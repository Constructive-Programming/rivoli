//! The rotary tables, and the one place a layer's class chooses between them.
//!
//! V4 carries TWO rotary parameter sets and picks per layer: a compressed layer rotates
//! against a large base WITH the YaRN interpolation, a plain one against the ordinary base
//! with YaRN off. The two tables have the same element type, the same stride and the same
//! shape, so nothing downstream can tell them apart — substituting one for the other is
//! `v4oracle::Defect::RopeNoYarn`, whose whole character is that the frequencies stay
//! plausible at every scale and the text stays fluent.
//!
//! That is why the selection is a function here and a memo in [`Tables`], rather than a
//! `match` arm at each of the two call sites that need a table.
//!
//! Ported from `old:src/kvcompress.rs`. No device, no feature gate — see the module header
//! in `super` for why the arithmetic half of this arm is not behind the backend.

use super::geometry::LayerKind;
use rivoli_artifact::v4_config::V4Config;

/// One resolved rotary parameter set — everything [`table`] needs for one layer.
///
/// `theta` and `original_seq_len` are the only two fields that differ between the model's two
/// sets; see [`for_layer`], which is the only thing that should ever set them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Rotary width. Only the LAST this-many dims of a row are rotated.
    pub rope_head_dim: usize,
    /// The base in `1 / base^(2i/dim)`.
    pub theta: f32,
    /// **Zero disables YaRN**, which is how a plain layer is expressed rather than by a
    /// separate flag — one field carrying the whole discriminant, so a caller cannot turn
    /// YaRN off and leave the compressed base on.
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Params {
    /// The COMPRESSED set, read from the config. The plain set is this one with the pair
    /// swapped, which is what [`for_layer`] does.
    pub fn compressed(cfg: &V4Config) -> Self {
        let rs = &cfg.rope_scaling;
        Self {
            rope_head_dim: cfg.qk_rope_head_dim,
            theta: cfg.compress_rope_theta as f32,
            original_seq_len: rs.original_max_position_embeddings,
            factor: rs.factor as f32,
            beta_fast: rs.beta_fast as f32,
            beta_slow: rs.beta_slow as f32,
        }
    }
}

/// The per-layer table selection the reference performs in `Attention.__init__`.
///
/// Takes the *compressed* set and the one value a plain layer swaps in, so the plain set is
/// built with `..compressed`. That is deliberate: the two sets differ ONLY in the pair, and
/// struct-update syntax makes that a fact the compiler maintains rather than a comment that
/// rots — a rotary parameter added later cannot silently apply to one set and not the other.
///
/// Selecting one value of the pair without the other is two DISTINCT silent-wrong outcomes
/// the oracle enumerates separately (`RopeNoYarn`, `RopeBaseThetaEverywhere`), and both are
/// the slip a caller makes when theta and `original_seq_len` travel as loose same-typed
/// arguments. Here they move together, once.
///
/// A function rather than a two-table struct: the struct held two values, was built once, and
/// its accessor was this `if`.
pub fn for_layer(compressed: Params, rope_theta: f32, kind: LayerKind) -> Params {
    // YaRN is keyed to COMPRESSION, not to layer index and not to overlap: every layer with a
    // compressor gets it, whatever its ratio.
    if kind.compressor_ratio().is_some() {
        return compressed;
    }
    // Not merely "skip the blend": the reference passes `original_seq_len = 0`, and the base
    // theta travels WITH it. Keeping the large base here while dropping YaRN is the insidious
    // half of the selection — the frequencies stay plausible at every scale.
    Params {
        theta: rope_theta,
        original_seq_len: 0,
        ..compressed
    }
}

/// `[seqlen * rope_head_dim/2]` interleaved `(cos, sin)` pairs, flattened — the layout the
/// rotary kernel indexes.
///
/// Faithful to two quirks that are easy to "clean up" into a one-ulp disagreement, and a
/// one-ulp angle can flip a top-k tie, which is a SET difference and not a tolerance one:
///
/// 1. **The correction range is computed in `dim` units and then applied to `dim/2`
///    frequency indices.** That is what the reference does; matching the *intent* instead of
///    the code shifts the interpolation band by a factor of two.
/// 2. **Only the correction range is f64.** The bounds go through double-precision `ln`,
///    `floor` and `ceil`; the frequencies, the blend, the outer product and the polar form
///    are all single. Computing the whole table in f64 leaves a sub-bf16-ulp angle error that
///    still surfaces as sporadic one-ulp disagreement against a faithful implementation.
pub fn table(p: Params, seqlen: usize) -> Vec<f32> {
    let half = p.rope_head_dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / p.theta.powf((2 * i) as f32 / p.rope_head_dim as f32))
        .collect();
    yarn_blend(&mut freqs, p);
    let mut out = Vec::with_capacity(seqlen * half * 2);
    for t in 0..seqlen {
        for f in &freqs {
            let a = t as f32 * f;
            out.push(a.cos());
            out.push(a.sin());
        }
    }
    out
}

/// The YaRN interpolation, in place — a no-op when `original_seq_len` is zero, which is how a
/// plain layer is expressed. Split out so [`table`]'s two quirks are readable separately from
/// its loop.
fn yarn_blend(freqs: &mut [f32], p: Params) {
    if p.original_seq_len == 0 {
        return;
    }
    let dim = p.rope_head_dim as f64;
    let base = f64::from(p.theta);
    // The correction dimension, in `dim` units and in double precision — quirk 2.
    let correction = |rot: f64| {
        dim * (p.original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * base.ln())
    };
    let low = correction(p.beta_fast as f64).floor().max(0.0);
    let high = correction(p.beta_slow as f64).ceil().min(dim - 1.0);
    // The reference's nudge against a zero denominator, kept because dropping it turns a
    // degenerate config into a NaN table rather than into the reference's (arbitrary but
    // finite) ramp. Applied to the upper bound alone, which is what makes it a nudge.
    let widened = if high == low { high + 0.001 } else { high };
    let (min, max) = (low as f32, widened as f32);
    // Quirk 1: the bounds above are in `dim` units and this indexes `dim/2` entries.
    for (i, f) in freqs.iter_mut().enumerate() {
        let smooth = 1.0 - ((i as f32 - min) / (max - min)).clamp(0.0, 1.0);
        *f = *f / p.factor * (1.0 - smooth) + *f * smooth;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

    use super::*;

    fn shipped() -> Params {
        Params {
            rope_head_dim: 64,
            theta: 160000.0,
            original_seq_len: 65536,
            factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    /// **The selection moves BOTH halves of the pair or neither.** This is the whole content
    /// of `for_layer`, and the failure it prevents — a plain table built with the compressed
    /// base — produces frequencies that are plausible at every scale.
    #[test]
    fn a_plain_layer_swaps_the_base_and_the_yarn_length_together() {
        let c = shipped();
        let plain = for_layer(c, 10000.0, LayerKind::from_ratio(0));
        assert_eq!(plain.theta, 10000.0, "a plain layer uses the base theta");
        assert_eq!(plain.original_seq_len, 0, "and has YaRN off");
        // Everything else is untouched — the property `..compressed` maintains.
        assert_eq!(
            (
                plain.rope_head_dim,
                plain.factor,
                plain.beta_fast,
                plain.beta_slow
            ),
            (c.rope_head_dim, c.factor, c.beta_fast, c.beta_slow)
        );
        for ratio in [4usize, 8, 128] {
            assert_eq!(
                for_layer(c, 10000.0, LayerKind::from_ratio(ratio)),
                c,
                "every layer WITH a compressor keeps the compressed set, ratio {ratio}"
            );
        }
    }

    /// The tables actually differ, and differ everywhere a defect would hide: not just at the
    /// long positions YaRN was designed for. Without this, the test above pins a struct field
    /// and proves nothing about the numbers it produces.
    #[test]
    fn the_two_tables_are_numerically_distinct_from_the_first_rotated_position() {
        let c = shipped();
        let (a, b) = (
            table(c, 4),
            table(for_layer(c, 10000.0, LayerKind::Plain), 4),
        );
        assert_eq!(a.len(), 4 * 32 * 2, "seqlen x rd/2 x (cos, sin)");
        assert_eq!(a.len(), b.len());
        // Position 0 is `cos(0), sin(0)` in BOTH tables whatever the frequencies are, so a
        // comparison that included it would be diluted by 32 pairs that cannot differ.
        assert!(a[..64] == b[..64], "position 0 cannot distinguish them");
        let differing = a[64..].iter().zip(&b[64..]).filter(|(x, y)| x != y).count();
        assert!(
            differing > 64,
            "only {differing} of {} entries differ past position 0 — the two rotary sets \
             have collapsed onto each other and RopeNoYarn would be invisible",
            a.len() - 64
        );
    }

    /// YaRN off must be exactly "no blend", not "a blend with neutral parameters" — the
    /// property `original_seq_len == 0` is carrying — and on must leave a RAMP rather than
    /// scaling everything, which is the shape quirk 1 is about.
    ///
    /// The input is `rope_head_dim / 2` entries long because that is what [`table`] passes,
    /// and the length is load-bearing here: the correction bounds are computed in
    /// `rope_head_dim` units, so a short probe array falls entirely below the band, comes
    /// back untouched, and reads as "YaRN did nothing". That is what a first draft of this
    /// test asserted a bug from.
    #[test]
    fn yarn_is_off_exactly_when_the_original_length_is_zero_and_on_it_is_a_ramp() {
        let flat = || vec![1.0f32; shipped().rope_head_dim / 2];
        let (mut with, mut without) = (flat(), flat());
        yarn_blend(&mut with, shipped());
        yarn_blend(
            &mut without,
            Params {
                original_seq_len: 0,
                ..shipped()
            },
        );
        assert_eq!(without, flat(), "off must not touch a value");
        assert_eq!(
            with[0], 1.0,
            "the low frequencies are outside the band, untouched"
        );
        assert!(
            (with[with.len() - 1] - 1.0 / shipped().factor).abs() < 1e-6,
            "the high frequencies are past the band and fully divided by the factor"
        );
        let moved = with.iter().filter(|v| **v != 1.0).count();
        assert!(
            moved > 0 && moved < with.len(),
            "{moved} of {} entries moved — YaRN must leave a RAMP, not scale everything and \
             not scale nothing",
            with.len()
        );
    }
}
