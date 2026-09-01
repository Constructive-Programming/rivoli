//! **Qwen3.8-Flash-Next's partial RoPE is `launch_rope_split_half`, measured** — a REUSE claim
//! turned into a comparison against the reference's own `rope.in`/`rope.out` captures.
//!
//! No new kernel, no census row: what is here is the evidence for reusing an M7 launcher on a
//! fifth architecture, which is the kind of claim this tree has learned not to make by reading two
//! implementations side by side. Split out of `kernel_qwen_indexer.rs` on 2026-09-01 when that file
//! crossed the 800-line soft cap — and the split is by cohesion rather than by line count: the
//! RoPE convention is a property of the attention module, and the indexer merely composes with it.
//!
//! # Why this triple is worth a suite of its own
//!
//! The QSA attention captures BOTH sides of the rotation at all three captured QSA layers and both
//! draws, and it is the only complete WEIGHTLESS triple in the decode goldens. Three things are
//! pinned at once and each is a named trap:
//!
//! * the **pairing** is split-half, `(j, j + rotary_dim/2)`, not interleaved `(2j, 2j+1)` — trap
//!   T16, and `mrope_interleaved: true` in the config invites exactly that confusion (it is about
//!   the mRoPE SECTIONS, not the pairing). `launch_rope_interleave` and `launch_rope_adjacent` are
//!   the two wrong doors, and they are separate entry points precisely so no call site can reach
//!   one by changing an argument;
//! * the **width** is the FIRST `rotary_dim` of `head_dim`, with the tail passing through. Needs
//!   `head_dim > rotary_dim` to be visible at all — 32 > 8 at these widths, 256 > 64 at the real
//!   ones — and the tail is asserted bit-identical, because agreement over the whole row cannot
//!   separate a partial rotation from a full one whose tail happened to land close;
//! * the **position** is `seq`, not `seq - 1`. The warm prefill occupies `0..seq`, so the decoded
//!   token sits at `seq`; both directions are asserted, because off by one here is a different
//!   rotation and not a rounding error.
//!
//! The frequency is `theta^(-2j/seg)` on both sides — MOD:96-116's
//! `inv_freq = 1 / theta ** (arange(0, 64, 2) / 64)` against `rope_pair`'s exponent — and at the
//! anchor's `rope_theta = 10.0` all four angles are spread across the circle, which is the
//! deviation track B's tiny config took *for this test*: at the real 1e7 over 8 rotary dims three
//! of the four frequency pairs are numerically inert and the whole claim would be vacuous.
//!
//! **YaRN is out of scope and pinned rather than guessed.** No released config enables it
//! (`rope_type: "default"`); the factor-4 form including the non-unit
//! `attention_factor = 0.1·ln(4) + 1` is recorded in `qwen-architecture.md` §7, and a kernel for it
//! is a later milestone, not a deferral of this one.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod kernel_qwen_harness;

use kernel_qwen_harness as h;

// `duplicate_mod`: the harness reaches `common/scoring.rs` by `#[path]` because `common/mod.rs` does
// not compile deviceless, and under `rocm` this declaration loads that same one file again through
// the umbrella. `kernel_qwen_gdn_recurrent.rs` carries the full argument.
#[cfg(feature = "rocm")]
#[allow(clippy::duplicate_mod)]
mod common;

/// The captured QSA layers — index 3 and every fourth after it runs the indexer, so these are the
/// three the decode goldens hold a rotation for.
const QSA_LAYERS: [usize; 3] = [3, 27, 47];

/// The worst the host rotation measures against the anchor over the twelve DECODE sites:
/// **1.2451e-7**, floor 3.0924e-8, so the spread is 4x.
///
/// There is no tolerance row for the RoPE itself, so the envelope is the `indexer` row it feeds —
/// the operator whose scores are wrong if the rotation is. (The prefill composition reads
/// 1.4596e-7 on `q` and 2.0805e-7 on `k`, where the position varies per row; that number belongs to
/// `kernel_qwen_indexer.rs`'s device chain and NOT here, and putting it here would loosen this
/// site's tripwire by 1.7x for no measurement.)
const HOST_WORST: f32 = 1.2451e-7;

/// The three wrong doors' separations, **minima over the twelve decode sites**, re-derived
/// independently 2026-09-01 (maxima beside them: interleaved to 1.2909e0, one-early to 9.7657e-1,
/// tail-dims to 2.1095e0).
///
/// > **These three were WRITTEN BEFORE THEY WERE MEASURED, and the suite went green anyway.** The
/// > first draft carried 4.9040e-1, 3.3286e-1 and 4.0175e-1 — one right and two guessed LOW — and
/// > every assertion passed, because a floor below the truth is a weaker gate that reports the same
/// > colour. Only re-deriving them from the vendored bytes found it. That is this tree's recorded
/// > "inherited numbers are unverified" failure arriving in a file whose whole subject is
/// > measurement, and the generalisation is the one that already applies to tolerances: a constant
/// > nobody re-derived is not evidence, and its greenness says nothing either way.
const INTERLEAVED: f32 = 6.1669e-1;
const ONE_POSITION_EARLY: f32 = 3.3286e-1;
const ON_TAIL_DIMS: f32 = 1.3245e0;

/// Which rotation an arm performs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Turn {
    Reference,
    /// `rope_interleaved_pairs`, the matrix's own row: the pair read as two adjacent elements.
    Interleaved,
    /// The rotation applied one position early — the warm prefill's LAST position instead of the
    /// decoded token's.
    OnePositionEarly,
    /// `rope_on_tail_dims`: the rotation moved to the LAST `rd` dims. The `[..rd]` head then passes
    /// through, so a fixture with `head_dim == rotary_dim` could not tell this from the reference —
    /// which is why the audit keeps `head_dim > rotary_dim` and why this arm asserts it.
    OnTailDims,
}

/// One rotation site: the widths it runs at and both sides of the reference's own capture.
///
/// A type, and ONE reader for it, because the deviceless arm and the device arm score the same
/// twelve sites — two loops reading `rope.in.{which}` and `rope.out.{which}` was this file's own
/// jscpd finding, and two readers are two places for the tensor names or the position to drift.
struct Site {
    at: String,
    hd: usize,
    rd: usize,
    pos: usize,
    theta: f64,
    src: Vec<f32>,
    want: Vec<f32>,
}

/// Two draws x three QSA layers x (q, k).
fn sites() -> Vec<Site> {
    let mut out = Vec::new();
    for d in h::decode_draws() {
        let (hd, rd) = (d.w("head_dim"), d.w("rotary_dim"));
        assert!(
            hd > rd,
            "head_dim {hd} must exceed rotary_dim {rd}, or the PARTIAL claim and the tail-dims \
             variant are both vacuous"
        );
        for layer in QSA_LAYERS {
            for (which, n) in [
                ("q", d.w("num_attention_heads")),
                ("k", d.w("num_key_value_heads")),
            ] {
                let a = format!("model.layers.{layer}.self_attn.rope");
                out.push(Site {
                    at: format!("{} L{layer} {which}", d.salt),
                    hd,
                    rd,
                    pos: d.decode_pos(),
                    theta: d.rope_theta(),
                    src: d.flat(&format!("{a}.in.{which}"), n * hd),
                    want: d.flat(&format!("{a}.out.{which}"), n * hd),
                });
            }
        }
    }
    assert_eq!(
        out.len(),
        12,
        "two draws times three QSA layers times q and k"
    );
    out
}

/// One arm's rotation over every head of one site, on the host.
fn turned(s: &Site, turn: Turn) -> Vec<f32> {
    let mut got = s.src.clone();
    for head in got.chunks_mut(s.hd) {
        match turn {
            Turn::Reference => h::rope_row(head, h::Pairing::SplitHalf, s.rd, s.pos, s.theta),
            // The one arm that reads a different pairing — `h::Pairing`'s own note argues why the
            // two conventions share a body in a test and never in a kernel.
            Turn::Interleaved => h::rope_row(head, h::Pairing::Interleaved, s.rd, s.pos, s.theta),
            Turn::OnePositionEarly => {
                h::rope_row(head, h::Pairing::SplitHalf, s.rd, s.pos - 1, s.theta)
            }
            Turn::OnTailDims => {
                let tail = s.hd - s.rd;
                h::rope_row(
                    &mut head[tail..],
                    h::Pairing::SplitHalf,
                    s.rd,
                    s.pos,
                    s.theta,
                );
            }
        }
    }
    got
}

// ── deviceless ──────────────────────────────────────────────────────────────────────────────

/// The host split-half rotation at position `seq` reproduces the reference, and the three wrong
/// doors do not.
///
/// The two halves are independent and the second is **blind to a wrong REFERENCE arm**: a wrong
/// door is scored against the anchor, never against the `Turn::Reference` rotation, so only the
/// `hold` loop above it would notice if that rotation itself drifted. `h::separations`' doc records
/// the four plants that measured this.
#[test]
fn the_host_split_half_rotation_is_the_reference_and_the_wrong_doors_are_not() {
    let b = h::Bar::at("indexer", HOST_WORST);
    let all = sites();
    for s in &all {
        b.hold(&s.at, "rotation", &turned(s, Turn::Reference), &s.want);
    }
    for (turn, name, floor) in [
        (Turn::Interleaved, "rope_interleaved_pairs", INTERLEAVED),
        (
            Turn::OnePositionEarly,
            "one position early",
            ONE_POSITION_EARLY,
        ),
        (Turn::OnTailDims, "rope_on_tail_dims", ON_TAIL_DIMS),
    ] {
        let rows: Vec<(String, f32)> = all
            .iter()
            .map(|s| (s.at.clone(), h::rel(&turned(s, turn), &s.want)))
            .collect();
        h::separations(h::Bar::at("indexer", HOST_WORST), name, floor, &rows);
    }
}

// ── on the device ───────────────────────────────────────────────────────────────────────────

/// The launcher reproduces the reference's rotation at every site, leaves the tail dims
/// bit-identical, and does NOT reproduce it one position early.
///
/// # RED-PROOF PLAN — for the integrator's first device run
///
/// This kernel is not new, so the plants belong to the CALL rather than to
/// `kernels/activation.hip`, and each is one argument away — which is the point of the suite:
///
/// * call `launch_rope_interleave` instead — ~6.2e-1, the whole of trap T16;
/// * pass `pos - 1` — ~3.3e-1;
/// * pass `seg = hd` instead of `seg = rd`, which rotates the whole head — the tail assertion
///   below catches it even where the magnitude would not;
/// * pass `stride = rd`, which walks the heads at the wrong pitch.
///
/// The host arm above must stay GREEN under all four: it is device-free, so a plant that reddens it
/// has broken the fixture and not the launch.
#[cfg(feature = "rocm")]
#[test]
fn the_split_half_rope_launcher_is_qwens_partial_rope() {
    use common::{back, dev, f32b, f32v, ok};
    use rivoli_backend::hip::launch_rope_split_half;

    let b = h::Bar::at("indexer", HOST_WORST);
    for s in sites() {
        let heads = s.want.len() / s.hd;
        let mut x = dev(&f32b(&s.src));
        let p = x.ptr_mut() as *mut f32;
        // SAFETY: `x` holds `heads · hd` f32 and the kernel rotates the first `rd` of each
        // `hd`-strided row in place; nothing else addresses the buffer while the stream runs.
        ok(
            unsafe { launch_rope_split_half(p, heads, s.hd, s.rd, s.pos, s.theta) },
            "rope_split_half",
        );
        let got = f32v(&back(&x));
        b.hold(&s.at, "rotation", &got, &s.want);
        // The dims above `rotary_dim` must be UNTOUCHED. Agreement over the whole row cannot
        // separate a partial rotation from a full one whose tail happened to land close, and at
        // `head_dim` 32 against `rotary_dim` 8 three quarters of every row is that tail.
        for (hd_i, (row, src)) in got.chunks(s.hd).zip(s.want.chunks(s.hd)).enumerate() {
            assert_eq!(
                &row[s.rd..],
                &src[s.rd..],
                "{} head {hd_i}: the tail dims moved, so this is not a PARTIAL rotation",
                s.at
            );
        }
        // And the position is load-bearing.
        let mut y = dev(&f32b(&s.src));
        let q = y.ptr_mut() as *mut f32;
        // SAFETY: as above.
        ok(
            unsafe { launch_rope_split_half(q, heads, s.hd, s.rd, s.pos - 1, s.theta) },
            "rope_split_half one position early",
        );
        b.separates(
            &s.at,
            "one position early",
            h::rel(&f32v(&back(&y)), &s.want),
        );
    }
}
