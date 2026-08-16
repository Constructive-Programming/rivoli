//! What a compressor comparison MEASURES, and the verdict it is held to — the bf16-code metric,
//! the width pair it buckets by, and the four constants those two are stated against.
//!
//! **Split out of `v4_compressor/mod.rs` 2026-08-16** under the 800-line soft cap, by COHESION:
//! the parent holds the launch sequence and the cells, and this holds the number they are scored
//! with. Bodies and their comments travelled verbatim, and `mod.rs` re-exports this module with a
//! glob, so every `use v4_compressor::{Widths, diff, gap, …}` in the two suites is untouched.

use rivoli_artifact::v4_config::V4Config as EngineV4Config;
use rivoli_core::num::f32_to_bf16;

// =======================================================================================
// the metric
// =======================================================================================

/// A comparison of two `[n, d]` block sets, split by whether `act_quant` touched the dim.
///
/// The split is the instrument. `act_quant` covers dims `[0, d - rd)` at block 64 and leaves the
/// RoPE tail `[d - rd, d)` in bf16 (model.py:378), so **the tail is a direct window onto the
/// pre-quantization arithmetic**. A disagreement confined to the quantized region cannot be a
/// pooling, norm or RoPE bug; one that appears in the tail must be.
///
/// [`Diff::worst_ratio`] exists because e4m3 is a STEP function: a pre-quantization value a hair
/// over a rounding boundary quantizes a whole step away, and one e4m3 step is [`E4M3_ULP`] =
/// **exactly 16 bf16 codes**. By magnitude alone that is indistinguishable from a real ~9%
/// arithmetic error; what distinguishes it is the RATIO landing on an e4m3 step and only a
/// handful of elements moving at all.
pub struct Diff {
    /// Max bf16 code gap over every element.
    pub max: u32,
    /// Max over dims `act_quant` rewrote.
    pub max_quant: u32,
    /// Max over the RoPE tail, which `act_quant` never touches.
    pub max_tail: u32,
    /// How many elements differ at all, out of how many.
    pub differing: usize,
    pub total: usize,
    /// `(dim within the row, want, got)` at the largest gap.
    pub worst: (usize, f32, f32),
}

impl Diff {
    /// `got / want` at the worst element. One e4m3 step is a ratio in **[1.0667, 1.125]** (or the
    /// reciprocal) depending where in the binade it lands — not `1.125` flat, which only holds at
    /// mantissa 0. Outside that range the disagreement is not a boundary flip.
    pub fn worst_ratio(&self) -> f32 {
        let (_, w, g) = self.worst;
        if w == 0.0 { f32::INFINITY } else { g / w }
    }

    pub fn one_line(&self, label: &str) -> String {
        let head = format!(
            "{label}: max={} (quant_dims={} rope_tail={}) differing={}/{} ({:.4}%)",
            self.max,
            self.max_quant,
            self.max_tail,
            self.differing,
            self.total,
            100.0 * self.differing as f64 / self.total as f64
        );
        // `worst` is only meaningful when something differed. Printing `ratio=inf` for a
        // bit-identical pass would read as the pathological "want is zero" case on the one line a
        // triager looks at.
        if self.differing == 0 {
            return format!("{head} bit-identical");
        }
        let (dim, w, g) = self.worst;
        format!(
            "{head} worst@dim{dim} want={w:e} got={g:e} ratio={:.4}",
            self.worst_ratio()
        )
    }
}

/// The two head widths every comparison here is stated against: the compressor's output width
/// `d`, and the RoPE tail `rd` that sits inside it and is NOT quantized.
///
/// A pair rather than two adjacent `usize` arguments, because [`diff`] splits its verdict on
/// `quant_dims = d - rd`: transposed, every element lands in the wrong bucket, so `max_quant` is
/// scored against the tail's bf16 floor and `max_tail` against the e4m3 allowance. Both totals
/// stay right and the verdict INVERTS.
///
/// **[`Widths::checked`] is what prevents it, not the subtraction.** An earlier version of this
/// comment claimed a transposed pair "would underflow and panic". That is PROFILE-DEPENDENT,
/// which is worse than simply wrong: on the dev profile `overflow-checks` is on and `64 - 512`
/// does panic, so the claim holds exactly where someone would check it. Under `--release` — where
/// nothing sets `overflow-checks` — it WRAPS, `quant_dims` becomes ~2^64, `dim < quant_dims`
/// holds for every element, `max_tail` stays 0 and the tail check passes vacuously. So the
/// protection evaporated in the one profile prescribed for measurement runs, and a transposition
/// would have LOOSENED the gate there in silence. An `assert!` is profile-independent and costs
/// nothing: it runs once per `Widths` built.
#[derive(Clone, Copy)]
pub struct Widths {
    pub d: usize,
    pub rd: usize,
}

impl Widths {
    /// **From the CHECKPOINT's own config**, never from the oracle's hard-coded transliteration
    /// — see `common::Configs` for why that direction is load-bearing.
    pub fn of(cfg: &EngineV4Config) -> Self {
        Self::checked(cfg.head_dim, cfg.qk_rope_head_dim)
    }

    /// The one constructor, so no `Widths` exists that [`diff`] could mis-bucket.
    ///
    /// `rd > 0` is a PRECONDITION here, not an accident: an empty RoPE tail leaves `max_tail`
    /// never assigned, so [`assert_clean`] would print a tail verdict of 0 for a region it never
    /// looked at. Every layer class shipped today has one; a future class without one needs
    /// `assert_clean` reconsidered, not this loosened.
    pub fn checked(d: usize, rd: usize) -> Self {
        // TWO asserts, not one conjunction: `(512, 0)` is not a transposition, and a single
        // message would send that reader hunting for a swap that is not there. It would also
        // leave both `#[should_panic]` tests matching the same substring, so neither would be
        // pinned to its own condition.
        assert!(
            rd < d,
            "Widths {{ d: {d}, rd: {rd} }}: the RoPE tail sits strictly inside the head width — \
             this pair is transposed, and `diff` would score every element against the wrong \
             bound rather than fail"
        );
        assert!(
            rd > 0,
            "Widths {{ d: {d}, rd: 0 }}: an empty RoPE tail leaves `max_tail` unassigned, so the \
             tail verdict would pass vacuously"
        );
        Self { d, rd }
    }
}

/// Compare two flattened `[n, d]` block sets in bf16 code space.
///
/// Both sides hold bf16 values — the kernel's last act on every row is `rbf16` and the oracle's
/// is `round_bf16` — so the unit is exact and no epsilon is chosen: re-encode both and difference
/// the codes. 0 is bit-identical, 1 is adjacent representable values.
///
/// Sign goes through a monotone ordering first. Raw bf16 codes across zero would report ~65000
/// for two values a hair apart, which would make the metric read as noise exactly where
/// cancellation put the interesting cases.
///
/// **This is a RELATIVE metric with a known blind spot at zero**: a code gap says nothing about
/// absolute magnitude, so an element near zero can report a large gap for a negligible absolute
/// difference. That is why [`Diff`] carries `differing` and `worst` — the count and the actual
/// pair are what separate that case from a real error, and the max alone cannot.
pub fn diff(want: &[f32], got: &[f32], w: Widths) -> Diff {
    let (d, rd) = (w.d, w.rd);
    assert_eq!(want.len(), got.len(), "diff: length mismatch");
    assert!(
        want.len().is_multiple_of(d),
        "diff: not a whole number of [d] rows"
    );
    let ord = |x: f32| -> i32 {
        let c = i32::from(f32_to_bf16(x) as i16);
        if c < 0 { -32768 - c } else { c }
    };
    let quant_dims = d - rd;
    let mut out = Diff {
        max: 0,
        max_quant: 0,
        max_tail: 0,
        differing: 0,
        total: want.len(),
        worst: (0, 0.0, 0.0),
    };
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        let e = ord(w).abs_diff(ord(g));
        if e == 0 {
            continue;
        }
        out.differing += 1;
        let dim = i % d;
        if dim < quant_dims {
            out.max_quant = out.max_quant.max(e);
        } else {
            out.max_tail = out.max_tail.max(e);
        }
        if e > out.max {
            out.max = e;
            out.worst = (dim, w, g);
        }
    }
    out
}

/// The verdict on a CLEAN comparison, stated in the unit each region actually ends in.
///
/// Three conditions, and the point is that no one of them can be satisfied by loosening another:
/// the RoPE tail is not quantized so it is held to the bf16 floor; the quantized dims may differ
/// by at most one e4m3 step; and only a sliver of elements may differ at all. A real arithmetic
/// error shows up in the tail, or moves more than one step, or moves the bulk of the elements —
/// this rejects all three. Returns the complaints rather than panicking, so a caller can measure
/// every cell before it aborts.
pub fn assert_clean(name: &str, dv: &Diff) -> Vec<String> {
    let mut bad = Vec::new();
    if dv.max_tail > CLEAN_ULP {
        bad.push(format!(
            "{name}: RoPE tail {} > {CLEAN_ULP} bf16 ULP — `act_quant` never touches those dims, \
             so this is the pooling, the norm or the rotation and not a rounding step",
            dv.max_tail
        ));
    }
    if dv.max_quant > E4M3_ULP {
        bad.push(format!(
            "{name}: quantized dims {} > one e4m3 step ({E4M3_ULP}) — more than a boundary flip",
            dv.max_quant
        ));
    }
    let frac = dv.differing as f64 / dv.total as f64;
    if frac > MAX_BOUNDARY_FLIPS {
        bad.push(format!(
            "{name}: {}/{} elements differ ({:.3}%) — a boundary flip is rare and this is \
             systematic, so the one-step allowance is covering a real error",
            dv.differing,
            dv.total,
            100.0 * frac
        ));
    }
    bad
}

/// Compare and PRINT. The number is the evidence — a comparison that passed at 0 and one that
/// passed at 3 look identical in a green run, and only one says the kernel reproduces the
/// reference.
pub fn gap(label: &str, want: &[f32], got: &[f32], w: Widths) -> u32 {
    let dv = diff(want, got, w);
    println!("{}", dv.one_line(label));
    dv.max
}

/// The bound every clean comparison is held to.
///
/// Not zero, and the reason is specific: the RMSNorm's sum-of-squares folds as a tree over 256
/// threads while the oracle folds it sequentially over 512 elements, and the wave reduction does
/// the same to both projection dots. That re-association moves the norm factor by a relative
/// ~1e-7, which the following bf16 store rounds away in almost every element and occasionally
/// does not. `expf` versus `f32::exp` adds the same order again.
///
/// 2 is "the re-association floor plus one". It applies **only to the RoPE tail**, which is the
/// last region still ending in a bf16 store.
pub const CLEAN_ULP: u32 = 2;

/// One e4m3 quantization step, in bf16 codes. **Exactly 16, in every binade.**
///
/// e4m3 carries 3 mantissa bits and bf16 carries 7, over the same exponent semantics: a value is
/// `2^E·(1 + m/8)` against `2^E·(1 + n/128)`, so `m → m+1` *is* `n → n+16`. There is no binade
/// dependence and no approximation in that.
///
/// **This is why the reference tree's first real-weights run read 16 and why widening
/// [`CLEAN_ULP`] would have been the wrong repair.** `act_quant` is the last thing that touches
/// dims `[0, d - rd)`, so those dims do not end in a bf16 store at all — they end in an e4m3 one,
/// and 16 codes is not "a 9% error", it is the SMALLEST nonzero disagreement those dims can
/// express. Holding a quantized output to a bf16 ULP was a unit error in the harness, not a
/// defect in the kernel.
///
/// The bound stays honest because it is one step and because two independent conditions sit
/// beside it: the untouched RoPE tail is still held to [`CLEAN_ULP`], and [`MAX_BOUNDARY_FLIPS`]
/// caps how many elements may move at all.
pub const E4M3_ULP: u32 = 16;

/// The fraction of elements allowed to sit on the far side of an e4m3 rounding boundary.
///
/// Derived, and then actually SET at the derivation. An element flips only if its
/// pre-quantization value lies within the two implementations' relative disagreement `ε` of a
/// boundary. The relative step is `(1/8)/(1 + m/8)` for `m ∈ [0,8)`, i.e. between **0.0667 and
/// 0.125** — so the expected flip fraction is `ε / 0.0667`, not `ε / 0.125`; taking the wide end
/// understates flips by up to 1.9x. Re-association puts `ε` near 1e-6, predicting well under one
/// flip in 32768 elements.
///
/// **It was 1% in the reference tree and that was wrong.** At 1% the three clean conditions were
/// jointly satisfiable by a real systematic error: a uniform ~0.1% relative error (a wrong
/// `norm_eps`, an `ape` scaled by 1e-3, a subtly wrong softmax) leaves the tail under one bf16
/// code — one code is 0.78% relative — flips the quantized dims by exactly one step where it
/// flips them at all, and lands near 0.8% of elements. All three green, real bug. At 0.1% that
/// example fails loudly, while still sitting ~40x above the worst `ε` the derivation admits. The
/// measured fraction is printed on every comparison so this can be tightened from data.
pub const MAX_BOUNDARY_FLIPS: f64 = 0.001;

/// How far a defect must move the output before the comparison is said to RESOLVE it.
///
/// Stated against the **quantization floor**, not against the clean gap. One e4m3 step is the
/// smallest disagreement a quantized dim can express, so anything within a step or two of that
/// is indistinguishable from a boundary flip. Four steps is the bound; the reference tree's first
/// real-weights run measured the `no-ape` separations at ~30000, nearly 2000 steps, so real
/// defects clear this by three orders of magnitude and the bound is not what decides them.
pub const RESOLVABLE: u32 = 4 * E4M3_ULP;
