//! **S3 item 1: the centered RMSNorm, and the one form this model cannot get wrong quietly.**
//!
//! `glimmer-architecture.md` §3 and §5: Glimmer's four per-layer sandwich norms are
//! `MuseGlimmerTextCenteredRMSNorm` — `_norm(x) * (1 + w)` with `w` initialised to **zeros** —
//! while its final norm, its weightless qk_norm and its embedding norm are the plain
//! `_norm(x) * w` with `w` ones. **Two formulas in one model**, and `rmsnorm_centered_rows` is
//! the first kernel in the tree that computes the first one.
//!
//! The tolerance row was measured on 2026-08-12 **before this kernel existed** (`norm`: floor
//! 7.701e-6, weakest targeting defect 2.024e-2, `Rel(7.70e-5)`), which is the only order in which
//! the number means anything — `glimmer-reference/anchor.md` §`norm`'s defect set.
//!
//! # What this can score, and what it CANNOT
//!
//! **Three of the four sandwich norms have their exact input in the goldens**, so the kernel is
//! scored against reference bytes: `embed_norm.out` → `L0.input_layernorm.out` (§5's order is embed
//! → embed_norm → layer 0), and — because §3's post-norms run on the BRANCH before the residual add
//! — `attn.o_proj.out` → `post_attention_layernorm.out` and `mlp.down_proj.out` →
//! `post_feedforward_layernorm.out` on every layer. `w` is not captured and is recovered per element.
//! `pre_feedforward_layernorm` is the fourth and has no chain: its input is a residual SUM that
//! nothing captures, as is `input_layernorm`'s above layer 0.
//!
//! **Two different form questions, and only one of them is unanswerable here.** The REFERENCE's form
//! is not identifiable from its own bytes: recovering the multiplier and feeding it to a plain kernel
//! reproduces the output exactly, and the multiplier's RANGE is no tell either, because the driver
//! draws a centered norm's `w` as `uniform_(-0.2, 0.2)` (so `1+w` ∈ (0.8, 1.2)) and a plain norm's as
//! `uniform_(0.8, 1.2)` — the same interval. But the KERNEL's form IS gated by the chain test, and
//! saying otherwise was this file's own overstatement (found by review 2026-08-12): the recovery fixes
//! the convention by returning `w = multiplier − 1`, so a kernel switched to `x·w` predicts
//! `x·inv·(mult−1)` against a reference of `x·inv·mult` and reddens on the first case — **MEASURED
//! 2026-08-12 at 8.618e-1, 1.12e4x the tolerance**, by patching the kernel to `x·w` and reverting.
//! Review computed the magnitude; running it is what makes it a number this file may state. `norm_not_centered` is still scored against a host oracle below, because that names the
//! substitution explicitly instead of relying on a red the chain test would give for many reasons.
//!
//! > **CORRECTED 2026-08-12.** This section read "the anchor captures norm OUTPUTS only … there is no
//! > (input, weight, output) triple to score a norm against" and stopped there, which is how item 1
//! > shipped with no reference-byte gate at all. The form argument above is the only part that
//! > survived; it had been used to skip the arithmetic and the eps, which it never covered. The
//! > dated record is in `docs/investigations/glimmer-integration.md` S3 item 1.
//!
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{dev, f32b, fill, rmsnorm, worst_rel, zeros};

/// Glimmer's hidden width and its two norm epsilons, from `glimmer-architecture.md` §1 —
/// `rms_norm_eps` on the two pre-norms, `post_norm_eps` on the two post-norms, three orders of
/// magnitude apart. `tests/glimmer_head.rs` pins its own constants against the vendored config;
/// these two are pinned there as `rms_norm_eps`/`post_norm_eps` already
/// (`model.rs::glimmer_shipped_config_parses_and_matches_the_reference_doc`).
const HIDDEN: usize = 6656;

/// The activation scale the four norms actually see, and it is **not** unit scale.
///
/// > **Corrected 2026-08-12 by two reviews, which computed this independently.** The first draft
/// > drew `x` at `fill(.., 1.0)`, i.e. mean(x²) ≈ 1/3 — and at that scale swapping the two eps
/// > moves the output by only `0.5·(1e-5 − 1e-8)/mean(x²)` = **1.5e-5, which is 0.19x the row's
/// > own 7.70e-5 tolerance.** So the fixture was scoring the kernel in a regime where the very
/// > defect the row was priced on is INVISIBLE, and a kernel handed the wrong eps would have
/// > passed. A test asserting the two eps were separable went red on exactly that and was
/// > deleted; deleting it was the wrong call — the scale was wrong, not the test.
/// >
/// > Both reviews recovered the reference's real post-norm inputs from the goldens
/// > (`attn.o_proj.out`, `mlp.down_proj.out`, 8 layers × 7 steps): **mean(x²) = 1.14e-3 …
/// > 6.39e-3**, 50–300x smaller than unit scale, where the eps swap is 1.9e-3–3.5e-3 — 25–45x
/// > OVER the tolerance.
/// >
/// > **CORRECTED AGAIN 2026-08-12, same day.** This said "`0.05` puts mean(x²) at 8.3e-4, inside that
/// > band's lower end" — and 8.3e-4 is BELOW 1.14e-3, so it never was inside; review caught the
/// > arithmetic. The census below now measures the row means from the reference's own bytes at
/// > **1.139e-3 … 6.55e-3**, confirming the band above, so 0.05 stood 1.4x under its low end. That is
/// > the SAFE direction — a smaller mean makes the eps signal larger, so the fixture overstated the
/// > row's power rather than understating it — and it is still the error class this constant exists to
/// > prevent. `0.11` puts mean(x²) at `0.11²/3` = **4.03e-3**, inside the band, where the eps
/// > substitution measures 1.239e-3 (16x the tolerance) instead of the 77x that standing low bought.
///
/// This is the `logits` row's lesson arriving one operator later: a threshold measured on the
/// reference means nothing against a fixture that does not stand where the reference stood.
const X_SCALE: f32 = 0.11;

/// The centered weight's scale, from the anchor's OWN draw rather than from an initialisation
/// fact. Review recovered `1 + w` from the goldens and found the tiny model's centered weights
/// span [-0.197, +0.198] — the driver draws `uniform_(-0.2, 0.2)`. The first draft used 0.02 and
/// called it "as shipped: centered weights sit near zero", which confused `w`'s INITIALISATION
/// (zeros, `modeling_muse_glimmer.py`) with a trained checkpoint's weights. The shipped 30B
/// scale is unknown until S4 converts it; 0.2 is the only centered weight anything here has seen.
const W_SCALE: f32 = 0.2;
const EPS_PRE: f32 = 1e-5;
const EPS_POST: f32 = 1e-8;

/// The host reference. `f64` accumulation deliberately: the point is to score the KERNEL's
/// reduction, and a host sum that re-associates in f32 the same way would hide an error in both.
fn host_centered(x: &[f32], w: &[f32], eps: f32, centered: bool) -> Vec<f32> {
    let inv = inv_rms(x, eps);
    x.iter()
        .zip(w)
        .map(|(v, k)| {
            let scale = if centered { 1.0 + *k as f64 } else { *k as f64 };
            (*v as f64 * inv * scale) as f32
        })
        .collect()
}

/// `mean(x²)`, in f64 — the quantity that decides whether an eps substitution is visible at all,
/// and therefore printed by every test here that claims a defect has power.
fn mean_sq(x: &[f32]) -> f64 {
    x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64
}

/// `rsqrt(mean(x²) + eps)` on the HOST, in f64. One spelling, so the two host oracles below cannot
/// disagree about the reduction.
///
/// > **DO NOT make `recover` get this scalar from the kernel, however tempting "then they agree"
/// > sounds.** The chain gate's whole power against a reduction defect is the ASYMMETRY: `recover`
/// > inverts this f64 host reduction while the prediction runs the device kernel, so a kernel that
/// > divides by `n-1` leaves a `sqrt(71/72)` = 6.97e-3 residual — which is exactly the 6.969e-3 the
/// > reverted red proof measured, so the mechanism is confirmed by its own number. Recover the scalar
/// > from the device and the defect cancels algebraically: the score drops to ~1e-7, under the tighter
/// > bar, and the gate goes permanently green on the defect it advertises. Review found this file's
/// > previous wording ("so a test that recovers and a test that predicts cannot disagree about the
/// > reduction they invert") pointing straight at that refactor, 2026-08-12.
fn inv_rms(x: &[f32], eps: f32) -> f64 {
    1.0 / (mean_sq(x) + eps as f64).sqrt()
}

// ------------------------------------------------------------------------------------------

/// The kernel against the host oracle at Glimmer's real width, at both epsilons.
///
/// **6656 is the width that matters and no golden reaches it** — every anchor capture is at
/// hidden 72, so the LDS reduction has only ever run 72 elements across 256 threads, i.e. most
/// threads contributing zero. At 6656 each thread strides 26 times and the tree reduction is
/// fully loaded, which is the same gap item 1 found at head_dim 8 against a production 128.
#[test]
fn the_centered_norm_matches_a_host_oracle_at_the_real_width() {
    let tol = fixture::rel_tolerance("norm");
    let mut worst = 0.0f32;
    for (label, eps) in [("rms_norm_eps", EPS_PRE), ("post_norm_eps", EPS_POST)] {
        let x = fill(HIDDEN, 1, X_SCALE);
        // `w` near zero, as the real checkpoint ships it — `w` is initialised to ZEROS for a
        // centered norm, so `1 + w` is near 1 and the norm is near the identity. Scoring at a
        // large `w` would be scoring a regime the model never occupies.
        let w = fill(HIDDEN, 2, W_SCALE);
        let got = rmsnorm(&x, &w, eps);
        let want = host_centered(&x, &w, eps, true);
        let r = worst_rel(&got, &want);
        assert!(r <= tol, "{label}: worst rel {r:e} > {tol:e}");
        worst = worst.max(r);
        println!("centered norm at {label} ({eps:e}), width {HIDDEN}: worst rel {r:e}");
    }
    // A second, tighter bar against this kernel's OWN measurement. The row's 7.70e-5 is derived
    // from the reference's fp32 floor over the whole `norm` bucket; one row of one norm produces
    // far less, so the row alone cannot see this kernel regress by two decades.
    //
    // **MEASURED 2026-08-12 on gfx1151 at 1.174e-7 (eps 1e-5) and 1.172e-7 (eps 1e-8)**, so this
    // bar is ~17x. The first version said "see the printout for the value this bar is 20x of" —
    // a pointer to something libtest captures on a passing test, i.e. no recorded measurement at
    // all, which review flagged as the inherited-number shape one turn after this port fixed the
    // same thing elsewhere. For scale: a correct kernel's analytic worst case here is ~1.15e-6
    // (26 strided adds per thread plus an 8-level ladder at 2^-24), so 2.0e-6 is deliberately
    // above that rather than snug against the observed value.
    assert!(
        worst <= 2.0e-6,
        "worst rel {worst:e} > 20x the 2026-08-12 measurement"
    );
}

/// **The form defect, run the way production would hit it.**
///
/// Both substitution directions, scored against a host oracle.
///
/// > **§5's "crashes into garbage" is FALSE, and the anchor run in this same commit disproves
/// > it.** This doc claimed the plain form on a centered weight "multiplies the residual by ≈0 …
/// > the loud direction and therefore the safe mistake". Review checked
/// > `norm_not_centered` on the goldens: **zero non-finite values across all 1103 tensors**, a
/// > 0.15x scaling of the branch, and seven tokens emitted normally. §9 trap 5 has it right —
/// > this substitution "runs clean and produces a wrong model" — and §5 contradicts itself.
/// > The two-entry-point design stands on THAT, not on a crash that does not happen.
///
/// `worst_rel` cannot rank the two directions (scaled-by-0 and scaled-by-2 both differ from the
/// reference by ~1x its magnitude), so this asserts only that each is far over the bar. Ranking
/// them needs a magnitude comparison, which is deliberately not claimed here.
#[test]
fn the_plain_form_on_a_centered_weight_is_caught() {
    let tol = fixture::rel_tolerance("norm");
    let x = fill(HIDDEN, 3, X_SCALE);
    let w = fill(HIDDEN, 4, W_SCALE);

    let right = rmsnorm(&x, &w, EPS_PRE);
    // The plain form on the HOST, not the device: `rmsnorm_rows` is not this diff's
    // kernel and is gated in three other files, and taking it here is what forced the fixture
    // wrapper to carry a form bool. Review, 2026-08-12.
    let wrong = host_centered(&x, &w, EPS_PRE, false);
    let signal = worst_rel(&wrong, &right);
    println!("plain form on a centered weight: {signal:e} against tol {tol:e}");
    assert!(
        signal > 100.0 * tol,
        "the plain form on a centered weight moved the output by only {signal:e}, {:.0}x the \
         tolerance — this fixture cannot tell the two forms apart",
        signal / tol
    );
    // The other direction — a plain weight (ones) through the centered kernel scales by ≈2. NOT
    // "smaller than the collapse above": that clause referred to a magnitude assert deleted the
    // same day, and both signals land at ~1 in this metric anyway.
    let ones = vec![1.0f32; HIDDEN];
    let quiet = worst_rel(
        &rmsnorm(&x, &ones, EPS_PRE),
        &host_centered(&x, &ones, EPS_PRE, false),
    );
    println!("centered form on a plain weight: {quiet:e}");
    assert!(
        quiet > 100.0 * tol,
        "a plain weight through the centered kernel moved the output by only {quiet:e}"
    );
}

/// The eps guard, driven — and the eps is the parameter whose defect SET this operator's row.
///
/// `post_norm_eps_shared` (passing `rms_norm_eps` where `post_norm_eps` belongs) scores 2.024e-2,
/// 56x weaker than the form defect and therefore the number the tolerance was derived from. The
/// guard cannot catch that substitution — both values are legal — so what it catches is the
/// config never having been read at all.
#[test]
fn the_launcher_refuses_an_eps_that_cannot_come_from_a_config() {
    let xb = dev(&f32b(&fill(8, 5, 1.0)));
    let wb = dev(&f32b(&[0.0f32; 8]));
    let y = zeros(8 * 4);
    for (n, eps, want, what) in [
        (0usize, EPS_PRE, Some(1001), "an empty row"),
        (8, 0.0, Some(1002), "a zero eps"),
        (8, -1e-5, Some(1002), "a negative eps"),
        (8, f32::NAN, Some(1002), "a NaN eps"),
        (8, f32::INFINITY, Some(1002), "an infinite eps"),
        (8, EPS_POST, None, "the real post_norm_eps, which must pass"),
    ] {
        // SAFETY: the rejected calls return before any launch; the accepted one writes 8 live
        // f32 into `y` from 8 live f32 in each of `xb` and `wb`.
        fixture::expect_guard(
            unsafe {
                fixture::rmsnorm_launch(
                    xb.ptr() as *const f32,
                    wb.ptr() as *const f32,
                    n,
                    eps,
                    y.ptr() as *mut f32,
                )
            },
            want,
            what,
        );
    }
    rivoli::backend::hip::device_sync().unwrap();
}

/// **The row's own weakest defect, reproduced — and this is what the scale correction bought.**
///
/// The `norm` row is priced on `post_norm_eps_shared`: `rms_norm_eps` where `post_norm_eps`
/// belongs. A tolerance derived from that defect is only meaningful if the defect exceeds it
/// HERE, and at unit activations it does not — 0.19x, which is why the first version of this test
/// was red. At the reference's own scale it clears the bar by ~78x.
///
/// It is not a red proof of the kernel (both eps are legal values, so no kernel is wrong for
/// accepting either) — it is a proof that the ROW can reject the thing it was measured from.
/// Without it the row is a number the fixture cannot use.
///
/// **MEASURED 2026-08-12 at 1.239e-3, 16x the tolerance**, at `X_SCALE`'s corrected 4.026e-3. The
/// earlier 77x was the fixture standing 4.8x under the reference's own row means; a bigger multiple
/// bought by standing somewhere the model does not stand is not more power, it is less honesty.
#[test]
fn the_rows_own_defect_exceeds_the_row_at_the_scale_the_norms_run_at() {
    let tol = fixture::rel_tolerance("norm");
    let x = fill(HIDDEN, 6, X_SCALE);
    let w = fill(HIDDEN, 7, W_SCALE);
    let d = worst_rel(&rmsnorm(&x, &w, EPS_PRE), &rmsnorm(&x, &w, EPS_POST));
    let m = mean_sq(&x);
    println!(
        "eps substitution at mean(x2) {m:.3e}: {d:e} against tol {tol:e} ({:.0}x)",
        d / tol
    );
    assert!(
        d > tol,
        "the eps substitution moves the output by only {d:e}, at or under the {tol:e} tolerance \
         derived FROM it — the fixture is standing at mean(x2) {m:.3e}, and the reference's \
         post-norms stand at 1.14e-3..6.39e-3"
    );
}

// ---- the anchor's three exact chains ------------------------------------------------------
//
// Everything above scores the kernel against a HOST oracle at a fixture's chosen scale.
// Everything below scores it against the reference's own bytes, at the reference's own scale,
// at hidden 72 — a width where 184 of the ladder's 256 threads contribute nothing and which no
// other check in this tree reaches.

/// One chain: the capture holding a norm's input, the capture holding its output, whether it is a
/// POST-norm (`post_norm_eps`) rather than a pre-norm (`rms_norm_eps`), and whether the pair exists
/// on every layer.
///
/// **`post` is a structural claim, not a value, and the value is read from the golden.** §3 assigns
/// the eps by position, so what belongs in a table is which position each norm holds; the number
/// then comes out of the `tiny_config` inside the file being scored (`eps_of` below) instead of a
/// constant pinned against the shipped 30B config. Review found the constants were not anchored to
/// the bytes they were scoring, 2026-08-12.
///
/// > **What this column CANNOT do, measured rather than argued.** Flipping chain 0's `post` would be
/// > a wrong statement about the model and **every assertion in this file would still pass** —
/// > review demonstrated it. `recover` and the prediction take the same eps, so a wrong choice
/// > largely cancels, and chain 0's input is a normalised vector whose mean(x²) is ≈1 for every row,
/// > which makes the cancellation total (~1e-10). Chains 1 and 2 escape it only because their
/// > mean(x²) varies 4x row to row, leaving ~2.6e-3 — 33x the tolerance. So the eps ASSIGNMENT is
/// > gated for the two post-norms and is decoration for the pre-norm one. The census below is where
/// > that asymmetry is measured (0.1x against 41.8x–56.6x) rather than asserted here.
///
/// **`per_layer` is false for exactly one chain and it is also a structural fact.** A layer's
/// `input_layernorm` consumes the residual stream, which is captured only where it happens to
/// coincide with another capture — at layer 0, where it IS `embed_norm.out`. At layer 1 and above the
/// residual is a SUM that nothing captures, which is also why `pre_feedforward_layernorm` has no
/// chain at all.
const CHAINS: [(&str, &str, bool, bool); 3] = [
    ("embed_norm.out", "input_layernorm.out", false, false),
    (
        "attn.o_proj.out",
        "post_attention_layernorm.out",
        true,
        true,
    ),
    (
        "mlp.down_proj.out",
        "post_feedforward_layernorm.out",
        true,
        true,
    ),
];

/// The eps one chain runs under, out of the golden's OWN `tiny_config`.
///
/// Cross-checked against this file's two constants, which are pinned against the vendored shipped
/// config elsewhere: that makes one assert cover both directions — the golden matching the shipped
/// model, and these constants matching the golden they score. Without it the 6656-wide host-oracle
/// tests and the chain tests could be running two different pairs of numbers.
fn eps_of(gold: &fixture::Golden, post: bool) -> f32 {
    let key = if post {
        "post_norm_eps"
    } else {
        "rms_norm_eps"
    };
    let v = gold.c[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{}: {key} is not a number in tiny_config", gold.name))
        as f32;
    let want = if post { EPS_POST } else { EPS_PRE };
    assert_eq!(
        v, want,
        "{}: tiny_config {key} is {v:e} against this file's {want:e}",
        gold.name
    );
    v
}

/// Every (input row, output row) pair one chain carries at one layer, across every step.
///
/// Rows are flattened across steps deliberately: step 0 carries `prompt_len` rows and each decode
/// step one, and the norm applies to each row independently, so a chain's evidence is the whole
/// set. The count is checked by the caller against the metadata.
///
/// **What anchors these rows to reality is `fixture::cap`'s shape assert, not the count.** `cap`
/// refuses a capture whose shape is not `[1, q, h]`, against the golden's own bytes; the caller's
/// `rows.len()` check is arithmetic over the same `geometry` this loop calls and cannot fail on its
/// own. Review made that explicit 2026-08-12 — the visible assertion was the tautological one.
fn chain_rows(gold: &fixture::Golden, chain: usize, l: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let h = gold.n("hidden_size");
    let (_, steps) = gold.steps();
    let (input, out, _, per_layer) = CHAINS[chain];
    let mut rows = Vec::new();
    for t in 0..=steps {
        let (q, _) = gold.geometry(t);
        // The asymmetry is in these two lines: chain 0's input is a model-level capture
        // (`t3.embed_norm.out`) while its output and both other chains are per-layer
        // (`t3.L0....`). One prefix for both would have to invent a layer for the embedding norm.
        let inp = if per_layer {
            format!("t{t}.L{l}.{input}")
        } else {
            format!("t{t}.{input}")
        };
        let x = fixture::cap(gold, &inp, &[1, q, h], false);
        let y = fixture::cap(gold, &format!("t{t}.L{l}.{out}"), &[1, q, h], false);
        for r in 0..q {
            rows.push((
                x[r * h..(r + 1) * h].to_vec(),
                y[r * h..(r + 1) * h].to_vec(),
            ));
        }
    }
    rows
}

/// Recover the norm's `w` per element, one element at a time, from the row where that element's
/// normalised input is LARGEST in magnitude.
///
/// Every row satisfies `out = x · rsqrt(mean(x²)+eps) · (1+w)` with one `w` shared across rows, so
/// the multiplier is `out / (x·inv)` and `w` is that minus one — returned already shifted, because
/// `w` is what the kernel takes and both callers were undoing the shift on the next line. Picking a
/// fixed row would divide by whatever that row happened to hold — with 72 elements and 18 rows some
/// element is near zero in any single row, and there the multiplier would be one rounding error over
/// another. Taking the per-element maximum makes the divisor as large as the evidence allows.
///
/// **This does not fit anything away.** One parameter per element against 18 observations, and the
/// parameter is a per-element SCALE while `inv` varies per row — so a wrong eps, a wrong reduction
/// or a wrong mean cannot be absorbed into it. That is what makes the prediction below a gate
/// rather than a restatement. Least squares over the rows would be the obvious alternative and is
/// WORSE here: it averages residual error into the fitted scale, where the max-pick leaves the
/// parameter determined by one observation and tested against the other seventeen.
fn recover(rows: &[(Vec<f32>, Vec<f32>)], eps: f32) -> Vec<f32> {
    assert!(!rows.is_empty(), "no rows to recover a weight from");
    let h = rows[0].0.len();
    let mut best = vec![0.0f64; h];
    let mut mult = vec![f64::NAN; h];
    for (x, y) in rows {
        let inv = inv_rms(x, eps);
        for i in 0..h {
            let xn = x[i] as f64 * inv;
            if xn.abs() > best[i] {
                best[i] = xn.abs();
                mult[i] = y[i] as f64 / xn;
            }
        }
    }
    mult.iter()
        .zip(&best)
        .enumerate()
        .map(|(i, (m, b))| {
            // A column that is zero in EVERY row leaves nothing to recover, and a NaN weight would
            // then poison the prediction into `worst_rel`'s INFINITY with a message about the
            // kernel. **`m.is_finite()` is NOT implied by `b > 0.0`** — review asserted it was, but
            // a non-finite `y[i]` divided by a healthy `xn` is non-finite, so this is the guard for
            // a corrupt capture and `b` is the guard for a dead column. Two failures, both refused.
            assert!(
                *b > 0.0 && m.is_finite(),
                "element {i}: normalised input peaks at {b:e} and the multiplier came out {m:e}, so \
                 `w` is not recoverable and this chain cannot be scored"
            );
            (*m - 1.0) as f32
        })
        .collect()
}

/// **The gate item 1 should have shipped with: the kernel against the reference's own bytes.**
///
/// For each chain, at each layer, `1+w` is recovered from the rows and then every row is predicted
/// through the DEVICE kernel and compared to the captured output under the `norm` row. A recovery
/// that were wrong about the reduction could not fit 18 rows at once, so the score is simultaneously
/// the arithmetic's gate and the recovery's own consistency check — the reviews that found these
/// chains confirmed `1+w` is constant across all 18 rows to ~3e-7 by hand, and this reproduces that
/// through the kernel instead of by hand.
///
/// **It DOES gate the kernel's form** — the recovery returns `w`, so a kernel computing `x·w` predicts
/// against a reference of `x·(1+w)` and reddens on the first case, measured at 8.618e-1 (1.12e4x). What these bytes cannot identify is
/// the REFERENCE's form; the module header separates the two, and this line asserted the wrong one of
/// them until review caught it (2026-08-12). `norm_not_centered` stays on the host oracle above
/// because naming the substitution beats a red the chain test would give for many reasons.
///
/// **Two red proofs, both run and reverted 2026-08-12.** Dividing the reduction by `n - 1` instead of
/// `n` reddens this at **6.969e-3, 90x the tolerance**; dropping the centering to `x·w` reddens it at
/// **8.618e-1, 1.12e4x**. Each takes 5 of the file's 8 tests with it. The eps census below is the standing version of the same proof — it drives this exact
/// recovery-and-predict path with a wrong eps every run, so the machinery's ability to go red is not
/// a one-off note about a reverted patch.
#[test]
fn the_centered_norm_reproduces_the_anchors_three_exact_chains() {
    let mut s = fixture::Scored::new("norm");
    let mut want_cases = 0usize;
    let mut span = (f32::INFINITY, f32::NEG_INFINITY);
    let mut pairs = 0usize;
    for gold in &fixture::goldens() {
        // `census_dims` carries the two refusals a count like `want_cases` owes — a zero of either
        // factor makes an expectation of zero against a loop that scored nothing, and the layer count
        // is cross-checked against `layer_is_sliding`, which the driver writes separately. Review
        // asked for this test to use the shared tool `glimmer_gate.rs` next door already uses.
        let (layers, steps) = fixture::census_dims(gold);
        let (prompt, _) = gold.steps();
        want_cases += (prompt + steps) * (1 + 2 * layers);
        for (c, (_, _, post, per_layer)) in CHAINS.iter().enumerate() {
            let eps = eps_of(gold, *post);
            for l in 0..if *per_layer { layers } else { 1 } {
                let rows = chain_rows(gold, c, l);
                assert_eq!(
                    rows.len(),
                    prompt + steps,
                    "{}: chain {c} L{l} rows",
                    gold.name
                );
                let w = recover(&rows, eps);
                // **Per (chain, layer), not folded over all of them — review finding 2026-08-12.**
                // A single global span says only "somewhere in the union something reached each
                // end", and the degenerate case it misses is the model's own initialisation: `w` is
                // initialised to ZEROS for a centered norm, so a regeneration that left one norm
                // undrawn would recover `w ≡ 0` there, its prediction rows would stop being able to
                // catch a kernel that ignores `w` at all, and the other 33 pairs would still fill
                // the interval. A 72-sample draw from a width-0.4 uniform has range < 0.3 with
                // probability ~1e-9, so this bar is safe per pair; the EXTREMES are not (P(min >
                // -0.19) ≈ 16% at n=72), which is why the outer bounds stay global below.
                let (lo, hi) = w
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                        (a.min(*v), b.max(*v))
                    });
                assert!(
                    hi - lo >= 0.3,
                    "{}: chain {c} L{l} recovered w spanning only [{lo}, {hi}] — a draw from \
                     uniform_(-0.2, 0.2) does not do that, so this norm was probably never drawn",
                    gold.name
                );
                span = (span.0.min(lo), span.1.max(hi));
                pairs += 1;
                for (x, want) in &rows {
                    s.case(&rmsnorm(x, &w, eps), want, || {
                        format!("{} chain {c} L{l} at eps {eps:e}", gold.name)
                    });
                }
            }
        }
    }
    assert_eq!(s.cases, want_cases, "chain rows scored");
    // **The coverage, as an absolute number, because both checks above are functions of the same
    // metadata the loop reads.** `want_cases` restates `steps` and `layers`, so it catches an edit to
    // CHAINS and nothing else; review was right that the printed 612 was otherwise prose nothing
    // gates. What makes THIS a gate is that it is not derived at all — it is the recorded coverage of
    // the vendored goldens (2 x 17 pairs x 18 rows), and a regeneration that changes it is a reviewed
    // change that updates this line, exactly as `Vendored::check_bytes` treats their hashes.
    assert_eq!(
        (s.cases, pairs),
        (612, 34),
        "the vendored goldens carry 612 chain rows over 34 (chain, layer) pairs"
    );
    println!(
        "{} reference rows over {pairs} (chain, layer) pairs at hidden 72: worst rel {:e} at {} \
         against tol {:e}; recovered w in [{:.4}, {:.4}]",
        s.cases, s.worst, s.at, s.tol, span.0, span.1
    );
    // The outer bounds, global: the driver draws a centered norm's `w` as `uniform_(-0.2, 0.2)`, so
    // the union over every pair must fill that interval and leave it — MEASURED 2026-08-12 at
    // [-0.2000, 0.1999]. **This is the only check that can see a reduction defect the host oracle
    // SHARES with the kernel** (give both `n-1` and the prediction cancels; only the recovered `w`
    // moves, by the 6.97e-3 that scale factor is), so its slop matters: 1e-5, not the 1e-4 first
    // written. Review priced that — the recovery carries ~1e-7, torch's `uniform_` is open at the top
    // only, and 1e-4 was 333x its own justification, widening the band a defect could hide in.
    //
    // Deliberately NOT per-pair: a single pair's 72 draws have expected extremes ±0.1945, so a
    // per-pair extremes check false-reds ~16% of the time (P(min > -0.19) at n=72). The per-pair
    // RANGE above is the per-pair check; what survives uncovered is a scale drift confined to one
    // (chain, layer), up to ~0.5%, and no mechanism produces a layer-dependent reduction defect.
    assert!(
        span.0 >= -0.2 - 1e-5 && span.1 <= 0.2 + 1e-5 && span.0 <= -0.19 && span.1 >= 0.19,
        "recovered w spans [{}, {}], which is not the driver's uniform_(-0.2, 0.2)",
        span.0,
        span.1
    );
    // A second, tighter bar against this gate's OWN measurement (3.238e-7 over 612 rows,
    // 2026-08-12 on gfx1151), at the ~20x this file and `glimmer_gate.rs` both use. The row's
    // 7.70e-5 is 238x above what these rows produce because it is derived from the whole `norm`
    // bucket's fp32 floor at the reference's widths; against 72 elements it cannot see this kernel
    // regress by two decades. The eps census below is what proves the ROW still has power here.
    assert!(
        s.worst <= 6.5e-6,
        "worst rel {:e} > 20x the 2026-08-12 measurement",
        s.worst
    );
}

/// **The eps is visible in the reference bytes — on two of the three chains, and the third is the
/// whole reason `X_SCALE` exists.**
///
/// `post_norm_eps_shared` is the defect the `norm` row was priced on, and its signal is
/// `≈ 0.5·(eps_wrong − eps_right)/mean(x²)`: it lives or dies on where the activations stand.
///
/// **MEASURED 2026-08-12, and it is the fixture's own X_SCALE argument confirmed against reference
/// bytes rather than reconstructed from them.** Every figure below is reported AT THE WORST-SIGNAL
/// ROW, not as a max over rows of each quantity separately: the first version paired a max-mean with
/// a max-signal and the two were 2.5x apart under the formula above, which review caught as a pair a
/// reader cannot relate. See the printout for the current numbers; the shape is that the two
/// post-norms consume branch outputs at mean(x²) ~1e-3 and clear the tolerance by more than an order,
/// while the pre-norm chain consumes `embed_norm.out` — a NORMALISED vector at mean(x²) ≈ 1 — and
/// falls a decade UNDER it. As measured: worst rows at mean(x²) **1.139e-3 … 1.545e-3** moving
/// **3.218e-3 … 4.358e-3**, i.e. **41.8x to 56.6x**, against the pre-norm chain's **5.10e-6 … 5.12e-6,
/// 0.1x**. Those now RECONCILE with the formula — `0.5·(1e-5)/1.139e-3` = 4.4e-3 against 4.358e-3
/// observed — which the first version's max-of-each pairing did not, and that is the whole point of
/// reporting one row's two numbers.
///
/// **Three magnitudes exist for this one named defect and they are three different scopes.**
/// `tolerance.rs` prices the `norm` row on `post_norm_eps_shared` at 2.024e-2: that is the max over
/// the whole 224-tensor `norm` bucket, including downstream residual accumulation. This test measures
/// ONE norm at ONE layer, so it is smaller by design. Neither is wrong and nothing else in the tree
/// says so, which is why it is said here.
///
/// So this asserts a CENSUS — exactly the two post-norm chains carry the power — rather than a
/// blanket "the eps is detectable", which is false here and was false in the fixture above until
/// 2026-08-12. Membership is `> 10x tol` and non-membership `<= tol`, so a chain drifting into the
/// gap between them fails BOTH censuses rather than sliding from one to the other: the siblings in
/// this file use 100x on the same argument (a trap that is loud in one place and quiet in another is
/// a trap the fixture does not catch), and 10x is inside the measured band with an order to spare.
#[test]
fn the_eps_a_chain_runs_under_is_visible_where_the_activations_are_small() {
    let tol = fixture::rel_tolerance("norm");
    let (mut powered, mut blind) = (Vec::new(), Vec::new());
    for gold in &fixture::goldens() {
        for (c, (_, out, post, _)) in CHAINS.iter().enumerate() {
            // Layer 0 for every chain: the substitution's magnitude is a property of the activation
            // scale, and one layer measures that. The prediction gate above covers all of them.
            let eps = eps_of(gold, *post);
            let rows = chain_rows(gold, c, 0);
            let w = recover(&rows, eps);
            // The mirror of `post_norm_eps_shared`: whichever of the two epsilons this norm does
            // NOT take. Both are legal config values, so no kernel is wrong for accepting either —
            // what is measured is whether the reference bytes can TELL. Injected on the PREDICTION
            // side only, which is the real defect's shape: the engine reads the wrong eps while the
            // checkpoint's weight is the true one. (Putting it on both sides is a different and much
            // weaker defect — see the CHAINS table's note on why chain 0 cannot see that one.)
            let wrong = if *post { EPS_PRE } else { EPS_POST };
            let (mut sig, mut at_mean) = (0.0f32, 0.0f64);
            for (x, want) in &rows {
                let r = worst_rel(&rmsnorm(x, &w, wrong), want);
                if r >= sig {
                    // The mean OF THIS ROW, so the printed pair is one row's two numbers and the
                    // formula above relates them. A max over rows of each separately does not.
                    (sig, at_mean) = (r, mean_sq(x));
                }
            }
            println!(
                "{}: {out} worst row at mean(x2) {at_mean:.3e}, eps {eps:e}->{wrong:e} moves it \
                 {sig:e} ({:.1}x tol)",
                gold.name,
                sig / tol
            );
            if sig > 10.0 * tol {
                powered.push((gold.name, c));
            }
            if sig <= tol {
                blind.push((gold.name, c));
            }
        }
    }
    // Built from the `const TEXT` table rather than from a second `goldens()` call: same two names,
    // no second parse of 629 KB per blob, and the expectation stops depending on the reader it is
    // checking. Chains 1 and 2 are the post-norms, chain 0 the pre-norm.
    let names = || fixture::TEXT.iter().map(|(n, _)| *n);
    assert_eq!(
        powered,
        names().flat_map(|n| [(n, 1), (n, 2)]).collect::<Vec<_>>(),
        "exactly the two post-norm chains must clear 10x the tolerance on their own eps substitution"
    );
    assert_eq!(
        blind,
        names().map(|n| (n, 0)).collect::<Vec<_>>(),
        "the pre-norm chain stands at mean(x2) ~1 and must be UNDER the tolerance there — if it \
         gained power, the goldens moved and the asymmetry needs re-reading, not silently keeping"
    );
}

/// The kernel may write into its own input, **bit for bit**.
///
/// Not a hypothetical convenience: S3's loop applies four norms per layer to a 6656-wide residual
/// stream, and `post_attention_layernorm`'s output feeds `o_proj` with nothing else reading its
/// input — so an in-place norm is one buffer the loop does not have to hold.
///
/// **The `__restrict__` question, which three reviews raised and none resolved.** All three operands
/// carry `__restrict__`, and passing one pointer twice is the thing that qualifier disclaims — so a
/// green run here would certify today's codegen, not the kernel, if that were the whole story. It is
/// not: `swiglu` and `swiglu_clamped_bf16` in the same file are `__restrict__` throughout and are
/// launched in place from `gpu.rs` and `f4gpu.rs` in production, on the argument that a write which
/// DEPENDS on its read licenses no observable reordering. This kernel needs one clause more, because
/// its first loop reads other threads' indices — the `__syncthreads()` inside `block_sum_lds` is what
/// separates those reads from every write. That argument now lives at the kernel, where a caller can
/// find it; it was missing, which is the real finding, and deleting the test would have left the tree
/// with the same silence one test lighter.
///
/// Scored bit-against-bit with the non-aliased launch rather than against the host oracle: same
/// kernel, same grid, same inputs, so any difference at all IS the aliasing. That is strictly
/// stronger than the `norm` row, and the first version's `r <= tol` was the one bar in this file with
/// no own-measurement bar behind it — 7.70e-5 asserted against 1.182e-7 measured, 651x of slack.
#[test]
fn the_kernel_may_write_into_its_own_input() {
    let x = fill(HIDDEN, 8, X_SCALE);
    let w = fill(HIDDEN, 9, W_SCALE);
    // BOTH epsilons: two of the three chains run under `post_norm_eps`, three decades from the other,
    // and the first version covered only `EPS_PRE`.
    for (label, eps) in [("rms_norm_eps", EPS_PRE), ("post_norm_eps", EPS_POST)] {
        let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
        // SAFETY: `xb` is HIDDEN live f32, readable and writable, and is passed as both operands
        // deliberately — the in-place contract is stated at the kernel. `wb` is HIDDEN live f32.
        // Both outlive the sync inside `sync_read`.
        unsafe {
            fixture::rmsnorm_launch(
                xb.ptr() as *const f32,
                wb.ptr() as *const f32,
                HIDDEN,
                eps,
                xb.ptr() as *mut f32,
            )
        }
        .expect("aliased rmsnorm_centered_rows launch");
        assert_eq!(
            fixture::sync_read(&xb),
            rmsnorm(&x, &w, eps),
            "in place at {label} is not bit-identical to the same launch with a distinct output"
        );
    }
}

/// **A width where the strided loop is RAGGED**, which neither 72 nor 6656 is.
///
/// Review found the coverage claim wrong: 6656 is exactly 26 x 256, so every thread runs the strided
/// accumulation the same 26 times, and 72 is under one block so 72 threads run once and 184 never.
/// The regime where some threads accumulate `k` terms and others `k+1` — the one where an off-by-one
/// in the loop bound or a partial-tail LDS write shows up — was untested at both.
///
/// Nothing in Glimmer reaches these widths today; hidden is 6656 for all four sandwich norms. This is
/// here because the claim was made, and because the first caller at a non-multiple width (a per-head
/// norm, a padded activation, the drafter's own hidden) should find the regime already gated.
#[test]
fn the_centered_norm_holds_at_widths_that_do_not_divide_the_block() {
    let tol = fixture::rel_tolerance("norm");
    let mut worst = 0.0f32;
    // 257 and 6657: one element past a block boundary, so exactly one thread carries an extra term —
    // the narrowest version of the imbalance. 1000: 232 threads at 4 terms and 24 at 3. 6655: one
    // element short, so one thread carries one FEWER. 72 and 6656 are covered above.
    for n in [257usize, 1000, 6655, 6657] {
        for eps in [EPS_PRE, EPS_POST] {
            let (x, w) = (fill(n, 10, X_SCALE), fill(n, 11, W_SCALE));
            let r = worst_rel(&rmsnorm(&x, &w, eps), &host_centered(&x, &w, eps, true));
            assert!(
                r <= tol,
                "width {n} at eps {eps:e}: worst rel {r:e} > {tol:e}"
            );
            worst = worst.max(r);
        }
    }
    println!("ragged widths 257/1000/6655/6657 at both eps: worst rel {worst:e}");
    // **MEASURED 2026-08-12 at 1.762e-7**, so this bar is ~11x — deliberately the same 2.0e-6 the
    // full-block width holds to rather than a snugger number, because a correct kernel's analytic
    // worst case at these depths is ~1.15e-6 and a bar under that would reject correct code.
    assert!(
        worst <= 2.0e-6,
        "worst rel {worst:e} over the ragged widths > the bar the full-block widths hold to"
    );
}
