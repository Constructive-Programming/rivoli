//! **S3 item 1: the centered RMSNorm, and the one form this model cannot get wrong quietly.**
//!
//! `glimmer-architecture.md` §3 and §5: Glimmer's four per-layer sandwich norms are
//! `MuseGlimmerTextCenteredRMSNorm` — `_norm(x) * (1 + w)` with `w` initialised to **zeros** —
//! while its final norm, its weightless qk_norm and its embedding norm are the plain
//! `_norm(x) * w` with `w` ones. **Two formulas in one model**, and `rmsnorm_centered_single` is
//! the first kernel in the tree that computes the first one.
//!
//! The tolerance row was measured on 2026-08-12 **before this kernel existed** (`norm`: floor
//! 7.701e-6, weakest targeting defect 2.024e-2, `Rel(7.70e-5)`), which is the only order in which
//! the number means anything — `glimmer-reference/anchor.md` §`norm`'s defect set.
//!
//! # What this can score, and what it CANNOT — the same wall item 4 hit
//!
//! **The anchor captures norm OUTPUTS only.** `t0.L0.input_layernorm.out` is there;
//! its input is not, and neither is `w`, which exists only as a draw inside the driver's python.
//! So there is no (input, weight, output) triple to score a norm against, exactly as there was
//! none for the MLP.
//!
//! One pair is *nearly* usable and is worth naming because S3's loop will use it:
//! `t0.embed_norm.out` IS the input to `t0.L0.input_layernorm` (§5's order is embed → embed_norm
//! → layer 0). With `w` recoverable as `out / (x · inv)` the arithmetic could be scored — but a
//! weight recovered from the output cannot then falsify the FORM, because recovering `1+w` and
//! feeding it to a plain kernel reproduces the output exactly. **Form and weight are not
//! separable from an output alone**, so this file does not pretend to separate them.
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
/// > OVER the tolerance. `0.05` puts mean(x²) at 8.3e-4, inside that band's lower end.
///
/// This is the `logits` row's lesson arriving one operator later: a threshold measured on the
/// reference means nothing against a fixture that does not stand where the reference stood.
const X_SCALE: f32 = 0.05;

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
    let mean: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean + eps as f64).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, k)| {
            let scale = if centered { 1.0 + *k as f64 } else { *k as f64 };
            (*v as f64 * inv * scale) as f32
        })
        .collect()
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
    // **MEASURED 2026-08-12 on gfx1151 at 1.179e-7 (eps 1e-5) and 1.172e-7 (eps 1e-8)**, so this
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
    // The plain form on the HOST, not the device: `rmsnorm_single` is not this diff's
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
#[test]
fn the_rows_own_defect_exceeds_the_row_at_the_scale_the_norms_run_at() {
    let tol = fixture::rel_tolerance("norm");
    let x = fill(HIDDEN, 6, X_SCALE);
    let w = fill(HIDDEN, 7, W_SCALE);
    let d = worst_rel(&rmsnorm(&x, &w, EPS_PRE), &rmsnorm(&x, &w, EPS_POST));
    let m: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
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
