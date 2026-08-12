//! **S3 item 2: the weightless QK-norm, and the 3.87 that follows it on Q alone.**
//!
//! `glimmer-architecture.md` §7 and §9 traps 2 and 3:
//!
//! ```text
//! q = qk_norm(q) * 3.87    # WEIGHTLESS RMSNorm over head_dim, THEN scale   M:341
//! k = qk_norm(k)           # normed, NOT scaled                             M:342
//! ```
//!
//! `qk_norm` is `MuseGlimmerRMSNorm(eps=rms_norm_eps, with_scale=False)` — it ships **no tensor**,
//! which is trap 2: a port that enumerates the checkpoint to decide what to implement skips it
//! silently. Trap 3 is giving K the 3.87 as well, or using it in place of the `1/sqrt(head_dim)`
//! softmax scale rather than as well as.
//!
//! The `qk_norm` tolerance row was measured on 2026-08-12 **before this kernel existed** (floor
//! 7.845e-6, weakest targeting defect `qk_norm_off` 1.483e0, `Rel(7.85e-5)`), with `qk_scale_on_k`
//! excluded by LOCALITY — it perturbs `k_proj`'s output, upstream of this operator, and an RMS norm
//! is scale-invariant but for its eps. `anchor.md` §`qk_norm`'s defect set.
//!
//! # What the goldens CAN pin here, and what they cannot
//!
//! **The norm's input is not captured.** `qk_norm.q` and `qk_norm.k` are the module's OUTPUT and
//! nothing upstream of them exists in the golden — no `q_proj.out`, no pre-norm q. So unlike the
//! sandwich norms, whose exact inputs are captured one operator earlier, there is no chain here and
//! **the norm's arithmetic is scored against a host oracle**, as the MLP's was.
//!
//! What the goldens do pin, exactly, is everything around it:
//!
//! 1. **The AXIS.** Each contiguous `head_dim` run of every `qk_norm.*` capture has `mean(y²)`
//!    within eps of 1 — which is true only if the reference normalised over head_dim per head. A
//!    port normalising over the whole hidden, or across rows, produces captures that fail it.
//! 2. **The SCALE, and that K does not get it.** `q.pre_rope` is captured on entry to
//!    `apply_rotary_pos_emb`, i.e. after norm AND after scale, so `q.pre_rope / qk_norm.q` is 3.87
//!    elementwise — and `k.pre_rope` is **bit-identical** to `qk_norm.k`. Trap 3 is refuted by the
//!    reference's own bytes rather than by a defect run: there IS no defect run for it, because the
//!    anchor's `qk_scale_on_k` scales upstream of a scale-invariant norm and is nearly a no-op
//!    (`anchor.md`, corrected 2026-08-12).
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

#[path = "common/glimmer_fixture.rs"]
mod fixture;
// **No `use fixture::{...}` line here, deliberately.** jscpd normalizes identifiers, so the
// four-line preamble every Glimmer fixture opens with — the two inner attributes, the `#[path]`
// include, and an import list — matches its neighbour's as a clone. `glimmer_fixture.rs`'s own
// header states the remedy: an import list is the one duplication Rust gives no way to factor, so
// have FEWER imports rather than an exemption. This file spells `fixture::` at each of its nine
// call sites instead.

/// Muse Glimmer's real head_dim, its qk-norm eps and Q's scale — **read from the vendored
/// `config.json` the fixture already includes, not restated here.**
///
/// `eps()` is the one that matters: it is fed to BOTH the kernel and the host oracle, so a wrong
/// value fails nothing and silently scores the kernel in a regime the model never occupies. That is
/// the restated-number class this port keeps finding, and the config is one `include_str!` away.
/// Review, 2026-08-12. (`qk_scale()` was already independently recovered from the goldens' own bytes
/// by `glimmer_anchor.rs::the_reference_scales_q_by_qk_scale_factor_and_leaves_k_alone`.)
fn shipped(key: &str) -> f64 {
    let cfg: serde_json::Value =
        serde_json::from_str(fixture::GLIMMER_SHIPPED_CONFIG).expect("the vendored config parses");
    cfg["text_config"][key]
        .as_f64()
        .unwrap_or_else(|| panic!("no numeric {key} in the vendored text_config"))
}
fn head_dim() -> usize {
    shipped("head_dim") as usize
}
fn eps() -> f32 {
    shipped("rms_norm_eps") as f32
}
fn qk_scale() -> f32 {
    shipped("qk_scale_factor") as f32
}

/// The host reference: a weightless RMS over `d`, times `scale`. `f64` accumulation deliberately —
/// scoring the KERNEL's reduction means the oracle must not re-associate the same way it does.
fn host_qk_norm(x: &[f32], d: usize, eps: f32, scale: f32) -> Vec<f32> {
    x.chunks(d)
        .flat_map(|row| {
            let mean = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
            let rs = scale as f64 / (mean + eps as f64).sqrt();
            row.iter().map(move |v| (*v as f64 * rs) as f32)
        })
        .collect()
}

/// The one place `launch_rmsnorm_weightless_batch` is spelled, returning its `Result`.
///
/// The scoring wrapper and the guard table must drive the SAME call or the guards prove something
/// about a second launch nobody uses — the argument `glimmer_fixture.rs::rmsnorm_launch` records,
/// and jscpd rejected the second spelling here too.
///
/// # Safety
/// `x` is `rows * d` live f32, written in place, live until the next `device_sync` — except when
/// the call is expected to be REFUSED, where the guard returns before any launch.
unsafe fn launch(x: *mut f32, rows: usize, d: usize, eps: f32, scale: f32) -> anyhow::Result<()> {
    // SAFETY: the caller's contract above. Null stream: every caller launches once and then joins.
    unsafe {
        rivoli::backend::hip::launch_rmsnorm_weightless_batch(
            x,
            rows,
            d,
            eps,
            scale,
            std::ptr::null_mut(),
        )
    }
}

/// `x` through the kernel, in place on a fresh device copy.
fn qk_norm(x: &[f32], d: usize, eps: f32, scale: f32) -> Vec<f32> {
    assert_eq!(
        x.len() % d,
        0,
        "{} values is not whole rows of {d}",
        x.len()
    );
    let b = fixture::dev(&fixture::f32b(x));
    // SAFETY: `b` holds exactly `x.len()` live f32, live until the sync inside `sync_read`.
    unsafe { launch(b.ptr() as *mut f32, x.len() / d, d, eps, scale) }
        .expect("rmsnorm_weightless_batch launch");
    fixture::sync_read(&b)
}

// ------------------------------------------------------------------------------------------

/// **The kernel against the host oracle**, over every width and both scale paths that matter.
///
/// One test and one accumulator, because the two halves were the same comparison to the same bar
/// with two different mechanisms (review, 2026-08-12): head_dim **128**, which is the width that
/// matters and no golden reaches — every capture is at 8 — and four RAGGED widths, where some
/// threads run the strided accumulation one more time than others. Neither 8 nor 128 is ragged:
/// both are under one 256-thread block, so the stride never wraps.
///
/// **Trap 2 is gated here and not by a test of its own.** A port that skips the norm returns its
/// input, and that scores 7.775e-1 against this oracle — 9,903x the tolerance. The dedicated test
/// that once stated it separately computed the signal on the HOST and never ran the kernel, which
/// made it a statement about `fill`'s dynamic range; it is now the anti-vacuity line below.
///
/// **Red proof, run and reverted 2026-08-12:** dropping `scale` from the kernel (`rs = 1.0f/...`)
/// reddens the Q path at **7.416e-1, 9,447x the tolerance**, and leaves the K path at 1.322e-7 —
/// exactly the shape the defect has, since K passes 1.0.
///
/// MEASURED 2026-08-12 on gfx1151: **1.322e-7** (K, unscaled), **1.367e-7** (Q, x3.87), and
/// 1.414e-7 worst over the ragged widths.
#[test]
fn the_qk_norm_matches_a_host_oracle_at_every_width_that_matters() {
    let (hd, e, qs) = (head_dim(), eps(), qk_scale());
    let mut s = fixture::Scored::new("qk_norm");
    // 96 heads: enough rows that the grid is more than a handful of blocks. (An earlier comment
    // claimed this was "more rows than any single block count this kernel will see per layer at
    // production" — false for any prefill of more than three rows, since production is 32 Q heads
    // times rows.)
    for (d, scale) in [
        (hd, 1.0f32),
        (hd, qs),
        (257, qs),
        (300, qs),
        (512, qs),
        (1000, qs),
    ] {
        let x = fixture::fill(96 * d, 21, 0.4);
        s.case(
            &qk_norm(&x, d, e, scale),
            &host_qk_norm(&x, d, e, scale),
            || format!("width {d} at scale {scale}"),
        );
    }
    println!(
        "qk_norm over {} widths: worst rel {:e} at {} against tol {:e}",
        s.cases, s.worst, s.at, s.tol
    );
    // **Anti-vacuity, and it is trap 2's signal.** The oracle must be far from the identity, or a
    // kernel that did nothing at all would pass every case above.
    let x = fixture::fill(96 * hd, 22, 0.4);
    let skipped = fixture::worst_rel(&x, &host_qk_norm(&x, hd, e, 1.0));
    println!(
        "a norm that does nothing scores {skipped:e} — {:.0}x tol",
        skipped / s.tol
    );
    assert!(
        skipped > 100.0 * s.tol,
        "skipping the norm moves the output only {skipped:e}, {:.0}x the tolerance — this fixture \
         cannot tell a normed vector from an unnormed one",
        skipped / s.tol
    );
    // A second, tighter bar against this kernel's OWN measurement, at the ~15x this port's other
    // kernels use. The row's 7.85e-5 comes from the whole `qk_norm` bucket's fp32 floor at the
    // reference's widths, so it cannot see this kernel regress by two decades.
    assert!(
        s.worst <= 2.0e-6,
        "worst rel {:e} at {} > 15x the 2026-08-12 measurement",
        s.worst,
        s.at
    );
}

/// The launcher's guards, driven.
#[test]
fn the_launcher_refuses_geometry_and_constants_a_config_cannot_produce() {
    let b = fixture::dev(&fixture::f32b(&fixture::fill(16, 23, 1.0)));
    let before = fixture::sync_read(&b);
    // SAFETY: the rejected calls return before any launch; the accepted ones write 16 live f32
    // in place.
    let call = |rows, d, eps, scale| unsafe { launch(b.ptr() as *mut f32, rows, d, eps, scale) };
    for (args, want, what) in [
        ((0usize, 8usize, eps(), 1.0f32), Some(1001), "no rows"),
        ((2, 0, eps(), 1.0), Some(1001), "a zero width"),
        ((2, 8, 0.0, 1.0), Some(1002), "a zero eps"),
        ((2, 8, f32::NAN, 1.0), Some(1002), "a NaN eps"),
        ((2, 8, eps(), 0.0), Some(1003), "a zero scale"),
        ((2, 8, eps(), -qk_scale()), Some(1003), "a negative scale"),
        (
            (2, 8, eps(), f32::INFINITY),
            Some(1003),
            "an infinite scale",
        ),
        (
            (2, 8, eps(), qk_scale()),
            None,
            "Q's real scale, which must pass",
        ),
        ((2, 8, eps(), 1.0), None, "K's unit scale, which must pass"),
    ] {
        let (rows, d, eps, scale) = args;
        fixture::expect_guard(call(rows, d, eps, scale), want, what);
    }
    // **The join, and it does more than join.** Reading the buffer back proves the two ACCEPTED
    // rows actually launched and wrote — a guard table whose accepted rows silently did nothing
    // would be a table of refusals wearing a census. `sync_read` opens with `device_sync`, so this
    // is also what keeps `b` alive past the last in-flight launch.
    //
    // **Compared against the bytes that went in, not against a constant.** The first version
    // asserted `any(|v| *v != 1.0)` on a buffer `fill` had drawn in [-1, 1) — true before any
    // launch, so the assert could not fail and was the exact census-wearing-refusals it names.
    // Review, 2026-08-12.
    assert_ne!(
        fixture::sync_read(&b),
        before,
        "the accepted rows left the buffer byte-identical, so nothing launched"
    );
}

/// **Where the eps sits, pinned without a golden — the one thing the oracle cannot check.**
///
/// `host_qk_norm` and the kernel are two spellings of one reading of one source line, so a
/// misreading of the eps SEMANTICS — `sqrt(mean)+eps`, or eps outside the mean — would be invisible:
/// the oracle would carry it too. That is the blind spot `mla.hip::qk_norm` names for itself ("a
/// shared blind spot created by agreeing with the instrument"), and review pointed out this file had
/// it without saying so.
///
/// The identity closes it from the OUTPUT alone: for `y = x·rsqrt(mean(x²)+eps)`,
/// `1 − mean(y²) = eps/(m+eps)` exactly, with `m` the input's own mean square. No other placement of
/// eps satisfies that. The reference's reading is confirmed independently at
/// `modeling_muse_glimmer.py:113-115` (`mean_squared = pow(2).mean(-1) + eps; h * pow(m, -0.5)`);
/// this is the arithmetic saying the same thing.
///
/// It also records why a `qk_norm.*` capture cannot be fed to this kernel as INPUT: every capture is
/// post-norm, and re-norming a normed vector moves it only ~9.6e-5 (= eps/2m at m = 0.4²/3), against
/// 7.8e-1 for skipping the norm on raw input. Four decades apart — the gap that makes a
/// capture-as-input test blind rather than merely weak.
#[test]
fn the_eps_is_inside_the_mean_and_the_identity_says_so() {
    let (hd, e) = (head_dim(), eps());
    let mut worst = 0.0f64;
    // **Small activations only, and that is forced rather than chosen.** The quantity being measured
    // is `eps/(m+eps)`, which shrinks as `1/m`, while the f32 storage error in `y` does not: at
    // mean(x²) = 3 the identity is 3.3e-6 against a noise floor near 1.2e-7, so the relative error is
    // ~4% of nothing. The first version of this test swept up to `scale_of_x = 3.0` and went red at
    // 5.789e-2 on exactly that — the bound was wrong by construction, not the kernel. These three
    // put the identity at 6.98e-2 / 1.19e-2 / 1.33e-3, and they are also where the model's own
    // post-norm activations sit.
    for scale_of_x in [0.02f32, 0.05, 0.15] {
        let x = fixture::fill(64 * hd, 25, scale_of_x);
        let y = qk_norm(&x, hd, e, 1.0);
        for (xr, yr) in x.chunks(hd).zip(y.chunks(hd)) {
            let ms =
                |v: &[f32]| v.iter().map(|q| (*q as f64) * (*q as f64)).sum::<f64>() / hd as f64;
            let (m, got) = (ms(xr), 1.0 - ms(yr));
            let want = e as f64 / (m + e as f64);
            worst = worst.max((got - want).abs() / want);
        }
    }
    println!("eps placement, worst relative error in 1 - mean(y^2): {worst:e}");
    // **Measured 1.4222e-4 — a 70x margin under the bound, and 1.6x ABOVE the f32 noise floor I
    // predicted (9.0e-5) when picking the regime.** The prediction is left here rather than quietly
    // replaced because the direction is the useful part: an f32 error estimate over a 128-term sum
    // undershoots, so a bound set at the predicted floor would have been a coin flip. 1e-2 is loose
    // against the MEASUREMENT and still tight against every other placement by orders —
    // `sqrt(mean)+eps` moves the identity by a factor of `sqrt(m)/eps`, ~1e3 at the narrowest scale.
    assert!(
        worst < 1e-2,
        "1 - mean(y^2) is {worst:e} away from eps/(m+eps) — the eps is not inside the mean"
    );
}
