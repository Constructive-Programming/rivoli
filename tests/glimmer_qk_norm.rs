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
//!    anchor's `qk_scale_on_k` scales upstream of a norm that cancels a scalar only up to the eps
//!    term — a residue of 3.7x to 7.9x this row's tolerance, NOT the no-op this line used to claim;
//!    `tolerance.rs` carries the correction and what it leaves open
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
/// with two different mechanisms (review, 2026-08-12). The widths cover four classes: **8**, which
/// is what every golden is at and therefore the width S3's wiring will score against; **128**, the
/// production head_dim, which no golden reaches; **512**, an exact multiple where all 256 threads
/// run the strided loop the same number of times; and 257/300/1000, RAGGED, where some threads run
/// it once more than others. 8 and 128 are both under one block, so there the stride never wraps.
///
/// **Trap 2 is gated here only on the KERNEL side, and the wiring half is open with trap 3's.** A
/// port that skips the norm returns its input, and that scores 7.775e-1 against this oracle —
/// 9,905x the tolerance, which the anti-vacuity line below prints. But trap 2 as defined at the
/// head of this file is a port that *never implements the norm at all*, and no test here can see
/// that: an engine that never calls `rmsnorm_weightless_batch` passes every assertion in this file,
/// and `tests/kernel_coverage.rs` records that the tree is in exactly that state today. The
/// dedicated test that once stated trap 2 separately computed the signal on the HOST and never ran
/// the kernel, which made it a statement about `fill`'s dynamic range.
///
/// **Red proof, run and reverted 2026-08-12:** dropping `scale` from the kernel (`rs = 1.0f/...`)
/// reddens the Q path by four decades and leaves the K path at the unscaled figure — exactly the
/// shape the defect has, since K passes 1.0. Its magnitudes are not quoted: the code that produced
/// them is gone, so no run can reproduce them.
///
/// MEASURED: the assert below prints ONE worst over all cases and that is the only figure to quote.
/// This comment previously carried three per-case numbers from a draft that reported per-case, one
/// of which (a "ragged worst" of 1.414e-7) was LARGER than the overall worst it sat under.
#[test]
fn the_qk_norm_matches_a_host_oracle_at_every_width_that_matters() {
    let (hd, e, qs) = (head_dim(), eps(), qk_scale());
    let mut s = fixture::Scored::new("qk_norm");
    // 96 heads: enough rows that the grid is more than a handful of blocks. (An earlier comment
    // claimed this was "more rows than any single block count this kernel will see per layer at
    // production" — false for any prefill of more than three rows, since production is 32 Q heads
    // times rows.)
    // **The eps column is not decoration, and every case but the last one would be blind without
    // it.** Everywhere else the shipped `e` goes to BOTH the kernel and the oracle, so a kernel that
    // ignored its `eps` argument and hardcoded 1e-5 scores clean at every width — and the next
    // test's bad-eps refusals cannot see it either, because they are in the C wrapper and return
    // before the kernel reads anything. The 1e-3 case is the only thing in this file that proves the
    // value reaches the arithmetic; at these activations it separates the two by ~9.3e-3, 118x tol.
    for (d, scale, eps) in [
        (8, qs, e),
        (hd, 1.0f32, e),
        (hd, qs, e),
        (257, qs, e),
        (300, qs, e),
        (512, qs, e),
        (1000, qs, e),
        (hd, qs, 1.0e-3f32),
    ] {
        let x = fixture::fill(96 * d, 21, 0.4);
        s.case(
            &qk_norm(&x, d, eps, scale),
            &host_qk_norm(&x, d, eps, scale),
            || format!("width {d} at scale {scale} eps {eps:e}"),
        );
    }
    // An absolute, not `DIMS.len()`: a derived count cannot notice a tuple being deleted, which is
    // the rule this tree wrote down for the sandwich-norm coverage one item ago.
    assert_eq!(s.cases, 8, "a width class was dropped from the sweep");
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
        "worst rel {:e} at {} > 14.2x the 2026-08-12 measurement of 1.410519e-7",
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
        // **Both guards are driven with BOTH non-finite classes, and that asymmetry was a real
        // hole.** Each guard is written `!(v > 0 && v < INFINITY)`, which rejects NaN and infinity
        // through different clauses; the first version drove NaN on eps and infinity on scale, so
        // for each guard one clause was untested. Rewritten the obvious way — `scale <= 0.0f ||
        // scale > 1e30f` — NaN fails both comparisons, falls through, and the kernel fills the
        // tensor with NaN while every assertion in this file stays green: no scored case passes a
        // non-finite scale, and the join below is an `assert_ne!`, which NaN satisfies.
        ((2, 8, 0.0, 1.0), Some(1002), "a zero eps"),
        ((2, 8, -eps(), 1.0), Some(1002), "a negative eps"),
        ((2, 8, f32::NAN, 1.0), Some(1002), "a NaN eps"),
        ((2, 8, f32::INFINITY, 1.0), Some(1002), "an infinite eps"),
        ((2, 8, eps(), 0.0), Some(1003), "a zero scale"),
        ((2, 8, eps(), -qk_scale()), Some(1003), "a negative scale"),
        ((2, 8, eps(), f32::NAN), Some(1003), "a NaN scale"),
        (
            (2, 8, eps(), f32::INFINITY),
            Some(1003),
            "an infinite scale",
        ),
    ] {
        let (rows, d, eps, scale) = args;
        fixture::expect_guard(call(rows, d, eps, scale), want, what);
    }
    // **Every row above is a REFUSAL, and the buffer proves they refused before launching.** A guard
    // that launched and then returned its code would satisfy `expect_guard` and be invisible.
    assert_eq!(
        fixture::sync_read(&b),
        before,
        "a refused call still wrote — the guard returns its code AFTER launching"
    );
    // **Each accept is proven SEPARATELY, against the bytes immediately before IT.** Capturing
    // `before` once and joining after both accepts is satisfied by either one alone: a launcher that
    // grew a `if (scale == 1.0f) return 0;` fast path for K would return Ok, satisfy
    // `expect_guard`, and hide behind Q's write. Review, 2026-08-13 — one round after this same test
    // shipped an anti-vacuity assert that could not fail, which is why the weaker form is named here
    // rather than just replaced.
    //
    // Running Q then K on the same buffer is not a no-op: after Q the rows stand at mean(y²) ≈
    // 3.87², so K's unit-scale pass renormalises them back to 1 and the bytes move again.
    // `sync_read` opens with `device_sync`, so this is also what keeps `b` alive past each launch.
    for (scale, what) in [(qk_scale(), "Q's real scale"), (1.0, "K's unit scale")] {
        let prior = fixture::sync_read(&b);
        fixture::expect_guard(call(2, 8, eps(), scale), None, what);
        assert_ne!(
            fixture::sync_read(&b),
            prior,
            "{what} was accepted but left the buffer byte-identical, so nothing launched"
        );
    }
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
/// post-norm, and re-norming a normed vector moves it only 9.863e-5 under this file's own metric
/// (host-recomputed on `fill(64*128, 25, 0.4)`; the closed form `eps/2m` predicts 9.375e-5 and the
/// difference is `worst_rel` dividing by a tensor-wide max while the displacement is per row), against
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
    // put the identity at 6.98e-2 / 1.19e-2 / 1.33e-3, well clear of the f32 floor.
    //
    // **They are BELOW the reference's own regime, deliberately, and an earlier version of this
    // comment claimed the opposite.** `x` here is the norm's INPUT, not a post-norm activation —
    // post-norm activations sit at mean(y²) ≈ 1 by construction, which is what the anchor's axis
    // test asserts. These scales give m = 1.33e-4 / 8.33e-4 / 7.50e-3, and `tolerance.rs` bounds the
    // reference's SMALLEST input mean-square at 1.233e-2 from that same axis run, so all three are
    // under it. That direction is the safe one — smaller m makes `eps/(m+eps)` larger and the test
    // more sensitive — but the constraint runs the other way from "this is where the model sits",
    // and a reader who believed that would widen the sweep upward and reproduce the original red.
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
    // `worst` starts at 0.0 and `worst < 1e-2` passes over an empty loop. The bounds are literals so
    // that is unreachable today, but this tree has shipped an `assert_eq!(0, 0)` once.
    assert!(worst > 0.0, "no rows were scored — the sweep ran empty");
    // **Measured 1.4222e-4 — a 70x margin under the bound, and 1.6x ABOVE the f32 noise floor I
    // predicted (9.0e-5) when picking the regime.** The prediction is left here rather than quietly
    // replaced because the direction is the useful part: an f32 error estimate over a 128-term sum
    // undershoots, so a bound set at the predicted floor would have been a coin flip. 1e-2 is loose
    // against the MEASUREMENT and tight against every other placement AT THESE SCALES — measured on
    // the host: eps outside the sqrt scores 9.77e-1 / 9.44e-1 / 8.27e-1, eps on the rsqrt result and
    // eps missing entirely both score ~1.0, and eps on the SUM instead of the mean 9.92e-1.
    //
    // **The discriminant has a ZERO, and the sweep must stay away from it — this is a precondition,
    // not a noise argument.** For `y = x/(sqrt(m)+eps)` the measured quantity lands at `2*eps/sqrt(m)`
    // against a `want` of `eps/m`, so the relative separation is exactly `|2*sqrt(m) - 1|` — no eps
    // in it, and it VANISHES at m = 1/4. At scale_of_x 0.866 (m = 0.25) that wrong placement scores
    // 2.93e-5, three decades UNDER this bound, and the test would pass it. The earlier version of
    // this line claimed the separation was `sqrt(m)/eps`, ~1e3, which is a different quantity
    // altogether and hid the zero. So there are two independent reasons the sweep stays small — the
    // f32 noise floor above, and this — and only the first was written down. Review, 2026-08-13.
    assert!(
        worst < 1e-2,
        "1 - mean(y^2) is {worst:e} away from eps/(m+eps) — the eps is not inside the mean"
    );
}
