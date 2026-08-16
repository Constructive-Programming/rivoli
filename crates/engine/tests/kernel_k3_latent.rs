//! **Kimi-K3's MoE latent sandwich — WHICH RMSNorm, and whether the trunk GEMV holds at
//! 7168.** Ported from `k3:tests/k3_kernels.rs` item 3 (banner at :1183); shared spine in
//! `tests/k3/mod.rs`.
//!
//! §6's order is `down(x) -> experts in latent space -> RMSNorm the AGGREGATE -> up(...)`,
//! and the port's answer to it is three kernels this engine already has: `gemm_bf16` twice
//! and one RMSNorm. None of the three is an M9 deferral — they are covered launchers — so
//! this suite's census weight is zero; it exists because the k3 tree's item 3 answers two
//! questions the covered suites do not ask: **which of this engine's two RMSNorms is K3's**
//! (the other one FAILS the fixture, asserted, not commented), and whether a matmul verified
//! at vocab 1024 / dim 512 holds at K3's 7168-wide trunk.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! * Swap [`the_latent_norm_matches_the_anchor_at_every_moe_layer`]'s launcher for
//!   `launch_rmsnorm_batch` (one line at [`device_rmsnorm_single`]). It must go RED at
//!   3.299e-3 against the 6.3e-4 tolerance — and
//!   [`the_batch_rmsnorm_would_fail_this_fixture`] must go red TOO, since the kernel it
//!   asserts fails now "passes": the pair flipping together is the proof both are live.
//! * In `kernels/linalg.hip`'s `gemm_bf16`, read the weight as fp16 instead of bf16.
//!   [`the_trunk_gemv_matches_an_f64_dot_at_k3_widths`] must go red in the 1e-3 region —
//!   its own comment says a miss that size means "reading the weights as something other
//!   than bf16", and this mutation is exactly that.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::{launch_rmsnorm_batch, launch_rmsnorm_single};
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};

mod common;
mod k3;

use k3::*;

/// The captured layers that own a `block_sparse_moe`.
///
/// Layer 0 is absent and its absence is load-bearing: `first_k_dense_replace` is 1, so layer
/// 0 is dense and has no MoE block at all. Naming the set keeps that a statement rather than
/// an accident of what happened to be in the file (`k3:tests/k3_kernels.rs:1190`).
const MOE_LAYERS: [usize; 5] = [1, 3, 12, 91, 92];

/// One latent RMSNorm: the expert aggregate in, the learned weight, and what the reference
/// made. The aggregate is the capture the k3 tree's item 3 added — `moe_infer`'s return is
/// not a module call, so no forward hook could see it and this operator had an output with
/// no input until 2026-08-12 (`k3:tests/k3_kernels.rs:1197`).
struct LatentNorm {
    x: Vec<f32>,
    w: Vec<f32>,
    want: Vec<f32>,
}

fn latent_norm(g: &GoldenSet, layer: usize) -> LatentNorm {
    let m = format!("model.layers.{layer}.block_sparse_moe.routed_expert_norm");
    let (ws, w) = float(g, &format!("{m}.weight"));
    let latent = ws[0];
    let (xs, x) = float(g, &format!("{m}.in"));
    let (os, want) = float(g, &m);
    assert_eq!(xs, [1, latent], "{m}: the aggregate is one row of latent");
    assert_eq!(os, [1, latent], "{m}: the norm is width-preserving");
    let [x, w, want] = [x, w, want].map(<[f32]>::to_vec);
    LatentNorm { x, w, want }
}

/// Every (draw, MoE layer) pair, with the reference's own eps in hand.
fn for_each_latent_norm(mut f: impl FnMut(&str, usize, f32, LatentNorm)) {
    for (salt, bytes) in GOLDENS {
        let g = load(bytes);
        let e = eps(&g);
        for layer in MOE_LAYERS {
            f(salt, layer, e, latent_norm(&g, layer));
        }
    }
}

/// `KimiRMSNorm.forward` in f64 — `weight * (x * rsqrt(mean(x²) + eps))`.
///
/// The statistic is f64 so this is a floor rather than a second f32 implementation, and the
/// eps goes INSIDE the mean's square root, which is the placement `rmsnorm_single` uses and
/// the one trap this three-line operator has (`k3:tests/k3_kernels.rs:1233`).
fn host_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    let rs = 1.0 / (ms + f64::from(eps)).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &q)| (f64::from(v) * rs * f64::from(q)) as f32)
        .collect()
}

/// `y[j] = Σ_i x[i]·bf16(w[j][i])` in f64 — `gemm_bf16`'s oracle at `m == 1`.
///
/// The weights are handed in ALREADY bf16-coded and widened here, so the only thing
/// separating this from the kernel is the summation order: the kernel's `wave_sum` shuffle
/// ladder against this sequential f64 accumulation. Rounding the weights inside the oracle
/// instead would let a kernel that silently read them as f32 agree with it
/// (`k3:tests/k3_kernels.rs:1247`).
fn host_gemv(x: &[f32], w: &[u16], n: usize, k: usize) -> Vec<f32> {
    sums64(n, |j| {
        (0..k)
            .map(|i| f64::from(x[i]) * f64::from(bf16_to_f32(w[j * k + i])))
            .sum()
    })
}

fn device_rmsnorm_single(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
    let mut yb = zeros(x.len() * 4);
    // SAFETY: `x`, `w` and `y` are each `n` live f32 and mutually non-aliasing, as required;
    // `back` synchronises before any buffer drops.
    ok(
        unsafe {
            launch_rmsnorm_single(
                xb.ptr() as *const f32,
                wb.ptr() as *const f32,
                x.len(),
                eps,
                yb.ptr_mut() as *mut f32,
            )
        },
        "rmsnorm_single",
    );
    f32v(&back(&yb))
}

/// `out[j] = Σ_i x[i]·bf16(w[j][i])` on the device, at `m == 1` — one decode row, through
/// the house `GemmBf16` bundle so the seven-argument launch is spelled once in the tree.
fn device_gemv(x: &[f32], w: &[u16], n: usize) -> Vec<f32> {
    let s = stream();
    let (xb, wb) = (dev(&f32b(x)), dev(&u16b(w)));
    let mut ob = zeros(n * 4);
    // SAFETY: `x` is `k` live f32, `w` is `n·k` live u16, `out` is `n` writable f32, none
    // aliasing; all outlive the stream's completion, which `back` synchronises on.
    unsafe {
        gemm_bf16_launch(
            GemmBf16 {
                x: xb.ptr() as *const f32,
                w: wb.ptr() as *const u16,
                out: ob.ptr_mut() as *mut f32,
                m: 1,
                n,
                k: x.len(),
            },
            s.raw(),
        );
    }
    f32v(&back(&ob))
}

/// **`rmsnorm_single` reproduces the latent norm at every MoE layer of both draws.**
///
/// `rmsnorm_single` and not `rmsnorm_batch`, and the choice is the content of this item —
/// see [`the_batch_rmsnorm_would_fail_this_fixture`]. At decode there is one row, which is
/// the case `dim3(1)` computes correctly; prefill needs a row-wise kernel that does NOT
/// bf16-round, which is neither of the two this engine has today
/// (`k3:tests/k3_kernels.rs:1303`).
#[test]
fn the_latent_norm_matches_the_anchor_at_every_moe_layer() {
    let tol = tolerance::rel_tolerance("moe_latent");
    for_each_latent_norm(|salt, layer, eps, ln| {
        let r = rel(&device_rmsnorm_single(&ln.x, &ln.w, eps), &ln.want);
        assert!(
            r <= tol,
            "{salt} layer {layer}: the latent norm differs by {r:e}, over {tol:e}"
        );
        // `tol` is 6.3e-4 and this kernel lands at 1.3e-7 — against `tol` alone a
        // THREE-order degradation would pass in silence. Measured worst over both draws and
        // all five layers, then 10x; three of the ten cells are BIT-EXACT
        // (`k3:tests/k3_kernels.rs:1315`).
        tripwire(
            r,
            Bars {
                tol,
                observed: 1.307e-7,
            },
            &format!("{salt} layer {layer} latent norm"),
        );
    });
}

/// **`rmsnorm_batch` would fail the fixture above, and that is why this item does not use
/// it.**
///
/// It is width-generic — and the width was never the problem. Its last line rounds its store
/// to bf16, because V4's `RMSNorm.forward` stores bf16 and that kernel is V4's.
/// `KimiRMSNorm.forward` is `self.weight * x.to(dtype)`, and in this fp32 reference
/// `to(dtype)` is a no-op — so the whole bf16 step is arithmetic the reference does not
/// perform. Measured: **3.299e-3 against the 6.3e-4 tolerance** (5.24x over; the assert
/// prints both). Asserted as a FAILURE rather than left as a comment: a claim that one of
/// two interchangeable kernels is wrong is exactly the claim that rots, and this goes red
/// the day someone changes either kernel's store.
///
/// **The obvious rescue does not work.** A fixture can absorb a store deviation by rounding
/// the REFERENCE's output the same way — score `rbf16(want)` — and that would let
/// `rmsnorm_batch` pass. It would also be wrong: `KimiRMSNorm` rounds to the input dtype
/// BEFORE the weight multiply, so the real bf16 model computes `w · bf16(x·rs)` while this
/// kernel computes `bf16(w · x·rs)` — a different function, not a differently-placed copy of
/// the same one. rivoli's trunk carries f32 activations throughout, which is the engine-wide
/// deviation this port inherits, and `rmsnorm_single`'s f32 store is what matches it
/// (`k3:tests/k3_kernels.rs:1324`).
#[test]
fn the_batch_rmsnorm_would_fail_this_fixture() {
    let tol = tolerance::rel_tolerance("moe_latent");
    let s = stream();
    let mut worst = 0.0f32;
    for_each_latent_norm(|_, _, eps, ln| {
        let mut xb = dev(&f32b(&ln.x));
        let wb = dev(&f32b(&ln.w));
        // SAFETY: `x` is `rows·d` live f32 written in place, `w` is `d` live f32; they do
        // not alias and both outlive the stream, which `back` synchronises on.
        ok(
            unsafe {
                launch_rmsnorm_batch(
                    xb.ptr_mut() as *mut f32,
                    wb.ptr() as *const f32,
                    1,
                    ln.x.len(),
                    eps,
                    s.raw(),
                )
            },
            "rmsnorm_batch",
        );
        worst = worst.max(rel(&f32v(&back(&xb)), &ln.want));
    });
    assert!(
        worst > tol,
        "`rmsnorm_batch` scored {worst:e}, INSIDE the {tol:e} this operator is held to. Its \
         bf16 store was the whole reason K3 uses `rmsnorm_single` instead. If the store has \
         been made optional, use it here and delete this test rather than loosening it."
    );
}

/// **`gemm_bf16` at K3's real trunk widths, against an f64 dot on the same bf16 weights.**
///
/// `gemm_bf16` carries the K3 trunk unchanged but is verified only at vocab 1024 / dim 512,
/// and the latent sandwich runs it at `7168 -> 3584` and `3584 -> 7168`. It has no anchor
/// bucket and cannot get one: the anchor is fp32 (one of its declared deviations) while
/// these weights are bf16, so an anchor comparison would be dominated by a ~2^-9
/// quantisation the reference never applied. What is left to check is the part that is
/// genuinely the kernel's — the `wave_sum` shuffle ladder re-associating a 7168-term sum
/// (`k3:tests/k3_kernels.rs:1386`).
#[test]
fn the_trunk_gemv_matches_an_f64_dot_at_k3_widths() {
    // **This bound is the test's own, NOT a tolerance-table row** — the table's numbers all
    // derive from the anchor's floor-vs-defect pair, and this operator has no bucket.
    // Measured worst over the four cases below, at the deepest reduction (n=3584, k=7168) as
    // the error model predicts, then 10x. An f32 accumulator over 7168 terms against an f64
    // one is the only difference here, so this is small on purpose: a number in the 1e-3
    // range would mean the kernel is reading the weights as something other than bf16, not
    // that it re-associated (`k3:tests/k3_kernels.rs:1399`).
    const OBSERVED_WORST: f32 = 2.705e-7;
    let mut r = Lcg(0x3EA7);
    // `(n, k)`: the sandwich's two projections at the real widths, then the tiny model's
    // pair so a failure that is about the width shows up as one.
    for &(n, k) in &[(3584usize, 7168usize), (7168, 3584), (96, 192), (192, 96)] {
        let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
        let w: Vec<u16> = (0..n * k).map(|_| f32_to_bf16(r.f())).collect();
        let got = device_gemv(&x, &w, n);
        let d = rel(&got, &host_gemv(&x, &w, n, k));
        assert!(
            d <= OBSERVED_WORST * 10.0,
            "n={n} k={k}: {d:e} exceeds {:e} — an f32 accumulator over {k} terms against an \
             f64 one should not drift this far, so the re-association is not the explanation",
            OBSERVED_WORST * 10.0
        );
    }
}

/// **The norm goes BEFORE the up projection, and doing it after is a different function.**
///
/// The fixture-level twin of `--defect LatentNormAfterUp`, which the anchor prices at
/// 2.05e+2 against a 6.287e-5 floor. Built the same way the defect is — the norm's weight
/// collapsed to its own mean so it is applicable at `hidden` width — so this asks about the
/// ORDER and not about the values. Synthetic, because the projection weights are
/// deliberately not in the goldens; what is being pinned is which of two orders the device
/// chain implements, and that does not need the reference's particular matrix
/// (`k3:tests/k3_kernels.rs:1424`).
#[test]
fn norming_after_the_up_projection_is_a_different_sandwich() {
    let (latent, hidden) = (3584usize, 7168usize);
    let eps = 1e-5f32;
    let mut r = Lcg(0x5A9D);
    let acc: Vec<f32> = (0..latent).map(|_| r.f()).collect();
    // `uniform(0.8, 1.2)`, the range `init_weights` draws every norm weight from — `r.f()`
    // is [-1, 1), not [0, 1). A norm weight near zero would make every downstream value a
    // denormal and the fixture a comparison of noise.
    let nw: Vec<f32> = (0..latent).map(|_| 1.0 + 0.2 * r.f()).collect();
    let up: Vec<u16> = (0..hidden * latent).map(|_| f32_to_bf16(r.f())).collect();

    let ordered = device_gemv(&device_rmsnorm_single(&acc, &nw, eps), &up, hidden);
    // The device chain is the specification's order, so the host oracle of that order must
    // agree. 10x the measured 2.35e-7: two kernels deep — the norm's f32 statistic and the
    // GEMV's f32 accumulation over 3584 terms — so it is looser than either alone and still
    // three orders under the separation asserted below.
    let want = host_gemv(&host_rmsnorm(&acc, &nw, eps), &up, hidden, latent);
    let good = rel(&ordered, &want);
    assert!(
        good <= 2.35e-6,
        "the specified order disagrees with its own oracle: {good:e}"
    );

    // Norm-after-up, at `hidden` width with the mean weight — the defect's own construction.
    //
    // This second half exists so the first assertion is not vacuous: a `good` of 1e-6 means
    // nothing unless the wrong order is somewhere else entirely. It is NOT a sensitivity
    // measurement — the two orders differ mostly in SCALE, because norming last leaves an
    // output of magnitude ~1 where norming first leaves the projection's own ~35, which is
    // also why the anchor prices this defect at 2.05e+2 rather than at something subtle
    // (`k3:tests/k3_kernels.rs:1458`).
    let mean = nw.iter().sum::<f32>() / latent as f32;
    let flipped = host_rmsnorm(
        &host_gemv(&acc, &up, hidden, latent),
        &vec![mean; hidden],
        eps,
    );
    let moved = rel(&flipped, &want);
    assert!(
        moved > 1.0e-2,
        "norming after the up projection moved the output by only {moved:e}, so the two \
         orders are not separated here and the agreement above proves nothing"
    );
}
