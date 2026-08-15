//! The kernels whose operand is a VECTOR rather than a weight matrix, each against a host
//! oracle written beside it: `argmax_reduce` and `vadd` (fwd.hip glue), `index_topk` (the
//! attention row selector), `swiglu` and `vq_encode` (linalg.hip), and `rope_interleave`
//! (mla.hip's rotary).
//!
//! **Split out of `kernel.rs` on 2026-08-15, with the MoE oracles that went to
//! `kernel_moe.rs` — by COHESION, not by size alone.** `kernel.rs` had reached 2263 lines
//! and 79 functions over nine unrelated kernel families; CodeScene scored it 8.03 on
//! "Low Cohesion" and on the function count. These six tests are what was left once the
//! GEMV/MLA/attend suites (which share `GemvIo`, `MlaIo`, `AttIo` and the batched-row
//! claim) and the MoE suite (which shares `ExpertDesc` construction and the fixed-point
//! accumulator) were each taken whole.
//!
//! **What they share is the shape of the test, and it is a real shape:** one launcher, one
//! fixture drawn from `Lcg`, one host reference spelled inline, no scaffolding held in
//! common with anything else in the tree. Four of them were uncovered until 2026-08-06 and
//! landed together under `kernel.rs`'s "`linalg.hip`'s remaining launchers" banner —
//! `tests/kernel_coverage.rs` named them, which is what that census is for, and the census
//! scans every `.rs` under `crates/engine/tests`, so it followed this move with no edit.
//!
//! `gemv_i4` is the one that did NOT come: it is a vector-against-matrix dot like its
//! `gemv_fp8`/`gemv_i8` neighbours and shares `GemvIo` with them, so it stayed.
//!
//! Every body below travelled VERBATIM with its comments — in this repo a comment carries
//! the measurement that justified the choice, so a re-worded one loses evidence.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_artifact::quant::{VQ_DIM, VQ_K, codebook_norms};
use rivoli_backend::hip::{
    device_sync, launch_argmax, launch_index_topk, launch_rope_interleave, launch_swiglu,
    launch_vadd, launch_vq_encode,
};

mod common;
use common::{
    DeviceBuf, Lcg, assert_bits, assert_bitwise, assert_close, assert_rel, back, dev, f32b, f32v,
    u16v, zeros,
};

/// `argmax_reduce` into the two output words, idx then val.
fn argmax(logits: &DeviceBuf, n: usize, idx: &mut DeviceBuf, val: &mut DeviceBuf) {
    // SAFETY: `logits` is n live f32 and each output buffer is one live word.
    unsafe {
        launch_argmax(
            logits.ptr() as *const f32,
            n,
            idx.ptr_mut() as *mut i32,
            val.ptr_mut() as *mut f32,
        )
    }
    .expect("launch argmax");
}

/// argmax_reduce: max value wins, ties → lowest index, NaN never wins. Plus vadd
/// as a residual-add smoke check (both fwd.hip glue).
#[test]
fn fwd_argmax_and_vadd() {
    // argmax: a plateau (tie → lowest index) with a NaN that must lose.
    let mut logits = vec![0.1f32, 0.5, 0.5, f32::NAN, 0.3, 0.5, -1.0];
    let want_idx = 1i32; // first 0.5
    let lb = dev(&f32b(&logits));
    let mut ib = dev(&[0u8; 4]);
    let mut vb = dev(&[0u8; 4]);
    argmax(&lb, logits.len(), &mut ib, &mut vb);
    device_sync().expect("sync");
    let w4 = |d: &DeviceBuf| -> [u8; 4] { d.copy_out().expect("out")[..4].try_into().expect("4") };
    let got_idx = i32::from_le_bytes(w4(&ib));
    let got_val = f32::from_le_bytes(w4(&vb));
    assert_eq!(got_idx, want_idx, "argmax idx");
    assert_eq!(got_val, 0.5, "argmax val");

    // vadd: x += y, elementwise.
    let y: Vec<f32> = logits
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    for l in logits.iter_mut() {
        if l.is_nan() {
            *l = 0.0;
        }
    }
    let mut xb = dev(&f32b(&logits));
    let yb = dev(&f32b(&y));
    unsafe {
        launch_vadd(
            xb.ptr_mut() as *mut f32,
            yb.ptr() as *const f32,
            logits.len(),
        )
        .expect("vadd");
    }
    device_sync().expect("sync");
    let got = f32v(&xb.copy_out().expect("out"));
    let want: Vec<f32> = logits.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_close(&want, &got, "vadd");
}

/// `index_topk` vs the host selection it replaces, on the shapes that actually occur.
///
/// The oracle is the engine's own two lines — `topk_into(scores, k, &mut sel)` then
/// `sel.sort_unstable()` — not a reimplementation, so this pins the kernel to what the
/// attend has always consumed rather than to my reading of it.
///
/// **The row buffer is sentinel-filled and the tail is asserted untouched.** Without
/// that, over-selection is invisible: the readback would be truncated to `want.len()`,
/// and whenever the correct answer is an index *prefix* — which it is for the
/// `ReLU-sparse` case, since its non-zero scores sit at indices 0..300 — a kernel that
/// emitted every tied row would still match on the first k. Measured against a serial
/// simulation of this kernel: of three mutations (drop the cross-chunk tie carry; use
/// `<= need` for the tie budget; drop the -0.0 canonicalisation), the tie carry is
/// caught by seven cases on selection alone, the canonicalisation only by
/// `mixed +0.0/-0.0`, and **`<= need` by nothing except the tail check**.
///
/// The `ReLU-sparse` and `scattered zeros` cases are tie-DOMINATED, which is the regime
/// where the index-ascending rule decides the bulk of the selection rather than a
/// handful of boundary entries. Whether the engine actually produces such an array is
/// unmeasured (docs/investigations/npu-offload.md), so these are chosen as the hardest case for the tiebreak,
/// not as a claim about production data. `scattered zeros` additionally makes the answer
/// non-prefix, which is the combination nothing else here covers — and note the two
/// differ in ORDER as well as scatter: `ReLU-sparse` is pre-sorted into the host
/// comparator's own order, which is its best case and a trap when timing rather than
/// checking. nt = 5185 and k = 2048 are the longer in-engine context and `index_topk`.
#[test]
fn index_topk_matches_host_selection() {
    fn host(scores: &[f32], k: usize) -> Vec<u32> {
        let mut sel = Vec::new();
        rivoli_core::routing::topk_into(scores, k, &mut sel);
        sel.sort_unstable();
        sel.iter().map(|&i| i as u32).collect()
    }
    let mut rng = Lcg(0x7071_C0DE);
    let nt = 5185usize;
    // Realistic shape, answer is a prefix: real scores at the front, rest ReLU'd to 0.0.
    let mut relu_sparse = vec![0.0f32; nt];
    for (i, x) in relu_sparse.iter_mut().enumerate().take(300) {
        *x = (300 - i) as f32 * 0.25;
    }
    // Realistic shape, answer is NOT a prefix: the same sparsity, scattered.
    let mut scattered = vec![0.0f32; nt];
    for j in 0..300 {
        scattered[(j * 7919) % nt] = (300 - j) as f32 * 0.25;
    }
    let dense: Vec<f32> = (0..nt).map(|_| rng.f() * 8.0).collect();
    let heavy_ties: Vec<f32> = (0..nt).map(|_| (rng.f() * 4.0).floor()).collect();
    let ramp = |n: usize, m: usize| -> Vec<f32> { (0..n).map(|i| (i % m) as f32).collect() };
    let signed_zeros: Vec<f32> = (0..4096)
        .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
        .collect();
    let negatives: Vec<f32> = (0..4096).map(|i| -((i % 11) as f32)).collect();
    let cases: Vec<(&str, Vec<f32>, usize)> = vec![
        ("mixed +0.0/-0.0", signed_zeros, 2048),
        ("negatives only", negatives, 2048),
        ("k == nt", ramp(2048, 7), 2048),
        ("k == nt - 1", ramp(2049, 7), 2048),
        ("k > nt (wrapper clamp)", ramp(500, 7), 2048),
        ("single block", ramp(200, 5), 64),
        (
            "ReLU-sparse (engine shape, prefix answer)",
            relu_sparse,
            2048,
        ),
        (
            "scattered zeros (engine shape, non-prefix answer)",
            scattered,
            2048,
        ),
        ("dense random", dense, 2048),
        ("heavy ties", heavy_ties, 2048),
    ];
    for (name, scores, k) in cases {
        check_topk_case(name, &scores, k, &host(&scores, k));
    }
}

/// One `index_topk` case: the kernel's rows against `want`, its tail against the sentinel,
/// and the order it promises. `want` is the host oracle's answer for the same `(scores, k)`.
fn check_topk_case(name: &str, scores: &[f32], k: usize, want: &[u32]) {
    const SENTINEL: u32 = 0xFFFF_FFFF;
    let n = scores.len();
    let written = k.min(n);
    let sb = dev(&f32b(scores));
    // Sentinel fill over `max(n, k)` slots: anything written past `min(k, n)` survives as a
    // non-sentinel word, which is what the tail assertion below reads.
    let mut rb = dev(&vec![0xFFu8; n.max(k) * 4]);
    // SAFETY: scores holds n f32; rows holds >= min(k,n) u32.
    unsafe {
        launch_index_topk(sb.ptr() as *const f32, n, k, rb.ptr_mut() as *mut u32)
            .expect("index_topk");
    }
    device_sync().expect("sync");
    let got = common::u32v(&rb.copy_out().expect("rows out"));
    assert_eq!(
        want.len(),
        written,
        "oracle wrote {} rows, expected {written} on {name}",
        want.len()
    );
    assert_eq!(
        &got[..written],
        want,
        "index_topk selection differs on {name}"
    );
    assert!(
        got[written..].iter().all(|&v| v == SENTINEL),
        "index_topk wrote past min(k,nt)={written} on {name} — over-selection"
    );
    assert!(
        got[..written].windows(2).all(|w| w[0] < w[1]),
        "index_topk output not strictly ascending on {name}"
    );
}

/// `swiglu` — the dense fp8 MLP's combine, `h = silu(g)·u`.
///
/// **The oracle is NOT `math::silu`, and that is the whole subtlety of this test.**
/// `math::silu` is `x·sigmoid(x)`, the MULTIPLY form; `linalg.hip::swiglu` is
/// `gv/(1 + e^-gv)`, the DIVISION form. `kernels/linalg.hip` records the pair as "one
/// rounding apart, which would normally vanish under the bf16 store ... except exactly at a
/// rounding boundary" — but there is no bf16 store on THIS path, so the difference reaches
/// the output directly and an oracle that reached for `silu` would be measuring the wrong
/// function at a tolerance loose enough to hide it.
///
/// Run IN PLACE (`h` aliases `g`), because that is how `gpu.rs:2010` and `f4gpu.rs:1406` —
/// the only two callers — launch it, and the aliasing is a documented safety claim rather
/// than an accident. A kernel that read `g[i]` after writing `h[i]` would still pass a
/// non-aliased test.
#[test]
fn swiglu_matches_the_division_form_in_place() {
    let n = 4096;
    let mut r = Lcg(0x5717);
    // ±6 rather than ±1: silu is very nearly linear on [-1, 1], so a defect that dropped
    // the sigmoid entirely would land inside a relative tolerance on that range. The
    // saturating tails are where the function has shape.
    let g: Vec<f32> = (0..n).map(|_| r.f() * 6.0).collect();
    let u: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| (gv / (1.0 + (-gv).exp())) * uv)
        .collect();

    let mut gb = dev(&f32b(&g));
    let ub = dev(&f32b(&u));
    // SAFETY: `g`, `u` and `h` are each `n` live device f32; `h == g` is the aliasing the
    // launcher's contract permits — every thread reads both operands, then writes once.
    let gp = gb.ptr_mut() as *mut f32;
    let null = std::ptr::null_mut();
    unsafe { launch_swiglu(gp as *const f32, ub.ptr() as *const f32, n, gp, null) }
        .expect("swiglu");
    let got = f32v(&back(&gb));

    // 1e-5 relative, not `assert_close`'s `1e-3·max + 1e-3`. The only honest disagreement
    // is device `expf` against Rust's, and a relative error `e` in `e^-g` becomes at most
    // `e` in `1/(1+e^-g)` — so the bound is a few ULP of the result. It is 1e-5 rather than
    // 1e-6 because ROCm's `expf` is specified to a few ULP, not correctly rounded, and one
    // device window is not the place to discover the difference. At this fixture's scale
    // the shared floor would be ~7% of the signal and would pass a kernel that had dropped
    // `u` entirely over half its range; 1e-5 is still ~700x tighter than that.
    assert_rel(&want, &got, "swiglu (in place)", 1e-5);
}

/// `rope_interleave` — the GLM rotary, `count` segments of `seg` at `stride`.
///
/// Two arms, and the first is the sharp one:
///
/// * **pos 0 is a bit-exact PERMUTATION.** The rotation is the identity there
///   (`cos 0 = 1`, `sin 0 = 0`), so `v[j] = v[2j]` and `v[half+j] = v[2j+1]` exactly — no
///   transcendental, no tolerance, one thread per element. This is what pins the layout
///   half of the kernel: the adjacent-pair READ and the half-split WRITE.
///   `kernels/mla.hip` calls confusing that permutation with V4's adjacent-pair rotation
///   "the single most likely silent-wrong" in the port, and both spellings produce fluent
///   text, so it is worth a gate that cannot be widened.
/// * **A real position** for the angles, at a tolerance, since `pow`/`cos`/`sin` are libm
///   on both sides and are not required to agree bit for bit.
///
/// `stride > seg` in both arms. Every production call but one passes `stride == seg`;
/// `gpu.rs:1886` ropes each of `h` query heads at `stride = qh` over `seg = rope`, so a
/// kernel that walked `seg` instead of `stride` would be wrong there alone.
#[test]
fn rope_interleaves_pairs_and_is_a_permutation_at_position_zero() {
    let (count, stride, seg, theta) = (5usize, 24usize, 16usize, 10000.0f64);
    let half = seg / 2;
    let mut r = Lcg(0x2909);
    let base: Vec<f32> = (0..count * stride).map(|_| r.f()).collect();

    let run = |pos: usize| -> Vec<f32> {
        let mut b = dev(&f32b(&base));
        // SAFETY: `base` is `count * stride` live device f32 for the whole call.
        unsafe { launch_rope_interleave(b.ptr_mut() as *mut f32, count, stride, seg, pos, theta) }
            .expect("rope_interleave");
        f32v(&back(&b))
    };

    let host = |pos: usize| -> Vec<f32> {
        let mut v = base.clone();
        for s in 0..count {
            let row = &base[s * stride..s * stride + seg];
            for j in 0..half {
                let (a, b) = (row[2 * j], row[2 * j + 1]);
                // f64 throughout, matching the kernel: it computes the angle in double and
                // narrows only the final cos/sin. Doing the ladder in f32 would disagree
                // with a CORRECT kernel by more than the tolerance below at large `pos`,
                // which is the arg-reduction the double is there for.
                let ang = pos as f64 * theta.powf(-2.0 * j as f64 / seg as f64);
                let (cs, sn) = (ang.cos() as f32, ang.sin() as f32);
                v[s * stride + j] = a * cs - b * sn;
                v[s * stride + half + j] = b * cs + a * sn;
            }
        }
        v
    };

    assert_bits(
        &host(0),
        &run(0),
        "rope at pos 0 (pure de-interleave permutation)",
    );
    assert_rel(&host(137), &run(137), "rope at pos 137", 1e-6);
}

/// `vq_encode` — the offline converter's argmin accelerator.
///
/// **The reference is computed a DIFFERENT way from the kernel, on purpose.** The kernel
/// minimises `‖c_k‖² − 2·s·c_k`, which is half the flops and the same argmin; this
/// minimises the true squared distance `Σ_d (s_d − c_kd)²`. `quant_vq`'s own `nearest` is
/// the first form and is a private closure, so re-spelling it here would be a
/// transliteration that shares the kernel's algebra — and a misread sign or tie-break would
/// be invisible because both sides carry it. `build.rs`'s duplication gate said the same
/// thing about the same block, which is the second reason it is not that.
///
/// The independent form is also what makes `cbnorm` an actual input rather than a shared
/// assumption: it is the ONE argument the kernel takes on trust, and if `codebook_norms`
/// were wrong the kernel would pick a non-nearest codeword and this would say so. That is
/// why the production precompute is used rather than a test-local one — the kernel is only
/// ever fed that function's output (`bin/convert.rs`).
#[test]
fn vq_encode_picks_the_nearest_codebook_entry() {
    let n = 512;
    let mut r = Lcg(0x7A11);
    let cb: Vec<f32> = (0..VQ_K * VQ_DIM).map(|_| r.f()).collect();
    let sub: Vec<f32> = (0..n * VQ_DIM).map(|_| r.f()).collect();
    let cbnorm = codebook_norms(&cb);

    let idxb = {
        let (sb, cbb, nb) = (dev(&f32b(&sub)), dev(&f32b(&cb)), dev(&f32b(&cbnorm)));
        let mut idxb = zeros(n * 2);
        // SAFETY: `sub` is n·VQ_DIM f32, the codebook VQ_K·VQ_DIM f32, `cbnorm` VQ_K f32,
        // and `idx` n u16 — all live for the call.
        unsafe {
            launch_vq_encode(
                sb.ptr() as *const f32,
                cbb.ptr() as *const f32,
                nb.ptr() as *const f32,
                n,
                idxb.ptr_mut() as *mut u16,
            )
        }
        .expect("vq_encode");
        u16v(&back(&idxb))
    };

    let dist2 = |i: usize, k: usize| -> f32 {
        (0..VQ_DIM)
            .map(|d| {
                let e = sub[i * VQ_DIM + d] - cb[k * VQ_DIM + d];
                e * e
            })
            .sum()
    };
    let mut want = Vec::with_capacity(n);
    let mut margin = f32::INFINITY;
    for i in 0..n {
        let (bk, gap) = nearest_two(VQ_K, |k| dist2(i, k));
        want.push(bk);
        margin = margin.min(gap);
    }

    // Exact index equality is only DECIDABLE while the winner is strictly separated: the
    // kernel's `‖c‖² − 2·s·c` and this `Σ(s−c)²` have the same argmin in exact arithmetic
    // and may round to different orders at a true tie, so a near-tie would turn a correct
    // kernel red. Printed as well as asserted — a seed whose margin collapsed would
    // otherwise fail with no clue that the fixture, not the kernel, had moved.
    println!("vq_encode: tightest runner-up margin over {n} subvectors = {margin:.3e}");
    // The threshold has to exclude margins the two ALGEBRAS cannot separate, not just exact
    // ties: at VQ_DIM 4 both discriminants carry ~1e-6 of independent f32 rounding, so a
    // 2e-6 margin would be undecidable. Measured at this seed the tightest is 9.94e-5, so
    // 1e-5 sits ~10x under the fixture and ~5x over the rounding floor.
    assert!(
        margin > 1e-5,
        "fixture has a near-tie; exact indices are not decidable"
    );
    assert_bitwise(&want, &idxb, "vq_encode indices");
}

/// `(argmin index, runner-up gap)` over `d2(k)` for `k` in `[0, n)`. The gap is what decides
/// whether an exact index comparison is meaningful at all, so the scan that finds the winner
/// is also the one that reports how far ahead it was — a second pass could disagree.
fn nearest_two(n: usize, d2: impl Fn(usize) -> f32) -> (u16, f32) {
    let (mut lo, mut second, mut bk) = (f32::INFINITY, f32::INFINITY, 0u16);
    for k in 0..n {
        match d2(k) {
            d if d < lo => (second, lo, bk) = (lo, d, k as u16),
            d if d < second => second = d,
            _ => {}
        }
    }
    (bk, second - lo)
}
