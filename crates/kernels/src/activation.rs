//! Per-token activation transforms — the twins of `kernels/activation.hip` at tag
//! `archive/hip-kernels`: the gated combines, RMSNorm in three forms, `act_quant_f8`, and
//! RoPE in two. That file's comments carry each kernel's contract (aliasing, which model
//! uses it, why a near-twin is a different function); they are not repeated here. What
//! this file adds is what the port had to decide: where hipcc fused, and what replaced
//! `blockDim`.
//!
//! In-place callers (`h == g`, `y == x`) are why no pointer here is `noalias`: HIP's
//! `__restrict__` on an aliased call is UB the C gets away with by one-read-then-one-write;
//! Rust raw pointers make no such promise to the optimizer in the first place.

use crate::gpu::{bid, fma, for_i32, lds, ocml, tid};
use crate::prelude::actquant::{
    ACT_QUANT_AMAX_FLOOR, FP8_MAX_INV, act_quant_roundtrip, fast_round_scale,
};
use crate::prelude::formats::rbf16;
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::{block_sum_lds, sync_threads, tree, wave_max};
use crate::prelude::{situ_glu, swiglu_clamped};
use core::intrinsics::{exp, sqrt};
use rivoli_abi::geom::{ACT_QUANT_BLOCK, ACT_QUANT_THREADS, ELEMENTWISE_BLOCK, ROW_BLOCK, WAVE};
use rivoli_abi::kernarg::{ActQuant, Gate, GateClamped, GateSitu, Rope, RowNorm, WeightlessNorm};

/// The gated combines' shared shape: `h[i] = f(g[i], u[i])` for the thread's one `i < n`.
/// The index is HIP's `blockIdx.x * blockDim.x + threadIdx.x` — unsigned, wrapping — read
/// as `int` for the signed bound test, exactly as the C does.
#[inline(always)]
unsafe fn gate_map(a: Gate, f: impl FnOnce(f32, f32) -> f32) {
    let i = bid().wrapping_mul(ELEMENTWISE_BLOCK).wrapping_add(tid()) as i32;
    if i < a.n {
        let i = i as usize;
        *a.h.add(i) = f(*a.g.add(i), *a.u.add(i));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn swiglu(a: Gate) {
    gate_map(a, |g, u| (g / (1.0 + exp(-g))) * u);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn swiglu_clamped_bf16(a: GateClamped) {
    gate_map(a.gate, |g, u| rbf16(swiglu_clamped(g, u, a.limit)));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn situ_glu_f32(a: GateSitu) {
    gate_map(a.gate, |g, u| situ_glu(g, u, a.b1, a.b2));
}

/// `Σ x[i]²` over this thread's `ROW_BLOCK`-strided slice of `x[0..n]`, fused per term the
/// way hipcc fuses `local += x[i] * x[i]`.
#[inline(always)]
unsafe fn strided_sum_sq(x: *const f32, n: i32) -> f32 {
    let mut local = 0.0f32;
    for_i32(tid(), n, ROW_BLOCK, |i| {
        let v = *x.add(i);
        local = fma(v, v, local);
    });
    local
}

/// `y[i] = x[i] · inv · weight(w[i])` over the thread's strided slice — `(x·inv)·weight`,
/// the C's association.
#[inline(always)]
unsafe fn row_scale(a: RowNorm, inv: f32, weight: impl Fn(f32) -> f32) {
    for_i32(tid(), a.n, ROW_BLOCK, |k| {
        *a.y.add(k) = *a.x.add(k) * inv * weight(*a.w.add(k));
    });
}

/// One row; the LDS ladder is spelled out rather than `block_sum_lds`, as in the C (which
/// argues why: that helper's trailing barrier is unmeasured on GLM's decode path).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_single(a: RowNorm) {
    let red = lds::<f32>();
    *red.add(tid() as usize) = strided_sum_sq(a.x, a.n);
    sync_threads();
    tree(tid() as usize, ROW_BLOCK as usize, |t, s| {
        *red.add(t) += *red.add(s)
    });
    let inv = 1.0 / sqrt(*red / a.n as f32 + a.eps);
    row_scale(a, inv, |w| w);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_centered_single(a: RowNorm) {
    let local = strided_sum_sq(a.x, a.n);
    let inv = 1.0
        / sqrt(
            block_sum_lds(local, lds(), tid() as usize, ROW_BLOCK as usize) / a.n as f32 + a.eps,
        );
    row_scale(a, inv, |w| 1.0 + w);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_weightless_batch(a: WeightlessNorm) {
    // `(size_t)blockIdx.x * d`: `d` sign-extends to the 64-bit product, as in C.
    let row =
        a.x.wrapping_add((bid() as usize).wrapping_mul(a.d as isize as usize));
    let p = strided_sum_sq(row, a.d);
    let rs = a.scale
        / sqrt(block_sum_lds(p, lds(), tid() as usize, ROW_BLOCK as usize) / a.d as f32 + a.eps);
    for_i32(tid(), a.d, ROW_BLOCK, |i| *row.add(i) *= rs);
}

/// One wave per `ACT_QUANT_BLOCK`-element group. `b >= nblk` is uniform per wave, so the
/// early return never splits a wave before `wave_max`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn act_quant_f8(a: ActQuant) {
    let lane = tid() & (WAVE - 1);
    let b = u64::from(bid()) * u64::from(ACT_QUANT_THREADS / WAVE) + u64::from(tid() / WAVE);
    if b >= a.nblk {
        return;
    }
    let seg = a.v.add(b as usize * ACT_QUANT_BLOCK as usize);
    let mut amax = 0.0f32;
    for_i32(lane, ACT_QUANT_BLOCK as i32, WAVE, |i| {
        amax = maxnum(amax, (*seg.add(i)).abs());
    });
    let s = fast_round_scale(maxnum(wave_max(amax), ACT_QUANT_AMAX_FLOOR), FP8_MAX_INV);
    for_i32(lane, ACT_QUANT_BLOCK as i32, WAVE, |i| {
        let e = seg.add(i);
        *e = act_quant_roundtrip(*e, s);
    });
}

/// Both RoPE conventions: thread `j` reads the pair at the indices `read(j, seg / 2)` names,
/// barriers (every pair is read before any is written — the in-place race-freedom), and
/// writes the rotation to `(j, seg/2 + j)`. Both products that multiply `cs` are the fused
/// ones — hipcc's ISA: `v_fma_f32 lo, a, cs, -(b·sn)` and `v_fmac_f32 hi += b·cs` over
/// `a·sn`. The block is `seg / 2` threads and nothing here reads blockDim.
#[inline(always)]
unsafe fn rope(a: Rope, read: impl Fn(i32, i32) -> (i32, i32)) {
    let j = tid() as i32;
    let half = a.seg / 2;
    // `(size_t)s * stride`, with C's sign extension of `stride`.
    let v = a
        .base
        .wrapping_add((bid() as usize).wrapping_mul(a.stride as isize as usize));
    let (ra, rb) = read(j, half);
    let (x, y) = (*v.add(ra as usize), *v.add(rb as usize));
    sync_threads();
    let inv = ocml::pow(a.theta, -2.0 * f64::from(j) / f64::from(a.seg));
    let ang = f64::from(a.pos) * inv;
    let (cs, sn) = (ocml::cos(ang) as f32, ocml::sin(ang) as f32);
    *v.add(j as usize) = fma(x, cs, -(y * sn));
    *v.add((half + j) as usize) = fma(y, cs, x * sn);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rope_interleave(a: Rope) {
    rope(a, |j, _| (2 * j, 2 * j + 1));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rope_split_half(a: Rope) {
    rope(a, |j, half| (j, half + j));
}
