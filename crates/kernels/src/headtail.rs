//! V4-Flash's head tail — the twins of `kernels/headtail.hip` at tag `archive/hip-kernels`
//! (which carries why the final norm and lm_head are NOT here, and why the gate is one wave
//! per token).
//!
//! **Fused sites, read from hipcc's gfx1151 ISA:** both strided dots of the gate are
//! `fma(x, w, acc)` (`v_fmac`); the mix is `fma(m, scale[0], base[j])` (`v_fma_f32 v6, v6,
//! s12, s20`); the blend's `acc += pre·x` is `fma(pre, x, acc)`. The two divides and the
//! `sqrt` are the correctly rounded ones hipcc expands (`v_div_scale`…`v_div_fixup`), which
//! `/` and `core::intrinsics::sqrt` lower to identically.

use crate::gpu::{bid, elementwise, fma, tid};
use crate::prelude::formats::{bf16f, rbf16};
use crate::prelude::llvm::wave_shuffle;
use crate::prelude::reduce::{WAVE, wave_sum};
use crate::prelude::rowview::for_strided;
use core::intrinsics::{exp, sqrt, unchecked_div, unchecked_rem};
use rivoli_abi::geom::ELEMENTWISE_BLOCK;
use rivoli_abi::kernarg::{EmbedBf16, HcHeadBlend, HcHeadGate};

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn embed_bf16_row_bcast(a: EmbedBf16) {
    let hidden = a.hidden as usize;
    elementwise(a.hidden, |i| {
        // `token >= 0` — the launcher refuses a negative one (1001).
        let v = bf16f(*a.w.add(a.token as usize * hidden + i));
        for c in 0..a.hc as usize {
            *a.x.add(c * hidden + i) = v;
        }
    });
}

/// `Σ a[i]·b[i]` over this lane's `WAVE`-strided columns of `0..n` — the C's `size_t` loop,
/// fused per term.
#[inline(always)]
unsafe fn lane_dot(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = 0.0f32;
    for_strided::<1>(tid() as usize, n, WAVE, |i| {
        acc = fma(*a.add(i), *b.add(i), acc)
    });
    acc
}

/// One wave per token. `wave_sum` leaves its total on lane 0 only, so `rs` is broadcast from
/// there (`__shfl(rs, 0)`) before every lane uses it.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn hc_head_collapse_gate(a: HcHeadGate) {
    let t = bid() as i32;
    if t >= a.s {
        return;
    }
    let lane = tid();
    let hc = a.hc as usize;
    let hcd = hc * a.dim as usize;
    let flat = a.h.add(t as usize * hcd);
    let rs = 1.0 / sqrt(wave_sum(lane_dot(flat, flat, hcd)) / hcd as f32 + a.eps);
    let rs = wave_shuffle(rs, 0);
    for j in 0..hc {
        let m = wave_sum(lane_dot(flat, a.fnw.add(j * hcd), hcd)) * rs;
        if lane == 0 {
            let z = fma(m, *a.scale, *a.base.add(j));
            *a.pre.add(t as usize * hc + j) = 1.0 / (1.0 + exp(-z)) + a.hc_eps;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn hc_head_collapse_blend(a: HcHeadBlend) {
    let id = bid() as usize * ELEMENTWISE_BLOCK as usize + tid() as usize;
    let (hc, dim) = (a.hc as usize, a.dim as usize);
    if id >= a.s as usize * dim {
        return;
    }
    // SAFETY: `dim > 0` — the launcher refuses `dim <= 0` (1001).
    let (t, d) = (unchecked_div(id, dim), unchecked_rem(id, dim));
    let flat = a.h.add(t * hc * dim);
    let mut acc = 0.0f32;
    for c in 0..hc {
        acc = fma(*a.pre.add(t * hc + c), *flat.add(c * dim + d), acc);
    }
    *a.y.add(id) = rbf16(acc);
}
