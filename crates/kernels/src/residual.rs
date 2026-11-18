//! The residual-stream mixers — the twins of `kernels/residual.hip` at tag
//! `archive/hip-kernels`: V4's mHC pair, Kimi-K3's `attn_res`, and Qwen's gated-residual
//! collapse. That file carries each kernel's contract and measurements; this one carries
//! what the port decided. Every `a * b + c` below is a [`fma`] because hipcc's backend fused
//! it (the C carries `contract` flags, not `fmuladd` — the fusion is in the ISA).

use crate::gpu::{bid, fma, for_i32, lds, tid};
use crate::prelude::formats::rbf16;
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::{sync_threads, tree, wave_sum};
use crate::prelude::rowview::for_strided;
use core::intrinsics::{exp, sqrt, unchecked_div, unchecked_rem};
use rivoli_abi::geom::{
    ATTN_RES_BLOCK, ATTN_RES_MAX_SRC, ELEMENTWISE_BLOCK, HC_MIX, HC_MULT, HC_PRE_BLOCK, HC_RED,
    WAVE,
};
use rivoli_abi::kernarg::{AttnRes, GatedResidual, HcPost, HcPre};

const HM: usize = HC_MULT as usize;

/// `hc_pre`'s two streaming loops: the C's `#pragma unroll 8` — schedule only, the fold
/// stays ascending. Without it the port ran 3.9x hipcc; with it, 1.002x (hipcc 14.58 µs, Rust
/// 14.61 µs; rust-kernels.md §5, Phase 0a leases 1 and 2 — measured before the `for_i32`
/// loops replaced the hand-written ones, so the next device run re-measures it).
const HC_UNROLL: usize = 8;

// `each_src` spells this many slots.
const _: () = assert!(ATTN_RES_MAX_SRC == 16, "each_src spells 16 slots");

/// `hc_pre`'s five LDS arrays, carved from the one dynamic allocation in
/// the order `rivoli_abi::geom::HC_PRE_LDS_BYTES` documents and sums.
struct HcLds {
    red: *mut f32,
    mixes: *mut f32,
    pre: *mut f32,
    post: *mut f32,
    comb: *mut f32,
}

impl HcLds {
    #[inline(always)]
    unsafe fn carve() -> Self {
        let red = lds::<f32>();
        let mixes = red.add(HC_RED as usize);
        let pre = mixes.add(HC_MIX as usize);
        let post = pre.add(HM);
        let comb = post.add(HM);
        Self {
            red,
            mixes,
            pre,
            post,
            comb,
        }
    }
}

/// `1 / (1 + e^-x)` — the division form every sigmoid in this file uses.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// `comb /= (comb.sum(axis) + eps)` along one axis, given as a (hold, run) stride pair.
#[inline(always)]
unsafe fn sinkhorn_norm(comb: *mut f32, hold: usize, run: usize, eps: f32) {
    for fixed in 0..HM {
        let c = comb.add(fixed * hold);
        let mut s = 0.0f32;
        for r in 0..HM {
            s += *c.add(r * run);
        }
        for r in 0..HM {
            *c.add(r * run) /= s + eps;
        }
    }
}

/// Row-wise `softmax + eps` over the `HC_MULT x HC_MULT` comb, max/exp/sum in the
/// reference's order.
#[inline(always)]
unsafe fn comb_softmax(comb: *mut f32, eps: f32) {
    for j in 0..HM {
        let row = comb.add(j * HM);
        let mut m = *row;
        for t in 1..HM {
            m = maxnum(m, *row.add(t));
        }
        let mut s = 0.0f32;
        for t in 0..HM {
            *row.add(t) = exp(*row.add(t) - m);
            s += *row.add(t);
        }
        for t in 0..HM {
            *row.add(t) = *row.add(t) / s + eps;
        }
    }
}

/// `kernel.py::hc_split_sinkhorn` for one token, on one thread (`residual.hip` argues why
/// serial, and why the first pass is a softmax). `scale`/`base` are global; the rest LDS.
#[inline(always)]
unsafe fn hc_sinkhorn(l: &HcLds, a: &HcPre) {
    let eps = a.hc_eps;
    let mix = |j: usize, k: usize| fma(*l.mixes.add(j), *a.scale.add(k), *a.base.add(j));
    for j in 0..HM {
        *l.pre.add(j) = sigmoid(mix(j, 0)) + eps;
        *l.post.add(j) = 2.0 * sigmoid(mix(j + HM, 1));
    }
    for j in 0..HM * HM {
        *l.comb.add(j) = mix(j + 2 * HM, 2);
    }
    comb_softmax(l.comb, eps);
    sinkhorn_norm(l.comb, 1, HM, eps);
    for _ in 1..a.iters {
        sinkhorn_norm(l.comb, HM, 1, eps);
        sinkhorn_norm(l.comb, 1, HM, eps);
    }
}

/// One token's flattened `[HC_MULT, dim]` stream: its base and its length `HC_MULT·dim`.
#[derive(Clone, Copy)]
struct HcRow {
    flat: *const f32,
    hcd: usize,
}

/// The token's `1/rms` over the FULL `HC_MULT·dim` row, at the frozen `HC_RED` width.
#[inline(always)]
unsafe fn hc_rms(HcRow { flat, hcd }: HcRow, norm_eps: f32, red: *mut f32) -> f32 {
    let t = tid();
    if t < HC_RED {
        let mut local = 0.0f32;
        for_strided::<HC_UNROLL>(t as usize, hcd, HC_RED as usize, |i| {
            local = fma(*flat.add(i), *flat.add(i), local);
        });
        *red.add(t as usize) = local;
    }
    sync_threads();
    tree(t as usize, HC_RED as usize, |t, s| {
        *red.add(t) += *red.add(s)
    });
    1.0 / sqrt(*red / hcd as f32 + norm_eps)
}

/// One wave per mix row, lane-strided then `wave_sum`.
#[inline(always)]
unsafe fn hc_mixes(HcRow { flat, hcd }: HcRow, fnw: *const f32, rs: f32, mixes: *mut f32) {
    let t = tid();
    let lane = t & (WAVE - 1);
    let mut j = t / WAVE;
    while j < HC_MIX {
        let wr = fnw.add(j as usize * hcd);
        let mut acc = 0.0f32;
        for_strided::<HC_UNROLL>(lane as usize, hcd, WAVE as usize, |i| {
            acc = fma(*flat.add(i), *wr.add(i), acc);
        });
        let acc = wave_sum(acc);
        if lane == 0 {
            *mixes.add(j as usize) = acc * rs;
        }
        j += HC_PRE_BLOCK / WAVE;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn hc_pre(a: HcPre) {
    let l = HcLds::carve();
    let t = tid() as usize;
    let dim = a.dim as isize as usize;
    let hcd = HM.wrapping_mul(dim);
    let tok = bid() as usize;
    let flat = a.h.add(tok * hcd);
    let row = HcRow { flat, hcd };
    let rs = hc_rms(row, a.norm_eps, l.red);
    hc_mixes(row, a.fnw, rs, l.mixes);
    sync_threads();
    if t == 0 {
        hc_sinkhorn(&l, &a);
    }
    sync_threads();
    let yo = a.y.add(tok * dim);
    for_i32(t as u32, a.dim, HC_PRE_BLOCK, |d| {
        let mut acc = 0.0f32;
        for c in 0..HM {
            acc = fma(*l.pre.add(c), *flat.add(c * dim + d), acc);
        }
        *yo.add(d) = rbf16(acc);
    });
    if t < HM {
        *a.post.add(tok * HM + t) = *l.post.add(t);
    }
    if t < HM * HM {
        *a.comb.add(tok * HM * HM + t) = *l.comb.add(t);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn hc_post(a: HcPost) {
    let i = bid() as usize * ELEMENTWISE_BLOCK as usize + tid() as usize;
    let dim = a.dim as isize as usize;
    if i >= (a.s as isize as usize).wrapping_mul(HM).wrapping_mul(dim) {
        return;
    }
    // SAFETY: `dim > 0` — the launcher refuses `dim <= 0` (1001), so no division by zero.
    let (q, d) = (unchecked_div(i, dim), unchecked_rem(i, dim));
    let (k, tok) = (q % HM, q / HM);
    // `post·x + comb₀·res₀` is `a·b + c·d`: hipcc fuses the LEFT product, `fma(post, x,
    // comb₀·res₀)` (its ISA: one `v_mul`, then four `v_fmac`). Fusing the right one instead
    // (this kernel's first spelling) passed every CPU check and went RED on the device:
    // 26 words in 23 of 192 cells, bit-identity arm A, 2026-11-18.
    let src = |j: usize| tok * HM + j;
    let cell = |j: usize| {
        (
            *a.comb.add(src(j) * HM + k),
            *a.residual.add(src(j) * dim + d),
        )
    };
    let (c0, r0) = cell(0);
    let mut acc = fma(*a.post.add(tok * HM + k), *a.x.add(tok * dim + d), c0 * r0);
    for j in 1..HM {
        let (c, r) = cell(j);
        acc = fma(c, r, acc);
    }
    *a.y.add(i) = rbf16(acc);
}

/// `attn_res`'s three LDS arrays, the order `rivoli_abi::geom::ATTN_RES_LDS_BYTES` documents and sums:
/// `(red_ss, red_dot, prob)`.
#[inline(always)]
unsafe fn attn_res_lds() -> [*mut f32; 3] {
    let red_ss = lds::<f32>();
    let red_dot = red_ss.add(ATTN_RES_BLOCK as usize);
    [red_ss, red_dot, red_dot.add(ATTN_RES_BLOCK as usize)]
}

/// Score source `s`: `<v, fold> / rms(v)`, both sums in one pass and one pair of trees;
/// thread 0 writes `prob[s]`. Ends on a barrier, so the next source may reuse the trees.
#[inline(always)]
unsafe fn attn_res_score(
    a: &AttnRes,
    v: *const f32,
    s: usize,
    [red_ss, red_dot, prob]: [*mut f32; 3],
) {
    let t = tid();
    let (mut ss, mut dot) = (0.0f32, 0.0f32);
    for_i32(t, a.n, ATTN_RES_BLOCK, |i| {
        let x = *v.add(i);
        ss = fma(x, x, ss);
        dot = fma(x, *a.fold.add(i), dot);
    });
    *red_ss.add(t as usize) = ss;
    *red_dot.add(t as usize) = dot;
    sync_threads();
    tree(t as usize, ATTN_RES_BLOCK as usize, |t, k| {
        *red_ss.add(t) += *red_ss.add(k);
        *red_dot.add(t) += *red_dot.add(k);
    });
    if t == 0 {
        *prob.add(s) = *red_dot / sqrt(*red_ss / a.n as f32 + a.eps);
    }
    sync_threads();
}

/// `f(s)` for `s = 0, 1, …, ns - 1`, as 16 straight-line guarded slots, so every index into
/// the private `p[16]` is a constant and `p` stays in registers, as hipcc keeps it. A loop
/// bounded by the runtime `nsrc` put `p` in scratch (`.private_segment_fixed_size` 144 per
/// `llvm-readelf --notes` on the code object; 80 with a constant-bound loop LLVM did not
/// unroll; 0 now) and `attn_res` ran 1.38x hipcc; with these slots it runs 1.063x (hipcc
/// 53.14 µs, Rust 56.51 µs; rust-kernels.md §5, Phase 0a leases 1 and 2). The remaining 6% is
/// OWED to Phase 1, and these slots are the suspect: the ISA read 1617 instructions to hipcc's
/// 725 and 10 static `s_barrier` to 3, because hipcc keeps a loop over sources that the slots
/// expand. (All measured before `for_i32`; the next device run re-measures them.)
/// Same operations as the C, in its ascending order.
#[inline(always)]
fn each_src(ns: usize, mut f: impl FnMut(usize)) {
    macro_rules! slots {
        ($($s:literal)*) => { $( if $s < ns { f($s); } )* };
    }
    slots!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
}

/// Max-subtracted softmax over the `nsrc` scores, redundantly per thread (the C argues why).
#[inline(always)]
unsafe fn attn_res_probs(prob: *const f32, ns: usize) -> [f32; ATTN_RES_MAX_SRC as usize] {
    let mut m = *prob;
    for s in 1..ns {
        m = maxnum(m, *prob.add(s));
    }
    let mut z = 0.0f32;
    for s in 0..ns {
        z += exp(*prob.add(s) - m);
    }
    let mut p = [0.0f32; ATTN_RES_MAX_SRC as usize];
    each_src(ns, |s| p[s] = exp(*prob.add(s) - m) / z);
    p
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_res(a: AttnRes) {
    let lds3 = attn_res_lds();
    let n = a.n as isize as usize;
    let ns = a.nsrc as usize;
    let v0 = a.src.add(bid() as usize * a.src_stride as usize);
    for s in 0..ns {
        attn_res_score(&a, v0.add(s * n), s, lds3);
    }
    let p = attn_res_probs(lds3[2], ns);
    let o = a.out.add(bid() as usize * n);
    for_i32(tid(), a.n, ATTN_RES_BLOCK, |i| {
        let mut acc = 0.0f32;
        each_src(ns, |s| acc = fma(p[s], *v0.add(s * n + i), acc));
        *o.add(i) = acc;
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gated_residual_collapse_f32(a: GatedResidual) {
    let t = tid() as i32;
    if bid() == 0 && t < a.n_r {
        *a.scale.add(t as usize) = 2.0 / (1.0 + exp(-*a.inject.add(t as usize) / a.n_r as f32));
    }
    let j = bid().wrapping_mul(ELEMENTWISE_BLOCK) as i32 + t;
    if j >= a.dim {
        return;
    }
    let mut acc = 0.0f32;
    for i in 0..a.n_r {
        let at = (i as usize) * (a.dim as usize) + j as usize;
        // The gate wraps `mix`, never `rhat` (the C's T-trap note).
        acc += *a.rhat.add(at) / (1.0 + exp(-*a.mix.add(at)));
    }
    *a.x.add(j as usize) = acc / a.n_r as f32;
}
