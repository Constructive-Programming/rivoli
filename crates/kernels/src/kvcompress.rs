//! V4-Flash's KV compressor (`Compressor.forward`, model.py:322-386) and the plain bf16 GEMM its
//! projections run through — the twins of `kernels/kvcompress.hip` at tag
//! `archive/hip-kernels`.
//! That file carries the contracts: the overlap index map, why the softmax recomputes its
//! exponential rather than hoisting the divide, the three bf16 stores of the finish, the deposit's
//! two-phase unification, and the decode window slide. What this file adds:
//!
//! - **The fused sites.** These were read off hipcc's ISA:
//!   - `acc += kv · (e / sum)` is `fma(kv, e/sum, acc)`;
//!   - the norm's `Σ x²` and the GEMM's dot are `v_fmac`;
//!   - the adjacent-pair RoPE fuses each LEFT product: `fma(a, c, -(b·s))` and `fma(a, s, b·c)`
//!     (`v_mul c·b; v_fmac += s·a` — `v[12:13]` is the `seg` pair the store writes back, so read
//!     the register roles off the loads and stores: a misread here once "fixed" it backwards).
//! - **The masked-`ape` add is kept.** The decode pool passes no position embedding, and the C
//!   still computes `score + 0.0f`, which turns a `-0.0` into `+0.0`. The fold `x + 0.0 → x` is
//!   illegal without `nsz`, so `ape` returning `0.0` reproduces the same add.
//! - **No blockDim, no gridDim.** Every block is a `geom` constant. The deposit's grid stride
//!   arrives as `Strided::threads`, which the launcher sets to the threads it launched.

use crate::gpu::{bid, fma, for_i32, for_usize, lds, tid};
use crate::prelude::formats::{bf16f, rbf16};
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::{block_sum_lds, sync_threads, wave_sum};
use core::intrinsics::{exp, sqrt, unchecked_div, unchecked_rem};
use rivoli_abi::geom::{COMPRESS_BLOCK, COMPRESS_ENTS_MAX, GEMM_BF16_BLOCK, WAVE};
use rivoli_abi::kernarg::{CompressDecode, CompressPrefill, Deposit, GemmBf16, Strided};
use rivoli_abi::{CompFinish, CompGeom};

/// `out[m, n] = x[m, k] · w[n, k]ᵀ` with `w` in bf16 — the compressor's (and the trunk's)
/// un-quantized `F.linear`: NO `act_quant` on the activation and no weight scales, which is why it
/// is not `gemv_fp8_bf16`; bf16 on device because the checkpoint stores it so. One wave per output
/// element; `wave_sum`'s fixed ladder re-associates against the oracle's sequential dot.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemm_bf16(a: GemmBf16) {
    let gw = (bid().wrapping_mul(GEMM_BF16_BLOCK).wrapping_add(tid()) / WAVE) as i32;
    let lane = tid() % WAVE;
    // `size_t`, as the C: in `int` the product overflows at ~2.1M outputs, every wave returns,
    // nothing is written and the launch still reports success (`tests/headtail.rs` cites this).
    if gw as usize >= a.m as usize * a.n as usize {
        return;
    }
    // SAFETY: `n > 0` (1001).
    let (r, c) = (unchecked_div(gw, a.n), unchecked_rem(gw, a.n));
    let xr = a.x.add(r as usize * a.k as usize);
    let wr = a.w.add(c as usize * a.k as usize);
    let mut acc = 0.0f32;
    for_i32(lane, a.k, WAVE, |i| {
        acc = fma(*xr.add(i), bf16f(*wr.add(i)), acc)
    });
    acc = wave_sum(acc);
    if lane == 0 {
        *a.out.add(gw as usize) = acc;
    }
}

/// The pooling window's LDS: per-entry pointer tables (`kvp`, `scp`, `apep`; a null `scp[j]` is a
/// masked entry) and the norm's tree, carved in `geom::COMPRESS_LDS_BYTES`' order.
struct Window {
    kvp: *mut *const f32,
    scp: *mut *const f32,
    apep: *mut *const f32,
    red: *mut f32,
}

impl Window {
    #[inline(always)]
    unsafe fn carve() -> Self {
        let kvp = lds::<*const f32>();
        let scp = kvp.add(COMPRESS_ENTS_MAX as usize);
        let apep = scp.add(COMPRESS_ENTS_MAX as usize);
        let red = apep.add(COMPRESS_ENTS_MAX as usize).cast::<f32>();
        Self {
            kvp,
            scp,
            apep,
            red,
        }
    }

    /// Point entry `j` at `kv + off`, `score + off` (decode's state slots, or prefill's rows).
    #[inline(always)]
    unsafe fn set(&self, j: usize, (kv, score): (*const f32, *const f32), off: usize) {
        *self.kvp.add(j) = kv.add(off);
        *self.scp.add(j) = score.add(off);
    }
}

/// `f(j)` for each live window entry, ascending — the C's `for (int j = 0; j < ents; ++j) { if
/// (!scp[j]) continue; … }`. As `(0..ents).filter(live)` it compiled to a nested find-next scan on
/// 64-bit counters, where hipcc unrolls the plain loop 8× with the loads in flight (Phase 1
/// integration, 2026-11-19; the L3 lease timed `kv_compress_decode` at 1.22× hipcc).
#[inline(always)]
unsafe fn each_live(w: &Window, ents: i32, mut f: impl FnMut(usize)) {
    for_i32(0, ents, 1, |j| {
        if !(*w.scp.add(j)).is_null() {
            f(j)
        }
    });
}

/// model.py:346 on one output row: per feature, the softmax over the window's live entries, then
/// the weighted sum, sequential in the entry order. `ape(j, e)` is the position embedding added to
/// each score (`0.0` at decode, where the deposit already added it — see the module note).
#[inline(always)]
unsafe fn pool_row(w: &Window, ape: impl Fn(usize, usize) -> f32, p: &CompGeom, out: *mut f32) {
    for_i32(tid(), p.d, COMPRESS_BLOCK, |e| {
        let score = |j: usize| *(*w.scp.add(j)).add(e) + ape(j, e);
        let mut mx = f32::NEG_INFINITY;
        each_live(w, p.ents, |j| mx = maxnum(mx, score(j)));
        // Every entry masked: the oracle's zero row (unreachable through either launcher).
        if !mx.is_finite() {
            *out.add(e) = 0.0;
            return;
        }
        let mut sum = 0.0f32;
        each_live(w, p.ents, |j| sum += exp(score(j) - mx));
        let mut acc = 0.0f32;
        each_live(w, p.ents, |j| acc = fma(*(*w.kvp.add(j)).add(e), exp(score(j) - mx) / sum, acc));
        *out.add(e) = acc;
    });
}

/// `row[e] = rbf16(row[e])` over the block's strided share.
#[inline(always)]
unsafe fn round_row(row: *mut f32, d: i32) {
    for_i32(tid(), d, COMPRESS_BLOCK, |e| {
        *row.add(e) = rbf16(*row.add(e))
    });
}

/// model.py:368-373 on one pooled row: the bf16 store, the compressor's RMSNorm, the adjacent-pair
/// RoPE on the last `rd` dims at absolute position `pos`, the final bf16 store.
#[inline(always)]
unsafe fn finish_row(row: *mut f32, (f, p): (&CompFinish, &CompGeom), pos: i32, red: *mut f32) {
    round_row(row, p.d);
    sync_threads();
    let mut part = 0.0f32;
    for_i32(tid(), p.d, COMPRESS_BLOCK, |e| {
        let v = *row.add(e);
        part = fma(v, v, part);
    });
    let sum = block_sum_lds(part, red, tid() as usize, COMPRESS_BLOCK as usize);
    let rs = 1.0 / sqrt(sum / p.d as f32 + p.eps);
    for_i32(tid(), p.d, COMPRESS_BLOCK, |e| {
        *row.add(e) = rbf16(*f.norm.add(e) * (*row.add(e) * rs))
    });
    sync_threads();
    let t = f.freqs.add(pos as usize * p.rd as usize);
    let seg = row.add((p.d - p.rd) as usize);
    for_i32(tid(), p.rd / 2, COMPRESS_BLOCK, |i| {
        let (c, s) = (*t.add(2 * i), *t.add(2 * i + 1));
        let (x, y) = (*seg.add(2 * i), *seg.add(2 * i + 1));
        *seg.add(2 * i) = fma(x, c, -(y * s));
        *seg.add(2 * i + 1) = fma(x, s, y * c);
    });
    sync_threads();
    round_row(row, p.d);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kv_compress_deposit(g: Strided<Deposit>) {
    let (a, p) = (g.a, g.a.p);
    // SAFETY: `ratio > 1` (1001); and for the `cd` divisions below, `d > 0`, `coff ∈ {1, 2}`
    // (1001) and `cd == coff · d` (1003, wrapping), so `cd as usize` is nonzero.
    let remainder = unchecked_rem(a.s, p.ratio);
    let cutoff = a.s - remainder;
    let overlap = p.coff == 2;
    let state_off = if overlap { p.ratio } else { 0 };
    let stride = g.threads as usize;
    let t = bid() as usize * COMPRESS_BLOCK as usize + tid() as usize;
    let cd = p.cd as usize;
    if overlap && cutoff >= p.ratio {
        let base = (cutoff - p.ratio) as usize;
        for_usize(t, p.ratio as usize * cd, stride, |i| {
            let src = (base + unchecked_div(i, cd)) * cd + unchecked_rem(i, cd);
            *a.kv_state.add(i) = *a.kv.add(src);
            *a.score_state.add(i) = *a.score.add(src) + *a.ape.add(i);
        });
    }
    if remainder > 0 {
        let dst0 = (state_off + a.slot0) as usize * cd;
        let ape0 = a.slot0 as usize * cd;
        for_usize(t, remainder as usize * cd, stride, |i| {
            let src = (cutoff as usize + unchecked_div(i, cd)) * cd + unchecked_rem(i, cd);
            *a.kv_state.add(dst0 + i) = *a.kv.add(src);
            *a.score_state.add(dst0 + i) = *a.score.add(src) + *a.ape.add(ape0 + i);
        });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kv_compress_prefill(a: CompressPrefill) {
    let w = Window::carve();
    let p = a.p;
    let b = bid() as i32;
    let overlap = p.coff == 2;
    for_i32(tid(), p.ents, COMPRESS_BLOCK, |j| {
        let ji = j as i32;
        // `overlap_transform` as an index map (the C's note): (source block, entry, half).
        let (sblk, src_j, half) = if !overlap {
            (b, ji, 0)
        } else if ji >= p.ratio {
            (b, ji - p.ratio, p.d)
        } else if b > 0 {
            (b - 1, ji, 0)
        } else {
            *w.kvp.add(j) = core::ptr::null();
            *w.scp.add(j) = core::ptr::null();
            *w.apep.add(j) = core::ptr::null();
            return;
        };
        let src = (sblk * p.ratio + src_j) as usize * p.cd as usize + half as usize;
        w.set(j, (a.kv, a.score), src);
        *w.apep.add(j) = a.ape.add(src_j as usize * p.cd as usize + half as usize);
    });
    sync_threads();
    let row = a.f.out.add(b as usize * p.d as usize);
    pool_row(&w, |j, e| *(*w.apep.add(j)).add(e), &p, row);
    sync_threads();
    finish_row(row, (&a.f, &p), b * p.ratio, w.red);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kv_compress_decode(a: CompressDecode) {
    let w = Window::carve();
    let p = a.p;
    let overlap = p.coff == 2;
    let cd = p.cd as usize;
    for_i32(tid(), p.ents, COMPRESS_BLOCK, |j| {
        let half = if overlap && j as i32 >= p.ratio {
            p.d
        } else {
            0
        };
        w.set(j, (a.kv_state, a.score_state), j * cd + half as usize);
    });
    sync_threads();
    pool_row(&w, |_, _| 0.0, &p, a.f.out);
    sync_threads();
    finish_row(
        a.f.out,
        (&a.f, &p),
        // SAFETY: `ratio > 1` (1001).
        unchecked_div(a.start_pos, p.ratio) * p.ratio,
        w.red,
    );
    sync_threads();
    if overlap {
        let n = p.ratio as usize * cd;
        // Both loads before either store: the C's `__restrict__` lets hipcc hoist the score load
        // over the kv store, and raw pointers do not, which left one more serial global round
        // trip per element (rust-kernels.md §5 "Perf follow-up"). The two buffers are distinct
        // and `[n, 2n)` never meets `[0, n)`, so the values are the same.
        for_usize(tid() as usize, n, COMPRESS_BLOCK as usize, |i| {
            let (kv, score) = (*a.kv_state.add(n + i), *a.score_state.add(n + i));
            *a.kv_state.add(i) = kv;
            *a.score_state.add(i) = score;
        });
    }
}
