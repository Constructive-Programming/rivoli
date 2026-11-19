//! MLA's absorb/value projections through the fp8 `kv_b`, and V4's attention-side helpers — the
//! twins of `kernels/mla.hip` at tag `archive/hip-kernels`. That file carries the contracts: why
//! the absorb is one thread per i-QUAD (one cache line per wave, measured 1.40×) and must not be
//! re-shaped into the value's wave-per-row form (that reassociates), the shift-not-divide tile
//! index, the fallback branch's two triggers, the `R`-row token batching, the prefix quantizer's
//! in-place contract, V4's bf16 QK-norm statistic, and the adjacent-pair RoPE.
//!
//! **Half this file is `contract(off)`, and that decides the bits.** The C's
//! `#pragma clang fp contract(off)` sits before `act_quant_f8_prefix` at FILE scope, so it
//! governs every kernel after it — `rmsnorm_batch`, `qk_norm`, `rope_adjacent` and
//! `gemv_fp8_bf16`'s own arithmetic carry no `contract` flags in hipcc's IR (`rmsnorm_batch`:
//! `%mul7 = fmul %8, %8; %add = fadd %p, %mul7`). So:
//! - the absorb's and value's dot accumulators are `v_fmac` (before the pragma), the absorb's
//!   `a += qd · (lut · s)` with `lut · s` hoisted and rounded first (the C's parenthesisation);
//! - `rmsnorm_batch`'s `Σ x²` and `rope_adjacent`'s rotation are plain products and sums — a
//!   first draft fused both, out of the habit of every other norm and RoPE in this tree; the ISA
//!   census caught it (hipcc `rope_adjacent`: 4 `v_mul`, 1 `v_add`, 1 `v_sub`, no `v_fma`);
//! - `gemv_fp8_bf16`'s dot is the prelude's (`dot_fp8.hpp` is included BEFORE the pragma, so it
//!   keeps its fused sites, and the census matches: 38 fma each).
//!
//! blockDim is never read: each block size is a `geom` constant its launcher launches with.

use crate::gpu::{bid, elementwise_id, fma, for_i32, fp8_lut, lds, tid};
use crate::prelude::actquant::{
    ACT_QUANT_AMAX_FLOOR, FP8_MAX_INV, act_quant_roundtrip, fast_round_scale,
};
use crate::prelude::dot_fp8::{FP8_SINGLE_ROW_UNROLL, Fp8Row, dot_fp8_wave_r};
use crate::prelude::formats::rbf16;
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::{ROWS_PER_BLOCK, WAVE, block_sum_lds, sync_threads};
use crate::prelude::rowview::{RowsView, blk_shift};
use core::intrinsics::sqrt;
use rivoli_abi::geom::{ROW_BLOCK, absorb_nquad};
use rivoli_abi::kernarg::{
    ActQuantPrefix, GemvFp8Bf16, MlaAbsorb, MlaValue, QkNorm, RmsNormRows, RopeAdjacent,
};

/// Scale-tile columns of an `n`-column fp8 weight, `ceil(n / block)` in the C's `int`.
#[inline(always)]
fn tiles(n: i32, block: i32) -> usize {
    ((n + block - 1) / block) as usize
}

/// One absorb thread's place: the LUT, its head, its quad's first column, the head's `kv_b` row
/// base `head · (nope + vh)`, and the two extents both paths index by.
struct Quad {
    lut: *const f32,
    head: usize,
    i0: i32,
    rbase: usize,
    /// `kvl` as an index, the tile shift `blk_shift(block)`, and the scale grid's columns —
    /// computed once per thread, not per `d`.
    kvl: usize,
    bsh: usize,
    sc_cols: usize,
}

/// The absorb's fallback (`kvl & 3` leaves rows unaligned for the dword load; `block < 4` splits
/// a quad across tiles): column by column, the C's `#pragma unroll 1` loop.
#[inline(always)]
unsafe fn absorb_columns<const R: usize>(
    a: &MlaAbsorb,
    q: &Quad,
    qrow: impl Fn(usize, usize) -> f32,
) {
    let (k, kvl, bsh) = (&a.kvb, q.kvl, q.bsh);
    let mut kk = 0;
    while kk < 4 && q.i0 + kk < k.kvl {
        let col = (q.i0 + kk) as usize;
        let mut acc = [0.0f32; R];
        for d in 0..k.nope as usize {
            let byte = *k.w.add((q.rbase + d) * kvl + col);
            let tile = ((q.rbase + d) >> bsh) * q.sc_cols + (col >> bsh);
            let ws = *q.lut.add(usize::from(byte)) * *k.scale.add(tile);
            for r in 0..R {
                acc[r] = fma(qrow(r, d), ws, acc[r]);
            }
        }
        let out = a.qabs.add(q.head * kvl + col);
        for r in 0..R {
            *out.add(r * k.h as usize * kvl) = acc[r];
        }
        kk += 1;
    }
}

/// `R` rows of four column accumulators to `out`, the rows `row_stride` apart. Its own function so
/// the quad path's loop nest stays shallow (CodeScene's bumpy-road finding, 2026-11-19).
#[inline(always)]
unsafe fn store_quads<const R: usize>(out: *mut f32, row_stride: usize, acc: [[f32; 4]; R]) {
    for r in 0..R {
        for c in 0..4 {
            *out.add(r * row_stride + c) = acc[r][c];
        }
    }
}

/// The absorb's fast path: one dword of `kv_b` (four columns, one scale tile) per `d`, the
/// products `lut · s` hoisted and rounded before they meet `q` — the C's parenthesisation. (Its
/// parameter order differs from `absorb_columns`' only because the identical signatures were a
/// jscpd clone.)
#[inline(always)]
unsafe fn absorb_quads<const R: usize>(
    q: &Quad,
    qrow: impl Fn(usize, usize) -> f32,
    a: &MlaAbsorb,
) {
    let (kvl, k) = (q.kvl, &a.kvb);
    let bsh = q.bsh;
    let sc_col = (q.i0 >> bsh) as usize;
    let mut w4 = k.w.add(q.rbase * kvl + q.i0 as usize).cast::<u32>();
    let mut acc = [[0.0f32; 4]; R];
    for d in 0..k.nope as usize {
        let p = *w4;
        let s = *k.scale.add(((q.rbase + d) >> bsh) * q.sc_cols + sc_col);
        let ls = |b: u32| *q.lut.add((b & 0xff) as usize) * s;
        let l = [ls(p), ls(p >> 8), ls(p >> 16), ls(p >> 24)];
        for r in 0..R {
            let qd = qrow(r, d);
            acc[r] = core::array::from_fn(|c| fma(qd, l[c], acc[r][c]));
        }
        w4 = w4.add(kvl >> 2);
    }
    store_quads(
        a.qabs.add(q.head * kvl + q.i0 as usize),
        k.h as usize * kvl,
        acc,
    );
}

/// One thread per (head, i-quad), `R` token rows through one read of `kv_b`.
#[inline(always)]
unsafe fn absorb<const R: usize>(a: MlaAbsorb) {
    let lut = fp8_lut();
    let k = &a.kvb;
    let nquad = absorb_nquad(k.kvl) as usize;
    // The C's `(size_t)blockIdx.x * blockDim.x + threadIdx.x` (its launcher's grid is elementwise).
    let idx = elementwise_id() as u32 as usize;
    if idx >= k.h as usize * nquad {
        return;
    }
    let head = idx / nquad;
    let q = Quad {
        lut,
        head,
        i0: ((idx % nquad) as i32) << 2,
        rbase: head * (k.nope + k.vh) as usize,
        kvl: k.kvl as usize,
        bsh: blk_shift(k.block) as usize,
        sc_cols: tiles(k.kvl, k.block),
    };
    let qh = a.q.add(head * a.qh as usize);
    let qrow = |r: usize, d: usize| *qh.add(r * k.h as usize * a.qh as usize + d);
    if (k.kvl & 3) != 0 || k.block < 4 {
        absorb_columns::<R>(&a, &q, qrow);
    } else {
        absorb_quads::<R>(&q, qrow, &a);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mla_absorb_fp8(a: MlaAbsorb) {
    absorb::<1>(a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mla_absorb_fp8_r2(a: MlaAbsorb) {
    absorb::<2>(a);
}

/// One WAVE per `kv_b` value row, `R` token rows (the prelude's R-row fp8 dot, un-unrolled as the
/// C's `dot_fp8_wave_r` is; R = 1 is bit-identical to the single-row form by construction).
#[inline(always)]
unsafe fn value<const R: usize>(a: MlaValue) {
    let lut = fp8_lut();
    let k = &a.kvb;
    let lane = tid() as usize & (WAVE - 1);
    let r = bid() as usize * ROWS_PER_BLOCK + tid() as usize / WAVE;
    let (h, vh, kvl) = (k.h as usize, k.vh as usize, k.kvl as usize);
    if r >= h * vh {
        return;
    }
    let (head, j) = (r / vh, r % vh);
    let row = head * (k.nope as usize + vh) + k.nope as usize + j;
    let w = Fp8Row {
        w: k.w.add(row * kvl),
        scale: k
            .scale
            .add((row >> blk_shift(k.block)) * tiles(k.kvl, k.block)),
        lut,
        i_dim: kvl,
        block: k.block,
    };
    let x = RowsView {
        x: a.clat.add(head * kvl),
        stride: h * kvl,
    };
    let acc = dot_fp8_wave_r::<R, 1>(x, w, lane);
    if lane == 0 {
        for rr in 0..R {
            *a.ctx.add(rr * h * vh + head * vh + j) = acc[rr];
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mla_value_fp8(a: MlaValue) {
    value::<1>(a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mla_value_fp8_r2(a: MlaValue) {
    value::<2>(a);
}

/// One thread per `block`-wide group of a row's first `n`; `src` may be `dst`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn act_quant_f8_prefix(a: ActQuantPrefix) {
    let nb = (a.n / a.block) as usize;
    // `as u32 as usize`: the unsigned 32-bit index widened, which equals the C's `size_t` product
    // for every grid under 2^32 threads (the launcher's `dim3` is 32-bit too).
    let id = elementwise_id() as u32 as usize;
    if id >= a.rows as usize * nb {
        return;
    }
    let off = (id / nb) * a.row_stride as usize + (id % nb) * a.block as usize;
    let (seg, out) = (a.src.add(off), a.dst.add(off));
    let mut amax = ACT_QUANT_AMAX_FLOOR;
    for i in 0..a.block as usize {
        amax = maxnum(amax, (*seg.add(i)).abs());
    }
    let s = fast_round_scale(amax, FP8_MAX_INV);
    for i in 0..a.block as usize {
        *out.add(i) = act_quant_roundtrip(*seg.add(i), s);
    }
}

/// `row`'s block-wide `Σ term(x)` through the `red` tree (`block_sum_lds`).
#[inline(always)]
unsafe fn row_sum(row: *const f32, d: i32, term: impl Fn(f32, f32) -> f32) -> f32 {
    let mut p = 0.0f32;
    for_i32(tid(), d, ROW_BLOCK, |i| p = term(*row.add(i), p));
    block_sum_lds(p, lds(), tid() as usize, ROW_BLOCK as usize)
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_batch(a: RmsNormRows) {
    let row = a.x.add(bid() as usize * a.d as usize);
    let sum = row_sum(row, a.d, |v, p| p + v * v);
    let rs = 1.0 / sqrt(sum / a.d as f32 + a.eps);
    for_i32(tid(), a.d, ROW_BLOCK, |i| {
        *row.add(i) = rbf16(*a.w.add(i) * (*row.add(i) * rs))
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn qk_norm(a: QkNorm) {
    let row = a.q.add(bid() as usize * a.d as usize);
    let var = rbf16(row_sum(row, a.d, |v, p| p + rbf16(v * v)) / a.d as f32);
    let rs = rbf16(1.0 / sqrt(rbf16(var + a.eps)));
    for_i32(tid(), a.d, ROW_BLOCK, |i| {
        *row.add(i) = rbf16(*row.add(i) * rs)
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rope_adjacent(a: RopeAdjacent) {
    let b = bid() as i32;
    let row = a.x.add(b as usize * a.row_len as usize);
    let seg = row.add((a.row_len - a.rd) as usize);
    let t = a
        .tbl
        .add((a.pos0 + b / a.rows_per_pos) as usize * a.rd as usize);
    for_i32(tid(), a.rd / 2, ROW_BLOCK, |i| {
        let c = *t.add(2 * i);
        let s = if a.inverse != 0 {
            -*t.add(2 * i + 1)
        } else {
            *t.add(2 * i + 1)
        };
        let (x, y) = (*seg.add(2 * i), *seg.add(2 * i + 1));
        *seg.add(2 * i) = x * c - y * s;
        *seg.add(2 * i + 1) = y * c + x * s;
    });
    sync_threads();
    for_i32(tid(), a.row_len, ROW_BLOCK, |i| {
        *row.add(i) = rbf16(*row.add(i))
    });
}

/// One WAVE per output; `x`'s row for output `j` is its group's (`groups` query heads sharing a
/// weight slice); the single-row fp8 dot at its measured unroll; the store bf16-rounded.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_fp8_bf16(a: GemvFp8Bf16) {
    let lut = fp8_lut();
    let id = bid() as usize * ROWS_PER_BLOCK + tid() as usize / WAVE;
    let n_out = a.n_out as usize;
    if id >= a.m as usize * n_out {
        return;
    }
    let (r, j) = (id / n_out, id % n_out);
    let k = a.k as usize;
    // The C's `int` division, `(size_t)(j / (n_out / groups))`.
    let group = (j as i32 / (a.n_out / a.groups)) as usize;
    let x = RowsView {
        x: a.x.add((r * a.groups as usize + group) * k),
        stride: 0,
    };
    let w = Fp8Row {
        w: a.w.add(j * k),
        scale: a
            .wscale
            .add((j >> blk_shift(a.block)) * tiles(a.k, a.block)),
        lut,
        i_dim: k,
        block: a.block,
    };
    let lane = tid() as usize & (WAVE - 1);
    let [acc] = dot_fp8_wave_r::<1, FP8_SINGLE_ROW_UNROLL>(x, w, lane);
    if lane == 0 {
        *a.out.add(r * n_out + j) = rbf16(acc);
    }
}
