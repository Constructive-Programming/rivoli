//! The batch-1 GEMVs and the VQ encoder — the twins of `kernels/linalg.hip` at tag
//! `archive/hip-kernels`, which carries each kernel's contract and measurements. What this
//! file adds is what the port decided.
//!
//! **Fused sites, read from hipcc's gfx1151 ISA:** the fp8/VQ/int4 dots are the prelude's
//! (`rowview::quad`, then `fma(s, quad, acc)`); `gemv_f32`'s `acc += x·w` is `fma(x, w, acc)`
//! (`v_fmac`); `gemv_i8`'s dword step is `acc + quad(x, w)` — the quad fused (`v_mul` + three
//! `v_fmac`), the accumulate a plain `v_add`, because no `s·` wraps it — and its tail is
//! `fma(x, w, acc)`. `vq_encode` is `#pragma clang fp contract(off)`: nothing fuses.
//!
//! `R` (token rows through one read of the weights) is a const generic, so each `_r2` kernel
//! is its own instantiation exactly as each C template instance was.

use crate::gpu::{bid, elementwise, fma, for_i32, fp8_lut, tid};
use crate::prelude::dot_fp8::{Fp8Row, dot_fp8_wave_r, fp8_dot_strided_r};
use crate::prelude::dot_nib::{I4_GROUP, I4Row, NibRow, dot_nib_wave_r};
use crate::prelude::dot_vq::{VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VQ_K, VqRow, dot_vq_wave_r};
use crate::prelude::formats::bf16f;
use crate::prelude::reduce::{ROWS_PER_BLOCK, WAVE, sync_threads, wave_sum};
use crate::prelude::rowview::{F32x4, LaneSpan, RowsView, U32x4, blk_shift, for_strided, quad};
use core::intrinsics::unchecked_div;
use rivoli_abi::geom::{FP8_LUT_ENTRIES, GEMV_BLOCK};
use rivoli_abi::kernarg::{Gemv, GemvFp8, Scaled, Vq, VqEncode};

/// The C's `int` extent as the `size_t` it widened to (sign-extending; every launcher refuses
/// non-positive extents first).
#[inline(always)]
fn ext(v: i32) -> usize {
    v as isize as usize
}

/// `x`'s token rows, `i_dim` apart.
#[inline(always)]
fn x_rows<W>(a: &Gemv<W>) -> RowsView {
    RowsView {
        x: a.x,
        stride: ext(a.i_dim),
    }
}

/// The wave-per-row skeleton every GEMV but split-K shares: the wave's output row `o =
/// blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE` (as `size_t`), a wave past `o_dim`
/// leaving (wave-uniform), then `dot(o, lane)`'s `R` reduced rows stored from lane 0 at
/// `y[r·o_dim + o]`.
#[inline(always)]
unsafe fn wave_per_row<W, const R: usize>(a: &Gemv<W>, dot: impl FnOnce(usize, usize) -> [f32; R]) {
    let t = tid() as usize;
    let (lane, o) = (t & (WAVE - 1), bid() as usize * ROWS_PER_BLOCK + t / WAVE);
    let o_dim = ext(a.o_dim);
    if o >= o_dim {
        return;
    }
    for (r, s) in dot(o, lane).into_iter().enumerate() {
        if lane == 0 {
            *a.y.add(r * o_dim + o) = s;
        }
    }
}

/// The fp8 weight row `o` and its tile-scale row (`o >> blk_shift(block)` of a
/// `ceil(i_dim / block)`-wide grid), over the block's LUT.
#[inline(always)]
unsafe fn fp8_row(a: &GemvFp8, o: usize, lut: *const f32) -> Fp8Row {
    let g = &a.gemv;
    // SAFETY: `block > 0` — the launcher refuses `block <= 0` (1001).
    let sc_cols = unchecked_div(g.i_dim + a.block - 1, a.block);
    Fp8Row {
        w: g.w.packed.add(o * ext(g.i_dim)),
        scale: g.w.scale.add((o >> blk_shift(a.block)) * ext(sc_cols)),
        lut,
        i_dim: ext(g.i_dim),
        block: a.block,
    }
}

/// `gemv_fp8_impl<R>`: wave-per-row, the R-row strided dot at unroll 1 (the C's
/// `fp8_dot_strided_r`, deliberately not unrolled — `dot_fp8.hpp`'s note).
#[inline(always)]
unsafe fn gemv_fp8_rows<const R: usize>(a: GemvFp8) {
    let lut = fp8_lut();
    wave_per_row(&a.gemv, |o, lane| {
        dot_fp8_wave_r::<R, 1>(x_rows(&a.gemv), fp8_row(&a, o, lut), lane)
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_fp8(a: GemvFp8) {
    gemv_fp8_rows::<1>(a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_fp8_r2(a: GemvFp8) {
    gemv_fp8_rows::<2>(a);
}

/// `gemv_fp8_splitk_impl<R>`: one row per block, every thread striding it, the waves'
/// partials folded in LDS by one thread per token row in fixed ascending order.
#[inline(always)]
unsafe fn gemv_fp8_splitk_rows<const R: usize>(a: GemvFp8) {
    let lut = fp8_lut();
    let part = lut.add(FP8_LUT_ENTRIES as usize);
    let (o, g) = (bid() as usize, &a.gemv);
    if o >= ext(g.o_dim) {
        return;
    }
    let t = tid() as usize;
    let span = LaneSpan {
        start: t,
        stride: GEMV_BLOCK as usize,
    };
    let acc = fp8_dot_strided_r::<R, 1>(x_rows(g), fp8_row(&a, o, lut), span);
    // Each wave's partial for each row → its lane 0 → `part[r][wave]`.
    let sums = acc.map(wave_sum);
    if t & (WAVE - 1) == 0 {
        for (r, s) in sums.into_iter().enumerate() {
            *part.add(r * ROWS_PER_BLOCK + t / WAVE) = s;
        }
    }
    sync_threads();
    if t < R {
        *g.y.add(t * ext(g.o_dim) + o) = fold_partials(part.add(t * ROWS_PER_BLOCK));
    }
}

/// One token row's `ROWS_PER_BLOCK` wave partials, folded from `0.0` in ascending wave order
/// — the single-row kernel's order, so the split-K combine is deterministic per row.
#[inline(always)]
unsafe fn fold_partials(part: *const f32) -> f32 {
    let mut s = 0.0f32;
    for w in 0..ROWS_PER_BLOCK {
        s += *part.add(w);
    }
    s
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_fp8_splitk(a: GemvFp8) {
    gemv_fp8_splitk_rows::<1>(a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_fp8_splitk_r2(a: GemvFp8) {
    gemv_fp8_splitk_rows::<2>(a);
}

/// `dot_vq_wave` is `dot_vq_wave_r<1>` (the C passed stride 0; at R = 1 no stride is read).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_vq(a: Gemv<Vq>) {
    let rb = ext(a.i_dim / VQ_DIM as i32) * VQ_INDEX_BITS / 8;
    let ng = ext(a.i_dim) / VQ_GROUP;
    wave_per_row(&a, |o, lane| {
        let w = VqRow {
            idx: a.w.indices.add(o * rb),
            scale: a.w.scales.add(o * ng),
            cb: a.w.codebook.cast::<f16>(),
            i_dim: ext(a.i_dim),
        };
        dot_vq_wave_r::<1>(x_rows(&a), w, lane)
    });
}

/// `R` fused accumulates per column over the C's `int` lane-strided loop from `start`:
/// `acc[r] = fma(x[r][i], w(i), acc[r])`. The accumulators move by value through `from_fn`:
/// threaded as `&mut` and stepped with `iter_mut`, `gemv_i8`'s landed in 8-12 B of scratch
/// where hipcc keeps them in registers (`llvm-readelf --notes`, 2026-11-19).
#[inline(always)]
unsafe fn lane_cols<W, const R: usize>(
    a: &Gemv<W>,
    start: u32,
    mut acc: [f32; R],
    w: impl Fn(usize) -> f32,
) -> [f32; R] {
    let x = x_rows(a);
    for_i32(start, a.i_dim, WAVE as u32, |i| {
        let wi = w(i);
        acc = core::array::from_fn(|r| fma(*x.at(r, i), wi, acc[r]));
    });
    acc
}

/// `gemv_f32_impl<R>`.
#[inline(always)]
unsafe fn gemv_f32_rows<const R: usize>(a: Gemv<*const f32>) {
    wave_per_row(&a, |o, lane| {
        let row = a.w.add(o * ext(a.i_dim));
        lane_cols(&a, lane as u32, [0.0; R], |i| *row.add(i)).map(wave_sum)
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_f32(a: Gemv<*const f32>) {
    gemv_f32_rows::<1>(a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_f32_r2(a: Gemv<*const f32>) {
    gemv_f32_rows::<2>(a);
}

/// The two bf16 weights packed in one dword, in ascending column order.
///
/// `bf16f` of the low half is `p << 16` and of the high half `p & 0xffff_0000` — the shift
/// drops the bits it would have masked, so this is one `v_lshlrev_b32` and one `v_and_b32`
/// beside the loads rather than two 16-bit extracts, and it is the same value either way.
#[inline(always)]
fn bf16_pair(p: u32) -> [f32; 2] {
    [f32::from_bits(p << 16), f32::from_bits(p & 0xffff_0000)]
}

fn fma_pair(x: [f32; 2], p: u32, acc: f32) -> f32 {
    let [w0, w1] = bf16_pair(p);
    crate::gpu::fma(x[1], w1, crate::gpu::fma(x[0], w0, acc))
}

/// One output row's dot, wave-cooperative: `Σ_i x[i]·bf16(w[i])` over this row's `i_dim`
/// codes.
///
/// **Eight columns per lane per step while BOTH spans are 16-byte aligned** — one 16 B weight
/// load against two 16 B activation loads, `WAVE·8 = 256` columns per wave step, four steps in
/// flight (`for_strided::<4>`: 2 KB of weights in flight per wave, no LDS). The trunk's widths
/// are all multiples of 256 — Mini's 2304/3072/1024 and Small's 4608/6144/1024 — so they run
/// the vector path end to end with no tail; the MessageGate rank (64) and `d_rel` (16) are the
/// widths that are tail only.
///
/// Two accumulators, the low and high four columns of each step, so the two `v_fmac` chains
/// cannot serialise on one another. The combine (`lo + hi`) and then the ascending scalar tail
/// leave the columns in order, and `wave_sum` reduces across lanes in its fixed ladder.
///
/// The alignment test is on BOTH spans and decides the whole row rather than a prefix: a
/// `global_load_b128` on a misaligned address faults, and `F32x4::load` is UB there, so a row
/// that is not aligned takes the scalar path for every column.
#[inline(always)]
unsafe fn dot_bf16_wave(x: *const f32, w: *const u16, i_dim: usize, lane: usize) -> f32 {
    // Columns per wave step: one lane's eight, across the wave.
    const STEP: usize = WAVE * 8;
    let vector = if w as usize & 15 == 0 && x as usize & 15 == 0 {
        i_dim / STEP * STEP
    } else {
        0
    };
    let (mut lo, mut hi) = (0.0f32, 0.0f32);
    for_strided::<4>(lane * 8, vector, STEP, |c| {
        // SAFETY: `c` is a multiple of 8 and both spans are 16-byte aligned, so the 16 B load
        // and the two `float4`s it is paired with are aligned and inside the row.
        let wp = U32x4::load(w.add(c)).0;
        let xl = F32x4::load(x.add(c)).0;
        let xh = F32x4::load(x.add(c + 4)).0;
        lo = fma_pair([xl[0], xl[1]], wp[0], fma_pair([xl[2], xl[3]], wp[1], lo));
        hi = fma_pair([xh[0], xh[1]], wp[2], fma_pair([xh[2], xh[3]], wp[3], hi));
    });
    let mut acc = lo + hi;
    // SAFETY: the tail columns are the ones the vector pass did not take, each inside `x` and
    // the weight row.
    for_strided::<1>(vector + lane, i_dim, WAVE, |i| {
        acc = crate::gpu::fma(*x.add(i), bf16f(*w.add(i)), acc);
    });
    wave_sum(acc)
}

/// `y[o] = Σ_i x[i]·bf16(w[o, i])` — one token row through a row-major `[o_dim, i_dim]` bf16
/// weight, the decode projections' shape (no C twin: `gemm_bf16` stays scalar for bit-identity;
/// scored by `kernel_gemv_bf16.rs`). See [`dot_bf16_wave`] for the wide path.
///
/// The grid is the GEMV family's row grid (`ceil(o_dim / ROWS_PER_BLOCK)` blocks of
/// `GEMV_BLOCK`), and `w` must be `o_dim · i_dim` codes: the launcher's extents are the only
/// bound the kernel has.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_bf16(a: Gemv<*const u16>) {
    let i_dim = ext(a.i_dim);
    wave_per_row::<*const u16, 1>(&a, |o, lane| {
        [dot_bf16_wave(a.x, a.w.add(o * i_dim), i_dim, lane)]
    });
}

/// `gemv_i8_impl<R>`: four columns per lane (one dword of int8 against one `float4` of `x`)
/// while the row and `x` are aligned — and, at `R > 1`, `i_dim % 4 == 0` — then a scalar
/// tail from where the quads stopped; the row scale multiplies the reduced sum.
#[inline(always)]
unsafe fn gemv_i8_rows<const R: usize>(a: Gemv<Scaled>) {
    wave_per_row(&a, |o, lane| {
        let row = a.w.packed.add(o * ext(a.i_dim));
        let x = x_rows(&a);
        let mut acc = [0.0f32; R];
        let mut base = 0i32;
        // `x + r·i_dim` stays float4-aligned only if `i_dim % 4 == 0` — a condition at R > 1
        // alone, so the single-row path keeps its quads on a ragged `i_dim` (the C's note).
        let rows_aligned = R == 1 || a.i_dim & 3 == 0;
        let aligned = row as usize & 3 == 0 && a.x as usize & 15 == 0;
        if aligned && rows_aligned {
            const STEP: usize = WAVE * 4;
            // A `usize` column stepped beside the `int` bound. Re-deriving it from `base` emitted
            // a `v_ashrrev_i32 …, 31` + `v_lshlrev_b64` per iteration where hipcc bumps pointers
            // by `0x80`/`0x200` (ISA read, unpriced; rust-kernels.md §5, the L1 device lease). An
            // array of advanced pointers lost the `x` load's global address space
            // (`flat_load_b128`). Same addresses either way.
            let mut col = lane * 4;
            while base + STEP as i32 <= a.i_dim {
                let p = *row.add(col).cast::<u32>();
                let wq = [p, p >> 8, p >> 16, p >> 24].map(|b| f32::from(b as i8));
                acc = core::array::from_fn(|r| acc[r] + quad(F32x4::load(x.at(r, col)).0, wq));
                col += STEP;
                base += STEP as i32;
            }
        }
        let acc = lane_cols(&a, base as u32 + lane as u32, acc, |i| {
            f32::from(*row.add(i) as i8)
        });
        let s = *a.w.scale.add(o);
        acc.map(|v| wave_sum(v) * s)
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_i8(a: Gemv<Scaled>) {
    gemv_i8_rows::<1>(a);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_i8_r2(a: Gemv<Scaled>) {
    gemv_i8_rows::<2>(a);
}

/// `dot_i4_wave` is `dot_nib_wave_r::<1, 1, I4>` (the prelude's note: R=1, never unrolled).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemv_i4(a: Gemv<Scaled>) {
    let rb = ext((a.i_dim + 1) / 2);
    let ng = ext((a.i_dim + I4_GROUP as i32 - 1) / I4_GROUP as i32);
    wave_per_row(&a, |o, lane| {
        let w: I4Row = NibRow {
            w: a.w.packed.add(o * rb),
            scale: a.w.scale.add(o * ng),
            dim: ext(a.i_dim),
        };
        dot_nib_wave_r::<1, 1, _>(x_rows(&a), w, lane)
    });
}

/// Unfused throughout (the C's `fp contract(off)`): Rust never contracts, so plain `*`/`+`
/// IS the C here — a fused dot would flip near-tied argmins on the learned codebook.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn vq_encode(a: VqEncode) {
    elementwise(a.n, |i| {
        let sub = a.sub.add(i * VQ_DIM);
        let sv: [f32; VQ_DIM] = core::array::from_fn(|d| *sub.add(d));
        // `int` counter and argmin, as the C. With `usize` (ISA read, 2026-11-19, unpriced) each
        // argmin update was two `v_cndmask_b32_e64` (a 64-bit select), and the range's guarded
        // step stayed a `v_cmp_gt_u64` in every unrolled copy, with four `s_load_b128` where
        // hipcc issues one `s_load_b512`. An `i32` range gives hipcc's loop, merge included.
        let (mut best, mut bk) = (3.4e38f32, 0i32);
        for k in 0..VQ_K as i32 {
            let c = a.codebook.add(k as usize * VQ_DIM);
            let mut dot = 0.0f32;
            for (d, s) in sv.iter().enumerate() {
                dot += s * *c.add(d);
            }
            let dist = *a.cbnorm.add(k as usize) - 2.0 * dot;
            if dist < best {
                best = dist;
                bk = k;
            }
        }
        *a.idx.add(i) = bk as u16;
    });
}
