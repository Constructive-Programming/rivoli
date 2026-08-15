// The fp8-e4m3 BLOCK-SCALED dot — the attention and dense projections' weight format —
// and the LDS LUT it decodes through. Split out of common.hpp 2026-08-15 with the other
// three dot families; the bodies and the measurements in their comments travelled
// verbatim, and the signatures were bundled into the views in `rowview.hpp` the same day
// (see that file for the aliasing argument that covers every pointer below).
//
// What separates this family from `dot_i4`/`dot_f4`/`dot_vq`, and why they share no code:
// fp8's scale is a **TILE grid** — `[ceil(o_dim/block), ceil(i_dim/block)]`, one scale
// shared by `block` OUTPUT rows — so a row's scale row is picked by the CALLER
// (`scale + (o >> blk_shift(block)) * sc_cols`) and indexed inside the loop by
// `i >> bsh`. The other three each carry ONE scale row per output row with the group
// running along the input dim only. Same shape of loop, different addressing, and the
// merge would need a policy parameter at the one line it exists to share.
#pragma once

#include <hip/hip_runtime.h>

#include "formats.hpp"
#include "reduce.hpp"
#include "rowview.hpp"

// Fill a 256-float LDS table with e4m3f(byte) so the hot GEMV decodes fp8 by an
// LDS read instead of the branchy exp2f path (decode was compute-bound, not load-
// width-bound — see the failed load-widening experiment). Bit-exact with e4m3f by
// construction. Caller passes threadIdx.x and __syncthreads() before use; needs a
// >=256-thread block (both callers launch ROWS_PER_BLOCK*WAVE = 256).
//
// It sits HERE, not with `e4m3f` in formats.hpp, because it is not an element codec: it
// builds the table `Fp8Row::lut` points at, and its only two callers are the fp8 GEMVs in
// `linalg.hip`/`mla.hip` that then hand that table to the dots below.
__device__ __forceinline__ void e4m3_lut_build(float* lut, int tid) {
    if (tid < 256) lut[tid] = e4m3f((unsigned char)tid);
}

// One packed fp8 weight row, everything needed to decode it, and nothing else. `w` =
// packed[o*i_dim..]; `scale` = the row's block-scale row, i.e. `scale + (o/block)*sc_cols`
// picked by the caller; `lut` is the block's `e4m3_lut_build` table. Element `i` uses
// `scale[i >> blk_shift(block)]` and `lut[w[i]]`. Matches quant.rs::matvec_fp8.
struct Fp8Row {
    const unsigned char* __restrict__ w;
    const float* __restrict__ scale;
    const float* __restrict__ lut;
    int i_dim;
    int block;
};

// One dword's worth of the weight row, already decoded: the column its four weights start
// at, the ONE block scale they share, and the four LUT values. Both amortise over the R
// activation rows, which is why they are hoisted to the call and arrive as a value.
//
// The four weights are SCALARS, not a pointer to the caller's local array. An array would
// have to survive SROA to stay in registers, and there is no reason to hand the optimizer
// that job when the values are already in registers at the call.
struct Fp8Quad {
    int i0;
    float s;
    float l0, l1, l2, l3;
};

// One column of the weight row: which column, its decoded weight, and its block scale.
// The remainder path's unit, where the dword read does not fit.
struct Fp8Col {
    int i;
    float l;
    float s;
};

// Strided fp8 dot accumulation (NO cross-thread reduction): Σ over the columns this
// thread owns of `x·lut[w]·scale`, using a uint-per-lane load (4 fp8 → 128B/wave) atop
// the LUT + a scalar tail. The caller reduces. [CORRECTED 2026-08-08: this said "shared
// by `dot_fp8_wave` and `gemv_fp8_splitk`" — the splitk kernels moved to
// `fp8_dot_strided_r` when the `_r` family landed, so `dot_fp8_wave` → `gemv_fp8_bf16` is
// the ONLY consumer, which is what scopes the M7 unroll's blast radius below.]
//
// `x` stays a `__restrict__` PARAMETER: this is the single-row form, so there is no row
// stride to bundle, and rowview.hpp's rule is to leave a pointer in parameter position
// whenever it can stay there.
__device__ __forceinline__ float fp8_dot_strided(const float* __restrict__ x, Fp8Row w,
                                                 LaneSpan span) {
    float acc = 0.0f;
    const unsigned int* w4 = (const unsigned int*)w.w;
    // The dword path applies ONE block scale to a quad's four columns, so it is only the
    // right scale when the tile is at least a quad wide — at block 1 or 2 the columns
    // past the tile boundary were silently given i0's. Both are powers of two, so guard
    // 1003 passes them. Zeroing `n4` hands the row to the per-column tail below, which
    // was already correct; block ≥ 4 (the engine runs 128) is untouched and bit-identical.
    // TWO KNOWN GAPS REMAIN, both recorded in docs/reference/vulkan-kernels.md
    // ("Known gaps in the fp8 dot"; this cited docs/PERF.md #4 until 2026-08-01,
    // a file split three ways whose item numbering no longer exists): the Vulkan twin
    // (vk/fp8.glsl) still has this bug and its oracle mirrors it, and `rivoli_gemv_fp8`
    // still does not guard the `i_dim % 4` this `w4` cast needs — a requirement that is
    // now CONDITIONAL, since at block < 4 the cast is never reached.
    int n4 = (w.block >= 4) ? (w.i_dim >> 2) : 0;
    int bsh = blk_shift(w.block);
    // Unrolled for memory-level parallelism, NOT for issue: un-unrolled, the emitted loop
    // issued its three vmem loads and then waited them all down (`s_waitcnt vmcnt(0)`
    // before the closing fmac, EVERY iteration), so each wave held exactly one 128-B
    // weight request in flight per GTT round-trip — M6 measured the only kernel built
    // from this loop (`gemv_fp8_bf16`) at 2–2.8× its bytes on all three of its spans while
    // issue sat at 42 instr/128 B, ~3× lighter than streaming needs. Unroll 8 puts 1 KB
    // of weight stream in flight per wave (read back from the ISA: loads `s_clause`'d,
    // waits counted down from vmcnt(23), ONE vmcnt(0) per 1 KB) at VGPR 86 = 96/wave =
    // still 16 waves/SIMD — the next granule (>96) drops to 12, so a change that grows
    // this body must re-read the ISA. Bit-identical by the M5 unroll argument: `acc` is
    // one serial FP chain and LLVM neither splits nor re-associates it without
    // fast-math; the fold order stays ascending `j`. Pinned on hardware by
    // `tests/f4_kernel.rs::the_fp8_dot_sums_in_source_order_through_both_loops`, which
    // also walks the unroll REMAINDER loop no engine dimension reaches (every real
    // per-lane trip count divides 8). `fp8_dot_strided_r` below is deliberately NOT
    // unrolled: its callers (GLM splitk, mla absorb/value) measured at budget, and R
    // multiplies the register cost of the same pragma. Details and the registered
    // prediction: docs/investigations/v4-decode-decomposition.md §M7.
#pragma unroll 8
    for (int j = span.start; j < n4; j += span.stride) {
        unsigned int p = w4[j];
        int i0 = j << 2;
        float s = w.scale[i0 >> bsh];
        acc += s * (x[i0]     * w.lut[(unsigned char)p]
                  + x[i0 + 1] * w.lut[(unsigned char)(p >> 8)]
                  + x[i0 + 2] * w.lut[(unsigned char)(p >> 16)]
                  + x[i0 + 3] * w.lut[(unsigned char)(p >> 24)]);
    }
    for (int i = (n4 << 2) + span.start; i < w.i_dim; i += span.stride)
        acc += x[i] * w.lut[w.w[i]] * w.scale[i >> bsh];
    return acc;
}

// R INPUT ROWS against one weight row — the fp8 twin of `dot_vq_wave_r`, and the same
// argument for existing. The attention projections read 165 MB of fp8 per layer (o_proj
// alone is 100 MB) against a 24 KB row of `x`, so the weight side is the cost and R rows
// through one read of it is what makes a speculative verify pass cheap.
//
// `x` rows are `x.stride` apart. NO cross-thread reduction here either — `dot_fp8_wave_r`
// wave-sums, `gemv_fp8_splitk_r` reduces in LDS.
//
// R=1 is BIT-IDENTICAL to the scalar form: hoisting the four `lut[]` reads into locals
// changes neither the values nor the order they are summed in.

// One dword's four columns folded into R accumulators — the weights already decoded
// through the LUT and the block scale already read, because both amortise over the rows.
// Lifted out of `fp8_dot_strided_r`'s inner loop 2026-08-15, SHAPE ONLY: the same four
// products in the same `s * (…)` grouping, the same ascending fold into `acc.v[r]`.
template <int R>
__device__ __forceinline__ void fp8_quad_accum(RowsView x, Fp8Quad q, Acc<R>& acc) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float* xr = x.x + (size_t)r * x.stride;
        acc.v[r] += q.s * (xr[q.i0] * q.l0 + xr[q.i0 + 1] * q.l1 + xr[q.i0 + 2] * q.l2
                           + xr[q.i0 + 3] * q.l3);
    }
}

// One column folded into R accumulators — the remainder path, where the dword read does
// not fit.
//
// `x * l * s`, NOT `x * (l*s)`. Folding the two weight-side factors first would be one
// fewer multiply per row and a DIFFERENT number — fp multiplication does not associate —
// which is exactly the kind of drift the bit-identity test exists to refuse. The loads are
// hoisted at the call; the arithmetic is not touched.
template <int R>
__device__ __forceinline__ void fp8_col_accum(RowsView x, Fp8Col c, Acc<R>& acc) {
#pragma unroll
    for (int r = 0; r < R; ++r) acc.v[r] += x.x[(size_t)r * x.stride + c.i] * c.l * c.s;
}

// Returns the R accumulators BY VALUE rather than filling a `float* __restrict__ acc` the
// caller owns — rowview.hpp rule 1: the accumulator is pure output here, and a value
// cannot alias the weight spans at all.
template <int R>
__device__ __forceinline__ Acc<R> fp8_dot_strided_r(RowsView x, Fp8Row w, LaneSpan span) {
    Acc<R> acc;
#pragma unroll
    for (int r = 0; r < R; ++r) acc.v[r] = 0.0f;
    const unsigned int* w4 = (const unsigned int*)w.w;
    // See the single-row note above for why `block < 4` zeroes the dword path.
    int n4 = (w.block >= 4) ? (w.i_dim >> 2) : 0;
    int bsh = blk_shift(w.block);
    for (int j = span.start; j < n4; j += span.stride) {
        unsigned int p = w4[j];
        int i0 = j << 2;
        fp8_quad_accum<R>(x,
                          Fp8Quad{i0, w.scale[i0 >> bsh], w.lut[(unsigned char)p],
                                  w.lut[(unsigned char)(p >> 8)],
                                  w.lut[(unsigned char)(p >> 16)],
                                  w.lut[(unsigned char)(p >> 24)]},
                          acc);
    }
    for (int i = (n4 << 2) + span.start; i < w.i_dim; i += span.stride)
        fp8_col_accum<R>(x, Fp8Col{i, w.lut[w.w[i]], w.scale[i >> bsh]}, acc);
    return acc;
}

// One WAVE reduces one row: strided MAC over the wave's lanes, then a wave-sum.
__device__ __forceinline__ float dot_fp8_wave(const float* __restrict__ x, Fp8Row w,
                                              int lane) {
    return wave_sum(fp8_dot_strided(x, w, LaneSpan{lane, WAVE}));
}

// R input rows, one wave, one weight row. R=1 is bit-identical to `dot_fp8_wave`:
// `fp8_dot_strided_r<1>` reproduces the scalar accumulation exactly and `wave_sum` is
// the same reduction.
template <int R>
__device__ __forceinline__ Acc<R> dot_fp8_wave_r(RowsView x, Fp8Row w, int lane) {
    Acc<R> acc = fp8_dot_strided_r<R>(x, w, LaneSpan{lane, WAVE});
    Acc<R> out;
#pragma unroll
    for (int r = 0; r < R; ++r) out.v[r] = wave_sum(acc.v[r]);
    return out;
}
