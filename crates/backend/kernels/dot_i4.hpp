// The int4 GROUP-SCALED dot — one f32 scale per I4_GROUP weights along the input dim, so
// the scale lives INSIDE the dot. Split out of common.hpp 2026-08-15 with the other three
// dot families; the bodies and the measurements in their comments travelled verbatim, and
// the signatures were bundled into the views in `rowview.hpp` the same day (that file
// carries the aliasing argument for every pointer below).
#pragma once

#include <hip/hip_runtime.h>

#include "reduce.hpp"
#include "rowview.hpp"

// int4 group-scale parameters — MUST match quant.rs (I4_GROUP). One f32 scale per
// I4_GROUP weights along the input dim.
#define I4_GROUP 128
#define I4_GROUP_SHIFT 7
static_assert((1 << I4_GROUP_SHIFT) == I4_GROUP, "I4_GROUP_SHIFT must be log2(I4_GROUP)");
static_assert(I4_GROUP % 8 == 0, "the dword fast path's 8 columns must not straddle a group");

// One packed int4 weight row and its group-scale row:
//   Σ_i v[i]·(nibble(i) − 8)·scale[i / I4_GROUP]
// `w` = packed[o*rb..], rb = (dim+1)/2; `scale` = scale + o*i4_groups(dim).
// Matches quant.rs::matvec_i4. NOTE the scale is applied per GROUP, inside the dot —
// under the old per-row format the caller applied one scale outside the dot, which is
// exactly what a group scale cannot express.
struct I4Row {
    const unsigned char* __restrict__ w;
    const float* __restrict__ scale;
    int dim;
};

// nibble k → signed weight.
//
// The fast paths below read a dword (8 nibbles = 8 consecutive columns) per lane when the
// row is 4-byte aligned (the dim/2 row stride, dim a multiple of 8, keeps every row
// aligned). Those 8 columns start at a multiple of 8 and I4_GROUP is a multiple of 8,
// so they always share ONE group scale — one extra multiply per 8 columns, and the
// scalar tail computes the same per-element product.
__device__ __forceinline__ float nib(unsigned int w, int k) {
    return (float)((int)((w >> (4 * k)) & 0xFu) - 8);
}

// `dot_i4_wave_r`'s dword fast path, returning the column `base` its scalar tail resumes
// from. A row that is not 4-byte aligned returns 0 and is handed to the tail whole — the
// same partition the wrapping `if` used to express, written as an early return so this
// loop sits at one level of nesting instead of two.
//
// Lifted out 2026-08-15, SHAPE ONLY. The `#pragma unroll 4` and every number the note
// inside it measures travel WITH the loop; the fold order, the two accumulators and the
// group-scale hoist are untouched.
//
// `a` is the caller's `AccPair<R>` BY REFERENCE, not two `float* __restrict__`: it is
// in/out, it is a local of the caller that never escapes, and after the `__forceinline__`
// it is SROA'd into registers — see rowview.hpp rule 1.
template <int R>
__device__ __forceinline__ int i4_dword_pass(RowsView v, I4Row w, int lane, AccPair<R>& a) {
    int base = 0;
    if ((((size_t)w.w) & 3u) != 0) return base;
    const unsigned int* rw = (const unsigned int*)w.w;
    // Unrolled 2026-08-09, MEASURED, not copied from fp4. Un-pragma'd, this loop issued
    // 4 (R=1) / 6 (R=2) loads and drained them all in-body — `vmcnt(3) lgkmcnt(0)` down
    // to `vmcnt(0)`, one iteration in flight, M7's disease and the same gap M11 fixed on
    // `dot_f4_wave_r`. `dot_bench glmi4` at the artifact's dims (6144x2048, 1.083 GB
    // rotating past the 32 MB MALL, two counterbalanced passes, benchmarks.md "GLM int4
    // MoE unroll round"): depth 4 is **+12.6% at R=1 (169.2 -> 190.6 GB/s), +16.4% at
    // R=2 (163.3 -> 190.1), +20.1% at e_count=1 (125.9 -> 151.3)**, fingerprint-identical
    // on BOTH token rows; depth 2 gave +11.6%/+3.0%, so depth 4 is the rung that pays at
    // the R=2 width speculative decode actually runs.
    //
    // The register cost the un-unrolled comment here worried about was measured FIRST and
    // does not bite: depth 4 is VGPR 88/123/83/95 across the four kernels, occupancy 16
    // everywhere except gateup_r2 at 10 waves/SIMD — exactly where fp4's winning arm sat —
    // zero spill, zero scratch. Two adjacent negatives, priced in the same round so nobody
    // re-tries them: AS1-typing these pointers (the fp4 `gu8p` treatment; flat_load ->
    // global_load, the lgkmcnt coupling gone) measured +-0.6% == nothing, and a ballast
    // with the whole nibble decode and every FMA removed measured +0.5%/-1.1% — the decode
    // is FREE at the un-unrolled schedule, so the entire gap was memory-level parallelism.
    //
    // Fold order is unchanged — each accumulator stays one serial fadd chain, ascending
    // `base` (the fingerprint gate would have caught anything else; arm X, a deliberate
    // even/odd split, moved it and was measured doing so). Multi-trip coverage INCLUDING
    // the remainder loop this pragma creates: tests/kernel.rs::
    // the_i4_dword_path_matches_the_oracle_at_multiple_trips (1280/1024 = 5 and 4 trips;
    // every engine dim is 0 mod 4, so only that fixture ever enters the epilogue).
#pragma unroll 4
    for (; base + WAVE * 8 <= w.dim; base += WAVE * 8) {
        int col = base + lane * 8;
        unsigned int p = rw[col >> 3];                // 8 nibbles = 8 consecutive columns
        float s = w.scale[col >> I4_GROUP_SHIFT];     // one group for all 8
        // Decoded ONCE for every row.
        float n0 = nib(p, 0), n1 = nib(p, 1), n2 = nib(p, 2), n3 = nib(p, 3);
        float n4 = nib(p, 4), n5 = nib(p, 5), n6 = nib(p, 6), n7 = nib(p, 7);
#pragma unroll
        for (int t = 0; t < R; ++t) {
            const float* vt = v.x + (size_t)t * v.stride;
            float4 x0 = *(const float4*)(vt + col);
            float4 x1 = *(const float4*)(vt + col + 4);
            a.a0.v[t] += s * (x0.x * n0 + x0.y * n1 + x0.z * n2 + x0.w * n3);
            a.a1.v[t] += s * (x1.x * n4 + x1.y * n5 + x1.z * n6 + x1.w * n7);
        }
    }
    return base;
}

// The per-column remainder, from `t.base` to `w.dim`. Folds into `a0` only — the dword
// path's `a1` exists to give it two independent FMA chains and there is no second chain
// here.
//
// The left-to-right `v[i] * (n−8) * scale` is the arithmetic, not an accident of
// spelling: hoisting `(n−8) * scale` out of the row loop would re-associate the product.
template <int R>
__device__ __forceinline__ void i4_tail_accum(RowsView v, I4Row w, TailSpan t, Acc<R>& a0) {
    for (int i = t.base + t.lane; i < w.dim; i += WAVE) {
        unsigned char b = w.w[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
#pragma unroll
        for (int r = 0; r < R; ++r)
            a0.v[r] += v.x[(size_t)r * v.stride + i] * (float)(n - 8)
                       * w.scale[i >> I4_GROUP_SHIFT];
    }
}

// R token rows against ONE read of the weight row — the nibble decode and the group scale
// amortise over the rows, which is the whole point of batching (the weight read is ~92% of
// an expert launch). `v.stride` is the row-minor stride of the activations.
//
// R=1 is BIT-IDENTICAL to `dot_i4_wave` below, and that is a constraint on how this is
// written, not a hope: the fast path's `s * (x·n + …)` grouping and the scalar tail's
// left-to-right `v[i] * (n−8) * scale` are both reproduced exactly. Hoisting the nibbles
// is safe (same values, same order); hoisting `(n−8) * scale` out of the tail would NOT be
// — it re-associates the product. See tests/kernel.rs.
//
// The two passes are a PARTITION of the columns, which is why `base` is threaded between
// them by value rather than each pass deciding its own bound.
template <int R>
__device__ __forceinline__ Acc<R> dot_i4_wave_r(RowsView v, I4Row w, int lane) {
    AccPair<R> a;
#pragma unroll
    for (int t = 0; t < R; ++t) {
        a.a0.v[t] = 0.0f;
        a.a1.v[t] = 0.0f;
    }
    int base = i4_dword_pass<R>(v, w, lane, a);
    i4_tail_accum<R>(v, w, TailSpan{base, lane}, a.a0);
    Acc<R> out;
#pragma unroll
    for (int t = 0; t < R; ++t) out.v[t] = wave_sum(a.a0.v[t] + a.a1.v[t]);
    return out;
}

__device__ __forceinline__ float dot_i4_wave(const float* __restrict__ v, I4Row w, int lane) {
    // Two accumulators + two float4 x-loads per lane per step: the 8 columns split
    // into independent FMA chains (ILP), and x streams as 2×16B vector loads instead
    // of 8 scalar loads. int4 is sequential-coalesced, so this saturates the L1/x
    // bandwidth and keeps the ALUs busy (nibble-decode) — unlike the VQ dot's random
    // codebook gather.
    float a0 = 0.0f, a1 = 0.0f;
    int base = 0;
    if ((((size_t)w.w) & 3u) == 0) {
        const unsigned int* rw = (const unsigned int*)w.w;
        for (; base + WAVE * 8 <= w.dim; base += WAVE * 8) {
            int col = base + lane * 8;
            unsigned int p = rw[col >> 3];             // 8 nibbles = 8 consecutive columns
            float s = w.scale[col >> I4_GROUP_SHIFT];  // one group for all 8
            float4 x0 = *(const float4*)(v + col);
            float4 x1 = *(const float4*)(v + col + 4);
            a0 += s * (x0.x * nib(p, 0) + x0.y * nib(p, 1) + x0.z * nib(p, 2) + x0.w * nib(p, 3));
            a1 += s * (x1.x * nib(p, 4) + x1.y * nib(p, 5) + x1.z * nib(p, 6) + x1.w * nib(p, 7));
        }
    }
    for (int i = base + lane; i < w.dim; i += WAVE) {
        unsigned char b = w.w[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        a0 += v[i] * (float)(n - 8) * w.scale[i >> I4_GROUP_SHIFT];
    }
    return wave_sum(a0 + a1);
}
