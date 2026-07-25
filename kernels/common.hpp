// Device helpers shared by the separately-compiled kernel translation units. The
// bf16/e4m3 pair MUST stay bit-exact with math.rs (the CPU oracle the kernel tests
// compare against), so one definition is a correctness property. int4-free.
#pragma once

#include <hip/hip_runtime.h>

// ponytail: WAVE=32 hardcoded for gfx1151 (RDNA3.5, native wave32); __shfl_down
// width=32 keeps it correct on a wave64 part too (reduces within 32-lane groups).
#define WAVE 32
#define ROWS_PER_BLOCK 8  // block = 256 threads = 8 waves → 8 output rows/block

// Sum a wave's partials into lane 0 (fixed __shfl_down order → deterministic).
__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}

// bf16 → f32: bf16 is the high 16 bits of f32 (matches math.rs::bf16_to_f32).
__device__ __forceinline__ float bf16f(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

// f32 → bf16, round-to-nearest-even (matches math.rs::f32_to_bf16). Non-finite
// keeps its top 16 bits verbatim (an RNE carry could turn a NaN into an Inf).
__device__ __forceinline__ unsigned short f2bf16(float x) {
    unsigned int b;
    __builtin_memcpy(&b, &x, sizeof(b));
    if (!isfinite(x)) return (unsigned short)(b >> 16);
    unsigned int r = ((b >> 16) & 1u) + 0x7fffu;
    return (unsigned short)((b + r) >> 16);
}

// fp8-e4m3 → f32 (matches math.rs::e4m3_to_f32). 1 sign / 4 exp (bias 7) / 3 mant;
// exp==0 subnormal = sign·(m/8)·2^-6; (exp==15, mant==7) = NaN.
__device__ __forceinline__ float e4m3f(unsigned char b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    int exp = (b >> 3) & 0x0f;
    float mant = (float)(b & 0x07);
    if (exp == 0) return sign * (mant * 0.125f) * 0.015625f; // 2^-6
    if (exp == 15 && mant == 7.0f) return __int_as_float(0x7fc00000);
    return sign * (1.0f + mant * 0.125f) * exp2f((float)(exp - 7));
}

// Fill a 256-float LDS table with e4m3f(byte) so the hot GEMV decodes fp8 by an
// LDS read instead of the branchy exp2f path (decode was compute-bound, not load-
// width-bound — see the failed load-widening experiment). Bit-exact with e4m3f by
// construction. Caller passes threadIdx.x and __syncthreads() before use; needs a
// >=256-thread block (both callers launch ROWS_PER_BLOCK*WAVE = 256).
__device__ __forceinline__ void e4m3_lut_build(float* lut, int tid) {
    if (tid < 256) lut[tid] = e4m3f((unsigned char)tid);
}

// Wave-cooperative fp8-e4m3 block-scaled dot for one output row `o`. `wrow` =
// packed[o*i_dim..], `scalerow` = scale + (o/block)*sc_cols (the row's block-scale
// row), so element i uses scalerow[i/block]. `lut` is the block's e4m3_lut_build
// table. Matches quant.rs::matvec_fp8.
// Strided fp8 dot accumulation (NO cross-thread reduction): Σ over the columns this
// thread owns of `x·lut[w]·scale`, using a uint-per-lane load (4 fp8 → 128B/wave) atop
// the LUT + a scalar tail. Shared by `dot_fp8_wave` (one wave, stride WAVE) and
// `gemv_fp8_splitk` (all block threads, split-K over one row). The caller reduces.
__device__ __forceinline__ float fp8_dot_strided(const float* __restrict__ x,
                                                 const unsigned char* __restrict__ wrow,
                                                 const float* __restrict__ scalerow,
                                                 int i_dim, int block,
                                                 const float* __restrict__ lut,
                                                 int start, int stride) {
    float acc = 0.0f;
    const unsigned int* w4 = (const unsigned int*)wrow;
    int n4 = i_dim >> 2;
    for (int j = start; j < n4; j += stride) {
        unsigned int p = w4[j];
        int i0 = j << 2;
        float s = scalerow[i0 / block];
        acc += s * (x[i0]     * lut[(unsigned char)p]
                  + x[i0 + 1] * lut[(unsigned char)(p >> 8)]
                  + x[i0 + 2] * lut[(unsigned char)(p >> 16)]
                  + x[i0 + 3] * lut[(unsigned char)(p >> 24)]);
    }
    for (int i = (n4 << 2) + start; i < i_dim; i += stride)
        acc += x[i] * lut[wrow[i]] * scalerow[i / block];
    return acc;
}

// One WAVE reduces one row: strided MAC over the wave's lanes, then a wave-sum.
__device__ __forceinline__ float dot_fp8_wave(const float* __restrict__ x,
                                              const unsigned char* __restrict__ wrow,
                                              const float* __restrict__ scalerow,
                                              int i_dim, int block, int lane,
                                              const float* __restrict__ lut) {
    return wave_sum(fp8_dot_strided(x, wrow, scalerow, i_dim, block, lut, lane, WAVE));
}

// Wave-cooperative per-row int4 dot for one output row: Σ_i v[i]·(nibble(i) − 8),
// result on lane 0 (the per-row scale is applied by the CALLER, outside). `row` =
// packed[o*rb..], rb = (dim+1)/2. Matches quant.rs::matvec_i4. The fast path reads a
// dword (8 nibbles = 8 consecutive columns) per lane when `row` is 4-byte aligned —
// colibri's per-row stride (dim/2, dim a multiple of 8) keeps every row aligned.
__device__ __forceinline__ float nib(unsigned int w, int k) {
    return (float)((int)((w >> (4 * k)) & 0xFu) - 8); // nibble k → signed weight
}
__device__ __forceinline__ float dot_i4_wave(const float* __restrict__ v,
                                             const unsigned char* __restrict__ row,
                                             int dim, int lane) {
    // Two accumulators + two float4 x-loads per lane per step: the 8 columns split
    // into independent FMA chains (ILP), and x streams as 2×16B vector loads instead
    // of 8 scalar loads. int4 is sequential-coalesced, so this saturates the L1/x
    // bandwidth and keeps the ALUs busy (nibble-decode) — unlike the VQ dot's random
    // codebook gather. Fast path needs `row` 4-byte aligned (colibri's dim/2 stride).
    float a0 = 0.0f, a1 = 0.0f;
    int base = 0;
    if ((((size_t)row) & 3u) == 0) {
        const unsigned int* rw = (const unsigned int*)row;
        for (; base + WAVE * 8 <= dim; base += WAVE * 8) {
            int col = base + lane * 8;
            unsigned int w = rw[col >> 3]; // 8 nibbles = 8 consecutive columns
            float4 x0 = *(const float4*)(v + col);
            float4 x1 = *(const float4*)(v + col + 4);
            a0 += x0.x * nib(w, 0) + x0.y * nib(w, 1) + x0.z * nib(w, 2) + x0.w * nib(w, 3);
            a1 += x1.x * nib(w, 4) + x1.y * nib(w, 5) + x1.z * nib(w, 6) + x1.w * nib(w, 7);
        }
    }
    for (int i = base + lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        a0 += v[i] * (float)(n - 8);
    }
    return wave_sum(a0 + a1);
}

#include <hip/hip_fp16.h>

// VQ-int3 codebook parameters — MUST match quant.rs (VQ_DIM/VQ_K/VQ_INDEX_BITS/VQ_GROUP).
#define VQ_DIM 4
#define VQ_K 4096
#define VQ_INDEX_BITS 12
#define VQ_GROUP 64
#define VQ_SUBS_PER_GROUP (VQ_GROUP / VQ_DIM)  // subvectors sharing one bf16 scale

// Wave-cooperative VQ-int3 dot for one output row, result on lane 0. Matches
// quant.rs::matvec_vq: each VQ_DIM-subvector t reads a packed 12-bit codebook
// index, dots x[t*VQ_DIM..] with codebook[idx*VQ_DIM..], and scales by the
// subvector's bf16 group scale (group = t / VQ_SUBS_PER_GROUP). Lane `l` owns
// subvectors t ≡ l (mod WAVE). The two-byte index read is in-bounds because
// i_dim is a multiple of 8 (nsub even; the last odd subvector's high byte is the
// row's last index byte — see quant.rs::get_idx / vq_row_bytes).
//
// The codebook is fp16: the random idx→cb[idx] gather is the latency-bound step,
// and at 8B/entry (VQ_K·VQ_DIM·2 = 32KB) the whole codebook fits in the 32KB L1,
// so the gather is an L1 hit instead of L2 (f32 was 64KB, L2-resident). fp16 keeps
// x in f32 for the products; its 10-bit mantissa on the centroids clears the
// oracle tol (bf16's 8 does not — see math::f32_to_f16).
__device__ __forceinline__ float dot_vq_wave(const float* __restrict__ x,
                                             const unsigned char* __restrict__ idxrow,
                                             const unsigned short* __restrict__ scalerow,
                                             const __half* __restrict__ cb, int i_dim, int lane) {
    int nsub = i_dim / VQ_DIM;
    float acc = 0.0f;
    for (int t = lane; t < nsub; t += WAVE) {
        int bitpos = t * VQ_INDEX_BITS;
        int byte = bitpos >> 3;
        int shift = bitpos & 7;
        unsigned int raw = (unsigned int)idxrow[byte] | ((unsigned int)idxrow[byte + 1] << 8);
        int idx = (int)((raw >> shift) & 0xFFFu);
        // 8B fp16 gather (two __half2) for the VQ_DIM=4 subvector; products in f32,
        // same 4 terms in the same order as the f32 path.
        const __half* c = cb + (size_t)idx * VQ_DIM;
        float2 c01 = __half22float2(*(const __half2*)c);
        float2 c23 = __half22float2(*(const __half2*)(c + 2));
        float4 xv = *(const float4*)(x + (size_t)t * VQ_DIM);
        float dot = xv.x * c01.x + xv.y * c01.y + xv.z * c23.x + xv.w * c23.y;
        acc += bf16f(scalerow[t / VQ_SUBS_PER_GROUP]) * dot;
    }
    return wave_sum(acc);
}
