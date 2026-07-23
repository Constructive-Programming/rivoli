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
__device__ __forceinline__ float dot_fp8_wave(const float* __restrict__ x,
                                              const unsigned char* __restrict__ wrow,
                                              const float* __restrict__ scalerow,
                                              int i_dim, int block, int lane,
                                              const float* __restrict__ lut) {
    float acc = 0.0f;
    for (int i = lane; i < i_dim; i += WAVE) acc += x[i] * lut[wrow[i]] * scalerow[i / block];
    return wave_sum(acc);
}

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
__device__ __forceinline__ float dot_vq_wave(const float* __restrict__ x,
                                             const unsigned char* __restrict__ idxrow,
                                             const unsigned short* __restrict__ scalerow,
                                             const float* __restrict__ cb, int i_dim, int lane) {
    int nsub = i_dim / VQ_DIM;
    float acc = 0.0f;
    for (int t = lane; t < nsub; t += WAVE) {
        int bitpos = t * VQ_INDEX_BITS;
        int byte = bitpos >> 3;
        int shift = bitpos & 7;
        unsigned int raw = (unsigned int)idxrow[byte] | ((unsigned int)idxrow[byte + 1] << 8);
        int idx = (int)((raw >> shift) & 0xFFFu);
        const float* c = cb + (size_t)idx * VQ_DIM;
        const float* xv = x + t * VQ_DIM;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < VQ_DIM; ++d) dot += xv[d] * c[d];
        acc += bf16f(scalerow[t / VQ_SUBS_PER_GROUP]) * dot;
    }
    return wave_sum(acc);
}
