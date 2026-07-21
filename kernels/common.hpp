// Device helpers shared by the separately-compiled kernel translation units.
// These were duplicated per-file; the bf16 pair in particular MUST stay bit-exact
// with math.rs (the scalar oracle the kernel tests compare against), so one
// definition is a correctness property, not just fewer lines.
#pragma once

#include <hip/hip_runtime.h>

// ponytail: WAVE=32 hardcoded for gfx1151 (RDNA3.5, native wave32); __shfl_down
// width=32 also keeps it correct on a wave64 part (reduces within 32-lane
// groups, matching the logical wave = threadIdx/32). A part with warpSize<32
// would need WAVE lowered.
#define WAVE 32
#define ROWS_PER_BLOCK 8  // block = 256 threads = 8 waves → 8 output rows/block

// Sum a wave's partials into lane 0 (fixed __shfl_down order → deterministic,
// so greedy decode stays schedule-independent).
__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}

// Wave-cooperative int4 dot, result on lane 0. Matches quant.rs: low nibble =
// even column, high = odd; signed value = nibble − 8.
//
// Lane `l` owns the CONTIGUOUS 8-column block [base + l*8, +8) of each
// WAVE*8 = 256-column stripe, so its packed bytes are one aligned `unsigned int`
// and a wave-load covers 128 B = a full cache line. The previous form gave lane
// `l` the columns i ≡ l (mod 32), which made lanes l and l+1 read the SAME byte
// and a whole wave touch only 16 B per load — one eighth of a line, so ~8x the
// load instructions for the same traffic. Measured on gfx1151 at the GLM MoE
// shapes: 74.4 → 192 GB/s (docs/probes/i4gemv_probe.cpp).
//
// The dword path needs `row` 4 B-aligned. Every snapshot tensor is (data offsets
// are ≡ 8 mod 16) and every row stride here is a multiple of 4, so the scalar
// fallback below is never taken in practice — but a misaligned dword load would
// fault, so the guard is not optional. The tail likewise never runs at GLM dims
// (6144/2048/12288/16384/512 all divide by 256); it keeps the helper total.
//
// Summation order differs from the scalar oracle's ascending sum by f32 rounding
// (well inside the kernel tests' 1e-3 int4 dequant tolerance) but is
// SCHEDULE-INDEPENDENT, so greedy decode stays deterministic.
__device__ __forceinline__ float dot_i4_wave(const float* __restrict__ v,
                                             const unsigned char* __restrict__ row,
                                             int dim, int lane) {
    float acc = 0.0f;
    int base = 0;
    if ((((size_t)row) & 3u) == 0) {
        const unsigned int* rw = (const unsigned int*)row;
        for (; base + WAVE * 8 <= dim; base += WAVE * 8) {
            int col = base + lane * 8;
            unsigned int w = rw[col >> 3];  // 8 nibbles = 8 consecutive columns
            const float* vv = v + col;
#pragma unroll
            for (int k = 0; k < 8; ++k)
                acc += vv[k] * (float)((int)((w >> (4 * k)) & 0xFu) - 8);
        }
    }
    for (int i = base + lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        acc += v[i] * (float)(n - 8);
    }
    return wave_sum(acc);
}

// int8 analog: lane `l` owns 4 consecutive columns as one dword, so a wave-load
// covers 128 B instead of the byte-per-lane form's 32 B. Values are signed bytes
// (matches quant.rs). Same alignment guard and tail as dot_i4_wave.
__device__ __forceinline__ float dot_i8_wave(const float* __restrict__ v,
                                             const unsigned char* __restrict__ row,
                                             int dim, int lane) {
    float acc = 0.0f;
    int base = 0;
    if ((((size_t)row) & 3u) == 0) {
        const unsigned int* rw = (const unsigned int*)row;
        for (; base + WAVE * 4 <= dim; base += WAVE * 4) {
            int col = base + lane * 4;
            unsigned int w = rw[col >> 2];
            const float* vv = v + col;
#pragma unroll
            for (int k = 0; k < 4; ++k)
                acc += vv[k] * (float)((signed char)((w >> (8 * k)) & 0xFFu));
        }
    }
    for (int i = base + lane; i < dim; i += WAVE) acc += (float)((signed char)row[i]) * v[i];
    return wave_sum(acc);
}

// bf16 → f32: bf16 is the high 16 bits of f32 (matches math.rs::bf16_to_f32).
__device__ __forceinline__ float bf16f(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

// f32 → bf16, round-to-nearest-even (matches math.rs::f32_to_bf16). Non-finite
// keeps its top 16 bits verbatim — the RNE carry could otherwise turn a NaN into
// an Inf, so Inf/NaN survive the round-trip as themselves.
__device__ __forceinline__ unsigned short f2bf16(float x) {
    unsigned int b;
    __builtin_memcpy(&b, &x, sizeof(b));
    if (!isfinite(x)) return (unsigned short)(b >> 16);
    unsigned int r = ((b >> 16) & 1u) + 0x7fffu;
    return (unsigned short)((b + r) >> 16);
}
