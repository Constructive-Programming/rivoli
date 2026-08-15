// Wave and block GEOMETRY, and the three reductions defined over it. Split out of
// common.hpp 2026-08-15 under the per-file line ceiling; the bodies and their
// measurements moved verbatim.
//
// The geometry macros live HERE rather than in the umbrella because they are what the
// reductions are parameterised by — `wave_sum`/`wave_max` fold over exactly WAVE lanes,
// and ROWS_PER_BLOCK is the same block's other axis (8 waves = 8 output rows). Every
// translation unit still sees both: common.hpp includes this header first.
//
// No `#pragma clang fp contract(off)` anywhere below, and none is needed: the only
// operators in this file are `+` and `fmaxf` over shuffles, so there is no multiply for an
// FMA to absorb. common.hpp's "V4 shared device helpers" note runs that same argument over
// the helpers it covers and names `f2e4m3_rne` (formats.hpp) as the one that DOES need the
// pragma, with the ISA diff that measured it.
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

// Max across a wave, result on EVERY lane. A butterfly rather than `wave_sum`'s ladder,
// which is legal here and not there: `fmaxf` is EXACTLY associative, so the two orders
// agree bit-for-bit, and the caller needs the result on every lane anyway.
//
// REQUIRES A FULL WAVE. `__shfl_xor` against an inactive lane is undefined, and unlike
// `wave_sum` — whose ladder leaves lane 0 right regardless — this promises every lane. Its
// one caller satisfies it: `act_quant_f8`'s early return is wave-uniform, because `b` is
// derived from `threadIdx.x / WAVE`.
__device__ __forceinline__ float wave_max(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WAVE));
    return v;
}

// Block-wide sum of `v` into every thread. `red` is caller-owned LDS of `blockDim.x`
// floats; on return every thread holds the total.
//
// **REQUIRES A POWER-OF-TWO `blockDim.x`.** The halving ladder drops the odd element at
// every level where it is not — at 6 threads `red[2]` never reaches `red[0]` — and the
// result is a quietly wrong sum, not a fault. Every caller launches 256. Stated because
// this is now reachable from `attn.hip`/`linalg.hip`/`moe.hip`/`fwd.hip`, where a
// 192-thread block is an ordinary edit; `wave_max` above documents its own precondition
// for the same reason.
//
// RE-ASSOCIATED relative to the oracle's sequential fold — this is the one place the V4
// kernels knowingly diverge from it, and it is the floor any tolerance built on the
// goldens has to clear.
//
// The trailing `__syncthreads()` is what makes `red` reusable for a second reduction in the
// same kernel; the other of the two spellings this was unified from lacked it. No shipped
// caller reduces twice, so it buys one barrier per row against a landmine for the first
// caller that does. That barrier is NEW for `rmsnorm_batch` and `qk_norm`, which had the
// spelling without it: one extra block barrier per token per norm, on the decode path.
// UNMEASURED — these kernels are bandwidth-bound so the expectation is that it is free,
// but that is an expectation, in the same sense as `mla.hip`'s pragma note.
__device__ __forceinline__ float block_sum_lds(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    float t = red[0];
    __syncthreads();
    return t;
}
