// MLA absorb + value kernels — the two ops through kv_b the generic GEMV can't
// express, head-batched into one launch each. kv_b is fp8-e4m3 now (was int4):
// [H·(nope+vh) rows, kvl cols], block-scaled (weight_scale_inv, `block`×`block`
// tiles), row stride = kvl bytes. Head `h` owns rows [h·(nope+vh) .. +(nope+vh)):
// the first `nope` are the absorb rows, the next `vh` the value rows. Block scale
// for element (r,c) is scale[(r/block)·sc_cols + c/block], sc_cols = ⌈kvl/block⌉.
// All pointers device-resident; launch-only. Return codes: 0 ok, POSITIVE arg
// guard, NEGATIVE -hipError.

#include "common.hpp"

// qabs[head][i] = Σ_d q[head·qh + d] · e4m3(kv_b[rbase+d][i]) · scale[block(rbase+d,i)],
// rbase = head·(nope+vh), i∈[0,kvl), d∈[0,nope) — absorb q_nope through kv_b's nope
// rows into latent space. Transposed: output indexed by column i, summed over nope
// rows. One thread per (head,i) — adjacent threads read adjacent bytes.
extern "C" __global__ void mla_absorb_fp8(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl, int block,
    float* __restrict__ qabs) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * kvl) return;
    int head = (int)(idx / kvl);
    int i = (int)(idx % kvl);
    int sc_cols = (kvl + block - 1) / block;
    size_t rbase = (size_t)head * (nope + vh);
    const float* qh_ptr = q + (size_t)head * qh;
    float acc = 0.0f;
    for (int d = 0; d < nope; ++d) {
        size_t row = rbase + d;
        float w = e4m3f(kvb[row * (size_t)kvl + i]) * kvb_scale[(row / block) * sc_cols + i / block];
        acc += qh_ptr[d] * w;
    }
    qabs[(size_t)head * kvl + i] = acc;
}

// ctx[head][j] = Σ_i clat[head][i] · e4m3(kv_b[rbase+nope+j][i]) · scale[block(row,i)],
// j∈[0,vh), i∈[0,kvl) — project the attention-weighted latent through kv_b's value
// rows. Per-head-row GEMV; one WAVE per output row (dot_fp8_wave folds the block
// scale inside the dot, since it varies along i).
extern "C" __global__ void mla_value_fp8(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl, int block,
    float* __restrict__ ctx) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)H * vh) return;
    int head = (int)(r / vh);
    int j = (int)(r % vh);
    int sc_cols = (kvl + block - 1) / block;
    size_t row = (size_t)head * (nope + vh) + nope + j;
    float acc = dot_fp8_wave(clat + (size_t)head * kvl, kvb + row * (size_t)kvl,
                             kvb_scale + (row / block) * sc_cols, kvl, block, lane);
    if (lane == 0) ctx[(size_t)head * vh + j] = acc;
}

extern "C" int rivoli_mla_absorb_fp8(
    const float* q, const unsigned char* kvb, const float* kvb_scale,
    int H, int qh, int nope, int vh, int kvl, int block, float* qabs) {
    if (H <= 0 || qh <= 0 || nope <= 0 || vh <= 0 || kvl <= 0 || block <= 0) return 1001;
    size_t total = (size_t)H * kvl;
    hipGetLastError();
    hipLaunchKernelGGL(mla_absorb_fp8, dim3((unsigned)((total + 255) / 256)), dim3(256), 0, 0,
                       q, kvb, kvb_scale, H, qh, nope, vh, kvl, block, qabs);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_mla_value_fp8(
    const float* clat, const unsigned char* kvb, const float* kvb_scale,
    int H, int nope, int vh, int kvl, int block, float* ctx) {
    if (H <= 0 || nope <= 0 || vh <= 0 || kvl <= 0 || block <= 0) return 1001;
    size_t total = (size_t)H * vh;
    hipGetLastError();
    hipLaunchKernelGGL(mla_value_fp8,
                       dim3((unsigned)((total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0,
                       clat, kvb, kvb_scale, H, nope, vh, kvl, block, ctx);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
