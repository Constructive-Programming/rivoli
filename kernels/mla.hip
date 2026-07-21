// M3 (4/4) MLA absorb + value kernels — the two int4 ops through kv_b that the
// generic GEMV can't express, head-batched into one launch each. All pointers
// device-resident; launch-only. Return codes: 0 ok, POSITIVE arg guard,
// NEGATIVE -hipError.
//
// kv_b is int4 [H*(nope+vh) rows, kvl cols]; head `h` owns rows
// [h*(nope+vh) .. +(nope+vh)) — the first `nope` are the absorb rows, the next
// `vh` the value rows. rb = ⌈kvl/2⌉ bytes/row.
//
// `mla_value` is a plain GEMV over a per-head row range, so it uses the shared
// wave-per-row `dot_i4_wave` (D3 + dword loads) like every other GEMV here; it
// was the last kernel still in the pre-D3 one-thread-per-row shape, where a
// wave's 32 lanes walked 32 DIFFERENT rows ⌈kvl/2⌉ B apart and every load
// instruction touched 32 separate cache lines. Its reduction order therefore
// differs from the scalar oracle (attn.rs matvec_i4_rows) by f32 rounding,
// within the kernel tests' int4 tolerance, and stays schedule-independent.
//
// `mla_absorb` keeps the thread-per-output-column form: it contracts over the
// `nope` ROWS into `kvl` columns, so adjacent threads already read adjacent
// bytes, and vectorising it measured only 1.10x (docs/probes/mla_proj_probe.cpp)
// — not worth a second load path.

#include "common.hpp"

// column-i nibble of a packed int4 row, offset −8 (matches quant.rs).
__device__ __forceinline__ int nib_i4(const unsigned char* row, int i) {
    unsigned char b = row[i >> 1];
    int n = (i & 1) ? (b >> 4) : (b & 0x0F);
    return n - 8;
}

// qabs[head][i] = Σ_d q[head·qh + d] · kv_b[rbase+d][i] · scale[rbase+d],
// rbase = head·(nope+vh), i∈[0,kvl), d∈[0,nope) — absorb q_nope through kv_b's
// nope rows into latent space (attn.rs addrow loop). Transposed: output indexed
// by column i, summed over the nope rows. One thread per (head,i).
extern "C" __global__ void mla_absorb(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb_packed,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl,
    float* __restrict__ qabs) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * kvl) return;
    int head = idx / kvl;
    int i = idx % kvl;
    size_t rb = (size_t)(kvl + 1) / 2;
    size_t rbase = (size_t)head * (nope + vh);  // 64-bit so the row index can't wrap
    const float* qh_ptr = q + (size_t)head * qh;
    float acc = 0.0f;
    for (int d = 0; d < nope; ++d) {
        const unsigned char* row = kvb_packed + (rbase + d) * rb;
        acc += qh_ptr[d] * (float)nib_i4(row, i) * kvb_scale[rbase + d];
    }
    qabs[(size_t)head * kvl + i] = acc;
}

// ctx[head][j] = scale[rbase+nope+j] · Σ_i clat[head][i] · kv_b[rbase+nope+j][i],
// j∈[0,vh), i∈[0,kvl) — project the attention-weighted latent through kv_b's
// value rows (attn.rs matvec_i4_rows). Standard GEMV over a per-head row range;
// ONE WAVE per output row, coalesced + shuffle-reduce.
extern "C" __global__ void mla_value(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb_packed,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl,
    float* __restrict__ ctx) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)H * vh) return;  // wave-uniform (all lanes share r)
    int head = (int)(r / vh);
    int j = (int)(r % vh);
    size_t rb = (size_t)(kvl + 1) / 2;
    size_t row_idx = (size_t)head * (nope + vh) + nope + j;  // 64-bit row index
    float acc = dot_i4_wave(clat + (size_t)head * kvl, kvb_packed + row_idx * rb, kvl, lane);
    if (lane == 0) ctx[(size_t)head * vh + j] = acc * kvb_scale[row_idx];
}

extern "C" int rivoli_mla_absorb(
    const float* q, const unsigned char* kvb_packed, const float* kvb_scale,
    int H, int qh, int nope, int vh, int kvl, float* qabs) {
    if (H <= 0 || qh <= 0 || nope <= 0 || vh <= 0 || kvl <= 0) return 1001;
    size_t total = (size_t)H * kvl;
    hipGetLastError();
    hipLaunchKernelGGL(mla_absorb, dim3((unsigned)((total + 255) / 256)), dim3(256), 0, 0,
                       q, kvb_packed, kvb_scale, H, qh, nope, vh, kvl, qabs);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_mla_value(
    const float* clat, const unsigned char* kvb_packed, const float* kvb_scale,
    int H, int nope, int vh, int kvl, float* ctx) {
    if (H <= 0 || nope <= 0 || vh <= 0 || kvl <= 0) return 1001;
    size_t total = (size_t)H * vh;  // one WAVE per output row
    hipGetLastError();
    hipLaunchKernelGGL(mla_value,
                       dim3((unsigned)((total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0,
                       clat, kvb_packed, kvb_scale, H, nope, vh, kvl, ctx);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
