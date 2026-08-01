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
// rows into latent space. Transposed: output indexed by column i, summed over nope rows.
//
// One thread per (head, i-QUAD): four adjacent columns, four accumulators. A quad's
// columns are contiguous in kv_b, so the weight read is one `global_load_b32` = 128 B per
// wave, exactly one cache line; the old one-column-per-thread form issued
// `global_load_u8` = 32 B/wave, a QUARTER of a line, and that load width was the cost.
// Measured 1.40× — docs/measurement/perf-roadmap.md #4.
//
// BIT-IDENTICAL: each output still sums over `d` in ascending order with the same
// expression, so no quality gate is needed. DO NOT "improve" this to the wave-per-row
// shape `mla_value` uses — that reassociates, and then it does need one.
//
// R TOKEN ROWS through one read of kv_b (14.7 MB/layer, 16% of the attention bytes): the
// `q` rows are `H·qh` apart, the `qabs` rows `H·kvl`, and the four column accumulators
// become `R`×four in registers. Row 0 is bit-identical to R=1 — the `d` loop, the
// expression and the summation order are untouched, only replicated.
template <int R>
__device__ __forceinline__ void mla_absorb_fp8_impl(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl, int block,
    float* __restrict__ qabs) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    int nquad = absorb_nquad(kvl);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * nquad) return;
    int head = (int)(idx / nquad);
    int i0 = (int)(idx % nquad) << 2;
    int sc_cols = (kvl + block - 1) / block;
    // Shift, NOT divide: `row` is a size_t, so `row / block` was a 64-bit unsigned
    // division that LLVM cannot fold to a magic multiply — it emitted an inline
    // Newton-Raphson reciprocal inside the d-loop, worth 72.0 → 36.5 µs on its own.
    // The launcher rejects a non-power-of-two tile. Bit-identical: `row` is unsigned
    // and the column index non-negative, so the index is the same.
    // Evidence in docs/measurement/benchmarks.md, "Read the ISA before you book the device".
    int bsh = blk_shift(block);
    size_t rbase = (size_t)head * (nope + vh);
    const float* qh_ptr = q + (size_t)head * qh;
    const unsigned char* wcol = kvb + rbase * (size_t)kvl + i0;
    float* out = qabs + (size_t)head * kvl + i0;
    // ONE fallback branch, two independent triggers. `kvl & 3` leaves rows unaligned for
    // the dword load (rows are `kvl` bytes apart); `block < 4` puts a quad's columns in
    // different scale tiles. Rejecting them in the launcher instead is not an option: the
    // Vulkan absorb launcher accepts both (src/vk.rs carries NO `kvl % 4` guard on absorb,
    // only on value, deliberately) and the backends must span one argument domain.
    //
    // THE COST IS A SECOND COPY OF THE DEQUANT. This body and the quad body below are two
    // expressions of the same math, and the GLM shape only ever runs the second — so a
    // change to either (lut indexing, scale layout, an fp8 variant) must be made twice,
    // and only the non-GLM oracle shapes in tests/kernel.rs would catch a divergence.
    if ((kvl & 3) || block < 4) {
        // Not unrolled: four copies of the d-loop tripled the kernel's static size
        // (698 → 1078 instructions) for a branch the engine never takes. The measured
        // build carried this pragma and the hot loop below is identical either way.
#pragma unroll 1
        for (int k = 0; k < 4 && i0 + k < kvl; ++k) {
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            for (int d = 0; d < nope; ++d) {
                size_t row = rbase + d;
                float ws = lut[wcol[(size_t)d * kvl + k]]
                           * kvb_scale[(row >> bsh) * sc_cols + ((i0 + k) >> bsh)];
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += qh_ptr[(size_t)r * H * qh + d] * ws;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) out[(size_t)r * H * kvl + k] = acc[r];
        }
        return;
    }
    // One scale load per iteration covers all four columns: past the `if` above the quad
    // sits inside one tile, and its column index does not depend on `d`.
    int sc_col = i0 >> bsh;
    // `w4 += stride4` rather than `w4[d * stride4]` — the latter cost a 64-bit
    // multiply-add in the loop body.
    const unsigned int* w4 = (const unsigned int*)wcol;
    int stride4 = kvl >> 2;
    float a[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) { a[r][0] = a[r][1] = a[r][2] = a[r][3] = 0.0f; }
    for (int d = 0; d < nope; ++d, w4 += stride4) {
        unsigned int p = *w4;
        float s = kvb_scale[((rbase + d) >> bsh) * sc_cols + sc_col];
        // `lut[..] * s` hoisted out of the row loop, exactly as the R=1 form wrote it —
        // the parenthesisation `qd * (lut * s)` is preserved, so row 0 is bit-identical.
        float l0 = lut[(unsigned char)p] * s;
        float l1 = lut[(unsigned char)(p >> 8)] * s;
        float l2 = lut[(unsigned char)(p >> 16)] * s;
        float l3 = lut[(unsigned char)(p >> 24)] * s;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float qd = qh_ptr[(size_t)r * H * qh + d];
            a[r][0] += qd * l0;
            a[r][1] += qd * l1;
            a[r][2] += qd * l2;
            a[r][3] += qd * l3;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float* o = out + (size_t)r * H * kvl;
        o[0] = a[r][0];
        o[1] = a[r][1];
        o[2] = a[r][2];
        o[3] = a[r][3];
    }
}

extern "C" __global__ void mla_absorb_fp8(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl, int block,
    float* __restrict__ qabs) {
    mla_absorb_fp8_impl<1>(q, kvb, kvb_scale, H, qh, nope, vh, kvl, block, qabs);
}

extern "C" __global__ void mla_absorb_fp8_r2(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl, int block,
    float* __restrict__ qabs) {
    mla_absorb_fp8_impl<2>(q, kvb, kvb_scale, H, qh, nope, vh, kvl, block, qabs);
}

// ctx[head][j] = Σ_i clat[head][i] · e4m3(kv_b[rbase+nope+j][i]) · scale[block(row,i)],
// j∈[0,vh), i∈[0,kvl) — project the attention-weighted latent through kv_b's value
// rows. Per-head-row GEMV; one WAVE per output row (dot_fp8_wave folds the block
// scale inside the dot, since it varies along i).
//
// R token rows through one read of kv_b — `clat` rows are `H·kvl` apart, `ctx` rows
// `H·vh`. Row 0 is bit-identical: `dot_fp8_wave_r<1>` IS `dot_fp8_wave`.
template <int R>
__device__ __forceinline__ void mla_value_fp8_impl(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl, int block,
    float* __restrict__ ctx) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)H * vh) return;
    int head = (int)(r / vh);
    int j = (int)(r % vh);
    int sc_cols = (kvl + block - 1) / block;
    size_t row = (size_t)head * (nope + vh) + nope + j;
    // Shift for the same reason as mla_absorb_fp8 — `row` is a size_t, so `/ block` was
    // a 64-bit divide. It is once per THREAD here, not once per iteration, and it was
    // still worth 20% (33.7 → 27.0 us, 249 → 310 GB/s): with H*vh = 16384 threads and a
    // 34 us kernel, one 64-bit division per thread is not a rounding error. An earlier
    // version of this comment called it housekeeping; the bench disagreed.
    float acc[R];
    dot_fp8_wave_r<R>(clat + (size_t)head * kvl, H * kvl, kvb + row * (size_t)kvl,
                      kvb_scale + (row >> blk_shift(block)) * sc_cols, kvl, block, lane, lut,
                      acc);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) ctx[(size_t)r * H * vh + (size_t)head * vh + j] = acc[r];
    }
}

extern "C" __global__ void mla_value_fp8(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl, int block,
    float* __restrict__ ctx) {
    mla_value_fp8_impl<1>(clat, kvb, kvb_scale, H, nope, vh, kvl, block, ctx);
}

extern "C" __global__ void mla_value_fp8_r2(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl, int block,
    float* __restrict__ ctx) {
    mla_value_fp8_impl<2>(clat, kvb, kvb_scale, H, nope, vh, kvl, block, ctx);
}

extern "C" int rivoli_mla_absorb_fp8(
    const float* q, const unsigned char* kvb, const float* kvb_scale,
    int H, int qh, int nope, int vh, int kvl, int block, int nrow, float* qabs) {
    if (H <= 0 || qh <= 0 || nope <= 0 || vh <= 0 || kvl <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    // The row count is a TEMPLATE parameter (accumulators live in registers), so only the
    // instantiated widths exist. Same contract as gemv_fp8 / moe_expert_range.
    if (nrow != 1 && nrow != 2) return 1004;
    size_t total = (size_t)H * (size_t)absorb_nquad(kvl); // one thread per (head, i-quad)
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? mla_absorb_fp8_r2 : mla_absorb_fp8,
                       dim3((unsigned)((total + 255) / 256)), dim3(256), 0, 0,
                       q, kvb, kvb_scale, H, qh, nope, vh, kvl, block, qabs);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_mla_value_fp8(
    const float* clat, const unsigned char* kvb, const float* kvb_scale,
    int H, int nope, int vh, int kvl, int block, int nrow, float* ctx) {
    if (H <= 0 || nope <= 0 || vh <= 0 || kvl <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    if (nrow != 1 && nrow != 2) return 1004;  // see rivoli_mla_absorb_fp8
    // `dot_fp8_wave` reads weights as 32-bit words and rows are `kvl` bytes apart, so a
    // ragged kvl leaves 3 rows in 4 misaligned for the dword load. The Vulkan twin has
    // rejected this since it was written (src/vk.rs::launch_mla_value_fp8) and this
    // launcher did not — a parity gap found when the absorb restructure added a kvl=37
    // oracle shape, which is legal for absorb (it reads single bytes) and drove
    // mla_value straight into the misaligned load. Absorb deliberately has NO such
    // guard; the asymmetry is the same one src/vk.rs documents.
    if (kvl & 3) return 1002;
    size_t total = (size_t)H * vh;
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? mla_value_fp8_r2 : mla_value_fp8,
                       dim3((unsigned)((total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0,
                       clat, kvb, kvb_scale, H, nope, vh, kvl, block, ctx);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
