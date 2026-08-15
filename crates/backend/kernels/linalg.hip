// Per-layer linear-algebra primitives — the projections the forward pass needs
// on-device (batch-1 GEMVs), int4-free. One WAVE cooperates on one output row:
// adjacent lanes read adjacent columns (coalesced), then a fixed-order wave-shuffle
// reduces. The reduction order differs from the scalar oracle's ascending sum by
// f32 rounding (within the tests' tolerance) but is SCHEDULE-INDEPENDENT, so greedy
// decode stays deterministic. All pointers device-resident; launchers only launch
// (caller joins via rivoli_device_sync). Return codes: 0 ok, POSITIVE arg guard,
// NEGATIVE -(hipError_t).
//
// **SPLIT 2026-08-15**: the per-token activation transforms — the norms, RoPE, the SwiGLU
// combines, `act_quant_f8` and the mHC residual mixing — moved to `activation.hip` whole,
// comments included, because this file was 1125 lines against an 800-line cap. The cut is
// by what a kernel READS: everything left here contracts a token against a full weight
// MATRIX, so its cost is the weight's bytes (P5); everything that moved transforms the
// activation stream and costs what the activation costs. Every symbol and every launcher
// signature is unchanged — the split is a move, not an edit — and `build.rs`'s KERNELS
// array is what makes the new unit compile at all: a `.hip` absent from that list is
// silently never built, and the archive then fails to link.

#include "common.hpp"

// y[o] = Σ_i x[i]·e4m3(packed[o·i_dim+i])·scale[(o/block)·sc_cols + i/block] — fp8
// block-scaled (attention/dense projections; matches quant.rs matvec_fp8).
// `R` token rows through one read of the weight row. `x` rows are `i_dim` apart and `y`
// rows `o_dim` apart — both already arguments, so batching costs no extra parameter and at
// R=1 every index is the single-row one.
template <int R>
__device__ __forceinline__ void gemv_fp8_impl(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    int sc_cols = (i_dim + block - 1) / block;
    // `x` rows are `i_dim` apart; the weight row and the block-scale row it indexes travel
    // as one `Fp8Row` (dot_fp8.hpp) — see rowview.hpp for what that costs in aliasing.
    Acc<R> acc = fp8_dot_strided_r<R>(
        RowsView{x, i_dim},
        Fp8Row{packed + o * (size_t)i_dim, scale + (o >> blk_shift(block)) * sc_cols, lut, i_dim,
               block},
        LaneSpan{lane, WAVE});
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // Uniform across the wave: R is a compile-time constant.
        float s = wave_sum(acc.v[r]);
        if (lane == 0) y[(size_t)r * o_dim + o] = s;
    }
}

extern "C" __global__ void gemv_fp8(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_impl<1>(x, packed, scale, o_dim, i_dim, block, y);
}

extern "C" __global__ void gemv_fp8_r2(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_impl<2>(x, packed, scale, o_dim, i_dim, block, y);
}

// Split-K fp8 GEMV for LONG reductions: ONE block per output row, its ROWS_PER_BLOCK
// waves each reduce a strided 1/K share of i_dim, combined in LDS in fixed order
// (deterministic). o_proj (o_dim=6144, i_dim=16384) has too few rows to fill the
// machine under wave-per-row (768 blocks, each wave grinding a 16384-long chain);
// one-row-per-block gives K× the blocks so DRAM saturates — measured 128→~250 GB/s.
// Same LUT + block-scaled e4m3 dot as gemv_fp8; the whole block strides coalesced.
template <int R>
__device__ __forceinline__ void gemv_fp8_splitk_impl(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    __shared__ float lut[256];
    __shared__ float part[R][ROWS_PER_BLOCK]; // K = ROWS_PER_BLOCK wave partials, per row
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    size_t o = blockIdx.x; // one output row per block
    if (o >= (size_t)o_dim) return;
    int wave = threadIdx.x / WAVE;
    int lane = threadIdx.x & (WAVE - 1);
    int sc_cols = (i_dim + block - 1) / block;
    const unsigned char* wrow = packed + o * (size_t)i_dim;
    // Shift, NOT divide: `o` is a size_t, so `o / block` was a runtime division once per
    // thread (~55 instructions). Same index, the launcher rejecting a non-power-of-two
    // tile. Measured −0.9%, INSIDE the noise band and not the 20% the identical fix
    // bought in mla_value_fp8 — why, and why it shipped anyway, in docs/measurement/perf-roadmap.md #2.
    const float* scalerow = scale + (o >> blk_shift(block)) * sc_cols;
    const int STRIDE = ROWS_PER_BLOCK * WAVE; // all 256 threads stride coalesced (split-K)
    // Same strided fp8 MAC as dot_fp8_wave, but split-K across every wave in the block —
    // which is the whole reason `LaneSpan` exists as a pair: wave-per-row passes
    // `{lane, WAVE}` above, this passes all 256 threads striding coalesced.
    Acc<R> acc = fp8_dot_strided_r<R>(RowsView{x, i_dim}, Fp8Row{wrow, scalerow, lut, i_dim, block},
                                      LaneSpan{(int)threadIdx.x, STRIDE});
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = wave_sum(acc.v[r]); // per-wave partial → its lane 0
        if (lane == 0) part[r][wave] = s;
    }
    __syncthreads();
    // One thread per token row, each folding its own K partials in the SAME fixed order as
    // the single-row kernel — so the split-K combine stays deterministic per row.
    if (threadIdx.x < R) {
        float s = 0.0f;
        for (int w = 0; w < ROWS_PER_BLOCK; ++w) s += part[threadIdx.x][w];
        y[(size_t)threadIdx.x * o_dim + o] = s;
    }
}

extern "C" __global__ void gemv_fp8_splitk(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_splitk_impl<1>(x, packed, scale, o_dim, i_dim, block, y);
}

extern "C" __global__ void gemv_fp8_splitk_r2(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_splitk_impl<2>(x, packed, scale, o_dim, i_dim, block, y);
}

// y[o] = Σ_subvec scale[o,grp]·codebook[idx]·x — VQ-int3, per-group bf16 scale
// applied inside the dot. `codebook` is shared across all rows.
extern "C" __global__ void gemv_vq(
    const float* __restrict__ x, const unsigned char* __restrict__ indices,
    const unsigned short* __restrict__ scales, const __half* __restrict__ codebook,
    int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    size_t rb = (size_t)(i_dim / VQ_DIM) * VQ_INDEX_BITS / 8; // vq_row_bytes
    size_t ng = (size_t)i_dim / VQ_GROUP;                     // vq_groups
    float acc = dot_vq_wave(x, VqRow{indices + o * rb, scales + o * ng, codebook, i_dim}, lane);
    if (lane == 0) y[o] = acc;
}

// y[r][o] = Σ_i x[r][i]·w[o·i_dim+i] — plain f32 GEMV (the MoE router gate), R token
// rows through one read of `w`. The gate is 6.3 MB per MoE layer, ~470 MB a pass, so a
// speculative verify wants it read once. `x` rows are `i_dim` apart, `y` rows `o_dim`.
// Row 0 is bit-identical: same `i` order, same expression, just replicated.
template <int R>
__device__ __forceinline__ void gemv_f32_impl(
    const float* __restrict__ x, const float* __restrict__ w, int o_dim, int i_dim,
    float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const float* row = w + o * (size_t)i_dim;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int i = lane; i < i_dim; i += WAVE) {
        float wi = row[i];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += x[(size_t)r * i_dim + i] * wi;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = wave_sum(acc[r]);
        if (lane == 0) y[(size_t)r * o_dim + o] = s;
    }
}

extern "C" __global__ void gemv_f32(
    const float* __restrict__ x, const float* __restrict__ w, int o_dim, int i_dim,
    float* __restrict__ y) {
    gemv_f32_impl<1>(x, w, o_dim, i_dim, y);
}

extern "C" __global__ void gemv_f32_r2(
    const float* __restrict__ x, const float* __restrict__ w, int o_dim, int i_dim,
    float* __restrict__ y) {
    gemv_f32_impl<2>(x, w, o_dim, i_dim, y);
}

// y[o] = scale[o]·Σ_i x[i]·(int8)packed[o·i_dim+i] — per-row int8 GEMV (lm_head to
// logits; matches quant.rs matvec_i8). One WAVE per output row, coalesced columns.
//
// FOUR columns per lane, not one. A scalar `row[i]` is a global_load_i8: 32 B/wave
// against a 128 B cache line, so the kernel asked DRAM for a quarter line at a time and
// measured 117 GB/s of a 256 GB/s part. Four contiguous int8 are one dword — a full line
// per wave — paired with one float4 of x. Same mechanism, and the same fix, as
// mla_absorb's i-quad (docs/measurement/perf-roadmap.md #4 and #5).
//
// NOT bit-identical to the scalar form: each lane now owns a contiguous quad instead of
// a stride-WAVE comb, which reassociates the f32 sum. lm_head feeds argmax directly, so
// this took the perplexity gate rather than a fingerprint.
//
// R token rows through one read of the 952 MB lm_head. Row 0 is bit-identical: the quad
// decode, the summation order and the final `acc * scale[o]` are unchanged, only
// replicated. `x` rows are `i_dim` apart, `y` rows `o_dim` (= vocab).
template <int R>
__device__ __forceinline__ void gemv_i8_impl(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const signed char* row = packed + o * (size_t)i_dim;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    int base = 0;
    // The quad path needs `row` dword-aligned and `x` float4-aligned. `packed` is a
    // device allocation, so the real condition is i_dim % 4 keeping every row aligned;
    // both are uniform per block and anything failing them falls to the scalar tail.
    // With R > 1 there is one more: `x + r·i_dim` must stay float4-aligned, i.e. i_dim
    // divisible by 4. Gated on `R == 1` (a compile-time constant) rather than added
    // unconditionally, so the single-row path keeps taking the quad route on a ragged
    // i_dim exactly as it did before — the oracle tests run such shapes.
    if ((((size_t)row) & 3u) == 0 && (((size_t)x) & 15u) == 0 && (R == 1 || (i_dim & 3) == 0)) {
        for (; base + WAVE * 4 <= i_dim; base += WAVE * 4) {
            int col = base + lane * 4;
            int w = *(const int*)(row + col); // 4 int8 = one dword = 128 B/wave
            float w0 = (float)(signed char)(w), w1 = (float)(signed char)(w >> 8);
            float w2 = (float)(signed char)(w >> 16), w3 = (float)(signed char)(w >> 24);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float4 xv = *(const float4*)(x + (size_t)r * i_dim + col);
                acc[r] += xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
            }
        }
    }
    for (int i = base + lane; i < i_dim; i += WAVE) {
        float wi = (float)row[i];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += x[(size_t)r * i_dim + i] * wi;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = wave_sum(acc[r]);
        if (lane == 0) y[(size_t)r * o_dim + o] = s * scale[o];
    }
}

extern "C" __global__ void gemv_i8(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    gemv_i8_impl<1>(x, packed, scale, o_dim, i_dim, y);
}

extern "C" __global__ void gemv_i8_r2(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    gemv_i8_impl<2>(x, packed, scale, o_dim, i_dim, y);
}

// y[o] = Σ_i x[i]·(nibble(o,i)-8)·scale[o, i/I4_GROUP] — group-scaled int4,
// wave-per-row (the exact dot_i4_wave the MoE kernels use). Exists so the dot's decode
// throughput can be measured in isolation against gemv_vq / gemv_fp8 at the MoE
// projection dims, free of the routing/miss-count confound the decode bench has.
extern "C" __global__ void gemv_i4(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    size_t rb = (size_t)((i_dim + 1) / 2);
    size_t ng = (size_t)((i_dim + I4_GROUP - 1) / I4_GROUP);
    float dv = dot_i4_wave(x, I4Row{packed + o * rb, scale + o * ng, i_dim}, lane);
    if (lane == 0) y[o] = dv;
}

// VQ encode: idx[i] = argmin_k (‖c_k‖² − 2·sub_i·c_k), tie → lowest k. Matches
// quant.rs::quant_vq's nearest exactly (FP contraction OFF, so the dot rounds like
// the CPU — a fused FMA would flip near-tied argmins on the learned codebook). The
// offline converter's argmin accelerator.
extern "C" __global__ void vq_encode(
    const float* __restrict__ sub, const float* __restrict__ codebook,
    const float* __restrict__ cbnorm, int n, unsigned short* __restrict__ idx) {
#pragma clang fp contract(off)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sv[VQ_DIM];
#pragma unroll
    for (int d = 0; d < VQ_DIM; ++d) sv[d] = sub[(size_t)i * VQ_DIM + d];
    float best = 3.4e38f;
    int bk = 0;
    for (int k = 0; k < VQ_K; ++k) {
        const float* c = codebook + (size_t)k * VQ_DIM;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < VQ_DIM; ++d) dot += sv[d] * c[d];
        float dist = cbnorm[k] - 2.0f * dot;
        if (dist < best) {
            best = dist;
            bk = k;
        }
    }
    idx[i] = (unsigned short)bk;
}

// ── launchers (grid = ceil(o_dim / ROWS_PER_BLOCK) blocks of 256 threads) ──────

#define ROW_GRID(o) dim3(((o) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)
#define ROW_BLOCK dim3(ROWS_PER_BLOCK * WAVE)

// Wave-per-row saturates DRAM once o_dim is large enough to fill the machine; below
// that (o_proj: o_dim=6144, i_dim=16384) too few blocks launch and DRAM starves, so
// switch to split-K (one block/row). Threshold on i_dim: the plain kernel measures
// near-peak (255-283 GB/s) for i_dim ≤ 2048 and drops (~128) for i_dim ≥ 8192.
#define GEMV_SPLITK_MIN_IDIM 4096

extern "C" int rivoli_gemv_fp8_splitk(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    int block, int nrow, float* y) {
    if (o_dim <= 0 || i_dim <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    if (nrow != 1 && nrow != 2) return 1004;
    hipGetLastError();
    if (nrow == 1) {
        hipLaunchKernelGGL(gemv_fp8_splitk, dim3(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    } else {
        hipLaunchKernelGGL(gemv_fp8_splitk_r2, dim3(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    }
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// `nrow` token rows (1 or 2) through ONE read of the fp8 weights: `x[r·i_dim + i]` →
// `y[r·o_dim + o]`. The grid is unchanged — the row axis lives inside each wave, not
// beside it — so this is the same launch doing nrow times the arithmetic on the same bytes.
extern "C" int rivoli_gemv_fp8(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    int block, int nrow, float* y) {
    // NO `i_dim % 4` guard, unlike the Vulkan twin and unlike rivoli_mla_value_fp8's
    // `kvl % 4` — a known gap, conditional on block >= 4. See dot_fp8.hpp fp8_dot_strided.
    if (o_dim <= 0 || i_dim <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    if (nrow != 1 && nrow != 2) return 1004;
    if (i_dim >= GEMV_SPLITK_MIN_IDIM)
        return rivoli_gemv_fp8_splitk(x, packed, scale, o_dim, i_dim, block, nrow, y);
    hipGetLastError();
    if (nrow == 1) {
        hipLaunchKernelGGL(gemv_fp8, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    } else {
        hipLaunchKernelGGL(gemv_fp8_r2, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    }
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_vq(
    const float* x, const unsigned char* indices, const unsigned short* scales,
    const __half* codebook, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_vq, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, indices, scales, codebook, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_i4(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_i4, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_i8(
    const float* x, const signed char* packed, const float* scale, int o_dim, int i_dim,
    int nrow, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    if (nrow != 1 && nrow != 2) return 1004;  // R is a template parameter — see gemv_fp8
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? gemv_i8_r2 : gemv_i8, ROW_GRID(o_dim), ROW_BLOCK, 0, 0,
                       x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// Takes a STREAM, unlike the other GEMVs in this file, and it is the odd one out because of
// DeepSeek-V4 rather than GLM: this is the router-gate GEMV (`Gate.forward`'s
// `linear(x.float(), weight.float())`), so on the V4 layer path it sits between two launchers
// that take one. `nullptr` is the null stream, so every existing caller is unchanged. The full
// argument, and why this is not a V4-only twin launcher, is on `hip.rs::launch_gemv_f32` —
// stated once, where a caller reads it.
extern "C" int rivoli_gemv_f32(const float* x, const float* w, int o_dim, int i_dim, int nrow,
                               float* y, void* stream) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    if (nrow != 1 && nrow != 2) return 1004;  // R is a template parameter — see gemv_fp8
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? gemv_f32_r2 : gemv_f32, ROW_GRID(o_dim), ROW_BLOCK, 0,
                       (hipStream_t)stream, x, w, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_vq_encode(
    const float* sub, const float* codebook, const float* cbnorm, int n, unsigned short* idx) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(vq_encode, dim3((n + 255) / 256), dim3(256), 0, 0, sub, codebook, cbnorm, n, idx);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// One join point per token: block until all launched kernels retire.
extern "C" int rivoli_device_sync() {
    hipError_t e = hipDeviceSynchronize();
    return e == hipSuccess ? 0 : -(int)e;
}

// Synchronous device-to-device copy — the routed arena's slot relocation (compaction).
// Blocks so the moved expert's bytes are in place before the layer's MoE compute (or a
// later cold read) touches the new location. `dst`/`src` are distinct slots, so the
// regions never overlap.
extern "C" int rivoli_memcpy_dtod(void* dst, const void* src, size_t bytes) {
    hipError_t e = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
    return e == hipSuccess ? 0 : -(int)e;
}

// STREAM-ORDERED device-to-device copy, for a pipeline whose kernels are on a stream.
//
// A separate entry point rather than a `stream` parameter on `rivoli_memcpy_dtod`, because
// the two are not the same operation with a knob — the one above BLOCKS THE HOST, and the
// arena relocation it serves needs exactly that. `memory/routed.rs` moves an expert's slot
// while later reads may still resolve to the old address, and this repo has a measured
// defect from a read outliving its layer and getting its slot copied out from under it
// (9 of 8452 reads at `--max-mem 30`, non-deterministic output; pins do not stop
// compaction). Making that path stream-ordered would replace a host barrier with a
// per-stream one and reopen it. So: two names, and the blocking one keeps its callers.
//
// Why this exists at all — DeepSeek-V4's attention block interleaves six device-to-device
// copies with sixteen kernel launches (`attn.rs::v4::attention`: `x -> xq`, `qr -> qrq`, and
// four KV ring writes, of which one or two run depending on the phase). Once those kernels take a stream, a blocking `hipMemcpy` between two of them
// does NOT wait on the stream: rivoli's streams are `hipStreamNonBlocking` (`async.hip`), so
// the null stream has no implicit ordering against them. The copy would read `s.qr` before
// `rmsnorm_batch` wrote it. Found 2026-08-05 while converting that block's six launchers —
// converting six of the seven operations would have introduced the very race the earlier
// decline of the partial fix predicted.
extern "C" int rivoli_memcpy_dtod_async(void* dst, const void* src, size_t bytes,
                                        void* stream) {
    hipError_t e = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice,
                                  (hipStream_t)stream);
    return e == hipSuccess ? 0 : -(int)e;
}
