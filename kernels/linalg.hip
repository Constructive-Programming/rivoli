// Per-layer linear-algebra primitives — the projections the forward pass needs
// on-device (batch-1 GEMVs), int4-free. One WAVE cooperates on one output row:
// adjacent lanes read adjacent columns (coalesced), then a fixed-order wave-shuffle
// reduces. The reduction order differs from the scalar oracle's ascending sum by
// f32 rounding (within the tests' tolerance) but is SCHEDULE-INDEPENDENT, so greedy
// decode stays deterministic. All pointers device-resident; launchers only launch
// (caller joins via rivoli_device_sync). Return codes: 0 ok, POSITIVE arg guard,
// NEGATIVE -(hipError_t).

#include "common.hpp"

// y[o] = Σ_i x[i]·e4m3(packed[o·i_dim+i])·scale[(o/block)·sc_cols + i/block] — fp8
// block-scaled (attention/dense projections; matches quant.rs matvec_fp8).
extern "C" __global__ void gemv_fp8(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    int sc_cols = (i_dim + block - 1) / block;
    float acc = dot_fp8_wave(x, packed + o * (size_t)i_dim,
                             scale + (o >> blk_shift(block)) * sc_cols, i_dim, block, lane, lut);
    if (lane == 0) y[o] = acc;
}

// Split-K fp8 GEMV for LONG reductions: ONE block per output row, its ROWS_PER_BLOCK
// waves each reduce a strided 1/K share of i_dim, combined in LDS in fixed order
// (deterministic). o_proj (o_dim=6144, i_dim=16384) has too few rows to fill the
// machine under wave-per-row (768 blocks, each wave grinding a 16384-long chain);
// one-row-per-block gives K× the blocks so DRAM saturates — measured 128→~250 GB/s.
// Same LUT + block-scaled e4m3 dot as gemv_fp8; the whole block strides coalesced.
extern "C" __global__ void gemv_fp8_splitk(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    __shared__ float lut[256];
    __shared__ float part[ROWS_PER_BLOCK]; // K = ROWS_PER_BLOCK wave partials
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
    // bought in mla_value_fp8 — why, and why it shipped anyway, in docs/PERF.md #2.
    const float* scalerow = scale + (o >> blk_shift(block)) * sc_cols;
    const int STRIDE = ROWS_PER_BLOCK * WAVE; // all 256 threads stride coalesced (split-K)
    // Same strided fp8 MAC as dot_fp8_wave, but split-K across every wave in the block.
    float acc = fp8_dot_strided(x, wrow, scalerow, i_dim, block, lut, threadIdx.x, STRIDE);
    acc = wave_sum(acc); // per-wave partial → its lane 0
    if (lane == 0) part[wave] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < ROWS_PER_BLOCK; ++w) s += part[w]; // fixed order
        y[o] = s;
    }
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
    float acc = dot_vq_wave(x, indices + o * rb, scales + o * ng, codebook, i_dim, lane);
    if (lane == 0) y[o] = acc;
}

// y[o] = Σ_i x[i]·w[o·i_dim+i] — plain f32 GEMV (the MoE router gate).
extern "C" __global__ void gemv_f32(
    const float* __restrict__ x, const float* __restrict__ w, int o_dim, int i_dim,
    float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const float* row = w + o * (size_t)i_dim;
    float acc = 0.0f;
    for (int i = lane; i < i_dim; i += WAVE) acc += x[i] * row[i];
    acc = wave_sum(acc);
    if (lane == 0) y[o] = acc;
}

// y[o] = scale[o]·Σ_i x[i]·(int8)packed[o·i_dim+i] — per-row int8 GEMV (lm_head to
// logits; matches quant.rs matvec_i8). One WAVE per output row, coalesced columns.
//
// FOUR columns per lane, not one. A scalar `row[i]` is a global_load_i8: 32 B/wave
// against a 128 B cache line, so the kernel asked DRAM for a quarter line at a time and
// measured 117 GB/s of a 256 GB/s part. Four contiguous int8 are one dword — a full line
// per wave — paired with one float4 of x. Same mechanism, and the same fix, as
// mla_absorb's i-quad (docs/PERF.md #4 and #5).
//
// NOT bit-identical to the scalar form: each lane now owns a contiguous quad instead of
// a stride-WAVE comb, which reassociates the f32 sum. lm_head feeds argmax directly, so
// this took the perplexity gate rather than a fingerprint.
extern "C" __global__ void gemv_i8(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const signed char* row = packed + o * (size_t)i_dim;
    float acc = 0.0f;
    int base = 0;
    // The quad path needs `row` dword-aligned and `x` float4-aligned. `packed` is a
    // device allocation, so the real condition is i_dim % 4 keeping every row aligned;
    // both are uniform per block and anything failing them falls to the scalar tail.
    if ((((size_t)row) & 3u) == 0 && (((size_t)x) & 15u) == 0) {
        for (; base + WAVE * 4 <= i_dim; base += WAVE * 4) {
            int col = base + lane * 4;
            int w = *(const int*)(row + col); // 4 int8 = one dword = 128 B/wave
            float4 xv = *(const float4*)(x + col);
            acc += xv.x * (float)(signed char)(w) + xv.y * (float)(signed char)(w >> 8) +
                   xv.z * (float)(signed char)(w >> 16) + xv.w * (float)(signed char)(w >> 24);
        }
    }
    for (int i = base + lane; i < i_dim; i += WAVE) acc += x[i] * (float)row[i];
    acc = wave_sum(acc);
    if (lane == 0) y[o] = acc * scale[o];
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
    float dv = dot_i4_wave(x, packed + o * rb, scale + o * ng, i_dim, lane);
    if (lane == 0) y[o] = dv;
}

// h[i] = silu(g[i])·u[i], silu(x)=x/(1+e^-x) — the SwiGLU combine for the dense
// fp8 MLP (moe.hip does this internally for VQ experts; the dense path is three
// gemv_fp8 + this). Safe in place (h may alias g): each thread reads g[i]/u[i] then
// writes h[i] independently.
extern "C" __global__ void swiglu(
    const float* __restrict__ g, const float* __restrict__ u, int n, float* __restrict__ h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i];
    h[i] = (gv / (1.0f + expf(-gv))) * u[i];
}

// y[i] = x[i]·(1/sqrt(mean(x²)+eps))·w[i] — RMSNorm, matching math.rs. One block
// reduces Σx² in LDS (tree order), then every thread scales.
extern "C" __global__ void rmsnorm(
    const float* __restrict__ x, const float* __restrict__ w, int n, float eps,
    float* __restrict__ y) {
    __shared__ float red[256];
    int t = threadIdx.x;
    float local = 0.0f;
    for (int i = t; i < n; i += blockDim.x) local += x[i] * x[i];
    red[t] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / (float)n + eps);
    for (int i = t; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

// Interleaved RoPE (matching attn.rs) on `count` segments of `seg` elements:
// segment s at base[s*stride..+seg]; pair (2j,2j+1) rotates by pos·theta^(-2j/seg)
// into (j, half+j). Double angles (long-context arg reduction); one block/segment,
// seg/2 threads; all pairs read before any write (barrier) → race-free in place.
extern "C" __global__ void rope_interleave(
    float* __restrict__ base, int stride, int seg, int pos, double theta) {
    int s = blockIdx.x;
    int j = threadIdx.x;
    int half = seg / 2;
    float* v = base + (size_t)s * stride;
    float a = v[2 * j], b = v[2 * j + 1];
    __syncthreads();
    double inv = pow(theta, -2.0 * (double)j / (double)seg);
    double ang = (double)pos * inv;
    float cs = (float)cos(ang), sn = (float)sin(ang);
    v[j] = a * cs - b * sn;
    v[half + j] = b * cs + a * sn;
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
    int block, float* y) {
    if (o_dim <= 0 || i_dim <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    hipGetLastError();
    hipLaunchKernelGGL(gemv_fp8_splitk, dim3(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_fp8(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    int block, float* y) {
    // NO `i_dim % 4` guard, unlike the Vulkan twin and unlike rivoli_mla_value_fp8's
    // `kvl % 4` — a known gap, conditional on block >= 4. See common.hpp fp8_dot_strided.
    if (o_dim <= 0 || i_dim <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    if (i_dim >= GEMV_SPLITK_MIN_IDIM)
        return rivoli_gemv_fp8_splitk(x, packed, scale, o_dim, i_dim, block, y);
    hipGetLastError();
    hipLaunchKernelGGL(gemv_fp8, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
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
    const float* x, const signed char* packed, const float* scale, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_i8, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_f32(const float* x, const float* w, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_f32, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, w, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_swiglu(const float* g, const float* u, int n, float* h) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(swiglu, dim3((n + 255) / 256), dim3(256), 0, 0, g, u, n, h);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rmsnorm(const float* x, const float* w, int n, float eps, float* y) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rmsnorm, dim3(1), dim3(256), 0, 0, x, w, n, eps, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rope(float* base, int count, int stride, int seg, int pos, double theta) {
    if (count <= 0 || seg <= 0 || (seg & 1) || (seg / 2) > 1024) return 1001;
    if (count > 1 && stride < seg) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rope_interleave, dim3(count), dim3(seg / 2), 0, 0, base, stride, seg, pos, theta);
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
