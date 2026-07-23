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
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    int sc_cols = (i_dim + block - 1) / block;
    float acc = dot_fp8_wave(x, packed + o * (size_t)i_dim, scale + (o / block) * sc_cols,
                             i_dim, block, lane);
    if (lane == 0) y[o] = acc;
}

// y[o] = Σ_subvec scale[o,grp]·codebook[idx]·x — VQ-int3, per-group bf16 scale
// applied inside the dot. `codebook` is shared across all rows.
extern "C" __global__ void gemv_vq(
    const float* __restrict__ x, const unsigned char* __restrict__ indices,
    const unsigned short* __restrict__ scales, const float* __restrict__ codebook,
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
extern "C" __global__ void gemv_i8(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const signed char* row = packed + o * (size_t)i_dim;
    float acc = 0.0f;
    for (int i = lane; i < i_dim; i += WAVE) acc += x[i] * (float)row[i];
    acc = wave_sum(acc);
    if (lane == 0) y[o] = acc * scale[o];
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

extern "C" int rivoli_gemv_fp8(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    int block, float* y) {
    if (o_dim <= 0 || i_dim <= 0 || block <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_fp8, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_vq(
    const float* x, const unsigned char* indices, const unsigned short* scales,
    const float* codebook, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_vq, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, indices, scales, codebook, o_dim, i_dim, y);
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
