// M3 (4/4) per-layer linear-algebra primitives — the projections the forward
// pass needs on-device alongside the fused MoE/attention kernels, so the whole
// token stays resident (no host round-trip = no per-layer join). Each is a
// batch-1 GEMV. All pointers are device-resident; the launchers only launch
// (caller joins via rivoli_device_sync). Return codes: 0 ok, POSITIVE arg guard,
// NEGATIVE -hipError.
//
// D3 COALESCED FORM (was: one thread per row, summing the whole row alone —
// adjacent threads strode one weight ROW apart, so every global load was a
// separate cache line; measured ~38% of decode). Now one WAVE cooperates on one
// output row: adjacent lanes read adjacent columns (coalesced cache lines), then
// a fixed-order wave-shuffle reduces the partials. The reduction order differs
// from the scalar oracle's ascending sum by f32 rounding (well within the test's
// 1e-3 dequant tolerance) but is SCHEDULE-INDEPENDENT, so greedy decode stays
// deterministic.

#include "common.hpp"

// y[o] = scale[o] · Σ_i x[i]·(nibble(row_o,i) - 8)   — int4, per-row scale.
extern "C" __global__ void gemv_i4(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    // size_t row index so a huge grid can't wrap to a negative int (the guard is
    // never approached at GLM dims, but keeps the address arithmetic honest).
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    size_t rb = (size_t)(i_dim + 1) / 2;
    float acc = dot_i4_wave(x, packed + o * rb, i_dim, lane);
    if (lane == 0) y[o] = acc * scale[o];
}

// y[o] = Σ_subvec scale[o,grp]·codebook[idx]·x  — VQ-int3, per-group bf16 scale
// applied inside the dot (no per-row post-scale). `codebook` is shared across all
// rows/experts; `indices`/`scales` are this matrix's packed indices + bf16 groups.
extern "C" __global__ void gemv_vq(
    const float* __restrict__ x, const unsigned char* __restrict__ indices,
    const unsigned short* __restrict__ scales, const float* __restrict__ codebook,
    int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    size_t rb = (size_t)(i_dim / VQ_DIM) * VQ_INDEX_BITS / 8;  // vq_row_bytes
    size_t ng = (size_t)i_dim / VQ_GROUP;                      // vq_groups
    float acc = dot_vq_wave(x, indices + o * rb, scales + o * ng, codebook, i_dim, lane);
    if (lane == 0) y[o] = acc;
}

// y[o] = scale[o] · Σ_i x[i]·(int8)packed[o·i_dim+i]   — int8, per-row scale.
extern "C" __global__ void gemv_i8(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    float acc = dot_i8_wave(x, packed + o * (size_t)i_dim, i_dim, lane);
    if (lane == 0) y[o] = acc * scale[o];
}

// grid = ceil(o_dim / ROWS_PER_BLOCK) blocks of 256 threads (one wave per row).
extern "C" int rivoli_gemv_i4(
    const float* x, const unsigned char* packed, const float* scale,
    int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_i4, dim3((o_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0, x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// grid = ceil(o_dim / ROWS_PER_BLOCK) blocks of 256 threads (one wave per row).
extern "C" int rivoli_gemv_vq(
    const float* x, const unsigned char* indices, const unsigned short* scales,
    const float* codebook, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_vq, dim3((o_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0, x, indices, scales, codebook,
                       o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_i8(
    const float* x, const unsigned char* packed, const float* scale,
    int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_i8, dim3((o_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0, x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// y[o] = Σ_i x[i]·w[o·i_dim+i]  — plain f32 GEMV, no scale. The MoE router gate
// (gate.weight is F32; matches quant.rs matvec_f32_bytes). Coalesced wave-per-row.
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

// y[i] = x[i] · (1/sqrt(mean(x²)+eps)) · w[i]  — RMSNorm, matching math.rs. One
// block reduces Σx² in LDS (tree order → ms differs from the scalar sequential
// sum by f32 rounding), then every thread scales. w is the f32 norm weight.
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

// Interleaved RoPE (matching attn.rs rope_interleave) on `count` segments of
// `seg` elements: segment s lives at base[s*stride .. +seg]; input pair
// (2j,2j+1) rotates by angle pos·theta^(-2j/seg) into outputs (j, half+j).
// Angles in double (as the reference) so long-context argument reduction doesn't
// drift. One block per segment, seg/2 threads; all pairs are read before any
// write (barrier), so the in-place de-interleave is race-free.
// `count` is the grid dim, not a kernel arg — one block per segment.
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

extern "C" int rivoli_gemv_f32(
    const float* x, const float* w, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_f32, dim3((o_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0, x, w, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rmsnorm(
    const float* x, const float* w, int n, float eps, float* y) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rmsnorm, dim3(1), dim3(256), 0, 0, x, w, n, eps, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rope(
    float* base, int count, int stride, int seg, int pos, double theta) {
    if (count <= 0 || seg <= 0 || (seg & 1) || (seg / 2) > 1024) return 1001;
    // Overlapping segments (stride < seg) would race across blocks (no inter-
    // block sync); the intended callers use stride ≥ seg (per-head qh, or count=1).
    if (count > 1 && stride < seg) return 1001;
    hipGetLastError();
    // block = seg/2 EXACTLY so every thread reaches the barrier (no divergence).
    hipLaunchKernelGGL(rope_interleave, dim3(count), dim3(seg / 2), 0, 0,
                       base, stride, seg, pos, theta);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
