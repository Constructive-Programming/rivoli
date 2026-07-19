// DSA lightning-indexer decode kernels (device port of src/indexer.rs). Per
// full layer, for one decoded token:
//   k   = rope(layernorm(wk·x))                      [index_head_dim, bf16 cache]
//   q_h = rope(wq_b·q_resid)[h]                       [n_heads · head_dim, f32]
//   w   = weights_proj·x                              [n_heads, f32]
//   I_t = Σ_{h∈active} w_h·wscale·ReLU(q_h·k_t·dscale)
// The top-index_topk scores are picked host-side (small D2H of scores, then the
// existing host topk), and the rows fed straight into the sparse mla_attend
// gather. All pointers device-resident; launch-only (no sync/alloc/copy).
// Return codes: 0 ok, POSITIVE arg guard, NEGATIVE -hipError (shared convention).

#include <hip/hip_runtime.h>

#define WAVE 32
#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float bf16f(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}
__device__ __forceinline__ unsigned short f2bf16(float f) {
    unsigned int u;
    __builtin_memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)(u >> 16); // NaN
    u += 0x7fffu + ((u >> 16) & 1u); // round-to-nearest-even
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}

// y[o] = Σ_i x[i]·bf16(w[o·i_dim+i])  — GEMV against a row-major bf16 weight
// (the indexer's wk / wq_b / weights_proj). Coalesced wave-per-row, mirroring
// gemv_f32/gemv_i4. x is f32 (the layer input / q-LoRA residual).
extern "C" __global__ void gemv_bf16(
    const float* __restrict__ x, const unsigned short* __restrict__ w,
    int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const unsigned short* row = w + o * (size_t)i_dim;
    float acc = 0.0f;
    for (int i = lane; i < i_dim; i += WAVE) acc += x[i] * bf16f(row[i]);
    acc = wave_sum(acc);
    if (lane == 0) y[o] = acc;
}

// y[i] = (x[i]-mean)/sqrt(var+eps)·w[i] + b[i]  — LayerNorm (the indexer k_norm,
// which unlike every model RMSNorm ships a bias). One block reduces Σx and Σx²
// in LDS, then every thread normalizes. w,b are f32 (widened at placement).
extern "C" __global__ void layernorm(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, int n, float eps, float* __restrict__ y) {
    __shared__ float sred[256];
    __shared__ float sqred[256];
    int t = threadIdx.x;
    float s = 0.0f, sq = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        float v = x[i];
        s += v;
        sq += v * v;
    }
    sred[t] = s;
    sqred[t] = sq;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) {
            sred[t] += sred[t + st];
            sqred[t] += sqred[t + st];
        }
        __syncthreads();
    }
    float mean = sred[0] / (float)n;
    float var = sqred[0] / (float)n - mean * mean;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = t; i < n; i += blockDim.x) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// kcache[pos·hd + i] = f2bf16(k[i])  — append one indexer key row (bf16), the
// single-stream analog of append_kv.
extern "C" __global__ void index_append(
    const float* __restrict__ k, unsigned short* __restrict__ kcache, int pos, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hd) kcache[(size_t)pos * hd + i] = f2bf16(k[i]);
}

// scores[t] = Σ_{h∈active} w[h]·wscale·ReLU((q_h·k_t)·dscale) over t∈[0,nt).
// One block per cached token; nh_threads threads (one per head-slot). `heads`
// (nullable) lists the `nact` active heads (MISA); null = all `nh` heads (DSA).
// Each active head-thread reduces its hd-dim dot against the LDS-staged key,
// then the block sums the per-head contributions. kcache bf16, q/w f32.
extern "C" __global__ void index_score(
    const float* __restrict__ q, const float* __restrict__ w,
    const unsigned short* __restrict__ kcache, const unsigned int* __restrict__ heads,
    int nt, int nh, int nact, int hd, float wscale, float dscale,
    float* __restrict__ scores) {
    extern __shared__ float smem[];
    float* kf = smem;             // [hd] staged key row
    float* part = kf + hd;        // [blockDim.x] per-head contribution
    int t = blockIdx.x;
    if (t >= nt) return;
    int tid = threadIdx.x;
    const unsigned short* krow = kcache + (size_t)t * hd;
    for (int i = tid; i < hd; i += blockDim.x) kf[i] = bf16f(krow[i]);
    __syncthreads();

    float contrib = 0.0f;
    if (tid < nact) {
        int h = heads ? (int)heads[tid] : tid;
        const float* qh = q + (size_t)h * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; ++i) dot += qh[i] * kf[i];
        contrib = w[h] * wscale * fmaxf(dot * dscale, 0.0f);
    }
    part[tid] = contrib;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[t] = part[0];
}

extern "C" int rivoli_gemv_bf16(
    const float* x, const unsigned short* w, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_bf16, dim3((o_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0, x, w, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_layernorm(
    const float* x, const float* w, const float* b, int n, float eps, float* y) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(layernorm, dim3(1), dim3(256), 0, 0, x, w, b, n, eps, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_append(
    const float* k, unsigned short* kcache, int pos, int hd) {
    if (hd <= 0 || pos < 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(index_append, dim3((hd + 255) / 256), dim3(256), 0, 0,
                       k, kcache, pos, hd);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_score(
    const float* q, const float* w, const unsigned short* kcache,
    const unsigned int* heads, int nt, int nh, int nact, int hd,
    float wscale, float dscale, float* scores) {
    if (nt <= 0 || nh <= 0 || nact <= 0 || nact > nh || hd <= 0) return 1001;
    // block = next pow2 ≥ nact (the reduction assumes it), ≥ enough to stage kf.
    int bd = 1;
    while (bd < nact) bd <<= 1;
    if (bd < 32) bd = 32;
    if (bd > 1024) return 1002;
    size_t lds = ((size_t)hd + bd) * sizeof(float);
    if (lds > 65536) return 1002;
    hipGetLastError();
    hipLaunchKernelGGL(index_score, dim3(nt), dim3(bd), lds, 0,
                       q, w, kcache, heads, nt, nh, nact, hd, wscale, dscale, scores);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
