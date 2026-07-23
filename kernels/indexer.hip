// DSA lightning-indexer decode kernels (device port of the old int4 tree's
// indexer, adapted to the fp8 checkpoint). Per full layer, for one decoded token:
//   k   = rope(layernorm(wk·x))                      [index_head_dim, bf16 cache]
//   q_h = rope(wq_b·q_resid)[h]                       [n_heads · head_dim, f32]
//   w   = weights_proj·x                              [n_heads, f32]
//   I_t = Σ_{h∈active} w_h·wscale·ReLU(q_h·k_t·dscale)
// The top-index_topk scores are picked host-side (small D2H of scores, then the
// host topk), and the rows fed straight into the sparse mla_attend gather. The
// projections wk/wq_b/weights_proj are fp8-e4m3 block-scaled in this checkpoint, so
// they run through the shared `gemv_fp8` (linalg.hip) — NOT a bf16 GEMV. k_norm
// stays bf16 (widened to f32 at placement) and feeds `layernorm` below.
// All pointers device-resident; launch-only (no sync/alloc/copy). Return codes:
// 0 ok, POSITIVE arg guard, NEGATIVE -hipError (shared convention).

#include "common.hpp"

// MISA router block size (pooled tokens per key). MUST match
// crate::indexer::MISA_BLOCK — the running-mean pool the router scores over.
#define MISA_BLOCK 1024

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
// One block per cached token; block staged with the key row in LDS. `heads`
// (nullable) lists the `nact` active heads (MISA); null = all `nh` heads (DSA).
extern "C" __global__ void index_score(
    const float* __restrict__ q, const float* __restrict__ w,
    const unsigned short* __restrict__ kcache, const unsigned int* __restrict__ heads,
    int nt, int nact, int hd, float wscale, float dscale,
    float* __restrict__ scores) {
    extern __shared__ float smem[];
    float* kf = smem;      // [hd] staged key row
    float* part = kf + hd; // [blockDim.x] per-head contribution
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

// MISA block pool: fold token `t`'s indexer key `k` (hd f32) into its block's
// running mean: block b = t/MISA_BLOCK covers [b·B,(b+1)·B); in_block = t%B==0 opens
// the block, else folds with weight 1/(in_block+1) — the exact block-key mean.
// Maintained EVERY token from the SAME roped key that lands in the key cache.
extern "C" __global__ void index_pool_push(
    const float* __restrict__ k, float* __restrict__ pool, int t, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hd) return;
    int in_block = t % MISA_BLOCK;
    float* m = pool + (size_t)(t / MISA_BLOCK) * hd;
    if (in_block == 0) {
        m[i] = k[i];
    } else {
        m[i] += (k[i] - m[i]) / (float)(in_block + 1);
    }
}

// MISA head router (paper arXiv 2605.07363 Eq. 7-8). One block per head j:
//   e[j] = (1/m_blocks) · Σ_b |w[j] · ReLU(Σ_i q[j·hd+i]·pool[b·hd+i])|
// Matches the scalar `(w[j]·dot.max(0)).abs()` sum exactly — no wscale/dscale in the
// router (raw ranking). q/w/pool/e all f32.
extern "C" __global__ void index_head_route(
    const float* __restrict__ q, const float* __restrict__ w,
    const float* __restrict__ pool, int m_blocks, int nh, int hd,
    float* __restrict__ e) {
    __shared__ float red[256];
    int j = blockIdx.x;
    if (j >= nh) return;
    int tid = threadIdx.x;
    const float* qj = q + (size_t)j * hd;
    float wj = w[j];
    float sum = 0.0f;
    for (int b = 0; b < m_blocks; ++b) {
        const float* kb = pool + (size_t)b * hd;
        float p = 0.0f;
        for (int i = tid; i < hd; i += blockDim.x) p += qj[i] * kb[i];
        red[tid] = p;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) sum += fabsf(wj * fmaxf(red[0], 0.0f));
        __syncthreads();
    }
    if (tid == 0) e[j] = sum / (float)(m_blocks > 0 ? m_blocks : 1);
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
    // block = next pow2 ≥ nact (the reduction assumes it), ≥ 32 to stage kf.
    int bd = 1;
    while (bd < nact) bd <<= 1;
    if (bd < 32) bd = 32;
    if (bd > 1024) return 1002;
    size_t lds = ((size_t)hd + bd) * sizeof(float);
    if (lds > 65536) return 1002;
    hipGetLastError();
    hipLaunchKernelGGL(index_score, dim3(nt), dim3(bd), lds, 0,
                       q, w, kcache, heads, nt, nact, hd, wscale, dscale, scores);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_pool_push(
    const float* k, float* pool, int t, int hd) {
    if (hd <= 0 || t < 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(index_pool_push, dim3((hd + 255) / 256), dim3(256), 0, 0,
                       k, pool, t, hd);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_head_route(
    const float* q, const float* w, const float* pool,
    int m_blocks, int nh, int hd, float* e) {
    if (m_blocks <= 0 || nh <= 0 || hd <= 0) return 1001;
    // block = next pow2 ≥ hd (the LDS reduction assumes it), capped at red[256].
    int bd = 1;
    while (bd < hd) bd <<= 1;
    if (bd > 256) bd = 256;
    hipGetLastError();
    hipLaunchKernelGGL(index_head_route, dim3(nh), dim3(bd), 0, 0,
                       q, w, pool, m_blocks, nh, hd, e);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : -(int)err;
}
