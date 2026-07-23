// Forward-pass glue kernels: the small device ops that stitch the GEMV / norm /
// attention / MoE kernels into one resident token, so nothing round-trips to the
// host mid-layer. All launch-only; caller joins via rivoli_device_sync. Return
// codes: 0 ok, POSITIVE arg guard, NEGATIVE -hipError.

#include "common.hpp"

// x[i] = (int8)embed[token·hidden + i] · scale[token]  — embedding row lookup.
extern "C" __global__ void embed_i8_row(
    const unsigned char* __restrict__ packed, const float* __restrict__ scale,
    int token, int hidden, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    x[i] = (float)((signed char)packed[(size_t)token * hidden + i]) * scale[token];
}

// Append one token's latent (kvl) and roped key (ropn) to the bf16 KV slabs at row
// `pos`. One flat launch over kvl+ropn threads.
extern "C" __global__ void append_kv(
    const float* __restrict__ latent, const float* __restrict__ rope,
    unsigned short* __restrict__ lc, unsigned short* __restrict__ rc,
    int pos, int kvl, int ropn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kvl) {
        lc[(size_t)pos * kvl + i] = f2bf16(latent[i]);
    } else if (i < kvl + ropn) {
        int d = i - kvl;
        rc[(size_t)pos * ropn + d] = f2bf16(rope[d]);
    }
}

// qrope[head·ropn + d] = q[head·qh + nope + d]  — gather each head's roped query
// segment out of the strided q[H·qh] into contiguous qrope[H·ropn] for the flash
// kernel.
extern "C" __global__ void gather_rope(
    const float* __restrict__ q, float* __restrict__ qrope,
    int h, int qh, int nope, int ropn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h * ropn) return;
    int head = idx / ropn;
    int d = idx % ropn;
    qrope[idx] = q[(size_t)head * qh + nope + d];
}

// x[i] += y[i]  — residual add.
extern "C" __global__ void vadd(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Greedy argmax over `logits[0..n]`, reproducing the host fold in gpu.rs::argmax
// EXACTLY: best=0, bv=-inf; for i: if logits[i] > bv { bv=logits[i]; best=i }. The
// combine below is associative + commutative and yields the SAME result set as
// that sequential `>` fold: the maximum VALUE wins; NaN never wins (every compare
// against NaN is false, so it loses to any finite value and to the -inf identity);
// on an exact tie the SMALLER index wins. So `best` = lowest index of the maximum
// value, returned value = logits[best] — matching the host's `bv`, so the caller's
// `!bv.is_finite()` bail is reproduced verbatim. One block, LDS tree reduce.
__device__ __forceinline__ void amax_combine(
    float va, int ia, float vb, int ib, float* vo, int* io) {
    if (vb > va)       { *vo = vb; *io = ib; }   // b strictly greater (b not NaN)
    else if (va > vb)  { *vo = va; *io = ia; }   // a strictly greater (a not NaN)
    else if (va != va) { *vo = vb; *io = ib; }   // a is NaN -> a loses, take b
    else if (vb != vb) { *vo = va; *io = ia; }   // b is NaN -> b loses, take a
    else               { if (ia <= ib) { *vo = va; *io = ia; }  // equal -> smaller index
                         else          { *vo = vb; *io = ib; } }
}

extern "C" __global__ void argmax_reduce(
    const float* __restrict__ logits, int n, int* __restrict__ out_idx,
    float* __restrict__ out_val) {
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    int t = threadIdx.x;
    // Identity mirrors the host initial (best=0, bv=-inf): only wins if EVERY
    // element is NaN/-inf, which the host bails on (value stays non-finite) too.
    float best = -INFINITY;
    int   bidx = 0;
    for (int i = t; i < n; i += blockDim.x) {
        amax_combine(best, bidx, logits[i], i, &best, &bidx);
    }
    sval[t] = best;
    sidx[t] = bidx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            amax_combine(sval[t], sidx[t], sval[t + s], sidx[t + s], &sval[t], &sidx[t]);
        }
        __syncthreads();
    }
    if (t == 0) {
        *out_idx = sidx[0];
        *out_val = sval[0];
    }
}

extern "C" int rivoli_embed_i8_row(
    const unsigned char* packed, const float* scale, int token, int hidden, float* x) {
    if (hidden <= 0 || token < 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(embed_i8_row, dim3((hidden + 255) / 256), dim3(256), 0, 0,
                       packed, scale, token, hidden, x);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_append_kv(
    const float* latent, const float* rope, unsigned short* lc, unsigned short* rc,
    int pos, int kvl, int ropn) {
    if (pos < 0 || kvl <= 0 || ropn <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(append_kv, dim3((kvl + ropn + 255) / 256), dim3(256), 0, 0,
                       latent, rope, lc, rc, pos, kvl, ropn);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gather_rope(
    const float* q, float* qrope, int h, int qh, int nope, int ropn) {
    if (h <= 0 || qh <= 0 || nope < 0 || ropn <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gather_rope, dim3((h * ropn + 255) / 256), dim3(256), 0, 0,
                       q, qrope, h, qh, nope, ropn);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_vadd(float* x, const float* y, int n) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(vadd, dim3((n + 255) / 256), dim3(256), 0, 0, x, y, n);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// Single-block argmax reduction (256 threads, grid-stride). `out_idx`/`out_val`
// are one device word each (the caller's 8-byte result buffer). Launch-only.
extern "C" int rivoli_argmax(
    const float* logits, int n, int* out_idx, float* out_val) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(argmax_reduce, dim3(1), dim3(256), 0, 0, logits, n, out_idx, out_val);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
