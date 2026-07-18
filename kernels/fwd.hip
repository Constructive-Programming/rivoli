// M3 (4/4) forward-pass glue kernels: the small device ops that stitch the
// GEMV / norm / attention / MoE kernels into one resident token, so nothing
// round-trips to the host mid-layer. All launch-only; caller joins via
// rivoli_device_sync. Return codes: 0 ok, POSITIVE arg guard, NEGATIVE -hipError.

#include <hip/hip_runtime.h>

// f32 → bf16, round-to-nearest-even, non-finite verbatim (matches math.rs).
__device__ __forceinline__ unsigned short f2bf16(float x) {
    unsigned int b;
    __builtin_memcpy(&b, &x, sizeof(b));
    if (!isfinite(x)) return (unsigned short)(b >> 16);
    unsigned int r = ((b >> 16) & 1u) + 0x7fffu;
    return (unsigned short)((b + r) >> 16);
}

// x[i] = (int8)embed[token·hidden + i] · scale[token]  — embedding row lookup.
extern "C" __global__ void embed_i8_row(
    const unsigned char* __restrict__ packed, const float* __restrict__ scale,
    int token, int hidden, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    x[i] = (float)((signed char)packed[(size_t)token * hidden + i]) * scale[token];
}

// Append one token's latent (kvl) and roped key (ropn) to the bf16 KV slabs at
// row `pos`. One flat launch over kvl+ropn threads.
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
