// M3 (4/4) per-layer linear-algebra primitives — the projections the forward
// pass needs on-device alongside the fused MoE/attention kernels, so the whole
// token stays resident (no host round-trip = no per-layer join). Each is a plain
// batch-1 GEMV: one thread per output row, summing i ascending to match the
// scalar oracle (quant.rs matvec_i4_rows / matvec_i8) bit-for-bit. All pointers
// are device-resident; the launchers only launch (caller joins via
// rivoli_device_sync). Return codes: 0 ok, POSITIVE arg guard, NEGATIVE -hipError.
//
// Coalescing is row-per-thread (adjacent lanes stride one weight row apart) —
// the D3 wave-coalesced-load tune applies here too, deferred to the measurement
// pass; correctness first.

#include <hip/hip_runtime.h>

// y[o] = scale[o] · Σ_i x[i]·(nibble(row_o,i) - 8)   — int4, per-row scale.
extern "C" __global__ void gemv_i4(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= o_dim) return;
    size_t rb = (size_t)(i_dim + 1) / 2;
    const unsigned char* row = packed + (size_t)o * rb;
    float acc = 0.0f;
    for (int i = 0; i < i_dim; ++i) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        acc += x[i] * (float)(n - 8);
    }
    y[o] = acc * scale[o];
}

// y[o] = scale[o] · Σ_i x[i]·(int8)packed[o·i_dim+i]   — int8, per-row scale.
extern "C" __global__ void gemv_i8(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= o_dim) return;
    const unsigned char* row = packed + (size_t)o * i_dim;
    float acc = 0.0f;
    for (int i = 0; i < i_dim; ++i) acc += (float)((signed char)row[i]) * x[i];
    y[o] = acc * scale[o];
}

extern "C" int rivoli_gemv_i4(
    const float* x, const unsigned char* packed, const float* scale,
    int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_i4, dim3((o_dim + 255) / 256), dim3(256), 0, 0,
                       x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_i8(
    const float* x, const unsigned char* packed, const float* scale,
    int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_i8, dim3((o_dim + 255) / 256), dim3(256), 0, 0,
                       x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
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
extern "C" __global__ void rope_interleave(
    float* __restrict__ base, int count, int stride, int seg, int pos, double theta) {
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
    if (count <= 0 || seg <= 0 || (seg & 1)) return 1001;
    hipGetLastError();
    // block = seg/2 EXACTLY so every thread reaches the barrier (no divergence).
    hipLaunchKernelGGL(rope_interleave, dim3(count), dim3(seg / 2), 0, 0,
                       base, count, stride, seg, pos, theta);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
