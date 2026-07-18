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
