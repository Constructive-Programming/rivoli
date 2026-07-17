// M0 toolchain probe: the smallest real kernel + launcher, to prove hipcc
// compiles, links against libamdhip64, and a launch reaches gfx1151. Replaced
// by moe_fused.hip in M2. extern "C" so Rust FFI binds without name mangling.

#include <hip/hip_runtime.h>

__global__ void axpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Runs a==2 axpy over n elements on-device; returns y[0] (== 2*1+0 == 2 on
// success) or a negative HIP error code. Zero device allocation persists —
// this is a liveness probe, not a compute path.
extern "C" int rivoli_probe(int n) {
    float *dx = nullptr, *dy = nullptr;
    size_t bytes = (size_t)n * sizeof(float);
    if (hipMalloc(&dx, bytes) != hipSuccess) return -1;
    if (hipMalloc(&dy, bytes) != hipSuccess) { hipFree(dx); return -2; }
    if (hipMemset(dx, 0, bytes) != hipSuccess) { hipFree(dx); hipFree(dy); return -3; }
    if (hipMemset(dy, 0, bytes) != hipSuccess) { hipFree(dx); hipFree(dy); return -4; }
    // dx = 1.0f everywhere via a tiny host copy of one page.
    float one = 1.0f;
    for (int i = 0; i < n && i < 1; ++i)
        (void)hipMemcpy(dx, &one, sizeof(float), hipMemcpyHostToDevice);

    dim3 block(256), grid((n + 255) / 256);
    hipLaunchKernelGGL(axpy, grid, block, 0, 0, 2.0f, dx, dy, n);
    if (hipDeviceSynchronize() != hipSuccess) { hipFree(dx); hipFree(dy); return -5; }

    float y0 = -1.0f;
    (void)hipMemcpy(&y0, dy, sizeof(float), hipMemcpyDeviceToHost);
    hipFree(dx);
    hipFree(dy);
    return (int)y0;  // 2 on success (2*1 + 0)
}
