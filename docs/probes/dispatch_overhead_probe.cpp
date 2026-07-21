// Probe: what does rivoli's dispatch pattern cost, independent of any real work?
// The engine issues ~1700 null-stream kernel launches and ~150 hipDeviceSynchronize
// calls per token (2 per layer: the mid-attention gate drain and the end-of-layer
// join). This measures each in isolation:
//   1) null-stream launch-to-launch gap for a trivial kernel
//   2) hipDeviceSynchronize round-trip when the queue is already empty
//   3) a small single-block kernel (rmsnorm's shape: grid=1, 256 threads)
//   4) the flash-attention launch shape (8 blocks only -> 20% of a 40-CU GPU)
//
// build: hipcc -O3 --offload-arch=gfx1151 overhead_probe.cpp -o overhead_probe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

__global__ void trivial(float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.0f; }

// rmsnorm's actual shape: one block, 256 threads, reduce 6144 then scale.
__global__ void rms_shape(const float* __restrict__ x, float* __restrict__ y, int n) {
    __shared__ float red[256];
    int t = threadIdx.x;
    float local = 0.0f;
    for (int i = t; i < n; i += blockDim.x) local += x[i] * x[i];
    red[t] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) red[t] += red[t+s]; __syncthreads(); }
    float inv = 1.0f / sqrtf(red[0] / (float)n + 1e-6f);
    for (int i = t; i < n; i += blockDim.x) y[i] = x[i] * inv;
}

#define CK(e) do { hipError_t _e=(e); if(_e!=hipSuccess){ \
    printf("HIP error %s at line %d\n", hipGetErrorString(_e), __LINE__); exit(1);} } while(0)

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

int main() {
    float* p; CK(hipMalloc(&p, 6144 * sizeof(float))); CK(hipMemset(p, 0, 6144 * sizeof(float)));
    float* y; CK(hipMalloc(&y, 6144 * sizeof(float)));
    const int N = 2000;

    // warm
    for (int i = 0; i < 100; ++i) trivial<<<1,64>>>(p);
    CK(hipDeviceSynchronize());

    // 1) launch-only cost (host-side dispatch, no sync between)
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i) trivial<<<1,64>>>(p);
    double dispatch_us = us_since(t0) / N;
    CK(hipDeviceSynchronize());

    // 2) launch + immediate sync (the serialized pattern)
    t0 = clk::now();
    for (int i = 0; i < N; ++i) { trivial<<<1,64>>>(p); CK(hipDeviceSynchronize()); }
    double launch_sync_us = us_since(t0) / N;

    // 3) bare sync on an empty queue
    CK(hipDeviceSynchronize());
    t0 = clk::now();
    for (int i = 0; i < N; ++i) CK(hipDeviceSynchronize());
    double bare_sync_us = us_since(t0) / N;

    // 4) rmsnorm shape (grid=1) launched back to back
    for (int i = 0; i < 100; ++i) rms_shape<<<1,256>>>(p, y, 6144);
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < N; ++i) rms_shape<<<1,256>>>(p, y, 6144);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
    double rms_us = ms * 1000.0 / N;

    // 4b) same work spread over 40 blocks instead of 1 (what a grid-strided
    //     two-pass rmsnorm would approach) - just to bound the single-block cost.
    CK(hipEventRecord(e0));
    for (int i = 0; i < N; ++i) rms_shape<<<40,256>>>(p, y, 6144);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    CK(hipEventElapsedTime(&ms, e0, e1));
    double rms40_us = ms * 1000.0 / N;

    printf("gfx1151 dispatch costs (microseconds):\n");
    printf("  host-side launch, no sync        %7.2f us\n", dispatch_us);
    printf("  bare hipDeviceSynchronize (idle) %7.2f us\n", bare_sync_us);
    printf("  launch + sync (serialized)       %7.2f us\n", launch_sync_us);
    printf("  rmsnorm shape grid=1,  256 thr   %7.2f us\n", rms_us);
    printf("  rmsnorm shape grid=40, 256 thr   %7.2f us\n", rms40_us);

    printf("\nrivoli per-token projection:\n");
    printf("  ~1700 launches x dispatch        %7.2f ms\n", 1700 * dispatch_us / 1000.0);
    printf("  ~150 device_sync x bare cost     %7.2f ms\n", 150 * bare_sync_us / 1000.0);
    printf("  ~234 rmsnorm (grid=1)            %7.2f ms\n", 234 * rms_us / 1000.0);
    return 0;
}
