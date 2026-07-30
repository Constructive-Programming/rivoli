#include <hip/hip_runtime.h>
#include <cstdio>
// Does hipStreamWaitValue64 give the WAITER an acquire view of the signaller's writes?
//
// v1 of this probe reported BROKEN and was itself broken: with one shared buffer and no
// back-pressure, the producer raced ahead overwriting the buffer for later iterations
// while the consumer was still checking an earlier one. Nearly every element mismatched,
// which is the signature of a bad test, not a subtle race. Fixed with a reverse timeline
// (`ack`) so the producer cannot refill until the consumer has read.
__global__ void fill(int* p, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}
__global__ void check(const int* p, int n, int want, int* bad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && p[i] != want) atomicAdd(bad, 1);
}
int main() {
  const int N = 1 << 22;                 // 16 MiB, still in flight when the signal lands
  int *buf = nullptr, *bad = nullptr;
  if (hipMalloc(&buf, N * sizeof(int)) != hipSuccess) return 1;
  if (hipHostMalloc(&bad, sizeof(int)) != hipSuccess) return 1;
  *bad = 0;
  uint64_t *sig = nullptr, *ack = nullptr;
  if (hipExtMallocWithFlags((void**)&sig, 8, hipMallocSignalMemory) != hipSuccess) return 2;
  if (hipExtMallocWithFlags((void**)&ack, 8, hipMallocSignalMemory) != hipSuccess) return 2;
  *sig = 0; *ack = 0;
  hipStream_t a, b;
  (void)hipStreamCreate(&a); (void)hipStreamCreate(&b);
  const int ITERS = 200;
  for (int it = 1; it <= ITERS; ++it) {
    // Producer B: wait until the consumer has finished the PREVIOUS round, then write and
    // publish. Without this the buffer is a shared mutable across overlapping iterations.
    (void)hipStreamWaitValue64(b, ack, (uint64_t)(it - 1), hipStreamWaitValueGte, ~0ull);
    hipLaunchKernelGGL(fill, dim3((N + 255) / 256), dim3(256), 0, b, buf, N, it);
    (void)hipStreamWriteValue64(b, sig, (uint64_t)it, 0);
    // Consumer A: enqueued BEFORE the producer has run, which is the real usage pattern.
    (void)hipStreamWaitValue64(a, sig, (uint64_t)it, hipStreamWaitValueGte, ~0ull);
    hipLaunchKernelGGL(check, dim3((N + 255) / 256), dim3(256), 0, a, buf, N, it, bad);
    (void)hipStreamWriteValue64(a, ack, (uint64_t)it, 0);
  }
  (void)hipDeviceSynchronize();
  printf("mismatches: %d over %lld checks\n", *bad, (long long)N * ITERS);
  printf("VISIBILITY: %s\n", *bad == 0 ? "OK — waiter sees signaller's writes"
                                       : "BROKEN — ordering without visibility");
  return *bad != 0;
}
