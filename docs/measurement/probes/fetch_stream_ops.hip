// fetch_stream_ops — what does the reaper pay per completed read, above the NVMe?
//
// WHY: a decode measures 10.06 GB/s of demand fetch (`io_wait` 18.3 s for 184 GB, int3-vq
// @100 GiB) while `fetch_batch.hip` shows the device+filesystem delivering 12.5-13.7 GB/s
// in the engine's own batch shape, into the engine's own pinned buffers. ~4.7 s is lost
// ABOVE the device, and `asyncfetch.rs::run_job`'s reap loop — which is what `io_wait`
// brackets — enqueues three HIP ops per completed read:
//
//   hipMemcpyAsync        the bounce->slot staging copy      (stream.rs `reap`)
//   hipStreamWriteValue64 the ticket a consumer waits on     (Timeline::signal)
//   hipLaunchHostFunc     the per-read Signal                (Signal::arm_on)
//
// The third is the suspect. A host function is a HOST ROUND TRIP recorded INTO the stream:
// everything behind it in that stream waits for a CPU thread to be scheduled and return.
// The other two are fire-and-forget descriptor writes.
//
// Measures the ENQUEUE cost (what the reaper thread pays, inside the io_wait clock) apart
// from the STREAM cost (what the copies behind it pay), because they are different budgets
// and only the first shows up as lost fetch bandwidth.
//
// build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/fetch_stream_ops docs/measurement/probes/fetch_stream_ops.hip
// run:   flock /var/run/sys-gpu.lock -c /tmp/fetch_stream_ops
#include <hip/hip_runtime.h>
#include <cstdio>
#include <ctime>

#define CK(x) do { hipError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } } while (0)

// One int3-vq expert, as `pin.rs` lays it out: 3.76 GiB/layer over 257 blocks.
static const size_t EXPERT = 15335424;
static const int N = 16;  // a full staging ring

static void noop(void *) {}

static double now_ms() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main() {
    void *host = nullptr, *dev = nullptr;
    CK(hipHostMalloc(&host, EXPERT * N, hipHostMallocDefault));
    CK(hipMalloc(&dev, EXPERT * N));
    // The timeline the ticket is written to: device-visible host memory, as
    // `rivoli_timeline_create` allocates it.
    unsigned long long *tl = nullptr;
    CK(hipHostMalloc((void **)&tl, sizeof(*tl), hipHostMallocDefault));
    *tl = 0;
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    // Warm the DMA path and the host-func machinery — the first of each pays setup no
    // steady-state layer pays.
    CK(hipMemcpyAsync(dev, host, EXPERT, hipMemcpyHostToDevice, s));
    CK(hipLaunchHostFunc(s, noop, nullptr));
    CK(hipStreamSynchronize(s));

    printf("per completed read, %d reads deep (what the reaper pays INSIDE the io_wait clock)\n\n",
           N);
    // `arm` selects whether the loop matches today's `run_job` (all three ops) or the same
    // loop with the unawaited Signal dropped.
    for (int arm = 1; arm >= 0; arm--) {
        double enq_best = 1e9, wall_best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            CK(hipStreamSynchronize(s));
            double t0 = now_ms();
            for (int i = 0; i < N; i++) {
                CK(hipMemcpyAsync((char *)dev + (size_t)i * EXPERT,
                                  (char *)host + (size_t)i * EXPERT,
                                  EXPERT, hipMemcpyHostToDevice, s));
                CK(hipStreamWriteValue64(s, tl, (uint64_t)(rep * N + i + 1), 0));
                if (arm) CK(hipLaunchHostFunc(s, noop, nullptr));
            }
            double t1 = now_ms();
            CK(hipStreamSynchronize(s));
            double t2 = now_ms();
            if (t1 - t0 < enq_best) enq_best = t1 - t0;
            if (t2 - t0 < wall_best) wall_best = t2 - t0;
        }
        printf("%-22s enqueue %7.3f ms (%6.1f us/read) | stream drains at %7.3f ms (%5.2f GB/s)\n",
               arm ? "memcpy+value+hostfunc" : "memcpy+value",
               enq_best, enq_best * 1e3 / N, wall_best,
               (double)EXPERT * N / (wall_best * 1e-3) / 1e9);
    }
    return 0;
}
