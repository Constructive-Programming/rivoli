// fetch_batch — where does the demand fetch lose 26% of the drive?
//
// A decode measures 10.06 GB/s of demand fetch (`io_wait` 18.3 s for 184 GB, int3-vq
// @100 GiB). A plain random-pread probe over the same files reaches 11.08 GB/s at QD1 and
// 14.2 at QD4, and weighting THAT by the engine's own per-layer miss distribution
// (`moe_us_by_miss`) predicts 13.6 s. So ~4.7 s is unexplained by queue depth.
//
// Two candidates this separates:
//
//   1. THE PINNED ARENA. `stream.rs` reads into `hipHostMalloc` memory so the staging copy
//      out of it is a DMA. O_DIRECT into pinned pages is a different write target than the
//      `posix_memalign` buffer the plain probe used. `-pinned` vs `-pageable` is that arm.
//
//   2. PER-BATCH DEAD TIME. The engine's ring is not continuously fed: one layer submits m
//      reads, waits for all m, and then the queue is EMPTY until the next layer routes.
//      A streaming probe never pays that ramp. `-batch` reproduces it (barrier, submit m,
//      join all, repeat); `-stream` is the continuously-fed arm.
//
// Threads, not io_uring: this is asking what the DEVICE and the BUFFER do, and one blocking
// pread per outstanding read expresses queue depth without a second variable. If both arms
// here reach ~13.5 GB/s, the loss is in the ring rather than below it.
//
//   3. LPDDR5 CONTENTION, added after 1 and 2 came back clean. This is a unified-memory
//      APU: the NVMe DMAs into the same LPDDR5 the GPU is streaming expert weights out of,
//      and the staging copy itself moves 2x15.3 MB per read at ~87 GB/s. A probe against an
//      IDLE GPU is measuring a machine the engine never runs on. `-hog` reproduces the load.
//
// build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/fetch_batch docs/probes/fetch_batch.hip
// run:   flock /tmp/rivoli-gpu.lock -c '/tmp/fetch_batch /var/db/rivoli/glm52-vq3-full'
#define _GNU_SOURCE
#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define EXPERT 15335424UL
#define ALIGN 4096UL
#define MAXQD 8

static int fds[128], nfile;
static off_t fsz[128];
static char *buf[MAXQD];
static int reps;              // batches (or reads) per thread
static pthread_barrier_t bar; // batch mode only
static int batched;
static volatile int hog_stop;

// A pure streaming read: no arithmetic worth the name, so it saturates LPDDR5 rather than
// the CUs — the same thing the MoE expert kernels do (they run at ~87 GB/s in-engine).
__global__ void hog(const float4 *__restrict__ src, float4 *__restrict__ sink, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 a = src[i % n];
    if (a.x == 1e30f) sink[0] = a; // never taken; defeats dead-code elimination
}

// Keeps a second stream permanently busy streaming `bytes` of device memory, so the preads
// below contend for the memory controller exactly as they do during a decode.
static void *hog_thread(void *arg) {
    size_t bytes = 1ull << 30;
    void *src = nullptr, *sink = nullptr;
    if (hipMalloc(&src, bytes) || hipMalloc(&sink, 1024)) return NULL;
    hipStream_t s;
    hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    size_t n = bytes / sizeof(float4);
    while (!hog_stop) {
        for (int i = 0; i < 32; i++)
            hipLaunchKernelGGL(hog, dim3(n / 256), dim3(256), 0, s, (const float4 *)src,
                               (float4 *)sink, n);
        hipStreamSynchronize(s);
    }
    hipStreamDestroy(s);
    hipFree(src);
    hipFree(sink);
    (void)arg;
    return NULL;
}

static int split;  // 1 = all threads read disjoint CHUNKS OF ONE expert, not separate experts
static int qd_now; // the current K, so a split worker knows its chunk size

static void *worker(void *arg) {
    long id = (long)arg;
    // In split mode every thread must land on the same (file, offset), so they share one
    // seed sequence and step it identically.
    unsigned seed = 12345u + (split ? 0 : id * 7919u);
    for (int i = 0; i < reps; i++) {
        // Every thread lands on the same barrier before AND after its read, so the batch
        // starts empty and is fully drained before the next — the engine's shape.
        if (batched) pthread_barrier_wait(&bar);
        int f = rand_r(&seed) % nfile;
        off_t off = (off_t)(rand_r(&seed) % ((fsz[f] - EXPERT) / ALIGN)) * ALIGN;
        size_t n = split ? EXPERT / qd_now : EXPERT;
        if (split) off += (off_t)id * n;
        if (pread(fds[f], buf[id], n, off) != (ssize_t)n) {
            perror("pread");
            exit(1);
        }
        if (batched) pthread_barrier_wait(&bar);
    }
    return NULL;
}

static double run(int qd, int nbatch) {
    reps = nbatch;
    qd_now = qd;
    pthread_barrier_init(&bar, NULL, qd);
    pthread_t t[MAXQD];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < qd; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < qd; i++) pthread_join(t[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &b);
    pthread_barrier_destroy(&bar);
    double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
    // Split mode reads ONE expert per batch (chunked qd ways); non-split reads qd of them.
    return (double)EXPERT * (split ? 1 : qd) * nbatch / 1e9 / s;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/var/db/rivoli/glm52-vq3-full";
    for (int l = 3; l < 78 && nfile < 128; l++) {
        char p[600];
        snprintf(p, sizeof p, "%s/L%02d.vq3", dir, l);
        int fd = open(p, O_RDONLY | O_DIRECT);
        if (fd < 0) continue;
        struct stat st;
        fstat(fd, &st);
        fds[nfile] = fd;
        fsz[nfile] = st.st_size;
        nfile++;
    }
    printf("%d files\n\n", nfile);

    for (int i = 0; i < MAXQD; i++)
        if (hipHostMalloc((void **)&buf[i], EXPERT, hipHostMallocDefault)) return 1;

    for (int hogging = 0; hogging <= 1; hogging++) {
        pthread_t h;
        if (hogging) {
            hog_stop = 0;
            pthread_create(&h, NULL, hog_thread, NULL);
            sleep(1); // let the hog reach steady state before timing anything
        }
        batched = 1; // the engine's shape: submit m, drain all m, then the queue is empty
        for (split = 0; split <= 1; split++) {
            printf("%-9s %-22s", hogging ? "GPU-BUSY" : "gpu-idle",
                   split ? "ONE expert split K ways" : "K separate experts");
            for (int qd = 1; qd <= 8; qd *= 2)
                printf("  K%d %5.2f GB/s (%4.2f ms/expert)", qd, run(qd, 96 / qd),
                       EXPERT / 1e9 / run(qd, 96 / qd) * 1e3);
            printf("\n");
        }
        if (hogging) {
            hog_stop = 1;
            pthread_join(h, NULL);
        }
    }
    for (int i = 0; i < MAXQD; i++) hipHostFree(buf[i]);
    return 0;
}
