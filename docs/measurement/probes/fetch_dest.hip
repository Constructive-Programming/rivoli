// fetch_dest — is O_DIRECT DMA into VMM pages STILL half the rate of DMA into pinned host?
//
// 2026-08-18. The question this exists for: `--direct-vmm-dma` was deleted 2026-08-01
// because DMA into the VMM pool ran 5.66 GB/s against 12.4 GB/s into the pinned bounce
// arena, so removing the staging hop more than DOUBLED the cost of a miss (1239 -> 2709
// us/expert, 2.59 -> 1.19 tok/s; docs/reference/architecture.md §3). Recovering the direct
// path would delete the arena, the H2D copy, `mod stage`, and all three coherence flags
// (--arena-refresh, --copy-by-kernel, --pinned-coherent) -- i.e. it deletes the LOCUS of
// the GLM nondeterminism bug rather than repairing it. So the simplification is worth
// having if, and only if, that bandwidth gap has closed.
//
// It might have. The 5.66 figure was measured on kernel 6.18.38; this box runs 6.18.39,
// with ROCm 7.14 / clang 23.1 since. That path is demonstrably volatile across point
// releases -- the amdgpu `get_user_pages` EFAULT that CREATED the arena was itself a
// 6.18.38 regression against <=6.18.35-r1 that then stopped reproducing.
//
// PRE-REGISTERED DECISION THRESHOLD: >= 11.4 GB/s in the VMM arm and the direct path is
// worth building. Derivation: direct starts ~4% ahead (it deletes the 38 us/layer hop,
// 0.74% of a token, plus the 1-3% --arena-refresh costs), a token spends 181 of its 386 ms
// on 146.65 misses, so direct may be 8.5% worse per miss and still tie -- 12.4/1.085.
//
// Two arms, one variable: WHERE the O_DIRECT read lands.
//   PINNED -- hipHostMalloc, hipHostMallocDefault. What stream.rs's bounce arena is.
//   VMM    -- the rivoli_vmm_alloc sequence verbatim (hipMemCreate device-local +
//             hipMemSetAccess for device THEN host). What the routed pool is, and what
//             --direct-vmm-dma read into.
// Everything else is held to the engine's shape, borrowed from fetch_batch.hip: real
// O_DIRECT reads of whole experts at random block-aligned offsets across the 75 layer
// files, submit-K-drain-all-K (the engine's ring is empty between layers), and a GPU-BUSY
// arm because this is a unified-memory APU -- the NVMe DMAs into the same LPDDR5 the MoE
// kernels stream out of, so an idle-GPU number measures a machine the engine never runs on.
//
// THE ARMS MUST BE PROVEN DISTINCT, or a silent fallback to host memory reports ~12 GB/s
// and reads as a win (see memory: a plant must change the tree). Two independent readbacks
// per arm, printed before any timing:
//   1. hipPointerGetAttributes -- the two arms must NOT report the same memory type.
//   2. DEVICE-SIDE READ BANDWIDTH off the buffer. VMM is device-local MTYPE_RW (~220 GB/s);
//      hipHostMalloc always maps system-domain (~87 GB/s in-engine). If these two numbers
//      match, the arm did not apply and every write-side number below is void.
// An EFAULT on the VMM arm is the 2026-07-17 kernel regression returning; it is reported
// per-arm and does NOT kill the process, so the PINNED control still prints.
//
// Three trials per cell, printed individually rather than averaged: fetch_batch.hip's own
// spread is +-25% at K1 across runs of the same probe, and an early "26% unexplained" was
// that spread mistaken for a finding.
//
// build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/fetch_dest docs/measurement/probes/fetch_dest.hip
// run:   flock /var/run/sys-gpu.lock -c '/tmp/fetch_dest /var/db/rivoli/glm52-vq3-full'
#define _GNU_SOURCE
#include <hip/hip_runtime.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define EXPERT 15335424UL // one gate|up|down vq3 expert, exactly 3744 * 4096
#define ALIGN 4096UL
#define MAXQD 8
#define TRIALS 3

// Bytes this process has pulled through the BLOCK LAYER, from /proc/self/io. The
// discriminator for an O_DIRECT read that was silently served buffered: if a cell moved
// 1.47 GB of `pread` and `read_bytes` did not move with it, the number is a page-cache
// number and not a disk number. Added 2026-08-18 after the first run of this probe
// reported 61 GB/s on the VMM arm -- 4x the drive's own ceiling in the PINNED control.
static unsigned long long read_bytes(void) {
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return 0;
    char k[64];
    unsigned long long v, out = 0;
    while (fscanf(f, "%63[^:]: %llu\n", k, &v) == 2)
        if (!strcmp(k, "read_bytes")) { out = v; break; }
    fclose(f);
    return out;
}

static int fds[128], nfile;
static off_t fsz[128];
static char *buf[MAXQD];
static int reps;
static pthread_barrier_t bar;
static volatile int hog_stop;
static volatile int read_errno; // first pread failure across all workers; 0 = clean
static unsigned trial_salt;      // varies the offset sequence per trial (see run())

// A pure streaming read: no arithmetic worth the name, so it saturates LPDDR5 rather than
// the CUs -- the same thing the MoE expert kernels do. Doubles as the memory-type readback.
__global__ void stream_read(const float4 *__restrict__ src, float4 *__restrict__ sink,
                            size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 a = src[i];
    if (a.x == 1e30f) sink[0] = a; // never taken; defeats dead-code elimination
}

static void *hog_thread(void *arg) {
    size_t bytes = 1ull << 30;
    void *src = nullptr, *sink = nullptr;
    if (hipMalloc(&src, bytes) || hipMalloc(&sink, 1024)) return NULL;
    hipStream_t s;
    hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    size_t n = bytes / sizeof(float4);
    while (!hog_stop) {
        for (int i = 0; i < 32; i++)
            hipLaunchKernelGGL(stream_read, dim3(n / 256), dim3(256), 0, s,
                               (const float4 *)src, (float4 *)sink, n);
        hipStreamSynchronize(s);
    }
    hipStreamDestroy(s);
    hipFree(src);
    hipFree(sink);
    (void)arg;
    return NULL;
}

static void *worker(void *arg) {
    long id = (long)arg;
    unsigned seed = 12345u + (unsigned)id * 7919u + trial_salt;
    for (int i = 0; i < reps; i++) {
        // Every thread lands on the same barrier before AND after its read, so the batch
        // starts empty and is fully drained before the next -- the engine's shape.
        pthread_barrier_wait(&bar);
        int f = rand_r(&seed) % nfile;
        off_t off = (off_t)(rand_r(&seed) % ((fsz[f] - EXPERT) / ALIGN)) * ALIGN;
        if (pread(fds[f], buf[id], EXPERT, off) != (ssize_t)EXPERT && !read_errno)
            read_errno = errno ? errno : EIO;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}

// GB/s of O_DIRECT reads into buf[0..qd), submit-qd-drain-all-qd, nbatch times.
static double run(int qd, int nbatch, double *disk_frac) {
    reps = nbatch;
    unsigned long long rb0 = read_bytes();
    pthread_barrier_init(&bar, NULL, qd);
    pthread_t t[MAXQD];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < qd; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < qd; i++) pthread_join(t[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &b);
    pthread_barrier_destroy(&bar);
    double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
    double asked = (double)EXPERT * qd * nbatch;
    *disk_frac = asked > 0 ? (double)(read_bytes() - rb0) / asked : 0.0;
    return asked / 1e9 / s;
}

// ---- the two destinations ---------------------------------------------------------------

static unsigned long long vmm_h[MAXQD];
static size_t vmm_m[MAXQD];

// hipMemCreate device-local + hipMemSetAccess(device) then (host). Copied from
// crates/backend/kernels/vmm.hip::rivoli_vmm_alloc so the probe tests the POOL's
// allocator and not a lookalike; keep the two in step.
static int vmm_alloc(void **out, unsigned long long *out_h, size_t *out_m, size_t size) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    size_t gran = 0;
    if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum))
        return 1;
    size_t msize = ((size + gran - 1) / gran) * gran;
    hipMemGenericAllocationHandle_t h;
    if (hipMemCreate(&h, msize, &prop, 0)) return 2;
    void *p = nullptr;
    if (hipMemAddressReserve(&p, msize, 0, nullptr, 0)) { hipMemRelease(h); return 3; }
    if (hipMemMap(p, msize, 0, h, 0)) {
        hipMemAddressFree(p, msize); hipMemRelease(h); return 4;
    }
    hipMemAccessDesc dd = {};
    dd.location.type = hipMemLocationTypeDevice; dd.location.id = 0;
    dd.flags = hipMemAccessFlagsProtReadWrite;
    hipMemAccessDesc hd = {};
    hd.location.type = hipMemLocationTypeHost; hd.location.id = 0;
    hd.flags = hipMemAccessFlagsProtReadWrite;
    if (hipMemSetAccess(p, msize, &dd, 1) || hipMemSetAccess(p, msize, &hd, 1)) {
        hipMemUnmap(p, msize); hipMemAddressFree(p, msize); hipMemRelease(h); return 5;
    }
    *out = p; *out_h = (unsigned long long)h; *out_m = msize;
    return 0;
}

// Device-side streaming read off `p`. The independent proof the two arms are different
// memory: device-local VMM reads ~220 GB/s, system-domain hipHostMalloc ~87.
static double device_read_gbs(void *p, size_t bytes) {
    void *sink = nullptr;
    if (hipMalloc(&sink, 1024)) return -1.0;
    size_t n = bytes / sizeof(float4);
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    const int iters = 64;
    // Warm the caches/TLB so the first launch's cost is not the measurement.
    hipLaunchKernelGGL(stream_read, dim3((n + 255) / 256), dim3(256), 0, 0,
                       (const float4 *)p, (float4 *)sink, n);
    hipDeviceSynchronize();
    hipEventRecord(a, 0);
    for (int i = 0; i < iters; i++)
        hipLaunchKernelGGL(stream_read, dim3((n + 255) / 256), dim3(256), 0, 0,
                           (const float4 *)p, (float4 *)sink, n);
    hipEventRecord(b, 0);
    hipDeviceSynchronize();
    float ms = 0;
    hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(sink);
    return ms > 0 ? (double)bytes * iters / 1e9 / (ms / 1e3) : -1.0;
}

static const char *memtype(void *p) {
    hipPointerAttribute_t at;
    if (hipPointerGetAttributes(&at, p) != hipSuccess) return "query-refused";
    switch (at.type) {
        case hipMemoryTypeHost: return "Host";
        case hipMemoryTypeDevice: return "Device";
        case hipMemoryTypeManaged: return "Managed";
        default: return "Unregistered";
    }
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
        fds[nfile] = fd; fsz[nfile] = st.st_size; nfile++;
    }
    printf("%d layer files under %s\n", nfile, dir);
    // RED-PROOF THE CACHED DETECTOR, before it is trusted to void a cell. A buffered
    // (no O_DIRECT) re-read of the same offset must move `read_bytes` the first time and
    // NOT the second; if it moves both times, /proc/self/io is not reporting what this
    // probe assumes and every "disk=100%" below is unearned.
    if (nfile) {
        char p2[600];
        snprintf(p2, sizeof p2, "%s/L03.vq3", dir);
        int bfd = open(p2, O_RDONLY); // deliberately NOT O_DIRECT
        void *tmp = malloc(64ull << 20);
        if (bfd >= 0 && tmp) {
            unsigned long long a0 = read_bytes();
            ssize_t r1 = pread(bfd, tmp, 64ull << 20, 1ull << 30);
            unsigned long long a1 = read_bytes();
            ssize_t r2 = pread(bfd, tmp, 64ull << 20, 1ull << 30);
            unsigned long long a2 = read_bytes();
            printf("cached-detector red-proof: buffered 64 MiB re-read moved read_bytes "
                   "%llu then %llu (%zd/%zd bytes) -- %s\n",
                   a1 - a0, a2 - a1, r1, r2,
                   (a2 - a1) * 10 < (a1 - a0) ? "DETECTOR ARMED"
                                              : "DETECTOR UNPROVEN, treat disk%% as decoration");
            close(bfd);
        }
        free(tmp);
    }
    if (nfile == 0) { fprintf(stderr, "no L*.vq3 files -- wrong artifact dir\n"); return 1; }

    for (int dest = 0; dest < 2; dest++) {
        const char *name = dest ? "VMM (pool)" : "PINNED (arena)";
        int failed = 0;
        for (int i = 0; i < MAXQD && !failed; i++) {
            if (dest) {
                int e = vmm_alloc((void **)&buf[i], &vmm_h[i], &vmm_m[i], EXPERT);
                if (e) { printf("\n%s: vmm_alloc failed at step %d -- arm SKIPPED\n", name, e); failed = 1; }
            } else if (hipHostMalloc((void **)&buf[i], EXPERT, hipHostMallocDefault)) {
                printf("\n%s: hipHostMalloc failed -- arm SKIPPED\n", name);
                failed = 1;
            }
        }
        if (failed) continue;

        // READBACKS FIRST. If these two lines match across the arms, the arm did not apply
        // and nothing below it means anything.
        printf("\n=== %s ===\n", name);
        size_t rbn = 256ull << 20; // > the 32 MiB MALL, so this measures DRAM not cache
        void *rb = nullptr;
        unsigned long long rbh = 0;
        size_t rbm = 0;
        double drd = -1.0;
        if (dest ? (vmm_alloc(&rb, &rbh, &rbm, rbn) == 0)
                 : (hipHostMalloc(&rb, rbn, hipHostMallocDefault) == hipSuccess)) {
            drd = device_read_gbs(rb, rbn);
            if (dest) {
                hipMemUnmap(rb, rbm);
                hipMemRelease((hipMemGenericAllocationHandle_t)rbh);
                hipMemAddressFree(rb, rbm);
            } else {
                hipHostFree(rb);
            }
        }
        printf("  readback: ptr %p  aligned %s  hipPointerGetAttributes type=%s  "
               "device-read %.1f GB/s over 256 MiB\n",
               (void *)buf[0], ((uintptr_t)buf[0] % ALIGN) ? "NO -- O_DIRECT WILL EFAULT" : "yes",
               memtype(buf[0]), drd);

        for (int hogging = 0; hogging <= 1; hogging++) {
            pthread_t h;
            if (hogging) {
                hog_stop = 0;
                pthread_create(&h, NULL, hog_thread, NULL);
                sleep(1); // let the hog reach steady state before timing anything
            }
            for (int trial = 0; trial < TRIALS; trial++) {
                printf("  %-9s trial %d ", hogging ? "GPU-BUSY" : "gpu-idle", trial + 1);
                for (int qd = 1; qd <= MAXQD; qd *= 2) {
                    read_errno = 0;
                    // Every cell reads a DIFFERENT offset sequence, so no cell can be
                    // warmed by the one before it.
                    trial_salt = (unsigned)(trial * 131071 + qd * 7919 + hogging * 65521);
                    double frac = 0;
                    double g = run(qd, 96 / qd, &frac);
                    if (read_errno)
                        printf(" K%d pread:%s", qd, strerror(read_errno));
                    else if (frac < 0.9)
                        printf(" K%d %5.2f !CACHED disk=%.0f%%", qd, g, frac * 100);
                    else
                        printf(" K%d %5.2f GB/s (%4.2f ms/exp)", qd, g,
                               EXPERT / 1e9 / g * 1e3);
                }
                printf("\n");
            }
            if (hogging) { hog_stop = 1; pthread_join(h, NULL); }
        }

        for (int i = 0; i < MAXQD; i++) {
            if (dest) {
                hipMemUnmap(buf[i], vmm_m[i]);
                hipMemRelease((hipMemGenericAllocationHandle_t)vmm_h[i]);
                hipMemAddressFree(buf[i], vmm_m[i]);
            } else {
                hipHostFree(buf[i]);
            }
        }
    }
    printf("\nthreshold: VMM >= 11.4 GB/s at the engine's K (>=4) means --direct-vmm-dma is\n"
           "worth recovering; ~5.7 reproduces 2026-07-30 and closes the question.\n");
    return 0;
}
