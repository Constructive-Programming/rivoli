// io_uring O_DIRECT batched reader — the cold-expert streaming engine. A single
// NVMe read at a time is latency-bound (~4 GB/s on this box); io_uring keeps the
// queue full and the same NVMe delivers more at depth (QD>=4). So a MoE layer
// submits ALL its cold reads at once and joins once, folding the old mmap-warm +
// memcpy into one overlapped stream.
//
// TWO DESTINATION MODES, chosen once at rivoli_ring_new:
//
//   DIRECT (bounce=0, `--direct-vmm-dma`): prep_read straight into the caller's
//   VMM device slot — the read DMA lands in device memory, no copy.
//
//   BOUNCE (bounce=1, the default): read into a PINNED HOST arena, then
//   hipMemcpy each into the VMM slot.
//   ------------------------------------------------------------------------
//   WORKAROUND (kernel bug, amdgpu, first seen 6.18.38-gentoo 2026-07-17):
//     io_uring / O_DIRECT DMA the read into the destination via get_user_pages,
//     which this kernel's amdgpu path REFUSES for VMM device-local host-mapped
//     pages — EFAULT (-14) on EVERY read. Isolated repro: docs/probes/
//     iouring_vmm.cpp faults into VMM; the same reads into a plain pinned/aligned
//     host buffer succeed (iou_host probe). Earlier kernels (<=6.18.35-r1) DMA'd
//     into VMM fine — a regression, not our design. The bounce reads into pinned
//     host (get_user_pages OK) and copies host->device into the VMM slot (220 GB/s
//     device-fill, ~free vs the NVMe read), keeping O_DIRECT + QD without ever
//     DMA'ing into VMM.
//     STATUS 2026-07-21: the EFAULT is GONE on this same kernel (6.18.38-gentoo) —
//     `--direct-vmm-dma` runs clean, with and without O_DIRECT. But do NOT flip the
//     default: DIRECT is now the slower arm for a DIFFERENT reason. Measured at 512
//     tokens, identical cache counters, only the destination varying:
//       O_DIRECT + bounce  fetch 506ms (4.17 ms/miss)  mlp  76ms  1.43 tok/s
//       O_DIRECT + VMM     fetch 560ms (4.62 ms/miss)  mlp 108ms  1.27 tok/s
//     The bounce's H2D is not waste — it is what puts the weights in DEVICE-LOCAL
//     memory. Reading them from host-mapped VMM instead costs ~40% on `mlp`, the
//     system-vs-device-domain read tax documented in docs/hip-apu-memory.md, which
//     is far more than the H2D copy it saves. Keep the bounce.
//   ------------------------------------------------------------------------
//
// A thin C ABI over liburing (on-box). The caller (src/stream.rs) does all the
// O_DIRECT alignment math and owns the O_DIRECT fds + VMM destination pointers.
// Compiled into librivolikernels.a like the rest (build.rs links -luring). Return
// codes: 0 ok, negative -errno / -(cqe res), -EIO short read, HIP_ERR_BASE-(hipError)
// on a bounce-copy failure.
#include <liburing.h>
#include <hip/hip_runtime.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>

// hipError_t codes are small positive ints; offset them well below -errno space so
// a copy failure is distinguishable from an io_uring/-errno failure in the log.
#define HIP_ERR_BASE (-100000)

struct Ring {
    struct io_uring uring;
    unsigned char* bounce;      // pinned host arena (bounce mode): entries*span; else null
    void** dst;                 // per-read VMM dst (bounce mode), indexed by user_data
    unsigned* nbytes;           // per-read aligned superset length (bounce mode)
    unsigned span;              // per-read bounce stride (max aligned superset)
    unsigned entries;
    int bounce_mode;            // 0 = DMA straight into dst; 1 = pinned-host bounce
};

// Create a ring for `entries` in-flight reads, each up to `span` bytes. If
// `bounce` != 0, allocate an `entries * span` pinned host arena + per-read
// bookkeeping (the kernel-bug workaround); otherwise reads DMA straight into the
// caller's VMM dst. Returns an opaque handle or null on failure.
//
// `sqpoll` != 0 asks the kernel to run a submission-poller thread, so submission
// becomes a shared-memory tail update with NO io_uring_enter syscall on the
// caller's thread. Measured, a plain submit of one expert's 6 O_DIRECT reads
// blocks the submitting thread ~2.96 ms — 61% of the 4.85 ms the read itself
// takes — because that thread drives the btrfs/blk-mq dispatch INLINE, one SQE
// at a time. Two consequences, and the second was missed for a long time:
//   1. an "async" submit is not async (this is the prefetch bug, fixed 2026-07-21);
//   2. the reads never reach the device CONCURRENTLY, because the inline dispatch
//      is serial. A batch of 6 SQEs behaves like queue depth 1. That is why the
//      demand path measured 6.97 ms per 19 MB expert — the array's P=1 rate
//      (2.53 GB/s) — while the same array does 6.69 GB/s at P>=4.
// So SQPOLL is wanted on EVERY ring that submits a batch it wants run in
// parallel, not just on rings whose submit must return immediately.
//
// `attach_fd` >= 0 asks this ring to SHARE that (already SQPOLL) ring's poller
// thread via IORING_SETUP_ATTACH_WQ, rather than starting a second kernel thread
// that spins on this box's already bandwidth-tight LPDDR5X. Fallback ladder:
// attached poller -> own poller -> plain ring.
extern "C" void* rivoli_ring_new(unsigned entries, unsigned span, int bounce, int sqpoll,
                                 int attach_fd) {
    Ring* r = (Ring*)calloc(1, sizeof(Ring));
    if (!r) return nullptr;
    int ok = -1;
    if (sqpoll && attach_fd >= 0) {
        struct io_uring_params p;
        memset(&p, 0, sizeof(p));
        p.flags = IORING_SETUP_SQPOLL | IORING_SETUP_ATTACH_WQ;
        p.sq_thread_idle = 2000;  // ms before the poller sleeps; re-armed on submit
        p.wq_fd = (unsigned)attach_fd;
        ok = io_uring_queue_init_params(entries, &r->uring, &p);
    }
    if (sqpoll && ok < 0) {
        struct io_uring_params p;
        memset(&p, 0, sizeof(p));
        p.flags = IORING_SETUP_SQPOLL;
        p.sq_thread_idle = 2000;
        ok = io_uring_queue_init_params(entries, &r->uring, &p);
    }
    if (ok < 0 && io_uring_queue_init(entries, &r->uring, 0) < 0) { free(r); return nullptr; }
    r->span = span;
    r->entries = entries;
    r->bounce_mode = bounce;
    if (bounce) {
        if (hipHostMalloc((void**)&r->bounce, (size_t)entries * span, hipHostMallocDefault)
                != hipSuccess) {
            io_uring_queue_exit(&r->uring); free(r); return nullptr;
        }
        r->dst = (void**)malloc((size_t)entries * sizeof(void*));
        r->nbytes = (unsigned*)malloc((size_t)entries * sizeof(unsigned));
        if (!r->dst || !r->nbytes) {
            hipHostFree(r->bounce); io_uring_queue_exit(&r->uring);
            free(r->dst); free(r->nbytes); free(r); return nullptr;
        }
    }
    return r;
}

// This ring's io_uring fd, for passing as another ring's `attach_fd` so the two
// share one submission-poller thread. Valid until rivoli_ring_free.
extern "C" int rivoli_ring_fd(void* ring) {
    return ((Ring*)ring)->uring.ring_fd;
}

// Queue one O_DIRECT read: `nbytes` from `fd` at `off`. DIRECT mode reads straight
// into `dst` (the VMM slot). BOUNCE mode reads into the bounce slot for `user_data`
// (< entries) and remembers `dst` for the post-wait copy. `off`/`nbytes` MUST be
// block-aligned (caller guarantees) and, in bounce mode, `nbytes <= span`. Returns
// 0, -1 if the SQ is full (drain then retry), -2 on a bad argument.
extern "C" int rivoli_ring_read(void* ring, int fd, void* dst, unsigned nbytes,
                                unsigned long long off, unsigned long long user_data) {
    Ring* r = (Ring*)ring;
    void* into = dst;
    if (r->bounce_mode) {
        if (user_data >= r->entries || nbytes > r->span) return -2;
        into = r->bounce + user_data * (size_t)r->span;
    }
    struct io_uring_sqe* s = io_uring_get_sqe(&r->uring);
    if (!s) return -1;
    io_uring_prep_read(s, fd, into, nbytes, off);
    io_uring_sqe_set_data64(s, user_data);
    if (r->bounce_mode) {
        r->dst[user_data] = dst;
        r->nbytes[user_data] = nbytes;
    }
    return 0;
}

// Submit the queued SQEs to the kernel WITHOUT waiting (the reads start running
// now — used by the prefetch ring so cross-layer reads proceed on the NVMe/DMA
// side during the current layer's GPU compute). A later `rivoli_ring_wait` reaps
// the same completions (its own submit is then a no-op, the SQ already drained).
// Returns the number of SQEs submitted (>= 0) or a negative -errno.
extern "C" int rivoli_ring_submit(void* ring) {
    Ring* r = (Ring*)ring;
    return io_uring_submit(&r->uring);
}

// Submit all queued reads, wait for `count` completions, and — in bounce mode, only
// if every read succeeded — hipMemcpy each bounce slot into its VMM dst (async on
// the null stream, one sync). Returns 0, or the first negative completion result
// (-errno), -EIO for a real mid-file short read, a submit error, or
// HIP_ERR_BASE-(hipError) on a copy failure. `count` must equal the reads queued
// since the last wait.
//
// `min_res` (len `count`, indexed by user_data) is the minimum byte count each
// completion MUST deliver — the sub-block offset plus the useful [begin,begin+len)
// length. O_DIRECT reads the aligned SUPERSET, which may run past EOF on a shard's
// last tensor, so a short completion is legitimate ONLY when it still covers the
// useful window (`res >= min_res[ud]`); shorter is a real truncation and errors.
// `min_res` may be null to skip the check.
// Nanosecond split of the last `rivoli_ring_wait`: [0] = waiting on the NVMe
// completions, [1] = the bounce H2D copy + its device sync. `fetch` is 73% of a
// token and both halves live inside it, so which one to attack is not guessable —
// the H2D moves ~2.7 GB/token and the reads are latency-bound, and those want
// opposite fixes. Accumulated, never reset by the shim; the caller reads deltas.
static unsigned long long g_read_ns = 0;
static unsigned long long g_copy_ns = 0;

static inline unsigned long long now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000ull + (unsigned long long)t.tv_nsec;
}

// Read the accumulated (read, copy) nanosecond totals.
extern "C" void rivoli_ring_timings(unsigned long long* read_ns, unsigned long long* copy_ns) {
    if (read_ns) *read_ns = g_read_ns;
    if (copy_ns) *copy_ns = g_copy_ns;
}

extern "C" int rivoli_ring_wait(void* ring, unsigned count,
                                const unsigned long long* min_res) {
    Ring* r = (Ring*)ring;
    if (count == 0) return 0;
    unsigned long long t0 = now_ns();
    int sub = io_uring_submit_and_wait(&r->uring, count);
    if (sub < 0) return sub;
    int err = 0;
    for (unsigned i = 0; i < count; i++) {
        struct io_uring_cqe* c;
        int w = io_uring_wait_cqe(&r->uring, &c);
        if (w < 0) { if (err == 0) err = w; break; }
        long long res = c->res;
        unsigned long long ud = io_uring_cqe_get_data64(c);
        if (res < 0) {
            if (err == 0) err = (int)res; // first failing read (-errno)
        } else if (min_res && (unsigned long long)res < min_res[ud]) {
            // Short read not covering the useful window — real truncation, not
            // benign EOF padding of the aligned superset.
            if (err == 0) err = -EIO;
        }
        io_uring_cqe_seen(&r->uring, c);
    }
    g_read_ns += now_ns() - t0;
    if (err != 0 || !r->bounce_mode) return err; // direct mode: reads landed in VMM
    unsigned long long t1 = now_ns();
    // Bounce mode: DMA each pinned-host slot into its VMM device slot. Pinned source
    // => async overlaps the copies; one sync joins them.
    for (unsigned i = 0; i < count; i++) {
        hipError_t he = hipMemcpyAsync(r->dst[i], r->bounce + (size_t)i * r->span,
                                       r->nbytes[i], hipMemcpyHostToDevice, 0);
        if (he != hipSuccess) return HIP_ERR_BASE - (int)he;
    }
    hipError_t he = hipDeviceSynchronize();
    g_copy_ns += now_ns() - t1;
    return he == hipSuccess ? 0 : HIP_ERR_BASE - (int)he;
}

extern "C" void rivoli_ring_free(void* ring) {
    Ring* r = (Ring*)ring;
    if (r->bounce_mode) {
        hipHostFree(r->bounce);
        free(r->dst);
        free(r->nbytes);
    }
    io_uring_queue_exit(&r->uring);
    free(r);
}
