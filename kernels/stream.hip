// io_uring O_DIRECT batched reader — the cold-expert streaming engine. A single
// NVMe read at a time is latency-bound (~4 GB/s on this box); io_uring keeps the
// queue full and the same NVMe delivers ~16 GB/s (QD>=4, docs/probes/iouring_vmm.cpp).
// So a MoE layer submits ALL its cold reads at once, straight into the VMM slots
// (device-local, host-mapped), and joins once — folding the old mmap-warm + memcpy
// into one overlapped DMA stream.
//
// A thin C ABI over liburing (on-box). The caller (src/stream.rs) does all the
// O_DIRECT alignment math and owns the O_DIRECT fds + destination buffers; this
// only drives the ring. Not HIP, but compiled into librivolikernels.a like the
// rest (build.rs links -luring). Return codes mirror the rest: 0 ok, negative
// -errno / -(cqe res).
#include <liburing.h>
#include <cstdlib>

// Create a ring sized for `entries` in-flight reads. Returns an opaque handle or
// null on failure.
extern "C" void* rivoli_ring_new(unsigned entries) {
    struct io_uring* r = (struct io_uring*)malloc(sizeof(struct io_uring));
    if (!r) return nullptr;
    if (io_uring_queue_init(entries, r, 0) < 0) {
        free(r);
        return nullptr;
    }
    return r;
}

// Queue one O_DIRECT read: `nbytes` from file `fd` at `off` into `buf`. All three
// MUST be block-aligned (the caller guarantees it). Returns 0, or -1 if the
// submission queue is full (caller must rivoli_ring_wait to drain, then retry).
extern "C" int rivoli_ring_read(void* ring, int fd, void* buf, unsigned nbytes,
                                unsigned long long off, unsigned long long user_data) {
    struct io_uring* r = (struct io_uring*)ring;
    struct io_uring_sqe* s = io_uring_get_sqe(r);
    if (!s) return -1;
    io_uring_prep_read(s, fd, buf, nbytes, off);
    io_uring_sqe_set_data64(s, user_data);
    return 0;
}

// Submit all queued reads and wait for exactly `count` completions, draining them.
// Returns 0 if all succeeded, else the first negative completion result (-errno)
// or a submit error. `count` must equal the number of reads queued since the last
// wait.
extern "C" int rivoli_ring_wait(void* ring, unsigned count) {
    struct io_uring* r = (struct io_uring*)ring;
    if (count == 0) return 0;
    int sub = io_uring_submit_and_wait(r, count);
    if (sub < 0) return sub;
    int err = 0;
    for (unsigned i = 0; i < count; i++) {
        struct io_uring_cqe* c;
        int w = io_uring_wait_cqe(r, &c);
        if (w < 0) {
            if (err == 0) err = w;
            break;
        }
        if (c->res < 0 && err == 0) err = c->res; // first failing read
        io_uring_cqe_seen(r, c);
    }
    return err;
}

extern "C" void rivoli_ring_free(void* ring) {
    struct io_uring* r = (struct io_uring*)ring;
    io_uring_queue_exit(r);
    free(r);
}
