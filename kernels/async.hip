// HIP stream + host-func async-signal bridge for the streaming MoE pipeline.
// The engine was single-null-stream + hipDeviceSynchronize; this adds the surface
// the futures pipeline needs: independent streams (so load DMA and compute overlap
// on the GPU) and a completion callback that wakes a Rust future when a stream
// reaches a point — no polling, no blocking join.
//
// The callback `cb(user)` is a Rust extern "C" fn; HIP runs it on a host thread
// once all prior work on the stream completes. Per the HIP contract it MUST NOT
// call any HIP API and must not block — ours only sets a flag + wakes a Waker.
#include <hip/hip_runtime.h>

#define HIP_ERR_BASE 1000000  // negative-return encoding shared with the other shims

// A non-blocking stream: work on it never implicitly serialises against the null
// stream, so a compute launch here truly overlaps a copy on another stream.
extern "C" void* rivoli_stream_create() {
    hipStream_t s = nullptr;
    if (hipStreamCreateWithFlags(&s, hipStreamNonBlocking) != hipSuccess) return nullptr;
    return (void*)s;
}

extern "C" int rivoli_stream_destroy(void* s) {
    return hipStreamDestroy((hipStream_t)s) == hipSuccess ? 0 : -1;
}

// Enqueue a host callback at the current point of `stream`: `cb(user)` fires once
// every prior op on the stream has completed. This is the async signal — the Rust
// side turns it into a resolved future.
extern "C" int rivoli_stream_host_signal(void* stream, void (*cb)(void*), void* user) {
    hipError_t e = hipLaunchHostFunc((hipStream_t)stream, cb, user);
    return e == hipSuccess ? 0 : HIP_ERR_BASE - (int)e;
}

// --- Timing events: bracket GPU work on a stream to recover accurate durations the
//     async overlap blurs out of wall-clock. `record` enqueues a timestamp on the
//     stream; `elapsed` reads the ms between two recorded events (both must have
//     completed — read it behind a join the forward already pays, so no added sync).
extern "C" void* rivoli_event_create() {
    hipEvent_t e = nullptr;
    if (hipEventCreate(&e) != hipSuccess) return nullptr;
    return (void*)e;
}

extern "C" int rivoli_event_record(void* ev, void* stream) {
    hipError_t e = hipEventRecord((hipEvent_t)ev, (hipStream_t)stream);
    return e == hipSuccess ? 0 : HIP_ERR_BASE - (int)e;
}

// Milliseconds from `start` to `end` (both recorded + completed) into *ms.
extern "C" int rivoli_event_elapsed(void* start, void* end, float* ms) {
    hipError_t e = hipEventElapsedTime(ms, (hipEvent_t)start, (hipEvent_t)end);
    return e == hipSuccess ? 0 : HIP_ERR_BASE - (int)e;
}

extern "C" void rivoli_event_destroy(void* ev) {
    hipEventDestroy((hipEvent_t)ev);
}

// --- io_uring bounce arena (src/stream.rs owns the ring via the io-uring crate; the
//     two HIP ops stay here, next to the other rivoli_* HIP wrappers). O_DIRECT reads
//     land in this PINNED host arena (get_user_pages OK on pinned host — the amdgpu
//     kernel-bug workaround from stream.rs), then an async H2D copy fills the VMM slot.
extern "C" void* rivoli_pinned_alloc(unsigned long long bytes) {
    void* p = nullptr;
    if (hipHostMalloc(&p, (size_t)bytes, hipHostMallocDefault) != hipSuccess) return nullptr;
    return p;
}

extern "C" void rivoli_pinned_free(void* p) {
    hipHostFree(p);
}

// Async host->device copy on `stream`: bounce slot -> VMM expert slot. The caller arms
// a signal on the same stream, so the load future resolves when the COPY lands (not
// merely the NVMe read). Returns 0 or HIP_ERR_BASE - hipError.
extern "C" int rivoli_memcpy_h2d_async(void* dst, const void* src, unsigned long long n,
                                       void* stream) {
    hipError_t e = hipMemcpyAsync(dst, src, (size_t)n, hipMemcpyHostToDevice,
                                  (hipStream_t)stream);
    return e == hipSuccess ? 0 : HIP_ERR_BASE - (int)e;
}
