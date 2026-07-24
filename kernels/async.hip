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
