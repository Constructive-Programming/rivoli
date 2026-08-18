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

// hipError_t → a distinct nonzero return the Rust side treats as an error (it only
// tests rc != 0; the exact value just has to not collide with 0 or an -errno).
#define HIP_ERR_BASE 1000000

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
// `coherent != 0` requests FINE-GRAINED (system-coherent) host memory instead of whatever
// `hipHostMallocDefault` yields on this runtime. It is a CANDIDATE FIX under evaluation, not a
// knob, and the argument is an ordering gap the rest of this file never addresses:
//
//   1. io_uring O_DIRECT has the NVMe DMA write this arena. The CQE establishes that those writes
//      are visible TO THE CPU — the kernel's completion path carries the barriers for that.
//   2. `rivoli_memcpy_h2d_async` below then reads the arena from the GPU side (copy engine or
//      blit), with NOTHING in between (`fetch/stream.rs::reap` consumes the CQE and enqueues the
//      copy back to back).
//   3. **Nothing establishes that the agent performing (2) observes the writes from (1).** The CPU
//      is neither the producer nor the consumer, so the CQE's guarantee is about the wrong agent.
//
// For fine-grained host memory the platform provides that guarantee by snooping. For coarse-grained
// it does not, and an acquire is required at a synchronisation point — and a stale line is exactly
// what the reuse pattern arranges, because a staging slot's previous contents were themselves read
// through the GPU on an earlier fetch.
//
// `hipHostMallocDefault` does not SAY which it gets: on ROCm that has been runtime- and
// environment-dependent. So this is not asserted, it is A/B-ed — which is why the flag exists and
// why the allocation logs what it asked for.
extern "C" void* rivoli_pinned_alloc(unsigned long long bytes, int coherent) {
    void* p = nullptr;
    unsigned flags = coherent ? hipHostMallocCoherent : hipHostMallocDefault;
    if (hipHostMalloc(&p, (size_t)bytes, flags) != hipSuccess) return nullptr;
    return p;
}

// READ BACK what the runtime actually gave us. `rivoli_pinned_alloc` says what it ASKED for; this
// says what it GOT, and the difference is the whole reason a `--pinned-coherent` arm could not be
// believed: a flag that never applied and a fix that does not work produce the same red.
//
// Returns 0 and writes `*out` on success, negative on failure. `hipHostGetFlags` is the canonical
// query — `hipPointerGetAttributes().allocationFlags` is the same information by a longer road.
extern "C" int rivoli_pinned_flags(void* p, unsigned* out) {
    unsigned f = 0;
    hipError_t e = hipHostGetFlags(&f, p);
    if (e != hipSuccess) return HIP_ERR_BASE - (int)e;
    *out = f;
    return 0;
}

// The bit `hipHostMallocCoherent` sets, exported so the Rust side does not hardcode a HIP constant
// it cannot see. A run reports the requested and returned flags and whether this bit survived.
extern "C" unsigned rivoli_pinned_coherent_bit(void) { return (unsigned)hipHostMallocCoherent; }

extern "C" void rivoli_pinned_free(void* p) {
    hipHostFree(p);
}

// Async host->device copy on `stream`: bounce slot -> VMM expert slot. The caller arms
// a signal on the same stream, so the load future resolves when the COPY lands (not
// merely the NVMe read). Returns 0 or HIP_ERR_BASE - hipError.
// COPY BY KERNEL instead of by the copy engine — Phase 3B, a candidate FIX.
//
// `hipMemcpyAsync` above dispatches to whatever path the runtime picks, typically an SDMA/blit
// engine. This does the same move as an ordinary shader read+write, so it goes through the normal
// vector-memory path with a kernel's own acquire at dispatch. The distinction is the whole
// experiment: the ablation matrix says suppression needs a device-side READ of the arena's own
// bytes, in bulk, and nothing weaker — which is what a kernel copy does and what a copy engine may
// not.
//
// `uint4` (16 B) per lane: the arena and every pool slot are `VQ_ALIGN`-aligned and the byte count
// is a multiple of the O_DIRECT block, so the vector width always divides the extent and no tail
// case exists. Grid-stride with a bounded grid, like the folds.
extern "C" __global__ void copy_h2d_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src,
                                          unsigned long long n4) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

// ONE entry point with the path as an argument, not two entry points. The two are one operation
// with a knob — the acceptance protocol switches between them by flag — and splitting them gave the
// Rust side two identical 4-parameter externs and two identical call arms, which the duplication
// gate reported twice. The branch belongs here, where both HIP calls already live.
extern "C" int rivoli_memcpy_h2d_async(void* dst, const void* src, unsigned long long n,
                                       void* stream, int by_kernel) {
    hipError_t e;
    if (by_kernel) {
        // 16 B granularity is a hard requirement, not a fast path: a remainder would be silently
        // dropped and the slot would hold a short expert. Refused loudly instead. Every caller
        // passes an O_DIRECT-block multiple, so this cannot fire in the engine.
        if (n == 0 || (n % sizeof(uint4)) != 0) return 1001;
        unsigned long long n4 = n / sizeof(uint4);
        unsigned long long blocks = (n4 + 255) / 256;
        if (blocks > 1024) blocks = 1024;
        hipGetLastError();
        hipLaunchKernelGGL(copy_h2d_kernel, dim3((unsigned)blocks), dim3(256), 0,
                           (hipStream_t)stream, (uint4*)dst, (const uint4*)src, n4);
        e = hipGetLastError();
    } else {
        e = hipMemcpyAsync(dst, src, (size_t)n, hipMemcpyHostToDevice, (hipStream_t)stream);
    }
    return e == hipSuccess ? 0 : HIP_ERR_BASE - (int)e;
}

// ---------------------------------------------------------------------------
// TIMELINE: a device-visible monotonic counter, waited on and signalled ON STREAM.
//
// This is the primitive the whole ticketed dataflow rests on, and the choice of API is
// load-bearing. `hipStreamWaitEvent` CANNOT be used here: it captures the event's state at
// the moment the wait is ENQUEUED, so a wait enqueued before the producer records its event
// is a silent no-op. Enqueuing consumers before their producers have run is the entire point
// of this design, so events would reproduce — invisibly — the very bug this replaces.
//
// `hipStreamWaitValue64` waits on a VALUE ("block until *ptr >= v"), which is well defined
// before the signaller has run. Probed on gfx1151 (2026-07-30): hipDeviceAttributeCanUse-
// StreamWaitValue = 1, and a wait enqueued BEFORE the write on another stream retires
// correctly rather than passing vacuously or deadlocking.
//
// The counter must live in signal memory (hipMallocSignalMemory), not ordinary device
// memory, or the wait has no coherent view of the writer.
extern "C" void* rivoli_timeline_create() {
  uint64_t* p = nullptr;
  if (hipExtMallocWithFlags((void**)&p, sizeof(uint64_t), hipMallocSignalMemory) != hipSuccess)
    return nullptr;
  *p = 0;
  return (void*)p;
}

extern "C" void rivoli_timeline_destroy(void* t) {
  if (t) (void)hipFree(t);
}

// Enqueue "wait until *t >= value" on `stream`. Safe to enqueue before anything signals it.
extern "C" int rivoli_timeline_wait(void* stream, void* t, unsigned long long value) {
  return (int)hipStreamWaitValue64((hipStream_t)stream, t, (uint64_t)value,
                                   hipStreamWaitValueGte, ~0ull);
}

// Enqueue "*t = value" on `stream`, ordered after everything already queued there — which
// is what makes the value mean "all work up to here has completed".
extern "C" int rivoli_timeline_signal(void* stream, void* t, unsigned long long value) {
  return (int)hipStreamWriteValue64((hipStream_t)stream, t, (uint64_t)value, 0);
}

// Host-side observation, for recycling staging slots without a sync.
extern "C" unsigned long long rivoli_timeline_completed(void* t) {
  return __atomic_load_n((unsigned long long*)t, __ATOMIC_ACQUIRE);
}

// Host-side RELEASE, for the teardown path only: force the counter to `value` so waits
// already enqueued against it retire, even though the work that would have signalled them
// will never run.
//
// This exists because a fetch error used to HANG the device. `hipStreamWaitValue64` has no
// error state — it blocks until the value arrives, forever — so when the reaper abandoned a
// poisoned ring without signalling that batch's tickets, every consumer already gated on
// them waited on a value nothing would ever write. Resolving the per-read `Signal`s (which
// is what the poison path did) released nothing: the ticket is the dependency, not the
// signal.
//
// Monotone via CAS, never a bare store: values only ever increase per slot, and dropping one
// backwards would un-release a slot some other consumer is already gated on — turning a
// clean teardown into the deadlock it exists to prevent.
extern "C" void rivoli_timeline_release(void* t, unsigned long long value) {
  unsigned long long* p = (unsigned long long*)t;
  unsigned long long cur = __atomic_load_n(p, __ATOMIC_ACQUIRE);
  while (cur < value &&
         !__atomic_compare_exchange_n(p, &cur, value, true, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE))
    ;
}
