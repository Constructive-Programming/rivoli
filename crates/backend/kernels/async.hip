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
//
// The suspected ordering gap on this path — the CQE guarantees the NVMe DMA's writes are
// visible to the CPU, while the agent that actually reads the arena is the GPU's copy path,
// and nothing orders those two — is what the GLM nondeterminism investigation localised and
// mitigated. A fine-grained (`hipHostMallocCoherent`) arena was one of its candidate fixes
// and measured RED with the flag observed to apply, so the allocation is plain
// `hipHostMallocDefault`; `docs/investigations/glm-nondeterminism-closeout.md` keeps that
// record, and `docs/investigations/glm-nondeterminism-worklog.md` the arm log.
extern "C" void* rivoli_pinned_alloc(unsigned long long bytes) {
    void* p = nullptr;
    if (hipHostMalloc(&p, (size_t)bytes, hipHostMallocDefault) != hipSuccess) return nullptr;
    return p;
}

extern "C" void rivoli_pinned_free(void* p) {
    hipHostFree(p);
}

// ARENA REFRESH — the only intervention measured to eliminate the divergence, promoted out of the
// probe and given its own kernel so it carries no hash nobody reads.
//
// WHAT IS KNOWN (2026-08-18; docs/investigations/glm-nondeterminism-worklog.md §7b): a FULL-WIDTH
// device read of the just-written arena window, enqueued on the fetch stream BEFORE the copy,
// makes GLM decode reproduce itself — four clean pairs over 6144 tokens against a rate that
// predicts P(all clean) ~ 1e-9. Everything else tried is RED: one cache line of the arena,
// stride-32 sampling, the same bulk from a decoy buffer, the bare dispatch, a kernel copy instead
// of SDMA, a fine-grained arena, a write-combined arena (each read back and confirmed to apply),
// and the staging hand-out (`slot_stalls` measured 0). The diagnostic flags that ran those arms
// were deleted 2026-08-20; docs/investigations/glm-nondeterminism-closeout.md keeps the record.
//
// WHAT IS NOT KNOWN: why. Fifteen mechanisms are eliminated and none of the survivors predicts
// "bulk AND arena-specific". So this is a MITIGATION with a measured price, not a root-cause fix,
// and it is labelled that way on purpose — the ceiling is that a future compiler, driver or cache
// geometry could stop making the read sufficient, and nothing here would notice except the
// determinism gate. Do not delete it without re-running `tests/determinism-glm.sh`.
__global__ void touch_region_kernel(const uint4* __restrict__ src, unsigned long long n4,
                                   unsigned long long* sink) {
    unsigned long long acc = 0;
    // Grid-stride over the whole region: FULL WIDTH is load-bearing — the investigation's
    // reduced-density arms (one 16 B unit per 64 B sector; one per 128 B line) were RED.
    for (unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         j < n4; j += (unsigned long long)blockDim.x * gridDim.x) {
        uint4 v = src[j];
        acc += (unsigned long long)v.x ^ (unsigned long long)v.y ^ (unsigned long long)v.z ^
               (unsigned long long)v.w;
    }
    // The store is what stops the compiler deleting the loads. The condition is never true for real
    // payloads, so the sink is never written and needs no synchronisation with anything.
    if (acc == 0xFFFFFFFFFFFFFFFFULL) {
        *sink = acc;
    }
}

// Read `n` bytes of `src` at full width on `stream`, discarding the value. `src` is the pinned
// arena window the NVMe just DMA'd into (`--arena-refresh`). The name stays neutral because the
// investigation aimed the same read at other regions before the arena won: reading the arena
// pre-copy suppresses the defect, reading the destination slot post-copy (`sc`) does NOT —
// docs/investigations/glm-nondeterminism-closeout.md keeps that matrix. 16 B granularity like
// the copy: every caller passes an O_DIRECT-block multiple.
extern "C" int rivoli_touch_region_async(const void* src, unsigned long long n, void* stream,
                                       unsigned long long* sink) {
    if (n == 0 || (n % sizeof(uint4)) != 0) return 1002;
    unsigned long long n4 = n / sizeof(uint4);
    unsigned long long blocks = (n4 + 255) / 256;
    if (blocks > 1024) blocks = 1024;
    hipGetLastError();
    hipLaunchKernelGGL(touch_region_kernel, dim3((unsigned)blocks), dim3(256), 0,
                       (hipStream_t)stream, (const uint4*)src, n4, sink);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : HIP_ERR_BASE - (int)e;
}

// Async host->device copy on `stream`: bounce slot -> VMM expert slot, `hipMemcpyAsync` (SDMA).
// The caller arms a signal on the same stream, so the load future resolves when the COPY lands
// (not merely the NVMe read). Returns 0 or HIP_ERR_BASE - hipError. A shader-copy alternative
// (`--copy-by-kernel`, Phase 3B) lived here as a candidate fix and was deleted 2026-08-20 with
// the other refuted arms — docs/investigations/glm-nondeterminism-closeout.md is the record.
extern "C" int rivoli_memcpy_h2d_async(void* dst, const void* src, unsigned long long n,
                                       void* stream) {
    hipError_t e = hipMemcpyAsync(dst, src, (size_t)n, hipMemcpyHostToDevice, (hipStream_t)stream);
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
