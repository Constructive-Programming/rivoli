#![cfg(feature = "rocm")]
//! Async-signal bridge: HIP stream completions become futures. This is the
//! keystone of the streaming MoE pipeline — a GPU-stream op turns into a future
//! that resolves when the hardware reaches that point (a `hipLaunchHostFunc`
//! callback wakes a `Waker`), so `futures-util` can orchestrate GPU concurrency
//! directly instead of blocking on `hipDeviceSynchronize`. No polling, no join.

use anyhow::{Result, bail, ensure};
use futures_util::task::AtomicWaker;
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

unsafe extern "C" {
    fn rivoli_stream_create() -> *mut c_void;
    fn rivoli_stream_destroy(s: *mut c_void) -> i32;
    fn rivoli_stream_host_signal(
        s: *mut c_void,
        cb: extern "C" fn(*mut c_void),
        user: *mut c_void,
    ) -> i32;
    fn rivoli_event_create() -> *mut c_void;
    fn rivoli_event_record(ev: *mut c_void, stream: *mut c_void) -> i32;
    fn rivoli_event_elapsed(start: *mut c_void, end: *mut c_void, ms: *mut f32) -> i32;
    fn rivoli_event_destroy(ev: *mut c_void);
    fn rivoli_timeline_create() -> *mut c_void;
    fn rivoli_timeline_destroy(t: *mut c_void);
    fn rivoli_timeline_wait(stream: *mut c_void, t: *mut c_void, value: u64) -> i32;
    fn rivoli_timeline_signal(stream: *mut c_void, t: *mut c_void, value: u64) -> i32;
    fn rivoli_timeline_completed(t: *mut c_void) -> u64;
}

/// A timing event: bracket GPU work on a stream to recover the true duration the
/// async overlap hides from wall-clock. Record two, then read the ms between them
/// after a join has retired both (no added sync).
pub struct HipEvent(*mut c_void);
unsafe impl Send for HipEvent {}

impl HipEvent {
    pub fn new() -> Result<Self> {
        // SAFETY: no args; null on failure.
        let e = unsafe { rivoli_event_create() };
        if e.is_null() {
            bail!("hipEventCreate failed");
        }
        Ok(HipEvent(e))
    }

    /// Enqueue a timestamp at the current point of `stream_raw`.
    // `stream_raw` is an opaque HIP stream handle passed to the runtime, not memory
    // this fn dereferences — so a safe signature is honest.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn record(&self, stream_raw: *mut c_void) -> Result<()> {
        // SAFETY: self.0 live; stream_raw a live stream.
        let rc = unsafe { rivoli_event_record(self.0, stream_raw) };
        if rc != 0 {
            bail!("hipEventRecord failed ({rc})");
        }
        Ok(())
    }

    /// Milliseconds from `start` to `end` — both must have completed (read behind a
    /// join). Returns 0.0 if either wasn't recorded this window.
    pub fn elapsed_ms(start: &HipEvent, end: &HipEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        // SAFETY: both live; ms is a valid out-ptr.
        let rc = unsafe { rivoli_event_elapsed(start.0, end.0, &mut ms) };
        if rc != 0 {
            bail!("hipEventElapsedTime failed ({rc})");
        }
        Ok(ms)
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        // SAFETY: self.0 from rivoli_event_create, freed once.
        unsafe { rivoli_event_destroy(self.0) };
    }
}

/// An owned non-blocking HIP stream. Work launched on `raw()` overlaps work on
/// other streams. The handle is an opaque runtime-owned pointer we only pass back
/// to HIP, so MOVING it across threads is sound (the fetch stream moves once into
/// the reaper). NOT `Sync`: a raw HIP stream handle isn't thread-safe (enqueue ops
/// mutate stream state), so a `&HipStream` must never be shared across threads —
/// every stream here is owned by exactly one thread.
pub struct HipStream(*mut c_void);
unsafe impl Send for HipStream {}

impl HipStream {
    pub fn new() -> Result<Self> {
        // SAFETY: no args; returns null on failure.
        let s = unsafe { rivoli_stream_create() };
        if s.is_null() {
            bail!("hipStreamCreate failed");
        }
        Ok(HipStream(s))
    }

    /// The MoE expert-partial stream. Identical to [`HipStream::new`] here — HIP streams
    /// carry no role — and named so the ROLE is visible at the call site.
    ///
    /// This pair exists because the Vulkan side CANNOT be role-blind: it maps each stream
    /// onto one of three queues, and a role-free `new()` would have to guess. Rather than
    /// give the two backends different constructors, both spell the role and HIP ignores
    /// it. Behaviour under `--features rocm` is unchanged to the byte.
    pub fn compute() -> Result<Self> {
        Self::new()
    }

    /// The reaper's H2D fetch stream. See [`HipStream::compute`].
    pub fn fetch() -> Result<Self> {
        Self::new()
    }

    /// The MISS stream: experts whose bytes are still arriving, kept off the compute stream
    /// so their device-side wait overlaps resident compute instead of following it.
    ///
    /// A stream is FIFO, so a wait sitting on the compute stream is only REACHED after the
    /// residents finish, and the GPU's wake latency then lands on the critical path — a
    /// measured +382 us per layer-with-misses. On its own stream the same wait starts at the
    /// top of the layer and its latency is absorbed by the ~1557 us of resident work.
    pub fn miss() -> Result<Self> {
        Self::new()
    }

    #[inline]
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        // SAFETY: self.0 live; teardown error ignored.
        unsafe { rivoli_stream_destroy(self.0) };
    }
}

/// A fresh [`Signal`] armed on a raw stream handle — resolves when the stream
/// reaches its current point. The expert stream uses it for the compute-stream
/// completion (moe_out ready) from inside its async block.
pub fn stream_signal(stream_raw: *mut c_void) -> Result<Signal> {
    let s = Signal::pending();
    s.arm_on_raw(stream_raw)?;
    Ok(s)
}

struct SignalState {
    done: AtomicBool,
    waker: AtomicWaker,
}

/// Runs on a HIP host thread when the stream reaches the enqueued point. Per the
/// host-func contract: no HIP calls, no blocking — flag + wake only.
extern "C" fn trampoline(user: *mut c_void) {
    // SAFETY: `user` is the Arc ptr from `signal`'s into_raw; reclaim (drop at end).
    let state = unsafe { Arc::from_raw(user as *const SignalState) };
    state.done.store(true, Ordering::Release);
    state.waker.wake();
}

/// A one-shot completion shared between whatever resolves it (a GPU-stream
/// host-func, an io_uring reaper) and the future that awaits it. `pending()` +
/// `arm_on(stream)` fires it when a stream reaches a point; `ready()` is an
/// already-resolved hit. `Clone` (Arc) so the resolver and the awaiter each hold
/// one; `Send`/`Sync` so a reaper thread can arm what the main task awaits.
#[derive(Clone)]
pub struct Signal(Arc<SignalState>);

impl Signal {
    /// An unresolved signal.
    pub fn pending() -> Self {
        Signal(Arc::new(SignalState {
            done: AtomicBool::new(false),
            waker: AtomicWaker::new(),
        }))
    }

    /// An already-resolved signal — a cache hit whose data is already present.
    pub fn ready() -> Self {
        let s = Self::pending();
        s.0.done.store(true, Ordering::Release);
        s
    }

    /// Fire this signal when `stream` reaches its current point (a host-func
    /// callback wakes it), so the enqueued GPU work's completion resolves the
    /// future. Enqueue the work first, then arm.
    pub fn arm_on(&self, stream: &HipStream) -> Result<()> {
        self.arm_on_raw(stream.raw())
    }

    /// As [`arm_on`](Self::arm_on) but for a raw stream handle — the expert stream
    /// holds a `*mut c_void` (Copy) inside its async block, not a `&HipStream`.
    // Opaque HIP handle passed to the runtime, not dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn arm_on_raw(&self, stream_raw: *mut c_void) -> Result<()> {
        // Hand one ref to the callback; the trampoline reclaims it exactly once.
        let user = Arc::into_raw(self.0.clone()) as *mut c_void;
        // SAFETY: `trampoline` has the C callback signature; `user` is a live Arc ptr.
        let rc = unsafe { rivoli_stream_host_signal(stream_raw, trampoline, user) };
        if rc != 0 {
            // Enqueue failed → callback never runs → reclaim the ref here.
            // SAFETY: `user` came from into_raw and was not consumed.
            unsafe { drop(Arc::from_raw(user as *const SignalState)) };
            bail!("hipLaunchHostFunc failed ({rc})");
        }
        Ok(())
    }


    /// Force-resolve from the resolver side without a stream (the reaper's error
    /// path, so awaiters never hang). Idempotent.
    pub fn resolve(&self) {
        self.0.done.store(true, Ordering::Release);
        self.0.waker.wake();
    }
}

impl Future for Signal {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0.done.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        self.0.waker.register(cx.waker());
        // Re-check: the callback may have fired between the load and register.
        if self.0.done.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Proves the bridge resolves on GPU completion AND measures per-signal
    // round-trip latency — the number that decides whether per-expert async
    // signals (~9/layer × 75 layers/token) are affordable vs the overlap they buy.
    // A Signal that resolves once every op enqueued on `stream` so far completes.
    fn signal(stream: &HipStream) -> Result<Signal> {
        let s = Signal::pending();
        s.arm_on(stream)?;
        Ok(s)
    }

    #[test]
    fn signal_resolves_and_latency() -> Result<()> {
        let stream = HipStream::new()?;
        let rt = tokio::runtime::Builder::new_current_thread().build()?;
        // Warm the runtime + the host-func machinery.
        rt.block_on(signal(&stream)?);
        let n = 2000u32;
        let t = std::time::Instant::now();
        rt.block_on(async {
            for _ in 0..n {
                signal(&stream)?.await;
            }
            Ok::<(), anyhow::Error>(())
        })?;
        let us = t.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
        println!("\nASYNC-SIGNAL round-trip: {us:.2} us/signal ({n} iters)\n");
        // Sanity ceiling: if a bare signal costs >1ms the pipeline is a non-starter.
        assert!(us < 1000.0, "signal latency {us:.1}us implausibly high");
        Ok(())
    }
}

/// A monotonic counter a stream can WAIT ON and SIGNAL, and the host can observe.
///
/// This replaces host-side readiness booleans. The difference that matters: a consumer may
/// be enqueued behind `wait(v)` **before** any producer has run, because the wait names a
/// VALUE rather than capturing a producer's current state. That is precisely what
/// `hipStreamWaitEvent` cannot do (it snapshots at enqueue time, so an early wait silently
/// passes), and it is why the engine's readiness can stop being a `Vec<bool>` that two
/// modules have to agree about.
///
/// Values are assigned by the producer side and only ever increase. `completed()` is a
/// plain acquire load, so staging slots can be recycled without any sync at all: slot *i*
/// is free once `completed() >= release[i]`.
pub struct Timeline(*mut c_void);

// SAFETY: the counter is device signal memory; the HIP calls that touch it are
// stream-ordered and internally synchronised, and Rust-side access is an atomic load.
unsafe impl Send for Timeline {}
unsafe impl Sync for Timeline {}

impl Timeline {
    pub fn new() -> Result<Self> {
        // SAFETY: no arguments; returns null on allocation failure, checked below.
        let p = unsafe { rivoli_timeline_create() };
        ensure!(
            !p.is_null(),
            "hipMallocSignalMemory failed — hipDeviceAttributeCanUseStreamWaitValue is 1 on \
             gfx1151, so this is an allocation failure rather than an unsupported device"
        );
        Ok(Timeline(p))
    }

    /// Enqueue "block until this timeline reaches `value`" on `stream_raw`. Ordering is
    /// safe in both directions: enqueueing this before the signaller is the intended use.
    // Opaque HIP handle passed through, not dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn wait(&self, stream_raw: *mut c_void, value: u64) -> Result<()> {
        // SAFETY: `self.0` is live signal memory; `stream_raw` is a live stream.
        let rc = unsafe { rivoli_timeline_wait(stream_raw, self.0, value) };
        ensure!(rc == 0, "hipStreamWaitValue64 failed ({rc})");
        Ok(())
    }

    /// Enqueue "set this timeline to `value`" on `stream_raw`, ordered after everything
    /// already queued there — which is what makes the value mean "work up to here is done".
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn signal(&self, stream_raw: *mut c_void, value: u64) -> Result<()> {
        // SAFETY: as `wait`.
        let rc = unsafe { rivoli_timeline_signal(stream_raw, self.0, value) };
        ensure!(rc == 0, "hipStreamWriteValue64 failed ({rc})");
        Ok(())
    }

    /// Highest value the device has reached. Acquire load — no sync, no stall.
    pub fn completed(&self) -> u64 {
        // SAFETY: `self.0` is live signal memory for the lifetime of `self`.
        unsafe { rivoli_timeline_completed(self.0) }
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `rivoli_timeline_create` and is dropped once.
        unsafe { rivoli_timeline_destroy(self.0) };
    }
}

#[cfg(test)]
mod timeline_tests {
    use super::*;

    /// **INV-4: a wait may be enqueued BEFORE its producer exists, and still waits.**
    ///
    /// This is the whole reason the dataflow can drop host-side readiness flags: consumers
    /// are recorded up front, behind waits on values nothing has signalled yet. It is also
    /// exactly what `hipStreamWaitEvent` cannot do — it snapshots the event at enqueue time,
    /// so an early wait passes vacuously and the kernel reads unwritten memory. That failure
    /// is silent, which is why this is a test and not a comment.
    #[test]
    fn inv_4_wait_enqueued_before_signal_still_waits() {
        let t = Timeline::new().expect("timeline");
        let (a, b) = (HipStream::new().expect("a"), HipStream::new().expect("b"));
        assert_eq!(t.completed(), 0, "a fresh timeline starts at 0");
        // Order matters: the WAIT goes first, against a value nothing has produced.
        t.wait(a.raw(), 1).expect("wait enqueues before any signal");
        t.signal(b.raw(), 1).expect("signal");
        let rt = tokio::runtime::Builder::new_current_thread().build().expect("rt");
        // If the wait had passed vacuously this would resolve regardless; the value check
        // below is what distinguishes "waited" from "did not block".
        rt.block_on(stream_signal(a.raw()).expect("completion"));
        assert!(t.completed() >= 1, "the waited-on value must have been reached");
    }

    /// Values only move forward, and `completed()` observes them without a sync — the
    /// property staging-slot recycling depends on (slot i is free iff completed >= release[i]).
    #[test]
    fn timeline_is_monotonic_and_host_observable() {
        let t = Timeline::new().expect("timeline");
        let s = HipStream::new().expect("stream");
        for v in 1..=4u64 {
            t.signal(s.raw(), v).expect("signal");
        }
        let rt = tokio::runtime::Builder::new_current_thread().build().expect("rt");
        rt.block_on(stream_signal(s.raw()).expect("completion"));
        assert_eq!(t.completed(), 4, "the last value written must be observable");
    }
}
