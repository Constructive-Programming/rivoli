#![cfg(feature = "rocm")]
//! Async-signal bridge: HIP stream completions become futures. This is the
//! keystone of the streaming MoE pipeline — a GPU-stream op turns into a future
//! that resolves when the hardware reaches that point (a `hipLaunchHostFunc`
//! callback wakes a `Waker`), so the decode loop can await GPU concurrency
//! directly instead of blocking on `hipDeviceSynchronize`. No polling, no join.
//!
//! [`Signal`] itself lives in `src/backend.rs` — it was identical on both backends. What
//! is HIP's alone, and is here, is the arming: the host-func trampoline.

use crate::backend::Signal;
use anyhow::{Result, bail, ensure};
use std::ffi::c_void;

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
    fn rivoli_timeline_release(t: *mut c_void, value: u64);
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

    // jscpd:ignore-start — mirrors `vkstream::Stream::raw`. Same name, same signature,
    // different body (a HIP stream handle here, a `Q` tag there); `backend.rs` cfg-selects
    // one of the two under the SAME path, so the signature is the contract.
    #[inline]
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
    // jscpd:ignore-end
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

/// Runs on a HIP host thread when the stream reaches the enqueued point. Per the
/// host-func contract: no HIP calls, no blocking — flag + wake only.
extern "C" fn trampoline(user: *mut c_void) {
    // SAFETY: `user` is the reference `arm_on_raw` shared out; reclaimed exactly once, here.
    let sig = unsafe { Signal::reclaim_raw(user) };
    sig.resolve();
}

/// The HIP half of [`Signal`]: arming one on a stream.
///
/// Its per-read use is GONE. `asyncfetch` armed one of these per cold read so `gpu.rs` could
/// await each expert's bytes; the ticketed dataflow moved that dependency onto the device
/// (INV-5) and the signals stayed on as a `hipLaunchHostFunc` per miss that nobody polled.
/// What is left is `stream_signal` — one arm per stream per layer, which the decode loop
/// genuinely awaits — so `ready()` (the resolved-on-arrival cache hit) has no callers either
/// and is deleted; `Ticket::RESIDENT` says that now. `arm_on(&HipStream)` went the same way
/// on 2026-08-01: `stream_signal` and the tests reach `arm_on_raw` directly.
impl Signal {
    /// Fire this signal when the stream reaches its current point (a host-func callback
    /// wakes it), so the enqueued GPU work's completion resolves the future. Enqueue the
    /// work first, then arm.
    // Opaque HIP handle passed to the runtime, not dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn arm_on_raw(&self, stream_raw: *mut c_void) -> Result<()> {
        // Hand one ref to the callback; the trampoline reclaims it exactly once.
        let user = self.share_raw();
        // SAFETY: `trampoline` has the C callback signature; `user` is a live Arc ptr.
        let rc = unsafe { rivoli_stream_host_signal(stream_raw, trampoline, user) };
        if rc != 0 {
            // Enqueue failed → callback never runs → reclaim the ref here.
            // SAFETY: `user` came from `share_raw` above and was not reclaimed.
            unsafe { drop(Signal::reclaim_raw(user)) };
            bail!("hipLaunchHostFunc failed ({rc})");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::block_on;

    // Proves the bridge resolves on GPU completion AND measures per-signal
    // round-trip latency — the number that decides whether per-expert async
    // signals (~9/layer × 75 layers/token) are affordable vs the overlap they buy.
    // A Signal that resolves once every op enqueued on `stream` so far completes.
    fn signal(stream: &HipStream) -> Result<Signal> {
        let s = Signal::pending();
        s.arm_on_raw(stream.raw())?;
        Ok(s)
    }

    #[test]
    fn signal_resolves_and_latency() -> Result<()> {
        let stream = HipStream::new()?;
        // Warm the host-func machinery.
        block_on(signal(&stream)?);
        let n = 2000u32;
        let t = std::time::Instant::now();
        block_on(async {
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

// jscpd:ignore-start — the twin `unsafe impl Send/Sync for Timeline` in `vk.rs` is two
// lines over a DIFFERENT type, discharged by a different argument (Vulkan semaphores are
// internally synchronised for wait/signal/query). Two types cannot share one impl.
//
// SAFETY: the counter is device signal memory; the HIP calls that touch it are
// stream-ordered and internally synchronised, and Rust-side access is an atomic load.
unsafe impl Send for Timeline {}
unsafe impl Sync for Timeline {}
// jscpd:ignore-end

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

    /// TEARDOWN ONLY: force the counter to `value` from the host, releasing waits whose
    /// producer will never run.
    ///
    /// `wait` has no error state — it blocks until the value arrives, forever — so a
    /// producer that dies owing a ticket hangs every consumer already gated on it. That is
    /// not hypothetical: the reaper's poison path abandoned a batch without signalling its
    /// tickets, and the decode hung on the device rather than surfacing the fetch error.
    /// Monotone, so releasing cannot move a slot backwards.
    pub fn release(&self, value: u64) {
        // SAFETY: `self.0` is live signal memory; the C side does a monotone CAS on it.
        unsafe { rivoli_timeline_release(self.0, value) };
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
    // A device handle either exists or the test cannot run at all, so `expect` IS the
    // assertion here — matching `asyncfetch.rs`'s test module, which already says so.
    // Without this the crate's `[lints.clippy] expect_used = "deny"` fails the whole
    // `--all-targets` clippy run, which is the command CLAUDE.md tells an agent to use.
    #![allow(clippy::expect_used)]
    use super::*;
    use crate::backend::block_on;

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
        // If the wait had passed vacuously this would resolve regardless; the value check
        // below is what distinguishes "waited" from "did not block".
        block_on(stream_signal(a.raw()).expect("completion"));
        assert!(
            t.completed() >= 1,
            "the waited-on value must have been reached"
        );
    }

    /// **INV-6: a wait can always be released, so a dead producer cannot hang the device.**
    ///
    /// The other half of INV-4. Enqueuing a consumer before its producer is the whole design,
    /// and `hipStreamWaitValue64` has no error state — it blocks until the value arrives,
    /// forever — so the producer failing has to be expressible. It was not: the reaper's
    /// poison path resolved a per-read `Signal` and left the timelines alone, which released
    /// exactly nothing once the ticketed dataflow made the timeline the dependency. A fetch
    /// error hung on the device instead of returning.
    ///
    /// What this actually proves is the mechanism, not the bookkeeping: a HOST store into
    /// signal memory retires a device-side wait that was enqueued before it. Nothing in the
    /// HIP docs promises that direction, and the teardown path is built on it.
    #[test]
    fn inv_6_a_host_release_retires_an_enqueued_wait() {
        let t = Timeline::new().expect("timeline");
        let s = HipStream::new().expect("stream");
        // The wait goes in first, against a value NOTHING on any stream will ever signal —
        // this is the dead-producer case, not a race with a slow one.
        // jscpd:ignore-start — INV-4's two halves. `tests/vk.rs` asserts the SAME property
        // against the Vulkan timeline, and the assertions are identical because the
        // invariant is: a wait enqueued before anything signals must still retire, and a
        // release must never move a timeline backwards. `rocm` and `vulkan` are mutually
        // exclusive features, so there is no module where one copy could compile for both —
        // a shared helper is not available at any price. Same category as the ABI walls in
        // hip.rs/vk.rs, and the reason INV-4 is registered once in architecture.md §8b
        // while being tested twice.
        t.wait(s.raw(), 7)
            .expect("wait enqueues against a value no stream will write");
        t.release(7);
        block_on(stream_signal(s.raw()).expect("completion"));
        assert!(
            t.completed() >= 7,
            "the host release must be what retired the wait"
        );
        // Monotone: releasing backwards would un-free a slot another consumer is gated on,
        // turning a clean teardown into the deadlock it exists to prevent.
        t.release(3);
        assert_eq!(
            t.completed(),
            7,
            "release must never move a timeline backwards"
        );
        // jscpd:ignore-end
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
        block_on(stream_signal(s.raw()).expect("completion"));
        assert_eq!(
            t.completed(),
            4,
            "the last value written must be observable"
        );
    }
}
