#![cfg(feature = "rocm")]
//! Async-signal bridge: HIP stream completions become futures. This is the
//! keystone of the streaming MoE pipeline — a GPU-stream op turns into a future
//! that resolves when the hardware reaches that point (a `hipLaunchHostFunc`
//! callback wakes a `Waker`), so `futures-util` can orchestrate GPU concurrency
//! directly instead of blocking on `hipDeviceSynchronize`. No polling, no join.

use anyhow::{Result, bail};
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
}

/// An owned non-blocking HIP stream. Work launched on `raw()` overlaps work on
/// other streams. The handle is an opaque runtime-owned pointer we only pass back
/// to HIP, so moving it across threads is sound.
pub struct HipStream(*mut c_void);
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn new() -> Result<Self> {
        // SAFETY: no args; returns null on failure.
        let s = unsafe { rivoli_stream_create() };
        if s.is_null() {
            bail!("hipStreamCreate failed");
        }
        Ok(HipStream(s))
    }

    #[inline]
    pub fn raw(&self) -> *mut c_void {
        self.0
    }

    /// A [`Signal`] that resolves once every op enqueued on this stream *so far*
    /// has completed on the GPU. Enqueue the work first, then call this.
    pub fn signal(&self) -> Result<Signal> {
        let s = Signal::pending();
        s.arm_on(self)?;
        Ok(s)
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        // SAFETY: self.0 live; teardown error ignored.
        unsafe { rivoli_stream_destroy(self.0) };
    }
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
        // Hand one ref to the callback; the trampoline reclaims it exactly once.
        let user = Arc::into_raw(self.0.clone()) as *mut c_void;
        // SAFETY: `trampoline` has the C callback signature; `user` is a live Arc ptr.
        let rc = unsafe { rivoli_stream_host_signal(stream.raw(), trampoline, user) };
        if rc != 0 {
            // Enqueue failed → callback never runs → reclaim the ref here.
            // SAFETY: `user` came from into_raw and was not consumed.
            unsafe { drop(Arc::from_raw(user as *const SignalState)) };
            bail!("hipLaunchHostFunc failed ({rc})");
        }
        Ok(())
    }

    /// True once resolved (non-blocking) — the collision guard checks this before
    /// reusing an in-flight slot.
    pub fn is_ready(&self) -> bool {
        self.0.done.load(Ordering::Acquire)
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
    #[test]
    fn signal_resolves_and_latency() -> Result<()> {
        let stream = HipStream::new()?;
        let rt = tokio::runtime::Builder::new_current_thread().build()?;
        // Warm the runtime + the host-func machinery.
        rt.block_on(stream.signal()?);
        let n = 2000u32;
        let t = std::time::Instant::now();
        rt.block_on(async {
            for _ in 0..n {
                stream.signal()?.await;
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
