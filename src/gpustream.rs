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

    /// A future that resolves once every op enqueued on this stream *so far* has
    /// completed on the GPU. Enqueue the work first, then call this.
    pub fn signal(&self) -> Result<StreamSignal> {
        let state = Arc::new(SignalState {
            done: AtomicBool::new(false),
            waker: AtomicWaker::new(),
        });
        // Hand one ref to the callback; the trampoline reclaims it exactly once.
        let user = Arc::into_raw(state.clone()) as *mut c_void;
        // SAFETY: `trampoline` has the C callback signature; `user` is a live Arc ptr.
        let rc = unsafe { rivoli_stream_host_signal(self.0, trampoline, user) };
        if rc != 0 {
            // Enqueue failed → callback never runs → reclaim the ref here.
            // SAFETY: `user` came from into_raw and was not consumed.
            unsafe { drop(Arc::from_raw(user as *const SignalState)) };
            bail!("hipLaunchHostFunc failed ({rc})");
        }
        Ok(StreamSignal(state))
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

/// Resolves when its stream's enqueued work has completed on the GPU.
pub struct StreamSignal(Arc<SignalState>);

impl Future for StreamSignal {
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
