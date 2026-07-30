#![cfg(feature = "vulkan")]
//! `gpustream.rs`'s Vulkan twin: the stream and event handles the decode loop threads
//! through the launchers, over ONE queue.
//!
//! # ONE QUEUE. Fetch does not overlap compute, and that is a ~3x cost.
//!
//! HIP runs three streams — the null stream for the forward pass, `compute_stream` for the
//! MoE expert partials, and the reaper's fetch stream for H2D weight copies — and the
//! overlap between the last two is what hides ~95% of fetch behind compute
//! (docs/benchmarks.md). `vk.rs` is one queue behind one `Mutex<Cmd>`, so integrating onto
//! it SERIALISES fetch against compute.
//!
//! This is deliberate for increment 1 and it is not fixed here. docs/VULKAN.md, "Phase 4
//! needs two queues, not one" predicts the slowdown and "The design, on paper" is the
//! three-queue plan that removes it — three queues from family 1, one `Mutex<Stream>` and
//! one timeline semaphore each, plus the command-buffer ring that closes `Gpu::signal`'s
//! submitted-vs-recorded gap. That is increment 2.
//!
//! **The acceptance gate cannot see any of this.** Token IDs depend on arithmetic, not on
//! overlap, so a fully serialised backend computes exactly the same numbers. Anyone
//! measuring this build must compare fetch-hidden % and tok/s against the ROCm figures at
//! matched `--max-mem` and `--cache-policy`, not just diff the token stream.

use crate::vk::{Signal, gpu};
use anyhow::Result;
use std::ffi::c_void;

/// A stream handle. **Carries nothing**, because there is nothing yet to carry: the single
/// queue lives inside `vk::Gpu`'s `Mutex<Cmd>` and every `vk.rs` launcher takes its
/// `_stream` argument and ignores it (see `launch_moe_expert_range`). Work "launched on
/// this stream" is recorded into the one command buffer in program order.
///
/// The type still EXISTS rather than being erased from `gpu.rs`, because increment 2 gives
/// it a queue, a command-buffer ring and a timeline of its own — and at that point every
/// call site that must pick a stream is already written. Deleting it would mean rewriting
/// them.
///
/// `raw()` is a null pointer, and that is honest for the same reason: the launchers do not
/// read it. If a Vulkan launcher ever starts consuming `_stream`, it must stop being null
/// on the same commit.
pub struct Stream;

impl Stream {
    pub fn new() -> Result<Self> {
        Ok(Stream)
    }

    #[inline]
    pub fn raw(&self) -> *mut c_void {
        std::ptr::null_mut()
    }
}

/// A timing event.
///
/// # `elapsed_ms` IS ALWAYS 0.0 ON VULKAN. Every timing span reads zero.
///
/// Vulkan times GPU work with `VK_QUERY_TYPE_TIMESTAMP` query pools, which are not wired
/// up (docs/VULKAN.md phase 5: "timestamp query pools wired into telemetry.rs"). Rather
/// than invent a plausible figure, this returns exactly zero, so the consequence is
/// visible in the telemetry instead of hidden in it:
///
/// - `compute_gpu_ms` / `tail_gpu_ms` / the indexer span in `ProfileSummary` are **0.0 on
///   a Vulkan build and mean nothing**. Do not compare them against the ROCm numbers in
///   docs/benchmarks.md — they are not small, they are absent.
/// - Wall-clock tok/s is unaffected and remains the number to use.
///
/// A plausible-looking value would be worse than a zero, because zero cannot be mistaken
/// for a measurement.
pub struct Event;

impl Event {
    pub fn new() -> Result<Self> {
        Ok(Event)
    }

    /// No-op: nothing records a timestamp yet. Returns `Ok` rather than failing, because
    /// `gpu.rs` records events unconditionally on the decode path and a hard error would
    /// turn missing instrumentation into a broken decode.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // takes no pointer it could deref; mirrors HipEvent
    pub fn record(&self, _stream_raw: *mut c_void) -> Result<()> {
        Ok(())
    }

    /// Always `0.0`. See the type's note — this is not a degraded measurement, it is the
    /// absence of one.
    pub fn elapsed_ms(_start: &Event, _end: &Event) -> Result<f32> {
        Ok(0.0)
    }
}

/// A fresh [`Signal`] armed on the queue — resolves once everything submitted so far has
/// retired. `gpustream::stream_signal`'s twin.
///
/// The HIP version arms on the *stream's current point*; this arms on the *queue's*, which
/// is the same thing while there is one queue. It also covers only SUBMITTED work, not
/// work still recorded in the open command buffer — see [`crate::vk::Gpu::signal`].
/// `gpu.rs`'s expert loop reaches it after a `device_sync`, which flushes recording, so
/// the distinction does not bite today. It will the moment a second queue exists, which is
/// why the ring and the queue are one design.
// The raw handle is unused for the same reason `Stream::raw` is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn stream_signal(_stream_raw: *mut c_void) -> Result<Signal> {
    gpu()?.signal()
}

/// `Signal::arm_on`, the half of the HIP surface `vk.rs` cannot declare on its own.
///
/// `asyncfetch.rs` hands out one pending [`Signal`] per queued read before any of them
/// completes, then arms each as its completion is reaped — so it needs to arm a signal it
/// already owns, not receive a fresh one. Lives here rather than in `vk.rs` because the
/// [`Stream`] it takes lives here; inherent impls may sit in any module of the defining
/// crate.
impl Signal {
    /// Fire this signal once everything submitted to `_stream` so far has retired.
    /// Enqueue the work first, then arm.
    pub fn arm_on(&self, _stream: &Stream) -> Result<()> {
        gpu()?.arm(self)
    }
}
