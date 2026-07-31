#![cfg(feature = "vulkan")]
//! `gpustream.rs`'s Vulkan twin: the stream and event handles the decode loop threads
//! through the launchers.
//!
//! # THREE QUEUES, one per HIP stream
//!
//! HIP runs three streams — the null stream for the forward pass, `compute_stream` for the
//! MoE expert partials, and the reaper's fetch stream for H2D weight copies — and the
//! overlap between the last two is what hides ~95% of fetch behind compute
//! (docs/benchmarks.md). Increment 1 ran all three on ONE queue, which serialised fetch
//! against compute and cost ~246 ms/token; `vk.rs` now has one [`Q`] per HIP stream, each
//! with its own command-buffer ring and its own timeline semaphore.
//!
//! A [`Stream`] here is just the NAME of one of them. The queue itself lives inside
//! `vk.rs`'s private `Mutex<Stream>` (a different type, same word — that one owns a
//! `VkQueue`, this one owns nothing), because `vkQueueSubmit` needs external
//! synchronisation and a guard is the only way to get it without a convention.
//!
//! **The acceptance gate still cannot see any of this.** Token IDs depend on arithmetic,
//! not on overlap, so a fully serialised backend computes exactly the same numbers.
//! Anyone measuring this build must compare fetch-hidden % and tok/s against the ROCm
//! figures at matched `--max-mem` and `--cache-policy`, not just diff the token stream —
//! which is now possible, because [`Event`] returns real GPU milliseconds.

use crate::backend::vk::{Q, Signal, Stamp, gpu};
use anyhow::Result;
use std::ffi::c_void;

/// A stream handle: which of the three queues work goes on.
///
/// The two constructors exist because the consumer is the only thing that knows its role,
/// exactly as under HIP where `gpu.rs` and `asyncfetch.rs` each create their own
/// `hipStream_t`. A single `new()` could not tell them apart, and inferring the role from
/// context is precisely the sort of thing that silently reintroduces the serialisation.
///
/// `raw()` is the [`Q`] tag, not a pointer to anything — see [`Q`]. It is the same
/// `*mut c_void` HIP passes a `hipStream_t` through, because that signature is shared by
/// both backends; the Vulkan launchers PARSE it (`Q::parse`) and never dereference it.
pub struct Stream(Q);

impl Stream {
    /// The MoE expert-partial stream. HIP's `compute_stream`.
    pub fn compute() -> Result<Self> {
        Ok(Stream(Q::Moe))
    }

    /// The reaper's H2D fetch stream.
    pub fn fetch() -> Result<Self> {
        Ok(Stream(Q::Fetch))
    }

    /// The MISS stream. **Maps to `Q::Moe` — the same queue as `compute()` — because this
    /// backend has exactly three queues and none to spare.**
    ///
    /// That is a lost optimisation, not a correctness gap. The overlap win (a miss's wait
    /// starting at the top of the layer rather than after the residents) simply does not
    /// happen here; every guarantee still does, because each miss is launched behind its own
    /// ticket wait either way. Nor can it deadlock: the join's `signal` is enqueued before
    /// its `wait` in program order, so on one queue the signal is always reached first.
    ///
    /// Give this its own `Q` variant when a fourth queue is available — `Q::COUNT` and the
    /// exhaustive matches on `Q` will name every site that needs updating.
    pub fn miss() -> Result<Self> {
        Ok(Stream(Q::Moe))
    }

    #[inline]
    pub fn raw(&self) -> *mut c_void {
        self.0.tag()
    }

    /// The queue this handle names — for `Signal::arm_on` below, so it does not have to go
    /// out through the raw tag and re-parse what it already knows.
    #[inline]
    pub(crate) fn queue(&self) -> Q {
        self.0
    }
}

/// A timing event — one point on a queue's GPU timeline.
///
/// # `elapsed_ms` IS A REAL MEASUREMENT NOW
///
/// It used to return exactly `0.0`, on the argument that a plausible-looking figure is
/// worse than a zero. Half right: `compute_gpu_ms` was then 0, so `ProfileSummary`'s
/// `fetch_hidden_pct` printed **0% as an arithmetic artifact** — a number that reads as a
/// finding. Phase 4 exists to preserve an overlap property, and an invariant that cannot be
/// measured is assumed rather than upheld, so the timestamp query pools moved forward out
/// of Phase 5. See [`Stamp`] for the mechanism (`vkCmdWriteTimestamp` into a `VkQueryPool`,
/// scaled by `limits.timestampPeriod`, masked to the family's `timestampValidBits`).
///
/// It can still refuse — a device whose queue family implements no timestamp bits, or a
/// pair read before the recording command buffer was submitted. Those return `Err`, which
/// is the honest spelling of "unavailable"; nothing here returns a zero standing in for a
/// measurement.
pub struct Event(Stamp);

impl Event {
    pub fn new() -> Result<Self> {
        Ok(Event(Stamp::new()?))
    }

    /// Record the timestamp on the queue `stream_raw` names. NULL is the main queue, which
    /// is HIP's null-stream convention.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // parses the tag; never dereferences it
    pub fn record(&self, stream_raw: *mut c_void) -> Result<()> {
        self.0.record(Q::parse(stream_raw)?)
    }

    /// GPU milliseconds between two events recorded on the SAME queue. Call after a
    /// `device_sync`, as the HIP twin requires.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        Stamp::elapsed_ms(&start.0, &end.0)
    }
}

/// A fresh [`Signal`] armed on the queue `stream_raw` names — resolves once everything
/// RECORDED OR SUBMITTED on it so far has retired. `gpustream::stream_signal`'s twin.
///
/// The recorded half is what the command-buffer ring bought: this arms on the stream's
/// current point the way the HIP version does, rather than on "whatever happened to be
/// submitted", so a caller no longer has to know that a `device_sync` must precede it.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // parses the tag; never dereferences it
pub fn stream_signal(stream_raw: *mut c_void) -> Result<Signal> {
    gpu()?.signal_on(Q::parse(stream_raw)?)
}

/// `Signal::arm_on`, the half of the HIP surface `vk.rs` cannot declare on its own.
///
/// `asyncfetch.rs` hands out one pending [`Signal`] per queued read before any of them
/// completes, then arms each as its completion is reaped — so it needs to arm a signal it
/// already owns, not receive a fresh one. Lives here rather than in `vk.rs` because the
/// [`Stream`] it takes lives here; inherent impls may sit in any module of the defining
/// crate.
impl Signal {
    /// Fire this signal once everything recorded or submitted to `stream` has retired.
    /// Enqueue the work first, then arm — the arming is what flushes it.
    pub fn arm_on(&self, stream: &Stream) -> Result<()> {
        gpu()?.arm_on(stream.queue(), self)
    }
}
