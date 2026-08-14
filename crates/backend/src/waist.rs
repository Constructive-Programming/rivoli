//! The compute-backend waist: one implementation, HIP/ROCm, selected at BUILD time.
//!
//! No trait, no dynamic dispatch — there is one backend, and even when there were two they
//! were never live at once, so a vtable in front of every `launch_*` on the decode hot path
//! would buy nothing.
//!
//! # This IS the seam. `gpu.rs`, `pin.rs` and `asyncfetch.rs` import it.
//!
//! An earlier version of this file was a plan with no consumers, and said so. Phase 4
//! increment 1 wired it up. The surface below is wider than a list of launchers because the
//! stream/event types are part of the boundary, not incidental: `gpu.rs` used to import
//! `crate::gpustream::{HipEvent, HipStream, Signal, stream_signal}` directly. They
//! are re-exported here under BACKEND-NEUTRAL names — [`Stream`], [`Event`] — so no module
//! above this one spells a backend into a type name. [`Signal`] is not re-exported but
//! DEFINED here — see its own note.
//!
//! # Why a "waist" with only one thing behind it — RETIRED 2026-08-06
//!
//! A second backend, Vulkan, lived behind this seam from its Phase-1 port until 2026-08-06,
//! when it was retired as an unfinished port rather than a feature: **6 of 36 mode-matrix
//! cells decoding, 16 of 29 kernels, ~1.9x slower on `--mode int3-vq --attn dense`,
//! refusing `int4`/`hybrid`/`dsa`/`misa` at startup, and no DeepSeek-V4 path at all.** The
//! code, the shaders and the full inventory of what it did and did not have are preserved
//! at the tag `archive/vulkan-backend-hb16`; the standing shader obligations and the
//! per-kernel table moved to `docs/investigations/vulkan-kernels.md`.
//!
//! Two things it forced on this file are worth keeping in view, because they still shape
//! code above the waist and would otherwise look arbitrary:
//!
//! - **The two backends did not agree on what a device pointer is.** Under HIP one number
//!   is both a host and a device address; under Vulkan they were unrelated. That asymmetry
//!   was never hidden here — it could not be. `device.rs`'s `VmmBuf` still hands out both
//!   bases and `pin.rs` still picks per consumer (descriptors take the device base, the
//!   io_uring DMA target takes the host base). Under HIP alone the two coincide, so the
//!   distinction now costs nothing and buys a `VmmBuf` that cannot silently conflate them.
//! - **A [`Stream`] NAMES a queue rather than owning one.** `Stream::compute()` and
//!   `Stream::fetch()` exist so the role is chosen by the consumer that knows it. HIP
//!   ignores the distinction; Vulkan could not. Kept because the call sites read better
//!   for it, not because anything now depends on the difference.

// NEITHER feature is a legal build — `cargo test` runs the backend-independent half of the
// crate (config, math, quant, arena, cache, telemetry) with no device at all. It is
// expressed by this module's ABSENCE rather than by a `compile_error!` here: `lib.rs` gates
// `pub mod backend` on `feature = "rocm"`, so a featureless build simply has no waist, and
// `main.rs` bails with a message naming the feature when asked to decode without one. A
// `compile_error!` for the neither case would fire on `cargo test` and break the very
// builds that keep the shared code honest.
//
// **That gate is why there are no `#[cfg(feature = "rocm")]` attributes below.** This file
// is only ever compiled when `rocm` is on, so an inner gate would be always-true — noise
// that reads like a live choice. It was a real choice until 2026-08-06, when the second
// backend was retired; do not re-add the attributes without first changing `lib.rs`, and
// do not "simplify" `lib.rs`'s gate away, because THAT is what keeps the featureless build
// compiling.

// The implementation lives under this module (`src/backend/`) rather than beside it, so the
// waist and the thing it selects stay one subtree. There were two until 2026-08-06 —
// `vk.rs` and `vkstream.rs`, chosen by a `vulkan` feature that this file made mutually
// exclusive with `rocm` via a `compile_error!`. Both are preserved at the tag
// `archive/vulkan-backend-hb16`.

// Re-exported under BACKEND-NEUTRAL names. Still worth doing with one backend: it is what
// stops every module above this one from spelling `HipStream` into its own signatures, and
// it is the whole reason the waist survived the second backend's removal as a seam rather
// than dissolving into `use crate::hip::*`.
pub use crate::gpustream::{HipEvent as Event, HipStream as Stream, Timeline, stream_signal};
pub use crate::hip::*;

/// "Deliberately not on a stream" — the null stream, named.
///
/// Every `launch_*` takes a trailing stream and accepts null, so a bare
/// `std::ptr::null_mut()` at a call site says only "a pointer" where it means "this work is
/// ordered by the null stream, on purpose". Naming it is the difference between a decision
/// and a default, and the launchers' own `# Safety` blocks now distinguish the two.
///
/// It also removes a token-level hazard that is not hypothetical: merging the layer-loop and
/// stream branches produced a jscpd clone between `gpu.rs` and `f4gpu.rs` that was **not in
/// either branch alone** — two unrelated call sites whose multi-line argument lists happened
/// to end in the same `null_mut(), )?; }` sequence, 24 tokens. `kernels/` is outside jscpd
/// and `git` saw no conflict, so the build script was the only thing that could catch it.
/// rustfmt manufactures that shape whenever a call gains an argument and gets reflowed.
pub const NULL_STREAM: *mut std::ffi::c_void = std::ptr::null_mut();

// ---------------------------------------------------------------------------
// The parts of the waist that are not backend-specific. They live here rather than in
// `hip.rs` because they were shared by both backends; with one left, "not backend-specific"
// is still the right home for them — see `Signal`'s note for why it did not move down.
// ---------------------------------------------------------------------------

use atomic_waker::AtomicWaker;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

/// A one-shot completion shared between whatever resolves it and the future that awaits it.
/// `Clone` (an `Arc`) so the resolver and the awaiter each hold one.
///
/// # ONE definition, not one per backend
///
/// This was two byte-identical copies, in `gpustream.rs` and `vk.rs` (the latter deleted
/// 2026-08-06 with the Vulkan backend; tag `archive/vulkan-backend-hb16`). `Signal` crosses
/// the waist — `gpu.rs` and `asyncfetch.rs` await one without knowing which backend built
/// it — so a drift between the copies would not have been a compile error on either build;
/// it would have been a `Future` that polls differently depending on a feature flag.
/// Nothing in it is backend-specific: it is a flag, a waker, and the double-check that
/// closes the race between them.
///
/// It stays HERE, above the one remaining backend, rather than moving down into
/// `gpustream.rs`: the reason it was hoisted was that consumers await it without knowing
/// the producer, and that is still true. What IS per-backend is only how a signal gets
/// ARMED — `hipLaunchHostFunc` today, a timeline-semaphore waiter thread on the Vulkan side
/// — and that stays in the backend's own module, as an inherent `impl` on this type (Rust
/// allows those in any module of the defining crate).
#[derive(Clone)]
pub struct Signal(Arc<SignalState>);

struct SignalState {
    done: AtomicBool,
    waker: AtomicWaker,
}

impl Signal {
    /// An unresolved signal.
    pub fn pending() -> Self {
        Signal(Arc::new(SignalState {
            done: AtomicBool::new(false),
            waker: AtomicWaker::new(),
        }))
    }

    /// Force-resolve from the resolver side, so awaiters never hang when the thing that was
    /// going to resolve this dies instead. Idempotent.
    pub fn resolve(&self) {
        self.0.done.store(true, Ordering::Release);
        self.0.waker.wake();
    }

    /// Hand ONE strong reference out as an opaque pointer for a C callback to own, balanced
    /// by exactly one [`Signal::reclaim_raw`]. Not `into_raw`: this SHARES a reference, it
    /// does not consume the signal — the caller keeps its own.
    ///
    /// The `Arc` arithmetic lives here rather than in `gpustream.rs` so the representation
    /// stays private to this module and the refcount has one place to be got wrong in.
    #[cfg(feature = "rocm")]
    pub(crate) fn share_raw(&self) -> *mut std::ffi::c_void {
        Arc::into_raw(self.0.clone()) as *mut std::ffi::c_void
    }

    /// Take back the reference [`Signal::share_raw`] handed out.
    ///
    /// # Safety
    /// `p` must come from one [`Signal::share_raw`] that has not been reclaimed yet.
    #[cfg(feature = "rocm")]
    pub(crate) unsafe fn reclaim_raw(p: *mut std::ffi::c_void) -> Self {
        // SAFETY: the caller's contract is exactly `Arc::from_raw`'s.
        Signal(unsafe { Arc::from_raw(p as *const SignalState) })
    }
}

impl Future for Signal {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0.done.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        self.0.waker.register(cx.waker());
        // Re-check: the resolver may have fired between the load and the register.
        if self.0.done.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Drive one future to completion on the calling thread, parking between polls.
///
/// # Why this is not `tokio`
///
/// The whole use of tokio in this crate was `Builder::new_current_thread().build()?
/// .block_on(..)` at eight sites — no `tokio::sync`, no `#[tokio::main]`, no `spawn`, no
/// timers, and the declared `sync`/`macros` features were dead. That is twelve crates for a
/// park/unpark loop `std::thread` already has, and `tests/vk.rs` had already written this
/// one with the note "fifteen lines of std beats pulling in a runtime". (That test file was
/// deleted 2026-08-06 with the Vulkan backend; the argument it made is why this is here.)
///
/// It is enough because the CONCURRENCY HERE IS GPU STREAMS, not CPU tasks (this module's
/// header says so): the decode loop awaits one thing at a time and the overlap it is waiting
/// on is happening on the device. A multi-threaded scheduler would have nothing to schedule.
/// `park` may wake spuriously; the loop re-polls, which is the correct response either way.
pub fn block_on<F: Future>(f: F) -> F::Output {
    use std::task::{Wake, Waker};
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}
