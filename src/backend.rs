//! The compute-backend waist: one implementation, chosen at BUILD time.
//!
//! No trait, no dynamic dispatch — the two backends are never live at once, so a vtable in
//! front of every `launch_*` on the decode hot path would buy nothing.
//!
//! # This IS the seam now. `gpu.rs`, `pin.rs` and `asyncfetch.rs` import it.
//!
//! An earlier version of this file was a plan with no consumers, and said so. Phase 4
//! increment 1 wired it up. Two things changed to make that possible, and both are the
//! reason the surface below is wider than a list of launchers:
//!
//! - **The stream/event types are part of the boundary, not incidental.** `gpu.rs` used to
//!   import `crate::backend::gpustream::{HipEvent, HipStream, Signal, stream_signal}` directly, and
//!   `HipStream` has no Vulkan analogue as such. They are re-exported here under
//!   BACKEND-NEUTRAL names — [`Stream`], [`Event`] — so no module above this one spells a
//!   backend into a type name. `src/backend/vkstream.rs` is the Vulkan side. [`Signal`] is
//!   not re-exported but DEFINED here — see its own note.
//! - **The two backends do not agree on what a device pointer is.** Under HIP one number
//!   is both a host and a device address; under Vulkan they are unrelated. That asymmetry
//!   is NOT hidden here — it cannot be. `device.rs`'s `VmmBuf` hands out both bases and
//!   `pin.rs` picks per consumer (descriptors take the device base, the io_uring DMA
//!   target takes the host base). See docs/investigations/vulkan-port.md, "Host pointer != device address".
//!
//! # What is NOT equal across the seam
//!
//! Compiling is not equivalence. Two of the three differences this list used to carry are
//! GONE as of Phase 4 increment 2 and are recorded here because their absence is the point:
//! the Vulkan side runs THREE QUEUES with the fetch↔compute overlap measured at 97%
//! (ROCm 96%), and `Event` returns real GPU milliseconds from a timestamp query pool rather
//! than a `0.0` that arithmetic downstream turned into "0% hidden".
//!
//! What remains unequal, and is load-bearing for anyone reading a Vulkan run's output:
//!
//! 1. **13 of 29 kernels refuse rather than run.** `Config::validate_backend` rejects the
//!    configurations that would reach them (`--attn dsa|misa`, `--mode int4|hybrid`) at
//!    startup; the launchers themselves return `Err` as a backstop. The `--mode` and
//!    `--attn auto` DEFAULTS therefore differ by backend — see `config::Mode`.
//! 1b. **Six PORTED kernels are single-row.** A different thing from a deferred kernel and
//!    worth its own line: `gemv_fp8`, `gemv_f32`, `gemv_i8`, `mla_absorb_fp8`,
//!    `mla_value_fp8` and `moe_expert_range` all run correctly here at `nrow == 1` and
//!    refuse above it, because the `.comp` shaders carry no row axis. That makes
//!    SPECULATIVE DECODE (the two-row verify pass, ARCHITECTURE §13) ROCm-only. Unlike the
//!    13, this is not gated at startup — an artifact carrying the MTP head fails at the
//!    first layer of the first token with a message naming `--features rocm`. Acceptable
//!    only because the Vulkan default is `--mode int3-vq` while the head rides `.vq3`, so
//!    the combination is reachable; if a Vulkan run is ever expected to carry one, this
//!    belongs in `validate_backend` instead.
//! 2. **The MoE kernels are ~2.1x slower than the HIP originals**, which is now the whole of
//!    the throughput gap (measured per-phase; docs/investigations/vulkan-port.md, "Increment 2: measured"). A
//!    Vulkan run's tok/s is not a statement about the engine's design, and after
//!    increment 2 it is not a statement about its scheduling either.
//! 3. **A `Stream` here NAMES a queue rather than owning one.** `Stream::compute()` and
//!    `Stream::fetch()` exist on both backends so the role is chosen by the consumer that
//!    knows it; HIP ignores the distinction, Vulkan cannot.

// One backend per build. Both at once is not a configuration that could work — the two
// `pub use` globs below would collide on every shared name, and the resulting hundred
// ambiguity errors would bury the one fact that matters.
#[cfg(all(feature = "rocm", feature = "vulkan"))]
compile_error!(
    "features `rocm` and `vulkan` are mutually exclusive: one compute backend per build, \
     selected at build time (docs/investigations/vulkan-port.md, \"Backend selection\"). Pick one."
);

// NEITHER feature is a legal build — `cargo test` runs the backend-independent half of the
// crate (config, math, quant, arena, cache, telemetry) with no device at all. It is
// expressed by this module's ABSENCE rather than by a `compile_error!` here: `lib.rs` gates
// `pub mod backend` on `any(rocm, vulkan)`, so a featureless build simply has no waist, and
// `main.rs` bails with a message naming both features when asked to decode without one. A
// `compile_error!` for the neither case would fire on `cargo test` and break the very
// builds that keep the shared code honest.

// The two implementations live under this module (`src/backend/`) rather than beside it,
// so the waist and the things it selects between are one subtree.
#[cfg(feature = "rocm")]
pub mod gpustream;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "vulkan")]
pub mod vk;
#[cfg(feature = "vulkan")]
pub mod vkstream;

#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
mod imp {
    pub use crate::backend::gpustream::{
        HipEvent as Event, HipStream as Stream, Timeline, stream_signal,
    };
    pub use crate::backend::hip::*;
}

#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
mod imp {
    pub use crate::backend::vk::*;
    pub use crate::backend::vkstream::{Event, Stream, stream_signal};
}

pub use imp::*;

/// "Deliberately not on a stream" — the null stream, named.
///
/// Every `launch_*` takes a trailing stream and accepts null, so a bare
/// `std::ptr::null_mut()` at a call site says only "a pointer" where it means "this work is
/// ordered by the null stream, on purpose". Naming it is the difference between a decision
/// and a default, and the launchers' own `# Safety` blocks now distinguish the two.
///
/// It also removes a token-level hazard that is not hypothetical: merging the layer-loop and
/// stream branches produced a jscpd clone between `gpu.rs` and `v4gpu.rs` that was **not in
/// either branch alone** — two unrelated call sites whose multi-line argument lists happened
/// to end in the same `null_mut(), )?; }` sequence, 24 tokens. `kernels/` is outside jscpd
/// and `git` saw no conflict, so the build script was the only thing that could catch it.
/// rustfmt manufactures that shape whenever a call gains an argument and gets reflowed.
pub const NULL_STREAM: *mut std::ffi::c_void = std::ptr::null_mut();

// ---------------------------------------------------------------------------
// Shared by both backends — the parts of the waist that are not backend-specific.
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
/// This was two byte-identical copies, in `gpustream.rs` and `vk.rs`. `Signal` crosses the
/// waist — `gpu.rs` and `asyncfetch.rs` await one without knowing which backend built it —
/// so a drift between the copies would not have been a compile error on either build; it
/// would have been a `Future` that polls differently depending on a feature flag. Nothing
/// in it is backend-specific: it is a flag, a waker, and the double-check that closes the
/// race between them.
///
/// What IS per-backend is only how a signal gets ARMED — `hipLaunchHostFunc` on one side, a
/// timeline-semaphore waiter thread on the other — and that stays in each backend's module,
/// as an inherent `impl` on this type (Rust allows those in any module of the defining
/// crate).
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
/// one with the note "fifteen lines of std beats pulling in a runtime".
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
