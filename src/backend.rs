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
//!   import `crate::gpustream::{HipEvent, HipStream, Signal, stream_signal}` directly, and
//!   `HipStream` has no Vulkan analogue as such. They are re-exported here under
//!   BACKEND-NEUTRAL names — [`Stream`], [`Event`] — so no module above this one spells a
//!   backend into a type name. `src/vkstream.rs` is the Vulkan side.
//! - **The two backends do not agree on what a device pointer is.** Under HIP one number
//!   is both a host and a device address; under Vulkan they are unrelated. That asymmetry
//!   is NOT hidden here — it cannot be. `device.rs`'s `VmmBuf` hands out both bases and
//!   `pin.rs` picks per consumer (descriptors take the device base, the io_uring DMA
//!   target takes the host base). See docs/VULKAN.md, "Host pointer != device address".
//!
//! # What is NOT equal across the seam
//!
//! Compiling is not equivalence, and three differences are load-bearing for anyone reading
//! a Vulkan run's output:
//!
//! 1. **One queue, so fetch does not overlap compute** — see `vkstream.rs`'s header.
//! 2. **Every GPU timing span reads 0.0 ms** — see [`Event`].
//! 3. **13 of 29 kernels refuse rather than run.** `Config::validate` rejects the
//!    configurations that would reach them (`--attn dsa|misa`, `--mode int4|hybrid`) at
//!    startup; the launchers themselves return `Err` as a backstop.

// One backend per build. Both at once is not a configuration that could work — the two
// `pub use` globs below would collide on every shared name, and the resulting hundred
// ambiguity errors would bury the one fact that matters.
#[cfg(all(feature = "rocm", feature = "vulkan"))]
compile_error!(
    "features `rocm` and `vulkan` are mutually exclusive: one compute backend per build, \
     selected at build time (docs/VULKAN.md, \"Backend selection\"). Pick one."
);

// NEITHER feature is a legal build — `cargo test` runs the backend-independent half of the
// crate (config, math, quant, arena, cache, telemetry) with no device at all. It is
// expressed by this module's ABSENCE rather than by a `compile_error!` here: `lib.rs` gates
// `pub mod backend` on `any(rocm, vulkan)`, so a featureless build simply has no waist, and
// `main.rs` bails with a message naming both features when asked to decode without one. A
// `compile_error!` for the neither case would fire on `cargo test` and break the very
// builds that keep the shared code honest.

#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
mod imp {
    pub use crate::gpustream::{HipEvent as Event, HipStream as Stream, Signal, stream_signal};
    pub use crate::hip::*;
}

#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
mod imp {
    pub use crate::vk::*;
    pub use crate::vkstream::{Event, Stream, stream_signal};
}

pub use imp::*;
