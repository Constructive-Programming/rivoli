//! rivoli — GLM-5.2 MoE decode engine (int3-vq / int4 / hybrid routed experts).
//! See docs/ARCHITECTURE.md for the module map and MODES.md for the run modes.
pub mod arena;
pub mod attn;
pub mod cache;
pub mod config;
pub mod device;
pub mod format;
pub mod hybrid;
pub mod indexer;
pub mod math;
pub mod model;
pub mod quant;
pub mod telemetry;
pub mod tokenizer;
pub mod watchdog;

/// The build-time backend switch (`rocm` XOR `vulkan`) — see docs/VULKAN.md. Gated on a
/// backend being selected: with neither feature the crate is its backend-independent half
/// (config, math, quant, arena, cache, telemetry), which has no waist to switch. That is
/// why `backend.rs` carries no `compile_error!` for the neither case — see the note there.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod backend;

// The engine proper: the decode loop, the resident weight set, and the streaming fetch
// pipeline. Backend-INDEPENDENT since phase 4 increment 1 — every device call goes through
// `crate::backend` — so these compile under either feature. `stream.rs` is io_uring and was
// never backend-specific except in its destination pointer and its staging copy.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod asyncfetch;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod gpu;
// Backend-neutral (pure counters + routing scratch), so it builds wherever `trace` does.
#[cfg(feature = "trace")]
pub mod looka;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod pin;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod stream;

#[cfg(feature = "rocm")]
pub mod gpustream;
#[cfg(feature = "rocm")]
pub mod hip;

#[cfg(feature = "vulkan")]
pub mod vk;
#[cfg(feature = "vulkan")]
pub mod vkstream;
