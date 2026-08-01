//! rivoli — GLM-5.2 MoE decode engine (int3-vq / int4 / hybrid routed experts).
//! See docs/reference/architecture.md for the module map and docs/reference/modes.md for the run modes.

/// The build-time backend switch (`rocm` XOR `vulkan`) — see docs/investigations/vulkan-port.md. Gated on a
/// backend being selected: with neither feature the crate is its backend-independent half
/// (config, math, quant, arena, cache, telemetry), which has no waist to switch. That is
/// why `backend.rs` carries no `compile_error!` for the neither case — see the note there.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod backend;

// Grouped by subsystem, mirroring docs/reference/architecture.md: where weights live (§1, §6), how
// they get there (§3, §4), and what they are written in (§7).
pub mod artifact;
pub mod fetch;
pub mod memory;

pub mod attn;
pub mod indexer;
pub mod math;
pub mod telemetry;
pub mod watchdog;

#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod gpu;

/// Teacher-forced scoring (`--ppl`). An instrument, not an engine feature — nothing in a
/// decode reaches it — so it is a module boundary AND a feature boundary, and the two
/// cannot drift apart. `bin/ppl`, which does the statistics over its output, is pure host
/// arithmetic and needs no gate at all.
#[cfg(all(feature = "teacher-forcing", any(feature = "rocm", feature = "vulkan")))]
pub mod eval;
