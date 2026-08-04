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

/// The DeepSeek-V4-Flash numerical oracle (S1b of `docs/investigations/v4-flash-port.md`).
///
/// Backend-independent and engine-independent by construction: it imports nothing from
/// `gpu`, `attn`, `math` or `artifact`, because an oracle that shares code with the
/// implementation it judges is blind to any bug they share.
///
/// **Ungated, unlike `eval` and the pred-probe.** Those are gated because they put work on
/// the per-token decode path; this has no runtime surface at all — nothing in a decode
/// reaches it — so a feature would buy only build time. And it would cost something real:
/// a feature-gated module here is compiled exactly as often as someone remembers to name
/// its feature, which is how `otlp` sat broken on an `E0609` for weeks (CLAUDE.md). The
/// oracle is what S2 and S3 are scored against; it must not be the thing that silently
/// stopped compiling.
pub mod v4oracle;
pub mod telemetry;
pub mod watchdog;

/// The OpenAI-compatible HTTP server (`--port`), for llama-swap and friends.
///
/// Backend-gated like `gpu`, since without one there is no engine to serve — but with
/// `test` added, because the half of it that matters for correctness (HTTP framing,
/// message flattening, the streaming detokenizer) is pure host code, and the featureless
/// build is the one CI runs. `serve()` itself, which drives a live engine, carries the
/// backend cfg a second time inside.
#[cfg(any(feature = "rocm", feature = "vulkan", test))]
pub mod serve;

#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub mod gpu;

/// Teacher-forced scoring (`--ppl`). An instrument, not an engine feature — nothing in a
/// decode reaches it — so it is a module boundary AND a feature boundary, and the two
/// cannot drift apart. `bin/ppl`, which does the statistics over its output, is pure host
/// arithmetic and needs no gate at all.
#[cfg(all(feature = "teacher-forcing", any(feature = "rocm", feature = "vulkan")))]
pub mod eval;
