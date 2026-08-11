//! rivoli — GLM-5.2 MoE decode engine (int3-vq / int4 / hybrid routed experts).
//! See docs/reference/architecture.md for the module map and docs/reference/modes.md for the run modes.

/// The compute backend. Gated on `rocm` being selected: with no feature the crate is its
/// backend-independent half (config, math, quant, arena, cache, telemetry), which has no
/// waist at all. That is why `backend.rs` carries no `compile_error!` for the neither case
/// — see the note there, and **do not collapse this `cfg` away**: it is the single line
/// that keeps the featureless build (all of CI's `host` job) compiling.
///
/// This was a `rocm` XOR `vulkan` switch until 2026-08-06. The Vulkan backend was retired
/// as an unfinished port — 6 of 36 mode-matrix cells decoding, 16 of 29 kernels, no V4 path
/// — and is preserved at the tag `archive/vulkan-backend-hb16`.
#[cfg(feature = "rocm")]
pub mod backend;

/// Which architecture an artifact is — the discriminant `artifact::model` reads out of the
/// manifest and `main` renders `--help` against. Top level and backend-free on purpose:
/// the offline converters resolve it too, and they build with no backend at all.
pub mod arch;

// Grouped by subsystem, mirroring docs/reference/architecture.md: where weights live (§1, §6), how
// they get there (§3, §4), and what they are written in (§7).
pub mod artifact;
pub mod fetch;
pub mod memory;

pub mod attn;
pub mod indexer;
pub mod math;

/// V4-Flash's KV compressor and sparse indexer (S2c of
/// `docs/investigations/v4-flash-port.md`).
///
/// The host half is backend-free for the same reason as `arch`: it is the two RoPE tables,
/// the per-layer shape discriminants and the two *arithmetic* selection paths, all of which
/// the offline converters and the CPU tests need without a device.
///
/// The pooling is `kernels/kvcompress.hip`, launched by the `rocm`-gated `device` submodule
/// — so this module is no longer *entirely* ungated, and the split is deliberate: everything
/// above `mod device` still compiles and tests with no feature and no GPU. The indexer's
/// scoring is still device work that does not exist yet; it needs S2a's e2m1/e8m0 block.
///
/// Separate from [`indexer`], which is GLM's DSA lightning indexer: V4's `Indexer` shares
/// the name and none of the structure — no `wk`, no `k_norm`, its own nested `Compressor`.
pub mod kvcompress;

/// The on-disk golden container, shared by every model's fixtures.
///
/// Lived under `v4oracle/` until 2026-08-11, and its own doc said to move it here "if a third
/// model arrives" rather than grow a third magic under a name that says V4. Muse Glimmer is that
/// third model. The layout is model-agnostic; only the eight-byte magic is not, and a module named
/// for the model that happened to introduce it is the naming defect this repo has now hit twice.
pub mod golden;

pub mod telemetry;
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
pub mod watchdog;

/// The OpenAI-compatible HTTP server (`--port`), for llama-swap and friends.
///
/// Backend-gated like `gpu`, since without one there is no engine to serve — but with
/// `test` added, because the half of it that matters for correctness (HTTP framing,
/// message flattening, the streaming detokenizer) is pure host code, and the featureless
/// build is the one CI runs. `serve()` itself, which drives a live engine, carries the
/// backend cfg a second time inside.
#[cfg(any(feature = "rocm", test))]
pub mod serve;

#[cfg(feature = "rocm")]
pub mod gpu;

/// DeepSeek-V4-Flash's layer loop — [`gpu`]'s counterpart, not a branch inside it.
///
/// **`rocm` only — which since 2026-08-06 is the only backend there is.** Every launcher it
/// drives is `crate::backend::hip`'s. The gate predates that: the Vulkan backend had no
/// `v4_*` twin and never would have, and "no V4 decode path at all" was one of the measured
/// reasons it was retired rather than finished.
#[cfg(feature = "rocm")]
pub mod f4gpu;

/// Teacher-forced scoring (`--ppl`). An instrument, not an engine feature — nothing in a
/// decode reaches it — so it is a module boundary AND a feature boundary, and the two
/// cannot drift apart. `bin/ppl`, which does the statistics over its output, is pure host
/// arithmetic and needs no gate at all.
#[cfg(all(feature = "teacher-forcing", feature = "rocm"))]
pub mod eval;
