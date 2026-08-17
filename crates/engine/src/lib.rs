//! # rivoli-engine — the imperative shell
//!
//! The interpreter for `rivoli-core`'s plans: io_uring fetch, slot/tier
//! executors, the per-architecture pins and layer loops (glm, v4, k3, glimmer —
//! four loop bodies in four files, a measured decision, not debt), the one
//! `enum Engine` seam that `serve` and `main` program against, the run record
//! (contention witness + paired-dNLL slots — an unwitnessed number is
//! structurally unciteable), and eval. Instrument features live here, each
//! behind a feature AND a flag, never an env var.

#[cfg(feature = "rocm")]
pub mod device;
pub mod fetch;
pub mod glimmer;
#[cfg(feature = "rocm")]
pub mod glm;
pub mod indexer;
/// `--divergence-log`: the (position, layer, quantity) localiser for a run-to-run divergence,
/// behind its own feature rather than `trace`'s because `trace` adds the class of
/// `device_sync` that is recorded to MASK this fault. The pure host fold it is scored against
/// lives in `rivoli_core::hash`, so a build without this feature still compiles that oracle.
#[cfg(all(feature = "rocm", feature = "corruption-probe"))]
pub mod probe;
// Ungated at the module and gated inside it, like `v4`: the fold schedule, the router
// weighting and the composed widths are arithmetic over a config that touches no device,
// and they are the half where a wrong answer is silent. `k3.rs` carries the arm's table.
pub mod k3;
/// The resident-set vocabulary every arm's pin shares — resolved weight types, the placers
/// that build them, and the weights-only [`rivoli_core::residency::Floor`]. Gated here rather
/// than inside, because every item takes a `DeviceTier`.
#[cfg(feature = "rocm")]
pub mod resident;
#[cfg(feature = "rocm")]
pub mod routed;
pub mod seam;
pub mod telemetry;
// Ungated at the module and gated inside it, unlike `glm`: this arm's layer classes, rotary
// tables and row selection are arithmetic over a config that touches no device, and they are
// the half where a wrong answer is silent. `v4.rs` carries the argument.
pub mod v4;

// The seam's vocabulary, hoisted to the crate root: `rivoli_engine::Engine` is what a
// consumer programs against, and making it name the module the enum happens to live in is
// one more detail the second consumer could get wrong.
pub use seam::{ArchCfg, DecodeStats, Decoded, Engine, GenSpec, OpenSpec, PoolKnobs, TokenSink};
