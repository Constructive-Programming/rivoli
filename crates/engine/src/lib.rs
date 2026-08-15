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
#[cfg(feature = "rocm")]
pub mod glm;
pub mod indexer;
#[cfg(feature = "rocm")]
pub mod routed;
pub mod seam;
pub mod telemetry;

// The seam's vocabulary, hoisted to the crate root: `rivoli_engine::Engine` is what a
// consumer programs against, and making it name the module the enum happens to live in is
// one more detail the second consumer could get wrong.
pub use seam::{DecodeStats, Decoded, Engine, GenSpec, OpenSpec};
