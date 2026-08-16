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
// UNGATED, and that is the whole point (2026-08-16, both code reviews). It was
// `#[cfg(feature = "teacher-forcing")]`, whose comment claimed "the deviceless test arm —
// the only arm CI compiles — is where they are exercised". **False, and checked:** CI runs
// `cargo test --workspace --locked --no-default-features`, CLAUDE.md prescribes that and
// `cargo test --workspace`, and `tests/feature-matrix.sh` runs `-p rivoli --test docs
// --test invariants` — NOT ONE of them enables `teacher-forcing`, so these tests were
// compiled by a clippy step and executed by nothing. Including the NaN-seed test that
// caught a live defect on its first run. A check nothing runs is not a check (P7).
//
// Ungating costs a stock build ~100 lines of host arithmetic it will not call, and buys
// every prescribed command the tests. `Scored` next door is ungated on exactly this
// argument. The per-arm `score()` loops stay behind the feature — they name a device.
pub mod score;
pub mod seam;
pub mod telemetry;
// Ungated at the module and gated inside it, unlike `glm`: this arm's layer classes, rotary
// tables and row selection are arithmetic over a config that touches no device, and they are
// the half where a wrong answer is silent. `v4.rs` carries the argument.
pub mod v4;

// The seam's vocabulary, hoisted to the crate root: `rivoli_engine::Engine` is what a
// consumer programs against, and making it name the module the enum happens to live in is
// one more detail the second consumer could get wrong.
pub use seam::{
    ArchCfg, DecodeStats, Decoded, Engine, GenSpec, OpenSpec, PoolKnobs, Scored, TokenSink,
};
