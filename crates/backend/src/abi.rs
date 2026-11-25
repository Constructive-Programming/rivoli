//! The `repr(C)` types that cross the kernel ABI wall, available featureless.
//!
//! Moved here from the old tree's `kvcompress.rs` when the backend became its own crate:
//! `hip.rs`'s launchers take these by pointer, and the backend depends on nothing above
//! libc, so the mirror structs live WITH the wall they mirror. The semantic constructors
//! (deriving geometry from a `LayerKind`) stay with the engine's compressor module, which
//! remains the sole intended producer — see `CompGeom::new`'s note.

// The two `repr(C)` structs moved to `rivoli-abi` 2026-11-18 — the `no_std` crate the Rust
// device crate compiles too — with their notes; re-exported so no path changed.
pub use rivoli_abi::{CompFinish, CompGeom, geom::SINK_ATTEND_MAX_N};

/// The four extents the sparse indexer's scoring contracts over.
///
/// A struct because they are four bare `usize` in a row and every one is plausible in any
/// other's position — `heads` and `hd` in particular are 64 and 128 on every layer that
/// has an indexer, so a transposed pair indexes a real row of `q` and produces a finite
/// score. What it does NOT buy, stated because the neighbouring types do enforce
/// something: there are no derived fields here and no relation between them to check, so
/// this is naming and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreDims {
    /// Query rows: the prompt length at prefill, 1 at decode.
    pub s: usize,
    /// Compressed blocks visible to this call, `(start_pos + seqlen) / ratio`.
    pub n_comp: usize,
    /// The indexer's OWN head count — NOT the attention's, which can coincide.
    pub heads: usize,
    /// The indexer's OWN head dim — NOT the attention's.
    pub hd: usize,
}
