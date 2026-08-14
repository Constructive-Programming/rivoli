//! # rivoli-core — pure planning, no device, no formats, no features
//!
//! Everything here is a total function over plain data: residency partitioning,
//! arena geometry, cache policies, traversal spans, the (arch × flag) legality
//! table, the gate taxonomy, tolerances-with-provenance, and the Belady replay.
//! The engine crate is the interpreter that spends the values emitted here on a
//! device; nothing in this crate can name a stream, a pointer, or a weight format.
//!
//! The layering is load-bearing, not stylistic: `Fmt` lives in `rivoli-artifact`,
//! which this crate does not depend on, so a residency decision that selects
//! arithmetic — the old tree's hybrid defect, where `--max-mem` changed the output
//! text — cannot be expressed here at all.

pub mod hash;
