//! # rivoli-oracles — frozen references
//!
//! CPU transliterations of first-party model code, the on-disk golden container,
//! and the anchor readers. The value of everything in this crate is that it was
//! written from the reference, not from rivoli's idea of the reference — so it
//! changes only when the reference does, and the engine must never call into it
//! on a decode path. Host-only and featureless.
