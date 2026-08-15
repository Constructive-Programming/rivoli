//! The GLM-5.2 engine arm: pin (resident placement + routed pool) and, next, the layer
//! loop. One arm per architecture — four loop bodies in four files is a measured
//! decision (`old:` zero cross-file clones between the loops), and a pin parameterised
//! by an arch flag is a GLM-shaped placement path one `if` away from running on the
//! wrong artifact.
//!
//! M4 scope (owner's design answers, 2026-08-15): single routed format per run, dense
//! attention, no MTP. The Rows dimension is designed in from day one anyway — Glimmer's
//! 72-minute prompt is what a rows retrofit costs — so [`MAXROW`] exists before anything
//! batches.

mod attn;
pub mod decode;
pub mod engine;
mod forward;
mod mlp;
pub mod pin;

/// The most token rows one forward pass may carry. 2 is the MTP verify-pass shape; MTP
/// itself is deferred past parity (M5+), but the pin's batch bound and the loop's
/// scratch are sized for it NOW because the union of `MAXROW` rows' expert picks is what
/// [`crate::routed::RoutedPool::submit`] must hold — a bound discovered mid-run is the
/// failure the pool's startup check exists to move to startup.
pub const MAXROW: usize = 2;
