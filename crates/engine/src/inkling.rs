//! The Inkling 2 engine arm: the looped decoder (Mini applies its 12 core layers three times),
//! block-attention-residual query banks, short convolutions on every k/v path, attention sinks,
//! and the per-pass message gates. One arm per architecture — the loop body is unlike the four
//! on record (glm/glimmer/k3/v4), so it lives in its own files rather than behind an arch flag.
//!
//! `geometry` is NOT gated on the backend (arithmetic over a config, touching no device, built
//! and tested by the deviceless CI arm); the pin, the launch composition (`forward`), and the
//! decode driver (`decode`) are `rocm`-gated.

pub mod geometry;
pub mod names;

#[cfg(feature = "rocm")]
pub mod decode;
#[cfg(feature = "rocm")]
pub mod engine;
#[cfg(feature = "rocm")]
pub mod forward;
/// Small's routed `.mxfp8` dispatch — the two-stream MoE block, split out of `decode` by the
/// 800-line soft cap.
#[cfg(feature = "rocm")]
pub mod moe_mxfp8;
#[cfg(feature = "rocm")]
pub mod pin;
