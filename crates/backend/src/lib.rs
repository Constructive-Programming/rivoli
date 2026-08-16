//! # rivoli-backend — the waist
//!
//! One backend (HIP/ROCm, gfx1151), one seam: streams, events, signals,
//! `block_on`, the kernel launchers and their ABI wall, and the compiled
//! `kernels/*.hip`. Everything above imports this seam and never a backend
//! symbol; "bytes and spans, no format names" is the proven contract — the old
//! tree's memory substrate crossed the V4 port without one line of change
//! because this boundary held. Owns the `rocm` feature; without it the crate
//! compiles empty and the workspace still builds and tests (that featureless
//! build is a real, CI-tested configuration, not breakage).

pub mod abi;

#[cfg(feature = "rocm")]
pub mod gpustream;
#[cfg(feature = "rocm")]
pub mod hip;
// The ABI wall's macro invocations, split out of `hip.rs` 2026-08-15 under the 800-line
// file ceiling — and a third, `hip_attn`, split out of `hip_blocks` 2026-08-16 when the M9
// launchers landed. PRIVATE and re-exported through `hip`, so the split is invisible from
// outside: `rivoli_backend::hip::launch_*` is still the only path, and which invocation file
// a launcher is declared in stays an authoring decision rather than an API one.
#[cfg(feature = "rocm")]
mod hip_attn;
#[cfg(feature = "rocm")]
mod hip_blocks;
#[cfg(feature = "rocm")]
mod hip_linalg;
// The waist itself — Signal, block_on, NULL_STREAM, and the backend-neutral re-exports.
// Its items sit at the crate root (`rivoli_backend::Stream`), exactly where the old tree's
// `crate::backend::Stream` sat relative to its consumers.
#[cfg(feature = "rocm")]
mod waist;
#[cfg(feature = "rocm")]
pub use waist::*;
