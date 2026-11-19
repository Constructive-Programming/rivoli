//! Launch geometry shared by a Rust kernel and its launcher — block sizes, dynamic-LDS
//! layouts, and the checkpoint constants a launcher refuses and a kernel indexes by.
//!
//! Every item is read by BOTH sides, or will be once its launcher lands (the two device-only
//! ones say so); a host-only constant stays beside its launcher. A
//! block size here is what the launcher passes as `blockDim` and what the kernel strides by
//! in place of reading it (the crate doc carries
//! the Phase 0c rule), and an LDS layout here is what the launcher sizes the dynamic
//! allocation by and what the kernel carves it with. A kernel that wants a different value
//! changes it HERE, and the launcher moves with it.
//!
//! One module per kernel family, re-exported flat (`rivoli_abi::geom::ROW_BLOCK`), each
//! constant's shape assert beside it, so a family's lane edits one file.

/// f32 is the only LDS element type so far; named so the byte counts read as such.
pub(crate) const F32: u32 = 4;

// ── shared by every family ───────────────────────────────────────────────────────────────

/// gfx11 is wave32. A kernel that assumes a wave width assumes THIS one.
pub const WAVE: u32 = 32;
/// Rows per block for the wave-per-row kernels (the prelude's dot families): 256 threads =
/// 8 waves = 8 output rows. The GEMV launchers' grid (`ceil(o_dim / ROWS_PER_BLOCK)`).
pub const ROWS_PER_BLOCK: u32 = 8;

/// `swiglu`, `swiglu_clamped_bf16`, `situ_glu_f32`, `hc_post`, `gated_residual_collapse_f32`:
/// `ceil(n / 256)` blocks of 256. The kernels index `blockIdx.x * ELEMENTWISE_BLOCK +
/// threadIdx.x`, so a launch at any other width would skip or repeat elements.
pub const ELEMENTWISE_BLOCK: u32 = 256;

mod activation;
mod attn;
mod blockindex;
mod fwd;
mod indexer;
mod kvcompress;
mod linalg;
mod mla;
mod residual;
mod touch;

pub use activation::*;
pub use attn::*;
pub use blockindex::*;
pub use fwd::*;
pub use indexer::*;
pub use kvcompress::*;
pub use linalg::*;
pub use mla::*;
pub use residual::*;
pub use touch::*;
