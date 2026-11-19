//! Kernel argument blocks: ONE `repr(C)` struct per Rust kernel parameter list, which the
//! kernel takes by value (it lands in the kernarg segment as-is) and the launcher builds
//! and hands to `hipModuleLaunchKernel` as the single `kernelParams` entry.
//!
//! **This is the wall's parameter list, written once.** Under hipcc every kernel's
//! parameters were spelled three times — the `__global__` signature, the C launcher's
//! `hipLaunchKernelGGL` call, the Rust `extern` mirror — and only the compiler-blind
//! convention of "same order" tied them. Here the device destructures the struct the host
//! filled, so a field cannot be transposed between the two sides: there is nothing to
//! transpose against. Field order is free (it no longer has to match a C signature), so it
//! is chosen for packing — pointers first.
//!
//! The kernels that share a shape share a struct (`Gate` under all three gated combines,
//! `RowNorm` under both single-row norms): a struct per kernel would restate identical
//! field lists, which is the duplication this module exists to remove.
//!
//! One module per kernel family, re-exported flat (`rivoli_abi::kernarg::Gate`). Each struct
//! is pinned by a size assert on the line after it, which compiles on BOTH targets — the one
//! way a single definition could still disagree with itself is a target whose ABI lays it
//! out differently, and either compile would fail.

// ── the launch wrappers: a family block plus what only the launcher knows ───────────────────

/// A kernel whose block size the LAUNCHER chooses at run time (`next_pow2` of an extent) takes it
/// here rather than from the dispatch packet (Phase 0c: ~1.1 µs per launch). `block` is exactly the
/// `blockDim.x` the launcher passes, set by the launcher and never by a caller.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Blocked<A> {
    pub a: A,
    pub block: u32,
}

/// A grid-strided kernel's argument block plus the launch's total thread count — `gridDim.x ·
/// blockDim.x`, which the kernel would otherwise read from the dispatch packet (Phase 0c). Set by
/// the launcher from the grid it launches, never by a caller.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Strided<A> {
    pub a: A,
    pub threads: u64,
}

// Spelled unlike `geom`'s identical family list on purpose (brace re-export, `pub(crate) mod`):
// the two lists grow in lockstep, one line per family, and as plain `pub use x::*;` / `mod x;`
// runs they cross jscpd's 5-line floor at the fourth family (measured 2026-11-19, 26 tokens).
pub use self::{activation::*, blockindex::*, indexer::*, kvcompress::*, residual::*, touch::*};

pub(crate) mod activation;
pub(crate) mod blockindex;
pub(crate) mod indexer;
pub(crate) mod kvcompress;
pub(crate) mod residual;
pub(crate) mod touch;
