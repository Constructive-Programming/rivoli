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

pub use activation::*;
pub use residual::*;
pub use touch::*;

mod activation;
mod residual;
mod touch;
