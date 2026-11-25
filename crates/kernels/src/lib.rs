//! # rivoli-kernels — the GPU kernels, in Rust, for `amdgcn-amd-amdhsa`
//!
//! Built ONLY by `crates/backend/build/device.rs` under the `rocm` feature (this crate's manifest
//! argues why it is not a workspace member), linked against ROCm's device libraries, and
//! embedded in `rivoli-backend` as one code object loaded once with `hipModuleLoadData`.
//!
//! Each kernel is a bit-for-bit twin of a hipcc kernel at tag `archive/hip-kernels`, and
//! `crates/engine/tests/bit_identity/` is the gate that says so — byte-identical outputs
//! over ≥ 64 seeds and adversarial inputs, or RED. The migration plan and its record are
//! `docs/investigations/rust-kernels.md`. What a port must get right, every time:
//!
//! - **FMA contraction.** hipcc fuses every `a * b + c` the backend can (`-ffp-contract=
//!   fast-honor-pragmas`); Rust never contracts. Every fused site is an explicit
//!   [`gpu::fma`], placed where hipcc's ISA shows a `v_fma`/`v_fmac` — read, never guessed.
//!   In `a·b + c·d` hipcc fuses the LEFT product: `fma(a, b, c·d)`. (The prelude spells its
//!   sites `fmuladd`, measured identical by 0b's device arms; unifying the two spellings is
//!   owed to a device run, not done blind.)
//! - **blockDim is never read from the dispatch packet** (Phase 0c, ~1.1 µs/launch). A
//!   launcher-fixed block size is a `rivoli_abi::geom` constant both sides name.
//! - **Math functions are ROCm's own.** A `.hip` call into `__ocml_*` (`pow`, `sin`, `cos`)
//!   is the SAME symbol here, LTO-linked from the device library hipcc links
//!   ([`gpu::ocml`]); `tanhf` is the prelude's transliteration of ocml's body. A
//!   `__builtin_*` call (`expf`, `sqrtf`) is the same LLVM intrinsic through
//!   `core::intrinsics`, and one `core` only has a lookalike of (`fmaxf` → `llvm.maxnum`,
//!   not `f32::max`) is in `prelude::llvm`.
#![no_std]
#![feature(abi_gpu_kernel, core_intrinsics, gpu_intrinsics, stdarch_amdgpu)]
#![feature(asm_experimental_arch, link_llvm_intrinsics, f16)]
// `core_intrinsics` is how the LLVM intrinsics hipcc's builtins lower to are reached at all.
#![allow(internal_features)]
// Kernels are raw-pointer code by construction: every pointer is a device buffer whose
// extent the launcher's guards establish, which is the `# Safety` contract the whole crate
// shares. Per-site `unsafe` blocks inside an `unsafe extern "gpu-kernel" fn` would restate
// it at every load.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
// Every kernel takes one `rivoli_abi::kernarg` struct by value. The lint warns that rustc's
// gpu-kernel ABI for aggregates is not promised stable; what is relied on here was CHECKED
// (2026-11-18): each kernel's metadata is ONE `by_value` arg of `size_of` the struct, and
// the ISA's `s_load` offsets are its `repr(C)` field offsets (`swiglu`: `n` at 0x18, `h` at
// 0x10). The bit-identity gate exercises every kernel with an output through that path —
// all but `touch_region`, whose only argued stand-in is its ISA and the determinism gate
// (`kernel_coverage.rs`'s `UNROWED`) — so a toolchain that changed the lowering would go red
// there, not silently.
#![allow(improper_gpu_kernel_arg)]

mod activation;
mod attn;
mod blockindex;
mod fwd;
mod gpu;
mod headtail;
mod indexer;
mod kvcompress;
mod linalg;
mod mla;
// The `common.hpp` port (Phase 0b, merged 2026-11-18). `pub` because in a cdylib with
// `warnings = deny` a private module's not-yet-called helpers (`dot_fp8`, the probes-only
// paths) are denied dead code; it can go private when Phase 1 calls them all. (The nibble and
// VQ dots and `moe_fixed` are called since L2's MoE port.)
pub mod prelude;
mod moe;
mod moe_f4;
mod moe_mxfp8;
#[cfg(feature = "probes")]
mod probes;
mod recurrent;
mod residual;
mod touch;

/// A kernel has no one to unwind to and no host to print to. A trap ends the dispatch with
/// a queue error the host sees as a failed launch — loud, where `loop {}` would hang the
/// stream and a silent `s_endpgm` would leave the output half-written and plausible.
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort_immediate()
}
