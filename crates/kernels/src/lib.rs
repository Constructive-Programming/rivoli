//! MINIMAL SKELETON (Phase 0b): Phase 0a owns this file. The prelude needs exactly these
//! features; everything else here is 0a's.
#![no_std]
#![feature(
    core_intrinsics,
    link_llvm_intrinsics,
    stdarch_amdgpu,
    asm_experimental_arch,
    f16
)]
// `core_intrinsics`/`link_llvm_intrinsics` are internal features, and they are the only way
// to name the exact LLVM intrinsic hipcc emits (`prelude::llvm`, `fmuladd`, `exp`).
#![allow(internal_features)]

pub mod prelude;
