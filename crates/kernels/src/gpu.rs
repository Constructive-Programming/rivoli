//! The device runtime the ported kernels stand on beside the prelude: thread ids, dynamic
//! LDS, explicit FMA, and the OCML bindings LTO-linked from ROCm's bitcode.

use core::arch::amdgpu;

/// `threadIdx.x`.
#[inline(always)]
pub fn tid() -> u32 {
    amdgpu::workitem_id_x()
}

/// `blockIdx.x`.
#[inline(always)]
pub fn bid() -> u32 {
    amdgpu::workgroup_id_x()
}

/// The dispatch's dynamic LDS, as `T`. Sized by the launcher's `sharedMemBytes`, which
/// every launcher takes from the `rivoli_abi::geom` layout the kernel carves it by. HIP's
/// kernels declared static `__shared__` arrays; Rust has no LDS statics, and where the
/// bytes come from is invisible to the arithmetic.
#[inline(always)]
pub fn lds<T>() -> *mut T {
    core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<T>()
}

/// A fused multiply-add, `a * b + c` with ONE rounding — `llvm.fma`, not `llvm.fmuladd`.
/// Used exactly where hipcc's ISA shows a `v_fma_f32`/`v_fmac_f32`: fmuladd would leave the
/// fuse-or-not decision to the backend, and the decision IS the bits.
#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    core::intrinsics::fma(a, b, c)
}

/// ROCm's OCML entry points, by the names HIP's `__clang_hip_math.h` maps the C calls to
/// (`pow`, `cos`, `sin`). The bodies come from `ocml.bc`, LTO-linked by
/// `crates/backend/build/device.rs` with the same `oclc_*` control libraries hipcc links, so
/// a call here inlines the code hipcc inlined.
pub mod ocml {
    unsafe extern "C" {
        #[link_name = "__ocml_pow_f64"]
        pub safe fn pow(x: f64, y: f64) -> f64;
        #[link_name = "__ocml_cos_f64"]
        pub safe fn cos(x: f64) -> f64;
        #[link_name = "__ocml_sin_f64"]
        pub safe fn sin(x: f64) -> f64;
    }
}
