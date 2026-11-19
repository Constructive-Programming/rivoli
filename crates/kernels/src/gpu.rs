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

/// The elementwise kernels' index, `blockIdx.x * ELEMENTWISE_BLOCK + threadIdx.x` — unsigned and
/// wrapping, as HIP computes it — read as the C's `int` for the signed bound test.
#[inline(always)]
pub fn elementwise_id() -> i32 {
    bid()
        .wrapping_mul(rivoli_abi::geom::ELEMENTWISE_BLOCK)
        .wrapping_add(tid()) as i32
}

/// The dispatch's dynamic LDS, as `T`. Sized by the launcher's `sharedMemBytes`, which
/// every launcher takes from the `rivoli_abi::geom` layout the kernel carves it by. HIP's
/// kernels declared static `__shared__` arrays; Rust has no LDS statics, and where the
/// bytes come from is invisible to the arithmetic.
#[inline(always)]
pub fn lds<T>() -> *mut T {
    core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<T>()
}

/// `for (int i = start; i < end; i += stride) body(i);` — the C's `int` strided loop, kept in
/// `i32` so the compare and the increment are the ones hipcc emitted, handing `body` the index
/// as `usize`. `start` and `stride` are thread/block quantities (`u32`), `end` the C's `int`
/// extent.
#[inline(always)]
pub fn for_i32(start: u32, end: i32, stride: u32, mut body: impl FnMut(usize)) {
    let mut i = start as i32;
    while i < end {
        body(i as usize);
        i += stride as i32;
    }
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
