//! The device runtime every kernel stands on: thread geometry, the workgroup barrier,
//! dynamic LDS, wave shuffles, explicit FMA, and the ROCm math-library bindings. Owned by
//! the infra phase — the `common.hpp` port (`prelude`) builds on this, not the reverse.

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

/// `__syncthreads()` — HIP's workgroup-scope release fence, `s_barrier`, acquire fence, as
/// gfx1151 (WGP mode) lowers them.
///
/// **Not the spike's barrier, which was under-fenced.** The spike spelled only
/// `s_waitcnt lgkmcnt(0)` before the barrier, copied from `rmsnorm_single`'s ISA — where
/// hipcc's waitcnt pass had ELIDED the global-memory waits because no global op was in
/// flight. `rope_interleave` shows the full release: `s_waitcnt vmcnt(0)` precedes its
/// barrier, because there each thread's global LOADS must complete before any thread's
/// in-place STORES land (a write-after-read across lanes). The asm cannot be elided the way
/// hipcc's waits are, so it carries all three counters; a wait on a counter already at zero
/// costs a cycle.
///
/// The barrier itself is the `llvm.amdgcn.s.barrier` intrinsic, not asm: the intrinsic is
/// `convergent`, so LLVM cannot duplicate it into divergent branches, and Rust inline asm
/// is not. The intrinsic is also `IntrNoMem` — it does not order memory for the optimizer —
/// which is what the two asm blocks are for: each is opaque memory access, so no load or
/// store moves across either one, and therefore none crosses the barrier between them.
#[inline(always)]
pub fn sync_threads() {
    // SAFETY: waits and a cache invalidate; no registers or memory are written.
    unsafe {
        core::arch::asm!("s_waitcnt vmcnt(0) lgkmcnt(0)", "s_waitcnt_vscnt null, 0x0");
    }
    amdgpu::s_barrier();
    // SAFETY: as above.
    unsafe {
        core::arch::asm!("buffer_gl0_inv");
    }
}

/// The LDS halving ladder every block reduction here shares: for `s = LEAVES/2, …, 1`,
/// threads `t < s` run `step(t, t + s)`, then the whole block barriers. The caller has
/// stored its leaves and barriered first; `step` is what one level does (one tree, or two
/// side by side). `LEAVES` replaces the C's `blockDim.x` — the Phase 0c rule.
///
/// # Safety
/// Every thread of the block calls this (it barriers), and `step` touches only LDS the
/// caller owns.
#[inline(always)]
pub unsafe fn tree<const LEAVES: u32>(mut step: impl FnMut(usize, usize)) {
    let t = tid();
    let mut s = LEAVES / 2;
    while s > 0 {
        if t < s {
            step(t as usize, (t + s) as usize);
        }
        sync_threads();
        s >>= 1;
    }
}

/// The dispatch's dynamic LDS, as `T`. Sized by the launcher's `sharedMemBytes`, which
/// every launcher takes from the `rivoli_abi::geom` layout the kernel carves it by. HIP's
/// kernels declared static `__shared__` arrays; Rust has no LDS statics, and where the
/// bytes come from is invisible to the arithmetic.
#[inline(always)]
pub fn lds<T>() -> *mut T {
    core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<T>()
}

/// `__lane_id()` on wave32.
#[inline(always)]
fn lane_id() -> u32 {
    amdgpu::mbcnt_lo(!0, 0)
}

/// `__shfl_down(v, o, WAVE)`: lane `l` reads lane `l + o`, or keeps its own value when that
/// is past the wave — HIP's exact rule (`amd_warp_functions.h`). A shuffle moves bits, so
/// any correct implementation is bit-identical to hipcc's (which lowers through
/// `llvm.amdgcn.wave.shuffle`).
#[inline(always)]
pub fn shfl_down(v: f32, o: u32) -> f32 {
    let l = lane_id();
    bpermute(
        if l + o >= rivoli_abi::geom::WAVE {
            l
        } else {
            l + o
        },
        v,
    )
}

/// `__shfl_xor(v, o, WAVE)` for `o < WAVE`, where the partner is always in the wave.
#[inline(always)]
pub fn shfl_xor(v: f32, o: u32) -> f32 {
    bpermute(lane_id() ^ o, v)
}

/// Read `v` from `src_lane`. Every call site runs on a whole wave (the reductions), so the
/// addressed lane is active and holds a defined value.
#[inline(always)]
fn bpermute(src_lane: u32, v: f32) -> f32 {
    // SAFETY: see the doc — the source lane is active.
    f32::from_bits(unsafe { amdgpu::ds_bpermute(src_lane << 2, v.to_bits()) })
}

/// A fused multiply-add, `a * b + c` with ONE rounding — `llvm.fma`, not `llvm.fmuladd`.
/// Used exactly where hipcc's ISA shows a `v_fma_f32`/`v_fmac_f32`: fmuladd would leave the
/// fuse-or-not decision to the backend, and the decision IS the bits.
#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    core::intrinsics::fma(a, b, c)
}

// `safe` items: pure scalar functions with no preconditions, so the declaration says so and
// no wrapper restates each signature to add an `unsafe` block.
unsafe extern "llvm-intrinsic" {
    /// `fmaxf` — `llvm.maxnum`, what HIP's `__builtin_fmaxf` emits. **Not `f32::max`**,
    /// which lowers to `llvm.maximumnum` (core's `maximum_number_nsz`): a different
    /// intrinsic with a different signalling-NaN rule. Flagged by the prelude port (0b),
    /// checked against core's source 2026-11-18.
    #[link_name = "llvm.maxnum.f32"]
    pub safe fn fmaxf(a: f32, b: f32) -> f32;
    /// `fminf` — `llvm.minnum`; see [`fmaxf`].
    #[link_name = "llvm.minnum.f32"]
    pub safe fn fminf(a: f32, b: f32) -> f32;
}

/// ROCm's OCML entry points, by the names HIP's `__clang_hip_math.h` maps the C calls to
/// (`tanhf`, `pow`, `cos`, `sin`). The bodies come from `ocml.bc`, LTO-linked by
/// `crates/backend/build.rs` with the same `oclc_*` control libraries hipcc links, so a call
/// here inlines the code hipcc inlined.
pub mod ocml {
    unsafe extern "C" {
        #[link_name = "__ocml_tanh_f32"]
        pub safe fn tanh(x: f32) -> f32;
        #[link_name = "__ocml_pow_f64"]
        pub safe fn pow(x: f64, y: f64) -> f64;
        #[link_name = "__ocml_cos_f64"]
        pub safe fn cos(x: f64) -> f64;
        #[link_name = "__ocml_sin_f64"]
        pub safe fn sin(x: f64) -> f64;
    }
}
