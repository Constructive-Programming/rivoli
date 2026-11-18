//! The LLVM intrinsics hipcc's IR calls that `core` does not expose under the SAME
//! intrinsic. Bit-identity is decided by which intrinsic reaches the backend, so each one
//! here names the exact call read out of `hipcc -O3 --cuda-device-only -S -emit-llvm` on
//! the `archive/hip-kernels` headers (ROCm clang 23.0.0git), not a lookalike:
//!
//! * `fminf`/`fmaxf` are `llvm.minnum`/`llvm.maxnum`; `f32::min`/`max` lower to
//!   `minimumnum`/`maximumnum`, which disagree on signalling NaN.
//! * `llrintf` is `llvm.rint` + `fptosi`; core's `round_ties_even` is `llvm.roundeven`.
//! * `exp2f((float)int)` is folded by clang to `llvm.ldexp(1.0, int)`.
//! * `tanhf` inlines ocml's body, which reciprocates with `llvm.amdgcn.rcp` (1 ulp, not a
//!   correctly-rounded divide).
//! * `__shfl_down`/`__shfl_xor` are `llvm.amdgcn.wave.shuffle`. The optimizer turns the
//!   pattern it recognises (`wave_sum`'s delta-16 step) into `ds_swizzle` and the rest into
//!   `ds_bpermute`, identically on both sides; core's `ds_bpermute` would skip the swizzle.

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.maxnum.f32"]
    pub safe fn maxnum(a: f32, b: f32) -> f32;
    #[link_name = "llvm.minnum.f32"]
    pub safe fn minnum(a: f32, b: f32) -> f32;
    #[link_name = "llvm.rint.f32"]
    pub safe fn rint(x: f32) -> f32;
    #[link_name = "llvm.ldexp.f32.i32"]
    pub safe fn ldexp(x: f32, e: i32) -> f32;
    #[link_name = "llvm.amdgcn.rcp.f32"]
    pub safe fn rcp(x: f32) -> f32;
    #[link_name = "llvm.amdgcn.wave.shuffle.i32"]
    safe fn wave_shuffle_i32(value: i32, lane: i32) -> i32;
}

/// `lane`'s `v`, as `__shfl` hands a float through the i32 intrinsic: a bitcast each way, so
/// every bit pattern (NaN payloads included) crosses unchanged.
#[inline(always)]
pub fn wave_shuffle(v: f32, lane: u32) -> f32 {
    f32::from_bits(wave_shuffle_i32(v.to_bits() as i32, lane as i32) as u32)
}
