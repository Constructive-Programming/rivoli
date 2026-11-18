//! The device prelude every kernel uses: port of `common.hpp` and the eight headers it
//! includes, at tag `archive/hip-kernels` (= migration base `abd065fe`). C++ templates are
//! const generics (`R` rows, `U` unroll depth) and a trait (`dot_nib::NibFormat`).
//!
//! | module         | C++ origin                | owns                                       |
//! |----------------|---------------------------|--------------------------------------------|
//! | `formats`      | formats.hpp               | scalar codecs: bf16, e4m3, e2m1            |
//! | `actquant`     | actquant.hpp              | block activation quantization              |
//! | `activation`   | common.hpp                | clamped SwiGLU, SiTU-GLU (+ ocml tanh)     |
//! | `reduce`       | reduce.hpp                | geometry, barrier, wave/block reductions   |
//! | `rowview`      | rowview.hpp               | by-value views, the shared quad, unrolling |
//! | `dot_fp8`      | dot_fp8.hpp               | fp8 tile-scaled dot + LUT                  |
//! | `dot_nib`      | dot_i4.hpp + dot_f4.hpp   | int4 / fp4 dots (one loop), e8m0 decode    |
//! | `dot_vq`       | dot_vq.hpp                | VQ-int3 codebook dot                       |
//! | `llvm`         | (hipcc's IR)              | intrinsics `core` has only lookalikes of   |
//!
//! Bit-identity against the C++ was measured on the device by 0b's probe harness, which still
//! lives in scratch (`rk-prelude/probe`, rust-kernels.md §5); moving its families into
//! `tests/bit-identity.sh` as rows is owed, so until then it cannot go red for anyone else.

pub mod activation;
pub mod actquant;
pub mod dot_fp8;
pub mod dot_nib;
pub mod dot_vq;
pub mod formats;
pub mod llvm;
pub mod reduce;
pub mod rowview;

pub use activation::{situ_glu, swiglu_clamped};

use llvm::{maxnum, minnum, rint};

/// Expert contributions are summed as integers at scale 2^-44, which makes the MoE sum
/// associative (independent of stream completion order). `MAX` derives from `SHIFT` so that
/// 16 clamped terms stay inside i64: `16·MAX·2^SHIFT <= 2^62`. The sizing argument (measured
/// partials |v| <= 15.25 over 21.5M samples) is in the tag's `common.hpp`.
pub const MOE_ACC_SHIFT: u32 = 44;
pub const MOE_ACC_SCALE: f32 = (1u64 << MOE_ACC_SHIFT) as f32;
pub const MOE_ACC_MAX: f32 = (1u64 << (58 - MOE_ACC_SHIFT)) as f32;

/// f32 → fixed point, SATURATING rather than wrapping: a wrapped overflow is a finite wrong
/// value `flag_nonfinite` cannot see. `maxnum`/`minnum` also launder a NaN into a finite
/// bound, as C++'s `fminf`/`fmaxf` do.
#[inline(always)]
pub fn moe_fixed(v: f32) -> u64 {
    let r = rint(minnum(maxnum(v, -MOE_ACC_MAX), MOE_ACC_MAX) * MOE_ACC_SCALE);
    // SAFETY: the clamp bounds `r` to ±2^58, finite and inside i64.
    unsafe { r.to_int_unchecked::<i64>() as u64 }
}

/// i-quads per head in `mla_absorb_fp8`. In C++ ONE definition served the kernel and the
/// launcher's grid so the two could not disagree (a short grid leaves columns unwritten,
/// silently); the Rust launcher is host code and cannot import this device crate. That crate
/// now exists (`rivoli_abi::geom`, merged 2026-11-18) and is where this moves when L3 ports
/// `mla_absorb_fp8` and its launcher — the tie stays OPEN until then.
#[inline(always)]
pub const fn absorb_nquad(kvl: i32) -> i32 {
    kvl.wrapping_add(3) >> 2
}
