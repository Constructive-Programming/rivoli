//! The slice of `kernels/common.hpp` (and the headers it includes) that `activation.hip`
//! and `residual.hip` use, ported inline so Phase 0a could migrate those two files before
//! the prelude port (Phase 0b) lands. **0b's module replaces this one wholesale**; nothing
//! here is meant to outlive that merge, and each item names its C source so the swap is a
//! name-for-name check.
//!
//! | here | C source |
//! |---|---|
//! | `formats::rbf16` (`bf16f`, `f2bf16`) | `formats.hpp` |
//! | `formats::{e4m3f, f2e4m3_rne}` (+ `e4m3_subnormal_mantissa`, `e4m3_normal_code`, `rne_rounds_up`) | `formats.hpp` |
//! | `actquant::{fast_round_scale, act_quant_roundtrip}`, `FP8_MAX`, `FP8_MAX_INV`, `ACT_QUANT_AMAX_FLOOR` | `actquant.hpp` |
//! | `reduce::{wave_sum, wave_max, block_sum_lds}` | `reduce.hpp` |
//! | `swiglu_clamped`, `situ_glu` (here) | `common.hpp` |

//! Split by C header, the layout 0b's port mirrors, so the replacement is file-for-file.

pub mod actquant;
pub mod formats;
pub mod reduce;

pub use actquant::{ACT_QUANT_AMAX_FLOOR, FP8_MAX_INV, act_quant_roundtrip, fast_round_scale};
pub use formats::rbf16;
pub use reduce::{block_sum_lds, wave_max, wave_sum};

use crate::gpu::{fmaxf, fminf, ocml};
use core::intrinsics::exp;

/// `swiglu_clamped`: `silu(min(bf16(g), L)) · clamp(bf16(u), ±L)`, multiply form, UNROUNDED.
/// The four load-bearing points are `common.hpp`'s.
#[inline(always)]
pub fn swiglu_clamped(g: f32, u: f32, limit: f32) -> f32 {
    let gt = fminf(rbf16(g), limit); // ASYMMETRIC: no lower clamp on the gate
    let ut = fminf(fmaxf(rbf16(u), -limit), limit);
    gt * (1.0 / (1.0 + exp(-gt))) * ut
}

/// `situ_glu`: Kimi-K3's SiTU-GLU, `b1·tanh(g/b1)·sigmoid(g) · b2·tanh(u/b2)`, UNROUNDED.
#[inline(always)]
pub fn situ_glu(g: f32, u: f32, b1: f32, b2: f32) -> f32 {
    let a = b1 * ocml::tanh(g / b1) * (1.0 / (1.0 + exp(-g)));
    a * (b2 * ocml::tanh(u / b2))
}
