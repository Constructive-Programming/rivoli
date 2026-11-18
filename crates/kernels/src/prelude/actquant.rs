//! `actquant.hpp`: the fp8 block-128 activation quantizer's per-element pieces.

use super::formats::{e4m3f, f2e4m3_rne};

pub const FP8_MAX: f32 = 448.0;
pub const FP8_MAX_INV: f32 = 1.0 / FP8_MAX;
pub const ACT_QUANT_AMAX_FLOOR: f32 = 1e-4;

/// `fast_round_scale`: the power of two at or above `amax * max_inv`, by `kernel.py`'s bit
/// surgery (rounding the log first would land one ulp below a power of two a binade low).
#[inline(always)]
pub fn fast_round_scale(amax: f32, max_inv: f32) -> f32 {
    let b = (amax * max_inv).to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + i32::from(b & 0x7f_ffff != 0);
    f32::from_bits(((e + 127) as u32) << 23)
}

/// `act_quant_roundtrip`: one element's fp8 quantize-dequantize. Comparisons rather than
/// `min`/`max` so a NaN propagates, as the C comment argues.
#[inline(always)]
pub fn act_quant_roundtrip(x: f32, s: f32) -> f32 {
    let q = x / s;
    let q = if q < -FP8_MAX {
        -FP8_MAX
    } else if q > FP8_MAX {
        FP8_MAX
    } else {
        q
    };
    e4m3f(f2e4m3_rne(q)) * s
}
