//! Block activation quantization: the fused quantize-then-dequantize a V4 GEMM input goes
//! through (`kernel.py::act_quant`, fp8 block 128) and the indexer's fp4 simulator
//! (`fp4_act_quant`, block 32). Port of `actquant.hpp` at tag `archive/hip-kernels`.

use super::formats::{e2m1f, e4m3f, f2e2m1, f2e4m3_rne};

pub const FP8_MAX: f32 = 448.0;
pub const FP8_MAX_INV: f32 = 1.0 / FP8_MAX;
/// `T.max(amax, 1e-4)`: keeps an all-zero block from dividing by zero.
pub const ACT_QUANT_AMAX_FLOOR: f32 = 1e-4;

pub const FP4_QUANT_BLOCK: usize = 32;
pub const FP4_MAX: f32 = 6.0;
pub const FP4_MAX_INV: f32 = 1.0 / FP4_MAX;
/// `6·2^-126`, exact (2^-126 is f32's smallest normal).
pub const FP4_QUANT_AMAX_FLOOR: f32 = FP4_MAX * f32::MIN_POSITIVE;

/// `fast_round_scale` from kernel.py: `amax·max_inv` rounded UP to a power of two by bit
/// surgery. Not `ceil(log2(x))`, which rounds the log first and lands a value one ulp below
/// a power of two in the wrong binade. `contract(off)` in C++; inert (the product feeds a
/// bitcast) and implicit in Rust.
#[inline(always)]
pub fn fast_round_scale(amax: f32, max_inv: f32) -> f32 {
    let b = (amax * max_inv).to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + i32::from(b & 0x7f_ffff != 0);
    f32::from_bits(((e + 127) as u32) << 23)
}

/// The clamp both round trips use. Comparisons, NOT `minnum`/`maxnum`: those return the
/// non-NaN operand and would launder a NaN into `-max`, where the oracle propagates it.
#[inline(always)]
fn clamp_nan_propagating(q: f32, max: f32) -> f32 {
    if q < -max {
        -max
    } else if q > max {
        max
    } else {
        q
    }
}

/// One element of a block-128 group through e4m3 and back.
#[inline(always)]
pub fn act_quant_roundtrip(x: f32, s: f32) -> f32 {
    e4m3f(f2e4m3_rne(clamp_nan_propagating(x / s, FP8_MAX))) * s
}

/// One element of a block-32 group through e2m1 and back.
#[inline(always)]
pub fn fp4_quant_roundtrip(x: f32, s: f32) -> f32 {
    e2m1f(f2e2m1(clamp_nan_propagating(x / s, FP4_MAX))) * s
}
