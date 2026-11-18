//! `formats.hpp`: the scalar codecs these kernels use — bf16 rounding, e4m3 decode/encode.

use crate::gpu::fma;
use core::intrinsics::{exp2, floor};

/// `f2bf16`: round-to-nearest-even to bf16, non-finite values truncated (a NaN keeps its
/// payload's top half, an infinity stays one).
#[inline(always)]
fn f2bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if !x.is_finite() {
        return (b >> 16) as u16;
    }
    let r = ((b >> 16) & 1) + 0x7fff;
    (b.wrapping_add(r) >> 16) as u16
}

/// `rbf16`: the f32 a bf16 store and reload produces.
#[inline(always)]
pub fn rbf16(x: f32) -> f32 {
    f32::from_bits(u32::from(f2bf16(x)) << 16)
}

/// `e4m3f`: OCP e4m3 decode. Operand order follows the C expression exactly; the one fused
/// site is `1 + mant * 0.125` (hipcc's ISA: `v_fma_f32 v, 0x3e000000, v, 1.0`).
#[inline(always)]
pub fn e4m3f(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((b >> 3) & 0x0f);
    let mant = f32::from(b & 0x07);
    if exp == 0 {
        return sign * (mant * 0.125) * 0.015625;
    }
    if exp == 15 && mant == 7.0 {
        return f32::from_bits(0x7fc0_0000);
    }
    sign * fma(mant, 0.125, 1.0) * exp2((exp - 7) as f32)
}

#[inline(always)]
fn rne_rounds_up<T: PartialOrd>(rem: T, half: T, keep: u32) -> bool {
    if rem != half {
        return rem > half;
    }
    keep & 1 != 0
}

/// Carries `#pragma clang fp contract(off)` in C; Rust never contracts, so nothing to say.
#[inline(always)]
fn e4m3_subnormal_mantissa(a: f32) -> u32 {
    let scaled = a * 512.0;
    let mut m = floor(scaled);
    let rem = scaled - m;
    if rne_rounds_up(rem, 0.5, m as u32) {
        m += 1.0;
    }
    m as u32
}

/// `formats.hpp`'s `SplitF32`: the three facts `f2e4m3_rne` has already extracted, handed
/// to the normal-range encoder as one value, as the C does.
struct SplitF32 {
    bits: u32,
    exp: i32,
    sign: u8,
}

#[inline(always)]
fn e4m3_normal_code(SplitF32 { bits, exp, sign }: SplitF32) -> u8 {
    let mant = bits & 0x7f_ffff;
    let mut m3 = mant >> 20;
    let rem = mant & 0xf_ffff;
    if rne_rounds_up(rem, 0x8_0000, m3) {
        m3 += 1;
    }
    let mut exp = exp + 7;
    if m3 == 8 {
        m3 = 0;
        exp += 1;
    }
    if exp >= 15 && m3 >= 7 {
        return sign | 0x7e;
    }
    sign | ((exp << 3) as u8) | (m3 as u8)
}

/// `f2e4m3_rne`: OCP e4m3 encode, round-to-nearest-even, saturating to ±448.
#[inline(always)]
pub fn f2e4m3_rne(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    // `signbit(x) ? 0x80 : 0`, read straight off the sign bit.
    let sign = (x.to_bits() >> 24) as u8 & 0x80;
    let a = x.abs();
    if a >= 464.0 {
        return sign | 0x7e;
    }
    if a <= 0.000_976_562_5 {
        return sign;
    }
    let b = a.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127;
    if e < -6 {
        let mi = e4m3_subnormal_mantissa(a);
        return if mi >= 8 {
            sign | 0x08
        } else {
            sign | mi as u8
        };
    }
    e4m3_normal_code(SplitF32 {
        bits: b,
        exp: e,
        sign,
    })
}
