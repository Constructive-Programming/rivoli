//! Scalar element codecs: bf16, fp8-e4m3, OCP MX e2m1. One number in, one number
//! out — port of `formats.hpp` at tag `archive/hip-kernels`.
//!
//! Every function is total over its input type and is proven bit-identical to its C++
//! original by an EXHAUSTIVE sweep of that type on the device (rust-kernels.md §5); the
//! measurement history behind each body stays in the tagged header, which this does not
//! repeat.
//!
//! One `fmuladd`, matching hipcc's ISA: the only fused site hipcc emits is `e4m3f`'s
//! `1 + m·0.125`, which is exact for every `m` in 0..=7, and it is reproduced anyway so the
//! two ISAs stay comparable. `e4m3_subnormal_mantissa` carries
//! `contract(off)` in C++; Rust never contracts, so its plain `*`/`-` IS that pragma.

use core::intrinsics::{floor, fmuladd};

use super::llvm::ldexp;

/// bf16 → f32: bf16 is the high half of an f32.
#[inline(always)]
pub fn bf16f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// f32 → bf16, round-to-nearest-even. Non-finite keeps its top half verbatim, because an RNE
/// carry could turn a NaN into an Inf.
#[inline(always)]
pub fn f2bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if !x.is_finite() {
        return (b >> 16) as u16;
    }
    let r = ((b >> 16) & 1) + 0x7fff;
    (b.wrapping_add(r) >> 16) as u16
}

/// bf16 round trip in f32.
#[inline(always)]
pub fn rbf16(x: f32) -> f32 {
    bf16f(f2bf16(x))
}

/// fp8-e4m3 → f32. exp==0 is subnormal `±(m/8)·2^-6`; `(exp,mant)==(15,7)` is NaN.
#[inline(always)]
pub fn e4m3f(b: u8) -> f32 {
    // Widened first, as C's integer promotion does: u8 arithmetic would run in 16-bit
    // registers and add a conversion hipcc's ISA does not have.
    let b = i32::from(b);
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (b >> 3) & 0x0f;
    let mant = (b & 0x07) as f32;
    if exp == 0 {
        return sign * (mant * 0.125) * 0.015625;
    }
    if exp == 15 && mant == 7.0 {
        return f32::from_bits(0x7fc0_0000);
    }
    sign * fmuladd(mant, 0.125, 1.0) * ldexp(1.0, exp - 7)
}

/// e2m1 nibble (low 4 bits of `nib`; the rest is ignored) → f32. The eight magnitudes doubled
/// are {0,1,2,3,4,6,8,12}, one nibble each of the immediate `0xC8643210`; the sign is OR'd
/// in, so code 8 is −0.0.
#[inline(always)]
pub fn e2m1f(nib: u32) -> f32 {
    let half = (0xC864_3210u32 >> ((nib & 7) << 2)) & 0xF;
    let mag = 0.5 * half as f32;
    f32::from_bits(mag.to_bits() | ((nib & 8) << 28))
}

/// The round-to-nearest-EVEN decision: past the midpoint rounds up, on it the low bit of what
/// is kept decides. Generic because the subnormal path compares floats and the normal path
/// integers.
#[inline(always)]
fn rne_rounds_up<T: PartialOrd>(rem: T, half: T, keep: u32) -> bool {
    if rem != half {
        return rem > half;
    }
    keep & 1 != 0
}

/// e4m3's subnormal mantissa for `a` in (2^-10, 2^-6): `m·2^-9`, m in 1..=8 (8 promotes to
/// the smallest normal; the caller applies that).
#[inline(always)]
fn e4m3_subnormal_mantissa(a: f32) -> u32 {
    let scaled = a * 512.0;
    let m = floor(scaled);
    let rem = scaled - m;
    // SAFETY (both): `a` < 2^-6 so `m` <= 8.0 and `m + 1.0` <= 9.0, finite, non-negative.
    // The `+ 1.0` stays in float, as C++ has it, so the two ISAs stay comparable.
    let up = rne_rounds_up(rem, 0.5, unsafe { m.to_int_unchecked::<u32>() });
    unsafe { (if up { m + 1.0 } else { m }).to_int_unchecked::<u32>() }
}

/// A finite magnitude taken apart: its raw f32 word, its unbiased exponent, and the sign
/// bit the caller already extracted (the word is the MAGNITUDE's, so it no longer carries
/// it). One value, as in C++ — and it keeps this file inside CodeScene's primitive-argument
/// budget, which the C++ header measured and this port hit on its first scoring.
#[derive(Clone, Copy)]
struct SplitF32 {
    bits: u32,
    exp: i32,
    sign: u8,
}

/// e4m3's normal path: keep 3 mantissa bits, RNE on the 20 dropped ones, re-bias, and attach
/// the sign.
#[inline(always)]
fn e4m3_normal_code(v: SplitF32) -> u8 {
    let mant = v.bits & 0x7f_ffff;
    let rem = mant & 0xf_ffff;
    let mut m3 = mant >> 20;
    if rne_rounds_up(rem, 0x8_0000, m3) {
        m3 += 1;
    }
    let mut e = v.exp + 7;
    if m3 == 8 {
        m3 = 0;
        e += 1;
    }
    // Unreachable behind `f2e4m3_rne`'s `a >= 464` return; the format's own bound.
    if e >= 15 && m3 >= 7 {
        return v.sign | 0x7e;
    }
    v.sign | (e << 3) as u8 | m3 as u8
}

/// f32 → e4m3, round-to-nearest-EVEN all the way down (CUDA's `cvt.rn.satfinite`),
/// saturating at ±448. NOT `fwd.hip`'s `f2e4m3`, which rounds subnormal ties away from zero.
#[inline(always)]
pub fn f2e4m3_rne(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let a = x.abs();
    // 464 is the midpoint between 448 and the absent next code.
    if a >= 464.0 {
        return sign | 0x7e;
    }
    // EXACTLY 2^-10 ties between 0 (even) and 2^-9 (odd), so `<=`.
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

/// f32 → e2m1 nibble, round-to-nearest-EVEN, saturating at ±6. Counts midpoints passed (the
/// count IS the code, so saturation falls out); ties go up at odd-indexed midpoints. NaN has
/// no midpoint below it, so it gives code 0 with NaN's sign — the oracle's answer too.
#[inline(always)]
pub fn f2e2m1(x: f32) -> u32 {
    const MID: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let a = x.abs();
    let code = MID
        .iter()
        .enumerate()
        // `|`/`&`, not `||`/`&&`: the same truth table, but the short-circuit forms compiled to
        // per-lane exec-mask branches (`v_cmpx_eq_f32` + `s_cbranch_execz` at the three odd
        // midpoints) where clang selects (Phase 1 integration, 2026-11-19).
        .map(|(i, &m)| u32::from((a > m) | ((a == m) & (i & 1 == 1))))
        .sum::<u32>();
    if x.is_sign_negative() { 8 | code } else { code }
}
