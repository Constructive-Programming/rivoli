//! The GLSL numeric helpers' ALGORITHMS, against `math.rs`, on the CPU.
//!
//! `kernels/vk/common.glsl`'s `f2e4m3` and `f2bf16` are a bit-exactness contract, not
//! conveniences: `append_kv` writes their output into the KV cache and the oracle
//! compares it as BYTES. The risk in them is not GPU behaviour — it is mistranscribed
//! branch logic (round-half-away on the subnormal path, the `m == 8` promotion, the
//! non-finite passthrough) carried from `common.hpp` into GLSL by hand.
//!
//! WHAT THIS DOES AND DOES NOT PROVE. The functions below are LITERAL, statement-for-
//! statement transcriptions of the GLSL, so this tests the algorithm as written, not
//! the shader as compiled — the glslc optimiser and the driver still sit between this
//! and the real answer, and a transcription can drift from its original. The shader
//! itself is covered by `append_kv`'s GPU oracle in `tests/vk.rs`, which is
//! authoritative but reaches a few hundred values and needs the device. This reaches
//! ~1.2 million and needs nothing, so the two are complementary rather than redundant.
//!
//! **If you edit `f2e4m3` or `f2bf16` in common.glsl, edit these to match.** There is a
//! pointer to this file next to them.
//!
//! Not feature-gated: it depends only on `math.rs`, so it runs in the default build
//! where the Vulkan backend is not even compiled.

/// Literal transcription of `common.glsl::f2e4m3`. Deliberately un-idiomatic — it must
/// mirror the GLSL statement for statement, not read like good Rust.
fn glsl_f2e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign: u32 = if (x.to_bits() & 0x8000_0000) != 0 { 0x80 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return (sign | 0x7e) as u8;
    }
    if a < 0.000_976_562_5 {
        return sign as u8;
    }
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    if e < -6 {
        let m = (a * 512.0 + 0.5).floor() as u32;
        return if m >= 8 { (sign | 0x08) as u8 } else { (sign | m) as u8 };
    }
    let mant = bits & 0x007f_ffff;
    let mut m3 = mant >> 20;
    let rem = mant & 0x000f_ffff;
    let half_ulp = 0x0008_0000;
    if rem > half_ulp || (rem == half_ulp && (m3 & 1) != 0) {
        m3 += 1;
    }
    let mut exp = e + 7;
    if m3 == 8 {
        m3 = 0;
        exp += 1;
    }
    if exp >= 15 && m3 >= 7 {
        return (sign | 0x7e) as u8;
    }
    (sign | ((exp as u32) << 3) | m3) as u8
}

/// Literal transcription of `common.glsl::f2bf16`.
fn glsl_f2bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 {
        return (b >> 16) as u16; // inf/nan verbatim, as common.hpp does
    }
    ((b + (((b >> 16) & 1) + 0x7fff)) >> 16) as u16
}

/// Sweep every exponent e4m3 can represent, stepping the mantissa and landing exactly
/// on the tie points where round-to-nearest-even and round-half-away disagree.
/// Saturation above 448 and flush below 2^-10 make wider exponents uninteresting.
#[test]
fn glsl_f2e4m3_matches_math_rs() {
    let mut checked = 0u64;
    for exp in -14i32..=10 {
        for mstep in 0..4096u32 {
            let tie = if mstep % 3 == 0 { 0x0008_0000 } else { 0 };
            let mant = (mstep * (0x0080_0000 / 4096)) | tie;
            let bits = (((exp + 127) as u32) << 23) | (mant & 0x007f_ffff);
            for v in [f32::from_bits(bits), -f32::from_bits(bits)] {
                let (want, got) = (rivoli::math::f32_to_e4m3(v), glsl_f2e4m3(v));
                assert_eq!(want, got, "f2e4m3({v:e}) bits={:#x}", v.to_bits());
                checked += 1;
            }
        }
    }
    for v in [
        0.0f32,
        -0.0,
        448.0,
        -448.0,
        447.9,
        464.0,
        1e30,
        -1e30,
        0.000_976_562_5,
        2f32.powi(-9),
        2f32.powi(-10),
        2f32.powi(-6),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        f32::EPSILON,
    ] {
        assert_eq!(rivoli::math::f32_to_e4m3(v), glsl_f2e4m3(v), "f2e4m3({v:e})");
        checked += 1;
    }
    // NaN is contractually 0x7f whatever the sign or payload.
    for nan in [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xffff_ffff),
    ] {
        assert_eq!(glsl_f2e4m3(nan), 0x7f, "f2e4m3(NaN {:#x})", nan.to_bits());
        assert_eq!(rivoli::math::f32_to_e4m3(nan), 0x7f);
        checked += 1;
    }
    println!("f2e4m3: {checked} values, 0 mismatches");
}

#[test]
fn glsl_f2bf16_matches_math_rs_on_finite() {
    let mut checked = 0u64;
    for exp in -30i32..=30 {
        for mstep in 0..8192u32 {
            let tie = if mstep % 5 == 0 { 0x0000_8000 } else { 0 };
            let mant = (mstep * (0x0080_0000 / 8192)) | tie;
            let bits = (((exp + 127) as u32) << 23) | (mant & 0x007f_ffff);
            for v in [f32::from_bits(bits), -f32::from_bits(bits)] {
                let (want, got) = (rivoli::math::f32_to_bf16(v), glsl_f2bf16(v));
                assert_eq!(want, got, "f2bf16({v:e}) bits={:#x}", v.to_bits());
                checked += 1;
            }
        }
    }
    println!("f2bf16 finite: {checked} values, 0 mismatches");
}

/// The GLSL mirrors **HIP**, not `math.rs`, and on NaN those two disagree: `math.rs`
/// goes through `half::bf16::from_f32`, which forces the quiet bit, while
/// `common.hpp::f2bf16` passes the top 16 bits through verbatim.
///
/// Asserted rather than described, so that if anyone "fixes" the GLSL to match
/// `math.rs` they get a failing test pointing at docs/VULKAN.md instead of a silently
/// divergent backend. The kernels never quantize a NaN key in practice; this pins the
/// contract, it does not endorse it.
#[test]
fn glsl_f2bf16_diverges_from_math_rs_on_nan_by_design() {
    let sig = f32::from_bits(0x7f80_0001);
    assert_eq!(glsl_f2bf16(sig), 0x7f80, "GLSL/HIP: top 16 bits verbatim");
    assert_eq!(
        rivoli::math::f32_to_bf16(sig),
        0x7fc0,
        "math.rs via half: quiet bit forced"
    );
}
