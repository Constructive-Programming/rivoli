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
//! **If you edit `f2e4m3` or `f2bf16` in common.glsl, edit these to match.** That is
//! not left to a comment: `transcriptions_still_match_the_glsl` hashes the two function
//! bodies out of common.glsl and fails if they have changed, telling you to re-verify
//! and update the constant. Naming a drift risk does not contain it — a stale
//! transcription would keep passing while testing a function the shader no longer has,
//! which is the same shape as a deleted helper or an empty log sink, one level removed.
//!
//! Not feature-gated: it depends only on `math.rs`, so it runs in the default build
//! where the Vulkan backend is not even compiled.
#![allow(clippy::expect_used)]

/// Literal transcription of `common.glsl::f2e4m3`. Deliberately un-idiomatic — it must
/// mirror the GLSL statement for statement, not read like good Rust.
fn transcribed_f2e4m3(x: f32) -> u8 {
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

/// Literal transcription of `common.glsl::e4m3f` — the DECODE direction, which arrived
/// with `gemv_fp8` and its LUT.
fn transcribed_e4m3f(b: u32) -> f32 {
    let sign: f32 = if (b & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 0 {
        return sign * (mant * 0.125) * 0.015625;
    }
    if exp == 15 && mant == 7.0 {
        return f32::from_bits(0x7fc0_0000);
    }
    sign * (1.0 + mant * 0.125) * f32::from_bits(((exp - 7 + 127) as u32) << 23)
}

/// Literal transcription of `common.glsl::f2bf16`.
fn transcribed_f2bf16(x: f32) -> u16 {
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
                let (want, got) = (rivoli::math::f32_to_e4m3(v), transcribed_f2e4m3(v));
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
        assert_eq!(rivoli::math::f32_to_e4m3(v), transcribed_f2e4m3(v), "f2e4m3({v:e})");
        checked += 1;
    }
    // NaN is contractually 0x7f whatever the sign or payload.
    for nan in [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0xffff_ffff),
    ] {
        assert_eq!(transcribed_f2e4m3(nan), 0x7f, "f2e4m3(NaN {:#x})", nan.to_bits());
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
                let (want, got) = (rivoli::math::f32_to_bf16(v), transcribed_f2bf16(v));
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
    assert_eq!(transcribed_f2bf16(sig), 0x7f80, "GLSL/HIP: top 16 bits verbatim");
    assert_eq!(
        rivoli::math::f32_to_bf16(sig),
        0x7fc0,
        "math.rs via half: quiet bit forced"
    );
}


// ---------------------------------------------------------------------------
// Drift guard
// ---------------------------------------------------------------------------

/// FNV-1a. Hand-rolled because `DefaultHasher` is explicitly NOT stable across Rust
/// releases — a checked-in constant computed from it would start failing on a toolchain
/// bump, and the natural response to a mystery failure is to paste in the new number,
/// which is exactly the reflex this guard exists to prevent.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The body of a GLSL function, from its signature to its matching close brace.
///
/// Deliberately narrower than hashing the whole file: comments and unrelated helpers in
/// common.glsl change often, and a guard that cries wolf on every edit gets its constant
/// updated without thought. This fires when the ALGORITHM changes, which is when the
/// transcriptions below actually need re-checking.
fn glsl_fn_body(src: &str, signature: &str) -> String {
    let start = src.find(signature).unwrap_or_else(|| {
        panic!(
            "kernels/vk/common.glsl no longer contains `{signature}`. If it was renamed \
             or removed, update the transcription in tests/glsl_numerics.rs to match \
             and fix this signature."
        )
    });
    let bytes = src.as_bytes();
    let mut i = start + signature.len(); // signature includes the opening brace
    let mut depth = 1usize;
    while depth > 0 {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    src[start..i].to_string()
}

/// Every GLSL function this file transcribes, as `(name, signature)`. The hash below
/// covers all of them, in this order.
///
/// A LIST, not two hardcoded calls, because the previous version pinned `f2bf16` and
/// `f2e4m3` BY NAME — which silently meant it did not cover anything else, and would not
/// have covered the next transcription either, with nothing to say so. That is the same
/// shape as a guard no test distinguishes from its absence: correct, present, and
/// narrower than a reader assumes. `every_transcription_is_locked` below turns
/// "remember to add it here" into a test failure.
const LOCKED: &[(&str, &str)] = &[
    ("f2bf16", "uint f2bf16(float x) {"),
    ("f2e4m3", "uint f2e4m3(float x) {"),
    ("e4m3f", "float e4m3f(uint b) {"),
];

/// Hash of every [`LOCKED`] function body as it stands in common.glsl, concatenated in
/// order. Update ONLY after checking that the transcriptions above still mirror the GLSL
/// statement for statement.
const GLSL_NUMERICS_HASH: u64 = 0x92c6_d0fe_121f_98ca;

/// The transcriptions above are only evidence while they still correspond to the
/// shader. This makes that correspondence a build-visible obligation rather than a
/// comment: touch the GLSL, and the test tells you to re-verify.
#[test]
fn transcriptions_still_match_the_glsl() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/vk/common.glsl");
    let src = std::fs::read_to_string(path).expect("read common.glsl");
    let mut joined = String::new();
    for (_, signature) in LOCKED {
        joined.push_str(&glsl_fn_body(&src, signature));
    }
    let got = fnv1a(joined.as_bytes());
    assert_eq!(
        got, GLSL_NUMERICS_HASH,
        "\n\nf2e4m3/f2bf16 in kernels/vk/common.glsl have CHANGED.\n\
         The literal Rust transcriptions in this file may no longer mirror them, and \
         until you check, this file's ~1.2M-value pass proves nothing about the \
         shader.\n\
         1. Re-read both GLSL functions against glsl_f2e4m3/glsl_f2bf16 here.\n\
         2. Update them if they diverged.\n\
         3. Set GLSL_NUMERICS_HASH = {got:#018x}\n"
    );
}


/// Every `glsl_*` transcription in this file must appear in [`LOCKED`].
///
/// THE EIGHTH MECHANISED RULE. Without it, the lock covers exactly what someone
/// remembered to list: adding a transcription and forgetting the entry leaves a function
/// diffed against `math.rs` but NOT pinned to the shader, so the GLSL can drift away from
/// it silently — the transcription keeps passing while testing something the shader no
/// longer contains. That is the failure this whole file exists to prevent, reintroduced
/// one level up.
///
/// The convention it relies on is `fn transcribed_<name>` mirroring GLSL `<name>`. The
/// prefix is deliberately distinct from anything a TEST would be called: the first
/// version keyed on `glsl_` and matched the test functions too, reporting
/// `glsl_f2e4m3_matches_math_rs` as an unlocked transcription. A convention has to be
/// unambiguous before it can be mechanically checked.
#[test]
fn every_transcription_is_locked() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/glsl_numerics.rs");
    let me = std::fs::read_to_string(path).expect("read this file");
    let mut found = Vec::new();
    for line in me.lines() {
        let line = line.trim_start();
        if let Some(name) = line
            .strip_prefix("fn transcribed_")
            .and_then(|rest| rest.split('(').next())
        {
            found.push(name.to_string());
        }
    }
    assert!(
        !found.is_empty(),
        "no `fn transcribed_*` found — the naming convention this check keys on has \
         changed, and it has been passing without examining anything"
    );
    for name in &found {
        assert!(
            LOCKED.iter().any(|(locked, _)| locked == name),
            "\n\n`transcribed_{name}` is here but is NOT in LOCKED.\n\
             It is therefore diffed against math.rs but not pinned to the shader, so \
             kernels/vk/common.glsl can drift away from it and this file will keep \
             passing while testing a function the shader no longer has.\n\
             Add (\"{name}\", \"<its GLSL signature up to the opening brace>\") to \
             LOCKED and update GLSL_NUMERICS_HASH.\n"
        );
    }
    println!("transcriptions locked: {found:?}");
}


/// EVERY e4m3 byte, decoded, against `math.rs`.
///
/// The standing debt from `kernels/vk/common.glsl`'s placeholder, now due: the LUT the
/// fp8 GEMVs build is 256 entries of `e4m3f`, so this is not a sample — 256 values IS
/// the whole domain, and a GEMV oracle over plausible weights would reach almost none of
/// the interesting ones. Specifically it would miss NaN at (exp==15, mant==7), the
/// exp==0 subnormal ladder, and the sign-symmetric edges, which is exactly why the
/// placeholder named them.
///
/// This test USED to carry an argument that `exp2` was safe in the normal branch: its
/// argument is an integer in [-6, 8], and exp2 of an exact integer is an exact power of
/// two "in any conforming implementation", so Vulkan's 3-ULP allowance could not bite.
/// The argument was wrong in one way and the evidence for it was wrong in another.
///
/// Wrong argument: Vulkan's 3-ULP allowance for `exp2` has NO exemption for integer
/// arguments. "Every implementation I can think of is exact there" is a prediction about
/// implementations, not a property of the contract — and predictions of that shape have
/// a poor record in this port.
///
/// Wrong evidence, and this is the part worth remembering: THIS TEST COULD NEVER HAVE
/// DETECTED THE PROBLEM. It runs `transcribed_e4m3f`, which called RUST's `f32::exp2` —
/// exact. The shader calls GLSL's `exp2` — the thing in question. A literal
/// transcription mirrors STATEMENTS, and an accuracy contract is precisely what a
/// transcription cannot transcribe. So the lock would have kept the two in perfect
/// correspondence while the only property at issue differed between them.
///
/// The fix removes the question rather than answering it: both sides now build the power
/// of two by bit manipulation, which is exact by construction and has no accuracy
/// contract to argue about. Generalise it — for any `exp2`/`inversesqrt`-class builtin,
/// a transcription test is NOT evidence of agreement, and the honest options are to
/// eliminate the builtin or to compare against the real shader on the device.
#[test]
fn e4m3f_decodes_all_256_bytes_bit_exactly() {
    for b in 0u32..256 {
        let want = rivoli::math::e4m3_to_f32(b as u8);
        let got = transcribed_e4m3f(b);
        if want.is_nan() {
            assert!(got.is_nan(), "e4m3f({b:#04x}): math.rs NaN, transcription {got}");
            continue;
        }
        assert_eq!(
            want.to_bits(),
            got.to_bits(),
            "e4m3f({b:#04x}): math.rs {want} ({:#010x}) vs transcription {got} ({:#010x})",
            want.to_bits(),
            got.to_bits()
        );
    }
    // The three classes the placeholder called out, asserted by name so a future edit
    // cannot quietly stop covering them.
    assert!(rivoli::math::e4m3_to_f32(0x7f).is_nan(), "0x7f is the NaN encoding");
    assert_eq!(rivoli::math::e4m3_to_f32(0x01), 2f32.powi(-9), "smallest subnormal");
    assert_eq!(
        rivoli::math::e4m3_to_f32(0x80),
        -rivoli::math::e4m3_to_f32(0x00),
        "sign symmetry at zero"
    );
    println!("e4m3f: all 256 byte values bit-exact");
}
