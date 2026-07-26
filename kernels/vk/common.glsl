// GLSL twin of kernels/common.hpp — the helpers every compute shader shares.
// Kept deliberately line-comparable with the .hip original: the two backends must
// stay NUMERICALLY IDENTICAL, and a reader has to be able to diff them by eye.
//
// #included, not compiled — build.rs tracks it with its own rerun-if-changed, since
// a stale shared header shipping old SPIR-V is a bug this repo has already hit once
// (see the common.hpp staleness fix in git history).
#ifndef RIVOLI_COMMON_GLSL
#define RIVOLI_COMMON_GLSL

// Requested HERE, not left to each includer. `wave_sum`/`wave_max` are declared
// unconditionally, so a shader that includes this header needs the extensions whether
// or not it calls them — making it the header's job. Leaving it to the caller means
// every new kernel starts with the same compile error.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

// WAVE and ROWS_PER_BLOCK are injected by build.rs with -D, from the SAME constants
// it uses to generate the Rust side's copy. They are deliberately NOT #defined here:
// the launcher's grid arithmetic and the shader's row mapping must agree, and a
// "must match" comment across two languages is not agreement. gfx1151 is native
// wave32 and the pipeline pins it with requiredSubgroupSize + REQUIRE_FULL_SUBGROUPS,
// so unlike the HIP side the wave width is driver-enforced rather than assumed.
#if !defined(WAVE) || !defined(ROWS_PER_BLOCK)
#error "WAVE/ROWS_PER_BLOCK must come from build.rs (-DWAVE=.. -DROWS_PER_BLOCK=..)"
#endif

// Sum a subgroup's partials into LANE 0. subgroupShuffleDown, NOT subgroupAdd:
// subgroupAdd's summation order is implementation-defined and greedy decode must be
// reproducible. Same fixed halving ladder as common.hpp::wave_sum, so the two backends
// round identically at lane 0. (build.rs also rejects any module that declares the
// GroupNonUniformArithmetic capability, so this is enforced, not merely requested.)
//
// LANE 0 ONLY. This is where the GLSL and the HIP diverge: HIP's
// __shfl_down(v, o, 32) returns the CALLER'S OWN value when lane+o >= 32, so every
// lane ends holding a well-defined partial. SPIR-V's OpGroupNonUniformShuffleDown
// leaves the result UNDEFINED in that case, so lanes >= o hold garbage here. The
// induction still puts the true total in lane 0 — at offset o, lane j < o reads lane
// j+o, whose value came from offset 2o over lanes j+o and j+3o < 4o <= 32, always in
// range — but a kernel that reads wave_sum's result on any other lane is correct
// under HIP and silently wrong under Vulkan. Every caller must guard with lane == 0.
float wave_sum(float v) {
    for (uint o = WAVE / 2u; o > 0u; o >>= 1) v += subgroupShuffleDown(v, o);
    return v;
}

// The two buffer-reference types every kernel needs. Declared once here rather than
// six times across the shaders — buffers reach a shader as device addresses in push
// constants (docs/VULKAN.md), so an f32 in and an f32 out is the shape of nearly every
// launcher. Kernels needing other widths (packed u8/u16 read as words, i32 outputs)
// declare those locally, where the unpacking that justifies them lives.
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RoF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer RwF32 { float v[]; };

// Max across a subgroup, into LANE 0. Same shuffle ladder and the same lane-0-only
// caveat as wave_sum above. Unlike a sum this is EXACT — max does no rounding, so the
// order genuinely does not matter — but it still uses shuffles rather than
// subgroupMax, because build.rs rejects the GroupNonUniformArithmetic capability
// outright and a per-op exemption is not worth the hole in that guard.
float wave_max(float v) {
    for (uint o = WAVE / 2u; o > 0u; o >>= 1) v = max(v, subgroupShuffleDown(v, o));
    return v;
}

// EDITING f2bf16 OR f2e4m3? Update tests/glsl_numerics.rs to match. It holds literal
// Rust transcriptions of both and diffs them against math.rs over ~1.2M values on the
// CPU, so it catches a mistranscribed branch without needing the GPU — but only while
// the transcription still mirrors what is written here.

// f32 -> bf16, round-to-nearest-even. MUST stay bit-exact with math.rs::f32_to_bf16 on
// the finite domain; append_kv's roped keys are compared as BYTES against it.
//
// Non-finite keeps its top 16 bits verbatim, matching common.hpp — an RNE carry could
// turn a NaN into an Inf. Note this DIVERGES from math.rs for NaN, which goes through
// half::bf16::from_f32 and forces the quiet bit. The divergence predates the port and
// the rule is to mirror HIP, not math.rs (docs/VULKAN.md, "Numerics").
uint f2bf16(float x) {
    uint b = floatBitsToUint(x);
    if ((b & 0x7f800000u) == 0x7f800000u) return b >> 16; // inf/nan: verbatim
    return (b + (((b >> 16) & 1u) + 0x7fffu)) >> 16;
}

// f32 -> OCP e4m3, bit-for-bit with math.rs::f32_to_e4m3 AND fwd.hip::f2e4m3: RNE on
// the normal mantissa, round-half-away-from-zero on the subnormal, saturating to +-448,
// 0x7f for NaN. The latent cache's quantizer; append_kv compares its output as bytes.
//
// The subnormal branch is the one that bites: it rounds half AWAY from zero (matching
// Rust's `.round()`), NOT to even like the normal path, and m==8 promotes to the
// smallest normal rather than clamping.
uint f2e4m3(float x) {
    if (isnan(x)) return 0x7fu;
    uint sign = (floatBitsToUint(x) & 0x80000000u) != 0u ? 0x80u : 0u;
    float a = abs(x);
    if (a >= 448.0) return sign | 0x7eu;
    if (a < 0.0009765625) return sign; // < 2^-10 rounds to zero
    uint bits = floatBitsToUint(a);
    int e = int((bits >> 23) & 0xffu) - 127;
    if (e < -6) {
        uint m = uint(floor(a * 512.0 + 0.5));
        return m >= 8u ? (sign | 0x08u) : (sign | m);
    }
    uint mant = bits & 0x007fffffu;
    uint m3 = mant >> 20;
    uint rem = mant & 0x000fffffu;
    const uint half_ulp = 0x00080000u;
    if (rem > half_ulp || (rem == half_ulp && (m3 & 1u) != 0u)) m3 += 1u;
    int exp = e + 7;
    if (m3 == 8u) { m3 = 0u; exp += 1; }
    if (exp >= 15 && m3 >= 7u) return sign | 0x7eu;
    return sign | (uint(exp) << 3) | m3;
}

// log2 of a power-of-two fp8 scale-tile size. Mirrors common.hpp::blk_shift, which is
// `31 - __clz(block)`; GLSL's findMSB is the same value for a positive power of two.
// Every launcher taking a `block` REJECTS a non-power-of-two (arg guard 1003), so
// agreement OUTSIDE that domain is irrelevant and is not claimed — `findMSB` returns the
// highest set bit, which for a non-power-of-two is a floor rather than a log.
//
// Bit-identical to the divide it replaces: same index, same order.
int blk_shift(int block) { return findMSB(block); }

// fp8-e4m3 -> f32 (matches math.rs::e4m3_to_f32 and common.hpp::e4m3f). 1 sign / 4 exp
// (bias 7) / 3 mantissa; exp==0 is subnormal = sign*(m/8)*2^-6; (exp==15, mant==7) is
// NaN. Bit-exactness is a CONTRACT here, not a nicety: the fp8 GEMVs decode weights
// through a LUT built from this, so an error is a wrong weight rather than a wrong bit.
float e4m3f(uint b) {
    float sign = (b & 0x80u) != 0u ? -1.0 : 1.0;
    int exp = int((b >> 3) & 0x0fu);
    float mant = float(b & 0x07u);
    if (exp == 0) return sign * (mant * 0.125) * 0.015625; // 2^-6
    if (exp == 15 && mant == 7.0) return uintBitsToFloat(0x7fc00000u);
    // The power of two is BUILT FROM BITS, not computed. `exp2(float(exp - 7))` is the
    // obvious transliteration of HIP's `exp2f` and of math.rs's `powi`, and it is the
    // same trap that produced `inversesqrt`: GLSL does not require `exp2` to be exact,
    // even on an exact integer argument, while Rust's `powi` and HIP's `exp2f` are. A
    // conformant implementation may return 2^3 with several ULP of error, and the line
    // above this function calls bit-exactness a CONTRACT.
    //
    // `exp` is 1..15 here (0 and the NaN encoding returned already), so `exp - 7 + 127`
    // is 121..135 — always a normal exponent, so the shift is exact and total over the
    // whole domain, with no accuracy contract to rely on.
    return sign * (1.0 + mant * 0.125) * uintBitsToFloat(uint(exp - 7 + 127) << 23);
}

// Filling the e4m3 LUT is a MACRO, not a function, and that is a correctness
// requirement rather than a style choice.
//
// GLSL passes parameters by VALUE-RESULT — copy-in/copy-out — including arrays, and
// including `shared` ones. Written as `void e4m3_lut_build(inout float lut[256], uint
// tid)` it compiled to a per-invocation 1 KB Function-storage copy of the whole shared
// array: each of 256 threads copied all 256 entries in, wrote one, and copied all 256
// back, so 255 of every thread's entries were the uninitialised values it had loaded.
// Last writer won, the table was noise, and every fp8 weight decoded to noise
// (err = 8.6e37). Nothing flagged it: clean compile, spirv-val clean, every capability
// and arithmetic guard clean, and GPU-AV silent because the reads were in-bounds — the
// CONTENTS were garbage, not the addresses.
//
// A macro writes the caller's shared variable directly, which is the whole point.
// build.rs now rejects array-typed function parameters so this cannot come back as a
// function.
//
// Caller barriers before use; needs a >=256-thread workgroup, which every caller has.
#define E4M3_LUT_BUILD(lut, tid) { if ((tid) < 256u) (lut)[(tid)] = e4m3f(tid); }

// bf16 -> f32. Mirrors common.hpp::bf16f exactly: the 16 bits become the TOP half of an
// f32 and the low half is zero, which is the whole of the format's definition. Exact and
// total over every bit pattern — no rounding and no branch.
//
// IT DIVERGES FROM math.rs FOR SIGNALLING NaN, and that was found by testing rather than
// assumed. `half::bf16::to_f32` QUIETS a signalling NaN, so 0x7f81 decodes to 0x7fc1_0000
// through math.rs and to 0x7f81_0000 here. The f2bf16 note above records the same
// disagreement in the ENCODE direction and was believed to be the only one; it is not.
// The rule stands either way — mirror HIP, not math.rs, since the backends are what must
// agree — and tests/glsl_numerics.rs pins the 126 diverging patterns so a change in
// either implementation is announced.
//
// Ported now because `mla_latent_attend` is its first caller (the roped keys stay bf16).
float bf16f(uint b) { return uintBitsToFloat((b & 0xffffu) << 16); }

// f2bf16's DECODE partner is above. The remaining note applies to anything still absent:
// they are a BIT-EXACTNESS CONTRACT with src/math.rs, and unexercised code carrying a
// correctness contract is worse than absent code, because it reads as coverage. Unused
// GLSL is optimised out of every module, so porting them ahead of a caller ships
// exactly that false signal.
//
// Port each alongside the first kernel that needs it — and note that a GEMV oracle
// over plausible weight data does NOT cover them. It reaches neither the encodings
// nor the branches that break. So, specifically:
//
//   e4m3f / e4m3_lut_build: add a standalone shader test that decodes ALL 256 byte
//   values and compares elementwise against math::e4m3_to_f32. A 256-element dispatch
//   settles the contract permanently; a MoE oracle will miss NaN (exp==15, mant==7),
//   the exp==0 subnormal ladder, and the sign-symmetric edges.
//
//   f2bf16: test the RNE-with-non-finite-passthrough branch directly — a NaN whose
//   payload an RNE carry would turn into Inf (the bug common.hpp's comment warns
//   about), ±Inf, values that carry on rounding, and the subnormal boundary. The
//   decode direction (bf16f) is cheap enough to test broadly.

#endif // RIVOLI_COMMON_GLSL
