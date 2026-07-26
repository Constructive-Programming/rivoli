// GLSL twin of kernels/common.hpp — the helpers every compute shader shares.
// Kept deliberately line-comparable with the .hip original: the two backends must
// stay NUMERICALLY IDENTICAL, and a reader has to be able to diff them by eye.
//
// #included, not compiled — build.rs tracks it with its own rerun-if-changed, since
// a stale shared header shipping old SPIR-V is a bug this repo has already hit once
// (see the common.hpp staleness fix in git history).
#ifndef RIVOLI_COMMON_GLSL
#define RIVOLI_COMMON_GLSL

// ponytail: WAVE=32 hardcoded for gfx1151 (RDNA3.5, native wave32), and the pipeline
// pins it with requiredSubgroupSize=32 + REQUIRE_FULL_SUBGROUPS — so unlike the HIP
// side this is enforced by the driver, not merely assumed.
#define WAVE 32
#define ROWS_PER_BLOCK 8  // block = 256 threads = 8 subgroups → 8 output rows/block

// Sum a subgroup's partials into lane 0. subgroupShuffleDown, NOT subgroupAdd:
// subgroupAdd's summation order is implementation-defined, and greedy decode must be
// reproducible. This is the same fixed halving ladder as common.hpp::wave_sum, so the
// two backends round identically.
float wave_sum(float v) {
    for (uint o = WAVE / 2u; o > 0u; o >>= 1) v += subgroupShuffleDown(v, o);
    return v;
}

// bf16 → f32: bf16 is the high 16 bits of f32 (matches math.rs::bf16_to_f32).
float bf16f(uint b) {
    return uintBitsToFloat(b << 16);
}

// f32 → bf16, round-to-nearest-even (matches math.rs::f32_to_bf16). Non-finite keeps
// its top 16 bits verbatim (an RNE carry could turn a NaN into an Inf).
uint f2bf16(float x) {
    uint b = floatBitsToUint(x);
    if ((b & 0x7f800000u) == 0x7f800000u) return b >> 16; // inf/nan
    return (b + (((b >> 16) & 1u) + 0x7fffu)) >> 16;
}

// fp8-e4m3 → f32 (matches math.rs::e4m3_to_f32). 1 sign / 4 exp (bias 7) / 3 mant;
// exp==0 subnormal = sign·(m/8)·2^-6; (exp==15, mant==7) = NaN.
float e4m3f(uint b) {
    float sign = (b & 0x80u) != 0u ? -1.0 : 1.0;
    int exp = int((b >> 3) & 0x0fu);
    float mant = float(b & 0x07u);
    if (exp == 0) return sign * (mant * 0.125) * 0.015625; // 2^-6
    if (exp == 15 && mant == 7.0) return uintBitsToFloat(0x7fc00000u);
    return sign * (1.0 + mant * 0.125) * exp2(float(exp - 7));
}

#endif // RIVOLI_COMMON_GLSL
