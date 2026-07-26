// GLSL twin of kernels/common.hpp — the helpers every compute shader shares.
// Kept deliberately line-comparable with the .hip original: the two backends must
// stay NUMERICALLY IDENTICAL, and a reader has to be able to diff them by eye.
//
// #included, not compiled — build.rs tracks it with its own rerun-if-changed, since
// a stale shared header shipping old SPIR-V is a bug this repo has already hit once
// (see the common.hpp staleness fix in git history).
#ifndef RIVOLI_COMMON_GLSL
#define RIVOLI_COMMON_GLSL

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

// bf16f / f2bf16 / e4m3f are NOT here yet, on purpose. They are not conveniences —
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
