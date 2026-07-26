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

// bf16f / f2bf16 / e4m3f are NOT here yet, on purpose. Unused GLSL functions are
// optimised out of every module, so porting them ahead of a caller ships code that
// looks verified and is not — and f2bf16's non-finite branch is a hand-rewrite of
// isfinite() into a bit test. Port each alongside the first kernel that needs it, so
// that kernel's oracle covers it.

#endif // RIVOLI_COMMON_GLSL
