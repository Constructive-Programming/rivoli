// The fp8-e4m3 block-scaled MAC, shared by every kernel that decodes fp8 weights.
// GLSL twin of kernels/common.hpp::fp8_dot_strided.
//
// SEPARATE HEADER, NOT common.glsl, AND THE REASON IS A GUARD. This file's function
// reads a `shared float lut[256]` that the INCLUDER declares — see the LUT note below —
// so it must be #included after that declaration, which common.glsl (included first, for
// wave_sum and the buffer-reference types) cannot be.
//
// Putting the LUT itself in common.glsl instead was considered and rejected: it would
// place a Workgroup variable in every module that reads it, and build.rs's barrier rule
// declines to judge any module with shared storage. The exempt set would have grown to
// cover kernels that have no LDS of their own — coverage waived as a side effect of
// deduplication. (An UNUSED shared array is dead-code-eliminated under -O, so the effect
// is confined to actual readers; that is narrower than a first draft of this comment
// claimed, and it is still the wrong direction to push a guard.)
#ifndef RIVOLI_FP8_GLSL
#define RIVOLI_FP8_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// EVERY INCLUDER FILLS THE LUT WITH E4M3_LUT_BUILD, WHICH NEEDS >= 256 THREADS.
//
// The macro writes one entry per invocation (`if (tid < 256u) lut[tid] = e4m3f(tid)`), so
// a workgroup smaller than 256 leaves the upper entries as uninitialised Workgroup
// storage. Since the high bit of an e4m3 byte is its SIGN, the first thing that would
// decode from garbage is every NEGATIVE weight — the err = 8.6e37 failure that already
// shipped once here, re-entering through a different door. Nothing downstream could see
// it: spirv-val is clean, and GPU-AV is silent because the shared reads are in bounds.
//
// Checked HERE, once, rather than in each of the three kernels that build a LUT. Every
// includer declares `local_size_x` as ROWS_PER_BLOCK*WAVE, so this is the same quantity,
// and lowering ROWS_PER_BLOCK is a plausible tuning change that would otherwise be silent.
#if (ROWS_PER_BLOCK * WAVE) < 256
#error "fp8.glsl: E4M3_LUT_BUILD fills one entry per thread and needs a >= 256-thread workgroup; ROWS_PER_BLOCK*WAVE is smaller"
#endif

// Packed fp8 weights are read as 32-bit WORDS — VK_KHR_8bit_storage is deliberately not
// required (docs/VULKAN.md) — so every caller's row base must be 4-aligned.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RoU32 { uint v[]; };

// THE LUT IS AN IMPLICIT PARAMETER, and that is forced rather than chosen.
//
// HIP passes `const float* lut` and C decays the array to a pointer. GLSL has no pointer
// decay: an array parameter is copy-in/copy-out, so `float lut[256]` would hand every
// invocation a private 1 KB copy of the shared table, and the writes would be lost. That
// exact bug shipped once and produced err = 8.6e37 with every other check clean; build.rs
// now rejects a whole-array OpLoad outright (rule 11).
//
// So the contract is a NAMING one: every includer declares `shared float lut[256]` before
// including this file, fills it with E4M3_LUT_BUILD, and barriers before the first call.
// A caller that forgets the declaration gets a compile error naming `lut`, which is the
// failure mode we want — loud, and at the right line.
//
// Σ over the columns this thread owns, with NO cross-thread reduction; the caller
// reduces. `x` and `scalerow` arrive already offset to their row, exactly as the HIP
// version's pointers do, and `wrow` is the row's absolute device address. Mirrors the HIP
// statement for statement — including the order of the four products inside the
// parentheses, which is what makes it bit-identical.
float fp8_dot_strided(RoF32 x, uint64_t wrow, RoF32 scalerow,
                      int i_dim, int block, uint start, uint stride) {
    float acc = 0.0;
    uint n4 = uint(i_dim) >> 2;
    int bsh = blk_shift(block);
    for (uint j = start; j < n4; j += stride) {
        uint p = RoU32(wrow + uint64_t(j) * 4ul).v[0];
        uint i0 = j << 2;
        float s = scalerow.v[i0 >> bsh];
        acc += s * (x.v[i0]      * lut[p & 0xffu]
                  + x.v[i0 + 1u] * lut[(p >> 8) & 0xffu]
                  + x.v[i0 + 2u] * lut[(p >> 16) & 0xffu]
                  + x.v[i0 + 3u] * lut[(p >> 24) & 0xffu]);
    }
    // Scalar tail, for an i_dim that is not a multiple of 4. Unreachable while the
    // launchers enforce that, and kept because the HIP source has it and the two must
    // stay diffable by eye.
    for (uint i = (n4 << 2) + start; i < uint(i_dim); i += stride) {
        uint word = RoU32(wrow + (uint64_t(i) & ~3ul)).v[0];
        uint byte = (word >> ((i & 3u) * 8u)) & 0xffu;
        acc += x.v[i] * lut[byte] * scalerow.v[i >> bsh];
    }
    return acc;
}

// One WAVE reduces one row, mirroring common.hpp::dot_fp8_wave. Result on LANE 0 only —
// wave_sum's ladder leaves every other lane holding garbage under SPIR-V, unlike HIP.
float dot_fp8_wave(RoF32 x, uint64_t wrow, RoF32 scalerow,
                   int i_dim, int block, uint lane) {
    return wave_sum(fp8_dot_strided(x, wrow, scalerow, i_dim, block, lane, WAVE));
}

#endif // RIVOLI_FP8_GLSL
