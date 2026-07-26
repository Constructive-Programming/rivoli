// VQ-int3 decode — the shared dot behind every MoE expert projection.
// GLSL twin of kernels/common.hpp::dot_vq_wave.
//
// Included AFTER the caller declares nothing in particular (this header owns no shared
// state, unlike fp8.glsl's LUT convention) — it is separate only because all three moe
// shaders need it.
#ifndef RIVOLI_VQ_GLSL
#define RIVOLI_VQ_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// MUST match quant.rs and common.hpp. Not single-sourced through build.rs for the same
// reason the MLA constants are not — that needs build.rs to feed the ROCm arm too.
#define VQ_DIM 4
#define VQ_INDEX_BITS 12
#define VQ_GROUP 64
#define VQ_SUBS_PER_GROUP (VQ_GROUP / VQ_DIM)

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RoU32v { uint v[]; };

// One byte from a device address, read through a 32-bit word.
//
// SLOW AND OBVIOUSLY CORRECT, ON PURPOSE. The 12-bit index unpack below reads two ADJACENT
// bytes, and when the first sits at offset ≡ 3 (mod 4) they land in DIFFERENT words. A
// single-word load plus a 16-bit shift — the transliteration a reader would write — is
// correct for three offsets out of four and silently wrong for the fourth, producing a
// valid-looking codebook index rather than a fault. That is the likeliest silent gather
// bug in this tranche, so each byte gets its own word load and the compiler is left to
// merge them when it legally can.
uint vq_byte(uint64_t base, uint64_t off) {
    uint word = RoU32v(base + (off & ~3ul)).v[0];
    return (word >> ((uint(off) & 3u) * 8u)) & 0xffu;
}

// Wave-cooperative VQ-int3 dot for one output row, result on LANE 0 only.
//
// Lane `l` owns subvectors t ≡ l (mod WAVE), matching common.hpp exactly — the summation
// order is the result, so this is a contract and not a scheduling detail.
//
// THE FUSION IS WRITTEN OUT, NOT LEFT TO THE COMPILER. hipcc lowers the four-term
// subvector dot to ONE plain multiply followed by THREE fused multiply-adds, then a fourth
// fma for the group scale — read off the isolated ISA:
//
//     v_mul_f32     v11, v11, v17        ; term 1, plain
//     v_fma_mix_f32 v10, v10, v14, v11   ; term 2, fused
//     v_fma_mix_f32 v10, v12, v15, v10   ; term 3, fused
//     v_fma_mix_f32 v10, v13, v15, v10   ; term 4, fused
//     v_dual_fmac_f32 v7, v10, v11       ; acc += scale * dot, fused
//
// Four multiplies feeding three adds admit many legal contractions and they give different
// results, so the port spells hipcc's choice explicitly rather than hoping ACO agrees —
// the exactness strategy's step 3 (docs/VULKAN.md).
//
// `v_fma_mix_f32` takes its fp16 operand without a separate convert. That is NOT a
// divergence: every finite fp16 value is exactly representable in f32 (10 mantissa bits
// into 23; even the 2^-24 subnormals are f32 normals), so converting first introduces no
// rounding and `fma(x, float(h), acc)` is exactly what the mix instruction computes. The
// conversion was never the hazard; the fusion boundary was, and it is pinned above.
float dot_vq_wave(RoF32 x, uint64_t idxrow, uint64_t scalerow, uint64_t cb,
                  int i_dim, uint lane) {
    uint nsub = uint(i_dim) / VQ_DIM;
    float acc = 0.0;
    for (uint t = lane; t < nsub; t += WAVE) {
        uint bitpos = t * VQ_INDEX_BITS;
        uint64_t byte = uint64_t(bitpos >> 3);
        uint shift = bitpos & 7u;
        uint raw = vq_byte(idxrow, byte) | (vq_byte(idxrow, byte + 1ul) << 8);
        uint idx = (raw >> shift) & 0xfffu;

        // The subvector's four fp16 centroids: 8 bytes = two words, two halves each.
        uint64_t c = cb + uint64_t(idx) * uint64_t(VQ_DIM) * 2ul;
        vec2 c01 = unpackHalf2x16(RoU32v(c).v[0]);
        vec2 c23 = unpackHalf2x16(RoU32v(c + 4ul).v[0]);

        uint xi = t * VQ_DIM;
        float dot = x.v[xi] * c01.x;
        dot = fma(x.v[xi + 1u], c01.y, dot);
        dot = fma(x.v[xi + 2u], c23.x, dot);
        dot = fma(x.v[xi + 3u], c23.y, dot);

        // The group scale is bf16; one scale per VQ_SUBS_PER_GROUP subvectors.
        uint64_t soff = uint64_t(t / VQ_SUBS_PER_GROUP) * 2ul;
        uint sw = RoU32v(scalerow + (soff & ~3ul)).v[0];
        uint sh = (sw >> ((uint(soff) & 3u) * 8u)) & 0xffffu;
        acc = fma(bf16f(sh), dot, acc);
    }
    return wave_sum(acc);
}

#endif // RIVOLI_VQ_GLSL
