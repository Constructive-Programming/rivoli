// Scalar NUMERIC CODECS: one element in, one element out. bf16, fp8-e4m3, the OCP MX pair
// e2m1/e8m0, and the two activation quantizers built on them. Split out of common.hpp
// 2026-08-15 under the per-file line ceiling; every body and its measurements moved
// verbatim.
//
// The bf16/e4m3 pair MUST stay bit-exact with math.rs (the CPU oracle the kernel tests
// compare against), so one definition is a correctness property — that claim is the reason
// this header exists as a header at all, and it is why a kernel that needs a decode
// includes common.hpp instead of writing three lines of bit surgery. `rbf16` below records
// what it cost the last time: THREE independent per-TU copies, and the only tool that ever
// caught them was the compiler.
//
// What is here and what is NOT: this header owns the ELEMENT codec and the constants that
// are part of a codec's own definition (the quantizer's block size and its ceiling —
// ACT_QUANT_BLOCK/FP8_MAX, FP4_QUANT_BLOCK/FP4_MAX). The WEIGHT-LAYOUT constants
// (I4_GROUP, F4_GROUP, VQ_*) stay in common.hpp beside the dot products that index rows
// with them: those describe a packed row's stride, not how one number is spelled.
#pragma once

#include <hip/hip_runtime.h>

// bf16 → f32: bf16 is the high 16 bits of f32 (matches math.rs::bf16_to_f32).
__device__ __forceinline__ float bf16f(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

// f32 → bf16, round-to-nearest-even (matches math.rs::f32_to_bf16). Non-finite
// keeps its top 16 bits verbatim (an RNE carry could turn a NaN into an Inf).
__device__ __forceinline__ unsigned short f2bf16(float x) {
    unsigned int b;
    __builtin_memcpy(&b, &x, sizeof(b));
    if (!isfinite(x)) return (unsigned short)(b >> 16);
    unsigned int r = ((b >> 16) & 1u) + 0x7fffu;
    return (unsigned short)((b + r) >> 16);
}

// bf16 round-trip in f32. V4 activations stay f32 on device and hold bf16-representable
// values, which is what `Oracle::round_bf16` does to an `f32` Vec — the reference stores
// bf16 at every point the kernels mark, and matching WHERE it rounds is what makes the
// goldens comparable at all.
//
// Dropping it leaves values that are fluent and slightly TOO PRECISE, which no magnitude
// check sees — `v4oracle::Defect::NoBf16Rounding`.
//
// `kernels/` is not watched by `build.rs`'s jscpd gate, so nothing mechanical finds a
// duplicated device helper. This one had THREE independent per-TU copies before it was
// hoisted, and the only tool that ever caught the duplication was the compiler — on a clean
// merge of two agents' identical hunks into this header, which does not overlap and so does
// not conflict. `moe.hip` spelled its copy `bf16r`; that one carries the magnitude the
// rounding is worth.
//
// No `#pragma clang fp contract(off)` here, and none needed: this is bit surgery, with no
// multiply for an FMA to absorb. That is NOT true of `f2e4m3_rne` below, which does carry
// one — see the note there for what an ISA diff caught, and common.hpp's "V4 shared device
// helpers" note for the same argument across the helpers it covers.
__device__ __forceinline__ float rbf16(float x) { return bf16f(f2bf16(x)); }
// NAMING RULE for `bf16` in kernel names, stated here because this helper is what the rule
// is about: a trailing `_bf16` names the STORE (`gemv_fp8_bf16`, `swiglu_clamped_bf16` —
// both end in `rbf16`); `bf16` elsewhere in a name is the INPUT dtype (`gemm_bf16` weights,
// `embed_bf16_row_bcast` source table). If a new kernel would break that split, rename
// rather than adding a third sense.

// fp8-e4m3 → f32 (matches math.rs::e4m3_to_f32). 1 sign / 4 exp (bias 7) / 3 mant;
// exp==0 subnormal = sign·(m/8)·2^-6; (exp==15, mant==7) = NaN.
__device__ __forceinline__ float e4m3f(unsigned char b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    int exp = (b >> 3) & 0x0f;
    float mant = (float)(b & 0x07);
    if (exp == 0) return sign * (mant * 0.125f) * 0.015625f; // 2^-6
    if (exp == 15 && mant == 7.0f) return __int_as_float(0x7fc00000);
    return sign * (1.0f + mant * 0.125f) * exp2f((float)(exp - 7));
}

// Fill a 256-float LDS table with e4m3f(byte) so the hot GEMV decodes fp8 by an
// LDS read instead of the branchy exp2f path (decode was compute-bound, not load-
// width-bound — see the failed load-widening experiment). Bit-exact with e4m3f by
// construction. Caller passes threadIdx.x and __syncthreads() before use; needs a
// >=256-thread block (both callers launch ROWS_PER_BLOCK*WAVE = 256).
__device__ __forceinline__ void e4m3_lut_build(float* lut, int tid) {
    if (tid < 256) lut[tid] = e4m3f((unsigned char)tid);
}

// e2m1 nibble → f32. 1 sign / 2 exp (bias 1) / 1 mantissa; no infinities, no NaN, so a
// bare arithmetic decode is TOTAL — every one of the 16 codes is a finite value.
//
// A REGISTER-immediate table, not memory and not a ternary. The 8 magnitudes are
// {0, .5, 1, 1.5, 2, 3, 4, 6}; doubled they are the integers {0,1,2,3,4,6,8,12}, which fit
// a nibble each, so the whole table is the one immediate 0xC8643210 and a lookup is
// shift-and-mask — no LDS, no scratch, no divergent index. The previous form here was the
// subnormal-aware ternary `e ? (1+m/2)·2^(e−1) : m/2`, and the M3a ISA read
// (docs/investigations/v4-decode-decomposition.md) measured what it compiles to: an
// exec-mask branch region PER NIBBLE, ~88 of the fp4 dot loop's 195 instructions per 128
// weight bytes, on a loop that is instruction-issue-bound. This form compiles branchless
// (v_bfe / v_lshrrev / v_cvt_f32 / v_mul; read the .s, do not trust the source).
//
// Bit-exact with the ternary on all 16 codes, including the two zeros: `half` is an exact
// small integer, `0.5f * (float)half` is exact (power-of-two scaling of an exact value),
// and the sign is OR'd into the payload bits, so code 8 decodes to -0.0f — the same
// negative zero the ternary's `-mag` produced and `v4oracle`'s F4_LUT holds.
// `tests/f4_kernel.rs::the_branchless_decodes_match_the_oracle_bitwise` sweeps all 16
// against `v4oracle::numerics::e2m1_decode` at the bit level.
__device__ __forceinline__ float e2m1f(unsigned int nib) {
    unsigned int half = (0xC8643210u >> ((nib & 7u) << 2)) & 0xFu;
    float mag = 0.5f * (float)half;
    return __uint_as_float(__float_as_uint(mag) | ((nib & 8u) << 28));
}

// e8m0 scale byte → f32, `2^(b − 127)` (matches v4oracle::numerics::e8m0_decode).
//
// Both endpoints are spelled out rather than left to the exponent-field shift. 0xff is the
// format's NaN, which is the right DECODE — reading it as `2^128` would be worse — but it
// does not poison anything downstream: `moe_fixed`'s saturating clamp launders a NaN into a
// finite ±2^14 (`fminf`/`fmaxf` return the non-NaN operand). So a 0xff scale byte must be
// REJECTED AT LOAD, in S3, alongside the `tid2eid` range check that `parse_tid2eid`
// performs host-side. (That obligation used to be stated at `moe.hip::moe_gate_v4` too;
// the device router was DELETED 2026-08-09 — `f4gpu.rs::route_row` carries why — so this
// is now the only place in `kernels/` that names it, which is the reason to keep it here
// rather than fold it into the deleted kernel's note.)
// b == 0 is 2^-127, which is BELOW f32's smallest normal, so `b << 23` would silently hand
// back +0 — a whole 32-weight group zeroed with no error anywhere.
//
// These bytes come off the CHECKPOINT (`<proj>.scale`, copied verbatim by the `.f4`
// repack), not from `fast_round_scale`, so "our quantizer cannot emit them" is not an
// argument that applies here: the FORMAT permits both, and what the converter happened to
// produce is not a property this decoder gets to assume.
//
// Branchless since 2026-08-08 (the M3a kernel-rate work): the two-`if` form compiled to an
// exec-mask branch region inside the fp4 dot loop's every iteration. The selects below are
// the same three cases — `umax` against the b == 0 subnormal is exact because every other
// `b<<23` is strictly larger, and the 0xff arm ORs the quiet bit onto what is then
// 0x7f800000, giving the identical 0x7fc00000 NaN. All 256 bytes are swept bitwise against
// `v4oracle::numerics::e8m0_decode` by
// `tests/f4_kernel.rs::the_branchless_decodes_match_the_oracle_bitwise`.
__device__ __forceinline__ float e8m0f(unsigned char b) {
    unsigned int t = (unsigned int)b << 23;
    unsigned int bits = t > (1u << 22) ? t : (1u << 22);
    bits |= (b == 0xff) ? (1u << 22) : 0u;
    return __uint_as_float(bits);
}

// ── fp8 activation quantization (`kernel.py::act_quant`, inplace=True) ──────────
//
// Every quantized `Linear` in V4 quantizes its ACTIVATION to fp8 before the GEMM — fp4
// weights included (`model.py::linear` line 120). So the numbers a `dot_f4` consumes are
// not the layer's activations, they are `e4m3(x/s)·s` at block 128 with a power-of-two `s`.
// Skipping it is silent: the magnitudes are within 2^-3 relative of the right ones.
//
// The shipped config sets `scale_fmt: "ue8m0"` (via `scale_dtype: "fp8"`, ModelArgs's
// default), so the scale is ALWAYS rounded UP to a power of two. That is not a knob here —
// a runtime `round_scale` would be a second code path with no caller.
#define ACT_QUANT_BLOCK 128
#define FP8_MAX 448.0f
#define FP8_MAX_INV (1.0f / FP8_MAX)
// `T.max(amax_local[i], 1e-4)` — keeps an all-zero block from dividing by zero.
#define ACT_QUANT_AMAX_FLOOR 1e-4f

// `fast_log2_ceil` / `fast_pow2` / `fast_round_scale` from kernel.py:22-38, by the same
// IEEE-754 bit surgery. Transcribed, not reimplemented: `ceilf(log2f(x))` agrees with this
// everywhere except that it rounds the log FIRST, so a value one ulp below a power of two
// lands on a different binade — one factor-of-two error per group, invisible in aggregate.
__device__ __forceinline__ float fast_round_scale(float amax, float max_inv) {
    // Paired with `f2e4m3_rne`'s pragma below — see the argument there. Inert today (the
    // product feeds a bitcast, not an add, so there is nothing to fuse); present so the
    // two halves of `act_quant` carry the same property rather than one of them.
#pragma clang fp contract(off)
    unsigned int b;
    float y = amax * max_inv;
    __builtin_memcpy(&b, &y, sizeof(b));
    int e = (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) ? 1 : 0);
    return __uint_as_float((unsigned int)(e + 127) << 23);
}

// f32 → e4m3, round-to-nearest-EVEN, saturating at ±448.
//
// NOT `kernels/fwd.hip`'s `f2e4m3` (which rounds half-away-from-zero below 2^-6 — rivoli's
// own rule, mirroring `math.rs::f32_to_e4m3`): V4 was trained against CUDA's
// `cvt.rn.satfinite.e4m3x2.f32`, which is RNE all the way down, and that is what
// `v4oracle::numerics::e4m3_encode` models. The two differ on exact halfway subnormals and
// NOWHERE else — established BY INSPECTION on 2026-08-05, and by nothing that would go red
// if it stopped being true (`act_quant_f8_is_bit_identical_to_the_oracle` gates the
// subnormal ties against the ORACLE, not against `fwd.hip`). Including the two places the
// two look different and are not: the early return here is at the RNE midpoint 464 where
// `fwd.hip`'s is at 448, but
// `[448, 464)` falls through to code 0x7e either way; and `a <= 2^-10` versus `a < 2^-10`
// IS one of the subnormal ties (k = 0.5), not a separate case.
//
// One ulp on a tie is not much, and it is still worth its own function: this is the only
// encoder that will ever be compared against the oracle, and a known difference inside a
// result whose whole job is to expose unknown ones is the thing this port cannot afford.
__device__ __forceinline__ unsigned char f2e4m3_rne(float x) {
    // **Load-bearing, and MEASURED IN THE ISA rather than argued — 2026-08-05.** This
    // function had a twin inside `mla.hip`, below that file's own file-scope
    // `#pragma clang fp contract(off)`. S3 requirement 11 deleted the twin and pointed
    // `act_quant_f8_prefix` here; `common.hpp` is included ABOVE that pragma and clang attaches FP
    // options per expression at parse time, so inlining into a `contract(off)` caller does
    // NOT restore the property — the subnormal branch's `scaled - m` (fed by `a * 512.0f`)
    // began fusing. Counted inside `act_quant_f8_prefix` at `--offload-arch=gfx1151 -O3`, and the
    // delta is **one `v_fma_f32`**: 4 at 78796eb, 5 with this pragma removed, 4 with it
    // (6 / 7 / 6 counting `v_fmac_f32` too). A third tier of 7 / 8 / 7 is
    // reachable by also counting the one `v_div_fmas_f32` — DON'T: that belongs to the f32
    // divide expansion (`v_div_scale` x2, `v_div_fmas`, `v_div_fixup`, all present here),
    // which `mla.hip`'s pragma note already excludes as contraction evidence. It shows up
    // in a naive grep only because "di*v_fma*s" contains the substring. The check was run in that order precisely so it was seen to go red
    // before being trusted green.
    //
    // **Count `v_fma_f32`/`v_fmac_f32`, and compare a DELTA rather than an absolute.** A
    // `v_fma|v_mac|v_mad` grep reads 15 / 16 / 15 here, because eight `v_mad_u64_u32` are
    // ADDRESS arithmetic that contraction neither does nor should touch — enough to make a
    // one-instruction regression look like noise. Found on `index_score_blocks`, where the
    // same pragma moved 1 `v_fmac_f32` to 0 while a naive count read 10 → 9.
    //
    // Values do not move either way: reaching this branch bounds `a` to (2^-10, 2^-6),
    // where `a` is normal and `a * 512.0f` is exact, so the FMA elides two roundings that
    // were already no-ops. The pragma is here for the CLAIM, not the arithmetic —
    // `mla.hip`'s "VERIFIED IN THE ISA" note and the `ULP_BUDGET = 1` it justifies in
    // `tests/f4_attn.rs` both rest on contraction being absent from the V4 path, and a
    // silent 8th instruction makes those two false while nothing goes red.
#pragma clang fp contract(off)
    if (isnan(x)) return 0x7f;
    unsigned char sign = signbit(x) ? 0x80 : 0x00;
    float a = fabsf(x);
    // 448 is the largest finite magnitude and 464 the midpoint to the absent next code, so
    // RNE sends everything below it to 448 and the format saturates above.
    if (a >= 464.0f) return sign | 0x7e;
    // Below half the subnormal quantum (2^-10) everything rounds to zero; EXACTLY 2^-10 is
    // a tie between 0 (mantissa 0, even) and 2^-9 (mantissa 1, odd), so it goes to zero too
    // — hence `<=`, not `<`.
    if (a <= 0.0009765625f) return sign;
    unsigned int b;
    __builtin_memcpy(&b, &a, sizeof(b));
    int e = (int)((b >> 23) & 0xffu) - 127;
    if (e < -6) {
        // Subnormal: value = m·2^-9, m in 1..=7 (m == 8 promotes to the smallest normal,
        // code 0x08). floor + explicit tie test rather than `rintf`, so the rule is on the
        // page instead of in the hardware's current rounding mode.
        //
        // `a * 512.0f` feeding `scaled - m` is the fusable pair the function's pragma
        // exists for.
        float scaled = a * 512.0f;
        float m = floorf(scaled), rem = scaled - m;
        if (rem > 0.5f || (rem == 0.5f && ((unsigned int)m & 1u))) m += 1.0f;
        unsigned int mi = (unsigned int)m;
        return mi >= 8u ? (unsigned char)(sign | 0x08) : (unsigned char)(sign | mi);
    }
    // Normal: keep 3 mantissa bits, RNE on the 20 dropped ones.
    unsigned int mant = b & 0x7fffffu, m3 = mant >> 20, rem = mant & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u))) m3 += 1u;
    int exp = e + 7;
    if (m3 == 8u) { m3 = 0u; exp += 1; }
    // Unreachable given the `a >= 464` return above (which caps m3 at 6 when exp is 15);
    // kept because it is the format's own bound and costs one compare.
    if (exp >= 15 && m3 >= 7u) return sign | 0x7e;
    return (unsigned char)(sign | (unsigned char)(exp << 3) | (unsigned char)m3);
}

// One element of a block-128 group, fused quantize-then-dequantize: the value a V4 GEMM
// actually sees. `e4m3f` is reused rather than rewritten because DECODE is unambiguous —
// only the ENCODE rule differs between the two engines. What pins the pair is the round
// trip, in `tests/f4_kernel.rs::act_quant_f8_is_bit_identical_to_the_oracle`, over all 254
// finite codes. That test's own doc states what the round trip can and cannot prove.
__device__ __forceinline__ float act_quant_roundtrip(float x, float s) {
    // Written with comparisons, NOT `fminf`/`fmaxf`, so a NaN PROPAGATES. Those two return
    // the non-NaN operand, which would sanitize a NaN activation into -448 — a large finite
    // value where the oracle's `f32::clamp` keeps the NaN and `e4m3_encode` maps it to 0x7f.
    // Unreachable from any fixture (no NaN inputs) and a NaN activation is already fatal
    // upstream, but `act_quant_f8_is_bit_identical_to_the_oracle` claims BIT identity, and a
    // known place where that does not hold is worth one line to remove rather than to note.
    float q = x / s;
    q = q < -FP8_MAX ? -FP8_MAX : (q > FP8_MAX ? FP8_MAX : q);
    return e4m3f(f2e4m3_rne(q)) * s;
}

// ── fp4 ACTIVATION quantization (`kernel.py::fp4_act_quant`, inplace=True) ──────
//
// The indexer's simulator, not the expert weights' codec. `dot_f4_wave_r` in common.hpp
// DECODES packed e2m1 nibbles that came off the checkpoint; this pair ENCODES an activation
// and decodes it straight back, which is how `Indexer.forward` simulates fp4 in a bf16
// tensor (model.py:376 and :422 — both `q` and the indexer's compressed kv go through it).
//
// Three constants differ from the fp8 block above and every one is silent-wrong if it
// drifts: the block is **32**, not 128; the ceiling is **6**, not 448; and there is no
// `round_scale` switch — `fp4_quant_kernel` has none, so the scale is ALWAYS rounded up to
// a power of two. Mirrors `v4oracle::numerics::fp4_act_quant_inplace`.
#define FP4_QUANT_BLOCK 32
#define FP4_MAX 6.0f
#define FP4_MAX_INV (1.0f / FP4_MAX)
// `6 * 2^-126` — the fp4 kernel's amax floor, where `act_quant`'s is 1e-4. 2^-126 is f32's
// smallest NORMAL (`f32::from_bits(1 << 23)` in the oracle), and `6 * 2^-126` is exactly
// `1.5 * 2^-124`, so the product is exact rather than a decimal approximation of one.
#define FP4_QUANT_AMAX_FLOOR (FP4_MAX * 1.17549435082228750797e-38f)

// f32 → e2m1 nibble, round-to-nearest-EVEN, saturating at ±6.
//
// COUNTS midpoints passed rather than searching for the nearest magnitude, and that is not
// a micro-optimization — it is the correctness argument. The eight magnitudes have seven
// midpoints, so the number passed IS the code, saturation falls out (a huge input passes
// all seven and lands on 7), and the tie rule is one term. A nearest-neighbour search does
// NOT saturate: `1e9 - 6.0f == 1e9` makes every candidate equidistant, the tie rule keeps
// the first, and the encoder returns ZERO. That was a real bug in the oracle, caught by
// `tests/v4_oracle.rs::e2m1_encode_is_nearest_ties_to_even`, and this is a transcription of
// the fixed version — `v4oracle::numerics::e2m1_encode`.
//
// Ties go to the even mantissa bit, which on an ascending table means "up at an odd-indexed
// midpoint, down at an even-indexed one": 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0,
// 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0.
//
// NaN gives code 0 with NaN's sign bit: every comparison against a NaN is false, so nothing
// is counted. The oracle does the same thing for the same reason. Unreachable through
// `fp4_quant_roundtrip`, whose clamp precedes it, and stated so the two agree by argument
// and not by luck.
__device__ __forceinline__ unsigned int f2e2m1(float x) {
    const float mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    float a = fabsf(x);
    unsigned int code = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i) code += (a > mid[i] || (a == mid[i] && (i & 1))) ? 1u : 0u;
    return (signbit(x) ? 8u : 0u) | code;
}

// One element of a block-32 group, fused quantize-then-dequantize: the value the indexer's
// einsum actually sees. `e2m1f` is reused rather than rewritten because DECODE is
// unambiguous — only the ENCODE rule is a choice.
//
// Comparisons, NOT `fminf`/`fmaxf`, for the reason `act_quant_roundtrip` states: those
// return the non-NaN operand and would launder a NaN into -6, where the oracle's
// `f32::clamp` propagates it.
__device__ __forceinline__ float fp4_quant_roundtrip(float x, float s) {
    float q = x / s;
    q = q < -FP4_MAX ? -FP4_MAX : (q > FP4_MAX ? FP4_MAX : q);
    return e2m1f(f2e2m1(q)) * s;
}
