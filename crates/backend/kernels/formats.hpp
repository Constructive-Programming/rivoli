// Scalar NUMERIC CODECS: one element in, one element out. bf16, fp8-e4m3, and the OCP MX
// e2m1 pair. Split out of common.hpp 2026-08-15 under the per-file line ceiling; every
// body and its measurements moved verbatim.
//
// The bf16/e4m3 pair MUST stay bit-exact with math.rs (the CPU oracle the kernel tests
// compare against), so one definition is a correctness property — that claim is the reason
// this header exists as a header at all, and it is why a kernel that needs a decode
// includes common.hpp instead of writing three lines of bit surgery. `rbf16` below records
// what it cost the last time: THREE independent per-TU copies, and the only tool that ever
// caught them was the compiler.
//
// What is here and what is NOT. This header owns the ELEMENT codec and nothing else.
// [NARROWED 2026-08-15, same day as the split above.] Three neighbours left, each to the
// one place that consumes it, and the cut in every case is "does this describe how ONE
// number is spelled, or what a BLOCK/ROW of them does":
//
//   * `e4m3_lut_build` → `dot_fp8.hpp`. It fills the 256-float LDS table `Fp8Row::lut`
//     points at; its only callers are the fp8 GEMVs that then hand that table to the dot.
//   * `e8m0f` → `dot_f4.hpp`. The e8m0 GROUP-SCALE decoder, whose only two callers are
//     that file's dword pass and tail. Its e2m1 partner stays here, because `f2e2m1` and
//     the indexer's fp4 round trip need it too — the pair's two halves have genuinely
//     different scopes, which is why splitting them costs nothing.
//   * `fast_round_scale` / `act_quant_roundtrip` / `fp4_quant_roundtrip` and the four
//     block constants → `actquant.hpp`. Those are BLOCK operations composed of the codecs
//     below, not codecs.
//
// The WEIGHT-LAYOUT constants (I4_GROUP, F4_GROUP, VQ_*) live with their own dot header,
// beside the loop that indexes a packed row with them.
//
// **Argument budget.** CodeScene's Primitive Obsession rule fires on a file with >= 7
// functions once the count of primitive-typed ARGUMENTS across them reaches 11 (measured
// against cs 1.0.36, 2026-08-15). This file is 10 functions and **9** such arguments,
// because a scalar codec's one argument IS a scalar and wrapping it would bury the
// math.rs/oracle correspondence the file exists for. One slot of headroom: a new codec
// that takes ONE primitive fits, a second one does not — at which point split the file
// rather than wrapping arguments.
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
// multiply for an FMA to absorb. That is NOT true of `e4m3_subnormal_mantissa` below, which
// does carry one — see the note there for what an ISA diff caught, and common.hpp's "V4
// shared device helpers" note for the same argument across the helpers it covers.
// [CORRECTED 2026-08-15: this named `f2e4m3_rne`, which held the pragma until it was split
// into three; the pragma now sits on the one branch whose `a * 512.0f` feeds a subtract.]
__device__ __forceinline__ float rbf16(float x) { return bf16f(f2bf16(x)); }
// NAMING RULE for `bf16` in kernel names, stated here because this helper is what the rule
// is about: a trailing `_bf16` names the STORE (`gemv_fp8_bf16`, `swiglu_clamped_bf16` —
// both end in `rbf16`); `bf16` elsewhere in a name is the INPUT dtype (`gemm_bf16` weights,
// `embed_bf16_row_bcast` source table). If a new kernel would break that split, rename
// rather than adding a third sense.
//
// **And what a trailing `_f32` means, because the split above does not cover it.** Settled
// here 2026-08-12 (ported with M9 from `k3:kernels/common.hpp`) rather than in one kernel's
// comment, on this block's own instruction to resolve a new sense at the rule: a trailing
// `_f32` names the STORE (`activation.hip::situ_glu_f32`,
// `recurrent.hip::gated_delta_recurrent_f32`) **unless the kernel is a member of a
// `gemv_<weight-format>` family**, where the suffix has always named the weight dtype
// instead (`gemv_f32` against `gemv_fp8`/`_i4`/`_i8`/`_vq`, whose weights really are
// `float`). That exception is the whole of the ambiguity; naming it is cheaper than
// renaming either side, and leaving it unnamed is how a third sense arrives.

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
//
// Split into the three pieces below 2026-08-15 — SHAPE ONLY, the arithmetic and its
// association are untouched. What moved and why is recorded at each piece.

// The round-to-nearest-EVEN tie rule, stated ONCE for both of `f2e4m3_rne`'s mantissa
// paths: `rem`/`half` are floats in the subnormal path and integers in the normal one,
// which is the whole reason this is a template and not two copies of one line. Strictly
// past the midpoint rounds up; strictly below rounds down; EXACTLY on it the tie goes to
// the even mantissa, so the low bit of what is KEPT decides.
//
// Bit-identical to the `rem > half || (rem == half && (keep & 1u))` pair it replaces,
// NaN included: every comparison against a NaN is false, so that form answered "no", and
// here `rem != half` is true and `rem > half` is then false. Neither caller can present a
// NaN (`f2e4m3_rne` returns on `isnan` before either path, and the normal path's operands
// are integers), and that is stated rather than relied on because the two forms agree
// there anyway.
template <typename T>
__device__ __forceinline__ bool rne_rounds_up(T rem, T half, unsigned int keep) {
    if (rem != half) return rem > half;
    return (keep & 1u) != 0u;
}

// e4m3's SUBNORMAL mantissa for `a` in (2^-10, 2^-6): value = m·2^-9, m in 1..=8, where
// m == 8 promotes to the smallest normal (code 0x08) and the caller applies that. floor +
// an explicit tie test rather than `rintf`, so the rule is on the page instead of in the
// hardware's current rounding mode.
//
// **The pragma is load-bearing, and MEASURED IN THE ISA rather than argued — 2026-08-05.**
// `f2e4m3_rne` had a twin inside `mla.hip`, below that file's own file-scope
// `#pragma clang fp contract(off)`. S3 requirement 11 deleted the twin and pointed
// `act_quant_f8_prefix` here; `common.hpp` is included ABOVE that pragma and clang attaches FP
// options per expression at parse time, so inlining into a `contract(off)` caller does
// NOT restore the property — this branch's `scaled - m` (fed by `a * 512.0f`)
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
//
// **The pragma travelled DOWN here with the fusable pair on 2026-08-15, and that direction
// is the safe one.** It is the same per-expression-at-parse-time rule the paragraph above
// turns on, read the other way: options attach to THIS function's expressions where they
// are parsed, so inlining into `f2e4m3_rne` — which no longer carries the pragma — cannot
// take them off again. The reverse (relying on a caller's pragma) is what was measured to
// fail.
__device__ __forceinline__ unsigned int e4m3_subnormal_mantissa(float a) {
#pragma clang fp contract(off)
    // `a * 512.0f` feeding `scaled - m` is the fusable pair the pragma exists for.
    float scaled = a * 512.0f;
    float m = floorf(scaled), rem = scaled - m;
    if (rne_rounds_up(rem, 0.5f, (unsigned int)m)) m += 1.0f;
    return (unsigned int)m;
}

// A finite magnitude taken apart: its raw f32 bits, its unbiased exponent, and the sign
// bit the caller already extracted. One decomposed float, which is why the three travel as
// one value — `bits` and `exp` are two readings of the same word, and `sign` is the half
// of the input `bits` no longer carries (it is the magnitude's bits, not the input's).
struct SplitF32 {
    unsigned int bits;
    int exp;
    unsigned char sign;
};

// e4m3's NORMAL path: keep 3 mantissa bits, RNE on the 20 dropped ones, then re-bias.
// Returns the finished code rather than half of one, which is why the sign rides in.
//
// No `contract(off)`, and none needed: every operator below is integer.
__device__ __forceinline__ unsigned char e4m3_normal_code(SplitF32 v) {
    unsigned int mant = v.bits & 0x7fffffu, m3 = mant >> 20, rem = mant & 0xfffffu;
    if (rne_rounds_up(rem, 0x80000u, m3)) m3 += 1u;
    int exp = v.exp + 7;
    if (m3 == 8u) { m3 = 0u; exp += 1; }
    // Unreachable given `f2e4m3_rne`'s `a >= 464` return (which caps m3 at 6 when exp is
    // 15); kept because it is the format's own bound and costs one compare.
    if (exp >= 15 && m3 >= 7u) return v.sign | 0x7e;
    return (unsigned char)(v.sign | (unsigned char)(exp << 3) | (unsigned char)m3);
}

__device__ __forceinline__ unsigned char f2e4m3_rne(float x) {
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
        unsigned int mi = e4m3_subnormal_mantissa(a);
        return mi >= 8u ? (unsigned char)(sign | 0x08) : (unsigned char)(sign | mi);
    }
    return e4m3_normal_code(SplitF32{b, e, sign});
}

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
// `fp4_quant_roundtrip` (actquant.hpp), whose clamp precedes it, and stated so the two
// agree by argument and not by luck.
__device__ __forceinline__ unsigned int f2e2m1(float x) {
    const float mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    float a = fabsf(x);
    unsigned int code = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i) code += (a > mid[i] || (a == mid[i] && (i & 1))) ? 1u : 0u;
    return (signbit(x) ? 8u : 0u) | code;
}
