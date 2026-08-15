// BLOCK activation quantization — the fused quantize-then-dequantize a V4 GEMM's input
// goes through, and the power-of-two scale rule both blocks share. Split out of
// formats.hpp 2026-08-15; the bodies and the measurements in their comments travelled
// verbatim.
//
// The seam is the one formats.hpp's own header already draws. That file owns the ELEMENT
// codec — one number in, one number out, bit-exact against math.rs or the oracle. What is
// here is a BLOCK operation: it needs the block's amax, the block's size, and the format's
// ceiling, and it composes two element codecs rather than defining one. Keeping the two
// apart is why neither file has to explain the other's constants.
#pragma once

#include <hip/hip_runtime.h>

#include "formats.hpp"

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

// ── fp4 ACTIVATION quantization (`kernel.py::fp4_act_quant`, inplace=True) ──────
//
// The indexer's simulator, not the expert weights' codec. `dot_f4_wave_r` (dot_f4.hpp)
// DECODES packed e2m1 nibbles that came off the checkpoint; the pair below ENCODES an
// activation and decodes it straight back, which is how `Indexer.forward` simulates fp4 in
// a bf16 tensor (model.py:376 and :422 — both `q` and the indexer's compressed kv go
// through it).
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

// **Argument budget, and why nothing below is wrapped in a view.** CodeScene's Primitive
// Obsession rule fires on a file with >= 7 functions once the count of primitive-typed
// ARGUMENTS across them reaches 11 (measured against cs 1.0.36, 2026-08-15). This file is
// **3** functions, so the rule cannot fire here at any argument count, and its arity rule
// starts at 5. Splitting these three out of formats.hpp is therefore the WHOLE fix: their
// signatures are untouched, which is what keeps `activation.hip`, `blockindex.hip` and
// `mla.hip` — none of them part of this refactor — compiling unchanged.

// `fast_log2_ceil` / `fast_pow2` / `fast_round_scale` from kernel.py:22-38, by the same
// IEEE-754 bit surgery. Transcribed, not reimplemented: `ceilf(log2f(x))` agrees with this
// everywhere except that it rounds the log FIRST, so a value one ulp below a power of two
// lands on a different binade — one factor-of-two error per group, invisible in aggregate.
__device__ __forceinline__ float fast_round_scale(float amax, float max_inv) {
    // Paired with `formats.hpp::e4m3_subnormal_mantissa`'s pragma — see the argument
    // there. Inert today (the product feeds a bitcast, not an add, so there is nothing to
    // fuse); present so the two halves of `act_quant` carry the same property rather than
    // one of them.
#pragma clang fp contract(off)
    unsigned int b;
    float y = amax * max_inv;
    __builtin_memcpy(&b, &y, sizeof(b));
    int e = (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) ? 1 : 0);
    return __uint_as_float((unsigned int)(e + 127) << 23);
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
