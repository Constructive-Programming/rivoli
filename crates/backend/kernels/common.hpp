// Device helpers shared by the separately-compiled kernel translation units, and the
// UMBRELLA every `.hip` includes: the headers below carry the scalar codecs, the
// wave/block reductions, the by-value argument views and the four weight-format dot
// products, so a kernel keeps seeing one name — `common.hpp` — for all of it.
//
// Split 2026-08-15 (formats/reduce) and again the same day (rowview, actquant and the four
// `dot_*` headers). Nothing was rewritten to make either split fit: the bodies and the
// measurements in their comments travelled verbatim, and the ONLY edit to a body was the
// argument bundling, which `rowview.hpp` argues and which a host equivalence harness
// proved bit-identical before it landed.
//
// | header        | owns                                                             |
// |---------------|------------------------------------------------------------------|
// | formats.hpp   | scalar element codecs: bf16, e4m3, e2m1                           |
// | actquant.hpp  | BLOCK activation quantization (fp8 block-128, fp4 block-32)       |
// | reduce.hpp    | wave/block geometry and the three reductions over it              |
// | rowview.hpp   | the by-value argument views + THE ALIASING RECORD for all of them |
// | dot_fp8.hpp   | fp8 TILE-scaled dot + its LDS LUT builder                         |
// | dot_i4.hpp    | int4 group-scaled dot                                             |
// | dot_f4.hpp    | OCP MX fp4 group-scaled dot + e8m0 + the AS1 span typedef         |
// | dot_vq.hpp    | VQ-int3 codebook dot                                              |
//
// What stayed HERE is what none of them owns and what more than one of them serves: the
// fixed-point MoE accumulator, the clamped SwiGLU both of V4's expert paths share, the
// SiTU-GLU both of K3's do (M9), and the `mla_absorb_fp8` grid bound.
#pragma once

#include <hip/hip_runtime.h>

#include "actquant.hpp"
#include "dot_f4.hpp"
#include "dot_fp8.hpp"
#include "dot_i4.hpp"
#include "dot_vq.hpp"
#include "formats.hpp"
#include "reduce.hpp"
#include "rowview.hpp"

// ── Fixed-point MoE accumulator ─────────────────────────────────────────────────
// Expert contributions are summed as INTEGERS at scale 2^-MOE_ACC_SHIFT, which makes
// the sum ASSOCIATIVE: the result stops depending on which stream finished first. That
// is what lets the per-expert partial rows, the `moe_reduce` over them (that kernel was
// deleted 2026-08-01, once nothing on the decode path called it), and the stream
// join that had to precede it all leave the decode path — the join disappears rather
// than moving, because the only thing that still needs every expert is the residual
// add, which already sits behind the end-of-layer barrier.
//
// ONE i64, not the 128 bits ARCHITECTURE.md §12 sketched. That sketch sized the width
// to represent the SMALLEST partial's full mantissa, which is the wrong question: the
// error that matters is absolute against an order-1 output, not relative against a
// 5e-11 term. Measured partials are |v| <= 15.25 over 21.5M samples, so at shift 44:
//   overflow    Σ over E<=16 clamped terms <= 2^62, a full binade of slack, and the
//               clamp is 1074x the observed worst case;
//   truncation  EXACT for |v| >= 2^-21 (scaling by a power of two only moves the
//               exponent, and llrintf of a value >= 2^24 is identity), and below that
//               bounded by 2^-45 per term — 9 terms give <= 2.6e-13 against the ~8e-6
//               of the f32 reduce it replaces.
// A second limb would buy range there is already 1000x of and precision three orders
// below what the f32 output can carry.
// Only SHIFT is a choice; the other two DERIVE, so raising precision cannot silently
// leave the overflow guard behind. MAX is the clamp that keeps E<=16 terms inside i64:
// 16·MAX·2^SHIFT <= 2^62, i.e. MAX = 2^(58-SHIFT).
#define MOE_ACC_SHIFT 44
#define MOE_ACC_SCALE ((float)(1ull << MOE_ACC_SHIFT))
#define MOE_ACC_MAX ((float)(1ull << (58 - MOE_ACC_SHIFT)))

// f32 → fixed. ponytail: saturate, do not wrap. |v| > 2^14 means the model is already
// broken, but integer overflow wraps to a FINITE wrong value that `flag_nonfinite`
// cannot see, where saturation is monotone and shows up as a stuck output.
__device__ __forceinline__ unsigned long long moe_fixed(float v) {
    return (unsigned long long)llrintf(fminf(fmaxf(v, -MOE_ACC_MAX), MOE_ACC_MAX)
                                       * MOE_ACC_SCALE);
}

// ── V4 shared device helpers ────────────────────────────────────────────────────
//
// None of the three carries a `contract(off)` pragma, and none needs one: `rbf16`
// (formats.hpp) is bit surgery, `block_sum_lds`'s (reduce.hpp) only operator is `+`, and
// `swiglu_clamped`'s below only `+` is `1.0f + expf(...)`, whose addends are a constant and
// a call return — there is no multiply for an FMA to absorb in any of them. (`situ_glu`,
// M9, is the same case as `swiglu_clamped`: its only `+` is the sigmoid denominator's
// constant-plus-call, so the argument covers it unchanged.) That is NOT
// true of `e4m3_subnormal_mantissa` in formats.hpp, which does carry one — see the note
// there for what an ISA diff caught. [CORRECTED 2026-08-15: this named `f2e4m3_rne`, which
// held the pragma until that function was split; it travelled down to the one branch with
// a fusable pair, and the split's note says why that direction is the safe one.]

// `Expert.forward`'s clamped SwiGLU intermediate, BEFORE the routing weight and before the
// bf16 store: `silu(min(bf16(g), L)) · clamp(bf16(u), ±L)` (model.py:601-608).
//
// ONE definition, because there are two callers and they must agree bit for bit or the
// comparison the whole port rests on stops meaning anything: `moe_f4.hip::moe_gateup_f4_impl`
// (the fp4 ROUTED experts) and `activation.hip::swiglu_clamped_bf16` (the fp8 SHARED
// expert). `MoE.__init__` passes `swiglu_limit` to `shared_experts` as well as to the
// routed ones (model.py:632), so the two run the same arithmetic on different weight
// formats. Hoisted here rather than copied because `build.rs`'s jscpd gate does not scan
// `kernels/` — a second copy is invisible to every tool in this tree except the compiler,
// and `rbf16` records that it had THREE copies before anyone noticed.
//
// There is deliberately NO unclamped mode. `limit <= 0` would make this the wrong function
// (`v4oracle::Defect::SwigluUnclamped`, one contribution in seven, fluent and wrong), and
// both launchers refuse it rather than letting a sentinel spell it — so this helper never
// has to decide what a non-positive limit means.
//
// Four things here are load-bearing and each is a defect the oracle can name:
//
//  1. **bf16 FIRST, then clamp.** `Linear` stores bf16 and `Expert.forward` clamps what it
//     read back (`self.w1(x).float()`). Clamping the f32 dot and rounding afterwards puts a
//     different value on the boundary, and the boundary is the only place a clamped SwiGLU
//     differs from an unclamped one at all.
//  2. **The clamp is ASYMMETRIC.** `up` is clamped both sides, `gate` only from above.
//     Clamping the gate from below too is `Defect::SwigluClampGateBothSides`.
//  3. **`x · sigmoid(x)`, not `x / (1 + e^-x)`.** `F.silu` is the multiply form, and this is
//     NOT `activation.hip::swiglu`'s division form. The two differ by one rounding that
//     would normally vanish under the bf16 store the callers apply — except exactly at a
//     rounding boundary, where it flips a whole bf16 ulp. Matching the reference's
//     association removes a systematic term from every comparison this port makes.
//  4. **The product is returned UNROUNDED**, in f32. Both callers round it, but not at the
//     same point: the fp4 path folds the routing weight in first and rounds
//     `bf16(silu·up·w)`, which is where the reference rounds (`weights * x` precedes
//     `x.to(dtype)`), while the shared expert has no routing weight at all
//     (model.py:648) and rounds the bare product. Rounding here would double-round the fp4
//     path — a second bf16 step the reference does not take.
__device__ __forceinline__ float swiglu_clamped(float g, float u, float limit) {
    float gt = rbf16(g), ut = rbf16(u);
    ut = fminf(fmaxf(ut, -limit), limit);
    gt = fminf(gt, limit);  // ASYMMETRIC: no lower clamp on the gate
    return gt * (1.0f / (1.0f + expf(-gt))) * ut;
}

// **SiTU-GLU — Kimi-K3's activation**, `k3-architecture.md` §8 and the reference's
// `SituAndMul.forward` (ported verbatim from `k3:kernels/common.hpp`, M9):
//
//     a = b1 · tanh(gate / b1) · sigmoid(gate)      b1 = activation_situ_beta        = 4
//     u = b2 · tanh(up   / b2)                      b2 = activation_situ_linear_beta = 25
//     y = a · u                                     so |y| <= b1·b2 = 100
//
// USED BY: Kimi-K3 only, and by all three of its MLP shapes — the dense layer 0, the shared
// MLP, and every routed expert. GLM-5.2 and DeepSeek-V4-Flash use `swiglu_clamped` above and
// never reach this.
//
// It lives HERE, next to `swiglu_clamped`, for that helper's own reason: `activation.hip`'s
// dense launcher and `moe_f4.hip`'s fused fp4 expert kernel must run the identical
// arithmetic, and `kernels/` is not watched by the jscpd gate, so a second copy would drift
// in silence. The plan says the same thing from the other end — "an implementer who changes
// the dense path alone watches it go right and every routed expert stay wrong".
//
// **It is NOT `swiglu_clamped` with different constants, and the merge that parameterises one
// into the other must never happen** — the identical hazard that entry already carries. Four
// differences, each of which is a defect a fixture can name:
//
//  1. **The sigmoid takes the UNCAPPED gate.** `tanh(g/b1)` bounds the first factor and
//     `sigmoid(g)` is handed the raw dot, so the two factors saturate at different rates.
//     Feeding the capped value to the sigmoid is smooth, plausible and wrong everywhere the
//     gate is large.
//  2. **`up` is transformed, not clamped.** `b2·tanh(up/b2)` saturates smoothly; a clamp is
//     piecewise-linear with a corner. They agree only near zero.
//  3. **Neither operand is bf16-rounded here.** `swiglu_clamped` rounds first because V4's
//     `Expert.forward` reads its operands back from a bf16 `Linear` store explicitly. K3's
//     `SituAndMul` does `.to(torch.float32)`, which is a widening no-op — whatever rounding
//     the real model does happened at the projection, and rivoli's trunk carries f32
//     activations throughout. Rounding here would be a step the reference does not take.
//  4. **The product is returned UNROUNDED**, and the rounding is the caller's. `moe_f4.hip`'s
//     fp4 caller stores `rbf16(y)` — pass 1's bf16 store — and applies the routing weight in
//     pass 2, while the dense and shared callers have no routing weight and no store here at
//     all. (CORRECTED 2026-08-12: this said "the fp4 caller folds the routing weight in and
//     rounds `bf16(y·w)`", which item 4b moved to pass 2. The conclusion was right and the
//     premise had become false — and re-folding `w` into pass 1 is exactly the defect
//     `folding_the_routing_weight_before_w2_is_not_the_same_function` exists to price.)
//
// `1.0f/(1.0f+expf(-g))` and not a multiply form: `torch.sigmoid` is the division form, and
// unlike `F.silu` there is no `x·sigmoid(x)` idiom here to match — the sigmoid's argument is
// not the factor it multiplies.
__device__ __forceinline__ float situ_glu(float g, float u, float b1, float b2) {
    const float a = b1 * tanhf(g / b1) * (1.0f / (1.0f + expf(-g)));
    return a * (b2 * tanhf(u / b2));
}

// i-quads per head in `mla_absorb_fp8` — one thread each. Shared by the kernel (which
// derives head/i0 from it) and its launcher (which sizes the grid from it), because a
// grid smaller than the kernel's own bound leaves output columns UNWRITTEN with no error
// code and no fault. `H * kvl` was self-evidently the same expression on both sides; a
// rounding formula stated twice is not.
__host__ __device__ __forceinline__ int absorb_nquad(int kvl) { return (kvl + 3) >> 2; }
