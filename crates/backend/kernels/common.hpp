// Device helpers shared by the separately-compiled kernel translation units, and the
// UMBRELLA every `.hip` includes: the two headers below carry the scalar codecs and the
// wave/block reductions, so a kernel keeps seeing one name — `common.hpp` — for all of it.
// Split 2026-08-15 under the per-file line ceiling. Nothing was rewritten to make the
// split fit: the bodies and the measurements in their comments travelled verbatim.
//
// What stayed here is what the two headers are COMPOSED into: the fixed-point MoE
// accumulator, the clamped SwiGLU both expert paths share, and the four wave-cooperative
// dot products (fp8 / int4 / fp4 / vq-int3) with the weight-layout constants they index
// packed rows with.
#pragma once

#include <hip/hip_runtime.h>

#include "formats.hpp"
#include "reduce.hpp"

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
// a call return — there is no multiply for an FMA to absorb in any of them. That is NOT
// true of `e4m3_subnormal_mantissa` in formats.hpp, which does carry one — see the note
// there for what an ISA diff caught. [CORRECTED 2026-08-15: this named `f2e4m3_rne`, which
// held the pragma until that function was split; it travelled down to the one branch with
// a fusable pair, and the split's note says why that direction is the safe one.]

// `Expert.forward`'s clamped SwiGLU intermediate, BEFORE the routing weight and before the
// bf16 store: `silu(min(bf16(g), L)) · clamp(bf16(u), ±L)` (model.py:601-608).
//
// ONE definition, because there are two callers and they must agree bit for bit or the
// comparison the whole port rests on stops meaning anything: `moe.hip::moe_gateup_f4_impl`
// (the fp4 ROUTED experts) and `linalg.hip::swiglu_clamped_bf16` (the fp8 SHARED expert).
// `MoE.__init__` passes `swiglu_limit` to `shared_experts` as well as to the routed ones
// (model.py:632), so the two run the same arithmetic on different weight formats. Hoisted
// here rather than copied because `build.rs`'s jscpd gate does not scan `kernels/`
// (build.rs:618) — a second copy is invisible to every tool in this tree except the
// compiler, and `rbf16` above records that it had THREE copies before anyone noticed.
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
//     NOT `linalg.hip::swiglu`'s division form. The two differ by one rounding that would
//     normally vanish under the bf16 store the callers apply — except exactly at a rounding
//     boundary, where it flips a whole bf16 ulp. Matching the reference's association
//     removes a systematic term from every comparison this port makes.
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

// log2 of a power-of-two fp8 scale-tile size. Every launcher taking a `block`
// REJECTS a non-power-of-two (arg guard 1003), so the hot loops index the block scale
// with a shift instead of a divide. `block` is a runtime int, and LLVM does
// strength-reduce `i / block` to a magic multiply — but the SIGNED quotient correction
// survives it, and that is what cost: measured 44 → 29 VALU per iteration in
// gemv_fp8_splitk, around 5 ops of real math, with the memory ops unchanged at 7.
// Bit-identical: same index, same order. See benchmarks.md, "Read the ISA before you
// book the device", for why grepping for `v_rcp_iflag_f32` finds nothing here.
__device__ __forceinline__ int blk_shift(int block) { return 31 - __clz(block); }

// i-quads per head in `mla_absorb_fp8` — one thread each. Shared by the kernel (which
// derives head/i0 from it) and its launcher (which sizes the grid from it), because a
// grid smaller than the kernel's own bound leaves output columns UNWRITTEN with no error
// code and no fault. `H * kvl` was self-evidently the same expression on both sides; a
// rounding formula stated twice is not.
__host__ __device__ __forceinline__ int absorb_nquad(int kvl) { return (kvl + 3) >> 2; }

// Wave-cooperative fp8-e4m3 block-scaled dot for one output row `o`. `wrow` =
// packed[o*i_dim..], `scalerow` = scale + (o/block)*sc_cols (the row's block-scale
// row), so element i uses scalerow[i>>bsh]. `lut` is the block's e4m3_lut_build
// table. Matches quant.rs::matvec_fp8.
// Strided fp8 dot accumulation (NO cross-thread reduction): Σ over the columns this
// thread owns of `x·lut[w]·scale`, using a uint-per-lane load (4 fp8 → 128B/wave) atop
// the LUT + a scalar tail. The caller reduces. [CORRECTED 2026-08-08: this said "shared
// by `dot_fp8_wave` and `gemv_fp8_splitk`" — the splitk kernels moved to
// `fp8_dot_strided_r` when the `_r` family landed, so `dot_fp8_wave` → `gemv_fp8_bf16` is
// the ONLY consumer, which is what scopes the M7 unroll's blast radius below.]
__device__ __forceinline__ float fp8_dot_strided(const float* __restrict__ x,
                                                 const unsigned char* __restrict__ wrow,
                                                 const float* __restrict__ scalerow,
                                                 int i_dim, int block,
                                                 const float* __restrict__ lut,
                                                 int start, int stride) {
    float acc = 0.0f;
    const unsigned int* w4 = (const unsigned int*)wrow;
    // The dword path applies ONE block scale to a quad's four columns, so it is only the
    // right scale when the tile is at least a quad wide — at block 1 or 2 the columns
    // past the tile boundary were silently given i0's. Both are powers of two, so guard
    // 1003 passes them. Zeroing `n4` hands the row to the per-column tail below, which
    // was already correct; block ≥ 4 (the engine runs 128) is untouched and bit-identical.
    // TWO KNOWN GAPS REMAIN, both recorded in docs/reference/vulkan-kernels.md
    // ("Known gaps in the fp8 dot"; this cited docs/PERF.md #4 until 2026-08-01,
    // a file split three ways whose item numbering no longer exists): the Vulkan twin
    // (vk/fp8.glsl) still has this bug and its oracle mirrors it, and `rivoli_gemv_fp8`
    // still does not guard the `i_dim % 4` this `w4` cast needs — a requirement that is
    // now CONDITIONAL, since at block < 4 the cast is never reached.
    int n4 = (block >= 4) ? (i_dim >> 2) : 0;
    int bsh = blk_shift(block);
    // Unrolled for memory-level parallelism, NOT for issue: un-unrolled, the emitted loop
    // issued its three vmem loads and then waited them all down (`s_waitcnt vmcnt(0)`
    // before the closing fmac, EVERY iteration), so each wave held exactly one 128-B
    // weight request in flight per GTT round-trip — M6 measured the only kernel built
    // from this loop (`gemv_fp8_bf16`) at 2–2.8× its bytes on all three of its spans while
    // issue sat at 42 instr/128 B, ~3× lighter than streaming needs. Unroll 8 puts 1 KB
    // of weight stream in flight per wave (read back from the ISA: loads `s_clause`'d,
    // waits counted down from vmcnt(23), ONE vmcnt(0) per 1 KB) at VGPR 86 = 96/wave =
    // still 16 waves/SIMD — the next granule (>96) drops to 12, so a change that grows
    // this body must re-read the ISA. Bit-identical by the M5 unroll argument: `acc` is
    // one serial FP chain and LLVM neither splits nor re-associates it without
    // fast-math; the fold order stays ascending `j`. Pinned on hardware by
    // `tests/f4_kernel.rs::the_fp8_dot_sums_in_source_order_through_both_loops`, which
    // also walks the unroll REMAINDER loop no engine dimension reaches (every real
    // per-lane trip count divides 8). `fp8_dot_strided_r` below is deliberately NOT
    // unrolled: its callers (GLM splitk, mla absorb/value) measured at budget, and R
    // multiplies the register cost of the same pragma. Details and the registered
    // prediction: docs/investigations/v4-decode-decomposition.md §M7.
#pragma unroll 8
    for (int j = start; j < n4; j += stride) {
        unsigned int p = w4[j];
        int i0 = j << 2;
        float s = scalerow[i0 >> bsh];
        acc += s * (x[i0]     * lut[(unsigned char)p]
                  + x[i0 + 1] * lut[(unsigned char)(p >> 8)]
                  + x[i0 + 2] * lut[(unsigned char)(p >> 16)]
                  + x[i0 + 3] * lut[(unsigned char)(p >> 24)]);
    }
    for (int i = (n4 << 2) + start; i < i_dim; i += stride)
        acc += x[i] * lut[wrow[i]] * scalerow[i >> bsh];
    return acc;
}

// R INPUT ROWS against one weight row — the fp8 twin of `dot_vq_wave_r`, and the same
// argument for existing. The attention projections read 165 MB of fp8 per layer (o_proj
// alone is 100 MB) against a 24 KB row of `x`, so the weight side is the cost and R rows
// through one read of it is what makes a speculative verify pass cheap.
//
// `x` rows are `x_stride` apart. NO cross-thread reduction here either — `dot_fp8_wave_r`
// wave-sums, `gemv_fp8_splitk_r` reduces in LDS.
//
// R=1 is BIT-IDENTICAL to the scalar form: hoisting the four `lut[]` reads into locals
// changes neither the values nor the order they are summed in.

// One dword's four columns folded into R accumulators — the weights already decoded
// through the LUT and the block scale already read, because both amortise over the rows.
// Lifted out of `fp8_dot_strided_r`'s inner loop 2026-08-15, SHAPE ONLY: the same four
// products in the same `s * (…)` grouping, the same ascending fold into `acc[r]`.
//
// The four weights arrive as SCALARS, not as a pointer to the caller's local array. An
// array would have to survive SROA to stay in registers, and there is no reason to hand
// the optimizer that job when the values are already in registers at the call.
template <int R>
__device__ __forceinline__ void fp8_quad_accum(const float* __restrict__ x, int x_stride, int i0,
                                               float s, float l0, float l1, float l2, float l3,
                                               float* __restrict__ acc) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float* xr = x + (size_t)r * x_stride;
        acc[r] += s * (xr[i0] * l0 + xr[i0 + 1] * l1 + xr[i0 + 2] * l2 + xr[i0 + 3] * l3);
    }
}

// One column folded into R accumulators — the remainder path, where the dword read does
// not fit.
//
// `x * l * s`, NOT `x * (l*s)`. Folding the two weight-side factors first would be one
// fewer multiply per row and a DIFFERENT number — fp multiplication does not associate —
// which is exactly the kind of drift the bit-identity test exists to refuse. The loads are
// hoisted at the call; the arithmetic is not touched.
template <int R>
__device__ __forceinline__ void fp8_col_accum(const float* __restrict__ x, int x_stride, int i,
                                              float l, float s, float* __restrict__ acc) {
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] += x[(size_t)r * x_stride + i] * l * s;
}

template <int R>
__device__ __forceinline__ void fp8_dot_strided_r(const float* __restrict__ x, int x_stride,
                                                  const unsigned char* __restrict__ wrow,
                                                  const float* __restrict__ scalerow,
                                                  int i_dim, int block,
                                                  const float* __restrict__ lut,
                                                  int start, int stride,
                                                  float* __restrict__ acc) {
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    const unsigned int* w4 = (const unsigned int*)wrow;
    // See the single-row note above for why `block < 4` zeroes the dword path.
    int n4 = (block >= 4) ? (i_dim >> 2) : 0;
    int bsh = blk_shift(block);
    for (int j = start; j < n4; j += stride) {
        unsigned int p = w4[j];
        int i0 = j << 2;
        fp8_quad_accum<R>(x, x_stride, i0, scalerow[i0 >> bsh], lut[(unsigned char)p],
                          lut[(unsigned char)(p >> 8)], lut[(unsigned char)(p >> 16)],
                          lut[(unsigned char)(p >> 24)], acc);
    }
    for (int i = (n4 << 2) + start; i < i_dim; i += stride)
        fp8_col_accum<R>(x, x_stride, i, lut[wrow[i]], scalerow[i >> bsh], acc);
}

// One WAVE reduces one row: strided MAC over the wave's lanes, then a wave-sum.
__device__ __forceinline__ float dot_fp8_wave(const float* __restrict__ x,
                                              const unsigned char* __restrict__ wrow,
                                              const float* __restrict__ scalerow,
                                              int i_dim, int block, int lane,
                                              const float* __restrict__ lut) {
    return wave_sum(fp8_dot_strided(x, wrow, scalerow, i_dim, block, lut, lane, WAVE));
}

// R input rows, one wave, one weight row. R=1 is bit-identical to `dot_fp8_wave`:
// `fp8_dot_strided_r<1>` reproduces the scalar accumulation exactly and `wave_sum` is
// the same reduction.
template <int R>
__device__ __forceinline__ void dot_fp8_wave_r(const float* __restrict__ x, int x_stride,
                                               const unsigned char* __restrict__ wrow,
                                               const float* __restrict__ scalerow,
                                               int i_dim, int block, int lane,
                                               const float* __restrict__ lut,
                                               float* __restrict__ out) {
    float acc[R];
    fp8_dot_strided_r<R>(x, x_stride, wrow, scalerow, i_dim, block, lut, lane, WAVE, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) out[r] = wave_sum(acc[r]);
}

// int4 group-scale parameters — MUST match quant.rs (I4_GROUP). One f32 scale per
// I4_GROUP weights along the input dim, so the scale lives INSIDE the dot.
#define I4_GROUP 128
#define I4_GROUP_SHIFT 7
static_assert((1 << I4_GROUP_SHIFT) == I4_GROUP, "I4_GROUP_SHIFT must be log2(I4_GROUP)");
static_assert(I4_GROUP % 8 == 0, "the dword fast path's 8 columns must not straddle a group");

// Wave-cooperative group-scaled int4 dot for one output row, result on lane 0:
//   Σ_i v[i]·(nibble(i) − 8)·scalerow[i / I4_GROUP]
// `row` = packed[o*rb..], rb = (dim+1)/2; `scalerow` = scale + o*i4_groups(dim).
// Matches quant.rs::matvec_i4. NOTE the scale is applied HERE, per group — under the
// old per-row format the caller applied one scale outside the dot, which is exactly
// what a group scale cannot express.
//
// The fast path reads a dword (8 nibbles = 8 consecutive columns) per lane when `row`
// is 4-byte aligned (the dim/2 row stride, dim a multiple of 8, keeps every row
// aligned). Those 8 columns start at a multiple of 8 and I4_GROUP is a multiple of 8,
// so they always share ONE group scale — one extra multiply per 8 columns, and the
// scalar tail below computes the same per-element product.
__device__ __forceinline__ float nib(unsigned int w, int k) {
    return (float)((int)((w >> (4 * k)) & 0xFu) - 8); // nibble k → signed weight
}
// `dot_i4_wave_r`'s dword fast path, returning the column `base` its scalar tail resumes
// from. A row that is not 4-byte aligned returns 0 and is handed to the tail whole — the
// same partition the wrapping `if` used to express, written as an early return so this
// loop sits at one level of nesting instead of two.
//
// Lifted out 2026-08-15, SHAPE ONLY. The `#pragma unroll 4` and every number the note
// inside it measures travel WITH the loop; the fold order, the two accumulators and the
// group-scale hoist are untouched.
template <int R>
__device__ __forceinline__ int i4_dword_pass(const float* __restrict__ v, int v_stride,
                                             const unsigned char* __restrict__ row,
                                             const float* __restrict__ scalerow, int dim, int lane,
                                             float* __restrict__ a0, float* __restrict__ a1) {
    int base = 0;
    if ((((size_t)row) & 3u) != 0) return base;
    const unsigned int* rw = (const unsigned int*)row;
    // Unrolled 2026-08-09, MEASURED, not copied from fp4. Un-pragma'd, this loop issued
    // 4 (R=1) / 6 (R=2) loads and drained them all in-body — `vmcnt(3) lgkmcnt(0)` down
    // to `vmcnt(0)`, one iteration in flight, M7's disease and the same gap M11 fixed on
    // `dot_f4_wave_r` below. `dot_bench glmi4` at the artifact's dims (6144x2048, 1.083 GB
    // rotating past the 32 MB MALL, two counterbalanced passes, benchmarks.md "GLM int4
    // MoE unroll round"): depth 4 is **+12.6% at R=1 (169.2 -> 190.6 GB/s), +16.4% at
    // R=2 (163.3 -> 190.1), +20.1% at e_count=1 (125.9 -> 151.3)**, fingerprint-identical
    // on BOTH token rows; depth 2 gave +11.6%/+3.0%, so depth 4 is the rung that pays at
    // the R=2 width speculative decode actually runs.
    //
    // The register cost the un-unrolled comment here worried about was measured FIRST and
    // does not bite: depth 4 is VGPR 88/123/83/95 across the four kernels, occupancy 16
    // everywhere except gateup_r2 at 10 waves/SIMD — exactly where fp4's winning arm sat —
    // zero spill, zero scratch. Two adjacent negatives, priced in the same round so nobody
    // re-tries them: AS1-typing these pointers (the fp4 `gu8p` treatment; flat_load ->
    // global_load, the lgkmcnt coupling gone) measured +-0.6% == nothing, and a ballast
    // with the whole nibble decode and every FMA removed measured +0.5%/-1.1% — the decode
    // is FREE at the un-unrolled schedule, so the entire gap was memory-level parallelism.
    //
    // Fold order is unchanged — each accumulator stays one serial fadd chain, ascending
    // `base` (the fingerprint gate would have caught anything else; arm X, a deliberate
    // even/odd split, moved it and was measured doing so). Multi-trip coverage INCLUDING
    // the remainder loop this pragma creates: tests/kernel.rs::
    // the_i4_dword_path_matches_the_oracle_at_multiple_trips (1280/1024 = 5 and 4 trips;
    // every engine dim is 0 mod 4, so only that fixture ever enters the epilogue).
#pragma unroll 4
    for (; base + WAVE * 8 <= dim; base += WAVE * 8) {
        int col = base + lane * 8;
        unsigned int w = rw[col >> 3];              // 8 nibbles = 8 consecutive columns
        float s = scalerow[col >> I4_GROUP_SHIFT];  // one group for all 8
        // Decoded ONCE for every row.
        float n0 = nib(w, 0), n1 = nib(w, 1), n2 = nib(w, 2), n3 = nib(w, 3);
        float n4 = nib(w, 4), n5 = nib(w, 5), n6 = nib(w, 6), n7 = nib(w, 7);
#pragma unroll
        for (int t = 0; t < R; ++t) {
            const float* vt = v + (size_t)t * v_stride;
            float4 x0 = *(const float4*)(vt + col);
            float4 x1 = *(const float4*)(vt + col + 4);
            a0[t] += s * (x0.x * n0 + x0.y * n1 + x0.z * n2 + x0.w * n3);
            a1[t] += s * (x1.x * n4 + x1.y * n5 + x1.z * n6 + x1.w * n7);
        }
    }
    return base;
}

// The per-column remainder, from `base` to `dim`. Folds into `a0` only — `a1` exists to
// give the dword path two independent FMA chains and there is no second chain here.
//
// The left-to-right `v[i] * (n−8) * scale` is the arithmetic, not an accident of
// spelling: hoisting `(n−8) * scale` out of the row loop would re-associate the product.
template <int R>
__device__ __forceinline__ void i4_tail_accum(const float* __restrict__ v, int v_stride,
                                              const unsigned char* __restrict__ row,
                                              const float* __restrict__ scalerow, int dim,
                                              int base, int lane, float* __restrict__ a0) {
    for (int i = base + lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
#pragma unroll
        for (int t = 0; t < R; ++t)
            a0[t] += v[(size_t)t * v_stride + i] * (float)(n - 8) * scalerow[i >> I4_GROUP_SHIFT];
    }
}

// R token rows against ONE read of the weight row — the nibble decode and the group scale
// amortise over the rows, which is the whole point of batching (the weight read is ~92% of
// an expert launch). `v_stride` is the row-minor stride of `v`.
//
// R=1 is BIT-IDENTICAL to `dot_i4_wave` below, and that is a constraint on how this is
// written, not a hope: the fast path's `s * (x·n + …)` grouping and the scalar tail's
// left-to-right `v[i] * (n−8) * scale` are both reproduced exactly. Hoisting the nibbles
// is safe (same values, same order); hoisting `(n−8) * scale` out of the tail would NOT be
// — it re-associates the product. See tests/kernel.rs.
//
// The two passes are a PARTITION of the columns, which is why `base` is threaded between
// them by value rather than each pass deciding its own bound.
template <int R>
__device__ __forceinline__ void dot_i4_wave_r(const float* __restrict__ v, int v_stride,
                                              const unsigned char* __restrict__ row,
                                              const float* __restrict__ scalerow,
                                              int dim, int lane, float* __restrict__ out) {
    float a0[R], a1[R];
#pragma unroll
    for (int t = 0; t < R; ++t) { a0[t] = 0.0f; a1[t] = 0.0f; }
    int base = i4_dword_pass<R>(v, v_stride, row, scalerow, dim, lane, a0, a1);
    i4_tail_accum<R>(v, v_stride, row, scalerow, dim, base, lane, a0);
#pragma unroll
    for (int t = 0; t < R; ++t) out[t] = wave_sum(a0[t] + a1[t]);
}

__device__ __forceinline__ float dot_i4_wave(const float* __restrict__ v,
                                             const unsigned char* __restrict__ row,
                                             const float* __restrict__ scalerow,
                                             int dim, int lane) {
    // Two accumulators + two float4 x-loads per lane per step: the 8 columns split
    // into independent FMA chains (ILP), and x streams as 2×16B vector loads instead
    // of 8 scalar loads. int4 is sequential-coalesced, so this saturates the L1/x
    // bandwidth and keeps the ALUs busy (nibble-decode) — unlike the VQ dot's random
    // codebook gather.
    float a0 = 0.0f, a1 = 0.0f;
    int base = 0;
    if ((((size_t)row) & 3u) == 0) {
        const unsigned int* rw = (const unsigned int*)row;
        for (; base + WAVE * 8 <= dim; base += WAVE * 8) {
            int col = base + lane * 8;
            unsigned int w = rw[col >> 3]; // 8 nibbles = 8 consecutive columns
            float s = scalerow[col >> I4_GROUP_SHIFT]; // one group for all 8
            float4 x0 = *(const float4*)(v + col);
            float4 x1 = *(const float4*)(v + col + 4);
            a0 += s * (x0.x * nib(w, 0) + x0.y * nib(w, 1) + x0.z * nib(w, 2) + x0.w * nib(w, 3));
            a1 += s * (x1.x * nib(w, 4) + x1.y * nib(w, 5) + x1.z * nib(w, 6) + x1.w * nib(w, 7));
        }
    }
    for (int i = base + lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        a0 += v[i] * (float)(n - 8) * scalerow[i >> I4_GROUP_SHIFT];
    }
    return wave_sum(a0 + a1);
}

// ── OCP MX FP4 (DeepSeek-V4-Flash's native routed experts) ──────────────────────
//
// e2m1 nibbles with ONE e8m0 (bare power-of-two) scale per F4_GROUP weights along the
// input dim. Three things separate this from the int4 block above and every one of them
// is silent-wrong if it drifts:
//
//   * the codebook is NON-UNIFORM ({0,.5,1,1.5,2,3,4,6}), not `nibble − 8`, so an int4
//     decode of fp4 bytes produces plausible magnitudes and wrong text;
//   * the group is 32, not 128 — a group stride read from I4_GROUP mis-scales 3 of every
//     4 groups while leaving the first one right, which is exactly the kind of bug that
//     looks like "mostly working";
//   * the scale grid is [o_dim, ceil(K/32)] row-major, one scale ROW per output row,
//     where fp8's is a 128×128 TILE grid shared by 128 output rows.
//
// MUST match quant.rs (F4_GROUP / F4_LUT / e8m0) and v4oracle::numerics (e2m1_decode /
// e8m0_decode); the decoders themselves are `e2m1f`/`e8m0f` in formats.hpp. The oracle
// is the reference; these two are independently written from the same format
// definition, which is the point — see v4oracle/numerics.rs's module doc.
#define F4_GROUP 32
#define F4_GROUP_SHIFT 5
static_assert((1 << F4_GROUP_SHIFT) == F4_GROUP, "F4_GROUP_SHIFT must be log2(F4_GROUP)");
static_assert(F4_GROUP % 8 == 0, "the dword fast path's 8 columns must not straddle a group");

// A GLOBAL-address-space (AMDGCN AS1) byte pointer, for spans whose globalness clang
// cannot infer. A pointer that arrives as a KERNEL ARGUMENT is promoted to AS1 by
// AMDGPUPromoteKernelArguments (that is why the fp4 dot's activation loads lower to
// `global_load_b128`), but a pointer LOADED from device memory — the six spans in an
// `ExpertDescF4` — reaches the loads as a generic value, and the M3a ISA read
// (docs/investigations/v4-decode-decomposition.md) measured the consequence: the weight
// dword and scale byte lower to `flat_load_*`, which on gfx11 take the slower path, count
// against `lgkmcnt` too, and put a mid-body `s_waitcnt` in an issue-bound loop. Typing the
// span AS1 at the source level is the fix that cannot be optimized away — a round-trip
// `(T*)(AS1 T*)p` cast is folded by instcombine, and an
// `assume(!is_shared && !is_private)` was measured not to reach InferAddressSpaces either
// (both tried 2026-08-08; the loads stayed flat). The generic→AS1 cast is a no-op on
// gfx11 (the representations coincide) and asserts only the address SPACE, nothing about
// aliasing.
typedef const unsigned char __attribute__((address_space(1)))* gu8p;

// Wave-cooperative group-scaled FP4 dot, R token rows against ONE read of the weight row:
//   out[t] = Σ_i v[t·v_stride + i] · e2m1(nibble(i)) · e8m0(scalerow[i / F4_GROUP])
// `row` = packed[o·(dim/2)..], `scalerow` = scale + o·ceil(dim/F4_GROUP). Result on lane 0.
//
// Same two-path shape as `dot_i4_wave_r` — a dword (8 nibbles = 8 consecutive columns) per
// lane when `row` is 4-byte aligned, a scalar tail otherwise — for the same reason: 8
// columns starting at a multiple of 8 always share ONE group scale, so the scale decode
// amortises 8×. The nibble ORDER is the checked half: nibble k of the dword is column
// col+k, i.e. a byte's LOW nibble is the EVEN column, which is what `convert.py`'s
// `stack([TABLE[low], TABLE[high]]).flatten()` and `WMat::Fp4::row` both mean. Reading it
// high-first is a permutation INSIDE each scale group, so it survives every summary
// statistic — `v4oracle::Defect::Fp4NibbleSwap` is what keeps it A/B-able.
//
// NOT merged with `dot_i4_wave_r` above, and the reason is arithmetic rather than caution:
// that one multiplies `v[i] · (n−8) · scale` PER ELEMENT because hoisting its f32 group
// scale would re-associate the product (see its own note, and the bit-identity constraint
// `tests/kernel.rs` pins on it). This one hoists the scale out of eight columns, which is
// legal only because e8m0 is a bare power of two and the multiply is therefore exact. A
// shared template would need a policy parameter meaning "is this scale exact?" — an
// abstraction whose two instantiations disagree at the one line it exists to share.
// `row` and `scalerow` are `gu8p` — AS1-typed, see the typedef above — because both come
// out of an `ExpertDescF4` at every call site and the flat_load they otherwise lower to is
// a measured cost on an issue-bound loop.
// `dot_f4_wave_r`'s dword fast path, returning the column `base` its scalar tail resumes
// from — the fp4 twin of `i4_dword_pass` above, and lifted out on the same day and for the
// same reason (SHAPE ONLY; the pragma, the fold order and every measured number below
// travel with the loop).
//
// Predicated on the WEIGHT row's 4-byte alignment, which is what the dword read needs; an
// unaligned row returns 0 and the tail takes it whole. The `float4` loads below are on the
// ACTIVATION and are NOT checked — a 16-byte-aligned `v` is an unchecked caller obligation
// (`launch_moe_expert_range_f4`'s `# Safety`), and a misaligned one faults rather than
// falling back to the tail.
template <int R>
__device__ __forceinline__ int f4_dword_pass(const float* __restrict__ v, int v_stride,
                                             gu8p __restrict__ row, gu8p __restrict__ scalerow,
                                             int dim, int lane, float* __restrict__ a0,
                                             float* __restrict__ a1) {
    int base = 0;
    if ((((size_t)row) & 3u) != 0) return base;
    const unsigned int __attribute__((address_space(1)))* rw =
        (const unsigned int __attribute__((address_space(1)))*)row;
    // Unrolled 2026-08-09 (M11). Un-pragma'd, this loop drained inside its own body
    // (`s_waitcnt vmcnt(1)`/`vmcnt(0)` in-body) — ONE iteration of loads in flight, ever,
    // which is M7's disease; `fp8_dot_strided` above was fixed for it and the exemption
    // beside that pragma names `fp8_dot_strided_r`, so this loop was never in that
    // conversation. `dot_bench v4res` MICROBENCHMARK at the engine's dims: 146.63 →
    // 195.27 GB/s. Depth is what buys it — a ballast with the decode and every FMA
    // removed buys only +12.8%, so the bound was never issue rate.
    //
    // A change that grows this body must re-read the ISA. Unroll 4 is 125 VGPR on
    // `moe_gateup_f4` (**10 waves/SIMD**, down from 16) and 93 on `moe_down_f4` (still 16);
    // no spill on either, 0 scratch. Unroll 2 keeps 16 waves everywhere at 74/66 VGPR and
    // measured 172.55; **both rungs are registered arms of an engine A/B and
    // `docs/investigations/v4-decode-decomposition.md` §M11b decides between them on the
    // engine wall, not this comment.**
    //
    // Fold order must stay the ascending-`base` sum, and this was READ OUT of the
    // compiler rather than argued: the device IR carries `contract` (FMA fusion, which
    // stock already relies on) but **zero `reassoc`/`fast`/`nnan`** at stock, unroll 2 and
    // unroll 4; each accumulator stays a SINGLE serial fadd chain terminating at the loop
    // phi — exactly two float phis in the loop header at all three depths, no partial-
    // accumulator split — and the chain is `phi + t(base) + t(base+256) + …`, ascending.
    // The epilogue remainder chains off the main loop's exit values, so it continues the
    // same order. `-ffast-math` is absent (`build.rs`), but `-ffp-contract=fast` is clang's
    // HIP default, which is why this was read and not assumed
    // (`tests/f4_kernel.rs::the_fp4_dispatch_hash_pins_the_clamp_hoist` records that
    // lesson).
    //
    // Multi-trip coverage is `tests/f4_kernel.rs::the_dword_path_matches_the_oracle_at_
    // multiple_trips` (1280/1024 = 5 and 4 trips; at `V4Config::toy` gate/up runs this loop
    // ONCE and down never enters it, so every other test executes only the remainder copy).
    // Measured 2026-08-09: it gates arithmetic wrong past the first trip — red at 65x
    // tolerance on an injected `n7 = 0 when base != 0`, while all 27 other tests stay green.
    // It does NOT gate this bound (measured 2026-08-09 by building both): `<=` -> `<` moves
    // the last trip into the scalar tail below, which resumes from `base`, and the test
    // still passes — the two paths are a PARTITION, so an off-by-one that SHORTENS this
    // loop is a performance bug no test will catch. Loosening it is a different animal:
    // reading `rw[col >> 3]` past the row faults or corrupts. The remainder is unreachable
    // at the engine's dims but reachable in production in principle — the launcher guards
    // `% ACT_QUANT_BLOCK` (128), not 256, so a conforming dim like 1152 or 3712 gives an
    // odd trip count.
#pragma unroll 4
    for (; base + WAVE * 8 <= dim; base += WAVE * 8) {
        int col = base + lane * 8;
        unsigned int w = rw[col >> 3];             // 8 nibbles = 8 consecutive columns
        float s = e8m0f(scalerow[col >> F4_GROUP_SHIFT]);  // one group for all 8
        // Decoded ONCE for every row — the weight side is the expensive half.
        float n0 = e2m1f(w), n1 = e2m1f(w >> 4), n2 = e2m1f(w >> 8), n3 = e2m1f(w >> 12);
        float n4 = e2m1f(w >> 16), n5 = e2m1f(w >> 20), n6 = e2m1f(w >> 24),
              n7 = e2m1f(w >> 28);
#pragma unroll
        for (int t = 0; t < R; ++t) {
            const float* vt = v + (size_t)t * v_stride;
            float4 x0 = *(const float4*)(vt + col);
            float4 x1 = *(const float4*)(vt + col + 4);
            a0[t] += s * (x0.x * n0 + x0.y * n1 + x0.z * n2 + x0.w * n3);
            a1[t] += s * (x1.x * n4 + x1.y * n5 + x1.z * n6 + x1.w * n7);
        }
    }
    return base;
}

// The per-column remainder, from `base` to `dim` — `i4_tail_accum`'s fp4 twin. Folds into
// `a0` only, for the same reason: `a1` is the dword path's second FMA chain and there is
// no second chain here. The scale is decoded PER COLUMN rather than hoisted, because this
// path has no group of eight to hoist it out of.
template <int R>
__device__ __forceinline__ void f4_tail_accum(const float* __restrict__ v, int v_stride,
                                              gu8p __restrict__ row, gu8p __restrict__ scalerow,
                                              int dim, int base, int lane,
                                              float* __restrict__ a0) {
    for (int i = base + lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        unsigned int n = (i & 1) ? (unsigned int)(b >> 4) : (unsigned int)(b & 0x0F);
        float s = e8m0f(scalerow[i >> F4_GROUP_SHIFT]);
#pragma unroll
        for (int t = 0; t < R; ++t) a0[t] += v[(size_t)t * v_stride + i] * e2m1f(n) * s;
    }
}

template <int R>
__device__ __forceinline__ void dot_f4_wave_r(const float* __restrict__ v, int v_stride,
                                              gu8p __restrict__ row,
                                              gu8p __restrict__ scalerow,
                                              int dim, int lane, float* __restrict__ out) {
    float a0[R], a1[R];
#pragma unroll
    for (int t = 0; t < R; ++t) { a0[t] = 0.0f; a1[t] = 0.0f; }
    int base = f4_dword_pass<R>(v, v_stride, row, scalerow, dim, lane, a0, a1);
    f4_tail_accum<R>(v, v_stride, row, scalerow, dim, base, lane, a0);
#pragma unroll
    for (int t = 0; t < R; ++t) out[t] = wave_sum(a0[t] + a1[t]);
}

#include <hip/hip_fp16.h>

// VQ-int3 codebook parameters — MUST match quant.rs (VQ_DIM/VQ_K/VQ_INDEX_BITS/VQ_GROUP).
#define VQ_DIM 4
#define VQ_K 4096
#define VQ_INDEX_BITS 12
#define VQ_GROUP 64
#define VQ_SUBS_PER_GROUP (VQ_GROUP / VQ_DIM)  // subvectors sharing one bf16 scale

// Wave-cooperative VQ-int3 dot for one output row, result on lane 0. Matches
// quant.rs::matvec_vq: each VQ_DIM-subvector t reads a packed 12-bit codebook
// index, dots x[t*VQ_DIM..] with codebook[idx*VQ_DIM..], and scales by the
// subvector's bf16 group scale (group = t / VQ_SUBS_PER_GROUP). Lane `l` owns
// subvectors t ≡ l (mod WAVE). The two-byte index read is in-bounds because
// i_dim is a multiple of 8 (nsub even; the last odd subvector's high byte is the
// row's last index byte — see quant.rs::get_idx / vq_row_bytes).
//
// The codebook is fp16: the random idx→cb[idx] gather is the latency-bound step,
// and at 8B/entry (VQ_K·VQ_DIM·2 = 32KB) the whole codebook fits in the 32KB L1,
// so the gather is an L1 hit instead of L2 (f32 was 64KB, L2-resident). fp16 keeps
// x in f32 for the products; its 10-bit mantissa on the centroids clears the
// oracle tol (bf16's 8 does not — see math::f32_to_f16).
// R INPUT ROWS against ONE weight row. `out[r]` (lane 0) is the dot of row `r` of `x`
// — stride `x_stride` — with the decoded weight row.
//
// This is where a speculative verify pass gets cheap. The weight side is the expensive
// half by a wide margin: an expert block is 15.34 MB and this kernel touches every byte
// of it, while a row of `x` is 24 KB. Decoding the row once and dotting it against R
// inputs turns R tokens into ONE read of the weights, so the pass that VERIFIES a draft
// costs barely more than the pass that would have produced a single token — which is the
// entire economic case for MTP on a fetch-bound engine.
//
// R is a template parameter so `acc[]` stays in registers and the inner loop unrolls; a
// runtime count would index an array dynamically and spill it to scratch.
//
// R=1 is BIT-IDENTICAL to the single-row form this replaced — the same four products in
// the same order, the same one `bf16f` scale multiply, the same `wave_sum`. That is load
// bearing: it means every existing oracle test and the perplexity gate still bind the
// batched kernel's R=1 path with no re-baselining.
template <int R>
__device__ __forceinline__ void dot_vq_wave_r(const float* __restrict__ x, int x_stride,
                                              const unsigned char* __restrict__ idxrow,
                                              const unsigned short* __restrict__ scalerow,
                                              const __half* __restrict__ cb, int i_dim, int lane,
                                              float* __restrict__ out) {
    int nsub = i_dim / VQ_DIM;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int t = lane; t < nsub; t += WAVE) {
        int bitpos = t * VQ_INDEX_BITS;
        int byte = bitpos >> 3;
        int shift = bitpos & 7;
        unsigned int raw = (unsigned int)idxrow[byte] | ((unsigned int)idxrow[byte + 1] << 8);
        int idx = (int)((raw >> shift) & 0xFFFu);
        // 8B fp16 gather (two __half2) for the VQ_DIM=4 subvector; products in f32,
        // same 4 terms in the same order as the f32 path.
        const __half* c = cb + (size_t)idx * VQ_DIM;
        float2 c01 = __half22float2(*(const __half2*)c);
        float2 c23 = __half22float2(*(const __half2*)(c + 2));
        float s = bf16f(scalerow[t / VQ_SUBS_PER_GROUP]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float4 xv = *(const float4*)(x + (size_t)r * x_stride + (size_t)t * VQ_DIM);
            float dot = xv.x * c01.x + xv.y * c01.y + xv.z * c23.x + xv.w * c23.y;
            acc[r] += s * dot;
        }
    }
    // Uniform across the wave: R is a compile-time constant, so every lane runs every
    // `wave_sum`. A runtime bound here would be a partial-wave reduction.
#pragma unroll
    for (int r = 0; r < R; ++r) out[r] = wave_sum(acc[r]);
}

__device__ __forceinline__ float dot_vq_wave(const float* __restrict__ x,
                                             const unsigned char* __restrict__ idxrow,
                                             const unsigned short* __restrict__ scalerow,
                                             const __half* __restrict__ cb, int i_dim, int lane) {
    float out;
    dot_vq_wave_r<1>(x, 0, idxrow, scalerow, cb, i_dim, lane, &out);
    return out;
}
