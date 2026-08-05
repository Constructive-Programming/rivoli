// Device helpers shared by the separately-compiled kernel translation units. The
// bf16/e4m3 pair MUST stay bit-exact with math.rs (the CPU oracle the kernel tests
// compare against), so one definition is a correctness property. int4-free.
#pragma once

#include <hip/hip_runtime.h>

// ponytail: WAVE=32 hardcoded for gfx1151 (RDNA3.5, native wave32); __shfl_down
// width=32 keeps it correct on a wave64 part too (reduces within 32-lane groups).
#define WAVE 32
#define ROWS_PER_BLOCK 8  // block = 256 threads = 8 waves → 8 output rows/block

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

// Sum a wave's partials into lane 0 (fixed __shfl_down order → deterministic).
__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}

// Max across a wave, result on EVERY lane. A butterfly rather than `wave_sum`'s ladder,
// which is legal here and not there: `fmaxf` is EXACTLY associative, so the two orders
// agree bit-for-bit, and the caller needs the result on every lane anyway.
//
// REQUIRES A FULL WAVE. `__shfl_xor` against an inactive lane is undefined, and unlike
// `wave_sum` — whose ladder leaves lane 0 right regardless — this promises every lane. Its
// one caller satisfies it: `act_quant_f8`'s early return is wave-uniform, because `b` is
// derived from `threadIdx.x / WAVE`.
__device__ __forceinline__ float wave_max(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WAVE));
    return v;
}

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
// the LUT + a scalar tail. Shared by `dot_fp8_wave` (one wave, stride WAVE) and
// `gemv_fp8_splitk` (all block threads, split-K over one row). The caller reduces.
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
        float s = scalerow[i0 >> bsh];
        float l0 = lut[(unsigned char)p];
        float l1 = lut[(unsigned char)(p >> 8)];
        float l2 = lut[(unsigned char)(p >> 16)];
        float l3 = lut[(unsigned char)(p >> 24)];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float* xr = x + (size_t)r * x_stride;
            acc[r] += s * (xr[i0] * l0 + xr[i0 + 1] * l1 + xr[i0 + 2] * l2 + xr[i0 + 3] * l3);
        }
    }
    for (int i = (n4 << 2) + start; i < i_dim; i += stride) {
        // `x * l * s`, NOT `x * (l*s)`. Folding the two weight-side factors first would be
        // one fewer multiply per row and a DIFFERENT number — fp multiplication does not
        // associate — which is exactly the kind of drift the bit-identity test exists to
        // refuse. The loads are hoisted; the arithmetic is not touched.
        float l = lut[wrow[i]];
        float s = scalerow[i >> bsh];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += x[(size_t)r * x_stride + i] * l * s;
    }
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
// R token rows against ONE read of the weight row — the nibble decode and the group scale
// amortise over the rows, which is the whole point of batching (the weight read is ~92% of
// an expert launch). `v_stride` is the row-minor stride of `v`.
//
// R=1 is BIT-IDENTICAL to `dot_i4_wave` below, and that is a constraint on how this is
// written, not a hope: the fast path's `s * (x·n + …)` grouping and the scalar tail's
// left-to-right `v[i] * (n−8) * scale` are both reproduced exactly. Hoisting the nibbles
// is safe (same values, same order); hoisting `(n−8) * scale` out of the tail would NOT be
// — it re-associates the product. See tests/kernel.rs.
template <int R>
__device__ __forceinline__ void dot_i4_wave_r(const float* __restrict__ v, int v_stride,
                                              const unsigned char* __restrict__ row,
                                              const float* __restrict__ scalerow,
                                              int dim, int lane, float* __restrict__ out) {
    float a0[R], a1[R];
#pragma unroll
    for (int t = 0; t < R; ++t) { a0[t] = 0.0f; a1[t] = 0.0f; }
    int base = 0;
    if ((((size_t)row) & 3u) == 0) {
        const unsigned int* rw = (const unsigned int*)row;
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
    }
    for (int i = base + lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
#pragma unroll
        for (int t = 0; t < R; ++t)
            a0[t] += v[(size_t)t * v_stride + i] * (float)(n - 8) * scalerow[i >> I4_GROUP_SHIFT];
    }
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

// One bf16 STORE: round an f32 through bf16 and back.
//
// V4's activations live in bf16 between operations, so every `.to(dtype)`, every
// `RMSNorm.forward` return and every `apply_rotary_emb` copy-back is one of these. Dropping
// one leaves values that are fluent and slightly too precise, which no magnitude check sees
// (`v4oracle::Defect::NoBf16Rounding`).
//
// Here rather than per-kernel because it had been written THREE times — `mla.hip::v4_rbf16`,
// `v4compress.hip::v4c_rbf16` and a third copy in `v4indexer.hip`, each `static` to its
// translation unit and so unreachable from the others. `kernels/` is not watched by
// `build.rs`'s jscpd gate, so nothing mechanical would ever have found them. `v4indexer.hip`
// now calls this one; collapsing the other two is the plan's requirement 11 and needs
// `mla.hip` edits no S2 agent was permitted to make.
__device__ __forceinline__ float rbf16(float x) { return bf16f(f2bf16(x)); }

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
// e8m0_decode). The oracle is the reference; these two are independently written from the
// same format definition, which is the point — see v4oracle/numerics.rs's module doc.
#define F4_GROUP 32
#define F4_GROUP_SHIFT 5
static_assert((1 << F4_GROUP_SHIFT) == F4_GROUP, "F4_GROUP_SHIFT must be log2(F4_GROUP)");
static_assert(F4_GROUP % 8 == 0, "the dword fast path's 8 columns must not straddle a group");

// e2m1 nibble → f32. 1 sign / 2 exp (bias 1) / 1 mantissa; no infinities, no NaN, so a
// bare arithmetic decode is TOTAL — every one of the 16 codes is a finite value.
//
// Arithmetic rather than a 16-entry lookup on purpose: the fast path decodes 8 nibbles
// from one dword and each is consumed once, so a table would be a divergent index into
// either scratch or LDS for values that cost two selects to compute. `(e - 1) & 3` keeps
// the shift in range on the e == 0 arm the ternary never takes.
__device__ __forceinline__ float e2m1f(unsigned int nib) {
    unsigned int e = (nib >> 1) & 3u, m = nib & 1u;
    float mag = e ? (1.0f + 0.5f * (float)m) * (float)(1u << ((e - 1u) & 3u))
                  : 0.5f * (float)m;
    return (nib & 8u) ? -mag : mag;
}

// e8m0 scale byte → f32, `2^(b − 127)` (matches v4oracle::numerics::e8m0_decode).
//
// Both endpoints are spelled out rather than left to the exponent-field shift. 0xff is the
// format's NaN, which is the right DECODE — reading it as `2^128` would be worse — but it
// does not poison anything downstream: `moe_fixed`'s saturating clamp launders a NaN into a
// finite ±2^14 (`fminf`/`fmaxf` return the non-NaN operand). So a 0xff scale byte must be
// REJECTED AT LOAD, in S3, alongside the `tid2eid` range check `moe_gate_v4` names.
// b == 0 is 2^-127, which is BELOW f32's smallest normal, so `b << 23` would silently hand
// back +0 — a whole 32-weight group zeroed with no error anywhere.
//
// These bytes come off the CHECKPOINT (`<proj>.scale`, copied verbatim by the `.f4`
// repack), not from `fast_round_scale`, so "our quantizer cannot emit them" is not an
// argument that applies here: the FORMAT permits both, and what the converter happened to
// produce is not a property this decoder gets to assume.
__device__ __forceinline__ float e8m0f(unsigned char b) {
    if (b == 0xff) return __int_as_float(0x7fc00000);
    if (b == 0) return __uint_as_float(1u << 22);
    return __uint_as_float((unsigned int)b << 23);
}

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
template <int R>
__device__ __forceinline__ void dot_f4_wave_r(const float* __restrict__ v, int v_stride,
                                              const unsigned char* __restrict__ row,
                                              const unsigned char* __restrict__ scalerow,
                                              int dim, int lane, float* __restrict__ out) {
    float a0[R], a1[R];
#pragma unroll
    for (int t = 0; t < R; ++t) { a0[t] = 0.0f; a1[t] = 0.0f; }
    int base = 0;
    // Predicated on the WEIGHT row's 4-byte alignment, which is what the dword read needs.
    // The `float4` loads below are on the ACTIVATION and are NOT checked — a 16-byte-aligned
    // `v` is an unchecked caller obligation (`launch_moe_expert_range_f4`'s `# Safety`), and
    // a misaligned one faults rather than falling back to the tail.
    if ((((size_t)row) & 3u) == 0) {
        const unsigned int* rw = (const unsigned int*)row;
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
    }
    for (int i = base + lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        unsigned int n = (i & 1) ? (unsigned int)(b >> 4) : (unsigned int)(b & 0x0F);
        float s = e8m0f(scalerow[i >> F4_GROUP_SHIFT]);
#pragma unroll
        for (int t = 0; t < R; ++t) a0[t] += v[(size_t)t * v_stride + i] * e2m1f(n) * s;
    }
#pragma unroll
    for (int t = 0; t < R; ++t) out[t] = wave_sum(a0[t] + a1[t]);
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
// trip, in `tests/v4_kernel.rs::act_quant_f8_is_bit_identical_to_the_oracle`, over all 254
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
// The indexer's simulator, not the expert weights' codec. `dot_f4_wave_r` above DECODES
// packed e2m1 nibbles that came off the checkpoint; this pair ENCODES an activation and
// decodes it straight back, which is how `Indexer.forward` simulates fp4 in a bf16 tensor
// (model.py:376 and :422 — both `q` and the indexer's compressed kv go through it).
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
