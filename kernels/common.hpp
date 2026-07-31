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
// is what lets the per-expert partial rows, the `moe_reduce` over them, and the stream
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
    // TWO KNOWN GAPS REMAIN, both recorded in docs/PERF.md #4: the Vulkan twin
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
