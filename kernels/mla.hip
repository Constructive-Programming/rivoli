// MLA absorb + value kernels — the two ops through kv_b the generic GEMV can't
// express, head-batched into one launch each. kv_b is fp8-e4m3 now (was int4):
// [H·(nope+vh) rows, kvl cols], block-scaled (weight_scale_inv, `block`×`block`
// tiles), row stride = kvl bytes. Head `h` owns rows [h·(nope+vh) .. +(nope+vh)):
// the first `nope` are the absorb rows, the next `vh` the value rows. Block scale
// for element (r,c) is scale[(r/block)·sc_cols + c/block], sc_cols = ⌈kvl/block⌉.
// All pointers device-resident; launch-only. Return codes: 0 ok, POSITIVE arg
// guard, NEGATIVE -hipError.

#include "common.hpp"

// qabs[head][i] = Σ_d q[head·qh + d] · e4m3(kv_b[rbase+d][i]) · scale[block(rbase+d,i)],
// rbase = head·(nope+vh), i∈[0,kvl), d∈[0,nope) — absorb q_nope through kv_b's nope
// rows into latent space. Transposed: output indexed by column i, summed over nope rows.
//
// One thread per (head, i-QUAD): four adjacent columns, four accumulators. A quad's
// columns are contiguous in kv_b, so the weight read is one `global_load_b32` = 128 B per
// wave, exactly one cache line; the old one-column-per-thread form issued
// `global_load_u8` = 32 B/wave, a QUARTER of a line, and that load width was the cost.
// Measured 1.40× — docs/measurement/perf-roadmap.md #4.
//
// BIT-IDENTICAL: each output still sums over `d` in ascending order with the same
// expression, so no quality gate is needed. DO NOT "improve" this to the wave-per-row
// shape `mla_value` uses — that reassociates, and then it does need one.
//
// R TOKEN ROWS through one read of kv_b (14.7 MB/layer, 16% of the attention bytes): the
// `q` rows are `H·qh` apart, the `qabs` rows `H·kvl`, and the four column accumulators
// become `R`×four in registers. Row 0 is bit-identical to R=1 — the `d` loop, the
// expression and the summation order are untouched, only replicated.
template <int R>
__device__ __forceinline__ void mla_absorb_fp8_impl(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl, int block,
    float* __restrict__ qabs) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    int nquad = absorb_nquad(kvl);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * nquad) return;
    int head = (int)(idx / nquad);
    int i0 = (int)(idx % nquad) << 2;
    int sc_cols = (kvl + block - 1) / block;
    // Shift, NOT divide: `row` is a size_t, so `row / block` was a 64-bit unsigned
    // division that LLVM cannot fold to a magic multiply — it emitted an inline
    // Newton-Raphson reciprocal inside the d-loop, worth 72.0 → 36.5 µs on its own.
    // The launcher rejects a non-power-of-two tile. Bit-identical: `row` is unsigned
    // and the column index non-negative, so the index is the same.
    // Evidence in docs/measurement/benchmarks.md, "Read the ISA before you book the device".
    int bsh = blk_shift(block);
    size_t rbase = (size_t)head * (nope + vh);
    const float* qh_ptr = q + (size_t)head * qh;
    const unsigned char* wcol = kvb + rbase * (size_t)kvl + i0;
    float* out = qabs + (size_t)head * kvl + i0;
    // ONE fallback branch, two independent triggers. `kvl & 3` leaves rows unaligned for
    // the dword load (rows are `kvl` bytes apart); `block < 4` puts a quad's columns in
    // different scale tiles. Rejecting them in the launcher instead is not an option: the
    // Vulkan absorb launcher accepts both (src/vk.rs carries NO `kvl % 4` guard on absorb,
    // only on value, deliberately) and the backends must span one argument domain.
    //
    // THE COST IS A SECOND COPY OF THE DEQUANT. This body and the quad body below are two
    // expressions of the same math, and the GLM shape only ever runs the second — so a
    // change to either (lut indexing, scale layout, an fp8 variant) must be made twice,
    // and only the non-GLM oracle shapes in tests/kernel.rs would catch a divergence.
    if ((kvl & 3) || block < 4) {
        // Not unrolled: four copies of the d-loop tripled the kernel's static size
        // (698 → 1078 instructions) for a branch the engine never takes. The measured
        // build carried this pragma and the hot loop below is identical either way.
#pragma unroll 1
        for (int k = 0; k < 4 && i0 + k < kvl; ++k) {
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            for (int d = 0; d < nope; ++d) {
                size_t row = rbase + d;
                float ws = lut[wcol[(size_t)d * kvl + k]]
                           * kvb_scale[(row >> bsh) * sc_cols + ((i0 + k) >> bsh)];
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += qh_ptr[(size_t)r * H * qh + d] * ws;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) out[(size_t)r * H * kvl + k] = acc[r];
        }
        return;
    }
    // One scale load per iteration covers all four columns: past the `if` above the quad
    // sits inside one tile, and its column index does not depend on `d`.
    int sc_col = i0 >> bsh;
    // `w4 += stride4` rather than `w4[d * stride4]` — the latter cost a 64-bit
    // multiply-add in the loop body.
    const unsigned int* w4 = (const unsigned int*)wcol;
    int stride4 = kvl >> 2;
    float a[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) { a[r][0] = a[r][1] = a[r][2] = a[r][3] = 0.0f; }
    for (int d = 0; d < nope; ++d, w4 += stride4) {
        unsigned int p = *w4;
        float s = kvb_scale[((rbase + d) >> bsh) * sc_cols + sc_col];
        // `lut[..] * s` hoisted out of the row loop, exactly as the R=1 form wrote it —
        // the parenthesisation `qd * (lut * s)` is preserved, so row 0 is bit-identical.
        float l0 = lut[(unsigned char)p] * s;
        float l1 = lut[(unsigned char)(p >> 8)] * s;
        float l2 = lut[(unsigned char)(p >> 16)] * s;
        float l3 = lut[(unsigned char)(p >> 24)] * s;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float qd = qh_ptr[(size_t)r * H * qh + d];
            a[r][0] += qd * l0;
            a[r][1] += qd * l1;
            a[r][2] += qd * l2;
            a[r][3] += qd * l3;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float* o = out + (size_t)r * H * kvl;
        o[0] = a[r][0];
        o[1] = a[r][1];
        o[2] = a[r][2];
        o[3] = a[r][3];
    }
}

extern "C" __global__ void mla_absorb_fp8(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl, int block,
    float* __restrict__ qabs) {
    mla_absorb_fp8_impl<1>(q, kvb, kvb_scale, H, qh, nope, vh, kvl, block, qabs);
}

extern "C" __global__ void mla_absorb_fp8_r2(
    const float* __restrict__ q, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int qh, int nope, int vh, int kvl, int block,
    float* __restrict__ qabs) {
    mla_absorb_fp8_impl<2>(q, kvb, kvb_scale, H, qh, nope, vh, kvl, block, qabs);
}

// ctx[head][j] = Σ_i clat[head][i] · e4m3(kv_b[rbase+nope+j][i]) · scale[block(row,i)],
// j∈[0,vh), i∈[0,kvl) — project the attention-weighted latent through kv_b's value
// rows. Per-head-row GEMV; one WAVE per output row (dot_fp8_wave folds the block
// scale inside the dot, since it varies along i).
//
// R token rows through one read of kv_b — `clat` rows are `H·kvl` apart, `ctx` rows
// `H·vh`. Row 0 is bit-identical: `dot_fp8_wave_r<1>` IS `dot_fp8_wave`.
template <int R>
__device__ __forceinline__ void mla_value_fp8_impl(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl, int block,
    float* __restrict__ ctx) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)H * vh) return;
    int head = (int)(r / vh);
    int j = (int)(r % vh);
    int sc_cols = (kvl + block - 1) / block;
    size_t row = (size_t)head * (nope + vh) + nope + j;
    // Shift for the same reason as mla_absorb_fp8 — `row` is a size_t, so `/ block` was
    // a 64-bit divide. It is once per THREAD here, not once per iteration, and it was
    // still worth 20% (33.7 → 27.0 us, 249 → 310 GB/s): with H*vh = 16384 threads and a
    // 34 us kernel, one 64-bit division per thread is not a rounding error. An earlier
    // version of this comment called it housekeeping; the bench disagreed.
    float acc[R];
    dot_fp8_wave_r<R>(clat + (size_t)head * kvl, H * kvl, kvb + row * (size_t)kvl,
                      kvb_scale + (row >> blk_shift(block)) * sc_cols, kvl, block, lane, lut,
                      acc);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) ctx[(size_t)r * H * vh + (size_t)head * vh + j] = acc[r];
    }
}

extern "C" __global__ void mla_value_fp8(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl, int block,
    float* __restrict__ ctx) {
    mla_value_fp8_impl<1>(clat, kvb, kvb_scale, H, nope, vh, kvl, block, ctx);
}

extern "C" __global__ void mla_value_fp8_r2(
    const float* __restrict__ clat, const unsigned char* __restrict__ kvb,
    const float* __restrict__ kvb_scale, int H, int nope, int vh, int kvl, int block,
    float* __restrict__ ctx) {
    mla_value_fp8_impl<2>(clat, kvb, kvb_scale, H, nope, vh, kvl, block, ctx);
}

extern "C" int rivoli_mla_absorb_fp8(
    const float* q, const unsigned char* kvb, const float* kvb_scale,
    int H, int qh, int nope, int vh, int kvl, int block, int nrow, float* qabs) {
    if (H <= 0 || qh <= 0 || nope <= 0 || vh <= 0 || kvl <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    // The row count is a TEMPLATE parameter (accumulators live in registers), so only the
    // instantiated widths exist. Same contract as gemv_fp8 / moe_expert_range.
    if (nrow != 1 && nrow != 2) return 1004;
    size_t total = (size_t)H * (size_t)absorb_nquad(kvl); // one thread per (head, i-quad)
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? mla_absorb_fp8_r2 : mla_absorb_fp8,
                       dim3((unsigned)((total + 255) / 256)), dim3(256), 0, 0,
                       q, kvb, kvb_scale, H, qh, nope, vh, kvl, block, qabs);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_mla_value_fp8(
    const float* clat, const unsigned char* kvb, const float* kvb_scale,
    int H, int nope, int vh, int kvl, int block, int nrow, float* ctx) {
    if (H <= 0 || nope <= 0 || vh <= 0 || kvl <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    if (nrow != 1 && nrow != 2) return 1004;  // see rivoli_mla_absorb_fp8
    // `dot_fp8_wave` reads weights as 32-bit words and rows are `kvl` bytes apart, so a
    // ragged kvl leaves 3 rows in 4 misaligned for the dword load. The Vulkan twin has
    // rejected this since it was written (src/vk.rs::launch_mla_value_fp8) and this
    // launcher did not — a parity gap found when the absorb restructure added a kvl=37
    // oracle shape, which is legal for absorb (it reads single bytes) and drove
    // mla_value straight into the misaligned load. Absorb deliberately has NO such
    // guard; the asymmetry is the same one src/vk.rs documents.
    if (kvl & 3) return 1002;
    size_t total = (size_t)H * vh;
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? mla_value_fp8_r2 : mla_value_fp8,
                       dim3((unsigned)((total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, 0,
                       clat, kvb, kvb_scale, H, nope, vh, kvl, block, ctx);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// ═══════════════════════════════════════════════════════════════════════════════════
// DeepSeek-V4-Flash attention front half — S2b of docs/investigations/v4-flash-port.md
// ═══════════════════════════════════════════════════════════════════════════════════
//
// V4 is MQA, not MLA: ONE 512-wide `wkv` entry is both the key and the value for all 64
// heads, and its last `rope_head_dim` dims carry the position. So none of the `mla_*`
// kernels above apply — there is no per-head K/V, no nope/rope tensor split, and no
// latent cache. What is here is `Attention.forward`'s elementwise and GEMV work
// (inference/model.py:490-548); the softmax core is `gather_attn_shared_kv` in attn.hip.
//
// EVERY defect available in this path is silent-wrong — a missing QK-norm, RoPE on the
// wrong pairing, the partial fp8 quantization applied to the whole entry. The gate is
// `src/v4oracle/`'s goldens, so each kernel below is written to reproduce the VALUE the
// reference produces, not merely its shape; where a value cannot be reproduced the
// reason is stated in place.
//
// FP CONTRACTION IS OFF for this section. `-ffp-contract=fast` would fuse `a*c - b*s`
// into an FMA, which is a different VALUE, not a different summation order — and the
// oracle's stated contract (src/v4oracle/forward.rs, "Fidelity") is that it reproduces
// every value change and declines only re-association. Leaving contraction on would put
// an unexplained delta into every comparison S2b makes, exactly as an fp8 `wo_a` would
// (the plan's DECIDED note of 2026-08-05). These kernels are elementwise and
// memory-bound, so the pragma costs nothing measurable; the fp8 dot below is
// deliberately outside it — `dot_fp8_wave` is defined in `common.hpp`, included ABOVE
// this line, so it never enters the pragma's lexical scope; that is the mechanism, not
// an accident of where the GEMV happens to sit in this file. It also already
// re-associates, so contraction is not the dominant term there.
//
// The cost has NOT been measured. These kernels are elementwise and bound by the bytes
// they move, so the expectation is that it is free — but that is an expectation, and S4
// is where it gets priced with the oracle available to bound the error either way.
//
// VERIFIED IN THE ISA, not assumed: at `--offload-arch=gfx1151 -O3`, `rope_adjacent`'s
// `a*c - b*s` emits 4 `v_mul_f32` and ZERO fma-class instructions, while
// `mla_absorb_fp8` above — same file, same flags, outside this pragma — emits 14. A
// pragma in the wrong place would have been invisible otherwise, since the wrong answer
// is still a plausible float.
//
// **"fma-class" here means `v_fma_f32` and `v_fmac_f32` ONLY** — float FMAs, the thing
// `contract(off)` governs. Re-measured 2026-08-05 and both numbers stand, but the
// distinction was not stated and it matters: a `v_fma|v_mac|v_mad` grep reports 1 for
// `rope_adjacent` and 32 for `mla_absorb_fp8`, because `v_mad_u64_u32`/`v_mad_i64_i32` are
// ADDRESS arithmetic that contraction neither does nor should touch. On a kernel with
// nine of those, a genuine one-instruction regression is a 10% move in a number nobody
// looks twice at — found by the indexer agent on `index_score_blocks`, where the pragma
// moved 1 `v_fmac_f32` to 0 while the naive count read 10 → 9. Count narrowly, and
// compare a DELTA rather than an absolute.
//
// (Counts elsewhere in this section are NOT evidence of contraction: `1.0f/sqrtf(x)`
// lowers to `v_rcp` plus FMA Newton steps, which is the divide's own implementation and
// not a fusion of anything written here.)
#pragma clang fp contract(off)


// `kernel.py::act_quant(x, block, scale_fmt="ue8m0", inplace=True)` — the fused
// quantize-then-dequantize that SIMULATES fp8 inside a bf16 tensor. Two call sites with
// different arguments, which is the entire reason this is one kernel and not two:
//   * every quantized `Linear` runs it over the whole activation row at block 128
//     (model.py:120/123), so the goldens are "what fp8 arithmetic produces", not
//     "f32 reference values";
//   * the KV entry runs it over dims [0, n) at block **64** with the RoPE tail left
//     alone (model.py:512), which is why `n` is separate from `row_stride`.
//
// The result needs no bf16 rounding afterwards: an e4m3 value has at most 4 significant
// bits and the scale is a bare power of two, so `e4m3 * s` lands exactly in bf16's 8 as
// long as the product stays normal — which it does here, since the 1e-4 amax floor puts
// the smallest scale this can produce at 2^-22 against a bf16 subnormal floor of 2^-133.
// The reference's `inplace` cast back to bf16 is therefore a no-op, and so is the
// oracle's `round_bf16` after `act_quant_inplace`. Checked over every e4m3 code against
// scale codes 40..=200 by
// `tests/v4_attn_host.rs::fp8_times_a_power_of_two_is_exact_in_bf16_over_the_range_the_checkpoint_uses`,
// which also exhibits the tiny-scale boundary where it stops holding.
//
// One thread per (row, block): `block` is 64 or 128, the amax scan and the write-back
// are serial over it, and the whole op moves ~2 KB per token. Parallelising inside a
// block would buy nothing and would re-associate an absmax that is exact today.
//
// `src`/`dst` rather than one `x` since M10 (docs/investigations/v4-decode-decomposition.md):
// the qkv chain ran `memcpy(dst, src)` + `act_quant(dst)` twice per layer purely so the
// SOURCE survived for its other readers (`x` for the compressor, `qr` for the reference's
// indexer — attn.rs `Scratch` names them). Reading `src` and writing `dst` satisfies the
// same contract in ONE launch: the values quantized are the bytes the copy would have
// delivered, and the per-block arithmetic below is untouched. `src == dst` is the
// reference's `inplace=True` and reproduces the pre-M10 body exactly — the amax scan
// completes before the first write either way. NO `__restrict__` on either pointer for
// that reason: they alias at every in-place call site.
extern "C" __global__ void act_quant_f8_prefix(const float* src, float* dst, int rows,
                                               int row_stride, int n, int block) {
    int nb = n / block;
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)rows * (size_t)nb) return;
    size_t off = (size_t)(id / nb) * row_stride + (size_t)(id % nb) * block;
    const float* seg = src + off;
    float* out = dst + off;
    // `T.max(amax_local[i], 1e-4)`: the floor keeps an all-zero block from producing a
    // zero scale and a division by zero. Seeding the scan with it is the same value.
    float amax = ACT_QUANT_AMAX_FLOOR;
    for (int i = 0; i < block; ++i) amax = fmaxf(amax, fabsf(seg[i]));
    float s = fast_round_scale(amax, FP8_MAX_INV);
    // `act_quant_roundtrip`, not a local clamp-and-encode. This loop body WAS a fourth
    // hand copy of `common.hpp`'s — the same divide, the same deliberate ternaries (which
    // exist so a NaN PROPAGATES rather than being laundered into -448 by `fminf`/`fmaxf`),
    // the same encode — sitting eight lines under a note claiming the duplication was
    // lifted. Found by review, 2026-08-05.
    for (int i = 0; i < block; ++i) out[i] = act_quant_roundtrip(seg[i], s);
}

// `RMSNorm.forward` (model.py:197-202) — `q_norm` and `kv_norm`. FP32 internals: the
// reference opens with an explicit `x = x.float()`, so the statistic, the epsilon and
// the reciprocal square root are all f32 no matter that `x` arrived as bf16. The
// learned weight multiplies AFTER the scale (`(self.weight * x)`, not `(w*v)*rs`), and
// the store is bf16.
//
// One block per row. `1.0f/sqrtf` and not `rsqrtf`: `rsqrtf` is a hardware
// approximation, and this factor multiplies the whole row, so its error is not a
// rounding difference — it is a systematic scale on every downstream logit.
extern "C" __global__ void rmsnorm_batch(float* __restrict__ x, const float* __restrict__ w,
                                         int d, float eps) {
    __shared__ float red[256];
    float* row = x + (size_t)blockIdx.x * d;
    float p = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) p += row[i] * row[i];
    float rs = 1.0f / sqrtf(block_sum_lds(p, red) / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) row[i] = rbf16(w[i] * (row[i] * rs));
}

// The QK-norm, model.py:504: `q *= rsqrt(q.square().mean(-1) + eps)` over `head_dim`,
// applied after the unflatten to (heads, head_dim), with **NO learnable weight**.
//
// TWO things here are invisible to the gate and had to be read off the reference:
//
// 1. It runs BEFORE the RoPE (model.py:504 then :505). The oracle CANNOT see the order
//    — `apply_rotary_emb` rotates adjacent pairs, so it preserves `mean(q^2)`, and a
//    scalar commutes with a rotation; `Defect::QkNormAfterRope` is demoted out of the
//    defect matrix for exactly that reason (tests/v4_oracle.rs::
//    qk_norm_order_is_a_rounding_difference_not_an_arithmetic_one). The launch order in
//    src/attn.rs is therefore justified by model.py, not by a green test.
// 2. It is computed in **BF16**, unlike `rmsnorm_batch`. `RMSNorm.forward` opens with
//    `x = x.float()`; model.py:504 does not, and `q` is bf16 out of `fp8_gemm`, so the
//    square, the mean, the `+ eps` and the `rsqrt` are all bf16-valued. Keeping the
//    statistic in f32 leaves the factor up to 2^-9 (~0.2%) off, identically for every
//    dim of a head, which scales that head's whole logit row.
//
// INHERITED INFERENCE, and the sharpest one in this file: torch reduces a bf16 `.mean`
// through an f32 accumulator (`acc_type`) with a bf16 result. `src/v4oracle/forward.rs`
// marks the same reading INFERRED, and this kernel copies it — so if it is wrong, both
// are wrong the same way and no golden can tell. That is a shared blind spot created by
// agreeing with the instrument, and it is stated here because nothing else in the tree
// states it.
extern "C" __global__ void qk_norm(float* __restrict__ q, int d, float eps) {
    __shared__ float red[256];
    float* row = q + (size_t)blockIdx.x * d;
    float p = 0.0f;
    // Per-element squares are bf16; the accumulator is f32 (the inference above). The
    // round-trip also blocks FMA contraction of the square into the accumulate.
    for (int i = threadIdx.x; i < d; i += blockDim.x) p += rbf16(row[i] * row[i]);
    float var = rbf16(block_sum_lds(p, red) / (float)d);
    float rs = rbf16(1.0f / sqrtf(rbf16(var + eps)));
    for (int i = threadIdx.x; i < d; i += blockDim.x) row[i] = rbf16(row[i] * rs);
}

// `apply_rotary_emb` (model.py:238-250) over the LAST `rd` dims of each row.
//
// **ADJACENT PAIRS `(2i, 2i+1)`**, because the reference rotates through
// `torch.view_as_complex`. This is NOT what `linalg.hip::rope_interleave` does: that one
// reads the adjacent pair and WRITES the half-split slots `(i, half+i)`, which is the
// GLM/HF convention and carries a permutation. Reusing it here would be
// `Defect::RopeHalfSplit`, which the oracle enumerates as a deliberate breakage —
// v4-flash-port.md calls this the single most likely silent-wrong in S2b's scope, and
// both rotations produce fluent text.
//
// The angles come from a precomputed `(cos, sin)` table rather than being computed here,
// because the per-layer YaRN selection lives in the table (model.py:481-488) and a
// kernel that computed `theta^(-2i/rd)` itself would have to re-derive the
// interpolation. Row `r` uses position `pos0 + r/rows_per_pos`: `rows_per_pos` is
// `n_heads` for q and for the output de-rotation (whose rows are per (token, head)) and
// 1 for the KV entry (one row per token).
//
// `inverse` conjugates the rotation — `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`
// at model.py:539, the output de-rotation. Omitting it, or applying the forward
// rotation there, are both silent and both in the defect matrix.
//
// One thread owns a whole pair, so the read-before-write is per-thread and needs no
// barrier. The trailing sweep is `round_bf16(row)` over the WHOLE row, as the oracle
// does: the untouched dims are already bf16 so it is a no-op on them, and doing it
// unconditionally removes a standing assumption about what the previous kernel left.
extern "C" __global__ void rope_adjacent(float* __restrict__ x, const float* __restrict__ tbl,
                                         int row_len, int rd, int pos0, int rows_per_pos,
                                         int inverse) {
    float* row = x + (size_t)blockIdx.x * row_len;
    float* seg = row + (row_len - rd);
    const float* t = tbl + (size_t)(pos0 + (int)blockIdx.x / rows_per_pos) * rd;
    for (int i = threadIdx.x; i < rd / 2; i += blockDim.x) {
        float c = t[2 * i], s = inverse ? -t[2 * i + 1] : t[2 * i + 1];
        float a = seg[2 * i], b = seg[2 * i + 1];
        seg[2 * i] = a * c - b * s;
        seg[2 * i + 1] = a * s + b * c;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) row[i] = rbf16(row[i]);
}

// **Stores bf16-rounded**, unlike `linalg.hip::gemv_fp8`, which stores raw f32. That is the
// first thing a caller has to know and the second thing the name says, so it leads here:
// `rbf16(acc)` matches the reference's store and every caller in this engine wants it, but a
// caller reaching across from the GLM path would silently gain a rounding.
//
// USED BY: DeepSeek-V4-Flash-0731 only (`qkv`, `wo_a`, `wo_b`, the shared expert).
// GLM-5.2 uses `linalg.hip::gemv_fp8`.
//
// **`_grouped` NAMES THE MISUSE HAZARD.** The two fp8 GEMVs differ on two axes and only one
// of them corrupts quietly: confuse the store dtype and you get a rounding, confuse the
// input slicing and every output row reads the wrong activations. Hence the suffix went to
// `groups` and the store went in this paragraph. Renamed from `v4_gemv_fp8` 2026-08-09.
//
// The quantized `Linear` of model.py:114-127, and the grouped `wo_a` einsum of
// model.py:541-544, as ONE kernel — they differ only in which slice of the input a given
// output row reads.
//
// `out[r][j] = bf16( Σ_i x_row(r, j)[i] · e4m3(w[j][i]) · wscale[j/block][i/block] )`
//
// with `x_row(r, j) = x + (r·groups + j/(n_out/groups))·k`, i.e. the input is `m` rows of
// `groups` consecutive k-wide slices and output row `j` reads slice `j/(n_out/groups)`.
//   * `Linear`: `groups = 1`, so every output row reads the whole activation row.
//   * `wo_a`: `groups = o_groups`. `o.view(b, s, n_groups, -1)` over a head-major
//     flattening makes group `g` the CONTIGUOUS run of heads [g·(n_heads/n_groups), …) —
//     all `head_dim` dims of each — so the einsum needs no gather, only the right base.
//
// ONE parameter, not three. The first cut took `x_row_stride`, `group_rows` and
// `x_group_stride` separately, and at every call site they were `k`, `n_out/groups` and
// `k` — so the only reachable configurations were the two above, and the other
// combinations were an unchecked way to read past the input. The bound is now exactly
// `m·groups·k` and follows from the arguments instead of being a relation
// (`(n_out/group_rows − 1)·x_group_stride + k ≤ x_row_stride`) that nothing verified.
//
// NO activation quantization here. `Linear.forward` does its `act_quant` before calling
// the GEMM, so `act_quant_f8_prefix` is a separate launch on the caller's side; `wo_a` gets none
// at all, which is the whole content of the plan's DECIDED note of 2026-08-05 — `wo_a`
// ships fp8 on disk and `convert.py` dequantizes it to bf16 (convert.py:123-127), and an
// fp8 GEMV here would not match the oracle bit-for-bit, making a real bug and the format
// choice indistinguishable. Reading the fp8 bytes and dequantizing here is NOT a third
// option: over the scale range any weight tensor uses it produces the same bf16 values
// `convert.py` would have written (4 significant bits into 8), measured by
// `tests/v4_attn_host.rs::fp8_times_a_power_of_two_is_exact_in_bf16_over_the_range_the_checkpoint_uses`.
//
// One wave per output row, via `dot_fp8_wave` — whose summation order is NOT the
// oracle's sequential fold. That re-association is the floor on the tolerance, and it is
// measured rather than assumed (tests/v4_attn.rs).
extern "C" __global__ void gemv_fp8_bf16(const float* __restrict__ x,
                                            const unsigned char* __restrict__ w,
                                            const float* __restrict__ wscale, int m, int n_out,
                                            int k, int block, int groups,
                                            float* __restrict__ out) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    size_t id = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (id >= (size_t)m * (size_t)n_out) return;
    int r = (int)(id / n_out), j = (int)(id % n_out);
    int bsh = blk_shift(block);
    const float* xr = x + ((size_t)r * groups + (size_t)(j / (n_out / groups))) * (size_t)k;
    float acc = dot_fp8_wave(xr, w + (size_t)j * k,
                             wscale + (size_t)(j >> bsh) * ((k + block - 1) / block), k,
                             block, (int)(threadIdx.x & (WAVE - 1)), lut);
    if ((threadIdx.x & (WAVE - 1)) == 0) out[(size_t)r * n_out + j] = rbf16(acc);
}

// ── V4 launchers ────────────────────────────────────────────────────────────────────
// Return codes follow this file's convention: 0 ok, POSITIVE argument guard, NEGATIVE
// -hipError. All pointers device-resident; launch-only.
//
// **Every one takes a `stream`, and they took none until 2026-08-05.** These five plus
// `attn.hip::rivoli_gather_attn_shared_kv` are the whole of `attn::v4::attention`, and they were
// declined a stream once on the grounds that "there is nothing to overlap with". That
// premise died when the `.f4` routed streaming pool landed and was measured at 1.082
// ms/miss, 12.36 GB/s over the real 137.06 GiB expert set: there is now a fetch to overlap
// compute with on every layer.
//
// They convert as a SET, and that is a correctness property rather than tidiness. rivoli's
// streams are created `hipStreamNonBlocking` (`async.hip`), so the null stream carries no
// implicit ordering against them — a stream-capable launcher next to a null-stream
// neighbour reading the same buffer is an unordered activation, i.e. a silent wrong, which
// is strictly worse than leaving the whole block on the null stream. The seventh member of
// the set is not a launcher at all: `linalg.hip::rivoli_memcpy_dtod` is a BLOCKING
// `hipMemcpy` and the block interleaves six of them with these launches, so it needed an
// `_async` twin for the same reason. See that function's note.
//
// `stream` is trailing on all of them, matching `rivoli_hc_pre`/`rivoli_gemm_bf16`,
// and `nullptr` is the null stream — so a caller that has no stream yet is unchanged.

extern "C" int rivoli_act_quant_f8_prefix(const float* src, float* dst, int rows, int row_stride,
                                          int n, int block, void* stream) {
    if (rows <= 0 || n <= 0 || block <= 0 || row_stride < n) return 1001;
    // A quant-FROM-SOURCE (`src != dst`) writes only dims [0, n) of each dst row, so a
    // partial-width one (`n < row_stride` — the KV entry's shape) would leave dst's tail
    // holding whatever was there before, which no caller can want: the copy this fuses
    // away delivered the whole row. Full-width or in-place only. 1002 is this file's
    // size-bound code, and "dst rows would be part-stale" is a bound on what `n` may
    // cover, not a divisibility fact (1004 below).
    if (src != dst && n != row_stride) return 1002;
    // `kernel.py::act_quant` asserts `N % block_size == 0`. Without this a ragged tail
    // block would be quantized against its own amax and produce values the reference
    // cannot produce — silently, since every shape in the model divides evenly today.
    // 1004 is this file's divisibility code (cf. rivoli_gemv_fp8_bf16 below); 1002 is a
    // size bound. Getting them the wrong way round costs nothing at the call site and
    // misleads everyone who learned the codes from the other ~60 launchers.
    if (n % block) return 1004;
    size_t total = (size_t)rows * (size_t)(n / block);
    hipGetLastError();
    hipLaunchKernelGGL(act_quant_f8_prefix, dim3((unsigned)((total + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, src, dst, rows, row_stride, n, block);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rmsnorm_batch(float* x, const float* w, int rows, int d, float eps,
                                    void* stream) {
    if (rows <= 0 || d <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rmsnorm_batch, dim3((unsigned)rows), dim3(256), 0, (hipStream_t)stream, x, w,
                       d, eps);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_qk_norm(float* q, int rows, int d, float eps, void* stream) {
    if (rows <= 0 || d <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(qk_norm, dim3((unsigned)rows), dim3(256), 0, (hipStream_t)stream, q, d,
                       eps);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rope_adjacent(float* x, const float* tbl, int rows, int row_len, int rd,
                                    int pos0, int rows_per_pos, int inverse, void* stream) {
    if (rows <= 0 || row_len <= 0 || rd <= 0 || rows_per_pos <= 0 || pos0 < 0) return 1001;
    if (rd > row_len) return 1002;
    // `view_as_complex` needs an even count to pair, and the table holds rd/2 pairs.
    // 1005, not 1003: 1003 is "power-of-two tile" everywhere else in this file, and a
    // parity requirement is not that.
    if (rd & 1) return 1005;
    if (rows % rows_per_pos) return 1004;
    hipGetLastError();
    hipLaunchKernelGGL(rope_adjacent, dim3((unsigned)rows), dim3(256), 0, (hipStream_t)stream, x, tbl,
                       row_len, rd, pos0, rows_per_pos, inverse);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_fp8_bf16(const float* x, const unsigned char* w, const float* wscale,
                                       int m, int n_out, int k, int block, int groups, float* out,
                                       void* stream) {
    if (m <= 0 || n_out <= 0 || k <= 0 || block <= 0 || groups <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    // A `groups` that does not divide `n_out` would send the last partial group's rows to
    // a slice index no input was sized for, and read past the end.
    if (n_out % groups) return 1004;
    size_t total = (size_t)m * (size_t)n_out;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_fp8_bf16,
                       dim3((unsigned)((total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)),
                       dim3(ROWS_PER_BLOCK * WAVE), 0, (hipStream_t)stream, x, w, wscale, m, n_out,
                       k, block, groups, out);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
