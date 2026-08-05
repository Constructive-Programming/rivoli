// Per-layer linear-algebra primitives — the projections the forward pass needs
// on-device (batch-1 GEMVs), int4-free. One WAVE cooperates on one output row:
// adjacent lanes read adjacent columns (coalesced), then a fixed-order wave-shuffle
// reduces. The reduction order differs from the scalar oracle's ascending sum by
// f32 rounding (within the tests' tolerance) but is SCHEDULE-INDEPENDENT, so greedy
// decode stays deterministic. All pointers device-resident; launchers only launch
// (caller joins via rivoli_device_sync). Return codes: 0 ok, POSITIVE arg guard,
// NEGATIVE -(hipError_t).

#include "common.hpp"

// y[o] = Σ_i x[i]·e4m3(packed[o·i_dim+i])·scale[(o/block)·sc_cols + i/block] — fp8
// block-scaled (attention/dense projections; matches quant.rs matvec_fp8).
// `R` token rows through one read of the weight row. `x` rows are `i_dim` apart and `y`
// rows `o_dim` apart — both already arguments, so batching costs no extra parameter and at
// R=1 every index is the single-row one.
template <int R>
__device__ __forceinline__ void gemv_fp8_impl(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    __shared__ float lut[256];
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    int sc_cols = (i_dim + block - 1) / block;
    float acc[R];
    fp8_dot_strided_r<R>(x, i_dim, packed + o * (size_t)i_dim,
                         scale + (o >> blk_shift(block)) * sc_cols, i_dim, block, lut, lane, WAVE,
                         acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // Uniform across the wave: R is a compile-time constant.
        float s = wave_sum(acc[r]);
        if (lane == 0) y[(size_t)r * o_dim + o] = s;
    }
}

extern "C" __global__ void gemv_fp8(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_impl<1>(x, packed, scale, o_dim, i_dim, block, y);
}

extern "C" __global__ void gemv_fp8_r2(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_impl<2>(x, packed, scale, o_dim, i_dim, block, y);
}

// Split-K fp8 GEMV for LONG reductions: ONE block per output row, its ROWS_PER_BLOCK
// waves each reduce a strided 1/K share of i_dim, combined in LDS in fixed order
// (deterministic). o_proj (o_dim=6144, i_dim=16384) has too few rows to fill the
// machine under wave-per-row (768 blocks, each wave grinding a 16384-long chain);
// one-row-per-block gives K× the blocks so DRAM saturates — measured 128→~250 GB/s.
// Same LUT + block-scaled e4m3 dot as gemv_fp8; the whole block strides coalesced.
template <int R>
__device__ __forceinline__ void gemv_fp8_splitk_impl(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    __shared__ float lut[256];
    __shared__ float part[R][ROWS_PER_BLOCK]; // K = ROWS_PER_BLOCK wave partials, per row
    e4m3_lut_build(lut, threadIdx.x);
    __syncthreads();
    size_t o = blockIdx.x; // one output row per block
    if (o >= (size_t)o_dim) return;
    int wave = threadIdx.x / WAVE;
    int lane = threadIdx.x & (WAVE - 1);
    int sc_cols = (i_dim + block - 1) / block;
    const unsigned char* wrow = packed + o * (size_t)i_dim;
    // Shift, NOT divide: `o` is a size_t, so `o / block` was a runtime division once per
    // thread (~55 instructions). Same index, the launcher rejecting a non-power-of-two
    // tile. Measured −0.9%, INSIDE the noise band and not the 20% the identical fix
    // bought in mla_value_fp8 — why, and why it shipped anyway, in docs/measurement/perf-roadmap.md #2.
    const float* scalerow = scale + (o >> blk_shift(block)) * sc_cols;
    const int STRIDE = ROWS_PER_BLOCK * WAVE; // all 256 threads stride coalesced (split-K)
    // Same strided fp8 MAC as dot_fp8_wave, but split-K across every wave in the block.
    float acc[R];
    fp8_dot_strided_r<R>(x, i_dim, wrow, scalerow, i_dim, block, lut, threadIdx.x, STRIDE, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = wave_sum(acc[r]); // per-wave partial → its lane 0
        if (lane == 0) part[r][wave] = s;
    }
    __syncthreads();
    // One thread per token row, each folding its own K partials in the SAME fixed order as
    // the single-row kernel — so the split-K combine stays deterministic per row.
    if (threadIdx.x < R) {
        float s = 0.0f;
        for (int w = 0; w < ROWS_PER_BLOCK; ++w) s += part[threadIdx.x][w];
        y[(size_t)threadIdx.x * o_dim + o] = s;
    }
}

extern "C" __global__ void gemv_fp8_splitk(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_splitk_impl<1>(x, packed, scale, o_dim, i_dim, block, y);
}

extern "C" __global__ void gemv_fp8_splitk_r2(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, int block, float* __restrict__ y) {
    gemv_fp8_splitk_impl<2>(x, packed, scale, o_dim, i_dim, block, y);
}

// y[o] = Σ_subvec scale[o,grp]·codebook[idx]·x — VQ-int3, per-group bf16 scale
// applied inside the dot. `codebook` is shared across all rows.
extern "C" __global__ void gemv_vq(
    const float* __restrict__ x, const unsigned char* __restrict__ indices,
    const unsigned short* __restrict__ scales, const __half* __restrict__ codebook,
    int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    size_t rb = (size_t)(i_dim / VQ_DIM) * VQ_INDEX_BITS / 8; // vq_row_bytes
    size_t ng = (size_t)i_dim / VQ_GROUP;                     // vq_groups
    float acc = dot_vq_wave(x, indices + o * rb, scales + o * ng, codebook, i_dim, lane);
    if (lane == 0) y[o] = acc;
}

// y[r][o] = Σ_i x[r][i]·w[o·i_dim+i] — plain f32 GEMV (the MoE router gate), R token
// rows through one read of `w`. The gate is 6.3 MB per MoE layer, ~470 MB a pass, so a
// speculative verify wants it read once. `x` rows are `i_dim` apart, `y` rows `o_dim`.
// Row 0 is bit-identical: same `i` order, same expression, just replicated.
template <int R>
__device__ __forceinline__ void gemv_f32_impl(
    const float* __restrict__ x, const float* __restrict__ w, int o_dim, int i_dim,
    float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const float* row = w + o * (size_t)i_dim;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int i = lane; i < i_dim; i += WAVE) {
        float wi = row[i];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += x[(size_t)r * i_dim + i] * wi;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = wave_sum(acc[r]);
        if (lane == 0) y[(size_t)r * o_dim + o] = s;
    }
}

extern "C" __global__ void gemv_f32(
    const float* __restrict__ x, const float* __restrict__ w, int o_dim, int i_dim,
    float* __restrict__ y) {
    gemv_f32_impl<1>(x, w, o_dim, i_dim, y);
}

extern "C" __global__ void gemv_f32_r2(
    const float* __restrict__ x, const float* __restrict__ w, int o_dim, int i_dim,
    float* __restrict__ y) {
    gemv_f32_impl<2>(x, w, o_dim, i_dim, y);
}

// y[o] = scale[o]·Σ_i x[i]·(int8)packed[o·i_dim+i] — per-row int8 GEMV (lm_head to
// logits; matches quant.rs matvec_i8). One WAVE per output row, coalesced columns.
//
// FOUR columns per lane, not one. A scalar `row[i]` is a global_load_i8: 32 B/wave
// against a 128 B cache line, so the kernel asked DRAM for a quarter line at a time and
// measured 117 GB/s of a 256 GB/s part. Four contiguous int8 are one dword — a full line
// per wave — paired with one float4 of x. Same mechanism, and the same fix, as
// mla_absorb's i-quad (docs/measurement/perf-roadmap.md #4 and #5).
//
// NOT bit-identical to the scalar form: each lane now owns a contiguous quad instead of
// a stride-WAVE comb, which reassociates the f32 sum. lm_head feeds argmax directly, so
// this took the perplexity gate rather than a fingerprint.
//
// R token rows through one read of the 952 MB lm_head. Row 0 is bit-identical: the quad
// decode, the summation order and the final `acc * scale[o]` are unchanged, only
// replicated. `x` rows are `i_dim` apart, `y` rows `o_dim` (= vocab).
template <int R>
__device__ __forceinline__ void gemv_i8_impl(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    const signed char* row = packed + o * (size_t)i_dim;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    int base = 0;
    // The quad path needs `row` dword-aligned and `x` float4-aligned. `packed` is a
    // device allocation, so the real condition is i_dim % 4 keeping every row aligned;
    // both are uniform per block and anything failing them falls to the scalar tail.
    // With R > 1 there is one more: `x + r·i_dim` must stay float4-aligned, i.e. i_dim
    // divisible by 4. Gated on `R == 1` (a compile-time constant) rather than added
    // unconditionally, so the single-row path keeps taking the quad route on a ragged
    // i_dim exactly as it did before — the oracle tests run such shapes.
    if ((((size_t)row) & 3u) == 0 && (((size_t)x) & 15u) == 0 && (R == 1 || (i_dim & 3) == 0)) {
        for (; base + WAVE * 4 <= i_dim; base += WAVE * 4) {
            int col = base + lane * 4;
            int w = *(const int*)(row + col); // 4 int8 = one dword = 128 B/wave
            float w0 = (float)(signed char)(w), w1 = (float)(signed char)(w >> 8);
            float w2 = (float)(signed char)(w >> 16), w3 = (float)(signed char)(w >> 24);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float4 xv = *(const float4*)(x + (size_t)r * i_dim + col);
                acc[r] += xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
            }
        }
    }
    for (int i = base + lane; i < i_dim; i += WAVE) {
        float wi = (float)row[i];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += x[(size_t)r * i_dim + i] * wi;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = wave_sum(acc[r]);
        if (lane == 0) y[(size_t)r * o_dim + o] = s * scale[o];
    }
}

extern "C" __global__ void gemv_i8(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    gemv_i8_impl<1>(x, packed, scale, o_dim, i_dim, y);
}

extern "C" __global__ void gemv_i8_r2(
    const float* __restrict__ x, const signed char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    gemv_i8_impl<2>(x, packed, scale, o_dim, i_dim, y);
}

// y[o] = Σ_i x[i]·(nibble(o,i)-8)·scale[o, i/I4_GROUP] — group-scaled int4,
// wave-per-row (the exact dot_i4_wave the MoE kernels use). Exists so the dot's decode
// throughput can be measured in isolation against gemv_vq / gemv_fp8 at the MoE
// projection dims, free of the routing/miss-count confound the decode bench has.
extern "C" __global__ void gemv_i4(
    const float* __restrict__ x, const unsigned char* __restrict__ packed,
    const float* __restrict__ scale, int o_dim, int i_dim, float* __restrict__ y) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t o = (size_t)blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x / WAVE);
    if (o >= (size_t)o_dim) return;
    size_t rb = (size_t)((i_dim + 1) / 2);
    size_t ng = (size_t)((i_dim + I4_GROUP - 1) / I4_GROUP);
    float dv = dot_i4_wave(x, packed + o * rb, scale + o * ng, i_dim, lane);
    if (lane == 0) y[o] = dv;
}

// h[i] = silu(g[i])·u[i], silu(x)=x/(1+e^-x) — the SwiGLU combine for the dense
// fp8 MLP (moe.hip does this internally for VQ experts; the dense path is three
// gemv_fp8 + this). Safe in place (h may alias g): each thread reads g[i]/u[i] then
// writes h[i] independently.
extern "C" __global__ void swiglu(
    const float* __restrict__ g, const float* __restrict__ u, int n, float* __restrict__ h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i];
    h[i] = (gv / (1.0f + expf(-gv))) * u[i];
}

// DeepSeek-V4's `Expert.forward` combine for the resident fp8 SHARED expert:
// h[i] = bf16( silu(min(bf16(g[i]), L)) · clamp(bf16(u[i]), ±L) ).
//
// Sits here, directly under `swiglu`, so that the two are readable side by side — because
// the interesting fact about them is that they are NOT the same function with a parameter,
// and that is not obvious from either one alone. At NO value of `L` do they agree. Beyond
// the clamp there are three differences, every one of them a defect the oracle can name:
//
//   * this bf16-rounds BOTH operands before combining (`Linear` stores bf16 and
//     `Expert.forward` reads it back with `.float()`) — `Defect::NoBf16Rounding`;
//   * this uses `F.silu`'s multiply form `x·sigmoid(x)`, not `swiglu`'s division form
//     `x/(1+e^-x)` — one rounding apart, and that rounding flips a whole bf16 ulp at a
//     boundary, which the store below then freezes in;
//   * this bf16-rounds the RESULT, which is the reference's `x.to(dtype)` before `w2`.
//
// So adding an `L` parameter to `swiglu` and passing something meaning "unclamped" from
// GLM's four call sites could not have produced this kernel — it would have produced GLM's
// arithmetic under a V4 name, which is the silent-wrong this whole port is built to avoid.
// GLM has no `swiglu_limit` and never should: `rivoli_v4_swiglu_clamped` refuses
// `L <= 0` (and NaN) exactly as `moe.hip`'s fp4 launcher does, so "unclamped" is not
// spellable THROUGH THIS ENTRY POINT. Note what that does and does not buy as of
// 2026-08-05: V4's shared expert has no caller yet, so nothing is currently prevented from
// reaching for `swiglu` above instead — the guard makes the mistake unspellable here, not
// unreachable everywhere. The launcher's Rust doc carries the same warning where a caller
// will read it.
//
// The clamp itself is `common.hpp::swiglu_clamped`, shared with `moe_gateup_f4_impl`. The
// routed fp4 experts and the shared fp8 expert must run the SAME arithmetic — `MoE.__init__`
// hands `swiglu_limit` to both (model.py:632) — and two copies in `kernels/` would be
// invisible to jscpd, which does not scan this directory.
//
// No routing weight, and that is the reference, not an omission: `MoE.forward` calls
// `self.shared_experts(x)` with no `weights` argument (model.py:648). So unlike the fp4
// path there is nothing to fold in before the store, and no `w == 0` NaN launder to carry.
//
// Safe in place (`h` may alias `g` or `u`), exactly as `swiglu` above and for the same
// reason: each thread performs both reads, then one write, and that write DEPENDS on both
// reads, so there is no ordering for the `__restrict__` qualifiers to license that would
// observe the overwrite. `gpu.rs` already launches `swiglu` this way (`h == g`), which is
// what the shared expert wants too — `g`'s buffer becomes `h` and then `w2`'s input, one
// fewer scratch allocation per layer.
extern "C" __global__ void v4_swiglu_clamped(
    const float* __restrict__ g, const float* __restrict__ u, int n, float limit,
    float* __restrict__ h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = rbf16(swiglu_clamped(g[i], u[i], limit));
}

// y[i] = x[i]·(1/sqrt(mean(x²)+eps))·w[i] — RMSNorm, matching math.rs. One block
// reduces Σx² in LDS (tree order), then every thread scales.
extern "C" __global__ void rmsnorm(
    const float* __restrict__ x, const float* __restrict__ w, int n, float eps,
    float* __restrict__ y) {
    __shared__ float red[256];
    int t = threadIdx.x;
    float local = 0.0f;
    for (int i = t; i < n; i += blockDim.x) local += x[i] * x[i];
    red[t] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(red[0] / (float)n + eps);
    for (int i = t; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

// `kernel.py::act_quant(x, 128, "ue8m0", …, inplace=True)` over `n_rows × row_len` f32,
// IN PLACE — the fused quantize-then-dequantize DeepSeek-V4 puts in front of EVERY
// quantized `Linear`, fp4-weight ones included (`model.py::linear` line 120).
//
// Its own kernel rather than folded into `dot_f4_wave_r` because the block is 128 wide and
// spans the whole reduction dim: a wave that owns one output row sees only its own strided
// slice of `x`, so it cannot compute the block amax. Hoisting it also makes it run ONCE per
// activation instead of once per output row — `w1` and `w3` consume the identical
// quantized `x`, and so does the resident fp8 shared expert.
//
// One wave per 128-element group. `row_len % ACT_QUANT_BLOCK == 0` is the reference's own
// `assert N % block_size == 0` (kernel.py:112) and the launcher enforces it, so the group
// tiling below never straddles a row boundary.
extern "C" __global__ void act_quant_f8(float* __restrict__ v, size_t nblk) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t b = (size_t)blockIdx.x * (blockDim.x / WAVE) + threadIdx.x / WAVE;
    if (b >= nblk) return;
    float* seg = v + b * ACT_QUANT_BLOCK;
    float amax = 0.0f;
    for (int i = lane; i < ACT_QUANT_BLOCK; i += WAVE) amax = fmaxf(amax, fabsf(seg[i]));
    amax = fmaxf(wave_max(amax), ACT_QUANT_AMAX_FLOOR);
    float s = fast_round_scale(amax, FP8_MAX_INV);
    for (int i = lane; i < ACT_QUANT_BLOCK; i += WAVE) seg[i] = act_quant_roundtrip(seg[i], s);
}

// ── mHC: hyper-connections (`model.py::Block.hc_pre` / `hc_post`) ───────────────
//
// V4 does not carry ONE residual stream, it carries `HC_MULT` copies. `hc_pre` reduces
// them to one with learned Sinkhorn-normalised weights before the attention/MoE sublayer;
// `hc_post` expands the sublayer's single output back to `HC_MULT`, mixing in the
// pre-sublayer copies through a learned combination matrix. A plain residual add in place
// of `hc_post` runs, converges to nothing useful, and never errors.
//
// HC_MULT is 4 and is not a knob: `hc_split_sinkhorn`'s `mix_hc = (2 + hc)·hc` layout is
// how the mHC weights are PACKED on disk, so another value is a different checkpoint. The
// launchers refuse anything else rather than pretending to generalise.
#define HC_MULT 4
#define HC_MIX ((2 + HC_MULT) * HC_MULT)

// `kernel.py::hc_split_sinkhorn` for one token, on ONE thread.
//
// Serial because it is 24 inputs and a 4x4 matrix: the whole routine is ~640 flops, and a
// parallel version would have to reproduce the reference's summation order across lanes to
// stay comparable at all. Note the asymmetry a port loses first — the FIRST normalisation
// is a row *softmax* followed by a column divide, and only the remaining `iters - 1`
// passes are plain row/column divides.
//
// `iters` is a parameter, not a constant, so the count comes from `config.json` through
// `V4Config` and cannot drift. It is NOT gated by any numerical comparison: at 20 passes a
// 4x4 positive matrix is far past convergence, so 19 and 20 agree bit-for-bit and no
// golden can tell them apart (`tests/v4_oracle.rs::sinkhorn_has_converged_long_before_
// iteration_20`). What IS gated is that the parameter is live — 2 vs 20 moves the output.
__device__ __forceinline__ void hc_sinkhorn(
    const float* __restrict__ mixes, const float* __restrict__ scale,
    const float* __restrict__ base, int iters, float eps, float* __restrict__ pre,
    float* __restrict__ post, float* __restrict__ comb) {
    // `2 * sigmoid(x)`, not `2 / (1 + e^-x)`: the reference multiplies the finished
    // sigmoid, and folding the 2 into the numerator moves the rounding.
    for (int j = 0; j < HC_MULT; ++j) {
        pre[j] = 1.0f / (1.0f + expf(-(mixes[j] * scale[0] + base[j]))) + eps;
        post[j] = 2.0f * (1.0f / (1.0f + expf(-(mixes[j + HC_MULT] * scale[1]
                                                + base[j + HC_MULT]))));
    }
    for (int j = 0; j < HC_MULT * HC_MULT; ++j)
        comb[j] = mixes[j + 2 * HC_MULT] * scale[2] + base[j + 2 * HC_MULT];
    // comb = comb.softmax(-1) + eps, row by row. Max/exp/sum in the reference's own order.
    for (int j = 0; j < HC_MULT; ++j) {
        float* row = comb + j * HC_MULT;
        float m = row[0];
        for (int t = 1; t < HC_MULT; ++t) m = fmaxf(m, row[t]);
        float s = 0.0f;
        for (int t = 0; t < HC_MULT; ++t) { row[t] = expf(row[t] - m); s += row[t]; }
        for (int t = 0; t < HC_MULT; ++t) row[t] = row[t] / s + eps;
    }
    // `comb / (comb.sum(-1) + eps)` and `comb / (comb.sum(-2) + eps)` differ only in which
    // index they hold fixed, so one normaliser takes that as a (hold, run) stride pair. Two
    // copies would be two places to get the eps or the axis wrong.
    auto norm = [&](int hold, int run) {
        for (int fixed = 0; fixed < HC_MULT; ++fixed) {
            float* c = comb + fixed * hold;
            float s = 0.0f;
            for (int r = 0; r < HC_MULT; ++r) s += c[r * run];
            for (int r = 0; r < HC_MULT; ++r) c[r * run] /= s + eps;
        }
    };
    // The reference's own sequence: ONE column pass, then `iters - 1` row/column PAIRS.
    // Written out rather than folded into a `2·iters − 1` loop with a parity switch — the
    // switch IS the axis, and swapping its two strides is exactly as silent as two loops
    // disagreeing, while this reads against `Oracle::sinkhorn` line for line.
    norm(1, HC_MULT);  // column: hold the dest index, run over the source
    for (int i = 1; i < iters; ++i) {
        norm(HC_MULT, 1);  // row: hold the source index, run over the dest
        norm(1, HC_MULT);
    }
}

// `Block.hc_pre`: one block per token. `h` is [s, HC_MULT, dim]; the rsqrt is over the FULL
// HC_MULT·dim flattened row, not per copy — a per-copy norm is a different (and plausible)
// statistic that leaves every magnitude in range.
//
// Returns the reduced `y` [s, dim] plus the `post`/`comb` the matching `hc_post` needs, so
// the Sinkhorn runs ONCE per token per sublayer rather than twice.
extern "C" __global__ void hc_pre(
    const float* __restrict__ h, const float* __restrict__ fnw, const float* __restrict__ scale,
    const float* __restrict__ base, int dim, int iters, float norm_eps, float hc_eps,
    float* __restrict__ y, float* __restrict__ post_out, float* __restrict__ comb_out) {
    __shared__ float red[256];
    __shared__ float mixes[HC_MIX];
    __shared__ float pre[HC_MULT], post[HC_MULT], comb[HC_MULT * HC_MULT];
    int t = threadIdx.x, nt = blockDim.x;
    size_t hcd = (size_t)HC_MULT * dim;
    const float* flat = h + (size_t)blockIdx.x * hcd;

    float local = 0.0f;
    for (size_t i = t; i < hcd; i += nt) local += flat[i] * flat[i];
    red[t] = local;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float rs = 1.0f / sqrtf(red[0] / (float)hcd + norm_eps);

    // One wave per mix row, three rounds at 256 threads (HC_MIX = 24, 8 waves).
    int lane = t & (WAVE - 1), w = t / WAVE, nw = nt / WAVE;
    for (int j = w; j < HC_MIX; j += nw) {
        const float* wr = fnw + (size_t)j * hcd;
        float acc = 0.0f;
        for (size_t i = lane; i < hcd; i += WAVE) acc += flat[i] * wr[i];
        acc = wave_sum(acc);
        if (lane == 0) mixes[j] = acc * rs;
    }
    __syncthreads();
    if (t == 0) hc_sinkhorn(mixes, scale, base, iters, hc_eps, pre, post, comb);
    __syncthreads();

    // y[d] = Σ_c pre[c]·flat[c·dim + d], then back to bf16 (`y.to(dtype)`).
    float* yo = y + (size_t)blockIdx.x * dim;
    for (int d = t; d < dim; d += nt) {
        float acc = 0.0f;
        for (int c = 0; c < HC_MULT; ++c) acc += pre[c] * flat[(size_t)c * dim + d];
        yo[d] = rbf16(acc);
    }
    if (t < HC_MULT) post_out[(size_t)blockIdx.x * HC_MULT + t] = post[t];
    if (t < HC_MULT * HC_MULT) comb_out[(size_t)blockIdx.x * HC_MULT * HC_MULT + t] = comb[t];
}

// `Block.hc_post`: y[t][k][d] = post[t][k]·x[t][d] + Σ_j comb[t][j][k]·residual[t][j][d].
//
// `comb` is indexed [SOURCE, DEST] — the Sinkhorn row-softmax runs over the dest index and
// the column normalisation over the source index. Transposing it keeps every output row a
// combination of the same vectors, so it survives every magnitude and norm check
// (`v4oracle::Defect::SinkhornCombTransposed`); only a numerical comparison sees it.
extern "C" __global__ void hc_post(
    const float* __restrict__ x, const float* __restrict__ residual,
    const float* __restrict__ post, const float* __restrict__ comb, int s, int dim,
    float* __restrict__ y) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)s * HC_MULT * dim) return;
    int d = (int)(i % dim);
    int k = (int)((i / dim) % HC_MULT);
    size_t tok = i / dim / HC_MULT;
    float acc = post[tok * HC_MULT + k] * x[tok * dim + d];
    for (int j = 0; j < HC_MULT; ++j)
        acc += comb[(tok * HC_MULT + j) * HC_MULT + k] * residual[(tok * HC_MULT + j) * dim + d];
    y[i] = rbf16(acc);  // `.type_as(x)` — the residual stream is bf16
}

// Interleaved RoPE (matching attn.rs) on `count` segments of `seg` elements:
// segment s at base[s*stride..+seg]; pair (2j,2j+1) rotates by pos·theta^(-2j/seg)
// into (j, half+j). Double angles (long-context arg reduction); one block/segment,
// seg/2 threads; all pairs read before any write (barrier) → race-free in place.
extern "C" __global__ void rope_interleave(
    float* __restrict__ base, int stride, int seg, int pos, double theta) {
    int s = blockIdx.x;
    int j = threadIdx.x;
    int half = seg / 2;
    float* v = base + (size_t)s * stride;
    float a = v[2 * j], b = v[2 * j + 1];
    __syncthreads();
    double inv = pow(theta, -2.0 * (double)j / (double)seg);
    double ang = (double)pos * inv;
    float cs = (float)cos(ang), sn = (float)sin(ang);
    v[j] = a * cs - b * sn;
    v[half + j] = b * cs + a * sn;
}

// VQ encode: idx[i] = argmin_k (‖c_k‖² − 2·sub_i·c_k), tie → lowest k. Matches
// quant.rs::quant_vq's nearest exactly (FP contraction OFF, so the dot rounds like
// the CPU — a fused FMA would flip near-tied argmins on the learned codebook). The
// offline converter's argmin accelerator.
extern "C" __global__ void vq_encode(
    const float* __restrict__ sub, const float* __restrict__ codebook,
    const float* __restrict__ cbnorm, int n, unsigned short* __restrict__ idx) {
#pragma clang fp contract(off)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sv[VQ_DIM];
#pragma unroll
    for (int d = 0; d < VQ_DIM; ++d) sv[d] = sub[(size_t)i * VQ_DIM + d];
    float best = 3.4e38f;
    int bk = 0;
    for (int k = 0; k < VQ_K; ++k) {
        const float* c = codebook + (size_t)k * VQ_DIM;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < VQ_DIM; ++d) dot += sv[d] * c[d];
        float dist = cbnorm[k] - 2.0f * dot;
        if (dist < best) {
            best = dist;
            bk = k;
        }
    }
    idx[i] = (unsigned short)bk;
}

// ── launchers (grid = ceil(o_dim / ROWS_PER_BLOCK) blocks of 256 threads) ──────

#define ROW_GRID(o) dim3(((o) + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)
#define ROW_BLOCK dim3(ROWS_PER_BLOCK * WAVE)

// Wave-per-row saturates DRAM once o_dim is large enough to fill the machine; below
// that (o_proj: o_dim=6144, i_dim=16384) too few blocks launch and DRAM starves, so
// switch to split-K (one block/row). Threshold on i_dim: the plain kernel measures
// near-peak (255-283 GB/s) for i_dim ≤ 2048 and drops (~128) for i_dim ≥ 8192.
#define GEMV_SPLITK_MIN_IDIM 4096

extern "C" int rivoli_gemv_fp8_splitk(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    int block, int nrow, float* y) {
    if (o_dim <= 0 || i_dim <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    if (nrow != 1 && nrow != 2) return 1004;
    hipGetLastError();
    if (nrow == 1) {
        hipLaunchKernelGGL(gemv_fp8_splitk, dim3(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    } else {
        hipLaunchKernelGGL(gemv_fp8_splitk_r2, dim3(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    }
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// `nrow` token rows (1 or 2) through ONE read of the fp8 weights: `x[r·i_dim + i]` →
// `y[r·o_dim + o]`. The grid is unchanged — the row axis lives inside each wave, not
// beside it — so this is the same launch doing nrow times the arithmetic on the same bytes.
extern "C" int rivoli_gemv_fp8(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    int block, int nrow, float* y) {
    // NO `i_dim % 4` guard, unlike the Vulkan twin and unlike rivoli_mla_value_fp8's
    // `kvl % 4` — a known gap, conditional on block >= 4. See common.hpp fp8_dot_strided.
    if (o_dim <= 0 || i_dim <= 0 || block <= 0) return 1001;
    if (block & (block - 1)) return 1003;  // blk_shift needs a power-of-two tile
    if (nrow != 1 && nrow != 2) return 1004;
    if (i_dim >= GEMV_SPLITK_MIN_IDIM)
        return rivoli_gemv_fp8_splitk(x, packed, scale, o_dim, i_dim, block, nrow, y);
    hipGetLastError();
    if (nrow == 1) {
        hipLaunchKernelGGL(gemv_fp8, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    } else {
        hipLaunchKernelGGL(gemv_fp8_r2, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, block, y);
    }
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_vq(
    const float* x, const unsigned char* indices, const unsigned short* scales,
    const __half* codebook, int o_dim, int i_dim, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_vq, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, indices, scales, codebook, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_i4(
    const float* x, const unsigned char* packed, const float* scale, int o_dim, int i_dim,
    float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gemv_i4, ROW_GRID(o_dim), ROW_BLOCK, 0, 0, x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gemv_i8(
    const float* x, const signed char* packed, const float* scale, int o_dim, int i_dim,
    int nrow, float* y) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    if (nrow != 1 && nrow != 2) return 1004;  // R is a template parameter — see gemv_fp8
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? gemv_i8_r2 : gemv_i8, ROW_GRID(o_dim), ROW_BLOCK, 0, 0,
                       x, packed, scale, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// Takes a STREAM, unlike the other GEMVs in this file, and it is the odd one out because of
// DeepSeek-V4 rather than GLM: this is the router-gate GEMV (`Gate.forward`'s
// `linear(x.float(), weight.float())`), so on the V4 layer path it sits between two launchers
// that take one. `nullptr` is the null stream, so every existing caller is unchanged. The full
// argument, and why this is not a V4-only twin launcher, is on `hip.rs::launch_gemv_f32` —
// stated once, where a caller reads it.
extern "C" int rivoli_gemv_f32(const float* x, const float* w, int o_dim, int i_dim, int nrow,
                               float* y, void* stream) {
    if (o_dim <= 0 || i_dim <= 0) return 1001;
    if (nrow != 1 && nrow != 2) return 1004;  // R is a template parameter — see gemv_fp8
    hipGetLastError();
    hipLaunchKernelGGL(nrow == 2 ? gemv_f32_r2 : gemv_f32, ROW_GRID(o_dim), ROW_BLOCK, 0,
                       (hipStream_t)stream, x, w, o_dim, i_dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_swiglu(const float* g, const float* u, int n, float* h) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(swiglu, dim3((n + 255) / 256), dim3(256), 0, 0, g, u, n, h);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// Takes a STREAM, unlike `rivoli_swiglu` directly above. Same reason `rivoli_hc_pre` gives
// below: rivoli's streams are `hipStreamNonBlocking` (`async.hip`), so the null stream does
// not implicitly order against them, and this runs between two `rivoli_v4_gemv_fp8` launches
// that are now stream-parameterised. A null-stream combine between two stream-ordered GEMVs
// reads `g`/`u` before they are written — an unordered activation, which is a silent wrong.
extern "C" int rivoli_v4_swiglu_clamped(const float* g, const float* u, int n, float limit,
                                        float* h, void* stream) {
    if (n <= 0) return 1001;
    // 1006, and deliberately the SAME code `moe.hip`'s fp4 launcher returns for the same
    // check on the same argument — a reader who learned the code there gets it here. Written
    // `!(limit > 0.0f)` rather than `limit <= 0.0f` so NaN is REJECTED: every comparison
    // against NaN is false, so `limit <= 0` admits it, and a NaN limit makes
    // `fminf(gt, limit)` return `gt` (fminf returns the non-NaN operand) — i.e. it degrades
    // silently to the unclamped form this launcher exists to make unspellable, rather than
    // producing NaN output someone would notice.
    if (!(limit > 0.0f)) return 1006;
    hipGetLastError();
    hipLaunchKernelGGL(v4_swiglu_clamped, dim3((unsigned)((n + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, g, u, n, limit, h);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rmsnorm(const float* x, const float* w, int n, float eps, float* y) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rmsnorm, dim3(1), dim3(256), 0, 0, x, w, n, eps, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// `n_rows` rows of `row_len` f32, quantized in place on `stream`.
//
// `row_len % ACT_QUANT_BLOCK` is REFUSED (1002) rather than handled with a short tail
// block: the reference asserts the same thing, and a tail block would quantize the last
// few columns against a scale the reference never computes — values it cannot produce,
// with no error anywhere. `nblk` is computed here and passed in, so the kernel's bound and
// the grid that covers it come from ONE expression.
extern "C" int rivoli_act_quant_f8(float* v, int n_rows, int row_len, void* stream) {
    if (n_rows <= 0 || row_len <= 0) return 1001;
    if (row_len % ACT_QUANT_BLOCK) return 1002;
    size_t nblk = (size_t)n_rows * (size_t)(row_len / ACT_QUANT_BLOCK);
    int waves = 256 / WAVE;
    hipGetLastError();
    hipLaunchKernelGGL(act_quant_f8, dim3((nblk + waves - 1) / waves), dim3(256), 0,
                       (hipStream_t)stream, v, nblk);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// One block per token, 256 threads — the LDS reduction tree in `hc_pre` needs a
// power-of-two block, and `HC_MIX = 24` needs at least 8 waves to finish in three rounds.
//
// Takes a STREAM, unlike `rmsnorm`/`swiglu` above. rivoli's streams are created
// `hipStreamNonBlocking` (`async.hip`), so the null stream does NOT implicitly order against
// them: an mHC result on stream 0 feeding an FP4 expert range on a caller stream would be
// unordered, and an unordered residual is a silent wrong.
//
// `hc` is taken and CHECKED rather than assumed: the caller reads it from `V4Config`, so a
// checkpoint with a different `hc_mult` fails here instead of silently reading the mHC
// weights at the wrong stride (`mix_hc = (2 + hc)·hc` is the packing).
extern "C" int rivoli_hc_pre(
    const float* h, const float* fnw, const float* scale, const float* base, int s, int hc,
    int dim, int iters, float norm_eps, float hc_eps, float* y, float* post, float* comb,
    void* stream) {
    if (s <= 0 || dim <= 0) return 1001;
    if (hc != HC_MULT) return 1002;
    if (iters < 1) return 1003;
    hipGetLastError();
    hipLaunchKernelGGL(hc_pre, dim3(s), dim3(256), 0, (hipStream_t)stream, h, fnw, scale, base,
                       dim, iters, norm_eps, hc_eps, y, post, comb);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_hc_post(
    const float* x, const float* residual, const float* post, const float* comb, int s, int hc,
    int dim, float* y, void* stream) {
    if (s <= 0 || dim <= 0) return 1001;
    if (hc != HC_MULT) return 1002;
    size_t n = (size_t)s * HC_MULT * dim;
    hipGetLastError();
    hipLaunchKernelGGL(hc_post, dim3((n + 255) / 256), dim3(256), 0, (hipStream_t)stream, x,
                       residual, post, comb, s, dim, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rope(float* base, int count, int stride, int seg, int pos, double theta) {
    if (count <= 0 || seg <= 0 || (seg & 1) || (seg / 2) > 1024) return 1001;
    if (count > 1 && stride < seg) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rope_interleave, dim3(count), dim3(seg / 2), 0, 0, base, stride, seg, pos, theta);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_vq_encode(
    const float* sub, const float* codebook, const float* cbnorm, int n, unsigned short* idx) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(vq_encode, dim3((n + 255) / 256), dim3(256), 0, 0, sub, codebook, cbnorm, n, idx);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// One join point per token: block until all launched kernels retire.
extern "C" int rivoli_device_sync() {
    hipError_t e = hipDeviceSynchronize();
    return e == hipSuccess ? 0 : -(int)e;
}

// Synchronous device-to-device copy — the routed arena's slot relocation (compaction).
// Blocks so the moved expert's bytes are in place before the layer's MoE compute (or a
// later cold read) touches the new location. `dst`/`src` are distinct slots, so the
// regions never overlap.
extern "C" int rivoli_memcpy_dtod(void* dst, const void* src, size_t bytes) {
    hipError_t e = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
    return e == hipSuccess ? 0 : -(int)e;
}

// STREAM-ORDERED device-to-device copy, for a pipeline whose kernels are on a stream.
//
// A separate entry point rather than a `stream` parameter on `rivoli_memcpy_dtod`, because
// the two are not the same operation with a knob — the one above BLOCKS THE HOST, and the
// arena relocation it serves needs exactly that. `memory/routed.rs` moves an expert's slot
// while later reads may still resolve to the old address, and this repo has a measured
// defect from a read outliving its layer and getting its slot copied out from under it
// (9 of 8452 reads at `--max-mem 30`, non-deterministic output; pins do not stop
// compaction). Making that path stream-ordered would replace a host barrier with a
// per-stream one and reopen it. So: two names, and the blocking one keeps its callers.
//
// Why this exists at all — DeepSeek-V4's attention block interleaves six device-to-device
// copies with sixteen kernel launches (`attn.rs::v4::attention`: `x -> xq`, `qr -> qrq`, and
// four KV ring writes, of which one or two run depending on the phase). Once those kernels take a stream, a blocking `hipMemcpy` between two of them
// does NOT wait on the stream: rivoli's streams are `hipStreamNonBlocking` (`async.hip`), so
// the null stream has no implicit ordering against them. The copy would read `s.qr` before
// `v4_rmsnorm` wrote it. Found 2026-08-05 while converting that block's six launchers —
// converting six of the seven operations would have introduced the very race the earlier
// decline of the partial fix predicted.
extern "C" int rivoli_memcpy_dtod_async(void* dst, const void* src, size_t bytes,
                                        void* stream) {
    hipError_t e = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice,
                                  (hipStream_t)stream);
    return e == hipSuccess ? 0 : -(int)e;
}
