// **The gated-delta-rule linear-attention family.** Kimi-K3's 69 KDA layers are this
// recurrence plus a short causal convolution and a gated head-wise norm around it
// (`k3-architecture.md` §4); this translation unit is the recurrence, and 5b/5c of the plan's
// item 5 add the other two beside it.
//
// Named for the arithmetic, not the model. The reference calls the block KDA — Kimi Delta
// Attention — and fla ships it as `fused_recurrent_kda`; both names carry a model into a
// kernel, which this tree has a rule against. What the operator IS is the gated delta rule:
// a per-head state decayed by a per-key-channel gate, corrected by a rank-one update towards
// the value, and read back with the query.

#include "common.hpp"

// One decode step of the gated delta rule, one block per head. `S[i][j]` has `i` = key
// channel and `j` = value channel (§4 step 7):
//
//   S[i][j] <- alpha[i] * S[i][j]                     decay rows by the per-key-channel gate
//   u[j]     = sum_i k[i] * S[i][j]                   from the ALREADY-DECAYED S
//   S[i][j] <- S[i][j] + k[i] * beta * (v[j] - u[j])  rank-one; (v-u) is the prediction error
//   o[j]     = sum_i q[i] * S[i][j]                   read the UPDATED S
//
// **Two passes over `S`, not four, and that is the whole reason this kernel exists.** The C
// reference walks the state four times because it has no reason not to; here `S` is 64 KB per
// head at the real widths and will not sit in LDS, so every pass is 64 KB of real traffic and
// 96 heads make it 6 MB per token per direction. Thread `t` owns value channel `t` for the
// whole kernel, so the decay folds into the `u` reduction and the rank-one update folds into
// the `o` reduction, and **no cross-thread reduction appears in the recurrence at all** — each
// column's two sums are private to its thread. The only block reductions are the two L2 norms
// before it.
//
// The state is read at `S[i*d + t]`, so consecutive threads read consecutive addresses within
// one key channel.
//
// **[K][V] is the only layout that can be simultaneously VECTORIZED and coalesced, and that —
// not coalescing alone — is why it is the right one.** The first version of this comment said
// the transposed store "would give each thread a private row and cost the coalescing", and a
// performance review priced that argument and found it the weaker half: with a per-thread row
// the DRAM traffic is the same (the 32 over-fetched lines per wave-instruction are consumed by
// the same thread over its next 15 iterations, out of a 32 KB L0 that a wave's 2 KB footprint
// fits), and the request-count penalty is a small multiple rather than 16x. What the transpose
// would also permit is a `float4` per-thread row — 512 B in flight per wave-instruction
// against this kernel's 128 B — on the axis this tree has twice measured as the limiter. So
// judged as WRITTEN the two shapes are close, and [K][V] pulls ahead only once a thread owns
// four consecutive value channels, which is a change this kernel has not made. Read that as a
// reason to make it, not as a licence to leave the scalar mapping alone.
//
// **The reference stores it the other way round, and this kernel deliberately does not.** fla
// is called with `transpose_state_layout=True`, so the anchor's `initial_state` and
// `out.state` are [V][K] — measured rather than read off the kwarg, because the state is
// square (128x128 in the model, 32x32 in the anchor) and no shape assertion can see the
// difference: `tests/k3_kernels.rs` scores both interpretations of the anchor's own state and
// they separate by 7.567e-1 against the oracle's own 2.481e-7 agreement. Note whose layout
// this is: `k3-architecture.md` §4 step 7 writes `S[i][j]` with `i` the key channel, so the C
// reference AGREES with this kernel and what is declined is fla's BUFFER order.
//
// Nothing forces that order on the port, whose state starts at zero and never leaves the
// device, so the fixture transposes at its boundary and the decode path pays nothing. **Two
// costs it does have**, both named by review rather than by measurement: a live state dumped
// at S3 and diffed against the anchor's will load, shape-check and disagree by exactly the
// magnitude these docs file as the signature of `--defect KdaStateLayout`; and a chunked
// prefill can only be ported from fla, whose buffer is [V][K], so transliterating its index
// math against this layout is wrong with every shape check green.
//
// Everything fla hides inside its kernel is here, because the anchor's captures put it here:
// `q`/`k` arrive PRE-L2-norm, `beta` arrives PRE-sigmoid, and `g` is the raw gate projection
// with `A_log` and `dt_bias` unapplied (`anchor.md` §what is captured).
//
// USED BY: Kimi-K3 (S3's layer loop; no engine caller yet — see `tests/kernel_coverage.rs`).
extern "C" __global__ void gated_delta_recurrent_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta_pre,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower_bound,
    float* __restrict__ state, float* __restrict__ out) {
    // sh = [ qn | kn | alpha | reduction scratch ], each `d` floats. Dynamic rather than a
    // fixed [128] because the anchor runs head_dim 32 and the real model 128, and a kernel
    // whose LDS silently sized for one of them would be a ceiling nobody stated.
    extern __shared__ float sh[];
    // **`d` is head_k_dim AND head_v_dim, and nothing can check that they are equal.** Thread `t`
    // is key channel `t` for `q`, `k`, `g`, `dt_bias` and its `alpha`, and value channel `t` for
    // `v`, `out` and its private state column — while both state loops run `i` over the same `d`.
    // The launcher takes ONE width, so the equality is not "guarded" (an earlier comment here said
    // it was): it is structurally unexpressible, and the square state makes every shape check pass
    // either way. True for K3 at 128/128; a gated-delta-rule model with head_v_dim = 2*head_k_dim
    // would be silently WRONG here rather than refused, which is worth knowing given this file is
    // named for the arithmetic and invites the family.
    const int d = (int)blockDim.x;
    const int h = (int)blockIdx.x;
    const int t = (int)threadIdx.x;
    float* qn = sh;
    float* kn = sh + d;
    float* alpha = sh + 2 * d;
    float* red = sh + 3 * d;

    const size_t off = (size_t)h * d + t;
    const float qv = q[off];
    const float kv = k[off];

    // L2Norm on q and k ONLY — `v` is left alone — with `eps` added to the SUM of squares
    // rather than to the mean (§4 step 3). That is a different convention from every RMSNorm
    // in this tree, so it is spelled out rather than reusing one.
    //
    // The C reference accumulates the sum in a `double`; this does not, because the anchor is
    // fla's fp32 triton kernel and that is what the tolerance was measured against
    // (`kda_op`, Rel(6.3e-4), from chunk-vs-recurrent). At d <= 128 the fp32 sum is nowhere
    // near needing it. If a chunked prefill path ever sums over the sequence too, revisit.
    //
    // TWO reductions back to back, which is what `block_sum_lds`'s trailing `__syncthreads` was
    // added for: its comment says no shipped caller reduces twice and that the barrier "buys one
    // barrier per row against a landmine for the first caller that does". This is that caller.
    const float qsum = block_sum_lds(qv * qv, red);
    const float ksum = block_sum_lds(kv * kv, red);
    // The `d^-0.5` goes on q AFTER the norm and on nothing else (§4 step 6). It is computed
    // from `d` rather than passed in: fla's `scale` defaults to `head_k_dim ** -0.5` and the
    // reference passes no override, so a parameter here would be a knob with one legal value.
    qn[t] = qv * (1.0f / sqrtf(qsum + 1e-6f)) * (1.0f / sqrtf((float)d));
    kn[t] = kv * (1.0f / sqrtf(ksum + 1e-6f));

    // Decay (§4 step 5). `a_log` is PER HEAD, `dt_bias` per (head, channel), and the bias goes
    // on BEFORE the scale. `lower_bound` MULTIPLIES the sigmoid — it is not a clamp; fla's own
    // docstring gives this form for `safe_gate=True` (fla 0.5.2's `fla/ops/kda/chunk.py:250-256`, the version `tests/k3_anchor.rs` pins), which is
    // the written reference this term has and the `KdaGateLowerBoundOff` defect only proves the
    // golden is sensitive to. `alpha == 1.0f` exactly is LEGITIMATE saturation, meaning perfect
    // retention: in fp32 the sigmoid underflows to zero below about -87.3, so a kernel
    // asserting `alpha < 1` fires on valid input.
    const float a = expf(a_log[h]);
    const float z = a * (g[off] + dt_bias[(size_t)h * d + t]);
    alpha[t] = expf(lower_bound * (1.0f / (1.0f + expf(-z))));
    __syncthreads();

    const float beta = 1.0f / (1.0f + expf(-beta_pre[h]));
    float* s = state + (size_t)h * d * d;

    // Decay and `u = S^T k` in one pass over the state.
    //
    // **`#pragma unroll` on both state loops, and it is not a guess.** Un-pragma'd, each loop
    // issues one dword load and consumes it immediately, so a wave holds exactly ONE request in
    // flight per memory round-trip — the disease `common.hpp::dot_f4_wave_r` names in its own
    // comment and that measured **146.63 -> 195.27 GB/s (+33.2%)** on this hardware when it was
    // fixed there. The fold order is preserved (the adds stay sequential; only the loads hoist
    // above the chain), which is `v4-decode-decomposition.md`'s argument and is why this is
    // byte-identical rather than a re-association. **The K3 magnitude is NOT measured** — there is
    // no engine caller to measure end to end until S3 — so the 33.2% is the precedent's number and
    // not a claim about this kernel.
    float u = 0.0f;
#pragma unroll 4
    for (int i = 0; i < d; ++i) {
        const float sv = alpha[i] * s[(size_t)i * d + t];
        s[(size_t)i * d + t] = sv;
        u += kn[i] * sv;
    }
    // The prediction error, scaled by the update gate. `beta` reaches this kernel pre-sigmoid
    // because fla takes `use_beta_sigmoid_in_kernel=True`.
    const float dv = beta * (v[off] - u);

    // The rank-one update and `o = S^T q` in one more pass — and `o` reads the UPDATED state,
    // which is the ordering the delta rule is defined by. No barrier between the passes: every
    // thread touches only column `t`.
    float o = 0.0f;
#pragma unroll 4
    for (int i = 0; i < d; ++i) {
        const float sv = s[(size_t)i * d + t] + kn[i] * dv;
        s[(size_t)i * d + t] = sv;
        o += qn[i] * sv;
    }
    out[off] = o;
}

extern "C" int rivoli_gated_delta_recurrent_f32(const float* q, const float* k, const float* v,
                                                const float* g, const float* beta_pre,
                                                const float* a_log, const float* dt_bias,
                                                int heads, int head_dim, float lower_bound,
                                                float* state, float* out, void* stream) {
    if (heads <= 0 || head_dim <= 0) return 1001;
    // `head_dim` IS the block, so it inherits the launch limit. The LDS needs no clause of its
    // own: four rows of `head_dim` floats is 16 KB at that maximum, a quarter of gfx1151's 64 KB,
    // so an `> 65536` disjunct here could never fire and was decoration that read as a live bound
    // (review 2026-08-12). It would start to matter above head_dim 4096 or at a fifth LDS row.
    if (head_dim > 1024) return 1002;
    // `block_sum_lds` reduces by halving `blockDim.x`, which drops elements unless the block is
    // a power of two. 32 (the anchor's) and 128 (the model's) both are; refuse the rest rather
    // than return a silently short norm. Same code and same expression as `blockindex.hip`'s
    // `rotate` guard, for the same reason.
    if (head_dim & (head_dim - 1)) return 1003;
    // fla range-checks `-5 <= lower_bound < 0` and the shipped config sits exactly on the
    // inclusive end (`anchor.md` §the defect runs). Refusing outside it refuses what the
    // reference itself refuses — and the `!(...)` form is deliberate: it also rejects NaN,
    // which would make every `alpha` NaN and every output NaN-quiet, the shape `swiglu_limit`
    // guards in this tree keep being widened to catch.
    if (!(lower_bound >= -5.0f && lower_bound < 0.0f)) return 1006;
    hipGetLastError();
    hipLaunchKernelGGL(gated_delta_recurrent_f32, dim3((unsigned)heads),
                       dim3((unsigned)head_dim), 4 * head_dim * sizeof(float),
                       (hipStream_t)stream, q, k, v, g, beta_pre, a_log, dt_bias, lower_bound,
                       state, out);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
