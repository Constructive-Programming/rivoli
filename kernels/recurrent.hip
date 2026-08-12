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
// one key channel. That is what makes the [K][V] axis order the right one for this mapping;
// the transposed store would give each thread a private row and cost the coalescing.
//
// **The reference stores it the other way round, and this kernel deliberately does not.** fla
// is called with `transpose_state_layout=True`, so the anchor's `initial_state` and
// `out.state` are [V][K] — measured rather than read off the kwarg, because the state is
// square (128x128 in the model, 32x32 in the anchor) and no shape assertion can see the
// difference: `tests/k3_kernels.rs` scores both interpretations of the anchor's own state and
// they separate by six orders. Nothing forces that layout on the port, whose state starts at
// zero and never leaves the device, so the fixture transposes at its boundary and the decode
// path pays nothing.
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
    const int d = (int)blockDim.x;  // head_dim == head_k_dim; the launcher guards it
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
    // docstring gives this form for `safe_gate=True` (`fla/ops/kda/chunk.py:250-256`), which is
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
    float u = 0.0f;
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
    // `head_dim` IS the block, so it inherits the launch limit; the LDS is four rows of it.
    if (head_dim > 1024 || (size_t)4 * head_dim * sizeof(float) > 65536) return 1002;
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
