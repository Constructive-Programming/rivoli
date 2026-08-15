// Per-token activation transforms — what the forward pass does to the activation stream
// BETWEEN the projections: RMSNorm in its three forms, RoPE in its two, the SwiGLU
// combines, `act_quant`'s fused quantize-dequantize, and DeepSeek-V4's mHC residual
// mixing.
//
// **Split out of `linalg.hip` on 2026-08-15**, which was 1125 lines against an 800-line
// cap. Whole and verbatim: same symbols, same launcher signatures, same `common.hpp`, so
// nothing outside `build.rs`'s KERNELS array can observe the move.
//
// The cut is by what a kernel READS. `linalg.hip` contracts a token against a full weight
// MATRIX, so its cost is the weight's bytes and P5 (bytes/token is the currency) is what
// tuning it is about; these read the activation and cost what the activation costs — which
// is why none of them has a split-K variant and why their launch geometry is a function of
// `n`, never of a weight layout.
//
// `hc_pre` is the one member that has to be ARGUED rather than seen: it does contract
// against a learned `fnw`. At 1.57 MB per layer against lm_head's 952 MB that is a
// per-token reduction and not a projection, and its output is the residual stream itself,
// which is the axis this file is cut on. It sits here with `hc_post`, whose `post`/`comb`
// it produces — splitting the pair would put half a residual contract in each file.
//
// All pointers device-resident; launchers only launch (the caller joins via
// `linalg.hip::rivoli_device_sync`). Return codes: 0 ok, POSITIVE arg guard,
// NEGATIVE -(hipError_t) — `linalg.hip`'s convention, unchanged by the move.

#include "common.hpp"

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

// `Expert.forward`'s combine for a resident fp8 SHARED expert:
// h[i] = bf16( silu(min(bf16(g[i]), L)) · clamp(bf16(u[i]), ±L) ).
//
// USED BY: DeepSeek-V4-Flash-0731 only. GLM-5.2 uses `swiglu` directly above and has no
// `swiglu_limit`.
//
// **THE `_bf16` SUFFIX IS LOAD-BEARING AND THE `_clamped` HALF IS NOT ENOUGH.** Renamed from
// `v4_swiglu_clamped` on 2026-08-09 — the model name was wrong (this is an arithmetic
// contract, not a vendor's kernel) but it was doing one useful job by accident: it stopped
// the name reading as "`swiglu`, with a clamp". It is not that, and a reader who believes it
// is will reach for the wrong one. Hence `_bf16`, which names the difference that survives
// when `L` is taken away.
//
// Sits here, directly under `swiglu`, so that the two are readable side by side — because
// the interesting fact about them is that they are NOT the same function with a parameter,
// and that is not obvious from either one alone. At NO value of `L` do they agree. Beyond
// the clamp there are three differences, every one of them a defect the oracle can name:
//
//   * this bf16-rounds BOTH operands before combining (`Linear` stores bf16 and
//     `Expert.forward` reads it back with `.float()`) — `Defect::NoBf16Rounding`;
//   * this uses `F.silu`'s multiply form `x·sigmoid(x)`, not `swiglu`'s division form
//     `x/(1+e^-x)` — one rounding apart, which `common.hpp::swiglu_clamped` point 3 says
//     "would normally vanish under the bf16 store ... except exactly at a rounding
//     boundary". **UNDEMONSTRATED, and say so: measured 2026-08-05, swapping the two forms
//     in `moe_gateup_f4_impl` left the fp4 dispatch BIT-IDENTICAL across all 512 outputs.**
//     That is the predicted result for a fixture that misses the boundary — it does not
//     refute the reasoning, but no test in this tree exercises the case. An earlier draft of
//     this line dropped the conditional and read as though the difference were routine;
//   * this bf16-rounds the RESULT, which is the reference's `x.to(dtype)` before `w2`.
//
// **So adding an `L` parameter to `swiglu` is the specific refactor that must never happen.**
// Passing something meaning "unclamped" from GLM's four call sites could not have produced
// this kernel — it would have produced GLM's arithmetic under this kernel's name, which is
// the silent-wrong this whole port is built to avoid. That is a live hazard rather than a
// historical one: the two now differ by a suffix rather than by a vendor prefix, so the
// merge looks more reasonable than it did, not less.
// GLM has no `swiglu_limit` and never should: `rivoli_swiglu_clamped_bf16` refuses
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
extern "C" __global__ void swiglu_clamped_bf16(
                                               const float* __restrict__ g, const float* __restrict__ u, int n, float limit,
                                               float* __restrict__ h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = rbf16(swiglu_clamped(g[i], u[i], limit));
}

// y[i] = x[i]·(1/sqrt(mean(x²)+eps))·w[i] — RMSNorm, matching math.rs. One block
// reduces Σx² in LDS (tree order), then every thread scales.
//
// USED BY: GLM-5.2 (`gpu.rs`, one launch per row). DeepSeek-V4-Flash uses
// `mla.hip::rmsnorm_batch`.
//
// **`_single` MEANS ONE ROW, AND THAT IS THE WHOLE POINT OF THE NAME.** The grid is
// `dim3(1)`, so this takes ONE statistic over whatever it is handed. Passing it `s x dim`
// therefore normalises every token against a single shared mean instead of per row — a real
// defect, modelled as `v4oracle::Defect::HeadNormOverAllTokens`, and **invisible at decode
// because `s == 1` there**, which is why it needs a name rather than a comment.
//
// Renamed from bare `rmsnorm` on 2026-08-09 for exactly that reason. A name claiming the
// general operation while silently implementing the single-row case is the same failure as
// naming a mechanism after a model: it tells the reader something that is not true, and the
// caller who believes it gets fluent wrong text. `mla.hip::rmsnorm_batch` is the row-wise
// one; neither name is now the unmarked default, which is deliberate — there is no correct
// answer to "which rmsnorm did you mean" that a caller should be allowed to skip.
extern "C" __global__ void rmsnorm_single(
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

// y[i] = x[i]·(1/sqrt(mean(x²)+eps))·(1 + w[i]) — the CENTERED RMSNorm, one row.
//
// USED BY: Muse Glimmer's four per-layer sandwich norms, and nothing else in the tree.
//
// **`(1 + w)`, not `w` — and this model ships BOTH forms.** The four sandwich norms are centered
// with `w` initialised to ZEROS (§3); the final norm, the weightless qk_norm and the embedding
// norm are plain with `w` ones (§5). Substituting either way RUNS: a centered weight through the
// plain kernel scales the residual by ~0.15x and a plain weight through this one scales by ~2 —
// and BOTH run clean. §5 says the first "crashes into garbage"; §9 trap 5 says it "runs clean and
// produces a wrong model", and the anchor settles it: `norm_not_centered` leaves ZERO non-finite
// values across 1103 captures and emits seven tokens normally (measured 2026-08-12 by review).
// **Neither direction is self-announcing**, which is the actual case for a separate entry
// point rather than a flag, for the reason `rope_pair` gives below; here the wrong form is the
// silent one, so a bool would put it one argument from every GLM and V4 call site.
//
// **`block_sum_lds` rather than a hand-rolled ladder, and `rmsnorm_single` above keeps its own.**
// `headtail.hip`'s note states the rule — re-implementing the ladder would be another copy of a
// helper `kernels/` has no duplication gate to catch — and `mla.hip` already spells this exact
// line. The first draft of this kernel instead factored a `rms_inv_row` helper OUT of
// `rmsnorm_single`, which was a fourth spelling of `block_sum_lds` and put a hand on GLM's live
// decode path to share ten lines; review found both, 2026-08-12. Converting `rmsnorm_single` is
// declined separately: `block_sum_lds` carries a trailing barrier that spelling lacks, and
// `common.hpp` prices that as UNMEASURED on the decode path.
//

// Safe in place (`y` may alias `x`), and the argument needs one more clause than `swiglu`'s above.
// There, each thread reads its own index and writes its own, so the write depending on the read is
// the whole proof. Here loop 1 reads OTHER threads' indices — so the `__syncthreads()` inside
// `block_sum_lds` is load-bearing: every read of `x` precedes it and every write of `y` follows it,
// and `__restrict__` licenses no reordering across a barrier intrinsic. Loop 2's thread `i` then
// reads and writes the same index and no other thread touches it, which is `swiglu`'s clause again.
// `gpu.rs` and `f4gpu.rs` already launch `swiglu` in place (`h == g`), so this is the tree's
// established position on `__restrict__` and aliasing rather than a new one — stated HERE because
// three reviews read the qualifier and not the argument (2026-08-12), which is what a kernel that
// supports in-place owes its callers.
//
// `tests/glimmer_norm.rs` scores it against `tolerance::GLIMMER`'s `norm` row, measured before
// this existed, and the aliased launch bit-against-bit with the non-aliased one.
extern "C" __global__ void rmsnorm_centered_single(
    const float* __restrict__ x, const float* __restrict__ w, int n, float eps,
    float* __restrict__ y) {
    __shared__ float red[256];
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
    float inv = 1.0f / sqrtf(block_sum_lds(local, red) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * (1.0f + w[i]);
}

// Muse Glimmer's WEIGHTLESS QK-norm, over `d` = head_dim, one block per (row, head), IN PLACE —
// and the `scale` that follows it, folded in. `glimmer-architecture.md` sections 7 and 9:
//
//     q = qk_norm(q) * qk_scale_factor      # 3.87, Q ALONE            M:341
//     k = qk_norm(k)                        # normed, NOT scaled       M:342
//
// So Q passes `qk_scale_factor` and K passes `1.0f`. It costs one association change against the
// reference, which applies the scale to the norm's OUTPUT: `x*(s*rs)` here against `(x*rs)*s` there,
// one rounding apart at ~6e-8 — four decades under this operator's row.
//
// **The alternative worth arguing against is not a second pass over q — it is `gqa_attend`, which
// ALREADY takes a runtime `float scale`** (`attn.hip`, applied once per (row, head, kv row) at the
// dot product). Folding `qk_scale_factor / sqrt(head_dim)` into that argument is arithmetically the
// same value, commutes with RoPE because a rotation commutes with a scalar, costs ZERO new lines —
// `attn.rs` already computes `1/sqrt(d)` there — and would make trap 3 unrepresentable in this
// kernel, since K's scale would stop existing as a parameter. It is genuinely cheaper. Review,
// 2026-08-13, and the earlier version of this comment argued against a strawman second kernel pass
// instead, which is not what a reader will find when they go looking.
//
// It is declined because rivoli's q intermediate would then be 3.87x smaller than the golden's
// `q.pre_rope` bytes, and this port's whole method is elementwise tensor-vs-capture scoring — the
// fixture already treats a LAYOUT deviation as something that must be named at the call site, so a
// VALUE deviation on the tensor every future chain gate compares is the larger concession. That is
// the trade: one kernel parameter and one open trap, against every q gate from here to S4 carrying
// a constant in its fixture.
//
// **NOTHING GATES A CALLER THAT PASSES 3.87 FOR K, which is section 9's trap 3.** An earlier version
// of this comment said `tests/glimmer_qk_norm.rs` reddens on it; that is false and review caught it.
// Every scoring path there passes the SAME scale to the kernel and to the oracle, so no test can
// observe which one a caller chose, and there is no engine call site yet to be wrong. What the
// goldens DO settle is that the reference leaves K unscaled
// (`glimmer_anchor.rs::the_reference_scales_q_by_qk_scale_factor_and_leaves_k_alone`) — a fact about
// the reference, not a constraint on this caller. The anchor has no defect run for the damaging form
// either. **The gate belongs to the wiring item, and until it lands this argument is unguarded.**
//
// **The near-miss a reader will reach for first is twenty lines above: `rmsnorm_centered_single`.**
// With `w` all zeros — the form Glimmer's sandwich norms actually ship — its arithmetic IS this
// one's. **The geometry is NOT the reason to decline it**, and an earlier version of this comment
// said it was: `dim3(1)` and one statistic per launch are a `rows` parameter and a `blockIdx.x`
// stride away, which is a smaller diff than this kernel, so a reader who checks that argument finds
// it refutable in two lines and concludes the kernel was unnecessary. Review, 2026-08-13.
//
// The real reason is the one this file already argues for the plain-vs-centered split twenty lines
// up, and it transfers verbatim: neither direction is self-announcing at a call site. Reaching a
// WEIGHTLESS norm by feeding a centered kernel the weights that happen to make it plain encodes a
// coincidence of Glimmer's parameterisation — that its sandwich `w` initialises to zeros — as this
// operator's definition. QK-norm is weightless because the reference ships no tensor for it
// (`with_scale=False`), not because some tensor happens to be zero, and the next reader should not
// have to re-derive `(1+0) = 1` on the attention path to know which one they are looking at.
//
// The two distant ones fail on arithmetic instead. `rmsnorm_batch` multiplies a learned weight and
// STORES bf16 — there is no weight here (`with_scale=False`, trap 2) and a bf16 store is 2^-9 of
// systematic scale, 25x this operator's tolerance. `mla.hip::qk_norm` is exactly this shape but
// rounds through bf16 at four points, deliberately, on an inference `mla.hip` marks INHERITED,
// because DeepSeek-V4's reference does; Glimmer's is f32 throughout.
//
// One block per row, `block_sum_lds` for the ladder (the rule `headtail.hip` states), and
// `1.0f/sqrtf` rather than `rsqrtf` for `rmsnorm_batch`'s reason: this factor multiplies a whole
// head, so a hardware approximation is a systematic scale rather than a rounding difference.
// **No bar in this tree can see that choice, and the comment above used to imply one could.**
// `v_rsq_f32` is ~1 ulp, so swapping to `rsqrtf` would land near 2.6e-7 — 8x under this operator's
// own 2.0e-6 test bar and 300x under its row. So this is a defensible default, not an enforced
// property, and it is held to a weaker standard than the bf16 store two paragraphs up, which IS
// 25x the tolerance. Stated rather than gated because a bar tight enough to see 1 ulp would reject
// correct code on a different compiler. Review, 2026-08-13.
//
// **IN PLACE, and `x` is DESTROYED.** The pre-norm q/k does not survive this call, and the sibling
// `rmsnorm_centered_single` twenty lines up is out of place — so a caller migrating between them
// silently converts a read-preserving operator into a destructive one, with no shape change in the
// signature to notice. The aliasing itself is safe here for the two-clause reason the sibling
// spells out: every read of `row` precedes `block_sum_lds`'s first `__syncthreads()`, and the second
// loop's thread `i` reads and writes only its own index. What that does NOT cover is a consumer
// enqueued AFTER this launch that expected the pre-norm values — a `--trace` or `--pred-probe`
// readback of q is the realistic one, and no fixture can see it. Review, 2026-08-13.
extern "C" __global__ void rmsnorm_weightless_batch(float* __restrict__ x, int d, float eps,
                                                    float scale) {
    __shared__ float red[256];
    float* row = x + (size_t)blockIdx.x * d;
    float p = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) p += row[i] * row[i];
    float rs = scale / sqrtf(block_sum_lds(p, red) / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) row[i] *= rs;
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
// **HC_MULT IS A CHECKPOINT CONSTANT WEARING A GENERIC NAME, AND THAT IS THE INVERSE OF THE
// HAZARD THE `v4_*` RENAMES OF 2026-08-09 ADDRESSED.** Naming a reusable mechanism after the
// model that introduced it tells the reader it is narrower than it is; naming a model's own
// constant generically tells them it is wider. Both put the same wrong idea in the same
// reader's head, so this one is documented rather than renamed:
//
// USED BY: DeepSeek-V4-Flash-0731 only, at the value 4. `hc_split_sinkhorn`'s
// `mix_hc = (2 + hc)·hc` layout is how the mHC weights are PACKED on disk, so **another
// value is a different checkpoint, not a tuning knob** — there is no configuration of this
// engine in which changing it is meaningful. The launchers refuse anything else rather than
// pretending to generalise.
//
// `headtail.hip::hc_head_collapse_gate` takes the same quantity as a runtime `hc` parameter, and
// that is not an inconsistency to fix in either direction: the collapse genuinely is
// width-agnostic, and this packing genuinely is not.
#define HC_MULT 4
#define HC_MIX ((2 + HC_MULT) * HC_MULT)

// The width `hc_pre`'s sum-of-squares reduction is FROZEN at, independent of blockDim.
// That phase's arithmetic depends on block shape twice over — a blockDim-strided comb
// into a blockDim-leaf LDS tree — so when the launch widened from 256 to 1024 threads
// (2026-08-08, the M5 hcn fix; docs/investigations/v4-decode-decomposition.md) the stride
// and the tree kept the old geometry, or every historic golden and the byte-identity A/B
// gate would have moved. Threads ≥ HC_RED idle through this phase; the mixes phase below
// is where the extra waves earn their launch.
#define HC_RED 256

// `kernel.py::hc_split_sinkhorn` for one token, on ONE thread.
//
// Serial because it is 24 inputs and a 4x4 matrix: the whole routine is ~640 flops, and a
// parallel version would have to reproduce the reference's summation order across lanes to
// stay comparable at all. Note the asymmetry a port loses first — the FIRST normalisation
// is a row *softmax* followed by a column divide, and only the remaining `iters - 1`
// passes are plain row/column divides.
//
// `iters` is a parameter, not a constant, so the count comes from `config.json` through
// `V4Config` and cannot drift. What the kernel tests gate is that the parameter is live —
// 2 vs 20 moves the output.
//
// CORRECTED 2026-08-07: this said the count is "NOT gated by any numerical comparison: at
// 20 passes a 4x4 positive matrix is far past convergence, so 19 and 20 agree bit-for-bit
// and no golden can tell them apart". That holds for the toy fixture the kernel tests use,
// not for the checkpoint, where 19 vs 20 moves 39,893/53,248 of `L0.pre.ffn_norm_out`. See
// `tests/v4_oracle.rs::sinkhorn_has_converged_long_before_iteration_20`.
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
    __shared__ float red[HC_RED];
    __shared__ float mixes[HC_MIX];
    __shared__ float pre[HC_MULT], post[HC_MULT], comb[HC_MULT * HC_MULT];
    int t = threadIdx.x, nt = blockDim.x;
    size_t hcd = (size_t)HC_MULT * dim;
    const float* flat = h + (size_t)blockIdx.x * hcd;

    // Sum of squares at HC_RED wide, NOT nt: stride and tree are this phase's arithmetic
    // (see HC_RED above). The unroll is schedule-only — the acc chain keeps its order, the
    // compiler hoists the now-independent loads; at 256 threads the stock ISA held ONE load
    // per `s_waitcnt vmcnt(0)`, so latency-hiding depth was the whole cost. Depth 8 is
    // UNTUNED: the first depth checked already clustered the loads in the emitted ISA
    // (`s_clause` batches, staggered `vmcnt(7..0)`), here and in the mixes loop below.
    if (t < HC_RED) {
        float local = 0.0f;
#pragma unroll 8
        for (size_t i = t; i < hcd; i += HC_RED) local += flat[i] * flat[i];
        red[t] = local;
    }
    __syncthreads();
    for (int s = HC_RED / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float rs = 1.0f / sqrtf(red[0] / (float)hcd + norm_eps);

    // One wave per mix row — ONE round at 1024 threads (HC_MIX = 24, 32 waves; 8 idle).
    // Which wave computes row `j` is not part of the arithmetic: the row's unit is the
    // lane-strided comb + `wave_sum` ladder below, bit-identical wherever it runs. At 256
    // threads this loop was the measured hcn sink — 8 waves × 3 serial rows, ~2 KB in
    // flight against 1.57 MB of `fnw` (M5's read: ~350–400 µs of the ~400 µs launch).
    int lane = t & (WAVE - 1), w = t / WAVE, nw = nt / WAVE;
    for (int j = w; j < HC_MIX; j += nw) {
        const float* wr = fnw + (size_t)j * hcd;
        float acc = 0.0f;
#pragma unroll 8
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

// The rotation itself, shared by the two kernels below. Both compute the SAME pair rotation at
// the same frequency; they differ only in which two elements of the segment are the pair, which
// is what "interleaved" and "split-half" name. Double angles are the long-context arg reduction.
static __device__ inline void rope_pair(
    float a, float b, int j, int seg, int pos, double theta, float* lo, float* hi) {
    double inv = pow(theta, -2.0 * (double)j / (double)seg);
    double ang = (double)pos * inv;
    float cs = (float)cos(ang), sn = (float)sin(ang);
    *lo = a * cs - b * sn;
    *hi = b * cs + a * sn;
}

// **TWO KERNELS AND NOT ONE WITH A FLAG, deliberately.** They differ by one read pattern, and
// `common.hpp`'s `swiglu`/`swiglu_clamped_bf16` note is the precedent: when two callers want
// different arithmetic under one name, the way to keep the wrong one from being spelled is to
// make it unspellable THROUGH THE ENTRY POINT, not to add a parameter and hope. Here the wrong
// spelling is `glimmer-architecture.md` §9 trap 9 — interleaved where split-half is meant —
// which produces fluent text and no error. A `bool split_half` on one launcher would put that
// trap one argument away from every existing GLM and V4 call site. The arithmetic is shared
// above, so this costs a read pattern, not a copy.

// Interleaved RoPE (matching attn.rs) on `count` segments of `seg` elements:
// segment s at base[s*stride..+seg]; pair (2j,2j+1) rotates by pos·theta^(-2j/seg)
// into (j, half+j). One block/segment, seg/2 threads; all pairs read before any write
// (barrier) → race-free in place. GLM-5.2 and DeepSeek-V4-Flash.
extern "C" __global__ void rope_interleave(
    float* __restrict__ base, int stride, int seg, int pos, double theta) {
    int s = blockIdx.x;
    int j = threadIdx.x;
    int half = seg / 2;
    float* v = base + (size_t)s * stride;
    float a = v[2 * j], b = v[2 * j + 1];
    __syncthreads();
    rope_pair(a, b, j, seg, pos, theta, &v[j], &v[half + j]);
}

// Split-half RoPE — transformers' `rotate_half`, where the pair is `(x[j], x[j+half])` rather
// than two adjacent elements. Same frequency, same output positions; only the READ moves.
// Muse Glimmer, whose `apply_rotary_pos_emb` is this convention (`glimmer-architecture.md` §6).
//
// §6 proposed avoiding this kernel by permuting `q_proj`/`k_proj` rows at conversion time, and
// `tests/glimmer_rope.rs` measured that route working at 1.41e-7. It was declined on 2026-08-12
// for a cost the proposal did not price: the permutation forces `q_proj`/`k_proj` out of
// `copy_verbatim`, so ~3 GB of the artifact stops being a borrowed mapping of the checkpoint and
// G1a's "round-trips bit-exact" clause stops covering them. Four lines of kernel is the cheaper
// side of that trade. The equivalence stays measured, and stays in that test.
extern "C" __global__ void rope_split_half(
    float* __restrict__ base, int stride, int seg, int pos, double theta) {
    int s = blockIdx.x;
    int j = threadIdx.x;
    int half = seg / 2;
    float* v = base + (size_t)s * stride;
    float a = v[j], b = v[half + j];
    __syncthreads();
    rope_pair(a, b, j, seg, pos, theta, &v[j], &v[half + j]);
}

// ── launchers (one per kernel above, in the same order) ───────────────────────

// `stream` is trailing and nullptr is the null stream, the same migration contract as
// `mla.hip`'s V4 launcher set: this runs inside V4's shared-expert chain, whose other
// four launches are stream-parameterised, and a null-stream member in a stream-ordered
// chain is an unordered activation (see that set's note). GLM's dense MLP passes null
// and is unchanged to the byte.
extern "C" int rivoli_swiglu(const float* g, const float* u, int n, float* h, void* stream) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(swiglu, dim3((n + 255) / 256), dim3(256), 0, (hipStream_t)stream, g, u, n,
                       h);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// Takes a STREAM, as `rivoli_swiglu` directly above does since M7. Same reason `rivoli_hc_pre` gives
// below: rivoli's streams are `hipStreamNonBlocking` (`async.hip`), so the null stream does
// not implicitly order against them, and this runs between two `rivoli_gemv_fp8_bf16` launches
// that are now stream-parameterised. A null-stream combine between two stream-ordered GEMVs
// reads `g`/`u` before they are written — an unordered activation, which is a silent wrong.
extern "C" int rivoli_swiglu_clamped_bf16(const float* g, const float* u, int n, float limit,
                                          float* h, void* stream) {
    if (n <= 0) return 1001;
    // 1006, and deliberately the SAME code AND the same expression as `moe.hip`'s fp4
    // launcher, for the same check on the same argument — a reader who learned it there gets
    // it here. Both refuse every value that would disable the clamp, and getting to that took
    // two passes:
    //
    //   * `limit <= 0.0f` admits NaN, because every comparison against NaN is false. A NaN
    //     limit makes `fminf(gt, limit)` return `gt` — `fminf` returns the non-NaN operand —
    //     so it degrades silently to the unclamped form rather than producing NaN output
    //     someone would notice. Hence `!(limit > 0.0f)`.
    //   * `!(limit > 0.0f)` admits +INFINITY, which is the SAME destination and quieter still:
    //     `fminf(gt, inf)` is `gt` and `fmaxf(ut, -inf)` is `ut`, so the clamp vanishes
    //     entirely. Found 2026-08-05 — the first bullet closed the case in mind and left the
    //     one beside it open, which is the shape this port keeps producing.
    //
    // The compare form rather than `isfinite()`, following `fwd.hip:224` — but NOT for the
    // reason that comment gives. It says the compare form "avoids depending on `-ffast-math`
    // having left `isfinite()` meaningful", and neither half holds here: `build.rs:67` passes
    // hipcc only `--offload-arch -O3 -fPIC`, so `-ffast-math` is not in play at all, and if it
    // were, `-ffinite-math-only` would fold `limit < INFINITY` to `true` exactly as readily as
    // it folds `isfinite()`. The compare form is kept for consistency with `fwd.hip` and
    // because it is one expression covering all four rejected values; it buys no extra
    // robustness. Corrected 2026-08-05 — the code was right and the reason was not.
    //
    // **This guard is the ONLY thing rejecting +inf, and an earlier version of this comment
    // said the opposite.** It claimed `V4Config::validate` required a finite positive
    // `swiglu_limit` upstream, making this the redundant half. It does not:
    // `artifact/model.rs:726` is `ensure!(self.swiglu_limit > 0.0, ...)`, which rejects NaN
    // and zero and **admits +inf** — the identical one-sided spelling, in the identical
    // shape, one layer up. Checked 2026-08-05 rather than assumed, which is the only reason
    // this line is not still wrong. The upstream hole is recorded in
    // `docs/investigations/v4-flash-port.md`; fixing it is `artifact/`'s call, and until it
    // happens this launcher is the whole defence.
    if (!(limit > 0.0f && limit < INFINITY)) return 1006;
    hipGetLastError();
    hipLaunchKernelGGL(swiglu_clamped_bf16, dim3((unsigned)((n + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, g, u, n, limit, h);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_rmsnorm_single(const float* x, const float* w, int n, float eps, float* y) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rmsnorm_single, dim3(1), dim3(256), 0, 0, x, w, n, eps, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// The centered form, one row. `stream` is trailing and nullptr is the null stream, the
// `rivoli_swiglu` contract — this sits INSIDE a layer between two stream-ordered launches from
// the day S3 writes the loop, so taking it now costs nothing and adding it later is the
// unordered-activation bug `rivoli_swiglu_clamped_bf16` describes.
//
// `eps` is refused unless positive and finite — this model has two eps three decades apart (§3),
// so it is never a constant here, and a zero or a NaN comes from a config never read. Spelled as
// one compare to match `rivoli_swiglu_clamped_bf16`'s `limit` guard 30 lines up.
extern "C" int rivoli_rmsnorm_centered_single(
    const float* x, const float* w, int n, float eps, float* y, void* stream) {
    if (n <= 0) return 1001;
    if (!(eps > 0.0f && eps < INFINITY)) return 1002;
    hipGetLastError();
    hipLaunchKernelGGL(rmsnorm_centered_single, dim3(1), dim3(256), 0, (hipStream_t)stream, x, w,
                       n, eps, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// `rows` heads of `d` f32, normed in place and scaled, on `stream`.
//
// Three guards, and `scale` earns its place for `eps`'s reason: it comes from a config field
// (`qk_scale_factor`), and a zero or non-finite one silently multiplies every Q head by nothing or
// by garbage. **They reject the unusable, not the implausible** — `eps` = 1e30 and `scale` = 1e-45
// both pass, and a numeric range that rejected them would be the guard restating the config.
extern "C" int rivoli_rmsnorm_weightless_batch(float* x, int rows, int d, float eps, float scale,
                                               void* stream) {
    if (rows <= 0 || d <= 0) return 1001;
    if (!(eps > 0.0f && eps < INFINITY)) return 1002;
    if (!(scale > 0.0f && scale < INFINITY)) return 1003;
    hipGetLastError();
    hipLaunchKernelGGL(rmsnorm_weightless_batch, dim3((unsigned)rows), dim3(256), 0,
                       (hipStream_t)stream, x, d, eps, scale);
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

// One block per token, 1024 threads — one WAVE per `HC_MIX = 24` mix row, in one round.
// 256 until 2026-08-08: at decode (s = 1) that was ONE 8-wave workgroup for the whole
// device, and M4 measured the hcn bucket at 41.2 ms/token against 0.8 ms of bytes — the
// mixes GEMV walking 3 rows per wave with two loads in flight per `s_waitcnt vmcnt(0)`.
// The sum-of-squares phase stays at `HC_RED` (see its comment — that freeze is what makes
// the widening schedule-only). 24 waves is the parallelism CEILING here: a row's reduction
// order is one lane-strided comb + one `wave_sum` ladder, so a second wave cannot share a
// row without moving bits; more width has to come from splitting the kernel, not the block.
//
// Takes a STREAM, unlike `rmsnorm_single`/`swiglu` above. rivoli's streams are created
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
    hipLaunchKernelGGL(hc_pre, dim3(s), dim3(1024), 0, (hipStream_t)stream, h, fnw, scale, base,
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

extern "C" int rivoli_rope_interleave(float* base, int count, int stride, int seg, int pos, double theta) {
    if (count <= 0 || seg <= 0 || (seg & 1) || (seg / 2) > 1024) return 1001;
    if (count > 1 && stride < seg) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rope_interleave, dim3(count), dim3(seg / 2), 0, 0, base, stride, seg, pos, theta);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// Same guards as its sibling — the launch geometry is identical and only the kernel differs.
extern "C" int rivoli_rope_split_half(float* base, int count, int stride, int seg, int pos, double theta) {
    if (count <= 0 || seg <= 0 || (seg & 1) || (seg / 2) > 1024) return 1001;
    if (count > 1 && stride < seg) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(rope_split_half, dim3(count), dim3(seg / 2), 0, 0, base, stride, seg, pos, theta);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
