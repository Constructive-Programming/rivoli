// The residual-stream MIXERS — what stands where a plain `x += y` residual would, for the
// two models that do not have one. DeepSeek-V4 carries `HC_MULT` residual copies and mixes
// them with learned Sinkhorn-normalised weights (`hc_pre`/`hc_post`); Kimi-K3 carries a
// stack of block snapshots and mixes them with a softmax over RMSNorm-scored dot products
// (`attn_res`). GLM's plain residual is `fwd.hip::vadd` and never comes through here.
//
// **Split out of `activation.hip` on 2026-08-16**, when K3's `attn_res` would have pushed
// that file past the 800-line cap. The mHC half moved WHOLE and verbatim — same symbols,
// same launcher signatures — so nothing outside `build.rs`'s KERNELS array can observe the
// move; `attn_res` is new with M9, ported verbatim from `k3:kernels/linalg.hip`. The cut is
// the one `activation.hip`'s old header already had to argue: these kernels' output IS the
// residual stream, where everything that stayed transforms the activation between
// projections.
//
// All pointers device-resident; launchers only launch (the caller joins via
// `linalg.hip::rivoli_device_sync`). Return codes: 0 ok, POSITIVE arg guard,
// NEGATIVE -(hipError_t).

#include "common.hpp"

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

// **Block Attention Residuals** — Kimi-K3's multi-residual-stream mixture, `k3-architecture.md`
// §3. `out = softmax(score) @ src`, where `score_s = <RMSNorm(src_s), fold>`.
//
// USED BY: Kimi-K3 only, twice per layer plus once model-level. GLM-5.2 and DeepSeek-V4-Flash
// have a plain residual and never reach this.
//
// **The softmax mixes the UNNORMALISED sources.** Normalisation exists only to score, and
// `--defect AttnResNormalisedValues` is the run that prices mixing `k` instead of `v` — it moves
// this operator by 1.80e0 against a 1.571e-5 fp32 floor, so the two readings are not close.
// `fold[i] = norm.weight[i] * proj.weight[i]`, collapsed at load time because the reference
// applies them consecutively and nothing between them depends on either alone.
//
// One block per token, `blockIdx.x` indexing `src`'s outer axis. Decode passes T=1; layer-major
// prefill keeps the whole prompt live (`architecture.md` §14) and passes all of it at once, which
// is the sizing §3 flags as a port decision rather than an inherited one.
//
// **`ss` and the dot product are reduced in ONE pass, and the scale applied once at the end.**
// The reference spells it `acc += (v[i] * inv) * fold[i]`, scaling every element before
// accumulating; `inv` is a scalar, so `inv * Σ(v·fold)` is the same number with one multiply
// instead of `n` — and strictly the more accurate of the two in fp32, since it rounds once rather
// than per element. It is NOT bitwise identical to the reference, which accumulates this loop in
// double regardless; that difference is what the operator's tolerance is for.
#define ATTN_RES_MAX_SRC 16

extern "C" __global__ void attn_res(const float* __restrict__ src, const float* __restrict__ fold,
                                    int nsrc, int n, size_t src_stride, float eps,
                                    float* __restrict__ out) {
    __shared__ float red_ss[256];
    __shared__ float red_dot[256];
    __shared__ float prob[ATTN_RES_MAX_SRC];
    const int t = threadIdx.x;
    // **`src_stride`, not `nsrc * n`.** §3's arena is `[T][9][hidden]` — NINE slots per token
    // whatever the current depth — so a token stride of `nsrc * n` reads token 1's stack out of
    // token 0's unused slots at every layer below 84, folding stale data into a plausible-looking
    // output. The kernel assumed the packed layout until an adversarial review 2026-08-12 traced
    // it against the spec; both fixtures built packed stacks, so nothing here could see it.
    // Passing the stride makes the arena shape the CALLER's statement and something a test can
    // vary, which is the only reason this is now checkable.
    const float* v0 = src + (size_t)blockIdx.x * src_stride;

    for (int s = 0; s < nsrc; s++) {
        const float* v = v0 + (size_t)s * (size_t)n;
        float ss = 0.0f, dot = 0.0f;
        for (int i = t; i < n; i += blockDim.x) {
            const float x = v[i];
            ss += x * x;
            dot += x * fold[i];
        }
        red_ss[t] = ss;
        red_dot[t] = dot;
        __syncthreads();
        for (int k = blockDim.x / 2; k > 0; k >>= 1) {
            if (t < k) {
                red_ss[t] += red_ss[t + k];
                red_dot[t] += red_dot[t + k];
            }
            __syncthreads();
        }
        // The score, not yet the probability — `prob` holds both in turn to spend one LDS array.
        if (t == 0) prob[s] = red_dot[0] / sqrtf(red_ss[0] / (float)n + eps);
        __syncthreads();
    }

    // Max-subtracted softmax over at most nine scores, computed redundantly by every thread. A
    // reduction for nine values would cost more in barriers than the arithmetic it saves, and
    // every thread needs every probability in the mixing loop below anyway.
    float m = prob[0];
    for (int s = 1; s < nsrc; s++) m = fmaxf(m, prob[s]);
    float z = 0.0f;
    for (int s = 0; s < nsrc; s++) z += expf(prob[s] - m);
    float p[ATTN_RES_MAX_SRC];
    for (int s = 0; s < nsrc; s++) p[s] = expf(prob[s] - m) / z;

    float* o = out + (size_t)blockIdx.x * (size_t)n;
    for (int i = t; i < n; i += blockDim.x) {
        float acc = 0.0f;
        // `v0`, the RAW source. Re-read rather than kept from the scoring pass: the stack is
        // nsrc*n floats (at the real width, 9*7168) and does not fit in LDS.
        for (int s = 0; s < nsrc; s++) acc += p[s] * v0[(size_t)s * (size_t)n + i];
        o[i] = acc;
    }
}

// One block per token. `nsrc` is REFUSED above ATTN_RES_MAX_SRC (1003) rather than clamped: the
// stack is one snapshot per `attn_res_block_size` layers plus the running prefix sum, so 93 layers
// at block 12 gives at most 9, and a larger value means the caller's block bookkeeping is wrong
// rather than that this kernel should grow. `nsrc == 0` is refused too — the reference's
// layer-level fold is guarded by an emptiness check on the caller's side (§3's layer loop), so an
// empty stack reaching here is that guard having gone missing, not a case to define away.
extern "C" int rivoli_attn_res(const float* src, const float* fold, int tokens, int nsrc, int n,
                               size_t src_stride, float eps, float* out) {
    if (n <= 0 || tokens <= 0) return 1001;
    if (nsrc <= 0 || nsrc > ATTN_RES_MAX_SRC) return 1003;
    // A stride shorter than the live stack would make consecutive tokens overlap.
    if (src_stride < (size_t)nsrc * (size_t)n) return 1005;
    hipGetLastError();
    hipLaunchKernelGGL(attn_res, dim3(tokens), dim3(256), 0, 0, src, fold, nsrc, n, src_stride,
                       eps, out);
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
// Takes a STREAM, unlike `activation.hip`'s `rmsnorm_single`/`swiglu`. rivoli's streams are created
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

