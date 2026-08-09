// The V4-Flash HEAD TAIL: the embedding gather, and `Block.hc_head`.
//
// Scored against `src/v4oracle/forward.rs` via `tests/v4_head_tail.rs`. Read that file's
// header before changing anything here — in particular, do NOT expect the gate to be
// bit-exact at real dimensions, and do not "fix" a kernel to make it so.
//
// WHAT IS NOT HERE, and why:
//
//   * the final `RMSNorm` — `kernels/mla.hip::rmsnorm_batch` already does exactly it, one
//     block per row, and it bf16-rounds on the way out as `RMSNorm.forward` does. The
//     other candidate, `kernels/linalg.hip::rmsnorm_single`, is `dim3(1)` — single-row — so
//     handing it `s x dim` would take ONE statistic over every token. That is a real
//     defect, modelled as `Defect::HeadNormOverAllTokens`, and it is invisible at decode
//     because `s == 1` there.
//
//   * the lm_head GEMV — `kernels/kvcompress.hip::gemm_bf16` already computes
//     `out[m,n] = x[m,k] . w[n,k]` for f32 activations against bf16 weights, which IS
//     `logits = head.weight @ x` at `m = 1, n = vocab_size, k = hidden`. A second kernel
//     differing only in extents would be pure duplication, and `kernels/` is NOT scanned
//     by jscpd (build.rs:619 covers `src`, `tests`, `build.rs` only), so nothing would
//     have caught it. `lm_head_is_the_existing_dense_gemm_at_vocab_shape` verifies the
//     mapping on device rather than asserting it here.
//
// EVERY HELPER COMES FROM `common.hpp` — `wave_sum`, `bf16f`, `rbf16`. None is redefined.
// That matters more than usual here, because `kernels/` is outside the duplication gate
// (`build.rs` covers `src`, `tests` and itself only) so nothing mechanical stops a third
// copy of a helper from appearing.
//
// CORRECTED 2026-08-09. This said `mla.hip::v4_rbf16` and `kvcompress.hip::v4c_rbf16` "are
// already two separate copies of `common.hpp::rbf16`". They WERE; both have since been
// folded into `common.hpp::rbf16` and neither identifier exists anywhere in `kernels/`. The
// warning stands, the example was stale — which is the same failure it describes, wearing a
// comment instead of a function.
#include "common.hpp"

// `Transformer.forward` lines 914-916: `h = embed(input_ids)`, then
// `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`.
//
// The expand is done HERE rather than by a follow-up copy because the copies are identical
// by construction and the read is the expensive part: one `[vocab, hidden]` row is fetched
// once and broadcast, instead of `hc_mult` passes over the same row.
//
// bf16 weights, not int8: V4's `embed.weight` is bf16 on disk and `convert_v4` keeps it
// that way (its `MODEL_LEVEL` marks it `Emit::Verbatim(Dtype::Bf16)`), so the int8 path
// `embed_i8_row` — which multiplies by a per-token scale — does not apply and there is no
// scale tensor to give it.
extern "C" __global__ void embed_bf16_row_bcast(const unsigned short* __restrict__ w, int token,
                                                int hidden, int hc, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    // `size_t`: at vocab 129280 x hidden 4096 the last row starts at 5.3e8, which still
    // fits `int`, but `token * hidden` in `int` overflows for any vocab past ~524k and
    // would silently read the wrong row rather than fault.
    float v = bf16f(w[(size_t)token * (size_t)hidden + i]);
    for (int c = 0; c < hc; ++c) x[(size_t)c * (size_t)hidden + i] = v;
}

// `Block.hc_head` part 1 (model.py:711-714): the per-copy gate vector.
//
// ONE WAVE PER TOKEN, deliberately. The reduction is over `hc * dim` = 16384 at real
// dimensions, and a block-wide reduction would need a shared-memory ladder — which
// `common.hpp::block_sum_lds` already implements. Re-implementing it here would be a third
// copy of a helper that is invisible to the duplication gate, so this uses `wave_sum` from
// `common.hpp` instead and accepts one wave's worth of parallelism. Decode runs a single
// token, so the occupancy this gives up is a prefill concern for S4, not a decode one.
//
// `scale` is `[1]` — ONE scalar broadcast over every mix. A `Block`'s `hc_*_scale` is `[3]`
// (pre, post, comb); indexing this one as if it were that reads past the tensor.
extern "C" __global__ void hc_head_collapse_gate(const float* __restrict__ h,
                                            const float* __restrict__ fnw,
                                            const float* __restrict__ base,
                                            const float* __restrict__ scale,
                                            float* __restrict__ pre, int s, int hc, int dim,
                                            float eps, float hc_eps) {
    int t = blockIdx.x;
    // The launcher sizes the grid at exactly `s`, so this cannot fire today. It is here
    // because the blend below bound-checks and this did not: an over-launched or grid-strided
    // caller would read `h` past its end and write a `pre` slot the blend then trusts.
    // Wave-uniform, so it does not break `wave_sum`'s full-wave requirement.
    if (t >= s) return;
    int lane = threadIdx.x;
    size_t hcd = (size_t)hc * (size_t)dim;
    const float* flat = h + (size_t)t * hcd;

    // `torch.rsqrt(x.square().mean(-1) + eps)` over the FULL flattened row — one statistic
    // for every mix, NOT one per hyper-connection copy. Getting that wrong type-checks and
    // produces the right shapes; it is `Defect::HeadHcRsqrtPerCopy`, and note that at real
    // dimensions its signal is the same order as this kernel's own re-association noise, so
    // the goldens CANNOT catch it — see tests/v4_head_tail.rs.
    float p = 0.0f;
    for (size_t i = lane; i < hcd; i += WAVE) p += flat[i] * flat[i];
    float rs = 1.0f / sqrtf(wave_sum(p) / (float)hcd + eps);
    // `wave_sum` leaves the total on lane 0 only; every lane needs `rs` for the dots below.
    rs = __shfl(rs, 0, WAVE);

    for (int j = 0; j < hc; ++j) {
        const float* wj = fnw + (size_t)j * hcd;
        float a = 0.0f;
        for (size_t i = lane; i < hcd; i += WAVE) a += flat[i] * wj[i];
        float m = wave_sum(a) * rs;
        // `torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps`. f32 throughout: every input
        // is an f32 parameter and there is no store to bf16 until the blend.
        //
        // `expf`, NOT `__expf`. The fast intrinsic carries ~2 ulp, which `moe.hip` states as
        // a rule, and `linalg.hip`'s hyper-connection gate — this same expression for this
        // same quantity in the non-V4 path — already uses `expf`. The oracle's `sigmoid`
        // uses a correctly-rounded `exp`, so the intrinsic would be an undeclared precision
        // deviation in a kernel whose whole justification is fidelity.
        if (lane == 0) pre[(size_t)t * hc + j] = 1.0f / (1.0f + expf(-(m * scale[0] + base[j]))) + hc_eps;
    }
}

// `Block.hc_head` part 2 (model.py:715-716): `sum(pre.unsqueeze(-1) * x, dim=2)`, then
// `y.to(dtype)`.
//
// One thread per output element, so the sum over `hc_mult` copies is SEQUENTIAL and in the
// oracle's own ORDER. It is NOT bit-identical to the oracle even so, and the reason applies to
// every kernel in this tree rather than just this one: build.rs passes hipcc only
// `--offload-arch -O3 -fPIC`, and clang's HIP default is `-ffp-contract=fast`, so `acc += p*x`
// becomes one FMA and the product is never rounded. MEASURED at dim 4096, s 8 (32768
// elements): 3 disagreements against a plain multiply-then-add, 0 against `mul_add`. So this
// is an FMA reduction, and `the_blend_is_bit_exact_given_the_gate` pins it against exactly
// that rather than against a rounding the build does not perform. `mla.hip` blocks the same
// contraction where it needs to; this does not need to, but the difference is stated rather
// than assumed away.
extern "C" __global__ void hc_head_collapse_blend(const float* __restrict__ h,
                                             const float* __restrict__ pre,
                                             float* __restrict__ y, int hc, int dim, int s) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= (size_t)s * (size_t)dim) return;
    size_t t = id / (size_t)dim, d = id % (size_t)dim;
    const float* flat = h + t * (size_t)hc * (size_t)dim;
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) acc += pre[t * hc + c] * flat[(size_t)c * dim + d];
    y[id] = rbf16(acc);
}

extern "C" int rivoli_embed_bf16_row_bcast(const unsigned short* w, int token, int hidden, int hc,
                                           float* x, void* stream) {
    if (hidden <= 0 || token < 0 || hc <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(embed_bf16_row_bcast, dim3((unsigned)((hidden + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, w, token, hidden, hc, x);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_hc_head_collapse(const float* h, const float* fnw, const float* base,
                                       const float* scale, float* pre, float* y, int s, int hc, int dim,
                                       float eps, float hc_eps, void* stream) {
    if (s <= 0 || hc <= 0 || dim <= 0) return 1001;
    hipGetLastError();
    // One wave per token for the gate; `wave_sum` requires a FULL wave, so the block is
    // exactly WAVE threads and every lane is active.
    hipLaunchKernelGGL(hc_head_collapse_gate, dim3((unsigned)s), dim3(WAVE), 0, (hipStream_t)stream, h,
                       fnw, base, scale, pre, s, hc, dim, eps, hc_eps);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    size_t n = (size_t)s * (size_t)dim;
    // Same stream, so the gate's writes precede the blend's reads without a host sync.
    hipLaunchKernelGGL(hc_head_collapse_blend, dim3((unsigned)((n + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, h, pre, y, hc, dim, s);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
