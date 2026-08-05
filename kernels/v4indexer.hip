// V4-Flash `Indexer.forward` (model.py:409-441) — the device-side scoring, and the
// Hadamard+fp4 "spread" its compressor shares with it.
//
// S2c-indexer of docs/investigations/v4-flash-port.md. Scored against `src/v4oracle/`'s
// `Oracle::indexer` by `tests/v4_indexer_kernel.rs`.
//
// WHAT IS HERE AND WHY IT IS ONE FILE. `rotate_activation` + `fp4_act_quant` is applied at
// TWO places in the reference — to the indexer's `q` rows (:420-422) and to its nested
// compressor's pooled rows (:374-376) — and the oracle factors them into one helper
// (`indexer_spread`) with the argument that "the Hadamard spread exists to make the fp4
// grouping well-conditioned, so applying one without the other is a different algorithm,
// not a partial one". `v4_indexer_spread` is that helper. `v4compress.rs::compress`
// launches it as the compressor's finish when `Geom::indexer` built the geometry, and
// `Geom::attention` cannot reach it.
//
// WHAT IS DELIBERATELY NOT HERE:
//
//   * The compressor's pooling. It is `kernels/v4compress.hip` and it is the SAME
//     arithmetic for both compressors — `Compressor.forward` branches only at the final
//     quantization (`if self.rotate:` at :374). Writing a second pooling kernel for the
//     indexer would duplicate the one piece of this path that is already verified.
//   * The causal mask and the top-k. `Oracle::indexer` exports the full pre-top-k score
//     matrix precisely so a consumer can be scored on it, and comparing that matrix is
//     STRICTLY STRONGER than comparing the selected sets: the shipped goldens are
//     set-invariant at `index_topk = 512` (the recorded "hole S3 inherits"), so a set
//     comparison accepts an arbitrarily wrong ranking while a score comparison cannot.
//   * `wq_b` and `weights_proj`. Both are ordinary projections already in the tree —
//     `v4_gemv_fp8` (wq_b ships fp8 + scale) and `v4_dense_gemm_bf16` (weights_proj is
//     bare bf16). A third GEMV here would be a third summation order to reconcile.
//
// DUPLICATION. `kernels/` is not watched by `build.rs`'s jscpd gate, so nothing mechanical
// stops this file re-deriving arithmetic another kernel has. It adds no format codec of its
// own: `f2e2m1`/`fp4_quant_roundtrip` went into `common.hpp` beside the fp8 pair they
// mirror, so there is one copy and it is where the next reader will look for it.

#include "common.hpp"

// The widest row `v4_indexer_spread` will rotate. `index_head_dim` is 128 in the shipped
// config; 256 leaves headroom without making the LDS request interesting (256 floats = 1 KB
// per block). Guard 1002 refuses anything larger rather than overrunning the tile.
#define V4I_D_MAX 256

// `model.py::rotate_activation` — the natural-order (Sylvester) Walsh-Hadamard transform,
// in place over `d` contiguous floats of LDS.
//
// THE BASIS ORDER IS THE SETTLED PART, and it was the port's highest-risk assumption until
// 2026-08-05: `rotate_activation` calls `fast_hadamard_transform.hadamard_transform`, which
// documents itself as `F.linear(x, scipy.linalg.hadamard(dim))` — Sylvester's construction,
// natural order, NOT sequency. `tests/v4_hadamard_basis.rs` carries the evidence chain and
// pins the oracle's implementation to an explicitly built Sylvester matrix. Getting it wrong
// would be invisible upstream (every candidate order is orthogonal, so the dot product is
// unchanged) and a different ranking downstream, because it permutes which coordinates share
// an fp4 scale group.
//
// BIT-IDENTICAL to `v4oracle::numerics::hadamard_rotate` by construction, not by tolerance.
// The butterfly visits the same operand pairs in the same stage order, and each element's
// value is produced by the same tree of f32 adds; the thread mapping changes only WHO
// evaluates a pair, never which pairs exist. Within a stage the pairs `(j, j+h)` are
// disjoint, so no thread reads an element another has written and one barrier per stage is
// the whole synchronization.
//
// `d` must be a power of two — the caller's guard 1003. The reference's `hadamard_transform`
// zero-pads a non-power-of-two width up to the next one, a fourth behaviour nothing here
// reproduces, so this refuses rather than silently transforming the wrong length.
//
// PRECONDITION: `sh` is fully written and the block has already barriered. "One barrier per
// stage is the whole synchronization" is true only given that — this helper barriers at the
// END of each stage, so the FIRST stage's reads are unsynchronized against whatever filled
// the tile. The one caller satisfies it; a second one must too.
__device__ __forceinline__ void v4i_hadamard(float* sh, int d) {
    for (int h = 1; h < d; h <<= 1) {
        for (int t = (int)threadIdx.x; t < (d >> 1); t += (int)blockDim.x) {
            // `t` enumerates butterflies; `off` is the position within a half-block and
            // `t - off` is `(t / h) * h`, so `j` is `(t / h) * 2h + off`. The mask works
            // because `h` is a power of two at every stage.
            int off = t & (h - 1);
            int j = ((t - off) << 1) + off;
            float a = sh[j], b = sh[j + h];
            sh[j] = a + b;
            sh[j + h] = a - b;
        }
        __syncthreads();
    }
}

// `Oracle::indexer_spread` (forward.rs:1130-1138) on one row: Hadamard, bf16 store, fp4
// block-32 simulation, bf16 store. One block per row.
//
// FOUR steps, and the bf16 store BETWEEN the rotation and the quantization is the one a port
// drops. It is not cosmetic: `fp4_act_quant` takes a per-block `amax` over exactly those
// values, so rounding after instead of before changes which scale each group of 32 gets, and
// the scale is the entire mechanism by which this stage affects the score at all.
// `rotate_activation` asserts a bf16 input in the reference, and `fp4_act_quant` receives a
// bf16 tensor, so both stores are the reference's own dtype and not an approximation of it.
//
// The row is staged through LDS because the Hadamard reads every element `log2(d)` times.
extern "C" __global__ void v4_indexer_spread(float* __restrict__ x, int d) {
    __shared__ float sh[V4I_D_MAX];
    float* row = x + (size_t)blockIdx.x * d;
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) sh[i] = row[i];
    __syncthreads();

    v4i_hadamard(sh, d);

    // `scale = x.size(-1) ** -0.5`, applied after the butterfly exactly as the oracle does.
    // `1.0f / sqrtf(d)`, NOT `rsqrtf`: the fast reciprocal square root is a 1-ulp
    // approximation on gfx1151 and the oracle computes `(d as f32).sqrt().recip()`, which is
    // a correctly-rounded sqrt and a true divide. hipcc is invoked without `-ffast-math`
    // (build.rs), so the divide survives. The factor multiplies every element of the row, so
    // a 1-ulp error in it does not average out.
    float scale = 1.0f / sqrtf((float)d);
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) sh[i] = rbf16(sh[i] * scale);
    __syncthreads();

    // `fp4_act_quant(row, 32, inplace=True)`. One thread owns a whole block of 32, so the
    // amax scan and the write-back are serial over it and in the oracle's own order — the
    // same mapping and the same argument as `mla.hip::v4_act_quant`. Parallelising inside a
    // block would re-associate an absmax that is exact today and buy nothing: the op moves
    // 512 bytes per row.
    int nb = d / FP4_QUANT_BLOCK;
    for (int b = (int)threadIdx.x; b < nb; b += (int)blockDim.x) {
        float* seg = sh + b * FP4_QUANT_BLOCK;
        // Seeded with the floor rather than folded from 0 and floored after. Identical for
        // `fmaxf`, which is associative here and returns the non-NaN operand either way, and
        // it is the idiom `v4_act_quant` already uses for the fp8 floor.
        float amax = FP4_QUANT_AMAX_FLOOR;
        for (int i = 0; i < FP4_QUANT_BLOCK; ++i) amax = fmaxf(amax, fabsf(seg[i]));
        float s = fast_round_scale(amax, FP4_MAX_INV);
        for (int i = 0; i < FP4_QUANT_BLOCK; ++i) seg[i] = fp4_quant_roundtrip(seg[i], s);
    }
    __syncthreads();

    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) row[i] = rbf16(sh[i]);
}

// `index_score = (einsum("bshd,btd->bsht", q, kv).relu_() * weights.unsqueeze(-1)).sum(2)`
// — model.py:425-427, one thread per (query row, compressed block).
//
// ONE THREAD FOR THE WHOLE (t, c) REDUCTION, which is a correctness choice and costs
// occupancy. Everything about this chain is order-sensitive:
//
//   * the `hd`-term dot is a SEQUENTIAL f32 fold in the oracle, so a wave-parallel
//     `wave_sum` here would re-associate it and the agreement would become a tolerance
//     instead of a bit result;
//   * the sum over `h` heads is a bf16 RUNNING fold — `acc = bf16(acc + bf16(dot * w))` —
//     so it cannot be split across lanes at all without changing the value.
//
// Every other kernel in this port re-associates something and pays for it in tolerance
// (`v4c_block_sum` and `wave_sum` both say so at their definitions). This one does not need
// to: at the shapes the model runs, `s * n_comp` threads each doing `h * hd` MACs is
// 512 threads x 8192 MACs at a 2048-token decode, which is small work in an awkward shape
// rather than a lot of work. If it ever becomes the bottleneck, the lever is a wave per
// (t, c) with lanes over `hd`, and the price is that this stops being bit-exact.
//
// THE bf16 STORES ARE THE POINT. `einsum` on bf16 tensors returns bf16, `relu_()` is in
// place on that bf16, `* weights` lands in bf16 (`weights_proj` is a bf16 `Linear`), and the
// oracle models the reduction as a bf16 running sum. This chain decides WHICH blocks are
// attended, so dropping a store leaves values that are fluent and slightly too precise — and
// a set comparison against the shipped goldens cannot see it, because at `index_topk = 512`
// the selection is determined entirely by the causal mask.
//
// KNOWN OPEN QUESTION, and this kernel matches the ORACLE rather than resolving it:
// PyTorch's `.sum(dim=2)` over a bf16 tensor accumulates through `acc_type`, i.e. in f32,
// and rounds ONCE at the end — not the per-term bf16 fold the oracle performs. If the oracle
// is wrong about that, this kernel is wrong with it and no comparison in this repo can see
// it. Recorded in the S2c-indexer report; not silently "fixed" here, because changing the
// oracle would invalidate every shipped golden and is not this stage's call.
// NO FMA CONTRACTION BELOW THIS LINE, and it is the difference between this kernel being
// bit-exact and merely close. `dot += q[i] * kv[i]` contracts to `v_fmac_f32` by default —
// hipcc is invoked with `-O3` only (build.rs), and HIP clang's default is
// `-ffp-contract=fast-honor-pragmas` — which rounds ONCE per term where the oracle's
// `dot += a * b` rounds twice, one per operation. The result stays finite and plausible, so
// nothing but a bit comparison would have shown it, and this file's whole claim is a bit
// comparison. `mla.hip`, `linalg.hip` and `attn.hip` each set this pragma for the same
// reason; found here by review before the kernel ever ran.
//
// Scoped to the scoring kernel: `v4_indexer_spread` above has no multiply-add to fuse (the
// Hadamard is pure adds and subtracts, and the fp4 scale is a single multiply), so widening
// the pragma would suppress nothing and would suggest it had.
#pragma clang fp contract(off)
extern "C" __global__ void v4_indexer_score(const float* __restrict__ q,
                                            const float* __restrict__ kv,
                                            const float* __restrict__ w,
                                            float* __restrict__ score, int s, int n_comp,
                                            int heads, int hd) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // `size_t`, matching the launcher's `(size_t)s * n_comp`. In `int` the product overflows
    // at ~2.1M entries — reachable at a long prefill, where `s` and `n_comp` both grow — and
    // a negative bound makes every thread return while `hipGetLastError` reports success.
    if (id >= (size_t)s * (size_t)n_comp) return;
    int t = (int)(id / (size_t)n_comp), c = (int)(id % (size_t)n_comp);
    const float* kvr = kv + (size_t)c * hd;
    float acc = 0.0f;
    for (int hh = 0; hh < heads; ++hh) {
        const float* qh = q + ((size_t)t * heads + hh) * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; ++i) dot += qh[i] * kvr[i];
        // bf16 FIRST, then `relu_()` — that is the order model.py:426-427 writes and the
        // order the oracle transliterates. Relu-then-round differs whenever the rounding
        // would have crossed zero, which is a small set and not an empty one.
        dot = rbf16(dot);
        dot = fmaxf(dot, 0.0f);
        acc = rbf16(acc + rbf16(dot * w[(size_t)t * heads + hh]));
    }
    score[id] = acc;
}

// ---------------------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------------------
// Return codes follow the tree's convention: 0 ok, POSITIVE argument guard, NEGATIVE
// -hipError. All pointers device-resident; launch-only, nothing synchronizes.

extern "C" int rivoli_v4_indexer_spread(float* x, int rows, int d, void* stream) {
    if (rows <= 0 || d <= 0) return 1001;
    // Three real limits, each able to fire, each exercised by
    // `the_spread_launcher_refuses_widths_the_hadamard_cannot_transform`. `V4I_D_MAX` bounds
    // the LDS tile; the power-of-two requirement is the Hadamard's, and `index_head_dim` is
    // the config field that would violate it; 1004 catches a width that is a power of two and
    // still SMALLER than the fp4 block, which would leave a ragged tail quantized against
    // nothing. An earlier comment called 1004 unreachable behind 1003 — it is not, `d = 16`
    // clears 1003 and fires 1004, and review caught the claim.
    if (d > V4I_D_MAX) return 1002;
    if (d & (d - 1)) return 1003;
    if (d % FP4_QUANT_BLOCK) return 1004;
    hipGetLastError();
    // One block per row; `d/2` butterflies at the widest stage, so 128 threads saturates
    // every stage at `d = 256` and wastes nothing at 128.
    hipLaunchKernelGGL(v4_indexer_spread, dim3((unsigned)rows), dim3(128), 0,
                       (hipStream_t)stream, x, d);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_v4_indexer_score(const float* q, const float* kv, const float* w,
                                       float* score, int s, int n_comp, int heads, int hd,
                                       void* stream) {
    if (s <= 0 || heads <= 0 || hd <= 0) return 1001;
    // NOT a no-op success. `n_comp == 0` is `end_pos / ratio` before the first block
    // completes — a real state the reference reaches — but there the reference produces an
    // EMPTY score tensor and never calls topk on it, so a caller that launched here believed
    // otherwise. Refusing names the disagreement; launching nothing would hand back an
    // unwritten buffer that reads as a row of zeros, which is a legal-looking score.
    if (n_comp <= 0) return 1005;
    size_t total = (size_t)s * (size_t)n_comp;
    hipGetLastError();
    hipLaunchKernelGGL(v4_indexer_score, dim3((unsigned)((total + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, q, kv, w, score, s, n_comp, heads, hd);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
