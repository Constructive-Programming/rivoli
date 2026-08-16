// The OCP MXFP4 routed-expert family: e2m1 nibble pairs, one e8m0 scale per 32 weights
// along the reduction dim, two-pass gate/up → down into the shared fixed-point `acc`.
// TWO models, TWO launchers, and the activation is where they part: DeepSeek-V4-Flash's
// pair (`rivoli_moe_expert_range_f4`) runs the clamped SwiGLU on an fp8-quantized
// activation with the routing weight folded into pass 1; Kimi-K3's pair
// (`rivoli_moe_expert_range_f4_situ`, M9, ported from `k3:kernels/moe.hip`) runs SiTU-GLU
// on a plain f32 activation with the routing weight applied in pass 2. Every difference is
// argued at the kernel it lives in.
//
// **Split out of `moe.hip` on 2026-08-16**, when the K3 pair would have pushed that file
// past the 800-line cap. The V4 half moved WHOLE and verbatim — same symbols, same
// launcher signature — so nothing outside `build.rs`'s KERNELS array can observe the move.
// GLM's VQ-int3 and int4 formats stay in `moe.hip`, together, because a HYBRID layer mixes
// them within one batch; nothing mixes either of them with FP4.

#include "common.hpp"

// ══ FP4 MoE — DeepSeek-V4-Flash's native routed experts ══════════════════════════════
//
// Same two-pass skeleton as the VQ and int4 paths, and FIVE differences from them, every
// one of which is silent-wrong if it is dropped. They are listed here because a reader who
// pattern-matches this onto the int4 twin in `moe.hip` will get all five wrong at once.
//
//  1. The ACTIVATION is fp8. `model.py::linear` quantizes `x` to e4m3 at block 128 with a
//     ue8m0 scale BEFORE the fp4 GEMM. This path therefore consumes an already-quantized
//     `x` (`activation.hip::act_quant_f8`), and `h` is re-quantized between the two passes.
//     Feeding raw activations is a ~2^-3 relative error on every product: fluent, wrong.
//  2. The SwiGLU is CLAMPED, and ASYMMETRICALLY — `up` to [-limit, +limit], `gate` from
//     ABOVE only (model.py:606-607). rivoli's SwiGLU is unclamped.
//  3. The ROUTING WEIGHT multiplies the SwiGLU intermediate, not the expert's output
//     (model.py:610). Algebraically the same; NOT the same here, because the reference
//     rounds the weighted intermediate to bf16 before `w2`.
//  4. bf16 lands between every step. `Linear` returns bf16, so `gate`, `up`, the weighted
//     intermediate and the expert output are each rounded — four roundings the GLM path
//     does not have.
//  5. `.f4` holds `n_experts` blocks and NO shared block: V4's shared expert is fp8 e4m3
//     at 128×128 and rides the resident path. A descriptor one past the end is not the
//     wrong bytes, it is the wrong ARITHMETIC — hence `n_desc` on the launcher.
//
// **Points 1-4 are the V4 pair's; Kimi-K3's pair below keeps the skeleton and swaps every
// one of them** — its own block comments carry the four substitutions.
//
// The six spans below are cast to `dot_f4.hpp::gu8p` (GLOBAL address space) at every use —
// the typedef's comment carries the measurement and the rejected alternatives. The cast is
// TRUE by construction: every span points at a device arena or heap allocation (f4gpu's
// placement, `tests/f4_kernel.rs`'s `DeviceBuf`s), and host GTT memory is still the global
// aperture on this APU; no LDS or scratch address can reach a descriptor. It asserts only
// the address SPACE — what the pointer may alias is untouched.

// Defined in activation.hip [CORRECTED 2026-08-16 with the move here: this said
// "linalg.hip", stale since the 2026-08-15 activation split]. Declared rather than called
// from Rust so the sequence gate/up → re-quantize → down cannot be issued with the middle
// step missing; see `rivoli_moe_expert_range_f4`.
extern "C" int rivoli_act_quant_f8(float* v, int n_rows, int row_len, void* stream);

struct ExpertDescF4 {
    const unsigned char* gate_packed;  // w1: [inter · hidden/2] e2m1 nibble pairs
    const unsigned char* gate_scale;   // [inter · f4_ng(hidden)] e8m0 bytes
    const unsigned char* up_packed;    // w3 — NOT w2; see quant.rs::V4_PROJ
    const unsigned char* up_scale;
    const unsigned char* down_packed;  // w2: [hidden · inter/2]
    const unsigned char* down_scale;   // [hidden · f4_ng(inter)]
};

// f4_groups inline (i_dim is the reduction dim of the projection) — matches
// quant.rs::f4_groups, the sizing `F4Expert::spans` lays the scale spans out with.
//
// CEIL here, FLOOR in the oracle (`WMat::Fp4::row` uses `cols / 32`). They disagree on a
// reduction dim that is not a multiple of 32, and the disagreement is a scale ROW STRIDE —
// every output row past the first would read the wrong scales, silently. What makes them
// agree is `rivoli_moe_expert_range_f4`'s `% ACT_QUANT_BLOCK` guard (128 % 32 == 0), which
// lives in neither expression; recorded here so a future non-128 reduction dim does not
// walk into it.
__device__ __forceinline__ size_t f4_ng(int i_dim) {
    return (size_t)((i_dim + F4_GROUP - 1) / F4_GROUP);
}

// The bf16 round-trip below is `formats.hpp::rbf16`. It was a local `bf16r` with an
// identical body until 2026-08-05 — `rbf16`/`bf16r` in one translation unit is a
// transposition between two names for the same thing, in a file where picking the wrong
// one would be invisible because both were correct. Every `Linear` in the reference
// returns bf16 and the residual stream is bf16, so this is where the reference's precision
// actually lives: dropping it is worth ~2^-8 relative per step, two orders above the
// summation-order floor the oracle's tolerance is built on.

// The row mapping and both projections — everything the two gate/up kernels share.
//
// **Factored 2026-08-12 (`k3:kernels/moe.hip`) because `kernels/` has no duplication
// gate.** `moe_gateup_f4_situ` was twelve lines of this restated verbatim with `hidden`
// spelled `expert_in`, and jscpd never scans this directory, so a divergence between the
// two would have been invisible to every tool here but the compiler. What the two kernels
// genuinely differ in is their ACTIVATION and their store, plus the parameters those need
// (`wexpert`/`limit` against the two betas), which is why they are still two kernels
// rather than one with a third template argument.
//
// Bit-neutrality is not assumed: the fp4 dispatch hash
// (`kernel_v4_fp4_decode.rs::the_fp4_dispatch_hash_is_a_tripwire_on_the_shared_clamp`, the
// M8 port of `old:tests/f4_kernel.rs`'s pin) holds this factoring to the same standard it
// held the `swiglu_clamped` hoist. Returns false when this wave has no row.
template <int R>
__device__ __forceinline__ bool f4_gate_up_dots(const float* __restrict__ x, int expert_in,
                                                int inter, int e_start, int e_count,
                                                const ExpertDescF4* __restrict__ descs, int& lane,
                                                int& e, int& j, Acc<R>& g, Acc<R>& u) {
    lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * inter) return false;
    e = e_start + (int)(r / inter);
    j = (int)(r % inter);
    const ExpertDescF4 d = descs[e];
    size_t rb = (size_t)(expert_in / 2), ng = f4_ng(expert_in);
    // `(gu8p)` on the four descriptor spans — flat_load → global_load, see the typedef in
    // dot_f4.hpp and the struct's own note for why the cast is true. Packed row and scale
    // row travel as one `F4Row`; the R token rows are `expert_in` apart.
    RowsView xv{x, expert_in};
    g = dot_f4_wave_r<R>(xv,
                         F4Row{(gu8p)d.gate_packed + (size_t)j * rb,
                               (gu8p)d.gate_scale + (size_t)j * ng, expert_in},
                         lane);
    u = dot_f4_wave_r<R>(
        xv, F4Row{(gu8p)d.up_packed + (size_t)j * rb, (gu8p)d.up_scale + (size_t)j * ng, expert_in},
        lane);
    return true;
}

// Pass 1: h[(e·R + t)·inter + j] = bf16(silu(min(bf16(w1_j·x), L)) · clamp(bf16(w3_j·x), ±L)
//                                       · wexpert[e·R + t])
//
// `x` must ALREADY be fp8-quantized (`act_quant_f8`) — this kernel cannot do it, see that
// kernel's note. `limit` is `swiglu_limit` from the config, passed rather than #defined so
// the kernel and `config.json` cannot drift; the launcher refuses every value that would
// DISABLE the clamp — `<= 0`, NaN and +/-inf alike — because an
// unclamped SwiGLU on a `.f4` expert is `v4oracle::Defect::SwigluUnclamped` and there is no
// V4 configuration that wants it.
template <int R>
__device__ __forceinline__ void moe_gateup_f4_impl(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescF4* __restrict__ descs, const float* __restrict__ wexpert, float limit,
    float* __restrict__ h_out) {
    int lane, e, j;
    Acc<R> g, u;
    if (!f4_gate_up_dots<R>(x, hidden, inter, e_start, e_count, descs, lane, e, j, g, u)) return;
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            // `common.hpp::swiglu_clamped` — bf16 both operands, clamp `up` both sides and
            // `gate` only from above, `x·sigmoid(x)`, return the product UNROUNDED. It was
            // these four statements inline here until 2026-08-05, hoisted when the fp8 SHARED
            // expert needed the identical arithmetic (`launch_swiglu_clamped_bf16`): the two
            // must agree bit for bit, `kernels/` is not watched by jscpd, and a second copy
            // would drift silently. The hoist was MEASURED bit-neutral, not assumed —
            // `tests/f4_kernel.rs::the_fp4_dispatch_hash_pins_the_clamp_hoist` carries the
            // hash and the argument.
            float sw = swiglu_clamped(g.v[t], u.v[t], limit);
            // ENFORCED, not inherited. `0 · finite` is exactly zero, but the gate has no
            // LOWER clamp, so a `w1` dot at -inf gives `silu = -inf · 0 = NaN`, and
            // `NaN · 0` is NaN, not 0. `moe_fixed` then turns that NaN into a FINITE extreme
            // (`fminf`/`fmaxf` return the non-NaN operand) — silent corruption of a row that
            // never asked for this expert. The int4 twin carries the same guard on the
            // same argument; folding the routing weight into pass 1 only moves the hazard a
            // stage earlier, it does not remove it.
            float w = wexpert[(size_t)e * R + t];
            // `sw * w`, and `swiglu_clamped` already associated as `(silu · up)`, so this is
            // the `((silu · up) · w)` the inline form computed. The bf16 store is HERE and
            // not in the helper because the reference rounds after the routing weight
            // (`weights * x` then `x.to(dtype)`), which the shared expert — having no
            // routing weight — does not do.
            h_out[((size_t)e * R + t) * inter + j] = w == 0.0f ? 0.0f : rbf16(sw * w);
        }
    }
}

// Pass 2: acc[t][o] += fixed(bf16(w2_o · h[e][t])), with the routing weight applied here or
// already folded into `h` depending on `WEIGHTED`.
//
// **The next three paragraphs are the `WEIGHTED == false` (DeepSeek-V4) arm only, and they were
// written before the other arm existed.** Scoped 2026-08-12 by review: every one of them is FALSE
// for `moe_down_f4_weighted`, and the third is the dangerous one — an S3 implementer who read it
// would insert `act_quant_f8` between K3's passes, which is precisely what
// `rivoli_moe_expert_range_f4_situ`'s own comment argues against.
//
// The routing weight is ALREADY in `h` (see the block comment's point 3), so unlike the VQ/int4
// twins the unweighted arm takes no `wexpert` — one source of truth for the routing, and no way
// for a mask to disagree with the data.
//
// A row that did not route to `e` gets `h == 0` exactly — `moe_gateup_f4_impl` ENFORCES that
// rather than deriving it (see the note there) — hence a dot of exactly 0.0 and
// `moe_fixed(0) == 0`. Adding it is a no-op, so riding through the arithmetic is bit-identical to
// skipping it. It is not free, but it is not the cost either: the weight read is, and that happens
// either way. **`moe_gateup_f4_situ` does NOT enforce it**: it writes a live activation for every
// expert in the range, and the weighted arm's `w != 0.0f` skip is what stands in.
//
// `h` must have been re-quantized to fp8 between the passes: `w2` is a quantized `Linear`
// like any other and quantizes its own input. **K3's `w2` does not** — §6 hands
// `k3_matmul_mxfp4` a plain fp32 activation, so the situ pair is two launches, not three.
//
// **`WEIGHTED` is Kimi-K3's half and it is ONE line, so it is argued here rather than left to
// the code.** `if (WEIGHTED)` is a template bool, so each instantiation compiles to exactly the
// arithmetic below it and the unweighted arm is bit-identical to what it was before the split.
// This is `moe_acc_drain`/`moe_acc_drain_to`'s pattern for the same reason: one difference the
// code cannot make visible, two names, both bodies readable side by side.
//
// **Why K3 needs the other arm at all.** DeepSeek-V4's `Expert.forward` applies the routing
// weight and THEN stores bf16 in front of `w2` (`weights * x`, then `x.to(dtype)`), so its pass 1
// folds `w` into `h` and this pass takes no `wexpert`. Kimi-K3 does the opposite: `moe_infer` ends
// `new_x.type(topk_weight.dtype).mul_(topk_weight).sum(dim=1)` on the expert's FULL output
// (`modeling_kimi_linear.py:867-874` at the pinned revision — the bare-colon form the tests use,
// spelled out once here because this is the first file under `kernels/` to cite it — and
// `k3-architecture.md` §6's `accL[i] += wt[j] * edn[i]`, where `edn` is
// the `w2` output). `w2` is linear, so `w2(w·h)` and `w·w2(h)` agree in exact arithmetic and the
// fold looks free — it is not, because there is a **bf16 store between them** and this pass
// accumulates in **fixed point**. `bf16(sw·w)` is not `bf16(sw)` scaled afterwards.
//
// So the order here is `rbf16(dv) * w`, NOT `rbf16(dv * w)`: the reference's `w2` produces bf16,
// and only then is the routing weight applied and summed. The VQ pass 2 has the weight in
// the same place and no `rbf16` at all, which is its own reference's shape — the three paths agree
// on none of this and each one's comment says which.
//
// The `w != 0.0f` skip is the VQ twin's, verbatim in argument: `0 * non-finite` is NaN and
// `moe_fixed`'s clamp turns NaN into a FINITE extreme, so it is silent corruption of a row that
// never asked for this expert, and skipping is bit-identical because adding zero to an integer
// accumulator is a no-op. **It is still needed here even though SiTU-GLU is bounded.** `situ_glu`
// maps ±inf to ±b1·b2 and only propagates a NaN it was given, so `h` cannot be the source — but
// `dv` is a fresh dot over `inter` terms of `h` against fp4 weights, and nothing bounds that.
template <int R, bool WEIGHTED>
__device__ __forceinline__ void moe_down_f4_impl(
    int hidden, int inter, int e_start, int e_count, const ExpertDescF4* __restrict__ descs,
    const float* __restrict__ h_in, const float* __restrict__ wexpert,
    unsigned long long* __restrict__ acc) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * hidden) return;
    int e = e_start + (int)(r / hidden);
    int o = (int)(r % hidden);
    const ExpertDescF4 d = descs[e];
    size_t rb = (size_t)(inter / 2), ng = f4_ng(inter);
    const float* he = h_in + (size_t)e * R * inter;
    Acc<R> dv = dot_f4_wave_r<R>(
        RowsView{he, inter},
        F4Row{(gu8p)d.down_packed + (size_t)o * rb, (gu8p)d.down_scale + (size_t)o * ng, inter},
        lane);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            const float y = rbf16(dv.v[t]);
            if (WEIGHTED) {
                const float w = wexpert[(size_t)e * R + t];
                if (w != 0.0f) atomicAdd(&acc[(size_t)t * hidden + o], moe_fixed(y * w));
            } else {
                atomicAdd(&acc[(size_t)t * hidden + o], moe_fixed(y));
            }
        }
    }
}

extern "C" __global__ void moe_gateup_f4(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescF4* __restrict__ descs, const float* __restrict__ wexpert, float limit,
    float* __restrict__ h_out) {
    moe_gateup_f4_impl<1>(x, hidden, inter, e_start, e_count, descs, wexpert, limit, h_out);
}

// **Kimi-K3's pass 1.** `situ_glu` in place of `swiglu_clamped`, and NO routing weight — §3b said
// "a new `moe_gateup_f4` variant", and the activation is the half it named. Three more differences
// it did not, each of which would pass every length check:
//
//  1. **No `wexpert` and no `w == 0` launder.** The weight belongs to pass 2 here
//     (`moe_down_f4_weighted`), so `h` is written for every expert in the range whether the row
//     routed to it or not. That is safe in a way V4's arrangement is not: `situ_glu` maps ±inf to
//     ±b1·b2 = ±100 and only propagates a NaN it was handed, so `h` is bounded even when a gate dot
//     overflows. The hazard moves to pass 2's multiply, where its guard is.
//  2. **No `limit`, and nothing to refuse.** SiTU-GLU has no clamp to disable — saturation is
//     `tanh`, which is in the function rather than in a parameter. The betas are refused instead,
//     at the launcher, and for the same class of reason.
//  3. **The store is `rbf16(y)` with no weight folded in**, which is the reference's
//     `KimiBlockSparseMLP.forward` handing `w2` a bf16 activation. V4 stores `rbf16(sw · w)`; those
//     are different values, not the same value written in a different order.
//
// `expert_in`, not `hidden` — for K3 this is the latent 3584 (§6). Same argument as
// `moe_down_f4_weighted`'s.
extern "C" __global__ void moe_gateup_f4_situ(
    const float* __restrict__ x, int expert_in, int inter, int e_start, int e_count,
    const ExpertDescF4* __restrict__ descs, float b1, float b2, float* __restrict__ h_out) {
    int lane, e, j;
    Acc<1> g, u;
    if (!f4_gate_up_dots<1>(x, expert_in, inter, e_start, e_count, descs, lane, e, j, g, u))
        return;
    // `situ_glu` is **two `tanhf` and one `expf`** where `swiglu_clamped` is one `expf`, and it is
    // evaluated under `if (lane == 0)` so 31 lanes are masked while the wave pays it. Priced by a
    // performance review at ~150-400 wave-cycles against a ~5,500-cycle per-wave budget: **~3-7% of
    // this pass and ~0% of a token**, since the pass is bandwidth-bound on 280 MB of expert weights.
    // Left alone deliberately — the exp2-identity rewrite of `tanhf` moves bits, and a tolerance is
    // the only thing holding this operator. Recorded so it is priced rather than rediscovered.
    if (lane == 0) h_out[(size_t)e * inter + j] = rbf16(situ_glu(g.v[0], u.v[0], b1, b2));
}

extern "C" __global__ void moe_down_f4(
    int hidden, int inter, int e_start, int e_count, const ExpertDescF4* __restrict__ descs,
    const float* __restrict__ h_in, unsigned long long* __restrict__ acc) {
    moe_down_f4_impl<1, false>(hidden, inter, e_start, e_count, descs, h_in, nullptr, acc);
}

// **Kimi-K3's pass 2**: the same fp4 down dot, with the routing weight applied to the bf16 `w2`
// output instead of having been folded into `h`. See the impl's `WEIGHTED` note.
//
// `expert_out` rather than `hidden`, because for K3 this dimension is the **latent** 3584 and not
// `cfg.hidden` 7168 — the experts run in latent space (§6). The shared kernel above calls the same
// parameter `hidden` because for GLM and V4 it is, and S1a item 3 renamed the Rust side of this
// distinction to `expert_in` for exactly this reason. A K3 caller binding `cfg.hidden` here passes
// every length check and computes a wrong row.
extern "C" __global__ void moe_down_f4_weighted(
    int expert_out, int inter, int e_start, int e_count, const ExpertDescF4* __restrict__ descs,
    const float* __restrict__ h_in, const float* __restrict__ wexpert,
    unsigned long long* __restrict__ acc) {
    moe_down_f4_impl<1, true>(expert_out, inter, e_start, e_count, descs, h_in, wexpert, acc);
}

// gate/up + the `h` re-quantization + down for the descriptor range
// [e_start, e_start+e_count) — the caller's placement, launch order since 2026-08-07 —
// on a caller-owned stream. THREE launches, not two:
// `act_quant_f8` between the passes is not an optimisation, it is `w2`'s own input
// quantization (see `moe_down_f4_impl`).
//
// `n_desc` is the length of `descs` — the guard `.f4`'s missing shared block needs, and the
// only argument this launcher has that its VQ and int4 twins do not. See point 5 of the FP4
// block comment above for why.
extern "C" int rivoli_moe_expert_range_f4(
    const float* x, int hidden, int inter, int e_start, int e_count, int n_desc,
    const void* descs, const float* wexpert, float swiglu_limit, float* h,
    unsigned long long* acc, int nrow, void* stream) {
    // `e_start < 0` is house style, matching the VQ and int4 twins, and is unreachable
    // through `launch_moe_expert_range_f4`, which takes a `usize`.
    if (hidden <= 0 || inter <= 0 || e_count <= 0 || e_start < 0 || n_desc <= 0) return 1001;
    // Both reduction dims must be whole ACT_QUANT_BLOCKs — which also makes them whole
    // F4_GROUPs, since 128 % 32 == 0. See `rivoli_act_quant_f8` for why a ragged tail is
    // refused rather than handled.
    if (hidden % ACT_QUANT_BLOCK || inter % ACT_QUANT_BLOCK) return 1002;
    // ONLY R=1 is instantiated. Not an oversight: the VQ and int4 twins carry R=2 because
    // speculative decode was MEASURED to pay there (1.108x at `--mtp-min-conf 0.8`), and no
    // such measurement exists for V4 — the S1b oracle is `bsz = 1` only, so a two-row FP4
    // kernel could not even be scored. `rivoli_moe_expert_range_i4` refused every nrow > 1
    // for the same reason until a caller existed, and that guard EARNED ITS KEEP on
    // 2026-07-31 by failing loudly at token 1 instead of silently computing a short row.
    if (nrow != 1) return 1003;
    if (e_start + e_count > n_desc) return 1004;
    // Every value that would DISABLE the clamp, refused. `x <= 0` would have admitted NaN;
    // `!(x > 0)` alone admits +inf, which is the same destination and quieter — `fminf(gt,
    // inf)` returns `gt`, so the clamp simply vanishes and the expert runs
    // `v4oracle::Defect::SwigluUnclamped` with no error anywhere. Found 2026-08-05, after the
    // one-sided form had been copied into `activation.hip::rivoli_swiglu_clamped_bf16` as a
    // deliberate match; both were fixed together.
    //
    // Rejects NaN (every compare against NaN is false), 0, negative, -inf and +inf. The
    // compare form follows `fwd.hip:224` for consistency and because it is one expression
    // covering all of them — NOT for the robustness that comment claims: `build.rs:67` gives
    // hipcc only `--offload-arch -O3 -fPIC`, so `-ffast-math` is not in play, and if it were,
    // `-ffinite-math-only` would fold `limit < INFINITY` as readily as `isfinite()`. Why
    // refusing at all: see `moe_gateup_f4_impl`.
    //
    // This is the launcher with REACHABLE callers, which is why it is the more urgent half:
    // the shared expert's twin has none yet.
    if (!(swiglu_limit > 0.0f && swiglu_limit < INFINITY)) return 1006;
    const ExpertDescF4* d = (const ExpertDescF4*)descs;
    hipStream_t s = (hipStream_t)stream;
    hipGetLastError();
    dim3 grid_a(((size_t)e_count * inter + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    hipLaunchKernelGGL(moe_gateup_f4, grid_a, dim3(256), 0, s, x, hidden, inter, e_start, e_count,
                       d, wexpert, swiglu_limit, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    // Only THIS range's `h` rows. Quantizing the whole buffer would race the experts other
    // streams are still writing, and would re-quantize rows a previous range already did —
    // idempotent only by luck, since the second pass's amax is the first pass's output.
    int rc = rivoli_act_quant_f8(h + (size_t)e_start * nrow * inter, e_count * nrow, inter, s);
    if (rc != 0) return rc;
    dim3 grid_b(((size_t)e_count * hidden + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    hipLaunchKernelGGL(moe_down_f4, grid_b, dim3(256), 0, s, hidden, inter, e_start, e_count, d, h,
                       acc);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// **Kimi-K3's routed expert range — TWO launches, not three.**
//
// `rivoli_moe_expert_range_f4` runs `act_quant_f8` between its passes and its comment insists that
// is "not an optimisation, it is `w2`'s own input quantization". True for DeepSeek-V4, whose `w2` is
// an fp8-quantized `Linear` that quantizes what it is handed. **K3's `w2` is not**: the routed
// expert weights are MXFP4 and the activation reaching them is plain fp32 —
// `k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, L, ...)` takes `act` directly (§6). Quantizing it here
// would introduce an error the reference does not have, in the one place the reference is exact.
// The fourth thing §3b did not say about this item.
//
// `expert_in` is the reduction dim of `w1`/`w3` AND the output dim of `w2` — 3584 for K3, both
// times, because the experts run in latent space and `[3072][3584]` transposes to `[3584][3072]`.
// One parameter is correct here and would not be for a model whose expert was not square in that
// sense; the name says which quantity it is rather than which config field a caller reached for.
//
// `acc` is the LATENT-wide accumulator `moe.hip::moe_acc_drain_to` drains (S1a item 4), not the
// residual — K3 norms and up-projects the aggregate before it becomes a residual at all.
extern "C" int rivoli_moe_expert_range_f4_situ(
    const float* x, int expert_in, int inter, int e_start, int e_count, int n_desc,
    const void* descs, const float* wexpert, float b1, float b2, float* h,
    unsigned long long* acc, int nrow, void* stream) {
    if (expert_in <= 0 || inter <= 0 || e_count <= 0 || e_start < 0 || n_desc <= 0) return 1001;
    // Both reduction dims must be whole F4_GROUPs — the scale rows are laid out per group and a
    // ragged tail would read past the last one. NOT `ACT_QUANT_BLOCK` as the fp8-activation twin
    // requires: nothing here quantizes an activation, so 128 is not a constraint this path has.
    // Both of K3's widths satisfy it either way (3584 and 3072 are 0 mod 128), which is exactly
    // why this needed stating rather than inheriting — the tighter check would never have fired.
    if (expert_in % F4_GROUP || inter % F4_GROUP) return 1002;
    // ONLY R=1, matching the fp8 twin: `num_nextn_predict_layers: 0`, so K3 has no speculative
    // decode and nothing batches (§3b). The guard earns its keep the way the int4 one did — by
    // failing loudly at token 1 rather than silently computing a short row.
    if (nrow != 1) return 1003;
    if (e_start + e_count > n_desc) return 1004;
    // The betas, refused rather than clamped — `activation.hip::rivoli_situ_glu_f32` carries the
    // full argument for all four rejected values. SiTU-GLU has no `swiglu_limit` and never should:
    // it has no clamp to disable, so there is nothing for a sentinel to spell.
    if (!(b1 > 0.0f && b1 < INFINITY) || !(b2 > 0.0f && b2 < INFINITY)) return 1006;
    const ExpertDescF4* d = (const ExpertDescF4*)descs;
    hipStream_t s = (hipStream_t)stream;
    hipGetLastError();
    dim3 grid_a(((size_t)e_count * inter + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    hipLaunchKernelGGL(moe_gateup_f4_situ, grid_a, dim3(256), 0, s, x, expert_in, inter, e_start,
                       e_count, d, b1, b2, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    dim3 grid_b(((size_t)e_count * expert_in + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    hipLaunchKernelGGL(moe_down_f4_weighted, grid_b, dim3(256), 0, s, expert_in, inter, e_start,
                       e_count, d, h, wexpert, acc);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
