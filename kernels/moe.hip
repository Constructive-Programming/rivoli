// Fused VQ-int3 MoE expert batch (descriptor-array form). One launch computes a
// whole batch of experts sharing input `x` and the (hidden, inter) shape:
//
//     out[o] += Σ_e w[e] · down_e( silu(gate_e·x) ⊙ up_e·x )[o]
//
// Each expert's six VQ tensors are addressed by device pointers in a per-expert
// descriptor, so experts live at arbitrary offsets (resident shared or cold stream
// slots) with no gather/repack. Per-projection codebooks: gate/up/down each decode
// against their own codebook (down_proj's weight distribution differs). The routed
// batch and the resident shared expert fold into ONE launch (shared appended,
// weight 1.0). Composes dot_vq_wave (common.hpp) — no LDS staging (measured to
// regress: 64 KB LDS caps occupancy, the codebook stays L2-hot).

#include "common.hpp"

struct ExpertDescVq {
    const unsigned char*  gate_indices; // [inter · vq_row_bytes(hidden)]
    const unsigned short* gate_scales;  // [inter · vq_groups(hidden)] bf16
    const unsigned char*  up_indices;
    const unsigned short* up_scales;
    const unsigned char*  down_indices; // [hidden · vq_row_bytes(inter)]
    const unsigned short* down_scales;  // [hidden · vq_groups(inter)]
};

__device__ __forceinline__ float siluf(float x) {
    return x / (1.0f + expf(-x));
}

// vq_row_bytes / vq_groups inline (i_dim is the reduction dim of the projection).
__device__ __forceinline__ size_t vq_rb(int i_dim) {
    return (size_t)(i_dim / VQ_DIM) * VQ_INDEX_BITS / 8;
}
__device__ __forceinline__ size_t vq_ng(int i_dim) {
    return (size_t)i_dim / VQ_GROUP;
}

// --- BATCHING OVER TOKEN ROWS (`R`) ---------------------------------------------------
//
// Everything below computes `R` token rows through ONE read of the expert weights. The
// row axis is a template parameter, never a grid dimension: putting it in the grid would
// read the weights R times and give back exactly what batching exists to save.
//
// Batched routing is per (row, expert): row `t` may not have selected expert `e` at all.
// That is expressed as `wexpert[e·R + t] == 0` rather than as a mask, so the caller builds
// the UNION of the rows' picks and fills a small dense weight matrix — no mask plumbing,
// no per-expert row lists. The rows that did not pick an expert still ride through its
// gate/up/down arithmetic, and that is deliberately fine: the weights are read either way,
// and reading them is the cost.
//
// EVERY row-major axis puts the token row FASTEST — `h[(e·R + t)·inter + j]`,
// `wexpert[e·R + t]`, `acc[t·hidden + o]`. Two consequences, both load-bearing:
//
//   - No stride parameters. Each stride is `R`, `inter` or `hidden`, all already in hand,
//     so batching adds exactly ONE argument to the launcher.
//   - At R=1 every one of those expressions collapses to the single-row form it replaced
//     (`h[e·inter+j]`, `wexpert[e]`, `acc[o]`). The shipping path's memory layout is not
//     merely compatible, it is unchanged — so nothing about it needs re-measuring.
//
// Pass 1: h[(e·R + t)·inter + j] = silu(gate_ej·x_t) · (up_ej·x_t). One WAVE per
// (expert,row), producing R values. Processes the ABSOLUTE expert range [e_start,
// e_start+e_count) of `descs`, writing its own `h` rows — so the streaming pipeline can
// launch experts (or a subrange) on independent streams and they all land in the same
// schedule-independent `h`.
template <int R>
__device__ __forceinline__ void moe_gateup_vq_impl(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescVq* __restrict__ descs, const __half* __restrict__ gate_cb,
    const __half* __restrict__ up_cb, float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * inter) return;
    int e = e_start + (int)(r / inter);
    int j = (int)(r % inter);
    const ExpertDescVq d = descs[e];
    size_t rb = vq_rb(hidden), ng = vq_ng(hidden);
    float g[R], u[R];
    dot_vq_wave_r<R>(x, hidden, d.gate_indices + (size_t)j * rb, d.gate_scales + (size_t)j * ng,
                     gate_cb, hidden, lane, g);
    dot_vq_wave_r<R>(x, hidden, d.up_indices + (size_t)j * rb, d.up_scales + (size_t)j * ng,
                     up_cb, hidden, lane, u);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t)
            h_out[((size_t)e * R + t) * inter + j] = siluf(g[t]) * u[t];
    }
}

// Pass 2: acc[t][o] += fixed(w[e][t] · (down_eo · h[e][t])). Every expert accumulates into
// ONE row per token row, atomically, in fixed point — so the result is schedule-independent
// not because the writes are disjoint but because integer addition is associative. Same
// absolute-range form. See common.hpp's MOE_ACC_SHIFT block for the width argument.
template <int R>
__device__ __forceinline__ void moe_down_vq_impl(
    int hidden, int inter, int e_start, int e_count, const ExpertDescVq* __restrict__ descs,
    const __half* __restrict__ down_cb, const float* __restrict__ wexpert,
    const float* __restrict__ h_in, unsigned long long* __restrict__ acc) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * hidden) return;
    int e = e_start + (int)(r / hidden);
    int o = (int)(r % hidden);
    const ExpertDescVq d = descs[e];
    size_t rb = vq_rb(inter), ng = vq_ng(inter);
    // This expert's R h-rows are ADJACENT (row-fastest), so the gather dot_vq_wave_r does
    // over them is stride `inter` and both rows share cache lines.
    const float* he = h_in + (size_t)e * R * inter;
    float dv[R];
    dot_vq_wave_r<R>(he, inter, d.down_indices + (size_t)o * rb, d.down_scales + (size_t)o * ng,
                     down_cb, inter, lane, dv);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            // Skip the rows that did not route here. Not an optimisation — `w == 0` times a
            // non-finite `dv` is NaN, and moe_fixed's clamp turns NaN into a FINITE extreme
            // (fmaxf returns the non-NaN operand), i.e. silent corruption of a row that
            // never asked for this expert. Skipping is also bit-identical: adding zero to
            // an integer accumulator is a no-op.
            float w = wexpert[(size_t)e * R + t];
            if (w != 0.0f) atomicAdd(&acc[(size_t)t * hidden + o], moe_fixed(w * dv[t]));
        }
    }
}

extern "C" __global__ void moe_gateup_vq(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescVq* __restrict__ descs, const __half* __restrict__ gate_cb,
    const __half* __restrict__ up_cb, float* __restrict__ h_out) {
    moe_gateup_vq_impl<1>(x, hidden, inter, e_start, e_count, descs, gate_cb, up_cb, h_out);
}

extern "C" __global__ void moe_gateup_vq_r2(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescVq* __restrict__ descs, const __half* __restrict__ gate_cb,
    const __half* __restrict__ up_cb, float* __restrict__ h_out) {
    moe_gateup_vq_impl<2>(x, hidden, inter, e_start, e_count, descs, gate_cb, up_cb, h_out);
}

extern "C" __global__ void moe_down_vq(
    int hidden, int inter, int e_start, int e_count, const ExpertDescVq* __restrict__ descs,
    const __half* __restrict__ down_cb, const float* __restrict__ wexpert,
    const float* __restrict__ h_in, unsigned long long* __restrict__ acc) {
    moe_down_vq_impl<1>(hidden, inter, e_start, e_count, descs, down_cb, wexpert, h_in, acc);
}

extern "C" __global__ void moe_down_vq_r2(
    int hidden, int inter, int e_start, int e_count, const ExpertDescVq* __restrict__ descs,
    const __half* __restrict__ down_cb, const float* __restrict__ wexpert,
    const float* __restrict__ h_in, unsigned long long* __restrict__ acc) {
    moe_down_vq_impl<2>(hidden, inter, e_start, e_count, descs, down_cb, wexpert, h_in, acc);
}

// acc → f32, folded into the residual add: x[o] += gain · (acc[o] · 2^-MOE_ACC_SHIFT),
// and the accumulator is RESET here for the next layer. Resetting in the drain rather
// than memsetting before the experts is what keeps the miss stream free of a start-of-
// layer dependency: the end-of-layer device_sync already separates this store from the
// next layer's first atomic.
//
// double, not float, for the convert: acc reaches 2^48 and f32 would round it away.
extern "C" __global__ void moe_acc_drain(
    float* __restrict__ x, unsigned long long* __restrict__ acc, int n, int rows, float gain) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    long long t = 0;
    for (int r = 0; r < rows; ++r) {
        t += (long long)acc[(size_t)r * n + o];
        acc[(size_t)r * n + o] = 0ull;
    }
    x[o] += gain * (float)((double)t / (double)MOE_ACC_SCALE);
}

// --- Streaming-pipeline launchers: gate/up + down for an ABSOLUTE expert range on
//     a caller-owned stream (so load DMA and this compute overlap on the GPU), and
//     a stream-scoped reduce. The GLM pipeline launches ONE expert per range
//     (e_count=1, per-expert Signal overlap); the [e_start,e_count) range form is the
//     hook for a batching of RESIDENT experts (whose Signals are already ready)
//     into a single launch, to cut host-gated launch bubbles. [2026-08-07: the V4 fp4
//     caller cashed that hook in — `v4gpu::routed_experts` batches its residents into one
//     range in LAUNCH order; this GLM vq/i4 path still launches one per range.]
// `nrow` token rows through ONE read of the expert weights. The grid is unchanged — the
// row axis is inside each wave, not beside it — so this is the same launch shape doing
// nrow times the arithmetic against the same bytes.
extern "C" int rivoli_moe_expert_range(
    const float* x, int hidden, int inter, int e_start, int e_count, const void* descs,
    const __half* gate_cb, const __half* up_cb, const __half* down_cb, const float* wexpert,
    float* h, unsigned long long* acc, int nrow, void* stream) {
    if (hidden <= 0 || inter <= 0 || e_count <= 0 || e_start < 0) return 1001;
    // Only the widths instantiated above. Depth-2 drafts measure 4.4% accepted, so a third
    // row buys 1.6% for a whole extra row of verify — 2 IS the answer, and a wider `nrow`
    // would be a template the data says not to write.
    if (nrow != 1 && nrow != 2) return 1003;
    const ExpertDescVq* d = (const ExpertDescVq*)descs;
    hipStream_t s = (hipStream_t)stream;
    hipGetLastError();
    dim3 grid_a(((size_t)e_count * inter + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    if (nrow == 1) {
        hipLaunchKernelGGL(moe_gateup_vq, grid_a, dim3(256), 0, s, x, hidden, inter, e_start,
                           e_count, d, gate_cb, up_cb, h);
    } else {
        hipLaunchKernelGGL(moe_gateup_vq_r2, grid_a, dim3(256), 0, s, x, hidden, inter, e_start,
                           e_count, d, gate_cb, up_cb, h);
    }
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    dim3 grid_b(((size_t)e_count * hidden + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    if (nrow == 1) {
        hipLaunchKernelGGL(moe_down_vq, grid_b, dim3(256), 0, s, hidden, inter, e_start, e_count,
                           d, down_cb, wexpert, h, acc);
    } else {
        hipLaunchKernelGGL(moe_down_vq_r2, grid_b, dim3(256), 0, s, hidden, inter, e_start,
                           e_count, d, down_cb, wexpert, h, acc);
    }
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_moe_acc_drain_s(float* x, unsigned long long* acc, int n, int rows,
                                      float gain, void* stream) {
    if (n <= 0 || rows <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(moe_acc_drain, dim3((n + 255) / 256), dim3(256), 0, (hipStream_t)stream,
                       x, acc, n, rows, gain);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// ══ FP4 MoE — DeepSeek-V4-Flash's native routed experts ══════════════════════════════
//
// Same two-pass skeleton as the VQ and int4 paths, and FIVE differences from them, every
// one of which is silent-wrong if it is dropped. They are listed here because a reader who
// pattern-matches this onto the int4 twin below will get all five wrong at once.
//
//  1. The ACTIVATION is fp8. `model.py::linear` quantizes `x` to e4m3 at block 128 with a
//     ue8m0 scale BEFORE the fp4 GEMM. This path therefore consumes an already-quantized
//     `x` (`linalg.hip::act_quant_f8`), and `h` is re-quantized between the two passes.
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
// The six spans below are cast to `common.hpp::gu8p` (GLOBAL address space) at every use —
// the typedef's comment carries the measurement and the rejected alternatives. The cast is
// TRUE by construction: every span points at a device arena or heap allocation (v4gpu's
// placement, `tests/v4_kernel.rs`'s `DeviceBuf`s), and host GTT memory is still the global
// aperture on this APU; no LDS or scratch address can reach a descriptor. It asserts only
// the address SPACE — what the pointer may alias is untouched.

// Defined in linalg.hip. Declared rather than called from Rust so the sequence
// gate/up → re-quantize → down cannot be issued with the middle step missing; see
// `rivoli_moe_expert_range_f4`.
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

// The bf16 round-trip below is `common.hpp::rbf16`. It was a local `bf16r` with an
// identical body until 2026-08-05 — `rbf16`/`bf16r` in one translation unit is a
// transposition between two names for the same thing, in a file where picking the wrong
// one would be invisible because both were correct. Every `Linear` in the reference
// returns bf16 and the residual stream is bf16, so this is where the reference's precision
// actually lives: dropping it is worth ~2^-8 relative per step, two orders above the
// summation-order floor the oracle's tolerance is built on.

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
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * inter) return;
    int e = e_start + (int)(r / inter);
    int j = (int)(r % inter);
    const ExpertDescF4 d = descs[e];
    size_t rb = (size_t)(hidden / 2), ng = f4_ng(hidden);
    float g[R], u[R];
    // `(gu8p)` on the four descriptor spans — flat_load → global_load, see the typedef and
    // the struct's own note for why the cast is true.
    dot_f4_wave_r<R>(x, hidden, (gu8p)d.gate_packed + (size_t)j * rb,
                     (gu8p)d.gate_scale + (size_t)j * ng, hidden, lane, g);
    dot_f4_wave_r<R>(x, hidden, (gu8p)d.up_packed + (size_t)j * rb,
                     (gu8p)d.up_scale + (size_t)j * ng, hidden, lane, u);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            // `common.hpp::swiglu_clamped` — bf16 both operands, clamp `up` both sides and
            // `gate` only from above, `x·sigmoid(x)`, return the product UNROUNDED. It was
            // these four statements inline here until 2026-08-05, hoisted when the fp8 SHARED
            // expert needed the identical arithmetic (`launch_v4_swiglu_clamped`): the two
            // must agree bit for bit, `kernels/` is not watched by jscpd, and a second copy
            // would drift silently. The hoist was MEASURED bit-neutral, not assumed —
            // `tests/v4_kernel.rs::the_fp4_dispatch_hash_pins_the_clamp_hoist` carries the
            // hash and the argument.
            float sw = swiglu_clamped(g[t], u[t], limit);
            // ENFORCED, not inherited. `0 · finite` is exactly zero, but the gate has no
            // LOWER clamp, so a `w1` dot at -inf gives `silu = -inf · 0 = NaN`, and
            // `NaN · 0` is NaN, not 0. `moe_fixed` then turns that NaN into a FINITE extreme
            // (`fminf`/`fmaxf` return the non-NaN operand) — silent corruption of a row that
            // never asked for this expert. The int4 twin below carries the same guard on the
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

// Pass 2: acc[t][o] += fixed(bf16(w2_o · h[e][t])). The routing weight is ALREADY in `h`
// (see the block comment's point 3), so unlike the VQ/int4 twins this takes no `wexpert`
// — one source of truth for the routing, and no way for a mask to disagree with the data.
//
// A row that did not route to `e` gets `h == 0` exactly — pass 1 ENFORCES that rather than
// deriving it (see the note there) — hence a dot of exactly 0.0 and `moe_fixed(0) == 0`.
// Adding it is a no-op, so riding through the arithmetic is bit-identical to skipping it.
// It is not free, but it is not the cost either: the weight read is, and that happens
// either way.
//
// `h` must have been re-quantized to fp8 between the passes: `w2` is a quantized `Linear`
// like any other and quantizes its own input.
template <int R>
__device__ __forceinline__ void moe_down_f4_impl(
    int hidden, int inter, int e_start, int e_count, const ExpertDescF4* __restrict__ descs,
    const float* __restrict__ h_in, unsigned long long* __restrict__ acc) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * hidden) return;
    int e = e_start + (int)(r / hidden);
    int o = (int)(r % hidden);
    const ExpertDescF4 d = descs[e];
    size_t rb = (size_t)(inter / 2), ng = f4_ng(inter);
    const float* he = h_in + (size_t)e * R * inter;
    float dv[R];
    dot_f4_wave_r<R>(he, inter, (gu8p)d.down_packed + (size_t)o * rb,
                     (gu8p)d.down_scale + (size_t)o * ng, inter, lane, dv);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t)
            atomicAdd(&acc[(size_t)t * hidden + o], moe_fixed(rbf16(dv[t])));
    }
}

extern "C" __global__ void moe_gateup_f4(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescF4* __restrict__ descs, const float* __restrict__ wexpert, float limit,
    float* __restrict__ h_out) {
    moe_gateup_f4_impl<1>(x, hidden, inter, e_start, e_count, descs, wexpert, limit, h_out);
}

extern "C" __global__ void moe_down_f4(
    int hidden, int inter, int e_start, int e_count, const ExpertDescF4* __restrict__ descs,
    const float* __restrict__ h_in, unsigned long long* __restrict__ acc) {
    moe_down_f4_impl<1>(hidden, inter, e_start, e_count, descs, h_in, acc);
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
    // one-sided form had been copied into `linalg.hip::rivoli_v4_swiglu_clamped` as a
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

// ── V4 router (`model.py::Gate.forward`, `score_func == "sqrtsoftplus"`) ─────────
//
// Takes the gate logits (a DENSE f32 GEMV — `Gate.forward` opens with
// `linear(x.float(), self.weight.float())`, so no activation quantization and no fp8) and
// produces the top-k routing weights and indices.
//
// Two things here are easy to lose and impossible to see afterwards:
//
//   * the load-balancing `bias` shifts the scores used for SELECTION and is absent from
//     the scores used as WEIGHTS (model.py:577-585). Using the shifted scores as weights
//     changes every routed magnitude by an amount that looks like ordinary variation.
//   * a HASH layer takes its indices from `tid2eid[input_id]` and bypasses the scores
//     entirely — but the gate still RUNS, and its scores still become the weights. Reading
//     `tid2eid` and skipping the gate leaves the weights uninitialised or uniform, which
//     decodes fluently and wrongly. That is why this kernel takes both and the launcher
//     refuses any combination but the model's own two.
//
// `sqrt(softplus(·))` uses PyTorch's DEFAULT `threshold = 20`: above it `softplus` IS the
// identity, and `ln(1 + e^x)` in f32 saturates to +inf near x = 88. Without the branch a
// large logit produces `sqrt(inf)` and the whole row renormalises to NaN.
//
// ONE BLOCK, and the selection runs on ONE THREAD. `torch.topk` is descending with ties to
// the LOWER index, and a k-pass serial argmax is that rule by construction, where a
// tree-reduced (value, index) pair is that rule only if the tie-break is written correctly
// at every level.
//
// The work is n_experts·k = 1536 LDS compares per layer per token. NOT MEASURED — S4 should
// price it before anyone parallelises it, and it is written here as a serial scan because
// being obviously right is worth more than being fast until that number exists.
extern "C" __global__ void moe_gate_v4(
    const float* __restrict__ logits, const float* __restrict__ bias,
    const long long* __restrict__ tid2eid, int input_id, int n_experts, int k,
    float route_scale, float* __restrict__ weights, int* __restrict__ indices) {
    extern __shared__ float gate_lds[];
    float* orig = gate_lds;                // sqrt(softplus(logit)) — the WEIGHT scores
    float* sel = gate_lds + n_experts;     // + bias — the SELECTION scores
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        float l = logits[e];
        // F.softplus's threshold=20 identity branch, which the reference relies on without
        // naming it. See the block comment.
        // `expf`/`logf`, NOT `__expf`/`__logf`: the fast intrinsics carry ~2 ulp, and this
        // value is compared against a CPU oracle whose claimed floor is re-association.
        float sp = l > 20.0f ? l : logf(1.0f + expf(l));
        orig[e] = sqrtf(sp);
        sel[e] = orig[e] + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int j = 0; j < k; ++j) {
        int pick;
        if (tid2eid) {
            pick = (int)tid2eid[(size_t)input_id * k + j];
        } else {
            // Descending, ties to the LOWER index: `>` (not `>=`) keeps the first-seen
            // maximum, and the scan runs in ascending expert order.
            pick = 0;
            float best = sel[0];
            for (int e = 1; e < n_experts; ++e) {
                if (sel[e] > best) { best = sel[e]; pick = e; }
            }
            sel[pick] = -INFINITY;  // mask, so pass j+1 finds the next
        }
        indices[j] = pick;
        weights[j] = orig[pick];  // the UNSHIFTED score — never `sel`
    }
    // `weights /= weights.sum()` then `*= route_scale`, in that order and only for the
    // non-softmax score functions (V4 is `sqrtsoftplus`, so always).
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) sum += weights[j];
    for (int j = 0; j < k; ++j) weights[j] = (weights[j] / sum) * route_scale;
}

// `n_experts` f32 of LDS for the weight scores and as many again for the selection scores.
//
// The two arrays are separate because the bias must NOT reach the weights: one array with
// the bias added in place is exactly `Defect::RouterBiasedWeights`, and it would be a
// one-line "simplification" that no shape or magnitude check could find afterwards.
//
// **Entries of `tid2eid` are NOT range-checked** — a value outside `[0, n_experts)` reads
// off the end of the LDS score array. Checking it per pick would put a branch on every
// selection to catch a defect in a table that is loaded once, so S3 validates the table at
// load instead. Recorded here because "the kernel checks it" is what a reader would assume.
//
// Takes a STREAM for the reason `rivoli_hc_pre` gives.
extern "C" int rivoli_moe_gate_v4(
    const float* logits, const float* bias, const long long* tid2eid, int input_id,
    int vocab_size, int n_experts, int k, float route_scale, float* weights, int* indices,
    void* stream) {
    if (n_experts <= 0 || k <= 0 || k > n_experts) return 1001;
    // A hash layer has `tid2eid` and NO bias; a scored layer has a bias and NO `tid2eid`
    // (model.py:561-566). The other two combinations are the two silent-wrongs this router
    // invites — routing by score on a hash layer, and biasing the weights — so they are
    // refused here rather than resolved by precedence inside the kernel.
    if ((tid2eid != 0) == (bias != 0)) return 1002;
    // `input_id < 0` was the guard here and it could not fire: the Rust launcher takes a
    // `usize`, so reaching a negative i32 needs a host index above 2^31. The reachable
    // overrun is off the OTHER end — `tid2eid[input_id * k + j]` past a `[vocab_size, k]`
    // table — so that is what is checked. `vocab_size` is ignored on a scored layer, where
    // there is no table to run off.
    if (tid2eid && (input_id < 0 || input_id >= vocab_size)) return 1003;
    hipGetLastError();
    hipLaunchKernelGGL(moe_gate_v4, dim3(1), dim3(256), 2 * n_experts * sizeof(float),
                       (hipStream_t)stream, logits, bias, tid2eid, input_id, n_experts, k,
                       route_scale, weights, indices);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// ── int4 MoE (group-scaled "warm expert" format): same two-pass shape as the VQ
//    path, decoding via dot_i4_wave with one f32 scale per I4_GROUP weights applied
//    INSIDE the dot. One descriptor per expert (six device pointers), absolute
//    [e_start,e_count) range on a stream, so it plugs into the same expert-stream +
//    moe_acc_drain as VQ.
struct ExpertDescI4 {
    const unsigned char* gate_packed;  // [inter · (hidden+1)/2]
    const float*         gate_scale;   // [inter · i4_ng(hidden)]
    const unsigned char* up_packed;
    const float*         up_scale;
    const unsigned char* down_packed;  // [hidden · (inter+1)/2]
    const float*         down_scale;   // [hidden · i4_ng(inter)]
};

// i4_groups inline (i_dim is the reduction dim of the projection) — matches
// quant.rs::i4_groups, the sizing i4_slot_offsets lays the scale spans out with.
__device__ __forceinline__ size_t i4_ng(int i_dim) {
    return (size_t)((i_dim + I4_GROUP - 1) / I4_GROUP);
}

// The int4 twins of the VQ pair above, and the same contract, axis for axis: `h` is
// `[(e·R + t)·inter + j]`, `wexpert` is `[e·R + t]`, `acc` is `[t·hidden + o]`. That the
// two formats agree on the layout is load-bearing rather than tidy — a HYBRID layer mixes
// int4 and VQ experts within one batch and they accumulate into the same `h` and `acc`.
template <int R>
__device__ __forceinline__ void moe_gateup_i4_impl(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescI4* __restrict__ descs, float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * inter) return;
    int e = e_start + (int)(r / inter);
    int j = (int)(r % inter);
    const ExpertDescI4 d = descs[e];
    size_t rb_h = (size_t)((hidden + 1) / 2), ng_h = i4_ng(hidden);
    float g[R], u[R];
    dot_i4_wave_r<R>(x, hidden, d.gate_packed + (size_t)j * rb_h,
                     d.gate_scale + (size_t)j * ng_h, hidden, lane, g);
    dot_i4_wave_r<R>(x, hidden, d.up_packed + (size_t)j * rb_h,
                     d.up_scale + (size_t)j * ng_h, hidden, lane, u);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t)
            h_out[((size_t)e * R + t) * inter + j] = siluf(g[t]) * u[t];
    }
}

template <int R>
__device__ __forceinline__ void moe_down_i4_impl(
    int hidden, int inter, int e_start, int e_count, const ExpertDescI4* __restrict__ descs,
    const float* __restrict__ wexpert, const float* __restrict__ h_in,
    unsigned long long* __restrict__ acc) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * hidden) return;
    int e = e_start + (int)(r / hidden);
    int o = (int)(r % hidden);
    const ExpertDescI4 d = descs[e];
    size_t rb_i = (size_t)((inter + 1) / 2), ng_i = i4_ng(inter);
    const float* he = h_in + (size_t)e * R * inter;
    float dv[R];
    dot_i4_wave_r<R>(he, inter, d.down_packed + (size_t)o * rb_i,
                     d.down_scale + (size_t)o * ng_i, inter, lane, dv);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            // Skip rows that did not route here — the same correctness argument as the VQ
            // twin, not thrift: `0 * dv` with a non-finite `dv` is NaN, and moe_fixed's
            // clamp turns NaN into a FINITE extreme. Adding zero to an integer accumulator
            // is a no-op, so skipping is also bit-identical.
            float w = wexpert[(size_t)e * R + t];
            if (w != 0.0f) atomicAdd(&acc[(size_t)t * hidden + o], moe_fixed(w * dv[t]));
        }
    }
}

extern "C" __global__ void moe_gateup_i4(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescI4* __restrict__ descs, float* __restrict__ h_out) {
    moe_gateup_i4_impl<1>(x, hidden, inter, e_start, e_count, descs, h_out);
}

extern "C" __global__ void moe_gateup_i4_r2(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescI4* __restrict__ descs, float* __restrict__ h_out) {
    moe_gateup_i4_impl<2>(x, hidden, inter, e_start, e_count, descs, h_out);
}

extern "C" __global__ void moe_down_i4(
    int hidden, int inter, int e_start, int e_count, const ExpertDescI4* __restrict__ descs,
    const float* __restrict__ wexpert, const float* __restrict__ h_in,
    unsigned long long* __restrict__ acc) {
    moe_down_i4_impl<1>(hidden, inter, e_start, e_count, descs, wexpert, h_in, acc);
}

extern "C" __global__ void moe_down_i4_r2(
    int hidden, int inter, int e_start, int e_count, const ExpertDescI4* __restrict__ descs,
    const float* __restrict__ wexpert, const float* __restrict__ h_in,
    unsigned long long* __restrict__ acc) {
    moe_down_i4_impl<2>(hidden, inter, e_start, e_count, descs, wexpert, h_in, acc);
}

extern "C" int rivoli_moe_expert_range_i4(
    const float* x, int hidden, int inter, int e_start, int e_count, const void* descs,
    const float* wexpert, float* h, unsigned long long* acc, int nrow, void* stream) {
    if (hidden <= 0 || inter <= 0 || e_count <= 0 || e_start < 0) return 1001;
    // Only the widths instantiated above — same reasoning as the VQ launcher. This used to
    // refuse every nrow > 1 (code 1004) because no batched int4 kernel existed, and that
    // guard EARNED ITS KEEP on 2026-07-31: emitting L78.i4 made the path reachable for the
    // first time and it failed loudly at token 1 instead of silently computing row 1 short
    // a few experts in a hybrid layer.
    if (nrow != 1 && nrow != 2) return 1003;
    const ExpertDescI4* d = (const ExpertDescI4*)descs;
    hipStream_t s = (hipStream_t)stream;
    hipGetLastError();
    dim3 grid_a(((size_t)e_count * inter + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    if (nrow == 1) {
        hipLaunchKernelGGL(moe_gateup_i4, grid_a, dim3(256), 0, s, x, hidden, inter, e_start,
                           e_count, d, h);
    } else {
        hipLaunchKernelGGL(moe_gateup_i4_r2, grid_a, dim3(256), 0, s, x, hidden, inter, e_start,
                           e_count, d, h);
    }
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    dim3 grid_b(((size_t)e_count * hidden + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    if (nrow == 1) {
        hipLaunchKernelGGL(moe_down_i4, grid_b, dim3(256), 0, s, hidden, inter, e_start,
                           e_count, d, wexpert, h, acc);
    } else {
        hipLaunchKernelGGL(moe_down_i4_r2, grid_b, dim3(256), 0, s, hidden, inter, e_start,
                           e_count, d, wexpert, h, acc);
    }
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
