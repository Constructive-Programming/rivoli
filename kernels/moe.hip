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

// out[o] = Σ_e partial[e][o], e ASCENDING. OFF THE DECODE PATH since the fixed-point
// accumulator replaced the partial slab — kept as the f32 REFERENCE the oracle tests
// pin against (and as the shape `tests/vk.rs` uses to probe cross-queue visibility).
extern "C" __global__ void moe_reduce(
    const float* __restrict__ partial, int e_count, int hidden, float* __restrict__ out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= hidden) return;
    float acc = 0.0f;
    for (int e = 0; e < e_count; ++e) acc += partial[(size_t)e * hidden + o];
    out[o] = acc;
}

// --- Streaming-pipeline launchers: gate/up + down for an ABSOLUTE expert range on
//     a caller-owned stream (so load DMA and this compute overlap on the GPU), and
//     a stream-scoped reduce. The pipeline currently launches ONE expert per range
//     (e_count=1, per-expert Signal overlap); the [e_start,e_count) range form is the
//     hook for a future batching of RESIDENT experts (whose Signals are already ready)
//     into a single launch, to cut host-gated launch bubbles.
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

extern "C" int rivoli_moe_reduce_s(const float* partial, int E, int hidden, float* out, void* stream) {
    if (hidden <= 0 || E <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(moe_reduce, dim3((hidden + 255) / 256), dim3(256), 0, (hipStream_t)stream,
                       partial, E, hidden, out);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// ── int4 MoE (group-scaled "warm expert" format): same two-pass shape as the VQ
//    path, decoding via dot_i4_wave with one f32 scale per I4_GROUP weights applied
//    INSIDE the dot. One descriptor per expert (six device pointers), absolute
//    [e_start,e_count) range on a stream, so it plugs into the same expert-stream +
//    moe_reduce as VQ.
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
