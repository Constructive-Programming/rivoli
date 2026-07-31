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

// Pass 1: h[e·inter + j] = silu(gate_ej·x) · (up_ej·x). One WAVE per (expert,row).
// Processes the ABSOLUTE expert range [e_start, e_start+e_count) of `descs`, writing
// its own `h` rows — so the streaming pipeline can launch experts (or a subrange) on
// independent streams and they all land in the same schedule-independent `h`.
extern "C" __global__ void moe_gateup_vq(
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
    float g = dot_vq_wave(x, d.gate_indices + (size_t)j * rb, d.gate_scales + (size_t)j * ng, gate_cb, hidden, lane);
    float u = dot_vq_wave(x, d.up_indices + (size_t)j * rb, d.up_scales + (size_t)j * ng, up_cb, hidden, lane);
    if (lane == 0) h_out[(size_t)e * inter + j] = siluf(g) * u;
}

// Pass 2: acc[o] += fixed(w[e] · (down_eo · h[e])). Every expert accumulates into ONE
// row, atomically, in fixed point — so the result is schedule-independent not because
// the writes are disjoint but because integer addition is associative. Same
// absolute-range form. See common.hpp's MOE_ACC_SHIFT block for the width argument.
extern "C" __global__ void moe_down_vq(
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
    const float* he = h_in + (size_t)e * inter;
    float dv = dot_vq_wave(he, d.down_indices + (size_t)o * rb, d.down_scales + (size_t)o * ng, down_cb, inter, lane);
    if (lane == 0) atomicAdd(&acc[o], moe_fixed(wexpert[e] * dv));
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
extern "C" int rivoli_moe_expert_range(
    const float* x, int hidden, int inter, int e_start, int e_count, const void* descs,
    const __half* gate_cb, const __half* up_cb, const __half* down_cb, const float* wexpert,
    float* h, unsigned long long* acc, void* stream) {
    if (hidden <= 0 || inter <= 0 || e_count <= 0 || e_start < 0) return 1001;
    const ExpertDescVq* d = (const ExpertDescVq*)descs;
    hipStream_t s = (hipStream_t)stream;
    hipGetLastError();
    size_t rows_a = (size_t)e_count * inter;
    hipLaunchKernelGGL(moe_gateup_vq, dim3((rows_a + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, s, x, hidden, inter, e_start, e_count, d, gate_cb, up_cb, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    size_t rows_b = (size_t)e_count * hidden;
    hipLaunchKernelGGL(moe_down_vq, dim3((rows_b + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, s, hidden, inter, e_start, e_count, d, down_cb, wexpert, h, acc);
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

extern "C" __global__ void moe_gateup_i4(
    const float* __restrict__ x, int hidden, int inter, int e_start, int e_count,
    const ExpertDescI4* __restrict__ descs, float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * inter) return;
    int e = e_start + (int)(r / inter);
    int j = (int)(r % inter);
    const ExpertDescI4 d = descs[e];
    size_t rb_h = (size_t)((hidden + 1) / 2), ng_h = i4_ng(hidden);
    float g = dot_i4_wave(x, d.gate_packed + (size_t)j * rb_h, d.gate_scale + (size_t)j * ng_h, hidden, lane);
    float u = dot_i4_wave(x, d.up_packed + (size_t)j * rb_h, d.up_scale + (size_t)j * ng_h, hidden, lane);
    if (lane == 0) h_out[(size_t)e * inter + j] = siluf(g) * u;
}

extern "C" __global__ void moe_down_i4(
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
    const float* he = h_in + (size_t)e * inter;
    float dv = dot_i4_wave(he, d.down_packed + (size_t)o * rb_i, d.down_scale + (size_t)o * ng_i, inter, lane);
    if (lane == 0) atomicAdd(&acc[o], moe_fixed(wexpert[e] * dv));
}

extern "C" int rivoli_moe_expert_range_i4(
    const float* x, int hidden, int inter, int e_start, int e_count, const void* descs,
    const float* wexpert, float* h, unsigned long long* acc, void* stream) {
    if (hidden <= 0 || inter <= 0 || e_count <= 0 || e_start < 0) return 1001;
    const ExpertDescI4* d = (const ExpertDescI4*)descs;
    hipStream_t s = (hipStream_t)stream;
    hipGetLastError();
    size_t rows_a = (size_t)e_count * inter;
    hipLaunchKernelGGL(moe_gateup_i4, dim3((rows_a + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, s, x, hidden, inter, e_start, e_count, d, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    size_t rows_b = (size_t)e_count * hidden;
    hipLaunchKernelGGL(moe_down_i4, dim3((rows_b + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, s, hidden, inter, e_start, e_count, d, wexpert, h, acc);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
