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
extern "C" __global__ void moe_gateup_vq(
    const float* __restrict__ x, int hidden, int inter, int E,
    const ExpertDescVq* __restrict__ descs, const float* __restrict__ gate_cb,
    const float* __restrict__ up_cb, float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * inter) return;
    int e = (int)(r / inter);
    int j = (int)(r % inter);
    const ExpertDescVq d = descs[e];
    size_t rb = vq_rb(hidden), ng = vq_ng(hidden);
    float g = dot_vq_wave(x, d.gate_indices + (size_t)j * rb, d.gate_scales + (size_t)j * ng, gate_cb, hidden, lane);
    float u = dot_vq_wave(x, d.up_indices + (size_t)j * rb, d.up_scales + (size_t)j * ng, up_cb, hidden, lane);
    if (lane == 0) h_out[r] = siluf(g) * u;
}

// Pass 2: partial[e·hidden + o] = w[e] · (down_eo · h[e]). Each expert writes its
// own partial row (no atomics → schedule-independent).
extern "C" __global__ void moe_down_vq(
    int hidden, int inter, int E, const ExpertDescVq* __restrict__ descs,
    const float* __restrict__ down_cb, const float* __restrict__ wexpert,
    const float* __restrict__ h_in, float* __restrict__ partial) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * hidden) return;
    int e = (int)(r / hidden);
    int o = (int)(r % hidden);
    const ExpertDescVq d = descs[e];
    size_t rb = vq_rb(inter), ng = vq_ng(inter);
    const float* he = h_in + (size_t)e * inter;
    float dv = dot_vq_wave(he, d.down_indices + (size_t)o * rb, d.down_scales + (size_t)o * ng, down_cb, inter, lane);
    if (lane == 0) partial[r] = wexpert[e] * dv;
}

// out[o] = Σ_e partial[e][o], e ASCENDING — fixed order → deterministic decode.
extern "C" __global__ void moe_reduce(
    const float* __restrict__ partial, int e_count, int hidden, float* __restrict__ out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= hidden) return;
    float acc = 0.0f;
    for (int e = 0; e < e_count; ++e) acc += partial[(size_t)e * hidden + o];
    out[o] = acc;
}

// Launch the batch over resident device memory: pass 1 → h (≥ E·inter f32), pass 2
// → partial (E·hidden f32), fixed-order reduce → out. No alloc/copy/sync; the caller
// owns everything and joins once per token.
extern "C" int rivoli_moe_experts_vq(
    const float* x, int hidden, int inter, int E, const void* descs,
    const float* gate_cb, const float* up_cb, const float* down_cb,
    const float* wexpert, float* h, float* partial, float* out) {
    if (hidden <= 0 || inter <= 0 || E <= 0) return 1001;
    const ExpertDescVq* d = (const ExpertDescVq*)descs;
    hipGetLastError();
    size_t rows_a = (size_t)E * inter;
    hipLaunchKernelGGL(moe_gateup_vq, dim3((rows_a + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, x, hidden, inter, E, d, gate_cb, up_cb, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    size_t rows_b = (size_t)E * hidden;
    hipLaunchKernelGGL(moe_down_vq, dim3((rows_b + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, hidden, inter, E, d, down_cb, wexpert, h, partial);
    e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    hipLaunchKernelGGL(moe_reduce, dim3((hidden + 255) / 256), dim3(256), 0, 0, partial, E, hidden, out);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
