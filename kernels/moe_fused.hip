// M3 fused MoE expert-batch kernel (descriptor-array form, D1). One launch
// computes a whole batch of experts sharing input `x` and the (hidden, inter)
// shape:
//
//     out[o] += Σ_e w[e] * down_e( silu(gate_e·x) ⊙ up_e·x )[o]
//
// Each expert's six weight tensors are addressed by DEVICE POINTERS carried in a
// per-expert descriptor, so the experts may live at arbitrary offsets in the
// resident device tier (or in cold-fetch slabs) — no contiguous-stride
// assumption, no gather/repack. This is the shape that reads the 11.3 GB/token
// of int4 weights exactly once (PLAN.md "M3 kernel contracts" D1). x, out, the
// descriptor array and the per-expert weights are all device-resident; the
// launcher only launches — the caller owns every allocation and joins once per
// token via rivoli_device_sync (bottleneck #3: ≤1 join/token).
//
// Numerics are identical to the M2 kernel (validated bit-close to quant.rs):
// per output row one thread sums i ascending, matching matvec_i4_rows.

#include "common.hpp"

struct ExpertDesc {
    const unsigned char* gate_packed;  // [inter * rb(hidden)]
    const float*         gate_scale;   // [inter]
    const unsigned char* up_packed;    // [inter * rb(hidden)]
    const float*         up_scale;     // [inter]
    const unsigned char* down_packed;  // [hidden * rb(inter)]
    const float*         down_scale;   // [hidden]
};

// dot_i4_wave now lives in common.hpp — linalg.hip and mla.hip share it, and the
// dword-per-lane load shape is a bandwidth property worth having one definition of.

__device__ __forceinline__ float siluf(float x) {
    return x / (1.0f + expf(-x));
}

// The fused MoE is UNFUSED into two coalesced, MANY-block GEMV passes so the grid
// carries the parallelism (the old one-block-per-expert form launched only E≤9
// workgroups — half the iGPU idle — and its thread-per-row reads were uncoalesced).
// Pass 1 (this) computes the SwiGLU hidden h[e][j] for every (expert, inter-row);
// pass 2 (moe_down) consumes h. One WAVE per output row, coalesced + shuffle-reduce
// (D3), grid = ceil(E*inter / ROWS_PER_BLOCK) blocks → full occupancy.
//
// h[e*inter + j] = silu(scale_g[j]·(gate_ej·x)) * scale_u[j]·(up_ej·x).
extern "C" __global__ void moe_gateup(
    const float* __restrict__ x, int hidden, int inter, int E,
    const ExpertDesc* __restrict__ descs, float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * inter) return; // wave-uniform (all lanes share r)
    int e = (int)(r / inter);
    int j = (int)(r % inter);
    const ExpertDesc d = descs[e];
    size_t rb_h = (hidden + 1) / 2;
    float g = dot_i4_wave(x, d.gate_packed + (size_t)j * rb_h, hidden, lane);
    float u = dot_i4_wave(x, d.up_packed + (size_t)j * rb_h, hidden, lane);
    if (lane == 0) h_out[r] = siluf(g * d.gate_scale[j]) * (u * d.up_scale[j]);
}

// Pass 2: partial[e][o] = w[e] * scale_d[o] · (down_eo · h[e]). Each expert writes
// its OWN partial[e] row (no atomicAdd → schedule-independent; moe_reduce sums in
// fixed order). One WAVE per output row, coalesced; grid = ceil(E*hidden / RPB).
extern "C" __global__ void moe_down(
    int hidden, int inter, int E, const ExpertDesc* __restrict__ descs,
    const float* __restrict__ wexpert, const float* __restrict__ h_in,
    float* __restrict__ partial) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * hidden) return;
    int e = (int)(r / hidden);
    int o = (int)(r % hidden);
    const ExpertDesc d = descs[e];
    size_t rb_i = (inter + 1) / 2;
    const float* he = h_in + (size_t)e * inter;
    float dv = dot_i4_wave(he, d.down_packed + (size_t)o * rb_i, inter, lane);
    if (lane == 0) partial[r] = wexpert[e] * dv * d.down_scale[o];
}

// ── VQ-int3 fused MoE (M3) ──────────────────────────────────────────────────
// Same two-pass shape as the int4 path, but each projection decodes via
// dot_vq_wave (12-bit codebook indices + per-group bf16 scales). The group scale
// is applied INSIDE the dot, so there is no per-row post-scale here (unlike int4).
// The codebook is ONE device buffer shared by every expert — passed to the
// launcher, not carried per-descriptor.
struct ExpertDescVq {
    const unsigned char*  gate_indices;  // [inter * vq_row_bytes(hidden)]
    const unsigned short* gate_scales;   // [inter * vq_groups(hidden)] bf16
    const unsigned char*  up_indices;    // [inter * vq_row_bytes(hidden)]
    const unsigned short* up_scales;     // [inter * vq_groups(hidden)]
    const unsigned char*  down_indices;  // [hidden * vq_row_bytes(inter)]
    const unsigned short* down_scales;   // [hidden * vq_groups(inter)]
};

// h[e*inter + j] = silu(gate_ej·x) * (up_ej·x), gate/up already group-dequantized.
extern "C" __global__ void moe_gateup_vq(
    const float* __restrict__ x, int hidden, int inter, int E,
    const ExpertDescVq* __restrict__ descs, const float* __restrict__ codebook,
    float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * inter) return;
    int e = (int)(r / inter);
    int j = (int)(r % inter);
    const ExpertDescVq d = descs[e];
    size_t rb_h = (size_t)(hidden / VQ_DIM) * VQ_INDEX_BITS / 8;
    size_t ng_h = (size_t)hidden / VQ_GROUP;
    float g = dot_vq_wave(x, d.gate_indices + (size_t)j * rb_h, d.gate_scales + (size_t)j * ng_h,
                          codebook, hidden, lane);
    float u = dot_vq_wave(x, d.up_indices + (size_t)j * rb_h, d.up_scales + (size_t)j * ng_h,
                          codebook, hidden, lane);
    if (lane == 0) h_out[r] = siluf(g) * u;
}

// partial[e][o] = w[e] * (down_eo · h[e]), down already group-dequantized.
extern "C" __global__ void moe_down_vq(
    int hidden, int inter, int E, const ExpertDescVq* __restrict__ descs,
    const float* __restrict__ codebook, const float* __restrict__ wexpert,
    const float* __restrict__ h_in, float* __restrict__ partial) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * hidden) return;
    int e = (int)(r / hidden);
    int o = (int)(r % hidden);
    const ExpertDescVq d = descs[e];
    size_t rb_i = (size_t)(inter / VQ_DIM) * VQ_INDEX_BITS / 8;
    size_t ng_i = (size_t)inter / VQ_GROUP;
    const float* he = h_in + (size_t)e * inter;
    float dv = dot_vq_wave(he, d.down_indices + (size_t)o * rb_i, d.down_scales + (size_t)o * ng_i,
                           codebook, inter, lane);
    if (lane == 0) partial[r] = wexpert[e] * dv;
}

// out[o] = Σ_e partial[e][o], e ASCENDING — a fixed, schedule-independent
// summation order, so the MoE output (and thus greedy decode) is deterministic.
extern "C" __global__ void moe_reduce(
    const float* __restrict__ partial, int e_count, int hidden, float* __restrict__ out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= hidden) return;
    float acc = 0.0f;
    for (int e = 0; e < e_count; ++e) acc += partial[(size_t)e * hidden + o];
    out[o] = acc;
}

// Launch the batch over already-resident device memory: pass 1 writes the SwiGLU
// hidden into `h` (caller-owned scratch, ≥ E*inter floats), pass 2 writes each
// expert's `partial[E*hidden]` row, then the fixed-order reduce into `out` (fully
// written, no pre-zero). Does NOT allocate/copy/synchronize — the caller owns
// x/descs/wexpert/h/partial/out and joins once per token. Return codes: 0 ok,
// POSITIVE arg guard, NEGATIVE -(hipError_t).
extern "C" int rivoli_moe_experts(
    const float* x, int hidden, int inter, int E,
    const void* descs, const float* wexpert, float* h, float* partial, float* out) {
    if (hidden <= 0 || inter <= 0 || E <= 0) return 1001;
    const ExpertDesc* d = (const ExpertDesc*)descs;
    hipGetLastError();  // drain any stale error so the check below is this launch's
    // Pass 1: gate/up → h (grid = ceil(E*inter / ROWS_PER_BLOCK) coalesced waves).
    size_t rows_a = (size_t)E * inter;
    hipLaunchKernelGGL(moe_gateup, dim3((rows_a + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, x, hidden, inter, E, d, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    // Pass 2: down · h → partial (grid = ceil(E*hidden / ROWS_PER_BLOCK)).
    size_t rows_b = (size_t)E * hidden;
    hipLaunchKernelGGL(moe_down, dim3((rows_b + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, hidden, inter, E, d, wexpert, h, partial);
    e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    hipLaunchKernelGGL(moe_reduce, dim3((hidden + 255) / 256), dim3(256), 0, 0,
                       partial, E, hidden, out);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// VQ analog of rivoli_moe_experts: same two-pass + fixed-order reduce, plus the
// shared `codebook` pointer. Caller owns x/descs/codebook/wexpert/h/partial/out
// and joins once per token. Return codes: 0 ok, POSITIVE arg guard, NEGATIVE err.
extern "C" int rivoli_moe_experts_vq(
    const float* x, int hidden, int inter, int E, const void* descs,
    const float* codebook, const float* wexpert, float* h, float* partial, float* out) {
    if (hidden <= 0 || inter <= 0 || E <= 0) return 1001;
    const ExpertDescVq* d = (const ExpertDescVq*)descs;
    hipGetLastError();
    size_t rows_a = (size_t)E * inter;
    hipLaunchKernelGGL(moe_gateup_vq, dim3((rows_a + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, x, hidden, inter, E, d, codebook, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    size_t rows_b = (size_t)E * hidden;
    hipLaunchKernelGGL(moe_down_vq, dim3((rows_b + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, hidden, inter, E, d, codebook, wexpert, h, partial);
    e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    hipLaunchKernelGGL(moe_reduce, dim3((hidden + 255) / 256), dim3(256), 0, 0,
                       partial, E, hidden, out);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// One join point per token: block until all launched kernels retire, surfacing
// any async execution fault. Returns 0 or -(hipError_t).
extern "C" int rivoli_device_sync() {
    hipError_t e = hipDeviceSynchronize();
    return e == hipSuccess ? 0 : -(int)e;
}
