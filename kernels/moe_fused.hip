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

#include <hip/hip_runtime.h>

struct ExpertDesc {
    const unsigned char* gate_packed;  // [inter * rb(hidden)]
    const float*         gate_scale;   // [inter]
    const unsigned char* up_packed;    // [inter * rb(hidden)]
    const float*         up_scale;     // [inter]
    const unsigned char* down_packed;  // [hidden * rb(inter)]
    const float*         down_scale;   // [hidden]
};

#define WAVE 32          // gfx1151 native wave32
#define ROWS_PER_BLOCK 8 // 256 threads = 8 waves → 8 output rows/block

// Sum a wave's partials into lane 0 (fixed __shfl_down order → deterministic).
__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}

// Wave-cooperative int4 dot: lane `l` sums columns i ≡ l (mod WAVE) — adjacent
// lanes read adjacent packed bytes (one coalesced transaction). Result on lane 0.
// Matches quant.rs: low nibble = even col, high = odd; signed value = nibble - 8.
__device__ __forceinline__ float dot_i4_wave(const float* __restrict__ v,
                                             const unsigned char* __restrict__ row,
                                             int dim, int lane) {
    float acc = 0.0f;
    for (int i = lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        acc += v[i] * (float)(n - 8);
    }
    return wave_sum(acc);
}

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

// --- batched (S positions) MoE for MTP speculative verify -------------------
// One launch applies the UNION of the S positions' experts to all S input rows,
// reading each int4 expert's weights ONCE (the fetch amortization: overlapping
// expert sets → far fewer unique reads than S separate S=1 calls). Each position
// weights only the experts it selected (wexpert[s*E+e] = 0 for the rest), so the
// output matches S independent S=1 MoE calls. x/out are [S*hidden] row-major;
// wexpert is [S*E]; h is [S*E*inter]; partial is [S*E*hidden].
#define MAXS 8 // max drafts verified per batch (1-token MTP head uses S=2)

// S wave-dots of the SAME int4 row against S input rows (base + s*xstride), so
// the row's nibbles are decoded once and reused across positions. Results (one
// per position) land on lane 0 in out[0..S).
__device__ __forceinline__ void dot_i4_wave_batched(
    const float* __restrict__ x, int dim, int S, size_t xstride,
    const unsigned char* __restrict__ row, int lane, float* out) {
    float acc[MAXS];
    for (int s = 0; s < S; ++s) acc[s] = 0.0f;
    for (int i = lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        float wv = (float)(((i & 1) ? (b >> 4) : (b & 0x0F)) - 8);
        for (int s = 0; s < S; ++s) acc[s] += x[(size_t)s * xstride + i] * wv;
    }
    for (int s = 0; s < S; ++s) out[s] = wave_sum(acc[s]);
}

extern "C" __global__ void moe_gateup_batched(
    const float* __restrict__ x, int hidden, int inter, int E, int S,
    const ExpertDesc* __restrict__ descs, float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * inter) return;
    int e = (int)(r / inter), j = (int)(r % inter);
    const ExpertDesc d = descs[e];
    size_t rb_h = (hidden + 1) / 2;
    float g[MAXS], u[MAXS];
    dot_i4_wave_batched(x, hidden, S, (size_t)hidden, d.gate_packed + (size_t)j * rb_h, lane, g);
    dot_i4_wave_batched(x, hidden, S, (size_t)hidden, d.up_packed + (size_t)j * rb_h, lane, u);
    if (lane == 0)
        for (int s = 0; s < S; ++s)
            h_out[(size_t)s * E * inter + r] = siluf(g[s] * d.gate_scale[j]) * (u[s] * d.up_scale[j]);
}

extern "C" __global__ void moe_down_batched(
    int hidden, int inter, int E, int S, const ExpertDesc* __restrict__ descs,
    const float* __restrict__ wexpert, const float* __restrict__ h_in,
    float* __restrict__ partial) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * hidden) return;
    int e = (int)(r / hidden), o = (int)(r % hidden);
    const ExpertDesc d = descs[e];
    size_t rb_i = (inter + 1) / 2;
    // position s's h for expert e: base h_in + e*inter, stride E*inter between s.
    float dv[MAXS];
    dot_i4_wave_batched(h_in + (size_t)e * inter, inter, S, (size_t)E * inter,
                        d.down_packed + (size_t)o * rb_i, lane, dv);
    if (lane == 0)
        for (int s = 0; s < S; ++s)
            partial[(size_t)s * E * hidden + r] = wexpert[(size_t)s * E + e] * dv[s] * d.down_scale[o];
}

// out[s*hidden + o] = Σ_e partial[s][e][o], e ASCENDING (deterministic).
extern "C" __global__ void moe_reduce_batched(
    const float* __restrict__ partial, int E, int hidden, int S,
    float* __restrict__ out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * hidden) return;
    int s = (int)(idx / hidden), o = (int)(idx % hidden);
    float acc = 0.0f;
    for (int e = 0; e < E; ++e) acc += partial[(size_t)s * E * hidden + (size_t)e * hidden + o];
    out[idx] = acc;
}

extern "C" int rivoli_moe_experts_batched(
    const float* x, int hidden, int inter, int E, int S,
    const void* descs, const float* wexpert, float* h, float* partial, float* out) {
    if (hidden <= 0 || inter <= 0 || E <= 0 || S <= 0) return 1001;
    if (S > MAXS) return 1002;
    const ExpertDesc* d = (const ExpertDesc*)descs;
    hipGetLastError();
    size_t rows_a = (size_t)E * inter;
    hipLaunchKernelGGL(moe_gateup_batched, dim3((rows_a + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, x, hidden, inter, E, S, d, h);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    size_t rows_b = (size_t)E * hidden;
    hipLaunchKernelGGL(moe_down_batched, dim3((rows_b + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
                       dim3(256), 0, 0, hidden, inter, E, S, d, wexpert, h, partial);
    e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    size_t rows_c = (size_t)S * hidden;
    hipLaunchKernelGGL(moe_reduce_batched, dim3((rows_c + 255) / 256), dim3(256), 0, 0,
                       partial, E, hidden, S, out);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// One join point per token: block until all launched kernels retire, surfacing
// any async execution fault. Returns 0 or -(hipError_t).
extern "C" int rivoli_device_sync() {
    hipError_t e = hipDeviceSynchronize();
    return e == hipSuccess ? 0 : -(int)e;
}
