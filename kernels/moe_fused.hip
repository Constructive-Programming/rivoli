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

// Σ_i v[i] * (nibble(row,i) - 8), matching quant.rs: low nibble = even col,
// high nibble = odd col; signed value = nibble - 8.
__device__ __forceinline__ float dot_i4(const float* __restrict__ v,
                                        const unsigned char* __restrict__ row,
                                        int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        acc += v[i] * (float)(n - 8);
    }
    return acc;
}

__device__ __forceinline__ float siluf(float x) {
    return x / (1.0f + expf(-x));
}

// grid = E blocks (one per expert); block = 256 threads; dynamic LDS = inter floats.
// Each expert writes its weighted contribution to its OWN partial[e] row — no
// atomicAdd — so the result is independent of GPU scheduling. The fixed-order sum
// happens in moe_reduce (determinism: greedy decode is reproducible).
extern "C" __global__ void moe_expert_batch(
    const float* __restrict__ x, int hidden, int inter,
    const ExpertDesc* __restrict__ descs, const float* __restrict__ wexpert,
    float* __restrict__ partial) {
    extern __shared__ float h[];  // [inter]
    int e = blockIdx.x;
    int t = threadIdx.x;
    int nt = blockDim.x;
    const ExpertDesc d = descs[e];

    size_t rb_h = (hidden + 1) / 2;  // packed bytes per gate/up row
    size_t rb_i = (inter + 1) / 2;   // packed bytes per down row

    // Phase A: h[j] = silu(scale_g[j]·(gate_j·x)) * scale_u[j]·(up_j·x).
    for (int j = t; j < inter; j += nt) {
        float g = dot_i4(x, d.gate_packed + (size_t)j * rb_h, hidden) * d.gate_scale[j];
        float u = dot_i4(x, d.up_packed + (size_t)j * rb_h, hidden) * d.up_scale[j];
        h[j] = siluf(g) * u;
    }
    __syncthreads();

    // Phase B: partial[e][o] = w[e] * scale_d[o] · (down_o · h). Own row, no atomic.
    float we = wexpert[e];
    float* prow = partial + (size_t)e * hidden;
    for (int o = t; o < hidden; o += nt) {
        float dv = dot_i4(h, d.down_packed + (size_t)o * rb_i, inter) * d.down_scale[o];
        prow[o] = we * dv;
    }
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

// Launch the batch over already-resident device memory: each expert into its own
// `partial[E*hidden]` row, then the fixed-order reduce into `out` (fully written,
// no pre-zero needed). Does NOT allocate/copy/synchronize — the caller owns
// x/descs/wexpert/partial/out and joins once per token. Return codes: 0 ok,
// POSITIVE arg guard, NEGATIVE -(hipError_t).
extern "C" int rivoli_moe_experts(
    const float* x, int hidden, int inter, int E,
    const void* descs, const float* wexpert, float* partial, float* out) {
    if (hidden <= 0 || inter <= 0 || E <= 0) return 1001;
    size_t lds = (size_t)inter * sizeof(float);
    // gfx1151 LDS ceiling is 64KB/workgroup (also asserted at model load).
    if (lds > 65536) return 1002;
    hipGetLastError();  // drain any stale error so the check below is this launch's
    hipLaunchKernelGGL(moe_expert_batch, dim3(E), dim3(256), lds, 0,
                       x, hidden, inter, (const ExpertDesc*)descs, wexpert, partial);
    hipError_t e = hipGetLastError();
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
