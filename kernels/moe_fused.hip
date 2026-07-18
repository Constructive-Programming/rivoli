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
extern "C" __global__ void moe_expert_batch(
    const float* __restrict__ x, int hidden, int inter,
    const ExpertDesc* __restrict__ descs, const float* __restrict__ wexpert,
    float* __restrict__ out) {
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

    // Phase B: out[o] += w[e] * scale_d[o] · (down_o · h).
    float we = wexpert[e];
    for (int o = t; o < hidden; o += nt) {
        float dv = dot_i4(h, d.down_packed + (size_t)o * rb_i, inter) * d.down_scale[o];
        atomicAdd(&out[o], we * dv);
    }
}

// Launch the batch over already-resident device memory. Does NOT allocate, copy,
// or synchronize — the caller owns x/descs/wexpert/out (out must be pre-zeroed)
// and joins once per token. Return-code convention (shared with attn.hip):
// 0 = success, POSITIVE = an argument guard here, NEGATIVE = -(hipError_t), so a
// device fault is never misreported as bad args (hipErrorNoDevice=100 etc.).
extern "C" int rivoli_moe_experts(
    const float* x, int hidden, int inter, int E,
    const void* descs, const float* wexpert, float* out) {
    if (hidden <= 0 || inter <= 0 || E <= 0) return 1001;
    size_t lds = (size_t)inter * sizeof(float);
    // gfx1151 LDS ceiling is 64KB/workgroup (also asserted at model load).
    if (lds > 65536) return 1002;
    hipLaunchKernelGGL(moe_expert_batch, dim3(E), dim3(256), lds, 0,
                       x, hidden, inter, (const ExpertDesc*)descs, wexpert, out);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// One join point per token: block until all launched kernels retire, surfacing
// any async execution fault. Returns 0 or -(hipError_t).
extern "C" int rivoli_device_sync() {
    hipError_t e = hipDeviceSynchronize();
    return e == hipSuccess ? 0 : -(int)e;
}
