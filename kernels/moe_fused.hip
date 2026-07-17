// M2 fused MoE expert-batch kernel. One launch computes a whole batch of
// experts that share the input `x` and the (hidden, inter) shape:
//
//     out[o] += Σ_e w[e] * down_e( silu(gate_e·x) ⊙ up_e·x )[o]
//
// One workgroup per expert: phase A computes h[j]=silu(gate_j)⊙up_j into LDS,
// a workgroup barrier, phase B does the down projection and atomicAdd's the
// weighted result into the shared `out`. No grid-wide barrier, no per-expert
// dispatch (colibri's loss) — the batch is one kernel launch.
//
// Numerics mirror the scalar oracle (quant.rs matvec_i4_rows): each output row
// is owned by one thread that sums i ascending, so the int4 dot products match
// the reference bit-for-bit; only silu(expf) and cross-expert atomicAdd order
// introduce ~1e-6 drift, well within the int4 dequant tolerance (M2 gate).

#include <hip/hip_runtime.h>

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
    const unsigned char* __restrict__ gate_packed, const float* __restrict__ gate_scale,
    const unsigned char* __restrict__ up_packed,   const float* __restrict__ up_scale,
    const unsigned char* __restrict__ down_packed, const float* __restrict__ down_scale,
    const float* __restrict__ wexpert,
    float* __restrict__ out) {
    extern __shared__ float h[];  // [inter]
    int e = blockIdx.x;
    int t = threadIdx.x;
    int nt = blockDim.x;

    size_t rb_h = (hidden + 1) / 2;  // packed bytes per gate/up row
    size_t rb_i = (inter + 1) / 2;   // packed bytes per down row

    const unsigned char* gp = gate_packed + (size_t)e * inter * rb_h;
    const unsigned char* upp = up_packed + (size_t)e * inter * rb_h;
    const float* gs = gate_scale + (size_t)e * inter;
    const float* us = up_scale + (size_t)e * inter;

    // Phase A: h[j] = silu(scale_g[j]·(gate_j·x)) * scale_u[j]·(up_j·x).
    for (int j = t; j < inter; j += nt) {
        float g = dot_i4(x, gp + (size_t)j * rb_h, hidden) * gs[j];
        float u = dot_i4(x, upp + (size_t)j * rb_h, hidden) * us[j];
        h[j] = siluf(g) * u;
    }
    __syncthreads();

    // Phase B: out[o] += w[e] * scale_d[o] · (down_o · h).
    const unsigned char* dp = down_packed + (size_t)e * hidden * rb_i;
    const float* ds = down_scale + (size_t)e * hidden;
    float we = wexpert[e];
    for (int o = t; o < hidden; o += nt) {
        float d = dot_i4(h, dp + (size_t)o * rb_i, inter) * ds[o];
        atomicAdd(&out[o], we * d);
    }
}

// Host launcher: copy inputs to device, zero `out`, launch the batch, copy the
// hidden-length contribution back. Allocates/frees per call — the M2 correctness
// path; the resident device tier (persistent buffers, zero-copy feed) is M3.
// Returns 0 on success or a negative HIP error code.
extern "C" int rivoli_moe_experts(
    const float* x, int hidden, int inter, int E,
    const unsigned char* gate_packed, const float* gate_scale,
    const unsigned char* up_packed, const float* up_scale,
    const unsigned char* down_packed, const float* down_scale,
    const float* wexpert, float* out) {
    if (hidden <= 0 || inter <= 0 || E <= 0) return -100;
    size_t rb_h = (size_t)(hidden + 1) / 2;
    size_t rb_i = (size_t)(inter + 1) / 2;
    size_t gate_bytes = (size_t)E * inter * rb_h;  // == up_bytes
    size_t down_bytes = (size_t)E * hidden * rb_i;
    size_t gscale_n = (size_t)E * inter;
    size_t dscale_n = (size_t)E * hidden;
    size_t lds = (size_t)inter * sizeof(float);
    // gfx1151 LDS ceiling is 64KB/workgroup; fail loudly rather than launch
    // with clamped shared memory (M2 shapes stay well under this).
    if (lds > 65536) return -101;

    hipError_t err;
    float *dx = 0, *dgs = 0, *dus = 0, *dds = 0, *dw = 0, *dout = 0;
    unsigned char *dgp = 0, *dupp = 0, *ddp = 0;
#define CK(call) do { err = (call); if (err != hipSuccess) goto fail; } while (0)
    CK(hipMalloc(&dx, (size_t)hidden * sizeof(float)));
    CK(hipMalloc(&dgp, gate_bytes));
    CK(hipMalloc(&dupp, gate_bytes));
    CK(hipMalloc(&ddp, down_bytes));
    CK(hipMalloc(&dgs, gscale_n * sizeof(float)));
    CK(hipMalloc(&dus, gscale_n * sizeof(float)));
    CK(hipMalloc(&dds, dscale_n * sizeof(float)));
    CK(hipMalloc(&dw, (size_t)E * sizeof(float)));
    CK(hipMalloc(&dout, (size_t)hidden * sizeof(float)));

    CK(hipMemcpy(dx, x, (size_t)hidden * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dgp, gate_packed, gate_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dupp, up_packed, gate_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(ddp, down_packed, down_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dgs, gate_scale, gscale_n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dus, up_scale, gscale_n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dds, down_scale, dscale_n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, wexpert, (size_t)E * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemset(dout, 0, (size_t)hidden * sizeof(float)));

    hipLaunchKernelGGL(moe_expert_batch, dim3(E), dim3(256), lds, 0,
                       dx, hidden, inter, dgp, dgs, dupp, dus, ddp, dds, dw, dout);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out, dout, (size_t)hidden * sizeof(float), hipMemcpyDeviceToHost));
    err = hipSuccess;
fail:
    hipFree(dx); hipFree(dgp); hipFree(dupp); hipFree(ddp);
    hipFree(dgs); hipFree(dus); hipFree(dds); hipFree(dw); hipFree(dout);
    return err == hipSuccess ? 0 : -(int)err;
#undef CK
}
