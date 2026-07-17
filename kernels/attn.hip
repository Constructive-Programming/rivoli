// M2 MLA attention kernel: flash online-softmax over the compressed KV cache in
// latent space. For one decoded token, per head:
//
//     clat_h = Σ_t softmax_t( (qabs_h·L_t + qrope_h·R_t) · scale ) · L_t
//
// where L_t (kv_lora latent) and R_t (roped key) are the bf16 cache rows. This
// is the O(nt·kvl) work that dominates at long context (the 200k bandwidth
// sink) — the absorb of q_nope through kv_b and the value projection back
// through kv_b stay on the reference path for M2 (small int4 GEMVs, folded into
// this kernel in M3). Mirrors attn.rs: per token the score sums kvl then rope,
// bf16 widened to f32; the flash form replaces the two-pass softmax with a
// single stable online pass (same result to f32 rounding).

#include <hip/hip_runtime.h>

// bf16 → f32: bf16 is the high 16 bits of f32 (matches math.rs bf16_to_f32).
__device__ __forceinline__ float bf16f(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

// One thread per head. `clat` (= global out) doubles as the online accumulator,
// rescaled in place each token — no LDS, correctness over speed.
//
// ponytail: one-thread-per-head with a global-memory accumulator is the M2
// correctness shape; the block-cooperative flash tiling that stages each L_t
// once and reuses it across heads (deferred P1) is the M3 perf step.
extern "C" __global__ void mla_latent_attend(
    const float* __restrict__ qabs,         // [H*kvl]
    const float* __restrict__ qrope,        // [H*rope]
    const unsigned short* __restrict__ Lc,  // [nt*kvl] bf16
    const unsigned short* __restrict__ Rc,  // [nt*rope] bf16
    int H, int nt, int kvl, int rope, float scale,
    float* __restrict__ clat) {             // [H*kvl]
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= H) return;
    const float* qa = qabs + (size_t)head * kvl;
    const float* qr = qrope + (size_t)head * rope;
    float* out = clat + (size_t)head * kvl;

    for (int i = 0; i < kvl; ++i) out[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int t = 0; t < nt; ++t) {
        const unsigned short* Lt = Lc + (size_t)t * kvl;
        const unsigned short* Rt = Rc + (size_t)t * rope;
        float s = 0.0f;
        for (int i = 0; i < kvl; ++i) s += qa[i] * bf16f(Lt[i]);
        for (int d = 0; d < rope; ++d) s += qr[d] * bf16f(Rt[d]);
        s *= scale;
        // Online softmax: rescale the running sum and accumulator to the new max.
        // At t=0, m=-inf ⇒ corr=exp(-inf)=0, p=1: out=L_0, l=1 (clean init).
        float m_new = fmaxf(m, s);
        float corr = expf(m - m_new);
        float p = expf(s - m_new);
        l = l * corr + p;
        for (int i = 0; i < kvl; ++i) out[i] = out[i] * corr + p * bf16f(Lt[i]);
        m = m_new;
    }
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int i = 0; i < kvl; ++i) out[i] *= inv;
}

// Host launcher: copy query + cache to device, launch one thread per head, copy
// the per-head latents back. Per-call alloc (M2 correctness); the resident KV
// slab the kernel reads zero-copy is M3/M4. Returns 0 or a negative HIP error.
extern "C" int rivoli_mla_attend(
    const float* qabs, const float* qrope,
    const unsigned short* Lc, const unsigned short* Rc,
    int H, int nt, int kvl, int rope, float scale,
    float* clat) {
    // Return-code convention (see moe_fused.hip): 0 = success, POSITIVE = arg
    // guard here, NEGATIVE = -(hipError_t). Guards stay out of HIP's enum range.
    if (H <= 0 || nt <= 0 || kvl <= 0 || rope <= 0) return 1001;
    size_t qabs_n = (size_t)H * kvl;
    size_t qrope_n = (size_t)H * rope;
    size_t l_n = (size_t)nt * kvl;
    size_t r_n = (size_t)nt * rope;

    hipError_t err;
    float *dqa = 0, *dqr = 0, *dcl = 0;
    unsigned short *dL = 0, *dR = 0;
#define CK(call) do { err = (call); if (err != hipSuccess) goto fail; } while (0)
    CK(hipMalloc(&dqa, qabs_n * sizeof(float)));
    CK(hipMalloc(&dqr, qrope_n * sizeof(float)));
    CK(hipMalloc(&dcl, qabs_n * sizeof(float)));
    CK(hipMalloc(&dL, l_n * sizeof(unsigned short)));
    CK(hipMalloc(&dR, r_n * sizeof(unsigned short)));
    CK(hipMemcpy(dqa, qabs, qabs_n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dqr, qrope, qrope_n * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dL, Lc, l_n * sizeof(unsigned short), hipMemcpyHostToDevice));
    CK(hipMemcpy(dR, Rc, r_n * sizeof(unsigned short), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(mla_latent_attend, dim3((H + 255) / 256), dim3(256), 0, 0,
                       dqa, dqr, dL, dR, H, nt, kvl, rope, scale, dcl);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(clat, dcl, qabs_n * sizeof(float), hipMemcpyDeviceToHost));
    err = hipSuccess;
fail:
    hipFree(dqa); hipFree(dqr); hipFree(dcl); hipFree(dL); hipFree(dR);
    return err == hipSuccess ? 0 : -(int)err;
#undef CK
}
