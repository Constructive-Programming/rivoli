// Forward-pass glue kernels: the small device ops that stitch the GEMV / norm /
// attention / MoE kernels into one resident token, so nothing round-trips to the
// host mid-layer. All launch-only; caller joins via rivoli_device_sync. Return
// codes: 0 ok, POSITIVE arg guard, NEGATIVE -hipError.

#include "common.hpp"

// x[i] = (int8)embed[token·hidden + i] · scale[token]  — embedding row lookup.
extern "C" __global__ void embed_i8_row(
    const unsigned char* __restrict__ packed, const float* __restrict__ scale,
    int token, int hidden, float* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    x[i] = (float)((signed char)packed[(size_t)token * hidden + i]) * scale[token];
}

// f32 → OCP e4m3 (mirrors math.rs::f32_to_e4m3 bit-for-bit: RNE on the normal
// mantissa, round-half-away on the subnormal, saturating to ±448, 0x7f = NaN) — the
// latent cache's quantizer.
__device__ __forceinline__ unsigned char f2e4m3(float x) {
    if (isnan(x)) return 0x7f;
    unsigned char sign = signbit(x) ? 0x80 : 0;
    float a = fabsf(x);
    if (a >= 448.0f) return sign | 0x7e;
    if (a < 0.0009765625f) return sign; // < 2^-10 → 0
    unsigned int bits;
    __builtin_memcpy(&bits, &a, sizeof(bits));
    int e = (int)((bits >> 23) & 0xff) - 127;
    if (e < -6) {
        // Subnormal e4m3: m/8 · 2^-6; round half away from zero (as Rust .round()).
        // m==8 rounds up to 2^-6 = smallest normal (exp=1,m3=0) — promote, don't clamp.
        int m = (int)floorf(a * 512.0f + 0.5f);
        if (m >= 8) return sign | 0x08;
        return sign | (unsigned char)m;
    }
    unsigned int mant = bits & 0x007fffffu;
    unsigned char m3 = (unsigned char)(mant >> 20);
    unsigned int rem = mant & 0x000fffffu;
    unsigned int half = 0x00080000u;
    if (rem > half || (rem == half && (m3 & 1u))) m3 += 1;
    int exp = e + 7;
    if (m3 == 8) { m3 = 0; exp += 1; }
    if (exp >= 15 && m3 >= 7) return sign | 0x7e;
    return sign | ((unsigned char)exp << 3) | m3;
}

// Append one token's latent as fp8-e4m3 + per-128 block scales (lc8/lscale) and its
// roped key as bf16 (rc), at row `pos`. One block of `kvl` threads: each 128-lane
// group reduces its block amax in LDS, writes scale = amax/448, then quantizes.
// n_blocks = kvl/128; the rope half (ropn ≤ kvl) rides the same block.
extern "C" __global__ void append_kv(
    const float* __restrict__ latent, const float* __restrict__ rope,
    unsigned char* __restrict__ lc8, float* __restrict__ lscale,
    unsigned short* __restrict__ rc, int pos, int kvl, int ropn, int n_blocks) {
    extern __shared__ float red[]; // kvl floats
    int i = threadIdx.x;
    red[i] = fabsf(latent[i]);
    __syncthreads();
    int base = (i / 128) * 128;
    int lane = i - base;
    for (int s = 64; s > 0; s >>= 1) {
        if (lane < s) red[base + lane] = fmaxf(red[base + lane], red[base + lane + s]);
        __syncthreads();
    }
    int b = i / 128;
    float amax = red[b * 128];
    float scl = amax > 0.0f ? amax / 448.0f : 1.0f;
    if (lane == 0) lscale[(size_t)pos * n_blocks + b] = scl;
    lc8[(size_t)pos * kvl + i] = f2e4m3(latent[i] / scl);
    if (i < ropn) rc[(size_t)pos * ropn + i] = f2bf16(rope[i]);
}

// qrope[head·ropn + d] = q[head·qh + nope + d]  — gather each head's roped query
// segment out of the strided q[H·qh] into contiguous qrope[H·ropn] for the flash
// kernel.
extern "C" __global__ void gather_rope(
    const float* __restrict__ q, float* __restrict__ qrope,
    int h, int qh, int nope, int ropn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h * ropn) return;
    int head = idx / ropn;
    int d = idx % ropn;
    qrope[idx] = q[(size_t)head * qh + nope + d];
}

// x[i] += y[i]  — residual add.
extern "C" __global__ void vadd(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Greedy argmax over `logits[0..n]`, reproducing the host fold in gpu.rs::argmax
// EXACTLY: best=0, bv=-inf; for i: if logits[i] > bv { bv=logits[i]; best=i }. The
// combine below is associative + commutative and yields the SAME result set as
// that sequential `>` fold: the maximum VALUE wins; NaN never wins (every compare
// against NaN is false, so it loses to any finite value and to the -inf identity);
// on an exact tie the SMALLER index wins. So `best` = lowest index of the maximum
// value, returned value = logits[best] — matching the host's `bv`, so the caller's
// `!bv.is_finite()` bail is reproduced verbatim. One block, LDS tree reduce.
__device__ __forceinline__ void amax_combine(
    float va, int ia, float vb, int ib, float* vo, int* io) {
    if (vb > va)       { *vo = vb; *io = ib; }   // b strictly greater (b not NaN)
    else if (va > vb)  { *vo = va; *io = ia; }   // a strictly greater (a not NaN)
    else if (va != va) { *vo = vb; *io = ib; }   // a is NaN -> a loses, take b
    else if (vb != vb) { *vo = va; *io = ia; }   // b is NaN -> b loses, take a
    else               { if (ia <= ib) { *vo = va; *io = ia; }  // equal -> smaller index
                         else          { *vo = vb; *io = ib; } }
}

extern "C" __global__ void argmax_reduce(
    const float* __restrict__ logits, int n, int* __restrict__ out_idx,
    float* __restrict__ out_val) {
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    int t = threadIdx.x;
    // Identity mirrors the host initial (best=0, bv=-inf): only wins if EVERY
    // element is NaN/-inf, which the host bails on (value stays non-finite) too.
    float best = -INFINITY;
    int   bidx = 0;
    for (int i = t; i < n; i += blockDim.x) {
        amax_combine(best, bidx, logits[i], i, &best, &bidx);
    }
    sval[t] = best;
    sidx[t] = bidx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            amax_combine(sval[t], sidx[t], sval[t + s], sidx[t + s], &sval[t], &sidx[t]);
        }
        __syncthreads();
    }
    if (t == 0) {
        *out_idx = sidx[0];
        *out_val = sval[0];
    }
}

extern "C" int rivoli_embed_i8_row(
    const unsigned char* packed, const float* scale, int token, int hidden, float* x) {
    if (hidden <= 0 || token < 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(embed_i8_row, dim3((hidden + 255) / 256), dim3(256), 0, 0,
                       packed, scale, token, hidden, x);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_append_kv(
    const float* latent, const float* rope, unsigned char* lc8, float* lscale,
    unsigned short* rc, int pos, int kvl, int ropn, int n_blocks) {
    if (pos < 0 || kvl <= 0 || ropn <= 0 || n_blocks <= 0) return 1001;
    // The one-block per-128 reduction needs kvl a multiple of 128 in [128,1024].
    if (kvl > 1024 || kvl < 128 || (kvl % 128) != 0 || ropn > kvl) return 1002;
    hipGetLastError();
    hipLaunchKernelGGL(append_kv, dim3(1), dim3(kvl), (size_t)kvl * sizeof(float), 0,
                       latent, rope, lc8, lscale, rc, pos, kvl, ropn, n_blocks);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_gather_rope(
    const float* q, float* qrope, int h, int qh, int nope, int ropn) {
    if (h <= 0 || qh <= 0 || nope < 0 || ropn <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(gather_rope, dim3((h * ropn + 255) / 256), dim3(256), 0, 0,
                       q, qrope, h, qh, nope, ropn);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_vadd(float* x, const float* y, int n) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(vadd, dim3((n + 255) / 256), dim3(256), 0, 0, x, y, n);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// Single-block argmax reduction (256 threads, grid-stride). `out_idx`/`out_val`
// are one device word each (the caller's 8-byte result buffer). Launch-only.
extern "C" int rivoli_argmax(
    const float* logits, int n, int* out_idx, float* out_val) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(argmax_reduce, dim3(1), dim3(256), 0, 0, logits, n, out_idx, out_val);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// If any of `x[0..n]` is non-finite, record `tag` in `*flag` — FIRST writer wins, so the
// value names the EARLIEST layer that went bad rather than the last.
//
// Exists because the production guard fires only at `argmax`, i.e. after every layer of
// every position, and so reports that a run is broken without saying where. The obvious
// alternative — copying `x` to the host per layer (`--checksum-x`) — perturbs timing
// enough to MASK the intermittent NaN entirely: a config that fails without it produced
// zero non-finite values across 78 layers of every position with it. So this stays on the
// device and adds NO sync: the flag rides the argmax D2H the tail already pays.
extern "C" __global__ void flag_nonfinite(
    const float* __restrict__ x, int n, unsigned tag, unsigned* __restrict__ flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    // isfinite() on a NaN or +/-Inf; the compare form avoids depending on -ffast-math
    // having left isfinite() meaningful.
    if (!(v > -INFINITY && v < INFINITY)) {
        atomicCAS(flag, 0u, tag);
    }
}

extern "C" int rivoli_flag_nonfinite(const float* x, int n, unsigned tag, unsigned* flag) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(flag_nonfinite, dim3((n + 255) / 256), dim3(256), 0, 0, x, n, tag, flag);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
