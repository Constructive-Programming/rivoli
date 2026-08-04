// vq_codebook — is the codebook gather the MoE kernel's wall, and does 16 KiB lift it?
//
// Probe A for perf-roadmap item #2 ("VQ_K=2048 L1-resident codebook"). The proposal is
// VQ_K 4096 → 2048, which does TWO separable things and the roadmap row prices them as
// one:
//
//   (i)  the fp16 device codebook drops 32 KiB → 16 KiB per projection — the "L1-resident"
//        claim. Needs ONLY a smaller K; the index container is irrelevant to it.
//   (ii) the index shrinks 12 → 11 bits, so an expert drops 15.34 → 14.16 MB (7.7%, not
//        8.3% — the bf16 group scales do not shrink). Needs the 11-bit PACKING, which is
//        where the cost is: at 12 bits an index never spans more than 2 bytes, at 11 bits
//        the period is 8 indices / 11 bytes and an index starting at bit 7 spans 3.
//
// Kernel speed does not depend on index VALUES, only on the codebook's size and the access
// pattern — so all of this is measurable with no requantization at all. Random bytes are
// masked to the arm's index width and the codebook is allocated at the arm's entry count.
//
// The arms, and what each one is for:
//
//   k4096_b12_f16   TODAY. The shipping kernel, reproduced from kernels/moe.hip.
//   k4096_b12_f32   CALIBRATION, not a proposal. The pre-2026 f32 codebook (64 KiB).
//                   common.hpp claims fp16 was a win BECAUSE 32 KiB fits L1 and 64 KiB
//                   does not. If this arm is not slower than the first, the probe is not
//                   sensitive to codebook size at all and every null result below is a
//                   probe artifact rather than a finding. Run it first, believe nothing
//                   until it separates.
//   k2048_b12_f16   Benefit (i) ALONE: 16 KiB codebook, unchanged 12-bit packing and
//                   unchanged bytes. This is the pure L1 question.
//   k2048_b11_f16   THE PROPOSAL: 16 KiB codebook AND 7.7% fewer bytes AND the 11-bit
//                   unpack.
//   k4096_b11_f16   The 11-bit unpack's cost ALONE (not a shippable format — 4096 entries
//                   do not fit in 11 bits — but the kernel does not care, and this is the
//                   only way to separate "the unpack got slower" from "the codebook got
//                   faster" inside the net number).
//   k256_b12_f16    THE ASYMPTOTE. 2 KiB, so gate+up together are 4 KiB and cannot miss
//                   L1 under any occupancy. If this arm does not beat k4096_b12_f16, the
//                   gather is not the wall and item #2's benefit (i) is worth zero — no
//                   value of K can beat K=256.
//
// A dword-load variant of the 11-bit unpack was measured and REMOVED: 1.029x against the
// three-byte form's 1.022x, i.e. the cost is the lost strength-reducible shift, not the
// extra byte load. benchmarks.md carries that; it does not need re-running.
//
// R=1 only. The row batch (`nrow=2`, speculative verify) amortizes the SAME gather over
// more x rows, so it can only dilute a codebook effect, never create one; measuring the
// dominant path is the conservative choice.
//
// Build and run (GPU is sole-tenant and shared — take the lock):
//   hipcc --offload-arch=gfx1151 -O3 -o /tmp/vq_codebook docs/measurement/probes/vq_codebook.hip
//   flock /var/run/sys-gpu.lock -c '/tmp/vq_codebook'

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── verbatim from kernels/common.hpp, so the arms are the shipping arithmetic ──────────
#define WAVE 32
#define ROWS_PER_BLOCK 8
#define MOE_ACC_SHIFT 44
#define MOE_ACC_SCALE ((float)(1ull << MOE_ACC_SHIFT))
#define MOE_ACC_MAX ((float)(1ull << (58 - MOE_ACC_SHIFT)))
#define VQ_DIM 4
#define VQ_GROUP 64
#define VQ_SUBS_PER_GROUP (VQ_GROUP / VQ_DIM)

__device__ __forceinline__ unsigned long long moe_fixed(float v) {
    return (unsigned long long)llrintf(fminf(fmaxf(v, -MOE_ACC_MAX), MOE_ACC_MAX) * MOE_ACC_SCALE);
}
__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}
__device__ __forceinline__ float bf16f(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}
__device__ __forceinline__ float siluf(float x) { return x / (1.0f + expf(-x)); }

// ── the two axes under test ───────────────────────────────────────────────────────────
enum Unpack { U12, U11 };

// Index `t` of a packed row. U12 is byte-for-byte the shipping expression in
// common.hpp::dot_vq_wave_r; the two U11 forms are the real 11-bit unpack, not a
// 16-bit-per-index stand-in — a stand-in would show a win that does not exist.
template <int BITS, Unpack U>
__device__ __forceinline__ int vq_idx(const unsigned char* __restrict__ row, int t) {
    int bitpos = t * BITS;
    int byte = bitpos >> 3;
    int shift = bitpos & 7;
    const unsigned int mask = (1u << BITS) - 1u;
    if (U == U12) {
        // 12 bits from shift ∈ {0,4} ends at bit 15 → never spans more than 2 bytes.
        unsigned int raw = (unsigned int)row[byte] | ((unsigned int)row[byte + 1] << 8);
        return (int)((raw >> shift) & mask);
    } else if (U == U11) {
        // 11 bits from shift ∈ {0..7} ends at bit 17 → 3 bytes.
        unsigned int raw = (unsigned int)row[byte] | ((unsigned int)row[byte + 1] << 8) |
                           ((unsigned int)row[byte + 2] << 16);
        return (int)((raw >> shift) & mask);
    }
}

// fp16 vs f32 codebook gather. Same four products in the same order either way — only the
// bytes per entry change (8 vs 16), which is the whole point of the calibration arm.
template <bool F32>
struct Cb;
template <>
struct Cb<false> {
    typedef __half T;
    __device__ static __forceinline__ float dot(const T* c, float4 x) {
        float2 c01 = __half22float2(*(const __half2*)c);
        float2 c23 = __half22float2(*(const __half2*)(c + 2));
        return x.x * c01.x + x.y * c01.y + x.z * c23.x + x.w * c23.y;
    }
};
template <>
struct Cb<true> {
    typedef float T;
    __device__ static __forceinline__ float dot(const T* c, float4 x) {
        float4 cv = *(const float4*)c;
        return x.x * cv.x + x.y * cv.y + x.z * cv.z + x.w * cv.w;
    }
};

// common.hpp::dot_vq_wave_r<1>, parameterized on index width, unpack form and codebook
// element type. `mask` is applied to the raw index so a random byte pattern always lands
// inside a K-entry codebook — the gather stays scattered (a constant index would turn it
// into a broadcast and time something the real kernel never does).
template <int BITS, Unpack U, bool F32, int K>
__device__ __forceinline__ float dot_vq(const float* __restrict__ x,
                                        const unsigned char* __restrict__ idxrow,
                                        const unsigned short* __restrict__ scalerow,
                                        const typename Cb<F32>::T* __restrict__ cb, int i_dim,
                                        int lane) {
    int nsub = i_dim / VQ_DIM;
    float acc = 0.0f;
    for (int t = lane; t < nsub; t += WAVE) {
        int idx = vq_idx<BITS, U>(idxrow, t) & (K - 1);
        float s = bf16f(scalerow[t / VQ_SUBS_PER_GROUP]);
        float4 xv = *(const float4*)(x + (size_t)t * VQ_DIM);
        acc += s * Cb<F32>::dot(cb + (size_t)idx * VQ_DIM, xv);
    }
    return wave_sum(acc);
}

struct Desc {
    const unsigned char* gate_idx;
    const unsigned short* gate_sc;
    const unsigned char* up_idx;
    const unsigned short* up_sc;
    const unsigned char* down_idx;
    const unsigned short* down_sc;
};

// Row stride in bytes at this index width — the reason the 11-bit arms move fewer bytes.
__device__ __host__ __forceinline__ size_t rb(int i_dim, int bits) {
    return (size_t)(i_dim / VQ_DIM) * bits / 8;
}
__device__ __host__ __forceinline__ size_t ng(int i_dim) { return (size_t)i_dim / VQ_GROUP; }

template <int BITS, Unpack U, bool F32, int K>
__global__ void gateup(const float* __restrict__ x, int hidden, int inter, int e_count,
                       const Desc* __restrict__ descs, const typename Cb<F32>::T* __restrict__ gcb,
                       const typename Cb<F32>::T* __restrict__ ucb, float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * inter) return;
    int e = (int)(r / inter), j = (int)(r % inter);
    const Desc d = descs[e];
    size_t s = rb(hidden, BITS), n = ng(hidden);
    float g = dot_vq<BITS, U, F32, K>(x, d.gate_idx + (size_t)j * s, d.gate_sc + (size_t)j * n,
                                      gcb, hidden, lane);
    float u = dot_vq<BITS, U, F32, K>(x, d.up_idx + (size_t)j * s, d.up_sc + (size_t)j * n, ucb,
                                      hidden, lane);
    if (lane == 0) h_out[(size_t)e * inter + j] = siluf(g) * u;
}

template <int BITS, Unpack U, bool F32, int K>
__global__ void down(int hidden, int inter, int e_count, const Desc* __restrict__ descs,
                     const typename Cb<F32>::T* __restrict__ dcb,
                     const float* __restrict__ wexpert, const float* __restrict__ h_in,
                     unsigned long long* __restrict__ acc) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)e_count * hidden) return;
    int e = (int)(r / hidden), o = (int)(r % hidden);
    const Desc d = descs[e];
    size_t s = rb(inter, BITS), n = ng(inter);
    float dv = dot_vq<BITS, U, F32, K>(h_in + (size_t)e * inter, d.down_idx + (size_t)o * s,
                                       d.down_sc + (size_t)o * n, dcb, inter, lane);
    if (lane == 0) {
        float w = wexpert[e];
        if (w != 0.0f) atomicAdd(&acc[o], moe_fixed(w * dv));
    }
}

// ── host ──────────────────────────────────────────────────────────────────────────────
#define HIDDEN 6144
#define INTER 2048
#define NDESC 9  // top-8 routed + 1 shared, the engine's per-layer batch
#define ITERS 40
#define WARMUP 3

#define CHK(x)                                                                       \
    do {                                                                             \
        hipError_t e_ = (x);                                                         \
        if (e_ != hipSuccess) {                                                      \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static size_t proj_bytes(int o, int i, int bits) { return (size_t)o * (rb(i, bits) + ng(i) * 2); }
static size_t expert_bytes(int bits) {
    return 2 * proj_bytes(INTER, HIDDEN, bits) + proj_bytes(HIDDEN, INTER, bits);
}

static unsigned long long rs = 0x2545F4914F6CDD1Dull;
static unsigned rnd() {
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (unsigned)(rs >> 32);
}

// The result of one arm, so the table can be printed after every arm has run and the
// baseline is known.
struct Row {
    const char* name;
    double gateup_us, down_us;
};

int main() {
    hipDeviceProp_t p;
    CHK(hipGetDeviceProperties(&p, 0));
    printf("%s  CUs=%d  L2=%d KiB  clock=%d MHz\n", p.name, p.multiProcessorCount,
           p.l2CacheSize / 1024, p.clockRate / 1000);
    printf("shapes: hidden=%d inter=%d experts=%d  R=1\n", HIDDEN, INTER, NDESC);
    printf("expert block: 12-bit %.2f MB, 11-bit %.2f MB (%.1f%% fewer bytes)\n\n",
           expert_bytes(12) / 1e6, expert_bytes(11) / 1e6,
           100.0 * (1.0 - (double)expert_bytes(11) / (double)expert_bytes(12)));

    // ONE slab at the widest stride; each arm lays its experts out at its OWN stride, so
    // the 11-bit arms genuinely touch fewer bytes. +64 B of slack covers the 11-bit unpack's
    // 3-byte read at the slab's last index.
    size_t slab = expert_bytes(12) * NDESC + 64;
    unsigned char* hw = (unsigned char*)malloc(slab);
    for (size_t i = 0; i < slab; ++i) hw[i] = (unsigned char)rnd();
    unsigned char* dw;
    CHK(hipMalloc(&dw, slab));
    CHK(hipMemcpy(dw, hw, slab, hipMemcpyHostToDevice));
    free(hw);

    // Codebooks at the widest entry count and widest element; every arm reads its prefix.
    // Small magnitudes so the products stay in range and `moe_fixed` never clamps.
    const int KMAX = 4096;
    float* hcb = (float*)malloc((size_t)KMAX * VQ_DIM * 4);
    for (int i = 0; i < KMAX * VQ_DIM; ++i) hcb[i] = ((float)(rnd() % 2048) - 1024.0f) / 1e5f;
    __half* hcbh = (__half*)malloc((size_t)KMAX * VQ_DIM * 2);
    for (int i = 0; i < KMAX * VQ_DIM; ++i) hcbh[i] = __float2half(hcb[i]);
    void *cb16[3], *cb32[3];
    for (int i = 0; i < 3; ++i) {
        CHK(hipMalloc(&cb16[i], (size_t)KMAX * VQ_DIM * 2));
        CHK(hipMemcpy(cb16[i], hcbh, (size_t)KMAX * VQ_DIM * 2, hipMemcpyHostToDevice));
        CHK(hipMalloc(&cb32[i], (size_t)KMAX * VQ_DIM * 4));
        CHK(hipMemcpy(cb32[i], hcb, (size_t)KMAX * VQ_DIM * 4, hipMemcpyHostToDevice));
    }
    free(hcb);
    free(hcbh);

    float* x;
    CHK(hipMalloc(&x, HIDDEN * 4));
    float* hx = (float*)malloc(HIDDEN * 4);
    for (int i = 0; i < HIDDEN; ++i) hx[i] = ((float)(rnd() % 2048) - 1024.0f) / 1e4f;
    CHK(hipMemcpy(x, hx, HIDDEN * 4, hipMemcpyHostToDevice));
    free(hx);
    float* wexp;
    CHK(hipMalloc(&wexp, NDESC * 4));
    float* hwe = (float*)malloc(NDESC * 4);
    for (int i = 0; i < NDESC; ++i) hwe[i] = 1.0f;  // all routed: no expert skips its atomic
    CHK(hipMemcpy(wexp, hwe, NDESC * 4, hipMemcpyHostToDevice));
    free(hwe);
    float* h;
    CHK(hipMalloc(&h, (size_t)NDESC * INTER * 4));
    unsigned long long* acc;
    CHK(hipMalloc(&acc, (size_t)HIDDEN * 8));
    CHK(hipMemset(acc, 0, (size_t)HIDDEN * 8));

    // Per-arm descriptors: expert e at `e * expert_bytes(bits)`, projections back to back
    // in the on-disk order gate‖up‖down, each indices-then-scales (quant.rs's layout).
    Desc* ddesc;
    CHK(hipMalloc(&ddesc, sizeof(Desc) * NDESC));
    auto build = [&](int bits) {
        Desc hd[NDESC];
        size_t es = expert_bytes(bits);
        size_t gi = 0, gs = (size_t)INTER * rb(HIDDEN, bits);
        size_t ui = proj_bytes(INTER, HIDDEN, bits), us = ui + (size_t)INTER * rb(HIDDEN, bits);
        size_t di = 2 * proj_bytes(INTER, HIDDEN, bits), ds = di + (size_t)HIDDEN * rb(INTER, bits);
        for (int e = 0; e < NDESC; ++e) {
            unsigned char* b = dw + (size_t)e * es;
            hd[e].gate_idx = b + gi;
            hd[e].gate_sc = (const unsigned short*)(b + gs);
            hd[e].up_idx = b + ui;
            hd[e].up_sc = (const unsigned short*)(b + us);
            hd[e].down_idx = b + di;
            hd[e].down_sc = (const unsigned short*)(b + ds);
        }
        CHK(hipMemcpy(ddesc, hd, sizeof(hd), hipMemcpyHostToDevice));
    };

    hipEvent_t e0, e1;
    CHK(hipEventCreate(&e0));
    CHK(hipEventCreate(&e1));
    dim3 ga(((size_t)NDESC * INTER + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), gb(256);
    dim3 gd(((size_t)NDESC * HIDDEN + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    Row rows[7];
    int nrows = 0;

    // One arm: warm up, then time the two kernels SEPARATELY, because they have different
    // reduction dims (6144 vs 2048) and therefore different index-bytes-per-gather ratios
    // — a single fused number would hide which of them the codebook actually helps.
    auto arm = [&](const char* name, int bits, auto launch_gu, auto launch_dn) {
        build(bits);
        for (int i = 0; i < WARMUP; ++i) {
            launch_gu();
            launch_dn();
        }
        CHK(hipDeviceSynchronize());
        float ms_gu, ms_dn;
        CHK(hipEventRecord(e0, 0));
        for (int i = 0; i < ITERS; ++i) launch_gu();
        CHK(hipEventRecord(e1, 0));
        CHK(hipDeviceSynchronize());
        CHK(hipEventElapsedTime(&ms_gu, e0, e1));
        CHK(hipEventRecord(e0, 0));
        for (int i = 0; i < ITERS; ++i) launch_dn();
        CHK(hipEventRecord(e1, 0));
        CHK(hipDeviceSynchronize());
        CHK(hipEventElapsedTime(&ms_dn, e0, e1));
        Row r = {name, ms_gu * 1000.0 / ITERS, ms_dn * 1000.0 / ITERS};
        rows[nrows++] = r;
        printf("%-16s gateup %8.1f us   down %8.1f us   total %8.1f us\n", r.name, r.gateup_us,
               r.down_us, r.gateup_us + r.down_us);
        fflush(stdout);
    };

// The launch pair for one (BITS, unpack, codebook-type, K) point. A macro rather than a
// template because the two kernels take different argument lists and only the template
// arguments vary — writing them out six times is what this hides.
#define ARM(NAME, BITS, U, F32, K)                                                              \
    arm(NAME, BITS,                                                                             \
        [&] {                                                                                   \
            hipLaunchKernelGGL((gateup<BITS, U, F32, K>), ga, gb, 0, 0, x, HIDDEN, INTER, NDESC, \
                               ddesc, (const Cb<F32>::T*)(F32 ? cb32[0] : cb16[0]),              \
                               (const Cb<F32>::T*)(F32 ? cb32[1] : cb16[1]), h);                \
        },                                                                                      \
        [&] {                                                                                   \
            hipLaunchKernelGGL((down<BITS, U, F32, K>), gd, gb, 0, 0, HIDDEN, INTER, NDESC,      \
                               ddesc, (const Cb<F32>::T*)(F32 ? cb32[2] : cb16[2]), wexp, h,     \
                               acc);                                                            \
        })

    ARM("k4096_b12_f16", 12, U12, false, 4096);   // shipping
    ARM("k4096_b12_f32", 12, U12, true, 4096);    // calibration: 64 KiB, pre-fp16
    ARM("k2048_b12_f16", 12, U12, false, 2048);   // benefit (i) alone
    ARM("k2048_b11_f16", 11, U11, false, 2048);   // the proposal, 3-byte unpack
    ARM("k4096_b11_f16", 11, U11, false, 4096);   // unpack cost alone
    ARM("k256_b12_f16", 12, U12, false, 256);     // asymptote: 2 KiB, cannot miss L1
#undef ARM

    // THE ALTERNATIVE ARM, and the reason the gateup/down split above is printed.
    //
    // `gateup` gathers through TWO codebooks at once, so its working set is 2·K entries —
    // 64 KiB at K=4096, not the 32 KiB the roadmap row quotes. `down` gathers through one.
    // That predicts gateup gains MORE from a smaller K than down does, which is exactly
    // what the arms show, and it means there is a second way to halve the gateup working
    // set: give gate and up the SAME codebook and keep K=4096.
    //
    // This arm measures the kernel half of that — both pointers at one 32 KiB codebook, K
    // and the index width untouched. It costs NOTHING in rate (still 3.00 bpw, still
    // 12-bit, still 4096 entries), so unlike every arm above it does not trade quality for
    // the gather. What it DOES cost is one shared fit across two weight distributions the
    // converter currently fits separately — a quality question this probe cannot answer,
    // and the reason `examples/vq_k_probe.rs` grew a joint-codebook arm to price it.
    arm(
        "shared_gu_k4096", 12,
        [&] {
            hipLaunchKernelGGL((gateup<12, U12, false, 4096>), ga, gb, 0, 0, x, HIDDEN, INTER,
                               NDESC, ddesc, (const __half*)cb16[0], (const __half*)cb16[0], h);
        },
        [&] {
            hipLaunchKernelGGL((down<12, U12, false, 4096>), gd, gb, 0, 0, HIDDEN, INTER, NDESC,
                               ddesc, (const __half*)cb16[2], wexp, h, acc);
        });

    double base = rows[0].gateup_us + rows[0].down_us;
    printf("\n%-16s %10s %10s\n", "arm", "total us", "vs today");
    for (int i = 0; i < nrows; ++i) {
        double t = rows[i].gateup_us + rows[i].down_us;
        printf("%-16s %10.1f %9.3fx\n", rows[i].name, t, base / t);
    }
    return 0;
}
