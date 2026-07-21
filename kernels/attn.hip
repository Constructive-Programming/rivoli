// M3 MLA attention kernel (D2): token-tiled flash online-softmax over the
// compressed KV cache in latent space. For one decoded token, per head:
//
//     clat_h = Σ_t softmax_t( (qabs_h·L_t + qrope_h·R_t) · scale ) · L_t
//
// where L_t (kv_lora latent) and R_t (roped key) are the bf16 cache rows shared
// by ALL heads (MLA — one latent per token, not per head). The M2 kernel ran one
// thread per head and re-streamed the whole cache H× from DRAM (~1.5 TB/token at
// 200k — fatal). Here a block owns HB heads (one wavefront each) and stages each
// token tile's L/R into LDS ONCE, reused across all HB heads: DRAM reads of the
// cache drop from H× to ⌈H/HB⌉× (PLAN.md D2). All pointers are device-resident.
//
// Numerics match attn.rs to f32 rounding: per-token score sums kvl then rope over
// bf16-widened values (wave-reduced, so summation order differs from the scalar
// oracle by ~1e-5 rel), and the flash online form replaces the two-pass softmax
// with one numerically-stable pass.
//
// HB/TILE are occupancy/LDS tuning knobs (LDS = (HB+TILE)*kvl + TILE*rope floats);
// the ≥1 tok/s pass (M3 4/4) tunes them. Correctness is independent of their
// value. NOTE: HB is also the DRAM KV re-read multiplier (⌈H/HB⌉×) — the
// dominant term at 200k. Raising HB past 14 needs the accumulator moved from LDS
// to registers (acc[HB*kvl] is what caps it); that's the measurement-driven
// 200k tune, FFI-unchanged.

#include "common.hpp"

#define HB 8        // heads per block (one width-32 subgroup each)
#define TILE 16     // KV tokens staged into LDS per step
// Score-reduction subgroup width. MUST stay a divisor of the physical wavefront
// (32 on gfx1151, 64 if ever compiled wave64) so a head's HB-th 32-lane group
// never straddles two wavefronts: with SUBW=32 and 32|64, head hb owns lanes
// [hb*32, hb*32+32), always inside one wave half, so __shfl(_,_,32) reduces and
// broadcasts within that head only. A future tuner must preserve this invariant.
#define SUBW 32

// OCP e4m3 → f32 (mirrors math.rs::e4m3_to_f32) — the fp8 latent cache's widener.
__device__ __forceinline__ float e4m3f(unsigned char b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    int exp = (b >> 3) & 0x0f;
    float mant = (float)(b & 0x07);
    if (exp == 0) return sign * (mant * 0.125f) * 0.015625f; // 2^-6
    if (exp == 15 && mant == 7.0f) return NAN;
    return sign * (1.0f + mant * 0.125f) * exp2f((float)(exp - 7));
}

// Dynamic LDS bytes: the per-head accumulators plus one staged L/R token tile.
// MUST match the smem carve-up in mla_attend_body below — one source of truth so
// a HB/TILE tune can't desync the launcher's request from the kernel's layout.
static inline size_t mla_attend_lds(int kvl, int rope) {
    return ((size_t)(HB + TILE) * kvl + (size_t)TILE * rope) * sizeof(float);
}

// grid = ⌈H/HB⌉ blocks; block = HB*SUBW threads; dynamic LDS holds the per-head
// accumulators plus the current L/R token tile.
//
// Sparse gather (dsa/streaming/misa modes): `rows` (nullable) lists the `nr`
// cache rows to attend, ascending — the tile loader indirects through it, and
// everything downstream is unchanged (the online softmax normalizes over
// whatever was staged). rows==nullptr means dense: row t0+tt is read directly,
// zero overhead vs the pre-gather kernel. The HOST guarantees every rows[i]
// < the cache's token count — the kernel cannot bounds-check the slab.
//
// `load_lat(t, i)` widens one cached latent element (token row `t`, lane `i`) to
// f32 — the ONLY thing that differs between the bf16 and the fp8 (--kv-fp8)
// latent caches. The head tiling, the sparse gather, and the flash online
// softmax are ONE implementation, instantiated once per cache layout below; the
// roped key Rc is bf16 in both.
template <typename LoadLat>
__device__ __forceinline__ void mla_attend_body(
    const float* __restrict__ qabs,         // [H*kvl]
    const float* __restrict__ qrope,        // [H*rope]
    const unsigned short* __restrict__ Rc,  // [tokens*rope] bf16
    const unsigned int* __restrict__ rows,  // [nr] gather rows, or nullptr = dense
    int H, int nr, int kvl, int rope, float scale,
    float* __restrict__ clat,               // [H*kvl]
    LoadLat load_lat) {
    extern __shared__ float smem[];
    float* acc = smem;              // [HB*kvl] online accumulator, persists all tiles
    float* Lt = acc + HB * kvl;     // [TILE*kvl] staged latent tile
    float* Rt = Lt + TILE * kvl;    // [TILE*rope] staged roped-key tile

    int tid = threadIdx.x;
    int lane = tid & (SUBW - 1);    // 0..31 within the head's subgroup
    int hb = tid / SUBW;            // 0..HB-1 which head in this block
    int head = blockIdx.x * HB + hb;
    bool active = head < H;

    // Online-softmax running max/sum, kept in registers replicated across the
    // head's lanes (every lane sees the same broadcast score, so they stay in
    // lockstep — no LDS/sync needed for m/l).
    float m = -INFINITY, l = 0.0f;
    if (active) {
        for (int i = lane; i < kvl; i += SUBW) acc[hb * kvl + i] = 0.0f;
    }

    const float* qa = qabs + (size_t)(active ? head : 0) * kvl;
    const float* qr = qrope + (size_t)(active ? head : 0) * rope;

    for (int t0 = 0; t0 < nr; t0 += TILE) {
        int tcount = min(TILE, nr - t0);
        // Cooperative tile load (all threads), widening bf16→f32 into LDS,
        // indirecting through `rows` when gathering. kvl/rope are powers of two
        // in GLM (512/64), so the div/mod compile to shifts.
        for (int idx = tid; idx < tcount * kvl; idx += blockDim.x) {
            int tt = idx / kvl, i = idx - tt * kvl;
            size_t t = rows ? (size_t)rows[t0 + tt] : (size_t)(t0 + tt);
            Lt[idx] = load_lat(t, i);
        }
        for (int idx = tid; idx < tcount * rope; idx += blockDim.x) {
            int tt = idx / rope, d = idx - tt * rope;
            size_t t = rows ? (size_t)rows[t0 + tt] : (size_t)(t0 + tt);
            Rt[idx] = bf16f(Rc[t * rope + d]);
        }
        __syncthreads();

        if (active) {
            for (int tt = 0; tt < tcount; ++tt) {
                // score = scale · (qa·L_tt + qr·R_tt), reduced across the subgroup.
                float part = 0.0f;
                for (int i = lane; i < kvl; i += SUBW) part += qa[i] * Lt[tt * kvl + i];
                for (int d = lane; d < rope; d += SUBW) part += qr[d] * Rt[tt * rope + d];
                for (int off = SUBW / 2; off > 0; off >>= 1)
                    part += __shfl_down(part, off, SUBW);
                float s = __shfl(part, 0, SUBW) * scale;  // broadcast the reduced sum

                // Online-softmax update (identical on every lane).
                float m_new = fmaxf(m, s);
                float corr = expf(m - m_new);
                float p = expf(s - m_new);
                for (int i = lane; i < kvl; i += SUBW)
                    acc[hb * kvl + i] = acc[hb * kvl + i] * corr + p * Lt[tt * kvl + i];
                l = l * corr + p;
                m = m_new;
            }
        }
        __syncthreads();  // tile fully consumed before the next overwrites LDS
    }

    if (active) {
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        for (int i = lane; i < kvl; i += SUBW)
            clat[(size_t)head * kvl + i] = acc[hb * kvl + i] * inv;
    }
}

// bf16 latent cache: Lc is [tokens*kvl] bf16, widened by the high-bits shift.
extern "C" __global__ void mla_latent_attend(
    const float* __restrict__ qabs, const float* __restrict__ qrope,
    const unsigned short* __restrict__ Lc, const unsigned short* __restrict__ Rc,
    const unsigned int* __restrict__ rows,
    int H, int nr, int kvl, int rope, float scale,
    float* __restrict__ clat) {
    mla_attend_body(qabs, qrope, Rc, rows, H, nr, kvl, rope, scale, clat,
                    [=](size_t t, int i) { return bf16f(Lc[t * kvl + i]); });
}

// fp8 latent cache (--kv-fp8): Lc8 is [tokens*kvl] e4m3, Lscale is
// [tokens*n_blocks] f32 per-128 block scales (n_blocks = kvl/128).
extern "C" __global__ void mla_latent_attend_fp8(
    const float* __restrict__ qabs, const float* __restrict__ qrope,
    const unsigned char* __restrict__ Lc8, const float* __restrict__ Lscale,
    const unsigned short* __restrict__ Rc, const unsigned int* __restrict__ rows,
    int H, int nr, int kvl, int rope, int n_blocks, float scale,
    float* __restrict__ clat) {
    mla_attend_body(qabs, qrope, Rc, rows, H, nr, kvl, rope, scale, clat,
                    [=](size_t t, int i) {
                        return e4m3f(Lc8[t * kvl + i]) * Lscale[t * n_blocks + i / 128];
                    });
}

// Launch the flash attention over already-resident device memory. Does NOT
// allocate/copy/synchronize — the caller owns qabs/qrope/Lc/Rc/clat and joins
// once per token via rivoli_device_sync. Return-code convention (shared with
// moe_fused.hip): 0 = success, POSITIVE = arg guard here, NEGATIVE = -(hipError_t).
extern "C" int rivoli_mla_attend(
    const float* qabs, const float* qrope,
    const unsigned short* Lc, const unsigned short* Rc,
    const unsigned int* rows, int H, int nr, int kvl, int rope, float scale,
    float* clat) {
    if (H <= 0 || nr <= 0 || kvl <= 0 || rope <= 0) return 1001;
    size_t lds = mla_attend_lds(kvl, rope);
    if (lds > 65536) return 1002;  // exceeds gfx1151 LDS (shrink TILE/HB)
    int blocks = (H + HB - 1) / HB;
    hipGetLastError();  // drain any stale error so the check below is this launch's
    hipLaunchKernelGGL(mla_latent_attend, dim3(blocks), dim3(HB * SUBW), lds, 0,
                       qabs, qrope, Lc, Rc, rows, H, nr, kvl, rope, scale, clat);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// fp8-cache variant of rivoli_mla_attend; same contract and return codes.
extern "C" int rivoli_mla_attend_fp8(
    const float* qabs, const float* qrope,
    const unsigned char* Lc8, const float* Lscale,
    const unsigned short* Rc, const unsigned int* rows,
    int H, int nr, int kvl, int rope, int n_blocks, float scale, float* clat) {
    if (H <= 0 || nr <= 0 || kvl <= 0 || rope <= 0 || n_blocks <= 0) return 1001;
    size_t lds = mla_attend_lds(kvl, rope);
    if (lds > 65536) return 1002;
    int blocks = (H + HB - 1) / HB;
    hipGetLastError();
    hipLaunchKernelGGL(mla_latent_attend_fp8, dim3(blocks), dim3(HB * SUBW), lds, 0,
                       qabs, qrope, Lc8, Lscale, Rc, rows, H, nr, kvl, rope, n_blocks,
                       scale, clat);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
