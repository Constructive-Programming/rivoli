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

// --- Split-KV (flash-decoding) launch geometry ---------------------------
//
// The head tiling alone yields grid = ⌈H/HB⌉ = 8 blocks at GLM dims (H=64,
// HB=8) — ~20% of a 40-CU gfx1151, with the ENTIRE KV sweep serialised inside
// each block's tile loop. So `attn` grew with context (77ms@128tok →
// 114ms@512tok) while every other bucket stayed flat.
//
// Fix: partition the `nr` attended rows into `n_splits` TILE-aligned chunks and
// give each (head-block, split) pair its own workgroup — grid.y = n_splits.
// Each computes a PARTIAL online-softmax result over its own row range (its own
// running max `m`, sum `l`, unnormalised accumulator), and a second small kernel
// (mla_attend_combine) merges them with the standard stable rescale.
//
// MLA_MAX_SPLITS bounds grid.y AND the caller's scratch (see
// rivoli_mla_attend_scratch_floats — the caller allocates for the max, once, so
// there is never a per-token hipMalloc: PLAN.md's M3 wiring contract).
#define MLA_MAX_SPLITS 16
// Target block count. LDS ≈ 52 KB/workgroup caps occupancy at ~1 workgroup per
// CU, so ~2× the 40 CUs gives the scheduler slack without a long tail.
#define MLA_TARGET_BLOCKS 80
// Floor on work per split: below this the split's own prologue/epilogue and the
// combine pass cost more than the parallelism buys.
#define MLA_MIN_TILES_PER_SPLIT 4

// How the `nr` rows are cut. `tps` = TILE steps per split, so split s owns rows
// [s*tps*TILE, min(nr, (s+1)*tps*TILE)) — TILE-aligned, so every split but the
// last runs full-width tiles. `n_splits` is recomputed from `tps` so the ceil
// division can never leave a trailing empty split.
//
// It is a pure function of (H, nr): the same context length always produces the
// same cut, hence the same summation order, hence bit-identical results run to
// run (greedy decode requires schedule-independence).
struct MlaSplitPlan {
    int n_splits;
    int tps;
};

static inline MlaSplitPlan mla_plan_splits(int H, int nr, bool have_scratch) {
    int ntiles = (nr + TILE - 1) / TILE;
    int hblocks = (H + HB - 1) / HB;
    int by_work = ntiles / MLA_MIN_TILES_PER_SPLIT;
    int by_grid = (MLA_TARGET_BLOCKS + hblocks - 1) / hblocks;
    int n = by_work < by_grid ? by_work : by_grid;
    if (n > MLA_MAX_SPLITS) n = MLA_MAX_SPLITS;
    if (n < 1 || !have_scratch) n = 1;  // degenerate = exactly the pre-split path
    int tps = (ntiles + n - 1) / n;
    MlaSplitPlan p;
    p.tps = tps;
    p.n_splits = (ntiles + tps - 1) / tps;
    return p;
}

// f32 scratch the caller must own for the partials, sized for the WORST case
// (MLA_MAX_SPLITS) so it is allocated once per session, never per token.
// Layout: [MLA_MAX_SPLITS*H*kvl] accumulators, then [MLA_MAX_SPLITS*H*2] (m,l)
// pairs. Both indexed [split*H + head]. Exported so src/gpu.rs sizes its
// DeviceBuf from this one definition instead of restating the arithmetic.
extern "C" size_t rivoli_mla_attend_scratch_floats(int H, int kvl) {
    if (H <= 0 || kvl <= 0) return 0;
    return (size_t)MLA_MAX_SPLITS * (size_t)H * ((size_t)kvl + 2);
}

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

// grid = (⌈H/HB⌉, n_splits) blocks; block = HB*SUBW threads; dynamic LDS holds
// the per-head accumulators plus the current L/R token tile.
//
// `blockIdx.y` is the split: this block attends only rows
// [y*tps*TILE, min(nr, (y+1)*tps*TILE)). When `partial` is null (n_splits==1)
// the block owns every row and writes the normalised result straight to `clat`
// — bit-identical to the pre-split kernel. Otherwise it writes its UNNORMALISED
// accumulator plus its (m, l) to `partial`, and mla_attend_combine merges.
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
    int H, int nr, int kvl, int rope, float scale, int tps,
    float* __restrict__ clat,               // [H*kvl], written iff partial==null
    float* __restrict__ partial,            // split scratch, or null = single split
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

    // This split's row range. Block-uniform, so every __syncthreads below is
    // still reached by all threads.
    int r_begin = (int)blockIdx.y * tps * TILE;
    int r_end = min(nr, r_begin + tps * TILE);

    for (int t0 = r_begin; t0 < r_end; t0 += TILE) {
        int tcount = min(TILE, r_end - t0);
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
        if (partial) {
            // Unnormalised partial: acc = Σ_t exp(s_t − m)·L_t over THIS split's
            // rows, with (m, l) alongside. Normalising here would lose the
            // information the combine needs. An empty split (r_begin ≥ r_end,
            // only reachable defensively) stores m = −inf, l = 0, acc = 0, which
            // the combine's exp(−inf − m_global) = 0 weight annihilates.
            float* pa = partial + ((size_t)blockIdx.y * H + head) * kvl;
            float* pml = partial + (size_t)MLA_MAX_SPLITS * H * kvl +
                         ((size_t)blockIdx.y * H + head) * 2;
            for (int i = lane; i < kvl; i += SUBW) pa[i] = acc[hb * kvl + i];
            if (lane == 0) {
                pml[0] = m;
                pml[1] = l;
            }
        } else {
            float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            for (int i = lane; i < kvl; i += SUBW)
                clat[(size_t)head * kvl + i] = acc[hb * kvl + i] * inv;
        }
    }
}

// Merge the per-split partials into the final `clat`. One block per head,
// blockDim threads striding the kvl columns.
//
// Numerical stability: each split stored acc_s = Σ exp(s_t − m_s)·L_t and
// l_s = Σ exp(s_t − m_s), both with a NON-POSITIVE exponent, so neither can
// overflow. Rescaling by w_s = exp(m_s − m_global) — also a non-positive
// exponent, since m_global = max_s m_s — turns acc_s into Σ exp(s_t −
// m_global)·L_t exactly. Summing those and dividing by Σ w_s·l_s is therefore
// the same globally-max-shifted softmax the single-block kernel computes; at no
// point is an unshifted (potentially overflowing) exponential formed.
//
// Determinism: both reductions walk splits in ascending index order in a plain
// loop — no atomics, no scheduling-dependent order — so greedy decode stays
// schedule-independent.
extern "C" __global__ void mla_attend_combine(
    const float* __restrict__ partial, int H, int kvl, int n_splits,
    float* __restrict__ clat) {
    __shared__ float w[MLA_MAX_SPLITS];  // exp(m_s − m_global) per split
    __shared__ float inv_l;

    int head = blockIdx.x;
    if (head >= H) return;
    const float* ml = partial + (size_t)MLA_MAX_SPLITS * H * kvl;

    if (threadIdx.x == 0) {
        float m_g = -INFINITY;
        for (int s = 0; s < n_splits; ++s) m_g = fmaxf(m_g, ml[((size_t)s * H + head) * 2]);
        // m_g non-finite ⇒ every split was empty (or every score was −inf):
        // exp(m_s − m_g) would be NaN, so short-circuit to the zero output the
        // single-block kernel's `l > 0 ? 1/l : 0` guard also produces.
        float l_g = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float* p = ml + ((size_t)s * H + head) * 2;
            w[s] = isfinite(m_g) ? expf(p[0] - m_g) : 0.0f;
            l_g += p[1] * w[s];
        }
        inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < kvl; i += blockDim.x) {
        float sum = 0.0f;
        for (int s = 0; s < n_splits; ++s)
            sum += partial[((size_t)s * H + head) * kvl + i] * w[s];
        clat[(size_t)head * kvl + i] = sum * inv_l;
    }
}

// Shared epilogue for both cache layouts: enqueue the combine when the attend
// was split. Same (null) stream, so it is ordered after the attend launch.
static inline int mla_launch_combine(int n_splits, int H, int kvl, const float* partial,
                                     float* clat) {
    if (n_splits <= 1) return 0;  // attend already wrote clat directly
    hipLaunchKernelGGL(mla_attend_combine, dim3(H), dim3(256), 0, 0, partial, H, kvl, n_splits,
                       clat);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// bf16 latent cache: Lc is [tokens*kvl] bf16, widened by the high-bits shift.
extern "C" __global__ void mla_latent_attend(
    const float* __restrict__ qabs, const float* __restrict__ qrope,
    const unsigned short* __restrict__ Lc, const unsigned short* __restrict__ Rc,
    const unsigned int* __restrict__ rows,
    int H, int nr, int kvl, int rope, float scale, int tps,
    float* __restrict__ clat, float* __restrict__ partial) {
    mla_attend_body(qabs, qrope, Rc, rows, H, nr, kvl, rope, scale, tps, clat, partial,
                    [=](size_t t, int i) { return bf16f(Lc[t * kvl + i]); });
}

// fp8 latent cache (--kv-fp8): Lc8 is [tokens*kvl] e4m3, Lscale is
// [tokens*n_blocks] f32 per-128 block scales (n_blocks = kvl/128).
extern "C" __global__ void mla_latent_attend_fp8(
    const float* __restrict__ qabs, const float* __restrict__ qrope,
    const unsigned char* __restrict__ Lc8, const float* __restrict__ Lscale,
    const unsigned short* __restrict__ Rc, const unsigned int* __restrict__ rows,
    int H, int nr, int kvl, int rope, int n_blocks, float scale, int tps,
    float* __restrict__ clat, float* __restrict__ partial) {
    mla_attend_body(qabs, qrope, Rc, rows, H, nr, kvl, rope, scale, tps, clat, partial,
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
    float* clat, float* partial) {
    if (H <= 0 || nr <= 0 || kvl <= 0 || rope <= 0) return 1001;
    size_t lds = mla_attend_lds(kvl, rope);
    if (lds > 65536) return 1002;  // exceeds gfx1151 LDS (shrink TILE/HB)
    int blocks = (H + HB - 1) / HB;
    MlaSplitPlan sp = mla_plan_splits(H, nr, partial != nullptr);
    float* part = sp.n_splits > 1 ? partial : nullptr;
    hipGetLastError();  // drain any stale error so the check below is this launch's
    hipLaunchKernelGGL(mla_latent_attend, dim3(blocks, sp.n_splits), dim3(HB * SUBW), lds, 0,
                       qabs, qrope, Lc, Rc, rows, H, nr, kvl, rope, scale, sp.tps, clat, part);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    return mla_launch_combine(sp.n_splits, H, kvl, partial, clat);
}

// fp8-cache variant of rivoli_mla_attend; same contract and return codes.
extern "C" int rivoli_mla_attend_fp8(
    const float* qabs, const float* qrope,
    const unsigned char* Lc8, const float* Lscale,
    const unsigned short* Rc, const unsigned int* rows,
    int H, int nr, int kvl, int rope, int n_blocks, float scale, float* clat,
    float* partial) {
    if (H <= 0 || nr <= 0 || kvl <= 0 || rope <= 0 || n_blocks <= 0) return 1001;
    size_t lds = mla_attend_lds(kvl, rope);
    if (lds > 65536) return 1002;
    int blocks = (H + HB - 1) / HB;
    MlaSplitPlan sp = mla_plan_splits(H, nr, partial != nullptr);
    float* part = sp.n_splits > 1 ? partial : nullptr;
    hipGetLastError();
    hipLaunchKernelGGL(mla_latent_attend_fp8, dim3(blocks, sp.n_splits), dim3(HB * SUBW), lds, 0,
                       qabs, qrope, Lc8, Lscale, Rc, rows, H, nr, kvl, rope, n_blocks,
                       scale, sp.tps, clat, part);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    return mla_launch_combine(sp.n_splits, H, kvl, partial, clat);
}
