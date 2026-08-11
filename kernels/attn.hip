// MLA flash attention over the compressed KV cache in latent space. For one
// decoded token, per head:
//
//     clat_h = Σ_t softmax_t( (qabs_h·L_t + qrope_h·R_t) · scale ) · L_t
//
// L_t (kv_lora latent) and R_t (roped key) are the cache rows shared by ALL heads
// (MLA — one latent per token, not per head). A block owns HB heads (one wavefront
// each) and stages each token tile's L/R into LDS ONCE (widened to f32), reused
// across all HB heads, so DRAM reads of the cache drop from H× to ⌈H/HB⌉×.
//
// The latent cache is fp8-e4m3 + per-128-block f32 scales (halves KV footprint at
// long context; DeepSeek's shipped MLA layout): Lc8 is [tokens*kvl] e4m3, Lscale is
// [tokens*n_blocks] f32, n_blocks = kvl/128, element (t,i) = e4m3(Lc8[t*kvl+i]) *
// Lscale[t*n_blocks + i/128]. The roped key stays bf16 (quantizing the rope half
// hurts accuracy — DeepSeek keeps it bf16 too).
//
// Attends either the whole causal prefix (dense, `rows` == null ⇒ token = row
// position) or a DSA-selected subset (`rows[j]` is the cached token index of the
// j-th attended row). The split-KV plan cuts the `nr` row POSITIONS, so only the
// tile load dereferences `rows` — everything else is row-position arithmetic.
// Numerics match the scalar oracle to f32 rounding: per-token score sums kvl then
// rope over the widened values (wave-reduced, so summation order differs by ~1e-5
// rel), flash online form for stability.
//
// HB/TILE are occupancy/LDS knobs (LDS = TILE*(kvl + rope) floats — the per-head
// accumulator is in REGISTERS, see MLA_ACC_REGS). Correctness is independent of their
// value. HB is also the DRAM KV re-read multiplier (⌈H/HB⌉×) — the dominant term at
// long context.

#include "common.hpp"

// 8 → 16 on 2026-08-02, measured. HB is the DRAM KV re-read multiplier (⌈H/HB⌉×), so
// doubling it halves the cache traffic: **226.5 → 108.8 µs at nr512 (2.08×) and 769.8 →
// 421.2 µs at nr2048 (1.83×)**, min-of-5 interleaved, and −3.2 ms/token of `route` in
// engine. Free in registers (47 VGPRs, no spill, at both values) and LDS is unchanged —
// the tile is TILE-sized and `acc` moved to registers earlier, which is what decoupled HB
// from LDS and made this possible at all.
//
// It is NOT bit-identical above nr≈640, and the roadmap said it would be. `by_work` =
// ⌈nr/TILE⌉/MLA_MIN_TILES_PER_SPLIT binds below that and gives both HB values the same
// cut; above it `by_grid` = ⌈MLA_TARGET_BLOCKS/⌈H/HB⌉⌉ binds, and doubling HB doubles it,
// so the split plan and therefore the f32 summation order change. Gated on that basis:
// paired dNLL **+0.00217 nats, 95% CI [−0.00243, +0.00676]** against a 0.00995 bar
// (ppl-corpus-5000, --attn dense, both arms reproduced byte-identically twice).
//
// `MLA_MIN_TILES_PER_SPLIT` 4→2 was swept alongside and is INERT at both HB values —
// `tps` rounds back — so it stays at 4. See docs/measurement/benchmarks.md, "The MLA HB sweep".
#define HB 16       // heads per block (one width-32 subgroup each)
#define TILE 16     // KV tokens staged into LDS per step
// Score-reduction subgroup width. MUST stay a divisor of the physical wavefront
// (32 on gfx1151) so a head's HB-th 32-lane group never straddles two wavefronts.
#define SUBW 32
// Per-lane online-accumulator registers. Lane `l` of a head owns exactly the columns
// i ≡ l (mod SUBW), so the accumulator is lane-private and needs ⌈kvl/SUBW⌉ registers
// — 16 at the GLM kv_lora_rank of 512. This CAPS the kernel's kvl at
// MLA_ACC_REGS*SUBW; `rivoli_mla_attend` enforces it (arg guard 1004) rather than
// letting a larger model silently drop columns.
#define MLA_ACC_REGS 16

// --- Split-KV (flash-decoding) launch geometry ---------------------------
// Head tiling alone yields grid = ⌈H/HB⌉ = 8 blocks at GLM dims (H=64, HB=8) —
// ~20% of a 40-CU part, with the ENTIRE KV sweep serialised inside each block. So
// attn grew with context (77ms@128 → 114ms@512). Fix: partition the `nr` attended
// rows into `n_splits` TILE-aligned chunks, one workgroup per (head-block, split)
// (grid.y = n_splits). Each computes a PARTIAL online-softmax over its row range;
// mla_attend_combine merges them with the standard stable rescale.
#define MLA_MAX_SPLITS 16       // bounds grid.y AND the caller's once-per-session scratch
#define MLA_TARGET_BLOCKS 80    // ~2× the 40 CUs
#define MLA_MIN_TILES_PER_SPLIT 4  // below this, split overhead + combine costs more than it buys

// How the `nr` rows are cut. `tps` = TILE steps per split, so split s owns rows
// [s*tps*TILE, min(nr, (s+1)*tps*TILE)) — TILE-aligned, every split but the last
// runs full-width tiles. A pure function of (H, nr): same context length → same
// cut → same summation order → bit-identical results (greedy decode needs it).
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

// f32 scratch the caller owns for the partials, sized for the WORST case
// (MLA_MAX_SPLITS) so it is allocated once per session, never per token. Layout:
// [MLA_MAX_SPLITS*H*kvl] accumulators, then [MLA_MAX_SPLITS*H*2] (m,l) pairs, both
// indexed [split*H + head]. Exported so src/gpu.rs sizes its DeviceBuf from this
// one definition instead of restating the arithmetic.
extern "C" size_t rivoli_mla_attend_scratch_floats(int H, int kvl) {
    if (H <= 0 || kvl <= 0) return 0;
    return (size_t)MLA_MAX_SPLITS * (size_t)H * ((size_t)kvl + 2);
}

// Dynamic LDS bytes: one staged L/R token tile (the per-head accumulators live in
// registers). MUST match the smem carve-up in mla_latent_attend — one source of truth
// so a HB/TILE tune can't desync the launcher's request from the kernel's layout.
static inline size_t mla_attend_lds(int kvl, int rope) {
    return ((size_t)TILE * kvl + (size_t)TILE * rope) * sizeof(float);
}

// grid = (⌈H/HB⌉, n_splits) blocks; block = HB*SUBW threads; dynamic LDS holds the
// current L/R token tile (the per-head accumulators are in registers, see
// MLA_ACC_REGS). `blockIdx.y` is the split:
// this block attends only rows [y*tps*TILE, min(nr, (y+1)*tps*TILE)). When
// `partial` is null (n_splits==1) the block owns every row and writes the
// normalised result straight to `clat`. Otherwise it writes its UNNORMALISED
// accumulator plus its (m, l) to `partial`, and mla_attend_combine merges.
extern "C" __global__ void mla_latent_attend(
    const float* __restrict__ qabs,         // [H*kvl]
    const float* __restrict__ qrope,        // [H*rope]
    const unsigned char* __restrict__ Lc8,  // [tokens*kvl] e4m3 latent cache
    const float* __restrict__ Lscale,       // [tokens*n_blocks] f32 per-128 block scale
    const unsigned short* __restrict__ Rc,  // [tokens*rope] bf16 roped key
    const unsigned int* __restrict__ rows,  // [nr] attended token indices, or null = dense
    int H, int nr, int kvl, int rope, int n_blocks, float scale, int tps,
    float* __restrict__ clat,               // [H*kvl], written iff partial==null
    float* __restrict__ partial) {          // split scratch, or null = single split
    extern __shared__ float smem[];
    float* Lt = smem;               // [TILE*kvl] staged latent tile
    float* Rt = Lt + TILE * kvl;    // [TILE*rope] staged roped-key tile

    int tid = threadIdx.x;
    int lane = tid & (SUBW - 1);    // 0..31 within the head's subgroup
    int hb = tid / SUBW;            // 0..HB-1 which head in this block
    int head = blockIdx.x * HB + hb;
    bool active = head < H;

    // INVARIANT: every unrolled acc loop below must bound on `k < nacc`, never on
    // `i < kvl`. `i = lane + k*SUBW` makes `i < kvl` lane-divergent, and LLVM then
    // predicates each unrolled step with its own exec-mask save/restore — which costs
    // more than the LDS traffic this whole change removes. `nacc` is wave-uniform.
    // kvl is a multiple of SUBW because the launcher rejects anything that is not a
    // multiple of 128. Measured both ways in docs/measurement/benchmarks.md, "Read the ISA before you
    // book the device".
    int nacc = kvl / SUBW;

    // Online accumulator, ONE REGISTER PER LANE-OWNED COLUMN. This was an [HB*kvl] LDS
    // array, but lane `lane` only ever touched columns i ≡ lane (mod SUBW) — it was
    // lane-private all along. Registers drop the LDS by HB*kvl floats and delete an LDS
    // read-modify-write from the innermost loop. Bit-identical: same columns, same
    // values, same order. Zeroing all MLA_ACC_REGS is safe even when nacc is smaller or
    // the lane is inactive — registers are private, so the extras are never read back.
    float acc[MLA_ACC_REGS] = {0};

    // Online-softmax running max/sum, kept in registers replicated across the
    // head's lanes (every lane sees the same broadcast score → lockstep, no sync).
    float m = -INFINITY, l = 0.0f;

    const float* qa = qabs + (size_t)(active ? head : 0) * kvl;
    const float* qr = qrope + (size_t)(active ? head : 0) * rope;

    // This split's row range. Block-uniform, so every __syncthreads is still
    // reached by all threads.
    int r_begin = (int)blockIdx.y * tps * TILE;
    int r_end = min(nr, r_begin + tps * TILE);

    for (int t0 = r_begin; t0 < r_end; t0 += TILE) {
        int tcount = min(TILE, r_end - t0);
        // Cooperative tile load (all threads), widening into LDS: latent fp8→f32
        // via e4m3 × its per-128 block scale, key bf16→f32. kvl/rope are powers of
        // two in GLM (512/64), so the div/mod compile to shifts.
        for (int idx = tid; idx < tcount * kvl; idx += blockDim.x) {
            int tt = idx / kvl, i = idx - tt * kvl;
            size_t t = rows ? (size_t)rows[t0 + tt] : (size_t)(t0 + tt);
            Lt[idx] = e4m3f(Lc8[t * kvl + i]) * Lscale[t * n_blocks + i / 128];
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
#pragma unroll
                for (int k = 0; k < MLA_ACC_REGS; ++k)
                    if (k < nacc) acc[k] = acc[k] * corr + p * Lt[tt * kvl + lane + k * SUBW];
                l = l * corr + p;
                m = m_new;
            }
        }
        __syncthreads();  // tile fully consumed before the next overwrites LDS
    }

    if (active) {
        if (partial) {
            // Unnormalised partial: acc = Σ_t exp(s_t − m)·L_t over THIS split's
            // rows, (m, l) alongside — normalising here would lose what the combine
            // needs. An empty split stores m = −inf, l = 0, acc = 0, which the
            // combine's exp(−inf − m_global) = 0 weight annihilates.
            float* pa = partial + ((size_t)blockIdx.y * H + head) * kvl;
            float* pml = partial + (size_t)MLA_MAX_SPLITS * H * kvl +
                         ((size_t)blockIdx.y * H + head) * 2;
#pragma unroll
            for (int k = 0; k < MLA_ACC_REGS; ++k)
                if (k < nacc) pa[lane + k * SUBW] = acc[k];
            if (lane == 0) {
                pml[0] = m;
                pml[1] = l;
            }
        } else {
            float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
#pragma unroll
            for (int k = 0; k < MLA_ACC_REGS; ++k)
                if (k < nacc) clat[(size_t)head * kvl + lane + k * SUBW] = acc[k] * inv;
        }
    }
}

// Merge the per-split partials into the final `clat`. One block per head,
// blockDim threads striding the kvl columns. Each split stored acc_s = Σ exp(s_t −
// m_s)·L_t and l_s = Σ exp(s_t − m_s), both non-positive exponent (can't overflow).
// Rescaling by w_s = exp(m_s − m_global) (also non-positive, m_global = max_s m_s)
// turns acc_s into Σ exp(s_t − m_global)·L_t exactly; summing and dividing by
// Σ w_s·l_s is the same globally-max-shifted softmax the single-block kernel
// computes. Both reductions walk splits in ascending order → schedule-independent.
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
        // m_g non-finite ⇒ every split empty (or every score −inf): exp(m_s − m_g)
        // would be NaN, so short-circuit to the zero output the single-block
        // kernel's `l > 0 ? 1/l : 0` guard also produces.
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

// Launch the flash attention over already-resident device memory. Does NOT
// allocate/copy/synchronize — the caller owns qabs/qrope/Lc/Rc/clat and joins once
// per token via rivoli_device_sync. Return codes: 0 = success, POSITIVE = arg
// guard, NEGATIVE = -(hipError_t).
extern "C" int rivoli_mla_attend(
    const float* qabs, const float* qrope,
    const unsigned char* Lc8, const float* Lscale, const unsigned short* Rc,
    const unsigned int* rows, int H, int nr, int kvl, int rope, int n_blocks, float scale,
    float* clat, float* partial) {
    if (H <= 0 || nr <= 0 || kvl <= 0 || rope <= 0 || n_blocks <= 0) return 1001;
    size_t lds = mla_attend_lds(kvl, rope);
    if (lds > 65536) return 1002;  // exceeds gfx1151 LDS (shrink TILE — HB no longer counts)
    // The accumulator is register-resident and indexed by k = (i - lane)/SUBW, so kvl
    // must fit MLA_ACC_REGS registers and tile SUBW exactly (`nacc` is the uniform bound
    // the unrolled loops test against). Enforced at 128 rather than SUBW because the
    // tile load's `Lscale[t*n_blocks + i/128]` needs kvl a multiple of 128 too: kvl=160
    // would satisfy %SUBW, give n_blocks=1, and silently read the NEXT token's block
    // scale for i in [128,160). 128 is a multiple of SUBW, so this subsumes both.
    if (kvl > MLA_ACC_REGS * SUBW || kvl % 128) return 1004;
    int blocks = (H + HB - 1) / HB;
    MlaSplitPlan sp = mla_plan_splits(H, nr, partial != nullptr);
    float* part = sp.n_splits > 1 ? partial : nullptr;
    hipGetLastError();  // drain any stale error so the check below is this launch's
    hipLaunchKernelGGL(mla_latent_attend, dim3(blocks, sp.n_splits), dim3(HB * SUBW), lds, 0,
                       qabs, qrope, Lc8, Lscale, Rc, rows, H, nr, kvl, rope, n_blocks, scale, sp.tps,
                       clat, part);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -(int)e;
    if (sp.n_splits <= 1) return 0;  // attend wrote clat directly
    hipLaunchKernelGGL(mla_attend_combine, dim3(H), dim3(256), 0, 0, partial, H, kvl,
                       sp.n_splits, clat);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// ═══════════════════════════════════════════════════════════════════════════════════
// DeepSeek-V4-Flash attention core — S2b of docs/investigations/v4-flash-port.md
// ═══════════════════════════════════════════════════════════════════════════════════
//
// `kernel.py::sparse_attn`. Nothing above applies: V4 is MQA over ONE `head_dim`-wide
// entry that is simultaneously the key and the value for all `h` heads (`T.gemm` is
// called twice against the same `kv_shared`), gathered by an explicit per-query index
// list rather than by a causal bound.
//
// FP CONTRACTION OFF, same argument as the V4 section of mla.hip: the logit dot and the
// value accumulation below run in the SAME order as `Oracle::sparse_attn`'s folds, so
// with contraction off the only divergence left in this kernel is `expf` and the two
// block reductions — which makes the goldens a far sharper instrument than they would
// otherwise be. This kernel reads 16 MB/token/layer out of `kv` (128 slots x 512 f32 x
// 64 head-blocks), so it should be bound by that and not by VALU — but the pragma's cost
// has NOT been measured, and S4 is where it gets priced rather than asserted.
#pragma clang fp contract(off)

#define GATHER_ATTN_THREADS 256
// Output dims per thread. Caps `head_dim` at GATHER_ATTN_THREADS * GATHER_ATTN_ACC = 1024; the
// launcher rejects more rather than dropping dims, which is what an unchecked bound
// would do — silently, and only in the tail of every head.
#define GATHER_ATTN_ACC 4

// o[t][head] = ( Σ_j softmax_j(q[t][head]·kv[idx[t][j]] · scale) · kv[idx[t][j]] )
// with `attn_sink[head]` added to the softmax DENOMINATOR ONLY.
//
// THE SINK IS NOT A KEY. `kernel.py:345-346` adds `exp(attn_sink[i] - scores_max[i])` to
// `sum_exp` after the last block and never adds a matching term to `acc_o`: it is a
// learned per-head leak of probability mass. Note that "a sink treated as a real key
// with a zero value vector" is algebraically THIS, so that is not a distinct mistake;
// the one this shape actually invites is forgetting the running-max shift, which weights
// the sink by a factor of exp(max) and is `Defect::AttnSinkNotMaxShifted`.
//
// `idx < 0` masks a slot (`get_window_topk_idxs` pads with -1 before the window fills,
// and masks the acausal half at prefill). Masked slots contribute nothing to the max,
// the sum or the accumulator, and are never dereferenced.
//
// A FULLY-MASKED QUERY POISONS ITS OUTPUT WITH NaN rather than producing 0/inf. The
// oracle asserts on this case (`query {t} head {hh} attends to nothing`); a kernel
// cannot, and 0/inf = 0 is a plausible-looking answer for a query that attended to
// nothing at all. NaN propagates to the comparison, which is the loudest thing available
// here. It is unreachable in the reference — every query's own position is in its window
// — so this guards a wiring mistake in the index list, not a model state.
//
// Two passes, not the reference's online (flash) softmax: the same mathematics, and the
// logits fit LDS at every `topk` this model reaches (128 window slots at ratio 0, plus
// `index_topk` compressed slots where S2c adds them). One block per (query row, head).
extern "C" __global__ void gather_attn_shared_kv(const float* __restrict__ q,
                                           const float* __restrict__ kv,
                                           const float* __restrict__ sink,
                                           const int* __restrict__ idxs, int h, int d,
                                           int topk, float scale, float* __restrict__ o) {
    extern __shared__ float lds[];
    float* qs = lds;          // [d]   this head's query row
    float* lg = lds + d;      // [topk] logits, then softmax weights
    __shared__ float red[GATHER_ATTN_THREADS];
    __shared__ float bc[2];   // running max, then the denominator

    const int tid = (int)threadIdx.x;
    const int head = (int)blockIdx.x, t = (int)blockIdx.y;
    const float* qrow = q + ((size_t)t * h + head) * d;
    const int* ix = idxs + (size_t)t * topk;
    for (int i = tid; i < d; i += GATHER_ATTN_THREADS) qs[i] = qrow[i];
    __syncthreads();

    // Logits. One thread owns a whole key, so the d-loop is sequential in the same order
    // as the oracle's `for i in 0..d { acc += qh[i] * k[i] }`.
    float mx = -INFINITY;
    for (int j = tid; j < topk; j += GATHER_ATTN_THREADS) {
        int p = ix[j];
        float l = -INFINITY;
        if (p >= 0) {
            const float* kr = kv + (size_t)p * d;
            float a = 0.0f;
            for (int i = 0; i < d; ++i) a += qs[i] * kr[i];
            l = a * scale;
        }
        lg[j] = l;
        mx = fmaxf(mx, l);
    }
    red[tid] = mx;
    __syncthreads();
    for (int s = GATHER_ATTN_THREADS >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    if (tid == 0) bc[0] = red[0];
    __syncthreads();
    mx = bc[0];

    float ps = 0.0f;
    for (int j = tid; j < topk; j += GATHER_ATTN_THREADS) {
        float e = (ix[j] >= 0) ? expf(lg[j] - mx) : 0.0f;
        lg[j] = e;
        ps += e;
    }
    red[tid] = ps;
    __syncthreads();
    for (int s = GATHER_ATTN_THREADS >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) bc[1] = isfinite(mx) ? red[0] + expf(sink[head] - mx) : NAN;
    __syncthreads();
    const float denom = bc[1];

    // Values. `j` ascends over ALL slots, which is the oracle's accumulation order; the
    // dims a thread owns are `GATHER_ATTN_THREADS` apart, so a wave reads 32 consecutive
    // floats of `kv` per step.
    float acc[GATHER_ATTN_ACC];
#pragma unroll
    for (int c = 0; c < GATHER_ATTN_ACC; ++c) acc[c] = 0.0f;
    for (int j = 0; j < topk; ++j) {
        int p = ix[j];
        if (p < 0) continue;
        const float* kr = kv + (size_t)p * d;
        const float e = lg[j];
#pragma unroll
        for (int c = 0; c < GATHER_ATTN_ACC; ++c) {
            int i = tid + c * GATHER_ATTN_THREADS;
            if (i < d) acc[c] += e * kr[i];
        }
    }
    // `sparse_attn` writes a bf16 tensor.
    float* orow = o + ((size_t)t * h + head) * d;
#pragma unroll
    for (int c = 0; c < GATHER_ATTN_ACC; ++c) {
        int i = tid + c * GATHER_ATTN_THREADS;
        if (i < d) orow[i] = bf16f(f2bf16(acc[c] / denom));
    }
}

// Takes a STREAM. Sixth and last of `attn::v4::attention`'s launcher set, all converted
// together on 2026-08-05 — see `mla.hip`'s "V4 launchers" banner for why the set cannot be
// converted in parts, and `linalg.hip::rivoli_memcpy_dtod_async` for the seventh member.
extern "C" int rivoli_gather_attn_shared_kv(const float* q, const float* kv, const float* sink,
                                            const int* idxs, int m, int h, int d, int topk,
                                            float scale, float* o, void* stream) {
    if (m <= 0 || h <= 0 || d <= 0 || topk <= 0) return 1001;
    // 1002 is the size-bound code this tree uses (attn.hip's flash launcher, indexer.hip).
    // THIS is the guard between a `head_dim` past the per-thread accumulator and output
    // dims that are silently never written — the tail of every head, with no error and no
    // fault. Nothing in the model approaches it (512 against a cap of 1024), which is
    // exactly why it has to be a hard rejection rather than a comment.
    if (d > GATHER_ATTN_THREADS * GATHER_ATTN_ACC) return 1002;
    size_t lds = ((size_t)d + (size_t)topk) * sizeof(float);
    // The static `red`/`bc` are counted here too: they come out of the same 64 KB.
    if (lds + sizeof(float) * (GATHER_ATTN_THREADS + 2) > 65536) return 1006;
    hipGetLastError();
    hipLaunchKernelGGL(gather_attn_shared_kv, dim3((unsigned)h, (unsigned)m), dim3(GATHER_ATTN_THREADS),
                       (unsigned)lds, (hipStream_t)stream, q, kv, sink, idxs, h, d, topk, scale,
                       o);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// ═══════════════════════════════════════════════════════════════════════════════════
// Grouped-query attention with an optional sliding-window ring
// ═══════════════════════════════════════════════════════════════════════════════════
//
// Used by Muse Glimmer (32 Q heads / 2 KV heads, head_dim 128, window 2048 on 3 of every
// 4 layers) — S2 item 1 of docs/investigations/glimmer-port.md. Nothing above fits it:
// `mla_latent_attend` has ONE latent row per position that is simultaneously key and
// value, and `gather_attn_shared_kv` is MQA over an explicit index list. This is the
// first kernel in the tree with a distinct V, more than one KV head, and a causal bound
// it derives instead of being handed.
//
//     out[i][h] = Σ_j softmax_j( q[i][h] · k[j][h/group] · scale ) · v[j][h/group]
//
// Four things about it are load-bearing, each recorded in
// docs/reference/glimmer-architecture.md §4/§9 as a defect that decodes FLUENTLY:
//
//  1. `kvh = head / group`, NOT `head % hkv`. The reference's `repeat_kv` is
//     `expand(b, hkv, group, s, d).reshape(b, hkv*group, s, d)` (M:249), which is a
//     per-head BLOCK, so Q heads 0..15 share KV head 0. Both mappings type-check, both
//     produce the right shapes, and `% hkv` interleaves them instead — trap 10.
//  2. The window is INCLUSIVE of the query's own position: `j ∈ [p-win+1, p]`, exactly
//     `win` rows. transformers' `sliding_window_overlay` is `kv_idx > q_idx - win`
//     and-ed with causal `kv_idx <= q_idx`; at win=3 row 4 sees {2,3,4}. Off-by-one
//     here is trap 14 and it is why `lo` is `p - win + 1` and not `p - win`.
//  3. `win == 0` means a GLOBAL layer — the whole causal prefix, no ring. A ring on a
//     global layer would silently drop everything older than `ring_cap`, so the two are
//     mutually exclusive and the launcher rejects the combination rather than picking one.
//  4. No mask array is taken. At Glimmer's 131072 context a [tq][s] mask is 68 GB of
//     nothing, so the bound is DERIVED from (start_pos, win) — and the S1b golden's
//     captured `attend.mask` is what proves the derivation, rather than being fed to it.
//
// Q must arrive POST-`qk_norm`, POST-`qk_scale_factor` and POST-rope, K the same minus the
// scale (§4: the 3.87 multiplies Q only, and the cache holds post-norm post-rope K); this
// kernel applies only `scale`, which stays `head_dim**-0.5`. Layout is the engine's
// `[row][head][dim]`, NOT the reference's heads-first `[b][head][row][dim]` — the caller
// transposes, and `attn.rs` names that deviation at its call site.
//
// `#pragma clang fp contract(off)` from the V4 section above is still in effect and is
// wanted here for the same reason: the dot below folds in the same order as the reference's,
// so the goldens measure `expf` and the wave reduction and nothing else.
//
// ponytail: one subgroup per (query row, Q head) — grid (hq, tq) is 32 blocks at decode
// against 40 CUs, and the whole KV sweep is serialised inside each. Correctness first;
// `mla_latent_attend`'s two levers are the upgrade path when S5 prices this — tile HB Q
// heads per block (all 16 of a group share one KV head, so the LDS tile is read once for
// the group rather than once per head) and split the row range with a combine pass.
#define GQA_THREADS 32   // one wave32 subgroup owns one (query row, Q head)
#define GQA_TILE 16      // KV rows staged into LDS per step; LDS = 2*GQA_TILE*d floats
// Lane-private accumulator/query registers. Lane `l` owns columns i ≡ l (mod GQA_THREADS),
// so this CAPS head_dim at GQA_THREADS*GQA_ACC = 256 — Glimmer is 128, and §9 trap 15
// records that head_dim is NOT hidden/heads (128, not 208), so the cap is checked rather
// than assumed. The launcher rejects a larger head_dim instead of dropping its tail.
#define GQA_ACC 8
extern "C" __global__ void gqa_attend(
    const float* __restrict__ q,   // [tq*hq*d]   post-norm, post-scale, post-rope
    const float* __restrict__ k,   // [cap*hkv*d] cache: post-norm, post-rope
    const float* __restrict__ v,   // [cap*hkv*d]
    int hq, int hkv, int d, int start_pos, int win, int ring_cap, float scale,
    float* __restrict__ out) {     // [tq*hq*d]
    extern __shared__ float smem[];
    float* Kt = smem;                     // [GQA_TILE*d] staged key tile
    float* Vt = Kt + GQA_TILE * d;        // [GQA_TILE*d] staged value tile

    const int lane = (int)threadIdx.x;
    const int head = (int)blockIdx.x, row = (int)blockIdx.y;
    const int kvh = head / (hq / hkv);    // item 1 above — a block repeat, not an interleave
    const int pos = start_pos + row;      // absolute position of this query row
    // Item 2: inclusive of `pos` itself, so exactly `win` rows once the window has filled.
    const int lo = (win > 0 && pos >= win) ? pos - win + 1 : 0;
    // Rounded UP, and every access below is guarded by `i < d`, so a head_dim that does not
    // tile the subgroup runs correctly with the tail lanes idle. `mla_latent_attend` refuses
    // exactly this and bounds on a wave-uniform `nacc` instead, because there the divergent
    // form cost more than the LDS traffic it saved — a measured decision on the hot path, at
    // one fixed width. Here it buys the thing that matters more: the S1b goldens are emitted
    // at head_dim 8, so a kernel that demanded a multiple of 32 could not be gated by its own
    // fixture at all. `gather_attn_shared_kv` takes the same guarded form for the same reason.
    const int nacc = (d + GQA_THREADS - 1) / GQA_THREADS;

    const float* qrow = q + ((size_t)row * hq + head) * d;
    float qv[GQA_ACC];
#pragma unroll
    for (int c = 0; c < GQA_ACC; ++c) {
        const int i = lane + c * GQA_THREADS;
        qv[c] = (c < nacc && i < d) ? qrow[i] : 0.0f;
    }

    float acc[GQA_ACC] = {0};
    float m = -INFINITY, l = 0.0f;

    for (int j0 = lo; j0 <= pos; j0 += GQA_TILE) {
        const int cnt = min(GQA_TILE, pos - j0 + 1);
        // Cooperative tile load. `ring_cap` maps an absolute position to its slot; 0 means
        // the cache is indexed by position directly (global layers, and any caller that
        // holds the whole prefix). K and V share the slot, so one index serves both.
        for (int idx = lane; idx < cnt * d; idx += GQA_THREADS) {
            const int tt = idx / d, c = idx - tt * d;
            const int j = j0 + tt;
            const size_t slot = ring_cap ? (size_t)(j % ring_cap) : (size_t)j;
            const size_t at = (slot * (size_t)hkv + (size_t)kvh) * (size_t)d + (size_t)c;
            Kt[idx] = k[at];
            Vt[idx] = v[at];
        }
        __syncthreads();

        for (int tt = 0; tt < cnt; ++tt) {
            float part = 0.0f;
#pragma unroll
            for (int c = 0; c < GQA_ACC; ++c) {
                const int i = lane + c * GQA_THREADS;
                if (c < nacc && i < d) part += qv[c] * Kt[tt * d + i];
            }
            for (int off = GQA_THREADS / 2; off > 0; off >>= 1)
                part += __shfl_down(part, off, GQA_THREADS);
            const float s = __shfl(part, 0, GQA_THREADS) * scale;

            const float mx = fmaxf(m, s);
            const float rescale = expf(m - mx), w = expf(s - mx);
#pragma unroll
            for (int c = 0; c < GQA_ACC; ++c) {
                const int i = lane + c * GQA_THREADS;
                if (c < nacc && i < d) acc[c] = acc[c] * rescale + w * Vt[tt * d + i];
            }
            l = l * rescale + w;
            m = mx;
        }
        __syncthreads();  // tile consumed before the next load overwrites it
    }

    // `l` is at least exp(0) = 1 for every reachable query, because item 2 puts `pos` in
    // its own window and the causal bound puts it in the prefix — a query here always
    // attends to at least itself. 0 rather than 0/0 if a caller ever wires `lo > pos`.
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    float* orow = out + ((size_t)row * hq + head) * d;
#pragma unroll
    for (int c = 0; c < GQA_ACC; ++c) {
        const int i = lane + c * GQA_THREADS;
        if (c < nacc && i < d) orow[i] = acc[c] * inv;
    }
}

// Takes a STREAM, like the V4 launchers above. Return codes: 0 = success, POSITIVE = arg
// guard, NEGATIVE = -(hipError_t).
extern "C" int rivoli_gqa_attend(const float* q, const float* k, const float* v, int hq,
                                 int hkv, int d, int tq, int start_pos, int win, int ring_cap,
                                 float scale, float* out, void* stream) {
    if (hq <= 0 || hkv <= 0 || d <= 0 || tq <= 0 || start_pos < 0 || win < 0 || ring_cap < 0)
        return 1001;
    // The group must be exact: `hq/group` truncating would send the tail of the Q heads to
    // a KV head that does not exist, and `head/group` is computed from the truncated value.
    if (hq % hkv) return 1003;
    // 1002 is the size-bound code this tree uses. Past the accumulator the tail of every head
    // is silently never written — no error, no fault, just wrong columns. A head_dim that
    // does not tile the subgroup is fine and is NOT rejected: the goldens are emitted at 8.
    if (d > GQA_THREADS * GQA_ACC) return 1002;
    // Item 3: a ring on a global layer drops the prefix past `ring_cap` with no error.
    // A ring shorter than the window is the same bug one step later — the oldest rows the
    // window still wants have already been overwritten.
    if (win == 0 && ring_cap != 0) return 1005;
    if (ring_cap != 0 && ring_cap < win) return 1005;
    const size_t lds = 2 * (size_t)GQA_TILE * (size_t)d * sizeof(float);
    if (lds > 65536) return 1006;
    hipGetLastError();
    hipLaunchKernelGGL(gqa_attend, dim3((unsigned)hq, (unsigned)tq), dim3(GQA_THREADS),
                       (unsigned)lds, (hipStream_t)stream, q, k, v, hq, hkv, d, start_pos, win,
                       ring_cap, scale, out);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
