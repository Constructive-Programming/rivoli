// DSA lightning-indexer decode kernels (device port of the old int4 tree's
// indexer, adapted to the fp8 checkpoint). Per full layer, for one decoded token:
//   k   = rope(layernorm(wk·x))                      [index_head_dim, bf16 cache]
//   q_h = rope(wq_b·q_resid)[h]                       [n_heads · head_dim, f32]
//   w   = weights_proj·x                              [n_heads, f32]
//   I_t = Σ_{h∈active} w_h·wscale·ReLU(q_h·k_t·dscale)
// The top-index_topk scores are picked host-side (small D2H of scores, then the
// host topk), and the rows fed straight into the sparse mla_attend gather. The
// projections wk/wq_b/weights_proj are fp8-e4m3 block-scaled in this checkpoint, so
// they run through the shared `gemv_fp8` (linalg.hip) — NOT a bf16 GEMV. k_norm
// stays bf16 (widened to f32 at placement) and feeds `layernorm` below.
// All pointers device-resident; launch-only (no sync/alloc/copy). Return codes:
// 0 ok, POSITIVE arg guard, NEGATIVE -hipError (shared convention).

#include "common.hpp"

// MISA router block size (pooled tokens per key). MUST match
// crate::indexer::MISA_BLOCK — the running-mean pool the router scores over.
#define MISA_BLOCK 1024

// y[i] = (x[i]-mean)/sqrt(var+eps)·w[i] + b[i]  — LayerNorm (the indexer k_norm,
// which unlike every model RMSNorm ships a bias). One block reduces Σx and Σx²
// in LDS, then every thread normalizes. w,b are f32 (widened at placement).
extern "C" __global__ void layernorm(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, int n, float eps, float* __restrict__ y) {
    __shared__ float sred[256];
    __shared__ float sqred[256];
    int t = threadIdx.x;
    float s = 0.0f, sq = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        float v = x[i];
        s += v;
        sq += v * v;
    }
    sred[t] = s;
    sqred[t] = sq;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) {
            sred[t] += sred[t + st];
            sqred[t] += sqred[t + st];
        }
        __syncthreads();
    }
    float mean = sred[0] / (float)n;
    float var = sqred[0] / (float)n - mean * mean;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = t; i < n; i += blockDim.x) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// kcache[pos·hd + i] = f2bf16(k[i])  — append one indexer key row (bf16), the
// single-stream analog of append_kv.
extern "C" __global__ void index_append(
    const float* __restrict__ k, unsigned short* __restrict__ kcache, int pos, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hd) kcache[(size_t)pos * hd + i] = f2bf16(k[i]);
}

// scores[t] = Σ_{h∈active} w[h]·wscale·ReLU((q_h·k_t)·dscale) over t∈[0,nt).
// One block per cached token; block staged with the key row in LDS. `heads`
// (nullable) lists the `nact` active heads (MISA); null = all `nh` heads (DSA).
extern "C" __global__ void index_score(
    const float* __restrict__ q, const float* __restrict__ w,
    const unsigned short* __restrict__ kcache, const unsigned int* __restrict__ heads,
    int nt, int nact, int hd, float wscale, float dscale,
    float* __restrict__ scores) {
    extern __shared__ float smem[];
    float* kf = smem;      // [hd] staged key row
    float* part = kf + hd; // [blockDim.x] per-head contribution
    int t = blockIdx.x;
    if (t >= nt) return;
    int tid = threadIdx.x;
    const unsigned short* krow = kcache + (size_t)t * hd;
    for (int i = tid; i < hd; i += blockDim.x) kf[i] = bf16f(krow[i]);
    __syncthreads();

    float contrib = 0.0f;
    if (tid < nact) {
        int h = heads ? (int)heads[tid] : tid;
        const float* qh = q + (size_t)h * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; ++i) dot += qh[i] * kf[i];
        contrib = w[h] * wscale * fmaxf(dot * dscale, 0.0f);
    }
    part[tid] = contrib;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[t] = part[0];
}

// MISA block pool: fold token `t`'s indexer key `k` (hd f32) into its block's
// running mean: block b = t/MISA_BLOCK covers [b·B,(b+1)·B); in_block = t%B==0 opens
// the block, else folds with weight 1/(in_block+1) — the exact block-key mean.
// Maintained EVERY token from the SAME roped key that lands in the key cache.
extern "C" __global__ void index_pool_push(
    const float* __restrict__ k, float* __restrict__ pool, int t, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hd) return;
    int in_block = t % MISA_BLOCK;
    float* m = pool + (size_t)(t / MISA_BLOCK) * hd;
    if (in_block == 0) {
        m[i] = k[i];
    } else {
        m[i] += (k[i] - m[i]) / (float)(in_block + 1);
    }
}

// MISA head router (paper arXiv 2605.07363 Eq. 7-8). One block per head j:
//   e[j] = (1/m_blocks) · Σ_b |w[j] · ReLU(Σ_i q[j·hd+i]·pool[b·hd+i])|
// Matches the scalar `(w[j]·dot.max(0)).abs()` sum exactly — no wscale/dscale in the
// router (raw ranking). q/w/pool/e all f32.
extern "C" __global__ void index_head_route(
    const float* __restrict__ q, const float* __restrict__ w,
    const float* __restrict__ pool, int m_blocks, int nh, int hd,
    float* __restrict__ e) {
    __shared__ float red[256];
    int j = blockIdx.x;
    if (j >= nh) return;
    int tid = threadIdx.x;
    const float* qj = q + (size_t)j * hd;
    float wj = w[j];
    float sum = 0.0f;
    for (int b = 0; b < m_blocks; ++b) {
        const float* kb = pool + (size_t)b * hd;
        float p = 0.0f;
        for (int i = tid; i < hd; i += blockDim.x) p += qj[i] * kb[i];
        red[tid] = p;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) sum += fabsf(wj * fmaxf(red[0], 0.0f));
        __syncthreads();
    }
    if (tid == 0) e[j] = sum / (float)(m_blocks > 0 ? m_blocks : 1);
}

// --- device row selection (docs/investigations/npu-offload.md): replaces a host D2H + CPU top-k ---------
// One workgroup: the kernel is ~5 sweeps of nt (nt <= max_ctx), so a grid-wide radix
// would add a global sync and a second launch to tens of microseconds of work.
#define TOPK_BLOCK 256
#define TOPK_BINS 256

// Order-preserving f32 -> u32 (ascending key iff ascending float), with -0.0
// canonicalised to +0.0. The host comparator ties -0.0 with +0.0 (`partial_cmp` says
// they are Equal) and separates them by index; a raw radix over bit patterns would
// instead order -0.0 strictly below, so the canonicalisation is what makes the two
// agree. NaN is out of contract on both sides — the host's comparator maps NaN to
// Equal, which is not a total order — and here a sign-clear NaN sorts above +inf while
// a sign-set NaN sorts below -inf.
__device__ __forceinline__ unsigned int topk_key(float v) {
    unsigned int b = (v == 0.0f) ? 0u : __float_as_uint(v);
    return b ^ ((unsigned int)((int)b >> 31) | 0x80000000u);
}

// Block-wide EXCLUSIVE scan of `v`, block total left in *total. Every thread of the
// block must call it: the barriers are unconditional.
__device__ __forceinline__ unsigned int topk_scan(
    unsigned int* tmp, unsigned int v, int tid, unsigned int* total) {
    tmp[tid] = v;
    __syncthreads();
    for (int off = 1; off < TOPK_BLOCK; off <<= 1) {
        unsigned int add = (tid >= off) ? tmp[tid - off] : 0u;
        __syncthreads(); // read before write, or the shifted read races the update
        tmp[tid] += add;
        __syncthreads(); // write before the next iteration's read
    }
    unsigned int incl = tmp[tid];
    *total = tmp[TOPK_BLOCK - 1];
    __syncthreads(); // both reads retired before a caller reuses `tmp`
    return incl - v;
}

// rows[0..min(k,nt)) = the DSA attend row set, ASCENDING: order by score descending
// and index ascending, take the first k, emit that set in index order.
//
// The index tiebreak is load-bearing whenever the score array carries ties at the k-th
// boundary: the selection is then decided by the index rule, and resolving ties by
// arrival order would be non-deterministic across runs. How tie-heavy the engine's
// scores actually are is UNMEASURED — `index_score` ReLUs every head contribution, which
// is the reason to expect ties, but docs/investigations/npu-offload.md records evidence against the array being
// tie-dominated. The rule is exact either way; only how much of the selection it decides
// is open.
extern "C" __global__ void index_topk(
    const float* __restrict__ s, int nt, int k, unsigned int* __restrict__ rows) {
    __shared__ unsigned int hist[TOPK_BINS];
    __shared__ unsigned int scan[TOPK_BLOCK];
    __shared__ unsigned int sh_pivot, sh_need;
    int tid = threadIdx.x;

    unsigned int pivot = 0u;             // key bits fixed so far
    unsigned int mask = 0u;              // which bits of `pivot` are fixed
    unsigned int need = (unsigned int)k; // top-k entries still to locate

    // Four 8-bit radix passes, MSB first, converging `pivot` on the exact key of the
    // k-th largest score and `need` on how many entries tied at it the selection takes.
    // Each pass re-reads nt and filters on the bits fixed so far: one extra sweep of an
    // array already in cache, against a compaction pass plus its scratch buffer.
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < TOPK_BINS; b += TOPK_BLOCK) hist[b] = 0u;
        __syncthreads();
        for (int i = tid; i < nt; i += TOPK_BLOCK) {
            unsigned int u = topk_key(s[i]);
            if ((u & mask) == pivot) atomicAdd(&hist[(u >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            // Walk bins high to low for the one carrying the k-th entry. The loop test
            // only accumulates after checking `acc + hist[b] < need`, so `acc < need`
            // holds on every exit and `need - acc >= 1` cannot underflow. Falling
            // through to b == 0 is a real path, not dead code — bin 0 is never tested —
            // and it is correct: there `acc = sum_{b>=1} hist[b]`, so
            // `acc + hist[0]` is the whole surviving candidate count, which is >= need
            // by the induction the chosen bin maintains.
            unsigned int acc = 0u;
            int b = TOPK_BINS - 1;
            for (; b > 0; --b) {
                if (acc + hist[b] >= need) break;
                acc += hist[b];
            }
            sh_pivot = pivot | ((unsigned int)b << shift);
            sh_need = need - acc;
        }
        __syncthreads();
        pivot = sh_pivot;
        need = sh_need;
        mask |= 0xFFu << shift;
        __syncthreads();
    }

    // Compaction, ascending. A selected element's output slot has a closed form —
    //   slot(i) = (# keys > pivot before i) + min(# keys == pivot before i, need)
    // — which is monotone in i, so one packed scan yields both prefixes and the rows
    // land already sorted. No atomics, so the order is deterministic. The 16-bit
    // packing is bounded by TOPK_BLOCK, not by nt: the scan only ever carries
    // within-chunk counts (<= 256), while the running totals are separate registers.
    unsigned int base_gt = 0u, base_eq = 0u;
    for (int base = 0; base < nt; base += TOPK_BLOCK) {
        int i = base + tid;
        bool live = i < nt;
        unsigned int u = live ? topk_key(s[i]) : 0u;
        unsigned int gt = (live && u > pivot) ? 1u : 0u;
        unsigned int eq = (live && u == pivot) ? 1u : 0u;
        unsigned int total = 0u;
        unsigned int pre = topk_scan(scan, gt | (eq << 16), tid, &total);
        unsigned int g = base_gt + (pre & 0xFFFFu);
        unsigned int e = base_eq + (pre >> 16);
        if (gt || (eq && e < need)) rows[g + (e < need ? e : need)] = (unsigned int)i;
        base_gt += total & 0xFFFFu;
        base_eq += total >> 16;
    }
}

extern "C" int rivoli_layernorm(
    const float* x, const float* w, const float* b, int n, float eps, float* y) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(layernorm, dim3(1), dim3(256), 0, 0, x, w, b, n, eps, y);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_append(
    const float* k, unsigned short* kcache, int pos, int hd) {
    if (hd <= 0 || pos < 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(index_append, dim3((hd + 255) / 256), dim3(256), 0, 0,
                       k, kcache, pos, hd);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_score(
    const float* q, const float* w, const unsigned short* kcache,
    const unsigned int* heads, int nt, int nh, int nact, int hd,
    float wscale, float dscale, float* scores) {
    if (nt <= 0 || nh <= 0 || nact <= 0 || nact > nh || hd <= 0) return 1001;
    // block = next pow2 ≥ nact (the reduction assumes it), ≥ 32 to stage kf.
    int bd = 1;
    while (bd < nact) bd <<= 1;
    if (bd < 32) bd = 32;
    if (bd > 1024) return 1002;
    size_t lds = ((size_t)hd + bd) * sizeof(float);
    if (lds > 65536) return 1002;
    hipGetLastError();
    hipLaunchKernelGGL(index_score, dim3(nt), dim3(bd), lds, 0,
                       q, w, kcache, heads, nt, nact, hd, wscale, dscale, scores);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_pool_push(
    const float* k, float* pool, int t, int hd) {
    if (hd <= 0 || t < 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(index_pool_push, dim3((hd + 255) / 256), dim3(256), 0, 0,
                       k, pool, t, hd);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_index_head_route(
    const float* q, const float* w, const float* pool,
    int m_blocks, int nh, int hd, float* e) {
    if (m_blocks <= 0 || nh <= 0 || hd <= 0) return 1001;
    // block = next pow2 ≥ hd (the LDS reduction assumes it), capped at red[256].
    int bd = 1;
    while (bd < hd) bd <<= 1;
    if (bd > 256) bd = 256;
    hipGetLastError();
    hipLaunchKernelGGL(index_head_route, dim3(nh), dim3(bd), 0, 0,
                       q, w, pool, m_blocks, nh, hd, e);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : -(int)err;
}

// Launch-only wrapper for `index_topk`. `k` is clamped to `nt`, matching
// `topk_into`'s `k.min(scores.len())`, so a caller may pass index_topk > nt. Note one
// divergence from `topk_into`: k == 0 is an argument error here, where the host returns
// an empty selection. Unreachable today (`index_topk` is validated non-zero at config
// load), but it is a difference, not an equivalence.
extern "C" int rivoli_index_topk(
    const float* scores, int nt, int k, unsigned int* rows) {
    if (nt <= 0 || k <= 0) return 1001;
    if (k > nt) k = nt;
    hipGetLastError();
    hipLaunchKernelGGL(index_topk, dim3(1), dim3(TOPK_BLOCK), 0, 0, scores, nt, k, rows);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// ============= Qwen3.8-Flash-Next's QSA block-key stage — `qwen-architecture.md` §3 =============

// Mean-pool `ratio` consecutive cached indexer keys into one block key and RMSNorm it, one block
// row per HIP block. This is TR Eq. 13 and MOD:679-682 fused: `pooled = keys.float().mean(dim=1)`
// then `k_layernorm(pooled)`.
//
// **Why this is not `index_pool_push`.** That kernel folds one token into a running block mean
// whose width is the compile-time `MISA_BLOCK` (1024). Qwen's `indexer_compress_ratio` is **4**,
// so the geometry is not reachable by any argument — and the reference pools a COMPLETE block
// once rather than accumulating, because the pool feeds a norm and then a RoPE at the block's
// first token position, and a running mean has no such moment.
//
// **Why the norm is FUSED in, and it is a measurement rather than a preference.**
// `RMSNorm(c·x) = RMSNorm(x)` for every `c > 0`, so the norm divides out any per-row scale the
// pool produced. Measured on the vendored prefill window: a pool that SUMS instead of averaging
// moves the composition by 3.39e-5 to 4.84e-5, which is the epsilon term becoming 16x smaller
// against a 4x larger row and not the 4x itself
// (`crates/engine/tests/kernel_qwen_indexer.rs::a_summing_pool_is_cancelled_by_the_norm_that_follows_it`).
// So there is no scoreable boundary between the two stages, splitting them would buy a second
// pass over the keys and no evidence, and — the part that matters for whoever ports this —
// **`.mean` versus `.sum` here has to be read off MOD:681; no fixture over these bytes can tell.**
//
// `eps` is the caller's `rms_norm_eps` and goes on the MEAN, inside the root. The three wrong eps
// forms measure 2.77e-5 to 4.65e-4 at this boundary against a 1.14e-4 tolerance, so only one of
// them is even distinguishable: like the pool's reduction, the epsilon is pinned by READING the
// config field rather than by scoring, which is the conclusion K3's anchor reached for the MLA eps
// on the same kind of arithmetic.
//
// **Only COMPLETE blocks are pooled.** `n_blocks` is `|visible| / ratio` FLOORED (MOD:672), and
// the `|visible| mod ratio` tail is always attended and never scored (MOD:700-701). A kernel that
// rounded up would read `ratio - 1` rows the cache does not hold on three of every four queries.
//
// The block is `next_pow2(hd)` with threads past `hd` masked, for the reason
// `recurrent.hip::gdn_recurrent_f32` states at length: `block_sum_lds` needs a power of two and
// this model's anchor runs `indexer_head_dim` **24** (128 at the real widths), so a guard would
// refuse the only fixture that can score the kernel.
//
// **The keys are f32 here and bf16 in the cache the real model will keep.** MOD pools in fp32 and
// casts back to the input dtype before the norm; at the anchor's fp32 that cast is a no-op, so
// this kernel's agreement says nothing about where a bf16 rounding belongs. Owed to S6 with the
// cache it will read.
//
// USED BY: Qwen3.8-Flash-Next (S6's QSA layers; no engine caller yet — see
// `crates/cli/tests/kernel_coverage.rs`).
extern "C" __global__ void index_pool_norm_f32(const float* __restrict__ keys,
                                              const float* __restrict__ weight, int ratio, int hd,
                                              float eps, float* __restrict__ out) {
    extern __shared__ float red[];
    const int b = (int)blockIdx.x;
    const int t = (int)threadIdx.x;
    // The masking is on the VALUE and never on the `block_sum_lds` call: the ladder is a
    // block-wide barrier and every thread must reach it.
    float m = 0.0f;
    if (t < hd) {
        float acc = 0.0f;
        for (int r = 0; r < ratio; ++r) acc += keys[(size_t)(b * ratio + r) * hd + t];
        m = acc / (float)ratio;
    }
    const float sum = block_sum_lds(m * m, red);
    if (t >= hd) return;
    out[(size_t)b * hd + t] = m * (1.0f / sqrtf(sum / (float)hd + eps)) * weight[t];
}

extern "C" int rivoli_index_pool_norm_f32(const float* keys, const float* weight, int n_blocks,
                                          int ratio, int hd, float eps, float* out, void* stream) {
    if (n_blocks <= 0 || ratio <= 0 || hd <= 0) return 1001;
    if (hd > 1024) return 1002;
    // `ratio == 1` is a per-token norm with a pooling loop around it, which means the caller read
    // `indexer_compress_ratio` as something other than TOKENS PER BLOCK — trap T8, where "budget
    // 2048" and "512 blocks" are the same fact stated twice. Refused rather than defined.
    if (ratio < 2) return 1004;
    // A non-finite or negative eps makes `rsqrt` of a negative variance NaN for the whole block
    // row; zero is legal and is what an exact RMSNorm is, so the bound is one-sided.
    if (!(eps >= 0.0f && eps < INFINITY)) return 1006;
    int block = 1;
    while (block < hd) block <<= 1;
    hipGetLastError();
    hipLaunchKernelGGL(index_pool_norm_f32, dim3((unsigned)n_blocks), dim3((unsigned)block),
                       block * sizeof(float), (hipStream_t)stream, keys, weight, ratio, hd, eps,
                       out);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// The same scoring `blockindex.hip::index_score_blocks` performs, in fp32 throughout.
//
// **This exists because the REUSE was REFUTED on silicon, 2026-09-01, and the deviceless arm
// could not have seen it.** Track C scored `launch_index_score_blocks` against qwen's anchor with
// `w` filled with `1/sqrt(hd)`, argued that a positive uniform scale distributes over the sum, and
// was right about the algebra and wrong about the arithmetic: that kernel rounds to **bf16 three
// times** — after each head's dot, after the per-head weight multiply, and once more on the way
// out — because DeepSeek-V4's indexer runs in bf16 and model.py:426-427 is what it transliterates.
// Qwen's does not: MOD:690-693 computes the whole score in fp32 (`qwen-architecture.md` §3). The
// first device arm measured **2.8701737e-3** against the `indexer` tolerance of 1.14e-4 — which is
// bf16's ~3.9e-3 resolution, not a scale, an ordering or a mask.
//
// **Why no host oracle could have caught it.** The fixture's oracle is f64 and the roundings live
// only inside the kernel, so every deviceless test agreed. That is the whole case for the device
// arm existing, and it is the second time in this port that two implementations of one formula
// differed in precision rather than in algebra.
//
// A separate entry point rather than a flag, which is this tree's standing rule for exactly this
// situation — `swiglu` against `swiglu_clamped_bf16`, `rope_split_half` against `rope_interleave`:
// a V4 call site must not be able to reach fp32 semantics by changing an argument, and a qwen call
// site must not be able to reach bf16 ones. Placed HERE beside `index_pool_norm_f32` rather than in
// `blockindex.hip` beside its bf16 twin, because the two kernels a qwen QSA layer runs back to back
// should be read together; the twin's file carries the V4 spread and compressor it belongs with.
//
// The causal mask and the top-k are the CALLER's, as they are for the twin: this writes the full
// pre-top-k score matrix, which is strictly stronger to compare than a selected set.
//
// USED BY: Qwen3.8-Flash-Next (S6's QSA layers; no engine caller yet — see
// `crates/cli/tests/kernel_coverage.rs`).
#pragma clang fp contract(off)
extern "C" __global__ void index_score_blocks_f32(const float* __restrict__ q,
                                                  const float* __restrict__ kv,
                                                  const float* __restrict__ w,
                                                  float* __restrict__ score, int s, int n_comp,
                                                  int heads, int hd) {
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // `size_t`, for the twin's reason: in `int` the product overflows at ~2.1M entries, reachable
    // at a long prefill, and a negative bound makes every thread return while the launch reports
    // success.
    if (id >= (size_t)s * (size_t)n_comp) return;
    int t = (int)(id / (size_t)n_comp), c = (int)(id % (size_t)n_comp);
    const float* kvr = kv + (size_t)c * hd;
    float acc = 0.0f;
    for (int hh = 0; hh < heads; ++hh) {
        const float* qh = q + ((size_t)t * heads + hh) * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; ++i) dot += qh[i] * kvr[i];
        // ReLU on the fp32 dot, with NO rounding before it. The twin rounds first because the
        // reference it mirrors does; here rounding first would move every score that the rounding
        // carried across zero, and MOD:693's `relu(...)` sees the fp32 value.
        acc += fmaxf(dot, 0.0f) * w[(size_t)t * heads + hh];
    }
    score[id] = acc;
}

extern "C" int rivoli_index_score_blocks_f32(const float* q, const float* kv, const float* w,
                                             float* score, int s, int n_comp, int heads, int hd,
                                             void* stream) {
    if (s <= 0 || n_comp <= 0 || heads <= 0 || hd <= 0) return 1001;
    size_t total = (size_t)s * (size_t)n_comp;
    hipGetLastError();
    hipLaunchKernelGGL(index_score_blocks_f32, dim3((unsigned)((total + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, q, kv, w, score, s, n_comp, heads, hd);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
