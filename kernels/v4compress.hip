// V4-Flash `Compressor.forward` (model.py:322-386) — the ATTENTION compressor, both
// branches (overlap at ratio 4, non-overlap at ratio 128) and both phases.
//
// S2c of docs/investigations/v4-flash-port.md. Scored against `src/v4oracle/`'s
// `Oracle::compressor` by `tests/v4_compress_kernel.rs`.
//
// WHAT IS DELIBERATELY NOT HERE, and where it is instead:
//
//   * `act_quant` — the partial block-64 fp8 quantization of dims [0, d-rd). It is a
//     SEPARATE call in the reference (`act_quant(kv[..., :-rd], 64, ...)` at model.py:378,
//     after `apply_rotary_emb`), and S2b already ships it as `mla.hip::v4_act_quant` /
//     `rivoli_v4_act_quant` with exactly the (row_stride, n, block) signature this needs.
//     `v4compress.rs::compress` launches it. Writing a second one here would be a third
//     copy of e4m3-RNE in this tree — see the note on `f2e4m3_rne` below.
//   * The indexer's compressor (`rotate = true`: Hadamard + fp4 instead of partial fp8).
//     Out of S2c's scope by the plan; it needs S2a's e2m1/e8m0 block.
//
// A NOTE ON DUPLICATION, because this file had to choose. `kernels/` is NOT watched by
// `build.rs`'s jscpd gate (which covers `src/`, `tests/` and itself), so nothing mechanical
// stops a kernel from re-deriving arithmetic another kernel already has — twice over, for
// two agents who could not see each other's work. S3 requirement 11 lifted the survivors
// into `common.hpp`; the gate is still blind to this directory, so the hazard stands. See
// `docs/investigations/v4-flash-port.md`, "The duplication gate cannot see `kernels/`".

#include "common.hpp"

// Entries pooled per output block: `coff * ratio` — 8 at ratio 4 (overlap doubles it), 128
// at ratio 128, which is the largest the shipped config produces and so the bound. It sizes
// the LDS pointer tables; guard 1004 refuses anything larger rather than letting a future
// ratio silently walk off them.
#define V4C_ENTS_MAX 128

// One compressor's geometry. A struct rather than seven trailing `int`s: `ratio`, `coff`,
// `d`, `cd`, `ents` and `rd` are all plausible in each other's positions, and `cd`/`ents`
// are DERIVED — `cd = coff*d`, `ents = coff*ratio` — so a caller that computed one from a
// stale `coff` would produce a kernel that reads the right shape with the wrong stride.
// `v4compress.rs::Geom` is the only constructor and derives all three.
struct V4Comp {
    int ratio;
    int coff;   // 1 + overlap (model.py:298)
    int d;      // this compressor's head_dim
    int cd;     // coff * d — the projection width of `wkv`/`wgate`, and `ape`'s row stride
    int ents;   // coff * ratio — entries in one pooling window
    int rd;     // rope_head_dim; only the LAST rd dims of a pooled row are rotated
    float eps;  // norm_eps
};

// What `v4c_finish_row` reads and writes. The three travel together because they must AGREE
// with each other, and nothing in a flat argument list makes them:
//
//   * `freqs` must be the layer's COMPRESSED table — `compress_rope_theta = 160000` with
//     YaRN. The ratio-0 table has the same type, the same stride and the same shape, and
//     substituting it is `Defect::RopeNoYarn`: plausible frequencies at every scale.
//   * `norm` is `[d]`, the compressor's own `RMSNorm` weight — not the layer's `kv_norm`,
//     which is also `[d]` on the attention path.
//   * `out` is `[nblk, d]` at prefill and one row at decode.
//
// Mirrored by `v4compress.rs::Finish`; the two are ONE `repr(C)` layout and must be edited
// together. Passed by pointer across the wall for the same reason `V4Comp` is.
struct V4Fin {
    const float* norm;
    const float* freqs;
    float* out;
};

// `out[m,n] = x[m,k] · w[n,k]^T` with `w` in bf16 — the un-quantized `F.linear` path.
//
// `Compressor.wkv`/`wgate` are `Linear(..., dtype=torch.float32)` (model.py:302-303), which
// in `model.py::linear` means NO `act_quant` on the activation and no weight scales: a
// plain dot in f32 against weights the checkpoint stores in bf16. That is why this is here
// and not `v4_gemv_fp8` — routing the compressor through the fp8 GEMV would quantize an
// activation the reference leaves alone, and the error would look like a pooling bug.
//
// bf16 weights on device, not f32: the values are bf16 on disk, so `bf16f` reproduces them
// exactly while halving the residency. At ratio 4 one layer's pair is [1024,4096] x2 =
// 16 MB bf16 against 32 MB f32, and there are 42 of them.
//
// One wave per output element; lanes stride the reduction dim, so the loads are coalesced.
// `wave_sum`'s fixed shuffle ladder re-associates against the oracle's sequential dot.
extern "C" __global__ void v4_dense_gemm_bf16(const float* __restrict__ x,
                                              const unsigned short* __restrict__ w,
                                              float* __restrict__ out, int m, int n, int k) {
    int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) / WAVE);
    int lane = (int)(threadIdx.x % WAVE);
    // `size_t`, matching the launcher's `(size_t)m * n`. In `int` the product overflows at
    // ~2.1M output elements, `gw >= <negative>` is true for every wave, and the kernel
    // writes NOTHING while `hipGetLastError` reports success — an unwritten scratch handed
    // to the pooling. Unreachable at any shipped `max_seq_len`; the cast costs nothing.
    if ((size_t)gw >= (size_t)m * n) return;
    int r = gw / n, c = gw % n;
    const float* xr = x + (size_t)r * k;
    const unsigned short* wr = w + (size_t)c * k;
    float acc = 0.0f;
    for (int i = lane; i < k; i += WAVE) acc += xr[i] * bf16f(wr[i]);
    acc = wave_sum(acc);
    if (lane == 0) out[gw] = acc;
}

// Softmax over the ENTRY axis, independently per feature, then the weighted sum —
// model.py:346 `(kv * score.softmax(dim=2)).sum(dim=2)`.
//
// One thread owns a whole feature, so the reduction over entries is SEQUENTIAL and in the
// oracle's own order. That is not an accident of the mapping: the softmax is over 8 or 128
// entries and the feature axis is 512 wide, so parallelising the short axis would both
// re-associate the sum and waste the wide one.
//
// `scp[j] == nullptr` is a MASKED entry — block 0's overlap half, which `overlap_transform`
// fills with `value` / `-inf` (model.py:313-318). The oracle spells it as an actual -inf in
// the score buffer and lets `exp(-inf - m)` produce 0; skipping is identical, since adding
// a true 0.0f leaves an f32 accumulator unchanged.
//
// `apep == nullptr` — the WHOLE table, not per entry, because the choice is call-uniform —
// means the position embedding is ALREADY in the score. Prefill adds it here, since at
// ratio 4 one source row is shared between two output blocks and pre-adding it into `score`
// would double-add for one of them; decode adds it once at the state deposit, model.py:349.
__device__ __forceinline__ void v4c_pool_row(const float* const* kvp, const float* const* scp,
                                             const float* const* apep, int ents, int d,
                                             float* __restrict__ outrow) {
    for (int e = threadIdx.x; e < d; e += blockDim.x) {
        float mx = -INFINITY;
        for (int j = 0; j < ents; ++j) {
            if (!scp[j]) continue;
            mx = fmaxf(mx, scp[j][e] + (apep ? apep[j][e] : 0.0f));
        }
        // Every entry masked. `exp(-inf - -inf)` is NaN in the reference; the oracle's
        // `softmax_strided` returns a zero row instead and this matches it. Unreachable
        // through either launcher today (prefill block 0 always has its normal half, and a
        // decode pool only runs on a position that filled the window), so it is a guard
        // against a future caller, not a live path.
        if (!isfinite(mx)) {
            outrow[e] = 0.0f;
            continue;
        }
        float sum = 0.0f;
        for (int j = 0; j < ents; ++j) {
            if (!scp[j]) continue;
            sum += expf(scp[j][e] + (apep ? apep[j][e] : 0.0f) - mx);
        }
        // The exponential is recomputed rather than the sum folded in. Hoisting the divide
        // — `(sum_j k_j·e_j)/S` instead of `sum_j k_j·(e_j/S)` — is a different rounding,
        // and this one is the oracle's. `expf` is deterministic, so the recompute is free
        // of correctness risk and costs one transcendental per element.
        float acc = 0.0f;
        for (int j = 0; j < ents; ++j) {
            if (!scp[j]) continue;
            acc += kvp[j][e] * (expf(scp[j][e] + (apep ? apep[j][e] : 0.0f) - mx) / sum);
        }
        outrow[e] = acc;
    }
}

// model.py:368-373 on one pooled row: the bf16 store, `self.norm`, then RoPE on the last
// `rd` dims at the block's FIRST absolute position.
//
// THREE bf16 roundings, and each one is a store the reference performs:
//   1. `kv.to(dtype)` before the norm — the pooling ran in f32, the norm takes bf16 in;
//   2. `RMSNorm.forward`'s `return (self.weight * x).to(dtype)` (model.py:202);
//   3. `apply_rotary_emb`'s single `y.copy_(x)` (model.py:249).
// Three, not four: an earlier version of this comment counted the `copy_` twice, and a
// reader auditing for the missing fourth would have inserted one and changed the numerics.
// The oracle performs three (`round_bf16` at forward.rs:1071, inside `rmsnorm` at :629, and
// at the tail of `rope_row` at :690). Dropping any leaves values that are fluent and
// slightly too precise, which no magnitude check sees — `Defect::NoBf16Rounding`.
//
// The RoPE pairing is ADJACENT (`2i`, `2i+1`), not rivoli's half-split — `view_as_complex`
// on the last axis. S1b flagged this as the port's easiest silent inversion.
//
// This is `mla.hip::v4_rope`'s rotation with a different position rule, which is the whole
// reason it is not a call to it: `v4_rope` derives its position as `pos0 + blockIdx.x /
// rows_per_pos`, i.e. one position per row going up by ONE. A compressor's rows are blocks,
// so consecutive output rows are `ratio` positions apart. Calling `v4_rope` per row would
// be one launch per compressed block — 512 launches for a 2048-token prefill at ratio 4.
__device__ __forceinline__ void v4c_finish_row(float* __restrict__ row, V4Fin f, int pos,
                                               V4Comp p, float* red) {
    for (int e = threadIdx.x; e < p.d; e += blockDim.x) row[e] = rbf16(row[e]);
    __syncthreads();

    float part = 0.0f;
    for (int e = threadIdx.x; e < p.d; e += blockDim.x) part += row[e] * row[e];
    // `1/sqrtf`, NOT `rsqrtf`: the fast reciprocal square root is a 1-ulp approximation on
    // gfx1151 and the oracle computes `(var + eps).sqrt().recip()`. The difference is a
    // single factor applied to every dim of the row, so it survives the bf16 rounding that
    // follows and does not average out.
    float rs = 1.0f / sqrtf(block_sum_lds(part, red) / (float)p.d + p.eps);
    for (int e = threadIdx.x; e < p.d; e += blockDim.x) {
        row[e] = rbf16(f.norm[e] * (row[e] * rs));
    }
    __syncthreads();

    const float* t = f.freqs + (size_t)pos * p.rd;
    float* seg = row + (p.d - p.rd);
    for (int i = threadIdx.x; i < p.rd / 2; i += blockDim.x) {
        float c = t[2 * i], s = t[2 * i + 1];
        float a = seg[2 * i], b = seg[2 * i + 1];
        seg[2 * i] = a * c - b * s;
        seg[2 * i + 1] = a * s + b * c;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < p.d; e += blockDim.x) row[e] = rbf16(row[e]);
}

// The state deposit — ONE kernel for both phases, because they are one operation.
//
// model.py:333-340 (prefill) writes the trailing partial window into `kv_state`/
// `score_state`; model.py:350-351 and :360-361 (decode) write that step's single row into
// slot `start_pos % ratio`. Read side by side they differ ONLY in which slot the run starts
// at: a prefill of `s` tokens deposits `s % ratio` rows beginning at slot 0, and a decode
// deposits 1 row beginning at slot `start_pos % ratio`. `slot0` is that start. With `s == 1`
// the prefill arithmetic — `remainder = 1`, `cutoff = 0`, source row `cutoff + j == 0` — is
// already the decode arithmetic, so unifying them removes a branch rather than adding one.
//
// This is not a tidiness argument. The two spellings had a five-pointer argument list in
// common at their call sites, `build.rs`'s duplication gate found it, and the fix it forced
// is the better factoring: `ape` is now indexed by `slot0 + j` in ONE place, and the
// off-by-one that indexes it by the state slot instead (`state_off + slot0 + j`, which
// differs by `ratio` on every overlapping layer) can no longer be made in one phase and not
// the other.
//
// It must be launched on EVERY call, including one that emits nothing: the reference writes
// the state and only THEN returns `None`. At ratio 128 that is every prompt under 128 tokens
// — the port's own 13-token emit included — and 127 of every 128 decode steps.
//
// The two loops write disjoint slots: [0, ratio) and [state_off + slot0, ...) with
// `state_off == ratio` whenever the first loop runs at all. One grid-strided kernel is safe.
extern "C" __global__ void v4_compress_state(const float* __restrict__ kv,
                                             const float* __restrict__ score,
                                             const float* __restrict__ ape,
                                             float* __restrict__ kv_state,
                                             float* __restrict__ score_state, V4Comp p, int s,
                                             int slot0) {
    int remainder = s % p.ratio;
    int cutoff = s - remainder;
    int overlap = p.coff == 2;
    int state_off = overlap ? p.ratio : 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    // model.py:335-336 — the last COMPLETE block also becomes the next one's overlap half.
    // Prefill only, and only when overlapping: `slot0` plays no part, since this run always
    // starts at slot 0.
    if (overlap && cutoff >= p.ratio) {
        size_t n = (size_t)p.ratio * p.cd;
        for (size_t i = tid; i < n; i += stride) {
            int j = (int)(i / p.cd), e = (int)(i % p.cd);
            size_t src = (size_t)(cutoff - p.ratio + j) * p.cd + e;
            kv_state[i] = kv[src];
            score_state[i] = score[src] + ape[i];
        }
    }
    if (remainder > 0) {
        size_t n = (size_t)remainder * p.cd;
        size_t dst0 = (size_t)(state_off + slot0) * p.cd;
        size_t ape0 = (size_t)slot0 * p.cd;
        for (size_t i = tid; i < n; i += stride) {
            int j = (int)(i / p.cd), e = (int)(i % p.cd);
            size_t src = (size_t)(cutoff + j) * p.cd + e;
            kv_state[dst0 + i] = kv[src];
            // `ape[slot0 + j]`, the position's index WITHIN its block. Not `ape[state_off +
            // slot0 + j]`: on an overlapping layer that is `ratio` rows further on, which is
            // past the end of a `[ratio, cd]` table.
            score_state[dst0 + i] = score[src] + ape[ape0 + i];
        }
    }
}

// Prefill pooling: one block per compressed output block, `gridDim.x == cutoff / ratio`.
//
// `overlap_transform` (model.py:312-318) is applied as an INDEX MAP rather than as the
// reference's materialised `[b, s, 2*ratio, d]` tensor. Entry j of output block b reads:
//   j >= ratio            → block b's own row (j - ratio), dims [d, 2d)   "normal half"
//   j <  ratio, b > 0     → block b-1's row j,             dims [0, d)    "overlap half"
//   j <  ratio, b == 0    → masked
// which is `new_tensor[:, :, ratio:] = tensor[..., d:]` and `new_tensor[:, 1:, :ratio] =
// tensor[:, :-1, ..., :d]` read backwards. At ratio 128 `coff == 1`, `ents == ratio`, and
// every entry takes the first arm with `half == 0` — the same code, no branch on the ratio.
extern "C" __global__ void v4_compress_prefill(const float* __restrict__ kv,
                                               const float* __restrict__ score,
                                               const float* __restrict__ ape, V4Fin f,
                                               V4Comp p) {
    __shared__ const float* kvp[V4C_ENTS_MAX];
    __shared__ const float* scp[V4C_ENTS_MAX];
    __shared__ const float* apep[V4C_ENTS_MAX];
    __shared__ float red[256];
    int b = (int)blockIdx.x;
    int overlap = p.coff == 2;

    for (int j = (int)threadIdx.x; j < p.ents; j += (int)blockDim.x) {
        int sblk, src_j, half;
        if (!overlap) {
            sblk = b;
            src_j = j;
            half = 0;
        } else if (j >= p.ratio) {
            sblk = b;
            src_j = j - p.ratio;
            half = p.d;
        } else if (b > 0) {
            sblk = b - 1;
            src_j = j;
            half = 0;
        } else {
            kvp[j] = nullptr;
            scp[j] = nullptr;
            apep[j] = nullptr;
            continue;
        }
        size_t src = (size_t)(sblk * p.ratio + src_j) * p.cd + half;
        kvp[j] = kv + src;
        scp[j] = score + src;
        apep[j] = ape + (size_t)src_j * p.cd + half;
    }
    __syncthreads();

    float* row = f.out + (size_t)b * p.d;
    v4c_pool_row(kvp, scp, apep, p.ents, p.d, row);
    __syncthreads();
    // Prefill's first block is block 0 (model.py:370 `freqs_cis[:cutoff:ratio]`), so the
    // rotation position is the block's own first token.
    v4c_finish_row(row, f, b * p.ratio, p, red);
}

// Decode pooling: the window is complete, so gather it out of the state and emit one block.
// `gridDim.x == 1`, which is what makes `__syncthreads()` a true whole-op barrier and the
// window slide at the end safe against the reads above it.
//
// It takes NO `kv`/`score`/`ape`: this step's row was already deposited by
// `v4_compress_state`, with its `ape` row added there. Launched ONLY when
// `(start_pos + 1) % ratio == 0` — the caller decides, and does not launch otherwise, which
// is why there is no `should` flag here. At ratio 128 that is one launch in 128.
extern "C" __global__ void v4_compress_pool_decode(float* __restrict__ kv_state,
                                                   float* __restrict__ score_state, V4Fin f,
                                                   V4Comp p, int start_pos) {
    __shared__ const float* kvp[V4C_ENTS_MAX];
    __shared__ const float* scp[V4C_ENTS_MAX];
    __shared__ float red[256];
    int overlap = p.coff == 2;

    for (int j = (int)threadIdx.x; j < p.ents; j += (int)blockDim.x) {
        // model.py:353-354 `cat([kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], dim=1)`
        // — the low slots contribute their overlap half, the high slots their normal half.
        // Without overlap `cd == d` and every slot contributes all of itself.
        int half = (overlap && j >= p.ratio) ? p.d : 0;
        kvp[j] = kv_state + (size_t)j * p.cd + half;
        scp[j] = score_state + (size_t)j * p.cd + half;
    }
    __syncthreads();

    // `nullptr` for `ape`: it was added at the state deposit (model.py:349).
    v4c_pool_row(kvp, scp, nullptr, p.ents, p.d, f.out);
    __syncthreads();
    // model.py:372 `freqs_cis[start_pos + 1 - ratio]`. Since this kernel only runs when
    // `start_pos + 1` is a multiple of `ratio`, that equals `(start_pos / ratio) * ratio` —
    // the block's first absolute position, the same rule prefill uses.
    //
    // `start_pos / ratio` ALONE is the block INDEX, not the position, and it is the
    // port-hostile slip here: at ratio 128 the first block completes at `start_pos` 127,
    // where the index is 0 and the position is 0 — they agree — and at the SECOND block
    // (255) they are 1 and 128. A probe that stops at the first block cannot tell them
    // apart, which is why `tests/v4_compress_kernel.rs` steps to the second.
    v4c_finish_row(f.out, f, (start_pos / p.ratio) * p.ratio, p, red);
    __syncthreads();

    // model.py:357-358 — the window slides: this block's normal half becomes the next
    // block's overlap half. Inside the emit path, matching the reference: a step that emits
    // nothing must not rotate the window, and this kernel only runs on steps that emit.
    if (overlap) {
        size_t n = (size_t)p.ratio * p.cd;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
            kv_state[i] = kv_state[n + i];
            score_state[i] = score_state[n + i];
        }
    }
}

// ---------------------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------------------

// Shared guards. Every one of these is a shape that would otherwise read plausible garbage
// rather than crash: a zero `ratio` divides by zero in the slot arithmetic, a `cd` that is
// not `coff*d` walks `ape` off its rows, and `ents > V4C_ENTS_MAX` overruns the LDS tables.
// Taken by POINTER across the `extern "C"` wall and by value only inside this file. A
// 28-byte struct passed by value is an ABI detail — SysV classifies it MEMORY, Rust's
// `#[repr(C)]` agrees today, and neither guarantees the other across a compiler bump. A
// pointer has one representation everywhere, and the deref happens on the host side of the
// wall where both halves are compiled by the same hipcc.
static int v4c_guard(V4Comp p) {
    if (p.ratio <= 1 || p.d <= 0 || p.rd <= 0 || p.coff < 1 || p.coff > 2) return 1001;
    if (p.rd > p.d || (p.rd & 1)) return 1002;
    if (p.cd != p.coff * p.d || p.ents != p.coff * p.ratio) return 1003;
    if (p.ents > V4C_ENTS_MAX) return 1004;
    // `coff` and `ratio` are NOT independent: `Compressor.overlap` is `compress_ratio == 4`
    // and nothing else (model.py:296). Without this, `{ratio: 128, coff: 2}` passes every
    // check above and runs the overlapping gather at ratio 128 — finite, plausible, wrong.
    // Unreachable through `Geom::new`, which derives `coff` from `LayerKind`; reachable
    // through the raw ABI, which is what this wall is for.
    //
    // `ratio <= 1` above rather than `<= 0`: at ratio 1 the state deposit computes
    // `remainder = 1 % 1 = 0` and writes NOTHING, and every other guard passes, so the pool
    // reads a slot this step never filled.
    if ((p.coff == 2) != (p.ratio == 4)) return 1010;
    return 0;
}

extern "C" int rivoli_v4_dense_gemm_bf16(const float* x, const unsigned short* w, float* out,
                                         int m, int n, int k, void* stream) {
    if (m <= 0 || n <= 0 || k <= 0) return 1001;
    hipGetLastError();
    size_t waves = (size_t)m * n;
    size_t threads = waves * WAVE;
    hipLaunchKernelGGL(v4_dense_gemm_bf16, dim3((unsigned)((threads + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, x, w, out, m, n, k);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_v4_compress_state(const float* kv, const float* score, const float* ape,
                                        float* kv_state, float* score_state, const V4Comp* pp,
                                        int s, int slot0, void* stream) {
    if (!pp) return 1000;
    V4Comp p = *pp;
    int g = v4c_guard(p);
    if (g) return g;
    if (s <= 0) return 1005;
    if (slot0 < 0 || slot0 + s % p.ratio > p.ratio) return 1008;
    hipGetLastError();
    // Grid-strided, so any positive grid is correct; this one covers the larger loop in a
    // single pass.
    int rem = s % p.ratio;
    size_t n = (size_t)(p.ratio > rem ? p.ratio : rem) * p.cd;
    hipLaunchKernelGGL(v4_compress_state, dim3((unsigned)((n + 255) / 256)), dim3(256), 0,
                       (hipStream_t)stream, kv, score, ape, kv_state, score_state, p, s, slot0);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_v4_compress_prefill(const float* kv, const float* score, const float* ape,
                                          const V4Fin* ff, const V4Comp* pp, int nblk,
                                          void* stream) {
    if (!pp || !ff) return 1000;
    V4Comp p = *pp;
    int g = v4c_guard(p);
    if (g) return g;
    // NOT a no-op success: `nblk == 0` means the caller believed a block would be emitted
    // when none can be, and silently launching nothing hands it an unwritten `out`.
    // `v4compress.rs::compress` decides emission from `should_compress` and does not call
    // this when it is false.
    if (nblk <= 0) return 1006;
    hipGetLastError();
    hipLaunchKernelGGL(v4_compress_prefill, dim3((unsigned)nblk), dim3(256), 0,
                       (hipStream_t)stream, kv, score, ape, *ff, p);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

extern "C" int rivoli_v4_compress_pool_decode(float* kv_state, float* score_state,
                                              const V4Fin* ff, const V4Comp* pp, int start_pos,
                                              void* stream) {
    if (!pp || !ff) return 1000;
    V4Comp p = *pp;
    int g = v4c_guard(p);
    if (g) return g;
    if (start_pos <= 0) return 1007;  // start_pos == 0 is the prefill path, not this one
    // The caller must only reach here on a position that COMPLETES a block. Refusing rather
    // than pooling a partial window: a half-filled window still produces finite, plausible
    // numbers, and the reference would have emitted nothing at all.
    if ((start_pos + 1) % p.ratio != 0) return 1009;
    hipGetLastError();
    // ONE block, always. `v4_compress_pool_decode` uses `__syncthreads()` to order the pool
    // against the window slide; with a wider grid that barrier would order nothing and the
    // slide would race the reads.
    hipLaunchKernelGGL(v4_compress_pool_decode, dim3(1), dim3(256), 0, (hipStream_t)stream,
                       kv_state, score_state, *ff, p, start_pos);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
