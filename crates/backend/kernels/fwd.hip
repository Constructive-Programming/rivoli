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

// x[i] *= sigmoid(g[i])  — Muse Glimmer's attention output gate, S2 item 3 of
// docs/investigations/glimmer-port.md. `x` is the attend output flattened to
// [rows][n_heads*head_dim] and `g` is `gate_proj(h)` at the same width; the product runs
// BEFORE `o_proj` (glimmer-architecture.md §4 item 3, M:369).
//
// **`g` IS NOT DERIVED FROM `x`, and that is the whole trap.** `gate_proj` consumes the LAYER
// INPUT — the post-`input_layernorm` activation — not the attention output. A port that gates on
// `x` has the right shapes, the right tensor and the wrong model (§9 trap 4). Nothing about this
// kernel can enforce that; it takes two pointers and multiplies. What holds the caller to it is
// `tests/glimmer_gate.rs`, which feeds the golden's own `attn.gate_proj.out` and compares against
// `attn.o_proj.in_gated` — two captures the anchor deliberately keeps apart for this reason.
//
// A separate launch rather than folded into `gqa_attend`'s output write. Folding saves one pass
// over [rows][4096] and is an optimisation with no measurement behind it; it would also put a
// second operand inside a kernel currently gated as pure attention, so its fixture would stop
// meaning "the attention is right". S5 prices the fusion if it is worth anything.
//
// Not `swiglu`: that is `silu(g)*u` = `g*sigmoid(g)*u`, which carries an extra factor of `g`.
// The two are one `* g` apart and neither is spellable as the other.
// A `g` of ±Inf yields the LIMIT (gate exactly 1 or exactly 0) — finite output from a
// non-finite input, so an overflowed `gate_proj` is invisible to `flag_nonfinite` on the
// residual. Reviewed 2026-08-12 and left as-is: the limits are what sigmoid converges to, a
// NaN `g` still propagates, and the softcap keeps the terminal detector. Recorded so the
// choice is a decision and not an accident.
//
// USED BY Kimi-K3 too (M9): its gated MLA applies exactly this — `attn *= sigmoid(gate)`,
// division form, between `attn.hip::mha_attend` and `o_proj` (`k3-architecture.md` §5). The
// k3 tree carried its own out-of-place copy (`k3:kernels/attn.hip`); the M9 port deliberately
// did NOT bring it, because a second entry point differing only in writing to a third buffer
// is the `swiglu`/`swiglu_clamped_bf16` merge hazard in reverse. What K3's MLA must NOT use
// is `recurrent.hip::rmsnorm_gate_heads_f32` — MLA gates with NO norm, KDA norms then gates,
// and that ordering is `k3-architecture.md` trap 10.
extern "C" __global__ void sigmoid_gate(
    float* __restrict__ x, const float* __restrict__ g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // `1/(1+exp(-v))` rather than a `tanh`-based form: it is what `torch.sigmoid` computes, and
    // the reference applies it in fp32 exactly here. **`expf`, not `__expf`** — the fast
    // intrinsic is a different function with its own error, and taking it here would price the
    // gate's tolerance against an approximation nobody measured. It is an S5 option, not a
    // correctness-stage default.
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}

// x[i] = cap * tanh(x[i] * mult / cap)  — Muse Glimmer's logit path, S2 item 5 of
// docs/investigations/glimmer-port.md. `mult` is `output_multiplier` and `cap` is
// `final_logit_softcapping`, both from the config and both PARAMETERS: a wrong multiplier is
// fluent, and hard-coding either would put this model's constants in a kernel the next model
// would reuse (glimmer-architecture.md §5).
//
// **THIS IS THE OPERATION EVERY GREEDY GATE IN THIS REPO IS PROVABLY BLIND TO.** `mult` > 0 and
// `tanh` is strictly increasing, so the composition cannot change which token an argmax picks:
// greedy equality, teacher-forced argmax and byte-identical output all pass with this kernel
// omitted entirely. It changes every PROBABILITY — NLL, perplexity, sampling, any confidence
// gate. The anchor measured exactly that: `softcap_off` moves 7 of 1103 captures and leaves
// `emitted.ids` identical (§9 trap 12, `glimmer-reference/anchor.md`). So this kernel's evidence
// has to come from the logits themselves, never from what was decoded.
//
// A separate launch from the head GEMV, for the reason the gate above is separate: folding it in
// would make the head's fixture stop meaning "the projection is right", and the fusion is worth
// one pass over 202048 floats that nobody has priced.
extern "C" __global__ void logit_softcap(
    float* __restrict__ x, int n, float mult, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        // Non-finite passes THROUGH, deliberately (review, 2026-08-12). `tanhf(±Inf)` is ±1, so
        // the plain form maps an overflowed logit to exactly ±cap — FINITE — and the engine's
        // only detector for a fault after the last layer is `argmax` returning a non-finite top
        // value (gpu.rs's bail). This is the last kernel before that check: laundering ±Inf here
        // deletes the detector, and the emitted token is whichever vocab entry overflowed first.
        // `tanhf`, not a `1 - 2/(exp(2z)+1)` rewrite: the reference calls `torch.tanh`, and a
        // rewrite would be a second approximation to price against a tolerance measured on the first.
        x[i] = isfinite(v) ? cap * tanhf(v * mult / cap) : v;
    }
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

// `stream` is trailing and nullptr is the null stream, the `rivoli_swiglu` contract. Not
// optional for this kernel (review, 2026-08-12): it sits between `gqa_attend` and the o_proj
// GEMV, both stream-parameterised, and rivoli's streams are `hipStreamNonBlocking` — a
// null-stream product between two stream-ordered launches reads its operands before they are
// written, a silent wrong no S2 fixture can see (each launches one kernel and joins).
extern "C" int rivoli_sigmoid_gate(float* x, const float* g, int n, void* stream) {
    if (n <= 0) return 1001;
    hipGetLastError();
    hipLaunchKernelGGL(sigmoid_gate, dim3((n + 255) / 256), dim3(256), 0, (hipStream_t)stream, x,
                       g, n);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}

// `stream` is trailing and nullptr is the null stream, the `rivoli_swiglu` contract — this sits
// between the head GEMV and `argmax`, and the head tail moving onto a stream is exactly the
// moment `f4gpu.rs`'s note says a null-stream member becomes an unordered read.
extern "C" int rivoli_logit_softcap(float* x, int n, float mult, float cap, void* stream) {
    if (n <= 0) return 1001;
    // `cap` divides, and `mult <= 0` would break the monotonicity the whole argmax-invariance
    // argument rests on — both are config values, so a zero here means the config was not read.
    // `!(x > 0)` rather than `x <= 0` is what rejects NaN; the finiteness check is what rejects
    // +Inf, which `!(x > 0)` admits (a +Inf cap NaNs every logit, a +Inf mult sign-quantises
    // them). `__builtin_isfinite` because this is HOST code in a .hip TU, where HIP's cmath
    // shim drops the float overload of `isfinite` (substitution failure, measured llvm-22).
    if (!(cap > 0.0f) || !(mult > 0.0f) || !__builtin_isfinite(cap) || !__builtin_isfinite(mult))
        return 1001;
    // The two scalars are adjacent, same-typed and TRANSPOSABLE, and swapping them is silent —
    // `0.196*tanh(102·x)` is a hard sign-quantiser every greedy gate still passes (this repo has
    // shipped a transposed-argument pair before: model.rs:2472's review record). For any model
    // with a softcap the pre-multiplier is a 1/sqrt(d)-scale value and the cap is O(10), so
    // order is checkable: refuse `mult >= cap` as its own code.
    if (mult >= cap) return 1002;
    hipGetLastError();
    hipLaunchKernelGGL(logit_softcap, dim3((n + 255) / 256), dim3(256), 0, (hipStream_t)stream, x,
                       n, mult, cap);
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

// XOR-fold the EXACT BITS of `x[0..n]` into `*out` — the divergence-localisation probe
// (`--divergence-log`). Two runs of one input must agree slot for slot; the first slot that
// differs names the (position, layer, quantity) where a run diverged, which is the question
// GLM's run-to-run divergence cannot answer from output text alone (a single changed token
// rewrites the whole tail, so the output says only "somewhere before here").
//
// XOR because it is commutative AND associative, so the fold is bit-identical whatever order
// the atomics land in. A float sum would be neither and would report divergence from
// scheduling jitter alone — the probe has to be quieter than the bug it hunts. This is the
// load-bearing property of the whole instrument, and it is the reason a `+` here would be a
// defect rather than a style choice.
//
// The INDEX is mixed in before the finalizer for two reasons: a permutation of the same
// values is then a different hash, and — because XOR is self-inverse — two elements that
// happen to hold the same bit pattern no longer cancel each other out of the fold.
// splitmix64's finalizer spreads a one-bit input change across the whole word, so two
// nearby single-ulp differences cannot cancel either.
//
// Stays on the DEVICE and adds NO sync, for a measured reason: the old tree's `--checksum-x`
// copied the residual to the host every layer, and a config that reproduced its fault
// without the probe produced a clean run with it (see `flag_nonfinite` above). An
// instrument that perturbs its own subject is not an instrument. The caller drains every
// slot in ONE D2H per token, at a point the end-of-layer sync has already idled.
extern "C" __global__ void hash_rows(const float* __restrict__ x, int n, int stride,
                                    unsigned long long i_base,
                                    unsigned long long* __restrict__ out) {
    // ONE ATOMIC PER BLOCK, not one per element, and this is a correctness-of-the-instrument fix
    // rather than a tuning choice.
    //
    // The first version did `atomicXor(out, …)` per element against a SINGLE global u64. That is
    // fine for the residual folds (6144 elements) but the fetch-path folds are 3.8 M elements each
    // and there are ~936 of them per token: 3.6e9 same-address atomics, which serialise. At any
    // plausible same-address atomic rate that is ~1-4 s/token against a 388 ms budget — the
    // instrument would be several times slower than what it measures, which is the `--checksum-x`
    // failure with a different mechanism. Reducing in-block first (3744x fewer atomics) makes it
    // bandwidth-bound instead.
    //
    // The reduction is EXACT and needs no tolerance argument: XOR is associative and commutative,
    // so a tree in shared memory yields the identical word a serial fold does. That is the property
    // the whole instrument rests on, used one level deeper. `rivoli_core::hash::xor_fold_from`
    // remains the reference and `fwd_kernel.rs` scores this against it bit for bit.
    //
    // `i_base` OFFSETS THE INDEX SPACE, which is how several slots fold into one accumulator while
    // staying sensitive to WHICH slot each payload came from. Folding N slots at `i_base = j*n`
    // treats them as one virtual array. Without it the fold mixes only the within-slot index, so it
    // is invariant under SWAPPING two experts' payloads between their slots — and a crossed
    // destination is exactly the "wrong bytes in a slot" class under investigation, which would
    // then leave every fold agreeing and be misread as something else (review, 2026-08-17).
    // `stride` SAMPLES the buffer: element 0, `stride`, `2*stride`, … < `n`. At `stride == 1`
    // this is the whole buffer and every earlier caller is unchanged. It exists so one arm can
    // touch every CACHE LINE of a slot while reading ~1/32 of its bytes — a sweep that covers the
    // slot can be a candidate FIX, where reading one line at the front can only ever repair one
    // line. The mixed-in index is the ELEMENT index, not the sample number, so folds at different
    // strides remain sensitive to where a difference is.
    __shared__ unsigned long long red[256];
    unsigned long long acc = 0;
    long long samples = stride > 0 ? (n + stride - 1) / stride : 0;
    // Grid-stride over SAMPLES, so the launch geometry is bounded and independent of `n`.
    for (long long s_i = blockIdx.x * blockDim.x + threadIdx.x; s_i < samples;
         s_i += (long long)blockDim.x * gridDim.x) {
        long long i = s_i * stride;
        unsigned int bits;
        __builtin_memcpy(&bits, &x[i], sizeof(bits));
        unsigned long long z = ((i_base + (unsigned long long)i) << 32) ^ (unsigned long long)bits;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        acc ^= z ^ (z >> 31);
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] ^= red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicXor(out, red[0]);
}

extern "C" int rivoli_hash_rows(const float* x, int n, int stride, unsigned long long i_base,
                                unsigned long long* out, void* stream) {
    if (n <= 0 || stride <= 0) return 1001;
    hipGetLastError();
    // Grid CAPPED at 1024 blocks: the kernel is grid-stride, so a bounded grid bounds the number
    // of atomics (one per block) independently of `n`. 1024 is ~25x the 40 CUs, enough to saturate
    // and few enough that the atomics are noise. Sized on SAMPLES, so a strided sweep launches the
    // blocks its sample count needs rather than the buffer's length.
    int samples = (n + stride - 1) / stride;
    int blocks = (samples + 255) / 256;
    if (blocks > 1024) blocks = 1024;
    hipLaunchKernelGGL(hash_rows, dim3(blocks), dim3(256), 0, (hipStream_t)stream, x, n, stride,
                       i_base, out);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
