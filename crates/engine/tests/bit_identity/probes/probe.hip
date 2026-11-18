// Prelude probe kernels, hipcc side. Each kernel exposes ONE common.hpp helper with plain
// arguments so its Rust twin (crates/kernels/src/probes) can be launched with the identical
// argument buffer and the two outputs compared byte for byte.
//
// Built against the archive/hip-kernels tag's headers (-I <tag checkout>/crates/backend/kernels).
// Sweep kernels ("ex_*") derive their input from the global id, so a 2^32 sweep moves no input.
#include "common.hpp"

#define GID (blockIdx.x * blockDim.x + threadIdx.x)
#define BITS (base + GID)
#define SWEEP(n) unsigned base, unsigned n, out_t* out
#define GUARD if (GID >= n) return

// ── scalar codecs: exhaustive over their whole input domain ───────────────────────
typedef float out_t;
extern "C" __global__ void ex_bf16f(SWEEP(n)) { GUARD; out[GID] = bf16f((unsigned short)BITS); }
extern "C" __global__ void ex_rbf16(SWEEP(n)) { GUARD; out[GID] = rbf16(__uint_as_float(BITS)); }
extern "C" __global__ void ex_e4m3f(SWEEP(n)) { GUARD; out[GID] = e4m3f((unsigned char)BITS); }
extern "C" __global__ void ex_e2m1f(SWEEP(n)) { GUARD; out[GID] = e2m1f(BITS); }
extern "C" __global__ void ex_e8m0f(SWEEP(n)) { GUARD; out[GID] = e8m0f((unsigned char)BITS); }
extern "C" __global__ void ex_frs(SWEEP(n), float max_inv) {
    GUARD; out[GID] = fast_round_scale(__uint_as_float(BITS), max_inv);
}
extern "C" __global__ void ex_aq8(SWEEP(n), float s) {
    GUARD; out[GID] = act_quant_roundtrip(__uint_as_float(BITS), s);
}
extern "C" __global__ void ex_aq4(SWEEP(n), float s) {
    GUARD; out[GID] = fp4_quant_roundtrip(__uint_as_float(BITS), s);
}
extern "C" __global__ void ex_nib(SWEEP(n), int k) { GUARD; out[GID] = nib(BITS, k); }
// g = bf16 high half, u = bf16 low half: swiglu_clamped rounds both to bf16 first, so this
// grid IS its whole domain for a given limit.
extern "C" __global__ void ex_swiglu(SWEEP(n), float limit) {
    GUARD; unsigned b = BITS;
    out[GID] = swiglu_clamped(bf16f((unsigned short)(b >> 16)), bf16f((unsigned short)b), limit);
}
extern "C" __global__ void ex_situ_g(SWEEP(n), float u, float b1, float b2) {
    GUARD; out[GID] = situ_glu(__uint_as_float(BITS), u, b1, b2);
}
extern "C" __global__ void ex_situ_u(SWEEP(n), float g, float b1, float b2) {
    GUARD; out[GID] = situ_glu(g, __uint_as_float(BITS), b1, b2);
}
#undef out_t
#define out_t unsigned short
extern "C" __global__ void ex_f2bf16(SWEEP(n)) { GUARD; out[GID] = f2bf16(__uint_as_float(BITS)); }
#undef out_t
#define out_t unsigned char
extern "C" __global__ void ex_f2e4m3(SWEEP(n)) { GUARD; out[GID] = f2e4m3_rne(__uint_as_float(BITS)); }
#undef out_t
#define out_t unsigned int
extern "C" __global__ void ex_f2e2m1(SWEEP(n)) { GUARD; out[GID] = f2e2m1(__uint_as_float(BITS)); }
extern "C" __global__ void ex_blk_shift(SWEEP(n)) { GUARD; out[GID] = (unsigned)blk_shift((int)BITS); }
extern "C" __global__ void ex_nquad(SWEEP(n)) { GUARD; out[GID] = (unsigned)absorb_nquad((int)BITS); }
#undef out_t
#define out_t unsigned long long
extern "C" __global__ void ex_moe_fixed(SWEEP(n)) { GUARD; out[GID] = moe_fixed(__uint_as_float(BITS)); }

// 256 threads: the LUT the fp8 GEMVs build, written straight to global.
extern "C" __global__ void k_lut(float* out) { e4m3_lut_build(out, threadIdx.x); }

// ── reductions ────────────────────────────────────────────────────────────────────
extern "C" __global__ void k_wave_sum(const float* in, float* out) { out[GID] = wave_sum(in[GID]); }
extern "C" __global__ void k_wave_max(const float* in, float* out) { out[GID] = wave_max(in[GID]); }
// Two reductions back to back over one `red` (recurrent.hip's shape), so the trailing barrier
// has a reader. `bd` is unused here: the Rust twin takes the block size as an argument (the
// 0c rule — no helper reads blockDim from the dispatch packet), so both sides share one ABI.
extern "C" __global__ void k_block_sum2(const float* a, const float* b, float* out, unsigned bd) {
    extern __shared__ float red[];
    float s = block_sum_lds(a[GID], red);
    float t = block_sum_lds(b[GID], red);
    out[2 * GID] = s;
    out[2 * GID + 1] = t;
}

// ── dots: one wave per weight row, every lane's result stored ─────────────────────
// Shared geometry: row o = blockIdx.x*8 + wave; out index (o*R + r)*32 + lane.
#define WAVE_ROW int lane = threadIdx.x & 31, o = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5); \
    if (o >= o_dim) return

#define FP8_ARGS const float* x, const unsigned char* w, const float* scale, int o_dim, int i_dim, \
    int block, int sc_cols, float* out
#define FP8_LUT __shared__ float lut[256]; e4m3_lut_build(lut, threadIdx.x); __syncthreads()
#define FP8_ROW Fp8Row{w + (size_t)o * i_dim, scale + (size_t)(o >> blk_shift(block)) * sc_cols, lut, i_dim, block}

extern "C" __global__ void k_fp8_wave(FP8_ARGS) {
    FP8_LUT; WAVE_ROW;
    out[o * WAVE + lane] = dot_fp8_wave(x, FP8_ROW, lane);
}
template <int R> __device__ void fp8_wave_r(FP8_ARGS) {
    FP8_LUT; WAVE_ROW;
    Acc<R> a = dot_fp8_wave_r<R>(RowsView{x, i_dim}, FP8_ROW, lane);
    for (int r = 0; r < R; ++r) out[(o * R + r) * WAVE + lane] = a.v[r];
}
extern "C" __global__ void k_fp8_wave_r1(FP8_ARGS) { fp8_wave_r<1>(x, w, scale, o_dim, i_dim, block, sc_cols, out); }
extern "C" __global__ void k_fp8_wave_r2(FP8_ARGS) { fp8_wave_r<2>(x, w, scale, o_dim, i_dim, block, sc_cols, out); }
// split-K form: one BLOCK per row, all 256 threads stride, every thread's partial stored.
#define SPLITK_ROW int o = blockIdx.x; LaneSpan sp{(int)threadIdx.x, ROWS_PER_BLOCK * WAVE}
extern "C" __global__ void k_fp8_splitk(FP8_ARGS) {
    FP8_LUT; SPLITK_ROW;
    out[o * 256 + threadIdx.x] = fp8_dot_strided(x, FP8_ROW, sp);
}
template <int R> __device__ void fp8_splitk_r(FP8_ARGS) {
    FP8_LUT; SPLITK_ROW;
    Acc<R> a = fp8_dot_strided_r<R>(RowsView{x, i_dim}, FP8_ROW, sp);
    for (int r = 0; r < R; ++r) out[(o * R + r) * 256 + threadIdx.x] = a.v[r];
}
extern "C" __global__ void k_fp8_splitk_r1(FP8_ARGS) { fp8_splitk_r<1>(x, w, scale, o_dim, i_dim, block, sc_cols, out); }
extern "C" __global__ void k_fp8_splitk_r2(FP8_ARGS) { fp8_splitk_r<2>(x, w, scale, o_dim, i_dim, block, sc_cols, out); }

// int4: w row stride rb = (dim+1)/2 bytes, scale row stride i4 groups.
#define I4_ARGS const float* v, const unsigned char* w, const float* scale, int o_dim, int dim, float* out
#define I4_ROW I4Row{w + (size_t)o * ((dim + 1) / 2), scale + (size_t)o * ((dim + I4_GROUP - 1) / I4_GROUP), dim}
extern "C" __global__ void k_i4_wave(I4_ARGS) { WAVE_ROW; out[o * WAVE + lane] = dot_i4_wave(v, I4_ROW, lane); }
template <int R> __device__ void i4_wave_r(I4_ARGS) {
    WAVE_ROW;
    Acc<R> a = dot_i4_wave_r<R>(RowsView{v, dim}, I4_ROW, lane);
    for (int r = 0; r < R; ++r) out[(o * R + r) * WAVE + lane] = a.v[r];
}
extern "C" __global__ void k_i4_wave_r1(I4_ARGS) { i4_wave_r<1>(v, w, scale, o_dim, dim, out); }
extern "C" __global__ void k_i4_wave_r2(I4_ARGS) { i4_wave_r<2>(v, w, scale, o_dim, dim, out); }

// fp4: w row stride dim/2 bytes (dim even), e8m0 scale row stride ceil(dim/32).
#define F4_ARGS const float* v, const unsigned char* w, const unsigned char* scale, int o_dim, int dim, float* out
template <int R> __device__ void f4_wave_r(F4_ARGS) {
    WAVE_ROW;
    F4Row row{(gu8p)(w + (size_t)o * (dim / 2)), (gu8p)(scale + (size_t)o * ((dim + F4_GROUP - 1) / F4_GROUP)), dim};
    Acc<R> a = dot_f4_wave_r<R>(RowsView{v, dim}, row, lane);
    for (int r = 0; r < R; ++r) out[(o * R + r) * WAVE + lane] = a.v[r];
}
extern "C" __global__ void k_f4_wave_r1(F4_ARGS) { f4_wave_r<1>(v, w, scale, o_dim, dim, out); }
extern "C" __global__ void k_f4_wave_r2(F4_ARGS) { f4_wave_r<2>(v, w, scale, o_dim, dim, out); }

// VQ: idx row stride i_dim/4*12/8 bytes, bf16 scale row stride i_dim/64, shared fp16 codebook.
#define VQ_ARGS const float* x, const unsigned char* idx, const unsigned short* scale, const __half* cb, \
    int o_dim, int i_dim, float* out
#define VQ_ROW VqRow{idx + (size_t)o * (i_dim / VQ_DIM * VQ_INDEX_BITS / 8), scale + (size_t)o * ((i_dim / VQ_DIM + VQ_SUBS_PER_GROUP - 1) / VQ_SUBS_PER_GROUP), cb, i_dim}
extern "C" __global__ void k_vq_wave(VQ_ARGS) { WAVE_ROW; out[o * WAVE + lane] = dot_vq_wave(x, VQ_ROW, lane); }
template <int R> __device__ void vq_wave_r(VQ_ARGS) {
    WAVE_ROW;
    Acc<R> a = dot_vq_wave_r<R>(RowsView{x, i_dim}, VQ_ROW, lane);
    for (int r = 0; r < R; ++r) out[(o * R + r) * WAVE + lane] = a.v[r];
}
extern "C" __global__ void k_vq_wave_r1(VQ_ARGS) { vq_wave_r<1>(x, idx, scale, cb, o_dim, i_dim, out); }
extern "C" __global__ void k_vq_wave_r2(VQ_ARGS) { vq_wave_r<2>(x, idx, scale, cb, o_dim, i_dim, out); }

// ── the gate's device-side compare (not probes) ──────────────────────────────────────
// Built here, by the REFERENCE toolchain, rather than in the Rust object under test: the
// exhaustive sweeps compare ~1.5 TB, which only a device compare can afford, and the thing
// doing the comparing must not be the thing being compared. Ported from 0b's harness.cpp.
// counts[0] = mismatching bytes; counts[1] = lowest mismatching byte index (init ~0).
extern "C" __global__ void cmp_bytes(const unsigned char* a, const unsigned char* b, size_t n,
                                     unsigned long long* counts) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) {
        atomicAdd(&counts[0], 1ull);
        atomicMin(&counts[1], (unsigned long long)i);
    }
}
// For f32 outputs: of the mismatching WORDS, counts[0] = NaN on both sides (a payload
// difference, equal under the owner's NaN-as-a-class rule), counts[1] = value differences.
extern "C" __global__ void cmp_nan_words(const unsigned* a, const unsigned* b, size_t n,
                                         unsigned long long* counts) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || a[i] == b[i]) return;
    bool na = (a[i] & 0x7fffffffu) > 0x7f800000u, nb = (b[i] & 0x7fffffffu) > 0x7f800000u;
    atomicAdd(&counts[na && nb ? 0 : 1], 1ull);
}
