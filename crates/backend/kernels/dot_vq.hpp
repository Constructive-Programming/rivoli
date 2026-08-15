// The VQ-int3 CODEBOOK dot — 12-bit indices into a shared fp16 codebook, one bf16 scale
// per VQ_GROUP weights. Split out of common.hpp 2026-08-15 with the other three dot
// families; the body and the measurements in its comments travelled verbatim, and the
// signature was bundled into the views in `rowview.hpp` the same day (that file carries
// the aliasing argument for every pointer below).
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "formats.hpp"
#include "reduce.hpp"
#include "rowview.hpp"

// VQ-int3 codebook parameters — MUST match quant.rs (VQ_DIM/VQ_K/VQ_INDEX_BITS/VQ_GROUP).
#define VQ_DIM 4
#define VQ_K 4096
#define VQ_INDEX_BITS 12
#define VQ_GROUP 64
#define VQ_SUBS_PER_GROUP (VQ_GROUP / VQ_DIM)  // subvectors sharing one bf16 scale

// One VQ-encoded weight row and the codebook it indexes. Matches quant.rs::matvec_vq:
// each VQ_DIM-subvector t reads a packed 12-bit index out of `idx`, dots x[t*VQ_DIM..]
// with `cb[index*VQ_DIM..]`, and scales by the subvector's bf16 group scale
// (group = t / VQ_SUBS_PER_GROUP). Lane `l` owns subvectors t ≡ l (mod WAVE). The
// two-byte index read is in-bounds because i_dim is a multiple of 8 (nsub even; the last
// odd subvector's high byte is the row's last index byte — see quant.rs::get_idx /
// vq_row_bytes).
//
// The codebook is fp16: the random idx→cb[idx] gather is the latency-bound step,
// and at 8B/entry (VQ_K·VQ_DIM·2 = 32KB) the whole codebook fits in the 32KB L1,
// so the gather is an L1 hit instead of L2 (f32 was 64KB, L2-resident). fp16 keeps
// x in f32 for the products; its 10-bit mantissa on the centroids clears the
// oracle tol (bf16's 8 does not — see math::f32_to_f16). `cb` is shared across all rows,
// which is why it rides the row view rather than being a separate argument: every caller
// that has one has the other.
struct VqRow {
    const unsigned char* __restrict__ idx;
    const unsigned short* __restrict__ scale;
    const __half* __restrict__ cb;
    int i_dim;
};

// R INPUT ROWS against ONE weight row. `out.v[r]` (lane 0) is the dot of row `r` of the
// activations — stride `x.stride` — with the decoded weight row.
//
// This is where a speculative verify pass gets cheap. The weight side is the expensive
// half by a wide margin: an expert block is 15.34 MB and this kernel touches every byte
// of it, while a row of `x` is 24 KB. Decoding the row once and dotting it against R
// inputs turns R tokens into ONE read of the weights, so the pass that VERIFIES a draft
// costs barely more than the pass that would have produced a single token — which is the
// entire economic case for MTP on a fetch-bound engine.
//
// R is a template parameter so `acc` stays in registers and the inner loop unrolls; a
// runtime count would index an array dynamically and spill it to scratch.
//
// R=1 is BIT-IDENTICAL to the single-row form this replaced — the same four products in
// the same order, the same one `bf16f` scale multiply, the same `wave_sum`. That is load
// bearing: it means every existing oracle test and the perplexity gate still bind the
// batched kernel's R=1 path with no re-baselining.
template <int R>
__device__ __forceinline__ Acc<R> dot_vq_wave_r(RowsView x, VqRow w, int lane) {
    int nsub = w.i_dim / VQ_DIM;
    Acc<R> acc;
#pragma unroll
    for (int r = 0; r < R; ++r) acc.v[r] = 0.0f;
    for (int t = lane; t < nsub; t += WAVE) {
        int bitpos = t * VQ_INDEX_BITS;
        int byte = bitpos >> 3;
        int shift = bitpos & 7;
        unsigned int raw = (unsigned int)w.idx[byte] | ((unsigned int)w.idx[byte + 1] << 8);
        int idx = (int)((raw >> shift) & 0xFFFu);
        // 8B fp16 gather (two __half2) for the VQ_DIM=4 subvector; products in f32,
        // same 4 terms in the same order as the f32 path.
        const __half* c = w.cb + (size_t)idx * VQ_DIM;
        float2 c01 = __half22float2(*(const __half2*)c);
        float2 c23 = __half22float2(*(const __half2*)(c + 2));
        float s = bf16f(w.scale[t / VQ_SUBS_PER_GROUP]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float4 xv = *(const float4*)(x.x + (size_t)r * x.stride + (size_t)t * VQ_DIM);
            float dot = xv.x * c01.x + xv.y * c01.y + xv.z * c23.x + xv.w * c23.y;
            acc.v[r] += s * dot;
        }
    }
    // Uniform across the wave: R is a compile-time constant, so every lane runs every
    // `wave_sum`. A runtime bound here would be a partial-wave reduction.
    Acc<R> out;
#pragma unroll
    for (int r = 0; r < R; ++r) out.v[r] = wave_sum(acc.v[r]);
    return out;
}

// The single-row form, result on lane 0. Still literally `dot_vq_wave_r<1>` with a stride
// nothing reads — a zero stride is what this has always passed, so keeping the wrapper
// costs one `RowsView` construction and keeps the one-row call sites reading as one row.
__device__ __forceinline__ float dot_vq_wave(const float* __restrict__ x, VqRow w, int lane) {
    return dot_vq_wave_r<1>(RowsView{x, 0}, w, lane).v[0];
}
