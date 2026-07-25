//! Weight decode + the VQ-int3 quantizer. CPU oracles the HIP kernels are
//! validated against, plus the offline `quant_vq` encode. The formats: VQ-int3 and
//! colibri int4 (routed experts), fp8-e4m3 block (attention/dense), int8 per-row
//! (embed/lm_head), bf16 (indexer), f32 (norms/gate). Functions take raw byte
//! slices + dims — no snapshot-struct coupling.

use crate::math::{bf16_to_f32, e4m3_to_f32, f32_to_bf16};

/// Read an F32 tensor's raw little-endian bytes into a `Vec<f32>`. For O-length
/// tensors only (norm weights, per-projection codebooks) — loaded once at startup.
pub fn read_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0, "F32 tensor length not a multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One little-endian f32 scale at row `o` of the raw per-row scale bytes.
#[inline]
fn scale_at(scale: &[u8], o: usize) -> f32 {
    let b = &scale[o * 4..o * 4 + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// GEMV against a per-row **int8** matrix `W[o_dim, i_dim]` (one signed byte per
/// weight + per-row f32 scale) — the lm_head projection to logits.
pub fn matvec_i8(y: &mut [f32], x: &[f32], packed: &[u8], scale: &[u8], i_dim: usize) {
    debug_assert_eq!(x.len(), i_dim);
    for (o, (yo, row)) in y.iter_mut().zip(packed.chunks_exact(i_dim)).enumerate() {
        let mut acc = 0.0f32;
        for (&b, &xi) in row.iter().zip(x) {
            acc += (b as i8) as f32 * xi;
        }
        *yo = acc * scale_at(scale, o);
    }
}

/// GEMV against an **fp8-e4m3** block-scaled matrix `W[o_dim, i_dim]` — the CPU
/// oracle the HIP `gemv_fp8` kernel (attention/dense projections) is validated
/// against. `scale` is the F32 `weight_scale_inv`, one value per `block × block`
/// tile: `w[o,i] = e4m3(packed[o·i_dim+i]) · scale[(o/block)·sc_cols + i/block]`.
pub fn matvec_fp8(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scale: &[f32],
    i_dim: usize,
    block: usize,
) {
    debug_assert_eq!(x.len(), i_dim);
    let sc_cols = i_dim.div_ceil(block);
    for (o, (yo, row)) in y.iter_mut().zip(packed.chunks_exact(i_dim)).enumerate() {
        let mut acc = 0.0f32;
        for (i, (&b, &xi)) in row.iter().zip(x).enumerate() {
            acc += e4m3_to_f32(b) * scale[(o / block) * sc_cols + i / block] * xi;
        }
        *yo = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector-quantized int3 experts (M1 scalar oracle — see docs/int3.md).
//
// We requantize the fp8 checkpoint's experts into a uniform container the streaming
// kernel reads. Each row's weights are split into `VQ_DIM`-wide subvectors, and each
// subvector is quantized to the nearest entry of a learned `VQ_K`-entry codebook
// (vector quantization — the E8/QuIP# idea, but a data-learned codebook, which on
// GLM-5.2 weights ties per-row int4 where a fixed E8 lattice fell short). A bf16
// scale per `VQ_GROUP` weights along the input dim carries the outer magnitude.
//
// On disk per row: `i_dim/VQ_DIM` codebook indices (VQ_INDEX_BITS each, packed) +
// `i_dim/VQ_GROUP` bf16 group scales. `w[subvec] ≈ scale_group · codebook[index]`.
// This module DEFINES the bytes the converter (M2) writes and the HIP kernel (M3)
// reads. The codebook is learned once by the converter, stored in the file header,
// and embedded in the kernel. Hadamard rotation (QuIP incoherence) is DEFERRED —
// it measured no gain on these already-well-conditioned weights (docs/int3.md).

/// Weights per learned codebook entry (subvector dimension).
pub const VQ_DIM: usize = 4;
/// Codebook entries. 4096 → 12-bit index = 3.0 bpw for the indices (before scales),
/// the rate at which learned VQ ties int4 on these weights.
pub const VQ_K: usize = 4096;
/// Bits per packed codebook index (`log2(VQ_K)`).
pub const VQ_INDEX_BITS: usize = 12;
/// Weights per bf16 group scale (along the input dim).
pub const VQ_GROUP: usize = 64;

/// ‖c‖² per codebook entry — the argmin-VQ precompute, so nearest is
/// `argmin(‖c‖² − 2·x·c)` (half the flops of the full squared distance, same
/// argmin/tie-break). Shared by the CPU encoder and the GPU converter.
pub fn codebook_norms(codebook: &[f32]) -> Vec<f32> {
    (0..VQ_K)
        .map(|k| {
            codebook[k * VQ_DIM..(k + 1) * VQ_DIM]
                .iter()
                .map(|&c| c * c)
                .sum()
        })
        .collect()
}

/// Packed-index byte stride of one VQ row: `i_dim/VQ_DIM` indices × `VQ_INDEX_BITS`,
/// rounded to bytes. `i_dim` is a multiple of `VQ_DIM` and 8 for all GLM-5.2 dims,
/// so this is exact (`i_dim·VQ_INDEX_BITS / (VQ_DIM·8)`).
pub fn vq_row_bytes(i_dim: usize) -> usize {
    debug_assert_eq!(
        (i_dim / VQ_DIM * VQ_INDEX_BITS) % 8,
        0,
        "VQ row not byte-aligned"
    );
    i_dim / VQ_DIM * VQ_INDEX_BITS / 8
}

/// Number of bf16 group scales in one VQ row (`i_dim / VQ_GROUP`).
pub fn vq_groups(i_dim: usize) -> usize {
    debug_assert_eq!(
        i_dim % VQ_GROUP,
        0,
        "i_dim {i_dim} not a multiple of VQ_GROUP"
    );
    i_dim / VQ_GROUP
}

/// On-disk bytes of one VQ-quantized projection `W[o_dim, i_dim]`: `o_dim` rows of
/// packed 12-bit indices, then `o_dim · vq_groups` bf16 group scales (2 bytes each).
/// The single source of truth the converter (M2) writes and the loader (M3) reads;
/// the two arrays are stored back-to-back so one projection is one contiguous span.
pub fn vq_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * vq_row_bytes(i_dim) + o_dim * vq_groups(i_dim) * 2
}

/// O_DIRECT alignment the converter pads each expert's on-disk block up to, so one
/// expert fetch is a single block-aligned read (mirrors int4's slot alignment).
pub const VQ_ALIGN: usize = 4096;

/// The three projections of one MoE expert in on-disk order, as `(o_dim, i_dim)`:
/// gate and up map hidden→moe_inter (`i_dim = hidden`); down maps moe_inter→hidden.
pub fn vq_expert_layout(hidden: usize, moe_inter: usize) -> [(usize, usize); 3] {
    [
        (moe_inter, hidden),
        (moe_inter, hidden),
        (hidden, moe_inter),
    ]
}

/// Unpadded on-disk bytes of one expert (gate‖up‖down concatenated).
pub fn vq_expert_bytes(hidden: usize, moe_inter: usize) -> usize {
    vq_expert_layout(hidden, moe_inter)
        .iter()
        .map(|&(o, i)| vq_proj_bytes(o, i))
        .sum()
}

/// Per-expert on-disk stride: [`vq_expert_bytes`] padded up to [`VQ_ALIGN`]. The
/// fixed stride at which experts sit in a per-layer `.vq3` file.
pub fn vq_expert_stride(hidden: usize, moe_inter: usize) -> usize {
    vq_expert_bytes(hidden, moe_inter).div_ceil(VQ_ALIGN) * VQ_ALIGN
}

/// One VQ projection borrowed from an on-disk expert block: packed 12-bit indices
/// then little-endian bf16 group scales. Run it through [`VqProj::gemv`].
pub struct VqProj<'a> {
    pub indices: &'a [u8],
    pub scales: &'a [u8],
    pub o_dim: usize,
    pub i_dim: usize,
}

impl VqProj<'_> {
    /// `y[o] = Σ_subvec scale · codebook[idx]·x`. Decodes the borrowed bf16 scale
    /// bytes into a scratch and dispatches to [`matvec_vq`] — the CPU reference the
    /// HIP VQ kernel is validated against, not the perf hot path.
    pub fn gemv(&self, x: &[f32], codebook: &[f32], y: &mut [f32]) {
        let scales: Vec<u16> = self
            .scales
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        matvec_vq(
            y,
            x,
            self.indices,
            &scales,
            codebook,
            self.o_dim,
            self.i_dim,
        );
    }
}

/// Slice expert `e`'s three [`VqProj`] out of a per-layer `.vq3` buffer (`n_experts`
/// blocks at [`vq_expert_stride`]). The layout the converter writes: each expert is
/// `gate ‖ up ‖ down`, each projection indices-then-scales.
pub fn vq_expert(layer: &[u8], e: usize, hidden: usize, moe_inter: usize) -> [VqProj<'_>; 3] {
    let stride = vq_expert_stride(hidden, moe_inter);
    let blk = &layer[e * stride..e * stride + vq_expert_bytes(hidden, moe_inter)];
    let dims = vq_expert_layout(hidden, moe_inter);
    let mut off = 0;
    core::array::from_fn(|k| {
        let (o_dim, i_dim) = dims[k];
        let idx_bytes = o_dim * vq_row_bytes(i_dim);
        let sc_bytes = o_dim * vq_groups(i_dim) * 2;
        let proj = VqProj {
            indices: &blk[off..off + idx_bytes],
            scales: &blk[off + idx_bytes..off + idx_bytes + sc_bytes],
            o_dim,
            i_dim,
        };
        off += idx_bytes + sc_bytes;
        proj
    })
}

/// Write a 12-bit `idx` for the `k`-th subvector of a packed row. Two 12-bit indices
/// occupy 3 bytes; `12k % 8 ∈ {0,4}`, so the value never spans more than 2 bytes.
/// Public so the GPU converter (`bin/fp82vq`) packs indices identically to `quant_vq`.
#[inline]
pub fn set_idx(row: &mut [u8], k: usize, idx: u16) {
    let (base, shift) = (k * VQ_INDEX_BITS / 8, (k * VQ_INDEX_BITS) % 8);
    let v = (u32::from(idx) & 0xFFF) << shift;
    row[base] |= v as u8;
    row[base + 1] |= (v >> 8) as u8;
}

/// Read the 12-bit codebook index of the `k`-th subvector of a packed row.
#[inline]
fn get_idx(row: &[u8], k: usize) -> usize {
    let (base, shift) = (k * VQ_INDEX_BITS / 8, (k * VQ_INDEX_BITS) % 8);
    (((row[base] as u16 | (row[base + 1] as u16) << 8) >> shift) & 0xFFF) as usize
}

/// Dequantize an fp8 (e4m3) block-scaled weight matrix `W[o_dim, i_dim]` to f32
/// (row-major). `scale` is the F32 `weight_scale_inv` tensor, one value per
/// `block × block` tile: `w[o,i] = e4m3(packed[o,i]) · scale[(o/block)·sc_cols +
/// i/block]`. The DeepSeek/GLM fp8 convention (mirrors colibri's converter); the
/// `_inv` name is historical — it is the dequant multiplier. Offline (converter)
/// use only: this materializes `o_dim·i_dim` f32.
pub fn dequant_fp8_block(
    packed: &[u8],
    scale: &[f32],
    o_dim: usize,
    i_dim: usize,
    block: usize,
) -> Vec<f32> {
    debug_assert_eq!(packed.len(), o_dim * i_dim);
    let sc_cols = i_dim.div_ceil(block);
    debug_assert_eq!(scale.len(), o_dim.div_ceil(block) * sc_cols);
    let mut out = vec![0.0f32; o_dim * i_dim];
    for o in 0..o_dim {
        for i in 0..i_dim {
            let s = scale[(o / block) * sc_cols + i / block];
            out[o * i_dim + i] = e4m3_to_f32(packed[o * i_dim + i]) * s;
        }
    }
    out
}

/// Nearest codebook entry (by squared Euclidean distance) to a `VQ_DIM`-subvector.
/// `codebook` is `VQ_K · VQ_DIM` f32, row-major. The converter's per-subvector
/// encode; ties break to the lowest index (deterministic).
pub fn vq_nearest(sub: &[f32], codebook: &[f32]) -> u16 {
    debug_assert_eq!(sub.len(), VQ_DIM);
    debug_assert_eq!(codebook.len(), VQ_K * VQ_DIM);
    let mut best = (f32::INFINITY, 0u16);
    for k in 0..VQ_K {
        let c = &codebook[k * VQ_DIM..(k + 1) * VQ_DIM];
        let d: f32 = sub.iter().zip(c).map(|(&a, &b)| (a - b) * (a - b)).sum();
        if d < best.0 {
            best = (d, k as u16);
        }
    }
    best.1
}

/// Quantize a row-major f32 weight matrix `W[o_dim, i_dim]` to VQ-int3 against a
/// learned `codebook` (`VQ_K · VQ_DIM` f32). Returns `(indices, scales)`: `indices`
/// is `o_dim · vq_row_bytes(i_dim)` packed 12-bit codebook indices; `scales` is
/// `o_dim · (i_dim/VQ_GROUP)` bf16 group scales.
///
/// Per group the encode alternates once: assign subvectors to nearest entries under
/// the amax-derived scale, refit the scale in closed form against the chosen entries
/// (least-squares: `s = Σ w·c / Σ c·c`), then re-assign under the refitted scale.
/// The stored bf16 scale is the one the final assignment used, so the oracle
/// reproduces the on-disk values exactly.
pub fn quant_vq(w: &[f32], o_dim: usize, i_dim: usize, codebook: &[f32]) -> (Vec<u8>, Vec<u16>) {
    debug_assert_eq!(w.len(), o_dim * i_dim);
    debug_assert_eq!(
        i_dim % VQ_GROUP,
        0,
        "i_dim {i_dim} not a multiple of VQ_GROUP"
    );
    const SUBS: usize = VQ_GROUP / VQ_DIM; // subvectors per group
    let ngroups = i_dim / VQ_GROUP;
    let rb = vq_row_bytes(i_dim);
    let mut indices = vec![0u8; o_dim * rb];
    let mut scales = vec![0u16; o_dim * ngroups];
    let norms = codebook_norms(codebook);
    let nearest = |sub: &[f32; VQ_DIM]| -> u16 {
        let mut best = (f32::INFINITY, 0u16);
        for k in 0..VQ_K {
            let c = &codebook[k * VQ_DIM..(k + 1) * VQ_DIM];
            let dot: f32 = sub.iter().zip(c).map(|(&a, &b)| a * b).sum();
            let d = norms[k] - 2.0 * dot;
            if d < best.0 {
                best = (d, k as u16);
            }
        }
        best.1
    };
    let assign = |seg: &[f32], inv: f32, idxs: &mut [u16; SUBS]| {
        let mut sub = [0.0f32; VQ_DIM];
        for (t, chunk) in seg.chunks_exact(VQ_DIM).enumerate() {
            for (d, &v) in chunk.iter().enumerate() {
                sub[d] = v * inv; // normalize into the codebook's scale
            }
            idxs[t] = nearest(&sub);
        }
    };
    for o in 0..o_dim {
        let wr = &w[o * i_dim..(o + 1) * i_dim];
        let ir = &mut indices[o * rb..(o + 1) * rb];
        for grp in 0..ngroups {
            let seg = &wr[grp * VQ_GROUP..(grp + 1) * VQ_GROUP];
            let amax = seg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let mut sb = f32_to_bf16(if amax > 0.0 { amax } else { 1.0 });
            let mut idxs = [0u16; SUBS];
            assign(seg, 1.0 / bf16_to_f32(sb), &mut idxs);
            // Refit: minimize Σ (w − s·c)² over s for the chosen entries; keep the
            // amax scale if the fit is degenerate (all-zero entries or s ≤ 0).
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for (t, chunk) in seg.chunks_exact(VQ_DIM).enumerate() {
                let c = &codebook[idxs[t] as usize * VQ_DIM..][..VQ_DIM];
                for (d, &v) in chunk.iter().enumerate() {
                    num += v * c[d];
                    den += c[d] * c[d];
                }
            }
            if den > 0.0 && num > 0.0 {
                let refit = f32_to_bf16(num / den);
                if refit != sb {
                    sb = refit;
                    assign(seg, 1.0 / bf16_to_f32(sb), &mut idxs);
                }
            }
            scales[o * ngroups + grp] = sb;
            for (t, &ix) in idxs.iter().enumerate() {
                set_idx(ir, grp * SUBS + t, ix);
            }
        }
    }
    (indices, scales)
}

/// GEMV oracle against a VQ-int3 matrix: `y[o] = Σ_subvec scale[o, group(subvec)] ·
/// codebook[idx(o,subvec)] · x[subvec]`. The scalar reference the HIP VQ kernel (M3)
/// is validated against. Codebook entries and group scales decoded inline; no
/// per-expert f32 materialization.
pub fn matvec_vq(
    y: &mut [f32],
    x: &[f32],
    indices: &[u8],
    scales: &[u16],
    codebook: &[f32],
    o_dim: usize,
    i_dim: usize,
) {
    debug_assert_eq!(y.len(), o_dim);
    debug_assert_eq!(x.len(), i_dim);
    let ngroups = i_dim / VQ_GROUP;
    let nsub = i_dim / VQ_DIM;
    let rb = vq_row_bytes(i_dim);
    for (o, yo) in y.iter_mut().enumerate() {
        let ir = &indices[o * rb..(o + 1) * rb];
        let mut acc = 0.0f32;
        for k in 0..nsub {
            let i0 = k * VQ_DIM;
            let s = bf16_to_f32(scales[o * ngroups + i0 / VQ_GROUP]);
            let c = &codebook[get_idx(ir, k) * VQ_DIM..][..VQ_DIM];
            let mut dot = 0.0f32;
            for (d, &cw) in c.iter().enumerate() {
                dot += x[i0 + d] * cw;
            }
            acc += s * dot;
        }
        *yo = acc;
    }
}

// ── int4 (colibri layout): the "warm expert" format ─────────────────────────
// Per-row symmetric int4: `W[o,i] = (nibble(o,i) − 8) · scale[o]`. `packed` = o_dim
// rows of `i4_row_bytes(i_dim)` bytes (LOW nibble = col 2j, HIGH = col 2j+1); one
// f32 `scale` per output row. Matches colibri's `.qs` and the old engine's
// matvec_i4, so the moe_*_i4 kernels are bit-comparable to this reference.

/// Row stride in bytes for a per-row int4 matrix (2 nibbles/byte).
pub fn i4_row_bytes(i_dim: usize) -> usize {
    i_dim.div_ceil(2)
}

/// Reference int4 GEMV `y[o] = scale[o] · Σ_i x[i]·(nibble(o,i) − 8)` — the CPU
/// oracle the `moe_gateup_i4`/`moe_down_i4` kernels validate against.
pub fn matvec_i4(y: &mut [f32], x: &[f32], packed: &[u8], scale: &[f32], o_dim: usize, i_dim: usize) {
    debug_assert_eq!(y.len(), o_dim);
    debug_assert_eq!(x.len(), i_dim);
    let rb = i4_row_bytes(i_dim);
    for (o, yo) in y.iter_mut().enumerate() {
        let row = &packed[o * rb..(o + 1) * rb];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            let byte = row[i >> 1];
            let n = (if i & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as i32 - 8;
            acc += xi * n as f32;
        }
        *yo = acc * scale[o];
    }
}

/// Quantize `w[o_dim·i_dim]` (row-major) → per-row symmetric int4 (packed bytes +
/// per-row f32 scale). `scale[o] = max|row|/7` so the extreme maps to nibble 15
/// (value +7); nibbles clamp to `[0,15]`. Round-trips through [`matvec_i4`]. Used to
/// build oracle inputs; the live path reads colibri's already-packed int4.
pub fn quant_i4(w: &[f32], o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(w.len(), o_dim * i_dim);
    let rb = i4_row_bytes(i_dim);
    let mut packed = vec![0u8; o_dim * rb];
    let mut scale = vec![0.0f32; o_dim];
    for o in 0..o_dim {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
        scale[o] = s;
        for (i, &wi) in row.iter().enumerate() {
            let q = ((wi / s).round() as i32 + 8).clamp(0, 15) as u8;
            let bi = o * rb + (i >> 1);
            if i & 1 == 0 {
                packed[bi] = (packed[bi] & 0xF0) | q;
            } else {
                packed[bi] = (packed[bi] & 0x0F) | (q << 4);
            }
        }
    }
    (packed, scale)
}

/// On-disk bytes of one int4 projection `W[o_dim, i_dim]`: `o_dim` packed rows then
/// `o_dim` f32 per-row scales, back-to-back (one projection = one contiguous span).
pub fn i4_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * i4_row_bytes(i_dim) + o_dim * 4
}

/// Unpadded on-disk bytes of one int4 expert (gate‖gate_scale‖up‖up_scale‖down‖
/// down_scale). Reuses [`vq_expert_layout`] — the (o,i) dims are format-independent.
pub fn i4_expert_bytes(hidden: usize, moe_inter: usize) -> usize {
    vq_expert_layout(hidden, moe_inter)
        .iter()
        .map(|&(o, i)| i4_proj_bytes(o, i))
        .sum()
}

/// Per-expert on-disk stride: [`i4_expert_bytes`] padded up to [`VQ_ALIGN`], so one
/// int4 expert is a single block-aligned read (mirrors the `.vq3` stride).
pub fn i4_expert_stride(hidden: usize, moe_inter: usize) -> usize {
    i4_expert_bytes(hidden, moe_inter).div_ceil(VQ_ALIGN) * VQ_ALIGN
}

/// The six byte offsets within one int4 expert block, in [`ExpertDescI4`] field
/// order: `[gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale]`.
/// Every packed span starts 4-byte aligned (rows are `i_dim/2`, i_dim a multiple of
/// 8), so `dot_i4_wave`'s dword fast path stays valid. `ExpertDescI4` — crate::hip.
pub fn i4_slot_offsets(hidden: usize, moe_inter: usize) -> [usize; 6] {
    let [(go, gi), (uo, ui), (dno, dni)] = vq_expert_layout(hidden, moe_inter);
    let gp = 0;
    let gs = gp + go * i4_row_bytes(gi);
    let up = gs + go * 4;
    let us = up + uo * i4_row_bytes(ui);
    let dp = us + uo * 4;
    let ds = dp + dno * i4_row_bytes(dni);
    [gp, gs, up, us, dp, ds]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i4_slot_offsets_are_contiguous_and_aligned() {
        let (hidden, inter) = (6144usize, 2048usize);
        let off = i4_slot_offsets(hidden, inter);
        assert_eq!(off[0], 0);
        for &o in &off {
            assert_eq!(o % 4, 0, "packed/scale span {o} not 4-byte aligned");
        }
        // last span (down_scale) ends exactly at i4_expert_bytes.
        assert_eq!(off[5] + hidden * 4, i4_expert_bytes(hidden, inter));
    }

    #[test]
    fn i4_quant_matvec_roundtrip() {
        // quant_i4 → matvec_i4 approximates the true GEMV within int4 error.
        let (o, i) = (16usize, 64usize);
        let mut w = vec![0.0f32; o * i];
        let mut s = 0x2468u64;
        let mut rf = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for v in w.iter_mut() {
            *v = rf();
        }
        let x: Vec<f32> = (0..i).map(|_| rf()).collect();
        let mut want = vec![0.0f32; o];
        for oo in 0..o {
            want[oo] = (0..i).map(|ii| w[oo * i + ii] * x[ii]).sum();
        }
        let (packed, scale) = quant_i4(&w, o, i);
        let mut got = vec![0.0f32; o];
        matvec_i4(&mut got, &x, &packed, &scale, o, i);
        // int4 per-row quant: err bounded by ~scale·Σ|x| worst case; check it tracks.
        let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let err = want.iter().zip(&got).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 0.15 * mx + 0.1, "i4 roundtrip err={err:.3} max={mx:.3}");
    }

    #[test]
    fn read_f32_roundtrips() {
        let vals = [1.5f32, -2.25, 0.0, 3.75];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(read_f32(&bytes), vals);
    }

    // ── VQ-int3 oracle (M1) ─────────────────────────────────────────────────

    /// A tiny deterministic codebook padded to VQ_K entries (unused rows pushed far
    /// away so they never win a nearest-lookup). Entries 0..n hold `pts`.
    fn tiny_codebook(pts: &[[f32; VQ_DIM]]) -> Vec<f32> {
        let mut cb = vec![1e30f32; VQ_K * VQ_DIM]; // far-away filler
        for (k, p) in pts.iter().enumerate() {
            cb[k * VQ_DIM..(k + 1) * VQ_DIM].copy_from_slice(p);
        }
        cb
    }

    #[test]
    fn vq_expert_slicing_roundtrips_gemv() {
        // Build a 1-expert `.vq3` block from quant_vq of synthetic gate/up/down, slice
        // it back with vq_expert, and check each projection's gemv equals matvec_vq on
        // the original arrays — the converter→loader byte contract for a whole expert.
        let (hidden, moe_inter) = (VQ_GROUP, VQ_GROUP); // tiny but valid (one group each)
        let cb = tiny_codebook(&[[1.0, -1.0, 0.5, -0.5], [0.25, 0.5, 0.75, 1.0]]);
        let stride = vq_expert_stride(hidden, moe_inter);
        let mut layer = vec![0u8; stride]; // one expert
        let dims = vq_expert_layout(hidden, moe_inter);
        // Deterministic weights per projection; keep originals to GEMV against.
        let mut originals = Vec::new();
        let mut off = 0;
        for (p, &(o_dim, i_dim)) in dims.iter().enumerate() {
            let mut w = vec![0.0f32; o_dim * i_dim];
            for (n, wv) in w.iter_mut().enumerate() {
                let e = (n + p) % 2;
                *wv = cb[e * VQ_DIM + (n % VQ_DIM)] * 0.5;
            }
            let (indices, scales) = quant_vq(&w, o_dim, i_dim, &cb);
            let ib = o_dim * vq_row_bytes(i_dim);
            layer[off..off + ib].copy_from_slice(&indices);
            for (s, dst) in scales.iter().zip(layer[off + ib..].chunks_exact_mut(2)) {
                dst.copy_from_slice(&s.to_le_bytes());
            }
            off += vq_proj_bytes(o_dim, i_dim);
            originals.push((indices, scales, o_dim, i_dim));
        }
        let projs = vq_expert(&layer, 0, hidden, moe_inter);
        for (proj, (indices, scales, o_dim, i_dim)) in projs.iter().zip(&originals) {
            let x: Vec<f32> = (0..*i_dim).map(|k| (k + 1) as f32).collect();
            let mut y_load = vec![0.0f32; *o_dim];
            let mut y_ref = vec![0.0f32; *o_dim];
            proj.gemv(&x, &cb, &mut y_load);
            matvec_vq(&mut y_ref, &x, indices, scales, &cb, *o_dim, *i_dim);
            assert_eq!(y_load, y_ref);
        }
    }

    #[test]
    fn vq_index_pack_unpack_all_positions() {
        // 12-bit indices, 2 per 3 bytes; every subvector slot must round-trip across
        // the nibble straddle (even k at shift 0, odd k at shift 4).
        let n = 6; // 6 subvectors → i_dim = 24
        let mut row = vec![0u8; vq_row_bytes(n * VQ_DIM)];
        let vals = [0u16, 4095, 1, 2048, 4094, 7];
        for (k, &v) in vals.iter().enumerate() {
            set_idx(&mut row, k, v);
        }
        for (k, &v) in vals.iter().enumerate() {
            assert_eq!(get_idx(&row, k), v as usize, "slot {k}");
        }
    }

    #[test]
    fn vq_nearest_picks_closest_and_ties_low() {
        let cb = tiny_codebook(&[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]);
        assert_eq!(vq_nearest(&[0.1, 0.0, 0.0, 0.0], &cb), 0); // nearest 0
        assert_eq!(vq_nearest(&[0.9, 1.0, 1.0, 1.0], &cb), 1); // nearest 1
        assert_eq!(vq_nearest(&[1.0, 1.0, 1.0, 1.0], &cb), 1); // exact tie 1==2 → lowest
    }

    #[test]
    fn vq_gemv_matches_codebook_dot() {
        // One full group (i_dim = VQ_GROUP), one row. Each subvector is one of two
        // codebook entries; amax over the group is 1.0 ⇒ scale 1, exact quant.
        let cb = tiny_codebook(&[[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]);
        let i_dim = VQ_GROUP;
        let nsub = i_dim / VQ_DIM;
        let entry = |t: usize| t % 2; // alternate entries 0,1,0,1,…
        let mut w = vec![0.0f32; i_dim];
        for t in 0..nsub {
            let e = entry(t);
            w[t * VQ_DIM..(t + 1) * VQ_DIM].copy_from_slice(&cb[e * VQ_DIM..(e + 1) * VQ_DIM]);
        }
        let (idx, scales) = quant_vq(&w, 1, i_dim, &cb);
        let x: Vec<f32> = (0..i_dim).map(|i| (i + 1) as f32).collect();
        let mut y = [0.0f32];
        matvec_vq(&mut y, &x, &idx, &scales, &cb, 1, i_dim);
        // Expected = Σ_t codebook[entry(t)] · x_subvec(t), with scale 1.
        let mut exp = 0.0f32;
        for t in 0..nsub {
            let e = entry(t);
            for d in 0..VQ_DIM {
                exp += cb[e * VQ_DIM + d] * x[t * VQ_DIM + d];
            }
        }
        assert!((y[0] - exp).abs() < 1e-4, "y0={} exp={}", y[0], exp);
    }

    #[test]
    fn vq_roundtrip_reconstructs_within_codebook() {
        // Unit-peak codebook entries and a bf16-exact group scale ⇒ normalized
        // subvectors land exactly on entries, so quant→dequant is exact — the
        // faithfulness the converter relies on. (Both entries peak at 1 so the group
        // amax = the scale S; S=0.5 is bf16-exact.)
        let cb = tiny_codebook(&[[1.0, -1.0, 0.5, -0.5], [0.25, 0.5, 0.75, 1.0]]);
        let i_dim = VQ_GROUP;
        let nsub = i_dim / VQ_DIM;
        let entry = |t: usize| t % 2;
        let scale = 0.5f32;
        let mut w = vec![0.0f32; i_dim];
        for t in 0..nsub {
            let e = entry(t);
            for d in 0..VQ_DIM {
                w[t * VQ_DIM + d] = cb[e * VQ_DIM + d] * scale;
            }
        }
        let (idx, scales) = quant_vq(&w, 1, i_dim, &cb);
        let s = bf16_to_f32(scales[0]);
        assert!((s - scale).abs() < 1e-6, "group scale {s} != {scale}");
        for t in 0..nsub {
            let e = &cb[get_idx(&idx, t) * VQ_DIM..][..VQ_DIM];
            for d in 0..VQ_DIM {
                assert!((e[d] * s - w[t * VQ_DIM + d]).abs() < 1e-4, "t{t} d{d}");
            }
        }
    }

    #[test]
    fn fp8_block_dequant_applies_tile_scale() {
        // 2×2 fp8 matrix, block=1 (per-element scale for the test) — check the tile
        // indexing and that e4m3 decode is wired. 0x38=1.0, 0x40=2.0, 0xB8=-1.0.
        let packed = [0x38u8, 0x40, 0xB8, 0x00]; // [[1, 2],[-1, 0]]
        let scale = [10.0f32, 100.0, 1000.0, 1.0]; // per element (block=1)
        let out = dequant_fp8_block(&packed, &scale, 2, 2, 1);
        assert_eq!(out, [10.0, 200.0, -1000.0, 0.0]);
    }

    #[test]
    fn matvec_fp8_matches_dequant_dot() {
        // Same 2×2 fp8 as above; GEMV must equal the dequant-then-dot reference.
        let packed = [0x38u8, 0x40, 0xB8, 0x00]; // [[1,2],[-1,0]] × per-elem scale
        let scale = [10.0f32, 100.0, 1000.0, 1.0];
        let x = [3.0f32, 5.0];
        let mut y = [0.0f32; 2];
        matvec_fp8(&mut y, &x, &packed, &scale, 2, 1);
        // row0: 10·3 + 200·5 = 1030 ; row1: -1000·3 + 0·5 = -3000
        assert_eq!(y, [1030.0, -3000.0]);
    }

    #[test]
    fn matvec_i8_matches_dot() {
        // W=[[1,-2],[3,4]] int8, scale=[0.5,2.0], x=[1,1].
        let packed = [1u8, (-2i8) as u8, 3, 4];
        let scale: Vec<u8> = [0.5f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let x = [1.0f32, 1.0];
        let mut y = [0.0f32; 2];
        matvec_i8(&mut y, &x, &packed, &scale, 2);
        assert_eq!(y, [(1.0 - 2.0) * 0.5, (3.0 + 4.0) * 2.0]);
    }
}
