//! The **VQ-int3** routed-expert format: its geometry, its on-disk block, the offline
//! encoder that writes it and the scalar oracle the HIP kernel is validated against.
//!
//! **Split out of `quant.rs` on 2026-08-15, by FORMAT.** CodeScene scored the single file
//! 8.54: past ~500 lines it prices a module's LCOM4, and `quant.rs` had grown four
//! independent formats plus a learner and a naming table with no call edge between them.
//! Everything here shares `VQ_DIM`/`VQ_GROUP` and the 12-bit index packing — one format,
//! one owner. What it does NOT own is the block geometry every routed format shares
//! (`expert_bytes`, `expert_stride`, `slot_offsets`, `vq_expert_layout`); those stay in
//! the parent, because a second copy of the walk is what `slot_offsets`'s own doc exists
//! to prevent.
//!
//! **Nothing was rewritten.** Every body and comment travelled verbatim — the comments here
//! carry the measurements that chose the constants, and a paraphrase drops the evidence and
//! keeps the number. The only edits: `subvec` and `group_amax_bf16` widened from private to
//! `pub(super)` because [`super::kmeans`] fits its codebook against the same two rules (a
//! second copy of either is a codebook fitted in a space the encoder never visits), and
//! intra-doc links that now cross a module boundary gained a `super::`.
//!
//! Every public name is re-exported by `quant.rs`, so `rivoli_artifact::quant::quant_vq`
//! and friends still resolve for `bin/convert` and the engine's kernel tests.

use super::kmeans::codebook_norms;
use super::{
    Subvectors, VqW, Weights, debug_check_gemv, expert_bytes, expert_stride, read_le, slot_offsets,
    vq_expert_layout,
};
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};

// ─────────────────────────────────────────────────────────────────────────────
// Vector-quantized int3 experts (M1 scalar oracle).
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
// and embedded in the kernel. Hadamard rotation (QuIP incoherence) is CLOSED, not merely
// deferred: this comment used to cite `docs/int3.md` for "no gain on these
// already-well-conditioned weights", and that file was never committed. The claim was
// re-derived from evidence that does exist and it holds — see
// docs/investigations/codebook-rotation.md, which also measures the OTHER argument (one
// codebook shared by 75 layers) and finds 0.09% to recover against a 2% bar.

/// Weights per learned codebook entry (subvector dimension).
pub const VQ_DIM: usize = 4;
/// Codebook entries. 4096 → 12-bit index = 3.0 bpw for the indices (before scales),
/// the rate at which learned VQ ties int4 on these weights.
pub const VQ_K: usize = 4096;
/// Bits per packed codebook index (`log2(VQ_K)`).
pub const VQ_INDEX_BITS: usize = 12;
/// Weights per bf16 group scale (along the input dim).
pub const VQ_GROUP: usize = 64;

/// Subvectors per [`VQ_GROUP`] scale — the encoder's inner loop count, and the width of the
/// index block one group produces.
const VQ_SUBS: usize = VQ_GROUP / VQ_DIM;

/// The `i`-th `VQ_DIM`-wide subvector of a flat `[n][VQ_DIM]` array. Codebook entry,
/// k-means sample point and weight subvector are the SAME indexing rule — spelled once, so
/// a stride mistake cannot survive in the encoder while the learner stays right (they are
/// the two halves of one round trip, and a half-right stride still produces a legal file).
/// `pub(super)` since the 2026-08-15 split, so [`super::kmeans`] indexes the sample and the
/// centroid table through this same rule rather than its own copy — the half-right stride
/// above is exactly what a second copy would produce.
#[inline]
pub(super) fn subvec(a: &Subvectors, i: usize) -> &Subvectors {
    &a[i * VQ_DIM..(i + 1) * VQ_DIM]
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

/// Decode one VQ projection to a DENSE row-major `W[o_dim, i_dim]` — the inverse of
/// [`quant_vq`], and the SINGLE VQ reader.
///
/// **RESTORED 2026-08-06 from tag `archive/i4-audit`.** Track H deleted this as having zero
/// callers *including tests*, which was true on that branch and false on `main`:
/// `examples/vq_k_probe.rs` (added 2026-08-04) calls it and names it "the shipping decoder".
/// Same shape as the `release.yml` consumer the same track found — a live user outside the
/// branch's field of view. A reachability claim is scoped to the tree that measured it.
///
/// Its historical caller was `bin/i4_audit`, which compared what this returns
/// `vq3 → int4` chain must decode identically or its "old set" baseline describes a
/// converter that never existed. (The same rule [`super::int4::write_i4_proj`] states for the `.i4`
/// writer.)
///
/// Materializes `o_dim·i_dim` f32 — offline use only; the decode path is
/// [`matvec_vq`], which never builds the dense matrix.
pub fn vq_decode_proj(p: &VqProj, codebook: &Subvectors) -> Vec<f32> {
    let (o_dim, i_dim) = (p.o_dim, p.i_dim);
    let (rb, ng, nsub) = (vq_row_bytes(i_dim), vq_groups(i_dim), i_dim / VQ_DIM);
    // Through `scales_u16` rather than re-spelling `u16::from_le_bytes` here: that spelling
    // is a layout rule, and this used to be one of the copies its doc names. The extra
    // `o_dim · ng` u16 is nothing beside the `o_dim · i_dim` f32 this function exists to
    // build — the argument that keeps `scales_u16` off the decode path does not apply here.
    let scales = p.scales_u16();
    let mut w = vec![0f32; o_dim * i_dim];
    for o in 0..o_dim {
        let ir = &p.indices[o * rb..(o + 1) * rb];
        for k in 0..nsub {
            let s = bf16_to_f32(scales[o * ng + (k * VQ_DIM) / VQ_GROUP]);
            for (d, &cw) in subvec(codebook, get_idx(ir, k)).iter().enumerate() {
                w[o * i_dim + k * VQ_DIM + d] = s * cw;
            }
        }
    }
    w
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

/// Unpadded on-disk bytes of one expert (gate‖up‖down concatenated).
pub fn vq_expert_bytes(expert_in: usize, moe_inter: usize) -> usize {
    expert_bytes(expert_in, moe_inter, vq_proj_bytes)
}

/// Per-expert on-disk stride: [`vq_expert_bytes`] padded up to [`super::VQ_ALIGN`]. The
/// fixed stride at which experts sit in a per-layer `.vq3` file.
pub fn vq_expert_stride(expert_in: usize, moe_inter: usize) -> usize {
    expert_stride(vq_expert_bytes(expert_in, moe_inter))
}

/// One VQ projection borrowed from an on-disk expert block: packed 12-bit indices
/// then little-endian bf16 group scales. Run it through [`matvec_vq`].
///
/// (CORRECTED 2026-08-15 during the split: this said `VqProj::gemv`, a method that has
/// never existed on this type — the entry point is the free function, and the dead link
/// only surfaced when `cargo doc` was run over the new module tree.)
pub struct VqProj<'a> {
    pub indices: &'a [u8],
    pub scales: &'a [u8],
    pub o_dim: usize,
    pub i_dim: usize,
}

impl VqProj<'_> {
    /// The borrowed group scales as the little-endian bf16 WORDS [`matvec_vq`] takes.
    ///
    /// Allocates, so it is for offline and test use only — the decode path never calls it
    /// (`matvec_vq` is handed words the loader already has). It exists because the
    /// byte→word convention is THIS
    /// module's: every caller that re-spelled `u16::from_le_bytes` over `chunks_exact(2)`
    /// was a copy of a layout rule that belongs here, and one of them lives in a binary
    /// that never sees a change to this format.
    pub fn scales_u16(&self) -> Vec<u16> {
        read_le(self.scales).map(u16::from_le_bytes).collect()
    }
}

/// Slice expert `e`'s three [`VqProj`] out of a per-layer `.vq3` buffer (`n_experts`
/// blocks at [`vq_expert_stride`]). The layout the converter writes: each expert is
/// `gate ‖ up ‖ down`, each projection indices-then-scales.
pub fn vq_expert(layer: &[u8], e: usize, expert_in: usize, moe_inter: usize) -> [VqProj<'_>; 3] {
    let stride = vq_expert_stride(expert_in, moe_inter);
    let blk = &layer[e * stride..e * stride + vq_expert_bytes(expert_in, moe_inter)];
    let dims = vq_expert_layout(expert_in, moe_inter);
    let off = vq_slot_offsets(expert_in, moe_inter); // the single source of the layout
    core::array::from_fn(|k| {
        let (o_dim, i_dim) = dims[k];
        let (io, so) = (off[k * 2], off[k * 2 + 1]); // indices start, scales start
        let sc_end = so + o_dim * vq_groups(i_dim) * 2;
        VqProj {
            indices: &blk[io..so], // indices span [io, so): scales begin at so
            scales: &blk[so..sc_end],
            o_dim,
            i_dim,
        }
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

/// Read the 12-bit codebook index of the `k`-th subvector of a packed row — the one place
/// the 12-bits-per-index packing is spelled out, so [`matvec_vq`] and the tests cannot drift
/// apart on it. Private since 2026-08-05: it was `pub` for "the offline readers", and both
/// of those (`bin/i4_audit`, `vq_decode_proj`) are now retired to tag `archive/i4-audit`.
#[inline]
fn get_idx(row: &[u8], k: usize) -> usize {
    let (base, shift) = (k * VQ_INDEX_BITS / 8, (k * VQ_INDEX_BITS) % 8);
    (((row[base] as u16 | (row[base + 1] as u16) << 8) >> shift) & 0xFFF) as usize
}

/// The closed-form group refit: least-squares `s = Σ w·c / Σ c·c` over the codebook
/// entries `idxs` already chose for `seg`, returned as bf16. `None` when the fit is
/// degenerate (all-zero entries, or `s ≤ 0`) — the caller keeps its amax scale.
///
/// Shared with `bin/convert`'s `--gpu` encoder, which is documented (and asserted by
/// `--validate`) to produce output BIT-IDENTICAL to [`quant_vq`]. Two copies of this
/// accumulation is two places for one to pick up a different summation order and turn
/// that guarantee into a mismatch report against the shipped bytes.
pub fn vq_refit(seg: &Weights, idxs: &[u16], codebook: &Subvectors) -> Option<u16> {
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for (t, chunk) in seg.chunks_exact(VQ_DIM).enumerate() {
        let c = subvec(codebook, idxs[t] as usize);
        for (d, &v) in chunk.iter().enumerate() {
            num += v * c[d];
            den += c[d] * c[d];
        }
    }
    (den > 0.0 && num > 0.0).then(|| f32_to_bf16(num / den))
}

/// A group's bf16 amax scale. ONE spelling, because it is both the encoder's starting scale
/// and the normalizer [`sample_subvectors`] draws the codebook sample under: a sample
/// normalized differently from the search fits the codebook in a space the encoder never
/// visits, and nothing about the resulting file looks wrong.
pub(super) fn group_amax_bf16(grp: &Weights) -> u16 {
    let amax = grp.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    f32_to_bf16(if amax > 0.0 { amax } else { 1.0 })
}

/// A codebook and its ‖c‖² precompute as one value: the argmin reads both, and a norm table
/// fitted to a different codebook still has the right length and still returns an index.
struct VqEncoder<'a> {
    codebook: &'a Subvectors,
    norms: Vec<f32>,
}

impl<'a> VqEncoder<'a> {
    fn new(codebook: &'a Subvectors) -> Self {
        VqEncoder {
            norms: codebook_norms(codebook),
            codebook,
        }
    }

    /// Nearest entry to an already-normalized subvector — see [`codebook_norms`] for why
    /// `argmin(‖c‖² − 2·x·c)` picks the same entry (and breaks ties the same way) as the
    /// full squared distance.
    fn nearest(&self, sub: &[f32; VQ_DIM]) -> u16 {
        let mut best = (f32::INFINITY, 0u16);
        for k in 0..VQ_K {
            let dot: f32 = sub
                .iter()
                .zip(subvec(self.codebook, k))
                .map(|(&a, &b)| a * b)
                .sum();
            let d = self.norms[k] - 2.0 * dot;
            if d < best.0 {
                best = (d, k as u16);
            }
        }
        best.1
    }

    /// Assign every subvector of one group to its nearest entry, under `inv = 1/scale`.
    fn assign(&self, seg: &Weights, inv: f32, idxs: &mut [u16; VQ_SUBS]) {
        for (t, chunk) in seg.chunks_exact(VQ_DIM).enumerate() {
            let mut sub = [0.0f32; VQ_DIM];
            for (d, &v) in chunk.iter().enumerate() {
                sub[d] = v * inv; // normalize into the codebook's scale
            }
            idxs[t] = self.nearest(&sub);
        }
    }

    /// Per group: assign under the amax scale, refit in closed form, re-assign under the
    /// refitted scale; a degenerate fit keeps the amax pass.
    fn encode_group(&self, seg: &Weights) -> (u16, [u16; VQ_SUBS]) {
        let mut sb = group_amax_bf16(seg);
        let mut idxs = [0u16; VQ_SUBS];
        self.assign(seg, 1.0 / bf16_to_f32(sb), &mut idxs);
        if let Some(refit) = vq_refit(seg, &idxs, self.codebook)
            && refit != sb
        {
            sb = refit;
            self.assign(seg, 1.0 / bf16_to_f32(sb), &mut idxs);
        }
        (sb, idxs)
    }

    /// One row of a projection: each `VQ_GROUP` of weights contributes one bf16 scale and
    /// `VQ_SUBS` packed 12-bit indices, in order.
    fn encode_row(&self, wr: &Weights, ir: &mut [u8], sr: &mut [u16]) {
        for (grp, (seg, out)) in wr.chunks_exact(VQ_GROUP).zip(sr.iter_mut()).enumerate() {
            let (sb, idxs) = self.encode_group(seg);
            *out = sb;
            for (t, &ix) in idxs.iter().enumerate() {
                set_idx(ir, grp * VQ_SUBS + t, ix);
            }
        }
    }
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
pub fn quant_vq(
    w: &Weights,
    o_dim: usize,
    i_dim: usize,
    codebook: &Subvectors,
) -> (Vec<u8>, Vec<u16>) {
    debug_assert_eq!(w.len(), o_dim * i_dim);
    // `vq_groups` carries the "multiple of VQ_GROUP" assertion this used to restate.
    let (rb, ngroups) = (vq_row_bytes(i_dim), vq_groups(i_dim));
    let mut indices = vec![0u8; o_dim * rb];
    let mut scales = vec![0u16; o_dim * ngroups];
    let enc = VqEncoder::new(codebook);
    for ((wr, ir), sr) in w
        .chunks_exact(i_dim)
        .zip(indices.chunks_exact_mut(rb))
        .zip(scales.chunks_exact_mut(ngroups))
    {
        enc.encode_row(wr, ir, sr);
    }
    (indices, scales)
}

/// GEMV oracle against a VQ-int3 matrix: `y[o] = Σ_subvec scale[o, group(subvec)] ·
/// codebook[idx(o,subvec)] · x[subvec]`. The scalar reference the HIP VQ kernel (M3)
/// is validated against. Codebook entries and group scales decoded inline; no
/// per-expert f32 materialization.
pub fn matvec_vq(y: &mut [f32], x: &[f32], w: VqW<'_>, shape: [usize; 2]) {
    let VqW {
        indices,
        scales,
        codebook,
    } = w;
    let [o_dim, i_dim] = shape;
    let ngroups = i_dim / VQ_GROUP;
    let nsub = i_dim / VQ_DIM;
    let rb = vq_row_bytes(i_dim);
    debug_check_gemv(y, x, o_dim, i_dim);
    for (o, yo) in y.iter_mut().enumerate() {
        let ir = &indices[o * rb..(o + 1) * rb];
        let mut acc = 0.0f32;
        for k in 0..nsub {
            let i0 = k * VQ_DIM;
            let s = bf16_to_f32(scales[o * ngroups + i0 / VQ_GROUP]);
            let c = subvec(codebook, get_idx(ir, k));
            let mut dot = 0.0f32;
            for (d, &cw) in c.iter().enumerate() {
                dot += x[i0 + d] * cw;
            }
            acc += s * dot;
        }
        *yo = acc;
    }
}

// `vq_decode_proj` — dense `.vq3` decode, the unfused inverse of `quant_vq` — was DELETED
// 2026-08-05 with `bin/i4_audit`, its only caller. It is at tag `archive/i4-audit` and comes
// back with the tool if a VQ question reopens.
//
// It was briefly kept as "the one unfused statement of the read convention", which is the
// wrong reason: nothing exercised it, not even a test, so it asserted a bit-identity with
// `matvec_vq` that nothing checked and that would drift silently. An unverified reference
// implementation is worse than none — a reader trusts it. `dequant_i4` reads as the same
// shape but is NOT: it has a round-trip test against `quant_i4`, which is exactly the
// difference, and why it stays.

/// The six byte offsets within one int3-VQ expert block:
/// `[gate.indices, gate.scales, up.indices, up.scales, down.indices, down.scales]`.
/// The SINGLE source of the `.vq3` slot layout — [`vq_expert`] slices by it and the pin
/// points its descriptors at it, so the two cannot disagree.
pub fn vq_slot_offsets(expert_in: usize, moe_inter: usize) -> [usize; 6] {
    slot_offsets(expert_in, moe_inter, |o, i| {
        (o * vq_row_bytes(i), o * vq_groups(i) * 2)
    })
}

#[cfg(test)]
mod tests {
    // The VQ fixtures are hand-built codebooks whose entries the encoder must land on
    // exactly, so an assertion failure here is a decode or packing bug and not a tolerance
    // question. Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

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
        let (expert_in, moe_inter) = (VQ_GROUP, VQ_GROUP); // tiny but valid (one group each)
        let cb = tiny_codebook(&[[1.0, -1.0, 0.5, -0.5], [0.25, 0.5, 0.75, 1.0]]);
        let stride = vq_expert_stride(expert_in, moe_inter);
        let mut layer = vec![0u8; stride]; // one expert
        let dims = vq_expert_layout(expert_in, moe_inter);
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
        let projs = vq_expert(&layer, 0, expert_in, moe_inter);
        for (proj, (indices, scales, o_dim, i_dim)) in projs.iter().zip(&originals) {
            let x: Vec<f32> = (0..*i_dim).map(|k| (k + 1) as f32).collect();
            let mut y_load = vec![0.0f32; *o_dim];
            let mut y_ref = vec![0.0f32; *o_dim];
            // GEMV the SLICED projection (decoding its borrowed bf16 scale bytes) and
            // the ORIGINAL arrays — the converter→loader byte contract for an expert.
            let ps = proj.scales_u16();
            matvec_vq(
                &mut y_load,
                &x,
                VqW::new(proj.indices, &ps, &cb),
                [proj.o_dim, proj.i_dim],
            );
            matvec_vq(
                &mut y_ref,
                &x,
                VqW::new(indices, scales, &cb),
                [*o_dim, *i_dim],
            );
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
    fn vq_gemv_matches_codebook_dot() {
        // One full group (i_dim = VQ_GROUP), one row. Each subvector is one of two
        // codebook entries; amax over the group is 1.0 ⇒ scale 1, exact quant.
        let cb = tiny_codebook(&[[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]);
        let i_dim = VQ_GROUP;
        let w = alternating_entries(&cb, i_dim, 1.0);
        let (idx, scales) = quant_vq(&w, 1, i_dim, &cb);
        let x: Vec<f32> = (0..i_dim).map(|i| (i + 1) as f32).collect();
        let mut y = [0.0f32];
        matvec_vq(&mut y, &x, VqW::new(&idx, &scales, &cb), [1, i_dim]);
        // Expected = Σ_t codebook[entry(t)] · x_subvec(t), with scale 1 — spelled straight
        // off `w`, which IS the codebook entries at this scale.
        let exp: f32 = w.iter().zip(&x).map(|(&wi, &xi)| wi * xi).sum();
        assert!((y[0] - exp).abs() < 1e-4, "y0={} exp={}", y[0], exp);
    }

    /// `i_dim` weights built from codebook entries 0 and 1 in alternation, each times
    /// `scale`. Both fixtures below need it: entries peak at 1, so the group amax IS
    /// `scale` and the normalized subvectors land exactly on entries.
    fn alternating_entries(cb: &Subvectors, i_dim: usize, scale: f32) -> Vec<f32> {
        (0..i_dim)
            .map(|n| cb[((n / VQ_DIM) % 2) * VQ_DIM + n % VQ_DIM] * scale)
            .collect()
    }

    #[test]
    fn vq_roundtrip_reconstructs_within_codebook() {
        // Unit-peak codebook entries and a bf16-exact group scale ⇒ normalized
        // subvectors land exactly on entries, so quant→dequant is exact — the
        // faithfulness the converter relies on. (Both entries peak at 1 so the group
        // amax = the scale S; S=0.5 is bf16-exact.)
        let cb = tiny_codebook(&[[1.0, -1.0, 0.5, -0.5], [0.25, 0.5, 0.75, 1.0]]);
        let (i_dim, scale) = (VQ_GROUP, 0.5f32);
        let w = alternating_entries(&cb, i_dim, scale);
        let (idx, scales) = quant_vq(&w, 1, i_dim, &cb);
        let s = bf16_to_f32(scales[0]);
        assert!((s - scale).abs() < 1e-6, "group scale {s} != {scale}");
        for (t, sub) in w.chunks_exact(VQ_DIM).enumerate() {
            let e = subvec(&cb, get_idx(&idx, t));
            for (d, (&ed, &wd)) in e.iter().zip(sub).enumerate() {
                assert!((ed * s - wd).abs() < 1e-4, "t{t} d{d}");
            }
        }
    }
}
