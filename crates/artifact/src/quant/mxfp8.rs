//! The **MX-FP8** (`.mxfp8`) routed-expert format: one e4m3 byte per weight under ue8m0
//! block scales, which `Inkling-2-Small-MXFP8` ships for all 48 of its layers, so rivoli
//! REPACKS rather than quantizes — exactly as [`super::f4`] does for the 4-bit sibling.
//!
//! **Why this is not `.f4` with a wider codebook.** The two formats share a block size (32)
//! and a scale encoding (a bare exponent byte), and nothing else: `.f4` packs two e2m1
//! nibbles per byte and `.f4`'s block is `o·i/2 + o·ceil(i/32)` bytes, where this one is
//! `o·i + o·ceil(i/32)`. A `.f4` block resolved through these offsets — or the reverse —
//! finds a projection at the wrong address, so the container discriminates on the header
//! magic ([`crate::format::header::MXFP8_MAGIC`]) and on the descriptor type, never on the
//! length.
//!
//! **The geometry is dictated by the source, not chosen here.** `MXFP8_GROUP` is the OCP MX
//! block size and is what the checkpoint's own `w13_weight_scale` shape carries: Small's
//! `[E, 4608, 144]` is one exponent per 32 weights along the input dim, ROW-WISE, which is
//! precisely the `[1, 32]` the root `quantization_config` declares — and NOT the `[32, 32]`
//! the `hf_quant_config.json` parameter group declares, which the header shapes refute
//! (`docs/reference/inkling2-architecture.md` §5, trap T1). Reading those scales as a 32×32
//! tile grid is the one mistake that produces a self-consistent wrong artifact, which is why
//! [`mxfp8_groups`] is the only place the grid is derived.

use super::{expert_bytes, expert_stride, slot_offsets};

// --- MXFP8 (`.mxfp8`) — Inkling-2-Small's native routed-expert format ------------------

/// Weights per ue8m0 scale along the input dim. 32 is the OCP MX block size; it coincides
/// with [`super::F4_GROUP`] and that coincidence is a property of the two SOURCES, not an
/// invitation to share the constant — a format that moved would move one model's artifact.
pub const MXFP8_GROUP: usize = 32;

/// The ue8m0 NaN, `0xFF`. Not decodable as a scale (`2^128` overflows f32), and the kernels
/// cannot refuse it: `prelude::dot_fp8`'s `e8m0f` returns a quiet NaN and `moe_fixed`'s
/// saturating clamp then launders it into a finite ±2^14 — 32 weights of plausible garbage
/// with no error anywhere. [`super::f4::e8m0`] refuses the same byte for the same reason;
/// the repack is the only path that reads every routed scale byte, so it is where the
/// refusal has to name the tensor, row and group.
pub const MXFP8_NAN_SCALE: u8 = 0xFF;

/// Number of ue8m0 group scales in one MXFP8 row (`ceil(i_dim / MXFP8_GROUP)`).
pub fn mxfp8_groups(i_dim: usize) -> usize {
    i_dim.div_ceil(MXFP8_GROUP)
}

/// Row stride in bytes for an MXFP8 matrix: one e4m3 byte per weight, so the row IS the
/// input dim (where `.f4` halves it).
pub fn mxfp8_row_bytes(i_dim: usize) -> usize {
    i_dim
}

/// On-disk bytes of one MXFP8-packed projection `W[o_dim, i_dim]`: the e4m3 rows, then one
/// ue8m0 byte per group. Mirrors `vq_proj_bytes`/`i4_proj_bytes`/`f4_proj_bytes`, so the
/// block layout and [`super::VQ_ALIGN`] stride math are shared across all four formats.
pub fn mxfp8_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * mxfp8_row_bytes(i_dim) + o_dim * mxfp8_groups(i_dim)
}

/// Unpadded on-disk bytes of one MXFP8 expert (w1‖w1_scale‖w3‖w3_scale‖w2‖w2_scale — see
/// [`super::naming::V4_PROJ`] for why that is gate/up/down order).
pub fn mxfp8_expert_bytes(expert_in: usize, moe_inter: usize) -> usize {
    expert_bytes(expert_in, moe_inter, mxfp8_proj_bytes)
}

/// Per-expert on-disk stride: [`mxfp8_expert_bytes`] padded up to [`super::VQ_ALIGN`], so one
/// MXFP8 expert is a single block-aligned read. Small's own block is already a multiple of
/// 4096 (32,845,824 B), which is a fact about its dims and not a property the function
/// assumes.
pub fn mxfp8_expert_stride(expert_in: usize, moe_inter: usize) -> usize {
    expert_stride(mxfp8_expert_bytes(expert_in, moe_inter))
}

/// The six byte offsets within one MXFP8 expert block, in `ExpertDescF4` field order:
/// `[w1, w1_scale, w3, w3_scale, w2, w2_scale]` — the gate/up/down order of
/// [`super::naming::V4_PROJ`], which the shared ABI (`MoeF4Situ`, six `*const u8`) spells
/// once for both byte-per-scale formats.
pub fn mxfp8_slot_offsets(expert_in: usize, moe_inter: usize) -> [usize; 6] {
    slot_offsets(expert_in, moe_inter, |o, i| {
        (o * mxfp8_row_bytes(i), o * mxfp8_groups(i))
    })
}

#[cfg(test)]
mod tests {
    // The subject is arithmetic over config dims, so nothing here touches a file. Crate-wide
    // `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// **Inkling-2-Small's exact block geometry, written out rather than derived.**
    ///
    /// The six spans and the block size are the numbers a Small expert occupies on disk, so
    /// they are pinned here against the shipped dims (hidden 4608, `intermediate_size` 2304,
    /// 384 routed experts) and not against this module's own arithmetic: a change to
    /// `mxfp8_groups` or `mxfp8_row_bytes` moves at least one of them, and a test that
    /// recomputed them would move with it.
    ///
    /// The last two assertions are the ones a WRONG layout cannot satisfy while keeping the
    /// size: every packed span starts on a 4-byte boundary (the device reads them as `u32`
    /// lanes), and the padded stride equals the unpadded size — Small's block is already a
    /// multiple of `VQ_ALIGN`, so a `.mxfp8` file for this model carries no padding at all.
    #[test]
    fn mxfp8_geometry_matches_inkling_small() {
        let (expert_in, inter) = (4608usize, 2304usize);
        // 2304 rows x 4608 e4m3 bytes, then 2304 x 144 ue8m0 bytes.
        let gate_packed = 2304 * 4608;
        let gate_scale = 2304 * 144;
        assert_eq!(gate_packed, 10_616_832);
        assert_eq!(gate_scale, 331_776);
        let off = mxfp8_slot_offsets(expert_in, inter);
        assert_eq!(
            off,
            [
                0,
                gate_packed,
                gate_packed + gate_scale,
                2 * gate_packed + gate_scale,
                2 * (gate_packed + gate_scale),
                3 * gate_packed + 2 * gate_scale,
            ],
            "the .mxfp8 slot layout moved"
        );
        // down is [hidden, inter] packed + [hidden, inter/32] scales — the same two numbers
        // by coincidence of dims, asserted rather than assumed.
        assert_eq!(off[4] - off[3], gate_scale);
        assert_eq!(expert_in * inter, gate_packed);
        assert_eq!(expert_in * 72, gate_scale);
        assert_eq!(mxfp8_expert_bytes(expert_in, inter), 32_845_824);
        assert_eq!(mxfp8_expert_stride(expert_in, inter), 32_845_824);
        assert_eq!(
            mxfp8_expert_stride(expert_in, inter) % super::super::VQ_ALIGN,
            0
        );
        for (i, &o) in off.iter().enumerate() {
            assert!(
                o.is_multiple_of(4),
                "packed span {i} starts at {o}, not a 4-byte boundary"
            );
        }
    }

    /// A 384-expert layer's file length is `VQ_ALIGN + 384 x 32,845,824` — 12.61 GB, which is
    /// the figure the streaming plan is built on. Asserted because it is the one number that
    /// says whether a Small layer fits an O_DIRECT read budget without recomputing geometry.
    #[test]
    fn a_small_layer_is_twelve_point_six_gigabytes() {
        let stride = mxfp8_expert_stride(4608, 2304);
        assert_eq!(384 * stride, 12_612_796_416);
        assert_eq!(super::super::VQ_ALIGN + 384 * stride, 12_612_800_512);
    }
}
