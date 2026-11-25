//! The **MX-FP8** (`.mxfp8`) routed-expert format: one e4m3 byte per weight under ue8m0
//! block scales, which `Inkling-2-Small-MXFP8` ships for all 48 of its layers, so rivoli
//! REPACKS rather than quantizes — exactly as [`super::f4`] does for the 4-bit sibling.
//!
//! **One source does have to be quantized INTO this container.** `Inkling 2 Mini` ships 20 of
//! its 24 layers at NVFP4 (e2m1 codes, e4m3 block-16 scales, a per-expert factor) and only the
//! last four already at MXFP8, so `convert_inkling --mxfp8` re-encodes those 20 through
//! [`quantize_mxfp8_row`] while the four native layers still travel the bit-exact repack path.
//! Row-wise `[1, 32]` is the only grid a quantizer here may produce: it is the one the
//! container's offsets and the kernels' `[1, 32]` decode both assume.
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

use anyhow::{Result, ensure};
use rivoli_core::num::{E4M3_MAX, f32_to_e4m3};

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

/// The ue8m0 exponent of 1.0 — `2^(127 − 127)`. What an all-zero group is written with.
const MXFP8_UNIT_SCALE: u8 = 127;

/// Quantize ONE f32 row to row-wise MXFP8: the weights into `weights` (one e4m3 byte each) and
/// the group exponents into `scales` (one ue8m0 byte per [`MXFP8_GROUP`] weights).
///
/// The layout produced is exactly the one [`mxfp8_slot_offsets`] points a descriptor at — `i`
/// weight bytes then `ceil(i/32)` scale bytes, per row — so a row quantized here drops into an
/// `.mxfp8` expert block with no reordering. `Inkling 2 Mini`'s NVFP4 layers reach the container
/// this way (module doc); nothing in the repack path calls it.
///
/// Each group takes the smallest power of two that keeps its largest weight inside e4m3's
/// `±448`, which spends the group's range on three mantissa bits rather than clipping it.
/// An all-zero group takes `s = 127` (`2^0`) and emits zeros: every exponent decodes an all-zero
/// group to zero, so 127 is chosen because it is the byte a decoder reproduces without the
/// subnormal path (`2^-127` at `s = 0` is representable but needlessly delicate).
///
/// **A non-finite weight is refused, not encoded.** `f32_to_e4m3` maps NaN to `0x7f` (e4m3's
/// NaN, which poisons the whole 32-weight dot) and `±inf` to a saturating `±448` (32 weights of
/// plausible garbage), and neither leaves a trace the loader could report — the same argument
/// [`super::quantize_fp8_block`] makes for its tiles.
pub fn quantize_mxfp8_row(row: &[f32], weights: &mut [u8], scales: &mut [u8]) -> Result<()> {
    let groups = mxfp8_groups(row.len());
    ensure!(
        row.len().is_multiple_of(MXFP8_GROUP),
        "MXFP8 row: {} weights is not a multiple of the {MXFP8_GROUP}-weight group",
        row.len()
    );
    ensure!(
        weights.len() == row.len() && scales.len() == groups,
        "MXFP8 row: {} weight bytes / {} scale bytes for a {}-weight row ({groups} groups)",
        weights.len(),
        scales.len(),
        row.len()
    );
    for (g, (group, out)) in row
        .chunks_exact(MXFP8_GROUP)
        .zip(weights.chunks_exact_mut(MXFP8_GROUP))
        .enumerate()
    {
        scales[g] = encode_mxfp8_group(group, out)?;
    }
    Ok(())
}

/// One 32-weight group: the ue8m0 exponent for `out`'s scale byte, and the encoded values.
fn encode_mxfp8_group(group: &[f32], out: &mut [u8]) -> Result<u8> {
    let mut amax = 0.0f32;
    for &v in group {
        ensure!(
            v.is_finite(),
            "MXFP8: non-finite weight {v} in a group of {MXFP8_GROUP} — an e4m3 encode would \
             turn it into a NaN byte or a silent ±{E4M3_MAX} saturation"
        );
        amax = amax.max(v.abs());
    }
    if amax == 0.0 {
        out.fill(0);
        return Ok(MXFP8_UNIT_SCALE);
    }
    let s = mxfp8_scale_byte(amax);
    let scale = f32::exp2(f32::from(s) - 127.0);
    for (b, &v) in out.iter_mut().zip(group) {
        *b = f32_to_e4m3(v / scale);
    }
    Ok(s)
}

/// The ue8m0 exponent for a non-zero group maximum: `ceil(log2(amax / 448)) + 127`, clamped to
/// the encodable `0..=254` (`255` is the format's NaN, [`MXFP8_NAN_SCALE`]).
///
/// The clamp cannot clip. An unclamped `s < 0` needs `amax / 448 < 2^-127`, where the clamped
/// scale `2^-127` still maps every weight of the group below 448; and `s > 254` needs
/// `amax / 448 > 2^127`, a weight 121 binades past e4m3's range. The one step back up below
/// covers a `log2`/`ceil` rounding that lands a hair low: without it `amax / scale` could
/// exceed 448 and `f32_to_e4m3` would SATURATE the group's largest weight, silently.
fn mxfp8_scale_byte(amax: f32) -> u8 {
    let e = (amax / E4M3_MAX).log2().ceil();
    let mut s = (e as i32 + 127).clamp(0, 254) as u8;
    if s < 254 && amax / f32::exp2(f32::from(s) - 127.0) > E4M3_MAX {
        s += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    // The subject is arithmetic over config dims, so nothing here touches a file. Crate-wide
    // `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use rivoli_core::num::e4m3_to_f32;

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

    /// Four groups: well-scaled positives, an all-zero one, one whose maximum is ~250x the
    /// smallest, and one six binades down — the three shapes a scale choice can get wrong.
    fn four_group_row() -> Vec<f32> {
        let group0 = (0..MXFP8_GROUP).map(|k| 0.5 + 0.1 * k as f32);
        let group2 = (0..MXFP8_GROUP).map(|k| 1000.0 - 30.0 * k as f32);
        let group3 = (0..MXFP8_GROUP).map(|k| 1e-6 * (1.0 + 0.2 * k as f32));
        let mut row: Vec<f32> = group0.collect();
        row.extend(std::iter::repeat_n(0.0, MXFP8_GROUP));
        row.extend(group2);
        row.extend(group3);
        row
    }

    /// **The quantizer's three behaviours, on one 4-group row.** A group of well-scaled values
    /// round-trips inside e4m3's OWN error (three mantissa bits, so a half-ulp is 2^-4 = 6.25%
    /// relative); an all-zero group takes the unit exponent and zero bytes; and no scale byte is
    /// `0xFF`. The tolerance is the format's, not the row's — a tighter one would be measuring
    /// `f32_to_e4m3`'s rounding rather than the scale choice this function owns.
    #[test]
    fn quantize_row_round_trips_labels_an_all_zero_group_and_never_writes_a_nan_scale() {
        let row = four_group_row();
        let mut weights = vec![0u8; row.len()];
        let mut scales = vec![0u8; mxfp8_groups(row.len())];
        quantize_mxfp8_row(&row, &mut weights, &mut scales).unwrap();
        assert_eq!(scales.len(), 4);
        assert_eq!(
            scales[1], MXFP8_UNIT_SCALE,
            "an all-zero group did not take s = 127"
        );
        for (g, &s) in scales.iter().enumerate() {
            assert_ne!(s, MXFP8_NAN_SCALE, "group {g} was written as the ue8m0 NaN");
        }
        for (k, &b) in weights[MXFP8_GROUP..2 * MXFP8_GROUP].iter().enumerate() {
            assert_eq!(b, 0, "all-zero group byte {k} is {b:#04x}, not zeros");
        }
        for (i, &want) in row.iter().enumerate() {
            let s = f32::exp2(f32::from(scales[i / MXFP8_GROUP]) - 127.0);
            let got = e4m3_to_f32(weights[i]) * s;
            assert!(
                (got - want).abs() <= 0.0625 * want.abs(),
                "weight {i}: {want} → {got}"
            );
        }
    }

    /// A non-finite weight is REFUSED rather than encoded, and a ragged row or a scale grid of
    /// the wrong length is refused by the geometry check. `f32_to_e4m3` answers the first with
    /// a NaN byte or a silent saturation, which decodes to plausible garbage.
    #[test]
    fn quantize_row_refuses_non_finite_and_ragged_lengths() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut row = vec![0.5f32; 2 * MXFP8_GROUP];
            row[MXFP8_GROUP + 3] = bad;
            let mut w = vec![0u8; row.len()];
            let mut s = vec![0u8; mxfp8_groups(row.len())];
            assert!(
                quantize_mxfp8_row(&row, &mut w, &mut s).is_err(),
                "a {bad} weight was encoded"
            );
        }
        let mut w = vec![0u8; MXFP8_GROUP + 1];
        let mut s = vec![0u8; 2];
        assert!(
            quantize_mxfp8_row(&[0.5f32; MXFP8_GROUP + 1], &mut w, &mut s).is_err(),
            "a row that is not a whole number of groups was accepted"
        );
        let mut w = vec![0u8; MXFP8_GROUP];
        assert!(
            quantize_mxfp8_row(&[0.5f32; MXFP8_GROUP], &mut w, &mut []).is_err(),
            "a missing scale grid was accepted"
        );
    }
}
