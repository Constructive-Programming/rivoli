//! The **MX-FP4** (`.f4`) routed-expert format: e2m1 nibbles under e8m0 block scales, which
//! DeepSeek-V4-Flash and Kimi-K3 already ship, so rivoli REPACKS rather than quantizes.
//!
//! **Split out of `quant.rs` on 2026-08-15, by FORMAT** — see [`super::vq`] for the measured
//! reason. Bodies and comments travelled verbatim, including [`e8m0`]'s settled `0xFF`
//! argument and the tests' `.f4`-vs-`.i4` band, both of which are long because both are
//! records of a question that was asked and answered.
//!
//! Its tests deliberately reach into [`super::int4`] and [`super::vq`]: the finding they
//! pin is that nothing STRUCTURAL separates an `.f4` block from an `.i4` one at either
//! model's dims, which can only be asserted by comparing the two formats' own functions.
//! Every public name is re-exported by `quant.rs`.
//!
//! **One source does have to be quantized INTO this container.** Glimmer, DeepSeek-V4-Flash and
//! Kimi-K3 all ship e2m1 already, so their `.f4` sets are repacks — but `Inkling 2` ships its
//! routed experts at NVFP4 (Mini) and MXFP8 (Small and Mini), so `--f4` on either member
//! converter re-encodes bf16/expert bytes through [`quantize_f4_row`]. The geometry does not
//! move: a row quantized here drops into an `.f4` expert block with no reordering, exactly as
//! [`super::mxfp8::quantize_mxfp8_row`] does for the byte-per-weight sibling.

use anyhow::{Result, ensure};

use super::{expert_bytes, expert_stride, slot_offsets};

// --- FP4 (`.f4`) — DeepSeek-V4-Flash's native routed-expert format --------------------
//
// Chosen 2026-08-03 to keep source fidelity: V4-Flash ships its 296.35B routed-expert
// params ALREADY at 4 bits (OCP MX e2m1 nibbles with e8m0 block scales), so producing a
// `.f4` artifact is a REPACK into rivoli's O_DIRECT-aligned block layout, not a
// quantization. There is no fit, no codebook, and no error introduced — which is the whole
// point: re-quantizing a 4-bit source into 3.25-bit int3-vq is the lossy-on-lossy chain
// docs/investigations/int4-scales.md records at PPL 73.43. See other-models.md §5.
//
// Unlike int4 (uniform levels, f32 group scale), e2m1 is a fixed NON-uniform 16-value
// codebook and the scale is a bare power of two, so dequant is a 16-entry lookup times a
// shift. That makes it cheaper on the GPU than either neighbour, not more expensive.

/// Weights per e8m0 scale along the input dim. 32 is the OCP MX block size, which is what
/// V4-Flash's `.scale` tensors carry (1×32 sub-blocks nested inside its 128×128 fp8
/// blocks). Distinct from `I4_GROUP`(128) and `VQ_GROUP`(64) on purpose — this one is
/// dictated by the source, not chosen by us.
pub const F4_GROUP: usize = 32;

/// The 16 e2m1 values, indexed by nibble. Bit 3 is sign, bits 2-1 exponent, bit 0
/// mantissa; there are no infinities and no NaN — every code is a finite value, which is
/// why a bare lookup is a complete decoder.
pub const F4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one e8m0 scale byte: a bare exponent, value `2^(b - 127)`. `0xFF` is the
/// format's NaN and must not appear in weights — a scale that is NaN poisons a whole
/// 32-weight block, so it is worth failing on rather than propagating.
///
/// > **SETTLED 2026-08-11 — Kimi-K3's `sb == 255` question (plan S1a item 2).** K3's C reference
/// > maps 255 to **zero** where this bails, and the plan flagged the disagreement as needing a
/// > decision because host and device must move together: `moe_fixed`'s saturating clamp would
/// > launder a device-side NaN into a finite ±2^14, so a divergence here is silent.
/// >
/// > **The bail stays**, on two grounds. First, measurement: 4,128,768 real K3 scale bytes (every
/// > scale tensor of experts 0-3, layer 1) hold **11 distinct codes in `0x70..=0x7a`, zero `0xFF`
/// > and zero `0x00`** — the same shape V4's shipped set showed, so the reference's 255 path is
/// > defensive rather than exercised. That is a 0.005% sample and settles nothing by itself.
/// > Second, and this is what settles it: **the repack is the only path that reads every ROUTED
/// > scale byte.** At decode they DMA from NVMe straight into a pool slot and the host never sees
/// > them, so `F4Expert::spans`'s check either passes over the whole checkpoint at conversion time
/// > or names the exact tensor, row and group that fails. `convert_k3` inherits that check through
/// > `RoutedRepack`, so no `.f4` artifact can contain a byte this function would reject.
/// >
/// > **ROUTED is load-bearing, because there is a second exhaustive host reader.**
/// > `SafeWriter::copy_fp8_e8m0` maps this function over every byte of every fp8 tensor's `.scale`
/// > grid. That does not weaken the conclusion — it is also conversion-time and also fails loudly —
/// > but it is weaker evidence, because it names the tensor and not the row or the group.
/// >
/// > **And there are THREE decoders with three behaviours, not two.** This one bails,
/// > the kernels' `e8m0f` (`crates/kernels/src/prelude/dot_nib.rs`) returns a quiet NaN, and
/// > `v4oracle::numerics::e8m0_decode` returns `f32::NAN` outright. A future decision to adopt 255
/// > → zero has three sites to move.
/// >
/// > Mapping 255 to zero instead would be adopting a rule for values the format forbids and this
/// > engine's own artifacts cannot contain — and it would have to be adopted in the kernels'
/// > `e8m0f` too, where nothing can report it. `docs/measurement/k3-reference/repack-one-expert.md`
/// > carries the distribution.
pub fn e8m0(b: u8) -> anyhow::Result<f32> {
    if b == 0xFF {
        anyhow::bail!("e8m0 scale byte 0xFF is NaN");
    }
    // exp2 rather than `bits << 23`: the bit form is exact only for b >= 1, and b = 0 is
    // 2^-127, an f32 SUBNORMAL that the shift encodes as +0.
    Ok(f32::exp2(b as f32 - 127.0))
}

/// Number of e8m0 group scales in one FP4 row (`ceil(i_dim / F4_GROUP)`).
pub fn f4_groups(i_dim: usize) -> usize {
    i_dim.div_ceil(F4_GROUP)
}

/// Row stride in bytes for an FP4 matrix (2 nibbles/byte) — same packing as int4, so the
/// `.i4` container's nibble addressing carries over unchanged.
pub fn f4_row_bytes(i_dim: usize) -> usize {
    i_dim.div_ceil(2)
}

/// On-disk bytes of one FP4-packed projection `W[o_dim, i_dim]`: packed nibbles, then one
/// e8m0 byte per group. Mirrors `vq_proj_bytes`/`i4_proj_bytes` so the block layout and
/// `VQ_ALIGN` stride math are shared across all three formats.
pub fn f4_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * f4_row_bytes(i_dim) + o_dim * f4_groups(i_dim)
}

/// Unpadded on-disk bytes of one FP4 expert (w1‖w1_scale‖w3‖w3_scale‖w2‖w2_scale — see
/// [`super::naming::V4_PROJ`] for why that is gate/up/down order).
pub fn f4_expert_bytes(expert_in: usize, moe_inter: usize) -> usize {
    expert_bytes(expert_in, moe_inter, f4_proj_bytes)
}

/// Per-expert on-disk stride: [`f4_expert_bytes`] padded up to [`super::VQ_ALIGN`], so one FP4
/// expert is a single block-aligned read (mirrors the `.vq3`/`.i4` stride).
pub fn f4_expert_stride(expert_in: usize, moe_inter: usize) -> usize {
    expert_stride(f4_expert_bytes(expert_in, moe_inter))
}

/// The six byte offsets within one FP4 expert block, in `ExpertDescF4` field order:
/// `[w1, w1_scale, w3, w3_scale, w2, w2_scale]` — see [`super::naming::V4_PROJ`] for why that is
/// gate/up/down order.
///
/// The scale span is ONE BYTE per group, not two (`.vq3` bf16) or four (`.i4` f32): e8m0 is
/// a bare exponent. That is the whole difference from its two siblings, and it is why
/// `backend::ExpertDescF4` carries `*const u8` scale pointers where `ExpertDesc` carries
/// `*const u16` — a `.f4` block resolved through the int4 offsets would put every
/// projection but the first at the wrong address, and the ones it did find it would decode
/// at group 128 against a uniform codebook.
pub fn f4_slot_offsets(expert_in: usize, moe_inter: usize) -> [usize; 6] {
    slot_offsets(expert_in, moe_inter, |o, i| {
        (o * f4_row_bytes(i), o * f4_groups(i))
    })
}

/// The largest e2m1 magnitude — codes 7 and 15, [`F4_LUT`]'s two ends. A block's whole dynamic
/// range lives in its e8m0 scale, so this is the value every scale is chosen against: the
/// smallest power of two that keeps the group's `amax` inside `±F4_MAX` spends the group's range
/// on the codebook's three mantissa bits instead of clipping it.
const F4_MAX: f32 = 6.0;

/// The ue8m0 exponent of 1.0 — `2^(127 − 127)`. What an all-zero group is written with, because
/// every exponent decodes an all-zero group to zero and 127 is the byte a decoder reproduces
/// without the subnormal path (`2^-127` at `s = 0` is representable but needlessly delicate).
/// [`super::mxfp8`] chose the same byte for the same reason.
const F4_UNIT_SCALE: u8 = 127;

/// Quantize ONE f32 row into MXFP4: e2m1 nibbles into `packed` and one ue8m0 exponent byte per
/// [`F4_GROUP`] weights into `scales`.
///
/// The packing is the container's, not a choice here: `packed` is [`f4_row_bytes`] bytes with
/// the LOW nibble holding the EVEN element and the HIGH nibble the odd one — the order
/// `crates/kernels/src/prelude/dot_nib.rs` decodes and `F4Expert`'s spans carry, so a row
/// quantized here drops straight into an `.f4` expert block.
///
/// **Lossy, and it is the only lossy `.f4` path in this tree.** Every other source ships e2m1
/// already (module doc) and rides the bit-exact repack; Inkling 2's NVFP4/MXFP8 experts do not,
/// so this function introduces the error and `--f4`'s paired-dNLL gate is what scores it.
///
/// **A non-finite weight is refused, not encoded.** e2m1 has no Inf and no NaN, so a NaN would
/// land on some legal code and an Inf would saturate to `±6` — 32 weights of plausible-looking
/// garbage with nothing on disk to report it. [`super::mxfp8::quantize_mxfp8_row`] refuses the
/// same inputs for the same reason.
pub fn quantize_f4_row(row: &[f32], packed: &mut [u8], scales: &mut [u8]) -> Result<()> {
    ensure!(
        row.len().is_multiple_of(F4_GROUP),
        "MXFP4 row: {} weights is not a multiple of the {F4_GROUP}-weight group",
        row.len()
    );
    ensure!(
        packed.len() == f4_row_bytes(row.len()) && scales.len() == f4_groups(row.len()),
        "MXFP4 row: {} packed / {} scale bytes for a {}-weight row",
        packed.len(),
        scales.len(),
        row.len()
    );
    for ((group, out), scale) in row
        .chunks_exact(F4_GROUP)
        .zip(packed.chunks_exact_mut(F4_GROUP / 2))
        .zip(scales.iter_mut())
    {
        *scale = encode_f4_group(group, out)?;
    }
    Ok(())
}

/// One 32-weight group: its e8m0 exponent, and the 16 nibble-pair bytes.
fn encode_f4_group(group: &[f32], out: &mut [u8]) -> Result<u8> {
    let amax = group.iter().try_fold(0.0f32, |m, &v| -> Result<f32> {
        ensure!(
            v.is_finite(),
            "MXFP4: non-finite weight {v} in a group of {F4_GROUP} — e2m1 has no Inf and no NaN, \
             so an encode would write a plausible code and leave no trace"
        );
        Ok(m.max(v.abs()))
    })?;
    if amax == 0.0 {
        // Every code of an all-zero group is zero; zeroing the bytes says so without relying
        // on the decoder to treat `-0.0`'s sign bit as insignificant.
        out.fill(0);
        return Ok(F4_UNIT_SCALE);
    }
    let code = f4_scale_byte(amax);
    let scale = f32::exp2(f32::from(code) - 127.0);
    for (j, pair) in group.chunks_exact(2).enumerate() {
        out[j] = f32_to_e2m1(pair[0] / scale) | (f32_to_e2m1(pair[1] / scale) << 4);
    }
    Ok(code)
}

/// The ue8m0 exponent for a non-zero group maximum: `ceil(log2(amax / 6)) + 127`, clamped to the
/// encodable `0..=254` (`255` is the format's NaN; [`e8m0`] refuses it, and `.f4` artifacts
/// cannot contain it because no producer here writes it).
///
/// The clamp cannot clip. `s < 0` needs `amax / 6 < 2^-127`, where the clamped `2^-127` still
/// maps every weight of the group below 6; `s == 254` needs `amax > 6·2^127 ≈ 1.02e39`, past
/// `f32::MAX`'s 3.4e38 — so the top byte is unreachable from an f32 and the clamp is
/// belt-and-braces rather than a behaviour. The one step back up covers a `log2`/`ceil` rounding
/// that lands a hair low, which would otherwise clip the group's largest weight to 6.0 silently.
fn f4_scale_byte(amax: f32) -> u8 {
    let mut code = ((amax / F4_MAX).log2().ceil() as i32 + 127).clamp(0, 254) as u8;
    let overshoots = amax / f32::exp2(f32::from(code) - 127.0) > F4_MAX;
    if overshoots && code < 254 {
        code += 1;
    }
    code
}

/// The e2m1 code of one scaled weight: sign in bit 3, magnitude by round-to-nearest-EVEN over
/// [`F4_LUT`]'s eight positive codes.
///
/// **The ties go to the even CODE, not to the even VALUE.** Code `c`'s LSB is 0 for the
/// magnitudes `0.0, 1.0, 2.0, 4.0` (codes 0, 2, 4, 6) and 1 for `0.5, 1.5, 3.0, 6.0`
/// (codes 1, 3, 5, 7), so the seven midpoints break as 0.25→0, 0.75→2, 1.25→2, 1.75→4,
/// 2.5→4, 3.5→6, 5.0→6. That is the same rule
/// `f32_to_e4m3` applies to the byte-per-weight sibling, which is what keeps the two formats'
/// error curves comparable at equal scales.
fn f32_to_e2m1(x: f32) -> u8 {
    const CUTS: [(f32, bool); 7] = [
        (0.25, true),
        (0.75, false),
        (1.25, true),
        (1.75, false),
        (2.5, true),
        (3.5, false),
        (5.0, true),
    ];
    let sign = u8::from(x.is_sign_negative()) << 3;
    let a = x.abs();
    let code = CUTS
        .iter()
        .position(|&(cut, inclusive)| if inclusive { a <= cut } else { a < cut })
        .unwrap_or(7) as u8;
    sign | code
}

#[cfg(test)]
mod tests {
    // The `.f4` geometry is pinned against a shipped file length measured on disk, which is
    // the one number none of these functions computes. Crate-wide `unwrap`/`expect` are
    // `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    // Named through the SIBLING modules rather than through `quant`'s re-exports: what
    // these tests pin is that `.f4` and `.i4` are indistinguishable at both models' dims,
    // and that claim is about two formats' own functions. Reading them at the flat path
    // would hide which format each one belongs to, which is the entire subject here.
    use super::*;
    use crate::quant::int4::{i4_expert_bytes, i4_slot_offsets};
    use crate::quant::vq::{vq_expert_bytes, vq_proj_bytes, vq_slot_offsets};

    /// **The `.f4` slot layout, pinned against the SHIPPED artifact's own geometry — and
    /// it is byte-identical to `.i4`'s, which the name says because the first draft of this
    /// test asserted the opposite and failed.**
    ///
    /// Not a restatement of `slot_offsets`: the six numbers are written out, and they are
    /// checked against a file length nothing here computes — `L00.f4` is
    /// `4096 + 256 × 13369344 = 3422556160` bytes, verified on disk. A layout that agreed
    /// with itself but not with the converter would move at least one of them.
    ///
    /// **And it asserts that `.f4` and `.i4` are byte-identical here, which is not what was
    /// expected and is the more useful half.** Both pack nibbles at `i_dim/2` bytes a row;
    /// `.f4` spends `ceil(i/32) × 1` byte on scales and `.i4` spends `ceil(i/128) × 4`. Those
    /// collide exactly when
    ///
    /// ```text
    /// ceil(i/32) == 4 · ceil(i/128)      i.e.  i mod 128 ∈ {0} ∪ {97..127}
    /// ```
    ///
    /// **which is 32 of every 128 dimensions — 25%, a BAND and not a special case.** Stating
    /// it as "`i_dim` is a multiple of 128" (as the first version of this doc did) is
    /// sufficient but badly incomplete, and misleading in a specific way: a reader who changes
    /// a dimension to 96 finds the layouts separate and may conclude the collision was fixed.
    /// It was not — 100 collides, 96 and 160 do not.
    ///
    /// **And the two things it governs are not the same thing.** The six OFFSETS collide iff
    /// `band(expert_in)`; the block SIZE collides iff `band(expert_in) && band(moe_inter)`.
    /// `moe_inter` reaches the scale grid only through w2, whose scale span begins AT `off[5]`
    /// and so appears in no offset — it changes `*_expert_bytes` and nothing the offset array
    /// can see. Both corrections are from 2026-08-05: the band from the coordinator
    /// reproducing the arithmetic, the `expert_in`-only part from the assertion below failing on
    /// `(4096, 96)` when it was written expecting symmetry.
    ///
    /// Both models are in the band on both dims (GLM 6144/2048, V4 4096/2048), so at V4's
    /// dims the two formats agree on all six offsets, on `*_expert_bytes`, and therefore on
    /// the whole file length.
    ///
    /// The consequence is worth stating where someone will read it: **nothing structural can
    /// separate an `.f4` from an `.i4`.** Not the length, not the slot offsets, not a
    /// descriptor's six addresses — an `.f4` block resolved through `i4_slot_offsets` finds
    /// every projection at exactly the right address and then decodes e2m1 nibbles as
    /// `n − 8` against a group-128 f32 scale read out of e8m0 bytes. The separation is the
    /// header magic (`ExpertHeader::from_bytes`, and `format.rs`'s
    /// `magic_separates_the_formats_when_the_length_cannot` is named for this) and the
    /// descriptor TYPE (`backend::ExpertDescF4` vs `ExpertDesc`). That is why
    /// `memory::routed::TierFmt` carries a `RoutedFmt` and not the `int4: bool` it replaced.
    ///
    /// They do diverge outside that band — the toy `(64, 32)` fixtures in
    /// `tests/f4_loading.rs` are such a case (32 and 64 are both `< 97 mod 128`) — so a check
    /// that compares them is not vacuous everywhere, only where it matters.
    #[test]
    fn f4_slot_offsets_match_the_shipped_block_and_are_indistinguishable_from_i4() {
        // DeepSeek-V4-Flash: expert_in 4096, moe_intermediate_size 2048.
        let (expert_in, inter) = (4096usize, 2048usize);
        let off = f4_slot_offsets(expert_in, inter);
        assert_eq!(
            off,
            [0, 4_194_304, 4_456_448, 8_650_752, 8_912_896, 13_107_200],
            "the .f4 slot layout moved"
        );
        // The last span (w2's scales) ends exactly at the block size, and the block size is
        // the shipped file's own stride.
        assert_eq!(off[5] + expert_in * f4_groups(inter), 13_369_344);
        assert_eq!(f4_expert_bytes(expert_in, inter), 13_369_344);
        assert_eq!(
            f4_expert_stride(expert_in, inter),
            13_369_344,
            "no padding at these dims"
        );
        assert_eq!(
            4096 + 256 * f4_expert_stride(expert_in, inter),
            3_422_556_160
        );

        // Every packed span 4-byte aligned, so `dot_f4_wave_r`'s dword fast path is taken
        // rather than falling back to its scalar tail. A PERFORMANCE property, not a
        // correctness one — the kernel predicates on the alignment and handles both.
        for k in [0, 2, 4] {
            assert_eq!(
                off[k] % 4,
                0,
                "packed span {} is not 4-byte aligned",
                off[k]
            );
        }

        // The coincidence, pinned so it is a known fact rather than a surprise at a call
        // site: at V4's dims `.f4` and `.i4` are the same layout AND the same size.
        assert_eq!(
            off,
            i4_slot_offsets(expert_in, inter),
            "at i_dim % 128 == 0 the two nibble formats tile identically — if this ever \
             stops being true, the claim in this test's doc has to change with it"
        );
        assert_eq!(
            f4_expert_bytes(expert_in, inter),
            i4_expert_bytes(expert_in, inter)
        );
        // `.vq3` is genuinely a different size (12-bit indices over VQ_DIM=4), so it is the
        // one format a length check does separate.
        assert_ne!(off, vq_slot_offsets(expert_in, inter));
        assert_ne!(
            f4_expert_bytes(expert_in, inter),
            vq_expert_bytes(expert_in, inter)
        );
        assert_the_band_is_where_f4_and_i4_collide(expert_in, inter);
    }

    /// **The six OFFSETS turn on `expert_in` ALONE — `moe_inter` cannot separate them.**
    /// Found by these assertions failing on `(4096, 96)`, which were written expecting the
    /// band to apply symmetrically. It does not: `off[2]` and `off[4]` are sums of w1's and
    /// w3's spans, whose `i_dim` is `expert_in`; `off[5]` adds w2's PACKED bytes, which are
    /// `i/2` in both formats. w2's scale length — the only place `moe_inter` reaches the
    /// scale grid — is past `off[5]` and appears in no offset at all. So `moe_inter` changes
    /// `*_expert_bytes` and nothing this array can see.
    ///
    /// The offsets therefore collide iff `band(expert_in)`, and the block SIZE collides iff
    /// `band(expert_in) && band(moe_inter)`. Both models are in the band on both dims
    /// (GLM 6144/2048, V4 4096/2048), so both collide completely.
    fn assert_the_band_is_where_f4_and_i4_collide(expert_in: usize, inter: usize) {
        let band = |i: usize| i.div_ceil(32) == 4 * i.div_ceil(128);
        assert!(
            band(expert_in) && band(inter),
            "both of V4's dims are in the band"
        );
        for h in [100usize, 128, 4096, 6144] {
            assert!(band(h));
            assert_eq!(
                f4_slot_offsets(h, inter),
                i4_slot_offsets(h, inter),
                "expert_in {h} is in the band, so the layouts must be indistinguishable"
            );
        }
        for h in [96usize, 160, 64] {
            assert!(!band(h));
            assert_ne!(
                f4_slot_offsets(h, inter),
                i4_slot_offsets(h, inter),
                "expert_in {h} is outside the band — if this collides, the band moved"
            );
        }
        // `moe_inter` out of band moves the block SIZE and leaves every offset alone. This is
        // the pair that made the point, so it is the pair that is pinned.
        assert_eq!(
            f4_slot_offsets(expert_in, 96),
            i4_slot_offsets(expert_in, 96)
        );
        assert_ne!(
            f4_expert_bytes(expert_in, 96),
            i4_expert_bytes(expert_in, 96)
        );
        // `tests/f4_loading.rs`'s toy fixtures are out of band on BOTH, which is the only
        // reason a test can SEE an `.f4` set resolved through `.i4`'s layout at all. There is
        // no runtime check for it — that is the finding above, not an omission.
        assert_ne!(f4_slot_offsets(64, 32), i4_slot_offsets(64, 32));
    }

    /// The e2m1 table decoded from its bit fields rather than restated, so this is a
    /// check and not a copy of the constant. Bit 3 sign, bits 2-1 exponent, bit 0
    /// mantissa; exponent 0 is the subnormal case (value = mantissa/2), every other
    /// exponent e is `2^(e-1) * (1 + mantissa/2)`. There is no Inf and no NaN, which is
    /// what makes a bare 16-entry lookup a COMPLETE decoder.
    #[test]
    fn f4_lut_matches_the_e2m1_bit_fields() {
        for code in 0u8..16 {
            let (sign, exp, man) = (code >> 3, (code >> 1) & 3, code & 1);
            let mag = if exp == 0 {
                man as f32 * 0.5
            } else {
                f32::exp2(exp as f32 - 1.0) * (1.0 + man as f32 * 0.5)
            };
            let want = if sign == 1 { -mag } else { mag };
            assert_eq!(
                F4_LUT[code as usize], want,
                "e2m1 code {code:#06b} decodes to {want}, table says {}",
                F4_LUT[code as usize]
            );
        }
        // The largest magnitude is 6.0, so a block's dynamic range is entirely in its
        // e8m0 scale — a fact the repack relies on to carry values through untouched.
        assert_eq!(F4_LUT.iter().cloned().fold(0.0f32, f32::max), 6.0);
    }

    /// e8m0 is a bare exponent, `2^(b-127)`, and both ends matter: b=0 lands on an f32
    /// subnormal (the `bits << 23` form silently returns +0 there) and 0xFF is the
    /// format's NaN, which would poison a whole 32-weight block.
    #[test]
    fn e8m0_covers_both_ends_and_rejects_nan() {
        assert_eq!(e8m0(127).unwrap(), 1.0);
        assert_eq!(e8m0(128).unwrap(), 2.0);
        assert_eq!(e8m0(126).unwrap(), 0.5);
        assert_eq!(e8m0(254).unwrap(), f32::exp2(127.0));
        let lo = e8m0(0).unwrap();
        assert!(lo > 0.0 && lo.is_finite(), "b=0 must be 2^-127, got {lo}");
        assert_eq!(lo, f32::exp2(-127.0));
        assert!(e8m0(0xFF).is_err(), "0xFF is NaN and must be refused");
    }

    /// `.f4` shares `.i4`'s nibble packing but carries a one-byte scale per 32 weights
    /// instead of an f32 per 128, so its rows are wider. GLM's widths stand in for the
    /// arithmetic; V4-Flash's own are expert_in 4096 / moe_inter 2048.
    #[test]
    fn f4_layout_sizes() {
        assert_eq!(f4_row_bytes(2048), 1024);
        assert_eq!(f4_groups(2048), 64);
        assert_eq!(f4_groups(2049), 65, "a ragged tail gets its own scale");
        // 4.25 bits/weight: 4 for the nibble, 8/32 for the scale.
        let (o, i) = (4096usize, 2048usize);
        assert_eq!(f4_proj_bytes(o, i), o * 1024 + o * 64);
        assert_eq!(f4_proj_bytes(o, i) as f64 * 8.0 / (o * i) as f64, 4.25);
        // Strictly larger than int3-vq's 3.25, and that is the trade we chose.
        assert!(f4_proj_bytes(o, i) > vq_proj_bytes(o, i));
    }

    /// Quantize one whole group through the public entry point and hand back its bytes.
    fn quantize_one_group(vals: &[f32]) -> (Vec<u8>, u8) {
        let mut packed = vec![0u8; f4_row_bytes(vals.len())];
        let mut scales = vec![0u8; f4_groups(vals.len())];
        quantize_f4_row(vals, &mut packed, &mut scales).unwrap();
        (packed, scales[0])
    }

    /// **Every one of the 16 codes round-trips EXACTLY, at five scale powers.** The row is the
    /// table twice (one group), so its amax is 6·2^k and the scale is exactly 2^k — which means
    /// `v / scale` is the table value with no rounding anywhere, and the expected byte is a
    /// function of the pair index alone. A sign, a nibble order or a scale-exponent error each
    /// move at least one of these bytes.
    #[test]
    fn quantize_row_round_trips_every_code_across_scale_powers() {
        for k in [-12i32, -1, 0, 1, 12] {
            let mult = f32::exp2(k as f32);
            let row: Vec<f32> = F4_LUT
                .iter()
                .chain(F4_LUT.iter())
                .map(|v| v * mult)
                .collect();
            let (packed, scale) = quantize_one_group(&row);
            assert_eq!(
                scale,
                (127 + k) as u8,
                "amax 6·2^{k} did not take exponent 127+{k}"
            );
            for (pair, &got) in packed.iter().enumerate() {
                // Both copies of the table give the same code, so the expected nibbles cycle
                // through 0..16 rather than running to 31.
                let (lo, hi) = (2 * (pair % 8), 2 * (pair % 8) + 1);
                assert_eq!(
                    got,
                    lo as u8 | ((hi as u8) << 4),
                    "at scale 2^{k}, pair {pair} is {got:#04x}"
                );
            }
        }
    }

    /// The nibble ORDER: the LOW nibble is the even element. Written as a swap, because a
    /// symmetric row would pass under either convention. The third element pins the group's
    /// amax at 6.0 so the scale is the unit exponent and the codes are the table's own.
    #[test]
    fn quantize_row_puts_the_even_element_in_the_low_nibble() {
        let mut row = vec![0.0f32; F4_GROUP];
        row[0] = 0.5;
        row[1] = 3.0;
        row[2] = 6.0;
        let (packed, scale) = quantize_one_group(&row);
        assert_eq!(scale, 127, "a 6.0 amax must take the unit exponent");
        assert_eq!(packed[0], 0x51, "0.5 (code 1) then 3.0 (code 5)");
        row.swap(0, 1);
        let (packed, _) = quantize_one_group(&row);
        assert_eq!(packed[0], 0x15, "3.0 (code 5) then 0.5 (code 1)");
    }

    /// An all-zero group is written with the unit exponent and zero bytes — not with a NaN
    /// scale and not with `-0.0`'s sign bit, which a `is_sign_negative` dispatch would emit for
    /// an exact `-0.0` weight in a mixed group but must not need to here.
    #[test]
    fn quantize_row_writes_the_unit_scale_for_an_all_zero_group() {
        let (packed, scale) = quantize_one_group(&[0.0f32; F4_GROUP]);
        assert_eq!(scale, F4_UNIT_SCALE);
        assert_eq!(packed, vec![0u8; F4_GROUP / 2]);
    }

    /// A non-finite weight is REFUSED rather than encoded (e2m1 has no Inf and no NaN, so an
    /// encode would be a plausible code), and every length disagreement is refused: a ragged
    /// row, a packed buffer of the wrong extent, and a missing scale grid.
    #[test]
    fn quantize_row_refuses_non_finite_and_ragged_lengths() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut row = vec![0.5f32; 2 * F4_GROUP];
            row[F4_GROUP + 3] = bad;
            let mut packed = vec![0u8; f4_row_bytes(row.len())];
            let mut scales = vec![0u8; f4_groups(row.len())];
            assert!(
                quantize_f4_row(&row, &mut packed, &mut scales).is_err(),
                "a {bad} weight was encoded"
            );
        }
        let mut packed = vec![0u8; f4_row_bytes(F4_GROUP + 1)];
        let mut scales = vec![0u8; f4_groups(F4_GROUP + 1)];
        assert!(
            quantize_f4_row(&[0.5f32; F4_GROUP + 1], &mut packed, &mut scales).is_err(),
            "a row that is not a whole number of groups was accepted"
        );
        let row = [0.5f32; F4_GROUP];
        let mut short = vec![0u8; F4_GROUP / 2 - 1];
        let mut scales = vec![0u8; 1];
        let mut full = vec![0u8; F4_GROUP / 2];
        assert!(quantize_f4_row(&row, &mut short, &mut scales).is_err());
        assert!(quantize_f4_row(&row, &mut full, &mut []).is_err());
    }

    /// The seven midpoints of the codebook, each beside the value one binade-step either side,
    /// so a change of tie rule (to-even VALUE, or round-half-up) reddens at least one row.
    #[test]
    fn f32_to_e2m1_rounds_ties_to_the_even_code() {
        for (x, want) in [
            (0.0, 0u8),
            (0.25, 0),
            (0.26, 1),
            (0.74, 1),
            (0.75, 2),
            (1.0, 2),
            (1.25, 2),
            (1.26, 3),
            (1.74, 3),
            (1.75, 4),
            (2.0, 4),
            (2.5, 4),
            (2.51, 5),
            (3.0, 5),
            (3.49, 5),
            (3.5, 6),
            (4.0, 6),
            (5.0, 6),
            (5.01, 7),
            (6.0, 7),
            (100.0, 7),
        ] {
            assert_eq!(f32_to_e2m1(x), want, "{x} took the wrong code");
        }
        // The sign bit is bit 3 and is independent of the magnitude code.
        for code in 0..8u8 {
            let v = F4_LUT[code as usize];
            assert_eq!(f32_to_e2m1(-v), code | 8, "-{v} lost its sign");
        }
    }

    /// The scale byte is `ceil(log2(amax / 6)) + 127`, clamped into `0..=253` for every f32.
    /// `254` is unreachable from an f32 (`amax > 6·2^127 ≈ 1.02e39` exceeds `f32::MAX`'s
    /// 3.4e38), so the clamp is belt-and-braces and the assertion is on what an f32 can produce.
    #[test]
    fn the_scale_byte_is_ceil_log2_against_the_top_code() {
        for (amax, want) in [
            (6.0, 127),
            (3.0, 126),
            (6.5, 128),
            (1e-40, 0),
            (f32::MAX, 253),
        ] {
            assert_eq!(f4_scale_byte(amax), want, "amax={amax}");
        }
    }
}
