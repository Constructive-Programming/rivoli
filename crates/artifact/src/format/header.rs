//! The routed-expert layer file's 40-byte header, and the [`RoutedFmt`] that says which
//! container is being read.
//!
//! Between the writer ([`super::layer`]) and the reader ([`super::set`]) because both must
//! agree on it and neither owns it: the header is written from the same [`LayerDims`] the
//! reader later confronts the file with. That pairing is the entire defence against a
//! transposed `(expert_in, moe_inter)`, which keeps the byte count and so passes every
//! length check ever written.

use anyhow::{Result, ensure};

use super::meta::FormatMeta;
use crate::quant::{
    f4_expert_bytes, f4_expert_stride, i4_expert_bytes, i4_expert_stride, vq_expert_bytes,
    vq_expert_stride,
};

// ── .vq3 / .i4 / .f4 expert files (streamed routed experts + resident shared) ────

/// Per-file header for a routed-expert layer file. Little-endian, 40 bytes, at the start
/// of the file. Self-describing: a dim/version mismatch or truncation fails loud on open.
///
/// **The dims are here because nothing else catches a transposition.** An expert's three
/// projections are `(moe_inter, expert_in) · 2 + (expert_in, moe_inter)`, so swapping the two
/// widths gives a file of EXACTLY the same size that passes every length check and streams
/// the wrong bytes. Which formats carry one, and why `.i4` does not, is
/// [`RoutedFmt::magic`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExpertHeader {
    /// [`VQ3_MAGIC`] or [`F4_MAGIC`] — which format's blocks follow.
    pub magic: [u8; 4],
    pub version: u32,
    pub layer: u32,
    /// ROUTED experts. `.vq3` holds `n_experts + 1` blocks (the last is the shared
    /// expert); `.f4` holds exactly `n_experts`, because V4's shared expert is fp8 e4m3
    /// and cannot share a file with FP4 blocks (see `quant::v4_expert_base`).
    pub n_experts: u32,
    pub expert_in: u32,
    pub moe_inter: u32,
    pub stride: u64, // per-expert O_DIRECT-aligned block stride
    pub reserved: u64,
}

pub const VQ3_MAGIC: [u8; 4] = *b"VQ3\0";
pub const F4_MAGIC: [u8; 4] = *b"FP4\0";
pub const EXPERT_HEADER_BYTES: usize = 40;

impl ExpertHeader {
    /// A header for one layer file. `stride` is the value the WRITER indexes blocks with,
    /// passed in rather than re-derived here: re-deriving would let the header disagree
    /// with the file it describes and still look self-consistent.
    pub fn new(magic: [u8; 4], d: LayerDims) -> Self {
        let (layer, n_experts) = (d.layer, d.n_experts);
        let (expert_in, moe_inter, stride) = (d.expert_in, d.moe_inter, d.stride);
        Self {
            magic,
            version: FormatMeta::VERSION,
            layer: layer as u32,
            n_experts: n_experts as u32,
            expert_in: expert_in as u32,
            moe_inter: moe_inter as u32,
            stride: stride as u64,
            reserved: 0,
        }
    }

    /// Serialize to the 40-byte on-disk header.
    pub fn to_bytes(&self) -> [u8; EXPERT_HEADER_BYTES] {
        let mut b = [0u8; EXPERT_HEADER_BYTES];
        b[0..4].copy_from_slice(&self.magic);
        b[4..8].copy_from_slice(&self.version.to_le_bytes());
        b[8..12].copy_from_slice(&self.layer.to_le_bytes());
        b[12..16].copy_from_slice(&self.n_experts.to_le_bytes());
        b[16..20].copy_from_slice(&self.expert_in.to_le_bytes());
        b[20..24].copy_from_slice(&self.moe_inter.to_le_bytes());
        b[24..32].copy_from_slice(&self.stride.to_le_bytes());
        b[32..40].copy_from_slice(&self.reserved.to_le_bytes());
        b
    }

    /// Parse and validate a layer file's header against the format the CALLER is reading
    /// for.
    ///
    /// The magic is a parameter rather than a constant because it is the only thing that
    /// separates a `.vq3` from an `.f4` up front, and the two differ in BLOCK COUNT as well
    /// as in content: reading one as the other addresses a whole expert stride past the end,
    /// or stops one short. It is the discriminant the length check alone cannot be.
    ///
    /// `fmt` rather than a `(magic, extension)` pair, so the two cannot be mismatched by a
    /// caller. A headerless format simply has no magic to check ([`RoutedFmt::magic`]).
    /// Takes the array, not a slice: the length check this used to carry became a
    /// compile-time truth once both callers switched to `[u8; EXPERT_HEADER_BYTES]` (the
    /// 40-byte read below, and `to_bytes()` in the tests). A guard that cannot fire is
    /// worse than none, so the type carries the length instead.
    pub(super) fn from_bytes(b: &[u8; EXPERT_HEADER_BYTES], fmt: RoutedFmt) -> Result<Self> {
        let h = Self {
            magic: b[0..4].try_into()?,
            version: u32::from_le_bytes(b[4..8].try_into()?),
            layer: u32::from_le_bytes(b[8..12].try_into()?),
            n_experts: u32::from_le_bytes(b[12..16].try_into()?),
            expert_in: u32::from_le_bytes(b[16..20].try_into()?),
            moe_inter: u32::from_le_bytes(b[20..24].try_into()?),
            stride: u64::from_le_bytes(b[24..32].try_into()?),
            reserved: u64::from_le_bytes(b[32..40].try_into()?),
        };
        if let Some(want) = fmt.magic() {
            ensure!(
                h.magic == want,
                "expected .{} magic {want:?}, file has {:?}",
                fmt.ext(),
                h.magic
            );
        }
        ensure!(
            h.version == FormatMeta::VERSION,
            ".{} version {} != {}",
            fmt.ext(),
            h.version,
            FormatMeta::VERSION
        );
        Ok(h)
    }
}

/// Which container a routed-expert set is opened as.
///
/// A `bool` (`i4: true/false`) while there were two, and the bool could not have carried
/// the property that actually separates the third: **`.f4` holds `n_experts` blocks and NO
/// shared one.** V4's shared expert is `F8_E4M3` at 128×128, not FP4 (see
/// [`crate::quant::v4_expert_base`]), so it cannot share a file with FP4 blocks —
/// it rides `resident.safetensors` instead. Verified against the shipped artifact:
/// `L00.f4` is `4096 + 256 × 13369344 = 3422556160` bytes exactly, so the `n_experts + 1`
/// this replaced demanded 13.37 MB that is not in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutedFmt {
    Vq3,
    I4,
    F4,
}

impl RoutedFmt {
    /// The per-layer file's extension. Public because it is also how a routed FORMAT
    /// names itself in a log line or an error — `RoutedPool` reports its tier with it.
    pub fn ext(self) -> &'static str {
        match self {
            RoutedFmt::Vq3 => "vq3",
            RoutedFmt::I4 => "i4",
            RoutedFmt::F4 => "f4",
        }
    }

    /// The magic the file's header must carry, or `None` for the headerless `.i4`.
    /// `.i4` gets away without one only because it is a twin of a `.vq3` that was already
    /// validated against the same dims; `.vq3` and `.f4` are standalone and carry one.
    pub(super) fn magic(self) -> Option<[u8; 4]> {
        match self {
            RoutedFmt::Vq3 => Some(VQ3_MAGIC),
            RoutedFmt::I4 => None,
            RoutedFmt::F4 => Some(F4_MAGIC),
        }
    }

    /// Byte offset of block 0. Derived from [`Self::magic`] rather than tabulated beside
    /// it: "has a header" and "reserves an aligned block for it" are the same fact.
    pub(super) fn hbytes(self) -> usize {
        match self.magic() {
            Some(_) => crate::quant::VQ_ALIGN,
            None => 0,
        }
    }

    /// Whether a shared-expert block follows the `n_experts` routed ones.
    ///
    /// **The one definition of that fact.** [`super::set::ExpertSet::open_routed`] sizes the file with it and
    /// [`super::set::ExpertSet::shared_block`] refuses without it, so a format cannot be sized for a
    /// shared block it does not have, or refuse to hand back one it does. Before `.f4` this
    /// lived only as a hard-coded `n_experts + 1` inside the length check — which is
    /// exactly the pair the old `from_bytes` error told S3 to relax together.
    pub(super) fn has_shared(self) -> bool {
        match self {
            RoutedFmt::Vq3 | RoutedFmt::I4 => true,
            RoutedFmt::F4 => false,
        }
    }

    /// The six byte offsets of this format's projections inside one expert block, in
    /// descriptor field order. Beside [`Self::geometry`] because the block SIZE and the
    /// layout INSIDE it are one fact about a format — and because **no downstream check can
    /// catch them being paired wrongly.** `.f4` and `.i4` tile a block identically for 25% of
    /// all `i_dim` (`ceil(i/32) == 4·ceil(i/128)`, i.e. `i mod 128 ∈ {0} ∪ {97..127}`), both
    /// models' dimensions included; `quant::f4_slot_offsets` has the arithmetic. Derived here,
    /// so the pairing is unrepresentable rather than merely warned about.
    pub(super) fn slot_offsets(self, expert_in: usize, moe_inter: usize) -> [usize; 6] {
        use crate::quant::{f4_slot_offsets, i4_slot_offsets, vq_slot_offsets};
        match self {
            RoutedFmt::Vq3 => vq_slot_offsets(expert_in, moe_inter),
            RoutedFmt::I4 => i4_slot_offsets(expert_in, moe_inter),
            RoutedFmt::F4 => f4_slot_offsets(expert_in, moe_inter),
        }
    }

    /// `(per-expert O_DIRECT-aligned stride, useful bytes in one block)`.
    pub(super) fn geometry(self, expert_in: usize, moe_inter: usize) -> (usize, usize) {
        match self {
            RoutedFmt::Vq3 => (
                vq_expert_stride(expert_in, moe_inter),
                vq_expert_bytes(expert_in, moe_inter),
            ),
            RoutedFmt::I4 => (
                i4_expert_stride(expert_in, moe_inter),
                i4_expert_bytes(expert_in, moe_inter),
            ),
            RoutedFmt::F4 => (
                f4_expert_stride(expert_in, moe_inter),
                f4_expert_bytes(expert_in, moe_inter),
            ),
        }
    }
}

/// The dims one layer file is confronted with at open — the CALLER's, not the file's own.
///
/// Grouped because the length check and the header check have to be made against the same five
/// numbers. A transposed `(expert_in, moe_inter)` keeps the byte count, so the header is the
/// only thing that can catch it — and a header checked against different dims than the length
/// was would catch nothing.
/// The five dimensions a layer file's header and its reader must agree on — ONE struct
/// consumed by both `ExpertHeader::new` and the reader's expectation, so the writer and
/// the checker cannot hold different lists (they were two copies until jscpd matched the
/// pair, 2026-08-15).
#[derive(Clone, Copy)]
pub struct LayerDims {
    pub layer: usize,
    pub n_experts: usize,
    pub expert_in: usize,
    pub moe_inter: usize,
    pub stride: usize,
}

#[cfg(test)]
mod tests {
    // The subject is a 40-byte round trip and the magic that separates two formats of
    // identical shape, so the fixtures are built in memory and nothing here touches a file.
    // Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// The GLM-shaped layer dims both header tests use; stride is the one axis that
    /// differs per format, so it is the one parameter.
    fn dims_6144x2048(stride: usize) -> LayerDims {
        LayerDims {
            layer: 7,
            n_experts: 256,
            expert_in: 6144,
            moe_inter: 2048,
            stride,
        }
    }

    #[test]
    fn vq3_header_roundtrips() {
        let h = ExpertHeader::new(VQ3_MAGIC, dims_6144x2048(vq_expert_stride(6144, 2048)));
        let back = ExpertHeader::from_bytes(&h.to_bytes(), RoutedFmt::Vq3).unwrap();
        assert_eq!(back.magic, VQ3_MAGIC);
        assert_eq!(back.layer, 7);
        assert_eq!(back.n_experts, 256);
        assert_eq!(back.expert_in, 6144);
        assert_eq!(back.moe_inter, 2048);
        assert_eq!(back.stride, vq_expert_stride(6144, 2048) as u64);
        // a corrupt magic must fail
        let mut bad = h.to_bytes();
        bad[0] = b'X';
        assert!(ExpertHeader::from_bytes(&bad, RoutedFmt::Vq3).is_err());
        // …and so must a WELL-FORMED header of the other format. This is the check the
        // length test cannot make: a `.vq3` and an `.f4` of the same nominal dims differ
        // in block count, so accepting either magic addresses past the last expert.
        assert!(ExpertHeader::from_bytes(&h.to_bytes(), RoutedFmt::F4).is_err());
        let f4 = ExpertHeader::new(F4_MAGIC, dims_6144x2048(f4_expert_stride(6144, 2048)));
        assert!(ExpertHeader::from_bytes(&f4.to_bytes(), RoutedFmt::Vq3).is_err());
        // Bidirectional: the correct pairing still parses, so the two refusals above are
        // the magic and not a header that stopped round-tripping.
        assert_eq!(
            ExpertHeader::from_bytes(&f4.to_bytes(), RoutedFmt::F4)
                .unwrap()
                .magic,
            F4_MAGIC
        );
    }
}
