//! What a V4 layer IS — its class, its compressor's shape, and the attention widths every
//! kernel on the path assumes. Arithmetic over `V4Config` and nothing else: no device, no
//! feature gate, and every relation checked at the point of use rather than trusted.
//!
//! Ported from `old:src/kvcompress.rs` (the host half) and `old:src/attn.rs::v4::Dims`.
//! The two lived apart there because the compressor and the attention block were written in
//! different stages; they are one file here because they answer one question — what shape is
//! this layer — and because a `Geom` built for one compressor and a `Dims` built for another
//! layer is precisely the pairing neither file could refuse on its own.

use anyhow::{Result, bail, ensure};
use rivoli_artifact::v4_config::V4Config;
use rivoli_backend::abi::CompGeom;

/// The block size V4's attention weights are quantized on — `weight_block_size: [128, 128]`
/// in the checkpoint's `quantization_config`, and also the block every quantized `Linear`
/// quantizes its ACTIVATION on. The KV entry's partial quantization is the one place that is
/// not 128; it is [`KV_QUANT_BLOCK`] and is spelled at its call site.
pub const FP8_BLOCK: usize = 128;

/// `act_quant(kv[..., :-rope_head_dim], **64**, …)` — the reference's KV entry finish, and
/// NOT the 128 every `Linear` uses.
///
/// Kept as a named constant beside [`FP8_BLOCK`] so the two are visibly different numbers
/// rather than one number someone rounded. The reference engine recorded a first-principles
/// argument that this choice "provably cannot be observed" — ue8m0 scales are powers of two
/// and e4m3 is exactly scale-invariant under those — and then MEASURED it wrong: the
/// invariance holds only while both blockings keep every value inside e4m3's range, and at a
/// rounding boundary the two blockings differ. Three readers passed over the derivation with
/// the contradicting numbers in front of them. Where a comment is about to say a choice
/// cannot be observed, check a run.
pub const KV_QUANT_BLOCK: usize = 64;

/// The only `compress_ratio` that carries a trained-in indexer.
const INDEXED_RATIO: usize = 4;

/// Which layer class this is, from its `compress_ratio`.
///
/// Three states rather than a `usize`, because every arithmetic path below divides by the
/// ratio and `0` is a legal entry in the config's table. A bare `usize` re-opens exactly that
/// division: read from one, `should_compress` was `seqlen >= 0` (always TRUE) at prefill and
/// `is_multiple_of(0)` (always false) at decode — two different answers for one layer, and
/// the prefill one sent every ratio-0 layer into a divide by zero. A decode-only smoke test
/// would not have shown it.
///
/// **This type does not catch the indexing mistake, and saying otherwise would be the
/// load-bearing lie.** [`LayerKind::from_ratio`] receives one already-extracted ratio and
/// never sees the layer index or the table's length — and that table is LONGER than the model
/// (its tail entries describe the checkpoint's speculative blocks, which this arm does not
/// run). The guard that catches a tail entry reaching a main-path layer is
/// `V4Config::compress_ratio`, which bounds-checks against `n_layers`. So prefer
/// [`LayerKind::from_config`], which goes through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// `compress_ratio == 0`: no compressor, no indexer, base `rope_theta`, **no YaRN**.
    Plain,
    /// `compress_ratio == 4`: an OVERLAPPING compressor, and the only class with an indexer.
    Overlap,
    /// Any other non-zero ratio: a non-overlapping compressor and no indexer; the selection
    /// is the arithmetic one in [`super::select`]. The ratio is CARRIED rather than baked
    /// into the name, because nothing in the reference fixes it — `overlap` keys on `== 4`
    /// alone, and this checkpoint's other value is a fact about this checkpoint.
    ///
    /// [`LayerKind::from_ratio`] is the only constructor, so `NonOverlap(0)` and
    /// `NonOverlap(4)` — each of which would make [`LayerKind::compressor_ratio`] disagree
    /// with [`LayerKind::overlap`] and [`LayerKind::has_indexer`] — are unrepresentable
    /// outside this module.
    NonOverlap(usize),
}

impl LayerKind {
    /// Classify `layer` through `V4Config::compress_ratio`, which bounds-checks against
    /// `n_layers` rather than against the ratio table's own length.
    ///
    /// Preferred over [`LayerKind::from_ratio`]: same classification, reached via the one
    /// accessor that refuses to hand back a speculative block's ratio for a main-path layer.
    /// `V4Config::layer_has_compressor` and `layer_has_indexer` answer two of the same
    /// questions one at a time; this returns the whole classification at once, so a caller
    /// cannot pair a compressor decision from one call with a pooling width from another.
    pub fn from_config(cfg: &V4Config, layer: usize) -> Result<Self> {
        Ok(Self::from_ratio(cfg.compress_ratio(layer)?))
    }

    /// Classify one layer from a raw ratio. Prefer [`LayerKind::from_config`] where a config
    /// is in hand — this one trusts the caller to have indexed the table correctly.
    pub fn from_ratio(ratio: usize) -> Self {
        match ratio {
            0 => Self::Plain,
            INDEXED_RATIO => Self::Overlap,
            r => Self::NonOverlap(r),
        }
    }

    /// The reference's `Compressor.overlap` — true iff the ratio is [`INDEXED_RATIO`].
    pub fn overlap(self) -> bool {
        match self {
            Self::Overlap => true,
            Self::Plain | Self::NonOverlap(_) => false,
        }
    }

    /// `coff = 1 + overlap`: the width multiplier on the positional table and on both
    /// compressor projections.
    ///
    /// At ratio 4 the compressor projects to `2 * head_dim` and splits it into an overlapping
    /// half and a normal half, so the table is `[4, 2*d]`; at any other ratio it is `[r, d]`.
    /// **A shape assumption that holds on one layer class breaks on the next**, which is why
    /// this is derived once here and read from [`Geom`] everywhere else.
    pub fn coff(self) -> usize {
        1 + usize::from(self.overlap())
    }

    /// The ratio, if this layer has a compressor at all — `None` for [`LayerKind::Plain`].
    ///
    /// The ONLY ratio accessor, deliberately. A companion returning a bare `usize` (0 for
    /// `Plain`) is the hazard this type was added to remove: offering both would make every
    /// future caller choose between a `0` that panics the divide and a `None` that cannot.
    /// The reference's own guard is `if self.compress_ratio:`.
    pub fn compressor_ratio(self) -> Option<usize> {
        match self {
            Self::Plain => None,
            Self::Overlap => Some(INDEXED_RATIO),
            Self::NonOverlap(r) => Some(r),
        }
    }

    /// Whether this layer carries a trained-in indexer — true only at [`INDEXED_RATIO`].
    pub fn has_indexer(self) -> bool {
        self.overlap()
    }
}

/// The tightest compressor ratio any layer class uses — what sizes shared selection-space
/// buffers and bounds the indexer's positional reach, since one buffer serves every class.
///
/// Read through [`LayerKind`] rather than spelled, so a change to which ratio carries an
/// indexer lands here — and only here: three call sites each spelling this read carried
/// the same comment making that promise, which three copies cannot keep (review 2026-08-16).
pub fn tightest_ratio() -> usize {
    LayerKind::Overlap.compressor_ratio().unwrap_or(1)
}

/// Which quantization a compressor finishes with — the reference's `rotate` flag, which is
/// the ONLY thing that differs between the attention compressor and the indexer's nested one.
///
/// An enum rather than a `bool`, and a FIELD of [`Geom`] rather than an argument beside it,
/// because the two are a different *algorithm* over an identical *shape*. Both arms accept
/// every geometry a `Geom` can hold, so nothing downstream would reject the wrong one:
/// block-64 e4m3 over a prefix where a Hadamard spread plus fp4 over the whole row was due is
/// finite, plausible, and wrong. Making it a field means the choice is made by whichever
/// constructor ran and cannot be made again at the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantize {
    /// `rotate = false` — block-64 e4m3 over dims `[0, d - rd)` only, leaving the rotary tail
    /// in bf16 to match how the checkpoint was trained.
    PartialFp8,
    /// `rotate = true` — the Hadamard spread then fp4 over the WHOLE row, rotary tail
    /// included. Takes no partial extent at all, which is the other half of the same trap.
    HadamardFp4,
}

/// One compressor's geometry **and the finish it owes**.
///
/// Built from a [`LayerKind`], so [`LayerKind::Plain`] — which has no compressor object in
/// the reference at all — cannot produce one.
///
/// **This module is the sole producer of [`CompGeom`]**, which is the rule `rivoli-backend`'s
/// `abi` states in the other direction: it owns the `repr(C)` layout and a compile-time
/// assert on it, and it deliberately owns no constructor, because the derivation
/// (`coff` from overlap, `cd` and `ents` from `coff`) is semantic and belongs where the
/// layer class is known. The kernel's own `compress_guard` catches a REORDER of the six
/// integers at run time and the layout assert catches a field being added or resized; what
/// neither can catch is a `coff` that went stale, and one derivation site is the answer to
/// that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geom {
    abi: CompGeom,
    quant: Quantize,
}

impl Geom {
    /// The **attention** compressor's geometry — the partial block-64 finish.
    ///
    /// **Named `attention`, not `new`, and that name is the guard.** The other compressor on
    /// a ratio-4 layer is the indexer's nested one, whose geometry is identical in shape and
    /// whose algorithm is not; a `Geom` built by the wrong constructor passes every dimension
    /// check there is.
    ///
    /// `d` is the *compressor's* `head_dim` — the attention's for this one, the much narrower
    /// `index_head_dim` for [`Geom::indexer`]. They are different geometries on the SAME
    /// layer, and taking one for the other still "works".
    ///
    /// `None` for [`LayerKind::Plain`]: there is no geometry to describe and every path below
    /// would divide by zero.
    ///
    /// Only the ratio is validated here, by the [`LayerKind`] that carries it. `d`,
    /// `rope_head_dim` and `norm_eps` come from the config, so a bad one is a bad checkpoint
    /// and is caught at the ABI wall where every launcher sees it. That split is deliberate
    /// rather than an omission — but it does mean a `Geom` can exist that no launcher will
    /// accept, and the error then names a guard code and not a field.
    pub fn attention(
        kind: LayerKind,
        d: usize,
        rope_head_dim: usize,
        norm_eps: f32,
    ) -> Option<Self> {
        // The ONE derivation of `cd`/`ents`: `Geom::indexer` refines what this returns rather
        // than repeating it, because two copies of the arithmetic are two places for a `coff`
        // to go stale.
        let ratio = kind.compressor_ratio()?;
        let coff = kind.coff();
        Some(Self {
            abi: CompGeom {
                ratio: ratio as i32,
                coff: coff as i32,
                d: d as i32,
                cd: (coff * d) as i32,
                ents: (coff * ratio) as i32,
                rd: rope_head_dim as i32,
                eps: norm_eps,
            },
            quant: Quantize::PartialFp8,
        })
    }

    /// The **indexer's** compressor geometry — the Hadamard-plus-fp4 finish.
    ///
    /// `None` unless the layer [`LayerKind::has_indexer`], which is stricter than
    /// [`Geom::attention`] accepts: an indexer exists only at [`INDEXED_RATIO`], so every
    /// other ratio is unrepresentable here rather than merely undocumented.
    ///
    /// Refining `attention`'s result is sound because [`Quantize`] is an input to none of the
    /// derived integers — and that is not left to this comment: it is what makes
    /// `a.abi() == i.abi()` for this pair a checkable claim rather than a hope.
    pub fn indexer(kind: LayerKind, d: usize, rope_head_dim: usize, norm_eps: f32) -> Option<Self> {
        // `has_indexer`, not `compressor_ratio().is_some()`: the question is whether the
        // layer HAS an indexer, so routing it through the accessor that answers exactly that
        // keeps a future change to which ratios carry one in a single place. The `?` below
        // cannot fire — `has_indexer()` implies a ratio `compressor_ratio()` answers `Some`
        // for — and is spelled rather than unwrapped because a total expression is cheaper
        // than a panic nobody could act on.
        if !kind.has_indexer() {
            return None;
        }
        Some(Self {
            quant: Quantize::HadamardFp4,
            ..Self::attention(kind, d, rope_head_dim, norm_eps)?
        })
    }

    /// The `repr(C)` half, for the launchers. Hands out a shared reference to an
    /// all-private-field value, so a caller across the ABI wall can pass one and cannot build
    /// or mutate one.
    pub fn abi(&self) -> &CompGeom {
        &self.abi
    }

    /// Which finish this geometry owes. See [`Quantize`].
    pub fn quantize(self) -> Quantize {
        self.quant
    }

    /// `compress_ratio`. Public because the launchers' safety contracts are stated in it —
    /// the positional table is `ratio * cd`, and a caller that cannot read the field cannot
    /// check the contract it is being asked to uphold.
    pub fn ratio(self) -> usize {
        self.abi.ratio as usize
    }

    /// This compressor's `head_dim` — the width of ONE pooled block. Not [`Geom::cd`], which
    /// is twice this on an overlapping layer.
    pub fn d(self) -> usize {
        self.abi.d as usize
    }

    /// `coff * d` — the width both projections produce, and the positional table's row
    /// stride.
    pub fn cd(self) -> usize {
        self.abi.cd as usize
    }

    /// Only the LAST this-many dims of a pooled row are rotated; the first `d - rd` are what
    /// the partial finish covers.
    pub fn rd(self) -> usize {
        self.abi.rd as usize
    }

    /// `coff * ratio` — entries in one pooling window, and the row count of both state
    /// buffers.
    pub fn ents(self) -> usize {
        self.abi.ents as usize
    }

    /// Elements in each pooling state buffer: `[ents, cd]`.
    pub fn state_len(self) -> usize {
        self.ents() * self.cd()
    }
}

/// Everything the attention block needs that does not vary between calls.
///
/// A struct because they are nine numbers in a row and every one is plausible in another's
/// position — `n_heads` and `index_n_heads` are both 64 on this checkpoint, `head_dim` and
/// `o_lora_rank` and `q_lora_rank` are all four figures, and a transposed pair indexes a real
/// row and produces a finite answer.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    /// `sliding_window` — the ring's size, and the base row of the compressed region.
    pub window: usize,
    pub norm_eps: f32,
}

impl Dims {
    /// Derive from the artifact's config, validating every relation the kernels assume.
    pub fn from_config(cfg: &V4Config) -> Result<Self> {
        // `f64` in the config because JSON numbers are, `f32` here because the kernels are.
        // The narrowing ROUNDS — this checkpoint's eps is representable in neither format —
        // but it lands on the same bits as the f32 literal, so there is no double-rounding
        // surprise, and the absolute error is thirteen orders below the norm it perturbs.
        let d = Self {
            dim: cfg.hidden,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            rope_head_dim: cfg.qk_rope_head_dim,
            q_lora_rank: cfg.q_lora_rank,
            o_groups: cfg.o_groups,
            o_lora_rank: cfg.o_lora_rank,
            window: cfg.sliding_window,
            norm_eps: cfg.rms_norm_eps as f32,
        };
        d.validate()?;
        Ok(d)
    }

    /// Every relation the kernels assume, checked against the values actually held.
    ///
    /// Called by [`Dims::from_config`] AND at the attention block's entry: this type is
    /// `Copy` with public fields, so a struct literal or a later field assignment skips the
    /// constructor entirely — and the struct literal is exactly what a test harness writes,
    /// which is the shape a layer loop copies by default. Sealing the fields stops the
    /// literal and not the mutation, and costs every reader an accessor; checking at the
    /// point of use stops both, for a handful of integer compares against a 4096-wide GEMV.
    pub fn validate(&self) -> Result<()> {
        self.no_extent_is_zero()?;
        self.rotary_fits_inside_a_head()?;
        self.every_quantized_row_is_a_whole_number_of_blocks()?;
        // The grouped output projection reads its input as `o_groups` contiguous runs of
        // [`Dims::group_width`], so a ragged split reshapes into the wrong stride rather than
        // failing. Stated as the WIDTH multiplying back, not as a divisibility, because that
        // width is the number the launcher is handed — the config's own validator asks the
        // divisibility question and this one asks what the call site actually passes.
        let outputs = self.n_heads * self.head_dim;
        ensure!(
            self.group_width() * self.o_groups == outputs,
            "v4 attention: {outputs} attention outputs do not split into {} equal o_groups",
            self.o_groups
        );
        Ok(())
    }

    /// One output group's width — the reduction extent the grouped projection is launched
    /// with. `o_groups` is non-zero by [`Dims::no_extent_is_zero`], which runs first.
    pub fn group_width(&self) -> usize {
        self.n_heads * self.head_dim / self.o_groups
    }

    /// EVERY extent, not just the three that read as counts.
    ///
    /// `is_multiple_of` admits zero (`0.is_multiple_of(128)` is true) and so do `0 > 0` and
    /// `0.is_multiple_of(2)`, so without this sweep a zero width passes every check below and
    /// surfaces as an opaque numeric guard code from whichever launcher happens to run first.
    /// `rope_head_dim == 0` is the interesting one: it means no rotary at all, which is a
    /// legal-looking config and a completely different model.
    fn no_extent_is_zero(&self) -> Result<()> {
        for (v, what) in [
            (self.window, "sliding_window"),
            (self.n_heads, "n_heads"),
            (self.o_groups, "o_groups"),
            (self.dim, "hidden"),
            (self.head_dim, "head_dim"),
            (self.rope_head_dim, "qk_rope_head_dim"),
            (self.q_lora_rank, "q_lora_rank"),
            (self.o_lora_rank, "o_lora_rank"),
        ] {
            if v == 0 {
                bail!("v4 attention: {what} is zero");
            }
        }
        Ok(())
    }

    fn rotary_fits_inside_a_head(&self) -> Result<()> {
        let (hd, rd) = (self.head_dim, self.rope_head_dim);
        ensure!(
            rd <= hd && rd.is_multiple_of(2),
            "v4 attention: qk_rope_head_dim {rd} must be even and at most head_dim {hd}"
        );
        // The one extent the zero sweep cannot reach, because it is DERIVED. A config with
        // `qk_rope_head_dim == head_dim` — "rotate the whole head", which looks entirely
        // ordinary — clears the bound above (`hd > hd` is false), is even, and then satisfies
        // the block test below because `0.is_multiple_of(64)` is TRUE. It reached the
        // quantizer with a zero extent and came back as an opaque guard code: late, and
        // exactly what the sweep exists to prevent.
        ensure!(
            hd != rd,
            "v4 attention: head_dim - qk_rope_head_dim is zero"
        );
        ensure!(
            (hd - rd).is_multiple_of(KV_QUANT_BLOCK),
            "v4 attention: head_dim - qk_rope_head_dim = {} is not a multiple of \
             {KV_QUANT_BLOCK}, the KV entry's partial quantization block",
            hd - rd
        );
        Ok(())
    }

    /// The activation quantizer asserts its row is a whole number of blocks. These are the
    /// three `Linear` inputs, all at [`FP8_BLOCK`]; the KV entry's partial span is checked
    /// above at its own block size.
    fn every_quantized_row_is_a_whole_number_of_blocks(&self) -> Result<()> {
        for (n, what) in [
            (self.dim, "hidden"),
            (self.q_lora_rank, "q_lora_rank"),
            (self.o_groups * self.o_lora_rank, "o_groups*o_lora_rank"),
        ] {
            ensure!(
                n.is_multiple_of(FP8_BLOCK),
                "v4 attention: {what} = {n} is not a multiple of {FP8_BLOCK}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

    use super::*;

    /// The classification's whole point: the three accessors that can disagree never do,
    /// whichever ratio produced the kind.
    #[test]
    fn a_layer_kinds_ratio_overlap_and_indexer_answers_cannot_disagree() {
        for r in [0usize, 1, 2, 4, 8, 128] {
            let k = LayerKind::from_ratio(r);
            assert_eq!(k.compressor_ratio(), (r != 0).then_some(r));
            assert_eq!(k.overlap(), r == INDEXED_RATIO);
            assert_eq!(k.has_indexer(), k.overlap());
            assert_eq!(k.coff(), 1 + usize::from(r == INDEXED_RATIO));
        }
    }

    /// Two geometries on ONE layer, differing only in the finish and the width — which is the
    /// pairing that has no runtime check anywhere downstream.
    #[test]
    fn the_indexers_geom_differs_from_the_attentions_only_in_its_finish_at_equal_widths() {
        let k = LayerKind::from_ratio(4);
        let a = Geom::attention(k, 128, 64, 1e-6).expect("ratio 4 has a compressor");
        let i = Geom::indexer(k, 128, 64, 1e-6).expect("ratio 4 has an indexer");
        assert_eq!(a.abi(), i.abi(), "the ABI half is the SHAPE and must match");
        assert_eq!(a.quantize(), Quantize::PartialFp8);
        assert_eq!(i.quantize(), Quantize::HadamardFp4);
        // And the derivation: overlap doubles the projected width and the window's entries.
        assert_eq!((a.d(), a.cd(), a.ents()), (128, 256, 8));
    }

    /// Neither constructor may produce a geometry for a layer that has no such compressor —
    /// the states this type exists to make unrepresentable.
    #[test]
    fn a_plain_layer_has_no_compressor_and_only_ratio_four_has_an_indexer() {
        assert!(Geom::attention(LayerKind::from_ratio(0), 512, 64, 1e-6).is_none());
        assert!(Geom::indexer(LayerKind::from_ratio(0), 128, 64, 1e-6).is_none());
        assert!(Geom::indexer(LayerKind::from_ratio(128), 128, 64, 1e-6).is_none());
        assert!(Geom::attention(LayerKind::from_ratio(128), 512, 64, 1e-6).is_some());
    }

    /// The dims a struct literal can hold that `from_config` would never produce, each
    /// refused by name. Every case starts from a set that PASSES, so no case can be passing
    /// for another's reason.
    #[test]
    fn validate_refuses_each_degenerate_width_by_name() {
        let ok = Dims {
            dim: 4096,
            n_heads: 64,
            head_dim: 512,
            rope_head_dim: 64,
            q_lora_rank: 1024,
            o_groups: 8,
            o_lora_rank: 1024,
            window: 128,
            norm_eps: 1e-6,
        };
        ok.validate().expect("the shipped widths must validate");
        /// One perturbation of an otherwise-valid `Dims` — a struct literal's worth of
        /// damage, which is the mutation path `validate` exists to survive.
        type Perturb = fn(&mut Dims);
        let cases: [(Perturb, &str); 5] = [
            (|d| d.rope_head_dim = 0, "qk_rope_head_dim is zero"),
            (|d| d.rope_head_dim = 63, "must be even"),
            (|d| d.rope_head_dim = 512, "is zero"),
            (|d| d.q_lora_rank = 1000, "not a multiple of"),
            (|d| d.o_groups = 7, "do not split into"),
        ];
        for (break_it, want) in cases {
            let mut d = ok;
            break_it(&mut d);
            let msg = format!("{}", d.validate().expect_err("must refuse"));
            assert!(msg.contains(want), "wrong refusal for {d:?}: {msg}");
        }
    }
}
