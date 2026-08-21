//! The on-disk artifact: the single source of truth for the layout the converter
//! writes and the loader reads. An artifact directory is:
//!
//! ```text
//! manifest.json          # HF config fields + a `format` section (FormatMeta)
//! codebooks.f32          # 3 per-projection codebooks (gate, up, down): VQ_K·VQ_DIM f32 each
//! resident.safetensors   # every resident weight (fp8 attn/dense, int8 embed, f32 norms, bf16 indexer)
//! L{03..NN}.vq3          # per MoE layer: header + (n_experts + 1) expert blocks,
//!                        #   block = gate‖up‖down; block n_experts = the shared expert
//! L{03..NN}.i4           # optional int4 twin of the .vq3 (bin/fp8_to_i4): headerless,
//!                        #   (n_experts + 1) blocks from offset 0. Streamed under --i4.
//! L{00..NN}.f4           # DeepSeek-V4-Flash (bin/convert_v4): header + exactly n_experts
//!                        #   FP4 blocks and NO shared block — V4's shared expert is fp8
//!                        #   e4m3 and rides `resident.safetensors`. See [`RoutedFmt`].
//! ```
//!
//! **This file is now the SHARED VOCABULARY; one job per submodule.** Split 2026-08-15 under
//! the 800-line ceiling: at 2741 lines it held four unrelated jobs — a safetensors reader and
//! writer, the FP4 repack, the routed-expert layer container, and `manifest.json` — with no
//! call edge between most of them. What stayed is the one type all four name: the [`Dtype`] a
//! tensor declares. What left went by job: [`tensors`] (the safetensors container, both
//! directions), [`expert`] and [`repack`] (the repack's per-expert source spans and its
//! per-layer driver), [`header`], [`layer`] and [`set`] (the routed container's header, its
//! writer and its reader), and [`meta`] (`manifest.json` — the format stamp, `.i4`
//! provenance, and publishing the artifact).
//!
//! **Two boundaries were moved by CodeScene rather than by taste, and they are the better
//! reading.** A module whose arguments are mostly `&str` scores as string-heavy, and the
//! 2741-line file had hidden that: the safetensors halves are name-keyed lookups on both
//! sides, so splitting them left two string-heavy modules where the joined one is not — and
//! joined is where the round-trip tests can build their own fixtures. `f4_source` /
//! `f4_layer_range` went to [`set`] for the same reason and the same benefit: they are the
//! loader's layer range, not a fact about `manifest.json`, and they sit beside the `SetDims`
//! that consumes them.
//!
//! **The split was a move, not a rewrite**: every body and every comment travelled verbatim,
//! because in this repo the comments carry the measurement that chose the constant. SIX
//! private methods widened to `pub(super)` so the sibling that already called them still
//! can ([`ExpertHeader`]'s `from_bytes` and [`RoutedFmt`]'s five accessors), a handful of
//! intra-doc links gained a `super::`, and the TWO test helpers more than one module uses
//! (`tmpdir` and `F4Fixture`) moved to a `#[cfg(test)]` `fixtures` module. Nothing else
//! changed, and every public name is re-exported below at its original path, so no caller
//! outside this file moved.

pub mod expert;
pub mod header;
pub mod layer;
pub mod meta;
pub mod repack;
pub mod set;
pub mod tensors;

#[cfg(test)]
mod fixtures;

/// Every name the submodules own, re-exported at the path it has always had.
///
/// The address of a public function is itself an interface: `bin/convert`, `bin/fp8_to_i4`,
/// `bin/add_indexer` and the engine's artifact tests all spell these
/// `rivoli_artifact::format::…`, and nothing was learned in the 2026-08-15 split that
/// justifies making them move. Adding a submodule is a file-layout decision; it is not an
/// API change, and this block is what keeps the two separate.
pub use expert::{F4_NAMING_K3, F4_NAMING_V4, F4Expert, F4Naming};
pub use header::{EXPERT_HEADER_BYTES, ExpertHeader, F4_MAGIC, LayerDims, RoutedFmt, VQ3_MAGIC};
pub use layer::{LAYER_WINDOW, write_expert_layer};
pub use meta::{ArtifactDirs, FormatMeta, I4Source, finish_artifact, require_aux};
pub use repack::RoutedRepack;
pub use set::{ExpertSet, SetDims, f4_layer_range, f4_source, load_codebooks};
pub use tensors::{SafeWriter, Safetensors};

/// A tensor's dtype in a safetensors header (only the ones this engine uses).
///
/// The one type the 2026-08-15 split left here, because two submodules name it and neither
/// owns it: [`tensors`] narrows the crate's dtype to it on read and writes its `name` back out
/// (37 mentions), and [`expert`] tabulates it per checkpoint in `F4Naming` (9) — the whole
/// point of that struct being that two checkpoints declare the same bytes under different
/// dtype strings. Its two private methods needed no widening: a child module sees its
/// ancestors' private items, so `narrow` and `name` stayed reachable from [`tensors`] and,
/// as before, from nowhere outside this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    U8,
    I8,
    I64,
    Bf16,
    F8E4M3,
    /// A bare 8-bit exponent, `2^(b-127)`. DeepSeek-V4-Flash's block scales — 128×128 on
    /// the fp8 attention tensors, one per 32 weights along the input dim on the FP4
    /// experts. Decoded by [`crate::quant::e8m0`].
    F8E8M0,
}

impl Dtype {
    /// Narrow the `safetensors` crate's dtype to the ones this engine decodes.
    ///
    /// This enum stays rather than becoming a re-export because the crate's is
    /// `#[non_exhaustive]` with ~20 variants: every `match` on it would need a `_` arm, and
    /// the point of a closed seven-variant enum is that adding a dtype is a COMPILE error at
    /// each decode site instead of a runtime fallthrough. The narrowing happens once, here.
    ///
    /// **That is not hypothetical, and 0.8.0 is why.** Its two additions over 0.7.0 are
    /// `F8_E4M3FNUZ` and `F8_E5M2FNUZ` — the AMD/Graphcore fp8 encodings, a different
    /// exponent bias and no signed zero, and this engine runs on AMD. The `_` arm below
    /// REFUSES them. A re-export would instead have let an FNUZ tensor reach
    /// `quant::dequant_fp8_block`, which decodes OCP e4m3 unconditionally: every weight off by
    /// a power of two, no error anywhere, fluent wrong output. The bytes do not say which
    /// encoding they are — only the dtype string does — so this match is the whole check.
    fn narrow(d: safetensors::Dtype) -> Option<Dtype> {
        Some(match d {
            safetensors::Dtype::F32 => Dtype::F32,
            safetensors::Dtype::U8 => Dtype::U8,
            safetensors::Dtype::I8 => Dtype::I8,
            safetensors::Dtype::I64 => Dtype::I64,
            safetensors::Dtype::BF16 => Dtype::Bf16,
            safetensors::Dtype::F8_E4M3 => Dtype::F8E4M3,
            safetensors::Dtype::F8_E8M0 => Dtype::F8E8M0,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I64 => "I64",
            Dtype::Bf16 => "BF16",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E8M0 => "F8_E8M0",
        }
    }
}
