//! `--f4`'s expert pass for `convert_inkling`: the routed experts as per-layer MXFP4 files.
//!
//! Split out of the bin root for the 800-line soft cap, and split HERE rather than into the
//! shared `inkling/f4.rs` because this half is Mini's alone: Small's experts arrive at MXFP8 and
//! never take the NVFP4 branch, so a shared module holding that branch would hand Small a dead
//! arm. What both members do share — the row decode, the block geometry, the atomic windowed
//! write — is `crate::f4`, included by both roots.
//!
//! **Both source encodings are re-quantized**, one row at a time: 0..20 arrive at NVFP4 (the
//! `(code · block_scale) · factor` decode stays in `inkling_dequant`) and 20..24 at MXFP8 (a
//! decode through the shared e4m3 table). This is the one LOSSY path in Mini's converter, and the
//! paired-dNLL gate is what scores it.

use super::{Nvfp4Layer, Nvfp4Plane, expert_prefix, f4};
use anyhow::{Result, bail};
use rivoli_artifact::format::{Dtype, Safetensors};
use rivoli_artifact::inkling_config::InklingTextConfig;
use std::collections::HashSet;

/// The `--f4` pass: the shard, the text config, and the artifact directory.
pub(super) struct Pass<'a> {
    pub(super) src: &'a Safetensors,
    pub(super) t: &'a InklingTextConfig,
    pub(super) out_dir: &'a str,
}

impl Pass<'_> {
    /// Mark and convert every layer's routed experts into `L{ll}.f4`.
    pub(super) fn layers(&self, consumed: &mut HashSet<String>) -> Result<()> {
        (0..self.t.num_hidden_layers).try_for_each(|l| self.layer(consumed, l))
    }
    /// One layer, dispatched on the SOURCE dtype of its `w13_weight` — the same match
    /// `Mxfp8Layers::layer` makes, for the same reason: an unrecognised encoding is refused by
    /// name rather than guessed at.
    fn layer(&self, consumed: &mut HashSet<String>, l: usize) -> Result<()> {
        let w13 = format!("{}w13_weight", expert_prefix(l));
        let (_, dt, _) = self.src.raw(&w13)?;
        let wrote = match dt {
            Dtype::F8E4M3 => self.quantize_mxfp8(consumed, l)?,
            Dtype::U8 => self.quantize_nvfp4(consumed, l)?,
            other => bail!(
                "{w13}: expert weight dtype {other:?} is neither NVFP4 (U8) nor MXFP8 \
                 (F8_E4M3) — --f4 has no path for it"
            ),
        };
        let path = format!("{}/L{l:02}.f4", self.out_dir);
        let what = if wrote { "written" } else { "kept" };
        eprintln!("convert_inkling: {path} ({what})");
        Ok(())
    }

    /// One layer file's identity and geometry. The `(hidden, inter)` pair comes from the config,
    /// so a layer cannot be written at another's width.
    fn spec(&self, l: usize) -> f4::F4LayerSpec<'_> {
        f4::F4LayerSpec {
            out_dir: self.out_dir,
            layer: l,
            n_experts: self.t.n_routed_experts,
            dims: (self.t.hidden_size, self.t.intermediate_size),
        }
    }

    /// A layer the checkpoint already holds at MXFP8: the shared reader checks the four shapes
    /// and walks the rows, so the de-interleave exists once for both converters.
    fn quantize_mxfp8(&self, consumed: &mut HashSet<String>, l: usize) -> Result<bool> {
        let p = expert_prefix(l);
        super::mark_mxfp8_consumed(consumed, &p);
        f4::quantize_layer_mxfp8(self.spec(l), self.src, &p)
    }

    /// A layer the checkpoint holds at NVFP4: decode one row at a time and quantize it into the
    /// FP4 block. The row walk of `quantize_expert_f4` is the shared one — the de-interleave and
    /// the offsets cannot drift between the two source encodings.
    fn quantize_nvfp4(&self, consumed: &mut HashSet<String>, l: usize) -> Result<bool> {
        let plane = Nvfp4Layer::load(self.src, self.t, l)?;
        plane.mark_consumed(consumed);
        let (h, i) = (self.t.hidden_size, self.t.intermediate_size);
        f4::write_f4_layer(self.spec(l), |e, slot| {
            let w13 = Nvfp4Rows {
                plane: &plane.w13,
                e,
            };
            let w2 = Nvfp4Rows {
                plane: &plane.w2,
                e,
            };
            f4::quantize_pair_f4(slot, &w13, &w2, (h, i))
        })
    }
}

/// One NVFP4 projection as a [`RowSource`](f4::RowSource): the plane plus the expert this pass is
/// packing, so the shared expert loop reaches `decode_expert_row` without knowing the plane's
/// shape or which projection it is.
struct Nvfp4Rows<'a> {
    plane: &'a Nvfp4Plane<'a>,
    e: usize,
}

impl f4::RowSource for Nvfp4Rows<'_> {
    fn row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        self.plane.decode_into((self.e, row), out)
    }
}
