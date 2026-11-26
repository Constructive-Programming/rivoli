//! `convert_inkling` — the Inkling 2 Mini checkpoint → a rivoli text artifact.
//!
//! **Text-only v1, both members' shared stack, Mini first.** The converter consumes the 822 TEXT
//! tensors of the single shard, excludes the 422 `model.visual.*` tensors BY NAME (counted, never
//! written), and never opens `mtp.safetensors` (a separate file, refused by `Flag::Mtp`). The
//! census closure (`census::inkling::Census::confront`) classifies every one of the 1244 names
//! before a byte is converted; a name matching no family is a hard error.
//!
//! **The experts are DEQUANTIZED to bf16 by default, and streamed as `.mxfp8` under `--mxfp8`.**
//! Mini's first-party loop path (`Inkling2LoopDequantizeExperts`, LOAD:772) dequantizes on load,
//! and the default artifact stores that result: NVFP4 (layers 0–19) and MXFP8 (20–23) decoded by
//! `inkling_dequant`, whose oracle is bit-compared against the transformers op
//! (`tests/inkling_dequant.rs`). `--mxfp8` instead writes one `L{ll}.mxfp8` per layer for the
//! streaming engine: the four native-MXFP8 layers are REPACKED bit-for-bit (and verified), and
//! the twenty NVFP4 ones are decoded and re-quantized row-wise by `quantize_mxfp8_row`, which
//! costs a re-rounding those layers do not otherwise pay. `w13` is de-interleaved into
//! `[gate_all; up_all]` in both modes; the two message-query banks merge to `(q,k,v,mlp)`;
//! `unembed` keeps the first 200058 rows. The output names are the first-party text model's
//! state-dict names (`text-model-names.tsv`, 693 params), gated by `tests/inkling_convert.rs`.
//!
//! **Two further format knobs, both P5 (bytes/token) and both orthogonal to the expert choice.**
//! `--f4` writes the 24 layers as `L{ll}.f4` — MXFP4, half of `--mxfp8`'s bytes — quantizing
//! whichever source encoding the layer ships (NVFP4 for 0..20, MXFP8 for 20..24) through the
//! shared [`f4`] module; it is the one LOSSY path here. `--fp8` quantizes the five large dense
//! attention projections per layer to block-scaled FP8 in `resident.safetensors`, cutting the
//! bf16 trunk roughly in half at `FP8_BLOCK`-tiled scales.
//!
//! Plan of record: `docs/investigations/inkling2-port.md` (stage S2).

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use rivoli_artifact::census::inkling::Census;
use rivoli_artifact::format::{
    Dtype, ExpertFile, ExpertHeader, FormatMeta, LAYER_WINDOW, LayerDims, LayerInputs, LayerShape,
    MXFP8_MAGIC, Mxfp8Layer, SafeWriter, Safetensors, TensorData, f4_source, finish_artifact,
    mxfp8_source, write_expert_layer,
};
use rivoli_artifact::inkling_config::{Inkling2LoopConfig, InklingTextConfig};
use rivoli_artifact::inkling_convert::{
    INKLING_AUX, VocabRows, copy_tiktoken, emit_lm_head, emit_verbatim_fmt, preflight,
    query_target, rename_verbatim,
};
use rivoli_artifact::inkling_dequant::{Mxfp8, Nvfp4, merge_message_queries};
use rivoli_artifact::quant::{
    ExpertBlocks, FP8_BLOCK, mxfp8_expert_bytes, mxfp8_expert_stride, mxfp8_groups,
    mxfp8_slot_offsets, quantize_mxfp8_row, read_f32,
};
use std::collections::HashSet;

// `#[path]` because a bin root resolves its children against `src/bin/` and cargo
// autobin-discovers every `src/bin/*.rs` as its own executable — so a module shared with the
// Small converter cannot sit beside the roots as a bare `.rs` (the `convert/gpu.rs` precedent).
#[path = "inkling/f4.rs"]
mod f4;

// `--f4`'s expert pass, which is Mini's alone (Small's experts arrive at MXFP8 and never take
// the NVFP4 branch), split out for the 800-line soft cap.
#[path = "convert_inkling/f4_pass.rs"]
mod f4_pass;

/// The single text shard. `mtp.safetensors` is NEVER named here — it is a separate file the text
/// converter does not open. `open_file` (not `open_dir`) so the directory's `mtp.safetensors` is
/// not mmap'd by accident.
const SHARD: &str = "model-00001-of-00001.safetensors";

/// Mini's two loop banks, which merge into `message_input.query` (trap T13). They are NOT in
/// `inkling_convert`: only Mini has them — Small ships no `middle_cycle` tensor at all, and its
/// converter must not grow a branch for a bank it can never see.
const MIDDLE_CYCLE_Q_MLP: &str = "model.llm.middle_cycle_cross_pass_queries";
const MIDDLE_CYCLE_KV: &str = "model.llm.middle_cycle_cross_pass_kv_queries";

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
#[derive(Parser)]
#[command(
    name = "convert_inkling",
    about = "Inkling 2 Mini checkpoint → a rivoli text artifact (routed experts dequantized to \
             bf16 or streamed as per-layer .mxfp8/.f4 files, dense attention optionally fp8, \
             vision excluded by name, MTP never opened)"
)]
struct Args {
    /// The checkpoint directory: config.json, model-00001-of-00001.safetensors, tiktoken/, and the
    /// tokenizer files copied into the artifact.
    src_dir: String,

    /// Artifact directory to write. Created if absent.
    out_dir: String,

    /// Emit `L{ll}.mxfp8` per-layer expert files instead of bf16 expert slabs in
    /// `resident.safetensors` — the layout the streaming engine opens. The four layers already at
    /// MXFP8 are repacked bit-for-bit and verified; the twenty NVFP4 layers are decoded and
    /// re-quantized row-wise (a re-rounding this flag is the only thing that pays for).
    #[arg(long)]
    mxfp8: bool,

    /// Emit `L{ll}.f4` per-layer MXFP4 expert files — half the bytes of `--mxfp8` and the format
    /// the FP4 streaming descriptor reads. LOSSY: both source encodings are re-quantized row-wise,
    /// so score the artifact with the paired-dNLL gate. Exclusive with `--mxfp8`.
    #[arg(long)]
    f4: bool,

    /// Quantize the five large dense attention projections per layer (q/k/v/attn-gate/o) to
    /// block-scaled FP8-E4M3 at `FP8_BLOCK` in `resident.safetensors`, with the matching
    /// `weight_scale_inv` grids. Absent, the trunk stays bf16.
    #[arg(long)]
    fp8: bool,
}

/// The three artifact-shape flags, bundled because `build_manifest` reads all of them and a
/// three-element tuple at the call site is a value no reader can check against the field order.
#[derive(Clone, Copy)]
struct Mode {
    mxfp8: bool,
    f4: bool,
    fp8: bool,
}

/// One expert projection: the source suffix, the target suffix, and whether `w13`'s gate/up rows
/// need de-interleaving (`w2` is a plain rename).
struct ExpertProj {
    src: &'static str,
    target: &'static str,
    deinterleave: bool,
}

const W13: ExpertProj = ExpertProj {
    src: "w13_weight",
    target: "gate_up_proj",
    deinterleave: true,
};
const W2: ExpertProj = ExpertProj {
    src: "w2_weight",
    target: "down_proj",
    deinterleave: false,
};

/// The conversion state: the source, the text config, and the writer. `consumed` is carried
/// SEPARATELY (a `&mut HashSet` argument) so marking a name consumed never borrows the source the
/// writer's payloads still mmap.
struct Conv<'a> {
    src: &'a Safetensors,
    t: &'a InklingTextConfig,
    w: SafeWriter<'a>,
    /// `--fp8`'s tile size, or `None` for a bf16 trunk — the value
    /// [`emit_verbatim_fmt`](rivoli_artifact::inkling_convert::emit_verbatim_fmt) dispatches on
    /// and the one the loader sniffs back from layer 0's `q_proj` dtype.
    fp8_block: Option<usize>,
}

impl<'a> Conv<'a> {
    /// Every layer's two expert projections, dequantized to bf16 and (for w13) de-interleaved —
    /// the DEFAULT mode. `--mxfp8` takes [`Mxfp8Layers`] instead, and the two modes are
    /// exclusive: writing both would either duplicate the routed experts or leave one of the two
    /// containers half-populated.
    fn experts(&mut self, consumed: &mut HashSet<String>) -> Result<()> {
        for l in 0..self.t.num_hidden_layers {
            self.expert_proj(consumed, l, &W13)?;
            self.expert_proj(consumed, l, &W2)?;
        }
        Ok(())
    }

    /// One projection of one layer. The weight's dtype selects the format: U8 is NVFP4 (amax
    /// present), F8_E4M3 is MXFP8. The companion scale/amax/denominator tensors are consumed here
    /// and never emitted — they are spent by the dequant, which writes bf16 bytes DIRECTLY (with
    /// the `w13` de-interleave fused in) so no f32 plane of a 128-expert layer is ever allocated.
    fn expert_proj(
        &mut self,
        consumed: &mut HashSet<String>,
        l: usize,
        p: &ExpertProj,
    ) -> Result<()> {
        let base = format!("model.llm.layers.{l}.mlp.experts.{}", p.src);
        let (wb, dt, wshape) = self.src.raw(&base)?;
        ensure!(
            wshape.len() == 3,
            "{base}: shape {wshape:?} is not [E, rows, cols]"
        );
        let e = wshape[0];
        let rows = wshape[1];
        let (bytes, cols) = match dt {
            Dtype::U8 => {
                let cols_packed = wshape[2];
                let (sb, _) = self.src.typed(&format!("{base}_scale"), Dtype::F8E4M3)?;
                let (ab, _) = self.src.typed(&format!("{base}_amax"), Dtype::F32)?;
                let (db, _) = self
                    .src
                    .typed(&format!("{base}_outer_scale_denominator"), Dtype::F32)?;
                ensure!(
                    db.len() == 4,
                    "{base}_outer_scale_denominator: expected 4 bytes, got {}",
                    db.len()
                );
                let amax = read_f32(ab);
                let denom = read_f32(db)[0];
                consumed.insert(format!("{base}_scale"));
                consumed.insert(format!("{base}_amax"));
                consumed.insert(format!("{base}_outer_scale_denominator"));
                let b = Nvfp4 {
                    weight: wb,
                    scale: sb,
                    amax: &amax,
                    denom,
                    rows,
                    cols_packed,
                }
                .dequant_to_bf16_le(p.deinterleave)?;
                (b, cols_packed * 2)
            }
            Dtype::F8E4M3 => {
                let cols = wshape[2];
                let (sb, _) = self.src.typed(&format!("{base}_scale"), Dtype::U8)?;
                consumed.insert(format!("{base}_scale"));
                let b = Mxfp8 {
                    weight: wb,
                    scale: sb,
                    rows,
                    cols,
                }
                .dequant_to_bf16_le(p.deinterleave)?;
                (b, cols)
            }
            other => bail!(
                "{base}: expert weight dtype {other:?} is neither NVFP4 (U8) nor MXFP8 (F8_E4M3)"
            ),
        };
        consumed.insert(base);
        let target = format!("model.layers.{l}.mlp.experts.{}", p.target);
        self.w.add(
            target,
            Dtype::Bf16,
            TensorData::new(vec![e, rows, cols], bytes),
        );
        Ok(())
    }

    /// The query banks: `attn_queries.{I}` feeds layer `I+1` (trap T12), `mlp_queries.{I}` feeds
    /// layer `I`, `output_query` is `final_input` — all three through
    /// [`query_target`](rivoli_artifact::inkling_convert::query_target), which is also what the
    /// Small converter calls — and Mini's two `middle_cycle` banks merge into
    /// `message_input.query` in `(q,k,v,mlp)` order (trap T13). Only Mini has those two: Small
    /// ships no `middle_cycle` tensor at all.
    fn queries(&mut self, consumed: &mut HashSet<String>) -> Result<()> {
        let names: Vec<String> = self.src.names().iter().map(|s| s.to_string()).collect();
        let mut q_mlp = None;
        let mut kv = None;
        for name in &names {
            if let Some(target) = query_target(name)? {
                self.copy_renamed(consumed, name, target)?;
            } else if name == MIDDLE_CYCLE_Q_MLP {
                q_mlp = Some(read_f32(self.src.typed(name, Dtype::F32)?.0));
                consumed.insert(name.clone());
            } else if name == MIDDLE_CYCLE_KV {
                kv = Some(read_f32(self.src.typed(name, Dtype::F32)?.0));
                consumed.insert(name.clone());
            }
        }
        let q = q_mlp.context("checkpoint lacks middle_cycle_cross_pass_queries")?;
        let k = kv.context("checkpoint lacks middle_cycle_cross_pass_kv_queries")?;
        let lc = self
            .t
            .loop_config
            .as_ref()
            .context("Mini carries loop_config")?;
        let loop_layers = lc.core_end - lc.core_start;
        let h = self.t.hidden_size;
        let merged = merge_message_queries(&q, &k, loop_layers, h)?;
        let bytes: Vec<u8> = merged.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.w.add(
            "model.message_input.query",
            Dtype::F32,
            TensorData::new(vec![loop_layers, 4, h], bytes),
        );
        Ok(())
    }

    /// `unembed` → `lm_head`, keeping the first `unpadded_vocab_size` rows (LOAD:856, trap T14)
    /// through `inkling_convert::emit_lm_head`, which the Small converter calls too.
    fn unembed(&mut self, consumed: &mut HashSet<String>) -> Result<()> {
        let name = "model.llm.unembed.weight";
        let (b, sh) = self.src.typed(name, Dtype::Bf16)?;
        let rows = VocabRows {
            vocab_size: self.t.vocab_size,
            unpadded: self.t.unpadded_vocab_size,
            hidden: self.t.hidden_size,
        };
        emit_lm_head(&mut self.w, name, (b, sh), rows)?;
        consumed.insert(name.to_string());
        Ok(())
    }

    /// Every remaining TEXT name, renamed and copied verbatim at its source dtype. Excluded
    /// (vision) names are skipped; expert/query/unembed names are already consumed.
    fn verbatim(&mut self, census: &Census, consumed: &mut HashSet<String>) -> Result<()> {
        let names: Vec<String> = self.src.names().iter().map(|s| s.to_string()).collect();
        for name in &names {
            if consumed.contains(name) {
                continue;
            }
            let (b, dt, sh) = self.src.raw(name)?;
            let f = census.family_of(name, dt, sh)?;
            if f.status.is_excluded() {
                continue;
            }
            let target = rename_verbatim(name)?;
            // The two engine-layout fixes (exact bf16→f32 widening of the f32-consumed
            // norms/clip-scales/sconv weights, and the `rel_proj` transpose for the m=heads
            // GEMM) live in `inkling_convert::emit_verbatim`, which the Small converter calls
            // too — one definition, so a fix to either lands in both artifacts. The `_fmt`
            // wrapper adds `--fp8`'s dense-trunk rewrite and delegates every other tensor, so
            // one loop still serves both artifact shapes.
            emit_verbatim_fmt(&mut self.w, target, (b, dt, sh), self.fp8_block)?;
            consumed.insert(name.clone());
        }
        Ok(())
    }

    fn copy_renamed(
        &mut self,
        consumed: &mut HashSet<String>,
        name: &str,
        target: String,
    ) -> Result<()> {
        let (b, dt, sh) = self.src.raw(name)?;
        consumed.insert(name.to_string());
        self.w.add(target, dt, TensorData::new(sh.to_vec(), b));
        Ok(())
    }
}

// ── `--mxfp8`: the routed experts as per-layer files ──────────────────────
//
// The default mode above writes bf16 slabs into `resident.safetensors`. This one writes the
// same 24 layers as `L{ll}.mxfp8`, which is the container the streaming engine opens, and the
// source decides which of its two paths a layer takes: 20..24 arrive at MXFP8 already and are
// repacked bit-for-bit, 0..20 arrive at NVFP4 and are re-quantized row-wise.

/// The `--mxfp8` expert pass: read the single text shard, write one layer file per layer.
struct Mxfp8Layers<'a> {
    src: &'a Safetensors,
    t: &'a InklingTextConfig,
    out_dir: &'a str,
}

impl<'a> Mxfp8Layers<'a> {
    /// Every layer's file, marking each layer's source tensors consumed as it goes — the same
    /// eight-or-four per layer the default mode spends, so the census closure is mode-agnostic.
    fn layers(&self, consumed: &mut HashSet<String>) -> Result<()> {
        for l in 0..self.t.num_hidden_layers {
            self.layer(consumed, l)?;
        }
        Ok(())
    }

    /// One layer, dispatched on the SOURCE dtype of its `w13_weight`: U8 is NVFP4 (an `amax`
    /// rides beside it) and F8_E4M3 is MXFP8. Anything else is refused by name rather than
    /// guessed at — the default mode makes the same match for the same reason.
    fn layer(&self, consumed: &mut HashSet<String>, l: usize) -> Result<()> {
        let w13 = format!("{}w13_weight", expert_prefix(l));
        let (_, dt, _) = self.src.raw(&w13)?;
        match dt {
            Dtype::F8E4M3 => self.repack_native(consumed, l),
            Dtype::U8 => self.requantize_nvfp4(consumed, l),
            other => bail!(
                "{w13}: expert weight dtype {other:?} is neither NVFP4 (U8) nor MXFP8 \
                 (F8_E4M3) — --mxfp8 has no path for it"
            ),
        }
    }

    /// A layer the checkpoint already holds at MXFP8: mark its four tensors and let
    /// [`Mxfp8Layer`] write AND verify it. Byte identity is the claim `--verify` there demands,
    /// which is exactly right for a repack that fits nothing and re-rounds nothing.
    fn repack_native(&self, consumed: &mut HashSet<String>, l: usize) -> Result<()> {
        mark_mxfp8_consumed(consumed, &expert_prefix(l));
        let inputs = LayerInputs {
            tool: "convert_inkling",
            out_dir: self.out_dir,
            src: self.src,
        };
        let shape = LayerShape {
            n_experts: self.t.n_routed_experts,
            expert_in: self.t.hidden_size,
            moe_inter: self.t.intermediate_size,
        };
        Mxfp8Layer::new(inputs, l, shape).mode(true, true).run()
    }

    /// A layer the checkpoint holds at NVFP4: decode one row at a time and re-quantize it into
    /// the `.mxfp8` block layout, streamed in windows by [`write_expert_layer`] — the same
    /// bounded-memory, atomic writer every other converter reaches.
    fn requantize_nvfp4(&self, consumed: &mut HashSet<String>, l: usize) -> Result<()> {
        let plane = Nvfp4Layer::load(self.src, self.t, l)?;
        plane.mark_consumed(consumed);
        let (h, i, n) = (
            self.t.hidden_size,
            self.t.intermediate_size,
            self.t.n_routed_experts,
        );
        let path = format!("{}/L{l:02}.mxfp8", self.out_dir);
        let stride = mxfp8_expert_stride(h, i);
        let dims = LayerDims {
            layer: l,
            n_experts: n,
            expert_in: h,
            moe_inter: i,
            stride,
        };
        let header = ExpertHeader::new(MXFP8_MAGIC, dims).to_bytes();
        let blocks = ExpertBlocks {
            stride,
            bytes: mxfp8_expert_bytes(h, i),
            count: n,
        };
        write_expert_layer(
            ExpertFile {
                path: &path,
                header: &header,
            },
            blocks,
            LAYER_WINDOW,
            |e, slot| plane.pack_expert(e, slot),
        )
        .with_context(|| format!("write layer {l} (NVFP4 → MXFP8 re-quantization)"))?;
        eprintln!("convert_inkling: wrote {path} ({n} experts re-quantized from NVFP4)");
        Ok(())
    }
}

/// Mark one native-MXFP8 layer's four source tensors consumed (shared by `--mxfp8` and `--f4`).
fn mark_mxfp8_consumed(consumed: &mut HashSet<String>, prefix: &str) {
    for suffix in [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ] {
        consumed.insert(format!("{prefix}{suffix}"));
    }
}

/// The checkpoint's tensor-name prefix for one layer's routed experts — one spelling, shared by
/// the dtype sniff, the consumption marks and the per-projection reads, so a rename cannot make
/// the three disagree about which tensors a layer owns.
fn expert_prefix(layer: usize) -> String {
    format!("model.llm.layers.{layer}.mlp.experts.")
}

/// One projection's NVFP4 source: the packed weight plane, its e4m3 block-scale grid, and the
/// per-expert `amax` / `outer_scale_denominator` pair that is that projection's factor.
struct Nvfp4Plane<'a> {
    weight: &'a [u8],
    scale: &'a [u8],
    amax: Vec<f32>,
    denom: f32,
    /// The `[E, rows, cols_packed]` tail of this projection's shape, kept so a row decode
    /// cannot be handed another projection's width.
    rows: usize,
    cols_packed: usize,
}

impl<'a> Nvfp4Plane<'a> {
    /// Read one projection's four tensors — `<base>weight`, `scale`, `amax`,
    /// `outer_scale_denominator` — and CHECK them against `[E, rows, cols_packed]`.
    ///
    /// Checked once per layer, not per row: the shapes decide every row offset below, and a
    /// transposed or mis-blocked source would copy the right number of bytes in the wrong order
    /// and fail no length check anywhere downstream.
    fn load(src: &'a Safetensors, base: &str, want: [usize; 3]) -> Result<Self> {
        let (weight, wsh) = src.typed(base, Dtype::U8)?;
        let (scale, ssh) = src.typed(&format!("{base}_scale"), Dtype::F8E4M3)?;
        let (ab, ash) = src.typed(&format!("{base}_amax"), Dtype::F32)?;
        let (db, _) = src.typed(&format!("{base}_outer_scale_denominator"), Dtype::F32)?;
        let [n, rows, cols_packed] = want;
        ensure!(
            wsh == want,
            "{base}: {wsh:?} != {want:?} (U8, two e2m1 codes per byte)"
        );
        ensure!(
            ssh == [n, rows, cols_packed / 8],
            "{base}_scale: {ssh:?} != [{n}, {rows}, {}] (one e4m3 scale per 16 values)",
            cols_packed / 8
        );
        ensure!(
            ash == [n] && ab.len() == 4 * n,
            "{base}_amax: {ash:?} / {} bytes for {n} experts",
            ab.len()
        );
        let denom = read_f32(db)
            .first()
            .copied()
            .with_context(|| format!("{base}_outer_scale_denominator is empty"))?;
        ensure!(
            denom > 0.0 && denom.is_finite(),
            "{base}_outer_scale_denominator: {denom} must be positive and finite"
        );
        Ok(Self {
            weight,
            scale,
            amax: read_f32(ab),
            denom,
            rows,
            cols_packed,
        })
    }

    /// The [`Nvfp4`] view of this plane — the row walk itself stays in `inkling_dequant`, so the
    /// `(code · block_scale) · factor` association exists once.
    ///
    /// Written spans-then-factor rather than in `Nvfp4`'s declaration order: the two spans and the
    /// row geometry are what the SOURCE shard fixes, while `amax`/`denom` are what this plane
    /// derived from that shard, and the literal reads in that order.
    fn view(&self) -> Nvfp4<'_> {
        Nvfp4 {
            weight: self.weight,
            scale: self.scale,
            rows: self.rows,
            cols_packed: self.cols_packed,
            amax: &self.amax,
            denom: self.denom,
        }
    }

    /// Decode `row` of `expert` into `out`, exactly `cols_packed * 2` values long.
    fn decode_into(&self, (expert, row): (usize, usize), out: &mut [f32]) -> Result<()> {
        self.view().decode_expert_row(expert, row, out)
    }
}

/// One layer's NVFP4 routed experts: `w13` (gate/up interleaved) and `w2` (down), each with its
/// own scale grid and its own per-expert factor.
struct Nvfp4Layer<'a> {
    /// [`expert_prefix`]'s value for this layer — the sole spelling of its tensor names.
    prefix: String,
    w13: Nvfp4Plane<'a>,
    w2: Nvfp4Plane<'a>,
    n_experts: usize,
    hidden: usize,
    inter: usize,
}

/// The eight tensors one NVFP4 layer spends, in the order its projections consume them.
const NVFP4_TENSORS: [&str; 8] = [
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_amax",
    "w13_weight_outer_scale_denominator",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_amax",
    "w2_weight_outer_scale_denominator",
];

impl<'a> Nvfp4Layer<'a> {
    /// Read and check both projections of one layer.
    ///
    /// `w13` is `[E, 2·inter, hidden/2]` packed and `w2` is `[E, hidden, inter/2]`, because the
    /// block is ENTERED at `hidden` — the same `(expert_in, moe_inter)` the `.mxfp8` offsets and
    /// the O_DIRECT stride are built from.
    fn load(src: &'a Safetensors, t: &InklingTextConfig, layer: usize) -> Result<Self> {
        let prefix = expert_prefix(layer);
        let (e, h, i) = (t.n_routed_experts, t.hidden_size, t.intermediate_size);
        let w13 = Nvfp4Plane::load(src, &format!("{prefix}w13_weight"), [e, 2 * i, h / 2])?;
        let w2 = Nvfp4Plane::load(src, &format!("{prefix}w2_weight"), [e, h, i / 2])?;
        Ok(Self {
            prefix,
            w13,
            w2,
            n_experts: e,
            hidden: h,
            inter: i,
        })
    }

    /// Mark this layer's eight source tensors consumed. The count is the default mode's, which
    /// is what keeps `consumed.len() == sum.text_tensors` true in both modes.
    fn mark_consumed(&self, consumed: &mut HashSet<String>) {
        for suffix in NVFP4_TENSORS {
            consumed.insert(format!("{}{suffix}", self.prefix));
        }
    }

    /// Re-quantize expert `e` into one MXFP8 block of `mxfp8_expert_bytes` when called by
    /// [`write_expert_layer`], writing every byte of it.
    ///
    /// The row order is the DE-INTERLEAVE, exactly as the native repack applies it: `w13`'s even
    /// rows are gate and its odd rows up (CONV:365–372), and `w2`'s rows are down in place.
    /// Treating `w13` as a `[gate; up]` concatenation is trap T2 — a file that opens, streams and
    /// decodes the gate against the up weights, which no length check can see.
    fn pack_expert(&self, e: usize, slot: &mut [u8]) -> Result<()> {
        let (h, i) = (self.hidden, self.inter);
        let bytes = mxfp8_expert_bytes(h, i);
        ensure!(
            e < self.n_experts && slot.len() == bytes,
            "{}expert {e}: {}-byte slot for {} experts of an {bytes}-byte MXFP8 block",
            self.prefix,
            slot.len(),
            self.n_experts
        );
        let off = mxfp8_slot_offsets(h, i);
        let (g_in, g_out) = (mxfp8_groups(h), mxfp8_groups(i));
        let mut row = vec![0f32; h.max(i)];
        for j in 0..i {
            self.w13.decode_into((e, 2 * j), &mut row[..h])?;
            put_mxfp8_row(slot, (off[0] + j * h, off[1] + j * g_in), &row[..h])?;
            self.w13.decode_into((e, 2 * j + 1), &mut row[..h])?;
            put_mxfp8_row(slot, (off[2] + j * h, off[3] + j * g_in), &row[..h])?;
        }
        for r in 0..h {
            self.w2.decode_into((e, r), &mut row[..i])?;
            put_mxfp8_row(slot, (off[4] + r * i, off[5] + r * g_out), &row[..i])?;
        }
        Ok(())
    }
}

/// Quantize one row into its slot: `at.0` is the packed span's start, `at.1` the ue8m0 grid's.
/// The two offsets come from [`mxfp8_slot_offsets`], the same function the streaming pool points
/// its descriptor at, so a row written here is a row the kernel reads.
///
/// The split is `split_at_mut` rather than two index borrows of `slot`, because a projection's
/// scale grid FOLLOWS its packed rows (`off[1] = off[0] + rows × cols`) and the two spans are
/// therefore disjoint with the packed one first — the one fact `split_at_mut(at.1)` relies on.
fn put_mxfp8_row(slot: &mut [u8], at: (usize, usize), row: &[f32]) -> Result<()> {
    let (packed, scales) = at;
    let n = row.len();
    let (head, tail) = slot.split_at_mut(scales);
    quantize_mxfp8_row(
        row,
        &mut head[packed..packed + n],
        &mut tail[..mxfp8_groups(n)],
    )
}

/// `manifest.json` = the source config VERBATIM (wrapper and all) + the `format` section +
/// `inkling_source` provenance, plus exactly one `<fmt>_source` block when the experts are
/// streamed (the loader's sole source for which layers exist).
///
/// **The two source blocks are exclusive and that is load-bearing**: the loader sniffs
/// `mxfp8_source` before it sniffs anything else, so an artifact carrying both would commit to
/// `.mxfp8` layer files an `--f4` run never wrote.
///
/// `fp8_block` in `format` is inert (Glimmer's precedent); `--fp8`'s presence is recorded in
/// `inkling_source.dense`, which nothing reads — it is there for a human six months later.
fn build_manifest(src_dir: &str, mode: Mode, n_layers: usize) -> Result<serde_json::Value> {
    let mut manifest = FormatMeta::manifest_from_config(src_dir, FP8_BLOCK)?;
    let experts = match (mode.f4, mode.mxfp8) {
        (true, _) => "f4-streamed",
        (false, true) => "mxfp8-streamed",
        (false, false) => "dequantized-to-bf16",
    };
    manifest["inkling_source"] = serde_json::json!({
        "tool": "convert_inkling",
        "src": src_dir,
        "member": "inkling2_loop",
        "experts": experts,
    });
    if mode.fp8 {
        manifest["inkling_source"]["dense"] = serde_json::json!("fp8-block128");
    }
    if mode.mxfp8 {
        manifest["mxfp8_source"] = mxfp8_source("convert_inkling", src_dir, 0..n_layers);
        // The shared producer's chain names ONE source encoding; Mini is mixed (0..20 NVFP4
        // re-quantized, 20..24 repacked), so the truthful string is stamped here.
        manifest["mxfp8_source"]["chain"] = serde_json::json!("nvfp4+fp8 -> fp8 (quantize/repack)");
    }
    if mode.f4 {
        manifest["f4_source"] = f4_source("convert_inkling", src_dir, 0..n_layers);
        // Same mixed-source argument as the `.mxfp8` chain above: `f4_source`'s literal says
        // `fp4->fp4 (repack)`, which is true of V4 and K3 and false of this checkpoint.
        manifest["f4_source"]["chain"] = serde_json::json!("nvfp4+fp8 -> fp4 (quantize)");
    }
    Ok(manifest)
}

fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        !(a.mxfp8 && a.f4),
        "--mxfp8 and --f4 both claim the routed experts, and they are two different containers: \
         the artifact would hold one set of layer files and a manifest promising the other. Pick \
         one."
    );
    let mode = Mode {
        mxfp8: a.mxfp8,
        f4: a.f4,
        fp8: a.fp8,
    };
    let dirs = preflight(rivoli_artifact::format::ArtifactDirs {
        src: &a.src_dir,
        out: &a.out_dir,
    })?;
    let cfg = Inkling2LoopConfig::load(&a.src_dir)?;
    let census = Census::load()?;
    let src = Safetensors::open_file(&format!("{}/{SHARD}", a.src_dir))?;
    let sum = confront_shard(&src, &census)?;
    std::fs::create_dir_all(&a.out_dir)?;
    let n = run_convert(
        &src,
        &cfg.text,
        (&census, &a.out_dir),
        (mode, sum.text_tensors),
    )?;
    copy_tiktoken(dirs)?;
    finish_artifact(
        "convert_inkling",
        dirs,
        &build_manifest(&a.src_dir, mode, cfg.text.num_hidden_layers)?,
        INKLING_AUX,
    )?;
    let label = match (mode.f4, mode.mxfp8) {
        (true, _) => "f4",
        (false, true) => "mxfp8-streamed",
        (false, false) => "bf16",
    };
    eprintln!(
        "convert_inkling: done — {} → {} ({n} TEXT tensors, experts {label})",
        a.src_dir, a.out_dir
    );
    Ok(())
}

fn confront_shard(
    src: &Safetensors,
    census: &Census,
) -> Result<rivoli_artifact::census::inkling::Summary> {
    let names = src.names();
    let mut tensors: Vec<(&str, Dtype, &[usize], u64)> = Vec::with_capacity(names.len());
    for n in &names {
        let (b, dt, sh) = src.raw(n)?;
        tensors.push((*n, dt, sh, b.len() as u64));
    }
    let sum = census.confront(&tensors)?;
    eprintln!("convert_inkling: {}", sum.line());
    Ok(sum)
}

fn run_convert(
    src: &Safetensors,
    t: &InklingTextConfig,
    (census, out_dir): (&Census, &str),
    (mode, want): (Mode, u64),
) -> Result<usize> {
    let mut consumed: HashSet<String> = HashSet::new();
    let mut conv = Conv {
        src,
        t,
        w: SafeWriter::new(),
        fp8_block: mode.fp8.then_some(FP8_BLOCK),
    };
    match (mode.f4, mode.mxfp8) {
        (true, _) => f4_pass::Pass { src, t, out_dir }.layers(&mut consumed)?,
        (false, true) => Mxfp8Layers { src, t, out_dir }.layers(&mut consumed)?,
        (false, false) => conv.experts(&mut consumed)?,
    }
    conv.queries(&mut consumed)?;
    conv.unembed(&mut consumed)?;
    conv.verbatim(census, &mut consumed)?;
    ensure!(
        consumed.len() as u64 == want,
        "consumed {} TEXT tensors but the census counts {want} — a family was dropped or double-counted",
        consumed.len(),
    );
    conv.w.write(&format!("{out_dir}/resident.safetensors"))?;
    Ok(consumed.len())
}
