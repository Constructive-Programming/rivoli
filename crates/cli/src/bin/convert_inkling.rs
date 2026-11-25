//! `convert_inkling` — the Inkling 2 Mini checkpoint → a rivoli text artifact.
//!
//! **Text-only v1, both members' shared stack, Mini first.** The converter consumes the 822 TEXT
//! tensors of the single shard, excludes the 422 `model.visual.*` tensors BY NAME (counted, never
//! written), and never opens `mtp.safetensors` (a separate file, refused by `Flag::Mtp`). The
//! census closure (`census::inkling::Census::confront`) classifies every one of the 1244 names
//! before a byte is converted; a name matching no family is a hard error.
//!
//! **The experts are DEQUANTIZED to bf16, not requantized.** Mini's first-party loop path
//! (`Inkling2LoopDequantizeExperts`, LOAD:772) dequantizes on load; the artifact stores that
//! result. NVFP4 (layers 0–19) and MXFP8 (20–23) are decoded by `inkling_dequant`, whose oracle is
//! bit-compared against the transformers op (`tests/inkling_dequant.rs`). `w13` is de-interleaved
//! into `[gate_all; up_all]`; the two message-query banks merge to `(q,k,v,mlp)`; `unembed` keeps
//! the first 200058 rows. The output names are the first-party text model's state-dict names
//! (`text-model-names.tsv`, 693 params), gated by `tests/inkling_convert.rs`.
//!
//! Plan of record: `docs/investigations/inkling2-port.md` (stage S2).

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use rivoli_artifact::census::inkling::Census;
use rivoli_artifact::format::{
    Dtype, FormatMeta, SafeWriter, Safetensors, TensorData, finish_artifact,
};
use rivoli_artifact::inkling_config::Inkling2LoopConfig;
use rivoli_artifact::inkling_convert::{
    INKLING_AUX, VocabRows, copy_tiktoken, emit_lm_head, emit_verbatim, preflight, query_target,
    rename_verbatim,
};
use rivoli_artifact::inkling_dequant::{Mxfp8, Nvfp4, merge_message_queries};
use rivoli_artifact::quant::{FP8_BLOCK, read_f32};
use std::collections::HashSet;

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
    about = "Inkling 2 Mini checkpoint → a rivoli text artifact (NVFP4/MXFP8 experts dequantized \
             to bf16, vision excluded by name, MTP never opened)"
)]
struct Args {
    /// The checkpoint directory: config.json, model-00001-of-00001.safetensors, tiktoken/, and the
    /// tokenizer files copied into the artifact.
    src_dir: String,

    /// Artifact directory to write. Created if absent.
    out_dir: String,
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
    t: &'a rivoli_artifact::inkling_config::InklingTextConfig,
    w: SafeWriter<'a>,
}

impl<'a> Conv<'a> {
    /// Every layer's two expert projections, dequantized to bf16 and (for w13) de-interleaved.
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
            // too — one definition, so a fix to either lands in both artifacts.
            emit_verbatim(&mut self.w, target, (b, dt, sh))?;
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

/// `manifest.json` = the source config VERBATIM (wrapper and all) + the `format` section +
/// `inkling_source` provenance. `fp8_block` is inert on this bf16 artifact (Glimmer's precedent).
fn build_manifest(src_dir: &str) -> Result<serde_json::Value> {
    let mut manifest = FormatMeta::manifest_from_config(src_dir, FP8_BLOCK)?;
    manifest["inkling_source"] = serde_json::json!({
        "tool": "convert_inkling",
        "src": src_dir,
        "member": "inkling2_loop",
        "experts": "dequantized-to-bf16",
    });
    Ok(manifest)
}

fn main() -> Result<()> {
    let Args { src_dir, out_dir } = Args::parse();
    // The aux, write-into-source and tiktoken refusals, shared with the Small converter: all
    // three must land before the weight pass, not after it.
    let dirs = preflight(rivoli_artifact::format::ArtifactDirs {
        src: &src_dir,
        out: &out_dir,
    })?;
    let cfg = Inkling2LoopConfig::load(&src_dir)?;
    let census = Census::load()?;
    let src = Safetensors::open_file(&format!("{src_dir}/{SHARD}"))?;

    // The exhaustive census closure before a byte is converted.
    let names = src.names();
    let mut tensors: Vec<(&str, Dtype, &[usize], u64)> = Vec::with_capacity(names.len());
    for n in &names {
        let (b, dt, sh) = src.raw(n)?;
        tensors.push((*n, dt, sh, b.len() as u64));
    }
    let sum = census.confront(&tensors)?;
    eprintln!("convert_inkling: {}", sum.line());

    std::fs::create_dir_all(&out_dir)?;
    let mut consumed: HashSet<String> = HashSet::new();
    let mut conv = Conv {
        src: &src,
        t: &cfg.text,
        w: SafeWriter::new(),
    };
    conv.experts(&mut consumed)?;
    conv.queries(&mut consumed)?;
    conv.unembed(&mut consumed)?;
    conv.verbatim(&census, &mut consumed)?;
    ensure!(
        consumed.len() as u64 == sum.text_tensors,
        "consumed {} TEXT tensors but the census counts {} — a family was dropped or double-counted",
        consumed.len(),
        sum.text_tensors
    );
    conv.w.write(&format!("{out_dir}/resident.safetensors"))?;

    // The tiktoken rank file: nested in the source, flat in the artifact (the loader's name).
    copy_tiktoken(dirs)?;
    finish_artifact(
        "convert_inkling",
        dirs,
        &build_manifest(&src_dir)?,
        INKLING_AUX,
    )?;
    eprintln!(
        "convert_inkling: done — {src_dir} → {out_dir} ({} TEXT tensors consumed)",
        consumed.len()
    );
    Ok(())
}
