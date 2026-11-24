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
    ArtifactDirs, Dtype, FormatMeta, SafeWriter, Safetensors, TensorData,
};
use rivoli_artifact::format::{finish_artifact, require_aux};
use rivoli_artifact::inkling_config::Inkling2LoopConfig;
use rivoli_artifact::inkling_dequant::{Mxfp8, Nvfp4, merge_message_queries};
use rivoli_artifact::quant::{FP8_BLOCK, read_f32};
use rivoli_core::num::bf16_to_f32;
use std::collections::HashSet;

/// The single text shard. `mtp.safetensors` is NEVER named here — it is a separate file the text
/// converter does not open. `open_file` (not `open_dir`) so the directory's `mtp.safetensors` is
/// not mmap'd by accident.
const SHARD: &str = "model-00001-of-00001.safetensors";

/// The flat aux files copied beside the weights. The tiktoken rank file ships NESTED
/// (`tiktoken/tokenizer.model`) and the loader reads it FLAT (`tiktoken.model`), so it is copied
/// and renamed explicitly in `main`, not through this list (which `finish_artifact` copies
/// same-name).
const AUX: &[&str] = &[
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
];

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

/// The verbatim layer-suffix renames (checkpoint suffix after `model.llm.layers.{L}.` → target
/// suffix after `model.layers.{L}.`). A lookup table, not a branch chain: the mapping IS data,
/// and `tests/inkling_convert.rs` confronts the emitted names against the first-party census.
const LAYER_RENAMES: &[(&str, &str)] = &[
    ("attn.attn_sink_proj.weight", "self_attn.sink_proj.weight"),
    ("attn.gate_attn.weight", "self_attn.o_proj.gate_proj.weight"),
    (
        "attn.k_sconv.static_w",
        "self_attn.k_proj.sconv.static_weight",
    ),
    (
        "attn.k_sconv.weight",
        "self_attn.k_proj.sconv.conv1d.weight",
    ),
    (
        "attn.k_sconv.weight_proj",
        "self_attn.k_proj.sconv.weight_proj.weight",
    ),
    ("attn.q_l2_clip.scale", "self_attn.q_proj.norm.scale"),
    (
        "attn.rel_l2_clip.scale",
        "self_attn.relative_position_bias.norm.scale",
    ),
    (
        "attn.rel_logits_proj.proj",
        "self_attn.relative_position_bias.proj",
    ),
    (
        "attn.v_sconv.static_w",
        "self_attn.v_proj.sconv.static_weight",
    ),
    (
        "attn.v_sconv.weight",
        "self_attn.v_proj.sconv.conv1d.weight",
    ),
    (
        "attn.v_sconv.weight_proj",
        "self_attn.v_proj.sconv.weight_proj.weight",
    ),
    ("attn.wk_dv.weight", "self_attn.k_proj.linear.weight"),
    ("attn.wo_ud.weight", "self_attn.o_proj.linear.weight"),
    ("attn.wq_du.weight", "self_attn.q_proj.linear.weight"),
    (
        "attn.wr_du.weight",
        "self_attn.relative_position_bias.linear.weight",
    ),
    ("attn.wv_dv.weight", "self_attn.v_proj.linear.weight"),
    ("attn_norm.weight", "input_layernorm.weight"),
    (
        "attn_sconv.weight",
        "post_attention_convolution.conv1d.weight",
    ),
    (
        "controllers.attn.down_weight",
        "q_message_gate.down_proj.weight",
    ),
    (
        "controllers.attn.forget_weight",
        "q_message_gate.forget_gate_proj.weight",
    ),
    (
        "controllers.attn.gain_weight",
        "q_message_gate.message_gate_proj.weight",
    ),
    (
        "controllers.k.down_weight",
        "k_message_gate.down_proj.weight",
    ),
    (
        "controllers.k.forget_weight",
        "k_message_gate.forget_gate_proj.weight",
    ),
    (
        "controllers.k.gain_weight",
        "k_message_gate.message_gate_proj.weight",
    ),
    (
        "controllers.mlp.down_weight",
        "mlp_message_gate.down_proj.weight",
    ),
    (
        "controllers.mlp.forget_weight",
        "mlp_message_gate.forget_gate_proj.weight",
    ),
    (
        "controllers.mlp.gain_weight",
        "mlp_message_gate.message_gate_proj.weight",
    ),
    (
        "controllers.v.down_weight",
        "v_message_gate.down_proj.weight",
    ),
    (
        "controllers.v.forget_weight",
        "v_message_gate.forget_gate_proj.weight",
    ),
    (
        "controllers.v.gain_weight",
        "v_message_gate.message_gate_proj.weight",
    ),
    ("k_input_norm.weight", "k_input_layernorm.weight"),
    ("mlp.gate.bias", "mlp.gate.e_score_correction_bias"),
    ("mlp.gate.weight", "mlp.gate.weight"),
    ("mlp_norm.weight", "post_attention_layernorm.weight"),
    ("mlp_sconv.weight", "post_mlp_convolution.conv1d.weight"),
    ("q_input_norm.weight", "q_input_layernorm.weight"),
    ("v_input_norm.weight", "v_input_layernorm.weight"),
];

fn layer_target(layer: usize, suffix: &str) -> Option<String> {
    LAYER_RENAMES
        .iter()
        .find(|(s, _)| *s == suffix)
        .map(|(_, t)| format!("model.layers.{layer}.{t}"))
}

/// The rename for a verbatim (non-expert, non-query, non-unembed) TEXT name.
fn rename_verbatim(name: &str) -> Result<String> {
    let rest = name
        .strip_prefix("model.llm.")
        .with_context(|| format!("{name:?}: not under model.llm."))?;
    if let Some(lr) = rest.strip_prefix("layers.") {
        let (ls, suffix) = lr.split_once('.').context("layer name has no suffix")?;
        let layer: usize = ls
            .parse()
            .with_context(|| format!("{name:?}: bad layer index"))?;
        layer_target(layer, suffix)
            .with_context(|| format!("{name:?}: no layer rename for {suffix:?}"))
    } else {
        match rest {
            "embed.weight" => Ok("model.embed_tokens.weight".to_string()),
            "embed_norm.weight" => Ok("model.embed_tokens.embed_norm.weight".to_string()),
            "norm.weight" => Ok("model.norm.weight".to_string()),
            _ => bail!("{name:?}: no global rename"),
        }
    }
}

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
    /// layer `I`, `output_query` is `final_input`, and the two `middle_cycle` banks merge into
    /// `message_input.query` in `(q,k,v,mlp)` order (trap T13).
    fn queries(&mut self, consumed: &mut HashSet<String>) -> Result<()> {
        let names: Vec<String> = self.src.names().iter().map(|s| s.to_string()).collect();
        let mut q_mlp = None;
        let mut kv = None;
        for name in &names {
            if let Some(idx) = name.strip_prefix("model.llm.attn_res.attn_queries.") {
                let i: usize = idx
                    .parse()
                    .with_context(|| format!("{name:?}: bad query index"))?;
                self.copy_renamed(
                    consumed,
                    name,
                    format!("model.layers.{}.attention_input.query", i + 1),
                )?;
            } else if let Some(idx) = name.strip_prefix("model.llm.attn_res.mlp_queries.") {
                let i: usize = idx
                    .parse()
                    .with_context(|| format!("{name:?}: bad query index"))?;
                self.copy_renamed(consumed, name, format!("model.layers.{i}.mlp_input.query"))?;
            } else if name == "model.llm.attn_res.output_query" {
                self.copy_renamed(consumed, name, "model.final_input.query".to_string())?;
            } else if name == "model.llm.middle_cycle_cross_pass_queries" {
                q_mlp = Some(read_f32(self.src.typed(name, Dtype::F32)?.0));
                consumed.insert(name.clone());
            } else if name == "model.llm.middle_cycle_cross_pass_kv_queries" {
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

    /// `unembed` → `lm_head`, keeping the first `unpadded_vocab_size` rows (LOAD:856, trap T14).
    fn unembed(&mut self, consumed: &mut HashSet<String>) -> Result<()> {
        let name = "model.llm.unembed.weight";
        let (b, sh) = self.src.typed(name, Dtype::Bf16)?;
        let h = self.t.hidden_size;
        let keep = self.t.unpadded_vocab_size;
        ensure!(
            sh == [self.t.vocab_size, h],
            "{name}: shape {sh:?} != [{}, {h}]",
            self.t.vocab_size
        );
        self.w.add(
            "lm_head.weight",
            Dtype::Bf16,
            TensorData::new(vec![keep, h], &b[..keep * h * 2]),
        );
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
            self.emit_verbatim(target, (b, dt, sh))?;
            consumed.insert(name.clone());
        }
        Ok(())
    }

    /// Verbatim emit with two engine-layout fixes (2026-11-24): the f32-consumed norms/clip-scales/
    /// sconv weights are widened bf16→f32 (exact — the rmsnorm/clip/sconv launchers take f32), and
    /// `rel_proj` is transposed `[d_rel, rel_extent]`→`[rel_extent, d_rel]` so the m=heads rel_logits
    /// GEMM (`gemm_bf16` computes x·wᵀ) reads it without a runtime transpose.
    ///
    /// The untouched branch BORROWS the source mmap (`b` is `&'a [u8]`) rather than copying it:
    /// the resident set is 8.6 GB of bf16 and every verbatim tensor is the checkpoint's own bytes,
    /// so a `to_vec()` here is host RAM spent for nothing.
    fn emit_verbatim(&mut self, target: String, raw: (&'a [u8], Dtype, &[usize])) -> Result<()> {
        let (b, dt, sh) = raw;
        if target.ends_with("relative_position_bias.proj") && dt == Dtype::Bf16 {
            ensure!(
                sh.len() == 2 && b.len() == sh[0] * sh[1] * 2,
                "{target}: expected 2-D bf16 shape, got {sh:?} ({} B)",
                b.len()
            );
            let bytes = transpose_bf16(b, sh[0], sh[1]);
            self.w.add(
                target,
                Dtype::Bf16,
                TensorData::new(vec![sh[1], sh[0]], bytes),
            );
            return Ok(());
        }
        if wants_f32(&target) && dt == Dtype::Bf16 {
            let bytes = widen_bf16_f32(b);
            self.w
                .add(target, Dtype::F32, TensorData::new(sh.to_vec(), bytes));
            return Ok(());
        }
        self.w.add(target, dt, TensorData::new(sh.to_vec(), b));
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

/// The engine consumes these as f32 (the rmsnorm/clip/sconv launchers take f32 operands) but the
/// checkpoint stores them bf16; the widening is exact. `sconv.static_weight` is the dynamic
/// sconv's per-layer `static_w` — the full-layer path (`pin.rs`) reads it as f32, so widening it
/// here keeps the artifact's dtype a function of the engine's operand type, not of the layer.
fn wants_f32(target: &str) -> bool {
    const F32_SUFFIXES: &[&str] = &[
        "layernorm.weight",
        "norm.scale",
        "conv1d.weight",
        "sconv.static_weight",
        "model.norm.weight",
        "embed_norm.weight",
    ];
    F32_SUFFIXES.iter().any(|s| target.ends_with(s))
}

/// `[r, c]` bf16 → `[c, r]` bf16 — the rel_proj transpose for the m=heads GEMM (x·wᵀ).
fn transpose_bf16(b: &[u8], r: usize, c: usize) -> Vec<u8> {
    let v: Vec<u16> = b
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect();
    let mut t = vec![0u16; r * c];
    for (idx, &val) in v.iter().enumerate() {
        let (i, j) = (idx / c, idx % c);
        t[j * r + i] = val;
    }
    let mut bytes = Vec::with_capacity(t.len() * 2);
    for x in t {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

/// Exact bf16 → f32 widening, re-serialized little-endian.
fn widen_bf16_f32(b: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(b.len() * 2);
    for x in b.chunks_exact(2) {
        let v = bf16_to_f32(u16::from_le_bytes([x[0], x[1]]));
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
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
    require_aux(&src_dir, AUX)?;
    let dirs = ArtifactDirs {
        src: &src_dir,
        out: &out_dir,
    };
    SafeWriter::refuse_writing_into_source(&dirs)?;
    let tiktoken_src = format!("{src_dir}/tiktoken/tokenizer.model");
    require_nonempty(&tiktoken_src, "the tiktoken vocabulary the runtime reads")?;
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
    std::fs::copy(&tiktoken_src, format!("{out_dir}/tiktoken.model"))
        .with_context(|| format!("copy {tiktoken_src} into the artifact"))?;
    require_nonempty(
        &format!("{out_dir}/tiktoken.model"),
        "the copied tiktoken rank file",
    )?;
    finish_artifact("convert_inkling", dirs, &build_manifest(&src_dir)?, AUX)?;
    eprintln!(
        "convert_inkling: done — {src_dir} → {out_dir} ({} TEXT tensors consumed)",
        consumed.len()
    );
    Ok(())
}
/// A file must exist and be non-empty. Applied to the tiktoken rank file at both ends (source and
/// artifact), which `finish_artifact` does not copy — it is nested-renamed. One helper so the two
/// checks are not a cloned `ensure!` pair (jscpd).
fn require_nonempty(path: &str, what: &str) -> Result<()> {
    ensure!(
        std::fs::metadata(path).is_ok_and(|m| m.len() > 0),
        "{path} is missing or empty — {what}"
    );
    Ok(())
}
