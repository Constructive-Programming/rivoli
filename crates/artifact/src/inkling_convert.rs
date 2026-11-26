//! The Inkling 2 checkpoint → rivoli rename and transform rules, shared by BOTH member
//! converters (`convert_inkling` for Mini, `convert_inkling_small` for Small-MXFP8).
//!
//! **Why this is a module and not a copy.** The two members are one text stack: the same
//! `model.llm.` root, the same layer-suffix vocabulary, the same `attn_res` query banks, the same
//! `unembed` unpad, the same three engine-layout fixes. What differs is what the EXPERTS are
//! (Mini dequantizes NVFP4/MXFP8 to bf16, Small repacks MXFP8 in place) and what the queries
//! become (Mini also merges the two `middle_cycle` banks for its loop). Everything else is one
//! table read by two binaries, because a second copy of it is how one member's converter gets
//! a rename fix the other's silently does not — the same argument
//! [`crate::format::repack`] makes for the `.f4` verify loop.
//!
//! The target names are the first-party text model's state-dict names (the Mini artifact is
//! gated against `text-model-names.tsv`, 693 params); the mapping is the Rosetta stone of
//! `docs/reference/inkling2-architecture.md` §6.

use anyhow::{Context, Result, bail, ensure};

use crate::format::{ArtifactDirs, Dtype, SafeWriter, TensorData, require_aux};
use rivoli_core::num::bf16_to_f32;

/// The artifact-side name of the tiktoken rank file: NESTED in both checkpoints
/// (`tiktoken/tokenizer.model`) and FLAT in the artifact, because that is the name the loader
/// reads. BPE tables, so the same file serves both members byte for byte.
pub const TIKTOKEN_ARTIFACT: &str = "tiktoken.model";

/// The two directories a run reads and writes, after the checks both member converters owe
/// before a weight is touched.
///
/// One function because it is the same five statements in both binaries and the ORDER is
/// load-bearing: the cheap refusals run before the multi-hour weight read, and a copy that
/// reordered them would be a converter which discovers a missing tokenizer at its last step —
/// the failure [`require_aux`] exists to move forward.
pub fn preflight<'a>(dirs: ArtifactDirs<'a>) -> Result<ArtifactDirs<'a>> {
    require_aux(dirs.src, INKLING_AUX)?;
    SafeWriter::refuse_writing_into_source(&dirs)?;
    let tiktoken = std::path::Path::new(dirs.src).join("tiktoken/tokenizer.model");
    require_nonempty(&tiktoken, "the tiktoken vocabulary the runtime reads")?;
    Ok(dirs)
}

/// Copy the nested tiktoken rank file to its flat artifact name and check it arrived.
///
/// Not part of [`INKLING_AUX`]: that list is what `finish_artifact` copies SAME-NAME, and this
/// one changes name on the way in.
pub fn copy_tiktoken(dirs: ArtifactDirs<'_>) -> Result<()> {
    copy_tiktoken_from(dirs, "tiktoken/tokenizer.model")
}

/// Copy a tiktoken rank file at relative path `rel` inside `dirs.src` to its flat artifact name.
pub fn copy_tiktoken_from(dirs: ArtifactDirs<'_>, rel: &str) -> Result<()> {
    let src = std::path::Path::new(dirs.src).join(rel);
    let dst = std::path::Path::new(dirs.out).join(TIKTOKEN_ARTIFACT);
    std::fs::copy(&src, &dst)
        .with_context(|| format!("copy {} into the artifact", src.display()))?;
    require_nonempty(&dst, "the copied tiktoken rank file")
}

/// The flat aux files copied beside the weights, identical for both members (the vendored
/// reference sets are the same files for Mini and Small). The tiktoken rank file is NOT in
/// this list: it ships nested (`tiktoken/tokenizer.model`) and is read flat
/// (`tiktoken.model`), so each converter copies and renames it explicitly.
pub const INKLING_AUX: &[&str] = &[
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
];

/// The verbose layer-suffix renames: checkpoint suffix after `model.llm.layers.{L}.` → target
/// suffix after `model.layers.{L}.`. A lookup table, not a branch chain: the mapping IS data,
/// and both converters' outputs are confronted against the first-party census.
///
/// The `controllers.*` rows are Mini-only (its loop's message gates); Small never reaches them
/// and does not need a second table to say so.
pub const LAYER_RENAMES: &[(&str, &str)] = &[
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

/// The two INDEXED `attn_res` query banks and what each becomes; `output_query` is the third
/// spelling and takes no index, so it is the constant below. One table, iterated, so
/// the two index rules (trap T12: `attn_queries.{i}` feeds layer `i+1`, `mlp_queries.{i}`
/// feeds layer `i`) are stated where they are applied rather than twice.
const QUERY_BANKS: &[(&str, usize, &str)] = &[
    ("model.llm.attn_res.attn_queries.", 1, "attention_input"),
    ("model.llm.attn_res.mlp_queries.", 0, "mlp_input"),
];
const OUTPUT_QUERY: &str = "model.llm.attn_res.output_query";

/// The checkpoint's two expert tensor-name stems. Each is a PREFIX of its own `_scale`
/// companion, so `contains` covers both without a second list.
const EXPERT_STEMS: &[&str] = &[".mlp.experts.w13_weight", ".mlp.experts.w2_weight"];

/// The target name for one verbatim layer suffix, or `None` when the suffix is not in
/// [`LAYER_RENAMES`] (which is a hard error at the caller, not a silent skip).
pub fn layer_target(layer: usize, suffix: &str) -> Option<String> {
    LAYER_RENAMES
        .iter()
        .find(|(s, _)| *s == suffix)
        .map(|(_, t)| format!("model.layers.{layer}.{t}"))
}

/// The rename for a verbatim (non-expert, non-query, non-unembed) TEXT name.
pub fn rename_verbatim(name: &str) -> Result<String> {
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

/// The target name for one `attn_res` query-bank tensor, or `None` when `name` is not one.
///
/// The two `middle_cycle_cross_pass*_queries` banks are NOT queries here: they merge into
/// `message_input.query` and only Mini has them, so its converter handles them itself.
pub fn query_target(name: &str) -> Result<Option<String>> {
    for (prefix, offset, slot) in QUERY_BANKS {
        if let Some(idx) = name.strip_prefix(*prefix) {
            let i: usize = idx
                .parse()
                .with_context(|| format!("{name:?}: bad query index"))?;
            return Ok(Some(format!("model.layers.{}.{slot}.query", i + offset)));
        }
    }
    Ok((name == OUTPUT_QUERY).then(|| "model.final_input.query".to_string()))
}

/// `unembed.weight` → `lm_head`, keeping the first `keep` rows (LOAD:856, trap T14) — the
/// EMBED keeps all `vocab_size` rows, so the two must not share this.
pub fn lm_head_rows(bytes: &[u8], keep: usize, hidden: usize) -> Result<&[u8]> {
    let want = keep * hidden * 2;
    ensure!(
        bytes.len() >= want,
        "lm_head: source has {} bytes, {keep} bf16 rows of {hidden} need {want}",
        bytes.len()
    );
    Ok(&bytes[..want])
}

/// The vocabulary widths `unembed` is unpadded against: the checkpoint's PADDED row count, the
/// live row count, and the hidden width.
#[derive(Clone, Copy)]
pub struct VocabRows {
    pub vocab_size: usize,
    pub unpadded: usize,
    pub hidden: usize,
}

/// Emit `lm_head.weight` from the checkpoint's `unembed`: the first `unpadded` rows of a
/// `[vocab_size, hidden]` bf16 matrix (LOAD:856, trap T14).
///
/// Shared because the row counts are the only difference between the two members here, and a
/// second copy of the shape check is how one member's artifact comes to keep the padded rows —
/// which decode 966 dead logits and fail nothing.
pub fn emit_lm_head<'a>(
    w: &mut SafeWriter<'a>,
    name: &str,
    raw: (&'a [u8], &[usize]),
    rows: VocabRows,
) -> Result<()> {
    let (b, sh) = raw;
    let VocabRows {
        vocab_size,
        unpadded,
        hidden,
    } = rows;
    ensure!(
        sh == [vocab_size, hidden],
        "{name}: shape {sh:?} != [{vocab_size}, {hidden}]"
    );
    // The EMBED keeps all `vocab_size` rows; only the head is unpadded (trap T14).
    w.add(
        "lm_head.weight",
        Dtype::Bf16,
        TensorData::new(vec![unpadded, hidden], lm_head_rows(b, unpadded, hidden)?),
    );
    Ok(())
}

/// True for any routed-expert tensor name — the weights AND their `_scale` companions, which
/// is why the stems match by containment. Used by both converters: Mini consumes them in the
/// dequant, Small excludes them from the resident set and selects their shards per layer.
pub fn is_expert_tensor(name: &str) -> bool {
    EXPERT_STEMS.iter().any(|stem| name.contains(*stem))
}

/// The engine consumes these as f32 (the rmsnorm/clip/sconv launchers take f32 operands) but
/// the checkpoint stores them bf16; the widening is exact. `sconv.static_weight` is the
/// dynamic sconv's per-layer `static_w` — the full-layer path (`pin.rs`) reads it as f32, so
/// widening it here keeps the artifact's dtype a function of the engine's operand type, not of
/// the layer.
pub fn wants_f32(target: &str) -> bool {
    const F32_SUFFIXES: &[&str] = &[
        "layernorm.weight",
        "norm.scale",
        "conv1d.weight",
        "sconv.static_weight",
        "model.norm.weight",
        "embed_norm.weight",
    ];
    F32_SUFFIXES.iter().any(|s| target.ends_with(*s))
}

/// `[r, c]` bf16 → `[c, r]` bf16 — the rel_proj transpose for the m=heads GEMM (x·wᵀ).
pub fn transpose_bf16(b: &[u8], r: usize, c: usize) -> Vec<u8> {
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
pub fn widen_bf16_f32(b: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(b.len() * 2);
    for x in b.chunks_exact(2) {
        let v = bf16_to_f32(u16::from_le_bytes([x[0], x[1]]));
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// The five large attention projections `--fp8` quantizes, as target-name SUFFIXES.
///
/// These are the dense trunk's matmuls — the only Inkling tensors whose every byte is spent in
/// a GEMV — so they are the ones a block-scaled fp8 container pays for: bf16→fp8 halves each,
/// which takes Mini's trunk from 2.72 GB to ~1.4 GB and Small's from 9.59 GB to ~4.8 GB, and the
/// bytes that stop being streamed become pinned experts under the same memory knob (P5's whole
/// currency). Everything else (`sink_proj`, the sconv weights, the norms, the router, `embed`,
/// `lm_head`) stays bf16: its quality is load-bearing at the PPL floor and its size is noise.
///
/// Suffix matching (rather than equality) because the caller holds a fully-qualified target
/// name, and the two members index layers differently — `model.layers.{L}.` for both, but Mini's
/// loop and Small's 48 layers are otherwise unrelated namespaces.
pub const DENSE_FP8_SUFFIXES: &[&str] = &[
    "self_attn.q_proj.linear.weight",
    "self_attn.k_proj.linear.weight",
    "self_attn.v_proj.linear.weight",
    "self_attn.o_proj.gate_proj.weight",
    "self_attn.o_proj.linear.weight",
];

/// True when `target` is one of the five projections [`DENSE_FP8_SUFFIXES`] names — the
/// predicate both `--fp8` converters and the engine's pin sniff are written against.
pub fn is_dense_fp8_target(target: &str) -> bool {
    DENSE_FP8_SUFFIXES
        .iter()
        .any(|suffix| target.ends_with(suffix))
}

/// Emit one verbatim TEXT tensor under its target name, applying the two engine-layout fixes:
/// the f32-consumed norms/clip-scales/sconv weights are widened bf16→f32 (exact — the
/// rmsnorm/clip/sconv launchers take f32), and `rel_proj` is transposed
/// `[d_rel, rel_extent]`→`[rel_extent, d_rel]` so the m=heads rel_logits GEMM computes x·wᵀ
/// without a runtime transpose.
///
/// The untouched branch BORROWS the source mmap (`b` is `&'a [u8]`) rather than copying it:
/// the resident set is gigabytes of bf16 and every verbatim tensor is the checkpoint's own
/// bytes, so a `to_vec()` here is host RAM spent for nothing.
pub fn emit_verbatim<'a>(
    w: &mut SafeWriter<'a>,
    target: String,
    raw: (&'a [u8], Dtype, &[usize]),
) -> Result<()> {
    let (b, dt, sh) = raw;
    if target.ends_with("relative_position_bias.proj") && dt == Dtype::Bf16 {
        ensure!(
            sh.len() == 2 && b.len() == sh[0] * sh[1] * 2,
            "{target}: expected 2-D bf16 shape, got {sh:?} ({} B)",
            b.len()
        );
        let bytes = transpose_bf16(b, sh[0], sh[1]);
        w.add(
            target,
            Dtype::Bf16,
            TensorData::new(vec![sh[1], sh[0]], bytes),
        );
        return Ok(());
    }
    if wants_f32(&target) && dt == Dtype::Bf16 {
        let bytes = widen_bf16_f32(b);
        w.add(target, Dtype::F32, TensorData::new(sh.to_vec(), bytes));
        return Ok(());
    }
    w.add(target, dt, TensorData::new(sh.to_vec(), b));
    Ok(())
}

/// [`emit_verbatim`] with the `--fp8` trunk rewrite: when `fp8_block` is `Some(block)` and
/// `target` is one of [`DENSE_FP8_SUFFIXES`], the bf16 matrix is quantized to block-scaled
/// fp8-e4m3 and emitted as the `<base>.weight` + `<base>.weight_scale_inv` pair the resident fp8
/// path reads ([`SafeWriter::copy_fp8`]'s convention, and the one `pin.rs` sniffs a layer by).
/// Every other tensor — and every tensor of a `--fp8`-less run — delegates to [`emit_verbatim`]
/// untouched, which is what keeps ONE conversion loop serving both artifact shapes.
///
/// **`None` is the compatibility arm, not a default.** A bf16 artifact has no `weight_scale_inv`
/// tensor anywhere, and the loader tells the two apart by layer 0's `q_proj.linear.weight`
/// dtype alone (`Dtype::Bf16` ⇒ bf16, `Dtype::F8E4M3` ⇒ fp8 at `FormatMeta::fp8_block`), so the
/// emitter and the pin sniff are two halves of one decision.
///
/// The quantization itself is [`crate::quant::quantize_fp8_block`]'s — the same function
/// Glimmer's bf16 trunk rides — because the scale rule is a property of the container, not of
/// the model: a second implementation here would be a second set of scales for one format.
pub fn emit_verbatim_fmt<'a>(
    w: &mut SafeWriter<'a>,
    target: String,
    raw: (&'a [u8], Dtype, &[usize]),
    fp8_block: Option<usize>,
) -> Result<()> {
    let quant = fp8_block.filter(|_| raw.1 == Dtype::Bf16 && is_dense_fp8_target(&target));
    match quant {
        Some(block) => emit_quantized_fp8(w, target, raw, block),
        None => emit_verbatim(w, target, raw),
    }
}

/// The fp8 half of [`emit_verbatim_fmt`]: bf16 bytes at `block`-sized tiles into the pair
/// `pin.rs` reads. `target` still carries `.weight`, so the base comes off the name that was
/// matched — one string, so the weight and its grid cannot be named apart.
fn emit_quantized_fp8<'a>(
    w: &mut SafeWriter<'a>,
    target: String,
    raw: (&'a [u8], Dtype, &[usize]),
    block: usize,
) -> Result<()> {
    let (b, _, sh) = raw;
    ensure!(
        sh.len() == 2 && b.len() == sh[0] * sh[1] * 2,
        "{target}: expected a 2-D bf16 weight, got {sh:?} ({} B)",
        b.len()
    );
    let (o_dim, i_dim) = (sh[0], sh[1]);
    let base = target
        .strip_suffix(".weight")
        .with_context(|| format!("{target}: an fp8 projection does not end in `.weight`"))?;
    let vals: Vec<f32> = b
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (packed, scale) = crate::quant::quantize_fp8_block(&vals, [o_dim, i_dim], block)
        .with_context(|| target.clone())?;
    let sbytes: Vec<u8> = scale.iter().flat_map(|s| s.to_le_bytes()).collect();
    // The grid's name comes off `base`, so it is built before `target` moves into `add`.
    let grid = format!("{base}.weight_scale_inv");
    w.add(target, Dtype::F8E4M3, TensorData::new(sh.to_vec(), packed));
    w.add(
        grid,
        Dtype::F32,
        TensorData::new(vec![o_dim.div_ceil(block), i_dim.div_ceil(block)], sbytes),
    );
    Ok(())
}

/// The range an artifact will HOLD after this run converts `want` on top of whatever `held`
/// already says — refusing the one shape a range cannot express: a GAP.
///
/// Both members' converters resume by range (`--from`/`--to`), and `*_source.layers` is one
/// interval, so a run that converts layer 9 into an artifact holding 0..3 would stamp 0..10 and
/// promise six layers that are not on disk. The loader would then walk into a missing
/// `L03.mxfp8` and report a damaged artifact. Refusing the gap is what keeps the stamp honest;
/// a run that needs a second island belongs in its own artifact directory.
pub fn merged_range(
    held: Option<std::ops::Range<usize>>,
    want: std::ops::Range<usize>,
) -> Result<std::ops::Range<usize>> {
    ensure!(
        want.start < want.end,
        "the requested layer range [{}, {}) is empty",
        want.start,
        want.end
    );
    let Some(held) = held else {
        return Ok(want);
    };
    ensure!(
        want.start <= held.end && held.start <= want.end,
        "converting layers [{}, {}) would leave a gap beside the [{}, {}) this artifact \
         already holds — `*_source.layers` is one interval, so the manifest would claim layers \
         that are not on disk. Convert the layers in between first, or use another artifact \
         directory for a second range.",
        want.start,
        want.end,
        held.start,
        held.end
    );
    Ok(held.start.min(want.start)..held.end.max(want.end))
}

/// A file must exist and be non-empty. Applied to the tiktoken rank file at both ends (source
/// and artifact), which `finish_artifact` does not copy — it is nested-renamed.
pub fn require_nonempty(path: &std::path::Path, what: &str) -> Result<()> {
    ensure!(
        std::fs::metadata(path).is_ok_and(|m| m.len() > 0),
        "{} is missing or empty — {what}",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    // Pure name arithmetic over the tables above, so nothing here touches a file. Crate-wide
    // `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// The query index rules (trap T12) and the two globals, at the names the checkpoint
    /// actually carries.
    #[test]
    fn query_banks_and_globals_land_on_their_targets() {
        assert_eq!(
            query_target("model.llm.attn_res.attn_queries.0").unwrap(),
            Some("model.layers.1.attention_input.query".to_string())
        );
        assert_eq!(
            query_target("model.llm.attn_res.mlp_queries.47").unwrap(),
            Some("model.layers.47.mlp_input.query".to_string())
        );
        assert_eq!(
            query_target("model.llm.attn_res.output_query").unwrap(),
            Some("model.final_input.query".to_string())
        );
        assert_eq!(query_target("model.llm.embed.weight").unwrap(), None);
        assert_eq!(
            rename_verbatim("model.llm.embed.weight").unwrap(),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            rename_verbatim("model.llm.layers.11.attn.k_sconv.static_w").unwrap(),
            "model.layers.11.self_attn.k_proj.sconv.static_weight"
        );
        // An unknown suffix must refuse rather than pass a name through.
        assert!(rename_verbatim("model.llm.layers.0.mlp.not_a_tensor").is_err());
    }

    /// The gap refusal: adjacent and overlapping ranges union, a disjoint one refuses. The
    /// cases are the ones `--from`/`--to` reach in practice — resume where the last run
    /// stopped, re-convert a middle slice, or ask for something the manifest cannot say.
    #[test]
    fn merged_range_unions_adjacent_and_overlapping_ranges_and_refuses_a_gap() {
        let h = || Some(0..3);
        assert_eq!(merged_range(None, 4..6).unwrap(), 4..6);
        assert_eq!(merged_range(h(), 3..6).unwrap(), 0..6);
        assert_eq!(merged_range(h(), 1..4).unwrap(), 0..4);
        assert_eq!(merged_range(h(), 0..2).unwrap(), 0..3);
        assert!(merged_range(Some(0..3), 5..6).is_err());
        assert!(merged_range(None, 4..4).is_err());
    }

    /// The five projections [`DENSE_FP8_SUFFIXES`] names — and the tensors that share their
    /// prefixes but must NOT be rewritten: the sconv weights, the norms, the near-miss suffix,
    /// the routed experts and the embedding.
    #[test]
    fn dense_fp8_targets_are_the_five_attention_projections() {
        assert_eq!(
            DENSE_FP8_SUFFIXES.len(),
            5,
            "a projection joined or left the fp8 trunk set"
        );
        for suffix in DENSE_FP8_SUFFIXES {
            assert!(
                is_dense_fp8_target(&format!("model.layers.7.{suffix}")),
                "{suffix} is a dense target but was not recognised"
            );
        }
        for other in [
            "model.layers.7.self_attn.sink_proj.weight",
            "model.layers.7.self_attn.q_proj.sconv.conv1d.weight",
            "model.layers.7.self_attn.k_proj.sconv.weight_proj.weight",
            "model.layers.7.self_attn.q_proj.norm.scale",
            "model.layers.7.self_attn.o_proj.gate_proj_bias.weight",
            "model.layers.7.mlp.experts.w2_weight",
            "model.embed_tokens.weight",
        ] {
            assert!(
                !is_dense_fp8_target(other),
                "{other} was taken for a dense fp8 target"
            );
        }
    }

    /// The `[8, 8]` fixture both emitter tests read, as f32 and as the bf16 bytes a converter
    /// holds: alternating signs, magnitudes `1/8 ..= 4`, so every value sits far enough above a
    /// tile's scale that e4m3's half-ulp is the only error the decode check can see.
    fn fixture_weights() -> (Vec<f32>, Vec<u8>) {
        let src: Vec<f32> = (0..64)
            .map(|k| {
                let mag = (k / 2 + 1) as f32 / 8.0;
                if k % 2 == 0 { mag } else { -mag }
            })
            .collect();
        let bytes = src
            .iter()
            .flat_map(|v| rivoli_core::num::f32_to_bf16(*v).to_le_bytes())
            .collect();
        (src, bytes)
    }

    /// **The quantizing arm, on disk.** A dense target under `Some(block)` becomes the
    /// `<base>.weight` + `.weight_scale_inv` pair the resident fp8 path reads, at the grid shape
    /// the block implies — and a block larger than both dims is ONE tile.
    ///
    /// The payload equality with [`crate::quant::quantize_fp8_block`] is a WIRING check, not a
    /// numerics one — the scale rule is scored in `quant::fp8` and restating it here would let
    /// this test pass while both copies were wrong. What it pins is the naming, the dtypes and
    /// the grid's shape, which are this function's own.
    #[test]
    fn emit_verbatim_fmt_quantizes_a_dense_target_into_an_fp8_pair() {
        let dir = std::env::temp_dir().join("rivoli_inkling_fmt_fp8");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| format!("{}/{name}", dir.display());

        let (src, bf16) = fixture_weights();
        let (o_dim, i_dim, block) = (8usize, 8usize, 4usize);
        let shape = [o_dim, i_dim];
        let dense = "model.layers.0.self_attn.q_proj.linear.weight".to_string();
        let raw: (&[u8], Dtype, &[usize]) = (&bf16, Dtype::Bf16, shape.as_slice());

        let mut q = SafeWriter::new();
        emit_verbatim_fmt(&mut q, dense.clone(), raw, Some(block)).unwrap();
        q.write(&path("fp8.safetensors")).unwrap();

        let quantized = crate::format::Safetensors::open_file(&path("fp8.safetensors")).unwrap();
        let (packed, psh) = quantized.typed(&dense, Dtype::F8E4M3).unwrap();
        assert_eq!(
            psh,
            shape.as_slice(),
            "the fp8 weight lost its [o, i] shape"
        );
        let grid = format!(
            "{}.weight_scale_inv",
            dense.strip_suffix(".weight").unwrap()
        );
        let (sb, ssh) = quantized.typed(&grid, Dtype::F32).unwrap();
        assert_eq!(ssh, &[o_dim.div_ceil(block), i_dim.div_ceil(block)][..]);
        let (want_packed, want_scale) =
            crate::quant::quantize_fp8_block(&src, shape, block).unwrap();
        assert_eq!(
            packed,
            want_packed.as_slice(),
            "the fp8 payload was rewritten"
        );
        let scales: Vec<f32> = sb
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(scales, want_scale);
        // And the pair DECODES back inside e4m3's three-mantissa-bit half-ulp (2^-4).
        let view = crate::quant::Fp8W {
            packed,
            scale: &scales,
            block,
        };
        let got = crate::quant::dequant_fp8_block(view, shape);
        for (k, want) in src.iter().enumerate() {
            assert!(
                (got[k] - want).abs() <= 0.0625 * want.abs() + 1e-6,
                "weight {k}: {want} decoded to {}",
                got[k]
            );
        }

        // A block larger than both dims — the block-128 shape every real Inkling projection takes
        // — is ONE tile, which is the `div_ceil` boundary the small matrix above cannot reach.
        let mut big = SafeWriter::new();
        emit_verbatim_fmt(&mut big, dense.clone(), raw, Some(128)).unwrap();
        let p128 = path("dense128.safetensors");
        big.write(&p128).unwrap();
        let coarse = crate::format::Safetensors::open_file(&p128).unwrap();
        assert_eq!(
            coarse.typed(&dense, Dtype::F8E4M3).unwrap().1,
            shape.as_slice()
        );
        assert_eq!(coarse.typed(&grid, Dtype::F32).unwrap().1, &[1, 1][..]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Non-dense targets and `fp8_block = None` stay verbatim bf16 without a scale grid.
    #[test]
    fn emit_verbatim_fmt_leaves_every_other_tensor_bf16_verbatim() {
        let dir = format!(
            "{}/rivoli_inkling_fmt_bf16_{}",
            std::env::temp_dir().display(),
            std::process::id()
        );
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| format!("{dir}/{name}");

        let (_, bf16) = fixture_weights();
        let shape = [8usize, 8];
        let dense = "model.layers.0.self_attn.q_proj.linear.weight".to_string();
        let sink = "model.layers.0.self_attn.sink_proj.weight".to_string();
        let raw: (&[u8], Dtype, &[usize]) = (&bf16, Dtype::Bf16, shape.as_slice());

        let mut b = SafeWriter::new();
        emit_verbatim_fmt(&mut b, dense.clone(), raw, None).unwrap();
        emit_verbatim_fmt(&mut b, sink.clone(), raw, None).unwrap();
        b.write(&path("plain.safetensors")).unwrap();
        let mut s = SafeWriter::new();
        emit_verbatim_fmt(&mut s, sink.clone(), raw, Some(128)).unwrap();
        s.write(&path("sink.safetensors")).unwrap();

        for name in ["plain.safetensors", "sink.safetensors"] {
            let verbatim = crate::format::Safetensors::open_file(&path(name)).unwrap();
            let (bytes, vsh) = verbatim.typed(&sink, Dtype::Bf16).unwrap();
            assert_eq!(vsh, shape.as_slice(), "{name}: the sink lost its shape");
            assert_eq!(
                bytes,
                bf16.as_slice(),
                "{name}: a bf16 tensor's bytes were rewritten"
            );
            assert!(
                verbatim
                    .typed(&format!("{sink}.weight_scale_inv"), Dtype::F32)
                    .is_err(),
                "{name}: a bf16 tensor grew a scale grid"
            );
        }
        let plain = crate::format::Safetensors::open_file(&path("plain.safetensors")).unwrap();
        let (bytes, _) = plain.typed(&dense, Dtype::Bf16).unwrap();
        assert_eq!(
            bytes,
            bf16.as_slice(),
            "fp8_block None rewrote a dense target"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
