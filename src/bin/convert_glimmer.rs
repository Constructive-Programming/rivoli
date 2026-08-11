//! `convert_glimmer` — build a Muse Glimmer-30B artifact from the HuggingFace checkpoint.
//!
//! **This converter quantizes nothing, on purpose.** Glimmer ships BF16 and nothing else, so
//! unlike GLM (fp8), DeepSeek-V4 (fp8 + e8m0) and Kimi-K3 (mxfp4) there is no publisher
//! decision to copy — `quantize_fp8_block` exists, but choosing its scales is *rivoli's*
//! quality decision and it has no measurement behind it yet. So the first artifact is
//! **bf16 verbatim**: 53.02 GB/token against fp8's 26.51, and zero new arithmetic between the
//! checkpoint and the kernels. `docs/investigations/glimmer-port.md` S1a item 2 carries the
//! dNLL gate the fp8 halving has to pass before it becomes the default, and this file is the
//! reference arm that gate is measured against.
//!
//! It is also why this converter has no memory problem. `SafeWriter` holds `Cow<'a, [u8]>`
//! and `copy_verbatim` **borrows the mapped source**, so a 55.7 GB resident set streams
//! through without a host copy. An fp8 pass would produce owned bytes and put ~26 GB in RAM;
//! that is the two-pass work item, and going bf16 first defers it rather than solving it.
//!
//! Separate from `bin/convert` and `bin/convert_k3` because it shares almost nothing with
//! either: no codebook to learn, no VQ encode, no fp8 dequant, no expert files at all. A
//! dense model's artifact is one `resident.safetensors` plus a manifest.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli::artifact::format::{Dtype, FormatMeta, SafeWriter, Safetensors, finish_artifact};
use rivoli::artifact::model::{
    GLIMMER_LAYER_PREFIX, GLIMMER_LAYER_TENSORS, GlimmerConfig, load_config,
};

/// Auxiliary files copied beside the weights. `generation_config.json` is **load-bearing and
/// not optional decoration**: it carries `eos_token_id: [200001, 200008]`, and the
/// `text_config` in `config.json` carries only the scalar `200001` (trap 13). A run that took
/// EOS from the config would stop on one of the two.
const AUX: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
];

// jscpd:ignore-start — the clap entry boilerplate every converter in this tree shares:
// `<last field>: <ty>, } fn main() -> Result<()> { let args = Args::parse();`. jscpd
// normalizes identifiers, so the match is on SHAPE, and the shape is prescribed by clap
// rather than chosen here — `convert.rs` is the twin it reports against. The alternatives
// were both worse: contort this binary's Args/main boundary purely to defeat the tokenizer,
// or drop clap for `std::env::args()` and lose `--help` on a tool whose two positional
// directories are easy to swap. Being a verbatim copy IS the point here.
#[derive(Parser)]
#[command(about = "Build a Muse Glimmer artifact (bf16 verbatim) from an HF checkpoint")]
struct Args {
    /// The HF checkpoint directory: `config.json` + `model.safetensors.index.json` + shards.
    src_dir: String,
    /// Where to write `resident.safetensors` and `manifest.json`.
    out_dir: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // jscpd:ignore-end
    // **Not into the source directory.** `SafeWriter::write` reads borrowed payloads at write
    // time, so the mapped shards are live while the output is being created; truncating a
    // file another mapping still holds is a SIGBUS — a fatal signal, not an error — with the
    // output left half-formed. `write` renames a sibling into place for exactly this reason,
    // and this refuses the case that argument does not cover.
    ensure!(
        std::fs::canonicalize(&args.src_dir).ok() != std::fs::canonicalize(&args.out_dir).ok(),
        "out_dir and src_dir resolve to the same directory; the writer maps the source while \
         it writes, and overwriting a mapped shard is a SIGBUS rather than an error"
    );
    // Parsed, and therefore validated, before a single tensor is read: `GlimmerConfig` is
    // where the layer/RoPE pairing invariant and the width checks live, and a checkpoint that
    // fails them must not produce a half-written artifact.
    let cfg: GlimmerConfig = load_config(&args.src_dir)?;
    let g = &cfg.text;
    std::fs::create_dir_all(&args.out_dir)?;

    // ONE predicate driving both which shards are opened and what is written — the pairing
    // `convert_k3` records as having broken from the other end, where the resident pass looped
    // over every tensor in every opened shard and so made the artifact a function of the
    // checkpoint's shard boundaries rather than of the request.
    let want = |n: &str| !is_vision(n);
    let src = Safetensors::open_indexed(&args.src_dir, want)?;

    let mut w = SafeWriter::new();
    let (mut verbatim, mut widened, mut skipped) = (0usize, 0usize, 0usize);
    let mut names: Vec<String> = src.names().iter().map(|s| s.to_string()).collect();
    names.sort();
    for name in &names {
        if is_vision(name) {
            skipped += 1;
            continue;
        }
        if is_norm(name) {
            w.add_widened(&src, name)?;
            widened += 1;
        } else {
            // Verbatim, and asserted BF16 rather than "whatever it is": `typed` refuses any
            // other dtype, so an fp8 or 4-bit export of this model refuses here instead of
            // being copied into an artifact that claims to be bf16.
            w.copy_verbatim(&src, name, Dtype::Bf16)?;
            verbatim += 1;
        }
    }

    // The tensors the decode path cannot run without. Checked by NAME here rather than left
    // to the pin: a missing `lm_head` in a partial checkpoint would otherwise surface as a
    // pin-time "tensor not found" long after the 55 GB write.
    for required in [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ] {
        ensure!(
            names.iter().any(|n| n == required),
            "the checkpoint has no {required} — this is not a complete Glimmer text model"
        );
    }
    // Per-layer completeness, so a truncated shard set is caught before the write rather than
    // at pin time. Nine tensors per layer: five projections, four norms.
    for l in 0..g.n_layers {
        for t in GLIMMER_LAYER_TENSORS {
            let n = format!("{GLIMMER_LAYER_PREFIX}.{l}.{t}.weight");
            ensure!(names.contains(&n), "the checkpoint has no {n}");
        }
    }

    let out = format!("{}/resident.safetensors", args.out_dir);
    w.write(&out).with_context(|| format!("write {out}"))?;
    eprintln!(
        "convert_glimmer: {verbatim} tensors bf16 verbatim, {widened} norms widened to f32, \
         {skipped} vision tensors skipped -> {out}"
    );

    // The manifest is the checkpoint's own `config.json` plus the `format` section, so
    // `load_config::<GlimmerConfig>` reads the artifact exactly as it read the source — the
    // wrapper and its `text_config` intact, which is what `Arch::from_manifest_str` needs.
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}/config.json", args.src_dir))?)?;
    // `FormatMeta::current` stamps the compiled-in VQ parameters even though this artifact has
    // no VQ tensors. That is inert rather than a lie: `FormatMeta::load` compares them against
    // the same constants, so they always agree, and the plan's "nullable VQ section" turned
    // out to be work that nothing needed. `fp8_block` likewise describes a format this
    // artifact does not use yet — it is what the fp8 pass will write.
    manifest["format"] =
        serde_json::to_value(FormatMeta::current(rivoli::artifact::quant::FP8_BLOCK))?;
    finish_artifact(
        "convert_glimmer",
        &args.out_dir,
        &args.src_dir,
        &manifest,
        &AUX,
    )?;
    Ok(())
}

/// The vision tower, its adapter and its projector — **out of scope, skipped explicitly.**
///
/// Explicitly rather than by omission: the text side is selected by a positive predicate
/// below, so an unrecognised vision tensor would simply not be copied and nothing would say
/// so. This function exists to make the *count* of skipped tensors a number the run prints,
/// which is what turns "the vision half was excluded" from an assumption into an observation.
fn is_vision(name: &str) -> bool {
    name.starts_with("model.vision_tower.")
        || name.starts_with("model.vision_adapter.")
        || name.starts_with("model.vision_projection")
}

/// Norms are widened bf16 -> f32; everything else is copied verbatim.
///
/// Every architecture in this engine reads its norm weights as f32 (`add_widened`'s doc lists
/// them), so this follows the house convention rather than inventing one. It is also cheap:
/// 5 norms per layer plus the final one is ~5.5 MB widened, against 55.7 GB of weights.
///
/// **Matched on the tail, not on a `layers.N.` prefix**, so the model-level `norm.weight` and
/// the four per-layer `*_layernorm.weight` take the same path — `"…layernorm.weight"` ends
/// with `"norm.weight"`, so one suffix covers both and a second clause for the layer norms
/// would be dead. Glimmer's QK-norm is weightless and ships no tensor, so there is nothing
/// else this can catch today; if one ever appears it is a norm and this is still right.
fn is_norm(name: &str) -> bool {
    name.ends_with("norm.weight")
}
