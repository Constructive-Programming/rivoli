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
use rivoli::artifact::tokenizer::eos_token_ids;

/// The one file carrying Muse Glimmer's stop tokens. Named because three places spell it and
/// [`eos_ids`] refuses on its contents — a literal repeated beside a check that reads it is one
/// rename away from checking a different file than the one it copies.
const GEN: &str = "generation_config.json";

/// Auxiliary files copied beside the weights. `generation_config.json` is **load-bearing and
/// not optional decoration**: it carries `eos_token_id: [200001, 200008]`, and the
/// `text_config` in `config.json` carries only the scalar `200001` (trap 13). A run that took
/// EOS from the config would stop on one of the two.
///
/// > **CORRECTED 2026-08-11**, by two reviews. That paragraph asserted a guarantee the code
/// > did not make: `finish_artifact` downgrades every aux-copy failure to
/// > `eprintln!("WARNING: {name} not copied")` and returns `Ok(())`, so a partial clone
/// > (`huggingface-cli download --include "*.safetensors" config.json tokenizer*` is enough)
/// > produced an artifact whose only EOS is the scalar — trap 13, live, after a multi-hour
/// > convert, announced by one warning line in the log. [`REQUIRED_AUX`] is the fix; the
/// > tolerance stays for the rest, because it is shared with three other converters.
const AUX: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    GEN,
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

/// The subset of [`AUX`] whose absence is a REFUSAL rather than a warning, checked before any
/// weight is read.
///
/// Both carry a fact that exists nowhere else in the source tree, and both have already cost
/// this repo a round in another model: `generation_config.json` is the only file with the
/// two-element `eos_token_id` (trap 13), and `chat_template.jinja` is the one the memory note
/// `artifact-drops-the-chat-template` records rivoli drifting away from for months precisely
/// because it lived only upstream. A converter that shrugs at their absence hands the problem
/// to whoever debugs the decode.
const REQUIRED_AUX: [&str; 2] = [GEN, "chat_template.jinja"];

/// The EOS ids `generation_config.json` carries, refusing a file that exists and says nothing.
///
/// **PRESENCE WAS CHECKED AND CONTENT WAS NOT, and that left trap 13 live in a worse spelling.**
/// [`REQUIRED_AUX`] was added because a partial download produced an artifact whose only EOS was
/// `text_config`'s scalar. It refuses a MISSING file — and until 2026-08-13 this tree's own Glimmer
/// fixture wrote `generation_config.json` as `{}`, which passes that check, copies into the
/// artifact, and yields `Tokenizer::load_eos` **no ids at all**. The engine's whole response is one
/// `warn!("decode won't stop on EOS")` and then `eos.contains(&t)` is false for every token, so
/// generation runs to `ngen` every time. That is not hypothetical damage: it is the exact signature
/// behind `docs/measurement/benchmarks.md`'s retraction, where across 56 runs not one terminated
/// naturally and the model drifted into list scaffolding and then looped.
///
/// So a port does not "stop on one of the two" — it stops on NONE, and it says so in a log line
/// nobody reads three hours into a convert. Checked here rather than in `artifact/tokenizer.rs`
/// because that path is shared with three other models and its tolerance is theirs to keep; this
/// binary converts one checkpoint and knows what that checkpoint has to carry.
///
/// **Non-empty, NOT "exactly two".** Glimmer's file carries `[200001, 200008]`, but pinning the
/// count here would be the same over-fit `layer_types`' doc argues against: the pair is a fact
/// about this checkpoint, not a rule about the architecture. What is architectural is that a decode
/// with zero stop tokens cannot terminate.
fn eos_ids(dir: &str) -> Result<Vec<u32>> {
    // **The ENGINE's parser, not a second one.** This function was fifteen lines re-spelling
    // `Tokenizer::load_eos`, with a comment asserting the two "match" — a claim the shared
    // function makes structural rather than asserted (review, 2026-08-13). What belongs here is
    // the REFUSAL: `artifact/tokenizer.rs` is shared with three other models and warns rather
    // than failing, which is its call to keep; this binary converts one checkpoint and knows what
    // that checkpoint has to carry.
    let ids = eos_token_ids(dir)?;
    ensure!(
        !ids.is_empty(),
        "{dir}/{GEN} carries no usable `eos_token_id`. The file exists, so REQUIRED_AUX is \
         satisfied and the artifact would ship — with NO stop token, which makes every decode run \
         to its token limit (glimmer-architecture.md §9 trap 13). Glimmer's is `[200001, 200008]`",
    );
    Ok(ids)
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
    for aux in REQUIRED_AUX {
        let p = std::path::Path::new(&args.src_dir).join(aux);
        ensure!(
            p.is_file(),
            "{} is missing from {}. It is not decoration — see REQUIRED_AUX — and \
             `finish_artifact` would only WARN about it, three hours into the convert",
            aux,
            args.src_dir
        );
    }
    // Parsed, and therefore validated, before a single tensor is read: `GlimmerConfig` is
    // where the layer/RoPE pairing invariant and the width checks live, and a checkpoint that
    // fails them must not produce a half-written artifact.
    let cfg: GlimmerConfig = load_config(&args.src_dir)?;
    let g = &cfg.text;
    // AFTER `load_config` so `g.vocab` is available, and still before any tensor is read — nothing
    // between the two touches the checkpoint. The bound is what separates "at least one number
    // parses" from "at least one token this model can emit": an id past the vocabulary is a stop
    // token no argmax can ever return, which is the same unstoppable decode one layer down
    // (review, 2026-08-13).
    //
    // PRINTED, not just checked. The ids decide whether a decode can terminate at all, they live
    // in an aux file rather than in the manifest, and the operator running this has the checkpoint
    // in front of them — so the one line that lets them notice a wrong set is worth more here than
    // after a decode has run to its token limit.
    let ids = eos_ids(&args.src_dir)?;
    for &id in &ids {
        ensure!(
            (id as usize) < g.vocab,
            "eos_token_id {id} is past this model's vocabulary of {} — no argmax can return it, \
             so it is a stop token that never fires",
            g.vocab
        );
    }
    eprintln!("convert_glimmer: eos_token_id {ids:?}");
    std::fs::create_dir_all(&args.out_dir)?;

    // ONE predicate driving both which shards are opened and what is written — the pairing
    // `convert_k3` records as having broken from the other end, where the resident pass looped
    // over every tensor in every opened shard and so made the artifact a function of the
    // checkpoint's shard boundaries rather than of the request.
    let want = |n: &str| !is_vision(n);
    let src = Safetensors::open_indexed(&args.src_dir, want)?;

    // **The skipped count comes from the INDEX, not from the opened shards.** `open_indexed`
    // selects whole SHARDS containing at least one wanted tensor, and `want` excludes vision —
    // so a shard holding only vision tensors is never opened and its tensors never appear in
    // `src.names()`. Counting them there under-reports by however many such shards the
    // checkpoint happens to have, which makes the number a function of its shard boundaries
    // rather than of the model. The fixture is single-shard so it could not show this; review
    // found it 2026-08-11, and the count is the whole reason `is_vision` exists as a named
    // predicate ("an observation, not an assumption").
    let skipped = {
        let idx = std::fs::read_to_string(
            std::path::Path::new(&args.src_dir).join("model.safetensors.index.json"),
        )?;
        let idx: serde_json::Value = serde_json::from_str(&idx)?;
        idx["weight_map"]
            .as_object()
            .context("model.safetensors.index.json has no weight_map")?
            .keys()
            .filter(|n| is_vision(n))
            .count()
    };

    let mut w = SafeWriter::new();
    let (mut verbatim, mut widened) = (0usize, 0usize);
    let mut names: Vec<String> = src.names().iter().map(|s| s.to_string()).collect();
    names.sort();
    for name in &names {
        if is_vision(name) {
            continue; // counted from the index above, not from whichever shards were opened
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
    // **AGAIN, against the ARTIFACT — the source-side check above does not establish this.**
    // `finish_artifact` downgrades every aux-copy failure to `eprintln!("WARNING: {name} not
    // copied")` and returns `Ok(())`, which is the defect [`REQUIRED_AUX`]'s own note was written
    // for. So a checkpoint that passes the check at the top can still produce an artifact with no
    // `generation_config.json` at all — the exact zero-stop-token state this file argues against —
    // whenever the copy fails: the filesystem fills between the 53 GB write and the aux copy, the
    // mount goes read-only, or the source file is removed during a three-hour run. The check at the
    // top narrows the window; this one closes it, and it is the invariant the engine actually
    // depends on, because the engine reads the ARTIFACT's copy and never the source's.
    //
    // Found by review 2026-08-13, against a version whose test already asserted the artifact-side
    // property while the shipped binary only asserted the source-side one.
    let placed = eos_ids(&args.out_dir)?;
    ensure!(
        placed == ids,
        "the artifact's eos_token_id is {placed:?} but the checkpoint's is {ids:?} — the aux copy \
         did not reproduce the file the engine will read"
    );
    Ok(())
}

/// The vision tower, its adapter and its projector — **out of scope, skipped explicitly.**
///
/// Explicitly, and the count is printed, which is what turns "the vision half was excluded"
/// from an assumption into an observation.
///
/// > **CORRECTED 2026-08-11**, by review. This said the text side is "selected by a positive
/// > predicate below, so an unrecognised vision tensor would simply not be copied". **The
/// > predicate is `|n| !is_vision(n)` — negative**, and the real behaviour is the inverse of
/// > what that argued: a fourth vision prefix would be copied verbatim INTO the artifact and
/// > counted as text. `tests/glimmer_names.rs`'s
/// > `every_family_is_either_implemented_or_deliberately_skipped` is what actually catches
/// > that, by restating these three prefixes and reconciling 627 + 809 against the shipped
/// > index — so the guard exists, it is just not the one this comment claimed.
fn is_vision(name: &str) -> bool {
    name.starts_with("model.vision_tower.")
        || name.starts_with("model.vision_adapter.")
        || name.starts_with("model.vision_projection")
}

/// Norms are widened bf16 -> f32; everything else is copied verbatim.
///
/// Every architecture in this engine reads its norm weights as f32 (`add_widened`'s doc lists
/// them), so this follows the house convention rather than inventing one. It is also cheap:
/// **4** norms per layer plus the final one — 209 tensors, ~5.5 MB widened, against 55.7 GB
/// of weights. (Said 5 until review corrected it 2026-08-11; the count that matters is
/// asserted, as `L * 4 + 1`, by `tests/glimmer_convert.rs`.)
///
/// **Matched on the tail, not on a `layers.N.` prefix**, so the model-level `norm.weight` and
/// the four per-layer `*_layernorm.weight` take the same path — `"…layernorm.weight"` ends
/// with `"norm.weight"`, so one suffix covers both and a second clause for the layer norms
/// would be dead. Glimmer's QK-norm is weightless and ships no tensor, so there is nothing
/// else this can catch today; if one ever appears it is a norm and this is still right.
fn is_norm(name: &str) -> bool {
    name.ends_with("norm.weight")
}
