//! `convert_glimmer` — build a Muse Glimmer-30B artifact from the HuggingFace checkpoint.
//! Ported from `old:src/bin/convert_glimmer.rs` (`wt/glimmer-s2` @ 6b7f496), comments
//! travelling with their code.
//!
//! **Default: quantizes nothing.** Glimmer ships BF16 and nothing else, so unlike GLM (fp8),
//! DeepSeek-V4 (fp8 + e8m0) and Kimi-K3 (mxfp4) there is no publisher decision to copy —
//! choosing the scales is *rivoli's* quality decision. So the default artifact is **bf16
//! verbatim**, and it is the reference arm every other rung of the ladder is measured against.
//!
//! > **AMENDED 2026-08-16 (M11), following the reference's own S4 amendment.** This said
//! > "quantizes nothing, on purpose", full stop, and deferred the fp8 pass as "the two-pass
//! > work item". `--fp8` now exists and this file is both arms — ported from the reference,
//! > which measured the pair on the old engine (its `benchmarks.md` S4 ladder: fp8-vs-bf16
//! > paired mean dNLL −0.00026, 95% CI [−0.00701, +0.00649] on the 762-token corpus — a pass,
//! > not an underpowered null). What has NOT changed: bf16 stays the DEFAULT, because it is
//! > the arm with no arithmetic of ours between the checkpoint and the kernels, and a ladder
//! > needs a rung nothing was done to. Which format ships as the default for decoding is
//! > settled by this tree re-measuring that dNLL through its own seam, not here.
//!
//! **`--fp8` costs host memory that the bf16 pass does not, and the asymmetry is structural.**
//! `SafeWriter` holds `Cow<'a, [u8]>` and `copy_verbatim` **borrows the mapped source**, so a
//! 55.7 GB bf16 resident set streams through with no host copy at all. Quantized bytes are
//! produced rather than borrowed, so `--fp8` holds ~25.2 GB before the write. That is accepted
//! for a one-off offline conversion on this host; `SafeWriter::add_quantized_fp8` carries the
//! upgrade path.
//!
//! Separate from `bin/convert` because it shares almost nothing with it: no codebook to learn,
//! no VQ encode, no fp8 dequant, no expert files at all. A dense model's artifact is one
//! `resident.safetensors` plus a manifest.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli_artifact::format::{
    ArtifactDirs, Dtype, FormatMeta, SafeWriter, Safetensors, finish_artifact,
};
use rivoli_artifact::glimmer::{GLIMMER_LAYER_PREFIX, GLIMMER_LAYER_TENSORS};
use rivoli_artifact::glimmer_config::GlimmerConfig;
use rivoli_artifact::tokenizer::eos_token_ids;

/// The one file carrying Muse Glimmer's stop tokens. Named because three places spell it and
/// [`eos_ids`] refuses on its contents — a literal repeated beside a check that reads it is one
/// rename away from checking a different file than the one it copies.
const GEN: &str = "generation_config.json";

/// Auxiliary files copied beside the weights. `generation_config.json` is **load-bearing and
/// not optional decoration**: it carries `eos_token_id: [200001, 200008]`, and the
/// `text_config` in `config.json` carries only the scalar `200001` (trap 13). A run that took
/// EOS from the config would stop on one of the two.
const AUX: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    GEN,
    "chat_template.jinja",
];

/// The subset of [`AUX`] whose absence is refused BEFORE any weight is read.
///
/// > **PORT NOTE 2026-08-16, and the argument is not the one the reference makes.** There this
/// > list existed because `finish_artifact` downgraded every aux-copy failure to
/// > `eprintln!("WARNING: {name} not copied")` and returned `Ok(())`, so a partial clone
/// > shipped an artifact with no stop tokens announced by one warning line. **This tree's
/// > `finish_artifact` already refuses a missing or empty aux file** — the old tree's defect
/// > was fixed at the shared function rather than per-converter — so absence is a hard failure
/// > either way.
/// >
/// > What this list still buys is WHEN. `finish_artifact` runs last; without this check the
/// > refusal lands after the 53 GB write, hours in, on a checkpoint that could have been
/// > rejected in milliseconds. Kept for that, and only that — the reference's stronger claim
/// > would be false here and is not restated.
///
/// Both files carry a fact that exists nowhere else in the source tree, and both have already
/// cost this repo a round: `generation_config.json` is the only file with the two-element
/// `eos_token_id` (trap 13), and `chat_template.jinja` is the one the memory note
/// `artifact-drops-the-chat-template` records rivoli drifting away from for months precisely
/// because it lived only upstream.
const REQUIRED_AUX: [&str; 2] = [GEN, "chat_template.jinja"];

// The old tree wraps this clap preamble in a `jscpd:ignore` region, on the argument that
// `<last field>: <ty>, } fn main() -> Result<()> { let args = Args::parse();` is a shape clap
// prescribes rather than one this file chose. **Here it needs none: measured 2026-08-16, jscpd
// reports 0 clones over `crates/` with the markers removed.** The nearest twin, `add_indexer.rs`,
// is a two-positional binary like this one but destructures (`let Args { .. } = Args::parse()`)
// and `convert.rs` carries six fields, so neither run matches at 15 tokens. Recorded rather than
// carried over: an exemption that suppresses nothing is a hole in the gate.
#[derive(Parser)]
#[command(about = "Build a Muse Glimmer artifact (bf16 verbatim, or --fp8) from an HF checkpoint")]
struct Args {
    /// The HF checkpoint directory: `config.json` + `model.safetensors.index.json` + shards.
    src_dir: String,
    /// Where to write `resident.safetensors` and `manifest.json`.
    out_dir: String,
    /// Quantize the per-layer projections to fp8-e4m3 at `FP8_BLOCK`. `embed_tokens`,
    /// `lm_head` and every norm are unaffected — see [`is_layer_proj`].
    #[arg(long)]
    fp8: bool,
}

/// The EOS ids `generation_config.json` carries, refusing a file that exists and says nothing.
///
/// **PRESENCE WAS CHECKED AND CONTENT WAS NOT, and that left trap 13 live in a worse spelling.**
/// [`REQUIRED_AUX`] refuses a MISSING file — and in the old tree its own Glimmer fixture wrote
/// `generation_config.json` as `{}`, which passes that check, copies into the artifact, and
/// yields the engine's tokenizer **no ids at all**. The whole response there is one
/// `warn!("decode won't stop on EOS")`, after which `eos.contains(&t)` is false for every token
/// and generation runs to `ngen` every time. That is not hypothetical damage: it is the exact
/// signature behind that tree's `benchmarks.md` retraction, where across 56 runs not one
/// terminated naturally and the model drifted into list scaffolding and then looped.
///
/// So a port does not "stop on one of the two" — it stops on NONE, and it says so in a log line
/// nobody reads three hours into a convert. Checked here rather than in `artifact/tokenizer.rs`
/// because that path is shared with three other models and its tolerance is theirs to keep;
/// this binary converts one checkpoint and knows what that checkpoint has to carry.
///
/// **Non-empty, NOT "exactly two".** Glimmer's file carries `[200001, 200008]`, but pinning the
/// count here would over-fit: the pair is a fact about this checkpoint, not a rule about the
/// architecture. What is architectural is that a decode with zero stop tokens cannot terminate.
fn eos_ids(dir: &str) -> Result<Vec<u32>> {
    // **The ENGINE's parser, not a second one.** In the reference this was fifteen lines
    // re-spelling `Tokenizer::load_eos`, with a comment asserting the two "match" — a claim the
    // shared function makes structural rather than asserted (review, 2026-08-13). What belongs
    // here is the REFUSAL: `artifact/tokenizer.rs` is shared with three other models and warns
    // rather than failing, which is its call to keep; this binary converts one checkpoint and
    // knows what that checkpoint has to carry.
    let ids = eos_token_ids(dir)?;
    ensure!(
        !ids.is_empty(),
        "{dir}/{GEN} carries no usable `eos_token_id`. The file exists, so REQUIRED_AUX is \
         satisfied and the artifact would ship — with NO stop token, which makes every decode \
         run to its token limit (glimmer trap 13). Glimmer's is `[200001, 200008]`",
    );
    Ok(ids)
}

/// Every guard that must fire before a single tensor is read, in the order it can fire.
///
/// Split out of `main` rather than left inline: the reference's `main` runs the checks, the
/// tensor walk, the completeness sweep and the manifest in one body, which is the shape the
/// code-health gate refuses. The split is also the file's own argument made structural — this
/// function is exactly "what is checked before the 53 GB write".
///
/// Returns the ids, because `main`'s last act is comparing the artifact's against them.
///
/// Takes the parsed [`Args`] whole rather than two bare directory strings: what this guards
/// is the INVOCATION, and two adjacent `&str` parameters are exactly the pair a caller can
/// swap without a type saying so — `refuse_writing_into_source` would then be checking the
/// hazard backwards and reporting it as absent.
fn refuse_before_writing(args: &Args) -> Result<(GlimmerConfig, Vec<u32>)> {
    let (src_dir, out_dir) = (args.src_dir.as_str(), args.out_dir.as_str());
    // **Not into the source directory.** The argument — and the words — moved to
    // `SafeWriter::refuse_writing_into_source` on 2026-08-16, when `convert_v4` became the
    // second converter with the same guard and jscpd reported the pair. It is the WRITER's
    // hazard, not this binary's, so it belongs beside `SafeWriter::write`.
    SafeWriter::refuse_writing_into_source(&ArtifactDirs {
        out: out_dir,
        src: src_dir,
    })?;
    for aux in REQUIRED_AUX {
        let p = std::path::Path::new(src_dir).join(aux);
        ensure!(
            p.is_file(),
            "{aux} is missing from {src_dir}. It is not decoration — see REQUIRED_AUX — and \
             `finish_artifact` would refuse it only at the END, three hours into the convert"
        );
    }
    // Parsed, and therefore validated, before a single tensor is read: `GlimmerConfig` is
    // where the layer/RoPE pairing invariant and the width checks live, and a checkpoint that
    // fails them must not produce a half-written artifact.
    let cfg = GlimmerConfig::load(src_dir)?;
    // AFTER the config so `vocab` is available, and still before any tensor is read — nothing
    // between the two touches the checkpoint. The bound is what separates "at least one number
    // parses" from "at least one token this model can emit": an id past the vocabulary is a
    // stop token no argmax can ever return, which is the same unstoppable decode one layer down
    // (review, 2026-08-13).
    let ids = eos_ids(src_dir)?;
    for &id in &ids {
        ensure!(
            (id as usize) < cfg.text.vocab,
            "eos_token_id {id} is past this model's vocabulary of {} — no argmax can return it, \
             so it is a stop token that never fires",
            cfg.text.vocab
        );
    }
    // PRINTED, not just checked. The ids decide whether a decode can terminate at all, they
    // live in an aux file rather than in the manifest, and the operator running this has the
    // checkpoint in front of them — so the one line that lets them notice a wrong set is worth
    // more here than after a decode has run to its token limit.
    eprintln!("convert_glimmer: eos_token_id {ids:?}");
    // The config the checks validated IS the config the manifest is built from — one
    // parse, one value; a second read is the changed-during-the-run class this file's
    // own EOS re-check paragraph treats as real (review 2026-08-16).
    Ok((cfg, ids))
}

/// How many tensors the checkpoint's INDEX declares as vision, which is what makes the
/// exclusion an observation rather than an assumption.
///
/// **Counted from the index, NOT from the opened shards.** `open_indexed` selects whole SHARDS
/// containing at least one wanted tensor, and the `want` predicate excludes vision — so a shard
/// holding only vision tensors is never opened and its tensors never appear in `src.names()`.
/// Counting them there under-reports by however many such shards the checkpoint happens to
/// have, which makes the number a function of its shard boundaries rather than of the model.
/// The fixture is single-shard so it could not show this; review found it 2026-08-11, and the
/// count is the whole reason [`is_vision`] exists as a named predicate.
fn vision_count(src_dir: &str) -> Result<usize> {
    let path = std::path::Path::new(src_dir).join("model.safetensors.index.json");
    let idx = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let idx: serde_json::Value = serde_json::from_str(&idx)?;
    Ok(idx["weight_map"]
        .as_object()
        .context("model.safetensors.index.json has no weight_map")?
        .keys()
        .filter(|n| is_vision(n))
        .count())
}

/// Every tensor the decode path cannot run without, by NAME, before the write.
///
/// Checked here rather than left to the pin: a missing `lm_head` in a partial checkpoint would
/// otherwise surface as a pin-time "tensor not found" long after the 55 GB write.
///
/// **Twelve** tensors per layer: five attention projections (q/k/v/o and the sigmoid gate),
/// three MLP (gate/up/down), four norms — [`GLIMMER_LAYER_TENSORS`] is the list and it is
/// `[&str; 12]`. The reference's comment here said "Nine ... five projections, four norms" and
/// so omitted the whole MLP; the loop always covered them, because it iterates the constant
/// rather than a count. Reconciled against the shipped checkpoint 2026-08-14: 52 x 12 + 3
/// model-level = 627 text tensors, and 627 + 809 vision = the index's 1436.
fn ensure_complete(names: &[String], n_layers: usize) -> Result<()> {
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
    for l in 0..n_layers {
        for t in GLIMMER_LAYER_TENSORS {
            let n = format!("{GLIMMER_LAYER_PREFIX}.{l}.{t}.weight");
            ensure!(names.contains(&n), "the checkpoint has no {n}");
        }
    }
    Ok(())
}

/// What the tensor pass counted — the numbers `main` prints and asserts, travelling as one
/// value so the print and the check cannot drift onto different tallies.
struct Counts {
    verbatim: usize,
    widened: usize,
    quantized: usize,
}

/// **The all-or-nothing check, made BEFORE the pass rather than after it** (review,
/// 2026-08-16). `is_layer_proj` is a pure predicate over `names`, and `names` is complete
/// at the call site, so the same claim the post-pass `ensure!` makes is decidable now — for
/// free, in milliseconds, instead of after ~34 GB of owned bytes and a multi-hour
/// quantization pass. That ordering is this file's own discipline (`refuse_before_writing`);
/// the post-pass check stays, because this one says the NAMES imply the count and that one
/// says the loop actually took the branch.
///
/// Returns the expected count so the post-pass check judges the same number this one did.
fn refuse_partial_quantization(names: &[String], cfg: &GlimmerConfig, fp8: bool) -> Result<usize> {
    let want_quantized = expected_quantized(&cfg.text, fp8)?;
    ensure!(
        names.iter().filter(|n| is_layer_proj(n)).count() * usize::from(fp8) == want_quantized,
        "this checkpoint has {} layer projections; its {} layers imply {want_quantized}",
        names.iter().filter(|n| is_layer_proj(n)).count(),
        cfg.text.n_layers
    );
    Ok(want_quantized)
}

/// The one pass over the sorted names: norms widened to f32, projections quantized under
/// `--fp8`, everything else bf16 verbatim. Split out of `main` on the same argument as
/// [`refuse_before_writing`] — this function is exactly "what the artifact's tensors are".
fn write_tensors<'a>(
    src: &'a Safetensors,
    names: &[String],
    fp8: bool,
) -> Result<(SafeWriter<'a>, Counts)> {
    let mut w = SafeWriter::new();
    let (mut verbatim, mut widened, mut quantized) = (0usize, 0usize, 0usize);
    for name in names {
        if is_vision(name) {
            continue; // counted from the index, not from whichever shards were opened
        }
        if is_norm(name) {
            w.add_widened(src, name)?;
            widened += 1;
        } else if fp8 && is_layer_proj(name) {
            let base = name
                .strip_suffix(".weight")
                .with_context(|| format!("{name} is a layer projection with no `.weight` tail"))?;
            w.add_quantized_fp8(src, base, rivoli_artifact::quant::FP8_BLOCK)?;
            quantized += 1;
        } else {
            // Verbatim, and asserted BF16 rather than "whatever it is": `copy_verbatim` refuses
            // any other dtype, so an fp8 or 4-bit export of this model refuses here instead of
            // being copied into an artifact that claims to be bf16.
            w.copy_verbatim(src, name, Dtype::Bf16)?;
            verbatim += 1;
        }
    }
    Ok((
        w,
        Counts {
            verbatim,
            widened,
            quantized,
        },
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (cfg, ids) = refuse_before_writing(&args)?;
    let skipped = vision_count(&args.src_dir)?;
    std::fs::create_dir_all(&args.out_dir)?;

    // ONE predicate driving both which shards are opened and what is written — the pairing
    // `convert_k3` records as having broken from the other end, where the resident pass looped
    // over every tensor in every opened shard and so made the artifact a function of the
    // checkpoint's shard boundaries rather than of the request.
    let want = |n: &str| !is_vision(n);
    let src = Safetensors::open_indexed(&args.src_dir, want)?;

    let mut names: Vec<String> = src.names().iter().map(|s| s.to_string()).collect();
    names.sort();
    let want_quantized = refuse_partial_quantization(&names, &cfg, args.fp8)?;
    let (w, counts) = write_tensors(&src, &names, args.fp8)?;
    // **The count is asserted, not printed and trusted.** `is_layer_proj` is two string
    // predicates over names this binary does not control; if either ever stopped matching, the
    // artifact would come out with some projections bf16 and some fp8, and the pin's header
    // loop would refuse it — correctly, but hours later and with a message about one tensor
    // rather than about the pass that skipped it.
    //
    // **The expected count is DERIVED, not the literal `8`** (review, 2026-08-16) — see
    // [`expected_quantized`], which is the one spelling of that arithmetic both this check and
    // [`refuse_partial_quantization`] judge against.
    //
    // Guarded on `args.fp8` rather than carrying an `else { 0 }` arm: `quantized` is only
    // incremented inside `write_tensors`' `fp8 &&` branch, so that arm could never fail — and
    // its failure message names a flag the run did not pass.
    if args.fp8 {
        ensure!(
            counts.quantized == want_quantized,
            "--fp8 quantized {} tensors; this checkpoint's {} layers imply {want_quantized}",
            counts.quantized,
            cfg.text.n_layers
        );
    }
    ensure_complete(&names, cfg.text.n_layers)?;

    let out = format!("{}/resident.safetensors", args.out_dir);
    w.write(&out).with_context(|| format!("write {out}"))?;
    eprintln!(
        "convert_glimmer: {} tensors bf16 verbatim, {} projections quantized to fp8, {} norms \
         widened to f32, {skipped} vision tensors skipped -> {out}",
        counts.verbatim, counts.quantized, counts.widened
    );

    // The manifest is the checkpoint's own `config.json` plus the `format` section, so
    // `GlimmerConfig::load` reads the artifact exactly as it read the source — the wrapper and
    // its `text_config` intact, which is what `arch::from_manifest_str` needs.
    //
    // > **MOVED 2026-08-16.** These two statements were spelled out here, with the `FormatMeta`
    // > paragraph beside them; `convert_v4` became the second converter to read `config.json`
    // > and jscpd reported the pair as a clone. They are now
    // > `FormatMeta::manifest_from_config`, which carries both arguments verbatim.
    let manifest =
        FormatMeta::manifest_from_config(&args.src_dir, rivoli_artifact::quant::FP8_BLOCK)?;
    finish_artifact(
        "convert_glimmer",
        ArtifactDirs {
            out: &args.out_dir,
            src: &args.src_dir,
        },
        &manifest,
        &AUX,
    )?;
    // **AGAIN, against the ARTIFACT, and this tree's `finish_artifact` does NOT already
    // establish it.** That function now refuses an aux file that is missing or zero-length
    // after the copy — the old tree's warn-and-continue defect is fixed at the shared function
    // — so what is left is the gap between "a non-empty file arrived" and "the ids the engine
    // will read are the ids this run validated". `std::fs::copy` of a source that changed
    // during a three-hour run reproduces neither, and the engine reads the ARTIFACT's copy and
    // never the source's. Compared as parsed ids rather than bytes because that is the thing
    // the decode depends on; the two are the same file in every non-pathological case.
    //
    // Found by review 2026-08-13 in the reference, against a version whose test already
    // asserted the artifact-side property while the shipped binary only asserted the
    // source-side one.
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
/// > counted as text. In the old tree `tests/glimmer_names.rs` is what actually catches that,
/// > by restating these three prefixes and reconciling 627 + 809 against the shipped index —
/// > so the guard exists, it is just not the one this comment claimed. **That gate has not
/// > been ported** (the shipped index is not on this machine); it arrives with the real
/// > checkpoint work, and until then this class is open (PORT NOTE 2026-08-16).
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
/// asserted, as `L * 4 + 1`, by `crates/cli/tests/glimmer_convert.rs`.)
///
/// **Matched on the tail, not on a `layers.N.` prefix**, so the model-level `norm.weight` and
/// the four per-layer `*_layernorm.weight` take the same path — `"…layernorm.weight"` ends
/// with `"norm.weight"`, so one suffix covers both and a second clause for the layer norms
/// would be dead. Glimmer's QK-norm is weightless and ships no tensor, so there is nothing
/// else this can catch today; if one ever appears it is a norm and this is still right.
fn is_norm(name: &str) -> bool {
    name.ends_with("norm.weight")
}

/// The eight per-layer projections — everything `--fp8` quantizes, and nothing else.
///
/// **`embed_tokens` and `lm_head` are deliberately NOT here**, though they are 5.380 GB of the
/// 55.712: requantizing two tensors read once per token each is its own quality question with
/// a dNLL measurement attached, and this pass does not take it up.
///
/// Composed from two existing predicates rather than listing eight suffixes, which would be a
/// fourth spelling of [`GLIMMER_LAYER_TENSORS`]; the count assertion at the call site is what
/// makes "these are the eight" an observation.
fn is_layer_proj(name: &str) -> bool {
    name.starts_with(GLIMMER_LAYER_PREFIX) && !is_norm(name)
}

/// How many tensors `--fp8` must quantize: the rank-2 entries of [`GLIMMER_LAYER_TENSORS`],
/// times the layer count. Zero without the flag.
///
/// **DERIVED, never the literal `8`.** The rank is the same discriminator
/// `geometry::layer_bytes`, `pin::layer_tails`, `pin::check_layer_headers` and
/// `glimmer_anchor::stored` all branch on, so this joins those four rather than becoming a
/// fifth authority on how many projections a layer has — which is precisely the extra
/// spelling [`is_layer_proj`]'s own doc declines to add.
fn expected_quantized(
    cfg: &rivoli_artifact::glimmer_config::GlimmerTextConfig,
    fp8: bool,
) -> Result<usize> {
    if !fp8 {
        return Ok(0);
    }
    let mut per_layer = 0usize;
    for t in GLIMMER_LAYER_TENSORS {
        per_layer += usize::from(cfg.layer_tensor_shape(t)?.len() == 2);
    }
    Ok(per_layer * cfg.n_layers)
}
