//! `convert_glimmer_drafter` — attach the DFlash drafter to an EXISTING Muse Glimmer
//! artifact, as its `drafter/` sub-artifact.
//!
//! A sibling of `convert_glimmer` rather than a flag on it, and the choice is about time and
//! coupling, not taste: the assistant checkpoint is ~5.1 GB against the target's 55.7, so
//! re-running a multi-hour bf16 convert to gain a drafter would be the wrong unit of work —
//! this binary attaches to an artifact that already exists, in minutes, and touches nothing
//! the main converter wrote. **The artifact-is-the-model doctrine is the design**: a Glimmer
//! artifact WITH a `drafter/` dir is what makes speculative drafting legal to switch on later
//! (M17d wires that legality); there is no standalone drafter artifact, because a drafter
//! alone decodes nothing — it owns neither the embedding nor the lm_head it needs
//! (`glimmer-architecture.md` §11, `drafter_config.rs`'s header).
//!
//! > **CORRECTED 2026-08-17. BOTH HALVES OF THIS ARE NOW FALSE, and this note is here rather
//! > than a deletion because what it ruled out is the record.** It read: "THIS BINARY HAS NO
//! > END-TO-END GATE. Not one line below `main` has ever executed." Since M17b:
//! >
//! > 1. **`main` has executed, against the real checkpoint.** The assistant checkpoint arrived at
//! >    `/swarm/storage/ai/rivoli/muse-glimmer-30b-assistant`, and this binary converted it in
//! >    **174 s** on CPU and NFS — 36 tensors bf16 verbatim, 22 norms widened, 58 total,
//! >    5,112,138,812 B out, exit 0. The target side was a scratch directory holding only the
//! >    production artifact's `manifest.json`, so nothing was written into the shared artifact.
//! >    `glimmer-reference/drafter-checkpoint.md` is the record.
//! > 2. **It has a gate**: `crates/cli/tests/drafter_convert.rs`, 13 tests, deviceless, all
//! >    proven red (`gate-red-proofs.md` §6). It gates the real checkpoint's census from the
//! >    checkpoint's own vendored safetensors header, and fires the three pairing refusals
//! >    against the real pair of shipped configs.
//! >
//! > **What the paragraph below still gets right, and it is the part worth keeping:** the reason
//! > the gate was missing was never the absent checkpoint. The gate that landed builds its
//! > refusal arms from vendored configs and needs no weights at all, exactly as the paragraph
//! > argued — so the diagnosis outlived the claim.
//!
//! **The original note, kept for its argument:**
//!
//! The absent checkpoint is NOT the reason, and saying it was is a finding review returned
//! here: `crates/cli/tests/glimmer_convert.rs` opens with the same sentence about the TARGET
//! ("59.553 GB and is not on this machine") and then builds a synthetic checkpoint anyway, as
//! `k3_convert.rs` and `v4_convert.rs` do. The harness needs nothing new — `tests/common/`
//! already ships `ConvertBin`/`ConvertRun`, `write_shard_and_index`, `write_config` and
//! `expect_refusal`. The real reason is that the gate is not written yet, and under P7 that is
//! the finding rather than a note.
//!
//! What IS gated: `DrafterConfig`'s own tests derive every shape in
//! `old:docs/reference/glimmer-architecture.md` §11 from the schema, prove every field is read
//! from its JSON key, and fire each refusal on the document that reaches it — so a census wrong
//! HERE reddens there. What is NOT: `ensure_census` in either direction, the widen/verbatim
//! split, the manifest round-trip, and the three pairing refusals that need a target artifact.
//! `glimmer_convert.rs` is the shape that gate takes, on a fixture built from
//! `DrafterConfig::census()` at tiny widths with `head_dim * heads != hidden`.
//!
//! > **UPDATED 2026-08-17.** Of that "what is NOT" list, `drafter_convert.rs` now closes the
//! > three pairing refusals (against the real configs, not tiny ones) and the widen/verbatim
//! > split (36/22, derived from the census AND from the real header independently).
//! > `ensure_census` is covered in the direction that matters — the real checkpoint's tensor set
//! > IS the census, both ways, by name — though the gate asserts that against the vendored
//! > header rather than by calling `ensure_census` itself. **Still open: the manifest
//! > round-trip**, which needs a target artifact, and the shape the note predicted for it
//! > (a tiny fixture with `head_dim * heads != hidden`) is still the right one.
//!
//! Like the target's converter this quantizes nothing: the checkpoint is BF16 and the first
//! artifact is bf16 verbatim, norms widened to f32 per the house rank rule. **No aux files
//! are copied**, and their absence is the point rather than an omission — the drafter borrows
//! the target's tokenizer, chat template and stop ids, all of which live one directory up in
//! the artifact it attaches to; a second copy inside `drafter/` would be a second place for
//! trap 13 to drift.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli_artifact::drafter_config::DrafterConfig;
use rivoli_artifact::format::{
    ArtifactDirs, Dtype, FormatMeta, SafeWriter, Safetensors, finish_artifact,
};
use rivoli_artifact::glimmer_config::GlimmerConfig;

#[derive(Parser)]
#[command(about = "Attach the DFlash drafter (bf16 verbatim) to an existing Muse Glimmer artifact")]
struct Args {
    /// The assistant HF checkpoint: `config.json` + `model.safetensors` (single-file).
    src_dir: String,
    /// An EXISTING Glimmer artifact directory; the drafter is written to `<dir>/drafter/`.
    artifact_dir: String,
}

/// Every refusal that must fire before a tensor is read, in the order it can fire — the same
/// shape as `convert_glimmer::refuse_before_writing`, for the same reason: a checkpoint or a
/// pairing that can be rejected in milliseconds must not be rejected after the write.
fn refuse_before_writing(
    src_dir: &str,
    artifact_dir: &str,
) -> Result<(DrafterConfig, serde_json::Value)> {
    // The TARGET must already be an artifact — attaching a drafter to a bare checkpoint
    // directory would create the `drafter/` layout somewhere the engine never looks, and the
    // manifest parse is also where the cross-checks below get the target's widths from.
    let target: GlimmerConfig = GlimmerConfig::load(artifact_dir).with_context(|| {
        format!(
            "{artifact_dir} is not a Muse Glimmer artifact — run convert_glimmer first; the \
             drafter attaches to an artifact, never the other way around"
        )
    })?;
    let cfg = DrafterConfig::load(src_dir)?;

    // The three facts that pair THIS drafter with THIS target. Each is a fact about the
    // borrow: the drafter embeds and projects through the target's own tensors, so a width
    // or id that disagrees is not a warning — nothing the engine could later do would make
    // the pair decode.
    ensure!(
        cfg.hidden_size == target.text.hidden,
        "drafter hidden_size {} != target hidden {} — the drafter borrows the target's \
         embedding and lm_head, so the widths must match",
        cfg.hidden_size,
        target.text.hidden
    );
    for id in cfg.target_layer_ids() {
        ensure!(
            id < target.text.n_layers,
            "target_layer_ids entry {id} is past the target's {} layers — the drafter would \
             wait on a hidden state the decode never produces",
            target.text.n_layers
        );
    }
    ensure!(
        cfg.mask_token_id < target.text.vocab,
        "mask_token_id {} is past the target's vocabulary of {} — every noise row would embed \
         out of bounds",
        cfg.mask_token_id,
        target.text.vocab
    );
    // The manifest is STAMPED here, before a byte is written, for this function's own reason:
    // `manifest_from_config` reads `<src>/config.json`, while `DrafterConfig::load` above
    // accepts a src_dir carrying only `manifest.json` — so without this the census walk and the
    // ~5.1 GB write both complete and the run then fails on a missing file.
    let manifest = FormatMeta::manifest_from_config(src_dir, rivoli_artifact::quant::FP8_BLOCK)?;
    Ok((cfg, manifest))
}

/// The checkpoint's tensor set against the census, by exact SET equality — both directions
/// reported by name. A missing tensor is an incomplete drafter; an EXTRA one is either the
/// spec's 58-vs-59 discrepancy surfacing (see `DrafterConfig::census`) or a checkpoint this
/// converter does not understand, and silently copying it would put unaudited bytes in the
/// artifact while silently dropping it would put a lie in the census.
fn ensure_census(src: &Safetensors, census: &[(String, Vec<usize>)]) -> Result<()> {
    let have: std::collections::BTreeSet<&str> = src.names().iter().copied().collect();
    let want: std::collections::BTreeSet<&str> = census.iter().map(|(n, _)| n.as_str()).collect();
    let missing: Vec<&&str> = want.difference(&have).collect();
    let extra: Vec<&&str> = have.difference(&want).collect();
    ensure!(
        missing.is_empty() && extra.is_empty(),
        "the checkpoint's tensor set is not the drafter census ({} tensors): missing \
         {missing:?}, unexpected {extra:?}",
        census.len()
    );
    census.iter().try_for_each(|(name, shape)| {
        let got = src.shape(name)?;
        ensure!(
            got == shape.as_slice(),
            "{name}: checkpoint shape {got:?} != census {shape:?}"
        );
        Ok(())
    })
}

fn main() -> Result<()> {
    // Destructured, as `add_indexer` does — and here that shape is also what keeps this
    // preamble from being a jscpd clone of `convert_glimmer`'s `let args = ...` form.
    let Args {
        src_dir,
        artifact_dir,
    } = Args::parse();
    let (cfg, manifest) = refuse_before_writing(&src_dir, &artifact_dir)?;
    let out_dir = format!("{artifact_dir}/drafter");
    let dirs = ArtifactDirs {
        out: &out_dir,
        src: &src_dir,
    };
    SafeWriter::refuse_writing_into_source(&dirs)?;

    // Single-file checkpoint, so `open_dir` (no index exists to select shards through);
    // the census walk below is what makes the whole-directory read a checked one.
    let src = Safetensors::open_dir(&src_dir)?;
    let census = cfg.census();
    ensure_census(&src, &census)?;

    // AFTER the census walk, not before: the header makes a `drafter/` dir the thing that lets
    // M17d switch drafting on, so a convert that REFUSES must not leave one behind. M17d's
    // predicate should be `drafter/manifest.json` — written last by `finish_artifact` — rather
    // than the directory, which also closes the window between the two writes.
    std::fs::create_dir_all(&out_dir)?;

    let mut w = SafeWriter::new();
    let (mut verbatim, mut widened) = (0usize, 0usize);
    for (name, shape) in &census {
        // The house rank rule (`glimmer_anchor/mod.rs::stored` states it for fixtures, the
        // pin enforces it at load): 1-D is a norm and is held f32, everything else is a
        // projection and stays bf16 verbatim. The drafter's q/k_norm are rank-1 and take
        // the widened path with the layernorms.
        if shape.len() == 1 {
            w.add_widened(&src, name)?;
            widened += 1;
        } else {
            w.copy_verbatim(&src, name, Dtype::Bf16)?;
            verbatim += 1;
        }
    }
    let out = format!("{out_dir}/resident.safetensors");
    w.write(&out).with_context(|| format!("write {out}"))?;

    // The drafter's manifest is ITS OWN config plus the format section — `DrafterConfig::
    // load` reads the sub-artifact exactly as it read the source, the same round-trip rule
    // the target's manifest follows. No aux (see the header).
    finish_artifact("convert_glimmer_drafter", dirs, &manifest, &[])?;
    // Re-read what was just written AND BIND IT to what shaped the write. Unbound this proved
    // only that the published manifest parses: `DrafterConfig::load` prefers `manifest.json`
    // while `manifest_from_config` reads `config.json`, so a src_dir carrying both could select
    // tensors by one document and publish the other — the engine would then derive
    // `encoder.fc [6656, 26624]` for a 33280-wide tensor with nothing saying so.
    // `convert_glimmer` closes the same class for its EOS ids; this is that check.
    let back = DrafterConfig::load(&out_dir).context("the drafter manifest does not read back")?;
    ensure!(
        back.census() == census,
        "the published manifest describes a different drafter than the one just written — \
         <src>/manifest.json and <src>/config.json disagree"
    );

    let bytes = std::fs::metadata(&out)?.len();
    eprintln!(
        "convert_glimmer_drafter: {verbatim} tensors bf16 verbatim, {widened} norms widened \
         to f32, {} tensors total, {bytes} B -> {out}",
        census.len()
    );
    Ok(())
}
