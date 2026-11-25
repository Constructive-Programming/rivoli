//! `convert_inkling_small` — the Inkling-2-Small-MXFP8 checkpoint → a rivoli STREAMING
//! artifact.
//!
//! **Text-only, like its Mini sibling, and MXFP8-PRESERVING where Mini dequantizes.** Small
//! ships every one of its 48 layers' routed experts already at 8 bits (one e4m3 value per
//! weight, one ue8m0 scale byte per 32 along the input dim), so the routed half of this
//! converter is a REPACK into `L{ll}.mxfp8` — nothing is fit, nothing is re-rounded, and
//! `--verify` demands byte identity because there is no loss to excuse. The resident half is
//! the same rename/widen/transpose stack Mini uses, through the same
//! [`rivoli_artifact::inkling_convert`] tables, because the two members are one text stack.
//!
//! **Shard selection is per layer, by tensor name.** `Safetensors::open_indexed` resolves a
//! predicate through the checkpoint's `model.safetensors.index.json`, so converting layer 7
//! mmaps the one or two shards that hold layer 7's four expert tensors — never the 619 GB
//! checkpoint. That is what makes a download-in-progress convertible layer by layer, and it
//! is why `--from`/`--to` exist.
//!
//! **What is excluded, and how.** `model.visual.*` by prefix and the routed experts by
//! [`is_expert_tensor`] — the vision tower is not text v1 and the experts are the layer
//! files' business. Unlike Mini's converter there is no vendored census here: the 48-layer
//! Small set is classified by three predicates, and any TEXT name that survives them must
//! match [`rename_verbatim`] or the run REFUSES — an unrecognised tensor is a hard error, not
//! a silently dropped one.
//!
//! Plan of record: `docs/investigations/inkling2-port.md` (stage S6).

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli_artifact::format::{
    Dtype, FormatMeta, LayerInputs, LayerShape, Mxfp8Layer, SafeWriter, Safetensors, TensorData,
    finish_artifact, mxfp8_expert_prefix, mxfp8_layer_range, mxfp8_source,
};
use rivoli_artifact::inkling_config::{Inkling2Config, InklingTextConfig};
use rivoli_artifact::inkling_convert::{
    INKLING_AUX, VocabRows, copy_tiktoken, emit_lm_head, emit_verbatim, is_expert_tensor,
    merged_range, preflight, query_target, rename_verbatim,
};
use rivoli_artifact::quant::FP8_BLOCK;

/// The multimodal wrapper's vision tower. Excluded BY PREFIX: every one of its names starts
/// with this, and none of them is a text tensor.
const VISION_PREFIX: &str = "model.visual.";

/// The head before unpadding — the one text tensor whose rows are cut rather than copied.
const UNEMBED: &str = "model.llm.unembed.weight";

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
#[derive(Parser)]
#[command(
    name = "convert_inkling_small",
    about = "Inkling-2-Small-MXFP8 checkpoint → a rivoli streaming artifact (routed experts \
             repacked to .mxfp8, resident text weights renamed; vision excluded by name)"
)]
struct Args {
    /// The checkpoint directory: config.json, model.safetensors.index.json, its shards,
    /// tiktoken/, and the tokenizer files copied into the artifact.
    src_dir: String,

    /// Artifact directory to write. Created if absent.
    out_dir: String,

    /// First routed-expert layer to convert.
    #[arg(long, default_value_t = 0)]
    from: usize,

    /// One past the last routed-expert layer. Defaults to the model's `num_hidden_layers`.
    #[arg(long)]
    to: Option<usize>,

    /// Re-read every layer file after writing it and compare it byte-for-byte with the source.
    #[arg(long)]
    verify: bool,

    /// Read-only: verify the layer files in range and write NOTHING (no manifest, no copy).
    #[arg(long)]
    verify_only: bool,

    /// Convert only the resident set and the manifest; no routed-expert layer files.
    #[arg(long)]
    resident_only: bool,

    /// Leave an existing resident set alone and convert layer files only. The manifest's
    /// layer range GROWS by this run's range, so the two runs cannot leave a gap.
    #[arg(long)]
    skip_resident: bool,
}

fn validate_flags(a: &Args, n_layers: usize) -> Result<std::ops::Range<usize>> {
    ensure!(
        !(a.verify_only && a.resident_only),
        "--verify-only is read-only by contract and --resident-only WRITES the resident \
         weights — the two flags contradict each other"
    );
    ensure!(
        !(a.resident_only && a.skip_resident),
        "--resident-only says to write the resident set and --skip-resident says to leave it \
         alone — that run would convert nothing at all"
    );
    let to = a.to.unwrap_or(n_layers);
    ensure!(
        a.from < to && to <= n_layers,
        "--from {} / --to {to} is not a non-empty range inside [0, num_hidden_layers={n_layers})",
        a.from
    );
    Ok(a.from..to)
}

fn resolve_manifest_range(
    out_dir: &std::path::Path,
    n_layers: usize,
    converting: Option<std::ops::Range<usize>>,
) -> Result<std::ops::Range<usize>> {
    let held = held_range(out_dir, n_layers)?;
    match converting {
        Some(want) => merged_range(held, want),
        None => held.with_context(|| {
            format!(
                "--resident-only on {}, which holds no .mxfp8 layers yet — the \
                 manifest would claim an empty layer range and the loader refuses that. \
                 Convert at least one layer first.",
                out_dir.display()
            )
        }),
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    let (src_dir, out_dir) = (a.src_dir.as_str(), a.out_dir.as_str());
    let dirs = preflight(rivoli_artifact::format::ArtifactDirs {
        src: src_dir,
        out: out_dir,
    })?;
    let cfg = Inkling2Config::load(src_dir)?;
    let t = &cfg.text;
    let want = validate_flags(&a, t.num_hidden_layers)?;
    std::fs::create_dir_all(out_dir)?;
    ensure!(
        !a.skip_resident || std::fs::metadata(format!("{out_dir}/resident.safetensors")).is_ok(),
        "--skip-resident was given but {out_dir}/resident.safetensors does not exist — this \
         artifact has no resident set yet, so the run would produce layer files nothing can \
         load. Drop the flag."
    );
    let range = resolve_manifest_range(
        std::path::Path::new(out_dir),
        t.num_hidden_layers,
        (!a.resident_only).then_some(want),
    )?;
    if !a.verify_only && !a.skip_resident {
        resident_set(dirs, t)?;
    }
    let written = match a.resident_only {
        true => 0,
        false => write_layer_files(dirs, t, &a)?,
    };
    if !a.verify_only {
        copy_tiktoken(dirs)?;
        finish_artifact(
            "convert_inkling_small",
            dirs,
            &build_manifest(std::path::Path::new(src_dir), range.clone())?,
            INKLING_AUX,
        )?;
    }
    eprintln!(
        "convert_inkling_small: done — {src_dir} → {out_dir} ({written} layer files written \
         or verified; artifact holds layers [{}, {}))",
        range.start, range.end
    );
    Ok(())
}

/// The layer range this artifact already claims, or `None` when there is no manifest yet.
///
/// A manifest that EXISTS but carries no `mxfp8_source` is an error rather than a fresh start:
/// it is another member's (or another format's) artifact, and stamping this one's provenance
/// over it would silently reinterpret every file in the directory.
fn held_range(
    out_dir: &std::path::Path,
    n_layers: usize,
) -> Result<Option<std::ops::Range<usize>>> {
    if std::fs::metadata(out_dir.join("manifest.json")).is_err() {
        return Ok(None);
    }
    mxfp8_layer_range(&out_dir.to_string_lossy(), n_layers).map(Some)
}

/// Every non-expert TEXT tensor, renamed into the first-party state-dict names and written to
/// `resident.safetensors` in one pass.
///
/// Every shard holding at least one non-expert TEXT tensor is mmap'd (all 48 shards on Small,
/// since each layer's dense weights sit beside its experts; mmap is demand-paged so untouched
/// expert pages are never faulted in). The loop filters each tensor name explicitly so co-located
/// vision and expert tensors never enter `resident.safetensors`.
fn resident_set(
    dirs: rivoli_artifact::format::ArtifactDirs<'_>,
    t: &InklingTextConfig,
) -> Result<usize> {
    let src = Safetensors::open_indexed(dirs.src, |n| {
        !n.starts_with(VISION_PREFIX) && !is_expert_tensor(n)
    })
    .with_context(|| format!("open {}'s non-expert text shards", dirs.src))?;
    let mut w = SafeWriter::new();
    let mut n = 0usize;
    for name in src.names() {
        if name.starts_with(VISION_PREFIX) || is_expert_tensor(name) {
            continue;
        }
        if name == UNEMBED {
            let (b, sh) = src.typed(name, Dtype::Bf16)?;
            let rows = VocabRows {
                vocab_size: t.vocab_size,
                unpadded: t.unpadded_vocab_size,
                hidden: t.hidden_size,
            };
            emit_lm_head(&mut w, name, (b, sh), rows)?;
        } else if let Some(target) = query_target(name)? {
            // The `attn_res` banks ship f32 and stay f32: the engine's query operands are
            // already that type, so there is nothing to widen.
            let (b, dt, sh) = src.raw(name)?;
            w.add(target, dt, TensorData::new(sh.to_vec(), b));
        } else {
            let (b, dt, sh) = src.raw(name)?;
            emit_verbatim(&mut w, rename_verbatim(name)?, (b, dt, sh))?;
        }
        n += 1;
    }
    w.write(&format!("{}/resident.safetensors", dirs.out))?;
    eprintln!(
        "convert_inkling_small: resident set — {n} TEXT tensors (vision and routed experts \
         excluded by name)"
    );
    Ok(n)
}

/// Convert (and optionally verify) every routed-expert layer in the requested range, one shard
/// set at a time.
///
/// A layer's four source tensors are selected by [`mxfp8_expert_prefix`] — the SAME spelling
/// the repacker reads them under — so a wrong prefix cannot select a shard whose tensors the
/// next step cannot find; it selects nothing and `open_indexed` refuses by name.
fn write_layer_files(
    dirs: rivoli_artifact::format::ArtifactDirs<'_>,
    t: &InklingTextConfig,
    a: &Args,
) -> Result<usize> {
    let to = a.to.unwrap_or(t.num_hidden_layers);
    for layer in a.from..to {
        let src =
            Safetensors::open_indexed(dirs.src, |n| n.starts_with(&mxfp8_expert_prefix(layer)))
                .with_context(|| format!("open the shard(s) holding layer {layer}'s experts"))?;
        let inputs = LayerInputs {
            tool: "convert_inkling_small",
            out_dir: dirs.out,
            src: &src,
        };
        let shape = LayerShape {
            n_experts: t.n_routed_experts,
            expert_in: t.hidden_size,
            moe_inter: t.intermediate_size,
        };
        Mxfp8Layer::new(inputs, layer, shape)
            .mode(!a.verify_only, a.verify || a.verify_only)
            .run()?;
    }
    // The gate: every layer in the range is on disk afterwards. `Mxfp8Layer` resumes by
    // SKIPPING an existing file without reading it, so "the run succeeded" and "the file
    // exists" are different claims and only the second one is what the loader needs.
    for layer in a.from..to {
        let path = format!("{}/L{layer:02}.mxfp8", dirs.out);
        ensure!(
            std::fs::metadata(&path).is_ok(),
            "{path} is missing after the conversion of layers [{}, {to}) — the run reported \
             success without producing it",
            a.from
        );
    }
    Ok(to - a.from)
}

/// `manifest.json` = the source config VERBATIM (wrapper and all) + the `format` section +
/// this member's two provenance blocks. `fp8_block` is inert here (Glimmer's precedent) —
/// `format` stamps this build's compiled-in value and the load side only checks its shape.
fn build_manifest(
    src_dir: &std::path::Path,
    layers: std::ops::Range<usize>,
) -> Result<serde_json::Value> {
    let src_str = src_dir.to_string_lossy();
    let mut manifest = FormatMeta::manifest_from_config(src_dir, FP8_BLOCK)?;
    manifest["inkling_source"] = serde_json::json!({
        "tool": "convert_inkling_small",
        "src": src_str.as_ref(),
        "member": "inkling2",
        "experts": "mxfp8-streamed",
    });
    manifest["mxfp8_source"] = mxfp8_source("convert_inkling_small", &src_str, layers);
    Ok(manifest)
}
