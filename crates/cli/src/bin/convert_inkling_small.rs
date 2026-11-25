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
//! **Three format knobs beyond the repack, all P5 (bytes/token).** `--f4` quantizes the routed
//! experts into `L{ll}.f4` — MXFP4, half of `.mxfp8`'s bytes, which nearly doubles the experts a
//! fixed pin can hold — through the shared [`f4`] module; it is LOSSY, so `--verify` is refused
//! and the paired-dNLL gate scores it. `--fp8` quantizes the five large dense attention
//! projections per layer to block-scaled FP8 in `resident.safetensors`. `--from-mxfp8` reads an
//! already-converted `.mxfp8` artifact (EITHER member's) instead of the checkpoint: the resident
//! set is its own output, so only the `--fp8` rewrite is left, and the expert half is a
//! HARDLINK when `--f4` is absent — 564 GB of Small experts for an inode instead of a copy.
//!
//! Plan of record: `docs/investigations/inkling2-port.md` (stage S6).

use clap::Parser;
use std::os::unix::fs::FileExt;

use anyhow::{Context, Result, ensure};
use rivoli_artifact::format::{
    ArtifactDirs, Dtype, FormatMeta, LayerInputs, LayerShape, Mxfp8Layer, SafeWriter, Safetensors,
    TensorData, f4_layer_range, f4_source, finish_artifact, has_f4_source, has_mxfp8_source,
    mxfp8_expert_prefix, mxfp8_layer_range, mxfp8_source, require_aux,
};
use rivoli_artifact::inkling_config::{Inkling2Config, InklingTextConfig};
use rivoli_artifact::inkling_convert::{
    INKLING_AUX, TIKTOKEN_ARTIFACT, VocabRows, copy_tiktoken, emit_lm_head, emit_verbatim_fmt,
    is_dense_fp8_target, is_expert_tensor, merged_range, preflight, query_target, rename_verbatim,
    require_nonempty,
};
use rivoli_artifact::quant::{
    FP8_BLOCK, VQ_ALIGN, mxfp8_expert_bytes, mxfp8_expert_stride, mxfp8_groups, mxfp8_slot_offsets,
};

// `#[path]` because a bin root resolves its children against `src/bin/` and cargo
// autobin-discovers every `src/bin/*.rs` as its own executable — so the module shared with the
// Mini converter cannot sit beside the roots as a bare `.rs` (the `convert/gpu.rs` precedent).
#[path = "inkling/f4.rs"]
mod f4;

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
             repacked to .mxfp8 or quantized to .f4, resident text weights renamed and \
             optionally fp8; vision excluded by name)"
)]
struct Args {
    /// The checkpoint directory: config.json, model.safetensors.index.json, its shards,
    /// tiktoken/, and the tokenizer files copied into the artifact. Under `--from-mxfp8`, the
    /// directory of an already-converted rivoli `.mxfp8` artifact instead.
    src_dir: String,

    /// Artifact directory to write. Created if absent.
    out_dir: String,

    /// First routed-expert layer to convert. Defaults to the first layer the source holds.
    #[arg(long)]
    from: Option<usize>,

    /// One past the last routed-expert layer. Defaults to the model's `num_hidden_layers`, or to
    /// the range the `--from-mxfp8` source artifact holds.
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

    /// Quantize the five large dense attention projections per layer (q/k/v/attn-gate/o) to
    /// block-scaled FP8-E4M3 at `FP8_BLOCK` in `resident.safetensors`, with the matching
    /// `weight_scale_inv` grids.
    #[arg(long)]
    fp8: bool,

    /// Quantize the routed experts into `L{ll}.f4` (MXFP4, half the bytes of `.mxfp8`) instead of
    /// repacking them. LOSSY by construction, so `--verify` is refused.
    #[arg(long)]
    f4: bool,

    /// Treat `src_dir` as an already-converted rivoli `.mxfp8` artifact — EITHER member's — and
    /// read its `resident.safetensors` and `L{ll}.mxfp8` files instead of a checkpoint. The
    /// expert half is HARDLINKED unless `--f4` is given, so an FP8-trunk artifact costs no extra
    /// disk for its experts.
    #[arg(long)]
    from_mxfp8: bool,
}

fn validate_flags(
    a: &Args,
    n_layers: usize,
    source: Option<std::ops::Range<usize>>,
) -> Result<std::ops::Range<usize>> {
    check_verify_contradictions(a)?;
    check_mode_contradictions(a)?;
    let (lo, hi) = source.map_or((0, n_layers), |r| (r.start, r.end));
    let (from, to) = (a.from.unwrap_or(lo), a.to.unwrap_or(hi));
    ensure!(
        from < to && to <= n_layers,
        "--from {from} / --to {to} is not a non-empty range inside [0, num_hidden_layers={n_layers})"
    );
    Ok(from..to)
}

fn check_verify_contradictions(a: &Args) -> Result<()> {
    ensure!(
        !(a.verify_only && a.resident_only),
        "--verify-only is read-only and --resident-only writes resident weights"
    );
    if a.verify || a.verify_only {
        ensure!(
            !a.f4,
            "--f4 quantizes experts, so --verify byte-identity does not apply"
        );
        ensure!(
            !a.from_mxfp8,
            "--from-mxfp8 hardlinks or quantizes; drop --verify/--verify-only"
        );
    }
    Ok(())
}

fn check_mode_contradictions(a: &Args) -> Result<()> {
    ensure!(
        !(a.resident_only && a.skip_resident),
        "--resident-only and --skip-resident contradict each other"
    );
    ensure!(
        !(a.f4 && a.resident_only),
        "--f4 asks for .f4 layer files and --resident-only writes none"
    );
    ensure!(
        !(a.fp8 && a.skip_resident),
        "--fp8 rewrites the resident set and --skip-resident leaves it alone"
    );
    Ok(())
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
                "--resident-only on {}, which holds no routed-expert layers yet — the \
                 manifest would claim an empty layer range and the loader refuses that. \
                 Convert at least one layer first.",
                out_dir.display()
            )
        }),
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    let dirs = preflight_dirs(
        ArtifactDirs {
            src: &a.src_dir,
            out: &a.out_dir,
        },
        a.from_mxfp8,
    )?;
    let text = load_text_config(&a)?;
    let source = a
        .from_mxfp8
        .then(|| mxfp8_layer_range(&a.src_dir, text.num_hidden_layers))
        .transpose()?;
    let want = validate_flags(&a, text.num_hidden_layers, source)?;
    let range = prepare_out_dir(&a, &text, &want)?;
    let written = execute_passes(&a, dirs, &text, &want)?;
    publish_artifact(dirs, &range, &a)?;
    eprintln!(
        "convert_inkling_small: done — {} → {} ({written} layer files; layers [{}, {}))",
        a.src_dir, a.out_dir, range.start, range.end
    );
    Ok(())
}

fn load_text_config(a: &Args) -> Result<InklingTextConfig> {
    match a.from_mxfp8 {
        true => text_config_from_manifest(std::path::Path::new(&a.src_dir)),
        false => Ok(Inkling2Config::load(&a.src_dir)?.text),
    }
}

fn prepare_out_dir(
    a: &Args,
    t: &InklingTextConfig,
    want: &std::ops::Range<usize>,
) -> Result<std::ops::Range<usize>> {
    std::fs::create_dir_all(&a.out_dir)?;
    ensure!(
        !a.skip_resident
            || std::fs::metadata(format!("{}/resident.safetensors", a.out_dir)).is_ok(),
        "--skip-resident was given but {}/resident.safetensors does not exist — drop the flag.",
        a.out_dir
    );
    let out_path = std::path::Path::new(&a.out_dir);
    if !a.resident_only {
        ensure_same_container(out_path, a.f4)?;
    }
    resolve_manifest_range(
        out_path,
        t.num_hidden_layers,
        (!a.resident_only).then_some(want.clone()),
    )
}

fn execute_passes(
    a: &Args,
    dirs: ArtifactDirs<'_>,
    t: &InklingTextConfig,
    want: &std::ops::Range<usize>,
) -> Result<usize> {
    if !a.verify_only && !a.skip_resident {
        match a.from_mxfp8 {
            true => resident_from_artifact(dirs, a.fp8)?,
            false => resident_set(dirs, t, a.fp8)?,
        };
    }
    match (a.resident_only, a.from_mxfp8) {
        (true, _) => Ok(0),
        (false, true) => layers_from_artifact(dirs, t, a.f4, want),
        (false, false) => write_layer_files(dirs, t, a, want),
    }
}

fn publish_artifact(
    dirs: ArtifactDirs<'_>,
    range: &std::ops::Range<usize>,
    a: &Args,
) -> Result<()> {
    if a.verify_only {
        return Ok(());
    }
    match a.from_mxfp8 {
        true => copy_flat_tiktoken(dirs)?,
        false => copy_tiktoken(dirs)?,
    }
    finish_artifact(
        "convert_inkling_small",
        dirs,
        &build_manifest(std::path::Path::new(&a.src_dir), range, a)?,
        INKLING_AUX,
    )
}

/// Refuse to change an artifact's expert CONTAINER in place.
///
/// The provenance key is what tells the loader which layer files exist, and one artifact holds
/// ONE container: a run that rewrote an `.mxfp8` artifact's layers as `.f4` would stamp
/// `f4_source` over a range in which the untouched layers are still `.mxfp8` — the loader would
/// walk to a `.f4` file that was never written, which reads as a damaged artifact. A fresh
/// directory (or the flag the artifact already matches) is the fix, and saying so costs one
/// comparison.
fn ensure_same_container(out_dir: &std::path::Path, f4: bool) -> Result<()> {
    if std::fs::metadata(out_dir.join("manifest.json")).is_err() {
        return Ok(());
    }
    let dir = out_dir.to_string_lossy();
    let held_f4 = has_f4_source(&dir);
    // A bf16 artifact carries neither key and has no container to disagree about; the `.mxfp8`
    // reader refuses it by name a few lines later, which is the message a reader wants.
    if !held_f4 && !has_mxfp8_source(&dir) {
        return Ok(());
    }
    ensure!(
        held_f4 == f4,
        "{} already holds a .{} expert set and this run writes .{} — one artifact holds ONE \
         container, so convert into a fresh directory or drop the flag",
        out_dir.display(),
        if held_f4 { "f4" } else { "mxfp8" },
        if f4 { "f4" } else { "mxfp8" }
    );
    Ok(())
}

/// The three refusals both members' converters owe before a weight is touched, with the tiktoken
/// path a function of what `src_dir` IS: a checkpoint keeps the rank file NESTED
/// (`tiktoken/tokenizer.model`) and an artifact keeps it FLAT (`tiktoken.model`) — the rename
/// [`copy_tiktoken`] performs on the way in. The order is [`preflight`]'s: the cheap refusals
/// before the multi-hour weight read.
fn preflight_dirs(dirs: ArtifactDirs<'_>, from_artifact: bool) -> Result<ArtifactDirs<'_>> {
    if !from_artifact {
        return preflight(dirs);
    }
    require_aux(dirs.src, INKLING_AUX)?;
    SafeWriter::refuse_writing_into_source(&dirs)?;
    require_nonempty(
        &std::path::Path::new(dirs.src).join(TIKTOKEN_ARTIFACT),
        "the tiktoken vocabulary the runtime reads",
    )?;
    Ok(dirs)
}

/// The shared text config out of an artifact's `manifest.json`, whichever MEMBER wrote it.
///
/// [`Inkling2Config::load`] and `Inkling2LoopConfig::load` each assert their own member's root
/// and absent-or-present fields — right for a checkpoint, wrong here: a `--from-mxfp8` run must
/// accept either member's artifact, and the shared struct is exactly the part both write.
fn text_config_from_manifest(dir: &std::path::Path) -> Result<InklingTextConfig> {
    let path = dir.join("manifest.json");
    let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&raw).with_context(|| format!("parse {}", path.display()))?;
    serde_json::from_value(v["text_config"].clone())
        .with_context(|| format!("{} carries no `text_config`", path.display()))
}

/// The layer range this artifact already claims, or `None` when there is no manifest yet.
///
/// A manifest that EXISTS but carries neither provenance block is an error rather than a fresh
/// start: it is another format's artifact, and stamping this one's provenance over it would
/// silently reinterpret every file in the directory.
///
/// The KEY decides which format's reader runs, and that is not cosmetic: an `--f4` run removes
/// `mxfp8_source` (the loader sniffs it first and would commit to files that are not there), so
/// asking the `.mxfp8` reader about an `.f4` artifact would refuse the converter's own output on
/// the next resume.
fn held_range(
    out_dir: &std::path::Path,
    n_layers: usize,
) -> Result<Option<std::ops::Range<usize>>> {
    if std::fs::metadata(out_dir.join("manifest.json")).is_err() {
        return Ok(None);
    }
    let dir = out_dir.to_string_lossy();
    match has_f4_source(&dir) {
        true => f4_layer_range(&dir, n_layers).map(Some),
        false => mxfp8_layer_range(&dir, n_layers).map(Some),
    }
}

/// Every non-expert TEXT tensor, renamed into the first-party state-dict names and written to
/// `resident.safetensors` in one pass.
///
/// Every shard holding at least one non-expert TEXT tensor is mmap'd (all 48 shards on Small,
/// since each layer's dense weights sit beside its experts; mmap is demand-paged so untouched
/// expert pages are never faulted in). The loop filters each tensor name explicitly so co-located
/// vision and expert tensors never enter `resident.safetensors`.
fn resident_set(dirs: ArtifactDirs<'_>, t: &InklingTextConfig, fp8: bool) -> Result<usize> {
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
            emit_verbatim_fmt(&mut w, rename_verbatim(name)?, (b, dt, sh), fp8_block(fp8))?;
        }
        n += 1;
    }
    write_resident(dirs, &w, (n, fp8), "checkpoint")
}

fn write_resident(
    dirs: ArtifactDirs<'_>,
    w: &SafeWriter<'_>,
    (n, fp8): (usize, bool),
    kind: &str,
) -> Result<usize> {
    w.write(&format!("{}/resident.safetensors", dirs.out))?;
    let dense = if fp8 { ", dense trunk fp8" } else { "" };
    eprintln!("convert_inkling_small: resident set ({kind}) — {n} tensors{dense}");
    Ok(n)
}

/// The `--fp8` tile, as the value the emitter dispatches on. A named function rather than
/// `fp8.then_some(FP8_BLOCK)` at two call sites so the `None` arm reads as the bf16 CONTRACT it
/// is (see [`emit_verbatim_fmt`]) rather than as an absent default.
fn fp8_block(fp8: bool) -> Option<usize> {
    fp8.then_some(FP8_BLOCK)
}

/// The resident half of a `--from-mxfp8` run.
///
/// The source's `resident.safetensors` is ALREADY renamed, widened, transposed and unpadded —
/// it is this converter's own output — so the only work left is the `--fp8` trunk rewrite and a
/// verbatim copy of everything else. Routing it through `emit_verbatim` would transpose
/// `rel_proj` a second time and widen the f32 norms again, which is a silently different model.
fn resident_from_artifact(dirs: ArtifactDirs<'_>, fp8: bool) -> Result<usize> {
    let path = format!("{}/resident.safetensors", dirs.src);
    let src = Safetensors::open_file(&path).with_context(|| format!("open {path}"))?;
    let mut w = SafeWriter::new();
    let mut n = 0usize;
    for name in src.names() {
        let (b, dt, sh) = src.raw(name)?;
        let quant_fp8 = fp8 && dt == Dtype::Bf16;
        if quant_fp8 && is_dense_fp8_target(name) {
            emit_verbatim_fmt(&mut w, name.to_string(), (b, dt, sh), fp8_block(fp8))?;
        } else {
            w.add(name.to_string(), dt, TensorData::new(sh.to_vec(), b));
        }
        n += 1;
    }
    write_resident(dirs, &w, (n, fp8), "from-mxfp8")
}

/// Convert (and optionally verify) every routed-expert layer in the requested range, one shard
/// set at a time.
///
/// A layer's four source tensors are selected by [`mxfp8_expert_prefix`] — the SAME spelling
/// the repacker reads them under — so a wrong prefix cannot select a shard whose tensors the
/// next step cannot find; it selects nothing and `open_indexed` refuses by name.
fn write_layer_files(
    dirs: ArtifactDirs<'_>,
    t: &InklingTextConfig,
    a: &Args,
    want: &std::ops::Range<usize>,
) -> Result<usize> {
    let dims = (t.hidden_size, t.intermediate_size);
    for layer in want.clone() {
        let prefix = mxfp8_expert_prefix(layer);
        let src = Safetensors::open_indexed(dirs.src, |n| n.starts_with(&prefix))
            .with_context(|| format!("open the shard(s) holding layer {layer}'s experts"))?;
        if a.f4 {
            let spec = f4::F4LayerSpec {
                out_dir: dirs.out,
                layer,
                n_experts: t.n_routed_experts,
                dims,
            };
            if f4::quantize_layer_mxfp8(spec, &src, &prefix)? {
                eprintln!("convert_inkling_small: wrote L{layer:02}.f4 (MXFP8 → MXFP4)");
            }
            continue;
        }
        let inputs = LayerInputs {
            tool: "convert_inkling_small",
            out_dir: dirs.out,
            src: &src,
        };
        let shape = LayerShape {
            n_experts: t.n_routed_experts,
            expert_in: dims.0,
            moe_inter: dims.1,
        };
        Mxfp8Layer::new(inputs, layer, shape)
            .mode(!a.verify_only, a.verify || a.verify_only)
            .run()?;
    }
    ensure_layers_present(dirs, want, if a.f4 { "f4" } else { "mxfp8" })?;
    Ok(want.end - want.start)
}

/// The gate: every layer in the range is on disk afterwards, with the extension this run writes.
///
/// `Mxfp8Layer` and [`f4::write_f4_layer`] both resume by SKIPPING an existing file without
/// reading it, so "the run succeeded" and "the file exists" are different claims and only the
/// second one is what the loader needs.
fn ensure_layers_present(
    dirs: ArtifactDirs<'_>,
    want: &std::ops::Range<usize>,
    ext: &str,
) -> Result<()> {
    for layer in want.clone() {
        let path = format!("{}/L{layer:02}.{ext}", dirs.out);
        ensure!(
            std::fs::metadata(&path).is_ok(),
            "{path} is missing after the conversion of layers [{}, {}) — the run reported \
             success without producing it",
            want.start,
            want.end
        );
    }
    Ok(())
}

/// The expert half of a `--from-mxfp8` run: `--f4` quantizes each source layer file into
/// `L{ll}.f4`, and without it the layer file is HARDLINKED.
///
/// The link is the whole point of the mode: the source's bytes are already exactly what the
/// artifact needs, so an FP8-trunk artifact costs an inode per layer instead of a copy of the
/// expert set (564 GB on Small).
fn layers_from_artifact(
    dirs: ArtifactDirs<'_>,
    t: &InklingTextConfig,
    f4_mode: bool,
    want: &std::ops::Range<usize>,
) -> Result<usize> {
    let dims = (t.hidden_size, t.intermediate_size, t.n_routed_experts);
    for layer in want.clone() {
        if f4_mode {
            quantize_artifact_layer(dirs, layer, dims)?;
        } else {
            hardlink_layer(dirs, layer)?;
        }
    }
    let ext = if f4_mode { "f4" } else { "mxfp8" };
    ensure_layers_present(dirs, want, ext)?;
    Ok(want.len())
}

// One reusable read buffer PER WORKER THREAD.
//
// `write_expert_layer` calls the fill closure from every thread it fans a window over, and a
// fresh `vec![0u8; bytes]` per expert would hand the allocator 16.9 MB, 18,432 times on Small —
// re-faulting 4,128 zero pages each round, a page-fault bill larger than the quantization it
// surrounds. One buffer per thread faults its pages once and reuses them; `RefCell` is sound here
// precisely because a `thread_local!` value is never shared between threads.
thread_local! {
    static BLOCK: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Quantize one `L{ll}.mxfp8` of the SOURCE artifact into `L{ll}.f4`.
///
/// The source file is read with `pread` per expert rather than mmap'd: the window
/// [`f4::write_f4_layer`] fills is walked by several threads at once, so a shared file handle
/// with positional reads is the one form that needs no lock, and the kernel's own read-ahead
/// covers the sequential half of the access pattern.
fn quantize_artifact_layer(
    dirs: ArtifactDirs<'_>,
    layer: usize,
    dims: (usize, usize, usize),
) -> Result<()> {
    let (h, i, n) = dims;
    let path = format!("{}/L{layer:02}.mxfp8", dirs.src);
    let file = std::fs::File::open(&path).with_context(|| format!("open {path}"))?;
    let (bytes, stride) = (mxfp8_expert_bytes(h, i), mxfp8_expert_stride(h, i));
    let len = file.metadata()?.len();
    ensure!(
        len == (VQ_ALIGN + n * stride) as u64,
        "{path}: {len} bytes, expected {} for {n} experts of an {stride}-byte MXFP8 block — this \
         is not the layer file the source manifest's dims describe",
        VQ_ALIGN + n * stride
    );
    let off = mxfp8_slot_offsets(h, i);
    let (g_in, g_out) = (mxfp8_groups(h), mxfp8_groups(i));
    let spec = f4::F4LayerSpec {
        out_dir: dirs.out,
        layer,
        n_experts: n,
        dims: (h, i),
    };
    let wrote = f4::write_f4_layer(spec, |e, slot| {
        BLOCK.with(|cell| {
            let mut block = cell.borrow_mut();
            block.resize(bytes, 0);
            file.read_exact_at(&mut block[..], (VQ_ALIGN + e * stride) as u64)?;
            let w13 = SplitW13 {
                gate: f4::Mxfp8Rows::new(
                    &block[off[0]..][..i * h],
                    &block[off[1]..][..i * g_in],
                    (i, h),
                )?,
                up: f4::Mxfp8Rows::new(
                    &block[off[2]..][..i * h],
                    &block[off[3]..][..i * g_in],
                    (i, h),
                )?,
            };
            let w2 = f4::Mxfp8Rows::new(
                &block[off[4]..][..h * i],
                &block[off[5]..][..h * g_out],
                (h, i),
            )?;
            f4::quantize_pair_f4(slot, &w13, &w2, (h, i))
        })
    })?;
    if wrote {
        eprintln!("convert_inkling_small: wrote L{layer:02}.f4 (MXFP8 → MXFP4, {n} experts)");
    }
    Ok(())
}

/// A source layer file's DEINTERLEAVED `w13` spans, presented as the interleaved view
/// [`f4::quantize_expert_f4`] walks: even rows are `gate`, odd rows `up`.
///
/// A `.mxfp8` layer file stores gate and up as separate spans (the repack's own de-interleave),
/// while a checkpoint's `w13_weight` interleaves them — so this adapter is what lets ONE expert
/// loop serve both sources instead of a second loop that would have to be kept in step.
struct SplitW13<'a> {
    gate: f4::Mxfp8Rows<'a>,
    up: f4::Mxfp8Rows<'a>,
}

impl f4::RowSource for SplitW13<'_> {
    fn row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        if row.is_multiple_of(2) {
            self.gate.row(row / 2, out)
        } else {
            self.up.row(row / 2, out)
        }
    }
}

/// Link the source artifact's `L{ll}.mxfp8` into the output, or keep the one already there.
///
/// A hardlink, not a copy: the bytes are already the artifact's, so a link costs an inode where a
/// copy costs the whole expert set. It needs both directories on ONE filesystem, and the error
/// says so rather than falling back to a copy that would silently fill the disk.
fn hardlink_layer(dirs: ArtifactDirs<'_>, layer: usize) -> Result<()> {
    let src = format!("{}/L{layer:02}.mxfp8", dirs.src);
    let dst = format!("{}/L{layer:02}.mxfp8", dirs.out);
    if std::fs::metadata(&dst).is_ok() {
        return Ok(());
    }
    std::fs::hard_link(&src, &dst).with_context(|| {
        format!(
            "link {src} → {dst}. A hardlink needs both directories on ONE filesystem: for a \
             cross-device artifact use --f4 (which rewrites the experts), or write the output \
             beside the source"
        )
    })?;
    Ok(())
}

/// `--from-mxfp8`'s tiktoken copy: an artifact already holds the rank file FLAT, so there is no
/// nested path to descend ([`copy_tiktoken`]'s job) — only the copy and the same arrival check.
fn copy_flat_tiktoken(dirs: ArtifactDirs<'_>) -> Result<()> {
    rivoli_artifact::inkling_convert::copy_tiktoken_from(dirs, TIKTOKEN_ARTIFACT)
}

/// `manifest.json` = the source config VERBATIM (wrapper and all) + the `format` section +
/// this member's provenance blocks.
///
/// Under `--from-mxfp8` the source is an ARTIFACT, so its manifest is loaded verbatim instead of
/// being rebuilt from `config.json` (an artifact need not carry one), and `format` is restamped
/// because this build's VQ parameters are what the loader checks.
///
/// **`mxfp8_source` and `f4_source` are mutually exclusive here**, for the reason
/// [`held_range`] gives: the loader sniffs `mxfp8_source` first, so an `.f4` artifact carrying
/// both would commit to layer files it does not hold.
fn build_manifest(
    src_dir: &std::path::Path,
    layers: &std::ops::Range<usize>,
    a: &Args,
) -> Result<serde_json::Value> {
    let src_str = src_dir.to_string_lossy();
    let mut manifest = match a.from_mxfp8 {
        true => {
            let path = src_dir.join("manifest.json");
            let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            serde_json::from_slice(&raw).with_context(|| format!("parse {}", path.display()))?
        }
        false => FormatMeta::manifest_from_config(src_dir, FP8_BLOCK)?,
    };
    manifest["format"] = serde_json::to_value(FormatMeta::current(FP8_BLOCK))?;
    let mut prov = manifest.get("inkling_source").cloned().unwrap_or_else(|| {
        serde_json::json!({
            "tool": "convert_inkling_small",
            "src": src_str.as_ref(),
            "member": "inkling2",
        })
    });
    prov["tool"] = serde_json::json!("convert_inkling_small");
    if a.fp8 {
        prov["dense"] = serde_json::json!("fp8-block128");
    }
    manifest["inkling_source"] = prov;
    // Which container the layer files are in is a fact about DISK, so a `--resident-only` run —
    // which writes none — must leave both keys exactly as it found them and only record the
    // trunk. Restamping `mxfp8-streamed` over an `.f4` artifact would be a manifest that lies
    // about every file in the directory.
    if !a.resident_only {
        let experts = if a.f4 {
            "f4-streamed"
        } else {
            "mxfp8-streamed"
        };
        manifest["inkling_source"]["experts"] = serde_json::json!(experts);
        if a.f4 {
            // `.f4` REPLACES the `.mxfp8` set, so the other key must go: the loader sniffs
            // `mxfp8_source` first and would commit to files this artifact does not hold.
            let _ = manifest
                .as_object_mut()
                .context("the manifest is not a JSON object")?
                .remove("mxfp8_source");
            manifest["f4_source"] = f4_source("convert_inkling_small", &src_str, layers.clone());
            // `f4_source`'s literal says `fp4->fp4 (repack)`, which is true of V4 and K3 and
            // false here: every expert this run wrote was fitted from MXFP8 by `quantize_f4_row`.
            manifest["f4_source"]["chain"] = serde_json::json!("mxfp8 -> fp4 (quantize)");
        } else if a.from_mxfp8 {
            // The layer files are HARDLINKS of the source's, so the chain is the source's own
            // repack with this artifact's directory and range.
            manifest["mxfp8_source"] =
                mxfp8_source("convert_inkling_small", &src_str, layers.clone());
            manifest["mxfp8_source"]["chain"] = serde_json::json!("fp8->fp8 (hardlink)");
        } else {
            manifest["mxfp8_source"] =
                mxfp8_source("convert_inkling_small", &src_str, layers.clone());
        }
    }
    Ok(manifest)
}
