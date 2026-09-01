//! `convert_qwen` — the Qwen3.8-Flash-Next FP8 checkpoint → a rivoli artifact.
//!
//! **Nothing is re-quantized.** The source ships block-128 fp8-e4m3 on its routed experts and a
//! per-TENSOR fp8 scale on its 51.2 G-element n-gram table; both are formats this engine already
//! reads, so every weight byte is copied and the only thing this tool produces is a widened copy
//! of each scale grid. That is `convert_k3`/`convert_v4`'s argument, and it is why re-rounding a
//! published 8-bit source is not on the table: it introduces error a gate would then have to
//! bound. Host RAM peak is the widened scales alone — 29.5 MB against 172.76 GiB of source.
//!
//! **The census IS the exclusion list, and it is exhaustive both ways.** Before a shard is
//! opened, every one of the source index's 152,089 names is collapsed to one of
//! `tensor-families.tsv`'s 109 patterns, its indices are bounded against the CONFIG (so
//! `layers.2.ple.*` is refused even though `layers.{L}.ple.*` is a real family — trap T12), and
//! every pattern's observed count is confronted with the census's. MTP's 35 families and
//! vision's 21 are excluded by NAME with their tensor and byte totals printed, never by a prefix
//! glob and never as "everything else": a census that balances a grand total while one class
//! vanished is the failure this closure exists to make impossible. A name matching no pattern is
//! a hard error, which is how an upstream revision that ADDS a family becomes loud.
//!
//! **What differs from the four converters before it:**
//!
//! * The routed scale grids are **BF16, not F32** — `copy_fp8` cannot read them — and their
//!   orientation is **per projection**: `down_proj` (weight [2560, 640]) has a [20, 5] grid
//!   while `gate_proj`/`up_proj` ([640, 2560]) have [5, 20]. Building one orientation for all
//!   three hands every 128x128 tile a different tile's scale over 80.5 GB, with no crash and no
//!   length error, so [`add_fp8_bf16_scaled`] derives the grid from each weight's own shape.
//! * The n-gram table is **128 ordered row ranges**, not a matrix: a decode gathers 16 rows of
//!   160 fp8 bytes per token, so the shards are copied verbatim in index order and asserted to
//!   tile `[0, padded_rows)` exactly.
//! * The three I64 hash buffers are passed through **and byte-compared** against
//!   `census::qwen::ngram_hash`'s re-derivation. They are the only tensors in the artifact whose
//!   VALUES this tool checks, and they are worth it: a wrong seed or a wrong `global_head_idx`
//!   hashes every token to the wrong rows and changes no shape.
//! * The chat template ships in **two homes that must agree** — `chat_template.jinja` and
//!   `tokenizer_config.json`'s `chat_template` key — and both are pinned equal at convert time.
//!   GLM's scar is a template that lived only in the fp8 SOURCE and drifted for months once
//!   hand-ported (`docs/investigations/k3-first-checkpoint.md`, and the artifact-provenance
//!   scar); here the checkpoint states it twice, so the converter is where the two are compared.
//!
//! **The manifest keeps `text_config` intact**, like `convert_k3`'s: `QwenConfig` descends
//! through it and the document ROOT is the only level that names the architecture, so a
//! converter that helpfully flattened the nesting would produce an artifact the engine refuses
//! on its own manifest.
//!
//! Plan of record: `docs/investigations/qwen-flash-next-port.md` (stage S4).

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use rivoli_artifact::census::qwen::{Census, NgramHash, Role, ngram_hash};
use rivoli_artifact::format::{
    ArtifactDirs, Dtype, FormatMeta, SafeWriter, Safetensors, finish_artifact, require_aux,
};
use rivoli_artifact::quant::FP8_BLOCK;
use rivoli_artifact::qwen_config::{QwenConfig, QwenTextConfig};

/// The files copied beside the weights, checked against the source before the convert starts and
/// copied when it ends — ONE list, so the check and the copy cannot drift (`require_aux`'s own
/// doc carries the K3 scar that rule came from: an aux name the checkpoint did not ship refused
/// a 1.42 TiB run at its LAST step).
///
/// Each entry carries something no other file in the artifact does:
/// * `tokenizer.json` — the vocabulary. **Verified as the TYPE first**: `"model": {"type":
///   "BPE"}` with `tokenizer_class: Qwen2Tokenizer`, so the `tokenizers` crate is the loader and
///   no tiktoken path is needed. That check came before 172.76 GiB moved, which is the whole
///   lesson of the K3 first-checkpoint investigation.
/// * `tokenizer_config.json` — the special-token decoder AND the second home of the chat
///   template; [`confront_chat_template`] compares it with the file below.
/// * `generation_config.json` — `eos_token_id: [248046, 248044]`, read by `Tokenizer::load` from
///   this exact filename. A converter that exited 0 without it once left the engine ZERO stop
///   tokens behind a single `warn!`.
/// * `chat_template.jinja` — the template itself, 8,952 B. It ships in the FP8 repo, unlike
///   GLM's, and both of its homes are pinned equal here.
const AUX: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
];

/// The source index every converter in this tree selects shards by.
const INDEX: &str = "model.safetensors.index.json";

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
#[derive(Parser)]
#[command(
    name = "convert_qwen",
    about = "Qwen3.8-Flash-Next FP8 checkpoint → a rivoli artifact (fp8 experts copied \
             verbatim, BF16 scale grids widened, MTP and vision excluded by name)"
)]
struct Args {
    /// The checkpoint directory: config.json, model.safetensors.index.json, the *.safetensors
    /// shards, and the four tokenizer files copied into the artifact.
    src_dir: String,

    /// Artifact directory to write. Created if absent; an existing per-layer expert file or
    /// n-gram shard is REUSED rather than rewritten, so a killed run resumes on the same command
    /// line.
    out_dir: String,

    /// First layer to convert. Every layer is MoE in this model, so this is 0 by default.
    #[arg(long, value_name = "L", default_value_t = 0)]
    from: usize,

    /// One past the last layer to convert (default: the whole model). The range is recorded in
    /// the manifest's `qwen_source`, so a partial artifact can never claim to be whole — and
    /// only the shards holding these layers are opened.
    #[arg(long, value_name = "L")]
    to: Option<usize>,
}

/// The layer range this run covers, refused before a single tensor is read.
///
/// `convert_k3::bounded_range` is the precedent; what that one needs and this one does not is a
/// dense-prefix floor — MoE runs on EVERY layer here, so the floor is 0. Refused rather than
/// clamped, for the same reason: `--to 999` silently converting 48 layers would look like it did
/// what was asked.
fn bounded_range(
    cfg: &QwenTextConfig,
    from: usize,
    to: Option<usize>,
) -> Result<std::ops::Range<usize>> {
    let to = to.unwrap_or(cfg.n_layers);
    ensure!(
        from < to && to <= cfg.n_layers,
        "layer range [{from}, {to}) is not inside this model's [0, {}) — every layer is MoE \
         here, so there is no dense prefix to skip",
        cfg.n_layers
    );
    Ok(from..to)
}

/// **The chat template ships TWICE and the two must be the same template.**
///
/// `chat_template.jinja` is the file; `tokenizer_config.json`'s `chat_template` key is the same
/// text as a JSON string. The S0 finding is that they agree byte-for-byte — track D measured both
/// at 8,952 B with NO trailing newline on either side — and that agreement is the reason this port
/// can copy either one instead of hand-porting, which is exactly where GLM drifted to another
/// family's role framing for months.
/// So the agreement is asserted at convert time rather than recorded in prose: if a future
/// revision lets them diverge, the port must choose deliberately, and this refusal is what makes
/// it choose.
///
/// Compared with trailing newlines trimmed from BOTH sides and both raw lengths named in the
/// refusal, because "which of the two is longer" is the actionable half. The trim is not what makes
/// the pinned revision agree — it already does, exactly — it is there so that a re-export adding a
/// newline to one home reads as the formatting artefact it is rather than as a divergence.
fn confront_chat_template(src_dir: &str) -> Result<()> {
    let jinja_path = format!("{src_dir}/chat_template.jinja");
    let jinja =
        std::fs::read_to_string(&jinja_path).with_context(|| format!("read {jinja_path}"))?;
    let cfg_path = format!("{src_dir}/tokenizer_config.json");
    let doc: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&cfg_path).with_context(|| format!("read {cfg_path}"))?,
    )
    .with_context(|| format!("parse {cfg_path}"))?;
    let embedded = doc["chat_template"].as_str().with_context(|| {
        format!(
            "{cfg_path} has no string `chat_template` key. This checkpoint states the template \
             TWICE and the converter pins the two equal; a missing second home means the port \
             must decide which one is authoritative rather than copying either"
        )
    })?;
    let (a, b) = (
        jinja.trim_end_matches('\n'),
        embedded.trim_end_matches('\n'),
    );
    ensure!(
        a == b,
        "chat_template.jinja ({} B) and tokenizer_config.json's `chat_template` ({} chars) are \
         not the same template. They agree byte-for-byte in the pinned revision (8,952 B each), \
         and copying either one is only safe while they do — the GLM scar is a \
         hand-ported template that drifted to another family's role framing for months",
        jinja.len(),
        embedded.len()
    );
    Ok(())
}

/// Every tensor name the source index declares, plus the `metadata.total_size` it claims.
///
/// The index rather than the shard headers, and BEFORE any shard is opened: it is 17.4 MB against
/// 172.76 GiB, it lists every tensor including the ones in shards this run will never open, and
/// it is therefore the only place the census closure can be made exhaustive.
fn index_names(src_dir: &str) -> Result<(Vec<String>, u64)> {
    let path = format!("{src_dir}/{INDEX}");
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("read {path}"))?)
            .with_context(|| format!("parse {path}"))?;
    let map = doc["weight_map"]
        .as_object()
        .with_context(|| format!("{path} has no `weight_map` object"))?;
    let total = doc["metadata"]["total_size"]
        .as_u64()
        .with_context(|| format!("{path} declares no `metadata.total_size`"))?;
    Ok((map.keys().cloned().collect(), total))
}

/// One routed expert projection: the fp8 weight verbatim, its BF16 grid widened to F32.
///
/// **The grid is derived from THIS weight's shape, per projection.** `[out/128, in/128]` is [20,
/// 5] for `down_proj` and [5, 20] for `gate_proj`/`up_proj`, and a grid read at the wrong
/// orientation is a legal index into a legal array: every tile takes another tile's scale, over
/// 80.5 GB of expert weight, with no NaN, no length error and no dtype mismatch. The census
/// makes the same check family-wide before anything is read; this makes it per tensor, because
/// the census is a claim about the pinned revision and this is a claim about the bytes in hand.
///
/// The names are `<base>.weight` / `<base>.weight_scale_inv` — the source's own spelling, which
/// is also what `SafeWriter::add_fp8_pair` writes for every other model, so nothing downstream
/// has to learn a fifth convention. `copy_fp8` is not used because it requires an F32 scale in
/// the SOURCE and this checkpoint's is BF16.
fn add_fp8_bf16_scaled<'a>(
    w: &mut SafeWriter<'a>,
    src: &'a Safetensors,
    base: &str,
) -> Result<usize> {
    let wname = format!("{base}.weight");
    let sname = format!("{base}.weight_scale_inv");
    let (bytes, shape) = src.typed(&wname, Dtype::F8E4M3)?;
    ensure!(shape.len() == 2, "{wname}: shape {shape:?} is not 2-D");
    let grid = src.shape(&sname)?;
    let want: Vec<usize> = shape.iter().map(|d| d.div_ceil(FP8_BLOCK)).collect();
    ensure!(
        grid == want.as_slice(),
        "{sname} is {grid:?} but its weight {shape:?} implies a [out/{FP8_BLOCK}, \
         in/{FP8_BLOCK}] grid of {want:?}. A transposed grid gives every 128x128 tile a \
         different tile's scale, which no length, dtype or NaN check downstream can see"
    );
    // The weight BORROWS the source mmap (this is the path 112.5 GiB rides); only the grid
    // materializes, at 4 B per tile.
    w.add(wname, Dtype::F8E4M3, shape.to_vec(), bytes);
    w.add_widened(src, &sname)?;
    Ok(shape.iter().product::<usize>())
}

/// `manifest.json` = the source config VERBATIM (wrapper and all) + the `format` section +
/// `qwen_source` provenance.
///
/// Returns the manifest instead of writing it, for `convert_k3::build_manifest`'s reason: ending
/// this function with `finish_artifact(...)` made its tail token-identical to that file's and
/// jscpd refused it. `num_hidden_layers` is deliberately left alone — `layer_types`,
/// `ple_layer_ids` and `full_attention_interval` are all indexed by the REAL layer id, so
/// renumbering a partial artifact as a small MODEL would mis-key every one of them.
fn build_manifest(src_dir: &str, layers: &std::ops::Range<usize>) -> Result<serde_json::Value> {
    let mut manifest = FormatMeta::manifest_from_config(src_dir, FP8_BLOCK)?;
    manifest["qwen_source"] = serde_json::json!({
        "tool": "convert_qwen",
        "src": src_dir,
        "layers": [layers.start, layers.end],
    });
    Ok(manifest)
}

/// Step 1: one file per layer, holding that layer's 512 experts x 3 projections.
///
/// REUSED rather than rewritten when present, for `convert_k3`'s reason — a killed run resumes
/// on the same command line, and this is 2.5 GB per layer.
fn write_routed_layer(
    src: &Safetensors,
    cfg: &QwenTextConfig,
    layer: usize,
    out_dir: &str,
) -> Result<()> {
    let path = format!("{out_dir}/L{layer:02}.experts.safetensors");
    if std::fs::metadata(&path).is_ok() {
        eprintln!("convert_qwen: {path} exists — reused");
        return Ok(());
    }
    let mut w = SafeWriter::new();
    let mut weights = 0usize;
    for e in 0..cfg.num_experts {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let base = format!("model.language_model.layers.{layer}.mlp.experts.{e}.{proj}");
            weights += add_fp8_bf16_scaled(&mut w, src, &base)?;
        }
    }
    // A guard that can actually fire: the loop above is driven by `num_experts` from the config,
    // and the census has already confronted that count against the index — so a zero here means
    // the config and the census agreed on a count of nothing, which is the one way both checks
    // pass and the layer is empty.
    ensure!(
        weights > 0,
        "layer {layer} produced no expert weights at num_experts {}",
        cfg.num_experts
    );
    w.write(&path)?;
    eprintln!(
        "convert_qwen: wrote {path} — {} experts x 3 projections, {weights} fp8 weights",
        cfg.num_experts
    );
    Ok(())
}

/// Step 2: the n-gram table, as the 128 **ordered row ranges** it is on disk.
///
/// Not a matrix and not reassembled: a decode gathers `ngram_heads` rows of `ngram_head_dim` fp8
/// bytes per token, so what the artifact needs is contiguous rows and a shard index it can
/// divide into. Each shard is copied verbatim and its rows are asserted to tile
/// `[0, padded_rows)` exactly — which is the check that turns a shape census into a claim about
/// a row SPACE. `split_ngram_parts` is a checkpoint-side convention with no counterpart in the
/// reference module (OPEN-2 in `qwen-architecture.md`), so the tiling is the only thing that
/// says the convention was read correctly.
fn write_ngram_shards(
    src: &Safetensors,
    cfg: &QwenTextConfig,
    hash: &NgramHash,
    host: usize,
    out_dir: &str,
) -> Result<()> {
    let head_dim = cfg.ngram_head_dim()?;
    let mut row = 0u64;
    for s in 0..cfg.split_ngram_parts {
        let name = format!(
            "model.language_model.layers.{host}.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
        );
        let (bytes, shape) = src.typed(&name, Dtype::F8E4M3)?;
        ensure!(
            shape
                == [
                    usize::try_from(hash.rows_per_shard).unwrap_or(usize::MAX),
                    head_dim
                ],
            "{name} is {shape:?}, but {} rows per shard x {head_dim} dims is what \
             split_ngram_parts {} and the derived {} padded rows imply",
            hash.rows_per_shard,
            cfg.split_ngram_parts,
            hash.padded_rows
        );
        let path = format!("{out_dir}/ngram.{s:03}.safetensors");
        if std::fs::metadata(&path).is_err() {
            let mut w = SafeWriter::new();
            // Named by its GLOBAL first row rather than by its shard number, because the row is
            // what a gather divides by and the shard number is an artefact of how the publisher
            // split the upload.
            //
            // **UNGATED ASSUMPTION, named here and in the plan doc's OWED list: the shard INDEX is
            // row ORDER.** The `ensure!` below gates that the 128 shards TILE `[0, padded_rows)` —
            // the widths sum — and says nothing about which rows are in which file. If
            // `shard_{s}` does not hold `[s * rows_per_shard, (s+1) * rows_per_shard)`, every row
            // a decode gathers is the wrong row with every shape, count, dtype and byte total
            // intact, no crash and no refusal. It cannot be gated here: the real bytes are not on
            // this box, and a check over a synthesized source would only assert the convention it
            // was built from. What settles it is one known token's 16 rows gathered from the REAL
            // shards through `hash.offsets` against the reference stack's PLE embedding output.
            w.add(
                format!("ngram.rows.{row}"),
                Dtype::F8E4M3,
                shape.to_vec(),
                bytes,
            );
            w.write(&path)?;
        }
        row += hash.rows_per_shard;
    }
    ensure!(
        row == hash.padded_rows,
        "the {} shards cover {row} rows, not the {} the derived head vocabularies imply — the \
         shards are ordered row RANGES and a gap or an overlap silently reads the wrong \
         embedding row",
        cfg.split_ngram_parts,
        hash.padded_rows
    );
    eprintln!(
        "convert_qwen: wrote {} n-gram shards — {} rows x {head_dim} fp8 bytes, tiled exactly",
        cfg.split_ngram_parts, hash.padded_rows
    );
    Ok(())
}

/// **The three I64 hash buffers, byte-compared against the re-derivation rather than trusted.**
///
/// They are the only tensor VALUES this converter checks, and the reason is that nothing else
/// can: a wrong seed, a wrong `ple_layer_index` or a wrong `global_head_idx` produces 16
/// plausible primes and 3 plausible odd multipliers that hash every token to different rows,
/// with every shape, count and byte total unchanged. `ngram_hash`'s own doc carries the near-miss
/// that makes the ORDER load-bearing: the wrong SplitMix64 form yields a sequence containing two
/// of the three correct multipliers in the wrong slots.
fn confront_hash_buffers(src: &Safetensors, hash: &NgramHash, host: usize) -> Result<()> {
    let stem = format!("model.language_model.layers.{host}.ple.ple_embedding");
    for (suffix, want) in [
        ("ngram_heads_vocab_sizes", &hash.vocab_sizes),
        ("ngram_heads_offsets", &hash.offsets),
        ("layer_multipliers", &hash.multipliers),
    ] {
        let name = format!("{stem}.{suffix}");
        let (bytes, shape) = src.typed(&name, Dtype::I64)?;
        let got: Vec<u64> = bytes
            .chunks_exact(8)
            .filter_map(|c| <[u8; 8]>::try_from(c).ok())
            .map(u64::from_le_bytes)
            .collect();
        ensure!(
            shape == [want.len()] && &got == want,
            "{name} is {shape:?} {got:?}, but the derivation from seed and ple_layer_index gives \
             {want:?}. This checkpoint's hash parameters are re-derivable \
             (`qwen-architecture.md` §5) and a mismatch means the seed, the PLE layer index or \
             the head-vocabulary walk is not the one the weights were trained with — which \
             changes no shape and every gathered row"
        );
    }
    Ok(())
}

/// Step 3: the resident set — everything in scope that is neither a routed expert nor an n-gram
/// shard, copied VERBATIM as BF16 (plus the I64 hash buffers and the one BF16 per-tensor scale).
///
/// ALWAYS rewritten, never reused, for `convert_v4::write_resident`'s reason: it is one file
/// spanning every layer the run covered, and reusing it after a different `--from/--to` leaves an
/// artifact whose expert files and resident set cover different ranges.
///
/// **Copied, not widened.** The norms stay BF16 exactly as they are on disk — `convert_v4`
/// widens because its loader reads norms as f32, and this port's loader does not exist yet, so
/// the decision belongs to whoever writes it. Recorded here rather than left to be discovered: a
/// BF16 tensor read as f32 is not a length error, it is half the rows at wrong magnitudes.
fn write_resident(
    src: &Safetensors,
    census: &Census,
    in_scope: &impl Fn(&str) -> bool,
    out_dir: &str,
) -> Result<()> {
    let mut w = SafeWriter::new();
    let (mut kept, mut skipped) = (0usize, 0usize);
    for name in src.names() {
        if !in_scope(name) {
            continue; // a tensor that rode along in an opened shard — see `resident_layers`
        }
        let (bytes, dtype, shape) = src.raw(name)?;
        let f = census.confront_tensor(name, dtype, shape)?;
        // **The excluded arm is a REFUSAL in the match, and it is deliberately not `in_scope`'s
        // question re-asked.** `in_scope` consults the same census, so a broken role predicate
        // would open those shards AND stop filtering them — `convert_k3`'s measured lesson, where
        // a structurally-zero counter printed "0 vision skipped" however broken the filter was.
        // Asking it here, in the arm, is what makes it unanswerable any other way: the question is
        // put against the role of the tensor actually in hand at the point of writing, and there is
        // no separate `ensure!` above whose deletion would leave an `unreachable!` to abort a
        // converter that must refuse.
        match f.role {
            Role::RoutedWeight | Role::RoutedScale | Role::NgramShard => skipped += 1,
            Role::Resident | Role::HashParam | Role::NgramScale => {
                w.add(name, dtype, shape.to_vec(), bytes);
                kept += 1;
            }
            Role::ExcludedMtp | Role::ExcludedVision => bail!(
                "{name} reached the resident set with role {} — this converter builds the TEXT \
                 arm only, and the MTP draft layer and the vision tower must never enter the \
                 artifact",
                f.role.label()
            ),
        }
    }
    ensure!(kept > 0, "the resident set is empty — nothing was in scope");
    let path = format!("{out_dir}/resident.safetensors");
    w.write(&path)?;
    eprintln!(
        "convert_qwen: wrote {path} — {kept} tensors ({skipped} routed/n-gram written elsewhere)"
    );
    Ok(())
}

fn main() -> Result<()> {
    // DESTRUCTURED, following `convert_v4`, `add_indexer` and `fp8_to_i4` rather than inventing a
    // shape: that file's own comment records `} <blank> fn main() -> Result<()> { let args =
    // Args::parse();` as a jscpd clone of `convert_glimmer`'s opening, and this converter is the
    // fifth to arrive at it.
    let Args {
        src_dir,
        out_dir,
        from,
        to,
    } = Args::parse();
    // ONE `ArtifactDirs`, borrowed for the refusal below and moved into `finish_artifact` at the
    // end — where the three converters before this one each construct the literal twice (or, in
    // `convert_v4`, route both sites through a free helper). Same pairing argument that type's own
    // doc makes; one construction is simply the stronger form of it, and it is also what keeps
    // this `main` off jscpd's report.
    let dirs = ArtifactDirs {
        src: &src_dir,
        out: &out_dir,
    };
    // Before anything is read: out_dir == src_dir is a SIGBUS mid-write, not an error.
    SafeWriter::refuse_writing_into_source(&dirs)?;
    // And before the config: a missing aux file is otherwise reported by `finish_artifact` at
    // the END, hours and 172.76 GiB later.
    require_aux(&src_dir, AUX)?;
    confront_chat_template(&src_dir)?;
    let cfg = QwenConfig::load(&src_dir)?;
    let t = &cfg.text;
    let census = Census::load()?;

    // **The exhaustive census closure, on the INDEX, before a shard is opened.** Every name is
    // classified, every index is bounded against the config, every pattern's count is
    // confronted, and the `weight_scale_inv` grid orientations are checked family-wide. After
    // this, `in_scope` below can treat an unclassifiable name as out of scope, because there is
    // none.
    let (names, declared_bytes) = index_names(&src_dir)?;
    let borrowed: Vec<&str> = names.iter().map(String::as_str).collect();
    let summary = census.confront_index(t, &borrowed)?;
    let (_, total_bytes) = summary.total();
    ensure!(
        total_bytes == declared_bytes,
        "the census accounts for {total_bytes} B but {INDEX} declares metadata.total_size \
         {declared_bytes} — every family count matched, so this is a byte-per-tensor \
         disagreement rather than a missing family"
    );
    eprintln!("convert_qwen: {}", summary.line());

    let range = bounded_range(t, from, to)?;
    let host = t.ple_host_layer()?;
    let hash = ngram_hash(t, 0)?;
    let ngram_in_range = range.contains(&host);
    std::fs::create_dir_all(&out_dir)?;

    // ONE list of layers, driving which shards get opened, what the resident writer emits, and
    // which expert files are written — the pairing `convert_k3`'s comment insists on. There is
    // no dense prefix to add: every layer is MoE.
    let wanted: Vec<String> = range
        .clone()
        .map(|l| format!("model.language_model.layers.{l}."))
        .collect();
    let in_scope = |n: &str| {
        // An unclassifiable name is impossible here — `confront_index` refused the whole run
        // above — so `false` is the safe reading of an `Err` rather than a silent drop.
        census.family_of(n).is_ok_and(|f| {
            !f.role.is_excluded()
                && (!n.contains(".layers.") || wanted.iter().any(|p| n.starts_with(p.as_str())))
        })
    };
    let src = Safetensors::open_indexed(&src_dir, in_scope)?;
    eprintln!(
        "convert_qwen: hidden={} layers {}..{} (of {}, {} QSA / {} GDN) experts={} top_k={} \
         ple host={host} ngram {} rows in {} shards",
        t.hidden,
        range.start,
        range.end,
        t.n_layers,
        t.sparse_attention_layers(),
        t.n_layers - t.sparse_attention_layers(),
        t.num_experts,
        t.top_k,
        hash.padded_rows,
        t.split_ngram_parts
    );

    if ngram_in_range {
        confront_hash_buffers(&src, &hash, host)?;
        write_ngram_shards(&src, t, &hash, host, &out_dir)?;
    } else {
        eprintln!(
            "convert_qwen: PLE host layer {host} is outside [{}, {}) — no n-gram shards written, \
             and the manifest's qwen_source records the range",
            range.start, range.end
        );
    }
    for l in range.clone() {
        write_routed_layer(&src, t, l, &out_dir)?;
    }
    write_resident(&src, &census, &in_scope, &out_dir)?;
    finish_artifact(
        "convert_qwen",
        dirs,
        &build_manifest(&src_dir, &range)?,
        AUX,
    )?;
    eprintln!("convert_qwen: done — {src_dir} → {out_dir}");
    Ok(())
}
