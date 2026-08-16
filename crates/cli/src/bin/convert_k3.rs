//! `convert_k3` — the Kimi-K3 checkpoint → a rivoli `.f4` artifact. Ported from
//! `k3:src/bin/convert_k3.rs`, comments travelling with their code.
//!
//! **The routed experts are a REPACK, not a quantization.** K3 ships its 896 experts per
//! layer already at 4 bits — OCP MX e2m1 nibbles with e8m0 group scales, `group_size: 32` —
//! which is byte-for-byte what rivoli's `.f4` container holds. Verified against the
//! checkpoint's own shard headers (`crates/artifact/tests/k3_names.rs`): the source is
//! `[o_dim, i_dim/2]` packed and `[o_dim, i_dim/32]` scales, exactly `f4_row_bytes` and
//! `f4_groups`, so each projection is two `copy_from_slice`s. Nothing is fit, nothing is
//! re-rounded, and no error is introduced — the same argument `convert_v4` makes, and the
//! reason re-quantizing a 4-bit source into int3-vq is not on the table (PPL 73.43,
//! `old:docs/investigations/int4-scales.md`).
//!
//! **What differs from `convert_v4`, and it is all naming and widths:**
//!
//! * Experts are entered at the **3584 latent** (`routed_expert_hidden_size`), not at
//!   `hidden_size` 7168. That is `F4Expert::expert_in`, and binding `hidden` there would
//!   produce a self-consistent file with every stride 2x wrong.
//! * Tensors are `<proj>.weight_packed` / `<proj>.weight_scale`, both `U8` — `F4_NAMING_K3`.
//! * Every text-side name carries a `language_model.model.` prefix that no document
//!   mentioned before the shipped index was read (2026-08-10).
//! * There is **no shared block** in `.f4` and none is wanted: K3's shared expert is one
//!   fused BF16 MLP at full width (`[7168, 6144]`), trunk-side, resident.
//! * Layer **0 is dense** (`first_k_dense_replace: 1`) and has no experts at all, so the
//!   routed set is layers 1..93.
//!
//! **The manifest keeps `text_config` intact.** `K3Config` descends through it and the
//! wrapper is the only level that names the architecture, so a converter that helpfully
//! flattened the nesting would produce an artifact the engine refuses on its own manifest.

use anyhow::{Result, ensure};
use clap::Parser;
use rivoli_artifact::format::{
    ArtifactDirs, F4_NAMING_K3, FormatMeta, RoutedRepack, SafeWriter, Safetensors, f4_source,
    finish_artifact,
};
use rivoli_artifact::k3_config::{K3Config, K3TextConfig};
use rivoli_artifact::quant::{FP8_BLOCK, K3_TEXT_PREFIX, k3_expert_base};
use std::ops::Range;

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
#[derive(Parser)]
#[command(
    name = "convert_k3",
    about = "Kimi-K3 checkpoint → the rivoli .f4 artifact (MXFP4 experts repacked, BF16 \
             trunk copied verbatim)"
)]
struct Args {
    /// The Kimi-K3 checkpoint directory: config.json, model.safetensors.index.json, the
    /// `*.safetensors` shards, and the tokenizer files copied into the artifact.
    src_dir: String,

    /// Artifact directory to write. Created if absent; an existing `L{ll}.f4` is REUSED
    /// rather than rewritten, so a killed run resumes on the same command line.
    out_dir: String,

    /// First MoE layer to convert. The dense prefix has no experts, so this is 1 by
    /// default and `--from 0` is refused rather than silently producing an empty `L00.f4`.
    #[arg(long, value_name = "L", default_value_t = 1)]
    from: usize,

    /// One past the last layer to convert (default: the whole model). The range is
    /// recorded in the manifest's `f4_source`, so a partial artifact can never claim to be
    /// whole — and only the shards holding these layers are opened.
    #[arg(long, value_name = "L")]
    to: Option<usize>,

    /// After writing each layer, read the FILE back and byte-compare every expert against
    /// the source tensors. The repack is a copy, so the only correct answer is zero
    /// differing bytes.
    #[arg(long)]
    verify: bool,

    /// Verify and write NOTHING. Implies `--verify`. Use this whenever the target is an
    /// artifact you want to keep — `--verify` alone still rewrites the resident set and
    /// the manifest for `--from/--to`, which TRUNCATES a whole artifact to the range.
    #[arg(long)]
    verify_only: bool,
}

/// Tensors that belong to the routed expert set, i.e. the ones `.f4` carries and the
/// resident safetensors must NOT duplicate.
///
/// A substring test on `.experts.` would also catch `shared_experts`, which is exactly
/// backwards: the shared MLP is trunk-side and MUST stay resident. Hence the anchored
/// `.experts.` under `block_sparse_moe`.
fn is_routed(name: &str) -> bool {
    name.contains("block_sparse_moe.experts.")
}

/// Is this tensor part of the multimodal front end?
///
/// Skipped explicitly rather than by omission: `vision_tower` (27 blocks) and
/// `mm_projector` are SIBLINGS of `language_model` in the name tree, so a filter phrased as
/// "not under language_model" would drop nothing, and copying everything would carry a
/// vision tower this engine has no path for. `quantization_config.ignore` names both,
/// which is the one part of that block worth believing.
///
/// **No byte figure here on purpose.** An earlier draft in the k3 tree said "~600 MB"; the
/// vendored `tensor-families.tsv` records `?` for every vision shape, so there is nothing
/// in this repo to derive it from and an estimate stated as a measurement is the defect
/// this port keeps catching.
fn is_vision(name: &str) -> bool {
    name.starts_with("vision_tower.") || name.starts_with("mm_projector.")
}

/// The layer range this run covers, refused before a single tensor is read.
/// `convert_v4::refuse_before_writing` is the precedent; here the guard that converter does
/// not need is the interesting one — the dense prefix has no experts, so the floor of the
/// range is `first_k_dense_replace`, not 0 — and it is what shapes this as a function of
/// the TEXT config rather than of the two directories (the SafeWriter guard and the config
/// load sit in `main`, before this can run).
fn bounded_range(k3: &K3TextConfig, from: usize, to: Option<usize>) -> Result<Range<usize>> {
    let to = to.unwrap_or(k3.n_layers);
    // `first_k_dense_replace` is the floor, not 0. Layer 0 ships `mlp.{gate,up,down}_proj`
    // and no experts, so `--from 0` would ask `F4Expert` for tensors that do not exist — a
    // confusing "tensor not found" instead of the real answer, which is that the dense
    // layer has none.
    ensure!(
        from >= k3.first_k_dense_replace,
        "layer {from} is inside the dense prefix (first_k_dense_replace = {}) and has no \
         routed experts — the MoE layers are {}..{}",
        k3.first_k_dense_replace,
        k3.first_k_dense_replace,
        k3.n_layers
    );
    // Refused, not clamped: `--to 999` silently converting 93 layers would look like it
    // did what was asked. The floor named in the message is the DENSE floor, because that
    // is the bound a K3 range can actually start at.
    ensure!(
        from < to && to <= k3.n_layers,
        "layer range [{from}, {to}) is not inside the MoE layers [{}, {})",
        k3.first_k_dense_replace,
        k3.n_layers
    );
    Ok(from..to)
}

/// Confront the config with the tensors ONCE per resident layer, before copying any of
/// them — `convert_v4::confront_config_with_tensors`'s argument, sharpened by this
/// converter's shape: **everything here is copied verbatim and the manifest carries the
/// config verbatim**, so without this the latent, the expert count and the fused shared
/// width would reach the engine's launches having never been compared to the weights they
/// describe. The k3 reference had no such check; the expert side was confronted by
/// `F4Expert::spans` and the trunk by nothing.
///
/// The MoE trunk only. The attention families' tensors are passed through uninterpreted —
/// their shapes are the engine arm's to confront when it launches against them — but the
/// MoE trunk is the set THIS tool splits an artifact around, so a disagreement here is
/// this tool's to refuse.
fn confront_moe_trunk(src: &Safetensors, k3: &K3TextConfig, l: usize) -> Result<()> {
    let moe = format!("{K3_TEXT_PREFIX}layers.{l}.block_sparse_moe");
    let (h, latent) = (k3.hidden, k3.expert_in);
    // The fused shared MLP: the config's COUNT of shared experts times their width, in ONE
    // tensor set — `num_shared_experts` is not a count of tensors (k3 G0 item 4).
    let fused = k3.n_shared * k3.moe_inter;
    for (name, want) in [
        // The router scores on FULL width, and its bias is per expert, selection-only.
        (format!("{moe}.gate.weight"), vec![k3.n_experts, h]),
        (
            format!("{moe}.gate.e_score_correction_bias"),
            vec![k3.n_experts],
        ),
        // The latent sandwich: down INTO the latent, norm ON it, up OUT of it.
        (
            format!("{moe}.routed_expert_down_proj.weight"),
            vec![latent, h],
        ),
        (format!("{moe}.routed_expert_norm.weight"), vec![latent]),
        (
            format!("{moe}.routed_expert_up_proj.weight"),
            vec![h, latent],
        ),
        (
            format!("{moe}.shared_experts.gate_proj.weight"),
            vec![fused, h],
        ),
        (
            format!("{moe}.shared_experts.up_proj.weight"),
            vec![fused, h],
        ),
        (
            format!("{moe}.shared_experts.down_proj.weight"),
            vec![h, fused],
        ),
    ] {
        confront(src, &name, &want)?;
    }
    Ok(())
}

/// One shape, confronted. A named function rather than the loop-body `ensure!` because the
/// inline form was token-identical to `convert_v4`'s confrontation tail and jscpd said so;
/// with one caller each, the two converters' loops now share only their idea.
fn confront(src: &Safetensors, name: &str, want: &[usize]) -> Result<()> {
    let got = src.shape(name)?;
    ensure!(
        got == want,
        "{name} is {got:?} where the config implies {want:?} — this trunk is copied \
         VERBATIM, so no later step would ever compare the two"
    );
    Ok(())
}

/// Step 2: the resident set — everything in scope that is neither routed nor vision,
/// copied VERBATIM. ALWAYS rewritten, never reused, for `convert_v4::write_resident`'s
/// reason: it is one file spanning every layer the run covered, and reusing it after a
/// different `--from/--to` leaves an artifact whose `.f4` set and resident set cover
/// different ranges.
///
/// BF16 and F32 tensors are copied rather than converted, so `SafeWriter`'s `Cow` borrows
/// them straight from the source mmap and host RAM peak stays the sum of the CONVERTED
/// tensors — which here is zero.
///
/// **The norms are copied VERBATIM as BF16, and `convert_v4` does not do that** — it
/// widens them to f32 because its loader reads norms as f32. K3's trunk is BF16 on disk
/// and this converter is a passthrough, so the widening question belongs to whoever writes
/// the K3 loader; it is recorded here rather than left for them to discover, because a
/// BF16 tensor read as f32 is not a length error, it is half the rows at wrong magnitudes.
fn write_resident(
    src: &Safetensors,
    k3: &K3TextConfig,
    in_scope: &impl Fn(&str) -> bool,
    moe_layers: usize,
    out_dir: &str,
) -> Result<()> {
    let mut w = SafeWriter::new();
    let (mut kept, mut skipped_routed, mut skipped_vision) = (0usize, 0usize, 0usize);
    for name in src.names() {
        if is_vision(name) {
            skipped_vision += 1;
            continue;
        }
        if is_routed(name) {
            skipped_routed += 1;
            continue;
        }
        if !in_scope(name) {
            continue; // a tensor that rode along in an opened shard — see `resident_layers`
        }
        let (bytes, dtype, shape) = src.raw(name)?;
        w.add(name, dtype, shape.to_vec(), bytes);
        kept += 1;
    }
    // **A guard that can actually fire**, replacing the k3 tree's `kept > 0`-with-a-wrong-
    // message (review 2026-08-11 there): if `is_routed`'s literal ever stops matching, 896
    // experts x 6 tensors x each converted layer land in `resident.safetensors` — 15.72 GB
    // per layer of duplicated weights, an artifact that still loads. `>=` rather than `==`
    // because an opened shard may hold more than one layer's experts, and this run only
    // accounts for its own.
    let want_skipped = moe_layers * k3.n_experts * 6;
    ensure!(
        skipped_routed >= want_skipped,
        "only {skipped_routed} routed tensors were skipped, expected at least \
         {want_skipped} ({moe_layers} layers x {} experts x 6) — `is_routed` is no longer \
         matching the checkpoint's expert names, and those weights are about to be \
         duplicated into the resident set",
        k3.n_experts
    );
    ensure!(kept > 0, "the resident set is empty — nothing was in scope");
    let rpath = format!("{out_dir}/resident.safetensors");
    w.write(&rpath)?;
    eprintln!(
        "convert_k3: wrote {rpath} — {kept} tensors ({skipped_routed} routed, \
         {skipped_vision} vision skipped)"
    );
    Ok(())
}

/// Step 3: `manifest.json` = the source config VERBATIM (wrapper and all — see the module
/// header) + the `format` section + `f4_source` provenance.
fn write_manifest(src_dir: &str, out_dir: &str, layers: std::ops::Range<usize>) -> Result<()> {
    // `manifest_from_config` re-reads the source `config.json` rather than re-serializing
    // the parsed `K3Config`, so the nested `text_config` — including every key the schema
    // deliberately does not bind — survives byte-for-byte in spirit and the artifact
    // re-parses exactly as the source did.
    let manifest = {
        let mut m = FormatMeta::manifest_from_config(src_dir, FP8_BLOCK)?;
        // `layers` is the range this artifact HOLDS; `num_hidden_layers` is deliberately
        // left alone — every per-layer structure in a K3 config (`linear_attn_config`'s
        // two one-based arrays, `first_k_dense_replace`) is indexed by the REAL layer id,
        // so renumbering a partial artifact as a small MODEL would mis-key all of them.
        m["f4_source"] = f4_source("convert_k3", src_dir, layers);
        m
    };
    // `ArtifactDirs` spelled inline and src-first, where `convert_v4` routes both call
    // sites through an `artifact_dirs` helper: with only two sites and no `&Args` to
    // shield against, the helper would be one more 45-token twin of that file's.
    finish_artifact(
        "convert_k3",
        ArtifactDirs {
            src: src_dir,
            out: out_dir,
        },
        &manifest,
        // No `chat_template.jinja` and no `generation_config.json`: K3's
        // `tokenizer_config.json` has no `chat_template` at all — there is no chat
        // encoding for this model in any tree, and the engine arm refuses `--port` rather
        // than inventing one.
        &["tokenizer.json", "tokenizer_config.json"],
    )
}

fn main() -> Result<()> {
    // NOT destructured, unlike `convert_v4`'s — its two reasons cut the other way here.
    // The hazard it names is a REBOUND `args.verify` meaning something the user did not
    // type; binding the resolved intent to a fresh name (and never reading `args.verify`
    // again) avoids that without the six-line pattern that was itself the clone jscpd
    // reported between the first drafts of these two `main`s.
    let args = Args::parse();
    // `--verify-only` implies `--verify` — one intent, resolved once, exactly as `--help`
    // says.
    let verify = args.verify || args.verify_only;
    let (src_dir, out_dir) = (args.src_dir, args.out_dir);
    // Not in the k3 reference, and added deliberately (the hazard and its argument live
    // with `SafeWriter`): out_dir == src_dir is a SIGBUS mid-write, not an error.
    SafeWriter::refuse_writing_into_source(&ArtifactDirs {
        src: &src_dir,
        out: &out_dir,
    })?;
    let cfg = K3Config::load(&src_dir)?;
    let k3 = &cfg.text;
    let range = bounded_range(k3, args.from, args.to)?;
    // **The latent, not `hidden`.** The single most consequential line in this file.
    let (expert_in, moe_inter, ne) = (k3.expert_in, k3.moe_inter, k3.n_experts);
    std::fs::create_dir_all(&out_dir)?;

    let layers: Vec<usize> = range.clone().collect();
    // **ONE list of layers, driving which shards get opened, what the resident writer
    // emits, AND what the confrontation walks** — the pairing `convert_v4`'s `MODEL_LEVEL`
    // comment insists on. The k3 tree broke it from the other end until review 2026-08-11
    // there: its resident pass looped over every tensor in every OPENED shard, so the
    // resident set was a function of the checkpoint's shard boundaries rather than of the
    // request (layer 0's dense `mlp.*` survived only because `embed_tokens` happened to
    // share shard 1 with them, and a partial run silently over-collected).
    //
    // The dense prefix is ALWAYS included, and that is not the same as "the requested
    // range": layer 0 carries no experts, so it is never in `from..to`; it is also trunk
    // the model cannot decode without. `convert_v4` has no dense layers and so never had
    // to say this.
    let resident_layers: Vec<usize> = (0..k3.first_k_dense_replace).chain(range.clone()).collect();
    let wanted: Vec<String> = resident_layers
        .iter()
        .map(|l| format!("{K3_TEXT_PREFIX}layers.{l}."))
        .collect();
    // A tensor this run is responsible for: model-level (no `layers.` at all —
    // `embed_tokens`, `norm`, `lm_head`, `output_attn_res_*`, verified against all 60
    // families in `tensor-families.tsv`) or belonging to one of `resident_layers`.
    let in_scope = |n: &str| {
        !is_vision(n)
            && (!n.contains("layers.") || wanted.iter().any(|p| n.starts_with(p.as_str())))
    };
    // Only the shards holding what we are about to read. A checkpoint still downloading
    // has truncated shards, and opening one would fail a run over layers we are not
    // converting. K3 is 1.42 TiB across 96 shards, so this is not a nicety.
    let src = Safetensors::open_indexed(&src_dir, in_scope)?;
    eprintln!(
        "convert_k3: hidden={} latent={expert_in} moe_inter={moe_inter} experts={ne} \
         layers {}..{} (of {}, dense prefix {})",
        k3.hidden, range.start, range.end, k3.n_layers, k3.first_k_dense_replace
    );

    // The confrontation runs BEFORE the repack loop, so a config/tensor disagreement
    // refuses with nothing written — including the `.f4` files, which `convert_v4` (whose
    // confrontation sits inside the resident walk) would already have produced.
    for &l in &layers {
        confront_moe_trunk(&src, k3, l)?;
    }

    // 1. Routed experts → one `.f4` per MoE layer. Pure repack, and `RoutedRepack` is the
    // shared loop — the ONLY thing this converter contributes here is `k3_expert_base` and
    // the latent.
    let repack = RoutedRepack {
        tool: "convert_k3",
        out_dir: &out_dir,
        src: &src,
        naming: &F4_NAMING_K3,
        expert_in,
        moe_inter,
        n_experts: ne,
        verify,
        write: !args.verify_only,
    };
    for &l in &layers {
        repack.layer(l, |e| k3_expert_base(l, e))?;
    }

    // A verification pass STOPS HERE — `convert_v4::main` records the near-miss (a narrow
    // `--verify` aimed at a whole artifact truncates its resident set and manifest to the
    // range, orphaning every other `.f4`).
    if args.verify_only {
        eprintln!("convert_k3: verify-only, resident set and manifest untouched");
        return Ok(());
    }

    write_resident(&src, k3, &in_scope, layers.len(), &out_dir)?;
    write_manifest(&src_dir, &out_dir, range)?;
    eprintln!("convert_k3: done — {src_dir} → {out_dir}");
    Ok(())
}
