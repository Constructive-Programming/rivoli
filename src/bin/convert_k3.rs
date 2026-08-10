//! Kimi-K3 checkpoint → rivoli artifact. **The routed experts are a REPACK, not a quantization.**
//!
//! K3 ships its 896 experts per layer already at 4 bits — OCP MX e2m1 nibbles with e8m0 group
//! scales, `group_size: 32` — which is byte-for-byte what rivoli's `.f4` container holds. Verified
//! against the checkpoint's own shard headers (`tests/k3_names.rs`): the source is
//! `[o_dim, i_dim/2]` packed and `[o_dim, i_dim/32]` scales, exactly `f4_row_bytes` and
//! `f4_groups`, so each projection is two `copy_from_slice`s. Nothing is fit, nothing is
//! re-rounded, and no error is introduced — the same argument `convert_v4` makes, and the reason
//! re-quantizing a 4-bit source into int3-vq is not on the table (PPL 73.43,
//! `docs/investigations/int4-scales.md`).
//!
//! **What differs from `convert_v4`, and it is all naming and widths:**
//!
//! * Experts are entered at the **3584 latent** (`routed_expert_hidden_size`), not at
//!   `hidden_size` 7168. That is `F4Expert::expert_in`, and binding `hidden` there would produce
//!   a self-consistent file with every stride 2x wrong — plan §2.
//! * Tensors are `<proj>.weight_packed` / `<proj>.weight_scale`, both `U8` — `F4_NAMING_K3`.
//! * Every text-side name carries a `language_model.model.` prefix that no document in this repo
//!   mentioned before 2026-08-10.
//! * There is **no shared block** in `.f4` and none is wanted: K3's shared expert is one fused
//!   BF16 MLP at full width (`[7168, 6144]`), trunk-side, resident.
//! * Layer **0 is dense** (`first_k_dense_replace: 1`) and has no experts at all, so the routed
//!   set is layers 1..93.
//!
//! **The manifest keeps `text_config` intact.** `K3Config` descends through it and the wrapper is
//! the only level that names the architecture, so a converter that helpfully flattened the nesting
//! would produce an artifact the engine refuses on its own manifest.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli::artifact::format::{
    F4_NAMING_K3, FormatMeta, RoutedRepack, SafeWriter, Safetensors, f4_source, finish_artifact,
};
use rivoli::artifact::model::{K3Config, load_config};
use rivoli::artifact::quant::{FP8_BLOCK, K3_TEXT_PREFIX, k3_expert_base};

#[derive(Parser)]
#[command(about = "Repack a Kimi-K3 checkpoint into a rivoli artifact")]
struct Args {
    src_dir: String,
    out_dir: String,
    /// First MoE layer to convert. The dense prefix has no experts, so this is 1 by default and
    /// `--from 0` is refused rather than silently producing an empty `L00.f4`.
    #[arg(long, default_value_t = 1)]
    from: usize,
    /// One past the last layer; defaults to `num_hidden_layers`.
    #[arg(long)]
    to: Option<usize>,
    /// Re-read each `.f4` and compare it against the source spans.
    #[arg(long)]
    verify: bool,
    /// Verify without writing anything new. Implies `--verify`.
    #[arg(long)]
    verify_only: bool,
}

/// Tensors that belong to the routed expert set, i.e. the ones `.f4` carries and the resident
/// safetensors must NOT duplicate.
///
/// A substring test on `.experts.` would also catch `shared_experts`, which is exactly backwards:
/// the shared MLP is trunk-side and MUST stay resident. Hence the anchored `.experts.` under
/// `block_sparse_moe`.
fn is_routed(name: &str) -> bool {
    name.contains("block_sparse_moe.experts.")
}

/// Is this tensor part of the multimodal front end?
///
/// Skipped explicitly rather than by omission: `vision_tower` (27 blocks) and `mm_projector` are
/// SIBLINGS of `language_model` in the name tree, so a filter phrased as "not under
/// language_model" would drop nothing, and copying everything would carry a vision tower this
/// engine has no path for. `quantization_config.ignore` names both `vision_tower` and
/// `mm_projector`, which is the one part of that block worth believing.
///
/// **No byte figure here on purpose.** An earlier draft said "~600 MB"; the vendored
/// `tensor-families.tsv` records `?` for every vision shape (none of the three shard headers
/// fetched covered them), so there is nothing in this repo to derive it from and an estimate
/// stated as a measurement is the defect this port keeps catching.
fn is_vision(name: &str) -> bool {
    name.starts_with("vision_tower.") || name.starts_with("mm_projector.")
}

fn main() -> Result<()> {
    let args = Args::parse();
    // `--verify-only` implies `--verify`; a local rather than a rebuilt `Args` so there is one
    // answer to "did the user ask for verification" and no field that can disagree with it.
    let verify = args.verify || args.verify_only;
    let cfg: K3Config = load_config(&args.src_dir)?;
    let k3 = &cfg.text;
    let to = args.to.unwrap_or(k3.n_layers);
    // `first_k_dense_replace` is the floor, not 0. Layer 0 ships `mlp.{gate,up,down}_proj` and no
    // experts, so `--from 0` would ask `F4Expert` for tensors that do not exist — a confusing
    // "tensor not found" instead of the real answer, which is that the dense layer has none.
    ensure!(
        args.from >= k3.first_k_dense_replace,
        "layer {} is inside the dense prefix (first_k_dense_replace = {}) and has no routed \
         experts — the MoE layers are {}..{}",
        args.from,
        k3.first_k_dense_replace,
        k3.first_k_dense_replace,
        k3.n_layers
    );
    ensure!(
        args.from < to && to <= k3.n_layers,
        "layer range [{}, {to}) is not inside [{}, {})",
        args.from,
        k3.first_k_dense_replace,
        k3.n_layers
    );
    // **The latent, not `hidden`.** The single most consequential line in this file.
    let (expert_in, moe_inter, ne) = (k3.expert_in, k3.moe_inter, k3.n_experts);
    std::fs::create_dir_all(&args.out_dir)?;

    let layers: Vec<usize> = (args.from..to).collect();
    // **ONE list of layers, driving both which shards get opened and what the resident writer
    // emits** — the pairing `convert_v4`'s `MODEL_LEVEL` comment insists on ("two things that
    // have to agree"). This converter broke it from the other end until review 2026-08-11: the
    // resident pass looped over `src.names()`, which is every tensor in every OPENED SHARD, so
    // the resident set was a function of the checkpoint's shard boundaries rather than of the
    // request. Two consequences, both real on this checkpoint:
    //
    //   * layer 0's dense `mlp.*` tensors landed in the resident set only because
    //     `embed_tokens` (no `layers.`, so always wanted) happens to share shard 1 with them.
    //     Luck, not design — and `--from 5 --to 6` on a different sharding would drop the dense
    //     layer while `ensure!(kept > 0)` stayed quiet.
    //   * a partial run silently over-collected: every trunk tensor that shared a shard with a
    //     requested layer, while the manifest claimed only `[from, to)`.
    //
    // The dense prefix is ALWAYS included, and that is not the same as "the requested range".
    // Layer 0 carries no experts, so it is never in `from..to`; it is also trunk the model
    // cannot decode without. `convert_v4` has no dense layers and so never had to say this.
    let resident_layers: Vec<usize> = (0..k3.first_k_dense_replace).chain(args.from..to).collect();
    let wanted: Vec<String> = resident_layers
        .iter()
        .map(|l| format!("{K3_TEXT_PREFIX}layers.{l}."))
        .collect();
    // A tensor this run is responsible for: model-level (no `layers.` at all — `embed_tokens`,
    // `norm`, `lm_head`, `output_attn_res_*`, verified against all 60 families in
    // `tensor-families.tsv`) or belonging to one of `resident_layers`.
    let in_scope = |n: &str| {
        !is_vision(n)
            && (!n.contains("layers.") || wanted.iter().any(|p| n.starts_with(p.as_str())))
    };
    // Only the shards holding what we are about to read. A checkpoint still downloading has
    // truncated shards, and opening one would fail a run over layers we are not converting.
    // K3 is 1.42 TiB across 96 shards, so this is not a nicety.
    let src = Safetensors::open_indexed(&args.src_dir, in_scope)?;
    eprintln!(
        "convert_k3: hidden={} latent={expert_in} moe_inter={moe_inter} experts={ne} \
         layers {}..{to} (of {}, dense prefix {})",
        k3.hidden, args.from, k3.n_layers, k3.first_k_dense_replace
    );

    // 1. Routed experts → one `.f4` per MoE layer. Pure repack, and `RoutedRepack` is the shared
    // loop — the ONLY thing this converter contributes here is `k3_expert_base` and the latent.
    let repack = RoutedRepack {
        tool: "convert_k3",
        out_dir: &args.out_dir,
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

    // A verification pass STOPS HERE, and the reason is a near-miss recorded in `convert_v4`:
    // `--verify --from 1 --to 3` against a whole artifact would otherwise truncate its resident
    // set to two layers and rewrite the manifest to claim them, orphaning every other `.f4`.
    if args.verify_only {
        eprintln!("convert_k3: verify-only, resident set and manifest untouched");
        return Ok(());
    }

    // 2. Everything else, verbatim. BF16 and F32 tensors are copied rather than converted, so
    // `SafeWriter`'s `Cow` borrows them straight from the source mmap and host RAM peak stays the
    // sum of the CONVERTED tensors — which here is zero.
    //
    // Driven off what the checkpoint HAS *within this run's scope* — `in_scope` above, the same
    // predicate that chose the shards — minus the routed experts, which `.f4` carries. Anything
    // else present lands in the resident set, which is the right default for a trunk that is
    // entirely BF16 (G0 item 3) and for a port whose next stages will need tensors this one does
    // not read.
    //
    // **The norms are copied VERBATIM as BF16, and `convert_v4` does not do that** — it widens
    // them to f32 (`add_widened`) because the loader reads norms as f32. K3's trunk is BF16 on
    // disk and this converter is a passthrough, so the widening question belongs to whoever
    // writes the K3 loader; it is recorded here rather than left for them to discover, because a
    // BF16 tensor read as f32 is not a length error, it is half the rows at wrong magnitudes.
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
    // **A guard that can actually fire.** The previous one was `kept > 0` with a message blaming
    // "the prefix or the two predicates" — but neither can produce zero: a wrong prefix makes the
    // resident set LARGER, and every opened shard carries trunk tensors. Review 2026-08-11.
    //
    // This one names the hazard that exists: if `is_routed`'s literal ever stops matching, 896
    // experts x 6 tensors x each converted layer land in `resident.safetensors` — 15.72 GB per
    // layer of duplicated weights, an artifact that still loads. `>=` rather than `==` because a
    // shard may hold more than one layer's experts, and this run only accounts for its own.
    let want_skipped = layers.len() * ne * 6;
    ensure!(
        skipped_routed >= want_skipped,
        "only {skipped_routed} routed tensors were skipped, expected at least {want_skipped} \
         ({} layers x {ne} experts x 6) — `is_routed` is no longer matching the checkpoint's \
         expert names, and those weights are about to be duplicated into the resident set",
        layers.len()
    );
    ensure!(kept > 0, "the resident set is empty — nothing was in scope");
    let path = format!("{}/resident.safetensors", args.out_dir);
    w.write(&path)?;
    eprintln!(
        "convert_k3: wrote {path} — {kept} tensors ({skipped_routed} routed, \
         {skipped_vision} vision skipped)"
    );

    // 3. The manifest. The source config VERBATIM, wrapper and all — see the module header.
    let text = std::fs::read_to_string(format!("{}/config.json", args.src_dir))
        .with_context(|| "read source config.json")?;
    let mut manifest: serde_json::Value = serde_json::from_str(&text)?;
    // **`format` is `FormatMeta`, the shape the only reader in the tree parses** (`FormatMeta::load`
    // → `version`/`vq_*`/`fp8_block`). Until review 2026-08-11 this wrote a bespoke object —
    // `routed`/`layers`/`expert_in`/`moe_inter`/`n_experts`/`expert_stride` — which was a THIRD
    // meaning for one key and would have failed `FormatMeta::load` on a K3 artifact. Four of those
    // six keys were also a second, unvalidated copy of `ExpertHeader`, which the pool reads and
    // checks at open; a JSON duplicate nothing cross-checks can only drift from the bytes.
    manifest["format"] = serde_json::to_value(FormatMeta::current(FP8_BLOCK))?;
    // **`f4_source` is NOT decoration, and omitting it made every artifact this tool produced
    // unopenable.** `f4_layer_range` is "the loader's only source for which layers exist" and
    // treats an absent `f4_source` as a hard error — so a K3 artifact without it failed at load
    // with "not a convert_v4 artifact", which is both a refusal and a lie. Found by review
    // 2026-08-11; the earlier code recorded the range inside its own `format` object, where
    // nothing looks.
    //
    // `layers` is the range this artifact HOLDS, and `num_hidden_layers` is deliberately left
    // alone — every per-layer structure in a K3 config (`linear_attn_config`'s two arrays,
    // `first_k_dense_replace`) is indexed by the REAL layer id, so renumbering a partial artifact
    // as a small MODEL would mis-key all of them.
    manifest["f4_source"] = f4_source("convert_k3", &args.src_dir, args.from..to);
    finish_artifact(
        "convert_k3",
        &args.out_dir,
        &args.src_dir,
        &manifest,
        // No `chat_template.jinja`: K3's `tokenizer_config.json` has no `chat_template` at all —
        // rendering lives in `encoding_k3.py` and is S4's to hand-transliterate (plan, S1a tail).
        &["tokenizer.json", "tokenizer_config.json"],
    )
}
