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
    F4_NAMING_K3, RoutedRepack, SafeWriter, Safetensors, finish_artifact,
};
use rivoli::artifact::model::{K3Config, load_config};
use rivoli::artifact::quant::{K3_TEXT_PREFIX, f4_expert_stride, k3_expert_base};

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
/// SIBLINGS of `language_model` in the name tree, so anything that filtered by "not under
/// language_model" would also drop nothing and anything that copied everything would carry ~600 MB
/// of weights this engine has no path for. `quantization_config.ignore` names both, which is the
/// one part of that block worth believing.
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
    // Only the shards holding what we are about to read. A checkpoint still downloading has
    // truncated shards, and opening one would fail a run over layers we are not converting.
    // K3 is 1.42 TiB across 96 shards, so this is not a nicety.
    let wanted: Vec<String> = layers
        .iter()
        .map(|l| format!("{K3_TEXT_PREFIX}layers.{l}."))
        .collect();
    let src = Safetensors::open_indexed(&args.src_dir, |n| {
        !is_vision(n)
            && (!n.contains("layers.") || wanted.iter().any(|p| n.starts_with(p.as_str())))
    })?;
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
    // Driven off what the checkpoint HAS, not off a list of names this port believes in. The two
    // exclusions are explicit and each is a one-line predicate above; anything else present in the
    // source lands in the resident set, which is the right default for a trunk that is entirely
    // BF16 (G0 item 3) and for a port whose next stages will need tensors this one does not read.
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
        let (bytes, dtype, shape) = src.raw(name)?;
        w.add(name, dtype, shape.to_vec(), bytes);
        kept += 1;
    }
    ensure!(
        kept > 0,
        "no resident tensors matched — every name was filtered, which means the prefix or the \
         two predicates are wrong rather than the checkpoint being empty"
    );
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
    manifest["format"] = serde_json::json!({
        "routed": "f4",
        "layers": [args.from, to],
        "expert_in": expert_in,
        "moe_inter": moe_inter,
        "n_experts": ne,
        // Recomputed rather than carried out of the loop: `RoutedRepack` derives it from
        // `expert_in`/`moe_inter` through the same `f4_expert_stride`, so there is one definition
        // and no local that could drift from the bytes on disk.
        "expert_stride": f4_expert_stride(expert_in, moe_inter),
    });
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
