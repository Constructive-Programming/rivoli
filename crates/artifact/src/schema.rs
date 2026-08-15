//! Model dimensions, parsed from the snapshot's `config.json`.
//!
//! **One type per architecture, and the type is the proof.** [`ModelConfig`] describes the
//! MLA (multi-head latent attention) + dense-prefix lineage — GLM-5.2, DeepSeek-V3.
//! [`V4Config`] describes DeepSeek-V4-Flash: shared-KV MQA, no dense layers, hash-routed
//! prefix, FP4 experts. [`K3Config`] describes Kimi-K3: KDA/MLA interleaved, routed experts
//! in a latent narrower than `hidden_size`, and — alone among the three — a config nested
//! behind a multimodal wrapper. Each refuses the others by name at [`crate::arch::Arch`],
//! *before* serde looks at a single dimension, so holding a value of any one of them is
//! evidence about which architecture the snapshot is.
//!
//! **Neither type may give an absent field a default.** V4's config lacks
//! `kv_lora_rank`, `qk_nope_head_dim`, `v_head_dim`, `intermediate_size` and
//! `first_k_dense_replace` *because it is not MLA and has no dense layers* — not because
//! they are optional. `#[serde(default)]` on those five would produce a `ModelConfig` that
//! parses, reports zeros, and launches the MLA decode path against an MQA model: fluent
//! output, wrong text, no crash. The same rule binds the other way and binds later
//! stages: a V4 field that S2/S3 needs is added as REQUIRED, or the guard rots.
//! `#[serde(default)]` survives here only on fields that are genuinely absent from *older
//! snapshots of the same architecture* — each one says so at its declaration.

use crate::arch::Arch;
use anyhow::{Context, Result, ensure};

/// Resolve a config document's architecture, from `model_type` and `architectures` BOTH.
///
/// Two independent statements of the same fact live in every config this engine has seen
/// (verified 2026-08-04 over all six manifests and source configs under `/var/db/rivoli`
/// and `/swarm/storage/ai/rivoli`: each carries `model_type` *and* `architectures`). When
/// both are present they must agree — a disagreement means the file was hand-edited, and
/// silently preferring one field would let that edit choose a decode path.
///
/// Absent-from-both and unrecognised are BOTH refusals. There is deliberately no fallback
/// to the architecture this engine happens to run today: an artifact whose architecture we
/// cannot name is one whose decode path we cannot choose, and choosing anyway is the exact
/// failure this port is built to avoid — it does not crash, it produces fluent wrong text.
///
/// Also returns the config string it resolved — so a refusal can quote the file rather
/// than only the enum variant.
fn arch_of_named(cfg: &serde_json::Value) -> Result<(Arch, String)> {
    let declared = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .into_iter()
        .chain(
            cfg.get("architectures")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str()),
        );
    let mut found: Option<(Arch, &str)> = None;
    for s in declared {
        let a = Arch::from_manifest_str(s)
            .with_context(|| format!("unsupported architecture {s:?}"))?;
        if let Some((prev, prev_s)) = found {
            ensure!(
                prev == a,
                "config disagrees with itself: {prev_s:?} and {s:?} name different architectures"
            );
        }
        // Keep the FIRST — `model_type` when present, which is the canonical field and the
        // one a reader will grep for. The agreement check above already compared it to
        // every later spelling, so nothing is lost by not overwriting.
        found.get_or_insert((a, s));
    }
    found.map(|(a, s)| (a, s.to_string())).context(
        "config declares neither `model_type` nor `architectures` — refusing rather than \
         assuming one. Every checkpoint and every artifact this engine has converted carries both",
    )
}

/// `<dir>/manifest.json` if present (a converted artifact), else `<dir>/config.json` (a
/// raw checkpoint). Shared by both configs' loaders so they cannot disagree on which file
/// describes a directory.
fn config_path(dir: &str) -> String {
    match std::fs::metadata(format!("{dir}/manifest.json")) {
        Ok(_) => format!("{dir}/manifest.json"),
        Err(_) => format!("{dir}/config.json"),
    }
}

// (`arch_of_artifact` — the one-call sniffing entry `Engine::open` will use — returns
// with its caller at M4; a pub fn with no caller is the shape this file already deleted
// twice. `arch_of_named` below is the whole mechanism.)

/// A config schema that describes exactly one architecture.
///
/// The binding of schema to architecture is a trait CONSTANT rather than a check each
/// impl remembers to write, so a third config added later cannot acquire a parse that
/// skips the discriminant — [`parse_config`] is the only constructor and it always
/// consults `ARCH`.
pub trait ArchConfig: Sized + serde::de::DeserializeOwned {
    /// The architecture a document must declare to parse as this type.
    const ARCH: Arch;
    /// Cross-field checks, run on every successful parse.
    fn validate(&self) -> Result<()>;
}

/// Parse one config document as `T`, refusing it unless it declares `T::ARCH`.
///
/// The arch check happens BEFORE serde looks at a dimension, so "wrong architecture" is
/// reported as itself rather than as whichever field the other architecture happens to
/// omit first. `ModelConfig` used to fail V4 with `missing field kv_lora_rank`, which
/// reads like a corrupt checkpoint rather than like a different model.
pub fn parse_config<T: ArchConfig>(text: &str) -> Result<T> {
    let doc: serde_json::Value = serde_json::from_str(text)?;
    let (got, declared) = arch_of_named(&doc)?;
    // The offending STRING as well as the resolved variant: `"deepseek_v4"` is what the
    // reader will grep the config for, and `DeepseekV4` is what the code calls it.
    ensure!(
        got == T::ARCH,
        "this snapshot declares {declared:?} ({got:?}), but this is the {:?} schema — the \
         two architectures do not share a decode path",
        T::ARCH
    );
    let cfg: T = serde_json::from_str(text)?;
    cfg.validate()?;
    Ok(cfg)
}

/// [`parse_config`] over `<dir>`'s manifest or config.
pub fn load_config<T: ArchConfig>(dir: &str) -> Result<T> {
    let path = config_path(dir);
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    parse_config(&text).with_context(|| format!("parse {path}"))
}

// (`ensure_f32_positive` returns with K3Config at M9 — its only caller.)

/// Both of an expert's input widths must divide the group-scale span exactly.
///
/// `vq_row_bytes`/`vq_groups` and their `.f4` counterparts round up with only a
/// `debug_assert` to catch a ragged dim, so in a RELEASE build a bad width silently
/// truncates every expert row instead of failing. Each width is an `i_dim` for some
/// projection — gate/up take `expert_in`, down takes `moe_inter` — so one check covers both.
///
/// `expert_in` is the routed block's entry width, not `hidden_size`; see
/// [`crate::quant::vq_expert_layout`] for why those differ on K3 and why this
/// takes the former. GLM-5.2 and V4 pass `cfg.hidden` because for them they are equal.
pub(crate) fn ensure_group_aligned(
    expert_in: usize,
    moe_inter: usize,
    group: usize,
    what: &str,
) -> Result<()> {
    // Named for the CONFIG KEY, not for the parameter. The reader of this message is holding a
    // `config.json` and needs to know which field to look at; `expert_in 6144 is not a multiple
    // of ...` makes them go find out what feeds `expert_in` first. Which key that is differs by
    // model, so both candidates are named.
    for (key, dim) in [
        ("hidden_size / routed_expert_hidden_size", expert_in),
        ("moe_intermediate_size", moe_inter),
    ] {
        ensure!(
            dim.is_multiple_of(group),
            "{key} is {dim}, not a multiple of {what} {group} — expert rows would \
             silently truncate in a release build"
        );
    }
    Ok(())
}

// (`ensure_f4_group_aligned`, the V4/K3 wrapper over the same check, returns with its
// first consumer at M8 — a helper with no caller is a warning under -D warnings, not a seam.)
