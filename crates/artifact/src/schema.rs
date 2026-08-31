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
/// **Both fields are read at the DOCUMENT ROOT, and both must be THERE — since 2026-08-31
/// that is a check rather than a habit.** `Qwen4ExpForConditionalGeneration`'s
/// `vision_config.model_type` is the root string verbatim (`qwen4_exp`), so a recursive
/// search would match the vision block and resolve a trimmed or vision-only config as the
/// whole model — the trap [`crate::arch::from_manifest_str`] documents in full. Root-only
/// reading closes the recursive case; requiring both fields closes the case root-only cannot,
/// which is a nested block PROMOTED to a document (`{"model_type": "qwen4_exp"}` plus vision
/// dimensions), because a sub-config carries `model_type` alone and never `architectures`.
///
/// > **STRENGTHENED 2026-08-31 (S1 review).** This accepted a document declaring only ONE of
/// > the two, and the paragraph above was a comment claiming a property nothing enforced. The
/// > doc's own evidence is what makes the check safe: every checkpoint and every artifact this
/// > engine has converted carries both (verified 2026-08-04 over all six manifests and source
/// > configs; a converted artifact's `manifest.json` is the source `config.json` plus a
/// > `format` section, so it inherits both fields). A file with one is hand-edited or
/// > promoted, and choosing a decode path for it is the failure mode this function exists to
/// > refuse.
///
/// Also returns the config string it resolved — so a refusal can quote the file rather
/// than only the enum variant.
fn arch_of_named(cfg: &serde_json::Value) -> Result<(Arch, String)> {
    const MODEL_TYPE: &str = "model_type";
    const ARCHITECTURES: &str = "architectures";
    let declared = cfg
        .get(MODEL_TYPE)
        .and_then(|v| v.as_str())
        .map(|s| (MODEL_TYPE, s))
        .into_iter()
        .chain(
            cfg.get(ARCHITECTURES)
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str())
                .map(|s| (ARCHITECTURES, s)),
        );
    let mut found: Option<(Arch, &str)> = None;
    let mut fields: Vec<&str> = Vec::new();
    for (field, s) in declared {
        let a = crate::arch::from_manifest_str(s)
            .with_context(|| format!("unsupported architecture {s:?}"))?;
        if let Some((prev, prev_s)) = found {
            ensure!(
                prev == a,
                "config disagrees with itself: {prev_s:?} and {s:?} name different architectures"
            );
        }
        fields.push(field);
        // Keep the FIRST — `model_type` when present, which is the canonical field and the
        // one a reader will grep for. The agreement check above already compared it to
        // every later spelling, so nothing is lost by not overwriting.
        found.get_or_insert((a, s));
    }
    let (arch, spelling) = found.context(
        "config declares neither `model_type` nor `architectures` — refusing rather than \
         assuming one. Every checkpoint and every artifact this engine has converted carries both",
    )?;
    ensure!(
        fields.contains(&MODEL_TYPE) && fields.contains(&ARCHITECTURES),
        "config names {spelling:?} in `{}` alone — the other of `model_type` / `architectures` \
         is absent or names nothing this engine recognises, so there is only ONE statement of \
         identity here and nothing for it to agree with. A sub-config carries `model_type` \
         without `architectures`, so this is the shape a nested block promoted to a document \
         has, and Qwen3.8-Flash-Next's `vision_config.model_type` is its root string verbatim",
        fields[0]
    );
    Ok((arch, spelling.to_string()))
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

/// Which architecture `<dir>` holds, from its manifest alone — **the sniff, and the only
/// one.**
///
/// > **ARRIVED 2026-08-16 with `main`'s second arm.** This was a comment reserving the name
/// > ("returns with its caller at M4; a pub fn with no caller is the shape this file already
/// > deleted twice"). The caller is now real: with two engine arms, `main` has to know which
/// > config type to load BEFORE it loads one, and the previous shape — load `ModelConfig`,
/// > read `ArchConfig::ARCH` off the type — could only ever answer "GLM", because that load
/// > refuses everything else.
///
/// It reads the same `model_type`/`architectures` pair [`parse_config`] does, through the
/// same [`arch_of_named`], so the sniff and the parse cannot disagree about a file: a
/// directory this accepts as Glimmer is one `GlimmerConfig::load` will accept, and the
/// per-type refusal stays in place behind it as the check that the two agreed.
///
/// Deliberately NOT a `--model`/`--arch` flag: the artifact IS the model, and a flag naming
/// the architecture is a flag that can disagree with the weights it describes. Disagreeing is
/// not a crash — it launches the wrong decode path and produces fluent wrong text.
pub fn arch_of_artifact(dir: &str) -> Result<Arch> {
    let path = config_path(dir);
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {path}"))?;
    Ok(arch_of_named(&doc)
        .with_context(|| format!("sniff {path}"))?
        .0)
}

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

// > **UPDATED 2026-08-16.** This read "(`ensure_f32_positive` returns with K3Config at M9 —
// > its only caller.)". It arrived at **M7** instead, with `GlimmerConfig`: the anticipated
// > shape was right and the milestone was wrong. `K3TextConfig::validate` became the second
// > caller later the same day (M9), which is what keeps this shared rather than inlined
// > into `glimmer_config.rs`.

/// Every f64 a kernel narrows to f32, checked in the **f32 domain** rather than only in the
/// f64 the JSON carries. Underflow (`x <= 2^-150` -> `0.0f32`) collapses the value; overflow
/// (`x > ~3.4e38` -> `inf`, which passes any bare `> 0.0` test) is the silent one. A `1e-46`
/// eps passes an f64 positivity test and reaches every RMSNorm as `0.0f32`.
///
/// Shared by `GlimmerTextConfig` and `K3TextConfig` because it is one rule about the
/// hardware, not a coincidence of two checkpoints — which is what separates it from the
/// dimension serde renames in each config, where the shared text IS a coincidence and stays
/// exempted rather than factored. Factored in the old tree 2026-08-11, when Glimmer's arrival
/// made jscpd report it; it arrives here already factored, for the same reason.
pub(crate) fn ensure_f32_positive(items: &[(&str, f64)]) -> Result<()> {
    for &(what, x) in items {
        let narrowed = x as f32;
        ensure!(
            narrowed > 0.0 && narrowed.is_finite(),
            "{what} {x} narrows to {narrowed} in f32, the domain the kernels work in"
        );
    }
    Ok(())
}

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

// > **ARRIVED 2026-08-16 with M8's `V4Config`.** This read "(`ensure_f4_group_aligned`, the
// > V4/K3 wrapper over the same check, returns with its first consumer at M8 — a helper with
// > no caller is a warning under -D warnings, not a seam.)". The consumer is now real and the
// > prediction held: it landed at M8, unchanged in shape.

/// [`ensure_group_aligned`] at [`crate::quant::F4_GROUP`] — the routed-expert scheme both
/// `.f4` models use, so both configs' `validate` want the same four arguments.
///
/// Wrapped rather than restated at each call: the five-line form was a duplication-gate failure
/// the moment K3 became the second caller in the old tree, and the group is not the interesting
/// part of either call. The interesting part is *which width* is `expert_in` — `cfg.hidden` on
/// V4, the 3584 latent on K3 — which is what the one-line call sites show.
///
/// Two callers ([`crate::v4_config::V4Config::validate`] and, since M9 the same day,
/// [`crate::k3_config::K3TextConfig::validate`] — which is the caller whose `expert_in` is
/// NOT `hidden`). It is here rather than inline in `v4_config.rs` for the reason
/// [`ensure_f32_positive`] is: the rule is about the FORMAT, which two architectures share,
/// not about either checkpoint.
pub(crate) fn ensure_f4_group_aligned(expert_in: usize, moe_inter: usize) -> Result<()> {
    ensure_group_aligned(
        expert_in,
        moe_inter,
        crate::quant::F4_GROUP,
        stringify!(F4_GROUP),
    )
}

#[cfg(test)]
mod tests {
    // tests: a firing panic IS the report, and the workspace denies both at lib level
    // (`drafter_config.rs`'s inline module carries the same line for the same reason).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// **The root-only, both-fields rule, against the SHIPPED document it exists for.**
    ///
    /// [`arch_of_named`]'s doc names Qwen3.8-Flash-Next's `vision_config.model_type` as the
    /// reason it reads the root and requires both fields. That reason was a comment until this
    /// test: `arch.rs`'s spelling test feeds four bare strings to `from_manifest_str` and
    /// never sees a nested document at all, so nothing in the tree ran a real wrapper config
    /// through the resolution path.
    ///
    /// The vendored `config.json` is the whole point — 71 KB of the shipped file, whose root
    /// carries the wrapper's two spellings, whose `text_config.model_type` is `qwen4_exp_text`
    /// and whose `vision_config.model_type` is `qwen4_exp` verbatim. Three cases, and the two
    /// refusals are the ones that matter: a `vision_config` block promoted to a document
    /// (which has no root identity at all) and a bare `model_type` with the vision dimensions
    /// trimmed off (which has exactly one, and is what a hand-edit produces).
    #[test]
    fn the_shipped_qwen_wrapper_resolves_from_its_root_and_a_promoted_block_does_not() {
        let doc: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/measurement/qwen-reference/config.json"
        ))
        .expect("the vendored config must parse");
        let (arch, spelling) = arch_of_named(&doc).expect("the shipped wrapper must resolve");
        assert_eq!(arch, Arch::QwenFlashNext);
        assert_eq!(
            spelling, "qwen4_exp",
            "the ROOT `model_type` is what resolved — not `text_config`'s `qwen4_exp_text`, \
             which names the text half, and not `vision_config`'s identical `qwen4_exp`"
        );
        // The near-miss, asserted rather than assumed: if upstream renamed the vision block's
        // model_type, this rule would still be right and its reason would be gone.
        assert_eq!(
            doc["vision_config"]["model_type"], doc["model_type"],
            "the vision block's spelling is no longer the root's — re-argue the rule below"
        );
        for (why, promoted) in [
            (
                "a vision_config block promoted to a document declares no root identity",
                serde_json::json!({"vision_config": {"model_type": "qwen4_exp"}}),
            ),
            (
                "a bare `model_type` is ONE statement of identity, which is the shape a \
                 hand-trimmed or promoted sub-config has",
                serde_json::json!({"model_type": "qwen4_exp"}),
            ),
            (
                "and the same, the other way round: `architectures` alone is also one",
                serde_json::json!({"architectures": ["Qwen4ExpForConditionalGeneration"]}),
            ),
        ] {
            let err = arch_of_named(&promoted)
                .map(|(a, s)| format!("resolved as {a:?} via {s:?}"))
                .expect_err(why);
            let err = format!("{err:#}");
            assert!(
                err.contains("model_type") && err.contains("architectures"),
                "{why}: the refusal must name both fields, since which one is missing is the \
                 actionable half — got {err}"
            );
        }
    }

    /// The both-fields rule binds every architecture, not just the fifth — so a GLM manifest
    /// with one field is refused too.
    ///
    /// Its own case because the rule was ADDED for qwen's near-miss and a reader will ask
    /// whether it was scoped to that row. It is not: two independent statements of identity is
    /// a property of every config this engine has seen, and a per-architecture exemption would
    /// be the "two authorities that can independently judge one configuration" hazard.
    #[test]
    fn one_field_is_refused_on_the_four_architectures_that_already_decode() {
        for (mt, arch) in [
            ("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
            ("deepseek_v4", "DeepseekV4ForCausalLM"),
            ("kimi_k3", "KimiK3ForConditionalGeneration"),
            ("muse_glimmer", "MuseGlimmerForConditionalGeneration"),
        ] {
            let both = serde_json::json!({"model_type": mt, "architectures": [arch]});
            assert!(
                arch_of_named(&both).is_ok(),
                "{mt}: a document with BOTH fields must still resolve — this rule tightens \
                 what is refused, and tightening it into refusing real checkpoints is the \
                 regression to catch here"
            );
            for one in [
                serde_json::json!({"model_type": mt}),
                serde_json::json!({"architectures": [arch]}),
            ] {
                assert!(
                    arch_of_named(&one).is_err(),
                    "{mt}: a document declaring one field alone must be refused, on every \
                     architecture and not only on the one whose near-miss motivated the rule"
                );
            }
        }
    }
}
