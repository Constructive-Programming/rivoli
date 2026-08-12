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
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

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
fn arch_of(cfg: &serde_json::Value) -> Result<Arch> {
    arch_of_named(cfg).map(|(a, _)| a)
}

/// [`arch_of`], also returning the config string it resolved — so a refusal can quote the
/// file rather than only the enum variant.
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

/// The architecture `dir`'s artifact declares — the one discriminant, read from the one
/// file.
///
/// No caller in this tree yet. It exists for the multi-model branch's `--help` rendering,
/// which re-renders against the artifact's architecture; it is here rather than there so
/// that the manifest is parsed in exactly one place. Today the live consumers of the
/// discriminant are [`parse_config`]'s refusals.
pub fn arch_of_artifact(dir: &str) -> Result<Arch> {
    let path = config_path(dir);
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let doc: serde_json::Value = serde_json::from_str(&text)?;
    arch_of(&doc).with_context(|| format!("parse {path}"))
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

/// Every f64 a kernel narrows to f32, checked in the **f32 domain** rather than only in the
/// f64 the JSON carries — the narrowing pair `V4Config`'s `swiglu_limit` documents at length.
/// Underflow (`x <= 2^-150` -> `0.0f32`) collapses the value; overflow (`x > ~3.4e38` ->
/// `inf`, which passes any bare `> 0.0` test) is the silent one. A `1e-46` eps passes an f64
/// positivity test and reaches every RMSNorm as `0.0f32`.
///
/// Shared by `K3TextConfig` and `GlimmerTextConfig` because it is one rule about the hardware,
/// not a coincidence of two checkpoints — which is what separates it from the dimension serde
/// renames above, where the shared text IS a coincidence and stays exempted rather than
/// factored. Factored 2026-08-11 when Glimmer's arrival made jscpd report it.
fn ensure_f32_positive(items: &[(&str, f64)]) -> Result<()> {
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
/// [`crate::artifact::quant::vq_expert_layout`] for why those differ on K3 and why this
/// takes the former. GLM-5.2 and V4 pass `cfg.hidden` because for them they are equal.
fn ensure_group_aligned(
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

/// [`ensure_group_aligned`] at [`crate::artifact::quant::F4_GROUP`] — the routed-expert
/// scheme both `.f4` models use, so both configs' `validate` want the same four arguments.
/// Wrapped rather than restated at each call: the five-line form was a duplication-gate
/// failure the moment K3 became the second caller, and the group is not the interesting part
/// of either call. The interesting part is *which width* is `expert_in` — `cfg.hidden` on V4,
/// the 3584 latent on K3 — which is what the one-line call sites now show.
fn ensure_f4_group_aligned(expert_in: usize, moe_inter: usize) -> Result<()> {
    ensure_group_aligned(
        expert_in,
        moe_inter,
        crate::artifact::quant::F4_GROUP,
        stringify!(F4_GROUP),
    )
}

/// GLM-5.2 nests theta under `rope_parameters`; we only need theta.
#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,

    // --- MLA attention ---
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    /// RoPE'd portion of each head's query/key.
    pub qk_rope_head_dim: usize,
    /// non-RoPE'd portion of each head's query/key.
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,

    // --- DSA lightning indexer (sparse attention; absent fields = no indexer
    // info, and the dsa/misa attention modes fail loudly at startup) ---
    #[serde(default)]
    pub index_n_heads: usize,
    #[serde(default)]
    pub index_head_dim: usize,
    #[serde(default)]
    pub index_topk: usize,
    /// Per-layer "full" (owns an indexer) / "shared" (reuses the nearest
    /// preceding full layer's top-k) — GLM-5.2's trained-in IndexShare.
    #[serde(default)]
    pub indexer_types: Vec<String>,

    // --- MoE ---
    #[serde(rename = "n_routed_experts")]
    pub n_experts: usize,
    #[serde(rename = "num_experts_per_tok")]
    pub top_k: usize,
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter: usize,
    /// Dense-layer MLP intermediate width (the first `dense_layers` layers).
    #[serde(rename = "intermediate_size")]
    pub dense_inter: usize,
    #[serde(rename = "n_shared_experts")]
    pub n_shared: usize,
    /// First `dense_layers` layers are dense MLP, not MoE (first_k_dense_replace).
    #[serde(rename = "first_k_dense_replace")]
    pub dense_layers: usize,
    #[serde(rename = "routed_scaling_factor")]
    pub routed_scale: f64,
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Router affinity. Absent on snapshots that predate the field; GLM-5.2 and the
    /// DeepSeek-V3 lineage are `sigmoid`, DeepSeek-V4 is `sqrtsoftplus`. Resolved by
    /// `scoring()` and rejected at load if unrecognised — a wrong affinity picks
    /// plausible-but-wrong experts and never crashes.
    #[serde(default)]
    scoring_func: Option<String>,

    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    pub rms_norm_eps: f64,
    /// GLM-5.2 nests theta; the DeepSeek/Llama lineage puts `rope_theta` at top
    /// level. Accept either — this is the first thing that fails on a non-GLM
    /// config, before any dimension is even looked at.
    #[serde(rename = "rope_parameters", default)]
    rope: Option<RopeParameters>,
    #[serde(rename = "rope_theta", default)]
    rope_theta_flat: Option<f64>,
}

impl ArchConfig for ModelConfig {
    const ARCH: Arch = Arch::GlmMoeDsa;

    /// Cheap semantic checks at the boundary, so a mismatched snapshot fails
    /// here with a clear message rather than as an out-of-bounds panic deep in
    /// the decode loop.
    fn validate(&self) -> Result<()> {
        if self.dense_layers > self.n_layers {
            anyhow::bail!(
                "dense_layers {} > n_layers {}",
                self.dense_layers,
                self.n_layers
            );
        }
        if self.top_k > self.n_experts {
            anyhow::bail!("top_k {} > n_experts {}", self.top_k, self.n_experts);
        }
        if self.n_shared < 1 {
            anyhow::bail!("n_shared_experts {} < 1", self.n_shared);
        }
        if !self.hidden.is_multiple_of(self.n_heads) {
            anyhow::bail!(
                "hidden {} not divisible by n_heads {}",
                self.hidden,
                self.n_heads
            );
        }
        // `softmax` is deliberately NOT accepted: the reference skips the top-k
        // renormalization for it (`if score_func != "softmax"`), so mapping it onto
        // this path would silently apply `norm_topk_prob` where the model expects
        // none. No model we target uses it.
        match self.scoring_func.as_deref() {
            None | Some("sigmoid") | Some("sqrtsoftplus") => {}
            Some(other) => anyhow::bail!(
                "unsupported scoring_func {other:?} — implemented: sigmoid, sqrtsoftplus"
            ),
        }
        if self.rope_theta() == 0.0 {
            anyhow::bail!(
                "no rope_theta: expected either a nested `rope_parameters.rope_theta` \
                 (GLM-5.2) or a top-level `rope_theta` (DeepSeek/Llama lineage)"
            );
        }
        // VQ_GROUP=64 is a multiple of the 8 the 12-bit packing needs, so this one check
        // covers the indices too. GLM-5.2: 6144, 2048 — both clean.
        ensure_group_aligned(
            self.hidden,
            self.moe_inter,
            crate::artifact::quant::VQ_GROUP,
            stringify!(VQ_GROUP),
        )?;
        // No ceiling on the intermediate widths: the LDS-staging MoE kernel that
        // imposed one (inter ≤ 16384, gfx1151's 64KB budget) is gone. `swiglu`
        // (linalg.hip) is elementwise with zero dynamic LDS, and moe.hip stages
        // nothing on purpose — LDS capped occupancy and measured slower. The old
        // guard would have refused the DeepSeek-V3 family on intermediate_size
        // 18432 for a constraint that no longer exists.
        Ok(())
    }
}

impl ModelConfig {
    /// Load from the artifact's `manifest.json` (the config fields live at top level
    /// alongside a `format` section; serde ignores the unknown key), falling back to a
    /// bare `config.json` for reading a raw checkpoint.
    ///
    /// Refuses a non-GLM snapshot by name. That refusal is the whole guard: every reader
    /// of this type's public fields — `gpu.rs`, `pin.rs`, `main.rs` — obtains its value
    /// from here, so a `ModelConfig` in hand IS the evidence that the snapshot is MLA.
    ///
    /// An inherent wrapper over [`load_config`] rather than a bare call, because ~8 call
    /// sites across `gpu.rs`/`pin.rs`/`main.rs`/`bin` already spell `ModelConfig::load`.
    /// `V4Config` is new and has none, so it uses the generic form directly.
    pub fn load(dir: &str) -> Result<Self> {
        load_config(dir)
    }

    /// The router affinity this model was trained with. Infallible because
    /// `validate` has already rejected anything it cannot map.
    pub fn scoring(&self) -> crate::math::Scoring {
        match self.scoring_func.as_deref() {
            Some("sqrtsoftplus") => crate::math::Scoring::SqrtSoftplus,
            _ => crate::math::Scoring::Sigmoid,
        }
    }

    pub fn rope_theta(&self) -> f64 {
        // `validate` has already established one of the two is present.
        self.rope
            .as_ref()
            .map(|r| r.rope_theta)
            .or(self.rope_theta_flat)
            .unwrap_or(0.0)
    }

    /// Validate the indexer config for the dsa/misa attention modes and return
    /// the per-layer full/shared flags (`true` = full). Called only when a
    /// sparse mode is requested — dense/streaming decode must keep working on
    /// snapshots whose config predates the indexer fields.
    pub fn indexer_layout(&self) -> Result<Vec<bool>> {
        if self.index_n_heads == 0 || self.index_head_dim == 0 || self.index_topk == 0 {
            anyhow::bail!(
                "config.json has no DSA indexer dims (index_n_heads/index_head_dim/index_topk)"
            );
        }
        if self.indexer_types.len() != self.n_layers {
            anyhow::bail!(
                "indexer_types has {} entries, expected n_layers={}",
                self.indexer_types.len(),
                self.n_layers
            );
        }
        let full: Vec<bool> = self
            .indexer_types
            .iter()
            .map(|t| match t.as_str() {
                "full" => Ok(true),
                "shared" => Ok(false),
                other => Err(anyhow::anyhow!("unknown indexer type {other:?}")),
            })
            .collect::<Result<_>>()?;
        if !full[0] {
            anyhow::bail!("layer 0 is 'shared' but has no preceding full layer");
        }
        Ok(full)
    }

    /// Total per-head query/key dimension (nope + rope).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Experts computed per MoE layer per token: the `top_k` routed picks the router
    /// selects (`num_experts_per_tok`) plus the `n_shared` always-on shared experts.
    /// Fixed by the trained model — the size of every MoE launch and of the expert
    /// stream, and the concurrency the stream needs to run them all at once.
    pub fn experts_per_layer(&self) -> usize {
        self.top_k + self.n_shared
    }
}

// ── DeepSeek-V4-Flash ───────────────────────────────────────────────────────────────
//
// A separate struct rather than optional fields on `ModelConfig`, and separate serde
// declarations rather than a shared core. The duplication is the POINT: a shared schema
// is exactly the mechanism by which a field added for one architecture would silently
// satisfy the other's parse, which is the defect this stage exists to prevent. Making
// them share would also move `ModelConfig`'s public fields behind a nested struct and
// rewrite every `cfg.hidden` in gpu.rs/pin.rs/main.rs for no semantic gain.

/// The `quantization_config` block. Checked rather than ignored: `.f4`'s repack and the
/// resident fp8 path each assume one specific scheme, and a checkpoint quantized another
/// way would decode to plausible-but-wrong weights with no error anywhere.
#[derive(Debug, Clone, Deserialize)]
struct QuantConfig {
    fmt: String,
    scale_fmt: String,
    weight_block_size: Vec<usize>,
}

/// DeepSeek-V4-Flash-0731. Shared-KV MQA (one `wkv` entry serving as both K and V for
/// all heads), grouped low-rank output projection, hyper-connection residuals, a
/// hash-routed prefix, and routed experts shipped as FP4 nibbles with e8m0 block scales.
///
/// **The model name is deliberate and load-bearing** (kept through the 2026-08-09
/// rename-for-behaviour pass): this struct IS that checkpoint's `config.json` — every
/// `#[serde(rename)]` below is one of its JSON keys, the unrenamed fields are its keys
/// verbatim — so deserializing another model's config through it is a refusal by design.
///
/// Every field is REQUIRED. See the module header for why that is not negotiable.
#[derive(Debug, Clone, Deserialize)]
pub struct V4Config {
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    #[serde(rename = "vocab_size")]
    pub vocab: usize,

    // --- shared-KV MQA. `head_dim` (512) is the FULL per-head width and the width of the
    // single KV entry; `qk_rope_head_dim` (64) is its RoPE'd tail. There is no
    // nope/rope SPLIT of separate tensors as in MLA — the last 64 dims of one 512-wide
    // vector are rotated in place, which is why `qk_nope_head_dim` has no meaning here.
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    /// Groups the output projection is split into (`wo_a` is `[o_groups·o_lora_rank, …]`).
    pub o_groups: usize,
    pub o_lora_rank: usize,
    /// One KV head, shared by all `n_heads` queries. Validated == 1: the whole attention
    /// frontend is written against a single shared entry.
    pub num_key_value_heads: usize,
    /// The sliding-window span, and the size of the KV ring. Required, not defaulted:
    /// `Attention.forward` indexes the cache as `kv_cache[:, start_pos % win]`, so a wrong
    /// `win` silently attends to the wrong rows rather than failing. S2b had to pass this
    /// in explicitly because it was missing here.
    pub sliding_window: usize,
    /// RMSNorm epsilon, used by `q_norm`/`kv_norm` AND by the weightless QK-norm. Required
    /// for the same reason: a default that differs from the checkpoint shifts every norm
    /// slightly and produces fluent, wrong text.
    pub rms_norm_eps: f64,

    // --- per-layer KV compression. `compress_ratios[l] == 0` means pure sliding-window
    // (no compressor, no indexer, base `rope_theta`, YaRN OFF); `!= 0` selects
    // `compress_rope_theta` WITH YaRN; `== 4` additionally carries an indexer.
    pub compress_ratios: Vec<usize>,

    // --- RoPE. Carried WHOLE, even though S1a reads none of it — the one exception to the
    // "a field must have a reader" rule above, and it is earned. `precompute_freqs_cis`
    // takes seven arguments and the reference's own `ModelArgs` DEFAULTS disagree with this
    // checkpoint on two: `rope_factor: 40` against the config's 16, `compress_rope_theta:
    // 40000.0` against 160000. A type exposing half the group invites S2 to take the rest
    // from those defaults and build a wrong RoPE table on all 41 compressor layers — no
    // error, just wrong positions. All of it, or none of it.
    pub compress_rope_theta: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub max_position_embeddings: usize,

    // jscpd:ignore-start — the four MoE serde renames, which coincide with `ModelConfig`'s
    // because both architectures declare these under the SAME JSON names.
    //
    // The copy is the point, for the same reason this module keeps two separate serde
    // declarations at all (see the module header): the two architectures agreeing on four
    // JSON names today is a coincidence of the checkpoints, not a shared contract, and a
    // shared struct would become the attractor for a fifth field that is NOT shared.
    //
    // PRICED, and rejected on the arithmetic rather than on scope. `#[serde(flatten)]` on
    // both sides nests `ModelConfig`'s public fields, so every `cfg.n_experts` /
    // `.top_k` / `.moe_inter` / `.n_shared` becomes `cfg.moe.*` — ~100 call sites across
    // gpu.rs, pin.rs, format.rs, convert.rs, main.rs and the tests — to
    // delete FOUR LINES of serde attribute. This is a cost exemption and the tree has
    // precedent for those: `gpu.rs:20` exempts the launcher import list because a glob
    // import would cost the compile-time check that every name exists.
    //
    // What must NOT be shared, and is not: the fields around them. `ModelConfig` gives
    // `norm_topk_prob` and `scoring_func` `#[serde(default)]` because GLM snapshots predate
    // them; on V4 a missing `scoring_func` or `swiglu_limit` is silent-wrong arithmetic,
    // not an old file, so both are required here.
    //
    // --- MoE ---
    #[serde(rename = "n_routed_experts")]
    pub n_experts: usize,
    #[serde(rename = "num_experts_per_tok")]
    pub top_k: usize,
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter: usize,
    #[serde(rename = "n_shared_experts")]
    pub n_shared: usize,
    // jscpd:ignore-end
    /// `Gate.forward`'s last line, `weights *= self.route_scale` (`model.py:588`), applied
    /// to every routed expert's weight after the top-k renormalization.
    ///
    /// **Required, and it is the RoPE block's argument again — the one place that exception
    /// was earned.** `ModelArgs.route_scale` defaults to `1.` (`model.py:56`) while this
    /// checkpoint ships `1.5` (both `config.json`'s `routed_scaling_factor` and the
    /// reference's own `inference/config.json` `"route_scale": 1.5`). A reader taking the
    /// default would scale every routed contribution by 1/1.5 — fluent, wrong, no crash.
    /// Carried with no reader in this crate yet, deliberately, so the MoE combine cannot
    /// reach for the default when it lands.
    ///
    /// There is no `norm_topk_prob` twin, and that is not an oversight: `Gate.forward`
    /// renormalizes on `score_func != "softmax"` (`model.py:587`), NOT on the config flag,
    /// and V4's `scoring_func` is `sqrtsoftplus` — so the renormalization is unconditional
    /// here and the config's `norm_topk_prob: true` has no effect to carry.
    #[serde(rename = "routed_scaling_factor")]
    pub routed_scale: f64,
    scoring_func: String,
    /// The first `n_hash_layers` layers route by a `tid2eid[vocab, top_k]` table indexed
    /// by token id — the router scores are computed but the SELECTION bypasses them.
    /// Those layers carry `ffn.gate.tid2eid` and NO `ffn.gate.bias`; the rest are the
    /// reverse. `layer_routes_by_hash` is the one reader.
    #[serde(rename = "num_hash_layers")]
    pub n_hash_layers: usize,
    /// SwiGLU clamp. rivoli's SwiGLU is unclamped, so a lost `10.0` here is silent-wrong,
    /// not a crash — required, and validated non-zero.
    pub swiglu_limit: f64,
    /// `"fp4"`. The `.f4` repack copies e2m1 nibbles verbatim; against an `"fp8"` export
    /// of the same model it would read fp8 bytes as nibble pairs and produce noise.
    pub expert_dtype: String,

    // --- attention-tensor quantization ---
    quantization_config: QuantConfig,

    // --- lightning indexer, on the `compress_ratio == 4` layers. `indexer.wq_b` is
    // `[index_n_heads · index_head_dim, q_lora_rank]`, confronted with the tensor by
    // `convert_v4::write_layer_resident`. ---
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    /// How many compressed blocks the lightning indexer keeps:
    /// `index_score.topk(min(index_topk, end_pos // ratio))` (`model.py:433`).
    ///
    /// **Required, and the reason is NOT the one `routed_scale` carries.** There the
    /// reference's default (`1.`) disagrees with the checkpoint (`1.5`), so a reader taking
    /// the default is silently wrong. Here they agree — `ModelArgs.index_topk` is 512
    /// (`model.py:77`) and so is the shipped config, a pairing `v4_base_matches_the_shipped_config`
    /// checks against the real file. The hazard is the other one: serde's `usize` default is
    /// **0**, which is a legal-looking value rather than an obviously-absent one.
    /// `topk(min(0, n))` yields a zero-width selection, `Attention.forward` concatenates it
    /// with the sliding-window list (`model.py:519`), and the `cat` is perfectly legal — so
    /// every `compress_ratio == 4` layer silently degrades to pure sliding-window attention.
    /// Fluent, wrong, no error: the same class as `routed_scale`, reached by the other route.
    /// (An earlier version of this comment claimed it "fails loudly"; it does not, and the
    /// oracle agrees — `forward.rs`'s `topk_idx(&score, 0)` returns an empty row.)
    ///
    /// No upper bound is available or needed: `min(index_topk, end_pos / ratio)` clamps from
    /// above, so an absurd value is a no-op, and nothing in this crate is sized by it.
    ///
    /// **Carried with no reader in this crate yet, deliberately** — like `routed_scale`, and
    /// for a sharper reason: THREE types declare a field of this name (`ModelConfig`'s, read
    /// by `gpu.rs`; `v4oracle::weights::V4Config`'s, hard-coded to 512; and this one), which
    /// is exactly the setup where the decode path reaches for the wrong one.
    ///
    /// For anyone who finds it looking inert: at 512 it does not truncate until **2052
    /// tokens** — `4 * (index_topk + 1)`, enforced by `tests/kvcompress.rs`'s
    /// `indexer_topk_never_cuts_at_the_emit_prompt` and recorded in
    /// `docs/investigations/v4-flash-port.md`, "A hole S3 inherits". Below that the
    /// selection is decided entirely by the causal mask, so a wrong value changes nothing
    /// observable. That is a property of the prompt length, not evidence the field is unused.
    pub index_topk: usize,

    /// Hyper-connection streams. `hc_*_fn` is `[_, hc_mult · hidden]`, likewise confronted
    /// with the tensor by `convert_v4`.
    pub hc_mult: usize,

    /// Sinkhorn passes in `hc_split_sinkhorn` — `Block.hc_pre`'s row/column normalization
    /// of the mHC mixing matrix (model.py:686).
    ///
    /// **Added 2026-08-05 by S3, because the layer loop could not be written without it.**
    /// `launch_hc_pre` takes it as a parameter *specifically* so the kernel and `config.json`
    /// cannot drift, and its doc says "passing it from `V4Config`" — but no such field
    /// existed, only `v4oracle::weights::V4Config::hc_sinkhorn_iters`, which is the oracle's
    /// own transliteration and must not be what the engine reads. That is the exact shape
    /// `index_topk`'s doc warns about: three types declare a field of one name and the decode
    /// path reaches for the wrong one.
    ///
    /// Required like every other field here, so the `ensure!` in `validate` reaches exactly
    /// one case: a config that writes `0` explicitly. `rivoli_hc_pre` already refuses that
    /// (`kernels/linalg.hip:642`, `if (iters < 1) return 1003`), so this is a load-boundary
    /// restatement of a check the kernel makes — worth the two lines because the load
    /// boundary names the FIELD and the kernel names a code, and because a config is read
    /// once while the launcher fires 43 times a token.
    ///
    /// **What a zero would cost is smaller than it sounds, and an earlier draft of this
    /// comment overstated it as "the matrix is never normalized at all".** It is not:
    /// `hc_sinkhorn` runs a row softmax and one column divide BEFORE the loop
    /// (`kernels/linalg.hip:370`, `norm(1, HC_MULT);`, and `kernel.py:401-415` agrees), and
    /// only the remaining `iters - 1` passes are the plain row/column refinements. So zero
    /// loses the refinement, not the normalization.
    ///
    /// The reason the value must be *read* rather than *checked* is unchanged: it is a
    /// config value, and the engine reading its own is not something a gate substitutes for.
    ///
    /// > **CORRECTED 2026-08-07.** This said a numeric gate *cannot* pin the shipped 20,
    /// > "because at 20 a 4x4 positive matrix is far past convergence and 19 and 20 agree
    /// > bit-for-bit". That is the toy fixture's behaviour, not the checkpoint's: on real
    /// > weights 19 vs 20 moves 39,893/53,248 of `L0.pre.ffn_norm_out` and all 78 router
    /// > weights, so a golden emitted from the checkpoint DOES distinguish them. The full
    /// > measurement and how the error happened are in
    /// > `tests/v4_oracle.rs::sinkhorn_has_converged_long_before_iteration_20`.
    ///
    /// **Carried with no reader in this crate yet, deliberately**, exactly as `index_topk`
    /// and `routed_scale` are: the layer loop that would read it does not exist. The
    /// alternative to declaring it now is the loop reaching for
    /// `v4oracle::weights::V4Config`, which is the whole reason this field is here.
    pub hc_sinkhorn_iters: usize,

    /// `hc_eps` — and it is **five** things, not the one an earlier draft of this comment
    /// named. It is the floor added to `hc_head`'s sigmoid gate (`model.py:714`,
    /// `pre = sigmoid(...) + hc_eps`), and inside `hc_split_sinkhorn` it is *also* the
    /// `+ eps` after the comb softmax and in every row and column divide — `2·iters - 1` of
    /// them per token (`kernel.py:408, :413, :419, :423`; `kernels/linalg.hip:347` and the
    /// `norm` lambda at :358). So a zero moves `comb` as well as `pre`, in opposite
    /// directions, and removes the guard from those divisions — harmless there, since the
    /// sums are strictly positive, but that is a reason rather than an absence.
    ///
    /// `model.py:686` is the `hc_split_sinkhorn(...)` CALL; the expression itself is in
    /// `inference/kernel.py`, a different file.
    ///
    /// Added with `hc_sinkhorn_iters` and for the same reason: `launch_hc_pre` and
    /// `launch_hc_head_collapse` both take it and nothing here supplied it. `f64` because JSON
    /// numbers are; the kernels narrow to f32 at the call, as `rms_norm_eps` already does.
    ///
    /// Required rather than defaulted for the reason a default of 0.0 would be *nearly*
    /// right — 1e-6 against a sigmoid output in (0, 1). It perturbs every gate by a hair, in
    /// the direction of less signal from every stream, uniformly across 43 layers. Small,
    /// systematic and unattributable is the worst shape a numeric error can have here.
    ///
    /// Carried with no reader yet, deliberately — see [`V4Config::hc_sinkhorn_iters`].
    pub hc_eps: f64,
}

/// The YaRN block. Required, and its `type` is checked — see the RoPE note in [`V4Config`].
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub beta_fast: u32,
    pub beta_slow: u32,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    pub kind: String,
}

impl ArchConfig for V4Config {
    const ARCH: Arch = Arch::DeepseekV4;

    /// Cross-field checks. Each one guards a failure that produces text rather than an
    /// error, which is the whole hazard class of this port.
    fn validate(&self) -> Result<()> {
        // `compress_ratios` is indexed by layer id in `Attention.__init__`. The shipped
        // config carries 46 entries for 43 layers (the tail belongs to the mtp blocks), so
        // this is a floor, not an equality — but one entry short is an index panic mid-load
        // or, worse, a layer silently treated as ratio 0.
        ensure!(
            self.compress_ratios.len() >= self.n_layers,
            "compress_ratios has {} entries, need at least n_layers={}",
            self.compress_ratios.len(),
            self.n_layers
        );
        // Only 0 (sliding-window), 4 (compressor + indexer) and 128 (compressor only)
        // appear, and `Attention.__init__` branches on `== 4` exactly. An unseen ratio
        // would land in the "compressor, no indexer" arm by default; refuse instead.
        for (l, &r) in self.compress_ratios.iter().take(self.n_layers).enumerate() {
            ensure!(
                matches!(r, 0 | 4 | 128),
                "compress_ratios[{l}] = {r}; implemented: 0, 4, 128"
            );
        }
        ensure!(
            self.n_hash_layers <= self.n_layers,
            "num_hash_layers {} > n_layers {}",
            self.n_hash_layers,
            self.n_layers
        );
        ensure!(
            self.top_k <= self.n_experts,
            "top_k {} > n_experts {}",
            self.top_k,
            self.n_experts
        );
        // `MoE.__init__` asserts this outright; the shared expert is a single always-on
        // FFN in both the reference and rivoli's resident set.
        ensure!(
            self.n_shared == 1,
            "n_shared_experts {} != 1",
            self.n_shared
        );
        ensure!(
            self.num_key_value_heads == 1,
            "num_key_value_heads {} != 1 — this is shared-KV MQA, one entry for all heads",
            self.num_key_value_heads
        );
        ensure!(
            self.qk_rope_head_dim < self.head_dim,
            "qk_rope_head_dim {} must be inside head_dim {}",
            self.qk_rope_head_dim,
            self.head_dim
        );
        // `wo_a` is viewed as `(o_groups, o_lora_rank, n_heads·head_dim/o_groups)`, so a
        // ragged split would reshape into the wrong stride rather than fail.
        ensure!(
            self.o_groups > 0 && (self.n_heads * self.head_dim).is_multiple_of(self.o_groups),
            "n_heads·head_dim ({}) not divisible by o_groups {}",
            self.n_heads * self.head_dim,
            self.o_groups
        );
        // A zero theta is a silently wrong RoPE, not a crash — `ModelConfig::validate`
        // refuses one for the same reason. Both are live: ratio-0 layers use `rope_theta`,
        // the other 41 use `compress_rope_theta`.
        for (what, theta) in [
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
        ] {
            ensure!(theta > 0.0, "{what} {theta} must be positive");
        }
        ensure!(
            self.rope_scaling.kind == "yarn"
                && self.rope_scaling.factor > 0.0
                && self.rope_scaling.original_max_position_embeddings > 0,
            "rope_scaling type {:?} / factor {} / original {}: only YaRN is implemented, \
             and a zero original length disables the interpolation branch entirely",
            self.rope_scaling.kind,
            self.rope_scaling.factor,
            self.rope_scaling.original_max_position_embeddings
        );
        ensure!(
            self.scoring_func == "sqrtsoftplus",
            "scoring_func {:?}: V4 is sqrtsoftplus. A wrong router affinity picks \
             plausible-but-wrong experts and never crashes",
            self.scoring_func
        );
        ensure!(
            self.index_topk > 0,
            "index_topk must be positive — at 0 the indexer selects no compressed blocks \
             and every compress_ratio == 4 layer silently falls back to sliding-window \
             attention"
        );
        // A zero or negative scale silently zeroes or flips every routed contribution
        // while the shared expert keeps working — degraded fluent text, not a crash.
        ensure!(
            self.routed_scale > 0.0,
            "routed_scaling_factor {} must be positive — it multiplies every routed \
             expert's weight (`Gate.forward`'s `weights *= route_scale`)",
            self.routed_scale
        );
        // Checked in the f32 domain the KERNELS work in, not only in the f64 JSON carries.
        // `kernels/moe.hip:413` guards `!(swiglu_limit > 0.0f)` — spelled that way rather than
        // `<= 0.0f` because NaN fails every comparison and `fminf(gt, NaN)` returns `gt`, so `<=`
        // would admit the one value that silently disables the clamp. A bare `> 0.0` here misses
        // BOTH narrowing failures, and they fail in opposite ways:
        //
        //   * UNDERFLOW. `as f32` rounds to nearest even, so any `0 < x <= 2^-150`
        //     (7.006492321624085e-46) becomes `0.0f32` — and 7.1e-46 does NOT, it rounds up to the
        //     min subnormal. The consequence is LOUD: guard 1006 at the first MoE layer of prefill.
        //   * OVERFLOW. Float-to-float `as` saturates, so any finite `x > ~3.4e38` becomes
        //     `f32::INFINITY`, which passes every `> 0.0` test — and `fminf(gt, inf) == gt`, so the
        //     clamp becomes a NO-OP. That is exactly `v4oracle::Defect::SwigluUnclamped`, SILENT,
        //     and wrong all the way to the output.
        //
        // The silent direction is the one worth the check, which is why this is `is_finite()` and
        // not just a positivity test. Both verified numerically 2026-08-05.
        let narrowed = self.swiglu_limit as f32;
        ensure!(
            narrowed > 0.0 && narrowed.is_finite(),
            "swiglu_limit {} narrows to {narrowed} in f32, which is the domain the MoE kernel's \
             clamp works in. V4's SwiGLU is CLAMPED (10.0 in the shipped config) and rivoli's is \
             not, so a zero here is a rejected launch and an infinity is silently unclamped \
             arithmetic",
            self.swiglu_limit
        );
        ensure!(
            self.expert_dtype == "fp4",
            "expert_dtype {:?}: the .f4 repack reads e2m1 nibble pairs, and an fp8 export \
             of the same model would decode as noise",
            self.expert_dtype
        );
        let q = &self.quantization_config;
        ensure!(
            q.fmt == "e4m3" && q.scale_fmt == "ue8m0",
            "quantization_config {:?}/{:?}: the resident path decodes e4m3 weights with \
             ue8m0 block scales",
            q.fmt,
            q.scale_fmt
        );
        ensure!(
            q.weight_block_size == [crate::artifact::quant::FP8_BLOCK; 2],
            "quantization_config.weight_block_size {:?} != [{}, {}]",
            q.weight_block_size,
            crate::artifact::quant::FP8_BLOCK,
            crate::artifact::quant::FP8_BLOCK
        );
        // Both mHC scalars are checked non-zero here rather than left to the kernels, and
        // the two cases differ. `hc_sinkhorn_iters == 0` IS refused downstream —
        // `kernels/linalg.hip:642` returns guard 1003 — so this is a restatement at the load
        // boundary, which names the field where the kernel names a code. `hc_eps == 0` is
        // refused by nothing: it is arithmetic, not a shape, and it perturbs every gate and
        // every Sinkhorn divide by 1e-6 uniformly, which no per-layer tolerance would read
        // as anything but depth. That one is the reason this pair is here at all.
        ensure!(
            self.hc_sinkhorn_iters > 0,
            "hc_sinkhorn_iters is 0 — `hc_split_sinkhorn`'s refinement passes would all be \
             skipped, on every layer (and `rivoli_hc_pre` refuses it with guard 1003)"
        );
        ensure!(
            self.hc_eps > 0.0,
            "hc_eps {} must be positive — it floors `hc_head`'s sigmoid gate AND every \
             row/column divide inside hc_split_sinkhorn",
            self.hc_eps
        );
        // The FP4 group scale runs along the INPUT dim, so both expert input widths must
        // divide it exactly — `f4_groups` rounds up, and a ragged tail would give the
        // last group a scale covering fewer weights than the kernel assumes.
        // `self.hidden` is the `expert_in` argument, and on V4 those are equal.
        ensure_f4_group_aligned(self.hidden, self.moe_inter)
    }
}

impl V4Config {
    /// `compress_ratios[layer]`, bounds-checked against `n_layers` rather than against the
    /// vector — the vector is longer (the mtp tail), and reading past `n_layers` would
    /// return an mtp block's ratio for a main-path layer.
    pub fn compress_ratio(&self, layer: usize) -> Result<usize> {
        ensure!(layer < self.n_layers, "layer {layer} >= {}", self.n_layers);
        Ok(self.compress_ratios[layer])
    }

    /// Whether this layer carries `attn.compressor.*` (ratio != 0). Also selects
    /// `compress_rope_theta` + YaRN over the base theta with YaRN off.
    pub fn layer_has_compressor(&self, layer: usize) -> Result<bool> {
        Ok(self.compress_ratio(layer)? != 0)
    }

    /// Whether this layer carries `attn.indexer.*`. `Attention.__init__` builds one only
    /// at ratio EXACTLY 4 — 21 of the 41 compressor layers in the shipped checkpoint.
    pub fn layer_has_indexer(&self, layer: usize) -> Result<bool> {
        Ok(self.compress_ratio(layer)? == 4)
    }

    /// Whether this layer's gate selects experts from `tid2eid` instead of from the
    /// scores. Such a layer has `ffn.gate.tid2eid` and no `ffn.gate.bias`.
    pub fn layer_routes_by_hash(&self, layer: usize) -> bool {
        layer < self.n_hash_layers
    }

    // No `experts_per_layer` twin of `ModelConfig`'s. On V4 the two kinds of expert are
    // not interchangeable: `top_k` routed experts are FP4 and stream from NVMe, while the
    // one shared expert is fp8 e4m3 and is resident. A single `top_k + n_shared` count is
    // right for a MoE launch and WRONG for per-token stream traffic, and it is the traffic
    // number this port keeps needing — so the two are spelled out separately at each use.
}

// ── Kimi-K3 ─────────────────────────────────────────────────────────────────────────
//
// A third struct, on the same argument the V4 section states: a shared schema is the
// mechanism by which a field added for one architecture silently satisfies another's parse.

/// The two layer arrays under `linear_attn_config`, which between them decide whether a
/// layer runs KDA or gated MLA.
///
/// **Both are required, and [`K3TextConfig::validate`] asserts the PARTITION** rather than
/// trusting either. The two reference implementations read opposite arrays — the C consumes
/// `full_attn_layers` and derives KDA as the complement, first-party `is_kda_layer` consumes
/// `kda_layers` and derives MLA — so neither is "the derived one", and a config where they
/// disagree would give two readers two different layer maps. The reference names that as the
/// mistake that "silently swaps KDA and MLA layers".
///
/// **Both arrays are ONE-BASED.** That is not a style note: zero-based MLA is
/// `[3, 7, …, 87, 91, 92]`, so reading them zero-based shifts every attention layer by one, in
/// a way no shape check sees — both families take `[hidden]` in and return `[hidden]`. The
/// tail is what makes the slip hard to eyeball: **91 and 92 are adjacent**, so the
/// every-fourth stride runs 3…91 (23 entries) and only 92 is off it.
/// [`K3TextConfig::layer_is_mla`] is the only reader that does the conversion, so there is one
/// place to get it right.
///
/// **The five scalars are here too, and their spellings came off the file** (vendored at
/// `docs/measurement/k3-reference/config.json`, revision
/// `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, fetched 2026-08-10). An earlier version of this
/// struct omitted them on the grounds that a guessed key refuses the real checkpoint — which was
/// the right rule and would have fired: `k3-architecture.md` §1 names them `kda_heads`,
/// `kda_head_dim` and `conv_k`, which are the **C reference's field names**, and the JSON calls
/// them `num_heads`, `head_dim` and `short_conv_kernel_size`. All three guesses were wrong.
///
/// `use_full_rank_gate` lives HERE, not on [`K3TextConfig`] — it was declared one level up until
/// the file was read, which would have refused every real K3 checkpoint on
/// `missing field \`use_full_rank_gate\``. Its default in the first-party modeling code is
/// `False` while every layer ships a `g_proj`, so the config value is the one that agrees with
/// the weights (G0 item 11).
///
/// Carried whole rather than field-by-field as S2 needs them, on `V4Config`'s RoPE argument: a
/// type exposing half a group invites the next stage to take the rest from a default that
/// happens to agree today.
#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttnConfig {
    pub full_attn_layers: Vec<usize>,
    pub kda_layers: Vec<usize>,
    /// 96 — KDA's own head count, which equals `num_attention_heads` in this checkpoint but is
    /// a separate field and must not be read from it.
    pub num_heads: usize,
    /// 128, and `d_k == d_v` for KDA.
    pub head_dim: usize,
    /// 4 — the depthwise causal conv's kernel width (`conv_k` in the C reference).
    pub short_conv_kernel_size: usize,
    /// **-5.0, and NEGATIVE is correct.** It multiplies the sigmoid rather than clamping or
    /// flooring it — trap 4 of `k3-architecture.md` §10 — so this is neither a bound nor an
    /// epsilon, and a positivity check on it would be wrong.
    pub gate_lower_bound: f64,
    /// True. See the struct doc: this field's LEVEL was the bug.
    pub use_full_rank_gate: bool,
}

/// Kimi-K3, as its `config.json` ships it: a `KimiK3ForConditionalGeneration` multimodal
/// wrapper around the text model.
///
/// The nesting is carried rather than flattened away, because it is load-bearing twice over.
/// The wrapper is the level that names the architecture ([`Arch::from_manifest_str`]), the
/// nested dict is the level that carries the dimensions, and `vision_config` — which this
/// port does not implement — is a sibling of `text_config` rather than of anything inside it.
/// Flattening would cost a hand-written `Deserialize` and would hide which level a field came
/// from, which §3e of the plan names as exactly how a key goes missing for the wrong reason.
#[derive(Debug, Clone, Deserialize)]
pub struct K3Config {
    #[serde(rename = "text_config")]
    pub text: K3TextConfig,
}

/// The `text_config` dict — Kimi-K3's text model, `KimiLinearForCausalLM`.
///
/// **Every field is REQUIRED**, as in [`V4Config`], and for the same reason: a defaulted
/// dimension does not crash, it produces fluent wrong text. A field whose JSON key this port
/// has not verified against the shipped file is *absent* rather than guessed — a wrong key on
/// a required field refuses the real checkpoint with `missing field`, which is loud and
/// fixable, while a guessed key with a `#[serde(default)]` is the silent version.
///
/// **Hold a [`K3Config`], not this.** This type is `pub` with `pub` fields and derives
/// `Deserialize`, so one can be produced by deserializing the inner dict alone or by a struct
/// literal — either of which skips [`parse_config`] and therefore skips both the architecture
/// check and `validate`. Only `K3Config` is evidence that those ran, which is the property the
/// module header claims and which holds for `ModelConfig`/`V4Config` because there the
/// validating type IS the carried type. Flagged by review 2026-08-10; the nesting is what
/// separates them, so the discipline is a convention here rather than a type guarantee, and
/// `main.rs`'s dispatch arm keeps the wrapper for exactly this reason.
///
/// `quantization_config` is absent on purpose, and it is the one omission that is a decision
/// rather than a gap: the block **mis-declares its own scope** (`targets: ["Linear"]` with an
/// `ignore` list that omits `routed_expert_{down,up}_proj` and `block_sparse_moe.gate.weight`,
/// all three of which ship BF16). The converter drives off the presence of `.weight_packed`
/// instead — `docs/investigations/k3-port.md` S1a item 5 — so a schema that read this block
/// would be reading a field nothing is allowed to trust.
#[derive(Debug, Clone, Deserialize)]
pub struct K3TextConfig {
    /// The NESTED architecture pair — `KimiLinearForCausalLM` / `kimi_linear`, which differs
    /// from the wrapper's. Carried so `validate` can assert it: this struct is reached by
    /// descending through `text_config`, and a descent that landed in some other dict of a
    /// multimodal config would otherwise be indistinguishable from the right one.
    pub model_type: String,
    pub architectures: Vec<String>,

    // jscpd:ignore-start — the four dimension serde renames, which coincide with
    // `V4Config`'s (and `ModelConfig`'s) because all three checkpoints declare these under
    // the SAME HuggingFace-standard JSON names.
    //
    // Exempted on exactly the argument the MoE-rename block above states, and the argument
    // is now stronger for having a third instance: three architectures agreeing on four
    // JSON names is a coincidence of the checkpoints, not a shared contract, and a shared
    // struct becomes the attractor for a FIFTH field that is not shared. K3 is the proof —
    // it agrees on these four and disagrees on `num_experts_per_token` (`_tok` everywhere
    // else), on `num_experts` (`n_routed_experts` on V4), and on nesting the whole dict
    // behind a wrapper. A shared core would have had to special-case all three.
    //
    // No factoring was priced this time, because the cheap one is refuted by construction:
    // `#[serde(flatten)]` over a shared `Dims` struct puts `cfg.dims.hidden` at every call
    // site in gpu.rs/pin.rs/format.rs/main.rs to delete four lines of attribute, which is
    // the arithmetic the MoE block already rejected.
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,
    // jscpd:ignore-end
    /// RMSNorm epsilon (1e-5 in the shipped config, and note the first-party MLA LoRA norms
    /// use 1e-6 where the C reference wrote 1e-5 — `k3-architecture.md` §5).
    ///
    /// `rms_norm_eps` was flagged in review as a key §1's table does not record — the doc spells
    /// it `rms_eps`, the C reference's field name. **Confirmed against the shipped file**
    /// 2026-08-10: the JSON key is `rms_norm_eps`, as on GLM and V4.
    pub rms_norm_eps: f64,
    /// `bfloat16`. The trunk dtype, asserted rather than assumed: G0 item 3 established the
    /// trunk is BF16, and an fp8 export of the same model read as BF16 is noise at every width.
    pub dtype: String,

    // --- attention. Which layer is which comes from the partition, not from a stride.
    pub linear_attn_config: LinearAttnConfig,
    /// The MLA head geometry, carried WHOLE for the reason `V4Config` carries all of RoPE: a
    /// type exposing half of it invites S2 to take the rest from `ModelArgs`-style defaults and
    /// build a wrong projection on all 24 layers. `num_key_value_heads` is 96 here — equal to
    /// `num_attention_heads`, i.e. NOT the MQA that V4 asserts `== 1`, and `validate` pins the
    /// equality so a copied V4 check cannot land here unnoticed.
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_key_value_heads: usize,
    /// NoPE, asserted POSITIVELY. Checking only that `rope_theta` is absent cannot tell "this
    /// model applies no rotation" from "we descended into the wrong dict" — plan §3e.
    pub mla_use_nope: bool,
    /// §3e's secondary reading, and the ONLY field here that must be **absent**. `Option` rather
    /// than a required field for that reason, which is the module header's rule read forward
    /// rather than broken: what is banned is a default standing in for a value the engine needs,
    /// and this is a value the engine must not find. `validate` refuses `Some`.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Defaults to `False` in the first-party modeling code, so it must come from the config
    /// rather than from a Rust default that happens to agree today (G0 item 11). Its partner
    /// `use_full_rank_gate` is on [`LinearAttnConfig`], one level down — see that struct.
    pub mla_use_output_gate: bool,
    /// AttnRes block size (12). The residual is taken across a block of layers rather than
    /// per layer — `k3-architecture.md` §3.
    pub attn_res_block_size: usize,

    // --- MoE. The routed experts run in a LATENT that is not `hidden_size`; see
    // `quant::vq_expert_layout` and plan §2 for what assuming otherwise costs.
    #[serde(rename = "num_experts")]
    pub n_experts: usize,
    /// `num_experts_per_token` — note the spelling. GLM and V4 both write
    /// `num_experts_per_tok`, and this checkpoint does not.
    ///
    /// **The field is `top_k` here and `text_config` ALSO has a key literally named `top_k`,
    /// which is 50 and has nothing to do with routing** — it is HuggingFace's sampling top-k,
    /// inherited from `PretrainedConfig`. Binding this from `top_k` selects 50 experts a token
    /// instead of 16: 3.1x the stream traffic, plausible output, no error. Noticed while reading
    /// the shipped file 2026-08-10; the `rename` is what keeps them apart, and the pinning test
    /// asserts the two values differ so this cannot be "simplified" later.
    #[serde(rename = "num_experts_per_token")]
    pub top_k: usize,
    /// Declared 2, but the checkpoint ships **one fused MLP** per layer (`down_proj`
    /// `[7168, 6144]` BF16) — G0 item 4. So this is the config's count of shared experts, not
    /// a count of tensors, and the converter must not go looking for two.
    #[serde(rename = "num_shared_experts")]
    pub n_shared: usize,
    /// `routed_expert_hidden_size` — the 3584-wide latent the routed experts are entered at,
    /// **not** `hidden_size` 7168. Named for the role, matching
    /// [`crate::artifact::quant::vq_expert_layout`]'s parameter.
    #[serde(rename = "routed_expert_hidden_size")]
    pub expert_in: usize,
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter: usize,
    /// RMSNorm on the routed AGGREGATE, in latent space, before the up-projection.
    pub latent_moe_use_norm: bool,
    pub moe_renormalize: bool,
    /// Grouped routing is **degenerate, not absent**: both are 1. Asserted rather than
    /// ignored — a checkpoint with real groups would need a grouped top-k this engine does
    /// not have, and would otherwise route through the ungrouped path with no error.
    pub num_expert_group: usize,
    pub topk_group: usize,
    /// `noaux_tc` — the first-party name for bias-on-selection-only, which is what
    /// `math::Scoring`'s sigmoid arm already implements.
    pub topk_method: String,
    /// `sigmoid`, and independent per expert — the scores do NOT sum to 1. Asserted for the
    /// reason `V4Config::scoring_func` is: a wrong router affinity picks plausible-but-wrong
    /// experts and never crashes.
    pub moe_router_activation_func: String,
    /// 1.0 today, and the multiply is kept anyway (`k3-architecture.md` §6's router block says
    /// so explicitly). A zero or negative scale silently zeroes or flips every routed
    /// contribution while the shared MLP keeps working — degraded fluent text, not a crash.
    #[serde(rename = "routed_scaling_factor")]
    pub routed_scale: f64,
    /// 1 — **every** layer after the dense prefix is MoE. Asserted rather than assumed: at 2
    /// only every other layer would be, and this port's layer loop would run the routed path on
    /// 46 layers that ship no expert tensors.
    pub moe_layer_freq: usize,
    /// SiTU-GLU's two betas (4.0 and 25.0), fused inside the fp4 expert kernel — plan §3b.
    ///
    /// **The second key is `activation_situ_linear_beta`, not `activation_linear_beta`.** §1 of
    /// the plan abbreviates the pair as "`activation_situ_beta` / `_linear_beta`", which reads as
    /// the latter; the file says otherwise (checked 2026-08-10). Declared wrong, this refused
    /// every real K3 checkpoint on `missing field` — the loud direction the port prefers, and it
    /// is why the vendored config and its pinning test now exist.
    pub activation_situ_beta: f64,
    #[serde(rename = "activation_situ_linear_beta")]
    pub activation_linear_beta: f64,

    // --- layer 0 is dense, and simultaneously a KDA layer and an AttnRes boundary.
    pub first_k_dense_replace: usize,
    #[serde(rename = "intermediate_size")]
    pub dense_inter: usize,
    /// `situ`. The dense layer and the shared MLP use the same SiTU-GLU as the routed experts,
    /// so a `silu` here would be a different activation on layer 0 and the shared path while the
    /// routed path stayed right — `k3-architecture.md` §3b's "watches the dense path go right
    /// and every routed expert stay wrong", in reverse.
    pub hidden_act: String,

    /// 0 — no MTP head, so no speculative decode and `MAXROW` is 1. Asserted rather than
    /// assumed: a non-zero value means tensors this port does not convert.
    pub num_nextn_predict_layers: usize,
    /// False. A tied head would read the output projection out of the embedding table, which
    /// is a different set of weights at the same shape.
    pub tie_word_embeddings: bool,
}

impl ArchConfig for K3Config {
    const ARCH: Arch = Arch::KimiK3;

    fn validate(&self) -> Result<()> {
        self.text.validate()
    }
}

impl K3TextConfig {
    /// Cross-field checks. As in [`V4Config`], each guards a failure that produces text
    /// rather than an error.
    fn validate(&self) -> Result<()> {
        // The descent check. `parse_config` already matched the WRAPPER's pair against
        // `Arch::KimiK3`; this is the nested pair, and it is what distinguishes "descended
        // into `text_config` of a Kimi-K3 config" from "descended into some other dict".
        // Both halves are quoted, and the pair is checked as ONE `ensure!` on purpose: which of
        // the two disagrees is not the interesting fact — "we are in the wrong dict" is, and a
        // reader holding the file wants both spellings in front of them to see that.
        ensure!(
            self.model_type == "kimi_linear"
                && self.architectures == ["KimiLinearForCausalLM".to_string()],
            "text_config declares {:?} / {:?} — a Kimi-K3 wrapper's text model is \
             \"kimi_linear\" / [\"KimiLinearForCausalLM\"]. Either this is not the dict we \
             think we descended into, or the checkpoint's text model is a different family",
            self.model_type,
            self.architectures
        );
        // §3e's SECONDARY check, and it is the reason this field exists at all. NoPE is asserted
        // positively below (`mla_use_nope`), because "this model applies no rotation" and "we
        // descended into the wrong dict" are otherwise the same observation — but the plan asks
        // for both readings, and without this the struct's lack of `deny_unknown_fields` means a
        // `rope_theta` sitting in `text_config` is silently ignored rather than being the signal
        // it is. An `Option` is the right shape here for once: the module bans defaults on
        // fields that must be PRESENT, and this one must be ABSENT.
        //
        // If a real K3 `text_config` turns out to carry a `rope_theta` for some path this port
        // does not walk, the fix is to delete this check and say so — not to relax it to a
        // value comparison, which would re-admit the wrong-dict case.
        ensure!(
            self.rope_theta.is_none(),
            "text_config carries rope_theta {:?}, but this model is NoPE (`mla_use_nope`) and \
             no rotation table is built. A rotary base in a dict that should have none is the \
             signal that the descent landed somewhere else",
            self.rope_theta
        );
        // A zero width passes every divisibility check below (0 is a multiple of anything)
        // and then sizes an expert row, an arena stride and a GEMV `dim` to nothing.
        for (what, dim) in [
            ("hidden_size", self.hidden),
            ("routed_expert_hidden_size", self.expert_in),
            ("moe_intermediate_size", self.moe_inter),
            ("intermediate_size", self.dense_inter),
            ("vocab_size", self.vocab),
            ("num_attention_heads", self.n_heads),
            ("num_hidden_layers", self.n_layers),
            ("attn_res_block_size", self.attn_res_block_size),
            // The two LoRA ranks belong here for a sharper reason than the rest, found by
            // review 2026-08-10: the MLA kernel's own guard does NOT catch a zero.
            // `kernels/attn.hip:293` is `if (kvl > MLA_ACC_REGS * SUBW || kvl % 128) return
            // 1004`, and `0 % 128 == 0` while `!(0 > 512)` — so a zero-width latent passes
            // guard 1004, and 24 layers of attention contribute nothing with no error
            // anywhere. `V4Config` omits the same pair; K3 is where §1 of the plan wrote the
            // constraint down, so it is restated here.
            ("q_lora_rank", self.q_lora_rank),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
            (
                "linear_attn_config.num_heads",
                self.linear_attn_config.num_heads,
            ),
            (
                "linear_attn_config.head_dim",
                self.linear_attn_config.head_dim,
            ),
            (
                "linear_attn_config.short_conv_kernel_size",
                self.linear_attn_config.short_conv_kernel_size,
            ),
        ] {
            ensure!(dim > 0, "{what} is 0");
        }
        // Not MQA. V4 asserts `num_key_value_heads == 1` because its whole attention frontend is
        // written against one shared KV entry; K3's MLA has one per query head, and a copied V4
        // check would refuse this checkpoint while a copied V4 *assumption* would size the cache
        // 96x too small. Pinned as the equality rather than as the literal 96.
        ensure!(
            self.num_key_value_heads == self.n_heads,
            "num_key_value_heads {} != num_attention_heads {} — K3's MLA is not MQA; every \
             query head has its own KV projection",
            self.num_key_value_heads,
            self.n_heads
        );
        // **Negative, and checked in f32.** `gate_lower_bound` MULTIPLIES the KDA gate's sigmoid
        // rather than clamping or flooring it (`k3-architecture.md` §10, trap 4), so its sign is
        // load-bearing and a positivity check on it would be exactly backwards. At 0 every gate
        // on all 69 KDA layers is zeroed and the recurrence contributes nothing — the model goes
        // quiet rather than wrong, which no tolerance downstream reads as an error. `-5.0` shipped.
        let gate_lb = self.linear_attn_config.gate_lower_bound as f32;
        ensure!(
            gate_lb < 0.0 && gate_lb.is_finite(),
            "linear_attn_config.gate_lower_bound {} narrows to {gate_lb} in f32; it must be \
             negative and finite — it MULTIPLIES the KDA gate's sigmoid (trap 4), so 0 silences \
             all {} KDA layers and a positive value inverts the decay",
            self.linear_attn_config.gate_lower_bound,
            self.linear_attn_config.kda_layers.len()
        );
        ensure!(
            self.moe_layer_freq == 1,
            "moe_layer_freq {} != 1 — this port's layer loop treats every layer past the dense \
             prefix as MoE, and at 2 half of them ship no expert tensors at all",
            self.moe_layer_freq
        );
        for (what, got, want) in [
            ("dtype", &self.dtype, "bfloat16"),
            ("hidden_act", &self.hidden_act, "situ"),
            (
                "moe_router_activation_func",
                &self.moe_router_activation_func,
                "sigmoid",
            ),
        ] {
            ensure!(
                got == want,
                "{what} is {got:?}, not {want:?} — each of these three changes the arithmetic \
                 without changing a single shape, so nothing downstream would refuse it"
            );
        }
        // The other half of guard 1004, restated at the load boundary because the boundary names
        // the FIELD where the kernel names a code. 512 (this checkpoint's value) sits exactly at
        // the cap: `MLA_ACC_REGS * SUBW` is 16 * 32, and `kernels/attn.hip:54` derives the 16
        // from the `⌈kvl/SUBW⌉` lane-private accumulator registers.
        //
        // TWO checks rather than one conjunction, so a refusal names which bound was crossed —
        // and so each test row proves its own half. As `a && b` both rows matched both halves of
        // the message and neither was a test of anything.
        ensure!(
            self.kv_lora_rank.is_multiple_of(128),
            "kv_lora_rank {} is not a multiple of 128 — `rivoli_mla_attend` refuses it with \
             guard 1004 (kernels/attn.hip:293)",
            self.kv_lora_rank
        );
        ensure!(
            self.kv_lora_rank <= 512,
            "kv_lora_rank {} exceeds the 512 (MLA_ACC_REGS * SUBW) the MLA kernel's \
             lane-private accumulator can hold — guard 1004, kernels/attn.hip:293",
            self.kv_lora_rank
        );
        self.validate_layer_partition()?;
        // TWO checks rather than one conjunction, for the reason `kv_lora_rank`'s pair above gives:
        // a refusal names which bound was crossed, and each test row proves its own half.
        ensure!(
            self.first_k_dense_replace > 0,
            "first_k_dense_replace is 0, so layer 0 would run the routed MoE path — and this \
             checkpoint ships no expert tensors for it, only the dense `intermediate_size` {} \
             pair, which would then go unused",
            self.dense_inter
        );
        ensure!(
            self.first_k_dense_replace < self.n_layers,
            "first_k_dense_replace {} is not below n_layers {} — every layer would be dense and \
             no routed expert would ever run",
            self.first_k_dense_replace,
            self.n_layers
        );
        ensure!(
            self.top_k > 0 && self.top_k <= self.n_experts,
            "num_experts_per_token {} is not in 1..={}",
            self.top_k,
            self.n_experts
        );
        ensure!(
            self.n_shared > 0,
            "num_shared_experts is 0 — the always-on MLP is a third of this layer's \
             arithmetic, and its absence is not something the routed path compensates for"
        );
        ensure!(
            self.num_expert_group == 1 && self.topk_group == 1,
            "num_expert_group {} / topk_group {}: grouped routing is degenerate in this \
             checkpoint and this engine has no grouped top-k. Real groups would route \
             through the ungrouped path with no error",
            self.num_expert_group,
            self.topk_group
        );
        ensure!(
            self.topk_method == "noaux_tc",
            "topk_method {:?}: only `noaux_tc` (bias on SELECTION only, never on the \
             returned weight) is implemented. Any other method picks plausible-but-wrong \
             experts and never crashes",
            self.topk_method
        );
        // Each of these defaults to `false` somewhere — in the first-party modeling code for
        // the two gates, and in Rust for any `bool` this port forgot to read. Requiring the
        // POSITIVE value means a config that omits one is a refusal, not a silent downgrade
        // to an architecture the weights were not trained for.
        for (what, flag) in [
            ("mla_use_nope", self.mla_use_nope),
            ("mla_use_output_gate", self.mla_use_output_gate),
            // One level down, and the label says so — the file puts it inside
            // `linear_attn_config`, which is where it belongs: it is a KDA property.
            (
                "linear_attn_config.use_full_rank_gate",
                self.linear_attn_config.use_full_rank_gate,
            ),
            ("latent_moe_use_norm", self.latent_moe_use_norm),
            ("moe_renormalize", self.moe_renormalize),
        ] {
            ensure!(
                flag,
                "{what} is false; this port implements only the true form and the shipped \
                 config sets it. Turning it off changes the arithmetic, not the shapes"
            );
        }
        ensure!(
            self.num_nextn_predict_layers == 0,
            "num_nextn_predict_layers {} != 0 — this checkpoint has no MTP head, so a \
             non-zero value means tensors nothing here converts and a batched verify pass \
             with no kernel behind it",
            self.num_nextn_predict_layers
        );
        ensure!(
            !self.tie_word_embeddings,
            "tie_word_embeddings is true — K3 ships a separate lm_head, and reading the \
             output projection out of the embedding table is a different set of weights at \
             the same shape"
        );
        // Every f64 the kernels narrow to f32, checked in the f32 domain rather than only in the
        // f64 JSON carries — the narrowing pair `V4Config`'s `swiglu_limit` documents at length.
        // Underflow (`x <= 2^-150` -> `0.0f32`) collapses the value; overflow (`x > ~3.4e38` ->
        // `inf`, which passes any bare `> 0.0` test) is the silent one.
        //
        // **`rms_norm_eps` belongs in this loop, and was checked in f64 alone until review
        // 2026-08-10.** `gpu.rs:1743` and `f4gpu.rs:325` both do `cfg.rms_norm_eps as f32`, and
        // `V4Config::hc_eps`'s own doc already said so ("the kernels narrow to f32 at the call,
        // as `rms_norm_eps` already does") — so the same check was being done two ways six lines
        // apart. A `1e-46` eps passes an f64 positivity test and reaches every RMSNorm as `0.0f32`.
        ensure_f32_positive(&[
            ("rms_norm_eps", self.rms_norm_eps),
            ("activation_situ_beta", self.activation_situ_beta),
            ("activation_situ_linear_beta", self.activation_linear_beta),
            // 1.0 today. Zero silently zeroes every routed contribution while the shared MLP
            // keeps working; negative flips them. Both are degraded fluent text, not a crash.
            ("routed_scaling_factor", self.routed_scale),
        ])?;
        // The routed experts' widths, and ONLY those: `expert_in` is the latent 3584, so this
        // says nothing about the trunk's 7168 or the shared MLP's 6144. Both of those happen
        // to be multiples of 32, so there is no hole today — but it is an accident of this
        // checkpoint, not something this call checks.
        ensure_f4_group_aligned(self.expert_in, self.moe_inter)
    }

    /// `full_attn_layers` and `kda_layers` must partition `1..=n_layers` — both present, no
    /// duplicates, no overlap, nothing missing, nothing out of range.
    ///
    /// Asserted rather than derived from one array, because the two reference implementations
    /// read opposite ones (see [`LinearAttnConfig`]). Every failure here is a layer running
    /// the wrong attention family, which is arithmetic rather than a shape — no length check
    /// downstream sees it.
    fn validate_layer_partition(&self) -> Result<()> {
        let (mla, kda) = (
            &self.linear_attn_config.full_attn_layers,
            &self.linear_attn_config.kda_layers,
        );
        ensure!(
            mla.len() + kda.len() == self.n_layers,
            "full_attn_layers ({}) + kda_layers ({}) = {} layers, but num_hidden_layers is {}",
            mla.len(),
            kda.len(),
            mla.len() + kda.len(),
            self.n_layers
        );
        // One pass over both, so overlap, duplication and gaps are all the same check: every
        // one-based id in range exactly once.
        let mut seen = vec![false; self.n_layers];
        for (what, ids) in [("full_attn_layers", mla), ("kda_layers", kda)] {
            for &one_based in ids {
                let l = one_based
                    .checked_sub(1)
                    .with_context(|| format!("{what} contains 0; these arrays are ONE-BASED"))?;
                let slot = seen.get_mut(l).with_context(|| {
                    format!(
                        "{what} contains layer {one_based}, past num_hidden_layers {}",
                        self.n_layers
                    )
                })?;
                ensure!(
                    !*slot,
                    "layer {one_based} appears twice across full_attn_layers/kda_layers — \
                     the two arrays must partition the layers, and an overlap means the two \
                     reference implementations would disagree about this layer's family"
                );
                *slot = true;
            }
        }
        // Unreachable while the length check above holds, and kept anyway: it is the
        // invariant the readers actually depend on, and the length check is the accident.
        if let Some(l) = seen.iter().position(|s| !s) {
            bail!(
                "layer {} (one-based) is in neither full_attn_layers nor kda_layers",
                l + 1
            );
        }
        Ok(())
    }

    /// Does zero-based `layer` run gated MLA? (Otherwise KDA.)
    ///
    /// **The one place the one-based → zero-based conversion happens.** The reference names
    /// getting it wrong as the mistake that "silently swaps KDA and MLA layers", and the
    /// swap is invisible downstream: both families take the same `[hidden]` input and return
    /// the same shape.
    ///
    /// **No caller in this tree yet** — `pub` for the S2 layer loop, and pinned meanwhile by
    /// `k3_baseline_parses`, which asserts the whole zero-based map including the adjacent tail.
    /// The linear scan is 24 elements against 93 layers, ~2.2k compares a token, which is
    /// nothing at this engine's rate; if it ever matters, `validate` has already proven the
    /// partition and can hand out a precomputed mask instead.
    pub fn layer_is_mla(&self, layer: usize) -> Result<bool> {
        ensure!(layer < self.n_layers, "layer {layer} >= {}", self.n_layers);
        Ok(self
            .linear_attn_config
            .full_attn_layers
            .contains(&(layer + 1)))
    }

    /// Is `layer` the dense-FFN prefix (no routed experts)? Layer 0 in the shipped config.
    ///
    /// No caller in this tree yet either. **Note the asymmetry with the sibling above: this one
    /// does not bounds-check**, because it cannot be wrong in a way a check would catch — an
    /// out-of-range id is not `< first_k_dense_replace` and reads as "not dense", which is the
    /// same answer it gives for every real layer but the first. `V4Config::layer_routes_by_hash`
    /// is the same shape for the same reason. `layer_is_mla` returns `Result` because there a
    /// missing id would read as "KDA", which is a positive claim about arithmetic.
    pub fn layer_is_dense(&self, layer: usize) -> bool {
        layer < self.first_k_dense_replace
    }
}

/// Muse Glimmer-30B, as its `config.json` ships it: a `MuseGlimmerForConditionalGeneration`
/// wrapper around the text model, with a sibling `vision_config` this port does not implement.
///
/// Same nesting as [`K3Config`], carried for the same reasons — see that type. `dtype` sits at
/// the WRAPPER level here rather than inside `text_config`, which is the one structural
/// difference and the reason this is not a copy of K3's shape.
#[derive(Debug, Clone, Deserialize)]
pub struct GlimmerConfig {
    #[serde(rename = "text_config")]
    pub text: GlimmerTextConfig,
    /// `bfloat16`. The whole checkpoint is BF16 — 59.553 GB across two shards, reconciled
    /// against the index's own `total_size` (`glimmer-architecture.md` §7). The model card's
    /// "approximately 4-bit precision, under 20 GB" describes separate GGUF releases; reading
    /// an fp8 or 4-bit export as BF16 is noise at every width, so this is asserted.
    pub dtype: String,
    /// Must be **absent**. `Option` for the reason `K3TextConfig::rope_theta` is one: what this
    /// module bans is a default standing in for a value the engine needs, and this is a value
    /// the engine must not find.
    ///
    /// **`dtype` alone does not prove the weights are unquantized**, which is the whole reason
    /// this field exists. The counter-example is vendored in this tree: K3's `config.json`
    /// declares `dtype: "bfloat16"` *and* a `quantization_config` with
    /// `format: "mxfp4-pack-quantized"`, `num_bits: 4`. Without this field serde would ignore
    /// such a block, and a packed Glimmer export would parse clean and be read as BF16 at
    /// every width — exactly what the `dtype` message above claims to prevent. Found by
    /// review 2026-08-11.
    ///
    /// Untyped `Value` on purpose: nothing may act on its contents. K3's schema records why a
    /// `quantization_config` is not trustworthy even when present — its `targets`/`ignore`
    /// lists mis-declare their own scope — so the only claim supported here is the negative
    /// one, that this converter reads unquantized checkpoints.
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

/// The `text_config` dict — Muse Glimmer's text model.
///
/// **Every field is REQUIRED**, as in [`V4Config`] and [`K3TextConfig`], and for the same
/// reason: a defaulted dimension does not crash, it produces fluent wrong text. **Hold a
/// [`GlimmerConfig`], not this** — same argument as `K3TextConfig`'s, which see.
///
/// The fields here are not the interesting part; what is absent is. Eight load-bearing
/// operations appear in no marketing surface. **Four have a config key and are therefore
/// guardable here** — `qk_scale_factor`, `post_norm_eps`, `output_multiplier`,
/// `final_logit_softcapping`, and `validate` checks all four. The other four are code-only
/// facts no schema can see: the weightless QK-norm, the centered `x*(1+w)` norm form, the
/// sandwich-norm placement, and the normed embedding. Those are S1b's fixtures.
/// `glimmer-architecture.md` §9 lists all fifteen traps.
///
/// > **CORRECTED 2026-08-11**, by review. This said "only two of them are visible as a config
/// > key at all (`qk_scale_factor`, `post_norm_eps`)" and routed "the rest" to S1b — which
/// > sent to a fixture the two fields `validate` deliberately checks HERE, and for the
/// > sharpest reason in this port: `output_multiplier` and `final_logit_softcapping` are
/// > argmax-invariant, so no greedy gate downstream can ever see them wrong.
#[derive(Debug, Clone, Deserialize)]
pub struct GlimmerTextConfig {
    /// The NESTED spelling, `muse_glimmer_text`. Carried so `validate` can assert the descent
    /// landed in the text dict rather than in `vision_config` — which is a sibling, declares
    /// `muse_glimmer_vision`, and has a `hidden_size` and a `num_attention_heads` of its own.
    /// That last point is why this check is not decoration: the vision dict would parse
    /// several of the fields below and refuse only on the ones it lacks.
    pub model_type: String,

    // jscpd:ignore-start — the four dimension serde renames, which coincide with
    // `K3TextConfig`'s, `V4Config`'s and `ModelConfig`'s because all four checkpoints declare
    // these under the SAME HuggingFace-standard JSON names.
    //
    // Exempted on exactly the argument the K3 block above states, and it is now a fourth
    // instance of it: four architectures agreeing on four JSON names is a coincidence of the
    // checkpoints, not a shared contract, and a shared struct becomes the attractor for a
    // FIFTH field that is not shared — `num_key_value_heads`, `head_dim`, `layer_types` and
    // `sliding_window` are each carried by some of these four and not others.
    //
    // > **CORRECTED 2026-08-11**, by review. This said Glimmer "needs `head_dim` as a
    // > FIRST-CLASS field, which none of the other three carry". **`V4Config` carries one**
    // > (`head_dim`, this file), and V4's is non-derivable for the same reason Glimmer's is:
    // > its pinned dims are hidden 4096 / 64 heads / head_dim **512**, and 4096/64 = 64.
    // > The wrong version invited a reader to "simplify" V4's field to `hidden / n_heads`,
    // > which is Glimmer's own trap 15 landed on V4.
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,
    // jscpd:ignore-end
    /// GQA: 2, against 32 query heads — 16 query heads per KV head.
    pub num_key_value_heads: usize,
    /// **128, and NOT `hidden / n_heads`.** 32 x 128 = 4096 while `hidden_size` is 6656, so
    /// `q_proj` is `[4096, 6656]` and is not square. Every other config here can derive a head
    /// dim and this one cannot; a port that derives it builds a 208-wide head and indexes past
    /// the end of every projection.
    pub head_dim: usize,
    #[serde(rename = "intermediate_size")]
    pub inter: usize,
    /// 1e-5, on `input_layernorm` and `pre_feedforward_layernorm` — and on the weightless
    /// QK-norm and embedding norm.
    pub rms_norm_eps: f64,
    /// **1e-8, and a different value on purpose**: the two POST norms
    /// (`post_attention_layernorm`, `post_feedforward_layernorm`) use this. Three orders of
    /// magnitude apart and assigned by position, so one eps for all four norms is wrong in a
    /// way nothing downstream reads as an error.
    pub post_norm_eps: f64,
    /// 3.87, multiplying **Q only** and applied AFTER the weightless QK-norm. It does not
    /// replace the `head_dim^-0.5` softmax scale; both apply.
    pub qk_scale_factor: f64,
    /// 1/sqrt(26). Pre-multiplies the logits before the tanh softcap below.
    pub output_multiplier: f64,
    /// 20.0. `logits = T * tanh(logits * output_multiplier / T)`.
    ///
    /// **Argmax-invariant, which makes it this model's gate blind spot** rather than a
    /// routine field: `tanh` is strictly increasing and `output_multiplier` is positive, so
    /// omitting both cannot change a greedy pick. Every probability, NLL and confidence value
    /// is wrong regardless. `glimmer-architecture.md` §5.
    pub final_logit_softcapping: f64,
    /// 2048, on the `sliding_attention` layers only. The window is inclusive of the current
    /// position: `[p-2047, p]`, exactly 2048 rows.
    pub sliding_window: usize,
    /// 52 entries. **Consumed as the array, never re-derived from a stride** — the
    /// `[s,s,s,full]` period is a fact about this checkpoint, not a rule, and a port that
    /// computes `i % 4 == 3` produces a model that is right until the first checkpoint whose
    /// pattern differs.
    ///
    /// Typed rather than `Vec<String>` so an unknown spelling is refused by **serde**, at
    /// deserialize time and unconditionally, instead of by a `validate` that a caller holding
    /// this struct directly can skip. The realistic wrong value is one dict away: this file's
    /// own `vision_config.layer_types` is `["window_attention", …]`.
    pub layer_types: Vec<LayerKind>,
    /// 52 entries: 500000.0 on sliding layers, **0 on full ones**.
    ///
    /// Read as a BOOLEAN, not as a per-layer base. The first-party code builds ONE cos/sin
    /// table from `rope_parameters.rope_theta` and passes it or `None` per layer, so a port
    /// that builds 52 tables is doing arithmetic nobody asked for — and one that reads the
    /// top-level theta and applies it everywhere rotates the 13 NoPE layers.
    pub layer_rope_theta: Vec<f64>,
    pub rope_parameters: GlimmerRope,
    pub max_position_embeddings: usize,
    /// False — `lm_head.weight` and `embed_tokens.weight` both ship, 2.690 GB each. The
    /// first-party class declares a tied-weights mapping that this checkpoint does not use, so
    /// the class is not evidence and the config is.
    pub tie_word_embeddings: bool,
    /// `silu`. Named `hidden_activation` here, where GLM and K3 say `hidden_act`.
    pub hidden_activation: String,
    /// False. No projection in the attention block carries a bias, and none ships.
    pub attention_bias: bool,
}

/// The twelve tensors every Muse Glimmer decoder layer ships, as `<layer prefix>.{}.weight`.
///
/// **One statement of this fact, read by both the converter and its test.** They had a copy
/// each until jscpd reported it 2026-08-11, and a shared list of *names* is exactly the thing
/// that must not be duplicated: two copies can disagree, and a name that exists but points at
/// the wrong tensor copies silently — `tests/k3_names.rs` exists because that failure mode
/// already cost this repo a round.
///
/// **Eight projections and four norms** — five projections in the attention block (`q`, `k`,
/// `v`, `o` and the output `gate`) and three in the MLP. The QK-norm is weightless and ships
/// nothing, and there is no bias anywhere (`attention_bias` is false and asserted).
///
/// > **CORRECTED 2026-08-11**, by review. This said "five projections and four norms", which
/// > is nine against a list of twelve — it counted the attention block and forgot the MLP.
/// > `pin.rs`'s `GlimmerLayerPin` had it right, so the tree disagreed with itself about the
/// > length of the one constant that exists to stop exactly that.
pub const GLIMMER_LAYER_TENSORS: [&str; 12] = [
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.gate_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// How many streaming slots a budgeted `GlimmerPin` keeps for the layers it could not pin.
///
/// **ONE at R1, and the number is NOT a correctness property — that was the first version's
/// mistake.** It said "two, and the number is a correctness requirement rather than a tuning
/// choice", arguing that one slot would be refilled while a kernel still read it. Two reviews
/// independently showed the argument does not work: kernel launches are ASYNCHRONOUS, so a host
/// that runs two layers ahead overwrites slot 0 under a live kernel with two slots exactly as
/// with one. **No finite slot count establishes write-after-read ordering; only a dependency
/// does** — `VmmBuf::ptr_mut`'s own contract covers the fill-then-read direction and names
/// `device_sync` as the mechanism for slot reuse on the analogous io_uring path.
///
/// So the count is purely how far a caller may run ahead, and R1's fill is SYNCHRONOUS: it
/// buys no overlap, so a second slot buys nothing and costs one extra streamed layer forever.
/// The arithmetic, at the shipped widths: `floor` charges every slot unconditionally, so each
/// one pins one layer fewer, and a streamed layer is **967.942 MB of host memcpy per token**.
///
/// **The dependency landed 2026-08-12 (S3 item 0), so raising this is no longer gated on writing
/// it.** [`GlimmerPin::layer`] performs a `device_sync` before every refill, and
/// `glimmer_residency.rs::a_slot_refill_cannot_land_under_a_live_kernel` gates it in both
/// directions. What raising the count still needs is the OTHER half: a synchronous fill buys no
/// overlap, so a second slot is only worth its extra streamed layer once the fill is async — and
/// that swap replaces the whole-device join here with an event on the fetch stream. This said "S5
/// raises this and adds the dependency in the same change"; the dependency went first.
pub const GLIMMER_STREAM_SLOTS: usize = 1;

/// Alignment slack in a `GlimmerPin`'s tier request. `DeviceTier::place` starts every
/// reservation at a 256-byte boundary and the pin makes at most `3 + 12·n_layers` = 627
/// placements, so the padding is under 160 KB; 1 MiB is ~6x that bound and 0.002% of the 55.7
/// GB it can sit beside.
pub const GLIMMER_PIN_SLACK: usize = 1 << 20;

/// The prefix Glimmer's text-side tensors carry. The `language_model.` segment is the
/// multimodal wrapper's, and it is on every text tensor — K3's port records the same shape as
/// a name nothing in its documentation mentioned.
pub const GLIMMER_LAYER_PREFIX: &str = "model.language_model.layers";

/// What a Glimmer layer's attention attends over. One entry per layer in
/// [`GlimmerTextConfig::layer_types`].
///
/// An enum rather than the checkpoint's raw string so that an unrecognised spelling cannot
/// reach the engine at all: serde refuses it while deserializing, which is before `validate`
/// and therefore not skippable. As `Vec<String>` a typo in any ONE of the three comparison
/// sites read as "not sliding" — i.e. as a positive claim of full attention over the whole
/// prefix, on a layer trained with a 2048 window. That is fluent wrong text, and the kind
/// this port cannot otherwise see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Attends over `[p-2047, p]` — exactly `sliding_window` rows, inclusive of `p`.
    SlidingAttention,
    /// Attends over the whole prefix, and carries no rotation (`layer_rope_theta` is 0).
    FullAttention,
}

/// Glimmer's `rope_parameters`. Distinct from GLM's private [`RopeParameters`] above rather
/// than an extension of it, on this module's one-type-per-architecture rule: GLM's carries
/// theta alone, and adding a `rope_type` there would either be a required field GLM's config
/// need not have or a defaulted one — and a default standing in for a scaling scheme is
/// exactly what this asserts against.
///
/// `rope_type` is asserted `default` rather than ignored: a scaling scheme silently
/// unimplemented is the V4 port's `Defect::RopeNoYarn`, where the frequencies stay plausible
/// at every scale and the text stays fluent.
#[derive(Debug, Clone, Deserialize)]
pub struct GlimmerRope {
    pub rope_theta: f64,
    pub rope_type: String,
}

impl ArchConfig for GlimmerConfig {
    const ARCH: Arch = Arch::MuseGlimmer;

    fn validate(&self) -> Result<()> {
        ensure!(
            self.dtype == "bfloat16",
            "dtype is {:?}, not \"bfloat16\" — this checkpoint is BF16 throughout (59.553 GB, \
             reconciled against the index's own total_size). The 4-bit artifacts the model \
             card advertises are separate GGUF releases and are not what this converter reads",
            self.dtype
        );
        // The other half of that claim. `dtype` says how the tensors are TYPED; this says they
        // are not additionally packed. K3 ships both at once, so neither check implies the
        // other and the `dtype` message would otherwise promise more than it delivers.
        ensure!(
            self.quantization_config.is_none(),
            "this config carries a quantization_config, so its weights are packed rather than \
             plain BF16 — `dtype: \"bfloat16\"` does not exclude that (Kimi-K3's checkpoint \
             declares both). This converter reads the unquantized release"
        );
        self.text.validate()
    }
}

impl GlimmerTextConfig {
    /// True when layer `layer` attends over a sliding window rather than the whole prefix.
    ///
    /// **`Result`, not `Option`, and the difference is not taste.** An absent answer here has
    /// to be an error the caller must handle, because every ergonomic way to collapse an
    /// `Option<bool>` — `unwrap_or(false)`, `unwrap_or_default()`, `matches!(_, Some(true))` —
    /// yields `false`, which is a *positive claim of full attention* about a layer that does
    /// not exist. `unwrap()` is not the escape hatch either: `Cargo.toml` denies
    /// `unwrap_used`. So the shape steered its only caller into the silently-wrong branch, and
    /// it did: `main.rs` wrote `.unwrap_or(false)` on the first try.
    ///
    /// `Result` collapses to `?` instead, which is what `K3TextConfig::layer_is_mla` does and
    /// for the identical reason recorded there. Flagged by both standing reviews 2026-08-11 —
    /// latent today (`validate` pins the length and the loop is `0..n_layers`), and it is the
    /// line S2's layer loop copies.
    pub fn layer_is_sliding(&self, layer: usize) -> Result<bool> {
        let kind = self.layer_types.get(layer).with_context(|| {
            format!(
                "layer {layer} is out of range: this model has {} layers",
                self.layer_types.len()
            )
        })?;
        Ok(*kind == LayerKind::SlidingAttention)
    }

    /// The shape of one [`GLIMMER_LAYER_TENSORS`] entry, derived from this config. `[o, i]`
    /// for a projection, `[hidden]` for a norm.
    ///
    /// **One shape table, because the alternatives were three.** The pin needs it to check
    /// what it places, [`Self::resident_bytes`] needs it to size the tier, and
    /// `tests/glimmer_names.rs` needs it to compare against the shipped checkpoint — and the
    /// third is what makes the other two trustworthy: that test resolves every entry here
    /// against `model.safetensors.index.json`, so this table is not a belief about the
    /// checkpoint, it is checked against it.
    ///
    /// The pairs matter more than the individual entries. `q_proj` and `self_attn.gate_proj`
    /// are both `[n_heads·head_dim, hidden]` and `k_proj`/`v_proj` are both
    /// `[kv_heads·head_dim, hidden]`, so within each pair a shape check proves nothing and
    /// only the NAME separates them; `o_proj` is the one that is transposed, and `down_proj`
    /// the other. Reading `hidden` for `n_heads·head_dim` — 6656 for 4096 — is the mistake
    /// this exists to make impossible, since `head_dim` is not `hidden / n_heads` here.
    pub fn layer_tensor_shape(&self, tensor: &str) -> Result<Vec<usize>> {
        let q = self.n_heads * self.head_dim;
        let kv = self.num_key_value_heads * self.head_dim;
        Ok(match tensor {
            "self_attn.q_proj" | "self_attn.gate_proj" => vec![q, self.hidden],
            "self_attn.k_proj" | "self_attn.v_proj" => vec![kv, self.hidden],
            "self_attn.o_proj" => vec![self.hidden, q],
            "mlp.gate_proj" | "mlp.up_proj" => vec![self.inter, self.hidden],
            "mlp.down_proj" => vec![self.hidden, self.inter],
            // All four norms, as a suffix rather than four literals.
            //
            // > **CORRECTED 2026-08-11**, by review. This said the suffix means "a fifth norm
            // > is covered rather than silently unmatched". Covered is not correct: the only
            // > fifth norm this architecture could grow is the QK-norm, which is per-head and
            // > `[head_dim]` = 128, not `[hidden]` = 6656 — so the arm would hand back a WRONG
            // > shape where the reader was promised a `bail!`. Inert today (the QK-norm is
            // > weightless and ships no tensor at all, `glimmer-architecture.md` §4 trap 2),
            // > and recorded because this is the one place where "covered" and "correct"
            // > diverge.
            t if t.ends_with("layernorm") => vec![self.hidden],
            _ => bail!(
                "{tensor} is not a Muse Glimmer layer tensor — GLIMMER_LAYER_TENSORS and this \
                 table disagree, which means one of them was extended and the other was not"
            ),
        })
    }

    /// Bytes the resident set occupies **in the artifact's dtypes** — bf16 projections, f32
    /// norms — from the config alone. 55.712 GB for the shipped model.
    ///
    /// Sized from [`Self::layer_tensor_shape`] rather than restating the shapes, so the number
    /// and the checks the pin makes cannot drift apart. It is what sizes the tier, which is
    /// what makes it load-bearing rather than documentation: under-count it and
    /// `DeviceTier::place` bails partway through the placement.
    ///
    /// **The norms are f32 here and bf16 in the checkpoint** — `convert_glimmer` widens them,
    /// the house convention every architecture follows — so this is 2.782 MB larger than the
    /// text half of the checkpoint. `tests/glimmer_names.rs` reproduces it from the shipped
    /// index's own shapes, which is where that 2.782 MB is accounted for rather than asserted.
    pub fn resident_bytes(&self) -> Result<usize> {
        Ok(self.global_bytes() + self.n_layers * self.layer_bytes()?)
    }

    /// Bytes ONE layer's twelve tensors occupy — **967.942 MB** for the shipped model.
    ///
    /// > Written as 967.889 MB until 2026-08-12. That is the figure with the four norms at
    /// > bf16, which is the CHECKPOINT's dtype — `convert_glimmer` widens them, so the artifact's
    /// > layer is 53,248 bytes larger. `resident_bytes`' own doc has always said so ("2.782 MB
    /// > larger"), and its 55.712 GB total only reconciles with the f32 figure, so the file
    /// > contradicted itself. Inherited from `glimmer-architecture.md` section 7, which is
    /// > describing the checkpoint and is right about it.
    ///
    /// Split out of [`Self::resident_bytes`] for `GlimmerPin`'s budget arithmetic: how many
    /// layers a budget pins is `(budget - globals - slots·this) / this`, and a second spelling
    /// of the per-layer size would be a number that could disagree with the one that sized the
    /// tier. Every layer is identical — Glimmer has no dense/MoE split — so this is exact
    /// rather than an average, and that is what makes a static partition expressible at all.
    pub fn layer_bytes(&self) -> Result<usize> {
        let mut per_layer = 0usize;
        for t in GLIMMER_LAYER_TENSORS {
            let shape = self.layer_tensor_shape(t)?;
            let n: usize = shape.iter().product();
            per_layer += n * if shape.len() == 1 { 4 } else { 2 };
        }
        Ok(per_layer)
    }

    /// Bytes the model-level tensors occupy — 5.380 GB for the shipped model.
    ///
    /// `embed_tokens` and `lm_head`, both `[vocab, hidden]` bf16 and both shipped —
    /// `tie_word_embeddings` is false, so this is 2x2.690 GB and not one of them — plus the
    /// final norm at f32.
    ///
    /// **These are unconditionally resident at every budget** and that is an arithmetic
    /// decision, not a convenience: they are read once per TOKEN each (5.380 GB against a
    /// layer's 0.968), so streaming them would buy 5.4 GB of residency and pay for it on
    /// every token, while pinning them costs 9.7% of the model. `GlimmerPin`'s floor
    /// includes them, so a budget that cannot hold them is refused rather than partitioned.
    pub fn global_bytes(&self) -> usize {
        2 * self.vocab * self.hidden * 2 + self.hidden * 4
    }

    /// The smallest budget this artifact can run under: model-level tensors, plus
    /// [`GLIMMER_STREAM_SLOTS`] layer slots, plus alignment.
    ///
    /// **This is a floor on WEIGHTS only, and saying so is the point.** KV at the configured
    /// context, activation scratch and the DFlash drafter are not here — they are not the pin's
    /// to size — so a budget clearing this floor can still be too small to decode. What it
    /// guarantees is that the failure happens at load, with a number, rather than at layer 40
    /// of the first token.
    pub fn floor_bytes(&self) -> Result<usize> {
        Ok(self.global_bytes() + GLIMMER_STREAM_SLOTS * self.layer_bytes()? + GLIMMER_PIN_SLACK)
    }

    /// How a device budget divides: `(layers pinned, tier bytes to request)`.
    ///
    /// `None` — and any budget at or above the whole model — pins every layer and allocates no
    /// slots, so an over-generous budget does not reserve slots nothing would ever fill. Below
    /// that, the model-level tensors and the slots come first and whole layers fill the rest in
    /// ascending order.
    ///
    /// **The partition is a fixed PREFIX, and that is the optimal policy rather than a
    /// simplification.** A dense model reads its layers in fixed cyclic order, which is LRU's
    /// pathological case: at any deficit LRU evicts exactly the layer needed next and the hit
    /// rate is **0**, not `pinned/n_layers`. Belady on a cyclic scan — evict the block whose
    /// next use is farthest, i.e. the one just used — degenerates to holding a fixed subset,
    /// and every fixed subset of size `k` has the same hit rate `k/n`. So the whole
    /// `--cache-policy` axis collapses to one answer here, which is why `run_glimmer` still
    /// refuses that flag while accepting `--max-mem`.
    ///
    /// **Here rather than on `GlimmerPin` because it is arithmetic over a config and touches no
    /// device.** `memory::pin` is `#[cfg(feature = "rocm")]`, so a `GlimmerPin::partition` was
    /// unreachable from a featureless build — and CI has no rocm job, so the gate that claimed
    /// to cover this could not run there at all. Found by trying it, 2026-08-12.
    pub fn partition(&self, budget: Option<usize>) -> Result<(usize, usize)> {
        let want = self.resident_bytes()?;
        let layer = self.layer_bytes()?;
        let Some(b) = budget.filter(|b| *b < want + GLIMMER_PIN_SLACK) else {
            return Ok((self.n_layers, want + GLIMMER_PIN_SLACK));
        };
        let floor = self.floor_bytes()?;
        ensure!(
            b >= floor,
            "a device budget of {:.3} GB is below this artifact's floor of {:.3} GB: the \
             model-level tensors are {:.3} GB (embed + lm_head + final norm, each read once per \
             TOKEN, so streaming them would cost more than it frees) and {GLIMMER_STREAM_SLOTS} \
             streaming slots are {:.3} GB. Weights only — KV at your context, activation scratch \
             and any drafter are on top of this",
            b as f64 / 1e9,
            floor as f64 / 1e9,
            self.global_bytes() as f64 / 1e9,
            (GLIMMER_STREAM_SLOTS * layer) as f64 / 1e9,
        );
        // Integer division, so the remainder stays unspent rather than pinning a layer the
        // budget cannot hold. `pinned < n_layers` ALWAYS here: pinning every layer needs
        // `floor + n_layers·layer` = `want + SLACK + SLOTS·layer`, which the filter above already
        // sent to the early return.
        let pinned = (b - floor) / layer;
        // **Ask for what the partition USES, never for the whole budget.** `DeviceTier::new`
        // allocates its capacity rather than treating it as a ceiling, AND feeds
        // `guard_capacity`, so an over-request both wastes GTT and can turn a workable budget
        // into a refusal. The waste is up to a whole layer — 967.942 MB on the shipped model,
        // at `b = want + SLACK - 1`, which pins 49 of 52 and uses `global + 51·layer`.
        //
        // > **CORRECTED 2026-08-12 by two independent reviews, and the first version of this
        // > comment described a bug that could not happen.** It read "a budget that turns out to
        // > pin every layer allocates no slots — so returning `b` there would reserve 2 x
        // > 967.889 MB", and guarded it with `if pinned == self.n_layers`. That branch is
        // > UNREACHABLE by the arithmetic above, so the guard was dead and the real
        // > over-allocation — this one, on the streaming path — went unfixed. The commit message
        // > for R1 claims the dead version as a fix; it was not one.
        let used =
            self.global_bytes() + (pinned + GLIMMER_STREAM_SLOTS) * layer + GLIMMER_PIN_SLACK;
        Ok((pinned, used))
    }

    /// Cross-field checks. Each guards a failure that produces text rather than an error.
    fn validate(&self) -> Result<()> {
        // The descent check. `parse_config` matched the WRAPPER's pair; this is the nested
        // spelling, and it is what separates "descended into text_config" from "descended into
        // vision_config" — a sibling dict that carries its own `hidden_size`,
        // `num_attention_heads`, `layer_types` and `rope_parameters`, and would therefore
        // satisfy a good fraction of the schema above before failing on the rest.
        ensure!(
            self.model_type == "muse_glimmer_text",
            "text_config declares model_type {:?} — a Muse Glimmer wrapper's text model is \
             \"muse_glimmer_text\". Either this is not the dict we think we descended into, or \
             (the case worth naming) it is `vision_config`, which carries several of the same \
             keys",
            self.model_type
        );
        // A zero width passes every divisibility check below and then sizes a projection, a
        // KV row or a GEMV `dim` to nothing.
        for (what, dim) in [
            ("hidden_size", self.hidden),
            ("intermediate_size", self.inter),
            ("vocab_size", self.vocab),
            ("num_attention_heads", self.n_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("num_hidden_layers", self.n_layers),
            ("sliding_window", self.sliding_window),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            ensure!(dim > 0, "{what} is 0");
        }
        // GQA, and the direction of the broadcast is the trap. 32 query heads share 2 KV
        // heads, so query head j reads KV head `j / 16` — NOT `j % 2`. Both mappings
        // type-check, both decode fluently, and only one is this model. The divisibility is
        // what makes `j / groups` well-defined at all.
        ensure!(
            // No `kv <= n_heads` conjunct: the zero loop above guarantees both are positive,
            // and for `0 < n_heads < kv` the multiple test is already false. A conjunct that
            // can never be the sole cause of a refusal makes the message ambiguous for free.
            self.n_heads.is_multiple_of(self.num_key_value_heads),
            "num_attention_heads {} is not a positive multiple of num_key_value_heads {} — \
             GQA needs a whole number of query heads per KV head",
            self.n_heads,
            self.num_key_value_heads
        );
        // **The two per-layer arrays must both be exactly n_layers long**, and this is the
        // load-bearing length check of the whole schema: everything downstream indexes them by
        // layer id. A short array is an out-of-bounds panic at best; a LONG one is worse,
        // because the extra entries are silently ignored and the file that was meant to
        // describe a different model parses cleanly.
        for (what, got) in [
            ("layer_types", self.layer_types.len()),
            ("layer_rope_theta", self.layer_rope_theta.len()),
        ] {
            ensure!(
                got == self.n_layers,
                "{what} has {got} entries but num_hidden_layers is {} — this array is indexed \
                 by layer id and is the only statement of which layers slide and which rotate",
                self.n_layers
            );
        }
        // The pairing invariant, and it is the reason both arrays are carried rather than one.
        // In this checkpoint a layer is sliding IFF it is rotated: `layer_rope_theta[i] == 0`
        // exactly on the `full_attention` layers. The two arrays are independent in the file,
        // so they CAN disagree — and a disagreement is not a shape error anywhere downstream.
        // It is a model that attends over the wrong rows or rotates a layer that must not be
        // rotated, and either one is fluent.
        //
        // This is the strongest statement the config alone can make, so it is made here rather
        // than left to a fixture. If a future Glimmer ships a rotated full layer, this refuses
        // it — correctly, because this port's attention would not implement it.
        // `zip` + `enumerate` rather than `0..n_layers` and two index expressions: the indices
        // are in bounds only by the statement order of the length check above, and nothing in
        // `Cargo.toml`'s lint table denies `indexing_slicing`. Iterating the pair is total, so
        // reordering this function cannot turn it into a panic.
        //
        // An unknown layer kind needs no check here — `LayerKind` refuses it at deserialize
        // time, which is earlier and not skippable.
        for (i, (kind, &theta)) in self
            .layer_types
            .iter()
            .zip(self.layer_rope_theta.iter())
            .enumerate()
        {
            let sliding = *kind == LayerKind::SlidingAttention;
            ensure!(
                sliding == (theta != 0.0),
                "layer {i} is {kind:?} with layer_rope_theta {theta} — in this architecture a \
                 layer is rotated IFF it slides. The arrays disagreeing is not a shape error \
                 downstream: it is a layer attending over the wrong rows, or rotated when it \
                 must not be, and both produce fluent text"
            );
            // Every rotated layer must share the one base the table is built from. The
            // first-party code builds a SINGLE cos/sin table from `rope_parameters.rope_theta`
            // and selects per layer, so a per-layer base that differed from it would be
            // silently ignored rather than honoured.
            ensure!(
                !sliding || theta == self.rope_parameters.rope_theta,
                "layer {i} asks for rope theta {theta} but rope_parameters.rope_theta is {} — \
                 one table is built for the whole model, so a differing per-layer base would \
                 be read and then ignored",
                self.rope_parameters.rope_theta
            );
        }
        ensure!(
            self.rope_parameters.rope_type == "default",
            "rope_parameters.rope_type is {:?}, not \"default\" — this port builds an \
             unscaled table, and an unimplemented scaling scheme keeps every frequency \
             plausible and the text fluent (the V4 port's `Defect::RopeNoYarn`)",
            self.rope_parameters.rope_type
        );
        ensure!(
            !self.tie_word_embeddings,
            "tie_word_embeddings is true — this port reads `lm_head.weight` and \
             `embed_tokens.weight` as two separate 2.690 GB tensors, and the shipped \
             checkpoint declares them untied"
        );
        ensure!(
            !self.attention_bias,
            "attention_bias is true — no projection in this port's attention block reads a \
             bias tensor, and none ships in the checkpoint"
        );
        // SwiGLU with a different activation changes the arithmetic without changing one
        // shape, so nothing downstream would refuse it.
        ensure!(
            self.hidden_activation == "silu",
            "hidden_activation is {:?}, not \"silu\" — the MLP is SwiGLU and the gate's \
             activation is not a shape",
            self.hidden_activation
        );
        // Narrowed to f32, the domain the kernels work in, for the reason K3's block states:
        // an f64 positivity test passes values that reach every kernel as 0.0f32.
        //
        // `output_multiplier` and `final_logit_softcapping` are here even though both are
        // argmax-invariant (see the field docs). That is exactly why they need a load-boundary
        // check: no greedy gate downstream can see them being wrong.
        ensure_f32_positive(&[
            ("rms_norm_eps", self.rms_norm_eps),
            ("post_norm_eps", self.post_norm_eps),
            ("qk_scale_factor", self.qk_scale_factor),
            ("output_multiplier", self.output_multiplier),
            ("final_logit_softcapping", self.final_logit_softcapping),
            (
                "rope_parameters.rope_theta",
                self.rope_parameters.rope_theta,
            ),
        ])
    }
}

/// The load boundary is the only place a foreign snapshot is inspected before its
/// dimensions reach a kernel, so every refusal here is one that would otherwise be an
/// out-of-bounds panic, a silently truncated expert row, or wrong experts with no crash.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    /// `base` as a JSON object with `overrides` applied — a present key is replaced, an
    /// empty value DELETES the key. Text rather than a struct literal because the serde
    /// renames ARE the contract under test.
    fn json_obj(base: &[(&str, &str)], overrides: &[(&str, &str)]) -> String {
        let mut fields: Vec<(String, String)> = base
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        for (k, v) in overrides {
            fields.retain(|(fk, _)| fk != k);
            if !v.is_empty() {
                fields.push((k.to_string(), v.to_string()));
            }
        }
        let body: Vec<String> = fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
        format!("{{{}}}", body.join(","))
    }

    /// DeepSeek-V4-Flash-0731's real `config.json` values, field for field. Pinned against
    /// the shipped file by `v4_base_matches_the_shipped_config` (skipped when absent),
    /// so this table cannot quietly become a fiction that only the unit tests believe.
    const V4_BASE: &[(&str, &str)] = &[
        ("model_type", r#""deepseek_v4""#),
        ("architectures", r#"["DeepseekV4ForCausalLM"]"#),
        ("num_hidden_layers", "43"),
        ("hidden_size", "4096"),
        ("vocab_size", "129280"),
        ("num_attention_heads", "64"),
        ("head_dim", "512"),
        ("qk_rope_head_dim", "64"),
        ("q_lora_rank", "1024"),
        ("o_groups", "8"),
        ("o_lora_rank", "1024"),
        ("num_key_value_heads", "1"),
        ("sliding_window", "128"),
        ("rms_norm_eps", "1e-06"),
        // 46 entries for 43 layers — the tail belongs to the mtp blocks. Layers 0 and 1
        // are ratio 0, then 4 and 128 alternate to layer 42.
        (
            "compress_ratios",
            "[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,\
             4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]",
        ),
        ("compress_rope_theta", "160000"),
        ("max_position_embeddings", "1048576"),
        (
            "rope_scaling",
            r#"{"beta_fast":32,"beta_slow":1,"factor":16,
                "original_max_position_embeddings":65536,"type":"yarn"}"#,
        ),
        ("rope_theta", "10000"),
        ("n_routed_experts", "256"),
        ("num_experts_per_tok", "6"),
        ("moe_intermediate_size", "2048"),
        ("n_shared_experts", "1"),
        ("routed_scaling_factor", "1.5"),
        ("scoring_func", r#""sqrtsoftplus""#),
        ("num_hash_layers", "3"),
        ("swiglu_limit", "10.0"),
        ("expert_dtype", r#""fp4""#),
        (
            "quantization_config",
            r#"{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8",
                "scale_fmt":"ue8m0","weight_block_size":[128,128]}"#,
        ),
        ("index_n_heads", "64"),
        ("index_head_dim", "128"),
        ("index_topk", "512"),
        ("hc_mult", "4"),
        ("hc_sinkhorn_iters", "20"),
        ("hc_eps", "1e-06"),
    ];

    fn v4_json(overrides: &[(&str, &str)]) -> String {
        json_obj(V4_BASE, overrides)
    }

    fn parse_v4(overrides: &[(&str, &str)]) -> Result<V4Config> {
        parse_config(&v4_json(overrides))
    }

    /// GLM-5.2's real values, as a JSON object that `overrides` can patch.
    fn cfg_json(overrides: &[(&str, &str)]) -> String {
        let base: Vec<(&str, &str)> = [
            ("model_type", r#""glm_moe_dsa""#),
            ("architectures", r#"["GlmMoeDsaForCausalLM"]"#),
            ("num_hidden_layers", "78"),
            ("hidden_size", "6144"),
            ("num_attention_heads", "64"),
            ("q_lora_rank", "2048"),
            ("kv_lora_rank", "512"),
            ("qk_rope_head_dim", "64"),
            ("qk_nope_head_dim", "192"),
            ("v_head_dim", "256"),
            ("n_routed_experts", "256"),
            ("num_experts_per_tok", "8"),
            ("moe_intermediate_size", "2048"),
            ("intermediate_size", "12288"),
            ("n_shared_experts", "1"),
            ("first_k_dense_replace", "3"),
            ("routed_scaling_factor", "2.5"),
            ("vocab_size", "154880"),
            ("rms_norm_eps", "1e-5"),
            ("rope_parameters", r#"{"rope_theta": 8000000}"#),
        ]
        .into();
        json_obj(&base, overrides)
    }

    fn parse(overrides: &[(&str, &str)]) -> Result<ModelConfig> {
        parse_config(&cfg_json(overrides))
    }

    #[test]
    fn glm_baseline_parses() {
        let c = parse(&[]).expect("GLM-5.2's own config must load");
        assert_eq!(c.rope_theta(), 8_000_000.0);
        assert_eq!(c.scoring(), crate::math::Scoring::Sigmoid);
        assert_eq!(c.experts_per_layer(), 9);
        assert_eq!(c.qk_head_dim(), 256);
    }

    /// GLM nests theta; the DeepSeek/Llama lineage puts it at top level. Both must
    /// load, and a snapshot carrying neither must say so instead of decoding at
    /// theta=0 — which would be a silently wrong RoPE, not a crash.
    #[test]
    fn rope_theta_from_either_nesting() {
        let flat = parse(&[("rope_parameters", ""), ("rope_theta", "10000")])
            .expect("top-level rope_theta must load");
        assert_eq!(flat.rope_theta(), 10_000.0);

        let err = parse(&[("rope_parameters", "")]).unwrap_err().to_string();
        assert!(err.contains("rope_theta"), "unhelpful message: {err}");
    }

    /// A wrong router affinity picks plausible-but-wrong experts and never crashes,
    /// so an unrecognised one must be refused at load. `softmax` is refused on
    /// purpose: the reference skips top-k renormalization for it.
    #[test]
    fn scoring_func_is_resolved_or_refused() {
        // Table-driven, so the message has to name WHICH spelling failed.
        for (raw, want) in [
            (r#""sqrtsoftplus""#, crate::math::Scoring::SqrtSoftplus),
            (r#""sigmoid""#, crate::math::Scoring::Sigmoid),
        ] {
            assert_eq!(
                parse(&[("scoring_func", raw)]).unwrap().scoring(),
                want,
                "scoring_func {raw} resolved wrong"
            );
        }
        for bad in [r#""softmax""#, r#""gumbel""#] {
            let err = parse(&[("scoring_func", bad)]).unwrap_err().to_string();
            assert!(err.contains("scoring_func"), "unhelpful message: {err}");
        }
    }

    /// `vq_row_bytes`/`vq_groups` only `debug_assert` this, so a release build would
    /// truncate every expert row with no diagnostic at all.
    /// Note the head count on the `hidden` case: at GLM's 64 heads the existing
    /// `hidden % n_heads` check already implies `% VQ_GROUP`, since both are 64. It is
    /// `moe_inter` — checked against nothing else — that this actually protects.
    #[test]
    fn ragged_vq_widths_are_refused() {
        for bad in [
            vec![("num_attention_heads", "8"), ("hidden_size", "6120")],
            vec![("moe_intermediate_size", "2000")],
        ] {
            let err = parse(&bad).unwrap_err().to_string();
            assert!(err.contains("VQ_GROUP"), "unhelpful message: {err}");
        }
        // 18432 is the DeepSeek-V3 family's intermediate_size, which the deleted
        // `MAX_FUSED_INTER` ceiling used to refuse for a kernel that no longer exists.
        parse(&[("intermediate_size", "18432")]).expect("no intermediate ceiling any more");
    }

    #[test]
    fn structurally_impossible_dims_are_refused() {
        for bad in [
            vec![("first_k_dense_replace", "99")],
            vec![("num_experts_per_tok", "300")],
            vec![("n_shared_experts", "0")],
            vec![("num_attention_heads", "63")],
        ] {
            assert!(parse(&bad).is_err(), "{bad:?} should not have loaded");
        }
    }

    // ── the architecture discriminant ──────────────────────────────────────────────

    /// The discriminant is read from `model_type` AND `architectures`, is never defaulted,
    /// and refuses a self-contradicting file. Absent-from-both is a refusal too: every
    /// config and manifest under `/var/db/rivoli` and `/swarm/storage/ai/rivoli` carries
    /// both (checked 2026-08-04, all six), so there is no artifact for a fallback to serve
    /// — only a foreign one, which is exactly what must not be guessed at.
    #[test]
    fn arch_is_declared_never_defaulted() {
        assert_eq!(
            arch_of(&serde_json::json!({"model_type": "glm_moe_dsa"})).unwrap(),
            Arch::GlmMoeDsa
        );
        // `architectures` alone is enough — that is the field the coordinator's `--help`
        // rendering keys on, and GLM's shipped manifest carries it.
        assert_eq!(
            arch_of(&serde_json::json!({"architectures": ["DeepseekV4ForCausalLM"]})).unwrap(),
            Arch::DeepseekV4
        );
        for (doc, want) in [
            (serde_json::json!({}), "neither"),
            (serde_json::json!({"model_type": "llama"}), "unsupported"),
            (
                serde_json::json!({"model_type":"glm_moe_dsa",
                                   "architectures":["DeepseekV4ForCausalLM"]}),
                "disagrees",
            ),
        ] {
            let err = format!("{:#}", arch_of(&doc).unwrap_err());
            assert!(err.contains(want), "expected {want:?} in: {err}");
        }
    }

    /// The heart of S1a. A V4 config must not become a zero-filled `ModelConfig`, and the
    /// refusal must NAME the architecture rather than blaming whichever MLA field V4
    /// happens to omit first (it used to say `missing field kv_lora_rank`, which reads
    /// like a corrupt checkpoint). Both directions, because a GLM config fed to the V4
    /// converter is the same defect wearing the other hat.
    #[test]
    fn each_config_refuses_the_other_architecture() {
        let err = format!(
            "{:#}",
            parse_config::<ModelConfig>(&v4_json(&[])).unwrap_err()
        );
        assert!(
            err.contains("deepseek_v4") && err.contains("GlmMoeDsa"),
            "refusal must name what the file says AND what was expected: {err}"
        );
        assert!(
            !err.contains("kv_lora_rank"),
            "refused on a missing field instead of on the architecture: {err}"
        );
        let err = format!(
            "{:#}",
            parse_config::<V4Config>(&cfg_json(&[])).unwrap_err()
        );
        assert!(
            err.contains("glm_moe_dsa") && err.contains("DeepseekV4"),
            "{err}"
        );
    }

    // ── V4Config ───────────────────────────────────────────────────────────────────

    #[test]
    fn v4_baseline_parses() {
        let c = parse_v4(&[]).expect("V4-Flash's own config must load");
        assert_eq!((c.n_layers, c.hidden, c.moe_inter), (43, 4096, 2048));
        assert_eq!((c.top_k, c.n_shared), (6, 1));
        // 1.5, NOT `ModelArgs.route_scale`'s default of 1. — see the field's doc.
        assert_eq!(c.routed_scale, 1.5);
        assert_eq!(c.index_topk, 512); // parsed, not defaulted — see the field's doc

        assert_eq!(c.n_heads * c.head_dim, 32768); // == wq_b's out_dim
    }

    /// **No TOP-LEVEL field of V4Config may be defaulted.** Dropping any one must fail the parse,
    /// not yield a zero — that is the failure mode this whole stage is built around, and
    /// it is checked field-by-field rather than by inspection so a `#[serde(default)]`
    /// added later cannot slip through.
    ///
    /// Driven off `V4_BASE` itself, so a field added to the struct and to the fixture is
    /// covered automatically; one added to the struct alone fails `v4_baseline_parses`.
    ///
    /// **Top-level only, and the doc above says so for that reason.** `V4_BASE`'s rows are
    /// whole JSON values, so dropping `rope_scaling` drops the table entire — the five
    /// fields of [`RopeScaling`] and the three of `QuantConfig` are never individually
    /// checked for requiredness. A `#[serde(default)]` on one of those would pass this
    /// test. Closing it means a second loop over the nested tables, which is worth doing
    /// the next time either grows a field.
    #[test]
    fn every_v4_field_is_required() {
        for (k, _) in V4_BASE {
            // The two discriminant keys are interchangeable — dropping ONE leaves the
            // other, which is the point of reading both. Dropping both is covered by
            // `arch_is_declared_never_defaulted`.
            if matches!(*k, "model_type" | "architectures") {
                continue;
            }
            // **On the serde error, not on `is_err()`.** A bare `is_err()` cannot tell
            // "the field is required" from "the field is defaulted and `validate()`
            // happens to reject the default" — and that is not hypothetical: `index_topk`
            // defaults to 0 under `#[serde(default)]`, and `validate()` refuses 0, so an
            // injected `#[serde(default)]` on it passed this test unchanged. Requiring the
            // MISSING-FIELD error separates the two.
            let err = format!("{:#}", parse_v4(&[(k, "")]).unwrap_err());
            // Naming the field too, not just "missing field": serde reports the JSON name,
            // so this also proves the error is about the key that was dropped rather than
            // some other one the fixture happens to be short of.
            assert!(
                err.contains(&format!("missing field `{k}`")),
                "dropping {k:?} did not fail as a missing field — it has acquired a \
                 default, and something downstream rejected that default instead. Got: {err}"
            );
        }
    }

    /// Each of these is silent-wrong if it slips through: a wrong router affinity picks
    /// plausible experts, an unclamped SwiGLU changes arithmetic, an fp8 export read as
    /// FP4 nibble pairs is noise, and a non-128 fp8 block mis-tiles every attention scale.
    #[test]
    fn v4_rejects_the_silently_wrong_settings() {
        // One YaRN block, varied — the two bad cases differ only in the field under test,
        // so neither can pass because of the other's value.
        let rope = |orig: usize, kind: &str| {
            format!(
                r#"{{"beta_fast":32,"beta_slow":1,"factor":16,
                    "original_max_position_embeddings":{orig},"type":"{kind}"}}"#
            )
        };
        let (no_yarn, not_yarn) = (rope(0, "yarn"), rope(65536, "linear"));
        for (bad, want) in [
            (vec![("scoring_func", r#""sigmoid""#)], "sqrtsoftplus"),
            (vec![("swiglu_limit", "0.0")], "CLAMPED"),
            // The two f64 -> f32 NARROWING failures, which a bare `> 0.0` on the f64 admits.
            // Both are device-free here and were not testable where this check first lived
            // (`F4Engine::new`, which allocates device buffers and needs a real pin).
            // `1e-46` underflows to `0.0f32`: LOUD, guard 1006 at the first MoE layer.
            (vec![("swiglu_limit", "1e-46")], "narrows to 0"),
            // `1e39` saturates to `f32::INFINITY`, passes every `> 0.0`, and makes
            // `fminf(gt, inf)` a no-op — SILENT `Defect::SwigluUnclamped`. This row is the one
            // that matters; the check was one-sided until a review found it.
            (vec![("swiglu_limit", "1e39")], "narrows to inf"),
            (vec![("expert_dtype", r#""fp8""#)], "e2m1"),
            (vec![("num_key_value_heads", "8")], "MQA"),
            (vec![("rope_theta", "0")], "must be positive"),
            (vec![("compress_rope_theta", "0")], "must be positive"),
            (vec![("rope_scaling", &no_yarn)], "interpolation branch"),
            (vec![("rope_scaling", &not_yarn)], "only YaRN"),
            (vec![("n_shared_experts", "2")], "n_shared_experts"),
            (vec![("num_experts_per_tok", "300")], "top_k"),
            (vec![("num_hash_layers", "99")], "num_hash_layers"),
            (vec![("o_groups", "7")], "o_groups"),
            // PRESENT but zero, like `index_topk` below: dropping either key fails as a
            // MISSING FIELD and never reaches the bound, so without these two rows both
            // `ensure!`s are unproved-reachable — the false-green shape this block's
            // `index_topk` note already calls out.
            (vec![("hc_sinkhorn_iters", "0")], "hc_sinkhorn_iters is 0"),
            (vec![("hc_eps", "0.0")], "must be positive"),
            (vec![("moe_intermediate_size", "2000")], "F4_GROUP"),
            // Full length, one bad entry — otherwise the LENGTH check fires first and this
            // case would pass without the value check existing at all.
            (
                vec![(
                    "compress_ratios",
                    "[0,0,7,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,\
                     4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]",
                )],
                "implemented: 0, 4, 128",
            ),
            (vec![("compress_ratios", "[0,0,4]")], "at least n_layers"),
            // PRESENT but zero — dropping the key fails as a MISSING FIELD and never
            // reaches this bound.
            (vec![("index_topk", "0")], "index_topk must be positive"),
            (
                vec![(
                    "quantization_config",
                    r#"{"fmt":"e5m2","scale_fmt":"ue8m0","weight_block_size":[128,128]}"#,
                )],
                "e4m3",
            ),
            (
                vec![(
                    "quantization_config",
                    r#"{"fmt":"e4m3","scale_fmt":"ue8m0","weight_block_size":[64,64]}"#,
                )],
                "weight_block_size",
            ),
        ] {
            let err = format!("{:#}", parse_v4(&bad).unwrap_err());
            assert!(
                err.contains(want),
                "expected {want:?} for {bad:?}, got: {err}"
            );
        }
    }

    /// The per-layer roles the converter and S1b both key on. Layer 0 is the one everyone
    /// checks and it is the LEAST representative layer in the model: ratio 0 means no
    /// compressor, no indexer, base theta and YaRN off. Layers 2 and 3 are the other two
    /// shapes. Cross-checked against the shipped checkpoint's tensor sets in
    /// `tests/v4_artifact.rs`.
    #[test]
    fn v4_layer_roles_follow_compress_ratios() {
        let c = parse_v4(&[]).unwrap();
        let role = |l: usize| {
            (
                c.compress_ratio(l).unwrap(),
                c.layer_has_compressor(l).unwrap(),
                c.layer_has_indexer(l).unwrap(),
                c.layer_routes_by_hash(l),
            )
        };
        assert_eq!(role(0), (0, false, false, true));
        assert_eq!(role(1), (0, false, false, true));
        assert_eq!(role(2), (4, true, true, true)); // last hash layer, first indexer
        assert_eq!(role(3), (128, true, false, false));
        assert_eq!(role(42), (4, true, true, false));
        // 21 indexers among 41 compressor layers — the count `other-models.md` quotes.
        let (comp, idx) = (0..c.n_layers).fold((0, 0), |(a, b), l| {
            (
                a + usize::from(c.layer_has_compressor(l).unwrap()),
                b + usize::from(c.layer_has_indexer(l).unwrap()),
            )
        });
        assert_eq!((comp, idx), (41, 21));
        // The mtp tail is past n_layers and must stay unreachable: reading it would give
        // an mtp block's ratio for a main-path layer.
        assert!(c.compress_ratio(c.n_layers).is_err());
    }

    /// `V4_BASE` is a hand-copy of the shipped `config.json`. This is what stops it from
    /// quietly becoming a fiction that only the unit tests believe.
    ///
    /// Structural, not a hand-listed tuple: EVERY key in `V4_BASE` is compared against the
    /// real document, so the check grows with the table instead of covering whichever
    /// seven fields someone happened to list. A drifted `o_groups` or `index_head_dim`
    /// would otherwise be believed by every other V4 test in this file.
    #[test]
    fn v4_base_matches_the_shipped_config() {
        const DIR: &str = "/var/db/rivoli/deepseek-v4-flash-0731";
        let Ok(text) = std::fs::read_to_string(format!("{DIR}/config.json")) else {
            eprintln!("SKIP v4_base_matches: no checkpoint at {DIR} — V4_BASE is UNPINNED");
            return;
        };
        let real: serde_json::Value = serde_json::from_str(&text).unwrap();
        for (k, v) in V4_BASE {
            let want: serde_json::Value = serde_json::from_str(v)
                .unwrap_or_else(|e| panic!("V4_BASE[{k}] is not valid JSON: {e}"));
            let got = real
                .get(*k)
                .unwrap_or_else(|| panic!("config.json has no {k:?}"));
            assert_eq!(
                got, &want,
                "V4_BASE[{k}] has drifted from the shipped config"
            );
        }
        // …and it parses, and is still refused as a ModelConfig from the file itself
        // rather than only from the fixture.
        parse_config::<V4Config>(&text).expect("the shipped config must parse");
        assert!(ModelConfig::load(DIR).is_err());
    }

    // ── K3Config ───────────────────────────────────────────────────────────────────

    /// One-based `full_attn_layers`, as `docs/reference/k3-architecture.md` §2 quotes it from
    /// the shipped config. The every-fourth stride runs 4…92 — **23** entries — and then
    /// **93 is adjacent to 92**, off the stride. That one off-pattern entry is why this is
    /// transcribed rather than generated from a step-4 range.
    const K3_MLA_ONE_BASED: [usize; 24] = [
        4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92,
        93,
    ];

    /// `linear_attn_config` with `kda_layers` DERIVED as the complement, so the baseline
    /// fixture cannot fail `validate_layer_partition` for a transcription slip in 69 numbers.
    /// The partition check earns its keep against `k3_layer_partition_must_be_a_partition`,
    /// which injects broken arrays on purpose.
    fn k3_linear_attn(layers: usize) -> String {
        // Clipped to `layers` so a shrunk fixture is still a partition — the small-model
        // positive control at the end of the partition test depends on it.
        let mla: Vec<usize> = K3_MLA_ONE_BASED
            .iter()
            .copied()
            .filter(|&l| l <= layers)
            .collect();
        let kda: Vec<usize> = (1..=layers).filter(|l| !mla.contains(l)).collect();
        k3_lac(&mla, &kda)
    }

    /// A `linear_attn_config` dict from two explicit arrays, with the five scalars at their
    /// shipped values — `gate_lower_bound` NEGATIVE, which `LinearAttnConfig` explains.
    ///
    /// Separate from [`k3_linear_attn`] because the partition test needs arrays that are NOT a
    /// partition, which that function cannot produce by construction. Both go through here so a
    /// scalar added to the struct has one place to appear.
    fn k3_lac(mla: &[usize], kda: &[usize]) -> String {
        format!(
            r#"{{"full_attn_layers":{mla:?},"kda_layers":{kda:?},"num_heads":96,"head_dim":128,
                 "short_conv_kernel_size":4,"gate_lower_bound":-5.0,"use_full_rank_gate":true}}"#
        )
    }

    /// The vendored `config.json`, the real one — `moonshotai/Kimi-K3` at revision
    /// `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, fetched 2026-08-10, byte-for-byte.
    ///
    /// `include_str!` rather than a path read, so [`k3_base_matches_the_shipped_config`] runs
    /// **always** instead of skipping when a checkpoint is absent — V4's twin skips, and this
    /// port has no checkpoint on this machine at all. 7 KB of metadata is the cheapest gate in
    /// this file and it is the one that caught two wrong key spellings.
    const K3_SHIPPED: &str = include_str!("../../docs/measurement/k3-reference/config.json");

    /// Kimi-K3's `text_config`, field for field. **Pinned to [`K3_SHIPPED`]** by
    /// [`k3_base_matches_the_shipped_config`], so it cannot become a fiction only the unit tests
    /// believe.
    ///
    /// Until 2026-08-10 these were §1 of the plan's values rather than the file's, and the doc
    /// here said so. Reading the file corrected **two** of them, both of which would have made
    /// this schema refuse every real K3 checkpoint: `activation_linear_beta` is spelled
    /// `activation_situ_linear_beta`, and `use_full_rank_gate` lives inside `linear_attn_config`
    /// rather than beside it.
    const K3_TEXT: &[(&str, &str)] = &[
        // The NESTED pair, which differs from the wrapper's on purpose — see
        // `Arch::from_manifest_str`.
        ("model_type", r#""kimi_linear""#),
        ("architectures", r#"["KimiLinearForCausalLM"]"#),
        ("num_hidden_layers", "93"),
        ("hidden_size", "7168"),
        ("vocab_size", "163840"),
        ("num_attention_heads", "96"),
        ("rms_norm_eps", "1e-05"),
        ("dtype", r#""bfloat16""#),
        // A placeholder: `k3_json` always overrides it with `k3_linear_attn`'s derived pair.
        // It is listed here anyway so `every_k3_field_is_required` covers the key — an
        // override with an empty value deletes it, which is what that test does.
        ("linear_attn_config", "null"),
        ("q_lora_rank", "1536"),
        ("kv_lora_rank", "512"),
        ("qk_nope_head_dim", "128"),
        ("qk_rope_head_dim", "64"),
        ("v_head_dim", "128"),
        ("num_key_value_heads", "96"),
        ("mla_use_nope", "true"),
        ("mla_use_output_gate", "true"),
        ("attn_res_block_size", "12"),
        ("num_experts", "896"),
        ("num_experts_per_token", "16"),
        ("num_shared_experts", "2"),
        ("routed_expert_hidden_size", "3584"),
        ("moe_intermediate_size", "3072"),
        ("latent_moe_use_norm", "true"),
        ("moe_renormalize", "true"),
        ("num_expert_group", "1"),
        ("topk_group", "1"),
        ("topk_method", r#""noaux_tc""#),
        ("moe_router_activation_func", r#""sigmoid""#),
        ("routed_scaling_factor", "1.0"),
        ("moe_layer_freq", "1"),
        ("activation_situ_beta", "4.0"),
        ("activation_situ_linear_beta", "25.0"),
        ("first_k_dense_replace", "1"),
        ("intermediate_size", "33792"),
        ("hidden_act", r#""situ""#),
        ("num_nextn_predict_layers", "0"),
        ("tie_word_embeddings", "false"),
    ];

    /// The whole document: the multimodal wrapper (which is the level that names the
    /// architecture) around a `text_config` built from [`K3_TEXT`]. `overrides` patch the
    /// NESTED dict, which is where every field this port reads lives.
    fn k3_json(overrides: &[(&str, &str)]) -> String {
        let lac = k3_linear_attn(93);
        // Injected first so a caller's own `linear_attn_config` override — including a
        // deletion — still wins; `json_obj` applies them in order.
        let mut ov: Vec<(&str, &str)> = vec![("linear_attn_config", &lac)];
        ov.extend_from_slice(overrides);
        let text = json_obj(K3_TEXT, &ov);
        format!(
            r#"{{"model_type":"kimi_k3",
                 "architectures":["KimiK3ForConditionalGeneration"],
                 "text_config":{text}}}"#
        )
    }

    fn parse_k3(overrides: &[(&str, &str)]) -> Result<K3Config> {
        parse_config(&k3_json(overrides))
    }

    /// The refusal message for a config that must not parse. Three tests below want it, and
    /// spelling `format!("{:#}", …unwrap_err())` out in each was a real duplication-gate
    /// failure waiting to happen — `build.rs` runs jscpd at `--min-tokens 15` over `src/`.
    fn k3_err(overrides: &[(&str, &str)]) -> String {
        format!("{:#}", parse_k3(overrides).unwrap_err())
    }

    #[test]
    fn k3_baseline_parses() {
        let c = parse_k3(&[]).expect("Kimi-K3's own config must load").text;
        assert_eq!((c.n_layers, c.hidden, c.vocab), (93, 7168, 163840));
        assert_eq!((c.n_experts, c.top_k, c.n_shared), (896, 16, 2));
        // **The assumption break this whole stage exists for**: the routed experts are
        // entered at 3584, not at `hidden_size`. If these two are ever read as equal, every
        // expert row stride and every fp4 dot `dim` on the MoE path is wrong by 2x.
        assert_eq!((c.expert_in, c.moe_inter), (3584, 3072));
        assert_ne!(c.expert_in, c.hidden);
        assert_eq!((c.q_lora_rank, c.kv_lora_rank), (1536, 512));
        assert_eq!((c.first_k_dense_replace, c.dense_inter), (1, 33792));

        // The layer map, zero-based, including the two ADJACENT ones at the end that the
        // every-fourth stride does not predict.
        let mla: Vec<usize> = (0..c.n_layers)
            .filter(|&l| c.layer_is_mla(l).unwrap())
            .collect();
        assert_eq!(mla.len(), 24);
        assert_eq!(&mla[..3], &[3, 7, 11]);
        assert_eq!(&mla[22..], &[91, 92]);
        // Layer 0 is simultaneously KDA, dense, and an AttnRes boundary — the least
        // representative layer in the model and the one everyone tests first.
        assert!(!c.layer_is_mla(0).unwrap() && c.layer_is_dense(0));
        assert!(!c.layer_is_dense(1));
        assert!(
            c.layer_is_mla(c.n_layers).is_err(),
            "no bound on the layer id"
        );
    }

    /// **No field of `text_config` may be defaulted**, checked field-by-field for the reason
    /// [`every_v4_field_is_required`] states: `is_err()` alone cannot tell "required" from
    /// "defaulted to something `validate` happens to reject".
    ///
    /// Unlike V4's, this loop skips nothing. The discriminant that `parse_config` reads lives
    /// on the WRAPPER, so the nested `model_type`/`architectures` are ordinary required fields
    /// here — dropping either must fail as a missing field, not fall back to the wrapper's.
    ///
    /// **`linear_attn_config`'s two arrays are covered too**, by the second loop. `K3_TEXT`'s
    /// rows are whole JSON values, so dropping that key drops the dict entire — the limitation
    /// V4's twin documents and accepts. An earlier draft of this doc claimed the two arrays'
    /// requiredness "rides on `k3_layer_partition_must_be_a_partition`"; **it did not.** Every
    /// case in that test emits both keys, so a `#[serde(default)]` on `kda_layers` would have
    /// left the whole suite green while this comment asserted otherwise. Caught by review
    /// 2026-08-10, and the fix was to make the claim true rather than to soften it.
    #[test]
    fn every_k3_field_is_required() {
        for (k, _) in K3_TEXT {
            let err = k3_err(&[(k, "")]);
            assert!(
                err.contains(&format!("missing field `{k}`")),
                "{k:?} has a default: dropping it was rejected by something other than \
                 serde, which is the shape that hides a zeroed dimension. Got: {err}"
            );
        }
        // One level down: **all SEVEN fields of `linear_attn_config`**, each dropped from an
        // otherwise complete dict.
        //
        // The first version of this loop dropped only the two arrays and supplied nothing else, so
        // it reported `missing field \`kda_layers\`` merely because serde names the FIRST missing
        // field in declaration order — the five scalars were absent from every arm and untested,
        // and reordering the struct would have made it pass for the wrong reason. Review
        // 2026-08-11. Building each arm by deletion from the full dict is what makes the failure
        // attributable to the named key.
        let full = k3_linear_attn(93);
        for k in [
            "full_attn_layers",
            "kda_layers",
            "num_heads",
            "head_dim",
            "short_conv_kernel_size",
            "gate_lower_bound",
            "use_full_rank_gate",
        ] {
            let doc: serde_json::Value = serde_json::from_str(&full).unwrap();
            let mut obj = doc.as_object().unwrap().clone();
            assert!(obj.remove(k).is_some(), "{k:?} is not in the fixture dict");
            let one_short = serde_json::to_string(&obj).unwrap();
            let err = k3_err(&[("linear_attn_config", &one_short)]);
            assert!(
                err.contains(&format!("missing field `{k}`")),
                "{k:?} inside linear_attn_config has a default. Got: {err}"
            );
        }
    }

    /// Every setting whose wrong value is arithmetic rather than a shape: a false gate, a
    /// routing method that picks plausible-but-wrong experts, a tied head reading the
    /// embedding table. None of these would fail a length check anywhere downstream.
    ///
    /// Single-key overrides, and each one PRESENT-but-wrong rather than absent — a dropped key
    /// fails as a missing field and never reaches the `ensure!`, which is the false-green
    /// shape `every_v4_field_is_required`'s notes call out.
    ///
    /// **Each `want` names the FIELD, not just the check.** Reviewed 2026-08-10: two rows
    /// shared `"F4_GROUP"`, and with that substring, transposing `ensure_f4_group_aligned`'s two
    /// arguments left both rows green while every refusal named the wrong config key. A `want`
    /// that only proves *some* check fired is not a test of the row it sits on.
    #[test]
    fn k3_rejects_the_silently_wrong_settings() {
        for (key, bad, want) in [
            // The nested-descent pair. Each row names its own conjunct's VALUE, so neither can
            // pass on the other's — the two share one `ensure!`.
            ("model_type", r#""deepseek_v4""#, r#""deepseek_v4" / "#),
            (
                "architectures",
                r#"["LlamaForCausalLM"]"#,
                "LlamaForCausalLM",
            ),
            ("mla_use_nope", "false", "mla_use_nope is false"),
            // §3e's secondary reading — the only field that must be ABSENT, so this row ADDS a
            // key the fixture does not carry rather than replacing one.
            ("rope_theta", "10000.0", "carries rope_theta"),
            ("mla_use_output_gate", "false", "output_gate is false"),
            // `use_full_rank_gate` is NOT here — it lives inside `linear_attn_config`, so an
            // override at this level is an unknown key and silently ignored. It is covered at the
            // bottom of this test, where the nesting can be patched.
            ("latent_moe_use_norm", "false", "use_norm is false"),
            ("moe_renormalize", "false", "moe_renormalize is false"),
            ("topk_method", r#""greedy""#, "noaux_tc"),
            // Same treatment: one `ensure!` covers both, so each row asserts on its own value.
            ("num_expert_group", "8", "num_expert_group 8"),
            ("topk_group", "4", "topk_group 4"),
            ("num_nextn_predict_layers", "1", "no MTP head"),
            ("tie_word_embeddings", "true", "separate lm_head"),
            ("num_experts_per_token", "900", "not in 1..="),
            ("num_experts_per_token", "0", "not in 1..="),
            ("num_shared_experts", "0", "always-on MLP"),
            // The two halves of one bound, each needle matching only its own — which is what the
            // split into two `ensure!`s buys and what a conjunction could not prove.
            ("first_k_dense_replace", "93", "every layer would be dense"),
            ("first_k_dense_replace", "0", "so layer 0 would run the"),
            // The three f64 -> f32 narrowing failures. `1e-46` underflows to `0.0f32`; `1e39`
            // saturates to infinity, which passes any bare `> 0.0` test — that is the silent
            // direction and the reason the check is `is_finite()` rather than a bound.
            // `rms_norm_eps` is in this set because `gpu.rs:1743` and `f4gpu.rs:325` both do
            // `cfg.rms_norm_eps as f32`; it was checked in f64 alone until review.
            ("rms_norm_eps", "1e-46", "narrows to 0"),
            ("activation_situ_beta", "1e-46", "narrows to 0"),
            // The JSON key, `activation_situ_linear_beta` — not the struct field's name. This row
            // read `activation_linear_beta` until the shipped config was fetched, which made it
            // an unknown key: silently ignored, baseline parse, `unwrap_err()` panic. A test that
            // patches by JSON key catches a wrong `rename`; one that patches by field name cannot.
            ("activation_situ_linear_beta", "1e39", "narrows to inf"),
            // Group alignment, on the LATENT width and on `moe_inter`. 3600 % 32 == 16 and
            // 3000 % 32 == 24 — neither is a multiple of `F4_GROUP`, and each `want` names its
            // own key so a transposed argument pair cannot pass both.
            (
                "routed_expert_hidden_size",
                "3600",
                "is 3600, not a multiple",
            ),
            (
                "moe_intermediate_size",
                "3000",
                "moe_intermediate_size is 3000",
            ),
            // Zero passes every divisibility check (0 is a multiple of anything) and then
            // sizes a row, a stride and a GEMV `dim` to nothing. One row per entry of
            // `validate`'s width table, so deleting any entry reddens exactly one row —
            // three of the ten were sampled until review pointed out the rest were free.
            ("hidden_size", "0", "hidden_size is 0"),
            ("routed_expert_hidden_size", "0", "expert_hidden_size is 0"),
            ("moe_intermediate_size", "0", "moe_intermediate_size is 0"),
            ("intermediate_size", "0", "intermediate_size is 0"),
            ("vocab_size", "0", "vocab_size is 0"),
            ("num_attention_heads", "0", "num_attention_heads is 0"),
            ("num_hidden_layers", "0", "num_hidden_layers is 0"),
            ("attn_res_block_size", "0", "attn_res_block_size is 0"),
            ("q_lora_rank", "0", "q_lora_rank is 0"),
            ("kv_lora_rank", "0", "kv_lora_rank is 0"),
            // Non-zero but refused by `rivoli_mla_attend`'s guard 1004, which the load boundary
            // restates. Two SEPARATE `ensure!`s, so each row proves its own half: 500 is not a
            // multiple of 128, and 640 is one but exceeds MLA_ACC_REGS*SUBW = 512. As one
            // conjunctive check both rows matched both substrings and neither proved anything.
            ("kv_lora_rank", "500", "not a multiple of 128"),
            ("kv_lora_rank", "640", "exceeds the 512"),
        ] {
            // **Every row's key must be one the schema actually reads.** `json_obj` will happily
            // add a key nothing deserializes, and serde ignores unknown keys — so a row naming a
            // stale or misspelled key patches nothing, parses the baseline clean, and dies in
            // `k3_err`'s `unwrap_err()` rather than reporting what is wrong. That happened twice
            // in one sitting: `use_full_rank_gate` after it moved a level down, and
            // `activation_linear_beta` after the file showed the key is
            // `activation_situ_linear_beta`. This assertion is the diagnosis those two needed.
            assert!(
                K3_TEXT.iter().any(|(t, _)| *t == key) || key == "rope_theta",
                "{key:?} is not a key of the fixture (and not the deliberately-absent \
                 `rope_theta`), so this row would be an unknown JSON key: ignored, and testing \
                 nothing"
            );
            let err = k3_err(&[(key, bad)]);
            assert!(err.contains(want), "{key}={bad}: want {want:?}, got: {err}");
        }
        // The two `linear_attn_config` scalars whose wrong values are arithmetic. These cannot
        // ride the loop above, and the reason is the bug that made this block necessary: an
        // override of `use_full_rank_gate` at `text_config` level is an UNKNOWN KEY, silently
        // ignored, so the row parsed clean and `unwrap_err()` panicked. That is what "the field
        // was declared one level too high" looks like from the test side.
        let lac = k3_linear_attn(93);
        for (from, to, want) in [
            (
                r#""use_full_rank_gate":true"#,
                "false",
                "use_full_rank_gate is false",
            ),
            // `gate_lower_bound` MULTIPLIES the sigmoid (trap 4), so 0.0 zeroes every KDA gate
            // on all 69 layers — the output goes quiet rather than wrong, and nothing refuses it.
            (r#""gate_lower_bound":-5.0"#, "0.0", "gate_lower_bound 0"),
        ] {
            let (key, _) = from.split_once(':').unwrap();
            let bad = lac.replace(from, &format!("{key}:{to}"));
            assert_ne!(bad, lac, "the {key} row no longer patches anything");
            let err = k3_err(&[("linear_attn_config", &bad)]);
            assert!(err.contains(want), "{key}={to}: want {want:?}, got: {err}");
        }
    }

    /// The two layer arrays must PARTITION `1..=n_layers`. Every case here is a layer running
    /// the wrong attention family — KDA where MLA belongs or the reverse — which the reference
    /// itself names as the mistake its one-based indexing invites, and which nothing
    /// downstream can see: both families take `[hidden]` in and return `[hidden]`.
    #[test]
    fn k3_layer_partition_must_be_a_partition() {
        let lac = k3_lac;
        let complement: Vec<usize> = (1..=93).filter(|l| !K3_MLA_ONE_BASED.contains(l)).collect();
        // **Every case but the first keeps both lengths right.** The length check runs first,
        // so a short array reports only that — and a case that trips it proves nothing about
        // the check it was written for. Two of these were exactly that until it was noticed.
        let mut overlap = complement.clone();
        overlap[0] = 4; // layer 4 is MLA in the real config; claim it for KDA as well
        // One-based read as zero-based: the specific slip the reference calls out. Lengths
        // stay right; layer 3 is claimed twice and layer 93 by nobody.
        let shifted: Vec<usize> = K3_MLA_ONE_BASED.iter().map(|l| l - 1).collect();
        // Out of range and zero, both inside a full-length array so the length check passes.
        let (mut oob, mut zero) = (K3_MLA_ONE_BASED, K3_MLA_ONE_BASED);
        (oob[0], zero[0]) = (94, 0);
        for (bad, want) in [
            (lac(&[4], &complement), "= 70 layers"),
            (lac(&K3_MLA_ONE_BASED, &overlap), "appears twice"),
            (lac(&shifted, &complement), "appears twice"),
            (lac(&oob, &complement), "past num_hidden_layers"),
            (lac(&zero, &complement), "ONE-BASED"),
        ] {
            let err = k3_err(&[("linear_attn_config", &bad)]);
            assert!(err.contains(want), "want {want:?} for {bad}, got: {err}");
        }
        // The positive control: a 4-layer model whose only MLA layer is 4. Without it every
        // row above could pass on a `validate_layer_partition` that refuses everything.
        let small = k3_linear_attn(4);
        parse_k3(&[("linear_attn_config", &small), ("num_hidden_layers", "4")])
            .expect("1..=4 with layer 4 as the only MLA layer IS a partition");
    }

    /// **The gate that makes every other K3 test mean something.** `K3_TEXT` is a hand-copy;
    /// this compares each of its keys against the vendored `config.json`, and then parses that
    /// file directly.
    ///
    /// It earned its keep before it was even committed. The schema had two fields that would
    /// have refused every real K3 checkpoint — `activation_linear_beta` for
    /// `activation_situ_linear_beta`, and `use_full_rank_gate` declared one level too high — and
    /// no amount of internal consistency between a fixture and a struct can catch that class.
    /// Only the file can. Structural rather than a hand-listed tuple, so the check grows with
    /// the table instead of covering whichever fields someone thought to list.
    #[test]
    fn k3_base_matches_the_shipped_config() {
        let real: serde_json::Value = serde_json::from_str(K3_SHIPPED).unwrap();
        let text = real.get("text_config").expect("no text_config");
        for (k, v) in K3_TEXT {
            // `linear_attn_config`'s row is the deliberate `null` placeholder; its real value is
            // checked field-by-field below, where the nesting can be compared properly.
            if *k == "linear_attn_config" {
                continue;
            }
            let want: serde_json::Value = serde_json::from_str(v)
                .unwrap_or_else(|e| panic!("K3_TEXT[{k}] is not valid JSON: {e}"));
            let got = text
                .get(*k)
                .unwrap_or_else(|| panic!("text_config has no {k:?}"));
            assert_eq!(
                got, &want,
                "K3_TEXT[{k}] has drifted from the shipped config"
            );
        }
        // The nested dict, including the five scalars whose spellings this port got wrong from
        // the C reference's field names before the file was read.
        let lac = text
            .get("linear_attn_config")
            .expect("no linear_attn_config");
        for (k, v) in [
            ("num_heads", "96"),
            ("head_dim", "128"),
            ("short_conv_kernel_size", "4"),
            ("gate_lower_bound", "-5.0"),
            ("use_full_rank_gate", "true"),
        ] {
            let want: serde_json::Value = serde_json::from_str(v).unwrap();
            assert_eq!(lac.get(k), Some(&want), "linear_attn_config.{k}");
        }
        // The layer arrays, against the transcribed constant rather than against themselves.
        let mla: Vec<usize> = serde_json::from_value(lac["full_attn_layers"].clone()).unwrap();
        assert_eq!(mla, K3_MLA_ONE_BASED, "full_attn_layers");
        assert_eq!(lac["kda_layers"].as_array().unwrap().len(), 69);
        // The wrapper's pair, which is the level `Arch::from_manifest_str` reads — and which
        // differs from the nested one. Both spellings, since either alone resolves.
        assert_eq!(real["model_type"], "kimi_k3");
        assert_eq!(real["architectures"][0], "KimiK3ForConditionalGeneration");
        // **`top_k` is HuggingFace's SAMPLING top-k, not the router's.** Asserted as a
        // difference, so a later "simplification" that binds `K3TextConfig::top_k` from the key
        // of that name — 50 experts a token instead of 16 — reddens here.
        assert_eq!(text["top_k"], 50);
        assert_ne!(text["top_k"], text["num_experts_per_token"]);
        // `quantization_config` is not in the schema on purpose (item 5). Pin the two facts that
        // decision rests on, so "do not trust this block" stays a measurement: its group size is
        // the `F4_GROUP` the repack assumes, and its `ignore` list does NOT name the three
        // families that ship BF16 — which is why the converter drives off `.weight_packed`.
        let q = text
            .get("quantization_config")
            .expect("no quantization_config");
        assert_eq!(q["format"], "mxfp4-pack-quantized");
        assert_eq!(
            q["config_groups"]["group_0"]["weights"]["group_size"],
            crate::artifact::quant::F4_GROUP
        );
        let ignore = serde_json::to_string(&q["ignore"]).unwrap();
        for missing in [
            "routed_expert_down_proj",
            "routed_expert_up_proj",
            "gate.weight",
        ] {
            assert!(
                !ignore.contains(missing),
                "the ignore list now names {missing:?} — item 5's argument for driving off \
                 `.weight_packed` was that it does NOT, so re-read the block before relying on it"
            );
        }
        // ...and the whole file parses through the real schema, from the bytes rather than from
        // the fixture. This is the assertion the two wrong spellings failed.
        parse_config::<K3Config>(K3_SHIPPED).expect("the shipped config must parse");
    }

    /// **K3 fits the routed pool's batch scratch only at ONE row**, and this is the arithmetic that
    /// says so — written before the K3 pin exists, because the pin's obvious first draft is a copy
    /// of GLM's line and that copy is wrong.
    ///
    /// `RoutedPool::submit` has a fixed `MAX_BATCH`-slot hit scratch, and a batched forward submits
    /// the UNION of every token row's picks: `top_k · rows + n_shared`. At K3's shipped scalars that
    /// is `16 · rows + 2`, so one row needs 18 of 32 and **two rows need 34 — over by two**.
    ///
    /// K3's row count is 1 because `num_nextn_predict_layers` is 0 and `validate` refuses otherwise.
    /// The live check is `pin.rs`'s, which multiplies by the GLOBAL `crate::gpu::MAXROW` — 2, because
    /// GLM ships a draft head — and would therefore refuse every K3 artifact at load. That is loud,
    /// not silent; what this test adds is the arithmetic in front of whoever writes the K3 pin.
    ///
    #[cfg(feature = "rocm")]
    #[test]
    fn k3_fits_the_routed_batch_scratch_at_one_row_and_not_at_two() {
        use crate::memory::routed::MAX_BATCH;
        let c = parse_k3(&[]).unwrap().text;
        // `k3_baseline_parses` already pins these against the vendored config; this reads them so
        // the arithmetic below is the config's rather than a literal's.
        let batch = |rows: usize| c.top_k * rows + c.n_shared;
        assert!(
            batch(1) <= MAX_BATCH,
            "K3 needs {} slots at one row and the scratch has {MAX_BATCH}",
            batch(1)
        );
        // Literal 2, NOT `crate::gpu::MAXROW`: the claim is about a two-row pass, and pinning it to
        // the global would make this fail when someone gives K3 its own row count of 1.
        assert!(
            batch(2) > MAX_BATCH,
            "K3 now fits a two-row pass ({} slots of {MAX_BATCH}). If that is intended it needs a \
             batched fp4 MoE kernel, which the tree does not have — `kernels/moe.hip` instantiates \
             R=1 for the f4 path (`nrow != 1` is refused) — and a draft head to fill the second row, \
             which `num_nextn_predict_layers: 0` says K3 has not got.",
            batch(2)
        );
    }

    /// A K3 config must not become a zero-filled config of either other architecture, nor
    /// the reverse. Same defect as [`each_config_refuses_the_other_architecture`], third hat:
    /// K3's wrapper is the only one of the three whose dimensions are not at the top level, so
    /// a foreign config fed to `K3Config` fails on the WRAPPER's discriminant rather than on a
    /// missing `text_config`.
    #[test]
    fn k3_refuses_the_other_architectures() {
        let k3 = k3_json(&[]);
        for err in [
            format!("{:#}", parse_config::<ModelConfig>(&k3).unwrap_err()),
            format!("{:#}", parse_config::<V4Config>(&k3).unwrap_err()),
        ] {
            assert!(
                err.contains("kimi_k3") && !err.contains("missing field"),
                "must refuse on the architecture, not on a field: {err}"
            );
        }
        for foreign in [cfg_json(&[]), v4_json(&[])] {
            let err = format!("{:#}", parse_config::<K3Config>(&foreign).unwrap_err());
            assert!(
                err.contains("KimiK3") && !err.contains("text_config"),
                "must refuse on the architecture, not on the missing nesting: {err}"
            );
        }
    }

    // ── Muse Glimmer ────────────────────────────────────────────────────────────────────
    //
    // These parse the SHIPPED config directly rather than assembling a base from transcribed
    // constants the way the K3 block above does. The K3 shape exists because its `config.json`
    // has to be reachable field-by-field to check five nested spellings the C reference got
    // wrong; Glimmer's schema has no such history, and a transcribed base is one more place
    // for the file's values to drift out of. Mutating the real document is also the stronger
    // gate: every refusal test below starts from a document that is known to parse.

    /// `include_str!` for the reason [`K3_SHIPPED`] gives — this port has no checkpoint on
    /// this machine, so a path read would skip rather than run. 5 KB, pinned at the HF
    /// revision `f84ecc3a0ea984a4c04542a84269e3d065350a6e`.
    const GLIMMER_SHIPPED: &str =
        include_str!("../../docs/measurement/glimmer-reference/config.json");

    /// The refusal message for the shipped config with one value replaced, or a panic naming
    /// the mutation that was wrongly ACCEPTED. The panic is the point: a refusal test whose
    /// subject silently parses is the false green this whole file exists to prevent.
    ///
    /// Goes through [`parse_config`] — the same entry the binary uses. A test that constructed
    /// a `GlimmerTextConfig` literal would skip both the architecture check and `validate`,
    /// which are the two things under test.
    fn glimmer_err(pointer: &str, value: serde_json::Value) -> String {
        let mut doc: serde_json::Value = serde_json::from_str(GLIMMER_SHIPPED).unwrap();
        let slot = doc
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("{pointer} is not a path in the shipped config"));
        *slot = value;
        match parse_config::<GlimmerConfig>(&doc.to_string()) {
            Ok(_) => panic!("{pointer} was mutated to a wrong value and the config still parsed"),
            Err(e) => format!("{e:#}"),
        }
    }

    /// **Every field the schema declares is REQUIRED — enforced, not just claimed.**
    ///
    /// `V4Config` and `K3TextConfig` each have this test and `GlimmerTextConfig` did not,
    /// while its doc made the identical claim across 22 fields. V4's twin records why a weaker
    /// `is_err()` check is insufficient: an injected `#[serde(default)]` on `index_topk` left
    /// it passing unchanged.
    ///
    /// Driven off the SHIPPED document rather than a list of field names, and asserted in
    /// **both** directions: deleting a key the schema needs must refuse, and deleting one it
    /// does not carry must still parse. The sets are compared whole, so a field that stops
    /// being required moves between them and reddens.
    ///
    /// **What it does and does not catch, measured 2026-08-11 rather than assumed.** The
    /// property is "removing this key is refused" — by serde *or* by `validate`, which is the
    /// property that actually matters, since either way the checkpoint is rejected. So:
    ///
    /// - `#[serde(default)]` on `attention_bias` **reddens** (default `false` is the
    ///   acceptable value, so the config would parse and run).
    /// - `#[serde(default)]` on `head_dim` **does not** — the default 0 is caught by the width
    ///   loop in `validate`, so the config is still refused and no defect exists. An earlier
    ///   draft of this doc named `head_dim` as the worked example and was wrong; the red-proof
    ///   run is what corrected it.
    ///
    /// The gap that leaves: a defaulted field whose default is both *acceptable to `validate`*
    /// and *wrong for this checkpoint*. Every such field here is a `bool` or `String` the
    /// shipped config pins, and each is asserted by value in
    /// [`glimmer_shipped_config_parses_and_matches_the_reference_doc`].
    #[test]
    fn every_glimmer_field_is_required() {
        // The `text_config` keys this schema does NOT bind. Everything else in the dict must
        // be load-bearing; a key added to the struct without being removed from here fails.
        // Sorted, because the comparison below is on the whole list. `top_k` was in here on
        // the first draft — it is K3's sampling key and Glimmer's `text_config` has none, and
        // this test caught it the first time it ran, which is the argument for the test.
        const NOT_IN_SCHEMA: [&str; 6] = [
            "attention_dropout",
            "bos_token_id",
            "eos_token_id", // EOS comes from `generation_config`, which lists TWO ids (trap 13)
            "initializer_range",
            "pad_token_id",
            "use_cache",
        ];
        let doc: serde_json::Value = serde_json::from_str(GLIMMER_SHIPPED).unwrap();
        let text = doc["text_config"]
            .as_object()
            .expect("the shipped config's text_config is not an object");
        let keys: Vec<String> = text.keys().cloned().collect();
        let (mut required, mut ignored) = (Vec::new(), Vec::new());
        for k in &keys {
            let mut d = doc.clone();
            d["text_config"].as_object_mut().unwrap().remove(k);
            match parse_config::<GlimmerConfig>(&d.to_string()) {
                Err(_) => required.push(k.clone()),
                Ok(_) => ignored.push(k.clone()),
            }
        }
        // Sorted HERE, because `serde_json` runs with `preserve_order` and so `text.keys()`
        // yields FILE order. It happens to equal sorted order for the vendored config, which
        // is why the whole-list comparison held — but a re-vendor from a source that emits
        // keys in declaration order would redden this with a diff of two identical SETS, and
        // point the reader at a serde change that did not happen. Review, 2026-08-11.
        ignored.sort();
        assert_eq!(
            ignored, NOT_IN_SCHEMA,
            "the set of text_config keys this schema tolerates as ABSENT has changed.\n  \
             required: {required:?}\n  If a field gained #[serde(default)] it moved into the \
             tolerated set, which is the silent-wrong-text case this test exists for."
        );
        // The wrapper's own fields. `quantization_config` is deliberately absent from the
        // shipped file, so it is the one key whose ABSENCE is correct — asserted separately
        // below rather than folded in here.
        for k in ["text_config", "dtype"] {
            let mut d = doc.clone();
            d.as_object_mut().unwrap().remove(k);
            assert!(
                parse_config::<GlimmerConfig>(&d.to_string()).is_err(),
                "removing the wrapper's {k:?} must refuse"
            );
        }
        assert!(
            doc.get("quantization_config").is_none(),
            "the shipped config must NOT carry a quantization_config — the schema refuses one"
        );
    }

    /// The shipped config parses, and every value this port acts on is what
    /// `glimmer-architecture.md` §1 records. Pinned against the FILE, so the doc and the
    /// schema cannot drift apart silently.
    #[test]
    fn glimmer_shipped_config_parses_and_matches_the_reference_doc() {
        let cfg: GlimmerConfig = parse_config(GLIMMER_SHIPPED).unwrap();
        assert_eq!(cfg.dtype, "bfloat16");
        let t = &cfg.text;
        assert_eq!(t.model_type, "muse_glimmer_text");
        assert_eq!(t.n_layers, 52);
        assert_eq!(t.hidden, 6656);
        assert_eq!(t.inter, 19968);
        assert_eq!(t.vocab, 202_048);
        assert_eq!(t.n_heads, 32);
        assert_eq!(t.num_key_value_heads, 2);
        assert_eq!(t.head_dim, 128);
        assert_eq!(t.sliding_window, 2048);
        assert_eq!(t.max_position_embeddings, 131_072);
        assert_eq!(t.rope_parameters.rope_theta, 500_000.0);
        assert_eq!(t.rope_parameters.rope_type, "default");
        assert_eq!(t.final_logit_softcapping, 20.0);
        assert_eq!(t.qk_scale_factor, 3.87);
        assert_eq!(t.rms_norm_eps, 1e-5);
        assert_eq!(t.post_norm_eps, 1e-8);
        assert!(!t.tie_word_embeddings);
        assert!(!t.attention_bias);
        assert_eq!(t.hidden_activation, "silu");

        // `head_dim` is NOT `hidden / n_heads` here (6656/32 = 208, against 128), and it is
        // not unique in that: `V4Config` carries its own for the same reason. The guard
        // against "drop the field and derive it" is that removing it fails to COMPILE, which
        // is stronger than any assertion — an `assert_ne!` over these three already-pinned
        // constants was deleted by review 2026-08-11 as unable to fail independently.

        // The layer map, counted from the arrays rather than from the [s,s,s,full] period —
        // the period is a fact about this checkpoint, the arrays are the contract. The exact
        // array equality below implies both the 39/13 split and that the LAST layer is full
        // (a named gate blind spot), so neither is asserted separately.
        let full: Vec<usize> = (0..t.n_layers)
            .filter(|&i| !t.layer_is_sliding(i).unwrap())
            .collect();
        assert_eq!(full, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]);
        // Out of range is an ERROR, not a silent "full" — see `layer_is_sliding`.
        assert!(t.layer_is_sliding(t.n_layers).is_err());
    }

    /// **The defect run.** Every load-bearing field, mutated one at a time, must refuse — and
    /// the assertion is on the MESSAGE, so a refusal that happens to fire for an unrelated
    /// reason does not count as this row passing.
    ///
    /// G1a asks that a config missing or contradicting a load-bearing field refuse at startup,
    /// "proven by feeding it one". This is that proof. Each row is a value that changes the
    /// arithmetic without changing a shape, which is the failure class this model is full of:
    /// nothing downstream crashes on any of them.
    #[test]
    fn glimmer_refuses_the_silently_wrong_settings() {
        use serde_json::json;
        for (pointer, value, want) in [
            // The descent. `vision_config` is a sibling carrying several of the same keys, so
            // landing in it is the realistic wrong-dict case rather than a hypothetical one.
            (
                "/text_config/model_type",
                json!("muse_glimmer_vision"),
                "muse_glimmer_text",
            ),
            // The two per-layer arrays are indexed by layer id everywhere downstream.
            (
                "/text_config/layer_types",
                json!(["sliding_attention"]),
                "1 entries",
            ),
            ("/text_config/num_hidden_layers", json!(51), "52 entries"),
            // An unknown layer kind, which would otherwise read as "not sliding" = full.
            (
                "/text_config/layer_types/0",
                json!("chunked_attention"),
                "expected",
            ),
            // **The pairing invariant, in BOTH directions** — the strongest claim the config
            // alone can make. Layer 0 is sliding and rotated; layer 3 is full and not.
            ("/text_config/layer_rope_theta/0", json!(0), "rotated IFF"),
            (
                "/text_config/layer_rope_theta/3",
                json!(500_000.0),
                "rotated IFF",
            ),
            // A rotated layer asking for a base the single shared table is not built from.
            (
                "/text_config/layer_rope_theta/0",
                json!(10_000.0),
                "read and then ignored",
            ),
            // GQA: 32 query heads do not divide into 3 KV heads.
            (
                "/text_config/num_key_value_heads",
                json!(3),
                "whole number of query heads",
            ),
            // **One row per entry of the width table**, so deleting any entry reddens exactly
            // one row. Three of the nine were sampled until review 2026-08-11 pointed out the
            // rest were free — K3's equivalent carries the same correction, which makes this a
            // repeat of a known finding rather than a fresh one.
            ("/text_config/hidden_size", json!(0), "hidden_size is 0"),
            (
                "/text_config/intermediate_size",
                json!(0),
                "intermediate_size is 0",
            ),
            ("/text_config/vocab_size", json!(0), "vocab_size is 0"),
            (
                "/text_config/num_attention_heads",
                json!(0),
                "num_attention_heads is 0",
            ),
            (
                "/text_config/num_key_value_heads",
                json!(0),
                "num_key_value_heads is 0",
            ),
            ("/text_config/head_dim", json!(0), "head_dim is 0"),
            (
                "/text_config/sliding_window",
                json!(0),
                "sliding_window is 0",
            ),
            (
                "/text_config/max_position_embeddings",
                json!(0),
                "max_position_embeddings is 0",
            ),
            // Scaling schemes and tying: both silently unimplemented rather than refused.
            (
                "/text_config/rope_parameters/rope_type",
                json!("yarn"),
                "RopeNoYarn",
            ),
            (
                "/text_config/tie_word_embeddings",
                json!(true),
                "declares them untied",
            ),
            ("/text_config/attention_bias", json!(true), "none ships"),
            ("/text_config/hidden_activation", json!("gelu"), "SwiGLU"),
            // f32 narrowing. `1e-46` passes any f64 positivity test and reaches every RMSNorm
            // as `0.0f32`; the softcap at 0 is a divide-by-zero the greedy path cannot see.
            ("/text_config/rms_norm_eps", json!(1e-46), "narrows to 0"),
            (
                "/text_config/final_logit_softcapping",
                json!(0.0),
                "narrows to 0",
            ),
            // "narrows to", not the field name: `:2472` records a review finding where two
            // rows shared a `want` substring and transposing an argument left both green.
            ("/text_config/output_multiplier", json!(-0.5), "narrows to"),
            ("/text_config/post_norm_eps", json!(0.0), "narrows to"),
            ("/text_config/qk_scale_factor", json!(0.0), "narrows to"),
            // NOT a row: `rope_parameters.rope_theta`. Its `ensure_f32_positive` entry is
            // unreachable by any SINGLE mutation — the pairing loop runs first and refuses
            // ("read and then ignored") as soon as the global base stops matching the 39
            // sliding layers' own. It is reachable only by mutating the global base and all
            // 39 together, which this one-pointer helper cannot express. Recorded rather than
            // asserted, so the gap is visible instead of looking like coverage.
            // The wrapper-level dtype: a 4-bit or fp8 export read as BF16 is noise at every
            // width, and the model card advertises exactly such a release.
            ("/dtype", json!("float8_e4m3fn"), "BF16 throughout"),
        ] {
            let err = glimmer_err(pointer, value.clone());
            assert!(
                err.contains(want),
                "{pointer} = {value} refused, but not for the reason under test\n  \
                 wanted the message to contain: {want}\n  got: {err}"
            );
        }

        // **A packed export must refuse even though its `dtype` is honest.** Not a table row
        // because the helper replaces an existing value and this key must be INSERTED — the
        // shipped file has none, which is the correct state and is asserted by
        // `every_glimmer_field_is_required`.
        //
        // The block below is K3's own, copied from its vendored config: `bfloat16` alongside
        // 4-bit packed weights. Before this guard existed, serde ignored it and the document
        // parsed clean.
        let mut doc: serde_json::Value = serde_json::from_str(GLIMMER_SHIPPED).unwrap();
        doc.as_object_mut().unwrap().insert(
            "quantization_config".into(),
            serde_json::json!({ "format": "mxfp4-pack-quantized", "num_bits": 4 }),
        );
        let err = format!(
            "{:#}",
            parse_config::<GlimmerConfig>(&doc.to_string()).unwrap_err()
        );
        assert!(
            err.contains("unquantized release"),
            "a config carrying a quantization_config must refuse even with dtype bfloat16, \
             which is exactly how Kimi-K3's checkpoint ships: {err}"
        );
    }

    /// The architecture check fires before serde reads a dimension, and it fires in both
    /// directions — a Glimmer document must not parse as another schema either. Without the
    /// second half this passes on a `parse_config` that refuses everything.
    #[test]
    fn glimmer_and_the_other_architectures_do_not_cross_parse() {
        // A Glimmer document, read as each other schema.
        assert!(parse_config::<ModelConfig>(GLIMMER_SHIPPED).is_err());
        assert!(parse_config::<V4Config>(GLIMMER_SHIPPED).is_err());
        assert!(parse_config::<K3Config>(GLIMMER_SHIPPED).is_err());
        // ...and K3's document read as Glimmer's. Both are `*ForConditionalGeneration`
        // wrappers with a `text_config`, which is what makes this pair worth asserting: the
        // shapes are similar enough that only the discriminant separates them.
        let err = format!(
            "{:#}",
            parse_config::<GlimmerConfig>(K3_SHIPPED).unwrap_err()
        );
        assert!(
            err.contains("MuseGlimmer") && err.contains("kimi_k3"),
            "must refuse on the architecture, naming both sides: {err}"
        );
    }
}
