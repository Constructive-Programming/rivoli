//! Model dimensions, parsed from the snapshot's `config.json`.
//!
//! **One type per architecture, and the type is the proof.** [`ModelConfig`] describes the
//! MLA (multi-head latent attention) + dense-prefix lineage — GLM-5.2, DeepSeek-V3.
//! [`V4Config`] describes DeepSeek-V4-Flash: shared-KV MQA, no dense layers, hash-routed
//! prefix, FP4 experts. Each refuses the other by name at [`crate::arch::Arch`], *before*
//! serde looks at a single dimension, so holding a value of either type is evidence about
//! which architecture the snapshot is.
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

/// Both of an expert's input widths must divide the group-scale span exactly.
///
/// `vq_row_bytes`/`vq_groups` and their `.f4` counterparts round up with only a
/// `debug_assert` to catch a ragged dim, so in a RELEASE build a bad width silently
/// truncates every expert row instead of failing. Each width is an `i_dim` for some
/// projection — gate/up take `hidden`, down takes `moe_inter` — so one check covers both.
fn ensure_group_aligned(hidden: usize, moe_inter: usize, group: usize, what: &str) -> Result<()> {
    for (name, dim) in [("hidden", hidden), ("moe_inter", moe_inter)] {
        ensure!(
            dim.is_multiple_of(group),
            "{name} {dim} is not a multiple of {what} {group} — expert rows would \
             silently truncate in a release build"
        );
    }
    Ok(())
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
        ensure_group_aligned(
            self.hidden,
            self.moe_inter,
            crate::artifact::quant::F4_GROUP,
            stringify!(F4_GROUP),
        )
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
}
