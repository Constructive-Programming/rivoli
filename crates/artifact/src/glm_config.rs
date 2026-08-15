//! GLM-5.2's config schema — one architecture, one file, per the rule that per-model
//! config types stay separate (a shared struct is the attractor for a field that is not
//! shared; the old tree's merge hazard caught two fields living in the wrong struct).
//! Ported verbatim from the old `artifact/model.rs`'s GLM slice.

use crate::arch::Arch;
use crate::schema::{ArchConfig, ensure_group_aligned, load_config};
use anyhow::Result;
use serde::Deserialize;

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
    /// GLM-5.2 nests theta under `rope_parameters` — REQUIRED, not optional-with-
    /// fallback. The old tree accepted a flat top-level `rope_theta` too, arguing "first
    /// thing that fails on a non-GLM config"; that rationale predates this tree's design
    /// (review 2026-08-15): `parse_config` refuses non-GLM architectures BEFORE serde
    /// reads a dimension, so a DeepSeek config never reaches this struct — and the real
    /// checkpoint carries only the nested form (verified against glm52-fp8's config).
    #[serde(rename = "rope_parameters")]
    rope: RopeParameters,
}

impl ArchConfig for ModelConfig {
    const ARCH: Arch = Arch::GlmMoeDsa;

    /// Cheap semantic checks at the boundary, so a mismatched snapshot fails
    /// here with a clear message rather than as an out-of-bounds panic deep in
    /// the decode loop.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.n_layers > 0,
            "num_hidden_layers is 0 — an empty model validates nothing and panics later \
             (review 2026-08-15: indexer_layout indexes [0] after a length check that 0 == 0 passes)"
        );
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
        // VQ_GROUP=64 is a multiple of the 8 the 12-bit packing needs, so this one check
        // covers the indices too. GLM-5.2: 6144, 2048 — both clean.
        ensure_group_aligned(
            self.hidden,
            self.moe_inter,
            crate::quant::VQ_GROUP,
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
    pub fn scoring(&self) -> rivoli_core::num::Scoring {
        match self.scoring_func.as_deref() {
            Some("sqrtsoftplus") => rivoli_core::num::Scoring::SqrtSoftplus,
            _ => rivoli_core::num::Scoring::Sigmoid,
        }
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope.rope_theta
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

// (`QuantConfig` — the fp8 `quantization_config` block — is V4Config's field, not GLM's;
// it returns with its consumer at M8.)
