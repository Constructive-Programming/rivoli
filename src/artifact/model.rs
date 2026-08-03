//! Model dimensions, parsed from the snapshot's `config.json`. GLM-5.2 is an
//! MLA (multi-head latent attention) + MoE architecture; the fields the decode
//! path needs are pulled and validated at load.

use anyhow::{Context, Result};
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
    /// GLM-5.2 nests theta; the DeepSeek/Llama lineage puts `rope_theta` at top
    /// level. Accept either — this is the first thing that fails on a non-GLM
    /// config, before any dimension is even looked at.
    #[serde(rename = "rope_parameters", default)]
    rope: Option<RopeParameters>,
    #[serde(rename = "rope_theta", default)]
    rope_theta_flat: Option<f64>,
}

impl ModelConfig {
    /// Load from the artifact's `manifest.json` (the config fields live at top level
    /// alongside a `format` section; serde ignores the unknown key), falling back to a
    /// bare `config.json` for reading a raw checkpoint.
    pub fn load(dir: &str) -> Result<Self> {
        let path = match std::fs::metadata(format!("{dir}/manifest.json")) {
            Ok(_) => format!("{dir}/manifest.json"),
            Err(_) => format!("{dir}/config.json"),
        };
        let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
        Self::from_json(&text).with_context(|| format!("parse {path}"))
    }

    /// Parse and validate one config document. Split out of `load` so the tests can
    /// drive the real boundary from a literal rather than re-implementing it — which
    /// is how a test starts passing against itself.
    fn from_json(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text)?;
        cfg.validate()?;
        Ok(cfg)
    }

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
        // `vq_row_bytes`/`vq_groups` divide by VQ_DIM and VQ_GROUP with only a
        // `debug_assert` to catch a ragged dim, so in a RELEASE build a bad width
        // silently truncates every expert row instead of failing. Both widths are
        // an `i_dim` for some projection (gate/up take hidden, down takes moe_inter),
        // and VQ_GROUP=64 is a multiple of the 8 the 12-bit packing needs, so this
        // one check covers both. GLM-5.2: 6144, 2048 — both clean.
        for (what, dim) in [("hidden", self.hidden), ("moe_inter", self.moe_inter)] {
            if !dim.is_multiple_of(crate::artifact::quant::VQ_GROUP) {
                anyhow::bail!(
                    "{what} {dim} is not a multiple of VQ_GROUP {} — int3-vq rows would \
                     silently truncate in a release build",
                    crate::artifact::quant::VQ_GROUP
                );
            }
        }
        // No ceiling on the intermediate widths: the LDS-staging MoE kernel that
        // imposed one (inter ≤ 16384, gfx1151's 64KB budget) is gone. `swiglu`
        // (linalg.hip) is elementwise with zero dynamic LDS, and moe.hip stages
        // nothing on purpose — LDS capped occupancy and measured slower. The old
        // guard would have refused the DeepSeek-V3 family on intermediate_size
        // 18432 for a constraint that no longer exists.
        Ok(())
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

/// The load boundary is the only place a foreign snapshot is inspected before its
/// dimensions reach a kernel, so every refusal here is one that would otherwise be an
/// out-of-bounds panic, a silently truncated expert row, or wrong experts with no crash.
#[cfg(test)]
mod tests {
    use super::*;

    /// GLM-5.2's real values, as a JSON object that `overrides` can patch. Built as
    /// text rather than a struct literal because the serde renames ARE the contract
    /// under test — `n_routed_experts`, the nested rope, `first_k_dense_replace`.
    fn cfg_json(overrides: &[(&str, &str)]) -> String {
        let mut fields: Vec<(String, String)> = [
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
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        for (k, v) in overrides {
            fields.retain(|(fk, _)| fk != k);
            if !v.is_empty() {
                fields.push((k.to_string(), v.to_string()));
            }
        }
        let body: Vec<String> = fields
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect();
        format!("{{{}}}", body.join(","))
    }

    fn parse(overrides: &[(&str, &str)]) -> Result<ModelConfig> {
        ModelConfig::from_json(&cfg_json(overrides))
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
        assert_eq!(
            parse(&[("scoring_func", r#""sqrtsoftplus""#)]).unwrap().scoring(),
            crate::math::Scoring::SqrtSoftplus
        );
        assert_eq!(
            parse(&[("scoring_func", r#""sigmoid""#)]).unwrap().scoring(),
            crate::math::Scoring::Sigmoid
        );
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
}
