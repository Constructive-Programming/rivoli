//! Model dimensions, parsed from the snapshot's `config.json`. GLM-5.2 is an
//! MLA (multi-head latent attention) + MoE architecture; the fields the decode
//! path needs are pulled and validated at load.

use anyhow::{Context, Result};
use serde::Deserialize;

/// `rope_parameters` is a nested object; we only need theta.
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

    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    pub rms_norm_eps: f64,
    #[serde(rename = "rope_parameters")]
    rope: RopeParameters,
}

impl ModelConfig {
    pub fn load(snapshot_dir: &str) -> Result<Self> {
        let path = format!("{snapshot_dir}/config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
        let cfg: Self = serde_json::from_str(&text).with_context(|| format!("parse {path}"))?;
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
        // The fused MoE kernel (moe_fused.hip) stages one SwiGLU intermediate row
        // per workgroup in dynamic LDS = inter*4 bytes, capped by gfx1151's 64KB
        // LDS budget → inter ≤ 16384. Fail at load with the cause, not at first
        // MoE launch. GLM-5.2: dense_inter=12288, moe_inter*n_shared=2048 — fit.
        const MAX_FUSED_INTER: usize = 16384;
        let shared_inter = self.moe_inter * self.n_shared;
        if self.dense_inter > MAX_FUSED_INTER || shared_inter > MAX_FUSED_INTER {
            anyhow::bail!(
                "intermediate width exceeds fused-kernel LDS ceiling {MAX_FUSED_INTER} \
                 (dense_inter={}, moe_inter*n_shared={shared_inter})",
                self.dense_inter
            );
        }
        Ok(())
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope.rope_theta
    }

    /// Total per-head query/key dimension (nope + rope).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}
