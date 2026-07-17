//! Model dimensions, parsed from the snapshot's `config.json`. GLM-5.2 is an
//! MLA + MoE architecture; only the fields the decode path needs are pulled.

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    #[serde(rename = "n_routed_experts")]
    pub n_experts: usize,
    #[serde(rename = "num_experts_per_tok")]
    pub top_k: usize,
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter: usize,
    /// First `dense_layers` layers are dense MLP, not MoE (first_k_dense_replace).
    #[serde(rename = "first_k_dense_replace")]
    pub dense_layers: usize,
    #[serde(rename = "vocab_size")]
    pub vocab: usize,
}

impl ModelConfig {
    pub fn load(snapshot_dir: &str) -> Result<Self> {
        let path = format!("{snapshot_dir}/config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
        serde_json::from_str(&text).with_context(|| format!("parse {path}"))
    }
}
