//! Model dimensions, parsed from the snapshot's `config.json`. GLM-5.2 is an
//! MLA + MoE architecture; only the fields the decode path needs are pulled.

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    /// First `dense_layers` layers are dense MLP, not MoE (first_k_dense_replace).
    pub dense_layers: usize,
    pub vocab: usize,
}

impl ModelConfig {
    pub fn load(snapshot_dir: &str) -> Result<Self> {
        let path = format!("{snapshot_dir}/config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
        let v = crate::json::parse(&text)?;
        let u = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|n| n as usize)
                .with_context(|| format!("config.json missing usize field {k}"))
        };
        Ok(Self {
            n_layers: u("num_hidden_layers")?,
            hidden: u("hidden_size")?,
            n_experts: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            moe_inter: u("moe_intermediate_size")?,
            dense_layers: u("first_k_dense_replace")?,
            vocab: u("vocab_size")?,
        })
    }

    /// A layer is MoE iff it is past the dense prefix.
    pub fn is_moe(&self, layer: usize) -> bool {
        layer >= self.dense_layers
    }
}
