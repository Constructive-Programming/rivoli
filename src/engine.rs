//! The decode loop — the M1 gate. Assembles the forward pass:
//!   embed → per layer [ rmsnorm(input_ln) → attn → +residual
//!                       → rmsnorm(post_ln) → mlp → +residual ]
//!         → rmsnorm(model.norm) → lm_head → greedy argmax.
//! Reference scalar path with on-demand weight reads (the pin/streaming feed and
//! the fused GPU kernels are later milestones). Greedy sampling for now — the
//! M1 gate is coherence, not sampling strategy.

use crate::attn::{AttnMode, AttnScratch, KvCache, attention};
use crate::indexer::Indexer;
use crate::math::rmsnorm_into_bytes;
use crate::model::ModelConfig;
use crate::moe::{MlpScratch, dense_mlp, moe_block};
use crate::quant::{dequant_int8_row, matvec_i8};
use crate::snapshot::Snapshot;
use anyhow::{Result, bail, ensure};

pub struct Engine<'a> {
    snap: &'a Snapshot,
    cfg: &'a ModelConfig,
    mode: AttnMode,
    indexer: Option<Indexer>,
    kv: KvCache,
    ascr: AttnScratch,
    mscr: MlpScratch,
    x: Vec<f32>,      // residual stream (hidden)
    xn: Vec<f32>,     // normed input to a sublayer (hidden)
    sub: Vec<f32>,    // sublayer output (hidden)
    logits: Vec<f32>, // vocab
}

impl<'a> Engine<'a> {
    /// Fails at construction (not mid-decode) when a sparse mode is requested
    /// but the snapshot/config lacks the indexer — the out-idx shard and the
    /// indexer dims are validated here.
    pub fn new(snap: &'a Snapshot, cfg: &'a ModelConfig, mode: AttnMode) -> Result<Self> {
        let indexer = match mode {
            AttnMode::Dsa | AttnMode::Misa { .. } => Some(Indexer::new(cfg)?),
            AttnMode::Dense | AttnMode::Streaming { .. } => None,
        };
        Ok(Self {
            snap,
            cfg,
            mode,
            indexer,
            kv: KvCache::new(cfg),
            ascr: AttnScratch::new(cfg),
            mscr: MlpScratch::new(cfg),
            x: vec![0.0; cfg.hidden],
            xn: vec![0.0; cfg.hidden],
            sub: vec![0.0; cfg.hidden],
            logits: vec![0.0; cfg.vocab],
        })
    }

    /// One forward pass for `token` at `pos`, leaving next-token logits in
    /// `self.logits`.
    fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let eps = cfg.rms_norm_eps as f32;

        // Embedding (int8 table, validated).
        let embed = self.snap.int8("model.embed_tokens", cfg.hidden)?;
        dequant_int8_row(&embed, token as usize, &mut self.x);

        for layer in 0..cfg.n_layers {
            let lb = format!("model.layers.{layer}");
            // Attention sublayer: rmsnorm(input_ln) into xn (residual x survives).
            let in_ln = self.snap.require(&format!("{lb}.input_layernorm.weight"))?;
            rmsnorm_into_bytes(&mut self.xn, &self.x, in_ln, eps);
            attention(
                self.snap,
                cfg,
                layer,
                &self.xn,
                pos,
                &self.mode,
                self.indexer.as_mut(),
                &mut self.kv,
                &mut self.ascr,
                &mut self.sub,
            )?;
            for (h, &a) in self.x.iter_mut().zip(&self.sub) {
                *h += a;
            }

            // MLP sublayer (dense for the first layers, MoE after).
            let post_ln = self
                .snap
                .require(&format!("{lb}.post_attention_layernorm.weight"))?;
            rmsnorm_into_bytes(&mut self.xn, &self.x, post_ln, eps);
            if layer < cfg.dense_layers {
                dense_mlp(
                    self.snap,
                    cfg,
                    layer,
                    &self.xn,
                    &mut self.mscr,
                    &mut self.sub,
                )?;
            } else {
                moe_block(
                    self.snap,
                    cfg,
                    layer,
                    &self.xn,
                    &mut self.mscr,
                    &mut self.sub,
                )?;
            }
            for (h, &m) in self.x.iter_mut().zip(&self.sub) {
                *h += m;
            }
        }

        // Final norm (into xn; x no longer needed) + lm_head → logits.
        let norm = self.snap.require("model.norm.weight")?;
        rmsnorm_into_bytes(&mut self.xn, &self.x, norm, eps);
        let head = self.snap.int8("lm_head", cfg.hidden)?;
        matvec_i8(&mut self.logits, &self.xn, &head);
        Ok(())
    }

    /// Greedy argmax over the current logits; errors if the winner is
    /// non-finite (a NaN anywhere in the forward pass would otherwise silently
    /// return token 0, i.e. coherent-looking garbage).
    fn argmax(&self) -> Result<u32> {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &l) in self.logits.iter().enumerate() {
            if l > bv {
                bv = l;
                best = i;
            }
        }
        if !bv.is_finite() {
            bail!("logits are non-finite (NaN/Inf in the forward pass)");
        }
        Ok(best as u32)
    }

    /// Greedy-decode up to `ngen` tokens continuing `prompt_ids`, stopping early
    /// on any `eos` id. Returns the generated ids (the caller detokenizes).
    pub fn generate(&mut self, prompt_ids: &[u32], ngen: usize, eos: &[u32]) -> Result<Vec<u32>> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        let mut pos = 0usize;
        // Prefill: run every prompt token; the last one's logits start decode.
        for &tok in prompt_ids {
            self.forward(tok, pos)?;
            pos += 1;
        }
        let mut generated = Vec::with_capacity(ngen);
        for _ in 0..ngen {
            let next = self.argmax()?;
            if eos.contains(&next) {
                break;
            }
            generated.push(next);
            self.forward(next, pos)?;
            pos += 1;
        }
        Ok(generated)
    }
}
