//! The decode loop — the M1 gate. Assembles the forward pass:
//!   embed → per layer [ rmsnorm(input_ln) → attn → +residual
//!                       → rmsnorm(post_ln) → mlp → +residual ]
//!         → rmsnorm(model.norm) → lm_head → greedy argmax.
//! Reference scalar path with on-demand weight reads (the pin/streaming feed and
//! the fused GPU kernels are later milestones). Greedy sampling for now — the
//! M1 gate is coherence, not sampling strategy.

use crate::attn::{AttnScratch, KvCache, attention};
use crate::math::rmsnorm;
use crate::model::ModelConfig;
use crate::moe::{MlpScratch, dense_mlp, moe_block};
use crate::quant::{dequant_int8_row, matvec_i8, read_f32};
use crate::snapshot::Snapshot;
use anyhow::Result;

pub struct Engine<'a> {
    snap: &'a Snapshot,
    cfg: &'a ModelConfig,
    kv: KvCache,
    ascr: AttnScratch,
    mscr: MlpScratch,
    x: Vec<f32>,      // residual stream (hidden)
    xn: Vec<f32>,     // normed input to a sublayer (hidden)
    sub: Vec<f32>,    // sublayer output (hidden)
    logits: Vec<f32>, // vocab
}

impl<'a> Engine<'a> {
    pub fn new(snap: &'a Snapshot, cfg: &'a ModelConfig) -> Self {
        let max_inter = cfg.dense_inter.max(cfg.moe_inter * cfg.n_shared);
        Self {
            snap,
            cfg,
            kv: KvCache::new(cfg),
            ascr: AttnScratch::new(cfg),
            mscr: MlpScratch::new(cfg.hidden, max_inter),
            x: vec![0.0; cfg.hidden],
            xn: vec![0.0; cfg.hidden],
            sub: vec![0.0; cfg.hidden],
            logits: vec![0.0; cfg.vocab],
        }
    }

    /// One forward pass for `token` at `pos`, leaving next-token logits in
    /// `self.logits`.
    fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let eps = cfg.rms_norm_eps as f32;

        // Embedding (int8 table).
        let eb = self.snap.require("model.embed_tokens.weight")?;
        let es = self.snap.require("model.embed_tokens.weight.qs")?;
        dequant_int8_row(eb, es, token as usize, &mut self.x);

        for layer in 0..cfg.n_layers {
            let lb = format!("model.layers.{layer}");
            // Attention sublayer.
            let in_ln = read_f32(self.snap.require(&format!("{lb}.input_layernorm.weight"))?);
            self.xn.copy_from_slice(&self.x);
            rmsnorm(&mut self.xn, &in_ln, eps);
            attention(
                self.snap,
                cfg,
                layer,
                &self.xn,
                pos,
                &mut self.kv,
                &mut self.ascr,
                &mut self.sub,
            )?;
            for (h, &a) in self.x.iter_mut().zip(&self.sub) {
                *h += a;
            }

            // MLP sublayer (dense for the first layers, MoE after).
            let post_ln = read_f32(
                self.snap
                    .require(&format!("{lb}.post_attention_layernorm.weight"))?,
            );
            self.xn.copy_from_slice(&self.x);
            rmsnorm(&mut self.xn, &post_ln, eps);
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

        // Final norm + lm_head → logits.
        let norm = read_f32(self.snap.require("model.norm.weight")?);
        rmsnorm(&mut self.x, &norm, eps);
        let hb = self.snap.require("lm_head.weight")?;
        let hs = self.snap.require("lm_head.weight.qs")?;
        matvec_i8(&mut self.logits, &self.x, hb, hs, cfg.hidden);
        Ok(())
    }

    fn argmax(&self) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &l) in self.logits.iter().enumerate() {
            if l > bv {
                bv = l;
                best = i;
            }
        }
        best as u32
    }

    /// Greedy-decode `ngen` tokens continuing `prompt_ids`. Calls `emit` with
    /// each generated id in order; stops early on `is_eos`. Returns the ids.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        ngen: usize,
        is_eos: &dyn Fn(u32) -> bool,
        mut emit: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let mut pos = 0usize;
        // Prefill: run every prompt token; the last one's logits start decode.
        for &tok in prompt_ids {
            self.forward(tok, pos)?;
            pos += 1;
        }
        let mut generated = Vec::with_capacity(ngen);
        for _ in 0..ngen {
            let next = self.argmax();
            if is_eos(next) {
                break;
            }
            generated.push(next);
            emit(next);
            self.forward(next, pos)?;
            pos += 1;
        }
        Ok(generated)
    }
}
