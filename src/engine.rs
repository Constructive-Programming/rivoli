//! The decode loop — the M1 gate. Assembles the forward pass:
//!   embed → per layer [ rmsnorm(input_ln) → attn → +residual
//!                       → rmsnorm(post_ln) → mlp → +residual ]
//!         → rmsnorm(model.norm) → lm_head → greedy argmax.
//! Reference scalar path with on-demand weight reads (the pin/streaming feed and
//! the fused GPU kernels are later milestones). Greedy sampling for now — the
//! M1 gate is coherence, not sampling strategy.

use crate::attn::{AttnMode, AttnScratch, AttnWeights, KvCache, attention};
use crate::indexer::Indexer;
use crate::math::rmsnorm_into_bytes;
use crate::model::ModelConfig;
use crate::moe::{MlpScratch, MlpWeights, MoeWeights, dense_mlp, moe_block};
use crate::quant::{dequant_int8_row, matvec_i8};
use crate::snapshot::{Dtype, Int8Matrix, Snapshot};
use anyhow::{Result, bail, ensure};

/// One layer's MLP weights: dense for the first `dense_layers`, MoE after.
enum LayerMlp<'a> {
    Dense(MlpWeights<'a>),
    Moe(MoeWeights<'a>),
}

/// One layer's resolved weights — everything `forward` reads, located and
/// validated once at [`Engine::new`] so the per-token path does no name
/// building or index probes (the old DEFERRED P3).
struct LayerWeights<'a> {
    input_ln: &'a [u8],
    post_ln: &'a [u8],
    attn: AttnWeights<'a>,
    mlp: LayerMlp<'a>,
}

pub struct Engine<'a> {
    cfg: &'a ModelConfig,
    mode: AttnMode,
    indexer: Option<Indexer<'a>>,
    layers: Vec<LayerWeights<'a>>,
    embed: Int8Matrix<'a>,
    lm_head: Int8Matrix<'a>,
    final_norm: &'a [u8],
    kv: KvCache,
    ascr: AttnScratch,
    mscr: MlpScratch,
    x: Vec<f32>,      // residual stream (hidden)
    xn: Vec<f32>,     // normed input to a sublayer (hidden)
    sub: Vec<f32>,    // sublayer output (hidden)
    logits: Vec<f32>, // vocab
}

impl<'a> Engine<'a> {
    /// Fails at construction (not mid-decode) when the snapshot lacks any
    /// weight the forward pass reads (incl. the out-idx shard for sparse
    /// modes) — every tensor is located and shape-validated here.
    pub fn new(
        snap: &'a Snapshot,
        cfg: &'a ModelConfig,
        mode: AttnMode,
        kv_fp8: bool,
    ) -> Result<Self> {
        let indexer = match mode {
            AttnMode::Dsa | AttnMode::Misa { .. } => Some(Indexer::new(snap, cfg)?),
            AttnMode::Dense | AttnMode::Streaming { .. } => None,
        };
        let layers = (0..cfg.n_layers)
            .map(|layer| {
                let lb = format!("model.layers.{layer}");
                Ok(LayerWeights {
                    input_ln: snap.typed(&format!("{lb}.input_layernorm.weight"), Dtype::F32)?,
                    post_ln: snap
                        .typed(&format!("{lb}.post_attention_layernorm.weight"), Dtype::F32)?,
                    attn: AttnWeights::load(snap, cfg, layer)?,
                    mlp: if layer < cfg.dense_layers {
                        LayerMlp::Dense(MlpWeights::load(
                            snap,
                            &format!("{lb}.mlp"),
                            cfg.hidden,
                            cfg.dense_inter,
                        )?)
                    } else {
                        LayerMlp::Moe(MoeWeights::load(snap, cfg, layer)?)
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg,
            mode,
            indexer,
            layers,
            embed: snap.int8("model.embed_tokens", cfg.hidden)?,
            lm_head: snap.int8("lm_head", cfg.hidden)?,
            final_norm: snap.typed("model.norm.weight", Dtype::F32)?,
            kv: KvCache::new(cfg, kv_fp8),
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

        // Embedding (int8 table, validated at construction).
        dequant_int8_row(&self.embed, token as usize, &mut self.x);

        for (layer, lw) in self.layers.iter().enumerate() {
            // Attention sublayer: rmsnorm(input_ln) into xn (residual x survives).
            rmsnorm_into_bytes(&mut self.xn, &self.x, lw.input_ln, eps);
            attention(
                &lw.attn,
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
            rmsnorm_into_bytes(&mut self.xn, &self.x, lw.post_ln, eps);
            match &lw.mlp {
                LayerMlp::Dense(w) => dense_mlp(w, &self.xn, &mut self.mscr, &mut self.sub),
                LayerMlp::Moe(w) => moe_block(cfg, w, &self.xn, &mut self.mscr, &mut self.sub)?,
            }
            for (h, &m) in self.x.iter_mut().zip(&self.sub) {
                *h += m;
            }
        }

        // Final norm (into xn; x no longer needed) + lm_head → logits.
        rmsnorm_into_bytes(&mut self.xn, &self.x, self.final_norm, eps);
        matvec_i8(&mut self.logits, &self.xn, &self.lm_head);
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
