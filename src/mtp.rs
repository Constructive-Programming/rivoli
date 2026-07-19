//! MTP (Multi-Token Prediction) scalar draft oracle — the M1 reference for
//! speculative decode. Given the main model's trunk hidden `h` (the residual
//! after the last main layer, pre `model.norm`) and the just-emitted token, it
//! drafts the NEXT-next token via the layer-`n_layers` MTP module (DeepSeek-V3
//! formulation; e=embedding, h=hidden):
//!
//!   h'  = eh_proj( [ enorm(embed(next)) | hnorm(h) ] )   // cat order per DS-V3
//!   h'' = mtp_layer(h')     // input_ln → attn → +res → post_ln → moe → +res
//!   draft = argmax( lm_head( rmsnorm(h'', shared_head.norm) ) )
//!
//! `eh_proj` is bf16 (int4 destroys its dynamic range — see docs/mtp.md); the
//! tied `lm_head`/`embed_tokens` are the main model's. The MTP attention has its
//! own KV (cache index = `n_layers`), maintained across decode steps; it runs
//! Dense here — below the 2048 sparsity threshold that equals the trained
//! index_share top-k. DEFERRED: wire index_share reuse for long-context drafts.
//! Everything but the enorm/hnorm/eh_proj glue is the engine's existing
//! `attention()` / `moe_block()` at layer `n_layers` (a full transformer layer).

use crate::attn::{AttnMode, AttnScratch, KvCache, attention};
use crate::math::{rmsnorm, rmsnorm_into_bytes};
use crate::model::ModelConfig;
use crate::moe::{MlpScratch, moe_block};
use crate::quant::{dequant_int8_row, matvec_bf16, matvec_i8, read_f32};
use crate::snapshot::{Dtype, Snapshot};
use anyhow::{Result, bail};

pub struct Mtp {
    layer: usize, // = cfg.n_layers (the MTP layer index, 78 for GLM-5.2)
    kv: KvCache,  // sized n_layers+1; the MTP attention uses index `layer`
    ascr: AttnScratch,
    mscr: MlpScratch,
    enorm: Vec<f32>,  // RMSNorm on the next-token embedding
    hnorm: Vec<f32>,  // RMSNorm on the trunk hidden
    shnorm: Vec<f32>, // pre-head RMSNorm (shared_head.norm)
    concat: Vec<f32>, // 2*hidden: [enorm(emb) | hnorm(h)]
    x: Vec<f32>,      // MTP residual stream (hidden)
    xn: Vec<f32>,     // normed sublayer input
    sub: Vec<f32>,    // sublayer output
    logits: Vec<f32>, // draft logits (vocab)
}

impl Mtp {
    pub fn new(snap: &Snapshot, cfg: &ModelConfig) -> Result<Self> {
        let l = cfg.n_layers;
        let b = format!("model.layers.{l}");
        Ok(Self {
            layer: l,
            kv: KvCache::new_n(l + 1, cfg, false),
            ascr: AttnScratch::new(cfg),
            mscr: MlpScratch::new(cfg),
            enorm: read_f32(snap.typed(&format!("{b}.enorm.weight"), Dtype::F32)?),
            hnorm: read_f32(snap.typed(&format!("{b}.hnorm.weight"), Dtype::F32)?),
            shnorm: read_f32(snap.typed(&format!("{b}.shared_head.norm.weight"), Dtype::F32)?),
            concat: vec![0.0; 2 * cfg.hidden],
            x: vec![0.0; cfg.hidden],
            xn: vec![0.0; cfg.hidden],
            sub: vec![0.0; cfg.hidden],
            logits: vec![0.0; cfg.vocab],
        })
    }

    /// Draft token t+2 from the trunk hidden `h` (which predicted t+1) and the
    /// token just emitted (t+1). `pos` is the MTP layer's current cached-token
    /// count (== the main model's `pos` for this step); the MTP attention
    /// appends its KV at `pos` and attends over `0..=pos`. Returns the greedy
    /// draft.
    pub fn draft(
        &mut self,
        snap: &Snapshot,
        cfg: &ModelConfig,
        h: &[f32],
        next_token: u32,
        pos: usize,
    ) -> Result<u32> {
        let hidden = cfg.hidden;
        let eps = cfg.rms_norm_eps as f32;

        // h' = eh_proj( [ enorm(embed(next)) | hnorm(h) ] )
        let embed = snap.int8("model.embed_tokens", hidden)?;
        dequant_int8_row(&embed, next_token as usize, &mut self.concat[..hidden]);
        rmsnorm(&mut self.concat[..hidden], &self.enorm, eps);
        self.concat[hidden..].copy_from_slice(h);
        rmsnorm(&mut self.concat[hidden..], &self.hnorm, eps);
        let eh = snap.bf16(
            &format!("model.layers.{}.eh_proj.weight", self.layer),
            2 * hidden,
        )?;
        matvec_bf16(&mut self.x, &self.concat, &eh); // x = h'  [hidden]

        // MTP transformer layer on x (input_ln → attn → +res → post_ln → moe → +res).
        let b = format!("model.layers.{}", self.layer);
        let in_ln = snap.typed(&format!("{b}.input_layernorm.weight"), Dtype::F32)?;
        rmsnorm_into_bytes(&mut self.xn, &self.x, in_ln, eps);
        attention(
            snap,
            cfg,
            self.layer,
            &self.xn,
            pos,
            &AttnMode::Dense,
            None,
            &mut self.kv,
            &mut self.ascr,
            &mut self.sub,
        )?;
        for (xi, &a) in self.x.iter_mut().zip(&self.sub) {
            *xi += a;
        }
        let post_ln = snap.typed(&format!("{b}.post_attention_layernorm.weight"), Dtype::F32)?;
        rmsnorm_into_bytes(&mut self.xn, &self.x, post_ln, eps);
        moe_block(
            snap,
            cfg,
            self.layer,
            &self.xn,
            &mut self.mscr,
            &mut self.sub,
        )?;
        for (xi, &m) in self.x.iter_mut().zip(&self.sub) {
            *xi += m;
        }

        // draft = argmax( lm_head( rmsnorm(h'', shared_head.norm) ) )  (tied head)
        self.xn.copy_from_slice(&self.x);
        rmsnorm(&mut self.xn, &self.shnorm, eps);
        let head = snap.int8("lm_head", hidden)?;
        matvec_i8(&mut self.logits, &self.xn, &head);
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in self.logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        if !bv.is_finite() {
            bail!("MTP draft logits non-finite (NaN/Inf in the MTP forward)");
        }
        Ok(best as u32)
    }
}
