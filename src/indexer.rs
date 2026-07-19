//! DSA lightning-indexer decode — scalar reference, mirroring the HF
//! `glm_moe_dsa` modeling code. Per *full* layer the indexer keeps one shared
//! 128-dim key per cached token (MQA-style, bf16 here) and scores every past
//! token against 32 low-rank query heads:
//!
//!   k_t   = rope(layernorm(wk·x))                 (first qk_rope dims roped)
//!   q_h   = rope(wq_b·q_resid)[h]                 (reuses the main q-LoRA residual)
//!   w     = weights_proj·x · n_heads^-0.5         (per-head gates, may be negative)
//!   I_t   = Σ_h w_h · ReLU(q_h·k_t · d^-0.5)
//!
//! Top-`index_topk` token rows (ascending) feed the sparse MLA gather. *Shared*
//! layers own no indexer at all (GLM-5.2's trained-in IndexShare) and reuse the
//! nearest preceding full layer's selection — layer 0 is always full, enforced
//! at load. When the cache holds ≤ index_topk tokens the selection is every
//! causal token, i.e. exactly dense — that equivalence is the oracle's test.
//!
//! MISA (block-pooled head routing, arXiv 2605.07363) plugs in here as a
//! restriction of the head sum to a routed subset; it lands as a follow-up.

use crate::attn::rope_interleave;
use crate::math::{layernorm, topk_into};
use crate::model::ModelConfig;
use crate::quant::{matvec_bf16, read_bf16};
use crate::snapshot::Snapshot;
use anyhow::{Result, ensure};

/// Per-full-layer indexer key cache + reusable scratch. One instance per
/// engine; layers index into `kcache` (empty Vec for shared layers).
pub struct Indexer {
    /// Per layer: `true` = full (owns an indexer), `false` = shared.
    full: Vec<bool>,
    /// Per layer, flat bf16 key rows, len = tokens * index_head_dim. Empty for
    /// shared layers. Same DEFERRED growth note as `KvCache` (see attn.rs).
    kcache: Vec<Vec<u16>>,
    /// Most recent full layer's selection this token (token rows, ascending).
    topk: Vec<u32>,
    // Scratch (allocation-free per step).
    k: Vec<f32>,      // index_head_dim
    q: Vec<f32>,      // index_n_heads * index_head_dim
    w: Vec<f32>,      // index_n_heads
    scores: Vec<f32>, // up to context length
    picks: Vec<usize>,
}

impl Indexer {
    pub fn new(cfg: &ModelConfig) -> Result<Self> {
        let full = cfg.indexer_layout()?;
        Ok(Self {
            kcache: vec![Vec::new(); full.len()],
            full,
            topk: Vec::new(),
            k: vec![0.0; cfg.index_head_dim],
            q: vec![0.0; cfg.index_n_heads * cfg.index_head_dim],
            w: vec![0.0; cfg.index_n_heads],
            scores: Vec::new(),
            picks: Vec::new(),
        })
    }

    /// Compute this layer's token selection for the current step and write it
    /// (ascending rows) into `out`. `x` is the attention input (post
    /// input_layernorm), `qr` the main path's normed q-LoRA residual, `pos` the
    /// current position (== cached tokens before this step's append).
    ///
    /// Full layers append this token's indexer key first, so the selection
    /// covers every causal token including the current one.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &mut self,
        snap: &Snapshot,
        cfg: &ModelConfig,
        layer: usize,
        x: &[f32],
        qr: &[f32],
        pos: usize,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if !self.full[layer] {
            // IndexShare: reuse the nearest preceding full layer's selection.
            // All layer caches hold the same token count, so rows transfer.
            ensure!(
                !self.topk.is_empty() || pos == 0,
                "shared layer {layer} selecting before any full layer ran"
            );
            out.clear();
            out.extend_from_slice(&self.topk);
            return Ok(());
        }

        let hd = cfg.index_head_dim;
        let nh = cfg.index_n_heads;
        let rope = cfg.qk_rope_head_dim;
        let theta = cfg.rope_theta();
        let base = format!("model.layers.{layer}.self_attn.indexer");

        // DEFERRED (same P3 as attn.rs): per-token weight re-resolution; the
        // resolved-tensor table lands with the pin milestone.
        let wk = snap.bf16(&format!("{base}.wk.weight"), cfg.hidden)?;
        let wq_b = snap.bf16(&format!("{base}.wq_b.weight"), cfg.q_lora_rank)?;
        let wproj = snap.bf16(&format!("{base}.weights_proj.weight"), cfg.hidden)?;
        ensure!(wk.o_dim == hd, "wk o_dim {} != {hd}", wk.o_dim);
        ensure!(
            wq_b.o_dim == nh * hd,
            "wq_b o_dim {} != {}",
            wq_b.o_dim,
            nh * hd
        );
        ensure!(
            wproj.o_dim == nh,
            "weights_proj o_dim {} != {nh}",
            wproj.o_dim
        );
        let kn_w = read_bf16(snap.require(&format!("{base}.k_norm.weight"))?);
        let kn_b = read_bf16(snap.require(&format!("{base}.k_norm.bias"))?);

        // Key for the current token: wk → LayerNorm → RoPE, then cache (bf16).
        matvec_bf16(&mut self.k, x, &wk);
        layernorm(&mut self.k, &kn_w, &kn_b, cfg.rms_norm_eps as f32);
        rope_interleave(&mut self.k[..rope], pos, theta);
        self.kcache[layer].extend(self.k.iter().map(|&v| crate::math::f32_to_bf16(v)));
        let nt = self.kcache[layer].len() / hd;

        // Everything causal fits in the budget → exactly dense; skip scoring.
        if nt <= cfg.index_topk {
            out.clear();
            out.extend(0..nt as u32);
            self.topk.clone_from(out);
            return Ok(());
        }

        // Query heads (from the shared q-LoRA residual) + per-head gates.
        matvec_bf16(&mut self.q, qr, &wq_b);
        for head in 0..nh {
            let off = head * hd;
            rope_interleave(&mut self.q[off..off + rope], pos, theta);
        }
        matvec_bf16(&mut self.w, x, &wproj);
        let wscale = 1.0 / (nh as f32).sqrt();
        let dscale = 1.0 / (hd as f32).sqrt();

        // Score every cached token: I_t = Σ_h w_h · ReLU(q_h·k_t · d^-0.5).
        if self.scores.len() < nt {
            self.scores.resize(nt, 0.0);
        }
        for (t, sc) in self.scores[..nt].iter_mut().enumerate() {
            let krow = &self.kcache[layer][t * hd..(t + 1) * hd];
            let mut acc = 0.0f32;
            for (head, &wh) in self.w.iter().enumerate() {
                let qh = &self.q[head * hd..(head + 1) * hd];
                let mut dot = 0.0f32;
                for (i, &kb) in krow.iter().enumerate() {
                    dot += qh[i] * crate::math::bf16_to_f32(kb);
                }
                acc += wh * wscale * (dot * dscale).max(0.0);
            }
            *sc = acc;
        }

        topk_into(&self.scores[..nt], cfg.index_topk, &mut self.picks);
        self.picks.sort_unstable(); // ascending token order for the gather
        out.clear();
        out.extend(self.picks.iter().map(|&i| i as u32));
        self.topk.clone_from(out);
        Ok(())
    }
}
