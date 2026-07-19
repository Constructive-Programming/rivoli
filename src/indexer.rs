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
//! MISA (block-pooled head routing, arXiv 2605.07363) is a restriction of the
//! head sum to a routed subset: a per-1024-token running-mean key pool feeds a
//! cheap per-head estimate E_j = mean_b |w_j·ReLU(q_j·k̄_b)|, and only the
//! top-`active_heads` run the O(nt) token scan (~4× cheaper at h=8 of 32; the
//! routed selection's IoU vs full DSA is asserted in tests/attn_modes.rs).

use crate::attn::rope_interleave;
use crate::math::{layernorm, topk_into};
use crate::model::ModelConfig;
use crate::quant::{matvec_bf16, read_bf16};
use crate::snapshot::{Dtype, Snapshot};
use anyhow::{Result, ensure};

/// MISA router block size (pooled tokens per key). The paper's validated
/// setting — an order of magnitude coarser than HISA's, because the router
/// only decides which HEADS matter, not which regions survive.
pub const MISA_BLOCK: usize = 1024;

/// Fold token `t`'s key into its block's running mean pool (`pooled` holds
/// ⌈(t+1)/B⌉ rows of `hd` f32 after the call). Block b covers tokens
/// [b·B, (b+1)·B); a partial tail block is the mean of what it holds so far.
fn pool_push(pooled: &mut Vec<f32>, k: &[f32], t: usize, hd: usize) {
    let in_block = t % MISA_BLOCK;
    if in_block == 0 {
        pooled.extend_from_slice(k);
        return;
    }
    let b = t / MISA_BLOCK;
    let m = &mut pooled[b * hd..(b + 1) * hd];
    let inv = 1.0 / (in_block + 1) as f32;
    for (mi, &ki) in m.iter_mut().zip(k) {
        *mi += (ki - *mi) * inv;
    }
}

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
    /// Per layer, MISA block-pooled keys: ⌈tokens/MISA_BLOCK⌉ rows of
    /// index_head_dim f32 (running mean). MISA-only bookkeeping — the engine's
    /// mode is fixed at construction, so DSA runs never maintain it. Empty for
    /// shared layers.
    pooled: Vec<Vec<f32>>,
    /// Per full layer, the k_norm LayerNorm weight+bias, widened once at
    /// construction (they never change; loading them per token was the only
    /// per-step heap allocation in the indexer path). None for shared layers.
    knorm: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    // Scratch (allocation-free per step).
    k: Vec<f32>,      // index_head_dim
    kf: Vec<f32>,     // index_head_dim — one widened cached key row
    q: Vec<f32>,      // index_n_heads * index_head_dim
    w: Vec<f32>,      // index_n_heads
    scores: Vec<f32>, // up to context length
    picks: Vec<usize>,
    head_scores: Vec<f32>, // index_n_heads router estimates E_j
    heads: Vec<usize>,     // routed active head set
}

/// The indexer's `k_norm` epsilon. Hardcoded in the HF reference
/// (`nn.LayerNorm(head_dim, eps=1e-6)` in modeling_deepseek_v32/glm_moe_dsa) —
/// NOT the model's rms_norm_eps (1e-5 for GLM-5.2).
pub const K_NORM_EPS: f32 = 1e-6;

impl Indexer {
    pub fn new(snap: &Snapshot, cfg: &ModelConfig) -> Result<Self> {
        let full = cfg.indexer_layout()?;
        let knorm = full
            .iter()
            .enumerate()
            .map(|(layer, &is_full)| {
                if !is_full {
                    return Ok(None);
                }
                let base = format!("model.layers.{layer}.self_attn.indexer.k_norm");
                let w = read_bf16(snap.typed(&format!("{base}.weight"), Dtype::Bf16)?);
                let b = read_bf16(snap.typed(&format!("{base}.bias"), Dtype::Bf16)?);
                ensure!(
                    w.len() == cfg.index_head_dim && b.len() == cfg.index_head_dim,
                    "layer {layer} k_norm dims {}/{} != {}",
                    w.len(),
                    b.len(),
                    cfg.index_head_dim
                );
                Ok(Some((w, b)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            kcache: vec![Vec::new(); full.len()],
            pooled: vec![Vec::new(); full.len()],
            knorm,
            full,
            topk: Vec::new(),
            k: vec![0.0; cfg.index_head_dim],
            kf: vec![0.0; cfg.index_head_dim],
            q: vec![0.0; cfg.index_n_heads * cfg.index_head_dim],
            w: vec![0.0; cfg.index_n_heads],
            scores: Vec::new(),
            picks: Vec::new(),
            head_scores: vec![0.0; cfg.index_n_heads],
            heads: Vec::new(),
        })
    }

    /// Compute this layer's token selection for the current step and write it
    /// (ascending rows) into `out`. `x` is the attention input (post
    /// input_layernorm), `qr` the main path's normed q-LoRA residual, `pos` the
    /// current position (== cached tokens before this step's append).
    /// `active_heads` = Some(h): MISA — route h of the index_n_heads via the
    /// block-pooled scorer and let only those heads run the token scan;
    /// None: full DSA (all heads).
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
        active_heads: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if let Some(h) = active_heads {
            ensure!(h > 0, "misa active_heads must be >= 1");
        }
        if !self.full[layer] {
            // IndexShare: reuse the nearest preceding full layer's selection.
            // All layer caches hold the same token count, so rows transfer.
            // Layer 0 is always full (enforced at config load), so a non-empty
            // topk exists by the time any shared layer runs — unconditionally.
            ensure!(
                !self.topk.is_empty(),
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
        let (kn_w, kn_b) = self.knorm[layer]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("full layer {layer} missing hoisted k_norm"))?;

        // Key for the current token: wk → LayerNorm (eps hardcoded to match
        // the HF reference, see K_NORM_EPS) → RoPE, then cache (bf16). The
        // MISA block pool is maintained only when routing is on — the mode is
        // fixed per engine, so a DSA run never reads it.
        matvec_bf16(&mut self.k, x, &wk);
        layernorm(&mut self.k, kn_w, kn_b, K_NORM_EPS);
        rope_interleave(&mut self.k[..rope], pos, theta);
        self.kcache[layer].extend(self.k.iter().map(|&v| crate::math::f32_to_bf16(v)));
        let nt = self.kcache[layer].len() / hd;
        if active_heads.is_some() {
            pool_push(&mut self.pooled[layer], &self.k, nt - 1, hd);
        }

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

        // Active head set: all of them (DSA), or the MISA-routed top-h. The
        // router (paper Eq. 7-8) estimates each head's contribution to the
        // final score from the block-pooled keys — E_j = mean_b |w_j ·
        // ReLU(q_j·k̄_b)| — and keeps the top h. O(nh·M) with M = ⌈nt/1024⌉,
        // negligible next to the O(h·nt) token scan it gates.
        self.heads.clear();
        match active_heads {
            Some(h) if h < nh => {
                let pooled = &self.pooled[layer];
                let m_blocks = pooled.len() / hd;
                for (j, e) in self.head_scores.iter_mut().enumerate() {
                    let qj = &self.q[j * hd..(j + 1) * hd];
                    let wj = self.w[j];
                    let mut sum = 0.0f32;
                    for b in 0..m_blocks {
                        let kb = &pooled[b * hd..(b + 1) * hd];
                        let dot: f32 = qj.iter().zip(kb).map(|(&a, &c)| a * c).sum();
                        sum += (wj * dot.max(0.0)).abs();
                    }
                    *e = sum / m_blocks.max(1) as f32;
                }
                topk_into(&self.head_scores, h, &mut self.heads);
            }
            _ => self.heads.extend(0..nh),
        }

        // Score every cached token with the active heads:
        // I_t = Σ_{h∈active} w_h · ReLU(q_h·k_t · d^-0.5).
        // Each key row is widened bf16→f32 ONCE into the kf scratch, then
        // reused across heads (the |heads|× re-widening the attn.rs P1 note
        // warns about, avoided here from the start).
        if self.scores.len() < nt {
            self.scores.resize(nt, 0.0);
        }
        for (t, sc) in self.scores[..nt].iter_mut().enumerate() {
            let krow = &self.kcache[layer][t * hd..(t + 1) * hd];
            for (f, &kb) in self.kf.iter_mut().zip(krow) {
                *f = crate::math::bf16_to_f32(kb);
            }
            let mut acc = 0.0f32;
            for &head in &self.heads {
                let wh = self.w[head];
                let qh = &self.q[head * hd..(head + 1) * hd];
                let dot: f32 = qh.iter().zip(&self.kf).map(|(&a, &b)| a * b).sum();
                acc += wh * wscale * (dot * dscale).max(0.0);
            }
            *sc = acc;
        }

        // Plain causal top-k, faithfully mirroring the HF reference
        // (`index_scores.topk(...)`): the current token is NOT force-included —
        // if its own key scores outside the top-k it is dropped, exactly as in
        // the trained model.
        topk_into(&self.scores[..nt], cfg.index_topk, &mut self.picks);
        self.picks.sort_unstable(); // ascending token order for the gather
        out.clear();
        out.extend(self.picks.iter().map(|&i| i as u32));
        self.topk.clone_from(out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_push_running_mean_and_block_boundaries() {
        let hd = 2;
        let mut pooled = Vec::new();
        // Tokens 0..MISA_BLOCK all in block 0; mean of k=[t, 1] is
        // [(B-1)/2, 1].
        for t in 0..MISA_BLOCK {
            pool_push(&mut pooled, &[t as f32, 1.0], t, hd);
        }
        assert_eq!(pooled.len(), hd);
        let want = (MISA_BLOCK - 1) as f32 / 2.0;
        assert!(
            (pooled[0] - want).abs() < want * 1e-4,
            "{} vs {want}",
            pooled[0]
        );
        assert!((pooled[1] - 1.0).abs() < 1e-5);
        // Next token opens block 1 verbatim.
        pool_push(&mut pooled, &[7.0, 9.0], MISA_BLOCK, hd);
        assert_eq!(pooled.len(), 2 * hd);
        assert_eq!(&pooled[hd..], &[7.0, 9.0]);
        // And a second token in block 1 averages in.
        pool_push(&mut pooled, &[9.0, 11.0], MISA_BLOCK + 1, hd);
        assert_eq!(&pooled[hd..], &[8.0, 10.0]);
    }
}
