//! The MLP path: dense SwiGLU (first `dense_layers` layers) and the
//! sigmoid-gated MoE block (all others). Reference scalar implementation — the
//! oracle the fused HIP kernel is validated against (M2). Weights are read
//! on-demand from the snapshot via [`Snapshot::int4`]; the pin/streaming feed
//! that keeps them resident is a later milestone.
//!
//! MoE gating matches colibri/GLM-5.2 exactly (glm.c): score = sigmoid(logit);
//! the correction bias is added only for SELECTION; the routing weight is the
//! original sigmoid score; selected weights are sum-normalized (norm_topk_prob)
//! then multiplied by routed_scaling_factor. The shared expert always applies.

use crate::math::{silu, topk_into};
use crate::model::ModelConfig;
use crate::quant::{matvec_f32_bytes, matvec_i4, read_f32};
use crate::snapshot::Snapshot;
use anyhow::{Result, ensure};

/// Upper bound on top-k so the routed-expert loop uses stack arrays (avoids
/// re-borrowing `MlpScratch` while `swiglu_accum` mutates it). GLM top-k = 8.
const MAX_TOPK: usize = 32;

/// Scratch buffers reused across layers/tokens so the MLP path allocates once —
/// the SwiGLU intermediates and the MoE router working set.
pub struct MlpScratch {
    gate: Vec<f32>, // intermediate (dense=12288 or moe=2048)
    up: Vec<f32>,
    expert_out: Vec<f32>, // hidden
    logits: Vec<f32>,     // n_experts (router gate output = scores in place)
    choice: Vec<f32>,     // n_experts (scores + bias, for selection)
    sel: Vec<usize>,      // top-k index buffer
}

impl MlpScratch {
    pub fn new(cfg: &ModelConfig) -> Self {
        let max_inter = cfg.dense_inter.max(cfg.moe_inter * cfg.n_shared);
        Self {
            gate: vec![0.0; max_inter],
            up: vec![0.0; max_inter],
            expert_out: vec![0.0; cfg.hidden],
            logits: vec![0.0; cfg.n_experts],
            choice: vec![0.0; cfg.n_experts],
            sel: Vec::with_capacity(cfg.n_experts),
        }
    }
}

/// One SwiGLU expert/MLP: `out += weight * down(silu(gate·x) ⊙ up·x)`, reading
/// the three int4 projections at `base` (`<base>.gate_proj` etc). `inter` is
/// the intermediate width. Accumulates into `out` scaled by `weight`.
fn swiglu_accum(
    snap: &Snapshot,
    base: &str,
    x: &[f32],
    inter: usize,
    weight: f32,
    scratch: &mut MlpScratch,
    out: &mut [f32],
) -> Result<()> {
    let hidden = x.len();
    // gate/up project hidden→inter (i_dim=hidden); down projects inter→hidden.
    let gate_w = snap.int4(&format!("{base}.gate_proj"), hidden)?;
    let up_w = snap.int4(&format!("{base}.up_proj"), hidden)?;
    let down_w = snap.int4(&format!("{base}.down_proj"), inter)?;
    // Widths must match the scratch slices; a mismatch would silently truncate
    // via zip in release, so fail loudly at the boundary instead.
    anyhow::ensure!(
        gate_w.o_dim == inter && up_w.o_dim == inter && down_w.o_dim == hidden,
        "{base}: projection width mismatch (gate {} up {} down {} vs inter {inter} hidden {hidden})",
        gate_w.o_dim,
        up_w.o_dim,
        down_w.o_dim
    );
    let g = &mut scratch.gate[..inter];
    let u = &mut scratch.up[..inter];
    matvec_i4(g, x, &gate_w);
    matvec_i4(u, x, &up_w);
    for (gi, &ui) in g.iter_mut().zip(u.iter()) {
        *gi = silu(*gi) * ui;
    }
    matvec_i4(&mut scratch.expert_out, g, &down_w);
    for (o, &e) in out.iter_mut().zip(scratch.expert_out.iter()) {
        *o += weight * e;
    }
    Ok(())
}

/// Dense SwiGLU MLP for a pre-MoE layer (`layer < dense_layers`). Writes the
/// full hidden output into `out`.
pub fn dense_mlp(
    snap: &Snapshot,
    cfg: &ModelConfig,
    layer: usize,
    x: &[f32],
    scratch: &mut MlpScratch,
    out: &mut [f32],
) -> Result<()> {
    out.fill(0.0);
    let base = format!("model.layers.{layer}.mlp");
    swiglu_accum(snap, &base, x, cfg.dense_inter, 1.0, scratch, out)
}

/// Sigmoid-gated MoE block for a layer (`layer >= dense_layers`): routed top-k
/// experts + the shared expert. Writes the full hidden output into `out`.
pub fn moe_block(
    snap: &Snapshot,
    cfg: &ModelConfig,
    layer: usize,
    x: &[f32],
    scratch: &mut MlpScratch,
    out: &mut [f32],
) -> Result<()> {
    out.fill(0.0);
    let lbase = format!("model.layers.{layer}.mlp");

    // Router gate (F32) → sigmoid scores (in scratch.logits) + correction bias
    // for selection (scratch.choice). Gate read inline from mmap bytes.
    ensure!(
        cfg.top_k <= MAX_TOPK,
        "top_k {} exceeds MAX_TOPK",
        cfg.top_k
    );
    let gate_bytes = snap.require(&format!("{lbase}.gate.weight"))?;
    let bias = read_f32(snap.require(&format!("{lbase}.gate.e_score_correction_bias"))?);
    ensure!(
        bias.len() == cfg.n_experts,
        "{lbase}.gate.e_score_correction_bias has {} entries, expected {}",
        bias.len(),
        cfg.n_experts
    );
    matvec_f32_bytes(&mut scratch.logits, x, gate_bytes, cfg.hidden);
    for l in scratch.logits.iter_mut() {
        *l = crate::math::sigmoid(*l); // logits now hold sigmoid SCORES
    }
    for ((c, &s), &b) in scratch.choice.iter_mut().zip(&scratch.logits).zip(&bias) {
        *c = s + b;
    }
    topk_into(&scratch.choice, cfg.top_k, &mut scratch.sel);

    // Snapshot the selection + weights into stack arrays so the expert loop
    // doesn't re-borrow scratch while swiglu_accum mutates it. Weight is the
    // ORIGINAL sigmoid score; sum-normalize (norm_topk_prob) then scale.
    let ke = scratch.sel.len();
    let mut sel = [0usize; MAX_TOPK];
    let mut w = [0.0f32; MAX_TOPK];
    let mut sm = 0.0f32;
    for (i, &e) in scratch.sel.iter().enumerate() {
        sel[i] = e;
        w[i] = scratch.logits[e];
        sm += w[i];
    }
    if cfg.norm_topk_prob {
        sm += 1e-20;
        for wi in w[..ke].iter_mut() {
            *wi /= sm;
        }
    }
    for wi in w[..ke].iter_mut() {
        *wi *= cfg.routed_scale as f32;
    }

    for i in 0..ke {
        let base = format!("{lbase}.experts.{}", sel[i]);
        swiglu_accum(snap, &base, x, cfg.moe_inter, w[i], scratch, out)?;
    }
    // Shared expert(s), always on (weight 1). The shared MLP's intermediate
    // width is moe_inter × n_shared (HF Glm4MoeMLP / colibri glm.c).
    let shared = format!("{lbase}.shared_experts");
    swiglu_accum(
        snap,
        &shared,
        x,
        cfg.moe_inter * cfg.n_shared,
        1.0,
        scratch,
        out,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::math::topk;

    /// The routing selection/weighting math in isolation (no snapshot): sigmoid
    /// scores, bias only for selection, weight = original score, norm + scale.
    #[test]
    fn routing_uses_score_not_choice_for_weight() {
        // 4 experts, top-2. Bias lifts expert 3 into selection though its score
        // is low, but its WEIGHT stays its (low) sigmoid score.
        let scores = [0.9f32, 0.1, 0.2, 0.3];
        let bias = [0.0f32, 0.0, 0.0, 1.0];
        let choice: Vec<f32> = scores.iter().zip(&bias).map(|(&s, &b)| s + b).collect();
        let sel = topk(&choice, 2);
        assert_eq!(sel, vec![3, 0]); // choice: [0.9,0.1,0.2,1.3] → 3 then 0
        let mut w: Vec<f32> = sel.iter().map(|&e| scores[e]).collect();
        assert!((w[0] - 0.3).abs() < 1e-6); // expert 3's weight is its score, not choice
        let sm: f32 = w.iter().sum();
        for wi in w.iter_mut() {
            *wi /= sm;
        }
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
