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
use crate::snapshot::{Dtype, Int4Matrix, Snapshot};
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

/// One SwiGLU MLP's three int4 projections, resolved and width-validated ONCE
/// at engine construction (the old per-token format! + HashMap re-resolution).
pub struct MlpWeights<'a> {
    gate: Int4Matrix<'a>,
    up: Int4Matrix<'a>,
    down: Int4Matrix<'a>,
    inter: usize,
}

impl<'a> MlpWeights<'a> {
    /// Locate + validate the three projections at `base` (`<base>.gate_proj`
    /// etc). `inter` is the intermediate width.
    pub fn load(snap: &'a Snapshot, base: &str, hidden: usize, inter: usize) -> Result<Self> {
        // gate/up project hidden→inter (i_dim=hidden); down projects inter→hidden.
        let gate = snap.int4(&format!("{base}.gate_proj"), hidden)?;
        let up = snap.int4(&format!("{base}.up_proj"), hidden)?;
        let down = snap.int4(&format!("{base}.down_proj"), inter)?;
        // Widths must match the scratch slices; a mismatch would silently truncate
        // via zip in release, so fail loudly at the boundary instead.
        ensure!(
            gate.o_dim == inter && up.o_dim == inter && down.o_dim == hidden,
            "{base}: projection width mismatch (gate {} up {} down {} vs inter {inter} hidden {hidden})",
            gate.o_dim,
            up.o_dim,
            down.o_dim
        );
        Ok(Self {
            gate,
            up,
            down,
            inter,
        })
    }
}

/// One MoE layer's resolved weights: the router gate + bias, all routed
/// experts, and the shared expert.
pub struct MoeWeights<'a> {
    /// Router gate F32 bytes `[n_experts, hidden]` (read inline from mmap).
    gate: &'a [u8],
    bias: Vec<f32>,
    experts: Vec<MlpWeights<'a>>,
    shared: MlpWeights<'a>,
}

impl<'a> MoeWeights<'a> {
    pub fn load(snap: &'a Snapshot, cfg: &ModelConfig, layer: usize) -> Result<Self> {
        let lbase = format!("model.layers.{layer}.mlp");
        let gate = snap.typed(&format!("{lbase}.gate.weight"), Dtype::F32)?;
        let bias =
            read_f32(snap.typed(&format!("{lbase}.gate.e_score_correction_bias"), Dtype::F32)?);
        ensure!(
            bias.len() == cfg.n_experts,
            "{lbase}.gate.e_score_correction_bias has {} entries, expected {}",
            bias.len(),
            cfg.n_experts
        );
        let experts = (0..cfg.n_experts)
            .map(|e| {
                MlpWeights::load(
                    snap,
                    &format!("{lbase}.experts.{e}"),
                    cfg.hidden,
                    cfg.moe_inter,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let shared = MlpWeights::load(
            snap,
            &format!("{lbase}.shared_experts"),
            cfg.hidden,
            cfg.moe_inter * cfg.n_shared,
        )?;
        Ok(Self {
            gate,
            bias,
            experts,
            shared,
        })
    }
}

/// One SwiGLU expert/MLP: `out += weight * down(silu(gate·x) ⊙ up·x)` from the
/// resolved projections `w`. Accumulates into `out` scaled by `weight`.
fn swiglu_accum(w: &MlpWeights, x: &[f32], weight: f32, scratch: &mut MlpScratch, out: &mut [f32]) {
    let g = &mut scratch.gate[..w.inter];
    let u = &mut scratch.up[..w.inter];
    matvec_i4(g, x, &w.gate);
    matvec_i4(u, x, &w.up);
    for (gi, &ui) in g.iter_mut().zip(u.iter()) {
        *gi = silu(*gi) * ui;
    }
    matvec_i4(&mut scratch.expert_out, g, &w.down);
    for (o, &e) in out.iter_mut().zip(scratch.expert_out.iter()) {
        *o += weight * e;
    }
}

/// Dense SwiGLU MLP for a pre-MoE layer (`layer < dense_layers`). Writes the
/// full hidden output into `out`.
pub fn dense_mlp(w: &MlpWeights, x: &[f32], scratch: &mut MlpScratch, out: &mut [f32]) {
    out.fill(0.0);
    swiglu_accum(w, x, 1.0, scratch, out);
}

/// Sigmoid-gated MoE block for a layer (`layer >= dense_layers`): routed top-k
/// experts + the shared expert. Writes the full hidden output into `out`.
pub fn moe_block(
    cfg: &ModelConfig,
    w: &MoeWeights,
    x: &[f32],
    scratch: &mut MlpScratch,
    out: &mut [f32],
) -> Result<()> {
    out.fill(0.0);

    // Router gate (F32) → sigmoid scores (in scratch.logits) + correction bias
    // for selection (scratch.choice). Gate read inline from mmap bytes.
    ensure!(
        cfg.top_k <= MAX_TOPK,
        "top_k {} exceeds MAX_TOPK",
        cfg.top_k
    );
    matvec_f32_bytes(&mut scratch.logits, x, w.gate, cfg.hidden);
    for l in scratch.logits.iter_mut() {
        *l = crate::math::sigmoid(*l); // logits now hold sigmoid SCORES
    }
    for ((c, &s), &b) in scratch.choice.iter_mut().zip(&scratch.logits).zip(&w.bias) {
        *c = s + b;
    }
    topk_into(&scratch.choice, cfg.top_k, &mut scratch.sel);

    // Snapshot the selection + weights into stack arrays so the expert loop
    // doesn't re-borrow scratch while swiglu_accum mutates it. Weight is the
    // ORIGINAL sigmoid score; sum-normalize (norm_topk_prob) then scale.
    let ke = scratch.sel.len();
    let mut sel = [0usize; MAX_TOPK];
    let mut wt = [0.0f32; MAX_TOPK];
    let mut sm = 0.0f32;
    for (i, &e) in scratch.sel.iter().enumerate() {
        sel[i] = e;
        wt[i] = scratch.logits[e];
        sm += wt[i];
    }
    if cfg.norm_topk_prob {
        sm += 1e-20;
        for wi in wt[..ke].iter_mut() {
            *wi /= sm;
        }
    }
    for wi in wt[..ke].iter_mut() {
        *wi *= cfg.routed_scale as f32;
    }

    for i in 0..ke {
        swiglu_accum(&w.experts[sel[i]], x, wt[i], scratch, out);
    }
    // Shared expert(s), always on (weight 1). The shared MLP's intermediate
    // width is moe_inter × n_shared (HF Glm4MoeMLP / colibri glm.c).
    swiglu_accum(&w.shared, x, 1.0, scratch, out);
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
