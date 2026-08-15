//! `Expert.forward` and `MoE.forward` — the clamped SwiGLU each expert runs, the gather and
//! scatter that batch the rows one expert owns, and the ascending-expert-id accumulation that
//! combines them with the shared expert.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). `run_layer` calls `moe` and nothing else here; the
//! router it opens with lives next door in [`crate::v4oracle::router`], on the reference's
//! own class boundary — `Gate.forward` returns `(weights, indices)` and this file reads only
//! those.
//!
//! Every body moved unchanged. This is a frozen transliteration — see `forward.rs`'s module
//! doc for what is reproduced exactly, what is reproduced only up to summation order, and
//! what is out of scope; all of it still governs this file. [`Oracle::expert`] keeps its
//! `pub` visibility; it is a method on `Oracle`, so every existing call site resolves
//! unchanged.

use crate::v4oracle::breakages::Defect;
use crate::v4oracle::capture::{Capture, Counters};
use crate::v4oracle::forward::Oracle;
use crate::v4oracle::layer::{ExpertW, LayerCtx};
use crate::v4oracle::numerics::silu;

/// `up`'s clamp, which the reference applies SYMMETRICALLY (model.py:606), and whether the
/// bound bit. The count is EVENTS, not elements: one element contributes here and again in
/// [`Oracle::clamp_gate`], which is what [`Counters::swiglu_clamp_events`] is named for.
fn clamp_up(ui: f32, limit: f32) -> (f32, usize) {
    let hit = usize::from(ui < -limit || ui > limit);
    (ui.clamp(-limit, limit), hit)
}

/// `expert id -> [(row, routing weight)]`, in ascending expert id.
type ExpertRows = std::collections::BTreeMap<usize, Vec<(usize, f32)>>;

/// One routing weight per ROW, applied across that row's whole width.
///
/// `Expert.forward` applies it to the SwiGLU intermediate, before the bf16 store that
/// precedes `w2`; `RouteWeightAfterW2` moves the same multiply to the output. The two are
/// identical in exact arithmetic and not identical here — which is why one function serves
/// both sites rather than each spelling its own loop.
fn scale_rows(v: &mut [f32], m: usize, w: &[f32]) {
    let width = v.len() / m;
    for t in 0..m {
        for i in 0..width {
            v[t * width + i] *= w[t];
        }
    }
}

/// The rows each expert must run, keyed by expert id.
///
/// A `BTreeMap` because `MoE.forward` accumulates in ASCENDING EXPERT ID, and re-ordering a
/// 7-term f32 sum is one more thing a consumer would have to allow for.
fn rows_by_expert(wts: &[f32], idx: &[usize], m: usize, k: usize) -> ExpertRows {
    let mut by_expert = ExpertRows::default();
    for t in 0..m {
        for j in 0..k {
            let row = (t, wts[t * k + j]);
            by_expert.entry(idx[t * k + j]).or_default().push(row);
        }
    }
    by_expert
}

/// One expert's rows gathered into a dense `[rows, dim]` batch, with their routing weights.
/// The reference groups by expert before calling `Expert.forward`, so the gather belongs to
/// `MoE.forward` rather than to the expert.
fn gather_rows(x: &[f32], rows: &[(usize, f32)], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut xs = Vec::with_capacity(rows.len() * dim);
    let mut ws = Vec::with_capacity(rows.len());
    for &(t, w) in rows {
        xs.extend_from_slice(&x[t * dim..(t + 1) * dim]);
        ws.push(w);
    }
    (xs, ws)
}

/// Add one expert's `[rows, dim]` output back into the `[m, dim]` accumulator, at the rows
/// it was gathered from. f32 throughout — the accumulator is rounded once, at the end.
fn scatter_add(y: &mut [f32], rows: &[(usize, f32)], o: &[f32], dim: usize) {
    for (r, &(t, _)) in rows.iter().enumerate() {
        for i in 0..dim {
            y[t * dim + i] += o[r * dim + i];
        }
    }
}
/// One expert call's operand block: the flattened rows, their count, and the routing
/// weight that scales the SwiGLU intermediate (None for the shared expert). Grouped
/// 2026-08-15 — the trio travelled as three loose args through every expert call.
pub struct ExpertOperand<'a> {
    pub x: &'a [f32],
    pub m: usize,
    pub weight: Option<&'a [f32]>,
}

impl Oracle {
    /// `Expert.forward` — SwiGLU with `swiglu_limit = 10.0`.
    ///
    /// The clamp is ASYMMETRIC: `up` is clamped to `[-limit, +limit]` and `gate` only from
    /// above (model.py:606-607). And the routing weight multiplies the SwiGLU intermediate,
    /// before the `.to(bf16)` that precedes `w2` — not the expert's output.
    pub fn expert(
        &self,
        e: &ExpertW,
        rows: ExpertOperand<'_>,
        counters: &mut Counters,
    ) -> Vec<f32> {
        let ExpertOperand { x, m, weight } = rows;
        let _inter = self.cfg.moe_inter_dim; // kept: the reference reads it two lines down in model.py
        let mut g = self.linear(x, m, &e.w1);
        let mut u = self.linear(x, m, &e.w3);
        self.round_bf16(&mut g);
        self.round_bf16(&mut u);
        let mut h = self.swiglu(&g, &u, counters);
        let apply_before = self.defect != Defect::RouteWeightAfterW2;
        if let Some(w) = weight
            && apply_before
        {
            scale_rows(&mut h, m, w);
        }
        self.round_bf16(&mut h);
        let mut out = self.linear(&h, m, &e.w2);
        self.round_bf16(&mut out);
        if let Some(w) = weight
            && !apply_before
        {
            scale_rows(&mut out, m, w);
            self.round_bf16(&mut out);
        }
        out
    }

    /// The SwiGLU proper: `silu(clamp(gate)) * clamp(up)` elementwise (model.py:606-609).
    ///
    /// `SwigluUnclamped` sets `swiglu_limit = 0`, which is how the reference itself spells
    /// "no clamp" — so the bound is tested rather than the branch removed.
    fn swiglu(&self, g: &[f32], u: &[f32], counters: &mut Counters) -> Vec<f32> {
        let limit = match self.defect {
            Defect::SwigluUnclamped => 0.0,
            _ => self.cfg.swiglu_limit,
        };
        let mut h = vec![0.0f32; g.len()];
        for (i, o) in h.iter_mut().enumerate() {
            let (mut gi, mut ui) = (g[i], u[i]);
            if limit > 0.0 {
                let (cu, eu) = clamp_up(ui, limit);
                let (cg, eg) = self.clamp_gate(gi, limit);
                counters.swiglu_clamp_events += eu + eg;
                ui = cu;
                gi = cg;
            }
            *o = silu(gi) * ui;
        }
        h
    }

    /// `gate`'s clamp, which the reference applies from ABOVE only (model.py:607), and
    /// whether the bound bit. `SwigluClampGateBothSides` is the symmetric mistake: it
    /// borrows `up`'s rule, which is one line above it in the reference.
    fn clamp_gate(&self, gi: f32, limit: f32) -> (f32, usize) {
        let both = self.defect == Defect::SwigluClampGateBothSides;
        let hit = if both {
            gi > limit || gi < -limit
        } else {
            gi > limit
        };
        let out = if both {
            gi.clamp(-limit, limit)
        } else {
            gi.min(limit)
        };
        (out, usize::from(hit))
    }

    /// `MoE.forward`. Accumulates in f32 in ASCENDING EXPERT ID, then adds the shared
    /// expert last — the reference's order, kept because it is free to keep and re-ordering
    /// a 7-term f32 sum is one more thing a consumer would have to allow for.
    pub(super) fn moe(&self, step: &LayerCtx, x: &[f32], cap: &mut Capture) -> Vec<f32> {
        let LayerCtx { lw, s: m, .. } = *step;
        let tag = step.tag();
        let c = &self.cfg;
        let k = c.n_activated_experts;
        let (wts, idx) = self.gate(step, x, &mut cap.counters);
        cap.push(&format!("{tag}.router_weights"), &[m, k], wts.clone());
        cap.push_i(
            &format!("{tag}.router_indices"),
            &[m, k],
            idx.iter().map(|&i| i as i64).collect(),
        );

        let mut y = vec![0.0f32; m * c.dim];
        let by_expert = rows_by_expert(&wts, &idx, m, k);
        for (e, rows) in &by_expert {
            let Some(ew) = lw.experts.get(e) else {
                // The driver loads exactly the experts a run reaches; a miss means the
                // caller and the router disagree, which must not be papered over.
                panic!("expert {e} was routed to but not loaded");
            };
            let (xs, ws) = gather_rows(x, rows, c.dim);
            let o = self.expert(
                ew,
                ExpertOperand {
                    x: &xs,
                    m: rows.len(),
                    weight: Some(&ws),
                },
                &mut cap.counters,
            );
            scatter_add(&mut y, rows, &o, c.dim);
        }
        let sw = if self.defect == Defect::SharedExpertWeighted {
            Some(vec![c.route_scale; m])
        } else {
            None
        };
        let sh = self.expert(
            &lw.shared,
            ExpertOperand {
                x,
                m,
                weight: sw.as_deref(),
            },
            &mut cap.counters,
        );
        for i in 0..m * c.dim {
            y[i] += sh[i];
        }
        // `y.type_as(x)`.
        self.round_bf16(&mut y);
        y
    }
}
