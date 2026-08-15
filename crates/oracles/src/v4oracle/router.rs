//! `Gate.forward` — the routing decision alone: `sqrt(softplus(·))` scoring, the
//! load-balancing bias that shifts SELECTION and not the weights, the hash-layer bypass, and
//! the renormalisation that closes it.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). The cut from [`crate::v4oracle::moe`] is the
//! reference's own class boundary: `Gate` is a separate `nn.Module` that `MoE.forward` calls
//! once and then never consults again — it hands back `(weights, indices)` and every
//! remaining stage reads only those. `softmax_row` came with it because
//! `Defect::RouterSoftmax` is its only reader.
//!
//! Every body moved unchanged. This is a frozen transliteration — see `forward.rs`'s module
//! doc for what is reproduced exactly, what is reproduced only up to summation order, and
//! what is out of scope; all of it still governs this file. [`Oracle::gate`] keeps its `pub`
//! visibility; it is a method on `Oracle`, so every existing call site resolves unchanged.

use crate::v4oracle::breakages::Defect;
use crate::v4oracle::capture::Counters;
use crate::v4oracle::forward::{Oracle, topk_idx};
use crate::v4oracle::layer::LayerCtx;
use crate::v4oracle::numerics::softplus;

/// `row.softmax(-1)` into `dst`, fp32, max-shifted. Contiguous and out-of-place, which is
/// why it is not [`softmax_strided`]: the router reads its logits and writes a separate
/// score buffer, and only `Defect::RouterSoftmax` reaches it at all.
fn softmax_row(row: &[f32], dst: &mut [f32]) {
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for (i, &v) in row.iter().enumerate() {
        let e = (v - mx).exp();
        dst[i] = e;
        s += e;
    }
    for v in dst.iter_mut() {
        *v /= s;
    }
}

/// `(weights, indices)`, both `[m, n_activated_experts]` — what `Gate.forward` returns.
type RouterPick = (Vec<f32>, Vec<usize>);

impl Oracle {
    /// `Gate.forward`. Returns `(weights, indices)`, both `[m, n_activated_experts]`.
    ///
    /// Two things here are easy to lose and impossible to see afterwards:
    /// - the load-balancing `bias` shifts the scores used for SELECTION and is absent from
    ///   the scores used as WEIGHTS (`original_scores`, model.py:577-585);
    /// - hash layers (`layer_id < n_hash_layers`) take their indices from
    ///   `tid2eid[input_id]` and bypass the scores entirely — but the gate still runs, and
    ///   its scores still become the weights.
    pub fn gate(&self, step: &LayerCtx, x: &[f32], cnt: &mut Counters) -> RouterPick {
        let (lw, m) = (step.lw, step.s);
        let logits = self.linear(x, m, &lw.gate_w);
        let original = self.router_scores(&logits, m, cnt);
        let selection = self.router_selection(&original, lw.gate_bias.as_deref(), m);
        self.router_topk(step, &original, &selection)
    }

    /// `original_scores`: `sqrt(softplus(logits))` per expert, or a plain `softmax` under
    /// `RouterSoftmax`. These are the scores that become the WEIGHTS — the load-balancing
    /// bias is absent from them, which is the half of model.py:577-585 that is easy to lose
    /// and impossible to see afterwards.
    fn router_scores(&self, logits: &[f32], m: usize, counters: &mut Counters) -> Vec<f32> {
        let n = self.cfg.n_routed_experts;
        let mut original = vec![0.0f32; m * n];
        for t in 0..m {
            let row = &logits[t * n..(t + 1) * n];
            let dst = &mut original[t * n..(t + 1) * n];
            if self.defect == Defect::RouterSoftmax {
                softmax_row(row, dst);
            } else {
                self.softplus_row(row, dst, counters);
            }
        }
        original
    }

    /// `sqrt(softplus(·))`, and the overflow census `RouterNoSoftplusThreshold` is gated on.
    /// The `threshold = 20` branch is load-bearing only where `e^x` reaches infinity — see
    /// [`Counters::softplus_overflows`] for why counting logits above 20 is the wrong
    /// instrument.
    fn softplus_row(&self, row: &[f32], dst: &mut [f32], counters: &mut Counters) {
        for (i, &v) in row.iter().enumerate() {
            if v.exp().is_infinite() {
                counters.softplus_overflows += 1;
            }
            let sp = if self.defect == Defect::RouterNoSoftplusThreshold {
                (1.0 + v.exp()).ln()
            } else {
                softplus(v)
            };
            dst[i] = sp.sqrt();
        }
    }

    /// The scores SELECTION reads: `original` shifted by the load-balancing bias on a layer
    /// that has one. A separate buffer, not an in-place shift, because the un-shifted copy
    /// is what the weights come from.
    fn router_selection(&self, original: &[f32], bias: Option<&[f32]>, m: usize) -> Vec<f32> {
        let n = self.cfg.n_routed_experts;
        let mut selection = original.to_vec();
        if let Some(b) = bias {
            for t in 0..m {
                for i in 0..n {
                    selection[t * n + i] += b[i];
                }
            }
        }
        selection
    }

    /// Which experts each row routes to, and with what weight.
    ///
    /// Hash layers (`layer_id < n_hash_layers`) take their indices from `tid2eid[input_id]`
    /// and bypass the scores entirely — but the gate still runs, and its scores still become
    /// the weights.
    fn router_topk(&self, step: &LayerCtx, orig: &[f32], selection: &[f32]) -> RouterPick {
        let (lw, m, input_ids) = (step.lw, step.s, step.input_ids);
        let c = &self.cfg;
        let (k, n) = (c.n_activated_experts, c.n_routed_experts);
        let mut idx = vec![0usize; m * k];
        let mut wts = vec![0.0f32; m * k];
        for t in 0..m {
            let sel: Vec<usize> = match &lw.tid2eid {
                Some(map) if self.defect != Defect::HashRoutingIgnored => {
                    let base = input_ids[t] as usize * k;
                    map[base..base + k].iter().map(|&e| e as usize).collect()
                }
                _ => topk_idx(&selection[t * n..(t + 1) * n], k),
            };
            let src = if self.defect == Defect::RouterBiasedWeights {
                selection
            } else {
                orig
            };
            for (j, &e) in sel.iter().enumerate() {
                idx[t * k + j] = e;
                wts[t * k + j] = src[t * n + e];
            }
            self.router_renorm(&mut wts[t * k..(t + 1) * k]);
        }
        (wts, idx)
    }

    /// `weights /= weights.sum()` then `* route_scale` (model.py:590-593).
    ///
    /// `RouterSoftmax` skips the renormalisation because a softmax row already sums to one
    /// — that is the reference's own `score_func == "softplus"` guard, not a defect of its
    /// own, which is why it shares the arm with `RouterNoRenorm` rather than being one.
    ///
    /// A disabled arm passes 1.0 rather than branching around the loop: `x / 1.0` and
    /// `x * 1.0` are exact in IEEE for every input, so the three defect arms differ from the
    /// clean one by value and never by rounding.
    fn router_renorm(&self, wts: &mut [f32]) {
        let skip = matches!(self.defect, Defect::RouterSoftmax | Defect::RouterNoRenorm);
        let sum: f32 = if skip { 1.0 } else { wts.iter().sum() };
        let scale = if self.defect == Defect::RouterNoScale {
            1.0
        } else {
            self.cfg.route_scale
        };
        for v in wts.iter_mut() {
            *v = *v / sum * scale;
        }
    }
}
