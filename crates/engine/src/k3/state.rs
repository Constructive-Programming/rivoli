//! The per-token state SCHEDULES of the K3 layer loop — which AttnRes sources exist at each
//! layer, and how the router's combining weights are formed. Pure arithmetic, deviceless,
//! because both are places where wrong code runs cleanly (`k3:docs/reference/k3-architecture.md`
//! §3, §10 trap 11) and the featureless build is the one CI compiles.
//!
//! # The residual arena this schedule indexes
//!
//! `engine.rs` keeps one `[res_blocks + 1][hidden]` device arena per token pass. Block
//! snapshots occupy rows `0..stack`, and the running prefix sum ALWAYS lives at row `stack` —
//! so a fold's sources are the contiguous rows `0..=stack` and a push is nothing but
//! `stack += 1`: the prefix row is REINTERPRETED as the newest snapshot and the next prefix
//! write lands one row up. That representation makes "push copies the wrong row" and "the
//! fold skipped the prefix" unwriteable, which is why the schedule below deals only in
//! COUNTS.

use anyhow::{Result, ensure};

/// What the AttnRes machinery does at one layer: the two fold widths and whether a snapshot
/// is pushed between them (`k3:docs/reference/k3-architecture.md` §3's layer loop, restated
/// as data).
#[derive(Clone, Copy)]
pub struct Fold {
    /// Sources of the layer-entry fold (`self_attention_res`): the block stack plus the
    /// prefix sum. `None` at layer 0 only — the stack is empty and the reference GUARDS this
    /// fold, where the mlp fold is unconditional.
    pub entry_sources: Option<usize>,
    /// Whether this layer is a block boundary (`layer % attn_res_block_size == 0`): the
    /// prefix row becomes a snapshot and the prefix restarts as NONE, re-seeded by the
    /// attention output.
    pub push: bool,
    /// Sources of the pre-FFN fold (`mlp_res`) — unconditional, no empty guard.
    pub mlp_sources: usize,
}

/// The schedule at zero-based `layer`.
///
/// The entry stack holds one snapshot per boundary STRICTLY BELOW `layer`, because a
/// boundary layer folds first and pushes after — the order the reference writes out and the
/// one the anchor's captures attest: layer 12's entry fold mixes ONE block (`[1, 1, 192]`)
/// and its mlp fold mixes two (`[1, 2, 192]`).
pub fn fold_at(layer: usize, res_block: usize) -> Fold {
    let stack = layer.div_ceil(res_block);
    let push = layer.is_multiple_of(res_block);
    Fold {
        entry_sources: (stack > 0).then_some(stack + 1),
        push,
        mlp_sources: stack + usize::from(push) + 1,
    }
}

/// Sources of the MODEL-LEVEL fold after the last layer: every snapshot plus the final
/// prefix. Skipping this aggregation is silent (`k3:docs/reference/k3-architecture.md` §7),
/// which is why it is a named function the loop calls and the widths gate counts.
pub fn final_sources(n_layers: usize, res_block: usize) -> usize {
    n_layers.div_ceil(res_block) + 1
}

/// The router's combining weights: the UNBIASED sigmoid scores of the selected experts,
/// renormalised over the selection, times `routed_scale` — trap 11 made a function.
///
/// `scores` and NEVER `choice`: the bias steers SELECTION only, and letting it reach the
/// weights changes every routed magnitude by an amount that reads as ordinary variation
/// (`k3:docs/reference/k3-architecture.md` §6). **The signature does NOT enforce that** —
/// `scores` and `choice` are both `Vec<f32>` of length `n_experts`, so
/// `combine_weights(&self.choice, ..)` compiles and runs wrong; this doc claimed the type
/// argued until review 2026-08-16. The owner is the ONE call site in `forward.rs::moe_ffn`,
/// and no gate sees a regression there (both parity arms would run the same wrong weights),
/// so treat that call as load-bearing text.
///
/// `out` is `[n_experts]` indexed by ABSOLUTE expert id and zero-filled first: the expert
/// kernel skips a zero weight, so the zeros are what make a wrongly-computed launch range
/// add nothing instead of a stale weight. The reference renormalises with a `+ 1e-20` guard;
/// here a non-positive or non-finite sum is a REFUSAL instead — on real weights it means the
/// forward pass already produced garbage, and decoding on through a 1e-20 division is how
/// that garbage becomes fluent text.
pub fn combine_weights(
    scores: &[f32],
    sel: &[usize],
    routed_scale: f32,
    out: &mut [f32],
) -> Result<()> {
    let sum: f32 = sel.iter().map(|&e| scores[e]).sum();
    ensure!(
        sum > 0.0 && sum.is_finite(),
        "routing weights sum to {sum} over {} picks",
        sel.len()
    );
    let scale = routed_scale / sum;
    out.fill(0.0);
    for &e in sel {
        out[e] = scores[e] * scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole 93-layer schedule at the real block size, walked as one stateful pass — the
    /// invariant form, so an off-by-one in `div_ceil` cannot agree with an off-by-one here.
    /// Then the four cells the anchor captured, pinned as values: those are the widths
    /// `k3-anchor-decode-*.bin` attests (`model.layers.{1,12,91}.*_res.in.block_residual`),
    /// so this test and the widths gate say the same thing from two directions.
    #[test]
    fn the_fold_schedule_walks_the_reference_loop() {
        let (mut stack, mut boundaries) = (0usize, 0usize);
        for l in 0..93 {
            let f = fold_at(l, 12);
            assert_eq!(
                f.entry_sources,
                (stack > 0).then_some(stack + 1),
                "layer {l} entry"
            );
            // Independent derivation: the next boundary is 12 past the last one counted.
            assert_eq!(f.push, l == boundaries * 12, "layer {l} boundary");
            if f.push {
                stack += 1;
                boundaries += 1;
            }
            assert_eq!(f.mlp_sources, stack + 1, "layer {l} mlp");
        }
        assert_eq!(
            boundaries, 8,
            "boundaries at 0, 12, .., 84 — the last block is 9 deep"
        );
        assert_eq!(final_sources(93, 12), 9, "8 snapshots + the final prefix");
        // The anchor's captured stack widths, as (layer, entry blocks, mlp blocks).
        for (l, entry, mlp) in [(0, 0, 1), (1, 1, 1), (12, 1, 2), (91, 8, 8), (92, 8, 8)] {
            let f = fold_at(l, 12);
            assert_eq!(f.entry_sources, (entry > 0).then_some(entry + 1), "L{l}");
            assert_eq!(f.mlp_sources, mlp + 1, "L{l}");
        }
    }

    /// Trap 11 priced as arithmetic: the weights come from the UNBIASED scores, sum to
    /// `routed_scale` exactly, and land at absolute ids with zeros everywhere else.
    #[test]
    fn combining_weights_renormalises_the_unbiased_scores() {
        let scores = [0.1f32, 0.9, 0.4, 0.6];
        let sel = [1usize, 3];
        let mut out = [f32::NAN; 4];
        assert!(combine_weights(&scores, &sel, 2.5, &mut out).is_ok());
        assert_eq!(out[0], 0.0, "unselected experts are zero, not stale");
        assert_eq!(out[2], 0.0);
        assert!((out[1] - 2.5 * 0.9 / 1.5).abs() < 1e-6);
        assert!((out[3] - 2.5 * 0.6 / 1.5).abs() < 1e-6);
        assert!(
            (out.iter().sum::<f32>() - 2.5).abs() < 1e-6,
            "sum is routed_scale"
        );
        // A dead selection refuses instead of dividing by the reference's 1e-20 guard.
        assert!(combine_weights(&[0.0; 4], &sel, 1.0, &mut out).is_err());
    }
}
