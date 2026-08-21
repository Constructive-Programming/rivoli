//! **The M15 set-equality gate: the engine's scored block selection against the frozen
//! oracle's own `Indexer.forward`.**
//!
//! The goldens are captured HERE, from the oracle, at test time — never from the engine
//! path under test (the named failure mode of this gate). `Oracle::indexer` is driven
//! directly with toy weights: it runs the indexer's own compressor, exports the full
//! pre-top-k score matrix, and returns the selection the reference would attend. The
//! engine half under test is `v4::select::scored_rows` — the pure arithmetic the device
//! path hands its D2H'd scores to — fed the ORACLE's matrix, so this compares top-k, tie
//! rule, causal mask and `-1` convention with the scores held bit-identical. The device
//! half of the chain (wq_b GEMV, RoPE, the Hadamard-fp4 spread, `index_score_blocks`) is
//! scored bit-identical separately in `kernel_v4_indexer.rs`; this file is the half a GPU
//! cannot make flaky.
//!
//! Sits in `crates/engine/tests/` and not `crates/oracles/tests/` because the workspace
//! DAG points oracles → engine: the oracle crate cannot name `scored_rows`. The oracle-side
//! invariants of the same captures (below-cap keeps everything; scores determine the
//! selection; perturbation resolution) live with the oracle, in
//! `crates/oracles/tests/v4_indexer_goldens.rs`.
//!
//! Deviceless and ungated: runs on every `cargo test`, including CI's
//! `--no-default-features` arm.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_engine::v4::geometry::LayerKind;
use rivoli_engine::v4::select::{Extent, Sel, SelRule, scored_rows};
use rivoli_oracles::v4oracle::forward::{Counters, Defect, LayerCtx, Oracle};
use rivoli_oracles::v4oracle::toy;
use rivoli_oracles::v4oracle::weights::{V4Config, fixed_bf16};
use std::collections::BTreeSet;

/// The toy's one indexed layer, its ratio, and its `index_topk` — 2, so the cap
/// (`4 * (2 + 1) = 12` tokens) is crossed by a 16-token prompt and by every decode step
/// after it, and the top-k genuinely truncates. At the shipped 512 nothing under 2052
/// tokens exercises ranking at all, which is the recorded "set-invariant goldens" trap.
const LAYER: usize = 2;
const TOPK: usize = 2;
const PROMPT: usize = 16;
const DECODE_STEPS: usize = 5;

/// One oracle indexer call's exports: the score matrix and the selection built from it.
struct Golden {
    at: Extent,
    offset: usize,
    n_comp: usize,
    scores: Vec<f32>,
    sel: Vec<Vec<i64>>,
}

impl Golden {
    /// The causal limit of query row `t` — how many blocks it may legally attend. Both
    /// gate tests classify rows by it, and two spellings of the phase branch is how one of
    /// them drifts to counting the other's rows.
    fn limit_of(&self, t: usize) -> usize {
        if self.at.is_prefill() {
            (t + 1) / 4
        } else {
            self.n_comp
        }
    }
}

/// Drive `Indexer.forward` through one prefill and `decode_steps` decodes on ONE
/// compressor state, exactly as a decode drives it — the state carry is what makes the
/// later steps' caches real rather than re-seeded.
fn capture_with(prompt: usize, decode_steps: usize) -> (V4Config, Vec<Golden>) {
    let cfg = V4Config::toy();
    let m = toy::build(&cfg);
    let o = Oracle::new(cfg.clone(), Defect::None);
    let lw = &m.layers[LAYER];
    let iw = lw.indexer.as_ref().expect("layer 2 carries the indexer");
    let mut rings = o.fresh_state(LAYER);
    let cs = rings.idx_comp.as_mut().expect("indexer compressor state");
    let mut out = Vec::new();
    let steps = std::iter::once((prompt, 0usize)).chain((0..decode_steps).map(|i| (1, prompt + i)));
    for (s, start_pos) in steps {
        let at = Extent {
            seqlen: s,
            start_pos,
        };
        // The reference's own offsets: the prompt length at prefill, the ring width at
        // decode (`v4oracle::attention::attn_compress_idxs`).
        let offset = if start_pos == 0 { s } else { cfg.window_size };
        let tag = format!("p{start_pos}");
        let x = fixed_bf16(&format!("x-{tag}"), s * cfg.dim, 1.0);
        let qr = fixed_bf16(&format!("qr-{tag}"), s * cfg.q_lora_rank, 1.0);
        let mut scores = Vec::new();
        let mut ctr = Counters::default();
        let step = LayerCtx {
            lw,
            layer: LAYER,
            s,
            start_pos,
            input_ids: &[],
            step_tag: &tag,
        };
        let sel = o.indexer(
            &step,
            iw,
            cs,
            &x,
            &qr,
            offset,
            o.freqs(LAYER),
            &mut ctr,
            &mut scores,
        );
        assert!(ctr.indexer_ran, "{tag}: the indexer must have run");
        let n_comp = (start_pos + s) / 4;
        assert_eq!(scores.len(), s * n_comp, "{tag}: exported matrix shape");
        out.push(Golden {
            at,
            offset,
            n_comp,
            scores,
            sel,
        });
    }
    (cfg, out)
}

/// The boundary-crossing fixture the two gate tests share.
fn capture() -> (V4Config, Vec<Golden>) {
    capture_with(PROMPT, DECODE_STEPS)
}

/// One golden's engine-side [`SelRule`], under the fixture's own constants — spelled once
/// so the three gate tests cannot rank under three slightly different rules.
fn rule_for(g: &Golden) -> SelRule {
    SelRule {
        kind: LayerKind::from_ratio(4),
        index_topk: TOPK,
        at: g.at,
        offset: g.offset,
    }
}

fn set_of_i64(row: &[i64]) -> BTreeSet<i64> {
    row.iter().copied().filter(|&v| v >= 0).collect()
}

fn set_of_i32(row: &[i32]) -> BTreeSet<i64> {
    row.iter()
        .filter(|&&v| v >= 0)
        .map(|&v| i64::from(v))
        .collect()
}

/// **The gate.** Per query row of every captured step: the engine's selection over the
/// oracle's own scores names exactly the set the oracle attends. Set-equality and not
/// list-equality, deliberately — the engine emits ascending where the oracle emits score
/// order, a summation-order difference the attend kernel's online softmax folds in
/// (`scored_rows`'s doc carries the argument).
#[test]
fn the_engines_selection_over_the_oracles_scores_names_the_oracles_set() {
    let (_, goldens) = capture();
    let mut truncated = 0usize;
    for g in &goldens {
        let got = scored_rows(&g.scores, g.n_comp, rule_for(g)).expect("legal scored rows");
        assert_eq!(got.len(), g.sel.len(), "{:?}: row count", g.at);
        for (t, (ours, theirs)) in got.iter().zip(&g.sel).enumerate() {
            assert_eq!(
                set_of_i32(ours),
                set_of_i64(theirs),
                "{:?} row {t}: the sets diverged",
                g.at
            );
            // The engine's own layout contract, checked where the data is real: ascending
            // survivors, then the -1 tail — nothing interleaved.
            let live: Vec<i32> = ours.iter().copied().filter(|&v| v >= 0).collect();
            assert!(live.is_sorted(), "{:?} row {t}: not ascending", g.at);
            assert!(
                ours[live.len()..].iter().all(|&v| v == -1),
                "{:?} row {t}: a -1 sits before a survivor",
                g.at
            );
            truncated += usize::from(g.limit_of(t) > TOPK);
        }
    }
    // Anti-vacuity: ranking must have DECIDED something, or every row above compared the
    // causal mask and nothing else.
    assert!(
        goldens.iter().any(|g| g.n_comp > TOPK),
        "no step ever offered more blocks than the top-k keeps"
    );
    assert!(
        truncated >= 3,
        "only {truncated} truncated rows reached the gate"
    );
}

/// **The keep-oldest sabotage, at selection level — the deviceless half of gate (c)'s
/// red-proof.** A selection that keeps the OLDEST blocks (`min(k, limit)` positional — the
/// exact bug `positional_context_limit` documents) must disagree with the scored set on
/// truncated rows. If it ever stops disagreeing, the NLL-cliff red-proof upstairs would
/// pass vacuously: the sabotaged build would BE the scored build.
#[test]
fn a_keep_oldest_selection_disagrees_with_the_scored_one_above_the_boundary() {
    let (_, goldens) = capture();
    let mut disagreements = 0usize;
    let mut truncated = 0usize;
    for g in goldens.iter().filter(|g| g.n_comp > TOPK) {
        let scored = scored_rows(&g.scores, g.n_comp, rule_for(g)).expect("legal");
        for (t, ours) in scored.iter().enumerate() {
            let limit = g.limit_of(t);
            if limit <= TOPK {
                continue;
            }
            truncated += 1;
            let oldest: BTreeSet<i64> = (0..TOPK.min(limit) as i64)
                .map(|c| c + g.offset as i64)
                .collect();
            disagreements += usize::from(set_of_i32(ours) != oldest);
        }
    }
    assert!(truncated >= 3, "the fixture stopped crossing the boundary");
    assert!(
        disagreements > 0,
        "keep-oldest equals the scored selection on every truncated row — the sabotage \
         red-proof would be vacuous ({truncated} rows checked)"
    );
}

/// The below-cap identity against the REAL oracle scores (the select.rs unit test proves
/// it for adversarial synthetic scores): every step and prefill row whose legal set fits
/// under the top-k gathers a buffer byte-identical to the positional path's.
#[test]
fn below_the_cap_the_scored_gather_is_the_positional_gather_on_real_scores() {
    // A fresh SUB-CAP capture — an 8-token prompt offers at most 2 blocks == TOPK, so
    // every row's legal set fits under the top-k and the boundary is never crossed.
    let (cfg, goldens) = capture_with(8, 0);
    let g = &goldens[0];
    let kind = LayerKind::from_ratio(4);
    let sel = Sel {
        win: cfg.window_size,
        kind,
        index_topk: TOPK,
        at: g.at,
    };
    let comp = scored_rows(&g.scores, g.n_comp, rule_for(g)).expect("legal");
    let (mut pos, mut scr) = (Vec::new(), Vec::new());
    let shape_pos = sel.gather(&mut pos).expect("positional path");
    let shape_scr = sel.gather_scored(&comp, &mut scr).expect("scored path");
    assert_eq!(shape_pos, shape_scr, "the two rectangles must agree");
    assert_eq!(
        pos, scr,
        "below the cap the two buffers must be byte-identical"
    );
    // And not vacuously: the sub-cap prompt really did offer blocks, and the oracle's own
    // selection for its last row is those same blocks.
    assert!(
        pos.iter().any(|&v| v >= 8),
        "no compressed column was filled"
    );
    assert_eq!(
        set_of_i64(g.sel.last().expect("rows exist")).len(),
        2,
        "the last row must see both closed blocks"
    );
}
