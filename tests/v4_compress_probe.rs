//! **Closing the three coverage holes, at real weights.**
//!
//! `tests/v4_compress.rs` records the holes as executable facts about the *shipped* goldens.
//! This file closes them, by driving `Oracle::compressor` and `Oracle::indexer` directly —
//! which the coordinator made possible at `6dd5a3e` by exposing them — on the real
//! checkpoint's compressor and indexer tensors, at probe lengths and an `index_topk` the
//! 13-token emit prompt never reaches.
//!
//! # Why this is not already covered by the toy
//!
//! `V4Config::toy` sets `index_topk: 2` and `compress_ratios: [0, 0, 4, 8]` precisely so the
//! top-k truncates and the non-overlapping branch runs — S1b closed the *structural* case
//! deliberately and says so. The holes are about the goldens emitted from the **checkpoint**,
//! at `index_topk = 512`, ratio 128, and a 13-token prompt. Toy shapes and real shapes differ
//! in exactly the ways that matter here — the three `ape` tensors on the two layers are
//! `[4, 1024]` (L2 attention), `[4, 256]` (L2 indexer, at `index_head_dim`) and `[128, 512]`
//! (L3) — so a toy verdict does not transfer.
//!
//! # The method: blind versus sighted, not "it runs"
//!
//! Showing that a longer probe produces output proves nothing about whether the gate can
//! *reject* anything. So each hole is closed the way the plan demands — by taking one of
//! S1b's `Defect` breakages, which is a real wrong implementation, and showing the
//! comparison is **blind** to it at the emit prompt and **sighted** at the probe. A test
//! that only showed the sighted half would not establish that the hole was ever real.
//!
//! Skips with a printed reason when the checkpoint is absent; there is no CI here and this
//! reads 167 GB of index metadata, so it must not be a hard failure on a machine without it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli::v4compress::{LayerKind, compress_offset, compress_topk};
use rivoli::v4oracle::forward::{Counters, Defect, LayerW, Oracle, Step};
use rivoli::v4oracle::weights::{V4Config, WMat};
use std::collections::HashMap;

mod common;
use common::{
    EMIT_LEN, PROBE_LEN, PROBE_REMAINDER_LEN, RATIO_128_FIRST_DECODE_BLOCK, checkpoint,
    compressor_w, indexer_w, probe,
};

// ---------------------------------------------------------------------------------------
// loading just the compressor/indexer tensors
// ---------------------------------------------------------------------------------------
//
// `compressor_w` and the probe lengths moved to `tests/common/mod.rs` when
// `v4_compress_kernel.rs` needed the same two compressors. `indexer_w` followed on
// 2026-08-05 when `v4_indexer_kernel.rs` became its second consumer -- the comment here used
// to argue that a second consumer did not exist, and the duplication gate found the copy the
// moment one did.

/// A `LayerW` carrier for [`Step`], whose weights are never read.
///
/// `Oracle::indexer` destructures `let Step { s, start_pos, .. } = *step` and touches nothing
/// else on it, so the layer here is a placeholder. Spelled as empty matrices rather than as
/// real ones so that if `indexer` ever starts reading `step.lw`, this fails loudly on a
/// zero-sized matrix instead of quietly scoring against the wrong layer's weights.
fn step_carrier() -> LayerW {
    let e = || WMat::Dense {
        rows: 0,
        cols: 0,
        v: Vec::new(),
    };
    LayerW {
        attn_sink: Vec::new(),
        wq_a: e(),
        q_norm: Vec::new(),
        wq_b: e(),
        wkv: e(),
        kv_norm: Vec::new(),
        wo_a: e(),
        wo_b: e(),
        attn_norm: Vec::new(),
        ffn_norm: Vec::new(),
        hc_attn_fn: Vec::new(),
        hc_attn_base: Vec::new(),
        hc_attn_scale: Vec::new(),
        hc_ffn_fn: Vec::new(),
        hc_ffn_base: Vec::new(),
        hc_ffn_scale: Vec::new(),
        gate_w: e(),
        gate_bias: None,
        tid2eid: None,
        compressor: None,
        indexer: None,
        experts: HashMap::new(),
        shared: rivoli::v4oracle::forward::ExpertW {
            w1: e(),
            w2: e(),
            w3: e(),
        },
    }
}

// =======================================================================================
// HOLE 1 — ratio-128 pooling has no golden at all
// =======================================================================================

/// The ratio-128 compressor emits nothing at 13 tokens and pools correctly at 256 — and the
/// pooling is defect-sensitive, so the branch is now genuinely gated rather than merely run.
///
/// At the emit prompt `Compressor.forward` returns `None` before it reaches the pooling,
/// so the oracle pushes no `L3.*.compressed` tensor and there is nothing for a comparison to
/// be right or wrong about. That is the hole. Here the same weights at `PROBE_LEN` produce
/// two blocks, and two of S1b's breakages move them — which is what makes the probe a gate.
#[test]
fn ratio_128_pooling_is_gated_at_a_long_probe_and_absent_at_the_emit_prompt() {
    let Some(ck) = checkpoint() else { return };
    let c = V4Config::v4_flash();
    let ratio = c.compress_ratio(3);
    assert_eq!(ratio, 128, "layer 3 is the ratio-128 class");
    let cw = compressor_w(&ck, "layers.3.attn.compressor", ratio, c.head_dim, false);
    let freqs_layer = 3;
    // Load the ratio-4 ATTENTION compressor purely so its [4, 1024] / coff-2 shape is
    // asserted too. Without it the only coff-2 tensor this file checks is the indexer's
    // [4, 256], and the width the brief actually warns about goes unverified.
    let l2 = compressor_w(&ck, "layers.2.attn.compressor", 4, c.head_dim, false);
    assert_eq!(
        l2.ape.len(),
        4 * 2 * c.head_dim,
        "L2 attention ape is [4, 1024]"
    );

    let run = |d: Defect, n: usize| {
        let o = Oracle::new(c.clone(), d);
        // The oracle's own state constructor rather than a re-derived one. On THIS path
        // only `cache` is read (see `PROBE_LEN`); the state matters in the remainder and
        // decode arms below.
        let mut cs = o
            .fresh_state(freqs_layer)
            .comp
            .expect("layer 3 has a compressor");
        let mut ctr = Counters::default();
        let x = probe("l3-probe", n, c.dim);
        let out = o.compressor(&cw, &mut cs, &x, n, 0, o.freqs(freqs_layer), &mut ctr);
        (out, ctr)
    };

    // The hole, reproduced at real weights: nothing at all comes out.
    let (short, ctr) = run(Defect::None, EMIT_LEN);
    assert!(
        short.is_none(),
        "13 tokens < ratio 128: no block, hence no golden"
    );
    assert_eq!(ctr.compressed_blocks, 0);

    // The hole, closed: two whole blocks, of the width the config implies.
    let (long, ctr) = run(Defect::None, PROBE_LEN);
    let base = long.expect("256 tokens pools two ratio-128 blocks");
    assert_eq!(ctr.compressed_blocks, 2);
    assert_eq!(base.len(), 2 * c.head_dim);
    assert!(base.iter().all(|v| v.is_finite()), "pooled rows are finite");

    // And the gate can REJECT. Both breakages are real transcription slips that leave the
    // shape and the magnitude intact, which is why "it produced numbers" was never enough.
    for d in [Defect::CompressorNoApe, Defect::CompressorRopeAtBlockEnd] {
        let (broken, _) = run(d, PROBE_LEN);
        let broken = broken.expect("the defect must not change WHETHER a block is emitted");
        assert_eq!(broken.len(), base.len(), "{d:?} changes values, not shape");
        assert!(
            broken.iter().zip(&base).any(|(a, b)| a != b),
            "{d:?} must move the pooled rows"
        );
    }

    // The silent half: a breakage that lives only in the overlapping branch must leave a
    // ratio-128 layer bit-identical. Note this one is inert BY CONSTRUCTION (`cw.overlap` is
    // already false, so the defect has no term to disable), so it is a regression pin and
    // NOT independent evidence that the probe has resolution -- the two defects above are.
    let (inert, _) = run(Defect::CompressorNoOverlap, PROBE_LEN);
    assert_eq!(
        inert.expect("still emits"),
        base,
        "CompressorNoOverlap must be inert where overlap is already false"
    );

    // A prefill with a remainder: the ONE prefill path that writes `kv_state`/`score_state`
    // on a non-overlapping compressor, and therefore the state a later decode would read.
    let (rem, ctr) = run(Defect::None, PROBE_REMAINDER_LEN);
    assert_eq!(
        ctr.compressed_blocks,
        PROBE_REMAINDER_LEN / 128,
        "300 tokens pools 2 blocks"
    );
    assert_eq!(rem.expect("emits").len(), 2 * c.head_dim);
}

/// **The other half of hole 1: ratio-128 at DECODE.**
///
/// The emit run leaves this as uncovered as the prefill half — its decode steps sit at
/// `start_pos` 13..25 and the first block completes at 127. And this is the more
/// port-hostile branch of the two: it indexes `ape` by `start_pos % ratio`, writes one state
/// slot per step, gathers the whole state to pool, and takes its RoPE position from
/// `start_pos / ratio` rather than from a prefill stride. None of that arithmetic is touched
/// by the prefill test above.
#[test]
fn ratio_128_decode_completes_exactly_one_block_at_start_pos_127() {
    let Some(ck) = checkpoint() else { return };
    let c = V4Config::v4_flash();
    let cw = compressor_w(&ck, "layers.3.attn.compressor", 128, c.head_dim, false);

    let run = |d: Defect| {
        let o = Oracle::new(c.clone(), d);
        let mut cs = o.fresh_state(3).comp.expect("layer 3 has a compressor");
        let mut ctr = Counters::default();
        // Seed with a short prefill, exactly as the emit run does, then step.
        let x = probe("l3-dec-pre", EMIT_LEN, c.dim);
        assert!(
            o.compressor(&cw, &mut cs, &x, EMIT_LEN, 0, o.freqs(3), &mut ctr)
                .is_none()
        );
        let mut emitted_at = Vec::new();
        let mut last = None;
        for start_pos in EMIT_LEN..=RATIO_128_FIRST_DECODE_BLOCK {
            let row = probe(&format!("l3-dec-{start_pos}"), 1, c.dim);
            if let Some(v) = o.compressor(&cw, &mut cs, &row, 1, start_pos, o.freqs(3), &mut ctr) {
                emitted_at.push(start_pos);
                last = Some(v);
            }
        }
        (emitted_at, last, ctr)
    };

    let (at, block, ctr) = run(Defect::None);
    assert_eq!(
        at,
        vec![RATIO_128_FIRST_DECODE_BLOCK],
        "exactly one block, exactly at 127"
    );
    assert_eq!(ctr.compressed_blocks, 1);
    let base = block.expect("a block at 127");
    assert_eq!(base.len(), c.head_dim, "decode emits one row, not a matrix");
    assert!(base.iter().all(|v| v.is_finite()));

    // And it is gated: the same two breakages move the decode-pooled row.
    for d in [Defect::CompressorNoApe, Defect::CompressorRopeAtBlockEnd] {
        let (at, block, _) = run(d);
        assert_eq!(
            at,
            vec![RATIO_128_FIRST_DECODE_BLOCK],
            "{d:?} changes values, not timing"
        );
        assert!(
            block
                .expect("still emits")
                .iter()
                .zip(&base)
                .any(|(a, b)| a != b),
            "{d:?} must move the decode-pooled row"
        );
    }
}

// =======================================================================================
// HOLE 2 — the ratio-128 selection golden is empty
// =======================================================================================

/// `compress_topk` is vacuous at 13 tokens and discriminating at 256, cross-checked against
/// the oracle's own copy now that it is public.
///
/// Note what this is and is not: the two implementations are not independent derivations
/// (`src/v4compress.rs`'s `jscpd:ignore` region states why), so agreement here is a **drift
/// tripwire**. The non-vacuity half is the part that closes the hole.
#[test]
fn ratio_128_selection_is_vacuous_at_the_emit_prompt_and_discriminating_at_the_probe() {
    let c = V4Config::v4_flash();
    let l3 = LayerKind::from_ratio(128);

    let mine = |n: usize| compress_topk(l3, n, 0, compress_offset(c.window_size, n, 0));
    let theirs = |n: usize| {
        rivoli::v4oracle::forward::compress_topk(128, n, 0, compress_offset(c.window_size, n, 0))
    };
    let as_i64 = |v: Vec<Vec<i32>>| -> Vec<Vec<i64>> {
        v.into_iter()
            .map(|r| r.into_iter().map(i64::from).collect())
            .collect()
    };

    // The hole: 13 rows of zero columns. Any implementation returning nothing agrees.
    let short = mine(EMIT_LEN);
    assert_eq!(short.len(), EMIT_LEN);
    assert_eq!(
        short.concat().len(),
        0,
        "the golden carries no values to be wrong about"
    );
    assert_eq!(as_i64(short), theirs(EMIT_LEN));

    // Closed: 2 real columns, and the causal structure is now observable — early queries see
    // nothing, the last query sees both blocks.
    let long = mine(PROBE_LEN);
    assert_eq!(long[0].len(), PROBE_LEN / 128);
    assert!(
        long[0].iter().all(|&x| x == -1),
        "query 0 may read no completed block"
    );
    assert_eq!(
        long[PROBE_LEN - 1].iter().filter(|&&x| x != -1).count(),
        2,
        "the last query may read both completed blocks"
    );
    assert_eq!(
        as_i64(long),
        theirs(PROBE_LEN),
        "drift tripwire against the oracle's copy"
    );
}

// =======================================================================================
// HOLE 3 — the ranking, which index_topk never truncates at 13 tokens
// =======================================================================================

/// **The priority hole.** At `index_topk = 512` and 13 tokens the top-k selects every
/// compressed block, so `.compress_idxs` is an invariant SET and set-comparison — which is
/// the comparison `forward.rs:772` correctly mandates — passes against an arbitrarily wrong
/// ranking. This shows exactly that, then shows the same defect being caught once `index_topk`
/// truncates.
///
/// `Counters::indexer_truncated` must read zero in the blind arm and non-zero in the sighted
/// one. It is **necessary, not sufficient**: it increments by `usize::from(k < n_comp)`,
/// which is the same for every row, so it is really `s * (index_topk < n_comp)` and says
/// nothing about the causal mask. It would read 13 even if every row were fully masked. What
/// makes the test sound is the set comparison, not this counter.
#[test]
fn indexer_ranking_is_blind_at_index_topk_512_and_sighted_when_it_truncates() {
    let Some(ck) = checkpoint() else { return };
    let base_cfg = V4Config::v4_flash();
    assert_eq!(
        base_cfg.compress_ratio(2),
        4,
        "layer 2 is the ratio-4 class and has an Indexer"
    );
    assert_eq!(base_cfg.index_topk, 512);
    let iw = indexer_w(&ck, 2, &base_cfg);
    let carrier = step_carrier();

    // One arm: a config, a probe length, and a defect. Everything else is held fixed, so the
    // only thing separating the blind arm from the sighted one is `index_topk`.
    let arm = |topk: usize, n: usize, d: Defect| {
        let mut c = base_cfg.clone();
        c.index_topk = topk;
        let o = Oracle::new(c.clone(), d);
        // `idx_comp`, not `comp`: the indexer's compressor is over `index_head_dim` (128)
        // with `rotate = true`, a different geometry from the attention compressor on the
        // same layer. Taking `comp` here would run at head_dim 512 and still "work".
        let mut cs = o.fresh_state(2).idx_comp.expect("layer 2 has an indexer");
        let mut ctr = Counters::default();
        let mut scores = Vec::new();
        let step = Step {
            lw: &carrier,
            layer: 2,
            s: n,
            start_pos: 0,
            input_ids: &[],
            phase: "pre",
        };
        let x = probe("l2-x", n, c.dim);
        let qr = probe("l2-qr", n, c.q_lora_rank);
        let idxs = o.indexer(
            &step,
            &iw,
            &mut cs,
            &x,
            &qr,
            compress_offset(c.window_size, n, 0),
            o.freqs(2),
            &mut ctr,
            &mut scores,
        );
        (idxs, scores, ctr)
    };

    // Compare the way a consumer must: as a SET per query row, never positionally.
    // `.compress_idxs` is score-ORDERED (`topk_idx`'s doc, forward.rs:776-782), so a
    // positional compare would
    // report a difference for a tie-break permutation and send the reader after a bug that
    // is not there.
    let as_sets = |v: &[Vec<i64>]| -> Vec<Vec<i64>> {
        v.iter()
            .map(|r| {
                let mut s = r.clone();
                s.sort_unstable();
                s
            })
            .collect()
    };

    // -- the blind arm: real index_topk, real prompt length ------------------------------
    let (base_idx, base_scores, ctr) = arm(512, EMIT_LEN, Defect::None);
    assert!(ctr.indexer_ran);
    assert_eq!(
        ctr.indexer_truncated, 0,
        "at index_topk 512 and 3 compressed blocks the top-k cannot cut -- this IS the hole"
    );

    // Two breakages that change WHICH blocks rank highest. Each moves the scores, and the
    // selected set does not move at all: the gate is blind to both.
    for d in [Defect::IndexerNoWeights, Defect::IndexerNoRelu] {
        let (idx, scores, _) = arm(512, EMIT_LEN, d);
        assert_ne!(
            scores, base_scores,
            "{d:?} must move `.indexer_scores` -- else it is inert"
        );
        assert_eq!(
            as_sets(&idx),
            as_sets(&base_idx),
            "{d:?}: the selection SET is invariant at index_topk 512. A gate resting on \
             `.compress_idxs` alone accepts this wrong ranking."
        );
    }

    // -- the sighted arm: index_topk lowered so the top-k truncates ----------------------
    // 1, not 2: with 3 compressed blocks and the causal mask, k = 2 still admits every
    // legal block for most query rows. k = 1 forces a genuine argmax over the scores.
    let (sharp_idx, sharp_scores, ctr) = arm(1, EMIT_LEN, Defect::None);
    assert!(
        ctr.indexer_truncated > 0,
        "index_topk must actually CUT, else the probe is not exercising the ranking and the \
         coverage is still absent"
    );
    assert_eq!(
        sharp_scores, base_scores,
        "lowering index_topk changes selection, never scoring"
    );

    // The truncated pick is driven by the SCORES, not by the causal mask's ordering. If the
    // top-k were degenerate -- always "the newest legal block" -- the picked index would rise
    // monotonically with the query row, and the set comparison would then be testing the mask
    // a second time rather than the ranking. Measured at real weights it does NOT: the picks
    // run [.., 13, 13, 15, 14], so row 12 selects a strictly older block than row 11.
    let picks: Vec<i64> = sharp_idx
        .iter()
        .flatten()
        .copied()
        .filter(|&x| x >= 0)
        .collect();
    assert!(
        picks.windows(2).any(|w| w[1] < w[0]),
        "the truncated selection must not be monotonic in the query row, else it is the \
         causal mask being re-tested and not the scoring: {picks:?}"
    );

    // At EMIT_LEN the top-k truncates but the CAUSAL MASK still leaves only rows 7..12 with
    // more than one legal candidate, so each defect moves exactly one row and two of the
    // four ranking defects move none. Truncation is necessary and the probe LENGTH is what
    // finally gates them -- measured, not assumed.
    let ranking_defects = [
        Defect::IndexerNoWeights,
        Defect::IndexerNoRelu,
        Defect::IndexerNoFp4Quant,
        Defect::IndexerNoHadamard,
    ];
    let (long_idx, _, ctr) = arm(1, 64, Defect::None);
    assert!(ctr.indexer_truncated > 0, "the long arm must truncate too");
    for d in ranking_defects {
        let (idx, scores, _) = arm(1, 64, d);
        assert!(
            scores
                .iter()
                .all(|v| v.is_finite() || *v == f32::NEG_INFINITY)
        );
        assert_ne!(
            as_sets(&idx),
            as_sets(&long_idx),
            "{d:?} must change the selected set once the top-k truncates AND the probe is \
             long enough for the causal mask to admit alternatives. All four are required: \
             `caught > 0` over two of them passed with a one-row margin."
        );
    }
}
