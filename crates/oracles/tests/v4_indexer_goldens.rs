//! **The scored-selection goldens: what `Indexer.forward` actually selects, captured from
//! the frozen oracle and pinned** — the M15 gate's oracle half.
//!
//! The engine's scored block selection (rivoli-engine `v4::select::scored_rows`) is gated
//! against these captures in `crates/engine/tests/v4_scored_selection.rs` — the DAG points
//! oracles → engine, so the cross-comparison lives there and THIS file owns everything the
//! oracle can witness about itself. Zero edits inside `src/v4oracle/`: every tensor here is
//! read out of the capture the transliteration already exports (`.indexer_scores` is the
//! full pre-top-k matrix, `.compress_idxs` the selection), which is exactly why those
//! exports exist.
//!
//! Three claims, each with the case that could refute it:
//!
//! 1. **Below the cap the indexer keeps every block it is offered.** This is the premise
//!    the whole pre-M15 arm stood on (`positional_context_limit`'s doc) and the premise the
//!    M15 arm's below-cap byte-identity still stands on — here it is CHECKED against the
//!    reference's own selection rather than argued from its topk shape.
//! 2. **Above the cap the exported scores DETERMINE the exported selection**, under the
//!    documented rule (descending score, ties toward the lower block index, causal mask by
//!    index, `-1` for masked slots). This is the contract `scored_rows` re-implements; if
//!    `topk_idx` ever changes its tie-break, this reddens before the engine drifts.
//! 3. **The harness can go red.** A perturbed score above the boundary must MOVE the
//!    recomputed set (resolution), and one below the cap must NOT (specificity — which is
//!    also claim 1 restated as an executable).
//!
//! Runs on the toy (`index_topk = 2`, so the cap is 12 tokens and truncation is REACHED),
//! deviceless, on every `cargo test`.
//!
//! **Both sides of the cap are COUNTED rather than trusted, because claims 1 and 2 live on
//! opposite sides of it and either side's row count can reach zero while the other's floor
//! is still met.** Rows above the cap come from every step; rows below it come only from
//! the prefill capture (every decode step here sits above). `steps()` extends from an
//! `Option` and a `filter_map`, neither of which asserts, so a renamed export would leave
//! the truncated floor satisfied by decode rows alone while claim 1 — the premise the
//! whole below-cap byte-identity argument rests on — examined nothing. Red-proved by
//! dropping the prefill capture: `only 0 non-empty below-cap rows`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_oracles::v4oracle::forward::{Capture, Defect};

#[path = "common/oracle_probe.rs"]
mod oracle_probe;
use oracle_probe::run;

/// The toy's one indexed layer (`compress_ratios = [0, 0, 4, 8]`).
const LAYER: usize = 2;
const RATIO: usize = 4;
const TOPK: usize = 2;

/// One captured step's indexer tensors: the full score matrix and the selection built
/// from it, both exactly as the oracle exported them.
struct Step {
    scores: Vec<f32>,
    n_comp: usize,
    sel: Vec<Vec<i64>>,
    /// `(t + 1) / ratio` per query row at prefill; `n_comp` for the one decode row.
    limits: Vec<usize>,
    offset: i64,
}

/// Pull one step's pair out of a capture, deriving the shapes from the capture's own
/// records rather than re-deriving them from the config — the golden is the authority here.
fn step(cap: &Capture, tag: &str, prefill: bool, offset: i64) -> Option<Step> {
    let (_, sshape, scores) = cap
        .floats
        .iter()
        .find(|(n, _, _)| n == &format!("L{LAYER}.{tag}.indexer_scores"))?;
    let (_, ishape, flat) = cap
        .ints
        .iter()
        .find(|(n, _, _)| n == &format!("L{LAYER}.{tag}.compress_idxs"))?;
    let (rows, n_comp) = (sshape[0], sshape[1]);
    assert_eq!(ishape[0], rows, "{tag}: selection rows != score rows");
    let sel = flat.chunks_exact(ishape[1]).map(<[i64]>::to_vec).collect();
    let limits = (0..rows)
        .map(|t| if prefill { (t + 1) / RATIO } else { n_comp })
        .collect();
    Some(Step {
        scores: scores.clone(),
        n_comp,
        sel,
        limits,
        offset,
    })
}

/// Every step of one toy run that has an indexer capture, prefill first.
fn steps(prompt: usize) -> Vec<Step> {
    let r = run(LAYER, prompt, Defect::None);
    let mut v = Vec::new();
    v.extend(step(&r.pre, "pre", true, prompt as i64));
    // The decode offset is the toy's `window_size` — the ring precedes the compressed
    // region in selection space. 8 is `V4Config::toy`'s value, asserted at use by the
    // in-range check below rather than re-read from a config this file does not open.
    v.extend(
        (0..oracle_probe::DECODE_STEPS).filter_map(|i| step(&r.dec, &format!("dec{i}"), false, 8)),
    );
    v
}

/// The documented selection rule, applied to an exported score matrix: causal mask BY
/// INDEX, top `min(TOPK, n_comp)` by descending score with ties toward the lower block
/// index, masked picks as `-1`, survivors offset — `topk_idx`'s observable contract,
/// restated once so a drift in either the oracle or this file's reading of it reddens.
fn recompute(s: &Step) -> Vec<Vec<i64>> {
    let k = TOPK.min(s.n_comp);
    s.limits
        .iter()
        .enumerate()
        .map(|(t, &limit)| {
            let row = &s.scores[t * s.n_comp..(t + 1) * s.n_comp];
            let mut order: Vec<usize> = (0..s.n_comp).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            order.truncate(k);
            order
                .into_iter()
                .map(|i| if i < limit { i as i64 + s.offset } else { -1 })
                .collect()
        })
        .collect()
}

/// The selected SET of one exported row — the non-masked entries, order discarded.
fn set_of(row: &[i64]) -> std::collections::BTreeSet<i64> {
    row.iter().copied().filter(|&v| v >= 0).collect()
}

/// One step's rows against both claims. Returns (truncated, non-empty below-cap) counts —
/// counted here, ASSERTED by the caller, so the anti-vacuity floors stay in the test body
/// beside the argument for their values.
fn check_step_rows(prompt: usize, s: &Step) -> (usize, usize) {
    let (mut truncated, mut below_cap) = (0usize, 0usize);
    for (t, (row, &limit)) in s.sel.iter().zip(&s.limits).enumerate() {
        let got = set_of(row);
        if limit <= TOPK {
            // Claim 1: below the cap, the selection IS every causally-legal block.
            // Counted only when there IS one — a `limit == 0` row compares two
            // empty sets and would let the caller's floor be met vacuously.
            below_cap += usize::from(limit >= 1);
            let want: std::collections::BTreeSet<i64> =
                (0..limit as i64).map(|c| c + s.offset).collect();
            assert_eq!(got, want, "prompt {prompt} row {t}: below-cap set");
        } else {
            truncated += 1;
            assert_eq!(got.len(), TOPK, "prompt {prompt} row {t}: k survivors");
            assert!(
                got.iter()
                    .all(|&v| v >= s.offset && v < s.offset + limit as i64),
                "prompt {prompt} row {t}: a selected block is causally illegal"
            );
        }
    }
    (truncated, below_cap)
}

/// Claims 1 and 2, over every captured step of two runs — one whose prefill crosses the
/// truncation boundary at its last row (prompt 12) and one that crosses it four rows deep
/// (prompt 16), plus every decode step, all of which sit above it.
#[test]
fn the_exported_scores_determine_the_exported_selection_and_below_the_cap_keep_everything() {
    let (mut truncated_rows, mut below_cap_rows) = (0usize, 0usize);
    for prompt in [12usize, 16] {
        let all = steps(prompt);
        assert!(all.len() > 1, "prompt {prompt}: prefill plus decode steps");
        for s in &all {
            // Claim 2: the exported matrix reproduces the exported selection EXACTLY —
            // list-equal, not merely set-equal, so the tie rule and the `-1` mapping are
            // both pinned.
            assert_eq!(recompute(s), s.sel, "prompt {prompt}: rule drifted");
            let (truncated, below_cap) = check_step_rows(prompt, s);
            truncated_rows += truncated;
            below_cap_rows += below_cap;
        }
    }
    // Anti-vacuity, BOTH sides, because the two claims live on opposite sides of the cap
    // and each one's rows can reach zero while the other's floor is still met.
    //
    // Above: the boundary was CROSSED, or claim 2 ran only where the causal mask decides
    // everything — the "dsa A/B under 2048 tokens" trap wearing a new name.
    assert!(
        truncated_rows >= 5,
        "only {truncated_rows} truncated rows — the fixture no longer crosses the cap"
    );
    // Below: claim 1 is reachable ONLY from the prefill capture (every decode row here
    // sits above the cap — `n_comp` is 3, 3, 3, 4 at prompt 12 and 4, 4, 4, 5 at 16, all
    // past `TOPK = 2`). So a `steps()` that silently lost `L2.pre.*` — it extends from an
    // `Option` and a `filter_map`, neither of which asserts — would leave the truncated
    // floor met by the four decode steps alone while the claim the whole below-cap
    // byte-identity argument rests on executed zero times.
    assert!(
        below_cap_rows >= 8,
        "only {below_cap_rows} non-empty below-cap rows — the prefill capture is gone and \
         claim 1 examined nothing"
    );
}

/// Claim 3, both directions. The perturbation is a SCRATCH copy of the capture — the
/// golden itself is never edited — and the boundary block is chosen from the scores, not
/// hard-coded, so the proof survives a re-drawn toy.
#[test]
fn a_perturbed_score_moves_the_set_above_the_boundary_and_cannot_below_it() {
    let all = steps(16);
    let above = all
        .iter()
        .find(|s| s.limits.iter().any(|&l| l > TOPK))
        .expect("a step above the boundary exists");
    let t = above.limits.iter().position(|&l| l > TOPK).unwrap();
    let baseline = recompute(above);
    // Promote the best currently-EXCLUDED legal block of row `t` past the current winners.
    let row = &above.scores[t * above.n_comp..(t + 1) * above.n_comp];
    let chosen = set_of(&baseline[t]);
    let excluded = (0..above.limits[t])
        .find(|&c| !chosen.contains(&(c as i64 + above.offset)))
        .expect("limit > k, so an excluded legal block exists");
    let mut scratch = Step {
        scores: above.scores.clone(),
        n_comp: above.n_comp,
        sel: above.sel.clone(),
        limits: above.limits.clone(),
        offset: above.offset,
    };
    scratch.scores[t * above.n_comp + excluded] = row.iter().fold(1.0f32, |m, v| m.max(*v)) + 1.0;
    let moved = recompute(&scratch);
    assert_ne!(
        set_of(&moved[t]),
        chosen,
        "promoting an excluded block must change the set — the comparator has no resolution"
    );
    // Specificity: the same promotion on a BELOW-cap row changes nothing, because there the
    // top-k keeps everything and the scores are not consulted for membership.
    let below = all
        .iter()
        .find(|s| s.limits.iter().any(|&l| (1..=TOPK).contains(&l)))
        .expect("a below-cap row with at least one block exists");
    let bt = below
        .limits
        .iter()
        .position(|&l| (1..=TOPK).contains(&l))
        .unwrap();
    let mut scratch = Step {
        scores: below.scores.clone(),
        n_comp: below.n_comp,
        sel: below.sel.clone(),
        limits: below.limits.clone(),
        offset: below.offset,
    };
    scratch.scores[bt * below.n_comp] += 100.0;
    assert_eq!(
        set_of(&recompute(&scratch)[bt]),
        set_of(&below.sel[bt]),
        "a below-cap set moved with a score — the below-cap identity is broken"
    );
}
