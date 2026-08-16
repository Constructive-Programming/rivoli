//! Teacher-forced scoring arithmetic — the host half of `--ppl`, shared by all four
//! arms' `score` loops so the softmax, the argmax fold, the coherence check and the
//! walk protocol exist exactly once. Ported from `old:src/eval.rs` (`nll_of` verbatim in
//! substance, re-typed over `&[f32]` because two of this tree's arms already hand back
//! decoded floats — Glimmer's and K3's `logits()` accessors — and the other two decode
//! their D2H bytes through `rivoli_core::num::f32s_le`).
//!
//! **UNGATED** — nothing here names a device, and it was behind `teacher-forcing` until
//! 2026-08-16, when two reviews independently traced that no prescribed command enables
//! that feature for a `cargo test`: CI, CLAUDE.md's two batteries and the feature matrix
//! all run featureless or `-p rivoli`, so these tests were compiled by a clippy step and
//! run by nothing. They now run in every arm, including CI's. The per-arm `score()` loops
//! that call this stay behind the feature — they name a device.
//!
//! **The coherence check is the module's second job, and the reason [`score_row`] exists
//! rather than a bare `nll_of`.** The named failure mode of a TF path is measuring a
//! DIFFERENT engine — a stale row, the wrong buffer, a desynced position. Every scored
//! row is therefore folded on the host with the device argmax kernel's exact rule
//! (strict `>`, ties keep the lowest index, NaN never wins — `glm/decode.rs` documents
//! the kernel as reproducing this fold), and the run REFUSES on the first position where
//! host and device disagree about their own maximum. That check costs one pass over a
//! row already on the host and turns "the TF path quietly diverged from the decode path"
//! from a silent wrong number into a loud coordinate.

use crate::seam::Scored;
use anyhow::{Result, ensure};

/// The scoring door, called by every arm BEFORE its first forward: a corpus that cannot
/// be scored is refused with the flag to turn, not discovered mid-walk.
pub fn admit(ids: &[u32], max_ctx: usize) -> Result<()> {
    ensure!(
        ids.len() >= 2,
        "need at least 2 corpus tokens to score a position"
    );
    ensure!(
        ids.len() <= max_ctx,
        "a {}-token corpus does not fit --ctx {} — raise it or score a shorter text",
        ids.len(),
        max_ctx
    );
    Ok(())
}

/// The forced-scoring walk shared by the sync arms (Glimmer, V4, K3): score position
/// `i-1`'s row against `ids[i]`, then force `ids[i]`. One author, because the first
/// draft had this loop four times and the duplication gate reported all the pairs.
///
/// `step` is one arm's turn: hand back the logits row the last forward produced
/// together with the device's own argmax over it, then — unless `next` is `None` (the
/// final position) — consume the forced `(token, position)` through the arm's OWN
/// decode-step calls (`forward`/`argmax`, `hidden_state`/`sample_x`). That reuse is the
/// whole fidelity argument: the TF path must BE the decode path. A closure rather than
/// a trait, deliberately — the first factoring used a per-arm `Steps` impl and the
/// three identical struct/impl/signature skeletons were themselves the next thing the
/// duplication gate reported.
///
/// GLM deliberately does NOT come through here: its forward is async and its whole
/// score runs inside ONE `block_on` (the decode loop's own shape — a per-token
/// `block_on` would be a different runtime pattern than the engine it claims to
/// measure), so it keeps a bespoke loop over the same [`Tally`].
pub fn walk(
    ids: &[u32],
    mut step: impl FnMut(Option<(u32, usize)>) -> Result<(Vec<f32>, u32)>,
) -> Result<Tally> {
    let mut tally = Tally::new(ids.len());
    for i in 1..ids.len() {
        let next = (i + 1 < ids.len()).then_some((ids[i], i));
        let (row, own) = step(next)?;
        tally.push(&row, own, ids[i])?;
    }
    Ok(tally)
}

/// The device argmax kernel's fold, on the host: strict `>` (ties keep the LOWEST
/// index), NaN never wins because `NaN > x` is false. Returns `(index, value)`.
///
/// **The seed is `(0, -inf)`, and it is the whole correctness of this function.** It is
/// what `argmax_reduce`'s identity is (`kernels/fwd.hip`: `float best = -INFINITY; int
/// bidx = 0;`), and seeding from `row[0]` instead — which this did until 2026-08-16, and
/// which its own NaN test caught — silently diverges on exactly one input: a row whose
/// FIRST element is NaN. `NaN > x` is false in both directions, so a `row[0]` seed makes
/// the NaN win the fold and index 0 the answer, while the device (whose `amax_combine`
/// spells `va != va` out) returns the largest finite index. That is the `f32::max`
/// NaN-swallowing class in its exact repo-native form: the two folds disagree, and the
/// disagreement surfaces as a bogus *coherence* refusal instead of the true "non-finite
/// logits" one. With the identity seed an all-NaN row yields `-inf`, which is what the
/// device returns for it too, and `nll_of`'s finite check then refuses with the accurate
/// reason.
fn host_argmax(row: &[f32]) -> Result<(usize, f32)> {
    ensure!(!row.is_empty(), "argmax over an empty logit row");
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    Ok(best)
}

/// `-ln softmax(row)[target]`, shifted by the row max before exponentiating.
///
/// The shift is load-bearing rather than tidy: GLM's raw logits routinely run past 80,
/// `exp(80)` overflows f32 to `inf`, and an `inf` in the sum yields a NaN NLL that would
/// propagate into a mean still *looking* like a plausible perplexity. Accumulated in f64
/// for the same reason — 154,880 f32 addends lose real precision to rounding. (Both
/// measurements are the old tree's, `old:src/eval.rs`, inherited with the code.)
pub fn nll_of(row: &[f32], target: usize) -> Result<f32> {
    ensure!(
        target < row.len(),
        "target {target} outside {} logits",
        row.len()
    );
    let (_, max) = host_argmax(row)?;
    ensure!(max.is_finite(), "non-finite logits");
    let sum: f64 = row.iter().map(|&z| f64::from(z - max).exp()).sum();
    let nll = (sum.ln() - f64::from(row[target] - max)) as f32;
    ensure!(nll.is_finite(), "non-finite NLL");
    Ok(nll)
}

/// Score one forced position: the coherence check, then the NLL.
///
/// `own` is the DEVICE's argmax over this row (the token the decode loop would have
/// consumed); the host fold over the D2H'd bytes must land on the same index, or the row
/// being scored is not the row the engine acted on and the whole run is refused. Exact
/// index equality, not a tolerance: both folds use the same tie rule over the same f32
/// bytes, so any disagreement is a defect, never noise.
pub fn score_row(row: &[f32], own: u32, target: u32) -> Result<f32> {
    let (hi, hv) = host_argmax(row)?;
    ensure!(
        hi == own as usize,
        "TF coherence check failed: device argmax {own} but the host fold over the \
         scored row gives {hi} (value {hv}) — the row being scored is not the row the \
         decode path produced (stale buffer, wrong offset, or a desynced position)",
    );
    nll_of(row, target as usize)
}

/// Running tally of a forced-scoring loop — the shared body of every arm's `score`, so
/// the four loops stay eight divergent lines each instead of forty identical ones (the
/// jscpd gate reported exactly that shape on the first draft of the four).
pub struct Tally {
    nlls: Vec<f32>,
    agree: usize,
    /// The corpus length this tally was opened for, kept only so [`Self::into_scored`]
    /// can refuse a SHORT walk. Without it a loop that exited early — a `break` added
    /// later, a `?` on a recoverable error, an arm whose closure stops advancing — closes
    /// into a perfectly well-formed `.nll` file with fewer positions than the text, and
    /// the only thing that would catch it is `bin/ppl` refusing to pair it against a
    /// full-length baseline. That is a check in a DIFFERENT tool, on a run that already
    /// happened, and only when someone pairs it (review, 2026-08-16).
    want: usize,
}

impl Tally {
    /// `n` = corpus length; the tally holds `n - 1` scores when complete.
    pub fn new(n: usize) -> Self {
        Self {
            nlls: Vec::with_capacity(n.saturating_sub(1)),
            agree: 0,
            want: n.saturating_sub(1),
        }
    }

    /// Score `row` (device argmax `own`) against the about-to-be-forced `target`.
    pub fn push(&mut self, row: &[f32], own: u32, target: u32) -> Result<()> {
        self.nlls.push(score_row(row, own, target)?);
        self.agree += usize::from(own == target);
        Ok(())
    }

    /// Close the run with the arm's rebased fetch counters, refusing a short walk.
    ///
    /// `ensure!`, not `debug_assert!`: benchmarks and every `--ppl` run of consequence are
    /// `--release` builds, where a debug assert enforces nothing at all, and a silently
    /// short `.nll` is exactly the failure that survives to become a quality claim.
    pub fn into_scored(self, hits: u64, misses: u64) -> Result<Scored> {
        ensure!(
            self.nlls.len() == self.want,
            "scored {} positions of {} — the forced walk ended early, and a short .nll \
             file is indistinguishable from a complete one once written",
            self.nlls.len(),
            self.want,
        );
        Ok(Scored {
            nlls: self.nlls,
            agree: self.agree,
            hits,
            misses,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // tests: panic-on-failure is the idiom
    use super::*;

    /// The toy row the old tree pinned `nll_of` against — hand-computed softmax over 4
    /// logits, carried over so the re-typing (`&[u8]` → `&[f32]`) provably preserved the
    /// arithmetic.
    #[test]
    fn nll_of_toy_row_matches_the_hand_computation() {
        let row = [1.0f32, 2.0, 3.0, 0.0];
        // softmax([1,2,3,0]): shift by max=3 -> [-2,-1,0,-3], exp -> [.1353,.3679,1,.0498],
        // sum = 1.5530. -log(p[2]) = log(1.5530) = 0.44024...
        let nll = nll_of(&row, 2).unwrap();
        assert!((nll - 0.440_24).abs() < 1e-3, "got {nll}");
        // The least likely target (index 3, logit 0) has the largest NLL of the four.
        assert!(nll_of(&row, 3).unwrap() > nll);
        // A target past the row is a refusal, not an index panic.
        assert!(nll_of(&row, 4).is_err());
    }

    /// Large logits overflow f32 exp without the max shift — the defect the shift
    /// exists for, pinned so a "simplification" cannot reintroduce it.
    #[test]
    fn the_max_shift_survives_logits_past_f32_exp_range() {
        let row = [90.0f32, 88.0, 10.0];
        let nll = nll_of(&row, 0).unwrap();
        assert!(nll.is_finite() && nll > 0.0 && nll < 1.0, "got {nll}");
        // All-NaN (and any non-finite max) refuses loudly — f32::max-style NaN
        // swallowing scored a broken kernel as a PERFECT match once; never again.
        assert!(nll_of(&[f32::NAN, f32::NAN], 0).is_err());
        assert!(nll_of(&[f32::INFINITY, 1.0], 0).is_err());
    }

    /// The host fold IS the device kernel's rule: strict `>` keeps the lowest index on
    /// ties, and NaN never wins — **including at index 0**, which is the one input a
    /// `row[0]` seed gets wrong (it did, until 2026-08-16; this test is what found it).
    #[test]
    fn host_argmax_ties_keep_lowest_and_nan_never_wins() {
        assert_eq!(host_argmax(&[1.0, 3.0, 3.0]).unwrap().0, 1, "tie -> lowest");
        assert_eq!(host_argmax(&[f32::NAN, 2.0, 1.0]).unwrap().0, 1);
        assert_eq!(host_argmax(&[2.0, f32::NAN]).unwrap().0, 0);
        // A row with no finite element returns the IDENTITY, `-inf` at index 0 — the
        // same pair `argmax_reduce` returns for it — so the caller's finite check sees a
        // non-finite max and refuses. An all-NaN row scoring as a plausible maximum is
        // how a broken kernel once passed 9/9.
        let (i, v) = host_argmax(&[f32::NAN, f32::NAN]).unwrap();
        assert_eq!(i, 0);
        assert!(!v.is_finite(), "an all-NaN row must not yield a finite max");
        assert!(host_argmax(&[]).is_err());
    }

    /// The coherence check refuses a row whose host argmax disagrees with the device's —
    /// the module's whole reason to exist, shown able to go red.
    #[test]
    fn score_row_refuses_a_desynced_row() {
        let row = [0.1f32, 5.0, 0.2];
        // Agreeing device argmax: scores.
        assert!(score_row(&row, 1, 2).is_ok());
        // Disagreeing device argmax: the run refuses rather than scoring a wrong row.
        let err = score_row(&row, 0, 2).unwrap_err().to_string();
        assert!(err.contains("coherence"), "unexpected error: {err}");
    }

    /// The tally accumulates scores and agreement, and closes into a `Scored`.
    #[test]
    fn tally_counts_scores_and_agreement() {
        let mut t = Tally::new(3);
        let row = [0.0f32, 4.0, 1.0];
        t.push(&row, 1, 1).unwrap(); // own == target
        t.push(&row, 1, 2).unwrap(); // own != target
        let s = t.into_scored(7, 3).unwrap();
        assert_eq!(s.nlls.len(), 2);
        assert_eq!(s.agree, 1);
        assert_eq!((s.hits, s.misses), (7, 3));
        assert!(s.nlls[0] < s.nlls[1], "the forced argmax scores lower");
    }

    /// The door refuses what the walk could not score, naming the knob.
    #[test]
    fn admit_refuses_short_corpora_and_overlong_ones() {
        assert!(admit(&[1], 100).is_err());
        assert!(admit(&[1, 2], 100).is_ok());
        assert!(admit(&[1, 2, 3], 2).is_err());
    }

    /// The walk against a host-arithmetic step: positions score in order, the last
    /// position gets `next = None` (nothing left to force), and every other advance
    /// carries the forced token at its own index — the exact protocol the three sync
    /// arms' closures implement.
    #[test]
    fn walk_scores_every_position_and_forces_all_but_the_last() {
        let ids = [10u32, 1, 2, 1];
        let mut advanced = Vec::new();
        let t = walk(&ids, |next| {
            if let Some(n) = next {
                advanced.push(n);
            }
            // Row whose argmax is index 1 — `own` must agree or score_row refuses.
            Ok((vec![0.0, 3.0, 1.0], 1))
        })
        .unwrap();
        let out = t.into_scored(0, 0).unwrap();
        assert_eq!(out.nlls.len(), 3, "one NLL per position with a target");
        assert_eq!(
            advanced,
            vec![(1, 1), (2, 2)],
            "every position but the last is forced, at its own index"
        );
        assert_eq!(out.agree, 2, "targets 1,2,1 against constant own=1");
    }
}
