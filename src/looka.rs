//! LOOKA — look-ahead router recall counters (CACHE_PILOT Step 1).
//!
//! Answers ONE question, and deliberately nothing else: if we ran layer `L+h`'s router
//! against layer `L`'s post-attention residual, how many of `L+h`'s real experts would we
//! name? That fraction — recall — is the gate on the whole speculative loader, because a
//! prefetcher that names the wrong experts moves *more* bytes, not fewer, and the engine is
//! fetch-bound (docs/ARCHITECTURE.md §3). Recall is unobservable offline: `bin/replay`'s
//! oracle saturates by construction (a decision needs `top_k` keys and `top_k` admissions
//! fit any pool that holds one batch), so this instrumentation is the only thing that
//! produces the number.
//!
//! Two horizons, because `top-m` needs L+2 and the L+1↔L+2 difference is a number nobody
//! has. Colibri's yardstick on the same architecture (unquantized, 48 greedy tokens):
//! **71.6% at L+1** vs **41.3%** for the previous-token null hypothesis.
//!
//! The null hypothesis is carried here on purpose. "Predict L+h from L" only earns the
//! loader if it beats "reuse what this layer picked last token", which costs zero compute
//! and zero D2H — so a pilot that ties it is a pilot not worth building.
//!
//! Behind `--features trace`: compiled out of default builds entirely, since the pilot
//! costs a real rmsnorm + gemv per horizon and one D2H per MoE layer that the forward pass
//! does not otherwise pay.

/// How far ahead to predict. Index into the per-layer stash is the position in this array.
pub const HORIZONS: [usize; 2] = [1, 2];

/// Ranks tracked for the precision curve. `top_k` is 8 today; the slack costs two words
/// and means a wider `top_k` does not silently truncate the curve.
const MAX_RANK: usize = 16;

/// Recall accumulators plus the routing scratch the pilot needs to top-k its own logits.
pub struct Looka {
    /// Summed per-decision recall (intersection size) and the matching denominator, per
    /// horizon. Kept as a byte count rather than a mean-of-means so layers with different
    /// `sel` lengths (a short final batch) weight correctly.
    hit: [u64; HORIZONS.len()],
    tot: [u64; HORIZONS.len()],
    /// Same, for "the experts this layer chose for the PREVIOUS token".
    prev_hit: u64,
    prev_tot: u64,
    /// **Precision at rank**: `rank_hit[h][r]` counts how often the pilot's r-th ranked
    /// guess (by gate score, descending — `topk_into` sorts value-desc) turned out to be
    /// one the layer really picked. This, not aggregate recall, is what gates the
    /// speculative loader: on a bandwidth-bound engine a wrong speculative fetch costs
    /// real throughput, so the loader needs a PREFIX it can trust, not a good average.
    rank_hit: [[u64; MAX_RANK]; HORIZONS.len()],
    rank_tot: [[u64; MAX_RANK]; HORIZONS.len()],
    /// `pred[target][h]` — written at layer `target - HORIZONS[h]`, read when `target`
    /// routes. Same-token and strictly write-before-read, so no generation tag is needed.
    pred: Vec<[Vec<u32>; HORIZONS.len()]>,
    /// `prev[layer]` — what this layer selected on the previous token.
    prev: Vec<Vec<u32>>,
    /// Routing scratch, kept off the hot path's `scores`/`choice`/`sel`/`cand` so a pilot
    /// can never perturb the real selection.
    pub p_scores: Vec<f32>,
    pub p_choice: Vec<f32>,
    pub p_sel: Vec<usize>,
    pub p_cand: Vec<usize>,
    /// Host landing zone for the pilot logits of BOTH horizons (one D2H, not two).
    pub host: Vec<u8>,
}

impl Looka {
    pub fn new(n_layers: usize, n_experts: usize) -> Self {
        Self {
            hit: [0; HORIZONS.len()],
            tot: [0; HORIZONS.len()],
            prev_hit: 0,
            prev_tot: 0,
            rank_hit: [[0; MAX_RANK]; HORIZONS.len()],
            rank_tot: [[0; MAX_RANK]; HORIZONS.len()],
            pred: (0..n_layers).map(|_| Default::default()).collect(),
            prev: vec![Vec::new(); n_layers],
            p_scores: vec![0.0; n_experts],
            p_choice: vec![0.0; n_experts],
            p_sel: Vec::with_capacity(16),
            p_cand: Vec::new(),
            host: Vec::new(),
        }
    }

    /// Record the pilot's guess for `target`, made at layer `target - HORIZONS[h]`.
    pub fn stash(&mut self, target: usize, h: usize, sel: &[usize]) {
        let slot = &mut self.pred[target][h];
        slot.clear();
        slot.extend(sel.iter().map(|&e| e as u32));
    }

    /// Score every stashed prediction for `layer` against what it actually routed, then
    /// roll `sel` into the previous-token baseline. Call once per MoE layer, right after
    /// the real `sel` is filled.
    pub fn score(&mut self, layer: usize, sel: &[usize]) {
        if sel.is_empty() {
            return;
        }
        let n = sel.len() as u64;
        for h in 0..HORIZONS.len() {
            // A prediction is absent for the first HORIZONS[h] layers of each token, and
            // whenever the source layer was dense (no router to run). Skipping keeps those
            // out of the denominator rather than scoring them as misses.
            if layer < HORIZONS[h] || self.pred[layer][h].is_empty() {
                continue;
            }
            let p = &self.pred[layer][h];
            self.hit[h] += sel.iter().filter(|&&e| p.contains(&(e as u32))).count() as u64;
            self.tot[h] += n;
            // Precision at each rank. Scored per-guess (was THIS guess right?), the
            // reverse direction of the recall test above (was this real pick named?).
            for (r, &e) in p.iter().take(MAX_RANK).enumerate() {
                self.rank_tot[h][r] += 1;
                if sel.contains(&(e as usize)) {
                    self.rank_hit[h][r] += 1;
                }
            }
            // Consume it: a stale guess must never be scored twice if a later token's
            // source layer goes dense and leaves this slot unwritten.
            self.pred[layer][h].clear();
        }
        if !self.prev[layer].is_empty() {
            let q = &self.prev[layer];
            self.prev_hit += sel.iter().filter(|&&e| q.contains(&(e as u32))).count() as u64;
            self.prev_tot += n;
        }
        self.prev[layer].clear();
        self.prev[layer].extend(sel.iter().map(|&e| e as u32));
    }

    fn pct(hit: u64, tot: u64) -> f64 {
        if tot == 0 {
            0.0
        } else {
            100.0 * hit as f64 / tot as f64
        }
    }

    /// One line beside the PROFILE summary. Reports the baseline alongside every horizon
    /// because the horizons are only meaningful as a margin over it.
    pub fn report(&self) -> String {
        let base = Self::pct(self.prev_hit, self.prev_tot);
        let mut s = String::from("  LOOKA recall:");
        for (h, &n) in HORIZONS.iter().enumerate() {
            let r = Self::pct(self.hit[h], self.tot[h]);
            s.push_str(&format!(
                " L+{}: {:.1}% ({:+.1}pp vs prev-token, n={})",
                n,
                r,
                r - base,
                self.tot[h]
            ));
        }
        s.push_str(&format!(
            " | prev-token baseline {:.1}% (n={})",
            base, self.prev_tot
        ));
        s
    }

    /// The precision curve, one line per horizon. `p@r` is "the r-th guess was right this
    /// often"; `top-N` is the CUMULATIVE precision of speculating on the first N guesses,
    /// which is the number a confidence-gated loader is actually priced by — at cumulative
    /// precision `p`, covering `u` useful experts costs `u/p` fetches, and everything above
    /// `u` is wasted bandwidth on an engine that has none to spare.
    pub fn rank_report(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (h, &dh) in HORIZONS.iter().enumerate() {
            let live = (0..MAX_RANK)
                .filter(|&r| self.rank_tot[h][r] > 0)
                .collect::<Vec<_>>();
            if live.is_empty() {
                continue;
            }
            let per: Vec<String> = live
                .iter()
                .map(|&r| format!("p@{}:{:.0}%", r, Self::pct(self.rank_hit[h][r], self.rank_tot[h][r])))
                .collect();
            let (mut ch, mut ct) = (0u64, 0u64);
            let cum: Vec<String> = live
                .iter()
                .map(|&r| {
                    ch += self.rank_hit[h][r];
                    ct += self.rank_tot[h][r];
                    format!("top{}:{:.1}%", r + 1, Self::pct(ch, ct))
                })
                .collect();
            out.push(format!(
                "  LOOKA precision L+{}: {} | cumulative {}",
                dh,
                per.join(" "),
                cum.join(" ")
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_is_intersection_over_actual() {
        let mut lk = Looka::new(8, 256);
        // Layer 3 is told, from layer 2 (h=0 → L+1), that it will pick 4 of these 8.
        lk.stash(3, 0, &[10, 11, 12, 13, 90, 91, 92, 93]);
        lk.score(3, &[10, 11, 12, 13, 20, 21, 22, 23]);
        assert_eq!(lk.hit[0], 4);
        assert_eq!(lk.tot[0], 8);
        // First sighting of a layer has no previous token, so the baseline stays empty
        // rather than scoring a free zero.
        assert_eq!(lk.prev_tot, 0);
        // Second token: the baseline now compares against the first token's picks.
        lk.score(3, &[10, 11, 20, 21, 30, 31, 32, 33]);
        assert_eq!(lk.prev_hit, 4);
        assert_eq!(lk.prev_tot, 8);
        // ...and the consumed prediction was not double-counted.
        assert_eq!(lk.tot[0], 8);
    }

    #[test]
    fn precision_is_scored_per_guess_in_rank_order() {
        let mut lk = Looka::new(8, 256);
        // Ranks 0 and 2 are right; 1 and 3 are wrong.
        lk.stash(3, 0, &[10, 77, 11, 88]);
        lk.score(3, &[10, 11, 12, 13, 20, 21, 22, 23]);
        assert_eq!((lk.rank_hit[0][0], lk.rank_tot[0][0]), (1, 1));
        assert_eq!((lk.rank_hit[0][1], lk.rank_tot[0][1]), (0, 1));
        assert_eq!((lk.rank_hit[0][2], lk.rank_tot[0][2]), (1, 1));
        assert_eq!((lk.rank_hit[0][3], lk.rank_tot[0][3]), (0, 1));
    }

    #[test]
    fn cumulative_precision_at_full_width_equals_recall() {
        // When |pred| == |actual|, precision over the whole prefix and recall are the
        // same ratio — a self-check that the two counters cannot silently disagree.
        let mut lk = Looka::new(8, 256);
        lk.stash(3, 0, &[10, 11, 12, 13, 90, 91, 92, 93]);
        lk.score(3, &[10, 11, 12, 13, 20, 21, 22, 23]);
        let rank_sum: u64 = lk.rank_hit[0].iter().sum();
        let rank_tot: u64 = lk.rank_tot[0].iter().sum();
        assert_eq!((rank_sum, rank_tot), (lk.hit[0], lk.tot[0]));
    }

    #[test]
    fn absent_prediction_is_excluded_not_zeroed() {
        let mut lk = Looka::new(8, 256);
        lk.score(5, &[1, 2, 3]); // nothing stashed: no denominator
        assert_eq!(lk.tot[0], 0);
        assert_eq!(lk.tot[1], 0);
    }
}
