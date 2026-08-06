//! Shared substrate for the routed-expert cache policies. The POLICIES themselves
//! (LRU / 2Q / ARC) live in [`crate::memory::hybrid`] — one byte-aware family used both LIVE
//! by the pin and offline by `bin/replay` (with unit strides). This module holds only
//! what both need: [`OrderedSet`] (the recency structure), [`Tier`] (which residency
//! tier a key landed in → which format slab), and [`TwoQSplit`] (2Q's Kin/Kout knobs).

use anyhow::{Result, ensure};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Recency-ordered set of keys: O(log n) MRU-insert / arbitrary-remove / pop-LRU
/// via a monotonic tick clock + a `BTreeMap` whose front is the LRU end.
#[derive(Default)]
pub(crate) struct OrderedSet {
    at: HashMap<u32, i64>,
    order: BTreeMap<i64, u32>,
    clock: i64, // ascending MRU inserts
}

impl OrderedSet {
    pub(crate) fn contains(&self, k: u32) -> bool {
        self.at.contains_key(&k)
    }
    pub(crate) fn len(&self) -> usize {
        self.at.len()
    }
    /// Insert `k`, or move it to the MRU end if already present.
    pub(crate) fn touch(&mut self, k: u32) {
        self.clock += 1;
        if let Some(&t) = self.at.get(&k) {
            self.order.remove(&t);
        }
        self.at.insert(k, self.clock);
        self.order.insert(self.clock, k);
    }
    pub(crate) fn remove(&mut self, k: u32) -> bool {
        if let Some(t) = self.at.remove(&k) {
            self.order.remove(&t);
            return true;
        }
        false
    }
    /// Evict and return the LRU (oldest) key.
    pub(crate) fn pop_lru(&mut self) -> Option<u32> {
        let (&t, &k) = self.order.iter().next()?;
        self.order.remove(&t);
        self.at.remove(&k);
        Some(k)
    }
    /// The LRU key (and tick) without removing — for cross-set recency comparison (the
    /// hybrid LRU picks the globally oldest across both tiers), skipping `must`.
    pub(crate) fn peek_lru_skip(&self, must: &HashSet<u32>) -> Option<(i64, u32)> {
        self.order
            .iter()
            .find(|(_, k)| !must.contains(k))
            .map(|(t, k)| (*t, *k))
    }
    /// Evict and return the LRU key, skipping `must`.
    ///
    /// `must` is the per-batch pin set: a key touched (hit or admitted) earlier in this
    /// batch has to stay resident or the pin cannot resolve its slot ("expert not resident
    /// after alloc"). That is a correctness constraint and is never relaxed.
    ///
    /// This took a second, ADVISORY skip set until 2026-07-31 — the LOOKA hint veto, which
    /// could never be allowed to fail an allocation and so fell back to `must` alone. The
    /// hint layer is gone (measured: the vetoes bound on ~0.9% of evictions and moved hit
    /// rate by at most +0.1pp), and with it the two-authority scan and its fallback.
    pub(crate) fn pop_lru_skip(&mut self, must: &HashSet<u32>) -> Option<u32> {
        let t = self.peek_lru_skip(must)?.0;
        let k = self.order.remove(&t)?;
        self.at.remove(&k);
        Some(k)
    }
}

/// Which residency tier a key landed in — the ONLY thing the pool needs to pick a
/// key's format slab (`Hot` → int4, `Cold` → int3-VQ). A policy-agnostic concept, NOT
/// a 2Q segment: `Hot` is the proven-frequent tier (2Q's Am / ARC's T2), `Cold` the
/// probation tier (2Q's A1in / ARC's T1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Cold,
    Hot,
}

/// 2Q's fixed split (the paper's Kin/Kout), expressed as PERCENTAGES of capacity so
/// one setting transfers across pool sizes. `Kin` bounds the A1in probation
/// (resident); `Kout` bounds the A1out ghost (key-only, not resident). The byte-aware
/// [`crate::memory::hybrid::HybridTwoQ`] scales these against its budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoQSplit {
    kin_pct: u32,
    kout_pct: u32,
}

impl Default for TwoQSplit {
    /// Tuned by the offline `replay` sweep on a CLEAN trace (i.e. captured after the
    /// 2Q mid-layer eviction fix) and confirmed on hardware: predicted 77.56 %
    /// residency, measured 77.4 %, 0.95 -> 1.14 tok/s at 512 tokens.
    ///
    /// The optimum is a broad plateau (kin 6-10 % x kout 15-25 %, all within 0.1 pp),
    /// so these are robust rather than knife-edge. Kout below ~5 % collapses — the
    /// ghost gets too small to promote anything — but that cliff is far away.
    ///
    /// **Kout is the axis that matters**, and small wins: 20 % scores 77.56 % where
    /// 100 % scores 71.43 %. A large ghost remembers keys evicted long ago, so a
    /// re-miss on a stale key promotes it into the protected set; with 13k unique
    /// experts over ~3.6k slots most such re-references are spurious, and the
    /// protected set fills with one-hit-wonders. A small ghost only promotes keys
    /// re-referenced RECENTLY, which is a much stronger signal at this breadth.
    /// (This is also the leading explanation for ARC underperforming here: its B1/B2
    /// ghosts are full-capacity by construction, pinning it to the bad end of this
    /// axis with no way to tune off it.)
    ///
    /// > **CORRECTED 2026-08-02 — two of the claims above no longer hold at today's
    /// > pool, and the numbers that produced them were measured at ~3.6k slots.**
    /// > Re-swept at **6874** slots (int3-vq, `--max-mem 115`) on two preserved traces: a
    /// > 769-token prefill (18,474 unique) and the 288-pass decode (14,982 unique) already
    /// > kept for the OPT work. See docs/measurement/benchmarks.md, "2Q kin/kout re-sweep".
    /// >
    /// > 1. **"Kout is the axis that matters" is now false, and the mechanism says why.**
    /// >    The 6.1 pp spread above (20 % 77.56 vs 100 % 71.43) is **0.46 pp** on prefill
    /// >    and **0.17 pp** on decode today. The stated cause — spurious promotions from a
    /// >    stale ghost — scales with pressure, and unique/slots fell from 13k/3.6k = 3.6
    /// >    to 2.7 and 2.2. The reasoning survives; the effect shrank out from under it.
    /// > 2. **The whole grid is flat, so these constants are a don't-care.** Across
    /// >    kin 3-66 % x kout 25-800 % the total range is **0.51 pp** (decode) and
    /// >    **0.92 pp** (prefill). The best cells beat this default by only 0.37 and
    /// >    0.23 pp, and they are not the same cell — 40 %/25 % on decode, 20 %/25 % on
    /// >    prefill — which is what a best cell drawn from noise looks like. Note kout 25 %
    /// >    now edges the shipped 20 % on both. **Do not retune on this evidence**; there
    /// >    is nothing here to win, and a knife-edge default is worse than a flat one.
    /// > 3. **The ARC parenthetical is not supported, and ARC's failure is CONDITIONAL.**
    /// >    At batch 1 ARC does not underperform at all — 119.46 reads/token against LRU
    /// >    117.66 and 2Q 119.62 on decode; 157.78 against 157.90 and 154.75 on prefill.
    /// >    It collapses only once a batch's working set exceeds capacity (point 5): on
    /// >    prefill at batch 32 it reads 219.79 against 2Q's 157.55, while on decode at
    /// >    batch 32 it reads 117.23 and BEATS 2Q. Where it does collapse, full-capacity
    /// >    ghosts cannot be the cause — the whole kout axis is worth 0.7 pp there against
    /// >    a ~10 pp gap. The surviving candidate is ARC's ADAPTIVE `p`, which grows the
    /// >    recency half on B1 ghost hits — spurious under a batch union, where a key
    /// >    recurs because the whole LAYER recurs. **Untested**; A1out and B1/B2 are not
    /// >    the same mechanism (A1out feeds promotion, B1/B2 drive adaptation), so this
    /// >    refutes the stated explanation without establishing its replacement.
    /// > 4. **Under batching these constants become load-bearing.** At batch 32 on prefill
    /// >    the same grid spans **1.81 % to 43.42 %** and kin 3 %/kout 50 % beats this
    /// >    default by **+8.26 pp**. Small kin wins because A1in is a hard cap on how much
    /// >    of a scan reaches the protected set.
    /// > 5. **What predicts all of the above is one number: a batch's working set,
    /// >    `distinct-experts-per-token x batch`, against capacity.** Under it, every
    /// >    policy is within a few reads/token and the grid is flat. Over it, nothing a
    /// >    batch loads survives into the next and recency-driven policies collapse. The
    /// >    two traces straddle the line at batch 32 — prefill 7,776 > 6874 (collapse),
    /// >    decode 6,621 < 6874 (no collapse) — which is what makes it a rule rather than
    /// >    a curve. Nothing in the engine batches the MoE beyond the 2-row verify pass
    /// >    today, so points 3-5 are prerequisites for batched prefill, not live tuning.
    fn default() -> Self {
        Self {
            kin_pct: 8,
            kout_pct: 20,
        }
    }
}

impl TwoQSplit {
    /// Validate a percentage pair. `kin_pct` is capped at 99 because at `kin >= cap`
    /// `reclaim` can never trim A1in, so with an empty Am it evicts nothing and
    /// residency grows past capacity — which would break the pin's slot invariant.
    /// `kout_pct` may exceed 100: A1out holds keys only (4 bytes), so a ghost larger
    /// than the resident set is cheap and remembers more second-access candidates.
    ///
    /// `anyhow`, not a typed error: this is reached once, from CLI parsing, and nothing
    /// ever matched on which bound was violated — the message IS the value. The typed
    /// enum it replaced cost 20 lines and two trait impls to say the same thing.
    pub fn new(kin_pct: u32, kout_pct: u32) -> Result<Self> {
        ensure!(
            (1..=99).contains(&kin_pct),
            "--2q-kin {kin_pct}: must be 1..=99 (percent of capacity)"
        );
        ensure!(
            (1..=1000).contains(&kout_pct),
            "--2q-kout {kout_pct}: must be 1..=1000 (percent of capacity)"
        );
        Ok(Self { kin_pct, kout_pct })
    }

    pub fn kin_pct(self) -> u32 {
        self.kin_pct
    }
    pub fn kout_pct(self) -> u32 {
        self.kout_pct
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The bounds, and that the message names the offending flag and value — the
    /// message is the whole payload now that the error is untyped.
    #[test]
    fn split_rejects_out_of_range() {
        for (kin, kout, want) in [
            (0, 50, "--2q-kin 0"),
            (100, 50, "--2q-kin 100"),
            (25, 0, "--2q-kout 0"),
            (25, 1001, "--2q-kout 1001"),
        ] {
            let e = TwoQSplit::new(kin, kout).expect_err("out of range must be rejected");
            assert!(
                e.to_string().contains(want),
                "message {e:?} should name {want}"
            );
        }
        assert!(TwoQSplit::new(8, 20).is_ok());
    }
}
