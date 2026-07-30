//! Byte-aware routed-expert policies for the format hybrid. Unlike the single-format
//! [`cache`](crate::cache) policies (which count uniform slots), these manage TWO tiers
//! of DIFFERENT byte sizes — COLD (int3-VQ) and HOT (int4) — against a byte budget, so
//! the split between them FLOATS with the workload (and, for ARC, with the adaptive
//! `p`). The pin backs them with the two-ended [`arena`](crate::arena); this module is
//! pure bookkeeping (host-testable).
//!
//! Contract: `get` reports a hit and keeps the key in its tier (a resident expert never
//! migrates format without a refetch). `admit` handles a MISS — it places the key in a
//! tier and evicts (by the policy's own rule) until the incoming slot's bytes fit the
//! budget, returning every victim. The pin frees each victim's slot, then places the new
//! one (compacting the arena as needed).

use crate::cache::{OrderedSet, Tier};
use std::collections::{HashMap, HashSet};

/// The result of admitting a miss: which tier the key landed in, and every resident key
/// evicted to make its bytes fit (possibly several, since the tiers differ in size).
pub struct Admission {
    pub tier: Tier,
    pub evicted: Vec<u32>,
}

/// A LOOKA look-ahead prediction: layer `L`'s router run against layer `L+horizon`'s gate,
/// saying "this key will probably be wanted `horizon` layers from now".
///
/// **A hint is NOT an access, and the distinction is the whole design.** Measured precision
/// by rank is 99/96/93/87/78/67/55/42% at L+1 (docs/CACHE_PILOT.md), so a hint is evidence,
/// not fact. Every policy therefore applies exactly ONE rule — *hints veto eviction, they
/// never promote* — because promotion on a guess corrupts the very signal each policy exists
/// to track: 2Q's whole point is that a single access does not promote (a prediction is
/// weaker than an access), and ARC's `p` adapts on real hits (feeding it 78%-precision
/// guesses degrades the adaptivity it exists for).
///
/// Because routing no longer consults residency (`top-m` is gone), hints can only change
/// WHICH experts stay cached, never WHICH are selected — so enabling them is
/// **output-bit-identical by construction**, and that is the acceptance test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hint {
    pub key: u32,
    /// Layers ahead this prediction is for; also its time-to-live in layers.
    pub horizon: u8,
    /// 0 = the router's top pick. Lower is more trustworthy, and it is the tiebreak when
    /// hints exceed the cap.
    pub rank: u8,
}

/// Ceiling on how much of a tier hint-vetoes may protect, in percent of its byte budget.
/// Without it a wide horizon pins so much that eviction starves and `admit` fails outright —
/// a self-inflicted OOM from an *advisory* signal. Over the cap the highest-`rank` (least
/// trustworthy) hints are dropped first.
pub const HINT_CAP_PCT: usize = 25;

/// The hint bookkeeping, shared by every policy so the cap and the decay exist exactly ONCE.
///
/// Each policy embedding this gets identical semantics for free; a policy hand-rolling its
/// own would be free to quietly skip the guards, which is how "capped and decaying" becomes
/// "neither" over a few refactors.
#[derive(Default)]
pub struct HintSet {
    /// key -> layers remaining before this veto expires.
    live: std::collections::HashMap<u32, u8>,
    /// Mirror of `live`'s keys, because eviction asks this question on every victim scan and
    /// wants a `HashSet` without rebuilding one.
    keys: std::collections::HashSet<u32>,
    /// Max keys allowed to hold a veto at once (0 = uncapped, used only by tests).
    cap: usize,
    /// Diagnostics: hints offered, and how many named an already-resident key.
    pub seen: u64,
    pub seen_resident: u64,
}

impl HintSet {
    /// `cap` is derived from the tier's slot count, not from the hint count — the resource
    /// being protected is eviction headroom.
    pub fn with_cap(cap: usize) -> Self {
        Self { cap, ..Default::default() }
    }

    /// Merge in a batch of predictions. Rank order decides who survives the cap: rank 0 is
    /// the router's top pick (measured 99% precision), rank 7 is a coin flip.
    pub fn insert(&mut self, hints: &[Hint]) {
        let mut sorted: Vec<&Hint> = hints.iter().collect();
        sorted.sort_by_key(|h| h.rank);
        for h in sorted {
            // Refresh an existing veto to the longer horizon rather than double-counting.
            if let Some(ttl) = self.live.get_mut(&h.key) {
                *ttl = (*ttl).max(h.horizon.max(1));
                continue;
            }
            if self.cap != 0 && self.live.len() >= self.cap {
                break; // over cap: the remaining (higher-rank, less trustworthy) hints drop
            }
            self.live.insert(h.key, h.horizon.max(1));
            self.keys.insert(h.key);
        }
    }

    /// One layer elapsed. A hint that never came true must decay, or the veto set only ever
    /// grows and eviction is permanently starved by predictions about a layer long past.
    pub fn tick(&mut self) {
        self.live.retain(|_, ttl| {
            *ttl -= 1;
            *ttl > 0
        });
        // `keys` mirrors `live`'s key set. Letting the two drift means eviction keeps
        // skipping a key nothing remembers, which never surfaces as an error — only as a
        // quietly worse hit rate — so it is re-derived here rather than patched in parallel.
        self.keys.retain(|k| self.live.contains_key(k));
    }

    /// Drop a veto once the key is genuinely resident-and-used; keeps the cap for live
    /// predictions rather than for confirmed ones the policy already tracks.
    pub fn confirm(&mut self, k: u32) {
        self.live.remove(&k);
        self.keys.remove(&k);
    }

    pub fn keys(&self) -> &std::collections::HashSet<u32> {
        &self.keys
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }
}

pub trait HybridPolicy {
    fn contains(&self, k: u32) -> bool;
    /// A hit refreshes recency IN its tier and returns true; a miss returns false.
    fn get(&mut self, k: u32) -> bool;
    /// Admit a known-miss key: pick its tier, evict until it fits the byte budget.
    fn admit(&mut self, k: u32) -> Admission;
    /// Start a new batch (the pin's `submit_layer`): clears the per-batch pin set so the
    /// next batch's evictions may reclaim the previous batch's keys again.
    fn begin_batch(&mut self);
    /// Pin a just-hit key for the rest of this batch — eviction must not take it, else
    /// the pin can't resolve its slot ("expert not resident after alloc"). Touching to
    /// MRU is NOT enough: a big/skewed batch can drain a whole tier past the MRU end.
    fn protect(&mut self, k: u32);
    /// Feed look-ahead predictions for upcoming layers. See [`Hint`]: these VETO EVICTION
    /// and must never promote, admit, or touch a policy's internal adaptation state.
    ///
    /// Defaulted to a no-op so a policy opts in rather than silently ignoring hints while
    /// appearing to support them. Implementations delegate to [`HintSet`], which owns the
    /// cap and the decay — the two guards that otherwise silently do not happen.
    fn hint(&mut self, _hints: &[Hint]) {}
    /// Keys currently vetoed by live hints; eviction skips these. Empty unless [`hint`] is
    /// implemented.
    fn hinted(&self) -> &std::collections::HashSet<u32> {
        static EMPTY: std::sync::OnceLock<std::collections::HashSet<u32>> =
            std::sync::OnceLock::new();
        EMPTY.get_or_init(Default::default)
    }
    /// Advance the hint clock one layer, expiring anything whose horizon has elapsed.
    /// Called once per MoE layer by the pin.
    fn tick_hints(&mut self) {}
    /// `(hints seen, of those already RESIDENT when hinted)`. A veto can only ever protect
    /// a resident key — hinting an absent one is a no-op, since eviction cannot take what
    /// is not there. If these two numbers are far apart the mechanism has nothing to do,
    /// which is a different finding from "the wiring is broken" and needs a different fix.
    fn hint_stats(&self) -> (u64, u64) {
        (0, 0)
    }
    /// Bytes currently resident — the pin/tests check this never exceeds the budget.
    fn resident_bytes(&self) -> usize;
}


/// Byte geometry shared by every policy: the budget and the two per-tier slot sizes.
#[derive(Clone, Copy)]
struct Geom {
    budget: usize,
    cold_stride: usize,
    hot_stride: usize,
}
impl Geom {
    fn stride(&self, tier: Tier) -> usize {
        match tier {
            Tier::Cold => self.cold_stride,
            Tier::Hot => self.hot_stride,
        }
    }
}

/// Construct a byte-aware hybrid policy. `split` is 2Q's Kin/Kout (ignored by lru/arc);
impl Geom {
    /// How many keys hints may veto at once: [`HINT_CAP_PCT`] of the pool's COLD-slot
    /// capacity. Derived from the smaller stride so the cap is a bound on slots in the
    /// worst case, and floored at 1 so a tiny test pool still admits one veto.
    fn hint_cap(&self) -> usize {
        ((self.budget / self.cold_stride) * HINT_CAP_PCT / 100).max(1)
    }
}

pub fn make(
    policy: &str,
    budget: usize,
    cold_stride: usize,
    hot_stride: usize,
    split: crate::cache::TwoQSplit,
) -> Option<Box<dyn HybridPolicy>> {
    let g = Geom { budget, cold_stride: cold_stride.max(1), hot_stride: hot_stride.max(1) };
    match policy {
        "lru" => Some(Box::new(HybridLru::new(g))),
        // Three policies, three implementations. (`top-m` was a fourth NAME over the same
        // LRU with router substitution switched on; it was removed 2026-07-30 — the LOOKA
        // hint layer steers eviction instead of selection and is output-neutral, which
        // top-m was not: +3.63% ppl on int3-vq, +12.7% on int4. See docs/CACHE_ROUTE.md.)
        // This repo deleted three duplicate
        // policy families in 08db745; a copy-pasted `HybridTopM` would be that mistake
        // with a new name.
        "2q" => Some(Box::new(HybridTwoQ::new(g, split))),
        "arc" => Some(Box::new(HybridArc::new(g))),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// LRU — recency eviction, frequency-counter admission (LRU has no native hot/cold
// signal). See MODES.md.
// ---------------------------------------------------------------------------
const LRU_HOT_THRESHOLD: u32 = 2;
/// Halve `freq` every this many accesses so the count tracks RECENT frequency (a cooled
/// expert drops below threshold). ~7 tokens at ~600 routed accesses/token; independent
/// of budget so it scopes "recent" by workload time, not pool size.
const LRU_DECAY: u64 = 4096;

struct HybridLru {
    g: Geom,
    cold: OrderedSet,
    hot: OrderedSet,
    freq: HashMap<u32, u32>,
    accesses: u64,
    pinned: HashSet<u32>,
    /// LOOKA eviction vetoes. Advisory: `cache::pop_lru_skip` drops them rather than fail.
    hints: HintSet,
}
impl HybridLru {
    fn new(g: Geom) -> Self {
        Self {
            g,
            cold: OrderedSet::default(),
            hot: OrderedSet::default(),
            freq: HashMap::new(),
            accesses: 0,
            pinned: HashSet::new(),
            hints: HintSet::with_cap(g.hint_cap()),
        }
    }
    fn bump(&mut self, k: u32) {
        *self.freq.entry(k).or_insert(0) += 1;
        self.accesses += 1;
        if self.accesses.is_multiple_of(LRU_DECAY) {
            self.freq.retain(|_, v| {
                *v /= 2;
                *v > 0
            });
        }
    }
    /// Evict the older of the two tiers' LRU keys. NOTE: `cold` and `hot` are separate
    /// OrderedSets with independent clocks, so this is NOT a true global LRU — the tick
    /// comparison drifts by the cold:hot access ratio, biasing eviction toward the less
    /// frequently touched tier. That segments recency per tier (protecting the busier
    /// HOT tier), which is fine here; a true global LRU would need a shared clock.
    fn evict_lru(&mut self) -> Option<u32> {
        // Skip keys pinned this batch (peek AND pop), so a same-batch key is never evicted.
        match (self.cold.peek_lru_skip(&self.pinned, self.hints.keys()), self.hot.peek_lru_skip(&self.pinned, self.hints.keys())) {
            (Some((tc, _)), Some((th, _))) => {
                if tc <= th {
                    self.cold.pop_lru_skip(&self.pinned, self.hints.keys())
                } else {
                    self.hot.pop_lru_skip(&self.pinned, self.hints.keys())
                }
            }
            (Some(_), None) => self.cold.pop_lru_skip(&self.pinned, self.hints.keys()),
            (None, Some(_)) => self.hot.pop_lru_skip(&self.pinned, self.hints.keys()),
            (None, None) => None,
        }
    }
}
impl HybridPolicy for HybridLru {
    fn contains(&self, k: u32) -> bool {
        self.cold.contains(k) || self.hot.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        // A veto exists to protect a key UNTIL it is used. Once it is, the policy
        // tracks it properly and holding the veto only consumes cap that a live
        // prediction could use.
        self.hints.confirm(k);
        self.bump(k);
        if self.cold.contains(k) {
            self.cold.touch(k);
            true
        } else if self.hot.contains(k) {
            self.hot.touch(k);
            true
        } else {
            false
        }
    }
    fn admit(&mut self, k: u32) -> Admission {
        self.hints.confirm(k);
        let tier = if self.freq.get(&k).copied().unwrap_or(0) >= LRU_HOT_THRESHOLD {
            Tier::Hot
        } else {
            Tier::Cold
        };
        let mut evicted = Vec::new();
        while self.resident_bytes() + self.g.stride(tier) > self.g.budget {
            match self.evict_lru() {
                Some(v) => evicted.push(v),
                None => break,
            }
        }
        match tier {
            Tier::Cold => self.cold.touch(k),
            Tier::Hot => self.hot.touch(k),
        }
        self.pinned.insert(k); // a just-admitted miss must survive the rest of the batch
        Admission { tier, evicted }
    }
    fn begin_batch(&mut self) {
        self.pinned.clear();
    }
    fn protect(&mut self, k: u32) {
        self.pinned.insert(k);
    }

    fn hint(&mut self, hints: &[Hint]) {
        for h in hints {
            self.hints.seen += 1;
            if self.contains(h.key) {
                self.hints.seen_resident += 1;
            }
        }
        self.hints.insert(hints);
    }
    fn hint_stats(&self) -> (u64, u64) {
        (self.hints.seen, self.hints.seen_resident)
    }
    fn hinted(&self) -> &HashSet<u32> {
        self.hints.keys()
    }
    fn tick_hints(&mut self) {
        self.hints.tick();
    }
    fn resident_bytes(&self) -> usize {
        self.cold.len() * self.g.cold_stride + self.hot.len() * self.g.hot_stride
    }
}

// ---------------------------------------------------------------------------
// 2Q — A1in probation (COLD) bounded by Kin bytes; Am (HOT) absorbs the rest and
// floats; A1out ghost promotes a returning key to HOT on its next miss.
// ---------------------------------------------------------------------------
struct HybridTwoQ {
    g: Geom,
    kin_bytes: usize, // A1in (cold) byte bound
    kout: usize,      // A1out ghost length bound
    a1in: OrderedSet,
    am: OrderedSet,
    a1out: OrderedSet,
    pinned: HashSet<u32>,
    /// LOOKA eviction vetoes. Advisory: `cache::pop_lru_skip` drops them rather than fail.
    hints: HintSet,
}
impl HybridTwoQ {
    fn new(g: Geom, split: crate::cache::TwoQSplit) -> Self {
        let kin_bytes = (g.budget * split.kin_pct() as usize / 100).max(g.cold_stride);
        let slots = g.budget / g.cold_stride.min(g.hot_stride);
        let kout = (slots * split.kout_pct() as usize / 100).max(1);
        Self {
            g,
            kin_bytes,
            kout,
            a1in: OrderedSet::default(),
            am: OrderedSet::default(),
            a1out: OrderedSet::default(),
            pinned: HashSet::new(),
            hints: HintSet::with_cap(g.hint_cap()),
        }
    }
    fn a1in_bytes(&self) -> usize {
        self.a1in.len() * self.g.cold_stride
    }
    fn trim_a1in(&mut self) -> Option<u32> {
        let v = self.a1in.pop_lru_skip(&self.pinned, self.hints.keys());
        if let Some(v) = v {
            self.a1out.touch(v); // ghost is key-only (not resident) — no pin skip needed
            while self.a1out.len() > self.kout {
                self.a1out.pop_lru();
            }
        }
        v
    }
    /// Free one resident (never a key pinned this batch). Evict Am (HOT) while A1in is
    /// WITHIN its Kin bound (protects the frequent tier); otherwise trim A1in probation
    /// into the ghost. Each branch falls back to the other if its preferred segment has
    /// only pinned keys left, so a pinned-heavy batch can't stall the eviction.
    fn reclaim(&mut self) -> Option<u32> {
        if self.a1in_bytes() < self.kin_bytes {
            self.am.pop_lru_skip(&self.pinned, self.hints.keys()).or_else(|| self.trim_a1in())
        } else {
            self.trim_a1in().or_else(|| self.am.pop_lru_skip(&self.pinned, self.hints.keys()))
        }
    }
}
impl HybridPolicy for HybridTwoQ {
    fn contains(&self, k: u32) -> bool {
        self.am.contains(k) || self.a1in.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        // A veto exists to protect a key UNTIL it is used. Once it is, the policy
        // tracks it properly and holding the veto only consumes cap that a live
        // prediction could use.
        self.hints.confirm(k);
        if self.am.contains(k) {
            self.am.touch(k);
            return true;
        }
        self.a1in.contains(k) // A1in hit stays put (FIFO); ghosts are not resident
    }
    fn admit(&mut self, k: u32) -> Admission {
        self.hints.confirm(k);
        // A second distinct access (via the ghost) promotes to Am/HOT; else A1in/COLD.
        let tier = if self.a1out.remove(k) { Tier::Hot } else { Tier::Cold };
        let mut evicted = Vec::new();
        while self.resident_bytes() + self.g.stride(tier) > self.g.budget {
            match self.reclaim() {
                Some(v) => evicted.push(v),
                None => break,
            }
        }
        match tier {
            Tier::Hot => self.am.touch(k),
            Tier::Cold => self.a1in.touch(k),
        }
        self.pinned.insert(k); // survives the rest of the batch
        Admission { tier, evicted }
    }
    fn begin_batch(&mut self) {
        self.pinned.clear();
    }
    fn protect(&mut self, k: u32) {
        self.pinned.insert(k);
        // A1in is a FIFO and `get` leaves hits in place, so also move an actively-used
        // key to the young end (the 2Q recency intent, beyond the batch pin).
        if self.a1in.contains(k) {
            self.a1in.touch(k);
        }
    }

    fn hint(&mut self, hints: &[Hint]) {
        for h in hints {
            self.hints.seen += 1;
            if self.contains(h.key) {
                self.hints.seen_resident += 1;
            }
        }
        self.hints.insert(hints);
    }
    fn hint_stats(&self) -> (u64, u64) {
        (self.hints.seen, self.hints.seen_resident)
    }
    fn hinted(&self) -> &HashSet<u32> {
        self.hints.keys()
    }
    fn tick_hints(&mut self) {
        self.hints.tick();
    }
    fn resident_bytes(&self) -> usize {
        self.a1in.len() * self.g.cold_stride + self.am.len() * self.g.hot_stride
    }
}

// ---------------------------------------------------------------------------
// ARC — adaptive split. T1 (recency, COLD) / T2 (frequency, HOT) resident; B1/B2
// key-only ghosts drive the target `p` (in BYTES), which chooses the eviction tier.
// ---------------------------------------------------------------------------
struct HybridArc {
    g: Geom,
    p: usize, // target BYTES for T1 (cold); floats with the ghost hits
    t1: OrderedSet,
    t2: OrderedSet,
    b1: OrderedSet,
    b2: OrderedSet,
    pinned: HashSet<u32>,
    /// LOOKA eviction vetoes. Advisory: `cache::pop_lru_skip` drops them rather than fail.
    hints: HintSet,
}
impl HybridArc {
    fn new(g: Geom) -> Self {
        Self {
            g,
            p: 0,
            t1: OrderedSet::default(),
            t2: OrderedSet::default(),
            b1: OrderedSet::default(),
            b2: OrderedSet::default(),
            pinned: HashSet::new(),
            hints: HintSet::with_cap(g.hint_cap()),
        }
    }
    fn t1_bytes(&self) -> usize {
        self.t1.len() * self.g.cold_stride
    }
    /// Evict one UNPINNED resident to a ghost, choosing the tier by `p`: shed COLD (T1)
    /// while it exceeds the target `p`, else shed HOT (T2). `in_b2` biases toward T1 at
    /// the tie. Falls back to the other tier if the preferred one has no unpinned victim
    /// (empty OR all pinned this batch), so it never stalls or evicts a batch key.
    fn replace(&mut self, in_b2: bool) -> Option<u32> {
        let t1b = self.t1_bytes();
        let prefer_cold = t1b > self.p || (in_b2 && t1b == self.p);
        let (v, from_cold) = if prefer_cold {
            match self.t1.pop_lru_skip(&self.pinned, self.hints.keys()) {
                Some(v) => (Some(v), true),
                None => (self.t2.pop_lru_skip(&self.pinned, self.hints.keys()), false),
            }
        } else {
            match self.t2.pop_lru_skip(&self.pinned, self.hints.keys()) {
                Some(v) => (Some(v), false),
                None => (self.t1.pop_lru_skip(&self.pinned, self.hints.keys()), true),
            }
        };
        if let Some(v) = v {
            let ghost = if from_cold { &mut self.b1 } else { &mut self.b2 };
            ghost.touch(v);
            // Bound each ghost to a budget's worth of keys (cheap; remembers returns).
            let bound = self.g.budget / self.g.cold_stride.min(self.g.hot_stride);
            while ghost.len() > bound {
                ghost.pop_lru();
            }
        }
        v
    }
    fn evict_until_fits(&mut self, incoming: usize, in_b2: bool, evicted: &mut Vec<u32>) {
        while self.resident_bytes() + incoming > self.g.budget {
            match self.replace(in_b2) {
                Some(v) => evicted.push(v),
                None => break,
            }
        }
    }
}
impl HybridPolicy for HybridArc {
    fn contains(&self, k: u32) -> bool {
        self.t1.contains(k) || self.t2.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        // A veto exists to protect a key UNTIL it is used. Once it is, the policy
        // tracks it properly and holding the veto only consumes cap that a live
        // prediction could use.
        self.hints.confirm(k);
        // A hit STAYS in its tier (no slab migration); refresh recency in-place.
        if self.t1.contains(k) {
            self.t1.touch(k);
            true
        } else if self.t2.contains(k) {
            self.t2.touch(k);
            true
        } else {
            false
        }
    }
    fn admit(&mut self, k: u32) -> Admission {
        self.hints.confirm(k);
        let mut evicted = Vec::new();
        // A ghost hit is a returning key → promote to T2/HOT and adapt `p`.
        let tier = if self.b1.remove(k) {
            let delta = (self.b2.len() * self.g.hot_stride / self.b1.len().max(1)).max(self.g.cold_stride);
            self.p = (self.p + delta).min(self.g.budget);
            self.evict_until_fits(self.g.hot_stride, false, &mut evicted);
            self.t2.touch(k);
            Tier::Hot
        } else if self.b2.remove(k) {
            let delta = (self.b1.len() * self.g.cold_stride / self.b2.len().max(1)).max(self.g.hot_stride);
            self.p = self.p.saturating_sub(delta);
            self.evict_until_fits(self.g.hot_stride, true, &mut evicted);
            self.t2.touch(k);
            Tier::Hot
        } else {
            // Fresh miss → T1/COLD probation.
            self.evict_until_fits(self.g.cold_stride, false, &mut evicted);
            self.t1.touch(k);
            Tier::Cold
        };
        self.pinned.insert(k); // survives the rest of the batch
        Admission { tier, evicted }
    }
    fn begin_batch(&mut self) {
        self.pinned.clear();
    }
    fn protect(&mut self, k: u32) {
        // `get` already promotes the hit to T1/T2 MRU, but a drained tier can evict past
        // the MRU end, so pin it explicitly for the batch.
        self.pinned.insert(k);
    }

    fn hint(&mut self, hints: &[Hint]) {
        for h in hints {
            self.hints.seen += 1;
            if self.contains(h.key) {
                self.hints.seen_resident += 1;
            }
        }
        self.hints.insert(hints);
    }
    fn hint_stats(&self) -> (u64, u64) {
        (self.hints.seen, self.hints.seen_resident)
    }
    fn hinted(&self) -> &HashSet<u32> {
        self.hints.keys()
    }
    fn tick_hints(&mut self) {
        self.hints.tick();
    }
    fn resident_bytes(&self) -> usize {
        self.t1.len() * self.g.cold_stride + self.t2.len() * self.g.hot_stride
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::cache::TwoQSplit;

    // The spec harness: drive a policy through a workload two-pass-per-key (get, then
    // admit on miss) while an independent tally tracks every resident key's tier. After
    // each admit it asserts the load-bearing invariants:
    //   1. byte accounting agrees (policy.resident_bytes == tally) and never exceeds budget
    //   2. every evicted key was resident and is now gone
    //   3. the admitted key is resident
    struct Spec {
        budget: usize,
        cs: usize,
        hs: usize,
        at: HashMap<u32, Tier>,
    }
    impl Spec {
        fn new(budget: usize, cs: usize, hs: usize) -> Self {
            Spec { budget, cs, hs, at: HashMap::new() }
        }
        fn bytes(&self) -> usize {
            self.at.values().map(|t| if *t == Tier::Hot { self.hs } else { self.cs }).sum()
        }
        fn access(&mut self, p: &mut dyn HybridPolicy, k: u32) -> Option<Tier> {
            p.begin_batch(); // each access is its own 1-key batch in this harness
            if p.get(k) {
                assert!(self.at.contains_key(&k), "hit on a key the tally thinks is gone: {k}");
                return None;
            }
            let a = p.admit(k);
            for e in &a.evicted {
                assert!(self.at.remove(e).is_some(), "evicted {e} was not resident");
                assert!(!p.contains(*e), "evicted {e} still contained");
            }
            self.at.insert(k, a.tier);
            assert!(p.contains(k), "admitted {k} not resident");
            assert_eq!(self.bytes(), p.resident_bytes(), "byte accounting drift");
            assert!(self.bytes() <= self.budget, "over budget: {} > {}", self.bytes(), self.budget);
            Some(a.tier)
        }
    }

    fn each_policy() -> Vec<(&'static str, Box<dyn HybridPolicy>)> {
        let (budget, cs, hs) = (100usize, 3usize, 4usize);
        ["lru", "2q", "arc"]
            .iter()
            .map(|&n| {
                let p = make(n, budget, cs, hs, TwoQSplit::default());
                (n, p.expect("known policy"))
            })
            .collect()
    }



    #[test]
    fn byte_budget_and_residency_hold_for_all() {
        for (name, mut p) in each_policy() {
            let mut s = Spec::new(100, 3, 4);
            // churny skewed workload well past capacity: a small hot core + rotating tail
            for i in 0..2000u32 {
                let k = if i % 3 == 0 { i % 7 } else { 100 + i };
                s.access(&mut *p, k);
            }
            assert!(!s.at.is_empty(), "{name}: policy emptied itself");
        }
    }

    #[test]
    fn fresh_is_cold_returning_is_hot() {
        // 2q and arc must place a first-seen key COLD and a returning (evicted then
        // re-missed) key HOT. Budget = 1 cold slot so the second key evicts the first.
        for name in ["2q", "arc"] {
            // budget 5 holds exactly one hot slot (4) — small enough that the 2nd admit
            // evicts the 1st, big enough that a HOT slot fits.
            let mut p = make(name, 5, 3, 4, TwoQSplit::default()).unwrap();
            p.begin_batch();
            assert!(!p.get(10));
            assert_eq!(p.admit(10).tier, Tier::Cold, "{name}: first-seen must be COLD");
            p.begin_batch();
            assert!(!p.get(20)); // evicts 10 (cold slot reused)
            let _ = p.admit(20);
            assert!(!p.contains(10), "{name}: 10 should have been evicted");
            p.begin_batch();
            assert!(!p.get(10)); // 10 returns via the ghost
            assert_eq!(p.admit(10).tier, Tier::Hot, "{name}: a returning key must be HOT");
        }
    }

    #[test]
    fn lru_admits_by_frequency() {
        // LRU has no ghost: placement is the decaying counter. First-seen COLD; a key
        // re-accessed after eviction crosses the threshold → HOT.
        let mut p = make("lru", 5, 3, 4, TwoQSplit::default()).unwrap();
        p.begin_batch();
        assert!(!p.get(10));
        assert_eq!(p.admit(10).tier, Tier::Cold);
        p.begin_batch();
        assert!(!p.get(20));
        let _ = p.admit(20); // evicts 10
        p.begin_batch();
        assert!(!p.get(10)); // second access → freq 2
        assert_eq!(p.admit(10).tier, Tier::Hot, "re-accessed key must be HOT");
    }

    #[test]
    fn arc_p_adapts_toward_frequency() {
        // A frequency-skewed workload (a hot core hit via the ghost) must drive ARC's
        // `p` DOWN from 0-start toward HOT... p rises on B1 hits (recency), falls on B2.
        // Here we just assert the hot core stays resident under churn (adaptivity works).
        let mut p = make("arc", 60, 3, 4, TwoQSplit::default()).unwrap();
        let core: Vec<u32> = (0..5).collect();
        for round in 0..50u32 {
            for &k in &core {
                p.begin_batch();
                if !p.get(k) {
                    p.admit(k);
                }
            }
            // churn distinct tail keys to pressure the cache
            for t in 0..8u32 {
                p.begin_batch();
                let k = 1000 + round * 8 + t;
                if !p.get(k) {
                    p.admit(k);
                }
            }
        }
        let core_resident = core.iter().filter(|&&k| p.contains(k)).count();
        assert!(core_resident >= 3, "frequent core evaporated: {core_resident}/5 resident");
    }

    // Mirrors the pin's submit_spine BATCH protocol (get()+protect() every hit, THEN
    // admit() every miss). A miss's eviction must never drop a key touched earlier in
    // the SAME batch, else the pin can't resolve its slot ("expert not resident after
    // alloc"). The other tests drive keys one-at-a-time, so they never hit this.
    #[test]
    fn batch_never_evicts_a_key_touched_this_batch() {
        for name in ["lru", "2q", "arc"] {
            let (budget, cs, hs) = (60usize, 3usize, 4usize);
            let mut p = make(name, budget, cs, hs, TwoQSplit::default()).unwrap();
            let mut resident: HashMap<u32, ()> = HashMap::new();
            let mut rng = 0x1234_5678u64;
            let next = |r: &mut u64| {
                *r ^= *r << 13;
                *r ^= *r >> 7;
                *r ^= *r << 17;
                *r
            };
            for _ in 0..5000 {
                p.begin_batch();
                let rv: Vec<u32> = resident.keys().copied().collect();
                let mut batch: Vec<u32> = Vec::new();
                while batch.len() < 9 {
                    let k = if !rv.is_empty() && next(&mut rng) % 2 == 0 {
                        rv[next(&mut rng) as usize % rv.len()]
                    } else {
                        (next(&mut rng) % 200) as u32
                    };
                    if !batch.contains(&k) {
                        batch.push(k);
                    }
                }
                let mut is_hit = vec![false; batch.len()];
                for (i, &k) in batch.iter().enumerate() {
                    if p.get(k) {
                        p.protect(k);
                        is_hit[i] = true;
                    }
                }
                for (i, &k) in batch.iter().enumerate() {
                    if is_hit[i] {
                        continue;
                    }
                    let a = p.admit(k);
                    for e in &a.evicted {
                        assert!(
                            !batch.contains(e),
                            "{name}: evicted {e}, a key touched in the current batch"
                        );
                        resident.remove(e);
                    }
                    resident.insert(k, ());
                }
                for &k in &batch {
                    assert!(p.contains(k), "{name}: batch key {k} not resident after its batch");
                }
            }
        }
    }
}

#[cfg(test)]
mod hint_tests {
    use super::*;
    use crate::cache::TwoQSplit;

    const POLICIES: [&str; 3] = ["lru", "2q", "arc"];

    fn pol(name: &str, budget: usize) -> Box<dyn HybridPolicy> {
        make(name, budget, 1, 1, TwoQSplit::default()).unwrap()
    }

    /// **INV-2: a hint never promotes, admits, or otherwise leaves residue in policy
    /// state.** This is the property that separates the hint layer from the `top-m` it
    /// replaced: a hint may only delay an eviction, never change what the policy believes.
    ///
    /// Observing "no promotion" directly would mean reaching into each policy's private
    /// segments (2Q's A1in/Am, ARC's T1/T2/p), which the trait deliberately does not
    /// expose. So it is observed behaviourally and end-to-end: run one policy that saw a
    /// burst of hints (then let them expire) and one that never did, through an IDENTICAL
    /// access sequence, and require the eviction order to match exactly. Any promotion,
    /// admission, or `p` nudge would reorder something.
    #[test]
    fn inv_2_hints_leave_no_residue_in_policy_state() {
        for name in POLICIES {
            let (mut hinted, mut clean) = (pol(name, 8), pol(name, 8));
            // Hint keys that are NOT resident, which is the case most likely to tempt an
            // implementation into admitting them.
            let hints: Vec<Hint> =
                (100..108).map(|k| Hint { key: k, horizon: 2, rank: 0 }).collect();
            hinted.hint(&hints);
            for k in 100..108u32 {
                assert!(!hinted.contains(k), "{name}: a hint must NOT admit key {k}");
            }
            // Expire them, so from here the two policies must be indistinguishable.
            hinted.tick_hints();
            hinted.tick_hints();
            assert!(hinted.hinted().is_empty(), "{name}: hints must expire");

            let (mut ev_h, mut ev_c) = (Vec::new(), Vec::new());
            for k in 0..24u32 {
                for (p, ev) in [(&mut hinted, &mut ev_h), (&mut clean, &mut ev_c)] {
                    p.begin_batch();
                    if !p.get(k % 10) {
                        ev.extend(p.admit(k % 10).evicted);
                    }
                    if !p.get(k) {
                        ev.extend(p.admit(k).evicted);
                    }
                }
            }
            assert_eq!(ev_h, ev_c, "{name}: expired hints changed the eviction order");
            assert_eq!(hinted.resident_bytes(), clean.resident_bytes(), "{name}: bytes");
        }
    }

    /// **INV-3: a hint can never fail an allocation.** Hints are predictions; the pin's
    /// per-batch pins are correctness. If honouring every veto would leave no victim, the
    /// veto is dropped — otherwise an advisory signal turns into "expert not resident after
    /// alloc", and `HINT_CAP_PCT` silently becomes load-bearing for correctness.
    #[test]
    fn inv_3_hints_can_never_starve_eviction() {
        for name in POLICIES {
            let mut p = pol(name, 4);
            p.begin_batch();
            for k in 0..4u32 {
                p.admit(k);
            }
            // Veto EVERY resident key, far past the cap, then force an admission.
            let hints: Vec<Hint> =
                (0..4u32).map(|k| Hint { key: k, horizon: 8, rank: 0 }).collect();
            p.hint(&hints);
            p.begin_batch();
            let adm = p.admit(99);
            assert!(
                !adm.evicted.is_empty() || p.contains(99),
                "{name}: a full pool + all-vetoed must still admit"
            );
            assert!(p.contains(99), "{name}: the new key must be resident");
        }
    }

    /// The cap bounds the veto set, and rank decides who survives it — rank 0 is the
    /// router's top pick at 99% measured precision, rank 7 is close to a coin flip.
    #[test]
    fn hint_cap_keeps_the_best_ranks() {
        let mut hs = HintSet::with_cap(3);
        let hints: Vec<Hint> = (0..8u32)
            .map(|i| Hint { key: 100 + i, horizon: 2, rank: (7 - i) as u8 })
            .collect();
        hs.insert(&hints);
        assert_eq!(hs.len(), 3, "cap must bound the veto set");
        // rank 0..2 are keys 107, 106, 105.
        for k in [107u32, 106, 105] {
            assert!(hs.keys().contains(&k), "the best-ranked hint {k} must survive the cap");
        }
    }

    /// Vetoes decay. Without this a hint about a layer long past pins forever and eviction
    /// is permanently starved by stale predictions.
    #[test]
    fn hints_decay_after_their_horizon() {
        let mut hs = HintSet::with_cap(0);
        hs.insert(&[Hint { key: 1, horizon: 1, rank: 0 }, Hint { key: 2, horizon: 3, rank: 0 }]);
        hs.tick();
        assert!(!hs.keys().contains(&1), "a horizon-1 hint must not survive one layer");
        assert!(hs.keys().contains(&2), "a horizon-3 hint must still be live");
        hs.tick();
        hs.tick();
        assert!(hs.is_empty(), "every hint must eventually expire");
    }
}
