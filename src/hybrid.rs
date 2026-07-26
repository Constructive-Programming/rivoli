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

/// Construct a byte-aware hybrid policy. `split` is 2Q's Kin/Kout (ignored by lru/arc).
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
        match (self.cold.peek_lru_skip(&self.pinned), self.hot.peek_lru_skip(&self.pinned)) {
            (Some((tc, _)), Some((th, _))) => {
                if tc <= th {
                    self.cold.pop_lru_skip(&self.pinned)
                } else {
                    self.hot.pop_lru_skip(&self.pinned)
                }
            }
            (Some(_), None) => self.cold.pop_lru_skip(&self.pinned),
            (None, Some(_)) => self.hot.pop_lru_skip(&self.pinned),
            (None, None) => None,
        }
    }
}
impl HybridPolicy for HybridLru {
    fn contains(&self, k: u32) -> bool {
        self.cold.contains(k) || self.hot.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
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
        }
    }
    fn a1in_bytes(&self) -> usize {
        self.a1in.len() * self.g.cold_stride
    }
    fn trim_a1in(&mut self) -> Option<u32> {
        let v = self.a1in.pop_lru_skip(&self.pinned);
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
            self.am.pop_lru_skip(&self.pinned).or_else(|| self.trim_a1in())
        } else {
            self.trim_a1in().or_else(|| self.am.pop_lru_skip(&self.pinned))
        }
    }
}
impl HybridPolicy for HybridTwoQ {
    fn contains(&self, k: u32) -> bool {
        self.am.contains(k) || self.a1in.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        if self.am.contains(k) {
            self.am.touch(k);
            return true;
        }
        self.a1in.contains(k) // A1in hit stays put (FIFO); ghosts are not resident
    }
    fn admit(&mut self, k: u32) -> Admission {
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
            match self.t1.pop_lru_skip(&self.pinned) {
                Some(v) => (Some(v), true),
                None => (self.t2.pop_lru_skip(&self.pinned), false),
            }
        } else {
            match self.t2.pop_lru_skip(&self.pinned) {
                Some(v) => (Some(v), false),
                None => (self.t1.pop_lru_skip(&self.pinned), true),
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
            .map(|&n| (n, make(n, budget, cs, hs, TwoQSplit::default()).expect("known policy")))
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
