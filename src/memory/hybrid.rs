//! Byte-aware routed-expert policies for the format hybrid. Unlike the single-format
//! [`cache`](crate::memory::cache) policies (which count uniform slots), these manage TWO tiers
//! of DIFFERENT byte sizes — COLD (int3-VQ) and HOT (int4) — against a byte budget, so
//! the split between them FLOATS with the workload (and, for ARC, with the adaptive
//! `p`). The pin backs them with the two-ended [`arena`](crate::memory::arena); this module is
//! pure bookkeeping (host-testable).
//!
//! Contract: `hit` reports and RECORDS an access, keeping the key in its tier (a resident
//! expert never migrates format without a refetch). `admit` handles a MISS — it places the key in a
//! tier and evicts (by the policy's own rule) until the incoming slot's bytes fit the
//! budget, returning every victim. The pin frees each victim's slot, then places the new
//! one (compacting the arena as needed).

use crate::memory::cache::{OrderedSet, Tier};
use std::collections::{HashMap, HashSet};

/// The result of admitting a miss: which tier the key landed in, and every resident key
/// evicted to make its bytes fit (possibly several, since the tiers differ in size).
pub struct Admission {
    pub tier: Tier,
    pub evicted: Vec<u32>,
}

pub trait HybridPolicy {
    /// The shared geometry and per-batch pin set. Every policy owns one, and treats it
    /// identically, which is why the two batch methods below are defaults over it rather
    /// than the same four lines written out once per policy.
    fn geom(&mut self) -> &mut TierGeomAndBudget;
    fn contains(&self, k: u32) -> bool;
    /// Record an access and report whether it landed. A hit refreshes recency IN its tier
    /// and returns true; a miss changes nothing and returns false — `admit` is its other half.
    fn hit(&mut self, k: u32) -> bool;
    /// Admit a known-miss key: pick its tier, evict until it fits the byte budget.
    fn admit(&mut self, k: u32) -> Admission;
    /// Bytes currently resident — the pin/tests check this never exceeds the budget.
    fn resident_bytes(&self) -> usize;

    /// Start a new batch (`RoutedPool::submit`): clears the per-batch pin set so the
    /// next batch's evictions may reclaim the previous batch's keys again.
    fn begin_batch(&mut self) {
        self.geom().pinned.clear();
    }
    /// Pin a just-hit key for the rest of this batch — eviction must not take it, else
    /// the pin can't resolve its slot ("expert not resident after alloc"). Touching to
    /// MRU is NOT enough: `hit` already promotes the access to its tier's MRU end, and a
    /// big/skewed batch can still drain a whole tier past that end.
    fn protect(&mut self, k: u32) {
        self.geom().pinned.insert(k);
    }
}

/// Byte geometry + the per-batch pin set: the state every policy carries identically.
///
/// `pub` only because [`HybridPolicy::geom`] is — every field is private and every method
/// is module-local, so nothing outside this file can do anything with one. It exists so
/// `begin_batch`/`protect` are written once instead of once per policy.
pub struct TierGeomAndBudget {
    budget: usize,
    cold_stride: usize,
    hot_stride: usize,
    /// Keys touched (hit or admitted) in the CURRENT batch. Eviction skips these — see
    /// [`HybridPolicy::protect`] and `OrderedSet::pop_lru_skip`.
    pinned: HashSet<u32>,
}
impl TierGeomAndBudget {
    fn stride(&self, tier: Tier) -> usize {
        match tier {
            Tier::Cold => self.cold_stride,
            Tier::Hot => self.hot_stride,
        }
    }
    /// A budget's worth of the SMALLER slot — the key count 2Q's and ARC's ghosts bound to.
    fn slots(&self) -> usize {
        self.budget / self.cold_stride.min(self.hot_stride)
    }
    /// Close an admission: the new key is pinned for the rest of the batch, because the
    /// pin allocates every miss slot BEFORE resolving final addresses and a key evicted
    /// by a LATER miss in the same batch could not then be resolved.
    fn admitted(&mut self, k: u32, tier: Tier, evicted: Vec<u32>) -> Admission {
        self.pinned.insert(k);
        Admission { tier, evicted }
    }
}

/// Emit [`HybridPolicy::geom`] for a policy that carries its [`TierGeomAndBudget`] as field `b`.
///
/// Three tokens of glue that cannot be a trait default — a default body cannot name a
/// field — so all three policies wrote it out, and `cargo fmt` made the three copies
/// literal (jscpd caught them fused to the head of `contains`, which follows in each
/// impl). This macro is the single place the "field `b`" convention is stated; the
/// alternative, storing `TierGeomAndBudget` outside the policy, would reshape the trait and every
/// call site in `pin.rs` for no behaviour change.
macro_rules! impl_geom {
    () => {
        fn geom(&mut self) -> &mut TierGeomAndBudget {
            &mut self.b
        }
    };
}

/// `Hot` iff the policy's promotion test passed, `Cold` otherwise.
///
/// Every `admit` opens with this branch — LRU on its frequency counter, 2Q on an A1out
/// ghost hit — and the two were the same five lines once rustfmt expanded the `if/else`
/// (it exceeds `single_line_if_else_max_width`). Naming the branch says what it decides.
fn tier_if_hot(promote: bool) -> Tier {
    if promote { Tier::Hot } else { Tier::Cold }
}

/// Close an admission: put `k` into `tier`'s resident set and record the victims.
///
/// The tier→set dispatch plus [`TierGeomAndBudget::admitted`] is the tail of both the LRU and the 2Q
/// `admit`; only the two sets differ, in the same way [`touch_either`]'s two do. ARC does
/// not use it — its tier is decided inside the ghost branches, which touch their set there.
fn place_in_tier(
    b: &mut TierGeomAndBudget,
    cold: &mut OrderedSet,
    hot: &mut OrderedSet,
    k: u32,
    tier: Tier,
    evicted: Vec<u32>,
) -> Admission {
    match tier {
        Tier::Cold => cold.touch(k),
        Tier::Hot => hot.touch(k),
    }
    b.admitted(k, tier, evicted)
}

/// The eviction half of every `admit`: shed victims by the policy's own `reclaim` rule
/// until `incoming` bytes fit `budget`, or until nothing unpinned is left to shed.
///
/// One copy of the LOOP, three victim rules. The `None => break` is the load-bearing part:
/// a batch whose keys are all pinned has no legal victim, and a copy that dropped it would
/// spin rather than admit over budget.
fn evict_until_fits<P: HybridPolicy>(
    p: &mut P,
    incoming: usize,
    budget: usize,
    reclaim: impl Fn(&mut P) -> Option<u32>,
) -> Vec<u32> {
    let mut evicted = Vec::new();
    while p.resident_bytes() + incoming > budget {
        match reclaim(p) {
            Some(v) => evicted.push(v),
            None => break,
        }
    }
    evicted
}

/// A hit refreshes recency IN whichever of the two resident tiers holds `k` — a resident
/// expert never migrates format without a refetch. Shared by LRU and ARC, whose tier pairs
/// differ only in name.
fn touch_either(a: &mut OrderedSet, b: &mut OrderedSet, k: u32) -> bool {
    for s in [a, b] {
        if s.contains(k) {
            s.touch(k);
            return true;
        }
    }
    false
}

/// Construct a byte-aware hybrid policy. `split` is 2Q's Kin/Kout (ignored by lru/arc);
pub fn policy_for(
    policy: &str,
    budget: usize,
    cold_stride: usize,
    hot_stride: usize,
    split: crate::memory::cache::TwoQSplit,
) -> Option<Box<dyn HybridPolicy>> {
    // Strides clamped to ≥1 exactly as `Arena::new` clamps them: `slots` divides by the
    // smaller of the two, and the policy's byte accounting has to agree with the arena's.
    let g = TierGeomAndBudget {
        budget,
        cold_stride: cold_stride.max(1),
        hot_stride: hot_stride.max(1),
        pinned: HashSet::new(),
    };
    match policy {
        "lru" => Some(Box::new(HybridLru::new(g))),
        // Three policies, three implementations. (`top-m` was a fourth NAME over the same
        // LRU with router substitution switched on; it was removed 2026-07-30 because
        // steering EVICTION rather than SELECTION is output-neutral, which top-m was
        // not: +3.63% ppl on int3-vq, +12.7% on int4. See docs/investigations/cache-conditional-routing.md.)
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
// signal). See docs/reference/modes.md.
// ---------------------------------------------------------------------------
const LRU_HOT_THRESHOLD: u32 = 2;
/// Halve `freq` every this many accesses so the count tracks RECENT frequency (a cooled
/// expert drops below threshold). ~7 tokens at ~600 routed accesses/token; independent
/// of budget so it scopes "recent" by workload time, not pool size.
const LRU_DECAY: u64 = 4096;

struct HybridLru {
    b: TierGeomAndBudget,
    cold: OrderedSet,
    hot: OrderedSet,
    freq: HashMap<u32, u32>,
    accesses: u64,
}
impl HybridLru {
    fn new(b: TierGeomAndBudget) -> Self {
        Self {
            b,
            cold: OrderedSet::default(),
            hot: OrderedSet::default(),
            freq: HashMap::new(),
            accesses: 0,
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
        let pin = &self.b.pinned;
        match (self.cold.peek_lru_skip(pin), self.hot.peek_lru_skip(pin)) {
            (Some((tc, _)), Some((th, _))) => {
                if tc <= th {
                    self.cold.pop_lru_skip(&self.b.pinned)
                } else {
                    self.hot.pop_lru_skip(&self.b.pinned)
                }
            }
            (Some(_), None) => self.cold.pop_lru_skip(&self.b.pinned),
            (None, Some(_)) => self.hot.pop_lru_skip(&self.b.pinned),
            (None, None) => None,
        }
    }
}
impl HybridPolicy for HybridLru {
    impl_geom!();
    fn contains(&self, k: u32) -> bool {
        self.cold.contains(k) || self.hot.contains(k)
    }
    fn hit(&mut self, k: u32) -> bool {
        self.bump(k);
        touch_either(&mut self.cold, &mut self.hot, k)
    }
    fn admit(&mut self, k: u32) -> Admission {
        let tier = tier_if_hot(self.freq.get(&k).copied().unwrap_or(0) >= LRU_HOT_THRESHOLD);
        let (incoming, budget) = (self.b.stride(tier), self.b.budget);
        let evicted = evict_until_fits(self, incoming, budget, HybridLru::evict_lru);
        place_in_tier(&mut self.b, &mut self.cold, &mut self.hot, k, tier, evicted)
    }

    fn resident_bytes(&self) -> usize {
        self.cold.len() * self.b.cold_stride + self.hot.len() * self.b.hot_stride
    }
}

// ---------------------------------------------------------------------------
// 2Q — A1in probation (COLD) bounded by Kin bytes; Am (HOT) absorbs the rest and
// floats; A1out ghost promotes a returning key to HOT on its next miss.
// ---------------------------------------------------------------------------
struct HybridTwoQ {
    b: TierGeomAndBudget,
    kin_bytes: usize, // A1in (cold) byte bound
    kout: usize,      // A1out ghost length bound
    a1in: OrderedSet,
    am: OrderedSet,
    a1out: OrderedSet,
}
impl HybridTwoQ {
    fn new(b: TierGeomAndBudget, split: crate::memory::cache::TwoQSplit) -> Self {
        let kin_bytes = (b.budget * split.kin_pct() as usize / 100).max(b.cold_stride);
        let kout = (b.slots() * split.kout_pct() as usize / 100).max(1);
        Self {
            b,
            kin_bytes,
            kout,
            a1in: OrderedSet::default(),
            am: OrderedSet::default(),
            a1out: OrderedSet::default(),
        }
    }
    fn a1in_bytes(&self) -> usize {
        self.a1in.len() * self.b.cold_stride
    }
    fn trim_a1in(&mut self) -> Option<u32> {
        let v = self.a1in.pop_lru_skip(&self.b.pinned);
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
            self.am
                .pop_lru_skip(&self.b.pinned)
                .or_else(|| self.trim_a1in())
        } else {
            self.trim_a1in()
                .or_else(|| self.am.pop_lru_skip(&self.b.pinned))
        }
    }
}
impl HybridPolicy for HybridTwoQ {
    impl_geom!();
    fn contains(&self, k: u32) -> bool {
        self.am.contains(k) || self.a1in.contains(k)
    }
    fn hit(&mut self, k: u32) -> bool {
        if self.am.contains(k) {
            self.am.touch(k);
            return true;
        }
        self.a1in.contains(k) // A1in hit stays put (FIFO); ghosts are not resident
    }
    fn admit(&mut self, k: u32) -> Admission {
        // A second distinct access (via the ghost) promotes to Am/HOT; else A1in/COLD.
        let tier = tier_if_hot(self.a1out.remove(k));
        let (incoming, budget) = (self.b.stride(tier), self.b.budget);
        let evicted = evict_until_fits(self, incoming, budget, HybridTwoQ::reclaim);
        place_in_tier(&mut self.b, &mut self.a1in, &mut self.am, k, tier, evicted)
    }
    fn protect(&mut self, k: u32) {
        self.b.pinned.insert(k);
        // A1in is a FIFO and `get` leaves hits in place, so also move an actively-used
        // key to the young end (the 2Q recency intent, beyond the batch pin).
        if self.a1in.contains(k) {
            self.a1in.touch(k);
        }
    }

    fn resident_bytes(&self) -> usize {
        self.a1in.len() * self.b.cold_stride + self.am.len() * self.b.hot_stride
    }
}

// ---------------------------------------------------------------------------
// ARC — adaptive split. T1 (recency, COLD) / T2 (frequency, HOT) resident; B1/B2
// key-only ghosts drive the target `p` (in BYTES), which chooses the eviction tier.
// ---------------------------------------------------------------------------
struct HybridArc {
    b: TierGeomAndBudget,
    p: usize, // target BYTES for T1 (cold); floats with the ghost hits
    t1: OrderedSet,
    t2: OrderedSet,
    b1: OrderedSet,
    b2: OrderedSet,
}
impl HybridArc {
    fn new(b: TierGeomAndBudget) -> Self {
        Self {
            b,
            p: 0,
            t1: OrderedSet::default(),
            t2: OrderedSet::default(),
            b1: OrderedSet::default(),
            b2: OrderedSet::default(),
        }
    }
    fn t1_bytes(&self) -> usize {
        self.t1.len() * self.b.cold_stride
    }
    /// Evict one UNPINNED resident to a ghost, choosing the tier by `p`: shed COLD (T1)
    /// while it exceeds the target `p`, else shed HOT (T2). `in_b2` biases toward T1 at
    /// the tie. Falls back to the other tier if the preferred one has no unpinned victim
    /// (empty OR all pinned this batch), so it never stalls or evicts a batch key.
    fn replace(&mut self, in_b2: bool) -> Option<u32> {
        let t1b = self.t1_bytes();
        let prefer_cold = t1b > self.p || (in_b2 && t1b == self.p);
        let (v, from_cold) = if prefer_cold {
            match self.t1.pop_lru_skip(&self.b.pinned) {
                Some(v) => (Some(v), true),
                None => (self.t2.pop_lru_skip(&self.b.pinned), false),
            }
        } else {
            match self.t2.pop_lru_skip(&self.b.pinned) {
                Some(v) => (Some(v), false),
                None => (self.t1.pop_lru_skip(&self.b.pinned), true),
            }
        };
        if let Some(v) = v {
            let ghost = if from_cold {
                &mut self.b1
            } else {
                &mut self.b2
            };
            ghost.touch(v);
            // Bound each ghost to a budget's worth of keys (cheap; remembers returns).
            let bound = self.b.slots();
            while ghost.len() > bound {
                ghost.pop_lru();
            }
        }
        v
    }
}
impl HybridPolicy for HybridArc {
    impl_geom!();
    fn contains(&self, k: u32) -> bool {
        self.t1.contains(k) || self.t2.contains(k)
    }
    fn hit(&mut self, k: u32) -> bool {
        // A hit STAYS in its tier (no slab migration); refresh recency in-place.
        touch_either(&mut self.t1, &mut self.t2, k)
    }
    fn admit(&mut self, k: u32) -> Admission {
        // Geometry read up front: `evict_until_fits` takes `&mut self`.
        let (cold, hot, budget) = (self.b.cold_stride, self.b.hot_stride, self.b.budget);
        // A ghost hit is a returning key → promote to T2/HOT and adapt `p`.
        let (tier, evicted) = if self.b1.remove(k) {
            let delta = (self.b2.len() * hot / self.b1.len().max(1)).max(cold);
            self.p = (self.p + delta).min(budget);
            let ev = evict_until_fits(self, hot, budget, |s| s.replace(false));
            self.t2.touch(k);
            (Tier::Hot, ev)
        } else if self.b2.remove(k) {
            let delta = (self.b1.len() * cold / self.b2.len().max(1)).max(hot);
            self.p = self.p.saturating_sub(delta);
            let ev = evict_until_fits(self, hot, budget, |s| s.replace(true));
            self.t2.touch(k);
            (Tier::Hot, ev)
        } else {
            // Fresh miss → T1/COLD probation.
            let ev = evict_until_fits(self, cold, budget, |s| s.replace(false));
            self.t1.touch(k);
            (Tier::Cold, ev)
        };
        self.b.admitted(k, tier, evicted)
    }

    fn resident_bytes(&self) -> usize {
        self.t1.len() * self.b.cold_stride + self.t2.len() * self.b.hot_stride
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::memory::cache::TwoQSplit;

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
            Spec {
                budget,
                cs,
                hs,
                at: HashMap::new(),
            }
        }
        fn bytes(&self) -> usize {
            self.at
                .values()
                .map(|t| if *t == Tier::Hot { self.hs } else { self.cs })
                .sum()
        }
        fn access(&mut self, p: &mut dyn HybridPolicy, k: u32) -> Option<Tier> {
            p.begin_batch(); // each access is its own 1-key batch in this harness
            if p.hit(k) {
                assert!(
                    self.at.contains_key(&k),
                    "hit on a key the tally thinks is gone: {k}"
                );
                return None;
            }
            let Admission { tier, evicted } = p.admit(k);
            for e in &evicted {
                assert!(self.at.remove(e).is_some(), "evicted {e} was not resident");
                assert!(!p.contains(*e), "evicted {e} still contained");
            }
            self.at.insert(k, tier);
            assert!(p.contains(k), "admitted {k} not resident");
            assert_eq!(self.bytes(), p.resident_bytes(), "byte accounting drift");
            assert!(
                self.bytes() <= self.budget,
                "over budget: {} > {}",
                self.bytes(),
                self.budget
            );
            Some(tier)
        }
    }

    /// One 1-key batch: the pin's protocol in miniature (begin, then hit-or-admit).
    fn touch(p: &mut dyn HybridPolicy, k: u32) {
        p.begin_batch();
        if !p.hit(k) {
            p.admit(k);
        }
    }

    fn each_policy() -> Vec<(&'static str, Box<dyn HybridPolicy>)> {
        let (budget, cs, hs) = (100usize, 3usize, 4usize);
        ["lru", "2q", "arc"]
            .iter()
            .map(|&n| {
                let p = policy_for(n, budget, cs, hs, TwoQSplit::default());
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
            let mut p = policy_for(name, 5, 3, 4, TwoQSplit::default()).unwrap();
            p.begin_batch();
            assert!(!p.hit(10));
            assert_eq!(
                p.admit(10).tier,
                Tier::Cold,
                "{name}: first-seen must be COLD"
            );
            p.begin_batch();
            assert!(!p.hit(20)); // evicts 10 (cold slot reused)
            let _ = p.admit(20);
            assert!(!p.contains(10), "{name}: 10 should have been evicted");
            p.begin_batch();
            assert!(!p.hit(10)); // 10 returns via the ghost
            assert_eq!(
                p.admit(10).tier,
                Tier::Hot,
                "{name}: a returning key must be HOT"
            );
        }
    }

    #[test]
    fn lru_admits_by_frequency() {
        // LRU has no ghost: placement is the decaying counter. First-seen COLD; a key
        // re-accessed after eviction crosses the threshold → HOT.
        let mut p = policy_for("lru", 5, 3, 4, TwoQSplit::default()).unwrap();
        p.begin_batch();
        assert!(!p.hit(10));
        assert_eq!(p.admit(10).tier, Tier::Cold);
        p.begin_batch();
        assert!(!p.hit(20));
        let _ = p.admit(20); // evicts 10
        p.begin_batch();
        assert!(!p.hit(10)); // second access → freq 2
        assert_eq!(p.admit(10).tier, Tier::Hot, "re-accessed key must be HOT");
    }

    #[test]
    fn arc_p_adapts_toward_frequency() {
        // A frequency-skewed workload (a hot core hit via the ghost) must drive ARC's
        // `p` DOWN from 0-start toward HOT... p rises on B1 hits (recency), falls on B2.
        // Here we just assert the hot core stays resident under churn (adaptivity works).
        let mut p = policy_for("arc", 60, 3, 4, TwoQSplit::default()).unwrap();
        let core: Vec<u32> = (0..5).collect();
        for round in 0..50u32 {
            for &k in &core {
                touch(&mut *p, k);
            }
            // churn distinct tail keys to pressure the cache
            for t in 0..8u32 {
                touch(&mut *p, 1000 + round * 8 + t);
            }
        }
        let core_resident = core.iter().filter(|&&k| p.contains(k)).count();
        assert!(
            core_resident >= 3,
            "frequent core evaporated: {core_resident}/5 resident"
        );
    }

    // Mirrors `RoutedPool::submit`'s BATCH protocol (hit()+protect() every hit, THEN
    // admit() every miss). A miss's eviction must never drop a key touched earlier in
    // the SAME batch, else the pin can't resolve its slot ("expert not resident after
    // alloc"). The other tests drive keys one-at-a-time, so they never hit this.
    #[test]
    fn batch_never_evicts_a_key_touched_this_batch() {
        for name in ["lru", "2q", "arc"] {
            let (budget, cs, hs) = (60usize, 3usize, 4usize);
            let mut p = policy_for(name, budget, cs, hs, TwoQSplit::default()).unwrap();
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
                    if p.hit(k) {
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
                    assert!(
                        p.contains(k),
                        "{name}: batch key {k} not resident after its batch"
                    );
                }
            }
        }
    }
}
