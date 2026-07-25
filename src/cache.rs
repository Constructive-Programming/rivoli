//! Routed-expert cache policies — LRU, 2Q, ARC — usable BOTH offline (key-only
//! hit-rate replay, `simulate`) and LIVE (the pin delegates eviction here while it
//! owns the slot data). Decode is hit-rate-bound and, once prefetch lands, the
//! misprediction stream is a scan the policy must resist — so the policy is a
//! runtime choice (`--cache-policy`).
//!
//! Live contract: the pin maps `key -> slot`; the policy owns residency + eviction
//! order. `insert`/`insert_cold` return the RESIDENT key they evicted (if any) so
//! the pin reuses that key's slot; `None` means spare capacity (the pin pops a free
//! slot). `insert_cold` parks a PREFETCHED key at the cold/probation end so an
//! unused prediction is evicted first and never pollutes the hot set. The pin's
//! `slot_of` keys stay in lockstep with `resident_len()` (a debug-asserted
//! invariant).
//!
//! Complexity gradient: LRU (recency) < 2Q (frequency, fixed split) < ARC
//! (frequency, adaptive split + ghosts). ARC fits well here: value ~18 MB, key 4
//! bytes, so ghost history is nearly free.

use std::collections::{BTreeMap, HashMap};

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
    /// The LRU (oldest) key and its tick, without removing — for cross-set recency
    /// comparison (the hybrid LRU picks the globally oldest across both tiers).
    pub(crate) fn peek_lru(&self) -> Option<(i64, u32)> {
        self.order.iter().next().map(|(&t, &k)| (t, k))
    }
}

/// A routed-expert cache policy. `get` reports a hit and promotes on hit;
/// `insert` admits a known-miss key and returns the RESIDENT key it evicted (for
/// the pin to reuse that slot), or `None` if spare capacity absorbed it.
/// `insert_cold` admits a PREFETCHED (predicted, maybe-wrong) key into the
/// policy's probation segment so it does NOT enter the scan-resistant/protected set
/// on a speculative guess — 2Q parks it in A1in (never promoting via the A1out
/// ghost), ARC in T1 (never adapting `p` or promoting via B1/B2). CRITICAL: a batch
/// of `insert_cold`s at capacity must COEXIST — each evicts an OLDER resident, never
/// a just-inserted batch sibling (a pure single-segment recency cache can't do both
/// coexist AND cold-first, so `Lru` uses the default = normal `insert`). `access_batch`
/// runs one layer's keys two-pass (hits first, then misses), mirroring
/// `submit_layer`. `seed` pre-fills the protected/frequency segment.
/// Which residency tier an [`insert`](Cache::insert) landed a key in — the ONLY thing
/// the pool needs to pick a key's slab (the format hybrid: `Hot` → int4, `Cold` →
/// int3-VQ). A policy-agnostic residency concept, NOT a 2Q segment: `Hot` is the
/// proven-frequent tier (2Q's Am), `Cold` the probation tier (2Q's A1in). Single-tier
/// policies (`Lru`) and `Arc` (for now) report `Cold` — a single-format fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Cold,
    Hot,
}

pub trait Cache {
    fn contains(&self, k: u32) -> bool;
    fn get(&mut self, k: u32) -> bool;
    /// Admit `k`; return `(evicted resident key, the tier k landed in)`.
    fn insert(&mut self, k: u32) -> (Option<u32>, Tier);
    /// Default = normal `insert` (correct for single-segment `Lru`: a batch coexists,
    /// evicting old normals). Segmented policies override to force probation (`Cold`).
    fn insert_cold(&mut self, k: u32) -> (Option<u32>, Tier) {
        self.insert(k)
    }
    /// Make `k` — just reported resident by [`Cache::get`] — safe from eviction by
    /// a subsequent `insert` in the SAME batch.
    ///
    /// The pin resolves a layer two-pass: every hit first, then every miss. That
    /// ordering is only protective if a `get` moves the key away from the eviction
    /// end, which is true for `Lru` (touch) and `Arc` (T1->T2 promote) but NOT for
    /// `TwoQ`: a hit on an A1in entry correctly leaves it in the FIFO, so a later
    /// `insert` can reclaim it — while the pin still holds its slot in `slots[i]`
    /// and hands that slot to a miss, whose read then overwrites weights a live
    /// descriptor points at. MEASURED: 2q at two pool sizes diverged at pos=8,
    /// layer 31, expert 110 — a HIT whose slot had been reassigned underneath it.
    ///
    /// Default is a no-op, correct for any policy whose `get` already promotes.
    fn protect(&mut self, _k: u32) {}
    fn seed(&mut self, keys: &[u32]);
    fn resident_len(&self) -> usize;

    fn access_batch(&mut self, keys: &[u32]) -> usize {
        // top_k is small (≤8 + shared); a fixed miss scratch avoids an alloc.
        let mut miss = [0u32; 32];
        let mut nm = 0;
        let mut hits = 0;
        for &k in keys {
            if self.get(k) {
                hits += 1;
            } else if nm < miss.len() {
                miss[nm] = k;
                nm += 1;
            }
        }
        for &k in &miss[..nm] {
            self.insert(k);
        }
        hits
    }
}

// ---------------------------------------------------------------------------
// LRU — pure recency (mirrors the pin's original hand-rolled Lru). In the format
// hybrid it gains a FREQUENCY-COUNTER admission layer (LRU has no ghost/frequency
// signal of its own): LRU still evicts (per segment), a decaying per-key count
// decides placement. See MODES.md.
// ---------------------------------------------------------------------------
/// Recent-frequency threshold: a miss whose key has been accessed ≥ this many times
/// within the current decay window is admitted to HOT/int4, else COLD/vq3.
const LRU_HOT_THRESHOLD: u32 = 2;

pub struct Lru {
    cap: usize,
    /// `None` = pure LRU (one segment). `Some((n_cold, n_hot))` = FIXED PARTITION for
    /// the hybrid: `set` = COLD/vq3 (cap n_cold), `hot` = HOT/int4 (cap n_hot), and
    /// `freq` gates admission. Each segment is an independent LRU, so an insert only
    /// evicts from the tier it enters → evicted_tier == insert_tier.
    fixed: Option<(usize, usize)>,
    set: OrderedSet, // dynamic: the pool; fixed: the COLD segment
    hot: OrderedSet, // fixed only: the HOT segment
    /// Fixed only: decayed per-key access count (recent frequency). Halved every `cap`
    /// accesses so a cooled expert falls below threshold instead of staying
    /// HOT-eligible forever (all-time counts would collapse the split to all-int4).
    freq: HashMap<u32, u32>,
    accesses: u64,
}
impl Lru {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            fixed: None,
            set: OrderedSet::default(),
            hot: OrderedSet::default(),
            freq: HashMap::new(),
            accesses: 0,
        }
    }

    /// Fixed-partition LRU for the format hybrid: `n_cold` = COLD/vq3 cap, `n_hot` =
    /// HOT/int4 cap. Placement is by recent frequency; eviction is per-segment LRU.
    pub fn fixed(n_cold: usize, n_hot: usize) -> Self {
        Self {
            cap: (n_cold + n_hot).max(1),
            fixed: Some((n_cold.max(1), n_hot.max(1))),
            set: OrderedSet::default(),
            hot: OrderedSet::default(),
            freq: HashMap::new(),
            accesses: 0,
        }
    }

    /// Count one access to `k` and periodically decay (halve) all counts so the signal
    /// tracks RECENT frequency. Fixed-partition only; called once per access (in `get`).
    fn bump(&mut self, k: u32) {
        *self.freq.entry(k).or_insert(0) += 1;
        self.accesses += 1;
        if self.accesses % self.cap as u64 == 0 {
            self.freq.retain(|_, v| {
                *v /= 2;
                *v > 0
            });
        }
    }
}
impl Cache for Lru {
    fn contains(&self, k: u32) -> bool {
        self.set.contains(k) || self.hot.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        if self.fixed.is_some() {
            self.bump(k); // one count per access; a hit stays in its segment (no migration)
            if self.set.contains(k) {
                self.set.touch(k);
                return true;
            }
            if self.hot.contains(k) {
                self.hot.touch(k);
                return true;
            }
            return false;
        }
        if self.set.contains(k) {
            self.set.touch(k);
            return true;
        }
        false
    }
    fn insert(&mut self, k: u32) -> (Option<u32>, Tier) {
        if let Some((n_cold, n_hot)) = self.fixed {
            // Miss (get already counted this access). Recent frequency picks the tier;
            // evict the chosen segment's LRU so evicted_tier == insert_tier.
            if self.freq.get(&k).copied().unwrap_or(0) >= LRU_HOT_THRESHOLD {
                let ev = (self.hot.len() >= n_hot).then(|| self.hot.pop_lru()).flatten();
                self.hot.touch(k);
                (ev, Tier::Hot)
            } else {
                let ev = (self.set.len() >= n_cold).then(|| self.set.pop_lru()).flatten();
                self.set.touch(k);
                (ev, Tier::Cold)
            }
        } else {
            let ev = (self.set.len() >= self.cap).then(|| self.set.pop_lru()).flatten();
            self.set.touch(k);
            (ev, Tier::Cold) // single segment → always the cold/vq3 tier
        }
    }
    fn protect(&mut self, k: u32) {
        // Mirror 2Q/ARC: keep an actively-used COLD hit off the block this batch. HOT
        // hits are already MRU-refreshed by `get`.
        if self.fixed.is_some() && self.set.contains(k) {
            self.set.touch(k);
        }
    }
    // insert_cold: uses the trait default (= insert). A recency cache cannot both
    // coexist a prefetch batch and cold-park it, so LRU prefetches land as normal
    // inserts (they decay if unused). Use 2q/arc to cold-park.
    fn seed(&mut self, keys: &[u32]) {
        // Seed the frequent segment when partitioned, else the single pool.
        let dst = if self.fixed.is_some() { &mut self.hot } else { &mut self.set };
        for &k in keys.iter().take(self.cap) {
            dst.touch(k);
        }
    }
    fn resident_len(&self) -> usize {
        self.set.len() + self.hot.len()
    }
}

// ---------------------------------------------------------------------------
// 2Q (Johnson & Shasha) — frequency with a FIXED split. A1in FIFO holds
// first-timers; a second distinct access (via the A1out ghost) promotes into the
// Am LRU, which scans can't pollute. Prefetches naturally park in A1in already.
// ---------------------------------------------------------------------------
/// 2Q's fixed split (the paper's Kin/Kout), expressed as PERCENTAGES of capacity so
/// one setting transfers across slot counts. `Kin` bounds the A1in probation FIFO
/// (resident); `Kout` bounds the A1out ghost (key-only, not resident).
///
/// [`Default`] is the pair that was hardcoded until now — 25 % / 50 %, i.e. `cap/4`
/// and `cap/2` — and reproduces it EXACTLY: `cap * 25 / 100 == cap / 4` and
/// `cap * 50 / 100 == cap / 2` for every `cap` (both are exact divisors of 100), so
/// leaving `--2q-kin`/`--2q-kout` unset is bit-identical to the 91.8 %-residency
/// baseline in benchmarks.md.
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
    fn default() -> Self {
        Self {
            kin_pct: 8,
            kout_pct: 20,
        }
    }
}

impl TwoQSplit {
    /// The classical 2Q split (the paper's `cap/4` probation, `cap/2` ghost) — the
    /// scan-resistant reference the tuned [`Default`] deliberately trades away for
    /// residency. Kept so that trade-off stays A/B-testable.
    pub const CLASSICAL: Self = Self {
        kin_pct: 25,
        kout_pct: 50,
    };
}

/// Why a `(kin_pct, kout_pct)` pair was rejected. Both carry the offending value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitError {
    /// A1in outside `1..=99` % of capacity.
    KinRange(u32),
    /// A1out outside `1..=1000` % of capacity.
    KoutRange(u32),
}

impl std::fmt::Display for SplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KinRange(v) => write!(f, "--2q-kin {v}: must be 1..=99 (percent of capacity)"),
            Self::KoutRange(v) => {
                write!(f, "--2q-kout {v}: must be 1..=1000 (percent of capacity)")
            }
        }
    }
}
impl std::error::Error for SplitError {}

impl TwoQSplit {
    /// Validate a percentage pair. `kin_pct` is capped at 99 because at `kin >= cap`
    /// `reclaim` can never trim A1in, so with an empty Am it evicts nothing and
    /// residency grows past capacity — which would break the pin's slot invariant.
    /// `kout_pct` may exceed 100: A1out holds keys only (4 bytes), so a ghost larger
    /// than the resident set is cheap and remembers more second-access candidates.
    pub fn new(kin_pct: u32, kout_pct: u32) -> Result<Self, SplitError> {
        if !(1..=99).contains(&kin_pct) {
            return Err(SplitError::KinRange(kin_pct));
        }
        if !(1..=1000).contains(&kout_pct) {
            return Err(SplitError::KoutRange(kout_pct));
        }
        Ok(Self { kin_pct, kout_pct })
    }

    pub fn kin_pct(self) -> u32 {
        self.kin_pct
    }
    pub fn kout_pct(self) -> u32 {
        self.kout_pct
    }

    /// Absolute A1in bound at `cap` slots, clamped to `1..=cap-1` so `reclaim` always
    /// makes progress (see [`TwoQSplit::new`]). At the default 25 % this is `cap/4`.
    fn kin(self, cap: usize) -> usize {
        let ceiling = cap.saturating_sub(1).max(1);
        (cap * self.kin_pct as usize / 100).clamp(1, ceiling)
    }

    /// Absolute A1out bound at `cap` slots. At the default 50 % this is `cap/2`.
    fn kout(self, cap: usize) -> usize {
        (cap * self.kout_pct as usize / 100).max(1)
    }
}

pub struct TwoQ {
    cap: usize,
    kin: usize,
    kout: usize,
    /// `None` = paper-dynamic (A1in ≤ kin, Am absorbs the rest, reclaim on total ≥
    /// cap). `Some(n_hot)` = FIXED PARTITION (the format hybrid): A1in hard-capped at
    /// `kin`, Am hard-capped at `n_hot`, each mapped to its own slab. Trades adaptivity
    /// for two right-sized slabs; an insert into a segment only evicts from THAT
    /// segment, so evicted_tier == insert_tier.
    am_cap: Option<usize>,
    a1in: OrderedSet,
    a1out: OrderedSet,
    am: OrderedSet,
}
impl TwoQ {
    pub fn new(cap: usize, split: TwoQSplit) -> Self {
        Self {
            cap,
            kin: split.kin(cap),
            kout: split.kout(cap),
            am_cap: None,
            a1in: OrderedSet::default(),
            a1out: OrderedSet::default(),
            am: OrderedSet::default(),
        }
    }

    /// Fixed-partition 2Q for the format hybrid: A1in (cold/vq3 slab) capped at
    /// `a1in_cap`, Am (hot/int4 slab) capped at `am_cap`, ghost = `kout`.
    pub fn fixed(a1in_cap: usize, am_cap: usize, kout: usize) -> Self {
        Self {
            cap: a1in_cap + am_cap,
            kin: a1in_cap.max(1),
            kout: kout.max(1),
            am_cap: Some(am_cap.max(1)),
            a1in: OrderedSet::default(),
            a1out: OrderedSet::default(),
            am: OrderedSet::default(),
        }
    }

    /// Evict A1in's LRU into the ghost (trimmed to `kout`) — frees one cold slot.
    fn evict_a1in(&mut self) -> Option<u32> {
        let v = self.a1in.pop_lru();
        if let Some(v) = v {
            self.a1out.touch(v);
            while self.a1out.len() > self.kout {
                self.a1out.pop_lru();
            }
        }
        v
    }
    /// Free one RESIDENT page for a new admission (paper's "reclaimfor"): trim an
    /// oversized A1in into the ghost, else evict Am's LRU. Returns the resident key
    /// that left residency (moved to ghost, or dropped from Am).
    fn reclaim(&mut self) -> Option<u32> {
        if self.a1in.len() + self.am.len() < self.cap {
            return None;
        }
        if self.a1in.len() > self.kin {
            self.evict_a1in()
        } else {
            self.am.pop_lru()
        }
    }
}
impl Cache for TwoQ {
    fn contains(&self, k: u32) -> bool {
        self.am.contains(k) || self.a1in.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        if self.am.contains(k) {
            self.am.touch(k);
            return true;
        }
        self.a1in.contains(k) // hit but stays put (FIFO); ghosts are not resident
    }
    fn insert(&mut self, k: u32) -> (Option<u32>, Tier) {
        // Promotion (a second distinct access via the ghost) → Am/Hot; else → A1in/Cold.
        let promote = self.a1out.remove(k);
        if let Some(am_cap) = self.am_cap {
            // FIXED PARTITION: evict from the SAME segment the key enters, so the freed
            // slot is in the destination slab (evicted_tier == insert_tier).
            if promote {
                let ev = (self.am.len() >= am_cap).then(|| self.am.pop_lru()).flatten();
                self.am.touch(k);
                (ev, Tier::Hot)
            } else {
                let ev = (self.a1in.len() >= self.kin).then(|| self.evict_a1in()).flatten();
                self.a1in.touch(k);
                (ev, Tier::Cold)
            }
        } else {
            // DYNAMIC (paper): reclaim on total, Am absorbs the slack.
            let ev = self.reclaim();
            if promote {
                self.am.touch(k);
                (ev, Tier::Hot)
            } else {
                self.a1in.touch(k);
                (ev, Tier::Cold)
            }
        }
    }
    /// A1in is a FIFO and `get` deliberately leaves hits in place, so the pin's
    /// hits-before-misses ordering does not protect them on its own. Moving an
    /// accessed entry to the young end of A1in makes it unreclaimable for the rest
    /// of the batch. This is a deliberate, minimal deviation from strict 2Q, applied
    /// ONLY to keys the current layer is actively using.
    fn protect(&mut self, k: u32) {
        if self.a1in.contains(k) {
            self.a1in.touch(k);
        }
    }
    fn insert_cold(&mut self, k: u32) -> (Option<u32>, Tier) {
        // Prefetch: force into A1in probation, NEVER promoting via the A1out ghost —
        // a speculative (maybe-wrong) prediction must not enter the protected Am set.
        // A batch coexists: reclaim evicts the oldest A1in (or Am, dynamic), never a
        // fresh sibling.
        self.a1out.remove(k); // drop any stale ghost; do NOT promote
        let ev = if self.am_cap.is_some() {
            (self.a1in.len() >= self.kin).then(|| self.evict_a1in()).flatten()
        } else {
            self.reclaim()
        };
        self.a1in.touch(k);
        (ev, Tier::Cold)
    }
    fn seed(&mut self, keys: &[u32]) {
        for &k in keys.iter().take(self.cap) {
            self.am.touch(k);
        }
    }
    fn resident_len(&self) -> usize {
        self.a1in.len() + self.am.len()
    }
}

// ---------------------------------------------------------------------------
// ARC (Megiddo & Modha) — frequency with an ADAPTIVE split. T1 (recency) / T2
// (frequency) resident; B1 / B2 key-only ghosts drive the target `p`.
// ---------------------------------------------------------------------------
pub struct Arc {
    c: usize,
    p: usize,
    /// `None` = dynamic ARC (adaptive `p`, promote-on-hit). `Some((n_cold, n_hot))` =
    /// FIXED PARTITION for the format hybrid: T1 (recency) hard-capped at `n_cold`
    /// (COLD/vq3 slab), T2 (frequency) at `n_hot` (HOT/int4). Like [`TwoQ::fixed`]: a
    /// hit stays in its tier (a resident expert can't migrate slabs without a refetch),
    /// promotion T1→T2 rides the B1/B2 ghost on a MISS, and an insert only evicts from
    /// its own tier → evicted_tier == insert_tier. Drops ARC's adaptivity for two
    /// right-sized slabs. The two ghosts are ARC's flavour vs 2Q's single A1out.
    fixed: Option<(usize, usize)>,
    /// Fixed-partition ghost bound (each of B1/B2 trimmed to this) — 0 in dynamic ARC,
    /// where the ghosts are bounded implicitly by the paper's invariants.
    kout: usize,
    t1: OrderedSet,
    t2: OrderedSet,
    b1: OrderedSet,
    b2: OrderedSet,
}
impl Arc {
    pub fn new(cap: usize) -> Self {
        Self {
            c: cap.max(1),
            p: 0,
            fixed: None,
            kout: 0,
            t1: OrderedSet::default(),
            t2: OrderedSet::default(),
            b1: OrderedSet::default(),
            b2: OrderedSet::default(),
        }
    }

    /// Fixed-partition ARC for the format hybrid (see the `fixed` field). `n_cold` =
    /// T1/COLD/vq3 cap, `n_hot` = T2/HOT/int4 cap, `kout` = each ghost's bound (how far
    /// back a returning expert is still remembered → promoted to HOT).
    pub fn fixed(n_cold: usize, n_hot: usize, kout: usize) -> Self {
        Self {
            c: (n_cold + n_hot).max(1),
            p: 0,
            fixed: Some((n_cold.max(1), n_hot.max(1))),
            kout: kout.max(1),
            t1: OrderedSet::default(),
            t2: OrderedSet::default(),
            b1: OrderedSet::default(),
            b2: OrderedSet::default(),
        }
    }

    /// Fixed-partition eviction: pop the tier's LRU into its ghost, trimmed to `kout`.
    /// `hot` picks T2/B2 (else T1/B1). Frees one slot in that tier's slab.
    fn evict_tier(&mut self, hot: bool) -> Option<u32> {
        let kout = self.kout;
        let (resident, ghost) = if hot {
            (&mut self.t2, &mut self.b2)
        } else {
            (&mut self.t1, &mut self.b1)
        };
        let v = resident.pop_lru();
        if let Some(v) = v {
            ghost.touch(v);
            while ghost.len() > kout {
                ghost.pop_lru();
            }
        }
        v
    }
    /// Admit `k` into T1 recency on a fresh miss (no ghost hit): make room per the
    /// paper's Case IV cases, evict one resident to a ghost if at capacity, then
    /// touch `k` into T1. Returns the evicted resident key (if any). Shared by
    /// `insert`'s cold-miss branch and `insert_cold` (which first strips any ghost).
    fn admit_t1(&mut self, k: u32) -> Option<u32> {
        let c = self.c;
        let total = self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len();
        let evicted = if self.t1.len() + self.b1.len() == c {
            if self.t1.len() < c {
                self.b1.pop_lru();
                self.replace(false)
            } else {
                self.t1.pop_lru() // resident T1 dropped
            }
        } else if total >= c {
            if total == 2 * c {
                self.b2.pop_lru();
            }
            self.replace(false)
        } else {
            None // spare capacity
        };
        self.t1.touch(k);
        evicted
    }

    /// Evict one RESIDENT page to a ghost (paper's REPLACE); returns the evicted
    /// resident key. `in_b2` biases toward evicting T1 at the boundary.
    fn replace(&mut self, in_b2: bool) -> Option<u32> {
        if self.t1.len() >= 1 && (self.t1.len() > self.p || (in_b2 && self.t1.len() == self.p)) {
            let v = self.t1.pop_lru();
            if let Some(v) = v {
                self.b1.touch(v);
            }
            v
        } else {
            let v = self.t2.pop_lru();
            if let Some(v) = v {
                self.b2.touch(v);
            }
            v
        }
    }
}
impl Cache for Arc {
    fn contains(&self, k: u32) -> bool {
        self.t1.contains(k) || self.t2.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        if self.fixed.is_some() {
            // Fixed partition: a hit STAYS in its tier (it can't migrate slabs) — just
            // refresh recency in-tier. Promotion happens only on a miss (via a ghost).
            if self.t1.contains(k) {
                self.t1.touch(k);
                return true;
            }
            if self.t2.contains(k) {
                self.t2.touch(k);
                return true;
            }
            return false;
        }
        if self.t1.remove(k) || self.t2.contains(k) {
            self.t2.touch(k);
            return true;
        }
        false
    }
    fn insert(&mut self, k: u32) -> (Option<u32>, Tier) {
        if let Some((n_cold, n_hot)) = self.fixed {
            // FIXED PARTITION: a ghost hit (seen before, in B1 or B2) → T2/HOT; a fresh
            // miss → T1/COLD. Evict from the SAME tier so evicted_tier == insert_tier.
            let promote = self.b1.remove(k) | self.b2.remove(k);
            if promote {
                let ev = (self.t2.len() >= n_hot).then(|| self.evict_tier(true)).flatten();
                self.t2.touch(k);
                (ev, Tier::Hot)
            } else {
                let ev = (self.t1.len() >= n_cold).then(|| self.evict_tier(false)).flatten();
                self.t1.touch(k);
                (ev, Tier::Cold)
            }
        } else {
            let c = self.c;
            let evicted;
            if self.b1.contains(k) {
                let delta = (self.b2.len() / self.b1.len().max(1)).max(1);
                self.p = (self.p + delta).min(c);
                evicted = self.replace(false);
                self.b1.remove(k);
                self.t2.touch(k);
            } else if self.b2.contains(k) {
                let delta = (self.b1.len() / self.b2.len().max(1)).max(1);
                self.p = self.p.saturating_sub(delta);
                evicted = self.replace(true);
                self.b2.remove(k);
                self.t2.touch(k);
            } else {
                evicted = self.admit_t1(k);
            }
            (evicted, Tier::Cold)
        }
    }
    fn protect(&mut self, k: u32) {
        // Mirror TwoQ: keep an actively-used COLD hit off the eviction block for the
        // rest of the batch. T2 hits are already MRU-refreshed by `get`.
        if self.fixed.is_some() && self.t1.contains(k) {
            self.t1.touch(k);
        }
    }
    fn insert_cold(&mut self, k: u32) -> (Option<u32>, Tier) {
        // Prefetch: force into T1 recency probation, NEVER adapting `p` or promoting
        // via the B1/B2 ghosts — a speculative prediction must not corrupt ARC's
        // adaptation signal. Strip any stale ghost, then admit to T1/COLD.
        self.b1.remove(k);
        self.b2.remove(k);
        if let Some((n_cold, _)) = self.fixed {
            let ev = (self.t1.len() >= n_cold).then(|| self.evict_tier(false)).flatten();
            self.t1.touch(k);
            (ev, Tier::Cold)
        } else {
            (self.admit_t1(k), Tier::Cold)
        }
    }
    fn seed(&mut self, keys: &[u32]) {
        for &k in keys.iter().take(self.c) {
            self.t2.touch(k);
        }
    }
    fn resident_len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }
}

/// Construct a policy by name (`lru`|`2q`|`arc`) at `cap` slots.
pub fn make(policy: &str, cap: usize, split: TwoQSplit) -> Option<Box<dyn Cache>> {
    match policy {
        "lru" => Some(Box::new(Lru::new(cap))),
        "2q" => Some(Box::new(TwoQ::new(cap, split))),
        "arc" => Some(Box::new(Arc::new(cap))),
        _ => None,
    }
}

/// One MoE layer of a captured trace: the experts the layer actually routed to, and
/// the experts the PREVIOUS layer's cross-layer predictor prefetched for it. An old
/// trace (or a `--trace` run with prefetch off) has `predicted` empty — which is a
/// materially different cache workload; see [`replay`].
#[derive(Clone, Copy, Debug)]
pub struct Layer<'a> {
    pub demand: &'a [u32],
    pub predicted: &'a [u32],
}

/// Per-access outcome of a replay, mirroring the engine's `expert source` split.
/// Only `loaded` is I/O-free — see the warning at the top of benchmarks.md. `hits`
/// is `loaded + preloading`, i.e. the number that "needed no demand read"; do not
/// quote it as residency.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency {
    /// Resident, and its read had already completed: the real residency metric.
    pub loaded: u64,
    /// Resident only because the immediately preceding layer's prefetch admitted it —
    /// the read is still in flight at resolve time.
    pub preloading: u64,
    /// Not resident: a demand read.
    pub cold: u64,
}

impl Residency {
    pub fn accesses(self) -> u64 {
        self.loaded + self.preloading + self.cold
    }
    /// `loaded` as a percentage of accesses; 0.0 for an empty trace.
    pub fn loaded_pct(self) -> f64 {
        let n = self.accesses();
        if n == 0 {
            0.0
        } else {
            100.0 * self.loaded as f64 / n as f64
        }
    }
    /// `(loaded + preloading)` as a percentage — the engine's `hit %`.
    pub fn hit_pct(self) -> f64 {
        let n = self.accesses();
        if n == 0 {
            0.0
        } else {
            100.0 * (self.loaded + self.preloading) as f64 / n as f64
        }
    }
}

/// Replay a captured trace through a fresh `policy` at `cap` slots, optionally
/// seeding the protected segment first.
///
/// This mirrors `Pin::resolve_layer` + `Pin::prefetch_layer` ordering exactly: the
/// predictions for layer L are cold-admitted (skipping any already-resident key)
/// BEFORE L resolves, because the engine issues them during L-1; then L's demands
/// are resolved hits-first, misses-second. A demand that hits a key admitted by the
/// immediately preceding prefetch counts as `preloading`, matching the engine's
/// accounting of a read still in flight.
///
/// Returns `None` if `policy` is not one of `lru`/`2q`/`arc`.
pub fn replay(
    policy: &str,
    cap: usize,
    split: TwoQSplit,
    seed: &[u32],
    trace: &[Layer<'_>],
) -> Option<Residency> {
    let mut cache = make(policy, cap, split)?;
    if !seed.is_empty() {
        cache.seed(seed);
    }
    let mut r = Residency::default();
    // Keys whose prefetch was issued for THIS layer, so a hit on one is still in
    // flight. Reused across layers to avoid a per-layer allocation.
    let mut in_flight: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut miss = [0u32; 32];
    for layer in trace {
        in_flight.clear();
        for &k in layer.predicted {
            if !cache.contains(k) {
                cache.insert_cold(k);
                in_flight.insert(k);
            }
        }
        let mut nm = 0;
        for &k in layer.demand {
            if cache.get(k) {
                if in_flight.contains(&k) {
                    r.preloading += 1;
                } else {
                    r.loaded += 1;
                }
            } else {
                r.cold += 1;
                if nm < miss.len() {
                    miss[nm] = k;
                    nm += 1;
                }
            }
        }
        for &k in &miss[..nm] {
            cache.insert(k);
        }
    }
    Some(r)
}

/// Replay demand-only `batches` (no prefetch modelled). Returns `(hits, accesses)`.
/// Thin wrapper over [`replay`]; prefer `replay` when the trace carries predictions,
/// because prefetch admission is what makes 2Q's probation split pay off at all.
/// Test-only: the live engine and `bin/replay` both call `replay` directly.
#[cfg(test)]
fn simulate(
    policy: &str,
    cap: usize,
    split: TwoQSplit,
    seed: &[u32],
    batches: &[Vec<u32>],
) -> (u64, u64) {
    let trace: Vec<Layer<'_>> = batches
        .iter()
        .map(|b| Layer {
            demand: b,
            predicted: &[],
        })
        .collect();
    let r = replay(policy, cap, split, seed, &trace)
        .unwrap_or_else(|| panic!("unknown policy {policy:?}"));
    (r.loaded + r.preloading, r.accesses())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Box a policy by name without `unwrap` (denied even in tests).
    fn boxed(p: &str, cap: usize) -> Box<dyn Cache> {
        match p {
            "2q" => Box::new(TwoQ::new(cap, TwoQSplit::default())),
            "arc" => Box::new(Arc::new(cap)),
            _ => Box::new(Lru::new(cap)),
        }
    }

    /// A key established as frequent must survive a later UNINTERRUPTED cold scan
    /// longer than the cache. Frequency policies promote it to a protected segment;
    /// pure recency ages it out. `hot` is promoted with just enough churn between
    /// touches to age it into the ghost and back, then left untouched through the scan.
    fn survives_cold_scan(mut c: Box<dyn Cache>, cap: usize) -> bool {
        let hot = 999_999u32;
        let mut churn = 1_000_000u32;
        let burst = cap / 4 + 1;
        for _ in 0..6 {
            c.access_batch(&[hot]);
            for _ in 0..burst {
                c.access_batch(&[churn]);
                churn += 1;
            }
        }
        c.access_batch(&[hot]);
        for _ in 0..(cap * 5) {
            c.access_batch(&[churn]);
            churn += 1;
        }
        c.contains(hot)
    }

    #[test]
    fn lru_has_no_scan_resistance() {
        assert!(
            !survives_cold_scan(boxed("lru", 8), 8),
            "recency LRU should drop the hot key under a cold scan"
        );
    }

    /// Scan resistance is the ALGORITHMIC property 2Q and ARC exist for, and it is
    /// asserted against 2Q's classical split. It depends on the A1out ghost being big
    /// enough to still remember the hot key when it is re-referenced after the scan.
    #[test]
    fn twoq_and_arc_survive_scan() {
        let twoq: Box<dyn Cache> = Box::new(TwoQ::new(8, TwoQSplit::CLASSICAL));
        assert!(survives_cold_scan(twoq, 8), "2Q must protect hot");
        assert!(
            survives_cold_scan(boxed("arc", 8), 8),
            "ARC must protect hot"
        );
    }

    /// The SHIPPED default deliberately trades that property away, and this records
    /// the trade rather than letting it disappear silently.
    ///
    /// `kout` 20 % beat 100 % by ~6 pp of residency on the measured workload (see
    /// `TwoQSplit::default`), because at 13k unique experts over ~3.6k slots most
    /// re-references of long-evicted keys are spurious and promoting them pollutes
    /// the protected set. The cost is that a small ghost cannot remember a hot key
    /// across a long scan — at `cap=8` the ghost is a single entry and scan
    /// resistance is gone entirely.
    ///
    /// OVERFITTING RISK, stated plainly: that +6 pp was tuned on ONE prompt's trace.
    /// A multi-request server workload — the eventual target — has genuine
    /// cross-request reuse that a large ghost is designed to capture, and this
    /// default may well be wrong there. Re-tune against a multi-prompt trace before
    /// trusting it outside the single-prompt bench.
    #[test]
    fn shipped_default_trades_scan_resistance_for_residency() {
        let tuned: Box<dyn Cache> = Box::new(TwoQ::new(8, TwoQSplit::default()));
        assert!(
            !survives_cold_scan(tuned, 8),
            "if the tuned default now survives a cold scan, the residency/scan \
             trade-off has changed and TwoQSplit::default's rationale needs re-checking"
        );
    }

    /// The percentage encoding of [`TwoQSplit::CLASSICAL`] must land on the paper's
    /// `cap/4` probation and `cap/2` ghost exactly — 4 and 2 divide 100, so
    /// `cap*25/100 == cap/4` and `cap*50/100 == cap/2` for EVERY cap, the same
    /// integer rather than a rounding coincidence. Checked across every capacity
    /// shape (odd, prime, powers of two) — the small caps also exercise `kin`'s
    /// `clamp(1, cap-1)` floor/ceiling.
    #[test]
    fn classical_split_maps_to_paper_fractions() {
        let d = TwoQSplit::CLASSICAL;
        for cap in [1usize, 2, 3, 4, 7, 8, 13, 64, 97, 100, 101, 1024, 4099] {
            let ceiling = cap.saturating_sub(1).max(1);
            assert_eq!(
                d.kin(cap),
                (cap / 4).max(1).min(ceiling),
                "kin drifted at cap={cap}"
            );
            assert_eq!(d.kout(cap), (cap / 2).max(1), "kout drifted at cap={cap}");
        }
    }

    /// A1in must stay a strict fraction of capacity, or `reclaim` can neither trim
    /// probation nor find an Am victim and residency grows past `cap` — the one thing
    /// the pin's slot bookkeeping cannot survive.
    #[test]
    fn any_admissible_split_respects_capacity() {
        for kin in [1u32, 25, 50, 99] {
            for kout in [1u32, 50, 1000] {
                let split = match TwoQSplit::new(kin, kout) {
                    Ok(s) => s,
                    Err(e) => panic!("{kin}/{kout} should be admissible: {e}"),
                };
                let mut c = TwoQ::new(16, split);
                for k in 0..500u32 {
                    if !c.get(k) {
                        let _ = if k.is_multiple_of(3) {
                            c.insert_cold(k)
                        } else {
                            c.insert(k)
                        };
                    }
                    assert!(c.resident_len() <= 16, "kin={kin} kout={kout}: over cap");
                }
            }
        }
    }

    #[test]
    fn split_rejects_out_of_range() {
        assert_eq!(TwoQSplit::new(0, 50), Err(SplitError::KinRange(0)));
        assert_eq!(TwoQSplit::new(100, 50), Err(SplitError::KinRange(100)));
        assert_eq!(TwoQSplit::new(25, 0), Err(SplitError::KoutRange(0)));
        assert_eq!(TwoQSplit::new(25, 1001), Err(SplitError::KoutRange(1001)));
    }

    // The format hybrid's load-bearing invariant: in fixed-partition 2Q an insert
    // only ever evicts from the SAME tier it enters (so the pool frees + allocates in
    // one slab), and each segment stays within its cap.
    #[test]
    fn fixed_partition_evicts_same_tier() {
        use std::collections::HashMap;
        let (n_cold, n_hot) = (4usize, 8usize);
        // Ghost bigger than the working set so a key survives in A1out between passes
        // (evicted from A1in → ghost → re-accessed → promoted to Am).
        let mut c = TwoQ::fixed(n_cold, n_hot, 64);
        let mut tier_of: HashMap<u32, Tier> = HashMap::new();
        for _pass in 0..6 {
            for k in 0..14u32 {
                if c.get(k) {
                    continue;
                }
                let (ev, tier) = c.insert(k);
                if let Some(ev) = ev {
                    assert_eq!(
                        tier_of.remove(&ev),
                        Some(tier),
                        "evicted {ev} was not in the {tier:?} tier that {k} entered"
                    );
                }
                tier_of.insert(k, tier);
                let hot = tier_of.values().filter(|&&t| t == Tier::Hot).count();
                let cold = tier_of.values().filter(|&&t| t == Tier::Cold).count();
                assert!(hot <= n_hot && cold <= n_cold, "over cap: hot {hot} cold {cold}");
                assert_eq!(hot + cold, c.resident_len(), "tier tally != resident_len");
            }
        }
        // Promotions must have actually happened (else the test proves nothing).
        assert!(tier_of.values().any(|&t| t == Tier::Hot), "no key ever promoted to Hot");
    }

    // Same load-bearing invariant for fixed-partition ARC: an insert only evicts from
    // the tier it enters, caps hold, and B1/B2 ghost hits promote to T2/HOT.
    #[test]
    fn arc_fixed_partition_evicts_same_tier() {
        use std::collections::HashMap;
        let (n_cold, n_hot) = (4usize, 8usize);
        let mut c = Arc::fixed(n_cold, n_hot, 64); // big ghost so keys survive to promote
        let mut tier_of: HashMap<u32, Tier> = HashMap::new();
        for _pass in 0..6 {
            for k in 0..14u32 {
                if c.get(k) {
                    continue;
                }
                let (ev, tier) = c.insert(k);
                if let Some(ev) = ev {
                    assert_eq!(
                        tier_of.remove(&ev),
                        Some(tier),
                        "evicted {ev} was not in the {tier:?} tier that {k} entered"
                    );
                }
                tier_of.insert(k, tier);
                let hot = tier_of.values().filter(|&&t| t == Tier::Hot).count();
                let cold = tier_of.values().filter(|&&t| t == Tier::Cold).count();
                assert!(hot <= n_hot && cold <= n_cold, "over cap: hot {hot} cold {cold}");
                assert_eq!(hot + cold, c.resident_len(), "tier tally != resident_len");
            }
        }
        assert!(tier_of.values().any(|&t| t == Tier::Hot), "no key ever promoted to Hot");
    }

    // Fixed-partition LRU (frequency-counter admission): the same slab invariant
    // (evicted_tier == insert_tier, caps hold) under a SKEWED workload — hot keys 0/1
    // reused each pass with ≥n_cold fillers between reuses, so they evict from COLD,
    // re-miss, and promote to HOT (a uniform workload never promotes — correctly).
    #[test]
    fn lru_fixed_partition_evicts_same_tier() {
        use std::collections::HashMap;
        let (n_cold, n_hot) = (8usize, 16usize);
        let mut c = Lru::fixed(n_cold, n_hot);
        let mut tier_of: HashMap<u32, Tier> = HashMap::new();

        fn touch(c: &mut Lru, k: u32, tier_of: &mut HashMap<u32, Tier>, n_cold: usize, n_hot: usize) {
            if c.get(k) {
                return;
            }
            let (ev, tier) = c.insert(k);
            if let Some(ev) = ev {
                assert_eq!(
                    tier_of.remove(&ev),
                    Some(tier),
                    "evicted {ev} was not in the {tier:?} tier that {k} entered"
                );
            }
            tier_of.insert(k, tier);
            let hot = tier_of.values().filter(|&&t| t == Tier::Hot).count();
            let cold = tier_of.values().filter(|&&t| t == Tier::Cold).count();
            assert!(hot <= n_hot && cold <= n_cold, "over cap: hot {hot} cold {cold}");
            assert_eq!(hot + cold, c.resident_len(), "tier tally != resident_len");
        }

        for pass in 0..12u32 {
            touch(&mut c, 0, &mut tier_of, n_cold, n_hot);
            touch(&mut c, 1, &mut tier_of, n_cold, n_hot);
            for f in 0..(n_cold as u32 + 2) {
                touch(&mut c, 1000 + pass * 100 + f, &mut tier_of, n_cold, n_hot);
            }
            // second reuse after the fillers evicted them → re-miss, freq climbs → HOT
            touch(&mut c, 0, &mut tier_of, n_cold, n_hot);
            touch(&mut c, 1, &mut tier_of, n_cold, n_hot);
        }
        assert!(tier_of.values().any(|&t| t == Tier::Hot), "no key ever promoted to Hot");
    }

    // A first-seen key admits COLD; a key re-accessed after eviction (freq crosses the
    // threshold) admits HOT — the whole point of frequency-counter admission. n_cold=1
    // so a cold key evicts on the very next cold miss, giving a clean re-miss on `x`.
    #[test]
    fn lru_hybrid_admits_by_frequency() {
        let mut c = Lru::fixed(1, 4); // cap 5 → decay only at 5 accesses; we stay under
        let x = 10u32;
        // First access to x: miss → COLD (freq 1).
        assert!(!c.get(x));
        assert_eq!(c.insert(x).1, Tier::Cold, "first-seen must admit COLD");
        // A different cold key evicts x from the 1-slot COLD segment.
        assert!(!c.get(20));
        assert_eq!(c.insert(20).1, Tier::Cold);
        // Second access to x: now a miss (evicted) with freq 2 → HOT.
        assert!(!c.get(x));
        assert_eq!(c.insert(x).1, Tier::Hot, "a re-accessed key must admit HOT");
    }

    /// `replay` must model the prefetch admission path the live engine depends on:
    /// a key cold-admitted for layer L and then demanded by L counts as `preloading`
    /// (its read is still in flight), and its NEXT demand — now promoted out of
    /// probation — counts as `loaded`. Without this, offline Kin/Kout tuning would be
    /// measuring a no-prefetch engine.
    #[test]
    fn replay_models_prefetch_admission() {
        let (a, b) = ([7u32], [7u32]);
        let trace = [
            Layer {
                demand: &a,
                predicted: &b,
            }, // prefetched, then demanded -> preloading
            Layer {
                demand: &a,
                predicted: &[],
            }, // already resident -> loaded
        ];
        let r = match replay("2q", 8, TwoQSplit::default(), &[], &trace) {
            Some(r) => r,
            None => panic!("2q must be a known policy"),
        };
        assert_eq!(
            r,
            Residency {
                loaded: 1,
                preloading: 1,
                cold: 0
            }
        );
        // Demand-only replay of the same keys: the first access is a cold miss.
        let (h, t) = simulate("2q", 8, TwoQSplit::default(), &[], &[vec![7u32], vec![7]]);
        assert_eq!((h, t), (1, 2), "no-prefetch replay must show the cold miss");
    }

    #[test]
    fn all_hit_when_working_set_fits() {
        let batches: Vec<Vec<u32>> = (0..50).map(|_| vec![1, 2, 3, 4]).collect();
        for pol in ["lru", "2q", "arc"] {
            let (hits, total) = simulate(pol, 8, TwoQSplit::default(), &[], &batches);
            assert_eq!(total - hits, 4, "{pol}: only cold-start misses expected");
        }
    }

    #[test]
    fn seed_gives_immediate_hits() {
        let batches = vec![vec![1u32, 2, 3, 4]];
        for pol in ["lru", "2q", "arc"] {
            let (hits, total) = simulate(pol, 8, TwoQSplit::default(), &[1, 2, 3, 4], &batches);
            assert_eq!(hits, total, "{pol}: seeded working set should fully hit");
        }
    }

    /// Residency never exceeds capacity, and evicted keys are always resident (the
    /// invariant the live pin relies on to reuse the evicted key's slot). Alternates
    /// insert and insert_cold so the prefetch cold-park path is covered too.
    #[test]
    fn eviction_returns_resident_and_respects_cap() {
        for pol in ["lru", "2q", "arc"] {
            let mut c = boxed(pol, 16);
            let mut resident = std::collections::HashSet::new();
            for k in 0..200u32 {
                if !c.get(k) {
                    let (ev, _tier) = if k.is_multiple_of(3) {
                        c.insert_cold(k)
                    } else {
                        c.insert(k)
                    };
                    if let Some(ev) = ev {
                        assert!(resident.remove(&ev), "{pol}: evicted {ev} was not resident");
                    }
                    resident.insert(k);
                }
                assert!(resident.len() <= 16, "{pol}: over cap ({})", resident.len());
                assert_eq!(resident.len(), c.resident_len(), "{pol}: pin/policy drift");
            }
        }
    }

    /// Finding A regression: a prefetch batch must COEXIST. At capacity, consecutive
    /// insert_cold calls must each evict an OLD resident, never a just-inserted batch
    /// sibling. The old LRU `touch_cold` made the 2nd cold insert evict the 1st.
    #[test]
    fn cold_batch_coexists_at_capacity() {
        for pol in ["lru", "2q", "arc"] {
            let mut c = boxed(pol, 16);
            for k in 0..16u32 {
                c.insert(k); // fill to capacity
            }
            let (a, b, d) = (1000u32, 1001, 1002);
            c.insert_cold(a);
            c.insert_cold(b);
            c.insert_cold(d);
            assert!(
                c.contains(a),
                "{pol}: 1st prefetch evicted by a later batch sibling"
            );
            assert!(c.contains(b), "{pol}: 2nd prefetch not resident");
            assert!(c.contains(d), "{pol}: 3rd prefetch not resident");
            assert!(c.resident_len() <= 16, "{pol}: over cap");
        }
    }

    /// Finding B regression: cold-inserting a key that is currently a GHOST must NOT
    /// promote it into the protected/frequency segment (2Q Am / ARC T2) — a
    /// speculative prefetch stays in probation, so a later probation-churning scan
    /// evicts it. If it had been promoted, it would survive the churn.
    #[test]
    fn cold_insert_does_not_promote_ghosts() {
        for pol in ["2q", "arc"] {
            let mut c = boxed(pol, 8);
            let g = 500u32;
            c.insert(g);
            for k in 0..8u32 {
                c.insert(k); // displace g → it becomes a ghost
            }
            assert!(
                !c.contains(g),
                "{pol}: setup — g should be a ghost, not resident"
            );
            c.insert_cold(g); // prefetch the ghost
            assert!(
                c.contains(g),
                "{pol}: g should be resident after cold insert"
            );
            for k in 100..140u32 {
                c.insert_cold(k); // probation-churning prefetch stream
            }
            assert!(
                !c.contains(g),
                "{pol}: ghost prefetch was promoted to the protected set (survived churn)"
            );
        }
    }
}
