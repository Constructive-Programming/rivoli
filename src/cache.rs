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
struct OrderedSet {
    at: HashMap<u32, i64>,
    order: BTreeMap<i64, u32>,
    clock: i64, // ascending MRU inserts
}

impl OrderedSet {
    fn contains(&self, k: u32) -> bool {
        self.at.contains_key(&k)
    }
    fn len(&self) -> usize {
        self.at.len()
    }
    /// Insert `k`, or move it to the MRU end if already present.
    fn touch(&mut self, k: u32) {
        self.clock += 1;
        if let Some(&t) = self.at.get(&k) {
            self.order.remove(&t);
        }
        self.at.insert(k, self.clock);
        self.order.insert(self.clock, k);
    }
    fn remove(&mut self, k: u32) -> bool {
        if let Some(t) = self.at.remove(&k) {
            self.order.remove(&t);
            return true;
        }
        false
    }
    /// Evict and return the LRU (oldest) key.
    fn pop_lru(&mut self) -> Option<u32> {
        let (&t, &k) = self.order.iter().next()?;
        self.order.remove(&t);
        self.at.remove(&k);
        Some(k)
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
pub trait Cache {
    fn contains(&self, k: u32) -> bool;
    fn get(&mut self, k: u32) -> bool;
    fn insert(&mut self, k: u32) -> Option<u32>;
    /// Default = normal `insert` (correct for single-segment `Lru`: a batch coexists,
    /// evicting old normals). Segmented policies override to force probation.
    fn insert_cold(&mut self, k: u32) -> Option<u32> {
        self.insert(k)
    }
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
// LRU — pure recency (mirrors the pin's original hand-rolled Lru).
// ---------------------------------------------------------------------------
pub struct Lru {
    cap: usize,
    set: OrderedSet,
}
impl Lru {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            set: OrderedSet::default(),
        }
    }
}
impl Cache for Lru {
    fn contains(&self, k: u32) -> bool {
        self.set.contains(k)
    }
    fn get(&mut self, k: u32) -> bool {
        if self.set.contains(k) {
            self.set.touch(k);
            return true;
        }
        false
    }
    fn insert(&mut self, k: u32) -> Option<u32> {
        let ev = (self.set.len() >= self.cap)
            .then(|| self.set.pop_lru())
            .flatten();
        self.set.touch(k);
        ev
    }
    // insert_cold: uses the trait default (= insert). A single-segment recency cache
    // cannot both coexist a prefetch batch and cold-park it, so LRU prefetches land
    // as normal MRU inserts (they decay normally if unused). Use 2q/arc to cold-park.
    fn seed(&mut self, keys: &[u32]) {
        for &k in keys.iter().take(self.cap) {
            self.set.touch(k);
        }
    }
    fn resident_len(&self) -> usize {
        self.set.len()
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
    /// Tuned by the offline `replay` sweep and confirmed on hardware (benchmarks.md):
    /// an 11x6 Kin/Kout grid over a real 512-token trace peaks at kin 8 % / kout 100 %,
    /// 92.21 % predicted -> 92.2 % measured, 2.15 -> 2.33 tok/s.
    ///
    /// The shape of that grid is the interesting part: Kout is IRRELEVANT (every
    /// column is flat to within 0.06 pp — the A1out ghost may as well not exist here),
    /// while Kin matters and *smaller is better* down to 8 %, falling off again at 3 %.
    /// Same mechanism as prefetch depth 1 beating depth 2: a tight probation queue
    /// promotes a prefetched expert into the protected Am set before churn can evict
    /// it. The historical 25 %/50 % is preserved as `LEGACY_SPLIT` for A/B.
    fn default() -> Self {
        Self {
            kin_pct: 8,
            kout_pct: 100,
        }
    }
}

impl TwoQSplit {
    /// The pre-sweep split (`cap/4` probation, `cap/2` ghost), kept so the tuning
    /// can be A/B'd and so the percentage encoding stays pinned to the original
    /// hardcoded constants by test.
    pub const LEGACY: Self = Self {
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
            a1in: OrderedSet::default(),
            a1out: OrderedSet::default(),
            am: OrderedSet::default(),
        }
    }
    /// Free one RESIDENT page for a new admission (paper's "reclaimfor"): trim an
    /// oversized A1in into the ghost, else evict Am's LRU. Returns the resident key
    /// that left residency (moved to ghost, or dropped from Am).
    fn reclaim(&mut self) -> Option<u32> {
        if self.a1in.len() + self.am.len() < self.cap {
            return None;
        }
        if self.a1in.len() > self.kin {
            let v = self.a1in.pop_lru();
            if let Some(v) = v {
                self.a1out.touch(v);
                while self.a1out.len() > self.kout {
                    self.a1out.pop_lru();
                }
            }
            v
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
    fn insert(&mut self, k: u32) -> Option<u32> {
        let ev = self.reclaim();
        if self.a1out.remove(k) {
            self.am.touch(k); // second distinct access → promote to protected
        } else {
            self.a1in.touch(k); // first sighting → probation FIFO
        }
        ev
    }
    fn insert_cold(&mut self, k: u32) -> Option<u32> {
        // Prefetch: force into A1in probation, NEVER promoting via the A1out ghost —
        // a speculative (maybe-wrong) prediction must not enter the protected Am set.
        // A batch coexists: reclaim evicts the oldest A1in/Am, never a fresh sibling.
        let ev = self.reclaim();
        self.a1out.remove(k); // drop any stale ghost; do NOT promote
        self.a1in.touch(k);
        ev
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
            t1: OrderedSet::default(),
            t2: OrderedSet::default(),
            b1: OrderedSet::default(),
            b2: OrderedSet::default(),
        }
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
        if self.t1.remove(k) || self.t2.contains(k) {
            self.t2.touch(k);
            return true;
        }
        false
    }
    fn insert(&mut self, k: u32) -> Option<u32> {
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
        evicted
    }
    fn insert_cold(&mut self, k: u32) -> Option<u32> {
        // Prefetch: force into T1 recency probation, NEVER adapting `p` or promoting
        // via the B1/B2 ghosts — a speculative prediction must not corrupt ARC's
        // adaptation signal. Strip any stale ghost, then take the same cold-miss T1
        // admission as `insert` (a batch coexists: replace evicts a T1/T2 LRU, never
        // a fresh sibling).
        self.b1.remove(k);
        self.b2.remove(k);
        self.admit_t1(k)
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
pub fn simulate(
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

    #[test]
    fn twoq_and_arc_survive_scan() {
        assert!(survives_cold_scan(boxed("2q", 8), 8), "2Q must protect hot");
        assert!(
            survives_cold_scan(boxed("arc", 8), 8),
            "ARC must protect hot"
        );
    }

    /// The tuning flags must be a pure refactor at their defaults: `TwoQSplit`'s
    /// 25 %/50 % has to land on the exact `cap/4` / `cap/2` that was hardcoded when
    /// benchmarks.md measured 91.8 % residency. Checked across every capacity shape
    /// (odd, prime, powers of two) since the claim rests on 4 and 2 dividing 100.
    #[test]
    fn default_split_reproduces_hardcoded_constants() {
        // The percentage encoding must reproduce the ORIGINAL hardcoded constants
        // exactly — that is what makes `--2q-kin/--2q-kout` a faithful
        // generalisation rather than a re-parameterisation. 4 and 2 divide 100, so
        // `cap*25/100 == cap/4` and `cap*50/100 == cap/2` for EVERY cap: the same
        // integer, not a rounding coincidence. (The DEFAULT has since moved to
        // 8 %/100 % on measured evidence; this pins the legacy split it replaced.)
        let d = TwoQSplit::LEGACY;
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
                    let ev = if k.is_multiple_of(3) {
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
