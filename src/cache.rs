//! Cache-policy simulator for the routed-expert pool. Pure key-only replay: given
//! the deterministic `(layer,expert)` access trace (one batch per MoE layer, the
//! same keys `resolve_layer` looks up), it computes the hit rate under LRU, 2Q, or
//! ARC at a chosen slot count — decoupled from the GPU, so policies A/B in
//! milliseconds instead of ~90s decode runs. The live pin keeps its own recency
//! LRU; this module proves a policy (and its usage seed) before wiring the winner.
//!
//! Why bother: decode is now hit-rate-bound (tok/s tracks hit%), and a plain
//! recency LRU has no scan resistance — a token's cold-tail experts evict the
//! recurring hot set. 2Q and ARC protect a frequency segment. Complexity gradient:
//! LRU (recency only) < 2Q (frequency, fixed split) < ARC (frequency, adaptive
//! split + ghosts). ARC fits unusually well here: the value is ~18 MB but the key
//! is 4 bytes, so ghost history is nearly free and can far exceed the cache.

use std::collections::{BTreeMap, HashMap};

/// Recency-ordered set of keys: O(log n) MRU-insert / arbitrary-remove / pop-LRU
/// via a monotonic tick clock + a `BTreeMap` whose front is the LRU end. FIFO
/// lists use `push` (as MRU) + `pop_lru` (oldest) and never re-touch on hit.
#[derive(Default)]
struct OrderedSet {
    at: HashMap<u32, u64>,
    order: BTreeMap<u64, u32>,
    clock: u64,
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
        if let Some(&t) = self.at.get(&k) {
            self.order.remove(&t);
        }
        self.clock += 1;
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

/// A replayable cache policy. `probe` reports a hit and promotes on hit (no
/// residency change on a miss); `load` admits a known-miss key (evicting/adapting
/// as the policy dictates). `access_batch` runs one layer's keys two-pass —
/// hits first, then misses — mirroring `resolve_layer` (a miss can't evict a
/// same-batch hit). `seed` pre-fills the policy's protected/frequency segment.
pub trait Cache {
    fn probe(&mut self, k: u32) -> bool;
    fn load(&mut self, k: u32);
    fn seed(&mut self, keys: &[u32]);

    fn access_batch(&mut self, keys: &[u32]) -> usize {
        // top_k is small (≤8 + shared); a fixed miss scratch avoids an alloc.
        let mut miss = [0u32; 32];
        let mut nm = 0;
        let mut hits = 0;
        for &k in keys {
            if self.probe(k) {
                hits += 1;
            } else if nm < miss.len() {
                miss[nm] = k;
                nm += 1;
            }
        }
        for &k in &miss[..nm] {
            self.load(k);
        }
        hits
    }
}

// ---------------------------------------------------------------------------
// LRU — pure recency (mirrors the live pin's Lru; the fidelity baseline).
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
    fn probe(&mut self, k: u32) -> bool {
        if self.set.contains(k) {
            self.set.touch(k);
            return true;
        }
        false
    }
    fn load(&mut self, k: u32) {
        if self.set.len() >= self.cap {
            self.set.pop_lru();
        }
        self.set.touch(k);
    }
    fn seed(&mut self, keys: &[u32]) {
        for &k in keys.iter().take(self.cap) {
            self.set.touch(k);
        }
    }
}

// ---------------------------------------------------------------------------
// 2Q (Johnson & Shasha) — frequency with a FIXED split. A1in FIFO holds
// first-timers; a second *distinct* access (via the A1out ghost) promotes into
// the Am LRU, which scans can't pollute. No adaptation.
// ---------------------------------------------------------------------------
pub struct TwoQ {
    cap: usize,
    kin: usize,  // A1in cap (first-timer FIFO)
    kout: usize, // A1out cap (ghost of A1in)
    a1in: OrderedSet,
    a1out: OrderedSet,
    am: OrderedSet,
}
impl TwoQ {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            kin: (cap / 4).max(1),
            kout: (cap / 2).max(1),
            a1in: OrderedSet::default(),
            a1out: OrderedSet::default(),
            am: OrderedSet::default(),
        }
    }
    /// Free one resident page for a new admission (paper's "reclaimfor"): trim an
    /// oversized A1in into the ghost, else evict Am's LRU.
    fn reclaim(&mut self) {
        if self.a1in.len() + self.am.len() < self.cap {
            return;
        }
        if self.a1in.len() > self.kin {
            if let Some(v) = self.a1in.pop_lru() {
                self.a1out.touch(v);
                while self.a1out.len() > self.kout {
                    self.a1out.pop_lru();
                }
            }
        } else {
            self.am.pop_lru();
        }
    }
}
impl Cache for TwoQ {
    fn probe(&mut self, k: u32) -> bool {
        if self.am.contains(k) {
            self.am.touch(k); // Am is LRU
            return true;
        }
        self.a1in.contains(k) // hit but stays put (FIFO); ghosts are not resident
    }
    fn load(&mut self, k: u32) {
        self.reclaim();
        if self.a1out.remove(k) {
            self.am.touch(k); // second distinct access → promote to protected
        } else {
            self.a1in.touch(k); // first sighting → probation FIFO
        }
    }
    fn seed(&mut self, keys: &[u32]) {
        for &k in keys.iter().take(self.cap) {
            self.am.touch(k); // seed straight into the scan-protected segment
        }
    }
}

// ---------------------------------------------------------------------------
// ARC (Megiddo & Modha) — frequency with an ADAPTIVE split. T1 (recency) / T2
// (frequency) resident; B1 / B2 key-only ghosts drive the target `p` that splits
// them. A B1 ghost-hit grows T1 (recency); a B2 ghost-hit grows T2 (frequency).
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
    /// Evict one resident page to a ghost list to make room (paper's REPLACE).
    /// `in_b2` = the key being admitted is a B2 ghost-hit (biases toward evicting
    /// from T1 at the boundary).
    fn replace(&mut self, in_b2: bool) {
        if self.t1.len() >= 1 && (self.t1.len() > self.p || (in_b2 && self.t1.len() == self.p)) {
            if let Some(v) = self.t1.pop_lru() {
                self.b1.touch(v);
            }
        } else if let Some(v) = self.t2.pop_lru() {
            self.b2.touch(v);
        }
    }
}
impl Cache for Arc {
    fn probe(&mut self, k: u32) -> bool {
        // Hit in T1 or T2 → move to MRU of T2 (it's now used ≥twice).
        if self.t1.remove(k) || self.t2.contains(k) {
            self.t2.touch(k);
            return true;
        }
        false
    }
    fn load(&mut self, k: u32) {
        let c = self.c;
        if self.b1.contains(k) {
            // Ghost-hit in B1: recency was undersized → grow p.
            let delta = (self.b2.len() / self.b1.len().max(1)).max(1);
            self.p = (self.p + delta).min(c);
            self.replace(false);
            self.b1.remove(k);
            self.t2.touch(k);
        } else if self.b2.contains(k) {
            // Ghost-hit in B2: frequency was undersized → shrink p.
            let delta = (self.b1.len() / self.b2.len().max(1)).max(1);
            self.p = self.p.saturating_sub(delta);
            self.replace(true);
            self.b2.remove(k);
            self.t2.touch(k);
        } else {
            // Cold miss. Keep the L1 (T1+B1) and total (T1+T2+B1+B2) invariants.
            if self.t1.len() + self.b1.len() == c {
                if self.t1.len() < c {
                    self.b1.pop_lru();
                    self.replace(false);
                } else {
                    self.t1.pop_lru();
                }
            } else if self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len() >= c {
                if self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len() == 2 * c {
                    self.b2.pop_lru();
                }
                self.replace(false);
            }
            self.t1.touch(k);
        }
    }
    fn seed(&mut self, keys: &[u32]) {
        for &k in keys.iter().take(self.c) {
            self.t2.touch(k); // seed straight into the frequency segment
        }
    }
}

/// Replay `batches` through a fresh `policy` at `cap` slots, optionally seeding the
/// protected segment first. Returns `(hits, accesses)`.
pub fn simulate(policy: &str, cap: usize, seed: &[u32], batches: &[Vec<u32>]) -> (u64, u64) {
    let mut cache: Box<dyn Cache> = match policy {
        "lru" => Box::new(Lru::new(cap)),
        "2q" => Box::new(TwoQ::new(cap)),
        "arc" => Box::new(Arc::new(cap)),
        other => panic!("unknown policy {other:?} (lru|2q|arc)"),
    };
    if !seed.is_empty() {
        cache.seed(seed);
    }
    let (mut hits, mut total) = (0u64, 0u64);
    for b in batches {
        hits += cache.access_batch(b) as u64;
        total += b.len() as u64;
    }
    (hits, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property that motivates the whole exercise: a key established as frequent
    /// must survive a later UNINTERRUPTED cold scan longer than the cache. A
    /// frequency policy promotes it to a protected segment (which the first-timer
    /// scan can't reach); pure recency ages it out. `hot` is promoted by touching it
    /// with just enough churn between touches to age it into the ghost and back
    /// (a burst of `kin+1`), then left untouched through the scan.
    fn survives_cold_scan<C: Cache>(mut c: C, cap: usize) -> bool {
        let hot = 999_999u32;
        let mut churn = 1_000_000u32;
        let burst = cap / 4 + 1; // push hot out of probation into the ghost, not past it
        for _ in 0..6 {
            c.access_batch(&[hot]);
            for _ in 0..burst {
                c.access_batch(&[churn]);
                churn += 1;
            }
        }
        c.access_batch(&[hot]); // final protected touch
        // Uninterrupted cold scan, 5x the cache, hot never touched.
        for _ in 0..(cap * 5) {
            c.access_batch(&[churn]);
            churn += 1;
        }
        c.probe(hot)
    }

    #[test]
    fn lru_has_no_scan_resistance() {
        assert!(
            !survives_cold_scan(Lru::new(8), 8),
            "recency LRU should drop the hot key under a cold scan"
        );
    }

    #[test]
    fn twoq_and_arc_survive_scan() {
        assert!(
            survives_cold_scan(TwoQ::new(8), 8),
            "2Q must protect the hot key"
        );
        assert!(
            survives_cold_scan(Arc::new(8), 8),
            "ARC must protect the hot key"
        );
    }

    #[test]
    fn all_hit_when_working_set_fits() {
        // Working set of 4 keys, cap 8: after warm-up every access hits, all policies.
        let batches: Vec<Vec<u32>> = (0..50).map(|_| vec![1, 2, 3, 4]).collect();
        for pol in ["lru", "2q", "arc"] {
            let (hits, total) = simulate(pol, 8, &[], &batches);
            // Only the first sighting of each of the 4 keys misses.
            assert_eq!(total - hits, 4, "{pol}: only cold-start misses expected");
        }
    }

    #[test]
    fn seed_gives_immediate_hits() {
        // With the working set pre-seeded, even the first batch is all hits.
        let batches = vec![vec![1u32, 2, 3, 4]];
        for pol in ["lru", "2q", "arc"] {
            let (hits, total) = simulate(pol, 8, &[1, 2, 3, 4], &batches);
            assert_eq!(hits, total, "{pol}: seeded working set should fully hit");
        }
    }
}
