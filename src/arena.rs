//! Two-ended byte arena for the format-hybrid routed pool. COLD (int3-VQ) slots pack
//! from the low end, HOT (int4) slots from the high end, tightly (no per-slab byte
//! waste). The boundary between them FLOATS with the cache policy's split.
//!
//! Because the two slot sizes differ (int4 > vq3), growing one tier past the shared
//! middle gap means COMPACTING the other: relocate its boundary slot into a freed hole
//! so its frontier can retreat and hand bytes to the gap. This module owns that integer
//! geometry and emits [`Reloc`] events; the pin executes each as a device memcpy of the
//! expert weights and remaps the key. Pure `usize` bookkeeping here — fully host-tested.
//!
//! Driver contract (the pin's alloc loop): call [`Arena::alloc_step`] repeatedly for the
//! wanted tier until it returns [`Step::Placed`]. On [`Step::Relocated`] copy the slot
//! bytes and remap that key, then call again; on [`Step::NeedFree`] evict one slot
//! (the OTHER tier first) and [`Arena::free`] it, then call again.

/// Relocate the slot at `from` to `to` within the `hot` tier (a device memcpy of one
/// expert's bytes). The key living at `from` moves to `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reloc {
    pub hot: bool,
    pub from: usize,
    pub to: usize,
}

/// One step of allocating a slot. See the module contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// A slot of the requested tier is ready at this index — done.
    Placed(usize),
    /// Copy the slot bytes `from`→`to` and remap that key, then call `alloc_step` again.
    Relocated(Reloc),
    /// No room and nothing to compact into — evict one slot, `free` it, then retry.
    NeedFree,
}

/// The two-ended slot arena. Cold grows up from 0, hot grows down from `budget`.
pub struct Arena {
    budget: usize,
    cold_stride: usize,
    hot_stride: usize,
    /// Frontiers: slot counts, so cold owns `[0, cold_hi·cs)` and hot owns
    /// `[budget − hot_hi·hs, budget)`. The gap between them is the shared free space.
    cold_hi: usize,
    hot_hi: usize,
    /// Freed slot indices strictly below each frontier (holes to reuse/compact into).
    /// `free` keeps the top of each region packed, so a frontier−1 slot is never here.
    cold_free: Vec<usize>,
    hot_free: Vec<usize>,
}

impl Arena {
    pub fn new(budget: usize, cold_stride: usize, hot_stride: usize) -> Self {
        Self {
            budget,
            cold_stride: cold_stride.max(1),
            hot_stride: hot_stride.max(1),
            cold_hi: 0,
            hot_hi: 0,
            cold_free: Vec::new(),
            hot_free: Vec::new(),
        }
    }

    /// Byte offset of slot `idx` in the `hot`/cold tier — where the pin points the
    /// expert descriptor and DMAs the cold read.
    pub fn offset(&self, hot: bool, idx: usize) -> usize {
        if hot {
            self.budget - (idx + 1) * self.hot_stride
        } else {
            idx * self.cold_stride
        }
    }

    fn stride(&self, hot: bool) -> usize {
        if hot { self.hot_stride } else { self.cold_stride }
    }
    fn frontier(&self, hot: bool) -> usize {
        if hot { self.hot_hi } else { self.cold_hi }
    }
    fn set_frontier(&mut self, hot: bool, v: usize) {
        if hot {
            self.hot_hi = v;
        } else {
            self.cold_hi = v;
        }
    }
    fn free_list(&mut self, hot: bool) -> &mut Vec<usize> {
        if hot { &mut self.hot_free } else { &mut self.cold_free }
    }

    /// Free bytes between the two frontiers (the only space either tier can grow into).
    fn gap(&self) -> usize {
        self.budget - self.cold_hi * self.cold_stride - self.hot_hi * self.hot_stride
    }

    /// Release slot `idx` of the `hot` tier. Cascades: if it (now) sits at the frontier,
    /// retreat the frontier — and keep retreating while the new top is also free — so
    /// the freed bytes rejoin the gap and the free list never holds a frontier−1 slot.
    pub fn free(&mut self, hot: bool, idx: usize) {
        self.free_list(hot).push(idx);
        loop {
            let f = self.frontier(hot);
            if f == 0 {
                break;
            }
            let top = f - 1;
            let list = self.free_list(hot);
            if let Some(p) = list.iter().position(|&x| x == top) {
                list.swap_remove(p);
                self.set_frontier(hot, top);
            } else {
                break;
            }
        }
    }

    /// One step toward placing a slot of the `hot` tier. See the module contract.
    pub fn alloc_step(&mut self, hot: bool) -> Step {
        let need = self.stride(hot);
        // 1) A hole in this tier — reuse it outright.
        if let Some(i) = self.free_list(hot).pop() {
            return Step::Placed(i);
        }
        // 2) The gap has room — grow this frontier into it.
        if self.gap() >= need {
            let i = self.frontier(hot);
            self.set_frontier(hot, i + 1);
            return Step::Placed(i);
        }
        // 3) Compact the other tier: relocate its top slot into one of its holes so its
        //    frontier retreats and the gap grows. Needs a hole to move into.
        let other = !hot;
        let ohi = self.frontier(other);
        if ohi == 0 {
            return Step::NeedFree; // other tier empty — caller must free THIS tier
        }
        let top = ohi - 1;
        if let Some(h) = self.free_list(other).pop() {
            self.set_frontier(other, top); // top vacated → gap grows by the other stride
            return Step::Relocated(Reloc { hot: other, from: top, to: h });
        }
        // 4) Other tier is packed solid (no holes) — caller must evict one of its slots.
        Step::NeedFree
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use std::collections::HashMap;

    // A stand-in for the pin: drives alloc_step, applies relocations, and — the point —
    // asserts that no two live slots ever overlap in bytes. `key` identity survives
    // relocation, so this catches a bad Reloc (wrong offset / lost key / overlap).
    struct Sim {
        a: Arena,
        at: HashMap<u32, (bool, usize)>, // key -> (hot, idx)
        next_free_victim: Vec<u32>,      // keys to evict on NeedFree, LRU-ish (front = oldest)
    }
    impl Sim {
        fn new(budget: usize, cs: usize, hs: usize) -> Self {
            Sim { a: Arena::new(budget, cs, hs), at: HashMap::new(), next_free_victim: Vec::new() }
        }
        fn byte_range(&self, hot: bool, idx: usize) -> (usize, usize) {
            let o = self.a.offset(hot, idx);
            (o, o + self.a.stride(hot))
        }
        fn assert_no_overlap(&self) {
            let mut spans: Vec<(usize, usize, u32)> =
                self.at.iter().map(|(&k, &(h, i))| {
                    let (lo, hi) = self.byte_range(h, i);
                    (lo, hi, k)
                }).collect();
            spans.sort();
            for w in spans.windows(2) {
                assert!(w[0].1 <= w[1].0, "overlap: key {} [{},{}) vs key {} [{},{})",
                    w[0].2, w[0].0, w[0].1, w[1].2, w[1].0, w[1].1);
                assert!(w[1].1 <= self.a.budget, "slot past budget");
            }
        }
        // Admit key `k` into `hot`, evicting the oldest OTHER-tier (then any) key on NeedFree.
        fn admit(&mut self, k: u32, hot: bool) {
            loop {
                match self.a.alloc_step(hot) {
                    Step::Placed(idx) => {
                        self.at.insert(k, (hot, idx));
                        self.next_free_victim.push(k);
                        break;
                    }
                    Step::Relocated(r) => {
                        // find the key at (r.hot, r.from), move it to (r.hot, r.to)
                        let moved = *self.at.iter()
                            .find(|&(_, &(h, i))| h == r.hot && i == r.from)
                            .expect("relocated slot must hold a key").0;
                        self.at.insert(moved, (r.hot, r.to));
                    }
                    Step::NeedFree => {
                        // evict the oldest victim in the OTHER tier if any, else oldest overall
                        let other = !hot;
                        let pick = self.next_free_victim.iter().position(|k| {
                            self.at.get(k).map(|&(h, _)| h == other).unwrap_or(false)
                        }).or_else(|| (!self.next_free_victim.is_empty()).then_some(0))
                            .expect("nothing to evict but arena full");
                        let v = self.next_free_victim.remove(pick);
                        let (h, i) = self.at.remove(&v).expect("victim resident");
                        self.a.free(h, i);
                    }
                }
            }
        }
    }

    #[test]
    fn packs_both_ends_without_overlap() {
        // vq3 3, int4 4, budget 40 (deliberately incommensurate strides).
        let mut s = Sim::new(40, 3, 4);
        // interleave cold/hot admissions well past capacity so compaction/eviction fire
        for n in 0..200u32 {
            s.admit(n, n % 3 == 0); // ~1/3 hot
            s.assert_no_overlap();
        }
        assert!(!s.at.is_empty(), "arena emptied itself");
    }

    #[test]
    fn cross_tier_growth_compacts() {
        // Fill mostly cold, then admit a run of hot — hot growth must compact cold and
        // stay non-overlapping, and end with hot slots resident.
        let mut s = Sim::new(60, 3, 4);
        for n in 0..30u32 {
            s.admit(n, false); // all cold first
            s.assert_no_overlap();
        }
        for n in 100..130u32 {
            s.admit(n, true); // now force hot growth → cross-tier compaction
            s.assert_no_overlap();
        }
        let hot_live = s.at.values().filter(|&&(h, _)| h).count();
        assert!(hot_live > 0, "no hot slot survived");
    }

    #[test]
    fn free_retreats_frontier_and_reuses() {
        let mut a = Arena::new(30, 3, 3);
        // grow 3 cold
        for i in 0..3 {
            assert_eq!(a.alloc_step(false), Step::Placed(i));
        }
        assert_eq!(a.cold_hi, 3);
        // free the top → frontier retreats to 2 (no hole left behind)
        a.free(false, 2);
        assert_eq!(a.cold_hi, 2);
        assert!(a.cold_free.is_empty());
        // free a middle → becomes a hole; next alloc reuses it (not a new frontier slot)
        a.free(false, 0);
        assert_eq!(a.cold_hi, 2);
        assert_eq!(a.alloc_step(false), Step::Placed(0));
    }

    #[test]
    fn offsets_are_tier_correct() {
        let a = Arena::new(100, 10, 20);
        assert_eq!(a.offset(false, 0), 0);
        assert_eq!(a.offset(false, 3), 30);
        assert_eq!(a.offset(true, 0), 80); // 100 - 20
        assert_eq!(a.offset(true, 1), 60);
    }
}
