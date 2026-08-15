//! The routed-expert streaming pool: residency, eviction, relocation and the io_uring
//! cold reads, over ONE routed format.
//!
//! Ported from `old:src/routed.rs` with one deliberate narrowing: **the pool is
//! single-format** (M4 design decision, 2026-08-15). The old pool carried two [`TierFmt`]s
//! and `submit` reported a per-expert `Vec<RoutedFmt>` — which tier residency placed an
//! expert in decided which ARITHMETIC decoded it. That channel is the old tree's #2 open
//! defect (`--max-mem` changes text in hybrid, a P4 violation), and it is deleted here
//! rather than tested against: the pool owns one [`RoutedGeom`], `submit` returns slots
//! and tickets only, and a caller that wants the format asks [`RoutedPool::fmt`], which
//! cannot vary per expert. When hybrid returns it comes as a `FormatPlan` fixed at
//! `open()` from startup inputs — format bound to identity, never to residency — and the
//! plan, not the pool, answers "which kernel".
//!
//! What made the port cheap is that the substrate below was already byte-parameterised:
//! [`Arena`] takes strides and never names a format, the policies account in bytes,
//! [`AsyncFetch`]/[`ReadSpec`]/[`Streamer`] move `(fd, begin, len) -> dst` spans, and
//! `ExpertSet` reads its geometry off `RoutedFmt`.
//!
//! What this does NOT own: which experts to fetch. Routing never consults residency
//! (INV-1), so the pool is told a selection and reports where the bytes are.
//!
//! Two siblings carry the halves that have no interaction with residency, moved out when
//! this file crossed the 800-line soft cap (`crates/cli/build.rs`) — a split by cohesion,
//! and both are re-exported here so every `crate::routed::X` path is unchanged:
//! [`mod geom`](geom) is the static layout read once from the artifact (key packing, slot
//! pointers, read-spec table) and [`mod trace`](trace) is the whole `--trace` v2 capture.
//! What is left in this file is the mutable part: the arena, the policy, the fetch ring.

mod geom;
mod trace;

pub use geom::{ExpertSlot, ProjSlot, RoutedGeom, expert_key, slot_at};
pub use trace::{RankWindow, TRACE_WINDOW};

use crate::device::VmmBuf;
use crate::fetch::asyncfetch::{AsyncFetch, ReadSpec, Ticket};
use crate::fetch::stream::{Streamer, slot_span};
use anyhow::{Context, Result, bail, ensure};
use rivoli_artifact::format::RoutedFmt;
use rivoli_backend::memcpy_dtod;
use rivoli_core::arena::{AllocOutcome, Arena, Reloc};
use rivoli_core::cache;
use rivoli_core::hybrid::HybridPolicy;
use std::collections::HashMap;
use trace::open_trace;

/// The largest selection one [`RoutedPool::submit`] call may carry.
///
/// It sizes exactly one thing: `submit`'s own `[bool; MAX_BATCH]` hit scratch, which is
/// why `submit` `ensure!`s against it rather than trusting a caller. The GLM loop sizes
/// its descriptor buffer from a RUNTIME `top_k · MAXROW + n_shared` (17 for GLM:
/// 8 · 2 + 1), not from this; the pin checks that value against this one at startup so
/// the friendly message arrives before the run rather than during it.
pub const MAX_BATCH: usize = 32;

/// [`RoutedPool::new`]'s knobs, bundled: every field is a startup-time decision (INV-1's
/// shape — nothing here can vary per token), and bundling them is what keeps `new` and
/// `submit` at signatures a reader can hold. Borrowed strs, `Copy`-cheap, built once.
#[derive(Clone, Copy)]
pub struct PoolCfg<'a> {
    /// Device bytes the pool may hold — see [`pool_budget`] for the alignment rule.
    pub budget: usize,
    /// One layer's demand count; sizes the io_uring ring, NOT the batch scratch.
    pub top_k: usize,
    /// The largest `submit` batch the caller will send; checked against [`MAX_BATCH`].
    pub max_batch: usize,
    /// `lru` | `2q` | `arc`.
    pub policy: &'a str,
    pub two_q: cache::TwoQSplit,
    /// `--trace` sink path, v2 format.
    pub trace: Option<&'a str>,
}

/// One layer's routed selection — the router's picks, in RANK order (load-bearing for
/// the trace: see [`RoutedPool::submit`]).
#[derive(Clone, Copy)]
pub struct Selection<'a> {
    pub layer: usize,
    pub experts: &'a [usize],
}

/// [`RoutedPool::submit`]'s result buffers, caller-owned so the per-layer hot path
/// reuses their capacity. Index i of each answers for `sel[i]`.
#[derive(Default)]
pub struct ResolvedBatch {
    /// Each selected expert's six projection pointers into the pool.
    pub slots: Vec<ExpertSlot>,
    /// Each selected expert's device-side data dependency. Resident experts carry
    /// [`Ticket::RESIDENT`]; there is no residency bool for anyone to branch on.
    pub tickets: Vec<Ticket>,
}

/// The routed-expert pool over the two-ended byte [`Arena`]. A byte-aware
/// [`HybridPolicy`] owns residency and the (floating) COLD/HOT split; on a cross-tier
/// rebalance the arena emits a relocation, which we execute as a synchronous device
/// memcpy of the expert's bytes and remap its key. `slot_of`/`key_at` are inverse maps.
pub struct RoutedPool {
    #[allow(dead_code)] // RAII owner of the pool VMM; addressed via `base`/`host_base`
    buf: VmmBuf,
    /// The DEVICE base: what every expert descriptor's six projection pointers are built
    /// from, and never dereferenced on the CPU.
    base: *mut u8,
    /// The HOST base: the io_uring O_DIRECT DMA target (`ReadSpec.dst`), and the only
    /// one of the two the CPU may touch.
    ///
    /// Under HIP these are the SAME NUMBER — unified addressing — so this field costs
    /// nothing and changes no behaviour. It existed because the retired Vulkan backend
    /// made the two unrelated numbers (2026-08-06); resolving both once here still keeps
    /// [`RoutedPool::ptr`] and [`RoutedPool::host_ptr`] a single `add` each on the fetch
    /// path, and keeps the two spellings from collapsing into one. The split is a naming
    /// CONVENTION now, not something the type system checks.
    host_base: *mut u8,
    arena: Arena,
    policy: Box<dyn HybridPolicy>,
    slot_of: HashMap<u32, (bool, usize)>, // key -> (hot, idx)
    key_at: HashMap<(bool, usize), u32>,  // (hot, idx) -> key, for relocation remap
    /// Keys whose bytes are known to have LANDED in their current slot.
    ///
    /// The engine has no other way to distinguish "the policy says resident" from "the
    /// bytes are actually there", and that distinction is the leading hypothesis for the
    /// old tree's intermittent non-finite-logits bug: a HIT carries `Ticket::RESIDENT`
    /// and the kernel reads the slot immediately, so if a key is ever counted resident
    /// before its load completed, the read is of uninitialised (-> NaN, the visible
    /// case) or stale (-> finite and WRONG, the silent case) memory.
    ///
    /// A key is removed on eviction and on relocation-into, and inserted only when its
    /// read signal has resolved. `trace` only — it costs a hash op per expert per layer.
    loaded: std::collections::HashSet<u32>,
    /// Misses submitted by the PREVIOUS layer, marked loaded at the top of the next
    /// `submit`. Correct because layer L's per-expert awaits and its unconditional
    /// end-of-layer `device_sync` both complete before layer L+1 submits — so by then
    /// every byte of L's batch has landed. Deferring this way avoids plumbing a
    /// completion callback through the layer loop.
    pending_loaded: Vec<u32>,
    geom: RoutedGeom,
    /// Per-expert async cold-fetch: owns the io_uring demand ring on a reaper thread
    /// and signals each miss's [`Ticket`] when its bytes land. The expert stream awaits
    /// these; there is no batch join.
    fetch: AsyncFetch,
    hits: u64,
    misses: u64,
    /// Optional access-trace sink (`--trace`), format v2: a `#` header line, then one
    /// line per resolved MoE layer — the `(layer,expert)` keys looked up in access
    /// order, then `|`, then the top-[`TRACE_WINDOW`] router candidates as
    /// `key:choice` in rank order. Feeds the offline `replay` simulator.
    trace: Option<std::io::BufWriter<std::fs::File>>,
}

/// The pool's share of the device budget: whatever `capacity` leaves after the resident
/// tier's `tier_cap`, rounded **down** to the O_DIRECT block.
///
/// The rounding is not thrift. HOT slots are anchored at the high end
/// (`budget − (idx+1)·stride`), so an unaligned `budget` makes every hot-slot DMA
/// destination violate the alignment `stream.rs` asserts — the base and the strides are
/// already aligned, so the budget is the only way in. It costs <4 KiB. One function
/// because every pin computes it and a rounding that differed between them would
/// misalign one pool's hot-slot DMA destinations and not the other's — a failure that
/// would look like a backend bug in exactly one architecture.
///
/// `saturating_sub` because a `--max-mem` below the resident footprint is a user error
/// with a number attached; it lands as "budget cannot hold one batch" in
/// [`RoutedPool::new`] rather than as a 16-exabyte wrap.
pub fn pool_budget(capacity: usize, tier_cap: usize) -> usize {
    capacity.saturating_sub(tier_cap) & !(crate::fetch::stream::ALIGN - 1)
}

impl RoutedPool {
    /// Build the pool over [`pool_budget`] device bytes, holding `geom`'s one format.
    pub fn new(cfg: PoolCfg<'_>, geom: RoutedGeom) -> Result<Self> {
        let stride = geom.stride;
        // A pool that cannot hold ONE BATCH cannot make progress: every key in a batch
        // is pinned, so `evict_until_fits` finds nothing to reclaim, `alloc_step`
        // returns `NeedFree` and `alloc` bails with "arena NeedFree after policy
        // eviction — byte-accounting bug". That message accuses the arena of a defect
        // the user's `--max-mem` caused. Refused here instead, at startup, with both
        // numbers.
        //
        // **`max_batch`, not `top_k`, and the two genuinely differ.** GLM submits the
        // UNION of `MAXROW` rows' picks, so its batch is `top_k · MAXROW + n_shared`.
        // Sizing this from `top_k` alone left GLM budgets between 8 and 16 slots
        // passing startup and failing mid-run with the arena message — the exact case
        // this check was added to replace. `top_k` still sizes the io_uring ring, which
        // is a per-layer demand count.
        let one_batch = cfg.max_batch * stride;
        ensure!(
            cfg.budget >= one_batch,
            "routed pool budget {:.2} GiB cannot hold one batch of {} experts \
             ({:.2} GiB) — raise --max-mem",
            cfg.budget as f64 / (1u64 << 30) as f64,
            cfg.max_batch,
            one_batch as f64 / (1u64 << 30) as f64,
        );
        let policy = rivoli_core::hybrid::policy_for(
            cfg.policy,
            cfg.budget,
            // One format, so both arena ends carry the same stride; the two-ended
            // arena is 2Q's shape, not a second format's.
            rivoli_core::hybrid::TierStrides {
                cold: stride,
                hot: stride,
            },
            cfg.two_q,
        )
        .with_context(|| format!("unknown --cache-policy {} (lru|2q|arc)", cfg.policy))?;
        // Every benchmark log is keyed on this line, so a mode that reads as another
        // mode is a measurement hazard, not a cosmetic one.
        tracing::info!(
            "routed pool [{} {}]: {:.1} GiB budget (~{} slots, {stride}B/slot)",
            cfg.policy,
            geom.fmt.ext(),
            cfg.budget as f64 / (1u64 << 30) as f64,
            cfg.budget / stride,
        );
        let mut buf = VmmBuf::new(cfg.budget)?;
        let base = buf.ptr_mut();
        // Both bases resolved ONCE, here. Under HIP `host_mut` and `ptr_mut` return the
        // same number, so this is a no-op — pinned by
        // `device.rs::vmmbuf_host_and_device_bases_coincide_under_hip`. The two
        // spellings are kept so the two consumers below cannot bake that coincidence
        // in; they were genuinely different numbers under the Vulkan backend, retired
        // 2026-08-06.
        let host_base = buf.host_mut();
        // Ring sized for one layer's worst case: one demand read per expert, one
        // aligned block each.
        let ring = (cfg.top_k + 4).next_power_of_two();
        let fetch = AsyncFetch::new(Streamer::new(ring as u32, slot_span(stride))?)?;
        Ok(Self {
            buf,
            base,
            host_base,
            arena: Arena::new(cfg.budget, stride, stride),
            policy,
            slot_of: HashMap::new(),
            key_at: HashMap::new(),
            loaded: std::collections::HashSet::new(),
            pending_loaded: Vec::new(),
            geom,
            fetch,
            hits: 0,
            misses: 0,
            trace: cfg.trace.map(|p| open_trace(p, cfg.top_k)).transpose()?,
        })
    }

    /// The pool's ONE routed format — what selects the kernel, and deliberately not a
    /// per-expert answer: residency moves bytes, never arithmetic.
    pub fn fmt(&self) -> RoutedFmt {
        self.geom.fmt
    }

    /// The slot's DEVICE address — the base every expert descriptor's six projection
    /// pointers are built from, and what `memcpy_dtod`/`fill_u32` take.
    ///
    /// Host-dereferenceable under HIP, where unified addressing makes this and
    /// [`RoutedPool::host_ptr`] the same number — but relying on that is what the split
    /// exists to prevent. It was NOT dereferenceable under the retired Vulkan backend
    /// (2026-08-06), which is where the rule came from.
    fn ptr(&self, hot: bool, idx: usize) -> *mut u8 {
        // SAFETY: arena.offset < budget, within the pool VMM.
        unsafe { self.base.add(self.arena.offset(hot, idx)) }
    }

    /// The slot's HOST address — the io_uring O_DIRECT destination (`ReadSpec.dst`).
    ///
    /// Same offset arithmetic as [`RoutedPool::ptr`], different base. The arena's slot
    /// stride and the pool base are both `crate::fetch::stream::ALIGN`-aligned (checked
    /// in `VmmBuf::new` and by the budget check in [`RoutedPool::new`]), so every
    /// result satisfies the O_DIRECT alignment the streamer asserts.
    fn host_ptr(&self, hot: bool, idx: usize) -> *mut u8 {
        // SAFETY: arena.offset < budget, within the pool VMM's host mapping.
        unsafe { self.host_base.add(self.arena.offset(hot, idx)) }
    }

    /// Record that `key`'s bytes have landed in its current slot. Called once per MISS,
    /// after that read's signal resolves.
    fn mark_loaded(&mut self, key: u32) {
        self.loaded.insert(key);
    }

    /// Has `key`'s data actually landed since it was last admitted? A HIT on a key for
    /// which this is false is a read of uninitialised or stale bytes — the fault this
    /// instrumentation exists to catch.
    fn is_loaded(&self, key: u32) -> bool {
        self.loaded.contains(&key)
    }

    fn slot(&self, key: u32) -> Option<(bool, usize)> {
        self.slot_of.get(&key).copied()
    }

    /// Admit a MISS: the policy evicts (by its own rule) until the incoming slot's bytes
    /// fit; free each victim's slot, then place the new key — compacting the arena (one
    /// device memcpy per relocation) as needed. Records the key's final slot.
    fn alloc(&mut self, key: u32) -> Result<()> {
        let adm = self.policy.admit(key);
        for ev in adm.evicted {
            let s = self
                .slot_of
                .remove(&ev)
                .context("evicted key had no slot")?;
            self.key_at.remove(&s);
            // Evicted: its bytes are no longer this key's, and the slot is about to be
            // handed to someone else.
            self.loaded.remove(&ev);
            self.arena.free(s.0, s.1);
        }
        let hot = adm.tier == cache::Tier::Hot;
        let idx = loop {
            match self.arena.alloc_step(hot) {
                AllocOutcome::Placed(idx) => break idx,
                AllocOutcome::Relocated(r) => self.relocate(r)?,
                AllocOutcome::NeedFree => {
                    bail!("arena NeedFree after policy eviction — byte-accounting bug")
                }
            }
        };
        self.slot_of.insert(key, (hot, idx));
        self.key_at.insert((hot, idx), key);
        // Freshly admitted: a slot with no bytes in it yet. `mark_loaded` clears this
        // once the read lands. A HIT observed while a key is in this state is the bug.
        self.loaded.remove(&key);
        // POISON the slot before its bytes land, so a read-before-write is
        // deterministic.
        //
        // Without this an unloaded slot holds whatever was there: uninitialised memory
        // (-> NaN, seen in ~6% of long runs) or the evicted expert's weights (-> finite,
        // plausible, SILENTLY wrong). 0x7FC0_7FC0 is a quiet NaN in f32 and in both
        // bf16 halves, so every format's scales read back non-finite and both cases
        // collapse into the loud one — which the per-layer localiser then pins to a
        // (pos, layer).
        //
        // Costs a ~20 MB device fill per miss (~3% of wall at 148 misses/token), which
        // is why it is `trace`-only. It is a diagnostic, not a safety net.
        //
        // It is WEAKER on `.f4` than on the other two and the reason is arithmetic:
        // 0x7FC0_7FC0 as e2m1 nibbles is an ordinary weight pattern and as e8m0 bytes
        // it is `0x7f`/`0xc0` — finite scales of 2^0 and 2^65. So an unloaded FP4 slot
        // decodes to large-but-finite garbage rather than to NaN, and only the
        // `READ-BEFORE-WRITE` report below catches it. Not fixed by a different
        // pattern: no 32-bit word is simultaneously non-finite as f32, as two bf16
        // halves AND as four e8m0 bytes, because every e8m0 byte but 0xff is finite.
        #[cfg(feature = "trace")]
        {
            let dst = self.ptr(hot, idx);
            // SAFETY: `dst` owns `stride` bytes in the pool VMM; the slot is not yet
            // handed to any kernel (that happens in the resolve phase, after this
            // returns).
            unsafe { rivoli_backend::fill_u32(dst, 0x7FC0_7FC0, self.geom.stride)? };
        }
        Ok(())
    }

    /// Execute one compaction relocation: memcpy the slot's bytes `from`→`to` (distinct,
    /// non-overlapping slots) and remap the key that lived there. Synchronous, so it
    /// lands before the layer's compute or any later cold read touches the new slot.
    fn relocate(&mut self, r: Reloc) -> Result<()> {
        let moved = self
            .key_at
            .remove(&(r.hot, r.from))
            .context("relocated slot had no key")?;
        let src = self.ptr(r.hot, r.from) as *const u8;
        let dst = self.ptr(r.hot, r.to);
        // SAFETY: distinct slots (non-overlapping), each `stride` bytes within the VMM.
        unsafe { memcpy_dtod(dst, src, self.geom.stride)? };
        self.slot_of.insert(moved, (r.hot, r.to));
        self.key_at.insert((r.hot, r.to), moved);
        // The relocation copies the bytes with the key, so `moved` stays loaded.
        // Nothing else changes state: the source slot is now free and holds no key.
        Ok(())
    }

    /// Enqueue the device-side wait for `t` on `stream_raw`. The ONLY way to consume a
    /// ticket — so a launch cannot happen without its dependency.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn wait_on(&self, t: Ticket, stream_raw: *mut std::ffi::c_void) -> Result<()> {
        self.fetch.wait(t, stream_raw)
    }

    /// Device bytes this pool may hold. Read by the startup log and by pool tests,
    /// which need it to ASSERT that their eviction case is one — a test whose premise
    /// ("the working set exceeds the budget") is assumed rather than checked passes
    /// silently the day a fixture grows.
    pub fn budget(&self) -> usize {
        self.arena.budget()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Accumulated reaper fetch wall (ns) — the off-main-thread load cost the expert
    /// stream's compute overlaps. The profile reads it against the MoE wall.
    pub fn fetch_ns(&self) -> u64 {
        self.fetch.fetch_ns()
    }

    /// Accumulated ns the reaper spent blocked in `io_uring` completions — the measured
    /// io-wait, taken at the ring rather than inferred from phase subtraction.
    pub fn io_wait_ns(&self) -> u64 {
        self.fetch.io_wait_ns()
    }

    /// Times a layer had to WAIT for a staging slot whose bounce copy had not retired.
    /// Should stay 0: a layer uses ~2 of 16 slots and a copy retires in ~1.2 ms against
    /// a ~3.5 ms layer. Non-zero means the ring is undersized for the lookahead —
    /// surfaced rather than merely counted, because a counter nobody reads is how the
    /// last two dead fields in the old engine got there.
    pub fn slot_stalls(&self) -> u64 {
        self.fetch.slot_stalls()
    }

    /// Is `(layer, expert)` resident? Deliberately routed through
    /// [`HybridPolicy::contains`], which takes `&self` and does NOT refresh recency —
    /// `get` would count the whole candidate window as an access and corrupt the
    /// eviction clock, which is the failure mode that would make cache-aware routing
    /// look like it works while destroying the cache underneath it.
    pub fn resident(&self, layer: usize, expert: usize) -> bool {
        self.policy.contains(expert_key(layer, expert))
    }

    /// Submit one layer's cold reads and resolve each selected expert to its
    /// [`ExpertSlot`] (device pointers into the pool) and its [`Ticket`] — the
    /// DEVICE-SIDE dependency its data is behind.
    ///
    /// Trace sink, then three phases over the arena pool, each a named method below:
    /// [`Self::touch_hits`] protects every HIT so a same-batch miss can't evict it;
    /// [`Self::admit_misses`] allocates every MISS — this is where the byte-aware
    /// policy evicts and the arena may RELOCATE resident slots; [`Self::resolve`] runs
    /// only after all relocations have settled, resolving each key's final slot and
    /// building the misses' cold reads — so a read never targets a slot that later
    /// moves.
    ///
    /// **There is no residency mask, and its absence is the point.** The old pool also
    /// returned `hit: Vec<bool>`, a second host-side encoding of "is this expert's data
    /// ready?" that the loop consumed to decide whether to await. When the two
    /// disagreed the bool won silently — a `hit` expert launched with no wait at all —
    /// so a slot still being written could be marked ready and the kernel would read
    /// it. A ticket cannot disagree with anything: it IS the dependency, and the only
    /// way to launch is to enqueue its wait ([`RoutedPool::wait_on`]). Resident experts
    /// carry [`Ticket::RESIDENT`], so resident / missing / in-flight are one code path.
    pub fn submit(
        &mut self,
        sel: Selection<'_>,
        win: RankWindow<'_>,
        out: &mut ResolvedBatch,
    ) -> Result<()> {
        out.slots.clear();
        out.tickets.clear();
        // **Range-check the whole selection BEFORE touching anything.** `spec` used to
        // be called in the resolve phase, after admission had already `admit`ed each
        // miss into the policy, placed an arena slot, bumped `misses` and (under
        // `trace`) poison-filled the slot. An out-of-range layer therefore returned
        // `Err` with the pool MUTATED: `resident()` then answered true for a key no
        // read ever targeted, and a second `submit` of it took the HIT path — returning
        // `Ok` with an `ExpertSlot` pointing at poison, or at the previous tenant's
        // weights. That is exactly the silent-wrong-bytes case the ticket protocol
        // exists to make impossible, reintroduced through the error path. Found by
        // review, 2026-08-05; the old `tests/f4_pool.rs` performs the first half of it
        // deliberately and asserts the pool is unchanged afterwards.
        for &e in sel.experts {
            self.geom.addressable(sel.layer, e)?;
        }
        // A real `ensure!`, not a `debug_assert!`: `is_hit` below is a fixed
        // `[bool; MAX_BATCH]` and an over-long `sel` would INDEX PAST IT, which under
        // `--release` (where `debug_assert!` is compiled out, CLAUDE.md) is a panic
        // mid-decode rather than an error. The pin checks the same bound at STARTUP
        // against the config numbers, which is the message a user should see; this is
        // the one that makes the array access total, and it costs one compare per
        // layer.
        ensure!(
            sel.experts.len() <= MAX_BATCH,
            "submit: {} experts exceeds the {MAX_BATCH}-slot batch scratch",
            sel.experts.len()
        );
        // New batch: the previous layer's reads have all landed (its awaits + its
        // end-of-layer sync), so its misses become `loaded` now; then clear the
        // policy's per-batch pin set — `touch_hits`'s protect() and `admit_misses`'s
        // admit() pin every touched key so a later miss's eviction can't reclaim it.
        {
            let done = std::mem::take(&mut self.pending_loaded);
            for k in done {
                self.mark_loaded(k);
            }
        }
        self.policy.begin_batch();
        self.write_trace(sel, win)?;
        let mut is_hit = [false; MAX_BATCH];
        self.touch_hits(sel, &mut is_hit);
        self.admit_misses(sel, &is_hit)?;
        self.resolve(sel, &is_hit, out)
    }

    /// Touch every hit first, so a later miss's admit can't evict it. The physical slot
    /// is deliberately NOT read here — [`Self::resolve`] takes it from `slot_of` after
    /// any same-batch relocation settles.
    fn touch_hits(&mut self, sel: Selection<'_>, is_hit: &mut [bool; MAX_BATCH]) {
        for (i, &e) in sel.experts.iter().enumerate() {
            let key = expert_key(sel.layer, e);
            if self.policy.hit(key) {
                self.hits += 1;
                // THE CHECK. A hit hands the kernel a slot pointer and a resolved
                // RESIDENT ticket, so nothing downstream waits. If the bytes never
                // landed, the kernel reads uninitialised memory (NaN) or another
                // expert's weights (finite, wrong, silent). Reported once per
                // occurrence rather than fataling, so a run keeps going and the
                // pattern is visible.
                if !self.is_loaded(key) {
                    tracing::error!(
                        "READ-BEFORE-WRITE: layer={} expert={e} counted as a cache HIT \
                         but its bytes never landed since admission. The kernel is \
                         about to read an unloaded slot — uninitialised memory (-> \
                         NaN) or a previous expert's weights (-> silently wrong).",
                        sel.layer
                    );
                }
                self.policy.protect(key);
                is_hit[i] = true;
            }
        }
    }

    /// Allocate the misses (evict + place + compact). Slots may relocate.
    fn admit_misses(&mut self, sel: Selection<'_>, is_hit: &[bool; MAX_BATCH]) -> Result<()> {
        let mut any_miss = false;
        for (i, &e) in sel.experts.iter().enumerate() {
            if !is_hit[i] {
                self.misses += 1;
                self.alloc(expert_key(sel.layer, e))?;
                any_miss = true;
            }
        }
        // The poison fills in `alloc` run on the DEFAULT stream; the reaper's
        // bounce->slot copies run on a `hipStreamNonBlocking` fetch stream, which does
        // not synchronise with it. Unordered, a fill could land AFTER the read and
        // destroy good data — the diagnostic would then cause the corruption it exists
        // to detect. One join per layer-with-misses orders them.
        //
        // CAVEAT, and it is the same trap the old `--checksum-x` fell into: this sync
        // may itself perturb the race being hunted. It sits at a different point (after
        // allocation, before reads are submitted) than the per-layer D2H that masked
        // the fault, so it is not the same barrier — but a clean run under poisoning is
        // NOT proof the bug is absent. Only a poison HIT is positive evidence.
        #[cfg(feature = "trace")]
        if any_miss {
            rivoli_backend::device_sync()?;
        }
        #[cfg(not(feature = "trace"))]
        let _ = any_miss; // the join is the diagnostic's ordering; without it, unused
        Ok(())
    }

    /// Relocations have settled — resolve each key's final slot into `out` and hand the
    /// misses' cold reads to the reaper, which queues+submits (all reads start at once)
    /// and signals each miss's ticket when its copy lands.
    fn resolve(
        &mut self,
        sel: Selection<'_>,
        is_hit: &[bool; MAX_BATCH],
        out: &mut ResolvedBatch,
    ) -> Result<()> {
        let mut reads: Vec<ReadSpec> = Vec::new();
        let mut miss_sel: Vec<usize> = Vec::new();
        for (i, &e) in sel.experts.iter().enumerate() {
            let (hot, idx) = self.slot(expert_key(sel.layer, e)).context(
                "expert not resident after alloc (batch exceeds pool — raise --max-mem)",
            )?;
            let b = self.ptr(hot, idx);
            // SAFETY: address arithmetic into the resolved slot. The offsets are inside
            // it by construction, not by check: `ExpertSet::open_routed` derives both
            // the slot stride and the six offsets from ONE `RoutedFmt`, so they cannot
            // describe different formats. The bytes land when `tickets[i]` is
            // satisfied.
            out.slots.push(unsafe { slot_at(b, &self.geom.off) });
            if !is_hit[i] {
                let (fd, begin, len) = self.geom.spec(sel.layer, e)?;
                reads.push(ReadSpec {
                    fd,
                    begin,
                    len,
                    dst: self.host_ptr(hot, idx),
                });
                miss_sel.push(i);
            }
        }
        // Queue this batch's misses to be marked loaded at the next layer.
        for &i in &miss_sel {
            self.pending_loaded
                .push(expert_key(sel.layer, sel.experts[i]));
        }
        let miss_tickets = self.fetch.submit(reads)?;
        // A resident expert's data is already there, so it carries the RESIDENT ticket
        // (value 0, satisfied on arrival). Every expert therefore has a ticket and the
        // caller has one code path — there is no residency bool for anyone to branch
        // on.
        out.tickets.resize(sel.experts.len(), Ticket::RESIDENT);
        for (k, &i) in miss_sel.iter().enumerate() {
            out.tickets[i] = miss_tickets[k];
        }
        Ok(())
    }
}

#[cfg(test)]
mod ticket_tests {
    use super::Ticket;

    /// **INV-5: an expert cannot be launched without enqueueing its data dependency.**
    ///
    /// The structural half of this is enforced by types and cannot be tested at
    /// runtime: [`RoutedPool::submit`] fills [`ResolvedBatch`] and returns no residency
    /// mask, so the loop has nothing to branch on and no way to spell "launch without
    /// waiting". What IS testable, and what actually broke before, is the encoding — a
    /// resident ticket must be a real satisfied dependency rather than a sentinel the
    /// consumer has to recognise and skip.
    ///
    /// Timelines start at 0, so `RESIDENT.value == 0` means "wait on 0", which every
    /// timeline satisfies on arrival. If it were, say, `u64::MAX` as an "N/A" marker,
    /// the consumer would need a branch to avoid deadlocking on it — and that branch is
    /// exactly the `hit` mask growing back.
    #[test]
    fn inv_5_every_descriptor_carries_a_ticket() {
        assert!(
            Ticket::RESIDENT.is_resident(),
            "the resident ticket must read as resident"
        );
        assert_eq!(
            Ticket::RESIDENT.value,
            0,
            "RESIDENT must be value 0 — a timeline starts there, so waiting on it is \
             satisfied immediately. A sentinel that had to be SKIPPED would put a \
             residency branch back in the consumer, which is the bug class this removed."
        );
        // A real fetch ticket is never confusable with a resident one: values are
        // assigned from 1 upward per slot.
        let fetched = Ticket { slot: 3, value: 1 };
        assert!(
            !fetched.is_resident(),
            "the first value a slot hands out must NOT read as already-satisfied"
        );
    }
}
