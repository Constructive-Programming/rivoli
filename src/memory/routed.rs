//! The routed-expert streaming pool: residency, eviction, relocation and the io_uring
//! cold reads, for every routed format ([`RoutedFmt`]).
//!
//! This was private to [`crate::memory::pin::Pin`] until DeepSeek-V4-Flash needed one too,
//! and the split is not a tidy-up. V4's routed set is **137 GiB of `.f4`** against the
//! ~115 GiB budget this machine runs at, so a V4 decode is blocked on a pool in a way GLM's
//! is not — and the two pins are deliberately separate types
//! (`pin.rs`'s "a single pin parameterised by an arch flag would be a GLM-shaped placement
//! path one `if` away from running on a V4 artifact"). The pool is the one part they must
//! share: it is ~300 lines of eviction/relocation/ticket protocol carrying INV-5 and three
//! diagnostics, and `build.rs`'s duplication gate would refuse a second copy — correctly,
//! because a second copy is a second place for the read-before-write rule to be wrong.
//!
//! **What made the split cheap is that the substrate below it was already byte-parameterised
//! and this was verified rather than assumed**: [`Arena`] takes two `usize` strides and
//! never names a format, [`HybridPolicy`] and [`cache`] account in bytes,
//! [`AsyncFetch`]/[`ReadSpec`]/[`Streamer`] move `(fd, begin, len) -> dst` spans, and
//! `ExpertSet::{open_routed, read_spec, expert_slot}` reads its geometry off `RoutedFmt`.
//! The format-dependent parts were exactly three, all of them here: the six intra-block
//! projection offsets, the per-slot stride, and which kernel decodes the result.
//!
//! What this does NOT own: which experts to fetch. Routing never consults residency
//! (INV-1), so the pool is told a selection and reports where the bytes are.

use crate::artifact::format::{ExpertSet, RoutedFmt};
use crate::backend::memcpy_dtod;
use crate::fetch::asyncfetch::{AsyncFetch, ReadSpec, Ticket};
use crate::fetch::stream::{Streamer, slot_span};
use crate::memory::arena::{AllocOutcome, Arena, Reloc};
use crate::memory::cache;
use crate::memory::device::VmmBuf;
use crate::memory::hybrid::HybridPolicy;
use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::os::fd::RawFd;

/// Width of the trace-v2 candidate window: the top-W router candidates recorded per
/// routing decision, on top of the `top_k` that actually ran. W bounds the largest M
/// the offline (J, M) substitution grid in docs/investigations/cache-conditional-routing.md can explore — an M
/// wider than this cannot be evaluated from a captured trace without recapturing.
/// 32 is 4× `top_k` (8) and an eighth of `n_experts` (256): far past any M where
/// promoting a resident-but-lower-ranked expert is still defensible, and only ~380
/// bytes a line.
pub const TRACE_WINDOW: usize = 32;

/// The largest selection one [`RoutedPool::submit`] call may carry.
///
/// It sizes exactly one thing: `submit`'s own `[bool; MAX_BATCH]` hit scratch, which is why
/// `submit` `ensure!`s against it rather than trusting a caller. `gpu.rs` sizes its
/// descriptor buffer from a RUNTIME `top_k · MAXROW + n_shared` (18 for GLM), not from this;
/// `Pin::build` checks that value against this one at startup so the friendly message
/// arrives before the run rather than during it. V4's demand is `top_k` — its FP4 kernel
/// refuses `nrow != 1` (`kernels/moe.hip:409`), so there is no batched union.
pub const MAX_BATCH: usize = 32;

/// One projection's two addresses inside an expert's pool slot.
///
/// **Both are `*const u8` and that is the point.** This carried a `*const u16` `scales`
/// while there were two formats, which was already a half-truth (`.i4`'s scales are f32,
/// "reinterpreted at the launch site") and becomes a wrong one at `.f4`, whose e8m0 scales
/// are ONE byte. A slot is six byte addresses; what they mean is the descriptor's business,
/// and `backend::ExpertDesc` vs `backend::ExpertDescF4` is where that is said once.
#[derive(Clone, Copy)]
pub struct ProjSlot {
    pub packed: *const u8,
    pub scale: *const u8,
}

/// One expert's three projections resolved to device addresses — what a launch descriptor
/// is built from. Field order is slot order everywhere in this engine: gate, up, down
/// (`quant::PROJ`, and `quant::V4_PROJ` for why V4's `w1, w3, w2` is that same order).
#[derive(Clone, Copy)]
pub struct ExpertSlot {
    pub gate: ProjSlot,
    pub up: ProjSlot,
    pub down: ProjSlot,
}

/// Resolve one projection's two pointers at slot-relative offsets `(poff, soff)` from an
/// expert-block base — the single builder shared by resident shared experts and the
/// streamed routed ones.
///
/// # Safety
/// Both offsets must lie within the expert block at `base`.
#[inline]
unsafe fn proj_at(base: *const u8, poff: usize, soff: usize) -> ProjSlot {
    // SAFETY: forwarded from this function's own contract.
    unsafe {
        ProjSlot {
            packed: base.add(poff),
            scale: base.add(soff),
        }
    }
}

/// Resolve all six offsets against one block base.
///
/// # Safety
/// Every offset in `off` must lie within the expert block at `base`.
#[inline]
pub unsafe fn slot_at(base: *const u8, off: &[usize; 6]) -> ExpertSlot {
    // SAFETY: forwarded from this function's own contract.
    unsafe {
        ExpertSlot {
            gate: proj_at(base, off[0], off[1]),
            up: proj_at(base, off[2], off[3]),
            down: proj_at(base, off[4], off[5]),
        }
    }
}

/// Pack `(layer, expert)` into the pool key. Both must fit in 16 bits — GLM is
/// ≤92 layers × 256 routed experts and V4 is 43 × 256, comfortably under 2^16.
pub fn expert_key(layer: usize, expert: usize) -> u32 {
    debug_assert!(
        layer < (1 << 16) && expert < (1 << 16),
        "layer {layer}/expert {expert} exceed the 16-bit pool key packing"
    );
    ((layer as u32) << 16) | expert as u32
}

/// One arena tier's format: the projection offsets, which format decodes it, the slot
/// stride, and the per-`(layer,expert)` O_DIRECT read-spec table. COLD/HOT are the SAME
/// format in single-format modes (uniform stride; any compaction is a cheap same-size move)
/// or int3-VQ vs int4 (hybrid). `.f4` is single-format: there is no second FP4 container to
/// pair it with.
#[derive(Clone)]
pub struct TierFmt {
    off: [usize; 6],
    fmt: RoutedFmt,
    stride: usize,
    /// `(fd, begin, len)` per `(layer - first_layer) * n_experts + expert`.
    table: Vec<(RawFd, usize, usize)>,
    /// The row basis of `table`, kept WITH the table rather than beside it in
    /// [`RoutedPool`]: `(layer - first_layer) * n_experts + expert` indexed with a
    /// `first_layer` from somewhere else reads a different layer's expert and fails no
    /// check. [`TierFmt::spec`] is the only reader, so there is no second copy to disagree
    /// — which is why the pool has no guard comparing its two tiers' bases, and why it
    /// should not grow one.
    first_layer: usize,
    n_experts: usize,
}

impl TierFmt {
    /// Tabulate one set's read specs, taking EVERYTHING from the set: the format, the six
    /// projection offsets, the slot stride, the layer range and the expert count.
    ///
    /// **One argument, and that is the design.** This took `fmt` and `off` and a `layers`
    /// range, with an `ensure!` that the offsets were ascending and inside the stride. That
    /// guard was written, and then asked what would have to be true for it to fire: nothing
    /// realistic. Every routed block is padded up to `VQ_ALIGN`, so `.vq3`'s layout on an
    /// `.f4` slot (9,961,472 against a 13,369,344 stride) sits comfortably inside it and
    /// passes — and `.f4` and `.i4` tile identically for 25% of all `i_dim`, both models'
    /// dimensions included (`quant::f4_slot_offsets` has the identity), so the pairing that
    /// actually costs correctness is invisible to any check at all. A guard that cannot fire
    /// is worse than none; the fix is that the set knows its own format, so there is nothing
    /// left to pair wrongly.
    pub fn new(src: &ExpertSet) -> Result<Self> {
        let layers = src.layers();
        let n_experts = src.n_experts();
        let first_layer = layers.start;
        let mut table = Vec::with_capacity(layers.len() * n_experts);
        for l in layers {
            for e in 0..n_experts {
                table.push(src.read_spec(l, e)?);
            }
        }
        Ok(Self {
            off: src.slot_offsets(),
            fmt: src.fmt(),
            stride: src.expert_slot(),
            table,
            first_layer,
            n_experts,
        })
    }

    /// This tier's cold-read spec for `(layer, expert)`, by ABSOLUTE layer id.
    ///
    /// The `layer - first_layer` subtraction lives here, with the table it indexes, and that
    /// is the point: [`RoutedPool`] used to hold one copy of the basis and apply it to both
    /// tiers, which needed an `ensure!` that the two agreed — a guard with no way to fire,
    /// since every mode builds both tiers from one `SetDims`. Indexing each table with its
    /// own basis removes the disagreement instead of checking for it.
    fn spec(&self, layer: usize, expert: usize) -> Result<(RawFd, usize, usize)> {
        self.row(layer)?;
        self.table
            .get(self.row(layer)? * self.n_experts + expert)
            .copied()
            .context("unreachable: `row` bounds both indices")
    }

    /// `layer`'s row in [`Self::table`], both ends checked.
    ///
    /// **`expert` is bounded here too, and a `table.get()` alone would not do it.** The index
    /// is `row * n_experts + expert`, so on any row but the LAST an `expert >= n_experts`
    /// lands inside the table on a later layer's row and comes back `Ok` with that layer's
    /// fd and offset — a silently wrong cold read, not an error. Concretely on the 3-layer
    /// `.f4` fixture, `(0, 256)` would return layer 1 expert 0. `ExpertSet::read_spec` bounds
    /// `expert`, but that ran at table-BUILD time; nothing re-checked it at lookup.
    ///
    /// Split out from [`Self::spec`] so [`RoutedPool::submit`] can run it BEFORE it mutates
    /// anything — see the range check at the top of `submit` for why that ordering is not a
    /// style preference.
    fn row(&self, layer: usize) -> Result<usize> {
        let rows = self.table.len() / self.n_experts;
        let row = layer.checked_sub(self.first_layer).filter(|&r| r < rows);
        row.with_context(|| {
            format!(
                "layer {layer} is outside a .{} tier over {rows} layers from layer {}",
                self.fmt.ext(),
                self.first_layer,
            )
        })
    }

    /// Is `(layer, expert)` addressable in this tier? The pre-flight half of [`Self::spec`].
    fn addressable(&self, layer: usize, expert: usize) -> Result<()> {
        self.row(layer)?;
        ensure!(
            expert < self.n_experts,
            "expert {expert} >= {} in a .{} tier",
            self.n_experts,
            self.fmt.ext(),
        );
        Ok(())
    }
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
    /// The HOST base: the io_uring O_DIRECT DMA target (`ReadSpec.dst`), and the only one
    /// of the two the CPU may touch.
    ///
    /// Under HIP these are the SAME NUMBER — unified addressing — so this field costs
    /// nothing there and changes no behaviour. It existed because the retired Vulkan backend
    /// made the two unrelated numbers (2026-08-06); resolving both once here still keeps
    /// [`RoutedPool::ptr`] and [`RoutedPool::host_ptr`] a single `add` each on the fetch
    /// path, and keeps the two spellings from collapsing into one. See
    /// [`crate::memory::device::VmmBuf::host_mut`] — under HIP the coincidence holds, and
    /// the split is a naming CONVENTION now, not something the type system checks.
    host_base: *mut u8,
    arena: Arena,
    policy: Box<dyn HybridPolicy>,
    slot_of: HashMap<u32, (bool, usize)>, // key -> (hot, idx)
    key_at: HashMap<(bool, usize), u32>,  // (hot, idx) -> key, for relocation remap
    /// Keys whose bytes are known to have LANDED in their current slot.
    ///
    /// The engine has no other way to distinguish "the policy says resident" from "the
    /// bytes are actually there", and that distinction is the leading hypothesis for the
    /// intermittent non-finite-logits bug: a HIT carries `Ticket::RESIDENT` and the kernel
    /// reads the slot immediately, so if a key is ever counted resident before its load
    /// completed, the read is of uninitialised (-> NaN, the visible case) or stale (->
    /// finite and WRONG, the silent case) memory.
    ///
    /// A key is removed on eviction and on relocation-into, and inserted only when its
    /// read signal has resolved. `trace` only — it costs a hash op per expert per layer.
    loaded: std::collections::HashSet<u32>,
    /// Misses submitted by the PREVIOUS layer, marked loaded at the top of the next
    /// `submit`. Correct because layer L's per-expert awaits and its unconditional
    /// end-of-layer `device_sync` both complete before layer L+1 submits — so by then
    /// every byte of L's batch has landed. Deferring this way avoids plumbing a
    /// completion callback through `gpu.rs`'s async expert loop.
    pending_loaded: Vec<u32>,
    cold: TierFmt,
    hot: TierFmt,
    /// Per-expert async cold-fetch: owns the io_uring demand ring on a reaper thread
    /// and signals each miss's [`Ticket`] when its bytes land. The expert
    /// stream awaits these; there is no batch join.
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
/// (`budget − (idx+1)·hot_stride`), so an unaligned `budget` makes every hot-slot DMA
/// destination violate the alignment `stream.rs` asserts — the base and the strides are
/// already aligned, so the budget is the only way in. It costs <4 KiB. One function because
/// both pins compute it and a rounding that differed between them would misalign one pool's
/// hot-slot DMA destinations and not the other's — a failure that would look like a backend
/// bug in exactly one architecture.
///
/// `saturating_sub` because a `--max-mem` below the resident footprint is a user error with
/// a number attached; it lands as "budget cannot hold one layer" in [`RoutedPool::new`]
/// rather than as a 16-exabyte wrap.
pub fn pool_budget(capacity: usize, tier_cap: usize) -> usize {
    capacity.saturating_sub(tier_cap) & !(crate::fetch::stream::ALIGN - 1)
}

impl RoutedPool {
    /// Build the pool over [`pool_budget`] device bytes. `cold`/`hot` are the same
    /// [`TierFmt`] in single-format modes and int3-VQ/int4 in GLM's hybrid. `top_k` is one
    /// layer's demand, which sizes the io_uring ring.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        budget: usize,
        top_k: usize,
        max_batch: usize,
        policy_name: &str,
        two_q: cache::TwoQSplit,
        trace_path: Option<&str>,
        cold: TierFmt,
        hot: TierFmt,
    ) -> Result<Self> {
        let (cold_stride, hot_stride) = (cold.stride, hot.stride);
        // A pool that cannot hold ONE BATCH cannot make progress: every key in a batch is
        // pinned, so `evict_until_fits` finds nothing to reclaim, `alloc_step` returns
        // `NeedFree` and `alloc` bails with "arena NeedFree after policy eviction —
        // byte-accounting bug". That message accuses the arena of a defect the user's
        // `--max-mem` caused. Refused here instead, at startup, with both numbers.
        //
        // **`max_batch`, not `top_k`, and the two genuinely differ.** GLM submits the UNION
        // of `MAXROW` rows' picks (`gpu.rs`), so its batch is `top_k · MAXROW + n_shared`;
        // V4's FP4 kernel refuses `nrow != 1` so its batch is `top_k`. Sizing this from
        // `top_k` alone left GLM budgets between 8 and 16 slots passing startup and failing
        // mid-run with the arena message — the exact case this check was added to replace.
        // `top_k` still sizes the io_uring ring, which is a per-layer demand count.
        let one_batch = max_batch * cold_stride.max(hot_stride);
        ensure!(
            budget >= one_batch,
            "routed pool budget {:.2} GiB cannot hold one batch of {max_batch} experts \
             ({:.2} GiB) — raise --max-mem",
            budget as f64 / (1u64 << 30) as f64,
            one_batch as f64 / (1u64 << 30) as f64,
        );
        let policy =
            crate::memory::hybrid::policy_for(policy_name, budget, cold_stride, hot_stride, two_q)
                .with_context(|| format!("unknown --cache-policy {policy_name} (lru|2q|arc)"))?;
        // BOTH tiers named. This printed `Mode` before the split and the naive replacement
        // was `cold.fmt.ext()` alone — which prints `[2q vq3]` for the DEFAULT hybrid mode,
        // naming half the pool and asserting a single-format one. Every benchmark log in
        // docs/measurement/benchmarks.md is keyed on this line, so a mode that reads as
        // another mode is a measurement hazard, not a cosmetic one.
        let fmts = if cold.fmt == hot.fmt {
            cold.fmt.ext().to_string()
        } else {
            format!("{}+{}", cold.fmt.ext(), hot.fmt.ext())
        };
        tracing::info!(
            "routed pool [{policy_name} {fmts}]: {:.1} GiB budget (~{} slots, cold {cold_stride}B / hot {hot_stride}B)",
            budget as f64 / (1u64 << 30) as f64,
            budget / cold_stride.min(hot_stride),
        );
        let mut buf = VmmBuf::new(budget)?;
        let base = buf.ptr_mut();
        // Both bases resolved ONCE, here. Under HIP `host_mut` and `ptr_mut` return the same
        // number, so this is a no-op — pinned by
        // `device.rs::vmmbuf_host_and_device_bases_coincide_under_hip`. The two spellings are
        // kept so the two consumers below cannot bake that coincidence in; they were genuinely
        // different numbers under the Vulkan backend, retired 2026-08-06.
        let host_base = buf.host_mut();
        // Ring sized for one layer's worst case: one demand read per expert, one aligned
        // block each, in every format.
        let ring = (top_k + 4).next_power_of_two();
        // Bounce span = the largest expert block across the tiers (one read).
        let span = slot_span(cold_stride.max(hot_stride));
        let fetch = AsyncFetch::new(Streamer::new(ring as u32, span)?)?;
        Ok(Self {
            buf,
            base,
            host_base,
            arena: Arena::new(budget, cold_stride, hot_stride),
            policy,
            slot_of: HashMap::new(),
            key_at: HashMap::new(),
            loaded: std::collections::HashSet::new(),
            pending_loaded: Vec::new(),
            cold,
            hot,
            fetch,
            hits: 0,
            misses: 0,
            trace: trace_path
                .map(|p| -> Result<_> {
                    use std::io::Write;
                    let mut w = std::io::BufWriter::new(
                        std::fs::File::create(p).with_context(|| format!("open trace {p}"))?,
                    );
                    // Version header. It is deliberately unparseable as data: `replay`
                    // reads each line for whitespace-separated u32s and drops the empty
                    // ones, so this line contributes nothing and a v2 trace replays
                    // through a v1 reader byte-identically.
                    writeln!(w, "# rivoli-trace v2 top_k={top_k} window={TRACE_WINDOW}")
                        .context("write trace")?;
                    Ok(w)
                })
                .transpose()?,
        })
    }

    fn tier(&self, hot: bool) -> &TierFmt {
        if hot { &self.hot } else { &self.cold }
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
    /// strides and the pool base are both `crate::fetch::stream::ALIGN`-aligned (checked in
    /// `VmmBuf::new` and by the budget check in [`RoutedPool::new`]), so every result
    /// satisfies the O_DIRECT alignment the streamer asserts.
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
        // POISON the slot before its bytes land, so a read-before-write is deterministic.
        //
        // Without this an unloaded slot holds whatever was there: uninitialised memory
        // (-> NaN, seen in ~6% of long runs) or the evicted expert's weights (-> finite,
        // plausible, SILENTLY wrong). 0x7FC0_7FC0 is a quiet NaN in f32 and in both bf16
        // halves, so every format's scales read back non-finite and both cases collapse
        // into the loud one — which the per-layer localiser then pins to a (pos, layer).
        //
        // Costs a ~20 MB device fill per miss (~3% of wall at 148 misses/token), which is
        // why it is `trace`-only. It is a diagnostic, not a safety net.
        //
        // It is WEAKER on `.f4` than on the other two and the reason is arithmetic:
        // 0x7FC0_7FC0 as e2m1 nibbles is an ordinary weight pattern and as e8m0 bytes it is
        // `0x7f`/`0xc0` — finite scales of 2^0 and 2^65. So an unloaded FP4 slot decodes to
        // large-but-finite garbage rather than to NaN, and only the `READ-BEFORE-WRITE`
        // report below catches it. Not fixed by a different pattern: no 32-bit word is
        // simultaneously non-finite as f32, as two bf16 halves AND as four e8m0 bytes,
        // because every e8m0 byte but 0xff is finite.
        #[cfg(feature = "trace")]
        {
            let stride = self.tier(hot).stride;
            let dst = self.ptr(hot, idx);
            // SAFETY: `dst` owns `stride` bytes in the pool VMM; the slot is not yet
            // handed to any kernel (that happens in phase 1c, after this returns).
            unsafe { crate::backend::fill_u32(dst, 0x7FC0_7FC0, stride)? };
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
        let stride = self.tier(r.hot).stride;
        let src = self.ptr(r.hot, r.from) as *const u8;
        let dst = self.ptr(r.hot, r.to);
        // SAFETY: distinct slots (non-overlapping), each `stride` bytes within the VMM.
        unsafe { memcpy_dtod(dst, src, stride)? };
        self.slot_of.insert(moved, (r.hot, r.to));
        self.key_at.insert((r.hot, r.to), moved);
        // The relocation copies the bytes with the key, so `moved` stays loaded. Nothing
        // else changes state: the source slot is now free and holds no key.
        Ok(())
    }

    /// Enqueue the device-side wait for `t` on `stream_raw`. The ONLY way to consume a
    /// ticket — so a launch cannot happen without its dependency.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn wait_on(&self, t: Ticket, stream_raw: *mut std::ffi::c_void) -> Result<()> {
        self.fetch.wait(t, stream_raw)
    }

    /// Device bytes this pool may hold. Read by the startup log and by
    /// `tests/f4_pool.rs`, which needs it to ASSERT that its eviction case is one — a test
    /// whose premise ("the working set exceeds the budget") is assumed rather than checked
    /// passes silently the day a fixture grows.
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
    /// Should stay 0: a layer uses ~2 of 16 slots and a copy retires in ~1.2 ms against a
    /// ~3.5 ms layer. Non-zero means the ring is undersized for the lookahead — surfaced
    /// rather than merely counted, because a counter nobody reads is how the last two dead
    /// fields in this engine got there.
    pub fn slot_stalls(&self) -> u64 {
        self.fetch.slot_stalls()
    }

    /// Is the `--trace` sink on? gpu.rs gates the candidate-window `topk_into` on this
    /// so a non-tracing decode pays literally nothing for trace v2.
    pub fn tracing(&self) -> bool {
        self.trace.is_some()
    }

    /// Is `(layer, expert)` resident? Deliberately routed through
    /// [`HybridPolicy::contains`], which takes `&self` and does NOT refresh recency —
    /// `get` would count the whole candidate window as an access and corrupt the
    /// eviction clock, which is the failure mode that would make `top-m` look like it
    /// works while destroying the cache underneath it.
    pub fn resident(&self, layer: usize, expert: usize) -> bool {
        self.policy.contains(expert_key(layer, expert))
    }

    /// Flush the trace sink. Called per token, because the trace CANNOT rely on
    /// `BufWriter`'s `Drop`: the wedge watchdog kills a hung decode with
    /// `std::process::exit`, which runs no destructors, and `Drop` discards flush errors
    /// anyway — so a wedged or ENOSPC run would leave a silently short capture with a
    /// clean exit code. A trace is ~30 minutes of sole-tenant GPU time; losing it quietly
    /// is far worse than one `write` per token. Errors propagate here, unlike in `Drop`.
    pub fn flush_trace(&mut self) -> Result<()> {
        if let Some(w) = &mut self.trace {
            use std::io::Write;
            w.flush().context("flush trace")?;
        }
        Ok(())
    }

    /// Submit one layer's cold reads and resolve each selected expert to its [`ExpertSlot`]
    /// (device pointers into the pool), its format, and its [`Ticket`] — the DEVICE-SIDE
    /// dependency its data is behind.
    ///
    /// Trace sink, then three phases over the arena pool. 1a: touch every HIT (protect it
    /// so a same-batch miss can't evict it). 1b: allocate every MISS — this is where the
    /// byte-aware policy evicts and the arena may RELOCATE resident slots. 1c: only NOW,
    /// after all relocations have settled, resolve each key's final slot into `out`/`fmt`
    /// and build the misses' cold reads — so a read never targets a slot that later moves.
    ///
    /// **There is no residency mask, and its absence is the point.** This used to also
    /// return `hit: Vec<bool>`, a second host-side encoding of "is this expert's data
    /// ready?" that `gpu.rs` consumed to decide whether to await. When the two disagreed the
    /// bool won silently — `gpu.rs` launches a `hit` expert with no wait at all — so a slot
    /// still being written could be marked ready and the kernel would read it. A ticket
    /// cannot disagree with anything: it IS the dependency, and the only way to launch is to
    /// enqueue its wait (`RoutedPool::wait_on`). Resident experts carry [`Ticket::RESIDENT`],
    /// so resident / missing / in-flight are one code path.
    ///
    /// `window`/`choice` feed the trace sink only: the ranked top-[`TRACE_WINDOW`]
    /// candidate expert ids and the full per-expert `choice` array they index into.
    /// Pass an empty `window` when not tracing — nothing else reads them.
    // ONE function, not the `submit_spine` + unwrap pair this was: the split's only artefact
    // was a `[Option<ResolvedSlot>; 32]` filled unconditionally over `sel` and then unwrapped
    // with a second `.context("unresolved expert slot")` that could never fire.
    // Seven arguments, all distinct runtime values on the per-layer hot path; bundling
    // them into a struct built once per layer would allocate to satisfy a lint.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        layer: usize,
        sel: &[usize],
        window: &[usize],
        choice: &[f32],
        out: &mut Vec<ExpertSlot>,
        fmt: &mut Vec<RoutedFmt>,
        tickets: &mut Vec<Ticket>,
    ) -> Result<()> {
        out.clear();
        fmt.clear();
        tickets.clear();
        // **Range-check the whole selection BEFORE touching anything.** `spec` used to be
        // called in phase 1c, after phase 1b had already `admit`ed each miss into the policy,
        // placed an arena slot, bumped `misses` and (under `trace`) poison-filled the slot.
        // An out-of-range layer therefore returned `Err` with the pool MUTATED: `resident()`
        // then answered true for a key no read ever targeted, and a second `submit` of it
        // took the phase-1a HIT path — returning `Ok` with an `ExpertSlot` pointing at
        // poison, or at the previous tenant's weights. That is exactly the silent-wrong-bytes
        // case the ticket protocol exists to make impossible, reintroduced through the error
        // path. Found by review, 2026-08-05; `tests/f4_pool.rs` performs the first half of it
        // deliberately and now asserts the pool is unchanged afterwards.
        //
        // Both tiers, because a miss may be admitted to either and `submit` cannot know which
        // until the policy has spoken — which is after the point of no return.
        for &e in sel {
            self.cold.addressable(layer, e)?;
            self.hot.addressable(layer, e)?;
        }
        // A real `ensure!`, not a `debug_assert!`: `is_hit` below is a fixed `[bool;
        // MAX_BATCH]` and an over-long `sel` would INDEX PAST IT, which under `--release`
        // (where `debug_assert!` is compiled out, CLAUDE.md) is a panic mid-decode rather
        // than an error. `Pin::build` checks the same bound at STARTUP against the config
        // numbers, which is the message a user should see; this is the one that makes the
        // array access total, and it costs one compare per layer.
        ensure!(
            sel.len() <= MAX_BATCH,
            "submit: {} experts exceeds the {MAX_BATCH}-slot batch scratch",
            sel.len()
        );
        // New batch: clear the policy's per-batch pin set. Phase 1a's protect() and 1b's
        // admit() then pin every touched key so a later miss's eviction can't reclaim it.
        // The previous layer's reads have all landed (its awaits + end-of-layer sync).
        {
            let done = std::mem::take(&mut self.pending_loaded);
            for k in done {
                self.mark_loaded(k);
            }
        }
        self.policy.begin_batch();
        // Trace sink (--trace), v2: the demand keys this layer looks up, then `|`, then
        // the top-`TRACE_WINDOW` candidates as `key:choice`.
        //
        // BOTH lists are in router RANK order, and that is LOAD-BEARING, not incidental.
        // `sel` and `window` both come out of `topk_into` over the same `choice` buffer
        // with the same comparator (value-desc, index-asc), and `topk_into` finishes with
        // a full sort — so `window[..sel.len()] == sel` element for element, and
        // `bin/replay` hard-fails a trace where that prefix does not hold. Reordering
        // `sel` for any local reason (coalescing reads by expert id, say) would silently
        // change the meaning of every captured trace. The debug_assert is the tripwire.
        debug_assert!(
            window.is_empty() || window.starts_with(sel),
            "trace v2: the candidate window must be the ranking that produced `sel`"
        );
        if let Some(w) = &mut self.trace {
            use std::io::Write;
            for (j, &e) in sel.iter().enumerate() {
                if j > 0 {
                    write!(w, " ").context("write trace")?;
                }
                write!(w, "{}", expert_key(layer, e)).context("write trace")?;
            }
            // ponytail: the `choice` values have no consumer yet — the (J, M) grid needs
            // only the RANK order, which the list already carries. Written anyway because
            // a capture is GPU-gated, sole-tenant and ~30 minutes, so these few bytes are
            // cheap now and unrecoverable later without another capture; and `route_kl`
            // (docs/investigations/cache-conditional-routing.md "Counters") is deferred, not cancelled, and needs the
            // mass distribution.
            write!(w, " |").context("write trace")?;
            for &e in window {
                write!(w, " {}:{:.6}", expert_key(layer, e), choice[e]).context("write trace")?;
            }
            writeln!(w).context("write trace")?;
        }
        // Phase 1a: touch every hit first, so a later miss's admit can't evict it.
        let mut is_hit = [false; MAX_BATCH];
        for (i, &e) in sel.iter().enumerate() {
            let key = expert_key(layer, e);
            // The physical slot is deliberately NOT read here — phase 1c takes it from
            // `slot_of` after any same-batch relocation settles.
            if self.policy.hit(key) {
                self.hits += 1;
                // THE CHECK. A hit hands the kernel a slot pointer and a resolved
                // RESIDENT ticket, so nothing downstream waits. If the bytes never landed, the
                // kernel reads uninitialised memory (NaN) or another expert's weights
                // (finite, wrong, silent). Reported once per occurrence rather than
                // fataling, so a run keeps going and the pattern is visible.
                if !self.is_loaded(key) {
                    tracing::error!(
                        "READ-BEFORE-WRITE: layer={layer} expert={e} counted as a cache \
                         HIT but its bytes never landed since admission. The kernel is \
                         about to read an unloaded slot — uninitialised memory (-> NaN) \
                         or a previous expert's weights (-> silently wrong)."
                    );
                }
                self.policy.protect(key);
                is_hit[i] = true;
            }
        }
        // Phase 1b: allocate the misses (evict + place + compact). Slots may relocate.
        #[cfg(feature = "trace")]
        let mut poisoned_any = false;
        for (i, &e) in sel.iter().enumerate() {
            if !is_hit[i] {
                self.misses += 1;
                self.alloc(expert_key(layer, e))?;
                #[cfg(feature = "trace")]
                {
                    poisoned_any = true;
                }
            }
        }
        // The poison fills above run on the DEFAULT stream; the reaper's bounce->slot
        // copies run on a `hipStreamNonBlocking` fetch stream, which does not synchronise
        // with it. Unordered, a fill could land AFTER the read and destroy good data —
        // the diagnostic would then cause the corruption it exists to detect. One join
        // per layer-with-misses orders them.
        //
        // This IS HIP-specific, and it is load-bearing: the poison fill is on the default
        // stream while the reaper's copy is on a non-blocking fetch stream, so nothing orders
        // them without the join. (A second paragraph here argued the same join was needed for
        // a different reason under Vulkan — a recorded-but-unsubmitted `vkCmdFillBuffer`
        // racing a host memcpy — and concluded "do not delete it as HIP-specific". That
        // backend was retired 2026-08-06; the HIP argument above stands alone and is the
        // reason the join is here.)
        //
        // CAVEAT, and it is the same trap `--checksum-x` fell into: this sync may itself
        // perturb the race being hunted. It sits at a different point (after allocation,
        // before reads are submitted) than the per-layer D2H that masked the fault, so it
        // is not the same barrier — but a clean run under poisoning is NOT proof the bug
        // is absent. Only a poison HIT is positive evidence.
        #[cfg(feature = "trace")]
        if poisoned_any {
            crate::backend::device_sync()?;
        }
        // Phase 1c: relocations have settled — resolve final slots and build the reads.
        let mut reads: Vec<ReadSpec> = Vec::new();
        let mut miss_sel: Vec<usize> = Vec::new();
        for (i, &e) in sel.iter().enumerate() {
            let (hot, idx) = self.slot(expert_key(layer, e)).context(
                "expert not resident after alloc (batch exceeds pool — raise --max-mem)",
            )?;
            let (b, t) = (self.ptr(hot, idx), self.tier(hot));
            // SAFETY: address arithmetic into the resolved slot. The offsets are inside it
            // by construction, not by check: `ExpertSet::open_routed` derives both the slot
            // stride and the six offsets from ONE `RoutedFmt`, so they cannot describe
            // different formats. The bytes land when `tickets[i]` is satisfied.
            out.push(unsafe { slot_at(b, &t.off) });
            fmt.push(t.fmt);
            if !is_hit[i] {
                // From the tier that OWNS the slot — its own read table, its own
                // `first_layer`. In hybrid the two tiers are different files.
                let (fd, begin, len) = t.spec(layer, e)?;
                reads.push(ReadSpec {
                    fd,
                    begin,
                    len,
                    dst: self.host_ptr(hot, idx),
                });
                miss_sel.push(i);
            }
        }
        // Phase 2: hand the whole batch to the reaper — it queues+submits (all reads
        // start at once) and signals each miss's ticket when its copy lands.
        // Queue this batch's misses to be marked loaded at the next layer.
        for &i in &miss_sel {
            self.pending_loaded.push(expert_key(layer, sel[i]));
        }
        let miss_tickets = self.fetch.submit(reads)?;
        // A resident expert's data is already there, so it carries the RESIDENT ticket
        // (value 0, satisfied on arrival). Every expert therefore has a ticket and the
        // caller has one code path — there is no residency bool for anyone to branch on.
        tickets.resize(sel.len(), Ticket::RESIDENT);
        for (k, &i) in miss_sel.iter().enumerate() {
            tickets[i] = miss_tickets[k];
        }
        Ok(())
    }
}

#[cfg(test)]
mod ticket_tests {
    use super::Ticket;

    /// **INV-5: an expert cannot be launched without enqueueing its data dependency.**
    ///
    /// The structural half of this is enforced by types and cannot be tested at runtime:
    /// [`RoutedPool::submit`] returns `Vec<Ticket>` and no longer returns a residency mask, so
    /// `gpu.rs` has nothing to branch on and no way to spell "launch without waiting". What
    /// IS testable, and what actually broke before, is the encoding — a resident ticket must
    /// be a real satisfied dependency rather than a sentinel the consumer has to recognise
    /// and skip.
    ///
    /// Timelines start at 0, so `RESIDENT.value == 0` means "wait on 0", which every
    /// timeline satisfies on arrival. If it were, say, `u64::MAX` as an "N/A" marker, the
    /// consumer would need a branch to avoid deadlocking on it — and that branch is exactly
    /// the `hit` mask growing back.
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
             satisfied immediately. A sentinel that had to be SKIPPED would put a residency \
             branch back in the consumer, which is the bug class this removed."
        );
        // A real fetch ticket is never confusable with a resident one: values are assigned
        // from 1 upward per slot.
        let fetched = Ticket { slot: 3, value: 1 };
        assert!(
            !fetched.is_resident(),
            "the first value a slot hands out must NOT read as already-satisfied"
        );
    }
}
