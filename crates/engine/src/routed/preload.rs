//! Preload a routed set that fits the pool ENTIRELY, so decode runs with zero misses.
//!
//! [`RoutedPool::submit`](super::RoutedPool::submit) fills the pool the way a cache wants
//! to be filled — on demand, one layer's picks at a time. That is the right shape for the
//! streaming arm and the wrong one for a set the placement already pinned in full: on
//! Inkling 2 Mini MXFP8 at 64 GiB every one of the 3072 experts (23.5 GiB) is resident by
//! decision, so every expert no prefill layer touched would be a cold miss on the demand
//! path — NVMe latency plus a bounce copy, spent before the same bytes are read again as
//! hits. [`RoutedPool::preload_all`] spends that once, at startup, in parallel, and hands
//! decode a pool that is 100% resident.
//!
//! **P4: this moves bytes, never text.** The slot each expert lands in, and the offset
//! within it, are exactly what the miss path would have produced (`geom.spec` for the
//! source, `arena` placement for the destination), so a preloaded run and a lazily-filled
//! one decode bit-identical ids — which is why the P4 arm's pinned side may now see
//! `first_misses == 0` while still agreeing with its streamed side.
//!
//! Three rules this module exists to keep:
//!
//! 1. **All or nothing.** A partial preload would admit keys into the policy before the
//!    run's own first access and so change which keys 2Q's admission order evicts later —
//!    a bytes/token change on the STREAMING arm, bought by an optimization for the
//!    resident one. `preload_all` returns `Ok(0)` unless the whole set fits.
//! 2. **`--trace` keeps its cold misses.** The capture exists to record the demand path,
//!    and the `trace`-feature poison fill exists to make a read-before-write loud; a
//!    preload would empty both. A capturing run skips the preload entirely.
//! 3. **A preload is not a miss.** `hits`/`misses` count DEMAND accesses, and every
//!    benchmark, every p4 arm and the seam's `DecodeStats` read them as such. Placement
//!    goes through `alloc` (policy + arena), never through `submit`, so both counters
//!    stay 0.

use super::{RoutedGeom, RoutedPool, expert_key};
use crate::fetch::stream::ALIGN;
use anyhow::{Context, Result, bail, ensure};
use std::fs::File;
use std::io;
use std::mem::ManuallyDrop;
use std::os::fd::{FromRawFd, RawFd};
use std::os::unix::fs::FileExt;
use std::time::Instant;

/// How many `pread`s run at once.
///
/// One blocking `pread` occupies one device queue slot, so this is a queue depth. The
/// measured array delivers 6.69 GB/s at P≥4 and 2.53 GB/s at depth 1 (`fetch::stream`'s
/// module docs, measured on this box), so four is where the NVMe stops being the limit.
const PRELOAD_THREADS: usize = 4;

/// One expert's preload read: everything the O_DIRECT `pread` needs, resolved on the main
/// thread so the workers share nothing mutable.
///
/// `dst` is a `usize` rather than a `*mut u8` for one reason: a raw pointer is not `Send`,
/// and the whole point is to hand disjoint slots to several threads. The cast back happens
/// at the single place that dereferences it ([`read_slot`]), under its SAFETY argument.
struct SlotRead {
    fd: RawFd,
    begin: usize,
    /// Useful bytes (`geom.spec`'s `len`) — the short-read floor.
    len: usize,
    /// The aligned superset, `len.next_multiple_of(ALIGN)`. `<= stride`, guaranteed by
    /// `RoutedGeom::check_reads_fit_their_slots` at `open()`, which is why no worker
    /// re-checks that the copy fits its slot.
    want: usize,
    /// This expert's slot base in the pool's HOST mapping.
    dst: usize,
}

impl RoutedPool {
    /// Place and fill EVERY expert this set holds, so the first demand access is a hit.
    ///
    /// Returns the number of experts preloaded, and `0` when there is nothing to do or it
    /// would not apply: an empty set, a set that does not fit the budget, or `--trace`
    /// capturing (this module's rule 2). Callers gate on the placement's own answer
    /// (`pinned == total`) and this is the byte-level restatement of it — the placement's
    /// number is a partition decision, the budget check here is what the arena will
    /// actually honour.
    pub fn preload_all(&mut self) -> Result<usize> {
        let total = self.geom.total_experts();
        if self.trace.is_some() || total == 0 {
            return Ok(0);
        }
        // The precondition, and the reason a partial preload is never attempted: with
        // `total · stride` inside the budget the policy's `evict_until_fits` cannot find a
        // victim (every policy sheds only while `resident + incoming > budget`) and the
        // arena's gap cannot close, so the whole set is placed with zero evictions and
        // zero relocations. Under it, `place_all`'s population check is unreachable — and
        // it is still there, because that is the claim this function is trusted on.
        if total.saturating_mul(self.geom.stride) > self.arena.budget() {
            return Ok(0);
        }
        let started = Instant::now();
        let population: Vec<(usize, usize)> = all_experts(&self.geom).collect();
        self.place_all(&population)?;
        // The population check, and it runs BEFORE a byte is written: a key evicted during
        // placement would have handed its slot to a later key, so filling the read list
        // first would write one expert's bytes into another's slot.
        ensure!(
            self.slot_of.len() == total,
            "routed pool preload placed {} of {total} experts under a budget that holds the \
             whole set — a key was evicted, so the byte accounting disagrees with the policy",
            self.slot_of.len(),
        );
        // A preload is placement, not a demand access: `hits`/`misses` are what the p4 arms
        // and every benchmark read as the run's cache behaviour, and this path must leave
        // them alone. Unreachable by construction (nothing here calls `submit`), asserted
        // because that is the claim the counters are trusted on.
        debug_assert_eq!(
            self.hits + self.misses,
            0,
            "a preload is not a demand access"
        );
        // The `trace` FEATURE poisons every slot `alloc` places — a device fill on the
        // DEFAULT stream — and the fill below is a CPU-driven DMA write that stream does
        // not order. A poison kernel still in flight would land on top of the real bytes,
        // so the fills are retired first: one join for the whole set, where `admit_misses`
        // needs one per layer for exactly this reason.
        #[cfg(feature = "trace")]
        rivoli_backend::device_sync()?;
        let reads = self.read_list(&population)?;
        fill_slots(&reads)?;
        self.mark_all_loaded();
        let secs = started.elapsed().as_secs_f64().max(1e-6);
        let gib = (total as f64 * self.geom.stride as f64) / (1u64 << 30) as f64;
        tracing::info!(
            "routed pool preload: all {total} routed experts loaded in {secs:.2} s ({:.1} GiB/s)",
            gib / secs,
        );
        Ok(total)
    }

    /// Place every expert in the policy and the arena, before a single byte is read.
    ///
    /// All on the calling thread, so `slot_of`/`key_at` are complete and every slot's
    /// address is FINAL before a worker touches one — which is what lets the fill hand out
    /// disjoint slices and share no mutable state at all.
    ///
    /// `begin_batch` after every key, not once: `pinned` is what stops an eviction from
    /// taking a key the batch still has to resolve, and a preload has no batch. Clearing it
    /// per key holds the set at one entry instead of 3072 and leaves it EMPTY, which is the
    /// state a fresh pool is in and the one `submit`'s own `begin_batch` produces. It cannot
    /// cost a placement: the precondition proved no eviction is possible, and the population
    /// check at the call site is what would say so if that ever stopped being true.
    fn place_all(&mut self, population: &[(usize, usize)]) -> Result<()> {
        for &(layer, expert) in population {
            self.alloc(expert_key(layer, expert))?;
            self.policy.begin_batch();
        }
        Ok(())
    }

    /// The per-expert read list, in population order: one aligned O_DIRECT `pread` per
    /// expert, straight into its slot's HOST mapping.
    fn read_list(&self, population: &[(usize, usize)]) -> Result<Vec<SlotRead>> {
        let mut reads = Vec::with_capacity(population.len());
        for &(layer, expert) in population {
            let (fd, begin, len) = self.geom.spec(layer, expert)?;
            let (hot, idx) = self.slot(expert_key(layer, expert)).with_context(|| {
                format!("preload: (layer {layer}, expert {expert}) has no slot")
            })?;
            reads.push(SlotRead {
                fd,
                begin,
                len,
                want: len.next_multiple_of(ALIGN),
                dst: self.host_ptr(hot, idx) as usize,
            });
        }
        Ok(reads)
    }

    /// Mark every placed key loaded — the `is_loaded` sweep the first `submit` needs
    /// (without it every preloaded expert would report `READ-BEFORE-WRITE` on its first
    /// hit).
    ///
    /// Swept from `slot_of` rather than re-walked: the keys that have a slot ARE the
    /// population the check above just counted, so there is no second traversal that could
    /// disagree with it about what was placed.
    fn mark_all_loaded(&mut self) {
        let keys: Vec<u32> = self.slot_of.keys().copied().collect();
        for k in keys {
            self.mark_loaded(k);
        }
    }
}

/// Every `(layer, expert)` this set holds, in table order.
///
/// Collected by the caller rather than iterated in place: placement needs `&mut self`
/// while the geometry is what it is walking, and one small `Vec` buys both loops the same
/// order without a borrow dance.
fn all_experts(geom: &RoutedGeom) -> impl Iterator<Item = (usize, usize)> + '_ {
    geom.layer_range()
        .flat_map(move |layer| (0..geom.n_experts()).map(move |expert| (layer, expert)))
}

/// Fill every slot from its file, on [`PRELOAD_THREADS`] threads.
///
/// **Straight into the pool slot, with no bounce arena and no device stream traffic.**
/// The streamer's bounce hop exists for the DEMAND path, where it buys 12.4 GB/s against
/// 5.66 GB/s of DMA into VMM pages (measured 2026-07-30, `fetch::stream`'s module docs)
/// on every miss of a run. A preload pays it once: 23.5 GiB at the direct rate is ~4 s of
/// startup against ~2 s of reads plus a second copy, and it buys the fill its whole
/// design — one thread per disjoint slot, no shared staging, nothing to synchronise. The
/// rate the log prints is measured, not this comment's arithmetic.
///
/// Coherence needs no fence of its own. The bytes are written by the kernel's DMA engine,
/// not by CPU stores, so the argument is the one `VmmBuf::ptr_mut` records for the
/// io_uring DMA fill: the reading kernel's dispatch packet carries a system-scope acquire
/// that invalidates the GPU caches, and here the preload additionally precedes every
/// launch that could have cached anything in this VMM.
fn fill_slots(reads: &[SlotRead]) -> Result<()> {
    let per_thread = reads.len().div_ceil(PRELOAD_THREADS).max(1);
    std::thread::scope(|scope| {
        // Every worker is spawned BEFORE the first join: joining as we go would serialize
        // the fill back to one outstanding read, which is the 2.53 GB/s arm.
        let mut workers = Vec::with_capacity(PRELOAD_THREADS);
        for part in reads.chunks(per_thread) {
            workers.push(scope.spawn(move || part.iter().try_for_each(read_slot)));
        }
        for w in workers {
            match w.join() {
                Ok(done) => done?,
                Err(_) => bail!("routed pool preload: a fill worker panicked"),
            }
        }
        Ok(())
    })
}

/// Read one expert's aligned superset into its slot.
///
/// Judged exactly as the streamer judges a completion: the useful window is `len` bytes,
/// so a short read that still covers it is the trailing-EOF truncation `min_completion`
/// accepts (the bytes past `len` are never read by any kernel), while one that does not is
/// a real mid-file short read.
fn read_slot(r: &SlotRead) -> Result<()> {
    // SAFETY: `r.fd` is one of the routed set's live O_DIRECT descriptors — owned by the
    // `ExpertSet` the geometry was tabulated from, which outlives the pool — and
    // `ManuallyDrop` makes this a BORROW: the wrapper must never close an fd it does not
    // own. `read_at` is `pread(2)`, which touches no shared file offset, so several
    // workers over the same fd are safe.
    let file = ManuallyDrop::new(unsafe { File::from_raw_fd(r.fd) });
    // SAFETY: `r.dst` is one slot's host base in the pool VMM and owns `stride >= want`
    // writable bytes; the read list gives every slot exactly one owner, so no other
    // thread (and no device work — the preload runs before the first launch) touches it.
    let buf = unsafe { std::slice::from_raw_parts_mut(r.dst as *mut u8, r.want) };
    let got = loop {
        match file.read_at(buf, r.begin as u64) {
            // A caught signal before the transfer started; the request is still the same
            // aligned one, so retry rather than poison a 23 GiB startup load.
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "routed pool preload: read of {} bytes at offset {} failed",
                        r.want, r.begin
                    )
                });
            }
            Ok(n) => break n,
        }
    };
    ensure!(
        got >= r.len,
        "routed pool preload: short read at offset {} — {got} of {} useful bytes (the aligned \
         superset was {})",
        r.begin,
        r.len,
        r.want,
    );
    Ok(())
}
