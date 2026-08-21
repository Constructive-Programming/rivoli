//! io_uring O_DIRECT cold-expert streamer. A single NVMe read is latency-bound
//! (~4 GB/s here); io_uring keeps the queue full and the NVMe delivers ~5.8 GB/s
//! random (QD≥4). So a MoE layer submits all its cold reads at once and joins
//! once — folding the old mmap-warm + memcpy-fetch into one overlapped stream.
//!
//! The ring is the `io-uring` crate (talks to the io_uring syscalls directly — no
//! liburing system lib). This module owns the O_DIRECT alignment math (block-aligned
//! offset/length/buffer), the fds, and the VMM destination pointers; the two BACKEND ops
//! (host staging arena, async H2D copy) are `rivoli_*` wrappers in `kernels/async.hip` under
//! `rocm` — see [`stage`], which was the whole of the per-backend difference until the
//! Vulkan half was deleted with that backend on 2026-08-06.
//!
//! There is ONE destination path and no flag to change it: every read lands in a HOST
//! staging arena and is async-copied from there into the VMM slot (`queue`'s `dst` is the
//! VMM slot; the arena is the hop in between). Two independent things forbid the obvious
//! alternative of DMA-ing the read straight into VMM, and a future reader needs both —
//! the first says WHERE the arena may live, the second says why the hop is not a cost.
//!
//! 1. **amdgpu could not `get_user_pages` device memory.** io_uring/O_DIRECT DMA into VMM
//!    device pages EFAULTed when this was found (6.18.38-gentoo, 2026-07-17; a regression
//!    vs ≤6.18.35-r1) and the staging hop began as the workaround. It no longer reproduces
//!    on 6.18.38, so that part is history — but it is the history that constrains the
//!    arena: it must be host memory the kernel can pin, NEVER device-local, or the read
//!    into it reintroduces the original fault. Under HIP the arena is `hipHostMalloc`'d and
//!    is host memory by construction; the retired Vulkan half had to exclude `DEVICE_LOCAL`
//!    explicitly for the same reason, which is the clearest statement of the constraint a
//!    future backend inherits.
//! 2. **DMA into VMM device pages runs at half the bandwidth.** 5.66 GB/s vs 12.4 GB/s
//!    into the pinned arena, so writing the read straight into VMM more than DOUBLES the
//!    cost of a miss. Measured 2026-07-30, int3-vq/dense/lru @512: marginal cost per
//!    missed expert 1239 us staged -> 2709 us direct, 2.59 -> 1.19 tok/s. The READ side is
//!    unchanged either way — the pool is device-local VMM regardless of how it was
//!    filled — so a zero-miss layer costs 1563 us vs 1525 us, equal within noise. Misses
//!    are the entire design, which makes the direct path strictly worse on every workload
//!    that matters. It was a `--direct-vmm-dma` flag until 2026-08-01; a flag with no
//!    workload that wants it is not a choice, so it was deleted. (An earlier note here
//!    blamed a ~40% `mlp` read tax from host-mapped VMM; that configuration is not what
//!    the flag produced.) The flag was recovered 2026-08-18 as a DIAGNOSTIC for the GLM
//!    nondeterminism investigation — it answered its question (direct mode diverges too)
//!    and was deleted again 2026-08-20 with the rest of the diagnostic arms;
//!    `docs/investigations/glm-nondeterminism-closeout.md` keeps the record.
//!
//! Repro of (1): `git show 3e1bd96:docs/probes/iouring_vmm.cpp` (faults into VMM) vs the
//! iou_host probe (pinned host) — both predate the empty-slate rebuild, neither is in the
//! tree.
//!
//! SQPOLL is requested on the ring (own poller thread). Without it, submit is an
//! `io_uring_enter` in which the CALLING thread walks the SQEs and drives the
//! btrfs/blk-mq dispatch inline, serially (~2.96 ms/expert at 6 SQEs): the call
//! doesn't return promptly AND the batch reaches the device at queue depth 1 (2.53
//! GB/s) instead of the 6.69 GB/s the array delivers at P≥4. The poller takes the whole
//! SQ tail at once, so the batch is genuinely concurrent. Falls back to a plain ring
//! (the QD1 perf arm, still correct) if SQPOLL setup is refused.
//!
//! Needs a backend — its sole consumer is the GPU decode pin. The
//! ring itself is backend-independent; only the two staging ops differ, and they are the
//! whole of [`stage`] below.
#![cfg(feature = "rocm")]

use crate::fetch::asyncfetch::FetchFolds;
#[cfg(feature = "corruption-probe")]
use crate::fetch::asyncfetch::FoldProbe;
use crate::fetch::stage; // the HIP FFI half — the ONLY backend-specific block here
use anyhow::{Result, ensure};
use io_uring::{IoUring, opcode, types};
use std::ffi::c_void;
use std::io;
use std::mem::ManuallyDrop;
use std::os::fd::RawFd;

/// O_DIRECT block alignment. 4 KiB is a safe superset of any real logical block
/// (512/4096) and matches the page/VMM granularity — offset, length, and buffer
/// must all be multiples of it.
pub const ALIGN: usize = 4096;

/// Destination bytes needed to O_DIRECT-read `len` bytes starting at an arbitrary
/// file offset: the aligned superset, upper-bounded independent of the offset so a
/// reused slot can be sized once. `align_up(len) + ALIGN` covers the worst-case
/// straddle (up to `ALIGN-1` leading pad + trailing round-up).
pub fn slot_span(len: usize) -> usize {
    len.next_multiple_of(ALIGN) + ALIGN
}

/// Minimum bytes an O_DIRECT completion must deliver to cover the useful window
/// `[begin, begin+len)`, given the read starts at the aligned-down offset: the
/// sub-block offset (`begin - align_down(begin)`) plus `len`. A completion of at
/// least this is fine even if the aligned SUPERSET was truncated by trailing EOF
/// padding; anything less is a real mid-file short read (stale slot-tail bytes).
fn min_completion(begin: usize, len: usize) -> u64 {
    let ab = begin & !(ALIGN - 1);
    ((begin - ab) + len) as u64
}

/// A ring of in-flight reads. `entries` caps how many can be queued before a
/// `reap` batch — sized to a layer's cold-read count with margin.
/// One read's byte extent. Grouped because `begin` and `len` are two bare `usize` a
/// caller can transpose and O_DIRECT alignment makes the wrong pair plausible.
#[derive(Clone, Copy)]
pub struct ReadSpan {
    pub begin: usize,
    pub len: usize,
}

/// The fetch path's knobs — what survives of the intervention matrix the GLM
/// nondeterminism investigation built. The nine refuted/answered diagnostic arms were
/// deleted end-to-end 2026-08-20; `docs/investigations/glm-nondeterminism-closeout.md`
/// keeps the record of what each measured. Still a struct rather than two `bool`
/// parameters: bundled knobs are what keeps [`Streamer::new`] out of the transposition
/// hazard this workspace refuses everywhere else.
///
/// [`FetchKnobs::default`] is the STOCK configuration (both arms off) — derivable now
/// that the deleted `bounce` field, whose derived default would have been the wrong
/// mode, is gone.
#[derive(Clone, Copy, Default)]
pub struct FetchKnobs {
    /// `--arena-refresh`: full-width device read of the just-written arena window,
    /// pre-copy. The ONE clean cell of the matrix — a mitigation, not a fix.
    pub arena_refresh: bool,
    /// `--copy-via-cpu`: the bounce→slot hop as a HOST memcpy on the reaper thread —
    /// the candidate FIX. No GPU agent then ever reads memory the NVMe wrote: the
    /// arena is read only by the CPU (the CQE's own guarantee, the one btrfs's datasum
    /// verification already relies on) and the slot is written only by the CPU (the
    /// CPU→GPU coherence `kernels/vmm.hip` was verified to have and the resident
    /// tier's 281 GB startup load already spends). The ticket is signalled on the
    /// fetch stream exactly as after an SDMA copy, so the consumer side is unchanged.
    pub cpu_copy: bool,
}

/// One read's destination: the aligned pointer, the slot whose ticket gates reuse, and the
/// optional divergence-fold target.
#[derive(Clone, Copy)]
pub struct ReadDst {
    pub ptr: *mut u8,
    pub slot: u32,
    /// `--divergence-log` only; see [`FetchFolds`] for why it is a named pair rather than a base
    /// pointer plus an offset.
    pub fold: FetchFolds,
}

pub struct Streamer {
    /// `ManuallyDrop` so `Drop` can tear the ring down BEFORE freeing the arena the
    /// ring's SQEs point into (Rust would otherwise run the `Drop` body — the arena
    /// free — before dropping this field). Access is transparent via `Deref`.
    ring: ManuallyDrop<IoUring>,
    /// Ring/arena capacity in staging slots. A batch is bounded by this (see `queue`), and
    /// the fetcher needs one timeline per slot.
    entries: u32,
    /// Per-read pinned-bounce stride: the largest aligned superset any single read
    /// may deliver. A `queue` whose superset exceeds this can't fit its bounce slot.
    span: usize,
    /// Staging arena, `entries * span` bytes: read slot `user_data` is
    /// `arena + user_data * span`. HIP-pinned under `rocm` — see [`stage`]. Never null:
    /// `new` refuses to build a `Streamer` around a failed allocation.
    arena: *mut u8,
    /// Per-SLOT VMM destination + aligned read length. `reap` copies
    /// `nbytes` from the arena slot into `dst`. Indexed by staging slot — which is the
    /// read's `user_data` — and NOT cleared per batch: a slot's lifetime is owned by its
    /// [`Ticket`](crate::fetch::asyncfetch::Ticket) now, not by the batch that happened to use it.
    dst: Vec<*mut u8>,
    nbytes: Vec<u32>,
    /// Per-SLOT divergence-fold targets, parallel to `dst`. [`FetchFolds::OFF`] = folds off.
    fold: Vec<FetchFolds>,
    /// ARENA REFRESH: enqueue a full-width device read of the just-written arena window on the
    /// fetch stream BEFORE the copy. The only intervention measured to make GLM decode
    /// reproduce itself; a MITIGATION with an unexplained mechanism, not a root-cause fix.
    /// Evidence, the fifteen refuted alternatives, and the ceiling: `kernels/async.hip` and
    /// `docs/investigations/glm-nondeterminism-closeout.md`.
    arena_refresh: bool,
    /// `--copy-via-cpu`: the bounce→slot hop as a host memcpy on the reaper thread — see
    /// [`FetchKnobs::cpu_copy`].
    cpu_copy: bool,
    /// One-shot: has the first arm application been LOGGED yet. Positive evidence that the
    /// intervention ran, not that it was asked for — two rounds of the nondeterminism
    /// investigation were lost to arms that never applied, and an arm that did not apply reds
    /// exactly like one that does not work.
    logged_refresh: bool,
    /// Device word the refresh kernel stores to only on an impossible value — it exists to stop
    /// the loads being optimised away, is never read, and needs no synchronisation.
    refresh_sink: *mut u64,
    /// Copies actually issued, per path — `[sdma-memcpy, host-cpu]`.
    ///
    /// **A COUNT, not the flag.** Two rounds of the nondeterminism investigation were spent on an
    /// arm that could not be believed because nothing observed whether the intervention applied:
    /// the log recorded the intent and the runtime was free to do something else. An intervention
    /// that never applied and one that does not work produce the same red, so the candidate fix
    /// reports what it DID.
    issued: [u64; 2],
    /// Per-SLOT minimum completion length (sub-block offset + useful len). `reap` compares
    /// the completion `res` against it so a real mid-file short read is caught while
    /// EOF-padding truncation is tolerated.
    min_res: Vec<u64>,
}

/// Which of the two bracket positions a divergence fold serves — see the BRACKET THE
/// COPY comment in [`Streamer::reap`]. The position selects everything: the accumulator,
/// the mode, the log label, and the subject buffer, because the buffer IS what the
/// position means (pre-copy = the arena window the read landed in, post-copy = the pool
/// slot the copy targeted).
#[cfg(feature = "corruption-probe")]
#[derive(Clone, Copy)]
enum FoldSide {
    /// Pre-copy: `bh` hashes what the drive delivered into the pinned arena.
    Bounce,
    /// Post-copy: `sc` hashes what arrived in the pool slot.
    Slot,
}

// SAFETY: the ring + arena are exclusively owned by whoever holds the `Streamer`.
// io_uring rings are NOT thread-safe, so this only asserts the handle can MOVE across
// threads — `AsyncFetch` moves it once into its reaper thread, which is then the sole
// accessor. Never share a `&Streamer`/`&mut Streamer` across threads.
unsafe impl Send for Streamer {}

impl Streamer {
    /// `entries` = max in-flight reads; `span` = the largest aligned superset a
    /// single read may deliver (`slot_span` of the biggest projection tensor); `knobs`
    /// selects the mitigation and the candidate fix (see [`FetchKnobs`]).
    /// Always allocates the `entries * span` host staging arena — it is the only
    /// destination path (see this module's header).
    pub fn new(entries: u32, span: usize, knobs: FetchKnobs) -> Result<Self> {
        let FetchKnobs {
            arena_refresh,
            cpu_copy,
        } = knobs;
        // The bounce arena has exactly `entries` slots and `queue` indexes it by the
        // SQE's user_data. io_uring rounds the SQ up to a power of two, so `entries` MUST
        // be a power of two for the arena to match the SQ capacity — otherwise a push
        // could succeed for a user_data past the arena's last slot (OOB pinned write).
        // The sole caller passes `(top_k+4).next_power_of_two()`.
        debug_assert!(
            entries.is_power_of_two(),
            "Streamer entries ({entries}) must be a power of two (arena ↔ SQ capacity)"
        );
        // Own SQPOLL poller (sq_thread_idle 2000ms, re-armed on submit); fall back to a
        // plain ring if the kernel refuses SQPOLL (the QD1 perf arm, still correct).
        let ring = IoUring::builder()
            .setup_sqpoll(2000)
            .build(entries)
            .or_else(|_| IoUring::new(entries))?;

        // `--copy-via-cpu` IS the copy path and it leaves no GPU-side reader of the arena, so
        // the refresh arm loses its subject. A silently-ignored knob is how an arm gets
        // attributed to the wrong cause.
        ensure!(
            !(cpu_copy && arena_refresh),
            "--copy-via-cpu leaves no GPU-side reader of the arena; --arena-refresh has no subject"
        );
        let arena_bytes = entries as usize * span;
        let arena = stage::alloc(arena_bytes);
        // Logged because the arena is the region the nondeterminism investigation localised
        // the defect to (`docs/investigations/glm-nondeterminism-closeout.md`): a run's
        // record has to say how big it was, or its result cannot be attributed.
        tracing::info!(
            "bounce arena: {:.0} MiB pinned host ({entries} slots x {span} B)",
            arena_bytes as f64 / (1u64 << 20) as f64,
        );
        ensure!(
            !arena.is_null(),
            "bounce arena alloc failed (entries={entries}, {:.0} MiB)",
            arena_bytes as f64 / (1u64 << 20) as f64
        );

        if cpu_copy {
            tracing::info!(
                "COPY VIA CPU (--copy-via-cpu): the bounce->slot hop is a host memcpy on the \
                 reaper thread; no GPU agent reads IO-written memory anywhere on the path"
            );
        }

        Ok(Self {
            ring: ManuallyDrop::new(ring),
            entries,
            span,
            cpu_copy,
            logged_refresh: false,
            arena,
            dst: vec![std::ptr::null_mut(); entries as usize],
            nbytes: vec![0; entries as usize],
            fold: vec![FetchFolds::OFF; entries as usize],
            arena_refresh,
            // The sink is a pinned word, not a device allocation: the kernel's store is never
            // taken, so the address only has to be writable and mapped. It reuses the arena's
            // first word, which avoids an allocation whose lifetime would have to be argued
            // against teardown order.
            refresh_sink: arena.cast::<u64>(),
            issued: [0; 2],
            min_res: vec![0; entries as usize],
        })
    }

    /// Log an arm's first APPLICATION, once — positive evidence the intervention ran, as
    /// against the flag that asked for it. Two rounds of the nondeterminism investigation
    /// were lost to arms that never applied; every arm's acceptance line comes through here.
    fn log_applied_once(&mut self, text: String) {
        if !self.logged_refresh {
            self.logged_refresh = true;
            tracing::info!("{text}");
        }
    }

    /// Queue an O_DIRECT read of `len` bytes at file offset `begin` (from `fd`)
    /// into `dst`, staged through bounce arena slot `slot`. Reads the aligned superset
    /// `[align_down(begin), align_up(begin+len))`, so `dst` must be `ALIGN`-aligned and
    /// own at least `slot_span(len)` bytes. Returns the sub-block offset in `dst` where
    /// the useful `len` bytes land (i.e. the caller reads `dst.add(returned) .. +len`).
    ///
    /// **`slot` is chosen by the caller, not by queue order.** It used to be a `queued++`
    /// counter reset per batch, which made a slot's lifetime the batch's lifetime; slot
    /// reuse is now gated on the slot's own timeline in `AsyncFetch::take_slot`.
    ///
    /// # Safety
    /// `dst` must be `ALIGN`-aligned and valid for `slot_span(len)` writable bytes
    /// until this read's [`reap`](Self::reap) completes. `slot` must not be in use by a
    /// read whose bounce copy has not yet retired (the caller's ticket gate).
    pub unsafe fn queue(&mut self, fd: RawFd, span: ReadSpan, dst: ReadDst) -> Result<usize> {
        let ReadSpan { begin, len } = span;
        let ReadDst {
            ptr: dst,
            slot,
            fold,
        } = dst;
        debug_assert_eq!(
            dst as usize % ALIGN,
            0,
            "O_DIRECT dst must be block-aligned"
        );
        let ab = begin & !(ALIGN - 1);
        let ae = (begin + len).next_multiple_of(ALIGN);
        let nbytes = ae - ab;
        ensure!(
            nbytes <= self.span,
            "read superset {nbytes} exceeds bounce span {} (raise Streamer span)",
            self.span
        );
        let sub = begin - ab; // useful bytes start `sub` into the aligned read
        let ud = slot;
        ensure!(
            ud < self.entries,
            "staging slot {ud} out of range (entries {})",
            self.entries
        );
        // The read lands in this slot's arena window and `reap` copies it on to `dst`.
        //
        // SAFETY: `ud < entries` (checked above), and the arena is `entries*span` with
        // every read's `nbytes <= span` (checked above), so this slot's window is owned
        // and in bounds.
        let into = unsafe { self.arena.add(ud as usize * self.span) };
        let read = opcode::Read::new(types::Fd(fd), into, nbytes as u32)
            .offset(ab as u64)
            .build()
            .user_data(u64::from(ud));
        // SAFETY: `into` is valid for `nbytes` writable bytes until this read's reap
        // (caller's `dst` contract, or our arena slot); the SQE references it by raw
        // pointer, so it must outlive the completion — it does.
        let pushed = unsafe { self.ring.submission().push(&read) };
        // io_uring owns the real SQ occupancy, so the capacity it ran out of is the only
        // number worth reporting — a parallel `queued` counter existed solely for this
        // message and could never have said anything the ring had not already decided.
        ensure!(
            pushed.is_ok(),
            "io_uring SQ full: the ring holds {} entries (raise it)",
            self.entries
        );
        self.dst[ud as usize] = dst;
        self.nbytes[ud as usize] = nbytes as u32;
        self.fold[ud as usize] = fold;
        // The completion must deliver at least the useful window `[begin,begin+len)`
        // from the aligned start; a shorter read is mid-file truncation (checked in
        // `reap` against `min_res`). Trailing EOF padding beyond this is fine.
        self.min_res[ud as usize] = min_completion(begin, len);
        Ok(sub)
    }

    /// Submit the queued reads to the kernel WITHOUT waiting, so they start running
    /// on the NVMe/DMA side immediately. The following per-read [`reap`](Self::reap)
    /// calls collect the same completions; the bookkeeping is deliberately left intact
    /// for them.
    ///
    /// Submitting here (rather than at reap time) starts the reads promptly and
    /// CONCURRENTLY — the whole batch reaches the device before any host work that
    /// follows, instead of dispatching serially at join time. The own-poller is what
    /// makes that concurrency real (see [`Streamer::new`]).
    pub fn submit(&self) -> Result<()> {
        // With SQPOLL this wakes the poller if idle; otherwise it does the io_uring_enter
        // submit. Either way the already-pushed SQEs are handed to the kernel and the
        // CQEs are collected by the matching `reap` calls. (The sole caller only submits
        // non-empty batches; an empty SQ would be a harmless no-op regardless.)
        // EINTR is retried for the same reason `reap` retries it: a stray signal on the
        // reaper thread must not poison a fetch whose SQEs are already pushed. `submit`
        // does not wait, so this is far less likely than in `reap` — but the two calls
        // were inconsistent, and `reap`'s own comment already grants that a signal can
        // land here. `io_uring_enter` is not in the SA_RESTART-able class, so nothing
        // retries it for us. (This also has to hold before any SIGPROF-based profiler
        // could ever be attached — see docs/measurement/traces.md, "Pyroscope".)
        loop {
            match self.ring.submit() {
                Ok(_) => return Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(anyhow::anyhow!("io_uring submit failed: {e}")),
            }
        }
    }

    /// Block for the NEXT completion and validate it: a nonnegative `res` that reaches
    /// the slot's minimum useful length (`min_res`). Returns the completed read's
    /// `user_data`, which is its staging slot.
    fn next_completed_slot(&mut self) -> Result<usize> {
        // Block for the next completion. A ready CQE is taken immediately; otherwise
        // submit_and_wait(1) parks until at least one more lands (submits nothing new —
        // the batch was already submitted).
        let (res, ud) = loop {
            if let Some(rd) = self.next_cqe() {
                break rd;
            }
            // Park until at least one more completion lands. A caught signal (EINTR) is
            // benign — the batch is still queued in the kernel — so retry rather than
            // poison the whole fetch on e.g. a SIGWINCH delivered to the reaper thread.
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(anyhow::anyhow!("io_uring wait failed: {e}")),
            }
        };
        ensure!(
            res >= 0,
            "io_uring read failed: {}",
            io::Error::from_raw_os_error(-res)
        );
        let ud = ud as usize;
        ensure!(
            res as u64 >= self.min_res[ud],
            "short read on expert slot {ud}: {res} < {} useful bytes",
            self.min_res[ud]
        );
        Ok(ud)
    }

    /// Per-read async reap: block for the NEXT read to complete, kick its bounce→slot
    /// copy on `stream` (or, under `--copy-via-cpu`, perform it on the spot as a host
    /// memcpy), and return the completed read's `user_data` (the index into the batch).
    /// The caller reaps exactly once per read it queued. Reads run concurrently on the
    /// NVMe; completions arrive in device order, each resolving its own expert.
    ///
    /// # Safety
    /// `stream` is a live HipStream handle; the copied read's `dst` slot must stay
    /// valid until that stream's completion signal fires.
    pub unsafe fn reap(&mut self, stream: *mut c_void) -> Result<usize> {
        let ud = self.next_completed_slot()?;
        // SAFETY: `ud < entries`, and every read's `nbytes <= span`, so this slot's arena window
        // is owned and in bounds.
        let src = unsafe { self.arena.add(ud * self.span) };
        // ARENA REFRESH, enqueued BEFORE the copy on the fetch stream. See the struct field and
        // `kernels/async.hip` for the evidence, the fifteen alternatives that did not work, and
        // the ceiling. Stream-ordered ahead of the copy, so it needs no sync of its own.
        // SAFETY: `src` is this slot's arena window, valid for `nbytes[ud]` bytes; the stream is
        // live; `refresh_sink` is a mapped, writable word the kernel never stores to.
        if self.arena_refresh {
            unsafe {
                stage::touch_region(src, self.nbytes[ud] as usize, stream, self.refresh_sink)
            }
            .map_err(|e| anyhow::anyhow!("arena refresh launch failed: {e}"))?;
        }
        // `--divergence-log`: BRACKET THE COPY. `bh` hashes what the drive delivered into the
        // pinned arena; `sc` hashes what arrived in the pool slot. Both are enqueued on the FETCH
        // stream around the copy, so the three are stream-ordered with no host sync and no
        // barrier — which is the whole reason this instrument can be pointed at a timing defect.
        //
        // A difference in `bh` across two runs means the READ delivered different bytes; `bh`
        // equal with `sc` differing isolates the copy itself. Both are folded at FULL WIDTH
        // (`nbytes`), so neither can miss a corruption by sampling.
        //
        // The fold reads the arena as f32 purely to reuse `hash_rows`; it folds raw bits and the
        // payload is packed indices plus bf16 scales, so the interpretation is irrelevant.
        // `nbytes` is a multiple of the O_DIRECT block and therefore of 4.
        //
        // WHY THESE FOLDS CANNOT RACE THE PROBE'S PER-PASS CLEAR, recorded here because the
        // argument spans three files and a reader of this one deserves it: each read's `bh`/copy/
        // `sc` are enqueued before THAT read's timeline signal (`asyncfetch.rs::run_job`), the miss
        // kernel waits on that value, `launch_moe` host-awaits the miss stream, and `run_layer`
        // ends every layer with an unconditional `device_sync`. `Probe::drain` runs only after
        // every layer of a pass and syncs again after its own clear. So no fold for pass N+1 can
        // be enqueued before pass N's clear has executed.
        #[cfg(feature = "corruption-probe")]
        if self.fold[ud].bh_armed() {
            // THE PRE-COPY POSITION — the one measured to SUPPRESS. Same ladder as post-copy, and
            // here it decides the FIX rather than the diagnosis: if `Nop` suppresses, what repairs
            // the hazard is the kernel dispatch's acquire and not the bytes read, so the fix is a
            // coherent arena (or an explicit cache operation) and any read is incidental.
            //
            // SAFETY: this slot's arena window holds the `nbytes[ud]` bytes just written by
            // the completed read; `stream` is live.
            unsafe { self.launch_fold(ud, FoldSide::Bounce, stream) };
        }
        unsafe { self.copy_bounce_to_slot(ud, src, stream) }?;
        #[cfg(feature = "corruption-probe")]
        if self.fold[ud].sc_armed() {
            // THE POST-COPY POSITION — the one measured to suppress the divergence. Three
            // alternatives at the same point in the stream, which is Phase 2's whole experiment:
            //
            //   Full  fold the entire slot: both DELAYS and READS, so on its own it cannot say
            //         which of the two repaired the hazard.
            //   Spin  the same launch geometry and trip count, touching NO memory. Suppression
            //         here means the hazard is pure TIME — a fixed-lag write-visibility problem.
            //   Line  read ONE cacheline. Suppression only here means TOUCHING the bytes repairs
            //         them, and at ~0% cost this is also the cheapest candidate fix.
            //
            // SAFETY: the pool slot holds `nbytes[ud]` bytes, live until this read's signal;
            // `stream` is live and the copy is already enqueued on it, so any arm that reads
            // the slot does so after it.
            unsafe { self.launch_fold(ud, FoldSide::Slot, stream) };
        }
        Ok(ud)
    }

    /// THE COPY — two paths, one destination:
    ///  - default: `hipMemcpyAsync` (SDMA) on the fetch stream, async;
    ///  - `--copy-via-cpu`: a HOST memcpy, right here on the reaper thread.
    ///
    /// The CPU path is the candidate FIX and its argument is a subtraction: after it, NO
    /// GPU agent anywhere reads memory the NVMe's DMA wrote. The arena is read only by the
    /// CPU — the visibility the io_uring CQE actually guarantees (and the one btrfs's
    /// datasum check already spends) — and the slot is written only by the CPU, whose
    /// writes to this VMM are verified GPU-coherent (`kernels/vmm.hip`; the resident
    /// tier's startup load spends the same property at 281 GB). The ticket still signals
    /// on the fetch stream after this returns, so the consumer side is byte-identical.
    ///
    /// # Safety
    /// `dst[ud]` is a live pool slot the pipeline keeps valid until this read's
    /// signal fires, and its HOST mapping is writable (that is what `host_ptr` is FOR);
    /// `src` is this slot's arena window holding the just-read `nbytes[ud]` bytes; arena
    /// and pool never alias. `stream` is a live HipStream handle.
    unsafe fn copy_bounce_to_slot(
        &mut self,
        ud: usize,
        src: *mut u8,
        stream: *mut c_void,
    ) -> Result<()> {
        if self.cpu_copy {
            // SAFETY: caller's contract — non-aliasing live src/dst of `nbytes[ud]` bytes.
            unsafe { std::ptr::copy_nonoverlapping(src, self.dst[ud], self.nbytes[ud] as usize) };
            self.issued[1] += 1;
            self.log_applied_once(format!(
                "COPY VIA CPU applied: host memcpy of {} B, arena slot {ud} -> pool slot",
                self.nbytes[ud]
            ));
        } else {
            // SAFETY: `dst[ud]` is a live pool slot the pipeline keeps valid until this
            // read's signal fires; the arena slot holds the just-read bytes; `stream` is
            // live. Copies the full aligned `nbytes` (a trailing-EOF short read leaves
            // stale bytes only past the useful window, never read).
            let r =
                unsafe { stage::copy_to_slot(self.dst[ud], src, self.nbytes[ud] as usize, stream) };
            if let Err(e) = r {
                anyhow::bail!("bounce staging copy failed on slot {ud} ({e})");
            }
            self.issued[0] += 1;
        }
        Ok(())
    }

    /// One divergence fold at either bracket position: pick the buffer extent the armed
    /// mode calls for and enqueue ONE `hash_rows` on the fetch stream. One call serves
    /// every arm; they differ only in WHICH buffer, HOW MUCH of it, and therefore how
    /// long they take. See `FoldProbe` for the ladder and what each rung means.
    ///
    /// `i_base = ud * n`, NOT 0: every cold read of a layer folds into ONE accumulator, so at
    /// 0 the fold would be invariant under two reads' payloads being SWAPPED between their
    /// destinations — a crossed destination is precisely the class under investigation. `ud`
    /// is deterministic given the miss sequence (INV-9), so it is comparable across runs.
    ///
    /// LOGGED, NOT `?`. A `?` here returns from `reap` after the CQE is consumed and (at the
    /// pre-copy position) BEFORE `copy_to_slot`, so the reaper poisons, the ticket is
    /// released from the host, and the miss kernel launches over a slot this layer never
    /// wrote — the INSTRUMENT changing what the engine computes, which it may never do.
    ///
    /// # Safety
    /// The side's subject buffer must own `nbytes[ud]` readable bytes — `Bounce`: this
    /// slot's arena window, just written by the completed read; `Slot`: the pool slot,
    /// live until this read's signal, with the copy already enqueued on `stream` so any
    /// fold that reads it does so after it. The side's decoy, when its mode selects it,
    /// is allocated slot-sized. The side's accumulator is one live device u64; `stream`
    /// is live.
    #[cfg(feature = "corruption-probe")]
    unsafe fn launch_fold(&self, ud: usize, side: FoldSide, stream: *mut c_void) {
        let f = &self.fold[ud];
        // SAFETY: `ud < entries` and every read's `nbytes <= span` (both checked in
        // `queue`), so this slot's arena window is owned and in bounds.
        let (mode, acc, label, base) = match side {
            FoldSide::Bounce => {
                let src = unsafe { self.arena.add(ud * self.span) };
                (f.bh_mode, f.bh, "bh", src as *const f32)
            }
            FoldSide::Slot => (f.sc_mode, f.sc, "sc", self.dst[ud] as *const f32),
        };
        let n = self.nbytes[ud] as usize / 4;
        let (buf, count, stride) = match mode {
            FoldProbe::Off => (base, 0, 1),
            FoldProbe::Full => (base, n, 1),
            // Every cache line of the buffer, ~1/32 of its bytes — a sweep that COVERS it,
            // so unlike reading one line at the front it could actually be a fix.
            FoldProbe::Line => (base, n, f.line_stride),
            // Same size, same bandwidth, same duration — a buffer that is NOT the subject.
            FoldProbe::Decoy => (f.decoy, n, 1),
            // Same launch and the same stream-boundary cache maintenance, ~no work.
            FoldProbe::Nop => (f.decoy, 1, 1),
        };
        if count == 0 {
            return;
        }
        // SAFETY: caller's contract — `buf` owns `count` readable f32, `acc` is one live
        // device u64, `stream` is live.
        let r = unsafe {
            rivoli_backend::launch_hash_rows(
                buf,
                count,
                stride,
                (ud as u64) * n as u64,
                acc,
                stream,
            )
        };
        if let Err(e) = r {
            tracing::error!("divergence probe: {label} fold failed on slot {ud} ({e:#})");
        }
    }

    /// Take one ready completion (if any), advancing the CQ head. Returns
    /// `(res, user_data)`; the guard's `sync` refreshes visible completions and its
    /// drop writes the consumed head back to the kernel.
    fn next_cqe(&mut self) -> Option<(i32, u64)> {
        let mut cq = self.ring.completion();
        cq.sync();
        cq.next().map(|c| (c.result(), c.user_data()))
    }

    /// Copies issued per path, `[hipMemcpyAsync, host memcpy]` — the OBSERVATION a
    /// candidate-fix arm is read off, as against the flag it was asked for.
    pub fn copies_issued(&self) -> [u64; 2] {
        self.issued
    }

    /// Staging-slot count — the ring's capacity, and the number of per-slot timelines the
    /// fetcher needs.
    pub fn entries(&self) -> u32 {
        self.entries
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        // Drop the ring BEFORE freeing the arena its SQEs point into. On every live path
        // a batch is fully reaped before the Streamer drops (normal: per-batch reap;
        // abandoned-poison: those reads complete into CQEs long before the reaper thread
        // exits and drops this), so no read is actually in flight at this point. The
        // ordering is belt-and-suspenders — it does NOT rely on io_uring teardown being a
        // synchronous drain (the kernel defers cancel-and-wait to a workqueue after
        // close()), only on not freeing first, which is strictly no worse than the
        // reverse. Rust runs this Drop body before dropping fields, so without ManuallyDrop
        // the arena would free first.
        // SAFETY: `ring` is never touched again; `ManuallyDrop::drop` runs exactly once
        // (single owner, no Clone).
        unsafe { ManuallyDrop::drop(&mut self.ring) };
        // SAFETY: `arena` came from `stage::alloc` and `new` refuses to build a `Streamer`
        // around a null one, so this is a live allocation; single owner, no Clone, so it
        // is freed exactly once. (`refresh_sink` borrows the arena's first word, so it is
        // freed with it.)
        unsafe { stage::free(self.arena) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_span_covers_worst_case_straddle() {
        // Any begin offset's superset fits in slot_span(len).
        for len in [1usize, 4095, 4096, 4097, 19_000_000] {
            for begin in [0usize, 1, 4095, 4096, 100_003] {
                let ab = begin & !(ALIGN - 1);
                let ae = (begin + len).next_multiple_of(ALIGN);
                assert!(ae - ab <= slot_span(len), "len={len} begin={begin}");
                assert_eq!(slot_span(len) % ALIGN, 0);
            }
        }
    }

    // The short-read guard's threshold arithmetic (the Rust-side logic; the reap only
    // compares `cqe.res` against it). Forcing a genuine mid-file short io_uring
    // completion is not unit-testable in this harness — O_DIRECT `open()` returns
    // EINVAL on tmpfs/overlayfs (the container test stage), and against a valid
    // snapshot a short read only ever occurs as trailing EOF padding, which the
    // guard deliberately tolerates. So test the threshold, not the completion.
    #[test]
    fn min_completion_covers_useful_window() {
        // Aligned begin: threshold is exactly the useful length.
        assert_eq!(min_completion(0, 100), 100);
        assert_eq!(min_completion(ALIGN, 4096), 4096);
        // Straddling begin: threshold includes the leading sub-block offset, so the
        // completion must reach past the pad into the useful bytes.
        assert_eq!(min_completion(1, 100), 101);
        assert_eq!(min_completion(4097, 4096), 1 + 4096);
        assert_eq!(min_completion(100_003, 10), (100_003 - 98_304 + 10) as u64);
        // The threshold never exceeds the aligned superset actually read.
        for len in [1usize, 4095, 4096, 4097, 1_000_000] {
            for begin in [0usize, 1, 4095, 4096, 100_003] {
                let ab = begin & !(ALIGN - 1);
                let superset = ((begin + len).next_multiple_of(ALIGN) - ab) as u64;
                assert!(
                    min_completion(begin, len) <= superset,
                    "len={len} begin={begin}"
                );
            }
        }
    }
}
