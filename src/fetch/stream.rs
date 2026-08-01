//! io_uring O_DIRECT cold-expert streamer. A single NVMe read is latency-bound
//! (~4 GB/s here); io_uring keeps the queue full and the NVMe delivers ~5.8 GB/s
//! random (QD≥4). So a MoE layer submits all its cold reads at once and joins
//! once — folding the old mmap-warm + memcpy-fetch into one overlapped stream.
//!
//! The ring is the `io-uring` crate (talks to the io_uring syscalls directly — no
//! liburing system lib). This module owns the O_DIRECT alignment math (block-aligned
//! offset/length/buffer), the fds, and the VMM destination pointers; the two BACKEND ops
//! (host staging arena, async H2D copy) are `rivoli_*` wrappers in `kernels/async.hip` under
//! `rocm` and `vk::Buf::staging` + `vk::copy_h2d_async` under `vulkan` — see [`stage`], the
//! whole of the difference.
//!
//! Two destination modes (chosen at `Streamer::new`, `queue`'s `dst` is the VMM slot
//! either way): BOUNCE (the default) reads into a host staging arena then
//! async-copies into VMM; DIRECT (`--direct-vmm-dma`) DMAs the read straight into
//! VMM. Bounce is the default AND a WORKAROUND for an amdgpu kernel bug (6.18.38-
//! gentoo, 2026-07-17) that EFAULTs on io_uring/O_DIRECT DMA into VMM device memory
//! (can't `get_user_pages` those pages; regression vs ≤6.18.35-r1). The EFAULT is gone
//! on 6.18.38 but bounce stays the default, and the reason is WRITE-side, not read-side:
//! DMA-ing into VMM device pages runs at 5.66 GB/s vs 12.4 GB/s into the pinned arena, so
//! DIRECT more than doubles the cost of a miss. Measured 2026-07-30, int3-vq/dense/lru
//! @512: marginal cost per missed expert 1239 us (bounce) -> 2709 us (DIRECT), 2.59 ->
//! 1.19 tok/s. The read side is UNCHANGED — `--direct-vmm-dma` only flips this module's
//! `bounce` flag and never touches pool allocation, so kernels read the same device-local
//! VMM either way: a zero-miss layer costs 1563 us (bounce) vs 1525 us (DIRECT), equal
//! within noise. (An earlier note here blamed a ~40% `mlp` read tax from host-mapped VMM;
//! that configuration is not what this flag produces.) Repro:
//! docs/measurement/probes/iouring_vmm.cpp (faults into VMM) vs the iou_host probe (pinned host).
//!
//! SQPOLL is requested on the ring (own poller thread). Without it, submit is an
//! `io_uring_enter` in which the CALLING thread walks the SQEs and drives the
//! btrfs/blk-mq dispatch inline, serially (~2.96 ms/expert at 6 SQEs): the call
//! doesn't return promptly AND the batch reaches the device at queue depth 1 (2.53
//! GB/s) instead of the 6.69 GB/s the array delivers at P≥4. The poller takes the whole
//! SQ tail at once, so the batch is genuinely concurrent. Falls back to a plain ring
//! (the QD1 perf arm, still correct) if SQPOLL setup is refused.
//!
//! Needs a backend (`rocm` or `vulkan`) — its sole consumer is the GPU decode pin. The
//! ring itself is backend-independent; only the two BOUNCE-mode staging ops differ, and
//! they are the whole of [`stage`] below.
#![cfg(any(feature = "rocm", feature = "vulkan"))]

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

/// The two BOUNCE-mode operations, per backend: allocate/free the staging arena, and move
/// one staged read into its pool slot. DIRECT mode uses neither — the read DMAs into the
/// slot and there is nothing to stage.
///
/// Nothing else in this file is backend-specific.
#[cfg(feature = "rocm")]
mod stage {
    use std::ffi::c_void;

    mod ffi {
        use std::ffi::c_void;
        unsafe extern "C" {
            /// Pinned host arena for the bounce path (kernels/async.hip). Null on failure.
            pub fn rivoli_pinned_alloc(bytes: u64) -> *mut c_void;
            pub fn rivoli_pinned_free(p: *mut c_void);
            /// Async H2D copy on `stream` (bounce slot → VMM slot). 0 ok, else negative.
            pub fn rivoli_memcpy_h2d_async(
                dst: *mut c_void,
                src: *const c_void,
                n: u64,
                stream: *mut c_void,
            ) -> i32;
        }
    }

    /// A HIP-PINNED host arena, which is what makes the copy below a DMA rather than a
    /// staged CPU memcpy. Null on failure.
    pub fn alloc(bytes: usize) -> *mut u8 {
        // SAFETY: no pointer args; null on failure.
        unsafe { ffi::rivoli_pinned_alloc(bytes as u64) as *mut u8 }
    }

    /// `_bytes` is unused — HIP tracks the pinned registration's size itself. It is in the
    /// signature so both backends' `free` are called identically; the Vulkan one needs the
    /// size because `std::alloc::dealloc` demands the original `Layout`.
    ///
    /// # Safety
    /// `p` came from [`alloc`] and is freed exactly once.
    pub unsafe fn free(p: *mut u8, _bytes: usize) {
        unsafe { ffi::rivoli_pinned_free(p as *mut c_void) };
    }

    /// ASYNC `hipMemcpyAsync` on the fetch stream: it returns before the bytes land, and
    /// the read's `Signal` (armed on the same stream) is what says they have. This is the
    /// op the load↔compute overlap is built on.
    ///
    /// # Safety
    /// `dst` owns `n` device bytes and stays valid until `stream`'s completion signal
    /// fires; `src` is a live arena slot holding `n` bytes; `stream` is a live handle.
    pub unsafe fn copy_to_slot(
        dst: *mut u8,
        src: *const u8,
        n: usize,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: the caller's contract, forwarded.
        let rc = unsafe {
            ffi::rivoli_memcpy_h2d_async(dst as *mut c_void, src as *const c_void, n as u64, stream)
        };
        if rc == 0 { Ok(()) } else { Err(format!("hip rc {rc}")) }
    }
}

/// The Vulkan half of [`stage`], and both operations are now the DMA-engine equivalents of
/// the HIP ones rather than host memory work.
///
/// # This was a synchronous CPU memcpy, and that was a violation of the design
///
/// Until Phase 4 increment 2, `copy_to_slot` here was `std::ptr::copy_nonoverlapping`: the
/// REAPER THREAD's CPU moving ~2.16 GB per token, synchronously, blocking the ring it was
/// supposed to be draining. It produced correct tokens, which is exactly why it survived a
/// merge — the engine's streaming design is not a tuning layer over the arithmetic, it IS
/// the architecture, and a backend that serialises fetch against compute does not implement
/// it (docs/investigations/vulkan-port.md).
///
/// What replaces it: the arena is a HOST-VISIBLE `vk::Buf` ([`crate::backend::vk::Buf::staging`], so
/// non-device-local where the device offers the choice — see below), and the move into the
/// pool slot is a `vkCmdCopyBuffer` recorded on the FETCH QUEUE. It returns before the bytes
/// land, and the read's `Signal` — armed by `asyncfetch.rs` on that same queue immediately
/// after — is what says they have. That is `rivoli_memcpy_h2d_async`'s contract, kept.
///
/// # The arena must NOT be device-local, and that is the same amdgpu bug bounce exists for
///
/// The arena is the destination of an O_DIRECT io_uring read, so the kernel must
/// `get_user_pages` it. That is precisely what amdgpu refused for device-local VMM pages
/// (this module's header: EFAULT, ≤6.18.35-r1 regression), which is why BOUNCE mode exists
/// at all. Allocating the bounce arena out of device-local memory would reintroduce the very
/// failure it works around, so [`crate::backend::vk::Buf::staging`] excludes `DEVICE_LOCAL` when the
/// device has any alternative — and warns when it does not.
///
/// DIRECT mode (`--direct-vmm-dma`) still skips the arena entirely by DMA-ing the read
/// straight into the pool mapping, and is untouched by any of this.
#[cfg(feature = "vulkan")]
mod stage {
    use crate::backend::vk::Buf;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::Mutex;

    /// Live staging arenas, keyed by the host base [`alloc`] handed out.
    ///
    /// The arena is a `Buf` (so that `vk::copy_h2d_async` can resolve its host pointer back
    /// to a `VkBuffer`), but `Streamer` holds only a `*mut u8` — the shape the HIP side's
    /// `hipHostMalloc` dictates and the shape io_uring needs. Something has to own the `Buf`
    /// in between, and a map keyed by base is the smallest thing that can: `free` takes the
    /// entry out and drops it, which releases the allocation exactly once.
    ///
    /// A `Mutex<HashMap>` for what is normally ONE entry looks heavy, and the alternative —
    /// a single `Mutex<Option<Buf>>` — is wrong: `cargo test` builds several `Streamer`s on
    /// parallel threads, and a one-slot cell would have the second arena evict the first
    /// while io_uring still had SQEs pointing into it. Touched twice per process per arena,
    /// never on the fetch path.
    static ARENAS: Mutex<Option<HashMap<usize, Buf>>> = Mutex::new(None);

    /// A HOST-VISIBLE device buffer whose mapping an O_DIRECT read may land in, and whose
    /// bytes a DMA copy may read. Null on failure — including a poisoned registry, because
    /// handing back memory nothing owns would leak it silently.
    pub fn alloc(bytes: usize) -> *mut u8 {
        let Ok(mut buf) = Buf::staging(bytes) else {
            return std::ptr::null_mut();
        };
        let host = buf.host_mut();
        // `vkMapMemory` guarantees `minMemoryMapAlignment`, which is page-sized on every
        // driver we have seen and is NOT required to be. An unaligned base makes every
        // O_DIRECT read fail EINVAL deep inside the reaper, so it is checked here rather
        // than assumed — the same guard `VmmBuf::new` carries for the pool.
        if !(host as usize).is_multiple_of(crate::backend::vk::O_DIRECT_ALIGN) {
            tracing::error!(
                "staging arena mapping {host:?} is not {}-byte aligned, so io_uring O_DIRECT \
                 reads into it would fail with EINVAL",
                crate::backend::vk::O_DIRECT_ALIGN
            );
            return std::ptr::null_mut();
        }
        let Ok(mut reg) = ARENAS.lock() else {
            return std::ptr::null_mut();
        };
        reg.get_or_insert_with(HashMap::new).insert(host as usize, buf);
        host
    }

    /// # Safety
    /// `p` came from [`alloc`]; freed exactly once. `bytes` is unused — the `Buf` knows its
    /// own size — and is in the signature so both backends' `free` are called identically.
    pub unsafe fn free(p: *mut u8, _bytes: usize) {
        // Taken out of the map and dropped OUTSIDE the guard: `Buf::drop` flushes the
        // device (`Gpu::sync`), and holding an unrelated lock across a device join is how
        // a lock-ordering deadlock gets built.
        let taken = ARENAS
            .lock()
            .ok()
            .and_then(|mut reg| reg.as_mut().and_then(|m| m.remove(&(p as usize))));
        drop(taken);
    }

    /// ASYNC device copy of the staged read into its pool slot, on the FETCH queue. Returns
    /// BEFORE the bytes land; the caller's `Signal`, armed on the same queue next, is what
    /// says they have.
    ///
    /// `stream` is accepted for signature parity and checked rather than used: the queue is
    /// [`crate::backend::vk::Q::Fetch`] by construction here, and a caller that passed some other
    /// stream would be describing a copy this function is not making.
    ///
    /// # Safety
    /// `dst` owns `n` writable bytes in the pool's host mapping and stays valid until the
    /// copy's signal fires; `src` is a live arena slot holding `n` bytes; the two do not
    /// overlap (distinct allocations).
    pub unsafe fn copy_to_slot(
        dst: *mut u8,
        src: *const u8,
        n: usize,
        _stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: the caller's contract, forwarded — `n` readable bytes at `src`, `n`
        // writable at `dst`, non-overlapping, both inside live `Buf` mappings.
        unsafe { crate::backend::vk::copy_h2d_async(dst, src, n) }.map_err(|e| format!("{e:#}"))
    }
}

/// Destination bytes needed to O_DIRECT-read `len` bytes starting at an arbitrary
/// file offset: the aligned superset, upper-bounded independent of the offset so a
/// reused slot can be sized once. `align_up(len) + ALIGN` covers the worst-case
/// straddle (up to `ALIGN-1` leading pad + trailing round-up).
pub fn slot_span(len: usize) -> usize {
    len.div_ceil(ALIGN) * ALIGN + ALIGN
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
pub struct Streamer {
    /// `ManuallyDrop` so `Drop` can tear the ring down BEFORE freeing the arena the
    /// ring's SQEs point into (Rust would otherwise run the `Drop` body — the arena
    /// free — before dropping this field). Access is transparent via `Deref`.
    ring: ManuallyDrop<IoUring>,
    /// Ring/arena capacity in staging slots. A batch is bounded by this (see `queue`), and
    /// the fetcher needs one timeline per slot.
    entries: u32,
    queued: u32,
    /// Bounce mode (the default): reads land in a pinned host arena and are
    /// `hipMemcpy`d into VMM. False (`--direct-vmm-dma`) = DMA straight into the
    /// VMM slot.
    bounce: bool,
    /// Per-read pinned-bounce stride (bounce mode only): the largest aligned
    /// superset any single read may deliver. A `queue` whose superset exceeds this
    /// can't fit its bounce slot. Unused (0) in direct mode.
    span: usize,
    /// Staging arena, `entries * span` bytes (bounce mode only; null in direct): read slot
    /// `user_data` is `arena + user_data * span`. HIP-pinned under `rocm`, plain
    /// `ALIGN`-aligned host memory under `vulkan` — see [`stage`].
    arena: *mut u8,
    /// `arena`'s byte size, kept because `stage::free` needs it (the Vulkan side's
    /// `dealloc` requires the original `Layout`). Zero in direct mode.
    arena_bytes: usize,
    /// Per-SLOT VMM destination + aligned read length (bounce mode only). `reap` copies
    /// `nbytes` from the arena slot into `dst`. Indexed by staging slot — which is the
    /// read's `user_data` — and NOT cleared per batch: a slot's lifetime is owned by its
    /// [`Ticket`](crate::fetch::asyncfetch::Ticket) now, not by the batch that happened to use it.
    dst: Vec<*mut u8>,
    nbytes: Vec<u32>,
    /// Per-SLOT minimum completion length (sub-block offset + useful len). `reap` compares
    /// the completion `res` against it so a real mid-file short read is caught while
    /// EOF-padding truncation is tolerated.
    min_res: Vec<u64>,
}

// SAFETY: the ring + arena are exclusively owned by whoever holds the `Streamer`.
// io_uring rings are NOT thread-safe, so this only asserts the handle can MOVE across
// threads — `AsyncFetch` moves it once into its reaper thread, which is then the sole
// accessor. Never share a `&Streamer`/`&mut Streamer` across threads.
unsafe impl Send for Streamer {}

impl Streamer {
    /// `entries` = max in-flight reads; `span` = the largest aligned superset a
    /// single read may deliver (`slot_span` of the biggest projection tensor).
    /// `bounce` selects the destination path: true (the default) reads into an
    /// `entries * span` pinned host arena then `hipMemcpy`s into VMM (kernel-bug
    /// workaround); false (`--direct-vmm-dma`) DMAs straight into the VMM slot (no
    /// arena allocated).
    pub fn new(entries: u32, span: usize, bounce: bool) -> Result<Self> {
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

        let arena_bytes = if bounce { entries as usize * span } else { 0 };
        let arena = if bounce {
            let p = stage::alloc(arena_bytes);
            ensure!(
                !p.is_null(),
                "bounce arena alloc failed (entries={entries}, {:.0} MiB)",
                arena_bytes as f64 / (1u64 << 20) as f64
            );
            p
        } else {
            std::ptr::null_mut()
        };

        Ok(Self {
            ring: ManuallyDrop::new(ring),
            entries,
            queued: 0,
            bounce,
            span,
            arena,
            arena_bytes,
            dst: vec![std::ptr::null_mut(); entries as usize],
            nbytes: vec![0; entries as usize],
            min_res: vec![0; entries as usize],
        })
    }

    /// Queue an O_DIRECT read of `len` bytes at file offset `begin` (from `fd`)
    /// into `dst`, staged through bounce arena slot `slot`. Reads the aligned superset
    /// `[align_down(begin), align_up(begin+len))`, so `dst` must be `ALIGN`-aligned and
    /// own at least `slot_span(len)` bytes. Returns the sub-block offset in `dst` where
    /// the useful `len` bytes land (i.e. the caller reads `dst.add(returned) .. +len`).
    ///
    /// **`slot` is chosen by the caller, not by queue order.** It used to be `queued++`,
    /// which made a slot's lifetime the batch's lifetime — see [`reset_batch`].
    ///
    /// # Safety
    /// `dst` must be `ALIGN`-aligned and valid for `slot_span(len)` writable bytes
    /// until this read's [`reap`](Self::reap) completes. `slot` must not be in use by a
    /// read whose bounce copy has not yet retired (the caller's ticket gate).
    pub unsafe fn queue(
        &mut self,
        fd: RawFd,
        begin: usize,
        len: usize,
        dst: *mut u8,
        slot: u32,
    ) -> Result<usize> {
        debug_assert_eq!(
            dst as usize % ALIGN,
            0,
            "O_DIRECT dst must be block-aligned"
        );
        let ab = begin & !(ALIGN - 1);
        let ae = (begin + len).div_ceil(ALIGN) * ALIGN;
        let nbytes = ae - ab;
        ensure!(
            !self.bounce || nbytes <= self.span,
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
        // BOUNCE reads into this slot's arena window; DIRECT reads straight into dst.
        let into = if self.bounce {
            // SAFETY: `ud < entries` (checked above), and the arena is `entries*span` with
            // every read's `nbytes <= span`, so this slot's window is owned and in bounds.
            unsafe { self.arena.add(ud as usize * self.span) }
        } else {
            dst
        };
        let read = opcode::Read::new(types::Fd(fd), into, nbytes as u32)
            .offset(ab as u64)
            .build()
            .user_data(u64::from(ud));
        // SAFETY: `into` is valid for `nbytes` writable bytes until this read's reap
        // (caller's `dst` contract, or our arena slot); the SQE references it by raw
        // pointer, so it must outlive the completion — it does.
        let pushed = unsafe { self.ring.submission().push(&read) };
        ensure!(
            pushed.is_ok(),
            "io_uring SQ full at {} reads (raise ring entries)",
            self.queued
        );
        if self.bounce {
            self.dst[ud as usize] = dst;
            self.nbytes[ud as usize] = nbytes as u32;
        }
        // The completion must deliver at least the useful window `[begin,begin+len)`
        // from the aligned start; a shorter read is mid-file truncation (checked in
        // `reap` against `min_res`). Trailing EOF padding beyond this is fine.
        self.min_res[ud as usize] = min_completion(begin, len);
        self.queued += 1;
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

    /// Per-read async reap: block for the NEXT read to complete, kick its bounce→slot
    /// copy on `stream`, and return the completed read's `user_data` (the index into
    /// the batch). The caller reaps exactly `queued` times, then
    /// [`reset_batch`](Self::reset_batch). Reads run concurrently on the NVMe;
    /// completions arrive in device order, each resolving its own expert.
    ///
    /// # Safety
    /// `stream` is a live HipStream handle; the copied read's `dst` slot must stay
    /// valid until that stream's completion signal fires.
    pub unsafe fn reap(&mut self, stream: *mut c_void) -> Result<usize> {
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
        if self.bounce {
            // SAFETY: `dst[ud]` is a live pool slot the pipeline keeps valid until this
            // read's signal fires; the arena slot holds the just-read bytes; `stream` is
            // live. Copies the full aligned `nbytes` (a trailing-EOF short read leaves
            // stale bytes only past the useful window, never read).
            let r = unsafe {
                stage::copy_to_slot(
                    self.dst[ud],
                    self.arena.add(ud * self.span),
                    self.nbytes[ud] as usize,
                    stream,
                )
            };
            if let Err(e) = r {
                anyhow::bail!("bounce staging copy failed on slot {ud} ({e})");
            }
        }
        Ok(ud)
    }

    /// Take one ready completion (if any), advancing the CQ head. Returns
    /// `(res, user_data)`; the guard's `sync` refreshes visible completions and its
    /// drop writes the consumed head back to the kernel.
    fn next_cqe(&mut self) -> Option<(i32, u64)> {
        let mut cq = self.ring.completion();
        cq.sync();
        cq.next().map(|c| (c.result(), c.user_data()))
    }

    /// Staging-slot count — the ring's capacity, and the number of per-slot timelines the
    /// fetcher needs.
    pub fn entries(&self) -> u32 {
        self.entries
    }

    /// Reset the SQ occupancy counter after a full batch has been [`reap`]ed, readying the
    /// ring for the next layer's batch.
    ///
    /// It used to clear `dst`/`nbytes`/`min_res` too, which silently recycled every staging
    /// slot — an integer reset with no relationship to whether the bounce copy OUT of those
    /// slots had retired. That was safe only because every demand read happens to be awaited
    /// inside its issuing layer: an emergent property of the consumer, written nowhere and
    /// enforced nowhere, and the reason the first speculative preloader corrupted. Slot reuse
    /// is now gated on the slot's timeline in `AsyncFetch::take_slot`.
    pub fn reset_batch(&mut self) {
        self.queued = 0;
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
        if !self.arena.is_null() {
            // SAFETY: `arena` came from `stage::alloc(self.arena_bytes)`, freed once.
            unsafe { stage::free(self.arena, self.arena_bytes) };
        }
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
                let ae = (begin + len).div_ceil(ALIGN) * ALIGN;
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
                let superset = ((begin + len).div_ceil(ALIGN) * ALIGN - ab) as u64;
                assert!(
                    min_completion(begin, len) <= superset,
                    "len={len} begin={begin}"
                );
            }
        }
    }
}
