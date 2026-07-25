//! io_uring O_DIRECT cold-expert streamer. A single NVMe read is latency-bound
//! (~4 GB/s here); io_uring keeps the queue full and the NVMe delivers ~5.8 GB/s
//! random (QD≥4). So a MoE layer submits all its cold reads at once and joins
//! once — folding the old mmap-warm + memcpy-fetch into one overlapped stream.
//!
//! The ring is the `io-uring` crate (talks to the io_uring syscalls directly — no
//! liburing system lib). This module owns the O_DIRECT alignment math (block-aligned
//! offset/length/buffer), the fds, and the VMM destination pointers; the two HIP ops
//! (pinned bounce arena, async H2D copy) are `rivoli_*` wrappers in `kernels/async.hip`.
//!
//! Two destination modes (chosen at `Streamer::new`, `queue`'s `dst` is the VMM slot
//! either way): BOUNCE (the default) reads into a pinned host arena then
//! `hipMemcpyAsync`s into VMM; DIRECT (`--direct-vmm-dma`) DMAs the read straight into
//! VMM. Bounce is the default AND a WORKAROUND for an amdgpu kernel bug (6.18.38-
//! gentoo, 2026-07-17) that EFAULTs on io_uring/O_DIRECT DMA into VMM device memory
//! (can't `get_user_pages` those pages; regression vs ≤6.18.35-r1). The EFAULT is gone
//! on 6.18.38 but bounce stays the default: reading weights from host-mapped VMM costs
//! ~40% on `mlp` (the system-vs-device read tax, docs/hip-apu-memory.md) — far more than
//! the H2D copy it saves. Repro: docs/probes/iouring_vmm.cpp (faults into VMM) vs the
//! iou_host probe (pinned host succeeds).
//!
//! SQPOLL is requested on the ring (own poller thread). Without it, submit is an
//! `io_uring_enter` in which the CALLING thread walks the SQEs and drives the
//! btrfs/blk-mq dispatch inline, serially (~2.96 ms/expert at 6 SQEs): the call
//! doesn't return promptly AND the batch reaches the device at queue depth 1 (2.53
//! GB/s) instead of the 6.69 GB/s the array delivers at P≥4. The poller takes the whole
//! SQ tail at once, so the batch is genuinely concurrent. Falls back to a plain ring
//! (the QD1 perf arm, still correct) if SQPOLL setup is refused.
//!
//! `rocm`-only (its sole consumer is the GPU decode pin).
#![cfg(feature = "rocm")]

use anyhow::{Result, ensure};
use io_uring::{IoUring, opcode, types};
use std::ffi::c_void;
use std::io;
use std::os::fd::RawFd;

/// O_DIRECT block alignment. 4 KiB is a safe superset of any real logical block
/// (512/4096) and matches the page/VMM granularity — offset, length, and buffer
/// must all be multiples of it.
pub const ALIGN: usize = 4096;

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
    ring: IoUring,
    queued: u32,
    /// Bounce mode (the default): reads land in a pinned host arena and are
    /// `hipMemcpy`d into VMM. False (`--direct-vmm-dma`) = DMA straight into the
    /// VMM slot.
    bounce: bool,
    /// Per-read pinned-bounce stride (bounce mode only): the largest aligned
    /// superset any single read may deliver. A `queue` whose superset exceeds this
    /// can't fit its bounce slot. Unused (0) in direct mode.
    span: usize,
    /// Pinned host arena, `entries * span` bytes (bounce mode only; null in direct):
    /// read slot `user_data` is `arena + user_data * span`.
    arena: *mut u8,
    /// Per-queued-read VMM destination + aligned read length (bounce mode only),
    /// indexed by the read's `user_data`. `reap` copies `nbytes` from the arena slot
    /// into `dst`. Built in queue order, cleared per batch (mirrors `min_res`).
    dst: Vec<*mut u8>,
    nbytes: Vec<u32>,
    /// Per-queued-read minimum completion length (sub-block offset + useful len),
    /// indexed by the read's `user_data`. `reap` compares the completion `res`
    /// against it so a real mid-file short read is caught while EOF-padding
    /// truncation is tolerated.
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
        // Own SQPOLL poller (sq_thread_idle 2000ms, re-armed on submit); fall back to a
        // plain ring if the kernel refuses SQPOLL (the QD1 perf arm, still correct).
        let ring = IoUring::builder()
            .setup_sqpoll(2000)
            .build(entries)
            .or_else(|_| IoUring::new(entries))?;

        let arena = if bounce {
            // SAFETY: no args; null on failure.
            let p = unsafe { ffi::rivoli_pinned_alloc(entries as u64 * span as u64) };
            ensure!(
                !p.is_null(),
                "bounce arena alloc failed (entries={entries}, {:.0} MiB pinned)",
                (entries as usize * span) as f64 / (1u64 << 20) as f64
            );
            p as *mut u8
        } else {
            std::ptr::null_mut()
        };

        Ok(Self {
            ring,
            queued: 0,
            bounce,
            span,
            arena,
            dst: Vec::with_capacity(entries as usize),
            nbytes: Vec::with_capacity(entries as usize),
            min_res: Vec::with_capacity(entries as usize),
        })
    }

    /// Queue an O_DIRECT read of `len` bytes at file offset `begin` (from `fd`)
    /// into `dst`. Reads the aligned superset `[align_down(begin), align_up(begin+
    /// len))`, so `dst` must be `ALIGN`-aligned and own at least `slot_span(len)`
    /// bytes. Returns the sub-block offset in `dst` where the useful `len` bytes
    /// land (i.e. the caller reads `dst.add(returned) .. +len`).
    ///
    /// # Safety
    /// `dst` must be `ALIGN`-aligned and valid for `slot_span(len)` writable bytes
    /// until this read's [`reap`](Self::reap) completes.
    pub unsafe fn queue(
        &mut self,
        fd: RawFd,
        begin: usize,
        len: usize,
        dst: *mut u8,
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
        let ud = self.queued;
        // BOUNCE reads into this slot's arena window; DIRECT reads straight into dst.
        let into = if self.bounce {
            // SAFETY: `ud < entries` (checked below via ensure on push failure); the
            // arena owns `entries * span` bytes, so `ud * span + nbytes <= arena len`.
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
            debug_assert_eq!(self.dst.len(), ud as usize);
            self.dst.push(dst);
            self.nbytes.push(nbytes as u32);
        }
        // The completion must deliver at least the useful window `[begin,begin+len)`
        // from the aligned start; a shorter read is mid-file truncation (checked in
        // `reap` against `min_res`). Trailing EOF padding beyond this is fine.
        debug_assert_eq!(self.min_res.len(), self.queued as usize);
        self.min_res.push(min_completion(begin, len));
        self.queued += 1;
        Ok(sub)
    }

    /// Submit the queued reads to the kernel WITHOUT waiting, so they start running
    /// on the NVMe/DMA side immediately. The following per-read [`reap`](Self::reap)
    /// calls collect the same completions; the bookkeeping is deliberately left intact
    /// for them. No-op if nothing is queued.
    ///
    /// Submitting here (rather than at reap time) starts the reads promptly and
    /// CONCURRENTLY — the whole batch reaches the device before any host work that
    /// follows, instead of dispatching serially at join time. The own-poller is what
    /// makes that concurrency real (see [`Streamer::new`]).
    pub fn submit(&self) -> Result<()> {
        if self.queued == 0 {
            return Ok(());
        }
        // With SQPOLL this wakes the poller if idle; otherwise it does the io_uring_enter
        // submit. Either way the already-pushed SQEs are handed to the kernel and the
        // CQEs are collected by the matching `reap` calls.
        self.ring
            .submit()
            .map_err(|e| anyhow::anyhow!("io_uring submit failed: {e}"))?;
        Ok(())
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
            self.ring
                .submit_and_wait(1)
                .map_err(|e| anyhow::anyhow!("io_uring wait failed: {e}"))?;
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
            // SAFETY: `dst[ud]` is a live VMM slot the pipeline keeps valid until this
            // read's signal fires; the arena slot holds the just-read bytes; `stream` is
            // live. Copies the full aligned `nbytes` (a trailing-EOF short read leaves
            // stale bytes only past the useful window, never read).
            let rc = unsafe {
                ffi::rivoli_memcpy_h2d_async(
                    self.dst[ud] as *mut c_void,
                    self.arena.add(ud * self.span) as *const c_void,
                    u64::from(self.nbytes[ud]),
                    stream,
                )
            };
            ensure!(rc == 0, "bounce H2D copy failed on slot {ud} (hip rc {rc})");
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

    /// Reset the per-batch bookkeeping after a full batch has been [`reap`]ed,
    /// readying the ring for the next layer's batch.
    pub fn reset_batch(&mut self) {
        self.queued = 0;
        self.dst.clear();
        self.nbytes.clear();
        self.min_res.clear();
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        if !self.arena.is_null() {
            // SAFETY: `arena` came from rivoli_pinned_alloc, freed exactly once.
            unsafe { ffi::rivoli_pinned_free(self.arena as *mut c_void) };
        }
        // `ring` (IoUring) tears itself down on drop.
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
