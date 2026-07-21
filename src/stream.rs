//! io_uring O_DIRECT cold-expert streamer. A single NVMe read is latency-bound
//! (~4 GB/s here); io_uring keeps the queue full and the NVMe delivers ~5.8 GB/s
//! random (QD≥4). So a MoE layer submits all its cold reads at once and joins
//! once — folding the old mmap-warm + memcpy-fetch into one overlapped stream.
//!
//! Two destination modes (chosen at `Streamer::new`, `queue`'s `dst` is the VMM
//! slot either way): BOUNCE (the default) reads into a pinned host arena then
//! `hipMemcpy`s into VMM; DIRECT (`--direct-vmm-dma`) DMAs the read straight into
//! VMM. Bounce is the default AND a WORKAROUND for an amdgpu kernel bug (6.18.38-
//! gentoo, 2026-07-17) that EFAULTs on io_uring/O_DIRECT DMA into VMM device
//! memory (can't `get_user_pages` those pages; regression vs ≤6.18.35-r1). Revert
//! path + repro in `kernels/stream.hip`.
//!
//! Thin Rust owner over `kernels/stream.hip`'s liburing ring: this side does the
//! O_DIRECT alignment math (block-aligned offset/length/buffer) and owns the fds
//! and VMM destination pointers.
//!
//! `rocm`-only (its sole consumer is the GPU decode pin).
#![cfg(feature = "rocm")]

use anyhow::{Result, ensure};
use std::ffi::c_void;
use std::os::fd::RawFd;

/// O_DIRECT block alignment. 4 KiB is a safe superset of any real logical block
/// (512/4096) and matches the page/VMM granularity — offset, length, and buffer
/// must all be multiples of it.
pub const ALIGN: usize = 4096;

mod ffi {
    use std::ffi::c_void;
    unsafe extern "C" {
        pub fn rivoli_ring_new(
            entries: u32,
            span: u32,
            bounce: i32,
            sqpoll: i32,
            attach_fd: i32,
        ) -> *mut c_void;
        pub fn rivoli_ring_fd(ring: *mut c_void) -> i32;
        pub fn rivoli_ring_read(
            ring: *mut c_void,
            fd: i32,
            buf: *mut c_void,
            nbytes: u32,
            off: u64,
            user_data: u64,
        ) -> i32;
        pub fn rivoli_ring_submit(ring: *mut c_void) -> i32;
        pub fn rivoli_ring_wait(ring: *mut c_void, count: u32, min_res: *const u64) -> i32;
        pub fn rivoli_ring_free(ring: *mut c_void);
        pub fn rivoli_ring_timings(read_ns: *mut u64, copy_ns: *mut u64);
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

/// How a ring's submission path is driven — the single most consequential choice
/// about a ring on this box.
///
/// A plain ring's submit is an `io_uring_enter` in which the CALLING thread walks
/// the SQEs and drives the btrfs/blk-mq dispatch inline, one at a time. Measured
/// at ~2.96 ms per expert (6 SQEs). That costs twice over: the call does not
/// return promptly, AND the reads reach the device serially, so a 6-read batch
/// behaves like queue depth 1 (2.53 GB/s) instead of the 6.69 GB/s the array
/// delivers at P≥4. A poller thread fixes both — it takes the whole SQ tail at
/// once, so the batch is genuinely concurrent.
pub enum Sqpoll<'a> {
    /// No poller: submit is a syscall with inline, serial dispatch. Correct, and
    /// the right choice only for a ring whose reads have no batch-parallelism to
    /// win (or where a poller thread cannot be had).
    Off,
    /// This ring gets its own kernel submission-poller thread.
    Own,
    /// Share the given ring's poller thread (`IORING_SETUP_ATTACH_WQ`) rather than
    /// spinning up a second one. Falls back to [`Sqpoll::Own`], then
    /// [`Sqpoll::Off`], if the kernel refuses.
    SharedWith(&'a Streamer),
}

/// A ring of in-flight reads. `entries` caps how many can be queued before a
/// `drain` — sized to a layer's cold-read count with margin.
pub struct Streamer {
    ring: *mut c_void,
    queued: u32,
    /// Bounce mode (the default): reads land in a pinned host arena and are
    /// `hipMemcpy`d into VMM. False (`--direct-vmm-dma`) = DMA straight into the
    /// VMM slot.
    bounce: bool,
    /// Per-read pinned-bounce stride (bounce mode only): the largest aligned
    /// superset any single read may deliver. A `queue` whose superset exceeds this
    /// can't fit its bounce slot. Unused (0) in direct mode.
    span: usize,
    /// Per-queued-read minimum completion length (sub-block offset + useful len),
    /// indexed by the read's `user_data`. `drain` hands this to the shim so a real
    /// mid-file short read is caught while EOF-padding truncation is tolerated.
    min_res: Vec<u64>,
}

impl Streamer {
    /// `entries` = max in-flight reads; `span` = the largest aligned superset a
    /// single read may deliver (`slot_span` of the biggest projection tensor).
    /// `bounce` selects the destination path: true (the default) reads into an
    /// `entries * span` pinned host arena then `hipMemcpy`s into VMM (kernel-bug
    /// workaround); false (`--direct-vmm-dma`) DMAs straight into the VMM slot (no
    /// arena allocated). `sqpoll` selects the submission path — see [`Sqpoll`];
    /// every ring that submits a batch it wants run in PARALLEL wants a poller.
    pub fn new(entries: u32, span: usize, bounce: bool, sqpoll: Sqpoll<'_>) -> Result<Self> {
        let (want_poll, attach) = match sqpoll {
            Sqpoll::Off => (0, -1),
            Sqpoll::Own => (1, -1),
            // SAFETY: `other.ring` is live for `other`'s lifetime, which outlives
            // this call; the shim only reads its `ring_fd`.
            Sqpoll::SharedWith(other) => (1, unsafe { ffi::rivoli_ring_fd(other.ring) }),
        };
        let ring = unsafe {
            ffi::rivoli_ring_new(entries, span as u32, i32::from(bounce), want_poll, attach)
        };
        ensure!(
            !ring.is_null(),
            "ring init failed (entries={entries}, bounce={bounce}, {:.0} MiB pinned)",
            if bounce {
                (entries as usize * span) as f64 / (1u64 << 20) as f64
            } else {
                0.0
            }
        );
        Ok(Self {
            ring,
            queued: 0,
            bounce,
            span,
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
    /// until the next [`Streamer::drain`] completes.
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
        let r = unsafe {
            ffi::rivoli_ring_read(
                self.ring,
                fd,
                dst as *mut c_void,
                nbytes as u32,
                ab as u64,
                u64::from(self.queued),
            )
        };
        ensure!(
            r == 0,
            "io_uring SQ full at {} reads (raise ring entries)",
            self.queued
        );
        // The completion must deliver at least the useful window `[begin,begin+len)`
        // from the aligned start; a shorter read is mid-file truncation (checked in
        // `drain`). Trailing EOF padding beyond this is fine.
        debug_assert_eq!(self.min_res.len(), self.queued as usize);
        self.min_res.push(min_completion(begin, len));
        self.queued += 1;
        Ok(sub)
    }

    /// Submit the queued reads to the kernel WITHOUT waiting, so they start running
    /// on the NVMe/DMA side immediately. A later [`Streamer::drain`] reaps the same
    /// completions (its `submit_and_wait` then submits nothing new); the
    /// `queued`/`min_res` bookkeeping is deliberately left intact for that drain.
    /// No-op if nothing is queued.
    ///
    /// Both rings use this. The prefetch ring wants the call to RETURN promptly (its
    /// reads must overlap the current layer's compute). The demand ring wants the
    /// reads to START promptly and CONCURRENTLY — submitting here rather than letting
    /// `drain`'s `submit_and_wait` do it puts the whole batch on the device before any
    /// host work that follows, instead of dispatching it serially at join time. Under
    /// [`Sqpoll::Off`] neither property holds, which is the point of the poller.
    pub fn submit(&self) -> Result<()> {
        if self.queued == 0 {
            return Ok(());
        }
        // SAFETY: `ring` is live; submitting only hands the already-prepped SQEs to
        // the kernel — the CQEs are reaped by the matching `drain`.
        let r = unsafe { ffi::rivoli_ring_submit(self.ring) };
        ensure!(
            r >= 0,
            "io_uring submit failed: {}",
            std::io::Error::from_raw_os_error(-r)
        );
        Ok(())
    }

    /// Submit all queued reads and block until every one completes; errors on the
    /// first failed read. No-op if nothing is queued.
    pub fn drain(&mut self) -> Result<()> {
        if self.queued == 0 {
            return Ok(());
        }
        let n = self.queued;
        // SAFETY: `ring` is live; exactly `n` reads were queued since the last drain,
        // and `min_res` holds `n` entries indexed by their user_data (0..n).
        let e = unsafe { ffi::rivoli_ring_wait(self.ring, n, self.min_res.as_ptr()) };
        self.queued = 0;
        self.min_res.clear();
        ensure!(
            e == 0,
            "io_uring: {n} reads, first error {}",
            std::io::Error::from_raw_os_error(-e)
        );
        Ok(())
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        // SAFETY: `ring` came from rivoli_ring_new, freed exactly once.
        unsafe { ffi::rivoli_ring_free(self.ring) };
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

    // The short-read guard's threshold arithmetic (the Rust-side logic; the C shim
    // only compares `cqe.res` against it). Forcing a genuine mid-file short io_uring
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

/// Accumulated `(nvme_read_ns, bounce_copy_ns)` across every drain on every ring.
/// Splits the `fetch` PROFILE bucket into the two halves that want opposite fixes:
/// waiting on NVMe completions (latency/queue-depth bound) versus the pinned-host
/// -> device copy and its sync (bandwidth bound, ~2.7 GB/token).
pub fn ring_timings() -> (u64, u64) {
    let (mut r, mut c) = (0u64, 0u64);
    // SAFETY: both out-pointers are valid; the shim only writes two u64s.
    unsafe { ffi::rivoli_ring_timings(&mut r, &mut c) };
    (r, c)
}
