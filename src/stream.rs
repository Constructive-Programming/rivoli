//! io_uring O_DIRECT cold-expert streamer. A single NVMe read is latency-bound
//! (~4 GB/s here); io_uring keeps the queue full and the NVMe delivers ~16 GB/s
//! (QD≥4, `docs/probes/iouring_vmm.cpp`). So a MoE layer submits all its cold
//! reads at once, straight into the VMM slots, and joins once — folding the old
//! mmap-warm + memcpy-fetch into one overlapped DMA stream.
//!
//! Thin Rust owner over `kernels/stream.hip`'s liburing ring: this side does the
//! O_DIRECT alignment math (block-aligned offset/length/buffer) and owns the fds
//! and destination buffers.
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
        pub fn rivoli_ring_new(entries: u32) -> *mut c_void;
        pub fn rivoli_ring_read(
            ring: *mut c_void,
            fd: i32,
            buf: *mut c_void,
            nbytes: u32,
            off: u64,
            user_data: u64,
        ) -> i32;
        pub fn rivoli_ring_wait(ring: *mut c_void, count: u32, min_res: *const u64) -> i32;
        pub fn rivoli_ring_free(ring: *mut c_void);
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
/// `drain` — sized to a layer's cold-read count with margin.
pub struct Streamer {
    ring: *mut c_void,
    queued: u32,
    /// Per-queued-read minimum completion length (sub-block offset + useful len),
    /// indexed by the read's `user_data`. `drain` hands this to the shim so a real
    /// mid-file short read is caught while EOF-padding truncation is tolerated.
    min_res: Vec<u64>,
}

impl Streamer {
    pub fn new(entries: u32) -> Result<Self> {
        let ring = unsafe { ffi::rivoli_ring_new(entries) };
        ensure!(!ring.is_null(), "io_uring_queue_init({entries}) failed");
        Ok(Self {
            ring,
            queued: 0,
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
