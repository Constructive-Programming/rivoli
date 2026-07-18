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
        pub fn rivoli_ring_wait(ring: *mut c_void, count: u32) -> i32;
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

/// A ring of in-flight reads. `entries` caps how many can be queued before a
/// `drain` — sized to a layer's cold-read count with margin.
pub struct Streamer {
    ring: *mut c_void,
    queued: u32,
}

impl Streamer {
    pub fn new(entries: u32) -> Result<Self> {
        let ring = unsafe { ffi::rivoli_ring_new(entries) };
        ensure!(!ring.is_null(), "io_uring_queue_init({entries}) failed");
        Ok(Self { ring, queued: 0 })
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
        self.queued += 1;
        Ok(begin - ab)
    }

    /// Submit all queued reads and block until every one completes; errors on the
    /// first failed read. No-op if nothing is queued.
    pub fn drain(&mut self) -> Result<()> {
        if self.queued == 0 {
            return Ok(());
        }
        let n = self.queued;
        // SAFETY: `ring` is live; exactly `n` reads were queued since the last drain.
        let e = unsafe { ffi::rivoli_ring_wait(self.ring, n) };
        self.queued = 0;
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
}
