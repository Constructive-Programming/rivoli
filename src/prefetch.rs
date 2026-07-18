//! Background page-cache warmer — the M4 streaming feed's overlap engine. The
//! decode thread's cold-expert fetch is a synchronous NVMe read (~86 ms of the
//! per-token time in the M3 measurement); this moves that read OFF the critical
//! path. A worker thread faults predicted cold-expert mmap pages into the OS page
//! cache ahead of need, so when the decode thread later `copy_in`s them to the
//! device it reads warm RAM, not cold disk.
//!
//! It does NO HIP — that stays on the decode thread (the single-HIP-thread rule).
//! It only touches host mmap bytes. The predicted set is the previous token's
//! routing (MoE selection is stable token-to-token), fed at each token's start.
//!
//! `rocm`-only (its sole consumer is the GPU engine).
#![cfg(feature = "rocm")]

use crossbeam_channel::{Sender, unbounded};
use std::thread::JoinHandle;

/// A batch of `(addr, len)` read-only mmap ranges to fault into the page cache.
type Batch = Vec<(usize, usize)>;

pub struct Prefetcher {
    tx: Option<Sender<Batch>>,
    handle: Option<JoinHandle<()>>,
}

impl Prefetcher {
    pub fn new() -> Self {
        let (tx, rx) = unbounded::<Batch>();
        let handle = std::thread::spawn(move || {
            let mut sink = 0u8;
            // Drain batches until the sender is dropped (Drop closes the channel).
            while let Ok(batch) = rx.recv() {
                for (addr, len) in batch {
                    // SAFETY: each (addr,len) is a read-only slice of the
                    // Snapshot's mmap, which outlives this thread (Prefetcher's
                    // Drop joins here before the Snapshot unmaps). Reading one
                    // byte per 4 KiB page faults the whole range into the page
                    // cache; the read is racy-safe (mmap is read-only, shared).
                    let p = addr as *const u8;
                    let mut off = 0usize;
                    while off < len {
                        sink = sink.wrapping_add(unsafe { *p.add(off) });
                        off += 4096;
                    }
                }
            }
            std::hint::black_box(sink); // keep the touches from being optimized out
        });
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Queue mmap ranges to warm. Non-blocking, best-effort (a full/closed queue
    /// just means the decode thread pays the cold read itself — never wrong).
    pub fn warm(&self, ranges: Batch) {
        if ranges.is_empty() {
            return;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(ranges);
        }
    }
}

impl Default for Prefetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Close the channel so the worker exits after draining, then join it
        // BEFORE the Snapshot it reads can unmap (the engine drops the
        // prefetcher while still holding the snapshot borrow).
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
