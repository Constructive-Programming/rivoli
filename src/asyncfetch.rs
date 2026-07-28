#![cfg(feature = "rocm")]
//! Per-expert async loads: the io_uring→future adapter under the expert stream.
//!
//! A reaper thread owns the demand ring and a dedicated fetch [`HipStream`]. Per
//! MoE layer the pipeline hands it a batch of cold reads; it queues+submits them
//! (so they run concurrently on the NVMe), then reaps completions one at a time.
//! Because [`Streamer::reap`] already kicks each read's bounce→slot copy on the
//! fetch stream, the reaper just arms that read's [`Signal`] on the same stream —
//! so `load(e)` resolves when the COPY lands, not merely the NVMe read. A cache hit
//! never enters here; its load is [`Signal::ready`].
//!
//! This is the whole concurrency surface: the `StreamExt` pipeline above stays
//! single-threaded on the decode loop, the reaper blocks off-thread, and the two
//! meet only through `Signal` wakers.

use crate::gpustream::{HipStream, Signal};
use crate::stream::Streamer;
use anyhow::{Result, anyhow};
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

/// One cold read: file range → VMM slot. `dst` is a device slot pointer valid
/// across threads (device memory the ring DMAs into; never CPU-dereferenced here).
pub struct ReadSpec {
    pub fd: RawFd,
    pub begin: usize,
    pub len: usize,
    pub dst: *mut u8,
}
// SAFETY: `dst` is only handed to io_uring / hipMemcpyAsync on the reaper thread,
// never dereferenced on the CPU; `fd` is a plain descriptor.
unsafe impl Send for ReadSpec {}

/// A layer's demand batch: the reads plus one [`Signal`] per read (index =
/// io_uring `user_data`) that the reaper resolves as each copy lands.
struct ReapJob {
    reads: Vec<ReadSpec>,
    signals: Vec<Signal>,
}

/// Owns the demand ring + fetch stream on a reaper thread; services one [`ReapJob`]
/// per MoE layer. Reads submitted via [`submit`](Self::submit) resolve their
/// per-read [`Signal`] when loaded.
pub struct AsyncFetch {
    tx: Option<Sender<ReapJob>>,
    reaper: Option<JoinHandle<()>>,
    /// Accumulated reaper wall (queue→submit→reap all misses), off the main thread —
    /// the true fetch cost the overlap hides. The profile reads it against the MoE
    /// wall to report how much fetch was buried behind compute.
    fetch_ns: Arc<AtomicU64>,
    /// Accumulated time the reaper spent BLOCKED IN `io_uring` completions — the
    /// `reap` loop only, excluding the queue/submit syscalls around it. Measured at
    /// the ring rather than inferred: the profile's old `io-wait` was
    /// `moe_wall - compute_gpu`, a host clock minus a GPU clock, which understated it.
    ///
    /// This runs on the reaper thread, so it OVERLAPS the decode thread's wall and is
    /// not a share of it. That is the point — see `ProfileSummary`'s class line.
    io_wait_ns: Arc<AtomicU64>,
    /// Set once by the reaper on any fetch error: the ring is left dirty by a
    /// mid-batch bail, so it's abandoned rather than reused (reusing it would index
    /// stale `user_data` → C-side OOB / signal-index panic). `submit` then fails
    /// fast so the decode returns a clean `Err` instead of streaming garbage.
    poisoned: Arc<AtomicBool>,
}

impl AsyncFetch {
    /// Take ownership of the demand `streamer` and spawn the reaper with its own
    /// fetch stream.
    pub fn new(streamer: Streamer) -> Result<Self> {
        let fetch = HipStream::new()?;
        let (tx, rx) = channel::<ReapJob>();
        let fetch_ns = Arc::new(AtomicU64::new(0));
        let io_wait_ns = Arc::new(AtomicU64::new(0));
        let poisoned = Arc::new(AtomicBool::new(false));
        let fn_reaper = fetch_ns.clone();
        let io_reaper = io_wait_ns.clone();
        let poison_reaper = poisoned.clone();
        let reaper = std::thread::Builder::new()
            .name("rivoli-reaper".into())
            .spawn(move || reaper_loop(streamer, fetch, rx, fn_reaper, io_reaper, poison_reaper))?;
        Ok(Self {
            tx: Some(tx),
            reaper: Some(reaper),
            fetch_ns,
            io_wait_ns,
            poisoned,
        })
    }

    /// Accumulated reaper fetch wall in ns (across all layers so far).
    pub fn fetch_ns(&self) -> u64 {
        self.fetch_ns.load(Ordering::Relaxed)
    }

    /// Accumulated ns the reaper spent blocked in `io_uring` completions — the measured
    /// io-wait, taken at the ring. Off-thread, so it overlaps the decode wall.
    pub fn io_wait_ns(&self) -> u64 {
        self.io_wait_ns.load(Ordering::Relaxed)
    }

    /// Submit a layer's cold reads; returns one pending [`Signal`] per read, in the
    /// same order (index = `user_data`). Reads run concurrently on the NVMe; each
    /// signal resolves when its bounce copy has landed on the fetch stream. Empty
    /// `reads` returns an empty vec (an all-hit layer).
    pub fn submit(&self, reads: Vec<ReadSpec>) -> Result<Vec<Signal>> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(anyhow!("AsyncFetch poisoned by an earlier reaper error"));
        }
        let signals: Vec<Signal> = reads.iter().map(|_| Signal::pending()).collect();
        if reads.is_empty() {
            return Ok(signals);
        }
        let job = ReapJob {
            reads,
            signals: signals.clone(),
        };
        self.tx
            .as_ref()
            .ok_or_else(|| anyhow!("AsyncFetch closed"))?
            .send(job)
            .map_err(|_| anyhow!("reaper thread gone"))?;
        Ok(signals)
    }
}

impl Drop for AsyncFetch {
    fn drop(&mut self) {
        // Close the channel → `reaper_loop`'s `for job in rx` ends → thread exits.
        self.tx.take();
        if let Some(h) = self.reaper.take() {
            let _ = h.join();
        }
    }
}

/// Reaper thread: one job per layer until the channel closes. Times each job's wall
/// into `fetch_ns` — the fetch cost the main-thread compute overlaps.
fn reaper_loop(
    mut streamer: Streamer,
    fetch: HipStream,
    rx: Receiver<ReapJob>,
    fetch_ns: Arc<AtomicU64>,
    io_wait_ns: Arc<AtomicU64>,
    poisoned: Arc<AtomicBool>,
) {
    for job in rx {
        // Once poisoned, the ring is dirty (a prior job bailed mid-batch, leaving
        // undrained CQEs and stale queued/min_res). Touching it again would index a
        // stale user_data → C-side OOB or a signals[u] panic that kills this thread
        // and hangs every later awaiter. So don't: just resolve so nothing hangs (the
        // slots hold stale bytes, which the forward's finiteness guard trips).
        if poisoned.load(Ordering::Acquire) {
            for s in &job.signals {
                s.resolve();
            }
            continue;
        }
        let t = std::time::Instant::now();
        let r = run_job(&mut streamer, &fetch, &job, &io_wait_ns);
        fetch_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Err(e) = r {
            // Fatal fetch error: poison (abandon the dirty ring for all later jobs),
            // log, and resolve this job's signals so no awaiter hangs.
            tracing::error!("reaper: {e:#}");
            poisoned.store(true, Ordering::Release);
            for s in &job.signals {
                s.resolve();
            }
        }
    }
}

fn run_job(
    streamer: &mut Streamer,
    fetch: &HipStream,
    job: &ReapJob,
    io_wait_ns: &AtomicU64,
) -> Result<()> {
    for r in &job.reads {
        // SAFETY: `dst` is an ALIGN-aligned device slot the pipeline keeps live until
        // this read's signal resolves; the VQ blocks are VQ_ALIGN-aligned so the
        // returned sub-offset is 0 (the slot start IS the block).
        let sub = unsafe { streamer.queue(r.fd, r.begin, r.len, r.dst)? };
        debug_assert_eq!(
            sub, 0,
            "VQ expert read must be block-aligned (sub-offset 0)"
        );
    }
    streamer.submit()?;
    // Everything above is CPU (building and submitting SQEs); everything in the loop
    // below is the thread parked in the kernel waiting for NVMe. Only the latter is
    // io-wait, which is why the clock starts here and not at the top of `run_job`.
    let t_io = std::time::Instant::now();
    for _ in 0..job.reads.len() {
        // SAFETY: `fetch` is a live stream; `reap` kicks this read's copy on it and
        // returns the completed read's user_data.
        let u = unsafe { streamer.reap(fetch.raw())? };
        // Resolve when the copy (enqueued by `reap`) lands on the fetch stream.
        job.signals[u].arm_on(fetch)?;
    }
    let e_io = std::time::Instant::now();
    io_wait_ns.fetch_add(e_io.duration_since(t_io).as_nanos() as u64, Ordering::Relaxed);
    // Emitted on the REAPER thread against the same shared anchor as the decode
    // thread's spans, so a trace viewer draws them on one timeline. This is the pair
    // whose overlap the whole streaming design is a bet on: io-wait bars should sit
    // underneath the decode thread's gpu-wait bars, not beside them.
    crate::telemetry::spans::record("io-wait/uring-reap", "reaper", t_io, e_io);
    streamer.reset_batch();
    Ok(())
}
