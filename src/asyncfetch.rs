#![cfg(any(feature = "rocm", feature = "vulkan"))]
//! Per-expert async loads: the io_uring→future adapter under the expert stream.
//!
//! Backend-independent: the fetch stream and the [`Signal`] both come from
//! [`crate::backend`], and it is a REAL dedicated stream on both — a `hipStream_t` under
//! `rocm`, its own `VkQueue` with its own command-buffer ring and timeline under `vulkan`.
//! Measured overlap: 96% of fetch hidden on ROCm, 97% on Vulkan. (Increment 1 of the Vulkan
//! port ran every stream on one queue and hid 0%; docs/VULKAN.md, "Increment 2: measured".)
//!
//! `Stream::fetch()` rather than `Stream::new()` at the construction below, and that is not
//! cosmetic: the Vulkan side maps the handle onto one of three queues and cannot infer which
//! from context, so the ROLE is named where it is known.
//!
//! A reaper thread owns the demand ring and a dedicated fetch [`Stream`]. Per
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

use crate::backend::{Signal, Stream, Timeline};
use crate::stream::Streamer;
use anyhow::{Result, anyhow};
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

/// One cold read: file range → pool slot. `dst` is the DMA TARGET for that slot, valid
/// across threads and never CPU-dereferenced here — under HIP the pool's unified pointer,
/// under Vulkan its host mapping (`ArenaPool::host_ptr`, not `ptr`).
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
    /// `tickets[i].value` is what the reaper signals on `tickets[i]`'s timeline once read
    /// `i`'s copy has been enqueued on the fetch stream. Assigned on the DECODE thread at
    /// submit, which is the whole point: a consumer can enqueue its wait before the reaper
    /// has even seen the completion.
    tickets: Vec<Ticket>,
}

/// A promise that some data will be present, redeemable as a DEVICE-SIDE wait.
///
/// This replaces the `hit: Vec<bool>` that used to tell `gpu.rs` whether to await. That
/// mask was a second, host-side encoding of "is this expert's data ready?", and when it
/// disagreed with the Signal it silently won — `gpu.rs` launches every `hit` expert with no
/// wait at all, so a mask that said "ready" made the kernel read unwritten memory. A ticket
/// cannot disagree with anything: it IS the dependency, and the only way to launch is to
/// enqueue its wait.
///
/// A resident expert carries [`Ticket::RESIDENT`] — value 0, which every timeline starts at,
/// so its wait is satisfied on arrival. Resident, missing and in-flight therefore take ONE
/// code path with no branch for anyone to get wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket {
    /// Which staging slot's timeline carries this dependency.
    pub slot: u16,
    /// The value that timeline reaches once the data has landed.
    pub value: u64,
}

impl Ticket {
    /// Data that is already present. Timelines start at 0, so waiting on 0 is free.
    pub const RESIDENT: Ticket = Ticket { slot: 0, value: 0 };

    #[inline]
    pub fn is_resident(&self) -> bool {
        self.value == 0
    }
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
    /// One timeline per staging slot, shared with the reaper. Per-SLOT rather than one for
    /// the whole stream because a ticket has to be known at submit time: read `i` of a batch
    /// deterministically lands in slot `i`, so its (slot, value) is computable on the decode
    /// thread before the read has been queued, let alone completed.
    slot_tl: Arc<Vec<Timeline>>,
    /// Next value to hand out per slot. Decode-thread only.
    slot_next: Vec<u64>,
}

impl AsyncFetch {
    /// Take ownership of the demand `streamer` and spawn the reaper with its own
    /// fetch stream.
    pub fn new(streamer: Streamer) -> Result<Self> {
        let fetch = Stream::fetch()?;
        let nslots = streamer.entries() as usize;
        let mut tls = Vec::with_capacity(nslots);
        for _ in 0..nslots {
            tls.push(Timeline::new()?);
        }
        let slot_tl = Arc::new(tls);
        let tl_reaper = slot_tl.clone();
        let (tx, rx) = channel::<ReapJob>();
        let fetch_ns = Arc::new(AtomicU64::new(0));
        let io_wait_ns = Arc::new(AtomicU64::new(0));
        let poisoned = Arc::new(AtomicBool::new(false));
        let fn_reaper = fetch_ns.clone();
        let io_reaper = io_wait_ns.clone();
        let poison_reaper = poisoned.clone();
        let reaper = std::thread::Builder::new()
            .name("rivoli-reaper".into())
            .spawn(move || {
                reaper_loop(streamer, fetch, rx, fn_reaper, io_reaper, poison_reaper, tl_reaper)
            })?;
        Ok(Self {
            tx: Some(tx),
            reaper: Some(reaper),
            fetch_ns,
            io_wait_ns,
            poisoned,
            slot_tl,
            slot_next: vec![0; nslots],
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
    pub fn submit(&mut self, reads: Vec<ReadSpec>) -> Result<(Vec<Signal>, Vec<Ticket>)> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(anyhow!("AsyncFetch poisoned by an earlier reaper error"));
        }
        let signals: Vec<Signal> = reads.iter().map(|_| Signal::pending()).collect();
        if reads.is_empty() {
            return Ok((signals, Vec::new()));
        }
        // Read `i` of this batch lands in staging slot `i` — `Streamer::queue` assigns
        // `ud = queued++` in exactly this order — so the ticket is computable here, on the
        // decode thread, before the reaper has touched anything. That is what lets a
        // consumer enqueue its wait ahead of the producer (INV-4).
        let tickets: Vec<Ticket> = reads
            .iter()
            .enumerate()
            .map(|(i, _)| {
                self.slot_next[i] += 1;
                Ticket { slot: i as u16, value: self.slot_next[i] }
            })
            .collect();
        let job = ReapJob {
            reads,
            signals: signals.clone(),
            tickets: tickets.clone(),
        };
        self.tx
            .as_ref()
            .ok_or_else(|| anyhow!("AsyncFetch closed"))?
            .send(job)
            .map_err(|_| anyhow!("reaper thread gone"))?;
        Ok((signals, tickets))
    }
}

impl AsyncFetch {
    /// Enqueue a DEVICE-SIDE wait for `t` on `stream_raw`. The only way to consume a
    /// ticket, so there is no path from "I have data" to "I launched" that skips the
    /// dependency — which is precisely the gap the `hit` mask left open.
    ///
    /// A resident ticket is value 0 and every timeline starts there, so the wait would be
    /// satisfied immediately; it is skipped rather than enqueued because ~7 of 9 experts per
    /// layer are resident and a no-op packet each is pure queue traffic. That is a LOCAL
    /// shortcut inside the one function that consumes tickets, not a contract a caller can
    /// get wrong.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn wait(&self, t: Ticket, stream_raw: *mut std::ffi::c_void) -> Result<()> {
        if t.is_resident() {
            return Ok(());
        }
        self.slot_tl[t.slot as usize].wait(stream_raw, t.value)
    }

    /// Has this ticket's data landed? Used to recycle staging slots by TIMELINE VALUE
    /// rather than by `reset_batch`'s blind integer reset — the latter recycled a slot with
    /// no relationship to whether the copy out of it had retired, which was safe only
    /// because every read happened to be awaited inside its own layer. That was an emergent
    /// property of the consumer, written nowhere and enforced nowhere.
    pub fn landed(&self, t: Ticket) -> bool {
        t.is_resident() || self.slot_tl[t.slot as usize].completed() >= t.value
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
    fetch: Stream,
    rx: Receiver<ReapJob>,
    fetch_ns: Arc<AtomicU64>,
    io_wait_ns: Arc<AtomicU64>,
    poisoned: Arc<AtomicBool>,
    slot_tl: Arc<Vec<Timeline>>,
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
        let r = run_job(&mut streamer, &fetch, &job, &io_wait_ns, &slot_tl);
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
    fetch: &Stream,
    job: &ReapJob,
    io_wait_ns: &AtomicU64,
    slot_tl: &[Timeline],
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
        // Publish the ticket: enqueued on the fetch stream AFTER `reap` queued this read's
        // copy, so the timeline reaching this value means the copy has completed. This is
        // what a consumer's `wait` is gated on.
        let t = job.tickets[u];
        slot_tl[t.slot as usize].signal(fetch.raw(), t.value)?;
        // Resolve when the copy (enqueued by `reap`) lands on the fetch stream. Kept as the
        // error/teardown channel; nothing awaits it per-expert any more.
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
