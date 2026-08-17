#![cfg(feature = "rocm")]
//! Per-expert async loads: the io_uring→future adapter under the expert stream.
//!
//! Backend-independent: the fetch stream and the [`Timeline`] both come from
//! the backend waist, and it is a REAL dedicated stream on both — a `hipStream_t` under
//! `rocm`, its own `VkQueue` with its own command-buffer ring and timeline under `vulkan`.
//! It is a real separate engine on both, and that is worth what the Vulkan port paid for it:
//! increment 1 ran every stream on one queue and hid nothing at all (docs/investigations/vulkan-port.md,
//! "Increment 2: measured").
//!
//! The "96% of fetch hidden on ROCm, 97% on Vulkan" that used to be claimed here came from a
//! metric that could not measure hiding — see `Profile::summary`, which now puts it at ~22%.
//! Separate queues are still what makes ANY overlap possible; they were never worth 96%.
//!
//! `Stream::fetch()` rather than `Stream::new()` at the construction below, and that is not
//! cosmetic: the Vulkan side maps the handle onto one of three queues and cannot infer which
//! from context, so the ROLE is named where it is known.
//!
//! A reaper thread owns the demand ring and a dedicated fetch [`Stream`]. Per
//! MoE layer the pipeline hands it a batch of cold reads; it queues+submits them
//! (so they run concurrently on the NVMe), then reaps completions one at a time.
//! Because [`Streamer::reap`] already kicks each read's bounce→slot copy on the
//! fetch stream, the reaper just signals that read's [`Ticket`] on the same stream —
//! so the ticket is satisfied when the COPY lands, not merely the NVMe read. A cache
//! hit never enters here; it carries [`Ticket::RESIDENT`].
//!
//! This is the whole concurrency surface, and it is now entirely device-side: the
//! decode loop stays single-threaded, the reaper blocks off-thread, and the two meet
//! only through the per-slot timelines. The one host round trip left — a `Signal` armed
//! per read with `hipLaunchHostFunc` — was deleted once nothing polled it.
//!
//! ## What bounds it
//!
//! The demand fetch runs at ~10 GB/s and the drive delivers 7.7 GB/s at QD1 rising to
//! ~13 GB/s at QD4 under the engine's own load (docs/measurement/probes/fetch_batch.hip), so a layer's
//! batch is close to what its queue depth can buy. What is NOT close is the duty cycle: the
//! ring only has work between a layer's routing and its MoE launch, so the drive idles
//! ~35% of every token. Nothing here can fix that — a read cannot be issued before the
//! router names it. See docs/reference/architecture.md §4.

use crate::fetch::stream::Streamer;
use anyhow::{Result, anyhow, ensure};
use rivoli_backend::{Stream, Timeline};
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

/// One cold read: file range → pool slot. `dst` is the DMA TARGET for that slot, valid
/// across threads and never CPU-dereferenced here — under HIP the pool's unified pointer,
/// under Vulkan its host mapping (`RoutedPool::host_ptr`, not `ptr`).
pub struct ReadSpec {
    pub fd: RawFd,
    pub begin: usize,
    pub len: usize,
    pub dst: *mut u8,
}
// SAFETY: `dst` is only handed to io_uring / hipMemcpyAsync on the reaper thread,
// never dereferenced on the CPU; `fd` is a plain descriptor.
unsafe impl Send for ReadSpec {}

/// A layer's demand batch: the reads plus the ticket each one redeems.
///
/// There was a `Vec<Signal>` here too, one per read, armed on the fetch stream as each copy
/// landed. Nothing awaited it — `gpu.rs` took the vec and dropped it — since the ticketed
/// dataflow moved the dependency onto the device. It cost a `hipLaunchHostFunc` per read
/// INSIDE the `io_wait` clock (7.2 us of enqueue each, 4% of the fetch stream's throughput;
/// docs/measurement/probes/fetch_stream_ops.hip), and worse, it made the teardown path look correct
/// while releasing nothing that anyone was waiting on.
struct ReapJob {
    reads: Vec<ReadSpec>,
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
/// per-read [`Ticket`] when loaded.
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
    /// Next value to hand out per slot — and, since it is bumped at hand-out, also the LAST
    /// value issued for that slot. So `slot_tl[s].completed() >= slot_next[s]` is exactly
    /// "slot `s`'s bytes have been copied out", which is the reuse gate. Decode-thread only.
    slot_next: Vec<u64>,
    /// Round-robin hand-out cursor, so slots age evenly instead of always retrying the one
    /// whose copy was enqueued most recently.
    cursor: usize,
    /// Times [`take_slot`](Self::take_slot) found every slot still in flight and had to
    /// park. Should stay 0: a layer uses ~2 of 16 slots and a copy retires in ~1.2 ms
    /// against a ~3.5 ms layer. Non-zero means the ring is undersized for the lookahead.
    slot_stalls: u64,
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
        let (tx, rx) = channel::<ReapJob>();
        let fetch_ns = Arc::new(AtomicU64::new(0));
        let io_wait_ns = Arc::new(AtomicU64::new(0));
        let poisoned = Arc::new(AtomicBool::new(false));
        let reaper_state = Reaper {
            streamer,
            fetch,
            fetch_ns: fetch_ns.clone(),
            io_wait_ns: io_wait_ns.clone(),
            poisoned: poisoned.clone(),
            slot_tl: slot_tl.clone(),
        };
        let reaper = std::thread::Builder::new()
            .name("rivoli-reaper".into())
            .spawn(move || reaper_state.run(rx))?;
        Ok(Self {
            tx: Some(tx),
            reaper: Some(reaper),
            fetch_ns,
            io_wait_ns,
            poisoned,
            slot_tl,
            slot_next: vec![0; nslots],
            cursor: 0,
            slot_stalls: 0,
        })
    }

    /// Times a slot hand-out had to park because every staging slot still had a copy in
    /// flight. See [`slot_stalls`](Self::slot_stalls).
    pub fn slot_stalls(&self) -> u64 {
        self.slot_stalls
    }

    /// Take a staging slot whose previous read's bounce copy has retired.
    ///
    /// This is the reuse gate the old per-batch `queued = 0` reset skipped: an integer
    /// reset with no relationship to whether the bounce copy OUT of a slot had retired.
    /// That was safe only because every demand read happens to be awaited inside its
    /// issuing layer — an emergent property of the consumer, written nowhere and enforced
    /// nowhere, and the reason the first speculative preloader corrupted.
    ///
    /// Bumping `slot_next` at hand-out makes the returned slot fail its own test until its
    /// copy lands, so a slot cannot be handed out twice — including twice within one batch.
    fn take_slot(&mut self) -> Result<usize> {
        loop {
            // Destructured so the closure can borrow the timelines while `cursor` moves.
            let Self {
                cursor,
                slot_tl,
                slot_next,
                ..
            } = self;
            if let Some(s) = scan_free(slot_next.len(), cursor, |s| {
                slot_tl[s].completed() >= slot_next[s]
            }) {
                return Ok(s);
            }
            // Nothing free. The reaper is the only thing that can advance a timeline, so if
            // it has died there is no wake-up coming — check before parking again.
            if self.poisoned.load(Ordering::Acquire) {
                return Err(anyhow!(
                    "AsyncFetch poisoned while waiting for a staging slot"
                ));
            }
            self.slot_stalls += 1;
            std::thread::yield_now();
        }
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

    /// Submit a layer's cold reads; returns one [`Ticket`] per read, in the same order.
    /// Reads run concurrently on the NVMe; a ticket is satisfied once its read's bounce copy
    /// has landed on the fetch stream. Empty `reads` returns an empty vec (an all-hit layer).
    pub fn submit(&mut self, reads: Vec<ReadSpec>) -> Result<Vec<Ticket>> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(anyhow!("AsyncFetch poisoned by an earlier reaper error"));
        }
        if reads.is_empty() {
            return Ok(Vec::new());
        }
        ensure!(
            reads.len() <= self.slot_next.len(),
            "batch of {} reads exceeds {} staging slots",
            reads.len(),
            self.slot_next.len()
        );
        // The staging slot is chosen HERE, on the decode thread, and travels with the read —
        // so the ticket is computable before the reaper has touched anything, which is what
        // lets a consumer enqueue its wait ahead of the producer (INV-4). It used to be the
        // read's position in the batch, which is what tied a slot's life to a batch's.
        let mut tickets: Vec<Ticket> = Vec::with_capacity(reads.len());
        for _ in &reads {
            let s = self.take_slot()?;
            self.slot_next[s] += 1;
            tickets.push(Ticket {
                slot: s as u16,
                value: self.slot_next[s],
            });
        }
        let job = ReapJob {
            reads,
            tickets: tickets.clone(),
        };
        self.tx
            .as_ref()
            .ok_or_else(|| anyhow!("AsyncFetch closed"))?
            .send(job)
            .map_err(|_| anyhow!("reaper thread gone"))?;
        Ok(tickets)
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
}

/// One round-robin sweep for a slot `landed` accepts, advancing `cursor` past whatever it
/// returns. `None` = every slot still has a copy in flight. Split out from
/// [`AsyncFetch::take_slot`] only so the hand-out order is testable without a device.
fn scan_free(n: usize, cursor: &mut usize, landed: impl Fn(usize) -> bool) -> Option<usize> {
    for _ in 0..n {
        let s = *cursor;
        *cursor = (*cursor + 1) % n;
        if landed(s) {
            return Some(s);
        }
    }
    None
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

/// Everything the reaper thread holds for its whole life: the demand ring and fetch stream
/// it owns outright, plus the four handles it shares with the [`AsyncFetch`] on the decode
/// thread. One struct because they are one unit — every field is moved into the thread at
/// spawn and read by every job — and because the alternative is threading seven parameters
/// through the loop and five more through each job.
struct Reaper {
    streamer: Streamer,
    fetch: Stream,
    /// Shares [`AsyncFetch::fetch_ns`]; this thread is the only writer.
    fetch_ns: Arc<AtomicU64>,
    /// Shares [`AsyncFetch::io_wait_ns`]; this thread is the only writer.
    io_wait_ns: Arc<AtomicU64>,
    /// Shares [`AsyncFetch::poisoned`]; this thread is the only writer.
    poisoned: Arc<AtomicBool>,
    /// Shares [`AsyncFetch::slot_tl`] — the timelines a ticket is redeemed against.
    slot_tl: Arc<Vec<Timeline>>,
}

impl Reaper {
    /// Reaper thread: one job per layer until the channel closes. Times each job's wall
    /// into `fetch_ns` — the fetch cost the main-thread compute overlaps.
    fn run(mut self, rx: Receiver<ReapJob>) {
        for job in rx {
            // Once poisoned, the ring is dirty (a prior job bailed mid-batch, leaving
            // undrained CQEs and stale queued/min_res). Touching it again would index a
            // stale user_data → C-side OOB or a panic that kills this thread and hangs
            // every later consumer. So don't: abandon the ring and release the tickets.
            if self.poisoned.load(Ordering::Acquire) {
                release(&job, &self.slot_tl);
                continue;
            }
            let t = std::time::Instant::now();
            let r = self.run_job(&job);
            self.fetch_ns
                .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if let Err(e) = r {
                // Fatal fetch error: poison (abandon the dirty ring for all later jobs),
                // log, and release this job's tickets so no consumer hangs. `run_job` may
                // have signalled some of them already; `release` is monotone.
                tracing::error!("reaper: {e:#}");
                self.poisoned.store(true, Ordering::Release);
                release(&job, &self.slot_tl);
            }
        }
    }

    fn run_job(&mut self, job: &ReapJob) -> Result<()> {
        // Destructured so the ring can be borrowed mutably while the stream, the counter
        // and the timelines are read alongside it.
        let Self {
            streamer,
            fetch,
            io_wait_ns,
            slot_tl,
            ..
        } = self;
        for (r, t) in job.reads.iter().zip(&job.tickets) {
            // SAFETY: `dst` is an ALIGN-aligned device slot the pipeline keeps live until
            // this read's signal resolves; the VQ blocks are VQ_ALIGN-aligned so the
            // returned sub-offset is 0 (the slot start IS the block). `t.slot` came from
            // `take_slot`, so its previous copy has retired.
            // The returned sub-offset is DISCARDED, and that is now safe to do rather than
            // merely customary. It used to feed a `debug_assert_eq!(sub, 0, "VQ expert read
            // must be block-aligned")` — a check that is compiled out under `--release`, which
            // is the profile every benchmark and every divergence run uses, so it enforced
            // nothing exactly where it mattered. `RoutedGeom::check_reads_fit_their_slots`
            // makes the property an `ensure!` at `open()` instead: block-aligned starts and
            // supersets that fit one slot, refused before the run rather than during it.
            // Keeping both would be two checks for one property, one of them inert.
            let _sub = unsafe {
                streamer.queue(
                    r.fd,
                    crate::fetch::stream::ReadSpan {
                        begin: r.begin,
                        len: r.len,
                    },
                    crate::fetch::stream::ReadDst {
                        ptr: r.dst,
                        slot: u32::from(t.slot),
                    },
                )?
            };
        }
        streamer.submit()?;
        // Everything above is CPU (building and submitting SQEs); everything in the loop
        // below is the thread parked in the kernel waiting for NVMe. Only the latter is
        // io-wait, which is why the clock starts here and not at the top of `run_job`.
        let t_io = std::time::Instant::now();
        for _ in 0..job.reads.len() {
            // SAFETY: `fetch` is a live stream; `reap` kicks this read's copy on it and
            // returns the completed read's user_data — which is now the STAGING SLOT, so
            // the batch position it belongs to has to be looked up. A batch is ≤ top_k
            // reads, so the scan is cheaper than carrying a slot→position map.
            let slot = unsafe { streamer.reap(fetch.raw())? };
            let u = job
                .tickets
                .iter()
                .position(|t| usize::from(t.slot) == slot)
                .ok_or_else(|| {
                    anyhow!("completion for slot {slot}, which this batch never queued")
                })?;
            // Publish the ticket: enqueued on the fetch stream AFTER `reap` queued this
            // read's copy, so the timeline reaching this value means the copy has completed.
            // This is what a consumer's `wait` is gated on, and now the only thing published
            // here — a `Signal::arm_on` used to follow, which is a `hipLaunchHostFunc` (a
            // host round trip recorded INTO this stream) for a future nobody polled.
            let t = job.tickets[u];
            slot_tl[t.slot as usize].signal(fetch.raw(), t.value)?;
        }
        let e_io = std::time::Instant::now();
        io_wait_ns.fetch_add(
            e_io.duration_since(t_io).as_nanos() as u64,
            Ordering::Relaxed,
        );
        // Emitted on the REAPER thread against the same shared anchor as the decode
        // thread's spans, so a trace viewer draws them on one timeline. This is the pair
        // whose overlap the whole streaming design is a bet on: io-wait bars should sit
        // underneath the decode thread's gpu-wait bars, not beside them.
        crate::telemetry::spans::record("io-wait/uring-reap", "reaper", t_io, e_io);
        Ok(())
    }
}

/// Teardown: force every ticket in `job` to its value from the HOST, so consumers already
/// gated on it stop waiting for a copy that will never be enqueued.
///
/// **This releases the ticket, not a `Signal`, and that is the whole point.** The poison
/// path used to resolve one `Signal` per read and leave the timelines untouched. Once the
/// ticketed dataflow moved the dependency onto the device, nothing awaited those signals and
/// everything waited on the timelines — so a fetch error stopped surfacing as an error and
/// started HANGING the device on a `hipStreamWaitValue64` whose value nothing would write.
/// The teardown path kept releasing the thing that had become vestigial.
///
/// The slot's bytes are stale, which is correct to allow: `submit` fails fast from here on
/// so the decode returns the real fetch error, and the forward's finiteness guard localises
/// whatever the released kernels computed in the meantime.
fn release(job: &ReapJob, slot_tl: &[Timeline]) {
    for t in &job.tickets {
        slot_tl[t.slot as usize].release(t.value);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;

    /// The hand-out rule, without a device: a slot is free iff its timeline has reached the
    /// last value issued for it, and bumping that value at hand-out is what stops the same slot
    /// being handed out twice — the property the old per-batch integer reset lacked.
    ///
    /// **INV-9: nothing on the host path can make two runs of one input a different program.**
    /// This test predates the invariant and is what carries it, which is why it was RENAMED
    /// rather than joined by a second one (review, 2026-08-17 — the second one asserted the same
    /// three things and its declared red proof reddened this one too).
    ///
    /// The connection: the one host decision on the routed path that reads DEVICE PROGRESS
    /// rather than its own inputs is this hand-out, since `landed(s)` is a timeline value, i.e.
    /// wall-clock. If it could vary between two runs of one input the runs would stop being the
    /// same program there. It cannot, because of a barrier somewhere else —
    /// `glm::forward::run_layer` ends every layer with an unconditional `device_sync`, so at the
    /// next `submit` every prior bounce copy has retired, `landed` is uniformly true, and
    /// `scan_free` is pure round-robin over the miss sequence. `AsyncFetch::slot_stalls()` is
    /// the observable that falsifies that precondition: non-zero means it did not hold and a
    /// determinism comparison over that run is unsound.
    ///
    /// What the assertions below have teeth against is the contrapositive — that an un-landed
    /// slot is never issued and does change the hand-out. Red-proofed 2026-08-17 by making
    /// `scan_free` ignore `landed`: the `None` assertion becomes `Some`.
    ///
    /// **Scope.** Two runs are the same PROGRAM, not the same OUTPUT. The output property is a
    /// device property; its gate is `tests/determinism-glm.sh` (GPU, real artifact, >=256 tokens
    /// for 82% power against the measured rate). See
    /// `docs/investigations/glm-nondeterminism.md`.
    #[test]
    fn inv_9_a_slot_is_not_reissued_until_its_copy_lands() {
        const N: usize = 4;
        let completed = [0u64; N]; // nothing has landed yet
        let mut next = [0u64; N]; // ...and nothing has been issued yet either
        let mut cursor = 0;
        let mut got = Vec::new();
        // Hand out a full batch. Each pick bumps `next[s]` past `completed[s]`, so the
        // sweep cannot return it again even though no copy has retired.
        for _ in 0..N {
            let s = scan_free(N, &mut cursor, |s| completed[s] >= next[s]).expect("a free slot");
            next[s] += 1;
            got.push(s);
        }
        got.sort_unstable();
        assert_eq!(got, [0, 1, 2, 3], "every slot handed out exactly once");
        // All four copies are still in flight: the ring is exhausted, and the caller must
        // park rather than reuse a slot whose bytes are still being read out.
        assert_eq!(scan_free(N, &mut cursor, |s| completed[s] >= next[s]), None);
        // ...and the hand-out above was a real sweep, not an early `None`. Without this, a
        // `scan_free` that always returned `None` would satisfy every assertion in this test
        // (review, 2026-08-17): `got` would be empty, `[]` sorts to `[]`, and the exhaustion
        // assertion would pass for the wrong reason.
        assert_eq!(next, [1u64; N], "every slot was issued exactly once");
        // One copy retires → exactly that slot comes back.
        let landed = [1u64, 0, 0, 0];
        assert_eq!(
            scan_free(N, &mut cursor, |s| landed[s] >= next[s]),
            Some(0),
            "the slot whose timeline advanced is the one reissued"
        );
    }

    /// Round-robin, not first-fit: a slot just handed back is the LAST candidate, so slots
    /// age evenly instead of hammering the one whose copy was enqueued most recently.
    #[test]
    fn hand_out_is_round_robin() {
        let mut cursor = 0;
        let picks: Vec<usize> = (0..5)
            .map(|_| scan_free(3, &mut cursor, |_| true).expect("all free"))
            .collect();
        assert_eq!(picks, [0, 1, 2, 0, 1]);
    }
}
