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
//!    the flag produced.)
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

/// The fetch path's knobs — the whole intervention matrix the GLM nondeterminism
/// investigation (`docs/investigations/glm-nondeterminism-closeout.md`) built, bundled
/// because [`Streamer::new`] was already seven parameters of which four were `bool` —
/// exactly the transposition hazard this workspace refuses everywhere else.
///
/// [`FetchKnobs::default`] is the STOCK configuration (bounce, no arms); every field
/// that is not it is one named arm of that investigation.
#[derive(Clone, Copy)]
pub struct FetchKnobs {
    /// `--pinned-coherent`: allocate the bounce arena FINE-GRAINED. RED 2026-08-18.
    pub coherent: bool,
    /// `--copy-by-kernel`: bounce→slot by shader copy instead of the copy engine. RED.
    pub by_kernel: bool,
    /// `--arena-refresh`: full-width device read of the just-written arena window,
    /// pre-copy. The ONE clean cell of the matrix — a mitigation, not a fix.
    pub arena_refresh: bool,
    /// `--arena-refresh-stride N`: the refresh at one 16 B unit per N — the dose-response
    /// sweep the investigation left unfinished. N=4 is the 64 B sector hypothesis's
    /// prediction; N=8 is the `bh-line` cell (RED @704). 0 = off. Conflicts with
    /// `arena_refresh` (two spellings of one arm).
    pub arena_refresh_stride: u64,
    /// `--arena-refresh-late`: the SAME full-width arena read enqueued AFTER the copy instead
    /// of before it — it cannot delay the copy, only the signal, so it separates "the repair
    /// must precede the COPY" from "...the CONSUMER". Clean = the copy reads the arena
    /// correctly without help and the defect is consumer-side; red = producer-side.
    pub arena_refresh_late: bool,
    /// `false` is DIRECT mode (`--direct-vmm-dma`): no arena, no copy. Diagnostic.
    pub bounce: bool,
    /// `--slot-refresh`: DIRECT-only full-width read of the just-DMA'd slot. RED.
    pub slot_refresh: bool,
    /// `--copy-via-cpu`: the bounce→slot hop as a HOST memcpy on the reaper thread —
    /// the candidate FIX. No GPU agent then ever reads memory the NVMe wrote: the
    /// arena is read only by the CPU (the CQE's own guarantee, the one btrfs's datasum
    /// verification already relies on) and the slot is written only by the CPU (the
    /// CPU→GPU coherence `kernels/vmm.hip` was verified to have and the resident
    /// tier's 281 GB startup load already spends). The ticket is signalled on the
    /// fetch stream exactly as after an SDMA copy, so the consumer side is unchanged.
    pub cpu_copy: bool,
    /// `--fetch-settle-us`: pure TIME between the CQE and the copy — a host sleep on
    /// the reaper thread, no device work, no memory traffic. The arm the ablation
    /// matrix never had: `bh-nop` has ~no delay and `bh-decoy` reads a DEVICE buffer
    /// (~10x faster than the host-memory `bh` it was meant to hold duration for), so
    /// "not a delay effect" was never established. CLEAN here and the repair was time.
    pub settle_us: u64,
    /// `--arena-refresh-decoy`: the `--arena-refresh` read aimed at a SECOND pinned
    /// host arena that the NVMe never writes — same kernel, same stream position,
    /// same bytes, same memory type, same per-slot cycling; different addresses.
    /// With `settle` it separates TIME from ADDRESS: both clean ⇒ the repair is the
    /// delay; both red ⇒ the repair is specific to the DMA'd region.
    pub arena_refresh_decoy: bool,
    /// `--cpu-retouch` (DIRECT mode only): after the CQE, the reaper CPU-reads the
    /// just-DMA'd pool slot into a scratch buffer and CPU-writes it back. The payload
    /// is unchanged, so nothing about the DATA differs — only the last WRITER changes,
    /// from the NVMe's DMA engine to the CPU. This is the `--copy-via-cpu` premise
    /// tested against the one condition that still fires: if CPU writes repair GPU
    /// visibility of the device pages, direct+retouch is CLEAN; if it is RED, CPU→GPU
    /// coherence into the pool is not reliable at this rate and the fix is dead.
    pub cpu_retouch: bool,
}

impl Default for FetchKnobs {
    /// The stock configuration: bounce mode, every arm off. Hand-written because a
    /// derived default would give `bounce: false` — DIRECT, a diagnostic mode, as the
    /// DEFAULT — the `Folds`-stride trap (`glm-nondeterminism.md`) wearing a new hat.
    fn default() -> Self {
        Self {
            coherent: false,
            by_kernel: false,
            arena_refresh: false,
            arena_refresh_stride: 0,
            arena_refresh_late: false,
            bounce: true,
            slot_refresh: false,
            cpu_copy: false,
            settle_us: 0,
            arena_refresh_decoy: false,
            cpu_retouch: false,
        }
    }
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
    /// BOUNCE (the default): reads land in [`Streamer::arena`] and are async-copied into
    /// the pool slot. `false` is DIRECT — `--direct-vmm-dma`, recovered 2026-08-18 as a
    /// DIAGNOSTIC and not as a shipping mode.
    ///
    /// **It is measurably the worse mechanism and is not a candidate.** Re-derived on
    /// kernel 6.18.39 / ROCm 7.14 with `docs/measurement/probes/fetch_dest.hip`: O_DIRECT
    /// DMA into the VMM pool runs **6.4 GB/s** against **13.3** into the pinned arena at the
    /// engine's queue depth with the GPU busy — a 2.08x gap that reproduces the 2026-07-30
    /// measurement (5.66 vs 12.4, 2.19x) which deleted the flag in the first place. The
    /// pre-registered bar for recovering it as a real mode was 11.4 GB/s.
    ///
    /// What it is FOR: direct mode has no arena, so it is the only arm that can say whether
    /// the arena is the LOCUS of the GLM nondeterminism defect or merely where the repair
    /// happened to land. `--arena-refresh` mitigates that defect without explaining it
    /// (`glm-bug.md` §14); this arm answers the question by REMOVAL instead of by repair.
    /// Divergence here would mean the missing guarantee is downstream of the arena entirely.
    bounce: bool,
    /// Staging arena, `entries * span` bytes: read slot `user_data` is
    /// `arena + user_data * span`. HIP-pinned under `rocm` — see [`stage`].
    /// Null in DIRECT mode, where nothing stages; never null in bounce mode.
    arena: *mut u8,
    /// Per-SLOT VMM destination + aligned read length. `reap` copies
    /// `nbytes` from the arena slot into `dst`. Indexed by staging slot — which is the
    /// read's `user_data` — and NOT cleared per batch: a slot's lifetime is owned by its
    /// [`Ticket`](crate::fetch::asyncfetch::Ticket) now, not by the batch that happened to use it.
    dst: Vec<*mut u8>,
    nbytes: Vec<u32>,
    /// Per-SLOT divergence-fold targets, parallel to `dst`. [`FetchFolds::OFF`] = folds off.
    fold: Vec<FetchFolds>,
    /// Phase 3B: move the bytes with a shader copy instead of `hipMemcpyAsync`.
    by_kernel: bool,
    /// ARENA REFRESH: enqueue a full-width device read of the just-written arena window on the
    /// fetch stream BEFORE the copy. The only intervention measured to make GLM decode
    /// reproduce itself; a MITIGATION with an unexplained mechanism, not a root-cause fix.
    /// Evidence, alternatives tried, and the ceiling: `kernels/async.hip`, `glm-bug.md` §7b.
    arena_refresh: bool,
    /// `--arena-refresh-stride`: the refresh at reduced density (16 B units), 0 = off.
    arena_refresh_stride: u64,
    /// `--arena-refresh-late`: the refresh enqueued AFTER the copy instead of before it.
    arena_refresh_late: bool,
    /// SLOT REFRESH — the arm that tests the only rule fitting every cell of the matrix:
    /// *a device-side reader can read STALE bytes from the region the NVMe just DMA'd into, and a
    /// prior full-width read by a compute kernel repairs it for the next consumer.*
    ///
    /// In BOUNCE mode the DMA target is the arena and the next consumer is the SDMA copy, so
    /// `--arena-refresh` is that read and it is CLEAN. In DIRECT mode the DMA target is the pool
    /// slot and the next consumer is the MoE kernel — and nothing reads it first, which is why
    /// direct diverges (91/512 @420, 2026-08-18). This enqueues the same full-width read of the
    /// SLOT, after the completion and before the ticket signal the miss kernel waits on.
    ///
    /// **Clean ⇒ the rule unifies every arm. Red ⇒ the rule is dead.** Either outcome is decisive,
    /// which is why it is built as its own flag rather than folded into `arena_refresh`.
    ///
    /// DIRECT only. In bounce mode this read is `sc`, already measured RED @236, so the CLI
    /// refuses the combination rather than re-running a known-red arm under a new name.
    slot_refresh: bool,
    /// `--copy-via-cpu`: the bounce→slot hop as a host memcpy on the reaper thread — see
    /// [`FetchKnobs::cpu_copy`]. Bounce mode only.
    cpu_copy: bool,
    /// `--cpu-retouch` (direct only) — see [`FetchKnobs::cpu_retouch`]. The scratch is `span`
    /// bytes of pinned host memory, allocated only when the arm is on; null otherwise.
    cpu_retouch: *mut u8,
    /// `--fetch-settle-us` as a Duration — see [`FetchKnobs::settle_us`].
    settle: std::time::Duration,
    /// `--arena-refresh-decoy` — see [`FetchKnobs::arena_refresh_decoy`]. The buffer is
    /// `entries * span` bytes like the arena itself and is indexed by the same `ud * span`, so
    /// the read's size, stride pattern, memory type and duration all match the real refresh —
    /// only the addresses (never DMA'd) differ. Null when the arm is off.
    arena_refresh_decoy: *mut u8,
    /// One-shot: has the first slot refresh been LOGGED yet. Positive evidence that the
    /// intervention ran, not that it was asked for — two rounds of this investigation were lost
    /// to arms that never applied, and an arm that did not apply reds exactly like one that
    /// does not work.
    logged_refresh: bool,
    /// Device word the refresh kernel stores to only on an impossible value — it exists to stop
    /// the loads being optimised away, is never read, and needs no synchronisation.
    refresh_sink: *mut u64,
    /// Copies actually issued, per path — `[sdma-memcpy, shader-kernel, host-cpu]`.
    ///
    /// **A COUNT, not the flag.** Two rounds of this investigation were spent on an arm that could
    /// not be believed because nothing observed whether the intervention applied: the log recorded
    /// the intent and the runtime was free to do something else. An intervention that never applied
    /// and one that does not work produce the same red, so every candidate fix now reports what it
    /// DID.
    issued: [u64; 3],
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
    /// single read may deliver (`slot_span` of the biggest projection tensor); `knobs`
    /// selects the mode and any investigation arm (see [`FetchKnobs`]).
    /// Always allocates the `entries * span` host staging arena — it is the only
    /// destination path (see this module's header).
    pub fn new(entries: u32, span: usize, knobs: FetchKnobs) -> Result<Self> {
        let FetchKnobs {
            coherent,
            by_kernel,
            arena_refresh,
            arena_refresh_stride,
            arena_refresh_late,
            bounce,
            slot_refresh,
            cpu_copy,
            settle_us,
            arena_refresh_decoy,
            cpu_retouch,
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

        // `--slot-refresh` in BOUNCE mode IS `sc`, already measured RED @236. Refused rather than
        // re-run under a new name; the CLI says the same thing with the number attached.
        ensure!(
            bounce || !arena_refresh,
            "--slot-refresh and --arena-refresh are the same read of different regions; pass one"
        );
        ensure!(
            !bounce || !slot_refresh,
            "--slot-refresh reads the destination slot, which in bounce mode is the `sc` arm — \
             already measured RED at first-divergence 236. It applies only with --direct-vmm-dma"
        );
        // The `--copy-via-cpu` refusal set: it IS the copy path, so every other copy-path knob
        // conflicts, and it leaves no GPU-side reader of the arena, so both refresh arms lose
        // their subject. A silently-ignored knob is how an arm gets attributed to the wrong cause.
        ensure!(
            bounce || !cpu_copy,
            "--direct-vmm-dma has no bounce arena and no copy; --copy-via-cpu has no subject"
        );
        ensure!(
            !(cpu_copy && by_kernel),
            "--copy-via-cpu and --copy-by-kernel are two answers to one question; pass one"
        );
        ensure!(
            !(cpu_copy && arena_refresh),
            "--copy-via-cpu leaves no GPU-side reader of the arena; --arena-refresh has no subject"
        );
        ensure!(
            !(cpu_copy && arena_refresh_decoy),
            "--copy-via-cpu leaves no GPU-side reader of the arena; --arena-refresh-decoy has no subject"
        );
        ensure!(
            !(arena_refresh && arena_refresh_decoy),
            "--arena-refresh and --arena-refresh-decoy are the same read of different regions; pass one"
        );
        ensure!(
            bounce || !arena_refresh_decoy,
            "--arena-refresh-decoy reads a pinned HOST buffer; direct mode's question is about \
             the slot. Refused rather than run an arm that decides nothing"
        );
        // The stride and late arms are the SAME refresh at a different density or position —
        // each combination that would make a cell ambiguous is refused rather than run.
        ensure!(
            !(arena_refresh && arena_refresh_stride > 0),
            "--arena-refresh IS --arena-refresh-stride 1; pass one spelling"
        );
        ensure!(
            !arena_refresh_late || !(arena_refresh || arena_refresh_decoy),
            "two refresh arms at once answer nothing — pass one"
        );
        ensure!(
            !(arena_refresh_stride > 0 && arena_refresh_late),
            "--arena-refresh-stride is the PRE-copy position; --arena-refresh-late the post-copy — pass one"
        );
        ensure!(
            bounce || (arena_refresh_stride == 0 && !arena_refresh_late),
            "--arena-refresh-stride/--arena-refresh-late read the arena; direct mode has none"
        );
        ensure!(
            !(cpu_copy && (arena_refresh_stride > 0 || arena_refresh_late)),
            "--copy-via-cpu leaves no GPU-side reader of the arena; the refresh arms lose their subject"
        );
        // `--cpu-retouch` is a DIRECT-mode arm: in bounce mode the slot's last writer is the
        // copy, and retouching the arena would test a property the copy then re-crosses.
        ensure!(
            !bounce || !cpu_retouch,
            "--cpu-retouch rewrites the slot the drive wrote; bounce mode's writer is the copy \
             — it applies only with --direct-vmm-dma"
        );
        ensure!(
            !(cpu_retouch && slot_refresh),
            "--cpu-retouch and --slot-refresh are two direct-mode arms; one question per arm"
        );
        // The retouch scratch: `span` bytes of pinned host memory, allocated only for the arm.
        let cpu_retouch = match cpu_retouch {
            false => std::ptr::null_mut(),
            true => {
                let s = stage::alloc(span, false);
                ensure!(!s.is_null(), "cpu-retouch scratch alloc failed");
                s
            }
        };
        // The decoy arena: same size and indexing as the real one (`entries * span`), so the
        // refresh read's footprint and cycling match exactly. Allocated ONLY when its arm is on.
        let arena_refresh_decoy = match arena_refresh_decoy {
            false => std::ptr::null_mut(),
            true => {
                let d = stage::alloc(entries as usize * span, coherent);
                ensure!(!d.is_null(), "arena-refresh decoy alloc failed");
                d
            }
        };
        // One page, only when DIRECT needs a sink and therefore cannot borrow the arena's.
        // Allocated before the arena so a single `Self` below can take both unconditionally.
        let refresh_sink = match (bounce, slot_refresh) {
            (false, true) => {
                let p = stage::alloc(ALIGN, false);
                ensure!(!p.is_null(), "slot-refresh sink alloc failed");
                p
            }
            _ => std::ptr::null_mut(),
        };
        // The arena is the ONE thing DIRECT mode changes: it allocates none, and that
        // absence is the intervention. Everything below is shared, so there is exactly one
        // `Self` — a second one is a jscpd clone and, worse, a place for the two modes to
        // drift apart.
        let arena = match bounce {
            false => {
                // The three staging knobs act on an arena this mode does not have. The CLI
                // refuses the combination; refused here too, because a silently-ignored
                // knob is how an arm gets attributed to the wrong cause.
                ensure!(
                    !arena_refresh && !by_kernel && !coherent,
                    "--direct-vmm-dma has no bounce arena, so --arena-refresh / \
                     --copy-by-kernel / --pinned-coherent cannot apply — pass none of them"
                );
                tracing::info!(
                    "DIRECT mode (--direct-vmm-dma): NO bounce arena, no H2D copy — every \
                     read DMAs straight into its pool slot. Diagnostic arm; measured 2.08x \
                     slower than bounce (probes/fetch_dest.hip, 2026-08-18)"
                );
                std::ptr::null_mut()
            }
            true => {
                let arena_bytes = entries as usize * span;
                let arena = stage::alloc(arena_bytes, coherent);
                // Logged because it is the one allocation whose MEMORY TYPE is under
                // investigation: a run's record has to say which it made, or its result
                // cannot be attributed. REQUESTED *and* RETURNED — every arm of the
                // coherence experiment is read off this line, so it reports the observation
                // and not the intent; a `?` means the runtime refused to say.
                let got = stage::flags(arena);
                tracing::info!(
                    "bounce arena: {:.0} MiB | requested {} | returned flags {} coherent-bit {}",
                    arena_bytes as f64 / (1u64 << 20) as f64,
                    match coherent {
                        true => "hipHostMallocCoherent",
                        false => "hipHostMallocDefault",
                    },
                    got.map_or("?".into(), |(f, _)| format!("0x{f:x}")),
                    got.map_or("?".into(), |(_, c)| c.to_string()),
                );
                // The one case that invalidates an arm outright: the flag was asked for and
                // did not stick. Loud, because it reads in a log exactly like a fix that
                // did not work.
                if let Some((_, c)) = got
                    && c != coherent
                {
                    tracing::error!(
                        "bounce arena: asked coherent={coherent} but the runtime returned \
                         coherent={c} — this arm did NOT apply the intervention it claims to test"
                    );
                }
                ensure!(
                    !arena.is_null(),
                    "bounce arena alloc failed (entries={entries}, {:.0} MiB)",
                    arena_bytes as f64 / (1u64 << 20) as f64
                );
                arena
            }
        };
        // BOUNCE borrows the arena's first word rather than holding a second allocation.
        let refresh_sink = match bounce {
            true => arena,
            false => refresh_sink,
        };

        if settle_us > 0 {
            // Logged at CONSTRUCTION: the arm is a host sleep, so there is no device-side
            // application to witness later — this line is the whole read-back.
            tracing::info!(
                "FETCH SETTLE (--fetch-settle-us): {settle_us} us of pure host time between \
                 every CQE and its copy/signal — the TIME-vs-ADDRESS discriminator"
            );
        }
        if cpu_copy {
            tracing::info!(
                "COPY VIA CPU (--copy-via-cpu): the bounce->slot hop is a host memcpy on the \
                 reaper thread; no GPU agent reads IO-written memory anywhere on the path"
            );
        }
        if !arena_refresh_decoy.is_null() {
            tracing::info!(
                "ARENA REFRESH DECOY (--arena-refresh-decoy): the refresh read is aimed at a \
                 second pinned arena the NVMe NEVER writes — the ADDRESS discriminator"
            );
        }

        Ok(Self {
            ring: ManuallyDrop::new(ring),
            entries,
            span,
            bounce,
            slot_refresh,
            cpu_copy,
            cpu_retouch,
            settle: std::time::Duration::from_micros(settle_us),
            arena_refresh_decoy,
            arena_refresh_stride,
            arena_refresh_late,
            logged_refresh: false,
            arena,
            dst: vec![std::ptr::null_mut(); entries as usize],
            nbytes: vec![0; entries as usize],
            fold: vec![FetchFolds::OFF; entries as usize],
            by_kernel,
            arena_refresh,
            // The sink is a pinned word, not a device allocation: the kernel's store is never
            // taken, so the address only has to be writable and mapped. In BOUNCE mode it reuses
            // the arena's first word, which avoids an allocation whose lifetime would have to be
            // argued against teardown order. DIRECT has no arena, so `--slot-refresh` gets one
            // page of its own — freed in `Drop`, and null when the flag is off.
            refresh_sink: refresh_sink.cast::<u64>(),
            issued: [0; 3],
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
        // BOUNCE: the read lands in this slot's arena window and `reap` copies it on to
        // `dst`. DIRECT: the read lands in `dst` — the pool slot — and there is nothing to
        // copy. `dst`'s ALIGN-alignment (asserted above) is what makes the direct case a
        // legal O_DIRECT destination; the pool maintains it via `routed::pool_budget`.
        //
        // SAFETY: `ud < entries` (checked above), and the arena is `entries*span` with
        // every read's `nbytes <= span` (checked above), so this slot's window is owned
        // and in bounds.
        let into = match self.bounce {
            true => unsafe { self.arena.add(ud as usize * self.span) },
            false => dst,
        };
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
        // FETCH SETTLE: pure TIME between the CQE and whatever consumes the read next. A host
        // sleep on the reaper thread, NOT a device delay kernel: a kernel would occupy the
        // fetch stream and carry a dispatch (bh-nop's confound), a sleep is only time. This is
        // the arm the ablation matrix never had — `bh-decoy` was recorded as the equal-duration
        // control but reads a DEVICE buffer (~10x the bandwidth of the host arena), so the
        // matrix's "not bandwidth or delay" cell never held duration constant. CLEAN here and
        // the repair is time; RED and the repair is specific to the arena's addresses.
        if !self.settle.is_zero() {
            std::thread::sleep(self.settle);
        }
        // DIRECT: the drive wrote the pool slot itself. No refresh, no copy, and no folds —
        // the `bh` fold hashes the arena, which does not exist here, and `sc` hashes a slot
        // no copy ever wrote. The divergence probe is a bounce-mode instrument by
        // construction, so this arm is measured with the determinism gate instead.
        if !self.bounce {
            // CPU RETOUCH: read the just-DMA'd slot into the scratch and write it straight
            // back. The bytes are unchanged; the last WRITER changes from the NVMe's DMA
            // engine to the CPU — the `--copy-via-cpu` premise (CPU writes to the pool are
            // GPU-visible, the property the resident tier's startup load spends) tested
            // against the one condition that still fires.
            if !self.cpu_retouch.is_null() {
                // SAFETY: `dst[ud]` is a live pool slot holding the `nbytes[ud]` bytes the
                // drive just wrote, host-mapped for read+write (what `host_ptr` is FOR); the
                // scratch is `span >= nbytes[ud]` bytes of pinned host memory. The two
                // copies never overlap (different allocations).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.dst[ud],
                        self.cpu_retouch,
                        self.nbytes[ud] as usize,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.cpu_retouch,
                        self.dst[ud],
                        self.nbytes[ud] as usize,
                    );
                }
                self.log_applied_once(format!(
                    "CPU RETOUCH applied: host read+write-back of {} B of pool slot {ud}, \
                     between the CQE and this slot's ticket signal",
                    self.nbytes[ud]
                ));
            }
            // SLOT REFRESH: the same full-width read `--arena-refresh` performs, aimed at the
            // region DIRECT mode actually DMA'd into. Enqueued on the fetch stream here, which is
            // before `asyncfetch::run_job` signals this slot's timeline — so it is stream-ordered
            // ahead of the miss kernel that waits on that value, exactly as the arena read is
            // ordered ahead of the copy.
            if self.slot_refresh {
                // SAFETY: `dst[ud]` is a live pool slot holding the `nbytes[ud]` bytes the drive
                // just wrote and valid until this read's signal fires; `stream` is live;
                // `refresh_sink` is a mapped, writable page the kernel never stores to.
                unsafe {
                    stage::touch_region(
                        self.dst[ud],
                        self.nbytes[ud] as usize,
                        stream,
                        self.refresh_sink,
                        1,
                    )
                }
                .map_err(|e| anyhow::anyhow!("slot refresh launch failed: {e}"))?;
                self.log_applied_once(format!(
                    "SLOT REFRESH applied: full-width read of {} B of pool slot {ud}, \
                     enqueued on the fetch stream ahead of this slot's ticket signal",
                    self.nbytes[ud]
                ));
            }
            return Ok(ud);
        }
        // SAFETY: `ud < entries`, and every read's `nbytes <= span`, so this slot's arena window
        // is owned and in bounds.
        let src = unsafe { self.arena.add(ud * self.span) };
        // ARENA REFRESH, enqueued BEFORE the copy on the fetch stream. See the struct field and
        // `kernels/async.hip` for the evidence, the fifteen alternatives that did not work, and
        // the ceiling. Stream-ordered ahead of the copy, so it needs no sync of its own.
        // Three arms select (region, density): `--arena-refresh` is (the arena, full width),
        // `--arena-refresh-decoy` is (the never-DMA'd second arena, full width) and
        // `--arena-refresh-stride` is (the arena, one 16 B unit per N).
        // SAFETY: `src` is this slot's arena window, valid for `nbytes[ud]` bytes; the decoy
        // branch substitutes its own `entries * span` arena at the same `ud` offset, so that
        // window is owned and in bounds too; the stream is live; `refresh_sink` is a mapped,
        // writable word the kernel never stores to.
        let refresh = if self.arena_refresh {
            Some((src, 1))
        } else if !self.arena_refresh_decoy.is_null() {
            // SAFETY: the decoy is `entries * span` and `ud < entries` — in-bounds window.
            Some((unsafe { self.arena_refresh_decoy.add(ud * self.span) }, 1))
        } else if self.arena_refresh_stride > 0 {
            Some((src, self.arena_refresh_stride))
        } else {
            None
        };
        if let Some((rsrc, stride4)) = refresh {
            unsafe {
                stage::touch_region(
                    rsrc,
                    self.nbytes[ud] as usize,
                    stream,
                    self.refresh_sink,
                    stride4,
                )
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
            // `i_base = ud * n`, NOT 0: every cold read of a layer folds into ONE accumulator, so at
            // 0 the fold would be invariant under two reads' payloads being SWAPPED between their
            // destinations — a crossed destination is precisely the class under investigation. `ud`
            // is deterministic given the miss sequence (INV-9), so it is comparable across runs.
            //
            // LOGGED, NOT `?`. A `?` here returns from `reap` after the CQE is consumed and BEFORE
            // `copy_to_slot`, so the reaper poisons, the ticket is released from the host, and the
            // miss kernel launches over a slot this layer never wrote — the INSTRUMENT changing what
            // the engine computes, which it may never do.
            let n = self.nbytes[ud] as usize / 4;
            let (buf, count, stride) = match self.fold[ud].bh_mode {
                crate::fetch::asyncfetch::FoldProbe::Off => (src as *const f32, 0, 1),
                crate::fetch::asyncfetch::FoldProbe::Full => (src as *const f32, n, 1),
                crate::fetch::asyncfetch::FoldProbe::Line => {
                    (src as *const f32, n, self.fold[ud].line_stride)
                }
                crate::fetch::asyncfetch::FoldProbe::Decoy => (self.fold[ud].decoy, n, 1),
                crate::fetch::asyncfetch::FoldProbe::Nop => (self.fold[ud].decoy, 1, 1),
            };
            if count > 0 {
                // SAFETY: `buf` owns `count` readable f32 — the arena slot (`nbytes` bytes, just
                // written by the completed read) or the decoy; `bh` is one live device u64;
                // `stream` is live.
                let r = unsafe {
                    rivoli_backend::launch_hash_rows(
                        buf,
                        count,
                        stride,
                        (ud as u64) * n as u64,
                        self.fold[ud].bh,
                        stream,
                    )
                };
                if let Err(e) = r {
                    tracing::error!("divergence probe: bh fold failed on slot {ud} ({e:#})");
                }
            }
        }
        // THE COPY — three paths, one destination:
        //  - default: `hipMemcpyAsync` (SDMA) on the fetch stream, async;
        //  - `--copy-by-kernel`: a shader copy on the fetch stream, async;
        //  - `--copy-via-cpu`: a HOST memcpy, right here on the reaper thread.
        //
        // The CPU path is the candidate FIX and its argument is a subtraction: after it, NO
        // GPU agent anywhere reads memory the NVMe's DMA wrote. The arena is read only by the
        // CPU — the visibility the io_uring CQE actually guarantees (and the one btrfs's
        // datasum check already spends) — and the slot is written only by the CPU, whose
        // writes to this VMM are verified GPU-coherent (`kernels/vmm.hip`; the resident
        // tier's startup load spends the same property at 281 GB). The ticket still signals
        // on the fetch stream after this returns, so the consumer side is byte-identical.
        //
        // SAFETY: `dst[ud]` is a live pool slot the pipeline keeps valid until this read's
        // signal fires, and its HOST mapping is writable (that is what `host_ptr` is FOR);
        // the arena slot holds the just-read `nbytes[ud]` bytes; arena and pool never alias.
        if self.cpu_copy {
            unsafe { std::ptr::copy_nonoverlapping(src, self.dst[ud], self.nbytes[ud] as usize) };
            self.issued[2] += 1;
            self.log_applied_once(format!(
                "COPY VIA CPU applied: host memcpy of {} B, arena slot {ud} -> pool slot",
                self.nbytes[ud]
            ));
        } else {
            // SAFETY: `dst[ud]` is a live pool slot the pipeline keeps valid until this
            // read's signal fires; the arena slot holds the just-read bytes; `stream` is
            // live. Copies the full aligned `nbytes` (a trailing-EOF short read leaves
            // stale bytes only past the useful window, never read).
            let r = unsafe {
                stage::copy_to_slot(
                    self.dst[ud],
                    src,
                    self.nbytes[ud] as usize,
                    stream,
                    self.by_kernel,
                )
            };
            if let Err(e) = r {
                anyhow::bail!("bounce staging copy failed on slot {ud} ({e})");
            }
            self.issued[usize::from(self.by_kernel)] += 1;
        }
        // ARENA REFRESH LATE: the same read as `--arena-refresh`, enqueued AFTER the copy — the
        // stream's FIFO order means it cannot hasten or delay the copy itself, only the signal
        // that follows, so it tests whether the repair must precede the COPY or only the
        // CONSUMER. Clean here re-locates the defect to the slot's consumer entirely.
        if self.arena_refresh_late {
            // SAFETY: as the pre-copy refresh above — same window, same sink, live stream.
            unsafe {
                stage::touch_region(src, self.nbytes[ud] as usize, stream, self.refresh_sink, 1)
            }
            .map_err(|e| anyhow::anyhow!("arena refresh (late) launch failed: {e}"))?;
        }
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
            // Logged rather than `?` for the reason given at `bh` above: a `?` here would abort a
            // copy the engine needs, i.e. the instrument changing what is computed.
            let n = self.nbytes[ud] as usize / 4;
            // ONE call serves every arm; they differ only in WHICH buffer, HOW MUCH of it, and
            // therefore how long they take. See `FoldProbe` for the ladder and what each rung means.
            let slot = self.dst[ud] as *const f32;
            let (buf, count, stride) = match self.fold[ud].sc_mode {
                crate::fetch::asyncfetch::FoldProbe::Off => (slot, 0, 1),
                crate::fetch::asyncfetch::FoldProbe::Full => (slot, n, 1),
                // Every cache line of the slot, ~1/32 of its bytes — a sweep that COVERS the slot,
                // so unlike reading one line at the front it could actually be a fix.
                crate::fetch::asyncfetch::FoldProbe::Line => (slot, n, self.fold[ud].line_stride),
                // Same size, same bandwidth, same duration — a buffer that is NOT the slot.
                crate::fetch::asyncfetch::FoldProbe::Decoy => (self.fold[ud].decoy, n, 1),
                // Same launch and the same stream-boundary cache maintenance, ~no work.
                crate::fetch::asyncfetch::FoldProbe::Nop => (self.fold[ud].decoy, 1, 1),
            };
            let r = match count {
                0 => Ok(()),
                // SAFETY: `buf` owns `count` readable f32 — the pool slot (`nbytes` bytes, live
                // until this read's signal) or the decoy (allocated slot-sized). `sc` is one live
                // device u64; `stream` is live and the copy is already enqueued on it, so any arm
                // that reads the slot does so after it.
                c => unsafe {
                    rivoli_backend::launch_hash_rows(
                        buf,
                        c,
                        stride,
                        (ud as u64) * n as u64,
                        self.fold[ud].sc,
                        stream,
                    )
                },
            };
            if let Err(e) = r {
                tracing::error!("divergence probe: sc fold failed on slot {ud} ({e:#})");
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

    /// Copies issued per path, `[hipMemcpyAsync, shader kernel, host memcpy]` — the
    /// OBSERVATION a candidate-fix arm is read off, as against the flag it was asked for.
    pub fn copies_issued(&self) -> [u64; 3] {
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
        // SAFETY: in bounce mode `arena` came from `stage::alloc` and `new` refuses to
        // build a `Streamer` around a null one, so this is a live allocation; single
        // owner, no Clone, so it is freed exactly once. DIRECT mode allocated none.
        match self.bounce {
            true => unsafe { stage::free(self.arena) },
            // DIRECT allocated no arena; its only allocation is the `--slot-refresh` sink page,
            // null when that flag is off.
            false if !self.refresh_sink.is_null() => unsafe {
                stage::free(self.refresh_sink.cast::<u8>())
            },
            false => {}
        }
        // The `--arena-refresh-decoy` arena and the `--cpu-retouch` scratch, each allocated
        // only when its arm is on (null otherwise).
        // SAFETY: as the arena above — live, single owner, freed once.
        if !self.arena_refresh_decoy.is_null() {
            unsafe { stage::free(self.arena_refresh_decoy) };
        }
        if !self.cpu_retouch.is_null() {
            unsafe { stage::free(self.cpu_retouch) };
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
