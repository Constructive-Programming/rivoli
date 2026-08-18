//! **The minimal exhibit for the bounce-arena hop, at rate.**
//!
//! Usage: `arena_repro <layer-file> <iters> [stride-bytes] [--coherent] [--copy-by-kernel]`
//! Run under `flock /var/run/sys-gpu.lock` — see "GPU tenancy" below.
//!
//! Phase 1 localised GLM's nondeterminism to the pinned bounce arena: every probe cell that does
//! not read it diverges, and both cells that do are clean. The suspected gap is that io_uring's CQE
//! establishes the NVMe DMA's writes are visible to the CPU while the agent that actually reads the
//! arena is the GPU's copy path, and nothing orders those two — see
//! `kernels/async.hip::rivoli_pinned_alloc` for the full chain.
//!
//! The engine reproduces that at roughly **one event per 40,000–80,000 reads**, i.e. a 20-minute
//! arm per sample. This does the same reads in a hot loop and verifies every one, which turns the
//! same statistics into minutes — and if it fires, it is the minimal driver-level exhibit, because
//! it contains no model, no MoE and no scheduling.
//!
//! ## What it does, and the three things that make it faithful
//!
//! It drives the engine's own [`Streamer`] — the same io_uring O_DIRECT ring, the same
//! `hipHostMalloc`'d arena, the same `hipMemcpyAsync` — rather than a re-implementation, so a
//! reproduction here is a reproduction of the shipping path and not of a lookalike.
//!
//! 1. **Staging slots are REUSED round-robin**, because that is the condition the hypothesis needs:
//!    a slot's previous contents were themselves read through the GPU, so the GPU may hold a cached
//!    copy of that host range when the NVMe DMA overwrites it.
//! 2. **Consecutive uses of one slot carry DIFFERENT payloads.** If a slot were refilled with the
//!    same bytes, a stale line would be invisible — the read would return the right answer for the
//!    wrong reason. So it cycles over `EXPERTS` distinct block offsets, and the count is chosen
//!    coprime to the slot count so (slot, previous tenant) pairs keep changing.
//! 3. **It never reads the arena.** Verification folds the DESTINATION on the device after the copy
//!    — the `sc` position, which was measured NOT to suppress. Reading the arena is the one thing
//!    that makes the defect disappear, so a repro that verified there would be a repro of nothing.
//!
//! No host sync sits between the copy and its verifying fold: both are enqueued on the fetch stream
//! and the whole batch is joined once, which is how the engine orders them too. The reference folds
//! are computed once from BUFFERED reads of the same ranges, so a mismatch means the O_DIRECT +
//! copy path delivered something the ordinary read path does not.
//!
//! ## GPU tenancy: this is NOT a deviceless test
//!
//! The plan that called for this described it as having "zero GPU tenancy". That is not true of this
//! implementation and could not be: `hipHostMalloc` initialises the HIP runtime and the copy and the
//! verifying fold are device work, so it holds a KFD entry and appears to every witness. **Take the
//! flock.** What it does avoid is a 281 GB model load and a decode, so it is cheap to run between
//! GPU cells — not free of them.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a repro harness dies loudly

use anyhow::{Context, Result, bail};
use rivoli_engine::fetch::stream::{ALIGN, ReadDst, ReadSpan, Streamer, slot_span};
use std::os::fd::AsRawFd;

/// Distinct block offsets cycled through, so a staging slot's payload changes every time it is
/// reused. 17 is coprime to the 16 staging slots, which keeps the (slot, previous tenant) pairing
/// moving instead of repeating every cycle.
const EXPERTS: usize = 17;
/// Staging slots — the io_uring ring depth, a power of two as `Streamer::new` requires.
const SLOTS: u32 = 16;

// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE — two examples sharing a module cannot avoid this line.
//
// The matched region is `mod common; fn main() -> Result<()> { let args = common::start();` — 26
// tokens of language scaffolding that IS the sharing. Cargo compiles each example independently,
// so both must declare the module and both must call it; the only way to stop the text matching is
// to stop sharing, which is strictly more duplication (the previous shape had the logging blocks
// themselves cloned, and widening `common` from `logging()` to `start()` removed the last of the
// real overlap). The alternative — reshaping one `main` to differ — would be changing code to suit
// the gate rather than the reader, which this repo rejects.
//
// The exemption is deliberately on THIS file only, and it is minimal: it covers the declaration and
// the first statement, nothing below. `glm_smoke.rs` carries no marker, so the gate still watches
// the whole of the other side of the pair.
mod common;

fn main() -> Result<()> {
    let args = common::start();
    // jscpd:ignore-end
    let coherent = args.iter().any(|a| a == "--coherent");
    // Phase 3B in the repro too: the same candidate fix, so a firing repro can be re-run against it
    // without touching the engine.
    let by_kernel = args.iter().any(|a| a == "--copy-by-kernel");
    // ARENA REFRESH: the engine-side mitigation, testable here too so a firing repro can be
    // re-run against it without the engine.
    let refresh = args.iter().any(|a| a == "--arena-refresh");
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [file, iters, rest @ ..] = pos.as_slice() else {
        bail!("usage: arena_repro <layer-file> <iters> [stride-bytes] [--coherent]");
    };
    let iters: usize = iters.parse().context("iters")?;
    // Default 15,335,424 B = the GLM `.vq3` expert stride (`4096 + 257 * stride` reproduces
    // `L03.vq3`'s size exactly). Overridable so the same harness can carry another format's width.
    let stride: usize = match rest.first() {
        Some(s) => s.parse().context("stride-bytes")?,
        None => 15_335_424,
    };
    if !stride.is_multiple_of(ALIGN) {
        bail!("stride {stride} must be a multiple of the {ALIGN}-byte O_DIRECT block");
    }

    // Two handles on purpose: O_DIRECT for the path under test, buffered for the reference. The
    // reference must come from a DIFFERENT path, or a corruption common to both would cancel.
    // O_DIRECT is 0o40000 on Linux; spelled here rather than pulling in a libc dependency for one
    // constant, and asserted by the open failing loudly if it is wrong for this platform.
    let direct = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0o40000)
        .open(file)
        .with_context(|| format!("open O_DIRECT {file}"))?;
    let buffered = std::fs::File::open(file).with_context(|| format!("open {file}"))?;

    // Reference folds, one per cycled offset, from the buffered path.
    let mut want = Vec::with_capacity(EXPERTS);
    let mut host = vec![0u8; stride];
    for e in 0..EXPERTS {
        let off = (ALIGN + e * stride) as u64;
        read_exact_at(&buffered, &mut host, off)?;
        want.push(rivoli_core::hash::xor_fold(as_f32(&host)));
    }
    tracing::info!(
        "reference: {EXPERTS} blocks of {:.2} MiB from the buffered path",
        stride as f64 / (1u64 << 20) as f64
    );

    // `true` = bounce. The repro exists to exercise the arena hop, so the arena-less
    // `--direct-vmm-dma` destination is not one of its arms by construction.
    let mut streamer = Streamer::new(SLOTS, slot_span(stride), coherent, by_kernel, refresh, true)?;
    let fetch = rivoli_backend::Stream::fetch()?;
    // One destination per staging slot, so a batch's copies never share a destination — the engine
    // has the same property (a slot's ticket gates its reuse).
    let mut dst: Vec<rivoli_engine::device::DeviceBuf> = (0..SLOTS)
        .map(|_| rivoli_engine::device::DeviceBuf::new(stride))
        .collect::<Result<_>>()?;
    let mut folds = rivoli_engine::device::DeviceBuf::new(SLOTS as usize * 8)?;
    let mut got = vec![0u8; SLOTS as usize * 8];

    let n_f32 = stride / 4;
    let t0 = std::time::Instant::now();
    let (mut reads, mut bad) = (0usize, 0usize);
    // One BATCH per outer step: `SLOTS` reads queued together, then reaped, each reap enqueueing its
    // copy and then its verifying fold on the same stream, then ONE join. That shape is the engine's
    // — no host sync between a copy and the read that consumes it.
    let mut e = 0usize;
    while reads < iters {
        let batch = SLOTS as usize;
        folds.copy_in_at(0, &vec![0u8; SLOTS as usize * 8])?;
        let mut expect = Vec::with_capacity(batch);
        for (slot, d) in dst.iter_mut().enumerate().take(batch) {
            let off = ALIGN + e * stride;
            // SAFETY: `dst[slot]` owns `stride` >= the aligned superset of this read, is
            // ALIGN-aligned (device allocation), and stays live for the whole batch.
            unsafe {
                streamer.queue(
                    direct.as_raw_fd(),
                    ReadSpan {
                        begin: off,
                        len: stride,
                    },
                    ReadDst {
                        ptr: d.ptr_mut(),
                        slot: slot as u32,
                        fold: rivoli_engine::fetch::asyncfetch::FetchFolds::OFF,
                    },
                )?
            };
            expect.push(want[e]);
            e = (e + 1) % EXPERTS;
        }
        streamer.submit()?;
        for _ in 0..batch {
            // SAFETY: `fetch` is a live stream; each destination outlives the batch.
            let slot = unsafe { streamer.reap(fetch.raw())? };
            // The VERIFYING fold, at the `sc` position: after the copy, on the same stream, reading
            // the DESTINATION. Never the arena — that is the read which makes the defect vanish.
            // SAFETY: `dst[slot]` holds `stride` bytes the copy just targeted; `folds` owns
            // `SLOTS * 8` bytes and `slot < SLOTS`.
            unsafe {
                rivoli_backend::launch_hash_rows(
                    dst[slot].ptr() as *const f32,
                    n_f32,
                    1,
                    0,
                    (folds.ptr_mut() as *mut u64).add(slot),
                    fetch.raw(),
                )?
            };
        }
        rivoli_backend::device_sync()?;
        folds.copy_out_into(&mut got)?;
        for (slot, w) in expect.iter().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&got[slot * 8..slot * 8 + 8]);
            let g = u64::from_le_bytes(b);
            if g != *w {
                bad += 1;
                tracing::error!(
                    "MISMATCH read #{} slot {slot}: got {g:016x} want {w:016x} — the O_DIRECT + \
                     hipMemcpyAsync path delivered bytes the buffered path does not",
                    reads + slot
                );
            }
        }
        reads += batch;
        if reads % (batch * 256) == 0 {
            tracing::info!(
                "{reads} reads, {bad} mismatches, {:.0} reads/s",
                reads as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    // The bound is the point of a clean run, so it is printed rather than left to the reader: at 0
    // events over N reads the 95% upper bound on the per-read rate is 3/N (the rule-of-three).
    println!(
        "arena_repro: {reads} reads, {bad} MISMATCHES, {:.0} reads/s, {:.1} s, arena={}, \
         copies=[memcpy {} / kernel {}]",
        reads as f64 / secs,
        secs,
        match coherent {
            true => "hipHostMallocCoherent",
            false => "hipHostMallocDefault",
        },
        streamer.copies_issued()[0],
        streamer.copies_issued()[1],
    );
    if bad == 0 {
        println!(
            "  clean: 95% upper bound on the per-read rate is {:.2e} (rule of three, 3/N). The \
             engine implies ~1/40,000-80,000, so a clean run at N >= 1e6 is a real exclusion.",
            3.0 / reads as f64
        );
    }
    Ok(())
}

/// `f32` view of a byte buffer, for the host fold. The payload is packed indices and bf16 scales,
/// not floats — the fold is over raw bits, so the interpretation is irrelevant and only the length
/// has to be right.
fn as_f32(b: &[u8]) -> &[f32] {
    // SAFETY: `b` comes from a `Vec<u8>` whose length is a multiple of 4 (a multiple of ALIGN), and
    // `f32` has no invalid bit patterns. Alignment: `Vec<u8>`'s buffer is only 1-aligned in general,
    // so this reads through a possibly-underaligned pointer — which is why it is
    // `align_to` rather than a cast, and the prefix must come back empty.
    let (head, mid, tail) = unsafe { b.align_to::<f32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "reference buffer is not f32-aligned ({} head, {} tail)",
        head.len(),
        tail.len()
    );
    mid
}

/// `pread` the whole buffer at `off`, looping over short reads.
fn read_exact_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.read_at(&mut buf[done..], off + done as u64)?;
        if n == 0 {
            bail!("short reference read at {off}: {done} of {}", buf.len());
        }
        done += n;
    }
    Ok(())
}

use std::os::unix::fs::OpenOptionsExt;
