//! Vulkan kernels vs their CPU oracles — the `tests/kernel.rs` story for the second
//! backend. Compiles to nothing without `vulkan`.
//!
//! Separate file because `tests/kernel.rs` is `#![cfg(feature = "rocm")]` end to end.
//! The helpers below are deliberately the same ones it uses, so the kernel-porting
//! phase can hoist both files onto a shared module instead of rewriting either.
//!
//! # Two rules for whoever writes the next tranche
//!
//! **1. Do not read the Vulkan shader sources while writing an oracle.** Derive it from the
//! HIP original and from `src/math.rs`, which are the specification. An oracle written
//! by someone who has seen the shader is a consistency check wearing a correctness
//! check's clothes: it agrees with the implementation because it was copied from it,
//! and it will happily ratify a shared misreading of the HIP. The `fwd.hip` oracles
//! here were written under that constraint deliberately. It costs a little rework when
//! the two disagree, which is the entire point — that disagreement is the signal.
//!
//! `tests/oracle_independence.rs` enforces the detectable half: no test file may name
//! the shader directory. That is why this paragraph does not spell the path out — a
//! rule that trips its own tripwire has to be allowlisted, and an allowlisted rule
//! stops being a rule.
//!
//! **2. A byte-exact oracle must prove its INPUTS are unambiguous.** Where a test
//! compares quantised bytes, the test DATA is a source of cross-driver flake
//! independent of the code: a value landing on a rounding midpoint of the target
//! format can legitimately quantise either way, so the comparison is decided by the
//! driver's arithmetic accuracy rather than by the shader. Green here, red on someone
//! else's machine, shader innocent — and whoever debugs it starts in the kernel,
//! because that is where the failure appears.
//!
//! Use [`assert_quantization_unambiguous`] with the margin the RELEVANT SPEC
//! guarantees, not the accuracy the hardware happens to deliver. Vulkan promises 2.5
//! ULP on `FDiv` and correctly-rounded `FAdd`/`FMul`; a dot product accumulates. This
//! matters most in the fp8 MoE tranche, which compares quantised bytes throughout and
//! is where a seed-dependent flake would be most expensive to diagnose. Make the
//! failure message name the SEED, not the shader.
#![cfg(feature = "vulkan")]
#![allow(clippy::expect_used)]

use rivoli::device::{DeviceBuf, DeviceTier};
use rivoli::quant::{matvec_fp8, matvec_i8};
use rivoli::math::{E4M3_BLOCK, E4M3_MAX, f32_to_bf16, f32_to_e4m3, silu};
use rivoli::vk::{
    Buf, ROWS_PER_BLOCK, VALIDATION_ERRORS, device_sync, gpu, launch_append_kv, launch_argmax,
    launch_embed_i8_row, launch_gather_rope, launch_gemv_f32, launch_gemv_fp8, launch_gemv_i8,
    launch_rmsnorm, launch_rope,
    launch_swiglu, launch_vadd, memcpy_dtod,
};
use std::sync::atomic::Ordering;

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32v(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn dev(b: &[u8]) -> Buf {
    let mut d = Buf::new(b.len()).expect("alloc");
    d.write_at(0, b).expect("fill");
    d
}
/// Report the max error AND the threshold it was compared against. Printing the
/// margin is the point: a green oracle that passed on 100x of headroom looks exactly
/// like one that passed on 2x, and only one of them is evidence.
fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = want
        .iter()
        .zip(got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let tol = 1e-3 * mx + 1e-3;
    println!("{label}: err={err:.3e} tol={tol:.3e} margin={:.1}x", tol / err.max(f32::MIN_POSITIVE));
    assert!(err <= tol, "{label}: err={err:.3e} > tol={tol:.3e}");
}

struct Lcg(u64);
impl Lcg {
    /// Uniform in [-1, 1).
    ///
    /// `>> 32`, not `>> 33`: the old shift left 31 bits, which over `u32::MAX` gives
    /// [0, 0.5) and therefore `*2 - 1` in [-1, 0) — EVERY SAMPLE NEGATIVE. In a GEMV
    /// oracle that makes every product positive, so the sums grow instead of
    /// cancelling, `mx` inflates, and the relative tolerance turns into ~100x of
    /// headroom. It also meant no oracle ever exercised cancellation — the one regime
    /// where summation order matters, and the entire reason `wave_sum` is a fixed
    /// ladder.
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Plain f32 GEMV, ascending summation — the same shape as `quant.rs`'s matvec
/// oracles. The kernel reduces in a fixed 32-lane shuffle ladder, so it differs by
/// f32 rounding only, which is what `assert_close`'s tolerance covers.
fn matvec_f32(y: &mut [f32], x: &[f32], w: &[f32], i_dim: usize) {
    for (o, out) in y.iter_mut().enumerate() {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        *out = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// One `gemv_f32` dispatch + sync, returning the raw output bytes.
fn gemv(x: &Buf, w: &Buf, y: &mut Buf, o_dim: usize, i_dim: usize) -> Vec<u8> {
    launch(x, w, y, o_dim, i_dim);
    device_sync().expect("sync");
    read(y, o_dim)
}

/// Enqueue only — no sync. Used to build a multi-dispatch command buffer.
fn launch(x: &Buf, w: &Buf, y: &mut Buf, o_dim: usize, i_dim: usize) {
    // SAFETY: live Buf device addresses of the documented sizes; nothing is dropped
    // before the caller's device_sync.
    unsafe {
        launch_gemv_f32(
            x.ptr() as *const f32,
            w.ptr() as *const f32,
            o_dim,
            i_dim,
            y.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
}

fn read(y: &Buf, n: usize) -> Vec<u8> {
    readb(y, n * 4)
}

/// `read` in BYTES. The fwd kernels write u8 and u16 as well as f32, and their
/// contracts are bit patterns rather than numbers, so most of the checks below never
/// leave byte space.
fn readb(y: &Buf, bytes: usize) -> Vec<u8> {
    let mut out = Vec::new();
    y.read_into(&mut out, bytes).expect("out");
    out
}

/// Bytes of guard band allocated past the end of every output buffer.
const GUARD: usize = 256;

/// An output buffer with every byte 0xFF. One poison pattern covers all four formats
/// these kernels emit — f32 NaN, bf16 NaN, e4m3 NaN, and a u8 no scale path produces —
/// so the guard band is the same check everywhere. `f32_to_e4m3` only ever returns
/// 0xFF for a NaN input, and no test here feeds one, so a surviving 0xFF is always
/// "nothing wrote here" rather than a legitimate result.
fn poison(bytes: usize) -> Buf {
    dev(&vec![0xFFu8; bytes])
}

/// Nothing was written at or past byte `end`. Cheap, and the standard — an overrun
/// into an allocation's rounding slack is otherwise invisible.
fn assert_untouched(got: &[u8], end: usize, label: &str) {
    let n = got[end..].iter().filter(|&&b| b != 0xFF).count();
    assert_eq!(n, 0, "{label}: wrote {n} bytes past byte {end}");
}

/// Byte-exact compare, reporting the FIRST difference and how many there are.
/// `assert_eq!` on two multi-kilobyte `Vec<u8>`s dumps both in full and buries the one
/// index that matters.
fn assert_bytes(want: &[u8], got: &[u8], label: &str) {
    assert_eq!(want.len(), got.len(), "{label}: length");
    let diff = want.iter().zip(got).filter(|(a, b)| a != b).count();
    match want.iter().zip(got).position(|(a, b)| a != b) {
        None => println!("{label}: {} bytes exact", want.len()),
        Some(i) => panic!(
            "{label}: {diff} of {} bytes differ; first at {i}: want {:#04x} got {:#04x}",
            want.len(),
            want[i],
            got[i]
        ),
    }
}

/// Snapshots the validation counter on construction and asserts NOTHING NEW arrived
/// by the time it is checked.
///
/// A delta, not an absolute. `VALIDATION_ERRORS` is process-global and cargo runs this
/// binary's tests on parallel threads, so asserting `== 0` means a message caused by
/// test A fails whichever test happens to read the counter next — the failure names
/// the wrong test and the maintainer debugs the wrong code. That misattribution is
/// worst for exactly the findings that matter here, since a threading diagnostic is
/// caused by the interaction BETWEEN two tests.
///
/// The delta still cannot attribute a concurrent message to its true source; it only
/// stops a test being blamed for one that predates it. Run with `--test-threads=1`
/// when a message actually appears.
struct Validation {
    at_entry: usize,
}

impl Validation {
    fn new() -> Self {
        Self {
            at_entry: VALIDATION_ERRORS.load(Ordering::Relaxed),
        }
    }

    /// Costs nothing when the layer is absent, and prints whether it was there at all
    /// — a green oracle from an unvalidated run must not read like a validated one.
    fn check(&self, label: &str) {
        let g = gpu().expect("vulkan init");
        let now = VALIDATION_ERRORS.load(Ordering::Relaxed);
        let n = now.saturating_sub(self.at_entry);
        println!(
            "{label}: validation layer {}",
            if g.validation() {
                "ON"
            } else {
                "OFF — THIS RUN IS UNVALIDATED"
            }
        );
        assert_eq!(n, 0, "{label}: {n} validation messages during this test");
    }
}

#[test]
fn gemv_f32_matches_oracle() {
    let v = Validation::new();
    let mut r = Lcg(0x3F2);
    // Shapes chosen for the edges, not for realism:
    //   256x5120  the router-gate shape the kernel exists for;
    //   255x512   o_dim not a multiple of ROWS_PER_BLOCK — the tail workgroup has
    //             idle subgroups that must not write;
    //   1x96      a single row, grid of one;
    //   33x97     NEITHER dim is a multiple of WAVE=32, so the strided inner loop
    //             leaves a partial final iteration and lanes diverge before wave_sum;
    //   7x33      both tiny and both ragged.
    for (o_dim, i_dim) in [
        (256usize, 5120usize),
        (255, 512),
        (1, 96),
        (33, 97),
        (7, 33),
    ] {
        let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();

        let mut want = vec![0.0f32; o_dim];
        matvec_f32(&mut want, &x, &w, i_dim);

        let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
        // Guard band: allocate a whole extra workgroup's worth of rows and poison
        // them. `o_dim = 255` leaves subgroup 7 of the tail workgroup out of range and
        // it must not write — but at exactly o_dim*4 bytes an overrun lands in the
        // allocation's rounding slack and nothing notices.
        let guard = ROWS_PER_BLOCK as usize;
        let mut yb = dev(&f32b(&vec![f32::NAN; o_dim + guard]));
        launch(&xb, &wb, &mut yb, o_dim, i_dim);
        device_sync().expect("sync");
        let out = f32v(&read(&yb, o_dim + guard));
        assert!(
            out[o_dim..].iter().all(|v| v.is_nan()),
            "gemv_f32 {o_dim}x{i_dim}: wrote past row {o_dim}"
        );
        assert_close(&want, &out[..o_dim], &format!("gemv_f32 {o_dim}x{i_dim}"));
    }
    v.check("gemv_f32_matches_oracle");
}

/// The inter-dispatch barrier in `Gpu::enqueue`, exercised hard enough that its
/// ABSENCE is detectable — and repeated often enough that one execution is a real
/// guard rather than a coin flip.
///
/// SIZING. Measured, not guessed. With the barrier deleted:
///
///   32-step chain, 2048x2048  ->  2 of 8 executions FAIL
///    4-step chain, 2048x2048  ->  0 of 8 (undetectable)
///    2-step chain, 64x96      ->  0 of 8 (undetectable, the original size)
///
/// and 8 of 8 pass in every configuration with the barrier present. So a missing
/// barrier is invisible below some threshold, and the shape that crosses it is: enough
/// workgroups to fill the machine several times over (256 per dispatch here), every
/// output row depending on every input element, and a chain long enough that one
/// escaped hazard survives to the end.
///
/// DETECTION RATE — MEASURED, AND WORSE THAN THE ARITHMETIC PREDICTS.
///
///   1 chain per execution, 7.4 s   ->  2 of 8 executions detect a removed barrier
///   16 chains per execution, 1.8 s ->  1 of 8
///   control (barrier present)      ->  0 of 8 in both configurations
///
/// Sixteen times the chains, HALF the detection. I predicted ~99% from
/// 1 - 0.75^16 and that was wrong, because the repeats are not independent and,
/// worse, because the change that added them also made each chain faster: fixing a
/// 512 MB duplicate-upload waste dropped the runtime 7.4 s -> 1.8 s, and those 32
/// interleaved 16 MB uploads were part of what opened the timing window. Optimising
/// the test removed the conditions that exposed the race.
///
/// So: **~12% detection per execution, measured.** A green run is very weak evidence
/// that a refactor kept the barrier. This test's value is now mostly that it CAN fail
/// at all — it is a smoke alarm with a flat battery, not a guard. Do not treat a
/// passing CI run as proof of ordering.
///
/// If you want a real guard back, the lever is not the repeat count: it is restoring
/// whatever made the chains slow and irregular. Interleaving unrelated large transfers
/// between the dispatches is the obvious candidate, since that is what was removed.
/// Measure the removal arm again after any change here — the whole point is that this
/// number does not follow from reasoning.
///
/// COST. The host oracle (32 chained 2048x2048 matvecs, in a debug build) dominates
/// and is computed once; the repeats add GPU work only. Worth it — this is the only
/// empirical check on the most consequential unverifiable property in the backend,
/// since synchronisation validation on this stack sees no compute hazard class at all
/// (docs/probes/README.md).
#[test]
fn chained_dispatch_respects_the_barrier() {
    let v = Validation::new();
    let mut r = Lcg(0xBA2);
    let n = 2048usize;
    const STEPS: usize = 32;
    /// Measured at ~12% detection per execution, NOT the ~99% that
    /// 1 - 0.75^REPEATS would suggest — see the docstring. Kept at 16 because it is
    /// cheap (1.8 s) and more chains cannot hurt; the number is not load-bearing.
    const REPEATS: usize = 16;

    // Near-identity: unit diagonal plus small noise. A random matrix chained 32 deep
    // either explodes or collapses into noise, and a corrupted intermediate then hides
    // inside the arithmetic. This keeps every step O(1) so a stale row survives to the
    // output as a visible error.
    //
    // A DISTINCT matrix per step, 32 x 16 MB = 512 MB. THIS IS NOT AN OVERSIGHT AND
    // MUST NOT BE "OPTIMISED" INTO ONE REUSED BUFFER. It was, once: reusing a single
    // matrix left it resident in cache, removed the DRAM traffic between dispatches,
    // took the runtime from 7.4 s to 1.8 s — and HALVED the detection rate, 2/8 to 1/8,
    // because the memory pressure was part of what opened the timing window. The waste
    // is the instrument. Re-measure the removal arm before changing it (see the
    // detection-rate note above); the number does not follow from reasoning.
    //
    // Cheap to build despite the size: only the diagonal and superdiagonal are nonzero,
    // so it is 2n random values over a zeroed allocation, not n^2.
    let mats: Vec<Vec<f32>> = (0..STEPS)
        .map(|_| {
            let mut m = vec![0.0f32; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
                m[i * n + (i + 1) % n] = 0.05 * r.f();
            }
            m
        })
        .collect();
    let x: Vec<f32> = (0..n).map(|_| r.f()).collect();

    let mut want = x.clone();
    for m in &mats {
        let mut next = vec![0.0f32; n];
        matvec_f32(&mut next, &want, m, n);
        want = next;
    }

    let mbufs: Vec<Buf> = mats.iter().map(|m| dev(&f32b(m))).collect();
    let xb = dev(&f32b(&x));
    let mut ping = dev(&f32b(&vec![0.0f32; n]));
    let mut pong = dev(&f32b(&vec![0.0f32; n]));

    for rep in 0..REPEATS {
        launch(&xb, &mbufs[0], &mut ping, n, n);
        for (i, mb) in mbufs.iter().enumerate().skip(1) {
            if i % 2 == 1 {
                launch(&ping, mb, &mut pong, n, n);
            } else {
                launch(&pong, mb, &mut ping, n, n);
            }
        }
        device_sync().expect("sync"); // ONE sync per chain, after all STEPS dispatches
        // Launch k writes ping for even k and pong for odd k, so after STEPS launches
        // the result is in pong when STEPS is even. (Had this inverted first; the test
        // then failed identically with AND without the barrier, which is what caught
        // it — see docs/probes/README.md, "A test built to fail needs its passing arm
        // checked too".)
        let out = if STEPS.is_multiple_of(2) { &pong } else { &ping };
        assert_close(&want, &f32v(&read(out, n)), &format!("{STEPS}-step chain #{rep}"));
    }
    v.check("chained_dispatch");
}

/// Greedy decode must be reproducible run to run, which is a STRICTER property than
/// the oracle's 1e-3 accuracy: a reduction whose order varies with workgroup
/// scheduling passes that tolerance happily. So compare BIT PATTERNS across repeats,
/// with a differently-shaped dispatch interleaved to perturb the scheduler.
///
/// This is why `wave_sum` is a fixed `subgroupShuffleDown` ladder and not
/// `subgroupAdd`. Every kernel with a reduction gets this test as it is ported —
/// `gemv_fp8_splitk` and `mla_attend_combine` especially, since they reduce across
/// workgroup partials in LDS rather than within one subgroup.
#[test]
fn gemv_f32_is_bit_reproducible() {
    let v = Validation::new();
    let mut r = Lcg(0xDE7);
    let (o_dim, i_dim) = (255usize, 1024usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let mut yb = dev(&vec![0u8; o_dim * 4]);

    // A decoy of a different shape, dispatched between repeats so the two runs do not
    // see identical queue state.
    let (dx, dw) = (dev(&f32b(&x[..96])), dev(&f32b(&w[..96 * 33])));
    let mut dy = dev(&[0u8; 33 * 4]);

    let first = gemv(&xb, &wb, &mut yb, o_dim, i_dim);
    for i in 1..5 {
        gemv(&dx, &dw, &mut dy, 33, 96);
        let again = gemv(&xb, &wb, &mut yb, o_dim, i_dim);
        assert_eq!(first, again, "gemv_f32 not bit-reproducible on repeat {i}");
    }
    // Guard against comparing two buffers of zeros and calling it determinism.
    assert!(
        f32v(&first).iter().any(|v| v.abs() > 1e-6),
        "output is all zero — the test proves nothing"
    );
    v.check("gemv_f32_is_bit_reproducible");
}

/// `memcpy_dtod` (arena slot relocation) reading what a dispatch just wrote, in ONE
/// command buffer. The copy's TRANSFER_READ must be ordered after the shader's
/// SHADER_WRITE, which is a COMPUTE -> TRANSFER dependency — a class the barrier in
/// `Gpu::enqueue` did not cover until this test forced it to.
///
/// Run this under `VK_LAYER_VALIDATE_SYNC=1`. The numeric assertion alone is weak
/// here: an unsynchronised copy usually still returns the right bytes on this driver,
/// which is exactly why the hazard is worth a checker rather than an oracle.
#[test]
fn memcpy_dtod_after_dispatch_is_ordered() {
    let v = Validation::new();
    let mut r = Lcg(0xC0B);
    let (o_dim, i_dim) = (128usize, 256usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    matvec_f32(&mut want, &x, &w, i_dim);

    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));
    let zb = dev(&f32b(&vec![f32::NAN; o_dim]));

    // Dispatch writes y, then the copy reads y — no sync in between, so both land in
    // the same command buffer and only the barrier separates them.
    launch(&xb, &wb, &mut yb, o_dim, i_dim);
    // SAFETY: both are live Buf device addresses, o_dim*4 bytes, distinct allocations.
    unsafe { memcpy_dtod(zb.ptr() as *mut u8, yb.ptr(), o_dim * 4).expect("dtod") };

    let got = f32v(&read(&zb, o_dim));
    assert_close(&want, &got, "memcpy_dtod after dispatch");
    v.check("memcpy_dtod_after_dispatch_is_ordered");
}

/// Drive a future to completion on this thread. Fifteen lines of std beats pulling in
/// a runtime for one test — `asyncfetch.rs` brings tokio when it is actually ported.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// The timeline-semaphore Signal resolves, and at what cost.
///
/// The HIP side gets this from `hipLaunchHostFunc` at ~19 us/signal, measured by
/// `gpustream::tests::signal_resolves_and_latency`. Vulkan has no host-callback-on-
/// queue, so this is one waiter thread in `vkWaitSemaphores` instead — and the plan
/// explicitly says to MEASURE whether a thread wakeup beats 19 us rather than assume.
/// The decode path arms roughly 9 per layer over 78 layers per token, so the number
/// decides whether the async overlap is affordable at all.
#[test]
fn timeline_signal_resolves_and_latency() {
    let v = Validation::new();
    let g = gpu().expect("vulkan init");
    // Warm the waiter thread and the first submit.
    block_on(g.signal().expect("arm"));

    let n = 200u32;
    let t = std::time::Instant::now();
    for _ in 0..n {
        block_on(g.signal().expect("arm"));
    }
    let us = t.elapsed().as_nanos() as f64 / f64::from(n) / 1000.0;
    println!("\nVK TIMELINE-SIGNAL round-trip: {us:.2} us/signal ({n} iters)\n");
    // Sanity ceiling only, matching the HIP test: if a bare signal costs >1 ms the
    // pipeline is a non-starter and the number matters more than the assertion.
    assert!(us < 1000.0, "signal latency {us:.1}us implausibly high");
    v.check("timeline_signal_resolves_and_latency");
}

/// A resolved signal is immediately ready, and `resolve` is idempotent — the error
/// path in the reaper depends on both, so awaiters never hang.
#[test]
fn signal_ready_and_resolve_are_immediate() {
    block_on(rivoli::vk::Signal::ready());
    let s = rivoli::vk::Signal::pending();
    s.resolve();
    s.resolve(); // idempotent
    block_on(s);
}

/// A kernel reads weights placed through `DeviceTier`, at the address `place`
/// returned.
///
/// This is the one line in the backend that converts between the two bases —
/// `place` writes through the HOST mapping and hands back a DEVICE address
/// (`slab.ptr() as usize + off`) — and docs/VULKAN.md calls that split the biggest
/// structural difference in the port. The unit test in device.rs proves only the
/// host-side arithmetic: it reads back through the host mapping, so a base swap, a
/// sign error, or an offset that does not track between the two mappings would pass
/// it and then produce garbage weights at integration.
///
/// Handing the returned address to a real dispatch is the only thing that closes
/// that. It is also a miniature of what `pin.rs` does with every resident weight.
#[test]
fn kernel_reads_weights_placed_through_the_tier() {
    let v = Validation::new();
    let mut r = Lcg(0x71E);
    let (o_dim, i_dim) = (64usize, 128usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    matvec_f32(&mut want, &x, &w, i_dim);

    let mut tier = DeviceTier::new(4 << 20).expect("tier");
    // Two placements, so the second sits at a non-zero bump offset — an offset that
    // failed to track between the host and device bases would only show up there.
    let wp = tier.place(&f32b(&w)).expect("place w");
    let xp = tier.place(&f32b(&x)).expect("place x");
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));

    // SAFETY: both are DEVICE addresses returned by `place`, sized as the launcher
    // documents, inside a tier that outlives the sync below.
    unsafe {
        launch_gemv_f32(
            xp as *const f32,
            wp as *const f32,
            o_dim,
            i_dim,
            yb.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    device_sync().expect("sync");
    assert_close(
        &want,
        &f32v(&read(&yb, o_dim)),
        "gemv over tier-placed weights",
    );
    v.check("kernel_reads_weights_placed_through_the_tier");
}

/// `DeviceBuf::copy_out_into` must see what a just-launched kernel wrote, WITHOUT the
/// caller syncing.
///
/// Its HIP twin is a blocking `hipMemcpy(..., D2H)` and every caller was written against
/// that contract. `gpu.rs:789` does `launch_gemv_f32(...)` then immediately
/// `gate_logits.copy_out_into(...)`; if the Vulkan version were a bare read of the host
/// mapping it would return the PREVIOUS token's gate logits, and routing would pick the
/// wrong experts on every layer of every token — coherently, with no error.
///
/// Found by review, not by a test, and untested until now because `gpu.rs` is rocm-only.
/// This closes it at unit scale, where a failure implicates one function, rather than at
/// integration where every kernel is simultaneously a suspect.
///
/// DETERMINISTIC, not a race: without the internal `device_sync` the dispatch is still
/// sitting unsubmitted in the open command buffer, so the read returns the sentinel
/// every time. No scaling needed — unlike the barrier test, which needed 32 chained
/// steps at 2048x2048 before a missing barrier became observable at all.
#[test]
fn copy_out_into_sees_the_dispatch_that_preceded_it() {
    let v = Validation::new();
    let mut r = Lcg(0x0D2);
    let (o_dim, i_dim) = (128usize, 256usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    matvec_f32(&mut want, &x, &w, i_dim);

    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    // A sentinel the kernel cannot produce, so "stale" and "correct" are never confusable.
    let sentinel = vec![-12345.0f32; o_dim];
    let mut out = DeviceBuf::new(o_dim * 4).expect("alloc");
    out.copy_in_at(0, &f32b(&sentinel)).expect("seed");

    // SAFETY: live device addresses of the documented sizes; nothing is dropped before
    // the copy_out_into below, which is what retires the dispatch.
    unsafe {
        launch_gemv_f32(
            xb.ptr() as *const f32,
            wb.ptr() as *const f32,
            o_dim,
            i_dim,
            out.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    // NO device_sync here. That is the whole test.
    let mut host = Vec::new();
    out.copy_out_into(&mut host).expect("copy_out_into");
    let got = f32v(&host);
    assert!(
        got.iter().all(|&g| g != -12345.0),
        "copy_out_into returned the SENTINEL — it read the host mapping without \
         retiring the dispatch, so a caller gets the previous contents (gpu.rs:789 \
         would route on the previous token's gate logits)"
    );
    assert_close(&want, &got, "copy_out_into after dispatch");
    v.check("copy_out_into_sees_the_dispatch_that_preceded_it");
}

/// `DeviceBuf::copy_in_at` must not overwrite bytes a recorded-but-unsubmitted dispatch
/// is going to read.
///
/// The mirror hazard of the test above, and the one that is easy to miss because it
/// runs the other way: HIP's H2D `hipMemcpy` blocks, so it is ordered AFTER any kernel
/// still reading the buffer. A bare host write into a mapped allocation is not — the
/// write lands before the submit, and the dispatch reads the NEW bytes. `gpu.rs:843`
/// (descs_vq / wexpert_buf) depends on the blocking behaviour.
///
/// Deterministic for the same reason: the dispatch has not been submitted, so without
/// the internal sync it is guaranteed to see the overwrite rather than merely likely to.
#[test]
fn copy_in_at_does_not_clobber_a_pending_dispatch() {
    let v = Validation::new();
    let mut r = Lcg(0x1A7);
    let (o_dim, i_dim) = (64usize, 128usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x_old: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    // Distinct enough that a result computed from the wrong one cannot pass the oracle.
    let x_new: Vec<f32> = x_old.iter().map(|v| v * 7.0 + 3.0).collect();
    let mut want = vec![0.0f32; o_dim];
    matvec_f32(&mut want, &x_old, &w, i_dim);

    let wb = dev(&f32b(&w));
    let mut xb = DeviceBuf::new(i_dim * 4).expect("alloc x");
    xb.copy_in_at(0, &f32b(&x_old)).expect("seed old");
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));

    // SAFETY: as above; `xb` outlives the sync inside copy_in_at.
    unsafe {
        launch_gemv_f32(
            xb.ptr() as *const f32,
            wb.ptr() as *const f32,
            o_dim,
            i_dim,
            yb.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    // Overwrite the INPUT the pending dispatch reads. copy_in_at must retire it first.
    xb.copy_in_at(0, &f32b(&x_new)).expect("clobber");
    device_sync().expect("sync");

    let got = f32v(&read(&yb, o_dim));
    assert_close(&want, &got, "gemv used x_old despite the overwrite");
    v.check("copy_in_at_does_not_clobber_a_pending_dispatch");
}

/// The guards report errors rather than tripping a Vulkan VUID or allocating nothing.
#[test]
fn guards_reject_degenerate_arguments() {
    let v = Validation::new();
    assert!(Buf::new(0).is_err(), "Buf::new(0) must be rejected");

    let mut b = dev(&[0u8; 64]);
    assert!(b.write_at(0, &[0u8; 65]).is_err(), "overlong write");
    assert!(b.write_at(64, &[0u8; 1]).is_err(), "write at the end");
    // The bounds check must not wrap: off + len overflows usize here, and an
    // `off + len <= self.len` test would compute a small number and let it through.
    assert!(b.write_at(usize::MAX, &[0u8; 8]).is_err(), "wrapping offset");
    assert!(b.read_into(&mut Vec::new(), 65).is_err(), "overlong read");

    let mut y = dev(&[0u8; 16]);
    // SAFETY: zero dims are rejected before any pointer is used.
    unsafe {
        assert!(
            launch_gemv_f32(b.ptr() as *const f32, b.ptr() as *const f32, 0, 4, y.ptr_mut() as *mut f32)
                .is_err(),
            "o_dim = 0 must be rejected"
        );
        assert!(
            launch_gemv_f32(b.ptr() as *const f32, b.ptr() as *const f32, 4, 0, y.ptr_mut() as *mut f32)
                .is_err(),
            "i_dim = 0 must be rejected"
        );
    }
    // Rejecting a bad argument must not itself trip the layer — a guard that returns
    // Err while leaving a half-built Vulkan object behind would show up here.
    v.check("guards_reject_degenerate_arguments");
}

// ---------------------------------------------------------------------------
// fwd.hip glue: embed_i8_row, gather_rope, vadd, append_kv, argmax_reduce.
//
// Oracles are written against kernels/fwd.hip and src/math.rs, NOT against the GLSL —
// a test derived from the shader it checks agrees with the shader by construction.
// ---------------------------------------------------------------------------

/// `x[i] = (i8)packed[token·hidden + i] · scale[token]`.
///
/// The shader loads a `u32` word and sign-extends one byte out of it by hand, so the
/// failure mode that matters is a byte read as UNSIGNED: invisible for every value
/// below 0x80 and a 256·scale error at every value above it. Every row therefore
/// starts with the four extremes {-128, -1, 127, 0}.
///
/// `hidden = 37` is neither a multiple of 4 nor of 32, so (a) the last workgroup has
/// idle threads and (b) the row base `token·37` walks byte phases 0,1,2,3,0 over the
/// five tokens — the same four extremes are extracted from a different position within
/// the word on each iteration, which is the whole reason to loop over tokens at all.
#[test]
fn embed_i8_row_sign_extends_and_scales() {
    let v = Validation::new();
    let (vocab, hidden) = (5usize, 37usize);
    let mut packed: Vec<u8> = (0..vocab * hidden)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
        .collect();
    for t in 0..vocab {
        packed[t * hidden..t * hidden + 4].copy_from_slice(&[0x80, 0xFF, 0x7F, 0x00]);
    }
    // Distinct per row, so a kernel indexing `scale[0]` instead of `scale[token]` is a
    // failure rather than a coincidence.
    let scale: Vec<f32> = (0..vocab).map(|t| 0.013 * (t as f32 + 1.0)).collect();
    let (pb, sb) = (dev(&packed), dev(&f32b(&scale)));

    for token in 0..vocab {
        let want: Vec<f32> = (0..hidden)
            .map(|i| f32::from(packed[token * hidden + i] as i8) * scale[token])
            .collect();
        let mut xb = poison(hidden * 4 + GUARD);
        // SAFETY: live Buf device addresses of the documented sizes — `packed` is
        // vocab·hidden bytes, `scale` vocab f32, `x` hidden f32 plus the guard band.
        // Nothing is dropped before the sync.
        unsafe {
            launch_embed_i8_row(pb.ptr(), sb.ptr() as *const f32, token, hidden, xb.ptr_mut() as *mut f32)
                .expect("launch");
        }
        device_sync().expect("sync");
        let got = readb(&xb, hidden * 4 + GUARD);
        let label = format!("embed_i8_row token {token}");
        assert_untouched(&got, hidden * 4, &label);
        // A sign-extension bug is a 256·scale error and clears this tolerance by three
        // orders of magnitude — `assert_close`'s printed margin says which happened.
        assert_close(&want, &f32v(&got[..hidden * 4]), &label);
    }
    v.check("embed_i8_row_sign_extends_and_scales");
}

/// `qrope[head·ropn + d] = q[head·qh + nope + d]` — gather each head's roped segment
/// out of the strided `q` into a contiguous buffer.
#[test]
fn gather_rope_gathers_the_strided_segment() {
    let v = Validation::new();
    let mut r = Lcg(0x604);
    // (h, qh, nope, ropn), all ragged. The caller guarantees nope + ropn <= qh; the
    // last shape sits EXACTLY at qh, so the final head reads q's very last element and
    // an off-by-one in the offset runs off the end of the allocation.
    for (h, qh, nope, ropn) in [
        (7usize, 100usize, 37usize, 51usize),
        (1, 64, 0, 64), // nope = 0: the no-offset path, single head, grid of one
        (33, 97, 65, 32),
    ] {
        let q: Vec<f32> = (0..h * qh).map(|_| r.f()).collect();
        let want: Vec<f32> = (0..h * ropn)
            .map(|i| q[(i / ropn) * qh + nope + i % ropn])
            .collect();
        let out_bytes = h * ropn * 4;

        let qb = dev(&f32b(&q));
        let mut ob = poison(out_bytes + GUARD);
        // SAFETY: `q` is h·qh f32, `qrope` h·ropn f32 plus the guard band; both live
        // until the sync below.
        unsafe {
            launch_gather_rope(qb.ptr() as *const f32, ob.ptr_mut() as *mut f32, h, qh, nope, ropn)
                .expect("launch");
        }
        device_sync().expect("sync");
        let got = readb(&ob, out_bytes + GUARD);
        let label = format!("gather_rope h{h} qh{qh} nope{nope} ropn{ropn}");
        assert_untouched(&got, out_bytes, &label);
        // BYTES, not `assert_close`. This kernel performs no arithmetic, so any
        // tolerance at all would be slack for a bug rather than for rounding.
        assert_bytes(&f32b(&want), &got[..out_bytes], &label);
    }
    v.check("gather_rope_gathers_the_strided_segment");
}

/// `x[i] += y[i]`, in place.
#[test]
fn vadd_adds_into_x_in_place() {
    let v = Validation::new();
    let mut r = Lcg(0xADD);
    // 1 (one live thread in one workgroup), 255/257 either side of the 256-thread
    // workgroup, then ragged multi-workgroup sizes. None is a multiple of 256.
    for n in [1usize, 255, 257, 1000, 4097] {
        let x: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let y: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();

        let mut xb = poison(n * 4 + GUARD);
        xb.write_at(0, &f32b(&x)).expect("fill x");
        let yb = dev(&f32b(&y));
        // SAFETY: `x` is n f32 plus the guard band, `y` is n f32; both live until the
        // sync. `x` is read AND written by the kernel, which is the contract.
        unsafe {
            launch_vadd(xb.ptr_mut() as *mut f32, yb.ptr() as *const f32, n).expect("launch");
        }
        device_sync().expect("sync");
        // Read back x's OWN buffer, so this also proves the sum landed in place rather
        // than in some scratch the launcher forgot to point at x.
        let got = readb(&xb, n * 4 + GUARD);
        let label = format!("vadd n={n}");
        assert_untouched(&got, n * 4, &label);
        assert_close(&want, &f32v(&got[..n * 4]), &label);
    }
    v.check("vadd_adds_into_x_in_place");
}

/// The CPU side of `append_kv`, exactly as fwd.hip specifies it. Returns
/// (lc8 bytes, block scales, rc bytes) for ONE row.
///
/// `f32_to_e4m3` and `f32_to_bf16` are bit-exact contracts — the kernel comment says
/// "mirrors math.rs::f32_to_e4m3 bit-for-bit" — so the caller compares bytes. A float
/// tolerance would accept a quantizer that rounds the other way at every tie, which is
/// precisely the bug this file exists to catch.
fn append_kv_oracle(latent: &[f32], rope: &[f32]) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let scales: Vec<f32> = latent
        .chunks_exact(E4M3_BLOCK)
        .map(|blk| {
            let amax = blk.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            // amax == 0 -> scale 1.0, or the quantizer would divide by zero.
            if amax > 0.0 { amax / E4M3_MAX } else { 1.0 }
        })
        .collect();
    let lc8 = latent
        .iter()
        .enumerate()
        .map(|(i, x)| f32_to_e4m3(x / scales[i / E4M3_BLOCK]))
        .collect();
    let rc = rope
        .iter()
        .flat_map(|x| f32_to_bf16(*x).to_le_bytes())
        .collect();
    (lc8, scales, rc)
}

/// One latent of `kvl` values whose 128-blocks sit in deliberately different scale
/// regimes, plus `ropn` rope values.
fn append_kv_input(kvl: usize, ropn: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut r = Lcg(seed);
    let latent: Vec<f32> = (0..kvl)
        .map(|i| match i / E4M3_BLOCK {
            1 => 0.0,            // amax == 0 -> the scale = 1.0 branch
            2 => r.f() * 3000.0, // amax >> 448 -> a scale above one
            _ => r.f(),          // amax ~ 1 -> a scale near 1/448
        })
        .collect();
    let mut rope: Vec<f32> = (0..ropn).map(|_| r.f()).collect();
    // bf16 is a pure bit operation, so these are unconditional: the RNE tie (rounds
    // DOWN to the even neighbour 0x3f80), the value just past it (0x3f81), zero, and a
    // magnitude f32 carries exactly but that leaves no mantissa room in bf16.
    // ponytail: no -0.0 here. Its bf16 IS 0x8000, but Vulkan does not promise signed
    // zero preservation without an execution mode, so a 0x0000 would be ambiguous
    // between a shader bug and a missing SignedZeroInfNanPreserve — untestable from
    // here, and a probe's job rather than an oracle's.
    let specials = [1.0 + 2f32.powi(-8), 1.0 + 2f32.powi(-8) + 2f32.powi(-16), 0.0, 65504.0];
    rope[..specials.len()].copy_from_slice(&specials);
    (latent, rope)
}

/// Every value in `values` must quantise to the same result `margin_ulp` either side
/// of itself — i.e. none of them sits near a rounding midpoint of the target format.
///
/// THE STANDING PRECONDITION FOR ANY BYTE-EXACT ORACLE (see the module header). A test
/// that compares quantised bytes is only meaningful if the inputs cannot legitimately
/// quantise both ways: otherwise the driver's arithmetic accuracy decides the result,
/// the test passes here and fails elsewhere, and the shader is innocent.
///
/// `margin_ulp` must come from what the SPEC guarantees for the operations that produce
/// these values, not from what this GPU happens to do. Vulkan guarantees 2.5 ULP on
/// `FDiv`; `FAdd`/`FMul` are correctly rounded but a reduction accumulates, so a dot
/// product needs a margin that grows with its length.
///
/// The message names the SEED deliberately. When this fires the data is wrong, not the
/// kernel, and the reader should not start by auditing a shader.
fn assert_quantization_unambiguous<T: PartialEq>(
    label: &str,
    values: &[f32],
    margin_ulp: u32,
    quantize: impl Fn(f32) -> T,
) {
    for (i, &q) in values.iter().enumerate() {
        if q == 0.0 || !q.is_finite() {
            continue; // exact or saturating; no midpoint to sit on
        }
        let up = f32::from_bits(q.to_bits().wrapping_add(margin_ulp));
        let dn = f32::from_bits(q.to_bits().wrapping_sub(margin_ulp));
        let want = quantize(q);
        assert!(
            want == quantize(up) && want == quantize(dn),
            "{label}[{i}] = {q:e} quantises ambiguously within {margin_ulp} ULP: this \
             SEED cannot support a byte-exact compare — pick another. The kernel is not \
             implicated; see the byte-exact-oracle rule in this file's header."
        );
    }
}

/// `append_kv`'s use of the rule above: 8 ULP covers both divisions in the chain
/// (amax/448, then latent/scale) at Vulkan's 2.5 ULP `FDiv` guarantee, with room.
fn assert_quantization_is_unambiguous(latent: &[f32], scales: &[f32]) {
    let quotients: Vec<f32> = latent
        .iter()
        .enumerate()
        .map(|(i, x)| x / scales[i / E4M3_BLOCK])
        .collect();
    assert_quantization_unambiguous("latent/scale", &quotients, 8, f32_to_e4m3);
}

/// `append_kv`: latent -> fp8-e4m3 with a per-128 block scale, roped key -> bf16, both
/// at row `pos`.
///
/// The block amax is an LDS tree reduction over 128 lanes, which is where the
/// reproducibility test below points; this one is about the quantizer agreeing with
/// `math.rs` bit for bit and about the row offset.
#[test]
fn append_kv_quantizes_bit_exactly() {
    let v = Validation::new();
    // kvl must be a multiple of 128 in [128, 1024], so it cannot be ragged — ropn is
    // the only free size here and 100 is deliberately not a multiple of 32, leaving the
    // rope write live on a partial subgroup.
    // Three shapes, and the two new ones each isolate ONE previously untested path so a
    // failure is attributable:
    //   512/100  the original: both paths in their already-covered state.
    //   1024/100 kvl = MAX_BLOCKS*128, so every subgroup carries a block and
    //            `gl_NumSubgroups == 8` is load-bearing rather than slack. Rope stays
    //            under THREADS, so this shape can only implicate the block mapping.
    //   512/300  ropn > THREADS, so the rope loop takes a second grid-stride iteration
    //            that was dead code until now. Block count stays at the covered 4, so
    //            this shape can only implicate the rope loop.
    // Combining them (1024/300) would exercise both at once and, on a failure, tell you
    // nothing about which.
    for (kvl, ropn, rows, pos) in [
        (512usize, 100usize, 5usize, 3usize),
        (1024, 100, 3, 2),
        (512, 300, 4, 1),
    ] {
    let n_blocks = kvl / E4M3_BLOCK;
    let (latent, rope) = append_kv_input(kvl, ropn, 0xA55);
    let (want_lc8, want_scl, want_rc) = append_kv_oracle(&latent, &rope);
    assert_quantization_is_unambiguous(&latent, &want_scl);
    // The two branches the test exists for, asserted on the oracle so a future edit to
    // `append_kv_input` cannot silently drop either.
    assert_eq!(want_scl[1], 1.0, "block 1 must exercise the amax == 0 branch");
    assert!(want_scl[2] > 1.0, "block 2 must exercise a large amax");

    let (lb, rb) = (dev(&f32b(&latent)), dev(&f32b(&rope)));
    // Whole slabs, poisoned. Rows other than `pos` are as much of a guard band as the
    // trailing GUARD bytes: a row-offset bug lands in one of them, and an `i < ropn`
    // check missing from the rope write spills straight into row pos+1 of `rc`.
    let (lc8_n, lscale_n, rc_n) = (rows * kvl, rows * n_blocks * 4, rows * ropn * 2);
    let mut lc8 = poison(lc8_n + GUARD);
    let mut lscale = poison(lscale_n + GUARD);
    let mut rc = poison(rc_n + GUARD);
    // SAFETY: `latent` is kvl f32 and `rope` ropn f32; the three slabs hold `rows` rows
    // of their documented stride plus a guard band, and row `pos` is in bounds. All
    // five outlive the sync.
    unsafe {
        launch_append_kv(
            lb.ptr() as *const f32,
            rb.ptr() as *const f32,
            lc8.ptr_mut(),
            lscale.ptr_mut() as *mut f32,
            rc.ptr_mut() as *mut u16,
            pos,
            kvl,
            ropn,
            n_blocks,
        )
        .expect("launch");
    }
    device_sync().expect("sync");
    let g_lc8 = readb(&lc8, lc8_n + GUARD);
    let g_scl = readb(&lscale, lscale_n + GUARD);
    let g_rc = readb(&rc, rc_n + GUARD);

    for (got, stride, name) in [
        (&g_lc8, kvl, "lc8"),
        (&g_scl, n_blocks * 4, "lscale"),
        (&g_rc, ropn * 2, "rc"),
    ] {
        assert!(
            got[..pos * stride].iter().all(|&b| b == 0xFF),
            "append_kv {name}: wrote BEFORE row {pos}"
        );
        assert_untouched(got, (pos + 1) * stride, &format!("append_kv {name} row {pos}"));
    }
    assert_bytes(&want_lc8, &g_lc8[pos * kvl..(pos + 1) * kvl], "append_kv lc8");
    assert_bytes(&want_rc, &g_rc[pos * ropn * 2..(pos + 1) * ropn * 2], "append_kv rc");

    // The scale is the one float here — a division, for which Vulkan promises 2.5 ULP.
    // Compared per element RELATIVE rather than through `assert_close`: these four
    // scales span 1/448 to ~6.7, and a shared `mx` would hand the smallest of them
    // three orders of magnitude of slack. 1e-6 is ~8 ULP.
    let got_scl = f32v(&g_scl[pos * n_blocks * 4..(pos + 1) * n_blocks * 4]);
    let rel = want_scl
        .iter()
        .zip(&got_scl)
        .fold(0.0f32, |m, (w, g)| m.max((w - g).abs() / w.abs()));
    println!("append_kv {kvl}/{ropn} lscale: max rel err={rel:.3e} tol=1.0e-6");
    assert!(rel <= 1e-6, "append_kv lscale: rel err {rel:.3e} > 1e-6, want {want_scl:?} got {got_scl:?}");
    }
    v.check("append_kv_quantizes_bit_exactly");
}

/// `append_kv` reduces the block amax in LDS, so it gets the bit-reproducibility test
/// every reduction in this backend gets — a scheduling-dependent reduce would still
/// clear any tolerance, and here it would silently move the quantization scale.
#[test]
fn append_kv_is_bit_reproducible() {
    let v = Validation::new();
    let (kvl, ropn, pos) = (512usize, 100usize, 2usize);
    let n_blocks = kvl / E4M3_BLOCK;
    let (latent, rope) = append_kv_input(kvl, ropn, 0xB16);
    let (lb, rb) = (dev(&f32b(&latent)), dev(&f32b(&rope)));

    // A decoy of a DIFFERENT shape (one block, half the rope, row 0) dispatched between
    // repeats, into its own buffers, so the two runs do not see identical queue state.
    let (dlat, drope) = append_kv_input(E4M3_BLOCK, 64, 0xD00);
    let (dlb, drb) = (dev(&f32b(&dlat)), dev(&f32b(&drope)));

    // Zero-filled, not poisoned: the all-zero guard at the end has to be able to fail.
    let rows = pos + 1;
    let mut lc8 = dev(&vec![0u8; rows * kvl]);
    let mut lscale = dev(&vec![0u8; rows * n_blocks * 4]);
    let mut rc = dev(&vec![0u8; rows * ropn * 2]);
    let (mut dlc8, mut dlscale, mut drc) =
        (dev(&[0u8; E4M3_BLOCK]), dev(&[0u8; 4]), dev(&[0u8; 128]));

    let run = |lat: &Buf,
                   rop: &Buf,
                   lc8: &mut Buf,
                   lscale: &mut Buf,
                   rc: &mut Buf,
                   pos: usize,
                   kvl: usize,
                   ropn: usize,
                   n_blocks: usize| {
        // SAFETY: sizes as documented above; every Buf outlives the sync.
        unsafe {
            launch_append_kv(
                lat.ptr() as *const f32,
                rop.ptr() as *const f32,
                lc8.ptr_mut(),
                lscale.ptr_mut() as *mut f32,
                rc.ptr_mut() as *mut u16,
                pos,
                kvl,
                ropn,
                n_blocks,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        let mut out = readb(lc8, (pos + 1) * kvl);
        out.extend(readb(lscale, (pos + 1) * n_blocks * 4));
        out.extend(readb(rc, (pos + 1) * ropn * 2));
        out
    };

    let first = run(&lb, &rb, &mut lc8, &mut lscale, &mut rc, pos, kvl, ropn, n_blocks);
    for i in 1..5 {
        run(&dlb, &drb, &mut dlc8, &mut dlscale, &mut drc, 0, E4M3_BLOCK, 64, 1);
        let again = run(&lb, &rb, &mut lc8, &mut lscale, &mut rc, pos, kvl, ropn, n_blocks);
        assert_bytes(&first, &again, &format!("append_kv repeat {i}"));
    }
    // Row `pos` of lc8 alone — rows 0..pos are legitimately still zero, so checking the
    // whole buffer would be a weaker guard than it looks.
    assert!(
        first[pos * kvl..(pos + 1) * kvl].iter().any(|&b| b != 0),
        "lc8 row {pos} is all zero — the test proves nothing"
    );
    v.check("append_kv_is_bit_reproducible");
}

/// One `argmax` dispatch + sync -> the two output words as RAW BYTES, idx then val.
///
/// Each word gets its own poisoned tail. They are written by thread 0 only, and a
/// shader that spilled the whole workgroup's partials would still leave the right
/// value in the first word.
fn argmax_raw(logits: &[f32]) -> Vec<u8> {
    let lb = dev(&f32b(logits));
    let mut ib = poison(4 + GUARD);
    let mut vb = poison(4 + GUARD);
    // SAFETY: `logits` is logits.len() f32; each output is one word plus a guard band.
    // All three live until the sync.
    unsafe {
        launch_argmax(
            lb.ptr() as *const f32,
            logits.len(),
            ib.ptr_mut() as *mut i32,
            vb.ptr_mut() as *mut f32,
        )
        .expect("launch argmax");
    }
    device_sync().expect("sync");
    let (gi, gv) = (readb(&ib, 4 + GUARD), readb(&vb, 4 + GUARD));
    assert_untouched(&gi, 4, "argmax out_idx");
    assert_untouched(&gv, 4, "argmax out_val");
    let mut out = gi[..4].to_vec();
    out.extend_from_slice(&gv[..4]);
    out
}

fn argmax(logits: &[f32]) -> (i32, f32) {
    let b = argmax_raw(logits);
    (
        i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
    )
}

/// Both halves of the contract, always. An argmax that returns the right index with
/// the wrong value still breaks the caller, whose `!bv.is_finite()` bail reads the
/// value and not the index.
fn chk_argmax(label: &str, logits: &[f32], want_idx: i32, want_val: f32) {
    let (gi, gv) = argmax(logits);
    assert_eq!(gi, want_idx, "{label}: index (value was {gv})");
    assert_eq!(gv, want_val, "{label}: value (index was {gi})");
    println!("argmax {label}: n={} -> ({gi}, {gv})", logits.len());
}

/// `argmax_reduce` against the host fold it reproduces, adversarially.
///
/// This is the last op before the sampled token, so a wrong answer is a wrong output
/// with no numeric smell — no tolerance to blow, no NaN downstream, just a different
/// word. The contract, from fwd.hip: the LOWEST index of the maximum value wins, NaN
/// NEVER wins, and the returned value is `logits[best]`.
#[test]
fn argmax_matches_the_host_fold() {
    let v = Validation::new();

    chk_argmax("max at index 0", &[9.0, 1.0, 2.0, -3.0, 0.5], 0, 9.0);

    // n = 7: fewer elements than the 256-thread workgroup, so most lanes contribute
    // nothing but the identity and the LDS tree must not let one of them win.
    chk_argmax("plateau -> lowest index", &[0.1, 0.5, 0.5, 0.3, 0.5, -1.0, 0.5], 1, 0.5);

    let mut last = vec![-1.0f32; 1000];
    last[999] = 5.0;
    chk_argmax("max at the final position", &last, 999, 5.0);

    // 261 = 5 + 256: with a 256-thread block the SAME lane sees both, so this tie is
    // broken inside the grid-stride loop rather than in the LDS tree. The cross-lane
    // case is the 100_003 one below.
    let mut same_lane = vec![0.0f32; 300];
    same_lane[5] = 2.0;
    same_lane[261] = 2.0;
    chk_argmax("plateau within one lane", &same_lane, 5, 2.0);

    chk_argmax("all elements equal", &vec![1.5f32; 300], 0, 1.5);

    // NaN loses to any finite value: every compare against it is false, so it never
    // displaces the running best — including when it is the running best's only rival.
    chk_argmax("NaN loses to finite", &[0.1, f32::NAN, 0.7, f32::NAN, 0.3, 0.7], 2, 0.7);
    // Index 0 specifically: it is the identity's index, so a NaN there is the one
    // position where "NaN loses" and "ties go to the lowest index" could conspire.
    chk_argmax("NaN at index 0", &[f32::NAN, 0.2, 0.9, 0.4, -7.0], 2, 0.9);

    // Every element is the identity: only the tie rule decides, and logits[0] is -inf
    // so the returned value is exact rather than merely non-finite.
    chk_argmax("all -inf", &vec![f32::NEG_INFINITY; 257], 0, f32::NEG_INFINITY);

    // All negative — catches an oracle (or a shader) that initialises the running best
    // to 0 instead of -inf, which is invisible whenever any element is positive.
    chk_argmax("all negative", &[-5.0, -2.0, -9.0, -2.5, -1e30], 1, -2.0);
    let big_neg: Vec<f32> = (0..1000).map(|i| -((i as f32 - 613.0).abs()) - 3.0).collect();
    chk_argmax("all negative at scale", &big_neg, 613, -3.0);

    // 100_003: ragged, ~391 grid-stride passes per lane, a NaN in lane 0's first slot,
    // and a plateau whose two members land on DIFFERENT lanes (12_345 % 256 = 57,
    // 98_765 % 256 = 205) so the tie is resolved by the LDS tree. `Lcg::f` is strictly
    // inside [-1, 1), so 3.0 is the unique maximum.
    let mut r = Lcg(0x9A2);
    let mut big: Vec<f32> = (0..100_003).map(|_| r.f()).collect();
    big[0] = f32::NAN;
    big[12_345] = 3.0;
    big[98_765] = 3.0;
    chk_argmax("cross-lane plateau at 100_003", &big, 12_345, 3.0);

    // ALL NaN. Nothing can win, so the reduction returns its identity. fwd.hip
    // documents that identity as best = 0, bv = -inf, chosen so the host's
    // `!bv.is_finite()` bail fires — the requirement is therefore not a particular
    // value but a NON-FINITE one at a defined index, and the value is printed rather
    // than asserted so a change in it is visible without being a failure.
    let (gi, gv) = argmax(&vec![f32::NAN; 300]);
    println!("argmax all-NaN: -> ({gi}, {gv}) bits {:#010x}", gv.to_bits());
    assert_eq!(gi, 0, "all-NaN: index");
    assert!(!gv.is_finite(), "all-NaN: value {gv} is finite — the caller's bail would not fire");

    v.check("argmax_matches_the_host_fold");
}

/// `argmax_reduce` is a reduction, so it gets the bit-reproducibility test. Weaker
/// than it sounds for the index (an i32 is exact either way) and exactly the point for
/// the value: a reduce whose order varies with scheduling returns a different
/// `logits[best]` bit pattern on a plateau, and greedy decode would drift run to run.
#[test]
fn argmax_is_bit_reproducible() {
    let v = Validation::new();
    let mut r = Lcg(0x1D3);
    let mut logits: Vec<f32> = (0..100_003).map(|_| r.f() * 10.0).collect();
    // Nonzero index AND nonzero value, so the all-zero guard below is not vacuous.
    logits[77_777] = 42.5;
    // A differently-shaped dispatch between repeats, to perturb queue state.
    let decoy: Vec<f32> = (0..1291).map(|_| r.f()).collect();

    let first = argmax_raw(&logits);
    for i in 1..5 {
        argmax_raw(&decoy);
        assert_bytes(&first, &argmax_raw(&logits), &format!("argmax repeat {i}"));
    }
    assert!(
        first.iter().any(|&b| b != 0),
        "both output words are zero — the test proves nothing"
    );
    v.check("argmax_is_bit_reproducible");
}

/// The fwd launchers' argument guards report errors rather than dispatching a
/// degenerate grid or tripping a VUID.
///
/// `token`, `pos` and `nope` are `usize` here, so the HIP guards' `< 0` arms are
/// unreachable from Rust and are not tested — the only negative value that could reach
/// the shader is one produced by the `as i32` narrowing inside the launcher, which is
/// its business and not this file's.
#[test]
fn fwd_guards_reject_degenerate_arguments() {
    let v = Validation::new();
    let src = dev(&[0u8; 4096]);
    let mut d1 = dev(&[0u8; 4096]);
    let mut d2 = dev(&[0u8; 4096]);
    let mut d3 = dev(&[0u8; 4096]);
    let (p, a, b, c) = (src.ptr(), d1.ptr_mut(), d2.ptr_mut(), d3.ptr_mut());

    let akv = |kvl: usize, ropn: usize, n_blocks: usize| {
        // SAFETY: every one of these must be rejected by a shape guard before a pointer
        // is used; the four buffers are live and 4096 bytes regardless.
        unsafe {
            launch_append_kv(
                p as *const f32,
                p as *const f32,
                a,
                b as *mut f32,
                c as *mut u16,
                0,
                kvl,
                ropn,
                n_blocks,
            )
        }
    };
    assert!(akv(0, 8, 1).is_err(), "append_kv kvl = 0");
    assert!(akv(128, 0, 1).is_err(), "append_kv ropn = 0");
    assert!(akv(128, 8, 0).is_err(), "append_kv n_blocks = 0");
    // The one-block per-128 reduction needs kvl a multiple of 128 in [128, 1024], and
    // the rope half rides the same block so ropn cannot exceed it.
    assert!(akv(64, 8, 1).is_err(), "append_kv kvl < 128");
    assert!(akv(200, 8, 1).is_err(), "append_kv kvl not a multiple of 128");
    assert!(akv(1152, 8, 9).is_err(), "append_kv kvl > 1024");
    assert!(akv(256, 300, 2).is_err(), "append_kv ropn > kvl");
    // The one guard deliberately STRICTER than HIP: the shader packs u16 keys into u32
    // words, so an odd ropn would straddle a word and drop the tail. Untested until
    // now, which meant the divergence from HIP was asserted only in a comment.
    assert!(akv(256, 99, 2).is_err(), "append_kv odd ropn");
    assert!(akv(256, 98, 2).is_ok(), "append_kv even ropn must still be accepted");
    // POSITIVE but wrong. Every previous n_blocks case was rejected by an earlier arm
    // (kvl > 1024 fires before the equality check), so `n_blocks == kvl/128` could have
    // been deleted with the whole suite still green.
    assert!(akv(256, 8, 1).is_err(), "append_kv n_blocks too small");
    assert!(akv(256, 8, 3).is_err(), "append_kv n_blocks too large");

    // SAFETY: as above — a zero dimension is rejected before any pointer is used.
    unsafe {
        assert!(
            launch_embed_i8_row(p, p as *const f32, 0, 0, a as *mut f32).is_err(),
            "embed_i8_row hidden = 0"
        );
        assert!(launch_vadd(a as *mut f32, p as *const f32, 0).is_err(), "vadd n = 0");
        assert!(
            launch_argmax(p as *const f32, 0, a as *mut i32, b as *mut f32).is_err(),
            "argmax n = 0"
        );
        for (h, qh, ropn, why) in [(0usize, 8usize, 8usize, "h"), (8, 0, 8, "qh"), (8, 8, 0, "ropn")] {
            assert!(
                launch_gather_rope(p as *const f32, a as *mut f32, h, qh, 0, ropn).is_err(),
                "gather_rope {why} = 0"
            );
        }
    }
    // Rejecting a bad argument must not itself trip the layer.
    v.check("fwd_guards_reject_degenerate_arguments");
}

// ---------------------------------------------------------------------------
// linalg.hip glue: swiglu, rmsnorm, rope_interleave.
//
// Oracles derived from kernels/linalg.hip and src/math.rs — the SPECIFICATION — and
// from nothing else. See the header: a test written from the shader it checks agrees
// with the shader by construction.
// ---------------------------------------------------------------------------

/// `h[i] = silu(g[i])·u[i]`, in and out of place.
///
/// TOLERANCE, NOT BYTES, AND DELIBERATELY. linalg.hip writes `gv / (1 + expf(-gv))`;
/// `math.rs::silu` writes `x * (1 / (1 + (-x).exp()))`. Those are the same function and
/// different roundings — a divide versus a multiply by a reciprocal — and underneath
/// both sits `exp`, a library routine whose last bits are unspecified in HIP and in
/// Vulkan alike (the SPIR-V extended instruction set promises 3 ULP on `Exp`, not
/// correct rounding). There is no bit pattern to agree on here, so the byte-exact rule
/// in this file's header does not apply and `assert_close` is the honest comparison.
///
/// ALIASING IS THE POINT OF THE SECOND ARM. linalg.hip states `h` may alias `g`, and
/// promises it by having each thread read `g[i]` before writing `h[i]`. A shader that
/// widened the write, reordered it ahead of a neighbouring read, or staged through a
/// shared tile would still be perfectly correct with distinct buffers and wrong here.
/// Running only the distinct case would leave the documented in-place contract untested.
#[test]
fn swiglu_matches_silu_times_u() {
    let v = Validation::new();
    let mut r = Lcg(0x51D);
    // 1 (one live thread in one workgroup), 255/257 either side of the 256-thread
    // workgroup, then ragged multi-workgroup sizes. None is a multiple of 256.
    for n in [1usize, 255, 257, 1000, 4097] {
        let mut g: Vec<f32> = (0..n).map(|_| r.f() * 12.0).collect();
        // Both saturating ends, planted rather than hoped for: silu(-30) is ~-2.8e-12
        // (the sigmoid has underflowed), silu(30) is 30 to within f32, and silu(0) is
        // 0. -0.5 leads so that even n = 1 exercises a value the shader has to compute
        // rather than one any implementation returns by accident.
        let specials = [-0.5f32, 0.0, -30.0, 30.0];
        let k = specials.len().min(n);
        g[..k].copy_from_slice(&specials[..k]);
        let u: Vec<f32> = (0..n).map(|_| r.f() * 2.0).collect();
        let want: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();

        let ub = dev(&f32b(&u));
        for alias in [false, true] {
            let mut gb = poison(n * 4 + GUARD);
            gb.write_at(0, &f32b(&g)).expect("fill g");
            let mut hb = poison(n * 4 + GUARD);
            let (gp, hp) = if alias {
                // ONE buffer, handed in as both operands — the in-place contract.
                let p = gb.ptr_mut();
                (p as *const f32, p as *mut f32)
            } else {
                (gb.ptr() as *const f32, hb.ptr_mut() as *mut f32)
            };
            // SAFETY: `g`/`u`/`h` are live Buf device addresses holding n f32 plus a
            // guard band; aliasing g and h is explicitly permitted by the kernel. None
            // is dropped before the sync.
            unsafe {
                launch_swiglu(gp, ub.ptr() as *const f32, n, hp).expect("launch");
            }
            device_sync().expect("sync");
            let out = if alias {
                readb(&gb, n * 4 + GUARD)
            } else {
                readb(&hb, n * 4 + GUARD)
            };
            let label = format!("swiglu n={n} {}", if alias { "aliased" } else { "distinct" });
            assert_untouched(&out, n * 4, &label);
            assert_close(&want, &f32v(&out[..n * 4]), &label);
            // With distinct buffers `g` is an input and must come back untouched. BYTES
            // here, unlike the result: no arithmetic is supposed to have happened to it,
            // so any tolerance would be slack for a bug. This is what makes a shader
            // that writes through the wrong pointer fail — in the aliased arm that bug
            // is invisible, because the wrong pointer is the right one.
            if !alias {
                assert_bytes(&f32b(&g), &readb(&gb, n * 4), &format!("{label}: g unmodified"));
            }
        }
    }
    v.check("swiglu_matches_silu_times_u");
}

/// `y[i] = x[i]·(1/sqrt(mean(x²)+eps))·w[i]`, exactly as linalg.hip orders it:
/// `(x·inv)·w`, so the only thing that can differ from the kernel is `inv` itself.
///
/// The mean accumulates in f64 — the kernel sums a strided partial per thread and then
/// an LDS tree, an order no host loop reproduces, so a "reference" f32 sum would be
/// just another arbitrary order rather than the right answer. f64 is the answer both
/// are approximating.
fn rmsnorm_oracle(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let mean = (sum / x.len() as f64) as f32;
    let inv = 1.0f32 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * inv * b).collect()
}

/// RMSNorm against the host oracle, over sizes and over two regimes the reduction can
/// get wrong in opposite directions.
#[test]
fn rmsnorm_matches_the_host_oracle() {
    let v = Validation::new();
    let mut r = Lcg(0x8A5);
    let eps = 1e-6f32;

    let chk = |x: &[f32], w: &[f32], label: &str| {
        let n = x.len();
        let want = rmsnorm_oracle(x, w, eps);
        let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
        let mut yb = poison(n * 4 + GUARD);
        // SAFETY: `x` and `w` are n f32, `y` is n f32 plus the guard band; all three
        // live until the sync.
        unsafe {
            launch_rmsnorm(
                xb.ptr() as *const f32,
                wb.ptr() as *const f32,
                n,
                eps,
                yb.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        let got = readb(&yb, n * 4 + GUARD);
        assert_untouched(&got, n * 4, label);
        assert_close(&want, &f32v(&got[..n * 4]), label);
    };

    // 1 (one live thread; the other 255 tree slots hold the identity and must not
    // corrupt it), 255/256/257 around the workgroup, and 5000 — ~20 grid-stride passes
    // per thread, ragged, so the last pass is partial.
    for n in [1usize, 255, 256, 257, 5000] {
        let x: Vec<f32> = (0..n).map(|_| r.f()).collect();
        // A DISTINCT weight per element: a kernel reading `w[0]`, or dropping `w`
        // entirely, is a coincidence away from passing on a uniform weight vector.
        let w: Vec<f32> = (0..n).map(|_| r.f()).collect();
        chk(&x, &w, &format!("rmsnorm n={n}"));
    }

    // LARGE DYNAMIC RANGE. One element carries the entire sum of squares: 500² = 2.5e5
    // against ~1e-8 apiece from the other 999. Lose that one slot and the mean falls by
    // ~1e8, `inv` rises ~1.6e4×, and EVERY output is wrong by that factor — so a
    // dropped reduction slot is visible rather than buried under the tolerance.
    // 777 % 256 = 9, so it lands in thread 9's strided partial and then in tree slot 9,
    // neither of which is the identity-heavy edge of the reduction.
    let n = 1000usize;
    let mut x: Vec<f32> = (0..n).map(|_| r.f() * 1e-4).collect();
    x[777] = 500.0;
    let w: Vec<f32> = (0..n).map(|_| r.f()).collect();
    // Only the spike's own output is checked meaningfully here — `assert_close` scales
    // its tolerance to the largest wanted value, which is the spike's. That is fine:
    // this case exists to move `inv`, and every other size above checks the elementwise
    // scaling at a uniform magnitude.
    chk(&x, &w, "rmsnorm dynamic range");

    // EPS-DOMINATED. mean(x²) ~ 1e-13, three orders below eps, so `inv` is decided
    // almost entirely by eps. Drop the eps term and `inv` goes from ~1e3 to ~1e6 — the
    // one arrangement in which the `+ eps` could be deleted and still look right
    // everywhere else, since at unit scale it moves the answer by 5e-7 relative.
    let x: Vec<f32> = (0..500).map(|_| r.f() * 1e-6).collect();
    let w: Vec<f32> = (0..500).map(|_| r.f()).collect();
    chk(&x, &w, "rmsnorm eps-dominated");

    v.check("rmsnorm_matches_the_host_oracle");
}

/// RMSNorm reduces Σx² in LDS, so it gets the bit-reproducibility test every reduction
/// in this backend gets. Stricter than the oracle above: a reduce whose order varies
/// with workgroup scheduling clears a 1e-3 tolerance easily, and here it would move
/// `inv` and therefore EVERY element of the output, silently, run to run.
///
/// The input spans six decades on purpose. Summation order only matters when the
/// addends differ in magnitude — over a uniform [-1,1) the f32 sum is nearly
/// order-independent and a nondeterministic reduction would produce identical bytes
/// anyway, making the test green for the wrong reason.
#[test]
fn rmsnorm_is_bit_reproducible() {
    let v = Validation::new();
    let mut r = Lcg(0x2E5);
    let (n, eps) = (5003usize, 1e-6f32);
    let x: Vec<f32> = (0..n)
        .map(|i| r.f() * 10f32.powi(i as i32 % 7 - 3))
        .collect();
    let w: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    // Zero-filled, not poisoned: the all-zero guard at the end has to be able to fail.
    let mut yb = dev(&vec![0u8; n * 4]);

    // A decoy of a different shape, dispatched between repeats so the two runs do not
    // see identical queue state. 301 is under one workgroup, so it is a different
    // grid-stride shape as well as a different size.
    let (dx, dw) = (dev(&f32b(&x[..301])), dev(&f32b(&w[..301])));
    let mut dy = dev(&vec![0u8; 301 * 4]);

    let run = |x: &Buf, w: &Buf, y: &mut Buf, n: usize| {
        // SAFETY: sizes as allocated above; every Buf outlives the sync.
        unsafe {
            launch_rmsnorm(
                x.ptr() as *const f32,
                w.ptr() as *const f32,
                n,
                eps,
                y.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        readb(y, n * 4)
    };

    let first = run(&xb, &wb, &mut yb, n);
    for i in 1..5 {
        run(&dx, &dw, &mut dy, 301);
        assert_bytes(&first, &run(&xb, &wb, &mut yb, n), &format!("rmsnorm repeat {i}"));
    }
    // Guard against comparing two buffers of zeros and calling it determinism.
    assert!(
        f32v(&first).iter().any(|v| v.abs() > 1e-6),
        "output is all zero — the test proves nothing"
    );
    v.check("rmsnorm_is_bit_reproducible");
}

/// One row of `rope_interleave`, transcribed from linalg.hip: the angle in f64, `cos`
/// and `sin` rounded to f32, and the rotation itself in f32.
///
/// EVERY READ COMES FROM `v` AND EVERY WRITE GOES TO `out`. That is the contract the
/// HIP spends a `__syncthreads()` on — the destination halves `(j, half+j)` overlap the
/// source pairs `(2j, 2j+1)`, so an implementation that writes as it goes shreds its own
/// input. An oracle that updated in place would reproduce that bug instead of catching
/// it.
fn rope_row(v: &[f32], seg: usize, pos: usize, theta: f64) -> Vec<f32> {
    let half = seg / 2;
    let mut out = v[..seg].to_vec();
    for j in 0..half {
        let (a, b) = (v[2 * j], v[2 * j + 1]);
        let inv = theta.powf(-2.0 * j as f64 / seg as f64);
        let ang = pos as f64 * inv;
        let (cs, sn) = (ang.cos() as f32, ang.sin() as f32);
        out[j] = a * cs - b * sn;
        out[half + j] = b * cs + a * sn;
    }
    out
}

/// Interleaved RoPE over `count` rows of `stride`, rotating each row's first `seg`.
///
/// WHAT THIS DOES AND DOES NOT PROVE. The trig table is a f64 `powf`/`cos`/`sin` chain,
/// and both this oracle and the launcher evaluate it with the same Rust libm — so the
/// ANGLES are a consistency check, not an independent one, and a misread exponent in
/// the HIP would be reproduced identically on both sides. What is independently checked
/// is everything downstream of the table: the interleave-to-halves permutation, the
/// rotation's sign convention, the per-row `stride` addressing, and the read-before-
/// write discipline. The `pos = 0` shape below is the strongest of these precisely
/// because it removes the trig from the picture entirely.
///
/// TOLERANCE for pos > 0: the rotation is `a·cs - b·sn` in f32, which SPIR-V may
/// contract into an FMA unless the shader forbids it. That is a legitimate one-ULP
/// difference from the oracle's two roundings, so bytes are not available.
#[test]
fn rope_interleave_rotates_each_row() {
    let v = Validation::new();
    let mut r = Lcg(0x60E);
    // (count, stride, seg, pos, theta):
    //   3/40/32/0      pos = 0 -> a pure permutation, checked as BYTES (see below).
    //   4/100/34/4096  half = 17: ragged, a partial subgroup, and every lane past 17
    //                  idle. pos = 4096 is where the angle actually has to be right —
    //                  the relative error in theta^(-2j/seg) is multiplied by pos.
    //   2/2100/2048    seg/2 = 1024, the documented cap, and four times the workgroup
    //                  so the in-block loop over `half` runs several iterations.
    //   1/16/16        count = 1, stride == seg: no in-row tail, grid of one.
    for (count, stride, seg, pos, theta) in [
        (3usize, 40usize, 32usize, 0usize, 10000.0f64),
        (4, 100, 34, 4096, 10000.0),
        (2, 2100, 2048, 7, 10000.0),
        (1, 16, 16, 1, 500000.0),
    ] {
        // Distinct data per row, so a kernel that ignored blockIdx and rotated row 0
        // `count` times fails rather than coincides.
        let rows: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..seg).map(|_| r.f()).collect())
            .collect();
        let bytes = count * stride * 4;
        let mut bb = poison(bytes + GUARD);
        for (s, row) in rows.iter().enumerate() {
            bb.write_at(s * stride * 4, &f32b(row)).expect("fill row");
        }
        // SAFETY: `base` is a live Buf device address holding count*stride f32 plus the
        // guard band; it outlives the sync.
        unsafe {
            launch_rope(bb.ptr_mut() as *mut f32, count, stride, seg, pos, theta)
                .expect("launch");
        }
        device_sync().expect("sync");
        let got = readb(&bb, bytes + GUARD);
        let label = format!("rope c{count} st{stride} seg{seg} pos{pos}");
        assert_untouched(&got, bytes, &label);

        for (s, row) in rows.iter().enumerate() {
            let base = s * stride * 4;
            // The row's tail past `seg` is a guard region of its own — the kernel is
            // documented to touch only the first `seg` elements of each row, and a
            // stride/seg mix-up (or a `half` computed from stride) lands here, INSIDE
            // the allocation where the trailing guard band cannot see it.
            assert!(
                got[base + seg * 4..base + stride * 4]
                    .iter()
                    .all(|&b| b == 0xFF),
                "{label} row {s}: wrote past element {seg} of the row"
            );
            let want = rope_row(row, seg, pos, theta);
            let g = &got[base..base + seg * 4];
            let rl = format!("{label} row {s}");
            if pos == 0 {
                // BYTES. At pos = 0 every angle is 0, so cs = 1.0 and sn = 0.0 exactly
                // and the kernel reduces to `v[j] = a`, `v[half+j] = b` — a pure
                // de-interleave in which no value is rounded at all, contraction or not.
                // The header's precondition for a byte-exact compare is therefore met
                // trivially: the inputs are the outputs. `Lcg::f` cannot return exactly
                // zero (it would need (u>>32) = 2147483647.5), so the one ambiguity
                // that could arise — a signed zero out of `a - b*0.0`, which Vulkan does
                // not promise to preserve — is out of reach of this data.
                assert_bytes(&f32b(&want), g, &rl);
            } else {
                assert_close(&want, &f32v(g), &rl);
            }
        }
    }
    v.check("rope_interleave_rotates_each_row");
}

/// The linalg launchers' argument guards report errors rather than dispatching a
/// degenerate grid or tripping a VUID.
///
/// Each rope guard gets a POSITIVE control beside its negative, because every one of
/// them could be deleted and replaced by something stricter with the negatives still
/// green — `stride >= seg` unconditionally would reject the legal single-row case, and
/// `seg <= 32` would reject everything this kernel is for.
#[test]
fn linalg_guards_reject_degenerate_arguments() {
    let v = Validation::new();
    let src = dev(&[0u8; 4096]);
    let mut dst = dev(&[0u8; 4096]);
    let (p, q) = (src.ptr(), dst.ptr_mut());

    let rope = |count: usize, stride: usize, seg: usize| {
        // SAFETY: `dst` is a live 4096-byte Buf and outlives the sync below; the cases
        // that are ACCEPTED here write at most 32 f32 into it. The rejected ones never
        // reach a pointer.
        unsafe { launch_rope(q as *mut f32, count, stride, seg, 3, 10000.0) }
    };

    // SAFETY: n = 0 is rejected before any pointer is used.
    unsafe {
        assert!(
            launch_swiglu(p as *const f32, p as *const f32, 0, q as *mut f32).is_err(),
            "swiglu n = 0"
        );
        assert!(
            launch_rmsnorm(p as *const f32, p as *const f32, 0, 1e-6, q as *mut f32).is_err(),
            "rmsnorm n = 0"
        );
    }

    assert!(rope(0, 16, 16).is_err(), "rope count = 0");
    assert!(rope(1, 16, 0).is_err(), "rope seg = 0");
    // Odd seg: `half = seg/2` would truncate and the last element would never be read
    // or written, so the guard is the only thing standing between that and silence.
    assert!(rope(1, 16, 15).is_err(), "rope odd seg");
    assert!(rope(1, 16, 16).is_ok(), "rope even seg must still be accepted");
    // seg/2 > 1024 — one past the cap. The ACCEPTED boundary, seg = 2048, is exercised
    // for real in `rope_interleave_rotates_each_row` rather than here, where the 4096
    // byte buffer could not hold its output.
    assert!(rope(1, 4096, 2050).is_err(), "rope seg/2 > 1024");
    // count > 1 with stride < seg: the rows would overlap and each block would rotate
    // bytes another block is mid-rotation on.
    assert!(rope(2, 8, 16).is_err(), "rope count > 1 with stride < seg");
    assert!(rope(1, 8, 16).is_ok(), "rope count = 1 ignores stride");
    assert!(rope(2, 16, 16).is_ok(), "rope stride == seg must be accepted");

    // Retire the accepted dispatches while their buffer is still alive. The guard test
    // is about the rejections, but the controls are real launches.
    device_sync().expect("sync");
    // Rejecting a bad argument must not itself trip the layer — a guard that returns
    // Err while leaving a half-built Vulkan object behind would show up here.
    v.check("linalg_guards_reject_degenerate_arguments");
}


// ---------------------------------------------------------------------------
// linalg.hip: the quantised GEMVs
// ---------------------------------------------------------------------------

/// `y[o] = scale[o] * Σ x[i]·(i8)packed[o·i_dim+i]` — lm_head to logits.
///
/// Like `embed_i8_row`, the shader reads packed int8 as u32 WORDS and sign-extends by
/// hand, so the failure that matters is a byte read as unsigned: invisible below 0x80,
/// a 256·scale error above it. Every row therefore carries the four extremes, and
/// `i_dim` is not a multiple of 4 so the extraction runs at every byte phase.
#[test]
fn gemv_i8_matches_the_host_oracle() {
    let v = Validation::new();
    let mut r = Lcg(0x18);
    // o_dim not a multiple of ROWS_PER_BLOCK (tail workgroup has idle subgroups);
    // i_dim not a multiple of 4 or 32 (byte phases, and a ragged wave stride).
    for (o_dim, i_dim) in [(37usize, 67usize), (256, 1024)] {
        let mut packed: Vec<u8> = (0..o_dim * i_dim)
            .map(|i| (i as u8).wrapping_mul(29).wrapping_add(7))
            .collect();
        for o in 0..o_dim {
            packed[o * i_dim..o * i_dim + 4].copy_from_slice(&[0x80, 0xFF, 0x7F, 0x00]);
        }
        let scale: Vec<f32> = (0..o_dim).map(|o| 0.011 * (o as f32 + 1.0)).collect();
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
        let mut want = vec![0.0f32; o_dim];
        matvec_i8(&mut want, &x, &packed, &scale, o_dim, i_dim);

        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = poison(o_dim * 4 + GUARD);
        // SAFETY: live Buf device addresses of the documented sizes; nothing dropped
        // before the sync.
        unsafe {
            launch_gemv_i8(
                xb.ptr() as *const f32,
                pb.ptr(),
                sb.ptr() as *const f32,
                o_dim,
                i_dim,
                yb.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        let got = readb(&yb, o_dim * 4 + GUARD);
        let label = format!("gemv_i8 {o_dim}x{i_dim}");
        assert_untouched(&got, o_dim * 4, &label);
        assert_close(&want, &f32v(&got[..o_dim * 4]), &label);
    }
    v.check("gemv_i8_matches_the_host_oracle");
}

/// `gemv_fp8` against `quant.rs::matvec_fp8`, in BOTH dispatch geometries.
///
/// The launcher picks split-K at `i_dim >= 4096`, so the two shapes below straddle that
/// threshold DELIBERATELY and each is reported separately. Testing only one side leaves
/// an entire geometry — a different thread-to-row mapping and a whole LDS combine —
/// unexecuted while the suite goes green, which is the same trap the 1024/300 append_kv
/// shape had. Combined shapes are efficient right up to the moment one goes red.
///
/// The block tile must be a power of two: `blk_shift` indexes the scale by a SHIFT since
/// the HIP side replaced its signed divide, and the launcher mirrors that rejection.
#[test]
fn gemv_fp8_matches_the_host_oracle() {
    let v = Validation::new();
    let block = 128usize;
    for (o_dim, i_dim, geometry) in [
        (256usize, 512usize, "wave-per-row"),
        (128, 4096, "split-K"),
    ] {
        let mut r = Lcg(0xF8 ^ i_dim as u64);
        let packed: Vec<u8> = (0..o_dim * i_dim).map(|_| f32_to_e4m3(r.f())).collect();
        let sc_cols = i_dim / block;
        let scale: Vec<f32> = (0..o_dim.div_ceil(block) * sc_cols)
            .map(|_| (r.f() * 0.1).abs() + 0.01)
            .collect();
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
        let mut want = vec![0.0f32; o_dim];
        matvec_fp8(&mut want, &x, &packed, &scale, i_dim, block);

        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = poison(o_dim * 4 + GUARD);
        // SAFETY: as above; `scale` is ⌈o_dim/block⌉·⌈i_dim/block⌉ f32 as documented.
        unsafe {
            launch_gemv_fp8(
                xb.ptr() as *const f32,
                pb.ptr(),
                sb.ptr() as *const f32,
                o_dim,
                i_dim,
                block,
                yb.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        let got = readb(&yb, o_dim * 4 + GUARD);
        let label = format!("gemv_fp8 {geometry} {o_dim}x{i_dim}");
        assert_untouched(&got, o_dim * 4, &label);
        assert_close(&want, &f32v(&got[..o_dim * 4]), &label);
    }
    v.check("gemv_fp8_matches_the_host_oracle");
}

/// The split-K path reduces across workgroup partials in a fixed-order LDS combine, so
/// it gets the reproducibility test every reduction here gets — bytes, repeats, a decoy
/// between them, and a not-all-zero guard.
#[test]
fn gemv_fp8_splitk_is_bit_reproducible() {
    let v = Validation::new();
    let (o_dim, i_dim, block) = (64usize, 4096usize, 128usize);
    let mut r = Lcg(0x5B7);
    let packed: Vec<u8> = (0..o_dim * i_dim).map(|_| f32_to_e4m3(r.f())).collect();
    let scale: Vec<f32> = (0..o_dim.div_ceil(block) * (i_dim / block))
        .map(|_| (r.f() * 0.1).abs() + 0.01)
        .collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));
    // A differently-shaped decoy, below the split-K threshold so it also perturbs which
    // pipeline geometry ran last.
    let (dx, dp, ds) = (
        dev(&f32b(&x[..512])),
        dev(&packed[..32 * 512]),
        dev(&f32b(&scale[..4])),
    );
    let mut dy = dev(&f32b(&[0.0f32; 32]));

    let run = |xb: &Buf, pb: &Buf, sb: &Buf, yb: &mut Buf, o: usize, i: usize| -> Vec<u8> {
        // SAFETY: live device addresses of the documented sizes.
        unsafe {
            launch_gemv_fp8(
                xb.ptr() as *const f32,
                pb.ptr(),
                sb.ptr() as *const f32,
                o,
                i,
                block,
                yb.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        readb(yb, o * 4)
    };

    let first = run(&xb, &pb, &sb, &mut yb, o_dim, i_dim);
    for k in 1..5 {
        run(&dx, &dp, &ds, &mut dy, 32, 512);
        let again = run(&xb, &pb, &sb, &mut yb, o_dim, i_dim);
        assert_eq!(first, again, "gemv_fp8 split-K not bit-reproducible on repeat {k}");
    }
    assert!(
        f32v(&first).iter().any(|v| v.abs() > 1e-6),
        "output is all zero — the test proves nothing"
    );
    v.check("gemv_fp8_splitk_is_bit_reproducible");
}

/// Twelve `(pos, j)` points of the RoPE trig table, against values computed OUTSIDE this
/// codebase's arithmetic.
///
/// WHY THIS EXISTS. `launch_rope` computes the table host-side in f64 — which is what
/// makes it bit-identical to HIP — and `rope_interleave_rotates_each_row`'s oracle
/// computes it the same way, with the same Rust libm. So that test is a CONSISTENCY
/// check on the table: a misread exponent would reproduce identically on both sides and
/// pass. The fix that bought bit-identity also removed the table from independent view,
/// and both halves are consequences of the same change.
///
/// These literals break the coupling. They were computed in Python with `decimal` at 60
/// digits: `inv = exp(-2j/seg · ln θ)` via Decimal's own ln/exp, exact range reduction
/// mod 2π against a 60-digit constant, then sin/cos by Taylor series — no `powf`, no
/// libm, nothing shared with the launcher.
///
/// SCOPE, stated so the coverage claim stays honest: THESE TWELVE POINTS are
/// independently verified. The rest of the table remains consistency-checked.
///
/// Extraction trick: with a row of pairs all set to (1, 0), the rotation reduces to
/// `v[j] = cos` and `v[half+j] = sin`, so the table is read straight out of the output.
#[test]
fn rope_trig_table_matches_independent_reference() {
    let v = Validation::new();
    // (pos, j, seg, cos bits, sin bits) — see the docstring for provenance.
    const REF: &[(usize, usize, usize, u32, u32)] = &[
        (0, 0, 64, 0x3f800000, 0x00000000),
        (0, 7, 64, 0x3f800000, 0x00000000),
        (1, 0, 64, 0x3f0a5140, 0x3f576aa4),
        (1, 1, 64, 0x3f3b54b0, 0x3f2e7ace),
        (4096, 0, 64, 0x3f4dd254, 0xbf183a75),
        (4096, 3, 64, 0x3f5241cc, 0xbf120a82),
        (4096, 31, 64, 0x3f5ac075, 0x3f04fadb),
        (137, 5, 64, 0x3ef4f90e, 0x3f60cbae),
        (100000, 0, 64, 0xbf7fd61c, 0x3d126d55),
        (100000, 17, 64, 0xbf15a6fd, 0x3f4fb3c1),
        (7, 2, 128, 0x3f02ee57, 0xbf5bfbf8),
        (65535, 63, 128, 0x3e908076, 0x3f7597c0),
    ];
    let theta = 10000.0f64;
    for &(pos, j, seg, want_cos, want_sin) in REF {
        let half = seg / 2;
        // All pairs (1, 0) so the rotation returns the table itself.
        let mut row = vec![0.0f32; seg];
        for k in 0..half {
            row[2 * k] = 1.0;
            row[2 * k + 1] = 0.0;
        }
        let mut b = dev(&f32b(&row));
        // SAFETY: one row of `seg` f32, live until the sync.
        unsafe { launch_rope(b.ptr_mut() as *mut f32, 1, seg, seg, pos, theta).expect("launch") };
        device_sync().expect("sync");
        let got = f32v(&read(&b, seg));
        let (gc, gs) = (got[j].to_bits(), got[half + j].to_bits());
        // Within 1 ULP, not bit-equal: the reference is a DIFFERENT algorithm at higher
        // precision, so the two f64 intermediates can straddle an f32 rounding boundary.
        // Exact agreement is expected and 1 ULP is the honest bound to assert.
        let ulp = |a: u32, b: u32| (a as i64 - b as i64).abs();
        assert!(
            ulp(gc, want_cos) <= 1,
            "rope cos(pos={pos}, j={j}, seg={seg}): got {gc:#010x}, independent reference              {want_cos:#010x} — the host-side trig table disagrees with a computation              that shares no arithmetic with it"
        );
        assert!(
            ulp(gs, want_sin) <= 1,
            "rope sin(pos={pos}, j={j}, seg={seg}): got {gs:#010x}, independent reference              {want_sin:#010x}"
        );
    }
    v.check("rope_trig_table_matches_independent_reference");
}
