//! Vulkan kernels vs their CPU oracles — the `tests/kernel.rs` story for the second
//! backend. Compiles to nothing without `vulkan`.
//!
//! Separate file because `tests/kernel.rs` is `#![cfg(feature = "rocm")]` end to end.
//! The helpers below are deliberately the same ones it uses, so the kernel-porting
//! phase can hoist both files onto a shared module instead of rewriting either.
#![cfg(feature = "vulkan")]
#![allow(clippy::expect_used)]

use rivoli::vk::{
    Buf, ROWS_PER_BLOCK, VALIDATION_ERRORS, device_sync, gpu, launch_gemv_f32, memcpy_dtod,
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
    let mut out = Vec::new();
    y.read_into(&mut out, n * 4).expect("out");
    out
}

/// Fail the run if the validation layer said anything. Costs nothing when the layer
/// is absent, and prints whether it was there at all — a green oracle from an
/// unvalidated run must not read like a validated one.
fn assert_validation_clean(label: &str) {
    let g = gpu().expect("vulkan init");
    let n = VALIDATION_ERRORS.load(Ordering::Relaxed);
    println!(
        "{label}: validation layer {}",
        if g.validation() {
            "ON"
        } else {
            "OFF — THIS RUN IS UNVALIDATED"
        }
    );
    assert_eq!(n, 0, "{label}: {n} validation messages");
}

#[test]
fn gemv_f32_matches_oracle() {
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
    assert_validation_clean("gemv_f32_matches_oracle");
}

/// The inter-dispatch barrier in `Gpu::enqueue` is the most consequential unvalidated
/// primitive in the backend, and until this test existed NOTHING covered it: every
/// other test records one dispatch and then syncs, so deleting the barrier entirely
/// would not have failed anything.
///
/// Two chained dispatches in ONE command buffer with ONE sync — the second consumes
/// the first's output as its input — so the result is only correct if the barrier
/// actually orders the write before the read.
#[test]
fn chained_dispatch_respects_the_barrier() {
    let mut r = Lcg(0xBA2);
    let (n, mid) = (64usize, 96usize);
    let a: Vec<f32> = (0..mid * n).map(|_| r.f()).collect(); // [mid, n]
    let b: Vec<f32> = (0..n * mid).map(|_| r.f()).collect(); // [n, mid]
    let x: Vec<f32> = (0..n).map(|_| r.f()).collect();

    let mut t = vec![0.0f32; mid];
    matvec_f32(&mut t, &x, &a, n);
    let mut want = vec![0.0f32; n];
    matvec_f32(&mut want, &t, &b, mid);

    let (xb, ab, bb) = (dev(&f32b(&x)), dev(&f32b(&a)), dev(&f32b(&b)));
    let mut tb = dev(&f32b(&vec![0.0f32; mid]));
    let mut yb = dev(&f32b(&vec![0.0f32; n]));

    launch(&xb, &ab, &mut tb, mid, n); // t = A·x
    launch(&tb, &bb, &mut yb, n, mid); // y = B·t   <- needs t complete
    device_sync().expect("sync"); // ONE sync for both
    assert_close(&want, &f32v(&read(&yb, n)), "chained A then B");
    assert_validation_clean("chained_dispatch");
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
    assert_validation_clean("gemv_f32_is_bit_reproducible");
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
    assert_validation_clean("memcpy_dtod_after_dispatch_is_ordered");
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
    assert_validation_clean("timeline_signal_resolves_and_latency");
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

/// The guards report errors rather than tripping a Vulkan VUID or allocating nothing.
#[test]
fn guards_reject_degenerate_arguments() {
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
}
