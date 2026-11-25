//! inkling's `vaxpy` (`x[i] += alpha·y[i]`) on the device, scored against the CPU oracle
//! `rivoli_oracles::inkling::vaxpy` — **no C twin** (owner direction 2026-11-24): oracle-validated
//! against the first-party reference rather than the C→Rust bit-identity harness. The update is
//! one multiply-add per element and the device may contract it to a single FMA, so the comparison
//! is RELATIVE at `1e-5` (the clip family's convention for an FMA-contracting expression) rather
//! than bitwise.
//!
//! `ORACLE_VALIDATED` in `kernel_coverage.rs` waives the C-twin row; this suite is the oracle it
//! cites.
//!
//! Run on the device under `flock /var/run/sys-gpu.lock`, single-threaded (`--test-threads=1`).
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::launch_vaxpy;
use rivoli_oracles::inkling::vaxpy as cpu_vaxpy;

mod common;
use common::*;

/// One extent's device launch, returning the launcher's `Result` so the guard arms enter through
/// the same function as the arithmetic arms. `x` is uploaded and updated in place; `n` comes from
/// its length, which is what the launcher is handed.
fn vaxpy_launch(x: &[f32], y: &[f32], alpha: f32) -> anyhow::Result<Vec<f32>> {
    let n = x.len();
    let mut xb = dev(&f32b(x));
    let yb = dev(&f32b(y));
    let s = stream();
    // SAFETY: `x`/`y` are `n` f32 each and live until the stream `back` syncs on; the launcher
    // guards the extent and `alpha` before any device access.
    unsafe {
        launch_vaxpy(
            xb.ptr_mut() as *mut f32,
            yb.ptr() as *const f32,
            alpha,
            n,
            s.raw(),
        )?;
    }
    Ok(f32v(&back(&xb)))
}

/// The update matches the CPU oracle over ragged extents — including `2304`, nine blocks of
/// `ELEMENTWISE_BLOCK`, so the grid is genuinely multi-block — with positive, negative, and zero
/// scales.
#[test]
fn vaxpy_matches_the_cpu_oracle() {
    for &n in &[1usize, 33, 128, 257, 2304] {
        let x = fill(n, 0x11 + n as u64, 2.0);
        let y = fill(n, 0x77 + n as u64, 2.0);
        for alpha in [1.5f32, -2.5, 0.0] {
            let want = cpu_vaxpy(&x, &y, alpha);
            let got = ok(vaxpy_launch(&x, &y, alpha), "gpu vaxpy");
            let label = format!("vaxpy n={n} alpha={alpha}");
            assert_rel(&want, &got, &label, 1e-5);
        }
    }
}

/// `n > 0` (1001) and `alpha` finite (1006) are guarded before launch; a zero or negative scale is
/// arithmetic, not a refusal.
#[test]
fn vaxpy_geometry_is_guarded() {
    let x = fill(4, 0x21, 2.0);
    let y = fill(4, 0x22, 2.0);
    assert_guard(vaxpy_launch(&[], &[], 1.0), Some(1001), "n = 0");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_guard(
            vaxpy_launch(&x, &y, bad),
            Some(1006),
            &format!("alpha = {bad}"),
        );
    }
    assert_guard(vaxpy_launch(&x, &y, 0.0), None, "alpha = 0.0");
    assert_guard(vaxpy_launch(&x, &y, -2.5), None, "alpha = -2.5");
}
