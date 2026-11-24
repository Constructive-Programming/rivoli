//! inkling's per-head L2 clip on the device, scored against the CPU oracle
//! `rivoli_oracles::inkling::l2_clip` — **no C twin** (owner direction 2026-11-24): this is a
//! new-model kernel validated against the first-party reference, not the C→Rust bit-identity
//! harness. The kernel reduces (an LDS tree), so the oracle comparison is RELATIVE, not bitwise:
//! the tree's accumulation order differs from the oracle's naive sum at fp32 rounding.
//!
//! **Tolerance provenance.** The anchor carves the fp32 floor per operator
//! (`docs/measurement/inkling2-reference/anchor.md` §Tolerances: 0 for both `L2Clip` forms, whose
//! anchor inputs leave every head below threshold, so the anchor constrains only the identity
//! path). These cases draw past the threshold so the scale is exercised, and `1e-5` relative is
//! the allowance for the LDS tree re-associating the oracle's naive fp32 sum — a 2^-22-scale
//! difference, not a fitted number.
//!
//! The census waiver is `kernel_coverage.rs::ORACLE_VALIDATED`; this suite is the oracle it names.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::{launch_l2_clip, launch_log_scaled_l2_clip, launch_scaled_l2_clip};
use rivoli_oracles::inkling::{
    l2_clip as cpu_l2_clip, log_scaled_l2_clip as cpu_log_scaled_l2_clip,
    scaled_l2_clip as cpu_scaled_l2_clip,
};

mod common;
use common::{
    assert_bits, assert_guard, assert_rel, back, dev, f32b, f32v, fill, ok, stream, zeros,
};

/// One BATCHED clip case: `x` (and the learned `scale`) each `rows·n` f32. Named fields because a
/// `rows`/`n` transposition would be a plausible wrong answer rather than a compile error; `tau`
/// selects the full-layer form (`Some`) from the sliding one (`None`).
#[derive(Clone, Copy)]
struct ClipCase<'a> {
    x: &'a [f32],
    scale: Option<&'a [f32]>,
    rows: usize,
    n: usize,
    threshold: f32,
    tau: Option<f32>,
}

impl<'a> ClipCase<'a> {
    /// The plain `l2_clip` geometry (no learned scale, no `tau`).
    fn plain(x: &'a [f32], geom: (usize, usize), threshold: f32) -> Self {
        let (rows, n) = geom;
        Self {
            x,
            scale: None,
            rows,
            n,
            threshold,
            tau: None,
        }
    }

    /// The scaled-layer geometry; the full-layer form is this with `tau: Some(…)` by struct
    /// update, so the two cannot drift in their operands.
    fn scaled(x: &'a [f32], scale: &'a [f32], geom: (usize, usize), threshold: f32) -> Self {
        Self {
            scale: Some(scale),
            ..Self::plain(x, geom, threshold)
        }
    }
}

/// One launch through the launcher callers use, returning its own `Result`. `alias` points `y` at
/// `x` — the contract each launcher documents (the reduction barrier separates every read of `x`
/// from every write of `y`); the result is read back from whichever buffer was written.
fn clip_launch(c: &ClipCase, alias: bool) -> anyhow::Result<Vec<f32>> {
    let s = stream();
    let mut xb = dev(&f32b(c.x));
    let mut yb = zeros(c.x.len().max(1) * 4);
    let sb = c.scale.map(|sc| dev(&f32b(sc)));
    let sp = || {
        sb.as_ref()
            .map_or(std::ptr::null(), |b| b.ptr() as *const f32)
    };
    let xp = xb.ptr() as *const f32;
    let y = if alias {
        xb.ptr_mut() as *mut f32
    } else {
        yb.ptr_mut() as *mut f32
    };
    let st = s.raw();
    // SAFETY: `x`, `y` (and `scale` when present) are each `rows·n` live f32, outliving the
    // stream `back` syncs on. `alias` makes them the SAME buffer, which the launchers document.
    let launched = match (c.scale, c.tau) {
        (None, _) => unsafe { launch_l2_clip(xp, c.rows, c.n, c.threshold, y, st) },
        (Some(_), None) => unsafe {
            launch_scaled_l2_clip(xp, sp(), c.rows, c.n, c.threshold, y, st)
        },
        (Some(_), Some(t)) => unsafe {
            launch_log_scaled_l2_clip(xp, sp(), c.rows, c.n, c.threshold, t, y, st)
        },
    };
    launched?;
    Ok(f32v(&back(if alias { &xb } else { &yb })))
}

/// The launch's `Result` alone, for the guard table (which never reads the destination back).
fn clip_guard(c: &ClipCase) -> anyhow::Result<()> {
    clip_launch(c, false).map(|_| ())
}

/// The per-row oracle of a batched case, so a row transposition cannot pass by comparing row 0.
fn per_row(want: impl Fn(&[f32]) -> Vec<f32>, c: &ClipCase) -> Vec<f32> {
    (0..c.rows)
        .flat_map(|r| want(&c.x[r * c.n..(r + 1) * c.n]))
        .collect()
}

/// A learned scale: a signed draw's magnitude plus 0.5, so every element is live and non-zero.
fn positive_scale(n: usize, salt: u64) -> Vec<f32> {
    fill(n, salt, 0.5).iter().map(|v| v.abs() + 0.5).collect()
}

/// The kernel reproduces the CPU oracle at ragged and degenerate widths (a block edge mid-wave).
#[test]
fn l2_clip_matches_the_cpu_oracle() {
    for &n in &[1usize, 33, 128, 257, 1024] {
        let x = fill(n, 0x1234 + n as u64, 4.0);
        let c = ClipCase::plain(&x, (1, n), 8.0);
        let got = ok(clip_launch(&c, false), "l2_clip");
        assert_rel(&cpu_l2_clip(&x, 8.0), &got, &format!("l2_clip n={n}"), 1e-5);
    }
}

/// BATCHED: one launch over many rows, each row clipped by its own norm — the whole point of the
/// `rows` argument. A kernel that reduced across rows, or fanned row 0 into every block, passes
/// the single-row case and fails here.
#[test]
fn l2_clip_batches_each_row_independently() {
    for &(rows, n) in &[(24usize, 128usize), (8, 16)] {
        let x = fill(rows * n, 0x5678 + n as u64, 4.0);
        let c = ClipCase::plain(&x, (rows, n), 2.0);
        let got = ok(clip_launch(&c, false), "l2_clip batched");
        let want = per_row(|row| cpu_l2_clip(row, 2.0), &c);
        assert_rel(&want, &got, &format!("l2_clip rows={rows} n={n}"), 1e-5);
    }
}

/// A long vector is scaled to norm == threshold; a short one is unchanged (the clamp at 1, T4).
#[test]
fn l2_clip_scales_long_vectors_and_leaves_short_ones() {
    let long = fill(128, 7, 4.0);
    let got = ok(
        clip_launch(&ClipCase::plain(&long, (1, 128), 2.0), false),
        "l2_clip long",
    );
    let norm = got.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 2.0).abs() < 1e-3, "long norm {norm} != 2.0");
    let short = vec![0.1f32; 128]; // norm ~1.13 < 2
    let got = ok(
        clip_launch(&ClipCase::plain(&short, (1, 128), 2.0), false),
        "l2_clip short",
    );
    assert_rel(&short, &got, "l2_clip short unchanged", 1e-6);
}

/// The zero vector is the T4 edge: `threshold/(0 + 1e-6)` is huge, so the clamp at 1 keeps the
/// zero vector zero instead of producing a 0·inf NaN.
#[test]
fn l2_clip_zero_vector_stays_zero() {
    for &n in &[1usize, 128, 257] {
        let x = vec![0.0f32; n];
        let zero = vec![0.0f32; n];
        let scale = vec![1.0f32; n];
        let plain = ClipCase::plain(&x, (1, n), 8.0);
        assert_bits(
            &zero,
            &ok(clip_launch(&plain, false), "zero plain"),
            "plain",
        );
        let scaled = ClipCase::scaled(&x, &scale, (1, n), 8.0);
        assert_bits(
            &zero,
            &ok(clip_launch(&scaled, false), "zero scaled"),
            "scaled",
        );
    }
}

/// `y` may alias `x` for ALL THREE forms, and the layer loop is invited to rely on it —
/// bit-identical to the separate-buffer launch.
#[test]
fn the_clip_family_output_may_alias_its_input() {
    let (rows, n) = (24usize, 128usize);
    let x = fill(rows * n, 99, 4.0);
    let scale = positive_scale(rows * n, 0x9abc);
    let scaled = ClipCase::scaled(&x, &scale, (rows, n), 8.0);
    let cases = [
        ClipCase::plain(&x, (rows, n), 8.0),
        scaled,
        ClipCase {
            tau: Some(1.5),
            ..scaled
        },
    ];
    for c in cases {
        let separate = ok(clip_launch(&c, false), "clip separate");
        let aliased = ok(clip_launch(&c, true), "clip alias");
        assert_bits(&separate, &aliased, &format!("clip alias tau={:?}", c.tau));
    }
}

/// `y[i] = x[i]·min(threshold/(‖x‖+1e-6), 1)` — the scaled and full-layer clips reproduce the
/// CPU oracle (scale before the reduction, tau after the clip) at ragged widths.
#[test]
fn scaled_and_log_scaled_clip_match_the_cpu_oracle() {
    for &n in &[1usize, 33, 128, 257] {
        let x = fill(n, 0x5678 + n as u64, 4.0);
        let scale = positive_scale(n, 0x9abc);
        // A NON-UNIT tau at every width: the LogScale factor is applied after the clip, and a
        // `tau = 1.0` arm would let a kernel that dropped the multiply pass (log_scaled_clip's
        // dedicated test is the sharper witness, but this loop must not be vacuous either).
        let tau = 1.0 + 0.001 * n as f32;
        let scaled = ClipCase::scaled(&x, &scale, (1, n), 8.0);
        for t in [None, Some(tau)] {
            let c = ClipCase { tau: t, ..scaled };
            let got = ok(clip_launch(&c, false), "scaled/log_scaled");
            let want = match t {
                None => cpu_scaled_l2_clip(&x, &scale, 8.0),
                Some(tt) => cpu_log_scaled_l2_clip(&x, &scale, 8.0, tt),
            };
            assert_rel(&want, &got, &format!("clip n={n} tau={t:?}"), 1e-5);
        }
    }
}

/// **The `tau` field is load-bearing.** At `pos = 16383` with the shipped `alpha = 0.1` and
/// `n_floor = 8192` the LogScale is `1 + 0.1·ln(2) ≈ 1.0693`, and 1.5 is an unmissable second
/// arm; each output must be exactly `tau` times the tau-free launch's. A kernel that dropped
/// `a.tau` (the defect this test exists for) would return the ratio 1.0 on both arms.
#[test]
fn log_scaled_clip_applies_a_non_unit_tau() {
    let x = fill(128, 11, 4.0);
    let scale = positive_scale(128, 13);
    let shipped = 1.0 + 0.1 * (16384.0f32 / 8192.0).ln();
    assert!(shipped > 1.0, "the shipped position must be past the floor");
    let base = ClipCase::scaled(&x, &scale, (1, 128), 2.0);
    let unit = ok(clip_launch(&base, false), "tau-free");
    for tau in [shipped, 1.5] {
        let c = ClipCase {
            tau: Some(tau),
            ..base
        };
        let got = ok(clip_launch(&c, false), "log_scaled");
        let want = cpu_log_scaled_l2_clip(&x, &scale, 2.0, tau);
        assert_rel(&want, &got, &format!("log_scaled tau={tau}"), 1e-5);
        let ratio = got[17] / unit[17];
        assert!(
            (ratio - tau).abs() < 1e-4,
            "tau={tau} must scale the clipped vector (ratio {ratio}); a dropped tau gives 1.0"
        );
    }
}

/// A unit scale reduces to the plain clip — the two kernels agree when the scale is inert.
#[test]
fn scaled_l2_clip_unit_scale_is_the_plain_clip() {
    let x = fill(128, 3, 4.0);
    let ones = vec![1.0f32; 128];
    let scaled = ok(
        clip_launch(&ClipCase::scaled(&x, &ones, (1, 128), 2.0), false),
        "scaled unit",
    );
    let plain = ok(
        clip_launch(&ClipCase::plain(&x, (1, 128), 2.0), false),
        "plain",
    );
    assert_rel(&plain, &scaled, "scaled unit == plain", 1e-6);
}

/// tau == 1 reduces to the scaled clip — the LogScale factor is inert below the floor (T9/T11).
#[test]
fn log_scaled_l2_clip_unit_tau_is_the_scaled_clip() {
    let x = fill(128, 17, 4.0);
    let scale = positive_scale(128, 19);
    let base = ClipCase::scaled(&x, &scale, (1, 128), 2.0);
    let logged = ok(
        clip_launch(
            &ClipCase {
                tau: Some(1.0),
                ..base
            },
            false,
        ),
        "log tau=1",
    );
    let scaled = ok(clip_launch(&base, false), "scaled");
    assert_rel(&scaled, &logged, "log tau=1 == scaled", 1e-6);
}

/// `rows = 0` and `n = 0` are refused (1001), and the threshold and `tau` must be finite and
/// positive — NaN, 0, negative, +inf each refused (1006).
#[test]
fn the_clip_launchers_guard_their_extents_and_scalars() {
    let x = fill(8, 1, 4.0);
    let scale = positive_scale(8, 2);
    let plain = ClipCase::plain(&x, (1, 8), 8.0);
    let scaled = ClipCase::scaled(&x, &scale, (1, 8), 8.0);
    assert_guard(
        clip_guard(&ClipCase { rows: 0, ..plain }),
        Some(1001),
        "rows = 0",
    );
    assert_guard(clip_guard(&ClipCase { n: 0, ..plain }), Some(1001), "n = 0");
    assert_guard(
        clip_guard(&ClipCase { rows: 0, ..scaled }),
        Some(1001),
        "rows = 0 scaled",
    );
    for bad in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
        let c = ClipCase {
            threshold: bad,
            ..plain
        };
        assert_guard(clip_guard(&c), Some(1006), &format!("threshold={bad}"));
    }
    // A non-unit tau is refused on its own account, and only in the full-layer form.
    for bad in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
        let c = ClipCase {
            tau: Some(bad),
            ..scaled
        };
        assert_guard(clip_guard(&c), Some(1006), &format!("tau={bad}"));
    }
    assert_guard(clip_guard(&plain), None, "valid plain");
    assert_guard(clip_guard(&scaled), None, "valid scaled");
    let logged = ClipCase {
        tau: Some(1.5),
        ..scaled
    };
    assert_guard(clip_guard(&logged), None, "valid log_scaled");
}
