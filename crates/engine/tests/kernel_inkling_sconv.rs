//! inkling's static short-convolution decode step on the device, scored against the CPU oracle
//! `rivoli_oracles::inkling::sconv` — **no C twin** (owner direction 2026-11-24): a new-model
//! kernel validated against the first-party reference, not the C→Rust bit-identity harness. The
//! k-tap dot is a short reduction whose accumulation may contract to FMA on the device, so the
//! comparison of the OUTPUT is RELATIVE (the clip family's convention for a sum); the state
//! advance is a pure shift+append and is compared bitwise.
//!
//! Tolerances: the anchor carves the fp32 floor per operator
//! (`docs/measurement/inkling2-reference/anchor.md` §Tolerances) and the sconv row is one of the
//! floors the S4 kernel suite owes there; `1e-5` relative here is the allowance for a k-tap dot
//! contracting to FMA, measured on the CPU oracle's own two association orders.
//!
//! `ORACLE_VALIDATED` in `kernel_coverage.rs` waives the C-twin row; this suite is the oracle it cites.
//!
//! Run on the device under `flock /var/run/sys-gpu.lock`, single-threaded (`--test-threads=1`).
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::launch_sconv;
use rivoli_oracles::inkling::sconv as cpu_sconv;

mod common;
use common::*;

/// One geometry's operands — named fields, not six positional args, because a `channels`/`k` swap
/// would be a wrong answer rather than a compile error. `Copy` so the guard test can vary one field
/// with struct-update syntax without consuming the base.
#[derive(Clone, Copy)]
struct SconvCase<'a> {
    weight: &'a [f32],
    state: &'a [f32],
    current: &'a [f32],
    channels: usize,
    k: usize,
    divisor: f32,
}

impl<'a> SconvCase<'a> {
    /// The static divisor `√(k+1)` (T15) is derived from the geometry, so the constructor stays at
    /// four arguments; the guard test overrides `divisor`/geometry with struct-update syntax.
    fn new(weight: &'a [f32], state: &'a [f32], current: &'a [f32], geom: (usize, usize)) -> Self {
        Self {
            weight,
            state,
            current,
            channels: geom.0,
            k: geom.1,
            divisor: ((geom.1 + 1) as f32).sqrt(),
        }
    }
}

/// One geometry's device launch, returning the launcher's own `Result` so the guard tests exercise
/// the same entry point. `alias` points `out` at `current` — the engine's own call shape
/// (`forward.rs` hands the projection scratch as both `current` and `out`); the kernel reads the
/// current tap before writing, so the two must agree bit for bit.
fn sconv_launch(c: &SconvCase, alias: bool) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let wb = dev(&f32b(c.weight));
    let sb = dev(&state_bytes(c.state));
    let mut cb = dev(&f32b(c.current));
    let s = stream();
    // `.max(1)` so the `channels = 0` guard arm still hands `DeviceBuf::new` a real allocation —
    // the launcher refuses before any device access, and a zero-byte buffer is not a test of that.
    let mut out = zeros(c.channels.max(1) * 4);
    let cur = cb.ptr() as *const f32;
    let dst = if alias {
        cb.ptr_mut() as *mut f32
    } else {
        out.ptr_mut() as *mut f32
    };
    // SAFETY: `weight` `channels·k` f32, `state` `channels·(k−1)` f32, `current`/`out` `channels`
    // f32; all outlive the stream `back` syncs on. The launcher guards geometry/divisor first.
    unsafe {
        launch_sconv(
            wb.ptr() as *const f32,
            sb.ptr() as *mut f32,
            cur,
            c.channels,
            c.k,
            c.divisor,
            dst,
            s.raw(),
        )?;
    }
    let written = if alias { &cb } else { &out };
    let o = f32v(&back(written));
    let advanced = f32v(&back(&sb));
    Ok((o, advanced))
}

/// The state's device bytes; a `k = 1` geometry has no history, and a zero-length allocation is a
/// device error rather than a test of the kernel, so one padding word stands in (never read).
fn state_bytes(state: &[f32]) -> Vec<u8> {
    if state.is_empty() {
        return f32b(&[0.0f32]);
    }
    f32b(state)
}

/// The static decode matches the CPU oracle over ragged channel counts and tap widths.
#[test]
fn sconv_matches_the_cpu_oracle() {
    for &(channels, k) in &[(1usize, 4), (33, 4), (128, 4), (257, 4), (64, 3)] {
        let weight = fill(channels * k, 0x1234 + channels as u64, 2.0);
        let state = fill(channels * (k - 1), 0x5678 + channels as u64, 2.0);
        let current = fill(channels, 0x9abc, 2.0);
        let c = SconvCase::new(&weight, &state, &current, (channels, k));
        let (want, want_state) = cpu_sconv(&weight, &state, &current, c.divisor);
        let (got, got_state) = ok(sconv_launch(&c, false), "gpu sconv");
        assert_rel(&want, &got, &format!("sconv c={channels} k={k}"), 1e-5);
        // The advance is a pure shift+append (no arithmetic), so it is bitwise.
        assert_bits(
            &want_state,
            &got_state,
            &format!("sconv state c={channels} k={k}"),
        );
    }
}

/// `out` may alias `current` (the engine's call shape): the kernel reads the current tap before
/// writing, so the aliased launch is bit-identical to the separate-buffer one — same output, same
/// advanced state.
#[test]
fn sconv_output_may_alias_the_current_tap() {
    let (channels, k) = (33usize, 4usize);
    let weight = fill(channels * k, 0x777, 2.0);
    let state = fill(channels * (k - 1), 0x888, 2.0);
    let current = fill(channels, 0x999, 2.0);
    let c = SconvCase::new(&weight, &state, &current, (channels, k));
    let (separate, separate_state) = ok(sconv_launch(&c, false), "separate");
    let (aliased, aliased_state) = ok(sconv_launch(&c, true), "alias");
    assert_bits(&separate, &aliased, "sconv alias out == separate out");
    assert_bits(
        &separate_state,
        &aliased_state,
        "sconv alias advances the same state",
    );
}

/// The recurrence, not just one step: three consecutive launches over the SAME device state must
/// reproduce the CPU chain step for step. A kernel that never shifted (or shifted the wrong way)
/// passes the one-step oracle case on a fresh state and fails here.
#[test]
fn sconv_advances_the_state_across_three_steps() {
    let (channels, k) = (3usize, 3usize);
    let weight = fill(channels * k, 0x111, 1.0);
    let divisor = ((k + 1) as f32).sqrt();
    let mut state = fill(channels * (k - 1), 0x222, 1.0);
    let wb = dev(&f32b(&weight));
    let sb = dev(&f32b(&state));
    let s = stream();
    for step in 0..3 {
        let current = fill(channels, 0x300 + step, 1.0);
        let cb = dev(&f32b(&current));
        let mut out = zeros(channels * 4);
        // SAFETY: `weight` `channels·k`, `state` `channels·(k−1)`, `current`/`out` `channels` f32;
        // all live, and `state` is the same buffer across steps (the advance is in place).
        let r = unsafe {
            launch_sconv(
                wb.ptr() as *const f32,
                sb.ptr() as *mut f32,
                cb.ptr() as *const f32,
                channels,
                k,
                divisor,
                out.ptr_mut() as *mut f32,
                s.raw(),
            )
        };
        ok(r, &format!("sconv step {step}"));
        let (want, want_state) = cpu_sconv(&weight, &state, &current, divisor);
        assert_rel(
            &want,
            &f32v(&back(&out)),
            &format!("sconv step {step}"),
            1e-5,
        );
        assert_bits(
            &want_state,
            &f32v(&back(&sb)),
            &format!("sconv advanced state step {step}"),
        );
        state = want_state;
    }
}

/// `k = 1`: no history at all — the window is the current token alone, the state stays empty, and
/// the divisor is `√2`. The one-tap case is where an off-by-one in `k − 1` reads past the state.
#[test]
fn sconv_k1_has_no_history() {
    let channels = 4usize;
    let weight = fill(channels, 0x444, 1.0);
    let current = fill(channels, 0x555, 1.0);
    let c = SconvCase::new(&weight, &[], &current, (channels, 1));
    let (want, want_state) = cpu_sconv(&weight, &[], &current, 2.0f32.sqrt());
    let (got, got_state) = ok(sconv_launch(&c, false), "sconv k=1");
    assert_rel(&want, &got, "sconv k=1 output", 1e-5);
    assert!(want_state.is_empty(), "k=1 has no state to advance");
    assert!(got_state.iter().all(|&v| v == 0.0), "the state stays empty");
}

/// T15: the static divisor is `√(k+1) = √5` for k=4. Unit weight on the current tap, zero past,
/// residual = current: the output is `(2·current)/√5`, not `current/√5` (a missing residual) nor
/// `2·current/3` (the dynamic variant's divisor).
#[test]
fn sconv_static_divisor_is_sqrt5() {
    let weight = vec![0.0, 0.0, 0.0, 1.0];
    let state = vec![0.0, 0.0, 0.0];
    let current = vec![2.0];
    let c = SconvCase::new(&weight, &state, &current, (1, 4));
    let (got, next) = ok(sconv_launch(&c, false), "sconv T15");
    let want = 4.0 / 5.0f32.sqrt();
    assert!(
        (got[0] - want).abs() < 1e-6,
        "T15: (conv+residual)/√5 = {want}, got {}",
        got[0]
    );
    // The state shifts left one and appends current: [0,0,0] → [0,0,2].
    assert_eq!(next, vec![0.0, 0.0, 2.0]);
}

/// `channels > 0`, `k > 0` (1001), and `divisor` positive finite (1006) are guarded before launch.
#[test]
fn sconv_geometry_is_guarded() {
    let weight = fill(4, 1, 1.0);
    let state = fill(3, 2, 1.0);
    let current = fill(1, 3, 1.0);
    let base = SconvCase::new(&weight, &state, &current, (1, 4));
    assert_guard(
        sconv_launch(
            &SconvCase {
                channels: 0,
                ..base
            },
            false,
        ),
        Some(1001),
        "channels = 0",
    );
    assert_guard(
        sconv_launch(&SconvCase { k: 0, ..base }, false),
        Some(1001),
        "k = 0",
    );
    for bad in [0.0, f32::NAN, -1.0, f32::INFINITY] {
        assert_guard(
            sconv_launch(
                &SconvCase {
                    divisor: bad,
                    ..base
                },
                false,
            ),
            Some(1006),
            &format!("divisor = {bad}"),
        );
    }
    assert_guard(sconv_launch(&base, false), None, "valid geometry");
}
