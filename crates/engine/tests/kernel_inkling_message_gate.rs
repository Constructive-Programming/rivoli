//! inkling's loop MessageGate combine on the device, scored against the CPU oracle
//! `rivoli_oracles::inkling::message_gate` — **no C twin** (owner direction 2026-11-24). The device
//! tanh is ocml's `tanhf`, ~1 ulp from the host's, so the comparison is RELATIVE, not bitwise.
//!
//! The `ORACLE_VALIDATED` census waiver (`kernel_coverage.rs`) cites this suite as the kernel's oracle.
//!
//! Device arm: `flock /var/run/sys-gpu.lock`, `--test-threads=1` (CLAUDE.md).
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // inkling message-gate oracle: panic-on-failure is the idiom

mod common;
use common::{assert_guard, assert_rel, dev, f32b, fill, launch_to_f32, ok};
use rivoli_backend::hip::launch_message_gate;
use rivoli_oracles::inkling::message_gate as cpu_message_gate;

/// One geometry's four operand vectors; `n` derives from the forget vector's length.
#[derive(Clone, Copy)]
struct MessageGateCase<'a> {
    forget_logit: &'a [f32],
    gain_logit: &'a [f32],
    hidden: &'a [f32],
    message: &'a [f32],
    n: usize,
}

impl<'a> MessageGateCase<'a> {
    fn new(
        forget_logit: &'a [f32],
        gain_logit: &'a [f32],
        hidden: &'a [f32],
        message: &'a [f32],
    ) -> Self {
        Self {
            forget_logit,
            gain_logit,
            hidden,
            message,
            n: forget_logit.len(),
        }
    }
}

/// One geometry's device launch, returning the launcher's own `Result` for the guard test.
fn message_gate_launch(c: &MessageGateCase) -> anyhow::Result<Vec<f32>> {
    let fb = dev(&f32b(c.forget_logit));
    let gb = dev(&f32b(c.gain_logit));
    let hb = dev(&f32b(c.hidden));
    let mb = dev(&f32b(c.message));
    // SAFETY: the four inputs and `out` are each `n` f32, outliving the stream `back` syncs on.
    launch_to_f32(c.n, |op, st| unsafe {
        launch_message_gate(
            fb.ptr() as *const f32,
            gb.ptr() as *const f32,
            hb.ptr() as *const f32,
            mb.ptr() as *const f32,
            c.n,
            op,
            st,
        )
    })
}

/// The combine reproduces the CPU oracle across ragged hidden widths.
#[test]
fn message_gate_matches_the_cpu_oracle() {
    for &n in &[1usize, 33, 128, 257, 1024] {
        let f = fill(n, 0x1234 + n as u64, 3.0);
        let g = fill(n, 0x5678 + n as u64, 3.0);
        let h = fill(n, 0x9abc, 3.0);
        let m = fill(n, 0xdef0, 3.0);
        let c = MessageGateCase::new(&f, &g, &h, &m);
        let want = cpu_message_gate(&f, &g, &h, &m);
        let got = ok(message_gate_launch(&c), "gpu message_gate");
        assert_rel(&want, &got, &format!("message_gate n={n}"), 1e-5);
    }
}

/// A large forget logit suppresses `hidden` (forget≈1); a large gain logit keeps `message`
/// (gain≈1). With hidden 5 and message 7 the output is ≈7, not ≈5 nor ≈12.
#[test]
fn message_gate_forget_suppresses_hidden_gain_keeps_message() {
    let f = vec![10.0];
    let g = vec![10.0];
    let h = vec![5.0];
    let m = vec![7.0];
    let c = MessageGateCase::new(&f, &g, &h, &m);
    let got = ok(message_gate_launch(&c), "message_gate gate");
    assert!(
        got[0] > 6.9 && got[0] < 7.1,
        "want ≈7 (message kept, hidden dropped), got {}",
        got[0]
    );
}

/// `n > 0` is guarded (1001) before any launch.
#[test]
fn message_gate_geometry_is_guarded() {
    let v = fill(1, 1, 3.0);
    let base = MessageGateCase::new(&v, &v, &v, &v);
    assert_guard(
        message_gate_launch(&MessageGateCase { n: 0, ..base }),
        Some(1001),
        "n = 0",
    );
    assert_guard(message_gate_launch(&base), None, "valid geometry");
}
