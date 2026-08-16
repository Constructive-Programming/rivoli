//! Muse Glimmer's two pointwise multiplies — `sigmoid_gate` and `logit_softcap` — against host
//! oracles written beside them.
//!
//! **One file because they share the argument that makes both of them hard to test at all: a
//! decode cannot see either one.** Each is one thread per element, each sits BETWEEN two GEMMs as
//! its own launch, and each is invisible to every gate this repo runs on emitted tokens:
//!
//! * `logit_softcap` is `x = cap·tanh(x·mult/cap)`, and `mult > 0` with `tanh` strictly
//!   increasing means **it cannot move an argmax**. Greedy equality, teacher-forced argmax and
//!   byte-identical output all pass with the kernel omitted entirely — the anchor measured that
//!   rather than arguing it, leaving `emitted.ids` bit-identical at both draws while the logits
//!   moved (`old:docs/reference/glimmer-architecture.md` §9 trap 12). It changes every
//!   PROBABILITY, so its evidence has to come from the logits and never from what was decoded.
//! * `sigmoid_gate` is `x *= sigmoid(g)`, and its `g` must be `gate_proj(LAYER INPUT)` — the
//!   post-`input_layernorm` activation — not anything derived from `x` (§9 trap 4). The wrong
//!   operand has the right shapes and the right dtype, so it substitutes cleanly and the model
//!   stays fluent. **No signature can prevent it**, which is what these tests stand in for.
//!
//! Both were also written as SEPARATE launches on the same argument, and it is worth keeping:
//! folding either into its neighbour saves one pass over a large buffer, has no measurement
//! behind it, and would make the neighbour's own fixture stop meaning "the projection is right".
//! Both take a trailing `stream` where null is the null stream, and both sit in a
//! stream-ordered chain where a null-stream member is an unordered read rather than a default.
//!
//! # Ported from `old:tests/glimmer_head.rs` and `old:tests/glimmer_gate.rs`, and what changed
//!
//! Those files score the gate against the anchor's own `attn.o_proj.in_gated` at 112 (salt,
//! layer, step) cases. The goldens live in `crates/oracles/` in this tree and the census that
//! owns this port scans `crates/engine/tests`, so **what came here is the half that needs no
//! captured bytes**: the arithmetic, the guards, the argmax-invariance claim, the non-finite
//! pass-throughs, and the two substitution red proofs. `the_gate_operand_is_not_recoverable_from_
//! the_attend_output` is a statement about the GOLDENS' power and stays with them.
//!
//! What is lost with them is named rather than left implicit: nothing here says where `g` came
//! from. That is a call-site question the layer loop answers, and no fixture at this level can.
//!
//! The bars are stated from the arithmetic rather than transcribed. `tolerance::GLIMMER` (in
//! `crates/oracles/tests/common/tolerance.rs`, a test-local module of another crate this binary
//! cannot name) carries the gate under `o_proj` at `Rel(8.29e-5)` and prices the logit path
//! **`ExactOnly`** — 4.879e-5 of movement over a 3.520e-6 floor is 13.9x, against the 297x that
//! table's own rule needs for a `Rel` row — so there is no logits threshold to inherit even if it
//! could be read. Both bars below are tighter than the `o_proj` row and are argued in place.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_backend::hip::{device_sync, launch_logit_softcap, launch_sigmoid_gate};

mod common;
use common::{Got, Lcg, Want, assert_guard, back, dev, f32b, f32v, ok, worst_rel};

/// Glimmer's text vocabulary — the dim `logit_softcap` is about, and one nothing else in this
/// tree runs a pointwise kernel over. Pinned against the shipped `config.json` in
/// `crates/artifact/tests/glimmer_config.rs`, together with [`CAP`].
const VOCAB: usize = 202_048;

/// `output_multiplier` and `final_logit_softcapping`.
///
/// > **`output_multiplier` has no value assertion anywhere in this tree**, checked 2026-08-16:
/// > `glimmer_config.rs`'s shipped-config test pins the vocabulary, the softcap, `qk_scale_factor`
/// > and both norm epsilons and does not pin this one, and `validate` only narrows it to positive
/// > and finite — which `0.5` passes. That gap belongs beside its siblings in the artifact crate's
/// > config gate and not here, so it is recorded rather than closed.
/// >
/// > **It costs these tests nothing, and that is by construction**: every oracle below takes
/// > `mult` and `cap` as arguments and scores against `cap·tanh(x·mult/cap)` for whatever pair it
/// > is handed, so the constants choose the REGIME and never the answer. The argmax claim needs
/// > only `mult > 0`.
const MULT: f32 = 0.196_116_14;
const CAP: f32 = 20.0;

/// The two together, and they are bundled because they are TRANSPOSABLE.
///
/// They are adjacent, same-typed and interchangeable to the compiler, and swapped they give
/// `0.196·tanh(102·x)` — a hard sign-quantiser that every greedy gate still passes. The launcher
/// refuses `mult >= cap` under its own code (1002) for exactly that reason, and
/// `the_softcap_launcher_refuses_what_it_cannot_compute` drives it. Carrying them as one value is
/// that same argument, made where the compiler can help instead of at the ABI wall.
#[derive(Clone, Copy)]
struct Softcap {
    mult: f32,
    cap: f32,
}

/// The shipped pair, in the order the launcher takes them.
const SHIPPED: Softcap = Softcap {
    mult: MULT,
    cap: CAP,
};

/// The softcap's bar. f32 in, f32 out, one `tanhf` apart — device `tanhf` against host libm over
/// 404,096 samples.
///
/// **MEASURED 2026-08-12 at 9.54e-8 at scale 400** in the ported file, where a reference magnitude
/// of ~19.98 has an f32 ULP of 1.9e-6 — so this admits roughly ten ULP of divergence between two
/// libms, which is the right order for a function neither is required to round correctly.
const SOFTCAP_TOL: f32 = 1.0e-6;

/// The gate's bar, argued rather than transcribed: the only honest disagreement is device `expf`
/// against Rust's, and a relative error `e` in `e^-g` becomes at most `e` in `1/(1+e^-g)`, so the
/// bound is a few ULP of the result. 1e-5 rather than 1e-6 because ROCm's `expf` is specified to a
/// few ULP and is not correctly rounded, and one device window is not the place to discover the
/// difference. It is still 8x tighter than the `o_proj` row the ported file scored against.
///
/// **Not carried from that file's 3.2e-6.** That bar was 20x a measurement taken on the
/// reference's own activations; this fixture draws its own, so the number would be a threshold
/// with nothing behind it here.
const GATE_TOL: f32 = 1.0e-5;

/// `n` draws uniform in `[-scale, scale)`.
///
/// > Written as a fill over a preallocated vector rather than as the `(0..n).map(…)` its sibling
/// > `kernel_glimmer_norm.rs` uses, and `kernel_glimmer_attend.rs` has a third spelling. **They
/// > cannot share a body**: the shared one belongs in `common/reference.rs` beside `Lcg` and
/// > `block_scales`, and `common/` is outside this port's file scope, so `build.rs`'s duplication
/// > gate is what keeps the three apart. That is a debt this port owes — hoist it and delete all
/// > three the next time `common/` is open.
fn fill(n: usize, salt: u64, scale: f32) -> Vec<f32> {
    let mut r = Lcg(salt);
    let mut v = vec![0.0f32; n];
    v.iter_mut().for_each(|x| *x = r.f() * scale);
    v
}

/// `logit_softcap` over a copy of `x`, returning what the device produced.
fn softcap(x: &[f32], s: Softcap) -> Vec<f32> {
    let mut b = dev(&f32b(x));
    // SAFETY: `b` holds exactly `x.len()` live writable f32 and outlives the join inside `back`.
    // Null stream: this fixture launches one kernel and joins, so there is nothing to order.
    let r = unsafe {
        launch_logit_softcap(
            b.ptr_mut() as *mut f32,
            x.len(),
            s.mult,
            s.cap,
            std::ptr::null_mut(),
        )
    };
    ok(r, "logit_softcap");
    f32v(&back(&b))
}

/// `x *= sigmoid(g)` on the device, returning the product.
fn gate(x: &[f32], g: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), g.len(), "the gate and the value must be one width");
    let (mut xb, gb) = (dev(&f32b(x)), dev(&f32b(g)));
    // SAFETY: both buffers hold exactly `x.len()` live f32 in DISTINCT allocations — the kernel's
    // two parameters are `__restrict__` and aliasing them is what that qualifier disclaims — and
    // both outlive the join inside `back`. Null stream, for the reason `softcap` gives.
    let r = unsafe {
        launch_sigmoid_gate(
            xb.ptr_mut() as *mut f32,
            gb.ptr() as *const f32,
            x.len(),
            std::ptr::null_mut(),
        )
    };
    ok(r, "sigmoid_gate");
    f32v(&back(&xb))
}

/// The host softcap — `cap·tanh(v·mult/cap)` — with the non-finite pass-through the kernel makes.
///
/// The pass-through is part of the REFERENCE here and not a special case bolted on: a host oracle
/// that mapped ±Inf to ±cap would agree with the naive kernel and disagree with the one that
/// ships, so the property would be untestable through the comparison and would need a second
/// fixture to say what this one already knows.
fn host_softcap(x: &[f32], s: Softcap) -> Vec<f32> {
    x.iter()
        .map(|v| {
            if v.is_finite() {
                s.cap * (v * s.mult / s.cap).tanh()
            } else {
                *v
            }
        })
        .collect()
}

/// The index of the largest element, ties going to the lowest index — `gpu.rs::argmax`'s own fold,
/// so what is asserted is the decision the engine actually makes.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &val)| {
            if val > bv { (i, val) } else { (bi, bv) }
        })
        .0
}

// ---- logit_softcap -------------------------------------------------------------------------

/// The softcap against a host `tanh`, at magnitudes where `tanh` is not its own tangent line.
///
/// **The input scale is the whole design of this test.** At `|x·mult| << cap` the operation is
/// nearly the identity, which is exactly why the anchor's tiny logits could not price it; feeding
/// small values here would reproduce that blindness in a test that then reports success. Scale 400
/// puts `x·mult/cap` around ±4, where `tanh` has saturated and the softcap is doing its job.
///
/// Each arm asserts ITS regime, so the scale-1 arm is the linear-region CONTRAST rather than a
/// loop pass with nothing to prove — a saturation census in both directions.
#[test]
fn the_softcap_matches_a_host_tanh_where_tanh_has_shape() {
    for scale in [1.0f32, 400.0] {
        let x = fill(VOCAB, 0x11, scale);
        let got = softcap(&x, SHIPPED);
        let r = worst_rel(Got(&got), Want(&host_softcap(&x, SHIPPED)));
        assert!(r <= SOFTCAP_TOL, "scale {scale}: worst rel {r:e}");
        // The property that survives any scale, and the reason the omission is invisible to a
        // greedy gate: nothing leaves the band.
        assert!(
            got.iter().all(|v| v.abs() <= CAP),
            "scale {scale}: a logit escaped the cap"
        );
        let saturated = got.iter().filter(|v| v.abs() > 0.9 * CAP).count();
        if scale > 100.0 {
            assert!(
                saturated > VOCAB / 100,
                "only {saturated} of {VOCAB} logits reached 90% of the cap at scale {scale}, so \
                 this case is still measuring tanh's linear region"
            );
        } else {
            assert_eq!(
                saturated, 0,
                "logits saturated at scale {scale}, so the linear-region arm is not one"
            );
        }
        println!("softcap at scale {scale}: worst rel {r:e}, {saturated} logits past 0.9*cap");
    }
}

/// **The argmax is unmoved — measured, not argued.** This is the invariance claim, and the reason
/// every greedy gate in this repo passes with this kernel omitted.
///
/// Asserted over the full vocabulary at a scale where the softcap changes every value materially,
/// so it is a statement about monotonicity rather than about small numbers. The second assertion
/// is what stops the first from being vacuous: if the values had not moved, "the argmax held"
/// would be a claim about a no-op.
#[test]
fn the_softcap_cannot_move_the_argmax() {
    let x = fill(VOCAB, 0x12, 400.0);
    let capped = softcap(&x, SHIPPED);
    // A post-cap argmax, not THE argmax: at this scale the top of tanh is so flat that
    // distinct top logits can round to the same f32 (top-two gap ~1.2e-6 against a 1.9e-6
    // ulp — review 2026-08-16), so index equality reddens on a CORRECT kernel for ~1 in 3
    // seeds. The mathematical claim is monotonicity: the pre-cap winner must still hold
    // the post-cap maximum VALUE, ties included.
    let best = capped.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(
        capped[argmax(&x)],
        best,
        "the softcap demoted the pre-cap argmax below the post-cap maximum, so its \
         invariance claim is false"
    );
    let moved = x.iter().zip(&capped).filter(|(a, b)| a != b).count();
    assert!(
        moved > VOCAB / 2,
        "only {moved} of {VOCAB} logits changed at all"
    );
    println!("pre-cap argmax kept the post-cap maximum while {moved} of {VOCAB} logits moved");
}

/// **The omission red proof, and the non-finite pass-through.**
///
/// Omission is the defect this operator's whole story is about — every greedy gate passes with the
/// kernel gone — so the fixture proves ITS metric sees it: unsoftcapped logits at scale 400 score
/// far over the bar the correct output is held to. **A greedy check could not do this**, which is
/// why the oracle reads logits.
///
/// And ±Inf must come back ±Inf. `tanhf(±Inf)` is ±1, so the naive form maps an overflowed logit
/// to exactly ±cap — FINITE — one launch before `argmax`'s non-finite bail, which is the engine's
/// only detector for a fault after the last layer. Laundering it there deletes the detector and
/// the emitted token becomes whichever vocab entry overflowed first. NaN must stay NaN for the
/// same reason.
#[test]
fn softcap_omission_is_loud_and_non_finites_survive() {
    let x = fill(4096, 0x13, 400.0);
    let r = worst_rel(Got(&x), Want(&host_softcap(&x, SHIPPED)));
    assert!(
        r > 1.0,
        "omitting the softcap moved this metric by only {r:e}, so it cannot see the one defect \
         this operator is about"
    );
    println!("softcap omission scores {r:e} against the {SOFTCAP_TOL:e} bar");

    let probe = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0, 400.0];
    let got = softcap(&probe, SHIPPED);
    assert_eq!(got[0], f32::INFINITY, "+Inf was laundered to {}", got[0]);
    assert_eq!(
        got[1],
        f32::NEG_INFINITY,
        "-Inf was laundered to {}",
        got[1]
    );
    assert!(got[2].is_nan(), "NaN was laundered to {}", got[2]);
    assert_eq!(got[3], 0.0, "zero must map to zero");
    assert!(
        got[4].is_finite() && got[4].abs() <= CAP,
        "a finite logit must still be capped"
    );
}

/// The launcher's guards, every clause — because an unexercised guard is how the sibling
/// `gqa_attend`'s ring bound sat wrong until review found it.
///
/// `!(cap > 0.0f)` is spelled that way in the kernel to reject NaN as well as non-positive, and
/// the NaN rows are what hold that spelling: rewritten as `cap <= 0.0f`, a NaN sails through and
/// those two rows go red. The `isfinite` rows hold the other half — `!(x > 0)` ADMITS +Inf, and a
/// +Inf cap NaNs every logit through `Inf * 0` while a +Inf multiplier sign-quantises them.
///
/// **The 1002 row holds the transposition guard**, and that is the sharpest one here. The two
/// scalars are adjacent, same-typed and interchangeable, and swapped they give
/// `0.196·tanh(102·x)` — a hard sign-quantiser that every greedy gate still passes. For any model
/// with a softcap the pre-multiplier is a `1/sqrt(d)`-scale value and the cap is O(10), so the
/// order is checkable and the launcher refuses `mult >= cap` under its own code.
#[test]
fn the_softcap_launcher_refuses_what_it_cannot_compute() {
    let mut b = dev(&f32b(&fill(8, 0x14, 1.0)));
    let x = b.ptr_mut() as *mut f32;
    for (n, mult, cap, want, what) in [
        (0usize, MULT, CAP, Some(1001), "zero logits"),
        (8, 0.0, CAP, Some(1001), "a zero multiplier"),
        (8, MULT, 0.0, Some(1001), "a zero cap, which divides"),
        (8, f32::NAN, CAP, Some(1001), "a NaN multiplier"),
        (8, MULT, f32::NAN, Some(1001), "a NaN cap"),
        (8, f32::INFINITY, CAP, Some(1001), "an infinite multiplier"),
        (8, MULT, f32::INFINITY, Some(1001), "an infinite cap"),
        (8, CAP, MULT, Some(1002), "the constants TRANSPOSED"),
        (8, MULT, CAP, None, "the real constants, which must pass"),
    ] {
        // SAFETY: the rejected calls return before any launch; the accepted one writes 8 f32 into
        // a live 8-f32 buffer that outlives the sync below.
        assert_guard(
            unsafe { launch_logit_softcap(x, n, mult, cap, std::ptr::null_mut()) },
            want,
            what,
        );
    }
    device_sync().expect("sync the accepted softcap dispatch");
}

// ---- sigmoid_gate --------------------------------------------------------------------------

/// The gate against a host sigmoid, and the red proof that it is **not `swiglu`**.
///
/// `swiglu` is `silu(g)·u` = `g·sigmoid(g)·u`, which carries an extra factor of `g`. The two are
/// one `* g` apart, neither is spellable as the other, and both take `(g, u, n)` — so the wrong
/// one substitutes without a shape changing. The red proof is that extra factor, measured over the
/// same fixture rather than argued from the algebra.
///
/// `g` is drawn at ±6 rather than ±1: the sigmoid is very nearly its own tangent line on [-1, 1],
/// so a defect that dropped it entirely would land inside a relative tolerance on that range. The
/// saturating tails are where the function has shape — and they are also where `swiglu`'s extra
/// factor is largest, so the two arms want the same draw.
#[test]
fn the_gate_multiplies_by_the_sigmoid_and_is_not_swiglu() {
    let n = 4096;
    let x = fill(n, 0x15, 1.0);
    let g = fill(n, 0x16, 6.0);
    let sigmoid: Vec<f32> = g.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let want: Vec<f32> = x.iter().zip(&sigmoid).map(|(a, s)| a * s).collect();
    let got = gate(&x, &g);
    let r = worst_rel(Got(&got), Want(&want));
    println!("sigmoid_gate over {n} elements: worst rel {r:e} against tol {GATE_TOL:e}");
    assert!(
        r <= GATE_TOL,
        "sigmoid_gate: worst rel {r:e} > {GATE_TOL:e}"
    );

    // `silu(g)·x` in place of `sigmoid(g)·x` — the swiglu form, on the host, because `swiglu` is
    // not this file's kernel and taking it here would score one kernel's defect through another
    // kernel's arithmetic.
    let swiglu: Vec<f32> = want.iter().zip(&g).map(|(h, gv)| h * gv).collect();
    let d = worst_rel(Got(&swiglu), Want(&want));
    println!("the swiglu form in the gate's place moves it {d:e}");
    assert!(
        d > 1.0e-1,
        "silu(g)*x differs from sigmoid(g)*x by only {d:e} at this draw, so the fixture cannot \
         tell the two combines apart"
    );
}

/// **A non-finite gate gates by the LIMIT — finite output from non-finite input, and that is a
/// reviewed decision rather than an accident.**
///
/// `sigmoid(+Inf)` is exactly 1 and `sigmoid(-Inf)` exactly 0, so an overflowed `gate_proj` passes
/// the value through untouched or annihilates it, and either way `flag_nonfinite` on the residual
/// sees nothing. It was reviewed 2026-08-12 and left as-is on three grounds: the limits are what
/// the sigmoid converges to, a NaN `g` still propagates, and the softcap keeps the terminal
/// detector. Recorded HERE as a driven case so the choice is a property and not a comment.
///
/// The NaN row is the one that makes the other two safe to accept, so it is asserted alongside
/// them rather than in a test of its own.
#[test]
fn a_non_finite_gate_gates_by_the_limit() {
    let x = [3.0f32, 3.0, 3.0, 3.0, 3.0];
    let g = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0, -30.0];
    let got = gate(&x, &g);
    assert_eq!(got[0], 3.0, "+Inf must gate by 1, got {}", got[0]);
    assert_eq!(got[1], 0.0, "-Inf must gate by 0, got {}", got[1]);
    assert!(got[2].is_nan(), "a NaN gate must propagate, got {}", got[2]);
    assert_eq!(got[3], 1.5, "sigmoid(0) is exactly 0.5, got {}", got[3]);
    // -30 is far enough into the tail that `expf` overflows nothing and the product underflows to
    // zero, which is the same answer as the limit — so the two rows above are not a discontinuity
    // the kernel introduces at infinity, they are where a smooth function already was.
    assert!(
        got[4].abs() < 1e-6 && got[4].is_finite(),
        "a deeply negative gate must approach zero smoothly, got {}",
        got[4]
    );
}

/// The launcher's one guard, driven — `n == 0` must be a code, not a zero-block launch.
#[test]
fn the_gate_launcher_refuses_an_empty_row() {
    let mut b = dev(&f32b(&[1.0f32; 4]));
    let x = b.ptr_mut() as *mut f32;
    // SAFETY: the call must return before any launch — that is what is being asserted — so the
    // aliased pointers are never dereferenced. Aliasing them is deliberate and safe only because
    // of that; the kernel's own parameters are `__restrict__` and a launched call must not.
    assert_guard(
        unsafe { launch_sigmoid_gate(x, x as *const f32, 0, std::ptr::null_mut()) },
        Some(1001),
        "an empty row",
    );
}
