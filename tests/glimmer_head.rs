//! **S2 item 5: the head at vocab 202048, and the softcap no gate in this repo can see.**
//!
//! Three of the plan's claims for this item did not survive reading it:
//!
//! * **"existing `gemv_i8`"** — wrong by dtype. Glimmer's artifact is bf16-verbatim, so
//!   `lm_head.weight` is bf16 and the kernel is `gemm_bf16`, the same one the MLP uses. `gemv_i8`
//!   would need an int8 checkpoint this port does not produce.
//! * **"check `ARGMAX_BYTES`"** — `ARGMAX_BYTES` is `MAXROW * 8 + 4` = **20 bytes**, the argmax's
//!   (index, value) output plus a shared non-finite tag. It has nothing to do with vocab width.
//!   The 808 KB/row the plan means is the LOGIT buffer, `202048 * 4`, which is a different
//!   allocation; both are fine, and conflating them is how a check gets written against the one
//!   that was never at risk.
//! * **"no new kernel"** (item 4's phrasing, inherited here) — the softcap had none. `tanh` appears
//!   nowhere in `kernels/` before `logit_softcap`.
//!
//! # The softcap is the point of this file
//!
//! `glimmer-architecture.md` §5: `logits = 20 * tanh(lm_head(h) * 0.196116 / 20)`. `output_multiplier`
//! is positive and `tanh` is strictly increasing, so **the composition cannot move an argmax**.
//! Greedy equality, teacher-forced argmax and byte-identical output are all blind to omitting it,
//! and the anchor proved that rather than arguing it: `softcap_off` leaves `emitted.ids`
//! **bit-identical** at both draws while the logits move.
//!
//! **And the anchor cannot price it either.** `softcap_off` moves the `logits` bucket by only
//! 4.879e-5 — 13.9x the fp32 floor, against the 297x a `Rel` policy needs — so `tolerance::GLIMMER`
//! carries `logits` as `ExactOnly`. The reason is the tiny model, not the instrument: at untrained
//! weights the logits sit in `tanh`'s linear region, where the softcap is very nearly the identity.
//! At vocab 202048 with trained weights it bites much harder, and S4 is where that can be measured.
//!
//! So this file scores the kernel against a host `tanh` at magnitudes where the function HAS shape,
//! and asserts the two properties that hold at any scale: the argmax is unmoved, and the output is
//! bounded by the cap. What it cannot do is tell you the softcap is present in a decode — nothing
//! at this stage can, which is why `glimmer-port.md` §G3 owes a probability-space check.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::backend::hip::{device_sync, launch_argmax, launch_logit_softcap};

#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{
    GLIMMER_SHIPPED_CONFIG, dev, f32b, fill, from_bf16, gemv_bf16, sync_read, to_bf16, worst_rel,
    zeros,
};

/// Glimmer's text vocabulary and the two logit-path constants, from `glimmer-architecture.md`
/// §1/§5 — not read from the tiny config, which shrinks the vocabulary that is the whole subject.
/// `the_constants_match_the_shipped_config` pins all three to the vendored `config.json`.
///
/// The hidden width is deliberately absent: this file checks the head's OUTPUT dim, and
/// `glimmer_mlp.rs` covers a 6656-wide reduction. See the projection test for why the product of
/// the two is not built.
const VOCAB: usize = 202_048;
const MULT: f32 = 0.196_116_14;
const CAP: f32 = 20.0;

/// What one row of logits costs: `202048 * 4` = **808,192 B**, printed by the projection test.
const LOGIT_BYTES: usize = VOCAB * 4;

/// The three constants above against the vendored `config.json`, because a review found
/// `output_multiplier` was the ONE logit-path constant with no value assertion anywhere in the
/// tree — `validate` narrows it to positive-and-finite, which `0.5` passes — and this file
/// otherwise hands its transcription to both sides of every comparison, so a wrong copy would
/// be structurally invisible on the one operation every greedy gate is blind to.
#[test]
fn the_constants_match_the_shipped_config() {
    let c: serde_json::Value = serde_json::from_str(GLIMMER_SHIPPED_CONFIG).unwrap();
    let t = &c["text_config"];
    assert_eq!(t["vocab_size"].as_u64().unwrap() as usize, VOCAB);
    assert_eq!(t["final_logit_softcapping"].as_f64().unwrap() as f32, CAP);
    assert_eq!(t["output_multiplier"].as_f64().unwrap() as f32, MULT);
}

fn softcap_on_device(x: &[f32], mult: f32, cap: f32) -> Vec<f32> {
    let b = dev(&f32b(x));
    // SAFETY: `b` holds exactly `x.len()` live f32 and outlives the sync inside `sync_read`.
    // Null stream: one kernel, then a join — nothing to order against.
    unsafe { launch_logit_softcap(b.ptr() as *mut f32, x.len(), mult, cap, std::ptr::null_mut()) }
        .expect("logit_softcap launch");
    sync_read(&b)
}

// ------------------------------------------------------------------------------------------

/// The softcap against a host `tanh`, at magnitudes where `tanh` is not its own tangent line.
///
/// **The input scale is the whole design of this test.** At `|x*mult| << cap` the operation is
/// nearly the identity, which is exactly why the anchor's tiny logits could not price it; feeding
/// small values here would reproduce that blindness in a test that then reports success. `scale`
/// 400 puts `x*MULT/CAP` around ±4, where `tanh` has saturated and the softcap is doing its job.
#[test]
fn the_softcap_matches_a_host_tanh_where_tanh_has_shape() {
    for scale in [1.0_f32, 400.0] {
        let x = fill(VOCAB, 1, scale);
        let got = softcap_on_device(&x, MULT, CAP);
        let want: Vec<f32> = x.iter().map(|v| CAP * (v * MULT / CAP).tanh()).collect();
        let r = worst_rel(&got, &want);
        // f32 in, f32 out, one `tanhf` apart — device tanhf vs host libm across 404,096 samples.
        // MEASURED 2026-08-12: 9.54e-8 at scale 400, where the reference magnitude ~19.98 has an
        // f32 ulp of 1.9e-6, so the bar admits ~10 ulps of divergence between the two libms.
        assert!(r <= 1e-6, "scale {scale}: worst rel {r:e}");
        // The property that survives any scale, and the reason the omission is invisible to a
        // greedy gate: nothing leaves the band.
        assert!(
            got.iter().all(|v| v.abs() <= CAP),
            "scale {scale}: a logit escaped the cap"
        );
        // Each arm asserts ITS regime — the saturation census in both directions, so the scale-1
        // arm is the linear-region CONTRAST rather than a loop pass with nothing to prove.
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

/// **The argmax is unmoved — measured, not argued.** This is §5's blindness claim, and the reason
/// every greedy gate in this repo passes without the softcap.
///
/// Asserted over the full vocabulary at a scale where the softcap changes every value materially,
/// so it is a statement about monotonicity rather than about small numbers.
#[test]
fn the_softcap_cannot_move_the_argmax() {
    let x = fill(VOCAB, 2, 400.0);
    let capped = softcap_on_device(&x, MULT, CAP);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &val)| {
                if val > bv { (i, val) } else { (bi, bv) }
            })
            .0
    };
    let (before, after) = (argmax(&x), argmax(&capped));
    assert_eq!(
        before, after,
        "the softcap moved the argmax, so §5's invariance claim is false"
    );
    // And the values DID move, or the check above is vacuous.
    let moved = x.iter().zip(&capped).filter(|(a, b)| a != b).count();
    assert!(
        moved > VOCAB / 2,
        "only {moved} of {VOCAB} logits changed at all"
    );
    println!("argmax held at {before} while {moved} of {VOCAB} logits moved");
}

/// The device argmax over a full-width logit row, with a planted maximum.
///
/// `argmax_reduce` is one block of 256 threads striding the row, so 202048 is 790 strides. Nothing
/// in the tree had run it past a few thousand, and the guard only rejects `n <= 0`.
#[test]
fn the_argmax_reduction_holds_at_the_full_vocabulary() {
    let mut x = fill(VOCAB, 3, 1.0);
    // Past the fill's range, and deliberately NOT at a stride boundary: 256-thread reductions
    // that mishandle their tail lose the last partial stride, and one that loses a whole stride
    // would still find a planted maximum sitting in the middle.
    let planted = VOCAB - 3;
    x[planted] = 99.0;
    let xb = dev(&f32b(&x));
    let out = zeros(8);
    // SAFETY: `xb` is `VOCAB` live f32; `out` is 8 writable bytes for one (i32, f32) pair. Both
    // outlive the `device_sync` below.
    unsafe {
        launch_argmax(
            xb.ptr() as *const f32,
            VOCAB,
            out.ptr() as *mut i32,
            (out.ptr() as *mut f32).wrapping_add(1),
        )
    }
    .expect("argmax launch");
    device_sync().unwrap();
    let raw = fixture::back(&out);
    let idx = i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
    let val = f32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    assert_eq!(
        idx, planted,
        "argmax found {idx}, not the planted {planted}"
    );
    assert_eq!(val, 99.0, "argmax returned {val}");
    println!("argmax over {VOCAB} found the planted maximum at {idx}");
}

/// The launcher's guards, driven — every clause, because an unexercised guard is how item 1's
/// ring bound sat wrong until review. `!(cap > 0.0f)` is spelled that way in the kernel to
/// reject NaN as well as non-positive, and the NaN rows are what hold that spelling: rewritten
/// as `cap <= 0.0f`, a NaN sails through and the two rows below go red. The `isfinite` rows
/// hold the other half (`!(x > 0)` ADMITS +Inf, which NaNs every logit through `Inf * 0`), and
/// the 1002 row holds the transposition guard — swapped constants are a hard sign-quantiser
/// that every greedy gate passes, and the two parameters are adjacent same-typed scalars.
#[test]
fn the_launcher_refuses_what_it_cannot_compute() {
    let b = dev(&f32b(&fill(8, 6, 1.0)));
    let x = b.ptr() as *mut f32;
    let cases: [(usize, f32, f32, Option<i32>, &str); 9] = [
        (0, MULT, CAP, Some(1001), "zero logits"),
        (8, 0.0, CAP, Some(1001), "a zero multiplier"),
        (8, MULT, 0.0, Some(1001), "a zero cap, which divides"),
        (8, f32::NAN, CAP, Some(1001), "a NaN multiplier"),
        (8, MULT, f32::NAN, Some(1001), "a NaN cap"),
        (8, f32::INFINITY, CAP, Some(1001), "an infinite multiplier"),
        (8, MULT, f32::INFINITY, Some(1001), "an infinite cap"),
        (8, CAP, MULT, Some(1002), "the constants TRANSPOSED"),
        (8, MULT, CAP, None, "the real constants, which must pass"),
    ];
    for (n, mult, cap, want, what) in cases {
        // SAFETY: the rejected calls return before any launch; the accepted one writes 8 f32
        // into a live 8-f32 buffer.
        fixture::expect_guard(
            unsafe { launch_logit_softcap(x, n, mult, cap, std::ptr::null_mut()) },
            want,
            what,
        );
    }
    device_sync().unwrap();
}

/// **The omission red proof, and the non-finite pass-through.**
///
/// Omission is the defect this operator's whole story is about — every greedy gate passes with
/// the kernel gone — so the fixture proves ITS metric sees it: unsoftcapped logits at scale 400
/// score ~19 against a 1e-6 bar. And ±Inf must come back ±Inf: `tanhf(±Inf)` is ±1, so the
/// naive kernel maps an overflowed logit to exactly ±cap — finite — one launch before `argmax`'s
/// non-finite bail, which is the engine's only detector for a fault after the last layer.
/// NaN must stay NaN for the same reason.
#[test]
fn omission_is_loud_and_non_finites_survive() {
    let x = fill(4096, 7, 400.0);
    let want: Vec<f32> = x.iter().map(|v| CAP * (v * MULT / CAP).tanh()).collect();
    let r = worst_rel(&x, &want);
    assert!(
        r > 1.0,
        "omitting the softcap moved this metric by only {r:e}, so the metric cannot see the one \
         defect this operator is about"
    );
    println!("softcap omission scores {r:e} against the 1e-6 bar");

    let probe = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0, 400.0];
    let got = softcap_on_device(&probe, MULT, CAP);
    assert_eq!(got[0], f32::INFINITY, "+Inf was laundered to {}", got[0]);
    assert_eq!(got[1], f32::NEG_INFINITY, "-Inf was laundered to {}", got[1]);
    assert!(got[2].is_nan(), "NaN was laundered to {}", got[2]);
    assert_eq!(got[3], 0.0, "zero must map to zero");
    assert!(
        got[4].is_finite() && got[4].abs() <= CAP,
        "a finite logit must still be capped"
    );
}

/// The head projection's OUTPUT width, which is the dim item 5 is about.
///
/// **`k` is deliberately small here and that is not laziness.** The full head weight is
/// `202048 x 6656` bf16 = 2.69 GB, and building it host-side means 1.34 G f32 before conversion —
/// minutes of scalar work and gigabytes of RAM on a machine whose `/tmp` is RAM-backed, for
/// arithmetic that `glimmer_mlp.rs` already exercises at k = 6656. The two dimensions are checked
/// independently and on purpose: that file covers the reduction width, this one covers 202048
/// outputs, and neither claim needs the product.
#[test]
fn the_head_projection_writes_every_one_of_the_202048_outputs() {
    const K: usize = 64;
    let x = fill(K, 4, 1.0);
    let w = to_bf16(&fill(VOCAB * K, 5, 1.0));
    // `gemv_bf16` seeds its output with zeros, which is what makes "every one of them" checkable
    // below rather than "the ones I sampled": an output the kernel never wrote is exactly 0.0, and
    // with this fill no true dot product is.
    let got = gemv_bf16(&x, &w, VOCAB, K);

    let unwritten = got.iter().filter(|v| **v == 0.0).count();
    assert_eq!(
        unwritten, 0,
        "{unwritten} of {VOCAB} logits were never written"
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "a logit came back non-finite"
    );
    for j in [0, 1, VOCAB / 2, VOCAB - 2, VOCAB - 1] {
        let want: f32 = (0..K).map(|i| x[i] * from_bf16(w[j * K + i])).sum();
        let d = (got[j] - want).abs() / want.abs().max(1.0);
        assert!(d <= 1e-5, "output {j}: {d:e}");
    }
    println!("head wrote all {VOCAB} logits ({LOGIT_BYTES} B); ends and midpoint verified");
}
