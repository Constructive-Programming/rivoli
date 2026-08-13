//! **S3 item 3: WHICH activation `gate_proj` consumes, scored against the reference's own bytes.**
//!
//! `glimmer-architecture.md` §9 trap 4: `gate_proj` consumes the **layer input** `h` — the
//! post-`input_layernorm` activation — and NOT the attention output, the residual, or the
//! post-attention norm. `tests/glimmer_gate.rs` (S2 item 3) proved the *kernel* `sigmoid_gate`
//! given the reference's own `gate_proj.out`, and says in its own header that where that operand
//! came from "is a call-site question that S3 answers". This is that answer.
//!
//! # The gate needs a weight, and the weight is NOT recoverable — that is the finding here
//!
//! Scoring `gate_proj` means computing it, which needs the projection. The anchor is
//! activations-only by design, and the obvious move — recover the weight from the captures the way
//! `glimmer_norm.rs` recovers each sandwich norm's `1 + w` — **does not merely lose power here, it
//! is vacuous.** `gate_proj` is 72 -> 48 and a layer sees 18 rows (12 prompt + 6 decode), so each
//! output element has 18 equations against 72 unknowns: underdetermined by 4x, and EVERY candidate
//! operand admits a weight reproducing the captures exactly. The norms escape that only because
//! they are elementwise, where one row recovers one parameter.
//!
//! So the driver exports it (`--dump-weights`, `weights_capture`) into a separate file per salt.
//! Separate, so the goldens' bytes and their four pinned FNVs do not move — and that was verified
//! by regenerating both goldens with the flag on and finding them BYTE-IDENTICAL.
//!
//! # What it measured (2026-08-13, both salts, every layer and step)
//!
//! | | `max|Δ| / max|reference|` |
//! |---|---:|
//! | `input_layernorm.out` through `gemm_bf16` — the correct operand | **5.301e-3** (text-1 L7 t3) |
//! | the same chain unrounded, in host f64 | 3.949e-7 (text-2 L7 t0) |
//! | **the bar** | **1e-2** |
//! | trap 4a: `post_attention_layernorm.out` instead, WEAKEST case | **9.104e-1** (text-2 L5 t1) |
//! | trap 4b: `pre_feedforward_layernorm.out` instead, WEAKEST case | **2.036e-1** (text-1 L7 t6) |
//!
//! **Both trap rows are the weakest case over all 112, and getting that wrong is how this file's
//! first three bars were set.** The figures written here first — 1.601e0 and 8.104e-1 — were a
//! single (layer, step) probe at L0 t0. The weakest `pre_feedforward` case is 4x smaller than that
//! probe, and a margin chosen from the probe went red. A red proof is only as strong as its
//! WEAKEST case, because one loud layer will happily carry a spelling that is invisible in the
//! other 111.
//!
//! 112 (salt, layer, step) cases over 288 rows, both counts asserted as absolutes.
//!
//! **The bar is set by the FORMAT, not by the operator, and that is why no `tolerance.rs` row
//! applies.** rivoli stores this projection bf16, as the real checkpoint does, and the reference
//! computed it in f32. **bf16 weight truncation is the ENTIRE floor**: recomputing this chain on the
//! host with the weight truncated exactly as `to_bf16` does reproduces 5.3012e-3 at the same
//! (salt, layer, step) the device reports, so the device contributes under 1e-6 of it. It is 64x
//! above the `o_proj` row `glimmer_gate.rs` scores against (8.29e-5) — a row from the anchor's fp32
//! defect runs cannot price a bf16 storage decision, and stretching one to fit is the trap
//! `attend`'s row records.
//!
//! > **CORRECTED 2026-08-13, by review, and the mistake is the interesting part.** This said
//! > "rounding the weight alone costs 3.703e-3 on the host; the device measures 5.301e-3, and the
//! > 1.43x between them is the GEMM's accumulation order". **There is no 1.43x.** My host probe
//! > rounded the weight to nearest-even and ALSO narrowed the activation, neither of which the
//! > device path does — `to_bf16` TRUNCATES and `gemm_bf16` takes `x` as f32. Same chain, right
//! > rounding: 5.3012e-3, four significant figures and the same location. Re-associating a 72-term
//! > f32 dot moves the result by ~1e-7, three orders too small to have been the cause. So I blamed
//! > the GPU for my own host code's rounding mode and wrote it up as a transferable lesson about
//! > host predictions under-shooting devices. The lesson that survives is narrower and duller:
//! > **make the host model the device's arithmetic exactly before comparing them at all.**
//!
//! **BAR is a NAMED EXCEPTION to `tolerance.rs`'s two gated rules, and not merely absent from
//! them.** That module requires a `Rel` threshold to sit at 10x its floor and 30x under the weakest
//! defect it must catch — so no `Rel` exists below a 297x margin. This comparison's margin is
//! 2.036e-1 / 5.301e-3 = **38x**, which is ExactOnly-class by that rule, and exact is impossible
//! against a reference that computed in f32 while the engine stores bf16. So BAR sits at 1.9x its
//! floor by exception, `const` rather than a `Tol` row, and `tolerances_leave_room` never sees it.
//! **The known cost of 1.9x:** narrowing the activation to bf16 as well — the natural spelling once
//! the engine holds `h` in bf16 — measures 8.466e-3, which is 1.18x UNDER the bar. A correct
//! implementation that is twice as hot as this chain reddens, and the failure message names neither
//! cause. Re-measure the floor when the arithmetic changes; do not widen the bar.
//!
//! # What this does NOT do
//!
//! **There is still no layer loop, so this does not gate the engine's wiring.** It gates the
//! comparison the wiring will be held to: it proves the goldens can DISCRIMINATE the operand (the
//! weakest of three wrong ones reddens at 38x the correct one), and it proves the device path
//! reproduces the reference given the right one.
//!
//! **And it hands S3 a bill, measured by review 2026-08-13.** `glimmer_gate.rs` scores
//! `attn.o_proj.in_gated` under the `o_proj` row at `Rel(8.29e-5)`, feeding the kernel the
//! REFERENCE's `gate_proj.out`. Propagate this file's bf16 gate through the sigmoid instead and
//! `in_gated` shifts by **1.634e-3 — 20x that row.** So the moment the loop feeds rivoli's own gate
//! value into `sigmoid_gate`, that gate reddens for a reason that is neither a kernel defect nor a
//! wiring defect, and it will look like both. S3 has to choose: `glimmer_gate.rs` keeps consuming
//! the reference's `gate_proj.out` and stops being a wiring gate, or the `o_proj` row is re-derived
//! against a bf16 gate. (The same measurement confirms this file's other claim with a number: the
//! sigmoid compresses 5.301e-3 to 1.634e-3, a factor of 3.2, which is why a margin borrowed from
//! sigmoid-space did not carry.) When S3's loop exists, the substitution is one line — its `gate_proj` output in
//! place of this file's `gemm_bf16` call — and until then item 3's own text ("the loop is scored
//! against `gate_proj.out` directly") is a contract, not a passing gate.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

#[path = "common/glimmer_fixture.rs"]
mod fixture;

/// The bar, and the two operands that must blow through it. See the header for why this is a
/// literal rather than a `tolerance.rs` row.
const BAR: f32 = 1.0e-2;

/// Score `<operand>.out` through each layer's own `gate_proj` against `attn.gate_proj.out`, for
/// every (salt, layer, step), handing each case's relative error and label to `f`.
///
/// One walker, because the correct operand and the wrong ones must travel the SAME path — if the
/// red proof took its own route it would be evidence about that route. jscpd refused the second
/// copy of this loop, which is the same conclusion arrived at mechanically.
fn sweep(operand: &str, mut f: impl FnMut(f32, &str)) {
    let (mut cases, mut rows_seen, mut scored) = (0usize, 0usize, 0usize);
    // **`zip` stops at the shorter side, silently, and the absolute counts below cannot see it.**
    // `glimmer-anchor.sh` regenerates a golden for every salt in `SALTS` and never passes
    // `--dump-weights`, so a third salt is exactly the round that adds a golden with no weight set:
    // zip would drop it, the loop would run 2x8x7 = 112 cases, and both censuses would pass while
    // this gate silently covered one fewer salt than every other Glimmer fixture. The step axis is
    // pinned by those counts; this is the salt axis. Review, 2026-08-13.
    assert_eq!(
        fixture::goldens().len(),
        fixture::weight_sets().len(),
        "a salt with no weight set is a salt this gate would skip without saying so"
    );
    for (gold, (wname, ws)) in fixture::goldens().iter().zip(fixture::weight_sets()) {
        assert_eq!(gold.name, wname, "weights and golden are out of order");
        // **The label check above proves the ORDER OF TWO ARRAYS OF STRING LITERALS, nothing more**
        // — swap the `include_bytes!` paths under unchanged labels and it stays green. The real tie
        // is in the bytes and was going unread: `--dump-weights` writes the golden's own metadata
        // into the weight file, so these four keys are the provenance. Review, 2026-08-13.
        for k in ["salt", "defect", "driver", "dtype"] {
            assert_eq!(
                ws.meta_get(k),
                gold.g.meta_get(k),
                "{}: weight file's {k} is not the golden's",
                gold.name
            );
        }
        let (hidden, layers) = (gold.n("hidden_size"), gold.n("num_hidden_layers"));
        // `steps()` is `(prompt_len, decode_steps)` — NOT a count of steps, which is what its name
        // reads as. The captures run `t0..=decode`: one prefill plus one per decode step.
        let (_, decode) = gold.steps();
        for l in 0..layers {
            // `n` off the weight's own shape rather than `len()/hidden`: the division would accept
            // a [72, 48] stored transposed, which is exactly the defect that has the same length.
            let (wshape, w) = fixture::float(&ws, &format!("L{l}.attn.gate_proj.weight"));
            assert_eq!(
                wshape[1], hidden,
                "L{l} gate_proj weight is {wshape:?}, not [n, {hidden}]"
            );
            let n = wshape[0];
            // BAR is 1.9x a floor measured at exactly these two widths, and a bf16 GEMM's error
            // moves with `k`. Neither width is otherwise pinned — `hidden` comes from tiny_config
            // and `n` from the weight file — so a re-vendored tiny model would move the floor while
            // the bar sat still. Review, 2026-08-13.
            assert_eq!(
                (hidden, n),
                (72, 48),
                "the tiny model moved to {hidden} -> {n}; BAR was measured at 72 -> 48 and has to \
                 be re-measured, not carried"
            );
            for t in 0..=decode {
                let (rows, _) = gold.geometry(t);
                // `[1, rows, w]`, batch included: `cap` asserts the FULL shape, which is the point
                // — a capture that changed rank would otherwise be reshaped into silence here.
                let x = fixture::cap(
                    gold,
                    &format!("t{t}.L{l}.{operand}.out"),
                    &[1, rows, hidden],
                    false,
                );
                let want = fixture::cap(
                    gold,
                    &format!("t{t}.L{l}.attn.gate_proj.out"),
                    &[1, rows, n],
                    false,
                );
                // bf16 weight: the format the engine stores this projection in, and the reference
                // computed it f32 — which is what sets BAR. See the header.
                let got = fixture::gemm_bf16(&x, &fixture::to_bf16(w), rows, n, hidden);
                let r = fixture::worst_rel(&got, &want);
                // **`worst_rel` returns INFINITY for a non-finite `got`, and the red proof below
                // takes a MINIMUM seeded at INFINITY — so an all-NaN GEMM would have satisfied its
                // bar.** That helper's own comment records a broken kernel passing 9 of 9 that way;
                // this file reintroduced the trap one level up, at the consumer. Review, 2026-08-13.
                assert!(
                    r.is_finite(),
                    "{} L{l} t{t} scored non-finite — the device path produced NaN or Inf, which \
                     every bar in this file would otherwise read as a large error",
                    gold.name
                );
                f(r, &format!("{} L{l} t{t}", gold.name));
                scored += 1;
                cases += 1;
                rows_seen += rows;
            }
        }
    }
    // **The census lives in the WALKER, not in one caller.** It was in the correct-operand test
    // only, so the red proof could have run a collapsed geometry — `cases` is invariant to that
    // while `rows_seen` is not, and a `geometry()` returning 1 row at t=0 keeps 112 cases while
    // dropping to 112 rows, exercising only the m=1 path this commit's gemv->gemm generalisation
    // exists to leave behind. `scored` is separate from `cases` because they are only equal by
    // adjacency: one `continue` between them and both bars go vacuous with the counts still right.
    assert_eq!(
        cases, 112,
        "{operand}: {cases} (salt, layer, step) cases, not 112"
    );
    assert_eq!(rows_seen, 288, "{operand}: {rows_seen} rows, not 288");
    assert_eq!(
        scored, cases,
        "{operand}: {scored} of {cases} cases reached the score"
    );
}

/// Every (salt, layer, step): the reference's `input_layernorm.out` through its own `gate_proj`
/// reproduces `attn.gate_proj.out`.
#[test]
fn the_gate_operand_is_the_layer_input_and_the_reference_says_so() {
    let (worst, at) = worst_correct();
    println!("gate operand: worst rel {worst:e} at {at}");
    // **A LOWER bar too, and it is not ceremony.** The upper bar prices bf16 storage; if the
    // rounding stopped happening — `to_bf16` widened to a no-op, f32 weights handed to the GEMM,
    // the kernel accumulating unrounded — the score drops to the host figure 2.463e-7 and passes,
    // while this file's whole "the bar is set by the FORMAT" argument silently stops being
    // exercised and the header still cites 5.301e-3 as its floor. Review, 2026-08-13.
    assert!(
        worst > 1.0e-3,
        "the correct operand scores {worst:e}, below the 3.7e-3 bf16 rounding alone costs — the \
         format this bar prices is not being paid, so BAR is measuring nothing"
    );
    assert!(
        worst < BAR,
        "gate_proj of input_layernorm.out is {worst:e} from the reference at {at} — either the \
         operand is not the layer input or the device path is wrong"
    );
}

/// The correct operand's worst case, which both tests need: one as the thing under test, the other
/// as the denominator that makes the red proof a ratio.
fn worst_correct() -> (f32, String) {
    let (mut worst, mut at) = (0.0f32, String::new());
    sweep("input_layernorm", |r, label| {
        if r > worst {
            worst = r;
            at = label.to_string();
        }
    });
    (worst, at)
}

/// The same chain fed the wrong operands. Without this the test above is a statement about
/// `gemm_bf16`, not about the operand: it would pass identically if every candidate were close, and
/// §9 warns that the trap-4 spellings "differ mostly by a scale".
///
/// **Which spellings, precisely — because this covers two of §9's three and neither is the hard
/// one.** §9 names the attention output, the pre-norm residual, and the post-attention norm.
/// `attn.o_proj.out` is the attention output and is captured, so it is scored here.
/// `post_attention_layernorm` and `pre_feedforward_layernorm` are activations from a different
/// point in the layer — the EASY discrimination. **The pre-norm residual is the one §9's "differs
/// mostly by a scale" is actually about**, since the norm is what stands between it and the correct
/// operand, and it is not captured at all: the driver taps module OUTPUTS, and `input_layernorm`'s
/// INPUT needs a `register_forward_pre_hook` and a re-vendor. Review found this, 2026-08-13; until
/// it lands, "the goldens discriminate the operand" means these three and not that one.
#[test]
fn feeding_gate_proj_the_wrong_activation_reddens_far_above_the_measured_floor() {
    // The denominator. A red proof that asserts only a LOWER bound is satisfied by every way the
    // shared path can break — NaN, a half-written output, the wrong `want` — because each of those
    // RAISES the measured error. Stated as a ratio against the correct operand's worst case, a
    // common-mode failure moves numerator and denominator together and the claim survives. Review,
    // 2026-08-13.
    let (correct, _) = worst_correct();
    for wrong in [
        "attn.o_proj",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
    ] {
        let (mut weakest, mut at) = (f32::INFINITY, String::new());
        sweep(wrong, |r, label| {
            if r < weakest {
                weakest = r;
                at = label.to_string();
            }
        });
        let ratio = weakest / correct;
        println!(
            "trap 4 via {wrong}: weakest {weakest:e} at {at}, {ratio:.1}x the correct operand"
        );
        // The WEAKEST case, not the worst: one loud layer must not carry a spelling that is
        // invisible everywhere else.
        //
        // **15x, and the two numbers that stood here first were both wrong.** 100x was inherited
        // from `glimmer_gate.rs`, which holds trap 4 to 100x its own bar — but that file scores the
        // SIGMOID of the gate, and sigmoid compresses everything into (0, 1), so the two margins are
        // not the same quantity. 50x was then set from a single-case host probe, and the weakest
        // case over all 112 turned out 4x smaller than that probe.
        //
        // Measured against the correct operand's 5.301e-3: post_attention 172x, pre_feedforward
        // 38x. 15x leaves 2.5x of headroom on the weaker of those two.
        assert!(
            ratio > 15.0,
            "gating on {wrong} is only {weakest:e} from correct at {at} — this fixture cannot tell \
             the two operands apart, so the test above proves nothing about the operand"
        );
    }
}
