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
//! | the same chain in host f32, weights unrounded | 2.463e-7 |
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
//! computed it in f32. Rounding the weight alone costs 3.703e-3 on the host; the device path
//! measures **5.301e-3**, and the 1.43x between them is the GEMM's accumulation order, which the
//! host estimate does not model — so the honest floor for this comparison is the measured one, not
//! the predicted one. Either way it is decades above the `o_proj` row `glimmer_gate.rs` scores
//! against (8.29e-5, 64x under the measurement): a row from the anchor's fp32 defect runs cannot
//! price a bf16 storage decision, and stretching one to fit is the trap `attend`'s row records.
//! 1e-2 is 1.9x the measured floor and 20.4x under the weaker of the two traps.
//!
//! # What this does NOT do
//!
//! **There is still no layer loop, so this does not gate the engine's wiring.** It gates the
//! comparison the wiring will be held to: it proves the goldens can DISCRIMINATE the operand (the
//! weaker of two wrong ones reddens at 38x the measured floor), and it proves the device path
//! reproduces the reference given the right one. When S3's loop exists, the substitution is one line — its `gate_proj` output in
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
fn sweep(operand: &str, mut f: impl FnMut(f32, &str)) -> (usize, usize) {
    let (mut cases, mut rows_seen) = (0usize, 0usize);
    for (gold, (wname, ws)) in fixture::goldens().iter().zip(fixture::weight_sets()) {
        assert_eq!(gold.name, wname, "weights and golden are out of order");
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
                f(
                    fixture::worst_rel(&got, &want),
                    &format!("{} L{l} t{t}", gold.name),
                );
                cases += 1;
                rows_seen += rows;
            }
        }
    }
    (cases, rows_seen)
}

/// Every (salt, layer, step): the reference's `input_layernorm.out` through its own `gate_proj`
/// reproduces `attn.gate_proj.out`.
#[test]
fn the_gate_operand_is_the_layer_input_and_the_reference_says_so() {
    let (mut worst, mut at) = (0.0f32, String::new());
    let (cases, rows_seen) = sweep("input_layernorm", |r, label| {
        if r > worst {
            worst = r;
            at = label.to_string();
        }
    });
    println!("gate operand: worst rel {worst:e} at {at} over {cases} cases, {rows_seen} rows");
    // Absolutes: two salts x 8 layers x 7 steps, and 12 prompt + 6 decode rows per (salt, layer).
    // Both are pinned rather than derived, because every count the loop could derive comes from the
    // same `tiny_config` the loop reads.
    assert_eq!(
        cases, 112,
        "the sweep covered {cases} (salt, layer, step) cases, not 112"
    );
    assert_eq!(
        rows_seen, 288,
        "the sweep covered {rows_seen} rows, not 288"
    );
    assert!(
        worst < BAR,
        "gate_proj of input_layernorm.out is {worst:e} from the reference at {at} — either the \
         operand is not the layer input or the device path is wrong"
    );
}

/// The same chain fed the two realistic wrong operands. Without this the test above is a statement
/// about `gemm_bf16`, not about the operand: it would pass identically if every candidate were
/// close, and §9 warns that the trap-4 spellings "differ mostly by a scale".
#[test]
fn feeding_gate_proj_the_wrong_activation_reddens_far_above_the_measured_floor() {
    for wrong in ["post_attention_layernorm", "pre_feedforward_layernorm"] {
        let (mut weakest, mut at) = (f32::INFINITY, String::new());
        let (cases, _) = sweep(wrong, |r, label| {
            if r < weakest {
                weakest = r;
                at = label.to_string();
            }
        });
        println!("trap 4 via {wrong}: weakest signal {weakest:e} at {at} over {cases} cases");
        assert_eq!(cases, 112, "the red proof covered {cases} cases, not 112");
        // The WEAKEST case, not the worst: one loud layer must not carry a spelling that is
        // invisible everywhere else.
        //
        // **10x, and the two larger numbers that stood here first were both wrong.** 100x was
        // inherited from `glimmer_gate.rs`, which holds trap 4 to 100x its own bar — but that file
        // scores the SIGMOID of the gate, and sigmoid compresses everything into (0, 1), so the two
        // margins are not the same quantity. 50x was then set from a single-case host probe, and
        // the weakest case over all 112 turned out 4x smaller than that probe.
        //
        // Measured weakest, over every case: post_attention 9.104e-1, pre_feedforward 2.036e-1 —
        // 91x and 20.4x BAR, and 172x and 38x the measured device floor of 5.301e-3. The
        // requirement is that the weakest trap sit an order of magnitude above BAR, which leaves 2x
        // of headroom on the weaker spelling. Stated against BAR rather than against the floor
        // because BAR is what the operand test asserts, so this is the margin between "passes" and
        // "would have been caught".
        assert!(
            weakest > 10.0 * BAR,
            "gating on {wrong} is only {weakest:e} from correct at {at} — this fixture cannot tell \
             the two operands apart, so the test above proves nothing about the operand"
        );
    }
}
