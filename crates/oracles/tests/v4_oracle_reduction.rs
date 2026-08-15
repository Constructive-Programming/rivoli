//! **Reduction ORDER: the two absolute gates, and the floor they imply.**
//!
//! Every other test in the `v4_oracle*` family is SELF-RELATIVE — a defected capture against
//! an undefected one — so an error the oracle shares with its own defect matrix cancels on
//! both sides and is invisible. Reduction order is exactly that class of error: the running
//! bf16 fold at model.py:427 was wrong for the life of the file and no relative test could
//! see it. So this binary asks the two questions only an ABSOLUTE comparison can answer:
//!
//! 1. Does the oracle's reduction reproduce **torch's**, and does the defect reproduce the
//!    **fold torch reproduces**? ([`bf16_reduction_matches_torch_and_not_a_running_fold`])
//! 2. What does a CORRECT but differently-ordered reduction cost against this oracle — the
//!    noise any real-dims gate has to clear?
//!    ([`the_reassociation_floor_bounds_any_tolerance_these_goldens_can_have`])
//!
//! Split out of `v4_oracle.rs` on 2026-08-15 for the 800-line ceiling; the family's
//! orientation stays in that file's header and the shared toy driver in
//! `common/oracle_probe.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/oracle_probe.rs"]
mod oracle_probe;

use oracle_probe::{all_bf16, bf16_round, model};
use rivoli_oracles::v4oracle::{
    forward::{Capture, Defect, HeadRows, HeadTailW, Oracle},
    weights::{NamedRng, V4Config, WMat},
};

/// `(terms, torch `.sum()`, the running bf16 fold)` — captured from CPU PyTorch, 2026-08-05.
///
/// Shaped like the indexer's real summand, `relu(einsum) * weights_proj(x)`: the `relu_` at
/// model.py:427 applies to the einsum output ONLY, and `weights_proj` is a bare
/// `ColumnParallelLinear` with no activation (model.py:400, :424) scaled by a positive
/// scalar — so the weights are **signed** and the terms can cancel. An earlier version of
/// this fixture assumed non-negative terms; that was wrong, and every conclusion about the
/// error having a systematic direction went with it (see `Oracle::bf16_sum`).
///
/// Two properties the rows are chosen for, because a fixture that merely "differs" proves
/// nothing:
/// - Row 1 is a CONTROL the two semantics agree on, so separation elsewhere is a fact about
///   the semantics and not about the data being uniformly hostile.
/// - The rest separate, at `n = 4` (the toy's `index_n_heads`) and `n = 64` (the model's).
///
/// These are SAMPLED, so their separation is a property of a seed. The case that separates
/// by construction is built in the test body rather than tabulated here — `vanishing_terms`.
type ReductionCase = (&'static [f32], f32, f32);
// **`#[rustfmt::skip]`, and it is what keeps the duplication gate armed over this table.**
//
// These are captured f32 values, one measurement per element. Formatted normally, rustfmt puts
// each on its own line, and several rows carry runs of `0.0` long enough that jscpd calls one
// five-line window a clone of the window one element over. There is nothing to factor — a
// repeated `0.0` here IS the measurement — so the first instinct was a `jscpd:ignore` region.
// That was wrong: an ignore would blanket the whole table INCLUDING every row added later, and
// a genuinely duplicated `ReductionCase` is a real defect class in these registries (the
// sibling compressor suite runs `assert_records_are_well_formed` for exactly that). Packing the
// rows instead removes the five-line windows without touching one measured value, and leaves
// the gate able to see a duplicated ROW -- rows are 7 to 17 lines, still over the window, and
// `bf16_reduction_matches_torch_and_not_a_running_fold` asserts it DIRECTLY as well, so the
// premise does not rest on a text gate that is skipped whenever `npx` is absent.
//
// The reflow was checked by extracting every numeric literal from the table before and after:
// **144**, same order, byte-identical. (Re-running that check gives 144, not the 156 an
// earlier version of this note claimed; 156 counts the twelve numerals inside the `//` row
// comments, which are not literals. The comments are byte-identical too.)
//
// `torch_head_tail`, now in the sibling `v4_oracle_head_tail.rs`, carries the same attribute.
// Its own doc argues nothing about rustfmt or jscpd -- it was authored packed for readability,
// before the interaction `build.rs` records was known -- so it is a precedent for the FORM,
// not a prior statement of this reason. The reason is stated here.
#[rustfmt::skip]
const TORCH_REDUCTIONS: &[ReductionCase] = &[
    // toy `index_n_heads` = 4.0: torch .sum() = -1.4765625, running fold = -1.484375
    (
        &[
            -0.016967773, -0.100097656, -0.49023438, -0.87109375,
        ],
        -1.4765625,
        -1.484375,
    ),
    // CONTROL: the two agree here: torch .sum() = -1.234375, running fold = -1.234375
    (
        &[
            -1.2578125, 0.0, 0.026245117, 0.0,
        ],
        -1.234375,
        -1.234375,
    ),
    // model `index_n_heads` = 64.0: torch .sum() = -2.109375, running fold = -2.09375
    (
        &[
            0.103515625, 0.0, 0.0, 0.26367188, 0.35546875, 0.1796875,
            0.0, -0.359375, 0.0, 0.33203125, 1.9375, 0.0,
            0.0, 0.0, 1.8125, -2.9375, 0.0, 0.0,
            -0.51953125, 0.0, -0.203125, 0.0, 0.06738281, -0.265625,
            0.0, -0.08105469, -0.5078125, 0.0, 0.75390625, 0.0,
            0.0, -0.9921875, 0.0, 0.0, -0.24707031, 0.5,
            -0.26367188, 0.0, -0.111328125, 0.0, 0.0, 0.0,
            -0.048095703, 0.0, 0.0, 0.0, 0.103027344, 0.0,
            0.0, -0.057617188, 0.0, -0.55078125, 0.0, -0.24804688,
            0.0, 0.0, 0.0, -1.1171875, 0.0, 0.0,
            0.0, 0.0, -0.0065307617, 0.0,
        ],
        -2.109375,
        -2.09375,
    ),
    // 64.0 heads, magnitudes spread 32x: torch .sum() = 94.5, running fold = 94.0
    (
        &[
            0.0, 5.90625, 0.0, 0.0, 0.0, 14.1875,
            -10.625, 70.0, 8.625, -4.59375, 0.0, 0.0,
            0.0, 2.65625, -22.5, -3.0625, 0.0, 103.0,
            0.0, 0.0, 8.9375, 18.0, 10.9375, 0.0,
            4.3125, 38.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 5.90625, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -27.625,
            -13.8125, 0.0, 7.65625, 0.0, 0.0, 0.0,
            5.25, -17.875, -0.056396484, -43.25, -33.75, -9.5,
            1.7421875, 7.78125, -2.921875, -9.625, 0.0, -28.625,
            -0.88671875, -3.96875, 0.0, 14.1875,
        ],
        94.5,
        94.0,
    ),
];

#[test]
fn bf16_reduction_matches_torch_and_not_a_running_fold() {
    // **The test that would have caught the 2026-08-05 indexer defect, and the reason it has
    // to be shaped like this.** Every other test in this family is SELF-RELATIVE -- a defected
    // capture against an undefected one -- so an error the oracle shares with its own defect
    // matrix cancels on both sides and is invisible. The running-bf16-fold reduction at
    // model.py:427 was exactly that for the life of the file. Nothing short of an ABSOLUTE
    // comparison against the reference's own semantics can see that class of bug.
    //
    // The expected values are PyTorch's, captured out of tree (see `TORCH_REDUCTIONS`), not
    // recomputed here from a restatement of the oracle.
    //
    // NO TWO ROWS ARE THE SAME CASE. The table's `#[rustfmt::skip]` comment argues that packing
    // it beats a `jscpd:ignore` region precisely because a duplicated `ReductionCase` stays
    // visible -- but jscpd is a text gate that is skipped when `npx` is absent and reports a
    // lower bound on a tree that is not rustfmt-clean. Leaving the premise to it while
    // asserting nothing here is the asymmetry `assert_records_are_well_formed` exists to close
    // in the sibling compressor suite, so it is closed here too. A duplicate row is dead
    // weight that reads as coverage: `separated` counts it twice and the sweep looks broader
    // than it is.
    for (i, (terms, _, _)) in TORCH_REDUCTIONS.iter().enumerate() {
        assert!(
            !TORCH_REDUCTIONS[..i].iter().any(|(t, _, _)| t == terms),
            "TORCH_REDUCTIONS row {i} repeats an earlier row's terms; it adds no case and \
             inflates the separation count below"
        );
    }

    let (cfg, _) = model();
    let good = Oracle::new(cfg.clone(), Defect::None);
    let bad = Oracle::new(cfg.clone(), Defect::IndexerBf16RunningSum);
    let mut separated = 0usize;
    for (i, (terms, torch_sum, torch_fold)) in TORCH_REDUCTIONS.iter().enumerate() {
        let got = good.bf16_sum(terms.iter().copied());
        assert_eq!(
            got.to_bits(),
            torch_sum.to_bits(),
            "row {i}: oracle {got:e} != torch .sum() {torch_sum:e} -- the reduction does not \
             accumulate in f32 and round once"
        );
        let fold = bad.bf16_sum(terms.iter().copied());
        assert_eq!(
            fold.to_bits(),
            torch_fold.to_bits(),
            "row {i}: the running-fold variant is not the fold torch reproduces, so the \
             defect does not model the bug that was actually here"
        );
        separated += usize::from(torch_sum.to_bits() != torch_fold.to_bits());
    }
    // Bidirectional: the data must be able to TELL THEM APART, and must also contain a case
    // where they legitimately agree -- a fixture on which everything differs would prove
    // nothing about resolution. Row 1 is that control.
    assert_eq!(
        separated,
        TORCH_REDUCTIONS.len() - 1,
        "the fixture's separation changed"
    );
    // The rows above separate because the seed found rows that do; reseed them and that could
    // change. This one separates because it cannot do anything else. bf16 keeps 7 explicit
    // mantissa bits, so the ulp at 1.0 is 2^-7 and 63 terms of 2^-10 -- an EIGHTH of an ulp
    // each -- are individually rounded away by a running fold, while together they are worth
    // 7.9 ulps: 1.0 against 1.0625, with no sampling in it. (Verified against this crate's
    // own codec: bf16(1.0 + 2^-8) == 1.0, bf16(1.0 + 2^-7) == 1.0078125, and
    // (1.0625 - 1.0) / 2^-7 == 8 exactly.)
    //
    // 1.0625 is PyTorch's answer for this vector, not arithmetic restated here — captured in
    // the same session as the table above.
    let vanishing_terms: Vec<f32> = std::iter::once(1.0f32)
        .chain(std::iter::repeat_n(2.0f32.powi(-10), 63))
        .collect();
    assert!(
        all_bf16(&vanishing_terms),
        "the construction must be exact in bf16 to mean anything"
    );
    let (kept, rounded_away) = (
        good.bf16_sum(vanishing_terms.iter().copied()),
        bad.bf16_sum(vanishing_terms.iter().copied()),
    );
    assert_eq!(
        kept, 1.0625,
        "f32 accumulation must keep 63 eighth-ulp terms; torch gives 1.0625"
    );
    assert_eq!(
        rounded_away, 1.0,
        "a running fold must round every one of them away"
    );
}

/// `kernels/common.hpp::wave_sum`, in the host: 32 strided partials, then the five-level
/// `__shfl_down` ladder. Not a model of the device -- a transcription of the reduction order
/// every V4 kernel actually uses, which is what makes the floor below a real number rather
/// than a guess about parallelism.
fn wave_sum(x: &[f32]) -> f32 {
    shfl_down_ladder(strided_partials(x))
}

/// A gfx1151 wave is 32 lanes wide, and the reduction order below is a fact about that width,
/// not a tunable.
const WAVE: usize = 32;

/// Phase one: lane `l` accumulates `x[l], x[l+32], x[l+64], ...` in that order, which is the
/// grid-stride loop every V4 kernel opens with.
fn strided_partials(x: &[f32]) -> [f32; WAVE] {
    let mut p = [0.0f32; WAVE];
    for (lane, acc) in p.iter_mut().enumerate() {
        *acc = x.iter().skip(lane).step_by(WAVE).sum();
    }
    p
}

/// Phase two: the five-level `__shfl_down` ladder. `snap` is what makes it a wave rather than
/// a serial fold — every lane reads the PRE-round value.
fn shfl_down_ladder(mut p: [f32; WAVE]) -> f32 {
    let mut off = WAVE / 2;
    while off > 0 {
        let snap = p;
        for lane in 0..WAVE - off {
            p[lane] = snap[lane] + snap[lane + off];
        }
        off >>= 1;
    }
    p[0]
}

/// `golden.rs::Diff.rel`, which is the metric any gate on these goldens will use.
fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (p, q)| m.max((p - q).abs() / scale))
}

/// The three quantities the floor argument turns on, plus the flip count that proves the
/// measurement was not comparing a function to itself.
struct Floor {
    noise_max: f32,
    percopy_min: f32,
    norsqrt_min: f32,
    flips: usize,
}

/// The head-tail weights the floor is measured through. `lm_head` is deliberately all zeros:
/// the measurement reads `final_norm_out`, and a live head would only add its own reduction
/// order to the thing being measured.
fn reassoc_head_weights(cfg: &V4Config) -> HeadTailW {
    let (dim, hcd) = (cfg.dim, cfg.hc_dim());
    let mut w = NamedRng::new("reassoc-floor-weights");
    HeadTailW {
        hc_head_fn: (0..cfg.hc_mult * hcd).map(|_| w.unit() * 0.05).collect(),
        hc_head_base: (0..cfg.hc_mult).map(|_| w.unit()).collect(),
        hc_head_scale: vec![1.0 + w.unit() * 0.5],
        norm: (0..dim).map(|_| bf16_round(1.0 + w.unit() * 0.3)).collect(),
        lm_head: WMat::Dense {
            rows: cfg.vocab_size,
            cols: dim,
            v: vec![0.0; cfg.vocab_size * dim],
        },
    }
}

/// Both quantities vary a lot per draw, so a single sample decides nothing -- the first
/// version of this test drew once, found signal above noise, and would have reported the
/// opposite conclusion. What a gate needs is a THRESHOLD, so the question is whether the two
/// RANGES overlap: worst-case noise against best-case signal.
fn measure_reassociation_floor(cfg: &V4Config, hw: &HeadTailW) -> Floor {
    let (dim, hcd) = (cfg.dim, cfg.hc_dim());
    let mut f = Floor {
        noise_max: 0.0,
        percopy_min: f32::INFINITY,
        norsqrt_min: f32::INFINITY,
        flips: 0,
    };
    for draw in 0..24 {
        let mut r = NamedRng::new(&format!("reassoc-floor-draw-{draw}"));
        let h: Vec<f32> = (0..hcd).map(|_| bf16_round(r.unit())).collect();
        let head = |d: Defect| {
            let mut cap = Capture::default();
            Oracle::new(cfg.clone(), d).head_tail(
                hw,
                HeadRows {
                    h: &h,
                    s: 1,
                    step_tag: "floor",
                },
                &mut cap,
            );
            cap.float("head.floor.final_norm_out")
                .expect("final_norm_out")
                .to_vec()
        };
        let truth = head(Defect::None);
        let percopy = rel_diff(&head(Defect::HeadHcRsqrtPerCopy), &truth);
        f.percopy_min = f.percopy_min.min(percopy);
        f.norsqrt_min = f
            .norsqrt_min
            .min(rel_diff(&head(Defect::HeadHcNoRsqrt), &truth));

        // NOISE: the same final RMSNorm, its 4096-term variance reduced by `wave_sum` instead
        // of sequentially. Both correct; only the order differs.
        let row: Vec<f32> = (0..dim).map(|_| bf16_round(r.unit())).collect();
        let sq: Vec<f32> = row.iter().map(|v| v * v).collect();
        let norm_with = |var: f32| -> Vec<f32> {
            let rs = (var / dim as f32 + cfg.norm_eps).sqrt().recip();
            (0..dim)
                .map(|i| bf16_round(hw.norm[i] * (row[i] * rs)))
                .collect()
        };
        let (sequential, waved) = (norm_with(sq.iter().sum::<f32>()), norm_with(wave_sum(&sq)));
        f.flips += waved
            .iter()
            .zip(&sequential)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        f.noise_max = f.noise_max.max(rel_diff(&waved, &sequential));
    }
    f
}

#[test]
fn the_reassociation_floor_bounds_any_tolerance_these_goldens_can_have() {
    // `forward.rs`'s module doc says the residual disagreement from re-association "is the
    // floor on any tolerance built on these goldens". It has never been QUANTIFIED, and the
    // number turns out to overturn a conclusion this family drew from toy dimensions.
    //
    // At `dim = 4096` the final `RMSNorm` sums 4096 f32 squares. The oracle sums them
    // sequentially; every V4 kernel sums them with `wave_sum`. Both are correct. The
    // measurement below is therefore what a CORRECT kernel costs against this oracle -- the
    // noise any real-dims gate has to clear -- and it is compared against the SIGNAL of a
    // real defect, taken from the oracle itself at the same dimensions.
    //
    // MEASURED 2026-08-05 at dim 4096 / hc_dim 16384. `rel` is max|a-b| / max|b|, and the
    // pairing is worst-case noise against best-case signal, because that is what a fixed
    // threshold has to survive:
    //
    //   noise, correct wave-reduced RMSNorm vs this oracle    3.6e-3 here, 7.1e-3 at 120
    //                                                         out-of-tree draws
    //   signal, HeadHcRsqrtPerCopy                            4.3e-3 here, 2.5e-3 there
    //   signal, HeadHcNoRsqrt                                 3.9e-2 here, 8.0e-3 there
    //
    // **`HeadHcRsqrtPerCopy` and the noise floor are the same order of magnitude, and which
    // is larger depends on the draw.** Here the signal leads by 1.2x; out of tree the noise
    // led by 2x. A quantity that swaps places with the noise between fixtures cannot be
    // gated at real dimensions by any fixed tolerance. `HeadHcNoRsqrt` clears by ~11x here
    // and is genuinely gateable.
    //
    // Two things follow, and the first CORRECTS the sibling head-tail binary. The guidance in
    // `the_head_mhc_rsqrt_is_load_bearing_in_every_case` read "the device-side head gate must
    // be bitwise". That was inferred from toy dimensions and is wrong at real ones: a CORRECT
    // wave-reduced kernel already differs from this oracle on ~0.08% of bf16 elements, so a
    // bitwise gate would reject correct code. It is exactly the extrapolation
    // v4-flash-port.md S3 item 16 warns against, and it was made there before being measured.
    //
    // Second: the mHC denominator's SCOPE cannot be settled by comparing full-width
    // activations at all. It has to be read out of the kernel, or pinned by a small-dim
    // absolute check where the reduction order is controlled -- which is what
    // `the_head_tail_matches_torch_absolutely` is for.
    let dim = 4096usize;
    let cfg = V4Config {
        dim,
        vocab_size: 16,
        ..V4Config::toy()
    };
    let hw = reassoc_head_weights(&cfg);
    let Floor {
        noise_max,
        percopy_min,
        norsqrt_min,
        flips,
    } = measure_reassociation_floor(&cfg, &hw);
    println!(
        "worst noise {noise_max:.3e} ({flips} bf16 flips over 24 draws); \
         best-case signal: PerCopy {percopy_min:.3e}, NoRsqrt {norsqrt_min:.3e}"
    );

    // A correct kernel is NOT bit-identical to this oracle at real dimensions. If that ever
    // stopped being true a bitwise device gate would be back on the table, so it is asserted.
    assert!(
        flips > 0 && noise_max > 1e-4,
        "wave and sequential reduction agreed at dim {dim} ({flips} flips, rel \
         {noise_max:.3e}) -- the re-association floor has vanished, which would change what a \
         device gate can do"
    );

    // A threshold needs MARGIN, not a favourable draw. Both quantities move by about 2x
    // across fixtures -- an out-of-tree run at 120 draws put the noise at 7.09e-3 and this
    // defect's signal at 2.5e-3, the opposite ordering to the one measured here -- so a
    // separation of order 1x is no separation at all. `SEPARABLE` is the margin a real gate
    // would need to survive that variance; it is a judgement, and the numbers it is applied
    // to are printed above so the judgement can be re-examined rather than trusted.
    const SEPARABLE: f32 = 4.0;
    assert!(
        percopy_min < SEPARABLE * noise_max,
        "HeadHcRsqrtPerCopy's weakest signal ({percopy_min:.3e}) now clears the worst \
         re-association noise ({noise_max:.3e}) by more than {SEPARABLE}x, so a real-dims \
         threshold could resolve it after all. Good news -- re-measure and rewrite the \
         guidance above rather than moving this assert"
    );
    // Not vacuous: another defect DOES clear it comfortably, so the bound measures this
    // defect's weakness and not a fixture too feeble to move anything.
    assert!(
        norsqrt_min > SEPARABLE * noise_max,
        "neither rsqrt defect clears the floor by {SEPARABLE}x (NoRsqrt {norsqrt_min:.3e} vs \
         noise {noise_max:.3e}), so this test cannot tell a real floor from a dead fixture"
    );
}
