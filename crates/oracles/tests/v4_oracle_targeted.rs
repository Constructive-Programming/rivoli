//! **The defects the grid cannot carry: magnitude-gated, toy-blind, or silent-half-less.**
//!
//! A matrix row asserts a defect BOTH ways in every reachable case. Some defects cannot make
//! that claim honestly — the clamp and the softplus threshold only bite above an activation
//! magnitude the toy never reaches; the split-k fold needs `k >= 4096` and the toy's largest
//! K is 256; `HcPreNoRsqrt` reaches every golden downstream of `hc_pre`, which is all of them,
//! so it has no silent half to declare. `expect()` returns `None` for each, and
//! `every_defect_carries_both_halves_of_its_claim` (sibling `v4_oracle.rs`) demands that a
//! `None` be paid for by a targeted test. These are those tests, and the payment is a DRIVEN
//! input: the loud half at a magnitude or dimension that reaches the defect, and the silent
//! half at one that provably does not.
//!
//! Split out of `v4_oracle.rs` on 2026-08-15 for the 800-line ceiling; the family's
//! orientation stays in that file's header and the shared toy driver in
//! `common/oracle_probe.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_oracles::golden::{diff, identical};
use rivoli_oracles::v4oracle::forward::{Capture, Defect, LayerCtx, Oracle, splitk_fold};
use rivoli_oracles::v4oracle::toy::ToyModel;
use rivoli_oracles::v4oracle::weights::{NamedRng, V4Config, WMat};

#[path = "common/oracle_probe.rs"]
mod oracle_probe;
use oracle_probe::{Run, bf16_round, fixed_ids, model, residual_probe, run};

/// Run one routed expert at a given input scale, returning `(output, clamped_count)`.
fn expert_at_scale(defect: Defect, scale: f32) -> (Vec<f32>, usize) {
    let (cfg, m) = model();
    let o = Oracle::new(cfg.clone(), defect);
    let mut r = NamedRng::new("swiglu-probe");
    let x: Vec<f32> = (0..cfg.dim).map(|_| bf16_round(r.unit() * scale)).collect();
    let mut counters = Default::default();
    let y = o.expert(&m.layers[0].experts[&0], &x, 1, None, &mut counters);
    (y, counters.swiglu_clamp_events)
}

#[test]
fn the_selection_golden_moves_when_topk_truncates() {
    // `.compress_idxs` exists because a wrong Hadamard basis or a wrong indexer score
    // changes WHICH blocks are attended while leaving every magnitude plausible -- something
    // no numeric tolerance can see. But it only carries information where `index_topk`
    // actually cuts: with k == n_compressed the selected SET is every block, invariant under
    // any scoring bug, and the golden is vacuous. That is the trap MEMORY.md records as
    // "a dsa A/B under 2048 tokens covers nothing".
    //
    // Both halves. Truncation is NECESSARY for the set to move -- without it the set is
    // every compressed block, and no scoring bug can change that. It is not SUFFICIENT: a
    // ranking can survive a defect. So the assertion is "some indexer defect moves the set
    // iff the top-k truncates", which is exactly as strong as the arithmetic allows.
    //
    // As a SET, not positionally: `topk_idx` returns descending-score order, so a scoring
    // change permutes the list even when it selects the same blocks. The set is what the
    // attention consumes and what S2 must compare.
    let selected = |r: &Run| {
        let mut v: Vec<i64> = [&r.pre, &r.dec]
            .iter()
            .flat_map(|c| c.ints.iter())
            .filter(|(n, _, _)| n.ends_with(".compress_idxs"))
            .flat_map(|(_, _, x)| x.iter().copied())
            .collect();
        v.sort_unstable();
        v
    };
    for (prompt, want_truncation) in [(5usize, false), (12, true)] {
        let base = run(2, prompt, Defect::None);
        let cut = base.pre.counters.indexer_truncated + base.dec.counters.indexer_truncated;
        assert_eq!(
            cut > 0,
            want_truncation,
            "prompt {prompt}: {cut} truncating query rows"
        );
        let want = selected(&base);
        assert!(
            !want.is_empty(),
            "prompt {prompt}: no selection golden at all"
        );
        let mut movers = Vec::new();
        for d in [
            Defect::IndexerNoHadamard,
            Defect::IndexerNoRelu,
            Defect::IndexerNoWeights,
        ] {
            if selected(&run(2, prompt, d)) != want {
                movers.push(d);
            }
        }
        assert_eq!(
            !movers.is_empty(),
            want_truncation,
            "prompt {prompt}: indexer defects that moved the selected SET = {movers:?}, but \
             the top-k truncated {cut} times. Without truncation NONE may move it; with \
             truncation the selection golden is worthless if none does."
        );
    }
}

#[test]
fn qk_norm_order_is_a_rounding_difference_not_an_arithmetic_one() {
    // `Defect::QkNormAfterRope` is mathematically INERT: `apply_rotary_emb` rotates adjacent
    // pairs, so it preserves `q.square().mean(-1)`, and a scalar scale commutes with a
    // rotation. Whatever the goldens show is bf16 rounding landing in a different place.
    //
    // Measured rather than argued: its relative move on `.q` must be no larger than what
    // dropping bf16 rounding altogether costs. If that ever stops holding, the two orders
    // are not equivalent after all and this belongs back in the matrix.
    let base = run(0, 12, Defect::None);
    let worst = |d: Defect| {
        diff(&base.pre, &run(0, 12, d).pre)
            .into_iter()
            .filter(|x| x.name.ends_with(".q"))
            .fold(0.0f32, |m, x| m.max(x.rel))
    };
    let (order, rounding) = (
        worst(Defect::QkNormAfterRope),
        worst(Defect::NoBf16Rounding),
    );
    assert!(
        rounding > 0.0,
        "NoBf16Rounding moved nothing, so there is no yardstick"
    );
    assert!(
        order <= rounding,
        "QK-norm order moved .q by {order:.3e}, more than dropping bf16 entirely \
         ({rounding:.3e}) -- it is not a pure rounding difference"
    );
}

#[test]
fn hc_pre_rsqrt_and_bf16_rounding_reach_the_whole_block() {
    // The two defects with no silent half. They are still real breakages and still must be
    // caught; what they cannot supply is a golden they leave alone, so they are asserted
    // here as "reaches everything from `attn_norm_out` onwards" rather than pretended into
    // the matrix on the strength of `.in`, which the driver fixes by construction.
    let base = run(2, 12, Defect::None);
    for d in [Defect::HcPreNoRsqrt, Defect::NoBf16Rounding] {
        let got = run(2, 12, d);
        let ds = diff(&base.pre, &got.pre);
        for suffix in [".attn_norm_out", ".q", ".attn_out", ".ffn_norm_out", ".out"] {
            assert!(
                ds.iter()
                    .filter(|x| x.name.ends_with(suffix))
                    .any(|x| x.changed > 0),
                "{d:?} left *{suffix} untouched"
            );
        }
        assert!(
            ds.iter()
                .filter(|x| x.name.ends_with(".in"))
                .all(|x| x.changed == 0),
            "{d:?} moved the driver-supplied input, which is impossible"
        );
    }
}

#[test]
fn sinkhorn_has_converged_long_before_iteration_20() {
    // **A SECOND LIMITATION, asserted rather than assumed.**
    //
    // `hc_sinkhorn_iters = 20` on a 4x4 positive matrix is far past convergence: the row and
    // column normalisations reach a fixed point at f32 precision after a handful of passes,
    // so iteration 20 changes nothing iteration 19 did not already give. This oracle
    // therefore CANNOT tell 19 iterations from 20, and neither can any golden built on it.
    //
    // What it can see is gross truncation, which is the failure that actually matters: a
    // port that ran two passes would be caught. Both halves below.
    //
    // > **CORRECTED 2026-08-07. The paragraph above is true of THIS FIXTURE and false of the
    // > checkpoint.** It was written as a claim about the algorithm ("a 4x4 positive matrix
    // > is far past convergence") and read that way ever since, including by the doc on
    // > `Defect::SinkhornIterCountProbe`, which said the variant changes nothing at the
    // > shipped count. `v4-oracle defects --layer 0 --decode-steps 1` on the real weights
    // > disagrees: 19 vs 20 moves **39,893/53,248** of `L0.pre.ffn_norm_out`, **all 78**
    // > router weights, 50,812/53,248 of `ffn_out` and 143,026/212,992 of `out`. Convergence
    // > is to within f32 rounding, and whether the last ulp settles is weight-dependent; the
    // > toy's mixes settle and the checkpoint's do not, after which `hc_post` and the MoE
    // > spread that difference across most of the block.
    // >
    // > The sweep reports differing-element COUNTS, not magnitudes, so this establishes
    // > non-identity on the real model and says nothing about size. The error came from
    // > generalising one fixture's bit-identity into a statement about the arithmetic —
    // > exactly the "most-trusted case is the blind spot" failure. The assertion below is
    // > still correct as a statement about the fixture, and is what it now claims to be.
    let (cfg, m) = model();
    let drive = |c: &V4Config, d: Defect| drive_layer0(c, m, d);
    let full = drive(cfg, Defect::None);
    assert!(
        identical(&full, &drive(cfg, Defect::SinkhornIterCountProbe)),
        "19 and 20 iterations disagree ON THE FIXTURE -- this oracle's blindness to the \
         cut is the whole claim here, and it has stopped holding"
    );
    let mut two = cfg.clone();
    two.hc_sinkhorn_iters = 2;
    assert!(
        !identical(&full, &drive(&two, Defect::None)),
        "the gate cannot even see the Sinkhorn cut from 20 passes to 2"
    );
}

/// The three structural claims behind excluding `SplitKFoldOrder` from the matrix, each the
/// half a wrong exclusion would hide (docs/investigations/v4-decode-decomposition.md §M9):
///
/// 1. **Toy-blind**: the dispatch predicate needs `k >= 4096` and the toy's largest K is
///    `dim = 256`, so no toy GEMV selects the fold and the capture is bit-identical to
///    `None` — the `SinkhornIterCountProbe` shape of exclusion, asserted so it goes red the
///    day a toy dimension grows past the predicate.
/// 2. **Partition-exact**: over integer-valued f32s (addition exact in any order below
///    2^24) the fold equals the ascending serial sum EXACTLY. A dropped, duplicated or
///    misassigned quad is an integer-sized error here, not rounding — this is the test that
///    the partition covers every column exactly once.
/// 3. **Live at real dims**: on smooth data at k = 4096 the fold differs from the serial
///    sum (else §M9's host measurement would be comparing a function to itself), and by a
///    relative amount in f32-reassociation territory, not a magnitude change.
/// 4. **Wired**: `Oracle::linear` reaches the fold exactly when `splitk_selects` says so —
///    positive at the wkv shape, negative at the m, k and format exclusions (see the
///    section's own comment for why a vacuously-true predicate is the hazard).
#[test]
fn the_splitk_fold_is_toy_blind_partition_exact_and_nonzero_at_real_dims() {
    assert_toy_cannot_select_the_fold();
    assert_partition_is_exact_and_the_fold_is_live_at_real_dims();
    assert_linear_reaches_the_fold_exactly_where_the_predicate_says();
}

/// Claim 1. Bit-identical to `None` on the toy, so the exclusion goes red the day a toy
/// dimension grows past the predicate.

/// Drive layer 0 of the toy under `d` and capture the prefill — the shared probe both
/// blindness assertions score (their closures became a jscpd clone once the ProbeLayer
/// literal went multi-line, 2026-08-15).
fn drive_layer0(cfg: &V4Config, m: &ToyModel, d: Defect) -> Capture {
    let o = Oracle::new(cfg.clone(), d);
    let mut h = residual_probe(cfg, "h-pre", 5);
    let ids = fixed_ids(cfg, "ids-pre", 5);
    oracle_probe::prefill_capture(
        &o,
        oracle_probe::ProbeLayer {
            idx: 0,
            w: &m.layers[0],
        },
        &ids,
        &mut h,
    )
}

fn assert_toy_cannot_select_the_fold() {
    let (cfg, m) = model();
    let drive = |d: Defect| drive_layer0(cfg, m, d);
    assert!(
        identical(&drive(Defect::None), &drive(Defect::SplitKFoldOrder)),
        "the toy can see the split-k fold: the matrix-exclusion argument is dead and this \
         defect needs a real Expect row"
    );
}

/// The ascending serial dot the fold is scored against — the reference order, spelled once so
/// both probes below are scored the same way.
fn serial_dot(x: &[f32], w: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (a, b) in x.iter().zip(w) {
        acc += a * b;
    }
    acc
}

/// Claims 2 and 3, from ONE seed in ONE order because both probes draw off the same
/// `NamedRng` and splitting them would re-base the second.
fn assert_partition_is_exact_and_the_fold_is_live_at_real_dims() {
    let k = 4096;
    let mut r = NamedRng::new("splitk-fold-probe");
    let xi: Vec<f32> = (0..k).map(|_| (r.unit() * 9.0).round()).collect();
    let wi: Vec<f32> = (0..k).map(|_| (r.unit() * 9.0).round()).collect();
    assert_eq!(
        splitk_fold(&xi, &wi).to_bits(),
        serial_dot(&xi, &wi).to_bits(),
        "integer probe: the split-k partition dropped or double-counted a column"
    );

    let x: Vec<f32> = (0..k).map(|_| r.unit() * 0.05).collect();
    let w: Vec<f32> = (0..k).map(|_| r.unit() * 0.05).collect();
    let (s, p) = (serial_dot(&x, &w), splitk_fold(&x, &w));
    assert_ne!(
        s.to_bits(),
        p.to_bits(),
        "smooth probe: the two folds agree bitwise at k = 4096 — the A/B would compare a \
         function to itself"
    );
    // Reassociating a 4096-term f32 sum moves it by rounding, not by magnitude. The bound
    // is generous on purpose (the probe's terms cancel toward zero, inflating RELATIVE
    // error); what it must catch is a fold that computes a different SUM, not worse
    // rounding.
    let scale = x
        .iter()
        .zip(&w)
        .map(|(a, b)| (a * b).abs())
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    assert!(
        ((s - p) / scale).abs() < 1e-5,
        "smooth probe: |serial - splitk| = {} against a term-magnitude sum of {scale} — \
         this is not reassociation noise",
        (s - p).abs()
    );
}

/// An fp8 weight of `rows x cols` with one 2^0 scale per 128x128 tile.
fn fp8_tiled(rows: usize, cols: usize, w: &[u8]) -> WMat {
    WMat::Fp8 {
        rows,
        cols,
        w: w.to_vec(),
        s: vec![127u8; rows.div_ceil(128) * cols.div_ceil(128)],
    }
}

/// Claim 4, the WIRING — `linear` must actually reach the fold when `splitk_selects` says so.
///
/// Review 2026-08-08: the claims above bypass the predicate entirely (the toy drive selects
/// nothing by design; the probes call `splitk_fold` directly), so a predicate typo that made
/// it never-true would reproduce §M9's "all tensors bit-identical" host measurement VACUOUSLY.
/// `linear` returns UNROUNDED values, so the raw fold delta (~59% of sums on real weights) is
/// observable here even though every golden downstream absorbs it in a bf16 store. wkv's
/// [512x4096] is the smallest captured shape; the negatives pin the m, k and format terms.
fn assert_linear_reaches_the_fold_exactly_where_the_predicate_says() {
    let (cfg, _) = model();
    let (rows, kk) = (512usize, 4096usize);
    let mut rw = NamedRng::new("splitk-wiring");
    let wb: Vec<u8> = (0..rows * kk)
        .map(|_| {
            loop {
                let b = rw.below(256) as u8;
                // The two e4m3 NaN codes poison a sum into NaN != NaN noise.
                if !matches!(b, 0x7f | 0xff) {
                    break b;
                }
            }
        })
        .collect();
    let wired = fp8_tiled(rows, kk, &wb);
    let xw: Vec<f32> = (0..13 * kk).map(|_| rw.unit() * 0.05).collect();
    let out = |d: Defect, m: usize, w: &WMat, k: usize| {
        Oracle::new(cfg.clone(), d).linear(&xw[..m * k], m, k, w)
    };
    let differs = |a: Vec<f32>, b: Vec<f32>| -> bool {
        a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits())
    };
    assert!(
        differs(
            out(Defect::None, 1, &wired, kk),
            out(Defect::SplitKFoldOrder, 1, &wired, kk)
        ),
        "the predicate no longer reaches the fold through `linear` at the wkv shape — \
         every downstream null host result is vacuous until this is red-to-green again"
    );
    let dense = WMat::Dense {
        rows,
        cols: kk,
        v: wb[..rows * kk].iter().map(|&b| b as f32 * 0.01).collect(),
    };
    for (name, m, w, k) in [
        ("m = 13 (recorded-prompt prefill)", 13usize, &wired, kk),
        (
            "k = 2048 (below the k bound)",
            1,
            &fp8_tiled(rows, 2048, &wb[..rows * 2048]),
            2048,
        ),
        // The format term is what keeps the compressor/gate/wo_a out.
        ("a Dense weight at a captured shape", 1, &dense, kk),
    ] {
        assert!(
            !differs(
                out(Defect::None, m, w, k),
                out(Defect::SplitKFoldOrder, m, w, k)
            ),
            "{name} must NOT select the split fold"
        );
    }
}

#[test]
fn swiglu_clamp_fires_only_above_its_limit() {
    // The bidirectional pair the defect matrix cannot supply: the clamp is magnitude-gated,
    // so at ordinary activation scales `swiglu_limit = 10` and rivoli's unclamped SwiGLU are
    // the SAME function, and only a driven input separates them. A test that only showed the
    // difference at large scale would not establish that the oracle is otherwise faithful.
    for d in [Defect::SwigluUnclamped, Defect::SwigluClampGateBothSides] {
        let (cold_ref, n_cold) = expert_at_scale(Defect::None, 0.3);
        let (cold_def, _) = expert_at_scale(d, 0.3);
        assert_eq!(
            n_cold, 0,
            "the probe was supposed to stay inside +/-10 ({d:?})"
        );
        assert_eq!(
            cold_ref, cold_def,
            "{d:?} moved an expert whose activations never clamp"
        );

        let (hot_ref, n_hot) = expert_at_scale(Defect::None, 300.0);
        let (hot_def, _) = expert_at_scale(d, 300.0);
        assert!(n_hot > 0, "the hot probe never reached the clamp ({d:?})");
        assert_ne!(hot_ref, hot_def, "{d:?} left a clamped expert unchanged");
    }
    // The two hot runs above already establish the asymmetry: `SwigluClampGateBothSides`
    // differs from the reference ONLY by clamping the gate from below, so its hot-vs-hot
    // disagreement IS the evidence that the reference does not. Restating it here as a
    // separate "asymmetry check" would be the same comparison wearing a second name.
}

#[test]
fn softplus_threshold_only_matters_for_large_router_logits() {
    // Same shape of argument at the router. The toy's own gate never reaches logit 20, so
    // the threshold is invisible there -- which is the silent half -- and a gate weight
    // scaled past it is the loud half.
    let (cfg, m) = model();
    let layer = 3; // score-routed
    let ids = fixed_ids(cfg, "ids-pre-5", 5);
    let x: Vec<f32> = residual_probe(cfg, "gate-x", 5)
        .into_iter()
        .take(5 * cfg.dim)
        .collect();

    for (scale, want_differ) in [(1.0f32, false), (400.0, true)] {
        // Swapping ONLY the gate weight keeps everything else identical, so any difference
        // is attributable to the softplus branch and nothing else.
        let mut layer_w = m.layers[layer].clone();
        if let WMat::Dense { v, .. } = &mut layer_w.gate_w {
            for e in v.iter_mut() {
                *e *= scale;
            }
        }
        let mut got = Vec::new();
        for d in [Defect::None, Defect::RouterNoSoftplusThreshold] {
            let o = Oracle::new(cfg.clone(), d);
            let step = LayerCtx {
                lw: &layer_w,
                layer,
                s: 5,
                start_pos: 0,
                input_ids: &ids,
                step_tag: "g",
            };
            let mut counters = Default::default();
            got.push((o.gate(&step, &x, &mut counters).0, counters));
        }
        let hits = got[0].1.softplus_overflows;
        // The counter is "logits where ln(1+e^x) OVERFLOWS", not "logits above 20": for
        // 20 < x < ~88 the two forms are bit-identical in f32, so a counter keyed to 20
        // would report the defect as reachable in a range where it provably is not.
        assert_eq!(
            hits > 0,
            want_differ,
            "at gate scale {scale} softplus overflowed {hits} times, expected {}",
            if want_differ { "some" } else { "none" }
        );
        assert_eq!(
            got[0].0 != got[1].0,
            want_differ,
            "at gate scale {scale}: threshold difference = {}, expected {want_differ}",
            got[0].0 != got[1].0
        );
    }
}
