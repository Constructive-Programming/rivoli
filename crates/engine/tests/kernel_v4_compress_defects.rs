//! **Whether the compressor comparison can REJECT** — the separation sweep over every breakage
//! the attention compressor could carry, and the coverage registry for the ones it provably
//! cannot see.
//!
//! The positive gate and the two exact impersonations are `kernel_v4_compress.rs`; the harness
//! both drive is `v4_compressor/mod.rs`. This file is the weaker but broader half, and being
//! plain about how much weaker is the point of the header below.
//!
//! Ported from `old:tests/kvcompress_kernel.rs`'s techniques 2, 3 and 4.
//!
//! # Three techniques, in descending order of strength
//!
//! 2. **Distance separation** for the breakages that live INSIDE the kernel and cannot be reached
//!    by perturbing an input (the RoPE pairing, the block-end position, the bf16 stores, the
//!    `act_quant` extent). For each, the distance from the GPU output to the defect-injected
//!    oracle must dwarf the distance to the clean one. **This proves the METRIC has resolution,
//!    not that this kernel would fail if broken in that specific way.** Technique 1 — exact
//!    impersonation, next door — is the strong half, and it reaches exactly two defects.
//! 3. **Named inertness.** A defect that cannot fire on a cell is PRINTED as inert and skipped, so
//!    "the kernel matched" there is recorded as proving nothing rather than counted as coverage.
//!    Note what this is NOT: the sweep does not *assert* inertness, so a defect that silently
//!    becomes inert on an unrecorded cell still passes with a printed line. Exactly one pair is
//!    genuinely asserted, by [`the_overlap_defect_is_inert_at_ratio_128_and_live_at_ratio_4`].
//! 4. **Recorded non-coverage, as an EXPECTED VALUE.** Where the metric provably cannot resolve a
//!    defect, the cell is listed in [`BELOW_RESOLUTION`] with its measured separation and asserted
//!    to reproduce that number exactly. A cell that GAINS resolution fires; an entry that stops
//!    being reached fires. That replaced a bare `sep >= RESOLVABLE` which had left the reference
//!    suite RED from its merge onward, because the decision not to require those cells lived only
//!    in a document.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no `rocm` CI arm, and no checkpoint or GPU for this port. Two mutations, and
//! the second is the one that matters:
//!
//! * In `kernels/kvcompress.hip`'s pooling, drop the `overlap_transform` term (make the
//!   overlapping branch behave like the non-overlapping one).
//!   [`the_overlap_defect_is_inert_at_ratio_128_and_live_at_ratio_4`] must go RED on its ratio-4
//!   arm — the kernel now IS the no-overlap kernel, so `sep` collapses toward the quantization
//!   floor — while its ratio-128 arm stays green, because at ratio 128 the defect had no term to
//!   disable. Both halves matter: a mutation that reddens the 128 arm too has broken the
//!   non-overlapping path, which is a different bug.
//! * Delete one row of [`BELOW_RESOLUTION`] and re-run
//!   [`each_in_scope_defect_is_further_from_the_gpu_than_the_clean_oracle_is`]. It MUST go red
//!   with "NOT RECORDED", and adding it back MUST make it green again. That is the proof the
//!   registry is load-bearing rather than decorative, and it needs no kernel change — run it
//!   first, because it is the cheapest thing here that can distinguish a working gate from a
//!   green one. Then change a recorded number by one and confirm "the record is stale" fires:
//!   the two failures have different messages on purpose.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;
mod v4_compressor;

use v4_compressor::{
    Cell, Defect, PROBE_LEN, PROBE_REMAINDER_LEN, RESOLVABLE, Run, Widths, assert_clean, cells,
    diff, gap, load_and_baseline,
};

/// **Cells the metric provably cannot resolve, with the separation MEASURED on hardware.**
///
/// `RESOLVABLE` is NOT lowered to admit `sep = 8`: that is the budget-not-measurement move, and
/// the non-coverage is encoded here — at the assertion, with its argument in place — instead.
///
/// **The scale-invariance argument this rests on is not universal, and this registry is the
/// evidence.** `KvActQuantBlock128` is inert "for a reason no threshold can fix" — ue8m0 scales
/// are powers of two and e4m3 is exactly scale-invariant under them. That holds at three of the
/// four cells, where the defect is INERT and the sweep prints it as covering nothing. At
/// `ratio4/prefill` it is **live**: `sep=16`, one e4m3 step, on 6 of 32768 elements,
/// `want=3.5 got=3.25` — adjacent codes in the `[2,4)` binade. Scale-invariance is exact only
/// while both scales keep every value inside the format's range; at a rounding boundary the two
/// blockings disagree. An entry can only appear here if it was REACHED, and reaching requires
/// `broken != clean`, so the presence of that row is itself the disproof of "at any threshold".
///
/// **An EXPECTED VALUE, not a skip.** Each entry is asserted to reproduce its recorded separation
/// exactly, so a cell that stops being unresolvable still fires; every entry must be reached, so a
/// stale one cannot silently swallow a case that no longer occurs; and a cell ABSENT from this
/// list must separate. An exclusion list that quietly absorbed a future regression would be the
/// same class of defect one level up — a guard that cannot fire.
///
/// **The list is broader than a first tabulation recorded**, which covered three
/// `act_quant`-argument defects at one cell. Measured across all four cells it is 13 entries. Two
/// that no such tabulation names: `KvActQuantWholeTensor` (29 and 38), and `NoBf16Rounding` on
/// both ratio-128 cells at `sep=16` — exactly one e4m3 step, i.e. the bf16 stores move the
/// ratio-128 output by less than the quantizer's own grain, and that one is not an `act_quant`
/// argument at all.
const BELOW_RESOLUTION: &[(&str, Defect, u32)] = &[
    ("ratio4/prefill", Defect::SkipKvActQuant, 14),
    ("ratio4/prefill", Defect::KvActQuantWholeTensor, 29),
    ("ratio4/prefill", Defect::KvActQuantBlock128, 16),
    ("ratio4/prefill", Defect::KvActQuantNoRoundScale, 23),
    ("ratio4/decode", Defect::SkipKvActQuant, 8),
    ("ratio4/decode", Defect::KvActQuantNoRoundScale, 22),
    ("ratio128/prefill+remainder", Defect::SkipKvActQuant, 8),
    (
        "ratio128/prefill+remainder",
        Defect::KvActQuantWholeTensor,
        38,
    ),
    (
        "ratio128/prefill+remainder",
        Defect::KvActQuantNoRoundScale,
        17,
    ),
    ("ratio128/prefill+remainder", Defect::NoBf16Rounding, 16),
    ("ratio128/decode", Defect::SkipKvActQuant, 8),
    ("ratio128/decode", Defect::KvActQuantNoRoundScale, 18),
    ("ratio128/decode", Defect::NoBf16Rounding, 16),
];

/// Every recorded value sits BELOW the floor, and no pair is recorded twice.
///
/// Without the first check the sweep asserts only `sep == want`, so an entry of 31215 would pass
/// — and the failure message would still print "(inside the quantization floor)", a false claim
/// emitted by the assertion itself. That is the exclusion list absorbing a SEPARATING cell, which
/// is the exact failure this registry is documented to prevent. It was found by review in the
/// reference tree and the arm was untested there, because no deliberate break reached it.
///
/// Duplicates make the second entry permanently unreachable, because both lookups take the FIRST
/// match — a guard that cannot fire, reported as a dead entry pointing at the wrong row.
fn assert_records_are_well_formed() {
    for (c, d, s) in BELOW_RESOLUTION {
        assert!(
            *s < RESOLVABLE,
            "BELOW_RESOLUTION {c}/{d:?} records {s} >= {RESOLVABLE} — a separating cell must not \
             be recorded as non-coverage"
        );
    }
    for (i, (c, d, _)) in BELOW_RESOLUTION.iter().enumerate() {
        assert!(
            !BELOW_RESOLUTION[..i]
                .iter()
                .any(|(c2, d2, _)| c2 == c && d2 == d),
            "BELOW_RESOLUTION has a duplicate {c}/{d:?}; the second can never be reached"
        );
    }
}

/// Does this breakage live anywhere the ATTENTION compressor touches?
///
/// Exhaustive and wildcard-free on purpose. `Defect` is a domain enum this repo owns, so a variant
/// added by a later stage must come back here and be classified rather than defaulting to "not
/// our problem" — which is how a real compressor defect ends up outside every list that claims to
/// cover the compressor.
fn in_compressor_scope(d: Defect) -> bool {
    match d {
        // The compressor's own three, the RoPE inside the finish, the four `act_quant` arguments
        // and the bf16 stores.
        Defect::CompressorNoOverlap
        | Defect::CompressorNoApe
        | Defect::CompressorRopeAtBlockEnd
        | Defect::RopeAllDims
        | Defect::RopeFirstDims
        | Defect::RopeHalfSplit
        | Defect::RopeNoYarn
        | Defect::SkipKvActQuant
        | Defect::KvActQuantWholeTensor
        | Defect::KvActQuantBlock128
        | Defect::KvActQuantNoRoundScale
        // In scope BECAUSE of the `if compressed` guard on its `Oracle::freqs` arm: base-theta
        // YaRN swaps the table on exactly the layers this sweep drives. Its twin
        // `RopeYarnEverywhere` guards `if !compressed` and is the excluded one below — the
        // first classification lumped the two as a pair and excluded both, with a stated
        // reason ("keys off a ratio-0 layer") that is true only of the twin (review
        // 2026-08-16). An excluded defect is simply never run, so the sweep itself could not
        // catch the misclassification.
        | Defect::RopeBaseThetaEverywhere
        | Defect::NoBf16Rounding => true,
        // `None` is the baseline, not a breakage. `RopeYarnEverywhere` keys off a ratio-0
        // layer (`if !compressed`), which by construction has no compressor at all.
        // Everything below belongs to the attention core, the router and MoE, the indexer,
        // or the head tail.
        //
        // `IndexerBf16RunningSum` is the indexer's per-head score reduction, and the indexer has
        // its OWN compressor — distinct instance, distinct algorithm (fp4 + Hadamard, not partial
        // fp8). It cannot reach the attention compressor. The six `Head*` variants live strictly
        // after the last block, downstream of everything here.
        Defect::None
        | Defect::RopeYarnEverywhere
        | Defect::SkipQkNorm
        | Defect::QkNormUsesQNormWeight
        | Defect::QkNormAfterRope
        | Defect::SkipAttnSink
        | Defect::AttnSinkNotMaxShifted
        | Defect::PrefillRingWritesFirstWindow
        | Defect::SkipOutputDerotation
        | Defect::OutputDerotationForward
        | Defect::WoGroupsSplitHeadDim
        | Defect::WoGroupsInterleaved
        | Defect::IndexerNoRelu
        | Defect::IndexerNoFp4Quant
        | Defect::IndexerNoHadamard
        | Defect::IndexerNoWeights
        | Defect::IndexerBf16RunningSum
        | Defect::SwigluUnclamped
        | Defect::SwigluClampGateBothSides
        | Defect::RouterSoftmax
        | Defect::RouterNoSoftplusThreshold
        | Defect::RouterBiasedWeights
        | Defect::RouterNoRenorm
        | Defect::RouterNoScale
        | Defect::HashRoutingIgnored
        | Defect::RouteWeightAfterW2
        | Defect::SharedExpertWeighted
        | Defect::Fp4NibbleSwap
        | Defect::SinkhornIterCountProbe
        | Defect::SinkhornCombTransposed
        | Defect::HcPostNoComb
        | Defect::HcPreNoRsqrt
        | Defect::HeadHcNoRsqrt
        | Defect::HeadHcRsqrtPerCopy
        | Defect::HeadNormSkipped
        | Defect::HeadNormNotBf16
        | Defect::HeadNormOverAllTokens
        | Defect::HeadLogitsFromFirstRow => false,
        // The split-k fold applies only to fp8-quantized GEMVs, and both compressor projections
        // are `Linear(..., dtype=torch.float32)` — Dense in the oracle, so the predicate cannot
        // select them at any shape.
        Defect::SplitKFoldOrder => false,
    }
}

/// One cell's CLEAN pair, carried into the per-defect scoring so the loop body is a function.
///
/// A struct because `clean` and `gpu` are both `&[f32]` of the same length and swapping them is
/// silent: every distance below would then be measured from the KERNEL rather than from the
/// oracle, which is the gate grading itself.
#[derive(Clone, Copy)]
struct Baseline<'a> {
    name: &'static str,
    clean: &'a [f32],
    gpu: &'a [f32],
}

/// The metric this sweep scores with, and the registry's reached-flags it ticks.
///
/// Bundled because they travel together and neither is an input to the arithmetic: `w` decides
/// which bucket each element lands in and `hit` is the anti-vacuity ledger. Splitting them across
/// two parameters is what pushed the loop body over CodeScene's argument rule when it was
/// extracted; carrying them as one value is what the extraction is worth.
struct Reach<'a> {
    w: Widths,
    hit: &'a mut [bool],
}

/// Score ONE (cell, defect) pair: report inertness, or measure the separation and hold it to the
/// registry. Returns the complaints rather than asserting, so the caller measures every pair
/// before it aborts — the same argument the clean sweep makes one level up.
///
/// **Extracted from the sweep 2026-08-16** under the CodeScene gate (`cc = 12`, `LoC = 73`, two
/// nesting bumps in one body). The cut is at the natural seam: the outer loop owns the CELLS and
/// this owns the VERDICT for one defect on one of them.
fn score_one(base: Baseline<'_>, def: Defect, broken: &[f32], r: Reach<'_>) -> Vec<String> {
    let name = base.name;
    if broken == base.clean {
        // INERT here, by construction. Printed rather than skipped silently: the point of naming
        // it is that this cell must not be counted as covering it.
        println!("{name}: {def:?} is INERT here — this cell covers it not at all");
        return Vec::new();
    }
    let sep = gap(&format!("{name} {def:?}"), broken, base.gpu, r.w);
    let known = BELOW_RESOLUTION
        .iter()
        .position(|(c, d, _)| *c == name && *d == def);
    if let Some(i) = known {
        r.hit[i] = true;
    }
    match known.map(|i| BELOW_RESOLUTION[i].2) {
        // Recorded non-coverage: must reproduce its measured separation EXACTLY. A cell that
        // gained resolution fails here rather than quietly passing, which is the whole difference
        // between an expected value and a skip.
        Some(want) if sep != want => vec![format!(
            "{name}/{def:?} sep={sep}, RECORDED {want} — the record is stale"
        )],
        // Not recorded, and the metric cannot see it.
        None if sep < RESOLVABLE => vec![format!(
            "{name}/{def:?} sep={sep} < {RESOLVABLE}, NOT RECORDED"
        )],
        Some(_) | None => Vec::new(),
    }
}

/// Every remaining in-scope breakage is measurably further from the GPU than the clean oracle is
/// — or is reported INERT on that cell and therefore claimed as coverage of nothing.
#[test]
fn each_in_scope_defect_is_further_from_the_gpu_than_the_clean_oracle_is() {
    let Some((ck, c, list)) = cells() else {
        return;
    };
    assert_records_are_well_formed();
    // Derived by EXHAUSTIVE match over `Defect::ALL` rather than spelled as a list. A list
    // silently omits any variant added later; the match makes one a compile error, which is the
    // same argument this repo makes about wildcards on domain enums — and the moment a new
    // breakage is added is exactly when someone must decide whether the compressor can see it.
    let in_scope: Vec<Defect> = Defect::ALL
        .iter()
        .copied()
        .filter(|d| in_compressor_scope(*d))
        .collect();
    assert!(
        in_scope.len() >= 10,
        "the scope filter selected almost nothing"
    );

    let mut bad = Vec::new();
    // Which `BELOW_RESOLUTION` entries this run actually reached. An exclusion list with a dead
    // entry is the failure this test exists to not become.
    let mut reached = vec![false; BELOW_RESOLUTION.len()];
    let w = Widths::of(&c.engine);
    for spec in &list {
        let (mut cell, clean, gpu) = load_and_baseline(&ck, &c, spec);
        let cd = diff(&clean, &gpu, w);
        println!("{}", cd.one_line(&format!("{} clean", spec.name)));
        // RECORDED, not asserted here. An over-budget clean comparison makes the separations below
        // uninterpretable, but it does not make them unmeasurable — and their pattern is
        // diagnostic in its own right, so measuring them beats aborting.
        bad.extend(assert_clean(spec.name, &cd));
        let base = Baseline {
            name: spec.name,
            clean: &clean,
            gpu: &gpu,
        };
        for &def in &in_scope {
            // The two impersonations have their own, stronger tests in `kernel_v4_compress.rs`.
            if matches!(def, Defect::CompressorNoApe | Defect::RopeNoYarn) {
                continue;
            }
            let (broken, _) = cell.run(Run {
                defect: def,
                ..Run::clean(&spec.script)
            });
            bad.extend(score_one(
                base,
                def,
                &broken,
                Reach {
                    w,
                    hit: &mut reached,
                },
            ));
        }
    }
    for (i, hit) in reached.iter().enumerate() {
        let (cell_name, def, s) = BELOW_RESOLUTION[i];
        assert!(
            hit,
            "BELOW_RESOLUTION records {cell_name}/{def:?} at sep={s}, but this run never measured \
             that pair — the entry is dead and would absorb a future regression silently. Either \
             the cell list changed; or the defect became INERT there (which the branch above \
             skips, and which is a DIFFERENT coverage statement); or the defect left \
             `in_compressor_scope`; or it is `CompressorNoApe`/`RopeNoYarn`, which this loop skips \
             because they have their own stronger tests, making such an entry unreachable by \
             construction."
        );
    }
    assert!(
        bad.is_empty(),
        "the metric cannot resolve these, so a kernel carrying the defect might well pass: {}",
        bad.join(" | ")
    );
}

/// `CompressorNoOverlap` must be inert at ratio 128 and LIVE at ratio 4 — the pin that says the
/// sweep's `INERT` branch reports a real structural fact rather than a defect that quietly
/// stopped working.
///
/// Without this, every defect could become inert everywhere, the sweep above would print a wall of
/// `INERT`, and it would pass.
#[test]
fn the_overlap_defect_is_inert_at_ratio_128_and_live_at_ratio_4() {
    let Some((ck, c, _)) = cells() else { return };
    let script_128 = vec![(PROBE_REMAINDER_LEN, 0)];
    let mut l3 = Cell::load(&ck, &c, 3);
    let (clean_128, _) = l3.run(Run::clean(&script_128));
    let (broken_128, _) = l3.run(Run {
        defect: Defect::CompressorNoOverlap,
        ..Run::clean(&script_128)
    });
    assert_eq!(
        clean_128, broken_128,
        "at ratio 128 `overlap` is already false, so this defect has no term to disable"
    );

    let script_4 = vec![(PROBE_LEN, 0)];
    let mut l2 = Cell::load(&ck, &c, 2);
    let (clean_4, gpu_4) = l2.run(Run::clean(&script_4));
    let (broken_4, _) = l2.run(Run {
        defect: Defect::CompressorNoOverlap,
        ..Run::clean(&script_4)
    });
    assert_ne!(clean_4, broken_4, "at ratio 4 the defect must bite");
    let sep = gap(
        "ratio4 no-overlap vs gpu",
        &broken_4,
        &gpu_4,
        Widths::of(&c.engine),
    );
    assert!(
        sep >= RESOLVABLE,
        "the overlapping branch is the half of the compressor ratio 128 never runs, and this cell \
         resolves it by only {sep} bf16 codes"
    );
}
