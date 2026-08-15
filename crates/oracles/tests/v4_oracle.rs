//! **Proving the V4 oracle can fail.**
//!
//! `src/v4oracle/` is the instrument S2 and S3 will be scored against. If it is blind to a
//! class of defect, the whole port ships silent-wrong and we find out at the benchmark. So
//! this suite does not test that the oracle produces numbers; it tests that the *gate* built
//! on those numbers rejects wrong implementations, and — the half that is usually missing —
//! that it stays quiet where the defect does not reach.
//!
//! Three layers of evidence, and where each one lives since the 800-line ceiling split the
//! file on 2026-08-15 (the toy driver they all share is `common/oracle_probe.rs`):
//!
//! 1. **Exhaustive codec tests** — `v4_oracle_codecs.rs`. Every fp8, fp4 and bf16 pattern,
//!    each against an independent brute-force reference rather than against itself.
//! 2. **The defect matrix** — THIS file. [`Defect`] enumerates ~40 deliberate breakages.
//!    Each is run across a grid of (layer class x prefill/decode x prompt length) and
//!    asserted BOTH ways: named goldens that must differ, named goldens that must stay
//!    bit-identical, and whole cases where the defect is unreachable and the entire capture
//!    must be unchanged. The defects a grid row cannot honestly carry are paid for by
//!    targeted tests in `v4_oracle_targeted.rs`, `v4_oracle_head_tail.rs`,
//!    `v4_oracle_reduction.rs` and `v4_oracle_codecs.rs` — see [`targeted_defects`].
//! 3. **Meta-guards** — THIS file. A defect with no declared silent evidence, or no
//!    reachable case, or no table entry at all, is itself a test failure — so the matrix
//!    cannot rot into a row of "differs everywhere", which proves nothing.
//!
//! Everything the matrix is BUILT ON — the comparator, the golden file format, the
//! safetensors reader, the `--defect` parser — is proved able to go red in `v4_oracle_gate.rs`.
//!
//! **The most-trusted case is the blind spot.** The grid deliberately does not privilege
//! layer 0. It has `compress_ratio = 0`: no compressor, no indexer, no YaRN, base theta —
//! the *least* representative layer in the model. The grid runs a ratio-0 layer, a ratio-4
//! layer (which has an `Indexer`) and a ratio-r layer (which does not), and several defects
//! are asserted to be inert on layer 0 precisely so that a fixture built only on it would
//! be visibly insufficient.
//!
//! Runs on the toy config (`V4Config::toy`), not the checkpoint: the questions here are
//! structural, and this way they are re-answered in seconds on every `cargo test` with
//! nothing on disk. `bin/v4-oracle defects` re-runs the same matrix against the real
//! weights so the toy's verdict is cross-checked rather than trusted.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_oracles::golden::{Diff, diff, identical};
use rivoli_oracles::v4oracle::forward::Defect;

#[path = "common/oracle_probe.rs"]
mod oracle_probe;
use oracle_probe::{Case, Phase, Run, cases, fingerprint, matching, model, run};

/// Goldens a defect MUST perturb, and goldens it MUST leave bit-identical.
///
/// `silent` is checked in every reachable case; `loud` requires at least one golden with
/// that suffix to differ (a decode case holds four steps and a compressor defect only
/// reaches the step that completes a block).
#[derive(Clone, Copy)]
struct Expect {
    loud: &'static [&'static str],
    silent: &'static [&'static str],
}

/// The attention half's goldens in PIPELINE order.
///
/// A row's silent half is almost always a PREFIX of this — "everything strictly upstream of
/// what the defect touches" — so the prefixes below are CUT from this one declaration rather
/// than retyped per row. Retyping is real duplication and not a formatting artefact: jscpd
/// reported the `DEROTATION`/`WO_GROUPS` pair on 2026-08-15, 28 tokens of identical prefix.
const PIPELINE: &[&str] = &[
    ".in",
    ".attn_norm_out",
    ".q",
    ".kv_entry",
    ".attn_core_out",
    ".attn_derot",
    ".attn_out",
];

/// Suffixes that are upstream of everything in the attention and are the strongest silent
/// evidence available: if a defect moves these, it is not the defect it claims to be.
const UPSTREAM: &[&str] = PIPELINE.split_at(2).0;

/// ...plus the query projection, which a KV-side defect must leave alone.
const PRE_KV: &[&str] = PIPELINE.split_at(3).0;

/// ...plus both projections — everything a defect *inside* the attention core is downstream of.
const PRE_ATTN: &[&str] = PIPELINE.split_at(4).0;

/// ...plus the core's own output, which the de-rotation is downstream of.
const PRE_DEROT: &[&str] = PIPELINE.split_at(5).0;

/// ...plus the de-rotation, which the output projection is downstream of.
const PRE_WO: &[&str] = PIPELINE.split_at(6).0;

/// What the router sits between: the attention half is upstream of it and the expert mix is
/// downstream, so a routing bug may move neither.
const AROUND_ROUTER: &[&str] = &[".in", ".attn_norm_out", ".attn_out", ".ffn_norm_out"];

/// Silent claims that no implementation can violate, so they must not count as evidence.
///
/// `run_layer` records `{tag}.in` from the `h` it was handed, before any defect-sensitive
/// code, and every driver here supplies a FIXED `h` per step on purpose. `*.in` is therefore
/// bit-identical for every defect in every case BY CONSTRUCTION. Harmless in a silent list
/// that carries other entries; fatal as a row's only entry, which is exactly the
/// "differs somewhere" row the meta-guard exists to forbid.
const TRIVIAL_SILENT: &[&str] = &[".in"];

/// QK-norm rescales `.q` after the projection, so nothing before the projection may move.
const QK_NORM: Expect = Expect {
    loud: &[".q"],
    silent: &[".in", ".attn_norm_out", ".kv_entry"],
};

/// RoPE writes the query AND the cache entry — a port that rotates only one of them is the
/// defect this row is here to catch.
const ROPE: Expect = Expect {
    loud: &[".q", ".kv_entry"],
    silent: UPSTREAM,
};

/// The KV activation quantizer touches the cache entry alone; in particular not `.q`, the
/// projection a port is most likely to quantize alongside it.
const KV_QUANT: Expect = Expect {
    loud: &[".kv_entry"],
    silent: PRE_KV,
};

/// The attention core READS the compressor's output, so `.compressed` is silent evidence here
/// in a way it is not for [`RING_SEED`].
const ATTN_CORE: Expect = Expect {
    loud: &[".attn_core_out"],
    silent: &[".in", ".attn_norm_out", ".q", ".kv_entry", ".compressed"],
};

/// Wrong ring seeding shows up only once something reads the cache, and the compressor is not
/// on that path at all — which is why this row cannot claim `.compressed`.
const RING_SEED: Expect = Expect {
    loud: &[".attn_core_out"],
    silent: PRE_ATTN,
};

/// De-rotation is the last thing before the output projection, so the core's own output is
/// upstream of it.
const DEROTATION: Expect = Expect {
    loud: &[".attn_derot"],
    silent: PRE_DEROT,
};

/// A mis-grouped `wo` moves only the projection's result; every attention golden feeding it is
/// silent evidence.
const WO_GROUPS: Expect = Expect {
    loud: &[".attn_out"],
    silent: PRE_WO,
};

/// The compressor consumes the cache entries and produces `.compressed`; it writes nothing
/// upstream of itself.
const COMPRESSOR: Expect = Expect {
    loud: &[".compressed"],
    silent: PRE_ATTN,
};

const INDEXER: Expect = Expect {
    // Only `.indexer_scores` here. `.compress_idxs` -- the SELECTION golden -- is
    // informative ONLY where `index_topk` truncates, which does not happen in every
    // case of the grid, so a matrix row demanding it would be false half the time.
    // `the_selection_golden_moves_when_topk_truncates` covers it where it bites.
    loud: &[".indexer_scores"],
    // The indexer has its OWN compressor; the attention compressor's output must be
    // untouched. That separation is exactly what a port is likely to conflate.
    silent: &[".in", ".attn_norm_out", ".q", ".kv_entry", ".compressed"],
};

/// A weighting bug moves the weights and leaves the SELECTION alone.
const ROUTER_WEIGHTS: Expect = Expect {
    loud: &[".router_weights"],
    silent: AROUND_ROUTER,
};

/// ...and a dispatch bug moves the selection and leaves the weights alone.
const ROUTER_INDICES: Expect = Expect {
    loud: &[".router_indices"],
    silent: AROUND_ROUTER,
};

/// Where the routing weight is applied changes the mix, never the routing that produced it.
const EXPERT_MIX: Expect = Expect {
    loud: &[".ffn_out"],
    silent: &[".ffn_norm_out", ".router_weights", ".router_indices"],
};

const FP4_NIBBLES: Expect = Expect {
    loud: &[".ffn_out"],
    // Attention is fp8 and the shared expert is fp8; only the ROUTED experts are
    // fp4, so nothing before the MoE may move.
    silent: &[
        ".in",
        ".attn_norm_out",
        ".q",
        ".kv_entry",
        ".attn_out",
        ".router_weights",
    ],
};

const COMBINATION: Expect = Expect {
    loud: &[".ffn_norm_out", ".out"],
    // `pre` comes straight from the mixes and never sees the Sinkhorn iterations, so the
    // attention half of the block is untouched by a combination-matrix bug -- ALL of it, which
    // is why this is the whole [`PIPELINE`]. (`.attn_derot` was missing from the hand-written
    // list this replaced; it is upstream of `.attn_out`, which the list already claimed, so
    // the row was understating a claim it already made.)
    silent: PIPELINE,
};

const HEAD_NORM: Expect = Expect {
    loud: &[".final_norm_out", ".logits"],
    // `.hc_head_out` is the real silent half here, and it is violable: an implementation
    // that fused `hc_head` with the final norm -- the obvious single-kernel shortcut,
    // since both are reductions over the same row -- would move it.
    silent: &[".hc_head_out"],
};

/// Slicing the wrong row changes the logits and nothing that produced them.
const HEAD_LOGITS: Expect = Expect {
    loud: &[".logits"],
    silent: &[".hc_head_out", ".final_norm_out"],
};

fn expect(d: Defect) -> Option<Expect> {
    // `None` here means "covered by a targeted test in a sibling binary", and the meta-guard
    // checks that every such defect really is.
    match d {
        Defect::None => None,

        Defect::SkipQkNorm | Defect::QkNormUsesQNormWeight => Some(QK_NORM),

        Defect::RopeAllDims | Defect::RopeFirstDims | Defect::RopeHalfSplit => Some(ROPE),
        Defect::RopeNoYarn | Defect::RopeYarnEverywhere | Defect::RopeBaseThetaEverywhere => {
            Some(ROPE)
        }

        Defect::SkipKvActQuant | Defect::KvActQuantWholeTensor | Defect::KvActQuantNoRoundScale => {
            Some(KV_QUANT)
        }
        // See `act_quant_block_size_is_almost_invisible_under_ue8m0_scales`: this one is
        // measurably undetectable on realistic activations, so putting it in the matrix
        // would claim a resolution the oracle does not have.
        Defect::KvActQuantBlock128 => None,

        Defect::SkipAttnSink | Defect::AttnSinkNotMaxShifted => Some(ATTN_CORE),
        Defect::PrefillRingWritesFirstWindow => Some(RING_SEED),

        Defect::SkipOutputDerotation | Defect::OutputDerotationForward => Some(DEROTATION),
        Defect::WoGroupsSplitHeadDim | Defect::WoGroupsInterleaved => Some(WO_GROUPS),

        Defect::CompressorNoOverlap
        | Defect::CompressorNoApe
        | Defect::CompressorRopeAtBlockEnd => Some(COMPRESSOR),
        Defect::IndexerNoRelu
        | Defect::IndexerNoFp4Quant
        | Defect::IndexerNoHadamard
        | Defect::IndexerNoWeights => Some(INDEXER),

        // MEASURED over the whole grid, 2026-08-05: this moves exactly ONE score element, at
        // (layer 2, prompt 12, decode step 2), out of ~60 live scores -- and never moves
        // `.compress_idxs` or anything downstream. A matrix row would be false in 15 of the
        // 16 cases.
        //
        // That is a statement about the FIXTURE, not about the defect. The toy runs
        // `index_n_heads = 4`, so the reduction has 3 rounding opportunities where the model's
        // 64 heads have 63, and at 64 heads the same fold disagrees with torch **72.6%** of
        // the time (`Oracle::bf16_sum`). Raising the toy's head count would let the grid see
        // it, and is deliberately NOT done here: `V4Config::toy` is the shared fixture for
        // `tests/f4_attn.rs` and `tests/f4_kernel.rs` and moving it would invalidate their
        // goldens. Covered absolutely instead, by
        // `bf16_reduction_matches_torch_and_not_a_running_fold`, which is the only kind of
        // check that could have caught this class at all -- see that test's header.
        Defect::IndexerBf16RunningSum => None,

        Defect::SwigluUnclamped
        | Defect::SwigluClampGateBothSides
        | Defect::RouterNoSoftplusThreshold => None,

        Defect::RouterSoftmax
        | Defect::RouterBiasedWeights
        | Defect::RouterNoRenorm
        | Defect::RouterNoScale => Some(ROUTER_WEIGHTS),
        Defect::HashRoutingIgnored => Some(ROUTER_INDICES),
        Defect::RouteWeightAfterW2 | Defect::SharedExpertWeighted => Some(EXPERT_MIX),
        Defect::Fp4NibbleSwap => Some(FP4_NIBBLES),

        // See `sinkhorn_has_converged_long_before_iteration_20`.
        Defect::SinkhornIterCountProbe => None,
        // A candidate design's fold, not a slip, and the toy is structurally blind to it:
        // the dispatch predicate needs `k >= 4096` and the toy's largest K is 256, so no
        // toy GEMV can select it and a matrix row would assert on 43 vacuous cells. See
        // `the_splitk_fold_is_toy_blind_partition_exact_and_nonzero_at_real_dims`.
        Defect::SplitKFoldOrder => None,
        Defect::SinkhornCombTransposed | Defect::HcPostNoComb => Some(COMBINATION),
        // Both of these reach EVERY golden downstream of `hc_pre` -- which is all of them --
        // so neither has a silent half to declare, and `.in` (fixed by the driver) would be
        // a claim no implementation could violate. Demoted to targeted tests, the same way
        // `KvActQuantBlock128` and `SinkhornIterCountProbe` were.
        Defect::HcPreNoRsqrt | Defect::NoBf16Rounding => None,

        // -- head tail ------------------------------------------------------------------
        // Same shape of problem as `HcPreNoRsqrt`: both reach every head golden and the only
        // thing upstream of them is the layer stack, which `head_tail` cannot touch because
        // it takes `&[f32]`. A silence Rust's own types enforce is not evidence about this
        // gate, so both are targeted rather than matrix rows.
        Defect::HeadHcNoRsqrt | Defect::HeadHcRsqrtPerCopy => None,

        Defect::HeadNormSkipped | Defect::HeadNormNotBf16 | Defect::HeadNormOverAllTokens => {
            Some(HEAD_NORM)
        }
        Defect::HeadLogitsFromFirstRow => Some(HEAD_LOGITS),

        // Mathematically INERT: `apply_rotary_emb` rotates adjacent pairs, so it PRESERVES
        // `q.square().mean(-1)`, and a scalar scale commutes with a rotation. The two orders
        // differ only in where the bf16 rounding lands. Keeping it in the matrix would
        // advertise a detection the gate does not have at any usable tolerance.
        Defect::QkNormAfterRope => None,
    }
}

/// Every defect [`expect`] returns `None` for, and the sibling binary that pays for it:
///
/// | defect group | test binary |
/// |---|---|
/// | `SwigluUnclamped`, `SwigluClampGateBothSides`, `RouterNoSoftplusThreshold`, `SinkhornIterCountProbe`, `SplitKFoldOrder`, `QkNormAfterRope`, `HcPreNoRsqrt`, `NoBf16Rounding` | `v4_oracle_targeted.rs` |
/// | `KvActQuantBlock128` | `v4_oracle_codecs.rs` |
/// | `HeadHcNoRsqrt`, `HeadHcRsqrtPerCopy` | `v4_oracle_head_tail.rs` |
/// | `IndexerBf16RunningSum` | `v4_oracle_reduction.rs` |
///
/// The list is a DECLARATION, and it moved one file away from the tests it names when the
/// 800-line ceiling split this suite on 2026-08-15 — so the table above is the only thing
/// tying a name here to a test there. `every_defect_carries_both_halves_of_its_claim` still
/// checks the half that matters most: a defect may not be in both this list and the matrix,
/// and may not be in neither.
fn targeted_defects() -> Vec<Defect> {
    vec![
        Defect::SwigluUnclamped,
        Defect::SwigluClampGateBothSides,
        Defect::RouterNoSoftplusThreshold,
        Defect::KvActQuantBlock128,
        Defect::SinkhornIterCountProbe,
        Defect::SplitKFoldOrder,
        Defect::QkNormAfterRope,
        Defect::HcPreNoRsqrt,
        Defect::NoBf16Rounding,
        Defect::HeadHcNoRsqrt,
        Defect::HeadHcRsqrtPerCopy,
        Defect::IndexerBf16RunningSum,
    ]
}

/// Where the defect can fire at all. Everything else must leave the WHOLE capture identical.
fn reachable(d: Defect, c: &Case, base: &Run) -> bool {
    let (cfg, _) = model();
    let ratio = cfg.compress_ratio(c.layer);
    let k = base.of(c.phase).counters;
    match d {
        // YaRN is selected per layer: compressed layers use it, ratio-0 layers do not.
        Defect::RopeNoYarn | Defect::RopeBaseThetaEverywhere => ratio != 0,
        Defect::RopeYarnEverywhere => ratio == 0,

        // The ring only rotates when the prompt outruns the window, and the wrong seeding is
        // only observable once something READS the cache, which prefill never does.
        Defect::PrefillRingWritesFirstWindow => {
            c.phase == Phase::Decode && base.pre.counters.prefill_evicted > 0
        }

        // Overlapping pooling exists only at ratio 4.
        Defect::CompressorNoOverlap => ratio == 4 && k.compressed_blocks > 0,
        Defect::CompressorNoApe | Defect::CompressorRopeAtBlockEnd => k.compressed_blocks > 0,

        // `Indexer` exists only where `compress_ratio == 4` -- 21 of the model's 43 layers.
        Defect::IndexerNoRelu
        | Defect::IndexerNoFp4Quant
        | Defect::IndexerNoHadamard
        | Defect::IndexerNoWeights => k.indexer_ran,

        // The load-balancing bias exists only on score-routed layers.
        Defect::RouterBiasedWeights => c.layer >= cfg.n_hash_layers,
        // ...and `tid2eid` only on hash layers.
        Defect::HashRoutingIgnored => c.layer < cfg.n_hash_layers,

        // Both are INERT at one row, by construction rather than by fixture: `x[:, -1]` IS
        // `x[:, 0]` when there is one row, and a per-token RMS over one token IS the joint
        // one. Every decode step here is `s == 1`, so the whole Decode capture must come back
        // bit-identical -- which is also why these two are dangerous in the field. A decode
        // smoke test cannot see either, and the engine spends almost all its life at s == 1.
        Defect::HeadNormOverAllTokens | Defect::HeadLogitsFromFirstRow => c.phase == Phase::Prefill,

        _ => true,
    }
}

fn first_change(ds: &[Diff]) -> String {
    ds.iter().find(|d| d.changed > 0).map_or_else(
        || "nothing".to_string(),
        |d| format!("{} ({} elements)", d.name, d.changed),
    )
}

/// The undefected run for every (layer, prompt) in the grid.
fn baselines() -> std::collections::HashMap<(usize, usize), Run> {
    let mut m = std::collections::HashMap::new();
    for c in cases() {
        m.entry((c.layer, c.prompt))
            .or_insert_with(|| run(c.layer, c.prompt, Defect::None));
    }
    m
}

/// One matrix row under test: the defect and the two halves of the claim it makes.
#[derive(Clone, Copy)]
struct Row {
    defect: Defect,
    expect: Expect,
}

/// Every golden the row says must MOVE really moved somewhere in this case.
fn assert_loud(row: Row, c: &Case, ds: &[Diff]) {
    let d = row.defect;
    for suffix in row.expect.loud {
        let hits = matching(ds, suffix);
        assert!(
            !hits.is_empty(),
            "{d:?} at {c:?}: no golden named *{suffix} exists, so the grid does not \
             exercise this defect at all"
        );
        assert!(
            hits.iter().any(|h| h.changed > 0),
            "{d:?} at {c:?}: left every *{suffix} bit-identical -- the gate would \
             pass a wrong implementation here"
        );
    }
}

/// ...and every golden it says must NOT move is bit-identical. This is the half that carries
/// the resolution: "differs somewhere" would pass without it.
fn assert_silent(row: Row, c: &Case, ds: &[Diff]) {
    let d = row.defect;
    for h in row.expect.silent.iter().flat_map(|s| matching(ds, s)) {
        assert_eq!(
            h.changed, 0,
            "{d:?} at {c:?}: perturbed {} ({} of {} elements, rel {:.3e}), which is \
             upstream of or beside what it claims to affect",
            h.name, h.changed, h.total, h.rel
        );
    }
}

/// The head tail must not be able to MASK an upstream error. Wherever a defect moved the
/// layer's residual output, the logits have to move too -- otherwise a per-layer golden could
/// fail while the token that comes out is unchanged, or, far worse, the reverse. Checked off
/// the `ds` the caller already computed: a second pass over the grid would double this file's
/// runtime for evidence that is free at this point.
///
/// Paired by STEP, not any-to-any across the capture. A Decode capture holds four steps, so an
/// any/any check would be satisfied by a defect that moved `.out` at `dec0` and the logits at
/// `dec3` -- which is not the claim. Both names carry the same `{tag}`, so the pairing is free.
///
/// Returns the number of (step) pairs checked, which the caller's anti-vacuity bound needs.
fn count_propagated(d: Defect, c: &Case, ds: &[Diff]) -> usize {
    let mut n = 0usize;
    for h in matching(ds, ".out").iter().filter(|h| h.changed > 0) {
        let tag = h.name.split('.').nth(1).unwrap_or_default();
        let want = format!("head.{tag}.logits");
        let lg = ds.iter().find(|x| x.name == want).unwrap_or_else(|| {
            panic!(
                "{d:?} at {c:?}: {} moved but there is no {want} to check",
                h.name
            )
        });
        assert!(
            lg.changed > 0,
            "{d:?} at {c:?}: moved {} but left {want} bit-identical -- the head tail \
             absorbed the error",
            h.name
        );
        n += 1;
    }
    n
}

/// What the grid actually exercised, so the bounds at the end of the matrix test assert over
/// something that was incremented rather than over a loop that never ran.
#[derive(Default)]
struct Tally {
    reached: usize,
    silenced: usize,
    propagated: usize,
}

impl Tally {
    /// One (defect, case) cell: assert it, count it, and return its fingerprint.
    fn check(&mut self, row: Row, c: &Case, base: &Run) -> u64 {
        let got = run(c.layer, c.prompt, row.defect);
        let (d, ds) = (row.defect, diff(base.of(c.phase), got.of(c.phase)));
        let print = fingerprint(got.of(c.phase));
        if !reachable(d, c, base) {
            assert!(
                identical(base.of(c.phase), got.of(c.phase)),
                "{d:?} is unreachable at {c:?} but changed {}",
                first_change(&ds)
            );
            self.silenced += 1;
            return print;
        }
        assert_loud(row, c, &ds);
        assert_silent(row, c, &ds);
        self.propagated += count_propagated(d, c, &ds);
        self.reached += 1;
        print
    }
}

/// Two defects with the SAME fingerprint vector are the same defect wearing two names, and the
/// matrix would then count one piece of evidence twice. This is not hypothetical: `RopeNoYarn`
/// and `RopeBaseThetaEverywhere` were exactly that until both stopped selecting the base-theta
/// table.
fn assert_no_twin(d: Defect, mine: &[u64], prints: &[(Defect, Vec<u64>)]) {
    for (other, theirs) in prints {
        assert_ne!(
            mine,
            theirs.as_slice(),
            "{d:?} and {other:?} compute the SAME thing in every case -- they are one \
             defect wearing two names, and the matrix is double-counting its evidence"
        );
    }
}

#[test]
fn defect_matrix_is_bidirectional() {
    let baselines = baselines();
    let mut tally = Tally::default();
    let mut prints: Vec<(Defect, Vec<u64>)> = Vec::new();
    for &d in Defect::ALL {
        let Some(expect) = expect(d) else { continue };
        let row = Row { defect: d, expect };
        let mine: Vec<u64> = cases()
            .iter()
            .map(|c| tally.check(row, c, &baselines[&(c.layer, c.prompt)]))
            .collect();
        assert_no_twin(d, &mine, &prints);
        prints.push((d, mine));
    }
    let (reached, silenced, propagated) = (tally.reached, tally.silenced, tally.propagated);
    assert!(
        reached > 200,
        "only {reached} reachable (defect, case) pairs were asserted"
    );
    assert!(
        silenced > 40,
        "only {silenced} unreachable pairs -- too little silent evidence"
    );
    // The propagation claim above is only worth anything if it was exercised. A change that
    // stopped `.out` from moving anywhere -- or that dropped the head tail out of `run` --
    // would leave the `if` cold and the assertion inside it vacuously satisfied.
    // MEASURED 2026-08-05: 1046 (defect, case, step) triples move a layer `.out`, and every
    // one of them moves that same step's logits. The bound is a witness, not the observation
    // -- set well under 1046 so ordinary fixture drift does not trip it, and far enough above
    // zero that dropping the head tail out of `run`, or a change that stopped `.out` moving,
    // would. (It read 417 while the check paired per CAPTURE rather than per step.)
    assert!(
        propagated > 400,
        "only {propagated} (defect, case, step) triples moved a layer .out -- 1046 did when \
         this was measured -- so the \"the head tail cannot mask an upstream error\" claim is \
         nearly untested"
    );
}

#[test]
fn every_defect_carries_both_halves_of_its_claim() {
    // The guard against the matrix rotting into "differs everywhere", which proves nothing
    // about the gate's resolution.
    let baselines = baselines();
    let targeted = targeted_defects();
    for d in Defect::breakages() {
        let Some(exp) = expect(d) else {
            assert!(
                targeted.contains(&d),
                "{d:?} has no matrix row and no targeted test"
            );
            continue;
        };
        assert!(!targeted.contains(&d), "{d:?} is covered twice; pick one");
        assert!(
            !exp.loud.is_empty(),
            "{d:?} declares nothing it must perturb"
        );
        let n_reach = cases()
            .iter()
            .filter(|c| reachable(d, c, &baselines[&(c.layer, c.prompt)]))
            .count();
        assert!(
            n_reach > 0,
            "{d:?} is unreachable in every case, so nothing tests it"
        );
        let real_silent = exp
            .silent
            .iter()
            .filter(|s| !TRIVIAL_SILENT.contains(s))
            .count();
        assert!(
            real_silent > 0 || n_reach < cases().len(),
            "{d:?} is reachable everywhere AND declares no NON-TRIVIAL golden it must leave \
             alone, so its matrix row is 'differs somewhere' and carries no information \
             about the gate's resolution"
        );
    }
}

#[test]
fn the_grid_actually_covers_three_layer_classes() {
    // A fixture check, not a behaviour check: if the toy config drifted so that (say) every
    // layer had an indexer, several defects above would silently stop being bidirectional
    // while every assertion still passed.
    let (cfg, _) = model();
    let classes: Vec<usize> = (0..cfg.n_layers).map(|l| cfg.compress_ratio(l)).collect();
    assert!(classes.contains(&0), "no ratio-0 layer");
    assert!(
        classes.contains(&4),
        "no ratio-4 layer (the only kind with an Indexer)"
    );
    assert!(
        classes.iter().any(|&r| r != 0 && r != 4),
        "no compressed layer WITHOUT an indexer -- layer 0 and layer 2 alone would leave the \
         ratio-128 class untested, and that class is 20 of the model's 43 layers"
    );
    for (l, &ratio) in classes.iter().enumerate() {
        let r = run(l, 12, Defect::None);
        let has_idx = r.pre.float(&format!("L{l}.pre.indexer_scores")).is_some();
        let has_comp = r.pre.float(&format!("L{l}.pre.compressed")).is_some();
        assert_eq!(
            has_idx,
            ratio == 4,
            "layer {l} (ratio {ratio}) indexer presence is wrong"
        );
        assert_eq!(
            has_comp,
            ratio != 0,
            "layer {l} (ratio {ratio}) compressor presence is wrong"
        );
        assert!(
            r.pre.int(&format!("L{l}.pre.router_indices")).is_some(),
            "layer {l} recorded no routing"
        );
        // and the goldens are not degenerate
        let out = r.pre.float(&format!("L{l}.pre.out")).expect("L{l}.pre.out");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "layer {l} produced non-finite output"
        );
        assert!(
            out.iter().any(|&v| v != 0.0),
            "layer {l} produced an all-zero output"
        );
    }
    // The ring must actually rotate at the long prompt and not at the short one, or
    // `PrefillRingWritesFirstWindow` has no silent case.
    assert_eq!(run(0, 5, Defect::None).pre.counters.prefill_evicted, 0);
    assert!(run(0, 12, Defect::None).pre.counters.prefill_evicted > 0);
}
