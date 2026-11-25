//! **Inkling 2 Small (`inkling2`, the MXFP8 member) composed end to end — the streaming gate.**
//!
//! `inkling_engine_anchor.rs` scores Mini's bf16 loop against the first-party anchor. Small is a
//! different composition: the same text stack with NO loop, and a routed-expert set that lives in
//! `.mxfp8` layer files streamed by `RoutedPool` through O_DIRECT rather than in one resident
//! slab. This file builds a TINY synthetic Small artifact (`inkling_small`, the module beside it,
//! on `/var/cache/rivoli/scratch` — never `/tmp`, where O_DIRECT does not work) and scores the
//! three facts only the composition can show:
//!
//! 1. **Residency invariance (P4).** The same prompt, decoded once with a budget that pins every
//!    expert and once with the smallest budget that opens at all, must produce **bit-identical**
//!    token ids AND logits. Bytes/token may move; text may not.
//! 2. **bf16 vs MXFP8 equivalence.** The SAME synthetic experts, dequantized to bf16 and written
//!    into a sibling resident-only artifact, must produce the same argmax and logits within the
//!    accumulation-order bound the two GEMV paths can differ by. That is what makes the
//!    `.mxfp8` path a representation of the model rather than a different model.
//! 3. **The config door.** `open_mxfp8` refuses `--ctx` past `SINK_ATTEND_MAX_N` before placing a
//!    byte — the door test opens a directory holding no weights at all.
//!
//! **Run it as the workspace runs every device suite:** `-- --test-threads=1`. The scored cells
//! each open their own pool and io_uring rings, and this tree's measured rule is that parallel
//! device tests wedge.
//!
//! **RED-PROOF RECIPE (P7 — owed for the device half).**
//! * P4: make the tight arm's pool hand back a stale slot (or skip the `wait` before reuse) and
//!   [`residency_moves_no_bit`] must redden on the token.
//! * cross-check: write the source's scale grid one exponent off and
//!   [`bf16_resident_experts_match_the_streamed_mxfp8_ones`] must redden on the argmax.
//! * door: delete `validate_scratch_cfg` from `open_mxfp8` and
//!   [`open_mxfp8_refuses_a_ctx_past_the_sink_attend_ceiling`] panics on `.expect(...)`.

#![cfg(feature = "rocm")]
// Tests: panic-on-failure is the idiom, and every fixture is generated rather than threaded out
// as a `Result`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod inkling_small;

use inkling_small::{Scratch, Storage, build, expert_total, write_config};
use rivoli_artifact::inkling_config::InklingTextConfig;
use rivoli_backend::abi::SINK_ATTEND_MAX_N;
use rivoli_engine::inkling::engine::InklingEngine;
use rivoli_engine::resident::PinCfg;
use rivoli_engine::seam::GenSpec;

const N_CTX: usize = 32;
const NGEN: usize = 4;
static PROMPT: [u32; 4] = [1, 7, 13, 29];

/// The one greedy request this file makes: the fixture prompt, `NGEN` new tokens, and no stop
/// token — the ids are compared bit-for-bit across budgets, so an EOS the fixture happened to
/// emit would make the comparison's width depend on which arm ran.
const SPEC: GenSpec<'static> = GenSpec {
    prompt: &PROMPT,
    ngen: NGEN,
    eos: &[],
};

/// The bf16-vs-MXFP8 tolerance, and where it comes from. Both artifacts hold the SAME values:
/// e4m3 (3 mantissa bits) widened to bf16 is exact, so the difference is not representation but
/// ACCUMULATION ORDER — a 64-term fp32 dot reduced in a wave for MXFP8 and a bf16-tensor GEMV for
/// the resident arm, compounded over 6 layers. Pure fp32 addition of 64 terms is bounded by
/// `64 · 2^-24 ≈ 4e-6` relative; the two paths' different reduction orders are what the remaining
/// factor covers. Measured evidence is owed with the device arm (see the header's recipe).
const MX_VS_BF16_TOL: f32 = 1e-3;

fn decode(engine: &mut InklingEngine<'_>) -> (Vec<u32>, Vec<f32>, u64) {
    let decoded = engine
        .decode(SPEC, &mut |_| true)
        .expect("the synthetic Small artifact must decode");
    let logits = engine.logits().expect("a sampled row is available");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "a decode row must be finite — an infinite or NaN logit is a broken composition, not a \
         tolerance question"
    );
    (decoded.ids, logits, decoded.stats.misses)
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

/// The smallest capacity that opens at all: every routed expert must stream from it. Found by
/// BISECTION on observed behaviour rather than by re-deriving the pin's floor in this file — the
/// pin's slack, the tier's 256-byte rounding and the pool's staging arena are the pin's
/// arithmetic, and a copy here would be a second author to drift.
fn open_tight<'a>(dir: &str, text: &'a InklingTextConfig) -> InklingEngine<'a> {
    let (mut lo, mut hi) = (0usize, 64 << 20);
    let opens =
        |cap: usize| InklingEngine::open_mxfp8(dir, text, PinCfg::stock(cap), N_CTX).is_ok();
    assert!(opens(hi), "a 64 MiB budget must open the ~2 MB fixture");
    // `PinCfg::stock(0)` is a LITERAL zero-byte budget, not "auto-size": the pin's floor must
    // refuse it. That is the invariant the bisection rests on, and it is asserted rather than
    // assumed because an auto-sizing capacity would silently make every later arm "pinned".
    assert!(
        !opens(0),
        "a zero-byte capacity must be refused by the pin's floor"
    );
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if opens(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    InklingEngine::open_mxfp8(dir, text, PinCfg::stock(hi), N_CTX)
        .expect("the minimal successful capacity opens by construction")
}

/// The door: a `--ctx` past the LDS ceiling is refused BEFORE anything is placed, so this runs
/// against a directory that holds a config and no weights at all.
#[test]
fn open_mxfp8_refuses_a_ctx_past_the_sink_attend_ceiling() {
    let scratch = Scratch::new("door");
    let cfg = write_config(&scratch.path());
    let over = SINK_ATTEND_MAX_N as usize + 1;
    let err = InklingEngine::open_mxfp8(&scratch.path(), &cfg.text, PinCfg::stock(64 << 20), over)
        .err()
        .expect("a --ctx past the sink_attend LDS ceiling must be refused");
    let message = format!("{err:#}");
    // Both halves are the contract: the first names WHAT refused, the second names the remedy.
    // A refusal that says only one of them is a message a caller cannot act on.
    for fragment in ["exceeds sink_attend LDS ceiling", "lower --ctx"] {
        assert!(
            message.contains(fragment),
            "the door must say {fragment:?}, not a failed malloc: {message}"
        );
    }
}

/// Teacher-forced scoring through the same streaming composition: one NLL per forced position,
/// each finite and non-negative. `p = 0` would be an infinity here rather than a rounding
/// artifact, so this is the NLL path's own composition check (and `score` refuses a host/device
/// argmax disagreement internally, which is what makes `agree` trustworthy at all).
#[cfg(feature = "teacher-forcing")]
#[test]
fn teacher_forced_scoring_is_finite_over_the_streamed_experts() {
    let scratch = Scratch::new("tf");
    let (dir, cfg) = build(&scratch, Storage::MxFp8);
    let mut engine = InklingEngine::open_mxfp8(&dir, &cfg.text, PinCfg::stock(64 << 20), N_CTX)
        .expect("the fixture opens for a scoring run");
    let scored = engine.score(&PROMPT).expect("teacher-forced scoring runs");
    assert_eq!(
        scored.nlls.len(),
        PROMPT.len() - 1,
        "one NLL per position that has a target"
    );
    for (i, nll) in scored.nlls.iter().enumerate() {
        assert!(
            nll.is_finite() && *nll >= 0.0,
            "position {i} scored {nll} — a negative or infinite NLL is p outside (0, 1]"
        );
    }
    assert!(scored.agree <= scored.nlls.len());
    eprintln!(
        "teacher-forced: {} positions, mean NLL {:.4}, {} argmax agreements, {} fills",
        scored.nlls.len(),
        scored.nlls.iter().sum::<f32>() / scored.nlls.len() as f32,
        scored.agree,
        scored.misses
    );
}

/// **P4, at the seam.** One prompt, two budgets: everything pinned, and the smallest capacity
/// that opens (so the routed set streams through `RoutedPool`'s O_DIRECT fill). The ids and the
/// logits must be BIT-identical — a token that moves when only residency changed is a pool
/// defect. The A-vs-A rerun is the floor: it separates "the pool is not state-free" from "this
/// arm does not repeat itself", which is the question a mismatch would raise.
#[test]
fn residency_moves_no_bit() {
    let scratch = Scratch::new("p4");
    let (dir, cfg) = build(&scratch, Storage::MxFp8);
    let total = expert_total(&cfg.text);

    let mut pinned = InklingEngine::open_mxfp8(&dir, &cfg.text, PinCfg::stock(64 << 20), N_CTX)
        .expect("the fixture opens with room for every expert");
    assert_eq!(
        pinned.residency(),
        (total, 0),
        "64 MiB must pin every one of the {total} experts and stream none"
    );
    let (ids, logits, first_misses) = decode(&mut pinned);
    let (ids_again, logits_again, warm_misses) = decode(&mut pinned);
    assert_eq!(
        warm_misses, 0,
        "the pinned arm must not miss once warmed (first decode had {first_misses} cold-start \
         fills, repeat had {warm_misses} — a budget that can hold every expert must never evict)"
    );
    assert_eq!(ids, ids_again, "the pinned arm must repeat its own ids");
    assert_eq!(
        bits(&logits),
        bits(&logits_again),
        "the pinned arm must repeat its own logits bit-for-bit"
    );

    let mut streamed = open_tight(&dir, &cfg.text);
    let (pinned_experts, streamed_experts) = streamed.residency();
    assert!(
        streamed_experts > 0,
        "the tight arm pinned all {pinned_experts} experts — a budget at the open boundary must \
         force the routed set through the pool"
    );
    let (ids_streamed, logits_streamed, streamed_misses) = decode(&mut streamed);
    assert!(
        streamed_misses > 0,
        "the tight arm recorded no miss — it never left the resident tier, so this run is not \
         the streaming arm it claims to be"
    );
    assert_eq!(
        ids, ids_streamed,
        "ids moved when only residency changed: pinned {ids:?} vs streamed {ids_streamed:?} — P4 \
         lets bytes/token move and forbids text from moving"
    );
    assert_eq!(
        bits(&logits),
        bits(&logits_streamed),
        "logits are not bit-identical across budgets — the pool is not state-free (a slot reused \
         under a live launch, or decoded from a stale block)"
    );
    eprintln!(
        "P4: {total} experts, pinned arm 0 fills, streaming arm {streamed_misses} fills, {} ids \
         and {} logits bit-identical",
        ids.len(),
        logits.len()
    );
}

/// **The representation cross-check.** The same experts, dequantized exactly (e4m3 → bf16 is
/// lossless), read by the resident bf16 arm and by the `.mxfp8` streaming arm: the argmax must
/// agree and the rows must agree within [`MX_VS_BF16_TOL`], which bounds accumulation order.
#[test]
fn bf16_resident_experts_match_the_streamed_mxfp8_ones() {
    let mx = Scratch::new("mx");
    let (mx_dir, mx_cfg) = build(&mx, Storage::MxFp8);
    let bf = Scratch::new("bf16");
    let (bf_dir, bf_cfg) = build(&bf, Storage::Bf16);

    let mut streamed =
        InklingEngine::open_mxfp8(&mx_dir, &mx_cfg.text, PinCfg::stock(64 << 20), N_CTX)
            .expect("the .mxfp8 fixture opens");
    let (ids_mx, logits_mx, _) = decode(&mut streamed);

    let mut resident = InklingEngine::open(&bf_dir, &bf_cfg.text, 64 << 20, N_CTX)
        .expect("the bf16 sibling opens through the resident path");
    let (ids_bf, logits_bf, _) = decode(&mut resident);

    assert_eq!(
        ids_mx, ids_bf,
        "the two representations of one model disagree on the argmax: mxfp8 {ids_mx:?} vs bf16 \
         {ids_bf:?}"
    );
    assert_eq!(
        logits_mx.len(),
        logits_bf.len(),
        "one row per vocabulary row"
    );
    let worst = logits_mx
        .iter()
        .zip(&logits_bf)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-3))
        .fold(0.0f32, f32::max);
    assert!(
        worst <= MX_VS_BF16_TOL,
        "MXFP8 and bf16 rows differ by {worst:e} relative, past the accumulation-order bound \
         {MX_VS_BF16_TOL:e} — the .mxfp8 blocks are not the model's own experts"
    );
    eprintln!("mxfp8 vs bf16: worst relative logit difference {worst:e}");
}
