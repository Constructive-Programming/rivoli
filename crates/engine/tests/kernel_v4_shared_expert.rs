//! **DeepSeek-V4's resident fp8 SHARED expert, and the clamped combine written for it** —
//! `swiglu_clamped_bf16`, driven both elementwise and through the whole expert chain.
//!
//! Ported from `old:tests/f4_kernel.rs` §7. `MoE.__init__` hands `swiglu_limit` to
//! `shared_experts` as well as to the routed ones (model.py:632) and `Expert.forward` clamps
//! both, but the shared expert is fp8 e4m3 at 128x128 rather than FP4 — a different kernel
//! chain, and one whose only available combine was GLM's unclamped `swiglu` until
//! `swiglu_clamped_bf16` existed. That is `v4oracle::Defect::SwigluUnclamped` on one
//! contribution in seven of every one of the 43 layers, fluent and wrong.
//!
//! # Why this is not `launch_swiglu` with a `limit`
//!
//! **At no value of `limit` would the two agree**, so a parameter could not have expressed it.
//! `launch_swiglu` is `(g/(1+e^-g))·u` and rounds nothing. Three differences besides the clamp:
//! both operands are bf16-rounded BEFORE the clamp (`Linear` stores bf16 and `Expert.forward`
//! reads it back with `.float()`); the product is bf16-rounded; and `F.silu`'s multiply form is
//! used, not the division form. GLM has no `swiglu_limit` and should never acquire one, so
//! passing an "unclamped" sentinel from its call sites would put a value in the tree that no
//! config can produce — this launcher refuses `<= 0`, NaN and `±inf` (guard 1006), so unclamped
//! is not spellable here at all.
//!
//! # What separates a clamped kernel from an unclamped one, and where
//!
//! **Nowhere, at ordinary activation scales.** `swiglu_limit` is 10.0 and the toy weights put
//! `|w1·x|` and `|w3·x|` around 1, so a clamp test built on the natural fixture passes against a
//! kernel with no clamp at all. So the fixture's reachability is MEASURED, from the oracle's own
//! `swiglu_clamp_events`, which is counted while the oracle computes and is independent of what
//! any kernel did. Both ends of the bracket are asserted: at scale 1 the count is zero and the
//! clamp must be BIT-INERT, at scale 48 the count is positive and the clamp must change the
//! answer.
//!
//! # What this file provably cannot detect
//!
//! * **The clamp's ASYMMETRY, through the expert.** `up` is clamped on both sides and `gate`
//!   only from above (model.py:606-607); the plausible wrong version clamps the gate from below
//!   too and reads as a tidier symmetry. Measured through the expert in the reference tree at
//!   `err=3.125e-2` against a `tol=9.424e-2` — a third of the bar — because for `g <= -limit`
//!   the difference is at most `|silu(-10)| = 4.540e-4` per element of `h` times `|up| <= 10`,
//!   i.e. **4.540e-3 per element**, and `silu` has already annihilated the operand before the
//!   lower clamp could matter. [`the_shared_expert_gate_clamp_matches_and_the_asymmetry_is_below_resolution`]
//!   records that as an EXPECTED non-separation; the asymmetry is gated at the COMBINE instead,
//!   by [`the_clamped_combine_is_bit_exact_elementwise`], where there is no `w2` accumulation to
//!   bury a 4.540e-3 term in and no tolerance to hide under.
//! * **Anything about a CALLER.** This gates the kernel. Whether the V4 layer loop reaches for
//!   this launcher rather than for GLM's `launch_swiglu` is that loop's gate, not this one's.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no `rocm` CI arm, no GPU for this port. Two mutations in
//! `kernels/linalg.hip::rivoli_swiglu_clamped_bf16`, each with the test that must move and the
//! tests that must not:
//!
//! * Clamp the gate from BELOW as well (`gt = fmaxf(fminf(g, limit), -limit)`).
//!   [`the_clamped_combine_is_bit_exact_elementwise`] must go RED, naming a probe pair whose
//!   gate is under `-10`; its own host-vs-host arm prints how many of the probe pairs the two
//!   clamp shapes differ on, so a green there means the probe table was narrowed rather than the
//!   kernel fixed. [`the_shared_expert_gate_clamp_matches_and_the_asymmetry_is_below_resolution`]
//!   must stay GREEN at `err ≈ 3.125e-2 <= tol ≈ 9.424e-2` — that is the recorded
//!   non-separation, and a red there is news about the metric, not about the clamp.
//! * Drop the bf16 round on the PRODUCT (`h = silu(gt) * ut` with no `rbf16`).
//!   [`the_clamped_combine_is_bit_exact_elementwise`] must go RED. This is the one
//!   `Defect::NoBf16Rounding` reaches on this path, and it is invisible to any tolerance —
//!   which is why the combine is gated bitwise and the expert chain is not.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_backend::hip::{
    launch_act_quant_f8_prefix, launch_gemv_fp8_bf16, launch_swiglu_clamped_bf16,
};
use rivoli_engine::device::DeviceBuf;
use rivoli_oracles::v4oracle::forward::{Counters, Defect, ExpertOperand, ExpertW, Oracle};
use rivoli_oracles::v4oracle::numerics::{bf16_decode, bf16_encode, e8m0_decode, silu};
use rivoli_oracles::v4oracle::weights::{V4Config, WMat, fixed_bf16};

mod common;
use common::{
    Want, assert_bits, assert_guards, assert_rel, assert_separates, back, dev, f32b, f32v, max_abs,
    stream, toy_fixture, zeros,
};

/// The tolerance the expert-chain comparisons are held to — two bf16 ulps, relative to the
/// largest expected element.
///
/// Not `common::assert_close`'s `1e-3·max + 1e-3`: that formula's ABSOLUTE floor dominates at
/// this fixture's scale, where a shared-expert output is ~2e-2, so the floor would be 5% of the
/// signal. The reference stores bf16 at every step, so any upstream difference — the wave
/// reduction's order against the oracle's sequential sum, `expf` against Rust's `exp` — flips an
/// element by a whole ulp rather than by its own magnitude. One ulp is the floor; two is the
/// margin for a flip in `h` that then propagates through `w2`.
const TOL: f32 = 1.0 / 128.0;

/// The block every fp8 `Linear` on this path quantizes and tiles its scale grid at.
const FP8_BLOCK: usize = 128;

/// "Effectively unclamped", spelled as a huge POSITIVE limit.
///
/// There is no way to ask for the unclamped form: the launcher refuses `0`, negatives, NaN and
/// `±inf`, which is the stronger guarantee and the reason every negative arm here goes the long
/// way round.
const NO_CLAMP: f32 = 1e6;

/// The shared expert's three fp8 weights on device — `(e4m3 bytes, f32 block scales)` per
/// projection, in `[w1, w2, w3]` order.
///
/// The scale bytes are widened here through `e8m0_decode`, which is exact: every e8m0 code is a
/// power of two. The oracle dequantizes the SAME bytes on its side through `WMat::row`, so a
/// disagreement between the two decoders is inside what this comparison covers.
///
/// Matched exhaustively rather than with a let-else: the shared expert being fp8 is the whole
/// reason it needs its own launcher, and a `.f4` block reaching here would be the wrong
/// ARITHMETIC rather than the wrong bytes.
fn upload_fp8_shared(e: &ExpertW) -> Vec<(DeviceBuf, DeviceBuf)> {
    [&e.w1, &e.w2, &e.w3]
        .into_iter()
        .map(|m| match m {
            WMat::Fp8 { w, s, .. } => {
                let widened: Vec<f32> = s.iter().map(|&c| e8m0_decode(c)).collect();
                (dev(w), dev(&f32b(&widened)))
            }
            WMat::Dense { .. } | WMat::Fp4 { .. } => panic!(
                "the shared expert is fp8 e4m3 at 128x128 — `MoE.__init__` passes \
                 `expert_dtype` only to the ROUTED experts"
            ),
        })
        .collect()
}

/// One fp8 `Linear` of the shared expert: `out[0..n_out] = bf16(w · x)`, `x` already quantized.
///
/// A helper rather than three spelled-out launches, and the reason is the argument list: ten
/// positional arguments where `n_out` and `k` are both bare `usize` is a place a mistake is a
/// wrong ANSWER rather than a compile error, and the three calls differ only in `(weight, n_out,
/// k, destination)`.
///
/// # Safety
/// `x` is `k` live quantized f32, `w` is the `[n_out, k]` e4m3 pair `upload_fp8_shared`
/// produced, `out` is `n_out` writable f32, none aliasing another, all live until the caller's
/// next join. `stream` is a live `hipStream_t`.
unsafe fn fp8_linear(
    x: *const f32,
    w: &(DeviceBuf, DeviceBuf),
    dims: (usize, usize),
    out: *mut f32,
    stream: *mut std::ffi::c_void,
) {
    let (n_out, k) = dims;
    // SAFETY: the caller's contract above.
    unsafe {
        launch_gemv_fp8_bf16(
            x,
            w.0.ptr(),
            w.1.ptr().cast(),
            1,
            n_out,
            k,
            FP8_BLOCK,
            1,
            out,
            stream,
        )
    }
    .expect("gemv_fp8_bf16");
}

/// One row of the resident fp8 shared expert on the GPU: three `gemv_fp8_bf16` and the clamped
/// combine — `Expert.forward` with `weights = None`.
///
/// The launch ORDER is the arithmetic, so it is spelled rather than abstracted: `act_quant(x)`
/// once (both `w1` and `w3` read the identical quantized row — the reference runs a separate
/// `act_quant` inside each `Linear`, on the same row at the same block, so the bytes are
/// identical), then `w1`/`w3`, then the combine, then `act_quant(h)` and `w2`. Every
/// `gemv_fp8_bf16` bf16-rounds its own output, which is where `Linear`'s bf16 store lives.
///
/// `limit` is a parameter and not `cfg.swiglu_limit` because the whole point below is an A/B on
/// it.
fn gpu_shared_expert(
    cfg: &V4Config,
    w: &[(DeviceBuf, DeviceBuf)],
    x: &[f32],
    limit: f32,
) -> Vec<f32> {
    let (dim, inter) = (cfg.dim, cfg.moe_inter_dim);
    let stream = stream();
    let st = stream.raw();
    let mut xq = dev(&f32b(x));
    let mut g = zeros(inter * 4);
    let mut u = zeros(inter * 4);
    let mut out = zeros(dim * 4);
    // SAFETY: `xq` is one row of `dim` f32; `g`/`u` are `inter`; `out` is `dim`; each weight is
    // `[o_dim, i_dim]` e4m3 with a 128x128 f32 scale grid by `upload_fp8_shared`'s contract. All
    // five outlive the join inside `back`.
    unsafe {
        launch_act_quant_f8_prefix(
            xq.ptr().cast(),
            xq.ptr_mut().cast(),
            1,
            dim,
            dim,
            FP8_BLOCK,
            st,
        )
        .expect("act_quant x");
        let xp = xq.ptr().cast::<f32>();
        fp8_linear(xp, &w[0], (inter, dim), g.ptr_mut().cast(), st);
        fp8_linear(xp, &w[2], (inter, dim), u.ptr_mut().cast(), st);
        // IN PLACE into `g`: `h` becomes `w2`'s input, which is one fewer allocation. Safe by
        // the kernel's own note — every thread reads both operands, then writes once, and that
        // write depends on both reads.
        launch_swiglu_clamped_bf16(
            g.ptr().cast(),
            u.ptr().cast(),
            inter,
            limit,
            g.ptr_mut().cast(),
            st,
        )
        .expect("clamped swiglu");
        launch_act_quant_f8_prefix(
            g.ptr().cast(),
            g.ptr_mut().cast(),
            1,
            inter,
            inter,
            FP8_BLOCK,
            st,
        )
        .expect("act_quant h");
        fp8_linear(
            g.ptr().cast(),
            &w[1],
            (dim, inter),
            out.ptr_mut().cast(),
            st,
        );
    }
    f32v(&back(&out))
}

/// `swiglu_clamp_events` for one shared-expert call at `defect`, and the oracle's answer.
///
/// The count comes from the ORACLE rather than from anything the kernel reports, which is what
/// makes the reachability claims below measurements instead of hopes.
fn oracle_shared(defect: Defect, x: &[f32], layer: usize) -> (Vec<f32>, usize) {
    let (cfg, m, _) = toy_fixture();
    let o = Oracle::new(cfg.clone(), defect);
    let mut c = Counters::default();
    let rows = ExpertOperand {
        x,
        m: 1,
        weight: None,
    };
    let y = o.expert(&m.layers[layer].shared, rows, &mut c);
    (y, c.swiglu_clamp_events)
}

/// The oracle produced something to compare. `assert_rel` scales its tolerance by
/// `max_abs(want)`, so an all-zero oracle result passes against an all-zero kernel result at
/// tol 0.
fn assert_non_vacuous(want: &[f32], what: &str) {
    assert!(
        max_abs(Want(want)) > 1e-6,
        "{what}: the oracle produced nothing to compare"
    );
}

/// The shared expert at an activation scale that NEVER reaches the clamp.
///
/// Two claims, and the second is the one a clamp test usually forgets. The fp8 path matches the
/// oracle; and where the oracle says the bound never binds, the clamp is **bit-inert** —
/// `limit = 10` and `limit = 1e6` produce identical bit patterns. That is the half of "the clamp
/// must separate exactly where it should and nowhere else" which says *nowhere else*, and
/// without it a kernel that clamped at the wrong threshold, or clamped `up` from the wrong side,
/// could still pass the positive gate below.
#[test]
fn the_shared_expert_matches_the_oracle_where_the_clamp_never_binds() {
    let (cfg, m, _) = toy_fixture();
    let x = fixed_bf16("shared-x", cfg.dim, 1.0);
    let (want, events) = oracle_shared(Defect::None, &x, 0);
    assert_eq!(
        events, 0,
        "this case is the UNCLAMPED half of the bracket — pick a lower scale"
    );
    assert_non_vacuous(&want, "clamp not binding");
    let w = upload_fp8_shared(&m.layers[0].shared);
    let got = gpu_shared_expert(cfg, &w, &x, cfg.swiglu_limit);
    assert_rel(&want, &got, "shared expert (fp8), clamp not binding", TOL);
    assert_bits(
        &got,
        &gpu_shared_expert(cfg, &w, &x, NO_CLAMP),
        "the clamp changed the answer on a case where the oracle counted ZERO clamp events, so \
         it is binding somewhere it must not — a wrong threshold, or `up` clamped on the wrong \
         side",
    );
}

/// The clamped SwiGLU on the shared expert, with the fixture MEASURED to reach it.
///
/// Four arms, and each is here because it rules out a way the other three could be green against
/// a wrong kernel:
///
/// 1. **The fixture reaches the clamp**, from the oracle's own event count. Without this the
///    remaining three compare a clamp that never fired.
/// 2. **The kernel matches the clamped oracle.** The positive gate.
/// 3. **`limit = 1e6` disagrees with the clamped oracle**, so the clamp is what separates them
///    at these inputs — and it must exceed the same [`TOL`] the positive arm passes at, which is
///    what `assert_separates` enforces.
/// 4. **`limit = 1e6` MATCHES the oracle running `Defect::SwigluUnclamped`.** This is the arm
///    that makes 3 mean something: it says the break is *precisely* the unclamped form rather
///    than some unrelated perturbation that happens to move the answer. A "break" that moved the
///    result for the wrong reason would pass 3 and fail this.
///
/// **There is deliberately no assertion that the defect oracle counted ZERO clamp events**, and
/// an earlier draft of this in the reference tree had one. It could not fail: `Oracle::expert`
/// sets `limit = 0.0` for that defect and both increments sit inside `if limit > 0.0`, so the
/// count is structurally zero. A guard nothing could make red is what this port has shipped
/// before; arm 4 makes the same claim from the numbers instead of from a counter.
#[test]
fn the_shared_expert_clamp_is_live_and_the_fixture_reaches_it() {
    let (cfg, m, _) = toy_fixture();
    let x = fixed_bf16("shared-x-big", cfg.dim, 48.0);
    let (want, events) = oracle_shared(Defect::None, &x, 0);
    assert!(
        events > 0,
        "the fixture never reaches `swiglu_limit`, so this test could not distinguish a clamped \
         kernel from an unclamped one — raise the activation scale"
    );
    println!("shared expert: {events} clamp events at scale 48");
    let w = upload_fp8_shared(&m.layers[0].shared);
    assert_rel(
        &want,
        &gpu_shared_expert(cfg, &w, &x, cfg.swiglu_limit),
        "clamped shared expert",
        TOL,
    );

    let unclamped = gpu_shared_expert(cfg, &w, &x, NO_CLAMP);
    assert_separates(&want, &unclamped, "the limit raised to 1e6", TOL);
    let (want_unclamped, _) = oracle_shared(Defect::SwigluUnclamped, &x, 0);
    assert_rel(
        &want_unclamped,
        &unclamped,
        "1e6 reproduces Defect::SwigluUnclamped",
        TOL,
    );
}

/// The clamp is ASYMMETRIC and **this comparison CANNOT gate that** — measured, not assumed, and
/// it is a property of the reference rather than of the fixture.
///
/// The plausible wrong version is `Defect::SwigluClampGateBothSides`. This test was written in
/// the reference tree to reject it through the expert and it does not, which was caught on the
/// GPU by the test's own anti-vacuity arm rather than by a reviewer:
///
/// ```text
/// shared expert: 12 gate values below -10 at scale 48
/// the two clamp shapes on this fixture: err=3.125e-2  tol=9.424e-2   (max |want| = 1.206e1)
/// ```
///
/// The fixture DOES reach the case — 12 elements of it — and the two clamp shapes still agree to
/// a third of the tolerance. The reason is a closed-form bound rather than a fixture accident:
/// for `g <= -limit` the asymmetric form computes `silu(g)` and the symmetric one `silu(-limit)`,
/// so the difference is at most `|silu(-10)| = 4.540e-4` times `|up| <= limit`, i.e.
/// **4.540e-3 per ELEMENT of `h`**.
///
/// **The per-element bound is what is proved; the observed 3.125e-2 is not it.** That figure is
/// max-abs on the expert's OUTPUT, after `w2` accumulates over `moe_inter_dim = 128` — 6.9x the
/// per-element bound, which is accumulation, not a contradiction.
///
/// **And the scale claim is narrower than it first looks.** The bound is per-element and
/// scale-free, but the NUMBER of affected elements grows with scale while the tolerance
/// saturates (`gt <= 10` and `|ut| <= 10` cap `h`, so `max|want|` stops growing). So: measured
/// unresolvable AT THIS FIXTURE'S SCALE, with a per-element bound explaining why pushing a
/// little harder will not help. Not a proof over all scales.
///
/// **This test still GATES the asymmetric clamp positively** — the `assert_rel` below is the
/// check that the kernel implements the reference's clamp, and deleting this test would lose it.
/// Only the *rejection* of the symmetric variant lives elsewhere, at
/// [`the_clamped_combine_is_bit_exact_elementwise`].
#[test]
fn the_shared_expert_gate_clamp_matches_and_the_asymmetry_is_below_resolution() {
    let (cfg, m, _) = toy_fixture();
    let x = fixed_bf16("shared-x-big", cfg.dim, 48.0);
    let (asym, events) = oracle_shared(Defect::None, &x, 0);
    let (sym, events_sym) = oracle_shared(Defect::SwigluClampGateBothSides, &x, 0);
    // The fixture reaches the case: each `g < -limit` is exactly one clamp event the asymmetric
    // form does not count, so the DIFFERENCE is the population size.
    assert!(
        events_sym > events,
        "the fixture has no gate value below -{}, so the measurement below would be reporting an \
         empty case rather than an unresolvable one ({events_sym} events vs {events})",
        cfg.swiglu_limit
    );
    println!(
        "shared expert: {} gate values below -{} at scale 48",
        events_sym - events,
        cfg.swiglu_limit
    );
    let w = upload_fp8_shared(&m.layers[0].shared);
    let got = gpu_shared_expert(cfg, &w, &x, cfg.swiglu_limit);
    assert_rel(&asym, &got, "gate clamped from above only", TOL);

    // The recorded non-separation, asserted so it cannot silently BECOME a separation nobody
    // noticed. Note what is and is not pinned: the 4.540e-3 figure in the doc bounds ONE ELEMENT
    // of `h`, while this is max-abs error on the expert's OUTPUT after `w2` accumulated over 128
    // of them. A red here means the recording is stale — re-measure before treating it as a new
    // gate.
    assert_rel(
        &asym,
        &sym,
        "the two clamp shapes (recorded as UNRESOLVABLE at 3.125e-2)",
        TOL,
    );
}

/// `swiglu_clamped_bf16` elementwise against a host transliteration, BIT FOR BIT.
///
/// Bitwise is legitimate here and nowhere else in this file: the kernel is one thread per element
/// with no reduction, so there is no summation order to diverge from. The retraction of a bitwise
/// gate over a WAVE-REDUCED kernel at dim 4096 is about the reduction, not about elementwise ops.
///
/// The inputs are adversarial by construction rather than by luck. `expf` is the only
/// transcendental and HIP's need not agree with Rust's to the last bit, so a disagreement is
/// reported with the offending element rather than swept into a tolerance — if this ever goes red
/// on `expf` alone, the right response is to say so, not to widen it.
#[test]
fn the_clamped_combine_is_bit_exact_elementwise() {
    let limit = 10.0f32;
    // Straddling the bound on both sides and at it, plus the values that make an asymmetric clamp
    // distinguishable from a symmetric one and a bf16-first clamp from a clamp-first one:
    // `10.001` rounds DOWN to 10.0 in bf16 (8 mantissa bits, so the codes near 10 are 2^-5 apart),
    // so clamping before the round and after it differ on it.
    let probes: Vec<f32> = vec![
        0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 9.9, -9.9, 9.999, -9.999, 10.0, -10.0, 10.001, -10.001,
        10.0625, -10.0625, 12.0, -12.0, 40.0, -40.0, 1e3, -1e3,
    ];
    let (mut g, mut u) = (Vec::new(), Vec::new());
    for &a in &probes {
        for &b in &probes {
            g.push(a);
            u.push(b);
        }
    }
    let n = g.len();
    // The host side, written as the reference reads: bf16 both, clamp `up` both sides, `gate` per
    // `gate_clamp`, `F.silu`'s MULTIPLY form, bf16 the product.
    //
    // ONE definition taking the gate clamp as a function, because the two arms below differ in
    // exactly that expression and nothing else — which is also precisely the difference between
    // model.py:607 and the tidier wrong version.
    let combine = |gate_clamp: &dyn Fn(f32) -> f32| -> Vec<f32> {
        g.iter()
            .zip(&u)
            .map(|(&gv, &uv)| {
                let gt = gate_clamp(bf16_decode(bf16_encode(gv)));
                let ut = bf16_decode(bf16_encode(uv)).clamp(-limit, limit);
                bf16_decode(bf16_encode(silu(gt) * ut))
            })
            .collect()
    };
    // `torch.clamp(gate, max=self.swiglu_limit)` — ABOVE only. The reference.
    let want = combine(&|x: f32| x.min(limit));
    let (gb, ub) = (dev(&f32b(&g)), dev(&f32b(&u)));
    let mut hb = zeros(n * 4);
    let stream = stream();
    // SAFETY: three live `n`-element f32 buffers, outliving the join inside `back`.
    unsafe {
        launch_swiglu_clamped_bf16(
            gb.ptr().cast(),
            ub.ptr().cast(),
            n,
            limit,
            hb.ptr_mut().cast(),
            stream.raw(),
        )
    }
    .expect("swiglu_clamped_bf16");

    // **THE ASYMMETRY GATE's anti-vacuity arm, and it is HOST vs HOST.** `sym` against `want`,
    // NOT against the device output, and the difference decides what a failure MEANS. Comparing
    // the symmetric host arm to the device conflates two causes: a narrowed probe table, and a
    // kernel that genuinely is symmetric. The second is the whole defect this arm exists to
    // catch, and it would have been reported as "the probe table has no gate value below -10",
    // sending the next reader to fix the fixture. Removing the kernel from the comparison makes
    // the claim unambiguous: *these two host functions differ on this input set*, therefore a
    // kernel can be asked which one it implements.
    let sym = combine(&|x: f32| x.clamp(-limit, limit));
    let moved = sym
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        moved > 0,
        "the symmetric and asymmetric gate clamps are BIT-IDENTICAL over {n} probe pairs, so no \
         comparison against them could tell the reference's clamp from the tidier wrong one — \
         the probe table has no gate value below -{limit}"
    );
    println!("asymmetric vs symmetric gate clamp: {moved}/{n} probe pairs differ");
    // An EDIT TRIPWIRE, not a measurement, and worth being plain about which: `probes` is a
    // literal fifteen lines up, so this folds to a constant on today's tree and can only go red
    // if someone narrows that table. That is exactly the change it is here to stop.
    assert!(
        g.iter()
            .zip(&u)
            .any(|(&a, &b)| a > limit || b.abs() > limit),
        "no probe crosses the limit — the table was narrowed"
    );
    assert_bits(&want, &f32v(&back(&hb)), "swiglu_clamped_bf16 elementwise");
}

/// The launcher's guards, by CODE — including the one that matters most.
///
/// Two rows matter here and they are the same defect from opposite ends of the float line — both
/// are values that make the clamp VANISH rather than values that make it fail loudly:
///
/// - **NaN**, which a `limit <= 0.0f` guard admits, because every comparison against NaN is
///   false. `fminf(gt, NaN)` returns `gt`.
/// - **+inf**, which `!(limit > 0.0f)` admits. `fminf(gt, inf)` is `gt` and `fmaxf(ut, -inf)` is
///   `ut`, so the clamp is simply gone.
///
/// The guard is therefore `!(limit > 0.0f && limit < INFINITY)` — the two-sided spelling. Code
/// 1006 is deliberately the same one `moe.hip`'s fp4 launcher returns for the same check on the
/// same argument, and that launcher had the identical hole.
#[test]
fn swiglu_clamped_bf16_guards() {
    let mut b = zeros(64);
    let (p, pm) = (b.ptr().cast::<f32>(), b.ptr_mut().cast::<f32>());
    let nul = std::ptr::null_mut();
    // `null_mut()` for the stream: every case is rejected before `hipLaunchKernelGGL`, so there
    // is no launch for a stream to order.
    // SAFETY: each call returns at an argument guard, before any pointer is read.
    let cases = unsafe {
        [
            (
                1001,
                "zero elements",
                launch_swiglu_clamped_bf16(p, p, 0, 10.0, pm, nul),
            ),
            (
                1006,
                "an unclamped limit",
                launch_swiglu_clamped_bf16(p, p, 16, 0.0, pm, nul),
            ),
            (
                1006,
                "a negative limit",
                launch_swiglu_clamped_bf16(p, p, 16, -1.0, pm, nul),
            ),
            (
                1006,
                "a NaN limit",
                launch_swiglu_clamped_bf16(p, p, 16, f32::NAN, pm, nul),
            ),
            // +inf is the case a `!(limit > 0.0f)` guard ADMITS, and it disables the clamp
            // exactly as thoroughly as `limit = 0` would. It sits next to the NaN row
            // deliberately — the two are the same defect from opposite ends of the float line,
            // and having only one of them on the page is how the other stayed open.
            (
                1006,
                "an infinite limit",
                launch_swiglu_clamped_bf16(p, p, 16, f32::INFINITY, pm, nul),
            ),
        ]
    };
    assert_guards(cases);
}
