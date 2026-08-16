//! **DeepSeek-V4's routed FP4 expert dispatch, scored against the frozen host oracle.**
//! `moe_expert_range_f4` and nothing else.
//!
//! Ported from `old:tests/f4_kernel.rs` §1 and §3. `crates/oracles/src/v4oracle/` is a CPU
//! transliteration of `inference/model.py` that was proved before it was trusted (~40
//! deliberate breakages, each asserted both to fire where it should and to be silent where it
//! should not). This file is the other half of that arrangement: it runs the routed-expert
//! kernel against the same toy checkpoint and asks whether it computes what the oracle
//! computes.
//!
//! # Why numerical comparison and nothing else
//!
//! Every defect available on this path is silent. An unclamped SwiGLU, a `w1`/`w3` swap, a
//! high-first nibble read, a group stride of 128 instead of 32, a missing activation
//! quantization, a bias that reached the routing weights — none crash, all leave every shape,
//! magnitude, norm and code histogram plausible, and `distinct`/`longest repeated block` fire
//! identically on all of them. So the tests below do not check that the kernel produces
//! numbers; each deliberate break is asserted VISIBLE at the same tolerance the positive gate
//! passes at, which is what makes the positive gate mean something.
//!
//! # What this file CANNOT detect, measured rather than assumed
//!
//! 1. **The real checkpoint's values.** Everything runs on `V4Config::toy`, which preserves
//!    every discriminant and shrinks every extent. `toy` has `moe_inter_dim = 128`, so
//!    `moe_down_f4` never enters `dot_f4_wave_r`'s 8-nibble dword fast path (it needs
//!    `WAVE·8 = 256` columns) — only `moe_gateup_f4` does, at exactly one iteration. Depth is
//!    covered separately by [`the_dword_path_matches_the_oracle_at_multiple_trips`], at
//!    5/4 trips; that gap is what let the reference tree's `#pragma unroll` ship with 27 green
//!    tests that never executed the unrolled body.
//! 2. **Batch > 1.** Nothing here covers it and nothing can: the oracle itself is `bsz = 1`
//!    only. The FP4 launcher therefore REFUSES `nrow != 1` rather than shipping an unscoreable
//!    second row — [`expert_range_f4_guards`] pins that, on BOTH sides of 1.
//! 3. **The e8m0 ENDPOINTS through the pipeline.** There is no exhaustive codec probe here and
//!    that is a decision, not an omission: the e8m0 range spans `2^-127` to `2^127` while
//!    `moe_fixed`'s accumulator is faithful only over roughly `[2^-21, 2^14]`, so rows outside
//!    that band would be truncated by the ACCUMULATOR and the test would measure the wrong
//!    thing. [`the_branchless_decodes_match_the_oracle_bitwise`] covers the two decode
//!    FORMULAS over every code including `0x00` and `0xff`, on the host;
//!    [`every_byte_pattern_decodes_right_in_both_dot_paths`] is the bridge to what hipcc
//!    emitted; and [`the_fixture_exercises_the_codes_the_decoders_are_credited_with`] measures
//!    what the end-to-end comparison actually reaches instead of assuming it is broad.
//! 4. **The shared expert.** It is fp8 e4m3 at 128x128, not FP4, and is a different kernel —
//!    `kernel_v4_shared_expert.rs` owns it.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! This suite has never executed: `--features rocm` has no CI arm in this tree, and the author
//! of this port had no GPU. Before trusting a green run, make it go red. One mutation, in
//! `kernels/moe.hip`, and the expected magnitude below it:
//!
//! * In `moe_gateup_f4_impl`, drop the `swiglu_clamped` call's `limit` to `1e6f`
//!   (or replace `swiglu_clamped(g, u, limit)` with the unclamped `silu(g) * u`).
//!   [`the_swiglu_clamp_is_live_and_the_fixture_reaches_it`] must go RED and
//!   [`routed_experts_match_the_oracle`] must stay GREEN — the first runs at activation scale
//!   48 where `old:` measured a positive `swiglu_clamp_events`, the second at scale 1 where it
//!   measured zero. Both halves are the proof: a mutation that reddens BOTH has changed the
//!   arithmetic somewhere other than the clamp.
//! * In `dot_f4_wave_r`, force the eighth nibble (`n7`) to zero when `base != 0`. Only
//!   [`the_dword_path_matches_the_oracle_at_multiple_trips`] may go red, at
//!   `err=8.133e-2 > tol=1.247e-3` — 65x, the figure `old:` recorded when it injected exactly
//!   this. Every other test here stays green, which is that test's whole reason to exist.
#![cfg(feature = "rocm")]
// No `.unwrap()` in this file — `expect` carries the message everywhere, so only that lint
// needs allowing.
#![allow(clippy::expect_used)]

use rivoli_backend::hip::{ExpertDescF4, device_sync, launch_moe_expert_range_f4};
use rivoli_oracles::v4oracle::forward::ExpertW;
use rivoli_oracles::v4oracle::weights::V4Config;

mod common;
mod v4_moe;

use common::moe::{Dims, moe_bufs};
use common::{assert_bits, assert_guard, assert_guards, build_toy, stream, toy_fixture, zeros};
use v4_moe::{
    Break, Case, CaseSpec, Dispatch, F4_COLS_PER_TRIP, F4Experts, Knobs, NO_CLAMP, PICKS, Wiring,
    assert_disagrees, assert_headroom, assert_matches,
};

// =======================================================================================
// 1. the FP4 expert
// =======================================================================================

/// The load-bearing comparison: one MoE layer's routed experts, GPU against oracle.
///
/// A failure here is any of — a wrong e2m1 codebook, a wrong e8m0 decode, a group stride of 128
/// instead of 32, a missing or misplaced bf16 store, the routing weight applied after `w2`
/// instead of to the intermediate, a missing activation quantization, or the clamp on the wrong
/// side of the gate. The tests that follow separate those out; this one is what would catch a
/// defect nobody thought to name.
#[test]
fn routed_experts_match_the_oracle() {
    let c = Case::new("moe-x", 1.0);
    assert_eq!(
        c.clamp_events, 0,
        "this case is the UNCLAMPED half of the clamp bracket"
    );
    assert_matches(&c.want, &c.gpu(), "routed experts (fp4)");
}

/// Three silent breaks, each asserted to be VISIBLE to the gate above.
///
/// Together they are the argument that [`routed_experts_match_the_oracle`] passing means
/// something. Each changes exactly one thing:
///
/// - **`w1`/`w3` swapped.** Identical shapes, identical byte counts, identical scale grids, and
///   the `.f4` repack maps both through the same name→slot table, so nothing structural can see
///   it. It is detectable ONLY because SwiGLU is asymmetric in its two operands — `silu` applies
///   to the gate alone. Were the combine `g · u`, the swap would be a no-op and no instrument
///   could ever find it.
/// - **Nibbles read high-first.** Swapping the BYTES is the same experiment from the other end
///   and needs no second kernel: a high-first kernel on real bytes and a low-first kernel on
///   swapped bytes compute the same wrong thing. The `.f4` repack cannot check this; a matvec
///   is where it becomes checkable, and this is that check.
/// - **`x` not fp8-quantized.** `model.py::linear` quantizes the activation in front of the fp4
///   GEMM. Dropping it leaves every magnitude within `2^-3` of right.
#[test]
fn the_silent_fp4_breaks_are_visible() {
    let c = Case::new("moe-x", 1.0);
    // The wiring breaks share their KNOBS with the reference — only the upload changes — so the
    // pair is spelled once here and each break names the one thing it moves.
    let swap = |c: &Case, wiring| Break {
        wiring,
        knobs: c.knobs(),
    };
    for (label, got) in [
        ("w1/w3 swapped", c.broken(swap(&c, Wiring::SwapGateUp))),
        (
            "nibbles read high-first",
            c.broken(swap(&c, Wiring::SwapNibbles)),
        ),
        (
            "x not fp8-quantized",
            c.broken(Break {
                wiring: Wiring::Correct,
                knobs: Knobs {
                    quantize_x: false,
                    ..c.knobs()
                },
            }),
        ),
    ] {
        assert_disagrees(&c.want, &got, label);
    }
}

/// The clamped SwiGLU (`swiglu_limit = 10.0`), which rivoli's own unclamped SwiGLU does not
/// have.
///
/// BIDIRECTIONAL, and the direction that matters is the second: a clamp test on activations
/// that never reach the limit passes for a kernel with no clamp at all. The oracle's own
/// `swiglu_clamp_events` is what makes the fixture's reachability a MEASUREMENT — it is counted
/// independently of whether any kernel clamped, and [`routed_experts_match_the_oracle`] asserts
/// the other end of the bracket (zero events).
#[test]
fn the_swiglu_clamp_is_live_and_the_fixture_reaches_it() {
    let c = Case::new("moe-x-big", 48.0);
    assert!(
        c.clamp_events > 0,
        "the fixture never reaches the clamp, so this test could not distinguish a clamped \
         kernel from an unclamped one — raise the activation scale"
    );
    assert_matches(&c.want, &c.gpu(), "clamped swiglu");
    // Effectively unclamped, but positive — the launcher refuses 0 outright, which is the
    // stronger guarantee and the reason this arm has to go the long way round.
    assert_disagrees(
        &c.want,
        &c.broken(Break {
            wiring: Wiring::Correct,
            knobs: Knobs {
                swiglu_limit: NO_CLAMP,
                ..c.knobs()
            },
        }),
        "swiglu limit raised to 1e6",
    );
}

/// An unrouted expert contributes exactly zero, and the routed sum does not depend on how many
/// unrouted experts rode along.
///
/// `moe_down_f4` takes no routing mask — the weight is already in `h` — so "did not route" is
/// `h == 0` and nothing else. That the contribution is EXACTLY 0.0 rather than merely small is
/// what makes the missing mask safe, and it is not obvious: it needs `0 · finite` to stay zero
/// through the fp8 re-quantization of `h` and through `moe_fixed`.
#[test]
fn an_unrouted_expert_contributes_exactly_zero() {
    let c = Case::new("moe-x", 1.0);
    let full = c.gpu();
    // The same two picks, dispatched with ONLY the two experts they name uploaded. Every
    // unrouted expert is gone rather than zero-weighted, so if any of them was contributing
    // anything at all — a denormal, a NaN turned finite by `moe_fixed`'s clamp — the two results
    // differ. Compared as BIT PATTERNS, not as f32: the claim is exactness, and `PartialEq` on
    // f32 reports `-0.0 == 0.0`, which a zero contribution can produce.
    let named: Vec<&ExpertW> = PICKS.iter().map(|&e| c.all[e]).collect();
    let two = F4Experts::upload(&named, Wiring::Correct);
    let w = PICKS.map(|e| c.wexpert[e]);
    let just_two = Dispatch::reference(c.cfg, &two, &c.x, &w).run();
    assert_bits(&full, &just_two, "an unrouted expert perturbed the sum");
    assert_headroom(&full, "both arms");
}

/// `wexpert`, `h` and `descs` are indexed by ABSOLUTE expert id, so a dispatch split into
/// ranges gives the same answer as one range over everything.
///
/// This is the only test that passes `e_start > 0` to anything but a rejected arm, and the
/// convention it pins is one a caller can get wrong silently: reading `wexpert` as
/// range-relative and sizing it `e_count` compiles, and runs off the end the first time a
/// pipeline splits experts across two streams.
///
/// Two ranges that are not adjacent and do not start at 0, so a kernel that quietly used
/// `r / inter` as an absolute index, or offset `h` by the range rather than by the expert, lands
/// on different weights.
#[test]
fn a_dispatch_split_into_ranges_matches_one_range() {
    let c = Case::new("moe-x", 1.0);
    // Exactly the two experts `Case` routes, each its own range — and nothing else, so the sum
    // is over the same terms as `c.want` by a different dispatch.
    let split = c
        .dispatch(&c.experts, c.knobs())
        .in_ranges(&[(PICKS[0], 1), (PICKS[1], 1)]);
    assert_matches(&c.want, &split, "routed experts, dispatched as two ranges");
    // And bit-identical to the single-range dispatch: the fixed-point accumulator makes the sum
    // associative, so splitting it must change nothing at all, not merely little.
    assert_bits(&c.gpu(), &split, "range split perturbed the sum");
}

/// The dword fast path at a MULTI-TRIP shape — the only test in this file that reaches one.
///
/// At `V4Config::toy` gate/up runs this loop exactly ONCE (`dim 256` = `WAVE * 8`) and
/// `moe_down_f4` never enters it at all (`moe_inter_dim 128`), so `#pragma unroll N` executes
/// only the remainder copy everywhere else in this file.
///
/// The multipliers ARE the coverage claim, so they are written as TRIP COUNTS rather than as
/// 1280/1024: 5 trips = unrolled body + remainder at unroll 2 AND at unroll 4; 4 = clean groups
/// at both. NOTHING machine-checks the trip counts — the launcher's rc 1002 guards
/// `% ACT_QUANT_BLOCK` (128), not 256, so 1152 or 1536 would launch fine at a different count.
/// The test keeps its power over multi-trip arithmetic if the pragma moves, but 5/4 stops
/// guaranteeing an unrolled group PLUS a remainder past depth 4, and a changed `WAVE` breaks
/// the counts outright.
///
/// **Measured against injected defects in the reference tree, so nobody over-trusts it:** it
/// FIRES on arithmetic wrong only past the first trip (`n7` forced to 0 when `base != 0`) at
/// `err=8.133e-2 > tol=1.247e-3`, 65x, while every other test stays green — that pair is the
/// whole claim for its existence. It is BLIND to a trip miscount (`<=` → `<`; the scalar tail
/// resumes from `base` and absorbs the dropped trip) and to a pure reassociation.
#[test]
fn the_dword_path_matches_the_oracle_at_multiple_trips() {
    // NOT a resized `toy`: `every_byte_pattern_decodes_right_in_both_dot_paths` asserts toy's
    // one-trip geometry on purpose for its own byte-position coverage. Leaked because `Case`
    // holds `&'static` into the fixture and exactly one test wants this shape.
    let cfg = V4Config {
        dim: 5 * F4_COLS_PER_TRIP,
        moe_inter_dim: 4 * F4_COLS_PER_TRIP,
        ..V4Config::toy()
    };
    let c = Case::at(
        Box::leak(Box::new(build_toy(cfg))),
        CaseSpec {
            layer: 0,
            tag: "unroll-trips",
            scale: 1.0,
        },
    );
    // Bound rather than inlined: `assert_matches(&c.want, &c.gpu(), ..)` is token-identical to
    // `routed_experts_match_the_oracle`'s call and `build.rs`'s jscpd gate rejects the build.
    // Inlining it back is a build error, not a style choice.
    let got = c.gpu();
    assert_matches(
        &c.want,
        &got,
        "fp4 routed experts at 5/4 dword trips (unrolled body + remainder)",
    );
}

/// Every launcher guard, by CODE. Accepting any error would pass a build where an unrelated
/// dimension check swallowed the case first.
#[test]
fn expert_range_f4_guards() {
    let (cfg, m, _) = toy_fixture();
    let lw = &m.layers[0];
    let experts = F4Experts::upload(&[&lw.experts[&0]], Wiring::Correct);
    let dims = Dims::new(cfg.dim, cfg.moe_inter_dim);
    let x = zeros(cfg.dim * 4);
    let w = zeros(4);
    let (mut h, mut acc, _out) = moe_bufs(1, 1, dims);
    let stream = stream();

    let mut go = |(hidden, inter, e_start, e_count, n_desc, limit, nrow)| {
        // SAFETY: every rejected case returns before a dereference; the accepted case is sized
        // by the buffers above.
        unsafe {
            launch_moe_expert_range_f4(
                x.ptr() as *const f32,
                hidden,
                inter,
                e_start,
                e_count,
                n_desc,
                experts.descs.ptr() as *const ExpertDescF4,
                w.ptr() as *const f32,
                limit,
                h.ptr_mut() as *mut f32,
                acc.ptr_mut() as *mut u64,
                nrow,
                stream.raw(),
            )
        }
    };
    let (hid, int, lim) = (cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit);

    assert_guard(go((hid, int, 0, 1, 1, lim, 1)), None, "the accepted case");
    // The accepted case LAUNCHED. Join before the buffers drop: a launcher's `Ok` is
    // `hipGetLastError()` immediately after the launch, so an asynchronous fault would otherwise
    // surface in whichever test calls `device_sync()` next.
    device_sync().expect("device sync");
    assert_guards([
        (1001, "zero hidden", go((0, int, 0, 1, 1, lim, 1))),
        // 129 is not a multiple of ACT_QUANT_BLOCK. `assert N % block_size == 0` is the
        // reference's own; a ragged tail would quantize against a scale it never computes.
        (
            1002,
            "hidden not a whole act-quant block",
            go((129, int, 0, 1, 1, lim, 1)),
        ),
        (
            1002,
            "inter not a whole act-quant block",
            go((hid, 96, 0, 1, 1, lim, 1)),
        ),
        // BOTH sides of 1, which is what separates `!= 1` from a `> 1` that would accept 0. The
        // FP4 path instantiates only R=1: no measurement justifies a second row, and the oracle
        // is bsz=1, so one could not be scored even if it existed.
        (1003, "nrow 2", go((hid, int, 0, 1, 1, lim, 2))),
        (1003, "nrow 0", go((hid, int, 0, 1, 1, lim, 0))),
        // THE `.f4` BOUNDARY. `.vq3`/`.i4` carry the shared expert as block `n_experts`; `.f4`
        // does not, because V4's shared expert is fp8 e4m3 at 128x128. One past the end here is
        // the wrong ARITHMETIC, not merely the wrong weights.
        (
            1004,
            "one expert past the descriptor array",
            go((hid, int, 0, 2, 1, lim, 1)),
        ),
        (
            1004,
            "e_start past the descriptor array",
            go((hid, int, 1, 1, 1, lim, 1)),
        ),
        // Every value that disables the clamp, and each row is chosen to distinguish a SPELLING
        // rather than to enumerate bad numbers. `-10.0` is absent because `x <= 0` would reject
        // it too, so it separates nothing. NaN separates `!(x > 0)` from `x <= 0`. **+inf
        // separates `!(x > 0 && x < INFINITY)` from `!(x > 0)`** — and that row was missing in
        // the reference tree until 2026-08-05, which is exactly how the infinity route stayed
        // open on the one clamp launcher that has callers: `fminf(gt, inf)` returns `gt`, so the
        // clamp is simply gone, on every fp4 expert of every layer, silently.
        (1006, "unclamped swiglu", go((hid, int, 0, 1, 1, 0.0, 1))),
        (
            1006,
            "NaN swiglu limit",
            go((hid, int, 0, 1, 1, f32::NAN, 1)),
        ),
        (
            1006,
            "infinite swiglu limit",
            go((hid, int, 0, 1, 1, f32::INFINITY, 1)),
        ),
    ]);
}
