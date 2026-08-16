//! **What the FP4 fixture's CODES actually cover, and what the compiled decoders do with every
//! one of them** — the half of the routed-expert gate that is about the DECODE rather than about
//! the dispatch.
//!
//! Split out of `kernel_v4_moe.rs` on 2026-08-16 under the 800-line soft cap; the dispatch harness
//! both files drive is `v4_moe/mod.rs`. The cut is by cohesion: next door asks whether the kernel
//! computes what the oracle computes, and this file asks what that comparison is entitled to claim
//! about `e2m1f` and `e8m0f`.
//!
//! # There is no exhaustive codec probe here, and that is a decision
//!
//! The obvious one — a synthetic weight whose columns cycle all 16 e2m1 codes and whose rows carry
//! every e8m0 code — cannot be read back through this pipeline: the e8m0 range spans `2^-127` to
//! `2^127` while `moe_fixed`'s accumulator is faithful only over roughly `[2^-21, 2^14]`
//! (`kernels/common.hpp`, `MOE_ACC_SHIFT`), so rows outside that band would be truncated or
//! clamped by the ACCUMULATOR and the test would be measuring the wrong thing.
//!
//! So the coverage is assembled from three tests that each say something a device comparison
//! alone cannot:
//!
//! 1. [`the_fixture_exercises_the_codes_the_decoders_are_credited_with`] MEASURES the code
//!    distribution the end-to-end comparison actually reaches, instead of assuming it is broad.
//! 2. [`the_branchless_decodes_match_the_oracle_bitwise`] pins both decode FORMULAS to the oracle
//!    over every code — including the three no device test can observe: e2m1 code 8 (`-0.0`,
//!    annihilated by the very next multiply) and e8m0's `0x00` and `0xff`.
//! 3. [`every_byte_pattern_decodes_right_in_both_dot_paths`] is the bridge to what hipcc actually
//!    emitted: a transliteration that drifted from the kernel passes 2 and fails 3.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no `rocm` CI arm, no GPU for this port. One mutation in
//! `kernels/common.hpp::e2m1f` — rotate the magnitude table's nibble constant by one
//! (`0xC864_3210` → `0x8643_210C`):
//!
//! * [`the_branchless_decodes_match_the_oracle_bitwise`] must go RED on a named code, since it
//!   compares the transliteration to the oracle on the host and the transliteration is the thing
//!   you edited — **so edit BOTH the kernel and the transliteration to test 3 alone**, which is
//!   the point of having the two tests: with only the kernel changed, 2 stays green and
//!   [`every_byte_pattern_decodes_right_in_both_dot_paths`] goes red, and that pairing is what
//!   says the second test observes the compiler rather than the comment.
//! * [`the_fixture_exercises_the_codes_the_decoders_are_credited_with`] must stay GREEN under
//!   either edit: it is a host histogram over weight BYTES and knows nothing about arithmetic. A
//!   red there means the toy's weights moved, not the decoder.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_oracles::v4oracle::forward::{Counters, ExpertOperand, ExpertW};
use rivoli_oracles::v4oracle::numerics::{bf16_decode, bf16_encode, e2m1_decode, e8m0_decode};
use rivoli_oracles::v4oracle::weights::{WMat, fixed_bf16};

mod common;
mod v4_moe;

use common::{assert_bits, byte_position_coverage, toy_fixture};
use v4_moe::{
    Case, Dispatch, F4Experts, PICKS, Wiring, assert_headroom, assert_matches, fp4_spans,
};

// =======================================================================================
// 2. what the fixture actually exercises
// =======================================================================================

/// What [`routed_experts_match_the_oracle`] passing does and does not say about the decoders.
///
/// The decoders are covered by the end-to-end comparison, over whatever code distribution the
/// fixture contains — and this test measures that distribution instead of assuming it is broad.
/// It is what turns "the expert test covers the decode" into a bounded claim, and it can go
/// red: shrink the toy's weight scale far enough and the e2m1 histogram collapses onto a
/// handful of codes while every other test here still passes.
#[test]
fn the_fixture_exercises_the_codes_the_decoders_are_credited_with() {
    // The experts `Case` actually ROUTES — not all `n_routed_experts`. An unrouted expert's
    // decode is annihilated inside the kernel (`wexpert == 0` gives `h == 0` exactly, which
    // `an_unrouted_expert_contributes_exactly_zero` proves), so counting its codes would credit
    // the comparison with coverage it cannot see. Host-only: it reads weights, not results, so
    // it builds no `Case` and touches no device.
    let (_, m, _) = toy_fixture();
    let mut nibbles = [0usize; 16];
    let mut scales = std::collections::BTreeSet::new();
    for &e in &PICKS {
        let ew = &m.layers[0].experts[&e];
        for w in [&ew.w1, &ew.w3, &ew.w2] {
            let (packed, s) = fp4_spans(w);
            for b in packed {
                nibbles[(b & 0x0f) as usize] += 1;
                nibbles[(b >> 4) as usize] += 1;
            }
            scales.extend(s.iter().copied());
        }
    }
    let missing: Vec<usize> = (0..16).filter(|&n| nibbles[n] == 0).collect();
    assert!(
        missing.is_empty(),
        "e2m1 codes never exercised by a ROUTED expert: {missing:?}"
    );
    // Printed, so it needs `-- --nocapture` to read: the BOUND is what a reader wants and any
    // threshold on it here would be a number picked to pass. `e8m0f`'s two special codes — 0x00
    // (2^-127, an f32 subnormal) and 0xff (NaN) — are decoded by nothing that runs on a device
    // in this file, which the assertion below states rather than prints.
    println!(
        "e2m1 code counts: {nibbles:?}\ne8m0 codes present: {} distinct, {:?}..={:?}",
        scales.len(),
        scales.iter().next(),
        scales.iter().next_back()
    );
    assert!(
        !scales.contains(&0u8) && !scales.contains(&0xffu8),
        "a special e8m0 code leaked in"
    );
}

/// The branchless decode FORMULAS against the oracle, bit for bit, over every code.
///
/// `kernels/common.hpp`'s `e2m1f`/`e8m0f` are branchless (the ternary forms compiled to an
/// exec-mask branch region per nibble, ~88 of the fp4 dot loop's 195 instructions). The two
/// functions below are line-for-line transliterations of those bodies — if you change the
/// kernel, change these or this test lies. What each half pins, and what it cannot:
///
/// - It proves the FORMULAS equal `v4oracle::numerics::{e2m1_decode, e8m0_decode}` at the bit
///   level — including code 8 → `-0.0` (the sign OR on a zero payload) and e8m0's `0x00` (the
///   f32 subnormal `2^-127`) and `0xff` (the `0x7fc00000` NaN), none of which any device test
///   can observe: a `-0.0` weight is annihilated by the very next multiply, and the e8m0
///   endpoints cannot ride through `moe_fixed` (module doc, item 3).
/// - It proves nothing about what hipcc COMPILED. That bridge is
///   [`every_byte_pattern_decodes_right_in_both_dot_paths`], which runs the real kernel over
///   every packed-byte pattern; a transliteration drifted from the kernel fails there.
#[test]
fn the_branchless_decodes_match_the_oracle_bitwise() {
    // kernels/common.hpp::e2m1f — magnitudes doubled are {0,1,2,3,4,6,8,12}, one immediate.
    fn e2m1f(nib: u32) -> f32 {
        let half = (0xC864_3210u32 >> ((nib & 7) << 2)) & 0xF;
        let mag = 0.5f32 * half as f32;
        f32::from_bits(mag.to_bits() | ((nib & 8) << 28))
    }
    // kernels/common.hpp::e8m0f — `max` against the b == 0 subnormal, quiet bit on 0xff.
    fn e8m0f(b: u8) -> f32 {
        let t = (b as u32) << 23;
        let quiet = if b == 0xff { 1u32 << 22 } else { 0 };
        f32::from_bits(t.max(1u32 << 22) | quiet)
    }
    let mine: Vec<f32> = (0..16u32).map(e2m1f).collect();
    let theirs: Vec<f32> = (0..16u8).map(e2m1_decode).collect();
    assert_bits(&theirs, &mine, "e2m1, all 16 codes");
    let mine: Vec<f32> = (0..=255u8).map(e8m0f).collect();
    let theirs: Vec<f32> = (0..=255u8).map(e8m0_decode).collect();
    assert_bits(
        &theirs,
        &mine,
        "e8m0, all 256 bytes including the endpoints",
    );
}

/// Packed bytes that put every one of the 256 values at every byte position of the dword fast
/// path: byte `i` carries `(i/4 + 64·(i%4) + salt) mod 256`, so position `p`'s bytes walk all
/// 256 values once per 256 consecutive dwords. `salt` decorrelates the three projections.
fn covering_bytes(n: usize, salt: u8) -> Vec<u8> {
    (0..n)
        .map(|i| {
            ((i >> 2) as u8)
                .wrapping_add((64 * (i & 3)) as u8)
                .wrapping_add(salt)
        })
        .collect()
}

/// The compiled kernel's decode over EVERY packed-byte pattern, in both dot paths.
///
/// [`the_branchless_decodes_match_the_oracle_bitwise`] pins the decode formulas; this is the
/// bridge to what hipcc actually emitted. One synthetic expert whose `w1`/`w3` bytes are
/// [`covering_bytes`] runs the toy geometry's single dword-path iteration (`dim = 256` =
/// `WAVE·8`), so every byte position of the weight dword decodes every value `0..=255` — a
/// wrong shift, mask or table constant at ANY position fails against the oracle on thousands of
/// terms. `w2` (`inter = 128` < `WAVE·8`) decodes entirely in the scalar tail, whose bytes cover
/// all 256 values too, both nibble-extraction parities included.
///
/// Coverage is COUNTED below, not trusted from the construction; scales cycle `2^-2..2^1` and
/// the activation is small, so everything stays inside `moe_fixed`'s faithful band (the
/// constraint that forbids sweeping e8m0 the same way) and outside the SwiGLU clamp, which
/// would otherwise mask a gate-row decode error behind a saturated `min`.
///
/// The coverage claim is about byte POSITIONS, not loop trips: at one iteration the dword loop's
/// advance never runs. Depth is [`the_dword_path_matches_the_oracle_at_multiple_trips`]'s.
#[test]
fn every_byte_pattern_decodes_right_in_both_dot_paths() {
    let (cfg, _, o) = toy_fixture();
    let (hidden, inter) = (cfg.dim, cfg.moe_inter_dim);
    // The coverage claims are geometry-bound; a resized toy silently voids them.
    assert_eq!(
        hidden, 256,
        "gate/up must be exactly one dword-path iteration"
    );
    assert_eq!(inter, 128, "w2 must decode entirely in the scalar tail");
    let scales = |rows: usize, k: usize| -> Vec<u8> {
        (0..rows * (k / 32)).map(|i| 125 + (i % 4) as u8).collect()
    };
    let (w1w, w3w, w2w) = (
        covering_bytes(inter * hidden / 2, 0),
        covering_bytes(inter * hidden / 2, 101),
        covering_bytes(hidden * inter / 2, 202),
    );
    for (label, w) in [("w1", &w1w), ("w3", &w3w)] {
        for p in 0..4 {
            let n = byte_position_coverage(w, p);
            assert_eq!(n, 256, "{label} dword byte position {p}: {n}/256 patterns");
        }
    }
    let mut seen = [false; 256];
    for &b in &w2w {
        seen[b as usize] = true;
    }
    let n = seen.iter().filter(|&&s| s).count();
    assert_eq!(n, 256, "scalar tail (w2): {n}/256 patterns");

    let mat = |rows, cols, w, s| WMat::Fp4 { rows, cols, w, s };
    let e = ExpertW {
        w1: mat(inter, hidden, w1w, scales(inter, hidden)),
        w2: mat(hidden, inter, w2w, scales(hidden, inter)),
        w3: mat(inter, hidden, w3w, scales(inter, hidden)),
    };
    let x = fixed_bf16("byte-pattern-sweep-x", hidden, 0.05);
    let mut counters = Counters::default();
    let rows = ExpertOperand {
        x: &x,
        m: 1,
        weight: Some(&[1.125]),
    };
    let want = o.expert(&e, rows, &mut counters);
    assert_eq!(
        counters.swiglu_clamp_events, 0,
        "a saturated clamp would mask gate-row decode errors — lower the activation scale"
    );
    assert_headroom(&want, "byte-pattern sweep");

    let experts = F4Experts::upload(&[&e], Wiring::Correct);
    let got = Dispatch::reference(cfg, &experts, &x, &[1.125]).run();
    assert_matches(&want, &got, "byte-pattern sweep (fp4)");
}

/// The bit pattern of the fp4 dispatch, PRINTED — the tripwire for any future edit to
/// `common.hpp::swiglu_clamped`, which the routed and shared paths share.
///
/// FMA contraction is uncontrolled tree-wide (`build.rs` gives hipcc only `--offload-arch -O3
/// -fPIC` and clang's HIP default is `-ffp-contract=fast`), so moving five lines across a
/// `__forceinline__` boundary can change codegen even where the arithmetic is identical. This
/// does not ASSERT a value — a hard-coded hash would be a golden tied to one compiler and one
/// GPU — it prints one, next to the oracle comparison that says the value is also correct.
#[test]
fn the_fp4_dispatch_hash_is_a_tripwire_on_the_shared_clamp() {
    let c = Case::new("moe-x-big", 48.0);
    assert!(
        c.clamp_events > 0,
        "the clamp must be exercised by the case that pins it"
    );
    let got = c.gpu();
    // FNV-1a over the bit patterns. Order-sensitive and 64-bit, so two runs that agree here
    // agree element for element; a sum or an XOR would not say that.
    let h = got.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, v| {
        (a ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    });
    println!("fp4 dispatch hash (shared-clamp tripwire): {h:#018x}");
    assert_matches(&c.want, &got, "fp4 dispatch behind the shared clamp");
}

/// The bf16 round-trip the fixture draw and the reference's stores both rely on.
///
/// Host-only, and it is the cheap guard under every claim above that a value "is
/// bf16-representable": if `bf16_decode(bf16_encode(x))` were not idempotent, every `draw_x`
/// would hand the kernel a value the oracle rounds and the kernel does not, and the whole file
/// would be measuring that instead of the arithmetic.
#[test]
fn the_bf16_round_trip_is_idempotent_over_the_fixture_draw() {
    let x = fixed_bf16("bf16-idem", 4096, 48.0);
    let again: Vec<f32> = x.iter().map(|v| bf16_decode(bf16_encode(*v))).collect();
    assert_bits(&x, &again, "fixture draw, re-rounded");
    assert!(
        x.iter().any(|v| v.abs() > 1.0) && x.iter().any(|v| *v < 0.0),
        "the draw must span both signs and cross unity, or it pins nothing about the format"
    );
}
