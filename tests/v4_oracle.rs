//! **Proving the V4 oracle can fail.**
//!
//! `src/v4oracle/` is the instrument S2 and S3 will be scored against. If it is blind to a
//! class of defect, the whole port ships silent-wrong and we find out at the benchmark. So
//! this file does not test that the oracle produces numbers; it tests that the *gate* built
//! on those numbers rejects wrong implementations, and — the half that is usually missing —
//! that it stays quiet where the defect does not reach.
//!
//! Three layers of evidence:
//!
//! 1. **Exhaustive codec tests.** Every fp8, fp4 and bf16 pattern, each against an
//!    independent brute-force reference rather than against itself.
//! 2. **The defect matrix.** [`Defect`] enumerates ~40 deliberate breakages. Each is run
//!    across a grid of (layer class x prefill/decode x prompt length) and asserted BOTH
//!    ways: named goldens that must differ, named goldens that must stay bit-identical, and
//!    whole cases where the defect is unreachable and the entire capture must be unchanged.
//! 3. **Meta-guards.** A defect with no declared silent evidence, or no reachable case, or
//!    no table entry at all, is itself a test failure — so the matrix cannot rot into a
//!    row of "differs everywhere", which proves nothing.
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

mod common;
use common::residual_probe;

use rivoli::v4oracle::forward::{Capture, Defect, HeadTailW, LayerCtx, Oracle};
use rivoli::v4oracle::golden::{Diff, GoldenSet, diff, identical};
use rivoli::v4oracle::numerics::{
    FP4_MAX, FP8_MAX, act_quant_inplace, bf16_decode, bf16_encode, e2m1_decode, e2m1_encode,
    e4m3_decode, e4m3_encode, e8m0_decode, fast_round_scale, fp4_act_quant_inplace,
    hadamard_rotate, softplus,
};
use rivoli::v4oracle::toy::{self, ToyModel};
use rivoli::v4oracle::weights::{NamedRng, V4Config, WMat};
use std::sync::OnceLock;

// =======================================================================================
// 1. codecs
// =======================================================================================

/// Every finite `float8_e4m3fn` magnitude, ascending, with its code.
fn e4m3_finite_codes() -> Vec<(u8, f32)> {
    (0u8..=0x7e)
        .map(|c| (c, e4m3_decode(c)))
        .filter(|(_, v)| v.is_finite())
        .collect()
}

/// Nearest-with-ties-to-even, by enumerating the format. Deliberately dumb and deliberately
/// not sharing a line with `e4m3_encode`: an encoder tested against a restatement of itself
/// tests nothing.
///
/// It DOES share the decoder, so this pins the encoder against the decoder and no further.
/// `e4m3_and_bf16_decode_match_the_format_by_hand` is what anchors the decoder to the format.
fn nearest_e4m3(a: f32) -> u8 {
    let codes = e4m3_finite_codes();
    let mut best = codes[0];
    for &(c, v) in &codes {
        let (dn, db) = ((a - v).abs(), (a - best.1).abs());
        // Tie -> the code with the even mantissa (low bit of the code clear).
        if dn < db || (dn == db && (best.0 & 1) != 0) {
            best = (c, v);
        }
    }
    best.0
}

#[test]
fn e4m3_and_bf16_decode_match_the_format_by_hand() {
    // Without this, `e4m3` is checked ONLY against itself: `nearest_e4m3` enumerates
    // `e4m3_decode`, and the round-trip test checks the decoder against the encoder. A
    // shared misunderstanding -- a bias of 8 written into both, a subnormal quantum of 2^-10
    // in both -- passes every other test in this file. These values are read off the format
    // definition (1-4-3, bias 7, no infinities, S.1111.111 = NaN), not off the code.
    for (code, want) in [
        (0x00u8, 0.0f32),
        (0x01, 1.0 / 512.0), // smallest subnormal: quantum 2^-9
        (0x07, 7.0 / 512.0), // largest subnormal
        (0x08, 1.0 / 64.0),  // smallest normal: 2^-6
        (0x38, 1.0),         // exp 7 == bias, mantissa 0
        (0x3f, 1.875),       // exp 7, mantissa 7 -> 1 + 7/8
        (0x78, 256.0),       // exp 15, mantissa 0
        (0x7e, 448.0),       // largest finite: 1.75 * 2^8
    ] {
        assert_eq!(e4m3_decode(code), want, "e4m3 code {code:#04x}");
        assert_eq!(
            e4m3_decode(code | 0x80),
            -want,
            "e4m3 code {:#04x}",
            code | 0x80
        );
    }
    assert!(
        e4m3_decode(0x7f).is_nan() && e4m3_decode(0xff).is_nan(),
        "S.1111.111 is NaN"
    );

    // bf16 is f32's top 16 bits; likewise pinned by hand rather than by round-tripping.
    for (bits, want) in [
        (0x0000u16, 0.0f32),
        (0x3f80, 1.0),
        (0x4000, 2.0),
        (0xbf80, -1.0),
        (0x3f00, 0.5),
    ] {
        assert_eq!(bf16_decode(bits), want, "bf16 {bits:#06x}");
    }
    assert!(bf16_decode(0x7f80).is_infinite() && bf16_decode(0x7fc0).is_nan());
}

#[test]
fn e4m3_encode_is_nearest_ties_to_even() {
    let mut r = NamedRng::new("e4m3-sweep");
    let mut checked = 0usize;
    // Sweep the whole dynamic range, and land exactly on every representable midpoint so the
    // tie rule is actually exercised rather than merely available.
    let codes = e4m3_finite_codes();
    let mut probes: Vec<f32> = codes.iter().map(|&(_, v)| v).collect();
    for w in codes.windows(2) {
        probes.push((w[0].1 + w[1].1) / 2.0);
    }
    for _ in 0..20_000 {
        probes.push(r.unit() * 512.0);
        probes.push(r.unit() * 0.01);
    }
    for a in probes {
        if a.abs() >= 464.0 {
            continue; // saturating range; covered separately
        }
        let want = nearest_e4m3(a.abs());
        let got = e4m3_encode(a.abs());
        assert_eq!(
            got,
            want,
            "e4m3_encode({a:e}) = {got:#04x} ({}) but nearest-ties-even is {want:#04x} ({})",
            e4m3_decode(got),
            e4m3_decode(want)
        );
        checked += 1;
    }
    // The random probes at +/-512 are half out of range, so the reachable count is well
    // below the number generated. Asserted so the sweep cannot quietly shrink to nothing.
    assert!(
        checked > 30_000,
        "only {checked} probes reached the assertion"
    );
    assert_eq!(e4m3_encode(1e30), 0x7e, "saturate, not overflow");
    assert_eq!(e4m3_encode(-1e30), 0xfe);
    assert!(e4m3_decode(0x7f).is_nan() && e4m3_decode(0xff).is_nan());
}

#[test]
fn e4m3_roundtrip_is_idempotent_on_every_code() {
    for c in 0u8..=0xff {
        let v = e4m3_decode(c);
        if v.is_nan() {
            continue;
        }
        assert_eq!(
            e4m3_decode(e4m3_encode(v)).to_bits(),
            v.to_bits(),
            "code {c:#04x} -> {v} did not survive a re-encode"
        );
    }
}

#[test]
fn e2m1_encode_is_nearest_ties_to_even() {
    // The seven midpoints between the eight magnitudes, each named with the value the
    // even-mantissa rule demands. Written out rather than computed: this is the one place a
    // shared helper would let a wrong rule agree with itself.
    for (probe, want) in [
        (0.25, 0.0),
        (0.75, 1.0),
        (1.25, 1.0),
        (1.75, 2.0),
        (2.5, 2.0),
        (3.5, 4.0),
        (5.0, 4.0),
    ] {
        assert_eq!(e2m1_decode(e2m1_encode(probe)), want, "e2m1 tie at {probe}");
        assert_eq!(
            e2m1_decode(e2m1_encode(-probe)),
            -want,
            "e2m1 tie at -{probe}"
        );
    }
    let mags = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut r = NamedRng::new("e2m1-sweep");
    for _ in 0..20_000 {
        let a = r.unit() * 8.0;
        let got = e2m1_decode(e2m1_encode(a));
        let best = mags.iter().fold(f32::INFINITY, |b: f32, &m| {
            if (a.abs() - m).abs() < (a.abs() - b).abs() {
                m
            } else {
                b
            }
        });
        assert!(
            (got.abs() - best).abs() < 1e-6 || a.abs() > 6.0,
            "e2m1({a}) = {got}, nearest magnitude is {best}"
        );
    }
    assert_eq!(e2m1_decode(e2m1_encode(1e9)), 6.0, "saturate at +6");
    assert_eq!(e2m1_decode(e2m1_encode(-1e9)), -6.0);
    for c in 0u8..16 {
        assert_eq!(
            e2m1_encode(e2m1_decode(c)),
            c,
            "code {c} is not its own nearest"
        );
    }
}

#[test]
fn e8m0_decodes_to_exact_powers_of_two() {
    for c in 0u8..0xff {
        let v = e8m0_decode(c);
        assert!(v > 0.0 && v.is_finite(), "e8m0 code {c} decoded to {v}");
        // 2^(c-127), checked against repeated multiplication rather than the same shift.
        let mut want = 1.0f64;
        for _ in 0..(c as i32 - 127).abs() {
            want *= if c >= 127 { 2.0 } else { 0.5 };
        }
        assert!(
            (v as f64 / want - 1.0).abs() < 1e-12,
            "e8m0 code {c}: {v} != 2^({} )",
            c as i32 - 127
        );
    }
    assert!(e8m0_decode(0xff).is_nan());
}

#[test]
fn bf16_roundtrip_is_exact_for_every_pattern() {
    for b in 0u32..=0xffff {
        let v = bf16_decode(b as u16);
        if v.is_nan() {
            continue;
        }
        assert_eq!(
            bf16_encode(v),
            b as u16,
            "bf16 pattern {b:#06x} did not survive"
        );
    }
}

#[test]
fn fast_round_scale_is_the_smallest_power_of_two_that_covers_amax() {
    let mut r = NamedRng::new("scale");
    for _ in 0..10_000 {
        let amax = (r.unit() * 6.0).exp().abs().max(1e-8);
        let s = fast_round_scale(amax, 1.0 / FP8_MAX);
        assert!(s.is_finite() && s > 0.0);
        assert_eq!(
            s.to_bits() & 0x007f_ffff,
            0,
            "scale {s} is not a power of two"
        );
        assert!(
            amax / s <= FP8_MAX * 1.0000001,
            "scale {s} does not cover amax {amax}"
        );
        assert!(
            amax / (s / 2.0) > FP8_MAX,
            "scale {s} is not the SMALLEST that covers {amax}"
        );
    }
}

#[test]
fn act_quant_is_partial_and_block_sized() {
    // The property the whole KV path turns on: quantizing [0:n) leaves [n:] untouched
    // BIT-for-bit, and a different block size gives a different answer. Both directions,
    // because "it changed something" would pass for a whole-tensor quantizer too.
    let mut r = NamedRng::new("act-quant");
    let orig: Vec<f32> = (0..256).map(|_| r.unit() * 3.0).collect();
    // The partial-quantization property, asked the same way of both scale modes — and the
    // message names WHICH one, because "the tail was modified" without that sends the reader
    // to whichever `act_quant_inplace` call they read first.
    let tail_intact = |v: &[f32], which: &str| {
        assert_eq!(
            &v[192..],
            &orig[192..],
            "{which}: the un-quantized tail was modified"
        );
    };
    // The largest error `act_quant` introduced over the quantized prefix. One spelling
    // because the fp8/fp4 comparison below is only meaningful if both sides are measured
    // against the same reference in the same way; two folds is two chances to slice one of
    // them differently, and the assertion would still read as a statement about the formats.
    let max_err = |v: &[f32]| {
        v[..192]
            .iter()
            .zip(&orig[..192])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let mut a = orig.clone();
    act_quant_inplace(&mut a[..192], 64, true);
    tail_intact(&a, "ue8m0-rounded");
    assert!(
        a[..192].iter().zip(&orig[..192]).any(|(x, y)| x != y),
        "nothing was quantized"
    );

    let mut c = orig.clone();
    act_quant_inplace(&mut c[..192], 64, false);
    assert_ne!(a, c, "ue8m0 scale rounding made no difference");
    tail_intact(&c, "unrounded scale");

    // fp4 saturates far earlier, so the same input must survive fp8 and not fp4.
    let mut d = orig.clone();
    fp4_act_quant_inplace(&mut d, 32);
    let (err_fp8, err_fp4) = (max_err(&a), max_err(&d));
    assert!(
        err_fp4 > err_fp8,
        "fp4 ({err_fp4}) should be coarser than fp8 ({err_fp8})"
    );
    assert!(d.iter().all(|v| v.abs() <= FP4_MAX * 1024.0));
}

#[test]
fn act_quant_block_size_is_almost_invisible_under_ue8m0_scales() {
    // **A LIMITATION OF THIS ORACLE, asserted rather than assumed.**
    //
    // `scale_fmt: "ue8m0"` makes every block scale a pure power of two, and e4m3 is exactly
    // scale-invariant under powers of two for any value inside its normal range. So the
    // block SIZE changes an element's value only when the re-blocking pushes it into e4m3's
    // subnormals (|x| < 2^-6 * s) or up against its 448 ceiling -- which needs an in-block
    // dynamic range of roughly 2^13. Activation data does not have that, so on realistic
    // input a block-64 and a block-128 `act_quant` are BIT-IDENTICAL and no golden built on
    // them can tell the two apart.
    //
    // That is why `Defect::KvActQuantBlock128` is not in the matrix: it would assert a
    // detection the oracle cannot make. S2 must take the KV block size from the reference
    // (64) by reading it, not by expecting the gate to catch it.
    let mut r = NamedRng::new("blocksize");
    let ordinary: Vec<f32> = (0..192).map(|_| r.unit() * 3.0).collect();
    let (mut a, mut b) = (ordinary.clone(), ordinary.clone());
    act_quant_inplace(&mut a, 64, true);
    act_quant_inplace(&mut b, 128, true);
    assert_eq!(
        a, b,
        "the invisibility claim above is wrong -- re-derive it before trusting this"
    );

    // The narrow window where it IS visible: a block spanning ~2^25, so the coarse scale
    // flushes the small elements to zero.
    let mut wide = ordinary.clone();
    wide[3] = 3000.0;
    for v in &mut wide[64..128] {
        *v = 1e-4;
    }
    let (mut a, mut b) = (wide.clone(), wide.clone());
    act_quant_inplace(&mut a, 64, true);
    act_quant_inplace(&mut b, 128, true);
    assert_ne!(
        a[64..128],
        b[64..128],
        "even a 2^25 in-block range did not separate the two"
    );
    assert!(
        b[64..128].iter().all(|&v| v == 0.0),
        "block 128 should flush the tiny run to zero"
    );
    assert!(
        a[64..128].iter().all(|&v| v != 0.0),
        "block 64 should resolve the tiny run"
    );

    // ...and the same thing THROUGH the defect, so this test covers `KvActQuantBlock128`
    // rather than merely covering the function it would have perturbed. Listing a defect in
    // `targeted_defects()` satisfies the meta-guard with a name in a Vec; running it is what
    // makes the coverage real.
    let base = run(0, 12, Defect::None);
    let got = run(0, 12, Defect::KvActQuantBlock128);
    assert!(
        identical(&base.pre, &got.pre) && identical(&base.dec, &got.dec),
        "KvActQuantBlock128 moved a golden on real-shaped activations; if that is now true, \
         put it back in the matrix -- it was excluded because it provably could not"
    );
}

#[test]
fn hadamard_is_its_own_inverse() {
    // H/sqrt(n) is orthogonal and symmetric, so applying it twice is the identity. This
    // pins the transform's SHAPE without settling its basis ORDER — no property here can
    // decide that, because every candidate order is orthogonal and symmetric and so passes
    // this test identically. The order is settled SEPARATELY, against the package's own
    // documented contract, in `tests/v4_hadamard_basis.rs`; it was marked INFERRED here
    // until 2026-08-05.
    let mut r = NamedRng::new("hadamard");
    for n in [2usize, 8, 128] {
        let orig: Vec<f32> = (0..n).map(|_| r.unit()).collect();
        let mut v = orig.clone();
        hadamard_rotate(&mut v);
        assert!(
            v.iter().zip(&orig).any(|(a, b)| a != b),
            "n={n}: transform was a no-op"
        );
        hadamard_rotate(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "n={n}: not involutive ({a} vs {b})");
        }
        // Norm preservation is the other half of orthogonality.
        let mut w = orig.clone();
        hadamard_rotate(&mut w);
        let (no, nw): (f32, f32) = (
            orig.iter().map(|x| x * x).sum(),
            w.iter().map(|x| x * x).sum(),
        );
        assert!(
            (no - nw).abs() < 1e-4 * no.max(1.0),
            "n={n}: norm {no} -> {nw}"
        );
    }
}

#[test]
fn softplus_threshold_is_load_bearing() {
    // Below 20 the two forms agree; above it the naive form overflows f32 and the
    // sqrt-softplus router would produce inf weights and then NaN after renormalisation.
    for x in [-30.0f32, -1.0, 0.0, 5.0, 19.9] {
        assert!(
            (softplus(x) - (1.0 + x.exp()).ln()).abs() < 1e-6,
            "disagreement at {x}"
        );
    }
    assert_eq!(softplus(100.0), 100.0);
    assert!(
        (1.0f32 + 100.0f32.exp()).ln().is_infinite(),
        "the naive form is expected to blow up"
    );
}

#[test]
fn fp4_nibbles_unpack_low_first_as_convert_py_documents() {
    // `inference/convert.py::cast_e2m1fn_to_e4m3fn` unpacks with
    //   low = x & 0x0F; high = (x >> 4) & 0x0F
    //   stack([TABLE[low], TABLE[high]], dim=-1).flatten()
    // so the LOW nibble carries the even (lower) K index. That is a fact about the
    // checkpoint, not a convention we get to pick, and everything downstream of a wrong
    // choice is a permutation inside each scale group -- invisible to every summary
    // statistic. Pinned here against a hand-built byte.
    let w = WMat::Fp4 {
        rows: 1,
        cols: 32,
        // byte 0 = 0x21 -> low nibble 1 (=0.5) at k=0, high nibble 2 (=1.0) at k=1.
        w: {
            let mut v = vec![0u8; 16];
            v[0] = 0x21;
            v
        },
        // e8m0 code 127 = 2^0.
        s: vec![127u8],
    };
    let mut row = Vec::new();
    w.row(0, &mut row);
    assert_eq!(row[0], 0.5, "k=0 must come from the LOW nibble");
    assert_eq!(row[1], 1.0, "k=1 must come from the HIGH nibble");
    assert!(row[2..].iter().all(|&v| v == 0.0));
}

// =======================================================================================
// 2. the defect matrix
// =======================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Case {
    layer: usize,
    /// Prompt length. 5 fits the toy's `window_size = 8`; 12 does not, which is what makes
    /// the ring rotation and the ratio-8 compressor reachable at all.
    prompt: usize,
    phase: Phase,
}

const PROMPTS: [usize; 2] = [5, 12];
/// Enough decode steps that BOTH compress ratios complete a block from either prompt
/// length: `(start_pos + 1) % ratio == 0` needs 4 steps to be guaranteed for ratio 4 and 8
/// at both starts.
const DECODE_STEPS: usize = 4;

/// Both captures for one (layer, prompt, defect).
struct Run {
    pre: Capture,
    dec: Capture,
}

impl Run {
    fn of(&self, p: Phase) -> &Capture {
        match p {
            Phase::Prefill => &self.pre,
            Phase::Decode => &self.dec,
        }
    }
}

fn model() -> &'static (V4Config, ToyModel) {
    static M: OnceLock<(V4Config, ToyModel)> = OnceLock::new();
    M.get_or_init(|| {
        let cfg = V4Config::toy();
        let m = toy::build(&cfg);
        (cfg, m)
    })
}

fn fixed_ids(cfg: &V4Config, tag: &str, s: usize) -> Vec<u32> {
    let mut r = NamedRng::new(tag);
    (0..s).map(|_| r.below(cfg.vocab_size) as u32).collect()
}

fn run(layer: usize, prompt: usize, defect: Defect) -> Run {
    let (cfg, m) = model();
    let o = Oracle::new(cfg.clone(), defect);
    let lw = &m.layers[layer];
    let mut st = o.fresh_state(layer);
    let mut caps = [Capture::default(), Capture::default()];
    // One state, driven prefill-then-decode as the engine will drive it. The two captures
    // are separate so a phase can be asserted on its own; the STATE is shared, which is what
    // lets a prefill-only defect show up at decode and nowhere else.
    let steps = std::iter::once((0usize, "pre".to_string(), prompt, 0usize))
        .chain((0..DECODE_STEPS).map(|i| (1usize, format!("dec{i}"), 1, prompt + i)));
    for (slot, tag, s, start_pos) in steps {
        let mut h = residual_probe(cfg, &format!("h-{tag}"), s);
        let ids = fixed_ids(cfg, &format!("ids-{tag}"), s);
        let step = LayerCtx {
            lw,
            layer,
            s,
            start_pos,
            input_ids: &ids,
            phase: &tag,
        };
        o.run_layer(&step, &mut st, &mut h, &mut caps[slot]);
        // The head tail, on THIS layer's output. `bin/v4-oracle` deliberately refuses to do
        // that (see `HeadTailW`) because a logits vector taken at 4 of 43 layers is not a
        // quantity the model computes and would be misread as one. Here nothing is ever read
        // as a model quantity -- the whole file is a structural gate on the toy -- and the
        // composition buys the thing a standalone head fixture could not: every layer defect
        // is shown to REACH the logits, so the head tail is proved unable to mask one.
        o.head_tail(&m.head_tail, &h, s, &tag, &mut caps[slot]);
    }
    let [pre, dec] = caps;
    Run { pre, dec }
}

/// Goldens a defect MUST perturb, and goldens it MUST leave bit-identical.
///
/// `silent` is checked in every reachable case; `loud` requires at least one golden with
/// that suffix to differ (a decode case holds four steps and a compressor defect only
/// reaches the step that completes a block).
struct Expect {
    loud: &'static [&'static str],
    silent: &'static [&'static str],
}

/// Suffixes that are upstream of everything in the attention and are the strongest silent
/// evidence available: if a defect moves these, it is not the defect it claims to be.
const UPSTREAM: &[&str] = &[".in", ".attn_norm_out"];

/// Silent claims that no implementation can violate, so they must not count as evidence.
///
/// `run_layer` records `{tag}.in` from the `h` it was handed, before any defect-sensitive
/// code, and every driver here supplies a FIXED `h` per step on purpose. `*.in` is therefore
/// bit-identical for every defect in every case BY CONSTRUCTION. Harmless in a silent list
/// that carries other entries; fatal as a row's only entry, which is exactly the
/// "differs somewhere" row the meta-guard exists to forbid.
const TRIVIAL_SILENT: &[&str] = &[".in"];

fn expect(d: Defect) -> Option<Expect> {
    // `None` here means "covered by a targeted test below", and the meta-guard checks that
    // every such defect really is.
    let e = |loud, silent| Some(Expect { loud, silent });
    match d {
        Defect::None => None,

        Defect::SkipQkNorm | Defect::QkNormUsesQNormWeight => {
            e(&[".q"], &[".in", ".attn_norm_out", ".kv_entry"])
        }

        Defect::RopeAllDims | Defect::RopeFirstDims | Defect::RopeHalfSplit => {
            e(&[".q", ".kv_entry"], UPSTREAM)
        }
        Defect::RopeNoYarn | Defect::RopeYarnEverywhere | Defect::RopeBaseThetaEverywhere => {
            e(&[".q", ".kv_entry"], UPSTREAM)
        }

        Defect::SkipKvActQuant | Defect::KvActQuantWholeTensor | Defect::KvActQuantNoRoundScale => {
            e(&[".kv_entry"], &[".in", ".attn_norm_out", ".q"])
        }
        // See `act_quant_block_size_is_almost_invisible_under_ue8m0_scales`: this one is
        // measurably undetectable on realistic activations, so putting it in the matrix
        // would claim a resolution the oracle does not have.
        Defect::KvActQuantBlock128 => None,

        Defect::SkipAttnSink | Defect::AttnSinkNotMaxShifted => e(
            &[".attn_core_out"],
            &[".in", ".attn_norm_out", ".q", ".kv_entry", ".compressed"],
        ),
        Defect::PrefillRingWritesFirstWindow => e(
            &[".attn_core_out"],
            &[".in", ".attn_norm_out", ".q", ".kv_entry"],
        ),

        Defect::SkipOutputDerotation | Defect::OutputDerotationForward => e(
            &[".attn_derot"],
            &[".in", ".attn_norm_out", ".q", ".kv_entry", ".attn_core_out"],
        ),
        Defect::WoGroupsSplitHeadDim | Defect::WoGroupsInterleaved => e(
            &[".attn_out"],
            &[
                ".in",
                ".attn_norm_out",
                ".q",
                ".kv_entry",
                ".attn_core_out",
                ".attn_derot",
            ],
        ),

        Defect::CompressorNoOverlap
        | Defect::CompressorNoApe
        | Defect::CompressorRopeAtBlockEnd => e(
            &[".compressed"],
            &[".in", ".attn_norm_out", ".q", ".kv_entry"],
        ),
        Defect::IndexerNoRelu
        | Defect::IndexerNoFp4Quant
        | Defect::IndexerNoHadamard
        | Defect::IndexerNoWeights => e(
            // Only `.indexer_scores` here. `.compress_idxs` -- the SELECTION golden -- is
            // informative ONLY where `index_topk` truncates, which does not happen in every
            // case of the grid, so a matrix row demanding it would be false half the time.
            // `the_selection_golden_moves_when_topk_truncates` covers it where it bites.
            &[".indexer_scores"],
            // The indexer has its OWN compressor; the attention compressor's output must be
            // untouched. That separation is exactly what a port is likely to conflate.
            &[".in", ".attn_norm_out", ".q", ".kv_entry", ".compressed"],
        ),

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
        // `tests/v4_attn.rs` and `tests/v4_kernel.rs` and moving it would invalidate their
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
        | Defect::RouterNoScale => e(
            &[".router_weights"],
            &[".in", ".attn_norm_out", ".attn_out", ".ffn_norm_out"],
        ),
        Defect::HashRoutingIgnored => e(
            &[".router_indices"],
            &[".in", ".attn_norm_out", ".attn_out", ".ffn_norm_out"],
        ),
        Defect::RouteWeightAfterW2 | Defect::SharedExpertWeighted => e(
            &[".ffn_out"],
            &[".ffn_norm_out", ".router_weights", ".router_indices"],
        ),
        Defect::Fp4NibbleSwap => e(
            &[".ffn_out"],
            // Attention is fp8 and the shared expert is fp8; only the ROUTED experts are
            // fp4, so nothing before the MoE may move.
            &[
                ".in",
                ".attn_norm_out",
                ".q",
                ".kv_entry",
                ".attn_out",
                ".router_weights",
            ],
        ),

        // See `sinkhorn_has_converged_long_before_iteration_20`.
        Defect::SinkhornIterCountProbe => None,
        Defect::SinkhornCombTransposed | Defect::HcPostNoComb => e(
            &[".ffn_norm_out", ".out"],
            // `pre` comes straight from the mixes and never sees the Sinkhorn iterations,
            // so the attention half of the block is untouched by a combination-matrix bug.
            &[
                ".in",
                ".attn_norm_out",
                ".q",
                ".kv_entry",
                ".attn_core_out",
                ".attn_out",
            ],
        ),
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

        // `.hc_head_out` is the real silent half here, and it is violable: an implementation
        // that fused `hc_head` with the final norm -- the obvious single-kernel shortcut,
        // since both are reductions over the same row -- would move it.
        Defect::HeadNormSkipped | Defect::HeadNormNotBf16 | Defect::HeadNormOverAllTokens => {
            e(&[".final_norm_out", ".logits"], &[".hc_head_out"])
        }
        Defect::HeadLogitsFromFirstRow => e(&[".logits"], &[".hc_head_out", ".final_norm_out"]),

        // Mathematically INERT: `apply_rotary_emb` rotates adjacent pairs, so it PRESERVES
        // `q.square().mean(-1)`, and a scalar scale commutes with a rotation. The two orders
        // differ only in where the bf16 rounding lands. Keeping it in the matrix would
        // advertise a detection the gate does not have at any usable tolerance.
        Defect::QkNormAfterRope => None,
    }
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

fn cases() -> Vec<Case> {
    let (cfg, _) = model();
    let mut v = Vec::new();
    for layer in 0..cfg.n_layers {
        for prompt in PROMPTS {
            for phase in [Phase::Prefill, Phase::Decode] {
                v.push(Case {
                    layer,
                    prompt,
                    phase,
                });
            }
        }
    }
    v
}

fn matching<'a>(ds: &'a [Diff], suffix: &str) -> Vec<&'a Diff> {
    // The whole suffix scheme rests on the leading dot. Without it `"norm_out"` would match
    // `head.*.final_norm_out` AND `.attn_norm_out` AND `.ffn_norm_out`, and `"_out"` would
    // sweep up `.attn_out`, `.ffn_out` and `.out` together -- a silent widening that reads
    // exactly like a correct row. With every suffix dotted there is no collision in the
    // current name set. That was previously only a naming convention and a comment; this is
    // what actually enforces it.
    assert!(
        suffix.starts_with('.'),
        "golden suffix {suffix:?} must carry its leading dot"
    );
    ds.iter().filter(|d| d.name.ends_with(suffix)).collect()
}

/// A fingerprint of a whole capture, for the "no two defects are the same defect" check.
///
/// Hashes the SERIALIZED form rather than walking the fields, so it covers names and shapes
/// as well as values -- and so there is only one place that knows how a capture is laid out.
fn fingerprint(c: &Capture) -> u64 {
    let mut buf = Vec::new();
    GoldenSet::from_capture(Vec::new(), c.clone())
        .write(&mut buf)
        .unwrap();
    buf.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
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

#[test]
fn defect_matrix_is_bidirectional() {
    let baselines = baselines();
    let mut reached = 0usize;
    let mut silenced = 0usize;
    let mut propagated = 0usize;
    // (defect -> its fingerprint in every case). Two defects with the SAME vector are the
    // same defect wearing two names, and the matrix would then count one piece of evidence
    // twice. This is not hypothetical: `RopeNoYarn` and `RopeBaseThetaEverywhere` were
    // exactly that until both stopped selecting the base-theta table.
    let mut prints: Vec<(Defect, Vec<u64>)> = Vec::new();
    for &d in Defect::ALL {
        let Some(exp) = expect(d) else { continue };
        let mut mine = Vec::new();
        for c in cases() {
            let base = &baselines[&(c.layer, c.prompt)];
            let got = run(c.layer, c.prompt, d);
            mine.push(fingerprint(got.of(c.phase)));
            let ds = diff(base.of(c.phase), got.of(c.phase));
            if !reachable(d, &c, base) {
                assert!(
                    identical(base.of(c.phase), got.of(c.phase)),
                    "{d:?} is unreachable at {c:?} but changed {}",
                    first_change(&ds)
                );
                silenced += 1;
                continue;
            }
            for suffix in exp.loud {
                let hits = matching(&ds, suffix);
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
            for suffix in exp.silent {
                for h in matching(&ds, suffix) {
                    assert_eq!(
                        h.changed, 0,
                        "{d:?} at {c:?}: perturbed {} ({} of {} elements, rel {:.3e}), which is \
                         upstream of or beside what it claims to affect",
                        h.name, h.changed, h.total, h.rel
                    );
                }
            }
            // The head tail must not be able to MASK an upstream error. Wherever a defect
            // moved the layer's residual output, the logits have to move too -- otherwise a
            // per-layer golden could fail while the token that comes out is unchanged, or,
            // far worse, the reverse. Checked here rather than in its own test because `ds`
            // is already computed: a second pass over the grid would double this file's
            // runtime for evidence that is free at this point.
            // Paired by STEP, not any-to-any across the capture. A Decode capture holds four
            // steps, so an any/any check would be satisfied by a defect that moved `.out` at
            // `dec0` and the logits at `dec3` -- which is not the claim. Both names carry the
            // same `{tag}`, so the pairing is free.
            for h in matching(&ds, ".out").iter().filter(|h| h.changed > 0) {
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
                propagated += 1;
            }
            reached += 1;
        }
        for (other, theirs) in &prints {
            assert_ne!(
                &mine, theirs,
                "{d:?} and {other:?} compute the SAME thing in every case -- they are one \
                 defect wearing two names, and the matrix is double-counting its evidence"
            );
        }
        prints.push((d, mine));
    }
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

#[test]
fn the_comparator_itself_can_go_red() {
    // Proving a gate green is worthless until you have seen it red. Three ways a naive
    // comparator fails open: identical inputs, a one-ulp change, and a missing tensor.
    let a = run(2, 12, Defect::None);
    assert!(identical(&a.pre, &a.pre), "a capture must equal itself");

    let mut b = a.pre.clone();
    // By NAME: an index into `floats` is an ordering detail, and a reorder would silently
    // move this probe onto a different tensor (or an empty one, which `v[0]` would panic on).
    let (_, _, v) = b
        .floats
        .iter_mut()
        .find(|(n, _, _)| n.ends_with(".q"))
        .expect("a .q golden");
    v[0] = f32::from_bits(v[0].to_bits() ^ 1);
    assert!(!identical(&a.pre, &b), "a one-ulp change went undetected");

    let mut c = a.pre.clone();
    c.floats.remove(0);
    assert!(!identical(&a.pre, &c), "a DELETED golden read as agreement");

    let mut e = a.pre.clone();
    e.floats[0].2.push(0.0);
    assert!(!identical(&a.pre, &e), "a length change read as agreement");

    let mut f = a.pre.clone();
    let (_, shape, _) = &mut f.floats[0];
    *shape = vec![shape.iter().product()];
    assert!(
        !identical(&a.pre, &f),
        "a RESHAPE with identical values read as agreement"
    );
}

#[test]
fn the_safetensors_reader_rejects_malformed_headers() {
    // The one component that reads the 167 GB checkpoint, and the only thing that exercised
    // it was a binary that needs the checkpoint present. These are synthetic files.
    let dir = std::env::temp_dir().join(format!("v4-oracle-st-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, hdr: &str, body: &[u8]| {
        let mut v = (hdr.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(hdr.as_bytes());
        v.extend_from_slice(body);
        std::fs::write(dir.join(name), v).unwrap();
    };
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map":{"ok":"a.st","short":"b.st","backwards":"c.st","past_end":"d.st"}}"#,
    )
    .unwrap();
    write(
        "a.st",
        r#"{"ok":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &1.0f32
            .to_le_bytes()
            .iter()
            .chain(2.0f32.to_le_bytes().iter())
            .copied()
            .collect::<Vec<u8>>(),
    );
    // shape [2] F32 needs 8 bytes; the header claims 4.
    write(
        "b.st",
        r#"{"short":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        &[0u8; 4],
    );
    // data_offsets reversed -- `b - a` would WRAP in release.
    write(
        "c.st",
        r#"{"backwards":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
        &[0u8; 8],
    );
    // well-formed header, truncated body: the shard is still downloading.
    write(
        "d.st",
        r#"{"past_end":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[0u8; 4],
    );

    let ck = rivoli::v4oracle::weights::Checkpoint::open(&dir).unwrap();
    assert_eq!(ck.get("ok").unwrap().to_f32().unwrap(), vec![1.0, 2.0]);
    assert!(ck.has_prefix("o") && !ck.has_prefix("zz"));
    for bad in ["short", "backwards", "past_end"] {
        assert!(ck.get(bad).is_err(), "{bad} was accepted");
    }
    assert!(ck.get("absent").is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn window_topk_matches_the_reference_by_hand() {
    // `model.py::get_window_topk_idxs` (lines 260-271), transcribed by hand from the Python
    // rather than from the Rust. The middle branch (0 < start_pos < window_size - 1, which
    // pads with -1) is the one a port forgets, and the grid reaches it only incidentally.
    let w = rivoli::v4oracle::forward::window_topk;
    // prefill, seqlen 5, window 8: causal, row t attends [max(0,t-7), t], -1 beyond.
    assert_eq!(w(8, 5, 0)[0], vec![0, -1, -1, -1, -1]);
    assert_eq!(w(8, 5, 0)[4], vec![0, 1, 2, 3, 4]);
    // prefill, seqlen 12, window 8: row 11 sees positions 4..=11.
    assert_eq!(w(8, 12, 0)[11], vec![4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(w(8, 12, 0)[3], vec![0, 1, 2, 3, -1, -1, -1, -1]);
    // decode inside the first window: F.pad(arange(sp+1), (0, win-sp-1), value=-1).
    assert_eq!(w(8, 1, 2), vec![vec![0, 1, 2, -1, -1, -1, -1, -1]]);
    // decode past it: cat([arange(sp%win+1, win), arange(0, sp%win+1)]) -- oldest first.
    assert_eq!(w(8, 1, 9), vec![vec![2, 3, 4, 5, 6, 7, 0, 1]]);
}

#[test]
fn the_golden_file_survives_a_round_trip() {
    // The writer and the reader are the only two halves of the format, and nothing else in
    // the tree exercises the reader -- so without this they could disagree on a length
    // prefix forever and every other test would still pass. S2 loads goldens through
    // `GoldenSet::read`.
    let cap = run(2, 12, Defect::None).pre;
    let want = GoldenSet::from_capture(vec![("k".to_string(), "v".to_string())], cap.clone());
    let mut buf = Vec::new();
    want.write(&mut buf).unwrap();
    let got = GoldenSet::read(&mut buf.as_slice()).unwrap();
    assert_eq!(got.meta, want.meta);
    assert_eq!(got.floats, want.floats);
    assert_eq!(got.ints, want.ints);
    assert!(
        !got.floats.is_empty() && !got.ints.is_empty(),
        "the round trip carried nothing"
    );
    // ...and it must reject something that is not a golden file, or the magic is decoration.
    assert!(GoldenSet::read(&mut b"not a golden".as_slice()).is_err());
}

#[test]
fn a_duplicate_golden_name_is_rejected() {
    // `Capture::float` returns the FIRST match, so a duplicate name silently shadows every
    // later tensor of that name -- which is what a four-layer emit did before `run_layer`
    // namespaced by layer. Proving the guard fires is the point; a guard nobody has seen go
    // red is not a guard.
    let mut c = Capture::default();
    c.push("x", &[2], vec![1.0, 2.0]);
    assert!(std::panic::catch_unwind(move || c.push("x", &[2], vec![3.0, 4.0])).is_err());
    let mut c = Capture::default();
    assert!(
        std::panic::catch_unwind(move || c.push("y", &[3], vec![1.0])).is_err(),
        "shape/len"
    );
}

// =======================================================================================
// 3. targeted tests for the magnitude-gated defects
// =======================================================================================

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
// `torch_head_tail` below carries the same attribute. Its own doc argues nothing about
// rustfmt or jscpd -- it was authored packed for readability, before the interaction
// `build.rs` records was known -- so it is a precedent for the FORM, not a prior statement of
// this reason. The reason is stated here.
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
    // to be shaped like this.** Every other test in this file is SELF-RELATIVE -- a defected
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
    assert_eq!(
        good.bf16_sum(vanishing_terms.iter().copied()),
        1.0625,
        "f32 accumulation must keep 63 eighth-ulp terms; torch gives 1.0625"
    );
    assert_eq!(
        bad.bf16_sum(vanishing_terms.iter().copied()),
        1.0,
        "a running fold must round every one of them away"
    );
}

/// The head tail at toy dimensions, captured from CPU PyTorch on 2026-08-05.
///
/// **This is the ABSOLUTE gate on the head tail, and the reason it exists is the same reason
/// `bf16_reduction_matches_torch_and_not_a_running_fold` exists.** Every other head-tail
/// assertion in this file is SELF-RELATIVE -- a defected capture against an undefected one --
/// so a transliteration error shared by `hc_head` and its own defect variants cancels on both
/// sides and passes silently. That is exactly how the indexer's running-fold survived. If
/// `hc_head`'s rsqrt were scoped wrong in both the base path and `HeadHcRsqrtPerCopy`, or the
/// final norm's eps landed outside the sqrt in both, the other 30 tests would still be green.
///
/// Generated by transliterating model.py:709-716 (`hc_head`), :197-202 (`RMSNorm`) and
/// :731-740 (`ParallelHead`, `full_logits=False`) straight into torch calls -- not by
/// restating the Rust. Dimensions are deliberately tiny (`hc_mult` 4, `dim` 8, `s` 2,
/// `vocab` 5) so the whole fixture is readable, and `s = 2` so `x[:, -1]` is a real choice.
///
/// Dtypes follow the checkpoint, and the test asserts that they do: `H`, `NORM_W` and
/// `LM_HEAD` are bf16-representable because the checkpoint stores them bf16, while `FN` is
/// full f32 because `hc_head_fn` is F32 on disk -- so the mixes dot exercises f32 mantissas
/// that a bf16 fixture would have hidden.
#[rustfmt::skip]
mod torch_head_tail {
    pub const HC_MULT: usize = 4;
    pub const DIM: usize = 8;
    pub const S: usize = 2;
    pub const VOCAB: usize = 5;
    pub const H: &[f32] = &[
        -0.05517578, -0.28710938, -0.14746094, -0.0078125, -1.796875, -0.2578125, 1.921875,
        -0.23632813, 2.21875, -0.25, 0.53125, 0.578125, 0.9375, 0.52734375, -0.2109375,
        0.51953125, -0.54296875, 1.390625, 0.08544922, -0.21777344, 0.31054688, 0.66796875,
        0.39648438, -0.33203125, -1.609375, -0.12011719, -0.06689453, -0.13671875, 0.6328125,
        -0.5390625, 1.390625, -0.96484375, -0.9375, 2.0, 0.10986328, -0.33984375, -0.796875,
        -0.74609375, -0.62109375, -1.4375, 1.9765625, -0.06738281, -1.625, -0.38867188,
        -0.546875, 0.10986328, 1.40625, -0.703125, 0.11816406, -0.37109375, -0.0021514893,
        0.59765625, 1.546875, 0.33789063, -0.5078125, 0.17285156, -0.15820313, -2.265625,
        1.4375, -0.13574219, 2.71875, 1.4609375, -1.328125, -0.11425781
    ];
    pub const FN: &[f32] = &[
        0.440603, -0.66241807, -0.24728929, -0.13473506, 0.6656537, 0.9849724, 0.039184332,
        0.093170285, -0.13936427, -0.40719315, 0.01683204, -0.8812992, -0.8122814, 0.3493402,
        -0.08314953, 0.021388406, 1.1909121, -1.0306041, -0.11617905, 0.19755034, -0.012285116,
        -0.4201789, -0.16754936, 0.050195172, 0.5776188, -0.03540141, 0.5788603, -0.065610915,
        0.031231571, 0.89210266, 1.1707971, 0.44770864, -0.20019218, -0.7726562, 0.5258006,
        0.56279606, -0.5170723, 0.27976602, 0.16123955, -0.3741502, 0.96389794, -0.21715029,
        -0.4851133, -0.112384826, 0.017547777, -0.15141359, -0.6811052, -0.36952764,
        0.18584248, 0.19504797, -0.93740416, -0.13653417, -0.3739525, -0.7811996, 0.65430826,
        0.04110952, 0.7716346, -0.4844922, -0.37774345, 0.156617, -0.35271907, -0.92980933,
        0.04721228, -0.39673546, -0.22678863, -0.006180152, 0.9809605, 0.3741006, -0.31392458,
        -0.70439285, 0.09982608, -0.14046784, -0.9091174, -0.21177873, -1.3339996, 0.27655193,
        0.3117143, -0.21688096, -0.31344137, -0.60645866, -0.38015386, -0.49603975,
        -0.12530302, -0.63456744, -0.047734298, -0.17878447, -0.35324326, -0.33005747,
        -0.4325743, -0.43922296, -0.2842426, 0.046071798, -0.9829653, -0.50217706, 0.75698155,
        0.17544046, 0.5727456, 0.29420432, 1.1298723, -0.33078817, 0.27781653, -0.36367008,
        0.824201, 0.16153276, -0.15936637, -0.9153573, -0.026848827, -1.0524422, 0.14650427,
        0.40450522, 0.51041394, 0.730945, 0.005767892, -0.6457507, 0.40119773, -0.0018267105,
        0.25621048, 0.17447938, -0.64957625, -0.077361606, 1.3280479, 0.0018734756,
        -0.17327635, 0.46909046, -1.0369962, 0.3421054, 0.30048266, 0.3960198
    ];
    pub const BASE: &[f32] = &[
        -1.8474996, -1.169862, -0.7948558, -0.084306315
    ];
    pub const SCALE: &[f32] = &[
        0.9867983
    ];
    pub const NORM_W: &[f32] = &[
        1.4375, -0.52734375, -1.484375, -0.10253906, -1.546875, 0.859375, -1.6484375,
        -0.5859375
    ];
    pub const LM_HEAD: &[f32] = &[
        -0.8125, 1.421875, 0.37695313, 0.54296875, -0.5234375, 1.3046875, -0.67578125,
        -1.0703125, -0.48242188, 0.38085938, -2.484375, 2.4375, -1.2109375, -1.3203125,
        -0.828125, -0.25195313, 0.3671875, -0.40820313, -1.8203125, -0.53125, -1.1171875,
        0.39257813, 0.55859375, 2.3125, 0.72265625, -0.031982422, -0.7109375, 0.109375,
        0.58203125, 0.40625, -0.6328125, 0.18164063, -1.1640625, 0.56640625, 0.953125,
        0.052001953, 0.1328125, -0.7890625, -0.953125, -0.65234375
    ];
    pub const HC_HEAD_OUT: &[f32] = &[
        1.75, -0.033935547, 0.45507813, 0.453125, 0.83984375, 0.515625, -0.092285156,
        0.36914063, 0.046875, -0.17871094, 0.07763672, -0.025878906, 0.28515625, 0.14257813,
        -0.11425781, -0.096191406
    ];
    pub const FINAL_NORM_OUT: &[f32] = &[
        3.328125, 0.02355957, -0.890625, -0.061279297, -1.7109375, 0.5859375, 0.20117188,
        -0.28515625, 0.46875, 0.65625, -0.8046875, 0.018432617, -3.078125, 0.85546875, 1.3125,
        0.39257813
    ];
    pub const LOGITS: &[f32] = &[
        1.6791062, 3.4799843, 6.7748985, -1.3114338, -3.5308638
    ];
}

/// `kernels/common.hpp::wave_sum`, in the host: 32 strided partials, then the five-level
/// `__shfl_down` ladder. Not a model of the device -- a transcription of the reduction order
/// every V4 kernel actually uses, which is what makes the floor below a real number rather
/// than a guess about parallelism.
fn wave_sum(x: &[f32]) -> f32 {
    const WAVE: usize = 32;
    let mut p = [0.0f32; WAVE];
    for (lane, acc) in p.iter_mut().enumerate() {
        let (mut a, mut i) = (0.0f32, lane);
        while i < x.len() {
            a += x[i];
            i += WAVE;
        }
        *acc = a;
    }
    let mut off = WAVE / 2;
    while off > 0 {
        // Every lane reads the PRE-round value, as a wave does.
        let snap = p;
        for lane in 0..WAVE {
            if lane + off < WAVE {
                p[lane] = snap[lane] + snap[lane + off];
            }
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

#[test]
fn the_reassociation_floor_bounds_any_tolerance_these_goldens_can_have() {
    // `forward.rs`'s module doc says the residual disagreement from re-association "is the
    // floor on any tolerance built on these goldens". It has never been QUANTIFIED, and the
    // number turns out to overturn a conclusion this file drew from toy dimensions.
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
    // Two things follow, and the first CORRECTS this file. The guidance in
    // `the_head_mhc_rsqrt_is_load_bearing_in_every_case` read "the device-side head gate must
    // be bitwise". That was inferred from toy dimensions and is wrong at real ones: a CORRECT
    // wave-reduced kernel already differs from this oracle on ~0.08% of bf16 elements, so a
    // bitwise gate would reject correct code. It is exactly the extrapolation
    // v4-flash-port.md S3 item 16 warns against, and it was made here before being measured.
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
    let hcd = cfg.hc_dim();
    let mut w = NamedRng::new("reassoc-floor-weights");
    let hw = HeadTailW {
        hc_head_fn: (0..cfg.hc_mult * hcd).map(|_| w.unit() * 0.05).collect(),
        hc_head_base: (0..cfg.hc_mult).map(|_| w.unit()).collect(),
        hc_head_scale: vec![1.0 + w.unit() * 0.5],
        norm: (0..dim)
            .map(|_| bf16_decode(bf16_encode(1.0 + w.unit() * 0.3)))
            .collect(),
        lm_head: WMat::Dense {
            rows: cfg.vocab_size,
            cols: dim,
            v: vec![0.0; cfg.vocab_size * dim],
        },
    };
    let bf = |x: f32| bf16_decode(bf16_encode(x));

    // Both quantities vary a lot per draw, so a single sample decides nothing -- the first
    // version of this test drew once, found signal above noise, and would have reported the
    // opposite conclusion. What a gate needs is a THRESHOLD, so the question is whether the
    // two RANGES overlap: worst-case noise against best-case signal.
    let (mut noise_max, mut percopy_min, mut norsqrt_min) = (0.0f32, f32::INFINITY, f32::INFINITY);
    let mut flipped_total = 0usize;
    for draw in 0..24 {
        let mut r = NamedRng::new(&format!("reassoc-floor-draw-{draw}"));
        let h: Vec<f32> = (0..hcd).map(|_| bf(r.unit())).collect();
        let run = |d: Defect| {
            let mut cap = Capture::default();
            Oracle::new(cfg.clone(), d).head_tail(&hw, &h, 1, "floor", &mut cap);
            cap.float("head.floor.final_norm_out")
                .expect("final_norm_out")
                .to_vec()
        };
        let truth = run(Defect::None);
        percopy_min = percopy_min.min(rel_diff(&run(Defect::HeadHcRsqrtPerCopy), &truth));
        norsqrt_min = norsqrt_min.min(rel_diff(&run(Defect::HeadHcNoRsqrt), &truth));

        // NOISE: the same final RMSNorm, its 4096-term variance reduced by `wave_sum` instead
        // of sequentially. Both correct; only the order differs.
        let row: Vec<f32> = (0..dim).map(|_| bf(r.unit())).collect();
        let sq: Vec<f32> = row.iter().map(|v| v * v).collect();
        let norm_with = |var: f32| -> Vec<f32> {
            let rs = (var / dim as f32 + cfg.norm_eps).sqrt().recip();
            (0..dim).map(|i| bf(hw.norm[i] * (row[i] * rs))).collect()
        };
        let (sequential, waved) = (norm_with(sq.iter().sum::<f32>()), norm_with(wave_sum(&sq)));
        flipped_total += waved
            .iter()
            .zip(&sequential)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        noise_max = noise_max.max(rel_diff(&waved, &sequential));
    }
    println!(
        "worst noise {noise_max:.3e} ({flipped_total} bf16 flips over 24 draws); \
         best-case signal: PerCopy {percopy_min:.3e}, NoRsqrt {norsqrt_min:.3e}"
    );

    // A correct kernel is NOT bit-identical to this oracle at real dimensions. If that ever
    // stopped being true a bitwise device gate would be back on the table, so it is asserted.
    assert!(
        flipped_total > 0 && noise_max > 1e-4,
        "wave and sequential reduction agreed at dim {dim} ({flipped_total} flips, rel \
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

/// Is every value in `v` exactly representable in bf16?
fn all_bf16(v: &[f32]) -> bool {
    v.iter()
        .all(|&x| bf16_decode(bf16_encode(x)).to_bits() == x.to_bits())
}

#[test]
fn the_head_tail_matches_torch_absolutely() {
    use torch_head_tail as t;
    // The fixture must be adversarial in the way it claims: bf16 where the checkpoint stores
    // bf16, and NOT bf16 for `hc_head_fn`, which is F32 on disk. If `FN` were bf16-valued the
    // mixes dot would never exercise an f32 mantissa and this gate would be weaker than it
    // reads.
    assert!(
        all_bf16(t::H),
        "the residual stream must be bf16, as the reference stores it"
    );
    assert!(
        all_bf16(t::NORM_W) && all_bf16(t::LM_HEAD),
        "norm/head weights are bf16 on disk"
    );
    assert!(
        !all_bf16(t::FN),
        "hc_head_fn is F32 on disk; a bf16 fixture would not exercise it"
    );

    let cfg = V4Config {
        dim: t::DIM,
        vocab_size: t::VOCAB,
        ..V4Config::toy()
    };
    assert_eq!(
        cfg.hc_mult,
        t::HC_MULT,
        "the fixture was captured at hc_mult 4"
    );
    let o = Oracle::new(cfg.clone(), Defect::None);
    let hw = HeadTailW {
        hc_head_fn: t::FN.to_vec(),
        hc_head_base: t::BASE.to_vec(),
        hc_head_scale: t::SCALE.to_vec(),
        norm: t::NORM_W.to_vec(),
        lm_head: WMat::Dense {
            rows: t::VOCAB,
            cols: t::DIM,
            v: t::LM_HEAD.to_vec(),
        },
    };
    let mut cap = Capture::default();
    o.head_tail(&hw, t::H, t::S, "abs", &mut cap);

    // Bitwise, not `assert_close`. At these dimensions a sequential f32 dot reproduces
    // torch's reduction exactly -- checked when the fixture was captured, all 5 logits equal
    // -- so there is no re-association slack to hide behind, and the two bf16 tensors are
    // quantized coarsely enough that any real disagreement clears the format anyway.
    for (name, want) in [
        ("hc_head_out", t::HC_HEAD_OUT),
        ("final_norm_out", t::FINAL_NORM_OUT),
        ("logits", t::LOGITS),
    ] {
        let got = cap
            .float(&format!("head.abs.{name}"))
            .unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(got.len(), want.len(), "head.abs.{name} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "head.abs.{name}[{i}]: oracle {g:e} != torch {w:e} -- the transliteration \
                 disagrees with the reference, and no self-relative test in this file can see it"
            );
        }
    }
}

#[test]
fn the_head_tail_stores_bf16_where_the_reference_does_and_f32_where_it_does_not() {
    // The dtype boundary is the whole of the head tail's precision contract, and it is a live
    // choice on the device: `hc_head` returns `y.to(dtype)` and `RMSNorm.forward` returns
    // `(...).to(dtype)` -- both bf16 -- while `ParallelHead.forward` is `F.linear(x.float(),
    // weight)` on an f32 parameter and is never stored back. Getting the last one wrong by
    // reusing a bf16-storing GEMV is silent: the argmax usually survives it.
    let (cfg, _) = model();
    let r = run(3, 12, Defect::None);
    // Prefill (12 rows in) and a decode step (1 row in), because the claim that `logits` is
    // ONE row is only a claim at all when more than one row went in.
    for (cap, phase) in [(&r.pre, "pre"), (&r.dec, "dec0")] {
        let get = |n: &str| {
            cap.float(&format!("head.{phase}.{n}"))
                .unwrap_or_else(|| panic!("head.{phase}.{n} is missing"))
                .to_vec()
        };
        let (hc, nm, lg) = (get("hc_head_out"), get("final_norm_out"), get("logits"));
        // The `[s, dim]` shapes are NOT re-asserted here: `Capture::push` already refuses a
        // tensor whose length disagrees with its declared shape, and `head_tail` declares
        // them from the same `cfg`. `logits` IS asserted, because it is the one shape a
        // correct-looking implementation gets wrong -- `ParallelHead.forward` slices
        // `x[:, -1]`, so 12 rows in must still give ONE row of logits out. An implementation
        // that returned all rows would push `[s, vocab]` and satisfy `push` perfectly.
        assert_eq!(
            lg.len(),
            cfg.vocab_size,
            "head.{phase}.logits is not a single row"
        );
        for (n, v) in [
            ("hc_head_out", &hc),
            ("final_norm_out", &nm),
            ("logits", &lg),
        ] {
            assert!(
                v.iter().all(|x| x.is_finite()),
                "head.{phase}.{n} has a non-finite value"
            );
            assert!(v.iter().any(|&x| x != 0.0), "head.{phase}.{n} is all zero");
        }
        assert!(
            all_bf16(&hc),
            "head.{phase}.hc_head_out is not bf16 -- `y.to(dtype)` was lost"
        );
        assert!(
            all_bf16(&nm),
            "head.{phase}.final_norm_out is not bf16 -- RMSNorm's store was lost"
        );
        // The other direction, and the load-bearing one. `all_bf16` on all three would pass
        // an implementation that rounded everything; this forbids it. Not a probabilistic
        // hope: each logit is an f32 dot product of `dim` bf16 terms, so landing on a bf16
        // grid point needs its low 16 mantissa bits to come out zero. If this fires, the
        // logits are being stored through bf16 somewhere -- the head is wrong, not unlucky.
        assert!(
            !all_bf16(&lg),
            "every one of head.{phase}.logits' {} values is bf16-representable -- the logits \
             were stored through bf16, which the reference never does",
            lg.len()
        );
    }
}

#[test]
fn the_head_mhc_rsqrt_is_load_bearing_in_every_case() {
    // `expect()` gives these two no matrix row: everything they can reach, they reach, and
    // the only thing upstream is the layer stack, which `head_tail` cannot touch because it
    // borrows `h` immutably. So what is assertable here is what a matrix row would otherwise
    // have given -- that each fires in EVERY case, and that they are not one defect wearing
    // two names.
    //
    // **The magnitudes are a NEGATIVE result and are recorded as one.** The prediction before
    // measuring was that both would clear one bf16 ulp, on the reasoning that dropping or
    // mis-scoping an rsqrt of ~1.7 moves the mHC gates by ~10%. Measured over the 16-case
    // grid, 2026-08-05, smallest relative movement of any head golden:
    //
    //   HeadHcNoRsqrt       4.899e-3  (median .hc_head_out 7.9e-2, max 2.1e-1)
    //   HeadHcRsqrtPerCopy  4.284e-4  (median .logits 3.6e-3, max 7.6e-3)
    //
    // The bf16 ulp at 1.0 is 2^-7 = 7.8125e-3, so **the prediction failed for BOTH** -- their
    // worst cases sit 1.6x and 18x BELOW one ulp of the tensor's own scale. (`Diff.rel`
    // normalises by `max |b|`, so the ulp at the tensor's max is the right comparison.) An
    // earlier revision of this comment put the ulp at 2^-8 and concluded that `HeadHcNoRsqrt`
    // cleared it; that was wrong -- bf16 keeps 7 explicit mantissa bits, not 8.
    //
    // Both are still caught here, because this file compares BIT-EXACTLY at TOY dimensions.
    //
    // **Do not carry that to the device.** An earlier revision of this comment concluded
    // "the device-side head gate must be bitwise, not `assert_close`". That is WRONG at real
    // dimensions, and `the_reassociation_floor_bounds_any_tolerance_these_goldens_can_have`
    // measures why: at dim 4096 a CORRECT wave-reduced kernel already differs from this
    // oracle on ~0.08% of bf16 elements, so a bitwise gate would reject correct code -- and
    // `HeadHcRsqrtPerCopy`'s signal is the same ORDER as that noise, the two swapping places
    // between fixtures. The mHC denominator's scope has to be settled by reading the kernel,
    // or at small dimensions against `the_head_tail_matches_torch_absolutely`, and not by
    // comparing full-width activations at all.
    //
    // The bound below is a fixture-regression pin at roughly a third of the measured worst
    // case, NOT a claim that any gate resolves either defect under tolerance.
    //
    // Why it is small is itself the reason not to widen it away: the toy draws the residual
    // copies i.i.d., so their per-copy RMS agree to ~5%, and the per-copy rsqrt is then nearly
    // the joint one. Whether the trained model's `hc_post` spreads them further is not
    // something this fixture can answer.
    let mut prints = Vec::new();
    for d in [Defect::HeadHcNoRsqrt, Defect::HeadHcRsqrtPerCopy] {
        let mut worst = f32::INFINITY;
        let mut fps = Vec::new();
        for c in cases() {
            let base = run(c.layer, c.prompt, Defect::None);
            let got = run(c.layer, c.prompt, d);
            let ds = diff(base.of(c.phase), got.of(c.phase));
            for suffix in [".hc_head_out", ".final_norm_out", ".logits"] {
                let hits = matching(&ds, suffix);
                assert!(
                    !hits.is_empty(),
                    "{d:?} at {c:?}: no *{suffix} golden exists"
                );
                assert!(
                    hits.iter().all(|h| h.changed > 0),
                    "{d:?} at {c:?}: left a *{suffix} golden bit-identical"
                );
                worst = hits.iter().fold(worst, |m, h| m.min(h.rel));
            }
            // The layer stack is untouched. Trivially true today -- `head_tail` takes
            // `&[f32]` -- and kept as a tripwire for the day someone gives it `&mut`.
            for h in matching(&ds, ".attn_out")
                .iter()
                .chain(matching(&ds, ".out").iter())
            {
                assert_eq!(
                    h.changed, 0,
                    "{d:?} at {c:?}: reached {} in the layer",
                    h.name
                );
            }
            fps.push(fingerprint(got.of(c.phase)));
        }
        println!("{d:?}: smallest relative movement over the grid = {worst:.3e}");
        assert!(
            worst > 1.4e-4,
            "{d:?}'s smallest movement fell to {worst:.3e}, below the 1.4e-4 pin (a third of \
             the 4.284e-4 measured on 2026-08-05). The fixture has become less able to \
             separate this defect, not more -- re-measure before moving the pin"
        );
        prints.push((d, fps));
    }
    assert_ne!(
        prints[0].1, prints[1].1,
        "the two head-rsqrt defects compute the same thing"
    );
}

fn targeted_defects() -> Vec<Defect> {
    vec![
        Defect::SwigluUnclamped,
        Defect::SwigluClampGateBothSides,
        Defect::RouterNoSoftplusThreshold,
        Defect::KvActQuantBlock128,
        Defect::SinkhornIterCountProbe,
        Defect::QkNormAfterRope,
        Defect::HcPreNoRsqrt,
        Defect::NoBf16Rounding,
        Defect::HeadHcNoRsqrt,
        Defect::HeadHcRsqrtPerCopy,
        Defect::IndexerBf16RunningSum,
    ]
}

/// Run one routed expert at a given input scale, returning `(output, clamped_count)`.
fn expert_at_scale(defect: Defect, scale: f32) -> (Vec<f32>, usize) {
    let (cfg, m) = model();
    let o = Oracle::new(cfg.clone(), defect);
    let mut r = NamedRng::new("swiglu-probe");
    let x: Vec<f32> = (0..cfg.dim)
        .map(|_| bf16_decode(bf16_encode(r.unit() * scale)))
        .collect();
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
    let ids = fixed_ids(cfg, "ids-pre", 5);
    let drive = |c: &V4Config, d: Defect| {
        let o = Oracle::new(c.clone(), d);
        let mut h = residual_probe(cfg, "h-pre", 5);
        common::prefill_capture(&o, &m.layers[0], 0, &ids, &mut h)
    };
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
                phase: "g",
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
