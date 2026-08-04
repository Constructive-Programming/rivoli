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

use rivoli::v4oracle::forward::{Capture, Defect, Oracle, Step};
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
    (0u8..=0x7e).map(|c| (c, e4m3_decode(c))).filter(|(_, v)| v.is_finite()).collect()
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
        (0x01, 1.0 / 512.0),          // smallest subnormal: quantum 2^-9
        (0x07, 7.0 / 512.0),          // largest subnormal
        (0x08, 1.0 / 64.0),           // smallest normal: 2^-6
        (0x38, 1.0),                  // exp 7 == bias, mantissa 0
        (0x3f, 1.875),                // exp 7, mantissa 7 -> 1 + 7/8
        (0x78, 256.0),                // exp 15, mantissa 0
        (0x7e, 448.0),                // largest finite: 1.75 * 2^8
    ] {
        assert_eq!(e4m3_decode(code), want, "e4m3 code {code:#04x}");
        assert_eq!(e4m3_decode(code | 0x80), -want, "e4m3 code {:#04x}", code | 0x80);
    }
    assert!(e4m3_decode(0x7f).is_nan() && e4m3_decode(0xff).is_nan(), "S.1111.111 is NaN");

    // bf16 is f32's top 16 bits; likewise pinned by hand rather than by round-tripping.
    for (bits, want) in
        [(0x0000u16, 0.0f32), (0x3f80, 1.0), (0x4000, 2.0), (0xbf80, -1.0), (0x3f00, 0.5)]
    {
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
            got, want,
            "e4m3_encode({a:e}) = {got:#04x} ({}) but nearest-ties-even is {want:#04x} ({})",
            e4m3_decode(got),
            e4m3_decode(want)
        );
        checked += 1;
    }
    // The random probes at +/-512 are half out of range, so the reachable count is well
    // below the number generated. Asserted so the sweep cannot quietly shrink to nothing.
    assert!(checked > 30_000, "only {checked} probes reached the assertion");
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
    for (probe, want) in
        [(0.25, 0.0), (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0)]
    {
        assert_eq!(e2m1_decode(e2m1_encode(probe)), want, "e2m1 tie at {probe}");
        assert_eq!(e2m1_decode(e2m1_encode(-probe)), -want, "e2m1 tie at -{probe}");
    }
    let mags = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut r = NamedRng::new("e2m1-sweep");
    for _ in 0..20_000 {
        let a = r.unit() * 8.0;
        let got = e2m1_decode(e2m1_encode(a));
        let best = mags.iter().fold(f32::INFINITY, |b: f32, &m| {
            if (a.abs() - m).abs() < (a.abs() - b).abs() { m } else { b }
        });
        assert!(
            (got.abs() - best).abs() < 1e-6 || a.abs() > 6.0,
            "e2m1({a}) = {got}, nearest magnitude is {best}"
        );
    }
    assert_eq!(e2m1_decode(e2m1_encode(1e9)), 6.0, "saturate at +6");
    assert_eq!(e2m1_decode(e2m1_encode(-1e9)), -6.0);
    for c in 0u8..16 {
        assert_eq!(e2m1_encode(e2m1_decode(c)), c, "code {c} is not its own nearest");
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
        assert_eq!(bf16_encode(v), b as u16, "bf16 pattern {b:#06x} did not survive");
    }
}

#[test]
fn fast_round_scale_is_the_smallest_power_of_two_that_covers_amax() {
    let mut r = NamedRng::new("scale");
    for _ in 0..10_000 {
        let amax = (r.unit() * 6.0).exp().abs().max(1e-8);
        let s = fast_round_scale(amax, 1.0 / FP8_MAX);
        assert!(s.is_finite() && s > 0.0);
        assert_eq!(s.to_bits() & 0x007f_ffff, 0, "scale {s} is not a power of two");
        assert!(amax / s <= FP8_MAX * 1.0000001, "scale {s} does not cover amax {amax}");
        assert!(amax / (s / 2.0) > FP8_MAX, "scale {s} is not the SMALLEST that covers {amax}");
    }
}

#[test]
fn act_quant_is_partial_and_block_sized() {
    // The property the whole KV path turns on: quantizing [0:n) leaves [n:] untouched
    // BIT-for-bit, and a different block size gives a different answer. Both directions,
    // because "it changed something" would pass for a whole-tensor quantizer too.
    let mut r = NamedRng::new("act-quant");
    let orig: Vec<f32> = (0..256).map(|_| r.unit() * 3.0).collect();
    let mut a = orig.clone();
    act_quant_inplace(&mut a[..192], 64, true);
    assert_eq!(&a[192..], &orig[192..], "the un-quantized tail was modified");
    assert!(a[..192].iter().zip(&orig[..192]).any(|(x, y)| x != y), "nothing was quantized");

    let mut c = orig.clone();
    act_quant_inplace(&mut c[..192], 64, false);
    assert_ne!(a, c, "ue8m0 scale rounding made no difference");
    assert_eq!(&c[192..], &orig[192..], "the un-quantized tail was modified");

    // fp4 saturates far earlier, so the same input must survive fp8 and not fp4.
    let mut d = orig.clone();
    fp4_act_quant_inplace(&mut d, 32);
    let err_fp8 = a[..192].iter().zip(&orig[..192]).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    let err_fp4 = d[..192].iter().zip(&orig[..192]).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    assert!(err_fp4 > err_fp8, "fp4 ({err_fp4}) should be coarser than fp8 ({err_fp8})");
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
    assert_eq!(a, b, "the invisibility claim above is wrong -- re-derive it before trusting this");

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
    assert_ne!(a[64..128], b[64..128], "even a 2^25 in-block range did not separate the two");
    assert!(b[64..128].iter().all(|&v| v == 0.0), "block 128 should flush the tiny run to zero");
    assert!(a[64..128].iter().all(|&v| v != 0.0), "block 64 should resolve the tiny run");

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
    // pins the transform's SHAPE without settling its basis ORDER, which is the part that
    // is INFERRED (see `numerics::hadamard_rotate`) and which no property here can decide.
    let mut r = NamedRng::new("hadamard");
    for n in [2usize, 8, 128] {
        let orig: Vec<f32> = (0..n).map(|_| r.unit()).collect();
        let mut v = orig.clone();
        hadamard_rotate(&mut v);
        assert!(v.iter().zip(&orig).any(|(a, b)| a != b), "n={n}: transform was a no-op");
        hadamard_rotate(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "n={n}: not involutive ({a} vs {b})");
        }
        // Norm preservation is the other half of orthogonality.
        let mut w = orig.clone();
        hadamard_rotate(&mut w);
        let (no, nw): (f32, f32) =
            (orig.iter().map(|x| x * x).sum(), w.iter().map(|x| x * x).sum());
        assert!((no - nw).abs() < 1e-4 * no.max(1.0), "n={n}: norm {no} -> {nw}");
    }
}

#[test]
fn softplus_threshold_is_load_bearing() {
    // Below 20 the two forms agree; above it the naive form overflows f32 and the
    // sqrt-softplus router would produce inf weights and then NaN after renormalisation.
    for x in [-30.0f32, -1.0, 0.0, 5.0, 19.9] {
        assert!((softplus(x) - (1.0 + x.exp()).ln()).abs() < 1e-6, "disagreement at {x}");
    }
    assert_eq!(softplus(100.0), 100.0);
    assert!((1.0f32 + 100.0f32.exp()).ln().is_infinite(), "the naive form is expected to blow up");
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

/// A fixed, bf16-representable residual stream. Fixed so that a defect at prefill cannot
/// change the decode step's INPUT: only the layer's own cached state carries the defect
/// forward, which is what makes "this case is unaffected" a statement about the defect
/// rather than about propagation.
fn fixed_h(cfg: &V4Config, tag: &str, s: usize) -> Vec<f32> {
    let mut r = NamedRng::new(tag);
    (0..s * cfg.hc_mult * cfg.dim).map(|_| bf16_decode(bf16_encode(r.unit()))).collect()
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
        let mut h = fixed_h(cfg, &format!("h-{tag}"), s);
        let ids = fixed_ids(cfg, &format!("ids-{tag}"), s);
        let step = Step { lw, layer, s, start_pos, input_ids: &ids, phase: &tag };
        o.run_layer(&step, &mut st, &mut h, &mut caps[slot]);
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

        Defect::SkipAttnSink | Defect::AttnSinkNotMaxShifted => {
            e(&[".attn_core_out"], &[".in", ".attn_norm_out", ".q", ".kv_entry", ".compressed"])
        }
        Defect::PrefillRingWritesFirstWindow => {
            e(&[".attn_core_out"], &[".in", ".attn_norm_out", ".q", ".kv_entry"])
        }

        Defect::SkipOutputDerotation | Defect::OutputDerotationForward => {
            e(&[".attn_derot"], &[".in", ".attn_norm_out", ".q", ".kv_entry", ".attn_core_out"])
        }
        Defect::WoGroupsSplitHeadDim | Defect::WoGroupsInterleaved => e(
            &[".attn_out"],
            &[".in", ".attn_norm_out", ".q", ".kv_entry", ".attn_core_out", ".attn_derot"],
        ),

        Defect::CompressorNoOverlap
        | Defect::CompressorNoApe
        | Defect::CompressorRopeAtBlockEnd => {
            e(&[".compressed"], &[".in", ".attn_norm_out", ".q", ".kv_entry"])
        }
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

        Defect::SwigluUnclamped
        | Defect::SwigluClampGateBothSides
        | Defect::RouterNoSoftplusThreshold => None,

        Defect::RouterSoftmax
        | Defect::RouterBiasedWeights
        | Defect::RouterNoRenorm
        | Defect::RouterNoScale => {
            e(&[".router_weights"], &[".in", ".attn_norm_out", ".attn_out", ".ffn_norm_out"])
        }
        Defect::HashRoutingIgnored => {
            e(&[".router_indices"], &[".in", ".attn_norm_out", ".attn_out", ".ffn_norm_out"])
        }
        Defect::RouteWeightAfterW2 | Defect::SharedExpertWeighted => {
            e(&[".ffn_out"], &[".ffn_norm_out", ".router_weights", ".router_indices"])
        }
        Defect::Fp4NibbleSwap => e(
            &[".ffn_out"],
            // Attention is fp8 and the shared expert is fp8; only the ROUTED experts are
            // fp4, so nothing before the MoE may move.
            &[".in", ".attn_norm_out", ".q", ".kv_entry", ".attn_out", ".router_weights"],
        ),

        // See `sinkhorn_has_converged_long_before_iteration_20`.
        Defect::SinkhornOneFewerIter => None,
        Defect::SinkhornCombTransposed | Defect::HcPostNoComb => e(
            &[".ffn_norm_out", ".out"],
            // `pre` comes straight from the mixes and never sees the Sinkhorn iterations,
            // so the attention half of the block is untouched by a combination-matrix bug.
            &[".in", ".attn_norm_out", ".q", ".kv_entry", ".attn_core_out", ".attn_out"],
        ),
        // Both of these reach EVERY golden downstream of `hc_pre` -- which is all of them --
        // so neither has a silent half to declare, and `.in` (fixed by the driver) would be
        // a claim no implementation could violate. Demoted to targeted tests, the same way
        // `KvActQuantBlock128` and `SinkhornOneFewerIter` were.
        Defect::HcPreNoRsqrt | Defect::NoBf16Rounding => None,

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

        _ => true,
    }
}

fn cases() -> Vec<Case> {
    let (cfg, _) = model();
    let mut v = Vec::new();
    for layer in 0..cfg.n_layers {
        for prompt in PROMPTS {
            for phase in [Phase::Prefill, Phase::Decode] {
                v.push(Case { layer, prompt, phase });
            }
        }
    }
    v
}

fn matching<'a>(ds: &'a [Diff], suffix: &str) -> Vec<&'a Diff> {
    ds.iter().filter(|d| d.name.ends_with(suffix)).collect()
}

/// A fingerprint of a whole capture, for the "no two defects are the same defect" check.
///
/// Hashes the SERIALIZED form rather than walking the fields, so it covers names and shapes
/// as well as values -- and so there is only one place that knows how a capture is laid out.
fn fingerprint(c: &Capture) -> u64 {
    let mut buf = Vec::new();
    GoldenSet::from_capture(Vec::new(), c.clone()).write(&mut buf).unwrap();
    buf.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn first_change(ds: &[Diff]) -> String {
    ds.iter()
        .find(|d| d.changed > 0)
        .map_or_else(|| "nothing".to_string(), |d| format!("{} ({} elements)", d.name, d.changed))
}

/// The undefected run for every (layer, prompt) in the grid.
fn baselines() -> std::collections::HashMap<(usize, usize), Run> {
    let mut m = std::collections::HashMap::new();
    for c in cases() {
        m.entry((c.layer, c.prompt)).or_insert_with(|| run(c.layer, c.prompt, Defect::None));
    }
    m
}

#[test]
fn defect_matrix_is_bidirectional() {
    let baselines = baselines();
    let mut reached = 0usize;
    let mut silenced = 0usize;
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
    assert!(reached > 200, "only {reached} reachable (defect, case) pairs were asserted");
    assert!(silenced > 40, "only {silenced} unreachable pairs -- too little silent evidence");
}

#[test]
fn every_defect_carries_both_halves_of_its_claim() {
    // The guard against the matrix rotting into "differs everywhere", which proves nothing
    // about the gate's resolution.
    let baselines = baselines();
    let targeted = targeted_defects();
    for d in Defect::breakages() {
        let Some(exp) = expect(d) else {
            assert!(targeted.contains(&d), "{d:?} has no matrix row and no targeted test");
            continue;
        };
        assert!(!targeted.contains(&d), "{d:?} is covered twice; pick one");
        assert!(!exp.loud.is_empty(), "{d:?} declares nothing it must perturb");
        let n_reach = cases().iter().filter(|c| reachable(d, c, &baselines[&(c.layer, c.prompt)])).count();
        assert!(n_reach > 0, "{d:?} is unreachable in every case, so nothing tests it");
        let real_silent = exp.silent.iter().filter(|s| !TRIVIAL_SILENT.contains(s)).count();
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
    assert!(classes.contains(&4), "no ratio-4 layer (the only kind with an Indexer)");
    assert!(
        classes.iter().any(|&r| r != 0 && r != 4),
        "no compressed layer WITHOUT an indexer -- layer 0 and layer 2 alone would leave the \
         ratio-128 class untested, and that class is 20 of the model's 43 layers"
    );
    for (l, &ratio) in classes.iter().enumerate() {
        let r = run(l, 12, Defect::None);
        let has_idx = r.pre.float(&format!("L{l}.pre.indexer_scores")).is_some();
        let has_comp = r.pre.float(&format!("L{l}.pre.compressed")).is_some();
        assert_eq!(has_idx, ratio == 4, "layer {l} (ratio {ratio}) indexer presence is wrong");
        assert_eq!(has_comp, ratio != 0, "layer {l} (ratio {ratio}) compressor presence is wrong");
        assert!(r.pre.int(&format!("L{l}.pre.router_indices")).is_some(), "layer {l} recorded no routing");
        // and the goldens are not degenerate
        let out = r.pre.float(&format!("L{l}.pre.out")).expect("L{l}.pre.out");
        assert!(out.iter().all(|v| v.is_finite()), "layer {l} produced non-finite output");
        assert!(out.iter().any(|&v| v != 0.0), "layer {l} produced an all-zero output");
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
    let (_, _, v) = b.floats.iter_mut().find(|(n, _, _)| n.ends_with(".q")).expect("a .q golden");
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
    assert!(!identical(&a.pre, &f), "a RESHAPE with identical values read as agreement");
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
    write("a.st", r#"{"ok":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, &1.0f32
        .to_le_bytes()
        .iter()
        .chain(2.0f32.to_le_bytes().iter())
        .copied()
        .collect::<Vec<u8>>());
    // shape [2] F32 needs 8 bytes; the header claims 4.
    write("b.st", r#"{"short":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, &[0u8; 4]);
    // data_offsets reversed -- `b - a` would WRAP in release.
    write("c.st", r#"{"backwards":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#, &[0u8; 8]);
    // well-formed header, truncated body: the shard is still downloading.
    write("d.st", r#"{"past_end":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, &[0u8; 4]);

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
    let want = GoldenSet::from_capture(
        vec![("k".to_string(), "v".to_string())],
        cap.clone(),
    );
    let mut buf = Vec::new();
    want.write(&mut buf).unwrap();
    let got = GoldenSet::read(&mut buf.as_slice()).unwrap();
    assert_eq!(got.meta, want.meta);
    assert_eq!(got.floats, want.floats);
    assert_eq!(got.ints, want.ints);
    assert!(!got.floats.is_empty() && !got.ints.is_empty(), "the round trip carried nothing");
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
    assert!(std::panic::catch_unwind(move || c.push("y", &[3], vec![1.0])).is_err(), "shape/len");
}

#[test]
fn defect_list_has_no_duplicates() {
    // `Defect::ALL` is hand-maintained (see its doc). This catches the half of the mistake
    // that is catchable: a variant listed twice, which would double-count its evidence in
    // the matrix. A variant MISSING from the list is caught by nothing here -- only by
    // `expect()`'s exhaustive match forcing the author to classify it.
    let mut seen = std::collections::HashSet::new();
    for &d in Defect::ALL {
        assert!(seen.insert(d), "{d:?} appears twice in Defect::ALL");
    }
    assert!(Defect::breakages().count() + 1 == Defect::ALL.len());
}

// =======================================================================================
// 3. targeted tests for the magnitude-gated defects
// =======================================================================================

fn targeted_defects() -> Vec<Defect> {
    vec![
        Defect::SwigluUnclamped,
        Defect::SwigluClampGateBothSides,
        Defect::RouterNoSoftplusThreshold,
        Defect::KvActQuantBlock128,
        Defect::SinkhornOneFewerIter,
        Defect::QkNormAfterRope,
        Defect::HcPreNoRsqrt,
        Defect::NoBf16Rounding,
    ]
}

/// Run one routed expert at a given input scale, returning `(output, clamped_count)`.
fn expert_at_scale(defect: Defect, scale: f32) -> (Vec<f32>, usize) {
    let (cfg, m) = model();
    let o = Oracle::new(cfg.clone(), defect);
    let mut r = NamedRng::new("swiglu-probe");
    let x: Vec<f32> = (0..cfg.dim).map(|_| bf16_decode(bf16_encode(r.unit() * scale))).collect();
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
        assert_eq!(cut > 0, want_truncation, "prompt {prompt}: {cut} truncating query rows");
        let want = selected(&base);
        assert!(!want.is_empty(), "prompt {prompt}: no selection golden at all");
        let mut movers = Vec::new();
        for d in [Defect::IndexerNoHadamard, Defect::IndexerNoRelu, Defect::IndexerNoWeights] {
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
    let (order, rounding) = (worst(Defect::QkNormAfterRope), worst(Defect::NoBf16Rounding));
    assert!(rounding > 0.0, "NoBf16Rounding moved nothing, so there is no yardstick");
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
                ds.iter().filter(|x| x.name.ends_with(suffix)).any(|x| x.changed > 0),
                "{d:?} left *{suffix} untouched"
            );
        }
        assert!(
            ds.iter().filter(|x| x.name.ends_with(".in")).all(|x| x.changed == 0),
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
    let (cfg, m) = model();
    let ids = fixed_ids(cfg, "ids-pre", 5);
    let drive = |c: &V4Config, d: Defect| {
        let o = Oracle::new(c.clone(), d);
        let mut st = o.fresh_state(0);
        let mut h = fixed_h(cfg, "h-pre", 5);
        let mut cap = Capture::default();
        let step =
            Step { lw: &m.layers[0], layer: 0, s: 5, start_pos: 0, input_ids: &ids, phase: "pre" };
        o.run_layer(&step, &mut st, &mut h, &mut cap);
        cap
    };
    let full = drive(cfg, Defect::None);
    assert!(
        identical(&full, &drive(cfg, Defect::SinkhornOneFewerIter)),
        "19 and 20 iterations disagree -- the convergence claim above is wrong, and \
         `SinkhornOneFewerIter` belongs back in the matrix"
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
        assert_eq!(n_cold, 0, "the probe was supposed to stay inside +/-10 ({d:?})");
        assert_eq!(cold_ref, cold_def, "{d:?} moved an expert whose activations never clamp");

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
    let x: Vec<f32> = fixed_h(cfg, "gate-x", 5).into_iter().take(5 * cfg.dim).collect();

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
            let step = Step { lw: &layer_w, layer, s: 5, start_pos: 0, input_ids: &ids, phase: "g" };
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
