//! **Layer 1 of the V4-oracle evidence: the codecs.**
//!
//! Every fp8, fp4 and bf16 pattern, each against an independent brute-force reference rather
//! than against itself. Split out of `v4_oracle.rs` on 2026-08-15 when the 800-line ceiling
//! landed; `v4_oracle.rs`'s header still carries the orientation for the whole family, and
//! the shared toy driver is `common/oracle_probe.rs`.
//!
//! Nothing here needs the grid except [`act_quant_block_size_is_almost_invisible_under_ue8m0_scales`],
//! which ends by driving the defect it argues is undetectable — the codec claim and the
//! matrix-exclusion it licenses are one question and stay in one place.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_oracles::golden::identical;
use rivoli_oracles::v4oracle::forward::Defect;
use rivoli_oracles::v4oracle::numerics::{
    FP4_MAX, FP8_MAX, act_quant_inplace, bf16_decode, bf16_encode, e2m1_decode, e2m1_encode,
    e4m3_decode, e4m3_encode, e8m0_decode, fast_round_scale, fp4_act_quant_inplace,
    hadamard_rotate, softplus,
};
use rivoli_oracles::v4oracle::weights::{NamedRng, WMat};

#[path = "common/oracle_probe.rs"]
mod oracle_probe;
use oracle_probe::run;

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
        if beats(a, v, best) {
            best = (c, v);
        }
    }
    best.0
}

/// The whole rounding rule in one place: strictly closer wins, and an exact tie goes to the
/// code with the even mantissa — so the incumbent loses a tie when ITS low bit is set.
fn beats(a: f32, cand: f32, best: (u8, f32)) -> bool {
    let (dn, db) = ((a - cand).abs(), (a - best.1).abs());
    let tie_to_even = dn == db && (best.0 & 1) != 0;
    dn < db || tie_to_even
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
    // The range the sweep skipped: e4m3 has no infinities, so encoding past 448 must CLAMP to
    // the largest finite code rather than run off the top of the format.
    let (up, down) = (e4m3_encode(1e30), e4m3_encode(-1e30));
    assert_eq!(up, 0x7e, "saturate, not overflow");
    assert_eq!(down, 0xfe);
    let nans = [e4m3_decode(0x7f), e4m3_decode(0xff)];
    assert!(nans.iter().all(|v| v.is_nan()));
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
        // Both halves of "smallest that covers": this scale fits amax inside fp8's range, and
        // the next one down does not.
        let (covered, halved) = (amax / s, amax / (s / 2.0));
        assert!(
            covered <= FP8_MAX * 1.0000001,
            "scale {s} does not cover amax {amax}"
        );
        assert!(
            halved > FP8_MAX,
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
    // documented contract, in `tests/hadamard_basis.rs`; it was marked INFERRED here
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
