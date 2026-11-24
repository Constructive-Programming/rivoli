//! inkling's expert dequant on the device — MXFP8 and NVFP4 — scored BIT-FOR-BIT against the
//! first-party CPU oracles `rivoli_artifact::inkling_dequant::{Mxfp8, Nvfp4}`. **No C twin** (owner
//! direction 2026-11-24): new-model kernels validated against the reference, not the C→Rust
//! bit-identity harness. Both decodes are elementwise (no reduction), so the shared fp8 LUT and
//! `ldexp(1, e)` match the oracles' `e4m3_to_f32` and `2f32.powi(e)` exactly and the comparison is
//! bitwise, not relative.
//!
//! The census waiver is `kernel_coverage.rs::ORACLE_VALIDATED`; this suite is the oracle it names.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::inkling_dequant::{Mxfp8, Nvfp4};
use rivoli_backend::hip::{launch_mxfp8_dequant, launch_nvfp4_dequant};

mod common;
use common::{assert_bits, assert_guard, dev, f32b, launch_to_f32, ok};

/// Deterministic LCG byte stream mapped through `map(index, raw_u64)`. Shared by the e4m3 weight
/// generator and the raw-exponent scale generator so both draw from one step rule.
fn gen_bytes(n: usize, seed: u64, mut map: impl FnMut(usize, u64) -> u8) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            map(i, s >> 33)
        })
        .collect()
}

/// A deterministic e4m3 byte spread. The two e4m3 NaN codes (`0x7F`/`0xFF`) are mapped to the max
/// finite `0x7E`: this suite scores the decode ARITHMETIC, and a NaN payload is not part of the
/// contract the LUT and `e4m3_to_f32` share.
fn weight_bytes(n: usize, seed: u64) -> Vec<u8> {
    gen_bytes(n, seed, |_, r| match r as u8 {
        0x7F | 0xFF => 0x7E,
        b => b,
    })
}

/// A deterministic RAW-exponent spread in `0..=254`, WITH `0` and `127` forced in: `0` is the
/// subnormal `2^-127` and `127` is `1.0` — the two values an e4m3 reading of the scale would turn
/// into `±0` and NaN and silently erase. `255` is excluded (`2^128` overflows fp32 `powi` to +inf).
fn exponents(n: usize, seed: u64) -> Vec<u8> {
    gen_bytes(n, seed, |i, r| match i % 95 {
        0 => 0,
        1 => 127,
        _ => (r % 255) as u8,
    })
}

/// One MXFP8 geometry's launch, returning the launcher's own `Result` so the guard tests exercise
/// the same entry point. `n` is the total element count (`E·rows·cols`), `cols` the logical width.
fn mxfp8_launch(weight: &[u8], scale: &[u8], n: usize, cols: usize) -> anyhow::Result<Vec<f32>> {
    let wb = dev(weight);
    let sb = dev(scale);
    // SAFETY: `weight` `n` bytes, `scale` `n/32`, `out` `n` f32 (owned by `launch_to_f32`); all
    // outlive the stream `back` syncs on. The launcher guards the geometry before device access.
    launch_to_f32(n, |op, st| unsafe {
        launch_mxfp8_dequant(wb.ptr(), sb.ptr(), n, cols, op, st)
    })
}

/// The decode reproduces the first-party oracle bit for bit across ragged expert/row/width grids.
#[test]
fn mxfp8_dequant_matches_the_cpu_oracle() {
    for &(e, rows, cols) in &[(1usize, 1, 32), (2, 3, 64), (4, 5, 128), (3, 7, 96)] {
        let n = e * rows * cols;
        let weight = weight_bytes(n, 0x1234 + n as u64);
        let scale = exponents(n / 32, 0x5678 + n as u64);
        let want = ok(
            Mxfp8 {
                weight: &weight,
                scale: &scale,
                rows,
                cols,
            }
            .dequant(),
            "cpu mxfp8",
        );
        let got = ok(mxfp8_launch(&weight, &scale, n, cols), "gpu mxfp8");
        assert_bits(&want, &got, &format!("mxfp8 e={e} rows={rows} cols={cols}"));
    }
}

/// T1: the scale is a RAW U8 exponent (`2^(s−127)`), NOT an e4m3 code. With every weight `0x38`
/// (e4m3 = 1.0) and the explicit exponents `[0, 127]`, the output is `[2^-127, 1.0]`; decoding the
/// scale as e4m3 would read `0x00 = 0` and `0x7F = NaN` and collapse the row.
#[test]
fn mxfp8_scale_is_a_raw_exponent_not_e4m3() {
    let (rows, cols) = (2usize, 32);
    let n = rows * cols;
    let weight = vec![0x38u8; n];
    // One scale byte per 32 values: 0 (2^-127) for row 0, 127 (1.0) for row 1.
    let scale = vec![0u8, 127];
    let got = ok(mxfp8_launch(&weight, &scale, n, cols), "mxfp8 T1");
    assert_bits(
        &got[..32],
        &[2.0f32.powi(-127); 32],
        "T1: exponent 0 is 2^-127",
    );
    assert!(
        got[32..].iter().all(|&x| x == 1.0),
        "T1: exponent 127 is 1.0, got {:?}",
        &got[32..36]
    );
}

/// `n > 0`, `cols > 0`, `cols % 32 == 0` and `n % cols == 0` are each guarded (1001) before any
/// launch.
#[test]
fn mxfp8_dequant_geometry_is_guarded() {
    let w = weight_bytes(64, 1);
    let s = exponents(2, 2);
    assert_guard(mxfp8_launch(&w, &s, 0, 32), Some(1001), "n = 0");
    assert_guard(mxfp8_launch(&w, &s, 64, 0), Some(1001), "cols = 0");
    assert_guard(mxfp8_launch(&w, &s, 64, 33), Some(1001), "cols % 32 != 0");
    assert_guard(mxfp8_launch(&w, &s, 48, 32), Some(1001), "n % cols != 0");
    assert_guard(mxfp8_launch(&w, &s, 64, 32), None, "valid geometry");
}

/// One NVFP4 geometry's operands — named fields, not seven positional args, because a bare
/// `rows`/`cols_packed` swap would be a wrong answer rather than a compile error. `Copy` so the
/// guard test can vary one field with struct-update syntax without consuming the base.
#[derive(Clone, Copy)]
struct Nvfp4Case<'a> {
    weight: &'a [u8],
    scale: &'a [u8],
    amax: &'a [f32],
    denom: f32,
    rows: usize,
    cols_packed: usize,
    /// An explicit output count, for the divisibility guard; `None` derives the oracle's `n`.
    n: Option<usize>,
}

impl<'a> Nvfp4Case<'a> {
    /// The tests' fixed `denom` (448, the e4m3 max) and a geometry tuple, so the constructor stays
    /// at four arguments. The guard test overrides `denom`/geometry with struct-update syntax.
    fn new(weight: &'a [u8], scale: &'a [u8], amax: &'a [f32], geom: (usize, usize)) -> Self {
        Self {
            weight,
            scale,
            amax,
            denom: 448.0,
            rows: geom.0,
            cols_packed: geom.1,
            n: None,
        }
    }

    /// The first-party CPU oracle over the same operands — one spelling of the field map, so the
    /// device case and the oracle cannot drift apart.
    fn oracle(&self) -> Nvfp4<'_> {
        Nvfp4 {
            weight: self.weight,
            scale: self.scale,
            amax: self.amax,
            denom: self.denom,
            rows: self.rows,
            cols_packed: self.cols_packed,
        }
    }
}

/// One NVFP4 geometry's launch. `n` (the output count `E·rows·cols_packed·2`) is derived here, as
/// the launcher derives it host-side, unless the case names one explicitly for a guard arm.
fn nvfp4_launch(c: &Nvfp4Case) -> anyhow::Result<Vec<f32>> {
    let n = c.n.unwrap_or(c.amax.len() * c.rows * c.cols_packed * 2);
    let wb = dev(c.weight);
    let sb = dev(c.scale);
    let ab = dev(&f32b(c.amax));
    // SAFETY: `weight` `n/2` bytes, `scale` `n/16`, `amax` `E` f32, `out` `n` f32; all outlive the
    // stream `back` syncs on. The launcher guards the geometry and `denom` before device access.
    launch_to_f32(n, |op, st| unsafe {
        launch_nvfp4_dequant(
            wb.ptr(),
            sb.ptr(),
            ab.ptr() as *const f32,
            c.denom,
            n,
            c.rows,
            c.cols_packed,
            op,
            st,
        )
    })
}

/// The decode reproduces the first-party oracle bit for bit, including the `(code·scale)·factor`
/// association (LOAD:813) and the low-nibble-first packing, across ragged grids.
#[test]
fn nvfp4_dequant_matches_the_cpu_oracle() {
    for &(e, rows, cols_packed) in &[(1usize, 1, 8), (2, 3, 16), (4, 5, 24), (3, 7, 8)] {
        let n = e * rows * cols_packed * 2;
        let weight = weight_bytes(e * rows * cols_packed, 0xabcd + n as u64);
        let scale = weight_bytes(e * rows * (cols_packed / 8), 0xdcba + n as u64);
        let amax: Vec<f32> = (0..e).map(|i| 1.0 + i as f32 * 0.5).collect();
        let c = Nvfp4Case::new(&weight, &scale, &amax, (rows, cols_packed));
        let want = ok(c.oracle().dequant(), "cpu nvfp4");
        let got = ok(nvfp4_launch(&c), "gpu nvfp4");
        assert_bits(
            &want,
            &got,
            &format!("nvfp4 e={e} rows={rows} cp={cols_packed}"),
        );
    }
}

/// T2: the even element is the LOW nibble. Byte `0x12` is low `2` (E2M1 = 1.0) then high `1`
/// (E2M1 = 0.5); with unit scale and factor the pair is `[1.0, 0.5]`, not `[0.5, 1.0]`.
#[test]
fn nvfp4_even_element_is_the_low_nibble() {
    // cols_packed 8 needs 8 weight bytes; the first is 0x12, the rest zero codes (E2M1[0] = 0).
    let weight = vec![0x12u8, 0, 0, 0, 0, 0, 0, 0];
    let scale = vec![0x38u8]; // e4m3 = 1.0
    let amax = vec![448.0f32];
    let c = Nvfp4Case::new(&weight, &scale, &amax, (1, 8));
    let got = ok(nvfp4_launch(&c), "nvfp4 T2");
    assert_eq!(
        got[0], 1.0,
        "T2: even element is the low nibble (code 2 = 1.0)"
    );
    assert_eq!(
        got[1], 0.5,
        "T2: odd element is the high nibble (code 1 = 0.5)"
    );
}

/// The negative E2M1 half INCLUDING code 8, which is `−0.0`: the codebook's sign bit alone, a
/// value `assert_eq!(−0.0, 0.0)` cannot see, so the comparison is on the raw bits.
#[test]
fn nvfp4_negative_codes_keep_their_sign_bit() {
    // Byte 0x98: low nibble 8 (−0.0), high nibble 9 (−0.5); byte 0xFE: low 14 (−4.0), high 15
    // (−6.0). Unit scale and factor 1.
    let weight = vec![0x98u8, 0xFE, 0, 0, 0, 0, 0, 0];
    let scale = vec![0x38u8];
    let amax = vec![448.0f32];
    let c = Nvfp4Case::new(&weight, &scale, &amax, (1, 8));
    let got = ok(nvfp4_launch(&c), "nvfp4 negative");
    assert_bits(&[-0.0f32], &got[..1], "code 8 is −0.0 (the sign bit alone)");
    assert_eq!(got[1], -0.5);
    assert_eq!(got[2], -4.0);
    assert_eq!(got[3], -6.0);
}

/// `n > 0`, `rows > 0`, `cols_packed > 0` and `% 8 == 0`, `n % (rows·cols_packed·2) == 0` (1001),
/// `denom` positive finite (1006).
#[test]
fn nvfp4_dequant_geometry_is_guarded() {
    let weight = weight_bytes(8, 1);
    let scale = weight_bytes(1, 2);
    let amax = vec![1.0f32];
    let base = Nvfp4Case::new(&weight, &scale, &amax, (1, 8)); // plane 16, n 16
    assert_guard(
        nvfp4_launch(&Nvfp4Case { rows: 0, ..base }),
        Some(1001),
        "rows = 0",
    );
    assert_guard(
        nvfp4_launch(&Nvfp4Case {
            cols_packed: 0,
            ..base
        }),
        Some(1001),
        "cols_packed = 0",
    );
    assert_guard(
        nvfp4_launch(&Nvfp4Case {
            cols_packed: 9,
            ..base
        }),
        Some(1001),
        "cols_packed % 8 != 0",
    );
    assert_guard(
        nvfp4_launch(&Nvfp4Case {
            n: Some(18),
            ..base
        }),
        Some(1001),
        "n % (rows·cols_packed·2) != 0",
    );
    for bad in [f32::NAN, -1.0, f32::INFINITY] {
        assert_guard(
            nvfp4_launch(&Nvfp4Case { denom: bad, ..base }),
            Some(1006),
            &format!("denom = {bad}"),
        );
    }
    assert_guard(nvfp4_launch(&base), None, "valid geometry");
}
