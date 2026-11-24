//! Inkling 2's dequant oracle gate (S2): the Rust transcription of LOAD:772–853 is scored against
//! the FIRST-PARTY transformers ops, not a transliteration. `dequant-golden.json` is vendored from
//! `Inkling2LoopDequantizeExperts` / `Interleave` / `Inkling2LoopMergeMessageQueries` run on real
//! Mini-shard experts (transformers @ d8f10a82, torch CPU); the live cell reads the same experts,
//! dequantizes through `rivoli_artifact::inkling_dequant`, narrows to bf16, and compares fnv1a64
//! over the bf16 bytes plus 64 spot values. The deviceless cells pin the arithmetic on tiny
//! hand-computed inputs so the module is gated even where the shard is absent.
//!
//! The deviceless cells also hold the FUSED bf16 emitter (`dequant_to_bf16_le`) to the two-step
//! `dequant` → `deinterleave_rows` → `f32_to_bf16` path byte for byte, with and without the `w13`
//! row permutation. The converter's 2.7 GB-per-layer memory saving is only sound if that emitter
//! is bit-identical, so the equivalence is a gate rather than a claim.
//!
//! The live cell skips unless `RIVOLI_INKLING_REQUIRED` is set, on `inkling_pins.rs`'s policy.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::format::{Dtype, Safetensors};
use rivoli_artifact::inkling_dequant::{Mxfp8, Nvfp4, deinterleave_rows, merge_message_queries};
use rivoli_artifact::quant::read_f32;
use rivoli_core::hash::fnv1a;
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};
use serde_json::Value;

const SHARD: &str = "/swarm/storage/ai/rivoli/inkling-2-mini-src/model-00001-of-00001.safetensors";
const GOLDEN: &str =
    include_str!("../../../docs/measurement/inkling2-reference/mini/dequant-golden.json");
const EXPERTS: usize = 2;

/// The bf16 narrowing of an f32 weight, little-endian — the artifact's stored bytes.
fn bf16_bytes(w: &[f32]) -> Vec<u8> {
    w.iter()
        .flat_map(|&x| f32_to_bf16(x).to_le_bytes())
        .collect()
}

/// fnv1a64 over [`bf16_bytes`].
fn bf16_fnv(w: &[f32]) -> u64 {
    fnv1a(&bf16_bytes(w))
}

/// The first `n` values narrowed to bf16 and back — exactly what the golden's spot fields hold.
fn bf16_spot(w: &[f32], n: usize) -> Vec<f32> {
    w.iter()
        .take(n)
        .map(|&x| bf16_to_f32(f32_to_bf16(x)))
        .collect()
}

/// A golden spot field (`spot_f32` or `deinterleave_spot_f32`) as f32.
fn spot_field(case: &Value, key: &str) -> Vec<f32> {
    case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn hex(v: &Value) -> u64 {
    u64::from_str_radix(v.as_str().unwrap(), 16).unwrap()
}

// ── deviceless arithmetic cells ────────────────────────────────────────────────────────────

/// One expert, one row, eight packed bytes (16 logical columns, one block-16 scale). `0x21` is
/// low nibble 1 (0.5), high nibble 2 (1.0); scale `0x38` is e4m3 1.0. Shared by the two NVFP4
/// cells so the input setup is written once (jscpd).
fn nvfp4_tiny(amax: f32, denom: f32) -> Vec<f32> {
    let weight = [0x21u8, 0, 0, 0, 0, 0, 0, 0];
    let scale = [0x38u8];
    let amax = [amax];
    Nvfp4 {
        weight: &weight,
        scale: &scale,
        amax: &amax,
        denom,
        rows: 1,
        cols_packed: 8,
    }
    .dequant()
    .unwrap()
}

/// Even element is the LOW nibble (trap T2), scale is e4m3 block-16 (trap T3).
#[test]
fn nvfp4_decodes_low_nibble_first_with_e4m3_block_scale() {
    let out = nvfp4_tiny(1.0, 1.0);
    assert_eq!(out.len(), 16);
    assert_eq!(out[0], 0.5); // even element, LOW nibble
    assert_eq!(out[1], 1.0); // odd element, HIGH nibble
    assert!(out[2..].iter().all(|&x| x == 0.0));
}

/// The per-expert factor `amax/denominator` scales every value (LOAD:804,814).
#[test]
fn nvfp4_applies_the_per_expert_amax_factor() {
    let out = nvfp4_tiny(6.0, 2.0); // factor 3
    assert_eq!(out[0], 0.5 * 3.0);
    assert_eq!(out[1], 1.0 * 3.0);
}

/// MXFP8: e4m3 value times `2^(scale−127)`, row-wise block 32. scale 128 → ×2.
#[test]
fn mxfp8_decodes_e4m3_times_power_of_two_row_scale() {
    let mut weight = [0u8; 32];
    weight[0] = 0x38; // e4m3 1.0
    weight[1] = 0x38;
    let scale = [128u8]; // one block of 32, exponent 128 → 2^1
    let out = Mxfp8 {
        weight: &weight,
        scale: &scale,
        rows: 1,
        cols: 32,
    }
    .dequant()
    .unwrap();
    assert_eq!(out[0], 2.0);
    assert_eq!(out[1], 2.0);
    assert!(out[2..].iter().all(|&x| x == 0.0));
}

/// De-interleave `[gate_0, up_0, gate_1, up_1]` → `[gate_0, gate_1, up_0, up_1]`.
#[test]
fn deinterleave_splits_gate_up_rows() {
    let w = [10.0, 20.0, 30.0, 40.0]; // e=1, rows=4, cols=1
    let out = deinterleave_rows(&w, 1, 4, 1).unwrap();
    assert_eq!(out, [10.0, 30.0, 20.0, 40.0]);
}

/// Merge `[q, mlp]` + `[k, v]` banks into `(q, k, v, mlp)` order (trap T13).
#[test]
fn merge_orders_queries_q_k_v_mlp() {
    let q_mlp = [1.0, 2.0, 3.0, 4.0]; // [q0,q1, mlp0,mlp1]
    let kv = [5.0, 6.0, 7.0, 8.0]; // [k0,k1, v0,v1]
    let out = merge_message_queries(&q_mlp, &kv, 1, 2).unwrap();
    assert_eq!(out, [1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0]);
}

// ── the fused bf16 emitter ────────────────────────────────────────────────────────────────

/// The two-step path the fused emitter must reproduce byte for byte: `dequant`, the optional
/// `deinterleave_rows`, then `f32_to_bf16` (via [`bf16_bytes`]).
fn reference_bf16(values: &[f32], shape: (usize, usize, usize), deinterleave: bool) -> Vec<u8> {
    let (e, rows, cols) = shape;
    let w = if deinterleave {
        deinterleave_rows(values, e, rows, cols).unwrap()
    } else {
        values.to_vec()
    };
    bf16_bytes(&w)
}

/// The fused emitter's two arms against the two-step reference, plus the byte length and a
/// non-vacuity check that the de-interleave actually permutes rows. One body for both schemes.
fn assert_fused(fused: impl Fn(bool) -> Vec<u8>, values: &[f32], shape: (usize, usize, usize)) {
    let (e, rows, cols) = shape;
    let plain = reference_bf16(values, shape, false);
    let permuted = reference_bf16(values, shape, true);
    assert_ne!(
        plain, permuted,
        "the de-interleave must move rows, or this cell is vacuous"
    );
    assert_eq!(fused(false), plain, "unpermuted bytes");
    assert_eq!(fused(true), permuted, "de-interleaved bytes");
    assert_eq!(fused(true).len(), e * rows * cols * 2, "bf16 byte length");
}

/// Four rows of `width` bytes whose leading codes differ, so the de-interleave permutation moves
/// real bytes. The row width IS the stride, so one body serves both schemes.
fn distinct_rows(width: usize, codes: [u8; 4]) -> Vec<u8> {
    let mut weight = vec![0u8; 4 * width];
    for (r, code) in codes.into_iter().enumerate() {
        weight[r * width] = code;
    }
    weight
}

/// `dequant_to_bf16_le` equals `dequant` → `deinterleave_rows` → `f32_to_bf16`, both arms.
#[test]
fn nvfp4_fused_bf16_matches_the_two_step_path() {
    let weight = distinct_rows(8, [0x21, 0x32, 0x43, 0x54]);
    let scale = [0x38u8; 4]; // e4m3 1.0, one block per row
    let nv = Nvfp4 {
        weight: &weight,
        scale: &scale,
        amax: &[6.0],
        denom: 2.0,
        rows: 4,
        cols_packed: 8,
    };
    let values = nv.dequant().unwrap();
    assert_fused(|d| nv.dequant_to_bf16_le(d).unwrap(), &values, (1, 4, 16));
}

/// The MXFP8 fused emitter, same equivalence and the same non-vacuity check. The four leading
/// e4m3 codes are 0.5, 1.0, 1.5 and 2.0.
#[test]
fn mxfp8_fused_bf16_matches_the_two_step_path() {
    let weight = distinct_rows(32, [0x30, 0x38, 0x3C, 0x40]);
    let scale = [128u8; 4]; // 2^1, one block per row
    let m = Mxfp8 {
        weight: &weight,
        scale: &scale,
        rows: 4,
        cols: 32,
    };
    let values = m.dequant().unwrap();
    assert_fused(|d| m.dequant_to_bf16_le(d).unwrap(), &values, (1, 4, 32));
}

/// The per-expert divisor is a DIVISOR: zero, negative and non-finite are refused by name, from
/// both outputs (the guard lives in the shared validation, so the fused path inherits it).
#[test]
fn nvfp4_refuses_a_non_positive_or_non_finite_denominator() {
    let weight = [0x21u8, 0, 0, 0, 0, 0, 0, 0];
    let scale = [0x38u8];
    for denom in [0.0f32, -1.0, f32::NAN] {
        let nv = Nvfp4 {
            weight: &weight,
            scale: &scale,
            amax: &[1.0],
            denom,
            rows: 1,
            cols_packed: 8,
        };
        for err in [
            nv.dequant().unwrap_err(),
            nv.dequant_to_bf16_le(false).unwrap_err(),
            nv.dequant_to_bf16_le(true).unwrap_err(),
        ] {
            let text = format!("{err:#}");
            assert!(
                text.contains("denom"),
                "denom {denom} must be refused by name: {text}"
            );
        }
    }
}

// ── the first-party cross-check ────────────────────────────────────────────────────────────

/// Every golden case, recomputed from the live shard through the Rust oracle and compared to the
/// transformers op's bf16 output. Skips unless required.
#[test]
fn the_rust_dequant_matches_the_first_party_golden() {
    if std::fs::metadata(SHARD).is_err() {
        assert!(
            std::env::var_os("RIVOLI_INKLING_REQUIRED").is_none(),
            "Inkling dequant gate REQUIRED but did not run: {SHARD} absent"
        );
        eprintln!("skip live dequant cross-check: {SHARD} absent (set RIVOLI_INKLING_REQUIRED=1)");
        return;
    }
    let golden: Value = serde_json::from_str(GOLDEN).unwrap();
    let src = Safetensors::open_file(SHARD).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert!(cases.len() >= 5, "examined cases");
    for case in cases {
        let label = case["label"].as_str().unwrap();
        match case["format"].as_str().unwrap() {
            "nvfp4" => check_nvfp4(&src, case, label),
            "mxfp8" => check_mxfp8(&src, case, label),
            "merge" => check_merge(&src, case),
            other => panic!("unknown golden format {other:?}"),
        }
    }
}

fn proj_of(label: &str) -> &'static str {
    if label.contains("w13") {
        "w13_weight"
    } else {
        "w2_weight"
    }
}

/// The bf16 fnv + spot + (optional) deinterleave comparison, shared by both format checks.
fn assert_dequant(case: &Value, label: &str, out: &[f32]) {
    let rows = case["rows"].as_u64().unwrap() as usize;
    let cols = case["cols"].as_u64().unwrap() as usize;
    assert_eq!(out.len(), EXPERTS * rows * cols);
    assert_eq!(
        bf16_fnv(out),
        hex(&case["bf16_fnv1a64"]),
        "{label} bf16 fnv"
    );
    assert_eq!(
        bf16_spot(out, 64),
        spot_field(case, "spot_f32"),
        "{label} spot"
    );
    if let Some(dh) = case.get("deinterleave_bf16_fnv1a64") {
        let deint = deinterleave_rows(out, EXPERTS, rows, cols).unwrap();
        assert_eq!(bf16_fnv(&deint), hex(dh), "{label} deinterleave fnv");
        assert_eq!(
            bf16_spot(&deint, 64),
            spot_field(case, "deinterleave_spot_f32"),
            "{label} deinterleave spot"
        );
    }
}

fn check_nvfp4(src: &Safetensors, case: &Value, label: &str) {
    let rows = case["rows"].as_u64().unwrap() as usize;
    let cols = case["cols"].as_u64().unwrap() as usize;
    let cols_packed = cols / 2;
    let base = format!("model.llm.layers.0.mlp.experts.{}", proj_of(label));
    let (wb, _) = src.typed(&base, Dtype::U8).unwrap();
    let (sb, _) = src.typed(&format!("{base}_scale"), Dtype::F8E4M3).unwrap();
    let (ab, _) = src.typed(&format!("{base}_amax"), Dtype::F32).unwrap();
    let (db, _) = src
        .typed(&format!("{base}_outer_scale_denominator"), Dtype::F32)
        .unwrap();
    let amax = read_f32(ab);
    let w = Nvfp4 {
        weight: &wb[..EXPERTS * rows * cols_packed],
        scale: &sb[..EXPERTS * rows * (cols / 16)],
        amax: &amax[..EXPERTS],
        denom: read_f32(db)[0],
        rows,
        cols_packed,
    };
    assert_dequant(case, label, &w.dequant().unwrap());
}

fn check_mxfp8(src: &Safetensors, case: &Value, label: &str) {
    let rows = case["rows"].as_u64().unwrap() as usize;
    let cols = case["cols"].as_u64().unwrap() as usize;
    let base = format!("model.llm.layers.20.mlp.experts.{}", proj_of(label));
    let (wb, _) = src.typed(&base, Dtype::F8E4M3).unwrap();
    let (sb, _) = src.typed(&format!("{base}_scale"), Dtype::U8).unwrap();
    let w = Mxfp8 {
        weight: &wb[..EXPERTS * rows * cols],
        scale: &sb[..EXPERTS * rows * (cols / 32)],
        rows,
        cols,
    };
    let out = w.dequant().unwrap();
    assert_dequant(case, label, &out);
}

fn check_merge(src: &Safetensors, case: &Value) {
    let loop_layers = case["loop_layers"].as_u64().unwrap() as usize;
    let h = case["hidden"].as_u64().unwrap() as usize;
    let (qb, _) = src
        .typed("model.llm.middle_cycle_cross_pass_queries", Dtype::F32)
        .unwrap();
    let (kb, _) = src
        .typed("model.llm.middle_cycle_cross_pass_kv_queries", Dtype::F32)
        .unwrap();
    let out = merge_message_queries(&read_f32(qb), &read_f32(kb), loop_layers, h).unwrap();
    let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(fnv1a(&bytes), hex(&case["f32_fnv1a64"]), "merge fnv");
    assert_eq!(
        out.iter().take(64).copied().collect::<Vec<f32>>(),
        spot_field(case, "spot_f32"),
        "merge spot"
    );
}
