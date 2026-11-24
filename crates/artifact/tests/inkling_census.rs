//! Inkling 2 Mini's census CLASSIFICATION gate (S2): every shard name lands in exactly one
//! family, the converter's by-name exclusion is exhaustive, and the confront refuses a count
//! drift. Distinct from `inkling_pins.rs`, which sums the TSV rows and reads the shard header's
//! totals but never classifies a NAME — this file is the name→family half the converter rests on.
//!
//! The live cell mmaps the local Mini shard and classifies all 1244 names through
//! `census::inkling::Census::confront`; it skips unless `RIVOLI_INKLING_REQUIRED` is set, on
//! `inkling_pins.rs`'s policy. The synthetic cells exercise the same confront deviceless, both
//! ends (a full expansion passes; a dropped tensor reddens), so the count check is proven armed
//! even where the shard is absent. No GPU, no network.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::census::inkling::{Census, FAMILIES, Summary, collapse, dtype_of};
use rivoli_artifact::format::Dtype;

const SHARD: &str = "/swarm/storage/ai/rivoli/inkling-2-mini-src/model-00001-of-00001.safetensors";

fn declared(label: &str) -> u64 {
    FAMILIES.declared(label).unwrap()
}

fn token(anchor: &str, keyword: &str) -> u64 {
    FAMILIES
        .header_token(anchor, keyword)
        .unwrap()
        .parse()
        .unwrap()
}

/// The five header tallies a confront must reproduce — shared by the synthetic and live cells.
fn assert_summary(sum: &Summary) {
    assert_eq!(sum.text_tensors, token("STATUS TEXT", "tensors "));
    assert_eq!(
        sum.vision_tensors,
        token("STATUS EXCLUDED-VISION", "tensors ")
    );
    assert_eq!(sum.text_bytes, token("STATUS TEXT", "bytes "));
    assert_eq!(sum.vision_bytes, token("STATUS EXCLUDED-VISION", "bytes "));
    assert_eq!(sum.total_tensors(), declared("tensors in the shard header"));
}

/// The census loads, its row count and byte closure match the header, and the status split is
/// the one the converter's exclusion is held to.
#[test]
fn the_census_loads_with_its_declared_closure() {
    let c = Census::load().unwrap();
    assert_eq!(c.families().len() as u64, declared("family rows"));
    assert!(!c.families().is_empty());

    let (mut text_rows, mut vision_rows) = (0u64, 0u64);
    let (mut tensors, mut bytes) = (0u64, 0u64);
    for f in c.families() {
        tensors += f.count;
        bytes += f.total;
        if f.status.is_excluded() {
            vision_rows += 1;
        } else {
            text_rows += 1;
        }
    }
    assert_eq!(tensors, declared("tensors in the shard header"));
    assert_eq!(bytes, declared("payload bytes (header data_offsets)"));
    assert_eq!(text_rows + vision_rows, c.families().len() as u64);
    // The two status row counts are derived here, not pinned in prose; the tensor tallies below
    // are the header's own.
    assert_eq!(text_rows, 59);
    assert_eq!(vision_rows, 26);
}

/// Every census row's EXAMPLE name collapses to its pattern and resolves back to its family —
/// the collapse and the (pattern, dtype, shape) key, exercised deviceless on all 85 rows.
#[test]
fn every_example_collapses_to_its_pattern_and_resolves() {
    let c = Census::load().unwrap();
    let rows = FAMILIES.rows(8).unwrap();
    assert_eq!(rows.len(), c.families().len());
    for r in &rows {
        let example = r.str_at(2).unwrap();
        let pattern = r.str_at(1).unwrap();
        assert_eq!(collapse(example), pattern, "collapse of {example:?}");
        let shape: Vec<usize> = r
            .dims_at(3)
            .unwrap()
            .into_iter()
            .map(|d| d as usize)
            .collect();
        let dtype = dtype_of(r.str_at(4).unwrap()).unwrap();
        let f = c.family_of(example, dtype, &shape).unwrap();
        assert_eq!(f.pattern, pattern);
        assert_eq!(f.shape, shape);
    }
}

/// The confront, both ends, deviceless: a full synthetic expansion of the census passes with the
/// header's text/vision tallies, and dropping one tensor reddens the count check. This proves the
/// arithmetic is armed without the shard; the live cell below proves it on the real names.
#[test]
fn the_confront_passes_a_full_expansion_and_refuses_a_missing_tensor() {
    let c = Census::load().unwrap();
    let owned = synthetic_tensors(&c);
    let borrowed = borrow_tensors(&owned);
    let sum = c.confront(&borrowed).unwrap();
    assert_summary(&sum);

    // Drop one TEXT tensor: its family's observed count falls one short, and the confront must
    // refuse rather than balance a grand total while a class vanished.
    let mut short = owned.clone();
    let drop_at = short
        .iter()
        .position(|t| !c.family_of(&t.0, t.1, &t.2).unwrap().status.is_excluded())
        .unwrap();
    short.remove(drop_at);
    let borrowed_short = borrow_tensors(&short);
    assert!(
        c.confront(&borrowed_short).is_err(),
        "a missing tensor must redden the confront"
    );

    // An unknown name is a hard error, not a silent drop.
    let mut extra = owned.clone();
    extra.push((
        "model.llm.layers.0.attn.not_a_real_family.weight".to_string(),
        Dtype::Bf16,
        vec![2304],
        4608,
    ));
    let borrowed_extra = borrow_tensors(&extra);
    assert!(
        c.confront(&borrowed_extra).is_err(),
        "an unknown name must redden the confront"
    );
}

/// The live Mini shard classifies into the census: all 1244 names, every family count matched,
/// the text/vision split exactly the header's. Reads only the header pages (mmap is lazy; the
/// payload is never touched). Skips unless required, on `inkling_pins.rs`'s policy.
#[test]
fn the_live_shard_classifies_into_the_census() {
    use rivoli_artifact::format::Safetensors;
    if std::fs::metadata(SHARD).is_err() {
        assert!(
            std::env::var_os("RIVOLI_INKLING_REQUIRED").is_none(),
            "Inkling census gate REQUIRED but did not run: {SHARD} absent"
        );
        eprintln!("skip live shard classification: {SHARD} absent (set RIVOLI_INKLING_REQUIRED=1)");
        return;
    }
    let c = Census::load().unwrap();
    let src = Safetensors::open_file(SHARD).unwrap();
    let names = src.names();
    let mut tensors: Vec<(&str, Dtype, &[usize], u64)> = Vec::with_capacity(names.len());
    for n in &names {
        let (b, dt, sh) = src.raw(n).unwrap();
        tensors.push((*n, dt, sh, b.len() as u64));
    }
    let sum = c.confront(&tensors).unwrap();
    assert_summary(&sum);
    assert!(!tensors.is_empty(), "examined tensors");
}

// ── helpers ──────────────────────────────────────────────────────────────────────────────

/// Expand each family into `count` distinct synthetic tensors whose names collapse to its
/// pattern. The indices need only be distinct, not legal: the confront checks classification and
/// counts, not index domains (the converter bounds those against the config).
fn synthetic_tensors(c: &Census) -> Vec<(String, Dtype, Vec<usize>, u64)> {
    let mut out = Vec::new();
    for f in c.families() {
        for k in 0..f.count {
            let name = f
                .pattern
                .replace("{L}", &k.to_string())
                .replace("{I}", &k.to_string());
            out.push((name, f.dtype, f.shape.clone(), f.bytes_per));
        }
    }
    out
}

fn borrow_tensors(owned: &[(String, Dtype, Vec<usize>, u64)]) -> Vec<(&str, Dtype, &[usize], u64)> {
    owned
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), *b))
        .collect()
}
