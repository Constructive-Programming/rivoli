//! The plant population, held equal to its tables at both ends, and every plant proven to
//! still apply to the tree it is meant to break.
//!
//! Two kinds of plant live under `tests/bit-identity/plants/`:
//! - **probe plants** at the root, one per prelude family, applied to a scratch copy of
//!   `crates/kernels` by `tests/bit-identity.sh --build-probes --plant=FAMILY` and judged by that
//!   script's `plant_spec` row;
//! - **launcher-row plants** under `launch/`, applied from the repo root by
//!   `tests/bit-identity/build-launch-plants.sh` (its `PLANTS` table names each plant's patches and
//!   the rows it must redden).
//!
//! Why this exists (2026-11-23): before it, a patch with no table row, a row with no patch, or a
//! patch whose context had drifted (L1's `pub fn tanh_ocml` once broke Phase 0's `situ` plant
//! silently) surfaced only as a refused build on the next GPU lease. Deviceless; needs `patch`.
#![allow(clippy::expect_used)] // a missing script or table is the test failing, loudly.

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn plants_dir() -> PathBuf {
    common::repo_root().join("tests/bit-identity/plants")
}

/// The `*.patch` stems directly under `dir`.
fn patch_stems(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("plants dir")
        .filter_map(|e| {
            let p = e.expect("dir entry").path();
            (p.extension()? == "patch").then(|| p.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect()
}

/// The `PLANTS="…"` table of `build-launch-plants.sh`: one `name|patches|rows` line each.
fn launch_table() -> Vec<(String, Vec<String>)> {
    let script = std::fs::read_to_string(
        common::repo_root().join("tests/bit-identity/build-launch-plants.sh"),
    )
    .expect("build-launch-plants.sh");
    let body = script
        .split_once("PLANTS=\"")
        .and_then(|(_, rest)| rest.split_once('"'))
        .expect("a PLANTS=\"…\" table")
        .0;
    body.lines()
        .map(|line| {
            let mut f = line.split('|');
            let name = f.next().expect("name").to_owned();
            let patches = f.next().expect("patches");
            assert!(
                f.next().is_some_and(|r| !r.is_empty()),
                "{name}: no expected-red rows"
            );
            let stems = patches
                .split_whitespace()
                .map(|p| p.trim_end_matches(".patch").to_owned());
            (name, stems.collect())
        })
        .collect()
}

/// The case labels of `tests/bit-identity.sh`'s `plant_spec`, the probe plants' judges.
fn plant_spec_labels() -> BTreeSet<String> {
    let script = std::fs::read_to_string(common::repo_root().join("tests/bit-identity.sh"))
        .expect("tests/bit-identity.sh");
    let body = script
        .split_once("plant_spec() {")
        .and_then(|(_, rest)| rest.split_once("esac"))
        .expect("plant_spec's case block")
        .0;
    body.lines()
        .filter_map(|l| {
            l.trim_start()
                .split_once(')')
                .map(|(label, _)| label.trim())
        })
        .filter(|label| {
            !label.is_empty() && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
        .map(str::to_owned)
        .collect()
}

/// `patch --dry-run` of `file` against `dir` with exact context, as the build scripts apply it.
fn applies(file: &Path, dir: &Path) -> bool {
    let input = std::fs::File::open(file).expect("open patch");
    Command::new("patch")
        .args(["-p1", "-F0", "--dry-run", "-s", "-d"])
        .arg(dir)
        .stdin(input)
        .output()
        .expect("run `patch` (needed by this test and the build scripts)")
        .status
        .success()
}

#[test]
fn the_launch_table_and_the_launch_patches_are_the_same_set() {
    let table = launch_table();
    let names: BTreeSet<_> = table.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(
        names.len(),
        table.len(),
        "a plant name appears twice in PLANTS"
    );
    let listed: BTreeSet<_> = table.into_iter().flat_map(|(_, p)| p).collect();
    let on_disk = patch_stems(&plants_dir().join("launch"));
    assert!(
        !on_disk.is_empty(),
        "no launcher-row plants found — the examined set is empty"
    );
    let unlisted: Vec<_> = on_disk.difference(&listed).collect();
    let missing: Vec<_> = listed.difference(&on_disk).collect();
    assert!(
        unlisted.is_empty() && missing.is_empty(),
        "plants/launch/ vs build-launch-plants.sh PLANTS: unlisted patches {unlisted:?}, \
         listed but missing {missing:?}"
    );
}

#[test]
fn every_probe_plant_has_a_plant_spec_row_and_back() {
    let on_disk = patch_stems(&plants_dir());
    let labels = plant_spec_labels();
    assert!(
        !on_disk.is_empty(),
        "no probe plants found — the examined set is empty"
    );
    assert_eq!(
        on_disk, labels,
        "plants/*.patch vs tests/bit-identity.sh plant_spec: they must name the same families"
    );
}

#[test]
fn every_plant_still_applies_to_the_tree_it_breaks() {
    let root = common::repo_root();
    let probes = patch_stems(&plants_dir()).into_iter().map(|s| {
        (
            plants_dir().join(format!("{s}.patch")),
            root.join("crates/kernels"),
        )
    });
    let launch = patch_stems(&plants_dir().join("launch"))
        .into_iter()
        .map(|s| (plants_dir().join(format!("launch/{s}.patch")), root.clone()));
    let all: Vec<_> = probes.chain(launch).collect();
    assert!(all.len() >= 2, "examined {} plants", all.len());
    let stale: Vec<_> = all
        .iter()
        .filter(|(p, d)| !applies(p, d))
        .map(|(p, _)| p)
        .collect();
    assert!(
        stale.is_empty(),
        "plants whose context no longer matches the tree: {stale:?}"
    );
}
