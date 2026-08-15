//! The matrix scripts' hand-written dimension lists must match what the manifests
//! actually declare — adding a feature and forgetting the matrix must be a test failure,
//! not a hole that opens quietly.
//!
//! Ported from the old tree's `tests/matrix.rs`, reduced to the one script this tree has
//! (`feature-matrix.sh`); the mode/policy/attn halves return with their scripts and the
//! CLI that parses those flags (M4+).

#![allow(clippy::expect_used)] // meta-gate: a missing script or manifest should panic loudly

mod common;

/// The `NAME=(a b c)` array literal from a shell script, split into items.
fn sh_array(script: &str, name: &str) -> Vec<String> {
    let line = script
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("{name}=(")))
        .unwrap_or_else(|| panic!("{name}=(...) not found in feature-matrix.sh"));
    let inner = line
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .map(|(l, _)| l)
        .expect("balanced parens in the array literal");
    inner.split_whitespace().map(str::to_string).collect()
}

#[test]
fn the_feature_matrix_lists_match_the_manifests() {
    let root = common::repo_root();
    let script = std::fs::read_to_string(root.join("tests/feature-matrix.sh"))
        .expect("tests/feature-matrix.sh");

    // OPTIONAL must equal the engine's non-backend features — the engine is where
    // instrument features are DECLARED (cli only forwards them).
    let engine =
        std::fs::read_to_string(root.join("crates/engine/Cargo.toml")).expect("engine Cargo.toml");
    let feats = engine
        .split("[features]")
        .nth(1)
        .expect("engine [features]")
        .split("\n[")
        .next()
        .expect("features table body");
    let mut declared: Vec<String> = feats
        .lines()
        .filter_map(|l| l.split_once(" = ").map(|(k, _)| k.trim().to_string()))
        .filter(|k| !k.starts_with('#') && k != "rocm")
        .collect();
    declared.sort();
    let mut optional = sh_array(&script, "OPTIONAL");
    optional.sort();
    assert_eq!(
        optional, declared,
        "feature-matrix.sh's OPTIONAL disagrees with crates/engine/Cargo.toml's \
         non-backend features. Update BOTH — a feature missing from the matrix is checked \
         exactly as often as someone remembers it."
    );
    // Anti-vacuity: the parse found a real feature set, not an empty table.
    assert!(!declared.is_empty(), "no non-backend features parsed");

    // BACKENDS must equal the backend crate's declared features — parsed from the
    // manifest like the OPTIONAL half, not pinned to a literal (review 2026-08-15: the
    // literal made the header's "match what the manifests declare" claim false).
    let backend = std::fs::read_to_string(root.join("crates/backend/Cargo.toml"))
        .expect("backend Cargo.toml");
    let bfeats = backend
        .split("[features]")
        .nth(1)
        .expect("backend [features]")
        .split("\n[")
        .next()
        .expect("features table body");
    let mut declared_backends: Vec<String> = bfeats
        .lines()
        .filter_map(|l| l.split_once(" = ").map(|(k, _)| k.trim().to_string()))
        .filter(|k| !k.starts_with('#'))
        .collect();
    declared_backends.sort();
    let mut backends = sh_array(&script, "BACKENDS");
    backends.sort();
    assert_eq!(
        backends, declared_backends,
        "BACKENDS drifted from crates/backend/Cargo.toml's [features]"
    );

    // The cli must forward every engine feature it names — a forward missing here makes a
    // matrix cell silently vacuous (the feature resolves but gates nothing).
    let cli = std::fs::read_to_string(root.join("crates/cli/Cargo.toml")).expect("cli Cargo.toml");
    for f in &declared {
        assert!(
            cli.contains(&format!("{f} = [\"rivoli-engine/{f}\"]")),
            "cli does not forward `{f}` to the engine — the matrix cell for it checks \
             nothing"
        );
    }
}
