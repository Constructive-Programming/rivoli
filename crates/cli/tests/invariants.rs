//! The INV-n registry check: every invariant documented in `docs/reference/architecture.md`
//! §8b must have a test named `inv_<n>_*` somewhere under `crates/*/src`, and every such
//! test must be documented there.
//!
//! Ported from the old tree, where it exists because prose drifted from behaviour
//! repeatedly and nothing noticed — §8 claimed "a compute launch before its bytes land is
//! ruled out" long after that stopped being true. A doc claim nobody can verify is worse
//! than no claim; this makes the orphan — in either direction — a failing test rather
//! than an archaeology exercise.
// The registry file and its §8b table either parse or this test cannot run at all, so the
// panic IS the assertion.
#![allow(clippy::expect_used, clippy::unwrap_used)]
use std::collections::BTreeSet;

mod common;

fn ids(hay: &str, prefix: &str) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let bytes = hay.as_bytes();
    let mut i = 0;
    while let Some(p) = hay[i..].find(prefix) {
        let start = i + p + prefix.len();
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start {
            out.insert(hay[start..end].parse().unwrap());
        }
        i = start.max(i + p + 1);
    }
    out
}

#[test]
fn every_documented_invariant_has_a_test_and_vice_versa() {
    let root = common::repo_root();
    let doc = std::fs::read_to_string(root.join("docs/reference/architecture.md"))
        .expect("architecture.md");
    // Only the registry table declares invariants; prose elsewhere may reference them.
    let table = doc
        .split("## 8b.")
        .nth(1)
        .expect("architecture.md must carry the §8b invariant registry");
    let documented = ids(table, "INV-");

    // WALK the workspace sources, do not list files — the old tree's hand-listed set
    // silently emptied when files moved, and the registry reported every INV as untested.
    let mut tested = BTreeSet::new();
    for krate in std::fs::read_dir(root.join("crates")).expect("crates/") {
        let src = krate.expect("dir entry").path().join("src");
        for p in common::walk(&src, "rs") {
            if let Ok(text) = std::fs::read_to_string(&p) {
                tested.extend(ids(&text, "fn inv_"));
            }
        }
    }

    let undocumented: Vec<_> = tested.difference(&documented).collect();
    let untested: Vec<_> = documented.difference(&tested).collect();
    assert!(
        untested.is_empty(),
        "INV-{untested:?} documented in architecture.md §8b with no `inv_<n>_*` test. \
         Either write the test or delete the claim — an unverifiable invariant is what \
         made the old tree's §8 wrong."
    );
    assert!(
        undocumented.is_empty(),
        "`inv_{undocumented:?}_*` test(s) exist with no entry in architecture.md §8b. \
         An invariant worth testing is worth stating where a reader will look for it."
    );
    assert!(!documented.is_empty(), "the registry must not be empty");
}
