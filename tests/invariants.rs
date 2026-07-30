//! The INV-n registry check: every invariant documented in `docs/ARCHITECTURE.md` §8b must
//! have a test named `inv_<n>_*`, and every such test must be documented there.
//!
//! This exists because prose drifted from behaviour repeatedly and nothing noticed. §8
//! claimed "a compute launch before its bytes land is ruled out" long after that stopped
//! being true, and a reader would reasonably have skipped a check on the strength of it.
//! A doc claim nobody can verify is worse than no claim; this makes the orphan — in either
//! direction — a failing test rather than an archaeology exercise.
use std::collections::BTreeSet;

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
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let doc = std::fs::read_to_string(root.join("docs/ARCHITECTURE.md")).expect("ARCHITECTURE.md");
    // Only the registry table declares invariants; prose elsewhere may reference them.
    let table = doc
        .split("## 8b.")
        .nth(1)
        .expect("ARCHITECTURE.md must carry the §8b invariant registry");
    let documented = ids(table, "INV-");

    let mut tested = BTreeSet::new();
    for f in ["src/math.rs", "src/hybrid.rs", "src/gpustream.rs", "src/pin.rs", "src/gpu.rs"] {
        if let Ok(src) = std::fs::read_to_string(root.join(f)) {
            tested.extend(ids(&src, "fn inv_"));
        }
    }

    let undocumented: Vec<_> = tested.difference(&documented).collect();
    let untested: Vec<_> = documented.difference(&tested).collect();
    assert!(
        untested.is_empty(),
        "INV-{untested:?} documented in ARCHITECTURE.md §8b with no `inv_<n>_*` test. \
         Either write the test or delete the claim — an unverifiable invariant is what \
         made §8 wrong."
    );
    assert!(
        undocumented.is_empty(),
        "`inv_{undocumented:?}_*` test(s) exist with no entry in ARCHITECTURE.md §8b. \
         An invariant worth testing is worth stating where a reader will look for it."
    );
    assert!(!documented.is_empty(), "the registry must not be empty");
}
