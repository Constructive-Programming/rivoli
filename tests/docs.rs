//! The docs registry check: every file under `docs/` carries `status:` + `verdict:` front
//! matter, and `docs/00-orientation/INDEX.md` lists it with the SAME verdict.
//!
//! This is the INV-n trick (see `tests/invariants.rs`) applied to documentation, and it
//! exists for the same reason: prose drifted from reality repeatedly and nothing noticed.
//! the old `docs/README.md` described `--pilot-k`, a flag that never existed under that
//! name;
//! `quant.rs` cited `docs/int3.md`, a file never committed; `PERF.md` claimed the engine was
//! compute-bound for weeks after the measurement that inverted it. Each was a doc nobody
//! could verify — and an unverifiable claim is worse than no claim, because a reader
//! reasonably skips a check on the strength of it.
//!
//! What this CANNOT check is whether a verdict is true. It checks that a verdict exists, is
//! classified, and is the same in both places — which is what makes a wrong one a visible
//! edit rather than a silent drift.

#![allow(clippy::expect_used)] // a missing doc file should panic loudly, not degrade

use std::path::{Path, PathBuf};

const INDEX: &str = "docs/00-orientation/INDEX.md";
const STATUSES: [&str; 5] = [
    "live",
    "closed-negative",
    "closed-shipped",
    "closed-mixed",
    "data",
];

fn docs_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs")
}

/// Every `.md` under `docs/`, repo-relative, sorted.
fn markdown_files() -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "md") {
                out.push(p);
            }
        }
    }
    let mut v = Vec::new();
    walk(&docs_root(), &mut v);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out: Vec<String> = v
        .iter()
        .map(|p| {
            p.strip_prefix(root)
                .unwrap_or(p)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    out.sort();
    out
}

/// `(status, verdict)` from a file's front matter, or `None` if it has none.
fn front_matter(body: &str) -> Option<(String, String)> {
    let rest = body.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    let mut status = None;
    let mut verdict = None;
    for line in rest[..end].lines() {
        if let Some(v) = line.strip_prefix("status:") {
            status = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("verdict:") {
            verdict = Some(v.trim().to_string());
        }
    }
    Some((status?, verdict?))
}

#[test]
fn every_doc_declares_a_status_and_a_verdict() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bad = Vec::new();
    for f in markdown_files() {
        let body = std::fs::read_to_string(root.join(&f)).expect(&f);
        match front_matter(&body) {
            None => bad.push(format!("{f}: no `status:`/`verdict:` front matter")),
            Some((s, v)) => {
                if !STATUSES.contains(&s.as_str()) {
                    bad.push(format!("{f}: status `{s}` is not one of {STATUSES:?}"));
                }
                if v.len() < 20 {
                    bad.push(format!("{f}: verdict is too short to rule the file out"));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "docs missing or malforming front matter:\n  {}\n\nEvery doc states what it is and \
         whether it is still true. A reader decides what NOT to open from the verdict.",
        bad.join("\n  ")
    );
}

#[test]
fn the_index_lists_every_doc_with_a_matching_verdict() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let index = std::fs::read_to_string(root.join(INDEX)).expect(INDEX);

    // Verdicts in the index are table cells; compare on prose with markdown emphasis and
    // link syntax stripped, so `**5.120**` in one place and `5.120` in the other is not a
    // failure. What must not drift is the CLAIM.
    let norm = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,%-→>".contains(*c))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let index_norm = norm(&index);

    let mut missing = Vec::new();
    let mut mismatched = Vec::new();
    for f in markdown_files() {
        if f == INDEX {
            continue; // the index does not list itself
        }
        let body = std::fs::read_to_string(root.join(&f)).expect(&f);
        let Some((_, verdict)) = front_matter(&body) else {
            continue; // reported by the test above
        };
        let name = f.rsplit('/').next().unwrap_or(&f);
        let linked = index.contains(name) || index.contains(&f);
        if !linked {
            missing.push(f.clone());
            continue;
        }
        // TOUR is linked from a prose row rather than a verdict table; its verdict is its
        // own front matter and nothing duplicates it.
        if f.ends_with("TOUR.md") {
            continue;
        }
        if !index_norm.contains(&norm(&verdict)) {
            mismatched.push(format!("{f}\n      front matter: {verdict}"));
        }
    }
    assert!(
        missing.is_empty(),
        "docs not listed in {INDEX}:\n  {}\n\nA doc nobody can find from the index is a doc \
         that will be rewritten from scratch by the next person.",
        missing.join("\n  ")
    );
    assert!(
        mismatched.is_empty(),
        "verdict drift between a doc and {INDEX}:\n  {}\n\nUpdate BOTH. The index is what a \
         reader uses to decide not to open the file, so a stale row there is worse than a \
         stale file.",
        mismatched.join("\n  ")
    );
}
