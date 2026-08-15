//! Source-file line caps, owner-set 2026-08-15 (800 hard → 1200 hard → the final form
//! the same day): **1200 is the HARD cap** (this test, cannot be reached) and **800 is
//! the SOFT cap** — crossing it emits a `cargo:warning` from `crates/cli/build.rs` on
//! every build, because a warning inside a passing libtest is captured and invisible
//! (this repo's recorded lesson), while build-script warnings print where the editor of
//! the file will actually see them. The soft warning's contract: the NEXT edit to that
//! file should shrink it. CodeScene still binds independently below both caps (Low
//! Cohesion fires from ~605 non-comment lines at LCOM4 >= 3, measured during the wave).
//!
//! Scope: everything the tree AUTHORS under `crates/` — Rust, HIP kernels and headers,
//! the anchor drivers, the regeneration scripts. Vendored binary fixtures have no lines;
//! docs are prose with their own registry. The limit is a ceiling on FILES, not on code:
//! the enforced move is a split by cohesion (module out, sibling test binary out), never
//! deletion of the comments that carry this repo's measurements.

#![allow(clippy::expect_used)] // meta-gate: panic loudly

mod common;

const LIMIT: usize = 1200;
const EXTS: [&str; 5] = ["rs", "hip", "hpp", "py", "sh"];

#[test]
fn no_authored_source_file_exceeds_the_line_limit() {
    let root = common::repo_root();
    let mut over = Vec::new();
    let mut seen = 0usize;
    for ext in EXTS {
        for p in common::walk(&root.join("crates"), ext) {
            let body =
                std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
            seen += 1;
            let lines = body.lines().count();
            if lines > LIMIT {
                let rel = p.strip_prefix(&root).unwrap_or(&p).display().to_string();
                over.push(format!("{lines:5} {rel}"));
            }
        }
    }
    // Anti-vacuity: the walk found the workspace (the .rs population alone is >40).
    assert!(seen > 40, "walked only {seen} source files — wrong tree");
    over.sort_by(|a, b| b.cmp(a));
    assert!(
        over.is_empty(),
        "\n\n{} source file(s) exceed {LIMIT} lines:\n  {}\n\nSplit by cohesion — a module \
         or sibling test binary moves out whole, comments travelling with their code. Do \
         not shrink by deleting the measurements.",
        over.len(),
        over.join("\n  ")
    );
}
