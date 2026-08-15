//! The workspace duplication gate. Ported from the old tree's `build.rs` (pinned reference
//! `wt/glimmer-s2` @ 6b7f496), where its semantics were measured; the comments carrying
//! those measurements travel with the code.
//!
//! It lives in the CLI crate because cli is in every workspace build — `cargo test
//! --workspace`, `cargo check -p rivoli`, the featureless CI job — so the gate stays armed
//! no matter which slice of the workspace someone builds. The scan set is the whole
//! `crates/` tree (every crate's `src`, `tests`, and build script), one list serving as
//! both the scan set and cargo's rerun set so the two cannot drift.
//!
//! The two outcomes are told apart by exit code, never by parsing the report:
//!
//! - `--exitCode 7` is jscpd's own "clones were found" signal and fires on
//!   `clones.length > 0`. `--threshold 0` is the obvious alternative and is WRONG for
//!   "strictly forbidden": it compares a percentage rounded to 2dp, so a small enough
//!   clone in a large enough tree reads as 0.00% and passes. It also throws instead of
//!   returning, so it exits 1 — indistinguishable from the tool being absent.
//! - Missing package, unreadable config, anything else: exit 1, which warns and carries on.
//!
//! `npx --no` is load-bearing: without it npx DOWNLOADS jscpd from the network mid-build.
//! So is the bare `--` — npm otherwise claims `--exitCode` as its own flag and refuses to
//! run ("Unknown cli config", exit 1), which this script would have read as "absent" and
//! skipped silently forever. CI pins jscpd's version and separately proves the gate is
//! armed (scanned-file count derived from `git ls-files`, never a constant — a constant
//! floor bricks a small tree and undershoots a grown one).

#![allow(clippy::expect_used)] // a build script that cannot find the workspace root should die loudly

use std::path::Path;
use std::process::Command;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/cli has a workspace root two levels up")
        .to_path_buf();

    // ONE list: the scan set and cargo's rerun set. `crates` covers every member's src,
    // tests, and build scripts; vendored binary fixtures under tests/ are not `.rs`, so
    // `.jscpd.json`'s `format: ["rust"]` skips them.
    const SCAN: &[&str] = &["crates"];
    for p in SCAN {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join(".jscpd.json").display()
    );

    // `-c` is explicit on purpose. jscpd's default is ".jscpd.json in <path>", and <path>
    // here is `crates` — a silent fall-back to the built-in minTokens 50 would leave the
    // gate more than three times looser than the file that is supposed to govern it.
    let out = match Command::new("npx")
        .current_dir(&root)
        .args([
            "--no",
            "--",
            "jscpd",
            "-c",
            ".jscpd.json",
            "--exitCode",
            "7",
        ])
        .args(SCAN)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            println!("cargo:warning=jscpd not run ({e}); Rust duplication unchecked");
            return;
        }
    };

    // A CLEAN RESULT IS ONLY MEANINGFUL ON A RUSTFMT-CLEAN TREE — the gate's correctness
    // precondition, not a style preference. jscpd tokenizes: two blocks that differ only
    // in line breaking tokenize differently enough to fall under `minTokens`. Measured in
    // the old tree 2026-08-06: 0 clones reported with 680 rustfmt hunks outstanding, and
    // 52 the moment `cargo fmt` ran. A WARNING, not a hard failure: `cargo build` on a
    // tree someone is mid-edit in must not refuse, and CI gates `cargo fmt --check` in
    // its own step anyway.
    let fmt_clean = Command::new("cargo")
        .current_dir(&root)
        .args(["fmt", "--check", "--quiet"])
        .output()
        .is_ok_and(|o| o.status.success());
    if !fmt_clean {
        println!(
            "cargo:warning=tree is not rustfmt-clean, so the jscpd result below is a LOWER \
             BOUND -- formatting differences hide clones from the tokenizer (measured in the \
             old tree 2026-08-06: 0 reported at 680 outstanding hunks, 52 after `cargo fmt`)"
        );
    }

    soft_line_cap(&root);

    match out.status.code() {
        Some(0) => {}
        // BOTH streams: the clone list is on stdout, but an invocation-level complaint can
        // land on either.
        Some(7) => panic!(
            "\n\njscpd found duplicated Rust. Duplicates are FORBIDDEN here, not \
             budgeted — .jscpd.json carries no `threshold`:\n\n{}\n{}\n",
            String::from_utf8_lossy(&out.stdout).trim(),
            String::from_utf8_lossy(&out.stderr).trim()
        ),
        _ => println!(
            "cargo:warning=jscpd did not run ({}); Rust duplication unchecked. Run \
             `npx --no -- jscpd -c .jscpd.json crates` at the workspace root to see why.",
            out.status
        ),
    }
}

/// The SOFT line cap: files over 800 lines warn on every build — the hard 1200 cap lives
/// in `tests/line_limit.rs`. A build-script warning, not a test eprintln, because libtest
/// captures test output and a captured warning is no warning (recorded lesson). The
/// contract: the next edit to a warned file should shrink it, not grow it.
fn soft_line_cap(root: &std::path::Path) {
    let mut stack = vec![root.join("crates")];
    while let Some(dir) = stack.pop() {
        // Every entry either descends or gets the per-file check, which filters
        // non-sources itself — one call per entry keeps this loop a single decision.
        // (A partition-based form was tried and was a token-for-token jscpd clone of
        // the test-side `common::walk`; the two must not converge textually.)
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            match e.path() {
                p if p.is_dir() => stack.push(p),
                p => warn_if_over_soft_cap(root, &p),
            }
        }
    }
}

/// One file's check: sources over 800 lines draw the warning (non-sources are skipped
/// here, which is what keeps the walk above branchless); the hard 1200 cap lives in
/// `tests/line_limit.rs`.
fn warn_if_over_soft_cap(root: &std::path::Path, p: &std::path::Path) {
    const SOFT: usize = 800;
    let source = p
        .extension()
        .is_some_and(|x| ["rs", "hip", "hpp", "py", "sh"].iter().any(|e| x == *e));
    if !source {
        return;
    }
    let lines = std::fs::read_to_string(p)
        .map(|s| s.lines().count())
        .unwrap_or(0);
    if lines > SOFT {
        println!(
            "cargo:warning={} is {lines} lines (soft cap {SOFT}) — the next edit here \
             should refactor it smaller, and 1200 is a hard gate",
            p.strip_prefix(root).unwrap_or(p).display()
        );
    }
}
