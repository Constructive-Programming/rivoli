//! The break corpus, checked without running it.
//!
//! `tests/refactor-gates/run-breaks.sh` costs fourteen `--release --features rocm` builds,
//! five of them device-bound, so it is run rarely — which is exactly when a corpus rots.
//! Everything here runs in milliseconds with no GPU and no cargo, and converts
//! "ANCHOR MISSING, hours in" into a red `cargo test`.
//!
//! Each assertion below exists because the corpus shipped that defect:
//!
//! - **`expect` matching the test's own name.** Row 7 was `EXPECT = "stream"` against
//!   `the_attention_block_is_entirely_on_its_stream`, and libtest echoes the test name at
//!   least twice on failure. The subject check was unconditionally true, so any red for any
//!   reason — GPU contention, an unrelated assertion — read as "message matches". It was
//!   also the one row whose message a commit claimed had been verified against real output.
//! - **An anchor that no longer occurs**, or occurs twice, so `replace(find, repl, 1)` edits
//!   the wrong site.
//! - **A `find` that equals its `repl`**, which applies a no-op and then reports whatever the
//!   suite already did.
//!
//! What this CANNOT check is whether a break actually makes its test fail. Only
//! `run-breaks.sh` can, and only on hardware. This is the cheap half.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use std::path::Path;

struct Row {
    file: String,
    find: String,
    repl: String,
    bin: String,
    test: Option<String>,
    expect: String,
    line: usize,
}

fn rows() -> Vec<Row> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(root.join("tests/refactor-gates/breaks.tsv"))
        .expect("read breaks.tsv");
    let v: Vec<Row> = raw
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(|(i, l)| {
            // Split on tab WITHOUT collapsing runs: an empty REPLACE column is legitimate
            // (row 1 deletes text) and `IFS=$'\t' read` collapsing it shifted every field
            // left, which is how the driver first ran the wrong test entirely.
            let f: Vec<&str> = l.split('\t').collect();
            assert!(
                f.len() >= 5,
                "breaks.tsv:{}: want 5 tab-separated columns, got {}",
                i + 1,
                f.len()
            );
            let (bin, test) = match f[3].split_once(' ') {
                Some((b, t)) => (b.to_string(), Some(t.to_string())),
                None => (f[3].to_string(), None),
            };
            Row {
                file: f[0].into(),
                find: f[1].into(),
                repl: f[2].into(),
                bin,
                test,
                expect: f[4].into(),
                line: i + 1,
            }
        })
        .collect();
    assert!(
        !v.is_empty(),
        "breaks.tsv parsed to zero rows — this test just went vacuous"
    );
    v
}

#[test]
fn every_break_anchor_occurs_exactly_once_in_a_file_that_exists() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for r in rows() {
        let p = root.join(&r.file);
        let src = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("breaks.tsv:{}: {} — {e}", r.line, r.file));
        let n = src.matches(&r.find).count();
        assert_eq!(
            n, 1,
            "breaks.tsv:{}: anchor occurs {n} times in {} (want exactly 1). \
             Zero means the corpus is stale for this tree; more than one means \
             `replace(find, repl, 1)` will edit whichever comes first, which may not be \
             the site the gate is about.\n  anchor: {:?}",
            r.line, r.file, r.find
        );
        assert_ne!(
            r.find, r.repl,
            "breaks.tsv:{}: find == repl, so the 'break' is a no-op and the row reports \
             whatever the suite already did",
            r.line
        );
    }
}

#[test]
fn every_break_names_a_test_that_exists() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for r in rows() {
        let p = root.join("tests").join(format!("{}.rs", r.bin));
        assert!(
            p.exists(),
            "breaks.tsv:{}: no test binary tests/{}.rs",
            r.line,
            r.bin
        );
        if let Some(t) = &r.test {
            let src = std::fs::read_to_string(&p).unwrap();
            // A cargo filter is a SUBSTRING match over test names, not an exact name, so
            // `every_defect_variant_reaches` legitimately selects
            // `every_defect_variant_reaches_the_all_list`. What must hold is that it selects
            // at LEAST one: a filter matching nothing prints `test result: ok. 0 passed`
            // and reads as GREEN, which is how the driver first reported a gate it never ran.
            let selected: Vec<&str> = src
                .match_indices("fn ")
                .filter_map(|(i, _)| src[i + 3..].split('(').next())
                .filter(|n| !n.contains(char::is_whitespace) && n.contains(t.as_str()))
                .collect();
            assert!(
                !selected.is_empty(),
                "breaks.tsv:{}: filter {t:?} selects NO test in tests/{}.rs. A filter that \
                 matches nothing prints `test result: ok. 0 passed` and reads as GREEN.",
                r.line,
                r.bin
            );
        }
    }
}

#[test]
fn no_expect_fragment_is_satisfiable_by_libtest_boilerplate() {
    // The corpus's contract is that a break fires red *with the right subject*. An `expect`
    // that appears in the test's own name, or in libtest's own vocabulary, cannot enforce
    // that — libtest prints `test <name> ... FAILED` and a `failures:` block on every red.
    const BOILERPLATE: &[&str] = &[
        "FAILED",
        "failures",
        "test result",
        "panicked",
        "error",
        "assertion",
        "left",
        "right",
    ];
    for r in rows() {
        assert!(
            r.expect.len() >= 8,
            "breaks.tsv:{}: expect {:?} is too short to be a subject — `inf` matched \
             `info`/`infer`/`Infinity` anywhere in build output",
            r.line,
            r.expect
        );
        if let Some(t) = &r.test {
            assert!(
                !t.contains(&r.expect),
                "breaks.tsv:{}: expect {:?} is a substring of the test NAME {:?}, so libtest \
                 echoing the name satisfies the check and any red reads as 'message matches'",
                r.line,
                r.expect,
                t
            );
        }
        for b in BOILERPLATE {
            assert!(
                !r.expect.eq_ignore_ascii_case(b),
                "breaks.tsv:{}: expect {:?} is libtest boilerplate present on every failure",
                r.line,
                r.expect
            );
        }
    }
}
