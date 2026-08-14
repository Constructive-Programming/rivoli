//! The docs registry check: every file under `docs/` carries `status:` + `scope:` +
//! `verdict:` front matter, and `docs/00-orientation/INDEX.md` lists it with the SAME
//! verdict and shows the same scope.
//!
//! Ported from the old tree (`wt/glimmer-s2` @ 6b7f496), where it exists because prose
//! drifted from reality repeatedly and nothing noticed: a README described a flag that
//! never existed under that name; a source file cited a doc never committed; a perf doc
//! claimed the engine was compute-bound for weeks after the measurement that inverted it.
//! An unverifiable claim is worse than no claim, because a reader reasonably skips a check
//! on the strength of it.
//!
//! What this CANNOT check is whether a verdict is true. It checks that a verdict exists, is
//! classified, and is the same in both places — which is what makes a wrong one a visible
//! edit rather than a silent drift.
//!
//! Two of the old file's tests are deliberately NOT here yet, on the "a gate lands with
//! what makes it non-vacuous" rule: the benchmarks-citation resolver and the benchmarks
//! line cap both gate `docs/measurement/benchmarks.md`, which this tree will not have
//! until the first measured number (M4/M5). Port them with that file.

#![allow(clippy::expect_used)] // a missing doc file should panic loudly, not degrade

use std::path::PathBuf;

mod common;

const INDEX: &str = "docs/00-orientation/INDEX.md";
const STATUSES: [&str; 5] = [
    "live",
    "closed-negative",
    "closed-shipped",
    "closed-mixed",
    "data",
];
// Whose evidence backs the verdict. A closed verdict rules its question out only for its
// scope: in the old tree, npu-offload.md's closed-negative was measured on GLM-5.2 and
// says nothing about the V4 port (2026-08-07, the correction that motivated this field).
const SCOPES: [&str; 5] = ["glm", "v4", "k3", "engine", "glimmer"];

/// Every `.md` under `docs/`, repo-relative, sorted.
fn markdown_files() -> Vec<String> {
    let root = common::repo_root();
    let v: Vec<PathBuf> = common::walk(&root.join("docs"), "md");
    let mut out: Vec<String> = v
        .iter()
        .map(|p| {
            p.strip_prefix(&root)
                .unwrap_or(p)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    out.sort();
    out
}

/// `(status, verdict, scope)` from a file's front matter, or `None` if it has none.
/// `scope` stays an `Option` so a missing one is its own finding, not a missing-front-matter.
fn front_matter(body: &str) -> Option<(String, String, Option<String>)> {
    let rest = body.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    let mut status = None;
    let mut verdict = None;
    let mut scope = None;
    for line in rest[..end].lines() {
        if let Some(v) = line.strip_prefix("status:") {
            status = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("verdict:") {
            verdict = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("scope:") {
            scope = Some(v.trim().to_string());
        }
    }
    Some((status?, verdict?, scope))
}

#[test]
fn every_doc_declares_a_status_and_a_verdict() {
    let root = common::repo_root();
    let files = markdown_files();
    // Anti-vacuity: an empty docs/ would make this test a no-op that looks like coverage.
    assert!(
        !files.is_empty(),
        "no .md files under docs/ — the walk found nothing, which is a failure of the \
         check, not a clean tree"
    );
    let mut bad = Vec::new();
    for f in &files {
        let body = std::fs::read_to_string(root.join(f)).expect(f);
        match front_matter(&body) {
            None => bad.push(format!("{f}: no `status:`/`verdict:` front matter")),
            Some((s, v, sc)) => {
                if !STATUSES.contains(&s.as_str()) {
                    bad.push(format!("{f}: status `{s}` is not one of {STATUSES:?}"));
                }
                if v.len() < 20 {
                    bad.push(format!("{f}: verdict is too short to rule the file out"));
                }
                match sc {
                    None => bad.push(format!(
                        "{f}: no `scope:` — whose evidence backs this? one of {SCOPES:?}"
                    )),
                    Some(sc) if !SCOPES.contains(&sc.as_str()) => {
                        bad.push(format!("{f}: scope `{sc}` is not one of {SCOPES:?}"));
                    }
                    Some(_) => {}
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
    let root = common::repo_root();
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
    let mut scope_drift = Vec::new();
    for f in markdown_files() {
        if f == INDEX {
            continue; // the index does not list itself
        }
        let body = std::fs::read_to_string(root.join(&f)).expect(&f);
        let Some((_, verdict, scope)) = front_matter(&body) else {
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
        // The scope is a table cell (`| glm |`), matched on the doc's own index row so one
        // doc's cell cannot satisfy another doc's check. Rows are collected, not `find`-ed:
        // a doc with TWO rows is how a stale verdict survives — the old tree's k3-port.md
        // carried two rows with DIFFERENT verdicts for days (a cross-branch INDEX merge
        // reintroduced a row instead of replacing it), found only because a red-proof of
        // this arm refused to go red. Matched on the row's LINK TARGET, not the filename
        // appearing anywhere in the row: two ports ship an `anchor.md`, and a basename
        // match would hand one doc the OTHER doc's row, letting the scope check pass by
        // reading a cell belonging to a different port. `ends_with` and not `==` because
        // the row's target is relative to `docs/00-orientation/` while `f` carries the
        // `docs/` prefix.
        let rows: Vec<&str> = index
            .lines()
            .filter(|l| l.starts_with("| ["))
            .filter(|l| {
                l.split_once("](")
                    .and_then(|(_, rest)| rest.split_once(')'))
                    .is_some_and(|(target, _)| f.ends_with(target.trim_start_matches("../")))
            })
            .collect();
        assert!(
            rows.len() <= 1,
            "{f} has {} rows in {INDEX}. Only the first is ever read, so the others are \
             unverifiable prose — and the one a reader happens to scroll to decides what they \
             believe. Delete the stale one.",
            rows.len()
        );
        let row = rows.first().copied();
        // A check that LOCATES its input by matching text has three outcomes, not two:
        // right, wrong, and ABSENT — and absent must not fall out as a silent skip. A doc
        // linked only from prose passes the `linked` test above, matches no table row, and
        // its scope would go unchecked while the suite stayed green.
        match (scope, row) {
            (Some(sc), Some(row)) if !row.contains(&format!("| {sc} |")) => {
                scope_drift.push(format!("{f}: front matter says `{sc}`, index row lacks it"));
            }
            (Some(_), None) => scope_drift.push(format!(
                "{f}: linked in the index but from no `| [` table row, so nothing checks its scope"
            )),
            _ => {}
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
    assert!(
        scope_drift.is_empty(),
        "scope drift between a doc and {INDEX}:\n  {}\n\nA verdict without a visible scope \
         reads as engine-wide, and closed-on-GLM has already been mistaken for closed.",
        scope_drift.join("\n  ")
    );
}

/// `CLAUDE.md`'s jscpd-exemption count is derived here, not asserted there.
///
/// In the old tree the count was wrong three times in two days across two parallel
/// branches; each fix was another hand-count, and each hand-count is a fresh chance to be
/// wrong. The semantics below were MEASURED there (jscpd 4.0.5, synthetic pairs carrying a
/// 141-token duplicate): a bare `//`, a `///` doc comment, and a mid-sentence prose
/// mention all act as live markers; only a string literal does not; an unpaired start does
/// NOT exempt to end of file. So the marker text may appear ONLY on a bare marker line —
/// the one convention under which counting is unambiguous and a prose mention is a visible
/// edit rather than a silent widening. This tree starts at ZERO exemptions; the first one
/// re-earns its marker with an argument in place, and this test keeps the ledger honest.
#[test]
fn the_jscpd_exemption_count_is_derived() {
    let root = common::repo_root();
    let mut files: Vec<PathBuf> = common::walk(&root.join("crates"), "rs");
    files.sort();
    // Anti-vacuity: the walk found the workspace, not an empty directory.
    assert!(
        files.len() >= 6,
        "walked {} .rs files under crates/ — fewer than the six crate roots, so the walk \
         is looking at the wrong tree",
        files.len()
    );

    // Assembled rather than written out, so this file holds no prose copy of the marker
    // either — it is scanned by jscpd like everything else.
    let (open, close) = ("jscpd:ignore-", "start");
    let (m_start, m_end) = (format!("{open}{close}"), format!("{open}end"));

    let mut total = 0usize;
    let mut per_file: Vec<String> = Vec::new();
    for f in &files {
        let body = std::fs::read_to_string(f).expect("read a source file");
        let rel = f.strip_prefix(&root).unwrap_or(f).to_string_lossy();
        let mut starts = 0usize;
        let mut ends = 0usize;
        for (n, l) in body.lines().enumerate() {
            for (marker, bare, tally) in [
                (&m_start, format!("// {m_start}"), &mut starts),
                (&m_end, format!("// {m_end}"), &mut ends),
            ] {
                if !l.contains(marker.as_str()) {
                    continue;
                }
                assert!(
                    l.trim_start().starts_with(&bare),
                    "{rel}:{} mentions the ignore marker without being one:\n  {}\nA doc \
                     comment or mid-sentence mention IS a marker to jscpd (measured in the \
                     old tree), so this silently moves where an exemption begins. Reword it \
                     to name the marker without spelling it.",
                    n + 1,
                    l.trim()
                );
                *tally += 1;
            }
        }
        assert_eq!(
            starts, ends,
            "{rel} has {starts} start markers against {ends} end markers. An unpaired start \
             does NOT exempt to end of file (measured), but it does mean either a region \
             someone meant to exempt is not exempt, or a pairing crossed two regions. Close it."
        );
        if starts > 0 {
            total += starts;
            per_file.push(format!("`{rel}` {starts}"));
        }
    }

    // Matched as a DIGIT, written only once. `contains` would be satisfied by a dated
    // correction quoting the superseded line; exactly-one forces a preserved quote to be
    // reworded — the same two-copies defect one more level out.
    let claude = std::fs::read_to_string(root.join("CLAUDE.md")).expect("read CLAUDE.md");
    let want = format!("**{total}** regions are exempt");
    let seen = claude.matches(&want).count();
    assert_eq!(
        seen,
        1,
        "CLAUDE.md says `**{total}** regions are exempt` {seen} times; it must say it \
         exactly once, and {total} is what crates/ carries. Per-file: {}",
        per_file.join(", ")
    );
}
