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

use std::path::{Path, PathBuf};

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

/// A doc's front matter. Named fields rather than a tuple because all three travel together
/// through both tests below, and position stopped being readable at the third one.
struct FrontMatter {
    status: String,
    verdict: String,
    /// `Option` so a missing scope is its own finding, not a missing-front-matter.
    scope: Option<String>,
}

/// One walked doc: repo-relative path and body. Carried together because no check here
/// wants one without the other — every finding a body produces has to name the file, and
/// the pair passed as two loose strings is exactly how a caller swaps them.
struct Doc {
    path: String,
    body: String,
}

/// Every `.md` under `docs/`, sorted by path and read.
fn docs() -> Vec<Doc> {
    let root = common::repo_root();
    let v: Vec<PathBuf> = common::walk(&root.join("docs"), "md");
    let mut paths: Vec<String> = v
        .iter()
        .map(|p| {
            p.strip_prefix(&root)
                .unwrap_or(p)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let body = std::fs::read_to_string(root.join(&path)).expect(&path);
            Doc { path, body }
        })
        .collect()
}

impl Doc {
    /// This doc's front matter, or `None` if it has none.
    fn front_matter(&self) -> Option<FrontMatter> {
        let rest = self.body.strip_prefix("---\n")?;
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
        Some(FrontMatter {
            status: status?,
            verdict: verdict?,
            scope,
        })
    }

    /// Everything wrong with THIS doc's front matter, as reader-facing lines; empty means
    /// clean. A list and not an early panic: the first run after a docs change should show
    /// every file it broke, not the alphabetically first one.
    fn front_matter_findings(&self) -> Vec<String> {
        let f = &self.path;
        let Some(fm) = self.front_matter() else {
            return vec![format!("{f}: no `status:`/`verdict:` front matter")];
        };
        let mut bad = Vec::new();
        if !STATUSES.contains(&fm.status.as_str()) {
            let s = fm.status;
            bad.push(format!("{f}: status `{s}` is not one of {STATUSES:?}"));
        }
        if fm.verdict.len() < 20 {
            bad.push(format!("{f}: verdict is too short to rule the file out"));
        }
        match fm.scope {
            None => bad.push(format!(
                "{f}: no `scope:` — whose evidence backs this? one of {SCOPES:?}"
            )),
            Some(sc) if !SCOPES.contains(&sc.as_str()) => {
                bad.push(format!("{f}: scope `{sc}` is not one of {SCOPES:?}"));
            }
            Some(_) => {}
        }
        bad
    }

    /// The filename alone — what a prose link in the index is written with.
    fn basename(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

#[test]
fn every_doc_declares_a_status_and_a_verdict() {
    let all = docs();
    // Anti-vacuity: an empty docs/ would make this test a no-op that looks like coverage.
    assert!(
        !all.is_empty(),
        "no .md files under docs/ — the walk found nothing, which is a failure of the \
         check, not a clean tree"
    );
    let bad: Vec<String> = all.iter().flat_map(Doc::front_matter_findings).collect();
    assert!(
        bad.is_empty(),
        "docs missing or malforming front matter:\n  {}\n\nEvery doc states what it is and \
         whether it is still true. A reader decides what NOT to open from the verdict.",
        bad.join("\n  ")
    );
}

/// A verdict reduced to its CLAIM: markdown emphasis and link syntax dropped, whitespace
/// collapsed. Verdicts in the index are table cells, so `**5.120**` in one place and
/// `5.120` in the other is not a failure. What must not drift is the claim.
fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,%-→>".contains(*c))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// One `| [name](target) | … |` row of the index.
#[derive(Clone, Copy)]
struct Row<'a>(&'a str);

impl<'a> Row<'a> {
    /// The row's markdown link target, as a `docs/`-relative path.
    ///
    /// A row is located by its LINK TARGET, not by the filename appearing anywhere in the
    /// row: two ports ship an `anchor.md`, and a basename match would hand one doc the
    /// OTHER doc's row, letting the scope check pass by reading a cell belonging to a
    /// different port. Targets are written relative to `docs/00-orientation/`, hence the
    /// `../` strip; the caller then compares EXACTLY rather than by suffix —
    /// `docs/measurement/glm-reference/anchor.md` ends_with `reference/anchor.md`, so the
    /// suffix rule this carried until review 2026-08-15 would let a future
    /// `docs/reference/anchor.md` row adopt every `*-reference/anchor.md` doc.
    fn target(self) -> Option<&'a str> {
        let (_, rest) = self.0.split_once("](")?;
        let (target, _) = rest.split_once(')')?;
        Some(target.trim_start_matches("../"))
    }
}

/// `INDEX.md` raw, plus its normalized whole-file text. The two are read together on every
/// doc, and normalizing the whole index once keeps the per-doc fallback cheap.
struct Index {
    raw: String,
    norm: String,
}

impl Index {
    fn read(root: &Path) -> Self {
        let raw = std::fs::read_to_string(root.join(INDEX)).expect(INDEX);
        let norm = norm(&raw);
        Self { raw, norm }
    }

    /// The doc's OWN table row, if it has one — and a hard failure if it has two.
    ///
    /// Rows are collected, not `find`-ed: a doc with TWO rows is how a stale verdict
    /// survives — the old tree's k3-port.md carried two rows with DIFFERENT verdicts for
    /// days (a cross-branch INDEX merge reintroduced a row instead of replacing it), found
    /// only because a red-proof of this arm refused to go red.
    fn row_for(&self, doc: &Doc) -> Option<Row<'_>> {
        let want = doc.path.strip_prefix("docs/");
        let rows: Vec<Row<'_>> = self
            .raw
            .lines()
            .filter(|l| l.starts_with("| ["))
            .map(Row)
            .filter(|r| r.target().is_some_and(|t| want == Some(t)))
            .collect();
        assert!(
            rows.len() <= 1,
            "{} has {} rows in {INDEX}. Only the first is ever read, so the others are \
             unverifiable prose — and the one a reader happens to scroll to decides what they \
             believe. Delete the stale one.",
            doc.path,
            rows.len()
        );
        rows.first().copied()
    }

    /// Does the doc's verdict still appear where a reader would look for it?
    ///
    /// Row-scoped when the doc HAS a row: the old whole-index `contains` let one doc's
    /// stale verdict be satisfied by ANOTHER doc's row text (review 2026-08-15 — the same
    /// one-doc's-cell hazard the scope arm guards). The whole-index fallback survives only
    /// for prose-linked docs, which have no row to scope to.
    fn verdict_agrees(&self, row: Option<Row<'_>>, fm: &FrontMatter) -> bool {
        let want = norm(&fm.verdict);
        match row {
            Some(r) => norm(r.0).contains(&want),
            None => self.norm.contains(&want),
        }
    }
}

/// The three ways a doc can drift from the index, kept apart because each has its own
/// remedy — and the failure a reader gets should name the one they have to apply.
#[derive(Default)]
struct Drift {
    missing: Vec<String>,
    mismatched: Vec<String>,
    scope: Vec<String>,
}

impl Drift {
    /// Record whatever one doc drifts on.
    fn check(&mut self, index: &Index, doc: &Doc, fm: &FrontMatter) {
        if !(index.raw.contains(doc.basename()) || index.raw.contains(&doc.path)) {
            self.missing.push(doc.path.clone());
            return;
        }
        // TOUR is linked from a prose row rather than a verdict table; its verdict is its
        // own front matter and nothing duplicates it.
        if doc.path.ends_with("TOUR.md") {
            return;
        }
        let row = index.row_for(doc);
        if !index.verdict_agrees(row, fm) {
            let (f, v) = (&doc.path, &fm.verdict);
            self.mismatched
                .push(format!("{f}\n      front matter: {v}"));
        }
        self.check_scope(doc, fm, row);
    }

    /// The scope is a table cell (`| glm |`), matched on the doc's own row so one doc's
    /// cell cannot satisfy another doc's check.
    ///
    /// A check that LOCATES its input by matching text has three outcomes, not two: right,
    /// wrong, and ABSENT — and absent must not fall out as a silent skip. A doc linked only
    /// from prose passes the linked test in `check`, matches no table row, and its scope
    /// would go unchecked while the suite stayed green.
    fn check_scope(&mut self, doc: &Doc, fm: &FrontMatter, row: Option<Row<'_>>) {
        let f = &doc.path;
        match (fm.scope.as_deref(), row) {
            (Some(sc), Some(r)) if !r.0.contains(&format!("| {sc} |")) => {
                self.scope
                    .push(format!("{f}: front matter says `{sc}`, index row lacks it"));
            }
            (Some(_), None) => self.scope.push(format!(
                "{f}: linked in the index but from no `| [` table row, so nothing checks its scope"
            )),
            _ => {}
        }
    }

    /// One failure per KIND, each carrying why that kind matters: a reader told only "the
    /// index disagrees" fixes whichever end is easier, which is how the drift started.
    fn assert_clean(&self) {
        assert!(
            self.missing.is_empty(),
            "docs not listed in {INDEX}:\n  {}\n\nA doc nobody can find from the index is a doc \
             that will be rewritten from scratch by the next person.",
            self.missing.join("\n  ")
        );
        assert!(
            self.mismatched.is_empty(),
            "verdict drift between a doc and {INDEX}:\n  {}\n\nUpdate BOTH. The index is what a \
             reader uses to decide not to open the file, so a stale row there is worse than a \
             stale file.",
            self.mismatched.join("\n  ")
        );
        assert!(
            self.scope.is_empty(),
            "scope drift between a doc and {INDEX}:\n  {}\n\nA verdict without a visible scope \
             reads as engine-wide, and closed-on-GLM has already been mistaken for closed.",
            self.scope.join("\n  ")
        );
    }
}

#[test]
fn the_index_lists_every_doc_with_a_matching_verdict() {
    let index = Index::read(&common::repo_root());
    let mut drift = Drift::default();
    for doc in docs() {
        if doc.path == INDEX {
            continue; // the index does not list itself
        }
        // A doc with no front matter at all is reported by the test above.
        if let Some(fm) = doc.front_matter() {
            drift.check(&index, &doc, &fm);
        }
    }
    drift.assert_clean();
}

/// One jscpd ignore marker. A newtype so the two cannot be handed to a counter in the
/// wrong order — they differ by one word and the failure would be a silent miscount.
struct Marker(String);

/// One walked `.rs` file: repo-relative path and text, paired for `Doc`'s reason — a marker
/// finding is only actionable with a file and a line number in it.
struct Source {
    rel: String,
    text: String,
}

impl Source {
    /// Lines that ARE `marker` lines, asserting that no line merely MENTIONS it.
    fn count(&self, marker: &Marker) -> usize {
        let (rel, bare) = (&self.rel, format!("// {}", marker.0));
        let mut seen = 0usize;
        for (n, l) in self.text.lines().enumerate() {
            if !l.contains(marker.0.as_str()) {
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
            seen += 1;
        }
        seen
    }
}

/// The opening and closing ignore markers, assembled rather than written out so this file
/// holds no prose copy of either — it is scanned by jscpd like everything else.
struct Markers {
    open: Marker,
    close: Marker,
}

impl Markers {
    fn new() -> Self {
        let (stem, first) = ("jscpd:ignore-", "start");
        Self {
            open: Marker(format!("{stem}{first}")),
            close: Marker(format!("{stem}end")),
        }
    }

    /// How many exempt regions one file opens, with its markers checked for pairing.
    fn regions_in(&self, src: &Source) -> usize {
        let starts = src.count(&self.open);
        let ends = src.count(&self.close);
        let rel = &src.rel;
        assert_eq!(
            starts, ends,
            "{rel} has {starts} start markers against {ends} end markers. An unpaired start \
             does NOT exempt to end of file (measured), but it does mean either a region \
             someone meant to exempt is not exempt, or a pairing crossed two regions. Close it."
        );
        starts
    }
}

/// `(total exempt regions, per-file tally)` over every `.rs` file under `crates/`.
fn exemptions_under_crates(root: &Path) -> (usize, Vec<String>) {
    let mut files: Vec<PathBuf> = common::walk(&root.join("crates"), "rs");
    files.sort();
    // Anti-vacuity: the walk found the workspace, not an empty directory.
    assert!(
        files.len() >= 6,
        "walked {} .rs files under crates/ — fewer than the six crate roots, so the walk \
         is looking at the wrong tree",
        files.len()
    );

    let markers = Markers::new();
    let mut total = 0usize;
    let mut per_file: Vec<String> = Vec::new();
    for f in &files {
        let src = Source {
            rel: f.strip_prefix(root).unwrap_or(f).to_string_lossy().into(),
            text: std::fs::read_to_string(f).expect("read a source file"),
        };
        let starts = markers.regions_in(&src);
        if starts > 0 {
            total += starts;
            per_file.push(format!("`{}` {starts}", src.rel));
        }
    }
    (total, per_file)
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
    let (total, per_file) = exemptions_under_crates(&root);

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
