//! The docs registry check: every file under `docs/` carries `status:` + `scope:` +
//! `verdict:` front matter, and `docs/00-orientation/INDEX.md` lists it with the SAME
//! verdict and shows the same scope.
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

mod common;

#[path = "common/k3_tolerance.rs"]
mod k3_tolerance;

const INDEX: &str = "docs/00-orientation/INDEX.md";
const STATUSES: [&str; 5] = [
    "live",
    "closed-negative",
    "closed-shipped",
    "closed-mixed",
    "data",
];
// Whose evidence backs the verdict. A closed verdict rules its question out only for its
// scope: npu-offload.md's closed-negative was measured on GLM-5.2 and says nothing about
// the V4 port (2026-08-07, the correction that motivated this field).
const SCOPES: [&str; 5] = ["glm", "v4", "k3", "engine", "glimmer"];

fn docs_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs")
}

/// Every `.md` under `docs/`, repo-relative, sorted.
fn markdown_files() -> Vec<String> {
    let v: Vec<PathBuf> = common::walk(&docs_root(), "md");
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
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bad = Vec::new();
    for f in markdown_files() {
        let body = std::fs::read_to_string(root.join(&f)).expect(&f);
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
        // doc's cell cannot satisfy another doc's check. A row is a line starting `| [`
        // whose link target ends in this filename — prose mentions (the header's
        // `docs/README.md`, the scope paragraph naming npu-offload.md) matched a bare
        // `contains(name)` on the first red run of this check.
        // Collected, not `find`-ed. A doc with TWO rows is how a stale verdict survives: `find`
        // takes the first, and the whole-file `index_norm.contains` above is satisfied by whichever
        // one is current, so the other sits there being wrong and nothing reads it. That is not
        // hypothetical — `k3-port.md` carried two rows with DIFFERENT verdicts from 2026-08-11
        // (`c6469d1`, a cross-branch INDEX.md resolution that reintroduced a row instead of
        // replacing it) until this check was written, and it was found only because a red-proof of
        // the arm below refused to go red.
        // Matched on the row's LINK TARGET, not on the filename appearing anywhere in the row. The
        // first version of this used `contains("{name})")` and reported `anchor.md` as duplicated,
        // because `k3-port.md`'s verdict cites `measurement/k3-reference/anchor.md)` in prose. The
        // `find` this replaced had the same weakness and survived it only by ordering luck —
        // `anchor.md`'s own row happens to come first, so the wrong match was never returned.
        let rows: Vec<&str> = index
            .lines()
            .filter(|l| l.starts_with("| ["))
            .filter(|l| {
                l.split_once("](")
                    .and_then(|(_, rest)| rest.split_once(')'))
                    .is_some_and(|(target, _)| target.rsplit('/').next() == Some(name))
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
        // A check that LOCATES its input by matching text has three outcomes, not two: right,
        // wrong, and **absent** — and absent used to fall out of this `if let` as a silent skip.
        // A doc linked only from prose passed the `linked` test above, matched no table row, and
        // its scope went unchecked while the suite stayed green. Zero docs are in that state
        // today, which is exactly when to close it: the hole opens the first time someone links a
        // doc from a sentence instead of the table, and nothing would say so.
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

const BENCH: &str = "docs/measurement/benchmarks.md";

/// Source text with line breaks and comment leaders removed, so a citation that wrapped
/// reads as one string.
///
/// This is the whole reason the check exists as code. `benchmarks.md` is cited ~155 times
/// and almost always BY SECTION NAME, and those citations wrap: `fp8_to_i4.rs` says
/// `benchmarks.md, "int4` / `//! provenance"`. A grep for the name finds neither half, so
/// hand-verification reports a citation as resolved when it dangles — which is exactly what
/// happened when this file was compacted from 4,070 lines to 375 on 2026-08-10. Three cited
/// `###` anchors were dropped and a by-hand pass twice declared zero unresolved.
fn flatten(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let t = line.trim_start();
        // Comment leaders, longest first: `///` and `//!` must not be read as `//`.
        // `>` is in the list because a citation that wraps inside a markdown blockquote
        // continues on a `>` line, and leaving it in reads as part of the section name:
        // `"V4 fp4 > kernel-rate A/B"`.
        let t = ["//!", "///", "//", "*", "#", ">"]
            .iter()
            .find_map(|p| t.strip_prefix(p))
            .unwrap_or(t);
        out.push(' ');
        out.push_str(t.trim());
    }
    out
}

/// The quoted names following a citation: the first must open within `budget` bytes, and
/// each one after it within `glue` of the previous close.
///
/// Byte budgets are compared against `find`/`rfind` results rather than used to slice, so
/// there is no char-boundary case to get wrong — this prose is full of `→`, `·` and `≥`.
fn chained_quotes(mut s: &str, budget: usize, glue: usize) -> Vec<&str> {
    let (mut out, mut allow) = (Vec::new(), budget);
    while let Some(open) = s.find('"').filter(|&o| o <= allow && is_glue(&s[..o])) {
        let rest = &s[open + 1..];
        let Some(close) = rest.find('"') else { break };
        out.push(&rest[..close]);
        (s, allow) = (&rest[close + 1..], glue);
    }
    out
}

/// The quoted name a citation puts BEFORE the filename, if its closing `"` is within
/// `budget` bytes of the end — `"Running these benches …" in benchmarks.md`.
fn quote_before(s: &str, budget: usize) -> Option<&str> {
    let close = s
        .rfind('"')
        .filter(|&c| s.len() - c <= budget && is_glue(&s[c + 1..]))?;
    Some(s[..close].rsplit('"').next().unwrap_or(""))
}

/// Whether the text between a quote and the filename is only connective tissue.
///
/// A clause boundary means the quote belongs to a neighbouring sentence, not to the
/// citation. `cache.rs` writes `benchmarks.md, "2Q kin/kout re-sweep". 1. **"Kout is the
/// axis that matters" is now false**` — the second quote is the claim being REFUTED, and
/// distance alone reads it as a second cited section. `dot_bench.rs` has the mirror case
/// before the filename: `separates "bit-identical" from "within tolerance". See
/// benchmarks.md, "A fingerprint …"`.
///
/// A bare `.` will not do: citers write relative paths, and `"int4 provenance" in
/// `../benchmarks.md`` (`cache-conditional-routing.md`) is a real citation whose glue is
/// mostly path separators. It takes a period FOLLOWED BY A SPACE to end a clause, which
/// `../` and `.md)` are not.
fn is_glue(s: &str) -> bool {
    !s.contains(". ") && !s.contains("; ")
}

/// Every quoted name cited beside `benchmarks.md` is still findable in it.
///
/// Deliberately checks "appears anywhere in the file", not "is a heading". What a reader
/// needs from `benchmarks.md, "V4 route split"` is to find that text; whether the
/// compaction left it as a `##`, a `###` or a line of prose is not the citer's business,
/// and heading-parsing would fail closed on the ones cited as `benchmarks.md's "half of
/// \`tail\` is in none of its kernels"`.
///
/// Both orders are scanned, because both are written: `benchmarks.md "V4 route split"` and
/// `"Running these benches — detach anything multi-cell" in benchmarks.md`
/// (`tests/ppl-sweep-powered.sh`). A forward-only scan reads the *following* sentence's
/// quote as that citation's name and reports a dangle that is not one.
///
/// The budgets are measured, not guessed. Over every citation in the tree the nearest quote
/// sits at most 34 bytes after the filename (`modes.md`, citing "`top-m` offline screen")
/// and at most 23 before it; the next candidate in either direction is at 50, and everything
/// from there up is a quote belonging to a neighbouring clause — "too small to measure",
/// "template the kernel wider", "I ran setsid". The gap between 34 and 50 is where this
/// check lives. A wider budget cries wolf, and a gate that cries wolf gets an allowlist
/// bolted on and then stops being read.
///
/// One citation can name SEVERAL sections, so the forward scan keeps taking quotes while
/// each is separated from the last by nothing but glue (`and`, `,`, `§`, `/`). This is not
/// a refinement: `modes.md` cites `§"`top-m` offline screen" and §"DECISION"`, and a
/// first-quote-only scan is satisfied by the name that resolves and never looks at the one
/// that does not. That is precisely how the fourth dropped anchor stayed hidden through two
/// hand-verification passes.
#[test]
fn benchmarks_citations_resolve() {
    const FWD: usize = 40;
    const BACK: usize = 30;
    /// Connective allowed between two names in one citation — ` and §`, `", "`, `" / "`.
    const GLUE: usize = 12;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let body = std::fs::read_to_string(root.join(BENCH)).expect("read benchmarks.md");

    // Every place a citation has ever been written: docs, both engine sources, the
    // benchmark examples, the kernels, and the two root-level orientation files.
    let mut sources: Vec<PathBuf> = ["docs", "src", "tests", "examples", "kernels"]
        .iter()
        .flat_map(|d| {
            ["md", "rs", "hip", "comp", "sh"]
                .iter()
                .flat_map(move |e| common::walk(&root.join(d), e))
        })
        .collect();
    sources.extend(["README.md", "CLAUDE.md"].map(|f| root.join(f)));
    sources.sort();

    let mut dangling = Vec::new();
    for path in sources {
        // The file cannot dangle against itself: its own cross-references are checked by
        // being in it, and its headings quote each other. This file is skipped for the
        // same reason a linter does not lint its own rule table — every string above is a
        // worked example of the syntax being matched.
        if path.ends_with("benchmarks.md") || path.ends_with("tests/docs.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue; // not UTF-8 (a fixture, a golden); it cites nothing
        };
        let flat = flatten(&text);
        let rel = path.strip_prefix(root).unwrap_or(&path).display();
        for (i, _) in flat.match_indices("benchmarks.md") {
            let (before, after) = (&flat[..i], &flat[i + "benchmarks.md".len()..]);
            let mut names = chained_quotes(after, FWD, GLUE);
            names.extend(quote_before(before, BACK));
            for name in names {
                // A citer may quote a long heading truncated with an ellipsis — `gpu.rs`
                // cites `"Batch coalescing…"`. That is a prefix, so match it as one;
                // demanding the whole title would be red on a correct citation.
                let name = name.trim_end_matches('…').trim_end_matches("...");
                if !name.is_empty() && !body.contains(name) {
                    dangling.push(format!("{rel}: {name:?}"));
                }
            }
        }
    }
    dangling.dedup();
    assert!(
        dangling.is_empty(),
        "citations of {BENCH} that resolve to nothing in it:\n  {}\n\nEither restore the \
         text under its old name or repoint the citer. Compacting the file is allowed and \
         encouraged -- silently taking a name out from under 155 references is not, because \
         the reader who follows one finds an absence and cannot tell a moved verdict from a \
         retracted one.",
        dangling.join("\n  ")
    );
}

/// `benchmarks.md` stays a verdicts file rather than growing back into a journal.
///
/// It reached 4,070 lines by being append-only, and the rule that replaced that
/// (`.claude/skills/rivoli-docs/SKILL.md`) is prose an agent skims. This is the same
/// argument the INV-n registry makes: an unenforced convention drifts until something
/// fails on it. A cap on the whole file is enough -- no lines-per-heading ratio, which
/// passes a file that regrows by adding empty headings and needs two failure modes
/// explained.
#[test]
fn benchmarks_stays_compact() {
    /// 492 lines after the 2026-08-10 compaction and the same day's restoration pass, plus
    /// room for a few rounds. Hitting this means retire a superseded section, not raise the
    /// number.
    ///
    /// It was first set at 450 against the 375 lines the compaction left, and went red the
    /// moment a review found the compaction had cut four cited anchors and nine fingerprints
    /// that no longer existed anywhere. Raising it then was correct — the cap exists to stop
    /// the file regrowing into a journal, not to force out artefacts that cost a device slot
    /// to re-derive. Raising it to avoid triage is the failure; five uncited ceremony
    /// sections came out in the same pass.
    const CAP: usize = 520;
    let n = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(BENCH))
        .expect("read benchmarks.md")
        .lines()
        .count();
    assert!(
        n <= CAP,
        "{BENCH} is {n} lines (cap {CAP}). Retire a superseded section instead of raising \
         the cap: a measurement that a later round replaced costs every future reader the \
         time to work out which one is current."
    );
}

/// `CLAUDE.md`'s jscpd-exemption count is derived here, not asserted there.
///
/// The count was wrong three times in two days across two parallel branches — stale at Ten,
/// re-derived as Fourteen, then asserted as Thirteen on a reading of jscpd that turned out to be
/// WRONG. Each round the fix was another hand-count, and each hand-count is a fresh chance to be
/// wrong; this is the last one.
///
/// **What was measured, 2026-08-11, jscpd 4.0.5, on synthetic pairs carrying a 141-token
/// duplicate.** These are the semantics the checks below are built on, and the earlier version of
/// this comment asserted the opposite of two of them from no measurement at all:
///
/// | form | exempts? |
/// |---|---|
/// | `// marker` (bare) | YES — the control |
/// | `/// marker` (doc comment) | **YES** |
/// | `// … marker … ` mid-sentence prose | **YES** |
/// | inside a Rust string literal | no |
/// | a start with no matching end | **NO — one clone, same as no markers** |
///
/// So a marker is any COMMENT that contains the text, and prose about the gate is a live marker.
/// `backend/hip.rs` had exactly that: a `///` line naming the marker 62 lines above the real one,
/// pairing with the `ignore-end` 1150 lines later, so the exemption began where nobody decided it
/// should. That is why the rule enforced here is not "count the anchored form" but **the text may
/// appear ONLY on a bare marker line** — the one convention under which counting is unambiguous
/// and a prose mention is a visible edit rather than a silent widening. It is why this comment
/// says "marker" instead of spelling it.
///
/// A start with no end does NOT exempt to end of file, contrary to what this said before it was
/// measured; the balance check is kept because an unpaired start means a region someone MEANT to
/// exempt is not exempt, or a pairing crossed two regions as `hip.rs`'s did.
///
/// **This models an unpinned dependency.** `build.rs` runs `npx --no -- jscpd` with no version;
/// the review that found the error measured 5.0.11 and agreed on every row above, but a 6.x could
/// differ and nothing here would notice.
#[test]
fn the_jscpd_exemption_count_is_derived() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<PathBuf> = ["src", "tests"]
        .iter()
        .flat_map(|d| common::walk(&root.join(d), "rs"))
        .collect();
    files.push(root.join("build.rs"));
    files.sort();

    // Assembled rather than written out, so this file holds no prose copy of the marker either —
    // it is scanned by jscpd like everything else, and its own doc comment above WAS a live
    // unpaired marker until this round.
    let (open, close) = ("jscpd:ignore-", "start");
    let (m_start, m_end) = (format!("{open}{close}"), format!("{open}end"));

    let mut total = 0usize;
    let mut per_file: Vec<String> = Vec::new();
    for f in &files {
        let body = std::fs::read_to_string(f).expect("read a source file");
        let rel = f.strip_prefix(root).unwrap_or(f).to_string_lossy();
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
                // Any comment carrying the text is a marker to jscpd — `///` and mid-sentence
                // included, both measured. So the text is allowed ONLY as a bare marker opening
                // its line, and anything else is reworded rather than counted.
                assert!(
                    l.trim_start().starts_with(&bare),
                    "{rel}:{} mentions the ignore marker without being one:\n  {}\nA doc comment \
                     or mid-sentence mention IS a marker to jscpd (measured), so this silently \
                     moves where an exemption begins — `backend/hip.rs` started one 62 lines \
                     early that way. Reword it to name the marker without spelling it.",
                    n + 1,
                    l.trim()
                );
                *tally += 1;
            }
        }
        assert_eq!(
            starts, ends,
            "{rel} has {starts} start markers against {ends} end markers. An unpaired start does \
             NOT exempt to end of file (measured), but it does mean either a region someone meant \
             to exempt is not exempt, or a pairing crossed two regions. Close it."
        );
        if starts > 0 {
            total += starts;
            per_file.push(format!("`{rel}` {starts}"));
        }
    }

    // Matched as a DIGIT and the count is written only once. It first read `**Thirteen (13)**`,
    // with the test checking the parenthesised digits — which passes `**Thirteen (14)**` green, two
    // copies of a number with one of them unchecked, i.e. the very defect this test exists to
    // close. Spelling it out in words costs a word table and buys nothing a digit does not.
    // Counted, not `contains`-ed. This repo corrects docs in place with a dated note that PRESERVES
    // the superseded sentence, so the first note quoting this line verbatim would satisfy a
    // `contains` from history while the live line said another number — the same two-copies defect
    // one more level out. Exactly one occurrence, so a preserved quote has to be reworded.
    let claude = std::fs::read_to_string(root.join("CLAUDE.md")).expect("read CLAUDE.md");
    let want = format!("**{total}** regions are exempt");
    let seen = claude.matches(&want).count();
    assert_eq!(
        seen,
        1,
        "CLAUDE.md says `**{total}** regions are exempt` {seen} times; it must say it exactly once, \
         and {total} is what `src/`, `tests/` and `build.rs` carry. Per-file: {}",
        per_file.join(", ")
    );
}

/// **`anchor.md`'s verdict must carry every `Rel` value the tolerance table actually holds.**
///
/// The table is the only place these numbers are CHECKED (`tolerances_leave_room` derives each `Rel`
/// from its floor). They are then transcribed into `anchor.md`'s verdict, into the `INDEX.md` row
/// that `the_index_lists_every_doc_with_a_matching_verdict` gates against it, and into prose in three
/// files — and on 2026-08-12 a re-measurement moved three of them and left four downstream arguments
/// citing the old values, one of which stated a ratio that was 2.2x wrong in three places at once.
///
/// So this makes the doc DERIVED from the table rather than agreeing with it by hand, the same way
/// `the_jscpd_exemption_count_is_derived` does for a count. It gates the verdict line specifically
/// because `INDEX.md` already has to match that, so one assertion covers both.
#[test]
fn the_anchor_verdict_carries_the_measured_tolerances() {
    let body = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/measurement/k3-reference/anchor.md"),
    )
    .expect("anchor.md");
    let verdict = front_matter(&body).expect("anchor.md front matter").1;
    for t in k3_tolerance::TOLERANCES {
        let k3_tolerance::Policy::Rel(tol) = t.policy else {
            continue; // `ExactOnly` has no value to carry.
        };
        // TWO significant figures, which is how the verdict writes them and how the table's own
        // `FLOOR_MULT` rule says an author writes them. Rust's bare `{:e}` drops a trailing zero —
        // `6.0e-4` becomes `6e-4` — and matching that against the doc fails for the one row whose
        // mantissa ends in zero, which is exactly how this check first went red.
        let want = format!("{tol:.1e}");
        assert!(
            verdict.contains(&want),
            "anchor.md's verdict does not carry `{}`'s tolerance {want} — the table moved and the \
             doc did not. Every argument in the tree that divides by this number is now stale; \
             re-derive them rather than editing this line alone.",
            t.operator
        );
    }
}
