//! The CodeScene code-health gate: every Rust file in the workspace scores **10/10**.
//!
//! Modeled on `docs.rs`, not on `build.rs`, for three reasons: `cs` needs a license token
//! and periodic network (the cloud license JWT refreshes every ~3 weeks), and a box
//! without CodeScene must still *build*; per-file review costs ~0.6–1 s cold, acceptable
//! in a cached test, not on every build; and `cs review` scores whole files, so it maps
//! 1:1 onto the same walk the other meta-gates use.
//!
//! **Classification is by OUTPUT, never exit code.** Measured 2026-08-15 on cs 1.0.36:
//! unlicensed `cs review` exits 1 with a "Personal Access Token" prose message — the same
//! exit code as other failures — so exit codes cannot distinguish "tool declines to run"
//! from "file failed review". JSON with a numeric `score` is a verdict; anything else is
//! tool-absent.
//!
//! **Absent/unlicensed policy: warn-and-skip locally, hard-fail in CI.** The license
//! expiring must not brick `cargo test` on the dev box; CI sets `RIVOLI_CS_REQUIRED=1`,
//! under which tool-absent is a panic. (An env var, which the instrument rule forbids for
//! the engine — the carve-out argument is `build.rs`'s: this configures the harness, has
//! no `--help` to be absent from, and CI is the only setter.) Honest limitation: libtest
//! captures the local skip's eprintln, so a skipped run looks green in the one-line
//! summary — "skips loudly" would be a false claim. CI is the enforcement point.
//!
//! **The gate proves it can go red on every run** (P7): the vendored fixture under
//! `tests/codescene-redproof/` must score BELOW 10, so "cs silently started scoring
//! everything 10" is a failure, not a pass.
//!
//! Exemptions: none yet, deliberately — see the note at the top of the file body. The
//! argued-in-place, checked-at-both-ends machinery returns WITH the first real row and
//! its red-proof; shipping it never-red was itself a P7 violation (review 2026-08-15).

#![allow(clippy::expect_used)] // meta-gate: a broken harness should panic loudly

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

mod common;

// Exemptions: none. The both-ends checking machinery (category match via `cs check`,
// file-exists sweep) was CUT in review 2026-08-15 as an arm that had never once been red
// — against P7 — and reviewers found it would misclassify an unlicensed `cs check` as
// "stale exemption, delete the row". It returns WITH the first real exemption, plus the
// red-proof that arms it.

/// One `cs review` outcome, classified by output.
enum Cs {
    /// JSON came back with a numeric score.
    Score(f64),
    /// JSON came back with `"score": null` — cs saw the file but found no scorable code
    /// (an empty lib.rs stub). Acceptable only for near-empty files.
    NoScorableCode,
    /// Spawn failure, license prose, or unparseable output: the tool did not review.
    Absent(String),
}

fn review(args: &[&str], stdin: Option<&[u8]>, cwd: &Path) -> Cs {
    let mut cmd = Command::new("cs");
    cmd.env("CS_DISABLE_VERSION_CHECK", "1")
        .current_dir(cwd)
        .args(["review", "--output-format", "json"])
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if stdin.is_some() {
        cmd.stdin(Stdio::piped());
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return Cs::Absent(format!("spawn failed: {e}")),
    };
    if let (Some(bytes), Some(mut pipe)) = (stdin, child.stdin.take()) {
        use std::io::Write as _;
        // A closed pipe here means cs died early; the output classification below reports it.
        let _ = pipe.write_all(bytes);
    }
    let out = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return Cs::Absent(format!("wait failed: {e}")),
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    match serde_json::from_str::<serde_json::Value>(&stdout) {
        Ok(v) => match v.get("score") {
            Some(serde_json::Value::Number(n)) => Cs::Score(n.as_f64().unwrap_or(f64::NAN)),
            Some(serde_json::Value::Null) => Cs::NoScorableCode,
            _ => Cs::Absent(format!("JSON without a `score` field: {}", trunc(&stdout))),
        },
        Err(_) => Cs::Absent(format!(
            "not JSON (unlicensed cs prints PAT prose here): {} {}",
            trunc(&stdout),
            trunc(&String::from_utf8_lossy(&out.stderr))
        )),
    }
}

fn trunc(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(120).collect()
}

/// Skip-or-panic on a tool that did not run, per the absent policy above. Returns only
/// in skip mode; the caller's next statement is its `return`.
fn tool_absent(context: &str, detail: &str) {
    if std::env::var_os("RIVOLI_CS_REQUIRED").is_some() {
        panic!("CodeScene gate REQUIRED but cs did not run ({context}): {detail}");
    }
    eprintln!(
        "CodeScene gate DID NOT RUN ({context}): {detail}\n\
         code health is UNCHECKED this run. Export CS_ACCESS_TOKEN to arm it."
    );
}

/// Cache keying: FNV-1a of contents + cs version, hash owned by `rivoli_core::hash` (this
/// file carried its own copy for one build, until the anchor port brought the second copy
/// and jscpd reported the pair — the gate's first live catch in this tree).
use rivoli_core::hash::fnv1a;

fn cache_path() -> PathBuf {
    // CARGO_TARGET_DIR when the environment sets one (this box's profile does — and an
    // env var OVERRIDES .cargo/config.toml's target-dir, so reading the config would be
    // wrong twice); otherwise the workspace-local target/. Review 2026-08-15: the first
    // comment here claimed the config file was honoured — it never was, and cargo does
    // not export the config's value to test processes.
    std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| common::repo_root().join("target"), PathBuf::from)
        .join("codescene-cache.json")
}

fn cs_version() -> Option<String> {
    let out = Command::new("cs")
        .env("CS_DISABLE_VERSION_CHECK", "1")
        .arg("version")
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[test]
fn every_workspace_rust_file_scores_ten() {
    let root = common::repo_root();
    let mut files: Vec<PathBuf> = common::walk(&root.join("crates"), "rs");
    files.sort();
    // Anti-vacuity: the reviewed set is the walked set, and the walk found the workspace.
    assert!(
        files.len() >= 6,
        "walked only {} .rs files under crates/ — wrong tree",
        files.len()
    );

    let Some(version) = cs_version() else {
        tool_absent("cs version", "binary not on PATH or not executable");
        return;
    };

    let cache_file = cache_path();
    let mut cache: serde_json::Map<String, serde_json::Value> =
        std::fs::read_to_string(&cache_file)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
    // A cs upgrade invalidates every entry: the score is a function of (file, reviewer).
    if cache.get("__cs_version__").and_then(|v| v.as_str()) != Some(version.as_str()) {
        cache.clear();
        cache.insert("__cs_version__".into(), version.clone().into());
    }

    let mut reviewed = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for f in &files {
        let rel = f
            .strip_prefix(&root)
            .unwrap_or(f)
            .to_string_lossy()
            .to_string();
        let body = std::fs::read(f).expect("read a walked file");
        let hash = format!("{:016x}", fnv1a(&body));
        if cache.get(&rel).and_then(|v| v.as_str()) == Some(hash.as_str()) {
            reviewed += 1; // cached green: same bytes, same reviewer, same verdict
            continue;
        }
        match review(&[&rel], None, &root) {
            Cs::Score(s) if s >= 10.0 => {
                reviewed += 1;
                cache.insert(rel, hash.into());
            }
            Cs::Score(s) => {
                reviewed += 1;
                failures.push(format!("{rel}: score {s}"));
            }
            Cs::NoScorableCode => {
                // cs scores FUNCTIONS; a file with none (pure data: repr(C) mirrors,
                // consts, module docs) legitimately scores null at any length — the
                // first licensed run refused 79-line abi.rs on a line-count rule, which
                // was the wrong classifier (2026-08-15). Vacuity is null WITH functions
                // present: that means the reviewer skipped real code.
                let fns = String::from_utf8_lossy(&body).matches("fn ").count();
                assert!(
                    fns == 0,
                    "{rel}: cs found no scorable code but the file declares {fns} fn(s) — \
                     the reviewer is skipping real code, which is vacuity, not health"
                );
                reviewed += 1;
                cache.insert(rel, hash.into());
            }
            Cs::Absent(d) => {
                tool_absent(&rel, &d);
                return;
            }
        }
    }
    let _ = std::fs::create_dir_all(cache_file.parent().unwrap_or(Path::new(".")));
    let _ = std::fs::write(
        &cache_file,
        serde_json::to_string(&serde_json::Value::Object(cache)).unwrap_or_default(),
    );
    assert_eq!(
        reviewed,
        files.len(),
        "reviewed {reviewed} of {} files — the gate did not see the whole tree",
        files.len()
    );
    assert!(
        failures.is_empty(),
        "files below 10/10 code health:\n  {}\n\nFix the code. If the code is right and \
         the rule is wrong, add an EXEMPT row with the argument — it is checked at both \
         ends and dies when stale.",
        failures.join("\n  ")
    );
}

/// The standing red-proof: a vendored, deliberately unhealthy file must score BELOW 10.
/// If it ever comes back 10/10, the reviewer has gone blind and every green above it is
/// meaningless — that is a failure of the GATE, reported as one.
#[test]
fn the_red_proof_fixture_scores_below_ten() {
    let root = common::repo_root();
    let fixture = root.join("crates/cli/tests/codescene-redproof/bad.rs.txt");
    let body = std::fs::read(&fixture).expect("vendored red-proof fixture");
    // `.txt` keeps it out of the rustc/jscpd/walk scan sets; `--file-name` tells cs it is
    // Rust anyway (measured: cs scores stdin under the declared name).
    match review(&["--file-name", "bad.rs"], Some(&body), &root) {
        Cs::Score(s) => assert!(
            s < 10.0,
            "the deliberately unhealthy fixture scored {s}/10 — the reviewer can no \
             longer go red, so every 10/10 in the other test is unfalsifiable"
        ),
        Cs::NoScorableCode => panic!(
            "cs found no scorable code in the red-proof fixture — it is not reviewing \
             what it is handed"
        ),
        Cs::Absent(d) => tool_absent("red-proof fixture", &d),
    }
}
