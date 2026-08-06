//! The matrix runners enumerate modes, cache policies, attention modes and features by
//! hand. This checks those lists still match what the engine accepts.
//!
//! It exists because that drift already happened and cost nothing to notice: `bench-matrix.sh`
//! kept `top-m` in its policy list for months after the policy was deleted from the engine,
//! so 8 of its 44 round-1 cells could only ever have died on `invalid value 'top-m'`. A
//! matrix that silently stops covering a dimension is worse than no matrix, because the
//! green summary is read as coverage.
//!
//! The runners are shell, so this parses them. That is deliberate rather than lazy: the
//! alternative is a shared list some third file owns, which is one more thing to forget to
//! update, and the assertion here is precisely "the list the script LOOPS OVER matches the
//! engine" — a list it merely declares alongside could drift from its own loop.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeSet;

/// The values inside a `clap` `value_parser = ["a", "b"]` attribute for `flag`.
fn cli_values(src: &str, flag: &str) -> BTreeSet<String> {
    const KEY: &str = "value_parser = [";
    let line = src
        .lines()
        .find(|l| l.contains(KEY) && l.contains(&format!("\"{flag}\"")))
        .unwrap_or_else(|| panic!("no value_parser list containing {flag:?} in main.rs"));
    // From the KEY, not from the first `[` — these lines open with `#[arg(`, so anchoring
    // on `[` swallows the whole attribute and yields nonsense that still parses.
    let rest = &line[line.find(KEY).unwrap() + KEY.len()..];
    let inner = &rest[..rest.find(']').expect("unterminated value_parser list")];
    inner
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The values a bash array literal `NAME=(a b c)` holds, from a matrix runner.
fn script_array(src: &str, name: &str) -> BTreeSet<String> {
    let pat = format!("{name}=(");
    let line = src
        .lines()
        .find(|l| l.trim_start().starts_with(&pat))
        .unwrap_or_else(|| panic!("no {name}=(...) array"));
    let inner = &line[line.find('(').unwrap() + 1..line.rfind(')').unwrap()];
    inner.split_whitespace().map(str::to_string).collect()
}

fn read(p: &str) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// Every mode/policy/attn the CLI accepts is exercised by the correctness matrix, and the
/// matrix names nothing the CLI would reject.
///
/// `auto` is the one deliberate exclusion: it resolves to dense or dsa at startup, so a
/// cell for it duplicates whichever it chose while hiding which that was.
#[test]
fn the_mode_matrix_covers_exactly_what_the_cli_accepts() {
    let main = read("src/main.rs");
    let cfg = read("src/artifact/config.rs");
    let sh = read("tests/mode-matrix.sh");

    let modes: BTreeSet<String> = cfg
        .lines()
        .filter_map(|l| l.split_once("\" => Ok(Mode::"))
        .filter_map(|(lhs, _)| lhs.rsplit_once('"').map(|(_, m)| m.to_string()))
        .collect();
    assert!(
        modes.len() >= 3,
        "parsed {modes:?} from Mode::parse — the match arms must have moved"
    );

    let mut attns = cli_values(&main, "dense");
    assert!(attns.remove("auto"), "--attn should still offer `auto`");

    for (dim, engine, script) in [
        ("modes", modes, script_array(&sh, "MODES")),
        (
            "policies",
            cli_values(&main, "2q"),
            script_array(&sh, "POLICIES"),
        ),
        ("attns", attns, script_array(&sh, "ATTNS")),
    ] {
        assert_eq!(
            engine, script,
            "tests/mode-matrix.sh {dim} drifted from the engine.\n  \
             engine accepts: {engine:?}\n  matrix runs:    {script:?}"
        );
    }
}

/// The same guard for `bench-matrix.sh`, which is where the drift actually happened. It
/// builds its cells in nested `for` loops rather than arrays, so this asserts the weaker
/// but sufficient property: it must not name a policy the CLI rejects.
#[test]
fn the_bench_matrix_names_no_policy_the_cli_rejects() {
    let main = read("src/main.rs");
    let sh = read("tests/bench-matrix.sh");
    let ok = cli_values(&main, "2q");
    let loop_line = sh
        .lines()
        .find(|l| l.contains("for pol in"))
        .expect("bench-matrix.sh should still loop over policies");
    let named: BTreeSet<String> = loop_line
        .split_whitespace()
        .skip_while(|w| *w != "in")
        .skip(1)
        .take_while(|w| *w != "do;" && *w != "do")
        .map(|w| w.trim_end_matches(';').to_string())
        .collect();
    let bogus: Vec<_> = named.difference(&ok).collect();
    assert!(
        bogus.is_empty(),
        "bench-matrix.sh loops over {bogus:?}, which --cache-policy rejects: every such \
         cell dies on `invalid value` and is counted as a crash, not as missing coverage"
    );
}

/// Every feature in `Cargo.toml` is a dimension of the feature matrix. Adding one and
/// forgetting the matrix is the exact shape of the `otlp` rot: a gated module nothing built.
#[test]
fn the_feature_matrix_covers_every_cargo_feature() {
    let toml = read("Cargo.toml");
    let sh = read("tests/feature-matrix.sh");

    let declared: BTreeSet<String> = toml
        .lines()
        .skip_while(|l| l.trim() != "[features]")
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('[') || l.contains("]  ="))
        .filter_map(|l| l.split_once('=').map(|(k, _)| k.trim().to_string()))
        .filter(|k| !k.is_empty() && !k.starts_with('#') && k != "default")
        .collect();
    assert!(
        declared.contains("rocm") && declared.contains("otlp"),
        "parsed {declared:?} from [features] — the section must have moved"
    );

    let mut covered = script_array(&sh, "BACKENDS");
    covered.extend(script_array(&sh, "OPTIONAL"));
    assert_eq!(
        declared, covered,
        "tests/feature-matrix.sh drifted from Cargo.toml [features].\n  \
         declared: {declared:?}\n  matrix:   {covered:?}"
    );

    // The decode sweep runs under the BACKEND features specifically — they are what changes
    // which kernels exist, so a cell can pass on one backend and not compile on another.
    // That was not hypothetical: `--features vulkan` stopped compiling the day
    // `prefill_layer_major` started calling a `copy_out_raw` the Vulkan `DeviceBuf` only had
    // under `trace`, and the prescribed `rocm,...,trace` union could never have shown it.
    //
    // Both lists hold one entry since 2026-08-06, but this compares TWO DIFFERENT FILES, so
    // it can still fire: the way to break it is to edit one script's `BACKENDS` and not the
    // other's. It costs nothing and keeps the two scripts' notion of a backend welded
    // together. (An earlier version of this comment claimed the assertion was vacuous. It is
    // not, and saying so was an invitation to delete a live check.)
    //
    // The assertion above it — BACKENDS ∪ OPTIONAL == Cargo.toml's [features] — is the one
    // that did the work here: it made deleting the `vulkan` feature a test failure until
    // both scripts were updated.
    assert_eq!(
        script_array(&sh, "BACKENDS"),
        script_array(&read("tests/mode-matrix.sh"), "BACKENDS"),
        "the feature matrix and the decode matrix disagree about what a backend is"
    );
}
