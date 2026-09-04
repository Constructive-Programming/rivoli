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
    let root = assert_this_checkout();

    // ONE list: the scan set and cargo's rerun set. `crates` covers every member's src,
    // tests, and build scripts; vendored binary fixtures under tests/ are not `.rs`, so
    // `.jscpd.json`'s `format: ["rust"]` skips them.
    const SCAN: &[&str] = &["crates"];
    declare_scan_reruns(&root, SCAN);

    // `None` means jscpd could not be started and its own warning has been printed; the
    // phases after it are skipped exactly as they were under this file's one-function form,
    // where an absent `npx` returned early. Preserved deliberately: changing it would make
    // the soft-cap warning appear or vanish depending on a tool's presence, and the two
    // gates are independent.
    let Some(out) = run_jscpd(&root, SCAN) else {
        return;
    };
    warn_if_format_is_a_lower_bound(&root);
    soft_line_cap(&root);
    judge_jscpd(&out);
}

/// Assert that THIS binary is running for THIS checkout, then return the workspace root.
fn assert_this_checkout() -> std::path::PathBuf {
    // THE GATE'S FIRST CLAIM IS THAT IT IS RUNNING IN THIS CHECKOUT. `env!` bakes the path
    // at the build script's OWN compile time; `var` is what cargo passes the binary it
    // decided to reuse. They differ exactly when a sibling worktree compiled this script
    // last and a shared target dir handed its binary to us — measured 2026-08-21: the
    // reused binary scanned `wave-m19/crates`, panicked, and jscpd never scanned this tree
    // at all, so a commit landed on a green nothing had established
    // (docs/measurement/gate-red-proofs.md §7e-bis). `assert!`, not `debug_assert!`: this
    // has to hold in the release build too.
    //
    // The two failure modes are kept apart because their remedies are unrelated. An ABSENT
    // variable means this binary was not started by cargo at all, so there is no build to
    // reason about; a DIFFERENT one is the 7e-bis defect. The remedy for the second is a
    // per-invocation `CARGO_TARGET_DIR`, not a config file: this box sets a box-wide
    // `CARGO_TARGET_DIR` in `/etc/security/pam_env.conf` for every login, and an env var
    // outranks `build.target-dir`, so a worktree's `.cargo/config.toml` is inert wherever
    // that variable is present (§12, where it was observed losing silently).
    let baked = env!("CARGO_MANIFEST_DIR");
    match std::env::var("CARGO_MANIFEST_DIR") {
        Err(e) => panic!(
            "CARGO_MANIFEST_DIR is unset or unreadable ({e}), so this build script was not \
             run by cargo. It cannot tell which checkout it belongs to and will not \
             pretend the duplication gate ran -- invoke it through cargo."
        ),
        Ok(live) => assert_eq!(
            live, baked,
            "stale build-script binary reused across worktrees through a shared target dir: \
             this build.rs was compiled for one checkout and run for another, so the \
             duplication gate did not scan the tree being built (gate-red-proofs.md 7e-bis). \
             Give THIS invocation its own target dir -- \
             `CARGO_TARGET_DIR=/var/cache/rivoli/target/<worktree> cargo ...`, or \
             `env -u CARGO_TARGET_DIR cargo ...` so this checkout's .cargo/config.toml can \
             bind. Writing the config alone does not move anything: the box-wide \
             CARGO_TARGET_DIR from pam_env outranks it (gate-red-proofs.md 12). \
             tests/new-worktree.sh writes that config and prints the prefix to use."
        ),
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/cli has a workspace root two levels up")
        .to_path_buf()
}

/// Tell cargo what the scan covers, so an edit under it re-runs this script.
fn declare_scan_reruns(root: &Path, scan: &[&str]) {
    for p in scan {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join(".jscpd.json").display()
    );
}

/// Run the duplication gate, or `None` if it could not be started.
fn run_jscpd(root: &Path, scan: &[&str]) -> Option<std::process::Output> {
    // `-c` is explicit on purpose. jscpd's default is ".jscpd.json in <path>", and <path>
    // here is `crates` — a silent fall-back to the built-in minTokens 50 would leave the
    // gate more than three times looser than the file that is supposed to govern it.
    match Command::new("npx")
        .current_dir(root)
        .args([
            "--no",
            "--",
            "jscpd",
            "-c",
            ".jscpd.json",
            "--exitCode",
            "7",
        ])
        .args(scan)
        .output()
    {
        Ok(o) => Some(o),
        Err(e) => {
            println!("cargo:warning=jscpd not run ({e}); Rust duplication unchecked");
            None
        }
    }
}

/// The gate's correctness precondition, in a warning rather than a failure.
fn warn_if_format_is_a_lower_bound(root: &Path) {
    // A CLEAN RESULT IS ONLY MEANINGFUL ON A RUSTFMT-CLEAN TREE — the gate's correctness
    // precondition, not a style preference. jscpd tokenizes: two blocks that differ only
    // in line breaking tokenize differently enough to fall under `minTokens`. Measured in
    // the old tree 2026-08-06: 0 clones reported with 680 rustfmt hunks outstanding, and
    // 52 the moment `cargo fmt` ran. A WARNING, not a hard failure: `cargo build` on a
    // tree someone is mid-edit in must not refuse, and CI gates `cargo fmt --check` in
    // its own step anyway.
    let fmt_clean = Command::new("cargo")
        .current_dir(root)
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
}

/// The verdict on a jscpd run: 0 is silence, 7 is a build error, anything else names a gate
/// that did not run.
fn judge_jscpd(out: &std::process::Output) {
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
