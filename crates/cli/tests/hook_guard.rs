//! The harness bash-guard's row table, as a STANDING gate.
//!
//! `.claude/hooks/bash-guard.sh` is a PreToolUse hook on the `Bash` matcher: it reads the
//! tool payload as JSON on stdin and answers with an exit code — **0 allows, 2 blocks and
//! hands the reason back on stderr**. Three rules: **R1** a command that reaches the device
//! without `flock /var/run/sys-gpu.lock` in the SAME segment, **R2** a cargo build verb with
//! no explicit `CARGO_TARGET_DIR` (the §7e-bis/§12 defect, whose mechanism is this box's
//! *default*), **R3** a mutating git verb in a tree other agents are writing to.
//!
//! **Why this is a cargo test and not a table in a doc.** It landed as a hand-run table whose
//! drivers lived in a session scratchpad, with "re-run after any edit to the hook" as its only
//! trigger — prose, enforced by nothing. That is the failure mode CLAUDE.md names outright: a
//! check whose examined-count can silently reach zero is not a check. Every row here is a pure
//! deviceless text decision, so the whole table runs on `cargo test --no-default-features` and
//! an edit to the matcher reddens a build instead of a memory.
//!
//! **Non-vacuity is measured, not asserted.** Driven against the PRE-FIX matcher — the
//! whole-string form this file's rows were written against, recovered byte-for-byte from a
//! reviewer's transcript because it was never committed — the table scored **36 of 78, 42
//! rows red** (2026-08-31): **33 false-ALLOWs** (`cargo r`/`b`/`c`/`d`/`t`, the `&&`-chained
//! second arm, a flock held by arm 1 only, a prose mention of the lock, an empty
//! `CARGO_TARGET_DIR=`, every relative artefact path, seven of R3's eight verbs, the `.bak`
//! lock path) and **9 false-BLOCKs** (`ls`/`du`/`rm`/`cat`/`find` under a build dir,
//! `--target-dir`, and a quoted `cargo test` in a `pgrep`/`grep` argument). The full pre/post
//! table is `docs/measurement/gate-red-proofs.md` §13.
//!
//! **Capture discipline, inherited from that file's standing rule.** Each payload is written
//! to a file and the FILE is redirected onto the hook's stdin; the exit code is read from the
//! hook process itself. No shell pipeline appears anywhere — §7e-bis's false green *was* a
//! pipeline's exit code. A `want: 2` row additionally asserts the stderr names a rule, so a
//! python traceback or a missing interpreter cannot be counted as a block.

#![allow(clippy::expect_used)] // meta-gate: a broken harness must panic loudly, not degrade

mod common;

// `Stdio` and `Command` only: the payload reaches the hook through a FILE, not a
// writer, so nothing here needs `io::Write`.
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

/// One row: `(id, the payload's `command`, the exit code the hook must answer with)`. The
/// argument for each row is the comment above it — it costs no code line and cannot drift
/// away from the row it explains.
type Row = (&'static str, &'static str, i32);

/// The population this gate landed with, and the floors it asserts. Named constants rather
/// than literals in the message because the two must move together: a row added or removed
/// changes the population, and a floor that no longer sits below it is a floor that cannot
/// fire. Measured 2026-08-31: 28 ALLOW rows and 48 BLOCK rows in `ROWS`.
const ALLOW_POP: usize = 28;
const BLOCK_POP: usize = 48;
const ALLOW_FLOOR: usize = 26;
const BLOCK_FLOOR: usize = 45;

/// The table. Rows 1–7 and A–G are the gate's original population; every row tagged
/// `PRE-FIX` is one a reviewer broke the first matcher with, kept here as the negative half.
const ROWS: &[Row] = &[
    // the canonical device battery, bare
    ("1", r#"cargo test --workspace -- --test-threads=1"#, 2),
    // row 2 REPAYLOADED: the lock is no longer an R2 exemption, so it carries the prefix
    (
        "2",
        r#"flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace -- --test-threads=1'"#,
        0,
    ),
    // the supersession itself: flocked but no target dir -> R2
    (
        "2b",
        r#"flock /var/run/sys-gpu.lock -c 'cargo test --workspace -- --test-threads=1'"#,
        2,
    ),
    // deviceless, no target dir -> R2
    ("3", r#"cargo test --workspace --no-default-features"#, 2),
    // THE canonical deviceless arm; a hook that blocks this is unconditionally red
    (
        "3b",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features"#,
        0,
    ),
    // the other remedy CLAUDE.md names
    (
        "3c",
        r#"env -u CARGO_TARGET_DIR cargo test --workspace --no-default-features"#,
        0,
    ),
    // R1 wins even with the prefix, and the fragment is the feature
    (
        "4",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features --features rocm"#,
        2,
    ),
    // R2 on a plain check
    ("5", r#"cargo check"#, 2),
    // plain git
    ("6a", r#"git status"#, 0),
    // fmt is not a build verb
    ("6b", r#"cargo fmt --check"#, 0),
    // R3
    ("7", r#"git stash"#, 2),
    // a target-dir BINARY with the prefix present, so only R1 can speak
    (
        "C",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli /var/cache/rivoli/target/rivoli/debug/rivoli decode"#,
        2,
    ),
    // C behind the lock
    (
        "D",
        r#"flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli /var/cache/rivoli/target/rivoli/debug/rivoli decode'"#,
        0,
    ),
    // the repo's own self-flocking scripts
    ("E", r#"tests/ppl-gates.sh profile"#, 0),
    // a target DIRECTORY: no profile segment, no device
    ("F", r#"ls -ld /var/cache/rivoli/target/rivoli"#, 0),
    // a binary under this box's pam_env DEFAULT target root
    (
        "G",
        r#"/var/cache/users/rhansen/cargo-target/debug/rivoli decode"#,
        2,
    ),
    // PRE-FIX 0: `r` is cargo's built-in alias for run
    (
        "Z1",
        r#"cargo r --release -- decode --model glm --prompt 'hi' --max-mem 100"#,
        2,
    ),
    // alias b = build -> R2
    ("X1", r#"cargo b --release"#, 2),
    // alias c = check -> R2
    ("X2", r#"cargo c --no-default-features"#, 2),
    // alias d = doc -> R2
    ("X3", r#"cargo d --no-deps"#, 2),
    // alias t = test -> R1
    ("X4", r#"cargo t --workspace"#, 2),
    // a third-party runner token must not hide `run`
    ("A6", r#"cargo nextest run --workspace"#, 2),
    // fix compiles AND rewrites source -> R2
    ("T8", r#"cargo fix --allow-dirty"#, 2),
    // cargo's short feature selector
    (
        "F1",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build -F rocm"#,
        2,
    ),
    // PRE-FIX 0: deviceless arm then the real one -- the most natural two-arm line here
    (
        "N2",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features && cargo test --workspace -- --test-threads=1"#,
        2,
    ),
    // N2's semicolon form
    (
        "N4",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features ; cargo test --workspace"#,
        2,
    ),
    // PRE-FIX 0: arm 2 holds no lock -- the `lock PER ARM of an A/B` scar
    (
        "L1",
        r#"flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace' && cargo test --workspace"#,
        2,
    ),
    // PRE-FIX 0: a PROSE mention of the lock armed the exemption
    (
        "L2",
        r#"echo 'run under flock /var/run/sys-gpu.lock next time' && cargo run --release -- decode"#,
        2,
    ),
    // an emptied lock arm followed by a bare bench
    (
        "L3",
        r#"flock /var/run/sys-gpu.lock -c 'true'; cargo bench"#,
        2,
    ),
    // the same across a newline
    (
        "X11",
        "flock /var/run/sys-gpu.lock -c 'true'\ncargo test --workspace",
        2,
    ),
    // PRE-FIX 0: a bash env prefix binds ONE command; arm 2 lands in the pam_env dir
    (
        "T3",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release && cargo test --workspace --no-default-features -- --test-threads=1"#,
        2,
    ),
    // the same shape through the env wrapper
    (
        "X12",
        r#"env -u CARGO_TARGET_DIR cargo check --no-default-features && cargo build"#,
        2,
    ),
    // PRE-FIX 0: an EMPTY value selects no directory
    (
        "T1",
        r#"CARGO_TARGET_DIR= cargo test --workspace --no-default-features"#,
        2,
    ),
    // PRE-FIX 0: a mention in an echo armed the exemption
    ("T7", r#"echo 'CARGO_TARGET_DIR=' && cargo check"#, 2),
    // PRE-FIX 2 (false-block): --target-dir is cargo's own flag and OUTRANKS the variable
    (
        "T2",
        r#"cargo test --workspace --no-default-features --target-dir /var/cache/rivoli/target/rivoli"#,
        0,
    ),
    // PRE-FIX 0: every root began with `/`, so the relative path was invisible
    (
        "Z3",
        r#"cd /home/rhansen/workspace/constructive.dev/rivoli && target/release/rivoli bench --model glm"#,
        2,
    ),
    // PRE-FIX 0: bare relative artefact
    ("P1", r#"target/debug/rivoli decode --prompt hi"#, 2),
    // the control that blocked pre-fix too
    ("P2", r#"./target/release/rivoli decode"#, 2),
    // PRE-FIX 0: the cd carries the target root
    ("P3", r#"cd target/debug && ./rivoli decode --prompt hi"#, 2),
    // PRE-FIX 0: the variable spelling of the same binary
    ("P4", r#"$CARGO_TARGET_DIR/debug/rivoli decode"#, 2),
    // PRE-FIX 2 (false-block): a trailing slash was the whole difference from row F
    ("F2", r#"ls -l /var/cache/rivoli/target/rivoli/debug/"#, 0),
    // PRE-FIX 2 (false-block)
    ("F3", r#"du -sh /var/cache/rivoli/target/rivoli/debug/"#, 0),
    // PRE-FIX 2 (false-block): reclaiming disk is routine here
    (
        "F4",
        r#"rm -rf /var/cache/rivoli/target/rivoli/debug/incremental"#,
        0,
    ),
    // PRE-FIX 2 (false-block): reading a build-script log
    (
        "F5",
        r#"cat /var/cache/rivoli/target/rivoli/debug/build/rivoli-cli-abc/output"#,
        0,
    ),
    // PRE-FIX 2 (false-block)
    (
        "F6",
        r#"find /var/cache/rivoli/target/rivoli/debug/ -name '*.d'"#,
        0,
    ),
    // the row 13 gap 4 had BACKWARDS: it exits 0, it was never a false-block
    ("X5", r#"ls -l target/debug/rivoli"#, 0),
    // OBSERVED LIVE 2026-08-31: the pre-fix hook refused the coordinator's read-only probe
    (
        "Q1",
        r#"ssh rh-anine 'pgrep -af "cargo test|test-threads"'"#,
        0,
    ),
    // PRE-FIX 2 (false-block): the fleet reading its own doctrine
    ("Q2", r#"grep -rn 'cargo test' docs/ CLAUDE.md"#, 0),
    // the pair to Q1: a REAL remote device arm still blocks
    ("Q3", r#"ssh rh-anine 'cargo test --workspace'"#, 2),
    // PRE-FIX 0
    ("G2", r#"git checkout main"#, 2),
    // PRE-FIX 0
    ("G3", r#"git checkout -- crates/engine/src/lib.rs"#, 2),
    // PRE-FIX 0: not even recoverable
    ("G4", r#"git reset --hard origin/main"#, 2),
    // PRE-FIX 0: would delete the untracked .claude/ tree
    ("G5", r#"git clean -fdx"#, 2),
    // PRE-FIX 0
    ("G6", r#"git restore ."#, 2),
    // PRE-FIX 0
    ("G8", r#"git switch main"#, 2),
    // PRE-FIX 0
    ("G9", r#"git rebase origin/main"#, 2),
    // PRE-FIX 0
    ("G10", r#"git merge --no-ff wave/m12"#, 2),
    // ALLOWED on purpose: workers commit as soon as a change verifies
    ("G7", r#"git add -A && git commit -m wip"#, 0),
    // the verb is found by POSITION: a message is not a checkout
    ("G11", r#"git commit -m 'fix the checkout path'"#, 0),
    // R3 sees the subcommand, not the whole line
    ("G12", r#"git stash pop"#, 2),
    // `git -C <path>` is still a stash
    ("G13", r#"git -C .claude/worktrees/wave-m12 stash"#, 2),
    // the wrong lock guards nothing
    (
        "L4",
        r#"flock /var/run/other.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace'"#,
        2,
    ),
    // PRE-FIX 0: a bare substring counted `.bak` as the lock
    (
        "L5",
        r#"flock /var/run/sys-gpu.lock.bak -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace'"#,
        2,
    ),
    // the rocm clippy arm: clippy compiles, it does not run the device
    (
        "CM1",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo clippy --workspace --all-targets"#,
        0,
    ),
    // the stub clippy arm
    (
        "CM2",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo clippy --workspace --all-targets --no-default-features"#,
        0,
    ),
    // the benchmark build
    (
        "CM3",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release"#,
        0,
    ),
    // KNOWN false-block, kept as a standing fixture: a heredoc's prose line is scanned as a segment. It reddened on this fix's own author while writing section 13
    (
        "H1",
        "cat > /var/tmp/note.txt <<'EOF'\ntarget/debug/rivoli decode was the false-allow\nEOF",
        2,
    ),
    // H1's pair: it is not `any heredoc blocks`, it is a prose line that parses as a command
    (
        "H2",
        "cat > /var/tmp/note.txt <<'EOF'\nthe guard reads the text, nothing more\nEOF",
        0,
    ),
    // the wrapper table: timeout's duration positional is skipped and the wrapped verb is judged
    ("W1", r#"timeout 300 cargo test --workspace"#, 2),
    // sh -c's command string is re-entered
    ("W2", r#"sh -c 'cargo test --workspace'"#, 2),
    // ssh's valued options are skipped before the host, and the remote deviceless arm is allowed
    (
        "W3",
        r#"ssh -o BatchMode=yes rh-anine 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features'"#,
        0,
    ),
    // R3 by POSITION: a search FOR the word stash is not a stash
    ("R4", r#"git log --grep=stash --oneline"#, 0),
    // KNOWN false-block, accepted: a read-only spelling costs a rephrase and this rule prefers it
    ("R5", r#"git stash list"#, 2),
    // a `|` splits the line without breaking either half's own verdict
    (
        "P5",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features 2>&1 | tee /var/tmp/log"#,
        0,
    ),
    // a lone `&` is NOT a separator, so `2>&1` survives tokenization
    (
        "P6",
        r#"CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release 2>&1"#,
        0,
    ),
    // an unbalanced quote falls back to a whitespace split rather than failing CLOSED on a typo
    ("N1", r#"echo 'oops"#, 0),
];

/// The hook script, located from this crate's manifest so the test does not depend on the
/// process's working directory.
///
/// **Fails LOUDLY rather than skipping.** A guard test that quietly passes when the guard is
/// absent is worse than no test: the harness reports "ok" for the exact tree state — hook
/// deleted, hook renamed, `.claude/` not checked out — the gate exists to notice.
fn hook() -> std::path::PathBuf {
    let p = common::repo_root().join(".claude/hooks/bash-guard.sh");
    assert!(
        p.is_file(),
        "the guard this gate scores is missing: {}\nA hook that is not on disk is not enforcing \
         anything, and this test must not pass without it.",
        p.display()
    );
    p
}

/// Drive one raw stdin body into the hook and hand back `(exit code, stderr)`.
///
/// The payload goes through a FILE, per §13's capture discipline. `wait_with_output` reads the
/// child's own status, so the code returned is the hook's and not some wrapper's.
fn drive(body: &str) -> (i32, String) {
    // A distinct scratch dir per call: `common::scratch` REMOVES the directory it
    // returns, so two of this file's tests running in parallel on one tag would delete
    // each other's payload mid-flight — a flake that would read as a matcher change.
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = common::scratch(&format!("hook-guard-{}", N.fetch_add(1, Ordering::Relaxed)));
    let path = dir.join("payload.json");
    std::fs::write(&path, body).expect("write the payload file");
    let file = std::fs::File::open(&path).expect("reopen the payload file");
    let child = Command::new("bash")
        .arg(hook())
        .stdin(Stdio::from(file))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the guard");
    let out = child.wait_with_output().expect("wait for the guard");
    common::clean(&dir);
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// A `Bash` payload carrying `cmd`, serialized the way the harness serializes it.
fn bash_payload(cmd: &str) -> String {
    serde_json::json!({ "tool_name": "Bash", "tool_input": { "command": cmd } }).to_string()
}

/// One row's verdict: `Ok(the rule tag, if it blocked)` or `Err(the reader-facing mismatch)`.
///
/// Split out of the test body so the loop below is a `for` over a `match` rather than four
/// levels of nesting, and so the two assertions that make a green MEAN something sit next to
/// the exit code they qualify: an ALLOW must be silent (a hook that printed a refusal and then
/// exited 0 would score green while telling the operator the opposite), and a BLOCK must name
/// a rule (a python traceback at exit 1, or the loud python3-missing branch, must not be able
/// to satisfy a `want: 2` row by accident).
fn score(id: &str, cmd: &str, want: i32) -> Result<Option<&'static str>, String> {
    let (got, err) = drive(&bash_payload(cmd));
    if got != want {
        return Err(format!(
            "{id}: want {want} got {got}  |  {cmd}\n      {err}"
        ));
    }
    if want == 0 {
        assert!(
            err.is_empty(),
            "row {id} allowed but wrote to stderr:\n{err}"
        );
        return Ok(None);
    }
    assert!(
        err.contains("BLOCKED (R"),
        "row {id} exited {got} without naming a rule — that is not a rule firing:\n{err}"
    );
    Ok(["R1", "R2", "R3"]
        .into_iter()
        .find(|tag| err.contains(&format!("BLOCKED ({tag}"))))
}

#[test]
fn every_row_gets_the_exit_code_it_expects() {
    let mut wrong = Vec::new();
    let (mut allows, mut blocks) = (0usize, 0usize);
    let mut rules = std::collections::BTreeSet::new();
    for (id, cmd, want) in ROWS {
        match score(id, cmd, *want) {
            Err(line) => wrong.push(line),
            Ok(None) => allows += 1,
            Ok(Some(tag)) => {
                blocks += 1;
                rules.insert(tag);
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "\n\n{} of {} guard rows disagree with the recorded table:\n    {}\n\nThe table is the \
         gate: fix the hook, or re-measure and move the row WITH its argument.",
        wrong.len(),
        ROWS.len(),
        wrong.join("\n    ")
    );
    // Anti-vacuity on the table itself. Each of these has been able to reach zero in some
    // version of this gate's history: a table of allows only would pass with R1 deleted, a
    // table of blocks only would pass with the hook replaced by `exit 2`, and a rule with no
    // row is a rule nothing scores.
    assert!(
        allows >= ALLOW_FLOOR,
        "only {allows} ALLOW rows scored ({ALLOW_POP} at the population this gate landed \
         with) — the table lost its greens, and the greens are what protect daily work"
    );
    assert!(
        blocks >= BLOCK_FLOOR,
        "only {blocks} BLOCK rows scored ({BLOCK_POP} at the population this gate landed \
         with) — the table lost its reds"
    );
    assert_eq!(
        rules.len(),
        3,
        "only these rules fired across the whole table: {rules:?} — a rule with no row is a \
         rule nothing scores"
    );
}

#[test]
fn the_matcher_scope_and_the_unparseable_payload_both_allow() {
    // Row A: the same device command under a DIFFERENT tool name. This is what separates
    // "the rules decided" from "the hook fails closed on everything".
    let read = serde_json::json!({
        "tool_name": "Read",
        "tool_input": { "command": "cargo test --workspace" }
    })
    .to_string();
    let (got, err) = drive(&read);
    assert_eq!(
        (got, err.as_str()),
        (0, ""),
        "row A: tool_name Read must be out of scope"
    );

    // Row B: stdin that is not JSON. The guard fails OPEN by design — the two layers below it
    // (flock, the per-arm contention witness) are still in place, and a guard that cannot read
    // its own input must not become a work stoppage.
    let (got, err) = drive("{not json at all");
    assert_eq!(
        (got, err.as_str()),
        (0, ""),
        "row B: an unparseable payload must fail OPEN"
    );
}

#[test]
fn the_guard_is_wired_up_in_the_versioned_settings() {
    // The rows above score the SCRIPT. This scores the wiring, which is the half a direct
    // drive cannot reach: a correct script that no settings.json invokes guards nothing.
    let body = std::fs::read_to_string(common::repo_root().join(".claude/settings.json"))
        .expect("read .claude/settings.json");
    let json: serde_json::Value = serde_json::from_str(&body).expect("settings.json is JSON");
    let pre = json
        .pointer("/hooks/PreToolUse/0")
        .expect("a PreToolUse hook entry");
    assert_eq!(
        pre.pointer("/matcher").and_then(|m| m.as_str()),
        Some("Bash")
    );
    let cmd = pre
        .pointer("/hooks/0/command")
        .and_then(|c| c.as_str())
        .expect("the hook's command string");
    assert!(
        cmd.contains(".claude/hooks/bash-guard.sh"),
        "the PreToolUse Bash hook does not invoke the guard this gate scores: {cmd}"
    );
}
