#!/usr/bin/env bash
# PreToolUse guard for harness Bash calls: the THIRD layer, not the first.
#
# HONEST BYPASS NOTE, up front because it bounds every claim below: this hook fires only for
# Bash calls made through the Claude Code harness. A raw terminal, an ssh session opened by
# hand, a cron job, another tenant's agent on this box — none of them are scanned. So the
# ordering of defences is unchanged: `flock /var/run/sys-gpu.lock` remains the cross-tenant
# guard, the per-arm contention witness (/dev/kfd holders + mem_info_gtt_used) remains the
# post-hoc detector, and this hook only catches the one class those two cannot — the agent
# that simply forgot, before it costs anyone a discarded arm.
#
# Design rule, RESTATED 2026-08-31 after review measured the first form failing open: the
# checks stay text checks, but the UNIT OF MEANING IS THE SEGMENT, not the command line. The
# first version asked its questions of the whole string, and four reviewers independently
# broke it the same way — one `flock …sys-gpu.lock`, one `--no-default-features` or one
# `CARGO_TARGET_DIR=` ANYWHERE on the line disarmed the rule for every later segment, so
# `… --no-default-features && cargo test --workspace` was silently ALLOWED (a full unlocked
# GPU battery), and so was a bare second arm after a flocked first — the "lock PER ARM of an
# A/B" scar exactly. The fix is closer to a deletion than an addition: split on `;`, `&&`,
# `||`, `|` and newlines, then evaluate all three rules per segment. Ambition is still
# bounded: no variable expansion, no subshell semantics, no `$(…)`. Two concessions to the
# shell are deliberate and are the ONLY places this file models it, because this repo's own
# discipline is WRITTEN in them: the split is quote-aware (`-c 'a && b'` is one segment), and
# a small table of WRAPPERS whose arguments are themselves a command (`flock`, `env`, `sh -c`,
# `ssh`, `timeout`) is peeled so the wrapped command is what gets judged.
#
# False-block is still cheaper than false-allow HERE (the remedy is one prefix), but three
# things must never be blocked because blocking them would make the hook itself the defect:
# the canonical deviceless arm, plain git/ls/grep — and, added 2026-08-31 after it happened
# live, a command that merely MENTIONS a device verb in an argument. `pgrep -af "cargo test"`
# and `grep -rn 'cargo test' CLAUDE.md` are read-only text work; the first version refused
# both, i.e. it refused the fleet the right to read its own doctrine. Hence the rules fire on
# the segment's EXECUTABLE POSITION (first token after any env assignments), never on a
# substring of an argument.
#
# The repo's own tests/*.sh are not matched, and that is by CONSTRUCTION rather than by an
# exception list: no rule below names a shell script. Those scripts self-flock and their
# scheduling is the coordinator's job.
#
# Commands sent over ssh are scanned by the same rules — the defect they guard against is the
# same defect on the remote node, and the text is the same text.
#
# Red proof (P7): `crates/cli/tests/hook_guard.rs` is the STANDING gate — the whole row table
# lives there and runs on every deviceless `cargo test`, so an edit here reddens a build
# rather than a table someone has to remember to re-run. Its argument and its pre-fix red
# evidence are docs/measurement/gate-red-proofs.md §13.
set -euo pipefail

# The guard is a python text filter; without python3 it cannot decide anything. Fail OPEN,
# but LOUDLY: the harness treats a non-zero hook exit as a block and a missing interpreter is
# not a reason to stop work, while a guard that silently evaporates is worse than no guard —
# an operator would keep believing R1 was live. Precedent for the shape: build.rs's
# `cargo:warning=jscpd not run ({e}); Rust duplication unchecked`.
if ! command -v python3 >/dev/null 2>&1; then
    echo "bash-guard: python3 missing — guard NOT enforced (R1/R2/R3 all skipped)" >&2
    exit 0
fi

PROG=$(cat <<'PY'
import json
import posixpath
import re
import shlex
import sys

ALLOW = 0
BLOCK = 2

# Matched as a whole TOKEN, never as a substring: `/var/run/sys-gpu.lock.bak` is a different
# file and guards nothing (review row L5 had it counting as the lock).
LOCK = "/var/run/sys-gpu.lock"

# A token that is an environment assignment (FOO=bar) is never the executable being run.
# Load-bearing, not tidiness: the canonical deviceless arm is spelled
# `CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features`,
# whose assignment contains a cargo target dir, and a hook that blocks THAT is unconditionally
# red (gate-red-proofs.md §13, row 3b).
ENV_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# A build ARTEFACT lives under <target-root>/<profile>/, and it is the artefact — not the
# directory — that runs on the GPU. Requiring the profile segment AFTER the root is what makes
# `ls -ld /var/cache/rivoli/target/rivoli` legal.
#
# The roots are NOT anchored at `/` any more. Every entry in the first version began with a
# slash, so `target/release/rivoli bench` — the commonest spelling of a built binary, since
# bash executes any path containing a slash directly — was invisible while `./target/…` and
# the absolute form both blocked. `cargo-target/` is here because pam_env makes
# /var/cache/users/<user>/cargo-target THIS BOX'S DEFAULT, so it is where a stray binary is
# most likely to be found (§12).
TARGET_ROOT = re.compile(r"(?:^|/)(?:target|cargo-target)/|^\$\{?CARGO_TARGET_DIR\}?/")
PROFILE_SEGMENTS = ("/debug/", "/release/")

# cargo's BUILT-IN one-letter aliases are real verbs: `cargo --list` gives b=build, c=check,
# d=doc, r=run, t=test. `cargo r --release -- decode` is a full device decode and the first
# version allowed it, silently, because every pattern spelled the verb out.
DEVICE_VERBS = frozenset({"run", "r", "bench", "test", "t"})
# Anything that COMPILES, which is what the shared-target-dir defect needs. `fix` rewrites
# source as well as compiling; `install` and `rustc` compile; `fmt`, `metadata`, `tree` and
# `clean` do not and stay out (a bare `cargo clean` against the box-wide default dir is a
# known uncovered class — §13 gap 4).
BUILD_VERBS = DEVICE_VERBS | frozenset(
    {"build", "b", "check", "c", "doc", "d", "clippy", "fix", "install", "rustc"}
)

# Verbs that discard or rewrite another agent's uncommitted work in this shared tree. `add`
# and `commit` are deliberately absent: workers COMMIT as soon as a change verifies, which is
# the remedy the recorded scar prescribes, not the defect.
GIT_MUTATORS = frozenset(
    {"stash", "checkout", "restore", "switch", "clean", "reset", "merge", "rebase"}
)
# git's own global options, the ones that take a separate value, so the SUBCOMMAND is found by
# position rather than by a substring search. Position matters both ways: `git -C wt stash` is
# a stash, and `git commit -m 'fix the checkout path'` is not a checkout.
GIT_VALUED = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"})

# Wrappers whose ARGUMENTS are a command. Each entry: (options taking a separate value,
# `-c` introduces a command string, how many positionals to drop, a `-c` is REQUIRED).
# This table is the one place this file models the shell, and it exists because the repo's
# discipline is written in two of these: `flock /var/run/sys-gpu.lock -c '…'` and
# `env -u CARGO_TARGET_DIR cargo …`.
WRAPPERS = {
    "flock": (frozenset({"-w", "-E", "--timeout", "--conflict-exit-code"}), True, 1, False),
    "env": (frozenset({"-u", "--unset", "-C", "--chdir"}), False, 0, False),
    "sh": (frozenset({"-o"}), True, 0, True),
    "bash": (frozenset({"-o"}), True, 0, True),
    "ssh": (
        frozenset(
            {"-o", "-i", "-p", "-l", "-F", "-J", "-b", "-c", "-D", "-E", "-e", "-I", "-L",
             "-m", "-O", "-Q", "-R", "-S", "-W", "-w"}
        ),
        False,
        1,
        False,
    ),
    "timeout": (frozenset({"-s", "--signal", "-k", "--kill-after"}), False, 1, False),
    "sudo": (frozenset({"-u", "-g", "-p"}), False, 0, False),
    "nohup": (frozenset(), False, 0, False),
    "time": (frozenset({"-o", "-f", "--output", "--format"}), False, 0, False),
}


def split_segments(cmd):
    """`cmd` cut at `;`, `&&`, `||`, `|` and newlines — OUTSIDE quotes.

    Quote-awareness is the whole reason `flock … -c 'a && b'` stays one segment, and a lone
    `&` is NOT a separator so that `cargo build 2>&1` survives intact."""
    segs, buf, quote, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if quote is not None:
            buf.append(c)
            if c == quote:
                quote = None
            i += 1
        elif c in "'\"":
            quote = c
            buf.append(c)
            i += 1
        elif c == "\\" and i + 1 < len(cmd):
            buf.append(cmd[i : i + 2])
            i += 2
        elif cmd[i : i + 2] in ("&&", "||"):
            segs.append("".join(buf))
            buf = []
            i += 2
        elif c in ";|\n":
            segs.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(c)
            i += 1
    segs.append("".join(buf))
    return [s for s in (s.strip() for s in segs) if s]


def tokenize(seg):
    """Tokens of one segment. Falls back to a whitespace split on unbalanced quotes — a
    guard that raised there would fail CLOSED on a typo, which is a work stoppage."""
    try:
        return shlex.split(seg)
    except ValueError:
        return seg.split()


def env_prefix(argv):
    """(assignments, rest) — the run of `FOO=bar` tokens a segment opens with."""
    i = 0
    while i < len(argv) and ENV_ASSIGN.match(argv[i]):
        i += 1
    return argv[:i], argv[i:]


def wrapped(base, rest):
    """What a wrapper wraps, as ("str", text) to re-split or ("tokens", argv) — or None."""
    valued, dash_c, drop, needs_c = WRAPPERS[base]
    i, dropped = 0, 0
    while i < len(rest):
        t = rest[i]
        if dash_c and t in ("-c", "--command"):
            return ("str", rest[i + 1]) if i + 1 < len(rest) else None
        if t in valued:
            i += 2
        elif t == "--":
            i += 1
        elif t.startswith("-") and len(t) > 1:
            i += 1
        elif dropped < drop:
            dropped += 1
            i += 1
        else:
            break
    if needs_c:
        # `sh script.sh` runs a FILE this guard has not read; there is no command text here.
        return None
    tail = rest[i:]
    if not tail:
        return None
    return ("str", tail[0]) if len(tail) == 1 else ("tokens", tail)


def is_build_artefact(tok):
    """A path to something cargo BUILT: a target root with a profile segment after it."""
    m = TARGET_ROOT.search(tok)
    if m is None:
        return False
    tail = tok[m.end() - 1 :]  # keep the root's trailing slash so `/debug/` can match
    return any(seg in tail for seg in PROFILE_SEGMENTS)


def cargo_verb(argv):
    """cargo's subcommand, seeing through a `+toolchain` selector, cargo's own flags, and one
    third-party runner token (`cargo nextest run`, which hid `run` from the first version)."""
    i = 1
    while i < len(argv) and (argv[i].startswith("+") or argv[i].startswith("-")):
        i += 1
    if i >= len(argv):
        return None
    if argv[i] == "nextest" and i + 1 < len(argv):
        return argv[i + 1]
    return argv[i]


def rocm_feature(argv):
    """`--features …rocm…` / `-F …rocm…` in THIS segment, as the fragment to quote.

    Substring on the VALUE, deliberately: a feature named `no-rocm` would match and be
    blocked. False-block, one-word remedy, and the alternative is a feature-name registry
    this hook has no way to read."""
    for i, t in enumerate(argv):
        if t in ("--features", "-F") and i + 1 < len(argv) and "rocm" in argv[i + 1]:
            return t + " " + argv[i + 1]
        if re.match(r"^(?:--features=|-F=)\S*rocm", t):
            return t
    return None


def git_subcommand(argv):
    """The verb, by POSITION: global options and their values skipped, the first bare token
    wins. A message that happens to contain `checkout` is not a checkout."""
    i = 1
    while i < len(argv):
        t = argv[i]
        if t in GIT_VALUED:
            i += 2
        elif t.startswith("-"):
            i += 1
        else:
            return t
    return None


def r1(base, exe, argv, ctx):
    """Reaches the device without holding the GPU lock."""
    fragment = None
    if base == "cargo":
        verb = cargo_verb(argv)
        if verb in DEVICE_VERBS and "--no-default-features" not in argv:
            fragment = "cargo " + verb + " (no --no-default-features: this is a GPU arm)"
        fragment = rocm_feature(argv) or fragment
    else:
        path = exe
        if not path.startswith("/") and ctx["cwd"]:
            path = posixpath.normpath(posixpath.join(ctx["cwd"], path))
        if is_build_artefact(path):
            fragment = exe
    if fragment is None or ctx["locked"]:
        return None
    return (
        "BLOCKED (R1, GPU): this command reaches the device without holding the GPU lock.\n"
        "  offending fragment: " + fragment + "\n"
        '  CLAUDE.md: "**The GPU is sole-tenant.** Wrap every GPU command in '
        "`flock /var/run/sys-gpu.lock -c`, build OUTSIDE the lock.\"\n"
        '  CLAUDE.md: "**The build IS the rocm build** … a bare `cargo test --workspace` is a '
        'GPU arm — flock it and serialize it like any device suite."\n'
        "  remedy: flock /var/run/sys-gpu.lock -c '<the command>'  — in the SAME segment as\n"
        "  the command it guards (a lock held by an earlier `&&` arm guards only that arm),\n"
        "  or run the deviceless arm instead: cargo test --workspace --no-default-features."
    )


def r2(base, exe, argv, ctx):
    """A cargo build verb with no explicit target dir — the §7e-bis/§12 defect."""
    if base != "cargo":
        return None
    verb = cargo_verb(argv)
    if verb not in BUILD_VERBS:
        return None
    if ctx["tdir"] or any(t == "--target-dir" or t.startswith("--target-dir=") for t in argv):
        return None
    return (
        "BLOCKED (R2, target dir): a cargo build verb with no explicit CARGO_TARGET_DIR.\n"
        "  offending fragment: cargo " + verb + "\n"
        "  why: /etc/security/pam_env.conf:6 sets a box-wide CARGO_TARGET_DIR for every login,\n"
        "  and an environment variable OUTRANKS build.target-dir in cargo precedence — so every\n"
        "  checkout on this box shares ONE target dir and the per-worktree .cargo/config.toml is\n"
        "  INERT. That is not a hypothetical: cargo then replays a sibling checkout's cached\n"
        "  build-script output, so the duplication gate scans the OTHER tree and the build carries\n"
        "  on green (gate-red-proofs.md §7e-bis and §12).\n"
        "  THE LOCK IS NOT AN EXEMPTION (changed 2026-08-31): device scheduling and build\n"
        "  isolation are independent, so a flocked cargo test needs the prefix too — the first\n"
        "  version exempted lock-prefixed commands and thereby exempted the single most-run\n"
        "  build on this box from the rule that exists to stop it.\n"
        "  remedy: prefix CARGO_TARGET_DIR=/var/cache/rivoli/target/<checkout-name> (the primary\n"
        "  checkout is `rivoli`; tests/new-worktree.sh prints yours), or `env -u CARGO_TARGET_DIR`,\n"
        "  or cargo's own --target-dir <path>, which outranks both."
    )


def r3(base, exe, argv, ctx):
    """A mutating git verb on a tree four agents are writing to."""
    if base != "git":
        return None
    verb = git_subcommand(argv)
    if verb not in GIT_MUTATORS:
        return None
    return (
        "BLOCKED (R3, shared tree): `git " + verb + "` in a tree other agents are writing to.\n"
        "  recorded scar: a foreign stash came back as CONFLICT MARKERS inside another agent's\n"
        "  files — one agent's stash/checkout silently reverts every other agent's uncommitted\n"
        "  work, and the damage is discovered by whoever compiles next. `reset --hard` and\n"
        "  `clean -fdx` are the same loss and are not even recoverable from the stash list.\n"
        "  `git add` and `git commit` are ALLOWED: committing as soon as a change verifies is\n"
        "  the remedy, not the defect. A read-only spelling (`git stash list`) is refused too —\n"
        "  that false-block costs a rephrase, and this rule prefers it.\n"
        "  remedy: leave your files alone (each agent owns an exclusive file list), or copy them\n"
        "  aside with cp into the scratch directory."
    )


def scan_tokens(argv, ctx, depth):
    """One segment, already tokenized: peel wrappers, then judge the executable."""
    if depth > 4:
        return None
    assigns, argv = env_prefix(argv)
    ctx = dict(ctx)
    for a in assigns:
        m = ENV_ASSIGN.match(a)
        # A non-empty VALUE, in THIS segment's own prefix. `CARGO_TARGET_DIR=` with nothing
        # after it selects no directory at all, and a bash env prefix binds only the one
        # command it prefixes — both were false-allows in the first version.
        if m and m.group(1) == "CARGO_TARGET_DIR" and m.group(2).strip():
            ctx["tdir"] = True
    if not argv:
        return None
    exe = argv[0]
    base = exe.rsplit("/", 1)[-1]
    if base in WRAPPERS:
        if base == "flock" and LOCK in argv[1:]:
            ctx["locked"] = True
        if base == "env":
            for a, b in zip(argv[1:], argv[2:]):
                if a in ("-u", "--unset") and b == "CARGO_TARGET_DIR":
                    ctx["tdir"] = True
        inner = wrapped(base, argv[1:])
        if inner is None:
            return None
        kind, payload = inner
        if kind == "str":
            return scan_command(payload, ctx, depth + 1)
        return scan_tokens(payload, ctx, depth + 1)
    for rule in (r3, r1, r2):
        reason = rule(base, exe, argv, ctx)
        if reason is not None:
            return reason
    return None


def scan_command(cmd, ctx, depth=0):
    """Every segment of `cmd`, each judged on its own, with `cd` threaded between them."""
    for seg in split_segments(cmd):
        argv = tokenize(seg)
        reason = scan_tokens(argv, ctx, depth)
        if reason is not None:
            return reason
        # `cd target/debug && ./rivoli decode` is the same invocation as
        # `target/debug/rivoli decode`, and only the join makes the second segment's
        # executable recognisable as an artefact.
        _, rest = env_prefix(argv)
        if len(rest) > 1 and rest[0].rsplit("/", 1)[-1] == "cd":
            ctx = dict(ctx, cwd=posixpath.join(ctx["cwd"], rest[1]))
    return None


def main():
    try:
        event = json.load(sys.stdin)
    except Exception:
        # A guard that cannot read its own input must not become a work stoppage: the two
        # layers below it (flock, the witness) are still in place.
        return ALLOW
    if event.get("tool_name") != "Bash":
        return ALLOW
    cmd = (event.get("tool_input") or {}).get("command") or ""
    reason = scan_command(cmd, {"locked": False, "tdir": False, "cwd": ""})
    if reason is None:
        return ALLOW
    sys.stderr.write(reason + "\n")
    return BLOCK


sys.exit(main())
PY
)

# `python3 -c` rather than a heredoc on stdin: stdin belongs to the hook payload.
exec python3 -c "$PROG"
