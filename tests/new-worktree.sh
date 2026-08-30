#!/usr/bin/env bash
# The ONLY way a worktree gets created in this repo. Never a raw `git worktree add`.
#
#   tests/new-worktree.sh <name> <branch> <base-sha>
#
# WHY A SCRIPT for three git commands: each of the three things it does is something that
# has already gone wrong here, and none of them is visible when it goes wrong.
#
#   1. THE BASE SHA IS REQUIRED. `git worktree add -b <branch> <path>` defaults the base to
#      the CURRENT checkout's HEAD, which for an agent launched from the primary tree is
#      the wrong commit whenever the work belongs on a sibling branch — a recorded lesson
#      ("isolation:worktree bases off the PRIMARY repo HEAD, not the active secondary
#      worktree — silently wrong branch"). A defaulted base is how that happens quietly, so
#      there is no default: three arguments or a refusal.
#   2. ONE TARGET DIR PER WORKTREE, and it is a CORRECTNESS rule. Cargo reuses the
#      build-script BINARY it finds in the target dir; `crates/cli/build.rs` bakes
#      `env!("CARGO_MANIFEST_DIR")` at ITS compile time, so a sibling worktree's binary
#      scans a `crates` directory this checkout does not have — the duplication gate did
#      not run at all for that invocation and a commit landed on the false green
#      (docs/measurement/gate-red-proofs.md 7e-bis). Hence the claimed-target-dir refusal:
#      two checkouts naming one target dir is that defect, spelled out in advance.
#   3. `/.cargo/` MUST BE IGNORED BY THE COMMON GITDIR. The per-worktree gitdir's
#      `info/exclude` is NOT read; only `git rev-parse --git-common-dir`'s is. The config
#      file is local machine state, not a repo change, so a worktree where it shows up as
#      untracked is a worktree that will eventually have it committed. This is VERIFIED
#      with `git check-ignore` rather than assumed, because the assumption is exactly the
#      kind that holds until someone re-clones.
#
# Every refusal exits 66 (this tree's loud-failure convention — see CLAUDE.md and the GPU
# lock scripts), and every refusal prints WHICH one it is: exit 66 alone does not tell the
# refusals apart, so read the message, not the code.
#
# Red proof (P7): docs/measurement/gate-red-proofs.md 12 runs the refusals plus the
# check-ignore one, and the shared-vs-per-worktree target dir pair that motivates the whole
# script.
set -euo pipefail

usage() {
    echo "usage: $0 <name> <branch> <base-sha>" >&2
    echo "  all three are REQUIRED: a defaulted base sha silently bases the worktree" >&2
    echo "  off the current checkout's HEAD instead of the commit you meant" >&2
    exit 66
}

[ $# -eq 3 ] || usage
name=$1
branch=$2
base=$3

# (a2) The name has to be one ordinary path component. It is interpolated into a filesystem
# path AND into a target-dir claim, so an empty name, a slash, or a `..` would quietly aim
# both of them somewhere other than where the message says.
if [[ ! $name =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "REFUSED: '$name' is not a usable worktree name" >&2
    echo "  allowed: A-Z a-z 0-9 . _ - and nothing else (no slashes, no empty name) —" >&2
    echo "  the name becomes both a path component and a /var/cache target-dir claim" >&2
    exit 66
fi

# Anchor on the COMMON gitdir's parent, never on `pwd` or on this script's own toplevel:
# run from inside a worktree, `--show-toplevel` answers with THAT worktree and the new one
# would nest under it. The common gitdir is the primary checkout's `.git` from anywhere,
# and `--path-format=absolute` is load-bearing — the bare form prints a path relative to
# `-C`'s directory, which resolves against the wrong `pwd` the moment they differ.
here=$(dirname "$(realpath "$0")")
common=$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)
repo=$(dirname "$common")
wt="$repo/.claude/worktrees/$name"
target="/var/cache/rivoli/target/$name"
# ONE spelling of the target dir: the claim check below searches for that PATH, and the file
# written further down builds its line out of the same variable, so what is searched for and
# what is written cannot drift.
line="target-dir = \"$target\""

# The registry is read ONCE into a variable, and every check below reads the variable. Read
# through a live pipe (`git … | grep -q`) it is a trap: grep exits on its first match, git
# dies of SIGPIPE, `pipefail` makes the pipeline non-zero, and under `||` a MATCH therefore
# reads as "no match" — the check inverted exactly when it had something to say.
registered=$(git -C "$repo" worktree list --porcelain)

# (b) The name is already taken. Both spellings are checked: the path on disk, and git's
# own registry — a `git worktree remove` that half-failed leaves one without the other, and
# `worktree add` then fails with a message about the half nobody looked at. `-L` as well as
# `-e` because a DANGLING symlink is invisible to `-e` and still blocks `worktree add`.
if [ -e "$wt" ] || [ -L "$wt" ] || grep -qxF "worktree $wt" <<<"$registered"; then
    echo "REFUSED: $name already exists as a worktree path or a registered worktree" >&2
    echo "  path: $wt" >&2
    exit 66
fi

# (b2) The branch is already taken. `worktree add -b` would refuse this too, but its message
# is about a ref, not about this script's arguments, and a refusal has to be readable where
# the mistake was made. Checked before anything is created.
if git -C "$repo" rev-parse --verify --quiet "refs/heads/$branch" >/dev/null; then
    echo "REFUSED: branch $branch already exists" >&2
    echo "  pick a fresh branch name, or delete that branch deliberately first" >&2
    exit 66
fi

# (c) The target dir is already claimed. Checked BEFORE creating anything, because the
# damage this prevents is two checkouts writing one build-script binary — the 7e-bis
# defect — and a worktree created and then removed has already touched the index.
#
# The set scanned is derived from git's own registry (plus the primary checkout), NOT from a
# glob under .claude/worktrees: a worktree registered anywhere else still holds a real claim.
# The match is on the PATH, not on a byte-exact line, so a hand-edited spelling of the same
# claim is still seen; the trailing quote keeps `target/rp-a` from matching `target/rp-ab`.
#
# Concurrent invocations are serialized by the coordinator, by convention. This script does
# not lock: the check-then-create window is a real TOCTOU and is accepted as such, because
# there is one writer.
mapfile -t configs < <(
    { printf '%s\n' "$repo"; grep '^worktree ' <<<"$registered" | cut -d' ' -f2-; } |
        sort -u | sed 's|$|/.cargo/config.toml|'
)
claimants=$(grep -lF -- "$target\"" "${configs[@]}" 2>/dev/null || true)
if [ -n "$claimants" ]; then
    echo "REFUSED: $target is already claimed by another checkout's config:" >&2
    while IFS= read -r f; do echo "  $f" >&2; done <<<"$claimants"
    echo "  one target dir per worktree — a shared one reuses a sibling's build-script" >&2
    echo "  binary and the jscpd gate silently does not run (gate-red-proofs.md 7e-bis)" >&2
    exit 66
fi

git -C "$repo" worktree add -b "$branch" "$wt" "$base"

# From here on the worktree EXISTS, so every remaining exit path has to undo it — a failed
# mkdir, a failed heredoc and a failed check are all half-made worktrees otherwise. One
# mechanism for all of them: a trap armed the moment `worktree add` succeeds and disarmed
# only after the last line of a successful run. The branch goes too, because `-b` created it
# and would otherwise refuse the re-run after the fix; `|| echo` so a cleanup that fails
# cannot swallow the exit status that caused it.
cleanup() {
    git -C "$repo" worktree remove --force "$wt" || echo "WARNING: $wt left behind" >&2
    git -C "$repo" branch -D "$branch" || echo "WARNING: branch $branch left behind" >&2
}
trap cleanup EXIT

mkdir -p "$wt/.cargo"
cat >"$wt/.cargo/config.toml" <<EOF
# Build output on the local btrfs cache subvolume, not this NFS worktree.
# See CLAUDE.md "Build cache and worktrees". Sets target-dir ONLY.
[build]
$line
EOF

# (d) Verify the ignore rather than trusting it. Unpiped on purpose: the status is read
# from the command itself, not from the tail of a pipeline (gate-red-proofs.md 7e-bis's
# other half). THREE outcomes, not two: 0 is the pass, 1 is "not ignored", and anything
# else is git itself failing — which leaves the question UNANSWERED, and an unanswered
# question is not a pass, so it refuses under its own message. The trap undoes the worktree
# on every one of these exits.
if git -C "$wt" check-ignore -q .cargo/config.toml; then
    ci=0
else
    ci=$?
fi
case $ci in
0) ;;
1)
    echo "REFUSED: .cargo/config.toml is NOT ignored, so it would show up as untracked" >&2
    echo "  and eventually be committed. Add '/.cargo/' to info/exclude in the COMMON" >&2
    echo "  gitdir ($common/info/exclude) — the per-worktree gitdir's info/exclude is" >&2
    echo "  NOT read, which is the trap this check exists for." >&2
    echo "  The worktree and branch just created are removed by the cleanup below." >&2
    exit 66
    ;;
*)
    echo "REFUSED: git check-ignore exited $ci, so whether .cargo/config.toml is ignored" >&2
    echo "  is UNKNOWN — git itself failed, which is a different finding from 'not" >&2
    echo "  ignored' and must not be reported as one. See what git says with:" >&2
    echo "    git -C $wt check-ignore -v .cargo/config.toml" >&2
    echo "  The worktree and branch just created are removed by the cleanup below." >&2
    exit 66
    ;;
esac

# Printed as lines a launch prompt can quote verbatim: the full 40-char oid, because an
# abbreviated sha in a prompt is ambiguous by the time anyone re-reads it.
echo
echo "worktree:  $wt"
echo "branch:    $branch"
echo "base sha:  $(git -C "$wt" rev-parse HEAD)"
echo "target-dir: $target"
echo
# WHY the variable and not just the config file: /etc/security/pam_env.conf sets a box-wide
# CARGO_TARGET_DIR for every login and an env var OUTRANKS build.target-dir, so the config
# written above is inert in any login shell (gate-red-proofs.md 12).
echo "run cargo in this worktree as (carry this prefix into the launch prompt):"
echo "  CARGO_TARGET_DIR=$target cargo <cmd>"

trap - EXIT
