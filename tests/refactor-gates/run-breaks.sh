#!/usr/bin/env bash
# G4b driver. See breaks.tsv for the contract.
#
# Three lessons are built in, each from a defect this harness itself had on first run:
#   1. `IFS=$'\t' read` COLLAPSES consecutive tabs, so an empty column shifts every field
#      left. Parsed with awk, which does not.
#   2. `cargo test --test X <filter>` matching NOTHING prints "test result: ok. 0 passed"
#      and reads as green. Every run asserts a test actually EXECUTED.
#   3. The restore step is `git checkout --`, which discards uncommitted work. Refuse a
#      dirty tree.
set -uo pipefail
cd "$(dirname "$0")/../.."
TSV="$(dirname "$0")/breaks.tsv"
pass=0; fail=0

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "REFUSING: working tree is dirty. Commit first — the restore step is destructive."
  exit 2
fi

# Restore on ANY exit, not just the happy path. A 10-minute harness timeout killed this
# script between "apply break" and "git checkout --" on 2026-08-05 and left the ORACLE
# modified -- the one file everything else is scored against. Without this trap the next
# measurement would have been taken against a deliberately broken reference.
BROKEN=""
restore() { [ -n "$BROKEN" ] && git checkout -- $BROKEN 2>/dev/null; BROKEN=""; }
trap 'restore; echo "  (interrupted — tree restored)"; exit 130' INT TERM HUP
trap restore EXIT

# "test result: ok. N passed" with N>0, or it did not run.
ran_and_passed() { grep -qE "test result: ok\. [1-9][0-9]* passed" <<<"$1"; }
ran_and_failed() { grep -qE "test result: FAILED\. [0-9]+ passed; [1-9]" <<<"$1"; }

while IFS= read -r line; do
  case "$line" in ''|\#*) continue ;; esac
  file=$(awk -F'\t' '{print $1}' <<<"$line")
  find=$(awk -F'\t' '{print $2}' <<<"$line")
  repl=$(awk -F'\t' '{print $3}' <<<"$line")
  test=$(awk -F'\t' '{print $4}' <<<"$line")
  expect=$(awk -F'\t' '{print $5}' <<<"$line")
  bin=${test%% *}
  filter=""; [ "$bin" != "$test" ] && filter=${test#* }

  # Five of these rows drive device suites. capture.sh takes the lock and this did not --
  # the one script in the gate suite that ignored its own standing hazard list. Build first,
  # OUTSIDE the lock (CLAUDE.md: never cargo build between the arms of a measurement), then
  # run under it with the witness re-checked inside.
  cargo test --release --features rocm --test "$bin" --no-run >/dev/null 2>&1
  before=$(flock -w 3600 /var/run/sys-gpu.lock bash -c '
      n=$(find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
      [ "$n" -eq 0 ] || { echo "FOREIGN GPU HOLDER: $n"; exit 66; }
      cargo test --release --features rocm --test '"$bin"' '"$filter"' -- --test-threads=1' 2>&1)
  if ! ran_and_passed "$before"; then
    printf '  %-44s SKIP: no test executed or already red\n' "$test"; fail=$((fail+1)); continue
  fi

  BROKEN="$file"
  if ! python3 -c '
import sys
p,f,r = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if f not in s: sys.exit(1)
open(p,"w").write(s.replace(f, r, 1))
' "$file" "$find" "$repl"; then
    printf '  %-44s ANCHOR MISSING — corpus is stale\n' "$test"; fail=$((fail+1)); continue
  fi

  BROKEN="$file"
  cargo test --release --features rocm --test "$bin" --no-run >/dev/null 2>&1
  out=$(flock -w 3600 /var/run/sys-gpu.lock bash -c '
      cargo test --release --features rocm --test '"$bin"' '"$filter"' -- --test-threads=1' 2>&1)
  git checkout -- "$file" || { echo "  RESTORE FAILED for $file"; exit 3; }
  BROKEN=""

  # Strip libtest's own status lines before checking the message. `grep -F stream` against
  # the raw output matches `test the_attention_block_is_entirely_on_its_stream ... FAILED`,
  # so that row's subject check was unconditionally true and any red for any reason -- GPU
  # contention, an unrelated assertion -- read as "message matches".
  body=$(grep -vE '^test [a-z0-9_]+ \.\.\. (ok|FAILED)$|^    [a-z0-9_]+$|^failures:' <<<"$out")
  if ran_and_failed "$out" && grep -qF "$expect" <<<"$body"; then
    printf '  %-44s RED, message matches\n' "$test"; pass=$((pass+1))
  elif grep -qE "error\[E0|^error: " <<<"$out"; then
    # NOT a pass. This branch is reached only when the test did not go red with the right
    # message, and a build failure means the REPLACEMENT text no longer type-checks -- i.e.
    # the row is stale for the tree it is gating. Scoring it green is how a corpus reports
    # 7/7 forever while gating nothing: Track 2 changes launcher arity, row 7's replacement
    # stops compiling, and the stream gate is never exercised again. hipcc says plain
    # "error:", rustc says "error[E0...]"; both count.
    printf '  %-44s COMPILE-BROKEN — row is stale, gate NOT exercised\n' "$test"; fail=$((fail+1))
  elif ran_and_failed "$out"; then
    printf '  %-44s RED but WRONG SUBJECT (wanted: %s)\n' "$test" "$expect"; fail=$((fail+1))
  else
    printf '  %-44s STAYED GREEN — the gate is gone\n' "$test"; fail=$((fail+1))
  fi
done < "$TSV"

git diff --quiet || { echo "TREE LEFT DIRTY — a restore failed"; fail=$((fail+1)); }
echo "breaks fired: $pass   problems: $fail"
exit $(( fail > 0 ))
