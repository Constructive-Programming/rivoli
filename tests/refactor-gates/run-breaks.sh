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

  before=$(cargo test --release --features rocm --test "$bin" $filter -- --test-threads=1 2>&1)
  if ! ran_and_passed "$before"; then
    printf '  %-44s SKIP: no test executed or already red\n' "$test"; fail=$((fail+1)); continue
  fi

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
  out=$(cargo test --release --features rocm --test "$bin" $filter -- --test-threads=1 2>&1)
  git checkout -- "$file"; BROKEN=""

  if ran_and_failed "$out" && grep -qF "$expect" <<<"$out"; then
    printf '  %-44s RED, message matches\n' "$test"; pass=$((pass+1))
  elif grep -q "error\[E0" <<<"$out"; then
    printf '  %-44s COMPILER caught it (stronger than a test)\n' "$test"; pass=$((pass+1))
  elif ran_and_failed "$out"; then
    printf '  %-44s RED but WRONG SUBJECT (wanted: %s)\n' "$test" "$expect"; fail=$((fail+1))
  else
    printf '  %-44s STAYED GREEN — the gate is gone\n' "$test"; fail=$((fail+1))
  fi
done < "$TSV"

echo "breaks fired: $pass   problems: $fail"
exit $(( fail > 0 ))
