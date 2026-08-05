#!/usr/bin/env bash
# G4b driver. See breaks.tsv for the contract.
set -uo pipefail
cd "$(dirname "$0")/../.."
TSV="$(dirname "$0")/breaks.tsv"
pass=0; fail=0

# Refuse to run on a dirty tree: the restore step is `git checkout --`, which would discard
# uncommitted work along with the break.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "REFUSING: working tree is dirty. Commit or stash first — the restore step is destructive."
  exit 2
fi

while IFS=$'\t' read -r file find repl test expect; do
  case "$file" in ''|\#*) continue ;; esac
  before=$(cargo test --release --features rocm --test ${test%% *} ${test#* } 2>&1)
  if ! grep -q "test result: ok" <<<"$before"; then
    printf '  %-46s SKIP (already red before the break)\n' "$test"; continue
  fi
  python3 - "$file" "$find" "$repl" <<'PY'
import sys, re
p, find, repl = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if find not in s: sys.exit(f"BREAK ANCHOR MISSING in {p}: {find!r}")
open(p,'w').write(s.replace(find, repl, 1))
PY
  [ $? -ne 0 ] && { printf '  %-46s ANCHOR MOVED — corpus is stale\n' "$test"; fail=$((fail+1)); continue; }
  out=$(cargo test --release --features rocm --test ${test%% *} ${test#* } 2>&1)
  git checkout -- "$file"
  if grep -q "test result: FAILED" <<<"$out" && grep -qF "$expect" <<<"$out"; then
    printf '  %-46s RED, message matches\n' "$test"; pass=$((pass+1))
  elif grep -q "test result: FAILED" <<<"$out"; then
    printf '  %-46s RED but WRONG SUBJECT (expected %q)\n' "$test" "$expect"; fail=$((fail+1))
  else
    printf '  %-46s STAYED GREEN — the gate is gone\n' "$test"; fail=$((fail+1))
  fi
done < "$TSV"

echo "breaks fired: $pass   problems: $fail"
exit $(( fail > 0 ))
