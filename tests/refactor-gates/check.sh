#!/usr/bin/env bash
# Re-run the gates and diff against a captured baseline. Exit non-zero if ANY moved.
#
# Read the exit code, not the output. A `| tail` or a trailing `[ $rc -eq 1 ]` swallows it;
# both have happened here.
set -uo pipefail
cd "$(dirname "$0")/../.."
BASE="${1:?usage: check.sh <baseline-dir>}"
NEW=$(mktemp -d)
trap 'rm -rf "$NEW"' EXIT

"$(dirname "$0")/capture.sh" "$NEW" >/dev/null 2>&1
fail=0
for g in g3-isa.txt g2-verify.log g1-glm.txt g1-v4.txt; do
  if [ ! -s "$BASE/$g" ]; then
    printf '  %-16s BASELINE ABSENT — gate unrun, not passed\n' "$g"; fail=1; continue
  fi
  if [ ! -s "$NEW/$g" ]; then
    printf '  %-16s CURRENT ABSENT — gate unrun, not passed\n' "$g"; fail=1; continue
  fi
  # g2's log carries timings; compare only the byte-difference verdicts.
  if [ "$g" = g2-verify.log ]; then
    a=$(grep -c "0 bytes differ" "$BASE/$g"); b=$(grep -c "0 bytes differ" "$NEW/$g")
  else
    a=$(sha256sum < "$BASE/$g" | cut -c1-16); b=$(sha256sum < "$NEW/$g" | cut -c1-16)
  fi
  if [ "$a" = "$b" ]; then printf '  %-16s OK    %s\n' "$g" "$a"
  else printf '  %-16s MOVED %s -> %s\n' "$g" "$a" "$b"; diff "$BASE/$g" "$NEW/$g" | head -20; fail=1; fi
done
[ "$fail" -eq 0 ] && echo "ALL GATES HELD" || echo "GATES MOVED — this is not a refactor"
exit "$fail"
