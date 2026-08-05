#!/usr/bin/env bash
# Re-run the gates and diff against a captured baseline. Non-zero if ANY moved OR was UNRUN.
set -uo pipefail
cd "$(dirname "$0")/../.."
BASE="${1:?usage: check.sh <baseline-dir>}"
NEW=$(mktemp -d); trap 'rm -rf "$NEW"' EXIT

# capture.sh's own exit code is load-bearing: 66 means a foreign GPU holder. The previous
# version sent all of it to /dev/null and then advised its caller to read exit codes.
"$(dirname "$0")/capture.sh" "$NEW" > "$NEW/capture.log" 2>&1
crc=$?
[ "$crc" -ne 0 ] && echo "  capture.sh rc=$crc (see $NEW/capture.log)"

fail=0

# 1. UNRUN before anything else. Both prior vacuity bugs produced NON-EMPTY output that
#    compared equal to itself, so emptiness was never the right signal.
for s in "$NEW"/*.status; do
  [ -f "$s" ] || continue
  g=$(basename "$s" .status)
  read -r st reason < "$s" || true
  [ "$st" = RAN ] || { printf '  %-14s UNRUN: %s — not passed\n' "$g" "${reason:-?}"; fail=1; }
done

# 2. Provenance: a gate that measures target/ says nothing about the source revision.
if [ -f "$BASE/provenance.txt" ] && [ -f "$NEW/provenance.txt" ]; then
  newest=$(awk '/^newest_src/{print $2}' "$NEW/provenance.txt")
  for b in rivoli convert_v4; do
    m=$(awk -v b="$b" '$1==b{print $3}' "$NEW/provenance.txt")
    if [ -n "$m" ] && [ -n "$newest" ] && [ "$m" -lt "$newest" ]; then
      printf '  %-14s STALE BINARY: target/release/%s predates the newest source file\n' provenance "$b"; fail=1
    fi
  done
fi

# 3. The hashes.
for g in g3-isa.txt g1-glm.txt g1-v4.txt; do
  [ -s "$BASE/$g" ] || { printf '  %-14s baseline absent\n' "$g"; continue; }
  [ -s "$NEW/$g" ]  || { printf '  %-14s current absent — unrun, not passed\n' "$g"; fail=1; continue; }
  a=$(sha256sum < "$BASE/$g" | cut -c1-16); b=$(sha256sum < "$NEW/$g" | cut -c1-16)
  if [ "$a" = "$b" ]; then printf '  %-14s OK    %s\n' "$g" "$a"
  else printf '  %-14s MOVED %s -> %s\n' "$g" "$a" "$b"; diff "$BASE/$g" "$NEW/$g" | head -20; fail=1; fi
done

# 4. G2 by the SET of verified layers, not a substring count. `grep -c "0 bytes differ"`
#    also matched "40 bytes differ", so a corruption of 5120 bytes left the count unchanged.
if [ -f "$BASE/g2-verify.log" ] && [ -f "$NEW/g2-verify.log" ]; then
  ext() { grep -oE '^convert_v4: verified L[0-9]+\.f4 .* 0 bytes differ$' "$1" | grep -oE 'L[0-9]+' | sort -u; }
  if diff <(ext "$BASE/g2-verify.log") <(ext "$NEW/g2-verify.log") > "$NEW/g2.diff"; then
    printf '  %-14s OK    %s layers\n' g2-verify "$(ext "$NEW/g2-verify.log" | wc -l)"
  else printf '  %-14s MOVED — verified-layer set changed\n' g2-verify; head -10 "$NEW/g2.diff"; fail=1; fi
fi

[ "$fail" -eq 0 ] && echo "ALL GATES HELD" || echo "GATES MOVED OR UNRUN — this is not a refactor"
exit "$fail"
