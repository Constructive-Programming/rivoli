#!/bin/bash
# Benchmark matrix runner — mode x attn x cache-policy, in rounds of decreasing width
# and increasing length (44 @ 512 -> 8 @ 2048 -> 4 @ 4096 -> 2 @ 10000).
#
# ONE PROCESS PER CELL, deliberately: cells sharing a process would leave the pool warm
# for whichever ran second, which is the confound this whole matrix exists to avoid.
#
# The runner's real job is NOT to launch 44 processes — it is to NOTICE when a cell goes
# wrong, because every failure mode here is silent by default:
#   - a crash / OOM leaves no PROFILE line, and an unchecked loop just moves on
#   - a wedged GPU hangs forever and would stall the matrix overnight
#   - a DEGENERATE generation still prints a perfectly plausible tok/s, and it is
#     inflated (a loop reuses a few experts -> high hit rate -> looks FAST)
# Each cell is therefore classified ok | DEGENERATE | TIMEOUT | CRASH, and the summary
# at the end refuses to rank anything that is not `ok`.
#
# usage: bench-matrix.sh <out-dir> [round]     round = 1|2|3|4 (default 1)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: bench-matrix.sh <out-dir> [round]}"
ROUND="${2:-1}"
BIN="${RIVOLI_BIN:-$W/target/release/rivoli}"
MODEL="${RIVOLI_MODEL:-/var/db/rivoli/glm52-vq3-full}"
MEM="${RIVOLI_MEM:-115}"
mkdir -p "$OUT"

case "$ROUND" in
  1) TOKENS=512   ;;
  2) TOKENS=2048  ;;
  3) TOKENS=4096  ;;
  4) TOKENS=10000 ;;
  *) echo "round must be 1..4"; exit 2 ;;
esac
# Generous: a 10k cell is ~80 min at ~2 tok/s. A cell that exceeds this is wedged, and
# killing it is the point — one hung cell must not cost the whole night.
TIMEOUT=$(( 600 + TOKENS * 2 ))

# Round 1 enumerates the full cross. Later rounds read the cells named in
# $OUT/roundN.cells (one "mode attn policy" per line) so selection stays a human
# decision — see the summary's warning about ranking on tok/s.
CELLS=()
if [ "$ROUND" = 1 ]; then
  for mode in int3-vq int4 hybrid; do
    for attn in dense streaming dsa misa; do
      for pol in lru 2q arc top-m; do
        # config.rs::validate rejects top-m + hybrid outright (the hybrid rank-driven
        # tier rule is not built; a fallback would credit its behaviour to top-m).
        [ "$pol" = top-m ] && [ "$mode" = hybrid ] && continue
        CELLS+=("$mode $attn $pol")
      done
    done
  done
else
  SEL="$OUT/round$ROUND.cells"
  [ -f "$SEL" ] || { echo "missing $SEL — write one 'mode attn policy' per line"; exit 2; }
  while read -r line; do [ -n "$line" ] && CELLS+=("$line"); done < "$SEL"
fi

echo "round $ROUND: ${#CELLS[@]} cells x $TOKENS tok, --max-mem $MEM, timeout ${TIMEOUT}s"
echo "binary: $BIN"
printf 'status\tmode\tattn\tpolicy\ttok_s\thit_pct\twall_ms\tlrb\tnote\n' > "$OUT/round$ROUND.tsv"

for cell in "${CELLS[@]}"; do
  read -r mode attn pol <<< "$cell"
  tag="$mode-$attn-$pol"
  log="$OUT/r$ROUND-$tag.log"
  # Resumable: an overnight matrix that cannot be restarted after one bad cell is a
  # matrix you run once and then cannot fix.
  if [ -s "$log" ] && grep -q 'PROFILE/tok' "$log"; then
    echo "  skip $tag (already has a result)"
    continue
  fi
  extra=()
  [ "$pol" = top-m ] && extra+=(--route-j 4 --route-m 9)
  [ "$attn" = streaming ] && extra+=(--sinks 4 --window 512)
  [ "$attn" = misa ] && extra+=(--misa-heads 32)

  start=$(date +%s)
  timeout -k 30 "$TIMEOUT" "$BIN" "$MODEL" \
      --mode "$mode" --cache-policy "$pol" --attn "$attn" --max-mem "$MEM" \
      -bench "$TOKENS" "${extra[@]}" > "$log" 2>&1
  rc=$?
  el=$(( $(date +%s) - start ))

  tok_s=$(grep -oE '\(([0-9.]+) tok/s' "$log" | tail -1 | tr -dc '0-9.')
  hit=$(grep -oE 'expert hit [0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+')
  wall=$(grep -oE ': [0-9.]+ms wall' "$log" | tail -1 | grep -oE '[0-9.]+')
  # Two independent degeneration signals: a verbatim tail CYCLE (late-stage) and an
  # oversized repeated block with no cycle (a RESTART — early-stage, and the one seen on
  # a real run). Either disqualifies the cell from ranking.
  loop=$(grep -oE 'DEGENERATE OUTPUT.*' "$log" | grep -oE '[0-9]+' | tr '\n' '/' | sed 's:/$::')
  suspect=$(grep -c 'SUSPECT OUTPUT' "$log")
  lrb=$(grep -oE 'longest repeated block [0-9]+' "$log" | grep -oE '[0-9]+$')

  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    status=TIMEOUT; note="killed after ${el}s — wedged?"
  elif [ "$rc" != 0 ]; then
    status=CRASH
    note=$(grep -oiE '(Error|error\[|panicked at|OOM|out of memory|foreign|refuse).*' "$log" | head -1 | cut -c1-120)
    note="rc=$rc ${note:-no error line}"
  elif [ -z "$tok_s" ]; then
    status=CRASH; note="exit 0 but no PROFILE line"
  elif [ -n "$loop" ]; then
    status=DEGENERATE; note="tail cycle; tok/s inflated by expert reuse — do not rank"
  elif [ "$suspect" != 0 ]; then
    status=SUSPECT; note="restart (lrb=$lrb), no cycle — tok/s suspect"
  else
    status=ok; note="${el}s"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$status" "$mode" "$attn" "$pol" "${tok_s:--}" "${hit:--}" "${wall:--}" "${lrb:--}" "$note" \
      >> "$OUT/round$ROUND.tsv"
  printf '  %-11s %-28s %sst %s tok/s\n' "$status" "$tag" "$el" "${tok_s:--}"
done

echo
echo "=== round $ROUND summary ==="
column -t -s$'\t' "$OUT/round$ROUND.tsv"
ok=$(grep -c '^ok' "$OUT/round$ROUND.tsv" || true)
deg=$(grep -cE '^(DEGENERATE|SUSPECT)' "$OUT/round$ROUND.tsv" || true)
bad=$(grep -cE '^(CRASH|TIMEOUT)' "$OUT/round$ROUND.tsv" || true)
echo
echo "ok=$ok degenerate/suspect=$deg crashed/timed-out=$bad"
[ "$deg" -gt 0 ] && echo "WARNING: $deg cell(s) looped. Their tok/s is inflated (a loop \
reuses a few experts -> high hit rate -> looks fast) and must NOT be ranked."
[ "$bad" -gt 0 ] && echo "WARNING: $bad cell(s) did not produce a result. Investigate \
before treating this round as complete."
echo "=== ROUND $ROUND DONE ==="
