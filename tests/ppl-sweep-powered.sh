#!/bin/bash
# The POWERED top-m quality gate: int3-vq only, ~5,000 tokens, shared baseline.
#
# int4 is dropped. It is not shipping (hybrid is parked and this artifact's .i4 is
# vq3-derived), and its one remaining scientific purpose — whether the swap->cost slope
# transfers across quantisations — was answered by the 762-token run: it does not, carried
# by the 17.6%-swap pair at 3.7 SE.
#
# J=4/M=9  is the SHIPPING CANDIDATE. At this corpus size it returns a clean PASS if its
#          true cost is under roughly 0.3%.
# J=4/M=10 is NOT expected to pass. It is here to be bounded tightly enough to REJECT, so
#          the higher-benefit cell cannot be relitigated later as "never properly measured".
# J=2/M=12 is absent on purpose: already decided and rejected (int4 +12.7% with the whole
#          interval past the bar; int3-vq lower bound +0.68%). More text only tightens a
#          failing interval around a failing value.
#
# One process per cell — sequencing cells inside one process lets an earlier cell leave the
# pool warm for a later one, which flatters whichever runs second.
#
# usage: ppl-sweep-powered.sh <out-dir> [max-mem-gib]
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: ppl-sweep-powered.sh <out-dir> [max-mem-gib]}"
MEM="${2:-100}"
MODEL=/var/db/rivoli/glm52-vq3-full
CORPUS="$W/tests/ppl-corpus-5000.txt"
mkdir -p "$OUT"

run() { # run <tag> <policy> [extra...]
  local tag=$1 policy=$2; shift 2
  echo "=== $tag ==="
  "$W/target/release/rivoli" "$MODEL" \
    --mode int3-vq --cache-policy "$policy" --attn dense --max-mem "$MEM" \
    --ppl "$CORPUS" --ppl-out "$OUT/$tag.nll" "$@" \
    > "$OUT/$tag.log" 2>&1
  echo "  exit=$? $(grep -o 'PPL: [0-9.]*' "$OUT/$tag.log" | head -1)"
}

run base  lru
run j4m9  top-m --route-j 4 --route-m 9
run j4m10 top-m --route-j 4 --route-m 10

echo
"$W/target/release/ppl" "$OUT/base.nll" "$OUT/j4m9.nll" "$OUT/j4m10.nll"
echo "=== POWERED SWEEP DONE ==="
