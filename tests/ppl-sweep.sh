#!/bin/bash
# `top-m` quality gate — teacher-forced perplexity over a fixed corpus.
#
# ONE PROCESS PER CELL, deliberately: running cells in sequence inside one process lets
# an earlier cell leave the pool warm for a later one, which silently improves whichever
# cell happens to run second and confounds exactly the comparison being made. A fresh
# process gives every cell an identical cold pool.
#
# Baseline is measured in this same harness with the same LRU family and the same budget
# — a baseline obtained any other way is not a baseline.
#
# usage: ppl-sweep.sh <out-dir> [max-mem-gib]
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: ppl-sweep.sh <out-dir> [max-mem-gib]}"
MEM="${2:-100}"
MODEL=/var/db/rivoli/glm52-vq3-full
CORPUS="$W/tests/ppl-corpus.txt"
mkdir -p "$OUT"

run() { # run <tag> <policy> <mode> [extra...]
  local tag=$1 policy=$2 mode=$3; shift 3
  echo "=== $tag ==="
  "$W/target/release/rivoli" "$MODEL" \
    --mode "$mode" --cache-policy "$policy" --attn dense --max-mem "$MEM" \
    --ppl "$CORPUS" --ppl-out "$OUT/$tag.nll" "$@" \
    > "$OUT/$tag.log" 2>&1
  echo "  exit=$? $(grep -o 'PPL: [0-9.]*' "$OUT/$tag.log" | head -1)"
}

for mode in int3-vq int4; do
  run "$mode-base"    lru   "$mode"
  run "$mode-j4m10"   top-m "$mode" --route-j 4 --route-m 10
  run "$mode-j2m12"   top-m "$mode" --route-j 2 --route-m 12
done

for mode in int3-vq int4; do
  echo; echo "########## $mode ##########"
  "$W/target/release/ppl" "$OUT/$mode-base.nll" "$OUT/$mode-j4m10.nll" "$OUT/$mode-j2m12.nll"
done
echo "=== PPL SWEEP DONE ==="
