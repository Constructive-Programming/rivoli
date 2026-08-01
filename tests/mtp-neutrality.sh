#!/usr/bin/env bash
# Speculative decode must move SPEED and never OUTPUT (docs/reference/architecture.md §13). This is
# the paired check for that, one pair per `--attn` mode: decode once with `--no-mtp` and
# once with the default, and compare token IDS. Not text — different id sequences can
# decode to identical text, so a text diff reports only a lower bound on divergence.
#
# THE TRAP THIS EXISTS TO SIDESTEP, and the reason it is a script rather than a line in
# bench-matrix.sh: **every sparse mode has a dense fast path, and a short run stays inside
# it.** `dsa_select_layer` returns dense while the context is <= index_topk (2048 on the
# reference artifact), and `streaming_rows` returns the whole prefix while it is <= sinks +
# window (8192 by default). So a 256-token pair under `--attn dsa` or `--attn streaming`
# executes none of the batched per-row selection it claims to test, and passes vacuously.
# That is exactly what happened on 2026-08-01 before this script existed.
#
# Two answers, one per mode, both of which force the selection to actually run:
#   - dsa/misa: also run the pair against a SHADOW artifact — a directory of symlinks to
#     every file, plus one manifest.json with index_topk lowered. Costs a directory, not a
#     copy of 60 GB, and beats prefilling 2048 tokens at ~0.38 s/token.
#   - streaming: `--window` is a CLI flag, so a small one is all it takes (see mode_flags).
#
# Usage: tests/mtp-neutrality.sh <artifact> [attn ...]     # default: dense dsa streaming
#   RIVOLI_NGEN=48  RIVOLI_LOW_TOPK=16  RIVOLI_WINDOW=16  RIVOLI_MAX_MEM=115  override.
#
# The artifact must carry the MTP head (`L78.i4`), or every arm silently decodes
# sequentially and the comparison is two identical sequential runs — checked below.
set -euo pipefail

ART=${1:?usage: mtp-neutrality.sh <artifact> [attn ...]}
shift || true
MODES=("${@:-}")
[ -z "${MODES[0]}" ] && MODES=(dense dsa streaming)

BIN=./target/release/rivoli
NGEN=${RIVOLI_NGEN:-48}
LOW_TOPK=${RIVOLI_LOW_TOPK:-16}
MAX_MEM=${RIVOLI_MAX_MEM:-115}
# A prompt that stays coherent: the default ("The sky is blue because") trips the
# degeneration warning on this checkpoint, which is a different regime.
PROMPT=${RIVOLI_PROMPT:-"Explain how a CPU cache hierarchy works, and why it exists."}
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

[ -x "$BIN" ] || { echo "no $BIN — cargo build --release --features rocm"; exit 2; }
[ -e "$ART/L78.i4" ] || echo "WARNING: $ART has no L78.i4; without the MTP head every arm decodes sequentially and this proves nothing"

# One arm. The GPU is sole-tenant (CLAUDE.md), so every invocation takes the lock — per
# arm rather than around the whole script, so a long matrix does not hold it for an hour.
arm() { # arm <artifact> <attn> <ids-out> [extra flags...]
  local art=$1 attn=$2 out=$3; shift 3
  flock -w 1800 /tmp/rivoli-gpu.lock -c \
    "timeout 1800 $BIN '$art' -bench $NGEN --mode int3-vq --attn $attn \
     --cache-policy 2q --max-mem $MAX_MEM --prompt '$PROMPT' --dump-ids '$out' $*" \
    2>&1 | grep -aE "tok/s|MTP:|speculative decode OFF" || true
}

# Flags that make a mode's row selection actually SPARSE inside a short run — without
# them the mode takes its own dense fast path and the pair proves nothing. `streaming`'s
# default window is 8192, i.e. wider than any run this script does.
mode_flags() {
  case $1 in
    streaming) echo "--sinks 4 --window ${RIVOLI_WINDOW:-16}" ;;
    misa) echo "--misa-heads 8" ;;
    *) echo "" ;;
  esac
}

pair() { # pair <artifact> <attn> <label>
  local art=$1 attn=$2 label=$3
  local extra; extra=$(mode_flags "$attn")
  echo "=== $attn $label ${extra:+[$extra]} ==="
  # shellcheck disable=SC2086  # extra is a deliberately word-split flag list
  arm "$art" "$attn" "$WORK/seq.ids" --no-mtp $extra
  # shellcheck disable=SC2086
  arm "$art" "$attn" "$WORK/spec.ids" $extra
  if cmp -s "$WORK/seq.ids" "$WORK/spec.ids"; then
    echo "PASS  $attn $label — speculation is output-neutral ($NGEN token ids identical)"
  else
    echo "FAIL  $attn $label — speculation CHANGED the output:"
    diff "$WORK/seq.ids" "$WORK/spec.ids" | head -10
    FAILED=1
  fi
}

# The shadow artifact: symlinks + one edited manifest, so the selector runs from token
# ~LOW_TOPK instead of token ~2048. Costs a directory, not a copy of 60 GB.
shadow() {
  local d="$WORK/shadow"
  mkdir -p "$d"
  for f in "$ART"/*; do
    [ "$(basename "$f")" = manifest.json ] || ln -sf "$f" "$d/"
  done
  python3 -c "
import json,sys
m=json.load(open(sys.argv[1])); m['index_topk']=int(sys.argv[3])
json.dump(m,open(sys.argv[2],'w'))" "$ART/manifest.json" "$d/manifest.json" "$LOW_TOPK"
  echo "$d"
}

FAILED=0
for attn in "${MODES[@]}"; do
  pair "$ART" "$attn" "(as shipped)"
  # Only dsa/misa consult index_topk; for dense the shadow arm would be the same run.
  case "$attn" in
    dsa | misa) pair "$(shadow)" "$attn" "(index_topk=$LOW_TOPK — selector actually runs)" ;;
  esac
done
exit $FAILED
