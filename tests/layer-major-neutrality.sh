#!/usr/bin/env bash
# Layer-major prefill (the default since 2026-08-03) must move the EXPERT READ COUNT
# and never the output. This is the
# paired check for that, one pair per `--attn` mode: prefill once token-major and once
# layer-major, compare token IDS, and compare the reads/token the new PREFILL line reports.
# Not text — different id sequences can decode to identical text, so a text diff reports
# only a lower bound on divergence (same reason as tests/mtp-neutrality.sh).
#
# WHY BOTH HALVES. Neutrality alone would pass on a build where the flag did nothing, and a
# read-count drop alone would pass on a build that produced garbage cheaply. The pair of
# assertions is what makes this a gate rather than a statistic.
#
# TWO WAYS TO PASS VACUOUSLY, both of which this sidesteps:
#
#   1. A SHORT PROMPT. Layer-major reorders the prompt against the layers, so at N tokens it
#      runs ceil(N/2) passes per layer. At N=6 that is 3, the pool holds every expert either
#      way, and both arms read the same bytes. The read reduction only appears once a
#      layer's experts would otherwise be evicted before the next token wants them, which
#      needs hundreds of tokens — hence the corpus prompt below rather than a sentence.
#   2. A SPARSE MODE'S DENSE FAST PATH. `dsa_select_layer` returns dense while the context
#      is <= index_topk (2048 as shipped), so a sub-2048 dsa run never writes the indexer's
#      `sel` buffer — the cross-pass IndexShare state that layer-major exists to have keyed
#      by position instead of by row slot. The shadow artifact below lowers index_topk so
#      that path actually executes. Same trick, same reason, as mtp-neutrality.sh.
#
# Usage: tests/layer-major-neutrality.sh <artifact> [attn ...]   # default: dense dsa streaming
#   RIVOLI_NGEN=8  RIVOLI_LOW_TOPK=64  RIVOLI_WINDOW=64  RIVOLI_MAX_MEM=115
#   RIVOLI_PROMPT_BYTES=3200  RIVOLI_PROMPT_FILE=tests/ppl-corpus-5000.txt   override.
set -euo pipefail

ART=${1:?usage: layer-major-neutrality.sh <artifact> [attn ...]}
shift || true
MODES=("${@:-}")
[ -z "${MODES[0]}" ] && MODES=(dense dsa streaming)

BIN=./target/release/rivoli
# /var/run is /run, which is TMPFS — this file does not survive a reboot, and /run is
# root-owned so flock cannot recreate it. That failure is loud (exit 66, "cannot open lock
# file"), never a silent unlocked run; restore with
#   sudo install -m 666 -o "$USER" /dev/null /run/sys-gpu.lock
#
# Overridable because a lock path can only be changed EVERYWHERE AT ONCE. Two cohorts on
# two paths are not serialised against each other at all — they are two GPU tenants that
# both believe they hold the mutex, which is strictly worse than no lock, since no lock at
# least makes people careful.
#
# The 2026-08-02 move off `/tmp/rivoli-gpu.lock` is COMPLETE — every consumer in this repo
# and llama-swap on rh-anine are all on `/var/run/sys-gpu.lock` (docs/reference/gpu-lock.md).
# This line used to say "run with RIVOLI_GPU_LOCK=/tmp/rivoli-gpu.lock to stay in their
# queue"; following that today would put you in a queue of one and create exactly the split
# the paragraph above warns about. The override stays for the NEXT such migration.
GPU_LOCK=${RIVOLI_GPU_LOCK:-/var/run/sys-gpu.lock}
NGEN=${RIVOLI_NGEN:-8}
LOW_TOPK=${RIVOLI_LOW_TOPK:-64}
MAX_MEM=${RIVOLI_MAX_MEM:-115}
PROMPT_FILE=${RIVOLI_PROMPT_FILE:-tests/ppl-corpus-5000.txt}
# Long enough that a layer's experts cannot all stay resident across the prompt, which is
# the only regime where the two orders differ. ~700-800 tokens at this corpus's byte rate.
PROMPT=$(head -c "${RIVOLI_PROMPT_BYTES:-3200}" "$PROMPT_FILE" | tr -d '\r')
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

[ -x "$BIN" ] || { echo "no $BIN — cargo build --release --features rocm"; exit 2; }

# One arm. The GPU is sole-tenant (CLAUDE.md), so every invocation takes the lock — per arm
# rather than around the whole script, so a long matrix does not hold it for an hour.
# `--no-mtp`: the head is a 79th MoE layer whose own reads would be folded into the same
# counter, and this gate is about the model's prefill. mtp-neutrality.sh covers the head.
#
# `flock LOCK cmd args...`, NOT `flock LOCK -c "string"`: the prompt is ~3200 bytes of
# English prose, and prose has apostrophes — a `-c` string re-parsed by a shell dies on the
# first `don't`. Not hypothetical: it is exactly how the first run of this script failed,
# and it failed by printing nothing rather than by saying anything.
arm() { # arm <artifact> <attn> <ids-out> <log-out> [extra flags...]
  local art=$1 attn=$2 ids=$3 log=$4; shift 4
  flock -w 3600 "$GPU_LOCK" \
    timeout 3600 "$BIN" "$art" -bench "$NGEN" --mode int3-vq --attn "$attn" --no-mtp \
    --cache-policy 2q --max-mem "$MAX_MEM" --prompt "$PROMPT" --dump-ids "$ids" "$@" \
    > "$log" 2>&1 || true
  # `|| true` on the grep: an arm that died leaves no PREFILL line, and under `set -e` a
  # grep matching nothing aborts the script before `pair` can report WHICH arm died.
  grep -aoE "PREFILL: [0-9]+ tokens in [0-9.]+ s \([a-z-]+\) \| [0-9]+ expert reads, [0-9.]+/token" \
    "$log" | tail -1 || true
}

# reads/token out of the PREFILL line — the field this flag exists to move.
rpt() { sed -E 's/.*, ([0-9.]+)\/token.*/\1/' <<<"$1"; }

mode_flags() {
  case $1 in
    streaming) echo "--sinks 4 --window ${RIVOLI_WINDOW:-64}" ;;
    misa) echo "--misa-heads 8" ;;
    *) echo "" ;;
  esac
}

pair() { # pair <artifact> <attn> <label>
  local art=$1 attn=$2 label=$3 extra a b ra rb
  extra=$(mode_flags "$attn")
  echo "=== $attn $label ${extra:+[$extra]} ==="
  # shellcheck disable=SC2086  # extra is a deliberately word-split flag list
  # The arms swapped on 2026-08-03, when `--layer-major-prefill` was deleted and
  # layer-major became the default. The remaining way to force TOKEN-major is `--trace`:
  # a v2 trace recovers token boundaries from the layer id descending, which a layer-major
  # prefill never does, so the engine falls back for a capture. That makes the control arm
  # `--trace` and the treatment arm bare. The trace file is a side effect we discard.
  #
  # It is a weaker control than the flag was — `--trace` also writes a trace per layer —
  # but it does not touch the arithmetic, so the IDS comparison below still means what it
  # meant. If that ever stops being true this script has no control left and should be
  # retired rather than trusted.
  a=$(arm "$art" "$attn" "$WORK/tok.ids" "$WORK/tok.log" --trace "$WORK/tok.trace" $extra)
  # shellcheck disable=SC2086
  b=$(arm "$art" "$attn" "$WORK/lay.ids" "$WORK/lay.log" $extra)
  echo "  token-major: ${a:-<no PREFILL line — arm failed, see below>}"
  echo "  layer-major: ${b:-<no PREFILL line — arm failed, see below>}"
  if [ -z "$a" ] || [ -z "$b" ]; then
    echo "FAIL  $attn $label — an arm produced no PREFILL line:"
    # `-n 5`, not `-5`: the obsolete form is rejected outright when tail is handed more
    # than one file here, so the diagnostic this branch exists to print printed nothing.
    tail -n 5 "$WORK/tok.log" "$WORK/lay.log"
    FAILED=1
    return
  fi
  ra=$(rpt "$a"); rb=$(rpt "$b")
  if cmp -s "$WORK/tok.ids" "$WORK/lay.ids"; then
    echo "PASS  $attn $label — reordering is output-neutral ($NGEN token ids identical)"
  else
    echo "FAIL  $attn $label — the reordering CHANGED the output:"
    diff "$WORK/tok.ids" "$WORK/lay.ids" | head -10
    FAILED=1
  fi
  # The read reduction, asserted rather than admired. A build where the flag silently did
  # nothing would sail through the id comparison above — this is the half that catches it.
  if awk -v a="$ra" -v b="$rb" 'BEGIN{exit !(b < a)}'; then
    echo "PASS  $attn $label — reads/token $ra -> $rb ($(awk -v a="$ra" -v b="$rb" 'BEGIN{printf "%.2fx", a/b}') fewer)"
  else
    echo "FAIL  $attn $label — layer-major did not reduce reads/token ($ra -> $rb)"
    FAILED=1
  fi
}

# Symlinks + one edited manifest, so the dsa selector runs from token ~LOW_TOPK instead of
# token ~2048. Costs a directory, not a copy of 60 GB.
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
  # Only dsa/misa consult index_topk, and only they carry the cross-pass reuse state this
  # flag had to re-key; for dense/streaming the shadow arm would be the same run.
  case "$attn" in
    dsa | misa) pair "$(shadow)" "$attn" "(index_topk=$LOW_TOPK — the sel path actually runs)" ;;
  esac
done
exit $FAILED
