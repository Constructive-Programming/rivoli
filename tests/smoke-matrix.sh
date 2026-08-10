#!/usr/bin/env bash
# Both models, every user-selectable setting, ~12 tokens each: does the engine still decode,
# and does it still refuse what it documents refusing? A smoke gate, not a benchmark.
#
# Born 2026-08-10 on the owner's requirement: "we test both models and the multiple
# settings (even if just for 12 tokens) to exercise and make sure we're still working with
# no regressions."
#
# HOW IT RELATES TO tests/mode-matrix.sh: that sweep is GLM-only, 36 cells total
# (3 modes x 4 attn x 3 cache policies), ~90 min — a pre-release gate. This one is both models, one cache policy,
# 15 cells, and exists to answer "did this change break decode anywhere" in tens of
# minutes. The classification rules are mode-matrix.sh's (a refusal on a GLM cell is a
# CRASH, not a category), plus one category that sweep retired and this one needs:
# REFUSED-AS-DOCUMENTED, for the V4 cells that assert a documented refusal actually
# fires. Asserting the refusal is the point — a V4 tree that silently ACCEPTED --attn
# would be a regression this gate must catch, so those cells are run, not skipped.
#
# WHAT A 12-TOKEN CELL DOES AND DOES NOT COVER:
#   * it exercises artifact load, prefill, decode, and the routed-expert path (every decode
#     step routes top-8, so experts stream from the first token);
#   * the `dsa` cells run the selector's DENSE FAST PATH, not the top-k path — the selector
#     only engages past index_topk tokens (docs: "A dsa A/B under 2048 tokens covers
#     nothing"). Accepted for a smoke gate; do not read those cells as dsa coverage.
#   * `hybrid` output is residency-dependent (architecture.md §8b under INV-1), so no cell
#     compares text across settings — every cell asserts only: decoded, exit 0, sane output.
#
# Usage: tests/smoke-matrix.sh
#   RIVOLI_GLM_ART=/var/db/rivoli/glm52-vq3-full   RIVOLI_V4_ART=/var/db/rivoli/v4-f4-full
#   RIVOLI_MAX_MEM=115 (GLM)  RIVOLI_MAX_MEM_V4=100  RIVOLI_NGEN=12  RIVOLI_TIMEOUT=900
#   Red-proof: `RIVOLI_MAX_MEM=1 RIVOLI_MAX_MEM_V4=1 tests/smoke-matrix.sh` must FAIL every
#     decode cell (budget below one layer) while the refusal cells stay green — prove the
#     gate can go red before trusting it green. (An earlier draft carried a sabotage env var
#     that APPENDED a second --max-mem; clap rejects duplicates, so it would have proven
#     clap's arm rather than the engine's. The existing knob does it right for free.)
#
# Artifact paths are the ones benchmarks.md records (7x v4-f4-full, 6x glm52-vq3-full).
# The GPU lock protocol is mode-matrix.sh's: flock -E 66 so "lock busy" is distinguishable
# from "binary exited 1", build OUTSIDE the lock, one fresh process per cell so no cell
# inherits a warm expert pool.
set -uo pipefail

GLM_ART="${RIVOLI_GLM_ART:-/var/db/rivoli/glm52-vq3-full}"
V4_ART="${RIVOLI_V4_ART:-/var/db/rivoli/v4-f4-full}"
# RIVOLI_BIN exists so the CLASSIFIER is testable without a device: point it at a stub that
# prints "tok/s", refuses, or emits a poison marker, and every verdict arm below can be
# driven host-side (and was, before this script's first real run). Overriding it skips the
# build — a stub needs none.
BIN="${RIVOLI_BIN:-./target/release/rivoli}"
NGEN="${RIVOLI_NGEN:-12}"
MEM_GLM="${RIVOLI_MAX_MEM:-115}"
MEM_V4="${RIVOLI_MAX_MEM_V4:-100}"
TIMEOUT="${RIVOLI_TIMEOUT:-900}"
LOCK_WAIT="${RIVOLI_LOCK_WAIT:-5400}"
GPU_LOCK="${RIVOLI_GPU_LOCK:-/var/run/sys-gpu.lock}"
# mode-matrix.sh's prompt, for its recorded reason: the default prompt trips the
# degeneration warning on this checkpoint, which would flag every cell.
PROMPT="${RIVOLI_PROMPT:-Explain how a CPU cache hierarchy works, and why it exists.}"

# GLM's settings, verbatim from the engine (src/main.rs's value_parser and
# arch.rs::attn_modes). `auto` excluded for mode-matrix.sh's reason: it resolves to dense
# or dsa at startup, so its cell duplicates one of these while hiding which.
GLM_MODES=(int3-vq int4 hybrid)
GLM_ATTNS=(dense streaming dsa misa)

LOG=$(mktemp -d); trap 'rm -rf "$LOG"' EXIT
declare -a ROWS; FAILN=0

# One cell: run under the lock, classify, append a table row.
#   cell <id> <artifact> <max-mem> <expect> [extra flags...]
# expect = "decode"  -> PASS iff rc=0, tok/s reported, no poison marker in the log
# expect = <pattern> -> REFUSED-AS-DOCUMENTED iff rc!=0 AND the log matches the pattern;
#                       anything else (including a successful decode!) is FAIL.
cell() {
  local id="$1" art="$2" mem="$3" expect="$4"; shift 4
  local out="$LOG/${id//\//-}.log" verdict detail rc
  local -a extra=("$@")
  flock -E 66 -w "$LOCK_WAIT" "$GPU_LOCK" \
    timeout -k 30 "$TIMEOUT" "$BIN" "$art" -bench "$NGEN" \
    --max-mem "$mem" --prompt "$PROMPT" "${extra[@]}" >"$out" 2>&1
  rc=$?
  if [ $rc -eq 66 ] && [ ! -s "$out" ]; then
    verdict=FAIL; detail="NO GPU — lock busy >${LOCK_WAIT}s, never ran"
  elif [ "$expect" = decode ]; then
    # Poison first: a decode that reported throughput AND flagged non-finite values is a
    # failure, and checking the witness first would classify it PASS.
    if grep -aqiE 'panicked|non-finite|NaN detected|degenerat' "$out"; then
      verdict=FAIL; detail="poison marker: $(grep -aioE 'panicked|non-finite|NaN detected|degenerat[a-z]*' "$out" | head -1)"
    # Decode witness is PER MODEL: the GLM bench epilogue prints `N tok/s`; run_v4 never
    # does — its always-on witness is the `PROFILE/tok:` line (f4gpu.rs). Requiring tok/s
    # alone made the v4/default cell permanently red on a healthy tree (review, 2026-08-10).
    elif [ $rc -eq 0 ] && grep -aqE 'tok/s|PROFILE/tok:' "$out"; then
      verdict=PASS
      detail="$(grep -aoE '[0-9.]+ tok/s' "$out" | tail -1)"
      [ -n "$detail" ] || detail="$(grep -aoE 'PROFILE/tok: [0-9.]+ms wall' "$out" | tail -1)"
    elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
      verdict=FAIL; detail="TIMEOUT ${TIMEOUT}s"
    else
      # NOT `grep ... | head -1 || tail -1`: head exits 0 on empty input, so the fallback
      # never ran and the diagnostic collapsed to a bare rc — the `| tail eats the exit
      # code` trap, caught by review before this script's first device run.
      verdict=FAIL
      detail="$(grep -aiE '^error|panicked' "$out" | head -1)"
      [ -n "$detail" ] || detail="$(tail -1 "$out")"
      detail="rc=$rc $detail"
    fi
  else
    # `-e` because the pattern itself may start with `--` (the --attn refusal does), and
    # bare `grep -aq "$expect"` reads it as options. Found by the stub self-test, arm 2 —
    # the exact class of bug this script exists to catch in the engine.
    if [ $rc -ne 0 ] && grep -aqe "$expect" "$out"; then
      verdict=REFUSED-AS-DOCUMENTED; detail="refused with the documented reason"
    elif [ $rc -eq 0 ]; then
      verdict=FAIL; detail="ACCEPTED a setting the engine documents refusing"
    else
      verdict=FAIL; detail="refused rc=$rc but NOT with the documented message: $(tail -1 "$out")"
    fi
  fi
  [ "$verdict" = FAIL ] && FAILN=$((FAILN + 1))
  ROWS+=("$(printf '%-28s %-22s %s' "$id" "$verdict" "$detail")")
  printf '%-28s %-22s %s\n' "$id" "$verdict" "$detail"
}

# Build OUTSIDE the lock (mode-matrix.sh's rule: a compile needs no GPU and holding the
# lock through one starves every other tenant). Skipped under RIVOLI_BIN: a stub needs none.
if [ -z "${RIVOLI_BIN:-}" ]; then
  if ! cargo build --release --quiet --features rocm 2>"$LOG/build.log"; then
    echo "FAIL — build failed:"; grep -m3 -E '^error' "$LOG/build.log" | sed 's/^/    /'
    exit 1
  fi
fi

echo "== GLM (${GLM_ART}): ${#GLM_MODES[@]} modes x ${#GLM_ATTNS[@]} attn, ${NGEN} tokens, --cache-policy lru --max-mem ${MEM_GLM} =="
for mode in "${GLM_MODES[@]}"; do
  for attn in "${GLM_ATTNS[@]}"; do
    cell "glm/$mode/$attn" "$GLM_ART" "$MEM_GLM" decode \
      --mode "$mode" --attn "$attn" --cache-policy lru
  done
done

echo "== V4 (${V4_ART}): 1 decode + 2 documented refusals, --max-mem ${MEM_V4} =="
cell "v4/default" "$V4_ART" "$MEM_V4" decode
# The two refusal messages are main.rs's, quoted distinctively enough to survive rewording
# around them: --attn is refused because arch.rs::attn_modes returns None for V4, and
# --mode because a .f4 artifact has no second routed-expert format to pick.
cell "v4/refuses-attn" "$V4_ART" "$MEM_V4" '--attn does not apply' --attn dense
cell "v4/refuses-mode" "$V4_ART" "$MEM_V4" 'selects a GLM routed-expert format' --mode int4

echo
echo "== model x mode x attn -> verdict =="
printf '%s\n' "${ROWS[@]}"
if [ "$FAILN" -gt 0 ]; then echo "$FAILN cell(s) FAILED"; exit 1; fi
echo "all $((${#ROWS[@]})) cells clean"
