#!/usr/bin/env bash
# Every mode x cache-policy x attention cell, under BOTH backend features. A correctness
# sweep, not a benchmark: short cells, full cross, and the only question is whether each
# combination still behaves — decodes, or refuses for a reason the engine states.
#
# tests/bench-matrix.sh covers the same three dimensions but ranks throughput, so its cells
# are long and its rounds are curated by a human. That makes it the wrong instrument for
# "did anything break": too slow to run after a change, and a combination that CRASHES is
# indistinguishable from one that simply was not selected this round.
#
# THERE IS NO HAND-WRITTEN LIST OF WHAT VULKAN SUPPORTS, deliberately. Vulkan ports 16 of 29
# kernels, so most cells cannot run there — and a declared support list is exactly the thing
# that drifts (bench-matrix.sh enumerated `top-m` for months after the policy was deleted).
# Instead every cell RUNS and the outcome is classified. `Config::validate_backend` refuses
# the unported combinations at startup with a message naming the backend, before the pin is
# built, so those cells cost seconds and are reported as `refused` rather than as failures.
# A newly ported kernel therefore flips a cell refused -> ok with nothing to update here.
#
# What that cannot tell you is a guard that STOPPED firing (refused -> ok looks like a port).
# `config.rs::vulkan_refuses_the_unported_modes` pins the guard itself; the two are
# complementary and neither replaces the other.
#
# Usage: tests/mode-matrix.sh [artifact] [ngen] [backend ...]
#   RIVOLI_MAX_MEM=115  RIVOLI_TIMEOUT=900  RIVOLI_GPU_LOCK=/var/run/sys-gpu.lock
#
# COST: 36 cells per backend, each a fresh process. Under rocm all 36 decode (~130 s of
# artifact load each, so ~90 min); under vulkan 6 decode and 30 refuse in seconds, ~15 min.
# A pre-release gate, not a per-commit one. One process per cell is deliberate: a shared one
# would leave the expert pool warm for whichever cell ran second.
#
# This sweep does NOT compare timings between cells, which is what makes it safe to take the
# lock per cell and to rebuild between backends — neither would be acceptable in a benchmark.
set -uo pipefail

ART="${1:-/var/db/rivoli/glm52-vq3-full}"
NGEN="${2:-8}"
shift 2 2>/dev/null || shift $#
BIN=./target/release/rivoli
MEM="${RIVOLI_MAX_MEM:-115}"
TIMEOUT="${RIVOLI_TIMEOUT:-900}"
# How long to WAIT for the shared GPU, which is a different quantity from how long a cell
# may run: other tenants' benchmarks routinely hold this machine for half an hour.
LOCK_WAIT="${RIVOLI_LOCK_WAIT:-5400}"
GPU_LOCK="${RIVOLI_GPU_LOCK:-/var/run/sys-gpu.lock}"

# Declared as arrays because tests/matrix.rs parses THESE LINES and compares them with what
# the CLI accepts and with Cargo.toml's backend features. Editing a loop without editing
# these is a test failure, which is the point.
BACKENDS=(rocm vulkan)
MODES=(int3-vq int4 hybrid)
POLICIES=(lru 2q arc)
# `auto` is excluded deliberately: it resolves to dense or dsa at startup, so a cell for it
# would duplicate whichever it picked while hiding which one that was.
ATTNS=(dense streaming dsa misa)

[ $# -gt 0 ] && BACKENDS=("$@")

# A prompt that stays coherent on this checkpoint. The default ("The sky is blue because")
# trips the degeneration warning, which is a different regime and would flag every cell.
PROMPT="${RIVOLI_PROMPT:-Explain how a CPU cache hierarchy works, and why it exists.}"

LOG=$(mktemp -d); trap 'rm -rf "$LOG"' EXIT
TOTAL_FAIL=0

for backend in "${BACKENDS[@]}"; do
  echo "===== backend: $backend ====="
  # OUTSIDE the lock: a compile needs no GPU and holding the lock through one starves
  # every other tenant on this machine.
  if ! cargo build --release --quiet --features "$backend" 2>"$LOG/build.log"; then
    echo "FAIL  $backend — build failed:"; grep -m3 -E "^error" "$LOG/build.log" | sed 's/^/        /'
    TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue
  fi

  PASS=0; REFUSED=0; FAIL=0; FAILED=(); REFUSALS=()
  for mode in "${MODES[@]}"; do
    for attn in "${ATTNS[@]}"; do
      for pol in "${POLICIES[@]}"; do
        cell="$mode/$attn/$pol"
        out="$LOG/$backend-$mode-$attn-$pol.log"
        # `-E 66`: without it, FAILING TO ACQUIRE THE LOCK exits 1 and prints nothing, which
        # is indistinguishable from the binary exiting 1 — so a busy machine was reported as
        # `CRASH rc=1` with no diagnostic, and the first cell of a run is the one most likely
        # to hit it. `-w` is the lock WAIT and is deliberately not the run timeout: on a
        # shared GPU the queue can legitimately be longer than any single cell.
        flock -E 66 -w "$LOCK_WAIT" "$GPU_LOCK" \
          timeout -k 30 "$TIMEOUT" "$BIN" "$ART" -bench "$NGEN" \
          --mode "$mode" --attn "$attn" --cache-policy "$pol" \
          --max-mem "$MEM" --prompt "$PROMPT" >"$out" 2>&1
        rc=$?
        if [ $rc -eq 66 ] && [ ! -s "$out" ]; then
          # Never ran. Not a verdict about the cell, so do not manufacture one.
          FAIL=$((FAIL + 1)); FAILED+=("$cell — NO GPU (lock busy > ${LOCK_WAIT}s)")
          printf 'FAIL     %-26s NO GPU — lock busy >%ss, cell never ran\n' "$cell" "$LOCK_WAIT"
        elif grep -aq "tok/s" "$out"; then
          PASS=$((PASS + 1))
          printf 'ok       %-26s %s\n' "$cell" "$(grep -ao "[0-9.]* tok/s" "$out" | tail -1)"
        elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
          FAIL=$((FAIL + 1)); FAILED+=("$cell — TIMEOUT ${TIMEOUT}s")
          printf 'FAIL     %-26s TIMEOUT\n' "$cell"
        elif [ $rc -ne 0 ] && grep -aqiE "Vulkan backend|not implemented on the Vulkan" "$out"; then
          # A stated refusal, not a breakage: the engine says which kernel is missing.
          REFUSED=$((REFUSED + 1)); REFUSALS+=("$cell")
          printf 'refused  %-26s (unported on %s)\n' "$cell" "$backend"
        elif [ $rc -ne 0 ]; then
          FAIL=$((FAIL + 1)); FAILED+=("$cell — CRASH rc=$rc")
          printf 'FAIL     %-26s CRASH rc=%s\n' "$cell" "$rc"
          # The known shapes first; then the tail REGARDLESS, because a crash whose output
          # matches none of these patterns is the one worth seeing and the grep alone
          # printed nothing at all for it.
          if ! grep -aiE "^error|panicked|guard rejected|not resident|non-finite" "$out" |
            head -3 | sed 's/^/           /' | grep -q .; then
            tail -n 3 "$out" | sed 's/^/           | /'
          fi
        else
          # Exited 0 without ever reporting throughput: it did not decode and did not say why.
          FAIL=$((FAIL + 1)); FAILED+=("$cell — NO OUTPUT rc=0")
          printf 'FAIL     %-26s NO OUTPUT (exit 0, never decoded)\n' "$cell"
        fi
      done
    done
  done

  echo "----- $backend: $PASS decoded, $REFUSED refused, $FAIL failed of $((PASS + REFUSED + FAIL))"
  [ "$REFUSED" -gt 0 ] && printf '      refused: %s\n' "${REFUSALS[*]}"
  if [ "$FAIL" -gt 0 ]; then
    printf '      FAILED: %s\n' "${FAILED[@]}"
    TOTAL_FAIL=$((TOTAL_FAIL + FAIL))
  fi
  echo
done

[ "$TOTAL_FAIL" -eq 0 ] || { echo "$TOTAL_FAIL cell(s) failed"; exit 1; }
echo "all backends clean"
