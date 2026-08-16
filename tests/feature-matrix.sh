#!/usr/bin/env bash
# Every feature combination must COMPILE and pass the tests it gates.
#
# This exists because the repo's documented recurring failure is a feature-gated module
# rotting unnoticed: `mod otlp` sat broken for weeks on an E0609 — a `ProfileSummary` field
# renamed out from under it — while every prescribed build command passed, because none of
# them named the feature. A feature is checked exactly as often as someone remembers it.
#
# The dimensions are declared as arrays and `tests/matrix.rs` asserts they match
# `Cargo.toml`'s `[features]`, so adding a feature and forgetting this file is a test
# failure rather than a hole that opens quietly.
#
# WHY THIS IS `cargo check` AND NOT `cargo test`, and why the two matrices are not nested:
# the mode x policy x attn matrix (tests/mode-matrix.sh) needs a GPU and ~145 s per cell.
# Running it under all 33 feature combinations would be 1188 GPU-bound cells, roughly two
# days of sole-tenant machine, to re-prove decode behaviour that features do not change —
# they gate instruments, not arithmetic. So: features are checked for COMPILATION here
# (plus the non-GPU tests, which is where feature-gated logic actually has assertions), and
# decode behaviour is checked once, on the union build, there.
#
# WORKSPACE ADAPTATION (2026-08-15): a virtual workspace refuses bare `--features` at the
# root, so every feature cell targets `-p rivoli` (the leaf), whose [features] forward down
# the DAG; the featureless cell checks the whole workspace, which is what CI's host job does.
#
# Usage: tests/feature-matrix.sh [quick|full]        # default: full
#   quick — backends x {none, each single feature, all}: catches a module that stopped
#           compiling. ~7 combos per backend.
#   full  — the whole powerset of the optional features per backend. ~33 combos, and the
#           only tier that can catch a PAIR of features interacting.
#   RIVOLI_BACKENDS="rocm" narrows the backends. There is only `rocm` since 2026-08-06, so
#           the knob is vestigial today; it is kept because BACKENDS is what tests/matrix.rs
#           asserts against Cargo.toml, and a second backend would want it back.
set -uo pipefail

MODE="${1:-full}"

# Mutually exclusive — one backend per build, no runtime selection. Never both in one cell.
BACKENDS=(rocm)
# The optional, non-backend features. `default` is empty and is not a cell.
OPTIONAL=(otlp teacher-forcing pred-probe trace stale-sel)

read -r -a BACKENDS <<<"${RIVOLI_BACKENDS:-${BACKENDS[*]}}"

# Non-GPU test targets. The rest of the suite needs the device and belongs behind the lock;
# these carry the assertions that feature-gated code actually has.
CHEAP_TESTS=(--test docs --test invariants)

combos() { # combos -> one space-separated feature subset per line ("" = none)
  local n=${#OPTIONAL[@]} i j sub
  if [ "$MODE" = quick ]; then
    printf '\n'                                  # none
    for f in "${OPTIONAL[@]}"; do printf '%s\n' "$f"; done
    printf '%s\n' "${OPTIONAL[*]}"               # all
    return
  fi
  # Full powerset: bit i of the counter selects OPTIONAL[i].
  for ((i = 0; i < (1 << n); i++)); do
    sub=""
    for ((j = 0; j < n; j++)); do
      (((i >> j) & 1)) && sub="$sub ${OPTIONAL[j]}"
    done
    printf '%s\n' "${sub# }"
  done
}

PASS=0; FAIL=0; FAILED=()
run() { # run <label> <cargo args...>
  local label=$1; shift
  if "$@" >/tmp/fm.$$ 2>&1; then
    PASS=$((PASS + 1)); printf 'ok    %s\n' "$label"
  else
    FAIL=$((FAIL + 1)); FAILED+=("$label"); printf 'FAIL  %s\n' "$label"
    # The first error only: a feature-gated break is usually one type error repeated.
    grep -m3 -E "^error" /tmp/fm.$$ | sed 's/^/        /'
  fi
  rm -f /tmp/fm.$$
}

echo "feature matrix ($MODE): backends=${BACKENDS[*]} optional=${OPTIONAL[*]}"
echo

# The featureless build FIRST, and it is not a formality: with no backend the crate compiles
# to a refusal stub, which is deliberate and has broken before when a module forgot its gate.
#
# `-D warnings` and `--all-targets` because this is the ONE cell that stands in for CI's
# `host` job, and without them it could not: `dead_code` is a warning, so this cell passed
# green from 243d438 to 68e83b3 over four ungated `BENCH_*` constants in main.rs that CI
# rejected outright. A gate that cannot go red is not a gate. Proven both ways before this
# line was written — un-gate one of those constants and this cell fails.
#
# The backend cells below deliberately keep the looser bar: no CI job builds a backend at
# all, so there is no bar to match them to, and raising it here would be an unreviewed
# change to a build nothing else checks.
# Every cell pins its EXACT feature set with --no-default-features: since the fuse (owner
# rule 2026-08-16) `default = ["rocm"]`, so without the pin every cell would silently
# include the backend and the deviceless half of the matrix would collapse into one build.
#
# The resolve check FIRST, and it is not a formality: --no-default-features strips only the
# TOP crate's defaults — an inter-crate dep edge without `default-features = false` re-arms
# rocm through feature unification, and on this box (hipcc present) every build cell below
# stays green over it. Found live 2026-08-16: an M9 agent's "deviceless" workspace run
# executed device binaries unlocked. A compile cell cannot see the leak; the resolve can.
# The non-empty guard is load-bearing: a failed `cargo tree` (stderr discarded) yields
# empty stdout, grep exits 1, and the bare negation would turn the cell GREEN on a broken
# command — the eaten-exit false-green class, found by review 2026-08-16 in the very cell
# that exists to catch the last false green.
run "(no features) — rocm absent from the resolve" \
  bash -c 'out=$(cargo tree -e features --no-default-features -p rivoli 2>/dev/null); [ -n "$out" ] && ! printf "%s" "$out" | grep -q "rocm"'
run "(no features) — refusal stub" \
  env RUSTFLAGS="-D warnings" cargo check --release --quiet --workspace --all-targets --no-default-features

while IFS= read -r sub; do
  for b in "${BACKENDS[@]}"; do
    feats="$b${sub:+,${sub// /,}}"
    run "check  --features $feats" cargo check --release --quiet -p rivoli --all-targets --no-default-features --features "$feats"
  done
done < <(combos)

# The union is what CLAUDE.md prescribes before claiming a change compiles; assert the
# cheap tests pass under it too, not just that it type-checks.
UNION="${BACKENDS[0]},$(IFS=,; echo "${OPTIONAL[*]}")"
run "test   --features $UNION ${CHEAP_TESTS[*]}" \
  cargo test --release --quiet -p rivoli --no-default-features --features "$UNION" "${CHEAP_TESTS[@]}"

echo
echo "----- $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
  printf 'failed: %s\n' "${FAILED[@]}"
  exit 1
fi
