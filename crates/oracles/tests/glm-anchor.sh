#!/usr/bin/env bash
# **Regenerate the GLM-5.2 anchor goldens and their defect runs.**
# `docs/measurement/glm-reference/anchor.md` is the record this writes;
# `crates/oracles/tests/glm_anchor_driver.py` is what it runs.
#
# NOT a `cargo test`. It needs a pinned python environment. **It does NOT need a GPU** — the
# reference is transformers' own glm_moe_dsa with a CPU path for every operator, so this takes
# no GPU lock and can run beside a benchmark. The goldens it produces are vendored; reading
# them needs neither python nor a device (`crates/oracles/tests/glm_anchor.rs`).
#
#     GLM_ANCHOR_VENV=/home/rhansen/glm-anchor/venv crates/oracles/tests/glm-anchor.sh
#
# **Configured by env var, and that is not the thing CLAUDE.md forbids.** That rule
# ("instruments go behind a feature AND a flag, never an env var") gives three reasons, all
# about the engine binary: an env var is invisible to `--help`, absent from the recorded
# command line, and silently active in a build that looks stock. None applies to a script that
# is not a cargo run, has no `--help`, and whose invocation is recorded on the line above and
# in `anchor.md`. The sibling anchor scripts carry the same exemption with the same argument.
#
# The env is checked ONCE, up front, against a vendored golden's recorded pins — and
# `GLM_ANCHOR_REPIN=1` skips the check entirely, because on a deliberate re-pin the vendored
# bytes are stale BY DEFINITION (the glimmer script's lesson: per-cell preflight exempted one
# cell of the matrix and left the rest to refuse the very run that would replace them).
set -euo pipefail

VENV=${GLM_ANCHOR_VENV:?set GLM_ANCHOR_VENV to the venv holding torch+transformers}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)   # workspace root; this script lives in crates/oracles/tests/
TESTS=$ROOT/crates/oracles/tests
OUT=$ROOT/target/glm-anchor
PY=$VENV/bin/python
DRIVER=$TESTS/glm_anchor_driver.py

# Derived from the driver rather than written here: a defect added there and forgotten here
# would be a defect nothing ever proves reddens.
mapfile -t DEFECTS < <("$PY" -c "
import sys; sys.path.insert(0, '$TESTS')
from glm_anchor_driver import DEFECTS as d
print('\n'.join(sorted(k for k in d if k != 'None')))")

# Anti-vacuity: mapfile hides the producer's exit status from set -e, so an import
# error would otherwise yield an empty list and a vacuous "all reddened" (review
# 2026-08-15 — the guard was accidental before, now it is designed).
[ "${#DEFECTS[@]}" -ge 10 ] || { echo "derived only ${#DEFECTS[@]} defects — the driver import failed"; exit 1; }

SALTS=(glm-anchor-1 glm-anchor-2)

mkdir -p "$OUT"

if [ -n "${GLM_ANCHOR_REPIN:-}" ]; then
    echo "== GLM_ANCHOR_REPIN set: not checking this env against the vendored goldens"
else
    "$PY" "$DRIVER" --preflight-against "$TESTS/glm-anchor-1.bin" \
        --out /dev/null --defect None >/dev/null 2>&1 || {
        "$PY" -c "
import sys; sys.path.insert(0, '$TESTS')
from glm_anchor_driver import preflight_env
preflight_env('$TESTS/glm-anchor-1.bin')"
    }
fi

echo "== regenerating into $OUT"
fail=0

for salt in "${SALTS[@]}"; do
    n=${salt##*-}
    clean=$OUT/clean-$n.bin
    "$PY" "$DRIVER" --salt "$salt" --out "$clean" --no-preflight >/dev/null

    vendored=$TESTS/glm-anchor-$n.bin
    if cmp -s "$clean" "$vendored"; then
        echo "  $salt  reproduces the vendored bytes"
    else
        echo "  $salt  DIFFERS from $vendored — find out why it moved before re-vendoring"
        fail=1
    fi

    for defect in "${DEFECTS[@]}"; do
        "$PY" "$DRIVER" --salt "$salt" --defect "$defect" --out "$OUT/d-$n-$defect.bin" \
            --no-preflight >/dev/null
        printf '  %-26s ' "$salt $defect"
        if "$PY" "$DRIVER" --compare "$clean" "$OUT/d-$n-$defect.bin" 2>&1 | tr '\n' ' '; then
            echo
        else
            echo "  <== FAILED"
            fail=1
        fi
    done
done

# Two salts are coverage, not redundancy: byte-identical salts mean the salt never reached
# the weights and every "both salts" claim in anchor.md is one claim.
if cmp -s "$TESTS/glm-anchor-1.bin" "$TESTS/glm-anchor-2.bin"; then
    echo "  THE TWO SALTS ARE IDENTICAL — the salt is not reaching the weights"
    fail=1
fi

[ $fail -eq 0 ] && echo "== all defect runs reddened where they should and held where they should"
exit $fail
