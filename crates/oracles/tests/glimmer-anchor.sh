#!/usr/bin/env bash
# **Regenerate the Muse Glimmer S1b anchor goldens and their defect runs.**
# `docs/measurement/glimmer-reference/anchor.md` is the record this writes;
# `tests/glimmer_anchor_driver.py` is what it runs.
#
# NOT a `cargo test`. It needs a pinned python environment. **It does NOT need a GPU** — unlike
# K3's anchor, whose KDA ops are triton kernels with no CPU path, this reference is plain PyTorch
# and the venv installs CPU torch on purpose. So this takes no GPU lock and can run beside a
# benchmark. The goldens it produces are vendored; reading them needs neither python nor a device.
#
#     GLIMMER_ANCHOR_VENV=/home/rhansen/glimmer-anchor/venv tests/glimmer-anchor.sh
#
# **Configured by env var, and that is not the thing CLAUDE.md forbids.** That rule ("instruments
# go behind a feature AND a flag, never an env var") gives three reasons, all about the engine
# binary: an env var is invisible to `--help`, absent from the recorded command line in
# `benchmarks.md`, and silently active in a build that looks stock. None applies to a script that
# is not a cargo run, has no `--help`, and whose invocation is recorded on the line above and in
# `anchor.md`. `build.rs`, `tests/k3-anchor.sh` and `tests/common/f4_artifact_dir.rs` each carry
# the same exemption with the same argument. `${VAR:?}` makes a missing one fail loudly.
#
# Writes `target/glimmer-anchor/<mode>-<salt>-<defect>.bin`, prints the reddening matrix, and
# `cmp`s each fresh `None` golden against its vendored twin — which is the reproducibility claim
# `anchor.md` makes, checked rather than asserted.
set -euo pipefail

VENV=${GLIMMER_ANCHOR_VENV:?set GLIMMER_ANCHOR_VENV to the venv holding torch+transformers}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)   # workspace root; this script lives in crates/oracles/tests/
TESTS=$ROOT/crates/oracles/tests
OUT=$ROOT/target/glimmer-anchor
PY=$VENV/bin/python
DRIVER=$TESTS/glimmer_anchor_driver.py

# Every defect the driver declares, split by the mode it applies to. Derived from the driver rather
# than written here: a defect added there and forgotten here would be a defect nothing ever proves
# reddens, which is the failure this whole file exists to prevent.
mapfile -t TEXT_DEFECTS < <("$PY" -c "
import sys; sys.path.insert(0, '$TESTS')
from glimmer_anchor_driver import TEXT_DEFECTS as d
print('\n'.join(sorted(d)))")
mapfile -t DRAFT_DEFECTS < <("$PY" -c "
import sys; sys.path.insert(0, '$TESTS')
from glimmer_anchor_driver import DRAFT_DEFECTS as d
print('\n'.join(sorted(d)))")

SALTS=(glimmer-anchor-1 glimmer-anchor-2)

mkdir -p "$OUT"

# **The env is checked ONCE, here, and not on every one of the 28 runs.**
#
# The driver's own `preflight_env` compares this interpreter against the versions recorded in a
# vendored golden, and that is right for a bare invocation. Per-run it is wrong twice over: it is
# 28 identical checks, and on a deliberate re-pin — where the vendored bytes are stale BY
# DEFINITION, because replacing them is the point — it refuses the very run that would replace
# them. An earlier version exempted one cell of the matrix and left the other 27 to refuse, which
# is a recipe that works until the day you need it.
#
# `GLIMMER_ANCHOR_REPIN=1` skips the check entirely. Re-vendoring is a reviewed change, so it
# should take a deliberate word rather than happening because a run was spelled slightly
# differently.
if [ -n "${GLIMMER_ANCHOR_REPIN:-}" ]; then
    echo "== GLIMMER_ANCHOR_REPIN set: not checking this env against the vendored goldens"
else
    "$PY" -c "
import sys; sys.path.insert(0, '$TESTS')
from glimmer_anchor_driver import preflight_env
preflight_env()"
fi

echo "== regenerating into $OUT"
fail=0

for salt in "${SALTS[@]}"; do
    n=${salt##*-}
    for mode in text draft; do
        if [ "$mode" = text ]; then defects=("${TEXT_DEFECTS[@]}"); else defects=("${DRAFT_DEFECTS[@]}"); fi
        for defect in "${defects[@]}"; do
            out=$OUT/$mode-$n-$defect.bin
            # The env was checked once above, so every run here skips it.
            "$PY" "$DRIVER" --mode "$mode" --salt "$salt" --defect "$defect" --out "$out" \
                --no-preflight >/dev/null
        done

        # The weight sets ride the clean text run: --dump-weights adds nothing to the golden, so
        # the cmp below still proves the golden reproduces, and the second cmp proves the weights do.
        # They were vendored with no regeneration path at all until review caught it (2026-08-13).
        if [ "$mode" = text ]; then
            "$PY" "$DRIVER" --mode text --salt "$salt" --defect None --out "$OUT/w-$n.bin" \
                --dump-weights "$OUT/weights-$n.bin" --no-preflight >/dev/null
            if cmp -s "$OUT/weights-$n.bin" "$TESTS/glimmer-anchor-weights-$n.bin"; then
                echo "  weights-$n  reproduces the vendored bytes"
            else
                echo "  weights-$n  DIFFERS — find out why it moved before re-vendoring"
                fail=1
            fi
        fi

        clean=$OUT/$mode-$n-None.bin
        vendored=$TESTS/glimmer-anchor-$mode-$n.bin
        if cmp -s "$clean" "$vendored"; then
            echo "  $mode-$n  reproduces the vendored bytes"
        else
            echo "  $mode-$n  DIFFERS from $vendored — find out why it moved before re-vendoring"
            fail=1
        fi

        for defect in "${defects[@]}"; do
            [ "$defect" = None ] && continue
            printf '  %-28s ' "$mode-$n $defect"
            if "$PY" "$DRIVER" --compare "$clean" "$OUT/$mode-$n-$defect.bin" 2>&1 | tr '\n' ' '; then
                echo
            else
                echo "  <== FAILED"
                fail=1
            fi
        done
    done
done

# Two salts are coverage, not redundancy: a property that holds at one draw may be a fact about
# those numbers rather than about the arithmetic. If the two ever agree byte for byte, the salt is
# not reaching the weights and every "both salts" claim in anchor.md is one claim.
for mode in text draft; do
    if cmp -s "$TESTS/glimmer-anchor-$mode-1.bin" "$TESTS/glimmer-anchor-$mode-2.bin"; then
        echo "  $mode: THE TWO SALTS ARE IDENTICAL — the salt is not reaching the weights"
        fail=1
    fi
done

[ $fail -eq 0 ] && echo "== all defect runs reddened where they should and held where they should"
exit $fail
