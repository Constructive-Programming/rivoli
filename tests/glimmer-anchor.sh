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
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/glimmer-anchor
PY=$VENV/bin/python
DRIVER=$ROOT/tests/glimmer_anchor_driver.py

# Every defect the driver declares, split by the mode it applies to. Derived from the driver rather
# than written here: a defect added there and forgotten here would be a defect nothing ever proves
# reddens, which is the failure this whole file exists to prevent.
mapfile -t TEXT_DEFECTS < <("$PY" -c "
import sys; sys.path.insert(0, '$ROOT/tests')
from glimmer_anchor_driver import TEXT_DEFECTS as d
print('\n'.join(sorted(d)))")
mapfile -t DRAFT_DEFECTS < <("$PY" -c "
import sys; sys.path.insert(0, '$ROOT/tests')
from glimmer_anchor_driver import DRAFT_DEFECTS as d
print('\n'.join(sorted(d)))")

SALTS=(glimmer-anchor-1 glimmer-anchor-2)

mkdir -p "$OUT"
echo "== regenerating into $OUT"
fail=0

for salt in "${SALTS[@]}"; do
    n=${salt##*-}
    for mode in text draft; do
        if [ "$mode" = text ]; then defects=("${TEXT_DEFECTS[@]}"); else defects=("${DRAFT_DEFECTS[@]}"); fi
        for defect in "${defects[@]}"; do
            out=$OUT/$mode-$n-$defect.bin
            # `--no-preflight` only on the clean run of salt 1, which is the file preflight would
            # compare against: on a deliberate re-pin the vendored bytes are stale by definition and
            # the check would refuse the very run that replaces them. Every other run keeps it.
            pf=""
            if [ "$defect" = None ] && [ "$n" = 1 ] && [ "$mode" = text ]; then pf=--no-preflight; fi
            "$PY" "$DRIVER" --mode "$mode" --salt "$salt" --defect "$defect" --out "$out" $pf >/dev/null
        done

        clean=$OUT/$mode-$n-None.bin
        vendored=$ROOT/tests/glimmer-anchor-$mode-$n.bin
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
    if cmp -s "$ROOT/tests/glimmer-anchor-$mode-1.bin" "$ROOT/tests/glimmer-anchor-$mode-2.bin"; then
        echo "  $mode: THE TWO SALTS ARE IDENTICAL — the salt is not reaching the weights"
        fail=1
    fi
done

[ $fail -eq 0 ] && echo "== all defect runs reddened where they should and held where they should"
exit $fail
