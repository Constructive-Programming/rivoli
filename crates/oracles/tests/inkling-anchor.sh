#!/usr/bin/env bash
# Regenerate the S3 Inkling 2 anchor goldens, the defect matrix, and the tolerance floors.
# `docs/measurement/inkling2-reference/anchor.md` is the record; `inkling_anchor_driver.py` runs.
#
# The `decode` goldens are the CACHED step: the driver prefills 0..16 with `use_cache=True` and
# then steps one token against that cache. The `prefill` golden is the same prompt uncached. Both
# protocols are recorded in each golden's own metadata (`protocol`), so a reader never has to
# infer which body produced the bytes.
#
# NOT a cargo test: it needs the pinned venv (torch CPU + transformers @ d8f10a82) and takes
# minutes, so it stays out of the deviceless battery; `crates/oracles/tests/inkling_anchor.rs`
# reads what this vendors.
#
#     INKLING_ANCHOR_VENV=/var/cache/rivoli/scratch/inkling-venv crates/oracles/tests/inkling-anchor.sh
#
# No GPU, no flock: the short convolutions fall back to the reference PyTorch body when
# causal_conv1d is absent (the driver asserts it), which is the arithmetic this anchor pins.
# Configured by env var, which CLAUDE.md permits for a script that is not a cargo run.
set -euo pipefail

VENV=${INKLING_ANCHOR_VENV:?set INKLING_ANCHOR_VENV to the venv holding torch + transformers @ d8f10a82}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
TESTS=$ROOT/crates/oracles/tests
OUT=${INKLING_ANCHOR_OUT:-/var/cache/rivoli/scratch/inkling-anchor/out}
PY=$VENV/bin/python
DRIVER=$TESTS/inkling_anchor_driver.py
export PYTHONDONTWRITEBYTECODE=1   # /home is NFS; a stale .pyc can shadow an edited source

mapfile -t DEFECTS < <(cd "$TESTS" && "$PY" -c 'from inkling_anchor_driver import DEFECTS; print("\n".join(d for d in DEFECTS if d != "None"))')
SALTS=(${INKLING_ANCHOR_SALTS:-inkling-1 inkling-2})
mkdir -p "$OUT"
cd "$TESTS"

echo "=== defect rows: ${#DEFECTS[@]} ==="

# The None goldens, both salts, decode + prefill (prefill salt 1 only — the structure claim).
for salt in "${SALTS[@]}"; do
    "$PY" "$DRIVER" --mode decode --defect None --salt "$salt" --out "$OUT/gold-decode-$salt-None.bin"
done
"$PY" "$DRIVER" --mode prefill --defect None --salt inkling-1 --out "$OUT/gold-prefill-inkling-1-None.bin"

# The defect matrix: each defect vs the None golden of the same salt, both salts.
rc=0
for salt in "${SALTS[@]}"; do
    for def in "${DEFECTS[@]}"; do
        "$PY" "$DRIVER" --mode decode --defect "$def" --salt "$salt" --out "$OUT/gold-decode-$salt-$def.bin"
        if ! "$PY" "$DRIVER" --compare "$OUT/gold-decode-$salt-None.bin" "$OUT/gold-decode-$salt-$def.bin" --defect "$def"; then
            echo "MATRIX RED: $def $salt"; rc=1
        fi
    done
done

# The floors, both draws.
for salt in "${SALTS[@]}"; do
    echo "=== tolerance $salt ==="
    "$PY" "$DRIVER" --tolerance --salt "$salt"
done

# The vendored bytes are the point: a regeneration that no longer reproduces them is the thing
# most worth knowing. Enumerated, not globbed; the exit code carries the census.
VENDORED=(
    "inkling-anchor-decode-inkling-1.bin:$OUT/gold-decode-inkling-1-None.bin"
    "inkling-anchor-decode-inkling-2.bin:$OUT/gold-decode-inkling-2-None.bin"
    "inkling-anchor-prefill-inkling-1.bin:$OUT/gold-prefill-inkling-1-None.bin"
)
verified=0
for pair in "${VENDORED[@]}"; do
    v=$TESTS/${pair%%:*}; fresh=${pair#*:}
    if [[ ! -f $v ]]; then echo "MISSING vendored: $v"; rc=1; continue; fi
    if cmp -s "$v" "$fresh"; then verified=$((verified+1)); else echo "DIFFERS: $v"; rc=1; fi
done
echo "vendored goldens verified: $verified of ${#VENDORED[@]}"
if ((verified < ${#VENDORED[@]})); then rc=1; fi
exit $rc
