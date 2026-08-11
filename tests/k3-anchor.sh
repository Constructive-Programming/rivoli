#!/usr/bin/env bash
# **Regenerate the S1b anchor golden and its defect runs.** `docs/measurement/k3-reference/anchor.md`
# is the record this writes; `tests/k3_anchor_driver.py` is what it runs.
#
# NOT a `cargo test`. It needs a pinned python environment and a GPU — `chunk_kda` and
# `fused_recurrent_kda` are triton kernels with no CPU path — and a gate that needs a device on
# this machine blocks correctness work that never wanted one. The goldens it produces are
# vendored; reading them needs neither.
#
#     K3_ANCHOR_VENV=/home/rhansen/k3-anchor/venv K3_ANCHOR_REF=/home/rhansen/k3-anchor/ref \
#         tests/k3-anchor.sh                       # add K3_ANCHOR_MODES=decode for one mode
#
# **Configured by env var, and that is not the thing CLAUDE.md forbids.** That rule ("instruments
# go behind a feature AND a flag, never an env var") gives three reasons, all about the engine
# binary: an env var is invisible to `--help`, absent from the recorded command line in
# `benchmarks.md`, and silently active in a build that looks stock. None applies to a script that
# is not a cargo run, has no `--help`, and whose invocation is recorded on the line above and in
# `anchor.md`. `build.rs` and `tests/common/f4_artifact_dir.rs` each carry the same exemption with
# the same argument. `${VAR:?}` below makes a missing one fail loudly, which is stronger than a
# flag's default would be.
#
# Writes `target/k3-anchor/gold-<mode>-<defect>.bin` and the reddening matrix to stdout, and
# compares the fresh `None` decode golden against the vendored bytes.
set -euo pipefail

VENV=${K3_ANCHOR_VENV:?set K3_ANCHOR_VENV to the venv holding torch+fla+transformers}
REF=${K3_ANCHOR_REF:?set K3_ANCHOR_REF to the dir holding the pinned reference .py files}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/target/k3-anchor
CONFIG=$ROOT/docs/measurement/k3-reference/config.json
VENDORED=$ROOT/tests/k3-anchor-decode.bin
PY=$VENV/bin/python
DRIVER=$ROOT/tests/k3_anchor_driver.py
# The repo's spelling, shared with `mode-matrix.sh`, `smoke-matrix.sh` and
# `layer-major-neutrality.sh` — which explains why the override exists: a lock path can only move
# EVERYWHERE AT ONCE, and two cohorts on two paths are two tenants that both believe they hold the
# mutex. This script read a bare `GPU_LOCK` for one day, which is exactly how a script gets left
# behind on the old path.
LOCK=${RIVOLI_GPU_LOCK:-/var/run/sys-gpu.lock}

DEFECTS=(
    None
    MlaLoraEps1e5 MlaScaleFromNope
    KdaNoQkL2Norm KdaGateLowerBoundOff KdaStateLayout KdaBetaSigmoidOutside
    ExpertW1W3Swap DenseMlpGateUpSwap RouterBiasInWeight LatentNormAfterUp
    AttnResNormalisedValues
)
MODES=(${K3_ANCHOR_MODES:-decode prefill})

mkdir -p "$OUT"

for mode in "${MODES[@]}"; do
    for d in "${DEFECTS[@]}"; do
        # One flock per invocation, not one around the loop: an arm that holds the device across
        # twelve model builds starves every other tenant for no reason, and each run is
        # independent. `-E 66` so "never acquired the lock" is distinguishable from "the driver
        # failed" — `flock -w` exits 1 on timeout, which is also what python exits on a traceback.
        flock -w 900 -E 66 "$LOCK" "$PY" "$DRIVER" \
            --ref "$REF" --config "$CONFIG" --mode "$mode" --defect "$d" \
            --out "$OUT/gold-$mode-$d.bin" >/dev/null
    done
    echo "=== $mode ==="
    # ONE `--compare` per defect, printing the matrix that was gated. It exits non-zero if the
    # defect changed nothing, if it reddened a layer upstream of itself, or if the two runs
    # captured different tensors — see the driver's `compare`. Two invocations (one to print, one
    # to gate) would be two independent scorings whose agreement is assumed.
    for d in "${DEFECTS[@]:1}"; do
        "$PY" "$DRIVER" --compare "$OUT/gold-$mode-None.bin" "$OUT/gold-$mode-$d.bin"
    done
done

# The vendored bytes are the whole point of generating these on a GPU once, so a regeneration
# that no longer reproduces them is the thing most worth knowing. `cmp` rather than a human
# diffing the paths the header used to print.
if [[ -f $OUT/gold-decode-None.bin ]]; then
    if cmp -s "$OUT/gold-decode-None.bin" "$VENDORED"; then
        echo "vendored decode golden reproduced byte-for-byte"
    else
        echo "DIFFERS from $VENDORED — re-vendor deliberately, or find out why it moved" >&2
        exit 1
    fi
fi
