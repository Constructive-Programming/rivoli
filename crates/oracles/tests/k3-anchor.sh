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
# Writes `target/k3-anchor/gold-<mode>-<salt>-<defect>.bin` and the reddening matrix to stdout, and
# `cmp`s each fresh `None` decode golden against its vendored twin.
set -euo pipefail

VENV=${K3_ANCHOR_VENV:?set K3_ANCHOR_VENV to the venv holding torch+fla+transformers}
REF=${K3_ANCHOR_REF:?set K3_ANCHOR_REF to the dir holding the pinned reference .py files}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)   # workspace root; this script lives in crates/oracles/tests/
TESTS=$ROOT/crates/oracles/tests
OUT=$ROOT/target/k3-anchor
CONFIG=$ROOT/docs/measurement/k3-reference/config.json
PY=$VENV/bin/python
DRIVER=$TESTS/k3_anchor_driver.py
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
# TWO independent weight draws, and both are vendored. One draw cannot show that a defect's
# localisation is a property of the arithmetic rather than of the particular numbers it landed on,
# and a kernel bug that is degenerate at one draw's values — a vanishing top-k weight, a `beta` that
# saturates the gate — hides completely. Every defect below is scored against both.
SALTS=(${K3_ANCHOR_SALTS:-k3-anchor-1 k3-anchor-2})

mkdir -p "$OUT"

for salt in "${SALTS[@]}"; do
for mode in "${MODES[@]}"; do
    for d in "${DEFECTS[@]}"; do
        # One flock per invocation, not one around the loop: an arm that holds the device across
        # twelve model builds starves every other tenant for no reason, and each run is
        # independent. `-E 66` so "never acquired the lock" is distinguishable from "the driver
        # failed" — `flock -w` exits 1 on timeout, which is also what python exits on a traceback.
        flock -w 900 -E 66 "$LOCK" "$PY" "$DRIVER" \
            --ref "$REF" --config "$CONFIG" --mode "$mode" --defect "$d" --salt "$salt" \
            --out "$OUT/gold-$mode-$salt-$d.bin" >/dev/null
    done
    echo "=== $mode $salt ==="
    # ONE `--compare` per defect, printing the matrix that was gated. It exits non-zero if the
    # defect changed nothing, if it reddened a layer upstream of itself, or if the two runs
    # captured different tensors — see the driver's `compare`. Two invocations (one to print, one
    # to gate) would be two independent scorings whose agreement is assumed.
    for d in "${DEFECTS[@]:1}"; do
        "$PY" "$DRIVER" --compare "$OUT/gold-$mode-$salt-None.bin" "$OUT/gold-$mode-$salt-$d.bin"
    done
done
done

# The vendored bytes are the whole point of generating these on a GPU once, so a regeneration
# that no longer reproduces them is the thing most worth knowing. `cmp` rather than a human
# diffing the paths the header used to print.
#
# Enumerated from the VENDORED files rather than from `SALTS`, and it reports a census. Driving
# the loop from `SALTS` made a narrowed run (`K3_ANCHOR_SALTS=k3-anchor-1`, which halves a ~25 min
# GPU-locked regeneration) print one cheerful "reproduced byte-for-byte" and exit 0 — while the
# other vendored golden went unchecked and looked, from the output, exactly like a full pass. The
# skip is legitimate; being quiet about it is not.
rc=0
verified=0
unverified=()
shopt -s nullglob
vendored_goldens=("$ROOT"/tests/k3-anchor-decode-*.bin)
shopt -u nullglob
for vendored in "${vendored_goldens[@]}"; do
    salt=$(basename "$vendored" .bin); salt=${salt#k3-anchor-decode-}
    fresh=$OUT/gold-decode-$salt-None.bin
    if [[ ! -f $fresh ]]; then
        unverified+=("$salt")
    elif cmp -s "$fresh" "$vendored"; then
        echo "vendored $salt decode golden reproduced byte-for-byte"
        verified=$((verified + 1))
    else
        echo "DIFFERS from $vendored — re-vendor deliberately, or find out why it moved" >&2
        rc=1
    fi
done
echo "vendored goldens verified: $verified of ${#vendored_goldens[@]}"
if ((${#unverified[@]})); then
    echo "NOT VERIFIED by this run (no fresh decode golden): ${unverified[*]}" >&2
    echo "  Their bytes are still FNV-pinned by tests/k3_anchor.rs, so they cannot drift" >&2
    echo "  unnoticed — but nothing here re-derived them from the reference." >&2
fi
# A regeneration that reproduces NOTHING is the failure this whole check exists to catch, and it
# is what an empty `$OUT` or a wrong `--out` path looks like from the outside.
if ((verified == 0 && rc == 0)); then
    echo "no vendored golden was reproduced at all — did the driver write to \$OUT?" >&2
    rc=1
fi
exit $rc
