#!/usr/bin/env bash
# The two device halves that gate-red-proofs.md §13 and release-v3.md §2 row 6 owe as RUNS:
# M17c's block-attend kernel (`kernel_glimmer_block_attend`, 18 tests) and M11's fp8
# anti-fallback gate (`glimmer_fp8_decode`). Both suites are written; neither had been
# executed with its evidence recorded — which is the only reason this file exists.
#
# **Why a script rather than an ssh one-liner.** §13's standing lesson, applied one level
# down: an arm driven from a session scratchpad is not evidence, because the examined count
# can reach zero and nobody notices. A hand-run arm has a second failure mode — no
# contention witness. The flock on this box is ADVISORY and peer agents skip it, so the
# witness (`run_arm`, tests/gpu-witness.sh: KFD holders + `mem_info_gtt_used` sampled every
# 5 s, DISCARD on a foreign tenant) is what separates a clean decode from one that shared
# the device. Run through this script an arm cannot be witnessed-optional.
#
# **Why it never builds.** `parity-glm.sh`'s rule, for two reasons that both drew blood
# here: a build inside the lock holds the device for compile time, and building between the
# two arms of a measurement evicts page cache — it moved ms/miss from 1.36 to 5.14 once. So
# build FIRST, outside, then run:
#
#   CARGO_TARGET_DIR=/var/cache/rivoli/target/<this-checkout> \
#     cargo test -p rivoli-engine --test kernel_glimmer_block_attend --test glimmer_fp8_decode --no-run
#   CARGO_TARGET_DIR=/var/cache/rivoli/target/<this-checkout> tests/device-halves.sh all
#
# usage: tests/device-halves.sh [attend|fp8|all]        (default all)
# env:   CARGO_TARGET_DIR — required, and it must name THIS checkout's own target dir.
#        §12: the box-wide pam_env value outranks `.cargo/config.toml`, and a shared dir
#        makes cargo replay a SIBLING's build-script output — the duplication gate then
#        scans the wrong tree and reports a green nothing established.
# exit:  0 both arms green · 1 an arm reddened · 2 refused at setup · 3 an arm was
#        DISCARDED on contention (run_arm's own exit, so the run is loud, not short)

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=tests/gpu-witness.sh
source "$HERE/gpu-witness.sh"

LOCK=${GPU_LOCK:-/var/run/sys-gpu.lock}
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/device-halves.XXXXXX")
CELL=${1:-all}

# The canonical lock path, not /run/sys-gpu.lock: §1780 records the 2026-09-02 window when a
# stray read-only bind over `/` made `/var/run -> ../run` dangle for every process on the box
# and peers took to locking the resolved file instead. Refusing here is the point — an arm
# that takes a different lock than its neighbours is an arm that believes it is sole tenant.
[ -e "$LOCK" ] || {
    echo "FAIL: $LOCK does not resolve. gate-red-proofs.md §1780 is the /var/run bind defect;" >&2
    echo "      do not fall back to /run/sys-gpu.lock — that hides the machine fault." >&2
    exit 2
}
TARGET=${CARGO_TARGET_DIR:-}
[ -n "$TARGET" ] || { echo "FAIL: CARGO_TARGET_DIR unset (§12 makes it the mechanism, not an optimisation)" >&2; exit 2; }
[ -d "$TARGET/debug/deps" ] || {
    echo "FAIL: no $TARGET/debug/deps — this gate never builds; run the --no-run line in the header" >&2
    exit 2
}

binary_for() { # $1 = test stem → newest built executable, skipping cargo's `.d` sidecars
    local b
    b=$(find "$TARGET/debug/deps" -maxdepth 1 -name "$1-*" ! -name '*.d' \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-) || true
    [ -n "$b" ] || {
        echo "FAIL: no built binary for --test $1 under $TARGET/debug/deps" >&2
        exit 2
    }
    printf '%s\n' "$b"
}

# A binary older than the kernels it launches tests a stale tree — §11's exact warning, that
# a fast rebuild proves nothing about whether hipcc ran. Warn rather than refuse: the
# operator may be scoring a branch deliberately, and parity-glm.sh made this a warning too.
warn_if_stale() { # $1 = binary
    local st
    st=$(find "$HERE/../crates/backend" "$HERE/../crates/engine" \( -name '*.hip' -o -name '*.hpp' \
          -o -name '*.rs' \) -newer "$1" 2>/dev/null | head -5) || true
    [ -z "$st" ] || echo "WARN: sources newer than $1 — this arm tests an older tree:" >&2
    while IFS= read -r f; do printf '        %s\n' "$f" >&2; done <<<"$st"
}

# One arm: witnessed, flocked, streams to files, and its OWN exit code read back unpiped
# (§7e-bis's false green was a pipeline's exit code).
arm() { # $1 = cell, $2 = test stem, $3 = 1 to skip if absent (arch-dependent cells)
    local name=$1 stem=$2 bin rc summary
    bin=$(binary_for "$stem")
    warn_if_stale "$bin"
    echo "=== $name: $bin"
    rc=0
    run_arm "$name" "$SCRATCH/$name.stdout" "$SCRATCH/$name.log" "$bin" --test-threads=1 --nocapture || rc=$?
    # libtest prints its result line on STDOUT and diagnostics on stderr, so classify over
    # BOTH files -- `parity-glm.sh` greps the pair for exactly this reason. The first version
    # of this line read only the stderr file, found nothing, printed "NO TEST RESULT LINE"
    # and still called the arm GREEN: a zero examined count reading as a pass is the defect
    # CLAUDE.md names outright, and it arrived through my own gate -- which is the argument
    # for the refusal below rather than for a bug report.
    summary=$(grep -hoE 'test result: [A-Za-z]+[.] [0-9]+ passed; [0-9]+ failed' \
        "$SCRATCH/$name.stdout" "$SCRATCH/$name.log" 2>/dev/null | tail -1) || true
    passed=$(printf '%s' "$summary" | sed -nE 's/.* ([0-9]+) passed.*/\1/p')
    failed=$(printf '%s' "$summary" | sed -nE 's/.*; ([0-9]+) failed.*/\1/p')
    if [ -z "$passed" ] || [ "$passed" -eq 0 ]; then
        # No count, or a zero count: the arm examined nothing, so no rc can make it GREEN.
        echo "RED $name: no scored tests (summary='${summary:-none}', rc=$rc) — an arm that" >&2
        echo "    examined nothing is not a pass; see $SCRATCH/$name.{stdout,log}" >&2
        return 1
    fi
    if [ "$rc" -eq 0 ] && [ "${failed:-1}" -eq 0 ]; then
        echo "GREEN $name: $summary"
        return 0
    fi
    echo "RED $name (rc=$rc): ${summary:-no result line} — see $SCRATCH/$name.log" >&2
    return 1
}

rc=0
case "$CELL" in
    attend) arm attend kernel_glimmer_block_attend || rc=1 ;;
    fp8)    arm fp8 glimmer_fp8_decode || rc=1 ;;
    all)
        arm attend kernel_glimmer_block_attend || rc=1
        arm fp8 glimmer_fp8_decode || rc=1
        ;;
    *) echo "usage: $0 [attend|fp8|all]" >&2; exit 2 ;;
esac

echo "   evidence: $SCRATCH (per-arm witness files, stdout, log; GTT and loadavg above)"
exit "$rc"
