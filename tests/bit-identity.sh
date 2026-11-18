#!/usr/bin/env bash
# The bit-identity gate (docs/investigations/rust-kernels.md §3): every Rust kernel against
# its hipcc twin at tag `archive/hip-kernels`, byte-identical or RED. The harness is
# crates/engine/tests/bit_identity/; this script owns the two things a cargo test cannot:
# the REFERENCE, built from the tag rather than from the working tree, and the GPU
# discipline (flock + contention witness through tests/gpu-witness.sh's run_arm).
#
# **The reference is the tagged C, never the tree.** `git archive` extracts
# crates/backend/kernels at the tag into scratch and hipcc builds REF_SOURCES there with the
# flags crates/backend/build.rs used for them (`--offload-arch=gfx1151 -O3 -fPIC`), as a
# shared object whose `rivoli_*` launchers the harness dlopens. Comparing against anything
# rebuilt from the working tree would let a port that edited the C beside the Rust compare
# against its own edit.
#
# **It never builds Rust** (device-halves.sh's rule: no build inside the lock, none between
# arms). Build first, outside:
#
#   CARGO_TARGET_DIR=/var/cache/rivoli/target/<checkout> cargo build -p rivoli-engine --test bit_identity
#   CARGO_TARGET_DIR=/var/cache/rivoli/target/<checkout> tests/bit-identity.sh
#
# usage: tests/bit-identity.sh [--expect-red=ROW[,ROW…]] [test-binary]
#        --expect-red=ROWS  inverts the verdict, for a recorded red proof: GREEN only if every
#                           row ran and examined something, EXACTLY the named rows are RED, and
#                           every other row is BIT-IDENTICAL — a plant must redden the assertion
#                           under test, not merely some assertion (gate-red-proofs.md's rule).
#        test-binary        a specific built binary (a planted copy); default the newest built.
#        BITIDENT_CLASSIFY=<stdout-file>  classify an existing arm's stdout and exit, no GPU and
#                           no build — the classifier's own red proofs run this way.
# exit:  0 verdict met · 1 verdict not met · 2 setup refused · 3 arm DISCARDED (contention)

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# shellcheck source=tests/gpu-witness.sh
source "$HERE/gpu-witness.sh"

TAG=archive/hip-kernels
# The tagged files with a Rust twin. Phase 1 lanes append theirs.
REF_SOURCES=(activation residual)
ARCH=${RIVOLI_OFFLOAD_ARCH:-gfx1151}

EXPECT_RED=
case "${1:-}" in --expect-red=?*) EXPECT_RED=${1#--expect-red=}; shift ;; esac

# The verdict, from an arm's stdout ($1) and its exit code ($2). Counted from the binary's
# per-row token lines and checked against the binary's own RESULT total — never from rc
# alone: a row that examined nothing prints VACUOUS and is neither verdict, and an arm that
# died part-way prints fewer rows than its RESULT line (or none) and is refused too.
classify() {
    local out=$1 rc=$2 rows total vac reds red_names
    rows=$(grep -cE '^(BIT-IDENTICAL|RED|VACUOUS) ' "$out")
    total=$(sed -nE 's/^RESULT [0-9]+ of ([0-9]+) rows.*/\1/p' "$out" | tail -1)
    vac=$(grep -cE '^VACUOUS ' "$out")
    reds=$(grep -cE '^RED ' "$out")
    red_names=$(grep -E '^RED ' "$out" | sed -E 's/^RED ([^:]+):.*/\1/' | sort | paste -sd, -)
    echo "rows=$rows of ${total:-?} red=$reds vacuous=$vac rc=$rc"
    if [ -z "$total" ] || [ "$rows" -ne "$total" ] || [ "$rows" -eq 0 ]; then
        echo "RED: the arm did not report every row (rows=$rows, RESULT total=${total:-none})" >&2
        return 1
    fi
    if [ "$vac" -ne 0 ]; then
        echo "RED: $vac row(s) VACUOUS — examined nothing, which is not a verdict" >&2
        return 1
    fi
    if [ -n "$EXPECT_RED" ]; then
        local want
        want=$(printf '%s' "$EXPECT_RED" | tr ',' '\n' | sort | paste -sd, -)
        if [ "$red_names" = "$want" ] && [ "$rc" -ne 0 ]; then
            echo "GREEN (expect-red): exactly [$want] reddened"
            return 0
        fi
        echo "RED (expect-red): wanted [$want] red, got [${red_names}] (rc=$rc)" >&2
        return 1
    fi
    if [ "$reds" -eq 0 ] && [ "$rc" -eq 0 ]; then
        echo "GREEN: every row BIT-IDENTICAL"
        return 0
    fi
    echo "RED: [${red_names}] differ from the tagged hipcc twin (rc=$rc)" >&2
    return 1
}

if [ -n "${BITIDENT_CLASSIFY:-}" ]; then
    classify "$BITIDENT_CLASSIFY" "${BITIDENT_RC:-0}"
    exit $?
fi

LOCK=${GPU_LOCK:-/var/run/sys-gpu.lock}
[ -e "$LOCK" ] || { echo "FAIL: $LOCK does not resolve (gate-red-proofs.md §1780)" >&2; exit 2; }
TARGET=${CARGO_TARGET_DIR:-}
[ -n "$TARGET" ] || { echo "FAIL: CARGO_TARGET_DIR unset (§12)" >&2; exit 2; }

BIN=${1:-$(find "$TARGET/debug/deps" -maxdepth 1 -name 'bit_identity-*' -type f -executable \
    -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)}
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "FAIL: no built bit_identity binary — run the cargo build line in this header" >&2
    exit 2
}

SCRATCH=/var/cache/rivoli/scratch/bit-identity/$(date +%Y%m%dT%H%M%S)-$$
mkdir -p "$SCRATCH/ref" || exit 2

# ── the reference, outside the lock ─────────────────────────────────────────────────────
TAG_SHA=$(git -C "$ROOT" rev-parse "$TAG^{commit}") || { echo "FAIL: tag $TAG missing" >&2; exit 2; }
git -C "$ROOT" archive "$TAG" crates/backend/kernels | tar -x -C "$SCRATCH/ref" || exit 2
KDIR=$SCRATCH/ref/crates/backend/kernels
SRCS=()
for s in "${REF_SOURCES[@]}"; do SRCS+=("$KDIR/$s.hip"); done
SO=$SCRATCH/ref/rivoli_ref.so
echo "reference: $TAG @ $TAG_SHA, $(hipcc --version 2>/dev/null | grep -m1 -i 'hip version')"
hipcc "--offload-arch=$ARCH" -O3 -fPIC -shared -Wl,-Bsymbolic "${SRCS[@]}" -o "$SO" \
    >"$SCRATCH/hipcc.log" 2>&1 || { echo "FAIL: hipcc (see $SCRATCH/hipcc.log)" >&2; exit 2; }
echo "binary:    $BIN"

# ── the arm ─────────────────────────────────────────────────────────────────────────────
rc=0
run_arm bit-identity "$SCRATCH/arm.stdout" "$SCRATCH/arm.log" \
    env RIVOLI_BITIDENT_REF="$SO" "$BIN" --ignored --exact \
    every_rust_kernel_is_byte_identical_to_its_tagged_hipcc_twin --test-threads=1 --nocapture || rc=$?
[ "$rc" -eq 3 ] && exit 3

grep -hE '^(BIT-IDENTICAL|RED|VACUOUS) |^    |^RESULT ' "$SCRATCH/arm.stdout" || true
echo "logs: $SCRATCH"
classify "$SCRATCH/arm.stdout" "$rc"
