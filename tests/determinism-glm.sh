#!/usr/bin/env bash
# The determinism gate: GLM greedy decode must reproduce ITSELF.
#
# Two runs of ONE binary with byte-identical arguments must produce byte-identical token ids.
# That is the weakest possible correctness claim an engine can make, and on 2026-08-17 GLM
# int3-vq did not meet it: 61 of 512 ids differed on a quiet box, 496 of 512 under CPU/NFS
# load, first difference anywhere from position 13 to 452.
#
#   green:      tests/determinism-glm.sh <artifact-dir> [ngen] [max-mem-GiB]
#   red-proof:  tests/determinism-glm.sh --self-test      (deviceless, no artifact, no GPU)
#
# ## LENGTH IS THE WHOLE POINT, and a short run is not a weaker gate but a USELESS one
#
# The defect passes at 32 tokens. Two no-MTP runs and an --mtp run were all byte-identical
# over 32 ids on the same tree, same artifact, same box, that produced 61/512 at 512. A
# 32-token determinism gate is therefore green on a broken engine — it is not conservative,
# it is vacuous. So `ngen` has a HARD FLOOR (see MIN_NGEN) and the gate refuses below it with
# the reason rather than reporting a green nobody should believe.
#
# The floor is 256, and it is the honest number rather than a round one: the earliest quiet-box
# onset actually observed is position 452, and the earliest onset observed at all is 13, so
# there is no length at which a single pair is guaranteed to catch it. 512 is the default
# because it is the length every recorded measurement of this defect used; 256 is the floor
# because below the earliest observed quiet onset a green says nothing at all, and a gate that
# can be dialled down to a length where it always passes is a gate someone will dial down.
# **A green at 512 bounds the rate; it does not prove determinism.** State the length with
# every green.
#
# ## What it does NOT do
#
# It does not use `--divergence-log`. That instrument adds three kernel launches per layer and
# a D2H per token, which is exactly the class of perturbation recorded to MASK this fault — so
# it is the tool you reach for AFTER this gate goes red, in a separate run, never inside it.
# For the same reason: never run this gate on a `--features trace` binary.
#
# It also never builds. A cargo run between the arms evicts page cache and poisons the
# comparison (measured in the old tree: ms/miss 1.36 -> 5.14), and a determinism gate is more
# sensitive to that than a throughput one because page-cache state is part of the timing this
# defect is a function of.
#
# Exit: 0 gate green at the stated length (or --self-test confirmed the comparator reddens)
#       1 gate RED — the two runs differ, or a run failed
#       2 setup error (usage, missing binary/artifact, non-id output)
#       3 arm discarded (foreign GPU tenant witnessed) — rerun
#       66 GPU lock file missing (house convention: /run is tmpfs, dies on reboot)
set -uo pipefail

BIN=${DETERMINISM_BIN:-${CARGO_TARGET_DIR:-$(dirname "$0")/../target}/debug/rivoli}
LOCK=/var/run/sys-gpu.lock
# The floor, argued in the header. A caller may raise `ngen`, never lower it past this.
MIN_NGEN=256

# --- the comparator, and its red proof ----------------------------------------------------
#
# One function, called by the gate and by --self-test. That is what makes the self-test a
# proof rather than a demonstration of a second implementation: the thing shown to redden IS
# the thing the gate runs.
compare_ids() { # $1 = arm A ids, $2 = arm B ids; 0 identical, 1 differ
    if cmp -s "$1" "$2"; then return 0; fi
    local n first
    n=$(paste "$1" "$2" | awk '$1 != $2' | wc -l)
    # 0-BASED, matching how every recorded measurement of this defect states its onset
    # ("first at position 13", "first at 452"): position 0 is the first generated token.
    first=$(paste "$1" "$2" | awk '$1 != $2 {print NR - 1; exit}')
    echo "RED: $n of $(wc -l <"$1") ids differ, first at position ${first:-?} (0-based)" >&2
    paste "$1" "$2" | awk -v F="${first:-0}" 'NR >= F && NR <= F + 4 {printf "  pos %d: A=%s B=%s\n", NR - 1, $1, $2}' >&2
    return 1
}

if [ "${1:-}" = "--self-test" ]; then
    # DEVICELESS RED PROOF of the comparator. It does not prove the engine is nondeterministic
    # (the 512-token arm does that, and did); it proves that when two id streams differ this
    # gate SAYS SO and exits non-zero — the half a green run can never demonstrate about
    # itself, and the half that silently breaks when someone "simplifies" the diff.
    d=$(mktemp -d "${TMPDIR:-/tmp}/determinism-selftest.XXXXXX")
    trap 'rm -rf "$d"' EXIT
    printf '10\n20\n30\n40\n' >"$d/a"
    printf '10\n20\n30\n40\n' >"$d/b"
    compare_ids "$d/a" "$d/b" || { echo "FAIL: identical streams must compare EQUAL" >&2; exit 2; }
    # One id changed, in the middle — the shape of the real defect (a late first difference).
    printf '10\n20\n99\n40\n' >"$d/c"
    if compare_ids "$d/a" "$d/c" 2>/dev/null; then
        echo "FAIL: the comparator did NOT redden on differing id streams — this gate cannot fail, so its greens mean nothing" >&2
        exit 2
    fi
    # And a truncated stream, which is what a crashed arm produces: `cmp -s` catches it, but a
    # naive `diff <(sort)`-style comparator would not, and that substitution has been made.
    printf '10\n20\n' >"$d/t"
    if compare_ids "$d/a" "$d/t" 2>/dev/null; then
        echo "FAIL: the comparator did NOT redden on a TRUNCATED stream (a crashed arm)" >&2
        exit 2
    fi
    echo "SELF-TEST ok: the comparator reddens on a changed id and on a truncated stream"
    exit 0
fi

ARTIFACT=${1:?usage: determinism-glm.sh <artifact-dir> [ngen] [max-mem-GiB] | --self-test}
NGEN=${2:-512}
MEM=${3:-115}

[ "$NGEN" -ge "$MIN_NGEN" ] 2>/dev/null || {
    cat >&2 <<EOF
FAIL: ngen=$NGEN is below the $MIN_NGEN floor.
This defect is byte-identical at 32 tokens and reproduces by 512 (measured 2026-08-17,
both on the same tree and artifact). A short determinism gate is not a conservative gate,
it is a green on a broken engine. Raise ngen or do not run this.
EOF
    exit 2
}
[ -x "$BIN" ] || { echo "FAIL: binary missing: $BIN (build it BEFORE the gate — this gate never builds)" >&2; exit 2; }
[ -e "$LOCK" ] || { echo "FAIL: GPU lock file missing: $LOCK" >&2; exit 66; }
[ -d "$ARTIFACT" ] || { echo "FAIL: artifact dir missing: $ARTIFACT" >&2; exit 2; }
# NO TRACE-BUILD PROBE, and its absence is deliberate rather than an omission.
#
# A `--features trace` binary has an extra `device_sync` per layer-with-misses and is recorded
# to MASK this defect, so a green from one is worse than no measurement. The obvious guard is
# to grep `--help` for `--trace` — and it is a check that CANNOT FAIL: `--trace` is declared
# unconditionally in `crates/cli/src/main.rs`, so it appears in every build's help and the
# probe warns on every run. A guard that always fires is noise that teaches its reader to
# ignore it, which is worse than no guard, and this repo bans the shape.
#
# There is no signal in the binary that distinguishes the feature set, so the rule stays where
# a rule with no enforcement belongs: written down, here and in the header, for the operator.
# **Do not run this gate on a binary built with `--features trace`.** If that ever needs
# enforcing, the honest fix is for `--version` to print its feature set, not a grep that
# guesses.

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/determinism-glm.XXXXXX")
echo "== determinism-glm | ngen=$NGEN mem=${MEM}GiB scratch=$SCRATCH"
echo "   bin: $(stat -c '%y' "$BIN") $BIN"

# --- contention witness, per arm ----------------------------------------------------------
# Lifted wholesale in SHAPE from tests/parity-glm.sh, whose header carries the full argument:
# the flock is advisory, peers skip it, ours is identified by DESCENT from the arm's own pid
# (never by binary path — a peer may be running the same binary), and every KFD entry is
# resolved against /proc before it is believed.
descends_from() { # $1 = candidate pid, $2 = ancestor pid
    local p=$1
    while [ "$p" -gt 1 ] 2>/dev/null; do
        [ "$p" = "$2" ] && return 0
        p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null) || return 1
        [ -n "$p" ] || return 1
    done
    return 1
}

witness() { # $1 = out-file, $2 = arm pid; runs until killed
    while :; do
        find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null |
            while read -r p; do
                [ -d "/proc/$p" ] || continue
                descends_from "$p" "$2" && continue
                cmd=$(tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null) || true
                echo "pid=$p cmd=${cmd:-?}" >>"$1"
            done
        sleep 5
    done
}

gtt_used() { # KFD is blind to Vulkan tenants, so the pre-arm baseline reads the allocator.
    cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

run_arm() { # $1 = arm name, $2 = ids out path
    local wfile="$SCRATCH/witness-$1" wpid apid gtt rc=0
    gtt=$(gtt_used)
    if [ "$gtt" -gt $((2 << 30)) ]; then
        echo "DISCARD arm '$1': $((gtt >> 20)) MiB GTT already held before the arm started — a ghost tenant KFD cannot see" >&2
        exit 3
    fi
    : >"$wfile"
    # Both arms take the SAME argument list, which is the entire experiment. Anything that
    # differs between them — including the ids path, which is why it is not a flag the engine
    # reads — would make this a comparison of two configurations instead of a repeatability
    # test.
    flock "$LOCK" "$BIN" "$ARTIFACT" --bench "$NGEN" --mode int3-vq --attn dense \
        --max-mem "$MEM" --dump-ids "$2" >"$SCRATCH/$1.out" 2>"$SCRATCH/$1.log" &
    apid=$!
    witness "$wfile" "$apid" &
    wpid=$!
    wait "$apid" || rc=$?
    kill "$wpid" 2>/dev/null || true
    wait "$wpid" 2>/dev/null || true
    if [ -s "$wfile" ]; then
        echo "DISCARD arm '$1': foreign GPU tenant witnessed:" >&2
        sort -u "$wfile" >&2
        exit 3
    fi
    [ "$rc" -eq 0 ] || { echo "FAIL: arm '$1' exited $rc — tail of its log:" >&2; tail -20 "$SCRATCH/$1.log" >&2; exit 1; }
    # The header line `--dump-ids` writes is stripped, so the comparison is ids only: a header
    # naming the arm would differ between arms for reasons that are not the measurement.
    grep -E '^[0-9]+$' "$2" >"$2.ids"
    local n
    n=$(wc -l <"$2.ids")
    [ "$n" -eq "$NGEN" ] || {
        echo "FAIL: arm '$1' produced $n ids, expected $NGEN. A SHORT arm is not a pass — it means the run stopped early (EOS or an error), and comparing two short arms of the same length would be a green over a decode that never happened." >&2
        exit 1
    }
}

run_arm a "$SCRATCH/a"
run_arm b "$SCRATCH/b"

if compare_ids "$SCRATCH/a.ids" "$SCRATCH/b.ids"; then
    echo "GREEN: two runs at identical flags produced identical ids over $NGEN tokens (mem=${MEM}GiB)."
    echo "       This BOUNDS the divergence rate at this length; it does not prove determinism."
    exit 0
fi
cat >&2 <<EOF
GATE RED: GLM greedy decode did not reproduce itself over $NGEN tokens.
  arm A ids: $SCRATCH/a.ids
  arm B ids: $SCRATCH/b.ids
Next step is a coordinate, not another id diff: re-run both arms on a
\`--features corruption-probe\` binary with --divergence-log and diff the two logs. The first
differing LINE is the (position, layer); the first differing COLUMN names the mechanism.
EOF
exit 1
