#!/usr/bin/env bash
# Shared GPU-arm discipline for the on-demand device gates: the flock, the contention
# witness, and the GTT baseline. Sourced, never executed.
#
#   source "$(dirname "$0")/gpu-witness.sh"
#   run_arm <name> <stdout-file> <stderr-file> <cmd...>   # returns the arm's rc
#
# **Extracted from parity-glm.sh on 2026-08-16, when ppl-gates.sh needed the same three
# functions.** The alternative was a second copy, and the copy is exactly what the house
# duplication gate would forbid if jscpd covered shell (`crates/cli/build.rs` scans
# `crates`, so it does not) — the rule is the repo's, not the tool's.
#
# CONTRACT with the sourcing script: it must have set `LOCK` (the flock path) and
# `SCRATCH` (a writable dir for the per-arm witness files) before calling `run_arm`, and
# it inherits these exit codes:
#   2  setup error   3  arm discarded (foreign GPU tenant witnessed) — rerun
# Both are parity-glm.sh's own numbering, kept so the two gates classify alike.

# The flock is advisory and other tenants skip it, so every arm carries a witness: KFD
# tenants sampled every 5 s while the arm runs. Ours is identified by DESCENT from the
# arm's own pid — never by binary path, because on this shared machine a peer agent may
# run the identical binary, and a path whitelist would wave it through (the exact
# false-green the witness exists to prevent). Each entry is resolved against /proc before
# being believed (the empty-dir/phantom trap: `ls` on an empty KFD dir returned the
# literal string "(empty)" and was twice read as a stale holder).
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

gtt_used() { # KFD is blind to Vulkan tenants (llama-swap once held 1.6 GB with zero KFD
    # entries), so the pre-arm baseline reads the allocator itself.
    #
    # **Prints nothing when no sysfs file matched, and that is the point.** The glob
    # summed with `awk '{s+=$1} END {print s+0}'` yields a bare `0` for "no tenant" AND
    # for "the path is wrong" — an unarmed guard that reads as a clean box. It has
    # already been read that way once: `docs/measurement/baseline-2026-08-16.md` records
    # two arms taken with "the GTT sysfs read was empty on this kernel (the card index
    # moved)", i.e. against a guard that could not fire. (Re-checked 2026-08-16 on this
    # box: `/sys/class/drm/card0/device/mem_info_gtt_used` exists and reads 18673664
    # idle, so whatever was seen then is not the state now — which is exactly why the
    # distinction must be structural rather than remembered.) `run_arm` turns the empty
    # answer into a setup refusal.
    local f n=0 s=0 v
    for f in /sys/class/drm/card*/device/mem_info_gtt_used; do
        [ -r "$f" ] || continue
        v=$(cat "$f" 2>/dev/null) || continue
        case $v in '' | *[!0-9]*) continue ;; esac
        s=$((s + v))
        n=$((n + 1))
    done
    [ "$n" -gt 0 ] && echo "$s"
}

run_arm() { # $1 = arm name, $2 = stdout file, $3 = stderr file, rest = command.
    # The arm's streams are redirected HERE, on the flock'd command only, so DISCARD
    # diagnostics reach the terminal instead of being buried in the arm's log.
    local wfile="$SCRATCH/witness-$1" wpid apid gtt rc=0
    gtt=$(gtt_used)
    if [ -z "$gtt" ]; then
        echo "FAIL: no readable mem_info_gtt_used under /sys/class/drm/card*/device — the ghost-tenant guard is UNARMED, and an unarmed guard reads exactly like a clean box. Find the node (the card index moves) before taking any number." >&2
        exit 2
    fi
    # 1 GiB. **Was 2 GiB, inherited from parity-glm.sh, and it sat ABOVE the incident it
    # cites** (both reviews, 2026-08-16): the comment on `gtt_used` argues the guard from
    # llama-swap holding 1.6 GB with zero KFD entries, and 1.6e9 < 2^31 — a repeat of the
    # exact event this exists for would have passed. Derivation of the replacement: idle on
    # this box reads 18,673,664 B (~17.8 MiB, measured 2026-08-16), so 1 GiB is 57x idle —
    # loose enough that a just-exited arm's un-reclaimed teardown does not flap the gate
    # into a rerun — and it is below the one ghost tenant on record, which is the bound
    # that matters.
    if [ "$gtt" -gt $((1 << 30)) ]; then
        echo "DISCARD arm '$1': $((gtt >> 20)) MiB GTT already held before the arm started — a ghost tenant (Vulkan/llama-swap?) KFD cannot see" >&2
        exit 3
    fi
    : >"$wfile"
    flock "$LOCK" "${@:4}" >"$2" 2>"$3" &
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
    return "$rc"
}
