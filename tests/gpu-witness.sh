# Contention witness for a GPU arm. `source` this; it defines functions and runs nothing.
#
# ONE file, not copy-paste, because these three functions ARE the false-green guard and the
# failure mode of a two-copy guard is that one copy quietly stops guarding — jscpd scans only
# Rust, so nothing here would compare them. Extracted from tests/parity-glm.sh 2026-08-17 when a
# second gate needed them; CLAUDE.md's measurement-discipline section carries the general rule.
#
# Each function encodes one trap that has already drawn blood; the argument is at the function.

# Is $1 a descendant of $2? Walks /proc PPid links to pid 1.
#
# DESCENT, never a binary path: on this shared box a peer agent may be running the identical
# binary, and a path whitelist would wave the real contender through — the exact false green the
# witness exists to prevent.
descends_from() { # $1 = candidate pid, $2 = ancestor pid
    local p=$1
    while [ "$p" -gt 1 ] 2>/dev/null; do
        [ "$p" = "$2" ] && return 0
        p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null) || return 1
        [ -n "$p" ] || return 1
    done
    return 1
}

# Append every FOREIGN KFD tenant to $1 every 5 s until killed. Runs in the background; the
# caller kills it when its arm ends and treats a non-empty $1 as a discard.
#
# `find`, never `ls`: `ls` on the empty directory returned the literal string `(empty)`, which
# reads as one phantom holder and was twice mistaken for a stale entry. Every candidate is then
# resolved against /proc before it is believed.
witness() { # $1 = out-file, $2 = arm pid
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

# Total GTT bytes the amdgpu allocator reports held, across cards. Read BEFORE an arm starts.
#
# KFD IS BLIND TO SOME TENANTS: llama-swap once held 1.6 GB of GTT with zero entries under
# /sys/class/kfd, so the baseline reads the allocator itself rather than trusting the process
# list. A caller's threshold must sit BELOW 1.6 GB or it would miss that tenant.
gtt_used() {
    cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | awk '{s+=$1} END {print s+0}'
}
