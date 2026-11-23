# Where the test scripts put their scratch files. Sourced, never executed.
#
#   source "$(dirname "$0")/scratch.sh"
#   SCRATCH=$(scratch_dir parity-glm)      # a fresh directory, printed
#
# CLAUDE.md: scratch goes to /var/cache/rivoli/scratch, never /tmp — /tmp is a 63 GB tmpfs in
# RAM that competes with the GPU's unified-memory budget, and it is session-scoped, so a run's
# evidence is gone next session. Until 2026-11-23 every gate script spelled
# `mktemp -d "${TMPDIR:-/tmp}/…"`, so an unset TMPDIR (the normal case) put GLM-sized id dumps
# and witness files in RAM.
#
# Order: an explicit TMPDIR still wins, because a caller who sets it (the lease drivers do) has
# chosen the place. Otherwise the house scratch subvolume when it exists, and /tmp only on a
# machine that has none (CI's hosted runner), where RAM competition with a GPU does not arise.
scratch_root() {
    if [ -n "${TMPDIR:-}" ]; then
        printf '%s\n' "$TMPDIR"
    elif [ -d /var/cache/rivoli/scratch ] && [ -w /var/cache/rivoli/scratch ]; then
        printf '%s\n' /var/cache/rivoli/scratch
    else
        printf '%s\n' /tmp
    fi
}

scratch_dir() { # $1 = name stem
    mktemp -d "$(scratch_root)/$1.XXXXXX"
}
