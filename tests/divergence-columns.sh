#!/usr/bin/env bash
# Which COLUMN of two `--divergence-log` files moved first — the coordinate AND the mechanism.
#
#   tests/divergence-columns.sh <a.log> <b.log>
#
# ## Why this is a script and not a snippet in the doc
#
# It was a snippet in `docs/investigations/glm-nondeterminism.md`, and the operator running the
# first real pair could not find it and wrote the comparison a second time (2026-08-17). A
# procedure that gets re-derived by whoever needs it is not recorded, whatever the doc says. So
# it lives here, it is executable, and the doc points at it.
#
# ## What it does, and why the FIRST DIFFERING COLUMN is the answer
#
# `--divergence-log` writes one row per (pass, layer) with fourteen fields. Six are device XOR
# folds; three are host-side FNV folds of what the router saw, picked and where the pool put it, and
# `misses`/`relocs` are plain counts rather than folds. The
# first row where the two runs disagree is the coordinate, and *which field disagrees there* names
# the subsystem:
#
#   xn      differs -> attention or its KV cache; the MLP has not run yet
#   h       differs, xn equal -> the gate/up expert BYTES, or that kernel
#   x       differs, xn and h equal -> the down projection, the accumulator, or the drain
#   gl      differs -> the router's INPUT moved, so the fault is upstream of routing
#   pk      differs, gl equal -> routing consulted something outside its inputs (INV-1)
#   sl      differs, pk equal -> the pool placed the same experts in different slots
#   misses  differs -> the two runs made different residency DECISIONS
#   relocs  differs -> the arena compacted in one run and not the other
#
# and the three fetch-path folds, which exist to split a `h`-differs verdict into WHICH HOP:
#
#   bh      the bounce arena, right after the NVMe read      differs -> the DRIVE delivered
#                                                                       different bytes
#   sc      the pool slot, right after the bounce->slot copy  bh equal, sc differs -> the COPY
#   se      the same slot again at end of layer               sc == se but h differs -> the bytes
#                                                            AT REST are right, so the kernel read
#                                                            them too early: a ticket/timeline
#                                                            ordering failure, not a bad payload
#                                                            sc != se -> something wrote the slot
#                                                            after the copy landed
#
# `se` covers only the experts the layer MISSED (folding the whole batch would be ~5x the cost), so
# it is silent about a RESIDENT expert corrupted on an earlier token. State that with any result.
#
# A `-` is NOT MEASURED, never zero: a dense layer has no `h` and no router. Rows are compared
# field-wise as TEXT, and a `-` on one side with a hash on the other is reported as a real
# difference, because it means the two runs disagree about whether a layer had a MoE at all.
#
# The two logs generally have DIFFERENT LENGTHS — after diverging, the arms generate different
# numbers of tokens — so this walks them in parallel and stops at the first disagreement rather
# than requiring equal lengths. That is also why it reports the row NUMBER: past the first
# difference nothing is comparable.
#
# THIS IS A READER, NEVER A GATE. It exits 0 whether or not it found a divergence — only its three
# refusals (unreadable input, missing header, headers that disagree) exit 2. A caller must not key
# on the exit code to decide whether the runs diverged; read the output. Making it exit 1 on "found"
# would turn a diagnostic into a gate, and the gate for this property is
# `tests/determinism-glm.sh`, which compares ids rather than instrumented folds and does not
# perturb the run.
#
# Deviceless. No GPU, no artifact, no lock.
set -uo pipefail

A=${1:-}; B=${2:-}
if [ -z "$A" ] || [ -z "$B" ]; then
    echo "usage: divergence-columns.sh <a.log> <b.log>" >&2
    exit 2
fi
for f in "$A" "$B"; do [ -r "$f" ] || { echo "FAIL: cannot read $f" >&2; exit 2; }; done

body() { grep -vE '^[[:space:]]*(#|$)' "$1"; }
for f in "$A" "$B"; do
    [ "$(body "$f" | wc -l)" -gt 0 ] || { echo "FAIL: $f has no data rows (header only?)" >&2; exit 2; }
done

echo "== $(basename "$A") ($(body "$A" | wc -l) rows)  vs  $(basename "$B") ($(body "$B" | wc -l) rows)"

# THE COLUMN NAMES COME FROM THE FILE'S OWN HEADER, not from a list in here.
#
# The header is `# rivoli-divergence vN <name> <name> ...`, so the writer names its own columns and
# this tool cannot drift from it. Hardcoding them meant that adding three columns silently turned a
# v2 log into a field-count error with a misleading verdict (2026-08-17), and — worse — a REORDER
# would have mislabelled every column while still "working". Both headers must agree, or the two
# runs were produced by different builds and are not comparable.
hdr() { grep -m1 -E '^# rivoli-divergence ' "$1" | sed -E 's/^# rivoli-divergence v[0-9]+ //'; }
HA=$(hdr "$A"); HB=$(hdr "$B")
# BOTH, separately. Checking only A blamed a missing B header on "different columns — different
# builds", which names the wrong cause (review 2026-08-17).
for f in "$A:$HA" "$B:$HB"; do
    [ -n "${f#*:}" ] || { echo "FAIL: ${f%%:*} has no 'rivoli-divergence' header line" >&2; exit 2; }
done
[ "$HA" = "$HB" ] || {
    echo "FAIL: the two logs declare different columns — different builds, not comparable:" >&2
    echo "  A: $HA" >&2
    echo "  B: $HB" >&2
    exit 2
}

# TRUNCATED TO THE COMMON PREFIX FIRST, and that is a fix rather than tidiness.
#
# `paste` pads the exhausted side with an EMPTY field, so once the shorter log runs out a row has
# `n` fields instead of `2n` — and the field-count guard below then reported
# "row 12001 has 11 fields but the header names 11 columns", a self-contradictory setup error, on
# the case this script's own header calls NORMAL (the arms generate different lengths by
# construction once diverged). It also made the "no differing row in the common prefix" message
# unreachable whenever the lengths differed, which is the honest verdict for an identical prefix.
# Reproduced deviceless by review 2026-08-17.
#
# Past the shorter log's end nothing is comparable anyway — there is no second run to compare
# against — so the common prefix IS the comparison, and cutting to it makes `NF == 2n` an invariant
# the guard can genuinely check.
NA=$(body "$A" | wc -l)
NB=$(body "$B" | wc -l)
NROWS=$((NA < NB ? NA : NB))
paste <(body "$A" | head -n "$NROWS") <(body "$B" | head -n "$NROWS") | awk -v cols="$HA" '
BEGIN { n = split(cols, F, " ") }
{
    # NO EARLY `exit`, deliberately. awk exiting on the first difference closes the pipe under
    # `paste`, which takes SIGPIPE, which `pipefail` turns into exit 141 — this script reported
    # the right answer and a failure code at the same time on its first real use. The logs are
    # ~25k rows; reading them all costs nothing and keeps the exit status honest.
    if (found || bad) next
    if (NF != 2 * n) {
        printf "FAIL: row %d has %d fields but the header names %d columns\n", NR, NF, n > "/dev/stderr"
        bad = 1
        next
    }
    for (i = 1; i <= n; i++) {
        if ($i != $(i + n)) {
            printf "FIRST DIVERGENCE at row %d:  %s=%s %s=%s %s=%s\n", NR, F[1], $1, F[2], $2, F[3], $3
            printf "  -> column %d (%s) moved first\n\n", i, F[i]
            printf "  %-8s %-20s %-20s\n", "column", "A", "B"
            for (k = 1; k <= n; k++)
                printf "  %-8s %-20s %-20s%s\n", F[k], $k, $(k + n), ($k != $(k + n) ? "   <-- DIFFERS" : "")
            found = 1
            break
        }
    }
}
END {
    if (bad) exit 2
    if (!found) print "no differing row in the common prefix — the shorter log is a prefix of the longer"
}'
