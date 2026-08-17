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
# `--divergence-log` writes one row per (pass, layer). The first row where the two runs disagree is
# the coordinate, and *which field disagrees there* names the subsystem — but the columns come in
# two kinds and they do NOT carry the same weight.
#
# CONSUMER-OUTPUT columns (xa, xn, h, ac, x) are what a kernel PRODUCED. The kernels are
# deterministic given their inputs, so equal output over equal other-inputs means the bytes the
# kernel actually consumed were equal. Walk them in pipeline order; the first that differs names
# the stage:
#
#   xa   the residual after attention   -> attention or its KV cache
#   xn   after the norm                 -> the norm (xa equal => attention is clear)
#   h    the SwiGLU intermediate        -> gate/up read WRONG BYTES
#   ac   the accumulator before drain   -> the DOWN projection read WRONG BYTES
#   x    the residual at layer exit     -> the drain or the residual add
#
# and the host columns, which say what the router saw and did:
#
#   gl differs -> the router's INPUT moved, upstream of routing
#   pk differs, gl equal -> routing consulted something outside its inputs (INV-1)
#   sl differs, pk equal -> the pool placed the same experts in different slots
#   misses / relocs differ -> the two runs made different residency DECISIONS
#
# BYTES-AT-AN-INSTANT columns (bh, sc, se) say what the payload looked like WHEN THE FOLD LOOKED,
# and they split a wrong-bytes verdict into WHICH HOP. Read them CROSS-RUN -- A's column against
# B's -- never against each other: sc folds the one slot just copied, se folds all ~9 the layer
# used.
#
#   bh   the bounce arena after the NVMe read    differs -> the DRIVE delivered different bytes
#   sc   the pool slot right after the copy      bh equal, sc differs -> the COPY
#   se   ALL the layer's slots, end of layer     bh+sc equal, se differs -> wrong AT REST and not
#                                                the slot just copied
#
# **THE ASYMMETRY, and it differs by kind.** A bytes-at-an-instant column AGREEING proves only that
# the bytes matched when it looked; a corruption landing between the fold and the consumer's read is
# invisible to it. **So no null on bh/sc/se exonerates a hop.** A consumer-output column agreeing is
# the stronger statement and does constrain what was consumed. Two recorded coordinates show why it
# matters: one where `h` differed (gate/up read wrong bytes) and one where `h` was IDENTICAL and
# only `x` moved -- the same mechanism reaching a different part of the slot, which is what `ac` was
# added to separate.
#
# A `-` is NOT MEASURED, never zero: the run did not enable that fold, or the layer had nothing for
# it (a dense layer has no h/ac/router; a layer with no miss has no bh/sc). A `~<hash>` is a PARTIAL
# fold -- sc-line covers every cache line but ~1/32 of the bytes, so its agreement is weaker. Both
# compare as real differences against a hash on the other side, because that means the two runs
# disagree about whether a quantity was measured at all.
#
# `se` folds every expert the layer used, each at its own index offset, so it is NOT blind to a
# resident expert nor to two payloads swapped between slots. (This block said the opposite --
# "covers only the experts the layer MISSED ... State that with any result" -- for one commit after
# `se` was widened to the union, i.e. it told every operator to attach a false caveat to every
# result. Corrected 2026-08-17.)
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
# THE FOLD CONFIGURATION IS PART OF THE COMPARISON'S VALIDITY, not metadata.
#
# The heavy probe (all three fetch-path folds) was measured to SUPPRESS the defect over 2,048
# tokens, so two logs taken under different `--divergence-folds` are not two samples of one
# experiment — they are two different experiments, and a `-` on one side against a hash on the other
# would read as a divergence in the payload. Refused.
# The fold config, DERIVED for older formats rather than refused.
#
# v4 states it on its own line. v2 and v3 do not need to: v2 predates the fetch-path folds
# entirely, so it is unambiguously `light`, and v3 had no way to disable them, so it is
# unambiguously all-on — which is the configuration measured to SUPPRESS the defect, and therefore
# the single most important thing to label. Deriving it keeps the historical logs (including the
# pair that produced the token-164 coordinate) readable while still refusing to compare across
# configurations.
# Returns ONLY the config, never its provenance: the note used to be inside the returned string, so
# a v3 log (derived "bh,sc,se (derived: …)") and a genuine v4 `--divergence-folds bh,sc,se` log —
# the SAME configuration — compared unequal and were refused. That defeated the derivation's own
# purpose, which is to keep the historical logs comparable. The provenance is printed separately.
folds() {
    local v line
    line=$(grep -m1 -E '^# rivoli-divergence-folds ' "$1" | sed -E 's/^# rivoli-divergence-folds //')
    if [ -n "$line" ]; then printf '%s' "$line"; return 0; fi
    v=$(grep -m1 -E '^# rivoli-divergence v[0-9]+ ' "$1" | sed -E 's/^# rivoli-divergence v([0-9]+) .*/\1/')
    case "$v" in
        2) printf 'light' ;;
        3) printf 'bh,sc,se' ;;
        *) printf '' ;;
    esac
}

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
FA=$(folds "$A"); FB=$(folds "$B")
# A v3-or-earlier log has no fold line at all. Refuse rather than assume: those were written by a
# build whose folds were ALL-ON, which is the suppressing configuration, and treating them as
# equal to a light log would compare an experiment against its own suppressor.
for f in "$A:$FA" "$B:$FB"; do
    [ -n "${f#*:}" ] || {
        echo "FAIL: ${f%%:*} declares no fold configuration and its version is not one this \
script can derive one from. Which folds were on decides what a difference MEANS — the all-on \
configuration suppresses the defect — so an unlabelled log cannot be compared." >&2
        exit 2
    }
done
[ "$FA" = "$FB" ] || {
    echo "FAIL: the two logs were taken under DIFFERENT fold configurations — two experiments, not two samples:" >&2
    echo "  A: $FA" >&2
    echo "  B: $FB" >&2
    exit 2
}
prov() { # where did the config come from?
    grep -qE '^# rivoli-divergence-folds ' "$1" && { printf 'declared'; return; }
    printf 'derived from the format version'
}
echo "   folds: $FA ($(prov "$A") / $(prov "$B"))"

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
