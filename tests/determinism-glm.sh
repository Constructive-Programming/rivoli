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
# ## LENGTH IS THE WHOLE POINT, and the floor is a POWER calculation
#
# The defect passes at 32 tokens. Two no-MTP runs and an --mtp run were all byte-identical over
# 32 ids on the same tree, same artifact, same box that produced 61/512 at 512 (inherited from
# `docs/investigations/glm-nondeterminism.md`, not re-derived here). A 32-token determinism gate
# is not a conservative gate, it is a GREEN ON A BROKEN ENGINE.
#
# That is not a threshold, it is a rate, and the rate is measured — from the four vendored
# teacher-forced arms, whose per-run first-divergence positions are 236, 362 and 375 with a
# fourth arm clean to 762. Two readings, and the gate is sized against the CONSERVATIVE one:
#
#   matched pair only (both arms at --max-mem 115)   2 events / 598   -> 1 per 299
#   all four arms, the clean one contributing its
#   full length as exposure                          3 events / 1735  -> 1 per 578
#
# A pair diverges if EITHER arm has an event, so P(detect) = 1 - exp(-2 * ngen / rate):
#
#            1/299        1/578
#     32       19%          10%
#    256       82%          59%
#    512       97%          83%     <- MIN_NGEN and the default
#
# 80% power needs ngen 241 at 1/299 and **465 at 1/578**, so the floor is 512 — the conservative
# reading rounded to the length every recorded measurement of this defect used. There is no knob
# below it, deliberately: a gate that can be dialled down to a length where it always passes is
# a gate someone will dial down.
#
# Re-derive all of it, and note that BOTH readings are printed so neither is buried:
#
#     tests/nll-divergence.sh docs/measurement/glm-divergence-evidence/{a,a2,b,ref}.nll
#     tests/nll-divergence.sh --power 2 598     # matched
#     tests/nll-divergence.sh --power 3 1735    # conservative
#
# **AND THE RATE ITSELF HAS A WIDE INTERVAL.** At k=3 the exact 95% Poisson bounds are 1 per
# [206, 2804], so 512 tokens gives between 31% and 99% power. A GREEN AT 512 BOUNDS THE RATE; IT
# DOES NOT PROVE DETERMINISM. State the length with every green. The rate also rose roughly
# thirtyfold under CPU/NFS load (452 vs 13, two n=1 runs — a ratio, not an interval), so a green
# on a loaded box bounds more than a green on a quiet one.
#
# That last point is about CPU and page-cache load, NOT about GPU tenancy: the witness below
# still DISCARDS any arm that shared the device, because a foreign GPU tenant changes what was
# computed and not merely when.
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
#
# `set -uo pipefail` WITHOUT `-e`, which DIFFERS from tests/parity-glm.sh's `set -euo` — an
# earlier comment here claimed it matched, and it does not (review 2026-08-17). The reason to
# differ: the id-census and the arm-rc checks below must REPORT a failed arm with its log and
# exit with this script's own code, and under `-e` an arm's non-zero rc, or a `grep` that matched
# nothing, would exit before reaching them.
set -uo pipefail

# RELEASE, not debug, and the default matters: every recorded measurement of this defect is a
# release run, the defect is timing-dependent (its rate rose ~30x under load), and a dev-profile
# green therefore does not bound the release binary at the same length. `DETERMINISM_BIN` exists
# because this gate must be able to test a binary it did not build and must never build one
# itself (a cargo run between arms evicts page cache); it is spelled like `PARITY_*` in the
# sibling gate rather than `RIVOLI_*`, which is the engine's own flat env namespace.
BIN=${DETERMINISM_BIN:-${CARGO_TARGET_DIR:-$(dirname "$0")/../target}/release/rivoli}
LOCK=/var/run/sys-gpu.lock
# The floor, argued in the header. A caller may raise `ngen`, never lower it past this.
MIN_NGEN=512

# THE PROMPT, RECORDED VERBATIM AND NOT REFERENCED.
#
# `docs/measurement/baseline-2026-08-16.md` records its command as `--prompt '<P>'` — a literal
# placeholder — and the text appears nowhere in that doc, its commit, or the tree, so that
# baseline's rows cannot be reproduced from their own record. A gate meant to outlive its authors
# must not repeat that, and a path into someone's scratchpad is the same mistake with extra steps.
#
# WHY THIS PROMPT AND NOT THE ENGINE'S DEFAULT: `--bench` with no `--prompt` uses "The sky is blue
# because", which hits EOS at 276 tokens, so a 512-token floor is UNREACHABLE and the gate refuses
# its own precondition (observed on device 2026-08-17). This is the essay prompt the wave
# standardised on; it is long-form enough that the model keeps generating.
PROMPT='Explain in depth how modern computer systems manage memory, from DRAM and the cache hierarchy through virtual memory, TLBs, cache coherence, NUMA, allocators, and garbage collection. For each mechanism, describe why it exists, the trade-offs it makes, and a concrete failure mode an engineer would meet in production.'

# PINNED, AND CHECKED ON EVERY RUN — not only under --self-test, which was the first version and
# never executed on the path it claimed to protect (review, 2026-08-17).
#
# The pin is not there to stop a silent edit: `git diff` already shows one. It is there so a
# RECORDED GREEN means something. Every green this gate prints cites the prompt's md5, and a green
# in a doc is only comparable to a later green if both ran the same text — the `corpus=<fnv>`
# discipline the .nll header already uses, applied to the one input this gate carries in-line.
# Costs one md5 of 317 bytes per run.
PLEN=$(printf '%s' "$PROMPT" | wc -c)
PMD5=$(printf '%s' "$PROMPT" | md5sum | cut -d' ' -f1)
if [ "$PLEN" != 317 ] || [ "$PMD5" != 18927a780b36b029d03450d2100e9242 ]; then
    echo "FAIL: the recorded prompt changed — $PLEN bytes, md5 $PMD5" >&2
    echo "      expected 317 bytes, md5 18927a780b36b029d03450d2100e9242" >&2
    echo "      Every green recorded by this gate was measured on the old text. If the change" >&2
    echo "      is intended, update BOTH numbers here in the same commit that edits it." >&2
    exit 2
fi

# --- the comparator, and its red proof ----------------------------------------------------
#
# One function, called by the gate and by --self-test. That is what makes the self-test a
# proof rather than a demonstration of a second implementation: the thing shown to redden IS
# the thing the gate runs.
compare_ids() { # $1 = arm A ids, $2 = arm B ids; 0 identical, 1 differ
    if cmp -s "$1" "$2"; then return 0; fi
    local n first
    # `-F'\t'` in BOTH awk calls, not the default FS. `paste` pads the exhausted side with an EMPTY
    # field, and under the default FS an empty leading field collapses — so when arm A is the
    # shorter one, awk reads B's id as `$1` and the excerpt prints it under the `A=` label with A's
    # count as the denominator. Reproduced by review 2026-08-17 (A=3, B=5: "2 of 3 differ, pos 3:
    # A=4 B=" when the truth is A had nothing there). Reporting a length divergence honestly is the
    # entire purpose of the branch that reaches this.
    n=$(paste "$1" "$2" | awk -F'\t' '$1 != $2' | wc -l)
    # 0-BASED, matching how every recorded measurement of this defect states its onset
    # ("first at position 13", "first at 452"): position 0 is the first generated token.
    first=$(paste "$1" "$2" | awk -F'\t' '$1 != $2 {print NR - 1; exit}')
    echo "RED: $n of $(wc -l <"$1") ids differ, first at position ${first:-?} (0-based)" >&2
    # `F` is 0-based and `NR` is 1-based, so the differing row is NR == F+1. Without the +1 the
    # first line printed is an IDENTICAL one, in the excerpt that is the gate's whole payload.
    paste "$1" "$2" | awk -F'\t' -v F="${first:-0}" 'NR >= F + 1 && NR <= F + 5 {printf "  pos %d: A=%s B=%s\n", NR - 1, $1, $2}' >&2
    return 1
}

# LENGTH IS CLASSIFIED BEFORE CONTENT, and the order is load-bearing.
#
# An earlier version checked each arm against $NGEN inside run_arm and exited 2. That mis-files the
# most interesting outcome there is: two arms that stop at DIFFERENT lengths have stopped at
# different EOS tokens, which IS the divergence — observed for real on 2026-08-17 (25,272 vs 22,386
# divergence-log rows from one pair) and reported as a setup error.
#
# In its own function so `--self-test` can exercise it. Reachable in the main flow only after two
# real 512-token GPU arms, so left inline it could never be shown red.
classify_lengths() { # $1 = A's ids, $2 = B's ids, $3 = ngen; echoes a verdict, returns its code
    local na nb
    na=$(wc -l <"$1"); nb=$(wc -l <"$2")
    if [ "$na" -ne "$nb" ]; then
        echo "RED: the two arms generated DIFFERENT LENGTHS ($na vs $nb ids)."
        echo "     A length difference is a divergence, not a setup problem: the arms reached EOS at"
        echo "     different points, so they had already stopped agreeing before then."
        return 1
    fi
    if [ "$na" -lt "$3" ]; then
        echo "FAIL (setup, not a verdict): both arms generated $na ids of the $3 requested, and the"
        echo "SAME number — so they agree, and the run simply hit EOS early. That is a prompt"
        echo "problem, not a determinism result: comparing two short arms of equal length would be a"
        echo "green over a decode that never happened, and $na is below the floor's power"
        echo "calculation anyway. Lengthen the recorded PROMPT at the top of this script."
        return 2
    fi
    return 0
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
    # THE LENGTH CLASSIFIER, exercised here because the main flow reaches it only after two real
    # 512-token GPU arms. All three branches, including the one that was WRONG (unequal lengths
    # were filed as a setup error when they are the divergence itself).
    printf '1\n2\n3\n' >"$d/l3"
    printf '1\n2\n3\n4\n' >"$d/l4"
    classify_lengths "$d/l3" "$d/l4" 4 >/dev/null && { echo "FAIL: unequal lengths must NOT pass" >&2; exit 2; }
    [ $? -eq 1 ] || { echo "FAIL: unequal lengths must be RED (1), not a setup error" >&2; exit 2; }
    classify_lengths "$d/l3" "$d/l3" 4 >/dev/null; [ $? -eq 2 ] || { echo "FAIL: equal-but-short must be a setup error (2)" >&2; exit 2; }
    classify_lengths "$d/l4" "$d/l4" 4 >/dev/null || { echo "FAIL: equal-and-full must pass (0)" >&2; exit 2; }

    echo "SELF-TEST ok: the comparator reddens on a changed id and on a truncated stream;"
    echo "             the length classifier separates RED (unequal) from setup (equal but short);"
    echo "             the recorded prompt is $PLEN bytes, md5 $PMD5 (checked above, every run)"
    exit 0
fi

# `${1:?}` would exit 1, which this script's own table reserves for "gate RED" — a wrapper
# keying on the code would report nondeterminism for a missing argument (review 2026-08-17).
if [ $# -lt 1 ]; then
    echo "usage: determinism-glm.sh <artifact-dir> [ngen] [max-mem-GiB] | --self-test" >&2
    exit 2
fi
ARTIFACT=$1
NGEN=${2:-512}
MEM=${3:-115}

[ "$NGEN" -ge "$MIN_NGEN" ] 2>/dev/null || {
    cat >&2 <<EOF
FAIL: ngen=$NGEN is below the $MIN_NGEN floor.
Two arms of N tokens detect a rate r with probability 1-exp(-2rN). Against the conservative
measured rate of 1 per 578 token-forwards that is 10% at 32, 59% at 256, 83% at $MIN_NGEN — and
80% power needs 465. Below the floor a green rules out nothing, and this defect IS byte-identical
at 32 tokens on the very tree that fails at 512.
  re-derive: tests/nll-divergence.sh --power 3 1735
EOF
    exit 2
}
[ -x "$BIN" ] || { echo "FAIL: binary missing: $BIN (build it BEFORE the gate — this gate never builds)" >&2; exit 2; }
[ -e "$LOCK" ] || { echo "FAIL: GPU lock file missing: $LOCK" >&2; exit 66; }
[ -d "$ARTIFACT" ] || { echo "FAIL: artifact dir missing: $ARTIFACT" >&2; exit 2; }
# NO TRACE-BUILD PROBE, and its absence is deliberate. The obvious guard — grep `--help` for
# `--trace` — CANNOT FAIL: that flag is declared unconditionally in `crates/cli/src/main.rs`, so
# it is in every build's help and the probe would warn on every run. A guard that always fires
# teaches its reader to ignore it. Nothing in the binary distinguishes the feature set, so the
# rule stays written down instead: **do not run this gate on a `--features trace` binary.** If
# it ever needs enforcing, `--version` should print its feature set.

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/determinism-glm.XXXXXX")
echo "== determinism-glm | ngen=$NGEN mem=${MEM}GiB scratch=$SCRATCH"
echo "   bin: $(stat -c '%y' "$BIN") $BIN"

# --- contention witness, per arm ----------------------------------------------------
# The contention witness lives in ONE file, sourced: the flock is advisory, peers skip it,
# and the failure mode of a two-copy false-green guard is that one copy stops guarding. See
# tests/gpu-witness.sh for the two traps each function encodes.
# shellcheck source=tests/gpu-witness.sh
. "$(dirname "$0")/gpu-witness.sh"

run_arm() { # $1 = arm name, $2 = ids out path
    local wfile="$SCRATCH/witness-$1" wpid apid gtt rc=0
    gtt=$(gtt_used)
    # 1 GiB, not 2: the one recorded ghost tenant held 1.6 GB (llama-swap, zero KFD entries —
    # see tests/gpu-witness.sh), so a 2 GiB threshold would have waved through the exact tenant
    # the guard was written for. 1 GiB sits below it and above this box's idle ~18 MiB.
    if [ "$gtt" -gt $((1 << 30)) ]; then
        echo "DISCARD arm '$1': $((gtt >> 20)) MiB GTT already held before the arm started — a ghost tenant KFD cannot see" >&2
        exit 3
    fi
    : >"$wfile"
    # Both arms take the SAME argument list, which is the entire experiment. Anything that
    # differs between them — including the ids path, which is why it is not a flag the engine
    # reads — would make this a comparison of two configurations instead of a repeatability
    # test.
    flock "$LOCK" "$BIN" "$ARTIFACT" --bench "$NGEN" --mode int3-vq --attn dense \
        --max-mem "$MEM" --prompt "$PROMPT" --dump-ids "$2" >"$SCRATCH/$1.out" 2>"$SCRATCH/$1.log" &
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
}

run_arm a "$SCRATCH/a"
run_arm b "$SCRATCH/b"

NA=$(wc -l <"$SCRATCH/a.ids")
NB=$(wc -l <"$SCRATCH/b.ids")

if ! classify_lengths "$SCRATCH/a.ids" "$SCRATCH/b.ids" "$NGEN" >"$SCRATCH/verdict"; then
    rc=$?
    cat "$SCRATCH/verdict" >&2
    if [ "$rc" -eq 1 ]; then
        compare_ids "$SCRATCH/a.ids" "$SCRATCH/b.ids" || true
        echo "  arm A ids: $SCRATCH/a.ids" >&2
        echo "  arm B ids: $SCRATCH/b.ids" >&2
    fi
    exit "$rc"
fi

if compare_ids "$SCRATCH/a.ids" "$SCRATCH/b.ids"; then
    echo "GREEN: two runs at identical flags produced identical ids over $NGEN tokens (mem=${MEM}GiB)."
    echo "       binary: $BIN"
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
