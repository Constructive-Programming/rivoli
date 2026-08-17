#!/usr/bin/env bash
# M5 parity gate: the rewrite must greedy-decode token-for-token what the pinned
# reference decodes, same artifact, same prompt. Token IDS, never text — different id
# sequences can decode to identical text, so a text diff reports only a lower bound on
# divergence (the reference's own --dump-ids doc makes this argument).
#
#   green:      tests/parity-glm.sh <artifact-dir> [ngen] [max-mem-GiB]
#   red-proof:  tests/parity-glm.sh --red-proof <ref-ids-file> <artifact-dir> [ngen] [max-mem-GiB]
#
# Red-proof decodes the REWRITE arm against a shadow artifact whose codebooks.f32 has
# one byte flipped and demands the comparison against <ref-ids-file> goes red. A gate
# that cannot be shown red proves nothing (P7). The pristine-artifact control is the
# green run that produced <ref-ids-file>: same binary, same prompt, minutes earlier —
# red-proof does not re-decode it. The measurement discipline this script encodes
# (never builds, flock + witness per arm, GTT baseline, loud prompt-id extraction) is
# argued in place at each site below.
#
# Exit: 0 gate green (or red-proof confirmed red)
#       1 gate RED — token mismatch, or the rewrite arm failed where the reference decoded
#       2 setup error (reference arm failed, extraction failed, usage)
#       3 arm discarded (foreign GPU tenant witnessed) — rerun
#       66 GPU lock file missing (house convention: /run is tmpfs, dies on reboot)
set -euo pipefail

REF_BIN=${PARITY_REF_BIN:-/var/cache/users/rhansen/ref-pin-target/release/rivoli}
NEW_BIN=${PARITY_NEW_BIN:-${CARGO_TARGET_DIR:-$(dirname "$0")/../target}/debug/examples/glm_smoke}
PROMPT=${PARITY_PROMPT:-"The sky is blue"}
LOCK=/var/run/sys-gpu.lock
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/parity-glm.XXXXXX")

RED_PROOF=""
if [ "${1:-}" = "--red-proof" ]; then
    RED_PROOF=${2:?red-proof needs a green-run ref-ids file}
    [ -f "$RED_PROOF" ] || { echo "FAIL: ref-ids file not found: $RED_PROOF" >&2; exit 2; }
    shift 2
fi
ARTIFACT=${1:?usage: parity-glm.sh [--red-proof <ref-ids>] <artifact-dir> [ngen] [max-mem-GiB]}
NGEN=${2:-32}
MEM=${3:-100}

# This gate NEVER builds: a cargo run between the arms evicts page cache and poisons
# the comparison (measured in the old tree: ms/miss 1.36 -> 5.14). Both binaries must
# arrive prebuilt or the gate refuses.
[ -x "$REF_BIN" ] || { echo "FAIL: reference binary missing: $REF_BIN (build it elsewhere first — this gate never builds)" >&2; exit 2; }
[ -x "$NEW_BIN" ] || { echo "FAIL: rewrite binary missing: $NEW_BIN (cargo build --example glm_smoke BEFORE the gate — rocm is default since the 2026-08-16 fuse — never between arms)" >&2; exit 2; }
[ -e "$LOCK" ] || { echo "FAIL: GPU lock file missing: $LOCK" >&2; exit 66; }
[ -d "$ARTIFACT" ] || { echo "FAIL: artifact dir missing: $ARTIFACT" >&2; exit 2; }

# Staleness is a warning, not a red: mtime skew on NFS must not fail a real green.
STALE=$(find "$(dirname "$0")/../crates" -name '*.rs' -newer "$NEW_BIN" 2>/dev/null | head -3 || true)
[ -z "$STALE" ] || echo "WARN: sources newer than $NEW_BIN — a stale binary tests the wrong tree:" "$STALE" >&2

echo "== parity-glm | ngen=$NGEN mem=${MEM}GiB prompt=\"$PROMPT\" scratch=$SCRATCH"
echo "   ref: $(stat -c '%y' "$REF_BIN") $REF_BIN"
echo "   new: $(stat -c '%y' "$NEW_BIN") $NEW_BIN"

# --- contention witness -------------------------------------------------------------
# The contention witness lives in ONE file, sourced: the flock is advisory, peers skip it,
# and the failure mode of a two-copy false-green guard is that one copy stops guarding. See
# tests/gpu-witness.sh for the two traps each function encodes.
# shellcheck source=tests/gpu-witness.sh
. "$(dirname "$0")/gpu-witness.sh"

run_arm() { # $1 = arm name, $2 = stdout file, $3 = stderr file, rest = command.
    # The arm's streams are redirected HERE, on the flock'd command only, so DISCARD
    # diagnostics reach the terminal instead of being buried in the arm's log.
    local wfile="$SCRATCH/witness-$1" wpid apid gtt rc=0
    gtt=$(gtt_used)
    if [ "$gtt" -gt $((2 << 30)) ]; then
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

# --- arm B, shared by green and red-proof -------------------------------------------
decode_new() { # $1 = artifact dir, $2 = out ids file; prompt ids read from $SCRATCH/prompt-ids
    # Returns the arm's rc — the CALLER classifies it (green: red gate; red-proof: a
    # loud refusal is itself a red). The stdout-purity and non-empty checks run only on
    # a successful decode: they gate the id-stream contract, not the failure path.
    local rc=0
    # shellcheck disable=SC2046 # the prompt ids are one-per-line integers by construction
    run_arm new "$2" "$SCRATCH/new.log" "$NEW_BIN" "$1" "$MEM" "$NGEN" \
        $(cat "$SCRATCH/prompt-ids") || rc=$?
    [ "$rc" -eq 0 ] || return "$rc"
    if grep -qvE '^[0-9]+$' "$2"; then
        echo "FAIL: non-id line on rewrite stdout (logs belong on stderr):" >&2
        grep -vE '^[0-9]+$' "$2" | head -3 >&2
        exit 2
    fi
    [ -s "$2" ] || { echo "FAIL: rewrite arm exited 0 but produced no ids — see $SCRATCH/new.log" >&2; exit 2; }
}

compare() { # $1 = expected (reference) ids, $2 = actual (rewrite) ids
    # The reference chat-frames its prompt, so its EOS is reachable and it may stop
    # before ngen; the smoke has no tokenizer yet (M6) and decodes with an empty EOS
    # set. Parity therefore means: the reference's WHOLE sequence is a prefix of the
    # rewrite's. A rewrite that produced fewer ids than the reference is a real red.
    local n
    n=$(wc -l <"$1")
    if [ "$(wc -l <"$2")" -lt "$n" ]; then
        echo "MISMATCH: rewrite produced $(wc -l <"$2") ids, reference $n" >&2
        return 1
    fi
    head -n "$n" "$2" >"$2.trunc"
    if cmp -s "$1" "$2.trunc"; then
        echo "PARITY: $n generated ids identical (reference full length)"
        return 0
    fi
    echo "MISMATCH: first divergence —" >&2
    diff "$1" "$2.trunc" | head -8 >&2
    return 1
}

# --- red-proof: perturbed shadow artifact through arm B only ------------------------
if [ -n "$RED_PROOF" ]; then
    grep -vE '^#' "$RED_PROOF" >"$SCRATCH/ref-ids" || true
    [ -s "$SCRATCH/ref-ids" ] || { echo "FAIL: no ids in $RED_PROOF" >&2; exit 2; }
    # ngen must cover the reference sequence, or compare()'s length check fires on the
    # SHORTFALL and the proof "reddens" without the perturbed codebook mattering.
    NREF=$(wc -l <"$SCRATCH/ref-ids")
    [ "$NGEN" -ge "$NREF" ] || { echo "FAIL: ngen=$NGEN < $NREF reference ids — a short run reddens vacuously" >&2; exit 2; }
    if [ -s "$(dirname "$RED_PROOF")/prompt-ids" ]; then
        cp "$(dirname "$RED_PROOF")/prompt-ids" "$SCRATCH/prompt-ids"
    else
        echo "FAIL: no non-empty prompt-ids file next to $RED_PROOF (a green run writes it; an empty one would redden on 'empty prompt', not on the perturbation)" >&2
        exit 2
    fi
    SHADOW="$SCRATCH/shadow-artifact"
    mkdir "$SHADOW"
    # realpath: a relative <artifact-dir> would otherwise mint links that resolve
    # relative to the shadow dir, dangle, and turn the red-proof into a vacuous
    # cannot-open refusal that never touched the perturbed byte.
    for f in "$ARTIFACT"/*; do ln -s "$(realpath "$f")" "$SHADOW/$(basename "$f")"; done
    rm "$SHADOW/codebooks.f32"
    cp "$ARTIFACT/codebooks.f32" "$SHADOW/codebooks.f32"
    # Sign-flip the ENTIRE first (gate-projection) codebook — finite wrong VALUES, not
    # a crafted NaN the engine would reject before any arithmetic ran. The scope is
    # measured, not chosen (2026-08-15, both smaller attempts decoded 8 tokens
    # IDENTICALLY): a 1-ulp f32 flip is annihilated by the pin's f32->fp16 narrowing
    # before it reaches the device, and a single codeword sign flip (0.23 -> -0.23,
    # fp16-survivable) touches ~0.05% of one projection's weights and dilutes below
    # greedy-argmax margins across 6144-wide dot products. Wrong arithmetic that the
    # gate CAN see must clear those margins; every gate projection inverting does.
    python3 - "$SHADOW/codebooks.f32" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
for i in range(3, len(b) // 3, 4):  # high byte of each f32 in codebook 0 of 3
    b[i] ^= 0x80
open(p, "wb").write(b)
EOF
    rc=0
    decode_new "$SHADOW" "$SCRATCH/new-ids" || rc=$?
    if [ "$rc" -ne 0 ]; then
        # Print WHAT refused so the recorded evidence attributes the red — a load
        # error here would be a broken proof, not a confirmed one.
        echo "RED-PROOF OK: perturbed decode refused loudly (rc=$rc); last lines of $SCRATCH/new.log:"
        tail -3 "$SCRATCH/new.log"
        exit 0
    fi
    if compare "$SCRATCH/ref-ids" "$SCRATCH/new-ids" >/dev/null 2>&1; then
        echo "RED-PROOF FAILED: one flipped codebook byte and the ids still match — the gate cannot see divergence" >&2
        exit 1
    fi
    echo "RED-PROOF OK: perturbed artifact diverged — the gate can go red"
    exit 0
fi

# --- green: both arms, reference first ----------------------------------------------
rc=0
run_arm ref "$SCRATCH/ref.stdout" "$SCRATCH/ref.log" \
    "$REF_BIN" "$ARTIFACT" --mode int3-vq --attn dense --no-mtp \
    --max-mem "$MEM" --prompt "$PROMPT" --bench "$NGEN" --dump-ids "$SCRATCH/ref-dump" || rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: reference arm rc=$rc — see $SCRATCH/ref.log and ref.stdout" >&2; exit 2; }

# Prompt ids from the reference's own tokenizer log line (it logs on stdout, so both
# streams are searched):
#   tokenizer: prompt "..." -> N tokens chat-framed [a, b, c]; eos=[...]
# The log prints AT MOST 12 ids, so the extracted count is asserted against the
# advertised count — a truncated list fails loudly instead of silently comparing a
# shorter prompt.
ADVERTISED=$(grep -h -oE '\-> [0-9]+ tokens chat-framed' "$SCRATCH/ref.log" "$SCRATCH/ref.stdout" | head -1 | grep -oE '[0-9]+' || true)
grep -h -oE 'chat-framed \[[0-9, ]+\]' "$SCRATCH/ref.log" "$SCRATCH/ref.stdout" | head -1 |
    grep -oE '[0-9]+' >"$SCRATCH/prompt-ids" || true
GOT=$(wc -l <"$SCRATCH/prompt-ids")
if [ -z "$ADVERTISED" ] || [ "$GOT" -ne "$ADVERTISED" ]; then
    echo "FAIL: extracted $GOT prompt ids but the reference framed ${ADVERTISED:-none} — the log prints at most 12; use a prompt that chat-frames to <=12 tokens" >&2
    exit 2
fi
grep -vE '^#' "$SCRATCH/ref-dump" >"$SCRATCH/ref-ids" || true
[ -s "$SCRATCH/ref-ids" ] || { echo "FAIL: reference dumped no ids — see $SCRATCH/ref.log" >&2; exit 2; }

rc=0
decode_new "$ARTIFACT" "$SCRATCH/new-ids" || rc=$?
[ "$rc" -eq 0 ] || { echo "RED: rewrite arm failed (rc=$rc) where the reference decoded — see $SCRATCH/new.log" >&2; exit 1; }
compare "$SCRATCH/ref-ids" "$SCRATCH/new-ids" || exit 1
echo "   evidence: $SCRATCH (ref-ids + prompt-ids feed --red-proof)"
