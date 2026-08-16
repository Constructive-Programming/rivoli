#!/usr/bin/env bash
# M15 exit gate: the V4 arm end to end, thin on purpose — smoke-glm.sh owns the server
# path and the clap-exclusivity cells (arch-independent), so this suite pins only what is
# V4's own: the --mtp refusal quotes V4's kernel-shaped reason; a --ctx past the old 2052
# ceiling loads, prefills ACROSS the boundary, and decodes finite tokens through the scored
# selection; and the sparse --attn values PROCEED loudly (the M15 legality flip) and
# provably toggle nothing (ids identical to the recorded dense run).
#
# **The ceiling cell runs BEFORE the ids cell, and the order is load-bearing.** The ids
# cell reds on a missing reference file, and `red` exits — so with the ids capture still
# owed, the original order meant the one cell that actually reaches the scored path never
# ran. The ids cell decodes at --ctx 2048, BELOW the boundary, where no indexer state is
# built at all; it pins the legality flip and the below-cap identity, not the selection.
#
#   tests/smoke-v4.sh <artifact-dir> [max-mem-GiB]
#
# GPU cells run under flock, serially, dev profile (correctness, not timing). Cost is
# ~20-30 min, dominated by one ~2.6k-token prefill at dev-profile speed. Exit 0 all cells
# green; first red exits 1 with the cell named.
set -euo pipefail

BIN=${SMOKE_BIN:-${CARGO_TARGET_DIR:-$(dirname "$0")/../target}/debug/rivoli}
ARTIFACT=${1:?usage: smoke-v4.sh <artifact-dir> [max-mem-GiB]}
MEM=${2:-100}
LOCK=/var/run/sys-gpu.lock
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/smoke-v4.XXXXXX")
# The recorded reference decode: captured at the PRE-M15 pin (264758c) with the same
# command cell 2 runs, then re-verified against the M15 tree before being committed —
# below the cap the scored arm is byte-identical by construction, so a drift here is a
# decode regression, full stop. Committed by the M15 GPU session; a missing file is RED,
# not a skip, so the cell cannot pass vacuously.
IDS_REF=$(dirname "$0")/v4-bench32-ctx2048.ids

[ -x "$BIN" ] || { echo "FAIL: rivoli binary missing: $BIN (cargo build first — rocm is default since the 2026-08-16 fuse)" >&2; exit 1; }
[ -e "$LOCK" ] || { echo "FAIL: GPU lock file missing: $LOCK" >&2; exit 66; }

pass=0
cell() { echo "== cell: $1"; }
ok() { pass=$((pass + 1)); echo "   ok: $1"; }
red() { echo "SMOKE RED in cell '$1': $2 — see $SCRATCH" >&2; exit 1; }

# --- refusal cell: no GPU, the legality check runs before any weight is placed ---------
# Quotes a fragment of core/legality.rs's V4_MTP_NEEDS_A_KERNEL; if the const is
# reworded at its source, reword it here — the fragment match keeps the cell from
# passing on some unrelated error.
cell "refuse: mtp"
rc=0
"$BIN" "$ARTIFACT" --mtp --bench 1 >"$SCRATCH/mtp.out" 2>"$SCRATCH/mtp.err" || rc=$?
[ "$rc" -ne 0 ] || red mtp "expected a refusal, got exit 0"
grep -q "missing KERNEL" "$SCRATCH/mtp.err" ||
    red mtp "refused (rc=$rc) but without V4's kernel-shaped reason"
ok "refused with the table's message"

# --- ceiling cell: the 2052 refusal is GONE and the scored selection decodes ----------
# The prompt is ~2.6k tokens of deterministic prose, so the PREFILL itself crosses the
# old boundary: past position 2052 every indexed layer's block set is decided by the
# indexer's scores, on all 21 indexed layers, in one whole-prompt pass — then every
# decode step scores again. Pre-M15 this exact invocation was refused at the door.
cell "ceiling gone: --ctx 4096 prefills across 2052 and decodes"
PROMPT=$(python3 - <<'EOF'
# 95 sentences (~2200 words) of varied, non-repeating prose -> comfortably past 2052
# tokens and comfortably under the 4095-row prompt bound; the NTOK assertions below
# catch either side drifting. Deterministic: no RNG, no date.
words = []
topics = ["memory", "bandwidth", "latency", "cache", "expert", "block", "stream",
          "kernel", "budget", "window", "index", "score", "token", "layer"]
for i in range(95):
    t = topics[i % len(topics)]
    words.append(f"Consider aspect {i} of {t} systems: the {t} path interacts with "
                 f"scheduling in ways that shift cost from {topics[(i+3)%len(topics)]} "
                 f"to {topics[(i+7)%len(topics)]} under load.")
print(" ".join(words))
EOF
)
# `--ctx 4096` is TYPED and not left to the CLI default, because both bounds below are
# derived from it — 4083 is `max_ctx - 1` minus the 12 decode steps.
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" --ctx 4096 \
    --prompt "$PROMPT" --bench 12 --dump-ids "$SCRATCH/ceiling.ids" \
    >"$SCRATCH/ceiling.out" 2>"$SCRATCH/ceiling.err" ||
    red ceiling "decode failed rc=$? — see ceiling.err (pre-M15 this refused at the door)"
# The boundary must actually have been CROSSED, or this cell is the vacuous-boundary
# trap: the engine's own PREFILL line carries the token count, so assert from it.
#
# **`|| true` inside every substitution, and it is the opposite of sloppiness.** Under
# `set -euo pipefail` a grep that matches NOTHING fails the pipeline, which fails the
# ASSIGNMENT, which exits the script — before the very `[ -n "$NTOK" ]` line whose job is
# to say the boundary could not be proven. The two anti-vacuity checks in this cell were
# unreachable in exactly the case they exist for; verified by running the shape standalone
# (exit 1, no message). Let the substitution come back empty and let the check speak.
NTOK=$(grep -oE 'PREFILL: [0-9]+ tokens' "$SCRATCH/ceiling.err" | grep -oE '[0-9]+' | head -1 || true)
[ -n "$NTOK" ] || red ceiling "no PREFILL line in stderr — cannot prove the boundary was crossed"
[ "$NTOK" -ge 2053 ] || red ceiling "prompt tokenized to $NTOK < 2053 — grow the prompt, the cell is vacuous"
[ "$NTOK" -le 4083 ] || red ceiling "prompt tokenized to $NTOK, too close to the 4095-row bound"
NIDS=$(grep -cvE '^#' "$SCRATCH/ceiling.ids" || true)
[ "${NIDS:-0}" -eq 12 ] || red ceiling "expected 12 decoded ids, got ${NIDS:-0}"
ok "prefilled $NTOK tokens across the old ceiling and decoded 12/12 finite tokens"

# --- bench cell: below the cap, ids pinned, and the M15 legality flip in one run ------
# --attn dsa on purpose: pre-M15 this refused at the door; now it must WARN with the
# rewritten const and then decode ids IDENTICAL to the recorded dense run — the
# strongest form of "the flag toggles nothing".
cell "bench 32 @ ctx 2048, --attn dsa falls back loudly"
[ -f "$IDS_REF" ] || red bench "recorded reference ids missing: $IDS_REF — capture per the file's header note"
flock "$LOCK" "$BIN" "$ARTIFACT" --attn dsa --max-mem "$MEM" --ctx 2048 \
    --bench 32 --dump-ids "$SCRATCH/bench.ids" \
    >"$SCRATCH/bench.out" 2>"$SCRATCH/bench.err" ||
    red bench "decode failed rc=$? — see bench.err"
grep -q "toggles nothing" "$SCRATCH/bench.err" ||
    red bench "--attn dsa proceeded without the fallback warning"
# `|| true` for the reason the ceiling cell's does: an all-comment (i.e. EMPTY) id dump
# would otherwise kill the script silently instead of failing this comparison by name.
WANT=$(grep -vE '^#' "$IDS_REF" | tr '\n' ' ' || true)
GOT=$(grep -vE '^#' "$SCRATCH/bench.ids" | tr '\n' ' ' || true)
[ -n "$WANT" ] || red bench "recorded ids file has no id lines: $IDS_REF"
[ "$GOT" = "$WANT" ] || red bench "ids [$GOT] != recorded [$WANT]"
ok "warned, then decoded 32/32 ids identical to the recorded run"

echo "SMOKE GREEN: $pass cells | evidence: $SCRATCH"
