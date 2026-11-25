#!/usr/bin/env bash
# S6 exit gate: Inkling 2 Small (`inkling2`, the MXFP8 member) end to end, on the SAME contract
# shape as tests/smoke-inkling.sh — the legality refusals fire with the table's own wording, the
# bench path decodes and is budget-invariant, and the server answers all three endpoints over a
# live decode.
#
#   tests/smoke-inkling-small.sh [--expect-red[=FRAGMENT]] <artifact-dir> [max-mem-GiB]
#
# `tests/smoke-inkling.sh` runs the Mini arm; this one runs Small, whose differences are exactly
# the routed pool's: the experts are native `.mxfp8` streamed through `RoutedPool`, so
# `--cache-policy`, `--max-mem`, `--ctx` and `--trace` are REAL choices here, `--mode` falls back
# loudly to MXFP8, and the refusals that remain are the text stack's (sparse attention, the draft
# stack) plus clap's own exclusivity.
#
# **Red-proofing a refusal fragment.** Plant a WRONG fragment in one `refuse` cell below (say
# `refuse mtp "NOT-THE-TABLE-S-WORDING" --mtp --bench 1`) and run
# `tests/smoke-inkling-small.sh --expect-red='without the expected message' <artifact-dir>`. The
# gate must exit 0 reporting that cell RED; if every cell comes out green the classifier has
# stopped seeing the plant and the run fails, because a gate that cannot go red is not evidence.
# Revert the plant afterwards — the same `red()` classifier serves both modes, which is what
# makes the proof a proof of THIS gate rather than of a bespoke reading.
#
# **P4 is the bench cell's real content.** Small's routed set is ~600 GB, so `--max-mem` decides
# how much of it is pinned and the deficit is streamed from NVMe. Two arms at the SAME prompt with
# DIFFERENT budgets must emit the SAME ids: a token that moves when only residency changed is a
# pool defect (a slot refilled under a live launch, an evicted expert decoded from a stale block),
# not a numerics difference. The A-vs-A rerun pins the other half — the arm's own determinism —
# so a mismatch can be attributed rather than argued about.
#
# GPU cells run under flock, serially, dev profile (correctness, not timing). Exit 0 all cells
# green; first red exits 1 with the cell named.
set -euo pipefail

EXPECT_RED="" EXPECT_FRAG=""
case "${1:-}" in
    --expect-red=*) EXPECT_RED=1 EXPECT_FRAG=${1#--expect-red=}; shift ;;
    --expect-red) EXPECT_RED=1; shift ;;
esac

BIN=${SMOKE_BIN:-${CARGO_TARGET_DIR:-$(dirname "$0")/../target}/debug/rivoli}
ARTIFACT=${1:?usage: smoke-inkling-small.sh [--expect-red[=FRAGMENT]] <artifact-dir> [max-mem-GiB]}
MEM=${2:-100}
# The tight arm's budget. It must sit ABOVE the always-resident floor and BELOW
# floor + the routed set (~600 GB), so the pool pins a small prefix and streams the rest.
# Small's floor is the embed + lm_head (~3.7 GB bf16) plus ~8 GB of dense projections plus the KV
# slab, so ~13 GB is the boundary and 24 GiB clears it while pinning ~350 of the 18,432 experts
# (one slot is `mxfp8_expert_stride(4608, 2304)` = 32,845,824 B). `SMOKE_TIGHT_MEM` overrides for
# an artifact whose dense share differs; a budget below the floor reddens this cell with the
# pin's own arithmetic, which is the loud failure rather than a silent one.
TIGHT=${SMOKE_TIGHT_MEM:-24}
LOCK=/var/run/sys-gpu.lock
source "$(dirname "$0")/scratch.sh"  # never /tmp: tests/scratch.sh
SCRATCH=$(scratch_dir smoke-inkling-small)
PORT=${SMOKE_PORT:-18175}

# PINNED_IDS — first-party Small greedy decode ids, for an operator who has measured them. No
# value ships by default: the first-party Small decode needs the 619 GB checkpoint, and a gate
# that refuses to run until that lands is not a gate this tree can hold green. The budget-invariant
# and A-vs-A cells below are the pin this milestone CAN carry; `SMOKE_PIN_IDS="a b c d"` adds the
# absolute pin on top without editing the file.
PINNED_IDS=${SMOKE_PIN_IDS:-}

[ -x "$BIN" ] || { echo "FAIL: rivoli binary missing: $BIN (cargo build first — rocm is default since the 2026-08-16 fuse)" >&2; exit 1; }
[ -e "$LOCK" ] || { echo "FAIL: GPU lock file missing: $LOCK" >&2; exit 66; }

pass=0
cell() { echo "== cell: $1"; }
ok() { pass=$((pass + 1)); echo "   ok: $1"; }
# The ONE classifier, green run and --expect-red run alike. Under --expect-red the cell is the
# proof: the red must be the one planted (`EXPECT_FRAG`), and a red for the wrong reason fails.
red() { # $1 cell name, $2 message
    if [ -n "$EXPECT_RED" ]; then
        if [ -n "$EXPECT_FRAG" ] && ! printf '%s' "$2" | grep -qF -- "$EXPECT_FRAG"; then
            echo "RED-PROOF FAILED: '$1' went red on '$2', which does not contain '$EXPECT_FRAG' — a red for the wrong reason is not the proof you planted." >&2
            exit 4
        fi
        echo "   RED (as the proof demanded) in '$1': $2"
        exit 0
    fi
    echo "SMOKE RED in cell '$1': $2 — see $SCRATCH" >&2
    exit 1
}

# --- refusal cells: no GPU, the legality check runs before any weight is placed ------
# The fragments quote crates/core/legality.rs's message consts (Small's row) and clap's own
# "cannot be used with". If a message is reworded at its source, reword it here — a fragment
# match keeps a cell from passing on some unrelated error.
refuse() { # $1 cell name, $2 expected stderr fragment, rest = args
    local name=$1 frag=$2; shift 2
    cell "refuse: $name"
    local rc=0
    "$BIN" "$ARTIFACT" "$@" >"$SCRATCH/$name.out" 2>"$SCRATCH/$name.err" || rc=$?
    [ "$rc" -ne 0 ] || red "$name" "expected a refusal, got exit 0"
    grep -q "$frag" "$SCRATCH/$name.err" ||
        red "$name" "refused (rc=$rc) but without the expected message '$frag'"
    ok "refused with the table's message"
}

# Small's attention is the text stack's hybrid of sliding-window and full layers — not a row
# selection — so the three sparse values refuse while `dense` is what its full layers genuinely do.
refuse dsa "does not offer the choice" --attn dsa --bench 1
refuse streaming "does not offer the choice" --attn streaming --bench 1
refuse misa "does not offer the choice" --attn misa --bench 1
# The draft stack is excluded by name on BOTH members.
refuse mtp "DRAFT STACK IS NOT IN THE ARTIFACT" --mtp --bench 1
# clap-level exclusivity — the `requires` spelling was measured inert (clap 4.6.6), so these
# cells watch the conflicts_with replacement stay alive.
refuse bench-vs-port "cannot be used with" --bench 1 --port "$PORT"
refuse think-needs-port "cannot be used with" --bench 1 --think

# --- cache cell: the policy IS a real choice on this arm, so it must be accepted ------
# Mini refuses `--cache-policy` (fixed layer prefix); Small's pool evicts by policy, so a refusal
# here would be the S5 row leaking into S6. One token is enough — the cell is the door, not
# the decode.
cell "cache-policy 2q"
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" --ctx 512 --cache-policy 2q \
    --prompt "Hi" --bench 1 --dump-ids "$SCRATCH/policy.ids" \
    >"$SCRATCH/policy.out" 2>"$SCRATCH/policy.err" ||
    red cache-policy "--cache-policy 2q refused rc=$? — see policy.err"
[ "$(grep -vE '^#' "$SCRATCH/policy.ids" | wc -l)" -eq 1 ] ||
    red cache-policy "--cache-policy 2q produced no id — the run did not decode"
ok "--cache-policy 2q accepted and decoded"

# --- fallback cell: --mode is a LOUD fallback, not a refusal, and the user must be told ---
# Small's artifact stores its experts as native MXFP8, so `--mode int4` names a format the run
# does not store. The run proceeds — and the warning on stderr is the whole contract: a silent
# degrade is how a benchmark ends up measuring something other than its command line.
cell "fallback-loudly: --mode int4"
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" --ctx 512 --mode int4 \
    --prompt "Hi" --bench 1 --dump-ids "$SCRATCH/int4.ids" \
    >"$SCRATCH/int4.out" 2>"$SCRATCH/int4.err" ||
    red int4 "--mode int4 decode failed rc=$? — see int4.err"
grep -qF "repacked byte-for-byte into .mxfp8" "$SCRATCH/int4.err" ||
    red int4 "--mode int4 produced no MXFP8 fallback warning on stderr — see int4.err"
grep -qF ".mxfp8" "$SCRATCH/int4.err" ||
    red int4 "the fallback warning does not name the .mxfp8 container — see int4.err"
[ "$(grep -vE '^#' "$SCRATCH/int4.ids" | wc -l)" -eq 1 ] ||
    red int4 "--mode int4 produced no id — the run did not decode"
ok "--mode int4 accepted with the MXFP8 fallback warning"

# --- bench cells: decode, then the two invariants smaller than a reference pin can carry ---
# One budget, 4 tokens: the ids cell.
cell "bench dense x mxfp8"
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" --ctx 512 --attn dense \
    --prompt "Hi" --bench 4 --dump-ids "$SCRATCH/bench.ids" \
    >"$SCRATCH/bench.out" 2>"$SCRATCH/bench.err" ||
    red bench "decode failed rc=$? — see bench.err"
GOT=$(grep -vE '^#' "$SCRATCH/bench.ids" | tr '\n' ' ')
[ "$(echo "$GOT" | wc -w)" -eq 4 ] ||
    red bench "expected 4 generated ids, got [$GOT]"
if [ -n "$PINNED_IDS" ]; then
    [ "$GOT" = "${PINNED_IDS% } " ] ||
        red bench "ids [$GOT] != pinned [$PINNED_IDS]"
    ok "4/4 ids match the pinned reference decode"
else
    ok "4 ids decoded (no absolute pin set; SMOKE_PIN_IDS pins one)"
fi

# A-vs-A: the SAME flags twice. GLM does not repeat itself at the byte level, so this cell is
# what tells a later mismatch between budgets apart from the arm's own run-to-run drift.
cell "determinism A-vs-A"
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" --ctx 512 --attn dense \
    --prompt "Hi" --bench 4 --dump-ids "$SCRATCH/bench-a2.ids" \
    >"$SCRATCH/bench-a2.out" 2>"$SCRATCH/bench-a2.err" ||
    red determinism "rerun failed rc=$? — see bench-a2.err"
GOT2=$(grep -vE '^#' "$SCRATCH/bench-a2.ids" | tr '\n' ' ')
[ "$GOT2" = "$GOT" ] ||
    red determinism "ids moved between identical runs: [$GOT] then [$GOT2] — a comparison below this floor is not a measurement"
ok "identical ids across two runs at identical flags"

# P4: a budget that pins a prefix and streams the rest must not move a single token. The budget
# is the only knob that changes; the prompt, the flags and the model are the same.
cell "p4: tight budget, identical ids"
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$TIGHT" --ctx 512 --attn dense \
    --prompt "Hi" --bench 4 --dump-ids "$SCRATCH/bench-tight.ids" \
    >"$SCRATCH/bench-tight.out" 2>"$SCRATCH/bench-tight.err" ||
    red p4 "tight-budget (--max-mem $TIGHT) decode failed rc=$? — see bench-tight.err"
GOT3=$(grep -vE '^#' "$SCRATCH/bench-tight.ids" | tr '\n' ' ')
[ "$GOT3" = "$GOT" ] ||
    red p4 "ids moved when only residency changed: [--max-mem $MEM] $GOT vs [--max-mem $TIGHT] $GOT3 — P4 says bytes/token may move, TEXT must not"
ok "P4: identical ids at --max-mem $MEM and --max-mem $TIGHT"

# --- serve cell: readiness contract + all three endpoints over a live decode --------
cell "serve"
setsid flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" \
    --ctx 512 --port "$PORT" >"$SCRATCH/serve.out" 2>"$SCRATCH/serve.err" &
SPID=$!
# Any red between here and the kill would otherwise orphan a server that holds BOTH the GPU and
# the flock, wedging every later flock on this machine (review 2026-08-16).
trap 'kill -- -"$SPID" 2>/dev/null || true' EXIT
# The port opens only once the model is loaded — that IS the readiness signal, so polling it is
# the contract under test, not a workaround. Dev-profile load takes minutes; 20 min is generous
# headroom, and a dead server ends the wait early.
up=""
for _ in $(seq 240); do
    kill -0 "$SPID" 2>/dev/null || red serve "server exited during load — see serve.err"
    if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >"$SCRATCH/health.json" 2>/dev/null; then
        up=1; break
    fi
    sleep 5
done
[ -n "$up" ] || red serve "port never opened within 20 min"
ok "readiness: port opened after load, /health 200"

curl -sf --max-time 10 "http://127.0.0.1:$PORT/v1/models" >"$SCRATCH/models.json" ||
    red serve "/v1/models failed"
grep -q '"id"' "$SCRATCH/models.json" || red serve "/v1/models carries no model id"
ok "/v1/models lists the model"

curl -sf --max-time 1200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4}' \
    >"$SCRATCH/chat.json" || red serve "non-stream completion failed"
# Inkling may legitimately spend all 4 tokens in `<|content_thinking|>`, so `content` alone is
# not the contract: the reply must arrive in ONE of the two channels.
python3 -c "
import json
d = json.load(open('$SCRATCH/chat.json'))
msg = d['choices'][0]['message']
text = msg.get('content') or msg.get('reasoning_content')
assert text, 'neither content nor reasoning_content'
print('   reply:', repr(text))
" || red serve "non-stream body malformed — see chat.json"
ok "non-stream completion returned content or reasoning_content"

# Inkling emits a short `<|content_thinking|>` block even at `Thinking effort level: 0` before
# opening `<|content_text|>` (measured 2026-11-25 on the first-party Mini checkpoint). Give 64
# tokens so the serve splitter is exercised on BOTH `reasoning_content` and `content` live —
# Small shares the text stack and the template, so the split is the same code.
curl -sf --max-time 1200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with one word: hi"}],"max_tokens":64,"enable_thinking":false}' \
    >"$SCRATCH/chat-nothink.json" || red serve "non-stream (enable_thinking=false) failed"
python3 -c "
import json
d = json.load(open('$SCRATCH/chat-nothink.json'))
msg = d['choices'][0]['message']
c = msg.get('content')
assert c, f'expected content in 64 tokens, got {msg!r}'
print('   content:', repr(c), 'reasoning:', repr(msg.get('reasoning_content')))
" || red serve "enable_thinking=false body malformed — see chat-nothink.json"
ok "enable_thinking=false returns content (channel split live)"

curl -sfN --max-time 1200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":2,"stream":true}' \
    >"$SCRATCH/stream.sse" || red serve "stream completion failed"
grep -q '^data: ' "$SCRATCH/stream.sse" || red serve "no SSE data frames"
grep -q 'data: \[DONE\]' "$SCRATCH/stream.sse" || red serve "stream never sent [DONE]"
ok "stream: SSE frames + [DONE] terminator"

kill -- -"$SPID" 2>/dev/null || true
wait "$SPID" 2>/dev/null || true
ok "server shut down"

if [ -n "$EXPECT_RED" ]; then
    echo "RED-PROOF FAILED: every cell came out green under --expect-red — the gate cannot see the planted defect. Debug the tree, not the proof." >&2
    exit 1
fi

echo "SMOKE GREEN: $pass cells | evidence: $SCRATCH"
