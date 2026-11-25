#!/usr/bin/env bash
# S5 exit gate: Inkling 2 Mini (`inkling2_loop`) end to end, on the SAME contract shape as
# tests/smoke-glm.sh — the legality refusals fire with the table's own wording, the bench path
# decodes, and the server answers all three endpoints over a live decode.
#
#   tests/smoke-inkling.sh [--expect-red[=FRAGMENT]] <artifact-dir> [max-mem-GiB]
#
# **Red-proofing a refusal fragment.** Plant a WRONG fragment in one `refuse` cell below (say
# `refuse mtp "NOT-THE-TABLE-S-WORDING" --mtp --bench 1`) and run
# `tests/smoke-inkling.sh --expect-red='without the expected message' <artifact-dir>`. The gate
# must exit 0 reporting that cell RED; if every cell comes out green the classifier has stopped
# seeing the plant and the run fails, because a gate that cannot go red is not evidence. Revert
# the plant afterwards — the same `red()` classifier serves both modes, which is what makes the
# proof a proof of THIS gate rather than of a bespoke reading.
#
# **What is deliberately NOT here.** No per-flag matrix: Mini's row of the legality table has
# exactly one supported matrix cell (dense attention, bf16 experts, fixed layer prefix), and the
# refusals below are the part that regresses silently — a flag that stops refusing is a scope cut
# nobody decided. Both members are covered by crates/core/src/legality/tests.rs; THIS script runs
# a Mini artifact only, so Small's row has no cell here until S6 gives it an arm.
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
ARTIFACT=${1:?usage: smoke-inkling.sh [--expect-red[=FRAGMENT]] <artifact-dir> [max-mem-GiB]}
MEM=${2:-100}
LOCK=/var/run/sys-gpu.lock
source "$(dirname "$0")/scratch.sh"  # never /tmp: tests/scratch.sh
SCRATCH=$(scratch_dir smoke-inkling)
PORT=${SMOKE_PORT:-18174}

# PINNED_IDS — first-party Mini greedy decode ids on "Hi" with reasoning_effort="none", measured
# 2026-11-25 via transformers `Inkling2LoopForConditionalGeneration.from_pretrained` @ d8f10a82:
# [200001, 200008, 371, 150409, 150239, 150330, 196980, 4896] (8/8 matched by rivoli Mini).
# `SMOKE_PIN_IDS` overrides without editing the file.
PINNED_IDS=${SMOKE_PIN_IDS:-"200001 200008 371 150409"}

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
# The fragments quote crates/core/legality.rs's message consts (Mini's row) and clap's own
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
    ok "refused with the table's own message"
}

# Mini's attention is hybrid sliding/full with relative bias and sinks — not a row selection,
# so the three sparse values refuse while `dense` is what its full layers genuinely do.
refuse dsa "does not offer the choice" --attn dsa --bench 1
refuse streaming "does not offer the choice" --attn streaming --bench 1
refuse misa "does not offer the choice" --attn misa --bench 1
# The block partition is a fixed layer prefix: the policy is not a choice either.
refuse cache-policy "fixed LAYER PREFIX" --cache-policy lru --bench 1
# No v2 expert-fetch trace on the bf16 arm.
refuse trace "expert-fetch trace" --trace "$SCRATCH/trace.bin" --bench 1
# The ONE cell that refuses even though the arm exists: the draft stack is excluded by name.
refuse mtp "DRAFT STACK IS NOT IN THE ARTIFACT" --mtp --bench 1
# clap-level exclusivity — the `requires` spelling was measured inert (clap 4.6.6), so these
# cells watch the conflicts_with replacement stay alive.
refuse bench-vs-port "cannot be used with" --bench 1 --port "$PORT"
refuse think-needs-port "cannot be used with" --bench 1 --think

# --- bench cell: the one supported cell, framed in Mini's own chat template ----------
cell "bench dense x bf16"
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" --ctx 512 \
    --prompt "Hi" --bench 4 --dump-ids "$SCRATCH/bench.ids" \
    >"$SCRATCH/bench.out" 2>"$SCRATCH/bench.err" ||
    red bench "decode failed rc=$? — see bench.err"
GOT=$(grep -vE '^#' "$SCRATCH/bench.ids" | tr '\n' ' ')
[ "$(echo "$GOT" | wc -w)" -eq 4 ] ||
    red bench "expected 4 generated ids, got [$GOT]"
if [ -n "$PINNED_IDS" ]; then
    [ "${GOT% }" = "${PINNED_IDS% }" ] ||
        red bench "ids [$GOT] != pinned [$PINNED_IDS]"
    ok "4/4 ids match the pinned reference decode"
else
    ok "4 ids produced, NOT PINNED (set PINNED_IDS once parity records them): [$GOT]"
fi

# --- fallback cell: --mode is a LOUD fallback, not a refusal, and the user must be told ---
# Mini's artifact stores its expert blocks DEQUANTIZED to bf16, so `--mode int4` names a format
# the run does not store. The run proceeds — and the warning on stderr is the whole contract: a
# silent degrade is how a benchmark ends up measuring something other than its command line.
cell "fallback-loudly: --mode int4"
flock "$LOCK" "$BIN" "$ARTIFACT" --max-mem "$MEM" --ctx 512 --mode int4 \
    --prompt "Hi" --bench 1 --dump-ids "$SCRATCH/int4.ids" \
    >"$SCRATCH/int4.out" 2>"$SCRATCH/int4.err" ||
    red int4 "--mode int4 decode failed rc=$? — see int4.err"
grep -q "DEQUANTIZED" "$SCRATCH/int4.err" ||
    red int4 "--mode int4 produced no DEQUANTIZED bf16 warning on stderr — see int4.err"
grep -q "bf16" "$SCRATCH/int4.err" ||
    red int4 "the fallback warning does not name the bf16 container — see int4.err"
[ "$(grep -vE '^#' "$SCRATCH/int4.ids" | wc -l)" -eq 1 ] ||
    red int4 "--mode int4 produced no id — the run did not decode"
ok "--mode int4 accepted with the DEQUANTIZED bf16 fallback warning"

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
# opening `<|content_text|>` (measured 2026-11-25 on the first-party checkpoint: `enable_thinking:
# false` emits ~25 thinking tokens then the answer). Give 64 tokens so the serve splitter is
# exercised on BOTH `reasoning_content` and `content` live.
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
