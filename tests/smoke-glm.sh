#!/usr/bin/env bash
# M6 exit gate: the CLI end to end — every legality refusal fires with the table's own
# message, the bench path decodes token-identically to the recorded reference run, and
# the server answers all three endpoints over a live decode. The old tree's
# smoke-matrix.sh covered mode x policy x attn cells; this tree's matrix is currently
# one supported cell (int3-vq x dense) plus a wall of refusals, and the REFUSALS are
# the part that regresses silently — a flag that stops refusing is a scope cut nobody
# decided (the clap `requires` defect this suite's --think cell pins was found live).
#
#   tests/smoke-glm.sh <artifact-dir> [max-mem-GiB]
#
# GPU cells run under flock, serially, dev profile (correctness, not timing). Cost is
# ~45 min, dominated by two prefills at dev-profile speed. Exit 0 all cells green;
# first red exits 1 with the cell named.
set -euo pipefail

BIN=${SMOKE_BIN:-${CARGO_TARGET_DIR:-$(dirname "$0")/../target}/debug/rivoli}
ARTIFACT=${1:?usage: smoke-glm.sh <artifact-dir> [max-mem-GiB]}
MEM=${2:-100}
LOCK=/var/run/sys-gpu.lock
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/smoke-glm.XXXXXX")
PORT=${SMOKE_PORT:-18173}

[ -x "$BIN" ] || { echo "FAIL: rivoli binary missing: $BIN (cargo build first — rocm is default since the 2026-08-16 fuse)" >&2; exit 1; }
[ -e "$LOCK" ] || { echo "FAIL: GPU lock file missing: $LOCK" >&2; exit 66; }

pass=0
cell() { echo "== cell: $1"; }
ok() { pass=$((pass + 1)); echo "   ok: $1"; }
red() { echo "SMOKE RED in cell '$1': $2 — see $SCRATCH" >&2; exit 1; }

# --- refusal cells: no GPU, the legality check runs before any weight is placed ------
# The legality cells quote fragments of core/legality.rs's message consts; the two
# clap-exclusivity cells quote clap's own "cannot be used with". If a message is
# reworded at its source, reword it here — a fragment match keeps a cell from passing
# on some unrelated error.
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

refuse hybrid "FormatPlan" --mode hybrid --bench 1
refuse dsa "only --attn dense decodes today" --attn dsa --bench 1
refuse streaming "only --attn dense decodes today" --attn streaming --bench 1
refuse mtp "deferred past parity" --mtp --bench 1
# clap-level exclusivity — the `requires` spelling was measured inert (clap 4.6.6),
# so these cells watch the conflicts_with replacement stay alive.
refuse bench-vs-port "cannot be used with" --bench 1 --port "$PORT"
refuse think-needs-port "cannot be used with" --bench 1 --think

# --- bench cell: the one supported matrix cell, ids pinned to the recorded run ------
# [13041, 1052, 0, 358] is the reference binary's own decode at the pin (6b7f496e,
# "Hi", recorded 2026-08-15 in docs/investigations/rewrite.md M4 and re-verified
# through this CLI the same night). A drift here is a decode regression, full stop.
cell "bench int3-vq x dense"
flock "$LOCK" "$BIN" "$ARTIFACT" --mode int3-vq --attn dense --max-mem "$MEM" \
    --prompt "Hi" --bench 4 --dump-ids "$SCRATCH/bench.ids" \
    >"$SCRATCH/bench.out" 2>"$SCRATCH/bench.err" ||
    red bench "decode failed rc=$? — see bench.err"
GOT=$(grep -vE '^#' "$SCRATCH/bench.ids" | tr '\n' ' ')
[ "$GOT" = "13041 1052 0 358 " ] ||
    red bench "ids [$GOT] != recorded [13041 1052 0 358]"
ok "4/4 ids match the recorded reference decode"

# --- serve cell: readiness contract + all three endpoints over a live decode --------
cell "serve"
setsid flock "$LOCK" "$BIN" "$ARTIFACT" --mode int3-vq --attn dense --max-mem "$MEM" \
    --port "$PORT" --ctx 4096 >"$SCRATCH/serve.out" 2>"$SCRATCH/serve.err" &
SPID=$!
# Any red between here and the kill would otherwise orphan a server that holds BOTH the
# GPU and the flock, wedging every later flock on this machine (review 2026-08-16).
trap 'kill -- -"$SPID" 2>/dev/null || true' EXIT
# The port opens only once the model is loaded — that IS the readiness signal, so
# polling it is the contract under test, not a workaround. Dev-profile load takes
# minutes; 20 min is generous headroom, and a dead server ends the wait early.
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
python3 -c "
import json, sys
d = json.load(open('$SCRATCH/chat.json'))
c = d['choices'][0]['message']['content']
assert c, 'empty content'
print('   content:', repr(c))
" || red serve "non-stream body malformed — see chat.json"
ok "non-stream completion returned content"

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

echo "SMOKE GREEN: $pass cells | evidence: $SCRATCH"
