#!/usr/bin/env bash
# Build the launcher-row PLANTED bit_identity binaries — one per row of PLANTS below — so a lease
# can judge each with `tests/bit-identity.sh --expect-red=ROWS <binary>`. Never touches the device.
#
#   CARGO_TARGET_DIR=/var/cache/rivoli/target/<checkout> tests/bit-identity/build-launch-plants.sh [NAME ...]
#
# Each plant = `git archive HEAD` + its patch(es) from `plants/launch/` (repo-root paths, applied
# with exact context), built in a scratch copy into `$CARGO_TARGET_DIR-plant`, so the checkout and
# its own target dir are never touched. After each build the patches are reverted and the copy is
# `diff -r`-checked against a pristine archive, so a plant that changed nothing, failed to apply,
# or failed to revert stops the script. Output: `<scratch>/bin/bit_identity-<name>` and a
# `manifest` of `name|expected-red-rows|sha256-prefix` lines ending in DONE.
#
# Brought in-tree 2026-11-23 from the Phase 1 integration's scratch script: until then the 15
# launcher-row plants could be rebuilt only from a file outside the repo, and they sat beside the
# prelude PROBE plants at the plants root, where `--build-probes --plant=moe` would build a probe
# object no row reads. The probe plants (judged by `--plant=FAMILY`) stay at the root.
#
# The table is the population: `crates/cli/tests/plants.rs` holds it equal to `plants/launch/`,
# both ends, and dry-runs every patch against the tree, so a stale or unlisted plant is a red
# deviceless test rather than a refused build on the next lease.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
source "$ROOT/tests/scratch.sh"
[ -n "${CARGO_TARGET_DIR:-}" ] || { echo "FAIL: set CARGO_TARGET_DIR to this checkout's own target dir" >&2; exit 2; }
T="$CARGO_TARGET_DIR-plant"

# name|patches under plants/launch/ (space-separated)|rows the plant must redden, and only those
PLANTS="linalg|linalg.patch|gemv_f32,gemv_i8
fwd|fwd.patch|sigmoid_gate
headtail|headtail.patch|hc_head_collapse
l2|moe.patch moe_f4.patch recurrent.patch|gated_delta_recurrent_f32,gdn_recurrent_f32,moe_expert_range,moe_expert_range_i4,moe_expert_range_f4_situ
indexer|indexer.patch|index_topk
blockindex|blockindex.patch|index_score_blocks
kvcompress|kvcompress.patch|gemm_bf16
mla|mla.patch|mla_absorb_fp8
attn-mla|attn-mla.patch|mla_attend
attn-gqa|attn-gqa.patch|gqa_attend,gqa_block_attend
attn-dense|attn-dense.patch|gather_attn_shared_kv,mha_attend
grid-y|grid-y.patch|mla_attend,gather_attn_shared_kv,gqa_attend,gqa_block_attend
rope|rope.patch|rope_interleave,rope_split_half"

[ -z "$(git -C "$ROOT" status --porcelain -- crates tests)" ] ||
    echo "WARN: uncommitted changes under crates/ or tests/ are NOT in the plants (built from HEAD)" >&2
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
S=$(scratch_dir launch-plants)
mkdir -p "$S/pristine" "$S/work" "$S/bin"
git -C "$ROOT" archive "$COMMIT" | tar -x -C "$S/pristine" || exit 2
git -C "$ROOT" archive "$COMMIT" | tar -x -C "$S/work" || exit 2
echo "commit $COMMIT" >"$S/manifest"
apply() { patch -p1 -F0 --no-backup-if-mismatch -s $2 -d "$S/work" <"$HERE/plants/launch/$1" >/dev/null; }

while IFS='|' read -r name patches red; do
    [ $# -eq 0 ] || [[ " $* " == *" $name "* ]] || continue
    for p in $patches; do apply "$p" "" || { echo "FAIL $name: $p does not apply to $COMMIT" >&2; exit 2; }; done
    ! diff -rq "$S/pristine" "$S/work" >/dev/null || { echo "FAIL $name: the plant changed nothing" >&2; exit 2; }
    (cd "$S/work" && CARGO_TARGET_DIR=$T cargo build -p rivoli-engine --tests --message-format=json) \
        >"$S/$name.json" 2>"$S/$name.err" || { echo "FAIL $name: build (see $S/$name.err)" >&2; exit 2; }
    bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="bit_identity" and .executable) | .executable' "$S/$name.json")
    [ -n "$bin" ] && cp "$bin" "$S/bin/bit_identity-$name" || { echo "FAIL $name: no bit_identity binary built" >&2; exit 2; }
    for p in $patches; do apply "$p" -R || { echo "FAIL $name: $p does not revert" >&2; exit 2; }; done
    diff -rq "$S/pristine" "$S/work" >/dev/null || { echo "FAIL $name: scratch tree not restored" >&2; exit 2; }
    echo "$name|$red|$(sha256sum <"$S/bin/bit_identity-$name" | cut -c1-16)" | tee -a "$S/manifest"
done <<<"$PLANTS"
echo DONE >>"$S/manifest"
echo "plants: $S/bin   manifest: $S/manifest"
