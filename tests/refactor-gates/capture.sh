#!/usr/bin/env bash
# Capture the refactor baseline: G1 decode output, G2 artifact bytes, G3 kernel ISA.
#
# A refactor is a change that moves NO number. These gates make that checkable instead of
# asserted. Run before a track starts; `check.sh` re-runs and diffs.
#
# Why hashes and not test results: the suite passing proves the assertions still hold, which
# is a weaker claim. `attn_out` is 57.9% wrong right now and the suite is green about it.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:?usage: capture.sh <manifest-dir>}"
mkdir -p "$OUT"
LOCK=/var/run/sys-gpu.lock
GLM=${GLM_ARTIFACT:-/var/db/rivoli/glm52-vq3-full}
V4=${V4_ARTIFACT:-/var/db/rivoli/v4-f4-full}
SRC=${V4_SRC:-/var/db/rivoli/deepseek-v4-flash-0731}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
# The witness is `find`, never `ls | wc -l`: on 2026-08-05 the latter returned 1 for an
# EMPTY directory and read as a phantom holder.
witness() { find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 2>/dev/null | wc -l; }

say "G3  kernel ISA — no device needed, so do it first and outside the lock"
: > "$OUT/g3-isa.txt"
find target/release/build -name '*.o' -newer Cargo.toml 2>/dev/null | sort | while read -r o; do
  # Hash the DISASSEMBLY, not the object: an object embeds paths and timestamps that move
  # for reasons a refactor is allowed to move them.
  if command -v llvm-objdump >/dev/null 2>&1; then
    printf '%s  %s\n' "$(llvm-objdump -d "$o" 2>/dev/null | grep -vE '^/|file format' | sha256sum | cut -c1-16)" "${o##*/}"
  fi
done >> "$OUT/g3-isa.txt"
wc -l < "$OUT/g3-isa.txt" | xargs echo "  objects hashed:"

say "G2  artifact bytes — converters must be byte-neutral"
if [ -d "$SRC" ] && [ -d "$V4" ]; then
  ./target/release/convert_v4 --verify --from 0 --to 3 "$SRC" "$V4" > "$OUT/g2-verify.log" 2>&1
  echo "  convert_v4 --verify rc=$? (layers 0..3)" | tee -a "$OUT/g2-verify.log"
  grep -c "0 bytes differ" "$OUT/g2-verify.log" | xargs echo "  layers verified:"
else
  echo "  SKIPPED: artifacts absent — this gate is NOT satisfied, it is unrun" | tee "$OUT/g2-verify.log"
fi

say "G1  decode output — the only gate that sees the whole pipeline"
if [ "$(witness)" -ne 0 ]; then echo "  REFUSING: GPU already held"; exit 66; fi
flock -w 3600 "$LOCK" bash -c '
  n=$(find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || { echo "  FOREIGN HOLDER inside the lock: $n"; exit 66; }
  set -x
  ./target/release/rivoli "'"$GLM"'" --mode hybrid --no-mtp -n 24 --bench > "'"$OUT"'/g1-glm.txt" 2>&1
  ./target/release/rivoli "'"$V4"'"  --no-mtp -n 16 --bench > "'"$OUT"'/g1-v4.txt"  2>&1
'
rc=$?
for f in g1-glm g1-v4; do
  [ -s "$OUT/$f.txt" ] && sha256sum "$OUT/$f.txt" | cut -c1-16 | xargs echo "  $f sha:"
done
echo "  G1 rc=$rc"

say "manifest written to $OUT"
