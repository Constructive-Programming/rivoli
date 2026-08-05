#!/usr/bin/env bash
# Capture the refactor baseline: G1 decode output, G2 artifact bytes, G3 kernel ISA.
#
# A refactor is a change that moves NO number. These gates make that checkable instead of
# asserted. `check.sh` re-runs and diffs.
#
# Hashes rather than test results on purpose: a green suite proves the assertions still hold,
# which is weaker. `attn_out` is 57.9% wrong today and the suite is green about it.
#
# EVERY gate writes a `<name>.status` file reading `RAN` or `UNRUN <reason>`. check.sh fails
# on any UNRUN *before* comparing anything — because both of this file's previous vacuity
# bugs produced non-empty outputs that compared equal to themselves.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:?usage: capture.sh <manifest-dir>}"
mkdir -p "$OUT"
LOCK=/var/run/sys-gpu.lock
GLM=${GLM_ARTIFACT:-/var/db/rivoli/glm52-vq3-full}
V4=${V4_ARTIFACT:-/var/db/rivoli/v4-f4-full}
SRC=${V4_SRC:-/var/db/rivoli/deepseek-v4-flash-0731}

say()   { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ran()   { echo "RAN"            > "$OUT/$1.status"; }
unrun() { echo "UNRUN $2"       > "$OUT/$1.status"; printf '  UNRUN: %s\n' "$2"; }
# `find`, never `ls | wc -l`: the latter returned 1 for an EMPTY directory on 2026-08-05.
witness() { find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 2>/dev/null | wc -l; }

say "provenance — a gate that measures target/ says nothing about the source"
{ echo "head $(git rev-parse HEAD)"
  echo "dirty $(git status --porcelain | wc -l)"
  for b in rivoli convert_v4; do
    [ -x "target/release/$b" ] && echo "$b mtime $(stat -c %Y "target/release/$b")"
  done
  # Newest source file. check.sh refuses if a binary predates it: otherwise "ALL GATES HELD"
  # can describe a binary built before the refactor it is certifying.
  echo "newest_src $(find src kernels -type f \( -name '*.rs' -o -name '*.hip' -o -name '*.hpp' \) -printf '%T@\n' | sort -rn | head -1 | cut -d. -f1)"
} > "$OUT/provenance.txt"
cat "$OUT/provenance.txt" | sed 's/^/  /'

say "G3  kernel ISA — AMDGCN, not the x86 launch stubs"
: > "$OUT/g3-isa.txt"; n_obj=0; n_gcn=0
# The device image lives in `.hip_fatbin`, a DATA section. `llvm-objdump -d` reads TEXT only,
# so disassembling the object yields the HOST launch stub — a function of the kernel
# SIGNATURE, not its body. Measured 2026-08-05: 0 AMDGCN instructions. A kernel-body rewrite
# left it byte-identical, which is precisely what this gate exists to catch.
outdir=$(find target/release/build -maxdepth 2 -name out -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$outdir" ]; then
  for o in "$outdir"/*.o; do
    [ -f "$o" ] || continue
    n_obj=$((n_obj+1))
    fat=$(mktemp); llvm-objcopy --dump-section=.hip_fatbin="$fat" "$o" /dev/null 2>/dev/null
    if [ -s "$fat" ]; then
      d=$(llvm-objdump -d --triple=amdgcn-amd-amdhsa "$fat" 2>/dev/null | grep -vE '^/|file format')
      c=$(grep -cE 's_endpgm|v_mov_b32|v_fmac_f32|v_add_f32' <<<"$d")
      n_gcn=$((n_gcn+c))
      printf '%s  %s\n' "$(sha256sum <<<"$d" | cut -c1-16)" "$o"
    fi
    rm -f "$fat"
  done >> "$OUT/g3-isa.txt"
fi
echo "  objects=$n_obj  amdgcn-instructions=$n_gcn  outdir=${outdir:-none}"
if [ "$n_obj" -eq 0 ] || [ "$n_gcn" -eq 0 ]; then
  unrun g3 "no device code disassembled (objects=$n_obj gcn=$n_gcn) — hashing nothing"
else ran g3; fi

say "G2  artifact bytes — READ-ONLY, all layers"
# --verify-only, NOT --verify. `--verify` is a flag on a CONVERTER: it still rewrites
# resident.safetensors and manifest.json for --from/--to, so a narrow verify against a whole
# artifact truncates it. This script was about to do that to the 146 GB v4-f4-full.
if [ -d "$SRC" ] && [ -d "$V4" ]; then
  ./target/release/convert_v4 --verify-only "$SRC" "$V4" > "$OUT/g2-verify.log" 2>&1
  rc=$?; echo "convert_v4 rc=$rc" >> "$OUT/g2-verify.log"
  v=$(grep -cE '^convert_v4: verified L[0-9]+\.f4 .* 0 bytes differ$' "$OUT/g2-verify.log")
  echo "  layers verified: $v  rc=$rc"
  # Anchored, and an explicit floor: `grep -c "0 bytes differ"` also matches "40 bytes
  # differ", so a corruption whose byte count ends in 0 kept the count unchanged.
  if [ "$rc" -eq 0 ] && [ "$v" -ge 43 ]; then ran g2; else unrun g2 "rc=$rc verified=$v (want rc=0, >=43)"; fi
else
  unrun g2 "artifacts absent: $SRC / $V4"
  : > "$OUT/g2-verify.log"
fi

say "G1  decode output — the only gate that sees the whole pipeline"
# `--mode int4`, not hybrid: hybrid's cache picks each expert's FORMAT, so residency selects
# the arithmetic and the text legitimately moves with --max-mem (architecture.md §8b, INV-1).
# `-bench N` is the recorded spelling; `-n N` is not a flag this binary has, and the clap
# error it produced was byte-STABLE, so G1 reported OK without ever loading a model.
if [ ! -d "$GLM" ] && [ ! -d "$V4" ]; then
  unrun g1 "no artifact present"
elif [ "$(witness)" -ne 0 ]; then
  unrun g1 "GPU already held before the lock"
else
  flock -w 3600 "$LOCK" bash -c '
    n=$(find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] || { echo "FOREIGN HOLDER inside the lock: $n"; exit 66; }
    r=0
    [ -d "'"$GLM"'" ] && { ./target/release/rivoli "'"$GLM"'" --mode int4 --no-mtp -bench 24 > "'"$OUT"'/g1-glm.raw" 2>&1 || r=1; }
    [ -d "'"$V4"'"  ] && { ./target/release/rivoli "'"$V4"'"  --no-mtp -bench 16 > "'"$OUT"'/g1-v4.raw"  2>&1 || r=1; }
    exit $r'
  rc=$?
  # Strip what legitimately moves: tracing's microsecond timestamps (main.rs:673) and every
  # "N.Ns" / "N.N tok/s" (main.rs:574,827,928 + telemetry.rs:607's always-on PROFILE line).
  # Hashing the raw log makes G1 red on EVERY run, and a gate that always fires gets deleted.
  for f in g1-glm g1-v4; do
    [ -s "$OUT/$f.raw" ] || continue
    sed -E 's/^[0-9]{4}-[0-9-]+T[0-9:.]+Z? +//; s/[0-9]+\.[0-9]+ ?(s|ms|tok\/s|%|GB)/N\1/g' \
      "$OUT/$f.raw" > "$OUT/$f.txt"
    sha256sum "$OUT/$f.txt" | cut -c1-16 | xargs echo "  $f sha:"
  done
  # A clap/usage error is non-empty and stable; require evidence a decode actually happened.
  if [ "$rc" -eq 0 ] && grep -qiE 'tok/s|tokens in' "$OUT"/g1-*.txt 2>/dev/null; then ran g1
  else unrun g1 "rc=$rc and no tok/s line — the binary did not decode"; fi
fi

say "manifest written to $OUT"
grep -H . "$OUT"/*.status | sed 's/^/  /'
