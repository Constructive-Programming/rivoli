#!/usr/bin/env bash
# M11 gate (a): the ported `convert_glimmer --fp8` reproduces, byte for byte, the artifact
# the OLD tree's converter wrote (or every diff is argued in writing in the M11 record).
#
#   tests/convert-parity-glimmer-fp8.sh <reference-artifact-dir> <candidate-artifact-dir>
#
# WHY A SCRIPT and not two ad-hoc sha256 calls: the comparison must be UNABLE to compare a
# file against itself and must print both paths, sizes and hashes — a pasted hash pair with
# no provenance is exactly the false-green this repo's verification notes warn about. Both
# refusals below exit 66 (the loud-failure convention the GPU flock also uses).
#
# Red-proof (P7): run it on a scratch copy of any small artifact against itself-with-one-
# byte-flipped and watch DIFFER; run it with the same directory twice and watch the refusal.
# The M11 record logs both runs.
set -euo pipefail

[ $# -eq 2 ] || {
    echo "usage: $0 <reference-artifact-dir> <candidate-artifact-dir>" >&2
    exit 2
}
ref=$(realpath "$1")
cand=$(realpath "$2")
# The self-comparison refusal: identical RESOLVED paths (so `dir` vs `dir/.` and symlinked
# spellings of one directory are caught, not just equal strings).
if [ "$ref" = "$cand" ]; then
    echo "REFUSED: both arguments resolve to $ref — a parity gate that can compare a file" >&2
    echo "against itself can never go red" >&2
    exit 66
fi

# Every file the artifact carries. resident.safetensors is the 28.5 GiB one that proves the
# quantizer; manifest.json proves the format stamp; the aux four prove the copies.
files="resident.safetensors manifest.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja"

fail=0
for f in $files; do
    a="$ref/$f"
    b="$cand/$f"
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then
        echo "$f: MISSING ($([ -f "$a" ] || echo "ref ")$([ -f "$b" ] || echo cand))"
        fail=1
        continue
    fi
    # Same resolved path for one member = self-comparison through a symlinked file.
    if [ "$(realpath "$a")" = "$(realpath "$b")" ]; then
        echo "REFUSED: $f resolves to the same file in both directories" >&2
        exit 66
    fi
    sa=$(stat -c%s "$a")
    sb=$(stat -c%s "$b")
    ha=$(sha256sum "$a" | cut -d' ' -f1)
    hb=$(sha256sum "$b" | cut -d' ' -f1)
    printf '%s\n  ref  %14d  %s  %s\n  cand %14d  %s  %s\n' \
        "$f" "$sa" "$ha" "$a" "$sb" "$hb" "$b"
    if [ "$sa" = "$sb" ] && [ "$ha" = "$hb" ]; then
        echo "  MATCH"
    else
        echo "  DIFFER"
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "PARITY: every file byte-identical"
else
    echo "PARITY FAILED: at least one file differs or is missing (argue each diff in writing, or fix the port)"
fi
exit $fail
