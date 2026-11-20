#!/usr/bin/env bash
# The bit-identity gate (docs/investigations/rust-kernels.md §3): every Rust kernel against
# its hipcc twin at tag `archive/hip-kernels`, byte-identical or RED. The harness is
# crates/engine/tests/bit_identity/; this script owns the two things a cargo test cannot:
# the REFERENCE, built from the tag rather than from the working tree, and the GPU
# discipline (flock + contention witness through tests/gpu-witness.sh's run_arm).
#
# **The reference is the tagged C, never the tree.** `git archive` extracts
# crates/backend/kernels at the tag into scratch (the tree has had no C++ since Rust-kernel
# Phase 2, 2026-11-20) and hipcc builds REF_SOURCES there with the flags
# crates/backend/build.rs used for them until then (`--offload-arch=gfx1151 -O3 -fPIC`), as a
# shared object whose `rivoli_*` launchers the harness dlopens. Comparing against anything
# rebuilt from the working tree would let a port that edited the C beside the Rust compare
# against its own edit.
#
# **Nothing builds inside the lock** (device-halves.sh's rule: no build inside the lock, none
# between arms). The test binary is built first, outside; the script itself builds only the
# hipcc reference, also outside, before `run_arm` takes the lock — and the Rust probe objects
# only when asked (`--build-probes`), never on an arm:
#
#   CARGO_TARGET_DIR=/var/cache/rivoli/target/<checkout> cargo build --workspace --tests --examples
#   CARGO_TARGET_DIR=/var/cache/rivoli/target/<checkout> tests/bit-identity.sh
#
# CARGO_TARGET_DIR must be THIS checkout's dir (`/var/cache/rivoli/target/<basename of the
# checkout>`): the binary, the device flags, the probe objects and their scratch source copies
# (`probes/<name>.src`, deleted and recreated by every build) all live under it, and the
# login default (a shared dir) would hand this checkout a sibling's (§7e-bis).
#
# usage: tests/bit-identity.sh [--expect-red=ROWS test-binary]          launcher rows
#        tests/bit-identity.sh --probes [--families=F,…]                  prelude probe rows
#        tests/bit-identity.sh --plant=FAMILY                             a probe plant's arm
#        tests/bit-identity.sh --build-probes [--plant=FAMILY]            deviceless build
#        tests/bit-identity.sh --census                                   deviceless barrier census
#        --expect-red=ROWS  inverts the verdict, for a recorded red proof: GREEN only if every
#                           row ran and examined something, EXACTLY the named rows are RED, and
#                           every other row is BIT-IDENTICAL — a plant must redden the assertion
#                           under test, not merely some assertion (gate-red-proofs.md's rule).
#        --probes           the PRELUDE probe rows (Phase 0b's families): the hipcc object is
#                           built from the in-tree probe.hip against the TAG's headers; the Rust
#                           one is the clean object `--build-probes` built.
#        --families=F,…     probe families to run (codec, reduce, fp8, i4, f4, vq).
#        --plant=FAMILY     the probe rows against that plant's object, judged by the family's
#                           row in plant_spec below: every red row matches its red RE (at least
#                           one red), and at least one row matches its must-hold RE and every
#                           such row is BIT-IDENTICAL (0b's check-arm.py rule).
#        --build-probes     build the Rust probe object (`--features probes`) from a scratch
#                           COPY of crates/kernels (rivoli-abi still read in place), with
#                           the device flags build/device.rs recorded for this checkout, into
#                           $CARGO_TARGET_DIR/probes/<FAMILY or clean>, and print its path. With
#                           --plant, tests/bit-identity/plants/FAMILY.patch is applied to the
#                           copy first: the plant is a small diff against the real prelude, so
#                           no red-proof branch lives in shipped source and nothing is copied.
#        --census           the per-kernel barrier census, deviceless (no lock): the clean
#                           probe object's static `s_barrier`/`buffer_gl0_inv` counts against
#                           hipcc's objects of the tag, each kernel equal or pinned in DIVERGE
#                           below. `--plant=barrier` is judged by this census, not by a device
#                           arm (plant_spec's CENSUS rows).
#        test-binary        a PLANTED binary, only with --expect-red; default: the one the header's
#                           build line produces, refused unless cargo reports it Fresh.
#        BITIDENT_CLASSIFY=<stdout-file>  classify an existing arm's stdout and exit, no GPU and
#                           no build — the classifier's own red proofs run this way.
# exit:  0 verdict met · 1 verdict not met · 2 setup refused · 3 arm DISCARDED (contention)

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# shellcheck source=tests/gpu-witness.sh
source "$HERE/gpu-witness.sh"

TAG=archive/hip-kernels
# The tagged files with a Rust twin — every one since the Phase 1 integration (2026-11-19).
REF_SOURCES=(activation residual attn indexer blockindex kvcompress mla linalg fwd headtail recurrent moe moe_f4)
ARCH=${RIVOLI_OFFLOAD_ARCH:-gfx1151}

# A plant's arm: the families its defect can reach (plus a clean anchor where it reaches every
# group of its own), the rows that must redden, and the rows that must hold. Derived from
# 0b's CPU defect sets and device arms (rust-kernels.md §"Phase 0b — the prelude"), respelled
# for this harness's row names (Rust float formatting: `s=NaN`). A patch plants the defect in
# the real helper, so it reaches EVERY probe calling that helper (formats: `f2e4m3_rne` also
# sits inside `ex_aq8`). `barrier` is a CENSUS row: a missing barrier is a race the bits may
# not show (0b's R-barrier arm: 0 red cells of 512), so it is judged by --census, its rows
# being kernels. It must redden exactly the kernels that call `block_sum_lds` (the planted
# helper; each call loses one of its 2 + log2(bd) barriers) and hold kernels whose barriers
# it cannot reach: the fp8 dots' combine, `mha_attend`, `index_topk`, `gather_attn_shared_kv`.
plant_spec() {
    case $1 in
        barrier) CENSUS=1
            RED_RE='^(k_block_sum2|qk_norm|index_pool_norm_f32|g(ated_delta|dn)_recurrent_f32'
            RED_RE+='|rmsnorm_(batch|centered_single|weightless_batch|gate_heads_f32)'
            RED_RE+='|kv_compress_(decode|prefill))$'
            HOLD_RE='^(k_fp8_|gemv_fp8|mha_attend$|index_topk$|gather_attn_shared_kv$)' ;;
        formats) FAMILIES=codec RED_RE='^ex_f2e4m3$|^ex_aq8 ' HOLD_RE='^ex_(aq4|rbf16|f2bf16)' ;;
        actquant) FAMILIES=codec RED_RE='^ex_aq[48] s=' HOLD_RE='^ex_aq[48] s=NaN$|^ex_f2e4m3$' ;;
        common) FAMILIES=codec RED_RE='^ex_swiglu ' HOLD_RE='^ex_situ_|^ex_moe_fixed$' ;;
        situ) FAMILIES=codec RED_RE='^ex_situ_' HOLD_RE='^ex_swiglu |^ex_moe_fixed$' ;;
        reduce) FAMILIES=reduce RED_RE='^k_wave_sum$' HOLD_RE='^k_wave_max$|^k_block_sum2' ;;
        fp8) FAMILIES=fp8 RED_RE='^k_fp8_' HOLD_RE='block=[12]$' ;;
        nib) FAMILIES=i4,f4 RED_RE='dim=(4224|1028|1000|999|130|7|1152|3712|136|2050|64)$'
            HOLD_RE='^k_(i4_wave[a-z0-9_]* dim=(6144|2048|1280|1024)|f4_wave_r[0-9] dim=(4096|2048|7168|1280))$' ;;
        # Every vq group is reached (0b: 15/15), so the anchor is a family it cannot touch.
        vq) FAMILIES=vq,reduce RED_RE='^k_vq_' HOLD_RE='^k_wave_(sum|max)$' ;;
        *) return 1 ;;
    esac
}

EXPECT_RED= RED_RE= HOLD_RE= PROBES= FAMILIES= PLANT= BUILD= CENSUS=
while [ $# -gt 0 ]; do
    case "$1" in
        --expect-red=?*) EXPECT_RED=${1#--expect-red=} ;;
        --probes) PROBES=1 ;;
        --families=?*) FAMILIES=${1#--families=} ;;
        --plant=?*) PLANT=${1#--plant=} PROBES=1 ;;
        --build-probes) BUILD=1 ;;
        --census) CENSUS=1 ;;
        --*) echo "FAIL: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
    shift
done
if [ -n "$BUILD" ]; then
    [ -z "$EXPECT_RED$FAMILIES$CENSUS$*" ] || { echo "FAIL: --build-probes takes only --plant" >&2; exit 2; }
elif [ -n "$CENSUS" ]; then
    [ -z "$EXPECT_RED$FAMILIES$PLANT$PROBES$*" ] ||
        { echo "FAIL: --census takes no other option (its plant is --plant=barrier)" >&2; exit 2; }
elif [ -n "$PLANT" ]; then
    [ -z "$EXPECT_RED$FAMILIES" ] || { echo "FAIL: --plant sets its own verdict and families" >&2; exit 2; }
    plant_spec "$PLANT" || { echo "FAIL: no plant_spec row for '$PLANT'" >&2; exit 2; }
    [ -z "$CENSUS" ] || [ -z "$*" ] || { echo "FAIL: a census plant scores no binary" >&2; exit 2; }
fi

# The verdict, from an arm's stdout ($1) and its exit code ($2). Counted from the binary's
# per-row token lines (a passing row is BIT-IDENTICAL, or SAME in the barrier census, where
# equal counts are not equal bits) and checked against the binary's own RESULT total — never from rc
# alone: a row that examined nothing prints VACUOUS and is neither verdict, and an arm that
# died part-way prints fewer rows than its RESULT line (or none) and is refused too.
classify() {
    local out=$1 rc=$2 rows total vac reds red_names
    rows=$(grep -cE '^(BIT-IDENTICAL|SAME|RED|VACUOUS) ' "$out")
    total=$(sed -nE 's/^RESULT [0-9]+ of ([0-9]+) rows.*/\1/p' "$out" | tail -1)
    vac=$(grep -cE '^VACUOUS ' "$out")
    reds=$(grep -cE '^RED ' "$out")
    red_names=$(grep -E '^RED ' "$out" | sed -E 's/^RED ([^:]+):.*/\1/' | sort | paste -sd, -)
    echo "rows=$rows of ${total:-?} red=$reds vacuous=$vac rc=$rc"
    if [ -z "$total" ] || [ "$rows" -ne "$total" ] || [ "$rows" -eq 0 ]; then
        echo "RED: the arm did not report every row (rows=$rows, RESULT total=${total:-none})" >&2
        return 1
    fi
    if [ "$vac" -ne 0 ]; then
        echo "RED: $vac row(s) VACUOUS — examined nothing, which is not a verdict" >&2
        return 1
    fi
    if [ -n "$RED_RE" ]; then
        local stray holds held_red
        stray=$(grep -E '^RED ' "$out" | sed -E 's/^RED ([^:]+):.*/\1/' | grep -vcE "$RED_RE")
        holds=$(grep -E '^(BIT-IDENTICAL|SAME|RED) ' "$out" | sed -E 's/^[A-Z-]+ ([^:]+):.*/\1/' | grep -cE "$HOLD_RE")
        held_red=$(grep -E '^RED ' "$out" | sed -E 's/^RED ([^:]+):.*/\1/' | grep -cE "$HOLD_RE")
        echo "plant $PLANT: red=$reds outside-RE=$stray must-hold rows=$holds of which red=$held_red"
        if [ "$reds" -gt 0 ] && [ "$stray" -eq 0 ] && [ "$rc" -ne 0 ] &&
            [ "$holds" -gt 0 ] && [ "$held_red" -eq 0 ]; then
            echo "GREEN (plant $PLANT): red only where the defect reaches, held where it cannot"
            return 0
        fi
        echo "RED (plant $PLANT): see the counts above (rc=$rc)" >&2
        return 1
    fi
    if [ -n "$EXPECT_RED" ]; then
        local want
        want=$(printf '%s' "$EXPECT_RED" | tr ',' '\n' | sort | paste -sd, -)
        if [ "$red_names" = "$want" ] && [ "$rc" -ne 0 ]; then
            echo "GREEN (expect-red): exactly [$want] reddened"
            return 0
        fi
        echo "RED (expect-red): wanted [$want] red, got [${red_names}] (rc=$rc)" >&2
        return 1
    fi
    if [ "$reds" -eq 0 ] && [ "$rc" -eq 0 ]; then
        echo "GREEN: every row BIT-IDENTICAL (census: SAME)"
        return 0
    fi
    echo "RED: [${red_names}] differ from the tagged hipcc twin (rc=$rc)" >&2
    return 1
}

if [ -n "${BITIDENT_CLASSIFY:-}" ]; then
    classify "$BITIDENT_CLASSIFY" "${BITIDENT_RC:-0}"
    exit $?
fi

TARGET=${CARGO_TARGET_DIR:-}
OWN=/var/cache/rivoli/target/$(basename "$ROOT")
[ "$TARGET" = "$OWN" ] || {
    echo "FAIL: CARGO_TARGET_DIR=${TARGET:-<unset>} is not this checkout's $OWN (§12)" >&2
    exit 2
}
# rustc names the code object `.elf`; `hipModuleLoad` takes it as-is.
probe_object() { echo "$TARGET/probes/$1/amdgcn-amd-amdhsa/release/rivoli_kernels.elf"; }
# What a probe object is built from: the crate sources (crates/abi included — the copy reads
# it in place) and the plant's patch. Stamped at build, re-derived by the arm, so an object
# left by an earlier build of an older tree or patch is refused rather than scored.
probe_stamp() {
    (cd "$ROOT" && find crates/kernels crates/abi -type f -print0 | sort -z | xargs -0 sha256sum |
        sha256sum) || return 1
    [ "$1" = clean ] || sha256sum <"$HERE/bit-identity/plants/$1.patch"
}
# The probe object for $1 (clean or a plant), refused when absent or stamped from other source.
current_probe() {
    local rp
    rp=$(probe_object "$1")
    [ -f "$rp" ] && [ "$(cat "$TARGET/probes/$1.stamp" 2>/dev/null)" = "$(probe_stamp "$1")" ] || {
        echo "FAIL: no current $rp — run --build-probes$([ "$1" = clean ] || echo " --plant=$1") first" >&2
        return 1
    }
    echo "$rp"
}
# The tagged C, extracted into $SCRATCH/ref: sets TAG_SHA and KDIR.
ref_tree() {
    TAG_SHA=$(git -C "$ROOT" rev-parse "$TAG^{commit}") || { echo "FAIL: tag $TAG missing" >&2; return 1; }
    mkdir -p "$SCRATCH/ref" || return 1
    git -C "$ROOT" archive "$TAG" crates/backend/kernels | tar -x -C "$SCRATCH/ref" || return 1
    KDIR=$SCRATCH/ref/crates/backend/kernels
    echo "reference: $TAG @ $TAG_SHA, $(hipcc --version 2>/dev/null | grep -m1 -i 'hip version')"
}
# hipcc for the reference, every call appending to one log.
ref_hipcc() {
    hipcc "--offload-arch=$ARCH" -O3 "$@" >>"$SCRATCH/hipcc.log" 2>&1 ||
        { echo "FAIL: hipcc (see $SCRATCH/hipcc.log)" >&2; return 1; }
}
# The prelude probes' hipcc object ($1): the in-tree probe.hip compiled against the TAG's
# headers, with 0b's flags (`--offload-device-only --no-gpu-bundle-output`, -O3) — a code
# object, not a .so.
hip_probes() {
    ref_hipcc --offload-device-only --no-gpu-bundle-output -I"$KDIR" \
        "$ROOT/crates/engine/tests/bit_identity/probes/probe.hip" -o "$1"
}

# ── Rust probe objects: deviceless, never inside the lock ────────────────────────────────
if [ -n "$BUILD" ]; then
    NAME=${PLANT:-clean}
    FLAGS=$(find "$TARGET/debug/build" -path '*/rivoli-backend-*/out/device-rustflags' \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$FLAGS" ] || { echo "FAIL: no device-rustflags under $TARGET — build the rocm arm first" >&2; exit 2; }
    PT=$TARGET/probes/$NAME
    SRC=$PT.src
    STAMP=$(probe_stamp "$NAME") || { echo "FAIL: cannot stamp '$NAME' (no such plant?)" >&2; exit 2; }
    # A failed build must not leave the previous object behind to be scored.
    rm -rf "$SRC" "$PT.stamp" "$(probe_object "$NAME")" || exit 2
    # The tree is copied whether or not a plant is applied, so a plant object and the clean one
    # differ by the patch and nothing else. `cp -r`, not `-a`: fresh mtimes, so the nested
    # cargo can never judge a copy Fresh against an object built from other source.
    mkdir -p "$SRC" && cp -r "$ROOT/crates/kernels" "$SRC/" || exit 2
    # rivoli-abi inherits its edition from the workspace root, so the copy points at it in
    # place — the one manifest line that differs, in clean and plant copies alike (a miss
    # leaves `../abi`, which does not exist beside the copy, and the build fails).
    sed -i "s|path = \"../abi\"|path = \"$ROOT/crates/abi\"|" "$SRC/kernels/Cargo.toml" || exit 2
    if [ -n "$PLANT" ]; then
        # Exact context (-F0): a patch whose anchor moved refuses instead of landing elsewhere.
        patch -p1 -F0 --no-backup-if-mismatch -d "$SRC/kernels" <"$HERE/bit-identity/plants/$NAME.patch" >&2 ||
            { echo "FAIL: the $NAME patch does not apply" >&2; exit 2; }
        diff -rq "$ROOT/crates/kernels/src" "$SRC/kernels/src" >/dev/null
        drc=$?
        [ "$drc" -eq 1 ] || { echo "FAIL: the $NAME patch changed nothing (diff rc=$drc)" >&2; exit 2; }
        [ "$(probe_stamp "$NAME")" = "$STAMP" ] ||
            { echo "FAIL: the checkout's crates/kernels changed under the patch" >&2; exit 2; }
    fi
    mkdir -p "$PT" || exit 2
    # A clean environment rather than a list of variables to strip (the outer shell's
    # CARGO_TARGET_DIR, RUSTFLAGS, RUSTUP_TOOLCHAIN, wrappers…): nothing reaches the nested
    # build but the flags read above, and rustup resolves the pin from the crate's own
    # rust-toolchain.toml because the build runs in its directory.
    (cd "$SRC/kernels" &&
        env -i PATH="$PATH" HOME="$HOME" ${RUSTUP_HOME:+RUSTUP_HOME="$RUSTUP_HOME"} \
            ${CARGO_HOME:+CARGO_HOME="$CARGO_HOME"} CARGO_ENCODED_RUSTFLAGS="$(cat "$FLAGS")" \
            cargo build --release --locked --target amdgcn-amd-amdhsa -Zbuild-std=core \
            --target-dir "$PT" --features probes) >"$PT.log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || { echo "FAIL: probe build rc=$rc (see $PT.log)" >&2; exit 2; }
    printf '%s\n' "$STAMP" >"$PT.stamp" || exit 2
    probe_object "$NAME"
    exit 0
fi

# ── the barrier census: deviceless, never inside the lock ────────────────────────────────
# A missing barrier is a race the bits may not show, so the barrier plant is judged here, from
# the ISA: per kernel, the static `s_barrier` and `buffer_gl0_inv` counts (`__syncthreads`
# on gfx11 is `s_barrier; buffer_gl0_inv`, `prelude::reduce::sync_threads` the same) in the
# probe object — every shipped kernel plus the prelude probes — against hipcc's objects of the
# tag, one per REF_SOURCES file with build.rs's flags (`-fgpu-rdc`, one object for all, would
# change the codegen) plus probe.hip. A row is a paired kernel with a barrier on either side;
# SAME is equal counts, or the exact counts DIVERGE pins. The clean census also holds the
# EMBEDDED object (the one `src/module.rs` ships) to the probe object, kernel by kernel, so
# scoring the probe build is not scoring a different binary.
#
# DIVERGE: `kernel hipcc_barrier hipcc_inv rust_barrier rust_inv`, one row per line, where the
# two differ by construction — hipcc loops an LDS tree on blockDim (one barrier in the loop
# body), Rust unrolls the block-size constant's tree (one per level): L1's `s_barrier 2 → 9`,
# L4's `20 against 6` (rust-kernels.md §5). Pinned exactly, so a plant that removes one of the
# unrolled barriers still reddens the row. `hash_rows`' hipcc elides one invalidate after a
# barrier; `sync_threads` always emits it.
DIVERGE='argmax_reduce 2 2 9 9
attn_res 3 3 10 10
hash_rows 2 1 9 9
kv_compress_decode 9 9 16 16
kv_compress_prefill 8 8 15 15
layernorm 2 2 9 9
mha_attend 6 6 20 20
qk_norm 3 3 10 10
rmsnorm_batch 3 3 10 10
rmsnorm_centered_single 3 3 10 10
rmsnorm_single 2 2 9 9
rmsnorm_weightless_batch 3 3 10 10'
# Kernels with no twin, each argued: the gate's device compares (hipcc-only by design, probes
# census), `k_vq_wave_r1` (LTO merges it into the identical `k_vq_wave` on the Rust side),
# `touch_region` (async.hip is not a REF_SOURCE: no output to compare, kernel_coverage's
# UNROWED). Each must be one-sided, both ends. Rust's `argmax` is hipcc's `argmax_reduce`.
ONE_SIDED='cmp_bytes cmp_nan_words k_vq_wave_r1 touch_region'
if [ -n "$CENSUS" ]; then
    # `join` pairs on field 1 while `sort` orders whole lines; under a collating locale `_` and
    # spaces are ignored and the two orders can disagree (review 2026-11-20, reproduced).
    export LC_ALL=C
    RP=$(current_probe "${PLANT:-clean}") || exit 2
    OBJDUMP=$(hipconfig --hipclangpath 2>/dev/null)/llvm-objdump
    [ -x "$OBJDUMP" ] || { echo "FAIL: no llvm-objdump beside hipcc's clang ($OBJDUMP)" >&2; exit 2; }
    SCRATCH=/var/cache/rivoli/scratch/bit-identity/census-$(date +%Y%m%dT%H%M%S)-$$
    mkdir -p "$SCRATCH/obj" || exit 2
    ref_tree || exit 2
    for s in "${REF_SOURCES[@]}"; do
        ref_hipcc -fPIC --offload-device-only --no-gpu-bundle-output "$KDIR/$s.hip" \
            -o "$SCRATCH/obj/$s.hsaco" || exit 2
    done
    hip_probes "$SCRATCH/obj/probes.hsaco" || exit 2
    sha256sum "$RP" | sed 's/^/rust:      /'
    counts() { # objects… → sorted `kernel barriers invalidates`, one line per kernel
        "$OBJDUMP" -d "--mcpu=$ARCH" "$@" | awk '
            /^[0-9a-f]+ <[^>]+>:$/ { k = substr($2, 2, length($2) - 3); b[k] += 0; g[k] += 0; next }
            k && /^[ \t]+s_barrier/ { b[k]++ }
            k && /^[ \t]+buffer_gl0_inv/ { g[k]++ }
            END { for (k in b) print k, b[k], g[k] }' | sort
    }
    counts "$SCRATCH"/obj/*.hsaco >"$SCRATCH/hipcc.txt" || exit 2
    counts "$RP" >"$SCRATCH/probe.txt" || exit 2
    if [ -z "$PLANT" ]; then
        EMB=$(find "$TARGET/debug/build" -path \
            '*/rivoli-backend-*/out/kernels-target/amdgcn-amd-amdhsa/release/rivoli_kernels.elf' \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        [ -n "$EMB" ] || { echo "FAIL: no embedded code object under $TARGET — build the rocm arm first" >&2; exit 2; }
        counts "$EMB" >"$SCRATCH/embedded.txt" || exit 2
        # Every embedded kernel, present in the probe object with the same counts.
        drift=$(join -a1 -e - -o 0,1.2,1.3,2.2,2.3 "$SCRATCH/embedded.txt" "$SCRATCH/probe.txt" |
            awk '$2 != $4 || $3 != $5') || exit 2
        [ -z "$drift" ] || {
            echo "FAIL: the embedded object ($EMB) differs from the probe object:" >&2
            echo "$drift" >&2
            exit 2
        }
        echo "embedded:  $(wc -l <"$SCRATCH/embedded.txt") kernels, counts equal to the probe object's ($EMB)"
    fi
    sed 's/^argmax /argmax_reduce /' "$SCRATCH/probe.txt" | sort >"$SCRATCH/rust.txt" || exit 2
    join --check-order -a1 -a2 -e - -o 0,1.2,1.3,2.2,2.3 "$SCRATCH/hipcc.txt" "$SCRATCH/rust.txt" \
        >"$SCRATCH/joined.txt" || { echo "FAIL: join (unsorted input?)" >&2; exit 2; }
    awk -v one="$ONE_SIDED" -v table="$DIVERGE" '
        BEGIN {
            n = split(one, o, " "); for (i = 1; i <= n; i++) side[o[i]] = 0
            n = split(table, t, "\n")
            for (i = 1; i <= n; i++) { split(t[i], f, " "); pin[f[1]] = f[2] " " f[3] " " f[4] " " f[5] }
        }
        {
            k = $1; got = $2 " " $3 " " $4 " " $5
            if ($2 == "-" || $4 == "-") {
                if (k in side) { side[k] = 1; next }
                print "RED " k ": no twin (hipcc " $2 "/" $3 ", rust " $4 "/" $5 ")"; rows++; reds++; next
            }
            if (k in side) { print "SETUP: " k " is listed one-sided but has both twins"; bad = 1; next }
            if (k in pin) { want = pin[k]; used[k] = 1 }
            else if ($2 + $3 + $4 + $5 == 0) next
            else want = $2 " " $3 " " $2 " " $3
            rows++
            split(want, w, " ")
            line = k ": s_barrier " $2 "/" $4 ", buffer_gl0_inv " $3 "/" $5 " (hipcc/rust)"
            if (got == want) print "SAME " line
            else { reds++; print "RED " line "; want " w[1] "/" w[3] ", " w[2] "/" w[4] }
        }
        END {
            for (k in side) if (!side[k]) { print "SETUP: one-sided " k " not found on exactly one side"; bad = 1 }
            for (k in pin) if (!(k in used)) { print "SETUP: DIVERGE row " k " names no paired kernel"; bad = 1 }
            print "RESULT " rows - reds " of " rows " rows"
            exit bad ? 2 : (reds ? 1 : 0)
        }' "$SCRATCH/joined.txt" >"$SCRATCH/census.stdout"
    rc=$?
    cat "$SCRATCH/census.stdout"
    echo "logs: $SCRATCH"
    [ "$rc" -eq 2 ] && { echo "FAIL: the census tables disagree with the objects (SETUP lines)" >&2; exit 2; }
    classify "$SCRATCH/census.stdout" "$rc"
    exit $?
fi

LOCK=${GPU_LOCK:-/var/run/sys-gpu.lock}
[ -e "$LOCK" ] || {
    echo "FAIL: $LOCK does not resolve (gate-red-proofs.md §16, the /var/run bind defect)" >&2
    exit 2
}

SCRATCH=/var/cache/rivoli/scratch/bit-identity/$(date +%Y%m%dT%H%M%S)-$$
mkdir -p "$SCRATCH" || exit 2

# ── the binary: built from THIS tree, by cargo's own fingerprint ────────────────────────
# Not "the newest by mtime", which is how a binary built before the last edit got scored:
# cargo is asked, with the header's exact build line, which bit_identity executable that
# line produces and whether it was already Fresh. Fresh means nothing builds here; a stale
# binary is rebuilt by that call (outside the lock) and the arm is REFUSED, so a run never
# scores code the tree no longer holds. A positional binary is a PLANTED copy, built from a
# planted tree on purpose, so it is accepted only with --expect-red and is named as such.
if [ -n "${1:-}" ]; then
    [ -n "$EXPECT_RED" ] || {
        echo "FAIL: a positional binary is a planted copy; it is accepted only with --expect-red" >&2
        exit 2
    }
    BIN=$1
    echo "binary is PLANTED (positional, --expect-red): its freshness is not checked"
else
    # In $ROOT: cargo resolves the workspace (and rustup the toolchain) from the directory it
    # runs in, so from anywhere else this would build ANOTHER checkout into this target dir.
    (cd "$ROOT" && cargo build --workspace --tests --examples --message-format=json) \
        >"$SCRATCH/fresh.json" 2>"$SCRATCH/fresh.err"
    crc=$?
    [ "$crc" -eq 0 ] || { echo "FAIL: cargo build rc=$crc (see $SCRATCH/fresh.err)" >&2; exit 2; }
    read -r BIN FRESH < <(jq -r 'select(.reason == "compiler-artifact" and
        .target.name == "bit_identity" and .executable) | "\(.executable) \(.fresh)"' \
        "$SCRATCH/fresh.json")
    [ -n "${BIN:-}" ] || { echo "FAIL: cargo named no bit_identity executable" >&2; exit 2; }
    [ "$FRESH" = true ] || {
        echo "FAIL: the bit_identity binary was STALE against the tree — cargo has just rebuilt" >&2
        echo "      it (outside the lock); rerun to score the binary the tree now builds" >&2
        exit 2
    }
fi
[ -x "$BIN" ] || { echo "FAIL: $BIN is not an executable" >&2; exit 2; }

# ── the reference, outside the lock ─────────────────────────────────────────────────────
ref_tree || exit 2
echo "binary:    $BIN"
ENVS=()
if [ -n "$PROBES" ]; then
    HP=$SCRATCH/ref/probes.hsaco
    hip_probes "$HP" || exit 2
    RP=$(current_probe "${PLANT:-clean}") || exit 2
    # The reference source is the working tree's probe.hip (its bodies are thin calls into the
    # tag's headers); its hash and both objects' are printed so an arm names what it compared.
    sha256sum "$ROOT/crates/engine/tests/bit_identity/probes/probe.hip" "$HP" "$RP" |
        sed 's/^/sha256:    /'
    TEST=every_prelude_probe_is_identical_to_its_hipcc_twin
    ENVS+=(RIVOLI_BITIDENT_PROBES_HIP="$HP" RIVOLI_BITIDENT_PROBES_RUST="$RP")
    [ -n "$FAMILIES" ] && ENVS+=(RIVOLI_BITIDENT_FAMILIES="$FAMILIES")
else
    SRCS=()
    for s in "${REF_SOURCES[@]}"; do SRCS+=("$KDIR/$s.hip"); done
    SO=$SCRATCH/ref/rivoli_ref.so
    ref_hipcc -fPIC -shared -Wl,-Bsymbolic "${SRCS[@]}" -o "$SO" || exit 2
    TEST=every_rust_kernel_is_byte_identical_to_its_tagged_hipcc_twin
    ENVS+=(RIVOLI_BITIDENT_REF="$SO")
fi

# ── the arm ─────────────────────────────────────────────────────────────────────────────
rc=0
run_arm bit-identity "$SCRATCH/arm.stdout" "$SCRATCH/arm.log" \
    env "${ENVS[@]}" "$BIN" --ignored --exact "$TEST" --test-threads=1 --nocapture || rc=$?
[ "$rc" -eq 3 ] && exit 3

grep -hE '^(BIT-IDENTICAL|RED|VACUOUS) |^    |^RESULT |^probes: ' "$SCRATCH/arm.stdout" || true
echo "logs: $SCRATCH"
classify "$SCRATCH/arm.stdout" "$rc"
