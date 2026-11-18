#!/usr/bin/env bash
# The bit-identity gate (docs/investigations/rust-kernels.md §3): every Rust kernel against
# its hipcc twin at tag `archive/hip-kernels`, byte-identical or RED. The harness is
# crates/engine/tests/bit_identity/; this script owns the two things a cargo test cannot:
# the REFERENCE, built from the tag rather than from the working tree, and the GPU
# discipline (flock + contention witness through tests/gpu-witness.sh's run_arm).
#
# **The reference is the tagged C, never the tree.** `git archive` extracts
# crates/backend/kernels at the tag into scratch and hipcc builds REF_SOURCES there with the
# flags crates/backend/build.rs used for them (`--offload-arch=gfx1151 -O3 -fPIC`), as a
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
# The tagged files with a Rust twin. Phase 1 lanes append theirs.
REF_SOURCES=(activation residual)
ARCH=${RIVOLI_OFFLOAD_ARCH:-gfx1151}

# A plant's arm: the families its defect can reach (plus a clean anchor where it reaches every
# group of its own), the rows that must redden, and the rows that must hold. Derived from
# 0b's CPU defect sets and device arms (rust-kernels.md §"Phase 0b — the prelude"), respelled
# for this harness's row names (Rust float formatting: `s=NaN`). A patch plants the defect in
# the real helper, so it reaches EVERY probe calling that helper (formats: `f2e4m3_rne` also
# sits inside `ex_aq8`). `barrier` has a patch but no row: a missing barrier is a race the
# bits may not show, and its judge (0b's scratch ISA census, `s_barrier` per kernel) is not
# in the tree yet — owed.
plant_spec() {
    case $1 in
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

EXPECT_RED= RED_RE= HOLD_RE= PROBES= FAMILIES= PLANT= BUILD=
while [ $# -gt 0 ]; do
    case "$1" in
        --expect-red=?*) EXPECT_RED=${1#--expect-red=} ;;
        --probes) PROBES=1 ;;
        --families=?*) FAMILIES=${1#--families=} ;;
        --plant=?*) PLANT=${1#--plant=} PROBES=1 ;;
        --build-probes) BUILD=1 ;;
        --*) echo "FAIL: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
    shift
done
if [ -n "$BUILD" ]; then
    [ -z "$EXPECT_RED$FAMILIES$*" ] || { echo "FAIL: --build-probes takes only --plant" >&2; exit 2; }
elif [ -n "$PLANT" ]; then
    [ -z "$EXPECT_RED$FAMILIES" ] || { echo "FAIL: --plant sets its own verdict and families" >&2; exit 2; }
    plant_spec "$PLANT" || { echo "FAIL: no plant_spec row for '$PLANT'" >&2; exit 2; }
fi

# The verdict, from an arm's stdout ($1) and its exit code ($2). Counted from the binary's
# per-row token lines and checked against the binary's own RESULT total — never from rc
# alone: a row that examined nothing prints VACUOUS and is neither verdict, and an arm that
# died part-way prints fewer rows than its RESULT line (or none) and is refused too.
classify() {
    local out=$1 rc=$2 rows total vac reds red_names
    rows=$(grep -cE '^(BIT-IDENTICAL|RED|VACUOUS) ' "$out")
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
        holds=$(grep -E '^(BIT-IDENTICAL|RED) ' "$out" | sed -E 's/^[A-Z-]+ ([^:]+):.*/\1/' | grep -cE "$HOLD_RE")
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
        echo "GREEN: every row BIT-IDENTICAL"
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

LOCK=${GPU_LOCK:-/var/run/sys-gpu.lock}
[ -e "$LOCK" ] || {
    echo "FAIL: $LOCK does not resolve (gate-red-proofs.md §16, the /var/run bind defect)" >&2
    exit 2
}

SCRATCH=/var/cache/rivoli/scratch/bit-identity/$(date +%Y%m%dT%H%M%S)-$$
mkdir -p "$SCRATCH/ref" || exit 2

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
TAG_SHA=$(git -C "$ROOT" rev-parse "$TAG^{commit}") || { echo "FAIL: tag $TAG missing" >&2; exit 2; }
git -C "$ROOT" archive "$TAG" crates/backend/kernels | tar -x -C "$SCRATCH/ref" || exit 2
KDIR=$SCRATCH/ref/crates/backend/kernels
echo "reference: $TAG @ $TAG_SHA, $(hipcc --version 2>/dev/null | grep -m1 -i 'hip version')"
echo "binary:    $BIN"
ENVS=()
if [ -n "$PROBES" ]; then
    # The prelude probes: the in-tree probe.hip compiled against the TAG's headers, with 0b's
    # flags (`--offload-device-only --no-gpu-bundle-output`, -O3) — a code object, not a .so.
    HP=$SCRATCH/ref/probes.hsaco
    hipcc "--offload-arch=$ARCH" -O3 --offload-device-only --no-gpu-bundle-output -I"$KDIR" \
        "$ROOT/crates/engine/tests/bit_identity/probes/probe.hip" -o "$HP" \
        >"$SCRATCH/hipcc.log" 2>&1 || { echo "FAIL: hipcc (see $SCRATCH/hipcc.log)" >&2; exit 2; }
    RP=$(probe_object "${PLANT:-clean}")
    [ -f "$RP" ] && [ "$(cat "$TARGET/probes/${PLANT:-clean}.stamp" 2>/dev/null)" = "$(probe_stamp "${PLANT:-clean}")" ] ||
        { echo "FAIL: no current $RP — run --build-probes${PLANT:+ --plant=$PLANT} first" >&2; exit 2; }
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
    hipcc "--offload-arch=$ARCH" -O3 -fPIC -shared -Wl,-Bsymbolic "${SRCS[@]}" -o "$SO" \
        >"$SCRATCH/hipcc.log" 2>&1 || { echo "FAIL: hipcc (see $SCRATCH/hipcc.log)" >&2; exit 2; }
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
