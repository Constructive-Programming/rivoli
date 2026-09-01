#!/usr/bin/env bash
# **Regenerate the S2 Qwen anchor goldens and the defect matrix.**
# `docs/measurement/qwen-reference/anchor.md` is the record this writes;
# `crates/oracles/tests/qwen_anchor_driver.py` is what it runs.
#
# NOT a `cargo test`. It needs a pinned python environment (transformers **v5.16.1** exactly --
# the model dir does not exist at v5.15) and it takes minutes, so it stays out of the deviceless
# battery; `crates/oracles/tests/qwen_anchor.rs` is the gate that reads what this vendors.
#
#     QWEN_ANCHOR_VENV=/var/cache/rivoli/scratch/qwen-anchor/venv crates/oracles/tests/qwen-anchor.sh
#
# **No GPU, no flock, no coordinator lease -- and that is a measured finding, not an omission.**
# K3's anchor takes the device because fla's KDA ops are triton kernels with no CPU path. Qwen's
# GatedDeltaNet ships pure-torch bodies inside transformers (MOD:266-398) that the
# kernel-from-hub decorator selects whenever `fla` is absent, and the driver ASSERTS `fla`,
# `kernels` and `causal_conv1d` are all absent rather than trusting the venv. If a future run
# needs the device, it needs a GO first -- nothing here may take the lock on its own.
#
# **Wall clock: 252 s on `nproc` 32, measured 2026-08-31** as the span from the first written
# golden to the last on a full run, so it is a LOWER BOUND -- it excludes this script's own
# startup and the class census before the first generation. This line said "~9 min ... ~2 GB peak
# RSS" until it was checked: the time was 2.1x the measurement and the RSS figure had never been
# measured at all, so it is deleted rather than corrected. A number nothing re-derives drifts.
#
# **Configured by env var, and that is not the thing CLAUDE.md forbids.** That rule ("instruments
# go behind a feature AND a flag, never an env var") gives three reasons, all about the engine
# binary: an env var is invisible to `--help`, absent from the recorded command line in
# `benchmarks.md`, and silently active in a build that looks stock. None applies to a script that
# is not a cargo run, has no `--help`, and whose invocation is recorded on the line above and in
# `anchor.md`. `${VAR:?}` below makes a missing one fail loudly.
#
# Writes `$OUT/gold-<mode>-<salt>-<defect>.bin` plus the reddening matrix and the class census to
# stdout, and `cmp`s each fresh `None` decode golden against its vendored twin.
set -euo pipefail

VENV=${QWEN_ANCHOR_VENV:?set QWEN_ANCHOR_VENV to the venv holding torch + transformers 5.16.1}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)   # workspace root; this script lives in crates/oracles/tests/
TESTS=$ROOT/crates/oracles/tests
OUT=${QWEN_ANCHOR_OUT:-/var/cache/rivoli/scratch/qwen-anchor/out}
CONFIG=$ROOT/docs/measurement/qwen-reference/config.json
PY=$VENV/bin/python
DRIVER=$TESTS/qwen_anchor_driver.py
# `/home` is NFS on this box and its attribute cache serves stale mtimes, so a `.pyc` written by
# an earlier run can shadow an edited source -- observed while authoring this matrix, as a
# traceback whose quoted line came from a different revision of the file than the code that ran.
# The same scar the port doc records for stale cargo binaries, one language over.
export PYTHONDONTWRITEBYTECODE=1

# Every defect row, read out of the driver's own table rather than restated here: a list in two
# places is a list that drifts, and `assert_class_coverage` already refuses a row that belongs to
# no class.
mapfile -t DEFECTS < <(cd "$TESTS" && "$PY" -c 'from qwen_anchor_defects import DEFECTS; print("\n".join(DEFECTS))')
MODES=(${QWEN_ANCHOR_MODES:-decode})
# TWO independent weight draws, and both are vendored. One draw cannot show that a defect's
# localisation is a property of the arithmetic rather than of the numbers it landed on, and a
# defect degenerate at one draw's values -- a router weight that happens to be peaked, a
# palindromic conv kernel -- hides completely. Every row below is scored against both.
SALTS=(${QWEN_ANCHOR_SALTS:-qwen-anchor-1 qwen-anchor-2})

mkdir -p "$OUT"
cd "$TESTS"

# The class census FIRST, because a matrix that runs for nine minutes and then turns out to cover
# seven classes has wasted the nine minutes.
"$PY" -c 'from qwen_anchor_defects import CLASSES, DEFECTS, assert_class_coverage
assert_class_coverage()
print(f"defect rows: {len(DEFECTS) - 1} across {len(CLASSES)} classes")
for name, rows in CLASSES.items():
    print(f"  {len(rows):2d}  {name}")'

for salt in "${SALTS[@]}"; do
for mode in "${MODES[@]}"; do
    for d in "${DEFECTS[@]}"; do
        "$PY" "$DRIVER" --config "$CONFIG" --mode "$mode" --defect "$d" --salt "$salt" \
            --out "$OUT/gold-$mode-$salt-$d.bin" >/dev/null
    done
    echo "=== $mode $salt ==="
    # ONE `--compare` per defect, printing the matrix that was gated. It exits non-zero if the
    # defect changed nothing, if its own declared first-touched bucket stayed identical, if a
    # bucket UPSTREAM of that one moved, or if the two runs captured different tensors -- see
    # `qwen_anchor_compare._gate_first_touch`. Two invocations (one to print, one to gate) would be
    # two independent scorings whose agreement is assumed.
    for d in "${DEFECTS[@]:1}"; do
        printf '%-34s ' "$d"
        "$PY" "$DRIVER" --compare "$OUT/gold-$mode-$salt-None.bin" "$OUT/gold-$mode-$salt-$d.bin" \
            | awk -F'\t' '$2 ~ /^[0-9]+$/ {n+=$2; d+=$3; if ($4+0 > m) m=$4+0}
                          END {printf "buckets-differing %4d/%-4d worst_rel %.3e\n", d, n, m}'
    done
done
done

# **One PREFILL golden, vendored beside the two decode ones, and it earns its bytes.** The
# dense-versus-selective boundary is the whole of trap T19 and half of T8, and it is only visible
# across QUERIES: in `decode` there is one query at 17 visible tokens, all of it selective, while
# a 16-position prefill holds 11 queries in the provably-dense regime and 5 above it. So the
# prefill golden is what lets `qwen_anchor.rs` assert BOTH regimes from the bytes, deviceless.
# Salt 1 only: the claim is about the reference's construction, not about a weight draw.
echo "=== prefill window (the dense/selective boundary) ==="
# `--capture-layers 3` -- the FIRST QSA layer alone. All seven layers at 16 positions is 2.7 MB
# of vendored bytes for a claim that lives in one layer's indexer, and the full-depth structure
# is already pinned by the two decode goldens.
"$PY" "$DRIVER" --config "$CONFIG" --mode prefill --defect None --salt qwen-anchor-1 \
    --capture-layers 3 --out "$OUT/gold-prefill-qwen-anchor-1-None.bin"

# The transcription gate. `qwen_anchor_taps.indexer_variant_forward` is the only copy of
# reference arithmetic in this anchor, and three defect rows substitute one line of it each -- so
# it is run against the reference at `variant=None` and any difference at all is a failure.
echo "=== indexer transcription identity ==="
"$PY" "$DRIVER" --config "$CONFIG" --mode indexer-identity

# The floors, before the kernels they will score. Two independent ones for the GDN operator (the
# reference ships two implementations of the same recurrence) and one for the MoE path
# (`grouped_mm` against the class body), plus the fp64 rounding floor per operator below.
echo "=== floors ==="
"$PY" "$DRIVER" --config "$CONFIG" --mode gdn-equiv
"$PY" "$DRIVER" --config "$CONFIG" --mode moe-equiv
for salt in "${SALTS[@]}"; do
    "$PY" "$DRIVER" --config "$CONFIG" --mode decode --defect None --salt "$salt" --dtype float64 \
        --out "$OUT/gold-decode-$salt-fp64.bin" >/dev/null
    echo "--- fp32 vs fp64, $salt"
    "$PY" "$DRIVER" --by-operator "$OUT/gold-decode-$salt-None.bin" "$OUT/gold-decode-$salt-fp64.bin"
done

# `rc` is declared HERE rather than beside the vendored census below, because the tolerance step
# now feeds it: `set -e` would abort the run at a refusal there and the census -- the one line that
# tells a reader how much of this run was real -- would never print. Every remaining check records
# into `rc` and the script exits once, at the bottom, carrying all of them.
rc=0

# The per-operator tolerance rows, DERIVED from the goldens this run just wrote rather than read
# off a table by hand. This is the command `common/tolerance.rs`'s QWEN block cites.
echo "=== tolerance rows ==="
if ! "$PY" "$DRIVER" --tolerance-table "$OUT"; then
    echo "the tolerance derivation REFUSED -- nothing above it is citable; see the reason" >&2
    rc=1
fi

# The real-parameter n-gram micro-anchor: the tiny config cannot see the 20M hash space, so the
# three multipliers, the 16 prime head vocabularies, the 16 offsets and a pinned window's ids are
# generated at FULL width and vendored separately.
echo "=== real-parameter n-gram micro-anchor ==="
"$PY" "$DRIVER" --config "$CONFIG" --mode ngram-real --out "$OUT/gold-ngram-real.bin"

# The vendored bytes are the whole point of generating these once, so a regeneration that no
# longer reproduces them is the thing most worth knowing.
#
# Enumerated from the VENDORED files rather than from `SALTS`, and it reports a census. K3's
# script learned this the hard way: driving the loop from `SALTS` made a narrowed run print one
# cheerful "reproduced byte-for-byte" and exit 0 while the other vendored golden went unchecked.
# (K3's own copy of this block also globs `$ROOT/tests/k3-anchor-decode-*.bin`, a path that has
# not existed since the anchors moved under `crates/oracles/tests/` -- so its census reports zero
# and it exits 1 on the "reproduced NOTHING" arm. Reported to the coordinator; not track B's file
# to fix.)
# The expected vendored set, as `<vendored basename>:<fresh path>` pairs. Enumerated rather than
# globbed, and that is a correction of the shape K3's copy of this block still carries: a
# `nullglob` array mixing a pattern with a LITERAL filename keeps the literal even when the file
# is absent, so a missing vendored golden reported as "DIFFERS" instead of as missing (observed
# here on the first run of this script). Enumerating also makes an unvendored file a named
# failure rather than a silently shorter loop.
VENDORED=(
    "qwen-anchor-decode-qwen-anchor-1:$OUT/gold-decode-qwen-anchor-1-None.bin"
    "qwen-anchor-decode-qwen-anchor-2:$OUT/gold-decode-qwen-anchor-2-None.bin"
    "qwen-anchor-prefill-qwen-anchor-1:$OUT/gold-prefill-qwen-anchor-1-None.bin"
    "qwen-anchor-ngram-real:$OUT/gold-ngram-real.bin"
)
verified=0
unverified=()
missing=()
for pair in "${VENDORED[@]}"; do
    base=${pair%%:*}
    fresh=${pair#*:}
    vendored=$TESTS/$base.bin
    if [[ ! -f $vendored ]]; then
        missing+=("$base")
    elif [[ ! -f $fresh ]]; then
        unverified+=("$base")
    elif cmp -s "$fresh" "$vendored"; then
        echo "vendored $base reproduced byte-for-byte"
        verified=$((verified + 1))
    else
        echo "DIFFERS from $vendored -- re-vendor deliberately, or find out why it moved" >&2
        rc=1
    fi
done
vendored_goldens=("${VENDORED[@]}")
if ((${#missing[@]})); then
    echo "NOT VENDORED IN TREE: ${missing[*]}" >&2
    echo "  Copy the fresh bytes over and update the pins in qwen_anchor.rs -- vendoring is a" >&2
    echo "  reviewed change, not a side effect of running this script." >&2
    rc=1
fi
# The census line stays, and it is what makes a recorded green interpretable: "verified: 4 of 4"
# is the claim, and a reader of a pasted log can see the denominator rather than infer it.
echo "vendored goldens verified: $verified of ${#vendored_goldens[@]}"
# **The EXIT CODE carries that census, and that is a 2026-09-01 correction.** This branch used to
# print to stderr and leave `rc` alone, and the `verified == 0` guard below could not fire while
# three of four had reproduced -- so
# `QWEN_ANCHOR_SALTS=qwen-anchor-1 crates/oracles/tests/qwen-anchor.sh` regenerated half the
# matrix, derived SINGLE-DRAW floors, printed "verified: 3 of 4" to stdout with the reason on
# stderr, and exited **0** while this file's own header says the run is two draws. OBSERVED, not
# reasoned: that arm was run on 2026-09-01 and returned exit 0 (`gate-red-proofs.md` section 14).
# A green whose denominator is wrong is the false-green class this whole anchor exists inside.
#
# A narrowed run is a legitimate thing to DO -- it is how a single row gets re-scored cheaply --
# and never a legitimate thing to PASS. So the rule is the census itself: exit 0 only when every
# vendored golden was re-derived from the reference on this run.
if ((${#unverified[@]})); then
    echo "NOT VERIFIED by this run (no fresh golden): ${unverified[*]}" >&2
    echo "  Their bytes are still FNV-pinned by crates/oracles/tests/qwen_anchor.rs, so they" >&2
    echo "  cannot drift unnoticed -- but nothing here re-derived them from the reference." >&2
fi
if ((verified < ${#vendored_goldens[@]})); then
    echo "NOT A FULL ANCHOR RUN: $verified of ${#vendored_goldens[@]} vendored goldens were" >&2
    echo "  re-derived. Re-run without QWEN_ANCHOR_SALTS/QWEN_ANCHOR_MODES before citing" >&2
    echo "  anything above -- the floors and weakest defects are per-draw worst cases." >&2
    # A regeneration that reproduces NOTHING is the extreme of the same failure, and it is what an
    # empty $OUT or a wrong --out path looks like from the outside.
    if ((verified == 0)); then
        echo "  no vendored golden was reproduced at all -- did the driver write to \$OUT?" >&2
    fi
    rc=1
fi
exit $rc
