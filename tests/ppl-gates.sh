#!/usr/bin/env bash
# M10 gates: the three claims the phase profile and the teacher-forced scorer make.
#
#   tests/ppl-gates.sh [--expect-red] <artifact-dir> [cell]
#   cell: profile | p4 | tf | all   (default all)
#
# | cell      | claim                                                        | red-proof |
# |-----------|--------------------------------------------------------------|-----------|
# | `profile` | the stamped buckets account for the decode wall              | drop one arm's accumulation (source, reverted) |
# | `p4`      | P4 at NLL: `--max-mem` moves speed, never text               | `--red-proof-corpus`, a one-word-different corpus in arm B |
# | `tf`      | the rewrite scores the reference's text equivalently         | off-by-one the TF position (source, reverted) |
#
# `--expect-red[=FRAGMENT]` inverts the classification so a red-proof is judged by the SAME
# code the green is; the argument is at the flag, below.
#
# **This gate NEVER builds.** Both binaries arrive prebuilt (a cargo run between arms
# evicts page cache: ms/miss 1.36 -> 5.14, measured). The source-mutation red-proofs DO
# require a rebuild, which is why they are separate invocations and not a mode: rebuild,
# run with --expect-red, `git checkout` the file, run again green.
#
# Cost, release profile, `tests/ppl-corpus.txt` (762 tokens) on GLM int3-vq at the
# baseline's 2.58 tok/s: profile ~4 min (512 decoded tokens), p4 ~18 min (THREE scoring
# arms since the re-spec — A, the control A', and B; plus a FOURTH, B'', only on the strict
# branch's red path, so ~24 min in that case), tf ~6 min (one reference arm; it
# reuses p4's arm A under `all`) — **~28 min for `all`**, plus ~6 min per red-proof re-run.
# Measured 2026-08-17 at 1.84 tok/s under a foreign host tenant, so budget ~40 min when the
# box is not idle. `PPL_CORPUS=tests/ppl-corpus-5000.txt PPL_CTX=8192` is the powered
# variant and costs ~6.5x; take it when a `tf` interval comes back INCONCLUSIVE, not
# before.
#
# Exit: 0 green (or, under --expect-red, confirmed red)
#       1 RED (or, under --expect-red, the gate refused to go red)
#       2 setup error   3 arm discarded (foreign GPU tenant) — rerun
#       66 GPU lock file missing (house convention: /run is tmpfs, dies on reboot)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${PPL_BIN:-${CARGO_TARGET_DIR:-$HERE/../target}/release/rivoli}
PPL_TOOL=${PPL_TOOL:-${CARGO_TARGET_DIR:-$HERE/../target}/release/ppl}
# The reference pin (tag `archive/glimmer-s2`, commit 6b7f496e) built WITH `--features teacher-forcing`
# — the parity gate's own PARITY_REF_BIN is not it: that binary has no `--ppl` at all
# (checked 2026-08-16), because the pin was built for `--bench`. Build the scoring one
# into its OWN target dir rather than over the parity gate's:
#   cd .claude/worktrees/ref-pin && CARGO_TARGET_DIR=/var/cache/users/rhansen/m10-ref-tf-target \
#     cargo build --release --locked --features rocm,teacher-forcing
REF_BIN=${PPL_REF_BIN:-/var/cache/users/rhansen/m10-ref-tf-target/release/rivoli}
# `ppl-corpus.txt` is a byte-exact PREFIX of `ppl-corpus-5000.txt` (52 of its 370 lines),
# so the powered run's first 762 tokens score the same text the default run does — edit
# one and you must edit the other, or the two stop being comparable.
CORPUS=${PPL_CORPUS:-$HERE/ppl-corpus.txt}
# **MEM_A is not just "the first budget" — the CONTROL pair runs there**, so MEM_A must be
# the budget you want the determinism answer for, and it is easy to set backwards. To ask
# "does streaming perturb the output?", put the STREAMING budget on MEM_A and the roomy one
# on MEM_B; setting it the other way round runs the control fully resident and answers a
# question nobody asked (caught while specifying the Glimmer experiment, 2026-08-17).
MEM_A=${PPL_MEM_A:-115}
MEM_B=${PPL_MEM_B:-70}
# 512, not a knob: it is what makes this run comparable to the row in
# `docs/measurement/baseline-2026-08-16.md`, and a knob with no caller that silently
# decouples the two is worth less than the line it costs. Same argument as `PROMPT` below.
NGEN=512
# The KV slab, and the reason it is a knob rather than the CLI default: `--ctx` defaults to
# 4096 and `tests/ppl-corpus-5000.txt` does not fit under it, so the powered re-run this
# file tells you to take (`tf` INCONCLUSIVE) would refuse at the door instead of scoring.
# ~51 KB of device memory per token, so raising it competes with --max-mem for the budget.
CTX=${PPL_CTX:-4096}
# The architecture-shaped flags, so `profile` and `p4` are not GLM-only. Caller today:
# experiment 0 in `docs/investigations/glm-nondeterminism.md` (Glimmer at two budgets,
# `PPL_MODE_FLAGS='--attn dense'`). A dense arm must NOT inherit `--mode`, which is
# `FallbackLoudly` there — the run would proceed with a warn and the `.nll` label would
# record a routed format nothing spent. (Residue, unfixed: `nll.rs` labels from
# `Args::mode`'s clap default, so a dense run's header still says `mode=int3-vq`. This gate
# compares bodies and does not care; a reader of the file should.)
read -ra MODE_FLAGS <<<"${PPL_MODE_FLAGS:---mode int3-vq --attn dense}"
# The `profile` cell's prompt. Pinned as a constant because
# `docs/measurement/baseline-2026-08-16.md` recorded its own only as `<P>` plus a prose
# description — so the 2.58 tok/s row is NOT byte-reproducible, and an attribution run
# that invented a fresh prompt each time would inherit that. This one matches the
# recorded description (a long technical essay; ~80 tokens after framing) and is now the
# reproducible one. A tok/s far from 2.58 means the regimes differ and the attribution
# needs saying so, not silently ranking against the old row.
PROMPT=${PPL_PROMPT:-"Write a long, detailed technical essay explaining how virtual memory works in a modern operating system: page tables, the TLB, demand paging, and what happens on a page fault. Then compare that machinery to how a GPU manages its own memory, and explain which ideas carry over and which do not."}
LOCK=/var/run/sys-gpu.lock
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/ppl-gates.XXXXXX")
# FORCED, not defaulted. Every subject this gate reads — `PROFILE/tok:`, `TF row-coherence
# held` — is a `tracing::info!`, and an inherited `RUST_LOG=warn` deletes both: the profile
# cell would go RED on a missing line and, under --expect-red, an environment variable
# would manufacture a confirmed proof. This gate measures an engine, not a log level, so it
# pins the one it reads instead of inheriting it. (`crates/cli/src/bench.rs` makes the same
# argument for putting BENCH on stderr outside tracing.)
export RUST_LOG=info
# gawk: the profile classifier needs 3-argument `match(s, re, arr)`. Under a POSIX awk that
# is a syntax error, the cell reds with "did not parse", and an environment fault reads as
# an engine RED — so it is checked here rather than left to chance.
awk 'BEGIN { if (match("a1", /([0-9])/, m) && m[1] == 1) exit 0; exit 1 }' </dev/null 2>/dev/null ||
    { echo "FAIL: awk here has no 3-argument match() — the profile classifier needs gawk. This would otherwise surface as a bogus 'PROFILE/tok line did not parse' RED." >&2; exit 2; }

# shellcheck source=tests/gpu-witness.sh
. "$HERE/gpu-witness.sh"

EXPECT_RED=""
EXPECT_FRAG=""
RED_CORPUS=""
while :; do
    case ${1:-} in
    # `--expect-red=FRAGMENT` requires the red MESSAGE to contain FRAGMENT. Without it any
    # red counts, and this repo has the receipt: gate-red-proofs.md §4 records a red-proof
    # that "passed" while reddening the WRONG test, against a stale binary. A missing
    # PROFILE line, an unparseable format and a dropped ffn bucket are three different
    # reds, and only one of them is the proof you planted. Bare `--expect-red` still works
    # and still accepts any red — it is the weaker claim, and the doc says which was used.
    --expect-red=*) EXPECT_RED=1; EXPECT_FRAG=${1#--expect-red=}; shift ;;
    --expect-red) EXPECT_RED=1; shift ;;
    # `p4`'s red-proof needs no rebuild: it is the COMPARISON that must be shown able to
    # see a difference, so arm B scores a corpus with one word changed. Scope, stated
    # plainly: this proves the byte comparison and the anti-vacuity check are live. It
    # does NOT simulate a residency-dependent format defect — the only real one on record
    # is `--mode hybrid`, whose format follows the cache, and hybrid does not decode in
    # this tree (it refuses at `FormatPlan`). When it lands, IT is this cell's red-proof.
    --red-proof-corpus) RED_CORPUS=1; EXPECT_RED=1; shift ;;
    *) break ;;
    esac
done
ARTIFACT=${1:?usage: ppl-gates.sh [--expect-red[=FRAGMENT]] [--red-proof-corpus] <artifact-dir> [profile|p4|tf|all]}
CELL=${2:-all}

[ -x "$BIN" ] || { echo "FAIL: rewrite binary missing: $BIN (cargo build --release --features teacher-forcing BEFORE the gate — never between arms)" >&2; exit 2; }
[ -e "$LOCK" ] || { echo "FAIL: GPU lock file missing: $LOCK" >&2; exit 66; }
[ -d "$ARTIFACT" ] || { echo "FAIL: artifact dir missing: $ARTIFACT" >&2; exit 2; }
[ -f "$CORPUS" ] || { echo "FAIL: corpus missing: $CORPUS" >&2; exit 2; }
# A build without `--features teacher-forcing` REFUSES `--ppl`, and the p4/tf cells would
# then "red" on a build flag that never reached a logit. Checked at the door — by PROBING,
# not by reading `--help`.
#
# **`--help` cannot answer this and the first draft of this check was vacuous for it.**
# `--ppl` is an unconditional clap field: `main.rs` carries no `#[cfg]` at all, deliberately
# ("a visible flag that names its build requirement beats one that vanishes"), so the flag
# appears in the help of every build including the stock one. The probe instead runs the
# real door — `Engine::ensure_scoring`, a compile-time cfg check that touches no device, so
# this needs no flock — against a directory that is not a model. A stock build refuses with
# `TF_SCORING_NOT_BUILT` before the tokenizer loads; an instrumented one gets past it and
# fails on the bogus artifact instead.
if [ "$CELL" != profile ]; then
    # The REAL artifact (the architecture sniff runs before either door and an empty dir
    # fails there instead — the first draft's mistake) plus a corpus that does not exist,
    # which fails AFTER both doors and BEFORE `Engine::open`. So the probe reaches every
    # compile-time refusal and never reaches the device: no flock, no GTT, no tenant. It
    # costs one 19 MB vocab parse.
    PROBE_CORPUS="$SCRATCH/no-such-corpus.txt"
    "$BIN" "$ARTIFACT" --ppl "$PROBE_CORPUS" --ppl-out "$SCRATCH/probe.nll" \
        >"$SCRATCH/probe.log" 2>&1 || true
    # THREE-WAY, with the unknown case fatal. Two greps and an implicit "else pass" would
    # be the silent-on-the-unknown-case classifier this repo has been bitten by: any
    # change to the startup order (a door moving, `Engine::open` moving earlier) would
    # make both greps miss and wave a broken build straight through to the arms.
    if grep -q 'NO compute backend' "$SCRATCH/probe.log"; then
        # `ensure_backend` runs BEFORE `ensure_scoring`, so this refusal comes first and
        # in different words. Easy to hit by accident: the last
        # `cargo test --no-default-features` in a worktree leaves exactly such a binary at
        # the default path, and it was this check's own first probe (2026-08-16).
        echo "FAIL: $BIN is a deviceless (--no-default-features) build and cannot decode at all. Rebuild: cargo build --release --features teacher-forcing" >&2
        exit 2
    elif grep -q 'not in this build' "$SCRATCH/probe.log"; then
        echo "FAIL: $BIN refuses --ppl — build it with --features teacher-forcing. Every scoring cell would otherwise redden on the build, not on the engine." >&2
        exit 2
    elif ! grep -qF "$PROBE_CORPUS" "$SCRATCH/probe.log"; then
        echo "FAIL: the teacher-forcing probe reached neither door nor the corpus read — the startup order changed and this check now proves nothing. See $SCRATCH/probe.log" >&2
        exit 2
    fi
    # `bin/ppl` is needed by both scoring cells and by neither profile — the same set this
#    block already guards, so it is asserted once here instead of in each cell.
    [ -x "$PPL_TOOL" ] || { echo "FAIL: bin/ppl missing: $PPL_TOOL (cargo build --release builds it alongside rivoli)" >&2; exit 2; }
    echo "== door: $BIN passes ensure_backend + ensure_scoring (probed, deviceless)"
fi

say() { echo "== cell: $1"; }
# ok/red record to FILES, not to shell variables, because each cell runs in a subshell
# (see `run_cell`) and a variable would not survive it. The files are also the evidence
# directory's own tally.
ok() { echo "$1" >>"$SCRATCH/passes"; echo "   ok: $1"; }
# The ONE classifier. Both a green run and an --expect-red run come through here, which
# is what makes the proof a proof of THIS gate rather than of a bespoke reading.
#
# Exits, and the CELL is what ends: a cell that has gone red has nothing further to say.
# `run_cell` catches that and moves to the next cell, which is the fix for the way this
# script behaved on its first real battery — `p4` reddened and `tf` NEVER RAN, so the one
# cell that validates scoring against the pinned reference was lost to an unrelated
# failure two cells earlier (2026-08-17). Cells are independent; the run should be too.
red() {
    if [ -n "$EXPECT_RED" ]; then
        if [ -n "$EXPECT_FRAG" ] && ! printf '%s' "$2" | grep -qF -- "$EXPECT_FRAG"; then
            echo "RED-PROOF FAILED: '$1' went red, but on '$2' — which does not contain '$EXPECT_FRAG'. A red for the wrong reason is not the proof you planted." >&2
            exit 4
        fi
        echo "   RED (as the proof demanded) in '$1': $2"
        : >"$SCRATCH/proof.$1"
        exit 0
    fi
    echo "PPL-GATE RED in cell '$1': $2 — see $SCRATCH" >&2
    : >"$SCRATCH/red.$1"
    exit 1
}

# `set -e` IS SUPPRESSED INSIDE EVERY CELL, and that is not a choice — it is bash. A
# subshell that is the left operand of `||` runs with `-e` disabled *inside* it, and the
# suppression cannot be lifted: `( set +e; set -e; ... )` and a bare `( ... ) || rc=$?` both
# still run past a failing command (measured 2026-08-17; only a fresh `bash -c` re-entry
# restores it, which is too heavy for a cell). **So every failure inside a cell is checked
# explicitly** — `run_arm`/`score_arm` return their rc into a tested variable, the `cmp`s sit
# in `if`, and the awk/grep parses are followed by an emptiness or row-count assertion. If
# you add a step to a cell, check its status; nothing else will.

# --- cell: profile — the stamped buckets account for the decode wall ------------------
# PRE-REGISTERED BAND, written here before the first run (2026-08-16, no number observed
# yet), and it has two independent halves because a band alone is not enough:
#
#  1. `other` (= wall - the four named buckets) in [-0.5%, +15%] of wall. The LOWER bound
#     is the sharp one and is tight by construction, not by measurement: every bucket is a
#     disjoint span on the decode thread, so named CANNOT exceed wall, and a negative
#     `other` means one span was stamped into two buckets. The UPPER bound is loose on
#     purpose — the unstamped work is host-side launch glue (`Emit::offer`, the sink, the
#     embed and flag launches, the per-layer pre-norm + gate GEMV launch, `Instant::now`
#     itself), microseconds against a ~390 ms token, so the honest prediction is well
#     under 1% and 15% is ~50x headroom. A band that a green only just clears is a band
#     tuned to pass; this one is not, and the first green run records the observed value
#     so it can be TIGHTENED with provenance.
#  2. A per-bucket CENSUS: every bucket the arm's table says is stamped must be > 0. This
#     is what catches a dropped accumulation the band cannot see — `head` is a few percent
#     of a GLM token, so dropping it moves `other` by less than the band's width while
#     making `head` exactly 0.0.
#
# `fetch-wait` is deliberately NOT in the census: 0.0 is its SUCCESS value (fetch fully
# hidden behind resident compute is the whole design), so requiring it non-zero would gate
# on the engine performing badly. The cost is a known blind spot — a dropped `fetch-wait`
# stamp is invisible to this cell — and it is stated rather than papered over.
#     DERIVED, not round: the true floor is 0 (disjoint spans on one thread cannot sum
#     past the wall they sit inside), and the only slack owed is print rounding. Six terms
#     at `{:.3}` ms carry at most 6 x 0.0005 = 0.003 ms of it, which on a ~390 ms token is
#     0.0008%. -0.05% is ~60x that, and still 300x tighter than the -0.5% this said until
#     the reviews pointed out that "tight by construction" and "-0.5" are different claims.
OTHER_LO=-0.05
OTHER_HI=15.0
#  3. A CROSS-CHECK of `other` against the four buckets on the same line. `other_ms` is the
#     only derived number the engine reports, and a gate that reads it without re-deriving
#     it is auditing arithmetic by asking the arithmetic. Concretely: hard-code
#     `other_ms: 0.0` in `from_decode` and halves 1 and 2 both stay green (0% is inside the
#     band; three buckets are still non-zero). Tolerance 0.02 ms — 4x the 0.003 ms of
#     rounding six `{:.3}` prints can carry, and ~5e-5 of a GLM token.
OTHER_EPS=0.02

cell_profile() {
    say "profile: bucket sum vs decode wall (${MODE_FLAGS[*]}, --bench $NGEN)"
    local rc=0
    # `PPL_REPLAY_LOG` re-classifies a SAVED arm log instead of decoding, so the
    # CLASSIFIER half of this cell's red proof needs no device (that half is PAID —
    # `docs/measurement/gate-red-proofs.md` §5a). A replayed result is evidence about the
    # parser and never about the engine, and the summary line says so.
    if [ -n "${PPL_REPLAY_LOG:-}" ]; then
        [ -f "$PPL_REPLAY_LOG" ] || { echo "FAIL: PPL_REPLAY_LOG not readable: $PPL_REPLAY_LOG" >&2; exit 2; }
        cp "$PPL_REPLAY_LOG" "$SCRATCH/profile.log"
        # A marker FILE, not a variable: this runs inside the cell's subshell, so a variable
        # would never reach the summary line — which is how the first version of this shipped,
        # printing a plain `PPL-GATES GREEN` for a classifier-only run and making it
        # indistinguishable from a real one. That is the false-green class this repo tracks,
        # in the code meant to prevent it (review, 2026-08-17).
        : >"$SCRATCH/replayed"
        echo "   REPLAY of $PPL_REPLAY_LOG — classifier only, no device, no claim about the engine"
    else
        run_arm profile "$SCRATCH/profile.out" "$SCRATCH/profile.log" \
            "$BIN" "$ARTIFACT" "${MODE_FLAGS[@]}" --ctx "$CTX" --max-mem "$MEM_A" \
            --prompt "$PROMPT" --bench "$NGEN" || rc=$?
        [ "$rc" -eq 0 ] || { echo "FAIL: profile arm rc=$rc — see $SCRATCH/profile.log" >&2; exit 2; }
    fi
    # Anti-vacuity: the line must EXIST. If the report is reworded or silently stops being
    # emitted, this cell must go red rather than find nothing and pass over it.
    local line
    line=$(grep -F 'PROFILE/tok:' "$SCRATCH/profile.log" | tail -1 || true)
    [ -n "$line" ] || red profile "no PROFILE/tok line on the run's log — the report was reworded, or Emit::finish stopped emitting it"
    echo "   $line"
    # One awk, one parse: the classifier reads the same six numbers the report printed.
    local verdict
    verdict=$(awk -v lo="$OTHER_LO" -v hi="$OTHER_HI" -v eps="$OTHER_EPS" '
        match($0, /wall ([0-9.]+)ms = attend ([0-9.]+) \+ ffn ([0-9.]+) \+ fetch-wait ([0-9.]+) \+ head ([0-9.]+) \+ other (-?[0-9.]+)/, m) {
            wall=m[1]+0; other=m[6]+0
            if (wall <= 0) { print "SETUP wall " wall " is not positive"; exit }
            # DATA, not three hand-written ifs, so the examined count is real rather than
            # true by construction. fetch-wait (m[4]) is excluded on purpose — see the
            # census note in the pre-registration block above.
            split("attend ffn head", name, " "); split("2 3 5", col, " ")
            n = 0
            for (k = 1; k in name; k++) {
                v = m[col[k]] + 0
                if (v <= 0) { printf "RED %s bucket is %.3f — the accumulation was dropped\n", name[k], v; exit }
                named += v; n++
            }
            if (n != 3) { print "SETUP census examined " n " buckets, expected 3"; exit }
            # Re-derive the one DERIVED number from the four measured ones on the same
            # line — item 3 of the pre-registration block above says why.
            derived = wall - (named + m[4])
            if ((derived - other) > eps || (other - derived) > eps) {
                printf "RED the reported other %.3f ms disagrees with wall - buckets = %.3f ms (eps %.3f) — `other` is not the remainder it claims to be\n", other, derived, eps
                exit
            }
            pct = 100*other/wall
            if (pct < lo) { printf "RED other is %.3f%% of wall, under the %.2f%% floor — a span is stamped into two buckets\n", pct, lo; exit }
            if (pct > hi) { printf "RED other is %.3f%% of wall, over the %.2f%% ceiling — wall is going somewhere unstamped\n", pct, hi; exit }
            printf "GREEN other %.3f%% of wall (band [%.2f, %.2f]), census 3/3 non-zero, remainder re-derived to within %.3f ms\n", pct, lo, hi, eps
            exit
        }
    ' <<<"$line")
    [ -n "$verdict" ] || red profile "PROFILE/tok line did not parse — the format changed: $line"
    case $verdict in
    SETUP*) echo "FAIL: $verdict" >&2; exit 2 ;;
    RED*) red profile "${verdict#RED }" ;;
    *) ok "${verdict#GREEN }" ;;
    esac
    # Not a gate, the DELIVERABLE: the per-phase decomposition this milestone owes.
    grep -F 'DECODE' "$SCRATCH/profile.log" | tail -1 || true
    # The cell TESTS a ratio, which survives a loaded box; it PRODUCES an absolute wall,
    # which is comparable only across comparable host states. 2.0 is the bound because this
    # arm's own decode thread plus its fetch reaper contribute ~1-2 on 32 cores, so above
    # that a foreign host tenant is running — see `gpu-witness.sh::host_load`. Recorded, not
    # enforced: green either way.
    awk -v l="${ARM_LOAD_BEFORE:-0}" 'BEGIN { exit (l+0 > 2.0) ? 0 : 1 }' && cat <<CAVEAT
   ATTRIBUTION CAVEAT: host loadavg was ${ARM_LOAD_BEFORE} when this arm started, so a
   foreign CPU/NFS tenant was running. The GATE above stands (it is a ratio), but the
   absolute wall and the per-phase milliseconds are NOT comparable to a baseline taken at
   another load. Re-run idle before citing them. On 2026-08-17 exactly this produced 1.84
   tok/s against a 2.58 baseline with the miss count UNCHANGED.
CAVEAT
    true
}

# --- cell: p4 — the memory knob moves speed, never text -------------------------------
# P4 at NLL. `--mode int3-vq` picks ONE arithmetic for every expert, so residency cannot
# select a format the way `--mode hybrid`'s cache does, and the budget must not move the
# text.
#
# > **RE-SPECIFIED 2026-08-17: byte-identity across budgets was a SPECIFICATION ERROR and
# > could never have passed** — the control (same budget twice) moves as much as the budget
# > does, so the cell was measuring nondeterminism and reddening whatever `--max-mem` did.
# > The differing-position COUNT, which it reported as evidence, turned out to have no
# > discriminating power at all. Measurements and bounds:
# > `docs/investigations/glm-nondeterminism.md`. Why the replacement has this shape and what
# > proved it: `docs/measurement/gate-red-proofs.md` §5a-2. Neither is restated here.
#
# **The cell now calibrates its own strictness from a control arm it has to run anyway**,
# rather than from a flag someone can set wrong or an arm name someone can forget to
# update:
#
#   A  @ MEM_A          the baseline
#   A' @ MEM_A          the CONTROL — same budget, same flags, same corpus
#   B  @ MEM_B          the test
#
#   if body(A) == body(A')   the arm repeats itself AT THIS LENGTH, so demand byte-identity
#                            of B. This is the STRICT gate, and it is what Glimmer gets —
#                            automatically, because Glimmer's control comes back identical.
#                            A differing B does NOT convict on the spot: it runs a fourth
#                            arm, B'' at MEM_B, because a one-off divergence would not
#                            repeat while a real budget effect would. See that branch for
#                            the false-conviction arithmetic and for why repeating the
#                            BUDGET beats repeating the control.
#   else                     the arm does not repeat itself, so the floor is MEASURED:
#                            `bin/ppl` pairs both A'→A and B→A, and the verdict reads the
#                            two 95% intervals as CALIBRATION and TEST.
#
#                              control EXCLUDES 0  -> UNCALIBRATED, exit 1, never a pass.
#                                  The jitter is not zero-mean, so an interval away from
#                                  zero cannot be attributed to the budget rather than to
#                                  whatever biases a repeat.
#                                  **Expect this ~5% of the time even on a healthy arm**: two
#                                  runs of one config differ by symmetry with mean zero, and
#                                  a 95% interval on a zero-mean quantity excludes zero one
#                                  time in twenty. So ONE UNCALIBRATED is a re-run, not a
#                                  finding; TWO in a row says the jitter is biased, which is
#                                  a real result and belongs in
#                                  docs/investigations/glm-nondeterminism.md.
#                              control contains 0, budget EXCLUDES 0  -> RED. A systematic
#                                  shift the noise does not account for: a P4 violation.
#                              both contain 0  -> GREEN, and the control's half-width is
#                                  reported as the resolution the run actually had.
#
# A'''s and B'''s intervals are both paired against A, so a "must not overlap" rule (used for
# one draft) double-counts A'''s noise and comes out ~2x less sensitive than the data
# supports; the interval already estimates the effect at the right SE, so the question is
# whether it clears zero, and the control'''s job is to prove zero is where a null lands.
# Derivation, the synthetic red-proof intervals, and the resolution this buys against the
# ladder'''s 0.0134-0.0172-nat gaps: `docs/measurement/gate-red-proofs.md` §5a-2.
#
# No magnitude floor is applied on purpose: P4 is an INVARIANT, not a budget, so a small
# systematic effect is still a violation and must not be excused by being small.
#
# **Stated blind spot:** a budget change that adds pure VARIANCE with zero mean shift is
# invisible to a test on the mean. The sd ratio is reported so a reader can see it; it is
# not gated, because no principled ratio has been measured.
nll_body() { grep -v '^#' "$1"; }
hit_of() { sed -n '1s/.*hit_pct=\([0-9.]*\).*/\1/p' "$1"; }
moved() { nll_body "$1" | paste - <(nll_body "$2") | awk '$1!=$2{c++} END{print c+0}'; }

# The 95% CIs from `bin/ppl`'s TABLE, one per line, in argument order.
#
# Truncated at the `1% PPL bar` line first, and that is not tidiness: the verdict block
# below it prints intervals too (`INCONCLUSIVE — interval [lo, hi]`), so an untruncated
# grep can return a verdict's interval as if it were a table row. The `tf` cell took
# `head -1` off an untruncated grep until this cell needed two rows and the hazard became
# visible (2026-08-17).
table_cis() { sed '/1% PPL bar/q' "$1" | grep -oE '\[[-+][0-9.]+, [-+][0-9.]+\]'; }

score_arm() { # $1 = arm name, $2 = max-mem, $3 = corpus, $4 = out .nll
    local rc=0
    run_arm "$1" "$SCRATCH/$1.out" "$SCRATCH/$1.log" \
        "$BIN" "$ARTIFACT" "${MODE_FLAGS[@]}" --ctx "$CTX" --max-mem "$2" \
        --ppl "$3" --ppl-out "$4" || rc=$?
    [ "$rc" -eq 0 ] || { echo "FAIL: scoring arm '$1' rc=$rc — see $SCRATCH/$1.log" >&2; exit 2; }
    [ -s "$4" ] || { echo "FAIL: arm '$1' exited 0 but wrote no NLLs to $4" >&2; exit 2; }
    # The coherence check is per-position and refuses the run, so reaching here means it
    # held — but the count is echoed so a silently SHORT run cannot read as a full one.
    grep -F 'TF row-coherence held' "$SCRATCH/$1.log" | tail -1 ||
        { echo "FAIL: arm '$1' wrote NLLs without logging the coherence line — a different scorer ran" >&2; exit 2; }
}

cell_p4() {
    say "p4: does --max-mem $MEM_A -> $MEM_B move the text FURTHER than a repeat does?"
    local b_corpus=$CORPUS
    if [ -n "$RED_CORPUS" ]; then
        # One word changed, first line — a perturbation the comparison must see. Bigger
        # than a byte on purpose: sub-threshold perturbations are this repo's recorded
        # red-proof trap (a 1-ulp flip was erased by fp16 narrowing; one sign flip sat
        # under argmax margins). A different TOKEN cannot be erased by anything, and it
        # must clear the measured noise floor rather than merely being non-zero — which is
        # exactly the discrimination the re-specified cell exists to make.
        b_corpus="$SCRATCH/red-corpus.txt"
        sed '1s/transformer/convolutional/' "$CORPUS" >"$b_corpus"
        cmp -s "$CORPUS" "$b_corpus" &&
            { echo "FAIL: the red-proof corpus is identical to the corpus — the substitution missed, and the proof would pass vacuously" >&2; exit 2; }
        echo "   red-proof: arm B scores a one-word-different corpus"
    fi
    score_arm "p4a" "$MEM_A" "$CORPUS" "$SCRATCH/a.nll"
    score_arm "p4a2" "$MEM_A" "$CORPUS" "$SCRATCH/a2.nll"
    score_arm "p4b" "$MEM_B" "$b_corpus" "$SCRATCH/b.nll"
    # **LENGTH goes in every verdict this cell prints.** Divergence on a streaming arm is
    # STOCHASTIC in the sequence position at which it first fires — GLM is byte-identical at
    # 32 generated tokens and differs at 512, and on a quiet box its first divergence moved
    # from position 13 to 452 (docs/investigations/glm-nondeterminism.md). So "byte-identical"
    # and "the control spread is X" are meaningless without the count they were measured
    # over, and a reader comparing two runs of this gate at two corpus lengths must be able
    # to see that they are not the same experiment.
    local npos
    npos=$(nll_body "$SCRATCH/a.nll" | wc -l)
    # A floor, because "a check whose examined count can silently reach zero is not a check"
    # and this one can get close: `nll.rs` admits any corpus of >=2 tokens, so a 2-token file
    # yields npos=1 and the strict branch would print a perfectly cheerful
    # `1 per-token NLLs byte-identical`. 32 is not arbitrary — it is the length at which GLM
    # IS byte-identical (docs/investigations/glm-nondeterminism.md), i.e. the shortest run
    # anyone has a reason to cite, and the paired branch needs far more than that for its
    # interval to mean anything.
    [ "$npos" -ge 32 ] || { echo "FAIL: only $npos scored positions — too few for either branch to mean anything (byte-identity is trivial and the interval is not estimable). Use a corpus of at least 33 tokens." >&2; exit 2; }
    echo "   length: $npos scored positions (from $(basename "$CORPUS")) — every verdict below is scoped to it"
    local ha ha2 hb
    ha=$(hit_of "$SCRATCH/a.nll"); ha2=$(hit_of "$SCRATCH/a2.nll"); hb=$(hit_of "$SCRATCH/b.nll")
    echo "   hit_pct: A=${ha} A'=${ha2} (control) B=${hb}"
    echo "   positions moved: A vs A' = $(moved "$SCRATCH/a.nll" "$SCRATCH/a2.nll") (noise), A vs B = $(moved "$SCRATCH/a.nll" "$SCRATCH/b.nll") (noise + budget), of $npos"
    # Anti-vacuity, and it too is now control-relative. `ha != hb` was the original check
    # and it is worthless once the baseline wobbles: A and A' differ in hit% (78.2643 vs
    # 78.2352 measured) with no budget change at all, so "the hit rates differ" is true of
    # two IDENTICAL runs. The budget must move residency further than a repeat does.
    if [ -z "$RED_CORPUS" ]; then
        awk -v a="$ha" -v a2="$ha2" -v b="$hb" 'BEGIN {
            ctl = (a > a2) ? a - a2 : a2 - a
            bud = (a > b)  ? a - b  : b - a
            exit (bud > 10 * ctl && bud > 1.0) ? 0 : 1
        }' || {
            echo "FAIL: --max-mem $MEM_A -> $MEM_B moved hit% from $ha to $hb, which is not decisively more than the repeat's $ha -> $ha2. The two budgets did not produce meaningfully different residency, so this cell would prove nothing whichever way it came out. Widen PPL_MEM_A/PPL_MEM_B." >&2
            exit 2
        }
    fi
    # --- strictness, chosen by the control ------------------------------------------
    if nll_body "$SCRATCH/a.nll" | cmp -s - <(nll_body "$SCRATCH/a2.nll"); then
        echo "   control is BYTE-IDENTICAL over $npos positions — this arm repeats itself at THIS LENGTH, so P4 is gated strictly"
        if nll_body "$SCRATCH/a.nll" | cmp -s - <(nll_body "$SCRATCH/b.nll"); then
            ok "strict: $npos per-token NLLs byte-identical across budgets, on an arm whose control proved it byte-reproducible at that length"
            return 0
        fi
        # **CONFIRM BEFORE CONVICTING — a fourth arm, on the red path only, repeating the
        # BUDGET and not the control.** A one-off divergence does not recur, so `B == B''`
        # cannot happen by chance, while a REAL budget effect is a different but STABLE
        # output and still convicts. Repeating the CONTROL instead — the first draft of this
        # fix — leaves the likelihood ratio unchanged and so carries no information about the
        # budget at all. Arithmetic, and the exception if the divergence turns out to be a
        # two-attractor race rather than diffuse: `docs/measurement/gate-red-proofs.md` §5a-2.
        echo "   B differs. A one-off divergence would not REPEAT, so a second budget arm decides whether this is the budget or the wobble"
        score_arm "p4b2" "$MEM_B" "$b_corpus" "$SCRATCH/b2.nll"
        if nll_body "$SCRATCH/b.nll" | cmp -s - <(nll_body "$SCRATCH/b2.nll"); then
            red p4 "STRICT: the control repeated byte-for-byte over $npos positions AND both budget-$MEM_B arms agreed with each other while differing from it — a STABLE, reproducible budget effect, which is a P4 violation. $(moved "$SCRATCH/a.nll" "$SCRATCH/b.nll") positions moved; first: $(nll_body "$SCRATCH/a.nll" | paste - <(nll_body "$SCRATCH/b.nll") | awk '$1!=$2{print NR": "$1" vs "$2; exit}')"
        fi
        echo "PPL-GATE UNCALIBRATED: the control repeated over $npos positions but the two budget-$MEM_B arms disagree with EACH OTHER ($(moved "$SCRATCH/b.nll" "$SCRATCH/b2.nll") positions), so this experiment CANNOT SEPARATE a budget effect from the wobble — a real effect plus wobble at B looks the same. Not attributable either way. Not a pass and not a P4 verdict — a finding for docs/investigations/glm-nondeterminism.md. Re-run at a shorter corpus for a verdict, or wait for wave/fix-glm-determinism." >&2
        exit 1
    fi
    echo "   control moved $(moved "$SCRATCH/a.nll" "$SCRATCH/a2.nll") of $npos positions — this arm does not repeat itself at this length, so the floor is MEASURED (see docs/investigations/glm-nondeterminism.md)"
    "$PPL_TOOL" "$SCRATCH/a.nll" "$SCRATCH/a2.nll" "$SCRATCH/b.nll" | tee "$SCRATCH/p4-paired.txt"
    local cis n
    cis=$(table_cis "$SCRATCH/p4-paired.txt")
    n=$(printf '%s\n' "$cis" | grep -c . || true)
    # Anti-vacuity on the parse itself: two cells in, two table rows out. One row would
    # mean bin/ppl refused a cell and the comparison silently became control-vs-nothing.
    [ "$n" -eq 2 ] || { echo "FAIL: expected 2 table rows from bin/ppl (control, test), parsed $n — its table format changed, or a cell was refused. See $SCRATCH/p4-paired.txt" >&2; exit 2; }
    local verdict
    verdict=$(printf '%s\n' "$cis" | awk -v npos="$npos" '
        { gsub(/[][,]/, " "); if (NR == 1) { clo = $1 + 0; chi = $2 + 0 } else { tlo = $1 + 0; thi = $2 + 0 } }
        END {
            half = (chi - clo) / 2
            if (clo > 0 || chi < 0) {
                printf "UNCALIBRATED over %d scored positions the CONTROL interval [%+.5f, %+.5f] excludes zero — two identical runs differ systematically, so nondeterminism here is not zero-mean and no interval can be attributed to the budget. Not a pass and not a P4 verdict. Expect this ~1 run in 20 even on a healthy arm, so ONE is a re-run and TWO in a row is a finding for docs/investigations/glm-nondeterminism.md\n", npos, clo, chi
            } else if (tlo > 0 || thi < 0) {
                printf "RED over %d scored positions the budget CI [%+.5f, %+.5f] EXCLUDES zero against a control of [%+.5f, %+.5f] that contains it — --max-mem moved the text systematically, beyond what a repeat accounts for. P4 violation\n", npos, tlo, thi, clo, chi
            } else {
                printf "GREEN budget CI [%+.5f, %+.5f] contains zero, control [%+.5f, %+.5f] contains zero — no systematic budget effect over %d scored positions (noise half-width %.5f nats, which IS the resolution this length bought)\n", tlo, thi, clo, chi, npos, half
            }
        }')
    case $verdict in
    RED*) red p4 "${verdict#RED }" ;;
    # Neither green nor a P4 red, and it gets its own exit rather than being folded into
    # either — exactly as `tf` treats INCONCLUSIVE. Folding it into RED would blame the
    # budget for the engine's own wobble, which is the mistake the first draft of this
    # whole cell made.
    UNCALIBRATED*) echo "PPL-GATE $verdict" >&2; exit 1 ;;
    *) ok "${verdict#GREEN }" ;;
    esac
}

# --- cell: tf — the rewrite scores the reference's text equivalently -------------------
# PRE-REGISTERED EQUIVALENCE BAND, written here before the first run (2026-08-16):
#
#   the 95% CI on paired mean dNLL must lie ENTIRELY inside +/- ln(1.01) = +/- 0.00995 nats
#
# Two-sided, unlike `bin/ppl`'s own one-sided verdict, and that is the whole difference in
# question: `bin/ppl` asks "is this cell's quality COST within budget", where coming out
# better is a pass. Here the claim is that two engines compute the SAME thing, so a
# rewrite that scored reliably BETTER than the reference would be evidence of a defect
# (most likely a misaligned position), not a free lunch — the interval has to clear zero
# from both sides. The magnitude is not invented: ln(1.01) is the repo's own 1%-perplexity
# bar, the constant `bin/ppl` already computes and the one every historical quality claim
# here was ranked against.
#
# An interval WIDER than the band is INCONCLUSIVE, never a pass — re-run with
# PPL_CORPUS=tests/ppl-corpus-5000.txt PPL_CTX=8192. That is the underpowered-null rule, and it is why
# this reads the interval rather than the point estimate.
TF_BAR=0.00995

cell_tf() {
    say "tf: rewrite vs pinned reference, paired dNLL over $(basename "$CORPUS")"
    # **GLM-only, structurally**: the other side of this comparison is the pinned GLM
    # reference at tag `archive/glimmer-s2`, commit 6b7f496e, so pointing the rewrite arm at another
    # architecture compares two different models. It would not fail quietly — the corpora
    # tokenize to different lengths and the position-count check below would fire — but it
    # would fire as a RED naming the tokenizer, which is a wrong diagnosis for an operator
    # error. Refused here instead, in the words of the thing that is wrong.
    [ "${MODE_FLAGS[*]}" = "--mode int3-vq --attn dense" ] || {
        echo "FAIL: the tf cell pairs against the pinned GLM reference, so it only means anything with the default PPL_MODE_FLAGS; got '${MODE_FLAGS[*]}'. Run the p4 cell alone for a non-GLM arm." >&2
        exit 2
    }
    [ -x "$REF_BIN" ] || { echo "FAIL: reference scoring binary missing: $REF_BIN (see the build line at the top of this file)" >&2; exit 2; }
    echo "   ref: $(stat -c '%y' "$REF_BIN") $REF_BIN"
    echo "   new: $(stat -c '%y' "$BIN") $BIN"
    local rc=0
    run_arm tfref "$SCRATCH/tfref.out" "$SCRATCH/tfref.log" \
        "$REF_BIN" "$ARTIFACT" --mode int3-vq --attn dense --no-mtp --ctx "$CTX" --max-mem "$MEM_A" \
        --ppl "$CORPUS" --ppl-out "$SCRATCH/ref.nll" || rc=$?
    [ "$rc" -eq 0 ] || { echo "FAIL: reference scoring arm rc=$rc — see $SCRATCH/tfref.log" >&2; exit 2; }
    # Reuse `p4`'s arm A when this invocation already produced it: same binary, same
    # budget, same corpus, same flags, same $SCRATCH — so it is the SAME arm, not a
    # comparable one, and re-running it would buy ~6 minutes of nothing on a box where
    # device time is the scarce resource. Scoped to one invocation because $SCRATCH is;
    # `tf` run alone still takes its own arm.
    if [ -s "$SCRATCH/a.nll" ] && [ -z "$RED_CORPUS" ]; then
        echo "   reusing p4's arm A (same budget, corpus and flags, same invocation)"
        cp "$SCRATCH/a.nll" "$SCRATCH/new.nll"
    else
        score_arm "tfnew" "$MEM_A" "$CORPUS" "$SCRATCH/new.nll"
    fi
    local na nb
    na=$(nll_body "$SCRATCH/ref.nll" | wc -l); nb=$(nll_body "$SCRATCH/new.nll" | wc -l)
    # A length mismatch is a RED, not a setup error: the two engines tokenized the same
    # bytes with the same artifact, so different position counts mean the rewrite's
    # tokenizer or its scored-position protocol diverged — exactly what this cell exists
    # to catch. `bin/ppl` would refuse to pair them, which would otherwise read as a
    # tooling complaint.
    [ "$na" = "$nb" ] ||
        red tf "reference scored $na positions, rewrite $nb — the same text produced different position counts (tokenizer or off-by-one in the walk)"
    "$PPL_TOOL" "$SCRATCH/ref.nll" "$SCRATCH/new.nll" | tee "$SCRATCH/paired.txt"
    local ci
    # `table_cis`, not a bare grep: the verdict block prints intervals too, so an
    # untruncated `head -1` can return a VERDICT's interval when the table row is missing —
    # reading a number off the wrong line and calling it a measurement. Found while giving
    # `p4` a two-row parse (2026-08-17).
    local cis n
    cis=$(table_cis "$SCRATCH/paired.txt")
    n=$(printf '%s\n' "$cis" | grep -c . || true)
    # One cell in, one table row out. This subsumes an emptiness check on `ci` — a count of
    # exactly 1 cannot leave it empty — so there is no second assertion.
    [ "$n" -eq 1 ] || { echo "FAIL: expected exactly 1 table row from bin/ppl, parsed $n — its table format changed, or the cell was refused. See $SCRATCH/paired.txt" >&2; exit 2; }
    ci=$(printf '%s\n' "$cis" | head -1)
    local verdict
    verdict=$(awk -v bar="$TF_BAR" -v ci="$ci" 'BEGIN {
        gsub(/[][,]/, " ", ci); split(ci, v, " ")
        lo = v[1] + 0; hi = v[2] + 0
        if (lo > hi) { print "SETUP CI bounds are inverted: " ci; exit }
        if (lo >= -bar && hi <= bar) { printf "GREEN 95%% CI [%+.5f, %+.5f] inside +/-%.5f nats\n", lo, hi, bar; exit }
        if (lo > bar || hi < -bar) { printf "RED 95%% CI [%+.5f, %+.5f] lies ENTIRELY outside +/-%.5f — the two engines score differently\n", lo, hi, bar; exit }
        printf "INCONCLUSIVE 95%% CI [%+.5f, %+.5f] is wider than the +/-%.5f band — not a pass; re-run with PPL_CORPUS=tests/ppl-corpus-5000.txt PPL_CTX=8192\n", lo, hi, bar
    }')
    case $verdict in
    SETUP*) echo "FAIL: $verdict" >&2; exit 2 ;;
    RED*) red tf "${verdict#RED }" ;;
    INCONCLUSIVE*)
        # Neither green nor red. Its own exit code would be a fourth classification for
        # one caller to get wrong; it exits 1 with the word INCONCLUSIVE in the message,
        # and an --expect-red run does NOT get to count it as a proof.
        echo "PPL-GATE $verdict" >&2; exit 1 ;;
    *) ok "${verdict#GREEN }" ;;
    esac
}

# **A RED in one cell must not cost the others their run**, and the first attempt at this
# did exactly that: `(run_cell "$c") || exit $?` put the subshell in a `||` list whose
# right-hand side re-raised the cell's exit code, so `p4` reddening still killed `tf` — the
# very failure it was written to fix. Reproduced and fixed 2026-08-17; the mapping now lives
# HERE, at the one place the codes are read, with no `exit` on a red.
#
#   0  green            1  RED (already recorded by `red()` in a marker file)
#   4  a red-proof that fired on the WRONG defect — hard stop, already explained
#   2  setup     3  arm discarded     both abort: the measurement could not be taken at all,
#                                     and the next cell would hit the same wall.
#
# Ordered cheapest-first so a battery cut short by a setup failure spends the least device
# time discovering it — and `tf` reuses `p4`'s arm A, so `p4` runs before it.
case $CELL in
profile | p4 | tf) CELLS=$CELL ;;
all) CELLS="profile p4 tf" ;;
*) echo "FAIL: unknown cell '$CELL' (profile|p4|tf|all)" >&2; exit 2 ;;
esac
for c in $CELLS; do
    rc=0
    ("cell_$c") || rc=$?
    case $rc in
    0 | 1) ;;
    4) exit 1 ;;
    *) exit "$rc" ;;
    esac
done

count() { find "$SCRATCH" -maxdepth 1 -name "$1" 2>/dev/null | wc -l; }
REPLAYED=""
[ -f "$SCRATCH/replayed" ] && REPLAYED=" (REPLAY — classifier only, NOT evidence about the engine)"
reds=$(count 'red.*')
proofs=$(count 'proof.*')
passes=0
[ -f "$SCRATCH/passes" ] && passes=$(wc -l <"$SCRATCH/passes")

if [ -n "$EXPECT_RED" ]; then
    [ "$proofs" -gt 0 ] || {
        echo "RED-PROOF FAILED: every cell came out green under --expect-red — the gate cannot see the planted defect. Debug the tree, not the proof." >&2
        exit 1
    }
    echo "RED-PROOF OK: $proofs cell(s) went red as demanded${REPLAYED:-} | evidence: $SCRATCH"
    exit 0
fi
# The suffix rides the FINAL line because that is the line a capture keeps; a mid-run
# caveat is invisible to `tail -1`.
if [ "$reds" -gt 0 ]; then
    echo "PPL-GATES RED: $reds cell(s) red, $passes check(s) green${REPLAYED:-} | evidence: $SCRATCH" >&2
    exit 1
fi
echo "PPL-GATES GREEN: $passes check(s)${REPLAYED:-} | evidence: $SCRATCH"
