#!/usr/bin/env bash
# Where two teacher-forced runs first stopped agreeing, and what that implies for a gate.
#
#   tests/nll-divergence.sh <a.nll> <b.nll> [more.nll ...]   # pairwise first-divergence
#   tests/nll-divergence.sh --power <events> <exposure>       # rate + interval + gate power
#   tests/nll-divergence.sh --se <a.nll> <b.nll>              # is bin/ppl's SE trustworthy here?
#
# ## Why the FIRST DIVERGING POSITION and never the differing COUNT
#
# Past the first divergence the two runs are computing over different state, so every later
# position differs whether or not anything else went wrong: a count of 526 of 762 is ONE
# event plus its wake, not 526 events. A count is therefore not a rate, not monotone in
# severity, and not comparable between runs — it is a function of WHERE the first event
# landed and nothing else. Ranking on it has already misled this investigation once
# (`docs/investigations/glm-nondeterminism.md`).
#
# The first position IS a rate: under teacher forcing every position is re-anchored to the
# committed corpus, so the walk is a sequence of near-identical trials and "position of the
# first event" is a geometric/exponential waiting time. That is the only quantity here that
# supports arithmetic.
#
# ## What the input is
#
# `.nll` files from `--features teacher-forcing --ppl`: a `#` header, then one f32 per
# predicted position as printed text. Compared AS TEXT, so "identical" means identical as
# recorded — no re-parse, no tolerance. A tolerance would be the wrong instrument twice over:
# the question is bit-reproducibility, and a per-position tolerance would hide exactly the
# small early perturbation whose position is the measurement.
#
# Deviceless. No GPU, no artifact, no flock. `docs/measurement/glm-divergence-evidence/`
# holds four vendored runs so every number this prints stays re-derivable.
set -uo pipefail

if [ "${1:-}" = "--se" ]; then
    shift
    [ $# -eq 2 ] || { echo "usage: nll-divergence.sh --se <a.nll> <b.nll>" >&2; exit 2; }
    # DOES `bin/ppl`'s `SE = sd/sqrt(n)` HOLD ON THIS DEFECT? It assumes the per-position dNLLs
    # are independent, and this defect makes one event contaminate every later position, so the
    # assumption is exactly the one it violates. `wave/m10-spine` raised that and could not
    # settle it because the `.nll` files had not been kept; they are vendored now, so this
    # settles it.
    #
    # Two deterministic diagnostics, no RNG — a bootstrap would need `rand()`, whose sequence is
    # implementation-dependent, and a re-derivation command that gives a different answer under
    # a different awk is not a re-derivation:
    #   lag-1 rho          positive rho would mean the naive SE UNDERSTATES; the inflation
    #                      factor for an AR(1) series is sqrt((1+rho)/(1-rho)).
    #   block SE (L=50)    the standard error of 15 non-overlapping block means, which makes no
    #                      independence assumption WITHIN a block and so absorbs any correlation
    #                      shorter than 50 positions.
    a=$1; b=$2
    for f in "$a" "$b"; do [ -r "$f" ] || { echo "FAIL: cannot read $f" >&2; exit 2; }; done
    paste <(grep -vE '^[[:space:]]*(#|$)' "$a") <(grep -vE '^[[:space:]]*(#|$)' "$b") | awk -v L=50 '
        { d[NR] = $2 - $1; n = NR }
        END {
            if (n < 2 * L) { print "FAIL: need at least 2*L positions"; exit 2 }
            for (i = 1; i <= n; i++) s += d[i]
            m = s / n
            for (i = 1; i <= n; i++) { v = d[i] - m; ss += v * v }
            sd = sqrt(ss / (n - 1)); se = sd / sqrt(n)
            for (i = 1; i < n; i++) num += (d[i] - m) * (d[i + 1] - m)
            rho = num / ss
            nb = int(n / L)
            for (k = 0; k < nb; k++) { bs = 0; for (i = 1; i <= L; i++) bs += d[k * L + i]; bm[k] = bs / L; bsum += bm[k] }
            bmean = bsum / nb
            for (k = 0; k < nb; k++) { v = bm[k] - bmean; bss += v * v }
            bse = sqrt(bss / (nb - 1)) / sqrt(nb)
            printf "n=%d  mean dNLL %+.6f  sd %.4f  mean |dNLL| ", n, m, sd
            for (i = 1; i <= n; i++) ad += (d[i] < 0 ? -d[i] : d[i])
            printf "%.6f\n", ad / n
            printf "  naive      SE %.5f   95%% CI [%+.5f, %+.5f]\n", se, m - 1.96 * se, m + 1.96 * se
            printf "  lag-1 rho  %+.4f  -> AR(1) inflation sqrt((1+rho)/(1-rho)) = %.3fx\n", rho, sqrt((1 + rho) / (1 - rho))
            printf "  block(L=%d) SE %.5f   95%% CI [%+.5f, %+.5f]   = %.2fx naive (%d blocks)\n", L, bse, m - 1.96 * bse, m + 1.96 * bse, bse / se, nb
            print ""
            if (rho > 0.05) print "rho > 0: the naive SE UNDERSTATES. Use the block SE."
            else print "rho <= 0: the naive SE does NOT understate here -- it is CONSERVATIVE, and the"
            if (rho <= 0.05) print "feared inflation does not occur. Note the block CI is TIGHTER, so a pair whose naive"
            if (rho <= 0.05) print "CI contains zero may still have a mean that is significantly non-zero."
        }'
    exit 0
fi

if [ "${1:-}" = "--power" ]; then
    shift
    [ $# -eq 2 ] || { cat >&2 <<'EOF'
usage: nll-divergence.sh --power <events> <exposure>

  events    number of FIRST-divergence events observed
  exposure  total position-forwards those runs survived, INCLUDING any run that
            finished with no event (its full length)

EXPOSURE IS EXPLICIT ON PURPOSE. An earlier version took a list of event positions and
summed them, which silently dropped every clean run's exposure and so over-stated the
rate — a censored arm is exactly the term a waiting-time estimate must not lose (review,
2026-08-17). Making the caller state both numbers means the reading is visible in the
command line that produced it.
EOF
        exit 2; }
    K=$1; E=$2
    awk -v k="$K" -v e="$E" 'BEGIN {
        if (k < 1 || e < 1) { print "FAIL: events and exposure must be >= 1" > "/dev/stderr"; exit 2 }
        rate = k / e
        # Garwood exact Poisson bounds, chi2 quantiles tabulated for the small k this evidence
        # supports. Beyond that the point estimate is all this prints, and says so.
        lo = 0; hi = 0
        if (k == 2) { lo = 0.4844 / 2; hi = 14.4494 / 2 }
        if (k == 3) { lo = 1.2373 / 2; hi = 16.8119 / 2 }
        printf "events=%d  exposure=%d position-forwards  rate = 1 per %.0f\n", k, e, 1 / rate
        if (hi > 0) printf "  exact 95%% Poisson interval on the rate: 1 per [%.0f, %.0f]\n", e / hi, e / lo
        else print "  (no interval printed: chi2 bounds are tabulated for k=2,3 only)"
        print ""
        printf "%-8s %-12s %s\n", "ngen", "P(detect)", "at the interval ends"
        n = split("32 64 128 256 512 1024", g, " ")
        for (i = 1; i <= n; i++) {
            N = g[i] + 0
            # The %% hugs its number: `%-11.0f%%` pads FIRST and prints a lonely % at column 20.
            printf "%-8d %-12s", N, sprintf("%.0f%%", 100 * (1 - exp(-2 * rate * N)))
            if (hi > 0)
                printf "  %.0f%% .. %.0f%%", 100 * (1 - exp(-2 * (hi / e) * N)), 100 * (1 - exp(-2 * (lo / e) * N))
            printf "\n"
        }
        printf "\nngen for 80%% power at the point estimate: %.0f\n", -log(0.2) / (2 * rate)
        print ""
        print "P(detect) is for TWO arms of ngen tokens each -- the gates POWER against THIS rate."
        print "The interval column is why a green is a bound and not a proof: at the pessimistic end"
        print "even a long run detects little. Compare against real pair outcomes yourself; this tool"
        print "deliberately prints no observations of its own."
    }' </dev/null
    exit 0
fi

[ $# -ge 2 ] || { echo "usage: nll-divergence.sh <a.nll> <b.nll> [more.nll ...] | --power <pos>..." >&2; exit 2; }
for f in "$@"; do [ -r "$f" ] || { echo "FAIL: cannot read $f" >&2; exit 2; }; done

# `${TMPDIR:-/tmp}` and a trap, like every sibling script: /tmp here is a 63 GiB RAM tmpfs that
# has hit 100%, and an early exit used to leak two files per pair.
WORK=$(mktemp -d "${TMPDIR:-/tmp}/nll-divergence.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# strip() keeps the value TEXT; see the header for why the comparison is textual.
strip() { grep -vE '^[[:space:]]*(#|$)' "$1"; }

# F4 (review 2026-08-17): without these, `/dev/null` vs `/dev/null` printed
# "identical -- no event" and exited 0, and `/dev/null` vs `a.nll` invented an event at
# position 0 (paste pads the missing column and awk's default FS reads the survivor as $1).
# A silent "no event" from a truncated or mis-parsed pair is a false green in the one tool
# the rate is read off.
declare -a N=()
for f in "$@"; do
    c=$(strip "$f" | wc -l)
    [ "$c" -gt 0 ] || { echo "FAIL: $f has no value lines (header only, or not a .nll)" >&2; exit 2; }
    N+=("$c")
done
for c in "${N[@]}"; do
    [ "$c" = "${N[0]}" ] || { echo "FAIL: files differ in length (${N[*]}) — a position-wise comparison needs one grid" >&2; exit 2; }
done
echo "== pairwise first divergence over ${N[0]} positions"
printf '%-14s %-14s %6s %8s %12s\n' arm-a arm-b ndiff first "|d| at first"
n=$#
for ((i = 1; i <= n; i++)); do
    for ((j = i + 1; j <= n; j++)); do
        a=${!i}; b=${!j}
        strip "$a" >"$WORK/a"
        strip "$b" >"$WORK/b"
        # paste + awk in one pass: the count, the first differing 0-based position, and the
        # magnitude AT that position — the last is what says whether the event is a rounding
        # difference (1e-7) or a wrong-bytes read (1e-2).
        read -r nd first mag < <(paste "$WORK/a" "$WORK/b" | awk '
            $1 != $2 { nd++; if (first == "") { first = NR - 1; mag = ($1 - $2 < 0 ? $2 - $1 : $1 - $2) } }
            END { printf "%d %s %s\n", nd + 0, (first == "" ? "-" : first), (first == "" ? "-" : mag) }')
        printf '%-14s %-14s %6s %8s %12s\n' "$(basename "$a")" "$(basename "$b")" "$nd" "$first" "$mag"
    done
done

# The wake profile of the FIRST pair only: one is enough to answer the one structural
# question a profile can answer, and printing it per pair buries the table above.
a=$1; b=$2
strip "$a" >"$WORK/a"
strip "$b" >"$WORK/b"
echo
echo "== |dNLL| after the event, $(basename "$a") vs $(basename "$b") (60-position windows)"
echo "   GROWING means the two runs' KV caches keep drifting apart — one event, permanent"
echo "   damage. DECAYING would mean the perturbation washes out. FLAT-FROM-ZERO with"
echo "   isolated nonzero windows would mean many independent events instead of one."
paste "$WORK/a" "$WORK/b" | awk '
    { d[NR - 1] = ($1 - $2 < 0 ? $2 - $1 : $1 - $2); if (first == "" && $1 != $2) first = NR - 1; n = NR }
    END {
        if (first == "") { print "   (identical — no event)"; exit }
        for (lo = first; lo < n; lo += 60) {
            hi = lo + 59; if (hi >= n) hi = n - 1
            c = 0; nz = 0; delete w
            for (p = lo; p <= hi; p++) { w[c++] = d[p]; if (d[p] > 0) nz++ }
            # median by insertion sort; windows are 60 wide, so this is not worth a better sort
            for (x = 1; x < c; x++) { v = w[x]; y = x - 1; while (y >= 0 && w[y] > v) { w[y + 1] = w[y]; y-- } w[y + 1] = v }
            printf "   pos %4d-%4d  median %.3e  nonzero %d/%d\n", lo, hi, w[int(c / 2)], nz, c
        }
    }'
