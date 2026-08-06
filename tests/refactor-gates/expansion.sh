#!/usr/bin/env bash
# G5 — the macro gate. Does a `macro_rules!` expand to the same code the hand-written form was?
#
# WHY THIS EXISTS RATHER THAN G3. G3 hashes the AMDGCN disassembly of `kernels/*.hip`, which is
# the right gate for a kernel edit and **structurally blind** to `src/backend/hip.rs`: the Rust
# launcher wrapper is not in those objects at all. Track 2 is "macro-generate the ABI wall", so
# taking G3's green as evidence would have been a gate that cannot fire — the exact defect this
# refactor's own review list puts first. Measured 2026-08-06: transposing two same-typed
# arguments in `launch_gemv_vq` left every G3 hash unchanged.
#
# WHAT IT COMPARES. `-Zunpretty=expanded` prints the crate after macro expansion, so the
# hand-written form and the generated form become the same kind of thing and can be diffed.
# Three normalisations, each for a reason:
#
#   1. **Doc comments and `#[allow]` are dropped.** The macro moves prose to the invocation
#      site and replaces 17 per-item `too_many_arguments` allows with one block-level allow.
#      Both are inert to codegen. Dropping them is what lets the rest be compared exactly.
#   2. **Statements are compared as a SORTED MULTISET, not in order.** A macro emits its items
#      where it is invoked, so every converted launcher legitimately relocates. Order carries no
#      meaning for an `extern` declaration or a `pub fn`, and demanding it would make the gate
#      fire on every conversion and therefore get ignored — a gate nobody can keep green is a
#      gate that gets disabled.
#   3. **Whitespace is collapsed.** The pretty-printer re-wraps at its own width.
#
# What survives all three is the thing that matters: for every statement in the crate, does one
# with identical text still exist? A macro that drops a cast, transposes two same-typed
# arguments, changes an error tag or emits a different type shows up as a -/+ pair.
#
# SEEN RED 2026-08-06, on the defect class a macro actually introduces: swapping `o_dim` and
# `i_dim` in `launch_gemv_vq`'s call — same types, compiles clean, silently wrong — produced a
# 2-statement diff. Do not trust a green from this script you have not first made go red.
#
#   tests/refactor-gates/expansion.sh capture  <file>     # before the change
#   tests/refactor-gates/expansion.sh check    <file>     # after; exit 1 on any difference
set -uo pipefail
cd "$(dirname "$0")/../.."

mode=${1:?usage: expansion.sh capture|check <snapshot-file>}
snap=${2:?usage: expansion.sh capture|check <snapshot-file>}

raw=$(mktemp); trap 'rm -f "$raw"' EXIT
# RUSTC_BOOTSTRAP: -Zunpretty is nightly-only and this is an instrument, not shipped code. It
# reads the same source the real build does; nothing here is compiled into a binary.
if ! RUSTC_BOOTSTRAP=1 cargo rustc --lib --features rocm --profile dev \
        -- -Zunpretty=expanded 2>/dev/null > "$raw"; then
  echo "expansion.sh: cargo rustc failed — the crate does not compile" >&2
  exit 2
fi

# Anti-vacuity. Both of capture.sh's historical bugs produced EMPTY output that compared equal
# to itself, so an emptiness check is not paranoia here, it is the specific thing that bit.
n=$(wc -l < "$raw")
if [ "$n" -lt 10000 ]; then
  echo "expansion.sh: only $n lines expanded (expected >10000) — refusing to compare nothing" >&2
  exit 2
fi

norm=$(mktemp); trap 'rm -f "$raw" "$norm"' EXIT
python3 - "$raw" > "$norm" <<'PY'
import re, sys

# Extract the ABI surface ITEM BY ITEM, by scanning to each item's own terminator.
#
# The first version of this split the expansion on ';' and sorted the pieces. That was wrong in
# a way worth recording: a ';'-delimited piece runs from one statement's end to the next one's,
# so it carries the HEAD of the following item. Relocating an item therefore changed the text of
# its neighbours, and the sort could not undo it — five spurious differences on a conversion
# that was in fact byte-identical. Sorting only cancels order if the units are self-contained.
txt = open(sys.argv[1]).read()

def scan(src, start_re, opener, closer):
    """Every item matching start_re, from its keyword to its own balanced terminator."""
    out = []
    for m in re.finditer(start_re, src):
        i, depth = m.start(), 0
        while i < len(src):
            c = src[i]
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
            elif c == ';' and depth == 0 and opener == '(':
                i += 1
                break
            if depth == 0 and c == closer and opener == '{':
                i += 1
                break
            i += 1
        out.append(' '.join(src[m.start():i].split()))
    return out

# `extern` declarations terminate at the ';' outside their parameter list; wrappers at the '}'
# closing their body. Both are compared as SETS: a macro emits items where it is invoked, so
# every converted launcher legitimately moves, and order carries no meaning for either.
externs  = scan(txt, r'\bfn rivoli_\w+\(', '(', ')')
wrappers = scan(txt, r'\bpub unsafe fn launch_\w+\(', '{', '}')

# ANTI-VACUITY, and not theoretical: this gate's predecessor recorded the sha256 of the empty
# string ten times and read green. Two empty sets compare equal, so the counts are asserted
# before the contents are printed.
if len(externs) < 45 or len(wrappers) < 40:
    sys.exit(f"expansion.sh: extracted only {len(externs)} externs / {len(wrappers)} wrappers "
             f"(expect >=45 / >=40) -- the scanner is broken, not the code")

print(f"# {len(externs)} extern declarations, {len(wrappers)} launcher wrappers")
for s in sorted(externs) + sorted(wrappers):
    print(s)
PY

case "$mode" in
  capture)
    mv "$norm" "$snap"; trap 'rm -f "$raw"' EXIT
    echo "captured $(wc -l < "$snap") statements -> $snap"
    ;;
  check)
    [ -f "$snap" ] || { echo "expansion.sh: no snapshot at $snap — run capture first" >&2; exit 2; }
    if d=$(diff "$snap" "$norm"); then
      echo "G5 HELD — $(wc -l < "$norm") statements, expansion identical"
    else
      echo "G5 FAILED — the macro does not expand to what was there before:"
      echo "$d" | head -40
      exit 1
    fi
    ;;
  *) echo "usage: expansion.sh capture|check <snapshot-file>" >&2; exit 2 ;;
esac
