"""Per-operator floors and signals: HOW MUCH a correct kernel may drift, not WHERE a defect landed.

Split out of `glimmer_anchor_driver.py` 2026-08-15 under the 800-line cap, verbatim. The section
comment below argued these were a separate question from `compare`'s before there was a second file;
the split makes the file boundary agree with the argument that was already written down. Builds no
model and imports no reference stack -- it reads the container and nothing else.

**Decomposed 2026-08-15 for the CodeScene 10.0 gate** (`operator_of` was cc 16, `by_operator` cc 12,
the file 9.23). Shape only: the classifier's if-chain became `_BUCKETS`, a table read TOP TO BOTTOM
in the chain's own order, and `by_operator`'s loop and printing became two helpers called in the old
order. No bucket name, no threshold and no comparison changed.
"""

import math

from glimmer_anchor_lib import read_golden

# ---------------------------------------------------------------------------------------------
# Per-operator scoring. A tolerance is a property of an OPERATOR, and `compare`
# (`glimmer_anchor_compare.py`) cannot state one: it groups by capture name to ask *where* a defect
# lives, which is a different question from *how much* arithmetic disagreement a correct kernel is
# allowed.

# `(test, needle, bucket)`, tried IN ORDER: this is the old if-chain transposed, first match wins,
# and the rows are in the chain's order because that is the order that was reviewed.
#
# **The order is not currently load-bearing, and that is a measurement, not an assumption.** Over
# all 1259 capture names in the six vendored goldens, exactly one row matches each name -- checked
# 2026-08-15 by counting matches per name rather than by reading the rules. So the table is a
# partition today. It is kept ordered anyway because the rules are not disjoint BY CONSTRUCTION
# (`rope.`/`.roped`/`.pre_rope` are three readings of one substring), and a capture added later
# that two rows match must resolve the way the `elif` cascade resolved it.
#
# The tests are the unbound `str` methods so that one flat sequence can mix prefix, suffix, exact
# and substring rules without splitting the chain into per-kind groups -- which is the split that
# would let the order drift silently.
_BUCKETS = (
    (str.startswith, "attend.", "attend"),
    (str.startswith, "qk_norm.", "qk_norm"),
    (str.endswith, ".roped", "rope"),
    (str.startswith, "rope.", "rope_table"),
    (str.endswith, ".pre_rope", "proj"),
    (str.startswith, "attn.gate_proj", "gate"),
    (str.startswith, "attn.o_proj", "o_proj"),
    (str.startswith, "mlp.", "mlp"),
    (str.__eq__, "logits", "logits"),
    (str.startswith, "embed_norm", "embed_norm"),
    (str.startswith, "final_norm", "final_norm"),
    (str.__contains__, "layernorm", "norm"),
)

# The four captures that are structure rather than arithmetic, matched on the WHOLE name because
# they carry no `tN.` step prefix to strip.
_ID_CAPTURES = ("prompt.ids", "emitted.ids", "layer_is_roped", "layer_is_sliding")


def _rest_of(name):
    """A capture name with its `tN.` step prefix, and its `LN.` layer prefix when it has one, gone.

    What is left is the part that names an OPERATOR, which is the only part `_BUCKETS` may see: a
    rule that could match `t0` or `L3` would bucket by position instead of by kernel.
    """
    parts = name.split(".")
    if len(parts) > 2 and parts[1].startswith("L"):
        return ".".join(parts[2:])
    return ".".join(parts[1:])


def operator_of(name):
    """The kernel a capture belongs to, keyed to `glimmer-port.md` §S2's five items.

    Buckets are the port's own decomposition, not the reference's module tree, because the number
    this feeds is a threshold a rivoli kernel will be scored against. `attend` is S2 item 1,
    `rope` item 2, `gate`/`o_proj` item 3, `mlp` item 4, `logits` item 5; `norm` and `qk_norm` are
    S3's, and `ids` is exact-or-not by construction.

    **`pre_rope` is deliberately NOT in `rope`.** It is the projection's output, so folding it in
    would price the projection's accumulated error as the rotation's — and the rotation is the
    cheaper operator of the two, so the floor would come out too generous in exactly the direction
    that hides a bug. Same reason `attend.q` and the two caches sit in `attend`: they are that
    kernel's inputs, and an input that already disagrees bounds what its output can prove.
    """
    rest = _rest_of(name)
    if name in _ID_CAPTURES:
        return "ids"
    for test, needle, bucket in _BUCKETS:
        if test(rest, needle):
            return bucket
    raise SystemExit(f"no operator bucket for capture {name!r} -- classify it before scoring")


def _relative_gap(va, vb):
    """The largest disagreement between two payloads of one capture, and its relative form.

    Scaled by the reference side's own magnitude, so the second number is a RELATIVE error and
    comparable across operators of different widths. Non-finite pairs are skipped rather than
    folded in: python's `max` silently discards a NaN unless it comes first, which would read as
    agreement.
    """
    scale = max((abs(v) for v in va if math.isfinite(v)), default=0.0) or 1e-30
    d = max(
        (abs(x - y) for x, y in zip(va, vb) if math.isfinite(x) and math.isfinite(y)),
        default=0.0,
    )
    return d, d / scale


def _tally(ta, tb):
    """Fold every capture of A into its operator bucket, against B's tensor of the same name.

    **Skipped rather than refused, and counted in its own column.** Three defects change the
    capture SET or a shape -- `window_off_by_one` and `full_layers_slide` resize the masks,
    `rope_on_nope_layers` makes 56 captures exist that do not exist cleanly -- and those three
    include the two that target the attend kernel hardest. An earlier version raised on any
    mismatch, which silently dropped exactly the rows S2 item 1 needed: the sweep produced no
    `attend` line for them at all, and a reader comparing defects would have concluded the
    window defect leaves attention alone. A shape that changed is not a value that can be
    subtracted; it is also not nothing, so it is reported.
    """
    per, skipped = {}, {}
    for name, (shape, va) in ta.items():
        op = operator_of(name)
        if name not in tb or tb[name][0] != shape:
            skipped[op] = skipped.get(op, 0) + 1
            continue
        vb = tb[name][1]
        n, moved, rel = per.get(op, (0, 0, 0.0))
        d, gap = _relative_gap(va, vb)
        per[op] = (n + 1, moved + (1 if d > 0 else 0), max(rel, gap))
    return per, skipped


def _print_table(meta_a, meta_b, per, skipped):
    """One row per operator, plus the skipped column, headed by which two runs produced it."""
    print(
        f"# {meta_b.get('defect')}/{meta_b.get('dtype')} vs "
        f"{meta_a.get('defect')}/{meta_a.get('dtype')}"
    )
    print("operator\ttensors\tdiffering\tmax_rel\tskipped")
    for op in sorted(set(per) | set(skipped)):
        n, moved, rel = per.get(op, (0, 0, 0.0))
        print(f"{op}\t{n}\t{moved}\t{rel:.3e}\t{skipped.get(op, 0)}")


def by_operator(a_path, b_path):
    """Per-operator agreement between two goldens: the shape a tolerance has to be stated in.

    Two uses, and a tolerance needs both numbers. `fp32 vs fp64` gives the **floor** no correct
    fp32 implementation can beat; `None vs <defect>` gives the **signal** a threshold has to stay
    under. `anchor.md`'s table is those two per operator, and the policy follows from their ratio.

    **The floor this reports is contaminated from below, and by a known amount.** `Capture.add`
    stores f32 because the container does (`golden.rs` reads f32), so an fp64 run is rounded on the
    way out and every comparison carries one f32 rounding, relative 2^-24 = 6.0e-8. A bucket whose
    floor lands within an order of magnitude of that is measuring the container, not the
    arithmetic, and must not be turned into a threshold -- widen the container first. Buckets well
    above it are unaffected: the fp32 run's own accumulated error dominates.
    """
    meta_a, ta = read_golden(a_path)
    meta_b, tb = read_golden(b_path)
    per, skipped = _tally(ta, tb)
    if not per:
        raise SystemExit(
            f"{a_path} and {b_path} share no comparable tensor at all -- these are not two runs "
            "of the same harness"
        )
    _print_table(meta_a, meta_b, per, skipped)
    extra = sorted(set(tb) - set(ta))
    if extra:
        print(f"# {len(extra)} captures exist only in B, e.g. {extra[:3]}")
