"""Per-operator floors and signals: HOW MUCH a correct kernel may drift, not WHERE a defect landed.

Split out of `glimmer_anchor_driver.py` 2026-08-15 under the 800-line cap, verbatim. The section
comment below argued these were a separate question from `compare`'s before there was a second file;
the split makes the file boundary agree with the argument that was already written down. Builds no
model and imports no reference stack -- it reads the container and nothing else.
"""

import math

from glimmer_anchor_lib import read_golden

# ---------------------------------------------------------------------------------------------
# Per-operator scoring. A tolerance is a property of an OPERATOR, and `compare`
# (`glimmer_anchor_compare.py`) cannot state one: it groups by capture name to ask *where* a defect lives, which is a different question from
# *how much* arithmetic disagreement a correct kernel is allowed.


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
    parts = name.split(".")
    rest = ".".join(parts[2:]) if len(parts) > 2 and parts[1].startswith("L") else ".".join(parts[1:])
    if name in ("prompt.ids", "emitted.ids", "layer_is_roped", "layer_is_sliding"):
        return "ids"
    if rest.startswith("attend."):
        return "attend"
    if rest.startswith("qk_norm."):
        return "qk_norm"
    if rest.endswith(".roped"):
        return "rope"
    if rest.startswith("rope."):
        return "rope_table"
    if rest.endswith(".pre_rope"):
        return "proj"
    if rest.startswith("attn.gate_proj"):
        return "gate"
    if rest.startswith("attn.o_proj"):
        return "o_proj"
    if rest.startswith("mlp."):
        return "mlp"
    if rest == "logits":
        return "logits"
    if rest.startswith("embed_norm"):
        return "embed_norm"
    if rest.startswith("final_norm"):
        return "final_norm"
    if "layernorm" in rest:
        return "norm"
    raise SystemExit(f"no operator bucket for capture {name!r} -- classify it before scoring")


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
    # **Skipped rather than refused, and counted in its own column.** Three defects change the
    # capture SET or a shape -- `window_off_by_one` and `full_layers_slide` resize the masks,
    # `rope_on_nope_layers` makes 56 captures exist that do not exist cleanly -- and those three
    # include the two that target the attend kernel hardest. An earlier version raised on any
    # mismatch, which silently dropped exactly the rows S2 item 1 needed: the sweep produced no
    # `attend` line for them at all, and a reader comparing defects would have concluded the
    # window defect leaves attention alone. A shape that changed is not a value that can be
    # subtracted; it is also not nothing, so it is reported.
    per, skipped = {}, {}
    for name, (shape, va) in ta.items():
        op = operator_of(name)
        if name not in tb or tb[name][0] != shape:
            skipped[op] = skipped.get(op, 0) + 1
            continue
        vb = tb[name][1]
        n, moved, rel = per.get(op, (0, 0, 0.0))
        # Scaled by the reference side's own magnitude, so this is a RELATIVE error and comparable
        # across operators of different widths. Non-finite pairs are skipped rather than folded in:
        # python's `max` silently discards a NaN unless it comes first, which would read as
        # agreement.
        scale = max((abs(v) for v in va if math.isfinite(v)), default=0.0) or 1e-30
        d = max(
            (abs(x - y) for x, y in zip(va, vb) if math.isfinite(x) and math.isfinite(y)),
            default=0.0,
        )
        per[op] = (n + 1, moved + (1 if d > 0 else 0), max(rel, d / scale))
    if not per:
        raise SystemExit(
            f"{a_path} and {b_path} share no comparable tensor at all -- these are not two runs "
            "of the same harness"
        )
    print(f"# {meta_b.get('defect')}/{meta_b.get('dtype')} vs {meta_a.get('defect')}/{meta_a.get('dtype')}")
    print("operator\ttensors\tdiffering\tmax_rel\tskipped")
    for op in sorted(set(per) | set(skipped)):
        n, moved, rel = per.get(op, (0, 0, 0.0))
        print(f"{op}\t{n}\t{moved}\t{rel:.3e}\t{skipped.get(op, 0)}")
    extra = sorted(set(tb) - set(ta))
    if extra:
        print(f"# {len(extra)} captures exist only in B, e.g. {extra[:3]}")
