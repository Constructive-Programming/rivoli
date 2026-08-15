"""Assert a defect's declared green set: WHERE the arithmetic moved, and that nothing else did.

Split out of `glimmer_anchor_driver.py` 2026-08-15 under the 800-line cap, verbatim.
`glimmer_anchor_operators.py` is the other half -- how MUCH a correct kernel may drift -- and the
two sit apart because they answer different questions, which the operator file's own header has
said since before either was a file.
"""

from glimmer_anchor_defects import DEFECTS
from glimmer_anchor_lib import read_golden

def compare(a_path, b_path):
    """Score two goldens against each other and ASSERT the perturbed one's declared green set.

    "Something changed" is the half of a defect proof that is easy and nearly worthless. The half
    that carries the evidence is that everything the defect does not touch stayed EXACTLY equal --
    that is what says the golden localises rather than merely reacts.
    """
    meta_a, ta = read_golden(a_path)
    meta_b, tb = read_golden(b_path)
    defect = meta_b.get("defect", "None")
    if meta_a.get("defect", "None") != "None":
        raise SystemExit(f"{a_path} is itself a defect run ({meta_a.get('defect')}); A must be None")
    _fn, green, extra_ok = DEFECTS[defect]
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    if only_a or (only_b and not extra_ok):
        raise SystemExit(
            f"tensor sets differ: only in A {only_a[:5]}, only in B {only_b[:5]}"
            + ("" if extra_ok else "\n(the defect does not declare extra_ok)")
        )
    if extra_ok:
        if not only_b:
            raise SystemExit(f"defect {defect!r} declares extra_ok but produced no extra captures")
        print(f"  {len(only_b)} captures exist only under the defect, e.g. {only_b[:3]}")

    def is_green(name):
        # A single negated entry inverts the rule; mixing the two forms is not supported, because a
        # set that is both "these held" and "everything but these held" says nothing.
        if len(green) == 1 and green[0].startswith("!"):
            return green[0][1:] not in name
        return any(g in name for g in green)

    moved, held, violations = [], [], []
    for name, (_shape, va) in sorted(ta.items()):
        if name not in tb:
            continue
        vb = tb[name][1]
        d = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        (moved if d > 0 else held).append(name)
        if is_green(name) and d > 0:
            violations.append(f"{name}: declared green, moved by {d:.3e}")
    print(f"defect {defect!r}: {len(moved)} captures moved, {len(held)} held")
    if not moved:
        raise SystemExit(f"defect {defect!r} moved NOTHING -- it is not a defect, or it did not apply")
    if green and not any(is_green(n) for n in held):
        raise SystemExit(f"defect {defect!r} declares green captures but none of them held")
    if violations:
        raise SystemExit(
            f"{len(violations)} declared-green captures moved:\n  " + "\n  ".join(violations)
        )
    print(f"  {sum(is_green(n) for n in held)} declared-green captures held" if green else "  (no green set)")
