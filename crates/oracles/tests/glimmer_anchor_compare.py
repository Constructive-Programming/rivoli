"""Assert a defect's declared green set: WHERE the arithmetic moved, and that nothing else did.

Split out of `glimmer_anchor_driver.py` 2026-08-15 under the 800-line cap, verbatim.
`glimmer_anchor_operators.py` is the other half -- how MUCH a correct kernel may drift -- and the
two sit apart because they answer different questions, which the operator file's own header has
said since before either was a file.

**Decomposed 2026-08-15 for the CodeScene 10.0 gate** -- `compare` scored 9.14 as one function
(cc 26, two bumps, a nested `or`/`and` conditional). Each helper below is ONE CONTIGUOUS RUN of the
old body, called from `compare` in the old order, so the sequence of prints and refusals a caller
sees is byte-identical. Nothing was renamed, no literal moved, and no comparison changed direction:
the venv that regenerates the goldens is gone, so a behaviour change here could not have been
caught by re-running anything -- which is exactly why none was made.
"""

from glimmer_anchor_defects import DEFECTS
from glimmer_anchor_lib import read_golden


def _assert_same_tensor_set(ta, tb, defect, extra_ok):
    """The two runs must have captured the same NAMES, unless the defect declares they will not.

    `extra_ok` both permits the asymmetry and requires it: a defect that declares it will produce
    captures the clean run lacks, and then produces none, has stopped applying.
    """
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    # Named rather than nested. The predicate is the old `only_b and not extra_ok` verbatim; both
    # operands are pure locals, so hoisting it reads the same values in the same run.
    undeclared_extras = only_b and not extra_ok
    if only_a or undeclared_extras:
        raise SystemExit(
            f"tensor sets differ: only in A {only_a[:5]}, only in B {only_b[:5]}"
            + ("" if extra_ok else "\n(the defect does not declare extra_ok)")
        )
    if extra_ok:
        if not only_b:
            raise SystemExit(f"defect {defect!r} declares extra_ok but produced no extra captures")
        print(f"  {len(only_b)} captures exist only under the defect, e.g. {only_b[:3]}")


def _green_rule(green):
    """The green set compiled to a predicate over capture names.

    A single negated entry inverts the rule; mixing the two forms is not supported, because a
    set that is both "these held" and "everything but these held" says nothing.

    Deciding which form is in play ONCE rather than per call is the only difference from the
    closure this replaces: `green` is a tuple in `DEFECTS` and nothing mutates it mid-comparison.
    """
    if len(green) == 1 and green[0].startswith("!"):
        excluded = green[0][1:]
        return lambda name: excluded not in name
    return lambda name: any(g in name for g in green)


def _score(ta, tb, is_green):
    """Split the shared captures into moved and held, and collect the green ones that moved."""
    moved, held, violations = [], [], []
    for name, (_shape, va) in sorted(ta.items()):
        if name not in tb:
            continue
        vb = tb[name][1]
        d = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        (moved if d > 0 else held).append(name)
        if is_green(name) and d > 0:
            violations.append(f"{name}: declared green, moved by {d:.3e}")
    return moved, held, violations


def _held_green(held, is_green):
    """The declared-green captures that survived. Its COUNT is the final line `compare` prints,
    and its EMPTINESS is the "declares green captures but none of them held" refusal -- the same
    two readings the old body took from one `any`/`sum` pair over `held`."""
    return [n for n in held if is_green(n)]


def _report(defect, green, is_green, scored):
    """Print the tally, then refuse on each of the three ways a defect proof can be worthless.

    A defect that moved nothing is not a defect; a green set none of whose members survived proves
    no localisation; and a green capture that moved is the failure the whole file exists to catch.
    """
    moved, held, violations = scored
    print(f"defect {defect!r}: {len(moved)} captures moved, {len(held)} held")
    if not moved:
        raise SystemExit(f"defect {defect!r} moved NOTHING -- it is not a defect, or it did not apply")
    green_held = _held_green(held, is_green)
    if green and not green_held:
        raise SystemExit(f"defect {defect!r} declares green captures but none of them held")
    if violations:
        raise SystemExit(
            f"{len(violations)} declared-green captures moved:\n  " + "\n  ".join(violations)
        )
    print(f"  {len(green_held)} declared-green captures held" if green else "  (no green set)")


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
    _assert_same_tensor_set(ta, tb, defect, extra_ok)
    is_green = _green_rule(green)
    _report(defect, green, is_green, _score(ta, tb, is_green))
