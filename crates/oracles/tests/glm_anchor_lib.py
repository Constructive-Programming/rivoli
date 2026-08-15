"""Container, scoring and fp32 floors for the GLM-5.2 anchor goldens.

Split out of `glm_anchor_driver.py` on 2026-08-15 under the 800-line-per-file cap. The cut
is by dependency, not by size: nothing in here loads the reference stack or knows what a
defect DOES. It reads and writes the byte container, scores a pair of goldens against a
defect matrix handed in by the caller, and measures the fp32 rounding floor per operator
bucket.

`score_goldens` and `measure_floors` take the defect matrix and the run function as
ARGUMENTS rather than importing them. The driver owns both — the matrix is what
`glm-anchor.sh` derives its cell list from — and importing back would be a cycle.

Nothing here may change what `write_golden` emits: `glm-anchor.sh` regenerates the goldens
and `cmp`s them against the vendored bytes, so a formatting change in this file is a gate
failure there.
"""

import math
import pathlib
import struct
import sys

import torch

MAGIC = b"RIVGMGLD"  # GLM golden; RIVGLGLD is Muse Glimmer's, RIVK3GLD Kimi-K3's.


# ---------------------------------------------------------------------------------------------
# Capture container (the glimmer driver's discipline restated).


class Capture:
    """Named tensors on their way past, in emission order; floats and ints kept apart."""

    def __init__(self):
        self.floats = []
        self.ints = []
        self.seen = set()

    def add(self, name, tensor):
        if name in self.seen:
            raise AssertionError(f"capture {name!r} written twice; names must be unique")
        self.seen.add(name)
        t = tensor.detach().to(torch.float32).contiguous()
        self.floats.append((name, list(t.shape), t.flatten().tolist()))

    def add_ints(self, name, values):
        if name in self.seen:
            raise AssertionError(f"capture {name!r} written twice; names must be unique")
        self.seen.add(name)
        vals = [int(v) for v in values]
        self.ints.append((name, [len(vals)], vals))


# ---------------------------------------------------------------------------------------------
# Container (byte-compatible with the glimmer container, distinct magic).


def _u64(n):
    return struct.pack("<Q", n)


def _s(x):
    b = x.encode()
    return _u64(len(b)) + b


def write_golden(path, meta, cap):
    out = bytearray(MAGIC)
    out += _u64(len(meta))
    for k, v in meta:
        out += _s(k) + _s(v)
    for items, fmt in ((cap.floats, "<f"), (cap.ints, "<q")):
        out += _u64(len(items))
        for name, shape, vals in items:
            out += _s(name) + _u64(len(shape))
            for d in shape:
                out += _u64(d)
            out += _u64(len(vals))
            for x in vals:
                out += struct.pack(fmt, x)
    pathlib.Path(path).write_bytes(bytes(out))
    return len(out)


def read_golden(path):
    buf = pathlib.Path(path).read_bytes()
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: not a GLM golden (magic {buf[:8]!r})")
    off = 8

    def u64():
        nonlocal off
        v = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        return v

    def s():
        nonlocal off
        n = u64()
        v = buf[off : off + n].decode()
        off += n
        return v

    meta = {}
    for _ in range(u64()):
        k = s()
        meta[k] = s()
    tensors = {}
    for fmt, size in (("<f", 4), ("<q", 8)):
        for _ in range(u64()):
            name = s()
            shape = [u64() for _ in range(u64())]
            n = u64()
            vals = list(struct.unpack_from(f"<{n}{fmt[1]}", buf, off))
            off += n * size
            tensors[name] = (shape, vals)
    return meta, tensors


def environment():
    import transformers

    return [
        ("python", sys.version.split()[0]),
        ("torch", torch.__version__),
        ("transformers", transformers.__version__),
    ]


def preflight_env(vendored):
    """Refuse to run against a drifted environment, reading the pin OUT OF the vendored
    golden rather than restating it here."""
    meta, _ = read_golden(vendored)
    drift = [
        f"{k}: venv {v} != vendored {meta[k]}"
        for k, v in environment()
        if k in meta and meta[k] != v
    ]
    if drift:
        raise SystemExit(
            "environment drift against the vendored golden:\n  " + "\n  ".join(drift)
        )


# ---------------------------------------------------------------------------------------------
# Scoring a defect run against its clean twin.


def _assert_clean_reference(a_path, meta_a):
    if meta_a.get("defect", "None") != "None":
        raise SystemExit(f"{a_path} is itself a defect run; A must be None")


def _assert_tensor_sets(ta, tb, defect, extra_ok):
    """A defect may only ADD captures, and only when it declares `extra_ok` — a SHRINKING
    tensor set means the defect deleted evidence instead of perturbing it."""
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    undeclared = only_b and not extra_ok
    if only_a or undeclared:
        raise SystemExit(
            f"tensor sets differ: only in A {only_a[:5]}, only in B {only_b[:5]}"
            + ("" if extra_ok else "\n(the defect does not declare extra_ok)")
        )
    if extra_ok and not only_b:
        raise SystemExit(f"defect {defect!r} declares extra_ok but produced no extra captures")


def _divergence(a_path, name, va, vb):
    """The worst element-wise gap, or +inf if the defect run went non-finite.

    NaN is not order-comparable, so max() DROPS it after the first element — the repo's own
    false-green class (`f32::max ignores NaN`). Any non-finite on either side is a
    divergence by definition; the clean side must be finite for the whole comparison to
    mean anything."""
    if any(not math.isfinite(x) for x in va):
        raise SystemExit(f"{a_path}: {name} carries a non-finite value in the CLEAN run")
    if any(not math.isfinite(y) for y in vb):
        return float("inf")
    return max((abs(x - y) for x, y in zip(va, vb)), default=0.0)


def _score_one(a_path, name, entry_a, entry_b):
    """One capture: did it move, and the phrase describing how (used only if it was
    declared green)."""
    shape_a, va = entry_a
    shape_b, vb = entry_b
    # Shape first: zip() silently truncates, so a length change under a defect would be
    # scored only over the overlap and a divergent tail would read as held.
    if shape_a != shape_b or len(va) != len(vb):
        return True, f"SHAPE moved {shape_a}->{shape_b}"
    d = _divergence(a_path, name, va, vb)
    return d > 0, f"moved by {d:.3e}"


def _score_shared(a_path, ta, tb, is_green):
    """Split every capture the two runs share into moved/held, collecting the ones that
    moved while declared green."""
    moved, held, violations = [], [], []
    for name in sorted(set(ta) & set(tb)):
        did_move, why = _score_one(a_path, name, ta[name], tb[name])
        (moved if did_move else held).append(name)
        if did_move and is_green(name):
            violations.append(f"{name}: declared green, {why}")
    return moved, held, violations


def _assert_it_reddened(defect, moved):
    if not moved:
        raise SystemExit(f"defect {defect!r} moved NOTHING — it is not a defect, or it did not apply")


def _assert_green_held(defect, green, green_held, violations):
    if green and not green_held:
        raise SystemExit(f"defect {defect!r} declares green captures but none of them held")
    if violations:
        raise SystemExit(
            f"{len(violations)} declared-green captures moved:\n  " + "\n  ".join(violations)
        )


def _verdict(defect, green, scored, is_green):
    """Both halves of the contract: something moved AND everything declared green held."""
    moved, held, violations = scored
    print(f"defect {defect!r}: {len(moved)} captures moved, {len(held)} held")
    _assert_it_reddened(defect, moved)
    green_held = [n for n in held if is_green(n)]
    _assert_green_held(defect, green, green_held, violations)
    print(f"  {len(green_held)} declared-green captures held" if green else "  (no green set)")


def score_goldens(a_path, b_path, defects):
    """Score two goldens and ASSERT the perturbed one's declared green set — both halves:
    something moved AND everything declared green held bit-identical."""
    meta_a, ta = read_golden(a_path)
    meta_b, tb = read_golden(b_path)
    defect = meta_b.get("defect", "None")
    _assert_clean_reference(a_path, meta_a)
    _fn, green, extra_ok = defects[defect]
    _assert_tensor_sets(ta, tb, defect, extra_ok)

    def is_green(name):
        return any(g in name for g in green)

    _verdict(defect, green, _score_shared(a_path, ta, tb, is_green), is_green)


# ---------------------------------------------------------------------------------------------
# The fp32 rounding floor, per operator bucket.


def bucket_of(name):
    """The operator bucket a capture belongs to — the granularity tolerances are keyed at."""
    for tag in ("router.", "experts.out", "shared.out", "moe.out", "mlp.out", "attend.",
                "q_resid", "kv_latent", "norm.", "logits", "attn.out",
                ".index.q.post_rope", ".index.k.post_rope", ".attn.q.post_rope", ".attn.k.post_rope"):
        if tag in name:
            return {".index.q.post_rope": "rope_index", ".index.k.post_rope": "rope_index",
                    ".attn.q.post_rope": "rope_attn", ".attn.k.post_rope": "rope_attn",
                    "router.": "router", "experts.out": "moe", "shared.out": "moe",
                    "moe.out": "moe", "mlp.out": "dense_mlp", "attend.": "attend",
                    "q_resid": "lora_norm", "kv_latent": "lora_norm", "norm.": "norm",
                    "logits": "logits", "attn.out": "attn_out"}[tag]
    return "other"


def _mask_gap(name, x, y):
    """The mask's sentinel is torch.finfo(dtype).min — dtype-dependent BY CONSTRUCTION
    (-3.4e38 at f32; -inf once the f64 sentinel is cast to f32 for capture). Selection is
    discrete, so what must agree across dtypes is the masked/kept CLASS of each position,
    never the sentinel: a floor for a mask would be a category error."""
    masked_x, masked_y = x < -1e30, y < -1e30
    if masked_x != masked_y:
        raise SystemExit(f"{name}: the f64 run SELECTED differently ({x} vs {y})")
    return 0.0 if masked_x else abs(x - y)


def _value_gap(name, x, y):
    if not (math.isfinite(x) and math.isfinite(y)):
        raise SystemExit(f"{name}: non-finite value in the floor comparison ({x} vs {y})")
    return abs(x - y)


def _floor_of(name, vals, ref):
    """The worst per-element gap between the f64 run and the vendored fp32 golden."""
    if name not in ref:
        raise SystemExit(f"{name}: captured at f64 but absent from the golden")
    rv = ref[name][1]
    if len(rv) != len(vals):
        raise SystemExit(f"{name}: length changed between dtypes")
    gap = _mask_gap if "mask" in name else _value_gap
    return max((gap(name, x, y) for x, y in zip(vals, rv)), default=0.0)


def _assert_floor_reference(meta, salt):
    if meta.get("defect", "None") != "None":
        raise SystemExit("floors must be measured against a CLEAN golden")
    if meta.get("salt") != salt:
        raise SystemExit(f"golden is salt {meta.get('salt')!r}, asked for {salt!r}")


def measure_floors(vendored_path, salt, run_f64):
    """The fp32 rounding floor per operator bucket: the same reference at float64 against
    the vendored fp32 golden. A tolerance for a kernel is chosen ABOVE its operator's
    floor and BELOW its weakest targeting defect; a floor from one draw is half a
    measurement (glimmer's attend floor differed 2.1x between draws), so run this at both
    salts and take the worse."""
    meta, ref = read_golden(vendored_path)
    _assert_floor_reference(meta, salt)
    cap = Capture()
    run_f64(salt, "None", cap, dtype=torch.float64)
    by_bucket = {}
    for name, _shape, vals in cap.floats:
        b = bucket_of(name)
        by_bucket[b] = max(by_bucket.get(b, 0.0), _floor_of(name, vals, ref))
    compared = len(cap.floats)
    if compared < 500:
        raise SystemExit(f"only {compared} captures compared — the f64 run is incomplete")
    print(f"fp32 floors at salt {salt} over {compared} captures:")
    for b in sorted(by_bucket):
        print(f"  {b:12s} {by_bucket[b]:.3e}")
    return by_bucket
