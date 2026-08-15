"""The deviceless half of the S1b K3 anchor: the tiny config, the container, the scoring.

Split out of `k3_anchor_driver.py` on 2026-08-15 under the 800-line-per-file gate. Every body
below moved verbatim, comments and measurements with it; nothing was rewritten, because the
goldens these produce cannot be regenerated without the pinned venv and the GPU.

**The cut is the one the driver already drew.** `k3_anchor_driver` binds `torch` inside `main`
rather than at import, so that `--compare` and `--by-operator` re-score vendored bytes on a
machine with no torch, no fla and no GPU. Everything on that side of the line lives here: the
tiny config derivation, the golden container, and the two scorers with the gate they assert.
Nothing in this module imports torch, and nothing here may.

`k3_anchor_driver` re-exports every name defined here. `tests/k3-anchor.sh` and every recipe in
`docs/measurement/k3-reference/anchor.md` name the driver as the entry point, so the split may
not narrow what that name resolves to.
"""

import json
import math
import pathlib
import struct


# ---------------------------------------------------------------------------------------------
# The tiny config.

# **Widths only, and no width may collapse a distinction the real config keeps.** That second
# rule was learned the hard way: the first version of this table set `hidden_size` 128,
# `intermediate_size` 128, `moe_intermediate_size` 32, `routed_expert_hidden_size` 64,
# `kv_lora_rank` 16 and `qk_nope_head_dim` 16, which made FOUR pairs accidentally equal that the
# real config separates —
#
#   | pair | tiny (old) | real |
#   |---|---|---|
#   | `kv_lora_rank` vs `qk_nope_head_dim` | 16 == 16 | 512 vs 128 |
#   | `2 * moe_intermediate_size` vs `routed_expert_hidden_size` | 64 == 64 | 6144 vs 3584 |
#   | `hidden_size` vs `intermediate_size` | 128 == 128 | 7168 vs 33792 |
#   | `hidden_size` vs KDA `num_heads * head_dim` | 128 == 128 | 7168 vs 12288 |
#
# and each equality deletes a hazard. A port that read the KV latent width from
# `qk_nope_head_dim`, or the shared expert's width from the latent instead of
# `num_shared_experts * moe_intermediate_size` — the `[hidden, 2*moe_inter]` coupling this port
# has a recorded trap for — produced a **bit-identical** fixture, not merely a shape-valid one.
# Found by review 2026-08-11, before any kernel was scored against it.
#
# Equalities the real config DOES have are kept: `qk_nope_head_dim == v_head_dim` (128 == 128),
# `num_attention_heads == linear_attn_config.num_heads` (96 == 96), and
# `routed_expert_hidden_size == hidden_size / 2` (3584 == 7168/2).
#
# One residual collision, kept deliberately: MLA's `kv_b_proj` output
# (`heads * (qk_nope + v_head)` = 128) equals the KDA projection width. Breaking it needs a
# non-power-of-two head count, which fla's triton kernels block over and refuse, and a port that
# confuses MLA's KV expansion with KDA's projection is confusing two different layer families.
TINY_TEXT = {
    "hidden_size": 192,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 256,
    "moe_intermediate_size": 24,
    "routed_expert_hidden_size": 96,
    "q_lora_rank": 32,
    "kv_lora_rank": 24,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "vocab_size": 256,
    "max_position_embeddings": 64,
    # The real ids (bos 163584, eos 163586, pad 163839) do not fit a 256-entry vocab, and
    # `nn.Embedding` refuses a `padding_idx` outside `num_embeddings` rather than ignoring it.
    # `pad` stays the last row so `padding_idx`'s zeroing still applies to it — see
    # `init_weights`, which has to put that row back.
    "bos_token_id": 250,
    "eos_token_id": 251,
    "pad_token_id": 255,
}
# `head_dim` stays a power of two: fla's triton kernels block over K and V and refuse degenerate
# widths. `num_heads` 4 keeps the real config's `num_attention_heads == num_heads`.
TINY_LINEAR_ATTN = {"head_dim": 32, "num_heads": 4}

# Structural fields the tiny config MUST inherit unchanged, asserted at generation time AND
# recorded in the golden as `structural_asserted` so `tests/k3_anchor.rs` can refuse to check a
# field this list does not cover. Two lists that must agree, with nothing keeping them in step,
# is the drift review found here on 2026-08-11: `gate_lower_bound` and `short_conv_kernel_size`
# were claimed asserted on both sides and were asserted on neither.
STRUCTURAL = [
    "num_hidden_layers",
    "first_k_dense_replace",
    "moe_layer_freq",
    "attn_res_block_size",
    "num_shared_experts",
    "routed_scaling_factor",
    "latent_moe_use_norm",
    "moe_renormalize",
    "moe_router_activation_func",
    "activation_situ_beta",
    "activation_situ_linear_beta",
    "hidden_act",
    "mla_use_nope",
    "mla_use_output_gate",
    "rms_norm_eps",
    "use_grouped_topk",
    "num_expert_group",
    "topk_group",
    "topk_method",
]
# The same, one level down. `linear_attn_config` is REBUILT by a dict merge rather than inherited
# whole, so its survivors need their own check -- and `gate_lower_bound` is a kernel kwarg with
# its own defect run below, which makes it the last field that should have gone unasserted.
STRUCTURAL_LINEAR_ATTN = ["gate_lower_bound", "short_conv_kernel_size", "use_full_rank_gate"]

# Layers whose every submodule output is captured. Chosen for what each one IS, since G1b names
# four coverage classes and these cover all of them plus the two structural boundaries:
#   0  — KDA (1-based list holds 1) AND the only dense `mlp` layer AND an attn-res block start
#   1  — the first MoE layer, KDA
#   3  — the first MLA layer (1-based 4)
#   12 — an attn-res block start that is not layer 0
#   91 — MLA, and the first of the two consecutive MLA layers the real map ends with
#   92 — the last layer, MLA
CAPTURE_LAYERS = (0, 1, 3, 12, 91, 92)


def build_config(cfg_mod, real_path):
    """The tiny `KimiLinearConfig`, derived from the vendored real one."""
    real = json.loads(pathlib.Path(real_path).read_text())["text_config"]
    d = dict(real)
    # `quantization_config` goes: it is what would make the experts MXFP4, and it is anchored on
    # real bytes elsewhere. `dtype`/`architectures` go because they are load-time hints, and
    # `_name_or_path` because it would put a stale path in the metadata.
    for k in ("quantization_config", "dtype", "architectures", "_name_or_path", "auto_map"):
        d.pop(k, None)
    d.update(TINY_TEXT)
    d["linear_attn_config"] = dict(real["linear_attn_config"], **TINY_LINEAR_ATTN)
    cfg = cfg_mod.KimiLinearConfig(**d)
    for k in STRUCTURAL:
        got, want = getattr(cfg, k), real[k]
        assert got == want, f"tiny config lost structural field {k}: {got!r} != real {want!r}"
    for k in STRUCTURAL_LINEAR_ATTN + ["kda_layers", "full_attn_layers"]:
        got, want = cfg.linear_attn_config[k], real["linear_attn_config"][k]
        assert got == want, f"tiny config lost linear_attn_config.{k}: {got!r} != {want!r}"
    return cfg


# ---------------------------------------------------------------------------------------------
# The container. Byte-for-byte the layout `src/v4oracle/golden.rs` reads, under a K3 magic.

MAGIC = b"RIVK3GLD"


def _u64(v):
    return struct.pack("<Q", v)


def _s(x):
    b = x.encode()
    return _u64(len(b)) + b


def read_golden(path):
    """The inverse of [`write_golden`], for `--compare`.

    Each tensor keeps its RAW payload bytes alongside the decoded values, because that is the
    only exact equality test: `-0.0 == 0.0` in python and `NaN != NaN`, and `golden.rs` compares
    `to_bits` for the same reason. No vendored tensor is non-finite today, so this is a guard
    rather than a fix.
    """
    b = pathlib.Path(path).read_bytes()
    assert b[:8] == MAGIC, f"{path}: not a rivoli K3 anchor golden"
    o = [8]

    def u64():
        v = struct.unpack_from("<Q", b, o[0])[0]
        o[0] += 8
        return v

    def s():
        n = u64()
        v = b[o[0]:o[0] + n].decode()
        o[0] += n
        return v

    meta = [(s(), s()) for _ in range(u64())]
    sections = []
    for width, code in ((4, "f"), (8, "q")):
        items = {}
        for _ in range(u64()):
            name = s()
            shape = [u64() for _ in range(u64())]
            n = u64()
            raw = b[o[0]:o[0] + n * width]
            vals = struct.unpack_from(f"<{n}{code}", b, o[0])
            o[0] += n * width
            # A duplicate name would collapse here, last-wins, and shrink the compared set --
            # while the Rust reader keeps both. Latent today; loud now.
            assert name not in items, f"{path}: duplicate tensor name {name}"
            items[name] = (tuple(shape), vals, raw)
        sections.append(items)
    assert o[0] == len(b), f"{path}: {len(b) - o[0]} trailing bytes"
    return dict(meta), sections[0], sections[1]


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


# ---------------------------------------------------------------------------------------------
# Scoring. Two reports over the same arithmetic, and the gate one of them asserts.

# **Which operator a tensor inside a layer belongs to, by the prefix of its name below the layer.**
# A table rather than a chain of `if`s, and ORDER IS SEMANTIC: `block_sparse_moe.routed_expert`
# must be tested before the `block_sparse_moe` it starts with. `self_attn` is not in the table
# because its answer depends on the layer (below); it is not shadowed by `self_attention_res`
# either way -- the two names diverge at the ninth character (`self_attn` vs `self_atte`).
_LAYER_OPERATORS = (
    ("kda.", "kda_op"),
    (("self_attention_res", "mlp_res"), "attn_res"),
    ("block_sparse_moe.routed_expert", "moe_latent"),
    ("block_sparse_moe", "moe_route"),
    ("mlp", "dense_mlp"),
)


def _layer_operator(rest, kda):
    """The operator owning `rest`, the part of a captured name below `model.layers.<n>.`.

    `self_attn` is the one prefix a name cannot answer alone: it is MLA on 24 layers and KDA's
    trunk on 69, which is why the caller has to read the partition out of the golden's own
    `tiny_config`.
    """
    if rest.startswith("self_attn"):
        return "kda_trunk" if kda else "mla"
    for prefix, operator in _LAYER_OPERATORS:
        if rest.startswith(prefix):
            return operator
    if "layernorm" in rest:
        return "norm"
    return "residual"


# **Which operator each captured tensor belongs to.** Layer numbers alone cannot say: `self_attn`
# is MLA on 24 layers and KDA on 69, so the classifier has to read the partition out of the
# golden's own `tiny_config`. Used only by `--by-operator`, which is how the per-operator
# tolerances in `anchor.md` are derived — a tolerance is a property of an OPERATOR, and a per-layer
# report cannot state one.
def operator_of(name, kda_layers):
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer, rest = int(parts[2]), ".".join(parts[3:])
        return _layer_operator(rest, layer in kda_layers)
    if name.startswith("model.output_attn_res"):
        return "attn_res"
    if name == "model.norm":
        return "norm"
    return "head"


def _score(items_a, items_b, key_of):
    """Group two tensor sections by `key_of(name)` and score each group.

    One scorer for both reports: `compare` groups by layer to ask about localisation,
    `by_operator` groups by operator to ask about tolerance, and the arithmetic is identical.
    """
    per = {}
    for name, (shape, y, raw_b) in items_b.items():
        key = key_of(name)
        n, diff, rel = per.get(key, (0, 0, 0.0))
        shape_a, x, raw_a = items_a[name]
        assert shape_a == shape, f"{name}: shape {shape_a} vs {shape}"
        scale = max((abs(v) for v in y if math.isfinite(v)), default=0.0) or 1e-30
        # Non-finite pairs are skipped rather than folded: python's `max` silently discards a NaN
        # unless it comes first, which would read as agreement.
        d = max(
            (abs(p - q) for p, q in zip(x, y) if math.isfinite(p) and math.isfinite(q)),
            default=0.0,
        )
        per[key] = (n + 1, diff + int(raw_a != raw_b), max(rel, d / scale))
    return per


def _both_sections(a_path, b_path):
    """Load two goldens and refuse a tensor-set mismatch, which is a harness bug either way."""
    ma, fa, ia = read_golden(a_path)
    mb, fb, ib = read_golden(b_path)
    if set(fa) != set(fb) or set(ia) != set(ib):
        only = (set(fa) ^ set(fb)) | (set(ia) ^ set(ib))
        raise SystemExit(
            f"{a_path} and {b_path} capture different tensors ({len(only)} on one side only, "
            f"e.g. {sorted(only)[:3]}) -- the harness is wrong, not the reference",
        )
    return (ma, mb), ((fa, fb), (ia, ib))


def _merged(sections, key_of):
    """Fold [`_score`] over both tensor sections into one `key -> (tensors, differing, max_rel)`.

    Shared by both reports, which differ only in what they key by: `compare` keys by layer to ask
    about localisation, `by_operator` keys by operator to ask about tolerance, and the merge is
    the same arithmetic either way.
    """
    per = {}
    for items_a, items_b in sections:
        for k, v in _score(items_a, items_b, key_of).items():
            n, diff, rel = per.get(k, (0, 0, 0.0))
            per[k] = (n + v[0], diff + v[1], max(rel, v[2]))
    return per


def _print_rows(header, per, keys):
    """The one table both reports print, in `keys` order: how many tensors the bucket holds, how
    many of them differ BYTE-for-byte, and the worst relative gap in it."""
    print(f"{header}\ttensors\tdiffering\tmax_rel")
    for k in keys:
        n, diff, rel = per[k]
        print(f"{k}\t{n}\t{diff}\t{rel:.3e}")


def by_operator(a_path, b_path):
    """Per-OPERATOR agreement between two goldens -- the shape a tolerance has to be stated in.

    Two uses, and the tolerance table needs both: `fp32 vs fp64` gives the floor an independent
    correct implementation cannot beat, and `None vs <defect>` gives the signal a tolerance has to
    stay under. `anchor.md`'s table is these two numbers per operator.
    """
    (ma, mb), sections = _both_sections(a_path, b_path)
    kda = _kda_zero_based(json.loads(ma["tiny_config"]))
    per = _merged(sections, lambda n: operator_of(n, kda))
    print(f"# {mb.get('defect')}/{mb.get('dtype')} vs {ma.get('defect')}/{ma.get('dtype')}")
    _print_rows("operator", per, sorted(per))


def _kda_zero_based(cfg):
    """`kda_layers` is 1-based on disk (`is_kda_layer` tests `layer_idx + 1`)."""
    return {i - 1 for i in cfg["linear_attn_config"]["kda_layers"]}


# **The captured layers each defect must leave BIT-IDENTICAL.** This is the half of `k3-port.md`
# §G rule 1 that "something changed" does not cover, and until 2026-08-11 it lived only in a
# markdown table that a human read. `--compare` asserts it.
#
# Only UPSTREAM layers can be listed: the goldens come from one forward pass, so a perturbation
# at layer 3 reaches layer 92 by construction. A defect whose first touch is layer 0 has no green
# layer at all, and that is recorded as an empty list rather than omitted, so a missing entry is
# an error instead of a silent pass.
EXPECT_GREEN = {
    "MlaLoraEps1e5": [0, 1],          # KDA layers, upstream of the first MLA layer (3)
    "MlaScaleFromNope": [0, 1],       # same
    "ExpertW1W3Swap": [0],            # layer 0 has no routed experts
    "RouterBiasInWeight": [0],        # same
    "LatentNormAfterUp": [0],         # same
    "DenseMlpGateUpSwap": [],         # layer 0's own dense MLP is the first thing it touches
    "AttnResNormalisedValues": [],    # layer 0's MLP fold is the first
    "KdaNoQkL2Norm": [],              # layer 0 is a KDA layer
    "KdaGateLowerBoundOff": [],
    "KdaStateLayout": [],
    "KdaBetaSigmoidOutside": [],
}


def _layer_of(name):
    """The bucket a captured tensor is reported under: its layer index, or the model-level fold."""
    return name.split(".")[2] if name.startswith("model.layers.") else "model"


def _layer_sort_key(key):
    """Numeric layers in index order, with the model-level bucket last."""
    return (key == "model", int(key) if key.isdigit() else 0)


def _captured_layers(per):
    """The numeric buckets in index order.

    Numeric keys ONLY: `per` also holds "model" for the model-level fold, and sorting the whole
    key set by `int` raises on it. That arm had never executed when it was written -- every
    exercise of the comparator tripped an earlier gate first -- which is how the crash was found.
    """
    return sorted((k for k in per if k.isdigit()), key=int)


def _gate_captured(defect, per, green):
    """Every layer a defect declares green must actually have been CAPTURED.

    An uncaptured one used to score as green, because `per.get(layer, (0, 0, 0))[1]` reads an
    absent layer as zero differing tensors. The localisation claim would then rest on an empty
    set: drop a layer from CAPTURE_LAYERS, or name a layer outside it in EXPECT_GREEN, and the
    matrix prints, finds nothing reddened, and exits 0. Found by review 2026-08-11.
    """
    absent = [layer for layer in green if layer not in per]
    if absent:
        raise SystemExit(
            f"--defect {defect} declares layer(s) {absent} green, but nothing captured them -- "
            f"captured: {sorted(per)}. An uncaptured layer is not evidence of localisation.",
        )


def _gate_reddened(defect, per, green):
    """...and every one of them must still be bit-identical, which is the localisation claim."""
    reddened = [layer for layer in green if per[layer][1]]
    if reddened:
        raise SystemExit(
            f"--defect {defect} reddened layer(s) {reddened}, which are upstream of it and must "
            f"stay bit-identical -- the localisation this golden claims is gone",
        )


def _gate_downstream(defect, per, green):
    """The POSITIVE half, which was only "something, somewhere, differs".

    EXPECT_GREEN encodes a boundary, so the first captured layer PAST it must actually redden --
    otherwise a perturbation that missed its operator and only disturbed something downstream
    reads as a localised, detected defect while the arithmetic the cell prices was never
    exercised.
    """
    downstream = [layer for layer in _captured_layers(per) if layer not in green]
    if not downstream:
        return
    if per[downstream[0]][1]:
        return
    raise SystemExit(
        f"--defect {defect} left layer {downstream[0]} bit-identical, the first captured layer "
        f"it does NOT declare green -- so whatever it changed, it was not this operator here. "
        f"Either the perturbation missed, or EXPECT_GREEN names the wrong boundary.",
    )


def _gate_defect(defect, base_defect, per):
    """`k3-port.md` §G rule 1 over one scored pair, in four parts.

      * the defect changed SOMETHING -- a defect that reddens nothing is not a defect;
      * every layer in `EXPECT_GREEN[defect]` was captured, and is BIT-IDENTICAL. Only upstream
        layers can be there, since one forward pass propagates everything downstream;
      * the first captured layer past that boundary did redden.

    A golden scored against itself (`base_defect` equal) is the `None`-vs-`None` case and gates
    nothing; a defect with no EXPECT_GREEN entry is an error, since an omitted entry would
    otherwise pass silently.
    """
    if defect == base_defect:
        return
    if defect not in EXPECT_GREEN:
        raise SystemExit(f"--defect {defect} has no EXPECT_GREEN entry; add one, even if empty")
    if not any(diff for _, diff, _ in per.values()):
        raise SystemExit(f"--defect {defect} changed NOTHING; that is not a defect run")
    green = list(map(str, EXPECT_GREEN[defect]))
    _gate_captured(defect, per, green)
    _gate_reddened(defect, per, green)
    _gate_downstream(defect, per, green)


def compare(a_path, b_path):
    """Score one defect run against the `None` golden, per captured layer, and GATE it.

    Reported per LAYER rather than per tensor because that is what G1b asks about, and 800 tensor
    lines cannot be read. What is asserted is `k3-port.md` §G rule 1 -- see [`_gate_defect`].

    The two tensor SETS must match exactly. They are a property of the config and the capture
    list, never of the numbers -- so a mismatch is a broken harness, not a defect finding, and it
    aborts. That distinction was measured into existence: while routed experts were captured
    individually, four of five defects reported `inf` for most layers because a moved routing
    fires a different set of expert modules.
    """
    (ma, mb), sections = _both_sections(a_path, b_path)
    per = _merged(sections, _layer_of)
    defect = mb.get("defect")
    print(f"# {defect} vs {ma.get('defect')}  mode={ma.get('mode')}")
    _print_rows("layer", per, sorted(per, key=_layer_sort_key))
    _gate_defect(defect, ma.get("defect"), per)
