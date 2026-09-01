"""Scoring two anchor captures against each other, and the gate one of the reports asserts.

Split out of `qwen_anchor_lib` on 2026-09-01, when that file crossed the 800-line soft cap. The
seam is the one the file already marked: everything here consumes goldens that
`qwen_anchor_lib.read_golden` has already parsed, and nothing here is needed to BUILD a config or
WRITE a capture. So the halves are "make the bytes" and "judge the bytes", and only the second
half knows what a defect row is.

**Still deviceless, and that property is load-bearing.** `qwen_anchor_lib`'s header records that
nothing on its side may import torch, so that `--compare` runs on a box with no GPU and no venv
beyond the standard library. This module inherits that rule for the same reason and by the same
means: `math` and `qwen_anchor_lib` are the only imports, and a torch import here would silently
take `--compare` away from every machine that cannot build the model.

`EXPECT_FIRST_TOUCH` moved with the code that reads it. Prose that cited it as
`qwen_anchor_lib.EXPECT_FIRST_TOUCH` was updated in the same commit rather than left to rot --
a stale module path in a comment is how the next reader learns to distrust the comments.
"""

import math

from qwen_anchor_lib import read_golden

# ---------------------------------------------------------------------------------------------
# Scoring: two reports over one piece of arithmetic, and the gate one of them asserts.

# **Which operator a captured tensor belongs to.** A table rather than a chain of `if`s, and
# ORDER IS SEMANTIC -- `self_attn.indexer` must be tested before the `self_attn` it starts with,
# and `linear_attn.norm` before `linear_attn`. The buckets are the kernels S3 writes, which is
# what a tolerance is a property of; a per-layer report cannot state one.
_LAYER_OPERATORS = (
    ("ple.ple_embedding", "ngram_hash"),
    ("ple", "ple_inject"),
    # The two folds are separate buckets and that is load-bearing, not tidiness: they sit on
    # OPPOSITE sides of the attention and the MoE (MOD:1222 and MOD:1238), so a single `hc`
    # bucket would put a post-MLP tensor upstream of the MoE in the ordering below and every
    # router row's localisation claim would fail on its own downstream fold.
    ("attn_hyper_connection", "hc_attn"),
    ("mlp_hyper_connection", "hc_mlp"),
    ("linear_attn.gated_delta_rule", "gdn_op"),
    ("linear_attn.conv", "gdn_conv"),
    ("linear_attn.norm", "gdn_out_norm"),
    # `in_proj_*` are UPSTREAM of the recurrence and `out_proj` plus the block's own output are
    # downstream of it, so they cannot share a bucket. Measured, not foreseen: with both under
    # one `gdn_proj`, `gdn_decay_sigmoid_gate` reddened 2 of its 6 tensors and the first-touch
    # gate correctly refused the run -- `out_proj` and `model.layers.0.linear_attn` were the two.
    ("linear_attn.in_proj", "gdn_proj"),
    ("linear_attn", "gdn_out"),
    ("self_attn.indexer", "indexer"),
    ("self_attn.q_proj", "qsa_proj"),
    ("self_attn.k_proj", "qsa_proj"),
    ("self_attn.v_proj", "qsa_proj"),
    ("self_attn.q_norm", "qk_norm"),
    ("self_attn.k_norm", "qk_norm"),
    ("self_attn", "qsa"),
    ("mlp.gate", "moe_route"),
    ("mlp.shared_expert", "moe_shared"),
    ("mlp", "moe"),
)


def operator_of(name):
    """The operator owning a captured tensor, from its name alone.

    Unlike K3's classifier this needs no partition lookup: a Qwen layer is `linear_attention`
    or `qwen_sparse_attention` and the two use DIFFERENT submodule names (`linear_attn` vs
    `self_attn`), so the name says which. That is a property of the reference, not a
    convenience -- and it is why a dense reading of the 12 aliased layers is a defect about
    behaviour rather than about naming.
    """
    if name.startswith("model.layers."):
        rest = ".".join(name.split(".")[3:])
        for prefix, operator in _LAYER_OPERATORS:
            if rest.startswith(prefix):
                return operator
        return "residual"
    if name.startswith("model.hyper_connection_mixer"):
        return "hc_collapse"
    return "head"


# **Where each captured tensor sits in the forward pass**, as an integer. The defect gate needs
# a linear order over buckets, not a set of layer numbers: every Qwen layer carries a MoE, an
# n-gram host, two hyper-connection folds and one attention family, so "the layers upstream of
# this defect" is far coarser than what the capture can actually distinguish. K3 declared a list
# of green LAYERS per defect and maintained it by hand; this declares ONE first-touched bucket
# and derives both halves of the claim from it.
#
# Within a layer the reference's order is fixed by `DecoderLayer.forward` (MOD:1207-1243):
# PLE injection, then the attention hyper-connection read, then the attention family, then the
# MLP hyper-connection read, then the MoE.
# Every position is read off the reference: `DecoderLayer.forward` is PLE, attention fold,
# attention family, MLP fold, MoE (MOD:1207-1243); `Attention.forward` is indexer, projections,
# q/k norms, rope, attend, gate, `o_proj` (MOD:790-833); `SparseMoeBlock.forward` runs the
# SHARED expert before the router (MOD:928-930).
_BUCKET_ORDER = (
    "ngram_hash",
    "ple_inject",
    "hc_attn",
    "gdn_proj",
    "gdn_conv",
    "gdn_op",
    "gdn_out_norm",
    "gdn_out",
    "indexer",
    "qsa_proj",
    "qk_norm",
    "qsa",
    "hc_mlp",
    "moe_shared",
    "moe_route",
    "moe",
    "residual",
)
# Buckets are compared LAYER-MAJOR: a defect that first touches layer 3's indexer must leave
# every tensor of layers 0-2 identical, whatever bucket it is in.


def bucket_of(name):
    """`(layer, operator)` for a captured tensor; `(BIG, ...)` for the model-level tail."""
    if name.startswith("model.layers."):
        return int(name.split(".")[2]), operator_of(name)
    return 1 << 30, operator_of(name)


def _bucket_rank(bucket):
    layer, operator = bucket
    return layer, _BUCKET_ORDER.index(operator) if operator in _BUCKET_ORDER else len(_BUCKET_ORDER)


def _score(items_a, items_b, key_of):
    """Group two tensor sections by `key_of(name)` and score each group.

    One scorer for both reports: `compare` groups by bucket to ask about localisation,
    `by_operator` groups by operator to ask about tolerance, and the arithmetic is identical.
    """
    per = {}
    for name, (shape, y, raw_b) in items_b.items():
        key = key_of(name)
        n, diff, rel = per.get(key, (0, 0, 0.0))
        shape_a, x, raw_a = items_a[name]
        assert shape_a == shape, f"{name}: shape {shape_a} vs {shape}"
        scale = max((abs(v) for v in y if math.isfinite(v)), default=0.0) or 1e-30
        # Non-finite pairs are skipped rather than folded: python's `max` silently discards a
        # NaN unless it comes first, which would read as agreement. The recorded mirror of
        # `f32::max` ignoring NaN.
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
    """Fold [`_score`] over both tensor sections into one `key -> (tensors, differing, max_rel)`."""
    per = {}
    for items_a, items_b in sections:
        for k, v in _score(items_a, items_b, key_of).items():
            n, diff, rel = per.get(k, (0, 0, 0.0))
            per[k] = (n + v[0], diff + v[1], max(rel, v[2]))
    return per


def _print_rows(header, per, keys, fmt=str):
    """The one table both reports print: how many tensors the bucket holds, how many differ
    BYTE-for-byte, and the worst relative gap in it."""
    print(f"{header}\ttensors\tdiffering\tmax_rel")
    for k in keys:
        n, diff, rel = per[k]
        print(f"{fmt(k)}\t{n}\t{diff}\t{rel:.3e}")


def operator_scores(a_path, b_path):
    """`operator -> max_rel` between two goldens. The scoring both reports and the tolerance
    table are derived from, so a row in `common/tolerance.rs` and a line of `--by-operator`
    output cannot disagree."""
    _, sections = _both_sections(a_path, b_path)
    return {k: v[2] for k, v in _merged(sections, operator_of).items()}


def by_operator(a_path, b_path):
    """Per-OPERATOR agreement -- the shape a tolerance has to be stated in.

    Two uses, and the tolerance table needs both: `fp32 vs fp64` gives the floor an independent
    correct implementation cannot beat, and `None vs <defect>` gives the signal a tolerance has
    to stay under. `anchor.md`'s table is those two numbers per operator.
    """
    (ma, mb), sections = _both_sections(a_path, b_path)
    per = _merged(sections, operator_of)
    print(f"# {mb.get('defect')}/{mb.get('dtype')} vs {ma.get('defect')}/{ma.get('dtype')}")
    _print_rows("operator", per, sorted(per))


# **Each defect's FIRST TOUCHED BUCKET.** Everything strictly before it must be bit-identical
# and the bucket itself must redden -- so one declaration carries both halves of the
# localisation claim, and neither can be satisfied vacuously. `None` means "the first thing the
# forward does", i.e. no captured bucket is upstream and only the reddening half has content.
#
# A defect with no entry is an ERROR rather than a pass, because an omitted entry is exactly
# what a hand-maintained list loses.
EXPECT_FIRST_TOUCH = {
    # GDN decay form: layer 0's own recurrence, the first arithmetic in the model after the
    # layer-0 hyper-connection read.
    "gdn_decay_sigmoid_gate": (0, "gdn_op"),
    "gdn_decay_clamped": (0, "gdn_op"),
    "gdn_dt_bias_outside_softplus": (0, "gdn_op"),
    "gdn_state_axis_swapped": (0, "gdn_op"),
    # The conv sits before the recurrence and after the projections.
    "gdn_conv_tap_reversed": (0, "gdn_conv"),
    "gdn_conv_tap_rotated": (0, "gdn_conv"),
    # The fused projection is the first thing the GDN block computes.
    "gdn_qkv_qk_swapped": (0, "gdn_proj"),
    "gdn_qkv_head_interleaved": (0, "gdn_proj"),
    # The output norm-gate is after the recurrence.
    "gdn_output_gate_silu": (0, "gdn_out_norm"),
    "gdn_output_gate_tanh": (0, "gdn_out_norm"),
    "gdn_output_gate_on_normed": (0, "gdn_out_norm"),
    "gdn_norm_unit_offset": (0, "gdn_out_norm"),
    "l2norm_eps_dropped": (0, "gdn_op"),
    # Everything that touches a norm the model uses everywhere starts at layer 0's
    # hyper-connection, which is the first module in the first layer.
    "hc_norm_no_offset": (0, "hc_attn"),
    "eps_outside_sqrt": (0, "hc_attn"),
    "hc_sum_not_mean": (0, "hc_attn"),
    "hc_residual_uses_normed": (0, "hc_attn"),
    "hc_lowrank_matrices_swapped": (0, "hc_attn"),
    "hc_mix_gate_static": (0, "hc_attn"),
    # Router and experts: layer 0 carries a MoE like every layer.
    "router_sigmoid": (0, "moe_route"),
    "router_bias_added": (0, "moe_route"),
    "router_no_renorm": (0, "moe_route"),
    "expert_gate_up_swapped": (0, "moe"),
    # The n-gram hash and the PLE injection live on layer 1, so layer 0 is a genuine captured
    # green boundary for all five.
    "ngram_seed_wrong": (1, "ngram_hash"),
    "ngram_order_swapped": (1, "ngram_hash"),
    "ngram_mix_add": (1, "ngram_hash"),
    "ngram_mod_round": (1, "ngram_hash"),
    "ple_layer_index_off_by_one": (1, "ngram_hash"),
    # The indexer and the QSA path first exist on layer 3: three captured layers upstream.
    "indexer_relu_off": (3, "indexer"),
    "indexer_budget_one_block_short": (3, "indexer"),
    "indexer_blocks_strided": (3, "indexer"),
    "indexer_rope_at_block_end": (3, "indexer"),
    "indexer_relu_after_sum": (3, "indexer"),
    "indexer_rope_before_pool": (3, "indexer"),
    "qsa_layer_is_dense": (3, "indexer"),
    "attn_gate_block_split": (3, "qsa_proj"),
    # RoPE reaches the indexer's queries before it reaches the attention's, and the indexer is
    # the first module inside layer 3's `self_attn`.
    "rope_interleaved_pairs": (3, "indexer"),
    "rope_on_tail_dims": (3, "indexer"),
}


def _gate_first_touch(defect, per):
    """The localisation claim, in three parts that cannot each pass vacuously.

      * the defect changed SOMETHING -- a defect that reddens nothing is not a defect;
      * the declared first-touched bucket was CAPTURED, and it reddens. An uncaptured bucket
        would score as green through a `.get(..., 0)` and the whole claim would rest on an
        empty set, which is the fail-open K3's review found;
      * every captured bucket strictly BEFORE it is bit-identical.
    """
    if defect not in EXPECT_FIRST_TOUCH:
        raise SystemExit(f"--defect {defect} has no EXPECT_FIRST_TOUCH entry; add one")
    if not any(diff for _, diff, _ in per.values()):
        raise SystemExit(f"--defect {defect} changed NOTHING; that is not a defect run")
    first = EXPECT_FIRST_TOUCH[defect]
    if first not in per:
        raise SystemExit(
            f"--defect {defect} declares {first} as its first touched bucket, but nothing "
            f"captured it -- captured: {sorted(per, key=_bucket_rank)[:8]}... An uncaptured "
            f"bucket is not evidence of localisation.",
        )
    if not per[first][1]:
        raise SystemExit(
            f"--defect {defect} left its OWN first bucket {first} bit-identical. Either the "
            f"perturbation missed the operator it names, or EXPECT_FIRST_TOUCH is wrong.",
        )
    rank = _bucket_rank(first)
    reddened = [b for b in per if _bucket_rank(b) < rank and per[b][1]]
    if reddened:
        raise SystemExit(
            f"--defect {defect} reddened {sorted(reddened, key=_bucket_rank)}, which are "
            f"UPSTREAM of {first} and must stay bit-identical -- the localisation is gone",
        )


def compare(a_path, b_path):
    """Score one defect run against the `None` golden, per bucket, and GATE it.

    The two tensor SETS must match exactly. They are a property of the config and the capture
    list, never of the numbers -- so a mismatch is a broken harness, not a defect finding, and
    it aborts. K3 measured that distinction into existence: while routed experts were captured
    individually, four of five defects reported `inf` for most layers because a moved routing
    fires a different set of expert modules. The same exclusion applies here (see
    `qwen_anchor_capture.py`).
    """
    (ma, mb), sections = _both_sections(a_path, b_path)
    per = _merged(sections, bucket_of)
    defect = mb.get("defect")
    print(f"# {defect} vs {ma.get('defect')}  mode={ma.get('mode')}")
    _print_rows(
        "bucket",
        per,
        sorted(per, key=_bucket_rank),
        fmt=lambda b: f"{'model' if b[0] == 1 << 30 else b[0]}.{b[1]}",
    )
    if defect != ma.get("defect"):
        _gate_first_touch(defect, per)
