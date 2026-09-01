"""The deviceless half of the S2 Qwen3.8-Flash-Next anchor: tiny config, container, scoring.

Nothing here imports torch, and nothing here may -- `qwen_anchor_driver` binds `torch` inside
`main` so that `--compare` and `--by-operator` re-score vendored bytes on a machine with no
torch and no venv, which is the property that lets a defect matrix be re-read from the bytes
long after the run. That cut is the same one `k3_anchor_lib.py` draws, for the same reason.

**What is different here, and it is the headline of the whole S2 stage: this reference runs on
a CPU.** Qwen3.8-Flash-Next's GatedDeltaNet ships PURE-TORCH implementations in transformers
itself -- `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`, MOD:266-398 --
behind a `@use_kernel_func_from_hub_with_fallback("...", "fla")` decorator whose fallback is
that torch body whenever `fla` is not importable. K3's anchor needed a GPU because fla's KDA
ops are triton-only and its "naive" twin took none of the seven kwargs; that finding does NOT
transfer, and it was checked rather than assumed (`qwen-reference/anchor.md` section "CPU or
GPU"). So these goldens are generated with no device, no lock and no lease.

Three implementation SELECTORS decide what arithmetic a golden pins, and every one of them
defaults to something other than the model file's own body:

  * `config._attn_implementation` defaults to **sdpa** (`modeling_utils.py`'s
    `get_correct_attn_implementation`). The indexer's own comment says only eager and sdpa are
    allowed, and the two take different branches inside it -- `attention_mask == 0` versus a
    bool mask, and a float versus a bool return. The driver pins **eager**.
  * `config._experts_implementation` defaults to **grouped_mm**, i.e.
    `torch._grouped_mm` out of `integrations/moe.py`, NOT `Qwen4ExpTextExperts.forward`'s own
    body at MOD:859-896 which `qwen-architecture.md` section 6 cites. The driver pins **eager**,
    which is that body -- and which is also the only one that compiles for fp64, so it is what
    makes the tolerance floors measurable at all. The two disagree by **4.673e-6** relative on the
    logits, and that disagreement is a first-party-versus-first-party floor in its own right.

    > **CORRECTED 2026-08-31.** This line said 1.760e-6, which disagreed with the same measurement
    > in `anchor.md`'s two statements of it (4.673e-6) -- one number, three places, two values, and
    > nothing re-derived any of them. Re-run rather than reconciled:
    > `qwen_anchor_driver.py --config docs/measurement/qwen-reference/config.json --mode moe-equiv`
    > printed `logits max_rel 4.673e-06` at salt `qwen-anchor-1`, exit 0, 3.0 s. That is the one
    > number, and `anchor.md` now cites this run rather than an inherited figure.
  * the fla / `causal_conv1d` / `kernels` fallback above. The driver ASSERTS all three are
    absent rather than trusting the venv.

A golden that did not record all three would be a golden of an unknown model.
"""

import json
import math
import pathlib
import struct

# ---------------------------------------------------------------------------------------------
# The tiny config.

# **Widths only, plus the two scalars argued below -- and no width may collapse a distinction
# the real config keeps.** That second rule is [`WIDTH_AUDIT`], which is ASSERTED at generation
# time rather than described here: K3's anchor learned the rule from a review that found four
# accidentally-equal pairs after the fact, and a rule that only lives in a comment gets broken
# by the next width edit. Depth is free and structure is what the traps live in, so all 48
# layers, the real `layer_types` (with its `full_attention` spelling, so the alias rewrite at
# CFG:180-184 is exercised), the real interval, `ple_layer_ids: [2]`, `hc_count: 4`, both conv
# kernels at 4, `indexer_compress_ratio: 4`, `ngram_size: 3` and `heads_per_ngram: 8` all
# survive untouched.
TINY_TEXT = {
    "hidden_size": 96,
    # 4 query heads over 2 KV heads: n_rep = 2, deliberately NOT the GDN 16:48 ratio of 3.
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    # 3 key heads / 9 value heads keeps the real 1:3 `repeat_interleave` exactly, and 3 is not
    # `num_key_value_heads` -- real separates them (16 vs 2) and so must this.
    "linear_key_head_dim": 20,
    "linear_value_head_dim": 20,
    "linear_num_key_heads": 3,
    "linear_num_value_heads": 9,
    # Real is 128, which is BOTH `head_dim / 2` and `2 * rotary_dim`. This is the one place the
    # tiny config deliberately breaks an equality the real config has: nothing in the reference
    # couples them (`validate_architecture` only demands `rotary_dim <= indexer_head_dim`), and
    # a port that read the index head width off `head_dim / 2` would produce a BIT-IDENTICAL
    # fixture at real widths. Separating them costs no realism and buys a real check.
    "indexer_head_dim": 24,
    # **Scaled so the selection path actually runs.** `block_topk = budget // ratio` and
    # MOD:695's `topk(min(block_topk, num_complete_blocks))` makes QSA DENSE BY CONSTRUCTION
    # whenever `|visible| <= budget + ratio - 1` -- 2051 tokens at the real budget, which is
    # why `qwen-architecture.md` T19 records the alias as invisible to every planned oracle.
    # At budget 8 that boundary drops to 11, so a 16-token window holds 11 dense queries AND 5
    # selective ones (measured: 8 of 16 tokens kept at the last position). T19 and the whole
    # selection-and-masking path become visible at tiny widths; that is what this number is for.
    "indexer_budget": 8,
    # 640 == 640 in the real config, so the equality is KEPT.
    "moe_intermediate_size": 28,
    "shared_expert_intermediate_size": 28,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    # hidden / 8, the real ratio (2560 / 8 = 320).
    "hc_lowrank": 12,
    # == hidden_size in the real config, so kept equal; 96 / 16 n-gram heads = 6 per head.
    "ple_embed_dim": 96,
    # The 16 head vocabularies are the 1st..16th primes strictly above `base - 1`. Shrinking the
    # base keeps the RULE (16 distinct primes, exclusive-prefix-sum offsets) while the table
    # fits in a fixture. The real-parameter hash is pinned separately and at FULL width by
    # `--mode ngram-real`, which is where the 20M space and the checkpoint's own I64 buffers
    # are compared.
    "ngram_vocab_size_base": 1024,
    "vocab_size": 512,
    "max_position_embeddings": 64,
    # The real ids (bos/eos 248044) do not fit a 512-row vocab. `eos` is load-bearing rather
    # than cosmetic: it pads the n-gram history (MOD:1076) and resets the context at every
    # document boundary (MOD:1053-1067), and `validate_architecture` refuses a PLE config
    # without it.
    "bos_token_id": 501,
    "eos_token_id": 500,
}

# `rope_theta` and `mrope_section` are the two NON-width overrides, and each is forced.
#
#   * `mrope_section` must index into `rotary_dim // 2` frequencies; the real [11, 11, 10] sums
#     to 32 for the real 64 rotary dims, and [2, 1, 1] sums to 4 for the tiny 8. It keeps the
#     real shape of two equal sections and one smaller. mRoPE stays inert for text either way
#     (MOD:140-155 overwrites `freqs[0]` slices with identical values), which is the point.
#   * `rope_theta` 1e7 is meaningless at 8 rotary dims. `inv_freq = theta ** -(0, .25, .5, .75)`
#     is (1, 1.78e-2, 3.16e-4, 5.62e-6) there, so over positions 0..15 three of the four
#     frequency pairs are numerically inert and BOTH partial-rope defect rows would be vacuous.
#     At theta 10 the four angles at position 15 are 15.0, 8.4, 4.7 and 2.7 radians -- spread
#     across the circle, which is what the real theta buys at the real width. The measured cost
#     of getting this wrong is in `anchor.md` section "rope_theta is a width".
TINY_ROPE = {"rope_theta": 10.0, "mrope_section": [2, 1, 1]}

# Keys dropped from the real `text_config` before the tiny overrides land. `mtp*` goes because
# v1 excludes the draft layer by name and the reference itself ignores it
# (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`); the rest are load-time hints that would
# put a stale path or a quantization scheme into the metadata.
DROP_KEYS = (
    "dtype",
    "architectures",
    "_name_or_path",
    "auto_map",
    "quantization_config",
    "mtp",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
)

# Structural fields the tiny config MUST inherit unchanged, asserted at generation time AND
# recorded in the golden as `structural_asserted` so `tests/qwen_anchor.rs` can refuse to check
# a field this list does not name. Two lists with nothing keeping them in step is the drift
# K3's review found; the metadata key is what ties them.
STRUCTURAL = [
    "num_hidden_layers",
    "full_attention_interval",
    "hc_count",
    "linear_conv_kernel_dim",
    "ple_conv_kernel_size",
    "ple_layer_ids",
    "ngram_size",
    "heads_per_ngram",
    "make_ngram_vocab_size_divisible_by",
    "split_ngram_parts",
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_compress_ratio",
    "output_gate_type",
    "hidden_act",
    "rms_norm_eps",
    "norm_topk_prob",
    "seed",
    "tie_word_embeddings",
    "attention_bias",
    "attention_dropout",
    "initializer_range",
    "layer_types",
]

# `norm_topk_prob` and `seed` are in that list and are **absent from the checkpoint's
# config.json** -- they come from `Qwen4ExpTextConfig`'s own defaults (True and 1234, CFG:163
# and CFG:156). Asserting them against the real config's *effective* value rather than against
# a literal is what makes trap T25 real: a port that reads only the file finds nothing to
# honour, and the default is on.
DEFAULTED_STRUCTURAL = ("norm_topk_prob", "seed")

# **The width-collision audit, as assertions -- and it is scored against the REAL config as well
# as the tiny one.** Each entry is `(label, lhs_key, rhs_key)` over a dict of derived widths.
#
# > **CORRECTED 2026-08-31, and the correction is why the audit now reads the real config.** This
# > block said "Left: pairs the real config keeps DISTINCT", which was FALSE for one row of the
# > nineteen: `half_head_dim` and `indexer_head_dim` are BOTH 128 in the real config, so the row
# > stated the tiny config's own choice as a property of the model. The deviation was argued in
# > `TINY_TEXT`'s `indexer_head_dim` comment and nowhere in the audit's DATA -- the split this
# > file elsewhere refuses. It is now [`REAL_COLLISIONS`], checked at BOTH ends, along with two
# > further real collisions the same evaluation found and nobody had written down: coverage
# > nobody records is coverage the next width edit deletes for free.
#
# Pairs the real config keeps DISTINCT, so a tiny width that collapses one would let a port read
# the wrong field and still produce a bit-identical fixture.
DISTINCT_PAIRS = [
    ("hidden vs attn head_dim", "hidden_size", "head_dim"),
    ("attn head_dim vs indexer head_dim", "head_dim", "indexer_head_dim"),
    ("attn head_dim vs GDN key head_dim", "head_dim", "linear_key_head_dim"),
    ("rotary_dim vs head_dim/2", "rotary_dim", "half_head_dim"),
    ("rotary_dim vs indexer head_dim", "rotary_dim", "indexer_head_dim"),
    ("attn heads vs KV heads", "num_attention_heads", "num_key_value_heads"),
    ("GDN key heads vs value heads", "linear_num_key_heads", "linear_num_value_heads"),
    ("GDN key heads vs attn KV heads", "linear_num_key_heads", "num_key_value_heads"),
    ("attn n_rep vs GDN key->value repeat", "attn_n_rep", "gdn_repeat"),
    ("hidden vs moe_intermediate", "hidden_size", "moe_intermediate_size"),
    ("hidden vs hc_lowrank", "hidden_size", "hc_lowrank"),
    ("hc_lowrank vs GDN key head_dim", "hc_lowrank", "linear_key_head_dim"),
    ("2*moe_intermediate vs hidden", "gate_up_width", "hidden_size"),
    ("experts vs experts_per_tok", "num_experts", "num_experts_per_tok"),
    ("ngram per-head dim vs GDN key head_dim", "ngram_head_dim", "linear_key_head_dim"),
    ("GDN conv_dim vs GDN value_dim", "conv_dim", "value_dim"),
    ("GDN value_dim vs hc stream width", "value_dim", "hc_width"),
    ("indexer qk width vs attn q width", "index_qk_width", "q_proj_width"),
]

# Equalities the real config HAS, kept so the fixture stays like the model -- including the
# square `[d_k][d_v]` GDN state, which is square at 128 == 128 in the real model too and is
# therefore a hazard the fixture must CARRY rather than hide (`gdn_state_axis_swapped` is what
# prices it).
#
# > **A fifth row died here 2026-08-31**, `("ple conv dilation vs ngram_size", ...)`: it was
# > TAUTOLOGICAL, because `derived_widths` computes `ple_conv_dilation` AS `cfg.ngram_size`, so
# > the row compared a value to itself and no width edit could have broken it. A gate row that
# > cannot fail is decoration, and this one read as coverage of the dilated conv's stride. The
# > claim is real and is now checked where it CAN fail -- the driver reads `dilation` off the
# > constructed `ple.conv1d` as `ple_conv_dilation_observed`, a number the module chose rather
# > than one this file restated. [`assert_width_audit`] now refuses any pair naming the same key
# > twice, so that shape of mistake cannot come back.
EQUAL_PAIRS = [
    ("moe vs shared expert width (640 == 640)", "moe_intermediate_size", "shared_expert_intermediate_size"),
    ("ple_embed_dim vs hidden (2560 == 2560)", "ple_embed_dim", "hidden_size"),
    ("GDN key vs value head_dim (128 == 128, the square state)", "linear_key_head_dim", "linear_value_head_dim"),
    ("hc_lowrank vs hidden/8 (320 == 2560/8)", "hc_lowrank", "hidden_over_8"),
]

# **Collisions the REAL config has and the tiny config deliberately BREAKS.** Checked at both
# ends: equal in the real widths, distinct in the tiny ones. Each is a place where the real model
# cannot score a port on which field it read, and the fixture can. None of the three is
# structural -- the reference couples none of these pairs and `validate_architecture` demands only
# `rotary_dim <= indexer_head_dim` -- so separating them costs no realism. What each buys:
REAL_COLLISIONS = [
    # 128 == 128 in real, and it is ALSO `2 * rotary_dim` there (64), so a port reading the index
    # head width off either `head_dim / 2` or twice the rotary width produces a BIT-IDENTICAL
    # fixture at real widths. Tiny: rotary 8, half_head 16, indexer 24 -- all three separated,
    # and `2 * 8 = 16 != 24` breaks the second reading too.
    ("head_dim/2 vs indexer head_dim (128 == 128 in real)", "half_head_dim", "indexer_head_dim"),
    # 10240 == 10240 in real: the hyper-connection stream stack (`4 * 2560`) and the GDN fused
    # conv width (`2*2048 + 6144`) coincide. A port that sized the conv buffer off the hc stack
    # would be right at real widths by accident. Tiny: 384 against 300.
    ("hc stream width vs GDN conv_dim (10240 == 10240 in real)", "hc_width", "conv_dim"),
    # 640 == 640 in real: the indexer's fused qk projection `(4+1)*128` and `moe_intermediate_size`
    # coincide -- two quantities from different subsystems. Tiny: 120 against 28.
    ("indexer qk width vs moe_intermediate (640 == 640 in real)", "index_qk_width", "moe_intermediate_size"),
]


def derived_widths(cfg):
    """Every width the audit and the shape assertions are stated in, derived from the config.

    Derived rather than written down, for `k3_anchor.rs`'s recorded reason: a literal agrees
    with a drifted config, and a derived width fails.
    """
    rope = cfg.rope_parameters
    rotary_dim = int(cfg.head_dim * rope.get("partial_rotary_factor", 1.0))
    ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    return {
        "hidden_size": cfg.hidden_size,
        "head_dim": cfg.head_dim,
        "half_head_dim": cfg.head_dim // 2,
        "rotary_dim": rotary_dim,
        "indexer_head_dim": cfg.indexer_head_dim,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "attn_n_rep": cfg.num_attention_heads // cfg.num_key_value_heads,
        "gdn_repeat": cfg.linear_num_value_heads // cfg.linear_num_key_heads,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "shared_expert_intermediate_size": cfg.shared_expert_intermediate_size,
        "gate_up_width": 2 * cfg.moe_intermediate_size,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "hc_lowrank": cfg.hc_lowrank,
        "hidden_over_8": cfg.hidden_size // 8,
        "hc_width": cfg.hc_count * cfg.hidden_size,
        "ple_embed_dim": cfg.ple_embed_dim,
        "ngram_heads": ngram_heads,
        "ngram_head_dim": cfg.ple_embed_dim // ngram_heads,
        "ngram_size": cfg.ngram_size,
        "ple_conv_dilation": cfg.ngram_size,
        "key_dim": cfg.linear_key_head_dim * cfg.linear_num_key_heads,
        "value_dim": cfg.linear_value_head_dim * cfg.linear_num_value_heads,
        "conv_dim": 2 * cfg.linear_key_head_dim * cfg.linear_num_key_heads
        + cfg.linear_value_head_dim * cfg.linear_num_value_heads,
        "index_qk_width": (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim,
        "q_proj_width": cfg.num_attention_heads * cfg.head_dim * 2,
        "block_topk": cfg.indexer_budget // cfg.indexer_compress_ratio,
        "dense_below": cfg.indexer_budget + cfg.indexer_compress_ratio - 1,
    }


def width_audit_pairs():
    """The audit as DATA, for the golden's metadata.

    Recorded in the bytes so `crates/oracles/tests/qwen_anchor.rs` can re-run it against the
    recorded widths instead of carrying a second copy of the pair lists. Two lists that must
    agree with nothing tying them together is the drift this repo has already paid for; here the
    Rust side reads the list the python side actually checked.
    """
    return {
        "distinct": [[label, a, b] for label, a, b in DISTINCT_PAIRS],
        "equal": [[label, a, b] for label, a, b in EQUAL_PAIRS],
        "collide_in_real": [[label, a, b] for label, a, b in REAL_COLLISIONS],
    }


def real_config_widths(cfg_cls, real_path):
    """The derived widths of the UNSHRUNK config, for the audit's real-side half.

    Built from the same file and the same `DROP_KEYS` as the tiny config and through the same
    `derived_widths`, so the two sides of every audit row are computed by one piece of code. No
    `validate_architecture` call and no model: this is arithmetic over the checkpoint's own
    numbers.

    Takes the CLASS rather than the module so `_metadata` can reach it as `type(cfg)`: that path
    has the built config and not the import, and re-importing would let the audit's two halves
    come from two different module objects.
    """
    real = json.loads(pathlib.Path(real_path).read_text())["text_config"]
    d = {k: v for k, v in real.items() if k not in DROP_KEYS}
    return derived_widths(cfg_cls(**d))


def assert_width_audit(w, real_w):
    """The audit, run against BOTH configs. A width edit that collapses a real distinction fails
    here, at generation time, before a single golden byte is written.

    Four checks, and the last two are what the 2026-08-31 correction added:

      * no row names the same key twice -- a tautological row cannot fail, and one had been
        sitting in `EQUAL_PAIRS` reading as coverage of the PLE conv's dilation;
      * every `DISTINCT_PAIRS` row is distinct in the TINY widths;
      * ...and distinct in the REAL widths too, because that is the claim the list makes. A row
        that is equal in real belongs in `REAL_COLLISIONS`, and the failure says so;
      * `REAL_COLLISIONS` at both ends -- equal in real AND distinct in tiny. Both halves earn
        their place: a row that stopped being a real collision is a stale claim about the
        checkpoint, and a row the tiny config stopped separating is lost coverage. Checking only
        one end is how an exemption list becomes permanent.
    """
    for label, a, b in DISTINCT_PAIRS + EQUAL_PAIRS + REAL_COLLISIONS:
        assert a != b, f"tautological audit row: {label} compares {a} to itself and cannot fail"
    for label, a, b in DISTINCT_PAIRS:
        assert w[a] != w[b], f"width collision: {label} -- both {w[a]}, so a port cannot be scored on which it read"
        assert real_w[a] != real_w[b], (
            f"{label} is EQUAL in the real config (both {real_w[a]}), so DISTINCT_PAIRS states "
            f"something false about the checkpoint -- move it to REAL_COLLISIONS and argue it"
        )
    for label, a, b in EQUAL_PAIRS:
        assert w[a] == w[b], f"lost a real equality: {label} -- {w[a]} != {w[b]}"
        assert real_w[a] == real_w[b], (
            f"{label} is NOT an equality in the real config ({real_w[a]} != {real_w[b]}), so the "
            f"tiny config is keeping a coincidence rather than the model's structure"
        )
    for label, a, b in REAL_COLLISIONS:
        assert real_w[a] == real_w[b], (
            f"{label} is no longer a collision in the real config ({real_w[a]} != {real_w[b]}) -- "
            f"the row is a stale claim; move it to DISTINCT_PAIRS"
        )
        assert w[a] != w[b], (
            f"{label} collides in the TINY config too (both {w[a]}) -- the fixture has stopped "
            f"separating a pair the real model cannot separate, which is coverage lost"
        )


# Layers whose every submodule output is captured, chosen for what each one IS:
#   0  -- the first GDN layer, and the first MoE
#   1  -- the PLE HOST. `ple_layer_ids: [2]` is ONE-indexed (MOD:1202 tests `layer_idx + 1`),
#         so the injection sits on `layer_idx` 1 -- the S0 correction, and the reason a capture
#         list written from the config's own digits would watch the wrong layer.
#   2  -- the GDN layer immediately downstream of the injection, which is what gives every
#         PLE-side defect a captured green boundary at layer 0 and a captured red at layer 1.
#   3  -- the FIRST QSA layer (`layer_types[3]` is the checkpoint's `full_attention`, aliased to
#         `qwen_sparse_attention` by CFG:180-184), so every indexer defect has three green
#         layers upstream of it.
#   27 -- a mid-pattern QSA layer, far from both ends.
#   46, 47 -- the last GDN and the last QSA: the pair the real 48-layer pattern ends with.
CAPTURE_LAYERS = (0, 1, 2, 3, 27, 46, 47)

# The window. 16 positions with `indexer_budget` 8 puts 11 queries in the provably-dense regime
# and 5 in the selective one, in ONE forward pass -- so T8's "both directions" and T19's alias
# are scored by the same golden.
SEQ = 16


def build_config(cfg_mod, real_path, attn_impl="eager", experts_impl="eager"):
    """The tiny `Qwen4ExpTextConfig`, derived from the vendored real `config.json`."""
    real = json.loads(pathlib.Path(real_path).read_text())["text_config"]
    d = {k: v for k, v in real.items() if k not in DROP_KEYS}
    d.update(TINY_TEXT)
    d["rope_parameters"] = dict(real["rope_parameters"], **TINY_ROPE)
    cfg = cfg_mod.Qwen4ExpTextConfig(**d)
    # The reference's own validation, run explicitly: `hc_count > 1`, the QSA field group,
    # `indexer_kv_heads == 1`, `budget % ratio == 0`, `rotary_dim <= indexer_head_dim`,
    # `ple_embed_dim % ngram_heads == 0`, one-indexed `ple_layer_ids` on a `linear_attention`
    # host, and an `eos_token_id`. A tiny config that fails it is not a tiny version of this
    # model.
    cfg.validate_architecture()
    reference = dict(real)
    for key in DEFAULTED_STRUCTURAL:
        reference.setdefault(key, getattr(cfg_mod.Qwen4ExpTextConfig, key))
    for key in STRUCTURAL:
        got, want = getattr(cfg, key), reference[key]
        if key == "layer_types":
            # The checkpoint spells the QSA layers `full_attention` and the reference REWRITES
            # that to `qwen_sparse_attention` before validating (CFG:180-184, comment: "layers
            # that are actually using an indexer"). So the survival check is on the ALIASED
            # form, entry by entry -- a partition that drifted in the middle is exactly what a
            # prefix or a count would pass.
            want = ["qwen_sparse_attention" if t == "full_attention" else t for t in want]
        assert got == want, f"tiny config lost structural field {key}: {got!r} != real {want!r}"
    assert_width_audit(
        derived_widths(cfg), real_config_widths(cfg_mod.Qwen4ExpTextConfig, real_path)
    )
    cfg._attn_implementation = attn_impl
    cfg._experts_implementation = experts_impl
    return cfg


# ---------------------------------------------------------------------------------------------
# The container. Byte-for-byte the layout `crates/oracles/src/golden.rs` reads, under the
# `RIVQWGLD` magic that landed at S1 -- before this driver existed, so that a shared file four
# other models read did not have to be edited by track B.

MAGIC = b"RIVQWGLD"


def _u64(v):
    return struct.pack("<Q", v)


def _s(x):
    b = x.encode()
    return _u64(len(b)) + b


def read_golden(path):
    """The inverse of [`write_golden`], for `--compare` and `--by-operator`.

    Each tensor keeps its RAW payload bytes beside the decoded values, because that is the only
    exact equality test: `-0.0 == 0.0` and `NaN != NaN` in python, and `golden.rs` compares
    `to_bits` for the same reason.
    """
    b = pathlib.Path(path).read_bytes()
    assert b[:8] == MAGIC, f"{path}: not a rivoli Qwen anchor golden"
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
            # A duplicate name would collapse here, last-wins, shrinking the compared set --
            # while the Rust reader keeps both. Latent; loud.
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
