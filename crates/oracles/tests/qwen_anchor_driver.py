#!/usr/bin/env python3
"""**The S2 anchor: goldens produced by Qwen3.8-Flash-Next's own first-party stack, not by ours.**

`docs/investigations/qwen-flash-next-port.md` census item 4 calls this mandatory, and the reason
is that four of the nine operator classes are attested by NOTHING but code -- the gated delta
rule's inner arithmetic, the QSA indexer's selection, the hyper-connection fold and the hashed
n-gram embedding are each described by the tech report only up to a formula, and the constants
that decide them live in `modeling_qwen4_exp.py`. A fixture derived from a reading of that file
would put the same misreading in the spec, in the golden and in the kernel checked against it,
so the golden has to come from RUNNING the reference. Nothing here re-implements the model; the
two places that transcribe reference arithmetic are named at their definitions and one of them
is gated by a bit-exact identity mode.

What runs is `transformers.models.qwen4_exp` at **v5.16.1** -- the earliest released tag that
carries the model dir at all, so the venv the other four ports share cannot be reused -- over a
**tiny config derived from the vendored real `config.json`**: all 48 layers, the real
`layer_types` including its `full_attention` spelling, the real interval, `ple_layer_ids: [2]`,
`hc_count: 4`, both conv kernels, `indexer_compress_ratio: 4`, `ngram_size: 3`,
`heads_per_ngram: 8`. Only widths shrink, and `qwen_anchor_lib.assert_width_audit` refuses a
width that collapses a distinction the real config keeps.

**No GPU, and that is a finding rather than a convenience.** K3's anchor needs a device because
fla's KDA ops are triton kernels with no CPU path; Qwen's GatedDeltaNet ships PURE-TORCH
implementations in transformers itself (MOD:266-398) behind a fallback decorator that selects
them whenever `fla` is not importable. This driver ASSERTS `fla`, `kernels` and `causal_conv1d`
are all absent rather than trusting the venv, because which of them is installed silently
decides what arithmetic the golden pins.

**Five declared deviations**, each recorded in the golden's own metadata:

  * `_attn_implementation` is set to **eager**; the default is `sdpa`. The indexer branches on
    the mask's dtype (`attention_mask == 0` vs a bool mask) and returns a float or a bool mask
    to match, so the two are different code paths inside the operator this anchor exists to pin.
  * `_experts_implementation` is set to **eager**; the default is **grouped_mm**, i.e.
    `torch._grouped_mm` from `integrations/moe.py` and NOT `Qwen4ExpTextExperts.forward`'s own
    body. Eager is the body `qwen-architecture.md` section 6 cites, and it is the only one that
    compiles for fp64 -- which is what makes the tolerance floors measurable. `--mode moe-equiv`
    measures the two against each other.
  * `Qwen4ExpForCausalLM` on the text config, not `Qwen4ExpForConditionalGeneration`. v1 is
    text-only and the wrapper contributes no text-side arithmetic. Recorded as `entry_point`.
  * The routed experts are plain fp32 parameters (`quantized=no`): `quantization_config` is
    dropped from the tiny config, and a block-128 scale grid does not exist at these widths. The
    fp8 grid is a converter assert at S4 and a real-byte micro-anchor, not this.
  * The model runs in torch's default **fp32** while the checkpoint is bf16. Right for a
    reference -- the point is to pin arithmetic, not one accumulation order -- but it is the
    number the tolerance decision starts from, so `dtype` is in the metadata.

Usage (`--config` is the vendored real config; `--ref` is not needed, the reference is the venv):

    python3 qwen_anchor_driver.py --config docs/measurement/qwen-reference/config.json \\
        --out golden.bin --mode decode --defect None --salt qwen-anchor-1

`--compare A B` scores two goldens and asserts the defect's declared first-touched bucket, which
is the half of "prove it can go red" that a bare "something changed" check leaves out.
`--by-operator A B` is the same arithmetic keyed by operator, which is the shape a tolerance has
to be stated in.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

# The moved halves, re-exported: `qwen_anchor_lib` is the deviceless side (no torch, so
# `--compare` still runs without one), `qwen_anchor_capture` holds the hooks, `qwen_anchor_taps`
# the values no hook can see and `qwen_anchor_defects` the matrix. All four sit beside this file,
# which is `sys.path[0]` whenever it runs as a script -- the only way it is ever run.
from qwen_anchor_capture import Capture, Tap, assert_captured, hook_model  # noqa: F401
from qwen_anchor_defects import CLASSES, DEFECTS, assert_class_coverage  # noqa: F401
from qwen_anchor_lib import (  # noqa: F401
    CAPTURE_LAYERS,
    DEFAULTED_STRUCTURAL,
    EXPECT_FIRST_TOUCH,
    MAGIC,
    SEQ,
    STRUCTURAL,
    TINY_ROPE,
    TINY_TEXT,
    build_config,
    by_operator,
    compare,
    operator_scores,
    derived_widths,
    operator_of,
    real_config_widths,
    width_audit_pairs,
    read_golden,
    write_golden,
)
from qwen_anchor_taps import (  # noqa: F401
    restore_reference,
    wrap_conv_ops,
    wrap_gdn_modules,
    wrap_gdn_ops,
    wrap_indexer,
    wrap_ple,
    wrap_rope,
    wrap_softplus,
)

# Bound by `main`, so `--compare` and `--by-operator` run with no torch and no venv -- the
# property that lets a defect run be re-scored from vendored bytes on any machine.
torch = None

# The packages whose presence would silently change what the golden pins. All three are checked
# and all three must be ABSENT: `fla` and `causal_conv1d` would replace the torch bodies through
# `use_kernel_func_from_hub_with_fallback`, and `kernels` would let `kernelize` pull a hub kernel
# on top of either.
FORBIDDEN_PACKAGES = ("fla", "kernels", "causal_conv1d")


# ---------------------------------------------------------------------------------------------
# Deterministic weights.


def _gen(name, salt):
    """A generator seeded by the parameter's NAME, not by its position in the module tree.

    One global seed would make every golden depend on construction order, so adding a capture or
    reordering a module would move numbers that nothing about the model changed. Keyed by name,
    a parameter's values are a property of its name alone -- which is also what makes
    `expert_gate_up_swapped` and `gdn_qkv_qk_swapped` non-vacuous: every block gets a distinct
    draw, so a swap cannot cancel.
    """
    h = hashlib.sha256(f"{salt}/{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def _draw(salt):
    """A named uniform draw, for the one defect that has to INVENT a tensor."""

    def draw(name, n, lo, hi):
        return torch.empty(n).uniform_(lo, hi, generator=_gen(name, salt))

    return draw


# The four draw families, and each range is a MEASURED property of the real checkpoint rather
# than a guess. `qwen-architecture.md`'s T1 provenance table is where the numbers come from.
#
#   * `Qwen4ExpTextRMSNorm.weight` is `torch.zeros`-initialised and the forward applies
#     `(1.0 + w)` -- ZERO-centred. Measured `layers.0.attn_hyper_connection.hc_norm.weight`:
#     mean -0.06355 over 10240 entries, min -5.9375, max +6.625.
#   * `Qwen4ExpTextRMSNormGated.weight` is `torch.ones`-initialised and the forward applies a
#     bare `w` -- ONES-centred. Measured `layers.0.linear_attn.norm.weight`: mean +0.96677 over
#     128 entries, min +0.875, max +1.02344. The 256 bytes are vendored in tree.
#   * `A_log` is the reference's own `log(uniform(0.01, 16))` (MOD:434-435). It is the decay
#     rate: too small and every head's state freezes, too large and it is erased, and either
#     way a kernel that ignored the term would still match.
#   * `dt_bias` on (-2, 2) rather than the reference's `torch.ones` placeholder. Measured real
#     range is [-8.0, +2.53] with mean -3.29, so this sits inside it -- and the range is chosen
#     so `softplus` and `sigmoid` SEPARATE over the heads, which is what
#     `gdn_decay_sigmoid_gate` needs to be non-vacuous (softplus(2)/sigmoid(2) = 2.41).
_ZERO_CENTRED = "Qwen4ExpTextRMSNorm"
_ONES_CENTRED = "Qwen4ExpTextRMSNormGated"


def _owner_types(model):
    """Every parameter's owning module type, by parameter name.

    Type rather than a substring of the name, which is what K3's init used: this model has two
    RMSNorm formulas whose parameters are BOTH called `<something>.weight`, and drawing them on
    one scale would put the fixture off the measured scale of one of them -- silently, and in the
    direction that makes `hc_norm_no_offset` and `gdn_norm_unit_offset` weak.
    """
    owners = {}
    for mod_name, mod in model.named_modules():
        for p_name, _ in mod.named_parameters(recurse=False):
            owners[f"{mod_name}.{p_name}" if mod_name else p_name] = type(mod).__name__
    return owners


def init_weights(model, salt):
    """Fill every parameter deterministically, in four families (see the note above)."""
    owners = _owner_types(model)
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = _gen(name, salt)
            flat = torch.empty(p.numel(), dtype=torch.float32)
            owner = owners.get(name, "")
            if name.endswith("A_log"):
                flat.uniform_(0.01, 16.0, generator=g).log_()
            elif name.endswith("dt_bias"):
                flat.uniform_(-2.0, 2.0, generator=g)
            elif owner == _ONES_CENTRED:
                flat.uniform_(0.8, 1.2, generator=g)
            elif owner == _ZERO_CENTRED:
                flat.uniform_(-0.2, 0.2, generator=g)
            else:
                flat.uniform_(-0.08, 0.08, generator=g)
            p.copy_(flat.view(p.shape).to(p.dtype).to(p.device))
    # No `padding_idx` row to restore: the real `text_config` sets `pad_token_id: null` and the
    # tiny config keeps it null, so `nn.Embedding`'s zeroing never applies. Stated because K3's
    # anchor had to put that row back and a reader will look for the same line here.


# ---------------------------------------------------------------------------------------------
# The reference, and refusing the wrong one.


def sha_of(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def _reference_paths():
    """The two pinned reference blobs, inside the venv. The model repo carries no `auto_map` and
    no `modeling_*.py`, so the in-tree transformers class IS the shipped implementation."""
    from transformers.models import qwen4_exp

    d = pathlib.Path(qwen4_exp.__file__).parent
    return d / "modeling_qwen4_exp.py", d / "configuration_qwen4_exp.py"


def preflight_env():
    """Refuse to generate under an environment other than the one the vendored goldens were made
    in, and under any environment that would silently swap an implementation.

    Two halves. The version pins are READ OUT OF THE VENDORED GOLDEN, never restated here: those
    bytes already carry what produced them and `qwen_anchor.rs` already asserts them, so a copy
    in this file would be a third statement of one fact. A deliberate re-pin therefore needs no
    edit here -- regenerate, re-vendor, and the new bytes become the new pin.

    The forbidden-package half is not a version check and cannot be read off a golden that was
    produced correctly: `fla`, `causal_conv1d` and `kernels` each REPLACE a torch body at import
    time through `use_kernel_func_from_hub_with_fallback`, and the replacement keeps the torch
    function's `__name__` (it is `functools.wraps`-ed), so the golden's own metadata would still
    say `torch_chunk_gated_delta_rule`. The name lies; the absence is what has to be asserted.
    """
    import transformers

    present = [p for p in FORBIDDEN_PACKAGES if importlib.util.find_spec(p) is not None]
    if present:
        raise SystemExit(
            f"qwen-anchor: {present} installed in this env. Each one REPLACES a pure-torch\n"
            "  reference body through the kernel-from-hub fallback while keeping its name, so\n"
            "  the golden would pin a different implementation and say otherwise. Use a venv\n"
            "  without them."
        )
    vendored = sorted(pathlib.Path(__file__).parent.glob("qwen-anchor-decode-*.bin"))
    if not vendored:
        print("qwen-anchor: no vendored golden to check this env against", file=sys.stderr)
        return
    pinned = read_golden(vendored[0])[0]
    live = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }
    drift = [f"{k}: golden says {pinned.get(k)!r}, this env has {v!r}" for k, v in live.items() if pinned.get(k) != v]
    if drift:
        raise SystemExit(
            f"qwen-anchor: this env is not the one that produced {vendored[0].name}:\n  "
            + "\n  ".join(drift)
            + "\n  Fix QWEN_ANCHOR_VENV, or re-vendor the bytes AND the versions qwen_anchor.rs asserts."
        )


def _build_model(mdl_mod, cfg, args):
    """The reference model, initialised, at the requested precision."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = mdl_mod.Qwen4ExpForCausalLM(cfg)
    # Set on the config BEFORE construction and re-asserted after, because a model class may
    # adjust either selector in `__init__` (`get_correct_attn_implementation` silently falls back
    # to eager when sdpa cannot dispatch, and `_check_and_adjust_experts_implementation` runs at
    # `post_init`). What the metadata records is what the model ENDED UP with.
    model.eval()
    init_weights(model, args.salt)
    if args.dtype == "float64":
        # AFTER `init_weights`, so the drawn values are identical to the fp32 run's to the bit
        # and the only difference measured is arithmetic. This works end to end here -- there is
        # no fp32 island to carve out, because the reference bodies are plain torch -- but ONLY
        # with `experts_implementation=eager`: `torch._grouped_mm` refuses Double, which is the
        # measured reason that selector is pinned rather than left at its default.
        model.double()
    return model


def _new_ctx(cfg, model, args):
    """The mutable per-run state every defect, tap and hook shares."""
    types = cfg.layer_types
    return {
        "model": model,
        "gdn_layer_ids": [i for i, t in enumerate(types) if t == "linear_attention"],
        "qsa_layer_ids": [i for i, t in enumerate(types) if t == "qwen_sparse_attention"],
        # ONE-indexed on disk (MOD:1202 tests `layer_idx + 1`), so the host is id - 1.
        "ple_layer_ids": [i - 1 for i in cfg.ple_layer_ids],
        "gdn_calls": 0,
        "conv_calls": 0,
        "indexer_calls": 0,
        "ple_calls": 0,
        "rope_calls": 0,
        "capturing": False,
        "draw": _draw(args.salt),
        "gdn_tags": set(),
        "conv_tags": set(),
    }


def _reset_counters(ctx):
    for key in ("gdn_calls", "conv_calls", "indexer_calls", "ple_calls", "rope_calls"):
        ctx[key] = 0


def _install(mdl_mod, tap, ctx):
    """Taps on, in the order that lets the defects compose with them.

    Defects run FIRST (see `_generate`): `rope_on_tail_dims` replaces
    `apply_rotary_pos_emb` and `rope_interleaved_pairs` replaces `rotate_half`, so `wrap_rope`
    must wrap whatever is there rather than the pristine function. `wrap_indexer` reads the
    variant the defect asked for out of `ctx`.
    """
    restore_reference(mdl_mod)
    wrap_softplus(mdl_mod, tap)
    wrap_gdn_ops(mdl_mod, tap)
    wrap_conv_ops(mdl_mod, tap)
    wrap_rope(mdl_mod, tap)
    wrap_ple(mdl_mod, tap)
    wrap_indexer(mdl_mod, tap, ctx.get("indexer_variant"))
    return wrap_gdn_modules(mdl_mod, tap)


def _capture_pass(model, tap, args, vocab):
    """Run the mode's forward pass with the hooks armed, and return the hooked names.

    `decode` warms the prefill UNHOOKED and captures exactly one step: the cache, the GDN
    recurrent state and the indexer's key cache are what make decode a different arithmetic path
    from prefill (`torch_recurrent_gated_delta_rule`, not `torch_chunk_gated_delta_rule`), and
    capturing the prefill too would triple the file for tensors the prefill golden already holds.
    """
    ids = torch.arange(1, args.seq + 1).unsqueeze(0) % vocab
    with torch.no_grad():
        if args.mode == "prefill":
            _reset_counters(tap.ctx)
            handles, expected = hook_model(model, tap.cap, tap.layers)
            tap.ctx["capturing"] = True
            out = model(input_ids=ids, use_cache=True)
        else:
            warm = model(input_ids=ids, use_cache=True)
            _reset_counters(tap.ctx)
            handles, expected = hook_model(model, tap.cap, tap.layers)
            tap.ctx["capturing"] = True
            nxt = torch.tensor([[(args.seq + 1) % vocab]])
            out = model(input_ids=nxt, past_key_values=warm.past_key_values, use_cache=True)
        tap.cap.add("logits", out.logits)
    for h in handles:
        h.remove()
    return expected


# ---------------------------------------------------------------------------------------------
# Metadata.


def _ple_conv_dilation(model, cfg):
    """The dilation the PLE's `nn.Conv1d` was actually built with, and the assert that it is
    `ngram_size`.

    Asserted here rather than in the width audit because this is the one form of the claim that
    can fail: the audit's version compared `derived_widths`' `ple_conv_dilation` -- defined AS
    `cfg.ngram_size` -- to `cfg.ngram_size`. `nn.Conv1d.dilation` is a tuple; the conv is 1-D, so
    there is exactly one entry and indexing it is not a choice.
    """
    host = cfg.ple_layer_ids[0] - 1
    dilation = model.model.layers[host].ple.conv1d.dilation
    assert len(dilation) == 1, f"the PLE conv is 1-D; dilation is {dilation}"
    assert dilation[0] == cfg.ngram_size, (
        f"the PLE dilated conv was built with dilation {dilation[0]}, not ngram_size "
        f"{cfg.ngram_size} -- the n-gram stride and the conv stride have come apart"
    )
    return dilation[0]


def _metadata(args, model, cfg, cap):
    """Everything the bytes have to carry about what produced them.

    A golden that hides what produced it is worse than no golden -- and here that includes the
    three implementation SELECTORS, because each of them defaults to something other than what
    this run pins and none of them is visible in the numbers.
    """
    import transformers

    mod_sha, cfg_sha = (sha_of(p) for p in _reference_paths())
    return [
        ("defect", args.defect),
        ("mode", args.mode),
        ("seq", str(args.seq)),
        ("salt", args.salt),
        ("dtype", str(next(model.parameters()).dtype)),
        ("entry_point", type(model).__name__),
        ("quantized", "no" if getattr(cfg, "quantization_config", None) is None else "yes"),
        ("torch", torch.__version__),
        ("transformers", transformers.__version__),
        ("python", ".".join(str(v) for v in sys.version_info[:3])),
        ("attn_implementation", model.config._attn_implementation),
        ("experts_implementation", model.config._experts_implementation),
        # Which pure-torch body actually ran, by name, and which conv entry point. Asserted
        # rather than assumed: the fallback decorator keeps the torch function's `__name__`
        # whatever it dispatched to, so this is only evidence alongside the forbidden-package
        # check in `preflight_env`.
        ("gdn_entry", ",".join(sorted(model.config.__dict__.get("_qwen_anchor_gdn_tags", ())))),
        ("conv_entry", ",".join(sorted(model.config.__dict__.get("_qwen_anchor_conv_tags", ())))),
        ("forbidden_absent", ",".join(FORBIDDEN_PACKAGES)),
        # The layers ACTUALLY hooked, not the module constant: a narrowed run whose metadata
        # claimed the full list would let `qwen_anchor.rs`'s census pass over a short capture.
        ("capture_layers", ",".join(str(i) for i in args.capture_layers_parsed)),
        # Which module names recurred within one forward, and how often. Empty for every decode
        # golden and thirteen-deep for the prefill one, whose indexer norm runs once per query
        # position -- see `Capture.add`. In the bytes because a reader counting distinct names
        # has to know the count is not the number of tensors, and because the ladder of shapes
        # under the repeated name IS the dense/selective boundary this fixture exists to show.
        ("repeated_captures", json.dumps(cap.repeated(), sort_keys=True)),
        ("structural_asserted", ",".join(STRUCTURAL)),
        ("defect_classes", json.dumps({k: list(v) for k, v in CLASSES.items()}, sort_keys=True)),
        ("tiny_config", json.dumps(cfg.to_dict(), sort_keys=True, default=str)),
        ("widths", json.dumps(derived_widths(cfg), sort_keys=True)),
        # The UNSHRUNK widths beside the tiny ones, so the audit's real-side half is re-runnable
        # from the bytes. Without them `qwen_anchor.rs` could only re-check that the tiny config
        # separates what the list says it separates -- never that the list's claim ABOUT THE
        # CHECKPOINT is true, which is the half that was false until 2026-08-31.
        ("real_widths", json.dumps(real_config_widths(type(cfg), args.config), sort_keys=True)),
        ("width_audit", json.dumps(width_audit_pairs(), sort_keys=True)),
        # The PLE dilated conv's dilation, read off the CONSTRUCTED module rather than restated
        # from `ngram_size`. It replaces an `EQUAL_PAIRS` row that compared `ngram_size` to
        # itself; see that list's dated note. The host layer is `ple_layer_ids` minus one,
        # because the field is one-indexed on disk (MOD:1202).
        ("ple_conv_dilation_observed", str(_ple_conv_dilation(model, cfg))),
        ("first_touch", json.dumps({k: list(v) for k, v in EXPECT_FIRST_TOUCH.items()}, sort_keys=True)),
        ("ref_modeling_sha256_16", mod_sha),
        ("ref_config_sha256_16", cfg_sha),
        ("real_config_sha256_16", sha_of(args.config)),
    ]


# ---------------------------------------------------------------------------------------------
# The three measurement modes. None of them writes a golden: what each produces is a number.


def _fresh(args, experts_impl=None, dtype=None, variant=None):
    """One built, initialised model plus its `(cfg, mdl_mod, tap)` -- the measurement modes each
    need two of these in one process and must not differ in anything but the axis measured."""
    from transformers.models.qwen4_exp import configuration_qwen4_exp as cfg_mod
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mdl_mod

    cfg = build_config(cfg_mod, args.config, experts_impl=experts_impl or args.experts_impl)
    stashed = args.dtype
    args.dtype = dtype or args.dtype
    model = _build_model(mdl_mod, cfg, args)
    args.dtype = stashed
    ctx = _new_ctx(cfg, model, args)
    if variant is not None:
        ctx["indexer_variant"] = variant
    tap = Tap(Capture(), ctx, args.capture_layers_parsed)
    _install(mdl_mod, tap, ctx)
    return cfg, mdl_mod, tap, model


def _logits_of(args, model, seq):
    ids = torch.arange(1, seq + 1).unsqueeze(0) % model.config.vocab_size
    with torch.no_grad():
        return model(input_ids=ids, use_cache=True).logits.detach()


def gdn_equiv(args):
    """**The GDN operator's tolerance floor, from the two implementations the reference ships.**

    `torch_chunk_gated_delta_rule` computes T positions at once with a chunk size of 64;
    `torch_recurrent_gated_delta_rule` walks one position at a time. Same mathematics, same
    authors, same file -- so their disagreement is exactly what "a correct implementation in
    fp32, associating its sums differently" costs, which is the number a HIP kernel's tolerance
    has to sit above. It is measured over EVERY GDN layer, because a bound wants the worst case.

    This is K3's `--mode kda-equiv` idea, and here it is a SECOND floor rather than the only one:
    the fp64 run works end to end, so the same operator also has a rounding floor from
    `--dtype float64`. Both are reported in `anchor.md` and the LARGER is what the table uses --
    the fp64 island measures sensitivity to slightly different inputs, while chunk-vs-recurrent
    measures two real implementations disagreeing, which is what a port will be.
    """
    _, _, tap, model = _fresh(args)
    ctx, seq = tap.ctx, args.seq
    sink = ctx["equiv_sink"] = {}
    ctx["capturing"] = False
    with torch.no_grad():
        ids = torch.arange(1, seq + 1).unsqueeze(0) % model.config.vocab_size
        _reset_counters(ctx)
        ctx["equiv_tag"] = "chunk"
        model(input_ids=ids, use_cache=True)
        ctx["equiv_tag"] = "steps"
        past = None
        for i in range(seq):
            _reset_counters(ctx)
            one = torch.tensor([[(i + 1) % model.config.vocab_size]])
            past = model(input_ids=one, past_key_values=past, use_cache=True).past_key_values
    rows = []
    for layer in ctx["gdn_layer_ids"]:
        chunked = sink.get(("chunk", layer))
        stepped = sink.get(("steps", layer))
        assert chunked and stepped and len(stepped) == seq, f"layer {layer}: {len(stepped or ())} steps for {seq}"
        rows.append((layer, _step_rel(chunked[0], stepped)))
    worst = max((r for _, r in rows), default=0.0)
    print("# torch_chunk_gated_delta_rule vs torch_recurrent_gated_delta_rule, same weights")
    print(f"gdn layers\t{len(rows)}\nworst max_rel\t{worst:.3e}")
    print("layer\tmax_rel")
    for layer, rel in rows[:6]:
        print(f"{layer}\t{rel:.3e}")


def _step_rel(chunked, stepped):
    """Worst relative disagreement between one layer's chunked output and its per-step twin.

    Scaled by the step's own largest magnitude, floored at 1e-30 so an all-zero position does not
    divide by zero.
    """
    rel = 0.0
    for i, step in enumerate(stepped):
        a, b = chunked[:, i].reshape(-1), step.reshape(-1)
        scale = max(abs(float(v)) for v in b) or 1e-30
        rel = max(rel, max(abs(float(x) - float(y)) for x, y in zip(a, b)) / scale)
    return rel


def moe_equiv(args):
    """The MoE path's floor, from `grouped_mm` against `eager` -- two first-party implementations
    of the same experts, on identical weights."""
    rels = []
    for impl in ("eager", "grouped_mm"):
        _, _, _, model = _fresh(args, experts_impl=impl)
        rels.append(_logits_of(args, model, args.seq))
    d = (rels[1] - rels[0]).abs().max() / rels[0].abs().max()
    print("# grouped_mm vs eager experts, same weights, same tokens")
    print(f"logits max_rel\t{float(d):.3e}")


def indexer_identity(args):
    """**The gate on this anchor's one transcription of reference arithmetic.**

    `qwen_anchor_taps.indexer_variant_forward` transcribes `QSAIndexer.forward` so that three
    defect rows can substitute one line each. A transcription that drifts from a future revision
    of the reference would move the goldens for a reason that has nothing to do with any defect,
    and nothing else in this harness could tell. So the transcription at `variant=None` is run
    against the reference in one process, on identical weights, and any difference at all is a
    failure -- not a tolerance.
    """
    ref = _logits_of(args, _fresh(args)[3], args.seq)
    got = _logits_of(args, _fresh(args, variant="identity")[3], args.seq)
    same = bool(torch.equal(ref, got))
    print(f"# indexer transcription vs reference, variant=identity\nbit_identical\t{same}")
    if not same:
        raise SystemExit(
            "the transcribed indexer is NOT bit-identical to the reference at variant=None "
            f"(max abs {float((ref - got).abs().max()):.3e}). The copy has drifted; fix it "
            "before believing any indexer defect row."
        )


# The window the real-parameter hash is pinned over. Two `eos` ids inside it, because
# `_shift_right_ignore_eos` resets the n-gram context at every EOS (MOD:1053-1067) and a window
# without one would leave that branch unscored -- and the history is padded with the config
# scalar 248044, not with `generation_config.json`'s list.
NGRAM_WINDOW = [1, 2, 3, 248044, 5, 6, 7, 248044, 9, 10, 11, 12]


def ngram_real(args):
    """**The real-parameter micro-anchor for the hashed n-gram embedding.**

    The tiny config shrinks `ngram_vocab_size_base` to 1024, which keeps the RULE but not the
    20M space -- so the row ids the real checkpoint's 128 shards are indexed by are outside
    everything the tiny anchor can see. This mode runs the reference's own
    `Qwen4ExpTextNGramEmbedding` at the REAL `vocab_size` 248320, base 20,000,000, `seed` 1234,
    `ngram_size` 3, `heads_per_ngram` 8 and `ple_layer_index` 0, and records the three
    multipliers, the 16 prime head vocabularies, the 16 offsets and the hashed ids of a pinned
    12-token window.

    Those first three are the buffers `qwen-architecture.md` section 5 byte-compared against the
    checkpoint's own I64 tensors, so this golden is what turns that comparison into a gate that
    S4's Rust re-derivation is scored against -- and a wrong `global_head_idx` cannot pass it,
    because the multipliers do not depend on `head_idx` at all while the primes do.

    One deviation, and it is the only way to run this at all: `embedding_dim` is 16 rather than
    2560, because the real table is 320,001,536 rows and `head_dim_per_ngram` affects the
    embedding WIDTH and nothing about the ids. 16 keeps `ple_embed_dim % ngram_heads == 0` and
    costs 1.3 GB instead of 205 GB. The ids are unchanged by construction; the embedding VALUES
    are not part of this fixture.
    """
    from transformers.models.qwen4_exp import configuration_qwen4_exp as cfg_mod
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mdl_mod

    real = json.loads(pathlib.Path(args.config).read_text())["text_config"]
    cls = cfg_mod.Qwen4ExpTextConfig
    cfg = cls(
        vocab_size=real["vocab_size"],
        ngram_size=real["ngram_size"],
        heads_per_ngram=real["heads_per_ngram"],
        ngram_vocab_size_base=real["ngram_vocab_size_base"],
        make_ngram_vocab_size_divisible_by=real["make_ngram_vocab_size_divisible_by"],
        eos_token_id=real["eos_token_id"],
        ple_embed_dim=16,
    )
    emb = mdl_mod.Qwen4ExpTextNGramEmbedding(cfg, 16, layer_idx=1, ple_layer_index=0)
    ids = torch.tensor([NGRAM_WINDOW])
    cap = Capture()
    emb.ngram_embedding.register_forward_hook(lambda _m, i, _o: cap.add("ngram_ids", i[0]))
    with torch.no_grad():
        emb(ids, None)
    cap.add("layer_multipliers", emb.layer_multipliers)
    cap.add("ngram_heads_vocab_sizes", emb.ngram_heads_vocab_sizes)
    cap.add("ngram_heads_offsets", emb.ngram_heads_offsets)
    cap.add("window", ids)
    cap.add("totals", torch.tensor([emb.total_vocab_size, emb.ngram_embedding.weight.shape[0]]))
    meta = [
        ("defect", args.defect),
        ("mode", args.mode),
        ("salt", args.salt),
        ("seq", str(len(NGRAM_WINDOW))),
        ("dtype", "int64"),
        ("torch", torch.__version__),
        ("python", ".".join(str(v) for v in sys.version_info[:3])),
        ("real_config_sha256_16", sha_of(args.config)),
        ("ref_modeling_sha256_16", sha_of(_reference_paths()[0])),
        ("ple_layer_index", "0"),
        # Present here too, so `qwen_anchor.rs` can assert the key on EVERY vendored golden
        # rather than on a subset -- a per-file exception is how a check's examined-count
        # silently reaches zero. This one is empty: nothing in this mode is called twice.
        ("repeated_captures", json.dumps(cap.repeated(), sort_keys=True)),
        ("embedding_dim_deviation", "16 instead of 2560; ids do not depend on it"),
        ("real_parameters", json.dumps({k: cfg.to_dict()[k] for k in
                                       ("vocab_size", "ngram_size", "heads_per_ngram",
                                        "ngram_vocab_size_base",
                                        "make_ngram_vocab_size_divisible_by", "seed")},
                                      sort_keys=True)),
    ]
    n = write_golden(args.out, meta, cap)
    print(f"qwen-anchor: {args.out} -- {len(cap.ints)} int tensors, {n} bytes, mode=ngram-real")


# ---------------------------------------------------------------------------------------------


def tolerance_table(directory):
    """**Derive the per-operator tolerance rows from the goldens already on disk.**

    A tolerance is a property of an OPERATOR and needs two measurements: the fp32 run's own
    rounding floor (`None` against the same run at `--dtype float64`) and the WEAKEST signal among
    the defect rows that TARGET that operator. Both are taken as the worst case over every weight
    draw found, because a floor measured at one draw is not a floor -- Glimmer's `attend` bucket
    came out 2.1x apart at two draws and the smaller would have placed the threshold at half what
    a correct kernel can need.

    "Targets" is not a judgement call here: `EXPECT_FIRST_TOUCH` already declares each row's
    first-touched bucket, and its operator IS the operator the row prices. A defect that reaches
    an operator by leaking downstream is not what that operator's tolerance is for, which is the
    rule `common/tolerance.rs` states.

    Runs deviceless over vendored or scratch bytes; no torch.
    """
    root = pathlib.Path(directory)
    salts = sorted({p.name.split("-None.bin")[0][len("gold-decode-"):] for p in root.glob("gold-decode-*-None.bin")})
    if not salts:
        raise SystemExit(f"no gold-decode-<salt>-None.bin under {root}")
    floors, weakest = {}, {}
    for salt in salts:
        base = root / f"gold-decode-{salt}-None.bin"
        fp64 = root / f"gold-decode-{salt}-fp64.bin"
        if fp64.exists():
            for op, rel in operator_scores(base, fp64).items():
                floors[op] = max(floors.get(op, 0.0), rel)
        for defect, bucket in EXPECT_FIRST_TOUCH.items():
            other = root / f"gold-decode-{salt}-{defect}.bin"
            if not other.exists():
                continue
            op = bucket[1]
            rel = operator_scores(base, other).get(op, 0.0)
            key = (op, defect)
            weakest[key] = min(weakest.get(key, float("inf")), rel)
    per_op = {}
    for (op, defect), rel in weakest.items():
        if rel < per_op.get(op, (float("inf"), ""))[0]:
            per_op[op] = (rel, defect)
    print(f"# salts {','.join(salts)}; floors from fp32-vs-fp64, weakest from the rows that target each operator")
    print("operator	floor	weakest_defect	margin	sets_the_row")
    for op in sorted(per_op):
        floor = floors.get(op, 0.0)
        rel, defect = per_op[op]
        margin = rel / floor if floor else float("inf")
        print(f"{op}	{floor:.4e}	{rel:.4e}	{margin:.1f}	{defect}")


def _parse_args():
    """The command line, and the one cross-argument rule argparse cannot state.

    `--config` and `--out` are required for a generation run and meaningless for a scoring one,
    so they are checked here rather than declared `required=True` -- which would break the
    deviceless `--compare` this whole split exists to keep.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "OTHER"),
                    help="score two goldens, assert the defect's first-touched bucket, exit; no torch")
    ap.add_argument("--by-operator", nargs=2, metavar=("BASE", "OTHER"),
                    help="score two goldens per OPERATOR instead of per bucket, exit; no torch")
    ap.add_argument("--tolerance-table", metavar="DIR",
                    help="derive the per-operator floor/weakest-defect rows from a matrix run; no torch")
    ap.add_argument("--config", help="the vendored real config.json")
    ap.add_argument("--out")
    ap.add_argument("--defect", default="None", choices=sorted(DEFECTS))
    ap.add_argument("--mode", default="decode",
                    choices=("prefill", "decode", "gdn-equiv", "moe-equiv", "indexer-identity", "ngram-real"))
    ap.add_argument("--seq", type=int, default=SEQ, help="prefill length")
    ap.add_argument("--salt", default="qwen-anchor-1", help="weight-init salt; part of the record")
    # Narrowable, and only the prefill window uses it. That window exists for ONE claim -- the
    # dense-versus-selective boundary across queries -- and capturing all seven layers at 16
    # positions costs 2.7 MB of vendored bytes for a fact that lives in one layer's indexer.
    ap.add_argument("--capture-layers", default=",".join(str(i) for i in CAPTURE_LAYERS),
                    dest="capture_layers", help="comma-separated layer indices to hook")
    # fp64 is not a mode anyone ships; it is how the TOLERANCE floor is measured. Running the
    # same reference at double precision and diffing gives the fp32 run's own rounding error,
    # which is the bound an independent correct kernel cannot beat.
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    # The experts selector, exposed because it is a DEVIATION and `--mode moe-equiv` measures it.
    ap.add_argument("--experts-impl", default="eager", choices=("eager", "grouped_mm"), dest="experts_impl")
    args = ap.parse_args()
    args.capture_layers_parsed = tuple(int(x) for x in args.capture_layers.split(",") if x != "")
    if args.compare or args.by_operator or args.tolerance_table:
        return args
    for req in ("config",) + (() if args.mode in ("gdn-equiv", "moe-equiv", "indexer-identity") else ("out",)):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required for --mode {args.mode}")
    return args


_MEASUREMENT_MODES = {
    "gdn-equiv": gdn_equiv,
    "moe-equiv": moe_equiv,
    "indexer-identity": indexer_identity,
    "ngram-real": ngram_real,
}


def _generate(args):
    """One generation run: build, perturb, tap, capture, write."""
    from transformers.models.qwen4_exp import configuration_qwen4_exp as cfg_mod
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mdl_mod

    assert_class_coverage()
    cfg = build_config(cfg_mod, args.config, experts_impl=args.experts_impl)
    model = _build_model(mdl_mod, cfg, args)
    ctx = _new_ctx(cfg, model, args)

    # Defects BEFORE the taps, so a defect that substitutes a free function is what the tap
    # wraps -- and before any forward pass, so a flag-only defect perturbs the warm prefill too.
    DEFECTS[args.defect](model, mdl_mod, ctx)
    patch = ctx.pop("patch_torch", None)
    if patch is not None:
        setattr(torch, patch[0], patch[1])
    tap = Tap(Capture(), ctx, args.capture_layers_parsed)
    _install(mdl_mod, tap, ctx)

    expected = _capture_pass(model, tap, args, cfg.vocab_size)
    assert_captured(tap, expected)
    model.config.__dict__["_qwen_anchor_gdn_tags"] = sorted(ctx["gdn_tags"])
    model.config.__dict__["_qwen_anchor_conv_tags"] = sorted(ctx["conv_tags"])
    cap = tap.cap
    n = write_golden(args.out, _metadata(args, model, cfg, cap), cap)
    print(
        f"qwen-anchor: {args.out} -- {len(cap.floats)} float, {len(cap.ints)} int tensors, "
        f"{n} bytes, defect={args.defect} mode={args.mode} dtype={args.dtype}"
    )


def main():
    args = _parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if args.by_operator:
        by_operator(*args.by_operator)
        return
    if args.tolerance_table:
        tolerance_table(args.tolerance_table)
        return

    global torch
    import torch

    # Before any weight is drawn: a wrong venv, or one with fla installed, should cost seconds.
    preflight_env()
    runner = _MEASUREMENT_MODES.get(args.mode)
    if runner is not None:
        runner(args)
        return
    _generate(args)


if __name__ == "__main__":
    main()
