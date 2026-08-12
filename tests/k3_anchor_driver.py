#!/usr/bin/env python3
"""**The S1b anchor: goldens produced by Kimi-K3's own first-party stack, not by ours.**

`docs/investigations/k3-port.md` G1b calls this MANDATORY, and the reason is in G0 item 11: four
of the twelve traps are **not attestable from the checkpoint**, because the KDA inner arithmetic
delegates to `fla-core` and the MXFP4 unpack to `compressed-tensors`. Everything rivoli knows
about those came from reading a third-party C reference. A fixture derived from that reading
would put the same misreading in the spec, in the golden, and in the kernel checked against it —
so the golden has to come from *running* the reference, and nothing here re-implements K3.

What runs is `modeling_kimi_linear.py` + `configuration_kimi_k3.py` at the pinned revision, over
a **tiny config derived from the real vendored `config.json`** — every structural field is the
real one (93 layers, the real 1-based KDA/MLA layer lists, `first_k_dense_replace`,
`attn_res_block_size` 12, `num_shared_experts` 2, both `situ` betas, `gate_lower_bound`), and only
*widths* shrink. Depth is free and structure is what the traps live in; width is what costs.

**Four declared deviations.** Each is recorded in the golden's own metadata, because a golden
that hides what produced it is worse than no golden:

  * `_attn_implementation` is forced to `eager` AFTER construction. `KimiLinearModel.__init__`
    overwrites whatever you pass with `flash_attention_2` (it logs "Ignoring the provided
    attention implementation"), and `KimiMLAAttention.forward` reads the field at call time, so
    the override lands. Eager is the semantics flash approximates, needs no GPU-only kernel, and
    skips only the pad-then-slice of `value_states` — which is numerically neutral.
  * The routed experts are **plain `nn.Linear`** (`quantized=no`), because `quantization_config`
    is dropped from the tiny config. The MXFP4 unpack is anchored separately and on real bytes by
    `docs/measurement/k3-reference/repack-one-expert.md`; nothing here would add to it, and a
    group-32 scale grid does not exist at these widths.
  * `KimiLinearForCausalLM` on the text config, not `KimiK3ForConditionalGeneration`. rivoli
    refuses vision, and the wrapper contributes no text-side arithmetic. Recorded as
    `entry_point`, because nothing else in the file would distinguish the two.
  * **The model runs in torch's default dtype, fp32, while the checkpoint is bf16.** Right for a
    reference — the point is to pin arithmetic, not to reproduce one accumulation order — but it
    is the number S2's tolerance decision starts from, so `dtype` is in the metadata. (Added
    2026-08-11 after review: three deviations were declared and this fourth was not.)

**KDA still needs a GPU**, and that is not a choice: `chunk_kda`/`fused_recurrent_kda` are triton
kernels with no CPU path (`triton.runtime.driver`: "0 active drivers"), and `fla.ops.kda.naive`'s
pure-torch twin takes none of the seven kwargs the model passes — `A_log`, `dt_bias`, the qk
l2-norm, the beta sigmoid and the gate lower bound all moved *inside* the kernel. Substituting it
would mean transliterating those by hand, which is the exact thing this anchor exists not to do.
So goldens are GENERATED on a GPU once and VENDORED; the gate that reads them needs no device.

Usage (`--ref` is the downloaded pinned reference, `--config` the vendored real config):

    python3 tests/k3_anchor_driver.py --ref <dir> --config docs/measurement/k3-reference/config.json \\
        --out golden.bin --mode decode --defect None

`--defect` perturbs the reference to prove a golden can go red — see `DEFECTS`. `--compare A B`
scores two goldens against each other and **asserts the defect's declared green layers**, which
is the half of `k3-port.md` §G rule 1 that a bare "something changed" check leaves out.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import pathlib
import struct
import sys

# Bound by `main` so `--compare` runs with no torch, no fla and no GPU — the property that lets a
# defect run be re-scored from vendored bytes on any machine.
torch = None

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
# Deterministic weights.


def _gen(name, salt):
    """A generator seeded by the parameter's NAME, not by its position in the module tree.

    One global seed would make every golden depend on construction order, so adding a capture or
    reordering a module would move numbers that nothing about the model changed. Keyed by name,
    a parameter's values are a property of its name alone.
    """
    h = hashlib.sha256(f"{salt}/{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def init_weights(model, salt):
    """Fill every parameter deterministically, in three families.

    Norm weights sit near 1 and everything else near 0, because that is what the real
    initialisers produce and a norm weight drawn near zero would make every downstream activation
    a denormal — a golden of numerical noise agrees with nothing.

    `A_log` and `dt_bias` are drawn on the reference's own scales: `A_log` is
    `log(uniform(1, 16))` in `KimiDeltaAttention.__init__`, and it is the decay rate, so a wrong
    scale would either freeze or erase the recurrent state and hide whatever the golden is for.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = _gen(name, salt)
            flat = torch.empty(p.numel(), dtype=torch.float32)
            # The OWNING module's name, because every norm here is a `<something>norm.weight` and
            # testing the leaf would only ever see "weight".
            owner = name.split(".")[-2]
            if name.endswith("A_log"):
                flat.uniform_(1.0, 16.0, generator=g).log_()
            elif name.endswith("dt_bias"):
                flat.uniform_(-4.0, 1.0, generator=g)
            elif "norm" in owner:
                flat.uniform_(0.8, 1.2, generator=g)
            else:
                flat.uniform_(-0.08, 0.08, generator=g)
            p.copy_(flat.view(p.shape).to(p.dtype).to(p.device))
        # `post_init` zeroed the `padding_idx` row and the loop above just overwrote it. Put it
        # back: the zeroing is the reference's own behaviour. Numerically inert here, since no
        # input id is ever `pad` -- which is exactly why review had to find this rather than a
        # test (2026-08-11); the comment in `TINY_TEXT` claimed the zero row was kept and it was
        # not.
        emb = model.get_input_embeddings()
        if emb.padding_idx is not None:
            emb.weight[emb.padding_idx].zero_()


# ---------------------------------------------------------------------------------------------
# Defects. Each one is a perturbation of the REFERENCE, applied after construction, whose whole
# purpose is to prove the goldens it should redden do redden and the rest stay green. A golden
# with no such run is not evidence -- it is a file.


def _mla_blocks(model):
    return [layer.self_attn for layer in model.model.layers if not layer.is_linear_attn]


def _moe_blocks(model):
    return [layer.block_sparse_moe for layer in model.model.layers if hasattr(layer, "block_sparse_moe")]


def defect_mla_lora_eps(model, ctx):
    """The divergence G0 item 11 found in the C reference: MLA's LoRA norms at eps 1e-5.

    First-party is 1e-6 -- `KimiRMSNorm`'s own default, taken because MLA constructs those two
    norms WITHOUT passing `config.rms_norm_eps` (which is 1e-5). A fixture transliterated from the
    C would have baked 1e-5 in, and this is what that would have cost.
    """
    for a in _mla_blocks(model):
        a.q_a_layernorm.variance_epsilon = 1e-5
        a.kv_a_layernorm.variance_epsilon = 1e-5


def defect_mla_scale_from_nope(model, ctx):
    """Softmax scale over `qk_nope_head_dim` instead of `q_head_dim` -- trap 8.

    The reference is `self.scaling = self.q_head_dim ** (-0.5)`, and `q_head_dim` counts the 64
    rope dims that NoPE never rotates but still scores. Taking the scale over the nope width
    alone is the natural reading of "the dims that carry information" and it is wrong.
    """
    for a in _mla_blocks(model):
        a.scaling = a.qk_nope_head_dim ** (-0.5)


def defect_expert_w1_w3_swap(model, ctx):
    """gate and up swapped in every routed expert -- the one repack error that is byte-clean.

    `V4_PROJ`'s doc says only a numerical oracle can see it, and `repack-one-expert.md` could pin
    only that `w2` is not in the up slot. This is that oracle.
    """
    with torch.no_grad():
        for b in _moe_blocks(model):
            for e in b.experts:
                w1 = e.w1.weight.clone()
                e.w1.weight.copy_(e.w3.weight)
                e.w3.weight.copy_(w1)


def defect_dense_mlp_gate_up_swap(model, ctx):
    """The same swap in the ONE dense layer, so a defect exists that touches layer 0 alone."""
    with torch.no_grad():
        mlp = model.model.layers[0].mlp
        g = mlp.gate_proj.weight.clone()
        mlp.gate_proj.weight.copy_(mlp.up_proj.weight)
        mlp.up_proj.weight.copy_(g)


def defect_router_bias_in_weight(model, ctx):
    """Take the routing WEIGHT from the biased score instead of the unbiased one.

    The reference selects on `scores + e_score_correction_bias` and then gathers from `scores`.
    Using the biased value for both is the natural way to write it and is wrong; it is trap 6.

    This is the one defect that restates reference arithmetic (the sigmoid, the `+1e-20`
    renormalisation, the `routed_scaling_factor`), against this file's own rule that nothing here
    re-implements K3. It is unavoidable: the biased weight is not recoverable from the reference's
    renormalised output, so there is nothing to reuse. The cost is real and worth stating -- if
    the reference's weight path ever changes, this copy reddens the goldens as though it were
    trap 6.
    """
    import torch.nn.functional as F

    for b in _moe_blocks(model):
        gate = b.gate
        orig = gate.forward

        def patched(hidden_states, _gate=gate, _orig=orig):
            idx, _ = _orig(hidden_states)
            h = hidden_states.view(-1, hidden_states.shape[-1])
            logits = F.linear(h.float(), _gate.weight.float(), None)
            biased = logits.sigmoid() + _gate.e_score_correction_bias.unsqueeze(0).float()
            w = biased.gather(1, idx)
            if _gate.moe_renormalize:
                w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
            return idx, w * _gate.routed_scaling_factor

        gate.forward = patched


def defect_latent_norm_after_up(model, ctx):
    """RMSNorm the latent sandwich's output AFTER the up projection instead of before it.

    The reference norms the AGGREGATE at latent width and then projects up. Norming after is the
    ordering trap, and it is shape-valid in both directions only because a norm is width-generic.

    Done by swapping the two modules' `forward`s rather than by replacing them, because the
    golden's tensor SET must not depend on the defect: a `Sequential` here would introduce
    `routed_expert_up_proj.0`/`.1` as new named submodules, the layer prefix would hook them, and
    `--compare` would abort on a set mismatch instead of scoring the defect. The norm's weight
    collapses to its own mean so it can be applied at `hidden` width -- which keeps the
    perturbation about the ORDER rather than about the values.
    """
    for b in _moe_blocks(model):
        norm, up = b.routed_expert_norm, b.routed_expert_up_proj
        w, eps, orig_up = norm.weight.detach().mean(), norm.variance_epsilon, up.forward
        norm.forward = lambda y: y

        def patched(y, _up=orig_up, _w=w, _eps=eps):
            z = _up(y).float()
            return (_w * z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + _eps)).to(y.dtype)

        up.forward = patched


def defect_attn_res_normalised_values(model, ctx):
    """Mix the NORMALISED sources instead of the raw ones in the AttnRes fold.

    `_apply_attn_res` normalises `v` only to score it, then mixes `v_float` -- the raw
    concatenation. Mixing `k` instead is a one-character slip that leaves every shape intact.
    Reaches all three folds (twice per layer, once model-level) because it replaces the free
    function they all resolve at call time.
    """
    ctx["attn_res_mix_normalised"] = True


def defect_kda_no_qk_l2norm(model, ctx):
    """Drop `use_qk_l2norm_in_kernel`, the normalisation that only exists inside fla's kernel."""
    ctx["kda_kwargs"]["use_qk_l2norm_in_kernel"] = False


def defect_kda_gate_lower_bound_off(model, ctx):
    """Drop `lower_bound` (-5.0 from the real config) AND `safe_gate` -- trap 4.

    The bound MULTIPLIES the gate's sigmoid rather than clamping it, and fla's own docstring says
    so — `fla/ops/kda/chunk.py:250-256` writes out both forms. An earlier version of this comment
    claimed nothing outside the kernel attested to it; that was wrong, and `anchor.md` carries the
    correction. **S2 should port the term from that docstring**, not infer its shape from a red
    cell. What this run buys is the other thing: proof the golden is SENSITIVE to the term, since a
    docstring can be stale in a way bytes cannot, and `gate_lower_bound` is inherited from the real
    config into the tiny one.

    THE TWO KWARGS MOVE TOGETHER BECAUSE FLA REFUSES TO SEPARATE THEM. `chunk.py:394` raises
    unless `lower_bound` is set whenever `safe_gate=True and use_gate_in_kernel`, so no run drops
    only the bound. This cell attests to the PAIR: a kernel with the right bound and the wrong
    clamp is not distinguished by it.
    """
    ctx["kda_kwargs"]["lower_bound"] = None
    ctx["kda_kwargs"]["safe_gate"] = False


def defect_kda_state_layout(model, ctx):
    """Store the recurrent state with its last two axes swapped -- the (K,V)-vs-(V,K) order.

    Invisible to any shape assertion: `head_k_dim == head_dim` in the tiny model AND in the real
    one (128 == 128), so the state is square either way. Only the state's VALUES can carry the
    layout, and only this run proves they do.

    **Done at the boundary rather than by flipping `transpose_state_layout`**, which is the kwarg
    that names this choice. That was the first implementation and it was abandoned after measuring
    it: with `transpose_state_layout=False`, fla went into triton for **25 minutes without
    finishing**, against ~30 s for every other defect, writing new cache entries the whole time.
    It is asking for a kernel variant nobody will ship, and a defect run that costs half an hour
    is a defect run nobody re-runs. Transposing the returned state asks the same question -- does
    the state's axis order reach the next token -- of the kernel that is actually used.
    """
    ctx["kda_transpose_out_state"] = True


def defect_kda_beta_sigmoid_outside(model, ctx):
    """Drop `use_beta_sigmoid_in_kernel`, so the kernel takes `beta` raw.

    `beta` is captured PRE-sigmoid (`b_proj(...).float()`), so whether the sigmoid happens inside
    the kernel is unattested — and it is a decision rivoli's own recurrence has to make.
    """
    ctx["kda_kwargs"]["use_beta_sigmoid_in_kernel"] = False


DEFECTS = {
    "None": lambda model, ctx: None,
    "MlaLoraEps1e5": defect_mla_lora_eps,
    "MlaScaleFromNope": defect_mla_scale_from_nope,
    "ExpertW1W3Swap": defect_expert_w1_w3_swap,
    "DenseMlpGateUpSwap": defect_dense_mlp_gate_up_swap,
    "RouterBiasInWeight": defect_router_bias_in_weight,
    "LatentNormAfterUp": defect_latent_norm_after_up,
    "AttnResNormalisedValues": defect_attn_res_normalised_values,
    "KdaNoQkL2Norm": defect_kda_no_qk_l2norm,
    "KdaGateLowerBoundOff": defect_kda_gate_lower_bound_off,
    "KdaStateLayout": defect_kda_state_layout,
    "KdaBetaSigmoidOutside": defect_kda_beta_sigmoid_outside,
}

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


# ---------------------------------------------------------------------------------------------
# Capture.


class Capture:
    """Named tensors, split by element type the way `src/v4oracle/golden.rs` splits them."""

    def __init__(self):
        self.floats = []
        self.ints = []
        self.seen = set()
        # Module names whose hook actually ran, recorded by the hook itself rather than inferred
        # from tensor names. Inferring would be prefix-shadowed: `model.layers.92`'s own hook could
        # look like it fired because `model.layers.92.self_attn` did.
        self.fired = set()

    def add(self, name, t):
        if not isinstance(t, torch.Tensor):
            return
        t = t.detach().to("cpu")
        shape = list(t.shape)
        self.seen.add(name)
        if t.dtype in (torch.int64, torch.int32):
            self.ints.append((name, shape, t.reshape(-1).to(torch.int64).tolist()))
        else:
            self.floats.append((name, shape, t.reshape(-1).to(torch.float32).tolist()))

    def add_any(self, name, out):
        """Flatten a module's output, whatever shape of container it came in."""
        if isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                self.add_any(f"{name}.{i}", o)
        else:
            self.add(name, out)


def _fire(cap, name, out):
    """Record that `name`'s hook ran, then capture its output."""
    cap.fired.add(name)
    cap.add_any(name, out)


def hook_model(model, cap, layers):
    """A forward hook on every submodule of the captured layers, plus the model-level tail.

    Every submodule rather than a chosen few: at these widths the whole set is a few hundred
    kilobytes, and a hand-picked list is a list someone has to remember to extend when a module
    appears.

    Returns `(handles, expected_names)`. The second half exists because a hook that never fires
    is silent, and that silence cost this harness its AttnRes coverage for a day — see
    `wrap_attn_res`.
    """
    # The trailing dot is load-bearing, and its absence was measured: an earlier version also
    # matched the bare `model.layers.{i}`, so `model.layers.1` caught layers 10-19 and
    # `model.layers.3` caught 30-39 -- 25 captured layers instead of 6. The layer's own output is
    # reached by the exact-name test below instead.
    prefixes = tuple(f"model.layers.{i}." for i in layers)
    exact = tuple(f"model.layers.{i}" for i in layers)
    tail = ("model.norm", "lm_head")
    handles, expected = [], []
    for name, mod in model.named_modules():
        wanted = name in tail or name in exact or name.startswith(prefixes)
        # Everything at or under `.experts` is excluded, and the reason is not size. `moe_infer`
        # calls only the experts that WON tokens, so which expert modules fire is
        # routing-dependent -- any defect that moves the routing changes the golden's tensor SET,
        # and every such comparison then scores "absent on one side" rather than a number.
        # Measured: the first defect matrix reported `inf` for most layers on four of five defects
        # for exactly this reason, drowning the real signal. `topk_idx`/`topk_weight` and the block
        # output are the routing fixture and are always present.
        #
        # `.experts` and not `.experts.`, so the `ModuleList` ITSELF goes too: its forward is never
        # called, so its hook could never fire, and the silent-hook assertion at the end of `main`
        # caught all five of them on its first run. `shared_experts` is untouched -- the substring
        # needs a dot before `experts`, and there the preceding character is an underscore.
        if not wanted or ".experts" in name:
            continue
        # The four AttnRes modules per layer are the OTHER never-fires case: `_apply_attn_res`
        # reads their `.weight` directly instead of calling them. `wrap_attn_res` captures the
        # fold; hooking them here would register four dead hooks per layer.
        if name.endswith(("_res_norm", "_res_proj")):
            continue
        handles.append(mod.register_forward_hook(lambda m, i, o, n=name: _fire(cap, n, o)))
        expected.append(name)
    return handles, expected


def wrap_attn_res(mod, model, cap, ctx, layers):
    """Capture the AttnRes fold, which no forward hook can see.

    **`_apply_attn_res` never calls its `proj` and `norm` modules** -- it reads
    `proj.weight.squeeze(0)`, `norm.weight` and `norm.variance_epsilon` inline (reference
    `:1075-1088`). A `register_forward_hook` only fires from `Module.__call__`, so the six
    AttnRes modules per layer and the two model-level ones produced NOTHING, and the golden
    contained no `*_res_*` tensor at all while three comments said otherwise. Found by review
    2026-08-11 -- and AttnRes is S2 item 1, so the anchor was missing a fixture for the first
    kernel the port writes.

    It is a module-level free function, and all three call sites (twice per layer, once
    model-level) resolve it from module globals at call time, so one `setattr` catches every
    fold. Folds are named by the identity of the `proj` they were handed, which is what
    distinguishes `self_attention_res` from `mlp_res` without depending on call order.
    """
    owners = {}
    for name, m in model.named_modules():
        if not name.endswith("_res_proj"):
            continue
        base = name[: -len("_proj")]
        if base.startswith("model.output") or any(
            base.startswith(f"model.layers.{i}.") for i in layers
        ):
            owners[id(m)] = base
    orig = mod._apply_attn_res

    def inner(prefix_sum, block_residual, proj, norm):
        if ctx.get("attn_res_mix_normalised"):
            out = _fold_mixing_normalised(prefix_sum, block_residual, proj, norm)
        else:
            out = orig(prefix_sum, block_residual, proj, norm)
        tag = owners.get(id(proj))
        if tag is not None and ctx["capturing"]:
            cap.add(f"{tag}.in.prefix_sum", prefix_sum)
            cap.add(f"{tag}.in.block_residual", block_residual)
            # **The fold, without which the other three are unscoreable.** The scoring vector is
            # the only *weight* this harness captures, and it is captured because AttnRes is the
            # one operator whose inputs alone do not determine its output: `out` depends on
            # `softmax(<RMSNorm(v), norm.weight * proj.weight>)`, and nothing in the golden
            # carried those two. S2 item 1 went to write the kernel, found inputs and an output
            # and no way to get from one to the other, and came back here (2026-08-11).
            #
            # The PRODUCT rather than the two factors: `fold[i] = norm[i] * proj[i]` is a
            # load-time collapse the port does in the loader (`k3-architecture.md` §3), so the
            # kernel never sees the factors and a fixture that carried them would be scoring the
            # elementwise multiply. `.float()` because the checkpoint ships these BF16 while the
            # reference holds them fp32 — this records what the arithmetic used.
            cap.add(f"{tag}.fold", norm.weight.float() * proj.weight.squeeze(0).float())
            cap.add(f"{tag}.out", out)
        return out

    mod._apply_attn_res = inner
    return orig


def _fold_mixing_normalised(prefix_sum, block_residual, proj, norm):
    """`_apply_attn_res` with `k` mixed instead of `v_float` -- the `AttnResNormalisedValues` body.

    A transcription of the reference's five lines with one substitution, which is the whole point
    of the defect; every other defect here perturbs a weight or a kwarg instead.
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    k = v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    scores = (k * (norm.weight.float() * proj.weight.squeeze(0).float())).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, k).squeeze(1).to(v.dtype)


def wrap_mla_attention(mod, model, cap, ctx, layers):
    """Capture the MLA attention CORE, which no forward hook can see.

    `KimiMLAAttention.forward` resolves `eager_attention_forward` from module globals at call time
    (`attention_interface: Callable = eager_attention_forward`) and the driver forces `eager`, so a
    single module-level setattr catches every MLA layer -- the same mechanism `wrap_attn_res` uses,
    and for the same reason: a free function is not a `Module`, so `register_forward_hook` never
    fires for it.

    **Without this, S2 item 2's three worst traps have no fixture at all.** The projections and
    norms around the attention are captured as module outputs, but the scores, the softmax and the
    weighted sum are not, so nothing in the golden could distinguish:

      * a softmax scale taken over `qk_nope` instead of the full 192 (`--defect MlaScaleFromNope`);
      * the unrotated 64 rope dims being DROPPED from the score rather than kept -- `k3-architecture.md`
        §5 calls this "the silent bug", and it is silent precisely because the projections either
        side of it are unchanged;
      * causality, which is unconditional here and lives entirely in `attention_mask`.

    `scaling` is captured as a one-element tensor rather than left in metadata: it is the trap, so
    it should be a VALUE a fixture reads, not prose a reader trusts. `probs` is captured too --
    the softmax is where a mask error shows up as a distribution rather than as a wrong number,
    and a port that attends across the causal boundary produces a plausible output and an obviously
    wrong row of probabilities.

    Named by the owning module's position in the tree, resolved through `module.layer_idx`, so the
    tag matches the rest of the layer's captures without depending on call order.
    """
    orig = mod.eager_attention_forward
    wanted = set(layers)

    def inner(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, probs = orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
        idx = getattr(module, "layer_idx", None)
        if idx in wanted and ctx["capturing"]:
            tag = f"model.layers.{idx}.self_attn.attend"
            cap.add(f"{tag}.in.q", query)
            cap.add(f"{tag}.in.k", key)
            cap.add(f"{tag}.in.v", value)
            if attention_mask is not None:
                cap.add(f"{tag}.in.mask", attention_mask[:, :, :, : key.shape[-2]])
            cap.add(f"{tag}.in.scaling", torch.tensor([scaling], dtype=torch.float32))
            cap.add(f"{tag}.out", out)
            cap.add(f"{tag}.probs", probs)
        return out, probs

    mod.eager_attention_forward = inner

    # **The gated value entering `o_proj`, by PRE-hook.** Trap 10 is an ORDER -- MLA gates with no
    # norm, KDA norms and then gates -- and an order is only attestable if both sides of it are in
    # the file. `attend.out` is the pre-gate value and `o_proj`'s OUTPUT is post-projection, so
    # without this the gate sits in a gap between two captures, and a port that normed before
    # gating, or skipped the gate entirely, would match every tensor here.
    #
    # A pre-hook rather than another free-function wrap, because `o_proj` IS a module and its input
    # is exactly the quantity wanted. `register_forward_hook` gives the output, which is already
    # captured and is a different claim.
    pre_hook_inputs(model, cap, ctx, layers, ".self_attn.o_proj", ".in_gated")
    return orig


def pre_hook_inputs(model, cap, ctx, layers, ends, suffix, params=()):
    """Capture the INPUT of every captured-layer module whose name ends with `ends`.

    A forward hook gives a module's output. Three of this port's operators are fed a value that
    no module produced -- the gated activation entering `o_proj`, the block's hidden state
    entering `routed_expert_down_proj`, and the expert aggregate entering `routed_expert_norm` --
    and for those the input IS the fixture. `params` additionally records named parameters of the
    module, for the same reason `wrap_attn_res` records the fold: a kernel fed an input and asked
    for an output cannot be scored without the weights in between.

    Returns the module names hooked, so the caller can assert a census rather than trust that a
    suffix matched anything. A hook that never fires is silent, and that silence cost this harness
    its AttnRes coverage for a day.
    """
    hooked = []
    for name, m in model.named_modules():
        if not name.endswith(ends) or not any(
            name.startswith(f"model.layers.{i}.") for i in layers
        ):
            continue

        def fire(mod_, inp, _n=name):
            if not ctx["capturing"]:
                return
            cap.add(f"{_n}{suffix}", inp[0])
            for p in params:
                cap.add(f"{_n}.{p}", getattr(mod_, p))

        m.register_forward_pre_hook(fire)
        hooked.append(name)
    return hooked


def wrap_latent_sandwich(model, cap, ctx, layers):
    """The MoE latent sandwich's two open ends -- S2 item 3.

    `k3-architecture.md` §6 is `down(x) -> experts in latent space -> RMSNorm the AGGREGATE ->
    up(...)`, and the golden held only the three module OUTPUTS. Two values a fixture needs were
    therefore absent:

    * **the aggregate**, `routed_expert_norm`'s input. It is `moe_infer`'s return, which is not a
      module call, and `.experts` is deliberately unhooked because which expert fires is
      routing-dependent. So the RMSNorm -- the one operator in this sandwich whose arithmetic is
      neither a plain matmul nor shared with another model -- had no input at all.
    * **the norm's weight**, drawn per-parameter from `sha256(salt/name)` and so not reproducible
      outside this file.

    The other two inputs are NOT captured, and in both cases because the file already holds them:
    `routed_expert_up_proj`'s input is `routed_expert_norm`'s output, and
    `routed_expert_down_proj`'s is `post_attention_layernorm`'s (reference `:964-966`, `:1035-1037`
    -- the block is called on the normed hidden state and on nothing else). A second copy under a
    second name would read as corroboration and be a tautology.

    **The two projection WEIGHTS are also not captured, and that is a decision rather than an
    oversight.** They are `[96,192]` and `[192,96]` here -- 36,864 floats per MoE layer against
    the whole golden's ~70,000 -- so capturing them at the five MoE layers would roughly quadruple
    both vendored files. What they would buy is an anchor-scored GEMV, and that comparison is weak
    where it is not free: rivoli's trunk matmul is `gemm_bf16`, whose weights are bf16, while this
    reference runs fp32 (one of the anchor's four declared deviations). A bf16 weight is ~2^-9
    relative off its fp32 twin -- 1.95e-3 against `moe_latent`'s 2.9e-4 tolerance -- so the
    fixture could only be stated at a tolerance seven times looser than the operator's, and it
    would still be at hidden 192 rather than 7168. `tests/k3_kernels.rs` scores `gemm_bf16`
    against an f64 host dot at K3's REAL widths instead, which is both cheaper and wider.
    """
    return pre_hook_inputs(
        model, cap, ctx, layers, ".routed_expert_norm", ".in", params=("weight",)
    )


def wrap_kda_ops(mod, cap, ctx, layers):
    """Record the KDA operator's own inputs and outputs, which no module hook can see.

    `q`/`k`/`v` after the short convolutions, the log-space gate, `beta`, `A_log`, `dt_bias` in;
    `o` and the recurrent state out. This IS the S2 KDA fixture: everything between these two
    points lives in fla's triton kernel and in no document.

    Installed BEFORE the warm prefill, and capture is gated on `ctx["capturing"]` instead.
    Review 2026-08-11: with the wrapper installed after the warm pass, a `--defect` that only
    sets a kernel kwarg perturbed exactly one call per layer, on a recurrent state the defect had
    never touched -- so `in.initial_state` was bit-identical between the runs BY CONSTRUCTION and
    the state-propagation claim was the one thing those defects could not test.
    """

    def wrap(fn, tag):
        def inner(**kw):
            kw = dict(kw, **ctx["kda_kwargs"])
            if ctx["kda_fp32_island"]:
                # **fla's KDA kernel cannot be compiled for fp64 at all** — it raises
                # `fp_downcast_rounding should be set only for truncating fp conversions` on an
                # internal fp32->fp64 conversion (measured 2026-08-11). So `--dtype float64` runs
                # the whole model in double EXCEPT this operator, which stays fp32 with its inputs
                # cast down and its outputs cast back. That makes the fp64 run a rounding-floor
                # reference for every operator but this one; KDA's own floor comes from
                # `--mode kda-equiv` instead, and would have needed a separate measurement anyway
                # because the kernel returns an fp32 state whatever the input dtype.
                # `q` by name, not "the first kwarg with a dtype". That took whatever the reference
                # happened to pass first -- today `q`, but a revision putting `cu_seqlens` or a
                # position index earlier would cast every KDA output to an INTEGER dtype, and the
                # only symptom would be a wrong floor in a measurement nobody re-derives.
                q = kw.get("q")
                assert q is not None and q.is_floating_point(), (
                    f"the fp32 island reads its dtype from `q`; got {type(q).__name__}. The "
                    f"reference calls this op with all-keyword arguments, so a rename upstream "
                    f"lands here rather than silently picking another kwarg's dtype."
                )
                dt = q.dtype
                kw = {
                    k: (v.float() if hasattr(v, "dtype") and v.is_floating_point() else v)
                    for k, v in kw.items()
                }
                out = fn(**kw)
                out = tuple(o.to(dt) if o is not None else None for o in out)
            else:
                out = fn(**kw)
            if ctx.get("kda_transpose_out_state") and out[1] is not None:
                out = (out[0], out[1].transpose(-1, -2).contiguous())
            sink = ctx.get("equiv_sink")
            if sink is not None:
                which_all = ctx["kda_layer_ids"][ctx["kda_calls"]]
                ctx["kda_calls"] += 1
                sink.setdefault((tag, which_all), []).append(out[0].detach().float().cpu())
                return out
            if ctx["capturing"]:
                # The nth KDA call of the capture pass is the nth KDA layer in index order, which
                # the model guarantees by iterating `self.layers` in order.
                which = ctx["kda_layer_ids"][ctx["kda_calls"]]
                ctx["kda_calls"] += 1
                if which in layers:
                    for k in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state"):
                        if kw.get(k) is not None:
                            cap.add(f"model.layers.{which}.kda.{tag}.in.{k}", kw[k])
                    cap.add(f"model.layers.{which}.kda.{tag}.out.o", out[0])
                    if out[1] is not None:
                        cap.add(f"model.layers.{which}.kda.{tag}.out.state", out[1])
            return out

        return inner

    for tag in ("chunk_kda", "fused_recurrent_kda"):
        setattr(mod, tag, wrap(getattr(mod, tag), tag))


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


# **Which operator each captured tensor belongs to.** Layer numbers alone cannot say: `self_attn`
# is MLA on 24 layers and KDA on 69, so the classifier has to read the partition out of the
# golden's own `tiny_config`. Used only by `--by-operator`, which is how the per-operator
# tolerances in `anchor.md` are derived — a tolerance is a property of an OPERATOR, and a per-layer
# report cannot state one.
def operator_of(name, kda_layers):
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer, rest = int(parts[2]), ".".join(parts[3:])
        kda = layer in kda_layers
        if rest.startswith("kda."):
            return "kda_op"
        if rest.startswith("self_attn.attend"):
            # **Its own bucket, split out 2026-08-11 for S2 item 2.** `mla` was measured before the
            # attention core was captured at all, and it is tabled ExactOnly on the strength of a
            # defect that lives in the LoRA NORMS -- the C reference's eps, 1.3x the bucket floor.
            # The core takes the reference's own q/k/v, so that eps cannot reach it, and a kernel
            # fixture cannot be bit-exact with torch in any case. Two different questions do not
            # share a threshold.
            return "mla_attend"
        if rest.startswith("self_attn"):
            return "kda_trunk" if kda else "mla"
        if rest.startswith(("self_attention_res", "mlp_res")):
            return "attn_res"
        if rest.startswith("block_sparse_moe.routed_expert"):
            return "moe_latent"
        if rest.startswith("block_sparse_moe"):
            return "moe_route"
        if rest.startswith("mlp"):
            return "dense_mlp"
        if "layernorm" in rest:
            return "norm"
        return "residual"
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


def by_operator(a_path, b_path):
    """Per-OPERATOR agreement between two goldens -- the shape a tolerance has to be stated in.

    Two uses, and the tolerance table needs both: `fp32 vs fp64` gives the floor an independent
    correct implementation cannot beat, and `None vs <defect>` gives the signal a tolerance has to
    stay under. `anchor.md`'s table is these two numbers per operator.
    """
    (ma, mb), sections = _both_sections(a_path, b_path)
    kda = _kda_zero_based(json.loads(ma["tiny_config"]))
    per = {}
    for items_a, items_b in sections:
        for k, v in _score(items_a, items_b, lambda n: operator_of(n, kda)).items():
            n, diff, rel = per.get(k, (0, 0, 0.0))
            per[k] = (n + v[0], diff + v[1], max(rel, v[2]))
    print(f"# {mb.get('defect')}/{mb.get('dtype')} vs {ma.get('defect')}/{ma.get('dtype')}")
    print("operator\ttensors\tdiffering\tmax_rel")
    for k in sorted(per):
        n, diff, rel = per[k]
        print(f"{k}\t{n}\t{diff}\t{rel:.3e}")


# Every fla module in the KDA path, by class name so this file need not import fla. All three are
# triton-backed and all three refuse fp64 — `ShortConvolution` and `FusedRMSNormGated` as much as
# the KDA ops themselves — which is why `--dtype float64` cannot simply be "the model in double".
FLA_MODULES = ("ShortConvolution", "FusedRMSNormGated")


def fp32_island(model):
    """Make every fla module compute in fp32 while the rest of the model runs in fp64.

    Measured 2026-08-11: a plain `model.double()` dies in triton with
    `fp_downcast_rounding should be set only for truncating fp conversions`, from inside
    `ShortConvolution` — so the island has to cover each fla boundary, not just the KDA op. What
    survives is a genuine fp64 reference for **AttnRes, MLA, the latent sandwich, SiTU/MoE, the
    norms and the head**: four of S2's five items. KDA's floor is `--mode kda-equiv`.
    """
    n = 0
    for m in model.modules():
        if type(m).__name__ not in FLA_MODULES:
            continue
        orig = m.forward

        def shim(*a, _orig=orig, **kw):
            down = lambda x: (  # noqa: E731
                x.float() if hasattr(x, "dtype") and x.is_floating_point() else x
            )
            out = _orig(*[down(x) for x in a], **{k: down(v) for k, v in kw.items()})
            if isinstance(out, tuple):
                return tuple(o.double() if hasattr(o, "dtype") and o.is_floating_point() else o for o in out)
            return out.double() if hasattr(out, "dtype") else out

        m.forward = shim
        n += 1
    return n


def kda_equiv(model, mdl_mod, ctx, seq, device, vocab):
    """**KDA's tolerance floor, measured against fla's own second implementation of it.**

    fp64 cannot serve here (see `wrap_kda_ops`), so the floor comes from the two paths fla ships
    for the same recurrence: `chunk_kda` over T positions at once, and `fused_recurrent_kda` one
    position at a time. They are the same mathematics by the same authors, so their disagreement is
    exactly what "a correct implementation in fp32, associating differently" costs — which is the
    number a HIP kernel's tolerance has to sit above.

    Reported over EVERY KDA layer, not just the captured six: a tolerance is a bound, and a bound
    wants the worst case.
    """
    chunk, steps = {}, {}
    with torch.no_grad():
        ctx["equiv_sink"], ctx["kda_calls"] = chunk, 0
        ids = torch.arange(1, seq + 1, device=device).unsqueeze(0) % vocab
        model(input_ids=ids, use_cache=True)
        ctx["equiv_sink"], ctx["kda_calls"] = steps, 0
        past = None
        for i in range(seq):
            one = torch.tensor([[(i + 1) % vocab]], device=device)
            out = model(input_ids=one, past_key_values=past, use_cache=True)
            past = out.past_key_values
            ctx["kda_calls"] = 0
    ctx["equiv_sink"] = None

    worst, rows = 0.0, []
    for (tag, layer), got in sorted(chunk.items(), key=lambda kv: kv[0][1]):
        assert tag == "chunk_kda", f"prefill went through {tag}"
        c = got[0]                                    # [1, T, H, D]
        r = steps.get(("fused_recurrent_kda", layer))
        assert r is not None and len(r) == seq, f"layer {layer}: {r and len(r)} steps for {seq}"
        rel = 0.0
        for i, step in enumerate(r):
            a, b = c[:, i].reshape(-1), step.reshape(-1)
            scale = max(abs(float(v)) for v in b) or 1e-30
            rel = max(rel, max(abs(float(x) - float(y)) for x, y in zip(a, b)) / scale)
        rows.append((layer, rel))
        worst = max(worst, rel)
    print("# chunk_kda vs fused_recurrent_kda, same weights, same tokens")
    print(f"kda layers\t{len(rows)}\nworst max_rel\t{worst:.3e}")
    print("layer\tmax_rel")
    for layer, rel in rows[:6]:
        print(f"{layer}\t{rel:.3e}")


def _kda_zero_based(cfg):
    """`kda_layers` is 1-based on disk (`is_kda_layer` tests `layer_idx + 1`)."""
    return {i - 1 for i in cfg["linear_attn_config"]["kda_layers"]}


def compare(a_path, b_path):
    """Score one defect run against the `None` golden, per captured layer, and GATE it.

    Reported per LAYER rather than per tensor because that is what G1b asks about, and 800 tensor
    lines cannot be read. Two things are asserted, which together are `k3-port.md` §G rule 1:

      * the defect changed SOMETHING -- a defect that reddens nothing is not a defect;
      * every layer in `EXPECT_GREEN[defect]` is BIT-IDENTICAL. Only upstream layers can be
        there, since one forward pass propagates everything downstream.

    The two tensor SETS must match exactly. They are a property of the config and the capture
    list, never of the numbers -- so a mismatch is a broken harness, not a defect finding, and it
    aborts. That distinction was measured into existence: while routed experts were captured
    individually, four of five defects reported `inf` for most layers because a moved routing
    fires a different set of expert modules.
    """
    (ma, mb), sections = _both_sections(a_path, b_path)
    per = {}
    layer_of = lambda n: n.split(".")[2] if n.startswith("model.layers.") else "model"  # noqa: E731
    for items_a, items_b in sections:
        for k, v in _score(items_a, items_b, layer_of).items():
            n, diff, rel = per.get(k, (0, 0, 0.0))
            per[k] = (n + v[0], diff + v[1], max(rel, v[2]))
    defect = mb.get("defect")
    print(f"# {defect} vs {ma.get('defect')}  mode={ma.get('mode')}")
    print("layer\ttensors\tdiffering\tmax_rel")
    for layer in sorted(per, key=lambda k: (k == "model", int(k) if k.isdigit() else 0)):
        n, diff, rel = per[layer]
        print(f"{layer}\t{n}\t{diff}\t{rel:.3e}")

    if defect == ma.get("defect"):
        return
    if defect not in EXPECT_GREEN:
        raise SystemExit(f"--defect {defect} has no EXPECT_GREEN entry; add one, even if empty")
    if not any(diff for _, diff, _ in per.values()):
        raise SystemExit(f"--defect {defect} changed NOTHING; that is not a defect run")

    # A declared-green layer that was never CAPTURED used to score as green, because
    # `per.get(layer, (0, 0, 0))[1]` reads an absent layer as zero differing tensors. The
    # localisation claim would then rest on an empty set: drop a layer from CAPTURE_LAYERS, or name a
    # layer outside it in EXPECT_GREEN, and the matrix prints, finds nothing reddened, and exits 0.
    # Found by review 2026-08-11.
    green = list(map(str, EXPECT_GREEN[defect]))
    absent = [layer for layer in green if layer not in per]
    if absent:
        raise SystemExit(
            f"--defect {defect} declares layer(s) {absent} green, but nothing captured them -- "
            f"captured: {sorted(per)}. An uncaptured layer is not evidence of localisation.",
        )
    reddened = [layer for layer in green if per[layer][1]]
    if reddened:
        raise SystemExit(
            f"--defect {defect} reddened layer(s) {reddened}, which are upstream of it and must "
            f"stay bit-identical -- the localisation this golden claims is gone",
        )

    # And the POSITIVE half, which was only "something, somewhere, differs". EXPECT_GREEN encodes a
    # boundary, so the first captured layer PAST it must actually redden -- otherwise a perturbation
    # that missed its operator and only disturbed something downstream reads as a localised,
    # detected defect while the arithmetic the cell prices was never exercised.
    # Numeric keys only: `per` also holds "model" for the model-level fold, and sorting the whole
    # key set by `int` raises on it. This arm had never executed when it was written -- every
    # exercise of the comparator tripped an earlier gate first -- which is how the crash was found.
    downstream = [
        layer for layer in sorted((k for k in per if k.isdigit()), key=int) if layer not in green
    ]
    if downstream and not per[downstream[0]][1]:
        raise SystemExit(
            f"--defect {defect} left layer {downstream[0]} bit-identical, the first captured layer "
            f"it does NOT declare green -- so whatever it changed, it was not this operator here. "
            f"Either the perturbation missed, or EXPECT_GREEN names the wrong boundary.",
        )


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


def load_reference(ref_dir):
    """Import the pinned reference as a package, because its files use relative imports.

    `modeling_kimi_linear.py` does `from .configuration_kimi_k3 import ...`, so the directory has
    to be a package. `__init__.py` is created here rather than shipped: the reference is
    downloaded at a pinned revision and never vendored (`repack-one-expert.md` gives the reason),
    and a file we add is a file the recipe has to account for.
    """
    ref = pathlib.Path(ref_dir).resolve()
    (ref / "__init__.py").touch()
    sys.path.insert(0, str(ref.parent))
    pkg = ref.name
    cfg = importlib.import_module(f"{pkg}.configuration_kimi_k3")
    mdl = importlib.import_module(f"{pkg}.modeling_kimi_linear")
    return cfg, mdl


def sha_of(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def preflight_env():
    """Refuse to generate under a python env other than the one the vendored goldens were made in.

    A second venv now exists on this machine: Muse Glimmer's S1b needs `muse_glimmer`, which is
    native to transformers **5.15.0.dev0**, while these goldens are transformers **4.56.2** — and
    `K3_ANCHOR_VENV` pointing at the wrong one is a two-character mistake. Without this check the
    symptom is a `cmp` mismatch at the END of a ~25 min GPU-locked regeneration, reported as
    "DIFFERS ... find out why it moved", which invites suspecting the driver.

    THE PIN IS READ OUT OF THE VENDORED GOLDEN, not restated here. Those bytes already carry the
    versions that produced them and `k3_anchor.rs` already asserts them, so a copy in this file
    would be a third statement of the same fact -- exactly the shape that made CLAUDE.md's
    exemption count wrong three times. A deliberate re-pin therefore needs no edit here: regenerate
    with the override, re-vendor, and the new bytes become the new pin.

    Not checked: the GPU name, which `k3_anchor.rs` pins on the bytes and which `--device cpu` runs
    legitimately do not have. This is about the venv, which is the resource two ports now share.
    """
    import fla
    import transformers
    import triton

    vendored = sorted(pathlib.Path(__file__).parent.glob("k3-anchor-decode-*.bin"))
    if not vendored:
        print("k3-anchor: no vendored golden to check this env against", file=sys.stderr)
        return
    pinned = read_golden(vendored[0])[0]
    live = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "fla": fla.__version__,
        "triton": triton.__version__,
    }
    drift = [
        f"{k}: golden says {pinned.get(k)!r}, this env has {v!r}"
        for k, v in live.items()
        if pinned.get(k) != v
    ]
    if not drift:
        return
    msg = f"this python env is not the one that produced {vendored[0].name}:\n  " + "\n  ".join(
        drift
    )
    if os.environ.get("K3_ANCHOR_ALLOW_ENV_DRIFT"):
        print(f"k3-anchor: WARNING, {msg}\n  (ALLOW_ENV_DRIFT set, continuing)", file=sys.stderr)
        return
    raise SystemExit(
        f"k3-anchor: {msg}\n"
        "  Fix K3_ANCHOR_VENV, or set K3_ANCHOR_ALLOW_ENV_DRIFT=1 to re-pin deliberately -- which\n"
        "  means re-vendoring the bytes AND updating the versions k3_anchor.rs asserts."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE", "OTHER"),
        help="score two goldens, assert the defect's green layers, and exit; no GPU, no torch",
    )
    ap.add_argument("--ref", help="dir holding the pinned reference .py files")
    ap.add_argument("--config", help="the vendored real config.json")
    ap.add_argument("--out")
    ap.add_argument("--defect", default="None", choices=sorted(DEFECTS))
    ap.add_argument("--mode", default="decode", choices=("prefill", "decode", "kda-equiv"))
    ap.add_argument("--seq", type=int, default=8, help="prefill length")
    # Kept although every recorded run is `cuda`: `--device cpu` gets a build error out of the
    # config, the weight init and the hooks in seconds, without taking the GPU lock, and that is
    # how three defects in this file were found. It cannot produce a golden -- the forward dies
    # in triton -- which is the point.
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--salt", default="k3-anchor-1", help="weight-init salt; part of the record")
    # fp64 is not a mode anyone ships; it is how the TOLERANCE floor is measured. Running the same
    # reference at double precision and diffing gives the fp32 run's own rounding error, which is
    # the bound an independent correct kernel cannot beat. See `anchor.md`'s tolerance table.
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument(
        "--by-operator",
        nargs=2,
        metavar=("BASE", "OTHER"),
        help="score two goldens per OPERATOR instead of per layer, and exit; no GPU, no torch",
    )
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if args.by_operator:
        by_operator(*args.by_operator)
        return
    for req in ("ref", "config", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required unless --compare is given")

    global torch
    import torch

    # Before `load_reference` and before any weight is drawn: a wrong venv should cost seconds.
    preflight_env()

    cfg_mod, mdl_mod = load_reference(args.ref)
    cfg = build_config(cfg_mod, args.config)

    torch.use_deterministic_algorithms(True, warn_only=True)
    model = mdl_mod.KimiLinearForCausalLM(cfg)
    # AFTER construction: `KimiLinearModel.__init__` overwrites the field with
    # `flash_attention_2` no matter what the config said. See the module docstring.
    model.config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"
    model.eval()
    init_weights(model, args.salt)
    model.to(args.device)
    if args.dtype == "float64":
        # AFTER `init_weights`, so the drawn values are identical to the fp32 run's to the bit and
        # the only difference measured is arithmetic. Every fla module stays fp32 (see
        # `fp32_island`): all three are triton-backed and none compiles for fp64, and the KDA
        # kernel returns an fp32 recurrent state whatever the input dtype anyway.
        model.double()
        print(f"k3-anchor: fp64 with {fp32_island(model)} fla modules held at fp32")

    ctx = {
        "kda_layer_ids": [i for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i)],
        "kda_kwargs": {},
        "kda_calls": 0,
        "capturing": False,
        "kda_fp32_island": args.dtype == "float64",
    }
    DEFECTS[args.defect](model, ctx)

    cap = Capture()
    # Wrappers go on BEFORE any forward pass so a kwarg-only defect perturbs the warm prefill
    # too; `ctx["capturing"]` is what decides whether a call is recorded.
    wrap_kda_ops(mdl_mod, cap, ctx, CAPTURE_LAYERS)
    wrap_attn_res(mdl_mod, model, cap, ctx, CAPTURE_LAYERS)
    wrap_mla_attention(mdl_mod, model, cap, ctx, CAPTURE_LAYERS)
    latent = wrap_latent_sandwich(model, cap, ctx, CAPTURE_LAYERS)
    ids = torch.arange(1, args.seq + 1, device=args.device).unsqueeze(0) % cfg.vocab_size

    if args.mode == "kda-equiv":
        # A measurement, not a golden: it writes nothing, because what it produces is one number
        # per KDA layer rather than a fixture anything scores against.
        kda_equiv(model, mdl_mod, ctx, args.seq, args.device, cfg.vocab_size)
        return

    with torch.no_grad():
        if args.mode == "prefill":
            handles, expected = hook_model(model, cap, CAPTURE_LAYERS)
            ctx["capturing"] = True
            out = model(input_ids=ids, use_cache=True)
        else:
            # Prefill UNHOOKED, then capture exactly one decode step. The cache and the KDA
            # recurrent state are what make decode a different arithmetic path from prefill
            # (`fused_recurrent_kda`, not `chunk_kda`), and capturing the prefill too would
            # triple the file for tensors the prefill golden already holds.
            warm = model(input_ids=ids, use_cache=True)
            handles, expected = hook_model(model, cap, CAPTURE_LAYERS)
            ctx["capturing"] = True
            nxt = torch.tensor([[args.seq + 1]], device=args.device) % cfg.vocab_size
            out = model(input_ids=nxt, past_key_values=warm.past_key_values, use_cache=True)
        cap.add("logits", out.logits)
    for h in handles:
        h.remove()

    # A hook that never fires is silent, and that silence is what hid the AttnRes gap for a day.
    # Every registered hook must have produced at least one tensor, and every KDA layer must have
    # been reached exactly once -- a short count would RELABEL captures (layer 1's operator
    # boundary written under layer 0's name) rather than produce a layer nothing expects, which
    # no assertion in `tests/k3_anchor.rs` could see.
    silent = sorted(set(expected) - cap.fired)
    assert not silent, f"{len(silent)} hooks never fired, e.g. {silent[:4]}"
    assert ctx["kda_calls"] == len(ctx["kda_layer_ids"]), (
        f"{ctx['kda_calls']} KDA calls for {len(ctx['kda_layer_ids'])} KDA layers"
    )
    for tag in ("self_attention_res", "mlp_res"):
        assert any(f".{tag}.out" in s for s in cap.seen), f"no {tag} fold was captured"
    assert any("output_attn_res.out" in s for s in cap.seen), "no model-level fold was captured"
    # The latent sandwich's pre-hooks get a COUNT, not an `any`. `first_k_dense_replace` decides
    # which captured layers own a `block_sparse_moe` at all, so a suffix that matched nothing and a
    # config that made layer 0 dense are the same shape of silence -- and one of them is correct.
    # Deriving the expected number from the config is what separates them.
    moe_layers = [i for i in CAPTURE_LAYERS if i >= cfg.first_k_dense_replace]
    assert len(latent) == len(moe_layers), (
        f"{len(latent)} latent-sandwich pre-hooks for {len(moe_layers)} MoE capture layers "
        f"{moe_layers} -- expected one `routed_expert_norm` each"
    )
    for tag in (".routed_expert_norm.in", ".routed_expert_norm.weight"):
        got = sum(1 for s in cap.seen if s.endswith(tag))
        assert got == len(moe_layers), f"{got} captures of {tag} for {len(moe_layers)} MoE layers"

    import fla
    import transformers
    import triton

    meta = [
        ("defect", args.defect),
        ("mode", args.mode),
        ("seq", str(args.seq)),
        ("salt", args.salt),
        ("dtype", str(next(model.parameters()).dtype)),
        ("entry_point", type(model).__name__),
        ("quantized", "no" if getattr(cfg, "quantization_config", None) is None else "yes"),
        ("torch", torch.__version__),
        ("transformers", transformers.__version__),
        ("fla", fla.__version__),
        ("triton", triton.__version__),
        ("gpu", torch.cuda.get_device_name(0)),
        ("attn_implementation", model.config._attn_implementation),
        ("capture_layers", ",".join(str(i) for i in CAPTURE_LAYERS)),
        ("structural_asserted", ",".join(STRUCTURAL + STRUCTURAL_LINEAR_ATTN)),
        ("tiny_config", json.dumps(cfg.to_dict(), sort_keys=True, default=str)),
        ("ref_modeling_sha256_16", sha_of(pathlib.Path(args.ref) / "modeling_kimi_linear.py")),
        ("ref_config_sha256_16", sha_of(pathlib.Path(args.ref) / "configuration_kimi_k3.py")),
        ("real_config_sha256_16", sha_of(args.config)),
    ]
    n = write_golden(args.out, meta, cap)
    print(
        f"k3-anchor: {args.out} — {len(cap.floats)} float, {len(cap.ints)} int tensors, "
        f"{n} bytes, defect={args.defect} mode={args.mode}",
    )


if __name__ == "__main__":
    main()
