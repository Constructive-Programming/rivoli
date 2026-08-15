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

What is left here is the generation path: the weights, the defects, the fp64 instruments and
`main`. The tiny config, the container and the two scorers are `k3_anchor_lib.py`; `Capture` and
the forward hooks are `k3_anchor_capture.py`; the two values no hook can see are
`k3_anchor_taps.py`. Split out 2026-08-15 under the 800-line-per-file gate with every body moved
unchanged -- these goldens cannot be regenerated without the pinned venv and a GPU, so a rewrite
of one would be a change nothing could score. All three are re-exported below, because this
file's name is the one `tests/k3-anchor.sh` and every recipe in `anchor.md` run.
"""

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import sys

# Imported for the SURFACE, not used here. `struct` went with the container and `math` with the
# scorer in the 2026-08-15 split, but this file's importable names are what `tests/k3-anchor.sh`
# and every recipe in `anchor.md` reach for, and a split is not allowed to narrow them.
import math  # noqa: F401
import struct  # noqa: F401

# The moved halves, re-exported for the same reason. `k3_anchor_lib` is the deviceless side (no
# torch, so `--compare` still runs without one); `k3_anchor_capture` holds the hooks, and
# `k3_anchor_taps` the two values no hook can see. All three sit beside this file, which is
# `sys.path[0]` whenever it runs as a script -- the only way it is ever run.
from k3_anchor_capture import Capture, Tap, _fire, hook_model  # noqa: F401
from k3_anchor_lib import (  # noqa: F401
    CAPTURE_LAYERS,
    EXPECT_GREEN,
    MAGIC,
    STRUCTURAL,
    STRUCTURAL_LINEAR_ATTN,
    TINY_LINEAR_ATTN,
    TINY_TEXT,
    _both_sections,
    _kda_zero_based,
    _s,
    _score,
    _u64,
    build_config,
    by_operator,
    compare,
    operator_of,
    read_golden,
    write_golden,
)
from k3_anchor_taps import (  # noqa: F401
    _fold_mixing_normalised,
    wrap_attn_res,
    wrap_kda_ops,
)

# Bound by `main` so `--compare` runs with no torch, no fla and no GPU — the property that lets a
# defect run be re-scored from vendored bytes on any machine.
torch = None

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


# Every fla module in the KDA path, by class name so this file need not import fla. All three are
# triton-backed and all three refuse fp64 — `ShortConvolution` and `FusedRMSNormGated` as much as
# the KDA ops themselves — which is why `--dtype float64` cannot simply be "the model in double".
FLA_MODULES = ("ShortConvolution", "FusedRMSNormGated")


def _down(x):
    """fp64 -> fp32 for anything floating; ints, `None` and flags pass through untouched."""
    return x.float() if hasattr(x, "dtype") and x.is_floating_point() else x


def _up(out):
    """The island's return trip, and the two arms test DIFFERENT predicates on purpose.

    A tuple member is raised only if it is floating point -- fla's ops return integer index
    tensors and `None` beside their activations. A bare return value is raised on `hasattr` alone,
    which is what these modules have always done; both halves are the pre-split behaviour and a
    golden that moved would be a golden nothing could re-derive.
    """
    if isinstance(out, tuple):
        return tuple(
            o.double() if hasattr(o, "dtype") and o.is_floating_point() else o for o in out
        )
    return out.double() if hasattr(out, "dtype") else out


def _fp32_shim(orig):
    """Wrap one fla module's `forward` so it computes in fp32 between fp64 neighbours."""

    def shim(*a, _orig=orig, **kw):
        out = _orig(*[_down(x) for x in a], **{k: _down(v) for k, v in kw.items()})
        return _up(out)

    return shim


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
        m.forward = _fp32_shim(m.forward)
        n += 1
    return n


def _kda_both_paths(model, ctx, args, vocab):
    """Run the same tokens twice, once through each of fla's two KDA implementations.

    The prefill goes through `chunk_kda` (T positions at once); the token-at-a-time loop goes
    through `fused_recurrent_kda`, which is the decode path. `equiv_sink` is what the KDA tap
    files each call's output into, and `kda_calls` is reset per step because the tap numbers
    layers by call order within one forward.
    """
    chunk, steps = {}, {}
    seq, device = args.seq, args.device
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
    return chunk, steps


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


def _steps_of(steps, layer, seq):
    """One layer's per-step outputs, asserting the decode loop reached it once per position."""
    r = steps.get(("fused_recurrent_kda", layer))
    assert r is not None and len(r) == seq, f"layer {layer}: {r and len(r)} steps for {seq}"
    return r


def _kda_equiv_rows(chunk, steps, seq):
    """`(layer, max_rel)` per KDA layer, in layer order."""
    rows = []
    for (tag, layer), got in sorted(chunk.items(), key=lambda kv: kv[0][1]):
        assert tag == "chunk_kda", f"prefill went through {tag}"
        c = got[0]                                    # [1, T, H, D]
        rows.append((layer, _step_rel(c, _steps_of(steps, layer, seq))))
    return rows


def kda_equiv(model, tap, args, vocab):
    """**KDA's tolerance floor, measured against fla's own second implementation of it.**

    fp64 cannot serve here (see `wrap_kda_ops`), so the floor comes from the two paths fla ships
    for the same recurrence: `chunk_kda` over T positions at once, and `fused_recurrent_kda` one
    position at a time. They are the same mathematics by the same authors, so their disagreement is
    exactly what "a correct implementation in fp32, associating differently" costs — which is the
    number a HIP kernel's tolerance has to sit above.

    Reported over EVERY KDA layer, not just the captured six: a tolerance is a bound, and a bound
    wants the worst case.
    """
    chunk, steps = _kda_both_paths(model, tap.ctx, args, vocab)
    rows = _kda_equiv_rows(chunk, steps, args.seq)
    # `default` rather than a `worst = 0.0` seed carried through the loop -- same value, including
    # the no-KDA-layer case the seed used to cover.
    worst = max((rel for _, rel in rows), default=0.0)
    print("# chunk_kda vs fused_recurrent_kda, same weights, same tokens")
    print(f"kda layers\t{len(rows)}\nworst max_rel\t{worst:.3e}")
    print("layer\tmax_rel")
    for layer, rel in rows[:6]:
        print(f"{layer}\t{rel:.3e}")


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


def _parse_args():
    """The command line, and the one cross-argument rule argparse cannot state.

    `--ref`/`--config`/`--out` are required for a generation run and meaningless for a scoring
    one, so they are checked here rather than declared `required=True` -- which would break the
    deviceless `--compare` this whole split exists to keep.
    """
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
    if args.compare or args.by_operator:
        return args
    for req in ("ref", "config", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required unless --compare is given")
    return args


def _build_model(mdl_mod, cfg, args):
    """The reference model, initialised, on the device, at the requested precision."""
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
    return model


def _new_ctx(cfg, args):
    """The mutable per-run state every defect, tap and hook shares."""
    return {
        "kda_layer_ids": [i for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i)],
        "kda_kwargs": {},
        "kda_calls": 0,
        "capturing": False,
        "kda_fp32_island": args.dtype == "float64",
    }


def _capture_pass(model, tap, args, vocab):
    """Run the mode's forward pass with the hooks armed, and return the names that were hooked.

    Handles are removed here rather than by the caller: a hook left registered would fire on
    whatever ran next, and nothing after this returns needs them.
    """
    ids = torch.arange(1, args.seq + 1, device=args.device).unsqueeze(0) % vocab
    with torch.no_grad():
        if args.mode == "prefill":
            handles, expected = hook_model(model, tap.cap, tap.layers)
            tap.ctx["capturing"] = True
            out = model(input_ids=ids, use_cache=True)
        else:
            # Prefill UNHOOKED, then capture exactly one decode step. The cache and the KDA
            # recurrent state are what make decode a different arithmetic path from prefill
            # (`fused_recurrent_kda`, not `chunk_kda`), and capturing the prefill too would
            # triple the file for tensors the prefill golden already holds.
            warm = model(input_ids=ids, use_cache=True)
            handles, expected = hook_model(model, tap.cap, tap.layers)
            tap.ctx["capturing"] = True
            nxt = torch.tensor([[args.seq + 1]], device=args.device) % vocab
            out = model(input_ids=nxt, past_key_values=warm.past_key_values, use_cache=True)
        tap.cap.add("logits", out.logits)
    for h in handles:
        h.remove()
    return expected


def _assert_captured(tap, expected):
    """Refuse a golden whose capture was SHORT.

    A hook that never fires is silent, and that silence is what hid the AttnRes gap for a day.
    Every registered hook must have produced at least one tensor, and every KDA layer must have
    been reached exactly once -- a short count would RELABEL captures (layer 1's operator boundary
    written under layer 0's name) rather than produce a layer nothing expects, which no assertion
    in `tests/k3_anchor.rs` could see.
    """
    cap, ctx = tap.cap, tap.ctx
    silent = sorted(set(expected) - cap.fired)
    assert not silent, f"{len(silent)} hooks never fired, e.g. {silent[:4]}"
    assert ctx["kda_calls"] == len(ctx["kda_layer_ids"]), (
        f"{ctx['kda_calls']} KDA calls for {len(ctx['kda_layer_ids'])} KDA layers"
    )
    for tag in ("self_attention_res", "mlp_res"):
        assert any(f".{tag}.out" in s for s in cap.seen), f"no {tag} fold was captured"
    assert any("output_attn_res.out" in s for s in cap.seen), "no model-level fold was captured"


def _metadata(args, model, cfg):
    """Everything the bytes have to carry about what produced them, including all four declared
    deviations -- a golden that hides what produced it is worse than no golden."""
    import fla
    import transformers
    import triton

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


def _generate(args):
    """One generation run: build, perturb, capture, and write the golden (or, in `kda-equiv`,
    print the floor and write nothing)."""
    cfg_mod, mdl_mod = load_reference(args.ref)
    cfg = build_config(cfg_mod, args.config)
    model = _build_model(mdl_mod, cfg, args)

    ctx = _new_ctx(cfg, args)
    DEFECTS[args.defect](model, ctx)

    tap = Tap(Capture(), ctx, CAPTURE_LAYERS)
    # Wrappers go on BEFORE any forward pass so a kwarg-only defect perturbs the warm prefill
    # too; `ctx["capturing"]` is what decides whether a call is recorded.
    wrap_kda_ops(mdl_mod, tap)
    wrap_attn_res(mdl_mod, model, tap)

    if args.mode == "kda-equiv":
        # A measurement, not a golden: it writes nothing, because what it produces is one number
        # per KDA layer rather than a fixture anything scores against.
        kda_equiv(model, tap, args, cfg.vocab_size)
        return

    _assert_captured(tap, _capture_pass(model, tap, args, cfg.vocab_size))
    cap = tap.cap
    n = write_golden(args.out, _metadata(args, model, cfg), cap)
    print(
        f"k3-anchor: {args.out} — {len(cap.floats)} float, {len(cap.ints)} int tensors, "
        f"{n} bytes, defect={args.defect} mode={args.mode}",
    )


def main():
    args = _parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if args.by_operator:
        by_operator(*args.by_operator)
        return

    global torch
    import torch

    # Before `load_reference` and before any weight is drawn: a wrong venv should cost seconds.
    preflight_env()
    _generate(args)


if __name__ == "__main__":
    main()
