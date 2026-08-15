#!/usr/bin/env python3
"""**The S1b anchor: goldens produced by Muse Glimmer's own first-party stack, not by ours.**

`docs/investigations/glimmer-port.md` G1b calls this mandatory. `docs/reference/glimmer-architecture.md`
was extracted by *reading* `modeling_muse_glimmer.py`, and a fixture derived from that reading
would put one misreading in the spec, in the golden, and in the kernel checked against it. So the
golden comes from *running* the reference. **Nothing here re-implements Muse Glimmer** — every
number below is produced by transformers' own modules; this file builds a tiny config, fills the
weights, taps values on their way past, and writes them out.

What runs is `transformers.models.muse_glimmer` (target) and `.muse_glimmer_assistant` (the DFlash
drafter) at the commit pinned in `glimmer-architecture.md`, over a **tiny config that keeps every
structural field real and shrinks only widths**. Depth and structure are where the traps live;
width is what costs. Kept real: `rms_norm_eps` 1e-5 *and* `post_norm_eps` 1e-8 (the two-eps
sandwich), `qk_scale_factor` 3.87, `output_multiplier`, `final_logit_softcapping` 20.0,
`rope_theta` 500000, the `[w,w,w,full]` layer-type pattern with its NoPE coupling, and GQA with a
group count that is neither 1 nor the head count.

**Widths are chosen so that no two structurally distinct quantities are accidentally equal.** K3's
anchor review found an assertion that looked like it pinned a coupling and was satisfied by the
wrong reading too, because four widths collided. Here: hidden 72, q width 48, kv width 16, head_dim
8, heads 6, kv heads 2, group 3, intermediate 216, vocab 61, layers 8, window 4, prompt 12. In
particular `hidden != num_heads * head_dim` (72 vs 48), which is the real model's shape (6656 vs
4096) and the one a port is most likely to collapse.

**Declared deviations**, each recorded in the golden's own metadata because a golden that hides
what produced it is worse than no golden:

  * `_attn_implementation` is forced to `eager`. Eager is the semantics the fused paths
    approximate, it needs no GPU kernel, and it is the only one this file can tap.
  * **The model runs in torch's default dtype, fp32, while the checkpoint is bf16.** Right for a
    reference — the point is to pin arithmetic, not to reproduce one accumulation order — but it is
    where S2's tolerance decision starts, so `dtype` is in the metadata.
  * `MuseGlimmerForConditionalGeneration` on a text-only input, because **the logit softcap lives
    only on that wrapper** (M:1253-1260); there is no text-only causal-LM class. The vision tower is
    constructed at toy widths and never runs. `entry_point` records this.
  * `bos/eos/pad` and the image/video token ids are shrunk into the tiny vocab. Unused by a forward
    pass, but a `@strict` config that names ids outside its own vocab is a lie in the metadata.
  * `output_multiplier` keeps the **released checkpoint's** 0.19611613513818404 rather than
    `1/sqrt(hidden/256)` recomputed at hidden 72. It is a config value the port reads, not a
    formula the port evaluates — recomputing it here would test arithmetic nothing performs.

Usage:

    python3 tests/glimmer_anchor_driver.py --mode text  --salt glimmer-anchor-1 --out golden.bin
    python3 tests/glimmer_anchor_driver.py --mode draft --salt glimmer-anchor-1 --out golden.bin

`--defect` perturbs the reference to prove a golden can go red — see `DEFECTS`. Each defect
declares the captures it must leave **green**, and `--compare A B` asserts both halves: that
something moved, and that nothing else did. A defect run that reddens everything proves nothing
about where the arithmetic lives.

**No GPU.** Unlike K3's anchor, this reference is plain PyTorch with a CPU path for every operator,
so goldens regenerate without the device lock. They are vendored anyway: the gate that reads them
needs no python, no venv and no network.
"""

import argparse
import json
import pathlib

import torch

# The fixture kit, the defect matrix and the two halves of the scoring, split out of this file
# 2026-08-15 to hold every authored file under the 800-line cap. The split moved bodies between
# files and changed nothing else; `main` stayed because it reads this module's `__doc__` for
# `--help`, so moving it would change what the CLI prints.
#
# Later the same day, for the CodeScene 10.0 gate (this file scored 8.52), what was left was
# decomposed IN PLACE: `install_text_taps` into its four numbered taps plus the two hook families,
# `run_draft` into `_draft_inputs`/`_draft_mask`/`_run_draft_block`, `main` into
# `_build_parser`/`_check_run_args`/`_metadata`, and the target build `run_text` and `run_draft`
# had each written out into `_build_target`. **Shape only, and deliberately so** -- the venv that
# regenerates these goldens is gone, so a behaviour change here could not have been caught by
# re-running anything. Every helper is one contiguous run of the body it came from, called in that
# body's order, with its comments attached to the lines they were about.
from glimmer_anchor_compare import compare
from glimmer_anchor_defects import (
    DEFECTS,
    DRAFT_DEFECTS,
    TEXT_DEFECTS,
    defect_draft_causal,
    defect_draft_context_unprojected,
    defect_embed_norm_off,
    defect_full_layers_slide,
    defect_gate_disabled,
    defect_kv_broadcast_blocked,
    defect_norm_not_centered,
    defect_post_norm_eps_shared,
    defect_qk_norm_off,
    defect_qk_scale_on_k,
    defect_rope_interleaved,
    defect_rope_on_nope_layers,
    defect_softcap_off,
    defect_window_off_by_one,
)
from glimmer_anchor_lib import (
    DECODE_STEPS,
    MAGIC,
    PROMPT_LEN,
    Capture,
    environment,
    init_weights,
    preflight_env,
    prompt_ids,
    read_golden,
    tiny_draft_config,
    tiny_text_config,
    tiny_vision_config,
    write_golden,
)
from glimmer_anchor_operators import by_operator, operator_of

# Bound but unused HERE. `dir(glimmer_anchor_driver)` is a surface: `tests/glimmer-anchor.sh`
# imports out of this module by name, and until the split every stdlib module the one-file version
# imported was importable from it. Four lines is a cheaper promise than auditing what every caller
# reads, and it makes the split provably surface-preserving rather than nearly so.
import hashlib  # noqa: F401
import math  # noqa: F401
import struct  # noqa: F401
import sys  # noqa: F401


# ---------------------------------------------------------------------------------------------
# Taps. A tap wraps a reference function or module and records what passed through it. It never
# computes anything the reference would not have computed.


class Taps:
    def __init__(self, cap):
        self.cap = cap
        self.step = None
        # The layer currently executing, published by the attention tap. **Not a call counter.**
        # The first version numbered rope captures by call index, which counts only ROTATED layers
        # -- so the six rotated layers of an eight-layer model were labelled L0..L5 and every NoPE
        # golden was mislabelled. A capture whose name lies about which layer it came from is worse
        # than a missing one, because the gate will compare it to the wrong thing and pass.
        self.layer = None
        self.qk_in_layer = 0
        self.rope_calls = 0
        self.attend_calls = 0
        self.qk_norm_calls = 0
        self.rotary_calls = 0
        self._undo = []

    def prefix(self, layer=None):
        p = f"t{self.step}"
        return p if layer is None else f"{p}.L{layer}"

    def patch(self, obj, attr, new):
        old = getattr(obj, attr)
        setattr(obj, attr, new)
        self._undo.append((obj, attr, old))
        return old

    def close(self):
        for obj, attr, old in reversed(self._undo):
            setattr(obj, attr, old)
        self._undo.clear()


def install_text_taps(mdl, model, taps):
    """Tap the four places the target's arithmetic is visible and a module hook is not enough.

    The six helpers are taps 1 to 4 as the comments below number them, the 1b layer-index publisher
    that two of them depend on, and the unnumbered module-hook family. Split out of this body
    2026-08-15 (it was 76 LoC): the patch order, the hook registration order and therefore the order
    of the returned list are the ones that produced the vendored goldens, and `Taps.close` still
    unwinds the patches in reverse.
    """
    _patch_rotary(mdl, taps)
    _patch_attention_layer(mdl, taps)
    _patch_rope(mdl, taps)
    _patch_attend(mdl, taps)
    return _hook_weightless_norms(model, taps) + _hook_modules(model, taps)


def _patch_rotary(mdl, taps):
    # 1. The rotary table. One table for the whole model (M:513); the per-layer NoPE decision is a
    #    flag applied at the CALL SITE, not a second table, and that is trap #1 in section 9.
    cap = taps.cap
    orig_rotary = mdl.MuseGlimmerTextRotaryEmbedding.forward

    def rotary(self, x, position_ids):
        cos, sin = orig_rotary(self, x, position_ids)
        taps.rotary_calls += 1
        cap.add(f"{taps.prefix()}.rope.cos", cos)
        cap.add(f"{taps.prefix()}.rope.sin", sin)
        return cos, sin

    taps.patch(mdl.MuseGlimmerTextRotaryEmbedding, "forward", rotary)


def _patch_attention_layer(mdl, taps):
    # 1b. The attention module, which is the only place the LAYER INDEX is in scope. It publishes it
    #     for the two function taps below, which see nothing but tensors.
    orig_attn = mdl.MuseGlimmerTextAttention.forward

    def attn_forward(self, *a, **kw):
        taps.layer = self.layer_idx
        taps.qk_in_layer = 0
        try:
            return orig_attn(self, *a, **kw)
        finally:
            taps.layer = None

    taps.patch(mdl.MuseGlimmerTextAttention, "forward", attn_forward)


def _patch_rope(mdl, taps):
    # 2. `apply_rotary_pos_emb`, so the golden holds q and k on BOTH sides of the rotation. A port
    #    that gets rotate_half vs interleaved wrong differs here and nowhere earlier.
    cap = taps.cap
    orig_rope = mdl.apply_rotary_pos_emb

    def roped(q, k, cos, sin, unsqueeze_dim=1):
        taps.rope_calls += 1
        qe, ke = orig_rope(q, k, cos, sin, unsqueeze_dim)
        p = taps.prefix(taps.layer)
        cap.add(f"{p}.q.pre_rope", q)
        cap.add(f"{p}.k.pre_rope", k)
        cap.add(f"{p}.q.roped", qe)
        cap.add(f"{p}.k.roped", ke)
        return qe, ke

    taps.patch(mdl, "apply_rotary_pos_emb", roped)


def _patch_attend(mdl, taps):
    # 3. `eager_attention_forward`, which sees the POST-CACHE key/value states -- that is the ring
    #    buffer's contents as the kernel must reproduce them, including eviction -- and returns the
    #    attend output BEFORE the sigmoid gate and o_proj.
    cap = taps.cap
    orig_attend = mdl.eager_attention_forward

    def attend(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        taps.attend_calls += 1
        out, weights = orig_attend(
            module, query, key, value, attention_mask, scaling, dropout, **kw
        )
        p = taps.prefix(taps.layer)
        cap.add(f"{p}.attend.q", query)
        cap.add(f"{p}.attend.k_cache", key)
        cap.add(f"{p}.attend.v_cache", value)
        if attention_mask is not None:
            # Clamped: the reference's additive mask carries the dtype minimum, which does not
            # survive a round trip through anything and is not what a port reproduces. What the
            # golden pins is WHICH positions are masked.
            cap.add(f"{p}.attend.mask", (attention_mask > -1.0).to(torch.float32))
        cap.add(f"{p}.attend.weights", weights)
        cap.add(f"{p}.attend.out", out)
        return out, weights

    taps.patch(mdl, "eager_attention_forward", attend)


def _hook_weightless_norms(model, taps):
    # 4. The two WEIGHTLESS norms, which ship no tensor and are therefore the two a port can omit
    #    without the checkpoint complaining: `embed_norm` and the per-layer `qk_norm`. The latter is
    #    one module called twice, q then k in that order (M:342-343), and the q call is captured
    #    BEFORE the 3.87 scale so the golden separates the norm from the scale.
    #
    #    **These are forward HOOKS, not a patched forward.** `nn.Module.__call__` runs hooks around
    #    whatever `self.forward` currently is, so a defect that replaces the norm with the identity
    #    still produces the capture -- holding the unnormalised value, which is the whole point. The
    #    first version patched the class method, so `qk_norm_off` and `embed_norm_off` deleted their
    #    own evidence and died in the tap census instead of reddening a golden.
    cap = taps.cap
    handles = []

    def qk_hook(_mod, _args, out):
        which = ("q", "k")[taps.qk_in_layer]
        taps.qk_in_layer += 1
        taps.qk_norm_calls += 1
        cap.add(f"{taps.prefix(taps.layer)}.qk_norm.{which}", out)

    def embed_hook(_mod, _args, out):
        taps.qk_norm_calls += 1
        cap.add(f"{taps.prefix()}.embed_norm.out", out)

    handles.append(
        model.model.language_model.embed_tokens.embed_norm.register_forward_hook(embed_hook)
    )
    for layer in model.model.language_model.layers:
        handles.append(layer.self_attn.qk_norm.register_forward_hook(qk_hook))
    return handles


def _hook_modules(model, taps):
    # Module hooks for everything that IS a module output. Cheap, and they cover the sandwich: four
    # norms per layer, the gate before o_proj, and the SwiGLU.
    cap = taps.cap
    handles = []
    for li, layer in enumerate(model.model.language_model.layers):
        for what, mod in (
            ("input_layernorm", layer.input_layernorm),
            ("post_attention_layernorm", layer.post_attention_layernorm),
            ("pre_feedforward_layernorm", layer.pre_feedforward_layernorm),
            ("post_feedforward_layernorm", layer.post_feedforward_layernorm),
            ("attn.gate_proj", layer.self_attn.gate_proj),
            ("attn.o_proj", layer.self_attn.o_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        ):
            handles.append(mod.register_forward_hook(_hook(cap, taps, li, what)))
    handles.append(
        model.model.language_model.norm.register_forward_hook(_hook(cap, taps, None, "final_norm"))
    )
    return handles


def _hook(cap, taps, layer, what):
    def fn(_mod, args, out):
        name = f"{taps.prefix(layer)}.{what}"
        if what == "attn.o_proj":
            # The o_proj INPUT is the gated attention output. Capturing it is how the golden pins
            # that the sigmoid gate lands BEFORE the projection (M:365-366) rather than after it,
            # which is the trap a port hits by reading the gate as an output scaling.
            cap.add(f"{name}.in_gated", args[0])
        cap.add(f"{name}.out", out)

    return fn


# ---------------------------------------------------------------------------------------------
# The runs.


def _build_target(cfg_mod, mdl, salt):
    """The target at tiny widths: eager, evaluated, and filled from the salt.

    **One builder for both modes, because the two must not drift.** `run_text` scores this model and
    `run_draft` uses it as the drafter's context source, and the drafter's goldens are only anchored
    to Muse Glimmer if that context comes from the SAME construction the text goldens pin. The two
    bodies were already identical statement for statement; making that structural is the whole of
    the change (2026-08-15 -- it also took `run_draft` from 94 LoC).

    Returns the text config beside the model because both callers need it: `run_text` for the vocab
    the prompt is drawn from, `run_draft` to force the drafter's `hidden_size` equal to it.
    """
    text = tiny_text_config(cfg_mod)
    top = cfg_mod.MuseGlimmerConfig(
        text_config=text,
        vision_config=tiny_vision_config(cfg_mod),
        image_token_id=59,  # shrunk into the tiny vocab
        video_token_id=58,  # shrunk into the tiny vocab
        out_hidden_size=32,
        projector_hidden_size=48,
    )
    top._attn_implementation = "eager"
    model = mdl.MuseGlimmerForConditionalGeneration(top)
    model.config.text_config._attn_implementation = "eager"
    model.eval()
    init_weights(model, salt)
    return text, model


def run_text(salt, defect, cap):
    from transformers.models.muse_glimmer import configuration_muse_glimmer as C
    from transformers.models.muse_glimmer import modeling_muse_glimmer as mdl

    text, model = _build_target(C, mdl, salt)

    fn = DEFECTS[defect][0]
    note = fn(mdl, model, model.config.text_config) if fn else ""

    taps = Taps(cap)
    handles = install_text_taps(mdl, model, taps)

    ids = prompt_ids(salt, text.vocab_size)
    cap.add_ints("prompt.ids", ids[0].tolist())
    # Structure the gate reads back rather than restating: which layers slide, which are NoPE.
    cfg = model.config.text_config
    cap.add_ints("layer_is_sliding", [int(t == "sliding_attention") for t in cfg.layer_types])
    cap.add_ints("layer_is_roped", [int(bool(t)) for t in cfg.layer_rope_theta])

    emitted = []
    past = None
    step_in = ids
    with torch.no_grad():
        for step in range(1 + DECODE_STEPS):
            taps.step = step
            taps.rope_calls = taps.attend_calls = taps.qk_norm_calls = taps.rotary_calls = 0
            out = model(input_ids=step_in, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            cap.add(f"t{step}.logits", logits)
            nxt = int(logits.argmax(-1))
            emitted.append(nxt)
            step_in = torch.tensor([[nxt]])
            _assert_taps_fired(taps, cfg, step)

    for h in handles:
        h.remove()
    taps.close()
    cap.add_ints("emitted.ids", emitted)
    return model, note, {"prompt_len": PROMPT_LEN, "decode_steps": DECODE_STEPS}


def _assert_taps_fired(taps, cfg, step):
    """A tap that silently stopped firing turns this whole file into a no-op that writes bytes.

    The counts are derived from the config, not written down: `apply_rotary_pos_emb` runs once per
    ROTATED layer, `eager_attention_forward` once per layer, and the weightless norm once for the
    embedding plus twice per layer.
    """
    roped = sum(1 for t in cfg.layer_rope_theta if t)
    want = {
        "rotary": 1,
        "rope": roped,
        "attend": cfg.num_hidden_layers,
        "qk_norm": 1 + 2 * cfg.num_hidden_layers,
    }
    got = {
        "rotary": taps.rotary_calls,
        "rope": taps.rope_calls,
        "attend": taps.attend_calls,
        "qk_norm": taps.qk_norm_calls,
    }
    if got != want:
        raise AssertionError(f"step {step}: taps fired {got}, expected {want} -- a tap is not installed")


def run_draft(salt, defect, cap):
    """One DFlash draft step: an anchor token plus masks, denoised against a target context.

    The target's hidden states are the CONTEXT, and they have to come from somewhere real -- a
    drafter fed random noise would pin the drafter's arithmetic against numbers no target produces,
    and the concatenation order of `target_layer_ids` (the thing a port gets wrong) would be
    unfalsifiable. So the target runs first, unperturbed, and its layer outputs are the input here.
    """
    from transformers.models.muse_glimmer import configuration_muse_glimmer as C
    from transformers.models.muse_glimmer import modeling_muse_glimmer as tgt_mdl
    from transformers.models.muse_glimmer_assistant import (
        configuration_muse_glimmer_assistant as DC,
    )
    from transformers.models.muse_glimmer_assistant import (
        modeling_muse_glimmer_assistant as mdl,
    )

    text, target = _build_target(C, tgt_mdl, salt)

    dcfg = tiny_draft_config(DC, text)
    dcfg._attn_implementation = "eager"
    draft = mdl.MuseGlimmerAssistantModel(dcfg)
    draft.eval()
    init_weights(draft, f"{salt}/draft")

    fn = DEFECTS[defect][0]
    note = fn(mdl, draft, dcfg) if fn else ""

    ids = prompt_ids(salt, text.vocab_size)
    cap.add_ints("prompt.ids", ids[0].tolist())
    cap.add_ints("target_layer_ids", dcfg.target_layer_ids)

    with torch.no_grad():
        ctx, noise = _draft_inputs(target, dcfg, ids, cap)
        dout = _run_draft_block((mdl, draft, dcfg), (ctx, noise), cap)

        # Logits from the TARGET's lm_head, then slice off index 0 -> block_size-1 candidates.
        logits = target.lm_head(dout.last_hidden_state)
        cap.add("draft.logits", logits)
        cap.add_ints("draft.candidates", logits[0, 1:, :].argmax(-1).tolist())

    return (
        draft,
        note,
        {
            "block_size": dcfg.block_size,
            "context_len": int(ctx.shape[1]),
            "use_cache": False,
            # No `draft_config` here: `main` already writes the returned model's config as
            # `tiny_config`, and for this mode that IS the drafter's. A second copy under a second
            # key is the two-frozen-copies shape this port has already been bitten by once.
        },
    )


def _draft_inputs(target, dcfg, ids, cap):
    """The drafter's two inputs, both produced by the UNPERTURBED target: context and noise block.

    Run inside `run_draft`'s `no_grad` block, where these statements were written.
    """
    tout = target(input_ids=ids, output_hidden_states=True, use_cache=False)
    # `hidden_states[i]` is the INPUT to layer i, so layer i's OUTPUT is index i+1. Off by one
    # here is a silent quality loss, and it is exactly the kind of thing a golden exists for.
    ctx = torch.cat([tout.hidden_states[i + 1] for i in dcfg.target_layer_ids], dim=-1)
    cap.add("draft.context_concat", ctx)

    # The draft block: the last accepted token, then masks. Embedded from the TARGET's
    # embedding matrix RAW -- section 11 item 3: the weightless embed-norm is skipped, so this
    # reaches past `MuseGlimmerTextNormedEmbedding.forward` to `nn.Embedding` on purpose.
    anchor = int(tout.logits[:, -1, :].argmax(-1))
    block = torch.tensor([[anchor] + [dcfg.mask_token_id] * (dcfg.block_size - 1)])
    emb = target.model.language_model.embed_tokens
    noise = torch.nn.functional.embedding(block, emb.weight)
    cap.add_ints("draft.block_ids", block[0].tolist())
    cap.add("draft.noise_embeds", noise)
    return ctx, noise


def _draft_mask(draft, dcfg, ctx, noise):
    """**The mask spans `context + block`, and the first draft call must be CACHELESS.**

    Q is the 4 block rows; K/V are `concat(context, block)`, 16 rows, and the two lengths
    differ inside one call. `MuseGlimmerAssistantModel.forward` builds its masks from
    `inputs_embeds=noise_embeds`, so the default mask is 4 wide and the reference raises on
    the add. Passing a 2D mask of length 16 fixes that -- but only with `use_cache=False`:
    measured 2026-08-11, `create_bidirectional_sliding_window_mask` takes `kv_length` from
    `past_key_values` when one is present, and a freshly built `DFlashCache` reports 0, so
    the 16-wide mask comes back 4 wide again and the same error returns.

    A cacheless first step is right for this anchor either way -- section 11 says the cycle is
    one forward pass with no denoising loop, and the cache exists to carry accepted context
    across cycles, which a single step has none of. Recorded in the metadata as `use_cache`.
    """
    attn_mask = torch.ones(1, int(ctx.shape[1]) + dcfg.block_size, dtype=torch.long)
    if getattr(draft, "_anchor_force_causal", False):
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        mk = dict(
            config=dcfg,
            inputs_embeds=noise,
            attention_mask=attn_mask,
            past_key_values=None,
        )
        attn_mask = {
            "full_attention": create_causal_mask(**mk),
            "sliding_attention": create_sliding_window_causal_mask(**mk),
        }
    return attn_mask


def _run_draft_block(drafter, inputs, cap):
    """The single draft forward pass, taps installed around it and torn down after.

    Both arguments bundle things that are useless apart. `drafter` is `(mdl, draft, dcfg)`: the
    reference module exists here only to be patched on the model built from that config. `inputs`
    is `(ctx, noise)` as `_draft_inputs` returns them -- the context is the K/V prefix of that
    noise block and means nothing without it.
    """
    mdl, draft, dcfg = drafter
    ctx, noise = inputs
    taps = Taps(cap)
    taps.step = 0
    handles = _install_draft_taps(mdl, draft, taps)
    attn_mask = _draft_mask(draft, dcfg, ctx, noise)
    dout = draft(
        noise_embeds=noise,
        context_hidden_states=ctx,
        attention_mask=attn_mask,
        use_cache=False,
    )
    cap.add("draft.last_hidden", dout.last_hidden_state)
    if taps.attend_calls != dcfg.num_hidden_layers:
        raise AssertionError(
            f"draft attend tap fired {taps.attend_calls}x, expected {dcfg.num_hidden_layers}"
        )
    for h in handles:
        h.remove()
    taps.close()
    return dout


def _install_draft_taps(mdl, draft, taps):
    cap = taps.cap
    orig_attend = mdl.eager_attention_forward

    def attend(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        i = taps.attend_calls
        taps.attend_calls += 1
        out, weights = orig_attend(
            module, query, key, value, attention_mask, scaling, dropout, **kw
        )
        p = f"draft.L{i}"
        # Q is the block only; K/V span context+block. That length mismatch inside one call is the
        # drafter's defining shape, and the golden has to carry both to pin it.
        cap.add(f"{p}.attend.q", query)
        cap.add(f"{p}.attend.k", key)
        cap.add(f"{p}.attend.v", value)
        if attention_mask is not None:
            cap.add(f"{p}.attend.mask", (attention_mask > -1.0).to(torch.float32))
        cap.add(f"{p}.attend.out", out)
        return out, weights

    taps.patch(mdl, "eager_attention_forward", attend)

    handles = [draft.encoder.register_forward_hook(_draft_hook(cap, "draft.encoder.out"))]
    for li, layer in enumerate(draft.layers):
        handles.append(
            layer.input_layernorm.register_forward_hook(
                _draft_hook(cap, f"draft.L{li}.input_layernorm.out")
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                _draft_hook(cap, f"draft.L{li}.post_attention_layernorm.out")
            )
        )
        handles.append(layer.mlp.register_forward_hook(_draft_hook(cap, f"draft.L{li}.mlp.out")))
    handles.append(draft.norm.register_forward_hook(_draft_hook(cap, "draft.final_norm.out")))
    return handles


def weights_capture(model):
    """The tiny model's WHOLE parameter set, for G3's comparison against the reference.

    **Extended 2026-08-13 from `attn.gate_proj` alone**, which was S3 item 3's need. The previous
    docstring specified this change and its price, so this is the deliberate decision it asked
    for rather than accretion:

        "Only `gate_proj` today. Items 4-5 and G3 will want q/k/v/o and the MLP; extend this
        rather than starting a second file, and note that the whole tiny model is ~475k floats,
        so exporting all of it is ~1.9 MB per salt -- a decision to make deliberately, not by
        accretion."

    **What it buys is the only comparison that is against MUSE GLIMMER.** Every other Glimmer
    gate in rivoli scores either a kernel against these captures, or the engine against a host
    reference transcribed from `glimmer-architecture.md` -- and a transcription shares its
    author's misreadings with the engine, so a defect written into both sides passes. With the
    parameters exported, rivoli can build a checkpoint from THEM, run its own loop, and compare
    logits to `tN.logits`: the reference model's own output, from the reference's own weights.

    The keeper of the old docstring's warning: `gate_proj` is not recoverable from the captures
    (72 -> 48, 18 rows seen, underdetermined 4x), and neither is any other projection. That is
    why this dump exists at all rather than a recover-and-predict gate.

    Still a SEPARATE file from the golden, unchanged: the goldens' bytes and their four pinned
    FNVs do not move, which is what lets `tests/glimmer-anchor.sh` prove this change is additive
    by regenerating and comparing.

    Names are the rivoli-side tensor names (`GLIMMER_LAYER_TENSORS` plus the three globals), not
    the reference's module paths, so the Rust side can build a checkpoint without a second
    mapping table to keep in step.
    """
    cap = Capture()
    lm = model.model.language_model
    cap.add("model.language_model.embed_tokens.weight", lm.embed_tokens.weight)
    cap.add("model.language_model.norm.weight", lm.norm.weight)
    cap.add("lm_head.weight", model.lm_head.weight)
    for li, layer in enumerate(lm.layers):
        a, m = layer.self_attn, layer.mlp
        for name, t in (
            ("input_layernorm", layer.input_layernorm.weight),
            ("post_attention_layernorm", layer.post_attention_layernorm.weight),
            ("pre_feedforward_layernorm", layer.pre_feedforward_layernorm.weight),
            ("post_feedforward_layernorm", layer.post_feedforward_layernorm.weight),
            ("self_attn.q_proj", a.q_proj.weight),
            ("self_attn.k_proj", a.k_proj.weight),
            ("self_attn.v_proj", a.v_proj.weight),
            ("self_attn.o_proj", a.o_proj.weight),
            ("self_attn.gate_proj", a.gate_proj.weight),
            ("mlp.gate_proj", m.gate_proj.weight),
            ("mlp.up_proj", m.up_proj.weight),
            ("mlp.down_proj", m.down_proj.weight),
        ):
            cap.add(f"model.language_model.layers.{li}.{name}.weight", t)
        # The old name, kept so `tests/glimmer_gate.rs` does not have to move in the same commit
        # that changes what this function exports. Delete it when that file reads the new name.
        cap.add(f"L{li}.attn.gate_proj.weight", a.gate_proj.weight)
    return cap


def _draft_hook(cap, name):
    def fn(_mod, _args, out):
        cap.add(name, out)

    return fn


# ---------------------------------------------------------------------------------------------


def _build_parser():
    """The CLI. `main` keeps reading THIS MODULE's `__doc__` for `--help`, which is why the split
    that moved the fixtures, the defects and the scoring out of this file left `main` behind."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("text", "draft"), default="text")
    ap.add_argument("--salt", default="glimmer-anchor-1")
    ap.add_argument("--defect", default="None", choices=sorted(DEFECTS))
    ap.add_argument("--out")
    ap.add_argument(
        "--dump-weights",
        metavar="PATH",
        help="also write the tiny model's WHOLE parameter set (see weights_capture); adds nothing "
        "to the golden, so the golden's bytes and FNV are unchanged by passing this",
    )
    ap.add_argument("--compare", nargs=2, metavar=("CLEAN", "DEFECT"))
    ap.add_argument("--by-operator", nargs=2, metavar=("A", "B"))
    # **fp64 needs no island here, and that is a real difference from K3's anchor.** K3 holds every
    # fla module at fp32 because its KDA ops are triton kernels that refuse double; Muse Glimmer's
    # reference is plain PyTorch throughout, so `--dtype float64` really is the whole model in
    # double and the floor it yields covers every operator at once. The weights are unaffected:
    # `init_weights` draws into an explicit f32 buffer and widens, so an fp64 run sees numerically
    # IDENTICAL weights and differs only in accumulation -- which is the entire point.
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--no-preflight", action="store_true")
    return ap


def _check_run_args(ap, a):
    """The refusals a generating run has to pass, in the order they were written.

    Order is behaviour here, because every `ap.error` exits: `--out` is reported before the
    mode/defect mismatch, which is reported before the `--dump-weights` trap. Only the nesting of
    that last predicate moved -- it is the same test, named.
    """
    allowed = TEXT_DEFECTS if a.mode == "text" else DRAFT_DEFECTS
    if a.defect not in allowed:
        ap.error(f"--defect {a.defect} does not apply to --mode {a.mode}")
    # Beside the other argument-compatibility refusal, and NOT after the run: this check used to sit
    # below `write_golden`, so `--defect X --dump-weights` built the model, ran all seven steps,
    # wrote a complete golden, and only then exited 2 -- aborting `glimmer-anchor.sh` mid-sweep with
    # a good-looking artefact already on disk. Review, 2026-08-13.
    weights_would_be_a_trap = a.mode != "text" or a.defect != "None"
    if a.dump_weights and weights_would_be_a_trap:
        ap.error("--dump-weights is for the clean text model; a defect run's weights are a trap")
    if not a.no_preflight:
        preflight_env()


def _metadata(a, model, note, extra):
    """The golden's self-describing header: what ran, under which environment, at which widths.

    A golden that hides what produced it is worse than no golden -- so each declared deviation in
    this module's docstring has a key here, and `preflight_env` reads the versions back OUT of
    these bytes rather than out of a constant.
    """
    cfg = model.config
    tiny = cfg.text_config.to_dict() if hasattr(cfg, "text_config") else cfg.to_dict()
    meta = [
        ("model", "muse-glimmer"),
        ("mode", a.mode),
        ("salt", a.salt),
        ("defect", a.defect),
        ("defect_note", note),
        ("driver", pathlib.Path(__file__).name),
        ("entry_point", type(model).__name__),
        ("dtype", str(torch.get_default_dtype())),
        ("attn_implementation", "eager"),
        ("tiny_config", json.dumps(tiny, sort_keys=True, default=str)),
    ]
    meta += [(k, str(v)) for k, v in sorted(environment().items())]
    meta += [(k, str(v)) for k, v in sorted(extra.items())]
    return meta


def main():
    ap = _build_parser()
    a = ap.parse_args()

    if a.compare:
        compare(*a.compare)
        return
    if a.by_operator:
        by_operator(*a.by_operator)
        return
    if not a.out:
        ap.error("--out is required unless --compare or --by-operator is given")
    # Before the model is built: the reference reads the default dtype at construction time for
    # every buffer it allocates, including the RoPE table.
    torch.set_default_dtype(getattr(torch, a.dtype))
    _check_run_args(ap, a)

    torch.manual_seed(0)  # nothing here should consume it; a stray draw would be a bug, not noise
    cap = Capture()
    runner = run_text if a.mode == "text" else run_draft
    model, note, extra = runner(a.salt, a.defect, cap)

    meta = _metadata(a, model, note, extra)
    n = write_golden(a.out, meta, cap)
    print(
        f"{a.out}: {n} B, {len(cap.floats)} float tensors, {len(cap.ints)} int tensors, "
        f"defect={a.defect}"
    )
    if a.dump_weights:
        w = weights_capture(model)
        wn = write_golden(a.dump_weights, meta, w)
        print(f"{a.dump_weights}: {wn} B, {len(w.floats)} weight tensors")


if __name__ == "__main__":
    main()
