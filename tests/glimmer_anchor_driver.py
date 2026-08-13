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
import hashlib
import json
import math
import pathlib
import struct
import sys

import torch

# Model-bound and matched by the reader in `src/v4oracle/golden.rs`. The container is the one V4
# and K3 already use; only these eight bytes say who wrote the file.
MAGIC = b"RIVGLGLD"

# The prompt is generated from the salt rather than written here, so a second draw is a second
# prompt as well as a second set of weights -- one draw cannot show that a property is a fact about
# the arithmetic rather than about the numbers it landed on.
PROMPT_LEN = 12
DECODE_STEPS = 6

# ---------------------------------------------------------------------------------------------
# The tiny configs. Every field is either the real one or a width, and the comment says which.


def tiny_text_config(cfg_mod):
    """The target's text config: real structure, toy widths.

    `num_hidden_layers` 8 is chosen so `__post_init__`'s "every 4th counted backward from the last"
    lands on layers 3 and 7 -- giving the real `[w, w, w, full]` pattern twice, a full layer that is
    NOT the last, and a last layer that IS full. Layer 0 is sliding, which is the real model too.
    """
    return cfg_mod.MuseGlimmerTextConfig(
        vocab_size=61,  # width (real 202048); prime, so it collides with nothing
        hidden_size=72,  # width (real 6656); NOT num_heads*head_dim, as in the real model
        intermediate_size=216,  # width (real 19968); keeps the real 3x hidden ratio
        num_hidden_layers=8,  # width (real 52); see docstring -- the pattern is real
        num_attention_heads=6,  # width (real 32)
        num_key_value_heads=2,  # width (real 2); group 3, which is neither 1 nor the head count
        head_dim=8,  # width (real 128)
        hidden_activation="silu",  # REAL
        max_position_embeddings=256,  # width (real 131072)
        rms_norm_eps=1e-5,  # REAL -- the pre-norm eps
        post_norm_eps=1e-8,  # REAL -- the post-norm eps, three orders apart, and that is the trap
        qk_scale_factor=3.87,  # REAL
        output_multiplier=0.19611613513818404,  # REAL, see the module docstring
        final_logit_softcapping=20.0,  # REAL
        sliding_window=4,  # width (real 2048); small enough for 18 positions to cross it
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},  # REAL
        attention_bias=False,  # REAL
        attention_dropout=0.0,  # REAL
        tie_word_embeddings=False,  # REAL
        bos_token_id=1,  # shrunk into the tiny vocab
        eos_token_id=2,  # shrunk into the tiny vocab
        pad_token_id=None,  # REAL -- the released config has none
    )


def tiny_vision_config(cfg_mod):
    """Constructed and never run. Toy widths throughout; nothing here reaches a golden."""
    return cfg_mod.MuseGlimmerVisionConfig(
        patch_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=16,
        patch_temporal=2,
        merge_size=2,
    )


def tiny_draft_config(cfg_mod, text_cfg):
    """The DFlash drafter, whose structure differs from the target's in almost every respect.

    Real and kept: 5 layers, all `sliding_attention`, plain two-norm pre-norm layers, WEIGHTED
    qk-norms, bidirectional attention, and a KV group count that is **not** the target's -- the
    target is 6Q/2KV here (group 3), so the drafter takes 6Q/3KV (group 2). The real pair is 32/2
    against 32/8; what matters is that the two group counts differ, because a port that reuses the
    target's attention shape silently passes when they agree.

    `hidden_size` must equal the target's: the drafter borrows the target's embedding matrix and
    lm_head, and `encoder.fc` consumes `len(target_layer_ids) * hidden`.
    """
    return cfg_mod.MuseGlimmerAssistantConfig(
        hidden_size=text_cfg.hidden_size,  # forced equal -- see docstring
        intermediate_size=216,  # width
        num_hidden_layers=5,  # REAL
        num_attention_heads=6,  # width
        num_key_value_heads=3,  # width; group 2, deliberately not the target's 3
        head_dim=8,  # width
        rms_norm_eps=1e-5,  # REAL
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},  # REAL
        max_position_embeddings=256,  # width
        sliding_window=4,  # width, matched to the target's tiny window
        attention_dropout=0.0,  # REAL
        hidden_act="silu",  # REAL
        block_size=4,  # width (real 16) -- one anchor token plus 3 masks
        mask_token_id=60,  # shrunk into the tiny vocab (real 201818)
        # REAL SHAPE: five layer ids spread across the target's depth, the last one at depth-3
        # exactly as [1, 13, 25, 37, 49] sits in 52. Values are scaled to 8 layers.
        target_layer_ids=[0, 1, 3, 5, 7],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


# ---------------------------------------------------------------------------------------------
# Capture.


class Capture:
    """Named tensors on their way past, in emission order.

    Float and int tensors are kept apart because the container does: a step index and a token id
    are not values a tolerance applies to, and storing them as floats would invite one.
    """

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
# Deterministic weights and prompt.


def _gen(name, salt):
    """A generator seeded by the parameter's NAME, not by its position in the module tree.

    One global seed would make every golden depend on construction order, so adding a capture or
    reordering a module would move numbers that nothing about the model changed. Keyed by name, a
    parameter's values are a property of its name alone.
    """
    h = hashlib.sha256(f"{salt}/{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def init_weights(model, salt):
    """Fill every parameter deterministically, in two families.

    **The centered norms are the reason this is not one uniform draw.** `MuseGlimmerTextCenteredRMSNorm`
    stores `w` and applies `(1 + w)`, so its weights are initialised near ZERO, not near one -- a
    centered norm filled near 1.0 doubles every activation it touches and a golden of exponentially
    growing numbers agrees with nothing. The drafter's plain `MuseGlimmerAssistantRMSNorm` and the
    target's final `MuseGlimmerRMSNorm` apply `x * w` and take the near-one draw. Telling them apart
    by module TYPE rather than by name is deliberate: both are called `...norm.weight`.
    """
    centered = set()
    for mod_name, mod in model.named_modules():
        if type(mod).__name__.endswith("CenteredRMSNorm"):
            centered.add(mod_name)
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = _gen(name, salt)
            flat = torch.empty(p.numel(), dtype=torch.float32)
            owner = name.rsplit(".", 1)[0]
            if owner in centered:
                flat.uniform_(-0.2, 0.2, generator=g)  # applied as (1 + w)
            elif "norm" in owner.split(".")[-1]:
                flat.uniform_(0.8, 1.2, generator=g)  # applied as (x * w)
            else:
                flat.uniform_(-0.08, 0.08, generator=g)
            p.copy_(flat.view(p.shape).to(p.dtype))
        # `post_init` zeroes the `padding_idx` row and the loop above just overwrote it. Put it
        # back: the zeroing is the reference's own behaviour.
        #
        # **The drafter raises here rather than returning None**, because it owns no embedding at
        # all -- it borrows the target's (section 11). That is a documented structural fact, so it
        # is caught rather than worked around, and a DIFFERENT failure would still propagate.
        try:
            emb = model.get_input_embeddings()
        except NotImplementedError:
            emb = None
        if emb is not None and getattr(emb, "padding_idx", None) is not None:
            emb.weight[emb.padding_idx].zero_()


def prompt_ids(salt, vocab, n=PROMPT_LEN):
    g = _gen("prompt", salt)
    return torch.randint(0, vocab, (1, n), generator=g)


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
    """Tap the four places the target's arithmetic is visible and a module hook is not enough."""
    cap = taps.cap

    # 1. The rotary table. One table for the whole model (M:513); the per-layer NoPE decision is a
    #    flag applied at the CALL SITE, not a second table, and that is trap #1 in section 9.
    orig_rotary = mdl.MuseGlimmerTextRotaryEmbedding.forward

    def rotary(self, x, position_ids):
        cos, sin = orig_rotary(self, x, position_ids)
        taps.rotary_calls += 1
        cap.add(f"{taps.prefix()}.rope.cos", cos)
        cap.add(f"{taps.prefix()}.rope.sin", sin)
        return cos, sin

    taps.patch(mdl.MuseGlimmerTextRotaryEmbedding, "forward", rotary)

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

    # 2. `apply_rotary_pos_emb`, so the golden holds q and k on BOTH sides of the rotation. A port
    #    that gets rotate_half vs interleaved wrong differs here and nowhere earlier.
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

    # 3. `eager_attention_forward`, which sees the POST-CACHE key/value states -- that is the ring
    #    buffer's contents as the kernel must reproduce them, including eviction -- and returns the
    #    attend output BEFORE the sigmoid gate and o_proj.
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

    # Module hooks for everything that IS a module output. Cheap, and they cover the sandwich: four
    # norms per layer, the gate before o_proj, and the SwiGLU.
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
# Defects. Each is a perturbation of the REFERENCE applied after construction, whose whole purpose
# is to prove the goldens it should redden do redden and the rest stay green. A golden with no such
# run is not evidence -- it is a file.
#
# `green` is a substring rule over capture names: a capture whose name contains any of these MUST
# NOT move. An empty tuple means the defect is expected to move everything downstream of layer 0,
# and `--compare` then only asserts that something moved.


def defect_post_norm_eps_shared(mdl, model, cfg):
    """The two-eps sandwich collapsed to one. 1e-5 vs 1e-8 is three orders on the post-norms."""
    for layer in model.model.language_model.layers:
        layer.post_attention_layernorm.eps = cfg.rms_norm_eps
        layer.post_feedforward_layernorm.eps = cfg.rms_norm_eps
    return "post-norms use rms_norm_eps instead of post_norm_eps"


def defect_norm_not_centered(mdl, model, cfg):
    """`x * w` where the reference does `x * (1 + w)` -- the other half of the sandwich trap."""
    orig = mdl.MuseGlimmerTextCenteredRMSNorm.forward

    def plain(self, x):
        out = self._norm(x.float()) * self.weight.float()
        return out.type_as(x)

    mdl.MuseGlimmerTextCenteredRMSNorm.forward = plain
    return f"centered norms applied as x*w (was {orig.__name__})"


def defect_qk_scale_on_k(mdl, model, cfg):
    """`qk_scale_factor` applied to K as well as Q. Q alone is scaled (M:342-343)."""
    for layer in model.model.language_model.layers:
        attn = layer.self_attn
        orig_k = attn.k_proj.forward
        attn.k_proj.forward = (
            lambda x, _f=orig_k, _s=attn.qk_scale_factor: _f(x) * _s
        )  # noqa: E731
    return "K scaled by qk_scale_factor too"


def defect_qk_norm_off(mdl, model, cfg):
    """The weightless QK-norm skipped. It ships no tensor, so a port can miss it entirely."""
    for layer in model.model.language_model.layers:
        layer.self_attn.qk_norm.forward = lambda x: x
    return "qk_norm replaced by identity"


def defect_rope_interleaved(mdl, model, cfg):
    """rivoli's interleaved RoPE where the reference uses rotate_half (split-half).

    Section 6 argues these are a row permutation apart and calls the conversion UNPROVEN. This is
    the golden that will settle it: a port applying the wrong convention reddens here, on rotated
    layers only, and the NoPE layers stay green.
    """

    def interleaved(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    mdl.rotate_half = interleaved
    return "rotate_half replaced by the interleaved convention"


def defect_rope_on_nope_layers(mdl, model, cfg):
    """Every layer rotated, including the NoPE ones. Trap #1: reading the top-level rope_theta."""
    cfg.layer_rope_theta = [cfg.rope_parameters["rope_theta"]] * cfg.num_hidden_layers
    return "layer_rope_theta forced non-zero everywhere (NoPE layers rotated)"


def defect_window_off_by_one(mdl, model, cfg):
    """The sliding window one position too wide. Invisible until a sequence crosses it."""
    cfg.sliding_window += 1
    for layer in model.model.language_model.layers:
        if layer.self_attn.sliding_window is not None:
            layer.self_attn.sliding_window = cfg.sliding_window
    return f"sliding_window widened to {cfg.sliding_window}"


def defect_full_layers_slide(mdl, model, cfg):
    """Every layer sliding. The full layers are the ones that must see the whole prefix."""
    cfg.layer_types = ["sliding_attention"] * cfg.num_hidden_layers
    for layer in model.model.language_model.layers:
        layer.self_attn.is_local_attention = True
        layer.self_attn.sliding_window = cfg.sliding_window
    return "layer_types forced to sliding_attention everywhere"


def defect_gate_disabled(mdl, model, cfg):
    """The sigmoid output gate saturated to 1, i.e. forgotten.

    A large constant rather than a removed multiply: the reference's own code still runs, and the
    perturbation is in the value the gate projection produces, which is what a defect is.
    """
    for layer in model.model.language_model.layers:
        layer.self_attn.gate_proj.forward = lambda x: torch.full(
            (*x.shape[:-1], layer.self_attn.config.num_attention_heads * layer.self_attn.head_dim),
            20.0,
            dtype=x.dtype,
            device=x.device,
        )
    return "attention gate saturated (sigmoid(20) ~ 1)"


def defect_kv_broadcast_blocked(mdl, model, cfg):
    """GQA broadcast as a block repeat instead of a per-head interleave.

    Only visible at group != 1 and kv heads != 1, which is why the tiny config keeps both.
    """

    def blocked(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        return hidden_states.repeat(1, n_rep, 1, 1)

    mdl.repeat_kv = blocked
    return "repeat_kv replaced by a block repeat"


def defect_softcap_off(mdl, model, cfg):
    """The tanh softcap effectively removed, by pushing T far above the logit scale.

    **This defect must NOT change the argmax**, and that is the point of it. Section 9 records that
    the logit path is argmax-invariant, so every greedy gate in the repo is blind to it; this golden
    is the only thing that can see it, and `--compare` asserts the token ids stayed green.
    """
    cfg.final_logit_softcapping = 1e9
    return "final_logit_softcapping raised to 1e9 (softcap effectively removed)"


def defect_embed_norm_off(mdl, model, cfg):
    """The weightless embedding norm skipped. Like qk_norm, it ships no tensor."""
    model.model.language_model.embed_tokens.embed_norm.forward = lambda x: x
    return "embed_norm replaced by identity"


def defect_draft_context_unprojected(mdl, model, cfg):
    """The drafter's context projection skipped -- target hidden states fed in raw.

    `H_t = output_norm_enc(fc(concat))` is computed ONCE and shared by all 5 layers; a port that
    recomputes it per layer, or skips the norm, lands here.
    """
    model.encoder.output_norm_enc.forward = lambda x: x
    return "encoder.output_norm_enc replaced by identity"


def defect_draft_causal(mdl, model, cfg):
    """The drafter's block made causal. It is bidirectional across the block (section 11 item 5).

    **Setting `self_attn.is_causal = True` does nothing**, and the first version of this defect did
    exactly that and moved zero captures. `eager_attention_forward` never reads the flag -- causality
    lives entirely in the MASK, which the model builds with `create_bidirectional_*`. So the defect
    is to hand the model transformers' own CAUSAL builders instead, which is precisely the mistake a
    port makes by reusing the target's mask path. `run_draft` reads the flag set here.
    """
    model._anchor_force_causal = True
    return "masks built with create_causal_mask instead of create_bidirectional_mask"


# `(fn, green, extra_ok)`. `green` is a substring rule over capture names -- a capture whose name
# contains any entry MUST NOT move -- with one negated form, `"!x"`, meaning every capture NOT
# containing `x` must hold. `extra_ok` allows the defect to produce captures the clean run does not.
#
# **Almost every green set is scoped to `t0.`, and that is a fact about the model, not caution.**
# A defect that shifts the argmax changes the token fed into step 1, so from t1 onward even layer 0
# differs for a reason that has nothing to do with where the defect lives. Only step 0 -- the
# prefill, whose input is the fixed prompt -- can localise anything. The first version declared
# unscoped green sets and every one of them failed at t6 on exactly this.
DEFECTS = {
    "None": (None, (), False),
    # The two-eps trap: everything up to and including the first attention is untouched; the first
    # POST-norm is where it starts.
    "post_norm_eps_shared": (
        defect_post_norm_eps_shared,
        ("rope.", "t0.embed_norm", "t0.L0.input_layernorm", "t0.L0.attend.", "t0.L0.attn."),
        False,
    ),
    "norm_not_centered": (defect_norm_not_centered, ("rope.", "t0.embed_norm"), False),
    # Q is untouched; K is not. Both are captured on both sides of the norm, so the golden says
    # which of the two moved.
    "qk_scale_on_k": (
        defect_qk_scale_on_k,
        ("rope.", "t0.embed_norm", "t0.L0.input_layernorm", "t0.L0.qk_norm.q", "t0.L0.q."),
        False,
    ),
    "qk_norm_off": (
        defect_qk_norm_off,
        ("rope.", "t0.embed_norm", "t0.L0.input_layernorm"),
        False,
    ),
    # The table itself is convention-free; only its APPLICATION changes.
    "rope_interleaved": (
        defect_rope_interleaved,
        (
            "rope.cos",
            "rope.sin",
            "t0.embed_norm",
            "t0.L0.input_layernorm",
            "t0.L0.qk_norm",
            "t0.L0.q.pre_rope",
            "t0.L0.k.pre_rope",
        ),
        False,
    ),
    # `extra_ok`: rotating the NoPE layers makes them CALL `apply_rotary_pos_emb`, so the defect run
    # carries q/k rope captures for layers 3 and 7 that the clean run does not have at all. That
    # asymmetry IS the defect's signature and the comparison runs over the intersection.
    "rope_on_nope_layers": (
        defect_rope_on_nope_layers,
        ("rope.", "t0.embed_norm", "t0.L0.", "t0.L1.", "t0.L2."),
        True,
    ),
    # The full layers' masks are the precise localisation: a window change must not reach them.
    "window_off_by_one": (
        defect_window_off_by_one,
        (
            "rope.",
            "t0.embed_norm",
            "t0.L0.input_layernorm",
            "t0.L0.qk_norm",
            "t0.L0.q.",
            "t0.L0.k.",
            "t0.L3.attend.mask",
            "t0.L7.attend.mask",
        ),
        False,
    ),
    # Layers 0-2 precede the first full layer, so at t0 they are green in their entirety.
    "full_layers_slide": (
        defect_full_layers_slide,
        ("rope.", "t0.embed_norm", "t0.L0.", "t0.L1.", "t0.L2."),
        False,
    ),
    # Everything into and through the attend is green; only the gate and what follows it moves.
    "gate_disabled": (
        defect_gate_disabled,
        (
            "rope.",
            "t0.embed_norm",
            "t0.L0.input_layernorm",
            "t0.L0.qk_norm",
            "t0.L0.q.",
            "t0.L0.k.",
            "t0.L0.attend.",
        ),
        False,
    ),
    # The attend's INPUTS are green; only the weights and the output move. That separates a
    # broadcast bug from a projection bug, which look identical one tensor later.
    "kv_broadcast_blocked": (
        defect_kv_broadcast_blocked,
        (
            "rope.",
            "t0.embed_norm",
            "t0.L0.input_layernorm",
            "t0.L0.qk_norm",
            "t0.L0.q.",
            "t0.L0.k.",
            "t0.L0.attend.q",
            "t0.L0.attend.k_cache",
            "t0.L0.attend.v_cache",
            "t0.L0.attend.mask",
        ),
        False,
    ),
    # **The whole model is declared green except the logits themselves** -- including `emitted.ids`,
    # which is the argmax invariance section 9 records, stated as an assertion for the first time.
    # Every greedy gate in this repo is blind to this defect; this golden is not.
    "softcap_off": (defect_softcap_off, ("!logits",), False),
    "embed_norm_off": (defect_embed_norm_off, ("rope.",), False),
    # `attend.q` is green because Q comes from the block alone -- the target context enters as extra
    # K/V entries and bypasses Q entirely (section 11 item 4). That is the drafter's defining shape
    # and this is the assertion that holds a port to it.
    "draft_context_unprojected": (
        defect_draft_context_unprojected,
        (
            "draft.context_concat",
            "draft.noise_embeds",
            "draft.block_ids",
            "draft.L0.input_layernorm",
            "draft.L0.attend.q",
            "attend.mask",
        ),
        False,
    ),
    # Everything up to the attend is green: only the mask, and what reads it, moves.
    "draft_causal": (
        defect_draft_causal,
        (
            "draft.context_concat",
            "draft.noise_embeds",
            "draft.block_ids",
            "draft.encoder",
            "draft.L0.input_layernorm",
            "draft.L0.attend.q",
            "draft.L0.attend.k",
            "draft.L0.attend.v",
        ),
        False,
    ),
}

TEXT_DEFECTS = {k for k in DEFECTS if not k.startswith("draft_")}
DRAFT_DEFECTS = {"None"} | {k for k in DEFECTS if k.startswith("draft_")}


# ---------------------------------------------------------------------------------------------
# The runs.


def run_text(salt, defect, cap):
    from transformers.models.muse_glimmer import configuration_muse_glimmer as C
    from transformers.models.muse_glimmer import modeling_muse_glimmer as mdl

    text = tiny_text_config(C)
    top = C.MuseGlimmerConfig(
        text_config=text,
        vision_config=tiny_vision_config(C),
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

    text = tiny_text_config(C)
    top = C.MuseGlimmerConfig(
        text_config=text,
        vision_config=tiny_vision_config(C),
        image_token_id=59,
        video_token_id=58,
        out_hidden_size=32,
        projector_hidden_size=48,
    )
    top._attn_implementation = "eager"
    target = tgt_mdl.MuseGlimmerForConditionalGeneration(top)
    target.config.text_config._attn_implementation = "eager"
    target.eval()
    init_weights(target, salt)

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

        taps = Taps(cap)
        taps.step = 0
        handles = _install_draft_taps(mdl, draft, taps)
        # **The mask spans `context + block`, and the first draft call must be CACHELESS.**
        #
        # Q is the 4 block rows; K/V are `concat(context, block)`, 16 rows, and the two lengths
        # differ inside one call. `MuseGlimmerAssistantModel.forward` builds its masks from
        # `inputs_embeds=noise_embeds`, so the default mask is 4 wide and the reference raises on
        # the add. Passing a 2D mask of length 16 fixes that -- but only with `use_cache=False`:
        # measured 2026-08-11, `create_bidirectional_sliding_window_mask` takes `kv_length` from
        # `past_key_values` when one is present, and a freshly built `DFlashCache` reports 0, so
        # the 16-wide mask comes back 4 wide again and the same error returns.
        #
        # A cacheless first step is right for this anchor either way -- section 11 says the cycle is
        # one forward pass with no denoising loop, and the cache exists to carry accepted context
        # across cycles, which a single step has none of. Recorded in the metadata as `use_cache`.
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
    """The tiny model's `attn.gate_proj` weight, per layer, for S3 item 3.

    Item 3 scores the gate OPERAND against the `attn.gate_proj.out` captures, and that needs the
    projection itself. **It cannot be recovered from the captures**: `gate_proj` is 72 -> 48, a
    layer sees 18 rows (12 prompt + 6 decode), and 18 equations against 72 unknowns per output
    element is underdetermined by 4x — so ANY candidate operand admits a weight that fits the
    captures exactly, and a recover-and-predict gate of the shape the sandwich norms use would be
    vacuous here rather than merely weak. The norms escaped that because they are ELEMENTWISE.

    Written to a SEPARATE file, not added to `cap`: the goldens' bytes and their four pinned FNVs
    do not move, which is what lets `tests/glimmer-anchor.sh` prove this change is additive by
    regenerating and comparing. Same container format, so the Rust side reads it with the reader it
    already has.

    Only `gate_proj` today. Items 4-5 and G3 will want q/k/v/o and the MLP; extend this rather than
    starting a second file, and note that the whole tiny model is ~475k floats, so exporting all of
    it is ~1.9 MB per salt — a decision to make deliberately, not by accretion.
    """
    cap = Capture()
    for li, layer in enumerate(model.model.language_model.layers):
        cap.add(f"L{li}.attn.gate_proj.weight", layer.self_attn.gate_proj.weight)
    return cap


def _draft_hook(cap, name):
    def fn(_mod, _args, out):
        cap.add(name, out)

    return fn


# ---------------------------------------------------------------------------------------------
# The container. Same layout as `src/v4oracle/golden.rs` reads; only MAGIC differs.


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
        raise ValueError(f"{path}: not a Muse Glimmer golden (magic {buf[:8]!r})")
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


# ---------------------------------------------------------------------------------------------


def preflight_env():
    """Refuse to generate under a python env other than the one the vendored goldens were made in.

    Two venvs exist on this machine -- K3's, pinned to an older transformers that has no
    `muse_glimmer` at all, and this port's. THE PIN IS READ OUT OF THE VENDORED GOLDEN, not restated
    here: those bytes already carry the versions that produced them, so a copy in this file would be
    a third statement of the same fact. A deliberate re-pin needs no edit here -- regenerate, re-vendor,
    and the new bytes are the new pin.
    """
    import numpy
    import transformers

    live = environment()
    vendored = sorted(pathlib.Path(__file__).parent.glob("glimmer-anchor-*.bin"))
    if not vendored:
        print("glimmer-anchor: no vendored golden to check this env against", file=sys.stderr)
        return
    pinned = read_golden(vendored[0])[0]
    drift = [
        f"{k}: golden says {pinned.get(k)!r}, this env has {v!r}"
        for k, v in live.items()
        if k in pinned and pinned[k] != v
    ]
    if drift:
        raise SystemExit(
            f"this python env is not the one that produced {vendored[0].name}:\n  "
            + "\n  ".join(drift)
            + f"\n(numpy {numpy.__version__}, transformers {transformers.__version__})"
        )


def environment():
    """The versions that produced a golden, read from the installed packages.

    `transformers_commit` comes from pip's own `direct_url.json` rather than from a constant: the
    package is installed from a git URL at a pinned revision, so the commit is a fact about the
    environment and restating it here would be a second copy to drift.
    """
    import numpy
    import transformers

    commit = "unknown"
    for d in pathlib.Path(transformers.__file__).parent.parent.glob("transformers-*.dist-info"):
        p = d / "direct_url.json"
        if p.exists():
            commit = json.loads(p.read_text()).get("vcs_info", {}).get("commit_id", "unknown")
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_commit": commit,
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------------------------
# Per-operator scoring. A tolerance is a property of an OPERATOR, and `compare` above cannot state
# one: it groups by capture name to ask *where* a defect lives, which is a different question from
# *how much* arithmetic disagreement a correct kernel is allowed.


def operator_of(name):
    """The kernel a capture belongs to, keyed to `glimmer-port.md` §S2's five items.

    Buckets are the port's own decomposition, not the reference's module tree, because the number
    this feeds is a threshold a rivoli kernel will be scored against. `attend` is S2 item 1,
    `rope` item 2, `gate`/`o_proj` item 3, `mlp` item 4, `logits` item 5; `norm` and `qk_norm` are
    S3's, and `ids` is exact-or-not by construction.

    **`pre_rope` is deliberately NOT in `rope`.** It is the projection's output, so folding it in
    would price the projection's accumulated error as the rotation's — and the rotation is the
    cheaper operator of the two, so the floor would come out too generous in exactly the direction
    that hides a bug. Same reason `attend.q` and the two caches sit in `attend`: they are that
    kernel's inputs, and an input that already disagrees bounds what its output can prove.
    """
    parts = name.split(".")
    rest = ".".join(parts[2:]) if len(parts) > 2 and parts[1].startswith("L") else ".".join(parts[1:])
    if name in ("prompt.ids", "emitted.ids", "layer_is_roped", "layer_is_sliding"):
        return "ids"
    if rest.startswith("attend."):
        return "attend"
    if rest.startswith("qk_norm."):
        return "qk_norm"
    if rest.endswith(".roped"):
        return "rope"
    if rest.startswith("rope."):
        return "rope_table"
    if rest.endswith(".pre_rope"):
        return "proj"
    if rest.startswith("attn.gate_proj"):
        return "gate"
    if rest.startswith("attn.o_proj"):
        return "o_proj"
    if rest.startswith("mlp."):
        return "mlp"
    if rest == "logits":
        return "logits"
    if rest.startswith("embed_norm"):
        return "embed_norm"
    if rest.startswith("final_norm"):
        return "final_norm"
    if "layernorm" in rest:
        return "norm"
    raise SystemExit(f"no operator bucket for capture {name!r} -- classify it before scoring")


def by_operator(a_path, b_path):
    """Per-operator agreement between two goldens: the shape a tolerance has to be stated in.

    Two uses, and a tolerance needs both numbers. `fp32 vs fp64` gives the **floor** no correct
    fp32 implementation can beat; `None vs <defect>` gives the **signal** a threshold has to stay
    under. `anchor.md`'s table is those two per operator, and the policy follows from their ratio.

    **The floor this reports is contaminated from below, and by a known amount.** `Capture.add`
    stores f32 because the container does (`golden.rs` reads f32), so an fp64 run is rounded on the
    way out and every comparison carries one f32 rounding, relative 2^-24 = 6.0e-8. A bucket whose
    floor lands within an order of magnitude of that is measuring the container, not the
    arithmetic, and must not be turned into a threshold -- widen the container first. Buckets well
    above it are unaffected: the fp32 run's own accumulated error dominates.
    """
    meta_a, ta = read_golden(a_path)
    meta_b, tb = read_golden(b_path)
    # **Skipped rather than refused, and counted in its own column.** Three defects change the
    # capture SET or a shape -- `window_off_by_one` and `full_layers_slide` resize the masks,
    # `rope_on_nope_layers` makes 56 captures exist that do not exist cleanly -- and those three
    # include the two that target the attend kernel hardest. An earlier version raised on any
    # mismatch, which silently dropped exactly the rows S2 item 1 needed: the sweep produced no
    # `attend` line for them at all, and a reader comparing defects would have concluded the
    # window defect leaves attention alone. A shape that changed is not a value that can be
    # subtracted; it is also not nothing, so it is reported.
    per, skipped = {}, {}
    for name, (shape, va) in ta.items():
        op = operator_of(name)
        if name not in tb or tb[name][0] != shape:
            skipped[op] = skipped.get(op, 0) + 1
            continue
        vb = tb[name][1]
        n, moved, rel = per.get(op, (0, 0, 0.0))
        # Scaled by the reference side's own magnitude, so this is a RELATIVE error and comparable
        # across operators of different widths. Non-finite pairs are skipped rather than folded in:
        # python's `max` silently discards a NaN unless it comes first, which would read as
        # agreement.
        scale = max((abs(v) for v in va if math.isfinite(v)), default=0.0) or 1e-30
        d = max(
            (abs(x - y) for x, y in zip(va, vb) if math.isfinite(x) and math.isfinite(y)),
            default=0.0,
        )
        per[op] = (n + 1, moved + (1 if d > 0 else 0), max(rel, d / scale))
    if not per:
        raise SystemExit(
            f"{a_path} and {b_path} share no comparable tensor at all -- these are not two runs "
            "of the same harness"
        )
    print(f"# {meta_b.get('defect')}/{meta_b.get('dtype')} vs {meta_a.get('defect')}/{meta_a.get('dtype')}")
    print("operator\ttensors\tdiffering\tmax_rel\tskipped")
    for op in sorted(set(per) | set(skipped)):
        n, moved, rel = per.get(op, (0, 0, 0.0))
        print(f"{op}\t{n}\t{moved}\t{rel:.3e}\t{skipped.get(op, 0)}")
    extra = sorted(set(tb) - set(ta))
    if extra:
        print(f"# {len(extra)} captures exist only in B, e.g. {extra[:3]}")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("text", "draft"), default="text")
    ap.add_argument("--salt", default="glimmer-anchor-1")
    ap.add_argument("--defect", default="None", choices=sorted(DEFECTS))
    ap.add_argument("--out")
    ap.add_argument(
        "--dump-weights",
        metavar="PATH",
        help="also write the tiny model's gate_proj weights (see weights_capture); adds nothing "
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

    allowed = TEXT_DEFECTS if a.mode == "text" else DRAFT_DEFECTS
    if a.defect not in allowed:
        ap.error(f"--defect {a.defect} does not apply to --mode {a.mode}")
    if not a.no_preflight:
        preflight_env()

    torch.manual_seed(0)  # nothing here should consume it; a stray draw would be a bug, not noise
    cap = Capture()
    runner = run_text if a.mode == "text" else run_draft
    model, note, extra = runner(a.salt, a.defect, cap)

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
    n = write_golden(a.out, meta, cap)
    print(
        f"{a.out}: {n} B, {len(cap.floats)} float tensors, {len(cap.ints)} int tensors, "
        f"defect={a.defect}"
    )
    if a.dump_weights:
        if a.mode != "text" or a.defect != "None":
            ap.error("--dump-weights is for the clean text model; a defect run's weights are a trap")
        w = weights_capture(model)
        wn = write_golden(a.dump_weights, meta, w)
        print(f"{a.dump_weights}: {wn} B, {len(w.floats)} weight tensors")


if __name__ == "__main__":
    main()
