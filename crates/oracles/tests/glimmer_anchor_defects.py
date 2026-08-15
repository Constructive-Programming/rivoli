"""The defect matrix: fourteen perturbations of the reference, each with its declared green set.

Split out of `glimmer_anchor_driver.py` 2026-08-15 under the 800-line cap, verbatim. It is its own
file for a reason beyond size: a defect is the only thing here that deliberately makes the reference
WRONG, and every other module in this port exists to be right. `DEFECTS` is the registry the
driver's `--defect`, `glimmer_anchor_compare.compare` and `tests/glimmer-anchor.sh`'s sweep all
read, so the list of proofs a golden must survive is stated once and derived everywhere.
"""

import torch

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
