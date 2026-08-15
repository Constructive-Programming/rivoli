"""Emit Muse-pattern anchor goldens for GLM-5.2 (glm_moe_dsa) from the FIRST-PARTY stack.

The reference is transformers' own `glm_moe_dsa` modeling code — the checkpoint at
/swarm/storage/ai/openclaw/glm52-fp8 declares `model_type: glm_moe_dsa` with NO
`auto_map`, so the in-tree class IS the shipped implementation. This driver runs it at
tiny widths with the REAL structure (dense prefix + MoE, MLA with q/kv-LoRA, interleaved
RoPE, DSA indexer with full/shared cross-layer sharing) and vendors what passed through.

    python3 glm_anchor_driver.py --salt glm-anchor-1 --out golden.bin
    python3 glm_anchor_driver.py --compare clean.bin defect.bin

Follows `glimmer_anchor_driver.py`'s contract exactly (same container discipline, same
defect-matrix semantics: a defect run must redden where it should AND hold bit-identical
where it declares green, with green sets scoped to t0 because a defect that shifts the
argmax contaminates every later step through the token it feeds back). The machinery is
re-stated rather than imported: this file must stay runnable with only the venv on the
path, and the glimmer driver's machinery carries glimmer-specific assertions.
"""

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import torch
import torch.nn.functional as F

MAGIC = b"RIVGMGLD"  # GLM golden; RIVGLGLD is Muse Glimmer's, RIVK3GLD Kimi-K3's.

# 12 > index_topk (4), so the DSA selection is REAL from mid-prompt on — the old tree's
# lesson that a dsa run shorter than index_topk exercises only the dense fast path.
PROMPT_LEN = 12
DECODE_STEPS = 6


# ---------------------------------------------------------------------------------------------
# Tiny config. Every width distinct from every other (the fixture-geometry lesson: at
# (2 heads, 1 kv, head_dim = dim) two candidate scales were THE SAME NUMBER and no
# tolerance could separate them). Trailing comments give the real GLM-5.2 value.


def tiny_config(C):
    cfg = C.GlmMoeDsaConfig(
        vocab_size=61,  # REAL: 154880
        hidden_size=48,  # REAL: 6144
        intermediate_size=96,  # REAL: 12288 (dense-layer MLP)
        moe_intermediate_size=24,  # REAL: 2048 (per expert)
        num_hidden_layers=6,  # REAL: 78
        num_attention_heads=4,  # REAL: 64
        num_key_value_heads=4,  # REAL: 64
        n_shared_experts=1,  # REAL: 1
        n_routed_experts=10,  # REAL: 256
        num_experts_per_tok=3,  # REAL: 8
        routed_scaling_factor=2.5,  # REAL: 2.5
        norm_topk_prob=True,  # REAL: True
        n_group=1,  # REAL: 1
        topk_group=1,  # REAL: 1
        kv_lora_rank=20,  # REAL: 512
        q_lora_rank=28,  # REAL: 2048
        qk_rope_head_dim=8,  # REAL: 64
        qk_nope_head_dim=14,  # REAL: 192  (qk_head 22 ≠ kv_lora 20, deliberately)
        v_head_dim=10,  # REAL: 256
        first_k_dense_replace=2,  # REAL: 3 — two dense + four sparse layers here
        index_topk=4,  # REAL: 2048 — must stay < PROMPT_LEN
        index_head_dim=16,  # REAL: 128
        index_n_heads=2,  # REAL: 32  (≠ num_attention_heads)
        rms_norm_eps=1e-5,  # REAL: 1e-5
        max_position_embeddings=256,  # REAL: 202752
        # F/S per layer. The real model derives this from freq/offset; the tiny one names
        # it so BOTH mechanisms are exercised: full indexers at 0/2/4, sharing at 1/3/5,
        # and layer 0 full (a shared layer 0 raises in the reference).
        index_topk_pattern="FSFSFS",
        attention_bias=False,
        attention_dropout=0.0,
    )
    assert cfg.indexer_types == ["full", "shared"] * 3, cfg.indexer_types
    assert cfg.mlp_layer_types == ["dense"] * 2 + ["sparse"] * 4, cfg.mlp_layer_types
    return cfg


# ---------------------------------------------------------------------------------------------
# Capture container + deterministic weights (the glimmer driver's discipline restated).


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


def _gen(name, salt):
    """Seeded by the parameter's NAME so values are a property of the name alone."""
    h = hashlib.sha256(f"{salt}/{name}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def _draw(name, numel, salt):
    g = _gen(name, salt)
    flat = torch.empty(numel, dtype=torch.float32)
    owner_leaf = name.rsplit(".", 2)[-2] if "." in name else name
    if name.endswith(".bias"):
        flat.uniform_(-0.05, 0.05, generator=g)  # LayerNorm/linear biases: near zero
    elif "norm" in owner_leaf:
        flat.uniform_(0.8, 1.2, generator=g)  # x * w norms: near one
    else:
        flat.uniform_(-0.08, 0.08, generator=g)
    return flat


def init_weights(model, salt):
    """Fill every parameter AND the router's correction-bias buffer deterministically.

    Two GLM-specific facts force the second half:
    - `GlmMoeDsaTopkRouter.weight` is ZERO-initialised by the reference; left that way,
      every expert scores identically and the top-k is an arbitrary tie — the routing
      goldens would pin tie-breaking order, not routing. The parameter loop covers it.
    - `e_score_correction_bias` is a `nn.Buffer`, not a parameter, so `named_parameters`
      never visits it — and zero bias would make the choice score equal the gate score,
      collapsing `router_bias_off` into a no-op defect. Drawn explicitly.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(_draw(name, p.numel(), salt).view(p.shape).to(p.dtype))
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                g = _gen(name, salt)
                flat = torch.empty(b.numel(), dtype=torch.float32)
                flat.uniform_(-0.3, 0.3, generator=g)
                b.copy_(flat.view(b.shape).to(b.dtype))
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "padding_idx", None) is not None:
            emb.weight[emb.padding_idx].zero_()


def prompt_ids(salt, vocab, n=PROMPT_LEN):
    g = _gen("prompt", salt)
    return torch.randint(0, vocab, (1, n), generator=g)


# ---------------------------------------------------------------------------------------------
# Taps.


class Taps:
    def __init__(self, cap):
        self.cap = cap
        self.step = None
        # Published by the attention wrap — NEVER a call counter (the glimmer lesson: rope
        # captures numbered by call index mislabelled every NoPE layer's golden).
        self.layer = None
        self.in_indexer = False
        self.rope_calls = 0
        self.attend_calls = 0
        self.router_calls = 0
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


def _norm_hook(cap, taps, what):
    def fn(_mod, _args, out):
        cap.add(f"{taps.prefix(taps.layer)}.{what}", out)

    return fn


def install_taps(mdl, model, taps):
    cap = taps.cap
    layers = model.model.layers

    # 1. Layer identity + per-layer attention I/O, published by wrapping Attention.forward.
    orig_attn = mdl.GlmMoeDsaAttention.forward

    def attn_forward(self, *a, **kw):
        # The decoder layer's pre-hook owns `taps.layer` for the WHOLE layer — attention
        # must not null it on exit, because post_attention_layernorm and the MLP hooks
        # fire after attention returns (the first run of this file mislabelled
        # `post_attn` with layer=None exactly that way).
        taps.layer = self.layer_idx
        out, weights, topk = orig_attn(self, *a, **kw)
        p = taps.prefix(self.layer_idx)
        cap.add(f"{p}.attn.out", out)
        # Sorted: the reference's topk order is an implementation detail of torch.topk
        # (sorted=False upstream in the router, indices here from topk on scores); the
        # SET is the contract a port must hit.
        cap.add_ints(f"{p}.topk_indices", sorted(topk[0, -1].tolist()))
        return out, weights, topk

    taps.patch(mdl.GlmMoeDsaAttention, "forward", attn_forward)

    # 2. The indexer boundary — sets the flag that disambiguates the TWO interleaved-rope
    # calls a full layer makes (attention's and the indexer's own).
    orig_idx = mdl.GlmMoeDsaIndexer.forward

    def idx_forward(self, *a, **kw):
        taps.in_indexer = True
        try:
            return orig_idx(self, *a, **kw)
        finally:
            taps.in_indexer = False

    taps.patch(mdl.GlmMoeDsaIndexer, "forward", idx_forward)

    # 3. Interleaved RoPE — the stated difference from DeepSeek-V3.2's half-split form.
    orig_rope = mdl.apply_rotary_pos_emb_interleave

    def rope_tap(q, k, cos, sin, *a, **kw):
        qo, ko = orig_rope(q, k, cos, sin, *a, **kw)
        taps.rope_calls += 1
        where = "index" if taps.in_indexer else "attn"
        p = f"{taps.prefix(taps.layer)}.{where}"
        cap.add(f"{p}.q.post_rope", qo)
        cap.add(f"{p}.k.post_rope", ko)
        return qo, ko

    taps.patch(mdl, "apply_rotary_pos_emb_interleave", rope_tap)

    # 4. The eager attend — inputs and outputs of the one matmul chain, so a broadcast
    # defect and a projection defect stop looking identical one tensor later.
    orig_eager = mdl.eager_attention_forward

    def eager_tap(module, query, key, value, attention_mask, scaling, **kw):
        taps.attend_calls += 1
        p = taps.prefix(taps.layer)
        out, weights = orig_eager(module, query, key, value, attention_mask, scaling, **kw)
        cap.add(f"{p}.attend.q", query)
        cap.add(f"{p}.attend.out", out)
        # The mask row for the LAST query position pins the DSA selection as arithmetic
        # (0 kept / -inf dropped), not just as an index list.
        if attention_mask is not None:
            cap.add(f"{p}.attend.mask_last_row", attention_mask[0, 0, -1, :])
        return out, weights

    taps.patch(mdl, "eager_attention_forward", eager_tap)

    # 5. The router — logits, weights, chosen experts, on every sparse layer.
    orig_router = mdl.GlmMoeDsaTopkRouter.forward

    def router_tap(self, hidden_states):
        taps.router_calls += 1
        logits, weights, idx = orig_router(self, hidden_states)
        p = taps.prefix(taps.layer)
        cap.add(f"{p}.router.logits", logits)
        cap.add(f"{p}.router.weights", weights)
        cap.add_ints(f"{p}.router.topk_last", sorted(idx[-1].tolist()))
        return logits, weights, idx

    taps.patch(mdl.GlmMoeDsaTopkRouter, "forward", router_tap)

    # 6. Module hooks — fire around whatever forward currently is (the glimmer lesson:
    # a patched class method deletes its own evidence when a defect patches the same slot).
    handles = []
    for i, layer in enumerate(layers):

        def with_layer(idx):
            def set_layer(_mod, _args):
                taps.layer = idx

            return set_layer

        handles.append(layer.register_forward_pre_hook(with_layer(i)))
        handles.append(
            layer.input_layernorm.register_forward_hook(_norm_hook(cap, taps, "norm.in"))
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                _norm_hook(cap, taps, "norm.post_attn")
            )
        )
        attn = layer.self_attn
        handles.append(
            attn.q_a_layernorm.register_forward_hook(_norm_hook(cap, taps, "q_resid"))
        )
        handles.append(
            attn.kv_a_layernorm.register_forward_hook(_norm_hook(cap, taps, "kv_latent"))
        )
        if type(layer.mlp).__name__ == "GlmMoeDsaMoE":
            handles.append(
                layer.mlp.register_forward_hook(_norm_hook(cap, taps, "moe.out"))
            )
            handles.append(
                layer.mlp.shared_experts.register_forward_hook(
                    _norm_hook(cap, taps, "shared.out")
                )
            )
            handles.append(
                layer.mlp.experts.register_forward_hook(_norm_hook(cap, taps, "experts.out"))
            )
        else:
            handles.append(layer.mlp.register_forward_hook(_norm_hook(cap, taps, "mlp.out")))
    return handles


def _assert_taps_fired(taps, cfg, step):
    """Derived from the config, not asserted as constants — the census that catches a tap
    silently detached by a defect or a transformers upgrade."""
    n = cfg.num_hidden_layers
    full = sum(1 for t in cfg.indexer_types if t == "full")
    sparse = sum(1 for t in cfg.mlp_layer_types if t == "sparse")
    want_rope = n + full  # attention on every layer + the indexer's own on full layers
    got = {
        "rope": (taps.rope_calls, want_rope),
        "attend": (taps.attend_calls, n),
        "router": (taps.router_calls, sparse),
    }
    bad = {k: v for k, v in got.items() if v[0] != v[1]}
    assert not bad, f"step {step}: taps misfired (got, want) = {bad}"


# ---------------------------------------------------------------------------------------------
# Defects. Perturbations of the reference applied after construction; `green` is a
# substring rule over capture names (scoped to t0 — localisation is only possible on the
# prefill), `extra_ok` permits new captures to exist only under the defect.

_DENSE_T0 = ("t0.L0.", "t0.L1.", "prompt.", "structure.")


def defect_router_softmax(mdl, model, cfg):
    orig = mdl.GlmMoeDsaTopkRouter.forward

    def fwd(self, hidden_states):
        hs = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hs.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.softmax(dim=-1)  # DEFECT: sigmoid is the reference
        scores_for_choice = scores + self.e_score_correction_bias
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        return router_logits, topk_weights * self.routed_scaling_factor, topk_indices

    mdl.GlmMoeDsaTopkRouter.forward = fwd
    return "router scores via softmax instead of sigmoid"


def defect_router_bias_off(mdl, model, cfg):
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.zero_()
    return "e_score_correction_bias zeroed: choice score == gate score"


def defect_router_norm_topk_off(mdl, model, cfg):
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.norm_topk_prob = False
    return "top-k weights unnormalised"


def defect_router_scaling_off(mdl, model, cfg):
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.routed_scaling_factor = 1.0
    return "routed_scaling_factor 2.5 -> 1.0"


def defect_shared_expert_off(mdl, model, cfg):
    # Zero the shared expert's down_proj rather than skipping its call: the call must
    # still happen so its capture exists (a skipped call orphans the hook and shrinks
    # the tensor set, which --compare refuses), but its contribution becomes exactly 0.
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer.mlp, "shared_experts"):
                layer.mlp.shared_experts.down_proj.weight.zero_()
    return "shared expert contribution zeroed out of the MoE sum"


def defect_rope_half_split(mdl, model, cfg):
    # DeepSeek-V3.2's non-interleaved form — THE documented difference. Uses the sibling
    # implementation transformers ships for models that split halves.
    def half_split(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        c = cos.unsqueeze(unsqueeze_dim)
        s = sin.unsqueeze(unsqueeze_dim)
        return (q * c) + (rotate_half(q) * s), (k * c) + (rotate_half(k) * s)

    mdl.apply_rotary_pos_emb_interleave = half_split
    return "half-split rope where the reference interleaves"


def defect_indexer_relu_off(mdl, model, cfg):
    orig = F.relu

    def no_relu(x, *a, **kw):
        return x

    # The indexer is the only F.relu caller in this model (SiLU activations elsewhere).
    F.relu = no_relu
    return "indexer score ReLU dropped"


def defect_indexer_select_all(mdl, model, cfg):
    for layer in model.model.layers:
        if layer.self_attn.indexer is not None:
            layer.self_attn.indexer.index_topk = 10_000
    return "index_topk raised past the sequence: DSA selects everything"


def defect_kv_norm_spans_rope(mdl, model, cfg):
    orig = mdl.GlmMoeDsaAttention.forward

    def fwd(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
            position_ids=None, prev_topk_indices=None, **kwargs):
        # DEFECT: normalise the WHOLE compressed projection, then split — the reference
        # normalises only the kv_lora half and leaves the rope slice raw.
        batch_size, seq_length = hidden_states.shape[:-1]
        whole = self.kv_a_proj_with_mqa(hidden_states)
        whole = self.kv_a_layernorm.weight * (
            whole * torch.rsqrt(whole.pow(2).mean(-1, keepdim=True) + 1e-5)
        )[..., : self.kv_lora_rank + self.qk_rope_head_dim].contiguous() if False else whole
        return orig(self, hidden_states, position_embeddings, attention_mask,
                    past_key_values, position_ids, prev_topk_indices, **kwargs)

    # Implemented as a patch of the layernorm itself: widen what it sees.
    class SpanningNorm(torch.nn.Module):
        def __init__(self, inner, rope_dim):
            super().__init__()
            self.inner = inner
            self.rope_dim = rope_dim

        def forward(self, x):
            pad = torch.zeros(*x.shape[:-1], self.rope_dim, dtype=x.dtype)
            wide = torch.cat([x, pad], dim=-1)
            var = wide.pow(2).mean(-1, keepdim=True)
            out = x * torch.rsqrt(var + self.inner.variance_epsilon)
            return self.inner.weight * out.to(x.dtype)

    for layer in model.model.layers:
        a = layer.self_attn
        a.kv_a_layernorm = SpanningNorm(a.kv_a_layernorm, cfg.qk_rope_head_dim)
    return "kv_a_layernorm variance computed as if the rope slice were in the row"


def defect_q_a_norm_off(mdl, model, cfg):
    for layer in model.model.layers:
        a = layer.self_attn
        a.q_a_layernorm.weight.data.fill_(1.0)
        a.q_a_layernorm.variance_epsilon = 0.0
        # Identity-ish is not identity; make it exact by replacing forward.
        a.q_a_layernorm.forward = lambda x: x
    return "q_a_layernorm bypassed"


def defect_expand_kv_swapped_split(mdl, model, cfg):
    orig = mdl.GlmMoeDsaAttention.expand_kv

    def expand_kv(self, kv_nope, k_rot):
        batch_size, _, seq_length, _ = kv_nope.shape
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        kv = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
        # DEFECT: value first, k_nope second — the 14/10 split read backwards.
        value_states, k_nope = torch.split(kv, [self.v_head_dim, self.qk_nope_head_dim], dim=-1)
        k_rot = k_rot.expand(-1, kv.shape[1], -1, -1)
        key_states = kv.new_empty(*kv.shape[:-1], self.qk_nope_head_dim + self.qk_rope_head_dim)
        key_states[..., : self.qk_nope_head_dim].copy_(
            k_nope[..., : self.qk_nope_head_dim]
            if k_nope.shape[-1] >= self.qk_nope_head_dim
            else F.pad(k_nope, (0, self.qk_nope_head_dim - k_nope.shape[-1]))
        )
        key_states[..., self.qk_nope_head_dim :].copy_(k_rot)
        return key_states, value_states

    mdl.GlmMoeDsaAttention.expand_kv = expand_kv
    return "expand_kv split order swapped (value taken where k_nope belongs)"


def defect_attn_scale_unscaled(mdl, model, cfg):
    for layer in model.model.layers:
        layer.self_attn.scaling = 1.0
    return "attention scaling 1/sqrt(qk_head_dim) dropped"


def defect_indexer_share_off(mdl, model, cfg):
    """Structural: the cross-layer top-k SHARING is the mechanism under test — a port that
    runs a full indexer on every layer instead of reusing the previous full layer's
    selection. Chosen over `first_k_dense_off` deliberately: that defect REPLACES the
    dense layers' captures (mlp.out vanishes, moe.* appears), and the compare contract
    refuses a shrinking tensor set — a defect must only ADD captures (extra_ok) or move
    existing ones. This one grows indexers on the shared layers, so the clean capture set
    survives intact and the new index-side rope pairs appear beside it."""
    shared = [i for i, t in enumerate(cfg.indexer_types) if t == "shared"]
    for i in shared:
        cfg.indexer_types[i] = "full"
        model.model.layers[i].self_attn = mdl.GlmMoeDsaAttention(cfg, i)
    # Deterministic by NAME, so re-running the whole-model init leaves every existing
    # parameter exactly as it was and draws only the new indexers' weights.
    init_weights(model, _CURRENT_SALT[0])
    return "shared indexer layers rebuilt as full: no cross-layer top-k reuse"


DEFECTS = {
    "None": (None, (), False),
    # Router family: the dense prefix (L0/L1) is upstream of every router and must hold.
    "router_softmax": (defect_router_softmax, _DENSE_T0, False),
    "router_bias_off": (defect_router_bias_off, _DENSE_T0, False),
    "router_norm_topk_off": (defect_router_norm_topk_off, _DENSE_T0, False),
    "router_scaling_off": (defect_router_scaling_off, _DENSE_T0, False),
    "shared_expert_off": (defect_shared_expert_off, _DENSE_T0, False),
    # Attention family: moves layer 0 itself, so only the prompt/structure hold at t0.
    "rope_half_split": (defect_rope_half_split, ("prompt.", "structure."), False),
    "kv_norm_spans_rope": (defect_kv_norm_spans_rope, ("prompt.", "structure."), False),
    "q_a_norm_off": (defect_q_a_norm_off, ("prompt.", "structure."), False),
    "expand_kv_swapped_split": (defect_expand_kv_swapped_split, ("prompt.", "structure."), False),
    "attn_scale_unscaled": (defect_attn_scale_unscaled, ("prompt.", "structure."), False),
    # Indexer family: everything before the selection holds on layer 0 at t0 — q_resid,
    # kv_latent, the norms, and both rope pairs are computed before the mask lands.
    "indexer_relu_off": (
        defect_indexer_relu_off,
        ("t0.L0.q_resid", "t0.L0.kv_latent", "t0.L0.norm.in", "t0.L0.attend.q",
         "prompt.", "structure."),
        False,
    ),
    "indexer_select_all": (
        defect_indexer_select_all,
        ("t0.L0.q_resid", "t0.L0.kv_latent", "t0.L0.norm.in", "t0.L0.attend.q",
         "prompt.", "structure."),
        False,
    ),
    # Structural: extra router/expert captures appear on the ex-dense layers.
    "indexer_share_off": (defect_indexer_share_off, ("prompt.",), True),
}


# ---------------------------------------------------------------------------------------------
# The run.


_CURRENT_SALT = [None]


def run(salt, defect, cap):
    _CURRENT_SALT[0] = salt
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa as C
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as mdl

    cfg = tiny_config(C)
    cfg._attn_implementation = "eager"
    model = mdl.GlmMoeDsaForCausalLM(cfg)
    model.config._attn_implementation = "eager"
    model.eval()
    init_weights(model, salt)

    fn = DEFECTS[defect][0]
    note = fn(mdl, model, cfg) if fn else ""

    taps = Taps(cap)
    handles = install_taps(mdl, model, taps)

    ids = prompt_ids(salt, cfg.vocab_size)
    cap.add_ints("prompt.ids", ids[0].tolist())
    cap.add_ints("structure.mlp_is_sparse", [int(t == "sparse") for t in cfg.mlp_layer_types])
    cap.add_ints("structure.indexer_is_full", [int(t == "full") for t in cfg.indexer_types])

    emitted = []
    past = None
    step_in = ids
    try:
        with torch.no_grad():
            for step in range(1 + DECODE_STEPS):
                taps.step = step
                taps.rope_calls = taps.attend_calls = taps.router_calls = 0
                out = model(input_ids=step_in, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                cap.add(f"t{step}.logits", logits)
                nxt = int(logits.argmax(-1))
                emitted.append(nxt)
                step_in = torch.tensor([[nxt]])
                # The census derives from cfg, which structural defects mutate in
                # place — so it applies to every run, clean or defective.
                _assert_taps_fired(taps, cfg, step)
    finally:
        for h in handles:
            h.remove()
        taps.close()
    cap.add_ints("emitted.ids", emitted)
    return cfg, note


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


def compare(a_path, b_path):
    """Score two goldens and ASSERT the perturbed one's declared green set — both halves:
    something moved AND everything declared green held bit-identical."""
    meta_a, ta = read_golden(a_path)
    meta_b, tb = read_golden(b_path)
    defect = meta_b.get("defect", "None")
    if meta_a.get("defect", "None") != "None":
        raise SystemExit(f"{a_path} is itself a defect run; A must be None")
    _fn, green, extra_ok = DEFECTS[defect]
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    if only_a or (only_b and not extra_ok):
        raise SystemExit(
            f"tensor sets differ: only in A {only_a[:5]}, only in B {only_b[:5]}"
            + ("" if extra_ok else "\n(the defect does not declare extra_ok)")
        )
    if extra_ok and not only_b:
        raise SystemExit(f"defect {defect!r} declares extra_ok but produced no extra captures")

    def is_green(name):
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
        raise SystemExit(f"defect {defect!r} moved NOTHING — it is not a defect, or it did not apply")
    if green and not any(is_green(n) for n in held):
        raise SystemExit(f"defect {defect!r} declares green captures but none of them held")
    if violations:
        raise SystemExit(
            f"{len(violations)} declared-green captures moved:\n  " + "\n  ".join(violations)
        )
    print(
        f"  {sum(is_green(n) for n in held)} declared-green captures held"
        if green
        else "  (no green set)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salt", default="glm-anchor-1")
    ap.add_argument("--defect", default="None", choices=sorted(DEFECTS))
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("CLEAN", "DEFECT"))
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--preflight-against", metavar="GOLDEN")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if args.preflight_against and not args.no_preflight:
        preflight_env(args.preflight_against)
    if not args.out:
        raise SystemExit("--out is required unless --compare")

    cap = Capture()
    cfg, note = run(args.salt, args.defect, cap)
    meta = [
        ("model", "glm_moe_dsa tiny"),
        ("salt", args.salt),
        ("defect", args.defect),
        ("defect_note", note or ""),
        ("config", json.dumps({
            "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "kv_lora_rank": cfg.kv_lora_rank, "q_lora_rank": cfg.q_lora_rank,
            "qk_rope_head_dim": cfg.qk_rope_head_dim,
            "qk_nope_head_dim": cfg.qk_nope_head_dim, "v_head_dim": cfg.v_head_dim,
            "first_k_dense_replace": cfg.first_k_dense_replace,
            "index_topk": cfg.index_topk, "index_head_dim": cfg.index_head_dim,
            "index_n_heads": cfg.index_n_heads,
            "routed_scaling_factor": cfg.routed_scaling_factor,
            "rms_norm_eps": cfg.rms_norm_eps,
        })),
        ("prompt_len", str(PROMPT_LEN)),
        ("decode_steps", str(DECODE_STEPS)),
    ] + environment()
    n = write_golden(args.out, meta, cap)
    print(f"wrote {args.out}: {n} bytes, {len(cap.floats)} float + {len(cap.ints)} int tensors")


if __name__ == "__main__":
    main()
