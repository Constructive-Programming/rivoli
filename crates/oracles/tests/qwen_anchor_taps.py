"""The taps: values a `register_forward_hook` cannot see, and how each is reached.

Four families, and each exists because the hook mechanism in `qwen_anchor_capture.py` is blind
to it:

  * **the GDN short convolution** -- `GatedDeltaNet.forward` hands `self.conv1d.weight.squeeze(1)`
    to the module-level `causal_conv1d_fn` / `causal_conv1d_update` and never calls the
    `nn.Conv1d` (MOD:477-496), so a hook on it fires zero times;
  * **the gated delta rule** -- `torch_chunk_gated_delta_rule` and
    `torch_recurrent_gated_delta_rule` are module-level functions, not modules. Everything the
    kernel S3 writes has to reproduce lives between their inputs and their outputs: the qk
    L2 norm at eps 1e-6, the `1/sqrt(d_k)` on `q` alone, the per-head scalar decay and the
    `[d_k][d_v]` state;
  * **partial RoPE** -- `apply_rotary_pos_emb` is a free function. Only the ATTENTION call is
    recorded (it is the one that passes `k`); the indexer's per-query key calls would put a
    tensor count in the golden that scales with the window, and the indexer's own output mask
    already carries their effect;
  * **the n-gram hash's own arithmetic** -- the multipliers, the 16 prime head vocabularies and
    the offsets are `nn.Buffer`s, so they are reachable, but the defects that move them have to
    be applied through the REFERENCE'S OWN `_build_layer_multipliers` and
    `_find_nth_prime_after` rather than through a re-derivation here. That is what
    `rebuild_ngram_tables` does, and it is the difference between scoring the model and scoring
    this file's reading of it.

The one place this module transcribes reference code is [`indexer_variant_forward`], and the cost
is paid explicitly: `--mode indexer-identity` runs the model twice in one process, with and
without the transcription at `variant=None`, and refuses any difference. A copy that silently
drifts from the reference is the failure mode; a copy with a bit-exact identity gate is a
substitution vehicle.
"""

import math

from qwen_anchor_capture import torch

# **Every reference symbol a tap or a defect replaces**, so a process that builds TWO models can
# start each from the pristine reference. Needed rather than tidy: `--mode indexer-identity` and
# `--mode moe-equiv` each build two, and the second `_install` wrapped the ALREADY-wrapped
# functions -- `_conv_binder` looked up `fn.__name__` and got `'inner'`. Wrapping a wrapper would
# also have double-counted every call, which the per-layer counts in `assert_captured` would have
# reported as a layer-count mismatch two steps from the cause.
_MODULE_SYMBOLS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "apply_rotary_pos_emb",
    "rotate_half",
    "l2norm",
)
_CLASS_SYMBOLS = (
    ("Qwen4ExpTextQSAIndexer", "forward"),
    ("Qwen4ExpTextPLELayer", "forward"),
    ("Qwen4ExpTextRMSNormGated", "forward"),
    ("Qwen4ExpTextRMSNorm", "forward"),
    ("Qwen4ExpTextRMSNorm", "_norm"),
    ("Qwen4ExpTextTopKRouter", "forward"),
    ("Qwen4ExpTextGatedResidual", "forward"),
)
_TORCH_SYMBOLS = ("relu", "bitwise_xor")
_PRISTINE = {}


def restore_reference(mdl_mod):
    """Put every replaced symbol back, snapshotting on the first call.

    Called at the top of the driver's `_install`, so each model built in a process sees the
    reference as shipped. `F.softplus` is on the list because `wrap_softplus` patches
    `torch.nn.functional` itself -- the model's only softplus call site, but a process-global
    edit either way, and one that must not survive into a second model.
    """
    if not _PRISTINE:
        for name in _MODULE_SYMBOLS:
            _PRISTINE[("mod", name)] = getattr(mdl_mod, name)
        for cls_name, attr in _CLASS_SYMBOLS:
            _PRISTINE[("cls", cls_name, attr)] = getattr(getattr(mdl_mod, cls_name), attr)
        for name in _TORCH_SYMBOLS:
            _PRISTINE[("torch", name)] = getattr(torch, name)
        _PRISTINE[("F", "softplus")] = mdl_mod.F.softplus
        return
    for name in _MODULE_SYMBOLS:
        setattr(mdl_mod, name, _PRISTINE[("mod", name)])
    for cls_name, attr in _CLASS_SYMBOLS:
        setattr(getattr(mdl_mod, cls_name), attr, _PRISTINE[("cls", cls_name, attr)])
    for name in _TORCH_SYMBOLS:
        setattr(torch, name, _PRISTINE[("torch", name)])
    mdl_mod.F.softplus = _PRISTINE[("F", "softplus")]


# ---------------------------------------------------------------------------------------------
# Which module is in flight.


def wrap_gdn_modules(mdl_mod, tap):
    """A forward-pre-hook per `GatedDeltaNet`, so the free-function taps below know whose call
    they are in -- and therefore see `A_log`, `dt_bias` and the conv weight.

    Not a convenience: the decay variants are stated as a RATIO against the reference's own `g`
    (see [`decay_variant`]), and a ratio needs the pre-softplus argument, which is
    `in_proj_a(x) + dt_bias` and lives nowhere a hook on the delta rule can reach.
    """
    cls = mdl_mod.Qwen4ExpTextGatedDeltaNet
    handles = []
    for m in tap.ctx["model"].modules():
        if isinstance(m, cls):
            handles.append(m.register_forward_pre_hook(lambda mod, _i, _t=tap: _enter_gdn(_t, mod)))
    return handles


def _enter_gdn(tap, module):
    tap.ctx["gdn_module"] = module
    tap.ctx["softplus"] = None


# ---------------------------------------------------------------------------------------------
# The decay, and the three wrong forms of it.


def wrap_softplus(mdl_mod, tap):
    """Stash `softplus`'s argument and result, because the decay variants are ratios of them.

    `F.softplus` appears EXACTLY ONCE in the whole model (MOD:519), so wrapping
    `torch.nn.functional.softplus` for the life of one defect run touches nothing else. Checked
    by grep against the pinned blob rather than assumed, and re-checked by
    `assert_captured`'s per-layer GDN count: a second caller would break the pairing loudly.
    """
    orig = mdl_mod.F.softplus

    def shim(x, *a, **kw):
        out = orig(x, *a, **kw)
        tap.ctx["softplus"] = (x, out)
        return out

    mdl_mod.F.softplus = shim
    tap.ctx["softplus_fn"] = orig
    return orig


def decay_variant(tap, g):
    """The reference's own `g`, rescaled into whichever wrong decay form the run asks for.

    `g = -exp(A_log) * softplus(a + dt_bias)` (MOD:519). Every variant below is stated as
    `g * (wrong(x) / softplus(x))`, with `x` the reference's own softplus argument -- so
    `A_log`, its `.float().exp()` and the `-` are never restated here. A defect that
    re-implements the formula it is testing would move the golden for reasons that have nothing
    to do with the defect, which is the cost K3's `RouterBiasInWeight` records paying.

      * `sigmoid` -- fla's GDN family and K3's KDA both ship a bounded-gate form, and softplus
        and sigmoid differ only in scale on small inputs, so the fixture needs `|a + dt_bias|`
        large enough that they separate. Measured separation is in `anchor.md`.
      * `clamped` -- KDA's `gate_lower_bound` carried into a model that has no such field. The
        clamp is on `g` itself, which is where a port that inherited the K3 launcher would put
        it.
      * `bias_outside` -- `-exp(A_log) * softplus(a) + dt_bias`: the bias read as an additive
        term on the decay rather than as an input to the softplus. One paren in Eq. 10.
    """
    form = tap.ctx.get("gdn_decay_form")
    if form is None:
        return g
    if form == "clamped":
        # -5.0 is fla's own inclusive lower bound for KDA's gate, which is the value a port
        # arriving from that family would carry (`k3:linear_attn_config.gate_lower_bound`).
        return g.clamp_min(-5.0)
    stash = tap.ctx.get("softplus")
    assert stash is not None, "the decay variant needs softplus's argument; wrap_softplus first"
    x, sp = stash
    x, sp = x.to(g.dtype), sp.to(g.dtype)
    if form == "sigmoid":
        return g * (torch.sigmoid(x) / sp)
    if form == "bias_outside":
        dt = tap.ctx["gdn_module"].dt_bias.to(g.dtype)
        # The UNWRAPPED softplus, kept by `wrap_softplus`: calling the shim here would overwrite
        # the stash this variant is reading, which is the kind of self-interference a tap that
        # doubles as an instrument invites.
        return g * (tap.ctx["softplus_fn"](x - dt) / sp) + dt
    raise AssertionError(f"unknown decay form {form}")


# ---------------------------------------------------------------------------------------------
# The gated delta rule.

GDN_OPS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
# The inputs that ARE the S3 fixture. `initial_state` is included because the recurrence is the
# decode path and a kernel that produces the right `o` from the wrong state agrees for exactly
# one token.
GDN_IN = ("query", "key", "value", "g", "beta", "initial_state")


def _gdn_layer(tap):
    """The layer this call belongs to, and the bookkeeping that makes that true.

    The nth GDN call of a forward pass is the nth `linear_attention` layer in index order, which
    the reference guarantees by iterating `self.layers` in order (MOD:1420).
    """
    ctx = tap.ctx
    which = ctx["gdn_layer_ids"][ctx["gdn_calls"]]
    ctx["gdn_calls"] += 1
    return which


def _gdn_record(tap, which, kw, out):
    if which not in tap.layers or not tap.ctx["capturing"]:
        return
    tag = f"model.layers.{which}.linear_attn.gated_delta_rule"
    for k in GDN_IN:
        if kw.get(k) is not None:
            tap.cap.add(f"{tag}.in.{k}", kw[k])
    tap.cap.add(f"{tag}.out.o", out[0])
    if out[1] is not None:
        tap.cap.add(f"{tag}.out.state", out[1])
    # The two learned per-V-head vectors, from the module in flight. Both are [48] in the real
    # checkpoint and both are LEARNED (measured ranges in `qwen-architecture.md` T17's row);
    # `__init__`'s `torch.ones` and `uniform_(0.01, 16)` are init placeholders, and a fixture
    # that omitted them could not tell a per-head gate from a per-channel one.
    module = tap.ctx["gdn_module"]
    tap.cap.add(f"{tag}.in.A_log", module.A_log)
    tap.cap.add(f"{tag}.in.dt_bias", module.dt_bias)


def _gdn_call(fn, tap, kw):
    """One wrapped delta-rule call: the decay variant, the call, the state variant, the record."""
    ctx = tap.ctx
    kw = dict(kw)
    if kw.get("g") is not None:
        kw["g"] = decay_variant(tap, kw["g"])
    which = _gdn_layer(tap)
    ctx["gdn_tags"].add(fn.__name__)
    out = fn(**kw)
    if ctx.get("gdn_transpose_out_state") and out[1] is not None:
        # `gdn_state_axis_swapped`: hand the next token a state with its last two axes swapped.
        # Invisible to every shape assertion -- `linear_key_head_dim == linear_value_head_dim`
        # in the tiny config AND in the real one (128 == 128), so the state is square either
        # way, and only its VALUES can carry the layout. `[key][value]` is what
        # `torch_recurrent_gated_delta_rule` indexes (`sum(dim=-2)` over the key axis, MOD:391)
        # and what TR p.3 pins, and it is already rivoli's coalescing order.
        out = (out[0], out[1].transpose(-1, -2).contiguous())
    sink = ctx.get("equiv_sink")
    if sink is not None:
        # `--mode gdn-equiv`: this call's output filed under `(arm, layer)`. The two recorders
        # are exclusive -- the equivalence mode runs no capture pass.
        sink.setdefault((ctx["equiv_tag"], which), []).append(out[0].detach().float().clone())
    else:
        _gdn_record(tap, which, kw, out)
    return out


def wrap_gdn_ops(mdl_mod, tap):
    """Record the delta rule's own inputs and outputs, which no module hook can see.

    Installed BEFORE any forward pass, with recording gated on `ctx["capturing"]`. K3's recorded
    reason: with the wrapper installed after a warm prefill, a kwarg-only defect perturbed
    exactly one call per layer on a recurrent state the defect had never touched, so
    `in.initial_state` was bit-identical BY CONSTRUCTION and the state-propagation claim was the
    one thing those defects could not test.
    """
    for tag in GDN_OPS:
        fn = getattr(mdl_mod, tag)
        setattr(mdl_mod, tag, _bind(fn, tap, _gdn_call))


def _bind(fn, tap, call):
    """A keyword-only closure over one reference entry point.

    A closure rather than `functools.partial`: the reference calls these ops with keyword
    arguments only, and `partial` would put `fn` into the same namespace as the kwargs.
    """

    def inner(*a, **kw):
        return call(fn, tap, dict(zip(("query", "key", "value"), a)) | kw)

    return inner


# ---------------------------------------------------------------------------------------------
# The short convolutions.

CONV_OPS = ("causal_conv1d_fn", "causal_conv1d_update")


def _conv_call(fn, tap, kw):
    ctx = tap.ctx
    which = ctx["gdn_layer_ids"][ctx["conv_calls"]]
    ctx["conv_calls"] += 1
    ctx["conv_tags"].add(fn.__name__)
    out = fn(**kw)
    if which in tap.layers and ctx["capturing"]:
        tag = f"model.layers.{which}.linear_attn.conv"
        tap.cap.add(f"{tag}.in", kw["hidden_states"])
        tap.cap.add(f"{tag}.weight", kw["weight"])
        tap.cap.add(f"{tag}.out", out)
    return out


def wrap_conv_ops(mdl_mod, tap):
    """Capture the GDN short convolution, whose `nn.Conv1d` is never called.

    The WEIGHT is captured beside the values, because tap order is a property of the weight
    layout and not of the activations: `F.conv1d` is cross-correlation with three zeros of left
    padding, so `out[t] = sum_j w[j] x[t-3+j]` and **`w[3]` multiplies the current token**. A
    fixture that recorded only inputs and outputs would let a port with a reversed kernel agree
    on a palindromic draw, which is why `gdn_conv_tap_reversed` requires a NON-palindromic one.
    """
    for tag in CONV_OPS:
        fn = getattr(mdl_mod, tag)
        setattr(mdl_mod, tag, _conv_binder(fn, tap))


# The two entry points take DIFFERENT positional orders and both call sites pass positionally
# (MOD:477-483 and MOD:490-496), so the names are recovered per function rather than per call.
# `causal_conv1d_fn` has no `conv_state`: its second positional is the weight.
_CONV_PARAMS = {
    "causal_conv1d_update": ("hidden_states", "conv_state", "weight", "bias", "activation"),
    "causal_conv1d_fn": ("hidden_states", "weight", "bias", "activation"),
}


def _conv_binder(fn, tap):
    names = _CONV_PARAMS[fn.__name__]

    def inner(*a, **kw):
        return _conv_call(fn, tap, dict(zip(names, a)) | kw)

    return inner


def wrap_ple(mdl_mod, tap):
    """Count PLE injections, so `assert_captured` can refuse a run where the host layer moved.

    `ple_layer_ids` is ONE-indexed (MOD:1202 tests `layer_idx + 1`), and a port reading it as a
    0-based index puts the injection on layer 2. This counter plus the per-layer capture is what
    makes that a scored row rather than a paragraph.
    """
    cls = mdl_mod.Qwen4ExpTextPLELayer
    orig = cls.forward

    def counted(self, *a, **kw):
        tap.ctx["ple_calls"] += 1
        return orig(self, *a, **kw)

    cls.forward = counted
    return orig


# ---------------------------------------------------------------------------------------------
# Partial RoPE.


def wrap_rope(mdl_mod, tap):
    """Record the attention's roped q and k -- the fixture for the 64-of-256 partial rotation.

    Only the call that passes `k` is recorded. `apply_rotary_pos_emb` is also called twice per
    QUERY inside the indexer (once for the query, once per block for the pooled keys), and
    recording those would make the golden's tensor count a function of the window length; the
    indexer's own output mask carries their effect, and `indexer_rope_at_block_end` is what
    prices the position they are taken at.
    """
    orig = mdl_mod.apply_rotary_pos_emb

    def shim(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
        out = orig(q, k, cos, sin, unsqueeze_dim)
        ctx = tap.ctx
        if k is not None and ctx["capturing"]:
            which = ctx["qsa_layer_ids"][ctx["rope_calls"]]
            ctx["rope_calls"] += 1
            if which in tap.layers:
                tag = f"model.layers.{which}.self_attn.rope"
                tap.cap.add(f"{tag}.in.q", q)
                tap.cap.add(f"{tag}.in.k", k)
                tap.cap.add(f"{tag}.out.q", out[0])
                tap.cap.add(f"{tag}.out.k", out[1])
        return out

    mdl_mod.apply_rotary_pos_emb = shim
    return orig


def rope_interleaved_rotate_half(x):
    """`rotate_half` with the OTHER pairing convention: `(i, i+1)` instead of `(i, i + d/2)`.

    The reference splits the rotary half in two and pairs `i` with `i + rotary_dim/2`
    (MOD:566-570). GLM and Qwen2 lineage code is interleaved in places and
    `mrope_interleaved: true` invites the confusion -- that flag is about the mRoPE SECTIONS,
    not about the pairing. `rotate_half` is the only substitution point the text path uses
    (MOD:598 and MOD:604; MOD:1728-1729 belong to the vision tower, which is never built here).
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_on_tail_dims(orig_rotate_half):
    """`apply_rotary_pos_emb` rotating the LAST `rotary_dim` dims instead of the first.

    A transcription of MOD:591-604 with one substitution, which is what the defect IS -- the
    reference's own five lines with `[..., :rotary_dim]` become `[..., -rotary_dim:]`. Requires
    `head_dim > rotary_dim` (32 > 8 here, 256 > 64 in the real model) or the row is vacuous,
    which the width audit's `rotary_dim vs head_dim/2` pair also guarantees.
    """

    def shim(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
        cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        rotary_dim = cos.shape[-1]

        def rot(t):
            head, tail = t[..., :-rotary_dim], t[..., -rotary_dim:]
            tail = (tail * cos) + (orig_rotate_half(tail) * sin)
            return torch.cat([head, tail], dim=-1)

        return (rot(q), rot(k)) if k is not None else rot(q)

    return shim


# ---------------------------------------------------------------------------------------------
# The n-gram tables, rebuilt through the reference's own functions.


def rebuild_ngram_tables(mdl_mod, model, *, seed=None, ple_layer_index=None, mode=None):
    """Move the hash by rebuilding its buffers with the REFERENCE'S OWN derivation.

    Three of the five n-gram defect rows are "the derivation was run with the wrong argument",
    and the only honest way to say that is to run it with the wrong argument:
    `_build_layer_multipliers(vocab, ngram_size, ple_layer_index, seed)` and
    `_find_nth_prime_after(base - 1, global_head_idx + 1)` are both module-level in MOD
    (MOD:986 and MOD:1009). Re-deriving the primes here would score this file's reading of
    section 5 of `qwen-architecture.md`.

    `mode="round"` is the exception and is not a wrong argument but a wrong RULE: every head
    reduced modulo `ngram_vocab_size_base` -- the round number the config field is named after
    -- instead of modulo its own prime. The offsets stay, so the 16 heads collide into
    overlapping row ranges: plausible-looking output, not a crash.
    """
    for m in model.modules():
        if type(m).__name__ != "Qwen4ExpTextNGramEmbedding":
            continue
        index = m.ple_layer_index if ple_layer_index is None else ple_layer_index
        if seed is not None or ple_layer_index is not None:
            m.layer_multipliers.copy_(
                mdl_mod._build_layer_multipliers(
                    m.unigram_vocab_size, m.ngram_size, index, m.seed if seed is None else seed
                ).to(m.layer_multipliers.device)
            )
        if ple_layer_index is not None:
            sizes = [
                mdl_mod._find_nth_prime_after(m.ngram_vocab_size_base - 1, index * m.ngram_heads + h + 1)
                for h in range(m.ngram_heads)
            ]
            _write_head_tables(m, sizes)
        if mode == "round":
            _write_head_tables(m, [m.ngram_vocab_size_base] * m.ngram_heads, keep_offsets=True)
        if mode == "order_swapped":
            # Bigram heads are 0..7 and trigram heads 8..15 (MOD:1098-1100, where
            # `start_idx = (ngram - 2) * heads_per_ngram`). Swapping the two halves of BOTH
            # tables is the "trigram first" reading, and because every head has its own prime a
            # swap changes every id.
            half = m.ngram_heads // 2
            for buf in (m.ngram_heads_vocab_sizes, m.ngram_heads_offsets):
                rolled = torch.cat([buf[half:], buf[:half]]).clone()
                buf.copy_(rolled)


def _write_head_tables(m, sizes, keep_offsets=False):
    m.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.long))
    if keep_offsets:
        return
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    m.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.long))
    _grow_table(m, total)


def _grow_table(m, total):
    """Grow the embedding when a shifted layer index needs more rows than the table has.

    **Measured, and it is a finding about the real model rather than a fixture detail.** At
    `ple_layer_index` 1 the 16 head vocabularies are the 17th..32nd primes above
    `ngram_vocab_size_base - 1` rather than the 1st..16th, so their sum EXCEEDS the padded row
    count the reference derived for index 0 -- the first golden attempt died in
    `F.embedding` with `index out of range in self`. That is true at the real widths too: index
    1's primes are 20000213..20000429 and their total is above the checkpoint's own
    320,001,536 rows, so a converter that hashed at the wrong index against a table built for
    the right one gets an INDEX ERROR, not a plausible-looking output. Loud, which is worth
    knowing; `qwen-architecture.md` section 5's re-derivation gate is what catches the
    consistent-but-wrong case.

    The extra rows repeat the existing table rather than being zeroed or drawn afresh: only this
    defect run reads them, a zero block would make the perturbed output look degenerate for a
    reason unrelated to the defect, and wrapping keeps the value distribution the fixture was
    built with.
    """
    weight = m.ngram_embedding.weight
    old_rows = weight.shape[0]
    divisor = 128
    need = -(-total // divisor) * divisor
    if need <= old_rows:
        return
    index = torch.arange(need) % old_rows
    m.ngram_embedding.weight = torch.nn.Parameter(weight.detach()[index].clone())
    m.ngram_embedding.num_embeddings = need


# ---------------------------------------------------------------------------------------------
# The indexer, and the one transcription in this file.


def indexer_variant_forward(variant, tap):
    """`QSAIndexer.forward` (MOD:632-717) transcribed, with exactly one substitution per variant.

    **This is the only transcription of reference arithmetic in the anchor, and it is gated.**
    Three of the six indexer defect rows -- the RoPE position the pooled keys are taken at, the
    order of pooling against normalisation, and the axis the per-head ReLU scores are summed
    over -- live inside this one 40-line function and cannot be reached by substituting a free
    function or setting an attribute. `--mode indexer-identity` runs the model with
    `variant=None` against the reference in one process and refuses any difference, so a copy
    that drifts from a future revision is loud rather than silent.

    Variants:
      * `rope_at_block_end` -- `group_starts = block_token_indices[:, -1]`. TR Eq. 13-14 and
        MOD:683-688 both take the block's FIRST token, whose stated reason is that pooling
        before roping "avoids averaging token representations with different rotary phases".
      * `rope_before_pool` -- rope the raw keys, then pool. The order the reason above forbids.
      * `relu_after_sum` -- `relu(sum_h <q_h, kbar_b>)` for `sum_h relu(<q_h, kbar_b>)`. TR
        Eq. 15 rectifies each head's agreement BEFORE aggregating, so a negative head cannot
        cancel a positive one; rectifying after lets it. See
        `qwen_anchor_defects.indexer_relu_after_sum` for the two readings this replaced and why.
      * `blocks_strided` -- blocks formed by STRIDE rather than contiguously:
        `local[: ncb*r].view(r, ncb).t()` for `.view(ncb, r)`. Same shapes, same block count,
        different membership, so it prices the reshape order MOD:673-675 fixes.

    **This body is on the critical path of every golden, `None` included, and that is deliberate.**
    It records four tensors the reference computes and discards -- the pooled block keys, the
    block start positions, the per-block scores and the selected block ids -- and without them the
    indexer's only observable is its final MASK, which is a discrete selection. Measured: three
    rows (`indexer_rope_at_block_end`, `indexer_max_over_heads`, `rope_interleaved_pairs`)
    perturbed the scores without moving the top-2 and left the whole `indexer` bucket
    BIT-IDENTICAL -- argmax-invariance, the same shape as this tree's recorded "ids moved by
    exactly 0.000e0" finding. Those tensors are also the boundary S3's indexer kernel is scored
    at, so capturing them is what the fixture was for.
    """

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = _rope(q, current_cos, current_sin, 2)

        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

        visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for b in range(batch_size):
            for qi in range(seq_length):
                local = torch.nonzero(visible[b, 0, qi], as_tuple=False).flatten()
                ncb = local.shape[-1] // self.compress_ratio
                if ncb > 0:
                    flat_visible = local[: ncb * self.compress_ratio]
                    if variant == "blocks_strided":
                        block = flat_visible.view(self.compress_ratio, ncb).t().contiguous()
                    else:
                        block = flat_visible.view(ncb, self.compress_ratio)
                    groups = raw_keys[b].index_select(0, block.flatten())
                    groups = groups.view(*block.shape, self.index_head_dim)
                    starts = block[:, -1] if variant == "rope_at_block_end" else block[:, 0]
                    cos = full_cos[b].index_select(0, starts)
                    sin = full_sin[b].index_select(0, starts)
                    if variant == "rope_before_pool":
                        flat = _rope(
                            groups.reshape(ncb * self.compress_ratio, 1, self.index_head_dim),
                            full_cos[b].index_select(0, block.flatten()),
                            full_sin[b].index_select(0, block.flatten()),
                            1,
                        )
                        pooled = flat.view(ncb, self.compress_ratio, self.index_head_dim)
                        pooled = pooled.float().mean(dim=1).to(raw_keys.dtype)
                        keys = self.k_layernorm(pooled)
                    else:
                        pooled = groups.float().mean(dim=1).to(raw_keys.dtype)
                        pooled = self.k_layernorm(pooled)
                        keys = _rope(pooled.unsqueeze(1), cos, sin, 1).squeeze(1)
                    scores = torch.matmul(q[b, qi].float(), keys.float().transpose(-1, -2))
                    scores = scores.transpose(-1, -2)
                    if variant == "relu_after_sum":
                        scored = torch.relu(scores.sum(dim=-1))
                    else:
                        scored = torch.relu(scores).sum(dim=-1)
                    scores = scored / math.sqrt(self.index_head_dim)
                    picked = scores.topk(min(self.block_topk, ncb), dim=0).indices
                    selected = block.index_select(0, picked).flatten()
                    _record_scores(tap, self, qi, seq_length, keys, starts, scores)
                else:
                    selected = torch.tensor([], device=hidden_states.device)
                tail = local[ncb * self.compress_ratio :]
                selected = torch.cat([selected, tail]).to(torch.int32)
                selected_token_indices[b, qi, : selected.numel()] = selected

        kv_length = attention_mask.shape[-1]
        mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1),
            device=attention_mask.device,
            dtype=torch.bool,
        )
        scatter = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
        mask = mask.scatter(-1, scatter, True)[..., :kv_length].unsqueeze(1)
        if attention_mask.is_floating_point():
            mask = torch.where(mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return mask

    return forward


def _record_scores(tap, indexer, query_idx, seq_length, keys, starts, scores):
    """The three tensors the reference computes and throws away, for the LAST query only.

    Last query rather than all of them because the tensor COUNT must not be a function of the
    window: in `decode` there is exactly one query, and in `prefill` the last one is the only
    position whose visible set reaches the selective regime at this fixture's budget.

    The SELECTED BLOCK IDS are deliberately not among them: their length is
    `min(block_topk, num_complete_blocks)`, so `indexer_budget_one_block_short` changed the SHAPE
    (2 to 1, measured) and the comparator refused the pair. The selection is fully determined by
    the scores and `block_topk`, and the module's own output mask carries it -- which is where
    that row reddens.
    """
    if query_idx != seq_length - 1 or indexer.layer_idx not in tap.layers or not tap.ctx["capturing"]:
        return
    tag = f"model.layers.{indexer.layer_idx}.self_attn.indexer"
    tap.cap.add(f"{tag}.pooled_keys", keys)
    tap.cap.add(f"{tag}.block_starts", starts)
    tap.cap.add(f"{tag}.scores", scores)


def _rope(q, cos, sin, unsqueeze_dim):
    """The reference's `apply_rotary_pos_emb`, resolved LATE.

    Late on purpose: `rope_interleaved_pairs` and `rope_on_tail_dims` substitute that function,
    and the transcription above must see the substitution the way the reference's own body
    would. Reaching into the module rather than importing at module scope is what makes the two
    defect families compose.
    """
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mdl

    return mdl.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=unsqueeze_dim)


def wrap_indexer(mdl_mod, tap, variant=None):
    """Count every indexer call, and install a variant or the dense bypass when asked.

    The counter is always on: `assert_captured` uses it to refuse a run in which the indexer did
    not reach all twelve `qwen_sparse_attention` layers, which is what a `layer_types` read at
    face value would produce -- 12 layers of dense global attention and no indexer at all
    (trap T19). At the real budget that defect is bit-identical below 2051 cached tokens and
    therefore invisible; at this fixture's budget of 8 the boundary is 11, so a 16-token window
    sees it.
    """
    cls = mdl_mod.Qwen4ExpTextQSAIndexer
    orig = cls.forward
    body = indexer_variant_forward(variant, tap)

    def counted(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        tap.ctx["indexer_calls"] += 1
        mask = body(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        if not tap.ctx.get("qsa_dense"):
            return mask
        # `qsa_layer_is_dense`: the selection computed and then DISCARDED, which is what a port
        # that read `layer_types` at face value produces -- dense global attention on the 12
        # aliased layers. An all-permitting mask is the defect's own arithmetic, not a
        # transcription of the reference's.
        #
        # **The body still runs, and that is the point.** Returning early skipped
        # `index_qk_proj`, `q_layernorm` and `k_layernorm` on all three captured QSA layers, so
        # NINE hooks stayed silent and the golden's tensor SET changed -- `--compare` would have
        # aborted on a set mismatch instead of scoring the row. Computing and discarding keeps
        # the set fixed, keeps those three tensors bit-identical (they are upstream of the
        # discard), and moves exactly the mask this row is about.
        shape = (*attention_mask.shape[:1], 1, *attention_mask.shape[2:])
        if attention_mask.is_floating_point():
            return attention_mask.new_zeros(shape)
        return torch.ones(shape, dtype=torch.bool, device=attention_mask.device)

    cls.forward = counted
    return orig
