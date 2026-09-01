"""The defect matrix: one perturbation of the REFERENCE per row, and what each one prices.

A golden with no defect runs is not evidence, it is a file. Every row here is a wrong-but-
plausible reading taken from `docs/reference/qwen-architecture.md`'s traps table, and the table's
class-to-row map is what this file answers: **nine named operator classes, two or more rows
each.** `qwen_anchor_compare.EXPECT_FIRST_TOUCH` carries each row's declared first-touched bucket, so
the matrix is gated in BOTH directions -- the row reddens where it should and holds where it
should -- rather than only proving that something, somewhere, moved.

**Three rules every row here follows.**

1. *The tensor SET must not depend on the defect.* K3 measured the cost of breaking this: while
   routed experts were captured individually, four of five defects reported `inf` for most
   layers because a moved routing fires a different set of expert modules, and `--compare`
   scores "absent on one side" rather than a number. So perturbations replace a `forward`, edit
   a weight in place, or set a flag -- never introduce a submodule.
2. *A row that restates reference arithmetic says so.* Most do not: they set an attribute, flip
   a kwarg, permute weight rows, or substitute one free function whose uniqueness was checked
   against the pinned blob. The exceptions are named in place, and each is either the DEFECT's
   own arithmetic (which the reference does not contain) or gated by an identity check.
3. *A row whose fixture makes it vacuous is not a row.* Three are called out below --
   `router_no_renorm` needs a FLAT router draw, `gdn_conv_tap_reversed` needs a
   non-palindromic kernel, `rope_on_tail_dims` needs `head_dim > rotary_dim` -- and each
   condition is asserted or measured rather than hoped for.
"""

from qwen_anchor_capture import torch
from qwen_anchor_taps import (
    rebuild_ngram_tables,
    rope_interleaved_rotate_half,
)
from qwen_anchor_taps import rope_on_tail_dims as _rope_tail_apply


def _gdn_blocks(model):
    return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextGatedDeltaNet"]


def _indexers(model):
    return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextQSAIndexer"]


def _gated_norms(model):
    return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextRMSNormGated"]


def _residuals(model):
    return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextGatedResidual"]


def _moe_blocks(model):
    return [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextSparseMoeBlock"]


# ---------------------------------------------------------------------------------------------
# Class 1 -- GDN decay form. `qwen-architecture.md` T20 and T17.
#
# All three are set as flags and applied by `qwen_anchor_taps.decay_variant`, which states each
# wrong form as a RATIO against the reference's own `g`. The formula is never restated.


def gdn_decay_sigmoid_gate(model, mdl_mod, ctx):
    """`softplus` -> `sigmoid`: fla's own safe-gate form, and K3's KDA carries it in this tree."""
    ctx["gdn_decay_form"] = "sigmoid"


def gdn_decay_clamped(model, mdl_mod, ctx):
    """A KDA-style lower bound on the decay, in a model whose config has no such field."""
    ctx["gdn_decay_form"] = "clamped"


def gdn_dt_bias_outside_softplus(model, mdl_mod, ctx):
    """`dt_bias` as an additive term on the decay instead of an input to the softplus."""
    ctx["gdn_decay_form"] = "bias_outside"


# ---------------------------------------------------------------------------------------------
# Class 2 -- conv tap order. T2 and T21.


def _conv_weights(model):
    """Every depthwise conv the model has: 36 GDN kernels (kernel 4, dense taps) and the ONE
    PLE kernel (kernel 4, **dilation 3**, so its taps are `{t-9, t-6, t-3, t}`)."""
    return [m.weight for m in model.modules() if isinstance(m, torch.nn.Conv1d)]


def gdn_conv_tap_reversed(model, mdl_mod, ctx):
    """The kernel flipped along its tap axis: `w[0]` multiplying the current token.

    `F.conv1d` is cross-correlation -- no kernel flip -- so with three zeros of left padding
    `out[t] = sum_j w[j] x[t-3+j]` and `w[3]` is the CURRENT token (MOD:245-253). The decode
    path agrees: `cat([state(4), x_t])` with `padding=0`, taps `{state[1..3], x_t}`.

    **Vacuous on a palindromic kernel**, which is why the weight draw is uniform and asserted
    non-palindromic by the driver's degeneracy check rather than assumed.
    """
    with torch.no_grad():
        for w in _conv_weights(model):
            w.copy_(w.flip(-1))


def gdn_conv_tap_rotated(model, mdl_mod, ctx):
    """The subtler case: the kernel rotated by ONE tap, which survives a symmetry check.

    A reversal is caught by any test that notices the weight multiset order; a cyclic rotation
    leaves the multiset intact and changes every output, so a fixture checking only sums or
    norms passes it. Recorded in T21 as the row a converter is most likely to produce.

    > **T21 CORRECTED here, 2026-08-31.** The traps table says a rotation moves the PLE conv's
    > taps to `{t-6, t-3, t, t+3}` -- "a FUTURE tap, so this plant must also be caught by a
    > causality assert". Read against MOD:1160-1165 that is wrong: `_short_conv` pads
    > `state_len` zeros on the LEFT and slices to `state_len + seq_len`, so `F.conv1d` reads
    > exactly the same input positions whatever the weights hold; rotating `w` re-PAIRS weights
    > with positions and cannot reach `t+3`. There is no causality violation to assert, and the
    > row's power is the re-pairing alone. The future-tap reading describes a port that rotates
    > its tap INDICES, which is a different defect and is not plantable in the reference.
    """
    with torch.no_grad():
        for w in _conv_weights(model):
            w.copy_(torch.roll(w, 1, dims=-1))


# ---------------------------------------------------------------------------------------------
# Class 3 -- output-gate activation. T5 and T22.


def gdn_output_gate_silu(model, mdl_mod, ctx):
    """SiLU, which is fla/GDN's own default and `RMSNormGated`'s `activation` default."""
    for m in _gated_norms(model):
        m.activation = "silu"


def gdn_output_gate_tanh(model, mdl_mod, ctx):
    """`tanh`, the OTHER bounded gate -- TR p.4 calls it "the bounded sigmoid gate"."""
    for m in _gated_norms(model):
        m.activation = "tanh"


def gdn_output_gate_on_normed(model, mdl_mod, ctx):
    """The activation wrapping the normed stream instead of `z`.

    `RMSNormGated.forward` takes both `hidden_states` and `gate`, so which argument the
    activation wraps is one identifier apart (MOD:199). Planted by swapping the two arguments,
    which is shape-valid because both are `[N, head_v_dim]` -- and that is exactly what makes
    the confusion survive every shape check.
    """
    cls = mdl_mod.Qwen4ExpTextRMSNormGated
    orig = cls.forward
    cls.forward = lambda self, hidden_states, gate=None: orig(self, gate, hidden_states)


def gdn_norm_unit_offset(model, mdl_mod, ctx):
    """The GDN output norm read as zero-centred `(1 + w)` like every other norm in the model.

    TR p.4 says "the same formulation is consistently applied to all other RMSNorm layers" and
    Fig. 2 labels it "Zero-Centered RMSNorm". **Code wins, and it was MEASURED**:
    `layers.0.linear_attn.norm.weight` has min +0.875 over its 128 entries -- ones-centred,
    matching `RMSNormGated`'s bare `w * x_hat` (MOD:198), while `hc_norm` has min -5.9375. The
    256 bytes that settle it are vendored as `qwen-reference/linear_attn_norm_l0.bin`.

    Applied as an exact ratio `(1 + w) / w` on the module's own output, so the norm is not
    restated. Safe because the driver draws these weights near 1 (the reference's `torch.ones`
    init), which is the measured scale.
    """
    cls = mdl_mod.Qwen4ExpTextRMSNormGated
    orig = cls.forward

    def shim(self, hidden_states, gate=None):
        out = orig(self, hidden_states, gate)
        w = self.weight.to(out.dtype)
        return out * ((1.0 + w) / w)

    cls.forward = shim


# ---------------------------------------------------------------------------------------------
# Class 9 -- eps homes. T23, restated against what v5.16.1 actually contains.
#
# > **T23 CORRECTED here, 2026-08-31.** The traps table lists `eps_from_the_wrong_config_field`
# > and says `GatedDeltaNet.__init__` "passes `layer_norm_epsilon` and not `rms_norm_eps`". At
# > this revision `self.layer_norm_epsilon = config.rms_norm_eps` (MOD:417) -- it is the same
# > value under a local name, and `Qwen4ExpTextConfig` has no `layer_norm_epsilon` field at all.
# > So that row has no referent and is replaced by `l2norm_eps_dropped`: `l2norm` carries its
# > own hard-coded 1e-6 (MOD:261), which is a genuinely different eps HOME even though this
# > checkpoint's `rms_norm_eps` happens to equal it.


def eps_outside_sqrt(model, mdl_mod, ctx):
    """`x / (sqrt(mean(x^2)) + eps)` for `x * rsqrt(mean(x^2) + eps)`.

    One paren apart, and they agree to about `eps / sqrt(mean)` on well-scaled inputs -- which
    is why this row's margin is MEASURED rather than assumed, and why the norm operator's
    tolerance may have to be `ExactOnly` (the shape K3's MLA eps has, at 1.3x its own floor).
    A transcription of `_norm`'s three lines with the one substitution.
    """
    cls = mdl_mod.Qwen4ExpTextRMSNorm

    def _norm(self, x):
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x / (x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    cls._norm = _norm


def l2norm_eps_dropped(model, mdl_mod, ctx):
    """The qk L2 norm's own 1e-6 dropped -- the eps that lives in a literal, not in the config.

    `use_qk_l2norm_in_kernel=True` at both delta-rule call sites (MOD:534, MOD:547) and `l2norm`
    hard-codes `eps=1e-6` (MOD:261). A port that routed `rms_norm_eps` here, or dropped the term
    because the two happen to be equal in this checkpoint, is not distinguishable from a correct
    one by any other row. Wrapped rather than transcribed.
    """
    orig = mdl_mod.l2norm
    mdl_mod.l2norm = lambda x, dim=-1, eps=1e-6: orig(x, dim=dim, eps=0.0)


def hc_norm_no_offset(model, mdl_mod, ctx):
    """The trunk RMSNorm read as ones-centred `w` instead of zero-centred `(1 + w)`.

    The mirror of `gdn_norm_unit_offset`, and the pair is the point: the model carries BOTH
    formulas, and each row must redden only its own norm. `Qwen4ExpTextRMSNorm.weight` is
    `torch.zeros`-initialised (MOD:161) and the measured checkpoint mean is -0.06355, so the
    ratio `w / (1 + w)` is well conditioned at the driver's own draw of (-0.2, 0.2).
    """
    cls = mdl_mod.Qwen4ExpTextRMSNorm
    orig = cls.forward

    def shim(self, x):
        out = orig(self, x)
        w = self.weight.to(out.dtype)
        return out * (w / (1.0 + w))

    cls.forward = shim


# ---------------------------------------------------------------------------------------------
# Class 5 -- hyper-connections. T9, T10, T11.


def hc_sum_not_mean(model, mdl_mod, ctx):
    """The branch collapse by SUM instead of MEAN -- exactly 4x, on every sublayer of 48 layers.

    MOD:966 is `.mean(dim=-2)`. `mean * hc_count` IS the sum, so scaling the module's own first
    return value is exact and restates nothing.
    """
    ctx["hc_scale_mixed"] = float(model.config.hc_count)
    _wrap_residual(model, mdl_mod, ctx)


def hc_residual_uses_normed(model, mdl_mod, ctx):
    """The residual base taken from the NORMED stream. MOD:969 returns the PRE-norm
    `hyper_input`, and `hyper_input` vs `hyper_input_normed` is one identifier apart."""
    ctx["hc_residual_normed"] = True
    _wrap_residual(model, mdl_mod, ctx)


def _wrap_residual(model, mdl_mod, ctx):
    """One wrapper serving both hyper-connection value rows, installed once.

    Wrapping the class rather than patching its body keeps the tensor set fixed and keeps the
    reference's own `hc_norm` in the path for the normed-residual row -- the defect calls the
    module, it does not re-implement it.
    """
    cls = mdl_mod.Qwen4ExpTextGatedResidual
    if getattr(cls.forward, "_qwen_anchor_wrapped", False):
        return
    orig = cls.forward

    def shim(self, hyper_input):
        out = orig(self, hyper_input)
        scale = ctx.get("hc_scale_mixed")
        if isinstance(out, tuple):
            mixed, residual, inject = out
            if scale is not None:
                mixed = mixed * scale
            if ctx.get("hc_residual_normed"):
                # `.forward` and not `__call__`: `hc_norm` is a HOOKED module, so calling it
                # normally fires its hook a second time and the golden gains a duplicate
                # `attn_hyper_connection.hc_norm` entry -- which the container's own reader
                # refuses, and which would otherwise have collapsed last-wins and shrunk the
                # compared set. Found by that assertion on this row's first run.
                residual = self.hc_norm.forward(hyper_input)
            return mixed, residual, inject
        return out * scale if scale is not None else out

    shim._qwen_anchor_wrapped = True
    cls.forward = shim


def hc_lowrank_matrices_swapped(model, mdl_mod, ctx):
    """`W_d` and `W_u` swapped by the converter, each transposed to fit.

    `hc_lowrank: 320` is the bottleneck rank `d/8`, with `input_mix_weight_down` **[320, 10240]**
    and `input_mix_weight_up` **[10240, 320]**. A converter that put one where the other belongs
    still type-checks once transposed, which is the whole of T11: a shape-agnostic port cannot
    tell. Non-crashing, and it changes every layer.
    """
    with torch.no_grad():
        for m in _residuals(model):
            down, up = m.input_mix_weight_down.weight, m.input_mix_weight_up.weight
            swapped = up.t().contiguous()
            up.copy_(down.t().contiguous())
            down.copy_(swapped)


def hc_mix_gate_static(model, mdl_mod, ctx):
    """The dynamic mix gate made STATIC -- T9's "static mHC" reading, with no `H_res` invented.

    TR section 2.2 drops the branch-mixing operator outright and finds the static terms bring
    "no improvement for GR", so the dynamic low-rank gate is the whole mechanism. Zeroing
    `input_mix_weight_down` makes `silu(0) = 0`, `up(0) = 0` and `sigmoid(0) = 0.5`: a constant
    half on every branch, which is precisely a static hyper-connection. One weight, zeroed --
    no tensor invented, which is what T9 says makes the Sinkhorn reading unplantable.
    """
    with torch.no_grad():
        for m in _residuals(model):
            m.input_mix_weight_down.weight.zero_()


# ---------------------------------------------------------------------------------------------
# Class 6 -- router and experts. T15, T24, T25.


def _wrap_router(model, mdl_mod, ctx):
    """One wrapper for the three router rows.

    **This is the file's second restatement of reference arithmetic, and the cost is real.** The
    router returns only its RENORMALISED weights (MOD:912-913), so a sigmoid score, a biased
    score and an un-renormalised weight are none of them recoverable from its output -- there is
    nothing to reuse, and the gather-and-renormalise has to be written here. K3's
    `RouterBiasInWeight` records paying the same price for the same reason. The consequence to
    state plainly: if the reference's weight path changes, these three rows redden the goldens
    as though the traps had fired. The `logits` the wrapper starts from are the reference's own.
    """
    cls = mdl_mod.Qwen4ExpTextTopKRouter
    if getattr(cls.forward, "_qwen_anchor_wrapped", False):
        return
    orig = cls.forward

    def shim(self, hidden_states):
        logits, scores, indices = orig(self, hidden_states)
        bias = ctx.get("router_bias")
        if bias is not None:
            probs = torch.nn.functional.softmax(logits.float() + bias.to(logits.device), dim=-1)
            indices = torch.topk(probs, self.top_k, dim=-1).indices
            probs = torch.nn.functional.softmax(logits.float(), dim=-1)
        elif ctx.get("router_sigmoid"):
            probs = torch.sigmoid(logits.float())
        else:
            probs = torch.nn.functional.softmax(logits.float(), dim=-1)
        kept = probs.gather(-1, indices)
        if self.norm_topk_prob and not ctx.get("router_no_renorm"):
            kept = kept / kept.sum(dim=-1, keepdim=True)
        return logits, kept.to(logits.dtype), indices

    shim._qwen_anchor_wrapped = True
    cls.forward = shim


def router_sigmoid(model, mdl_mod, ctx):
    """Sigmoid scoring, which every other MoE in this tree uses. MOD:910 is a softmax over all
    512 experts, and TR says nothing about router scoring -- so the code is the only source."""
    ctx["router_sigmoid"] = True
    _wrap_router(model, mdl_mod, ctx)


def router_bias_added(model, mdl_mod, ctx):
    """A top-k bias correction, selected on the biased score and weighted from the unbiased one.

    **There is no bias tensor to zero, so the plant has to ADD one** (T15) -- the absence is
    structural here, unlike GLM/DeepSeek where the bias is an `nn.Buffer` invisible to
    `named_parameters`. Drawn from the run's salt so the two weight draws price it independently.
    """
    ctx["router_bias"] = ctx["draw"]("router_bias_correction", model.config.num_experts, -0.5, 0.5)
    _wrap_router(model, mdl_mod, ctx)


def router_no_renorm(model, mdl_mod, ctx):
    """The kept probabilities used as softmax gave them. `norm_topk_prob` is ABSENT from the
    checkpoint's config and DEFAULTS to True (CFG:163), so a port reading only the file finds
    nothing to honour.

    **Vacuous on a peaked router**, which is the whole trap: the ten kept probabilities sum to
    about 1 on a trained router and the error is a small per-token scale. This fixture's router
    weight is drawn uniform on a 96-wide hidden state, so the softmax is nearly flat and the two
    kept probabilities sum to about `top_k / num_experts` = 0.25 -- the un-renormalised output
    is roughly 4x too small. The measured kept-sum is in `anchor.md` and is asserted by
    `qwen_anchor.rs` on the `None` golden, so the row cannot go quietly vacuous.
    """
    ctx["router_no_renorm"] = True
    _wrap_router(model, mdl_mod, ctx)


def expert_gate_up_swapped(model, mdl_mod, ctx):
    """The fused `gate_up_proj` halves swapped, so SiLU lands on `up`.

    The checkpoint ships `gate_proj` and `up_proj` as SEPARATE per-expert tensors while the
    reference wants one fused 3-D parameter, so the CONVERTER chooses the order -- and "w1/w3"
    naming disagrees across families about which is which. MOD:889 is
    `gate, up = linear(...).chunk(2, -1)` then `act_fn(gate) * up`, so `gate` is the FIRST half.
    Needs distinct gate and up draws, which a per-parameter-name draw guarantees.
    """
    with torch.no_grad():
        for m in _moe_blocks(model):
            w = m.experts.gate_up_proj
            inter = w.shape[1] // 2
            w.copy_(torch.cat([w[:, inter:], w[:, :inter]], dim=1))


# ---------------------------------------------------------------------------------------------
# Class 7 -- the n-gram hash. T12, T13, T14.


def ngram_seed_wrong(model, mdl_mod, ctx):
    """The multipliers reseeded. `base_seed = seed + 10007 * ple_layer_index` with `seed`
    defaulting to 1234 (CFG:156, absent from the checkpoint), so the seed is a default nobody
    reading the file would find."""
    rebuild_ngram_tables(mdl_mod, model, seed=model.config.seed + 1)


def ple_layer_index_off_by_one(model, mdl_mod, ctx):
    """`ple_layer_index` 1 instead of 0 -- and this row is where the prime rule earns its keep.

    The layer index is in BOTH halves of the per-PLE-layer indexing: `base_seed` carries it for
    the multipliers, and `global_head_idx = ple_layer_index * ngram_heads + head_idx` carries it
    for the 16 vocabularies (MOD:1038-1039). Stating only one of them is the trap, and it is
    the specialisation that made `ple_layer_index == 0` MEASURABLE: index 0 reproduces the
    checkpoint's 20000003..20000171 and its three multipliers byte-exactly while index 1 gives
    20000213..20000429. Rebuilt through `_build_layer_multipliers` and `_find_nth_prime_after`
    themselves, so what is scored is the reference's derivation at the wrong argument.
    """
    rebuild_ngram_tables(mdl_mod, model, ple_layer_index=1)


def ngram_order_swapped(model, mdl_mod, ctx):
    """Bigram heads and trigram heads swapped. Heads 0..7 are bigram and 8..15 trigram
    (MOD:1098-1100, `start_idx = (ngram - 2) * heads_per_ngram`), and because each head owns its
    own prime a swap changes every id."""
    rebuild_ngram_tables(mdl_mod, model, mode="order_swapped")


def ngram_mod_round(model, mdl_mod, ctx):
    """Every head reduced modulo `ngram_vocab_size_base` -- the round 20,000,000 the config field
    is named after -- instead of modulo its own prime. The offsets stay, so the heads collide
    into overlapping row ranges: plausible-looking output, not a crash."""
    rebuild_ngram_tables(mdl_mod, model, mode="round")


def ngram_mix_add(model, mdl_mod, ctx):
    """The hash mixed with `+` instead of XOR.

    `torch.bitwise_xor` appears EXACTLY ONCE in the model (MOD:1103), checked by grep against
    the pinned blob, so substituting it for the life of one run touches nothing else. Addition
    is the natural way to combine multiplied shifts and is what a port that read "mix" rather
    than the code would write.
    """
    ctx["patch_torch"] = ("bitwise_xor", torch.add)


# ---------------------------------------------------------------------------------------------
# Class 4 -- the QSA indexer. T6, T7, T8, T19.


def indexer_relu_off(model, mdl_mod, ctx):
    """The ReLU dropped from the block scores.

    TR Eq. 15 is `sum_h ReLU(<q_h, kbar_b>)` and MOD:693 is `relu(scores).sum(dim=-1)`. Without
    it, negative per-head agreements cancel positive ones and the selection changes.
    `torch.relu` appears EXACTLY ONCE in the model -- inside this indexer -- so the substitution
    needs no transcription. T6's own `indexer_head_weights` reading cannot be planted at all:
    `index_qk_proj` is the only indexer projection the checkpoint ships, so there is no `w_h` to
    move, and the absence is structural.
    """
    ctx["patch_torch"] = ("relu", lambda x: x)


def indexer_budget_one_block_short(model, mdl_mod, ctx):
    """`block_topk - 1`: one block too few kept.

    T8's confusion is "2048 blocks or 512 tokens", and the plantable off-by-one is the SHORT
    direction. Two other readings were tried and are not rows:

      * `block_topk = token_budget` selects every block at both real and tiny widths, so it is
        bit-identical to `qsa_layer_is_dense` and counting both would count one row twice;
      * **`block_topk + 1` CRASHES the reference**, measured: `selected_token_indices` is exactly
        `indexer_budget + indexer_compress_ratio - 1` wide (MOD:661-666), so keeping one extra
        block writes `(block_topk + 1) * ratio` tokens into a buffer sized for
        `block_topk * ratio + (ratio - 1)` and raises "expanded size of the tensor (11) must
        match the existing size (12)". True at the real widths too -- 2052 selected tokens into a
        2051 buffer. So that direction of the off-by-one is LOUD in the reference and needs no
        oracle; the short direction is the silent one, and it is the row.

    Non-vacuous wherever `num_complete_blocks >= 2`, i.e. from 8 visible tokens on.
    """
    for m in _indexers(model):
        m.block_topk -= 1


def indexer_blocks_strided(model, mdl_mod, ctx):
    """Blocks formed by STRIDE over the visible list instead of contiguously.

    `block_token_indices = local_visible_indices[: ncb * r].view(ncb, r)` (MOD:673-675) groups
    CONSECUTIVE visible tokens; `.view(r, ncb).t()` groups every r-th one. Same block count, same
    shapes, same token multiset -- and a completely different pooling, which is what makes it the
    reshape-order trap rather than a size error.

    > **T8 CORRECTED here, 2026-08-31.** The plan asks for a "pool ratio 2" row and it is not
    > plantable: `compress_ratio` decides `num_complete_blocks`, so halving it changes the SHAPE
    > of the pooled-key and score tensors ([4, 24] became [8, 24], measured), and the comparator
    > refuses a shape mismatch because a defect that moves the tensor set scores "absent on one
    > side" rather than a number. `indexer_compress_ratio` is instead pinned STRUCTURALLY -- it is
    > in `qwen_anchor_lib.STRUCTURAL`, inherited unchanged from the checkpoint and re-asserted
    > from the golden's own bytes -- and the reshape ORDER, which no config field pins, is what
    > this row prices.
    """
    ctx["indexer_variant"] = "blocks_strided"


def indexer_rope_at_block_end(model, mdl_mod, ctx):
    """The pooled keys roped at the block's LAST token position instead of its first."""
    ctx["indexer_variant"] = "rope_at_block_end"


def indexer_rope_before_pool(model, mdl_mod, ctx):
    """RoPE before pooling -- the order TR Eq. 13-14 exists to forbid, because pooling first
    "avoids averaging token representations with different rotary phases"."""
    ctx["indexer_variant"] = "rope_before_pool"


def indexer_relu_after_sum(model, mdl_mod, ctx):
    """The ReLU applied AFTER the aggregation over heads instead of before it.

    TR Eq. 15 is `I_ib = sum_h ReLU(<q_i^h, kbar_b>)` and MOD:693 is `relu(scores).sum(dim=-1)`:
    each head's agreement is rectified first, so a negative head cannot cancel a positive one.
    Rectifying after lets it, which changes every block score with a negative head in it --
    distinct from `indexer_relu_off`, which drops the rectification entirely.

    > **T6/T7 CORRECTED here, 2026-08-31, and TWO readings were rejected on MEASUREMENT.** The
    > traps table names `indexer_sum_over_blocks`, summing the wrong axis. That is not plantable:
    > summing `dim=-2` yields `indexer_n_heads` scores and MOD:695 then indexes
    > `block_token_indices` with a head id, raising whenever
    > `num_complete_blocks < indexer_n_heads`. The wrong axis is a SHAPE error, not a silent
    > defect. A max-over-heads reading was tried next and REJECTED as draw-dependent: it left the
    > FIRST QSA layer's whole indexer bucket bit-identical at draw 1 (8 tensors, 0 differing),
    > because at most one head scored positive per block there, so max and sum agreed exactly --
    > while layers 27 and 47 reddened. A row that is vacuous at its own declared first-touched
    > bucket on one draw is not a row, which is the rule this file's header states.
    """
    ctx["indexer_variant"] = "relu_after_sum"


def qsa_layer_is_dense(model, mdl_mod, ctx):
    """**T19: the 12 `full_attention` entries read as dense global attention.**

    It is the checkpoint's own spelling, it is what every other HF config means by it, and MOD
    uses the identical string as a causal-mask dictionary key (MOD:1404, 1423). The reference
    ALIASES it -- CFG:180-184 rewrites `full_attention` to `qwen_sparse_attention` before
    `validate_architecture` will accept the config, with the comment "layers that are actually
    using an indexer".

    `qwen-architecture.md` records this as the one trap invisible to every gate the port plans,
    because MOD:695's `topk(min(block_topk, num_complete_blocks))` makes QSA DENSE BY
    CONSTRUCTION at or below `budget + ratio - 1` = 2051 cached tokens -- the regime of the free
    oracle, the 64-token parity window and the smoke decode cell alike. **This fixture closes
    that hole by scaling the boundary instead of the context**: at `indexer_budget: 8` it sits at
    11, and the 16-token window holds 5 queries above it (8 of 16 tokens kept at the last
    position, measured). So the row reddens here, deviceless, at a width the port can afford --
    and the arch doc's "needs > 2051 cached tokens" is true of the REAL budget only.
    """
    ctx["qsa_dense"] = True


# ---------------------------------------------------------------------------------------------
# The remaining classes: fused-QKV segmentation (T3), the attention gate split (T4), and
# partial-rope width (T16). Each is a defect this tree has been bitten by on another port.


def _swap_rows(weight, a, b, n):
    with torch.no_grad():
        rows = weight[a:a + n].clone()
        weight[a:a + n] = weight[b:b + n]
        weight[b:b + n] = rows


def gdn_qkv_qk_swapped(model, mdl_mod, ctx):
    """`in_proj_qkv` split as `[K|Q|V]`.

    `torch.split(..., [key_dim, key_dim, value_dim])` puts Q at rows 0..key_dim and K next
    (MOD:502-515). 10240 = 2*2048 + 6144 admits several segmentations, and LC#27742 reports its
    own architecture test could not tell three of them apart. A `[V|K|Q]` order is not
    shape-valid here (60/60/180), so the plantable confusion is the Q/K swap -- and it needs
    DISTINCT per-head draws, which a name-keyed draw gives.
    """
    m = _gdn_blocks(model)[0]
    key_dim = m.key_dim
    for block in _gdn_blocks(model):
        _swap_rows(block.in_proj_qkv.weight, 0, key_dim, key_dim)


def _head_minor_permutation(heads, dim):
    """The index map from head-major `[h][d]` to channel-major `[d][h]`, as a row permutation."""
    return [d * heads + h for h in range(heads) for d in range(dim)]


def gdn_qkv_head_interleaved(model, mdl_mod, ctx):
    """The Q/K/V blocks read channel-major instead of head-major.

    Each block is `reshape(batch, seq, -1, head_dim)` (MOD:509-513), i.e. head-major: head `h`
    owns rows `[dim*h, dim*h + dim)`. A port that interleaved the heads instead reads
    `[d*heads + h]`, which is shape-valid and silently wrong. Planted as the row permutation, so
    the reference's own reshape produces the wrong reading.
    """
    with torch.no_grad():
        for block in _gdn_blocks(model):
            w = block.in_proj_qkv.weight
            perm = []
            offset = 0
            for heads, dim in (
                (block.num_k_heads, block.head_k_dim),
                (block.num_k_heads, block.head_k_dim),
                (block.num_v_heads, block.head_v_dim),
            ):
                perm += [offset + i for i in _head_minor_permutation(heads, dim)]
                offset += heads * dim
            w.copy_(w[perm])


def attn_gate_block_split(model, mdl_mod, ctx):
    """`q_proj` read as `[query | gate]` in two blocks instead of per head.

    It is per HEAD: `view(*input_shape, -1, head_dim * 2)` then `chunk(2, dim=-1)`
    (MOD:805-807), so head `h` owns rows `[2*head_dim*h, +head_dim)` for the query and the next
    `head_dim` for the gate. The block reading takes the first half of all 6144 rows as query.
    Planted as the row permutation that makes the reference's own per-head view select the block
    layout's contents.
    """
    with torch.no_grad():
        for m in model.modules():
            if type(m).__name__ != "Qwen4ExpTextAttention":
                continue
            heads, dim = m.config.num_attention_heads, m.head_dim
            perm = [
                part * heads * dim + h * dim + d
                for h in range(heads)
                for part in range(2)
                for d in range(dim)
            ]
            m.q_proj.weight.copy_(m.q_proj.weight[perm])


def rope_interleaved_pairs(model, mdl_mod, ctx):
    """The `(i, i+1)` pairing instead of `(i, i + rotary_dim/2)`."""
    mdl_mod.rotate_half = rope_interleaved_rotate_half


def rope_on_tail_dims(model, mdl_mod, ctx):
    """The LAST `rotary_dim` dims rotated instead of the first.

    Partial RoPE covers dims 0..63 of 256 as 32 `(i, i+32)` pairs and 64..255 pass through
    (MOD:591-600). Both this row and the interleaved one need `head_dim > rotary_dim`, which the
    width audit's `rotary_dim vs head_dim/2` pair guarantees (32 > 8 here, 256 > 64 real).
    """
    mdl_mod.apply_rotary_pos_emb = _rope_tail_apply(mdl_mod.rotate_half)


DEFECTS = {
    "None": lambda model, mdl_mod, ctx: None,
    "gdn_decay_sigmoid_gate": gdn_decay_sigmoid_gate,
    "gdn_decay_clamped": gdn_decay_clamped,
    "gdn_dt_bias_outside_softplus": gdn_dt_bias_outside_softplus,
    "gdn_state_axis_swapped": lambda model, mdl_mod, ctx: ctx.__setitem__("gdn_transpose_out_state", True),
    "gdn_conv_tap_reversed": gdn_conv_tap_reversed,
    "gdn_conv_tap_rotated": gdn_conv_tap_rotated,
    "gdn_qkv_qk_swapped": gdn_qkv_qk_swapped,
    "gdn_qkv_head_interleaved": gdn_qkv_head_interleaved,
    "gdn_output_gate_silu": gdn_output_gate_silu,
    "gdn_output_gate_tanh": gdn_output_gate_tanh,
    "gdn_output_gate_on_normed": gdn_output_gate_on_normed,
    "gdn_norm_unit_offset": gdn_norm_unit_offset,
    "hc_norm_no_offset": hc_norm_no_offset,
    "eps_outside_sqrt": eps_outside_sqrt,
    "l2norm_eps_dropped": l2norm_eps_dropped,
    "hc_sum_not_mean": hc_sum_not_mean,
    "hc_residual_uses_normed": hc_residual_uses_normed,
    "hc_lowrank_matrices_swapped": hc_lowrank_matrices_swapped,
    "hc_mix_gate_static": hc_mix_gate_static,
    "router_sigmoid": router_sigmoid,
    "router_bias_added": router_bias_added,
    "router_no_renorm": router_no_renorm,
    "expert_gate_up_swapped": expert_gate_up_swapped,
    "ngram_seed_wrong": ngram_seed_wrong,
    "ngram_order_swapped": ngram_order_swapped,
    "ngram_mod_round": ngram_mod_round,
    "ngram_mix_add": ngram_mix_add,
    "ple_layer_index_off_by_one": ple_layer_index_off_by_one,
    "indexer_relu_off": indexer_relu_off,
    "indexer_budget_one_block_short": indexer_budget_one_block_short,
    "indexer_blocks_strided": indexer_blocks_strided,
    "indexer_rope_at_block_end": indexer_rope_at_block_end,
    "indexer_rope_before_pool": indexer_rope_before_pool,
    "indexer_relu_after_sum": indexer_relu_after_sum,
    "qsa_layer_is_dense": qsa_layer_is_dense,
    "attn_gate_block_split": attn_gate_block_split,
    "rope_interleaved_pairs": rope_interleaved_pairs,
    "rope_on_tail_dims": rope_on_tail_dims,
}

# **The nine named classes, and the rows that cover each.** Asserted by
# `assert_class_coverage`, not described: the plan's requirement is two or more rows per class,
# and a count of rows cannot tell whether the distribution met it. The traps table's own
# class-to-row map is what this answers.
CLASSES = {
    "gdn decay form": (
        "gdn_decay_sigmoid_gate",
        "gdn_decay_clamped",
        "gdn_dt_bias_outside_softplus",
    ),
    "conv tap order": ("gdn_conv_tap_reversed", "gdn_conv_tap_rotated"),
    "output-gate activation": (
        "gdn_output_gate_silu",
        "gdn_output_gate_tanh",
        "gdn_output_gate_on_normed",
    ),
    "indexer relu/pool/budget": (
        "indexer_relu_off",
        "indexer_budget_one_block_short",
        "indexer_blocks_strided",
        "indexer_rope_at_block_end",
        "indexer_rope_before_pool",
        "indexer_relu_after_sum",
        "qsa_layer_is_dense",
    ),
    "hc transpose + lowrank order": (
        "hc_sum_not_mean",
        "hc_residual_uses_normed",
        "hc_lowrank_matrices_swapped",
        "hc_mix_gate_static",
    ),
    "router bias/norm/w1w3": (
        "router_sigmoid",
        "router_bias_added",
        "router_no_renorm",
        "expert_gate_up_swapped",
    ),
    "n-gram seed/order/layer": (
        "ngram_seed_wrong",
        "ngram_order_swapped",
        "ngram_mod_round",
        "ngram_mix_add",
        "ple_layer_index_off_by_one",
    ),
    "partial-rope width": ("rope_interleaved_pairs", "rope_on_tail_dims"),
    "eps homes": ("eps_outside_sqrt", "l2norm_eps_dropped"),
    # The four classes carried without being on the plan's list, each a defect this tree has
    # already been bitten by on another port.
    "norm offset": ("gdn_norm_unit_offset", "hc_norm_no_offset"),
    "fused-QKV segmentation": ("gdn_qkv_qk_swapped", "gdn_qkv_head_interleaved"),
    "attention gate split": ("attn_gate_block_split",),
    "state axis order": ("gdn_state_axis_swapped",),
}
# The nine the plan counts; the four extras above are coverage, not a quota.
PLAN_CLASSES = tuple(list(CLASSES)[:9])


def assert_class_coverage():
    """Two or more rows for each of the nine named classes, and every row in a class.

    Both halves earn their place. A class with one row reads as covered in a count of 39; a row
    in no class is a perturbation nobody decided what it prices, and `EXPECT_FIRST_TOUCH` would
    not notice.
    """
    for name in PLAN_CLASSES:
        rows = CLASSES[name]
        assert len(rows) >= 2, f"class {name!r} has only {len(rows)} defect row(s), the rule is >= 2"
    named = {r for rows in CLASSES.values() for r in rows}
    unclassified = sorted(set(DEFECTS) - named - {"None"})
    assert not unclassified, f"defect rows in no class: {unclassified}"
    missing = sorted(named - set(DEFECTS))
    assert not missing, f"classes name rows that do not exist: {missing}"
