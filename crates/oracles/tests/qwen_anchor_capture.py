"""The captured tensors, and the forward hooks that a `register_forward_hook` can supply.

**This half is the mechanism that works by itself.** `hook_model` puts one hook on every
submodule of the captured layers, and `Capture` holds what they produce. The values no forward
hook can reach live in `qwen_anchor_taps.py`, and each one's docstring records why the hook is
blind to it -- the GDN short convolution is the sharp case: `Qwen4ExpTextGatedDeltaNet.forward`
never calls `self.conv1d`, it hands `self.conv1d.weight.squeeze(1)` to the module-level
`causal_conv1d_fn`, so a hook on that `nn.Conv1d` fires ZERO times. That is exactly the shape of
the bug K3's anchor carried for a day with three comments claiming otherwise, and it is why
[`assert_captured`] refuses a run in which any registered hook stayed silent.
"""

import collections


# **`torch` resolved on first attribute access, never at import.** `qwen_anchor_driver` binds its
# own `torch` inside `main` for one stated reason -- `--compare` re-scores vendored bytes with no
# torch and no venv -- and it imports this module at the top to re-export the names below. A
# plain `import torch` here would take that property away. The proxy costs one attribute lookup
# on a path already calling into torch, and leaves every body saying `torch.x`.
class _LazyTorch:
    def __getattr__(self, name):
        import torch

        return getattr(torch, name)


torch = _LazyTorch()


# **The three things every tap and the capture pass need**, in one value: where tensors go, the
# run's mutable `ctx` (which the defects write into), and which layers are captured. Bundled
# because otherwise each of those functions takes five or six parameters whose real arity is two.
Tap = collections.namedtuple("Tap", "cap ctx layers")


class Capture:
    """Named tensors, split by element type the way `crates/oracles/src/golden.rs` splits them."""

    def __init__(self):
        self.floats = []
        self.ints = []
        self.seen = set()
        # Module names whose hook actually RAN, recorded by the hook itself rather than inferred
        # from tensor names. Inferring would be prefix-shadowed: `model.layers.47`'s own hook
        # could look like it fired because `model.layers.47.self_attn` did.
        self.fired = set()
        # How many times each BARE name has been offered. The ordinal suffix below is derived
        # from it, and `repeated()` reports the ones that recurred.
        self.calls = collections.Counter()

    def add(self, name, t):
        """Record one tensor, giving a REPEAT CALL its own name rather than a colliding one.

        **A module called more than once per forward used to write two entries under one name,
        and that was a live defect in the prefill golden.** `qwen_anchor_lib.read_golden` asserts
        against duplicates and says of the alternative "the Rust reader keeps both. Latent;
        loud" -- but nothing had ever read the prefill golden back, so the latent half was what
        shipped. Measured 2026-08-31 on the inherited `gold-prefill-qwen-anchor-1-None.bin`:
        THIRTEEN entries named `model.layers.3.self_attn.indexer.k_layernorm`, because the
        transcribed indexer body calls that norm once per query position with `ncb > 0`
        (`qwen_anchor_taps.py`, the `for qi in range(seq_length)` loop) and 13 of 16 positions
        qualify. `golden_read::float` finds the FIRST match, so twelve of the thirteen were
        unreachable by name and any census over distinct names was silently short.

        The `NEVER_FIRES` note below records the same class caught once before, on
        `mlp.experts.act_fn` -- and the fix there was exclusion, which is right when the call
        count is a function of the ROUTING (a moved routing then changes the tensor SET and
        `--compare` scores "absent on one side" instead of the arithmetic). It is the wrong fix
        here: this count is a function of the window and the compress ratio, both fixed by the
        config, and the LADDER of thirteen shapes -- (1,24) x4, (2,24) x4, (3,24) x4, (4,24) --
        is the per-query complete-block count, which is exactly the dense-versus-selective
        boundary the prefill golden exists to make visible (T19, half of T8). Losing it to keep
        one name unique would have thrown away the fixture's whole point.

        So the first call keeps the bare name -- which is why every DECODE golden's bytes are
        unchanged by this, verified by regeneration and `cmp` rather than argued -- and the k-th
        keeps `#k`. `operator_of` and `bucket_of` classify by prefix and substring, so a
        suffixed name still lands in its own bucket; `seen` keeps the BARE name so
        `assert_captured`'s fragment tests are untouched.
        """
        if not isinstance(t, torch.Tensor):
            return
        t = t.detach().to("cpu")
        shape = list(t.shape)
        self.seen.add(name)
        self.calls[name] += 1
        k = self.calls[name]
        stored = name if k == 1 else f"{name}#{k}"
        if t.dtype in (torch.int64, torch.int32, torch.bool):
            self.ints.append((stored, shape, t.reshape(-1).to(torch.int64).tolist()))
        else:
            self.floats.append((stored, shape, t.reshape(-1).to(torch.float32).tolist()))

    def repeated(self):
        """Every name offered more than once, and how often -- for the golden's own metadata.

        In the bytes rather than left implicit, because a reader counting distinct tensor names
        has to know that thirteen of them share a module. A DECODE golden's map is empty, which
        is itself the statement that its per-name lookups are unambiguous.
        """
        return {name: n for name, n in sorted(self.calls.items()) if n > 1}

    def add_any(self, name, out):
        """Flatten a module's output, whatever container it came in."""
        if isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                self.add_any(f"{name}.{i}", o)
        else:
            self.add(name, out)


def fire(cap, name, out):
    """Record that `name`'s hook ran, then capture its output."""
    cap.fired.add(name)
    cap.add_any(name, out)


# The model-level tail: the 4-to-1 collapse (`hyper_connection_mixer`, a fifth `GatedResidual`
# with `use_combine=False`) and the head. **There is no `model.norm`** -- that collapse IS the
# model's only final norm (MOD:1330/1430, and the checkpoint carries no such tensor), which is
# the one structural fact about the tail a port is most likely to add by habit.
TAIL = ("model.hyper_connection_mixer", "lm_head")

# Submodules that must NOT get a hook, as name SUFFIXES rather than a substring test -- so
# `ple.conv1d` cannot be swept up by a rule written for the GDN one.
#
#   * `linear_attn.conv1d` would be DEAD: its weight is handed to the module-level
#     `causal_conv1d_fn` / `causal_conv1d_update` (MOD:477-496) and the `nn.Conv1d` is never
#     called. `qwen_anchor_taps.wrap_conv_ops` captures those instead, and
#     [`assert_captured`]'s silent-hook check is what found this. (`ple.conv1d` is NOT here:
#     MOD:1165 really does call it, because the dilated conv has no fused counterpart.)
#   * `act_fn` is an `nn.SiLU` INSTANCE, so it is a submodule -- and inside
#     `Qwen4ExpTextExperts.forward` it is called once per HIT EXPERT (MOD:890). Its call count
#     is therefore a property of the routing, and any defect that moves the routing changes the
#     golden's tensor SET rather than its numbers: `--compare` then scores "absent on one side"
#     and the real signal drowns. Found here as a DUPLICATE NAME in the container's own reader
#     (two `mlp.experts.act_fn` entries in the first golden written), which is the same class of
#     failure K3 measured as `inf` rows on four of five defects. A stateless activation whose
#     output is a function of the input it was handed is also the cheapest thing to lose.
NEVER_FIRES = ("linear_attn.conv1d", "act_fn")


def _name_filters(layers):
    """The two name tests `_is_captured` applies, built once per hook pass.

    The trailing dot on `prefixes` is load-bearing and its absence is a MEASURED bug in K3's
    harness: without it `model.layers.2` also matched layers 20-29 and `model.layers.1` matched
    10-19, capturing 25 layers instead of 6. A layer's own output is reached by the exact-name
    test instead.
    """
    return tuple(f"model.layers.{i}." for i in layers), tuple(f"model.layers.{i}" for i in layers)


def _is_captured(name, prefixes, exact):
    """Whether a submodule gets a hook: inside the captured set, and not a never-fires family."""
    wanted = name in TAIL or name in exact or name.startswith(prefixes)
    return wanted and not name.endswith(NEVER_FIRES)


# Modules whose INPUT is the fixture, not just their output. One entry, and it is the n-gram
# hash: `Qwen4ExpTextNGramEmbedding.forward` computes the 16 hashed row ids and immediately looks
# them up, so the ids exist only as the argument to this `nn.Embedding`. They are what every
# n-gram defect moves and what S4's Rust re-derivation has to reproduce exactly, so capturing
# only the looked-up vectors would score the hash through a dense projection.
CAPTURE_INPUT = ("ple.ple_embedding.ngram_embedding",)


def _hook(cap, name):
    def run(_m, inputs, out):
        if name.endswith(CAPTURE_INPUT) and inputs:
            cap.add(f"{name}.in", inputs[0])
        fire(cap, name, out)

    return run


def hook_model(model, cap, layers):
    """A forward hook on every submodule of the captured layers, plus the model-level tail.

    Every submodule rather than a chosen few: at these widths the whole set is a few hundred
    kilobytes, and a hand-picked list is a list someone has to remember to extend when a module
    appears. Returns `(handles, expected_names)` -- the second half exists because a hook that
    never fires is silent.
    """
    prefixes, exact = _name_filters(layers)
    handles, expected = [], []
    for name, mod in model.named_modules():
        if not _is_captured(name, prefixes, exact):
            continue
        handles.append(mod.register_forward_hook(_hook(cap, name)))
        expected.append(name)
    return handles, expected


def assert_captured(tap, expected):
    """Refuse a golden whose capture was SHORT.

    Four independent counts, because each failure is silent on its own:

      * every registered hook produced at least one tensor. A dead hook is how an operator ends
        up with no fixture while the golden looks like a few hundred plausible tensors;
      * the GDN operator was reached once per `linear_attention` layer, and the conv once per
        call. A short count RELABELS captures -- layer 2's boundary written under layer 1's name
        -- which no assertion in `tests/qwen_anchor.rs` could see;
      * the indexer ran on every `qwen_sparse_attention` layer;
      * the PLE host ran exactly once, on the layer `ple_layer_ids` names.
    """
    cap, ctx = tap.cap, tap.ctx
    silent = sorted(set(expected) - cap.fired)
    assert not silent, f"{len(silent)} hooks never fired, e.g. {silent[:4]}"
    for key, want, what in (
        ("gdn_calls", ctx["gdn_layer_ids"], "GDN layers"),
        ("conv_calls", ctx["gdn_layer_ids"], "GDN short convolutions"),
        ("indexer_calls", ctx["qsa_layer_ids"], "QSA indexer layers"),
        ("ple_calls", ctx["ple_layer_ids"], "PLE hosts"),
    ):
        assert ctx[key] == len(want), f"{ctx[key]} {what} reached, expected {len(want)}"
    # And each tapped family left a fixture behind, named -- **but only where a layer of that
    # family is in the capture list.** A count above can be right while the capture wrote nothing,
    # because `ctx["capturing"]` gates the recording and the counters do not; and an unconditional
    # list would refuse the narrow prefill window, whose whole purpose is one QSA layer. The
    # condition is derived from the layer lists rather than declared, so narrowing the capture
    # cannot silently drop a family that IS in it.
    families = (
        ("gdn_layer_ids", ("gated_delta_rule.out.o", "linear_attn.conv.out")),
        ("qsa_layer_ids", ("self_attn.indexer.scores",)),
        ("ple_layer_ids", ("ple.ple_embedding.ngram_embedding.in",)),
    )
    for key, fragments in families:
        if not set(ctx[key]) & set(tap.layers):
            continue
        for fragment in fragments:
            assert any(fragment in name for name in cap.seen), f"nothing captured for {fragment}"
