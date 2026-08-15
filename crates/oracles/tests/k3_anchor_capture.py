"""The captured tensors, and the forward hooks that a `register_forward_hook` can supply.

Split out of `k3_anchor_driver.py` on 2026-08-15 under the 800-line-per-file gate. Every body
below moved verbatim, comments and measurements with it.

**This half is the mechanism that works by itself.** `hook_model` registers one hook per
submodule of the captured layers and `Capture` holds what they produce. The two values that no
forward hook can reach -- the AttnRes fold, whose modules are read rather than called, and fla's
KDA operator, which is a free function -- need a wrapper each, and those live in
`k3_anchor_taps.py`. The split is that boundary; each tap's own docstring records what its
absence cost when the harness had only the hooks.
"""

import collections


# **`torch` resolved on first attribute access, never at import.** `k3_anchor_driver` binds its
# own `torch` inside `main` for one stated reason -- `--compare` re-scores vendored bytes with no
# torch, no fla and no GPU -- and it imports this module at its top to re-export the names below.
# A plain `import torch` here would take that property away, and a `torch = None` that `main`
# assigned would change `main`; the split is a pure move and may change no body. The proxy costs
# one attribute lookup per access, on a path that is already calling into torch, and leaves every
# body below saying `torch.x` exactly as it did before the split. `k3_anchor_taps` imports this
# object rather than restating it -- rebinding never happens, so the import is the whole story.
class _LazyTorch:
    def __getattr__(self, name):
        import torch

        return getattr(torch, name)


torch = _LazyTorch()


# **The three things every tap and the capture pass all need**, in one value: where tensors go,
# the run's mutable `ctx` (which the DEFECTS write into), and which layers are captured. Bundled
# because each of those functions takes exactly these three alongside one or two arguments of its
# own, which put five and six parameters on functions whose real arity is two. A `namedtuple` and
# not a class: it holds no behaviour, and the three fields are read positionally nowhere.
Tap = collections.namedtuple("Tap", "cap ctx layers")


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


# The model-level tail, captured alongside the per-layer submodules: the final norm and the head
# are the two boundaries no layer prefix reaches.
TAIL = ("model.norm", "lm_head")


def _name_filters(layers):
    """The three name tests `_is_captured` applies, built once per hook pass.

    The trailing dot on `prefixes` is load-bearing, and its absence was measured: an earlier
    version also matched the bare `model.layers.{i}`, so `model.layers.1` caught layers 10-19 and
    `model.layers.3` caught 30-39 -- 25 captured layers instead of 6. The layer's own output is
    reached by the exact-name test instead.
    """
    return tuple(f"model.layers.{i}." for i in layers), tuple(f"model.layers.{i}" for i in layers)


def _is_captured(name, prefixes, exact):
    """Whether a submodule name gets a hook: in the captured set, and not one of the two
    never-fires families.

    Everything at or under `.experts` is excluded, and the reason is not size. `moe_infer` calls
    only the experts that WON tokens, so which expert modules fire is routing-dependent -- any
    defect that moves the routing changes the golden's tensor SET, and every such comparison then
    scores "absent on one side" rather than a number. Measured: the first defect matrix reported
    `inf` for most layers on four of five defects for exactly this reason, drowning the real
    signal. `topk_idx`/`topk_weight` and the block output are the routing fixture and are always
    present.

    `.experts` and not `.experts.`, so the `ModuleList` ITSELF goes too: its forward is never
    called, so its hook could never fire, and the silent-hook assertion at the end of `main`
    caught all five of them on its first run. `shared_experts` is untouched -- the substring needs
    a dot before `experts`, and there the preceding character is an underscore.

    The four AttnRes modules per layer are the OTHER never-fires case: `_apply_attn_res` reads
    their `.weight` directly instead of calling them. `wrap_attn_res` captures the fold; hooking
    them here would register four dead hooks per layer.
    """
    wanted = name in TAIL or name in exact or name.startswith(prefixes)
    never_fires = ".experts" in name or name.endswith(("_res_norm", "_res_proj"))
    return wanted and not never_fires


def _hook_targets(model, layers):
    """The `(name, module)` pairs that get a hook, in `named_modules` order."""
    prefixes, exact = _name_filters(layers)
    return [(n, m) for n, m in model.named_modules() if _is_captured(n, prefixes, exact)]


def hook_model(model, cap, layers):
    """A forward hook on every submodule of the captured layers, plus the model-level tail.

    Every submodule rather than a chosen few: at these widths the whole set is a few hundred
    kilobytes, and a hand-picked list is a list someone has to remember to extend when a module
    appears.

    Returns `(handles, expected_names)`. The second half exists because a hook that never fires
    is silent, and that silence cost this harness its AttnRes coverage for a day — see
    `wrap_attn_res`.
    """
    handles, expected = [], []
    for name, mod in _hook_targets(model, layers):
        handles.append(mod.register_forward_hook(lambda m, i, o, n=name: _fire(cap, n, o)))
        expected.append(name)
    return handles, expected
