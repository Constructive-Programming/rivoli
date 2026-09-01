"""What this run WAS, and the refusal that keeps it worth recording.

Split out of `qwen_anchor_driver` on 2026-09-01, when that file crossed the 800-line soft cap.
One concern in two halves, and they belong together because the second is only trustworthy
while the first holds: `preflight_env` refuses a venv that could have produced different
arithmetic, and `_metadata` stamps into every golden the facts a later reader would otherwise
have to take on faith -- versions, file hashes, config widths, and the forbidden-package list
that was asserted absent.

**`FORBIDDEN_PACKAGES` moved with both of its users and nothing else reads it.** `fla`,
`kernels` and `causal_conv1d` each substitute a different implementation under the same call,
so a venv carrying any of them yields goldens that are not the ones this anchor claims to pin.
The driver ASSERTS their absence rather than trusting the venv, and stamps that assertion into
the file -- which is why the constant is provenance rather than configuration.

This module imports torch and transformers, so unlike `qwen_anchor_lib` and
`qwen_anchor_compare` it is NOT part of the deviceless half. Nothing here is needed to read a
golden back or to score two of them, and that is the line the split was cut along.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import torch
import transformers

from qwen_anchor_compare import EXPECT_FIRST_TOUCH
from qwen_anchor_defects import CLASSES
from qwen_anchor_lib import (
    STRUCTURAL,
    derived_widths,
    read_golden,
    real_config_widths,
    width_audit_pairs,
)

# The packages whose presence would silently change what the golden pins. All three are checked
# and all three must be ABSENT: `fla` and `causal_conv1d` would replace the torch bodies through
# `use_kernel_func_from_hub_with_fallback`, and `kernels` would let `kernelize` pull a hub kernel
# on top of either.
FORBIDDEN_PACKAGES = ("fla", "kernels", "causal_conv1d")
# ---------------------------------------------------------------------------------------------
# The reference, and refusing the wrong one.


def sha_of(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def _reference_paths():
    """The two pinned reference blobs, inside the venv. The model repo carries no `auto_map` and
    no `modeling_*.py`, so the in-tree transformers class IS the shipped implementation."""
    from transformers.models import qwen4_exp

    d = pathlib.Path(qwen4_exp.__file__).parent
    return d / "modeling_qwen4_exp.py", d / "configuration_qwen4_exp.py"


def preflight_env():
    """Refuse to generate under an environment other than the one the vendored goldens were made
    in, and under any environment that would silently swap an implementation.

    Two halves. The version pins are READ OUT OF THE VENDORED GOLDEN, never restated here: those
    bytes already carry what produced them and `qwen_anchor.rs` already asserts them, so a copy
    in this file would be a third statement of one fact. A deliberate re-pin therefore needs no
    edit here -- regenerate, re-vendor, and the new bytes become the new pin.

    The forbidden-package half is not a version check and cannot be read off a golden that was
    produced correctly: `fla`, `causal_conv1d` and `kernels` each REPLACE a torch body at import
    time through `use_kernel_func_from_hub_with_fallback`, and the replacement keeps the torch
    function's `__name__` (it is `functools.wraps`-ed), so the golden's own metadata would still
    say `torch_chunk_gated_delta_rule`. The name lies; the absence is what has to be asserted.
    """
    import transformers

    present = [p for p in FORBIDDEN_PACKAGES if importlib.util.find_spec(p) is not None]
    if present:
        raise SystemExit(
            f"qwen-anchor: {present} installed in this env. Each one REPLACES a pure-torch\n"
            "  reference body through the kernel-from-hub fallback while keeping its name, so\n"
            "  the golden would pin a different implementation and say otherwise. Use a venv\n"
            "  without them."
        )
    vendored = sorted(pathlib.Path(__file__).parent.glob("qwen-anchor-decode-*.bin"))
    if not vendored:
        print("qwen-anchor: no vendored golden to check this env against", file=sys.stderr)
        return
    pinned = read_golden(vendored[0])[0]
    live = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }
    drift = [f"{k}: golden says {pinned.get(k)!r}, this env has {v!r}" for k, v in live.items() if pinned.get(k) != v]
    if drift:
        raise SystemExit(
            f"qwen-anchor: this env is not the one that produced {vendored[0].name}:\n  "
            + "\n  ".join(drift)
            + "\n  Fix QWEN_ANCHOR_VENV, or re-vendor the bytes AND the versions qwen_anchor.rs asserts."
        )


# ---------------------------------------------------------------------------------------------
# Metadata.


def _ple_conv_dilation(model, cfg):
    """The dilation the PLE's `nn.Conv1d` was actually built with, and the assert that it is
    `ngram_size`.

    Asserted here rather than in the width audit because this is the one form of the claim that
    can fail: the audit's version compared `derived_widths`' `ple_conv_dilation` -- defined AS
    `cfg.ngram_size` -- to `cfg.ngram_size`. `nn.Conv1d.dilation` is a tuple; the conv is 1-D, so
    there is exactly one entry and indexing it is not a choice.
    """
    host = cfg.ple_layer_ids[0] - 1
    dilation = model.model.layers[host].ple.conv1d.dilation
    assert len(dilation) == 1, f"the PLE conv is 1-D; dilation is {dilation}"
    assert dilation[0] == cfg.ngram_size, (
        f"the PLE dilated conv was built with dilation {dilation[0]}, not ngram_size "
        f"{cfg.ngram_size} -- the n-gram stride and the conv stride have come apart"
    )
    return dilation[0]


def _metadata(args, model, cfg, cap):
    """Everything the bytes have to carry about what produced them.

    A golden that hides what produced it is worse than no golden -- and here that includes the
    three implementation SELECTORS, because each of them defaults to something other than what
    this run pins and none of them is visible in the numbers.
    """
    import transformers

    mod_sha, cfg_sha = (sha_of(p) for p in _reference_paths())
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
        ("python", ".".join(str(v) for v in sys.version_info[:3])),
        ("attn_implementation", model.config._attn_implementation),
        ("experts_implementation", model.config._experts_implementation),
        # Which pure-torch body actually ran, by name, and which conv entry point. Asserted
        # rather than assumed: the fallback decorator keeps the torch function's `__name__`
        # whatever it dispatched to, so this is only evidence alongside the forbidden-package
        # check in `preflight_env`.
        ("gdn_entry", ",".join(sorted(model.config.__dict__.get("_qwen_anchor_gdn_tags", ())))),
        ("conv_entry", ",".join(sorted(model.config.__dict__.get("_qwen_anchor_conv_tags", ())))),
        ("forbidden_absent", ",".join(FORBIDDEN_PACKAGES)),
        # The layers ACTUALLY hooked, not the module constant: a narrowed run whose metadata
        # claimed the full list would let `qwen_anchor.rs`'s census pass over a short capture.
        ("capture_layers", ",".join(str(i) for i in args.capture_layers_parsed)),
        # Which module names recurred within one forward, and how often. Empty for every decode
        # golden and thirteen-deep for the prefill one, whose indexer norm runs once per query
        # position -- see `Capture.add`. In the bytes because a reader counting distinct names
        # has to know the count is not the number of tensors, and because the ladder of shapes
        # under the repeated name IS the dense/selective boundary this fixture exists to show.
        ("repeated_captures", json.dumps(cap.repeated(), sort_keys=True)),
        ("structural_asserted", ",".join(STRUCTURAL)),
        ("defect_classes", json.dumps({k: list(v) for k, v in CLASSES.items()}, sort_keys=True)),
        ("tiny_config", json.dumps(cfg.to_dict(), sort_keys=True, default=str)),
        ("widths", json.dumps(derived_widths(cfg), sort_keys=True)),
        # The UNSHRUNK widths beside the tiny ones, so the audit's real-side half is re-runnable
        # from the bytes. Without them `qwen_anchor.rs` could only re-check that the tiny config
        # separates what the list says it separates -- never that the list's claim ABOUT THE
        # CHECKPOINT is true, which is the half that was false until 2026-08-31.
        ("real_widths", json.dumps(real_config_widths(type(cfg), args.config), sort_keys=True)),
        ("width_audit", json.dumps(width_audit_pairs(), sort_keys=True)),
        # The PLE dilated conv's dilation, read off the CONSTRUCTED module rather than restated
        # from `ngram_size`. It replaces an `EQUAL_PAIRS` row that compared `ngram_size` to
        # itself; see that list's dated note. The host layer is `ple_layer_ids` minus one,
        # because the field is one-indexed on disk (MOD:1202).
        ("ple_conv_dilation_observed", str(_ple_conv_dilation(model, cfg))),
        ("first_touch", json.dumps({k: list(v) for k, v in EXPECT_FIRST_TOUCH.items()}, sort_keys=True)),
        ("ref_modeling_sha256_16", mod_sha),
        ("ref_config_sha256_16", cfg_sha),
        ("real_config_sha256_16", sha_of(args.config)),
    ]
