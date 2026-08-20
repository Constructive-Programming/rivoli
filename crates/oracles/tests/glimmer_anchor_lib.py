"""The S1b anchor's fixture kit and golden container -- everything that is not a run.

Split out of `glimmer_anchor_driver.py` 2026-08-15 to hold every authored file under the 800-line
cap; the driver imports and re-exports it, and **nothing here changed in the move** -- the bodies,
their comments and the constants are the ones that produced the vendored goldens.

Three things live here because they share one property: none of them runs the reference. The tiny
configs SAY what to build, `Capture` and the container say how a run is written down and read back,
and `init_weights`/`prompt_ids` say what numbers a salt names. The runs and the taps that watch them
are the driver's; the defects are `glimmer_anchor_defects.py`'s.

`preflight_env` reads `__file__`'s directory, and that survived the move because this file is a
sibling of the driver in `crates/oracles/tests/` -- the vendored goldens it globs are the same six.
"""

import hashlib
import json
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
        # width. **NOT the target's tiny 4, and the difference is the whole point.** ctx is
        # PROMPT_LEN 12 and block_size 4, so kv_len is 16 and the block's own K/V rows sit at
        # 12..16. The reference indexes queries by ROW (`q_offset = 0`, no cache) while RoPE
        # places them at ctx.., so the mask is `|q_row - kv| <= w`: at w=4 the furthest query
        # reached kv 7 and the block-vs-block submatrix summed to EXACTLY 0.0 — no query ever
        # attended the block, so §11 step 5 (attention is bidirectional ACROSS THE BLOCK) was
        # pinned by the mask pattern and by no value at all, and `defect_causal_mask`'s red was
        # the re-selection of CONTEXT rows rather than the property under test. Measured and
        # re-vendored 2026-08-16.
        #
        # 13 is the MINIMUM value that produces a strictly-bidirectional pair (a query attending
        # a LATER block row -- exactly what a causal mask forbids), and minimal on purpose: 13 of
        # the 16 block-vs-block pairs attend and 3 stay masked, so the window still binds inside
        # the block, where w >= 15 would make the whole mask all ones and unable to fail. The
        # binding constraint is bidirectionality, NOT reach: at w=11 six block pairs already
        # attend (the block is wholly out of reach only at w <= 8), but a query attending a LATER
        # block row needs 12 + r - q <= w with r > q, i.e. w >= 13.
        #
        # THE COST, derived rather than discovered later: at w >= 12 the CONTEXT half of the mask
        # is all ones, so this fixture no longer exercises window-masking of the context. That
        # trade is FORCED, not a tuning choice -- swept over w at ctx 12 / block 4:
        #
        #   w               :  2 .. 8   9   10   11   12   13   14   15+
        #   ctx cols masked : 31 .. 6   3    1    0    0    0    0    0
        #   bidir pairs     :  0 .. 0   0    0    0    0    3    5    6
        #
        # The two never overlap: context masking needs w <= 10, bidirectionality needs w >= 13.
        # It follows from the reference's own mask form -- `q_offset = 0` indexes queries by ROW
        # (0..block) while K/V spans ctx+block, so any w letting q0 see kv=ctx also lets it see
        # every kv < ctx. No geometry with this mask has both, at any ctx. Section 11 step 5 is
        # the property under test, so the block wins; a context-window defect is now this
        # fixture's DECLARED blind spot, asserted in glimmer_draft_oracle.rs.
        sliding_window=13,
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


def _centered_norm_owners(model):
    """The module paths whose weight is applied as `(1 + w)` instead of `x * w`.

    Telling the two families apart by module TYPE rather than by name is deliberate: both are
    called `...norm.weight`, so a name rule would have to guess.
    """
    centered = set()
    for mod_name, mod in model.named_modules():
        if type(mod).__name__.endswith("CenteredRMSNorm"):
            centered.add(mod_name)
    return centered


def _draw_into(flat, owner, centered, g):
    """The three-way draw, keyed to how the owning module APPLIES the weight it is given.

    A centered norm filled near 1.0 doubles every activation it touches, and a golden of
    exponentially growing numbers agrees with nothing -- see `init_weights`.
    """
    if owner in centered:
        flat.uniform_(-0.2, 0.2, generator=g)  # applied as (1 + w)
    elif "norm" in owner.split(".")[-1]:
        flat.uniform_(0.8, 1.2, generator=g)  # applied as (x * w)
    else:
        flat.uniform_(-0.08, 0.08, generator=g)


def _restore_padding_row(model):
    """`post_init` zeroes the `padding_idx` row and the fill loop overwrote it. Put it back:
    the zeroing is the reference's own behaviour.

    Called from INSIDE `init_weights`' `no_grad` block, where it was written -- the write is
    in-place on a leaf parameter and needs that block as much as the fill loop does.

    **The drafter raises here rather than returning None**, because it owns no embedding at
    all -- it borrows the target's (section 11). That is a documented structural fact, so it
    is caught rather than worked around, and a DIFFERENT failure would still propagate.
    """
    try:
        emb = model.get_input_embeddings()
    except NotImplementedError:
        emb = None
    if emb is not None and getattr(emb, "padding_idx", None) is not None:
        emb.weight[emb.padding_idx].zero_()


def init_weights(model, salt):
    """Fill every parameter deterministically, in two families.

    **The centered norms are the reason this is not one uniform draw.** `MuseGlimmerTextCenteredRMSNorm`
    stores `w` and applies `(1 + w)`, so its weights are initialised near ZERO, not near one -- a
    centered norm filled near 1.0 doubles every activation it touches and a golden of exponentially
    growing numbers agrees with nothing. The drafter's plain `MuseGlimmerAssistantRMSNorm` and the
    target's final `MuseGlimmerRMSNorm` apply `x * w` and take the near-one draw. Telling them apart
    by module TYPE rather than by name is deliberate: both are called `...norm.weight`.

    The three steps below -- collect the centered set, draw into each parameter, put the padding row
    back -- are the ones this body always had; `_centered_norm_owners`, `_draw_into` and
    `_restore_padding_row` are those steps under their own names, split 2026-08-15 for the CodeScene
    10.0 gate (cc 10, two bumps). The draw order, the generator per parameter and every bound are
    untouched, which is what lets the vendored goldens still be this function's output.
    """
    centered = _centered_norm_owners(model)
    with torch.no_grad():
        for name, p in model.named_parameters():
            g = _gen(name, salt)
            flat = torch.empty(p.numel(), dtype=torch.float32)
            owner = name.rsplit(".", 1)[0]
            _draw_into(flat, owner, centered, g)
            p.copy_(flat.view(p.shape).to(p.dtype))
        _restore_padding_row(model)


def prompt_ids(salt, vocab, n=PROMPT_LEN):
    g = _gen("prompt", salt)
    return torch.randint(0, vocab, (1, n), generator=g)


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

    `transformers_commit` comes from pip's own `direct_url.json` rather than from a constant: when
    the package is installed from a git URL at a pinned revision, the commit is a fact about the
    environment and restating it here would be a second copy to drift.

    `assistant_modeling_sha256` is emitted alongside it, unconditionally, and is for this
    model the better pin. A PyPI
    RELEASE has no `direct_url.json` and so no commit, and the sha of a transformers git revision
    would not tell you the content of the four files that actually produce these captures anyway.
    This hashes those four directly, sorted by name, so the provenance is the SOURCE rather than a
    label on the repository it came from. Added 2026-08-16, when the venv holding the pinned
    revision stopped existing and the only stack left on the machine was `transformers==5.15.0`
    from PyPI — whose assistant modeling files were proven to reproduce all 50 captures of both
    vendored draft goldens bit-identically before anything was re-vendored under it.
    """
    import numpy
    import transformers

    commit = "unknown"
    for d in pathlib.Path(transformers.__file__).parent.parent.glob("transformers-*.dist-info"):
        p = d / "direct_url.json"
        if p.exists():
            commit = json.loads(p.read_text()).get("vcs_info", {}).get("commit_id", "unknown")
    src = pathlib.Path(transformers.__file__).parent / "models" / "muse_glimmer_assistant"
    h = hashlib.sha256()
    for f in sorted(src.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_commit": commit,
        "assistant_modeling_sha256": h.hexdigest(),
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
    }
