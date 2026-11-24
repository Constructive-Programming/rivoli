#!/usr/bin/env python3
"""The S3 Inkling 2 anchor: goldens from the first-party `transformers.models.inkling2_loop`
stack (transformers @ d8f10a82), not a transliteration. Runs the text model
(`Inkling2LoopForCausalLM`) over a tiny config derived from the real Mini config — widths shrink,
the 24-layer / block-6 / loop-6-18-x3 structure stays — with seeded weights, on CPU fp32.

Two protocols, both the model's own:

- `--mode prefill` runs the prompt `0..PREFILL_SEQ` uncached;
- `--mode decode` prefills `0..PREFILL_SEQ` with `use_cache=True` (no hooks) and then steps ONE
  token against the resulting `DynamicCache`. That is real incremental decode: the KV append,
  `causal_conv1d_update`'s conv-state shift instead of a full-window conv, and the loop passes'
  per-pass `cache_offset` (T16) are all exercised, and the buckets are `[1, 1, …]`.

Capture layers inside the loop range `[6, 18)` run once per loop pass, so each stores every pass
(`model.layers.{L}.pass{p}.hidden`) beside the last-pass `model.layers.{L}.hidden`.

The golden container is byte-for-byte the layout `crates/oracles/src/golden.rs` reads under the
`RIVILGLD` magic that landed at S1. `--compare A B` scores two goldens and asserts the defect's
declared first-touched bucket; `--tolerance` measures the fp32-vs-fp64 floor per operator on the
same weights and the same decode protocol. No GPU: the short convolutions fall back to the
reference PyTorch body when `causal_conv1d` is absent (asserted), which is the arithmetic this
anchor pins.

Usage:
    python3 inkling_anchor_driver.py --mode decode --defect None --salt inkling-1 --out gold.bin
    python3 inkling_anchor_driver.py --compare A.bin B.bin --defect l2clip_no_clamp
    python3 inkling_anchor_driver.py --tolerance --salt inkling-1
"""
import argparse
import hashlib
import math
import json
import pathlib
import struct
import sys

import torch
from transformers.models.inkling2_loop import Inkling2LoopForCausalLM, Inkling2LoopTextConfig
import transformers.models.inkling2_loop.modeling_inkling2_loop as M

MAGIC = b"RIVILGLD"
CONFIG = pathlib.Path(__file__).resolve().parents[3] / "docs/measurement/inkling2-reference/mini/anchor-config.json"
# Capture layers: 0 (first sliding, first MoE), 5 (first full / block end), 6 (first loop layer),
# 11 (full block end inside the loop), 17 (last loop layer), 23 (last full / block end).
CAPTURE_LAYERS = (0, 5, 6, 11, 17, 23)
DECODE_SEQ = 17
PREFILL_SEQ = 16


def assert_no_fast_conv():
    for mod in ("causal_conv1d",):
        try:
            __import__(mod)
        except ImportError:
            continue
        raise SystemExit(f"{mod} is installed — the conv would not be the reference body this anchor pins")


# ── golden container (mirrors qwen_anchor_lib; golden.rs reads it) ──────────────────────────

def _u64(v):
    return struct.pack("<Q", v)


def _s(x):
    b = x.encode()
    return _u64(len(b)) + b


def write_golden(path, meta, floats, ints):
    out = bytearray(MAGIC)
    out += _u64(len(meta))
    for k, v in meta:
        out += _s(k) + _s(v)
    for items, fmt in ((floats, "<f"), (ints, "<q")):
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
    b = pathlib.Path(path).read_bytes()
    assert b[:8] == MAGIC, f"{path}: not a rivoli Inkling anchor golden"
    o = [8]

    def u64():
        v = struct.unpack_from("<Q", b, o[0])[0]
        o[0] += 8
        return v

    def s():
        n = u64()
        v = b[o[0]:o[0] + n].decode()
        o[0] += n
        return v

    meta = [(s(), s()) for _ in range(u64())]
    sections = []
    for width, code in ((4, "f"), (8, "q")):
        items = {}
        for _ in range(u64()):
            name = s()
            shape = [u64() for _ in range(u64())]
            n = u64()
            raw = b[o[0]:o[0] + n * width]
            vals = struct.unpack_from(f"<{n}{code}", b, o[0])
            o[0] += n * width
            assert name not in items, f"{path}: duplicate tensor {name}"
            items[name] = (tuple(shape), vals, raw)
        sections.append(items)
    assert o[0] == len(b), f"{path}: trailing bytes"
    return dict(meta), sections[0], sections[1]


# ── model + capture ─────────────────────────────────────────────────────────────────────────

def build_model(salt, dtype=torch.float32):
    cfg = Inkling2LoopTextConfig(**json.loads(CONFIG.read_text()))
    cfg.eos_token_id = None
    seed = int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    model = Inkling2LoopForCausalLM(cfg).to(dtype)
    model.eval()
    return model, cfg, seed


def input_ids(cfg, seq):
    # A fixed, salt-independent prompt so a defect's effect is the arithmetic, not the tokens.
    return torch.arange(seq).remainder(cfg.vocab_size).unsqueeze(0)


def run_pair(model, cfg, mode, arm=None):
    """One forward under the anchor's protocol, returning the final CausalLM output.

    `prefill` is the whole prompt `0..PREFILL_SEQ` uncached. `decode` is the REAL cached step:
    the prompt is prefilled with `use_cache=True` and NO hooks (that pass only populates the
    `DynamicCache` the reference builds for a hybrid model — 48 layer entries, because the
    text model appends `loop_num_passes - 1` deep copies of the loop layers' caches — plus four
    conv states per layer), then token `PREFILL_SEQ` is stepped against that cache. That is the
    only way the golden exercises the incremental body: `causal_conv1d_update`'s state shift
    instead of a full-window conv, the KV append, and the loop passes' `cache_offset` (T16).

    `arm()` installs the caller's hooks and returns its own teardown; it runs BETWEEN the two
    forwards, so a decode golden's buckets are the stepped token and never the prefill.
    """
    ids = input_ids(cfg, DECODE_SEQ if mode == "decode" else PREFILL_SEQ)
    cache = None
    if mode == "decode":
        with torch.no_grad():
            cache = model(input_ids=ids[:, :PREFILL_SEQ], use_cache=True).cache
    disarm = arm() if arm is not None else None
    try:
        with torch.no_grad():
            if cache is None:
                return model(input_ids=ids, use_cache=False)
            return model(input_ids=ids[:, PREFILL_SEQ:], cache=cache, use_cache=True)
    finally:
        if disarm is not None:
            disarm()


def capture(model, cfg, mode):
    """Hook every capture layer's hidden-state output plus the final logits.

    A capture layer inside `[loop_layer_index_start, loop_layer_index_end)` is applied once per
    loop pass, so it fires `loop_num_passes` times in ONE forward. `model.layers.{idx}.hidden`
    keeps the LAST pass (the value the following layer consumed) and every pass is also stored
    as `model.layers.{idx}.pass{p}.hidden` — the three passes are different arithmetic (pass 2+
    restart from the pre-loop states with a message mixed in, MOD:542), so a golden that kept
    only the last one could not see the recurrence at all. `capture_census` accepts the pass
    names because `captured_layers` reads the index before the first field dot.

    Returns (floats, ints, looped-layer-indices) for the golden and its metadata."""
    taps = {}
    calls = {}
    lo, hi = cfg.loop_layer_index_start, cfg.loop_layer_index_end

    def layer_hook(idx):
        def fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach()
            calls[idx] = calls.get(idx, 0) + 1
            taps[f"model.layers.{idx}.hidden"] = h
            if lo <= idx < hi:
                taps[f"model.layers.{idx}.pass{calls[idx]}.hidden"] = h
        return fn

    def arm():
        hooks = [model.model.layers[idx].register_forward_hook(layer_hook(idx)) for idx in CAPTURE_LAYERS]

        def disarm():
            for h in hooks:
                h.remove()
        return disarm

    out = run_pair(model, cfg, mode, arm)
    taps["logits"] = out.logits.detach()

    looped = [idx for idx in CAPTURE_LAYERS if lo <= idx < hi]
    for idx in looped:
        assert calls.get(idx) == cfg.loop_num_passes, (
            f"layer {idx} fired {calls.get(idx)} times, not {cfg.loop_num_passes} loop passes")

    floats = []
    for name, t in taps.items():
        flat = t.reshape(-1).to(torch.float32)
        floats.append((name, list(t.shape), flat.tolist()))
    return floats, [], looped


# ── the defect matrix: each row plants one trap in the reference body ───────────────────────
# Converter-level traps (T1/T2/T3/T12/T13/T14) are gated by S2's dequant/census/name gates, not
# here; these are the FORWARD-operable traps. Each entry: (name, first_touched_bucket, applier).

def _patch(cls, method, fn):
    setattr(cls, method, fn)


def defect_l2clip_no_clamp():
    orig = M.Inkling2LoopL2Clip.forward
    def fwd(self, hidden_states):
        input_dtype = hidden_states.dtype
        h = hidden_states.float()
        norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
        # T4 family: drop the clamp(max=1.0) so vectors SHORTER than threshold are scaled UP.
        # The +1e-6 epsilon alone is sub-fp32 at these norms (measured vacuous), so the clamp is
        # the observable L2Clip misreading at tiny widths.
        return (h * (self.threshold / (norm + 1e-6))).to(input_dtype)
    _patch(M.Inkling2LoopL2Clip, "forward", fwd)
    return lambda: _patch(M.Inkling2LoopL2Clip, "forward", orig)


def defect_softcap_sigmoid_capped():
    orig = M.Inkling2LoopSoftcappedSwiGLU.forward
    def fwd(self, gate, up):
        input_dtype = gate.dtype
        gate, up = gate.float(), up.float()
        g = self.gate_softcap(gate)
        return (g * torch.sigmoid(g) * self.up_softcap(up)).to(input_dtype)  # T6: sigmoid the CAPPED gate
    _patch(M.Inkling2LoopSoftcappedSwiGLU, "forward", fwd)
    return lambda: _patch(M.Inkling2LoopSoftcappedSwiGLU, "forward", orig)


def defect_sconv_no_divisor():
    orig = M.Inkling2LoopShortConvolution.forward
    def fwd(self, hidden_states, *a, **k):
        out = orig(self, hidden_states, *a, **k)
        return out * math.sqrt(self.conv_kernel_size + 1)  # T15: remove the /sqrt(k+1) divisor
    _patch(M.Inkling2LoopShortConvolution, "forward", fwd)
    return lambda: _patch(M.Inkling2LoopShortConvolution, "forward", orig)


def defect_router_no_route_scale():
    orig = M.Inkling2LoopTopKRouter.forward
    def fwd(self, hidden_states):
        logits, weights, indices = orig(self, hidden_states)
        # T5 family: drop the routed_scaling_factor. (The bias-in-weights plant is vacuous at
        # random init — e_score_correction_bias is a zero buffer — so route_scale is the
        # observable router misreading here.)
        return logits, weights / self.routed_scaling_factor, indices
    _patch(M.Inkling2LoopTopKRouter, "forward", fwd)
    return lambda: _patch(M.Inkling2LoopTopKRouter, "forward", orig)


def defect_logscale_wrong_floor():
    orig = M.Inkling2LoopLogScale.forward
    def fwd(self, hidden_states, position_ids):
        input_dtype = hidden_states.dtype
        scale = 1.0 + self.log_scaling_alpha * torch.log(
            ((position_ids + 1).float() / 128000.0).clamp(min=1.0)  # T9: class default floor, not config
        )
        return (hidden_states.float() * scale[..., None, None]).to(input_dtype)
    _patch(M.Inkling2LoopLogScale, "forward", fwd)
    return lambda: _patch(M.Inkling2LoopLogScale, "forward", orig)


DEFECTS = {
    "None": (None, None),
    "l2clip_no_clamp": ("model.layers.0.hidden", defect_l2clip_no_clamp),
    "softcap_sigmoid_capped": ("model.layers.0.hidden", defect_softcap_sigmoid_capped),
    "sconv_no_divisor": ("model.layers.0.hidden", defect_sconv_no_divisor),
    "router_no_route_scale": ("model.layers.0.hidden", defect_router_no_route_scale),
    # LogScale lives in the FULL layers' LogScaledL2Clip; layer 0 is sliding, so the first
    # captured bucket this defect touches is layer 5 (the first full / block-end layer).
    "logscale_wrong_floor": ("model.layers.5.hidden", defect_logscale_wrong_floor),
}


# ── modes ───────────────────────────────────────────────────────────────────────────────────

def run_capture(mode, defect, salt, out):
    assert_no_fast_conv()
    seq = DECODE_SEQ if mode == "decode" else PREFILL_SEQ
    restore = None
    if defect != "None":
        _, applier = DEFECTS[defect]
        restore = applier()
    model, cfg, seed = build_model(salt)
    floats, ints, looped = capture(model, cfg, mode)
    if restore:
        restore()
    meta = [
        ("model", "inkling2_loop"),
        ("entry_point", "Inkling2LoopForCausalLM"),
        ("mode", mode),
        # The protocol is metadata, not a convention: `decode` is the cached step (prefill with
        # `use_cache=True`, then ONE token against that cache), so a reader can tell a golden of
        # the incremental body from a golden of a full forward of the same length.
        ("protocol", "cached-decode" if mode == "decode" else "uncached-prefill"),
        ("prefill_seq", str(PREFILL_SEQ)),
        ("defect", defect),
        ("salt", salt),
        ("seed", str(seed)),
        ("dtype", "float32"),
        ("seq", str(seq)),
        ("transformers", __import__("transformers").__version__),
        ("capture_layers", ",".join(map(str, CAPTURE_LAYERS))),
        ("capture_passes", ",".join(map(str, looped))),
        ("loop_passes", str(cfg.loop_num_passes)),
    ]
    n = write_golden(out, meta, floats, ints)
    print(f"wrote {out}: {n} bytes, {len(floats)} tensors, defect={defect} salt={salt}")


def compare(a_path, b_path, defect):
    _, af, _ = read_golden(a_path)
    _, bf, _ = read_golden(b_path)
    assert set(af) == set(bf), "tensor sets differ"
    changed = []
    for name in af:
        if af[name][2] != bf[name][2]:
            changed.append(name)
    first = DEFECTS[defect][0]
    if defect == "None":
        assert not changed, f"None defect changed {changed}"
        print("compare None: identical")
        return
    assert changed, f"defect {defect} changed NOTHING"
    assert first in changed, f"defect {defect}: first-touched {first} not in changed {changed}"
    print(f"compare {defect}: {len(changed)} tensors changed, first-touched {first} present")


def _floor(f, x):
    """max|f(x) in fp32 - f(x) in fp64|: the operator's own rounding, on ONE captured input."""
    return (f(x.float()) - f(x.double()).float()).abs().max().item()


def _weightless_rms(x, eps):
    """`Inkling2LoopUnweightedRMSNorm` WITHOUT its fp32 cast, so the fp64 arm stays fp64.

    The reference body does `hidden_states.float()` first, which would make a "fp64" carve
    compute in fp32 and report a floor of zero — the same fp32-locked trap the model-level
    comparison falls into.
    """
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _carve_rmsnorm(mod):
    h, w, eps = mod._anchor_inputs[0], mod.weight, mod.variance_epsilon

    def f(x):
        return w.to(x.dtype) * _weightless_rms(x, eps)
    return _floor(f, h), "captured input, weight cast"


def _carve_swiglu(mod):
    gate, up = mod._anchor_inputs[:2]
    gc, uc = mod.gate_softcap.softcap, mod.up_softcap.softcap

    def f(g):
        return (torch.tanh(g / gc) * gc) * torch.sigmoid(g) * (torch.tanh(up.to(g.dtype) / uc) * uc)
    return _floor(f, gate), "captured gate/up"


def _carve_clip(mod):
    """`Inkling2LoopL2Clip` / `Inkling2LoopScaledL2Clip`, on a NON-VACUOUS input.

    At the tiny widths every captured head vector's norm sits below `threshold`, so
    `clamp(max=1.0)` is the identity and the fp32-vs-fp64 floor is exactly 0 — a carve that
    measures nothing (it was reported as 0 for both clips until 2026-11-24). The captured input
    is therefore scaled up until `threshold / (norm + 1e-6) < 1.0` for the largest row, which is
    the branch that carries the clamp, the division and the scale multiply all at once; the
    assertion below is what keeps the scaling honest if the fixture ever changes.
    """
    h = mod._anchor_inputs[0]  # ScaledL2Clip also takes position_ids; only the stream is carved
    th = mod.threshold
    scale = getattr(mod, "scale", None)

    def f(v):
        x = v if scale is None else v * scale.to(v.dtype)
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return x * (th / (n + 1e-6)).clamp(max=1.0)

    x = h.float() if scale is None else h.float() * scale.float()
    norm_max = torch.linalg.vector_norm(x, dim=-1, keepdim=True).max().item()
    boost = max(1.0, 2.0 * th / norm_max)
    clipped = ((th / (torch.linalg.vector_norm(x * boost, dim=-1, keepdim=True) + 1e-6)) < 1.0).any().item()
    assert clipped, f"{type(mod).__name__}: the scaled input still never enters the clip branch"
    return _floor(f, x * boost), f"captured input scaled x{boost:.4g} into the clip branch"


def _carve_sconv(mod):
    """`(conv(window) + residual) / sqrt(k+1)` over the captured stream and the module's taps.

    The reference computes the conv on the fp32 stream with zero left padding; the fp64 arm
    widens both the stream and the taps, so the floor is the divisor's and the accumulation's
    rounding, not a dtype mismatch.
    """
    x = mod._anchor_inputs[0]  # the projection output; a `dynamic_input` slot may follow it
    taps = mod.conv1d.weight.detach()
    k = mod.conv_kernel_size

    def f(v):
        conv = torch.nn.functional.conv1d(
            v.transpose(1, 2), taps.to(v.dtype), padding=k - 1, groups=v.shape[-1])
        return (conv[:, :, : v.shape[1]].transpose(1, 2) + v) / math.sqrt(k + 1)
    return _floor(f, x), f"captured stream, {taps.shape[0]} channels, k={k}"


def _carve_message_gate(mod):
    """`(1 - tanh(forget)) * hidden + tanh(gain) * message`, projections widened per arm."""
    h, m = mod._anchor_inputs[:2]
    eps = mod.norm.eps
    down, forget = mod.down_proj.weight, mod.forget_gate_proj.weight
    gain = mod.message_gate_proj.weight

    def f(v):
        u = m.to(v.dtype)
        g = torch.cat([_weightless_rms(v, eps), _weightless_rms(u, eps)], dim=-1)
        g = g @ down.to(v.dtype).T
        return (1 - torch.tanh(g @ forget.to(v.dtype).T)) * v + torch.tanh(g @ gain.to(v.dtype).T) * u
    return _floor(f, h), "captured hidden/message, three GEMVs"


def _carve_sink_attend(tap):
    """The sink softmax and its AV sum, on the tap a real `sink_attend` call consumed.

    `attn_weights = cat([q·kᵀ·scaling + bias + mask, sink], -1)` then one softmax over the
    extended row, and the sink column's mass (`[..., :-1]`) multiplies `v`. Under `--mode
    decode`'s protocol the tap is a SINGLE query row against the cached keys.
    """
    query, key, value, mask, bias, sinks, scaling = tap
    assert query.shape[-2] == 1, f"the sink tap is not a single-query row: {tuple(query.shape)}"
    n_rep = query.shape[1] // key.shape[1]
    key_rep = M.repeat_kv(key, n_rep)
    val_rep = M.repeat_kv(value, n_rep)

    def f(q):
        logits = q @ key_rep.to(q.dtype).transpose(-1, -2) * scaling
        if bias is not None:
            logits = logits + bias.to(q.dtype)
        if mask is not None:
            logits = logits + mask.to(q.dtype)
        weights = torch.softmax(torch.cat([logits, sinks.to(q.dtype)], dim=-1), dim=-1)[..., :-1]
        return weights @ val_rep.to(q.dtype)
    return _floor(f, query), (
        f"one query row, {query.shape[1]} heads, {key.shape[-2]} keys, sink logit in the softmax")


CARVERS = {
    M.Inkling2LoopRMSNorm: _carve_rmsnorm,
    M.Inkling2LoopSoftcappedSwiGLU: _carve_swiglu,
    M.Inkling2LoopL2Clip: _carve_clip,
    M.Inkling2LoopScaledL2Clip: _carve_clip,
    M.Inkling2LoopShortConvolution: _carve_sconv,
    M.Inkling2LoopMessageGate: _carve_message_gate,
}


def _record_inputs(mod, args):
    mod._anchor_inputs = args


def _tap_attention(taps):
    """Wrap the attention interface so the sink softmax's own inputs are captured.

    The residual-only calls (`Inkling2LoopBlockAttentionResidual`, the message residual) pass
    `sinks=None`; only a layer's attention passes the per-head sink logit, so the LAST tap with
    a sink is the operator this row is about.
    """
    orig = M.ALL_ATTENTION_FUNCTIONS.get_interface

    def get_interface(impl, default):
        inner = orig(impl, default)

        def wrapped(module, query, key, value, attention_mask=None, **kwargs):
            taps.append((query, key, value, attention_mask,
                         kwargs.get("position_bias"), kwargs.get("sinks"), kwargs.get("scaling")))
            return inner(module, query, key, value, attention_mask, **kwargs)
        return wrapped
    M.ALL_ATTENTION_FUNCTIONS.get_interface = get_interface
    return lambda: setattr(M.ALL_ATTENTION_FUNCTIONS, "get_interface", orig)


def _collect_carve_targets(model):
    """Pick the first instance of each operator class in CARVERS."""
    targets = {}
    for mod in model.modules():
        cls = next((c for c in CARVERS if isinstance(mod, c) and c not in targets), None)
        if cls is not None:
            targets[cls] = mod
    return targets


def _report_floor(name, floor, note):
    # A floor of exactly 0 is the vacuous carve this table exists to avoid: it means the fp32 and
    # fp64 arms computed the same bits (operator skipped, degenerate input, or reference forced fp32).
    assert floor > 0, f"{name}: carved floor is 0 ({note}) — the carve is vacuous"
    print(f"{name}\t{floor:.6e}\t{note}")


def tolerance(salt):
    """Per-operator fp32 rounding floors, measured by CARVING each continuous operator
    fp32-vs-fp64 on the inputs a real fp32 forward fed it. The inkling reference is fp32-LOCKED
    (the residual stream and most params force `.float()`), so a whole-model fp64 run computes
    fp32 internally and shows zero diff — the floor lives in each operator's own fp32
    arithmetic, carved here. The discrete router is excluded: its top-k selection is not a
    continuous rounding error.

    The forward is the anchor's own `decode` protocol, so the streams carved are the ones the
    shipped decode path sees (a one-token step against the cache), not a fresh prefill."""
    assert_no_fast_conv()
    model, cfg, seed = build_model(salt, torch.float32)
    targets = _collect_carve_targets(model)
    taps = []

    def arm():
        hooks = [mod.register_forward_pre_hook(_record_inputs) for mod in targets.values()]
        restore_interface = _tap_attention(taps)

        def disarm():
            for h in hooks:
                h.remove()
            restore_interface()
        return disarm

    out = run_pair(model, cfg, "decode", arm)
    assert out.logits.shape[-1] == cfg.unpadded_vocab_size, "the tolerance forward did not run"
    print(f"# tolerance floors (carved fp32 vs fp64, salt {salt}, seed {seed})")
    for cls, mod in targets.items():
        floor, note = CARVERS[cls](mod)
        _report_floor(cls.__name__, floor, note)
    tap = next((t for t in reversed(taps) if t[5] is not None), None)
    assert tap is not None, "no attention call carried a sink logit — the sink_attend carve is missing"
    floor, note = _carve_sink_attend(tap)
    _report_floor("sink_attend", floor, note)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="decode", choices=["decode", "prefill"])
    ap.add_argument("--defect", default="None", choices=list(DEFECTS))
    ap.add_argument("--salt", default="inkling-1")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--tolerance", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1], args.defect)
    elif args.tolerance:
        tolerance(args.salt)
    else:
        assert args.out, "--out required for capture"
        run_capture(args.mode, args.defect, args.salt, args.out)


if __name__ == "__main__":
    main()
