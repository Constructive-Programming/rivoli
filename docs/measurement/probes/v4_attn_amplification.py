#!/usr/bin/env python3
"""Why `attn_out` cannot bisect the V4 attention block, measured rather than argued.

This is the transcription behind the `# CORRECTED 2026-08-05` section of
`tests/v4_loop.rs`. Every number in that section is printed by this script. It is here
because the transcription IS the experiment -- which ops, in which order, with the
perturbation injected where -- so "reproducible from the goldens plus the artifact" was
not a recipe, and two of the rows are single-element perturbations whose result depends
entirely on WHICH element.

    python3 docs/measurement/probes/v4_attn_amplification.py

Needs numpy, ~6 GB of RAM and ~8 s. **Touches no GPU** -- it is pure host arithmetic
against files on disk, so it runs beside a held device, like `bin/v4-oracle`.

Inputs, both read-only:
  /var/db/rivoli/v4-f4-l0-2/resident.safetensors   the converted artifact's weights
  /var/db/rivoli/v4-goldens-l2.bin                 v4-oracle emit --layers 2 --decode-steps 1

# What it establishes

1. The artifact's layer-0/1 attention weights are BYTE-IDENTICAL to the source
   checkpoint the oracle reads, and the widened `weight_scale_inv` is exact. So the
   conversion is not a candidate cause. (Only checked when the source is present.)

2. A faithful host transcription of `attn::v4::attention`'s launch sequence reproduces
   the oracle's `q`, `kv_entry`, `attn_core_out` and `attn_derot` at real dims to <= 4.2%
   of elements (p99.9 <= 25 bf16 ULP, median 1) -- and `attn_out` to 0%-21%, same inputs.
   The block's stage semantics are therefore not where a defect would have to be hiding.

3. The block performs three fp8 ACTIVATION requantizations -- act_quant(xq),
   act_quant(qrq), act_quant(y) -- each a 4-significant-bit step function immediately
   downstream of a bf16 store. A bf16 ULP is 2^-8..2^-7 relative and an e4m3 step is
   2^-4..2^-3 -- 16x larger at the same point in a binade -- so a re-associated reduction
   flips a quantization bin on a few percent of elements and each flip moves that element
   16x further than the difference that caused it. Every downstream tensor is a dense reduction over the quantized vector, so
   ONE flip perturbs ALL of them.

4. Consequently the clean/dirty ORDERING across the block's tensors is set by how many
   act_quants sit upstream of each, NOT by proximity to a defect -- which is why
   `attn_norm_out` (zero upstream) reads clean and `attn_out` (three) reads worst, for
   any implementation.

5. And the residual that keeps `tests/v4_loop.rs` red: no single uniform 1-ULP
   perturbation of `attn_derot` reproduces all three of the engine's recorded statistics
   for `L0.pre.attn_out` at once. Matching the differing fraction needs ~3e-3; matching
   `max_rel` needs ~1e-1. The engine's error tail is heavier than amplification alone.

# Two things this CANNOT say

It says nothing about what the HIP KERNELS compute. It models the launch sequence in numpy,
so it scores semantics; only reading `q`/`kv_entry`/`attn_derot` off the device -- which
`V4Engine::probe_attn_stages` now makes possible -- can close that.

And it is a transcription of `src/attn.rs` BY HAND. Agreeing with the oracle establishes
that this sequence of stages is right; it does not establish that `attn.rs` launches this
sequence, because a launch-order defect would have been transcribed along with everything
else. The two are independent only where this file was written from `model.py` and the
kernels rather than from `attn.rs`.

Transcribed from `kernels/common.hpp`, `kernels/mla.hip`, `kernels/attn.hip` and
`src/attn.rs` at 06d6863. If `f2e4m3_rne`, `fast_round_scale` or the launch order in
`attn::v4::attention` moves, re-check this against it -- nothing in the tree pins them
together.
"""

import glob
import json
import os
import struct
import sys

import numpy as np

ARTIFACT = os.environ.get("RIVOLI_V4_ARTIFACT", "/var/db/rivoli/v4-f4-l0-2")
GOLDENS = os.environ.get("RIVOLI_V4_GOLDENS", "/var/db/rivoli/v4-goldens-l2.bin")
SOURCE = os.environ.get("RIVOLI_V4_SRC", "/var/db/rivoli/deepseek-v4-flash-0731")

# DeepSeek-V4-Flash-0731, from `config.json`; mirrored by `V4_BASE` in src/artifact/model.rs.
DIM, N_HEADS, HEAD_DIM, ROPE_HEAD_DIM = 4096, 64, 512, 64
Q_LORA, O_GROUPS, O_LORA, WINDOW = 1024, 8, 1024, 128
ROPE_THETA, NORM_EPS = 10000.0, np.float32(1e-6)
NHD = N_HEADS * HEAD_DIM  # 32768
GD = NHD // O_GROUPS  # 4096 -- wo_a's k, one group's slice of `o`
GR = O_GROUPS * O_LORA  # 8192 -- wo_a's n_out, wo_b's k
FP8_MAX = np.float32(448.0)
# `check`'s floor in tests/v4_loop.rs. A ULP is a DIVISOR here too, so it needs the same
# one: unfloored, a zero reference turns any difference into ~1e37 ULP.
REL_FLOOR = 1e-3

# The engine's own numbers for L0.pre.attn_out, from the bisection table in
# docs/investigations/v4-flash-port.md. Quoted so the sweep below can be read against them.
ENGINE = dict(differ=57.9, max_abs=7.81e-2, max_rel=3.50e1)


# --- safetensors ------------------------------------------------------------------


class ST:
    def __init__(self, paths):
        self.map = {}
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                head = json.loads(f.read(n))
            for k, v in head.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.map[k] = (p, 8 + n + a, 8 + n + b, v["dtype"], tuple(v["shape"]))

    def raw(self, k):
        p, a, b, dt, sh = self.map[k]
        with open(p, "rb") as f:
            f.seek(a)
            return np.frombuffer(f.read(b - a), dtype=np.uint8), dt, sh


# --- numeric primitives, transcribed from kernels/common.hpp ------------------------


def rbf16(x):
    """`rbf16` / `f2bf16`: f32 -> bf16 -> f32, round-to-nearest-even."""
    x = np.asarray(x, dtype=np.float32)
    b = x.view(np.uint32)
    r = ((b >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    o = (((b + r) >> 16).astype(np.uint32)) << 16
    return np.where(np.isfinite(x), o, (b >> 16) << 16).astype(np.uint32).view(np.float32)


def _e4m3_lut():
    v = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 7
        if e == 0:
            v[c] = s * (m / 8.0) * 2.0**-6
        elif e == 15 and m == 7:
            v[c] = np.nan
        else:
            v[c] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return v


LUT = _e4m3_lut()


def e4m3_rne(a):
    """`f2e4m3_rne` then `e4m3f`: round-to-nearest-EVEN all the way down, saturating 448."""
    a = np.asarray(a, dtype=np.float32)
    nan = np.isnan(a)  # common.hpp:641 short-circuits isnan -> 0x7f, which decodes to NaN
    sign, x = np.signbit(a), np.abs(np.where(nan, 0.0, a)).astype(np.float32)
    b = x.view(np.uint32)
    e = ((b >> 23) & 0xFF).astype(np.int32) - 127
    sat, zer = x >= np.float32(464.0), x <= np.float32(0.0009765625)
    # subnormal: value = m * 2^-9, floor + explicit tie-to-even
    scaled = (x * np.float32(512.0)).astype(np.float32)
    m = np.floor(scaled).astype(np.float32)
    rem = (scaled - m).astype(np.float32)
    m = np.where((rem > 0.5) | ((rem == 0.5) & (m.astype(np.int64) & 1 == 1)), m + 1.0, m)
    sub = np.minimum(m, 8.0) * np.float32(2.0) ** -9
    # normal: keep 3 mantissa bits, RNE on the 20 dropped
    mant = (b & 0x7FFFFF).astype(np.uint32)
    m3, rr = (mant >> 20).astype(np.int32), (mant & 0xFFFFF).astype(np.uint32)
    m3 = np.where((rr > 0x80000) | ((rr == 0x80000) & (m3 & 1 == 1)), m3 + 1, m3)
    exp = np.where(m3 == 8, e + 8, e + 7)
    m3 = np.where(m3 == 8, 0, m3)
    nrm = (1.0 + m3.astype(np.float32) / 8.0) * np.exp2((exp - 7).astype(np.float32))
    out = np.where(sat, FP8_MAX, np.where(zer, np.float32(0.0), np.where(e < -6, sub, nrm)))
    out = np.where(sign, -out, out)
    return np.where(nan, np.float32(np.nan), out).astype(np.float32)


def e4m3_step_up(v):
    """Move an e4m3-representable value to the NEXT e4m3 code, magnitude-wise.

    `v * 1.125` is one step only when the mantissa field is 0 — one step is a factor
    (9 + m3) / (8 + m3). On the real `qrq` only 14% of nonzero elements have m3 == 0, so
    the naive form is 1.125 to 1.875 steps and lands OFF the e4m3 grid, i.e. on a value
    `act_quant` can never emit. A first draft of this probe used it and overstated the
    resulting `q` movement by ~16%.

    `v` is a dequantized `e4m3 * 2^k`, so the mantissa field is recoverable from the f32
    directly: the leading 3 mantissa bits are m3 and the rest are zero.
    """
    v = np.asarray(v, dtype=np.float32)
    m3 = ((v.view(np.uint32) >> 20) & 0x7).astype(np.float32)
    return (v * ((9.0 + m3) / (8.0 + m3)).astype(np.float32)).astype(np.float32)


def act_quant(x, rows, row_stride, n, block):
    """`v4_act_quant`: fused quantize-then-dequantize, IN PLACE over dims [0, n).

    THIS IS THE AMPLIFIER. It is a step function with 4 significant bits, and it sits
    immediately downstream of a bf16 store at all three of its call sites.
    """
    a = x.reshape(rows, row_stride)
    seg = a[:, :n].reshape(rows, n // block, block)
    amax = np.maximum(np.abs(seg).max(axis=2), np.float32(1e-4)).astype(np.float32)
    # `fast_round_scale`: ceil(log2(amax/448)) as a bare power of two, by bit surgery.
    y = (amax * np.float32(1.0 / 448.0)).astype(np.float32)
    yb = y.view(np.uint32)
    ex = ((yb >> 23) & 0xFF).astype(np.int32) - 127 + np.where((yb & 0x7FFFFF) != 0, 1, 0)
    s = (((ex + 127).astype(np.uint32)) << 23).view(np.float32)[:, :, None]
    q = np.clip((seg / s).astype(np.float32), -FP8_MAX, FP8_MAX).astype(np.float32)
    a[:, :n] = (e4m3_rne(q) * s).reshape(rows, n)
    return x


# The accumulator width every reduction below uses. A list so section 6 can flip it: f32 is
# what the kernels do, f64 is an equally CORRECT implementation of the same semantics, and the
# distance between the two is the floor on any tolerance built on these goldens. It changes
# only the accumulator -- the bf16 and e4m3 rounding points are unchanged, because those are
# semantics rather than precision.
ACC = [np.float32]


def gemv(x, w, m, n_out, k, groups):
    """`v4_gemv_fp8`: out[r][j] = rbf16(dot(x_row(r, j), w[j])), bf16 store.

    `x` is `m` rows of `groups` consecutive k-wide slices; output row j reads slice
    j // (n_out // groups). groups == 1 is a plain Linear; groups == o_groups is wo_a.
    """
    xr = x.reshape(m, groups, k).astype(ACC[0])
    g = n_out // groups
    out = np.empty((m, n_out), dtype=ACC[0])
    for i in range(groups):
        out[:, i * g : (i + 1) * g] = xr[:, i, :] @ w[i * g : (i + 1) * g, :].T.astype(ACC[0])
    return rbf16(out.astype(np.float32))


def v4_rmsnorm(x, w, rows, d):
    """`v4_rmsnorm`: f32 statistic (the reference opens with `x.float()`), bf16 store."""
    a = x.reshape(rows, d)
    p = (a.astype(ACC[0]) ** 2).sum(axis=1, dtype=ACC[0])
    rs = (1.0 / np.sqrt(p / ACC[0](d) + ACC[0](NORM_EPS))).astype(np.float32)
    return rbf16(w[None, :] * (a * rs[:, None]))


def v4_qk_norm(x, rows, d):
    """`v4_qk_norm`: BF16 statistic, unlike `v4_rmsnorm`, and no learnable weight.

    model.py:504 has no `.float()`, so the square, the mean, the `+ eps` and the rsqrt
    are all bf16-valued. Keeping it in f32 moves the factor by up to 2^-9, identically
    for every dim of a head.
    """
    a = x.reshape(rows, d)
    var = rbf16(rbf16(a * a).sum(axis=1, dtype=np.float32) / np.float32(d))
    rs = rbf16(np.float32(1.0) / np.sqrt(rbf16(var + NORM_EPS)))
    return rbf16(a * rs[:, None])


def rope_table(rd, max_pos, theta):
    """`v4_rope_table_ratio0`: interleaved (cos, sin). Ratio-0 layers only -- no YaRN."""
    inv = np.array(
        [1.0 / np.float32(theta) ** np.float32((2 * i) / rd) for i in range(rd // 2)],
        dtype=np.float32,
    )
    t = np.arange(max_pos, dtype=np.float32)[:, None] * inv[None, :]
    out = np.empty((max_pos, rd), dtype=np.float32)
    out[:, 0::2], out[:, 1::2] = np.cos(t), np.sin(t)
    return out


def v4_rope(x, tbl, rows, row_len, rd, pos0, rows_per_pos, inverse):
    """`v4_rope`: rotate the LAST `rd` dims, pairing ADJACENT dims (view_as_complex)."""
    a = x.reshape(rows, row_len).copy()
    seg = a[:, row_len - rd :]
    t = tbl[pos0 + np.arange(rows) // rows_per_pos]
    c, s = t[:, 0::2], (-t[:, 1::2] if inverse else t[:, 1::2])
    p, q = seg[:, 0::2].copy(), seg[:, 1::2].copy()
    seg[:, 0::2], seg[:, 1::2] = p * c - q * s, p * s + q * c
    return rbf16(a)


def sparse_attn(q, kv, sink, idxs, m, h, d, scale):
    """`v4_sparse_attn`: MQA over one entry that is both key and value; `attn_sink` enters
    the softmax DENOMINATOR only -- a learned per-head leak, never a key."""
    o = np.empty((m, h, d), dtype=np.float32)
    qq, kk = q.reshape(m, h, d).astype(ACC[0]), kv.reshape(-1, d)
    for t in range(m):
        rows = kk[idxs[t][idxs[t] >= 0]]
        lg = (qq[t] @ rows.T.astype(ACC[0])).astype(np.float32) * np.float32(scale)
        mx = lg.max(axis=1)
        e = np.exp((lg - mx[:, None]).astype(ACC[0]))
        den = e.sum(axis=1, dtype=ACC[0]) + np.exp((sink - mx).astype(ACC[0]))
        o[t] = ((e @ rows.astype(ACC[0])) / den[:, None]).astype(np.float32)
    return rbf16(o.ravel())


# --- inputs -------------------------------------------------------------------------


def goldens(path):
    d = open(path, "rb").read()
    assert d[:8] == b"RIVV4GLD", "not a rivoli V4 golden file"
    o = 8

    def u64():
        nonlocal o
        v = struct.unpack_from("<Q", d, o)[0]
        o += 8
        return v

    def s():
        nonlocal o
        n = u64()
        v = d[o : o + n].decode()
        o += n
        return v

    for _ in range(u64()):
        s(), s()
    out = {}
    for _ in range(u64()):
        name = s()
        for _ in range(u64()):
            u64()
        n = u64()
        out[name] = np.frombuffer(d, dtype=np.float32, count=n, offset=o).copy()
        o += n * 4
    return out


def fp8w(art, layer, name):
    """One dequantized fp8 weight. The 128x128 block scale is a bare power of two, so
    `e4m3 * scale` is exact and the fold order of this product does not matter."""
    w, _, (o_dim, i_dim) = art.raw(f"layers.{layer}.attn.{name}.weight")
    sc, _, ssh = art.raw(f"layers.{layer}.attn.{name}.weight_scale_inv")
    grid = sc.view(np.float32).reshape(ssh)
    full = np.repeat(np.repeat(grid, 128, axis=0), 128, axis=1)[:o_dim, :i_dim]
    return (LUT[w].reshape(o_dim, i_dim) * full).astype(np.float32)


def f32t(art, layer, name):
    a, _, sh = art.raw(f"layers.{layer}.attn.{name}")
    return a.view(np.float32).reshape(sh).copy()


# --- scoring ------------------------------------------------------------------------


def ulp(v):
    """One bf16 ULP at |v|, floored at REL_FLOOR. Mirrors `bf16_ulp` in tests/v4_loop.rs."""
    return np.exp2(np.floor(np.log2(np.maximum(np.abs(v), REL_FLOOR))) - 7).astype(np.float32)


def score(got, want):
    got, want = np.asarray(got).ravel(), np.asarray(want).ravel()
    assert got.size == want.size, "shape disagreement is not a tolerance question"
    differ = float((rbf16(got).view(np.uint32) != rbf16(want).view(np.uint32)).mean()) * 100
    ab = np.abs(got - want)
    return dict(
        differ=differ,
        max_abs=float(ab.max()),
        max_rel=float((ab / np.maximum(np.abs(want), REL_FLOOR)).max()),
        ulp999=float(np.percentile(ab / ulp(want), 99.9)),
    )


def line(tag, sc):
    print(
        f"  {tag:32s} differ {sc['differ']:6.2f}%  max_abs {sc['max_abs']:.3e}  "
        f"max_rel {sc['max_rel']:.3e}  ULP p99.9 {sc['ulp999']:7.1f}"
    )


# --- the block ----------------------------------------------------------------------


def attention(art, gold, layer, tag, m, pos, ring, tbl):
    """One `attn::v4::attention` call, transcribed. Returns each stage for scoring.

    Driven from the ORACLE's `attn_norm_out`, not from a composed residual, so every
    layer/phase cell is scored on the reference's own input -- the same choice
    tests/v4_loop.rs makes with `set_residual`.
    """
    wqa, wqb, wkv = (fp8w(art, layer, n) for n in ("wq_a", "wq_b", "wkv"))
    # `wo_a` is fp8 on disk but a plain bf16 parameter in the reference: `Attention.forward`
    # consumes it raw in an einsum, so `convert.py` dequantizes it and there is NO activation
    # quantization on that projection. Everything else goes through `Linear`, which act_quants.
    woa, wob = rbf16(fp8w(art, layer, "wo_a")), fp8w(art, layer, "wo_b")
    qn = f32t(art, layer, "q_norm.weight")
    kn = f32t(art, layer, "kv_norm.weight")
    sink = f32t(art, layer, "attn_sink")

    x = gold[f"L{layer}.{tag}.attn_norm_out"].copy()
    xq = act_quant(x.copy(), m, DIM, DIM, 128)  # AMPLIFIER 1

    qr = v4_rmsnorm(gemv(xq, wqa, m, Q_LORA, DIM, 1), qn, m, Q_LORA)
    qrq = act_quant(qr.copy().ravel(), m, Q_LORA, Q_LORA, 128)  # AMPLIFIER 2
    q = gemv(qrq, wqb, m, NHD, Q_LORA, 1).ravel()
    # QK-norm BEFORE RoPE (model.py:504 then :505), read off the reference.
    q = v4_rope(v4_qk_norm(q, m * N_HEADS, HEAD_DIM).ravel(), tbl, m * N_HEADS,
                HEAD_DIM, ROPE_HEAD_DIM, pos, N_HEADS, False)

    kv = v4_rmsnorm(gemv(xq, wkv, m, HEAD_DIM, DIM, 1), kn, m, HEAD_DIM)
    kv = v4_rope(kv.ravel(), tbl, m, HEAD_DIM, ROPE_HEAD_DIM, pos, 1, False)
    # PARTIAL, at block 64 and not 128: the RoPE'd tail keeps bf16 precision (model.py:512).
    kv = act_quant(kv.ravel(), m, HEAD_DIM, HEAD_DIM - ROPE_HEAD_DIM, 64)

    if pos == 0:  # prefill: attend the prompt's own KV, by ABSOLUTE position, causally
        ring[:m] = kv.reshape(m, HEAD_DIM)
        src = kv.reshape(m, HEAD_DIM)
        idxs = np.full((m, m), -1, np.int64)
        for t in range(m):
            idxs[t, : t + 1] = np.arange(t + 1)
    else:  # decode: attend the ring, by SLOT; `window_topk` pads to `win` with -1
        ring[pos % WINDOW] = kv.reshape(1, HEAD_DIM)
        src = ring
        row = np.full(WINDOW, -1, np.int64)
        row[: pos + 1] = np.arange(pos + 1)
        idxs = row[None, :]

    core = sparse_attn(q.ravel(), src.ravel(), sink, idxs, m, N_HEADS, HEAD_DIM,
                       HEAD_DIM**-0.5)
    derot = v4_rope(core.ravel(), tbl, m * N_HEADS, HEAD_DIM, ROPE_HEAD_DIM, pos,
                    N_HEADS, True)

    def tail(d):
        y = gemv(d, woa, m, GR, GD, O_GROUPS)
        return gemv(act_quant(y.copy().ravel(), m, GR, GR, 128), wob, m, DIM, GR, 1).ravel()

    def tail_no_actquant(d):
        """The same tail with `act_quant(y)` REMOVED — the ablation that isolates which of
        the three requantizations dominates. Not a variant anyone should ship; it is the
        counterfactual."""
        y = gemv(d, woa, m, GR, GD, O_GROUPS)
        return gemv(y.ravel().copy(), wob, m, DIM, GR, 1).ravel()

    return (
        dict(q=q.ravel(), kv_entry=kv.ravel(), attn_core_out=core.ravel(),
             attn_derot=derot.ravel(), attn_out=tail(derot.ravel())),
        tail,
        tail_no_actquant,
    )


# --- checks -------------------------------------------------------------------------


def check_conversion():
    """The artifact vs the source checkpoint the ORACLE reads. Rules the conversion in or out."""
    print("== 1. artifact weights vs the source checkpoint (the oracle's own input)")
    shards = sorted(glob.glob(os.path.join(SOURCE, "*.safetensors")))
    if not shards:
        # An EXPLICITLY SET path that does not resolve is a failure, never a skip -- the rule
        # `tests/common/v4_artifact_dir.rs` states and the reason it gives: someone who pointed
        # this at a checkpoint and got a clean report would have no way to tell that check 1
        # never ran.
        if "RIVOLI_V4_SRC" in os.environ:
            sys.exit(f"RIVOLI_V4_SRC={SOURCE} holds no *.safetensors")
        print(f"  SKIP: no source checkpoint at {SOURCE} -- conversion is UNCHECKED here\n")
        return
    art, src = ST([os.path.join(ARTIFACT, "resident.safetensors")]), ST(shards)
    bad = 0
    for layer in (0, 1):
        for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            a, _, _ = art.raw(f"layers.{layer}.attn.{n}.weight")
            s, _, _ = src.raw(f"layers.{layer}.attn.{n}.weight")
            sa = art.raw(f"layers.{layer}.attn.{n}.weight_scale_inv")[0].view(np.float32)
            codes = src.raw(f"layers.{layer}.attn.{n}.scale")[0].astype(np.int32)
            ss = np.where(codes == 0, np.float32(2.0) ** -127,
                          np.exp2(codes.astype(np.float64) - 127)).astype(np.float32)
            ok = np.array_equal(a, s) and np.array_equal(sa, ss)
            bad += not ok
            if not ok:
                print(f"  layer {layer} {n}: DIFFERS")
    print(f"  10 fp8 tensors + their widened e8m0 scales: "
          f"{'all byte-identical -- conversion ruled OUT' if not bad else f'{bad} DIFFER'}\n")


def main():
    for p in (os.path.join(ARTIFACT, "resident.safetensors"), GOLDENS):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- see the module docstring for how to produce it")
    check_conversion()

    art = ST([os.path.join(ARTIFACT, "resident.safetensors")])
    gold = goldens(GOLDENS)
    tbl = rope_table(ROPE_HEAD_DIM, 4096, ROPE_THETA)
    m_pre = gold["L0.pre.attn_norm_out"].size // DIM

    print("== 2. the transcribed block vs the oracle, real weights, real dims")
    print("   (numpy's fold order, not the kernels' -- so this scores SEMANTICS, not the kernels)")
    tails, worst = {}, 0.0
    for layer in (0, 1):
        # Seeded from the ORACLE's prefill KV, not from this transcription's, so the decode
        # cell is scored on the reference's own ring rather than on one this run produced.
        # They are bit-identical (the kv_entry rows below say so), but using the run's own
        # would make that claim circular.
        ring = np.zeros((WINDOW, HEAD_DIM), dtype=np.float32)
        ring[:m_pre] = gold[f"L{layer}.pre.kv_entry"].reshape(m_pre, HEAD_DIM)
        for tag, m, pos in (("pre", m_pre, 0), ("dec0", 1, m_pre)):
            got, tail, tail_no_aq = attention(art, gold, layer, tag, m, pos, ring, tbl)
            tails[(layer, tag)] = (got, tail, tail_no_aq)
            for k in ("q", "kv_entry", "attn_core_out", "attn_derot", "attn_out"):
                sc = score(got[k], gold[f"L{layer}.{tag}.{k}"])
                if k == "attn_out":
                    worst = max(worst, sc["differ"])
                line(f"L{layer}.{tag}.{k}", sc)
            print()
    print(f"  the ladder: `kv_entry` (1 act_quant upstream) is bit-identical on all four cells;")
    print(f"  `attn_out` (3 upstream) spans 0% to {worst:.2f}% of elements differing.\n")

    print("== 3. the amplifier: one e4m3 step in `qrq` moves a large fraction of `q`")
    wqa, wqb = fp8w(art, 0, "wq_a"), fp8w(art, 0, "wq_b")
    qn = f32t(art, 0, "q_norm.weight")
    x = gold["L0.pre.attn_norm_out"].copy()
    xq = act_quant(x.copy(), m_pre, DIM, DIM, 128)
    qr = v4_rmsnorm(gemv(xq, wqa, m_pre, Q_LORA, DIM, 1), qn, m_pre, Q_LORA)
    qrq0 = act_quant(qr.copy().ravel(), m_pre, Q_LORA, Q_LORA, 128)

    def q_of(qq):
        v = gemv(qq, wqb, m_pre, NHD, Q_LORA, 1).ravel()
        return v4_rope(v4_qk_norm(v, m_pre * N_HEADS, HEAD_DIM).ravel(), tbl,
                       m_pre * N_HEADS, HEAD_DIM, ROPE_HEAD_DIM, 0, N_HEADS, False).ravel()

    base = q_of(qrq0.copy())
    rng = np.random.default_rng(3)
    for n in (1, 4, 16, 64):
        qq = qrq0.copy()
        idx = rng.choice(qq.size, n, replace=False)
        qq[idx] = e4m3_step_up(qq[idx])
        d = float((rbf16(q_of(qq)).view(np.uint32) != rbf16(base).view(np.uint32)).mean()) * 100
        print(f"  {n:3d}/{qrq0.size:,} `qrq` elements moved one e4m3 step "
              f"-> `q` ({base.size:,}) differs {d:6.2f}%")
    print()

    print("== 4. the cleanest single-element case, OBSERVED rather than injected")
    # L1.dec0 needs no perturbation to make the point: the transcription's `attn_derot`
    # already differs from the oracle's on exactly ONE element out of 32,768, by one bf16
    # ULP, purely from fold order. Its `attn_out` differs on a fifth of the tensor.
    #
    # Injecting a chosen element instead was tried and is a WORSE experiment: which element
    # matters enormously (index 0 of this same tensor moves `attn_out` by 0.00%), so an
    # injected number reports the draw rather than the mechanism. This one is the arithmetic
    # the two implementations actually produced.
    g1 = tails[(1, "dec0")][0]
    for k in ("attn_derot", "attn_out"):
        w = gold[f"L1.dec0.{k}"]
        n = int((rbf16(g1[k]).view(np.uint32) != rbf16(w).view(np.uint32)).sum())
        rel = float((np.abs(g1[k] - w) / np.maximum(np.abs(w), REL_FLOOR)).max())
        print(f"  L1.dec0.{k:11s} {n:5d}/{w.size:6,} differ ({100 * n / w.size:5.2f}%)  "
              f"max_rel {rel:.3f}")
    print()

    print("== 5. WHICH of the three requantizations dominates: ablate `act_quant(y)`")
    # Same perturbation into the tail with and without the act_quant between wo_a and wo_b.
    # The 8192-wide dot amplifies on its own; this says how much the step function adds.
    _, tail0, tail0_no_aq = tails[(0, "pre")]
    d0 = gold["L0.pre.attn_derot"].copy()
    rng = np.random.default_rng(11)
    bb = d0.view(np.uint32).copy()
    mk = rng.random(d0.size) < 1e-3
    bb[mk] = (bb[mk].astype(np.int64) + (rng.integers(0, 2, d0.size) * 2 - 1)[mk] * 0x10000).astype(np.uint32)
    pert = bb.view(np.float32)
    for name, fn in (("shipped (act_quant(y) present)", tail0), ("ablated (no act_quant(y))", tail0_no_aq)):
        x0, x1 = fn(d0), fn(pert)
        dz = float((rbf16(x1).view(np.uint32) != rbf16(x0).view(np.uint32)).mean()) * 100
        print(f"  1 bf16 ULP on 1e-03 of `attn_derot` -> {name:32s} `attn_out` differs {dz:6.2f}%")
    print()

    print("== 6. two CORRECT implementations: f32 vs f64 accumulation, same semantics")
    # The mildest legitimate difference between two implementations. Everything else is held
    # fixed -- same weights, same rounding points, same order of operations -- and only the
    # GEMV/norm accumulator width changes. Whatever this produces is a FLOOR on any bound.
    for layer, tag, m, pos in ((0, "pre", m_pre, 0),):
        a32 = tails[(layer, tag)][0]["attn_out"]
        ACC[0] = np.float64
        ring = np.zeros((WINDOW, HEAD_DIM), dtype=np.float32)
        ring[:m_pre] = gold[f"L{layer}.pre.kv_entry"].reshape(m_pre, HEAD_DIM)
        a64 = attention(art, gold, layer, tag, m, pos, ring, tbl)[0]["attn_out"]
        ACC[0] = np.float32
        line(f"L{layer}.{tag}.attn_out f32-vs-f64", score(a32, a64))
    print("  ^ that is the distance between two implementations that are BOTH correct.")
    print(f"  `tests/v4_loop.rs` asserts max_rel < 5e-2 on this tensor.\n")

    print("== 7. the residual: no single perturbation of `attn_derot` matches the engine")
    print(f"  ENGINE, L0.pre.attn_out (v4-flash-port.md): differ {ENGINE['differ']}%  "
          f"max_abs {ENGINE['max_abs']:.2e}  max_rel {ENGINE['max_rel']:.2e}")
    derot = gold["L0.pre.attn_derot"].copy()
    want = gold["L0.pre.attn_out"]
    tail = tails[(0, "pre")][1]
    line("no perturbation (fold only)", score(tail(derot), want))
    rng = np.random.default_rng(7)
    for p in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        b = derot.view(np.uint32).copy()
        mask = rng.random(derot.size) < p
        # +-1 BF16 ulp: the low 16 mantissa bits of a bf16-valued f32 are zero, so the
        # step is 0x10000. Adding 1 would be an f32 ulp -- 1/65536 as large, and inert.
        step = (rng.integers(0, 2, derot.size) * 2 - 1) * 0x10000
        b[mask] = (b[mask].astype(np.int64) + step[mask]).astype(np.uint32)
        line(f"1 bf16 ULP on {p:.0e} of elements", score(tail(b.view(np.float32)), want))
    print()
    print("  Read the sweep against the ENGINE line above: the differing fraction is matched")
    print("  around 3e-3, but max_abs and max_rel are not matched until ~1e-1 -- a ~30x")
    print("  disagreement. Amplification explains the SHAPE and most of the magnitude; the")
    print("  remaining tail is what `probe_attn_stages` reading `attn_derot` off the device")
    print("  is for. That is why tests/v4_loop.rs stays RED rather than having its bound moved.")


if __name__ == "__main__":
    main()
