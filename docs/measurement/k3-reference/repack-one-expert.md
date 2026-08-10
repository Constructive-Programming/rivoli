---
status: data
scope: k3
verdict: The `.f4` repack is bit-exact on real Kimi-K3 tensors — all six spans of a genuine expert survive unchanged, in gate/up/down order, verified independently of rivoli's own code.
---

# The `.f4` repack, on real Kimi-K3 bytes

**Measured 2026-08-10.** G1a's first bullet asks for the repack to be *bit-exact both directions on
real tensors*. K3 is 1.42 TiB and does not fit here, so the tensors were fetched by HTTP Range
from the shipped shard — **one real expert, 17,547,264 bytes** — and converted. This is the record
of what was run and what came back, so it can be re-run without re-deriving the offsets.

## What was fetched

`moonshotai/Kimi-K3`, revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`,
`model-00002-of-000096.safetensors`. Header is 818,696 bytes, so tensor data begins at
`base = 8 + 818,696 = 818,704`.

**Ranges are given as the header's own `data_offsets`, not as absolutes.** An earlier version of
this table listed absolute byte ranges and the first row was mistranscribed — both endpoints
344,000 low — which a re-run would have turned into 5.5 MB of the wrong bytes and a hash mismatch
blamed on the revision. Caught by review 2026-08-11. The measurement itself was never affected:
the fetch used ranges generated from the header, not this table. Absolutes are `base + offset`, and
the six spans are exactly contiguous, so each row's start is the previous row's end.

Layer 1, expert 0 — `language_model.model.layers.1.block_sparse_moe.experts.0.*`:

| tensor | dtype | shape | bytes | `data_offsets` (relative) | sha256, first 8 bytes |
|---|---|---|---|---|---|
| `w1.weight_packed` | U8 | 3072×1792 | 5,505,024 | 1,267,744,256 – 1,273,249,279 | `7ce56d721eea1dbc` |
| `w1.weight_scale` | U8 | 3072×112 | 344,064 | 1,273,249,280 – 1,273,593,343 | `2711ff31127e735b` |
| `w2.weight_packed` | U8 | 3584×1536 | 5,505,024 | 1,273,593,344 – 1,279,098,367 | `f4dbcdbb0f7d2d02` |
| `w2.weight_scale` | U8 | 3584×96 | 344,064 | 1,279,098,368 – 1,279,442,431 | `525101a1b9d83ade` |
| `w3.weight_packed` | U8 | 3072×1792 | 5,505,024 | 1,279,442,432 – 1,284,947,455 | `405f4a03860b03a3` |
| `w3.weight_scale` | U8 | 3072×112 | 344,064 | 1,284,947,456 – 1,285,291,519 | `7fdf3a03ac6cd1a2` |

So `w1.weight_packed` is HTTP range `1268562960-1274067983`, and the rest follow contiguously.

Assembled into a one-tensor-per-name `model.safetensors` plus a matching
`model.safetensors.index.json`, alongside a **shadow config**: the real `config.json` with
`num_experts` 1, `num_experts_per_token` 1, `num_hidden_layers` 2 and
`linear_attn_config.{full_attn_layers: [], kda_layers: [1,2]}` — the smallest edit that is still a
legal partition and still passes `K3TextConfig::validate`. One synthetic BF16
`routed_expert_norm` (zeros, shipped shape `[3584]`) so the resident pass has an input; the
resident path is a verbatim copy and is not what this measures.

## What came back

```
convert_k3: hidden=7168 latent=3584 moe_inter=3072 experts=1 layers 1..2 (of 2, dense prefix 1)
convert_k3: wrote .../L01.f4 (17551360 bytes)
convert_k3: verified L01.f4 — 1 experts, 0 bytes differ
convert_k3: wrote .../resident.safetensors — 1 tensors (6 routed, 0 vision skipped)
```

`L01.f4` sha256, first 16 bytes: `56d288cfb03aba135e39f61c3abd61a5`. 17,551,360 = `VQ_ALIGN` 4096 + 17,547,264,
so the expert's stride needed no padding — `f4_expert_bytes` is already 4096-aligned at these
widths (17,547,264 / 4096 = 4,284 exactly).

**Then verified again in Python, without using rivoli's code**, because `--verify` and the writer
share `F4Expert::spans`: a shared-layout check can only prove the arithmetic agrees with itself.
Slot offsets were recomputed from the widths alone and each span compared to the fetched file:

| slot offset | span | bytes | identical to |
|---|---|---|---|
| 0 | gate packed | 5,505,024 | `w1.weight_packed` |
| 5,505,024 | gate scale | 344,064 | `w1.weight_scale` |
| 5,849,088 | **up** packed | 5,505,024 | **`w3`**`.weight_packed` |
| 11,354,112 | up scale | 344,064 | `w3.weight_scale` |
| 11,698,176 | **down** packed | 5,505,024 | **`w2`**`.weight_packed` |
| 17,203,200 | down scale | 344,064 | `w2.weight_scale` |

All six bit-identical. The bolded rows are the part worth having: **the up slot holds `w3` and the
down slot holds `w2`**, which is the gate/up/down order `k3-architecture.md` §6 fixes from the
reference's forward pass. A `w1`/`w3` swap is the one repack error that is internally consistent
and byte-clean — `V4_PROJ`'s doc says only a numerical oracle can see it — and while that remains
true for a *swap of the two same-shaped tensors*, this at least pins that the down projection is
not in the up slot. Header parsed blind: magic `FP4\0`, then layer 1, 1 expert, 3584, 3072, stride
17,547,264. Padding to `VQ_ALIGN` is all zero.

## The e8m0 `0xff` question (S1a item 2), with data

The reference maps `sb == 255` to zero; `quant::e8m0` returns a bail. **4,128,768 real scale bytes**
were sampled — every scale tensor of experts 0–3 in layer 1 — and:

| | |
|---|---|
| distinct codes | **11**, all in `0x70..=0x7a` (2⁻¹⁵ … 2⁻⁵) |
| `0xff` (the e8m0 NaN) | **0** |
| `0x00` (2⁻¹²⁷) | **0** |

Distribution is sharply peaked: `0x79` is 82% and `0x78` 17%, the other nine codes together 1%.
That is the same shape V4's set showed (9 codes in `0x76..=0x7e`, zero of both ends), so
**rivoli's refusal is green on this sample and the reference's 255→zero path is defensive rather
than exercised.**

**This is a sample, and a small one: 4 of 82,432 experts, ~0.005% of the ~85 GB of scale bytes in
the checkpoint.** It does not settle item 2 by itself. What settles it is that the repack is the
only path that reads *every* scale byte — at decode they DMA from NVMe into the pool slot and the
host never sees them — so `F4Expert::spans`'s existing check will either pass over the whole set at
conversion time or name the exact tensor, row and group that fails. That is the same argument V4's
measurement rests on, and it is why the check lives at repack.

## Re-running it

Everything above is reproducible from the ranges in the first table plus
`docs/measurement/k3-reference/tensor-families.tsv`. Nothing is vendored: 17.5 MB of weights is
too much for the repo, and the fixture is worth less than the recipe — the ranges are stable
because the revision is pinned.
