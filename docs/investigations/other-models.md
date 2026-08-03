---
status: live
verdict: Kimi K3 is a BITRATE problem, not a capacity one — 2.72T expert params is 1.03 TiB at int3-vq's 3.25 bits/weight against ~940 GiB reclaimable, and ≤2.5 bits fits; the real wall is a linear-attention kernel family for 69 of its 93 layers. DeepSeek-V4-Flash-0731 fits at ~120 GiB and its sqrt(softplus) router SHIPPED 2026-08-03, but it is not MLA, not residual-additive, and llama.cpp measured it compute-bound — so it barely exercises expert streaming.
---

# Can rivoli run a model other than GLM-5.2?

**Asked 2026-08-03**, against two named targets. Answered from the shipped `config.json` of
each model, not from the tech reports — several load-bearing fields are config-only, and two
claims in the reports were wrong in the direction that would have made this look easier.

## STATE — the answer in fifteen lines

The engine is *more* portable than it looks and *less* portable than we hoped, in different
places than expected.

- **The kernels are not the problem.** No model dimension is compiled into any HIP kernel.
  `hidden`/`inter`/`kvl`/`H` are runtime launch arguments. `build.rs` injects only `WAVE` and
  `ROWS_PER_BLOCK`, both hardware. The compile-time `#define`s (`VQ_DIM`, `VQ_K`, `I4_GROUP`,
  `MOE_ACC_SHIFT`) describe the **quantization format**, which is ours, not the model's.
- **The MoE half is genuinely reusable**: expert streaming, io_uring fetch, the byte arena,
  the residency cache, int3-vq/int4 codecs, fixed-point accumulation. That is the expensive,
  hard-won half of this engine and it survives a model change intact.
- **The attention half is not.** rivoli is MLA-with-q-LoRA and nothing else, with no
  GQA/MHA/linear path, and applies interleaved RoPE unconditionally with no scaling.
- **Kimi K3's capacity problem is a bitrate problem** (§2). It does not fit at int3-vq's
  3.25 bits/weight; it fits at ≤2.5, and the VQ constants are ours to change. The wall that
  is *not* negotiable is the 69 KDA linear-attention layers.
- **DeepSeek-V4-Flash-0731 fits, and would barely stream at all** (§3) — which is worth
  saying plainly, because streaming is the point of this engine. llama.cpp measured it
  compute-bound.
- **The router already ships** (§4). `sqrt(softplus(·))` behind a `Scoring` enum, verified
  against the reference implementation, with INV-1's frozen oracle still pinning sigmoid.

## 1. What was already GLM-specific, and what got fixed

Three defects found here were GLM-specific bugs in their own right and are fixed on this
branch regardless of whether either port proceeds:

| what | where | why it was wrong |
|---|---|---|
| `MAX_FUSED_INTER = 16384` ceiling | `artifact/model.rs` | Guarded LDS pressure in `moe_fused.hip`, a kernel that **no longer exists**. `swiglu` (`linalg.hip`) is elementwise with zero dynamic LDS and `moe.hip` stages nothing *on purpose* — LDS capped occupancy and measured slower. The guard rejected any `intermediate_size > 16384` for a constraint that had been deleted. |
| `rope_parameters` required nested | `artifact/model.rs` | GLM-5.2 nests theta; the entire DeepSeek/Llama lineage puts `rope_theta` at top level. This was the *first* thing to fail on a foreign config — a serde error before any dimension was inspected. Now accepts either, and bails loudly if neither is present. |
| VQ divisibility only `debug_assert`ed | `artifact/quant.rs` → checked in `model.rs` | `vq_row_bytes`/`vq_groups` divide by `VQ_DIM`/`VQ_GROUP` with a `debug_assert`, so in a **release** build a width not a multiple of 64 silently truncates every expert row with no diagnostic. Hoisted to one loud check at load, where `hidden` and `moe_inter` are both known. |

Not fixed, because the target does not need it: **group-limited routing** (`n_group` /
`topk_group`) is unimplemented and appears nowhere in `src/` or `kernels/`. GLM-5.2 ships
`1`/`1`, so it is a no-op today. DeepSeek-V3 ships `8`/`4` and *would* have routed wrongly and
silently — but V4-Flash's config carries neither field, and the V4 paper says the constraint
was removed outright. Left alone deliberately; see "what we did not build" below.

## 2. Kimi K3 — capacity is solvable, the attention is not (yet)

`moonshotai/Kimi-K3`, `model_type: kimi_k3`, 2.78T total / 104.2B active.

> **CORRECTED 2026-08-03, same day.** This section first said "REFUSED, does not fit".
> That conflated *disk capacity* with *bitrate*: the artifact does not fit **at int3-vq's
> 3.25 bits/weight**, which is not a floor. Unsloth ships a dynamic low-bit K3 build at
> **594 GB**, which fits here with room to spare. The error came from treating the one
> quantization this engine happens to implement as the size of the model. Capacity is a
> format decision; the KDA layers below are the actual wall.

**Capacity, at the bitrate we have today.** 93 layers, 896 routed experts,
`moe_intermediate_size` 3072, hidden 7168. Per expert 3 × 3584 × 3072 = 33.0M; × 896 experts
× 92 MoE layers = **2.72T params in routed experts** (this reproduces the 2.78T headline, so
the layout is right). int3-vq stores 3 bits/weight plus one bf16 scale per 64 → 3.25 bits =
0.406 B/weight, giving a **~1.03 TiB expert artifact**.

The disk is btrfs RAID0 across both NVMes: **1.69 TiB total, 265 GiB free**, and the GLM-5.2
artifact is 675 GiB of it. Deleting GLM-5.2 outright frees 940 GiB = 0.92 TiB. **Still ~115
GiB short**, and that is before `resident.safetensors`. There is no third drive; the
apparently-unformatted `nvme0n1p2` is already a member of the pool. NVMe is not substitutable
here — O_DIRECT streaming is the design, and `/swarm/storage` is NFS.

**The bitrate that would fit.** 2.72T expert params against ~940 GiB of reclaimable disk
needs **≤2.9 bits/weight**; against the 630 GiB freed by dropping only the `.i4` twins,
**≤1.9 bits**. int3-vq is 3.25 (12-bit index over `VQ_DIM=4` weights, plus one bf16 scale
per 64). The knobs already exist as paired constants in `quant.rs` and `common.hpp`:

| VQ_DIM | VQ_K | bits/weight | expert artifact | codebook |
|---|---|---|---|---|
| 4 | 4096 | 3.25 (today) | 1.03 TiB | 32 KB — fits L1 |
| 4 | 256 | 2.25 | 713 GiB | 2 KB |
| 8 | 1024 | 1.50 | 475 GiB | 16 KB |
| 8 | 4096 | 1.75 | 554 GiB | **64 KB — exceeds the 32 KB L1** |

Not a constant flip: `dot_vq_wave_r` (`common.hpp`) reads a `float4` and two `__half2` per
subvector and unpacks 12 bits with a fixed two-byte read, so `VQ_DIM≠4` or
`VQ_INDEX_BITS≠12` needs a new inner loop and a matching packer in `quant.rs`. The L1
column is load-bearing — the comment there records that moving the codebook from f32 (64 KB,
L2-resident) to fp16 (32 KB, L1) was a measured win, so `VQ_DIM=8, VQ_K=4096` would give
that back.

**This is measurable before anything is downloaded.** The quality question — does a
1.5–2.25-bit VQ hold up — is answerable on GLM-5.2, which is already on disk, using the
existing `bin/ppl` paired-dNLL harness. Re-quantizing one GLM artifact at a candidate
bitrate and scoring it costs no network and answers whether K3 is worth 594 GB of transfer.
Do that first.

**Architecture, which would settle it anyway.** Four independent walls:

- **Hybrid attention.** `linear_attn_config` lists 24 full-attention layers and **69 KDA
  layers** — Kimi Delta Attention, a linear-attention recurrence with a fixed-size state
  rather than a growing KV cache. rivoli has no linear-attention kernel family, and KDA's
  state is a different memory model entirely, not a different set of dimensions.
- **NoPE.** Confirmed by the config carrying no `rope_theta` at all: position comes from KDA's
  decay recurrence. `qk_rope_head_dim: 64` is allocated-but-unrotated width — reading it as a
  RoPE dimension, which rivoli would, is a real bug rather than a graceful degradation.
- **`num_shared_experts: 2`.** `gpu.rs` has a hard `ensure!(cfg.n_shared == 1)`, and the
  `.vq3` file *format* is `n_experts + 1` blocks with the last one being the shared expert.
  Two shared experts is an artifact-format change, not a constant.
- **MXFP4 source.** `quantization_config` is `compressed-tensors` / `mxfp4-pack-quantized`.
  The converter requires `<name>.weight` as F8E4M3 plus `<name>.weight_scale_inv` as F32
  (`artifact/format.rs`). It cannot read this checkpoint at all.

Also: the config keys are `num_experts` / `num_experts_per_token` / `num_shared_experts`, not
DeepSeek's `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts`; and the whole
model config is nested under `text_config` behind a `KimiK3ForConditionalGeneration`
multimodal wrapper. And `num_nextn_predict_layers: 0` — the MTP layer exists in the trained
model but **is not in the release**, so speculative decode is off from the start.

## 3. DeepSeek-V4-Flash-0731 — feasible, but it is a new frontend

`deepseek-ai/DeepSeek-V4-Flash-0731`, `model_type: deepseek_v4`, 284B total / 13B active,
43 layers, hidden 4096, `moe_intermediate_size` 2048, 256 experts top-6 + 1 shared.

**It fits, easily.** Per expert 3 × 4096 × 2048 = 25.2M; × 256 × 43 layers = 277B in experts
(again reproducing the headline). At 0.406 B/weight that is **~105 GiB of experts, ~120 GiB
with residents** — comfortable inside the 265 GiB already free, no deletion required.

**But it would barely stream.** Active experts are top-6 + 1 shared = 7 × 25.2M = 176M
params/layer = 71 MB/layer at int3-vq, × 43 layers = **3.1 GB/token**, against GLM-5.2's
9 × 37.7M × 75 = **10.3 GB/token**. That is 3.4× less traffic per token — and a ~120 GiB
artifact against a ~115 GiB practical `--max-mem` is *nearly fully resident*. The expert
streaming that is this engine's entire reason for existing would be close to idle. That is a
good outcome for tok/s and a strange one for the project's thesis; worth deciding on purpose
rather than discovering after the port.

**What is genuinely missing.** Confirmed from the config, not inferred:

| field in `config.json` | what rivoli has |
|---|---|
| no `kv_lora_rank`, no `qk_nope_head_dim`, no `v_head_dim` | **It is not MLA.** Shared-K=V MQA: one 512-d compressed entry is both key and value for all 64 heads, 64 RoPE / 448 NoPE. `gpu.rs`'s forward is unconditionally `q_a_proj → q_a_layernorm → q_b_proj` + `kv_a_proj_with_mqa` + `mla_absorb_fp8`. No alternative path exists. |
| `hc_mult: 4`, `hc_sinkhorn_iters: 20`, `hc_eps` | **mHC** — the plain residual add is replaced by a 4-stream hyper-connection with per-token A/B/C matrices Sinkhorn-projected onto the Birkhoff polytope. This touches every layer boundary, twice. |
| `num_hash_layers: 3`, no `first_k_dense_replace` | **Zero dense layers.** The first 3 are MoE with static token-ID hash routing — no router logits, no correction bias. rivoli decides dense-vs-MoE purely by `l < cfg.dense_layers`. |
| `scoring_func: "sqrtsoftplus"` | `math.rs` hardcodes `sigmoid(logit) + bias`. Wrong scoring function is the **silent** failure mode: plausible-looking text, wrong experts. |
| `rope_scaling: {type: yarn, factor: 16, …}` | rivoli reads only `rope_theta`; no scaling of any kind, and no `mscale` in the softmax scale (`1/sqrt(qk_head_dim)`, `gpu.rs`). |
| `sliding_window: 128` | A windowed branch on every compressed layer. No equivalent. |
| `quantization_config.scale_fmt: "ue8m0"` | The converter expects F32 `weight_scale_inv`. Block size 128×128 does match `FP8_BLOCK`. |

Three more are **absent from the config and would be found only by reading the paper**: a
negative-position RoPE applied to the attention *output* (the KV entry doubles as the value,
so outputs carry absolute position and it must be undone), a learnable per-head attention-sink
logit in the softmax denominator, and QK-norm (RMSNorm per query head and on the single KV
head). Each is silent-wrong if missed.

`num_nextn_predict_layers: 1` and `tie_word_embeddings: false` do match, and hidden 4096 /
moe_inter 2048 are both clean multiples of `VQ_GROUP`.

## 4. V4-Flash port — what shipped, and what the reference says

Read from `inference/model.py` in the model repo, not inferred.

**SHIPPED 2026-08-03: the router.** `sqrt(softplus(·))` — the reference computes it as
`F.softplus(scores).sqrt()`, so the literal reading of the name is right. Everything
downstream was *already* correct: the reference adds `e_score_correction_bias` for
**selection only** and gathers the weights from the **pre-bias** scores
(`weights = original_scores.gather(1, indices)`), then renormalizes and applies
`route_scale`. That is exactly what `route_into` + `gpu.rs` already did for GLM. So the
whole change is one activation behind a `Scoring` enum read from `scoring_func`, and INV-1's
frozen oracle still pins the sigmoid path bit-for-bit. `softmax` is refused at load rather
than mapped, because the reference skips the top-k renormalization for it
(`if score_func != "softmax"`) and silently applying `norm_topk_prob` there would be wrong.

Implemented in the stable form `max(x,0) + ln1p(exp(-|x|))`: the naive `ln(1+exp(x))`
overflows to `inf` near x=88 in f32 and gate logits are unbounded. Note sqrt-softplus is
**unbounded above** where sigmoid is confined to (0,1) — the correction bias is trained
against this scale, so applying the wrong affinity is the silent-wrong-experts failure, not
a crash.

**The hazard rivoli is primed to hit.** The llama.cpp port reports its worst bug was a q8_0
KV cache producing *"`=`-loops, single-character output, or `"Mirror …"`-style noise instead
of coherent text. No error, no crash."* Cause: V4 pre-quantizes K activations through fp8
e4m3 before cache storage, and re-quantizing already-constrained values corrupts attention
into coherent-looking wrong output. They pin `type_k`/`type_v` to F16 in two places
regardless of user flags. **rivoli has `--kv-fp8`, which does exactly the harmful thing.**
It must be refused on this architecture, not merely discouraged.

**Still unbuilt**, in dependency order: shared-K=V MQA attention (one `wkv: Linear(dim,
head_dim)` + `kv_norm`, one entry serving as both K and V for all 64 heads, per-head QK-norm
via `rsqrt(q.square().mean(-1))`, a per-head f32 `attn_sink` logit in the softmax
denominator, and `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)` — a **de-rotation
of the output**, needed because the KV entry doubles as the value and so carries absolute
position); mHC (`hc_pre`/`hc_post` around both attention and MoE, with `hc_split_sinkhorn`);
hash routing for the first 3 layers (an int32 `tid2eid[vocab, top_k]` table indexed by token
ID — selection bypasses the scores entirely, but the gate weights still produce the
*weights*); YaRN; the 128-token sliding-window branch; and an fp8-`ue8m0`/FP4 converter path,
since V4 ships **no bf16/fp16 distribution at all**.

**The thesis problem, stated plainly.** llama.cpp measured V4-Flash **compute-bound** on the
indexer/sinkhorn/routing path — Q8_0 beat Q4_K_M by 1.7× *despite reading 1.76× more bytes
per token*. Combined with the ~120 GiB artifact against a ~115 GiB `--max-mem`, this model
is close to fully resident and not bandwidth-limited. rivoli's entire advantage is hiding
NVMe latency behind compute. On V4-Flash there is little latency to hide and the compute is
the bottleneck. The port is worth doing as a second architecture; it is not worth doing as a
demonstration of expert streaming.

## 5. Source weight formats — measured from the repos, 2026-08-03

`convert` requires `<name>.weight` as **F8E4M3** plus `<name>.weight_scale_inv` as **F32**
(`artifact/format.rs`). Neither target provides that for its routed experts. Per-dtype
parameter counts from each repo's `safetensors` metadata, and repo sizes from its file tree:

| | DeepSeek-V4-Flash-0731 | Kimi K3 |
|---|---|---|
| download | **162.5 GB**, 48 shards | **1.55 TB**, 96 shards |
| total params | 304.18B | 2.78T |
| routed experts | **296.35B as I8** (FP4 nibbles, 2/byte, e8m0 scales) | **2.72T as U8** (MXFP4, QAT) |
| attention | **6.30B as F8_E4M3** ← the one thing that maps today | — |
| everything else | 1.48B BF16, 37.7M F32, 2.33M I64 | 57.18B BF16, 11.1M F32 |

**Both models' experts are already 4-bit.** That is the fact that decides the artifact
format, and it cuts against re-quantizing to int3-vq:

> `int4-scales.md` records that `vq3_to_i4` — deriving int4 from the *already lossy* vq3
> rather than from fp8 — produced **PPL 73.43**, and the chain was DELETED. int4 became
> usable (5.120) only when derived DIRECTLY from the fp8 source. Feeding a 4-bit expert into
> a 3.25-bit VQ fit is the same shape of chain against the same kind of source.

The difference between the two targets is whether we can afford to keep native precision:

- **V4-Flash: we can.** Native 4-bit experts in the existing `.i4` container (nibble +
  group scale) is ~157 GB, against ~120 GB for int3-vq. Spending 37 GB of an 860 GiB budget
  to avoid a lossy-on-lossy chain is not a close call. Attention stays fp8 — it already is
  fp8, and that half maps onto the resident path unchanged.
- **K3: we cannot.** Native MXFP4 experts are ~1.45 TB. Even with GLM-5.2 deleted
  (709 GB reclaimed → ~1.63 TB free) that leaves ~180 GB of headroom, and the 1.55 TB source
  cannot be local at the same time — it has to stream from `/swarm/storage` during
  conversion, exactly as GLM's fp8 source did. int3-vq at ~1.10 TB is the affordable option
  and still needs GLM deleted.

**The K3 penalty is measurable today, with no network.** GLM-5.2 carries BOTH `.i4` (4-bit,
derived from fp8) and `.vq3` (3.25-bit, derived from fp8) twins for every MoE layer. Quantize
the `.i4` into a vq3 and score it against the shipped vq3-from-fp8 on paired dNLL
(`bin/ppl`, `tests/ppl-corpus-5000.txt`). That is the 4-bit → 3.25-bit chain K3 would take,
measured on data already on disk. Do it before committing to a 1.55 TB transfer.

## What we did not build, and why

No model-family abstraction, no `trait Architecture`, no registry, no per-model plugin. One
supported model plus one candidate is not enough evidence to know where the seams go, and the
seams this investigation actually found (attention frontend, residual topology, router
scoring) are not the seams a speculative abstraction would have cut. The three fixes above are
each a defect repaired in place. If the V4-Flash port proceeds, the abstraction should be
extracted from two working implementations, not designed ahead of one.

Group-limited routing is likewise left unimplemented: no model we intend to run uses it.
