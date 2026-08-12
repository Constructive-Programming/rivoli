---
scope: k3
status: live
verdict: The implementation plan for Kimi-K3, a required capability. Six stages behind six correctness gates; a gate is MET or NOT MET and must be proven able to go red before it is trusted green. S0 IS DONE and G0 IS MET (2026-08-10; the corrections it produced are recorded inline in reference/k3-architecture.md, which is now first-party-verified for everything except the KDA inner arithmetic and the MXFP4 unpack — those live in fla-core and compressed-tensors, and S1b's mandatory anchor now COVERS both by running the first-party stack on gfx1151 over a tiny config that keeps the real 93-layer structure — ELEVEN defect runs, each GATED on the layers it must leave bit-identical, at TWO weight draws, and per-operator tolerances MEASURED: the C reference's MLA LoRA-norm eps divergence is priced at 1.9e-5 relative, which is BELOW that operator's own fp32 rounding floor (0.33x), so MLA is exact-only and the eps must be pinned structurally rather than numerically; measurement/k3-reference/anchor.md). Traffic is 25.83 GB/token of experts (first-party confirmed) plus 108.81 GB of trunk when not resident; predicted ~0.48-0.57 tok/s with the resident set held, from rivoli's measured 12.39-14.76 GB/s at the expert-read shape — an earlier 0.27 came from a dd QD1 figure, the same instrument-class error found in the reference's own 3.2. The memory budget is the binding constraint (~5.7 GB residual at --max-mem 115; the resident set does not fit on the auto path) and nothing converts until SafeWriter streams. Only the 69 KDA layers are a wholly new kernel family, plus AttnRes. S1-S3 run on fixtures; S4 converts and decodes real weights on /swarm; S5 needs ~200 GiB more NVMe.
---

# Kimi-K3 — implementation plan

**K3 is a capability this engine must have.** This is the plan to get there, not an assessment
of whether to.

Target: `moonshotai/Kimi-K3`, text-only. 93 layers, 2.78T total / 104.2B active.
The forward pass is specified in **`docs/reference/k3-architecture.md`**; the third-party
measurements it cites are vendored under **`docs/measurement/k3-reference/`**.

## STATE

- **Six stages, six gates.** S0 ground truth → S1 artifact + harness → S2 kernels → S3 layer
  loop → S4 real weights → S5 residency. §G.
- **G0 is MET** (2026-08-10). §G0 for the record; corrections live inline in
  `k3-architecture.md`.
- **Only the 69 KDA layers are a wholly new kernel family**, plus AttnRes. The 24
  full-attention layers are MLA + q-LoRA, GLM's own family — but rivoli caches the **fp8
  latent** where the reference caches expanded fp32, a deviation under G3's zero-tolerance
  gate and a ~150x smaller KV budget. §3.
- **Traffic is 25.83 GB/token of experts** (first-party confirmed), **plus 108.81 GB of trunk
  when it is not resident.** §4.
- **Predicted ~0.48–0.57 tok/s** with the resident set held. §4c. This replaces an earlier
  0.27 that was computed from a queue-depth-1 disk figure.
- **The memory budget is the binding constraint.** At `--max-mem 115` the 113.51 GB resident
  set leaves ≈5.7 GB; on the **auto** path the budget is `MemAvailable − 16 GiB` ≈ 107 GB and
  **the resident set does not fit at all**. §4d.
- **Both writers stream as of 2026-08-10** — `SafeWriter` for the resident tensors and
  `write_expert_layer` for the routed ones, the latter in 1 GiB windows against a 15.72 GB
  layer. Conversion is unblocked. §S1a item 1. *(This said "the ROUTED writer is still owed"
  until 2026-08-10, contradicting item 1 in the same document; corrected after review.)*
- **The load boundary is in and refuses**: `Arch::KimiK3`, `K3Config`, and a dispatch arm that
  parses before it bails. §S1a items 6 and 7.
- **`convert_k3` exists and the repack is verified on real bytes.** Names and shapes come from the
  checkpoint's own index (vendored reduction, `tests/k3_names.rs`); one real expert was fetched by
  HTTP Range and converted with 0 bytes differing, re-checked independently of rivoli's code.
  §S1a items 5 and 6, `docs/measurement/k3-reference/repack-one-expert.md`.
- **Three of G1a's four bullets are MET** — the repack (on a one-expert sample, recipe recorded),
  the refusal at startup, and the existing artifacts still opening byte- and offset-identically
  with their byte accounting reproduced from the config (`tests/artifact_compat.rs`, 805 GiB in
  0.1 s).
- **S1a IS DONE.** Item 2 settled (the e8m0 `0xff` bail stays — the repack is the only path that
  reads every ROUTED scale byte); item 4's `moe_acc_drain_to` written, templated against its sibling,
  and **executed on gfx1151 — 8 elements bit-identical, its `1001` dimension guard refusing both
  rows** (it took a `gain` and a second guard for one day; see item 4 for why both went); item 7's `MAX_BATCH`
  arithmetic settled and asserted, **K3 fits the routed batch scratch at ONE row and not at two**
  (18 of 32 against 34). The whole device sweep passed in the same window (`kernel` 24/24,
  `f4_kernel` 24/24, `kvcompress_kernel` 10/10, `--lib` 141/141).
- **S1b's ANCHOR exists and runs, 2026-08-11** — the first-party stack (`modeling_kimi_linear.py` at
  the pinned revision, fla-core 0.5.2, transformers 4.56.2) executed on gfx1151 over a tiny config
  that keeps the **real 93-layer structure**, with ELEVEN defect runs, each GATED on the layers it
  must leave bit-identical rather than merely on having changed something, at TWO independent weight
  draws. Both decode goldens are vendored (324 KiB each) and read with no GPU, no
  python and no network. `docs/measurement/k3-reference/anchor.md`.
- **The per-operator TOLERANCES are measured and gated**, `tests/common/k3_tolerance.rs`. The MLA
  LoRA-norm eps divergence sits at **0.33x that operator's own fp32 rounding floor** — *below* it —
  so **no tolerance can separate it and the eps must be pinned structurally**, by reading the
  constant. Every other operator has 90,000x to 7M x of room and is set at 10x its floor.
  **S1b IS DONE**; S3's layer loop remains for the pin.
  *(This read "2.2e-5 relative, which is only 1.3x" until 2026-08-12. That was the ONE-DRAW reading
  taken before item 2 captured the attention core; re-measured over both draws and with `mla_attend`
  split out, the margin fell to 0.33x — a stronger form of the same finding, and `anchor.md`
  §"Re-measured on both draws" carries the numbers.)*
- **S2 items 1-4 and 5a are DONE, each gated before the next** (2026-08-11/12): `attn_res`, the
  gated MLA core (`mha_attend` + `sigmoid_gate`), the latent sandwich, SiTU-GLU + the fused fp4
  expert, and the gated delta recurrence (`gated_delta_recurrent_f32`). `tests/k3_kernels.rs`,
  26 tests.
  **Each of the first three found the anchor could not score its operator, and only by trying to write
  the fixture** — the fold weights, the attention core and `o_proj`'s input, the expert aggregate and
  the norm weight. Goldens re-vendored three times, now at **272** tensors. Item 3 also found the
  first place the anchor's fp32-vs-bf16 deviation bites: an operator whose rivoli kernel rounds to
  bf16 cannot be scored against this anchor at the anchor's own tolerance. **5a found that the
  reference stores its recurrent state `[value][key]` and that rivoli should NOT** — the first
  reference convention this port measured and then declined, at no cost, because the state never
  leaves the device. **5b and 5c are next and share ONE regeneration** (the conv taps and
  `o_norm.weight`); then 5d's tolerance row, item 6 (the router) and item 7 (trunk GEMV speed).

## G. The gate model

A gate is **met or not met** — no partial credit, no proceeding-while-noting. When a gate is
not met, the work is to make it met. Three rules bind every gate here.

**1. Prove it can go red.** State what would have to be true for the gate to reject a wrong
implementation, then break the implementation deliberately and confirm it reddens **at every
case the defect touches and stays green at every case it does not**. A gate that reddens
everywhere proves nothing; one that never reddens is decoration.

**2. Name the blind spot.** Layer 0 is dense FFN, no routing, and is what everyone checks —
it is also simultaneously a KDA layer and an AttnRes boundary, so it is the *least*
representative layer. Gates must cover a KDA layer, an MLA layer, layer 0, and layer 92.

**3. Fluent wrong text is the failure mode.** None of this port's likely defects crash.
`distinct` and `longest repeated block` cannot see any of them and have misled three
investigations here. Gates are numeric or they are nothing.

| gate | after | asserts |
|---|---|---|
| **G0** | S0 | every unknown has a recorded answer and a source |
| **G1a** | S1a | `.f4` repack bit-exact both ways; existing artifacts still open byte-identically |
| **G1b** | S1b | every golden demonstrably reddens on a deliberate defect; independent anchor in place |
| **G2** | S2 | each kernel passes its operator fixture **and** its defect run, in order |
| **G3** | S3 | the three zero-tolerance model gates pass on the tiny model; KDA state carries |
| **G4** | S4 | real-weight decode coherent and on-task; byte accounting reproduced from the artifact |
| **G5** | S5 | throughput inside the registered band; output byte-identical to G4 |

## 1. Ground truth — the config

`moonshotai/Kimi-K3/config.json`, read 2026-08-09. The text model is nested under
`text_config`, behind a `KimiK3ForConditionalGeneration` multimodal wrapper.

> **VENDORED 2026-08-10.** The file itself is now at
> `docs/measurement/k3-reference/config.json` — `moonshotai/Kimi-K3` revision
> `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, 7,006 bytes, sha256
> `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` — and
> `model.rs::k3_base_matches_the_shipped_config` pins every fixture value to it via
> `include_str!`, so it runs always rather than skipping like V4's twin.
>
> **Reading it corrected three things this table had wrong or vague**, each of which had already
> become a defect in `K3Config`: the `activation_situ_linear_beta` spelling (below);
> `use_full_rank_gate` living **inside `linear_attn_config`**, not beside it; and the "+5 scalars"
> hand-wave, whose real keys are `num_heads`, `head_dim`, `short_conv_kernel_size`,
> `gate_lower_bound` and `use_full_rank_gate` — **not** the `kda_heads` / `kda_head_dim` /
> `conv_k` that `k3-architecture.md` §1 lists, which are the C reference's field names. Two of
> the three would have refused every real K3 checkpoint on `missing field`.
>
> **And one trap the table never mentioned:** `text_config` carries a key literally named
> `top_k`, which is **50** — HuggingFace's sampling top-k, inherited from `PretrainedConfig`,
> nothing to do with routing. Binding the router from it selects 50 experts a token instead of
> 16: 3.1x the stream traffic, plausible output, no error. The MoE count is
> `num_experts_per_token`.

| field | value | consequence |
|---|---|---|
| `architectures` / `model_type` | top-level `KimiK3ForConditionalGeneration` / `kimi_k3`; **nested `text_config` says `KimiLinearForCausalLM` / `kimi_linear`** | recognise on the **top** level and assert the nested pair as a secondary check — S1a descends into `text_config`, so a recogniser that descends first will refuse the real checkpoint |
| `num_hidden_layers` | 93 | 1 dense + 92 MoE |
| `linear_attn_config` | dict: `full_attn_layers` (24, one-based) + `kda_layers` (69) + 5 scalars | 24 + 69 = 93. §2 |
| `num_experts` / `num_experts_per_token` | 896 / **16** | the traffic figure, §4 |
| `num_shared_experts` | 2 | one **fused** MLP on disk, §3. **The 2 is load-bearing for a SHAPE and only a test knows it**: `shared_experts.down_proj` is `[hidden, 2·moe_inter]` = `[7168, 6144]`, pinned by `tests/k3_names.rs`, while `validate` accepts any positive value. Nothing is wrong today — no shared-expert code exists — but S3 inherits an unchecked coupling, so read the width from `n_shared · moe_inter` rather than from 6144 |
| `routed_expert_hidden_size` | **3584** | the latent, **not** `hidden_size` 7168. §2 |
| `moe_intermediate_size` | 3072 | |
| `hidden_size` / `num_attention_heads` | 7168 / 96 | |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | `kernels/attn.hip:293` needs `kvl % 128 == 0` and `kvl <= MLA_ACC_REGS*SUBW`; 512 passes both |
| `mla_use_nope` | **true** | assert this **positively**; §3e |
| `mla_use_output_gate`, `latent_moe_use_norm`, `moe_renormalize` | true | each an explicit `K3Config::validate` assertion, never a defaulted field |
| `num_expert_group` / `topk_group` / `topk_method` | 1 / 1 / `noaux_tc` | grouped routing is **degenerate, not absent** — assert both are 1 and refuse otherwise. `noaux_tc` is the first-party name for bias-on-selection-only |
| `activation_situ_beta` / **`activation_situ_linear_beta`** | 4.0 / 25.0 | SiTU-GLU, fused into the fp4 kernel — §3b. **CORRECTED 2026-08-10**: this row abbreviated the second key as "`_linear_beta`", which reads as `activation_linear_beta` — and that is how S1a first declared it, which would have refused every real checkpoint on `missing field`. Never abbreviate a key in this table |
| `first_k_dense_replace` | 1 | layer 0 dense, `intermediate_size` 33792 |
| `attn_res_block_size` | 12 | AttnRes, `k3-architecture.md` §3 |
| `vocab_size` | 163,840 | 163,584 BPE + a 256-id reserved block; `tie_word_embeddings` false |
| `num_nextn_predict_layers` | 0 | no MTP, **no speculative decode — so `MAXROW` is 1 and no `_r2` twin has a caller** |
| `quantization_config` | `mxfp4-pack-quantized`, group 32 | **its `ignore` list mis-declares its own scope** — see S1a |
| `vision_config` | 27 blocks + `mm_projector` | **out of scope**; the converter must skip them explicitly |

## 2. The shape that produces fluent wrong text

**Routed experts are 3584 wide; `hidden_size` is 7168.** The MoE block routes on full width,
down-projects 7168→3584, runs the experts at 3584, RMSNorms the **aggregate**, and up-projects
back. rivoli assumes expert input width **is** `hidden` throughout the MoE path, and that
assumption is load-bearing in the arena's row stride and the fp4 dot's `dim`.

The layer map needs no inference: `full_attn_layers` is explicit, and the *weights* confirm it
independently — zero-based MLA is **[3, 7, …, 87, 91, 92]**, the last two **adjacent**. Full
spec and the twelve order-of-operations traps are in `k3-architecture.md`.

## 3. What is new, what is reused

Every row verified against the tree 2026-08-09.

| piece | status |
|---|---|
| `.f4` E2M1/E8M0 **value encoding** | **reuse** — bit-identical; the container differs |
| expert streaming, io_uring fetch, byte arena, residency cache | **reuse verbatim** |
| sigmoid router with selection-only bias | **reuse** — `Scoring` enum ships it |
| `Arch` enum, per-arch help and refusal | **reuse** — `arch.rs` is already the multi-arm shape |
| bf16 trunk GEMV | **exists** — `gemm_bf16` (`kvcompress.hip:82`), already V4's lm_head. A *performance* problem, not a new family |
| `resident.safetensors` bf16 | **already accepted** — `Dtype::Bf16`, `format.rs:100` |
| `.f4` shared block | **VOID** — `has_shared()` is false for F4, which is already correct: K3's shared MLP is bf16 and trunk-side |
| **KDA, 69 layers** | **new kernel family** — the bulk of S2 |
| **AttnRes** | **new**, and unplanned until S0 found it |
| **fp4 expert dot + SiTU-GLU** | **new variant** — §3b |
| MLA + q-LoRA, 24 layers | reuse `mla.hip` minus RoPE — but the **gate is new kernel work**, and the **KV representation deviates**, below |
| expert width 3584 ≠ hidden 7168 | assumption break, §2 |
| tiktoken 163,840 vocab | new tokenizer path + a hand-ported `encode_k3` |

**§3b — SiTU-GLU is fused inside the fp4 expert kernel.** `kernels/moe.hip:329` computes
`swiglu_clamped(g[t], u[t], limit)` *inside* `moe_gateup_f4_impl`, `limit` passed at `:406`,
and the launcher refuses any value that disables the clamp (`:439`). There is no separate
activation launch on the `.f4` path. So SiTU-GLU means **a new `moe_gateup_f4` variant**, plus
the SiTU form in `linalg.hip` for dense layer 0 and the shared MLP. An implementer who changes
`linalg.hip` alone watches the dense path go right and every routed expert stay wrong.
**No `_r2` twin** — `num_nextn_predict_layers: 0`, so nothing batches.

**§3c — the MLA KV representation is a real deviation.** rivoli **absorbs** and caches the
**fp8 latent** (`gpu.rs:943`, `mla_absorb_fp8`) at **15,744 B/pos** across 24 layers; the reference
caches **expanded fp32** per-head k/v at 2.37 MB/pos. ~150x smaller, and a different
algorithm. Two consequences: the budget line is ~150x smaller than a reading of the reference
implies, and **G3's zero-tolerance token match is asserted across a representation rivoli does
not implement.** Run the incremental gate first with the quantization disabled, so a mismatch
is attributable to the recurrence rather than the cache format.

**§3e — NoPE: require the positive flag.** `mla_use_nope: true` is stated in the config.
Asserting only the *absence* of `rope_theta` cannot distinguish "this model is NoPE" from "we
descended into the wrong dict" — and this plan descends into `text_config` behind a wrapper,
which is exactly how a key goes missing for the wrong reason. Require `mla_use_nope`, assert
`rope_theta` absent as a secondary check, and refuse to construct a rotation table.

## 4. Traffic, bandwidth, and what it predicts

Per token every MoE layer selects 16 of 896 experts:

```
16 × 17,547,264 B = 280,756,224 B/layer × 92 = 25,829,572,608 B = 25.83 GB/token
```

**Confirmed first-party**: the shard header gives `w1.weight_packed` U8 `[3072, 1792]` and
`w1.weight_scale` U8 `[3072, 112]`, which reconstruct to 16,515,072 + 1,032,192 =
**17,547,264 B/expert** exactly. Plus **108.81 GB of trunk** whenever it is not resident —
`k3_trunk.c:361`, "the trunk is read 93 times per token".

### 4a. The expert cache is weak here, and not for the reason first assumed

The reference reports 0.0% expert hit below ~36 GB of arena, 43.8% at 224 GB. That is **not**
flatness: 36.39 GB holds 2,074 of 82,432 experts = 2.52% residency returning a 29.9% hit,
**11.9× uniform**. The distribution is skewed; what produces the 0% rungs is budget
competition with the trunk, or a threshold effect, and the reference's data cannot separate
them.

**The design consequence follows from bytes alone and needs no timing:** the trunk is 108.81
GB that every token needs in full, while at 2.5–7.5% residency the cache returns a fraction of
25.83 GB. Hold the larger, wholly-required object.

### 4b. Two bandwidth errors, both ours

**The reference's 3.2 GB/s is a `dd` queue-depth-1 figure**, and its engine sustains 5373–6064
MB/s (`k3-reference/environment.txt:36-42`). An earlier draft used 3.2 as a floor, found the
ladder beneath it, and called the trunk-miss model unestablished. There was no paradox.

**But the corrected reading does not confirm the model either.** The reference's own README
shows **2709 and 5874 MB/s on bit-identical work** — a 2.17× spread — and `replication.tsv`
records a **33.1% noise floor** across three back-to-back runs of one configuration. At the
low arm the ladder is again sub-floor; at the high arm it admits any trunk model from 0× to
1.5× re-read. **The ladder cannot discriminate the trunk-miss model in either direction.** It
is confirmed by `k3_trunk.c`'s bind count, not by timing.

**And rivoli's own 7.0 GB/s is the same instrument class** — `other-models.md:336`,
"`dd iflag=direct`, 4 GiB". rivoli's probes at the expert-read shape (15.34 MB, against K3's
17.55 MB) measure **12.39 GB/s at QD1 and 14.76 at QD16** (`benchmarks.md` "Storage: sequential ordering buys nothing at QD>=2"). The MoE layer
issues 16 concurrent expert reads, so QD16 is the operating point.

### 4c. The predicted operating point

Resident set held, expert arena effectively zero, reads at the measured QD band:

```
25.83 GB / 12.39 GB/s = 2.08 s/token = 0.48 tok/s      (QD1)
25.83 GB / 14.76 GB/s = 1.75 s/token = 0.57 tok/s      (QD16)
```

**Register ~0.48–0.57 tok/s**, a band and not a point, and record which QD the MoE layer
actually achieves. This supersedes an earlier 0.27 computed from the `dd` figure — a 1.8×
error, larger than anything S5 could hope to detect.

### 4d. The budget, which is the binding constraint

`--max-mem 115` GiB = 123.5 GB deducts **no** OS reserve (`config.rs:110` — "the user asked
for it"). Resident set 113.51 GB, `DeviceTier::HEADROOM` 4 GiB → **≈5.7 GB residual**, against:

| claimant | size |
|---|---|
| KDA state | 464.6 MB (69 layers) or 626 MB as the reference allocates |
| AttnRes stack, `[T][9][7168]` fp32 | **T × 258 KB** — 1.06 GB at 4k, 2.11 at 8k, 4.2 at 16k |
| MLA KV, rivoli's fp8 latent | 656 B/layer x 24 = **15,744 B/pos** -> 129 MB at 8k |
| `RoutedPool` one-batch floor | 281 MB (16 × 17.55 MB) |
| residual stream, scratch, io_uring buffers | unpriced |

**Three claimants, one budget, and AttnRes at prefill is the largest.** On the **auto** path
(`MemAvailable − OS_RESERVE` ≈ 107 GB) the resident set does not fit at all. S5's band must
come from a budget with these subtracted.

## 5. Capacity

```
experts   2,722,740,830,208 params × 0.53125 B = 1.4465 TB
resident set (trunk 108.81 + embed/lm_head 4.70) = 113.51 GB
  (the C reference rounds this to 113.49; its components are what sum)
index metadata.total_size                        = 1,560,860,324,864 B = 1.4196 TiB
```

Pool measured 2026-08-09: **1.69 TiB total, 431.72 GiB free**, GLM at ~~675~~ **659.25** GiB, V4 at
**145.97** GiB. *(CORRECTED 2026-08-11 by `tests/artifact_compat.rs`, which derives both from the
config and confronts the disk: GLM is 76 x `.vq3` + 76 x `.i4` + 2 resident = 707,865,529,324 B =
659.25 GiB, so the 675 was 15.75 GiB high and no directory here measures it. V4's ~146 was right —
and a first pass that summed only its `.f4` files got 137.06 GiB and nearly reported the plan wrong
in the other direction.)*
Deleting both frees 1.223 TiB — **200 GiB short**. But **`/swarm/storage` has 7.7 TiB**, so the
artifact can be stored and converted today; NFS at 154 MB/s is useless for throughput and fine
for a bounded correctness run.

| work | where | blocked? |
|---|---|---|
| convert, verify byte accounting, real-weight correctness run | `/swarm` | **no** |
| throughput measurement, residency study | NVMe | **yes** — ~200 GiB short |

Fidelity is native MXFP4, decided at planning time.

---

# Stages

## S0 — ground truth. No code, no weights.

Reference pinned at **`ff11dce858a2eb8a781224facdffd33a1fa48d25`** (2026-08-07). Measurements
vendored under `docs/measurement/k3-reference/`. Forward pass extracted to
`docs/reference/k3-architecture.md` and re-verified against raw pinned source.

### G0 — **MET 2026-08-10** (third pass; reopened twice, both times because "no checkpoint
download" never excluded metadata).

| # | question | answer | source |
|---|---|---|---|
| 1 | 7168→3584 | Latent sandwich. `routed_expert_down_proj` **[3584,7168] BF16**, `up` **[7168,3584]**, `norm` **[3584]**; experts `w1` U8 **[3072,1792]** nibble-packed, `w2` U8 **[3584,1536]** | index + shard header |
| 2 | layer map | MLA = layers carrying `kv_a_proj_with_mqa`: zero-based **[3,7,…,87,91,92]**, n=24; KDA n=69; disjoint, union 0..92 | index (weights) + `config.json` |
| 3 | trunk dtype | **BF16.** Trunk 108.81 GB; +4.70 embed/lm_head = 113.51 resident; index total 1.4196 TiB; non-expert 114.40 GB incl. vision | index + header |
| 4 | shared experts | **ONE fused MLP per layer**, `down_proj` **[7168,6144] BF16**. Not two | index + header |
| 5a | expert byte layout | U8 packed + U8 group-32 scales reconstruct to **17,547,264 B/expert** exactly | header |
| 5b | nibble order, `sb==255` | Low nibble = even; 255 → zero. **THIRD-PARTY, unconfirmed** — a safetensors header cannot express nibble order | `k3_ops.c` only |
| 6 | vocab | 163,584 BPE + **256 reserved ids 163584–163839** (16 defined) = 163,840 | `tokenizer_config.json` |
| 7 | trunk re-read | **Confirmed by bind count** (93/token), **not by timing** — §4b | `k3_trunk.c:361` |
| 8 | pin the reference | Ladder, split sweep, `environment.txt`, `replication.tsv` vendored | — |
| 9 | correct `other-models.md` §2 | done, dated note in place | — |
| 10 | first-party index | **Done.** 497,220 tensors → **60 families, 48 of them text-side** (CORRECTED 2026-08-11: this said "50 families" under an unstated reduction rule. The rule is now stated and mechanical — collapse `.layers.<n>.`, `.experts.<n>.` and `.blocks.<n>.` — and the result is vendored at `docs/measurement/k3-reference/tensor-families.tsv`, asserted by `tests/k3_names.rs`. A family count with no rule behind it is not a measurement), **every text-side family maps to a documented section**; `g_proj`/`o_proj` on all 93 (both families gate), `o_norm` on 69 (KDA only — confirming MLA gates without a norm), `mlp.*_proj` on exactly 1. **No tensor in the checkpoint is unexplained** | index |
| **11** | **first-party modeling code** | **DONE 2026-08-10** — results below | `modeling_kimi_linear.py` @ HF, read raw |

**Item 11 — results.** ~156 KB of first-party source read raw and checked line-by-line
against `k3-architecture.md`. **8 of 12 traps confirmed** (MLA, AttnRes, MoE, router, SiTU-GLU
— the `M:` citations are recorded inline in that doc); **2 corrected** (the layer-array
guidance was inverted, `k3-architecture.md` §2; `A_log` ships [128] on disk against the
modeling code's own [96], §4); **1 divergence found in the C reference itself** (MLA's LoRA
norms: eps 1e-6 first-party vs the C's 1e-5, §5); **4 not attestable from the checkpoint** —
the KDA inner arithmetic delegates to `fla-core` and the MXFP4 unpack to compressed-tensors
(§4, §9), which is what S1b's anchor now exists to cover.

Two plan-side actions fell out: the MXFP4 `sb == 255` semantics must be settled against the
compressed-tensors unpack or real scale bytes (S1a item 2), and `use_full_rank_gate` /
`mla_use_output_gate` **default False in code** — assert them true from config (§1).

> **BOTH DONE 2026-08-10 for the second action, with a correction.** `mla_use_output_gate` is on
> `text_config`; **`use_full_rank_gate` is inside `linear_attn_config`**, so "assert them from
> config" needed two different levels and the S1a code asserted both at the outer one — which
> refused nothing, because serde ignores unknown keys, and would have refused every real
> checkpoint on `missing field`. Both are now asserted at their real levels, and the vendored
> config makes that permanent. Worth stating plainly: the config-vs-code disagreement this item
> found is real and the config wins — the weights agree with it, since every layer ships a
> `g_proj`.

## S1 — foundation. No GPU.

### S1a — artifact: config, `Arch`, naming, `.f4`

1. **`SafeWriter` streams — DONE 2026-08-10.** It carries a `Cow` per tensor
   (`format.rs:169`), so verbatim copies borrow the source mmap and host-RAM peak is the sum of
   the CONVERTED tensors. `write` also became atomic: a borrowed payload is read at write time,
   so truncating a path that is also a mapped source would SIGBUS.
   **The ROUTED writer — DONE 2026-08-10.** Both converters buffered a whole layer through
   `write_atomic(&path, &buf)`: 3.4 GB for V4, 3.7 GB for GLM, **15.7 GB** for K3
   (896 x 17.55 MB) on a 128 GB host whose RAM the GPU shares. `write_expert_layer`
   (`format.rs`) streams it in 1 GiB windows and keeps `fill_expert_blocks`'s thread-parallel
   pack inside each window — at K3's stride that is 61 blocks per window against ~32 threads,
   so nothing serialises. Asserted byte-identical to the buffered form, short final window
   covered. `write_atomic` went with it — after this there were no callers left.
2. **Settle e8m0 `0xff` — SETTLED 2026-08-11: the bail STAYS.** The reference maps 255 → zero;
   rivoli's `e8m0f` returns a quiet NaN and `quant::e8m0` **bails**. Host and device must move
   together or the divergence is silent — `moe_fixed`'s clamp launders NaN into a finite ±2^14.
   **Two grounds, and the second is the one that decides it.** Measurement: 4,128,768 real K3 scale
   bytes (every scale tensor of experts 0-3, layer 1) hold 11 distinct codes in `0x70..=0x7a`, zero
   `0xff`, zero `0x00` — the same shape V4's shipped set showed, so the reference's 255 path is
   defensive rather than exercised. That is a **0.005% sample and settles nothing alone.** What
   settles it: **the repack is the only path that reads every scale byte** — at decode they DMA from
   NVMe into a pool slot and the host never sees them — so `F4Expert::spans`'s existing check either
   passes over the whole checkpoint at conversion or names the exact tensor, row and group. Every
   `.f4` writer goes through it, `convert_k3` included. Adopting 255 → zero would mean adopting a
   rule for values the format forbids and this engine's artifacts cannot contain, in
   `common.hpp::e8m0f` as well, where nothing can report it. Recorded at `quant::e8m0` and in
   `docs/measurement/k3-reference/repack-one-expert.md`.
3. **Thread `moe_latent` (3584) separately from `hidden` (7168).** *(Tier 1. Host half DONE
   2026-08-10.)* The fp4 kernels already take the widths as runtime arguments
   (`moe.hip:300`) and `ACT_QUANT_BLOCK=128` divides both 3584 and 3072, so the work is
   entirely in what Rust binds.
   **Done:** the expert-geometry layer's `hidden` is now `expert_in`, named for the role —
   `quant::vq_expert_layout` (the chokepoint all six `*_expert_{bytes,stride}` /
   `*_slot_offsets` go through) plus the three structs a K3 call site fills: `SetDims`,
   `F4Expert`, `ExpertHeader`. Rename-only; `ExpertHeader`'s 40-byte on-disk layout is
   unchanged. Cited by name rather than line, since these move:
   `quant::vq_expert_layout` carries the argument, `model::ensure_group_aligned` the check.
   **Still to do, and the list matters more than the rename:** every call site still binds
   `cfg.hidden` positionally, which is right for GLM and V4 and wrong for K3. The latent sites
   are `SetDims::new` in both pins, and the `*_expert_bytes` / `*_slot_offsets` calls in
   `pin.rs`, `convert*.rs` and `fp8_to_i4.rs`.
   **Sites that must stay FULL width (7168), because a mechanical "bind latent wherever the
   parameter is now called `expert_in`" pass breaks them silently:**
   - `f4gpu.rs` shared-expert dispatch (two sites) — the trunk-side `[7168,6144]`.
   - `pin.rs:373`/`:375` — the SHARED expert's resident footprint, computed with
     `i4_expert_bytes`/`vq_expert_bytes`, i.e. the routed geometry. Correct for GLM only
     because its shared expert has the routed dims.
   - `pin.rs:466` — `shared_off` reuses the routed `*_slot_offsets`.

   Decode sites where a wrong value passes every length check: `f4gpu.rs` MoE dispatch, and
   the accumulator, which must be latent-wide (item 4).
   Note for K3: `ensure_group_aligned(latent, moe_inter, …)` stops covering the trunk widths,
   so its "one check covers both" comment no longer holds there. No practical hole — 7168 and
   6144 are multiples of both 32 and 64 — but the comment needs the caveat when K3 lands.
4. **The MoE accumulator drains into the residual — kernel DONE, executed on hardware.**
   `moe_acc_drain` fuses de-fixed-point into the residual add at one width; K3 needs the aggregate
   intercepted **in latent space** for the RMSNorm and up-projection first. `moe_acc_drain_to`
   (`kernels/moe.hip`) is that kernel: `out[o] = (Σ_r acc[r][o])·2⁻⁴⁴`, accumulator reset, with
   `launch_moe_acc_drain_to`. The two kernels share ONE templated body and differ in exactly one
   line, `=` against `+=` — the one difference the code cannot make visible, so it is argued at the
   kernel and pinned by `tests/kernel.rs::moe_acc_drain_to_writes_the_latent_aggregate_and_resets`,
   whose destination is pre-filled with a poison value.
   **It takes no `gain`, and that is the S3 trap this item exists to spell out.** It had one, with a
   guard, for a day. A positive scalar applied to this buffer is erased by the RMSNorm that
   immediately follows it, so the parameter could not be used correctly and an inert knob is worse
   than no knob (review 2026-08-11). `routed_scaling_factor` is not a candidate either: it
   multiplies the ROUTER WEIGHTS inside the sum, which is where S3 must put it.
   **EXECUTED AND PASSING 2026-08-11** on gfx1151: 8 elements bit-identical, guard 1001 refusing
   both non-positive dimensions. The full device sweep ran in the same window: `kernel` 24/24,
   `f4_kernel` 24/24, `kvcompress_kernel` 10/10, `headtail` 5/5, `f4_pin`/`f4_pool`/`f4_loading`,
   and `--lib` 141/141 — including the two `DeviceTier` tests that fail under contention. **Deleting
   the `gain` changed the kernel's ABI, so both device suites were re-run the same day: `kernel`
   24/24 (179.41 s), `f4_kernel` 24/24 (16.17 s), sole tenant, 0 KFD holders at start.**
   The GPU was freed by **unloading the `ai/llama-swap` models over its ClusterIP**
   (`POST http://10.43.48.47:8080/unload`, 41.4 GB of GTT → 174 MB), which is reversible — they
   reload on demand and the service never went down. That is the procedure; the two heavier ones are
   worse. **A `kubectl cordon` + `drain` does NOT hold the node empty**: tried 2026-08-11, the
   ReplicaSet re-scheduled onto the cordoned node within seconds because its tolerations do not cover
   `unschedulable`, so a run that looks sole-tenant is not. Scaling the deployment to 0 does work but
   takes the AI service down for the window. `CLAUDE.md`'s measurement-discipline list now carries
   this, because it is where someone hunting for GPU tenancy will look.
5. **Converter naming.** Expert tensors are `.weight_packed` / `.weight_scale`
   (compressed-tensors), *not* `.weight` / `.weight_scale_inv`. And **`quantization_config`
   mis-declares its own scope**: `targets: ["Linear"]` with an `ignore` list that omits
   `routed_expert_{down,up}_proj` and `block_sparse_moe.gate.weight`, all three of which ship
   **BF16** on disk. Drive off the presence of `.weight_packed`; do not trust the config.
   Skip `vision_tower` and `mm_projector` explicitly.
6. `.f4` repack, `Arch::KimiK3` plumbing. **Plumbing DONE 2026-08-10; the repack and the
   converter are NOT.**
   **Done:** `Arch::KimiK3` and its six arms, recognised on the **top** level only
   (`KimiK3ForConditionalGeneration` / `kimi_k3`) — the nested `KimiLinearForCausalLM` /
   `kimi_linear` pair is deliberately *rejected* by the recogniser, because it names the
   linear-attention family rather than this checkpoint; `K3Config::validate` asserts it
   instead, where the key it descended through is available to quote. `attn_modes` is `None`
   and the same four flags V4 hides are hidden. `K3Config` + `K3TextConfig` +
   `impl ArchConfig`, every field required, plus the `main.rs` dispatch arm — which **parses
   the config and then bails**, in that order, so the schema is reachable from the binary
   rather than only from unit tests. Verified end to end against a hand-written manifest: the
   good one logs `93 layers (24 MLA / 69 KDA) … latent 3584` and refuses to decode; one with
   `mla_use_nope: false` refuses at startup naming the field.
   **`lib.rs` needed nothing** — `artifact::model` is already `pub`, so the plan line asking
   for an export was wrong.
   **Still to do:** the `.f4` repack and `src/bin/convert_k3.rs`. **`convert_k3` must copy the
   source `config.json` into the manifest with its `text_config` wrapper INTACT** (adding
   `format`, as `convert_v4.rs` and `convert.rs` both do). A converter that helpfully flattens
   the nesting away produces an artifact the shipped binary refuses on its own manifest —
   `K3Config` requires `text_config`, and the wrapper is also the only level that names the
   architecture. Surfaced by review 2026-08-10, before it cost a re-conversion.
   **The refusal obligation moved, it did not go away.** There is no `run_k3` yet, and an
   unconditional bail refuses every flag by refusing the run — so the `--port`/`--mode`/
   `--attn` bails are absent on purpose, with that reasoning recorded at the dispatch arm.
   **When the decode path lands, `run_k3` must hand-write them**: they are bespoke bails in
   `run_v4`, not match arms, so a `run_k3` that omits them compiles clean and silently accepts
   the flags. `arch.rs` hiding a flag from `--help` is not the parser rejecting it.
7. Assert the config scalars of §1 rather than defaulting them — **DONE 2026-08-10**, in
   `K3TextConfig::validate`: the five positive flags (`mla_use_nope`, `mla_use_output_gate`,
   `use_full_rank_gate`, `latent_moe_use_norm`, `moe_renormalize`), the degenerate routing
   groups, `topk_method == "noaux_tc"`, `num_nextn_predict_layers == 0`,
   `!tie_word_embeddings`, both SiTU betas in the **f32** domain the kernel works in, the
   zero-width widths (0 passes every divisibility check), and the `full_attn_layers` /
   `kda_layers` **partition** of `1..=n_layers`. `every_k3_field_is_required` is red-proved
   by injecting `#[serde(default)]` on `mla_use_nope`: it fails with "has a default", which
   is the false-green shape V4's `index_topk` note describes.
   **§3e is implemented in BOTH readings**, which an earlier draft of this item got wrong by
   treating them as alternatives: `mla_use_nope` is required true, *and* `rope_theta` is carried
   as an `Option` that `validate` refuses when present. Without the second, `K3TextConfig`'s
   lack of `deny_unknown_fields` means a rotary base sitting in `text_config` is silently
   ignored — which is the wrong-dict signal §3e wanted the second reading for.
   **Also added after review:** `kv_lora_rank` is checked against guard 1004's two bounds
   (`% 128 == 0`, `<= MLA_ACC_REGS * SUBW` = 512) that §1 recorded and item 7 had dropped. A
   **zero** passes the kernel's own guard — `0 % 128 == 0` and `!(0 > 512)` — so 24 layers of
   attention would contribute nothing with no error anywhere.
   **Two config-shaped gaps, both deliberate and both loud if wrong.** `linear_attn_config`'s
   five scalars (`kda_heads`, `kda_head_dim`, `conv_k`, the gate lower bound, one more) are
   absent because their JSON key *spellings* are not among the fields §1 verified against the
   shipped file, and a guessed key on a required field refuses the real checkpoint. So is
   `quantization_config`, for the opposite reason — item 5 says not to trust it.
   **The `MAX_BATCH` bound — arithmetic SETTLED 2026-08-11, and it is binding.**
   `RoutedPool::submit`'s hit scratch is a fixed 32 slots, and a batched forward submits the UNION of
   every row's picks: `top_k · rows + n_shared`. At K3's scalars that is `16 · rows + 2` — **rows=1
   needs 18, rows=2 needs 34, two over.** K3 fits at one row and only at one row, which it has
   because `num_nextn_predict_layers` is 0. `pin.rs:409` is GLM's copy of this check and reaches for
   the global `crate::gpu::MAXROW` (2, fixed by GLM's own acceptance measurements): a K3 pin that
   copies that line refuses every K3 artifact at load, and one that copies it after someone raises
   `MAX_BATCH` silently sizes a batched pass K3 has no kernel for (`kernels/moe.hip` instantiates
   R=1 only for the f4 path; the VQ and int4 paths do instantiate R=2). **Both directions** are
   asserted by `model.rs::k3_fits_the_routed_batch_scratch_at_one_row_and_not_at_two` — red-proved by
   raising `MAX_BATCH` to 64 — written before the pin so its author meets the constraint instead of
   discovering it. The second assertion uses a LITERAL 2 rather than `crate::gpu::MAXROW`: review
   caught that pinning it to the global would make it fire on exactly the change K3 needs (a
   per-model row count of 1), contradicting the first assertion.
   **Still to do:** the refusal inside the K3 pin, when S3 writes one. The test above is the
   arithmetic; it is not a load-time check.

Tokenizer work defers to S4: **there is no `chat_template`** in `tokenizer_config.json`;
rendering lives in `encoding_k3.py` with an XML-ish `<message role=...>` framing, so the port
hand-transliterates that module. No gate through G3 consumes a template.

### G1a — met when

- `.f4` repack **bit-exact both directions on real tensors**, asserted. **MET 2026-08-10, on a
  ONE-EXPERT sample.** K3 is 1.42 TiB and does not fit here, so layer 1 expert 0 (17,547,264 B) was
  fetched by HTTP Range from the shipped shard and converted: `--verify` reports 0 bytes differ, and
  a second pass in Python — at slot offsets recomputed from the widths alone, so it shares no code
  with the writer — finds all six spans bit-identical, with `w3` in the up slot and `w2` in the
  down. Two runs are byte-identical. `docs/measurement/k3-reference/repack-one-expert.md` carries
  the byte ranges and hashes. **What is NOT covered: 4 of 82,432 experts.** A full-checkpoint pass
  belongs to S4, where the bytes exist.
- The existing GLM (**659.25** GiB) and V4 (**145.97** GiB) artifacts still open **byte- and
  offset-identically**, proven by a test that opens them. **MET 2026-08-11** —
  `tests/artifact_compat.rs` opens both full artifacts (152 layer files, all three routed formats)
  and confronts every file length and the end layers' expert offsets with the geometry derived from
  each model's config. It reads no weight: 805 GiB in 0.1 s, no GPU. It also pins the `has_shared`
  fork — `.f4` has no shared block, `.vq3` and `.i4` do — and `.i4`'s headerlessness, which is the
  case where a stray 4 KiB header would shift every offset in the artifact.
- A config missing or contradicting any load-bearing field **refuses at startup**, proven by
  feeding it one. **MET 2026-08-10** — `mla_use_nope: false` in a hand-written manifest is
  refused by the shipped binary before it reads a dimension. That case is one of the **35** rows
  in `k3_rejects_the_silently_wrong_settings`; `k3_layer_partition_must_be_a_partition` adds 5
  broken layer maps plus a positive control, and `every_k3_field_is_required` covers all 30
  `text_config` fields one at a time plus both arrays inside `linear_attn_config`. Three of
  those gates are red-proved by deliberate injection (a `#[serde(default)]`, an emptied
  `hidden_flags`, and a transposed width pair).
- Byte accounting reproduced **from the artifact**, both halves. **MET 2026-08-11** — the same test
  derives each artifact's total from its config and asserts it against the disk: V4
  147,169,914,880 B routed + 9,557,453,182 B resident = 145.97 GiB; GLM 299,531,812,864 (`.vq3`) +
  391,695,040,512 (`.i4`) = 643.75 GiB routed, 659.25 GiB with everything. **The plan's GLM figure
  was 675 GiB and that was 15.75 GiB high** — corrected above and in `other-models.md`. V4's ~146
  was right, and a first pass that summed only its `.f4` files got 137.06 GiB and nearly reported
  the opposite.

**Three of the four are MET.** The open one is the repack at full scale, which needs the
checkpoint.

### S1b — the gate harness

Owns fixtures and the harness. **Must not touch `gpu.rs`, `attn.rs` or any kernel.**

The reference ships a tiny model with three zero-tolerance gates, per-operator fixtures with
their own tolerances, a synthetic expert shard, and tokenizer goldens — ~9 MB, no checkpoint.
That is far less work than V4's 117 KB hand oracle, but it is **cheaper, not better**: those
goldens are one implementation's output, so a misreading is in the spec, in the goldens, and in
anything checked against them.

**The anchor: run the first-party stack** — `modeling_kimi_linear.py` + pinned fla — at tiny
dimensions, emitting golden activations per module. **Mandatory**, not optional: the
reference's own parity evidence is one position of a junk prompt, the KDA arithmetic lives
only in fla, and item 11 already caught one divergence (the LoRA-norm eps) that fixtures
inherited from the C would have baked in. Record the fla version pin next to the goldens.

> **DONE 2026-08-11.** `tests/k3_anchor_driver.py` runs it, `tests/k3-anchor.sh` reproduces it,
> `tests/k3-anchor-decode-k3-anchor-{1,2}.bin` are the vendored decode goldens (two independent
> weight draws), `tests/k3_anchor.rs` is the gate that
> reads it, and `docs/measurement/k3-reference/anchor.md` is the record — including the four
> declared deviations and why the goldens are vendored rather than regenerated.
>
> **Two things this settled that were guesses here.** *The tiny model is stored in nothing*: the
> weights are generated deterministically from a per-parameter-NAME seed, so there is no fixture
> format to choose and no converter in the loop — a global seed would have made every golden depend
> on module construction order. And *KDA cannot run on CPU at all*: fla's ops are triton kernels,
> and its pure-torch `naive_recurrent_kda` takes **none** of the seven kwargs the model passes
> (`A_log`, `dt_bias`, the qk l2-norm, the beta sigmoid, the gate flag, `safe_gate`,
> `lower_bound`), because all of that moved *inside* the kernel. Substituting it would have meant
> hand-transliterating the exact arithmetic the anchor exists to not transliterate.
>
> **The eps divergence is priced at 2.2e-5 relative** — which is the argument for the anchor made
> concrete: a fixture with any ordinary tolerance would have passed the C reference's value.
>
> **AttnRes had no fixture for a day, and it is S2 item 1.** `_apply_attn_res` reads
> `proj.weight`/`norm.weight` inline and never calls either module, so forward hooks on them fired
> zero times while three comments claimed the fold was captured. The fold is now captured by
> wrapping the reference's free function, and the driver asserts every registered hook fired —
> which caught five more dead hooks immediately. **Anything in this plan that assumes a module hook
> sees an operator should be checked against how the reference actually invokes it.**
>
> **A width that collapses a distinction the real config keeps is not "only widths shrink".** The
> first tiny config made `kv_lora_rank == qk_nope_head_dim`, `2·moe_inter == latent`,
> `hidden == intermediate` and `hidden == KDA projection` — four pairs the real config separates —
> so a port reading the KV latent width off `qk_nope_head_dim`, or the shared expert's width off the
> latent instead of `num_shared_experts · moe_inter` (§1's own trap), produced a **bit-identical**
> fixture. Fixed before any kernel was scored. Every future fixture in this port inherits the rule.

Also unscoped: which converter binary K3 uses. *(The other half of this — what format the tiny
model is stored in — is answered above: none.)*

### G1b — met when

- Every golden has a **recorded defect run** showing it reddens at exactly the cases the defect
  touches and stays green elsewhere. A golden without one does not count.
  > **Met for the anchor's goldens, with one reading fixed 2026-08-11.** "Stays green elsewhere"
  > can only mean **upstream**: the goldens come from one forward pass, so a perturbation at layer
  > 3 reaches layer 92 by construction, and a defect that reddened *nothing* downstream would mean
  > the capture was disconnected. `anchor.md` carries the matrix for all eleven defects.
  >
  > **The green half is now GATED, not read.** Until the same day, `tests/k3-anchor.sh` asserted only
  > that a defect changed *something* — so a regression that broke the localisation would have
  > printed a matrix nobody reads and exited 0. Each defect now declares the layers it must leave
  > bit-identical (`EXPECT_GREEN` in the driver) and `--compare` fails if one of them reddens. That
  > is the load-bearing half of §G rule 1, and it was decoration for a day.
- Defect runs cover a KDA layer, an MLA layer, layer 0, and layer 92.
  > **Met 2026-08-11** — captured layers are 0, 1, 12 (KDA), 3, 91, 92 (MLA), and
  > `DenseMlpGateUpSwap` exists specifically so a defect touches layer 0 alone at first.
- The independent anchor — **S1b's first-party-stack goldens, as defined above** — exists and
  passes. Citing the reference's own validation does not satisfy this; a source attesting to
  itself is not independence.
  > **Met 2026-08-11.** `anchor.md`. **Eleven defects**, including all four of the fla kernel kwargs
  > that exist only inside triton (`use_qk_l2norm_in_kernel`, the −5.0 gate `lower_bound`, the
  > state's axis order, `use_beta_sigmoid_in_kernel`) — the arithmetic this anchor exists for, and
  > un-red-proved until review asked for it. Against §10's twelve traps this covers 6, 8, and trap 4
  > in part.
  >
  > **The per-operator TOLERANCES are measured, 2026-08-11** — `tests/common/k3_tolerance.rs`, which
  > S2's kernel tests and the anchor's own gate share. Each row's policy is *derived* from two
  > measured numbers (the fp32 rounding floor, and the weakest defect targeting that operator) and
  > `tests/k3_anchor.rs` fails if the numbers stop supporting it.
  >
  > **MLA came out exact-only, and it changes an S2/S3 action.** The C reference's LoRA-norm eps moves
  > that operator by 2.22e-5 against its own fp32 rounding floor of 1.70e-5 — a margin of **1.3×**, so
  > no threshold admits a correct kernel and rejects that eps. **The eps must be pinned structurally:
  > read the constant and assert it.** `KimiMLAAttention` constructs both LoRA norms without passing
  > `config.rms_norm_eps`, so 1e-6 is `KimiRMSNorm`'s own default and that is a fact about the source,
  > not about any output. Downstream the same defect sits at 0.3–0.9× the floor — *below* the
  > reference's own rounding error — so no tolerance-based fixture could have caught it anywhere,
  > which is the retrospective case for this anchor being exact bytes. Every other operator has
  > 90,000× to 7M× of room.
  > **The one-draw limit is closed**: two salts are vendored, all eleven defects are scored against
  > both, and salt 2 reproduces salt 1's green cells exactly — so the localisation is a property of
  > the arithmetic, not of the numbers it landed on. Degeneracy is asserted per draw (no routed
  > weight under 5% of the largest, `|beta| < 8`) rather than hoped for.
  >
  > Still open, and disclosed rather than fixed: the anchor is **fp32 while the checkpoint is bf16**.
  > The tolerances above are fp32-vs-fp32 numbers; what bf16 accumulation costs is an S4 question.

## S2 — kernels. Each item gates before the next.

**Order: AttnRes → MLA → latent sandwich → SiTU/MoE → KDA.** Specs and the twelve traps are in
`k3-architecture.md`; **each trap is a G2 defect-run candidate.**

1. **AttnRes** — **DONE 2026-08-11, G2 met for this item.** `kernels/linalg.hip::attn_res` plus
   `launch_attn_res`, scored by `tests/k3_kernels.rs` against all twelve folds of both draws:
   worst **3.08e-7**, against a 7.052e-5 floor and a 7.1e-4 tolerance, several folds bit-exact.
   *(Floor and tolerance restated 2026-08-12: both were one-draw readings. The measured 3.08e-7 is
   unchanged and the tripwire at 10x it is what actually binds — `k3_tolerance.rs` header.)*
   The defect run is the second test — mixing the NORMALISED sources fails the fixture — and the
   kernel is red-proved four ways (uniform weights, no fold in the score, source 0 only, eps
   outside the mean).

   Two things it cost, both recorded in `measurement/k3-reference/anchor.md`: the anchor had
   **no fold weights**, so the operator's inputs did not determine its output and no fixture was
   writeable until `wrap_attn_res` captured `norm.weight * proj.weight` (223 → 235 tensors, both
   goldens re-vendored). And the fixture is ~50x TIGHTER than the operator tolerance, because it
   feeds the kernel the reference's own inputs while the floor was measured on whole-model runs
   carrying upstream drift — so it also carries a regression tripwire at 10x the observed worst,
   marked as not being the contract.

   **Still open, and deliberately not done here:** the `[T][9][7168]` stack and its **prefill**
   sizing (§4d), and the layer loop's push/reset bookkeeping. The kernel takes an assembled stack;
   `every_fold_mixes_the_depth_the_layer_loop_implies` pins the depth each fold should see, but
   nothing yet builds it. That is S3's. Note the tensors ship **BF16** and are named
   `self_attention_res_*`, not `attn_res_*`.
2. **Gated MLA** — **DONE 2026-08-11, G2 met for this item.** `kernels/attn.hip::mha_attend` (dense
   MHA over the per-head k/v `kv_b` expands — NOT the absorbed `mla_latent_attend`, which is V4's
   and shares no arithmetic) plus `sigmoid_gate`, scored by `tests/k3_kernels.rs` at all three MLA
   layers of both draws. Red-proved seven ways.

   **The anchor could score none of it.** `eager_attention_forward` is a free function, so the
   golden held the projections either side of the attention and nothing from within it — the
   192-scale, the still-scored rope dims and causality were all unscoreable. And `o_proj`'s input
   was uncaptured, so trap 10 sat in a gap. Both fixed (`wrap_mla_attention`, plus a
   `register_forward_pre_hook`); goldens re-vendored at **262** tensors. `scaling` is now a captured
   VALUE, which is what lets the fixture see `MlaScaleFromNope` rather than trust a comment.

   **`mla_attend` is its own tolerance row**, floor 4.103e-5 / defect 6.578e-1 / `Rel(4.10e-4)`,
   split from `mla` because the eps that makes `mla` ExactOnly cannot reach an operator fed the
   reference's own q/k/v. Re-measuring `mla` across both draws moved its margin from 1.3x to
   **0.33x** — the eps now sits BELOW its own floor, which is stronger than the original finding.

   **Two blind spots the goldens could not have caught**, both found by red-proofing and both now
   covered by a synthetic sweep: the decode masks are **all zero**, so causality masks nothing and
   a kernel ignoring `mask` stayed green; and the softmax's max-subtraction is unobservable at these
   magnitudes. Plus the width gap — 4 heads of 24/16 against a real 96 of 192/128.
3. **The latent sandwich** — two bf16 trunk GEMVs plus a norm, with the accumulator interception of
   S1a.4. Scored by `tests/k3_kernels.rs`.

   > **CORRECTED 2026-08-12.** This item read "plus `rmsnorm_batch` (`mla.hip:346`, already
   > width-generic)". It is width-generic, and **the width was never the problem.** Its last line is
   > `row[i] = rbf16(w[i] * (row[i] * rs))` — it rounds its store to bf16, because V4's
   > `RMSNorm.forward` stores bf16 and that kernel is V4's. `KimiRMSNorm.forward` is
   > `self.weight * x.to(dtype)`, and against this fp32 anchor `to(dtype)` is a no-op, so the bf16
   > step is arithmetic the reference does not perform: **measured 3.299e-3 against the 6.3e-4
   > tolerance, 11.4× over**. **The kernel is `linalg.hip::rmsnorm_single`** (f32 store, out-of-place),
   > correct at decode's one row. `the_batch_rmsnorm_would_fail_this_fixture` asserts the failure
   > rather than leaving it as a comment, because a claim that one of two interchangeable kernels is
   > wrong is exactly the claim that rots.
   >
   > **Prefill needs a third kernel and does not have one.** `rmsnorm_single` is `dim3(1)` — one
   > statistic over whatever it is handed, which at `T` tokens is `Defect::HeadNormOverAllTokens`
   > and invisible at `T == 1`. `rmsnorm_batch` is row-wise and rounds. K3's latent norm at prefill
   > wants row-wise **and** f32-store, which is neither. Deliberately not written here: S2 is
   > decode-first, and an unused third variant is a variant nothing scores.

   **The projections are `gemm_bf16`, and the anchor cannot score them.** Not for want of captures —
   because the anchor runs fp32 (a declared deviation) while these weights are bf16, so any such
   comparison is dominated by a ~2⁻⁹ quantisation the reference never applied. What is left to check
   is the part that is genuinely the kernel's, the `wave_sum` shuffle ladder re-associating a
   7168-term sum, and `the_trunk_gemv_matches_an_f64_dot_at_k3_widths` scores exactly that at
   `7168→3584` and `3584→7168` against an f64 dot on the same bf16 codes. **That closes half of item
   7's complaint** — `gemm_bf16` was verified only at vocab 1024 / dim 512 and now has a stated
   bound at K3's real trunk widths. Its bound is the test's own and is NOT a `k3_tolerance` row:
   every number in that table derives from the anchor's floor-vs-defect pair, and this one cannot.

   The ordering trap gets a fixture-level twin of `--defect LatentNormAfterUp`
   (`norming_after_the_up_projection_is_a_different_sandwich`), built the same way the defect is —
   the norm weight collapsed to its own mean so it applies at `hidden` width — so it asks about the
   ORDER and not about the values. Synthetic, because the projection weights are deliberately absent
   from the goldens; `anchor.md` carries that argument and the ~4× file growth it declines.

   **The anchor gap this one cost:** `routed_expert_norm` had an output and no input. It is fed
   `moe_infer`'s return, which is a method call rather than a module call, so no forward hook could
   see it — the fourth instance of the pattern `anchor.md` now states as a rule. `wrap_latent_sandwich`
   captures the aggregate and the norm weight by pre-hook; goldens re-vendored at **272** tensors.
4. **SiTU-GLU + fp4 MoE** — §3b. **DONE 2026-08-12, G2 met for both halves.** The split is §3b's own: the activation has
   two call sites, `linalg.hip` for the dense layer 0 and the shared MLP, and `moe.hip` fused inside
   the fp4 expert kernel because there is no separate activation launch on the `.f4` path.

   **4a — `common.hpp::situ_glu` + `linalg.hip::situ_glu_f32`.** The helper lives in `common.hpp`
   for `swiglu_clamped`'s own reason: the two call sites must agree bit for bit and `kernels/` is
   not scanned by jscpd, so a second copy would drift unseen. Scored by `tests/k3_kernels.rs`
   against all six captured MLPs of both draws — worst **1.454e-7**, against a 9.4e-6 tolerance at
   layer 0's dense MLP. Red-proved five ways (capped sigmoid, betas swapped, `up` clamped instead of
   `tanh`'d, plain silu, guard removed), plus real widths, the `|y| <= b1·b2 = 100` bound §8 states,
   and a beta-guard test — the first refusal-code test in this file, the other three items having
   left theirs untested and said so.

   **The fixture needed NO regeneration, and that is worth recording.** `SituAndMul` is an
   `nn.Module`, so `hook_model` had been capturing its output as `<mlp>.act_fn` all along, and its
   input is `torch.cat([gate_proj(x), up_proj(x)])` — both halves separately captured. Four items in
   a row found the anchor could not score their operator; the fifth found it already could. Checking
   what is in the file is cheaper than a 25-minute GPU-locked regeneration.

   **Two limits found while scoring it, both recorded at the tests:**
   * `operator_of` buckets the shared experts as **`moe_route`**, the ROUTER's tolerance, because
     the name prefix does not separate them. A classification artifact, not a judgement; the fix
     belongs with item 6, the other occupant.
   * At those shared experts the capped-sigmoid defect separates by only **4.10e-3 against
     `moe_route`'s 6.0e-4 — 6.8x**, under the 30x `DEFECT_MARGIN` the table requires of a `Rel`
     policy. So **the bucket tolerance could not be relied on to catch that defect there**; the
     fixture's own tripwire catches it by 2,800x. True at the old 2.5e-4 too (16x), so it is a
     property of the shared expert's small `moe_intermediate_size`, not of this round's loosening.

   **4b, still owed — and BIGGER than this item said.** §3b scoped it as "a new `moe_gateup_f4`
   variant", i.e. one activation swap in pass 1. Reading the reference to write it found a second,
   structural difference, in the pass §3b does not mention.

   > **K3 applies the routing weight AFTER the down projection; rivoli's fp4 path folds it in
   > BEFORE.** Found 2026-08-12. `moe_infer` ends
   > `new_x.view(...).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(-1)).sum(dim=1)`
   > (reference `:867-874`), which is §6's `accL[i] += wt[j] * edn[i]` — `edn` is the expert's `w2`
   > output. V4's `Expert.forward` instead does `weights * x` and THEN `x.to(dtype)` in front of
   > `w2`, which is why `moe_gateup_f4_impl` stores `rbf16(sw * w)` and `moe_down_f4_impl`
   > "takes no `wexpert` — one source of truth for the routing, and no way for a mask to disagree
   > with the data".
   >
   > `w2` is linear, so `w2(w·h)` and `w·w2(h)` agree in exact arithmetic and the fold looks free.
   > **It is not free here**, for two reasons that are both in the existing kernels' own comments:
   > there is a **bf16 store between the two** (`rbf16` in pass 1), and pass 2 accumulates in
   > **fixed point** (`MOE_ACC_SHIFT 44`). Rounding `bf16(sw·w)` is not rounding `bf16(sw)` and
   > scaling afterwards. So K3 needs a variant of the DOWN pass too, and pass 1 must round without
   > the weight.
   >
   > **The `w == 0` NaN launder moves with it.** Pass 1 currently enforces `w == 0.0f ? 0.0f : ...`
   > because a `w1` dot at `-inf` gives `silu = -inf · 0 = NaN` and `NaN · 0` is NaN, which
   > `moe_fixed` then turns into a finite extreme — silent corruption of a row that never asked for
   > this expert. With the weight applied in pass 2 the same hazard sits at the same multiply in a
   > different kernel, and SiTU-GLU's own bound (`|y| <= 100`) does not remove it: the dot feeding
   > it is unbounded. Whatever 4b does here has to be argued, not inherited.

   **4b as built:** `moe_gateup_f4_situ`, `moe_down_f4_weighted` and
   `rivoli_moe_expert_range_f4_situ`. Pass 2 is ONE templated body with the existing kernel,
   `WEIGHTED` deciding a single line — `moe_acc_drain`/`moe_acc_drain_to`'s pattern, for its reason:
   one difference the code cannot make visible, two names, both bodies readable side by side. **Two
   launches, not three:** no `act_quant_f8` between the passes, because K3's `w2` takes a plain fp32
   activation (§6's `k3_matmul_mxfp4(edn, act, ...)`) where V4's fp8 `Linear` quantizes its own
   input — quantizing here would add an error the reference does not have, in the one place the
   reference is exact. The group-alignment guard is `F4_GROUP` (32), not the inherited
   `ACT_QUANT_BLOCK` (128); the tighter check would never have fired, since 3584 and 3072 are both 0
   mod 128, so keeping it would have been a constraint nothing measured.

   The routed experts have **no anchor fixture and cannot get one** — `.experts` is unhooked on
   purpose, because `moe_infer` calls only the experts that won tokens, so which modules fire is
   routing-dependent and any defect that moved the routing would change the golden's tensor SET.
   So 4b is scored against a host oracle composed of parts pinned elsewhere:
   `v4oracle::numerics::{e2m1_decode, e8m0_decode}` for the codes, `repack-one-expert.md`'s
   real-byte verification for the layout, and `host_situ` for the activation, which 4a pinned
   against the reference at both draws. **`Oracle::expert` is NOT reusable** — it is V4's, with
   `swiglu_clamped`, V4's three bf16 roundings and V4's weight placement — and parameterising a
   frozen oracle to serve two models is the refactor `common.hpp` warns against, one level up.

   Red-proved six ways, six reds: the weight folded into pass 1, pass 2 rounding AFTER the weight,
   pass 2 dropping the weight, `swiglu_clamped` in place of `situ_glu`, gate/up swapped, the beta
   guard removed. **The second of those is the one this item exists for** — `rbf16(dv)·w` against
   `rbf16(dv·w)` is invisible in exact arithmetic, and the fixture sees it.

   > **How it is scored, because neither obvious bar works.** `common/mod.rs::assert_bitwise` records
   > that a correct wave-reduced kernel differs from its oracle on **~0.08% of bf16 elements at dim
   > 4096** — the f32 and f64 dots land on opposite sides of a bf16 boundary. So a tight bound
   > rejects correct code (two of three cases here are BIT-EXACT and the third is 2.59e-11, which is
   > luck, not a contract), and a loose one sees nothing (one crossing is a whole bf16 ulp, ~3.9e-3,
   > so admitting it admits 3.9e-3 of anything else). The gate is both: **no element differs by more
   > than one bf16 ulp, and no more than `2 + len/100` differ at all.** A pure percentage was tried
   > and FAILED at `expert_in = 64` on 1 element in 64 — a rate bound is unusable at small n, and
   > small n is where the index-error case lives, so the absolute allowance is what keeps that case
   > runnable. At the real widths 2 of 3584 is 0.06%, under the measured crossing rate, so the
   > fraction still binds where it matters.
   >
   > **The real geometry is not run and that is deliberate.** The host oracle is ~220M f64 operations
   > at 3584x3072; on the dev profile this repo prescribes for correctness work, one case took over
   > ten minutes before it was abandoned. Each pass's REDUCTION is at its real depth instead — one
   > case at `expert_in = 3584`, one at `inter = 3072` — because that is what the arithmetic depends
   > on, while the row counts only exercise the grid mapping. Anyone who needs the full shape should
   > reach for `--release`.
5. **KDA** — the new family and the bulk of the stage. Heads are independent and head-parallel is
   bit-identical in the reference, so one block per head maps directly; `S` is 64 KB/head and will
   not sit in LDS, so **fusing the four passes is the HIP win the C does not have.** No chunked
   prefill exists in the reference, and a chunked port must reinstate two correctness conditions
   (`k3-architecture.md` §4).

   **DECOMPOSED 2026-08-12, from what the anchor can and cannot score.** §4's ten steps do not map
   to ten units — fla fuses them into three observable boundaries, and the golden's capture set is
   what decides where the sub-gates go. Enumerated rather than guessed, from the vendored bytes:

   | unit | §4 steps | anchor boundary | fixture today |
   |---|---|---|---|
   | **5a recurrence** | 3-7 | `kda.fused_recurrent_kda.in.{q,k,v,g,beta,A_log,dt_bias,initial_state}` → `.out.{o,state}` | **DONE 2026-08-12** |
   | **5b ShortConv+SiLU** | 2 | `self_attn.{q,k,v}_proj` → `self_attn.{q,k,v}_conv1d.{0,1}` | needs the conv **weights** |
   | **5c fused norm+gate** | 8-9 | `.out.o` + `self_attn.g_proj` → `self_attn.o_norm` | needs **`o_norm.weight`** |
   | **5d projections** | 1, 10 | `self_attn.{q,k,v,b,f_a,f_b,g,o}_proj` | complete, but see the tolerance note |

   **Do 5a FIRST, and not only because the plan said "decode recurrence first".** It is the one unit
   whose fixture is already complete — everything inside fla's kernel is captured on both sides, and
   the four KDA defect runs (`KdaNoQkL2Norm`, `KdaGateLowerBoundOff`, `KdaStateLayout`,
   `KdaBetaSigmoidOutside`) price exactly this boundary's arithmetic, each reddening 16 of layer 0's
   40 tensors while leaving the 24 upstream alone. It is also the largest kernel in the port. Four
   items in a row began with a regeneration; this one does not need one.

   **5a as built: `gated_delta_recurrent_f32`, in a new `kernels/recurrent.hip`.** One block per
   head, thread `t` owning value channel `t` for the whole kernel, which makes the decay fold into
   the `u = Sᵀk` reduction and the rank-one update fold into the `o = Sᵀq` reduction — **two passes
   over `S` instead of the C's four, and no cross-thread reduction in the recurrence at all**
   (each column's two sums are private to its thread; the only block reductions are the two L2
   norms). That is the HIP win this item was predicted to have, and it is bigger than "fuse the four
   passes": the fusion falls out of the thread mapping rather than being arranged.

   Named for the arithmetic, not the block. `kda`/`fused_recurrent_kda` are Kimi's and fla's names
   for it; what it computes is the gated delta rule, and this tree has a rule against a model's name
   on a kernel. The file is named for the family so 5b and 5c land beside it.

   > **THE FINDING: the reference stores the state `[value][key]`, and this kernel deliberately does
   > not.** `transpose_state_layout=True` is in the driver's kwargs and names the choice, but the
   > state is SQUARE at the tiny widths (32) and at the real ones (128), so no shape assertion can
   > see it and §4's `S[i][j]` is prose either way. Scoring both interpretations of the anchor's own
   > `initial_state` settles it: with the transpose the recurrence agrees to **2.5e-7**, without it
   > to **2.2e-1 – 5.6e-1**, unanimously across three layers and both draws.
   >
   > The port keeps `[key][value]` regardless, and pays nothing for it: rivoli's state starts at zero
   > and never leaves the device, so nothing forces the reference's axis order on it, while
   > `S[i*d + t]` is what makes consecutive threads read consecutive addresses. The transpose is a
   > FIXTURE boundary — three lines, once per case — and `KdaStateLayout`'s red-proof is what shows
   > the fixture can tell the two apart. **This is the first item where a reference convention was
   > measured and then declined**, rather than measured and adopted.

   Everything else in §4 steps 3-7 confirmed exactly as written, by the same six-site sweep: the L2
   norm on q and k only with `eps` on the SUM, `beta = sigmoid(b_proj)`, `alpha =
   exp(lb·sigmoid(exp(A_log)·(g + dt_bias)))` with the bound MULTIPLYING the sigmoid, the decay on
   the KEY axis, the `d^-0.5` on q alone, and `o` read from the UPDATED state. Worth stating because
   the alternative to each was tried in the same sweep and each lands two to six orders away.

   Red-proved against the DEVICE six ways, six reds — the four `Kda*` defect runs plus the two steps
   no defect covers (`o` read before the update, and the `d^-0.5`). The host oracle carries the same
   five variants, so the fixture's sensitivity and its connectedness are proved separately.

   **What 5a does NOT cover, and it is the same gap AttnRes has**: the fixture runs ONE step from a
   supplied `initial_state`. That 69 states stay alive across a sequence and are never reset
   mid-decode is S3's, and no anchor capture can gate it — the golden holds one step of one decode.

   **The launcher takes no token count, and that is a scope cut with a price.** It is one step, so a
   KDA prefill is `T` launches of it — 69 layers x T, each a 2-pass sweep of the whole state. That is
   correct and it is what the reference's `fused_recurrent` path does at `q_len == 1`; it is also why
   first-party defaults to `chunk_kda` for prefill. Porting the chunked form is the throughput item
   this section's header flags, and it must reinstate the two conditions §4 names.

   **5b and 5c each need ONE more capture, and both are parameters.** Same shape of gap as the
   AttnRes fold and the latent norm: an input and an output do not determine an operator when a
   weight sits between them. `ShortConvolution`'s depthwise taps are `[channels][k]` (§4 step 2 —
   oldest→newest, `w[k-1]` on the current token, SiLU fused into the output) and
   `FusedRMSNormGated`'s is `[head_dim]`. At the tiny widths that is 3 x [128][4] + [32] per KDA
   layer over three captured KDA layers (0, 1, 12 — the other three are MLA), about 19 KB per
   golden. **Batch them into one regeneration**, and take 5b's conv CACHE semantics with it: `.1` is
   the returned history, and at decode the history is what makes the conv stateful.

   > **5c is where trap 10 finally becomes checkable, and only halfway.** `o_norm` is
   > `FusedRMSNormGated(head_dim, activation='sigmoid')` called as `o_norm(o, g)`, so norm-then-gate
   > is ONE module and the intermediate is unobservable — item 2 recorded that and deferred it here.
   > What 5c can prove is the composition end to end (both inputs and the output are in the file);
   > what it cannot prove is the ORDER within the fusion. That has to come from the reference's
   > source, as the MLA side's did. Note also that `o_norm`'s output and `o_proj.in_gated` are the
   > same tensor under two names — a free cross-check that the pre-hook and the module hook agree.

   **5d has no tolerance row, and that is a documented GAP rather than a decision.** `operator_of`
   buckets a KDA layer's projections as `kda_trunk`, one of the four buckets `anchor.md` says "S2
   must not score against a threshold — compare them exactly, or measure the floor the same way and
   add a row". The fp64 island runs from item 3 already contain it: **7.680e-6 on draw 1, 2.292e-5
   on draw 2**, so the floor is 2.292e-5 and a `Rel(2.3e-4)` row is one `--by-operator` away from
   being defensible. Its weakest targeting defect is the open question — no defect in the eleven
   targets a KDA projection — so 5d may have to stay exact-only for want of a ceiling, which would
   be the second `ExactOnly` in the table and for a different reason than `mla`'s.
6. **Router** — sigmoid, bias on selection only, weights from the **unbiased** score,
   renormalised over the 16.
7. **Trunk GEMV, for speed.** `gemm_bf16` is correct and carries S2/S3 unchanged. Its inner loop is
   one wave per output *element* with scalar u16 loads; at 108.81 GB/token that loop is the decode.

   > **UPDATED 2026-08-12 by item 3.** This said it is "verified only at vocab 1024 / dim 512
   > (`f4gpu.rs:2253` says so outright), so it is an oracle that itself needs a tolerance before it
   > can settle one". The accuracy half is now answered:
   > `the_trunk_gemv_matches_an_f64_dot_at_k3_widths` scores it at `7168→3584` and `3584→7168`
   > against an f64 dot on the same bf16 codes, so it has a stated bound at the widths K3 runs it
   > at. **The speed half stands unchanged** — that is what this item is for.

### G2 — met when

Each item passes **its operator fixture and its defect run**, in order, before the next begins.
Kernel-by-kernel, not stage-at-the-end.

## S3 — layer loop, first decode

A `src/k3gpu.rs` and a `main.rs` K3 branch. Name **every deviation from the reference at its
call site** — V4's three named deviations are what let its reviews catch two criticals before
the GPU did. §3c's KV deviation is the first entry.

**Do not write `k3gpu.rs` by mirroring `f4gpu.rs`** — `build.rs` runs `jscpd --min-tokens 15`
and panics on any clone, at zero budget. Factor when it fires; do not pre-emptively design a
three-model skeleton.

### G3 — met when

- The three zero-tolerance gates pass on the tiny model: teacher forcing, greedy decode,
  incremental decode with KV cache.
- **KDA state carries across positions**, proven by a deliberately broken state advance going
  red.
- The incremental gate is run **first with the fp8 KV quantization disabled**, so a mismatch is
  attributable to the recurrence rather than to §3c's cache-format deviation.

## S4 — real weights on `/swarm`. Not blocked.

1. Convert the full artifact; reproduce §4's byte accounting from it, **both halves**.
2. Tokenizer and the hand-ported `encode_k3`, with a byte-level gate.
3. A bounded greedy run. **This is the only check that exercises §2's traps against trained
   weights** — the tiny model may collapse the very distinctions in question.

### G4 — met when

Byte accounting matches; the greedy run produces **coherent, on-task text read by a human**
(not `distinct`, not longest-repeated-block); output deterministic across two runs.

## S5 — throughput and residency. Blocked on ~200 GiB of NVMe.

1. Register **~0.48–0.57 tok/s** (§4c) before the first run, from a budget with §4d's claimants
   subtracted. Score **total** bytes/token, and record the achieved QD.
2. Pin the trunk, on §4a's byte argument — not on the reference's `s/tok`, which its own 33.1%
   noise floor disqualifies.
3. Measure what Belady, the residency policies and prefetch do against a top-16-of-896
   distribution under budget competition. Reproducing the reference's *sweep* would need
   partial trunk residency, which rivoli cannot express (`DeviceTier` is all-or-nothing) —
   restate the question as the one this engine can ask.

### G5 — met when

Throughput is inside the registered band, or the miss is explained and recorded; and **output
is byte-identical to G4** at the same prompt and settings.

## Standing rules

`CLAUDE.md` § "Measurement discipline" and § "Build and test" apply in full — dev profile for
development, `-- --test-threads=1` on device suites, `flock` and a contention witness per arm,
no `cargo build` between arms, instruments behind a feature *and* a flag, jscpd is a build
error and clippy-green is not duplication-green, and the feature union for anything touching
`telemetry.rs`/`eval.rs`/`gpu.rs`. **CI has no `rocm` arm and no GPU arm**, which is why the
gates above are the whole safety net.
