---
scope: k3
status: live
verdict: The implementation plan for Kimi-K3, a required capability. Six stages behind six correctness gates; a gate is MET or NOT MET, and must be proven able to go red before it is trusted green. S0 IS DONE and G0 is MET (2026-08-10, third pass — reopened twice, both times by the same lesson: "no checkpoint download" never excluded metadata, and the checkpoint's ~156 KB of first-party modeling code was metadata too). Item 11 verified the architecture doc against modeling_kimi_linear.py: MLA, AttnRes, MoE, router and SiTU-GLU confirmed line-by-line; the kda_layers guidance was INVERTED (each implementation consumes the opposite array — assert the partition); A_log ships [128] on disk against the modeling code's own [96]; and the C reference itself DIVERGES from first-party on MLA's LoRA-norm eps (1e-6 vs 1e-5). The 4 KDA-arithmetic traps live in fla-core, which first-party delegates to, so S1b's mandatory anchor RUNS the first-party stack at tiny dims and emits goldens. The index already confirmed the structure first-party: latent sandwich 7168->3584->7168, MLA at zero-based 3,7,...,87,91,92 with the last two ADJACENT, ONE fused shared MLP [7168,6144] BF16, per-expert bytes exactly 17,547,264, trunk 108.81 GB bf16 (113.49 with embed and lm_head), index total 1.4196 TiB, and every tensor family maps to a documented section. TWO BANDWIDTH ERRORS, both ours and both now fixed: the reference's 3.2 GB/s is a dd QD1 figure, and so is rivoli's own 7.0 — rivoli's probes measure 12.39-14.76 GB/s at the expert-read shape, so the prediction moves from ~0.27 to ~0.48-0.57 tok/s. The trunk IS re-read every token (93 binds), but that is confirmed by the bind count, NOT by timing: the reference's device delivered 2709-5874 MB/s on bit-identical work, a 2.17x spread with a 33.1% replication noise floor, so its ladder cannot discriminate the model either way. Reuse is wider than first thought (a bf16 GEMV exists, resident.safetensors takes Bf16, the .f4 shared-block work is void, the fp4 kernels are width-parametric) and one Tier-1 blocker remains: SafeWriter buffers every resident tensor in host RAM, so nothing converts until it streams. Only the 69 KDA layers are a wholly new kernel family, plus Block Attention Residuals which the plan did not know existed.
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
- **G0 is MET** (2026-08-10, third pass). Item 11 read the first-party modeling code raw:
  8 of 12 traps confirmed, 2 doc claims corrected (`kda_layers` was inverted; `A_log` ships
  [128] against the modeling code's own [96]), and the 4 KDA-arithmetic traps are attestable
  only against **fla-core**, which first-party delegates to — S1b's anchor is defined to cover
  exactly that.
- **Only the 69 KDA layers are a wholly new kernel family**, plus AttnRes. The 24
  full-attention layers are MLA + q-LoRA, GLM's own family — but rivoli caches the **fp8
  latent** where the reference caches expanded fp32, a deviation under G3's zero-tolerance
  gate and a ~170× smaller KV budget. §3.
- **Traffic is 25.83 GB/token of experts** (first-party confirmed), **plus 108.81 GB of trunk
  when it is not resident.** §4.
- **Predicted ~0.48–0.57 tok/s** with the resident set held. §4c. This replaces an earlier
  0.27 that was computed from a queue-depth-1 disk figure.
- **The memory budget is the binding constraint.** At `--max-mem 115` the 113.49 GB resident
  set leaves ≈5.7 GB; on the **auto** path the budget is `MemAvailable − 16 GiB` ≈ 107 GB and
  **the resident set does not fit at all**. §4d.
- **Nothing converts until `SafeWriter` streams.** §S1a.

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

| field | value | consequence |
|---|---|---|
| `architectures` / `model_type` | top-level `KimiK3ForConditionalGeneration` / `kimi_k3`; **nested `text_config` says `KimiLinearForCausalLM` / `kimi_linear`** | recognise on the **top** level and assert the nested pair as a secondary check — S1a descends into `text_config`, so a recogniser that descends first will refuse the real checkpoint |
| `num_hidden_layers` | 93 | 1 dense + 92 MoE |
| `linear_attn_config` | dict: `full_attn_layers` (24, one-based) + `kda_layers` (69) + 5 scalars | 24 + 69 = 93. §2 |
| `num_experts` / `num_experts_per_token` | 896 / **16** | the traffic figure, §4 |
| `num_shared_experts` | 2 | one **fused** MLP on disk, §3 |
| `routed_expert_hidden_size` | **3584** | the latent, **not** `hidden_size` 7168. §2 |
| `moe_intermediate_size` | 3072 | |
| `hidden_size` / `num_attention_heads` | 7168 / 96 | |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | `kernels/attn.hip:293` needs `kvl % 128 == 0` and `kvl <= MLA_ACC_REGS*SUBW`; 512 passes both |
| `mla_use_nope` | **true** | assert this **positively**; §3e |
| `mla_use_output_gate`, `latent_moe_use_norm`, `moe_renormalize` | true | each an explicit `K3Config::validate` assertion, never a defaulted field |
| `num_expert_group` / `topk_group` / `topk_method` | 1 / 1 / `noaux_tc` | grouped routing is **degenerate, not absent** — assert both are 1 and refuse otherwise. `noaux_tc` is the first-party name for bias-on-selection-only |
| `activation_situ_beta` / `_linear_beta` | 4.0 / 25.0 | SiTU-GLU, fused into the fp4 kernel — §3b |
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
| bf16 trunk GEMV | **exists** — `v4_dense_gemm_bf16` (`v4compress.hip:82`), already V4's lm_head. A *performance* problem, not a new family |
| `resident.safetensors` bf16 | **already accepted** — `Dtype::Bf16`, `format.rs:99` |
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
**fp8 latent** (`gpu.rs:943`, `mla_absorb_fp8`) at ~13.8 KB/pos across 24 layers; the reference
caches **expanded fp32** per-head k/v at 2.37 MB/pos. ~170× smaller, and a different
algorithm. Two consequences: the budget line is 170× smaller than a reading of the reference
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
17.55 MB) measure **12.39 GB/s at QD1 and 14.76 at QD16** (`benchmarks.md:2430`). The MoE layer
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
for it"). Resident set 113.49 GB, `DeviceTier::HEADROOM` 4 GiB → **≈5.7 GB residual**, against:

| claimant | size |
|---|---|
| KDA state | 464.6 MB (69 layers) or 626 MB as the reference allocates |
| AttnRes stack, `[T][9][7168]` fp32 | **T × 258 KB** — 1.06 GB at 4k, 2.11 at 8k, 4.2 at 16k |
| MLA KV, rivoli's fp8 latent | ~13.8 KB/pos × 24 layers ≈ 129 MB at 8k |
| `RoutedPool` one-batch floor | 281 MB (16 × 17.55 MB) |
| residual stream, scratch, io_uring buffers | unpriced |

**Three claimants, one budget, and AttnRes at prefill is the largest.** On the **auto** path
(`MemAvailable − OS_RESERVE` ≈ 107 GB) the resident set does not fit at all. S5's band must
come from a budget with these subtracted.

## 5. Capacity

```
experts   2,722,740,830,208 params × 0.53125 B = 1.4465 TB
resident set (trunk 108.81 + embed/lm_head 4.70) = 113.49 GB
index metadata.total_size                        = 1,560,860,324,864 B = 1.4196 TiB
```

Pool measured 2026-08-09: **1.69 TiB total, 431.72 GiB free**, GLM at 675 GiB, V4 at ~146 GiB.
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

### G0 — **MET 2026-08-10**, on the third pass.

> Item 11 ran: the first-party modeling code was read raw and every claim in
> `k3-architecture.md` checked against it. **8 of 12 traps first-party confirmed; 2 doc claims
> corrected; the 4 KDA-arithmetic traps are NOT attestable from the checkpoint at all** — its
> own `modeling_kimi_linear.py` delegates them to the external `fla-core` library
> (`chunk_kda` / `fused_recurrent_kda` / `ShortConvolution` / `FusedRMSNormGated`). That is a
> recorded answer with a source, not an unknown: the truth for those four lives in fla, and
> S1b's anchor is now defined to cover exactly that gap.

| # | question | answer | source |
|---|---|---|---|
| 1 | 7168→3584 | Latent sandwich. `routed_expert_down_proj` **[3584,7168] BF16**, `up` **[7168,3584]**, `norm` **[3584]**; experts `w1` U8 **[3072,1792]** nibble-packed, `w2` U8 **[3584,1536]** | index + shard header |
| 2 | layer map | MLA = layers carrying `kv_a_proj_with_mqa`: zero-based **[3,7,…,87,91,92]**, n=24; KDA n=69; disjoint, union 0..92 | index (weights) + `config.json` |
| 3 | trunk dtype | **BF16.** Trunk 108.81 GB; +4.70 embed/lm_head = 113.49 resident; index total 1.4196 TiB; non-expert 114.40 GB incl. vision | index + header |
| 4 | shared experts | **ONE fused MLP per layer**, `down_proj` **[7168,6144] BF16**. Not two | index + header |
| 5a | expert byte layout | U8 packed + U8 group-32 scales reconstruct to **17,547,264 B/expert** exactly | header |
| 5b | nibble order, `sb==255` | Low nibble = even; 255 → zero. **THIRD-PARTY, unconfirmed** — a safetensors header cannot express nibble order | `k3_ops.c` only |
| 6 | vocab | 163,584 BPE + **256 reserved ids 163584–163839** (16 defined) = 163,840 | `tokenizer_config.json` |
| 7 | trunk re-read | **Confirmed by bind count** (93/token), **not by timing** — §4b | `k3_trunk.c:361` |
| 8 | pin the reference | Ladder, split sweep, `environment.txt`, `replication.tsv` vendored | — |
| 9 | correct `other-models.md` §2 | done, dated note in place | — |
| 10 | first-party index | **Done.** 497,220 tensors → 50 families, **every text-side family maps to a documented section**; `g_proj`/`o_proj` on all 93 (both families gate), `o_norm` on 69 (KDA only — confirming MLA gates without a norm), `mlp.*_proj` on exactly 1. **No tensor in the checkpoint is unexplained** | index |
| **11** | **first-party modeling code** | **DONE 2026-08-10** — results below | `modeling_kimi_linear.py` @ HF, read raw |

**Item 11 — results.** ~156 KB of first-party source (`modeling_kimi_linear.py` 51.5 KB,
`configuration_kimi_k3.py`, `encoding_k3.py`, `tokenization_kimi.py`), admitted by the same
"metadata, not weights" argument as the index, verified line-by-line against
`k3-architecture.md`.

**First-party CONFIRMS, verbatim:** MLA in full (scale `192^-0.5` at M:357; `assert
self.use_nope` and `rotary_emb = None` at M:396/403; rope dims scored unrotated; `kv_a_norm`
latent-only; gate-then-`o_proj` with no norm at M:470-473; the cache holds expanded per-head
k/v). AttnRes in full (**zero-based** `layer_idx % block_size == 0`; push-then-clear;
softmax over the raw sources at M:1080-1087; the model-level third aggregation at M:1215).
MoE in full (route-before-down-projection; RMSNorm the aggregate; the shared MLP is **one
`KimiMLP` with `intermediate = moe_inter × n_shared`** at M:798-801, applied to the original
input, unweighted, after up-projection). Router in full (bias to `scores_for_choice` only at
M:723; weights gathered from the **unbiased** sigmoid at M:750; renorm over the 16 with
`+1e-20`; the `routed_scaling_factor` multiply kept). SiTU-GLU exactly as specified, sigmoid
on the uncapped gate, computed at f32.

**First-party CONTRADICTS, two corrections applied to the docs:**

- **`kda_layers` guidance was inverted.** The C reference consumes only `full_attn_layers`;
  first-party `is_kda_layer` (C:152) consumes **only `kda_layers`** and derives MLA as the
  complement — the exact mirror image. Neither array is "the derived one". The port must
  **assert the partition**: both present, disjoint, union = 1..93, counts 69/24.
- **`A_log`: the checkpoint disagrees with its own modeling code.** `modeling_kimi_linear.py`
  declares it `[num_heads]` = **[96]** (M:520); the shard header ships **F32 [128]**. The C
  reference — accept either length, use the first 96 per head — matches the *disk*, which is
  what the converter reads. Accept [128], slice to [96], assert the rest.

**First-party CANNOT ATTEST (4 of 12 traps): the KDA inner arithmetic.** The forward
delegates to `fla-core` with the whole contract in kwargs: `use_qk_l2norm_in_kernel`,
`use_gate_in_kernel`, `use_beta_sigmoid_in_kernel`, `safe_gate`, `lower_bound`,
`transpose_state_layout=True` (M:609-645). The decay formula, recurrence order,
`a*(z+dt_bias)` grouping and the `d_k^-0.5` q-scale live in fla's `chunk_kda` /
`fused_recurrent_kda`. The plumbing around them **is** confirmed (per-head `A_log`, per-channel
`dt_bias`, the shared `f_a→f_b` pair, per-head `beta`, three `ShortConvolution(k=4,
activation='silu')`, norm-then-gate-then-project via `FusedRMSNormGated`).

**New numerics findings, recorded in `k3-architecture.md`:**

- **The C reference DIVERGES from first-party on MLA's two LoRA norms**: first-party
  `q_a_layernorm`/`kv_a_layernorm` take `KimiRMSNorm`'s default **eps 1e-6** (M:368/383,
  no eps argument), while the C passes `rms_eps` = 1e-5. Small, real, and exactly the class
  item 11 exists to catch.
- `KimiRMSNorm` multiplies the weight **in bf16 after the cast** (M:232-236); AttnRes casts
  its mixed output back to bf16 per call. The C holds fp32 throughout. Tolerance notes.
- **Chunked prefill is the first-party default** (`self.mode = "chunk"`, M:481);
  `fused_recurrent` only at cached q_len == 1.
- MXFP4 dequant appears **nowhere in the shipped Python** — it lives in the quantization
  library named by `quantization_config` (compressed-tensors). Item 5b's nibble order and
  `sb == 255` stay third-party until checked against that library's unpack or real scale
  bytes (S1a item 2).
- `moe_layer_freq` (default 1) participates in the dense-vs-MoE choice; `use_full_rank_gate`
  and `mla_use_output_gate` **default False** in code and must be asserted true from config.

## S1 — foundation. No GPU.

### S1a — artifact: config, `Arch`, naming, `.f4`

1. **`SafeWriter` must stream.** *(Tier 1, blocks everything.)* `format.rs:148-154` holds every
   resident tensor as `Vec<u8>` until `write`, sized for a ~10 GiB set. 113.49 GB does not fit.
   Two-pass: header and offsets, then stream bytes. `write_atomic` (`:703`) needs the same, and
   so does the routed path — `convert_v4.rs:298` buffers a whole layer, which for K3 is
   896 × 17.55 MB = **15.7 GB**.
2. **Settle e8m0 `0xff`.** *(Tier 1.)* The reference maps 255 → zero; rivoli's `e8m0f` returns
   a quiet NaN and `quant.rs:748` **bails**. Item 11 settles the semantics for free; only the
   *presence* question needs bytes. Host and device must change together or the divergence is
   silent — `moe_fixed`'s clamp launders NaN into a finite ±2^14.
3. **Thread `moe_latent` (3584) separately from `hidden` (7168).** *(Tier 1.)* The fp4 kernels
   already take `hidden`/`inter` as runtime arguments (`moe.hip:300`), and `ACT_QUANT_BLOCK=128`
   divides both 3584 and 3072 — so the work is entirely in what Rust binds. Loader sites:
   `pin.rs:870` (`SetDims::new`, and a second at `:381`), `quant.rs:184`/`:777`/`:1037`,
   `format.rs:522`/`:1228`, `model.rs:147`. Decode sites, where a wrong value passes every
   length check: `v4gpu.rs:1540` (dispatch), `:1111` (the accumulator must be latent-wide).
   **`v4gpu.rs:1789`/`:1843` are the shared expert and must stay FULL width** — they are not
   latent sites, and listing them as such points at the substitution that breaks all 92 layers.
4. **The MoE accumulator drains into the residual.** `moe_acc_drain` fuses de-fixed-point into
   the residual add at one width; K3 needs the aggregate intercepted **in latent space** for
   the RMSNorm and up-projection first. That is a new drain-to-buffer kernel with its own
   launcher, guards and `tests/kernel_coverage.rs` oracle.
5. **Converter naming.** Expert tensors are `.weight_packed` / `.weight_scale`
   (compressed-tensors), *not* `.weight` / `.weight_scale_inv`. And **`quantization_config`
   mis-declares its own scope**: `targets: ["Linear"]` with an `ignore` list that omits
   `routed_expert_{down,up}_proj` and `block_sparse_moe.gate.weight`, all three of which ship
   **BF16** on disk. Drive off the presence of `.weight_packed`; do not trust the config.
   Skip `vision_tower` and `mm_projector` explicitly.
6. `.f4` repack, `Arch::KimiK3` plumbing (`arch.rs` six arms + a both-directions recogniser
   test, `K3Config` + `impl ArchConfig`, `main.rs:979` dispatch + `run_k3`, `lib.rs`,
   `src/bin/convert_k3.rs`). **`run_k3` must hand-write the `--port` and attention-flag
   refusals** — those are bespoke bails in `run_v4` (`main.rs:729`), not matches, so omitting
   them compiles clean and silently accepts the flags.
7. Assert the config scalars of §1 rather than defaulting them, and write the K3 pin's own
   `top_k * rows + n_shared <= MAX_BATCH` check — `pin.rs:341` exists only on GLM's path.

Tokenizer work defers to S4: **there is no `chat_template`** in `tokenizer_config.json`;
rendering lives in `encoding_k3.py` with an XML-ish `<message role=...>` framing, so the port
hand-transliterates that module. No gate through G3 consumes a template.

### G1a — met when

- `.f4` repack **bit-exact both directions on real tensors**, asserted.
- The existing GLM (675 GiB) and V4 (~146 GiB) artifacts still open **byte- and
  offset-identically**, proven by a test that opens them.
- A config missing or contradicting any load-bearing field **refuses at startup**, proven by
  feeding it one.
- Byte accounting reproduced **from the artifact**, both halves.

### S1b — the gate harness

Owns fixtures and the harness. **Must not touch `gpu.rs`, `attn.rs` or any kernel.**

The reference ships a tiny model with three zero-tolerance gates, per-operator fixtures with
their own tolerances, a synthetic expert shard, and tokenizer goldens — ~9 MB, no checkpoint.
That is far less work than V4's 117 KB hand oracle, but it is **cheaper, not better**: those
goldens are one implementation's output, so a misreading is in the spec, in the goldens, and in
anything checked against them.

**Item 11 changed the answer here, twice.** The anchor is **mandatory** (the reference's own
parity evidence is one position of a junk prompt), and it cannot be a static read: the KDA
arithmetic lives in **fla-core**, not in the checkpoint's Python. So the anchor is **run the
first-party stack** — `modeling_kimi_linear.py` + pinned fla — at tiny dimensions, and emit
golden activations per module. That covers the four traps no document can attest, catches the
eps-1e-6 class of divergence item 11 already found in the C reference, and replaces both the
hand transliteration and the trust in the reference's fixtures. Record the fla version pin
next to the goldens; fla is the one dependency of record.

Also unscoped: what format the tiny model is stored in (rivoli's converters read HF
safetensors; a C fixture may be its own layout), and which converter binary K3 uses.

### G1b — met when

- Every golden has a **recorded defect run** showing it reddens at exactly the cases the defect
  touches and stays green elsewhere. A golden without one does not count.
- Defect runs cover a KDA layer, an MLA layer, layer 0, and layer 92.
- The independent anchor exists and passes. **Citing the reference's own validation does not
  satisfy this** — a source attesting to itself is not independence.

## S2 — kernels. Each item gates before the next.

**Order: AttnRes → MLA → latent sandwich → SiTU/MoE → KDA.** Specs and the twelve traps are in
`k3-architecture.md`; **each trap is a G2 defect-run candidate.**

1. **AttnRes** — a ≤9-source softmax mixture, twice per layer plus once model-level.
   Arithmetically trivial; the work is the `[T][9][7168]` stack and its **prefill** sizing
   (§4d). First, because it is structural. Note the tensors ship **BF16** and are named
   `self_attention_res_*`, not `attn_res_*`.
2. **Gated MLA** — RoPE removed but the 64 rope dims **still cached and still scored**, softmax
   scale over **192**, output gate **before `o_proj` with no norm**.
3. **The latent sandwich** — two bf16 trunk GEMVs plus `v4_rmsnorm` (already width-generic),
   with the accumulator interception of S1a.4.
4. **SiTU-GLU + fp4 MoE** — §3b.
5. **KDA** — the new family and the bulk of the stage. **Decompose into units with their own
   G2 sub-gates.** Heads are independent and head-parallel is bit-identical in the reference,
   so one block per head maps directly; `S` is 64 KB/head and will not sit in LDS, so **fusing
   the four passes is the HIP win the C does not have.** Decode recurrence first — no chunked
   prefill exists in the reference, and a chunked port must reinstate two correctness
   conditions (`k3-architecture.md` §4).
6. **Router** — sigmoid, bias on selection only, weights from the **unbiased** score,
   renormalised over the 16.
7. **Trunk GEMV, for speed.** `v4_dense_gemm_bf16` is correct and carries S2/S3 unchanged — but
   it is **verified only at vocab 1024 / dim 512** (`v4gpu.rs:2207` says so outright), so it is
   an oracle that itself needs a tolerance before it can settle one. Its inner loop is one wave
   per output *element* with scalar u16 loads; at 108.81 GB/token that loop is the decode.

### G2 — met when

Each item passes **its operator fixture and its defect run**, in order, before the next begins.
Kernel-by-kernel, not stage-at-the-end.

## S3 — layer loop, first decode

A `src/k3gpu.rs` and a `main.rs` K3 branch. Name **every deviation from the reference at its
call site** — V4's three named deviations are what let its reviews catch two criticals before
the GPU did. §3c's KV deviation is the first entry.

**Do not write `k3gpu.rs` by mirroring `v4gpu.rs`** — `build.rs` runs `jscpd --min-tokens 15`
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
