---
scope: k3
status: live
verdict: The implementation plan for Kimi-K3, a required capability. Six stages behind six correctness gates; a gate is MET or NOT MET and must be proven able to go red before it is trusted green. S0 IS DONE and G0 IS MET (2026-08-10; the corrections it produced are recorded inline in reference/k3-architecture.md, which is now first-party-verified for everything except the KDA inner arithmetic and the MXFP4 unpack — those live in fla-core and compressed-tensors, so S1b's mandatory anchor RUNS the first-party stack at tiny dims and emits goldens). Traffic is 25.83 GB/token of experts (first-party confirmed) plus 108.81 GB of trunk when not resident; predicted ~0.48-0.57 tok/s with the resident set held, from rivoli's measured 12.39-14.76 GB/s at the expert-read shape — an earlier 0.27 came from a dd QD1 figure, the same instrument-class error found in the reference's own 3.2. The memory budget is the binding constraint (~5.7 GB residual at --max-mem 115; the resident set does not fit on the auto path) and nothing converts until SafeWriter streams. Only the 69 KDA layers are a wholly new kernel family, plus AttnRes. S1-S3 run on fixtures; S4 converts and decodes real weights on /swarm; S5 needs ~200 GiB more NVMe.
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
  0.1 s). What remains of S1a: the latent-wide accumulator drain kernel (item 4), the e8m0 `0xff`
  decision (item 2), and the K3 pin's `MAX_BATCH` check (item 7).

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
| `num_shared_experts` | 2 | one **fused** MLP on disk, §3 |
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
2. **Settle e8m0 `0xff`.** *(Tier 1.)* The reference maps 255 → zero; rivoli's `e8m0f` returns
   a quiet NaN and `quant.rs:748` **bails**. Item 11 settles the semantics for free; only the
   *presence* question needs bytes. Host and device must change together or the divergence is
   silent — `moe_fixed`'s clamp launders NaN into a finite ±2^14.
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
   **Still to do:** the K3 pin's own `top_k * rows + n_shared <= MAX_BATCH` check —
   `pin.rs:409` exists only on GLM's path, and there is no K3 pin yet.

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

Also unscoped: what format the tiny model is stored in (rivoli's converters read HF
safetensors; a C fixture may be its own layout), and which converter binary K3 uses.

### G1b — met when

- Every golden has a **recorded defect run** showing it reddens at exactly the cases the defect
  touches and stays green elsewhere. A golden without one does not count.
- Defect runs cover a KDA layer, an MLA layer, layer 0, and layer 92.
- The independent anchor — **S1b's first-party-stack goldens, as defined above** — exists and
  passes. Citing the reference's own validation does not satisfy this; a source attesting to
  itself is not independence.

## S2 — kernels. Each item gates before the next.

**Order: AttnRes → MLA → latent sandwich → SiTU/MoE → KDA.** Specs and the twelve traps are in
`k3-architecture.md`; **each trap is a G2 defect-run candidate.**

1. **AttnRes** — a ≤9-source softmax mixture, twice per layer plus once model-level.
   Arithmetically trivial; the work is the `[T][9][7168]` stack and its **prefill** sizing
   (§4d). First, because it is structural. Note the tensors ship **BF16** and are named
   `self_attention_res_*`, not `attn_res_*`.
2. **Gated MLA** — RoPE removed but the 64 rope dims **still cached and still scored**, softmax
   scale over **192**, output gate **before `o_proj` with no norm**.
3. **The latent sandwich** — two bf16 trunk GEMVs plus `rmsnorm_batch` (`mla.hip:346`, already width-generic),
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
7. **Trunk GEMV, for speed.** `gemm_bf16` is correct and carries S2/S3 unchanged — but
   it is **verified only at vocab 1024 / dim 512** (`f4gpu.rs:2253` says so outright), so it is
   an oracle that itself needs a tolerance before it can settle one. Its inner loop is one wave
   per output *element* with scalar u16 loads; at 108.81 GB/token that loop is the decode.

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
