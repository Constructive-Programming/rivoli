---
scope: k3
status: live
verdict: The implementation plan for Kimi-K3 — a required capability, not a candidate. Six stages behind six correctness gates; every gate is a hard stop that must be MET before the next stage begins, and every gate must be proven able to go red before it is trusted green. S0 IS DONE (2026-08-09, reference pinned at ff11dce) and G0 is MET but for one item; the extracted forward pass is now reference/k3-architecture.md. S0 overturned four of this plan's own claims: a bf16 GEMV ALREADY EXISTS (v4_dense_gemm_bf16, already V4's lm_head), resident.safetensors already accepts Bf16, the .f4 shared-block work is VOID because K3's two shared experts are ONE fused bf16 resident MLP, and the fp4 kernels are already width-parametric so the latent is a Rust-binding change. It added one Tier-1 blocker the plan never named: SafeWriter holds every resident tensor in host RAM until write, and the 113.49 GB bf16 trunk does not fit. 25.83 GB/token STANDS — the width resolved to a latent sandwich (7168 down to 3584, experts at 3584, up to 7168), refuting the 34.4 alternative. Newly discovered and unplanned: Block Attention Residuals, a multi-residual-stream softmax mixture over 8 snapshots, and KDA's short causal conv with fused SiLU. Only the 69 KDA layers are a wholly new kernel family. Predicted ~0.27 tok/s with the trunk resident, which leaves only ~10 GB, so the expert arena is effectively zero. The trunk-miss model stays OPEN: O_DIRECT is confirmed, yet two rungs still decode below their own bandwidth floor. S1-S3 run on 8.9 MB of fixtures, S4 converts and decodes real weights on /swarm's 7.7 TiB, S5 needs ~200 GiB more NVMe than an empty pool provides.
---

# Kimi-K3 — implementation plan

**K3 is a capability this engine must have.** This document is the plan to get there, not an
assessment of whether to. Written 2026-08-09 from the shipped `config.json` and the C
reference at `github.com/FareedKhan-dev/kimi-k3-in-c`, with every claim about rivoli's own
source verified against the tree the same day.

Target: `moonshotai/Kimi-K3`, text-only. 93 layers, 2.78T total / 104.2B active.

## STATE

- **Six stages, six gates.** S0 ground truth → S1 artifact + harness → S2 kernels → S3 layer
  loop → S4 real weights → S5 residency. Each gate is a **correctness stop point**: work does
  not proceed past it until it is met. §G.
- **Every gate must be proven able to fail.** A gate accepted without a deliberate-defect run
  is not a gate. This is the discipline that caught two criticals in the V4 port before the
  GPU did, and the failure class here — fluent wrong text — is invisible to every other check
  this repo has. §G.
- **Reuse is real but narrower than it first looks.** The `.f4` *value* encoding transfers
  bit-identically; the container does not, and `.f4` has no shared block at all. SiTU-GLU is
  **fused inside** `moe.hip`'s fp4 expert kernel, so that is a new kernel variant rather than
  an activation swap. The trunk's dtype is unknown and rivoli's resident path is fp8-only.
  §3.
- **Only the 69 KDA layers are a wholly new kernel family.** The 24 full-attention layers are
  MLA + q-LoRA — GLM-5.2's own family, which `mla.hip` already serves — minus RoPE, plus a
  gate. §3.
- **Two config shapes will produce fluent wrong text if assumed.** `linear_attn_config` has
  94 entries for 93 layers, and routed experts are 3584 wide where `hidden_size` is 7168.
  Both are S0 blockers. §2.
- **Traffic is 25.83 GB/token of expert reads, provisional.** It rests on the unresolved 3584
  width; if `down_proj` targets 7168 the figure is 34.4. Predicted **~0.27 tok/s** with the
  trunk resident — disk bandwidth divided by bytes, which no kernel work moves. §4.
- **The trunk-miss model is OPEN.** The reference's own ladder puts two rungs below their
  bandwidth floor, so "the trunk is re-read every token" is not established. Trunk *sizes*
  are sound; trunk *rates* are not. S0 resolves it. §4.
- **S4 is not blocked.** `/swarm/storage` has 7.7 TiB, enough to convert and run real weights.
  Only S5's throughput and residency work needs the ~200 GiB of NVMe that does not exist. §5.

## G. The gate model

Every stage ends in a gate. A gate is **met or not met**; there is no partial credit and no
proceeding-while-noting. When a gate is not met, the work is to make it met.

Three rules bind every gate in this document.

**1. Prove it can go red.** For each gate, state what would have to be true for it to reject a
wrong implementation. Then break the implementation deliberately and confirm it reddens **at
every case the defect touches and stays green at every case it does not**. A gate that reddens
everywhere proves nothing; a gate that never reddens is decoration. Record the defect runs
next to the gate.

**2. Name the blind spot.** The most-trusted case is the least informative one. Layer 0 is
dense FFN with no routing and is what everyone checks. Gates must cover a KDA layer, an MLA
layer, layer 0, and layer 92 — the pattern-breaking last MLA layer (§2).

**3. Fluent wrong text is the failure mode.** None of this port's likely defects crash. Wrong
router scoring, a missing QK-norm, an un-suppressed RoPE, a mis-scaled FP4 group, a sign
error in the KDA decay — all produce readable, on-topic, wrong output. `distinct` and
`longest repeated block` cannot see any of them and have misled three investigations in this
repo. Gates are numeric or they are nothing.

| gate | after | asserts |
|---|---|---|
| **G0** | S0 | every unknown has a recorded answer and a source |
| **G1a** | S1a | `.f4` repack bit-exact both ways; existing artifacts still open byte-identically |
| **G1b** | S1b | every golden demonstrably reddens on a deliberate defect; independent anchor in place |
| **G2** | S2 | each kernel passes its operator fixture **and** its defect run, in order |
| **G3** | S3 | the three zero-tolerance model gates pass on the tiny model; KDA state carries |
| **G4** | S4 | real-weight decode is coherent and on-task; byte accounting reproduced from the artifact |
| **G5** | S5 | throughput inside the registered band; output byte-identical to G4 |

## 1. Ground truth — the config

Read from `moonshotai/Kimi-K3/config.json` 2026-08-09. The text model is nested under
`text_config`, behind a `KimiK3ForConditionalGeneration` multimodal wrapper.

| field | value | consequence |
|---|---|---|
| `architectures` / `model_type` | `KimiK3ForConditionalGeneration` / `kimi_k3` | new `Arch` arm; must REFUSE, never fall back to GLM |
| `num_hidden_layers` | 93 | 1 dense + 92 MoE |
| `linear_attn_config` | 24 full + 69 KDA, **94 entries** | **94 entries for 93 layers** — §2 |
| `hidden_size` | 7168 | |
| `num_attention_heads` | 96 | |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | MLA + q-LoRA. `gpu.rs:931` caps kv_lora at exactly 512 — K3 sits on the boundary |
| `qk_nope_head_dim` / `qk_rope_head_dim` | 128 / 64 | the 64 are **allocated but unrotated** |
| `num_experts` / `num_experts_per_token` | 896 / **16** | the traffic figure, §4 |
| `num_shared_experts` | **2** | three `ensure!` sites, and `.f4` has no shared block — §3 |
| `moe_intermediate_size` | 3072 | |
| routed expert hidden | **3584** | **not `hidden_size`** — §2 |
| `hidden_act` | `situ` | SiTU-GLU β₁=4 β₂=25 — and it is fused, §3 |
| `vocab_size` | 163,840 | vs the tokenizer's 163,584 entries — reconcile in S0 |
| `rms_norm_eps` | 1e-5 | |
| `max_position_embeddings` | 1,048,576 | sizes the MLA KV slabs |
| `num_nextn_predict_layers` | 0 | no MTP, no speculative decode |
| `quantization_config` | `mxfp4-pack-quantized`, group 32 | = `.f4` values, §3 |
| `vision_config` | 27 layers, patch 14 | **out of scope** |

## 2. The two shapes that produce fluent wrong text

**`linear_attn_config` has 94 entries for 93 layers.** This is the shape that cost V4's S1b
layer 42: `compress_ratios` had 46 entries for 43 layers, the first cut assumed clean
alternation, and it silently dropped the last layer's compressor and indexer. **Do not infer
the KDA/MLA assignment from a pattern.** Index it explicitly, assert 69 and 24, and assert
what the last layer is. The reference places MLA layers at one-based 4, 8, 12, … 92, **and
93** — so the last layer is MLA, breaking the every-fourth pattern the first 92 entries
suggest.

**Routed experts are 3584 wide; `hidden_size` is 7168.** Each expert is
`3 × 3584 × 3072 = 33,030,144` params, which reproduces the reference's stated per-expert size
exactly. rivoli assumes expert input width **is** `hidden` throughout the MoE path, and that
assumption is load-bearing in the arena's row-stride arithmetic and the fp4 dot loop's `dim`
argument. Resolve what the 7168→3584 reduction is — a projection, two halves, a gated split
feeding SiTU-GLU — from `k3_bind.c` / `k3_ops.c` / `include/k3/k3_cfg.h` **before any
container work**.

This also decides the traffic figure: if `down_proj` must project back to 7168, an expert is
`2×3584×3072 + 3072×7168 = 44.0M` params and §4's number becomes 34.4 GB/token.

## 3. What is new, what is reused

Every row verified against the tree 2026-08-09.

| piece | status |
|---|---|
| `.f4` E2M1/E8M0 **value encoding** | **reuse** — bit-identical, §3a |
| expert streaming, io_uring fetch, byte arena, residency cache | **reuse verbatim** |
| sigmoid router, per-expert bias steering selection only | **reuse** — `Scoring` enum ships it; verify the bias semantics |
| MLA + q-LoRA, q-norm, 24 layers | **reuse `mla.hip`** minus RoPE — the **gate is new kernel work** |
| `Arch` enum, per-arch help and refusal | **reuse** — `arch.rs` is already the multi-arm shape |
| **KDA, 69 layers** | **new kernel family** — the bulk of S2 |
| **fp4 expert dot + SiTU-GLU** | **new variant** — the activation is fused in, §3b |
| **trunk GEMV at K3's dtype** | **unknown, possibly a second new family**, §3c |
| 2 shared experts | `.f4` has **no shared block**; three `n_shared` sites, §3d |
| expert width 3584 ≠ hidden 7168 | assumption break, §2 |
| tiktoken, 163,840 vocab | new tokenizer path **and** a third hand-ported chat template |
| vision tower, MTP | **out of scope** |
| second backend | **does not exist** — Vulkan retired 2026-08-06; `rocm` only |

### 3a. The value encoding is already in the tree

K3 ships `mxfp4-pack-quantized`, group 32 — `value = E2M1[nibble] · 2^(E8M0 − 127)`, one
shared 8-bit exponent per 32 elements, low nibble is the even element. rivoli's `.f4` is the
same statement: `kernels/common.hpp:466`, `:481` (`F4_GROUP 32`), `:549` (`e8m0f`).

So the expert weights are a **repack, not a requantization** — values pass through untouched,
and M3c's tuned fp4 dot loop (branchless decode, 105 instructions per 128 weight bytes) needs
no numerical change.

**The containers are not identical**, and the verdict must not be read as saying so. rivoli's
`.f4` is a 4096-byte header plus `f4_expert_stride`-padded blocks in which each projection's
nibbles and its scale row are separate spans. K3's is packed differently. What transfers is
the arithmetic, which is what makes it a repack.

**Nibble order is one bit between correct and garbage.** Assert it against the reference's
operator fixture in S1a, not in S3.

```
33,030,144 params × (4 bits + 8 bits/32) = × 0.53125 B = 17,547,264 B/expert
```

### 3b. SiTU-GLU is fused inside the fp4 expert kernel

`kernels/moe.hip:329` computes `float sw = swiglu_clamped(g[t], u[t], limit);` **inside**
`moe_gateup_f4_impl`, with `limit` passed as an argument (`:406`) and the launcher refusing
any value that would disable the clamp (`:439`). There is no separate activation launch on the
`.f4` MoE path — `linalg.hip`'s `swiglu` serves the dense MLP and the fp8 shared expert only.

So SiTU-GLU means **a new `moe_gateup_f4` variant plus its `_r2` twin, its launcher, its
argument guards, and its own bit-exactness gate.** An implementer who changes `linalg.hip`
alone will watch the dense path go right and every routed expert stay wrong — the exact
fluent-wrong-text class §G rule 3 exists to catch. `linalg.hip` also needs the SiTU form for
dense layer 0 and the shared experts, so **both** paths change.

### 3c. The trunk's dtype — RESOLVED, and this section was wrong

> **CORRECTED 2026-08-09 by S0.** The trunk is **bf16, 113.49 GB**. And the conclusion below
> is **false**: `kernels/v4compress.hip:82` `v4_dense_gemm_bf16` is a working bf16 GEMV,
> already V4's lm_head and already through the ABI wall, and `resident.safetensors` already
> accepts `Dtype::Bf16`. The trunk GEMV is a **performance** problem, not a new kernel family
> — see "What G0 changed". The paragraph below is kept for the inventory, which is accurate
> about `linalg.hip` and `mla.hip` and was simply not the whole tree.

`mla.hip` ships `mla_absorb_fp8`, `mla_value_fp8`, `v4_gemv_fp8`. `linalg.hip` ships
`gemv_fp8`, `gemv_fp8_splitk`, `gemv_vq`, `gemv_f32`, `gemv_i8`, `gemv_i4` — **no fp4 and no
bf16 GEMV**. `artifact/format.rs` describes `resident.safetensors` as fp8 attn/dense, int8
embed, f32 norms; `model.rs` hard-refuses anything but `e4m3` + `ue8m0` at `[128, 128]`.

K3's quantization is group-32 along the input dim, not a 128×128 tile, and an INT8 trunk at
~56.7 GB against 113.49 GB implies 2 B/param, i.e. bf16. Either way the 24 MLA layers, the 69
KDA layers, the dense layer and a 163,840-row head need a trunk GEMV family that does not
exist here. **S0 settles this**; it is the largest unpriced item in the plan.

### 3d. `.f4` has no shared block — and that is already CORRECT for K3

> **CORRECTED 2026-08-09 by S0. This entire work item is VOID.** K3's two shared experts are
> one fused bf16 MLP living in `resident.safetensors`, so `.f4` carrying exactly `n_experts`
> blocks is already right. `has_shared()` does not become a count, `shared_block` does not
> gain a range, and none of the three `n_shared` assertions need lifting — `model.rs:673` is
> `V4Config::validate` and `gpu.rs:845` is GLM's `GpuEngine`, neither of which K3 reaches.
> What survives is one **new** guard: `top_k * rows + n_shared <= MAX_BATCH` exists only on
> GLM's `Pin::build` (`pin.rs:338`), and K3 at top-16 sits at exactly 32 of `MAX_BATCH = 32`
> at two rows. Write it for K3 rather than inheriting V4's gap.

The `n_experts + 1` layout belongs to `.vq3` and `.i4`. `format.rs:1190` sizes files as
`hbytes + (n_experts + usize::from(has_shared)) * stride`, where `has_shared()` (`:1048`) is a
**boolean** and is **false for F4** — V4's single shared expert is fp8 and rides
`resident.safetensors`.

The work is: turn `has_shared() -> bool` into a shared-block **count**; generalise
`ExpertSet::shared_block`'s single-index accessor to a range; and lift **three** sites —
`model.rs:251` (`n_shared < 1` bail), `model.rs:673` (`== 1`), `gpu.rs:845` (`== 1`) — plus the
test at `model.rs:1190` asserting `n_shared_experts: 2` is rejected. Also check `pin.rs`'s
`top_k * MAXROW + n_shared <= MAX_BATCH` (32): K3 is 18 at one row, 34 if the fp4 path ever
batches.

Whether K3's two shared experts are MXFP4 (so `.f4` gains a shared-block concept it was
deliberately built without) or full-width and resident (so `.f4` is untouched) is **S0's
call**, not this item's.

### 3e. NoPE — refuse the rotation, do not just skip it

`qk_rope_head_dim: 64` exists and the config carries **no `rope_theta` at all**; position
comes from KDA's decay recurrence. rivoli applies interleaved RoPE unconditionally. Rotating
those 64 dims because they are present and named `rope` produces fluent wrong text on all 24
MLA layers. Assert the absence of `rope_theta` and make the K3 path **refuse to construct a
rotation table**, so the failure is a startup error rather than bad output.

### 3f. KDA

A linear-attention recurrence with fixed-size state rather than a growing KV cache: L2Norm on
q and k only, a channel-wise forget gate `g = g_min · sigmoid(e^A · z)` with `g_min = −5` and
`A` indexed per head, and a per-head β from sigmoid. State is 626 MB across all 93 layers and
is **always resident** — it never streams. At decode this is a rank-1 state update per token,
so it is compute- and bandwidth-trivial; the work is getting the recurrence numerically right,
not making it fast.

## 4. Traffic and the throughput this implies

Per token, every MoE layer selects 16 of 896 experts:

```
16 × 17,547,264 B = 280,756,224 B/layer × 92 = 25,829,572,608 B = 25.83 GB/token
```

The reference measures the same figure independently. **Both derivations share unverified
inputs** — the 3584 width, 92 MoE layers, top-16 — so agreement shows the two read the config
the same way, not that either read it right. **Provisional until G0**, and if `down_proj`
targets 7168 it becomes 34.4 GB/token.

Against the other models here: GLM ~1.4 GB/token, V4 3.449, K3 **25.83** — expert reads only.

### 4a. The expert cache is weak here, but not for the reason first assumed

The reference reports 0.0% expert hit below ~36 GB of arena, rising to 43.8% at 224 GB. That
does **not** mean the distribution is flat: a 36.39 GB cache holds 2,074 of 82,432 experts =
**2.52% residency returning a 29.9% hit — 11.9× uniform**. At 108.98 GB: 7.53% residency,
43.8% hit, 5.8× uniform. The distribution is **skewed**, and something else produces the 0%
rungs — either budget competition with the trunk, or a threshold effect (23.59 GB of cache
gives 0.0% while 36.39 GB gives 29.9%, which no smooth curve produces).

**Design consequence:** hold the trunk, and do not size the expert arena expecting it to earn
its keep. That follows from bytes alone — the trunk is 113.49 GB every token needs in full,
while at 2.5–7.5% residency the cache returns a fraction of 25.83 GB. It does **not** follow
from the reference's `s/tok` column, which §4b disqualifies.

### 4b. The trunk-miss model is OPEN — do not build on it

Applying "the trunk is re-read every token when not resident" to the reference's own ladder
puts two rungs **below** the bandwidth floor, which is impossible:

| rung | expert GB | trunk miss GB | total | floor @3.2 GB/s | measured |
|---|---:|---:|---:|---:|---:|
| 8 GB | 25.83 | 113.49 | 139.32 | **43.54 s** | 32.69 s |
| 96 GB | 18.11 | 67.53 | 85.64 | **26.76 s** | 24.40 s |

Most likely explanation, unverified: the reference's host has **228 GiB of RAM** and the
ladder caps the *application's* arena, not the OS page cache — so a trunk the app records as
missed is served from cache at memory speed. **rivoli uses O_DIRECT and gets no such help**, so
if that is the cause, none of the reference's trunk *rates* transfer.

| claim | status |
|---|---|
| 25.83 GB/token of expert reads | stands (provisional on §2's width) |
| trunk is 113.49 GB and must be held or fetched | stands — a size, not a rate |
| trunk is re-read in full every token when not resident | **OPEN** |
| ~0.27 tok/s with the trunk resident | stands — trunk traffic is zero there |

### 4c. The predicted operating point

NVMe 7.0 GB/s O_DIRECT, trunk resident, ~14.7 GB of expert cache at ~0% hit:

```
25.83 GB / 7.0 GB/s = 3.69 s/token = 0.27 tok/s
```

That is disk bandwidth divided by bytes. No kernel work moves it, and every span this engine
has optimized hides underneath it. Register it before S5's first run so the result is scored
against a prediction rather than against an expectation.

**This is also the only allocation rivoli can currently express.** `architecture.md` §1 splits
the device budget into a `DeviceTier` for resident weights (`src/memory/device.rs:151`,
bump-allocated, filled once, freed as a unit) and a `RoutedPool` for streamed **expert blocks
only** (`src/memory/routed.rs:234`). Nothing streams trunk weights; there is no partial trunk
pin. The reference's "27 pinned layers / trunk hit 25.4%" rows are not expressible here.

**Budget check.** The device budget is ≈115 GiB of 116 GiB GTT, and the 113.49 GB trunk leaves ≈10.0 GB
spends all of it before the 626 MB KDA state, the 24 MLA layers' KV slabs at up to 1,048,576
positions, the layer-major residual stream, activation scratch, io_uring registered buffers,
and `OS_RESERVE = 16 << 30`. **S5's registered band must come from a budget with those already
subtracted**, or it is unreachable by construction.

## 5. Capacity and where the weights live

```
experts   2,722,740,830,208 params × 0.53125 B = 1.4465 TB = 1.3155 TiB
trunk (bf16)                         113.49 GB =            0.1032 TiB
                                                 ─────────────────────
total                                1.560 TB  =            1.419 TiB
```

Pool measured 2026-08-09 (`btrfs fi usage /`): **1.69 TiB total, 431.72 GiB free**, with GLM
at 675 GiB and V4's `.f4` at ~146 GiB. Deleting both frees 1.223 TiB — still **200 GiB short**.

**`/swarm/storage` has 9.8 TiB with 7.7 TiB available** (`df`, 2026-08-09), so the artifact can
be stored and converted today. It is NFS at a measured 154 MB/s, so streaming 25.83 GB/token
across it is ~168 s/token — fine for a bounded correctness run, useless for throughput.

| work | where | blocked? |
|---|---|---|
| convert, verify byte accounting, real-weight correctness run | `/swarm` | **no** |
| throughput measurement, residency study | NVMe | **yes** — needs ~200 GiB |

Fidelity is native MXFP4, decided at planning time. int3-vq requant would be lossy-on-lossy on
already-4-bit weights (`int4-scales.md` records that chain at PPL 73.43), and lower-bitrate VQ
needs a new `dot_vq_wave_r` inner loop and packer.

---

# Stages

## S0 — ground truth. No code, no checkpoint download.

Everything here is a question whose wrong answer produces fluent wrong text later. Cheapest
possible place to be wrong.

1. **Resolve the 7168 → 3584 expert width** (§2) from `k3_bind.c` / `k3_ops.c` /
   `include/k3/k3_cfg.h`. Record which tensor does it and the true MoE input shape.
   **Blocks S1a** — the arena's row stride depends on it — and moves §4's traffic figure.
2. **Extract the exact KDA/MLA layer assignment** from `linear_attn_config`'s 94 entries.
   Assert 69 and 24; record what the 94th entry is and what layer 92 is.
3. **Settle the trunk's numeric format** (§3c). Decides whether `mla.hip` is reusable or a
   second GEMV family is new work. **Blocks S2.**
4. **Settle shared-expert width and dtype** (§3d). MXFP4-and-streamed puts them in `.f4`;
   full-width-and-resident puts them in `resident.safetensors` as V4's is. **Blocks S1a.**
5. **Confirm nibble order and the E8M0 bias** against the reference's operator fixture versus
   `common.hpp`'s `e8m0f`.
6. **Reconcile the vocab**: `vocab_size` 163,840 against the tokenizer's 163,584 entries.
   Decides the embed and lm_head row counts and the id space for special tokens.
7. **Resolve §4b's contradiction**: does the reference run the trunk through O_DIRECT or the
   page cache, and is the trunk genuinely re-read per token? **Blocks S5's registered band.**
8. **Pin the reference.** Record its commit SHA and retrieval date; vendor the memory-ladder
   and `trunk-cache-split.tsv` into `docs/measurement/` with their source paths. Every
   quantitative claim not from the config comes from a live third-party repo.
9. **Correct `other-models.md` §2 in place with a dated note** — its four walls missed top-k,
   it priced capacity at int3-vq rather than native MXFP4, and it read the MLA half as foreign
   when it is GLM's own family.

### G0 — met when

Every item above has a recorded answer **and its source**, in a table with no unknown cells.
Items 1, 3 and 4 additionally carry the code path they imply, because each changes what S1a
and S2 build rather than merely what they assume.

### G0 — RESULTS, 2026-08-09. MET except item 7.

Reference pinned at **`ff11dce858a2eb8a781224facdffd33a1fa48d25`** (2026-08-07, "Release
v1.0.0: verified end to end"). The extracted forward pass is now
**`docs/reference/k3-architecture.md`**; this table records the answers and what each changes.

| # | question | answer | source |
|---|---|---|---|
| 1 | 7168→3584 | **A latent sandwich.** MoE routes on full width, `down` 7168→3584, experts run at 3584, RMSNorm the **aggregate**, `up` 3584→7168. `routed_expert_hidden_size = latent = 3584` | `k3_ops.c` `k3_moe`; `k3.h` `K3Cfg` |
| 2 | layer map | **Two explicit arrays**, `full_attn_layers` (24) + `kda_layers` (69) = 93. No inference needed. Zero-based MLA = 3,7,…,87,**91,92** — the last two adjacent | `config.json` `linear_attn_config`; `k3_is_mla` |
| 3 | trunk dtype | **bf16 (`uint16_t`), 113.49 GB** — not 108.81 | `k3.h` storage summary; `wdt` = `K3_WBF16` |
| 4 | shared experts | **ONE fused wider MLP** at full hidden 7168, intermediate `moe_inter*n_shared` = 6144, **bf16 and resident**, added **unweighted after** the up-projection. Not MXFP4, not streamed, not two MLPs | `k3_ops.c` `sh1/sh3/sh2`; `k3_bind.c` |
| 5 | nibble order | **Low nibble = even element** ✓ matches rivoli. Bias 127. **But `sb == 255` → ZERO in the reference**, where rivoli returns NaN and the converter bails | `k3_ops.c` MXFP4 decode vs `common.hpp:549`, `quant.rs:748` |
| 6 | vocab | **163,840**, no tied embeddings (`embed_tokens.weight` and `lm_head.weight` are separate) | `config.json`; `k3_bind.c` |
| 7 | trunk re-read / O_DIRECT | **PARTIAL — see below** | `k3_portable_io.h` |
| 8 | pin the reference | SHA above; the ladder and `trunk-cache-split.tsv` still to vendor | GitHub API |
| 9 | correct `other-models.md` §2 | done, dated note in place | — |

**Item 7 is not met and stays open.** The reference **does** use `O_DIRECT` on Linux for both
trunk and expert reads, with a documented buffered fallback if it fails. That *weakens* the
page-cache explanation §4b offered for the sub-floor contradiction without resolving it — two
rungs still decode faster than a 3.2 GB/s floor allows. The leading remaining hypothesis is
that **3.2 GB/s is a single-stream figure** the parallel reader beats. **S5's registered band
must not be derived until this is settled**; nothing before S5 depends on it.

### What G0 changed

**In rivoli's favour — three plan claims were wrong:**

- **§3c is FALSE: a bf16 GEMV already exists.** `kernels/v4compress.hip:82`
  `v4_dense_gemm_bf16` computes `out[m,n] = x[m,k] · w[n,k]ᵀ` for f32 activations against bf16
  weights, and is already V4's lm_head (`v4gpu.rs:2204`) and its compressor GEMV
  (`v4gpu.rs:320`), already through the ABI wall (`hip.rs:993`). The trunk GEMV is **not a new
  kernel family** — it is a *performance* problem, because its inner loop
  (`v4compress.hip:96`) is one wave per output element with scalar u16 loads. At 113.49 GB of
  trunk per token that loop **is** the decode. The correct kernel exists and is a free
  correctness oracle for the fast one.
- **`resident.safetensors` already accepts bf16.** `Dtype::Bf16` is in the closed enum
  (`format.rs:99`), `copy_verbatim` already writes it, and `convert_v4.rs:226` already emits
  bf16 `embed`/`head`. The `e4m3`/`ue8m0`/`[128,128]` refusal is a **`V4Config::validate`
  check** (`model.rs:766-781`), not a resident-file check — `K3Config` simply never writes it.
- **§3d's `.f4` shared-block work is VOID.** `has_shared()` returns `false` for `F4`
  (`format.rs:1048`) and `.f4` is exactly `n_experts` blocks — which is **already correct** for
  a resident bf16 shared expert. `has_shared` does not become a count and `shared_block` does
  not gain a range. Nor do the three `n_shared` assertions need lifting: `model.rs:673` is
  `V4Config::validate` and `gpu.rs:845` is GLM's `GpuEngine`, and K3 reaches neither.
- **The fp4 kernels are already width-parametric.** `moe.hip:300` takes `hidden` and `inter`
  as runtime arguments and derives the row stride and dot `dim` from them, so passing 3584
  is correct as written. `ACT_QUANT_BLOCK=128` divides both 3584 and 3072 — checked.

**Against — one new Tier-1 blocker the plan never named:**

- **`SafeWriter` holds every resident tensor in host RAM until `write`.**
  `format.rs:148-154`: `tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>`, serialised only at
  the end, sized for "the resident set is ~10 GiB". **A 113.49 GB bf16 trunk does not fit on a
  128 GB machine.** `convert_k3` cannot exist until `SafeWriter` becomes a two-pass streaming
  writer. The reader is fine — `open_dir` mmaps.

**And the budget is tighter than §4c said.** At the corrected 113.49 GB trunk against ≈123.5 GB
of device budget, the residual is **≈10.0 GB** for the KDA state, 24 layers of MLA KV, the
residual stream, scratch, io_uring buffers and `OS_RESERVE`. The expert arena is **effectively
zero**, which turns §4a's "do not expect the arena to earn its keep" into "there is no arena".
The ~0.27 tok/s prediction is unchanged — it already assumed ~0% expert hit.

**Architecture the plan did not know about**, all now in `k3-architecture.md`:
**Block Attention Residuals** (a multi-residual-stream scheme: 8 snapshots at zero-based
layers 0,12,…,84, mixed by softmax over ≤9 sources, twice per layer plus once model-level —
a new small kernel and a `[T][9][7168]` stack live across the layer loop); **KDA's short causal
convolution** (k=4, depthwise, SiLU fused, with its own `[P][3]`×3 state); the full delta-rule
recurrence and its `[96][128][128]` fp32 state; **MLA's softmax scale over 192**, its
unrotated-but-scored rope dims, and its gate-without-norm; the router's unbiased combining
weights; and SiTU-GLU's exact form.

**25.83 GB/token STANDS.** Item 1 resolved the width against the *latent*, not `hidden` —
`down_proj` maps 3072→3584, so an expert is `3 × 3584 × 3072` and the 34.4 GB/token
alternative is **refuted**.

## S1 — foundation. No GPU.

### S1a — artifact: config, `Arch`, naming, `.f4`

Owns `src/artifact/{model,config,format,quant}.rs`, `src/arch.rs`, and a converter binary.

1. `Arch::KimiK3`, recognising `KimiK3ForConditionalGeneration` / `kimi_k3`, refusing anything
   unknown. K3 has no `--attn` choice — the layer kinds are fixed by the weights — so it takes
   the `attn_modes() -> None` path V4 established and hides the attention-shaped flags.
2. `ModelConfig`: descend into `text_config`; accept K3's key spellings (`num_experts` /
   `num_experts_per_token` / `num_shared_experts`). Absent fields are **optional behind an
   explicit architecture discriminant**, never defaulted to 0 — a default that yields a
   runnable-looking config is the failure to avoid.
3. **`SafeWriter` must become streaming.** *(Tier 1, added by S0, blocks everything.)*
   `format.rs:148-154` holds every resident tensor as `Vec<u8>` until `write`, sized for a
   ~10 GiB resident set. **113.49 GB does not fit in host RAM**, so `convert_k3` cannot exist
   until this is a two-pass writer: compute the header and offsets, then stream bytes.
   `write_atomic` (`:703`) takes `&[u8]` and needs the same.
4. **Settle e8m0 `0xff` against K3's real scale tensors.** *(Tier 1, added by S0.)* The
   reference maps 255 → **zero**; rivoli's `e8m0f` returns a quiet NaN (`common.hpp:549`) and
   `quant.rs:748` **bails**. Scan K3's `.scale` tensors first. If no `0xff` appears, keep the
   refusal — it is the stronger guarantee. If it does, host and device must change **together**
   and `tests/v4_kernel.rs`'s bitwise sweep be re-pinned; changing one alone is a silent
   host/device divergence that `moe_fixed`'s clamp launders into a finite ±2^14.
5. **Thread `moe_latent` (3584) separately from `hidden` (7168).** *(Tier 1, added by S0.)*
   The fp4 kernels are already width-parametric — `moe.hip:300` takes `hidden`/`inter` as
   runtime arguments — so the work is entirely in what Rust binds. Ten sites, of which the
   decode-loop ones are dangerous because a wrong value passes every length check:
   `pin.rs:867` (arena row stride), `quant.rs:184` (`vq_expert_layout` → `[(3072,3584),
   (3072,3584), (3584,3072)]`), `quant.rs:777`/`:1037`, `format.rs:520`/`:1244`,
   **`v4gpu.rs:1540`** (MoE dispatch), **`v4gpu.rs:1111`** (the fixed-point accumulator must be
   latent-wide), `v4gpu.rs:1789`/`:1843`, `model.rs:143` (`ensure_group_aligned`).
6. `.f4` repack for K3 experts, per §3a.
7. **Refuse the absence of `rope_theta`** rather than defaulting a rotation table (§3e).
8. Converter binary: V4 shipped `src/bin/convert_v4.rs` alongside `convert.rs` rather than
   extending it. Follow that. Establish what format the reference's tiny model is stored in —
   rivoli's converters read HuggingFace safetensors, and a C reference's fixture may be its own
   layout, in which case a reader is needed here.
9. `Arch::KimiK3` plumbing is narrow and every `Arch` match is exhaustive, so omissions are
   compile errors: `arch.rs` (variant + six arms + a both-directions recogniser test),
   `model.rs` (`K3Config` + `impl ArchConfig`), `main.rs:979` (dispatch + `run_k3`),
   `lib.rs` (`pub mod k3gpu`), `src/bin/convert_k3.rs` (auto-discovered, no `Cargo.toml`
   edit). Note `tests/kernel_coverage.rs` is **not** feature-gated and fails the build for any
   new `launch_*` without an oracle.

Tokenizer work (tiktoken → `tokenizers` form, and a third hand-ported `encode_k3` chat
template — `tokenizer.rs` carries no Jinja engine) is **deferred to S4**, where the first real
prompts run. No gate through G3 consumes a chat template.

### G1a — met when

- The `.f4` repack is **bit-exact in both directions on real tensors**, asserted, not assumed.
- The existing GLM (675 GiB) and V4 (~146 GiB) artifacts still open **byte- and
  offset-identically** under the widened layout, proven by a test that opens them. These
  cannot be cheaply regenerated and the change is being made on their behalf, not K3's.
- A config missing or contradicting any load-bearing field **refuses at startup**, proven by
  feeding it one.
- Byte accounting reproduced **from the artifact**, not from the index.

### S1b — the gate harness

Owns fixtures and the test harness. **Must not touch `gpu.rs`, `attn.rs` or any kernel.**

The reference ships, in ~9 MB and needing no checkpoint: a tiny model whose tensor graph
matches the released architecture, with three zero-tolerance gates (teacher forcing, greedy
decode, incremental decode with KV cache); per-operator fixtures with per-fixture tolerances,
adversarial by construction; a synthetic expert shard for prefetch/eviction; and tokenizer
goldens.

That is far less work than V4's 117 KB hand-transliterated oracle — but it is **cheaper, not
better**, and the difference matters. `common.hpp:480` states the doctrine: the oracle and the
kernel "are independently written from the same format definition, which is the point." The
reference's goldens are the output of **one** third-party implementation, so any misreading it
makes is baked into every golden and rivoli passes by reproducing the same defect. Checking
that fixtures are *adversarial* does not address this — a fixture can be maximally
discriminating and discriminate toward the wrong answer.

So: find and record whether the reference was itself validated against the official
HuggingFace forward pass, on what, and at what tolerance. If it was, cite it. If it was not,
build **one independently sourced anchor** — an HF-generated logit trace on the tiny model, or
a hand transliteration of the two or three operators most likely to be misread, the KDA
recurrence first.

### G1b — met when

- Every golden has a **recorded defect run** showing it reddens at exactly the cases the
  defect touches and stays green elsewhere (§G rule 1). A golden without one does not count.
- Defect runs cover a KDA layer, an MLA layer, layer 0, and layer 92 (§G rule 2).
- The independent anchor exists and passes, or the reference's own HF validation is cited with
  its tolerance.

## S2 — kernels. Each item gates before the next starts.

**Order: AttnRes → MLA → latent sandwich → SiTU/MoE → KDA.** Structural and cheapest first;
MLA before KDA because it is the only one checkable against a path already known to work.

Specifications for all of these are in **`docs/reference/k3-architecture.md`**, whose §10
lists twelve order-of-operations traps. **Each trap is a G2 defect-run candidate** — every one
of them runs cleanly and produces a wrong model.

1. **AttnRes** (`k3-architecture.md` §3) — *added by S0; the plan did not know this existed.*
   A ≤9-source softmax mixture over a residual snapshot stack, twice per layer plus once
   model-level. Arithmetically trivial (~24 Mflop/token); the work is keeping `[T][9][7168]`
   live across the layer loop. First, because it is structural — every later gate runs inside
   it.
2. **Gated MLA** — `mla.hip` with RoPE removed but the 64 rope dims **still cached and still
   scored**, softmax scale over **192** not `qk_nope`, and the output gate applied **before
   `o_proj` with no norm** (the opposite of KDA's norm-then-gate). Watch `gpu.rs:931`'s
   exact-512 `kv_lora_rank` cap.
3. **The latent sandwich** — `down` 7168→3584 and `up` 3584→7168 as bf16 trunk GEMVs, plus an
   RMSNorm on the **aggregate, never per expert**. `v4_rmsnorm` (`mla.hip:334`) is already
   width-generic, so this is two GEMV launches and an existing kernel.
4. **SiTU-GLU + fp4 MoE** — per §3b, a new `moe_gateup_f4` variant plus its `_r2` twin,
   launcher and guards; and the SiTU form in `linalg.hip` for dense layer 0 and the shared
   MLP. Exact form and the `b1*b2 = 100` cap in `k3-architecture.md` §8.
5. **KDA** (`k3-architecture.md` §4) — the new family and the bulk of the stage. Short causal
   conv k=4 with **fused SiLU** and its own `[P][3]`×3 state; L2Norm on q and k only; the
   per-head `A_log` / per-channel `dt_bias` split; the four-pass delta rule over a
   `[96][128][128]` fp32 state. **Decompose into units with their own G2 sub-gates** — 69 of
   93 layers with no prior art in this tree, and one gate at the end of it all is not a gate.
   Heads are independent and the reference asserts head-parallel is bit-identical, so one
   block per head maps directly — but `S` is 64 KB/head and will not sit in LDS, so **fusing
   the four passes is the HIP win the C does not have**. No chunked prefill exists in the
   reference; decode recurrence first.
6. **Router** — sigmoid, bias on **selection only**, combining weights from the **unbiased**
   score, renormalised over the 16. The reference's fixture reorders its top-2 on 5 of 6 rows
   specifically to catch an implementation that ignores the bias.
7. **Trunk GEMV, for speed.** *(Reframed by S0 — it is not a new family.)*
   `v4_dense_gemm_bf16` is already correct and carries S2/S3 unchanged, so **score the fast
   kernel against it — a free oracle V4 already paid for.** But its inner loop
   (`v4compress.hip:96`) is one wave per output *element* with scalar u16 loads, and at
   113.49 GB/token that loop is the whole decode. Needs 128-bit vectorized loads
   (`common.hpp:517`'s `gu8p` pattern), grid-per-row rather than per-element, split-K on
   `gemv_fp8_splitk_impl`'s shape (`linalg.hip:59`), and an `_r2` twin if K3 ever batches.

**Sizing.** The comparable port is V4: `v4gpu.rs` 165 KB + `v4oracle/forward.rs` 117 KB +
`dsv4_encoding.rs` 137 KB + `v4_attn.rs` 142 KB + `v4_kernel.rs` 125 KB ≈ 690 KB of roughly
1 MB total. K3 adds a linear-attention family V4 did not have and reuses an oracle V4 had to
write. Treat V4 as the floor.

### G2 — met when

Each of the five passes **its operator fixture and its defect run**, in order, before the next
begins. Kernel-by-kernel, not stage-at-the-end.

## S3 — layer loop, first decode

A `src/k3gpu.rs` and a `main.rs` K3 branch. Name **every deviation from the reference at its
call site** — V4's three named deviations are what let its reviews catch two criticals before
the GPU did.

**Do not write `k3gpu.rs` by mirroring `v4gpu.rs`.** `build.rs` runs `jscpd --min-tokens 15`
over `src/` and panics on **any** clone, with no threshold; the tree is at zero clones. A
165 KB file shaped like another 165 KB file will not build. This is an architectural decision
to take before S3 starts, not one to discover on the first `cargo test`: **factor a shared
engine skeleton across the three models.** Three is where a skeleton pays for itself, and the
V4 port already declined to exempt `compress_topk` rather than widen the exemption list.

### G3 — met when

- The reference's three zero-tolerance gates pass on the tiny model: teacher forcing, greedy
  decode, incremental decode with KV cache. Exact token match, zero tolerance.
- **KDA state carries correctly across positions** — the incremental gate must be shown to
  detect a deliberately broken state advance, or it is not testing the recurrence.
- Every deviation from the reference is named at its call site.

## S4 — real weights on `/swarm`. Not blocked.

1. Convert the full artifact to `/swarm`; reproduce §4's byte accounting from it — **both
   halves**, expert and trunk. The reference's `GB read/tok` column is expert-only, and
   conflating it with total traffic is what produced §4b's open question.
2. Tokenizer and chat template (deferred from S1a): tiktoken conversion, `encode_k3`, and a
   byte-level template gate. The template lives only in the source repo and rivoli has drifted
   on this before.
3. A bounded greedy run, ~10 tokens at ~168 s/token ≈ 28 minutes.

**This is the only check that exercises §2's traps against trained weights.** The tiny model
may collapse the very distinctions in question — the 3584-vs-7168 width, the NoPE rotation,
the KDA gate sign — because its tensor graph matching the architecture does not guarantee its
*dimensions* do. Do not defer this waiting for NVMe.

### G4 — met when

- Byte accounting from the artifact matches §4/§5, both halves.
- The greedy run produces **coherent, on-task text**, read by a human. Not `distinct`, not
  longest-repeated-block — §G rule 3.
- Output is deterministic across two runs of the same prompt.

## S5 — throughput and residency. Blocked on ~200 GiB of NVMe.

1. Register the predicted band in `benchmarks.md` **before** the first run, derived from a
   budget with the KDA state, KV, residual stream and scratch already subtracted (§4c). Score
   **total** bytes/token, not expert bytes.
2. **Pin the trunk**, on the byte argument of §4a — not on the reference's `s/tok` column.
3. Measure what Belady, the residency policies and prefetch do against a top-16-of-896
   distribution under budget competition. Note that reproducing the reference's *sweep* would
   need a partial-trunk-residency subsystem rivoli does not have (§4c); scope that
   explicitly or restate the question as the one this engine can ask.

### G5 — met when

- Measured throughput is inside the registered band, or the miss is explained and recorded.
- **Output is byte-identical to G4's** at the same prompt and settings. A performance change
  that alters output is a defect, not a result.

## Standing rules

- **Develop on the dev profile** (`cargo test --features rocm`), where `debug_assert!` is live.
  `--release` is for benchmarks only.
- **`-- --test-threads=1` on any suite that touches the device.** Each device test builds its
  own tier, pool and io_uring ring; in parallel they wedge.
- The GPU is sole-tenant: `flock /var/run/sys-gpu.lock`, build outside the lock, lock per arm
  of an A/B, discard any arm with a non-empty contention witness.
- Never `cargo build` between the two arms of a benchmark.
- Instruments go behind a feature **and** a flag, never an env var.
- Rank quality on paired dNLL from `bin/ppl`; an interval straddling zero is inconclusive.
- Duplication is a build error, and **clippy-green is not duplication-green** — run something
  that re-runs `build.rs`.
- Run the feature union, not just `--features rocm`, on anything touching `telemetry.rs`,
  `eval.rs`, `gpu.rs` or a `ProfileSummary` field.
- CI has **no `rocm` arm and no GPU arm**. Every device path is checked exactly as often as
  someone runs it here — which is why the gates above are the whole safety net.
