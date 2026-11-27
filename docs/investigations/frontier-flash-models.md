---
status: live
scope: engine
verdict: THE KEEP-UP PLAN (owner goal 2026-11-27) — the five open Flash-tier families are tracked at pinned revisions: DeepSeek-V4.1-Flash-1017 (763B, CED + CSA2 + Engram + DSpark, FP8/FP4), GLM-5.3-Flash (320B/18B, KDA + Gated DSA + mHC, FP8, MIT), Kimi K3.1 Pro (2.8T/104B, 69 KDA + 24 Gated DSA, MXFP4), Qwen3.8-Flash-Next (125B/6B + 51B n-gram, GDN + QSA, fp8) and Inkling 2 Small/Mini (594B/17B and 26.4B/3.4B), ALREADY SHIPPED text-only on MXFP8 and FP8+MXFP4 at 7182854 (vision and MTP refused by owner scope). A standard intake ladder I0-I7 (pin, anchor, kernels, artifact, doors, arm, gates, closeout) is the reusable seam; milestones F0-F8 run it in expected-speed order — Qwen first (S0-S5 already landed), then GLM-5.3-Flash, DeepSeek-V4.1-Flash and Kimi K3.1 Pro — with FP8/MXFP4, KDA, Gated DSA, mHC and the sparse-indexer scaffold built once and shared. No model is claimed until greedy ids are token-identical to its first-party reference, every model passes a 1M-context prefill/decode gate, ppl sits inside a pre-registered band, A-vs-A determinism is measured, and every gate has a recorded red proof. Targets: 5/5 at 1M and a measured time-to-first-decode baseline; no tok/s is claimed from a third-party number.
---

# Frontier flash models: the keep-up plan

> **This is the plan of record, not evidence.** Every number about this box arrives through a
> gate with a recorded red proof. Numbers from vendor blogs, papers and other engines are
> cited as leads and carry their source; they are not measurements of rivoli and must never
> be quoted as if they were.

## 0. What changed on 2026-11-27

The foundation is shipped: the rewrite is M0–M9 complete, all HIP kernels are Rust, the
bit-identity battery is green, Inkling 2 Mini/Small decode on this box, and the GLM, V4,
Kimi-K3 and Qwen S0–S5 seams exist. The old backlog is retired rather than finished:

| retired doc | what it still owned | what carries forward |
|---|---|---|
| `qwen-flash-next-port.md` | S6–S8: arm, `tests/smoke-qwen.sh`, parity/ppl/determinism, closeout | F2 finishes the same checkpoint |
| `release-v3.md` | v3.0.0 gate day and tag | F8 owns the release labels; no v3.0.0 claim |
| `v4-scored-selection.md` | NLL-vs-position gate, CodeScene score | F5 re-enters through V4.1-Flash |
| `k3-first-checkpoint.md` | K3 chat encoding, ctx-scaled allocation, two checkpoint leads | F6 re-enters through K3.1 Pro |
| `glm-nondeterminism-closeout.md` | four closure items | §10 keeps the limit as a release label and a determinism gate |

Their bodies remain under `status: obsoleted` for provenance. Nothing in this plan resumes
their unstarted work; the new intake starts from pinned current checkpoints.

## 1. The target set, pinned 2026-11-27

**Baseline.** This plan was first written against HEAD `87dfa70` (2026-11-25). Fresh
`origin/main` is `7182854` (2026-11-26), which adds the Inkling Waves 1–3 performance
commit; it is **not merged into this branch** (the tree has uncommitted work and the
harness blocks mutating git verbs). The `tree today` column below is read at `7182854`;
each model's I0 re-pins its own revision.

The user's goal is the open **Flash tier** of five families, not the closed flagships. The
rows below name what is actually downloadable today; the exact revision SHA, config hash and
artifact byte count are pinned at each model's I0, never here.

| family | checkpoint | tree today | released | total / active | context | weights | attention | license |
|---|---|---|---|---|---|---|---|---|
| DeepSeek | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | V4-Flash fp4 arm; V4.1 not | 2026-10-17 | 763B / 8B prefill, 16B decode | 1M | FP8 block 32×32 ue8m0, FP4 routed experts | CED 20+20, CSA2 Full/Reindex/Reuse, SWA 128, hierarchical indexer | MIT |
| GLM | `zai-org/GLM-5.3-Flash` | GLM-5.2 arm; 5.3-Flash not | 2026-08-26 | 320B / 18B | 1M | FP8 e4m3 | 45 layers: 34 KDA linear + 11 Gated DSA, mHC, no RoPE | MIT |
| Kimi | `moonshotai/Kimi-K3.1-Pro` | K3 arm; K3.1 not | 2026-11-04 | 2.8T / 104B | 1M | MXFP4 weights, MXFP8 activations | 93 layers: 69 KDA + 24 Gated DSA (replaces K3's Gated MLA) | custom |
| Qwen | `Qwen/Qwen3.8-Flash-Next` | S0–S5; no arm | 2026-08-26 | 125B / 6B, plus 51B n-gram and 4B MTP | 262K native, 1M extended | fp8 (measured 185,502,232,570 B in tree) | 48 layers: 12 × (3 GDN + 1 QSA), gated residual | Qwen community, gated |
| Thinking Machines | `thinkingmachines/Inkling-2-Small` and `Inkling-2-Mini` | **shipped + Waves 1–3 at 16,128; 1M pending (F7)** (text-only; vision/MTP refused) | 2026-11-12 / 2026-11-20 | 594B / 17B and 26.4B / 3.4B | 1M | Small: MXFP8 (first-party). Mini: NVFP4/MXFP8/bf16. Both now support FP8 trunk + MXFP4 experts | Small: 48 layers, 8 of 384 experts, hybrid local/global; Mini: 24 looped layers | Apache 2.0 |

Notes that change the intake:

- **DeepSeek** is no longer the V4 architecture the tree shipped. V4.1-Flash is a
  Causal Encoder–Decoder: the decoder's global KV is projected from the last encoder layer,
  so prefill activates 8B and decode 16B. It adds CSA2 (cross-layer KV and index reuse in
  three static modes), a hierarchical sparse indexer, Single-Pass mHC, 196B Engram
  parameters, DSpark speculative decoding, and FP4 main-KV caching (E2M1 with one E4M3
  scale per 16 channels, 890 B/token global KV). The 1017 checkpoint is a later weight
  update of the same architecture, not a new one.
- **GLM** is the first hybrid in the family: 34 KDA linear layers and 11 Gated DSA layers,
  with mHC and no RoPE (`qk_rope_head_dim` is 0). It is not a GLM-5.2 refresh; the DSA
  layers are new work even though K3 already ships KDA.
- **Kimi K3.1 Pro** is a post-train of K3 with the full-attention Gated MLA layers replaced
  by Gated DSA. The KDA layers, MoE shape and MXFP4 format are inherited, so the intake is
  a refresh plus one new layer family, not a full port. `K3.1 Pro Highspeed` is API-only.
  Kimi has no separately named open Flash checkpoint as of 2026-11-27; K3.1 Pro (104B
  active) is the open K3.1 entry, and any Flash variant goes on the watchlist.
- **Qwen** is the checkpoint the tree already has through S5. The 51B n-gram table is
  explicitly offload-friendly; the QSA indexer works on 512 micro-blocks / 2048 tokens.
- **Inkling 2 Mini is the newest Thinking Machines release** (2026-11-20) and the only
  target that can be fully resident on this box. Small is the streaming tier. Both are
  already supported and benchmarked at 7182854: Small on MXFP8, Mini on NVFP4/MXFP8/bf16,
  plus the FP8 trunk and MXFP4 experts from Waves 1–3. F7 is regression/watch only.

### 1a. Watchlist, deliberately not targets

| family | item | why not a target |
|---|---|---|
| GLM | GLM-5.4 (loop transformer) | no open weights as of 2026-11-27 |
| Qwen | Qwen4-Max, Qwen4 Plus/Flash/27B | Max is closed; open Qwen4 models have no date |
| Kimi | K3.1 Pro Highspeed | API-only; no weights |
| DeepSeek | V4.1-Pro | announced, not launched |
| Thinking Machines | Inkling 2 Medium/Large | not released |

The watchlist is reviewed monthly. A row moves to the target table only when weights are
downloadable under a license that permits local use, and only with a pinned revision.

### 1b. Expected speed order (estimates; replaced by measurement)

The ordering is a **work-order aid**, not a performance claim. It is derived from active
bytes per token, the measured Inkling Small/Mini and V4-Flash cells, and the pinned fraction
each artifact gets at a ~100 GiB budget. Rows are ordered by the central estimate; GLM and
DeepSeek overlap, and the first decode arbitrates.

| rank | model | expected decode tok/s | basis |
|---|---|---|---|
| 1 | Qwen3.8-Flash-Next | **8–20** | 6B active vs Inkling Small's 17B measured at 2.67–7.54; 185.5 GB total gives a high pinned fraction; 36 of 48 layers are linear GDN |
| 2 | GLM-5.3-Flash | **4–10** | 18B active, 320 GB total (higher hit rate than Small's 564 GB); 34 of 45 layers are KDA |
| 3 | DeepSeek V4.1-Flash-1017 | **3–9** | 16B decode active; V4-Flash's 13B measured 9.17; 763B total plus 196B Engram; CED and CSA2 cut prefill, not decode |
| 4 | Kimi K3.1 Pro | **0.7–2** | 104B active, about 6× Inkling Small; 2.8T total; MXFP4 |
| — | Inkling 2 Small | 3.05–7.54 measured | already running: MXFP8 3.05, FP8+F4 7.54 warmed |
| — | Inkling 2 Mini | 21.78–27.04 measured | already running |

**Rule:** each model's first decode replaces its estimate with a measured number. If the
remaining order changes, only the model-specific arm re-queues: F3 is cut to serve GLM, and
DeepSeek and Kimi reuse its parts, so the shared work is already spent.

### 1c. Shared work and build order

Build each part once, with an oracle and a recorded red proof, before its first consumer.

| shared part | consumers | built in | first consumer |
|---|---|---|---|
| intake seam (pin, header census, artifact arithmetic, license, refusals) | all | F1 | Qwen and every new model |
| FP8 block-scaled GEMM/dequant | GLM, DeepSeek, Qwen, Inkling | F3.1 | GLM |
| Gated DSA plus sparse-indexer scaffold (score, block-max, top-k, gather, policy hook) | GLM, Kimi; Qwen QSA and DeepSeek CSA2 plug in | F3.2 | GLM |
| mHC fused residual | GLM, DeepSeek | F3.3 | GLM |
| MXFP4/FP4 expert GEMM/dequant | DeepSeek, Kimi, Inkling | F5 | DeepSeek |
| per-model KV and pin policy | all | F3.4 | GLM |
| KDA linear attention | GLM, Kimi | **reuse shipped K3** | GLM |
| tokenizer/chat/doors helpers | per family | per model | Qwen |
| speculative-decode harness | DeepSeek DSpark, Qwen MTP, GLM MTP | F8 | after all decode |

Why this order:

- **Qwen first, immediately.** It is the only target whose kernels, anchor, converter and
  chat encoding already exist (S0–S5). The critical path is `crates/engine/src/qwen/**`, the
  smoke and the S7 battery; no shared work gates it. It is also the fastest expected, so the
  first working new model is also the cheapest.
- **GLM second.** Its new surface is the smallest of the three new architectures: KDA is
  already shipped for K3, and DSA, mHC and FP8 are the exact parts DeepSeek and Kimi need.
  Building them here makes the next two models cheaper.
- **DeepSeek third.** Reuses FP8, mHC and the indexer scaffold; it builds the 4-bit expert
  path for its FP4 experts, which Kimi then reuses, and its own work is CED, CSA2 mode
  policy, Engram prefetch, DSpark and FP4 KV.
- **Kimi fourth.** Slowest expected, so it is last; it reuses KDA, MXFP4 and the DSA
  scaffold, and its own work is the 2.8T artifact, tokenizer/chat and the K3.1 DSA variant.
- **Inkling already runs at 16,128.** F7 adds the 1M kernel/KV work; its measured speeds anchor the table.

### 1d. The 1M requirement (owner decision 2026-11-27)

**Every model must pass a 1M-context gate before its milestone closes.** First decode is
"up"; 1M is "done". The gate is not "the door accepts `--ctx 1048576`": it is a 1M-token
prefill, a decode, a quality comparison and a memory witness (§6). The table is the path,
not a claim.

| model | 1M path | KV at 1M | kernel work |
|---|---|---|---|
| Qwen3.8-Flash-Next | YaRN extension from 262K native; QSA long-context kernel | ~23 GiB f32 (12 QSA layers, 2 KV heads × 256) | chunked/blocked QSA; the indexer budget already bounds top-k |
| GLM-5.3-Flash | native 1M; 11 DSA + 34 KDA | ~7 GiB fp8 + 16 GiB residual | chunked prefill; DSA top-k 2048 |
| DeepSeek V4.1-Flash | native 1M; CED + CSA2 + FP4 KV | ~2 GiB (890 B/token plus indexer/SWA) + 20 GiB residual | chunked prefill; CSA2 modes |
| Kimi K3.1 Pro | native 1M; 24 Gated DSA + 69 KDA | ~33 GiB bf16 + 28 GiB residual | chunked Gated DSA; KV dtype chosen by the quality gate |
| Inkling 2 Small/Mini | native 1M; 8/4 full layers + sliding 1024 | Small: 8 full × 8 KiB/token = 64 GiB f32 (16 GiB fp8); Mini: 4 full = 32 GiB f32 (8 GiB fp8) | per-layer KV sizing (window vs full), KV dtype, and chunked sink-attend beyond 16,128 |

The two current hard caps are removed by this requirement, not worked around: K3's
`ATTEND_MAX_KV = 8192` and Inkling's `SINK_ATTEND_MAX_N = 16,128` are single-launch LDS
limits. K3.1's Gated DSA kernel and Inkling's full-layer attention must be chunked
(split-K/online softmax) from the start. Inkling's current scratch allocates `ctx` KV for all
48/24 cache layers; the 1M path sizes sliding layers at the 1,024 window and only full
layers at `ctx`. The gate records prefill and decode time; it does not claim a tok/s.

## 2. What the frontier changed, as an architecture ledger

| mechanism | first seen here | targets | engine consequence |
|---|---|---|---|
| Causal Encoder–Decoder | DeepSeek V4.1 | DeepSeek | decoder KV projected from encoder hidden state; prefill path is not the decode path |
| CSA2 cross-layer KV and index reuse | DeepSeek V4.1 | DeepSeek | per-layer static mode (Full/Reindex/Reuse); sharing is a checkpoint property, not a runtime guess |
| Hierarchical sparse indexer | DeepSeek V4.1 | DeepSeek | first Full layer builds a block candidate pool; later indexers score only that pool |
| SWA bounded replay | DeepSeek V4.1 | DeepSeek | replay `n_win` tokens instead of persisting SWA KV; a storage/compute trade with its own gate |
| FP4 main KV | DeepSeek V4.1 | DeepSeek | QAT checkpoint; dequantize before attention; per-model KV dtype, never a global switch |
| Engram conditional memory | DeepSeek V4.1 | DeepSeek | 196B sparsely addressed parameters; deterministic prefetch is the only way this streams |
| DSpark speculative decode | DeepSeek V4.1 | DeepSeek | draft is part of the checkpoint family; acceptance and end-to-end gain must be measured |
| KDA linear attention | Kimi K3 | Kimi, GLM | already shipped for K3; reuse, do not re-port |
| Gated DSA | Kimi K3.1, GLM-5.3 | Kimi, GLM | sparse selection with an output gate; new kernel and census row |
| mHC residual mixing | DeepSeek V4.1, GLM-5.3 | DeepSeek, GLM | 4 residual streams; fusion changes activation traffic |
| QSA block-sparse attention | Qwen3.8 | Qwen | block-level selection; indexer budget 512 blocks / 2048 tokens |
| GDN hybrid + gated residual | Qwen3.8 | Qwen | already shipped through S3; arm is the gap |
| n-gram embeddings | Qwen3.8 | Qwen | 51B table, deterministic hash addressing, offload candidate |
| NVFP4 / MXFP8 / FP8 + MXFP4 | Inkling 2 | Inkling | shipped: Small is MXFP8 (first-party), Mini NVFP4/MXFP8, both now support FP8 trunk + MXFP4 experts |
| looped layers | Inkling 2 Mini | Inkling | the same block runs multiple passes with a message gate |

## 3. State of the art in local execution: what this plan adopts

### 3.1 Weight precision

Adopt the vendor format the checkpoint ships. The evidence says precision is the highest
leverage and the least portable:

- The Inference Engineering Pareto Atlas (arXiv 2609.17863, 2026-09-15) measured FP8 weights
  retaining **99.4%** of baseline GSM8K accuracy at **0.61–0.65×** latency across L4, A100
  and H100, and FP8 appeared in three of four regime winners. AWQ 4-bit reached 0.34×
  latency but lost **5.9%** strict GSM8K accuracy.
- The same study found a **naive FP8 KV cache answered none of 200 questions correctly**
  while maintaining normal throughput. Speed without a quality gate is not a result.
- The Apple Silicon study (arXiv 2610.31479, 2026-10-22) found quantization gains vary by
  machine and that optimizations do not transfer as properties of the technique alone.

Decision: consume FP8 block-scaled, MXFP4, MXFP8 and NVFP4 as shipped. Do not introduce a
second quantization pipeline (GGUF K-quants, AWQ) for the engine. Every new format gets a
bit-identity or tolerance gate before an arm can use it.

### 3.2 KV cache

The frontier is moving KV from a per-model afterthought to a checkpoint-defined object:
DeepSeek V4.1 ships FP4 main KV and cross-layer reuse; Kimi and GLM use linear-attention
state plus sparse full-attention KV; Qwen uses GDN recurrent state plus QSA blocks.

Decision: KV dtype and layout are **per-model**, declared in the config census and charged
in the pin floor. FP4 KV is admitted only for a checkpoint trained with it (DeepSeek V4.1)
and only behind a quality gate. The VeriCache result (arXiv 2605.17613: compressed-KV draft
plus full-KV verification, up to 4× throughput with identical outputs) is the model for any
lossy KV experiment: exactness is the acceptance criterion, not the compression ratio.

### 3.3 Speculative decoding

The local evidence is mixed and hardware-specific:

- The Pareto Atlas measured n-gram speculative decoding at **0.90–0.98×** baseline on its
  stack: no benefit.
- The Apple Silicon study found **all** evaluated speculative-decoding methods reduced
  throughput on Apple, despite 2–6× server-GPU reports.
- BitNest (arXiv 2610.02800) gets 1.48–1.61× on 7–8B edge models with a bit-nested draft
  that shares the target's physical weights, at 95.2% acceptance.

Decision: native drafts only (DSpark, MTP, DFlash), default **off** until an acceptance
histogram and an end-to-end release-profile gain are measured on gfx1151. A self-spec draft
that shares weights is preferred over a second model because the memory budget is the
binding constraint.

### 3.4 Attention and prefill

Sparse and linear attention are now architecture, not an optimization toggle. CSA2's
three modes, QSA's block budget, KDA's recurrent state and DSA's top-k are checkpoint
properties. Implement them as declared, and gate each mode separately. CED and SWA bounded
replay are prefill/decode trades that must be measured: a bounded replay that recomputes
`n_win` tokens can beat persisting SWA KV on NVMe, but only if the recompute is cheaper than
the read.

### 3.5 Streaming and memory

The box has 128 GB of unified LPDDR5X and NVMe behind it. Every target except Inkling 2 Mini
is bigger than memory, so the existing design — expert-granular streaming overlapped with
compute — remains the only viable one. Two new offload surfaces are deterministic and
therefore prefetchable: Qwen's n-gram table and DeepSeek's Engram table. Both are indexed by
token identity, so their reads can be issued before the layer that consumes them.

Third-party Strix Halo reports (a 35B MoE at ~51 tok/s, MTP +70% on dense, Vulkan vs ROCm)
are **leads, not baselines**. They are recorded here only to justify measuring the same
mechanisms on this box; no number from them is citable.

## 4. The standard intake ladder (I0–I7)

Every model, refresh or new architecture, runs the same ladder. A refresh may skip a stage
only with a written argument that the stage's subject did not change, and the argument is a
row in the model's milestone table.

| stage | artifact | exit gate | red proof |
|---|---|---|---|
| I0 pin | config, generation config, tokenizer files, safetensors header; license; revision SHA; artifact byte count | every vendored byte is hash-gated and recomputed from the live file; the artifact size is derived from the header, not from parameter counts | flip one hash digit; delete one required aux file |
| I1 anchor | first-party reference at tiny widths with the real structure; ≥2 weight draws; goldens; defect matrix ≥2 rows per named class | every defect reddens its declared bucket and holds upstream; tolerances come from fp32/fp64 floors on both draws, measured before the kernel | delete one EXPECT_GREEN row; move a tolerance under its floor |
| I2 kernels | launchers with oracle rows; bit-identity against a C reference or the first-party CPU oracle; ≥64 seeds; aliasing cases | census both ends: every launcher has an oracle or a milestone-named deferral; every Rust kernel has a bit-identity row | swap two operands; rename a launcher so its row names something absent |
| I3 artifact | converter with exact `ensure!` counts; name census; resident/routed split; pin floor | counts consumed and emitted match the header; the pin floor charges KV, vision, draft and scratch | drop one `ensure!`; remove one tensor from the census |
| I4 doors | legality cells, CLI/serve/bench refusals, tokenizer and chat encoding | every refusal quotes its own fragment; chat cases are id-pinned against the first-party encoder | change a fragment; mis-frame one chat turn |
| I5 arm | engine arm; prefill and decode; greedy parity window; ppl cells; A-vs-A determinism | greedy ids token-identical to the first-party reference on a pinned prompt; ppl inside a pre-registered band; determinism measured with its control | reverse the argmax tie-break; truncate one stream |
| I6 gates | smoke, docs registry, jscpd, CodeScene, line caps | the full battery is green and every row has a recorded red proof | remove an INDEX row; plant a 5-line clone |
| I7 closeout | verdict, INDEX row, census, deferrals | N/N/0 or every deferral milestone-named; no bare TODO | delete a deferral row while the gap exists |

## 5. Milestones, in expected-speed order

The order is a **work order**, not a performance claim: fastest expected first, with the
shared substrate cut to what the next model needs. Qwen starts first because it is the only
target whose kernels, anchor, converter and chat encoding already exist — the critical path
is the arm alone. F3 runs in parallel and never gates F2.

**"Up" is not "done."** A model is up at its first token-identical decode; it is done only
when the 1M-context gate in §1d and §6 is green. A model's milestone stays open until then,
and the next model's closeout does not start until the current 1M gate passes. Shared
substrate may still land early because it is reused.

### F0 — retire, register, watch (this change)

Scope: the five docs move to `status: obsoleted`; this plan is registered; the target and
watch tables above are dated; no code. Exit: the docs registry test is green and the INDEX
has exactly one row for this file. Red proof: remove this file's INDEX row; the registry
test names it.

### F1 — the intake seam

One reusable path for pin, header census, artifact-size arithmetic, license record and
legality refusal, so the next release is a data change rather than a new port. Exit: a
synthetic checkpoint and a real safetensors index both produce an exact census; a missing
aux file is refused by name; the artifact-size prediction reconciles to a header. Red proofs:
a one-byte header edit; a removed tensor row; a license field left empty.

### F2 — Qwen3.8-Flash-Next arm (the retired S6–S8, re-based)

Scope: `crates/engine/src/qwen/**`, `tests/smoke-qwen.sh`, the S7 battery and S8 closeout.
The kernels, anchor, converter and chat encoding already exist. Exit: greedy ids identical
to the first-party reference; the parity window and ppl cells run; A-vs-A determinism is
measured before any dNLL claim; smoke is green; the legality rows flip atomically. This is
the first TTFD baseline. The milestone does not close until the 1M gate is green: YaRN from
the 262K native window, a 1M prefill/decode, and ppl/needle parity inside the pre-registered
band. Red proofs: a wrong smoke fragment; a missing reference id; a truncated determinism
stream; a truncated 1M positional index.

### F3 — shared substrate (minimum for GLM; reused by DeepSeek and Kimi)

Build these once, in this order. Each lands with an oracle and a bit-identity row before its
first consumer; nothing here blocks F2.

1. **FP8 block-scaled GEMM/dequant** — GLM-5.3-Flash, DeepSeek V4.1, Qwen (already has its
   S3 kernels), Inkling (already has an FP8 trunk). First consumer: GLM.
2. **Gated DSA plus sparse-indexer scaffold** — score, block-max, top-k, gather and a
   per-layer policy hook. GLM and Kimi use it directly; Qwen's QSA and DeepSeek's CSA2 plug
   into it with their own policy data. First consumer: GLM.
3. **mHC fused residual** — GLM and DeepSeek. First consumer: GLM.
4. **Per-model KV and pin policy** — every model; charges KV, linear-attention state,
   vision, draft, SWA replay scratch, and the ctx-scaled allocation class. First consumer:
   GLM.

KDA is **reused from the shipped K3 arm**, not rebuilt; the MXFP4/FP4 expert path is built
with DeepSeek (F5) and reused by Kimi. Exit: each shared kernel has its own test and recorded
red proof; no shared part merges without a consumer in the same milestone. Red proofs: FP8
block scale off by one; DSA top-k off by one; mHC stream count wrong; a pin floor that
accepts an allocation it cannot honour.

### F4 — GLM-5.3-Flash (second model up)

Scope: 34 KDA + 11 Gated DSA layers, mHC, FP8 block GEMM, no-RoPE positional path, vision,
arm, gates. Uses F3 items 1–4; KDA is the shipped K3 kernel. Exit: anchor and defect
matrix; kernel bit-identity; artifact census; parity; ppl; determinism; smoke; the 1M gate
(1M prefill/decode, ppl/needle in band, memory witness). Red proofs: DSA layer order
swapped; mHC stream count wrong; FP8 scale block size wrong; a truncated 1M DSA candidate
pool.

### F5 — DeepSeek V4.1-Flash-1017 (third model up)

Scope: the largest intake — CED encoder/decoder split, CSA2 three modes, hierarchical
indexer, FP4 KV, SWA bounded replay, Single-Pass mHC, Engram, DSpark, vision, arm, gates.
Reuses F3 items 1, 3 and 4 and the sparse-indexer scaffold; the new work is CED, CSA2's mode
policy, Engram's deterministic prefetch, DSpark's draft/verify, FP4 KV, the MXFP4/FP4
expert GEMM/dequant, and chunked prefill. Exit: every mode has its own oracle and
bit-identity row; the FP4 KV path is exact against a dequantized reference; the pin floor
charges Engram, KV, SWA replay scratch and the DSpark draft; greedy parity; ppl; determinism;
smoke; the 1M gate (1M prefill/decode, ppl/needle in band, memory witness). Red proofs: a
Reuse-mode layer that recomputes instead of reusing; an Engram table lookup off by one; a KV
scale applied per tensor instead of per 16 channels; an MXFP4/FP4 group shifted by one; a
CSA2 candidate pool truncated at the 1M boundary.

### F6 — Kimi K3.1 Pro (fourth model up; slowest expected)

Scope: artifact, tokenizer, chat encoding, the Gated DSA variant and census row, arm, gates.
Reuses KDA, the F5 MXFP4/FP4 expert path and the F3 sparse-indexer scaffold; the new work is
the 2.8T artifact, the K3.1 tokenizer/chat path, and the DSA configuration that replaces K3's
Gated MLA. Exit: DSA bit-identity; 2.8T artifact census; greedy parity; ppl; determinism;
smoke; the 1M gate (1M prefill/decode with chunked Gated DSA, bf16/fp8 KV, ppl/needle in
band, memory witness). The K3 arm's `ATTEND_MAX_KV = 8192` cap is not inherited: the DSA
kernel is chunked from the start. The K3 checkpoint's retired items are **not** resumed:
chat encoding is re-derived from K3.1's first-party source, and the ctx-scaled allocation
class is charged in the new pin floor. Red proofs: DSA index top-k off by one; a swapped
gate; a census row deleted; a DSA selection truncated at the 1M boundary.

### F7 — Inkling 2 at 1M (long-context work, not a port)

Inkling 2 is already shipped at `7182854` at **16,128** (Mini bf16/MXFP8/NVFP4, Small MXFP8,
FP8 trunk + MXFP4 experts, 17/17 Small smoke). The 1M requirement adds three changes, none of
which is a model port:

1. **Per-layer KV sizing.** The current scratch allocates `ctx` K/V for all 48/24 cache
   layers. At 1M that is 384/192 GiB f32 and cannot fit. Sliding layers need only the 1,024
   window; only the 8 (Small) / 4 (Mini) full-attention layers need `ctx`.
2. **KV dtype.** With the split, Small's 8 full layers at 1M are 64 GiB f32 / 16 GiB fp8;
   Mini's 4 are 32 GiB f32 / 8 GiB fp8. Choose the dtype with the quality gate, not by
   default.
3. **Chunked full-layer attention.** `sink_attend` refuses past 16,128; full layers at 1M
   need split-K/online-softmax chunking. Sliding layers stay at the 1,024 window.

Exit: 1M prefill + decode on Small and Mini, ppl/needle inside the band, memory witness
clean, existing smokes green. Red proofs: a sliding layer sized at `ctx` (the witness blows);
a full layer sized at the window (quality collapses); a chunk boundary off by one.

### F8 — optimization and cadence closeout

Four cross-family tracks, each with a pre-registered acceptance and a default-off rule:

1. **KV policy** — per-model dtype, cross-layer reuse, bounded replay; exactness gate.
2. **Speculative decode** — native drafts only; acceptance histogram plus end-to-end gain.
3. **Streaming scheduler** — prefetch for n-gram/Engram tables; hit/miss and ms/miss.
4. **Prefill/decode split** — CED, QSA and DSA block budgets; measure the crossover.

No track changes a default until its measurement is witnessed and its red proof recorded.
Exit: 5/5 families pass the 1M-context gate and decode token-identical to their first-party
references on pinned prompts; every model's ppl and determinism rows are green; the watch
table has a dated review; the
release labels from the retired v3 plan are either claimed with evidence or explicitly
dropped; the docs registry, jscpd, CodeScene and line caps are green. The first measured TTFD
becomes the baseline for the cadence target.

## 6. Acceptance battery, per model

| gate | what green means | red proof |
|---|---|---|
| pin | vendored bytes hash and recompute; license and revision recorded | one hash digit flipped |
| census | consumed and emitted tensor counts equal the header, both ends | one tensor row removed |
| anchor | every named defect reddens its bucket and holds upstream | one EXPECT_GREEN row deleted |
| kernel | bit-identical to the C reference or the CPU oracle, ≥64 seeds, aliasing included | one operand swapped |
| artifact | converter counts exact; resident/routed split exact; pin floor charges every term | one `ensure!` dropped |
| doors | every refusal quotes its own fragment; chat ids pinned | one fragment changed |
| parity | greedy ids identical on the pinned window; flips only at measured near-ties | argmax tie reversed |
| ppl | paired dNLL inside the pre-registered band; INCONCLUSIVE is not a pass | a 1% NLL shift injected |
| determinism | A-vs-A with its control; the comparator reddens on a changed id and a truncated stream | stream truncated |
| 1M context | 1M-token prefill and decode; ppl/needle inside the pre-registered band; KV and scratch inside the pin floor; witness clean | truncate the positional/KV index by one block; the gate reddens |
| smoke | CLI door-to-door: refusals, bench ids, serve readiness, SSE, shutdown | one cell's expected id changed |
| docs | registry, INDEX, jscpd, CodeScene, line caps | one INDEX row removed |

## 7. Fit arithmetic and the pin floor

Planning envelope only; every number is replaced at I0 by a header-derived byte count.

| model | stored bytes, planning bound | fits resident on 128 GB? | streamed share |
|---|---|---|---|
| DeepSeek V4.1-Flash-1017 | ~0.5–1.0 B/param over 763B, so ~380–760 GB | no | experts, Engram, vision |
| GLM-5.3-Flash | ~320 GB at FP8 plus scales | no | experts |
| Kimi K3.1 Pro | ~1.4 TB at MXFP4 | no | experts |
| Qwen3.8-Flash-Next | 185,502,232,570 B measured in tree | no | n-gram table, experts |
| Inkling 2 Small | ~600 GB MXFP8 source; FP8+MXFP4 variants measured on hot NVMe | no | experts |
| Inkling 2 Mini | 19,526,019,168 B source (19.5 GB); fits | **yes** | none |

The pin floor must charge, per model: resident weights, KV at the model's dtype and
layout, linear-attention state, vision encoder and image tokens, the draft model, SWA replay
scratch, and the ctx-scaled allocation class the retired K3 doc found uncharged (a 288 MiB
`hipMalloc` that failed after the pin at `--ctx 4096`). A door that accepts a budget the box
cannot honour is a defect, not a warning.

## 8. Cadence and the keep-up metric

- **TTFD** — time from a pinned release's weights being available to the first greedy
  token-identical decode on this box. The first measurement is F2; the target for a refresh
  is ≤ 10 working days, and for a new architecture ≤ 8 weeks. These are targets, not claims,
  until the first TTFD is recorded.
- **Coverage** — the fraction of the five families' latest open Flash-tier checkpoints that
  decode here. Target 5/5 at F8.
- **Watch review** — monthly; each review records its date and every row it moved. A release
  with weights available and no triage row is a process failure, not a green.

## 9. Non-claims and standing risks

- **No v3.0.0.** The cut plan is obsoleted; no tag, notes or gate-day claim is made.
- **No third-party tok/s.** Vendor and community numbers are leads. Only witnessed
  release-profile measurements on this box are citable.
- **Vision is not v1 for a new arm** unless the text path requires image tokens. The vision
  encoder is a separate milestone with its own anchor and bit-identity rows.
- **Speculative decode is off by default** until its acceptance and end-to-end gain are
  measured. The Apple Silicon and Pareto Atlas results are the reason.
- **FP4 KV is per-checkpoint**, only where QAT trained it, and only behind an exactness or
  pre-registered quality gate.
- **Licenses matter.** Kimi is custom and Qwen is gated; weights are not vendored, only
  headers and configs.
- **Churn is the default.** Targets are pinned revisions, never "latest". A new release can
  invalidate a target; the watch table is how that becomes a visible edit.
- **GLM nondeterminism remains an accepted operational limit**, carried as a release label
  and a determinism gate, not as an open investigation.
- **CodeScene debt is tree-wide.** New arms must not add to the standing below-10 files;
  the gate is the same one every package already carries.

## 10. Worklog

- **2026-11-27** — plan registered. Five retired docs moved to `status: obsoleted`; target
  table pinned to the releases above; F0 complete pending the docs test. No model code
  written under this plan yet.
- **2026-11-27 (order)** — milestones re-queued by expected decode speed: Qwen first, then
  GLM, DeepSeek and Kimi. F3 is cut to GLM's minimum shared substrate (FP8, DSA/indexer,
  mHC, KV/pin); the MXFP4/FP4 expert path moves to DeepSeek's F5 and is reused by Kimi.
- **2026-11-27 (1M)** — owner requirement: every model must pass a 1M-context gate before
  its milestone closes. Added §1d, per-model 1M exits, the acceptance-battery row, and
  rewrote F7 as Inkling 1M (per-layer KV sizing, KV dtype, chunked sink-attend).
- **2026-11-27 (re-base)** — the first pass evaluated `87dfa70`; fresh `origin/main` is
  `7182854` (2026-11-26), which adds the Inkling Waves 1–3 commit. Inkling 2 is already
  shipped text-only (Mini bf16/MXFP8/NVFP4, Small MXFP8, FP8 trunk + MXFP4 experts, 17/17
  Small smoke), so F6 is a regression-and-watch row and the target table carries a
  `tree today` column. The fresh commit is not merged into this branch.
