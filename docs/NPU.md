# rivoli — NPU offload plan: the DSA indexer

Status: **proposed, gated. Not started.** The NPU is live on this node — XRT 2.21, amdxdna
0.1, firmware 1.1.2, `/dev/accel/accel0`, Strix Halo NPU at `0000:c8:00.1` — and 100% idle
during decode. This plan is **exclusive to one workload: the DSA sparse indexer.** The
other NPU candidate (a spec-decode drafter) was analysed and set aside — spec decode is a
verified-correct *negative result* on this engine (`deadend/mtp`), and the NPU changes only
the draft cost, not the verify-union penalty that sank it; see the prior revision of this
file / that branch. This document does not revisit it.

## The value proposition: hidden, not fast

**The NPU indexer does not need to beat the GPU or the CPU at the indexer.** Its only job
is to run the indexer *concurrently*, so the GPU stops spending time on it. Success is:

> `route` drops by the indexer's former GPU cost, **provided the NPU indexer finishes
> inside a window of GPU work it can overlap.**

A slower NPU is fine. The NPU is otherwise idle, so anything it absorbs off the GPU
critical path is free wall-clock — *if and only if it is hidden*. So the whole plan turns
on one question, and it is a **hideability** question, not a **throughput** question:

> Is there a stretch of GPU work, independent of the indexer's output, that is at least as
> long as the NPU indexer's latency (plus the two handoffs)?

That inverts the usual kernel gate. We are not asking "can the NPU do this faster." We are
asking "can we arrange for the GPU to be usefully busy while the NPU does this."

## What the indexer is (as it exists on the GPU today)

The DSA lightning indexer (`--attn dsa`, `indexer.hip`, `IndexerPin`) is a small, mostly
**dense** attention that scores past tokens and emits a **top-k token selection** (past the
2048 `index_topk`) that the main flash-attend then restricts to. It is **NPU-shaped**:

- `weights_proj` and `k_norm` are already **bf16→f32** — native for the NPU.
- only `wk` / `wq_b` are **fp8** → a small bf16 (or int8) copy is the one added resident.
- kernels: `layernorm`, `index_score`, `index_head_route`, `index_pool_push` — dense
  matmul + reductions, no int3-vq gather (the thing the NPU is bad at).

Its cost **grows with context** (it scans the KV history each step), so at short context it
is cheap and at long context it becomes a real fraction of `route` — which is *good* for
this plan two ways: the bigger it gets, the more GPU time offloading it removes, and it is
the same regime where the `mla_latent_attend` follow-up in `PERF.md` already wants
attention-phase work reduced.

## The concurrency structure — where the hiding window is

Within a layer the exact dependency is:

```
q_proj ──▶ indexer(scores past keys) ──▶ top-k selection ──▶ attend(selected) ──▶ MoE
                    ▲                                              ▲
              needs q + KV history                         needs the selection
```

The indexer sits **on the critical path into the attend** — the attend cannot start until
the selection lands. So hiding it means overlapping it with GPU work that does **not** need
the selection. Two windows, smallest-first:

1. **Exact / tight window — `kv_proj` + KV-append (same token).** The current token's
   kv-path (kv_a/kv_b → append to the KV cache) is needed for the attend but is independent
   of the *selection*. Run `indexer(NPU)` ∥ `{kv_proj, KV-append}(GPU)`; the attend consumes
   both. **Small window (~a few ms)** — hides only a short (short-context) indexer. Exact:
   the selection is computed from the current query, identical to today.

2. **Decoupled / large window — stale-or-periodic selection (approximation).** The top-k
   selection changes slowly across steps (the salient history is stable). If the attend may
   use a selection computed from a **one-step-stale query**, or **refreshed every N tokens**,
   the indexer leaves the critical path entirely and can overlap the **MoE (210 ms)** — the
   biggest GPU window there is. This hides an indexer of essentially *any* size, which is
   exactly what long context needs. It is an **approximation** and must clear a quality gate
   (the selection is a heuristic already; the question is whether staleness moves
   perplexity). This is the design that makes hideability scale with context.

Handoffs are **per layer (×75/token)**, so each must be sub-ms; the shared `gpu_sched`
between amdgpu and amdxdna may itself serialize them (verify in the toolchain spike).

## De-risking milestones — gated on hideability, not speed

- **M0 — indexer GPU cost vs context (NO NPU, ~½ day).** Profile `index_score` /
  `layernorm` / `index_head_route` / `index_pool_push` at context 128 → 8k → 32k. This is
  the **prize size**: the ms we remove from `route` by offloading, at each context. There is
  no minimum-speed bar here — any hideable positive amount is a win — but if it is
  sub-millisecond even at 32k, the handoff overhead will exceed it and the whole plan is
  moot. **Gate: indexer GPU cost > handoff overhead in the context range we care about.**

- **M1 — the hiding window (NO NPU).** Measure the two candidate windows: (1) `kv_proj` +
  KV-append duration (the exact window) and (2) the MoE duration (the decoupled window).
  Answer, per context: *is a window ≥ the indexer's latency available?* For window (1) at
  short context, likely yes; for long context the indexer outgrows it → window (2) is
  required, which means the approximation. **Gate: a window exists (possibly requiring the
  stale-selection design) that covers a realistic NPU indexer latency + 2 handoffs.**

- **M1a — stale-selection quality (NO NPU, only if window (2) is needed).** Simulate a
  1-step-stale / periodic-refresh selection offline against the exact selection; measure the
  perplexity delta on the fixed corpus. **Gate: quality delta within budget** (the selection
  is already a top-k heuristic; small staleness should be cheap, but prove it).

- **Toolchain spike (shared prerequisite for any NPU code).** Compile + run one dense bf16
  kernel on the NPU via whatever flow exists (MLIR-AIE / Peano / IREE — presence unverified;
  `xrt-smi` running ≠ an xclbin compiler). Measure: NPU latency for the indexer's shapes,
  **GPU↔NPU concurrent bandwidth** on the shared 256 GB/s bus, and **handoff round-trip**
  latency. These feed the M1 window comparison.

- **M2 — indexer on the NPU.** Port the indexer kernels (bf16 `weights_proj`/`k_norm`
  native; a bf16/int8 copy of `wk`/`wq_b`). Validate the **top-k selection is equivalent**
  to the GPU indexer on a fixed long-context input (the selected set, not bit-exact scores).

- **M3 — overlap in the engine.** Wire `indexer(NPU)` ∥ (window 1 or 2) in the `--attn dsa`
  path. Measure `route` vs GPU-only DSA, **net of concurrent-bandwidth contention**.
  **Gate: `route` drops (any amount) and output is equivalent** (exact for window 1; within
  the M1a quality budget for window 2).

## Risks specific to this workload

- **Bandwidth contention is the real hideability threat.** The indexer *reads the KV
  history*, which is bandwidth-hungry and grows with context — exactly when we most want it
  hidden. On the shared bus it competes with the GPU work it overlaps; the "hidden" claim is
  a bandwidth claim, not just a latency claim. Measure concurrent, never assume.
- **Per-layer handoff (×75).** Sub-ms required; `gpu_sched` serialization is the open
  question the toolchain spike must answer.
- **The stale-selection approximation** trades exactness for a large hiding window. Keep the
  exact (window-1) path as the always-correct default; the decoupled path is opt-in and
  quality-gated.
- **amdxdna 0.1 / firmware 1.1.2** — early stack; expect rough edges.

## Sequence (cheap gates first, NPU code last)

1. **M0** (indexer cost vs context) and **M1** (hiding-window measurement) — both NO-NPU,
   together decide whether a hideable win exists at all.
2. **M1a** if the long-context window needs the stale-selection approximation.
3. **Toolchain spike** — only if M0/M1 clear.
4. **M2 → M3** — build and wire, gated on `route` dropping net of bandwidth.

The framing to hold onto: this is not a "make the indexer fast" project. It is a "make the
indexer disappear from the GPU timeline" project, and it succeeds the moment the GPU is
doing something else useful while the NPU carries it — however slowly.
