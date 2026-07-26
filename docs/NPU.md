# rivoli — NPU offload plan (spec-drafter + DSA indexer)

Status: **proposed, gated. Not started.** The NPU is live on this node — XRT 2.21,
amdxdna 0.1, firmware 1.1.2, `/dev/accel/accel0`, Strix Halo NPU at `0000:c8:00.1` — and
100% idle during decode. This plan covers exactly two concurrency candidates: **(1)** a
speculative-decode drafter on the NPU, and **(4)** the DSA sparse indexer on the NPU. Both
are **gated on a cheap analytical/measurement milestone before any NPU code is written** —
because one of them (spec) has already landed on main as a *definitive negative result*,
and the other is a long-context-only play with zero benefit at the contexts we bench today.

## What the NPU is, and what it can't do here

The XDNA NPU is a **dense INT8/BF16 matmul dataflow engine** on the same unified LPDDR5 as
the GPU (no copies) but sharing the one **256 GB/s bus**. Two hard constraints follow:

- **It is bad at random gather.** The int3-vq MoE dot (`cb[idx]` L1 gather) is its worst
  case, so the NPU **cannot** attack the 210 ms `moe-gpu` phase directly. Its only role is
  to run *dense* work concurrently so the GPU spends more of its time on the gather-bound
  MoE.
- **Concurrency shares bandwidth.** Every "hidden" claim below is a **hypothesis until the
  concurrent-bandwidth is measured** — same discipline as `PERF.md`'s ISA section. On the
  shared bus, NPU weight-reads can steal from the GPU critical path.

## What has already landed on main (this plan must not re-litigate it)

- **Decode is compute-bound warm, bandwidth-bound cold.** route ~104 ms (post the ISA
  tranche), moe-gpu ~210 ms, fetch ~94% hidden. Best config hybrid+lru 2.85 tok/s.
- **Speculative decode / MTP is a closed, *verified-correct* negative result**
  (`deadend/mtp`, `--spec` opt-in). It was built to M5 (tree drafting) with prefetched
  batched verify and **loses in both regimes**:
  - **cold:** the batch reads the **union** of the drafted positions' experts →
    **+16% bytes/token** (184 vs 159 misses/tok); on the NVMe floor the round-count drop
    can't overcome it.
  - **warm:** the **draft itself is a full extra MoE-layer of GPU compute** — the GLM MTP
    head at layer 78 *is* a complete MoE layer (MLA attn + 256 routed experts + shared +
    indexer), not a light head.
  - Verdict on record: needs **>70% accept AND high union overlap**, unreachable with the
    1-token MTP head. Measured 43–84% accept.
- **DSA indexer is implemented on the GPU** (`--attn dsa`, `indexer.hip`, `IndexerPin`):
  a small extra attention that scores past tokens for a sparse top-k selection, engaged
  past the 2048 index_topk. Weights: `wk`/`wq_b` fp8, `weights_proj`/`k_norm` bf16→f32.
- An **int8-MTP** container exists (`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`), and
  the layer-78 MTP head is in the GLM-5.2-FP8 checkpoint (download ~complete).

---

## Candidate 1 — NPU speculative-decode drafter

**The idea:** draft K tokens on the NPU while the GPU verifies, so the draft cost leaves
the GPU entirely — attacking the *warm-regime* loss mechanism (draft = a full extra MoE
layer). **The idea's problem is on record:** the NPU changes **only** the draft cost. It
does **not** touch the two things that actually sank spec:

1. **The verify union (+16% bytes)** is paid by the **GPU** regardless of where drafting
   runs — it is intrinsic to batched MoE verification, and it was the cold-regime killer.
2. **The natural drafter is a MoE layer** (the GLM MTP head) — the NPU's *worst* workload
   (gather). Running it on the NPU is not just hard, it's the wrong engine. The only
   NPU-shaped drafter is a **separate small dense model**, which (a) is a new model to
   source/distill, and (b) drafts a *different* expert distribution → **worse** union
   overlap on verify, making #1 worse, not better.

So candidate 1 is a **long shot**, and the honest move is to price it out on paper before
spending a device slot.

### De-risking milestones

- **M0 — free-draft analysis (NO NPU, ~1 day).** On `deadend/mtp` (the verified `--spec`
  loop), re-derive throughput with **draft cost set to zero** — the NPU idealization.
  Reuse the captured accept rates and union byte-counts. **Gate:** does a *free* draft beat
  hybrid+lru 2.85 in *any* regime? The recorded numbers suggest **no** (warm: even removing
  the draft, the verify-union +16% at ≤84% accept nets a loss; cold: +16% on the bandwidth
  floor is drafter-independent). **If M0 fails, candidate 1 is dead — stop here.** This
  milestone is the whole point: it kills or greenlights the biggest-ceiling idea for the
  price of a spreadsheet, and it respects the landed negative result instead of rebuilding
  into it.
- **M1 — accept-rate ceiling (NO NPU, only if M0 is marginal).** The union penalty is only
  survivable at high accept + high overlap. Measure whether a **wider/better drafter** (not
  the 1-token MTP head) could reach >70% accept with a shared-expert-union — the
  `index_share_for_mtp_iteration` path hints the checkpoint already shares selection. If no
  reachable drafter clears the bar, stop.
- **M2 — NPU dense drafter feasibility (needs the toolchain spike, § shared).** Only if
  M0/M1 clear: pick an NPU-shaped **dense** drafter (small int8 model; the int8-MTP
  container is the starting point *if* its head can be made dense), compile it for the NPU,
  and validate greedy-equivalent drafting against a CPU reference.
- **M3 — pipelined draft‖verify.** Overlap the NPU draft of round r+1 with the GPU verify
  of round r; measure end-to-end vs 2.85 and vs the concurrent-bandwidth cost. **Gate:**
  net win after the shared bus contention.

**Expected outcome, stated honestly:** M0 most likely closes this. Documented so the next
person doesn't reopen it without new information (a genuinely dense high-accept drafter, or
a verify path that avoids the union — neither exists today).

---

## Candidate 4 — DSA indexer on the NPU (long-context)

**The idea:** the DSA lightning indexer is a small, mostly-dense attention that scores past
tokens for the sparse top-k. Its cost **grows with context** (it scans the KV history),
while the current benches are short so it barely registers. At long context it becomes a
real fraction of `route`, and it is **NPU-shaped** (`weights_proj`/`k_norm` already bf16;
only `wk`/`wq_b` are fp8 → a small bf16/int8 copy). Run it on the NPU **concurrent with the
GPU's dense attention projections**; its top-k selection then feeds the GPU flash-attend.
This hides the indexer behind the projections at long context. **Zero benefit at short
context** — this is explicitly a long-context lever, aligned with the `mla_latent_attend`
follow-up in `PERF.md`.

### De-risking milestones

- **M0 — indexer cost vs context (NO NPU, ~½ day).** Profile the GPU indexer kernels
  (`index_score`/`layernorm`/`index_head_route`) across context 128 → 8k → 32k. **Gate:**
  does the indexer grow to a *material* fraction of `route` (say >5 ms) within the context
  range we actually care about? If it stays <2 ms even at 32k, it is not worth an NPU
  handoff — stop.
- **M1 — dependency & overlap window (NO NPU).** Confirm the concurrency is real: the
  indexer needs the layer's query (from the q-projection) and produces the top-k that the
  attend consumes. The exploitable overlap is **indexer(NPU) ∥ {kv-projection, KV append}
  (GPU)**, then attend(GPU) waits on the selection. Measure that window — if the indexer is
  longer than the GPU work it would overlap, it stalls the attend and there is no win.
- **M2 — indexer on the NPU (needs the toolchain spike, § shared).** Port
  `index_score`/`layernorm`/`route` to the NPU (bf16 `weights_proj`/`k_norm` native; a
  bf16/int8 copy of `wk`/`wq_b`). Validate the **selection is equivalent** to the GPU
  indexer on a fixed long-context input (the top-k set, not bit-exact scores).
- **M3 — overlap in the engine.** Wire indexer(NPU) ∥ projections(GPU) in the `--attn dsa`
  path; measure `route` at 8k/32k vs GPU-only DSA, net of concurrent-bandwidth. **Gate:**
  faster route at long context, output-equivalent.

**Expected outcome:** viable *if* M0 shows the indexer actually grows to matter. It is the
safer of the two candidates (no dead-end baggage, NPU-shaped weights, clean-ish overlap),
but its payoff is entirely in the long-context regime we do not yet bench.

---

## Shared infrastructure & cross-cutting risks

- **Toolchain spike (blocks M2 of both).** XRT runs, but building an **xclbin** needs a
  kernel compiler (MLIR-AIE / Peano / IREE-for-AIE) whose presence is **unverified**.
  First shared step: compile + run **one int8 GEMM** on the NPU, measure (a) NPU GEMV
  latency for our shapes and (b) **GPU-vs-NPU concurrent bandwidth** on the shared bus.
  Both candidates' M3 gates depend on (b).
- **Bandwidth contention** — the shared 256 GB/s. MoE is L1-gather-bound (~24% of bus) so
  there is likely DRAM headroom during `moe-gpu`, but `route` is more bandwidth-hungry;
  measure, never assume.
- **Handoff latency** — coarse for #1 (a whole draft round) amortizes it; #4 overlaps
  *within* the attention phase, per layer, so its handoff must be sub-ms.
- **NPU-format weights** — extra bf16/int8 copies (indexer `wk`/`wq_b`; any drafter).
- **amdxdna 0.1 / firmware 1.1.2** — early stack; expect rough edges, and that the shared
  `gpu_sched` scheduler between amdgpu and amdxdna may itself serialize handoffs (verify in
  the toolchain spike).

## De-risking sequence (cheap gates first, NPU code last)

1. **1-M0** (free-draft analysis) — cheapest, decides the biggest-ceiling idea. Do first;
   likely closes candidate 1.
2. **4-M0** (indexer-cost-vs-context) — decides whether candidate 4 is worth any NPU work.
3. **Toolchain spike** — only if 1-M0 or 4-M0 clears. Establishes the whole NPU path and
   the concurrent-bandwidth number both need.
4. Then the surviving candidate's build milestones (M2/M3).

No NPU kernel is written until steps 1–2 have justified it on paper. The most probable
result of this plan is a *documented decision*: candidate 1 stays closed, candidate 4
proceeds only when long context is on the roadmap.
