# rivoli — NPU offload plan: the DSA indexer

Status: **M0 and M1 MEASURED (2026-07-26). M0 clears ≥4k. M1 clears via the decoupled
window only. A no-NPU change takes half the prize and gates the rest.** The NPU is live on
this node — XRT 2.21, amdxdna 0.1, firmware 1.1.2, `/dev/accel/accel0`, Strix Halo NPU at
`0000:c8:00.1` — and 100% idle during decode. This plan is **exclusive to one workload: the
DSA sparse indexer.** The other NPU candidate (a spec-decode drafter) was analysed and set
aside — spec decode is a verified-correct *negative result* on this engine (`deadend/mtp`),
and the NPU changes only the draft cost, not the verify-union penalty that sank it.

## The finding, in five lines

1. **The prize is real above `index_topk` = 2048 and larger than the microbench said.** Measured
   in-engine: **8.6 ms/token at 2.4k context, 11.6 ms/token at 5.2k**, against a ≥0.68
   ms/token handoff floor. Below `index_topk` = 2048 it is ~0.32 ms/token, *less than the
   handoff*, so the offload would make the engine slower there.
2. **Half to three-quarters of the prize is not GPU work.** The host score-D2H + CPU top-k
   + row upload is **52% of the prize at 2.4k and 60% at 5.2k, measured**, rising with
   context. A device top-k kernel recovers it exactly — no approximation, no quality gate,
   no NPU, no toolchain.
3. **The exact-overlap design (window 1) fails on engine data, at every context measured.**
   The window is ~369 µs in-engine; the indexer plus its selection step needs 409 µs at
   2.4k and 552 µs at 5.2k. It is not close, and the gap widens with context.
4. **The decoupled design (window 2) is established only to 5.2k.** It clears there
   comfortably. At 32k it clears on one reading of the MoE window and fails on the other,
   and the favourable reading rests on treating ~1.4 ms/layer as GPU idle when the only
   *measured* idle is 0.12 ms/layer. It also misses the 3 dense-MLP full layers at 32k
   under every reading. And where it does depend on fetch-stall idle, it is coupled to the
   engine staying fetch-bound — which the residency programme is actively trying to end.
5. **The measured wall under `--attn dsa` is 391 ms/token at 2.4k and 438 ms/token at
   5.2k** — the estimate this document previously used (~430 ms) was close, but the
   estimate's *reasoning* was wrong (see below). The prize is **2.2% of wall at 2.4k and
   2.7% at 5.2k**, measured, rising to ~9% at 32k by extrapolation.

**A methodological warning that belongs at the top.** Window 1's verdict moved **four
times** under review: 22 µs (window under-scoped) → 158 µs (weights cache-resident, `q_b`
reading 372 GB/s — above the bus) → 291 µs (rotated) → *does not clear* (the budget omitted
the cost of producing the selection). Three defects in the instrument and one in the
analysis built on it, every one found by review rather than by the rig, and every one moving
the same conclusion. **Every validity control passed throughout.** A passing control shows
the rig is measuring something; it never shows it is measuring the thing you named it after.
The in-engine run below is the first instrument in this document that could have contradicted
the microbench, and on one of the two quantities it did — by 2×.

## The value proposition: hidden, not fast

**The NPU indexer does not need to beat the GPU or the CPU.** Its only job is to run
concurrently, so the GPU stops spending time on it:

> `route` drops by the indexer's former GPU cost, **provided the NPU indexer finishes
> inside a window of GPU work it can overlap.**

A slower NPU is fine — the NPU is otherwise idle, so anything it absorbs off the critical
path is free wall-clock *if and only if it is hidden*. So the gate is **hideability**, not
throughput: is there a stretch of GPU work, independent of the indexer's output, at least
as long as the NPU's latency plus two handoffs plus the selection step?

## What the indexer is (as it exists on the GPU today)

The DSA lightning indexer (`--attn dsa`, `indexer.hip`, `IndexerPin`) is a small, mostly
**dense** attention that scores past tokens and emits a **top-k token selection** (past the
2048 `index_topk`) that the main flash-attend then restricts to. It is **NPU-shaped**:
`weights_proj` and `k_norm` are already bf16→f32, and only `wk` / `wq_b` are fp8 (a small
bf16/int8 copy is the one added resident). Kernels an NPU port must cover: `layernorm`,
`rope` ×2 (key, then the 32 query heads), `index_append`, `index_score`, and three GEMVs
(`wk`, `wq_b`, `weights_proj`).

Four corrections to the earlier draft, all from reading the engine:

- **`index_head_route` and `index_pool_push` do not run on `--attn dsa`.** Both are gated on
  `active_heads.is_some()` in `dsa_select_layer`, which is `Some` only under `--attn misa`.
  They were named as M0 targets on the dsa path; they are dead code there.
- **The indexer runs on 21 layers per token, not 78.** `indexer_types` is 21 `full` / 57
  `shared`, and a shared layer relaunches nothing. Every per-token figure below is ×21, and
  the handoff count is **×21/token**, not the "×75" assumed — a 3.6× relief.
- **At or below `index_topk` the indexer barely exists.** At `nt <= 2048`
  `dsa_select_layer` appends the key and returns dense *before* the scoring path and
  *before* the host round-trip, so the host cost there is **zero**, not small.
- **A large part of "the indexer" is not on the GPU** — the D2H, the CPU top-k and the row
  upload, per full layer, are the biggest single component past 8k.

## The concurrency structure — where the hiding window is

```
q_proj ──▶ indexer(scores past keys) ──▶ top-k selection ──▶ attend(selected) ──▶ MoE
                    ▲                                              ▲
              needs q + KV history                         needs the selection
```

The attend cannot start until the selection lands, so hiding the indexer means overlapping
it with GPU work that does **not** need the selection. Two windows:

1. **Exact / tight — the rest of attention phase 1.** The indexer's inputs (`xn` post
   input_layernorm, `qr` post q_a_ln) are ready after the *second* rmsnorm, and the first
   consumer of the selection is `launch_attend`. Everything between is available:

   ```
   gemv_fp8(q_b) → gemv_fp8(kv_a) → rmsnorm(kv_a_ln) → rope(key) → rope(query, 64 heads)
     → append_kv → mla_absorb_fp8(kv_b) → gather_rope        [then attend, which needs it]
   ```

   The earlier draft scoped this to "`kv_proj` + KV-append" — 22.6 µs, under a tenth of it,
   omitting `q_b` (33.5 MB) and `mla_absorb_fp8` (14.7 MB). **Measured whole, with cold
   weights: 291.25 µs.** Under-scoping a window is not conservative; it produces a confident
   false negative.

2. **Decoupled / large — stale-or-periodic selection (approximation).** If the attend may
   use a selection from a **one-step-stale query**, the indexer for layer `l` at token `t+1`
   runs during layer `l`'s MLP at token `t` — inputs already in hand, nothing downstream
   needing the result until the next token reaches that layer. **Measured: 1261.9 µs per MoE
   layer.** Note the unit: the earlier draft's "MoE (210 ms)" is the whole token across 75
   MoE layers, but the handoff is per full layer, so the window is **1.26 ms, not 210 ms** —
   it still clears, but the margin is 167× smaller than the plan argued from.

## MEASURED — M0 and M1

Instrument: `examples/indexer_bench.rs`, gfx1151, sole tenant, 2026-07-26. Same launch
wrappers and GLM-5.2 dims as the engine, every constant sourced to a manifest key. **Rows,
controls and methodology are recorded in `benchmarks.md`, "DSA indexer round"**; this
section is the interpretation. Figures below are from that round's final run unless a
superseded run is named explicitly.

### Handoff floor

`gpustream::tests::signal_resolves_and_latency`, run 2026-07-26: **16.18 µs** per
host↔device stream-signal round-trip. A hidden indexer needs two per full layer (release the
GPU, resume it), so the floor is **≥32.4 µs/layer = ≥0.68 ms/token over 21 layers**.

**HYPOTHESIS, flagged per docs/PERF.md.** This is a host↔GPU HIP host-func round-trip
standing in for a host↔NPU one, which goes through XRT submit/fence — a different mechanism,
cheaper or dearer unknown, and first on the spike's list. The test prints the figure and
asserts only `us < 1000`, so **nothing in the tree records it**; it is quoted here with its
date because no artifact holds it.

### M0 — the prize

Per full layer: key path (`gemv_fp8` wk + `layernorm` + `rope` + `index_append`) **15.32 µs**
pipelined, **20.48 µs** with the engine's per-layer sync — these four kernels are launch-
bound, not GPU-bound, so read the sub-2048 row as overhead rather than as work an NPU
absorbs. Score path, only above `index_topk`: `gemv_fp8` wq_b **78.27 µs** (107.2 GB/s),
`rope` query 4.90, `gemv_f32` weights_proj 34.74 (22.6 GB/s — 32 output rows, grid-starved),
**fixed subtotal 117.91 µs**; then `index_score`, 4.4 → 239.1 µs, the only part that scales.

Per token (×21 full layers), against the ≥0.68 ms/token handoff floor:

| context | indexer/layer µs | indexer GPU ms/tok | host round-trip ms/tok | **total prize** | vs floor |
|---:|---:|---:|---:|---:|---|
| 128 | 15.3 | 0.322 | — *(not run)* | **0.322** | **0.47× — FAILS** |
| 2048 | 15.3 | 0.322 | — *(not run)* | **0.322** | **0.47× — FAILS** |
| 4096 | 165.2 | 3.468 | 3.364 | **6.832** | 10.0× — clears |
| 8192 | 192.4 | 4.041 | 3.842 | **7.883** | 11.6× — clears |
| 16384 | 248.2 | 5.213 | 7.417 | **12.630** | 18.6× — clears |
| 32768 | 372.3 | 7.819 | 11.625 | **19.444** | 28.6× — clears |

**M0 gate: CLEARS at ≥4k, FAILS at or below `index_topk` = 2048** — the plan's own kill
condition firing early rather than late; there the handoff costs 2.1× what the indexer does.

Three things in that table were not in the plan:

- **The host round-trip is the largest single component past 8k**, 3.36 → 11.63 ms/token,
  more than the entire GPU-side indexer. Per layer at 32k it is 554 µs — 1.9× the whole of
  window 1. *These are the isolated-microbench figures; the in-engine section below measures
  them **2.0–2.2× higher**, so treat this row as a floor.*
- **On the GPU side the dominant cost is fixed, not context-scaling.** `wq_b` is 1.64
  ms/token regardless of context, and `index_score` does not overtake it until ~**11k**
  (linear fit through the 8k/16k rows). At 8k the context-independent work is 133/192 =
  **69%** of the per-layer GPU cost.
- **The prize as a share of wall** is left to the in-engine section below, which measures
  the denominator instead of estimating it. The estimate this section previously carried
  (~430 ms/token) happened to land close to the measured 391–438 ms, but it was built by
  adding an `mla_attend` nr=2048 term to a `--attn dense` benchmark, and the in-engine
  decomposition shows that reasoning was wrong: under dsa the attention phase is *flat* in
  context, and the term that actually grows is the indexer itself. A right answer from a
  wrong mechanism is the failure mode `docs/PERF.md` opens with.

### In-engine confirmation — two runs, and the microbench does not survive intact

`--attn dsa --mode hybrid --cache-policy lru --max-mem 115 -bench 48`, GLM-5.2 full
artifact, sole tenant, 2026-07-26/27. Two profile buckets added to `dsa_select_layer`
(`idx_gpu_ns` via a HIP-event pair, `idx_host_ns` via a clock started *after* the sync that
path already pays), so neither adds a join. Prompts differ because reaching 5.2k needs more
text than reaching 2.4k — which turns out to matter.

| | run A | run B |
|---|---:|---:|
| prompt tokens / mean nt during decode | 2432 / **2456** | 5185 / **5209** |
| **wall ms/token** | **391** | **438** |
| `route` (post-selection attention + host routing) | 156 | 158 |
| `moe` wall (gpu) | 201 (192) | 242 (232) |
| **indexer GPU** ms/tok — µs/layer | **4.1** — **194.9** | **4.6** — **218.1** |
| **indexer host** ms/tok — µs/layer | **4.5** — **214.2** | **7.0** — **334.1** |
| scoring layers / token | **21.0** | **21.0** |
| tok/s · expert hit% · miss/tok | 2.56 · 81.4 · 111.4 | 2.28 · 76.9 · 138.9 |
| residual (wall − route − moe − indexer) | 25.4 | 26.4 |

**Confirmed.**

- **21.000 scoring layers per token** — the 21-full-of-78 reading of `indexer_types` was a
  prediction from the manifest and the engine reproduces it. Every ×21 in this document
  rests on it. (The counter is now printed at 3 decimals for exactly this reason; the runs
  above were logged at 1 decimal, i.e. 21.0 means [20.95, 21.05), so this is confirmed to
  the printed precision and not to the unit.)
- **`route` is flat: 156 → 158 ms across a 2.1× context increase.** This is the first direct
  evidence that DSA caps the attend at `index_topk` rows, and it means the post-selection
  attention phase does not grow with context. The indexer is the only part that does.
- **The wall accounts for all but ~25 ms**, twice. Stated carefully, because it is easy to
  over-read: `wall`, `route` and `moe` are printed at 1 ms resolution, so the residual is
  **25 ± 2 and 26 ± 2 ms** and its stability is inside the inputs' own error bar — that
  agreement is not itself evidence. And the residual is an **unbucketed remainder, not "the
  tail"**: `lm_head` alone is a 951 MB int8 GEMV, ~5 ms at the engine's demonstrated 190
  GB/s, and rmsnorm + argmax are sub-millisecond, so ~20 ms belongs to something not named.
  A candidate nothing measures: between the `route` clock closing and the `moe` clock
  opening, every MoE layer runs `submit_layer` (cache lookup + I/O submit for 9 experts),
  the weight normalisation, descriptor construction and two H2D uploads — ×75/token, in no
  bucket. Also note `route_ns` accrues only on the 75 MoE layers, so the 3 dense layers'
  attention lands in `moe_wall_ns`; the bucket names are approximations of their contents.

**Refuted, and this is the important half.**

- **The microbench under-predicts the indexer's GPU span by 27%** — 1.271× at 2.4k and
  1.264× at 5.2k, two independent contexts agreeing to 0.6%. **The size is solid; the
  mechanism is NOT established.** The obvious candidate is launch bubbles between the 8
  kernels, but the rig's own measurement of that overhead — 5.16 µs for a *four*-kernel
  group with one sync — predicts under 10 µs, not the observed ~41 µs, so it under-predicts
  by 4–8×. A second candidate the instrument cannot exclude: the span's endpoints are
  barrier packets whose own dispatch cost falls inside it. Recorded as measured-but-
  unexplained, which is the status `benchmarks.md` gave the previous 27% surplus and the
  status this one has earned. **Do not read the earlier 27% as corroboration** — that
  figure is a ratio of two *deltas* between arms of one binary, in which any fixed
  per-launch overhead cancels exactly, so it cannot share a bubble-count cause. Two
  unexplained under-predictions of similar size, not one explained one.
  **`idx_gpu_ns` is a timeline SPAN, not a sum of kernel times; it is the right quantity
  for "what would offloading remove from the timeline" and must not be differenced against
  a microbench per-kernel total without this correction.**
- **The host round-trip is 2.0–2.2× the isolated microbench** (214.2 vs ~97 µs at 2.4k;
  334.1 vs ~166 µs at 5.2k). So the synthetic score distribution *under*-stated it, and the
  bracket this document previously quoted — "3.4–11.6 ms/token at 32k" — is therefore
  presumptively ~2× too low, since it was built entirely from the microbench. (The measured
  engine values, 4.5 and 7.0 ms/token, sit *inside* that bracket — but at contexts 6–13×
  shorter, so the like-for-like comparison is the µs/layer one above, not the bracket.)
  Mechanism
  unresolved between a harder real distribution and in-situ CPU-cache contention from the
  expert streamer moving 1.7–2.1 GB/token through host memory; both push the same way, so
  the direction is safe even though the split is not known.
- **Wall is NOT cleanly comparable between these two runs.** The +47 ms is dominated by +41
  ms of MoE, alongside expert hit% falling 81.4 → 76.9 and ms/miss rising 76 → 134. Run B's
  prompt is the first 12,000 characters of run B's — wholly contained in it — so prompt
  content and context length are perfectly confounded and **n = 2 cannot say which caused
  it** — longer context changing the routing distribution, different subject matter routing
  elsewhere, and the longer run accumulating more NVMe/page-cache pressure all fit. The
  honest statement is that the MoE term moved for reasons this pair cannot separate.
  **This applies to every context sweep this engine can run, not just to 32k** — reaching a
  longer context always requires more text. And it cuts both ways: the extrapolated 32k
  wall below is built from `route` and `moe` taken from these same two runs, so **that
  denominator inherits the same confound** and the "~9% of wall" figure is softer than its
  components suggest.

**Extrapolation, flagged as such — and it does not have one form.** The engine's host cost
fits `107.2 + 0.04355·nt` µs exactly through both points; the functional form is justified a
priori (an O(n) index-vector build, an O(n) quickselect, a fixed 2048-element sort), not
chosen to fit. The GPU span is worse posed: the correction over the microbench sum is either
**multiplicative** (×1.267) or **additive** (+41.5 µs at 2.4k, +45.5 at 5.2k), and two points
2.1× apart cannot discriminate — 1.271 vs 1.264 is equally consistent with a fixed offset.
Since the mechanism is unestablished (above), both are carried:

| context | indexer GPU (additive → multiplicative) | host round-trip | **prize** | host share |
|---:|---:|---:|---:|---:|
| 2456 *(measured)* | 4.1 ms/tok | 4.5 ms/tok | **8.6** | **52%** |
| 5209 *(measured)* | 4.6 | 7.0 | **11.6** | **60%** |
| 8192 *(extrapolated)* | 5.0 – 5.1 | 9.7 | **14.7 – 14.9** | 65 – 66% |
| 16384 *(extrapolated)* | 6.1 – 6.6 | 17.2 | **23.3 – 23.8** | 72 – 74% |
| 32768 *(extrapolated)* | 8.7 – 9.9 | 32.2 | **40.9 – 42.1** | 76 – 79% |

At 32k that is a prize of ~41 ms against a wall of roughly 157 (route) + ~25 (unbucketed) +
41 (indexer) + 200–242 (moe) = **423–465 ms**, i.e. **~9%** — double the 4.5% this document
estimated from the microbench, and three-quarters of it is the CPU top-k. The denominator
carries the confound noted above.

### M1 — the windows

| window | measured | verdict |
|---|---:|---|
| 1 — selection-independent phase 1 (exact) | **291.25 µs** | **does not clear as the engine stands** |
| 2 — MoE compute floor + the engine's *measured* idle | **~1380 µs** | clears ≤5.2k; **fails at 32k** |
| 2 — the engine's whole MoE wall (`moe` ÷ 75), idle *inferred* | **2680–3230 µs** | clears everywhere, on an unmeasured inference |
| 2 — dense fp8 SwiGLU MLP, the 3 dense full layers (no MoE phase) | **1174.67 µs** | clears ≤5.2k; **fails at 32k** |

Window 1 decomposed: `gemv_fp8 q_b` **213.35** (4 rotating copies, 157 GB/s) + kv_a/rmsnorm/
rope/append 22.62 + `rope` query 7.45 + `mla_absorb_fp8` **45.64** (4 rotating copies) +
`gather_rope` 2.20. *Two residual optimisms, both favourable:* 4 copies cycle 193 MB against
a 32 MB MALL so ~1 read in 6 may still hit, where the engine holds 78 distinct weights; and
`kv_a` is a single replayed 3.5 MB copy, comfortably MALL-resident. A fully cold window
would be **larger** than 291 µs, widening window 1's margin. 291 µs is a lower bound.

**The budget must hold three things, not two: the indexer, two handoffs, and the production
of the selection itself.** The offload delivers 2048 row indices, not scores — M2's own gate
is selection equivalence — so wherever that top-k runs it is serial with the NPU compute and
sits inside the window.

**Window 1 was never measured in-engine.** No bucket covers it. The figure below is the
microbench's 291.25 µs scaled by the indexer's measured 1.267× span ratio, on the assumption
that the same unexplained overhead applies — 369 µs, budget 337 µs after handoffs. That
borrowing is *conservative for the verdict*: it scales the window up, making a FAILS harder
to reach.

**Window 2's usable size is bounded below by measurement and above by inference, and the two
differ by 10×.** What the engine demonstrates is `moe_wall − compute_gpu` = 9–10 ms/token =
**0.12–0.13 ms/layer** of idle beyond the compute-stream span. The larger figure this
document previously used (~1.4 ms/layer, from the engine's MoE wall minus the microbench's
9-expert batch) is an *inference across different launch shapes*: the engine issues nine
separate single-expert launches gated on host fetch signals, the microbench issues one
batched 9-expert launch, so the gap may be launch-shape cost that keeps the GPU busy rather
than idle an NPU could use. `gpu.rs` says as much about `compute_gpu` being an upper bound.

| context | indexer + selection | window 1 (~337 µs, scaled) | window 2, measured-idle floor (~1350 µs) | window 2, inferred ceiling (~3200 µs) |
|---:|---:|---|---|---|
| 2456 *(measured)* | 409 µs | **FAILS** | clears 3.3× | clears 7.8× |
| 5209 *(measured)* | 552 µs | **FAILS** | clears 2.4× | clears 5.8× |
| 32768 *(extrapolated)* | ~2000 µs | **FAILS ~6×** | **FAILS** | clears 1.6× |

**Window 1 fails at both measured contexts**, by 72 µs at 2.4k and 215 µs at 5.2k, widening
with context because the selection step grows. The earlier microbench-only reading —
"clears at 4k–8k" — was an artifact of lacking the 27% span correction and of costing the
selection in isolation rather than in situ.

**Window 2 is established only to 5.2k.** At 32k its verdict flips on which bound you take,
and the favourable one is an inference the run does not support. Note also that **the 3
dense-MLP full layers (0/1/2) have no MoE phase at all** — their window is the dense SwiGLU,
1174.67 − 32.4 = 1142 µs, which the ~2000 µs 32k requirement exceeds under every figure. So
even the optimistic reading of window 2 covers 18 of 21 full layers at 32k, not 21.

**With a device top-k the question dissolves**: the selection term collapses, the 32k
requirement drops from ~2000 µs to ~470 µs, and window 2 clears against the *measured-idle
floor* — no inference required — with 2.8× to spare, dense layers included. That is a third
independent reason to build it first.

Restated in the unit the toolchain spike can test — the NPU must sustain, per layer:

| context | indexer bytes/layer | to hide in window 2 |
|---:|---:|---:|
| 4096 | 11.0 MB | 10.2 GB/s |
| 8192 | 12.1 MB | 11.4 GB/s |
| 16384 | 14.2 MB | 15.8 GB/s |
| 32768 | 18.4 MB | 26.3 GB/s |

### The bandwidth risk — not binding for window 2, unmeasured across the NPU boundary

The plan's stated central risk: hideability is a **bandwidth** claim, because the indexer
reads KV history and contends with the work it overlaps.

Measured: the MoE batch reads 138.0 MB in 1261.9 µs = **109.4 GB/s** — 43% of the 256 GB/s
theoretical peak and **56% of the 193.8 GB/s this rig actually achieves** on a big streaming
fp8 read. The achievable figure is the honest denominator, leaving ~84 GB/s, against which
the indexer's 10–26 GB/s is **12–31%**. Not binding.

**But note what this argument is.** It is arithmetic over GPU-side measurements. It says
nothing about NPU DMA efficiency, whether amdxdna reaches DRAM at comparable bandwidth, or
coherency/snoop traffic between two engines on the same pages. Two caveats sharpen it: the
microbench window holds all 9 experts resident, whereas the engine's real window also
carries the hidden expert stream (~2.0 GB/token, 94% overlapped ≈ 6–12 GB/s), so read ~72–78
GB/s of headroom; and the rig's VQ indices come from a repeating 4 KiB block, so its
codebook gather is friendlier to L1 than the engine's — shortening the window (conservative
for hideability) and understating the bus share it commits.

A direct GPU∥GPU concurrency probe was built, run and deleted; it could not answer the
question, for reasons recorded in benchmarks.md. **A GPU∥GPU experiment cannot stand in for
a GPU∥NPU one.**

### What was NOT measured

- **Any context beyond 5.2k, in-engine.** The 8k/16k/32k rows are extrapolated from two
  measured points through a functional form the algorithm justifies, not measured. A 32k run
  costs **~3.6 h of sole-tenant prefill** (measured prefill rate: 0.34–0.39 s/token,
  token-by-token — there is no batched prefill; that is `docs/PERF.md` Path A, unbuilt) and,
  more decisively, **it would not produce a clean wall number anyway**: reaching 32k needs a
  different prompt, and prompt changes move expert hit% and therefore the MoE phase by more
  than context moves the indexer. The two runs here differ by 47 ms of wall of which 41 ms is
  MoE residency. A third confounded point is not worth 3.6 h; a device top-k plus a re-run
  would be.
- **An un-instrumented control run.** The 391/438 ms wall is reported by a build carrying
  the new buckets (42 event enqueues + 21 clock reads per token). The perturbation is
  bounded by argument at ~0.05%, not by measurement; closing it costs one re-run of run A
  with the buckets compiled out.
- **Per-layer GPU-busy time inside the MoE phase** — the difference between window 2's
  measured floor and its inferred ceiling, and the thing the 32k verdict turns on.
- **A device top-k's cost** — which now decides whether the exact path exists at all, and
  which by these numbers is 52–76% of the prize.
- **Whether the host round-trip's 2× in-situ penalty is distribution or cache contention.**
  Separating them needs a dump of the engine's real score array; both push the same way, so
  the reported figure is safe, but the *cause* determines whether a device top-k removes all
  of it or only most.
- **Bandwidth contention during either window**, and anything across the amdxdna boundary.
- **Anything NPU.** No NPU code was written or run, per the plan's own sequencing.

## Milestones

- ~~**M0**~~ **DONE**, microbench + in-engine. Clears above `index_topk`; fails at or below it.
- ~~**M1**~~ **DONE**, microbench + in-engine. Window 1 fails at every measured context;
  window 2 clears, at 32k only against the engine's real MoE wall rather than its compute
  floor.

- **Device top-k (NO NPU). DO THIS FIRST.** MEASURED in-engine at **52% of the prize at
  2.4k and 60% at 5.2k**, extrapolating to ~76% at 32k — the host round-trip is 4.5 → 7.0
  ms/token measured, all of it GPU-idle. One kernel: same selection, exact, no quality gate,
  no staleness, no toolchain, no per-layer handoff. Three independent reasons it comes
  first: it is the largest single win available; it is what would make window 1 (and so any
  exact path) even arguable; and it is what decouples window 2's long-context viability from
  the engine staying fetch-bound. It also removes a per-layer `device_sync` from the decode
  path and the distribution-dependence that makes the largest line item uncertain.

- **Toolchain spike — lead with the handoff.** Compile + run one dense bf16 kernel on the
  NPU via whatever flow exists (MLIR-AIE / Peano / IREE — presence unverified; `xrt-smi`
  running ≠ an xclbin compiler). Measure, in order: **(1) handoff round-trip**; (2) NPU
  sustained bandwidth at the indexer's shapes against the 10–26 GB/s window-2 bar; (3)
  GPU↔NPU concurrent bandwidth.

- **M1a — stale-selection quality (NO NPU). THE GATE FOR EVERYTHING DOWNSTREAM, AND IT IS
  NOT YET IMPLEMENTABLE.** Window 1 has failed, so the decoupled window is the only one
  left, and it is reachable only through a stale selection — which means every remaining
  milestone (the spike, M2, M3) is behind a quality result nobody can produce today.
  **Nothing in the engine currently computes a selection from a one-step-stale query**:
  `dsa_select_layer` derives it from the current token's `qr` and consumes it in the same
  layer. Producing one requires an engine change — a per-full-layer selection cache in
  `DeviceIndexer`, written at token `t` and read at `t+1`, behind a flag — plus a decision
  about what layers 0/1/2 do on the first token. That change is unscoped, unestimated, and
  is a prerequisite for the measurement, not part of it. **This is a blocker, not a
  scheduling detail.** Simulate a
  1-step-stale / periodic-refresh selection offline against the exact selection; measure the
  perplexity delta on `tests/ppl-corpus-5000.txt` via `--ppl`.

- **M2 — indexer on the NPU.** Port the kernels; validate the **selected set** matches the
  GPU indexer on a fixed long-context input (not bit-exact scores).

- **M3 — overlap in the engine.** Wire `indexer(NPU)` ∥ window 2 in `--attn dsa`. Measure
  `route` vs GPU-only DSA, **net of concurrent-bandwidth contention**. **Gate: `route` drops
  and output is within the M1a budget.**

A second, smaller no-NPU lever, marked as what it is: **`gemv_fp8 wq_b` runs at 107.2 GB/s
where `o_proj` reaches 193.8 GB/s** on the same rig and run — a measured 1.8× gap on a 1.64
ms/token context-independent cost. **HYPOTHESIS, not a measurement:** that the gap is
recoverable. `wq_b` [4096×2048] stays on the wave-per-row `gemv_fp8` path while `o_proj` at
i_dim=16384 dispatches to `gemv_fp8_splitk` — different kernels, different grid regimes, so
the gap may be structural. Per docs/PERF.md, read the ISA before booking the device.

## Risks

- **There is no exact path today.** Window 1 fails on the selection step, so everything is
  downstream of either a fast device top-k or M1a's approximation — and M1a is unscoped.
- **Window 1's viability, if the device top-k is built, then rests on an unmeasured handoff
  cost.** At ~16 µs a side there is room at 4k–8k; at ~64 µs a side there is none anywhere.
- **Bandwidth** is not binding for window 2 (12–31% of real headroom) but is **unmeasured
  across the amdxdna/amdgpu boundary**, which is where it would bite.
- **The prize is 2.2% of wall at 2.4k and 2.7% at 5.2k, measured**, extrapolating to ~9% at
  32k — and 52–76% of it is recoverable without the NPU. This is the number the go/no-go
  rests on, and the NPU's share of it shrinks as context grows.
- **This analysis has been wrong four times on one number**, three times in the instrument
  and once in the reasoning, always understating what had to fit or overstating the room.
- **amdxdna 0.1 / firmware 1.1.2** — early stack; expect rough edges.

The framing to hold onto: this is not a "make the indexer fast" project but a "make the
indexer disappear from the GPU timeline" project. What the measurements change is that half
the indexer is not on the GPU timeline at all — it is a CPU top-k the GPU is blocked behind
— and that fixing *that*, with no NPU involved, is both the largest single win available and
the precondition for the exact overlap the plan wanted.
