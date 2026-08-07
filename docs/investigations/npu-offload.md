---
scope: glm
status: closed-negative
verdict: DSA indexer offload to the NPU: not worth it — M1a (measured 2026-08-07) closed the decoupled window for GLM's DSA indexer (verbatim 1-step-stale costs +0.89 nats; only the diagonal-patched variant is unmeasured). The device top-k it recommended shipped instead (−9.4 ms/token).
---

# rivoli — NPU offload plan: the DSA indexer

> **NAV — 57 KB. The answer is the next section, "The finding, in five lines".**
> Read that and stop unless you are implementing the offload. **One line:** the prize is
> real only above `index_topk` = 2048, half to three-quarters of it was never GPU work,
> and the part that was — a device top-k kernel — **is built and shipped** (−9.4 ms/token,
> selection bit-identical to the host over 10,752 real layers). The NPU itself remains
> unbuilt and the exact-overlap design was falsified on engine data — **and 2026-08-07 the
> decoupled design fell too, on quality (M1a: verbatim 1-step-stale, +0.89 nats)**.

Status: **M0 and M1 MEASURED (2026-07-26). The no-NPU device top-k is now WIRED and
MEASURED (2026-07-27): −9.4 ms/token, 2.1% of wall, selection exact.** M0 clears ≥4k. M1
clears via the decoupled window only, and everything past it still sits behind M1a, which
~~remains not implementable as written~~ (**2026-08-07: implemented behind `--features
stale-sel` + `--stale-sel`, and MEASURED the same day — verbatim 1-step-stale FAILS,
tail mean dNLL +0.89 nats, closing the decoupled window for GLM's DSA indexer; see the
MEASURED note under the M1a milestone**). The NPU is live on
this node — XRT 2.21, amdxdna 0.1, firmware 1.1.2, `/dev/accel/accel0`, Strix Halo NPU at
`0000:c8:00.1` — and 100% idle during decode. This plan is **exclusive to one workload: the
DSA sparse indexer.** The other NPU candidate (a spec-decode drafter) was analysed and set
aside — spec decode is a verified-correct *negative result* on this engine (`deadend/mtp`),
and the NPU changes only the draft cost, not the verify-union penalty that sank it.

## The finding, in five lines

1. **The prize is real above `index_topk` = 2048 and larger than the microbench said.** Measured
   in-engine: **8.6 ms/token at 2.4k context, 11.6 ms/token at 5.2k** (session 1; **session 2
   measured 13.1–15.3 ms/token at the same 2.4k context and identical flags** — the prize is
   not a stable quantity, see "Wired"), against a ≥0.68 ms/token handoff floor. Below `index_topk` = 2048 it is ~0.32 ms/token, *less than the
   handoff*, so the offload would make the engine slower there.
2. **Half to three-quarters of the prize is not GPU work.** The host score-D2H + CPU top-k
   + row upload is **52% of the prize at 2.4k and 60% at 5.2k, measured**, rising with
   context. A device top-k kernel recovers it exactly — no approximation, no quality gate,
   no NPU, no toolchain. **This is now BUILT, WIRED and MEASURED: −9.4 ms/token, 2.1% of
   wall, selection bit-identical to the host on 10,752 real layers** (see "Wired" below).
   The 52/60% shares are session 1; **session 2 put the host share at 69–73%** at the same
   context. The host half is not a stable quantity — 429–533 µs/layer in one session, 214
   in another — so every share figure in this document is softer than it looks.
3. **The exact-overlap design (window 1) fails on engine data, at every context measured.**
   The window is ~369 µs in-engine; the indexer plus its selection step needs 409 µs at
   2.4k and 552 µs at 5.2k. It is not close, and the gap widens with context.
4. **The decoupled design (window 2) is established only to 5.2k.** (**2026-08-07: the
   timing clearance is now moot — M1a measured verbatim 1-step-stale at +0.89 nats tail
   dNLL, so window 2 is closed on QUALITY for GLM's DSA indexer; see the MEASURED note
   under the M1a milestone.**) It clears there
   comfortably. At 32k it clears on one reading of the MoE window and fails on the other,
   and the favourable reading rests on treating ~1.4 ms/layer as GPU idle when the only
   *measured* idle is 0.12 ms/layer. It also misses the 3 dense-MLP full layers at 32k
   under every reading. And where it does depend on fetch-stall idle, it is coupled to the
   engine staying fetch-bound — which the residency programme is actively trying to end.
5. **The measured wall under `--attn dsa` is 391 ms/token at 2.4k and 438 ms/token at
   5.2k** — the estimate this document previously used (~430 ms) was close, but the
   estimate's *reasoning* was wrong (see below). The prize is **2.2% of wall at 2.4k and
   2.7% at 5.2k**, measured, rising to ~9% at 32k by extrapolation. **Session 2 measured
   447–452 ms wall and a 2.9–3.4% prize at the same 2.4k context and identical flags.** The
   15% wall difference between sessions is UNEXPLAINED, and every share-of-wall percentage
   in this document is divided by it.

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
wrappers and GLM-5.2 dims as the engine, every constant sourced to a manifest key.
**The rig is DELETED** (source at `77b5500:examples/indexer_bench.rs`): the in-engine
confirmation below both superseded it and refuted its central figure by 27%, and the
engine's own always-on `idx_gpu_ns` bucket now measures in situ what it approximated from
outside. (`idx_host_ns` went with the host selection arm — see "Wired" below; on the shipped
device path it was measuring zero.) Restore it only to re-derive a per-kernel decomposition; for
totals, read a run's PROFILE line. **Rows,
controls and methodology are recorded in `docs/measurement/benchmarks.md`, "DSA indexer round"**; this
section is the interpretation. Figures below are from that round's final run unless a
superseded run is named explicitly.

### Handoff floor

`gpustream::tests::signal_resolves_and_latency`, run 2026-07-26: **16.18 µs** per
host↔device stream-signal round-trip. A hidden indexer needs two per full layer (release the
GPU, resume it), so the floor is **≥32.4 µs/layer = ≥0.68 ms/token over 21 layers**.

**HYPOTHESIS, flagged per docs/measurement/perf-roadmap.md.** This is a host↔GPU HIP host-func round-trip
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
  wrong mechanism is the failure mode `docs/measurement/perf-roadmap.md` opens with.

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
  unexplained, which is the status `docs/measurement/benchmarks.md` gave the previous 27% surplus and the
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
  **This is a standing constraint on any future measurement here, not a quirk of this
  pair.** Varying context while holding content fixed does not happen by default — a longer
  prompt is a different prompt. Anyone measuring context-dependence in this engine has to
  construct it deliberately: the same text padded to several lengths, or several distinct
  prompts at matched length so content varies *within* a context point rather than between
  them, or a synthetic-KV harness that skips prefill entirely. Absent one of those, a wall
  or MoE difference between two context points is uninterpretable, and only `route` — which
  docs/measurement/benchmarks.md notes is structurally insulated from fetch variance — can be compared
  directly. And it cuts both ways: the extrapolated 32k
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

### The device top-k, measured — after a fixture artifact was removed

Kernel `index_topk`. Correctness gate `tests/kernel.rs::index_topk_matches_host_selection`
**passes**: bit-identical to `topk_into + sort_unstable` on all 10 cases, sentinel tail
included. Cost, host and device timed in the **same rig on the same buffer on the same
data** — the only comparison that may be divided (µs per full layer, host → device):

| nt | dense (few ties) | scattered (heavy ties, random order) | sorted-sparse (**artifact**) |
|---:|---:|---:|---:|
| 2456 | 86.6 → 35.8 — **2.42×** | 54.4 → 45.6 — **1.19×** | 28.8 → 45.3 — 0.64× |
| 5209 | 101.4 → 41.2 — **2.46×** | 74.9 → 59.7 — **1.25×** | 41.1 → 60.9 — 0.67× |
| 8192 | 126.4 → 52.7 — 2.40× | 96.6 → 82.7 — 1.17× | 46.8 → 79.2 — 0.59× |
| 16384 | 344.5 → 83.3 — 4.14× | 144.3 → 126.6 — 1.14× | 65.7 → 127.6 — 0.52× |
| 32768 | 578.1 → 157.6 — 3.67× | 191.8 → 215.0 — **0.89×** | 144.6 → 215.0 — 0.67× |

**The third column is a trap, and an earlier revision of this document fell in it.** It
reported "the device kernel is 1.6–1.9× slower than the CPU it was built to replace" from
that column alone, and attributed the gap to ties making quickselect cheaper. The real
cause was mostly the *fixture*: `topk_into` seeds its workspace with the identity
permutation and orders by (score desc, index asc), and that fixture's values descend from
index 0, so the identity **is** the sorted order — quickselect and the trailing sort both
got an already-sorted slice, their best case, which the device kernel cannot exploit.
`scattered` has the identical tie structure in random order and is the honest test. It
moves the ratio from 0.64× to 1.19×, so **roughly 1.8× of the claimed regression was
fixture, not ties.**

**Corrected finding.** The device kernel is faster than the host on every distribution
measured except tie-heavy at 32k. The margin is strongly distribution-dependent —
**2.4–4.1× on dense, 1.13–1.25× on tie-heavy, 0.89× at tie-heavy 32k** *(fixtures, not the
engine — the engine's own arrays measure 2.01×; see the retraction below)* — because ties
make quickselect cheaper *and* the radix histogram dearer (tied keys collide on one LDS
bin). It is never the regression the previous revision reported. Any single-number
speedup for this kernel is meaningless without naming the distribution.

**Which shape the engine sees is still unmeasured, and that is now the whole question.**
The in-engine host cost is 214.2 µs/layer at 2456 and 334.1 at 5209. For the engine to be
producing each shape, the in-situ penalty would have to be **2.5–3.3× (dense), 3.9–4.5×
(scattered), or 7.4–8.1× (sorted-sparse)**. Dense needs the smallest and is the only one
close to the ~2.1× penalty measured elsewhere in this document — but note that penalty is
itself the in-engine cost divided by a *dense* host microbench, so it is a reproducibility
check on that row rather than independent evidence, and the argument partly assumes its
conclusion. The honest position: dense-like is the best-supported guess, it is not
established, and the kernel's value ranges from **~2.4× down to break-even** across the
shapes still in play. *(Superseded twice over: the distribution was then measured — see the
next section, which retracts ~2.4× in favour of **2.01×** on the engine's own arrays — and
the kernel has since been wired and measured in-engine, below.)*

**What that does to the share-of-prize numbers.** Earlier text said the host round-trip is
"52–60% of the prize" and a device top-k "recovers it exactly". Recomputed from the
matched table: the kernel recovers **59% of the host cost on dense (≈31–36% of the prize)
and 16–20% on tie-heavy (≈8–12%)**. The 52–60% figure is the size of the *target*, not of
the win.

**That blocking measurement has since been taken — see the next section. The answer is
`dense`. But do not carry the ~2.4× forward: the next section measures the real arrays
directly at **2.01×**, and that is the figure of record.**

### The engine's actual score distribution — measured, and it refutes the premise

`RIVOLI_DUMP_SCORES` (a `trace`-gated dump of `index_score`'s raw output, added for this),
2026-07-27: **64 (layer, token) records, all 21 full layers, 131,202 token-scores** at
nt = 2049–2052. Characterised on its own terms first; the fixture comparison comes after.

> **The dump was deleted 2026-08-01.** `RIVOLI_DUMP_SCORES` was an env var, which
> `CLAUDE.md` forbids — an env var is invisible to `--help`, absent from the command line
> `benchmarks.md` records, and silently active in a build that looks stock. It served this
> investigation, which is **closed-negative with its one recommendation already shipped**
> (the device top-k, −9.4 ms/token), so the instrument had no remaining question. `src/gpu.rs`
> now reads zero env vars. **The distribution it captured is tabulated below and is the
> record**; if the premise is ever re-opened, re-add the dump behind a *feature and a flag*
> the way `--ppl`, `--pred-probe` and `--spans` are — that is the shape the rule asks for,
> and this is one of the two env-var instruments that motivated writing it down.

*Provenance:* `--features rocm,trace`, GLM-5.2 full artifact, `--attn dsa --mode hybrid
--cache-policy lru --max-mem 115 -bench 4`, 2432-token prompt, sole tenant. **The records
are all prefill-phase** — scoring begins at nt = 2049 and the budget exhausted immediately
after — which is also the reason the dump cannot have perturbed the run's own profile
numbers: `Profile::default()` resets the buckets after prefill, so nothing was written
during the measured decode window. That is a structural argument, not an empirical one; do
not read the trace run's agreement with the earlier non-trace runs (wall 388 vs 391 ms,
route 157 vs 156) as evidence of non-perturbation, because the instrument was idle by then.
One prompt is n = 1 on the axis most likely to vary.

| property | measured |
|---|---|
| exactly `0.0` | **0 of 131,202 (0.0000%)** |
| negative | **90.0%** |
| distinct values / nt | **0.9985 – 1.000**, uniform across layers 0–74 |
| per-layer heterogeneity | large — layer 1 is **100% non-negative** (median +126), layer 34 is 99.85% negative. The 90% is an aggregate, not a per-layer property |
| largest tie group | 1–2 entries (0.05–0.1% of the array) |
| tie group at the k-th boundary | **1 entry, in 0 of 64 records** |
| ordering | 47.4–53.5% of adjacent pairs descending; ~1000 ascending runs in ~2050 elements |
| spread | roughly −184 … +145; per-record median **−125.6 … +126.4** |

**The ReLU-sparse premise is refuted on every axis it asserted.** It predicted "a large
fraction of tokens score exactly 0.0"; the answer is *none*, not *few*. It assumed the
scores were non-negative ReLU'd sums; 90% are negative. It claimed the k-th boundary lands
inside a large tie group so the index tiebreak decides the bulk of the selection; the
boundary tie group is a single entry in every record.

**Why the premise was wrong, stated so it is not re-derived.** `index_score` computes
`Σ_h w_h·wscale·ReLU(q_h·k_t·dscale)`. The ReLU clips each head's *dot*, not the sum, and
`w_h` comes from `weights_proj` and is frequently negative — so a clipped head contributes
`±0.0` while an unclipped one with a negative gate contributes a negative number. For the
*sum over 32 heads* to be exactly `0.0`, every head must clip simultaneously. That never
happened once in 131,202 samples. I had reasoned from "the ReLU produces many per-head
zeros" to "the summed score is often zero", which does not follow.

**Which fixture this implies: `dense`, on the host-side determinants.** On tie fraction
and presortedness — what drives a comparison sort — the real array is 0.02% ties and ~50%
presorted; `dense` is ~0% and ~50%; `scattered` and the `sorted-sparse` trap are tie-heavy,
and the trap is additionally 100% presorted. It does not sit between the fixtures; it is at
the `dense` end.

**But the fixture argument is no longer the evidence, because the real arrays were measured
directly.** `scores.bin` holds 64 real arrays, so the matched rig now carries a fourth
column fed from the engine's own scores — no fixture-matching inference at all:

| nt | dense | REAL engine scores | tiling contamination |
|---:|---:|---:|---|
| **2456** | 68.8 → 32.8 (2.10×) | **65.5 → 32.6 — 2.01×** | 16.5% duplicated |
| 4096 | 126.0 → 30.8 (4.10×) | 70.3 → 32.4 (2.17×) | 50% — not trustworthy |
| 5209 | 118.4 → 38.3 (3.09×) | 77.5 → 39.0 (1.99×) | 61% — not trustworthy |
| 32768 | 407.5 → 155.4 (2.62×) | 136.7 → 159.3 (0.86×) | **94% — measures tiling** |

**Read only the 2456 row.** The dumped arrays are nt ≈ 2050, so for larger `nt` the rig
tiles them, and tiling duplicates every value — manufacturing exactly the ties the whole
analysis is sensitive to. At 32768 the array is 16× tiled, every value repeated 16 times,
which is why that row resembles the tie-heavy `scattered` column rather than the engine.
That is a fixture artifact of the same family as the `sorted-sparse` trap, caught before
publication this time rather than after.

**So: 2.01× on real engine data at nt=2456, measured.** The `dense` fixture predicted 2.10×
at the same point, which is why the fixture identification was right — but the number to
quote is 2.0×, not the ~2.4× an earlier revision inferred. Note also the host column moved
between runs (86.6 → 68.8 µs at 2456 for the same fixture), so **these ratios are good to
one significant figure**, not two.

**A claim from an earlier revision, now withdrawn.** It said the real array's pass-1 radix
histogram was "slightly better for the kernel than dense" (14 of 256 bins, 43% maximum).
That statistic was computed from record 0 and is inverted for the other 63: across all 64
records the mean pass-1 maximum bin is ~85%, and 34 of 64 put ≥95% of the array in a single
bin. The real array is *more* pass-1 contended than dense, not less. The direct measurement
above supersedes the argument entirely, which is the better reason to stop making it.

**And this settles an open question by elimination.** This document has carried "whether
the host round-trip's 2× in-situ penalty is distribution or cache contention" as unresolved.
The real distribution is structurally `dense`, so the in-engine/microbench gap is **not**
distribution. But name the residual honestly: it is **2.5× at nt=2456 and 3.3× at 5209** —
it *grows with context*, which is itself unexplained. "In-situ cost" is a label for what
the microbench does not reproduce, **not a diagnosed mechanism** (HYPOTHESIS, per
docs/measurement/perf-roadmap.md, which warns that a bucket gives a total and not a decomposition).
That matters for the wiring estimate in the favourable direction: the device kernel does
not run on the CPU, so it should not pay that penalty, and the in-engine win may exceed the
2.0× the microbench shows. **It did.** Wired and measured, the host round-trip costs
429–533 µs/layer in-engine against the kernel's 34 µs/layer — see "Wired" below.

**Limitation, and what the data says about it.** The dump captured the *first* 64 scoring
records, at nt ≈ 2050 — just past `index_topk`, where scoring begins during prefill — while
the refuted premise was specifically about *long* context. The obvious worry is that a token
32k positions back is less similar to the query than one 2k back, so clipping might grow
with distance and the ties might appear at scale.

The same data speaks to that, because distance is a within-record variable. Bucketing every
record by position decile (decile 0 = oldest tokens, 9 = most recent):

| decile | 0 | 3 | 6 | 9 |
|---|---:|---:|---:|---:|
| mean score | −59.8 | −58.4 | −53.6 | −33.4 |
| distinct / n | 0.9999 | 0.9997 | 0.9998 | 0.9998 |
| scores with \|x\| < 1e−6 | 0 | 0 | 0 | 0 |

**The trend runs the wrong way for the worry.** Older tokens do not drift toward zero; they
drift *more negative*, because unclipped heads with negative gates dominate further from the
query. Distinctness is flat to four decimals across the whole span and there is not a single
near-zero score in any decile. Extrapolating that trend predicts *fewer* ties at long
context, not more.

This is evidence, not proof — it is a 2050-token span standing in for a 32k one, and a
monotone trend can still turn. Dumping later in a run is a one-line change to the record
budget and costs one prefill, and is worth doing before anyone relies on the 32k rows.

### Wired — the device top-k in the engine, and both wins are real

`RIVOLI_TOPK` selected the arm; three timing arms plus a correctness arm from **one
binary**, interleaved, 2432-token prompt, `-bench 128`, sole tenant, 2026-07-27. **The
switch is gone** — the engine now always selects on device, the rejected `device-nosync`
option is recorded below rather than shipped, and `verify`'s comparison lives in
`tests/kernel.rs::index_topk_matches_host_selection`. Re-running the A/B means restoring
the arms from git (`77b5500:src/gpu.rs`). Rows, the bucket
table and the full caveat list are in docs/measurement/benchmarks.md, "Device top-k WIRED"; this is the
interpretation.

**Read the buckets, not the wall — and decide which buckets can respond before looking.**
`moe` carries 7–10 ms of within-arm spread here against effects of 9.4 and 2.5 ms, so at
n=2 `wall` resolves neither. Two revisions of this analysis were wrong for exactly that
reason, in opposite directions, before the rule was applied.

| change | measured in | r1 | r2 | mean |
|---|---|---:|---:|---:|
| the device top-k (`host` → `device`) | indexer bucket | −10.5 | −8.3 | **−9.4 ms/token, 2.1% of wall** |
| the sync deletion (`device` → `device-nosync`) | route + unbucketed | −3.2 | −1.8 | **−2.5 ms/token, 0.6%** |

**The top-k.** `idx_host` (11.2, 9.0 ms/token) goes to zero, `idx_gpu` rises 0.72 for the
kernel, and **the unbucketed remainder is unchanged to ±0.4 ms** — nothing else moved. The
kernel costs **34 µs/layer** in-engine against a host round-trip of **429–533 µs/layer**.
Within this session the prize was 13.1–15.3 ms/token, of which the host half was 69–73%,
and the top-k recovered **63–68%** of it: the host share less the kernel's own cost. The
mechanism closes inside one session, with no cross-session differencing.

**The sync deletion is worth ~2.5 ms, not nothing.** The plan predicted it "can then be
deleted outright" and demanded it be costed apart — which was right, and the answer is that
it is real but **4× smaller than the top-k**. `route` rises +12.6 / +10.1 as the wait
relocates to the gate-logits D2H, the unbucketed remainder falls −15.8 / −11.9, and the
difference is the win. **The default keeps the sync anyway**: 0.6% of wall at n=2 does not
buy making `route` incomparable with every historical row in docs/measurement/benchmarks.md. That is a
judgement, not a measurement; re-run at n≥4 and flip if it holds.

**Do not use `wall` for the sync arm.** Its wall delta changes sign between replicates
(−9.2, +8.2) entirely because `moe` swings −6.0 / +10.0 — 14× the 0.7 ms this change can
physically move (3 dense full layers × 229 µs, the only path where the wait lands in `moe`).
An earlier revision read that sign flip as "the second win does not exist"; it was noise
admitted into the comparator.

**One instrument reproduced and one did not, and it matters more than the win.** `idx_gpu`
returned 195.2 / 195.9 µs/layer, within 0.15–0.51% of a different session's 194.9. `idx_host`
returned **533.2 / 428.9 µs/layer — 24% apart within one session**, against 214.2 in that
earlier one. **The quantity this document denominates the entire prize in is the unstable
one, and every share-of-prize figure above inherits that.** Candidate causes, the
counter-evidence against them, and the unexplained 15% wall difference between sessions: see
docs/measurement/benchmarks.md. Nothing is diagnosed.

**Correctness: 10,752 full layers matched the host selection exactly**, sentinel intact,
output byte-identical across all seven runs (docs/measurement/benchmarks.md for the count's derivation and
for why the exit status is not the evidence).

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

**Window 1 fails at both measured contexts** with the CPU top-k, by 72 µs at 2.4k and
215 µs at 5.2k. **With the device top-k it partially revives — at 4k, and only at 4k.**
Budget = 291.25 − indexer − 32.4, against the measured device kernel on the sparse shape:

| context | window-1 budget | device top-k | verdict |
|---:|---:|---:|---|
| 4096 | 93.7 µs | 52.3 µs | **FITS**, 41.4 µs spare |
| 8192 | 66.5 µs | 79.0 µs | **fails**, over by 12.6 µs |
| 16384 | 10.7 µs | 127.4 µs | fails, over by 116.7 µs |

**That table is computed from the WRONG COLUMN, and is left standing only as a marker.**
Its "device top-k" figures (52.3 / 79.0 / 127.4 µs) come from the tie-heavy `scattered` /
`sorted-sparse` fixtures, which the distribution dump above refutes as models of the engine.
The `dense` column is 32.5 / 52.7 / 83.3 µs at the same contexts, which would move the 8192
row from "fails" to "fits" — so the verdicts here are not just imprecise, one of them
flips. **Not recomputed into a new verdict**, because the budget it is differenced against
is itself a microbench scaled across contexts, and this document has been wrong four times
on exactly that kind of derived number. Window 1 needs a measurement, not another
subtraction.

**Say the qualification in the same breath as the result.** That budget assumes the NPU
runs the indexer at *exactly GPU speed*. Charge the window with the indexer, the top-k
and both handoffs and the NPU slowdown budget is **1.25× at 4k, 0.93× at 8k, 0.53× at
16k** — so even where window 1 fits, it demands an NPU within 25% of GPU speed — a 1.25x slowdown budget — over
a single context point, which is not the "a slower NPU is fine" premise the plan rests
on. This is the fifth revision of the window-1 number (22 → 158 → 291 µs → fails →
fits at 4k only); it is recorded because the measurement says so, not because the exact
path is worth reviving on this evidence.

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
question, for reasons recorded in docs/measurement/benchmarks.md. **A GPU∥GPU experiment cannot stand in for
a GPU∥NPU one.**

### What was NOT measured

- **Any context beyond 5.2k, in-engine.** The 8k/16k/32k rows are extrapolated from two
  measured points through a functional form the algorithm justifies, not measured. A 32k run
  costs **~3.6 h of sole-tenant prefill** (measured prefill rate: 0.34–0.39 s/token,
  token-by-token — there is no batched prefill; that is `docs/measurement/perf-roadmap.md` Path A, unbuilt) and,
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

- ~~**Device top-k (NO NPU)**~~ **DONE — WIRED, DEFAULT ON, AND MEASURED IN-ENGINE.**
  `index_topk` (kernels/indexer.hip) computes the selection exactly; `dsa_select_layer` now
  launches it instead of the score D2H + CPU top-k + row H2D. **Worth −9.4 ms/token, 2.1%
  of wall** (r1 −10.5, r2 −8.3, read off the indexer bucket with the unbucketed remainder
  unchanged). Correctness: 10,752 full layers matched the host selection exactly on the
  engine's real scores, over-selection sentinel intact, all seven runs byte-identical.
  **The mid-layer `device_sync` deletion, costed separately as the plan demanded, is worth
  −2.5 ms/token — real, consistently signed, and 4× smaller than the top-k.** The default
  keeps the sync: 0.6% of wall at n=2 does not buy making `route` incomparable with every
  historical row. Full rows and caveats: docs/measurement/benchmarks.md, "Device top-k WIRED".
  `idx.last_nr` is now `min(topk, nt)` by construction, checked by the `verify` arm's
  sentinel rather than by an assertion on the arm that never needed one.

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

  > **UPDATE 2026-08-07 — the mechanism now EXISTS; the measurement is planned but NOT
  > yet run.** M1a is implementable as of this date, behind `--features stale-sel` plus
  > `--stale-sel` (feature AND flag, per the house rule; no env var). Scope note first:
  > everything here is about **GLM-5.2's DSA indexer**, like the rest of this document —
  > whatever this measurement decides, it decides for this workload, not for the
  > DeepSeek-V4 port's indexer, whose NPU question is a separate and open one.
  >
  > **What was built** (`src/gpu.rs::dsa_select_layer`, `src/indexer.rs::StaleShare`):
  > per full layer, a `index_topk`-row device cache (plain `hipMalloc`, deliberately NOT a
  > routed-arena slot — the arena compacts under long-lived reads, the recorded relocation
  > defect). At every scored token the layer **serves** the selection stored at token
  > `t−1` (a D2D into the layer's `sel` slot, enqueued before `index_topk` overwrites its
  > source; null-stream program order makes read-then-overwrite safe) and **stores** the
  > selection `index_topk` just computed from token `t`'s query. The fresh selection is
  > still computed every token — that is the work an NPU would absorb, so the run prices
  > *staleness*, not skipped work. Shared layers reuse the SERVED selection, as they would
  > under the real design. The flag off (or the feature absent) compiles/branches to the
  > exact path unchanged.
  >
  > **The first-token decision the old text asked for:** at the crossing token
  > (`nt = index_topk + 1`) nothing is stored, and the layer attends DENSE over the whole
  > prefix — exact, one row more than a top-k attend, and what the real decoupled design
  > would do (dense needs no selection at all). It cannot flatter the stale arm past that
  > single token. Below `index_topk` there is nothing to be stale about — the indexer
  > computes no selection there — so both arms are identical by construction. The served
  > stale selection is used **verbatim**: it was scored before the current token's key
  > existed, so the diagonal may be absent. Patching the diagonal back in is a different,
  > better variant; it gets measured only if verbatim fails, as its own arm.
  >
  > **Refusals, all loud:** not under `--ppl` → startup error (single-row forwards are the
  > only shape with a well-defined per-layer predecessor; speculation/prefill rows are
  > refused again per-layer as defense); not `--attn dsa` → startup error (dense/streaming
  > have no selection, misa's head-route was never in the decoupled budget); corpus not
  > longer than `index_topk + 2` tokens → startup error (a scored NLL at position `p`
  > comes from the forward at `p−1`, and the last forward is never scored, so the first
  > corpus length where a SCORED position saw a stale selection is `index_topk + 3`); and
  > a `--stale-sel` run that finishes with ZERO stale serves **refuses to return its
  > NLLs** rather than emit an `.nll` indistinguishable from a baseline arm — the "dsa
  > A/B under 2048 tokens covers nothing" trap, now a gate that goes red. Engagement is
  > counted (`stale_served`) and printed.
  >
  > **The planned measurement** (paired, per docs/measurement/perf-roadmap.md's standard;
  > needs the sole-tenant GPU): one binary, `--features rocm,teacher-forcing,stale-sel`;
  > arm A `--ppl tests/ppl-corpus-5000.txt` exact, arm B identical plus `--stale-sel`;
  > `--attn dsa --mode int3-vq` (single-format — hybrid's residency-picks-arithmetic
  > defect forbids quality A/Bs there), cache settings held fixed; no `cargo build`
  > between arms; flock + contention witness per arm. Rank on **paired dNLL from
  > `bin/ppl`**, not the PPL column; an interval straddling zero is inconclusive.
  > **Power caveat, stated up front:** the first `index_topk` positions are identical by
  > construction and the crossing token is exact-dense, so only ~2.9k of the corpus's
  > ~5.0k positions carry any signal — the full-file paired test dilutes its t-statistic
  > by roughly √(2.9/5.0), and the honest secondary read is the same statistics over the
  > positions past 2049 only. **Built-in contamination control:** the first-2048 NLL
  > prefix must match between arms bit-for-bit; any difference means something other than
  > staleness moved (contention, the timing-race class), and the pair is discarded. If
  > more power is needed, the recorded remedy is a symlinked shadow artifact with
  > `index_topk` lowered in its manifest, SHARED by both arms — with the caveat that a
  > lowered threshold is a different selection regime (fewer rows, each mis-pick heavier),
  > so a shadow-run verdict screens but does not settle the k=2048 question.
  >
  > **What a failure means, scoped:** if verbatim 1-step-stale measurably damages quality
  > here, the decoupled window — the only window left — is closed for **GLM's DSA
  > indexer**, and with it this plan's M2/M3. That is this document's workload; it says
  > nothing about the V4 indexer.

  > **MEASURED 2026-08-07 — verbatim 1-step-stale FAILS, decisively. The decoupled
  > window is CLOSED for GLM's DSA indexer, and with it M2/M3.** The paired A/B ran as
  > planned above (sole tenant, flock, per-arm witness sampling kfd + `mem_info_gtt_used`
  > + the llama-swap probe every 60 s — all samples clean on both arms; one binary,
  > `--mode int3-vq --cache-policy lru --attn dsa --max-mem 100`, no build between arms):
  >
  > | read | mean dNLL (nats) | 95% CI | worse% | PPL |
  > |---|---|---|---|---|
  > | full file, 5184 positions | **+0.53923** | [+0.50362, +0.57484] | 49.3 | 4.1126 → 7.0518 |
  > | tail only (past 2049), 3136 positions | **+0.89138** | [+0.83587, +0.94689] | 81.4 | 3.7844 → 9.2283 |
  >
  > The 1% bar is 0.00995 nats; the tail interval sits ~90× past it, and `worse% = 81.4`
  > says the damage is broad, not a few outliers. Every planned control passed: the gate
  > proof refused a 763-token corpus with the vacuous-corpus error and wrote no `.nll`;
  > the first-2048 prefix was **bit-identical** between arms (divergence begins exactly at
  > prediction 2049, the crossing token); and the engagement count was **65,835 = 21 full
  > layers × 3,135 scored stale forwards** — exactly the arithmetic predicts, so the
  > instrument served precisely the selections it claims. Side observation, mechanically
  > expected: the stale arm's expert-cache hit rate moved (71.96% → 69.15%) because a
  > different attend output routes differently; a reminder that this flag, like hybrid's
  > residency defect, changes the trajectory and must never be on in a run that isn't
  > this measurement. Raw rows: `docs/measurement/m1a-stale-selection/` (exact.nll /
  > stale.nll, engine logs, per-arm witnesses, and vacuous.log — the gate-refusal proof).
  >
  > **What this closes, and the one thing it does not.** VERBATIM 1-step-stale — the form
  > window 2 gets for free — is dead here: no overlap saving of ≤2.7% of wall survives a
  > +0.89-nat quality collapse. Scope: **GLM-5.2's DSA indexer on this engine**; it says
  > nothing about the V4 indexer. The variant registered above before the measurement —
  > patching the current token's diagonal into the served selection (one extra attend row,
  > no extra NPU work) — remains unmeasured. Anyone reopening this line owes that arm
  > FIRST, under this same harness; nothing else in M2/M3 is worth touching unless it
  > passes.

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
the gap may be structural. Per docs/measurement/perf-roadmap.md, read the ISA before booking the device.

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

---

## Successor brief — picking this up at the wiring seam

**The wiring described below is DONE** (2026-07-27; see "Wired" above). Kept because its
reading order, its two invariants and its measurement lessons all held up — and because
"instrument the case you trust most first" caught a `verify` gate that passed while
comparing nothing. Everything below the wiring step is still unbuilt.

### Read in this order

1. **This document's "MEASURED" sections**, for what is established and what is not.
2. **`docs/measurement/benchmarks.md`, "DSA indexer round"** and the two sections after it — the rows, the
   methodology, and three recorded measurement traps.
3. **`src/gpu.rs::dsa_select_layer`** — the function being changed, ~90 lines.
4. **`kernels/indexer.hip`, the `index_topk` block** — the kernel that replaces its host half.
5. **`docs/measurement/perf-roadmap.md`'s opening section** on how to write a performance claim here. It is the
   house standard and this work violated it twice before complying.

### State

| item | status |
|---|---|
| `index_topk` kernel + launcher | committed, correctness gate **passes** |
| `tests/kernel.rs::index_topk_matches_host_selection` | 10 cases, sentinel tail, **passes** |
| Cost vs the host, matched rig | measured — **2.01×** on the engine's real arrays at nt=2456 (NOT ~2.4×; that is the `dense` *fixture* and was retracted above) |
| Engine score distribution | measured at nt≈2050: `dense`-like, 0 exact zeros in 131k |
| Wiring into `dsa_select_layer` | **DONE**, default on — **−9.4 ms/token (2.1% of wall)** |
| Selection matches the host in-engine | **10,752 full layers exact**, sentinel intact, output byte-identical across all arms |
| `device_sync` deletion | costed separately: **−2.5 ms/token**, consistently signed, 4× smaller than the top-k. Sync KEPT — 0.6% of wall at n=2 vs `route`'s comparability |
| M1a / anything NPU | ~~untouched, and M1a is still not implementable~~ **CORRECTED 2026-08-07: M1a is BUILT and MEASURED — FAILS; see the MEASURED note under the M1a milestone.** Only the diagonal-patched variant remains unmeasured. Anything NPU: still untouched, and now moot for GLM's DSA indexer (the V4 indexer is out of scope) |

### The wiring, and its two invariants

Replace the D2H + `topk_into` + upload in `dsa_select_layer` with one `launch_index_topk`
onto `rows_buf`. Two things change that are not obvious from the diff:

1. **`idx.last_nr` becomes an implicit invariant.** Today it is `idx.rows.len()`, an
   *observed* count. After wiring, nothing reads the rows back, so it must be
   `min(topk, nt)` by construction. That is sound — the path only runs when `nt > topk`, so
   it is exactly `topk` — but it is a state invariant with no test. Add one, or the first
   change to the threshold logic breaks the attend's row count silently.
2. **The mid-layer `device_sync` has TWO consumers, not one.** It makes the score D2H safe
   *and* it retires the `idx_ev_start`/`idx_ev_end` pair that `HipEvent::elapsed_ms` reads
   into `prof.idx_gpu_ns` — buckets that are not trace-gated and that feed the
   `indexer/tok` telemetry line the A/B below depends on. Deleting it "outright" removes
   the instrument the third arm needs. Either drop the per-layer GPU-span bucket or replace
   the sync with an event query that does not stall the stream. (MISA takes a separate
   `device_sync` for its head-route D2H; that one is untouched by this work.) With the selection device-resident, the attend consumes `rows_buf` on the same
   stream and program order is the whole requirement. This is a *second, separate* win —
   one sync per full layer per token, 21/token — and it must be **costed separately from
   the top-k in the A/B**. If both land in one number nobody will ever know which paid.
   Suggested: three arms, host+sync (baseline), device+sync, device+no-sync.

### Which outcomes at wiring are findings, not nuisances

- **Output tokens change.** That is a **bug**, not a tolerance. The kernel is bit-identical
  to the host selection by test; if the engine's tokens move, the wiring is wrong, the
  test's case list has a hole, or — the third possibility, excluded by neither — a NaN
  reached the score array, where the kernel's contract and `topk_into`'s
  (`partial_cmp(..).unwrap_or(Equal)`, not a total order) legitimately differ. No NaN or
  inf appears in the 131,202 dumped scores, so this is not a live hazard, but it is the
  explanation to check third rather than not at all. Do not rationalise it as numerical noise — the selection
  is integer indices.
- **`route` drops by much less than ~4 ms/token.** A **finding**. It would mean the host
  round-trip was not on the critical path the way the buckets imply, and the `idx_host_ns`
  bucket needs re-reading.
- **`route` drops by much more.** Also a finding, and the likelier surprise: the `device_sync`
  deletion may be worth more than the top-k. That is why they are costed separately.
- **The kernel is slower in-engine than 44 µs/layer.** Expected to some degree — the
  microbench times 60 launches behind one sync and the engine syncs per layer — but a large
  gap would indicate the same launch-bubble effect that made the indexer's GPU span 27%
  above its per-kernel sum, and would be worth attributing rather than absorbing.
- **Degenerate output.** Always a severe bug here, never a benchmark confound.

### The measurement lesson this work kept relearning

Three separate defects in this branch were the same shape: **the case trusted most was the
one never instrumented.**

- A "ReLU-sparse" fixture was introduced *because* it was believed most realistic. It was
  used to justify the tiebreak's importance, named in the test as the discriminating case,
  and it was simultaneously (a) the case that masked over-selection, because when the answer
  is an index prefix a truncated readback matches anyway, and (b) the case whose *timing*
  was an artifact, because its scores descend from index 0 and `topk_into` seeds the
  identity permutation, so a comparison sort was handed an already-sorted slice — its best
  case, and one no device kernel can exploit. It produced a false 1.6–1.9× regression.
- Then the distribution dump showed the fixture was not realistic at all: the engine
  produces **zero** exact zeros, 90% negative scores, and ~100% distinct values.

Three operational rules, in decreasing generality:

1. **Instrument the case you trust most first.** Confidence concentrates exactly where you
   have not looked, and that is where a blind spot costs the most.
2. **When timing a comparison-based algorithm, randomise input order** — otherwise you are
   measuring your generator. Hold the structure you care about fixed and vary only order,
   as the `scattered` vs `sorted-sparse` pair does.
3. **After a retraction, grep the whole document for the retracted number.** A corrected
   section three pages from an uncorrected headline silently re-publishes the error; this
   document did exactly that in one commit.

And one that is not about measurement: **a passing control proves the rig measures
something, never that it measures the thing you named it after.** Every validity control
here passed while window 1 was wrong by 13×.
