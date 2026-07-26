# rivoli — NPU offload plan: the DSA indexer

Status: **M0 and M1 MEASURED (2026-07-26). M0 clears ≥4k. M1 clears via the decoupled
window only. A no-NPU change takes half the prize and gates the rest.** The NPU is live on
this node — XRT 2.21, amdxdna 0.1, firmware 1.1.2, `/dev/accel/accel0`, Strix Halo NPU at
`0000:c8:00.1` — and 100% idle during decode. This plan is **exclusive to one workload: the
DSA sparse indexer.** The other NPU candidate (a spec-decode drafter) was analysed and set
aside — spec decode is a verified-correct *negative result* on this engine (`deadend/mtp`),
and the NPU changes only the draft cost, not the verify-union penalty that sank it.

## The finding, in four lines

1. **The prize is real above 4k context** — 6.8 ms/token at 4k rising to 19.4 ms/token at
   32k, against a ≥0.68 ms/token handoff floor. Below `index_topk` = 2048 it is 0.32
   ms/token, *less than the handoff*, so the offload would make the engine slower there.
2. **Half the prize is not GPU work.** The host score-D2H + CPU top-k + row upload is
   **49–60%** of everything the offload removes. A device top-k kernel recovers it exactly —
   no approximation, no quality gate, no NPU, no toolchain.
3. **The exact-overlap design (window 1) does not clear as the engine stands.** The window
   is 291 µs, and after the indexer and two handoffs it leaves only **94 / 66 / 11 µs** at
   4k / 8k / 16k to produce the selection — which today costs **152 / 168 / 334 µs** on the
   CPU. Window 1 becomes viable only *if* the device top-k in (2) is built and is fast.
4. **The decoupled design (window 2) clears at every context**, with 1.9–6.5× of NPU
   slowdown budget even if the selection stays at today's CPU cost. It is the
   approximation, so **M1a is the gate** — and M1a is not yet implementable.

**A methodological warning that belongs at the top.** Window 1's verdict moved **four
times** under review: 22 µs (window under-scoped) → 158 µs (weights cache-resident, `q_b`
reading 372 GB/s — above the bus) → 291 µs (rotated) → *does not clear* (the budget omitted
the cost of producing the selection). Three defects in the instrument and one in the
analysis built on it, every one found by review rather than by the rig, and every one moving
the same conclusion. Treat any row here that an in-engine run has not confirmed accordingly.

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
  window 1.
- **On the GPU side the dominant cost is fixed, not context-scaling.** `wq_b` is 1.64
  ms/token regardless of context, and `index_score` does not overtake it until ~**11k**
  (linear fit through the 8k/16k rows). At 8k the context-independent work is 133/192 =
  **69%** of the per-layer GPU cost.
- **The prize is ~4.5% of wall at 32k** and ~1.9% at 8k. **ESTIMATED denominator:** wall
  under `--attn dsa` has never been measured at any context. ~430 ms/token takes
  benchmarks.md's 351 ms/token `hybrid / lru` row, swaps its short-context attend for
  `mla_attend` at nr=2048 (778 µs × 78 = 60.7 ms), and adds this table's own 19.4 ms
  indexer. Treat it as an order of magnitude.

**Caveat on the host top-k, the biggest line item.** `topk_into` is comparison-driven
(`select_nth_unstable_by` + `sort_by` over an index vector), so its cost is
**distribution-dependent**. At nt=32768 the round-trip totals **162 µs/layer** against a
degenerate tie-heavy array (superseded run `m0m1-v2`) and **554 µs/layer** against a
synthetic heavy-tailed one (final run). The table uses the latter. The engine's true
distribution is neither, so read the 32k host cost as **3.4–11.6 ms/token**. This single
uncertainty is wider than most of the rest of the analysis.

### M1 — the windows

| window | measured | verdict |
|---|---:|---|
| 1 — selection-independent phase 1 (exact) | **291.25 µs** | **does not clear as the engine stands** |
| 2 — MoE batch, 9 vq3 experts + reduce (stale) | **1261.88 µs** | **CLEARS at every context** |
| 2 — dense fp8 SwiGLU MLP (stale), the 3 dense full layers | **1174.67 µs** | **CLEARS** |

Window 1 decomposed: `gemv_fp8 q_b` **213.35** (4 rotating copies, 157 GB/s) + kv_a/rmsnorm/
rope/append 22.62 + `rope` query 7.45 + `mla_absorb_fp8` **45.64** (4 rotating copies) +
`gather_rope` 2.20. *Two residual optimisms, both favourable:* 4 copies cycle 193 MB against
a 32 MB MALL so ~1 read in 6 may still hit, where the engine holds 78 distinct weights; and
`kv_a` is a single replayed 3.5 MB copy, comfortably MALL-resident. A fully cold window
would be **larger** than 291 µs, widening window 1's margin. 291 µs is a lower bound.

**The budget must hold three things, not two: the indexer, two handoffs, and the production
of the selection itself.** The offload has to deliver 2048 row indices, not scores — M2's
own gate is selection equivalence. Wherever that top-k runs it is serial with the NPU
compute and sits inside the window. Budget left for it after the indexer and 2×16.18 µs:

| context | indexer/layer | window 1 leaves | window 2 leaves | selection costs today (CPU) |
|---:|---:|---:|---:|---:|
| 4096 | 165.2 µs | **93.7 µs** | 1064.4 µs | 152.1 µs |
| 8192 | 192.4 µs | **66.5 µs** | 1037.1 µs | 167.7 µs |
| 16384 | 248.2 µs | **10.7 µs** | 981.3 µs | 334.5 µs |
| 32768 | 372.3 µs | **negative** | 857.2 µs | 530.7 µs |

**Window 1 does not clear**: the selection step would have to be **1.6× faster than the CPU
at 4k, 2.5× at 8k, 31× at 16k**, and no amount of speed rescues 32k. That is not a
statement about the NPU's speed — it holds even for an infinitely fast NPU indexer. Window 1
becomes viable only if the device top-k below is built *and* lands under ~90 µs/layer at 4k
or ~66 µs at 8k, which is unmeasured.

**Window 2 clears everywhere, even with the selection left on the CPU at today's cost.**
Budget for the NPU indexer alone = 1229.5 µs − the selection cost:

| context | NPU indexer budget | may be this much slower than the GPU |
|---:|---:|---:|
| 4096 | 1077.4 µs | **6.5×** |
| 8192 | 1061.8 µs | **5.5×** |
| 16384 | 895.0 µs | **3.6×** |
| 32768 | 698.9 µs | **1.9×** |

(The 3 dense-MLP full layers have a 1142.3 µs budget instead of 1229.5, so their ×figures
are ~7% tighter. Window 2 still clears there.)

**M1 gate: CLEARS via window 2 only.** The plan intended window 1 as the always-correct
default with the decoupled path opt-in and quality-gated. That default is not available
without first making the selection step much cheaper. **M1a is the gate.**

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

- **In-engine confirmation — the highest-value follow-up.** Everything above is microbench.
  `docs/PERF.md` records the microbench mispredicting `route` by 27% on the last per-kernel
  tranche, and this rig has been wrong by more than that twice. Needs a decode run at >2048
  context (~15 min of prefill) plus two profile buckets in `dsa_select_layer`. Also settles
  the 3.4×-wide host top-k bracket. **Before the toolchain spike, not after.**
- **A device top-k's cost** — which now decides whether the exact path exists at all.
- **Wall at long context under `--attn dsa`** — never measured; the ~4.5% is an estimate.
- **Bandwidth contention during either window**, and anything across the amdxdna boundary.
- **Anything NPU.** No NPU code was written or run, per the plan's own sequencing.

## Milestones

- ~~**M0**~~ **DONE.** Clears ≥4k, fails ≤2048.
- ~~**M1**~~ **DONE.** Window 2 clears; window 1 is contingent on a cheap selection step.

- **Device top-k (NO NPU). DO THIS FIRST.** It recovers 49–60% of the prize exactly — the
  host round-trip is 3.36 → 11.63 ms/token, all of it GPU-idle — and it is the precondition
  for window 1 and therefore for the plan having any exact path. One kernel: same selection,
  no quality gate, no staleness, no toolchain, no per-layer handoff. It also removes a
  per-layer `device_sync` and the distribution-dependence that makes the largest line item
  uncertain. **Measure its per-layer cost against the 94 / 66 / 11 µs window-1 budgets.**

- **In-engine confirmation of M0 (NO NPU).** Two buckets in `dsa_select_layer`, one
  long-context decode run.

- **Toolchain spike — lead with the handoff.** Compile + run one dense bf16 kernel on the
  NPU via whatever flow exists (MLIR-AIE / Peano / IREE — presence unverified; `xrt-smi`
  running ≠ an xclbin compiler). Measure, in order: **(1) handoff round-trip**; (2) NPU
  sustained bandwidth at the indexer's shapes against the 10–26 GB/s window-2 bar; (3)
  GPU↔NPU concurrent bandwidth.

- **M1a — stale-selection quality (NO NPU). The gate for the whole programme.** Simulate a
  1-step-stale / periodic-refresh selection offline against the exact selection; measure the
  perplexity delta on `tests/ppl-corpus-5000.txt` via `--ppl`. **Not yet implementable:**
  nothing in the engine produces a stale selection, so this needs an engine change (a
  one-step selection cache in `DeviceIndexer` behind a flag) that this list does not scope.

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
- **The prize is ~4.5% of wall at 32k and ~1.9% at 8k**, about half of it recoverable
  without the NPU. This is the number the go/no-go rests on.
- **This analysis has been wrong four times on one number**, three times in the instrument
  and once in the reasoning, always understating what had to fit or overstating the room.
- **amdxdna 0.1 / firmware 1.1.2** — early stack; expect rough edges.

The framing to hold onto: this is not a "make the indexer fast" project but a "make the
indexer disappear from the GPU timeline" project. What the measurements change is that half
the indexer is not on the GPU timeline at all — it is a CPU top-k the GPU is blocked behind
— and that fixing *that*, with no NPU involved, is both the largest single win available and
the precondition for the exact overlap the plan wanted.
