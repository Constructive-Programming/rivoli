# rivoli — performance plan

Status: **analysis + roadmap.** Grounds every lever in the current profile so we chase
the phases that actually cost, not the ones that look expensive. Cross-references the
standalone proposals ([CACHE_ROUTE](CACHE_ROUTE.md), [CACHE_PILOT](CACHE_PILOT.md)) where
they already own a lever.

## Where the time goes (hybrid+lru, 512 tok, the best coherent config)

```
2.85 tok/s = 351 ms/tok
  route      115 ms   attention: MLA projections (fp8 GEMV) + absorb/value + flash attend
  moe-gpu    210 ms   routed-expert compute  ← 60% of wall, the dominant phase
  fetch      159 ms   NVMe O_DIRECT expert streaming — 94% HIDDEN (~9 ms exposed)
  tail       ~16 ms   final rmsnorm + lm_head (gemv_i8, vocab 154880) + device argmax
```

**We are compute-bound, not fetch-bound.** Fetch (159 ms) overlaps compute almost
entirely — only ~9 ms is exposed. So `wall ≈ route + moe-gpu + tail = 325 ms of GPU
compute + ~16 ms tail`. Two consequences that shape the whole plan:

1. **Attacking fetch wall buys almost nothing** (it's already hidden). Residency work
   (CACHE_ROUTE/PILOT) pays off a *different* way: fewer misses → fewer host-gated
   per-expert launch **bubbles** inside `moe-gpu`. That's a compute win, not a fetch win.
2. **The prize is `moe-gpu` (210 ms) and `route` (115 ms).** Everything below is ranked
   by how much of those two it moves.

Hardware: gfx1151, 40 CUs, 256 GB/s LPDDR5 (unified, GTT ceiling 116 GiB).

---

## Per-phase levers

### MoE compute — `moe-gpu` 210 ms (the main event)

What it is: per token, top-8 routed + 1 shared expert × 75 layers. In hybrid, hot experts
decode int4 (sequential nibble, no gather), cold decode int3-vq (random `cb[idx]` L1
gather). Two independent cost components:

**(a) The int3-vq gather-throughput wall.** The VQ dot reads ~10.3 GB/tok of indices+
scales at **~53 GB/s = 24% of the 256 GB/s bus, ~2% of fp32 peak** — neither bandwidth-
nor compute-bound, but throughput-bound on random 8-byte L1 lookups. The kernel is
already at the hardware ceiling (16 waves/SIMD, 0 spill), so occupancy/ILP can't move it
(confirmed: the ×4 unroll was a wash). Only **fewer/bigger gathers** move it — a format
change. Levers, best-first:

- **L1-resident codebook via a smaller VQ codebook.** Today the fp16 codebook is 32 KB
  (VQ_K=4096) — one fits L1, but `moe_gateup` needs *two* (gate+up) = 64 KB and spills to
  L2. A **VQ_K=2048** codebook is 16 KB → both fit L1 with headroom → the gather becomes a
  reliable L1 hit. Cost: a re-quantize (learn 2048-entry codebooks) and a quality check
  (perplexity). Est: pushes the 53 GB/s wall up materially on the cold path. **Effort:
  medium (requant + oracle). Risk: quality — measure ppl before/after.**
- **More experts in int4 (raise `--hot-pct`).** int4 has *no gather* — ~1.8× faster
  compute (microbench: 669 vs 353 GElem/s). Hybrid already runs hot experts int4; the
  ceiling is residency (18.9 vs 15.3 MB). The **fp8-derived `.i4`** (in progress, higher
  fidelity than vq3-derived) may let a larger hot fraction stay coherent — re-run the
  `--hot-pct` sweep once `fp8_to_i4` lands. **Effort: low (re-tune). Risk: residency
  cliff — use the replay sim + a fixed-token bench, not free-running tok/s.**

**(b) Host-gated launch bubbles.** Each miss's per-expert compute launches only after its
load Signal resolves on the host, so the compute stream idles between host-gated launches;
those bubbles inflate `moe-gpu`. Fewer misses ⇒ fewer bubbles. Two proposals own this:

- **[CACHE_ROUTE](CACHE_ROUTE.md) (top-m cache-conditional routing)** — training-free,
  reported >50% cache-miss reduction at +0.1–3% ppl. Biases routing toward resident
  experts. Lands as fewer bubbles in `moe-gpu`. **Highest residency ROI; proposal ready.**
- **[CACHE_PILOT](CACHE_PILOT.md) (cross-layer prefetch)** — router-piloted prefetch of
  L+1/L+2 experts during current compute. Same bubble mechanism.
- **Deeper: a device-side expert loop.** The host-gating is structural — a persistent MoE
  kernel that pulls experts as their bytes land (no host round-trip per expert) removes
  the bubble class entirely. **Effort: high (kernel rework). Impact: caps the bubble
  component of moe-gpu.** File as a stretch item.

### Attention — `route` 115 ms

Context-independent projections dominate at short context; the flash-attend grows with
context and will dominate at 32k+. Sub-costs (measured):

- **o_proj (~62 ms, half of route).** [6144,16384] batch-1 fp8 GEMV, split-K'd to 185
  GB/s — **1.45× headroom to the 256 GB/s peak.** Lever: tune the split-K (ROWS_PER_BLOCK,
  more splits to fill 40 CUs). **Effort: low-medium. Est: ~10–15 ms.**
- **mla_absorb_fp8 (~5 ms @ 99 GB/s).** Structural transpose direction (1 thread/(head,i),
  kvl-strided single-byte loads, low ILP). Restructure toward `mla_value`'s wave-per-row
  float4 form → ~2× → save ~2–3 ms. **Effort: medium. Est: ~2–3 ms.**
- **mla_value_fp8 (~2.5 ms @ 254 GB/s)** — at the HW ceiling. Leave it.
- **mla_latent_attend (~20 ms @ nr512, grows ~linearly with context).** LDS-capped to 1
  WG/CU (dynamic LDS ((HB+TILE)·kvl+TILE·rope)·4 = 53 KB). Levers: move `acc[HB·kvl]` (16
  KB) to registers *or* HB 8→4 → 2 WG/CU; lower `MLA_MIN_TILES_PER_SPLIT` 4→2 so short
  context spawns enough splits to fill 40 CUs. **Est: ~5–7 ms at short ctx, much more at
  long ctx.** **Do this before any long-context push** — it's the one that scales.

Route total realistic gain: ~20–25 ms (~6–7% of wall) at short context, and it's what
keeps long-context viable.

### Fetch — 159 ms, 94% hidden

Already hidden; there is no fetch-wall win at this budget. The only reason to touch it is
the bubble mechanism above (CACHE_ROUTE/PILOT), which is really a `moe-gpu` win. At
*smaller* budgets fetch is exposed and dominates — but we run at 115/116 GiB GTT, near the
ceiling. Raising residency needs more physical RAM (128 GB node) or fewer bytes/expert
(the VQ_K=2048 codebook also shrinks the streamed expert). Do not re-add speculative
prefetch as a fetch-wall play — it was deleted for good reason (overlap adds no bandwidth).

### Tail — ~16 ms

`lm_head` is a [154880, 6144] int8 GEMV — the one big tail cost. Split-K it (same shape
argument as o_proj: many rows, long reduction) for a few ms. Low priority; do it if
touching linalg.hip anyway.

---

## Global levers

### Speculative decode / MTP — the biggest ceiling, the most work

Amortize the 325 ms of per-token compute across K tokens drafted then verified in **one**
forward. GLM ships a layer-78 MTP head — **and it's in the fp8 checkpoint we're
re-downloading**, so this is newly feasible (it was previously absent from the artifact).

- **Why it's bigger now.** The old estimate capped this at ~1.35× — but that was on the
  disk-bound engine (fetch was 58% of wall then, and fetch doesn't amortize across drafts).
  We are now compute-bound with fetch hidden, so amortizing the 325 ms compute lands much
  closer to the accept-rate ceiling. Single-draft MTP at ~85% accept → ~1.5–1.7×;
  tree/multi-token drafting → more.
- **The hard prerequisite: batched-GEMV kernels.** Every kernel today is batch-1
  (wave-per-row). Verifying K draft tokens is a K-row forward; without batched kernels
  you'd run K sequential forwards and gain nothing. So step 1 is a batched GEMV/MoE path
  (K rows share the weight read — which also *reduces bytes/token* on the verify pass).
- **Plus:** MTP-head wiring, draft/verify loop, and KV rollback on rejection.
- **Verdict:** highest ceiling (≥1.5×), highest effort. Sequence it *after* the batched-
  kernel work, which is independently useful.

### Batched-GEMV kernels — the enabler

A K-row GEMV amortizes the weight read across K tokens. Directly enables MTP, and on its
own turns the batch-1 memory-bound GEMVs (o_proj, projections, lm_head) into
compute-bound ones with K× better arithmetic intensity. This is the highest-leverage
*infrastructure* item — do it first among the global levers.

### Format / quant program (attacks the 210 ms structurally)

Bundle the MoE-format levers into one program with a shared quality gate (perplexity on
fixed text, never free-running tok/s):
1. `fp8_to_i4` — higher-fidelity int4 from the original fp8 (in progress) → enables (2).
2. `--hot-pct` re-tune on the fp8-derived `.i4`.
3. VQ_K=2048 (or product/residual VQ) → L1-resident codebooks → lifts the gather wall.
4. Re-bench the full 3×3 (+ hot-pct sweep) as the acceptance test.

### Residency

Near the GTT ceiling already. The realistic knob is *bytes per expert* (item 3 above
shrinks both the resident codebook and the streamed expert), plus CACHE_ROUTE/PILOT to
raise effective hit rate without more RAM.

---

## Ranked roadmap (impact × effort)

| # | Lever | Phase | Est. impact | Effort | Status |
|---|---|---|---|---|---|
| 1 | `fp8_to_i4` + `--hot-pct` re-tune | moe-gpu | med (more int4, faster compute) | low | fp8 downloading |
| 2 | CACHE_ROUTE (top-m) | moe-gpu bubbles | med–high (>50% miss cut) | med | [proposal](CACHE_ROUTE.md) |
| 3 | VQ_K=2048 L1-resident codebook | moe-gpu | med (lifts gather wall) + smaller experts | med (requant) | new |
| 4 | Batched-GEMV kernels | route + tail + enabler | med now, unlocks MTP | med–high | new |
| 5 | MTP / speculative decode | global | high (≥1.5×) | high | needs #4 + fp8 MTP head |
| 6 | mla_latent_attend occupancy | route (scales with ctx) | ~5–7 ms now, huge at long ctx | med | new |
| 7 | o_proj / lm_head split-K tuning | route + tail | ~10–15 ms | low | new |
| 8 | mla_absorb restructure | route | ~2–3 ms | med | new |
| 9 | CACHE_PILOT (cross-layer prefetch) | moe-gpu bubbles | low–med (overlaps #2) | med | [proposal](CACHE_PILOT.md) |
| — | device-side expert loop | moe-gpu bubbles | caps bubble class | high | stretch |

**Suggested sequence:** land #1 (cheap, already in flight) → #2 and #3 as the MoE
program (biggest structural wins on the 210 ms) → #4 batched kernels (enabler + route/tail
wins) → #5 MTP (the multiplier) → #6–8 as incremental route polish, #6 mandatory before
any long-context work.

**Measurement discipline (non-negotiable, learned the hard way):** rank format/numerics
changes by (a) the replay residency sim, (b) a fixed forced-token wall-clock bench, and
(c) perplexity for quality — **never** free-running greedy tok/s (it's confounded by
output degeneration; a broken run looks fastest). See [MODES.md](MODES.md) and
[benchmarks.md](../benchmarks.md).
