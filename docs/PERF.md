# rivoli — performance plan

Status: **analysis + roadmap.** Leads with the **structural paths** (the higher-level
goals that move throughput by a multiplier or restructure a whole phase); the per-kernel
findings are **follow-up polish** to that structural work and to the existing improvement
proposals ([CACHE_ROUTE](CACHE_ROUTE.md), [CACHE_PILOT](CACHE_PILOT.md), the fp8-int4
work).

## Where the time goes (hybrid+lru, 512 tok, the best coherent config)

```
2.85 tok/s = 351 ms/tok
  route      115 ms   attention: MLA projections (fp8 GEMV) + absorb/value + flash attend
  moe-gpu    210 ms   routed-expert compute  ← 60% of wall, the dominant phase
  fetch      159 ms   NVMe O_DIRECT expert streaming — 94% HIDDEN (~9 ms exposed)
  tail       ~16 ms   final rmsnorm + lm_head (gemv_i8, vocab 154880) + device argmax
```

**We are compute-bound, not fetch-bound.** Fetch overlaps compute almost entirely (~9 ms
exposed), so `wall ≈ route + moe-gpu + tail`. Two consequences that shape the plan:

1. **Fetch-wall wins buy nothing** — residency work pays a *different* way: fewer misses →
   fewer host-gated launch **bubbles** inside `moe-gpu`. A compute win, not a fetch win.
2. **The prize is `moe-gpu` (60%) and `route`.** The structural paths below attack those
   two by a multiplier or a whole-phase restructure; the per-kernel follow-ups shave the
   remaining milliseconds.

Hardware: gfx1151, 40 CUs, 256 GB/s LPDDR5 (unified, GTT ceiling 116 GiB).

---

## Structural paths — the higher goals

### Path A — Batched-GEMV kernels → speculative decode (MTP): the throughput multiplier

The single biggest ceiling. Amortize the ~325 ms of per-token compute across K tokens
drafted then verified in **one** forward.

- **Enabler: batched-GEMV kernels.** Every kernel today is batch-1 (wave-per-row).
  A K-row GEMV amortizes the weight read across K tokens — this alone turns the
  memory-bound batch-1 GEMVs (o_proj, projections, lm_head) into compute-bound ones with
  K× the arithmetic intensity, and it is the hard prerequisite for MTP. **Do this first.**
- **MTP / speculative decode.** GLM ships a layer-78 MTP head — **and it's in the fp8
  checkpoint we're re-downloading**, so this is newly feasible (previously absent from the
  artifact). The old ~1.35× cap was a *disk-bound* artifact (fetch was 58% of wall then and
  doesn't amortize across drafts); compute-bound with fetch hidden, single-draft MTP at
  ~85% accept lands ~1.5–1.7×, tree/multi-token drafting more. Needs the batched kernels,
  MTP-head wiring, a draft/verify loop, and KV rollback on rejection.

**Ceiling: ≥1.5×. Effort: high. Sequence: batched kernels (independently useful) → MTP.**

### Path B — MoE format program: restructure the 210 ms phase

The MoE dot has two separable costs — the int3-vq **gather-throughput wall** (~53 GB/s,
24% of bus, already at max occupancy, so only a *format* change moves it) and int4's
**residency tradeoff** (no gather, ~1.8× compute, but 18.9 vs 15.3 MB). Attack both as one
program behind a shared quality gate (perplexity on fixed text — never free-running tok/s):

1. **`fp8_to_i4`** — derive int4 from the original fp8 (higher fidelity than via vq3;
   download in flight) → enables a larger coherent hot fraction.
2. **`--hot-pct` re-tune** on the fp8-derived `.i4` — more experts in int4 = less gather,
   bounded by the residency cliff (judge with the replay sim + a fixed-token bench).
3. **Smaller, L1-resident codebook** (per-kernel follow-up #1) — the lever that lifts the
   gather wall itself.

**Impact: structural on the 210 ms (60% of wall). Effort: medium; requant + quality gate.**

### Path C — Cache-conditional residency: kill the `moe-gpu` bubbles

Every avoided miss avoids a host-gated per-expert launch, so it removes a compute bubble
from `moe-gpu` (not fetch wall — fetch is hidden). Two existing proposals own this:

- **[CACHE_ROUTE](CACHE_ROUTE.md)** (top-m cache-conditional routing) — training-free,
  reported >50% miss reduction at +0.1–3% ppl; biases routing toward resident experts.
  Highest residency ROI.
- **[CACHE_PILOT](CACHE_PILOT.md)** (router-piloted cross-layer prefetch) — fetch L+1/L+2
  experts during current compute; same bubble mechanism.
- **Stretch — device-side expert loop.** The host-gating is structural; a persistent MoE
  kernel that consumes experts as their bytes land (no host round-trip per expert) removes
  the bubble class entirely. High effort; caps the bubble component of `moe-gpu`.

---

## Per-kernel follow-ups

Tactical, a few ms each — done after or alongside the structural path each supports.
Grounded in the measured kernel profile.

1. **VQ_K=2048 L1-resident codebook** *(feeds Path B).* The fp16 codebook is 32 KB
   (VQ_K=4096) — one fits L1, but `moe_gateup` needs *two* (gate+up) = 64 KB and spills to
   L2. A **VQ_K=2048** codebook is 16 KB → both fit L1 with headroom → the random gather
   becomes a reliable L1 hit, *and* the streamed expert shrinks. Cost: re-quantize + a
   perplexity check.

2. **o_proj split-K tuning** *(route).* o_proj [6144,16384] is ~half of route; split-K has
   it at 185 GB/s — **1.45× headroom to the 256 GB/s peak.** Tune the split-K
   (`ROWS_PER_BLOCK`, more splits to fill the 40 CUs). Est ~10–15 ms.

3. **`mla_latent_attend` occupancy** *(route; scales with context — do before any
   long-context push).* ~20 ms @ nr512, grows ~linearly with context. LDS-capped to 1
   WG/CU (dynamic LDS `((HB+TILE)·kvl + TILE·rope)·4 = 53 KB`). Levers: move `acc[HB·kvl]`
   (16 KB) to registers *or* HB 8→4 → 2 WG/CU; lower `MLA_MIN_TILES_PER_SPLIT` 4→2 so short
   context spawns enough splits to fill the 40 CUs. Est ~5–7 ms at short ctx, much more at
   long ctx.

4. **`mla_absorb_fp8` restructure** *(route).* ~5 ms @ 99 GB/s — structural transpose
   direction (1 thread/(head,i), kvl-strided single-byte loads, low ILP). Restructure
   toward `mla_value`'s wave-per-row float4 form. Est ~2–3 ms.

5. **`lm_head` split-K** *(tail).* [154880, 6144] int8 GEMV — the one big tail cost. Split-K
   it (same shape argument as o_proj: many rows, long reduction). A few ms.

---

## Ranked roadmap

| # | Item | Path | Est. impact | Effort | Status |
|---|---|---|---|---|---|
| 1 | `fp8_to_i4` + `--hot-pct` re-tune | B | med (more int4, faster compute) | low | fp8 downloading |
| 2 | CACHE_ROUTE (top-m) | C | med–high (>50% miss cut) | med | [proposal](CACHE_ROUTE.md) |
| 3 | VQ_K=2048 L1-resident codebook | B / follow-up #1 | med (lifts gather wall) + smaller experts | med (requant) | new |
| 4 | Batched-GEMV kernels | A | med now, unlocks MTP | med–high | new |
| 5 | MTP / speculative decode | A | high (≥1.5×) | high | needs #4 + fp8 MTP head |
| 6 | `mla_latent_attend` occupancy | follow-up #3 | ~5–7 ms now, huge at long ctx | med | new |
| 7 | o_proj / lm_head split-K | follow-up #2, #5 | ~10–15 ms | low | new |
| 8 | `mla_absorb` restructure | follow-up #4 | ~2–3 ms | med | new |
| 9 | CACHE_PILOT (cross-layer prefetch) | C | low–med (overlaps #2) | med | [proposal](CACHE_PILOT.md) |
| — | device-side expert loop | C (stretch) | caps bubble class | high | stretch |

**Suggested sequence:** land #1 (cheap, in flight) → **Path B + Path C** (#2, #3 — the
biggest structural wins on the 210 ms) → **Path A** (#4 batched kernels → #5 MTP, the
multiplier) → per-kernel follow-ups #6–8 as route/tail polish (#6 mandatory before any
long-context work).

**Measurement discipline (learned the hard way):** rank format/numerics changes by (a) the
replay residency sim, (b) a fixed forced-token wall-clock bench, and (c) perplexity for
quality — **never** free-running greedy tok/s (confounded by output degeneration; a broken
run looks fastest). See [MODES.md](MODES.md) and [benchmarks.md](../benchmarks.md).
