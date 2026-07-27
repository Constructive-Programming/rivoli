# rivoli — performance plan

Status: **analysis + roadmap.** Leads with the **structural paths** (the higher-level
goals that move throughput by a multiplier or restructure a whole phase); the per-kernel
findings are **follow-up polish** to that structural work and to the existing improvement
proposals ([CACHE_ROUTE](CACHE_ROUTE.md), [CACHE_PILOT](CACHE_PILOT.md), the fp8-int4
work). Residency / cache-conditional routing is **not** covered here — it is owned in full
by those two proposals.

## How to read — and write — this document

**A phase profile localises cost; it does not explain it.** Any mechanism attributed to a
hot kernel without reading its ISA is a **HYPOTHESIS**. Mark it as one when writing, and
confirm with `hipcc -S` before implementing — that costs a compile, not a device slot.

This is not a general caution; it is the measured result of the first per-kernel tranche.
**Four of the five per-kernel items below had the wrong mechanism.** The profile was right
about *where* the time went in all five cases and wrong about *why* in four:

| # | Mechanism in the plan | Mechanism in the ISA |
|---|---|---|
| 2 o_proj | too few blocks to fill the machine | signed 32-bit divide in the inner loop |
| 3 attend | LDS caps occupancy at 1 WG/CU | still 1 WG/CU after the fix; the real prize was an LDS read-modify-write, and a lever the item never listed (HB) |
| 4 absorb | transpose direction, single-byte loads, low ILP | 64-bit divide LLVM cannot strength-reduce |
| 5 lm_head | split-K, "same shape argument as o_proj" | not grid-starved at all (19,360 blocks); it is load width |

Every one of those corrections came from a compiler that was available the whole time,
free, while the GPU was the bottleneck. See benchmarks.md, "Read the ISA before you book
the device", for the invocations and the two ways an instruction count lies.

**The same error has a magnitude form, and this document made it too.** Item #5's estimate
was revised from "a few ms" up to "~10 ms" by reasoning that `tail` ≈ 16 ms is "almost
entirely" lm_head, implying <100 GB/s. Measured: lm_head is **8.12 ms**, roughly *half*
the bucket, and the original "a few ms" was closer than the revision. **A bucket gives you
a total, not a decomposition** — inferring a component's cost from its phase budget is the
same mistake as inferring its mechanism from its phase budget, and both were made here on
the same afternoon. Decompose by measurement before you estimate from a bucket.

**The structural paths below have NOT been through this filter.** Path A and Path B were
written in the same style, from the same profile, and their mechanism claims are
localisation plus a reasoned guess rather than localisation plus a diagnosis. They may
well be right — but treat them as un-ISA'd until someone checks, and expect the per-kernel
hit rate to apply.

## Where the time goes (hybrid+lru, 512 tok, the best coherent config)

**Status of `route`: the per-kernel tranche on `feat/perf` measures `route` 112 → 104 ms
in-engine** (interleaved A/B, flat control, identical miss counts, byte-identical output).
The 115 ms below is the pre-tranche figure. See benchmarks.md, "In-engine confirmation".

```
2.85 tok/s = 351 ms/tok
  route      115 ms   attention: MLA projections (fp8 GEMV) + absorb/value + flash attend
  moe-gpu    210 ms   routed-expert compute  ← 60% of wall, the dominant phase
  fetch      159 ms   NVMe O_DIRECT expert streaming — 94% HIDDEN (~9 ms exposed)
  tail       ~16 ms   final rmsnorm + lm_head (gemv_i8, vocab 154880) + device argmax
```

**We are compute-bound, not fetch-bound.** Fetch overlaps compute almost entirely (~9 ms
exposed), so `wall ≈ route + moe-gpu + tail`. Two consequences that shape the plan:

1. **Fetch-wall wins buy nothing.** Residency work pays a *different* way — fewer misses →
   fewer host-gated launch bubbles inside `moe-gpu` — and that lever is already owned by
   [CACHE_ROUTE](CACHE_ROUTE.md) / [CACHE_PILOT](CACHE_PILOT.md); it is out of scope here.
2. **The prize for this plan is `moe-gpu` (60%) and `route`.** The structural paths below
   attack those two by a multiplier or a whole-phase restructure; the per-kernel follow-ups
   shave the remaining milliseconds.

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
   **Mechanism corrected by ISA inspection:** the first-order cost was not the grid shape
   but a **signed integer division in the inner loop** (`scalerow[i0 / block]`, `block` a
   runtime `int`) — 8 quotient-correction ops around 5 FMAs, 44 VALU per iteration. A
   shift takes it to 29 VALU with the memory ops unchanged. **But a 34% VALU cut bought
   only 2.3% (541.6 → 529.0 µs, −0.98 ms/tok) — a real defect that was NOT the binding
   constraint.** Those are two separate findings and collapsing them into "the hypothesis
   was wrong" would be as inaccurate as claiming the win: the waste was genuine and
   removing it was correct, and the kernel was never issue-bound. At 74% of peak the live
   hypothesis is now **x re-read amplification** — all 6144 blocks stream the whole 64 KB
   of x for 16 KB of weights, 402 MB of x traffic against 100 MB of weights — which is a
   cache-hierarchy question the ISA cannot answer and `ROWS_PER_BLOCK` tiling is the fix
   for. See benchmarks.md, "Read the ISA before you book the device".
   **THIS ITEM IS MIS-SCOPED AS AN o_proj FIX — it is route-wide.** `fp8_dot_strided` is
   the shared helper behind *every* fp8 block-scaled GEMV: `o_proj`, `q_a`, `q_b`, `kv_a`
   and the dense MLP. Measured in-engine by a three-arm decomposition, the shift was worth
   **−2.5 ms/tok, 2.5× the −0.98 ms o_proj alone accounts for.** Any further work on this
   helper — load widening, x re-read tiling — inherits the same multiplier, so it is worth
   more than its o_proj row suggests.

   **x RE-READ AMPLIFICATION IS REFUTED. Measured, and refuted in the direction opposite
   to the prediction.** `ROWS_PER_BLOCK`-style tiling was implemented (one pass over `x`
   feeding R weight rows, bit-identical) and swept interleaved against the untiled arm,
   5 samples each, min-of-N because the bus was contended:

   | `SPLITK_ROWS` | x traffic | o_proj min µs | vs untiled |
   |---|---|---:|---:|
   | 1 (untiled, shipped) | 402 MB | **515.7** | — |
   | 1 (via the tiled helper) | 402 MB | 615.0 | +19% |
   | 2 | 201 MB | 508.3 | −1.4% |
   | 4 | 100 MB | 531.5 | +3.1% |
   | 8 | 50 MB | 573.6 | +11.2% |

   If the x re-read were binding, R=8 would be the fastest arm. It is the **slowest**, and
   the trend is monotone the wrong way for R ≥ 2. The best arm (R=2) buys 1.4%, inside a
   noise band that spanned 515–1141 µs on the untiled arm alone. **The tiling was reverted
   — it added a templated multi-row helper, pointer arrays and a tail-block clamp for an
   effect indistinguishable from zero.**

   **The arithmetic said so before the device did.** The 402 MB + 100 MB in 529 µs this
   item rests on is 950 GB/s on a 256 GB/s part — 3.7× over, so it was never DRAM traffic
   and one division would have retired the item without a device slot. See benchmarks.md,
   "Divide by the peak before you book the slot".

   **What o_proj actually is: at the roofline for the traffic it cannot avoid.** The
   100.7 MB of weights it must read from DRAM is 393 µs at peak; it measures 515.7 µs
   min = **76% of peak, with a competing memory-bound job on the same unified bus.** The
   remaining headroom is ~24% and no restructuring of the x side can reach it.

   **The `o / block` scale-row divide (a separate defect, found in the same ISA pass) is
   fixed and measured.** `size_t o / int block` in both `gemv_fp8` and `gemv_fp8_splitk`
   emitted a runtime division per thread — a 32-bit fast path plus a full 64-bit one
   behind `v_cmp_ge_u64`, ~55 instructions, and the split-K prologue dropped 379 → 349
   static instructions when it became a shift. Worth **−0.9% (min-of-6, 518.6 → 513.9 µs)
   — inside the noise band**, NOT the 20% the identical fix bought in `mla_value_fp8`,
   because `o` is uniform per block here so LLVM emits it in SALU and 256 threads amortize
   one sequence. Shipped anyway: bit-identical, free, and the waste was real. This is the
   third instance of the pattern and the second time it did not pay — *finding* an integer
   division in a hot kernel is now a reliable prediction; **its magnitude is not.**

3. **`mla_latent_attend` occupancy** *(route; scales with context — do before any
   long-context push).* ~20 ms @ nr512, grows ~linearly with context. LDS-capped to 1
   WG/CU (dynamic LDS `((HB+TILE)·kvl + TILE·rope)·4 = 53 KB`). Levers: move `acc[HB·kvl]`
   (16 KB) to registers *or* HB 8→4 → 2 WG/CU; lower `MLA_MIN_TILES_PER_SPLIT` 4→2 so short
   context spawns enough splits to fill the 40 CUs. Est ~5–7 ms at short ctx, much more at
   long ctx.
   **`acc` → registers is done: measured −12.0% at nr512 (258.0 → 227.2 µs, −2.41 ms/tok)
   and −11.2% at nr2048 (876.3 → 778.5 µs)**, i.e. a roughly constant *fraction*, so the
   absolute saving grows with context as this item predicted. And the bigger effect was
   not occupancy — it deletes an LDS read-modify-write from the innermost loop (the
   rescale did 2 LDS reads + 1 LDS write per owned column per attended token; now 1 read
   and a register FMA). LDS 52 → 36 KB, VGPRs 33 → 47, no spill. The stated ~5–7 ms
   estimate was for the whole item including the occupancy work, which is NOT done —
   see below. Two consequences for the remaining levers:
   - 36 KB is **still 1 WG/CU** (needs ≤32 KB). `TILE` 16→14 would reach 31.5 KB, but
     `TILE` feeds `ntiles` → the split plan → summation order, so it is a numerics change
     and needs the gate. **Prefer the HB route below** — it reaches the same occupancy
     without a numerics change, and if it holds up, `TILE` 16→14 should be struck rather
     than kept as an alternative: two routes to one goal, one gated and one not, is an
     invitation to take the wrong one.
   - `MLA_MIN_TILES_PER_SPLIT` 4→2 appears **inert at nr=512 — but only at HB=8**, and
     the qualifier is load-bearing. `by_grid` = ⌈MLA_TARGET_BLOCKS/hblocks⌉ = 10 binds
     before `by_work`, and `tps` rounds back to 4. Raise HB and `hblocks` halves, so
     `by_grid` doubles and this knob starts to bite. Do not read "inert" as a dead end.
   - **`HB` is now decoupled from LDS entirely**, which was not true before. HB is the
     DRAM KV re-read multiplier (⌈H/HB⌉×) and this file already calls that "the dominant
     term at long context". Raising HB 8→16 halves it, and is **free in registers**
     (measured: 47 VGPRs, 0 scratch, no spill at both HB=8 and HB=16 — allocation is
     per-thread and `acc` is sized by kvl/SUBW, so nothing in it scales with HB). It also
     doubles waves/SIMD at 1 WG/CU (4 → 8), which is the occupancy win `TILE` was for.
     **But it halves `grid.x`**: at nr=512 the default plan drops to 4×8 = 32 blocks on
     40 CUs, 8 idle. Total waves are unchanged (32×16 vs 64×8), and packing them onto
     fewer CUs trades more latency hiding per CU against fewer independent memory
     pipelines — which wins is not derivable from wave counts. So this is a **two-parameter
     sweep (HB × MLA_MIN_TILES_PER_SPLIT), not a one-line change**, and deserves its own
     entry and its own measured slot.

4. **`mla_absorb_fp8` restructure** *(route).* ~5 ms @ 99 GB/s.
   **DIAGNOSIS SUPERSEDED.** This item blamed the transpose direction (1 thread/(head,i),
   kvl-strided single-byte loads, low ILP) and prescribed `mla_value`'s wave-per-row
   float4 form. The ISA says the dominant cost was neither: `kvb_scale[(row / block) *
   sc_cols + ...]` with `size_t row` is a **64-bit unsigned division inside the `d`
   loop**, which LLVM cannot strength-reduce — it emitted an inline Newton-Raphson
   reciprocal, 498 static instructions around 10 memory ops. Fixed with a shift:
   **measured 72.0 → 36.5 µs, 87.4 → 172.3 GB/s, 1.97×** (−2.77 ms/tok over 78 layers).
   **The restructure IS still worth doing, and the reason the old target was misleading:**
   this item judged absorb's 99 GB/s against "`mla_value`'s 254" — but `mla_value` carried
   the *same* 64-bit divide, so the reference was depressed. Post-fix the real comparison
   is **172.3 vs 310.3**, absorb still ~1.8× off its sibling. Prefer one thread per
   **(head, i-quad)** — 4 columns, 4 accumulators — over the wave-per-row form: same
   coalescing and ILP, but it keeps each output's sum over `d` ascending, so it is
   bit-identical and needs no quality gate. Wave-per-row is not.

   **DONE, and the i-quad prescription was right on both counts.** One thread per
   (head, i-quad): **35.9 → 25.7 µs, 175 → 245 GB/s, 1.40×** on the 6.29 MB of kv_b it
   reads (min-of-N over interleaved A/B; `mla_value` held at 26.1 µs in both arms as an
   unchanged control). Over 78 layers that is **−0.80 ms/tok**, on top of the divide
   fix's −2.77. Both figures are min-of-N from this branch's own A/B and are NOT the
   36.5 µs / 172.3 GB/s recorded above, which came from a different session — compare
   within a run, not across them.

   **The mechanism was LOAD WIDTH, exactly as `lm_head`'s (#5) is.** One column per
   thread meant `global_load_u8` — 32 lanes × 1 byte = **32 B/wave against a 128 B cache
   line**. A quad's four columns are contiguous, so the same bytes arrive as one
   `global_load_b32` = 128 B/wave, a full line. Normalized per four columns the ISA goes
   from 4 weight loads / 4 scale loads / 66 VALU to **1 / 1 / 27** — 4× fewer memory
   instructions and 2.4× less VALU for identical bytes moved. VGPRs 29, no spill, no
   scratch, occupancy unchanged at 16 waves/SIMD, and `s_and_saveexec_b32` went 2 → 1
   (removing a cost without adding a neighbouring one — the check benchmarks.md asks for).

   **Bit-identical, and proved rather than argued:** absorb's output fingerprint is
   `0925c147afeea3fb` in **all 14** interleaved runs across both arms. No quality gate
   needed, as the item predicted. See benchmarks.md, "A fingerprint is the only instrument
   that shows bit-identity".

   **Two preconditions the quad path cannot meet, both routed to the original scalar body
   rather than rejected in the launcher:** `kvl % 4 != 0` (rows unaligned for a dword
   load) and `block < 4` (a quad would straddle two scale tiles). Both kept legal because
   the Vulkan launcher accepts them and the backends must span one domain.

   **This turned up a latent correctness defect in `fp8_dot_strided` itself.** Its dword
   path has always applied ONE block scale to a quad's four columns, which is wrong for
   `block` 1 or 2 — both powers of two, so guard 1003 passes them, and
   `gemv_fp8_rejects_non_power_of_two_block` *explicitly requires* block=1 to be accepted.
   Every fp8 GEMV and `mla_value_fp8` was silently mis-scaling three columns in four at
   those tiles. Found by the block=2 shape added alongside the absorb work, fixed by
   zeroing `n4` so the already-correct per-column tail takes the row, bit-identical at
   block ≥ 4. **The Vulkan twin (`kernels/vk/fp8.glsl`) still has it, and tests/vk.rs's
   oracle mirrors the bug so that backend cannot see it either** — recorded in
   `kernels/common.hpp`, not fixed, because this branch had no Vulkan device slot.
   No shipped model is affected: every fp8 checkpoint uses `weight_block_size` 128.

5. **`lm_head` split-K** *(tail).* [154880, 6144] int8 GEMV — the one big tail cost. Split-K
   it (same shape argument as o_proj: many rows, long reduction).
   **MEASURED: 8.12 ms at 117 GB/s, so the ceiling on this item is ~4.4 ms** (8.12 → 3.71
   at peak). An earlier revision of this line said "~10 ms, not a few" by assuming `tail`
   was almost entirely lm_head; it is about half. The original "a few ms" was right.
   **`tail` cannot be fixed here**: lm_head 8.12 + argmax 0.088 + rmsnorm 0.008 ≈ 8.2 ms
   of a ~16 ms bucket, and those are its only kernels — see benchmarks.md, "Open question:
   half of `tail` is in none of its kernels".
   **And split-K is probably not the fix.** o_dim=154880 already launches 19,360 blocks,
   so the machine is full — the grid argument that motivated o_proj's split-K does not
   transfer. The ISA shows the real defect: the inner loop is a single `global_load_i8`,
   **one byte per lane per iteration** = 32 B/wave against a 128 B cache line, ~19
   instructions per weight byte (o_proj's fixed loop is ~7). The fix is the load width —
   4 int8/lane plus a float4 `x`, i.e. exactly `dot_i4_wave`'s existing shape. Unlike the
   route items this repartitions each lane's columns, so it is **not bit-identical** (f32
   reassociation) and, since lm_head feeds argmax directly, it takes the full ppl gate.

---

## Ranked roadmap

| # | Item | Path | Est. impact | Effort | Status |
|---|---|---|---|---|---|
| 1 | `fp8_to_i4` + `--hot-pct` re-tune | B | med (more int4, faster compute) | low | fp8 downloading |
| 2 | VQ_K=2048 L1-resident codebook | B / follow-up #1 | med (lifts gather wall) + smaller experts | med (requant) | new |
| 3 | Batched-GEMV kernels | A | med now, unlocks MTP | med–high | new |
| 4 | MTP / speculative decode | A | high (≥1.5×) | high | needs #3 + fp8 MTP head |
| 5 | `mla_latent_attend` occupancy | follow-up #3 | ~5–7 ms now, huge at long ctx | med | new |
| 6 | o_proj / lm_head split-K | follow-up #2, #5 | lm_head only (o_proj refuted) | low | partly closed |
| 7 | `mla_absorb` restructure | follow-up #4 | **−0.80 ms/tok, measured** | med | **done** |

**Suggested sequence:** land #1 (cheap, in flight) → **Path B** (#2 — the biggest
structural win on the 210 ms) → **Path A** (#3 batched kernels → #4 MTP, the multiplier) →
per-kernel follow-ups #5–7 as route/tail polish (#5 mandatory before any long-context
work).

**Measurement discipline (learned the hard way):** rank format/numerics changes by (a) the
replay residency sim, (b) a fixed forced-token wall-clock bench, and (c) perplexity for
quality — **never** free-running greedy tok/s (confounded by output degeneration; a broken
run looks fastest). See [MODES.md](MODES.md) and [benchmarks.md](../benchmarks.md).
