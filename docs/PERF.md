# rivoli — performance plan

> **NAV — 39 KB. The live part is the "Ranked roadmap" table at the BOTTOM.**
> Everything above it is the evidence for one row. Jump there first
> (`grep -n "Ranked roadmap" docs/PERF.md`), then read only the section a row cites.
> **Closed as negative — do not re-open without reading why:** batched-GEMV/MTP
> speculative decode (#4, re-derived 2026-07-31 at 0.93–0.95×), o_proj split-K (#6b,
> refuted and reverted), `--hot-pct` (#1b, flag deleted).

Status: **analysis + roadmap.** Leads with the **structural paths** (the higher-level
goals that move throughput by a multiplier or restructure a whole phase); the per-kernel
findings are **follow-up polish** to that structural work and to the existing improvement
proposals ([CACHE_ROUTE](CACHE_ROUTE.md), [CACHE_PILOT](CACHE_PILOT.md), the fp8-int4
work). Residency / cache-conditional routing is **not** covered here — it is owned in full
by those two proposals.

**A correctness defect found by this document's per-kernel work does NOT live in this
document.** The `route` tranche turned up an fp8 block scale mis-applied at `block < 4`,
affecting every fp8 block-scaled GEMV — it is written up in **benchmarks.md, "Bugs found
and fixed"**, because someone auditing numerics reads that file and someone tuning kernels
reads this one. Two known-broken twins are recorded there too. Perf items cross-reference
the bug; they do not host it.

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

**All four corrected mechanisms have now been implemented and measured, and the ISA was
right every time** — absorb's load width 1.40×, lm_head's load width 1.78×, attend's LDS
read-modify-write −12%, and o_proj's divide, which was real but not binding. So the
column that matters is not "the plan was wrong four times out of five"; it is that **the
ISA reading was right four times out of four.** One caveat carries forward, from o_proj:
the ISA tells you the mechanism reliably, and **the magnitude not at all** — a 34% VALU
cut bought 2.3% there, while the same class of fix bought 78% here. Use it to choose what
to implement, then still measure what it was worth.

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

## What the time IS — the CLASS axis (always on, `class/tok`)

The phase buckets below (`route` / `moe-gpu` / `tail`) say **where** time goes. They are
*regions*, and each mixes host compute with blocking waits — which is why `tail` spent this
document's whole life with most of itself attributable to no kernel. The `class/tok` line
cuts the same work by **activity**, and every term is a stamped span:

```
class/tok [spans overlap; no residual]:
    gpu-wait 321.4ms (95% of wall) | io-wait 165.1ms (49%) | cpu 6.2ms (2%)
    cpu = launch 2.6ms + route 0.31ms + submit 0.87ms + tokio-poll 2.4ms
split/tok: route = 101.1ms gpu-wait + 0.3ms host-routing
           tail wait 5.5ms, of which 5.5ms is GPU
```
*(hybrid+lru, `-bench 128`, `--max-mem 100`, wall 338.2 ms/tok.)*

These are **spans, not just counters** — set `RIVOLI_SPANS` and they export as real OTLP
spans with true start/end times across both threads, so a trace viewer draws the overlap
instead of implying it. See [TRACES.md](TRACES.md).

**These spans OVERLAP and do not sum to wall — by design.** `io-wait` is the reaper thread
blocked in `io_uring` while the decode thread computes; it is 54% of wall precisely because
it is concurrent with it, and 95% of it is hidden. Forcing these into a partition is what
broke the first version: it made `io-wait` a *derived* `moe_wall − compute_gpu` (a host
clock minus a GPU clock) which reported **8.4 ms**, and `cpu` a residual. The measured
io-wait is **183.7 ms — 20× larger**. The derived number was not measuring io-wait at all;
it was measuring the *unhidden remainder*, which is a different and much smaller thing.

**The price, accepted deliberately: no residual is reported.** Measured `cpu` is 6.5 ms
where the old residual said 8.8 — so ~2.3 ms/tok is genuinely unattributed. It is now
invisible rather than dressed up as host compute. A residual bucket absorbs every error in
every other term, which makes it the one number in a profile that cannot be wrong and
cannot be useful.

**Three things this settles.**

1. **We are GPU-bound, and host compute is negligible — 6.2 ms, under 2%.** Now measured
   in four named pieces rather than inferred: kernel launch 2.6 ms (~1.5k driver calls),
   tokio poll 2.4 ms, `submit_layer` 0.87 ms, `route_into` 0.31 ms. Any plan premised on
   host overhead is chasing 2%, and we know which 2%.
2. **`route` is not routing.** Of its 101 ms, **101.1 ms is a blocking D2H wait and 0.31 ms
   is the actual host routing** — top-k over 256 experts across 75 layers costs nothing.
   `route` is an attention-GPU phase wearing a host-phase name.
3. **`tail`'s missing half was never a kernel.** The argmax D2H is 5.5 ms and effectively
   all of it is GPU. The rest of the old `tail` bucket is decode-loop host work now
   itemised above. **benchmarks.md's "half of `tail` is in none of its kernels" is
   answered, and the answer demotes it.**

**Measuring io-wait properly exposed a pre-existing bug in `fetch_wall_ms`.** `hits` and
`misses` are rebased when the profile resets after prefill; `fetch_ns` never was, so every
`fetch` number this project has published folded the prefill's cold, expensive fetch into
the decode average. It was invisible at `-bench 512` (5 prompt tokens amortize away) and
obvious the moment the new io-wait was read at `-bench 8`, where it reported **136% of
wall**. Both counters are now baselined: at `-bench 128` **`fetch` drops 184 → 165 ms/tok,
an 11% correction** to a long-standing figure. Older `fetch` numbers in benchmarks.md are
high by roughly this much, and by more at small `-bench`.

### How far to trust each bucket

| bucket | confidence | why |
|---|---|---|
| `gpu-wait` | **high** as *host state* | Stamped at every blocking call; audit found none unwrapped on this arm. |
| `io-wait` | **high** | Stamped around `run_job`'s reap loop at the ring, excluding queue/submit. Independently ≈ `fetch_wall` (165.1 vs 165.0), which is the expected agreement. |
| `cpu` | **high, but a lower bound** | Four stamped regions. Anything host-side outside them is unattributed and *not shown* — hence the ~2.3 ms gap to the old residual. |

**`gpu-wait` is not utilisation, and the gap is measured.** Sampled through a decode,
`rocm-smi` reports **83.9%** GPU busy (n=65 steady-state, 77–91%) against 95% gpu-wait.
Both are right and measure different things: ~11 points (~38 ms/tok) is the host blocked on
a GPU that is *not* executing — launch gaps, driver and queue-drain overhead. **Reading
`gpu-wait` as utilisation overstates it by an eighth.**

**But put wide error bars on the 38 ms.** `gpu-wait` is stamped and trustworthy; the 83.9%
it is differenced against is an *instantaneous SMU register read*, not a time-integrated
counter, from a tool that in the same breath reports 512 MiB of VRAM (the real unified pool
is 116 GiB of GTT), a 0% fan, and throws `map::at` on every invocation. It is also coarse
and rolling-averaged — ~12 s to track a step.

**~6 ms of the 38 is now measured rather than guessed:** the device-side kernel dispatch
floor is 1.97 µs × ~1500 kernels ≈ **3.0 ms/tok**, and the host→GPU join tax is 11–20 µs ×
~180 joins ≈ **2.7 ms/tok**. A clean negative came with it — `hipDeviceSynchronize`,
`hipStreamSynchronize` and `hipEventSynchronize` all cost the same, and spinning on
`hipEventQuery` is *strictly worse* — so there is no cheap win in swapping sync primitives.
**The remaining ~30 ms has a suspect already named in `src/gpu.rs`: per-expert host-gated
launch bubbles, which fall inside `compute_gpu_ns` and are why it is an upper bound.**
Resolving this needs GPU-timeline spans on the host clock — see [GPU_TRACE.md](GPU_TRACE.md),
which also shows that is ~4 hours of work and that the clock problem is already solved.

**Audit — what is stamped.** Every blocking call in `src/gpu.rs` was enumerated. On this
arm (`--attn dense`, `topk=device`) the decode path blocks in exactly three places: the
gate-logits D2H, the argmax D2H, and the end-of-layer `device_sync`. `pin`/`hybrid`/
`stream`/`arena` contain **no** blocking calls; expert fetch is async throughout, and the
one place it really waits — the reap loop — is where `io-wait` is now taken. Two indexer
D2Hs were unwrapped and would have corrupted the `--attn dsa` arms (invisible under
`--attn dense`, where `dsa_select_layer` never runs); both are fixed. Left unwrapped
deliberately: the `TopkPath::Verify` debug arm, the `--ppl` logits D2H, and `--checksum-x`.

**One overclaim corrected.** The tail HIP-event span was documented as the tail kernels'
execution time; it is a *bracket*. It measures 5.50 ms against a 4.66 ms microbench sum, so
**~0.84 ms (15%) is inter-kernel gap** — the caveat `idx_gpu_ns` already carried. It is an
upper bound, and the reported "0% overhead" is 0.1 ms display rounding, not a measured zero.

Expensive per-kernel detail stays behind the `trace` feature; this axis is free — it rides
joins the forward pass already pays and adds no sync.

## Where the time goes (hybrid+lru, 512 tok, the best coherent config)

**Status of `route`: the per-kernel tranche on `feat/perf` measures `route` 112 → 104 ms
in-engine** (interleaved A/B, flat control, identical miss counts, byte-identical output).
The 115 ms below is the pre-tranche figure. See benchmarks.md, "In-engine confirmation".

**Status of `tail`: lm_head's quad-load fix (follow-up #5) takes it 8.12 → 4.56 ms, and the
bucket measures 14.5 → 11.3 ms in-engine.** The ~16 ms below is the pre-fix figure. Two
caveats on reading it: ~62% of the bucket is now in none of its kernels, so do not treat
11.3 as "lm_head plus change" — and **this 3.2 ms is only ~1% of wall, which no end-to-end
bench here can resolve** against 16 ms of moe/fetch drift. `tail` is ~4% of the token; it
is not where this engine is slow.

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

**THIS PATH WAS ALREADY BUILT ONCE, MEASURED, AND CLOSED AS A NEGATIVE RESULT — and this
document was written as if it never happened.** Branch `deadend/mtp` carries the whole
chain: weight extractor → scalar oracle → device draft → **a batched-S fused MoE kernel
(this is exactly the "batched-GEMV" enabler above)** → a greedy-equivalent spec loop →
prefetch, closed by `479c430` "docs(mtp): close investigation — 256-tok sweep,
regime-dependent negative result". Read `git show 479c430:docs/mtp.md` before re-opening
either item. Measured @256 tok: **baseline 1.05 tok/s, warm-budget 0.95, union-tree
width-1 0.68** — speculation lost in *both* regimes, for opposite reasons. Cold is
NVMe-bound and a 2-position union reads **+16% bytes/tok**; warm is compute-bound and the
layer-78 draft is a full extra 256-expert MoE layer, so the `warm-budget` arm hit **84%
accept and still lost 10%**. Note that 84% is essentially the "~85% accept" the estimate
above assumes — **the accept rate was not the thing that was wrong; the cost of the draft
was.**

**The caveat that keeps this from being a closed door.** Those runs sit on merge-base
`b5c2ed8`, and main was rebuilt from scratch at `fd18238` ("empty slate — rebuild
int3-vq-only engine from scratch"). They predate int3-vq, the group-128 `.i4` fix, the DSA
device top-k, and the route/absorb wins. So this is strong evidence about a *previous*
engine, not a measurement of this one. But it inverts the burden of proof: the next person
does not get to estimate ≥1.5× from accept rate, because that estimate has already been
run and it lost at 84% accept. **What must be re-derived first is the draft's cost against
today's `moe-gpu`, and that is a one-afternoon estimate, not a re-implementation.**

**Ceiling: ≥1.5× on paper, measured <1.0× once. Effort: high. Sequence: re-derive the
draft cost → batched kernels (independently useful) → MTP.**

### Path B — MoE format program: restructure the 210 ms phase

The MoE dot has two separable costs — the int3-vq **gather-throughput wall** (~53 GB/s,
24% of bus, already at max occupancy, so only a *format* change moves it) and int4's
**residency tradeoff** (no gather, ~1.8× compute, but 18.9 vs 15.3 MB). Attack both as one
program behind a shared quality gate (perplexity on fixed text — never free-running tok/s):

1. **`fp8_to_i4`** — derive int4 from the original fp8 (higher fidelity than via vq3;
   download in flight) → enables a larger coherent hot fraction.
   **DONE, and it took a second fix to actually land.** `src/bin/fp8_to_i4.rs` ships, but
   the first set was per-row-scaled and int4 PPL was **73.43** — unusable. The mechanism
   was scale *granularity*, not provenance: `I4_GROUP = 128` scales along the input dim
   took int4 to **PPL 5.120** and hybrid to **5.189**, beating the int3-vq control's
   5.275. Hybrid is now the best config in the engine on quality *and* speed. Full
   write-up in [INT4.md](INT4.md).
2. ~~**`--hot-pct` re-tune**~~ — **STRUCK: the flag does not exist.** It was introduced
   (`dca66a7`), replaced by a cold-slab floor (`b842c60`), and then deleted outright with
   the fixed-partition cache variants and the replay `--hybrid` split simulator
   (`c876a8d`) — *"the split self-sizes now"*. The hot/cold boundary is emergent today:
   the two-ended byte arena packs cold from the low end and hot from the high end and the
   boundary **floats** with the policy's tier decisions (`src/arena.rs:1`, `src/pin.rs:674`).
   **So this row cannot be run as written, and the substitute is not `--max-mem`** — a
   cross-budget hybrid A/B is confounded by construction, because changing the budget
   changes which experts sit in which format, i.e. changes the arithmetic being compared
   ([MODES.md](../MODES.md)). Re-specifying it means re-introducing a floor override to
   sweep against. Until then the honest status is *unrunnable*, not *pending*.
   (`MODES.md` and `README.md` still document `--hot-pct` and the deleted `::fixed`
   variants as live — three stale sites, worth a pass.)
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

   **This work also turned up a CORRECTNESS defect in `fp8_dot_strided`** — the block
   scale was mis-applied at `block < 4` in every fp8 GEMV. It is a numerics bug, not a
   perf one, so it is written up under **benchmarks.md, "Bugs found and fixed"** rather
   than here, along with the two known-broken twins left in place (the Vulkan shader,
   whose oracle mirrors the defect, and `rivoli_gemv_fp8`'s missing `i_dim % 4` guard).
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

   **DONE, and this is the first item in this document whose mechanism was right the first
   time.** One dword of weights (4 int8) plus one float4 of `x` per lane:
   **8121.8 → 4555.1 µs, 117.2 → 208.9 GB/s, 1.78×** — min-of-5, interleaved
   base/fix/base/fix, spread inside each arm ~0.1%. That is **−3.57 ms/tok**, capturing
   **81% of the ~4.4 ms ceiling** this item computed. Controls measured in the same runs
   held flat: `o_proj` 527.9 → 526.2 µs, `mla_absorb` 25.9 → 26.0, `rmsnorm` 7.7 → 7.7,
   `argmax` 87.2 → 87.2.

   **The ISA predicted the magnitude before the device confirmed it**, normalized per four
   columns: **8 memory instructions → 2** (one `global_load_b32` for the weight quad and
   one `global_load_b128` for `x`, replacing 4× `global_load_i8` + 4 scalar `x` loads) and
   **36 VALU → 17**, for identical bytes moved. The sign-extends compiled to three
   `v_bfe_i32` plus one `v_ashrrev_i32`. VGPRs 18, zero scratch, zero spill, occupancy 16
   waves/SIMD (the maximum) — and, the neighbouring-cost check benchmarks.md asks for,
   **`s_and_saveexec_b32` inside the hot loop stayed at 0**; the kernel's two are the
   prologue guard and the tail-loop entry, both outside it.

   **Quality gate: PASS, and not an underpowered one.** Teacher-forced over the 762-token
   corpus, hybrid + lru + `--max-mem 100`, one process per arm: **PPL 5.189425 → 5.189426**,
   mean dNLL 0.00000, 95% CI [−0.00000, +0.00000], `worse%` 41.5. The `ppl` tool warns that
   a null may be absence of evidence rather than evidence of absence; that warning does not
   apply here, because **the per-token sd is 0.0000, not merely the mean** — the
   reassociation moved logits at f32 rounding only. The base arm reproduced
   [INT4.md](INT4.md)'s recorded hybrid 5.189 exactly, so this is paired against the
   published baseline rather than a fresh one.

   **Two guards, and why the fast path is not silently dead in the engine.** The quad path
   needs `row` dword-aligned and `x` float4-aligned; anything failing either falls to the
   original scalar loop, so `i_dim % 4 != 0` stays legal (the Vulkan launcher accepts it,
   and the backends must span one domain). Both hold in the engine — `hipMalloc` is
   256-byte aligned and lm_head's i_dim is 6144 — but **a microbench allocates fresh
   buffers and would show the win even if the engine's `x` were misaligned**, so the
   alignment was traced to the call site rather than assumed. `tests/kernel.rs` now sweeps
   6144 / 6148 / 100 / 6143 to reach the pure fast path, fast-path-plus-tail, tail-only,
   and the de-aligned fallback.

   **The Vulkan twin was deliberately left alone** — it is the portability backend, not the
   perf path, and it carries its own oracle against the same host reference. It keeps the
   1-byte-per-lane shape and does not get the speedup.

   **This makes `tail`'s unexplained half worse, not better.** lm_head 4.56 + argmax 0.089
   + rmsnorm 0.008 ≈ 4.65 ms now sits in a bucket that should have fallen to ~12.4 ms, so
   the share attributable to none of its kernels goes from about half to **roughly 62%**.
   That open question is now the larger of the two and should outrank any further lm_head
   work — the remaining headroom here is only ~0.84 ms (82% of peak already).

   **IN-ENGINE: the bucket moved exactly as predicted, and the wall did not noticeably.**
   `-bench 256`, hybrid + lru + `--max-mem 100`, interleaved base/fix ×3. Miss counts were
   **identical in all six runs** (40202 miss, 148.55/tok, 2.28 GB) so the greedy sequence
   never diverged — the free-running-decode confound MODES.md warns about did not fire, and
   was checked rather than assumed. `route` held at 102.3–102.9 ms in **both** arms, a clean
   control confirming the change is isolated to `tail`.

   | | base (3 reps) | fix (3 reps) | Δ (min) |
   |---|---|---|---:|
   | `tail` (wall − route − moe) | 14.5 / 14.8 / 16.5 ms | **11.3 / 11.7 / 12.4** | **−3.2 ms** |
   | wall | 345.3 / 348.4 / 362.8 ms | 344.2 / 351.3 / 358.7 | −1.1 ms (noise) |
   | tok/s | 2.90 / 2.87 / 2.76 | 2.91 / 2.85 / 2.79 | — |

   **`tail` fell by 3.2 ms in every one of the three pairs** (−3.2, −3.1, −4.1), matching
   the −3.57 ms microbench prediction. **Wall did not resolve it**, because `moe`/`fetch`
   drifted upward by 16 ms over the six runs (moe 228 → 244, fetch 178 → 193) — drift an
   order of magnitude larger than the effect. Two lessons, both cheap to reuse:
   - **Judge a per-kernel change on its own bucket, not on wall.** `tail` is ~4% of wall,
     so even a perfect fix here is a ~1% end-to-end change and no end-to-end bench at this
     n will ever see it. The phase profile resolved it on the first pair.
   - **The interleave order biased against the fix and it still won.** With
     base,fix,base,fix,… and monotone upward drift, the fix arm always ran *later* inside
     each pair and so absorbed more drift. Alternating the leading arm would remove this;
     it was not worth a re-run here because the bucket delta survived the bias.

   **So: real, measured, and imperceptible.** ~2.90 → ~2.93 tok/s, about +1%. This is the
   strongest available argument for the document's own thesis — `tail` polish cannot move
   this engine, and the prize is `moe-gpu` (60%) and `route`.

---

## Ranked roadmap

| # | Item | Path | Est. impact | Effort | Status |
|---|---|---|---|---|---|
| 1 | `fp8_to_i4` | B | int4 PPL 73.43 → **5.12**, hybrid → **5.19** | low | **done** |
| 1b | ~~`--hot-pct` re-tune~~ | B | — | — | **struck — flag deleted, unrunnable** |
| 2 | VQ_K=2048 L1-resident codebook | B / follow-up #1 | med (lifts gather wall) + smaller experts | med (requant) | new |
| 3 | Batched-GEMV kernels | A | med now, unlocks MTP | med–high | **done, on `main`** — 6 kernels take `nrow`, bit-identical per row |
| 4 | MTP / speculative decode | A | ≥1.5× on paper, **0.93–0.95× measured** | high | **RE-DERIVED 2026-07-31 and closed again — the cause is now known** |

> **Item 4 was re-derived as this table asked, and reached the same verdict by a route that
> explains it.** Shipped end to end (`docs/ARCHITECTURE.md` §13): 2.50 vs 2.69 tok/s at 128
> tokens, 2.49 vs 2.63 at 512, output byte-identical. The mechanism is arithmetic, not
> tuning: the MoE is 67% of the pass and a batched pass launches the **UNION** of both rows'
> routing — 14.5 experts against a single row's 9, so **1.61× the weight reads** — while the
> second row per expert is genuinely free (178 vs 176 µs on 0-miss layers). Attention
> behaved as designed (0.83× per token). Break-even is **1.53 tokens/pass ≈ 53% acceptance**
> and measured acceptance is 42–54%.
>
> So it is a coin flip landing slightly wrong, not a structural impossibility, and the ONLY
> lever is acceptance — skipping zero-weight rows inside the kernel would recover ~8%,
> because ~92% of an expert launch is the weight read. **Do not re-open without a draft head
> that clears 53%.** GLM-5.2 ships one MTP layer and depth-2 chains accept at 4.4%, so that
> head is not available in this checkpoint.
| 5 | `mla_latent_attend` occupancy | follow-up #3 | ~5–7 ms now, huge at long ctx | med | **partial** — `acc`→regs done (−12%); HB sweep not run |
| 6a | lm_head load width | follow-up #5 | kernel **1.78×**; `tail` **−3.2 ms** in-engine; wall **~+1%, not noticeable** | low | **done** |
| 6b | o_proj split-K / x-tiling | follow-up #2 | — | — | **refuted and reverted** |
| 7 | `mla_absorb` restructure | follow-up #4 | **−0.80 ms/tok, measured** | med | **done** |

**Suggested sequence, revised.** The original sequence led with #1 and treated #4 as the
big multiplier; #1 has landed and #4 has a measured loss against it, so:

1. **#5's HB × `MLA_MIN_TILES_PER_SPLIT` sweep** — the cheapest unclaimed win, and still
   mandatory before any long-context work. Note it is a **4-site change** (`kernels/attn.hip`,
   `kernels/vk/mla_latent_attend.comp`, and the two mirrored launcher constants in
   `src/vk.rs`), which the item's text does not say.
2. **Path B (#2)** — the biggest remaining structural lever, on the 210 ms that is 60% of
   wall. Now the *only* live item in Path B, since 1b is struck.
3. ~~**`tail`'s missing ~62%**~~ — **ANSWERED and struck.** The CLASS axis shows it is
   decode-loop host CPU (~6 ms of the 8.9 ms `cpu` bucket), not a hidden kernel. Promoting
   it was the right call on the evidence available; measuring it cost one run and demoted
   it, because total host compute is under 3% of the token.
4. **Path A (#3 → #4)** only after the draft cost is re-derived against today's `moe-gpu`.
   Do not re-estimate it from accept rate: 84% accept has already been measured, and lost.

**Measurement discipline (learned the hard way):** rank format/numerics changes by (a) the
replay residency sim, (b) a fixed forced-token wall-clock bench, and (c) perplexity for
quality — **never** free-running greedy tok/s (confounded by output degeneration; a broken
run looks fastest). See [MODES.md](MODES.md) and [benchmarks.md](../benchmarks.md).
