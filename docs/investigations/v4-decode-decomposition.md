---
scope: v4
status: live
verdict: OPEN, four levers LANDED and the route span SPLIT. M3b (2026-08-07, launch geometry): moe 70.6 → 54.7 ms/token, +9.6% tok/s. M3c (2026-08-08, branchless e2m1/e8m0 + global-AS descriptor loads, fp4 dot loop 195 → 105 instr per 128 weight bytes): moe 54.9 → 49.6, wall 165.3 = 6.048 tok/s — the registered prediction (−4..−10, point −6) landed IN BAND at −5.3, the first of the series to do so. M4 (2026-08-08) split route with four event-pair sub-spans, no new join: attn 53.2 | cmp 9.1 | hcn 41.2 | gate 3.2 | win 107.7 (resid 1.0) — WRONG at the headline: hcn (hyper-connections + norms) measured 41.2 against a 4–8 band and 0.8 ms of bytes, the engine's largest above-bytes excess; cmp and gate closed at budget; resid 1.0 kills the gate-D2H width micro-lever. M5 (2026-08-08, the hcn read + fix, schedule-only: hc_pre 256 → 1024 threads = one wave per mix row, sum-of-squares frozen at HC_RED, unroll-8): hcn 40.7 → 5.6, wall 130.7 = 7.652 tok/s (+26.5% over M3c's 6.048; arm-to-arm +28.7%), Δhcn −35.1 vs the registered −15..−30 band — the error family a third time, the good side a second (M4 missed high); the width and unroll levers MULTIPLY (24× loads in flight); hcn CLOSED at this rung, +4.8 above bytes. Output byte-identical and counters identical across every arm (M5: reply-prefix md5 recorded, escape-decoding to M4's recorded 1983-byte md5). Residue to 10 tok/s (~31 ms off the 130.7 wall): moe ~+31 over its 18 ms byte floor (miss exposure, shared GEMV unpriced) > attn +28.9 > hcn +4.8; remainder's non-tail 15.6 is no longer ranked apart — it overlaps the ranked spans (win − d2h = 14.8). M2's floor (~62 ms/token ≈ 16 tok/s ceiling) stands; buckets still not certified free (recorded stock-class wall spread ±1.5%, and M5's arm S sat +1.75% over the recorded 165.3). M6 (2026-08-08, staged, no GPU yet): the double split is INSTRUMENTED — attn cut into qkv/attend/oproj by two marks inside `attention` (shared endpoints, so they tile the whole-call span), moe into a seven-span host tiling (resid ≥ 0 by construction) plus shared/res/miss device pairs read behind its own closing sync, no new join anywhere; predictions registered with kill conditions (attn: qkv 17–24 / attend 2–6 / oproj 26–32; moe: sync2 26–36 with gpu res 17–24, miss 6–12, shared 6–10 — the closure hypothesis for moe's +31); one control run staged, wall gate ±3% of 130.7, reply md5 75b19fcde806059b45c515259feb16d2.
---

# Where do V4-Flash's 185 ms/token go?

## The measured budget (2026-08-07, benchmarks.md "GLM-5.2 vs DeepSeek-V4-Flash")

512 tokens in 95.0 s = **185.5 ms/token**, release, sole tenant, witnessed. From that run's
own log lines (43 layers, 256 experts top-6, expert slot 13,369,344 B fp4, resident
footprint 8.90 GiB, pool 91.08 GiB against a 137.06 GiB routed set = 66.5% residency, 17.0
misses/token at a 95.4% hit rate):

| term | bytes/token | at 193.8 GB/s achievable |
|---|---:|---:|
| routed experts, 43 x 6 x 13.37 MB | 3.45 GB | ~18 ms |
| shared expert (fp8, resident) | ~0.5 GB | ~3 ms |
| non-expert resident weights, read once | 8.90 GiB | ~46 ms |
| **floor** | **~13 GB** | **~65-70 ms = ~15 tok/s** |

NVMe side: 17 misses x 13.37 MB = **227 MB/token**, ~32 ms serial at 7 GB/s — overlappable,
overlap quality unmeasured. The engine runs at **2.8x its bandwidth floor**; ~120 ms/token
is unattributed. Every number above is arithmetic over one run's summary line — the
decomposition does not exist because **v4gpu has no phase buckets**.

*[CORRECTED 2026-08-07, by M2 below — three of the four rows are superseded: the 17.0
miss/token divides prefill+decode misses by decode tokens (decode-only is 4.96, 66
MB/token); the 8.90 GiB resident term includes the ~1.06 GB embed, a row gather not a
per-token read; and the "shared expert ~0.5 GB" row is both undersized (43 × 3 × 2048 ×
4096 fp8 ≈ 1.08 GB) and double-counted — shared weights are pin placements inside the
8.90 GiB. Floor re-derived ~62 ms/token ≈ 16 tok/s. The table is kept as the state the
investigation opened on.]*

## Candidate sinks, in guessed order (all UNMEASURED — that is the point)

1. **Unhidden fetch stalls** — up to ~32 ms/token if serial.
2. **fp4 MoE kernel rate** — GLM's MoE GEMV ran at ~half achievable bandwidth until HB 8→16
   (2.08x kernel, measured, roadmap #5). Whether `kernels/moe.hip`'s fp4 path carries that
   lesson is checkable by reading the kernel — do it host-side, before the GPU.
   *[CORRECTED 2026-08-07, by the M1b read below: HB 8→16 was `mla_latent_attend` — the
   attention kernel — not the MoE GEMV; roadmap #5 and benchmarks.md "The MLA HB sweep"
   both say so. The kernel-rate question itself stays live; the cited precedent was wrong.]*
3. **Dense/attention phase** — the 8.9 GiB re-read is the largest single term; `hc_mult 4`
   hyper-connections add elementwise volume on top.
4. **Launch/host overhead x 43 layers** — GLM's unbucketed remainder was ~25 ms/token.
5. **Speculation structurally off** (fp4 kernel instantiates R=1 only) — a multiplier for a
   later stretch, not a sink; GLM's gated MTP measures 1.108x.

## Milestones

- **M1 — the buckets (no GPU). DONE 2026-08-07, §"M1 — DONE" below.** Port GLM's per-phase telemetry (`route`/`moe`/`fetch`
  walls, miss count, ms/miss, plus the unbucketed remainder) into `src/v4gpu.rs`, matching
  GLM's bucket semantics so the two engines' PROFILE lines read the same way. Buckets must
  not add joins the decode does not already pay (GLM's precedent: HIP event pairs, clocks
  started after syncs the path already pays). GLM behavior byte-identical; V4 output
  byte-identical with buckets compiled in.
- **M1b — the kernel read (no GPU). DONE 2026-08-07, §"M1b — DONE" below.** Read `kernels/moe.hip`'s fp4 path against the HB-16
  lesson and `docs/measurement/how-to-measure.md`'s ISA-first rule; record whether the GLM
  fix applies, as prose in this doc.
- **M2 — one instrumented decode (GPU, via the coordinator). DONE 2026-08-07 — §"M2 — MEASURED" below.** Same command as the recorded
  512-token benchmark. Record the decomposition in `docs/measurement/benchmarks.md` and
  update this doc's verdict with the measured ranking.
- **M3 — attack in measured order.** Out of scope for this stretch; the M2 table decides it.

## M1 — DONE 2026-08-07: the buckets are in (host-verified; no GPU has run them yet)

`src/v4gpu.rs` (`V4Profile`, always-on like GLM's) accumulates two decode-thread buckets
and prints, once per `generate`:

```
PROFILE/tok: <w>ms wall | route <r>ms | moe <m>ms | fetch <f>ms | <n> miss, <k>ms/miss, <g> GB | remainder <x>ms
```

What each term brackets, exactly — named like GLM's and approximate like GLM's:

- **wall** — decode wall / generated tokens; same numbers as the `v4 decode:` line.
- **route** — the gate-logits D2H plus `route_row`'s host math. The D2H is the layer's
  FIRST blocking call and everything before it rides the null stream, so its wait drains
  the attention half, the hc/norm work and the gate GEMV still in flight: **most of `route`
  is attention GPU time**, exactly as in GLM, where the same D2H put the HB=16 attention
  win in the `route` column. Candidate sink 3 therefore lands HERE, not in the remainder.
- **moe** — `shared_expert` + `routed_experts`: the launches, the pool submit, both
  *existing* `device_sync`s (expert compute plus whatever fetch was not hidden behind it),
  and the accumulator drain launch. The analog of GLM's MoE block_on wall. Candidate
  sinks 1 and 2 land here.
- **fetch** — the reaper's off-thread wall (`RoutedPool::fetch_ns` delta), the same counter
  GLM's summary reads. It overlaps the decode wall and is NOT a share of it. miss,
  ms/miss and GB use the pool's **decode-only** miss delta — so miss/tok here can read
  slightly below the recorded 17.0, which spans prefill too.
- **remainder** — `wall − route − moe`, computed and printed rather than left to the
  reader: attention/hc/norm *launch* time (their GPU time drains into `route`), the
  end-of-layer sync (acc drain + `hc_post`), and the whole head tail — `hc_head`, final
  norm, the bf16 lm_head GEMV at `(1, 129280, 4096)`, argmax sync + D2H. Candidate sink 4
  lands here. Non-negative by construction (both buckets are sub-spans of the wall on one
  thread); `debug_assert`ed in dev.

Two precision notes for reading M2's numbers (review-surfaced, info-level): the fetch
delta books a fetch's whole wall at reap time, so boundary-token fetches straddling the
prefill/decode line land wholly on one side; and `-bench 512`'s wall covers 511 forwards
divided by 512 tokens (the prefill argmax supplies the first) — a ~0.2% understatement
applied identically to wall and every bucket, so the decomposition stays one arithmetic.

Cost bound, by argument as GLM's is: 7 `Instant` reads per MoE layer per token (~300
reads/token, O(10 µs) against a 185 ms token) and **no new sync, event, or join** — every
bracket closes at a blocking call the decode already pays. The real control is M2 itself:
its wall must sit within ~1% of the recorded 185.5 ms/token or the buckets are not free,
and that finding would precede any reading of the decomposition. GLM's decode and telemetry
are untouched; V4's generated token stream is byte-identical (clock reads and one log line).

## M1b — DONE 2026-08-07: the HB fix does not transfer, and the kernel-rate question narrows

The correction first (also noted at candidate 2 above): **HB 8→16 was `mla_latent_attend`**,
the MLA attention kernel — roadmap #5 and benchmarks.md "The MLA HB sweep" — not the MoE
GEMV. Its mechanism: HB is *heads per block*, and every head in a block shares one read of
the latent-KV tile, so doubling HB halves KV bytes fetched per unit of attention work — a
shared-operand-reuse lever.

Read against `kernels/moe.hip`'s fp4 path (`moe_gateup_f4_impl` / `moe_down_f4_impl` /
`common.hpp::dot_f4_wave_r`), that lever has **no purchase**:

- At the only instantiated `R = 1`, each expert weight byte is read exactly **once** per
  token, and weights are the traffic (13.37 MB/expert against a 16 KB activation that stays
  L2-hot). A one-read stream has no redundant read for a larger blocking factor to
  eliminate — there is no HB analog to raise.
- The reuse axis that *does* exist is `R` itself — token rows per weight read — and that is
  roadmap #10 (general-R MoE kernels) = candidate 5 here, structurally off: guard 1003
  refuses `nrow != 1`, and the S1b oracle is `bsz = 1` so an R=2 fp4 kernel could not be
  scored. A later stretch's multiplier, not this one's sink.
- What the source does show about load geometry: per lane-iteration the weight side is one
  4-byte dword (8 nibbles, one e8m0 scale decoded once for all 8 columns; 32 lanes = 128
  contiguous weight bytes per wave iteration), activations via two `float4` loads. Whether
  that reaches achievable bandwidth is exactly what `how-to-measure.md`'s ISA-first rule
  refuses to answer from source: any wider-load or occupancy claim is a **hypothesis** until
  the ISA is read. Un-ISA'd, unmeasured, and deliberately not acted on this stretch.
- Decision rule for M2: the `moe` bucket prices the question before anyone reads ISA. The
  routed bytes are 43 × 6 × 13.37 MB = 3.45 GB/token ≈ 18 ms at 193.8 GB/s; a `moe` wall
  far above the bytes (net of the fetch the same bucket exposes) makes the fp4 kernel rate
  the next read — if not, candidate 2 dies without an ISA read.

One geometry observation recorded while reading, as M2 candidate mechanism rather than
claim: `routed_experts` launches **one expert per call** (`e_count = 1`) — 3 kernel
launches + a ticket wait per expert, ~19 host launches per MoE layer for the routed path,
~820/token, issued serially by the host loop onto two streams.

## M2 — MEASURED 2026-08-07: the decomposition, the ranking, and the kill-condition check

Full record in `docs/measurement/benchmarks.md` "V4 decode decomposition"; this section
carries the reading. Run as staged below, witnessed clean, output **byte-identical** to the
recorded head-to-head.

**The gate missed first: wall 190.9 ms/token vs the recorded 185.5, +2.9% against a ±1%
control** — variance suspected (the bucket cost bound is ~500x smaller than the delta, see
the record), but at n=1 the buckets are *not certified free*, and that caveat rides on
every number here.

```
PROFILE/tok: 190.9ms wall | route 78.9ms | moe 71.9ms | fetch 11.9ms | 4.96 miss, 2.40ms/miss, 0.07 GB | remainder 40.0ms
```

- **Candidate 1 (unhidden fetch) is DEAD.** Decode-only fetch is 4.96 miss/token = 66
  MB/token, 11.9 ms/token *off-thread*; the 17.0 miss/token that priced the "~32 ms if
  serial" fear was prefill-polluted.
- **Candidate 3 (dense/attention) is measured at route ≈ 78.9 ms — ~46 ms above its ~33 ms
  of bytes.** The phase runs at ~2.4x its traffic; *which* op eats it needs the phase's own
  split (or a kernel microbench) before a lever is named.
- **Candidate 2 (fp4 kernel rate) is now WARRANTED an ISA read** by M1b's own decision
  rule: moe ≈ 71.9 ms against ~23 ms of routed+shared bytes, ~49 ms excess that fetch
  overlap cannot explain (all of fetch is 11.9). What replaces the dead HB premise: read
  `dot_f4_wave_r`/`moe_gateup_f4`'s ISA for load width and wave occupancy, and price the
  one-expert-per-launch geometry (M1b's geometry observation above — the kernel already
  takes an `e_count`).
- **Candidate 4 (launch/host overhead) is the remainder: 40.0 ms/token**, holding the
  ~820/token routed-path launches, 43 end-of-layer syncs, and the whole head tail (incl. the
  `(1, 129280, 4096)` bf16 head GEMV the port flagged as a shape objection). Cheapest next
  instrument: a tail bracket at the existing argmax join.

**Kill-condition check — the floor survives, corrected ~10% down.** Three errors found in
the opening table, none fatal: the 8.90 GiB resident term includes the ~1.06 GB embed
matrix (a row gather, not a per-token read); the "shared expert ~0.5 GB" row was undersized
(≈1.08 GB) AND double-counted — shared weights sit inside the 8.90 GiB resident footprint,
not beside it; and the NVMe term used the prefill-polluted miss rate. Re-derived: routed
3.45 GB + per-token resident ~8.5 GB ≈ 11.95 GB ≈ **~62 ms/token ≈ 16 tok/s ceiling**. 10 tok/s (≤100 ms/token) needs ~91 of the ~129 ms/token
now measured above bytes — still no quality tradeoff required, and the three excesses
above are the ranked budget for it.

## M3a — the fp4 ISA read (2026-08-07, no GPU), and the registered prediction

`hipcc --offload-arch=gfx1151 -O3 -fPIC -S kernels/moe.hip` — the build's own flags
(`build.rs:76`), read before any code was written, per `how-to-measure.md`'s ISA-first rule.
Both fp4 kernels compile to 0 LDS, 0 scratch, 48 VGPR (`moe_gateup_f4`) / 38 VGPR
(`moe_down_f4`) — 16 waves/SIMD, so **occupancy is not the limiter**. The story is the
inner loop. One iteration of `dot_f4_wave_r`'s dword fast path — one wave consuming 256
columns = 128 packed weight bytes + 8 scale bytes — is **195 instructions, 117 of them
VALU, of which ~10 are the FMAs the loop exists to do**:

- **The e2m1 decode is eight exec-mask branch regions per dword.** The `e ? (1+m/2)·2^(e−1)
  : m/2` ternary compiles to `v_cmpx_ne_u32` + `s_mov exec` / `s_or exec` around a 5-VALU
  else-arm, PER NIBBLE — ~11 instructions each, ~88 of the 195. The `(e−1)&3` trick kept
  the shift in range but did not make the select branchless.
- **The sign pass is 8 more `v_and`+`v_cmp`+`v_cndmask` triplets**, and the e8m0 scale
  decode is a further exec-branchy region (the 0xff/0 special cases) per iteration.
- **The weight dword and the scale byte lower to `flat_load_b32` / `flat_load_d16_u8`** —
  generic-address-space loads, because both pointers come out of the `ExpertDescF4` struct
  read from memory and clang cannot prove them global. On gfx11 flat loads take the slower
  path and count against `lgkmcnt` too; the loop carries three `s_waitcnt`s, one mid-body.
  The activation side is clean (`s_clause`'d `global_load_b128` ×2).

Balance arithmetic (~2.9 GHz, 80 SIMDs, 1 wave-instr/SIMD/cycle): the loop delivers 128 B
per 195 issue slots ≈ **0.66 B/cycle/SIMD, against the 0.84 needed to stream 193.8 GB/s**
— instruction-issue-bound at ~79% of achievable bandwidth in the IDEAL case (VOPD dual-issue
buys some back; the flat-load latency and mid-loop waits take more). So the ISA puts the
routed compute floor at ≥ ~23 ms/token against the 18 ms of bytes — **kernel rate explains
~5–10 ms of the 49 ms excess, not most of it.** The fixes it names — branchless e2m1,
global-address-space descriptor pointers — are kernel changes, out of this stretch's scope
(plumbing only), recorded as the M3 kernel-rate lever.

**Launch geometry, what the source + ISA say together:** `routed_experts` launches one
expert per `rivoli_moe_expert_range_f4` call — 3 kernels each (gate/up, a 16-block
`act_quant_f8`, down), ~19 launches/MoE-layer on the routed path, ~820/token — although the
kernel's `[e_start, e_count)` range form was written as the hook for exactly this batching
(`kernels/moe.hip:173-176`). Batching the RESIDENT experts of a layer into one range call
removes ~630 launches/token (device dispatch floor 1.97 µs measured on GLM), ~5 of 6
`act_quant` micro-launches per layer, and the drain-to-16-block-kernel pipeline bubbles at
each of the ~12 removed chain boundaries per layer. Misses must stay per-expert launches on
the miss stream — batching them would gate the whole batch on the LAST fetch (2.40 ms/miss
measured), serialising hits behind misses.

**Prediction, registered before the measurement so it can be wrong:** resident-batching
moves the `moe` bucket by **−3 to −8 ms/token (point: −5)**, byte-identical output (integer
fixed-point accumulation is order-free by design — `MOE_ACC_SHIFT`'s contract,
`kernels/common.hpp:14-37`), fetch bucket unchanged. The remaining ~40 ms of the 49 is
predicted to split: instruction-issue-bound decode ~5–10 ms, miss fetch exposed inside the
layer's closing `device_sync` ~8–10 ms (4.96 misses × 2.40 ms, less the ~0.5 ms/layer of
resident compute that can hide it), shared-expert fp8 GEMV rate + per-layer H2D/sync fixed
costs the rest — i.e. **launch geometry is predicted the MINOR half of the moe excess**,
and if the measured delta is under ~3 ms/token it is inside the recorded ±2.9% wall
variance's bucket-level noise and the arms need replicates before any claim.

> **SCORED 2026-08-07, by the M3b A/B: wrong on the good side — measured Δmoe is −15.9
> ms/token against the −3..−8 band.** Direction, byte-identity and a non-growing fetch (it
> fell) were as predicted; the magnitude was priced on the wrong constant — the 1.97 µs
> device dispatch floor, where the recovery divides out to the ~25 µs host-bubble class.
> That makes the geometry twice the top of its band and the largest single priced
> component, though still smaller than the ~32 ms residual. The full accounting, and why
> the residual stays named-not-priced, is in benchmarks.md "V4 launch-geometry A/B".

## M3b — the geometry change, MEASURED 2026-08-07: moe 70.6 → 54.7, wall 186.2 → 169.9 = 5.887 tok/s

Full record and every gate result in `docs/measurement/benchmarks.md` "V4 launch-geometry
A/B" — all gates passed. Δmoe = **−15.9 ms/token**, all of it in the wall (+9.6% tok/s);
`tail` measured 8.2 of the 39.1 ms remainder, ranking item (3)'s first number. The
prediction above is SCORED at its own paragraph. New measured state: **169.9 ms/token;
10 tok/s needs ~70 more off the wall** — the ranked residue is in the verdict.

*The staging protocol below is kept as run:*

Landed on `wt/v4-moe-launch`: `routed_experts` writes descriptors and routing weights in
LAUNCH order (residents compacted at `[0, n_res)`, misses after), launches the residents as
ONE `[0, n_res)` range call on the compute stream, and keeps every miss a separate launch
on the miss stream behind its own ticket — the straggler story M3a's prediction depends on.
Plus the tail bracket at the argmax join (remainder's head share, printed inside the
remainder term) and the raw decode-miss integer M2 found itself unable to recover.

**Two release arms, both binaries built BEFORE the device was requested, nothing built
between arms** *(staged 2026-08-07 pre-GO, run the same day as written)*:

- arm S (stock): `c656eac` via `git archive` into a scratch tree, `cargo build --release
  --features rocm` there.
- arm B (batched): this branch's committed HEAD, same build line.
- command per arm, flag-identical to M2's (the 218-token prompt verbatim in benchmarks.md's
  head-to-head CORRECTED note), each under the exclusive flock with the per-minute
  KFD+GTT+llama-swap witness, stock first:
  `flock /var/run/sys-gpu.lock -c '<arm-binary> /var/db/rivoli/v4-f4-full -bench 512 --prompt "<prompt>"'`

**Gates, registered with the prediction:** (1) correctness — generated text byte-identical
across arms, and hit/miss counts identical (routing and submit order are untouched); any
token difference is a BUG in the regrouping, not noise. (2) the instrument is the `moe`
bucket, NOT wall — the recorded +2.9% run-to-run wall variance swallows a 5 ms win; if
|Δmoe| < ~3 ms/token the arms need replicates before any claim. (3) `fetch` must not grow —
growth is the tell that batching serialised hits behind misses. (4) `tail` from arm B is
M2 ranking item (3)'s first number, free from the same run.

## M4 — the route split, MEASURED 2026-08-08: hcn 41.2 ms is the surprise

Run as staged the same day (record and all four gates in benchmarks.md "V4 route split"):

```
ROUTE-SPLIT/tok: attn 53.2ms | cmp 9.1ms | hcn 41.2ms | gate 3.2ms | win 107.7ms (resid 1.0ms) | d2h 76.3ms + host 0.4ms
```

wall 172.9 (+1.8%, in band), counters identical to M3b, resid ≥ 0, spans+resid ≡ win
exact. The prediction below is SCORED at its own paragraph; the one-line outcome: three of
four bands hit, and the headline was wrong — the **hyper-connection/norm chain measured
41.2 ms against a 4–8 band and 0.8 ms of bytes**, the engine's largest above-bytes excess.
The ranked levers are in the verdict; the named next step is a host-side read of the
hc/norm kernels (`kernels/linalg.hip` — the 20-pass Sinkhorn in `hc_pre` and GLM's
`dim3(1)`-rmsnorm precedent are the candidates, per benchmarks.md's named-not-priced
list). *[Done — M5 below, 2026-08-08: the read attributed it, the fix removed 35.1 of
the 41.2, and the verdict now carries M5's residue ranking; M4's own ranking survives in
benchmarks.md "V4 route split".]*

*The staging record below is kept as written pre-GO:*

`route` is the largest sink — ~76 ms against ~31 ms of bytes — and M2's reading said its
excess "needs the phase's own split before a lever is named". This stretch lands the split:
six HIP-event marks per layer on the null stream (`src/v4gpu.rs`, `V4Profile` "The route
split"), four spans read at the gate-logits D2H — the join the route bracket already closes
on — so there is **no new sync, event wait, or join**. The spans are SPANS in GLM's
`idx_gpu_ns` sense: gaps included, not kernel sums (GLM measured its span 27% above one).

**What each bracket covers, exactly (marks are program order on one stream):**

- `hcn` — marks 0→1 + 3→4: `hc_pre` + `attn_norm`, and `hc_post` + `hc_pre` + `ffn_norm` —
  the hyper-connection application and both sublayer norm chains (two spans, one bucket:
  one lever class).
- `cmp` — marks 1→2: `compress_and_place` (the compressor deposit and both blocking
  placement copies) plus the GPU idle while the host builds and uploads the positional
  selection. The marks record on EVERY layer — the two ratio-0 layers just read ~zero-width
  — so the accumulation is uniform across layer classes rather than conditional, which is
  how the fires-on-some-layers failure mode is excluded by construction. There is no
  learned-indexer kernel on this path (positional selection, valid to 2052 tokens; this
  benchmark peaks at position ~730), so `cmp` IS the whole per-layer selection cost.
- `attn` — marks 2→3: the `attn::v4::attention` call, whole — q/kv projections, cache
  write, `sparse_attn`, o_proj. **The intra-attention split (qkv vs attend vs o_proj) stays
  unbucketed this stretch and this line says so:** it needs marks inside one straight-line
  function in `src/attn.rs`, outside this stretch's file ownership — no new join, just
  ownership; three more marks there are the follow-up if `attn` dominates.
- `gate` — marks 4→5: the gate GEMV, the `xq` copy and its `act_quant` — the last launches
  before the D2H.
- `win` — the HOST wall containing the four spans (they tile marks 0→5 inside it): layer
  top to the D2H's return. Printed with `resid = win − (hcn+cmp+attn+gate)`, which is
  **the summing check by construction, not by hope**: the window contains every pair
  (mark 0 records after the window clock starts; mark 5 retires before the D2H's data
  lands), so `resid ≥ 0` up to GPU-vs-host clock-rate skew — a dev-profile assert — and
  holds the D2H copy plus the pre-mark-0 lag (layer 0's includes the step's embed gather).
  The residual is defined against `win` and NOT against `route`, deliberately: `route` =
  D2H wait + `route_row` host math, `win` = pre-D2H host traversal + D2H wait, so the event
  sum can legitimately exceed `route` — GPU time that overlapped host traversal is
  invisible to `route`'s clock but visible to the events. The two are reconciled on the
  same line: `d2h + host` restates `route` as its halves (`route_host_ns` splits it at the
  clock reads the bracket already pays), so `win − d2h` = the traversal share the PROFILE
  remainder holds.

One geometry observation recorded while reading, M1b-style (mechanism candidate, not a
claim): the gate-logits D2H copies the WHOLE `gate_logits` buffer every layer —
`copy_out_into` has no row count, and the buffer is sized `max_m × n_experts` f32 = 218 ×
256 × 4 ≈ 223 KB at this prompt — where decode reads one row (1 KB). ~9.6 MB/token of
blocking D2H, lands in `resid`/`d2h`; if `resid` measures large, this is the first suspect
and a one-argument fix (`copy_out_prefix` already exists).

No host-side test is added for the accumulation and that is a decision, not an omission:
the pairing is five fixed index pairs feeding four adds, HIP events cannot exist without a
device, and a test restating the constants would be the tautological-assertion class this
repo has shipped twice. The printed residual is the check, on every run.

**Bytes budget per sub-span** — host arithmetic over `resident.safetensors`'s own tensor
shapes (read from the artifact header 2026-08-08), decode m=1, per token = 43 layers:

| span | read per layer | per token | at 193.8 GB/s |
|---|---|---:|---:|
| hcn | `hc_attn_fn`+`hc_ffn_fn` [24,16384] f32 = 3.15 MB, norms 0.03, ~0.5 MB h/xw activations | 0.158 GB | 0.8 ms |
| cmp | compressor `wkv`+`wgate` f32: [1024,4096]×2 = 33.6 MB on the 21 ratio-4 layers, [512,4096]×2 = 16.8 MB on the 20 ratio-128, 0 on the 2 ratio-0 | 1.040 GB | 5.4 ms |
| attn | fp8 `wq_a` 4.19 + `wq_b` 33.55 + `wkv` 2.10 + `wo_a` 33.55 + `wo_b` 33.55 + scales 0.03 = 107.0 MB, + ≤1 MB selected KV + activations | 4.640 GB | 23.9 ms |
| gate | `ffn.gate.weight` [256,4096] f32 = 4.19 MB + 0.08 activation | 0.184 GB | 0.9 ms |
| **window** | | **~6.02 GB** | **~31.1 ms** |

(M2's aggregate priced route at "~6.4 GB ≈ 33 ms"; the per-span sum re-derives ~6.0 GB —
the drift is M2's lump including rows now booked to other buckets. The indexer's 17.8
MB/ratio-4-layer of weights is resident but never read: the engine selects positionally.)

**Prediction, registered before the measurement so it can be wrong:** `attn` dominates the
window — **45–62 ms of a predicted win ≈ 76–86 ms (point: attn 55, cmp 10, hcn 6, gate 5,
resid 4, win 80)** — i.e. attention runs ~2.3× its 23.9 ms of bytes and carries most of
route's ~46 ms excess, with the fp8 GEMV projections (103 of the 107 MB/layer) the suspected
mechanism in the M3a family (issue-bound decode, unverified for fp8) alongside
`sparse_attn`'s gathered reads. `cmp` second at 8–12 ms (5.4 of bytes + two blocking
placement copies + the selection stall), `hcn` 4–8 and `gate` 3–6 (both far above their
<1 ms of bytes — the ~25 µs host-bubble launch class M3b measured, not traffic). **What each
outcome kills:** `attn` under half the window kills the fp8-kernel-rate lever (and the ISA
read it would warrant) as the primary attack and hands the ranking to the fixed-overhead
class; `cmp` landing at its bytes kills the compressor-overpricing worry; `hcn+gate` above
~15 ms makes the launch class, not any kernel, the next lever.

> **SCORED 2026-08-08, by the run above: three of four bands hit, wrong at the headline.**
> attn 53.2 (45–62, near the 55 point ✓), cmp 9.1 (8–12 ✓), gate 3.2 (3–6 ✓) — and **hcn
> 41.2 against the 4–8 band, 5–10× out and ~50× its bytes**, which pushed win to 107.7
> above its 76–86 band; resid 1.0 under its 4 point. Both registered kill conditions
> fired, one at its edge — the resolutions and the ranked levers are in benchmarks.md and
> the verdict. The error family is M3a/M3b's again —
> mechanism right (attention is far above bytes), magnitude priced on the wrong component:
> the band priced hcn as small elementwise kernels plus launch bubbles, and the measured
> 41.2 says the kernels themselves are ~7× the bubble arithmetic (~5.4 ms at the ~25 µs
> class). Full accounting in benchmarks.md "V4 route split".

**Staged command (GO, one run):** release binary built at this branch's committed HEAD
BEFORE the device is requested; flag-identical to M2/M3b, the 218-token prompt verbatim
from benchmarks.md's head-to-head CORRECTED note; exclusive flock with the per-minute
KFD+GTT+llama-swap witness:

```
flock /var/run/sys-gpu.lock -c 'target/release/rivoli /var/db/rivoli/v4-f4-full -bench 512 --prompt "<prompt>"'
```

**Gates, registered with the prediction:** (1) wall within ±3% of the recorded 169.9
ms/token (this branch bases on the batched-geometry main; the instrument's bound is O(1.5
ms/token) — 258 event enqueues + 215 completed-event queries, argued at `V4Profile`) — a
breach is reported BEFORE any reading of the split. (2) output byte-identical to the M3b
batched arm (2025 bytes) and expert lookups identical (179389 hit / 8693 miss): clock reads
and event records touch no data. (3) `resid ≥ 0` and the spans + resid sum to `win` exactly
(printed identity). (4) coherence, expected not enforced: `win − d2h` (the traversal) should
land near the non-tail remainder share M2 attributed to per-layer launches; a traversal far
above `remainder − tail` says the window books time the PROFILE line books elsewhere.
## M3c — the kernel-rate levers, IMPLEMENTED 2026-08-08 (no GPU yet), and the registered prediction

M3a named two kernel fixes and priced their bucket at ~5–10 ms of the moe excess. Both are
now implemented, on `wt/v4-e2m1`, and read back out of the compiler per
`how-to-measure.md`'s ISA-first rule — same command as M3a's read
(`hipcc --offload-arch=gfx1151 -O3 -fPIC -S`, `build.rs:76`'s own flags) — BEFORE any
device time was requested.

**Lever 1, branchless e2m1/e8m0 decode** (`kernels/common.hpp::e2m1f`/`e8m0f`). The
subnormal-aware ternary became a register-immediate table: the 8 magnitudes doubled are the
integers {0,1,2,3,4,6,8,12}, one nibble each, so the whole table is the immediate
`0xC8643210` and a decode is bfe → shift → mask → `v_cvt_f32` → `·0.5f`, with the sign
OR'd into the payload bits (code 8 still decodes to `-0.0f`, bitwise). The e8m0 special
cases became a `umax` + one select. Both are bit-exact by exhaustive host sweep, not by
argument — `tests/v4_kernel.rs::the_branchless_decodes_match_the_oracle_bitwise` (all 16
e2m1 codes, all 256 e8m0 bytes, against the oracle at the bit level) and
`every_byte_pattern_decodes_right_in_both_dot_paths` (all 256 packed-byte patterns at every
dword byte position through the REAL kernels, fast path and scalar tail, GPU-gated).

**Lever 2, global-address-space descriptor loads.** Two source-level idioms do NOT work
and are recorded in `common.hpp`'s `gu8p` comment so nobody re-tries them: a round-trip
`(T*)(AS1 T*)p` cast (instcombine folds the cancelling pair) and
`__builtin_assume(!is_shared && !is_private)` (never reaches InferAddressSpaces; loads
stayed flat in the emitted ISA). What works is typing the span AS1 at the front end:
`dot_f4_wave_r`'s `row`/`scalerow` are now `gu8p` (`__attribute__((address_space(1)))`),
cast once at the descriptor in `moe.hip::as_global`.

**The ISA after, against M3a's 195.** `moe_gateup_f4`'s inner dword iteration — one wave
consuming 128 packed weight bytes + scale — is **105 instructions, was 195**. The eight
per-nibble exec-mask branch regions, the eight sign cmp+cndmask triplets and the e8m0
exec region are GONE (the loop body's only remaining selects are the two branchless e8m0
cndmasks; the only branch is the backedge); every load in every loop is `global_load_*`
(zero `flat_*` in either fp4 kernel); `s_waitcnt` 4 → 2 with none mid-decode;
`s_delay_alu` 40 → 17. VGPR 48 → 49 / 38 → 39, which stays inside the same allocation
granule — occupancy still 16 waves/SIMD. Balance arithmetic, same model as M3a: 128 B per
105 slots ≈ **1.22 B/cycle/SIMD against the 0.84 needed to stream 193.8 GB/s** — the loop
goes from issue-bound at 79% of achievable to ~45% issue headroom, so the routed-compute
floor drops from ~23 ms/token (issue) to the **18 ms/token byte floor**. At the real dims
(4096/2048, both multiples of WAVE·8 = 256) the dword path is the entire loop for all
three projections.

A review pass compiled the next rung and priced it dead: a `v_perm_b32` byte-table decode
(selector `w & 0x07070707`, doubled magnitudes as packed bytes, `v_cvt_f32_ubyte0..3` in
place) emits **88** for the same loop, bit-identical — the compiler already fuses the
staged form's shift+mask into one `v_bfe` against the table immediate and the sign into
one `v_and_or_b32`, ~7 VALU/nibble — but 88 only widens headroom past a bound that
stopped binding at 105, so it is recorded here as the known next lever IF the 18 ms byte
floor ever moves (smaller format, faster memory), not taken. A `__constant__`-array or
LDS table is NOT that lever: the first re-introduces the in-order-vmcnt mid-loop waits
this change removed, the second trades them for `lgkmcnt` plus a barrier the kernel's
early-return structure cannot host.

**Blast radius, measured not assumed.** Stock-vs-new ISA diff over every kernel file:
all ten GLM kernels in `moe.hip` (`*_vq*`, `*_i4*`, `moe_acc_drain`, `moe_gate_v4`)
**bit-identical**; every other kernel file identical except `v4indexer.hip` —
`v4_indexer_spread` inlines `e2m1f` via `fp4_quant_roundtrip`, so its SCHEDULE changes
while its values cannot (the decode is bit-exact); worth knowing when reading non-moe
buckets in the A/B.

**Prediction, registered before the measurement so it can be wrong:** the A/B below moves
the `moe` bucket by **−4 to −10 ms/token (point: −6)** from the measured 54.7, output
byte-identical and hit/miss identical (the change touches decode arithmetic that is
bit-exact and load ADDRESS SPACE, not order, routing or residency), `fetch` unchanged.
The −5 ms core is M3a's issue-bound excess (23 → 18); the spread above it is the
flat-load latency/`lgkmcnt` serialisation the slot model never priced (M3b's lesson:
the recovered constant can be the bigger one); the floor of the band is where the miss
stream and the closing `device_sync` hide most of what the kernel no longer spends. If
|Δmoe| < 3 ms/token, that is inside recorded bucket noise — replicates before any claim,
and a null result gets recorded WITH the 105-count that proves the instructions were
really removed: that dissociation would itself be the finding (the bound was elsewhere).

> **SCORED 2026-08-08, by the A/B below: measured Δmoe = −5.3 ms/token — INSIDE the
> −4..−10 band, 0.7 off the −6 point.** First M3-series prediction to land in its band.
> Byte-identity and hit/miss identity as predicted; fetch BYTES identical, fetch time
> +0.6 ms — inside the recorded 9.6–11.9 replicate spread. Wall 170.3 → 165.3 =
> 6.048 tok/s (+3.0%). Full record and every gate result in benchmarks.md "V4 fp4
> kernel-rate A/B". The −5.3 sits where the −5 issue-bound core predicted, which prices
> THIS lever's slot model right; the point alone cannot apportion the two unpriced terms
> it sits between (flat-load latency above, sync-hidden compute below), so neither is
> claimed. The residual ~31.6 ms of moe above the byte floor is the M3a split (miss
> exposure, shared-expert GEMV, per-layer fixed costs), still named-not-priced.

*Host validation, all green 2026-08-08: dev-profile `cargo test --features rocm --no-run`
(jscpd gate re-armed via `touch build.rs` after rustfmt — it has manufactured clones here
before), the host-only bitwise sweep run and passing, `clippy --release --features rocm
--all-targets` and the six-feature union both clean, file rustfmt-clean. The device half
(`v4_kernel` suite incl. the byte-pattern sweep, then the A/B) WAITS FOR GO under the
resource protocol.*

**The A/B, staged (not run):** M3b's own pattern, verbatim — arm S = `a2a0b8c` via
`git archive` into a scratch tree, arm B = this branch's committed HEAD, both release
binaries built BEFORE the device is requested, nothing built between arms, per-arm
exclusive flock with the per-minute KFD+GTT+llama-swap witness, the M2 command
flag-identical (218-token prompt verbatim in benchmarks.md's head-to-head CORRECTED note).
Gates as M3b's: (1) byte-identical text AND identical hit/miss = correctness, any
difference is a BUG in the decode or the cast, not noise; (2) the instrument is the `moe`
bucket, not wall; (3) `fetch` must not grow; (4) the record lands in benchmarks.md when it
exists, beside the "V4 launch-geometry A/B".

## M5 — the hc/norm kernel read (2026-08-08, no GPU), and the registered prediction

M4 measured `hcn` at 41.2 ms/token against 0.8 ms of bytes and bounded launch bubbles at
~5.4, naming the kernels themselves the prime suspect. This is the host-side read —
source plus ISA (`hipcc --offload-arch=gfx1151 -O3 -fPIC -S kernels/linalg.hip`,
`build.rs:76`'s own flags, per `how-to-measure.md`'s ISA-first rule) — written BEFORE any
device time.

**What the bucket holds, counted from the code.** Per layer the two hcn spans cover five
launches: `hc_pre` ×2, `v4_rmsnorm` ×2 (`mla.hip`, `dim3(rows)` — one block at decode),
`hc_post` ×1 (the FFN-side `hc_post` drains at the end-of-layer sync, outside the window).
43 layers → 215 launches/token, 86 of them `hc_pre`. There is no 20-launch Sinkhorn: the
20 passes are a `for` loop INSIDE `hc_pre`, on ONE THREAD (`linalg.hip::hc_sinkhorn`,
called at `if (t == 0)`) — the 4×4 matrix never leaves that thread, so the inter-pass
synchronization the algorithm needs is program order on a single lane, not a device-wide
barrier. That kills the fused-20-launches lever the plan named (there is nothing to fuse)
and makes the Sinkhorn a serial-compute term, not a launch term.

**Where the time is, attributed by arithmetic that closes.** `hc_pre` launches `dim3(s)` ×
256 threads — at decode, ONE 8-wave workgroup for the whole device. Its mixes phase (the
`[24, 16384]` `hc_*_fn` GEMV) streams 1.57 MB of weights plus 24 re-reads of the 64 KB
flattened residual (~3.1 MB of loads) through that one workgroup, and the ISA shows the
inner loop issues exactly TWO scalar `global_load_b32` then `s_waitcnt vmcnt(0)` EVERY
iteration — no unroll, no pipelining, so the whole workgroup holds ~2 KB in flight against
LPDDR5/GTT latency. Cross-calibrating with GLM's measured `dim3(1)` rmsnorm (7.7 µs for
~96 KB touched ≈ 12 GB/s for one such workgroup, benchmarks.md "Closed questions"): ~3.1
MB at single-digit-GB/s effective ≈ **~350–400 µs per `hc_pre` launch**. The books then
close: 86 × ~400 µs ≈ 34.5 ms + rmsnorms ~0.7 (GLM's 7.7 µs precedent) + `hc_post` ~0.3
(64 blocks, 320 KB) + bubbles ~5.4 ≈ **41 ms against the measured 41.2**. Within one
`hc_pre`: mixes ~340–370 µs, sum-of-squares ~15–25 (same one-load-per-wait pattern over
64 KB), the 20-pass serial Sinkhorn ~10–30 (≈640 f32 divides + 24 `expf` on one lane),
combine ~5–10.

**The fix, and its byte-identity argument.** Two schedule-only levers, no arithmetic
change:

1. **Widen the block to 1024 threads** (launcher `dim3(256)` → `dim3(1024)`). The mixes
   loop `for (j = w; j < HC_MIX; j += nw)` then gives each of 24 waves ONE row instead of
   8 waves × 3 serial rows — 3× the loads in flight. The per-row unit is untouched: the
   same lane-strided comb (`i = lane; i += WAVE`) and the same `wave_sum` ladder, executed
   by one wave, so every `mixes[j]` is bit-identical regardless of WHICH wave computes it.
   The sum-of-squares phase is the one part whose arithmetic depends on block shape (a
   `blockDim`-strided comb into a `blockDim`-leaf LDS tree), so it is FROZEN at the
   current geometry: first 256 threads, stride 256, 256-leaf tree — bit-identical by
   construction, waves 8–31 idle at the barrier. The combine phase's per-element c-order
   is fixed; only the thread→element map widens. Sinkhorn stays `t == 0`, untouched.
2. **`#pragma unroll` the two strided load loops** (mixes, sum-of-squares). Unrolling a
   serial `acc += a·b` chain preserves its order — the FMAs stay one dependent sequence,
   the compiler merely hoists the now-independent loads above the chain — so the bits
   cannot move; the gain is loads-in-flight per wave. Verified against the emitted ISA
   before commit: same FMA sequence, loads clustered, no per-iteration `vmcnt(0)` in the
   main bodies (the ≤7-iteration peel/remainder loops keep the scalar pattern — at the
   checkpoint's `hcd = 16384` both trip counts divide by 8 and they never execute).

No new launch, no new sync, no cross-block communication: one token's whole tensor is
owned by one block (grid = s), which is why intra-block `__syncthreads()` was and remains
the only synchronization the kernel needs — grid-sync/cooperative-launch territory never
opens. `v4_rmsnorm` is left alone on GLM's measured precedent (7.7 µs; "do not re-flag it
from the launch shape alone"). Reducing Sinkhorn passes is quality, not optimization
(real-weights goldens: 19 vs 20 moves 39,893/53,248 of `ffn_norm_out`), and nothing here
touches the iteration count.

**Prediction, registered before the measurement so it can be wrong:** the A/B moves the
`hcn` sub-bucket by **−15 to −30 ms/token (point: −22)**, i.e. `hc_pre` from ~400 µs to
~50–225 µs/launch (point ~145), and the wall by ≈ the same amount (the window is serial
on the null stream; nothing overlaps it). Output byte-identical, hit/miss and raw-miss
counters identical (clock-free, residency-free, arithmetic-order-free change).
`attn`/`cmp`/`gate` within ±1.5 ms, `moe`/`fetch`/`tail` unchanged within recorded
spread. Kill conditions: |Δhcn| < 3 ms ⇒ the latency model is wrong, replicates before
any claim and the finding is that the time hides where stream events cannot see it; Δhcn
between −5 and −14 ⇒ the single-workgroup ceiling is real but lower than modelled — next
rung is splitting the mixes across workgroups (a 2-kernel form, +1 launch), not more
waves in one. At the point the bucket lands at ~19 ms; the modelled FLOOR under this
kernel shape is ~10–15 (5.4 of bubbles + ~1 of rmsnorm/`hc_post` + 86 launches whose
1.57 MB and serial Sinkhorn cannot go below ~50–100 µs in one workgroup), so a −22
result still leaves a priced next rung and −30 is the model's hard edge. The record
lands in benchmarks.md beside "V4 route split"; a passing byte-identity gate also keeps
`tests/v4_loop.rs` §8's envelope valid without a re-run (its own note names `hc_pre`
changes as the trigger) — a failing one owes §8 a re-run on top of being a bug.

> **SCORED 2026-08-08, by the A/B: measured Δhcn = −35.1 ms/token — outside the −15..−30
> band on the GOOD side, the M3-series error family a third time (the good side a
> second; M4 missed high).** Byte-identity, identical counters, flat tail and
> attn/cmp/gate ≤0.1 all as predicted; wall −37.5 (Δhcn plus −2.4 of unbanked moe/fetch
> movement). What the band missed: it priced the width lever (3×) and treated the unroll
> as garnish, but the levers MULTIPLY (24× loads in flight), so `hc_pre` fell to the
> registered floor's own edge instead of the ~145 µs point. New state 130.7 ms/token =
> 7.652 tok/s; hcn CLOSED at this rung. Full record, every gate, the wall-win
> attribution and the retired bubble bound: benchmarks.md "V4 hcn A/B".

## M6 — the double split (attn × moe), INSTRUMENTED 2026-08-08 (no GPU yet), and the registered predictions

The residue ranking after M5 names two spans with no internal decomposition — moe 48.9
over ~23.6 of bytes and attn 52.8 over 23.9 — and the instrument-first pattern has now
paid three times (M2 → M3b, M4 → M5). This stretch lands BOTH splits, instrumentation
only, no perf change: three sub-spans inside `attn` and a host tiling plus three device
attributions inside `moe`. No new sync, event wait, or join anywhere — every event pair
is read behind a join the decode already pays (the gate-logits D2H for the attn marks,
the routed path's own closing `device_sync` for the moe pairs), and every host bracket
is an `Instant` read beside a call the path already makes.

**The attn split — two marks recorded INSIDE `attn::v4::attention`** (via its new
`SplitMarks` argument; `None` from every probe/test caller is bit-identical to the
pre-M6 call), cutting M4's 2→5 span into three that SHARE endpoints, so they tile it by
construction and the printed `resid` is an identity check (either sign, per-query float
rounding only; dev assert at |resid| ≤ 5e-2). What each bracket covers, exactly:

- `qkv` — mark 2 (selection uploaded) → `qkv_done` (recorded after the KV entry's
  partial `act_quant`): the q and kv chains — `xq` copy + `act_quant`, `wq_a` GEMV,
  `q_norm`, `qrq` copy + `act_quant`, `wq_b` GEMV, `qk_norm`, RoPE(q), `wkv` GEMV,
  `kv_norm`, RoPE(kv), KV `act_quant` — 12 device ops, 3 of them fp8 GEMVs.
- `attend` — `qkv_done` → `attend_done` (recorded after `launch_v4_sparse_attn`): the
  phase-dependent cache/ring write copies (one row at decode) and `sparse_attn` itself.
- `oproj` — `attend_done` → mark 5 (`attention` returned): the de-rotating RoPE,
  `wo_a` GEMV, `act_quant`, `wo_b` GEMV.

**The moe split — seven host spans that tile `moe_ns`** (disjoint sub-intervals of the
two brackets `moe` already sums, so `resid ≥ 0` by construction, dev-asserted), **plus
three same-stream device pairs** read behind the closing `device_sync` (attributions,
NOT addends — `shared` overlaps `route_row`'s host math; `res` and `miss` overlap each
other across streams). Coverage, exactly:

- `sh_enq` — the `shared_expert` call: five launch ENQUEUES (its GPU time is `shared`).
- `desc` — routed entry → first H2D: `RoutedPool::submit` (miss fetches enter flight
  here), the format check, the 256-descriptor launch-order rebuild.
- `h2d` — the two blocking H2D copies (~13.3 KB): legacy null-stream semantics order
  them behind the shared chain still in flight, so the chain's UNHIDDEN tail exposes
  here, not in `sync1`.
- `sync1` — the first `device_sync`: expected ~0 (the H2D just drained the null
  stream); it is the `xq`-read guarantee, not the exposure site.
- `launch` — ticket waits + the resident range launch + per-miss launches, host enqueue.
- `sync2` — the closing `device_sync`: resident-batch compute and miss stragglers
  expose here, whichever stream drains last.
- `drain` — the accumulator drain launch enqueue (its GPU time retires at the
  end-of-layer sync, in the PROFILE remainder — the bracket does not contain it).
- gpu `shared` — null-stream pair around the shared chain: gate/up GEMVs, swiglu,
  `act_quant`, down GEMV.
- gpu `res` — compute-stream pair around the ONE resident range launch (M3b's boundary,
  unmoved): resident-batch expert compute.
- gpu `miss` — miss-stream pair, before the first straggler's ticket wait → after the
  last straggler's launch: fetch exposure + straggler compute. Both records are SKIPPED
  on a no-miss layer (and `res` when `n_res = 0`): a reused event retains its previous
  recording, so reading an unrecorded pair would book a stale span, not zero.

What stays unbucketed, said here as the rule requires: the per-miss split of `miss`
into fetch-wait vs kernel time (needs an event pair per miss or a mid-stream query —
the off-thread `fetch`/`ms/miss` counters already price that side), and the drain
kernel's execution (retires outside `moe`, in the remainder's end-of-layer sync).

Instrument cost, argued as M4's: ≤8 added records (2 attn + 2 shared + 2 res + 2 miss,
the miss pair usually skipped at 0.115 miss/layer) and ≤6 added completed-event queries
per layer — worst case ~600 records + ~470 queries per token against M4's 258+215,
O(2 ms) — plus 8 `Instant` reads per routed call (noise). The control is the run's wall
gate, not the argument.

**Byte budgets per sub-span** — attn rows from M4's per-tensor table (same artifact
header read), moe rows from the config's `moe_inter = 2048`, `hidden = 4096`, the
13.37 MB fp4 expert slot, and the recorded 2538 decode misses / 511 tokens = 0.1155
misses per layer-token:

| span | per layer | per token | at 193.8 GB/s |
|---|---|---:|---:|
| qkv | `wq_a` 4.19 + `wq_b` 33.55 + `wkv` 2.10 + norms/scales ~0.03 MB | 1.714 GB | 8.8 ms |
| attend | ≤1 MB gathered KV + sink/idxs | ≤0.043 GB | ≤0.2 ms |
| oproj | `wo_a` 33.55 + `wo_b` 33.55 + scales ~0.02 MB | 2.887 GB | 14.9 ms |
| shared | 3 × [2048, 4096] fp8 = 25.17 MB + scales/activations | 1.084 GB | 5.6 ms |
| res | 5.885 residents × 13.37 MB = 78.7 MB | 3.383 GB | 17.5 ms |
| miss | 0.1155 × 13.37 MB (compute side; the fetch is off-thread, 1.99 ms/miss) | 0.066 GB | 0.34 ms |
| desc/h2d/launch/drain/sync1 | ~13.3 KB H2D, rest launch/host | ~0.001 GB | ~0 |

(attn sub-spans sum to M4's 4.640 GB ≈ 23.9 ms; moe device spans sum to ~23.4 ms of
bytes, the floor `moe`'s 48.9 is ranked against.)

**Prediction — attn split, registered 2026-08-08 before the measurement so it can be
wrong** (attn measured 52.8; the three must tile it, so the bands are a partition):
**qkv 17–24 (point 20), attend 2–6 (point 4), oproj 26–32 (point 28.5)** — both
GEMV-heavy spans at ~2.0–2.3× their bytes, the M3a family (fp8 GEMV suspected
issue/latency-bound at m = 1, unverified — that suspicion is what the split prices),
with qkv additionally carrying ~7–9 small-kernel/copy ops of the hcn latency class.
**What each outcome selects:** attend > 10 ⇒ `sparse_attn` is an hc_pre-class
wrong-shaped kernel and its read/widening outranks the GEMV lever; qkv ≈ oproj ± 20%
despite the 8.8-vs-14.9 byte ratio ⇒ the excess follows op COUNT, the per-launch/small-
kernel class is the lever (fuse/batch the chain), not GEMV ISA; both GEMV spans ≈ 2×
bytes with attend small ⇒ the fp8 GEMV kernel-rate ISA read is warranted (M3a pattern),
corroborated independently if moe's gpu `shared` also lands ≥ 1.5× its 5.6.

**Prediction — moe split, registered 2026-08-08 with it** (moe measured 48.9; the host
spans must tile it): **sh_enq 0.5–2 (1), desc 2–5 (3.5), h2d 4–8 (7), sync1 0–1 (0.4),
launch 1–3 (1.5), sync2 26–36 (33), drain 0.5–2 (1.2), resid 0.5–2 (1.3)** — point sum
48.9 — **and gpu shared 6–10 (7.5), res 17–24 (20), miss 6–12 (9)**. This IS the
closure hypothesis for moe's +31 over bytes: resident compute near its 17.5 byte floor
(M3c bought the fp4 loop ~45% issue headroom), miss exposure ≈ 4.96 × 1.99 less
overlap, the shared chain modestly above its 5.6, host residue ~7. **Kill conditions:**
res > 24 ⇒ the fp4 loop is NOT stream-bound despite the 105-instr count — the
memory-system/geometry question re-opens (the 88-instr rung does NOT follow; it widens
issue, not bytes); miss ≥ 10 ⇒ miss exposure is the top moe lever (overlap stragglers
past the closing sync — a device-side join, the stream-ordered accumulate — or buy
residency); gpu shared ≥ 9 ⇒ the fp8 GEMV rate lever, jointly with the attn split's
verdict — one ISA read serves both; host residue (sh_enq+desc+launch+drain+resid) ≥ 10
⇒ the host loop is the lever (descriptor rebuild caching, one H2D); sync2 < 24 with moe
unchanged ⇒ the sit-of-time model is wrong somewhere visible in the other spans —
replicates before any claim. |any span − its point| inside recorded bucket noise
(~±1.5 ms) is NOT a finding either way at n = 1.

**The staged run (GO, one run — control, not an A/B: the splits ride the recorded
130.7 baseline):** release binary built at this branch's committed HEAD BEFORE the
device is requested; flag-identical to M2..M5, the 218-token prompt verbatim from
benchmarks.md's head-to-head CORRECTED note (md5 `bc71afa745d980be7d21860f70ad96aa`);
exclusive flock with the per-minute KFD+GTT+llama-swap witness:

```
flock /var/run/sys-gpu.lock -c 'target/release/rivoli /var/db/rivoli/v4-f4-full -bench 512 --prompt "<the 218-token prompt>"'
```

**Gates, registered with the predictions:** (1) wall within ±3% of the recorded 130.7
ms/token — a breach is reported BEFORE any reading of either split. (2) output
byte-identical: the reply's escape-decoded full text must md5 to the recorded
`75b19fcde806059b45c515259feb16d2` (the convention since M5), and expert lookups
identical (179389 hit / 8693 miss, 2538 raw decode misses). (3) both splits sum to
their parents: ATTN-SPLIT resid within ±0.05, MOE-SPLIT resid ≥ 0 (printed identities;
dev asserts are compiled out in release, so the gate is the printed numbers). (4)
coherence, expected not enforced: attn/cmp/hcn/gate/win within their recorded spreads
of the M5 arm-B line (the whole-call `attn` span is kept precisely so the split cannot
drift from it unnoticed), and `h2d + sync1` ≈ gpu `shared` − the part `route_row`'s
~0.6 ms of host math hides. The record lands in benchmarks.md beside "V4 hcn A/B"; the
deliverable is the measured double decomposition and the ranked lever list it selects —
the attack itself is the next stretch.

## M2 — provenance of the command

The recorded head-to-head wrote a literal `"<prompt>"` where its prompt should be — the
218-token text was in no doc. Recovered 2026-08-07 from the run's own log (startup
tokenizer line, byte-complete, 218 tokens confirmed) and recorded verbatim in
benchmarks.md's head-to-head CORRECTED note. The instrumented command as run — flag-identical,
flock-wrapped, release built before the lock, with the per-minute KFD+GTT+llama-swap
witness (`docs/reference/gpu-lock.md`'s corrected note is why KFD alone is insufficient) —
is in the benchmarks.md record, the canonical home for a recorded command.

## Kill condition

If M2 shows the floor arithmetic above is wrong (e.g. the dense phase does not re-read
8.9 GiB, or achievable bandwidth on this path is far below 193.8 GB/s), say so and re-derive
the ceiling before proposing any lever — a target computed from a wrong floor is how GLM's
early perf rounds went wrong, per `docs/measurement/perf-roadmap.md`'s opening.

*[Applied 2026-08-07: M2 tested it. The floor was wrong by ~10%, in the direction and for
the reasons recorded in the CORRECTED bracket under "The measured budget" — re-derived ~62
ms/token, ceiling ~16 tok/s, no kill. The levers above are ranked against the corrected
floor.]*
