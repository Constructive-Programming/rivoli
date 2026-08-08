---
scope: v4
status: live
verdict: OPEN, two levers LANDED. M3c measured 2026-08-08: branchless e2m1/e8m0 decode + global-address-space descriptor loads (fp4 dot loop 195 → 105 instr per 128 weight bytes) took moe 54.9 → 49.6 ms/token and wall 170.3 → 165.3 = 6.048 tok/s (+3.0%), output byte-identical, hit/miss identical, fetch flat in bytes — the registered prediction (−4..−10, point −6) landed IN BAND at −5.3, the first M3-series prediction to do so. M3b (launch geometry, moe 70.6 → 54.7, 2026-08-07) stands under it. Residue to 10 tok/s (~65 ms off the wall): route ~76 (attention phase still needs its own split) > moe ~31.6 above the 18 ms byte floor (issue-bound excess cashed out; miss exposure and shared GEMV unpriced) > remainder ~31 non-tail. M2's decomposition and floor (~62 ms/token ≈ 16 tok/s ceiling) stand; buckets still not certified free (wall spread ±1.5% over three stock-class runs).
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
