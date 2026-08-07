---
scope: v4
status: live
verdict: OPEN. V4 decodes at 5.389 tok/s (185.5 ms/token) against a ~65-70 ms/token bandwidth floor — 2.8x, so ~120 ms/token is not bytes. M1 DONE 2026-08-07 (route/moe/fetch buckets + printed remainder in v4gpu, GLM semantics, no new syncs); M1b DONE (the HB 8→16 fix does not transfer — it was mla_latent_attend, not the MoE GEMV, and at R=1 the fp4 path has no shared-operand reuse axis for it). M2 — one instrumented decode on the recorded command — is prepared and NOT yet run; it ranks the levers.
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
- **M2 — one instrumented decode (GPU, via the coordinator). PREPARED, not run — §"M2 — PREPARED" below.** Same command as the recorded
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

Cost bound, by argument as GLM's is: ~8 `Instant` reads per MoE layer per token (~350
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
~850/token, issued serially by the host loop onto two streams.

## M2 — PREPARED 2026-08-07, not run: the exact command

The recorded head-to-head wrote a literal `"<prompt>"` where its prompt should be — the
218-token text was in no doc. Recovered 2026-08-07 from the run's own log (startup
tokenizer line, byte-complete, 218 tokens confirmed) and recorded in benchmarks.md beside
the run, in a dated note. The instrumented decode is flag-identical to the recorded one:

```
flock /var/run/sys-gpu.lock -c \
  'target/release/rivoli /var/db/rivoli/v4-f4-full -bench 512 --prompt "<the 218-token prompt now recorded in benchmarks.md>"'
```

Release build BEFORE taking the lock. Witness: `docs/reference/gpu-lock.md`'s corrected
note establishes that KFD is blind to llama-swap's Vulkan tenants; the sampling plan built
on it is this investigation's own — per-minute KFD holder count AND `mem_info_gtt_used`
AND the llama-swap `/running` HTTP probe, any step-change beyond the run's own footprint
discarding the run with a timestamp. Control = the recorded 95.0 s / 185.5 ms/token wall,
±~1% or the buckets are reported as non-free before the decomposition is read.

## Kill condition

If M2 shows the floor arithmetic above is wrong (e.g. the dense phase does not re-read
8.9 GiB, or achievable bandwidth on this path is far below 193.8 GB/s), say so and re-derive
the ceiling before proposing any lever — a target computed from a wrong floor is how GLM's
early perf rounds went wrong, per `docs/measurement/perf-roadmap.md`'s opening.
