---
scope: v4
status: live
verdict: OPEN. V4 decodes at 5.389 tok/s (185.5 ms/token) against a ~65-70 ms/token bandwidth floor — 2.8x, so ~120 ms/token is not bytes and nobody has measured where it goes; v4gpu emits one summary line and no phase buckets. Port GLM's route/moe/fetch buckets, run one instrumented decode, then rank the levers. 10 tok/s needs no quality tradeoff if the decomposition confirms the floor.
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
3. **Dense/attention phase** — the 8.9 GiB re-read is the largest single term; `hc_mult 4`
   hyper-connections add elementwise volume on top.
4. **Launch/host overhead x 43 layers** — GLM's unbucketed remainder was ~25 ms/token.
5. **Speculation structurally off** (fp4 kernel instantiates R=1 only) — a multiplier for a
   later stretch, not a sink; GLM's gated MTP measures 1.108x.

## Milestones

- **M1 — the buckets (no GPU).** Port GLM's per-phase telemetry (`route`/`moe`/`fetch`
  walls, miss count, ms/miss, plus the unbucketed remainder) into `src/v4gpu.rs`, matching
  GLM's bucket semantics so the two engines' PROFILE lines read the same way. Buckets must
  not add joins the decode does not already pay (GLM's precedent: HIP event pairs, clocks
  started after syncs the path already pays). GLM behavior byte-identical; V4 output
  byte-identical with buckets compiled in.
- **M1b — the kernel read (no GPU).** Read `kernels/moe.hip`'s fp4 path against the HB-16
  lesson and `docs/measurement/how-to-measure.md`'s ISA-first rule; record whether the GLM
  fix applies, as prose in this doc.
- **M2 — one instrumented decode (GPU, via the coordinator).** Same command as the recorded
  512-token benchmark. Record the decomposition in `docs/measurement/benchmarks.md` and
  update this doc's verdict with the measured ranking.
- **M3 — attack in measured order.** Out of scope for this stretch; the M2 table decides it.

## Kill condition

If M2 shows the floor arithmetic above is wrong (e.g. the dense phase does not re-read
8.9 GiB, or achievable bandwidth on this path is far below 193.8 GB/s), say so and re-derive
the ceiling before proposing any lever — a target computed from a wrong floor is how GLM's
early perf rounds went wrong, per `docs/measurement/perf-roadmap.md`'s opening.
