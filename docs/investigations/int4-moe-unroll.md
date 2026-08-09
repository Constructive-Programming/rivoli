---
scope: glm
status: live
verdict: OPEN, registered 2026-08-09 and not yet measured. M11 measured +17.7%/+33.2% serial rate on `dot_f4_wave_r` from a one-line `#pragma unroll` (V4, R=1, byte-identical, 9.38 → 9.61 tok/s). `dot_i4_wave_r` — GLM's int4 and hybrid-HOT resident MoE path, i.e. the shipping default's expert compute — carries the IDENTICAL un-unrolled dword loop with the same 8:1 activation:weight byte ratio and the same in-body drain. It is unmeasured, and the fp4 number must NOT be copied across: GLM instantiates R = 2 for speculative decode, so the same pragma multiplies register cost against a loop that already carries 2R accumulators and 2R float4 activation reads per step. The wall question is separately hard — GLM decodes 2.07 tok/s at 67.7% expert hit and 180.4 miss/token x 1.44 ms/miss, so roughly HALF the token is fetch exposure and a kernel saving of a few ms is below the wall's resolving power. The deliverable is therefore the KERNEL RATE and a shippability decision, not a wall delta.
---

# Does GLM's int4 MoE loop have the same MLP gap fp4 had?

## Where this comes from

M11 (`v4-decode-decomposition.md`) established, on V4's fp4 resident kernel:

- the loop issued 4 loads and drained them in the same body — **one iteration in flight**;
- stripping the *entire* decode and every FMA bought only **+12.8%**, so issue rate was
  never the bound;
- one line of `#pragma unroll` bought **+17.7%** (depth 2) and **+33.2%** (depth 4), serial,
  **fingerprint-identical**, and **+0.23 tok/s** through the engine byte-identically;
- at depth 4 the decode's cost falls to **2.5%** — the loop becomes, to within 2%, purely
  memory-bound at 195.3 GB/s.

`kernels/common.hpp::dot_i4_wave_r` is the same shape: two-path (aligned dword fast path +
scalar tail), one dword = 8 nibbles = 8 consecutive columns per lane, one scale per 8, two
`float4` activation reads per `R` per step, `wave_sum` fold at the end, **no `#pragma unroll`
on the dword loop**. The comment recording this gap is already in place at the loop.

**This is not a V4 path.** `dot_i4_wave_r` is GLM-5.2's int4 expert compute, which is
`--mode int4` outright and, through the HOT format, `--mode hybrid` — **the shipping
default**.

## Why the fp4 result does not transfer, stated before anyone measures

1. **R = 2 exists here.** fp4 refuses `nrow != 1` (`moe.hip`, return 1003). GLM instantiates
   R = 2 because speculative decode was measured to pay (1.108x at `--mtp-min-conf 0.8`).
   The loop body holds `2R` accumulators and reads `2R` `float4`s per step; an unroll of
   depth D multiplies the in-flight register cost by D **on top of** R. fp4 at depth 4 already
   cost `moe_gateup_f4` 10 of 16 waves/SIMD at R = 1. The same pragma at R = 2 could
   plausibly collapse occupancy — or not, since M11 also measured that **in-flight iterations
   per SIMD, not wave count, is the quantity that predicts throughput** (C2 won at 10 waves x
   depth 4 = 40 over C1's 16 x 2 = 32). Both readings are live; that is why this is measured.
2. **GLM's residency and access pattern differ.** 67.7% hit against V4's ~98%, 180.4
   miss/token against 4.96, and a fetch stream that is doing far more work alongside the
   compute. A loop that is memory-latency-bound in an idle microbench may be bound by
   something else entirely when the fetch path is saturating the same controllers.
3. **There are three loops here, not one.** `dot_i4_wave_r<R>` (the MoE resident path),
   `dot_i4_wave` (non-`_r`, a separate copy with the same un-unrolled dword loop, called from
   `linalg.hip`), and `dot_vq_wave_r` (int3-vq — a **random codebook gather**, explicitly a
   different memory pattern, and the COLD half of hybrid). Scope this stretch to
   `dot_i4_wave_r` first; report whether the other two share the gap without attacking them.

## The wall problem, named up front so nobody grades this on the wrong number

GLM's recorded decode is **2.07 tok/s = 483 ms/token** with **180.4 miss/token at 1.44
ms/miss ≈ 260 ms/token of fetch exposure**. Even a total elimination of int4 expert compute
could not move the wall by more than what compute actually occupies, and a +17%-class kernel
win on a fraction of that is **below the wall's resolving power at any n this project can
afford**.

**That is not a reason to skip it, and the ranking rule this repo already carries says so:**
a perf win counts even when it hides behind another bottleneck — efficiency is recurring
cost, and today's bottleneck is not permanent (residency rises with budget, artifact and
cache work). What it *is* a reason for: **do not stage a wall A/B and report "no effect".**
Measure the kernel, and measure the compute span if one is instrumented.

## Milestones

**G1 — the serial rate, no engine.** Extend `examples/dot_bench.rs` (it already carries int4
rows, and M11's `v4res` section is the pattern) with a GLM-shaped `glmi4` section: the real
expert dims read from the artifact manifest — **verify them, do not assume 6144/inter from
prose** — at **R = 1 and R = 2**, depths 1 / 2 / 4, over a **≥ 1 GB rotating working set**
with an ~80 MB single-range control printed beside it, exactly as M11 did. The control is
what turns the MALL confound from a worry into a measurement.

**Registered prediction, R = 1: +10..+25% at depth 2** (fp4 gave +17.7% on the identical
shape). **R = 2: +0..+15% at depth 2**, wide because register pressure and in-flight-iteration
count pull opposite ways and M11 showed the second one winning once already.
**KILL: any arm slower than stock by >3% ⇒ report and stop** — that is the register-pressure
outcome and it ends the stretch as a negative.
**KILL: depth 4 at R = 2 drops below 6 waves/SIMD ⇒ do not measure it as a candidate**, record
the ISA and say so.

**G2 — the static read, free, do it first.** VGPR/SGPR/occupancy for `moe_gateup_i4` and
`moe_down_i4` at R ∈ {1,2} x depth ∈ {1,2,4}, plus the `vmcnt` structure per iteration. M11's
occupancy model was WRONG in its ranking (it put depth 2 first; depth 4 won), so this read
**ranks candidates, it does not eliminate them** — an arm predicted worse still gets measured
unless it trips the ≥6-wave kill.

**G3 — bit-identity, if any arm wins.** The fingerprint discipline M11 landed:
non-degenerate, and **demonstrated red against a deliberately reassociated body** before any
"byte-identical" claim is made. A gate only ever seen green is not a gate.

**G4 — the shippability decision.** If a depth wins on rate and holds the fingerprint, the
question is whether it merges. It needs: a **multi-trip test that executes the unrolled body
at the winning depth** (`v4_kernel.rs::the_dword_path_matches_the_oracle_at_multiple_trips`
is the template, including its non-vacuity argument — GLM's toy configs will have the same
one-trip blindness), and an engine run that is **byte-identical on `--mode int4`**.

**Use `--mode int4`, NOT hybrid, for any engine check.** Hybrid's cache picks each expert's
*format*, so residency selects the arithmetic (INV-1 exception, `architecture.md` §8b) and a
hybrid A/B cannot separate a kernel change from a residency difference. Hold `--max-mem` and
`--cache-policy` fixed regardless.

## Kill condition for the stretch

If the loop is already at its memory-bound ceiling at R = 2 — i.e. the ballast-vs-real gap
that M11 measured at 12.8% is small here — then the drain is not the limiter on this path and
the stretch closes as a negative with the ISA read as its evidence. **A well-evidenced
negative is a complete result**; the fp4 win does not oblige this one to exist.
