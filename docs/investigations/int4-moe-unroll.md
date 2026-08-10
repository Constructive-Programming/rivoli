---
scope: glm
status: closed-shipped
verdict: CLOSED-SHIPPED 2026-08-10: `#pragma unroll 4` on `dot_i4_wave_r`'s dword loop, merged. Measured +12.6% (R=1) / +16.4% (R=2) / +20.1% (e_count=1) serial rate at the artifact's dims over a 1.083 GB rotating set, fingerprint-identical both token rows, two counterbalanced passes; the two adjacent negatives priced in the same round (AS1/`gu8p` typing ±0.6%, decode-removed ballast +0.5% — the whole gap was memory-level parallelism, and the static reads mis-ranked twice: R=2 does NOT collapse occupancy, and the lgkmcnt coupling was worth nothing). Engine A/B at `--mode int4` (128 tokens, the recorded prompt, MTP active): BYTE-IDENTICAL — reply md5 `ba97d99d983f1641469d4d0ca6aaf086`, 143013 hit / 42197 miss, MTP 63/84, all identical between arms, so the R=2 path is byte-neutral on the engine, not just the microbench. NO wall claim: n=1, 128 tokens, fetch-dominated — the value is recurring-cost efficiency per the ranking rule. Gate: tests/kernel.rs multi-trip pair at 1280/1024 (5 and 4 trips, depth-4 remainder entered; tolerance red-target 479.6×), landed after the first fixture drove moe_fixed into ±2^14 saturation and failed 845× against a STOCK kernel — the saturation precondition is now asserted per-contribution in the reference. `dot_i4_wave` (non-_r) and `dot_vq_wave_r` were NOT touched: the former is a separate copy with the same gap (unmeasured), the latter a random gather (different pattern). Status moves live→closed-shipped with the merge.
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

> **RUN 2026-08-09 by the coordinator, after the authoring agent was killed mid-work: the
> G4 test as committed FAILS, and the fault is the FIXTURE, not the kernel.**
>
> - `the_i4_multi_trip_tolerance_can_see_a_past_first_trip_defect` (host-only, no device)
>   **PASSES**: `err=5.059e4` vs `tol=1.055e2` = **479.6x**. The red-target is decisive and
>   the tolerance is sensitive to a past-first-trip perturbation, as designed.
> - `the_i4_dword_path_matches_the_oracle_at_multiple_trips` (device) **FAILS**:
>   `err=8.910e4 > tol=1.055e2`, against `max=1.055e5` — the error is the same order as the
>   data, i.e. a gross mismatch, not a numerics margin.
> - **This was measured against the STOCK kernel.** `git diff c9c6f3d..HEAD -- kernels/` is
>   empty on this branch; the unroll arms were scratch-tree patches and were never applied
>   here. So the failure cannot be attributed to `#pragma unroll`.
> - **The kernel is exonerated by the three pre-existing int4 tests, which all pass** on the
>   same binary and the same device: `moe_i4_matches_reference` (err 9.766e-4 / tol 2.465),
>   `moe_i4_real_data_matches_cpu` — **which reaches 24 and 8 dword trips on real artifact
>   data** — at `cosine(GPU,CPU)=1.0000`, err 1.222e-6, and `moe_i4_real_data_vs_fp8_ground_truth`
>   in band. A kernel that agrees with the reference to 1e-6 at 24 trips is not broken at 5.
>
> So `i4_multi_trip_fixture` (or the `i4_reference`/launch path for that shape) is
> constructed wrong and the test cannot certify anything yet. It had never been executed —
> the authoring agent was killed before it ran, so "seen only green" overstated it.
> **The +12.6/+16.4/+20.1% rate result is untouched by this**: those were fingerprint-gated
> rate measurements, not reference comparisons.
>
> Witness caveat: `whisper-large-v3-turbo` held 1.67 GB of GTT throughout (unwrapped, no
> flock — the known `hr-fleet` gap). It cannot explain an 845x numerical disagreement, and
> the three control tests passed under the identical condition, but the arm is not
> witness-clean and is recorded as such.

> **ROOT-CAUSED 2026-08-10, host-only, by the fixture's author: `moe_fixed` SATURATION.**
> The identification is arithmetic, not narrative — the GPU clamps every per-expert
> contribution at ±`MOE_ACC_MAX` = 2^(58−44) = **16384** (`common.hpp::moe_fixed`); the CPU
> reference does not; and the measured failure is **`err = max − 2^14` to three significant
> figures** (1.055e5 − 1.6384e4 = 8.912e4 against the measured 8.910e4). The first fixture
> drew weights up to ±4, which at 1280 columns puts σ(gate·x) ≈ 31, h = silu(g)·u at O(10³–10⁴),
> and the down pass at O(10⁵) — 6.4× past the clamp. **The hazard was already written down, by
> the same author, in `dot_bench.rs::run_glm_i4`** ("scales small enough that the partials stay
> far under `moe_fixed`'s ±2^14 saturation") — the probe respected it and the test fixture
> ignored it. This is also why the host red-target passed while the device test failed: the
> red-target never crosses the GPU, so it certifies the tolerance and nothing about the launch.
>
> **Fixed by measurement, and double-guarded:** the fixture's weight draw is scaled by 0.02
> (reference now peaks O(1), four orders under the clamp; the red-target re-proven at the new
> magnitudes: `err=3.227e-1` vs `tol=1.694e-3` = **190×**, and re-proven RED with the injection
> neutered), and `moe_reference` itself now **asserts every per-expert contribution is under
> the clamp** — proven to fire on the exact defective fixture (`|2.124e4| >= 2^14 at expert 0
> output 0`), so this class cannot reach a device again: it dies at the first host run.
> `moe_i4_matches_reference`'s magnitudes pass the new guard, and its 9.766e-4 on-device error
> is itself proof its contributions never clamped. Device re-verification of the multi-trip
> test is queued for the next grant, alongside the broken-kernel red proof.

**G4 — the shippability decision.** *[SHARPENED 2026-08-09 by the G1 correctness review, which
went and looked at what the suite actually covers rather than assuming the fp4 precedent
transfers. The concrete hole and the fixture that closes it are recorded below; the test is
NOT written yet, because G4 is gated on an arm winning.]*

**Today, nothing checks `dot_i4_wave_r`'s dword loop past its first trip on a box without the
artifact, and nothing checks it past the first UNROLL trip on any box.** Two fixtures exist and
neither closes it:

- `tests/kernel.rs::moe_i4_matches_reference` is `hidden = 256, inter = 128`, so gate/up runs
  the dword loop **exactly once** and `moe_down_i4` **never enters it at all** (128 < `WAVE*8`
  = 256) — the same one-trip blindness `V4Config::toy` had on the fp4 side.
- `tests/kernel.rs::moe_i4_real_data_matches_cpu` does reach 24 and 8 trips against a CPU
  oracle, but it **`return`s early when `/var/db/rivoli/glm52-vq3-full/L03.i4` is absent**, and
  24 and 8 are both divisible by 2 and by 4 — so at `#pragma unroll 2` *or* `4` **the unroll
  remainder is never executed**.

The fixture that closes both: an `ne = 1` case at **`hidden = 1280, inter = 1024`** — 5 and 4
trips, both multiples of `I4_GROUP` = 128 — beside `moe_i4_matches_reference`. That is the int4
twin of `v4_kernel.rs::the_dword_path_matches_the_oracle_at_multiple_trips` (1280/1024), which
commit `c9c6f3d` added for exactly this reason, and it inherits that test's non-vacuity
argument: it must be shown RED against an injected defect that only bites past the first trip,
not merely observed green.

**The original G4 text follows unchanged.** If a depth wins on rate and holds the fingerprint, the
question is whether it merges. It needs: a **multi-trip test that executes the unrolled body
at the winning depth** (`v4_kernel.rs::the_dword_path_matches_the_oracle_at_multiple_trips`
is the template, including its non-vacuity argument — GLM's toy configs will have the same
one-trip blindness), and an engine run that is **byte-identical on `--mode int4`**.

**Use `--mode int4`, NOT hybrid, for any engine check.** Hybrid's cache picks each expert's
*format*, so residency selects the arithmetic (INV-1 exception, `architecture.md` §8b) and a
hybrid A/B cannot separate a kernel change from a residency difference. Hold `--max-mem` and
`--cache-policy` fixed regardless.

## G2 — DONE 2026-08-09, no GPU: the shape is confirmed, the R = 2 occupancy fear is NOT, and there is a SECOND gap this plan did not register

Method: `kernels/` copied to a scratch tree, `#pragma unroll D` inserted above
`dot_i4_wave_r`'s dword loop only (the patch script asserts there are exactly three such
loops and that the first is inside `dot_i4_wave_r`, so a refactor cannot silently patch
`dot_i4_wave` or `dot_f4_wave_r` instead), then `hipcc --offload-arch=gfx1151 -O3 -fPIC` —
build.rs's own flags — with `-Rpass-analysis=kernel-resource-usage` and
`--cuda-device-only -S`. **No shipped kernel was modified.**

### Registers and occupancy — the registered kill does NOT fire

| kernel | depth 1 (stock) | depth 2 | depth 4 |
|---|---|---|---|
| `moe_gateup_i4` (R=1) | VGPR 39, **occ 16** | 54, **16** | 88, **16** |
| `moe_gateup_i4_r2` (R=2) | 49, **16** | 73, **16** | 123, **10** |
| `moe_down_i4` (R=1) | 32, **16** | 49, **16** | 83, **16** |
| `moe_down_i4_r2` (R=2) | 42, **16** | 67, **16** | 95, **16** |

SGPR 20/22 throughout. **Zero VGPR spill, zero SGPR spill, 0 bytes scratch on all twelve
cells.**

**The "R = 2 register pressure could collapse occupancy" worry that this plan is built on is
not supported.** Depth 2 keeps 16 waves/SIMD at *both* widths and on *both* kernels — the
same rung fp4's winning-but-safer C1 sat on. The worst cell in the table is R = 2 depth 4 at
**10 waves/SIMD**, which is *exactly* where fp4's `moe_gateup_f4` landed at depth 4 (10
waves, 125 VGPR) — and that is the arm that WON there, at +33.2%. The registered
"depth 4 at R = 2 drops below 6 waves/SIMD ⇒ do not measure it" **does not fire**; all six
arms are measurable.

Per §G2's own rule this ranks and does not eliminate, and M11's precedent says the ranking is
not to be trusted anyway (its occupancy model put depth 2 first and depth 4 won).

### The drain is there, confirmed in the ISA, at both widths

Stock `moe_gateup_i4`'s dword loop (`.LBB8_7`, 68 instructions) issues **4 vector loads and
waits every one of them down inside the same body**: `s_waitcnt vmcnt(3) lgkmcnt(0)` →
`vmcnt(2)` → `vmcnt(1)` → `vmcnt(0)`, with the final two `v_fmac_f32` after the last wait.
**One iteration of loads in flight, ever** — M7's disease, byte for byte the diagnosis M11
made on `dot_f4_wave_r`. R = 2 is the same picture with 6 loads (`.LBB9_7`, 82 instructions,
`vmcnt(5)`→`(4)`→`(2)`→`(1)`→`(0)`).

Unrolling does what it did on fp4 — it stops the drain being per-iteration. **Two wait columns,
and the gap between them is the finding:**

| arm | loads/iteration | ENTRY wait (first `s_waitcnt` in body) | deepest wait in body |
|---|---|---|---|
| R=1 stock | 2 `flat_load_b32` + 2 `global_load_b128` | `vmcnt(3)` | `vmcnt(3)` |
| R=1 unroll 2 | 4 + 4 | `vmcnt(5)` | `vmcnt(5)` |
| R=1 unroll 4 | 8 + 8 | **`vmcnt(11)`** | `vmcnt(12)` |
| R=2 stock | 2 + 4 | `vmcnt(5)` | `vmcnt(5)` |
| R=2 unroll 2 | 4 + 8 | `vmcnt(9)` | `vmcnt(9)` |
| R=2 unroll 4 | 8 + 16 (24 loads) | **`vmcnt(4)`** | `vmcnt(19)` |
| *`moe_gateup_f4` as SHIPPED (unroll 4)* | *4 `global_load_b32` + 4 `global_load_u8` + 8 `global_load_b128`* | *`vmcnt(15)`* | *`vmcnt(15)`* |

> **CORRECTED 2026-08-09, before any device time, by the G1 code-quality review.** This table
> had ONE column, labelled "first wait counts down from", carrying `vmcnt(12)` and `vmcnt(19)`
> for the two depth-4 rows. Those are the *deepest* waits, not the first ones; every other row
> happened to have them coincide, which is why the mislabel survived. Re-read from the `-S`
> output: **R = 2 depth 4 stalls at `vmcnt(4)` on entry** — it drains the previous iteration
> down to four outstanding loads before doing anything, despite having 24 in flight. That is a
> **pre-registered reason for C2 to disappoint at R = 2**, and it is exactly the arm at 123 VGPR
> / 10 waves/SIMD that this round is most curious about. Recording it now, before the device,
> so a weak C2-at-R=2 is a confirmed prediction rather than a post-hoc explanation.

### The thing this plan did not register, and it is not small

**`dot_i4_wave_r` issues its WEIGHT dword and its GROUP SCALE as `flat_load_b32`. The fp4
loop issues `global_load_b32`/`global_load_u8`.** That difference is not incidental — it is
exactly the `gu8p` (`__attribute__((address_space(1)))`) typing M3c put on `dot_f4_wave_r`'s
`row`/`scalerow`, whose own comment says the flat lowering "is a measured cost on an
issue-bound loop". The int4 path never got it: `ExpertDescI4` holds plain
`const unsigned char*` / `const float*`, so the compiler cannot prove global address space
and falls back to flat.

Two consequences visible in the ISA:

1. **The body's ENTRY wait is `lgkmcnt(0)`-coupled**, where every wait in the fp4 loop is pure
   `vmcnt`. On GFX11 a flat load increments both counters, so the first `s_waitcnt` of each
   iteration reads `vmcnt(3) lgkmcnt(0)` (R = 1) / `vmcnt(5) lgkmcnt(0)` (R = 2) — and that is
   the wait the drain sits behind. A conservative `lgkmcnt(0)` at the top of the body is a
   second, independent reason this loop cannot keep work in flight, and it is one an unroll may
   not remove.

   > **CORRECTED 2026-08-09, before any device time, by the G1 code-quality review, which
   > re-derived this from its own `hipcc` run.** This point was written as "**Every** int4 wait
   > is `lgkmcnt`-coupled … `vmcnt(3) lgkmcnt(0)` … `vmcnt(0) lgkmcnt(0)`". Measured, stock
   > `.LBB8_7` is `vmcnt(3) lgkmcnt(0)`, `vmcnt(2)`, `vmcnt(1)`, `vmcnt(0)` — **one of four**
   > carries `lgkmcnt`, and the closing wait is bare. Same shape at R = 2 (one of five) and in
   > `moe_down_i4`. The substantive point stands and is restated above; the word "every" did
   > not, and it had reached the `verdict:` line, which is what most readers see.
2. **Address arithmetic per iteration.** Flat needs a 64-bit VGPR address pair, and the body
   carries the `v_add_co_u32` / `v_add_co_ci_u32_e64` pairs to build it.

**Registered as an additional, independent, bit-neutral candidate arm (call it G) — NOT taken
in this stretch without a decision from the coordinator**, because it changes `ExpertDescI4`
and `dot_i4_wave_r`'s signature rather than adding one pragma. It is also a *confound for the
unroll question in one direction*: if the `lgkmcnt` coupling is what forces the conservative
drain, unrolling buys less here than it did on fp4, and the honest reading of a small unroll
win would be "the wrong gap was closed", not "the loop is at its ceiling".

### Issue rate, by M3a's slot model, is not binding — but this is a model, not a measurement

A wave-iteration reads 128 weight bytes. Stock R = 1 is 68 instructions for those bytes =
1.88 B/cycle/SIMD; R = 2 is 82 for the same 128 = 1.56. By the constant M11 calibrated
(88 instr = 1.45 B/cycle/SIMD ≈ 337 GB/s) that is ≈ **437 and ≈ 362 GB/s**, both far above
anything this box has measured on a decoding loop. Consistent with fp4's probe B1, which killed
the issue-rate suspect dynamically at +12.8%.

Instruction counts here **exclude the block label and include `s_delay_alu`**, which is a real
SOPP on this target; counting the block terminator as well gives 69 and 83. The derived
B/cycle and GB/s figures use the 68/82 convention, matching M11's 88 for the fp4 loop.

> **CORRECTED 2026-08-09, before the device.** This paragraph ended *"**No ballast arm is
> planned here** — B1 already answered this question on the identical loop shape, and re-running
> it would cost device time to re-derive a number, not to test a claim."* The G1 code-quality
> review pointed out that the stretch's kill condition is written in terms of the
> ballast-vs-real gap **here**, which no other arm measures — so that sentence made the kill
> unfireable. **A ballast arm (B1) is now staged**; see the round below for its patch and why
> its `DEGENERATE` fingerprint is expected. The issue-rate model above is unchanged and is still
> a model, not a measurement.

### Two shape facts that matter downstream

- **The scalar tail is unreachable at GLM's dims.** `WAVE * 8` = 256; hidden 6144 = 24 trips,
  inter 2048 = 8. Both divide by 2 and by 4, so **no unroll remainder is entered either**, at
  any planned depth. G4's multi-trip test therefore has the same one-trip blindness fp4's
  did, and needs the same treatment — a config whose trip count exceeds the depth.
- **The activation:weight byte ratio is 8:1 at R = 1 and 16:1 at R = 2** (two `float4` = 32 B
  per row per 4 packed weight bytes). M11 recorded that ratio as the un-priced candidate for
  why the drain costs more on fp4 than it did on fp8; at R = 2 it doubles again. That is a
  *request-volume* reason R = 2 might behave differently, distinct from the register-pressure
  reason this plan registered and the ISA just refuted.

### The other two loops, reported and not attacked (§"Why the fp4 result does not transfer" point 3)

Read from the same ISA dumps, free:

| loop | kernel read | loads/iteration | in-body drain? | weight/scale addressing |
|---|---|---|---|---|
| `dot_i4_wave` (non-`_r`) | `gemv_i4` (`linalg.hip`) | 2 `global_load_b32` + 2 `global_load_b128` | **yes**, `vmcnt(3)`→`(0)` | **global** |
| `dot_vq_wave_r` | `moe_gateup_vq` / `moe_down_vq` | 2 `flat_load_u16` + 1 `global_load_b64` + 1 `global_load_b128` | **yes**, ends `vmcnt(0)` | **flat** |

Both share the un-unrolled in-body drain. Two things follow that are worth having on record:

- **`dot_i4_wave`'s loads are GLOBAL where `dot_i4_wave_r`'s identical source lines are FLAT.**
  Same loop, same header, different lowering — because `gemv_i4` takes its pointers as kernel
  arguments (provably global) while `dot_i4_wave_r`'s come out of an `ExpertDescI4` read from
  memory. That is direct evidence for the AS1 finding above rather than an inference from it.
  `gemv_i4` has no caller in `src/` (it is the microbench/oracle kernel), so this costs nothing
  today; it means an unroll measured on `gemv_i4` would NOT be measuring the MoE path.
- **`dot_vq_wave_r`'s per-iteration read is a random codebook gather** (`global_load_b64`, one
  `VQ_DIM=4` f16 entry). A drain around a *random* access is a stronger latency exposure than
  one around a sequential stream, so if the unroll pays here it plausibly pays more there — and
  vq3 is `--mode int3-vq` outright and hybrid's COLD half. **Registered, not attacked.**

### Dims, verified against the artifact and not against prose

`/var/db/rivoli/glm52-vq3-full/manifest.json`: `hidden_size` **6144**, `moe_intermediate_size`
**2048**, `n_routed_experts` **256**, `n_shared_experts` 1, `first_k_dense_replace` 3,
`num_hidden_layers` 78, `i4_source.group` **128** (= `I4_GROUP`). One int4 expert is
`2·(2048·3072 + 2048·48·4) + 6144·1024 + 6144·16·4` = **20,054,016 B**, and `L03.i4` is
**5,153,882,112 B = 257 × 20,054,016** exactly — 256 routed plus the shared expert. The
scales are **f32**, so a span is `groups · 4` bytes where the fp4 twin's e8m0 scale is one byte
per group.

> **CORRECTED 2026-08-09, before any device time, by the G1 code-quality review — which
> recomputed the claim instead of reading it.** This section and the matching code comment said
> *"carrying `f4_*`'s formula across with the names swapped would understate the per-expert
> bytes by 786 KB and inflate every GB/s by ~4%."* **Wrong three ways.** (1) The named error is
> exactly **zero**: `I4_GROUP`(128) / `F4_GROUP`(32) = 4 = `sizeof(f32)`, so
> `i4_groups(d)·4 == f4_groups(d)·1` at every dim here (6144: 48·4 = 192·1; 2048: 16·4 = 64·1)
> and `f4_expert_bytes(6144,2048) == i4_expert_bytes(6144,2048) == 20,054,016`. Swapping the
> helpers in verbatim gives the identical number. (2) The error that *does* cost bytes is using
> `i4_groups` with a **one-byte** scale, and it is **884,736 B (864 KiB, 4.41%)**, not 786 KB —
> 786,432 corresponds to nothing in this layout. (3) The direction is backwards:
> `report_bytes` computes `bytes / us`, so understating `bytes` reports GB/s **low**, not
> inflated. **The real hazard is the scale WIDTH, not the helper names**, and it would have made
> the probe look ~4.4% slower than it is.

## G1 — MEASURED 2026-08-09: the unroll TRANSFERS (+16.4% at R=2, fingerprint-identical); the addressing does not matter; the ballast inverts M11

Record: `benchmarks.md` "GLM int4 MoE unroll round". Seven arms, two counterbalanced passes
(order reversed in pass 2), primary rows replicated to within 1.6%, no live foreign KFD holder
and an empty fleet in all 28 witness samples.

**Two scope limits on that sentence, stated here rather than left to be discovered.** The whole
14-run round spans **10 seconds** (19:57:07 → 19:57:17), so the replicate bounds *short-timescale
repeatability*; it does not bound session-to-session drift, and the counterbalancing cannot
cancel slow effects like thermal drift or page-cache warmth that need longer than 5 s to differ.
And two arms (**B1, both passes**) exited **rc 101** on a harness assert *after* printing every
measurement row — their rates are intact and used, but 2 of 14 runs terminated in a panic.

| arm | R=1 | R=2 | e1 | verdict |
|---|---:|---:|---:|---|
| A stock | 169.2 | 163.3 | 125.9 | baseline |
| B1 ballast | +0.5% | −1.1% | −1.0% | **the decode is FREE** |
| C1 unroll 2 | +11.6% | +3.0% | +15.4% | qualifies at R=1 |
| **C2 unroll 4** | **+12.6%** | **+16.4%** | **+20.1%** | **qualifies at both** |
| G AS1 only | +0.2% | −0.3% | −0.6% | **not a lever** |
| GC2 AS1+unroll 4 | +11.6% | +16.2% | +19.5% | ≈ C2, purely additive |
| X reassociated | −0.5% | −17.5% | −0.8% | fingerprint RED, as required |

**Which of the four registered outcomes fired: none.** G alone is not ≥ +10%; GC2 is not
super-additive (it is additive to within 1%); C1/C2 are not flat, so the false negative did not
occur; and no candidate arm is >3% slower than stock. The result is the plain one the plan
allowed for but did not predict: **the fp4 lever transfers to GLM's int4 loop, and the
addressing asymmetry this investigation discovered is real, correctly diagnosed, and worth
nothing.**

**The three headline findings.**

1. **`#pragma unroll 4` is worth +12.6% (R=1), +16.4% (R=2) and +20.1% at `e_count = 1`,
   fingerprint-identical on both token rows.** `e_count = 1` is the LOW END of the engine's
   run lengths, and those are **derived, not counted** — at the recorded 67.7% decode hit the
   expected run is 1/(1 − 0.677) ≈ 3.1, so ~1–3. Calling it "the engine's real launch size" (as
   an earlier draft of this section and the verdict both did) upgrades a derived range to a
   measured fact, and it happens to select the cell with the largest gain.
   **The registered bands were for DEPTH 2, and both HIT:** C1 measured **+11.6% at R=1**
   (band +10..+25) and **+3.0% at R=2** (band +0..+15). *An earlier draft scored the depth-2
   R=2 band against depth 4 (+16.4%) and called it "missed high" — a band miss manufactured by
   swapping arms, in the flattering direction. Depth 4 was never banded.*
   The plan's stated worry, that R=2's register pressure would eat the gain, was wrong twice
   over: the ISA refuted the occupancy story before the device, and R=2 then gained *more* than
   R=1.
2. **The AS1/`flat_load` gap is real and free — and arm G was a JOINT registration, not one
   agent's pet theory.** The asymmetry was found here; the coordinator promoted it to a 2 x 2
   and wrote the false-negative argument that justified spending two arms on it. **It is the
   second time in this investigation that a static read ranked candidates wrongly** — M11's
   occupancy model put depth 2 ahead of depth 4 and depth 4 won; here the ISA made a mechanism
   look compelling that did nothing on the machine. The insurance was still worth buying: it
   cost two arms of a round that had the device anyway, and it **eliminated an alternative
   explanation that would otherwise have shadowed every number in the table** — without G, a
   flat C1/C2 would have been unattributable between "the unroll does not transfer" and "the
   addressing is masking it". Recorded as a priced negative, not as wasted work. G eliminated `flat_load_b32` 18 → 0 device-wide
   and removed the `lgkmcnt` coupling from the body's entry wait, at identical VGPR, occupancy
   and instruction count — and moved the rate by +0.2% / −0.3% / −0.6%. **Arm G is closed as a
   measured negative.** It was this investigation's own discovery and its own highest-rated
   suspect after the unroll; it does not survive contact with a number.
3. **The ballast inverts M11.** fp4's B1 bought +12.8%; int4's buys nothing. **Scoped
   precisely: the decode is free AT THE UN-UNROLLED SCHEDULE**, which is the only regime B1
   measures — there the loop drains to `vmcnt(0)` every iteration and is latency-bound, so issue
   rate could not bind whatever the decode cost. Whether the decode binds at depth 4 is **not
   measured here**; M11 priced that with a matched-depth ballast (its B3) and this round has no
   equivalent arm. "No issue-rate component at all" overstates what B1 can say.

**The stretch's kill condition was mis-specified and must not be reused as written.** It closed
the stretch as a negative "if the ballast-vs-real gap is small here". It *is* small — and the
unroll still won +16%. A small ballast gap means the **decode** is not the limiter, which is
perfectly consistent with the **drain** being the limiter. B1 prices the decode; only the C arms
price the drain. Had the round been staged without C arms on the strength of that kill, it would
have closed a stretch that had just found its lever.

**A registered prediction that failed.** The G2 ISA read registered "R=2 depth 4 stalls at
`vmcnt(4)` on entry despite 24 loads in flight — a pre-registered reason for C2 to disappoint at
R=2." **C2 at R=2 is the best arm in the round.** The arm that disappointed at R=2 was C1
(+3.0%, against +11.6% at R=1). Entry-wait depth does not predict throughput; M11's
"in-flight iterations per SIMD" reading stands and this refinement of it does not.

**The MALL control justified the whole discipline on the row that nearly shipped without one.**
The 120 MB control tracks its 1.083 GB row within **1.1% on every arm at R=1**. The `e1` control
at 20 MB — below the 32 MB MALL — reads a **two-pass mean of 314.2 GB/s, 123% of the 256 GB/s
bus**, against 125.9 for the same kernel over 1.003 GB: a naive one-range harness reports this
kernel **2.49× faster than it is**.

*Corrected before commit: an earlier draft quoted 328.9 / 128% / 2.6×, which is pass 2 alone
against a section whose every other figure is a two-pass mean.*

**And the control rows are NOT uniformly well-behaved — the two excursions are reported rather
than dropped, because they are the round's only visible evidence of a transient.** `X` pass 1's
R=2 control read 93.0 GB/s against its own rotating row's 134.8 (**−31%**, 1293.4 µs vs 886.4 µs
in pass 2, a 37.3% pass-to-pass spread) while X's main R=2 row was steady at 134.8/134.6; `GC2`
pass 2's R=2 control read 155.2 against 189.2 (**−18%**), in the same pass that carries GC2's
largest primary outlier (R=1 190.3 → 187.3). Both excursions are *slower*, so they are not cache
service — but "every arm replicated, max spread 1.6%" is true of the **three primary rows only**,
and the GC2 perturbation sits underneath the additivity residual this section reads as
sub-additive.

**Two harness defects the round exposed, and what is and is not fixed:**
- **The witness gate keyed on `kfd_holders=0` and flagged 100% of a clean round** — 12 of the
  14 non-zero samples were STALE entries for the arm that had just exited, and 2 named nobody at
  all. The corrected classifier reports `live_foreign_holders`. **Three caveats the record must
  carry:** the round ran under the PRE-FIX witness, so no stored sample contains the new field
  and re-running the new gate over them flags all 28; the new gate has been exercised only
  against synthetic fixtures, never against a real foreign holder; and it classifies by `comm`,
  so a *sibling agent's* `dot_bench` — the shared-machine case the flock is advisory against —
  would be misclassified as our own. **The round's cleanliness was established by hand from the
  raw samples, not by the corrected gate.**
- **A pre-registered discard rule was relaxed post-hoc.** "Any arm with a non-empty witness is
  DISCARDED, not explained" would have discarded 12 of 14 arms. They were explained instead,
  using a classifier written after the data was in hand. The reasoning is right; the sequence is
  the thing this repo's norms forbid doing quietly, so it is stated.
- **The `row 1 != row 0` assert fired on the B1 ballast** (rc 101, both passes), whose residual
  is a saturated constant by design — a guard red on a *planned* arm. Its first "fix" was worse:
  an `if/else` that printed in both branches and **could never fail**, while its own comment and
  this document both claimed it was "gated on the same non-degeneracy condition". It was gated
  on nothing, because `distinct` never left `run_glm_i4`. **Caught by review before commit**, and
  now a real `assert!` inside `run_glm_i4` where `distinct` is in scope — proven RED on
  (non-degenerate, rows equal) and silent on (degenerate, rows equal).
- **B1's rates are valid but B1's FINGERPRINTS ARE NOT.** Its residual is constant, so all four
  of its cross-arm fingerprint checks passed **vacuously**. A `DEGENERATE` row supports no
  bit-identity claim; B1's reading is its rate alone.

## G4 — the multi-trip test, BUILT 2026-08-09 (no device), and the proposed dims were a trap

`tests/kernel.rs::the_i4_dword_path_matches_the_oracle_at_multiple_trips`, plus its
device-free non-vacuity half
`the_i4_multi_trip_tolerance_can_see_a_past_first_trip_defect`.

One dword trip is `WAVE * 8 = 256` columns, so trips = `dim / 256`, and depth D leaves a
remainder of `trips % D`:

| dim | trips | rem @ d2 | rem @ d4 | covers |
|---|---:|---:|---:|---|
| hidden 6144 (engine) | 24 | 0 | 0 | — |
| inter 2048 (engine) | 8 | 0 | 0 | — |
| **hidden 1280** | 5 | **1** | **1** | the epilogue nothing else reaches |
| **inter 1024** | 4 | 0 | 0 | the CLEAN case — the production geometry |

> **CORRECTED 2026-08-09, before commit, and the correction is against ME.** I first read
> `inter = 1024` as a trap — 4 trips divides by both planned depths, so `moe_down_i4`'s
> remainder never runs — switched the fixture to `inter = 768` (3 trips, remainder at both
> depths), and wrote that `v4_kernel.rs`'s fp4 twin "covers the remainder on gate/up only — the
> same partial gap". **Then I read that test's comment, which explains the choice: *"5 trips =
> unrolled body + remainder at unroll 2 AND at unroll 4; 4 = clean groups at both."*** The pair
> is deliberate and covers TWO different cases. Making both dims leave a remainder tests the
> epilogue twice and stops testing the clean case at all — and the clean case is the one every
> engine dimension actually runs. **Reverted to 1280/1024.** I criticised a design as a gap
> without reading the paragraph that justified it; the code won, as it should.

**NOTHING machine-checks the trip counts**, here or in the fp4 twin: the int4 launcher guards
`hidden`/`inter` against `I4_GROUP` (128), not against `WAVE * 8` (256), so a conforming dim
like 1152 would launch fine at a different count, and a changed `WAVE` breaks them outright.

**The non-vacuity proof needs no GPU, and that is the point.** The tolerance half injects
M11's `n7`-zeroed-when-`base != 0` into the same quantized bytes — setting the stored nibble to
8, which decodes to exactly 0.0 since `nib()` returns `nibble - 8` — and asserts the
disagreement exceeds `err_tol`'s `1e-3·max + 1e-3`. **Measured at 1280/1024: `err = 5.059e4` against `tol = 1.055e2`, a factor of 480.** And it is **demonstrated RED**: neutering the injection so
the two oracles are identical drives `err = 0.000e0` and the assert fires with its own message.
Green and red both witnessed, on the host, in 0.3 s.

**What this does NOT prove**, stated because the distinction is the whole value: it shows the
*tolerance* can see the defect, not that the *GPU test* is wired up to see it. That needs a
deliberately broken kernel and a device, and is the one piece of G4 still outstanding. The
defect is aimed at ARITHMETIC, never at fold order — a reassociation is invisible to any
tolerance test by construction, which is the fingerprint's job and which the `glmi4` round's X
arm already discharged.

The duplication gate rejected the first draft of this test (3 clones, then 5). Factoring
produced `I4Case`, `i4_reference`, `gpu_i4_moe` and `i4_launch_drain` — one launch path for
every int4 MoE test, so a descriptor-layout change cannot be fixed in one test and left stale
in another. That is the gate doing its job, recorded because "duplication is a build error"
reads like bureaucracy until it prevents exactly this.

## G5 — the engine A/B, BANDS REGISTERED 2026-08-09 BEFORE the device

**The arithmetic, stated so the band cannot be walked back afterwards.** GLM's MoE compute is
what C2 speeds up, and only part of it. Per token: 75 MoE layers x (top-8 routed + 1 shared),
of which ~67.7% are resident hits computing on the compute stream and the rest arrive as
misses. C2's measured serial gain is **+16.4% at R=2** — which is the shipping width, since
MTP is on by default — and **+20.1% at `e_count = 1`**, the low end of the engine's run
lengths.

**Three discounts, each measured or registered elsewhere, applied in order:**

1. **Only the loop scales.** M11b found 23.07 of 24.25 ms was the loop and the rest launch and
   stream mechanics. Take ~95%.
2. **M9/M11b's transfer discount.** M11b measured **55-69%** of an equivalent fp4 kernel saving
   reaching the wall. That is the closest analogue this project has and it is not favourable.
3. **This plan's own point 2, UNTESTED.** "A loop that is memory-latency-bound in an idle
   microbench may be bound by something else entirely when the fetch path is saturating the
   same controllers." GLM runs 180.4 miss/token against V4's 4.96, so the fetch stream is
   vastly busier here than in the round M11b's transfer figure came from. **There is no
   measurement of this term in either direction.**

**Registered bands for `moe` ms/token** (the bucket, not the wall — it is the narrowest
instrument that can see this):

| outcome | `moe` delta | reading |
|---|---|---|
| **expected** | **−3 to −12%** | the transfer discount holds and point 2 is mild |
| plausible | 0 to −3% | point 2 bites; the kernel win is real and mostly absorbed |
| **registered as a genuine possibility** | **~0, indistinguishable** | the compute this speeds up is not on the critical path at 67.7% hit — **this outcome does NOT retract the kernel result** |
| kill | **any `moe` INCREASE beyond noise** | something other than the loop changed; do not ship |

**And the wall band is deliberately wider than the `moe` band: −0 to −8 ms/token, with "no
resolvable change" inside it.** I would rather register a band that admits the wall may not
move than one that has to be walked back. **A null wall result is a publishable outcome here
and is not a failure of the kernel change** — the ranking rule this repo already carries says
a perf win counts even when it hides behind another bottleneck, and at 67.7% hit and ~260 ms
of fetch latency per token, hiding is the expected case.

**Protocol, fixed now:** `--mode int4` (never hybrid — residency selects the arithmetic there,
INV-1 exception), `--max-mem` and `--cache-policy` held fixed and identical across arms, the
218-token prompt recorded in `benchmarks.md`, **byte-identity of the reply checked BEFORE any
span is read** (a differing reply voids the comparison whatever the timings say), counterbalanced
arm order with n = 2 per side, `flock` with the witness either side of every arm and any arm
with a live foreign holder discarded. Both binaries built before the device is requested and
nothing rebuilt between arms.

> **LEANED 2026-08-10 at the coordinator's instruction, and the trade is stated rather than
> hidden: n = 1 per side at `-bench 128`, ~65 s/arm at GLM's ~2 tok/s.** Byte-identity is the
> gate that matters here — it is binary at any n — and the span movement is reported against
> the registered bands with the caveat that **n = 1 cannot separate a small span delta from
> run-to-run variance** (this artifact's decode carries a known OPEN run-to-run divergence,
> `benchmarks.md` "Long-run divergence", so a single pair's timing delta is indicative only;
> the moe-band verdict at n = 1 is "moved / did not move / inconclusive", never a point
> estimate with authority). Order still counterbalanced (S,U then U,S would need n = 2 — at
> n = 1 the two arms simply run once each, order recorded). Everything else above stands.

## What G3 and G4 still need before anything ships

*[UPDATED 2026-08-10 — the goal is now MERGE, per the coordinator. State of each item:]*

C2 qualifies on the decision rule (fingerprint-identical AND ≥ +10%). **`#pragma unroll 4` is
now APPLIED on this branch** (`kernels/common.hpp::dot_i4_wave_r`, comment carries the
measurements; in-tree ISA re-read reproduces the C2 arm exactly — VGPR 88/123/83/95, occupancy
16/10/16/16, zero spill). Between it and MERGE-READY:

- **G4's multi-trip test EXISTS and is host-proven; its device half is QUEUED.** Fixture at
  1280/1024, saturation root-caused and double-guarded (see the dated notes under G3/G4). The
  next device grant must show: **green on stock**, **red on an injected `n7` break** (scratch
  tree), **green on the unrolled kernel**, plus the full int4 suite.
- **the G5 engine A/B on `--mode int4`**, protocol and bands registered above, leaned to
  n = 1 / 128 tokens per the dated note.

## G1 — the probe as staged (SUPERSEDED 2026-08-09 by the measurement above; kept for the staging argument)

### The probe as built

`examples/dot_bench.rs` gains a `glmi4` section: `run_glm_i4(name, hidden, inter, e_count,
ranges, nrow)`, five rows —

| row | e_count | ranges | working set | why |
|---|---:|---:|---:|---|
| `glm i4 R1` | 6 | 9 | 1.083 GB | engine condition, 33× the 32 MB MALL |
| `glm i4 R1 mall-ctl` | 6 | 1 | 120 MB | the naive harness, MEASURED not argued |
| `glm i4 R2` | 6 | 9 | 1.083 GB | the width fp4 structurally cannot run |
| `glm i4 R2 mall-ctl` | 6 | 1 | 120 MB | |
| `glm i4 R1 e1` | 1 | 50 | 1.003 GB | what launch size costs — see below |
| `glm i4 R1 e1 mall-ctl` | 1 | 1 | 20 MB | *added 2026-08-09*: the `e1` row's own control |

- **GB/s is WEIGHT bytes over time at both widths**, because one read of the weight row serves
  both token rows. So R = 1 and R = 2 are not competing arms; only depth-vs-depth *within* a
  width is a comparison.
- **The `e1` row exists because GLM's dispatch is not V4's, and it is GRADED, not decoration.**
  `gpu.rs` batches each *run of consecutive resident selections* among the 9 descriptors
  (top-8 + shared) and launches every miss singly on the miss stream, so real `e_count` is 1–3,
  not "the layer's residents in one call". So this row is the one measured at the launch size
  the engine actually issues, and **its A-vs-C1/C2 delta is part of the read**: a win that
  appears only at `e_count = 6` and not at 1 is a win the shipping dispatch would not collect,
  and that outcome changes the G4 decision rather than being a footnote. It also measures what
  launch size costs instead of asserting the grid is saturated regardless (at `e_count = 1`,
  gate/up is already 2048 waves and down 6144, against ~1280 machine slots).
- **The cross-arm gate has content, and it gates BOTH token rows.** `run_glm_i4` returns
  `(row 0, row 1 if R = 2)`. Row 0's arithmetic is independent of `nrow` (`a0[t]` accumulates
  per `t`, no cross-row term; `moe_down_i4_impl` reads `he = h_in + e*R*inter`, so `t = 0` is
  always the same slice) and of `ranges`, so all four `e_count = 6` rows must return the same
  row-0 value. **Row 1 is gated too, and that was a G1-review finding, not the original
  design:** row 0 is bit-identical at `nrow = 1` by construction, so a row-0-only gate would
  sail past an unroll that broke `vt = v + t * v_stride` for `t = 1` past the first trip — and
  R = 2 is the width this whole stretch exists to price.
  `tests/kernel.rs::batched_rows_are_bit_identical_to_single_rows` already covers row 0 against
  `nrow = 1`; nothing covered row 1 at depth. The `e1` row is deliberately excluded from both:
  it drains one expert, not six.
- **The band banner prints only on the `e_count = 6` rows.** Same review: gating it on
  `ranges > 1` alone would print `KILL ≤ 90` over the `e1` row, where M8's 66 GB/s at
  512–1024 waves makes a sub-90 reading plausible *and correct*, and an operator obeying it
  would discard a valid row.
- **Every ≥ 1 GB rotating row has a single-range control beside it.** *The `e1` row shipped
  without one and it was added 2026-08-09 at the coordinator's prompt* — the rule is not "the
  section has a control", it is "each rotating row has one", and a rotating row whose control is
  missing is exactly the confound the other two convert into a number. At GLM's shape a
  6-expert range is **120 MB**, not M11's ~80 MB: one int4 expert is 20.05 MB against fp4's
  13.37, so there is no way to make the `e_count = 6` control 80 MB. What matters is the
  property — one range, replayed, against a 32 MB MALL — and 120 MB clears it by 3.8×. The
  `e1` control is **20 MB, i.e. BELOW the MALL**, which makes it the strongest cache-served
  reading available at this shape: an upper bound on what the MALL can do for this kernel
  rather than merely a naive-harness comparison.
- **No ms/token is derived from any row**, unlike `run_v4_res`. GLM's wall cannot resolve this,
  and a derived engine-unit line is the single most liftable thing in the output.
- **Probe A's band, registered here before any device time: 130–175 GB/s. KILL ≥ 200 or ≤ 90.**
  Floor: the `.vq3` MoE batch — a random codebook gather — measured 109.4 GB/s
  (`benchmarks.md` "DSA indexer round: `examples/indexer_bench`", the per-layer rows; that row
  may itself be partly cache-served, which only makes it a weaker floor), and a
  sequential int4 read must beat it. Ceiling: M11's fp4 ballast arms, decode and every FMA
  removed, topped out at 200.20 GB/s on this box, so a stock *decoding* loop above that is
  reading cache. **This band is a SANITY band, not a reproduction gate**, and that is a real
  weakness against M11: V4 had a booked 24.3 ms serial `res` span for probe A to reproduce,
  and there is no booked GLM int4 expert-compute span to reproduce. The `mall-ctl` row is what
  actually makes the number trustworthy here.

## The probe patches, verbatim (M11's precedent: the engine never builds these)

Every arm is the SAME `examples/dot_bench.rs glmi4` binary with **one** patch to
`kernels/common.hpp`, scoped strictly inside `dot_i4_wave_r`. The other two dword loops in
that header — `dot_i4_wave` immediately below it and `dot_f4_wave_r` further down — have
byte-identical-looking bodies, so **an edit that lands on the wrong one changes a different
kernel silently**; the applying script asserts the header holds exactly three such loops and
that the one it patches is inside `dot_i4_wave_r`. Nothing here is committed to `kernels/`,
which is also why there is no second copy of this decode loop in the tree to drift.

**R = 1 and R = 2 are rows of one binary, not separate builds** — the `glmi4` section runs
both widths in one process — so the six unroll arms cost three builds, and the round is five binaries once B1 and X are counted.

- **A — stock.** The worktree at HEAD, unpatched.
- **G — the AS1 typing alone, no unroll.** *Added 2026-08-09 by the coordinator, and it changes
  the round's design rather than merely extending it — see "why G is insurance" below.*
  `dot_i4_wave_r`'s `row`/`scalerow` become `address_space(1)` pointers (`gi4u8p`/`gi4f32p`,
  declared just above it — `gu8p` itself is declared ~115 lines later, inside the fp4 block),
  the `rw` dword reinterpret carries the address space too, and **`moe.hip`'s three call sites
  cast**, because the asymmetry is at the CALL SITE and not in the struct: `ExpertDescI4`'s
  fields stay generic pointers exactly as `ExpertDescF4`'s do. This is the ONLY arm that
  patches two files, and the restore covers both.
  **Bit-neutral by construction** — an address space is a property of the pointer, not of the
  value loaded through it; no arithmetic, no ordering, no operand changes. So G's fingerprint
  must equal A's, and a disagreement would mean the patch did something it was not supposed to.
  **Measured statically before the device:** VGPR 39/49/32/42 and occupancy 16 — *identical to
  A* — and 68/82 instructions, *identical to A*. What moves is exactly the intended thing:
  the loop's 2 `flat_load_b32` become 2 `global_load_b32`, **`flat_load_b32` goes 18 → 0 across
  the whole device image**, and the body's entry wait goes `vmcnt(3) lgkmcnt(0)` → **`vmcnt(3)`,
  pure `vmcnt`, matching the fp4 loop.**
- **GC2 — the AS1 typing PLUS `unroll 4`.** G's patch with the same `#pragma unroll 4` C2 uses.
  VGPR 88/123/83/95 at occupancy 16/10/16/16 — identical to C2, so the addressing costs no
  registers at depth either.
- **B1 — the ballast.** In `dot_i4_wave_r`'s dword fast path, delete the eight `nib(w, k)`
  decodes and replace the two accumulate lines with
  `a0[t] += x0.x + x0.y + x0.z + x0.w + (float)w;` and
  `a1[t] += x1.x + x1.y + x1.z + x1.w + s;`, leaving the `w`, `s` and two `float4` loads
  exactly where they were. Every loaded value is still consumed into `a0`/`a1`, which reach
  `out` → `h` → the fixed-point accumulator → the drained residual the harness reads back, so
  no load is dead.
  **Confirmed in the emitted ISA before this was recorded, not argued** — a ballast the
  compiler deletes measures nothing: `moe_gateup_i4`'s loop keeps **2 `flat_load_b32` + 2
  `global_load_b128`** (R = 2: + 2 more `b128`) and still drains to `vmcnt(0)` in body, at
  **41 instructions against stock's 68** (R = 2: 54 against 82), VGPR 34/44/27/37, occupancy 16
  everywhere. **Its fingerprint will read `DEGENERATE`** — the raw dwords drive `moe_fixed` past
  its ±2^14 clamp so the residual is constant — and that marker is the confirmation the
  arithmetic is meaningless as designed, not a failure. **B1's reading is its RATE.**
- **C1 / C2 — the candidate levers.** `#pragma unroll 2` (C1) or `#pragma unroll 4` (C2)
  inserted on the line immediately above
  `        for (; base + WAVE * 8 <= dim; base += WAVE * 8) {` **in `dot_i4_wave_r`** — the
  FIRST of the three occurrences of that line in `common.hpp`. Nothing else changes.
- **X — the fingerprint's positive control.** Stock plus an even/odd split of the serial fold
  over `base`: declare `b0[R]`/`b1[R]` beside `a0`/`a1` and zero them in the same loop, carry
  an `int xit = 0` incremented in the fast-path loop's third clause, hoist the two products
  into `p0`/`p1`, accumulate `if (xit & 1) { b0[t] += p0; b1[t] += p1; } else { a0[t] += p0;
  a1[t] += p1; }`, and close with
  `out[t] = wave_sum((a0[t] + b0[t]) + (a1[t] + b1[t]))`. The scalar tail is untouched.
  **A real reassociation and deliberately NOT a schedule change** — verified in the ISA before
  it was recorded: the loop keeps the same 2 `flat_load_b32` + 2 (R=1) or 4 (R=2)
  `global_load_b128` and the same in-body countdown to `vmcnt(0)`, at VGPR 41/47/34/42 and
  occupancy 16 everywhere, against stock's 39/49/32/42 and 16. It is 75 instructions against
  stock's 68 (112 against 82 at R = 2), so **X is expected to be slightly SLOWER than A** —
  M11's X came in at −1.9% — and its rate is not the reading. **X is a throwaway: never
  committed, and it must FAIL the bit-identity check that C1 and C2 pass.** A gate only ever
  seen green is not a gate.

## The staged GPU round (SUPERSEDED 2026-08-09 — SEVEN arms ran, not five)

> **Superseded 2026-08-09.** The round as run was **seven** arms — `A B1 C1 C2 X G GC2`, pass 2
> reversed — after the coordinator added G and GC2 to make the design a 2x2 of addressing x
> unroll. The five-arm staging below is kept for its argument (build-before-device, nothing
> built between arms, witness either side) which the seven-arm round followed unchanged.

### The five-arm staging, as written

All five built BEFORE the device is requested, from this one worktree, with
`CARGO_TARGET_DIR=<worktree>/target-wt` (**never `/tmp`**: it is a 63 GB tmpfs in RAM).
Each build's `target-wt/release/examples/dot_bench` is copied to `target-wt/arms/dot_bench_<arm>`
immediately after it links, and `kernels/common.hpp` is restored with
`git checkout -- kernels/common.hpp` (**never `sed`**) between patches. Then the device is
asked for once and nothing is rebuilt until every arm has run.

| arm | reads |
|---|---|
| A | the sanity band (130–175 GB/s, KILL ≥200/≤90) and what the 120 MB `mall-ctl` rows do |
| B1 | **the ballast** — decode and every FMA removed at constant schedule; prints `DEGENERATE` |
| C1 | the candidate lever at depth 2, both widths |
| C2 | the second rung at depth 4 — R = 2 costs 10 waves/SIMD there, R = 1 costs nothing |
| X | the bit-identity gate's RED, without which "C1 is byte-identical" is a claim from an instrument never shown able to say otherwise |

> **B1 ADDED 2026-08-09, before the device, by the G1 code-quality review — which caught that
> the stretch's own kill condition could not be evaluated by the round that was staged.** The
> kill reads "if **the ballast-vs-real gap** that M11 measured at 12.8% is small **here**, the
> drain is not the limiter and the stretch closes as a negative", while §G2 said "no ballast
> arm is planned — B1 already answered this on the identical loop shape". The gap *here* is
> measurable only with a ballast, so as staged the kill had no input and could never fire. That
> is the precise shape this repo has burned itself on before, so the arm goes back in rather
> than the kill being quietly softened.
>
> **B1 for int4**, in `dot_i4_wave_r`'s dword fast path: replace everything from
> `unsigned int w = rw[col >> 3];` through the closing brace of the `for (int t = 0; t < R; ++t)`
> block with a body that keeps all four (six at R = 2) loads and consumes every loaded value
> into `a0`/`a1` additively — `a0[t] += x0.x + x0.y + x0.z + x0.w + (float)w;`
> `a1[t] += x1.x + x1.y + x1.z + x1.w + s;` — dropping the nibble decode and every FMA. **The
> loads must be confirmed surviving in the emitted ISA before this arm is believed**, exactly as
> M11 discharged it statically; a ballast the compiler deletes measures nothing. Its fingerprint
> will read `DEGENERATE` (raw dwords drive `moe_fixed` past its ±2^14 clamp, so the residual is
> a constant) and that marker is the confirmation, not a failure. Its reading is its RATE.
>
> Two arms of the drain question now have inputs: **B1 vs A** is the decode's cost at matched
> (un-unrolled) schedule — the 12.8% the kill is written against — and **C1/C2 vs A** is what
> the pragma buys. If M11's pattern repeats, B1 lands modestly above A and C2 lands near B1's
> unrolled ceiling; if B1 ≈ A *and* C1/C2 ≈ A, the loop is at its ceiling and the stretch closes
> negative with evidence rather than by assumption.

**ARM ORDER IS COUNTERBALANCED, not fixed.** *Added 2026-08-09 at the coordinator's
instruction, and the precedent is M11b's own: it ran S, C1, C2 in the same order every pass and
so aliased arm with position, which its untouched-span controls only just rescued.* Two passes
over all five arms, the second in **reverse order** — pass 1 `A, B1, C1, C2, X`, pass 2
`X, C2, C1, B1, A`. Position effects (page-cache warmth, thermal drift, a tenant arriving
mid-session) therefore enter the two passes with opposite sign and cancel in the per-arm mean,
instead of being confounded with the arm. **The order actually used goes in the record**, per
pass, and any arm whose two passes disagree by more than the round's own spread is reported as
unreplicated rather than averaged.

`flock /var/run/sys-gpu.lock -c '<arm-binary> glmi4'` per arm, exit code checked explicitly
(`flock -w` exits 1 silently on timeout, and an empty log is not a failing test). Witness —
KFD holders resolved through `/proc/<pid>/comm`, `mem_info_gtt_used`, and
`curl -s http://10.42.2.44:8080/running` — sampled either side of every arm; **any arm with a
non-empty witness is DISCARDED, not explained.** No engine decode this round: nothing has
earned an A/B yet.

### Why G is insurance, and the three outcomes registered BEFORE the device

**The round as first staged could have returned a false negative on its own central question.**
If the `flat_load`/`lgkmcnt` coupling is what stops this loop keeping work in flight, C1 and C2
could measure flat *because of the addressing*, and the honest reading would be "the unroll does
not transfer to int4" — which would be wrong. With A, C1, C2, G, GC2 the design is a
**2 × 2 (addressing × unroll) plus the depth ladder**, so the interaction is measured, not
inferred.

Scored, not narrated — the report says which of these fired:

1. **G alone ≥ +10%** ⇒ the addressing is a real independent lever, ranking on its own merits.
2. **GC2 super-additive over C2 and G** ⇒ the coupling hypothesis is **confirmed** and the
   unroll's value was being masked by the addressing.
3. **C1/C2 flat but GC2 not flat** ⇒ the false negative G was bought to insure against, and it
   becomes **the round's headline**.
4. **Any arm slower than stock by > 3%** ⇒ the registered kill; report and stop.

B1 and X stay: B1 is the only arm that can evaluate this stretch's kill condition, and X is the
fingerprint's demonstrated red.

**The decision rule, registered before the device.** A variant earns an engine A/B only if it
is BOTH fingerprint-identical to A on token row 0 AND ≥ +10% serial rate at R = 1 *or* R = 2.
Below +10%, record it and stop. If it wins at only one width, that is a finding about R, not a
tie-break — say which. **Report the `e1` row's delta beside them**: a lever that pays only at
`e_count = 6` is a lever the engine's short resident runs would not collect.

## Kill condition for the stretch — MIS-SPECIFIED, do not reuse as written

> **MIS-SPECIFIED, 2026-08-09, established by the round this kill was written to govern.** The
> ballast-vs-real gap IS small here (+0.5% / −1.1% / −1.0%) and `unroll 4` still won +12.6% to
> +20.1%. The kill's `i.e.` asserts an equivalence that is false by construction of B1: the
> ballast keeps every load and the identical drain schedule and removes only the decode and the
> FMAs, so B1 ≈ A isolates "not ALU-bound" and says nothing about latency exposure. A loop bound
> by insufficient memory-level parallelism gives B1 ≈ A *and* a large unroll gain — exactly what
> happened. Firing this kill would have closed a stretch that had just found its lever. See §G1.
> The text is kept unchanged below because what a registration got wrong is part of the record.
>
> **What the correct condition would have been**, so the next stretch inherits a rule and not
> just a warning: *"If a matched-depth ballast is within a few percent of the corresponding
> unrolled real kernel — B_D ≈ C_D at the same depth D — then the loop at that depth is
> essentially memory-bound and further DECODE compression cannot pay. And if no unrolled arm
> beats stock by the acceptance bar, the drain was not the limiter."* Two arms, two different
> questions. M11 got this right by accident: its B3 (ballast at depth 4) existed only because
> the first reading compared C2 against an unmatched-depth B2, and that arm is what let it say
> "C2 runs at 97.5% of its matched-depth no-decode ceiling". **The un-unrolled ballast alone
> (B1 vs A) answers neither question** — it isolates "not ALU-bound at a schedule that is
> already latency-bound", which is nearly a tautology and is why it read +0.5% here.

If the loop is already at its memory-bound ceiling at R = 2 — i.e. the ballast-vs-real gap
that M11 measured at 12.8% is small here — then the drain is not the limiter on this path and
the stretch closes as a negative with the ISA read as its evidence. **A well-evidenced
negative is a complete result**; the fp4 win does not oblige this one to exist.
