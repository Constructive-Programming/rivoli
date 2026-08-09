---
scope: glm
status: live
verdict: OPEN, G2 done 2026-08-09 (static, no GPU); the G1 probe is written and unmeasured. The ISA confirms the shape: `dot_i4_wave_r`'s dword loop issues 4 (R=1) or 6 (R=2) vector loads and waits every one of them down IN BODY — one iteration in flight, the same gap M11 measured +17.7%/+33.2% on `dot_f4_wave_r` (V4, R=1, byte-identical, 9.38 -> 9.61 tok/s). The fp4 number still must NOT be copied across, but THE REASON THIS PLAN GAVE FOR THAT IS REFUTED: R = 2 does not collapse occupancy. Depth 2 holds 16 waves/SIMD on all four int4 kernels, depth 4's worst cell is 10 — exactly where fp4's WINNING arm sat — and there is zero spill and zero scratch across all twelve cells, so the registered >=6-wave kill does not fire and all six arms are measurable. What the ISA does suggest is a different reason to doubt depth 4 at R = 2: it stalls at `vmcnt(4)` on ENTRY to the body despite 24 loads in flight, which is registered here as a prediction rather than kept for a post-hoc explanation. A SECOND gap this plan did not register turned up as well: the int4 loop loads its weight dword and group scale as `flat_load_b32` where the fp4 loop uses `global_load_*`, because `ExpertDescI4` never got the AS1 (`gu8p`) typing M3c gave the fp4 path — so the body's ENTRY wait is `lgkmcnt(0)`-coupled where every fp4 wait is pure `vmcnt`, an independent reason it cannot keep work in flight and one an unroll may not remove. Registered as candidate arm G, not taken. The wall cannot settle any of this — GLM decodes 2.07 tok/s at 67.7% expert hit and 180.4 miss/token x 1.44 ms/miss, so roughly half the token is fetch exposure — and the deliverable is therefore the KERNEL RATE and a shippability decision, not a wall delta.
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

## G1 — the probe, written and unmeasured (no GPU yet)

`examples/dot_bench.rs` gains a `glmi4` section: `run_glm_i4(name, hidden, inter, e_count,
ranges, nrow)`, five rows —

| row | e_count | ranges | working set | why |
|---|---:|---:|---:|---|
| `glm i4 R1` | 6 | 9 | 1.083 GB | engine condition, 33× the 32 MB MALL |
| `glm i4 R1 mall-ctl` | 6 | 1 | 120 MB | the naive harness, MEASURED not argued |
| `glm i4 R2` | 6 | 9 | 1.083 GB | the width fp4 structurally cannot run |
| `glm i4 R2 mall-ctl` | 6 | 1 | 120 MB | |
| `glm i4 R1 e1` | 1 | 50 | 1.003 GB | what launch size costs — see below |

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

## The staged GPU round (five binaries, one session, nothing built between arms)

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

`flock /var/run/sys-gpu.lock -c '<arm-binary> glmi4'` per arm, exit code checked explicitly
(`flock -w` exits 1 silently on timeout, and an empty log is not a failing test). Witness —
KFD holders resolved through `/proc/<pid>/comm`, `mem_info_gtt_used`, and
`curl -s http://10.42.2.44:8080/running` — sampled either side of every arm; **any arm with a
non-empty witness is DISCARDED, not explained.** No engine decode this round: nothing has
earned an A/B yet.

**The decision rule, registered before the device.** A variant earns an engine A/B only if it
is BOTH fingerprint-identical to A on token row 0 AND ≥ +10% serial rate at R = 1 *or* R = 2.
Below +10%, record it and stop. If it wins at only one width, that is a finding about R, not a
tie-break — say which. **Report the `e1` row's delta beside them**: a lever that pays only at
`e_count = 6` is a lever the engine's short resident runs would not collect.

## Kill condition for the stretch

If the loop is already at its memory-bound ceiling at R = 2 — i.e. the ballast-vs-real gap
that M11 measured at 12.8% is small here — then the drain is not the limiter on this path and
the stretch closes as a negative with the ISA read as its evidence. **A well-evidenced
negative is a complete result**; the fp4 win does not oblige this one to exist.
