---
scope: engine
status: data
verdict: The measured record — one verdict per round, not a journal. Carries what cannot be re-derived without a device: the canonical 218-token prompt, the byte-identity md5s and kernel fingerprints, the recorded command forms, and the RETRACTION. Per-arm rows are in git history at `e8526bd^`; the arguments are in `investigations/`. Live state at the top.
---

# Benchmarks

> ## STATE — the numbers that are true today
>
> **This file was compacted 2026-08-10** from an append-only journal (4,070 lines) down to the
> verdict of each round. It is **no longer append-only**: record a new result as a short
> section, and when one is superseded, replace it and say so. The arguments live in
> `investigations/`; the full journal lives in git history before `HEAD`.
>
> **Where a section states a verdict but not the rows behind it, the rows are in git history
> at `e8526bd^`** — per-arm tables were what made this file 4,070 lines. A verdict here is
> citable; a row you need to re-read is a `git show` away. What is NOT recoverable that way
> is anything needing a device, so fingerprints, byte-identity hashes and recorded command
> forms stay in the text.
>
> | | |
> |---|---|
> | quality ladder | int4 **5.120** > hybrid **5.189** (default) > int3-vq **5.275** |
> | GLM-5.2, 512 tok | **2.07 tok/s** (hybrid, dsa, MTP gated) |
> | V4-Flash, 512 tok | **9.10-9.17 tok/s** (`.f4`, MTP structurally off) |
> | layer-major prefill | **2.15x**, byte-identical, default since 2026-08-03 |
> | speculative decode | **1.108x** at `--mtp-min-conf 0.8`; ungated it is 0.93-0.95x, a loss |
>
> **Three things that invalidate comparisons, and have each drawn blood:**
>
> 1. **The prompt framing changed 2026-08-01.** Free-running text is not comparable across that
>    date. `encode_chat` used to emit GLM-4's `<|role|>\n{content}` and end at
>    `<|assistant|>\n`; this checkpoint has **no separator after the role token** and ends at
>    `<|assistant|><think></think>`.
> 2. **Long runs are non-deterministic** — ~40% of 5k-token scores are silently wrong, and the
>    rate itself is unestablished (those runs had no contention witness). See "Long runs are
>    NON-DETERMINISTIC" and "Long-run divergence" below.
> 3. **Everything in the 512->10k matrix at 2048 tokens and above is RETRACTED** — it measured
>    degeneration, not throughput.
>
> **Never rank on `distinct` or longest-repeated-block**, and never `cargo build` between the
> two arms of an A/B (it evicts page cache; it moved `ms/miss` 1.36 -> 5.14 in one measured
> pair). Method lives in `how-to-measure.md`.

## The canonical 218-token prompt

Every V4 A/B from the head-to-head through M11b, and the GLM int4 engine A/B, ran this exact
text. It is recorded here because it exists nowhere else, and 12 places cite it: an arm that
does not use it verbatim is not comparable to any of them. Recovered 2026-08-07 from a run's
own startup tokenizer line after `"<prompt>"` turned out to be the only record — 218 tokens,
1065 chars, and the int4 run script now REFUSES to start unless the file reads exactly that.

```
You are the lead engineer for a distributed log-ingestion platform handling 4 TB/day across 12 regions. Design a backpressure-aware streaming pipeline that replaces our current at-least-once Kafka consumer fleet, which is suffering duplicate writes during regional failovers and 95th-percentile end-to-end latency spikes above 30 seconds. Your answer must: (1) propose a concrete architecture naming each component and the protocol between them; (2) explain how exactly-once semantics survive a mid-flight region loss, including the idempotency-key scheme and its storage cost at our volume; (3) give pseudocode for the consumer's flow-control loop, including how it detects and sheds load under downstream store degradation; (4) enumerate the top five failure modes of your own design and the monitoring signal that detects each one before customers do; and (5) lay out a three-phase migration plan from the current fleet with rollback criteria at each phase. Be specific about numbers: partition counts, batch sizes, timeout values, and the reasoning behind each.
```

```
rivoli /var/db/rivoli/glm52-vq3-full --max-mem 115 -bench 512 --prompt "<the prompt above>"
rivoli /var/db/rivoli/v4-f4-full                  -bench 512 --prompt "<the prompt above>"
```

## Storage: sequential ordering buys nothing at QD>=2

`probes/seq_vs_random.c`, btrfs Data RAID0 across two NVMe (**`nvme1n1p3` + `nvme0n1p2`**,
1.68 TiB, `ssd`, no compression), under the flock. Same 15.34 MB request size, 246 requests,
same layer file, arm order alternated; mean of 4 reps, GB/s:

| QD | rand | seq | delta |
|---|---|---|---|
| 1 | 12.39 | 12.91 | +4.2% |
| 2 | 13.96 | 13.88 | -0.6% |
| 16 | 14.76 | 14.54 | -1.5% |

At 15.34 MB the seek is already amortised. **These are the achieved rates to reason from — the
7.0 GB/s quoted elsewhere is a `dd iflag=direct` queue-depth-1 figure** and understates what
the io_uring path gets.

## Bugs found and fixed

- **fp8 block scale mis-applied at `block < 4`** — `common.hpp::fp8_dot_strided` read four fp8
  weights per lane as one dword and applied ONE scale to all four. Numerics, not perf; it sat
  under every fp8 block-scaled GEMV. Fixed.
- **First-failure build masking** — `build.rs` compiled shaders in sorted order and aborted on
  the first failure. Now compiles all and fails once with the whole list.

## `top-m` offline screen (CACHE_ROUTE, arXiv:2412.00099)

Offline `bin/replay` over three 512-token captured routing traces, one per mode — no engine
change, so free of the decode-trajectory confound. **`top-m` is RETIRED and removed from the
engine.** The screen — the (J, M) grid, its hit/swap columns and the powered cell below — is
the record of why.

### DECISION: `top-m` ships opt-in and UNCERTIFIED

The powered cell that decided the feature: `int3-vq`, **5,184 teacher-forced positions**,
`--max-mem 100`, shared baseline, one process per cell. Baseline (lru) **PPL 4.130637**, hit
72.25%.

| cell | PPL | dPPL% | 95% CI (nats) | hit% | swap% | verdict |
|---|---:|---:|---|---:|---:|---|
| J=4/M=9 | 4.15252 | +0.529% | [−0.00207, +0.01263] | **77.69%** (+5.44pp) | 5.79% | **INCONCLUSIVE** — interval contains zero |
| J=4/M=10 | 4.16786 | +0.901% | [+0.00077, +0.01717] | 81.47% (+9.22pp) | 9.80% | **COST ESTABLISHED, MAGNITUDE UNRESOLVED** |

**J=4/M=9 shipped**, and INCONCLUSIVE is not "small cost confirmed": the interval contains
zero, so no cost is established *and* it is not certified within the bar either. J=4/M=10 is
not ship-able — its lower bound clears zero. This is the origin of the four-verdict
vocabulary (PASS / FAIL / COST ESTABLISHED / INCONCLUSIVE) that `vulkan-port.md`'s staged
acceptance gate reuses.

## Per-kernel round: matched A/B, `examples/dot_bench`

Matched A/B on one instrument binary, three INTERLEAVED repeats (base/fix/base/fix/base/fix)
so drift shows as within-arm spread rather than as the effect. GLM dims from the manifest:
H=64, qk_head_dim=256. The interleaving discipline is the transferable part.

### Section tokens recorded rounds invoke

`examples/dot_bench` takes a section name, and rounds below are recorded by it. These tokens
are **frozen**: `moe`, `gemv`, `v4gemv`, `v4res`, `glmi4`, `mla`, `attend`, `tail`. `v4gemv`
and `glmi4` still carry model-derived names where the kernels they drive were renamed for
behaviour on 2026-08-09, deliberately — a recorded command that no longer runs cannot be
re-run to settle a question. `dot_bench.rs`'s `main` cites this section for that argument.

### A fingerprint is the only instrument that shows bit-identity

`assert_close` cannot tell a bit-identical restructure from a reassociating one — both pass,
and the margin print does not separate them. `examples/dot_bench` prints an FNV-1a hash of
each kernel's raw output bytes; the absorb restructure's claim rests on
**`0925c147afeea3fb`**, unchanged across 14 interleaved runs of both arms.

**It only works if the inputs VARY.** `run_fp8` and `run_mla` used constant `x`, `q` and
`clat` — correct for throughput, since traffic does not depend on values, but a constant
input leaves the output insensitive to summation order, so the fingerprint would have been
green for a change that reassociated. The instrument and the input generator are ONE
instrument; a fingerprint over degenerate data is a fingerprint of nothing.

## In-engine confirmation — the number a merge decision rests on

`-bench 256 --mode int3-vq --cache-policy lru --max-mem 100 --attn dense`, fixed prompt,
interleaved base/fix. **`-bench` is a fixed-token bench even though decode is greedy**: in
`int3-vq` the token count is the work, so the arms are comparable.

### Closed questions

- **`rmsnorm`'s `dim3(1)` launch is not a problem** — 7.7 us, 0.05% of the `tail` bucket. At
  hidden=6144 there is not enough work for the geometry to matter. Measured; do not re-flag it
  from the launch shape alone.
- **`mla_value` was not a healthy reference.** `mla_absorb`'s 99 GB/s was judged against
  "`mla_value`'s 254", but `mla_value` carried the same 64-bit divide, so the yardstick was
  depressed too. Post-fix 172.3 vs 310.3. **Check that a reference point is itself healthy
  before measuring against it.**

### ~~Open question: half of `tail` is in none of its kernels~~ — ANSWERED

Closed by the CLASS axis (`perf-roadmap.md`): the unattributed time is decode-loop **host CPU**,
not a hidden kernel — ~6 ms/tok of 6.2 ms host compute, itemised as kernel launch, tokio poll
(term deleted 2026-08-01 with tokio), `submit_layer` and `route_into`.

## Read the ISA before you book the device

**The GPU is the scarce resource; the compiler is not.** `hipcc` answers a large class of
kernel questions on the CPU in seconds with no queue, and answers some of them BETTER than a
bench because it gives the mechanism rather than a number. Four of five perf questions in one
round were settled this way. Prescribed in `how-to-measure.md`.

## Running these benches — detach anything multi-cell

**A GPU run longer than the harness's background-task lifetime must be detached into its own
process group**, or a task reap kills the engine with it. Invisible from the code; cost a cell
before it was understood.

**Verify detachment rather than assuming it** — "I ran setsid" and "it is actually detached"
are different claims, and only the second matters:

```sh
ps -o pid,ppid,pgid,cmd -C rivoli
# PID 2005651  PPID 2005649  PGID 2005649  -> own process group, not a harness child
```

If `PGID` equals the process's own `PID`, and `PPID` is not the harness shell, the run
survives a reap. `tests/ppl-sweep-powered.sh` cites this check.

## Measurement caveat

**Free-running greedy tok/s cannot rank modes.** A degenerate run routes to the same few
experts -> inflated hit% -> artificially FAST; the early int4 rows posted the highest tok/s
BECAUSE they degenerated. Gate on output quality first, then read the rate.

**And the distinct-token gate cannot be that quality gate.** 2026-07-27 a branch-gain sweep
tripled PPL (**73 -> 216**) while distinct-ratio held. It separates a crash from a completion,
nothing more — every cell of the retracted matrix passed it.

## DSA indexer round: `examples/indexer_bench`

Instrument for the NPU-offload gates (`investigations/npu-offload.md` M0/M1), gfx1151 sole
tenant, 2026-07-26. **The rig is deleted** (`77b5500:examples/indexer_bench.rs`) and its
GPU-span figure was refuted by 27% by the engine's own indexer buckets. Recorded as a
superseded instrument, not as a baseline — its rows, methodology and three measurement traps
are in git history at `e8526bd^`.

### Device top-k WIRED: three-arm in-engine A/B, 2026-07-27

`--attn dsa --mode hybrid --cache-policy lru --max-mem 115 -bench 128`, 2432-token prompt,
sole tenant, arms selected by `RIVOLI_TOPK` from ONE binary, interleaved. **The switch no
longer exists** — `device` shipped; `host`/`device-nosync`/`verify` were deleted once these
rows were recorded (`77b5500:src/gpu.rs` restores them). Rows recorded after 2026-07-30 carry
no `[topk=...]` tag and no `idx_host` term, which is what makes `route` incomparable with
older rows. Cited by `src/gpu.rs` and `investigations/npu-offload.md`.

**The evidence is the match count, not the exit status**, because an earlier revision of this
gate **exited 0 having compared zero layers** whenever the context stayed under `index_topk`.
The repaired gate was then confirmed able to fail: `RIVOLI_TOPK=verify … -bench 4` on the
default short prompt exits 1 with `compared 0 layers: the context never exceeded
index_topk=2048`. Re-run on the shipped binary after the comparison loop was rewritten in
review: **8,736 layers matched** at `-bench 32` (21 × [384 prefill + 32 decode]), all seven
runs byte-identical (564 chars, sha256 `778387fa557c4e9d…`), coherent prose. **Not
established:** one prompt, one context (nt ≈ 2496 mean), n = 2 per arm.

## RETRACTION: the 512->10k matrix's long-context results are invalid

**Everything in the 512->10k matrix at 2048 tokens and above measured DEGENERATION, not
throughput.** The headline — `int3-vq/streaming/2q` as "the only cell that gets FASTER with
context" — is an artifact. All 58 cells were reclassified from their logs with a structural
repetition check. **Cited by `README.md` and `artifact/tokenizer.rs` as "the retraction".**

## Benchmark matrix: mode x attn x cache-policy, 115 GiB, 512 -> 10k tokens

Bracket 44 -> 8 -> 4 -> 2 at 512/2048/4096/10000 tokens, `--max-mem 115`, one process per
cell, ~10 h. **Invalid at 2048+ — see the RETRACTION above.** The 512-token column stands.

## `--mode int4` vs int3-vq — the point estimate favours int4, the test cannot confirm it

2026-07-31, one binary, one session, `tests/ppl-corpus.txt` (762 teacher-forced tokens),
2q / `--max-mem 115` / dense. `--ppl` never enters `generate`, so speculation is not a
variable. **Point estimate favours int4; the test cannot confirm it** — 762 tokens is
underpowered. `tests/ppl-corpus-5000.txt` exists for this reason.

### int4 provenance — MEASURED, and it inverts hybrid's stated premise

The int4/hybrid numbers of that era used `.i4` re-derived from **vq3**, itself a lossy 3-bit
quantization, because colibri's own int4 was a mismatched per-row RTN (R≈0.96 against vq3,
scales 5–9% inflated) that decoded degenerately under the fp8 router. So the chain was
`fp8 → vq3 → int4`, and **the int4 set could not be better than the vq3 it came from, by
construction** — which is the whole reason `bin/fp8_to_i4` exists and `bin/vq3_to_i4` is
deleted.

> **SUPERSEDED 2026-07-27 — `docs/investigations/int4-scales.md`.** Two claims here measured
> false. The set rebuilt from fp8 is **strictly more accurate** and **8× worse end to end**
> (PPL 73.43 vs 5.28), so the deficit was never "the arithmetic of double quantization": the
> cause is **per-row scaling**, one scale per 6144 weights. The gs64/`pack_i4` fix this
> section recommended was right for the wrong reason, and `int4-scales.md` re-endorses it on
> the correct one.

## The MTP confidence gate — 1.108×, 2026-07-31

**1.108x via `--mtp-min-conf 0.8`** (2026-07-31), and this is the live verdict on the feature.
Calibration is prompt-invariant. Shared-GPU caveat: ran under the flock with another tenant
present, so pin-build times varied — the ratio is the claim, not the wall.

## Layer-major prefill (`--layer-major-prefill`) — prefill 2.15x, output byte-identical (2026-08-02)

**Prefill 2.15x, reads 159.56 -> 28.20 per token (the floor), output byte-identical, every
`--attn` mode.** Default since 2026-08-03; the flag is deleted and `--trace` falls back to
token-major. Decode pays a ONE-OFF ~2.7 s warm-up = 1.8% of the prefill saving; the
"1.55x slower decode" reading was a 13-pass artifact. Closing the sweep token-major was tried
and REVERTED as useless.

```
rivoli /var/db/rivoli/glm52-vq3-full -bench 16 --mode int3-vq   # the recorded form
```

## The Belady bound on residency: the cache-policy lever is spent at 115 GiB, live at 61 — 2026-08-02

**The cache-policy lever is SPENT at `--max-mem` 115 GiB and LIVE at 61.** The online policies
had only ever been ranked against each other, never against the offline optimum, so a 2Q that
beat LRU by 5 pp could equally have been 5 pp from perfect or 30. Cited by
`perf-roadmap.md`.

## Batch coalescing, alternative policies, and the 2Q kin/kout re-sweep (2026-08-02)

**Batching the MoE does NOT reduce bytes per token: 2 rows read 1.61x the experts, not 1x.**
The union of two rows' picks is larger than one row's. Break-even acceptance ~53%. This priced
`perf-roadmap.md` Path A, whose case assumed the opposite. The 2Q kin/kout re-sweep found no
better split; `cache.rs`'s default stands.

## The MLA HB sweep: HB 8→16 is 2.08× on the kernel and −3.2 ms/token — SHIPPED 2026-08-02

**HB 8 -> 16 is 2.08x on the kernel and -3.2 ms/token — SHIPPED 2026-08-02.** Run as the two-
parameter (HB x `MLA_MIN_TILES_PER_SPLIT`) sweep roadmap #5 asked for; only HB is live.
Interleaved, min-of-5, `examples/dot_bench attend`, four binaries built up front so no
`cargo build` ran between arms.

Fingerprints, because an HB sweep is bit-identical only while the split plan is unchanged —
this is what tells the cells apart, and re-deriving it costs a device slot:

| cell | µs | fp (HB) | fp (split) |
|---|---:|---|---|
| HB=8, MIN=4 *(was shipped)* | 226.5 | `6eb5576d…` | `91d2fa2a…` |
| HB=8, MIN=2 | 226.4 | `6eb5576d…` | `91d2fa2a…` |
| **HB=16, MIN=4 (shipped)** | **108.8** | `6eb5576d…` | `4c2cf2d9…` |
| HB=16, MIN=2 | 117.9 | `faf6f182…` | `4c2cf2d9…` |

MIN=2 at HB=16 is the cell that lands on a different `n_splits`, which is why its first
column moves while the shipped cell's does not.

## Long runs are NON-DETERMINISTIC: ~40% of 5k-token scores are silently wrong — 2026-08-02

**~40% of 5k-token scores are silently wrong (2026-08-02).** Confirms the "worse than the
crash" case: a NaN needs a slot that was never written, and on a warm pool the same race reads
a *stale* slot instead — plausible numbers, no crash. This is why `distinct` cannot be trusted
and why long-run PPL needs a repeat.

## Long-run divergence: a RACE, not residency — INV-1 exonerated, and the first witnessed pairs — 2026-08-05

**Still not root-caused** — the repo's #1 OPEN defect. Two whole classes are closed, and the
earlier measurement was wrong in a way worth keeping.

**The earlier numbers had no contention witness.** `flock /var/run/sys-gpu.lock` is
*advisory* and another agent on this box runs GPU tests without it (observed 2026-08-04).
The tell is general: the arms WITHOUT the extra probe ran SLOWER than the arms with it
(2485.6/2428.9 s vs 2247.3/2301.4 s), and a probe that adds work cannot make a run faster,
so the ordering being backwards is contention. A reproduction claim on that pair was
**retracted**. Every pair below carries a per-arm witness log — every PID holding `/dev/kfd`
or a render node, sampled every 20 s, minus the arm's own — and the rule was fixed before the
data: a non-empty witness means the arm is **discarded, not interpreted**.

> This puts a question mark over the `~40%` and `token 4042` / `17 nats` figures in the two
> sections above: those runs had no witness either. Not hereby wrong, but the *rate* is
> unestablished and a single unwitnessed pair should not be quoted as one.

`--mode int3-vq --no-mtp --attn dsa --max-mem 90`, `ppl-corpus-5000`, sole-tenant, witness
empty on both arms of both pairs:

| pair | arm 1 | arm 2 | first divergence (pos, layer) |
|---|---:|---:|---|
| A | 4.126882 | 4.175782 | (258, 62) |
| B | 4.848965 | 4.302220 | (265, 60) |

It reproduces on a clean machine, so the fault is real. Pair B's 0.55 PPL spread is the ~0.5
recorded originally; pair A's 0.049 is not. **The divergence POSITION moves between pairs**,
which rules out anything deterministic about token 4042 and is the signature of a race.

**INV-1 is EXONERATED by direct measurement** — but note the instrument is **not in this
tree**. `--checksum-route` / `--features corruption-probe` exists only in tag
`archive/belady-residency-bound` (`544fea7`), on the `9ffb468` base (2026-08-04) that predates
the `v4gpu.rs`→`f4gpu.rs` rename (`0f39cc4`) and the device-router deletion (`b8ff613`), so
re-running this needs a forward-port first — its `src/gpu.rs` hunks are the +211 lines to
re-place. What it did: hash, per MoE layer, the gate logits the router SAW and the
experts it PICKED — pure host-side, so no device traffic and no I/O during the run. Across
**388,875 records per arm** (5185 positions × 75 MoE layers), in both pairs: **rows where the
logits AGREE and the picks DIFFER — 0.** Where picks diverge the logits diverged first, at the
same row: routing faithfully reflecting an already-corrupted residual. The
`hit_pct`-tracks-output correlation that motivated the check is a symptom, not a cause, and
`--mode int3-vq` remains output-neutral to residency. **Localised to a timing race in layer
*L-1*'s MoE compute against layer *L*'s attention** — stated here rather than cross-referenced,
because the `architecture.md` §6 write-up went down with `perf/belady-residency-bound` and
§6 in this tree is the byte-arena pool.

## VQ_K=2048: the codebook shrink and the byte saving are the SAME item and they cancel — 2026-08-04

**The codebook shrink and the byte saving are the SAME item and they cancel.** `VQ_K` 4096 ->
2048 gives a 16 KiB fp16 codebook that fits L1 AND an 11-bit index, but 1.189x on the MoE
kernels costs +18.7% relFrob. Priced by two probes with no requant, because kernel speed does
not depend on index values. Live on `perf-roadmap.md` #2, needs a real dNLL gate.

## GLM-5.2 vs DeepSeek-V4-Flash — 512-token decode, one complex prompt (2026-08-07)

First head-to-head on the merged tree, release. `-bench 512`, sole tenant under the exclusive
flock with a per-minute GTT+KFD witness, both arms clean.

| | GLM-5.2 (hybrid, dsa, MTP gated) | V4-Flash (.f4, MTP off) |
|---|---:|---:|
| prefill, 218 tokens | 72.8 s (layer-major) | **20.12 s** |
| decode | 512 in 320.5 s = **2.07 tok/s** | 512 in 95.0 s = **5.389 tok/s** |
| expert hit (decode) | 67.7%, 180.4 miss/tok, 1.44 ms/miss | **~98.1%**, 4.96 miss/tok |
| output | 512/512, coherent, on-task | 512/512, coherent, on-task |

prompt md5 `bc71afa745d980be7d21860f70ad96aa`. **CORRECTED 2026-08-11:** this said *reply* md5.
It is the md5 of the 1065-byte canonical **prompt** (no trailing newline) — recomputed from the
prompt file 2026-08-11, it matches. So **no reply md5 was ever recorded for this pair**, and the
`output` row above is the only claim about what either model emitted. **CORRECTED 2026-08-07:** the V4 "17.0 miss/tok"
and "95.4% hit" originally printed here divided TOTAL misses (prefill included) by decode
tokens; decode-only is 4.96 miss/token, and "V4's ~0.23 GB/token" becomes ~0.07.

The GLM hybrid arm emitted two stray in-word `odesk` tokens in otherwise clean prose. n=1,
greedy, `--mode hybrid` — the mode whose arithmetic is residency-dependent (INV-1 exception).
Not filed from one run; A/B on a single-format mode if it recurs.

## V4 decode decomposition — route 78.9 / moe 71.9 / remainder 40.0 ms/token (2026-08-07)

route 78.9 / moe 71.9 / remainder 40.0 ms/token (2026-08-07). The instrumented decode M2 asked
for, flag-identical to the head-to-head. **Superseded through M11b — the ranked levers, every
band and every gate live in `investigations/v4-decode-decomposition.md`, which is the
narrative home.** The rounds below keep only their headline and their gate.

## V4 launch-geometry A/B — moe 70.6 → 54.7 ms/token, output byte-identical (2026-08-07)

moe 70.6 -> 54.7 ms/token, +9.6% tok/s, **output byte-identical** (M3b, launch geometry:
resident experts of a layer in ONE range launch).

## V4 route split — hcn 41.2 ms is the surprise: attn 53.2 / hcn 41.2 / cmp 9.1 / gate 3.2 of win 107.7 ms/token (2026-08-08)

attn 53.2 / hcn 41.2 / cmp 9.1 / gate 3.2 / win 107.7, resid 1.0 (M4). **hcn 41.2 against a
4-8 band was the surprise** — the engine's largest above-bytes excess at the time.

## V4 fp4 kernel-rate A/B — moe 54.9 → 49.6 ms/token, output byte-identical (2026-08-08)

moe 54.9 -> 49.6 ms/token, wall 165.3 = 6.048 tok/s, **byte-identical** (M3c: branchless
e2m1/e8m0 + global-AS descriptor loads, fp4 dot 195 -> 105 instr per 128 weight bytes). The
registered prediction (-4..-10, point -6) landed IN BAND at -5.3.

## V4 hcn A/B — hcn 40.7 → 5.6 ms/token, wall 168.2 → 130.7 = 7.652 tok/s, output byte-identical (2026-08-08)

hcn 40.7 -> 5.6, wall 168.2 -> 130.7 = **7.652 tok/s**, byte-identical (M5) — reply prefix
md5 `0ebcf62c20c6981b0ad7ca04ccfff270`, escape-decoding to the **1983-byte** reply prefix M4
established. Schedule-only:
`hc_pre` 256 -> 1024 threads = one wave per mix row, unroll-8. The width and unroll levers
MULTIPLY (24x loads in flight).

## V4 double split — attn = qkv 20.3 + attend 3.2 + oproj 29.6; moe attributed res 24.2 / miss 9.7 / shared 15.8 (overlapping, not addends); the fp8 GEMV fires in BOTH splits (2026-08-08)

attn = qkv 20.3 + attend 3.2 + oproj 29.6; moe res 24.2 / sync2 31.6 / shared 15.8 / miss 9.7
(M6). Control wall 130.6, both summing identities exact. The fp8-GEMV kill fired on qkv
(2.31x bytes) and oproj (1.99x).

## V4 fp8-MLP + shared-overlap A/B — wall 129.5 → 109.0 ms/token = 9.175 tok/s (+18.8%), output byte-identical; the wall band MISSED LOW, first bad-side miss of the series (2026-08-08)

wall 129.5 -> 109.0 ms/token = **9.175 tok/s** (+18.8%), byte-identical (M7: `#pragma unroll 8`
on the fp8 GEMV + the shared chain on its own stream). Fell 1.5 SHORT of the registered band —
the series' first bad-side miss.

## V4 M8 replicate + serial-rate round — every conditional replicates; wq_b streams 222 GB/s (87% of bus) and refutes the uniform-floor hypothesis; the stretch closes at 9.10–9.17 tok/s (2026-08-08)

Every conditional replicates; **wq_b streams 222 GB/s = 87% of the 256 bus at full grid while
wq_a/wkv sit at 66** — the excess is grid starvation, not per-wave cost. ~13.6 ms/token of
kernel headroom is measured but unreachable under byte-identity. **Closes the last mile as a
successful negative: 9.10-9.17 tok/s, and 10 tok/s is not reachable.**

## V4 M9 split-k — measured on branch `wt/v4-splitk`, kernel NOT merged (2026-08-08)

**BUILT and REJECTED — DO NOT ENABLE.** Kept unmerged on `wt/v4-splitk` (`697246c` kernel,
`6642312` record) because merging enables it flaglessly. 17/512 argmax flips, every one at a
near-tie; perf NULL (wall +0.8). The measured trade: real tie-breaking drift for zero speed.

## V4 M10 qkv width-fusion A/B — Δqkv −1.7 REPLICATED in band; the wall does not resolve a ±2 lever at this noise; byte-identical on all four arms (2026-08-09)

Δqkv **-1.7 REPLICATED in band**; the wall does not resolve at this noise (pairings -2.1/+1.6).
n=2 per side. Byte-identical on all four arms, reply md5
`75b19fcde806059b45c515259feb16d2`. B1's 106.5 = 9.394 tok/s is the fastest witnessed decode
of this benchmark, claimed no further.

## V4 M11 fp4 resident-kernel round — memory-level parallelism is the limiter; `unroll 4` runs at 97.5% of its matched no-decode ceiling, fingerprint-identical (2026-08-09)

**No engine decode ran** — `dot_bench v4res` microbench only. Memory-level parallelism is the
limiter and the lever is one line: `#pragma unroll 4` measures **195.27 vs 146.63 GB/s
(+33.2%)**, fingerprint-identical (the `v4res` fnv, NOT an engine reply md5). Issue rate DEAD.

## V4 M11b fp4-unroll engine A/B — byte-identical 9/9, Δwall −1.70/−2.53 in band, 77–85% of the MoE bracket reaches the wall, arm order confounded (2026-08-09)

**Byte-identical 9/9**, Δwall -1.70 (unroll 2) / -2.53 (unroll 4), both in band = 9.54/9.61
tok/s. 77-85% of the MoE bracket's saving reaches the wall, 55-69% of the kernel's. Arm order
was FIXED S,C1,C2 every pass, so **the wall separation cannot be attributed** — the next round
randomises rather than adding n.

## GLM int4 MoE unroll round — `dot_i4_wave_r`, seven arms, +16.4% at R=2 fingerprint-identical (2026-08-09)

`dot_i4_wave_r`, seven arms, two counterbalanced passes, nothing built between arms.
**+12.6% (R=1) / +16.4% (R=2) / +20.1% (e_count=1)** serial rate, fingerprint-identical.
The two adjacent negatives priced in the same round: AS1/`gu8p` typing ±0.6%, decode-removed
ballast +0.5% — the whole gap was memory-level parallelism.

**The gate is demonstrated in BOTH directions, and the red half is these hashes.** Every
shipping arm holds row-0 fnv `b2407d1121848fd5` — stock, unroll 2, unroll 4, AS1, AS1+unroll 4
alike. Arm **X**, deliberately reassociated, moves all three
(`924efb78f6743e16` / `01ff6e0362de32b4` / `7de5301cb611fd2e`) while staying non-degenerate:
proof the fingerprint can go red on a change `assert_close` would have passed.

## GLM int4-unroll engine A/B — byte-identical at --mode int4, 128 tokens, MTP active (2026-08-10)

**Byte-identical at `--mode int4`**, 128 tokens, MTP active: reply md5
`ba97d99d983f1641469d4d0ca6aaf086`, 143013 hit / 42197 miss, MTP 63/84, all identical between
arms. NO wall claim (n=1, fetch-dominated). The run script REFUSES to start unless the prompt
file reads exactly 1065 chars — added after a first attempt ran both arms on an EMPTY prompt.

## Rename merge gate — union suite 334/0, smoke matrix 15/15, red-proof 13/13 (2026-08-10)

**Union suite 334/0, smoke matrix 15/15, red-proof 13/13** (2026-08-10), on the union tree
(renames + int4 unroll). Suite was `--release`, union features, **all 35 targets** — which
counts the 7 `src/bin/*` a `--lib`/`--test` run never builds. Red-proof:
`RIVOLI_MAX_MEM=1` fails all 13 decode cells while both refusal cells keep their verdict, so
the classifier separates a broken engine from a documented refusal in both directions.

## K3 S1a SafeWriter gate — suite 324/0 on a fully-idle GPU (2026-08-10)

**324 passed / 0 failed** across all 26 binaries at the **dev profile** (`--features rocm`,
`--test-threads=1`, each suite under the flock). Host-side change (`SafeWriter` borrows
verbatim tensors, `write` became atomic), so this is a regression gate — no tok/s claimed.

Not comparable to the 334/0 above: that was `--release`, union, all 35 targets. Earlier
attempts were discarded under a witnessed foreign tenant (33.6-39.5 GiB GTT, 0 kfd —
`reference/gpu-lock.md`); cleared by draining `rh-anine`, GTT 39517 -> **17 MiB**.
