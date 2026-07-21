# rivoli benchmarks

Every number here was measured on **rh-anine** (AMD Ryzen AI MAX+ 395, Radeon
8060S gfx1151 40 CU, 128 GB LPDDR5X, btrfs RAID0 over 2 NVMe) against the local
GLM-5.2 int4 snapshot at `/var/db/llama-server/glm52-colibri-int4`. Reproduction
commands are given with each table.

**Read this before quoting any number below:**

- **`hit %` is not a residency metric.** It counts "needed no *demand* read at
  resolve time", which includes experts a prefetch had just streamed off disk.
  Use the `loaded / preloading / cold` split instead — only `loaded` is I/O-free.
  This distinction has invalidated conclusions twice; see *Retired claims*.
- **tok/s comparisons are only valid when the routed expert sequence is fixed.**
  Anything that perturbs f32 rounding changes the sampled tokens, which changes
  routing, which changes the hit rate, which changes `fetch`. Kernel changes do
  this; compare their `attn`/`mlp` ms buckets instead, which are hit-rate
  independent. Cache-config A/Bs (policy, depth, Kin/Kout) ARE safe — but only
  since commit `f060823`. Before it, a 2Q eviction bug made the cache silently
  change computed weights, so the workload itself varied with cache settings; see
  the retraction below.
- **Check the cache counters before believing any delta.** They are exact, so
  identical counters across runs prove the workload was identical and any wall-clock
  difference is environmental. Every wrong conclusion recorded here was caught this
  way, and one was hidden for hours by a `sort -u` that deduplicated the single
  differing record.
- Cache counters are **exact and reproducible** (the two `2q`+depth-1 runs below
  produced byte-identical `loaded`/`cold` counts). Wall-clock carries ~±3 %.

---

## Current defaults

`--cache-policy 2q --2q-kin 8 --2q-kout 20 --prefetch-depth 1`, prefetch on,
O_DIRECT cold reads, SQPOLL on both io_uring rings, split-KV attention,
12 GiB OS reserve capped at an 88 GiB total device budget.

```
rivoli /var/db/llama-server/glm52-colibri-int4 -bench 512 --pre-seed
```
```
device pool budget 88.0 GiB (free 100.0 GiB - 12 GiB OS reserve, capped at 88 GiB)
routed pool [2q]: 4415 slots (77.9 GiB)
PROFILE/tok: 577ms wall | fetch 397ms 69% (96 miss, 4.10ms/miss, submit 7ms + join 389ms)
           | attn 87ms 15% | mlp 76ms 13% | lmhead 4ms 1% | route 3ms | other 10ms
512 tokens in 300.1s - 1.71 tok/s | hit 83.7%
expert source: loaded 82.0% | preloading 1.7% | cold 16.3%
disk traffic: 112.8 expert reads/tok (2.13 GB/tok), 3.4% wasted
```

### RETRACTION — everything below the correctness fix was re-measured (2026-07-21)

An earlier revision of this file reported **2.12-2.33 tok/s at 91.8-92.2 %
residency**. Those numbers were produced by a build with a silent
weight-corruption bug (commit `f060823`): `TwoQ::get` reports a hit on an A1in
entry without moving it, so the pin's hits-before-misses ordering did not protect
it, and a same-layer miss could reassign its slot underneath a live descriptor.

The corruption perturbed the token stream into a **different, much easier
workload** — 10 994 unique experts instead of 13 259 — and the high hit rate was
real for that stream. It just was not the model's actual output. **The bug was
2Q-only**; LRU and ARC promote on `get` and were never affected, so their
historical numbers stand.

Retracted: the "2q + depth 1 sharp peak", the 91.8 %/92.2 % residency figures, the
first Kin/Kout sweep (run against a corrupted trace), and every tok/s number
derived from them. **Not** affected, because they move hit-rate-independent
buckets: the int4 GEMV vectorisation, split-KV attention, and the SQPOLL work.

Honest trajectory, all 512 tok, `--pre-seed`, correct decode:

| state | tok/s | ms/tok | loaded | GB/tok |
|---|---|---|---|---|
| measured baseline, start of 2026-07-20 | 0.90 | 1093 | — | 3.28 |
| + all kernel + I/O work, defaults tuned on corrupt data | 0.95 | 1037 | 71.7 % | 3.33 |
| + 2Q Kin/Kout re-tuned on a clean trace | 1.14 | 860 | 77.4 % | 2.67 |
| + O_DIRECT cold reads | 1.47 | 682 | 77.4 % | 2.67 |
| **+ pool 3621 -> 4415 slots** | **1.71** | **577** | **82.0 %** | **2.13** |

So the day's honest gain is **0.90 -> 1.71 tok/s (1.90x)**, not the 2.4x this file
previously claimed on corrupted output. Compute is genuinely down (`attn` 175 -> 89 ms, `mlp` 203 ->
78 ms); the engine is now overwhelmingly disk-bound at 79 % of wall.

### Cache configuration, re-decided on a clean trace

Policy, at cap 3621 with the tuned split unavailable to LRU/ARC:

| policy | loaded (replay) | loaded (engine) |
|---|---|---|
| **2q, kin 8 / kout 20** | **77.56 %** | **77.4 %** |
| 2q, kin 8 / kout 100 (previous default) | 71.43 % | 71.7 % |
| lru | 71.19 % | 71.0 % |
| arc | 70.77 % | — |

`replay` predicted the tuned cell to **0.16 pp**. With the corruption gone it also
reproduces LRU (71.19 vs 71.0), so cross-policy comparison from the simulator is
now trustworthy — the >21 pp disagreement documented earlier was the corrupted
workload, not a model error.

**Kout is the axis that matters, and small wins** (20 % -> 77.56 %, 100 % ->
71.43 %). The optimum is a broad plateau, kin 6-10 % x kout 15-25 %, all within
0.1 pp; kout below ~5 % collapses. Mechanism: a large ghost remembers keys evicted
long ago, so a re-miss on a stale key promotes it into the protected set — at 13k
unique experts over 3.6k slots most such re-references are spurious and the
protected set fills with one-hit-wonders. This is also the leading explanation for
**ARC** trailing: its B1/B2 ghosts are full-capacity by construction, pinning it to
the bad end of this axis with no way to tune off it.

**Two caveats on this tuning.** It sacrifices 2Q's scan resistance (pinned by
`shipped_default_trades_scan_resistance_for_residency`), and it was tuned on ONE
prompt's trace. A multi-request server workload has real cross-request reuse that a
large ghost is designed to capture, so re-tune before trusting this outside the
single-prompt bench.

Prefetch depth, at the tuned split:

| depth | wall | fetch (miss) | loaded | reads/tok | wasted | tok/s |
|---|---|---|---|---|---|---|
| 0 | 850 | 678 (132) | 77.7 % | 135.1 | 0.0 % | 1.16 |
| 1 (default) | 860 | 678 (121) | 77.4 % | 141.4 | 3.4 % | 1.14 |
| 2 | 886 | 703 (111) | 76.9 % | 153.4 | 8.9 % | 1.11 |

`loaded` is FLAT across all three — prefetch does not improve residency, it
relabels cold reads as preloading and adds waste. The tok/s spread is inside
run-to-run noise, but disk traffic is exact and rises monotonically with depth on
an engine that is 79 % disk-bound. Depth 1 is kept only because the throughput
difference is not resolvable; **prefetch is a candidate for removal** and should be
decided by a repeat measurement, not by this single set.

---

## How much residency is left: Belady, and what closes the gap

`docs/probes/` has no harness for this; it is a short script over the trace. Belady
(evict the entry whose NEXT use is farthest away) is unimplementable online but is
the exact upper bound for ANY eviction policy, so it says how much room a smarter
policy could possibly have. 1 pp of hit rate = ~6 expert reads/token = ~25 ms.

| cap | OPT (Belady) | 2Q tuned | gap |
|---|---|---|---|
| 3000 | 83.09 % | 74.37 % | 8.7 pp |
| 3621 | 85.99 % | 78.96 % | 7.0 pp |
| **4415 (shipped)** | **88.88 %** | **84.13 %** | **4.75 pp** |
| 5000 | 90.86 % | 84.74 % | 6.1 pp |
| 8000 | 95.56 % | 92.27 % | 3.3 pp |

**Do not build an LFU.** A PERFECT static frequency oracle — hold the N most-used
experts forever — scores 76.54 % at cap 3621, *below* 2Q's 78.96 %. Global frequency
knowledge alone loses to what we already ship, so recency is load-bearing and the
Belady gap is not a frequency-estimation problem. It is phase structure: experts run
hot for a stretch of tokens and then go cold, which only future knowledge exploits.

What frequency IS good for is admission. The reuse distribution is heavy-tailed:

| reuse count | experts | % of unique | % of accesses |
|---|---|---|---|
| 1 | 1759 | 13.3 % | **0.57 %** |
| 1-7 | 5670 | 42.8 % | 5.6 % |
| 64+ | 1066 | 8.0 % | **45.7 %** |

1759 one-hit-wonders would occupy up to 49 % of the pool while serving 0.57 % of
traffic, and LRU/2Q/ARC all admit unconditionally on a miss.

**W-TinyLFU: implemented, measured, NOT shipped.** An LRU admission window in front
of an SLRU main cache, gated by a 4-bit count-min sketch with aging
(`--cache-policy wtlfu`, window/protected swept via `--2q-kin`/`--2q-kout`).

| cap | best W-TinyLFU (loaded) | 2Q tuned (loaded) |
|---|---|---|
| 3621 | 77.76 % (win 8 % / prot 90 %) | 76.70 % |
| **4415** | **82.52 %** | **82.74 %** |

It wins by 1.06 pp at the OLD capacity and LOSES at the shipped one. Kept
selectable and documented rather than deleted, because `replay` re-tests it in
seconds if the capacity or workload changes — but 2Q remains the default.

Also not levers, each measured rather than assumed: per-layer capacity allocation
(per-layer unique counts are 87-210, median 187, so global sharing is already
near-right), and workload priming (the `.coli_usage` seed is worth ~0.7 pp at 512
tokens — 77.4 % seeded vs 76.7 % cold — which independently confirms PLAN.md's
decision to park it).

---

## Pool capacity — the biggest single residency lever

`OS_RESERVE` was 26 GiB because PLAN.md measured a ~84 GiB footprint as SLOWER,
"thrash[ing] via OS page reclaim". That mechanism is page-cache contention, and
cold-expert reads are O_DIRECT now and never touch the page cache — so the
confound was removed and the ceiling was re-tested:

| OS reserve | slots | loaded | miss/tok | tok/s |
|---|---|---|---|---|
| 26 GiB (old default) | 3621 | 77.4 % | 121 | 1.47 |
| 20 GiB | 3961 | 79.6 % | 109 | 1.58 |
| 16 GiB | 4188 | 80.9 % | 102 | 1.65 |
| **12 GiB (new default)** | **4415** | **82.0 %** | **96** | **1.71** |

Monotone, no corruption: a 64-token run at 4415 slots produced a routed-expert
workload byte-identical to one at 3621 slots.

**The reserve is not the safety bound — `MAX_BUDGET` is.** The budget derives from
`MemAvailable`, so on a memory-rich boot a reserve alone would size past the point
where the driver can no longer durably back the VMM pool, and decode reads back NaN
(PLAN.md: ~92 GiB total, around token 290). The 88 GiB cap holds the footprint at
the verified point regardless of free memory, with ~4 GiB of margin under the
recorded cliff.

---

## Hardware ceilings

Measured, not vendor figures. Both are needed to read the tables below.

**NVMe array** — O_DIRECT 19 MB random reads (expert-sized), btrfs RAID0 striping
evenly across `nvme0n1p6` + `nvme1n1` (byte counters match to <1 %):

| parallelism | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| GB/s | 2.53 | 5.30 | **6.69** | 6.29 | 6.47 | 6.41 |

**~6.5 GB/s at P≥4.** This *retires* the 16 GB/s figure from the earlier
`iouring_vmm` probe, which was sequential 1 MiB and did not survive contact with
expert-sized random reads.

**LPDDR5X** — ~230 GB/s peak; the shipped `uint` int4 GEMV reaches 190.5 GB/s
(83 %). A `uint2` variant reaches 211.8 GB/s but is NOT shipped — see below.

**Dispatch** (`docs/probes/dispatch_overhead_probe.cpp`): host launch 1.29 µs,
`hipDeviceSynchronize` on an idle queue 0.10 µs, `rmsnorm` identical at grid=1 and
grid=40 (6.73 vs 6.90 µs — launch-latency bound, not occupancy bound). rivoli's
~1800 launches and ~150 syncs per token therefore cost **~3 ms/token total**.

---

## Cache policy × prefetch depth (512 tok)

```
for d in 1 2; do for p in lru arc 2q; do
  rivoli <snap> -bench 512 --pre-seed --prefetch-depth $d --cache-policy $p
done; done
```

| depth | policy | wall | fetch (miss) | attn | mlp | **loaded** | reads/tok | GB/tok | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lru | 1293 | 1083 (156) | 114 | 77 | 71.1 % | 178.0 | 3.36 | 0.76 |
| 1 | arc | 1189 | 978 (159) | 114 | 78 | 70.7 % | 183.1 | 3.46 | 0.83 |
| **1** | **2q** | **505** | **300 (43)** | 114 | 74 | **91.8 %** | **50.6** | **0.96** | **1.92** |
| 2 | lru | 1323 | 1112 (144) | 115 | 78 | 70.6 % | 185.5 | 3.51 | 0.75 |
| 2 | arc | 1228 | 1015 (147) | 115 | 78 | 70.6 % | 193.6 | 3.66 | 0.80 |
| 2 | 2q | 1033 | 826 (120) | 113 | 75 | 75.5 % | 154.6 | 2.92 | 0.95 |

**`2q` + depth 1 is a sharp peak, not a trend.** Every other cell sits at
70.6–75.5 % residency and 0.75–0.95 tok/s; that one cell is 91.8 % and 1.92 tok/s.
LRU and ARC are indistinguishable from each other and barely respond to depth
(70.6–71.1 % across all four of their cells), so the effect is specifically
**2Q's structure × exactly one high-confidence prediction**.

Mechanism: 2Q's A1in probation queue is bounded. At depth 1 the single best
prediction enters A1in and its first real `get` promotes it into the protected Am
set, so the hot set converges fast and holds. At depth 2 the cold-insert rate into
A1in doubles, churning entries out of probation *before* their `get` lands — fewer
promotions, worse Am quality. LRU and ARC have no probation/protection split for
this to exploit. **Prediction confidence rank matters far more than count; more
lookahead actively destroys the cache.**

Consistency check: `attn` is 113–115 ms and `mlp` 74–78 ms in **all six cells** —
the compute buckets are invariant to cache configuration, as they must be.

---

## The prefetch submit bug (fixed 2026-07-21)

`prefetch_layer` instrumented in three parts, depth 2, 128 tok:

```
prefetch cost: alloc 15ms (0.12ms/tok) | sqe-prep 3ms (0.02ms/tok)
             | io_uring_submit 14959ms (116.87ms/tok)      <-- 99%
```

`io_uring_submit` blocked the decode thread **2.96 ms per queued expert** — 61 % of
the 4.85 ms the read itself takes. The ring was created with plain
`io_uring_queue_init`, so submit is an `io_uring_enter` syscall in which the
submitting task drives the btrfs/blk-mq dispatch inline. **The "async, overlaps GPU
compute" prefetch was doing most of the read synchronously**, then hiding the cost
in `other` — the one part of the forward pass no PROFILE bucket covers.

Fix: `IORING_SETUP_SQPOLL`, with graceful fallback. `io_uring_submit` →
**0.00 ms/tok**.

**Correction (later the same day): the DEMAND ring needed it too.** It was
initially left on a plain ring, reasoning that "its submit is immediately followed
by a blocking drain, so it gains nothing." That conflates two different things:
the thread blocks either way, but the reads do not ISSUE concurrently either way.
Without a poller the calling thread walks the submission queue and hands each read
to the block layer one at a time before it starts waiting, so a 6-SQE batch behaves
like queue depth 1. The evidence fit: 6.65 ms for a 19 MB expert is 2.8 GB/s,
essentially the array's measured **P=1** rate (2.53), not its P≥4 rate (6.69).
Both rings now use SQPOLL and share one poller via `IORING_SETUP_ATTACH_WQ`;
`ms/miss` 6.65 → 6.31, tok/s 1.99 → 2.08. Still only ~40 % of the P≥4 rate, so
demand-read concurrency has headroom left.

Effect at 128 tok, `--cache-policy 2q`:

| config | wall | fetch (miss) | loaded | GB/tok | tok/s |
|---|---|---|---|---|---|
| no prefetch | 1003 | 834 (175) | 70.2 % | 3.51 | 0.95 |
| **SQPOLL depth 1** | **631** | 456 (84) | **83.7 %** | **1.94** | **1.45** |
| SQPOLL depth 2 | 1280 | 1099 (126) | 73.6 % | 3.21 | 0.75 |

---

## Kernel work: int4 GEMV vectorisation

`docs/probes/i4gemv_probe.cpp` — one MoE layer's routed batch (E=9, hidden 6144,
moe_inter 2048, 162 MiB int4, past any cache):

| variant | ms/layer | GB/s | speedup | ms/token (75 layers) |
|---|---|---|---|---|
| byte per lane (was shipped) | 2.282 | 74.4 | 1.00× | 171 |
| `uint` (4 B/lane) | 0.883 | 192.4 | 2.58× | 66 |
| `uint4` (16 B/lane) | 0.791 | 214.6 | 2.88× | 59 |

(`uint` is the SHIPPED width. `uint2` below is faster but was rejected in
end-to-end measurement — read that section before changing this.)

The old form gave lane `l` the columns `i ≡ l (mod 32)`, so lanes `l` and `l+1`
read the *same* byte and a wave touched only 16 B per load — one eighth of a cache
line. D3 had fixed the access *stride* but never the *width*.

**`uint4` is unshippable on alignment:** the safetensors header is 101512 bytes,
so all 852 tensors in a shard sit at data offset ≡ 8 (mod 16), and pool placement
preserves the skew — a `uint4` pointer cast is misaligned on every expert.

### `uint2` — faster kernel, REJECTED end-to-end (2026-07-21)

8 mod 16 is still **8-byte** aligned, so `uint2` (8 B/lane = 16 columns, wave-load
256 B) is legal on the real layout and had never been measured. It is a
genuinely faster kernel and it is NOT shipped; the end-to-end result below is why. `docs/probes/i4gemv_u64_probe.cpp`, same shapes, **min of 9
interleaved rounds** — run-to-run spread on this part is ~5 % and a single ordered
pass systematically penalises whichever variant is measured first, so configs are
swept round-robin:

| variant | base % 16 | ms/layer | GB/s | ms/token |
|---|---|---|---|---|
| `uint` (4 B/lane) | 0 | 0.903 | 188.1 | 68 |
| `uint2` (8 B/lane) | 0 | 0.814 | 208.6 | 61 |
| `uint4` (16 B/lane) | 0 | 0.780 | 217.7 | 59 |
| `uint` (4 B/lane) — was shipped | **8** | 0.892 | 190.5 | 67 |
| **`uint2` (8 B/lane) — not shipped** | **8** | **0.802** | **211.8** | **60** |

**`uint2` takes 80 % of the gap to `uint4` and pays nothing for the skew.** Where
`uint` loses ~2 % at the real ≡8 offset (`i4_align_probe.cpp`), `uint2` measures
0.814 aligned vs 0.802 skewed — identical within noise, because its 256 B span
straddles the same number of cache lines either way. So no runtime dispatch or
alignment tier is needed: the single 8 B guard is satisfied by every tensor in the
snapshot. Worth **~7 ms/token**; `mlp` should land near 67 ms against its 74.

`dot_i8_wave` (lm_head, `i8gemv_u64_probe.cpp`, vocab 154880 × hidden 6144, 952 MB,
min of 7 rounds at skew 8) got the same treatment: **207.5 → 223.6 GB/s**
(4.59 → 4.25 ms), 97 % of peak. Only ~0.3 ms/token — it rides along with the int4
change rather than justifying itself.

The implementation makes both helpers three tiers (uint2 → uint → scalar), each
guarded on the actual row pointer; at GLM dims only the uint2 tier runs. Verified
against a host scalar oracle over 15 dims × 6 base offsets, plus 50-launch
bit-identical determinism. It is correct and it is faster. **It is still not
shipped**, for a reason no micro-benchmark could show:

| build (512 tok, defaults + `--pre-seed`) | `mlp` | miss/tok | **loaded** | tok/s |
|---|---|---|---|---|
| without `uint2` | 74 ms | 44 | 91.6 % | **2.12** |
| with `uint2` | **69 ms** | **70** | **87.0 %** | 1.71 |

`uint2` delivered its 5 ms of `mlp` exactly as predicted — and cost ~120 ms of
`fetch`. Changing the summation order perturbs f32 rounding, which changes the
sampled tokens, which reroutes experts, which collapsed residency from 91.6 % to
87.0 %.

**This is a routing lottery, not a regression.** The same kernel on a different
prompt could land the other way; nothing about `uint2` is worse. But trading a
~1 % compute gain for a ~15 % routing variance is not a bet worth taking on the
default path when the only evidence available is one prompt. Revisit it against a
prompt SET, where the routing draw averages out and the 5 ms is what remains.
The patch lives on branch `worktree-agent-add1d8c9d69004b34`.

This is the sharpest illustration of the tok/s-comparability rule at the top of
this file: the buckets said ship it, the wall clock said don't, and both were
right about different things.

In-engine effect, 128 tok (buckets, which are hit-rate independent):

| bucket | before | after |
|---|---|---|
| `attn` | 175 ms | **87 ms** (2.01×) |
| `mlp` | 203 ms | **102 ms** (1.99×) |
| compute (wall − fetch) | 481 ms | **327 ms** |

`mla_value` was the last kernel in the pre-D3 thread-per-row shape (32 lanes
walking 32 different rows, so 32 cache lines per load instruction); rewritten
wave-per-row, measured 5.45× in isolation but worth only ~7 ms/token.
`mla_absorb` deliberately left alone — 1.10× measured, not worth a second path.

---

## Split-KV attention (2026-07-21)

`mla_latent_attend` launched `grid = ceil(H/HB)` = **8 workgroups on a 40-CU GPU**
(~20 % occupancy), and `attn` was the only bucket that grew with context (77 ms at
128 tok → 113 ms at 512). The attended rows are now partitioned into `n_splits`
chunks so `grid = ceil(H/HB) * n_splits` (~80 blocks at 512 rows); each split
computes a partial online-softmax and a combine kernel merges them by rescaling to
the global max.

**`attn` 113 → 88 ms (−22 %).** Stability: every exponent formed anywhere is
non-positive (`s − m_split` inside a split, `m_split − m_global` in the combine),
so no unshifted exponential ever exists. Determinism: partials are combined in
fixed split order, no atomics, and `n_splits` depends only on `(H, nr)` — so a
given context length always produces the same summation order. Scratch is 2.1 MB
allocated once at engine construction (the M3 contract forbids per-token
`hipMalloc`). Below 64 rows it degenerates to bit-identical single-split
behaviour. Cache counters moved only 91.8 % → 91.6 %, consistent with the one
extra rounding layer the combine adds.

---

## Retired claims

Recorded because each was believed, acted on, and then measured false.

| claim | status |
|---|---|
| "NVMe delivers 16 GB/s at QD≥4" | **Wrong.** Sequential-1 MiB artifact. Real ceiling ~6.5 GB/s for 19 MB random reads. |
| "prefetch depth 2 is validated optimal" | **Wrong.** Measured while `io_uring_submit` blocked; depth 1 wins once submit is async. |
| "prefetch is a net loss, disable it" | **Wrong**, same cause. With SQPOLL it is worth +53 % at 128 tok. |
| "ARC is the best policy" | **Wrong.** Worst of the three at depth 1 (0.83 vs 2q's 1.92). |
| "hit rate ≈ residency" | **Wrong.** Prefetched-off-disk experts count as hits; `loaded` is the real metric. |
| "≤1 join/token is a throughput lever" (bottleneck #3) | **Retired.** Inherited from colibri's OpenMP costs; on HIP all ~150 syncs cost ~0.9 ms/token. |
| "prefetched entries are evicted before use" (hypothesis) | **Refuted.** Only 127/1205 (depth 2) and 609/15179 (depth 8) wasted reads were later demanded, and the share *falls* with depth. Waste is plain misprediction. |
| "the demand ring gains nothing from SQPOLL" | **Wrong.** Blocking the thread and issuing reads concurrently are different things; the batch was behaving like QD=1. Worth +4.5 %. |
| "prefetch fetch depth and cache admission are separable" (hypothesis) | **Refuted.** A disjoint side table starves the policy: residency 91.8 % -> 71.7 % at fetch-depth 4. |
| "resident-tier VMM → ~2× experts fit" | **Retired** earlier (double-counting; madvise already banked it). See PLAN.md. |

---

## Reproducing

```bash
cargo build --release --features rocm
cargo test  --release --features rocm -- --test-threads=1   # 52 tests; serial: sole-tenant guard
./target/release/rivoli /var/db/llama-server/glm52-colibri-int4 -bench 512 --pre-seed
```

Probes (`hipcc -O3 --offload-arch=gfx1151 <file> -o <bin>`):
`docs/probes/i4gemv_probe.cpp`, `i4gemv_u64_probe.cpp`, `i8gemv_u64_probe.cpp`,
`mla_proj_probe.cpp`, `i4_align_probe.cpp`, `dispatch_overhead_probe.cpp`.

The GPU must be sole-tenant (the engine refuses to start on >1 GiB foreign GTT),
and the snapshot must be on **local** storage — io_uring→VMM EFAULTs on NFS fds
under kernel 6.18.38.
