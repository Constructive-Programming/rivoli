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
  routing, which changes the hit rate, which changes `fetch`. Cache-config A/Bs
  (policy, depth) are safe — routing is deterministic given the weights. Kernel
  changes are **not**; compare their `attn`/`mlp` ms buckets instead, which are
  hit-rate independent.
- Cache counters are **exact and reproducible** (the two `2q`+depth-1 runs below
  produced byte-identical `loaded`/`cold` counts). Wall-clock carries ~±3 %.

---

## Current defaults

`--cache-policy 2q --prefetch-depth 1 --2q-kin 8 --2q-kout 100`, prefetch on,
SQPOLL on BOTH io_uring rings, split-KV attention.

```
rivoli /var/db/llama-server/glm52-colibri-int4 -bench 512 --pre-seed
```
```
PROFILE/tok: 458ms wall | fetch 281ms 61% (44 miss, 6.31ms/miss, submit 11ms + join 269ms)
           | attn 88ms 19% | mlp 74ms 16% | lmhead 5ms 1% | route 2ms | other 9ms
512 tokens in 242.0s — 2.12 tok/s | hit 92.2%
expert source: loaded 91.6% | preloading 0.7% | cold 7.8%
disk traffic: 52.2 expert reads/tok (0.99 GB/tok), 2.3% wasted
```

Trajectory over 2026-07-20/21, same command, 512 tok:

| state | tok/s | ms/tok | loaded | GB/tok |
|---|---|---|---|---|
| M4 gate as recorded in PLAN.md | 1.31 | 763 | — | — |
| measured baseline before this work | 0.90 | 1093 | — | — |
| + vectorised int4 kernels | 0.96 | 1033 | 71.3 % | 3.28 |
| + SQPOLL prefetch ring, 2q, depth 1 | 1.99 | 488 | 91.8 % | 0.96 |
| + SQPOLL demand ring + split-KV attn | 2.12 | 458 | 91.6 % | 0.99 |
| + swept 2Q Kin/Kout (8 %/100 %) | **2.10–2.33** | **416–462** | **92.2 %** | **0.95** |

**~2.4x over the day.** Note the two halves: the kernel work took compute from
481 to 174 ms/token, and the I/O work took `fetch` from ~600 to 281 ms while
cutting disk traffic 3.7x. Neither alone would have crossed 2 tok/s.

### Parallel-agent validation round (2026-07-21)

Five candidate optimisations, each built in an isolated worktree, then validated
serially (the GPU is sole-tenant and the NVMe is shared — see *Measurement
hygiene*). 512 tok, defaults + `--pre-seed`, quiet machine:

| candidate | wall | fetch (miss) | attn | mlp | loaded | tok/s | verdict |
|---|---|---|---|---|---|---|---|
| baseline | 488 | 287 (43) | 113 | 71 | 91.8 % | 1.99 | — |
| demand-ring SQPOLL | 465 | 261 (43) | 114 | 74 | 91.8 % | 2.08 | **shipped** |
| split-KV attention | 473 | 295 (44) | **88** | 74 | 91.6 % | 2.05 | **shipped** |
| **both** | **458** | 281 (44) | **88** | 74 | 91.6 % | **2.12** | **shipped** |
| both + `uint2` GEMV | 571 | 401 (70) | 88 | **69** | 87.0 % | 1.71 | rejected (routing) |
| prefetch fetch-depth 4 | 1199 | 984 (106) | 118 | 80 | **71.7 %** | 0.82 | rejected |
| 2Q Kin/Kout sweep | — | — | — | — | — | — | tooling only |

The two winners compose additively and independently — one is pure I/O
concurrency, the other pure kernel occupancy, and each preserved the other's
bucket unchanged.

**`--prefetch-fetch-depth` refuted the hypothesis it was built to test.** The idea
was that prefetch's *fetch* volume and its *cache admission* are separable, so we
could fill the NVMe queue without polluting 2Q's A1in probation. Fetch-depth 1
reproduced baseline byte-identically (the side table is correctly inert), but at
fetch-depth 4 residency COLLAPSED 91.8 % -> 71.7 % and traffic went 50.6 -> 219.8
reads/tok. Mechanism: `prefetch_layer` skips keys resident in *either* tier, so a
hot expert parked in the side table is never re-admitted to the policy and can only
age out — the side table starves the cache it was meant to protect. Branch
`worktree-agent-ab265f8e452e3d3f3` if someone wants to try promotion-on-use.

### Measurement hygiene

Running the five agent builds concurrently with a benchmark cost **18 % of
throughput** and would have been invisible without the counters:

| | quiet machine | during 5 concurrent builds |
|---|---|---|
| cache counters | 286885 / 23315 | **286885 / 23315** (byte-identical) |
| `attn` / `mlp` | 113 / 71 ms | **113 / 71 ms** (identical) |
| `ms/miss` | 6.65 | **9.48** (+36 %) |
| tok/s | 1.99 | 1.58 |

Cache behaviour and compute were bit-for-bit unchanged; only disk service time
moved. Benchmark on an idle machine, and always check the counters before
believing a delta.

---

## Offline cache simulation — `replay` (2026-07-21)

The routed-expert sequence does NOT depend on the cache: routing is a deterministic
function of the weights and the token stream, so which experts each layer asks for
is fixed no matter what is resident. That makes cache configuration answerable
offline. `rivoli --trace <path>` captures the access sequence from one real run;
`bin/replay` replays it through LRU / 2Q / ARC at any capacity and any 2Q Kin/Kout,
**on CPU in ~2 s**. The 6-cell policy grid above cost 66 minutes of GPU; the 66-cell
Kin/Kout sweep below cost 2.3 seconds.

Two bugs made the pre-existing `simulate` useless for this and are now fixed: it
only ever called `insert` (never `insert_cold`, so A1in probation admission was
never exercised), and the trace format **never recorded the predictions** — the
information was not in the file. Any sweep on the old format modelled a
*no-prefetch* 2Q, i.e. the ~70 % row, not the ~92 % one. `cache::replay` now
mirrors `resolve_layer` + `prefetch_layer` ordering exactly (cold-admit predictions,
then all demand hits, then all misses) and returns the loaded/preloading/cold split.

**Calibration, 512-tok trace, cap = 3621 slots, seeded from `.coli_usage`:**

| | live engine | `replay` | error |
|---|---|---|---|
| **2q (default)** | **91.6 %** | **91.74 %** | **0.14 pp** |
| lru | 71.1 % | 92.85 % | **21.8 pp** |
| arc | 70.7 % | 93.27 % | **22.6 pp** |

**Use this tool for 2Q parameter work ONLY.** It reproduces 2Q to a seventh of a
point and is wildly wrong for LRU and ARC — it claims both would beat 2Q, when the
engine measured them 20 points worse. The cause is unresolved (a leading suspect is
the seed path: `Pin::build`'s pre-seed goes through `pool.alloc` -> `policy.insert`,
whereas `replay` calls `Cache::seed`, and the two land in different segments). Until
that is chased down, **cross-policy comparisons from `replay` are not evidence.**

### 2Q Kin/Kout sweep

```
rivoli <snap> -bench 512 --pre-seed --trace /tmp/rivoli.trace
replay /tmp/rivoli.trace 3621 --seed <snap> --sweep
```

`loaded %`, 11 Kin x 6 Kout:

| kin\kout | 25% | 50% | 100% | 200% | 400% | 800% |
|---|---|---|---|---|---|---|
| 3% | 91.83 | 91.87 | 91.92 | 91.92 | 91.92 | 91.92 |
| 5% | 92.09 | 92.07 | 92.13 | 92.12 | 92.12 | 92.12 |
| **8%** | 92.19 | 92.16 | **92.21** | 92.20 | 92.20 | 92.20 |
| 12% | 92.11 | 92.08 | 92.10 | 92.10 | 92.10 | 92.10 |
| 16% | 92.00 | 92.00 | 92.04 | 92.04 | 92.04 | 92.04 |
| 20% | 91.93 | 91.89 | 91.98 | 91.97 | 91.97 | 91.97 |
| 25% (was default) | 91.75 | 91.74 | 91.80 | 91.81 | 91.81 | 91.81 |
| 33% | 91.49 | 91.57 | 91.63 | 91.63 | 91.63 | 91.63 |
| 40% | 91.19 | 91.29 | 91.33 | 91.33 | 91.33 | 91.33 |
| 50% | 90.68 | 90.79 | 90.84 | 90.84 | 90.84 | 90.84 |
| 66% | 89.46 | 89.63 | 89.77 | 89.82 | 89.82 | 89.82 |

**Kout is irrelevant** — every column is flat to within 0.06 pp, so the A1out ghost
may as well not exist on this workload. **Kin matters and smaller is better**, down
to 8 %, falling off again at 3 %. That is the same mechanism as prefetch depth 1
beating depth 2: a tight probation queue lets a prefetched expert reach the
protected Am set before churn evicts it. Both knobs say the same thing — 2Q wins
here because of *fast promotion*, and anything that slows promotion costs residency.

**Hardware confirmation** (kin 8 % / kout 100 %, now the default):

| | before (kin 25/kout 50) | after (kin 8/kout 100) |
|---|---|---|
| predicted `loaded` | 91.74 % | 92.21 % |
| **measured `loaded`** | **91.6 %** | **92.2 %** |
| miss/tok | 44 | **40** |
| GB/tok | 0.99 | **0.95** |
| tok/s | 2.12 / 2.15 | **2.10 / 2.33** |

The model predicted the residency gain to within 0.15 pp. Note the tok/s column:
the same config measured 2.10 and 2.33 on two runs with **byte-identical cache
counters**, so wall-clock noise here is ~±10 % and the residency/traffic figures
are the trustworthy ones. Do not read 2.33 as "the" number.

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
