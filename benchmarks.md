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

`--cache-policy 2q --prefetch-depth 1`, prefetch on, SQPOLL prefetch ring.

```
rivoli /var/db/llama-server/glm52-colibri-int4 -bench 512 --pre-seed
```
```
PROFILE/tok: 505ms wall | fetch 300ms 59% (43 miss, 6.97ms/miss)
           | attn 114ms 23% | mlp 74ms 15% | lmhead 4ms 1% | route 2ms | other 10ms
512 tokens in 266.5s — 1.92 tok/s | hit 92.5%
expert source: loaded 91.8% | preloading 0.6% | cold 7.5%
disk traffic: 50.6 expert reads/tok (0.96 GB/tok), 2.3% wasted
```

Trajectory over 2026-07-20/21, same command, 512 tok:

| state | tok/s | ms/tok | loaded | GB/tok |
|---|---|---|---|---|
| M4 gate as recorded in PLAN.md | 1.31 | 763 | — | — |
| measured baseline before this work | 0.90 | 1093 | — | — |
| + vectorised int4 kernels | 0.96 | 1033 | 71.3 % | 3.28 |
| + SQPOLL prefetch ring, 2q, depth 1 | **1.92** | **505** | **91.8 %** | **0.96** |

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

**LPDDR5X** — ~230 GB/s peak; the vectorised int4 GEMV reaches 214.6 GB/s (93 %).

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

Fix: `IORING_SETUP_SQPOLL` on the prefetch ring only (the main ring's submit is
immediately followed by a blocking drain, so it gains nothing), with graceful
fallback. `io_uring_submit` → **0.00 ms/tok**.

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
| **`uint` (4 B/lane)** | **0.883** | **192.4** | **2.58×** | **66** |
| `uint4` (16 B/lane) | 0.791 | 214.6 | 2.88× | 59 |

The old form gave lane `l` the columns `i ≡ l (mod 32)`, so lanes `l` and `l+1`
read the *same* byte and a wave touched only 16 B per load — one eighth of a cache
line. D3 had fixed the access *stride* but never the *width*.

**`uint` chosen over `uint4` on alignment evidence:** the safetensors header is
101512 bytes, so all 852 tensors in a shard sit at data offset ≡ 8 (mod 16), and
pool placement preserves the skew — a `uint4` pointer cast is misaligned on every
expert. The `uint` path needs only 4 B alignment and costs 2 % for the skew
(203.9 GB/s aligned vs 199.7 at the real ≡8 offset), so `uint4`'s extra ~7 ms/token
is not worth a runtime dispatch and fallback tier.

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
| "prefetched entries are evicted before use" (my hypothesis) | **Refuted.** Only 127/1205 (depth 2) and 609/15179 (depth 8) wasted reads were later demanded, and the share *falls* with depth. Waste is plain misprediction. |
| "resident-tier VMM → ~2× experts fit" | **Retired** earlier (double-counting; madvise already banked it). See PLAN.md. |

---

## Reproducing

```bash
cargo build --release --features rocm
cargo test  --release --features rocm -- --test-threads=1   # 52 tests; serial: sole-tenant guard
./target/release/rivoli /var/db/llama-server/glm52-colibri-int4 -bench 512 --pre-seed
```

Probes (`hipcc -O3 --offload-arch=gfx1151 <file> -o <bin>`):
`docs/probes/i4gemv_probe.cpp`, `mla_proj_probe.cpp`, `i4_align_probe.cpp`,
`dispatch_overhead_probe.cpp`.

The GPU must be sole-tenant (the engine refuses to start on >1 GiB foreign GTT),
and the snapshot must be on **local** storage — io_uring→VMM EFAULTs on NFS fds
under kernel 6.18.38.
