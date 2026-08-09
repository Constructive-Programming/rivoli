---
scope: k3
status: data
verdict: The C reference's own measurements, vendored at ff11dce so they cannot move under us. The load-bearing correction they carry: the 3.2 GB/s in PERFORMANCE.md is a `dd` QUEUE-DEPTH-1 number, and the engine's own measured trunk rate is 5373-6064 MB/s — the repo says so in environment.txt:36-42. Every "impossible" sub-bandwidth-floor reading in k3-port.md came from taking 3.2 as the floor. The trunk IS re-read in full every token when not pinned (93 binds/token), so real traffic is 25.83 GB of experts PLUS up to 108.81 GB of trunk = ~135 GB/token, which BENCHMARKING.md states outright while PERFORMANCE.md's "GB read/tok" column reports experts only. Each ladder rung is 8 generated tokens from a 5-token prompt, and step 0 alone costs 53 s against ~8.9 s steady state, so the ladder's s/tok is an 8-token average, not a steady-state rate.
---

# The C reference's measurements, vendored

Copied verbatim 2026-08-09 from `github.com/FareedKhan-dev/kimi-k3-in-c` at
**`ff11dce858a2eb8a781224facdffd33a1fa48d25`** (2026-08-07). Vendored because
`docs/investigations/k3-port.md` §4 reasons from these numbers and a live third-party repo can
be force-pushed. Nothing here is rivoli's measurement.

| file | source path |
|---|---|
| `memory-ladder.tsv` | `docs/data/memory-ladder.tsv` |
| `trunk-cache-split.tsv` | `docs/data/trunk-cache-split.tsv` |
| `environment.txt` | `docs/data/environment.txt` |

## The correction these carry

`environment.txt:36-42`, verbatim:

```
--- storage bandwidth, measured ---
O_DIRECT cold : 3.2 GB/s     (dd bs=4M iflag=direct after drop_caches)
buffered warm : 2.3 GB/s
engine, trunk : 5373-6064 MB/s sustained during real runs
NOTE O_DIRECT is FASTER than buffered here. That is the opposite of the usual
     expectation, and it is why the engine opens the trunk O_DIRECT. Worth
     re-checking on any target device before assuming it holds.
```

**3.2 GB/s is a `dd` queue-depth-1 figure and is not the device's ceiling.** The engine issues
up to 16 concurrent 17.55 MB O_DIRECT expert reads and runs a separate trunk reader thread, and
measures **5373–6064 MB/s** doing it, on virtio-backed storage. `tools/devbw.py` in the
reference exists specifically to make this point: "dd is one sequential stream at queue depth 1
… a dd number cannot tell you whether a device is the bottleneck for THIS engine."

This resolves what `k3-port.md` §4b recorded as an impossible sub-floor reading. There was no
paradox — the floor was computed from the wrong number. **The trunk-miss model is confirmed,
not refuted:** `k3_trunk.c:361` states "the trunk is read 93 times per token", and the whole-run
counter (374.99 GB at the 128 GB rung) reconstructs to within rounding.

## How to read the ladder without repeating our mistakes

- **`GB read/tok` is EXPERT bytes only.** The trunk is a separate counter. Real per-token
  traffic at zero trunk residency is `25.83 + 108.81 = 134.64 GB`, which
  `docs/BENCHMARKING.md:47` states as "~135 GB per token".
- **`s/token` is an 8-token average, not steady state.** Each rung generates 8 tokens from a
  5-token prompt (`benchmarks/memory-ladder.sh`, `GEN=8`). Step 0 costs ~53 s against ~8.9 s
  for later steps, so the average is inflated. The reference's own sustained figures over
  16–32 tokens are **10.66–11.79 s/token**.
- **`trunk_hit` counts layer binds, not bytes.** 8 passes × 93 layers = 744 binds; hits are
  `7 × pinned_layers` because token 0's pin fill counts as misses. Every row reproduces.
- **`memory-ladder.tsv` was not produced by the shipped harness at this commit** — its columns
  and rung count differ from `benchmarks/memory-ladder.sh`, and its `io_share` uses a
  definition `k3_run.c:1432` describes having since fixed. The data is internally consistent
  and reproduces the engine's measured device rates, but it cannot be regenerated as shipped.

## What this implies for rivoli, and the trap to avoid twice

**Check whether rivoli's own 7.0 GB/s is a queue-depth-1 number before using it as a floor.**
That is exactly the mistake this page documents, and `docs/measurement/benchmarks.md` should be
consulted for how it was obtained. If it is a `dd`-class figure, rivoli's io_uring path may
beat it the same way the reference's beats 3.2 — and every derived tok/s in `k3-port.md` §4c
moves with it.
