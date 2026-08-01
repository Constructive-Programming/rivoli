# docs index — read this, not the directory

~490 KB lives here and **most of it is investigation log, not reference.** The logs are kept
deliberately: what an approach *eliminated* is worth as much as what it found, and several
have been re-opened and re-closed on that record. But they are written chronologically, so
the current answer is usually buried under the history that produced it.

**So: find your question below, read the one section it names.** Every doc ≥25 KB opens with
a STATE block that gives the current answer in ~15 lines.

## By question

| You want to know | Go to | Don't |
|---|---|---|
| How the engine works, end to end | **`ARCHITECTURE.md`** — read whole | — |
| What `--mode` / `--cache-policy` do, and which to pick | `../MODES.md` (16 KB, read whole) | — |
| Which mode has the best quality | `INT4.md` §0 table. int4 5.120 > hybrid 5.189 > int3-vq 5.275 | ranking on `benchmarks.md`'s top table — it predates the `.i4` rebuild |
| Why int4 was once unusable | `INT4.md` §10 — per-row scales; group-128 fixed it, 73.43 → 5.120 | quoting 73.43 as current; it is the pre-fix number |
| Where the time goes, and what to optimise next | `PERF.md` §"Ranked roadmap" (bottom) | reading the 39 KB above it |
| Whether speculative decode is worth it | `ARCHITECTURE.md` §13. **Yes, 1.108× — but only gated** (`--mtp-min-conf 0.8`); ungated it is 0.93–0.95× | quoting the ungated number as the verdict; it was the answer until 2026-07-31 |
| What Vulkan can and cannot run | `VULKAN.md` §"Kernel inventory" | the other 110 KB — it is a port journal |
| Whether the NPU offload is worth it | `NPU.md` §"The finding, in five lines" | the other 55 KB |
| How the expert cache decides residency | `../MODES.md` §"Cache policies", then `CACHE_ROUTE.md` §"Design" | `CACHE_ROUTE.md` §"RETIRED" onwards — `top-m` is gone |
| Whether cross-layer prefetch works | `CACHE_PILOT.md` header. **No — settled 2026-08-01.** Not for want of accuracy (82.7% recall) or bandwidth (the drive idles 35%): the idle window is 1.13 ms and one expert read is ~2 ms | the two reasons previously recorded — "prediction is too hard" and "overlap creates no bandwidth" — both measured false |
| Whether the disk fetch can be made faster | `ARCHITECTURE.md` §3, "What the drive actually does". **No** — it is already at what its queue depth buys; the drive is *idle* 35% of a token and only prediction fills that | trusting `fetch_hidden_pct` from before 2026-08-01; it read ~97% on every configuration, including ones running at half speed |
| How to read a trace / profile | `TRACES.md` §"What to actually look at" | `GPU_TRACE.md` unless you are attaching a profiler |

## The docs, one line each

**Reference — meant to be read.**
- **`ARCHITECTURE.md`** (44 KB) — the engine as it is. Memory regions, decode pipeline,
  io_uring streamer, async-signal bridge, MoE launch, caching, the **§8b INV-n registry**
  (enforced by `tests/invariants.rs`), fixed-point accumulation (§12), speculative decode
  (§13).
- **`../MODES.md`** (16 KB) — the format × policy matrix, and which knob does what.
- **`../benchmarks.md`** (105 KB) — **measurements, append-only.** Never read whole; grep
  for the config. Its top table predates the `.i4` rebuild and says so.

**Closed investigations — grep, don't read.**
- **`INT4.md`** (25 KB) — why int4 was unusable (per-row scales) and the group-128 fix.
  **RESOLVED.** §1 and §9 outlive int4 and are worth reading on their own.
- **`VULKAN.md`** (117 KB) — the second backend, port journal across four phases. Current
  state is the inventory section; the rest is how it got there.
- **`NPU.md`** (57 KB) — DSA indexer offload to the NPU. Answer is in the first 40 lines;
  the device top-k it recommended is **built and shipped** (−9.4 ms/token).
- **`CACHE_ROUTE.md`** (29 KB) — routing-aware caching. `top-m` **RETIRED 2026-07-30**.
- **`CACHE_PILOT.md`** (28 KB) — LOOKA + the `--hint-k` veto layer. **Removed from the engine 2026-07-31**; the doc is the record of why. `bin/replay`'s offline `Pilot` is unaffected.
- **`PERF.md`** (39 KB) — the performance roadmap. The ranked table at the bottom is the
  live part; everything above is the evidence for a row in it.
- **`TRACES.md`** (16 KB) / **`GPU_TRACE.md`** (9 KB) — OTLP spans, and why ROCm GPU
  profiling does not work on this part.

## If you are writing here

Correct **in place with a dated note**; do not delete and do not silently overwrite. Half
this session's doc work was fixing staleness banners that had themselves gone stale, and
claims that contradicted the code (`--pilot-k` does not exist; a flag documented as "always
on" defaults to off). A correction that erases what was believed before makes the next
reader repeat the experiment.
