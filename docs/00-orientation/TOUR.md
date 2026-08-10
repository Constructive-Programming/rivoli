---
scope: engine
status: live
verdict: New here? Read this and stop. The engine in two pages, plus the five things that will bite you.
---

# rivoli — the two-page tour

A decode engine for **GLM-5.2**, a 78-layer mixture-of-experts model, running on one **AMD
Strix Halo** APU (gfx1151, 40 CUs, unified LPDDR5). Rust, on HIP/ROCm — the only backend
since a second, Vulkan, was retired 2026-08-06 (`docs/investigations/vulkan-kernels.md`).

## The one fact the whole design follows from

Each layer has **256 routed experts**; a token uses **8 of them**, plus one shared expert.
At int3-vq an expert is 15.34 MB, so a layer's experts are ~3.9 GB and the model's are
~290 GB. **They do not fit in memory.** The device budget holds maybe 115 GiB.

So the experts **stream from NVMe while the resident ones compute**, and that overlap is
the engine. Almost every design decision in this repo is downstream of it:

```
  attention ──▶ router picks 8 ──▶ which are resident?
                                     ├── hit  ─────────────▶ compute now
                                     └── miss ──▶ NVMe read ──▶ compute when it lands
```

A token touches ~150 experts it doesn't have. That is **2.25 GB of reads per token**, and
it is why "how fast can we read" matters more here than "how fast can we multiply".

## How a token is produced

1. **Attention** — MLA with a DSA sparse indexer selecting rows.
2. **Route** — gate logits come off the device, the host does sigmoid + bias + top-k. This
   is a *pure function of (logits, bias, top_k)* and never consults the cache (**INV-1**),
   which is what makes every caching change output-bit-identical by construction.
3. **Submit** — the pool is asked for those 8 experts. Hits get pinned; misses get a slot
   and an io_uring O_DIRECT read. Each expert comes back with a **`Ticket`**, a value on a
   device timeline.
4. **Launch** — residents first (order cost 20% once), then misses, on two streams. Every
   kernel enqueues a device-side wait on its ticket (**INV-5**), so the GPU waits, not the
   host.
5. **Accumulate** — fixed-point, one accumulator row per stream, drained into the residual.

Read [`reference/architecture.md`](../reference/architecture.md) for the real version. It is
44 KB and it is **the one doc meant to be read whole.**

## Five things that will bite you

1. **The GPU is sole-tenant, and several agents share this machine.** Wrap every GPU command
   in `flock /var/run/sys-gpu.lock -c '…'`. Build *outside* the lock. Two concurrent
   benchmarks do not just contend — they make `DeviceTier::new` fail to allocate.

   **`/var/run` is `/run`, which is tmpfs — the lock file does not survive a reboot**, and
   `/run` is root-owned so `flock` cannot recreate it as an ordinary user. The failure is
   loud rather than silent (`flock: cannot open lock file …: Permission denied`, exit 66)
   so nothing ever runs *believing* it holds a lock it does not — but every GPU command
   stays broken until someone restores the file:

   ```bash
   sudo install -m 666 -o "$USER" /dev/null /run/sys-gpu.lock
   ```

2. **Never `cargo build` between the two arms of an A/B.** It evicts page cache and moved
   `ms/miss` from 1.36 to 5.14 in one measured pair.

3. **Free-running `tok/s` cannot rank a quality change.** A degenerate run looks *fastest*.
   Rank on paired dNLL from `bin/ppl`; an interval straddling zero is inconclusive, not a
   pass. `distinct` and `longest repeated block` measure nothing — they fire identically on
   a repetition loop, on spliced corruption, and on prose that restates a paragraph.

4. **Cache changes cannot move the output.** Policy, `--max-mem` and prefetch are
   output-neutral by construction (INV-1). If tokens move when only those changed, you have
   found a bug, not a trade. *(One live exception: `--mode hybrid` picks each expert's
   format by residency, so its output does follow the cache. Don't A/B quality in hybrid.)*

5. **`docs/` is mostly closed investigation.** Read
   [`INDEX.md`](INDEX.md) and use the verdict column to decide what *not* to open. Reading
   `vulkan-port.md` end to end will cost you an afternoon and teach you mostly about rejected
   options. `benchmarks.md` is the exception — it is readable whole, and its STATE table is
   the live state.

## Build and test

```bash
cargo build --release --features rocm        # the only backend
flock /var/run/sys-gpu.lock -c 'cargo test --release --features rocm'
cargo clippy --release --features rocm --all-targets
```

A featureless build compiles to a refusal stub. That is deliberate, not breakage.

## Where to go next

| you want to | go to |
|---|---|
| understand the engine properly | [`reference/architecture.md`](../reference/architecture.md) |
| pick a `--mode` / `--cache-policy` | [`reference/modes.md`](../reference/modes.md) |
| make it faster | [`measurement/perf-roadmap.md`](../measurement/perf-roadmap.md), then [`how-to-measure.md`](../measurement/how-to-measure.md) |
| measure anything at all | [`measurement/how-to-measure.md`](../measurement/how-to-measure.md) — read it *before* booking the GPU |
| know whether an idea was already tried | [`INDEX.md`](INDEX.md), investigations table |
