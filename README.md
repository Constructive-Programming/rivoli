# rivoli

<img src="docs/rivoli-hummingbird.jpg" alt="Rivoli's hummingbird (Eugenes fulgens)" width="480">

*Rivoli's hummingbird (Eugenes fulgens) — Bernard Gagnon, [CC0](https://creativecommons.org/publicdomain/zero/1.0/), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Eugenes_fulgens_in_Costa_Rica.jpg).*

A GLM-5.2 mixture-of-experts decode engine for the **AMD Strix Halo** APU
(Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151), written in Rust with a tokio
streaming architecture and HIP/ROCm compute. Goal: **stable ≥ 1 tok/s**
single-stream int4 decode of the full 744B model on one 128 GB machine.

## Lineage

rivoli is the successor to **colibri** — a from-scratch C engine that ran the
same model on the same box. *Colibri* is Spanish for hummingbird; **Rivoli's
hummingbird** keeps the name and honors François Victor Masséna, 2nd Duke of
Rivoli, for whom the species is named. rivoli inherits colibri's hard-won
evidence and its usage-ranking snapshot format directly, but rebuilds the
architecture around what that campaign proved:

- the decode is **dispatch-bound, not bandwidth-bound** — at 0.72 tok/s the
  memory bus sat ~96 % idle; the cost was ~920k OpenMP fork/joins per run and
  single-threaded glue, not weight traffic;
- **pinning the hot experts** (usage-ranked, ~95 % hit) takes NVMe off the
  critical path;
- **one fused kernel launch per layer** beats per-expert dispatch (colibri's
  ~4800 submits/token lost seconds to fence-waits);
- **streaming the feed** — overlapping cold-expert fetch with resident
  compute — is what keeps the engine from starving serially.

rivoli's design law: **win by removing bottlenecks** — NVMe bandwidth, RAM
bandwidth, synchronization points, and synchronization cost. See
[`PLAN.md`](PLAN.md) for the full evidence trail and milestone gates.

## Design in one breath

- **Single engine, no CPU fallback.** The GPU computes every expert; the CPU
  only routes, samples, and drives the feed. A run with zero kernel launches
  is a hard error, never a silent fallback.
- **Zero knobs.** One flag: `rivoli <snapshot-dir> -bench <tokens>`. Memory
  budgets, device tier, and thread pool are auto-discovered and printed as the
  run's first line. (An OpenAI-compatible server is the flagless default,
  later.)
- **NVMe → iGPU direct.** Cold experts stream from disk into cacheable,
  coherent unified-memory slabs the fused kernels read in place — zero
  intermediate copies, leveraging the APU's unified LPDDR5X.
- **Sole tenant.** rivoli refuses to start if another process holds the GPU;
  large-allocation co-tenancy is the proven path to an amdgpu wedge.

## Build & run

```sh
cargo build                      # CPU-side dev; no GPU toolchain needed
cargo build --features rocm      # compiles the HIP kernels (needs hipcc)
./target/release/rivoli <snapshot-dir> -bench 128
```

## Format note

The engine currently reuses colibri's per-row int4 snapshot. A follow-up
converter — HF GLM-5.2's fp8 (e4m3, block-scaled 128×128) → **group-scaled
int4** — is planned once the ≥1 tok/s model is proven; it improves accuracy at
the same ~4 bits/weight. Hardware MX formats (MXFP4/MXINT4) are **not**
available on gfx1151 (they are CDNA4/Blackwell); RDNA 3.5 offers plain int4
WMMA, which matters only for the future batched server path, not single-stream
GEMV decode.

## License

MIT. See [`LICENSE`](LICENSE).
