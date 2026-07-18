# colibri vs rivoli — architecture head-to-head (2026-07-18)

Both run GLM-5.2 (int4, MoE, MLA) by streaming cold experts from NVMe over a
resident pin. colibri (`~/workspace/opensource/colibri`, `c/glm.c`) is a mature
**CPU** engine; rivoli is a from-scratch **iGPU** engine for Strix Halo. This
compares them to find rivoli's path past 1 tok/s.

## The table

| | **colibri** | **rivoli** |
|---|---|---|
| Compute | **CPU** AVX2/AVX-512-VNNI/NEON + OpenMP; GPU/NPU only for *resident* tensors (streaming stays CPU by design) | **iGPU** (gfx1151): coalesced wave-per-row int4 GEMV, LDS-staged fused MoE |
| Weight memory | `pread` into heap + `fadvise(DONTNEED)`; **single copy**; pin = `posix_memalign` slabs, optional `mlock`; page cache = free L2 | VMM **device-local host-fillable** pin (single copy, device bandwidth); cold VMM slots |
| Cold I/O | coalesced ~19 MB `pread`, **O_DIRECT twin fd**, **PIPE** (8 pthread workers, QD≈8) + **PILOT** cross-layer router prefetch (71.6% recall) | **io_uring O_DIRECT**, per-layer batch (QD≈26 in-layer), one join |
| Cold caching | **3-tier**: mlock pin → per-layer **LRU `ecache`** → OS page cache | pin → cold-reload. **No cross-token cache** (O_DIRECT re-reads every miss every token) |
| Quant | int4 packed 2/byte, **per-row** scales, dequant-on-use; IDOT int8-activation path | int4 per-row (identical), int8 embed/lm_head |
| MoE combine | host top-k; **fixed-order serial** accumulate (no atomics) | host top-k; **fixed-order `moe_reduce`** (per-expert partial, no atomics) — same idea |
| Attention | MLA weight-absorption, hand-written causal softmax (not flash); 57× compressed KV | MLA absorb, token-tiled flash; bf16 latent+roped KV |
| Extras | **MTP speculative decode**, adaptive top-p, DSA indexer, grammar/n-gram spec | none yet |
| Pin ranking | `.coli_usage` freq histogram, written every turn, self-improves; live re-pin heat swap | `.coli_usage` (colibri's), **read-only, no priming** |

## The number that matters

On the **same box class** (AMD Ryzen AI Max+ = Strix Halo):

- **colibri: 0.40 tok/s @ ~71% hit** (its own primed usage, CPU).
- **rivoli: ~0.55 tok/s @ 44.6% hit** (mismatched bench prompt, iGPU).

**rivoli is already ~1.4× faster despite a far worse hit rate** — because the
iGPU int4 GEMV beats colibri's AVX2 CPU kernel. That's the core structural win.

And colibri only *reaches* 1 tok/s when **disk is eliminated by hit rate**: 1.0
tok/s @ 98% hit (430 GB EPYC), 2.06 @ 72.5% hit (M5 Max Metal). Its own stated
bottleneck order: (1) cold disk bandwidth, (2) warm int4 matmul. rivoli's profile
agrees exactly — fetch 53%, mlp 37%.

## What colibri does better (and we should steal)

1. **PILOT — cross-layer prefetch.** colibri runs layer L+1's router on L's
   post-attention state (71.6% recall) and prefetches next-layer experts *while
   computing L*. This is precisely rivoli's "sustain queue depth across layers"
   gap — our per-layer batch drains before the next submits, so we get ~6.6 GB/s
   not the 16 GB/s io_uring can do. **Biggest I/O lever.**
2. **Cross-token cold LRU (`ecache`).** colibri caches recently-streamed experts;
   a repeat hit costs RAM, not NVMe. rivoli's O_DIRECT **bypasses cache and
   re-reads every cold expert every token** — a real regression vs colibri on
   multi-token runs. Need an LRU of warm cold-slots (or drop O_DIRECT's cache
   bypass for repeat experts).
3. **Usage priming.** colibri's `.coli_usage` is written every turn and matches
   its workload → 71–98% hit. rivoli deferred priming and runs colibri's ranking
   read-only → 44.6% on a mismatched prompt. **This is the single biggest reason
   we're not already at 1 tok/s.**
4. **Adaptive top-p** (trim k when disk-bound) and **MTP speculative decode**
   (amortize forwards) — both cut effective cold bytes/token.

## What rivoli does better

1. **GPU compute.** Coalesced int4 GEMV on the iGPU vs CPU AVX2 — the 1.4×-at-
   worse-hit result. colibri deliberately keeps streaming on the CPU ("PCIe
   bottleneck"); on an **APU there is no PCIe** — device memory *is* system RAM —
   so our VMM direct-to-device load sidesteps colibri's core reason to avoid the GPU.
2. **io_uring queue depth** (16 GB/s measured) vs colibri's 8-worker pread PIPE.
3. **Direct-to-device VMM** — single copy, device bandwidth, no CPU matmul.

## Path to 1 tok/s (ranked by leverage)

1. **Usage priming** → 44.6%→~71–85% hit. Cuts cold bytes ~2×; disk stops
   dominating. colibri proves 1 tok/s needs high hit, not more FLOPs.
2. **Cross-token cold LRU cache** — stop re-reading repeat experts every token
   (colibri's `ecache`). Compounds with priming.
3. **Cross-layer io_uring prefetch (PILOT-style)** — sustain 16 GB/s instead of
   6.6; overlap cold reads with compute.
4. **MoE inner-dot GEMV coalescing** (mlp 37%) — the warm compute lever, same
   wave-per-row as the attention GEMV.
5. **MTP speculative decode** — later; multiplies once the above land.

## Cruft to remove (rivoli)

- ✅ prefetch.rs (mmap-warm Prefetcher) + rayon — removed with io_uring.
- Resident tier still loads via **buffered pread** (the +9 s build); switch to
  io_uring O_DIRECT batch or accept it (build-time only).
- `Profile.warm_ns`/`prefetch_ns` — always 0 now; drop from the struct/report.
- The coherent-pin exploration (git `stash@{0}`, env-knob CoherentBuf) — knowledge
  captured in `docs/hip-apu-memory.md`; stash is droppable.
- `main-fast-mmap` branch/tag once `feat/direct-load` merges.
