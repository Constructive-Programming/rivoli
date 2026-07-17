# rolibri — GLM-5.2 MoE decode engine in Rust + ROCm for Strix Halo

Goal: **stable ≥ 1 tok/s** GLM-5.2 (744B int4, 78 layers, 256 experts, top-8)
single-stream decode on rh-anine (AMD Ryzen AI MAX+ 395, Radeon 8060S gfx1151,
128 GB LPDDR5X unified, 16 Zen5 cores). Written from scratch in Rust, tokio-based
streaming architecture from day one, HIP/ROCm compute.

## Why a rewrite gets to 1 tok/s (evidence from the colibri campaign)

Every number below was measured on this box, July 15–17 2026.

1. **The memory bus was never the limit.** 0.72 tok/s = ~8 GB/s of weight
   traffic vs ~230 GB/s available (~3.5%). The C engine is *dispatch-bound*:
   ~920k OpenMP fork/joins per 512-token run (3 matmuls × ~600 experts × token),
   plus single-threaded glue (silu/gather/accumulate ×307k) that at the tuned
   8-thread point costs 2.4× the matmul itself.
2. **Thread scheduling alone was worth +19%.** OMP 32→8 threads: matmul
   70.1s → 10.9s (SMT siblings add zero throughput on batch-1 GEMV and double
   barrier cost). CPU-only @ 8 threads = **0.87 tok/s**, beating every
   GPU-assisted colibri run.
3. **Pinning works.** 64 GB usage-ranked pin ⇒ **91–95% expert hit**; disk
   drops off the critical path when prefetch overlaps (PIPE + pilot). The pin
   ranking file (`.coli_usage`, 2.2M selections) transfers directly.
4. **Per-expert GPU dispatch loses.** Vulkan resident tier, chunked ≤16-expert
   submits: 0.65 tok/s with 38s/128tok of fence-wait — the CPU finished its
   share and *waited on the iGPU*. Small dispatches + submit overhead waste the
   GPU. One fused launch per layer (batch-union of the 8 routed experts) is the
   only shape worth building.
5. **MTP/speculation is out.** MoE expert reads scale with drafted *positions*,
   not forwards (each position routes its own experts). Break-even needed
   tok/fw ≈ 2.8; hardware delivered 1.4–1.9 short, ~1.0 at 512 tokens.
6. **The amdgpu large-GTT kernel bug is real** (BO_VA -12 storm →
   drm_suballoc deadlock → unkillable proc or DEVICE_LOST; 10 reboots).
   Mitigations that must be *designed in*, not bolted on:
   - allocate the device tier **once at startup** and never again (first-alloc
     of a fresh context is far safer; mid-session re-allocations are what die);
   - never share the GPU with llama-swap (its 21 GB GTT load mid-run was the
     aggravator behind most wedges) — refuse to start if
     `mem_info_gtt_used` > threshold;
   - watchdog for D-state + `drm_suballoc` wchan stays in the runner.
7. **Boot params that matter** (already on rh-anine): `ttm.pages_limit=26214400`
   `ttm.page_pool_size=1048576` `transparent_hugepage=madvise` `amd_iommu=off`
   (+6% bandwidth). `amdgpu.vm_update_mode=3` optional seatbelt (needs the
   `amdgpu.` prefix).

### Per-token byte budget (the physics)

- Active experts/token: 75 sparse layers × 8 = ~600 × 18.9 MB int4 = **11.3 GB**
- Dense + attention + shared: ~0.4 GB
- At a conservative 60 GB/s *effective* fused-GEMV streaming rate (GPU or CPU,
  unified LPDDR5X): **~5 tok/s ceiling** → 1 tok/s has 5× margin. The whole
  game is keeping engines fed and dispatches coarse.

## Architecture

Single crate, flat modules (ollama-router layout), edition 2024, tokio.

```
rolibri/
  Cargo.toml
  build.rs              # feature "rocm": compile kernels/*.hip via hipcc
  kernels/
    moe_fused.hip       # per-layer fused batch: int4 dequant × (gate,up) → silu⊙ → down → weighted-accum
  src/
    main.rs             # CLI: rolibri <snapshot> [--ngen N]; tokio runtime (multi_thread, workers = cfg)
    lib.rs
    config.rs           # env-first config (RAM_GB, PIN_GB, DEV_GB, THREADS, …) — printed in full at startup
    snapshot.rs         # safetensors mmap; tensor index; int4/int8 layouts (colibri-compatible snapshot)
    usage.rs            # .coli_usage reader/writer — pin ranking + online accumulation
    pin.rs              # resident store: device tier + host tier, built ONCE at startup
    stream.rs           # the streaming spine: Stream<ExpertBatch> with bounded channels & backpressure
    pilot.rs            # cross-layer prefetch predictor (colibri PILOT_REAL port) as a stream stage
    router.rs           # gate softmax + top-8 (CPU, trivial)
    hip.rs              # minimal HIP FFI (hand-rolled extern "C": hipMalloc/Memcpy/LaunchKernel/Stream/Event)
    moe.rs              # per-layer batch-union assembly → ONE fused kernel launch per layer
    cpu.rs              # AVX-512 VNNI int4 GEMV fallback path; fixed 8-thread pool (NO per-matmul fork/join)
    attn.rs             # MLA attention (CPU first; NPU candidate later)
    engine.rs           # decode loop; per-token profile buckets
    metrics.rs          # colibri-style PROFILE line + submit/launch counters; refuses to report a "GPU run" with 0 launches
  tests/
    kernel_test.rs      # GPU kernel vs CPU reference, per-layer bit-exactness tolerance
    stream_test.rs      # backpressure/ordering of the feed pipeline
  build.sh test.sh .githooks/   # ollama-router conventions; version-guarded build
```

### The streaming spine (tokio from day one)

Decode itself is a tight synchronous loop — async buys nothing inside a token.
Async owns the **feed side**, where colibri serialized and starved:

```
[NVMe pread pool]──►(bounded mpsc, N=32 slabs)──►[decode-ready slabs]
        ▲                                              │
  [pilot predictor]◄──(router outputs, layer l)────[decode loop]
        │                                              │
  [usage accumulator]◄────────────────────────────(selections)
```

- `stream.rs` exposes `ExpertFeed`: `poll_expert(layer, id) -> Ready | Pending`;
  misses are queued to the pread pool (spawn_blocking / io-uring later), decode
  overlaps GPU compute of resident experts with fetch of the ~5% cold set — the
  measured colibri pattern (PIPE + VK_OVERLAP) rebuilt on honest primitives.
- Bounded channels give backpressure for free; no unbounded queue can OOM us.
- The GPU stream is fed via HIP events; CPU cold-set compute joins by channel,
  not pthread_join.

### GPU strategy (gfx1151-specific)

- **One fused kernel launch per layer** (≤75 launches + 1 attention region per
  token, vs colibri's ~4800 submits): kernel takes the batch-union of routed
  experts (device-resident weights, indices, per-row weights) and does
  dequant→gate/up→silu→down→accumulate in LDS-tiled wave32 workgroups.
- Weights live in **unified device memory allocated once** (hipMalloc up to
  DEV_GB; start 32 GB — never device-lost in 3 days of testing — grow to 48/64
  only as stability data accumulates; the rest of the 64 GB pin stays host-side
  for the CPU path).
- rocBLAS/hipBLASLt have no int4 GEMV path worth using at batch-1 — custom
  kernel from the start; correctness pinned by `kernel_test.rs` against cpu.rs.
- `HSA_OVERRIDE_GFX_VERSION` not needed (gfx1151 native in current ROCm), but
  config plumbs HSA env through and prints it.

### CPU path is a first-class citizen, not a fallback

0.87 tok/s @ 8 threads is the bar the GPU must beat per-layer. `cpu.rs` uses a
**fixed worker pool = physical-core count (default 8)**, parallelism *across*
experts (one expert per task), silu/accumulate inside the task — zero global
barriers per token. Engine picks GPU/CPU **per layer** from live measurements
(first 16 tokens are a calibration window), so the faster engine wins
empirically, per this machine, per run.

### NPU (stretch, M6)

XDNA2 spike measured int8 GEMM at 2.78 TOPS (10× one CPU core). Dense mlp +
attention projections are NPU-shaped (static shapes, resident weights). Out of
scope until ≥1 tok/s is banked.

## Milestones — each gated on a measured number

- **M0 — toolchain + skeleton.** Emerge HIP (`dev-util/hip`, hipcc) on
  rh-anine; `cargo check` clean; snapshot mmap + tensor index reads GLM-5.2
  snapshot; config prints full env. *Gate: parse + index the snapshot < 5 s.*
- **M1 — CPU reference decode.** router + cpu.rs + attn.rs + engine.rs; pin.rs
  host tier from `.coli_usage`. *Gate: coherent 32-token output; ≥ 0.8 tok/s
  CPU-only (colibri parity with sane threading).*
- **M2 — HIP kernel correctness.** `moe_fused.hip` vs CPU reference.
  *Gate: max abs error within int4 dequant tolerance on all 75 layers.*
- **M3 — GPU resident tier.** One-shot 32 GB device tier, per-layer fused
  launches, calibration chooser. *Gate: ≥ 1.0 tok/s over 128 tokens, launch
  count ≤ 100/token, zero DEVICE_LOST.*
- **M4 — streaming feed.** pilot.rs + cold-set overlap + usage accumulation.
  *Gate: ≥ 1.0 tok/s sustained over 512 tokens at ≥ 93% hit; disk wait
  ≤ 5% of token time.*
- **M5 — hardening.** GTT co-tenancy guard, wedge watchdog in-process, PROFILE
  metrics + optional OTLP (ollama-router telemetry pattern), build.sh/test.sh,
  git hooks. *Gate: 3 consecutive 512-token runs, variance < 10%, llama-swap
  loaded on purpose → clean refusal, not a wedge.*
- **M6 (stretch) — NPU dense offload; DEV_GB 48/64 stability ladder.**

## Risks

| Risk | Mitigation |
|---|---|
| ROCm gfx1151 maturity (known PERMISSION_FAULT reports upstream) | one-shot allocation; calibration chooser means CPU path always shippable; kernel bug watchdog |
| Same amdgpu underneath → BO_VA -12 class bugs reachable from HIP too | DEV_GB=32 default; single allocation; GTT guard; grow tier only with stability data |
| hipcc/Gentoo packaging friction | M0 is exactly this, nothing else blocks on it; CPU milestones (M1) proceed in parallel |
| int4 layout mismatch vs colibri snapshot | snapshot.rs ports colibri's exact packing (glm.c `pack_int4`); kernel_test locks it |

## Conventions

- **No `unwrap`/`expect` outside tests** (workspace lint, deny).
- Every benchmark/report prints its **full env + config line first** and its
  engine engagement counters (launches, submits) — a "GPU number" with zero
  launches is reported as CPU fallback, loudly.
- ollama-router repo conventions: flat modules, `tests/`, `.githooks`
  pre-commit (fmt+clippy) / pre-push (test), version-guarded `build.sh`.
