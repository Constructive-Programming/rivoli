# rivoli — GLM-5.2 MoE decode engine in Rust + ROCm for Strix Halo

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
   - **sole-tenant rule**: never share the GPU with another process (a 21 GB
     foreign GTT allocation landing mid-run was the aggravator behind most
     wedges) — refuse to start if `mem_info_gtt_used` shows another tenant;
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

### The four bottlenecks — the design wins by removing them

Every module decision answers to one of these; anything that serves none of
them is cut.

1. **NVMe bandwidth** — never on the critical path: 95%+ residency via the
   usage-ranked pin, misses prefetched by the pilot ahead of need, demand
   fetches overlapped with resident compute. Disk time budget: ≤5% of a token.
2. **RAM bandwidth** — the only *honest* limit (11.3 GB/token). Spend it once:
   weights stream through exactly one engine per layer, no double reads, no
   host↔device copies (unified memory, zero-copy by construction).
3. **Synchronization points** — colibri died here (~920k barriers/run,
   4800 GPU submits, one big fence per layer-chunk). Budget: ≤100 kernel
   launches and ≤1 join point per token; feed↔decode contact is lock-free
   bounded channels only.
4. **Synchronization cost** — what remains must be cheap: no fork/join pools
   (persistent workers), no fence spins (HIP events + async streams), no
   SMT-oversubscription (pool = physical cores, sized once at startup).

## Zero-knob operation

**No environment variables. One flag.**

```
rivoli <snapshot-dir> -bench <tokens>     # benchmark mode: decode N tokens, print PROFILE
rivoli <snapshot-dir>                     # (later) OpenAI-compatible server mode
```

Everything else is **auto-discovered at startup** and printed as the first
line of the run:

- **Memory budget** = `MemAvailable` − **16 GB OS reserve** − engine overhead
  (dense weights ~9.6 GB, KV cache, slab pool, dispatch scratch — computed
  from the snapshot config, not guessed). What remains is the expert pin,
  filled byte-accurately from the usage ranking, hottest first.
- **Device tier** = as much of the pin as the GPU can hold, from live free
  GTT/unified memory at startup (single allocation, sole tenant already
  verified). A stability ladder may cap it from *observed* device-loss data,
  never from a config knob.
- **Feed pool** (tokio workers + pread tasks) = physical cores ÷ 2 (the
  measured optimum; SMT-logical count is the proven pathology). Detected,
  not configured. The CPU never computes experts — it routes, samples, and
  keeps the GPU fed.
- **Usage ranking** = `<snapshot>/.coli_usage`, read at startup, accumulated
  during the run, written back at exit.

## Architecture

Single crate, flat modules (ollama-router layout), edition 2024, tokio.

```
rivoli/
  Cargo.toml
  build.rs              # feature "rocm": compile kernels/*.hip via hipcc
  kernels/
    moe_fused.hip       # per-layer fused batch: int4 dequant × (gate,up) → silu⊙ → down → weighted-accum
  src/
    main.rs             # CLI: rivoli <snapshot> [-bench N]; tokio runtime (multi_thread, workers = discovered)
    lib.rs
    config.rs           # zero-knob auto-discovery (memory budgets, device tier, CPU pool) — printed in full at startup
    snapshot.rs         # safetensors mmap; tensor index; int4/int8 layouts (colibri-compatible snapshot)
    usage.rs            # .coli_usage reader/writer — pin ranking + online accumulation
    pin.rs              # resident store: device tier + host tier, built ONCE at startup
    stream.rs           # the streaming spine: Stream<ExpertBatch> with bounded channels & backpressure
    pilot.rs            # cross-layer prefetch predictor (colibri PILOT_REAL port) as a stream stage
    router.rs           # gate softmax + top-8 (CPU, trivial)
    hip.rs              # minimal HIP FFI (hand-rolled extern "C": hipMalloc/Memcpy/LaunchKernel/Stream/Event)
    moe.rs              # per-layer batch-union assembly → ONE fused kernel launch per layer
    attn.rs             # MLA attention (HIP kernels; NPU candidate later)
    engine.rs           # decode loop; per-token profile buckets
    metrics.rs          # colibri-style PROFILE line + submit/launch counters; refuses to report a "GPU run" with 0 launches
  tests/
    reference.rs        # scalar CPU reference impl (TEST-ONLY: kernel correctness oracle)
    kernel_test.rs      # GPU kernel vs scalar reference, per-layer tolerance
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

**Slab ownership (M4 contract, decided by review).** `ExpertSlab` must NOT own
its bytes (no `bytes::Bytes`, no `Vec<u8>`): an owned host buffer is a
per-expert heap allocation *and* not GPU-visible, so it would force a copy into
the unified pool — exactly colibri's copy-pool, the thing bottleneck #2
forbids. The feed channel carries a **pooled slab handle** — an index/offset
into the one `hipHostMalloc` coherent pool allocated at startup — that NVMe
`pread` writes into directly and the kernel reads in place; on drop the handle
returns to a free-list. The bounded channel then bounds slab *occupancy*, not a
stream of heap buffers. (The M0 skeleton `stream.rs`/`metrics.rs` were removed
after review — they had no consumer and encoded the wrong, owned-bytes shape;
they return in M3/M4 built against a real one.)

- the feed exposes `poll_expert(layer, id) -> Ready | Pending`;
  misses are queued to the pread pool (spawn_blocking / io-uring later), decode
  overlaps GPU compute of resident experts with fetch of the ~5% cold set — the
  measured colibri pattern (PIPE + VK_OVERLAP) rebuilt on honest primitives.

**NVMe → iGPU direct (the unified-memory play).** The slab pool is allocated
once as *cacheable, coherent* GPU-visible unified memory (`hipHostMalloc`,
APU-coherent path). NVMe reads land **directly in those slabs** and the fused
kernels consume them in place — the byte path is NVMe DMA → LPDDR5X → GPU
read, with **zero intermediate copies** (colibri paid a memcpy per cold expert
into its copy-pool, and its design-c attempt at direct pread died on
write-combined/uncached Vulkan mappings — the HIP coherent path is the fix
this hardware actually offers). Refinement inside M4: io_uring with O_DIRECT
into the registered slabs, so cold-tail experts skip the page cache entirely
(they are rarely re-read at 95% residency; skipping saves the page-cache
copy and halves the RAM traffic of every miss).
- Bounded channels give backpressure for free; no unbounded queue can OOM us.
- The GPU stream is fed via HIP events; cold-set fetches land in unified-memory
  slabs the same fused kernels consume — completion joins by channel, never
  pthread_join.

### GPU strategy (gfx1151-specific)

- **One fused kernel launch per layer** (≤75 launches + 1 attention region per
  token, vs colibri's ~4800 submits): kernel takes the batch-union of routed
  experts (device-resident weights, indices, per-row weights) and does
  dequant→gate/up→silu→down→accumulate in LDS-tiled wave32 workgroups.
- Weights live in **unified device memory allocated once**, sized from live
  free memory at startup (see Zero-knob operation); the remainder of the pin
  stays in host-side unified slabs the GPU reads zero-copy on demand. A
stability ladder caps the device tier only from
  *observed* device-loss events, never from configuration.
- rocBLAS/hipBLASLt have no int4 GEMV path worth using at batch-1 — custom
  kernel from the start; correctness pinned by `kernel_test.rs` against cpu.rs.
- `HSA_OVERRIDE_GFX_VERSION` not needed (gfx1151 native in current ROCm).

### One engine — no CPU fallback

The GPU is the only compute path. Cold (non-resident) experts are fetched into
unified-memory slabs the GPU reads zero-copy, and the same per-layer fused
kernel consumes resident and freshly-fetched experts alike — one code path,
no engine chooser, no divergent numerics. The CPU's jobs are exactly: routing
(softmax/top-8), sampling, and driving the feed pipeline. A scalar CPU
implementation exists **only in tests** as the correctness reference for the
kernels (colibri's measured 0.87 tok/s CPU number remains the external
sanity bar the GPU must clear, but it is not shipped code). Zero kernel
launches in a run is a **hard error**, not a fallback — the silent-CPU runs
of the colibri campaign are impossible by construction.

### NPU (stretch, M6)

XDNA2 spike measured int8 GEMM at 2.78 TOPS (10× one CPU core). Dense mlp +
attention projections are NPU-shaped (static shapes, resident weights). Out of
scope until ≥1 tok/s is banked.

## Weight format (decided)

Reuse colibri's per-row int4 snapshot for now. Verified on this hardware
(clang builtin gating, 2026-07-17): **MXFP4/MXINT4 microscaling matmul is
CDNA4-only** (`gfx950-insts`); gfx1151 (RDNA 3.5) has plain int4 WMMA
(`gfx11-insts`) but no block-scale operand, and WMMA is a matrix instruction
that a batch-1 GEMV can't fill anyway — so MX buys us nothing here (and its
4.25 bits/weight is *worse* for our bandwidth ceiling than per-row int4's
~4.0). **Follow-up, after ≥1 tok/s is proven:** a converter from HF GLM-5.2's
fp8 (e4m3, block-scaled 128×128) to **group-scaled int4** (group ~32–128) —
an accuracy gain at the same ~4 bits/weight, no bandwidth penalty. int4 WMMA
is revisited only for the future batched/server path (S>1 rows share weights).

## Milestones — each gated on a measured number

- **M0 — toolchain + skeleton.** HIP (hipcc) installed on rh-anine;
  `cargo check` clean; snapshot mmap + tensor index reads GLM-5.2 snapshot;
  auto-discovered config printed as first line. *Gate: parse + index the
  snapshot < 5 s.*
- **M1 — reference decode (test-only path).** router + engine skeleton + pin.rs
  from `.coli_usage`; scalar reference impl in `tests/`. *Gate: coherent
  32-token output through the reference path (correctness, not speed).*
- **M2 — HIP kernel correctness.** `moe_fused.hip` + attention kernels vs the
  scalar reference. *Gate: max abs error within int4 dequant tolerance on all
  75 layers.*
- **M3 — GPU resident tier.** One-shot auto-sized device tier, per-layer fused
  launches, calibration chooser. *Gate: ≥ 1.0 tok/s over 128 tokens, launch
  count ≤ 100/token, zero DEVICE_LOST.*
- **M4 — streaming feed.** pilot.rs + cold-set overlap + usage accumulation.
  *Gate: ≥ 1.0 tok/s sustained over 512 tokens at ≥ 93% hit; disk wait
  ≤ 5% of token time.*
- **M5 — hardening.** Sole-tenant guard, wedge watchdog in-process, PROFILE
  metrics + optional OTLP (telemetry pattern from the router project),
  build.sh/test.sh, git hooks. *Gate: 3 consecutive 512-token runs, variance
  < 10%; a foreign GPU tenant present at startup → clean refusal, not a wedge.*
- **M6 (stretch) — NPU dense offload; device-tier stability ladder toward the
  full pin.**

## Milestone status — 2026-07-20 (post branch-integration)

All feature branches merged into `main` (36 commits); `main` == `feat/direct-load`.
Redundant branches (`moe-gemv`, `xlayer-prefetch`, `prefetch`, `misa-gpu-routing`)
deleted; the MTP salvage branches are archived as `deadend/*`.

- **M0 toolchain + skeleton — DONE.** hipcc live on rh-anine; snapshot indexes
  ~118k tensors in < 0.15 s.
- **M1 reference decode — DONE.** scalar reference path + `pin.rs` from
  `.coli_usage`; coherent output verified.
- **M2 HIP kernel correctness — DONE.** `moe_fused` + attention kernels vs the
  scalar reference within int4 tolerance. Since extended with pluggable
  attention modes (Dense / StreamingLLM / DSA), MISA 8-of-32 indexer head
  routing, a device-side DSA indexer, and fp8-e4m3 latent-KV kernels — all
  code-reviewed.
- **M3 GPU resident tier — MET (warm).** One-shot auto-sized device tier
  (`--max-mem`, default `free − 8 GiB`), per-layer fused launches, coalesced
  wave-per-row GEMV (attn 1450→155 µs; mlp 2.6×), device-side argmax, zero-copy
  upload. **Warm 256-tok = 1.05 tok/s** overall, warm windows 1.0–1.32, no
  DEVICE_LOST. (The strict "1.0 over 128 overall" gate is borderline at 0.87;
  it clears at 256.)
- **M4 streaming feed — MET (2026-07-20).** *Landed:* io_uring O_DIRECT
  cold-expert streaming (`817744c`, beats fast-mmap; the per-layer batched
  submit-then-drain already fills the NVMe queue to QD≥4), unified VMM+pread
  direct-to-device load, pinned-host bounce for cold reads (default), cross-layer
  expert prefetch (default, depth 2 — validated optimal), ARC/LRU/2Q cache
  policies (ARC default), adaptive routed-expert LRU + warm-start seeding from
  `.coli_usage`. **512-token gate: 1.32 tok/s at 90.9% hit, drain-wait ~1% of
  wall — PASS** (target ≥ 1.0, disk ≤ 5%).
  The blocker was NOT throughput (warm decode already sustains 1.3–1.5 tok/s);
  it was a **VMM over-commit bug** that deterministically NaN'd decode at token
  ~290 and so capped every prior run at ≤ 256 tokens. `hipMemGetInfo` reports
  ~100 GiB free but the driver won't durably back a ~92 GiB device footprint
  under live decode; high VMM pool slots got reclaimed and streamed experts read
  back NaN. Fixed by raising `OS_RESERVE` 8→26 GiB so the default footprint
  (~74 GiB) stays in the verified-safe zone (see `config.rs`). This retires
  commit `16bae7f`'s "grow pool to ~92 GiB" as unsafe — it was never run past
  256 tokens. Footgun still stands: io_uring→VMM EFAULTs on NFS fds (kernel
  6.18.38) — the model must live on **local** `/var/db`, not `/swarm`.
  Pool sizing is settled — no memory left to recover. The ~26 GiB reserve is not
  waste: a larger footprint either corrupts (≥ ~92 GiB total) or thrashes via OS
  page reclaim (~84 GiB total measured *slower*, 0.98 vs 1.31 tok/s at lower hit),
  so ~64 GiB is the throughput-optimal pool. Sharding into multiple VMM handles
  would not help — the ceiling is total device footprint, not per-handle size (a
  single 82 GiB handle commits+reads back fine in isolation). And the slots are
  already byte-optimal: int4 weights with 0.13% O_DIRECT alignment padding
  (~24 KiB per 18 MiB slot); fp8 would *double* each expert (int4 is 4-bit, fp8
  8-bit) and halve capacity. The only lever left for more resident experts is a
  sub-int4 quant, which is a quality tradeoff, not a layout change.
- **M5 hardening — MET (2026-07-20).** Sole-tenant guard
  (`device::DeviceTier::guard_sole_tenant`, refuses to start on foreign GTT
  > 1 GiB — the gate's "clean refusal, not a wedge"), in-process **wedge
  watchdog** (`src/watchdog.rs`: a background thread aborts with a clear message
  if no token lands for 60 s, since a hung `hipDeviceSynchronize` can't be caught
  in-loop), PROFILE metrics, `build.sh`/`test.sh`, git hooks, and the `--kv-fp8`
  / `--direct-io` / `--cache-policy` / attention-mode knobs. **Stability gate: 3
  consecutive 512-token runs = 1.31 / 1.31 / 1.31 tok/s at 90.9% hit, 0.0%
  variance — PASS** (target < 10%). **OTLP span export** (`src/telemetry.rs`,
  mirroring the ollama-router pattern): opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT`,
  batch-exports a `rivoli.decode` span (tokens / tok_per_s / hit_pct attributes +
  the PROFILE/summary events) over OTLP-HTTP to the fleet Tempo pipeline; unset ⇒
  log-only. **M5 CLOSED** — all items done.
- **M6 stretch — NPU dense offload — not started.** The colibri-npu spike proved
  int8 GEMM at 2.78 TOPS (~10× CPU); nothing wired into rivoli yet.

**MTP / speculation — CLOSED (negative result), confirming item 5 above.** Built
and measured all three salvage ideas at 256 tokens; all lose. Decode is
bandwidth-bound cold (+16% bytes/tok) but *compute*-bound warm (the MTP draft is
a full extra 256-expert MoE layer every round → `other` 638 ms/tok at 84%
accept), so speculation loses in both regimes. The cache — not speculation — is
the lever that crosses 1 tok/s. Full write-up + numbers in `docs/mtp.md`; branches
`deadend/{mtp,spec-overlap-gate,spec-warm-budget,spec-union-tree}`.

## OPEN: attention modes are unmeasured where they differ (followup, 2026-07-20)

The `--attn dense|streaming|dsa|misa` knobs and `--kv-fp8` are shipped,
code-reviewed and kernel-validated, but **no mode has ever been benchmarked or
quality-checked in the regime where the modes actually differ.** Recording it
here because these gaps were previously written down only in
`docs/review-log.md`, which lives on `deadend/*` and is invisible from `main`.

What is NOT known (do not cite a mode ranking until this runs):

- **No valid throughput comparison exists.** The one recorded figure —
  dsa/misa 1.28 vs dense 0.80 tok/s @ 512 (`git show
  8d267c5:docs/review-log.md`) — is self-labeled "unconfirmed", collapses dsa
  and misa into ONE number, and attributes the gain to the **MoE prefetch path,
  not attention** (hit 79→91→95%, `attn` flat ~204 ms across all modes). The
  `--no-prefetch` A/B that would isolate the cause was interrupted at ~112/512.
- **`streaming` has never been measured at all** — no throughput or quality
  number exists for it anywhere.
- **The sparse regime has never been run end-to-end.** `index_topk` = 2048
  (model `config.json`), so below 2048 context every mode takes the dense
  fallback and is bit-identical by construction — which is all
  `tests/attn_modes.rs` certifies. Benchmarks at 64/128/512 tokens say NOTHING
  about mode choice.
- **No quality measurement in the sparse regime.** No perplexity, no
  cross-check vs the HF reference. The MISA-vs-DSA IoU 0.84
  (`tests/attn_modes.rs:151`) is selection overlap on synthetic activations;
  its own comment says "true routing QUALITY is a real-decode long-context
  eval, not this harness's job". (An earlier 0.996 IoU claim was retracted in
  `f58d70e` as near-vacuous.)
- **`--kv-fp8` is uncalibrated and unvalidated at long context.** Measured only
  as within 5 % of bf16, dense, short context. vLLM reported MLA-fp8 accuracy
  drift without calibration.

**Run dsa/misa against `~/glm52-snap`, NOT the bare colibri dir.** The colibri
converter skipped the DSA indexer (AGENTS.md "Weights the snapshot doesn't
have"), so `/var/db/llama-server/glm52-colibri-int4` alone bails with `required
tensor model.layers.0.self_attn.indexer.wk.weight not found in snapshot`.
`~/glm52-snap` is the documented symlink-overlay: symlinks to all 147 real
shards + the range-extracted `out-idx-00000.safetensors` (412 MB, 110 indexer
tensors). Against it, `--attn misa` indexes 118587 tensors vs 118478 and runs.
Note the repo's `.snapshot_path` still points at the bare colibri dir, so any
tooling that reads it gets the indexer-less path.

### Measured 2026-07-20: at 128 tok the modes are indistinguishable, and the fp8 "win" is a routing artefact

16 runs (4 modes × `--kv-fp8` on/off × 2 reps, 128 tok, `~/glm52-snap`).
Reproducibility is tight (rep1 vs rep2 within 0.01 tok/s), so the spreads below
are real — but they are **not** caused by what they appear to be.

| config | tok/s | hit % | miss | attn ms |
|---|---|---|---|---|
| dense / streaming / dsa / misa, fp8 **off** | 0.85–0.86 | 78.1–78.3 | 122–123 | 173–174 |
| dense + fp8, streaming + fp8 | 0.89–0.90 | 79.9 | 112 | 173–176 |
| dsa + fp8, misa + fp8 | 0.81–0.82 | 76.6 | 132 | 173–175 |

Two things this establishes:

1. **All four modes are identical below `index_topk`** — with fp8 off, every
   mode produced byte-identical output (md5 `285c4923`) at identical cost. The
   dense-fallback equivalence is now confirmed end-to-end on the real snapshot,
   not just by construction in `tests/attn_modes.rs`.
2. **`attn` is FLAT at 173–176 ms across all 16 runs** — invariant to mode and
   to fp8. Meanwhile `ms/miss` is constant at 4.78–4.90, so
   `fetch ≈ miss_count × 4.85 ms`, and tok/s is a strict monotonic function of
   the expert cache hit rate. **The entire tok/s spread is expert-fetch, not
   attention.**

Mechanism: fp8 is lossy, so it perturbs the sampled token stream (different
output md5), which routes to different experts, which changes the cache hit
rate, which changes fetch time. The sign flips by mode — fp8 *helps*
dense/streaming (+1.6 pp hit) and *hurts* dsa/misa (−1.5 pp hit) — which is
conclusive that this is a routing draw, not a bandwidth property. At 128 tok the
KV cache is ~133 rows, far too small for fp8's 656-vs-1152 B/token/layer saving
to register, and the flat `attn` confirms it does not.

**Do not read "dense+fp8 = 0.90" as fp8 being faster.** This is the same
confound that made the un-merged 512-token dsa/misa figure uninterpretable.
Any future A/B must hold the token stream fixed (or average over many prompts)
before attributing a tok/s delta to an attention or KV change.

The run that would settle it:

1. Context **> 2048** (the whole point — ≥ 8k, ideally toward 40k) so the sparse
   paths actually engage. Fixed prompt, fixed token count, all four modes.
2. **`--no-prefetch` as a control arm** alongside the default prefetch arm, to
   separate attention cost from the prefetch hit-rate effect that confounded the
   original measurement. Report the PROFILE `attn` ms/tok, not just tok/s.
3. **A quality arm**: agreement vs `--attn dense` output at the same context
   (note dense itself is "mildly out-of-distribution" beyond index_topk per
   `attn.rs:26`, so it is a reference, not ground truth).
4. dsa and misa reported **separately**; `--kv-fp8` crossed with each mode, since
   fp8 composes with all of them and its KV-bandwidth saving (656 vs 1152
   B/token/layer) only pays off once the cache is large.

Until then `--attn dense`, `--kv-fp8` off remain the defaults — chosen as the
conservative exact-model path, NOT because they won a benchmark.

## Measured bottlenecks & the coherent-pin fix (2026-07-18, real snapshot)

Profiled the M3 (4/4) engine on the LOCAL snapshot (`/var/db/llama-server/
glm52-colibri-int4`, not the NFS `/swarm/storage` copy — the network was masking
everything). Per-token buckets, "sky is blue", 48 GiB pin:

- **COMPUTE-bound, not I/O-bound.** attn 36–39% | fetch 34–38% | mlp 16% |
  lmhead 9%. Local storage cut per-miss 86 ms → 4.6 ms (NFS was the whole story).
- **#1 bottleneck = the attention int4 GEMV** (row-per-thread, uncoalesced),
  hit-rate-independent. The fused MoE kernel is 2× more efficient at 8× the work,
  because it's LDS-staged/batched and the attention projections aren't. → a
  coalesced/LDS int4 GEMV (the D3 tune) is the single largest tok/s lever.
- **Routing is CORRECT** — 0.2 % of selections absent from `.coli_usage`, 0 misses
  below the pin cutoff. The 45 % hit is pure workload BREADTH (this prompt routes
  to rank ~9000). Pin-vs-hit curve: 45/53/67 % at 48/64/94 GiB (the 94 GiB
  one-shot alloc SURVIVED — no wedge, de-risks large pins). 85 % would need
  ~10 000 experts ≈ 189 GiB (2× the device) — **pinning cannot reach 85 %; a
  workload-matched priming pass can** (routing being correct is what makes
  priming clean).

**THE MEMORY FIX (apply to the resident pin, not just the cold feed).** Per-expert
bytes are already minimal and identical to colibri (gate+up+down int4 @ 4 bits +
~40 KB f32 scales ≈ 18.9 MB; int4 is the floor). The waste is that the M3 pin
DOUBLE-STORES on the unified APU: `pread`/mmap lands the expert in the **page
cache** (host copy) and then `hipMemcpy` H2D duplicates it into the `hipMalloc`
**device tier** — ~2× LPDDR5X per pinned expert. colibri (CPU) used the single
host copy AS its pin. This is why the 94 GiB tier starved the page cache and
per-miss ROSE to 6.17 ms (device tier + duplicate cache saturated 128 GB → cold
set thrashed).

Fix: **pin into `hipHostMalloc` coherent unified memory** (the "NVMe → iGPU
direct" design, already specced for the cold feed — now extend it to the resident
pin). `pread` NVMe straight into the coherent slab; the GPU reads it in place
(APU-coherent); no device carveout, no page-cache duplicate → **~halves RAM per
expert**, so ~2× more experts fit the same 128 GB AND the page cache stays free
for the cold set (fast misses). Interim cheap version: `madvise(MADV_DONTNEED)`
each pinned expert's mmap range right after the H2D copy to reclaim the duplicate
(~40 GB at a 48 GiB tier). This does NOT replace the GEMV work — it makes the pin
memory-efficient so a primed/matched pin can hold the working set.

Priorities from the data: (1) coalesced int4 GEMV kernel (biggest tok/s lever,
workload-independent); (2) coherent-memory pin (~2× RAM efficiency); (3) usage
priming to lift hit past the 67 % pinning ceiling toward 85 %.

### Update 2026-07-18 (post-D3) — reconciling the coherent pin & priming with evidence

**(1) DONE.** D3 coalesced wave-per-row GEMV shipped (`a89d7ea`): attn ~1450→155
ms/tok (~9×), lm_head ~350→7 ms, wall ~3900→2140 ms/tok (~1.8×). This **moved the
bottleneck**: the engine is now **I/O-bound, not compute-bound** — per-token
`warm` 904 ms (42 %) + `fetch` 439 ms (21 %) = **63 % cold-expert paging**; `mlp`
616 ms (29 %, the MoE fused kernel's still-uncoalesced inner dot — the next
compute lever); `attn` 155 ms (7 %). ~0.47 tok/s wall.

**(2) coherent pin — DO NOT schedule as a plain task; it's an UNMEASURED
HYPOTHESIS gated behind priming.** What we actually ran was the coherent **cold
pool** (`hipHostMalloc` on the cold-fetch slots), NOT the resident pin:
- Result: **`mlp` 617→675 ms, +9 % REGRESSION** → git `stash@{0}` ("revisit for
  async streaming"). Cause: coherent reads measured **207 GB/s vs 220 GB/s device
  (~6 % slower)**, and cold misses are *read*-bound (GPU reads the expert in place
  during the MoE dot), not copy-bound — so the tax hit compute with no offsetting
  H2D saving.
- The 6 % read tax applies to the **resident pin too**, which is the most
  read-heavy memory in the engine (every token, every GEMV + MoE). That
  experiment ran while we were compute-bound, where the tax was clearly bad.
- Post-D3 the calculus *may* have flipped (now I/O-bound: 6 % of ~770 ms compute
  ≈ 46 ms vs halving RAM/expert → ~2× more resident experts → fewer of the 55 %
  misses → cuts into 1343 ms warm+fetch). **This is a hypothesis to MEASURE.**
- **UNGATED FROM PRIMING (2026-07-18).** Earlier this was "test only after
  priming"; that coupling is removed — the coherent approach is being re-measured
  on its own merits now that the profile is I/O-bound, independent of whether we
  ever prime. The `madvise(MADV_DONTNEED)` interim (`5b2212b`) is the safe half
  and is already shipped.
- **RE-RAN the coherent COLD pool post-D3 (2026-07-18) — now a WASH, not a
  regression.** vs D3 baseline: `fetch` 439→**368 ms** (1.30→1.09 ms/miss, the
  host-memcpy fill skips 3× H2D/miss) but `mlp` 616→**672 ms** (+9 %, the 6 %
  coherent read tax on the cold-expert dot); wall 2140→**2135 ms** (net ≈0). The
  6 % tax is RECONFIRMED. This isolates the TAX (real, ~6 % on coherent reads) but
  NOT the resident-pin's RAM-fit benefit — the cold pool is read once/miss WITH a
  fill offset; the RESIDENT pin is read every token with NO offset, so A3 would
  pay the tax on all hit-reads and must justify it purely by fitting ~2× experts
  → higher hit → less I/O. **A3 (resident pin) still needs its OWN build+measure.**
  Conflict-resolved coherent artifact preserved in `git stash@{0}` as the base.
- **TAX-ELIMINATION EXPERIMENT (2026-07-18) — both clever levers NULL.** Ran a
  4-way A/B (32-tok, same pin) via env knobs (`RIVOLI_COH_FLAGS`,
  `RIVOLI_HUGEPAGE`, in stash@{0}): baseline `mlp` 616 | fine-grained coherent 672
  | **L1 non-coherent** (`hipHostMallocNonCoherent` 0x80000002) **678** | **L2
  coherent+hugepage** (`madvise(MADV_HUGEPAGE)`) **673** | both 676. The `mlp` tax
  is INVARIANT (672–678, never near 616) → it is NOT fine-grained-coherence (L1
  disproved) and NOT TLB/page-size (L2 disproved; madvise likely a no-op on
  non-THP HSA memory). The behavioral read was "the tax can't be engineered away"
  — **that conclusion was WRONG (corrected via kernel source, below).**
- **SOURCE-CONFIRMED MECHANISM (2026-07-18, /usr/src/linux amdgpu/amdkfd on
  rh-anine).** gfx1151 = GC IP 11.5.1. `amdgpu_amdkfd_gpuvm.c::get_pte_flags`
  sets `AMDGPU_VM_MTYPE_DEFAULT` → `gmc_v11_0_get_vm_pte` maps DEFAULT/NC →
  **MTYPE_NC (L2-CACHEABLE)**, BUT its tail forces **MTYPE_UC (uncached, L2
  BYPASSED)** when the BO has `AMDGPU_GEM_CREATE_COHERENT/EXT_COHERENT/UNCACHED`
  (set from the KFD `COHERENT` alloc flag, line ~1756). So fine-grained coherent
  host mem → MTYPE_UC → the ~6 % tax: CONFIRMED, definitive. `svm_range_get_pte_
  flags` (managed path) agrees: gfx11 falls to `default: coherent?UC:NC`.
- **RESOLVED via full userspace trace (ROCm/clr amd-staging) + live topology.**
  Traced hipHostMallocNonCoherent end to end: ihipHostMalloc clears
  CL_MEM_SVM_ATOMICS → getHostMemorySegment (rocmemory.hpp) ATOMICS==0 → kNoAtomics
  → getHostMemoryPool (rocdevice.cpp) kNoAtomics → coarse_grain_pool IF it exists →
  `rocminfo` confirms the CPU agent DOES expose a COARSE GRAINED ~128GB pool. So L1
  DID land in the coarse (cached, MTYPE_NC) pool — and still didn't help. **The tax
  is NOT coherence-grain: our own data shows ALL host-pool variants 672–678 ms
  regardless of UC vs NC, only the DEVICE pool (hipMalloc) hits 616.** The real
  axis is SYSTEM-memory domain vs DEVICE-LOCAL domain: baseline = GPU-agent
  device-local pool (MTYPE_RW, GPU-owned, large pages, no snoop); every hipHostMalloc
  grain = CPU-agent system pool (MTYPE_NC/UC + AMDGPU_PTE_SYSTEM + PTE_SNOOPED, 4 KiB
  pages). The coherence flag flips UC↔NC within the system domain but can't cross
  into device-local, so it never recovers bandwidth. **CONCLUSION (now source-
  grounded): no hipHostMalloc flag eliminates the tax — A3's "single host copy read
  in place" is inherently a system-domain read that pays ~9%; only device-local
  (hipMalloc = the double-store) avoids it.** You can't have both single-copy AND
  device bandwidth from hipHostMalloc. Narrow escape hatches left, uncertain payoff:
  large-page system mapping via hipHostRegister over MAP_HUGETLB (attacks only the
  TLB component), or VMM/hipMemPool APIs to make a device-local pool host-fillable.
  Epistemic note: first "fundamental, no flag fixes it" was asserted w/o source
  (right direction, wrong stated mechanism); kernel source alone over-corrected to
  "avoidable"; the full trace confirms no-flag-fixes-it for the RIGHT reason
  (system-vs-device domain). Read the source before saying "definitively".
- **VMM ESCAPE HATCH — DONE + WORKS (2026-07-18).** `device::VmmBuf` (C ABI shim
  `kernels/vmm.hip`; probe `docs/probes/vmm_probe.cpp`): `hipMemCreate` device-local
  physical + `hipMemSetAccess(hipMemLocationTypeHost)` → CPU fills in place, GPU
  reads at **220 GB/s = device speed** (vs 215 hipHostMalloc), verified CPU→GPU
  coherent. So the "can't have both single-copy AND device bandwidth" wall is FALSE
  for VMM — it crosses the domain boundary hipHostMalloc can't. **Caveat (measured,
  decisive):** device bandwidth holds ONLY write-once-read-many; interleaved
  CPU-rewrite+GPU-read halves it (220→112 GB/s). Wired into the cold pool as a
  proof: correct/coherent decode, but `mlp` stayed 673 (cold = refill-every-miss =
  the 112 antipattern), kept only for the fill-offset net win (wall 2086 vs 2136,
  fetch 447→364). **The real payoff is the RESIDENT PIN (A3): filled once, read all
  run = the 220 path → single host copy (drop BOTH the mmap page-cache dup and the
  device-tier hipMalloc slab, no double-store) + device bandwidth + ~2× experts
  fit.** Next concrete A3 step: convert the resident DeviceTier to VMM-backed,
  pread experts straight in. See docs/hip-apu-memory.md.
- Robust lever-independent win regardless = the fetch-fill offset (`fetch`
  447→~365 ms, host memcpy skips 3× H2D/miss). A3 arithmetic still favors
  build-and-measure: ~+46 ms/tok tax (IF unavoidable) vs ~500 ms potential from
  halving RAM → fitting the 67 %-hit set in ~48 GiB → cutting the 55 % miss that
  drives 1339 ms warm+fetch. If the MTYPE_NC path can be forced, the tax may
  largely vanish and A3 gets stronger.

- **CORRECTION + REVIEW OUTCOME (2026-07-18) — supersedes the "~2× experts fit"
  claims above (lines ~284, ~329, ~384) and the "resident-tier VMM is the payoff"
  bullet.** Two errors, now fixed:
  1. **"VMM resident pin → ~2× experts fit" is WRONG — double-counting.** The
     resident tier is ALREADY single-copy: it's a device-local `hipMalloc` slab and
     we already `madvise(MADV_DONTNEED)` the mmap page-cache dup after each H2D
     (commit 5b2212b). Steady state = 1× device-local. VMM device-local is the SAME
     bytes in the SAME domain → same RAM, same 220 GB/s read. So converting the
     resident tier to VMM is **perf-neutral** (no RAM saving, no bandwidth gain),
     not a 2×-experts payoff. The "~500 ms potential" above assumed a RAM halving
     that madvise already banked.
  2. **VMM delivers NO unique net perf win in the current architecture.** Cold pool
     = rewrite-per-miss → can't use VMM's device bandwidth (112 GB/s interleaved);
     its ~2% wall win is the fill offset, which a plain coherent `hipHostMalloc`
     buffer gives too. Resident pin = already device-local → VMM redundant. So
     VMM's unique property (device-BW + host-fill) isn't load-bearing anywhere here.
  - **DECISION — resident-tier VMM conversion: NOT doing it.** Only upside is ~10
     fewer lines + unifying the two fill paths (H2D placement + host-memcpy cold)
     onto one — a modest code dedup, perf-neutral, on the hot core allocator. Not
     worth the churn/re-test. (Ponytail review independently: don't unify
     VmmBuf/DeviceBuf into a trait/enum either — the load-bearing methods diverge.)
  - **DECISION — cold-pool VmmBuf: keep as-is (marginal net win, primitive built,
     correct/coherent), but CONDITIONAL.** It's the "wrong tier" for VMM; if we ever
     want to simplify, revert it to a coherent `hipHostMalloc` buffer (same fill
     offset, less machinery) and drop `VmmBuf`/`vmm.hip` as YAGNI. Trigger: if no
     write-once VMM consumer ever lands.
  - **REVIEWS (both clean).** Correctness: no blockers/bugs — cross-slot reuse
     protected by the per-layer `device_sync`, `VmmBuf` `!Send`/drop-once-safe,
     decode byte-identical; only asked to document the CPU→GPU ordering source (done,
     commit 863eb48). Ponytail: already lean — one path-ref fix (done), everything
     else endorsed minimal. VMM stays a validated, documented primitive + the
     hip-apu-memory.md findings; its lasting value is the KNOWLEDGE, not a tok/s win.
  - **UNIFIED VMM+pread LOAD PATH — TRIED, MEASURED REGRESSION, REVERTED
     (2026-07-18).** Built the full unification: resident `DeviceTier` VMM-backed +
     both resident and cold loaded by `pread` straight from the shard file into the
     device-local slab (Snapshot kept the shard `File`s + a `read_into` pread;
     `DeviceTier::place`→`reserve`; `drop_pages` madvise→`posix_fadvise` evict). It
     WORKS (decode coherent, 28 tests green) but is a **net loss, not the expected
     marginal win**: pin build 24.7→33.9 s (+37%), cold fetch 1.08→1.96 ms/miss
     (+81%), wall 2086→2326 ms (+11%). TWO measured causes: (1) a VMM resident tier
     must be **CPU-filled** (pread/memcpy) whereas the `hipMalloc` tier fills via
     **`hipMemcpy` H2D = DMA engine** — DMA wins for the one-time 48 GiB bulk load
     (VMM-resident is read-neutral, but I'd ignored *fill* cost); (2) **`pread`
     `copy_to_user` into device-local host-granted pages is ~2× slower than a
     userspace `memcpy`** from the warm mmap. The two load paths diverge for a
     REASON: resident = bulk one-time → DMA-H2D; cold = per-miss host-write →
     VMM+memcpy. Reverted to `a52a800`. Probe `scratchpad/pread_vmm.cpp` confirms
     pread→VMM is correct+GPU-coherent — the regression is throughput, not
     correctness. DO NOT retry unless a bulk async DMA-into-VMM path exists.

**(3) priming — DEFERRED indefinitely; we TRUST colibri's priming.** The shipped
`.coli_usage` IS colibri's priming artifact, and routing was verified correct
against it (0.2 % absent, 0 misses below cutoff). Re-running our own
workload-matched priming pass is NOT on the near-term roadmap — we accept
colibri's ranking as the pin order. Consequence: the 67 % pinning ceiling / 45 %
hit on the "sky is blue" bench is a *known, accepted* breadth gap for now, not a
bug to chase. (Priming being parked no longer parks the coherent pin — see the
"UNGATED FROM PRIMING" note above; the two are now decoupled.)

**Near-term levers that remain (given priming is parked):** (a) MoE fused-kernel
inner-dot coalescing — the 29 % `mlp` bucket, the last big *compute* win; (b) NVMe
fault throughput for the 63 % warm+fetch I/O (io-uring / pread pool — the I/O
itself, threading already fixed). Device-side routing / ≤1-join stays SKIPPED
(blocked while cold-fetching; only reachable fully-resident, i.e. post-priming).

## HIP KERNEL REVIEW → target 2 tok/s warm (2026-07-20)

Broadened from "(a) MoE inner-dot coalescing" above into a full review of all six
kernel translation units. Everything below is **measured on rh-anine**, not
projected from theory; four new probes in `docs/probes/` reproduce it.

### Baseline

`rivoli /var/db/llama-server/glm52-colibri-int4 -bench 128 --pre-seed`:

```
PROFILE/tok: 1080ms wall | fetch 596ms 55% (121 miss, 4.89ms/miss)
           | attn 175ms 16% | mlp 203ms 19% | lmhead 7ms 1% | route 2ms | other 96ms
0.88 tok/s overall, expert hit 79.1% (climbing 71%→88% across the run)
```

**Compute = wall − fetch = 481 ms/token.** That is the budget this section
attacks. Note the arithmetic that frames everything: at 481 ms of compute, a
*zero-miss* run would already be 2.08 tok/s. So "2 tok/s warm" is a statement
about **compute AND hit rate jointly** — see "What 2 tok/s actually requires".

### THE finding: every int4 kernel reads ONE BYTE per lane per step

`dot_i4_wave` (moe_fused.hip), `gemv_i4` (linalg.hip) and `nib_i4` (mla.hip) all
have lane `l` load `row[i>>1]` for column `i ≡ l (mod 32)`. Lanes `l` and `l+1`
read the *same* byte, so a 32-lane wave touches **16 consecutive bytes per load
instruction** — one eighth of a 128 B cache line. The loop therefore issues ~8×
more loads than the traffic needs and keeps far too few bytes in flight. The D3
rewrite fixed the *stride* (adjacent lanes now read adjacent addresses) but never
fixed the *width*.

`docs/probes/i4gemv_probe.cpp` — one MoE layer's routed batch (E=9, hidden 6144,
moe_inter 2048, 162 MB int4, far past any cache):

| variant | ms/layer | GB/s | speedup | ms/token (75 layers) |
|---|---|---|---|---|
| A byte (shipped, 16 B/wave-load) | 2.282 | 74.4 | 1.00× | 171 |
| B `uint` (4 B/lane, 128 B/wave-load) | 0.883 | 192.4 | 2.58× | 66 |
| C `uint4` (16 B/lane, 512 B/wave-load) | 0.791 | **214.6** | **2.88×** | **59** |

214.6 GB/s is **93 % of the ~230 GB/s LPDDR5X peak** — variant C is DRAM-bound,
i.e. there is nothing left in the MoE inner dot after this change. The shipped
form runs at 32 % of peak. The same `dot` is used by every other int4 kernel.

**Take variant B (`uint`), not C — the snapshot's alignment decides it.** Checked
the real shards: the safetensors header is 101512 bytes, so **all 852 tensors in a
shard sit at data offset ≡ 8 (mod 16)** (relative offsets are 16-aligned; the
header length shifts every one of them by 8). Pool placement preserves that skew —
`packed = slot_base + slot_dst + (file_begin & 4095)` with both slot terms
4096-aligned, so `packed % 16 == file_begin % 16 == 8` for **every** expert. A
`uint4` pointer cast is therefore misaligned on all of them.

`docs/probes/` alignment probe: the `uint` path costs only **2 %** for the skew
(203.9 GB/s at 16 B-aligned vs **199.7 GB/s at the real ≡8 offset**), and needs
only 4 B alignment, which every tensor satisfies. Variant C's extra ~7 ms/token
is not worth a runtime alignment dispatch and a fallback tier. **One code path,
`uint`, no tiering** — worth **~105 ms/token** on `mlp` alone.

### Per-kernel review

Weight bytes/token at GLM-5.2 dims (hidden 6144, q_lora 2048, kv_lora 512,
qk_nope 192, qk_rope 64, v_head 256, H=64, 78 layers, 75 sparse):

| kernel | file | bytes/tok | shipped | after `uint` | verdict |
|---|---|---|---|---|---|
| `moe_gateup`/`moe_down` | moe_fused.hip | 12.7 GB | ~171 ms | ~66 ms | **FIX — byte loads** |
| `gemv_i4` (q_a,q_b,kv_a,o_proj) | linalg.hip | 5.9 GB | ~79 ms | ~31 ms | **FIX — byte loads** |
| `mla_value` | mla.hip | 0.33 GB | ~9 ms | ~2 ms | **FIX — worst pattern, small payoff** |
| `mla_absorb` | mla.hip | 0.25 GB | ~4 ms | ~3 ms | fix opportunistically |
| `gemv_i8` (lm_head) | linalg.hip | 0.95 GB | 7 ms | ~5 ms | minor |
| `gemv_f32` (gate + predict) | linalg.hip | 0.95 GB | ~5 ms | n/a | already coalesced, leave |
| `mla_latent_attend` | attn.hip | KV-only | ~15 ms | — | **grid = ⌈H/HB⌉ = 8 blocks** (see below) |
| `rmsnorm`/`layernorm`/`argmax` | linalg/indexer/fwd | — | ~1.6 ms | — | leave (measured: not occupancy-bound) |

`mla_value` deserves a note even though the payoff is small: it is the *only*
kernel still in the pre-D3 thread-per-row shape, so a wave's 32 lanes read 32
different rows 256 B apart — **32 separate cache lines per load instruction**.
`docs/probes/mla_proj_probe.cpp` measures the fix at **5.45×** (wave-per-row alone
gives 2.57×; the probe’s uint4 variant reaches 5.45×, the shipped-alignment `uint` form lands between). It is a correctness-neutral 10-line change
and removes the last inconsistency with the D3 contract.

### Measured NON-problems — do not spend effort here

`docs/probes/` overhead probe, gfx1151:

- **host launch dispatch 1.29 µs** → ~1800 launches/token (≈23/layer × 78) ≈ **2.3 ms**. Not a lever.
- **`hipDeviceSynchronize` on an idle queue 0.10 µs**; launch+sync serialized
  7.49 µs vs 1.29 µs pipelined, so the drain penalty is ~6.2 µs. The engine's
  **~150 syncs/token cost ~0.9 ms.**
- **`rmsnorm` at grid=1 is 6.73 µs; at grid=40 it is 6.90 µs** — identical. It is
  launch-latency bound, not occupancy bound. 234/token ≈ 1.6 ms.

**This retires the "≤1 join/token" line of work as a throughput lever.** PLAN's
bottleneck #3 budget (≤100 launches, ≤1 join per token) was written from colibri's
OpenMP fork/join costs; on HIP the same pattern costs ~3 ms/token total. Device-side
routing to reach ≤1 join is **not** worth building for speed. (It stays interesting
only if it enables overlapping fetch with compute — a *different* argument, on the
I/O side.) Likewise: do not multi-block the norms.

### What 2 tok/s actually requires

Vectorising every int4 dot onto the `uint` path: compute **481 → ~315 ms/token**
(−166 ms).

| hit rate | fetch ms/tok | wall | tok/s |
|---|---|---|---|
| 79.1 % (today's 128-tok bench) | 596 | 911 | 1.10 |
| 90.9 % (today's 512-tok warm) | ~270 | ~585 | **1.71** |
| 95 % | ~147 | ~462 | **2.16** |
| 100 % (unreachable, bound) | 0 | 315 | 3.17 |

So **the kernel work alone does not reach 2 tok/s at the current 90.9 % warm hit
rate** — it reaches ~1.71. Closing the last gap needs one of:

1. **Cold-path bandwidth.** 4.89 ms per 18.9 MB miss = **3.9 GB/s**, against the
   16 GB/s the `iouring_vmm` probe measured at QD≥4. A 4× gap already documented
   and never chased. At 8 GB/s, 90.9 % hit gives ~132 ms fetch → **2.28 tok/s**.
   This is now the single largest remaining lever, larger than any kernel.
2. Hit rate to ~95 % (priming — deliberately parked, see above).

**Consequence to internalise: after the kernel fixes, rivoli is I/O-bound again**
(fetch ~270 of ~585 ms = 46 %). The kernel work is necessary but not sufficient.

### Unexplained: `other` = 96 ms/token (9 %)

Dispatch accounts for only ~3 ms of it. The rest is host-side work between syncs —
descriptor assembly, pin bookkeeping, prefetch accounting, tokenizer, argmax
readback. 96 ms/token is 9 % of wall and nobody has ever profiled it. Worth one
`perf record` pass before assuming it is irreducible.

### Implementation order

1. **`dot_i4_wave` → `uint` vectorised load** in `common.hpp` (one definition, all
   callers inherit it): main loop over `uint` words while `dim - i ≥ WAVE*8`, then
   a scalar tail. Lane `l` owns columns `[base + l*8, +8)`, so a wave-load covers
   128 B = one cache line instead of 16 B.
   - Every contraction dim divides by `WAVE*8 = 256` (6144, 2048, 12288, 16384,
     512), so the tail is dead code at GLM dims — but keep it, it is three lines
     and makes the helper total.
   - Alignment holds: every `rb` is a multiple of 4 (3072/1024/8192/256/6144) and
     every tensor offset is ≡ 8 mod 16, hence 4 B-aligned. **No runtime dispatch,
     no fallback tier** — measured 2 % skew cost, see above.
2. **`gemv_i4` + `gemv_i8`** onto the same helper (int8 needs its own 4-byte
   unpack — it is already 32 B/wave-load, so the win there is smaller).
3. **`mla_value` → wave-per-row** (matching every other D3 kernel), then the same
   helper. `mla_absorb` opportunistically.
4. Re-measure. Stop here if `mlp` lands near 66 ms and `attn` near 95 ms.
5. Only then: `mla_latent_attend`'s 8-block grid — a split-KV (flash-decoding)
   grid of `⌈H/HB⌉ × n_splits` blocks. **Low priority at short context** (~15 ms)
   but it is the kernel that decides the still-unrun >2048-context benchmark, so
   fix it as part of that work, not this one.

### SHIPPED + MEASURED (2026-07-20) — compute halved, tok/s unchanged

Implemented steps 1–3: `dot_i4_wave`/`dot_i8_wave` moved into `common.hpp` on the
dword-per-lane shape, `gemv_i4`/`gemv_i8` onto them, `mla_value` rewritten
wave-per-row. `mla_absorb` deliberately left alone (1.10× measured, not worth a
second load path). **52/52 tests pass** against the scalar oracle; decode output
stays coherent.

Same command as the baseline (`-bench 128 --pre-seed`, local snapshot):

| bucket | before | after | change |
|---|---|---|---|
| `attn` | 175 ms | **87 ms** | **2.01×** |
| `mlp` | 203 ms | **102 ms** | **1.99×** |
| `lmhead` | 7 ms | 5 ms | 1.4× |
| `other` | 96 ms | 133 ms | ↑ (tracks miss count) |
| **compute (wall − fetch)** | **481 ms** | **327 ms** | **1.47×** |
| `fetch` | 596 ms (121 miss) | 757 ms (156 miss) | ↑ hit 79.1 % → 73.6 % |
| wall / tok-s | 1080 ms / 0.88 | 1086 ms / 0.88 | **unchanged** |

**The compute prediction was right (327 measured vs ~315 projected) and the
throughput win was eaten by a hit-rate draw** — exactly the confound this section
warned about. Different f32 rounding → different sampled tokens → different expert
routing → 35 more misses/token → +161 ms of fetch that cancels the −154 ms of
compute. `ms/miss` is rock stable across every run ever recorded (4.89 / 4.85 /
4.84), so fetch is a strict linear function of miss count and the two effects can
be separated: **normalised to the baseline's 121 misses, wall goes 1080 → 914 ms,
i.e. +18 %.** That is the honest size of the kernel win.

512-token run: `1093ms wall | fetch 712ms 65% | attn 118ms | mlp 104ms | lmhead
4ms | other 153ms`, 0.90 tok/s at 75.3 % hit. Note `attn` grows 87 → 118 ms with
context while `mlp` is flat — that is `mla_latent_attend`'s 8-block grid (step 5),
now the only kernel whose cost scales with context.

**Conclusion: the kernel side is done and rivoli is decisively I/O-bound.** Fetch
is 65–70 % of wall; compute is 30 %. Halving compute again would buy under 15 %.
**Every remaining path to 2 tok/s runs through the cold path** (3.9 GB/s per miss
vs 16 GB/s probed) or the hit rate — not through HIP kernels. Two follow-ups that
are now visible and were not before:

- **`other` is 133–153 ms/token (12–14 % of wall), larger than `attn` or `mlp`.**
  It scales with miss count, so it is likely per-miss host bookkeeping in
  `resolve_layer`/descriptor assembly. Never profiled. This is now a bigger lever
  than any remaining kernel.
- The 512-token hit rate (75.3 %) is far below the 90.9 % this PLAN records for
  the M4 gate. Either that figure came from an already-warm pool or the token
  stream matters more than assumed — **worth re-establishing before any future
  run cites 90.9 % as the warm baseline.**

### PREFETCH IS NOW A NET LOSS — measured A/B (2026-07-20)

Prompted by the miss-count rise above. **This A/B is clean**: prefetch depth does
not change the token stream (routing is a deterministic function of the weights),
so all four arms route the *identical* expert sequence and only cache/IO behaviour
differs. No hit-rate confound. 128 tok, `--pre-seed`, local snapshot.

| depth | wall | fetch (miss) | attn | mlp | **other** | tok/s | hit | recall | nvme0 GB |
|---|---|---|---|---|---|---|---|---|---|
| **0 (`--no-prefetch`)** | **1040** | 870 (185) | 77 | 77 | **10** | **0.91** | 68.7 % | — | 368 |
| 2 (default) | 1085 | 755 (156) | 87 | 101 | 134 | 0.88 | 73.6 % | 79.6 % | 382 |
| 4 | 1174 | 658 (131) | 88 | 112 | 310 | 0.81 | 77.9 % | 73.5 % | 417 |
| 8 | 1557 | 529 (94) | 85 | 113 | **823** | 0.61 | 84.2 % | 59.1 % | 556 |

**Prefetch does its job and still loses.** It monotonically improves hit rate
(68.7 → 84.2 %), cuts demand misses (185 → 94) and cuts `fetch` (870 → 529 ms).
But `other` explodes 10 → 823 ms, and total disk bytes rise 368 → 556 GB (+51 %).
Net: deeper is strictly worse, and **the current default (depth 2) is already a
net loss versus no prefetch at all.**

Mechanism: `other` is wall minus the timed buckets, and `prefetch_layer` is the
one call in the forward pass that is not inside a timed region — so `other` is
**disguised I/O wait**, `Streamer::submit` blocking once the ring fills against a
saturated disk. The host-side work itself (`pool.contains` / `alloc_cold` /
6× `io_uring_prep_read`) is microseconds; nothing in it can cost 1.4 ms/expert.

The root cause is simple: **prefetch buys hit rate with disk bandwidth, and disk
bandwidth is precisely the scarce resource.** At recall 59 % (depth 8), 41 % of
prefetched bytes are thrown away against a device that is the bottleneck.

**Why this regressed now:** depth 2 was validated as optimal when compute was
481 ms/token. The kernel work cut compute to ~170 ms (no-prefetch arm: attn 77 +
mlp 77 + lmhead 5 + other 10 + route 1), so **the compute window prefetch used to
hide under shrank by ~2.8×.** The prefetch tuning simply did not survive making
the GPU faster.

### Measured hardware ceiling — the disk is NOT 16 GB/s

`/` is **btrfs RAID0 across both NVMes** (`nvme0n1p6` 1.45 TiB + `nvme1n1`
465 GiB); reads stripe evenly (byte counters match to <1 % in every sample). Direct
O_DIRECT 19 MB random reads, caches not dropped, parallelism swept:

| parallelism | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| GB/s | 2.53 | 5.30 | **6.69** | 6.29 | 6.47 | 6.41 |

**The array tops out at ~6.5 GB/s at P≥4.** This *retires the 16 GB/s figure* from
the io_uring probe note above — that number was sequential 1 MiB and did not
survive contact with expert-sized random reads. rivoli's no-prefetch arm sustains
185 × 18.9 MB / 870 ms = **4.0 GB/s, about 62 % of the ceiling** — so the cold-path
headroom is ~1.6×, not the ~4× this PLAN previously claimed.

**Revised path to 2 tok/s** (compute now ~170 ms, so fetch must be ≤ 330 ms):
at the 6.5 GB/s ceiling, 330 ms buys ~113 experts/token, so it needs **both** the
I/O path taken to ~ceiling (1.6×) **and** hit rate ≥ 81 %. Notably depth-8 prefetch
already *achieves* 84.2 % hit / 94 misses — 94 × 18.9 MB at 6.5 GB/s would be
273 ms fetch → **2.26 tok/s** — but only if that hit rate can be had without the
+51 % wasted traffic. **The lever is prediction ACCURACY, not depth.**

### Double-fetch: dedup exists, with one gap

Answering "do we load the same expert twice (prefetch + cold miss)?" — **no, not
on the normal path.** `prefetch_layer` skips already-resident keys
(`pool.contains`) and `alloc_cold` *binds the key to its slot at submit time*, so
`resolve_layer` drains the prefetch ring first and the predicted-correct experts
then surface in phase 1a as ordinary hits. No second read is issued.

Two consequences worth writing down:

1. **The reported hit rate counts prefetched-off-disk experts as HITS.** "hit
   73.6 %" means "did not need a *demand* read at resolve time", not "was already
   resident". True disk traffic is `misses + prefetch-queued`, which is why the
   nvme byte counter rises with depth while the miss count falls. Any future
   residency claim must use bytes, not the hit counter.
2. **Gap (hypothesis) — TESTED AND REFUTED, see below.**

### loaded / preloading / cold instrumentation (2026-07-20)

The single `hits` counter conflated "already resident" with "a prefetch just
streamed it off disk". Split three ways in `Pool` (`pending_pf` set + `hit_loaded`
/ `hit_preload` / `misses`), reported as `expert source:` and `disk traffic:`.
128 tok, `--pre-seed`, identical routed sequence in every arm:

| depth | **loaded** | preloading | cold | reads/tok | GB/tok | wasted | tok/s |
|---|---|---|---|---|---|---|---|
| 0 | **68.7 %** (54794) | 0 | 31.3 % (25006) | 195.4 | 3.69 | — | **0.96** |
| 2 | **68.8 %** (54888) | 4.8 % (3855) | 26.4 % (21057) | 204.0 | 3.86 | 4.6 % | 0.89 |
| 4 | **68.6 %** (54769) | 9.2 % (7357) | 22.1 % (17674) | 226.6 | 4.28 | 17.6 % | 0.83 |
| 8 | **68.1 %** (54368) | 16.1 % (12859) | 15.8 % (12573) | 317.3 | 6.00 | 37.4 % | 0.62 |

**`loaded` is FLAT at 68.1–68.8 % across every arm.** Prefetch does not improve
true residency by even one point — it only relabels disk reads as "hits". The
apparent hit-rate climb 68.7 → 84.2 % is **entirely fictitious**: the same experts
are read off the same disk, just earlier and under a different counter. Meanwhile
real traffic rises 3.69 → 6.00 GB/tok (+63 %). At the measured 6.5 GB/s array
ceiling, depth 8's 6.00 GB/tok needs ~920 ms of pure disk time per token — which
is the whole story of its 1509 ms wall.

**Never quote the hit rate again without the split.** It is not a residency metric.

### Eviction-before-use hypothesis: REFUTED

Direct test (`pf_evict_then_missed`): mark every prefetched key evicted while
still unclaimed, then count demand misses that land on one — i.e. the expert
really was wanted, we really did read it, and we threw it away in time to pay for
it twice.

| depth | wasted (evicted unused) | of those, later demanded | share |
|---|---|---|---|
| 2 | 1205 of 5060 queued (23.8 %) | **127** | 10.5 % |
| 8 | 15179 of 28038 queued (54.1 %) | **609** | 4.0 % |

Eviction-before-use is real but **negligible** — 127 and 609 re-reads against
26 k and 41 k total reads (0.5 % and 1.5 %). And the share *falls* with depth
(10.5 → 4.0 %), the opposite of what the hypothesis predicted. The
`n_slots >= top_k + prefetch_depth` invariant plus the phase-1a-before-phase-1b
ordering in `resolve_layer` (all `get`s precede any `alloc`) really is sufficient.

**The waste is plain misprediction.** At depth 8, 54 % of queued prefetch reads
are for experts that are simply never selected. The prefetcher's accuracy, not
the cache's eviction discipline, is what fails.

### Depth 1, and the policy re-examination at the winning depth (2026-07-20)

Depth 1 completes the sweep. It is far better behaved than depth 2 — traffic
within 0.5 % of no-prefetch, waste 1.6 %, and the best recall of any depth (82.3 %) —
but wall still rises monotonically with depth:

| depth | wall | fetch (miss) | other | loaded | reads/tok | GB/tok | waste | recall | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| **0** | **1026** | 857 (185) | **10** | 68.7 % | 195.4 | 3.69 | 0 % | — | **0.93** |
| 1 | 1041 | 792 (167) | 63 | 69.0 % | 196.3 | 3.71 | 1.6 % | 82.3 % | 0.91 |
| 2 | 1075 | 747 (156) | 130 | 68.8 % | 204.0 | 3.86 | 4.6 % | 79.6 % | 0.89 |

The trade is explicit: depth 1 moves 65 ms out of `fetch` and puts 53 ms into
`other` (net +15 ms); depth 2 moves 110 ms and puts back 120 ms. **Prefetch's
overlap benefit is real but its submit cost is always slightly larger.** Note
`other`/prefetched-expert ≈ 3.3 ms at depth 1 — about one expert read — which says
`Streamer::submit` is effectively *synchronous* against a busy disk rather than
fire-and-forget. If that is fixable, depth 1 becomes a small win rather than a
small loss; it is the only version of prefetch worth revisiting.

**Cache policy at depth 0 (`--no-prefetch`).** These deltas are EXACT, not noisy:
with the routed sequence fixed and prefetch off, the cache simulation is
deterministic, so the miss counts are reproducible to the unit.

| policy | wall | fetch (miss) | **loaded** | reads/tok | GB/tok | tok/s |
|---|---|---|---|---|---|---|
| `arc` (shipped default) | 1021 | 852 (185) | 68.7 % | 195.4 | 3.69 | 0.93 |
| `lru` | 1007 | 837 (179) | 69.7 % | 189.0 | 3.57 | 0.94 |
| **`2q`** | **1003** | 834 (175) | **70.2 %** | **185.6** | **3.51** | **0.95** |

**ARC — the shipped default — is the WORST of the three.** 2Q gives +1.5 pp
residency and −5.0 % disk traffic over it; even plain LRU beats it. ARC was
presumably chosen while prefetch was on, where `insert_cold` feeds its ghost
lists; with prefetch off that machinery never runs and ARC's adaptivity only
costs. Since residency is now the critical path (below), a policy that buys
1.5 pp of `loaded` for free is worth more than it looks.

~~**Recommended default change: `--no-prefetch` + `--cache-policy 2q`.**~~
**SUPERSEDED — the reason prefetch lost was a BUG, see next section.** 2Q vs ARC
stands; "prefetch off" does not.

## THE PREFETCH BUG: io_uring_submit was synchronous (2026-07-21)

Chasing why `other` grew ~60 ms per unit of prefetch depth, `prefetch_layer` was
instrumented in three parts. The answer is not subtle:

```
prefetch cost: alloc 15ms (0.12ms/tok) | sqe-prep 3ms (0.02ms/tok)
             | io_uring_submit 14959ms (116.87ms/tok)      <-- 99%
```

**`io_uring_submit` blocked the decode thread for 2.96 ms per queued expert** —
about 61 % of the 4.85 ms the read itself takes. The ring was created with
`io_uring_queue_init(entries, &r->uring, 0)`, so `io_uring_submit` is an
`io_uring_enter` syscall in which the submitting task drives the btrfs/blk-mq
dispatch inline. **The "asynchronous, overlaps GPU compute" prefetch was doing
most of the read synchronously**, then hiding the cost in the one part of the
forward pass no PROFILE bucket covers. Every conclusion built on prefetch being
free was measuring an artifact — including this PLAN's own "depth 2 validated
optimal" and yesterday's "prefetch is a net loss".

**Fix: `IORING_SETUP_SQPOLL` on the prefetch ring** (kernel poller thread; submit
becomes a shared-memory tail write, no syscall), with a graceful fallback to a
plain ring. Main ring left alone — its submit is immediately followed by a
blocking drain, so it has nothing to gain. `io_uring_submit` → **0.00 ms/tok**.
52/52 tests pass; output stays coherent.

### Result — this is the largest single win in the project

128 tok, `--cache-policy 2q`:

| config | wall | fetch (miss) | **loaded** | reads/tok | GB/tok | tok/s |
|---|---|---|---|---|---|---|
| no prefetch | 1003 | 834 (175) | 70.2 % | 185.6 | 3.51 | 0.95 |
| **SQPOLL depth 1** | **631** | 456 (84) | **83.7 %** | **102.7** | **1.94** | **1.45** |
| SQPOLL depth 2 | 1280 | 1099 (126) | 73.6 % | 169.7 | 3.21 | 0.75 |

**512 tok warm, SQPOLL + 2q + depth 1:**
```
PROFILE/tok: 531ms wall | fetch 325ms 61% (43 miss, 7.55ms/miss)
           | attn 114ms 22% | mlp 74ms 14% | lmhead 5ms | route 3ms | other 11ms
1.83 tok/s | expert hit 92.5% | loaded 91.8% | 50.6 reads/tok (0.96 GB/tok)
```

**1.83 tok/s**, against the M4 gate's recorded 1.31 and this morning's measured
0.90. Disk traffic fell 3.51 → **0.96 GB/tok (−73 %)**.

**Why residency jumped (70 → 92 %), which scheduling alone cannot explain:** the
routed sequence is identical (79800 selections either way), so prefetch is
changing *admission*, not selection. Under 2Q a demand miss enters A1in and needs
a SECOND access to reach the protected Am set; a prefetched expert enters A1in and
its first real `get` promotes it straight to Am. Prefetch is therefore acting as a
one-touch admission filter that converges the hot set far faster. That is a real
cache-quality effect, and it is why prefetch + 2Q compounds rather than just adds.

**Depth 1 only.** Depth 2 is now *worse than no prefetch* (0.75 tok/s, ms/miss
doubles to 8.67): with submit genuinely async the extra reads really do run in the
background and oversubscribe the disk, and the lower-confidence second prediction
pollutes A1in, displacing entries before they can promote. The confidence ranking
matters more than the count — consistent with recall falling with depth.

### Full 512-token grid: depth {1,2} × policy {lru,arc,2q} (2026-07-21)

All six cells, `--pre-seed`, SQPOLL prefetch ring, identical routed sequence:

| depth | policy | wall | fetch (miss) | attn | mlp | **loaded** | reads/tok | GB/tok | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lru | 1293 | 1083 (156) | 114 | 77 | 71.1 % | 178.0 | 3.36 | 0.76 |
| 1 | arc | 1189 | 978 (159) | 114 | 78 | 70.7 % | 183.1 | 3.46 | 0.83 |
| **1** | **2q** | **505** | **300 (43)** | 114 | 74 | **91.8 %** | **50.6** | **0.96** | **1.92** |
| 2 | lru | 1323 | 1112 (144) | 115 | 78 | 70.6 % | 185.5 | 3.51 | 0.75 |
| 2 | arc | 1228 | 1015 (147) | 115 | 78 | 70.6 % | 193.6 | 3.66 | 0.80 |
| 2 | 2q | 1033 | 826 (120) | 113 | 75 | 75.5 % | 154.6 | 2.92 | 0.95 |

**`2q` + depth 1 is not a trend, it is a sharp peak.** Everything else clusters at
70.6–75.5 % residency and 0.75–0.95 tok/s; that one cell is 91.8 % and 1.92 tok/s
— 2× the next best. LRU and ARC are indistinguishable from each other and barely
respond to depth at all (70.6–71.1 % in all four of their cells), so the
interaction is specifically **2Q's structure × exactly one high-confidence
prediction**.

Mechanism, consistent with the numbers: 2Q's A1in probation queue is bounded. At
depth 1 the single best prediction enters A1in and its first real `get` promotes
it to the protected Am set, so the hot set converges fast and holds. At depth 2
the cold-insert rate into A1in doubles, churning entries out of probation *before*
their `get` arrives — fewer promotions, worse Am quality, residency collapses back
to 75.5 %. LRU and ARC have no probation/protection split for this to exploit,
which is why they sit flat. **The prediction's confidence rank matters far more
than the number of predictions.**

Also note `attn` is 113–115 ms and `mlp` 74–78 ms in **all six cells** — the
compute buckets are completely invariant to cache configuration, as they should
be. That is a good internal consistency check on the whole grid.

### Where 2 tok/s stands now

**505 ms/token against a 500 ms target — 1 % short, i.e. at the threshold within
run variance** (the two `2q`+depth-1 runs measured 531 ms / 1.83 tok/s and 505 ms
/ 1.92 tok/s, with byte-identical cache counters — so the spread is pure
wall-clock noise, not behaviour). The profile has inverted again:

- `fetch` 325 ms (61 %) but only **43 misses at 7.55 ms/miss** and 0.96 GB/tok =
  **2.9 GB/s, against the 6.5 GB/s ceiling.** The cold path is now
  LATENCY-bound, not bandwidth-bound — too few reads per layer to fill the queue.
  Batching misses across layers, or a deeper (but still depth-1-confidence)
  lookahead, is the lever.
- `attn` 114 ms (22 %) is now the #2 bucket and the only one that grows with
  context — that is `mla_latent_attend`'s ⌈H/HB⌉ = **8-block grid** (step 5 of the
  kernel plan, previously deprioritised). At 22 % it is worth doing now.

Either one alone plausibly covers the 31 ms.

### Consequence for 2 tok/s

Compute is ~170 ms/token. For a 500 ms budget, fetch must be ≤ 330 ms, which at
the 6.5 GB/s ceiling is ~2.15 GB/tok ≈ 113 expert reads/tok. Today's best arm
(no-prefetch) reads 195/tok at 68.7 % residency.

So **2 tok/s needs residency ~83 %, and prefetch provably cannot supply it** —
`loaded` does not move. The only levers on residency are pool contents: priming
(parked), a better eviction policy, or a bigger pool (settled as byte-optimal).
That is now the critical path, and it is a cache-admission problem, not an I/O or
kernel problem.

### Measurement protocol — the confound this project keeps hitting

Changing the summation order changes f32 rounding, which perturbs the sampled
token stream, which routes to different experts, which changes the cache hit
rate, which changes `fetch`, which changes tok/s. This is **exactly** the
artefact that made the `--kv-fp8` and dsa/misa numbers uninterpretable (see the
attention-modes section above). Therefore:

- Gate correctness on `tests/kernel_test.rs` against the scalar oracle (1e-3
  int4 dequant tolerance), not on output identity.
- Report **`attn` and `mlp` ms/token**, which are hit-rate-independent, as the
  primary result. Report tok/s only alongside the hit rate.
- Determinism is preserved: lane `l` owning a contiguous column block and then
  `wave_sum` is still schedule-independent, so greedy decode stays reproducible.
- The `i4gemv_probe` numbers are the ceiling; if the in-engine `mlp` bucket does
  not approach 66 ms, the gap is dispatch/serialisation, not the dot.

## M3 kernel contracts (locked by the M2 review)

The M2 kernels are the correctness oracle; M3 rewrites them for residency. Two of
these must change *before* wiring the resident tier or they silently reintroduce
colibri's copy-pool (bottleneck #2). Numbers below are per-token at GLM-5.2 dims
(hidden 6144, dense_inter 12288, moe_inter 2048, n_shared 1, 64 heads, 78 layers,
kv_lora 512, 200k KV) and ~60 GB/s effective streaming.

- **D1 (MoE signature → per-expert descriptor array) — MUST FIX for M3.** The M2
  kernel addresses expert `e` as `base + e*stride`, assuming all batched experts
  are contiguous and each projection is one flat array. In M3 the routed experts
  live at arbitrary pin-pool offsets; satisfying the contiguous signature forces
  a gather+stage+re-read repack ≈ 34 GB/token (~3× the 11.3 GB honest budget).
  Fix: pass a descriptor `{gate_ptr,up_ptr,down_ptr,gate_scale,up_scale,down_scale,
  inter,hidden}` per block so the kernel dereferences whatever the pin/slab holds
  and reads the weights once. (The batch-union `atomicAdd`-into-one-`out` shape is
  already M3-correct.)
- **D2 (attention → token-tiled flash) — MUST FIX for M3** (this is the deferred
  P1; the review confirms it is mandatory, not a nicety). One-thread-per-head
  re-streams the whole KV cache H≈64× ⇒ ~1.5 TB/token at 200k (fatal). Fix:
  stage each `L_t` tile into LDS once and let all heads in the block consume it —
  H× DRAM → 1× DRAM + H× LDS. The `Lc/Rc + nt` signature is already the resident
  per-layer KV-slab shape, so only the loop changes, not the interface.
- **D3 (wave-coalesced weight load) — M3 perf.** Row-per-thread makes adjacent
  lanes read addresses `rb_h`≈3 KB apart, so wavefront loads don't burst-coalesce
  — a cap on the achievable fraction of the ~225 GB/s peak. Fix: cooperative
  wave32 co-load of a row's bytes into LDS/regs then reduce; int4 packing stays
  bit-locked (quant.rs). Prototype alongside D1 (same inner loop).
- **D6 (fold the shared expert into the routed batch) — M3 launch-count.** With
  n_shared=1, the shared expert's inter (2048) equals the routed inter, so it can
  be the 9th block of the routed batch (weight 1.0) instead of its own launch —
  folds ~78 launches/token toward the ≤100 budget. If a future config has
  n_shared>1, D1's descriptor array handles the mixed width instead.

Confirmed M2-ok (no change): `x` re-read from L2 in MoE phase A (<0.1% of budget);
the attention `out[i]` RMW accumulator (2 KB/head, L2-resident — the P1 note
targets the *KV* read, not this); per-call `hipMemcpy` (M4 resident-slab replaces
it, no signature change needed).

## M3 (4/4) wiring contracts (locked by the M3 residency review)

The DeviceTier + descriptor MoE kernel are sound, but raw device pointers erase
the borrow checker's lifetime tracking. When engine.rs wires the resident path,
these are mandatory — several are the exact bottleneck-#6 wedge trigger if
ignored:

- **Async lifetime = join, not call.** `launch_moe` enqueues and returns; the
  kernel reads x/out/wexpert/descs and the weights they point at *after* it
  returns. Everything referenced must stay valid until the next `device_sync()`
  RETURNS. A `?` between launch and sync that drops a buffer = GPU UAF.
- **Own the device memory in one place with a sync-on-Drop.** A `DeviceCtx`
  holding the tier + per-token buffers, whose `Drop` calls `device_sync()`
  before any `hipFree`, so teardown (incl. unwinding) never frees under a live
  kernel. `device_sync()` must be reached on every path, error paths included.
  The type-honest form is a launch→join guard.
- **No per-token `hipMalloc`.** The test idiom (`from_bytes`/`zeroed` per call)
  is the mid-session re-allocation the amdgpu GTT wedge punishes. Hoist
  `x_buf`/`out_buf`/`descs_buf` into the Engine, allocate once, reuse via
  `copy_in`/`zero` (already alloc-free). Reusable `[ExpertDesc; MAX_TOPK+1]`.
- **Zero-copy host→device bridge.** Do NOT use the test's `f32_bytes` (allocs a
  Vec/token). Reinterpret `&[f32]→&[u8]` with `bytemuck::cast_slice` (host is
  LE) straight into `copy_in`. Add a borrowing `DeviceBuf::copy_out_into(&mut
  [u8])` for logits readback into the engine's existing `logits` Vec.
- **Pin builder feeds the tier from mmap directly.** Pass `Int4Matrix.packed` /
  `.scale` straight to `place` — never `read_f32` (it dequants to host f32, 4×
  the traffic). Resolve `(layer, expert)→device ptr` ONCE into a flat table
  indexed `layer*n_experts+id` (P3), so per-token descriptor assembly is pointer
  reads, not `format!`-key map lookups.
- **Threading.** `DeviceTier`/`DeviceBuf` are `!Send` by design and that's
  correct — `block_on` needs no `Send`, only `tokio::spawn` does (the engine
  never spawns). Every HIP call (place, launch, sync, Drop) runs on the one
  `block_on` thread. NEVER `unsafe impl Send` — HIP null-stream ordering isn't
  thread-safe.
- **Descriptor shape is uniform (D1/D6 reconcile).** The built `ExpertDesc`
  carries pointers only; `hidden`/`inter` are uniform kernel params. This folds
  the shared expert into the routed batch when `n_shared==1` (D6, GLM-5.2) but
  CANNOT express mixed-width experts — a future `n_shared>1` config would need
  per-descriptor `inter`/`hidden` added to the struct (validated at model load).
- **KV cache = per-layer resident slab (P2), grown in place.** The attention
  kernel's `Lc[nt*kvl]/Rc[nt*rope] + nt` signature is already device-slab-shaped
  (verified) — do NOT change it. Wiring: one `DeviceBuf` per layer sized to
  `max_ctx` up front (200k → ~205MB L + 26MB R/layer, ~18GB total, fits with the
  64GB pin), append = write ONE token's 1152 bytes at `nt*row_bytes` via a new
  `DeviceBuf::copy_in_at(off, bytes)` + increment `nt`. NEVER `from_bytes` the
  whole cache per token (18GB H2D + a per-token hipMalloc = the wedge). The
  append value is `s.comp[..kvl]` f32 → convert to bf16 into a hoisted
  `[u16; kvl+rope]` scratch, then `copy_in_at`.
- **Value projection + o_proj stay on device.** The kernel writes `clat`
  device-side. If the `clat→ctx` projection through `kv_b` value rows (attn.rs
  matvec_i4_rows) or o_proj run on host, every layer must D2H-copy clat + join to
  read it = ~78 joins/token, breaking ≤1 join. Keep the whole layer a device
  pipeline with one token-end join. (qabs/qrope PRODUCTION can stay host — a
  pre-launch H2D input adds no join; only post-kernel outputs force a join.)
- **Attention KV re-read is a 200k tune, not a gate blocker.** HB=8 → ⌈H/HB⌉=8×
  DRAM KV re-read: fatal at 200k (~640ms–2.4s/token) but negligible at the M3
  128-token gate (KV ≪ the 11.3GB expert stream there). Fix is FFI-unchanged
  (internal kernel): move the accumulator LDS→registers so HB can rise to 32 (2×)
  or SUBW=16→HB=64 (1× honest). Do it in the measurement pass (VGPR-vs-occupancy
  is empirical), AFTER the 128-token gate proves the pipeline.
- **Per-token fault gate.** `device_sync()` returns the async execution fault
  (the launch return code only catches launch-config errors); the decode loop
  must check its `Result` every token.

## Risks

| Risk | Mitigation |
|---|---|
| ROCm gfx1151 maturity (known PERMISSION_FAULT reports upstream) | one-shot allocation; kernel-bug watchdog; single-engine design means a driver regression blocks loudly instead of silently degrading |
| Same amdgpu underneath → BO_VA -12 class bugs reachable from HIP too | single startup allocation; sole-tenant guard; stability ladder grows the tier only from observed device-loss data |
| hipcc/Gentoo packaging friction | M0 is exactly this, nothing else blocks on it; CPU milestones (M1) proceed in parallel |
| int4 layout mismatch vs colibri snapshot | snapshot.rs ports colibri's exact packing (glm.c `pack_int4`); kernel_test locks it |

## Conventions

- **No `unwrap`/`expect` outside tests** (workspace lint, deny).
- **No environment variables, no config files** — one CLI flag (`-bench`);
  everything else discovered and printed.
- Every benchmark/report prints its **full discovered config first** and its
  engine engagement counters (launches, submits) — a "GPU number" with zero
  launches is reported as CPU fallback, loudly.
- ollama-router repo conventions: flat modules, `tests/`, `.githooks`
  pre-commit (fmt+clippy) / pre-push (test), version-guarded `build.sh`.

### io_uring O_DIRECT queue-depth probe — GO (2026-07-18, feat/direct-load)

The direct-load regression traced to access pattern, not bandwidth. Probe
(`docs/probes/iouring_vmm.cpp`, O_DIRECT NVMe→VMM, coherent/MATCH):
QD=1 4.1 GB/s | QD=4 16.3 | QD=16 16.0 | QD=128 17.0. NVMe is latency-bound at
QD=1; at depth ≥4 it delivers **~16 GB/s = ~3× the current cold-path effective
bandwidth** (warm 793 + fetch 364 ≈ 1157 ms for ~6.3 GB/token ≈ 5.4 GB/s).
Projected: fold warm+fetch into one overlapped O_DIRECT stream → cold I/O
~390-630 ms/token → wall ~1320-1560 ms → **~0.64-0.76 tok/s** (vs 0.47 now;
colibri 0.8). GO. NEXT: implement io_uring O_DIRECT streaming into the cold VMM
slots (replace mmap-warm + memcpy-fetch) + O_DIRECT for the resident build.
Caveats to verify in-engine: random ~19 MB expert reads vs the probe's sequential
1 MiB; some of 16 GB/s may be drive-side.
