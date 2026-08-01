# rivoli

A from-scratch GLM-5.2 MoE decode engine for the **AMD Strix Halo** APU (gfx1151),
in Rust + HIP/ROCm. It streams a 256-expert / top-8 / 78-layer model through a
device-side routed-expert pool backed by unified LPDDR5 (host RAM *is* GPU memory
via GTT), decoding at ~3 tok/s on a single Strix Halo node.

**The artifact is the model.** You convert the fp8 checkpoint once into a
self-contained directory, then point `rivoli` at it — no separate config, no
external weight dirs. The machine is auto-discovered; every CLI flag is a
benchmark/diagnostic override, not required setup.

## Measured performance

Full cross of mode x attention x cache-policy at `--max-mem 115`, bracketed
44 -> 8 -> 4 -> 2 cells over 512 / 2048 / 4096 / 10000 tokens (~10 h). Details and the
retraction below in **[docs/measurement/benchmarks.md](docs/measurement/benchmarks.md)**.

**Only the 512-token round is valid.** At 2048 tokens and beyond, free-running greedy
decode degenerates into template loops **regardless of mode, attention or cache policy** —
0/42 cells degenerate at 512, 1/8 at 2048, **4/4 at 4096 and 2/2 at 10000**. Long-context
throughput on this engine is currently not measurable by free-running decode.

Valid results, 512 tokens, `--max-mem 115`:

| mode | cells ok | tok/s | mean |
|---|---:|---|---:|
| `int3-vq` | 16/16 | 2.70-3.22 | **2.91** |
| `hybrid` | 12/12 | 2.12-2.76 | 2.45 |
| `int4` | 14/16 | 1.77-2.42 | 2.10 |

That table predates the group-128 `.i4` rebuild (`docs/investigations/int4-scales.md`, RESOLVED 2026-07-27):
its `int4` and `hybrid` rows were measured against the per-row-scaled set, and `int4` is
now the best-QUALITY mode in the engine (PPL 5.120 against int3-vq's 5.275) while staying
the slowest — its slot is 31% larger, so the same budget holds fewer experts. Rank quality
on `docs/investigations/int4-scales.md` §10 and `docs/measurement/benchmarks.md`, not on tok/s here.

Two caveats on that table. `top-m` holds the top four slots (3.10-3.22) but substitutes
~5.5% of experts away from the true top-K, so its 85-86% hit rates are not comparable.
And `int4` lost two cells to intermittent `NaN/Inf` logits — non-reproducing, no
predictive combination, `int3-vq` and `hybrid` 28/28 clean, not root-caused.

**Why a degenerate run must never be ranked: it benchmarks FASTER.** A loop re-routes to
the same few experts, so the cache hit rate climbs. Measured on one configuration as it
collapsed over increasing context — hit rate **81.6 -> 83.2 -> 85.4 -> 89.6%** while the
distinct-word ratio fell **0.474 -> 0.366 -> 0.288 -> 0.244**. It looked like the only
configuration that got *faster* with context. It was the one degenerating hardest.

`-bench` therefore classifies its own output and warns, on three independent signals: a
verbatim tail cycle, the longest repeated block, and **structural repetition** (most-
repeated line count + distinct-word ratio). The third exists because the first two both
passed a run that was 329 repetitions of `**Memory Product.**` — the loop had a varying
label, so no exact match was long enough to catch it. See [docs/reference/modes.md](docs/reference/modes.md) on why
free-running `tok/s` cannot rank modes; a fixed forced-token harness is the fix and is not
built yet.

## Weight formats

| weights | format | resident? |
|---|---|---|
| routed experts | **int3-vq** (12-bit codebook idx + bf16 g64 scale) and/or **int4** (nibble + f32 g128 scale) | streamed (pooled) |
| shared expert | int3-vq | resident |
| attention projections (q/kv/o) | **fp8** e4m3 + 128-block scale | resident |
| dense-layer MLPs (first 3 layers) | fp8 | resident |
| embed / lm_head | int8 + per-row scale | resident |
| norms, router gate | f32 | resident |
| DSA indexer (wk/wq_b fp8; weights_proj/k_norm bf16→f32) | fp8 / f32 | resident (optional) |

The routed experts are the bandwidth driver — smaller = more fit in the pool =
higher hit rate. Which format a routed expert decodes from is the **run mode**
(`--mode`, below); everything else is fixed and always resident. The KV cache is
fp8-e4m3 latent + f32 block scales + bf16 roped key, grown on device.

## Run modes

`--mode int3-vq | int4 | hybrid` (default **hybrid**) picks the routed-expert
format; `--cache-policy lru | 2q | arc | top-m` (default **2q**) picks the pool eviction
policy. Hybrid keeps the frequently-reused ("hot") experts int4 for its ~1.8×
faster compute and streams the rest as small cheap int3-vq slots, in one
byte-arena pool whose hot/cold split floats with the workload. **[docs/reference/modes.md](docs/reference/modes.md)**
is the full story — the format tradeoffs, the cache policies, the interaction
matrix, and (important) why free-running decode `tok/s` cannot rank modes.

**The defaults are not the throughput winner, deliberately.** `hybrid` holds the better
perplexity (5.189 vs int3-vq's 5.275), while the matrix above measures `int3-vq` ~19%
faster at a 115 GiB budget. Which one is right depends on whether you are spending your
memory budget on quality or on slots, and the answer moves with `--max-mem`: at 100 GiB
the ranking reported in [docs/investigations/int4-scales.md](docs/investigations/int4-scales.md) is the other way round.

## Build

```
cargo build --release --features rocm
```

`build.rs` compiles `kernels/*.hip` via `hipcc` — needs ROCm and a gfx1151 target.
Without `--features rocm` the crate builds host-only (formats, converter logic,
cache/replay sims), which is what CI/clippy runs — `cargo test` and
`cargo clippy --all-targets` are both clean with no features. The binaries and the
`moe_bench` example still BUILD in that configuration; they refuse at runtime, naming the
feature to rebuild with, because a rivoli that cannot decode should say so rather than
fail to link.

Optional features: `otlp` (export the decode as OTLP traces + metrics, opt-in at runtime
via `OTEL_EXPORTER_OTLP_ENDPOINT`; set `RIVOLI_SPANS` to also emit a per-token/per-layer
span timeline — see [docs/measurement/traces.md](docs/measurement/traces.md)); `trace` (expensive correctness probes
`--checksum-x`/`XSUM` + fine-grained per-op timing). The cheap per-token PROFILE
summary is always on, no feature needed.

## Convert a checkpoint → artifact

```
# fp8 GLM-5.2 checkpoint → int3-vq artifact (learns 3 codebooks, GPU-encodes experts)
cargo run --release --features rocm --bin convert -- <fp8-dir> <out-dir> --gpu

# add int4 expert files (L{l}.i4) beside the .vq3, derived from the fp8 source,
# to enable --mode int4 / hybrid
cargo run --release --features rocm --bin fp8_to_i4 -- <fp8-dir> <out-dir>

# add the DSA sparse-indexer weights (enables --attn dsa; auto-detected)
cargo run --release --bin add_indexer -- <out-dir> <indexer-stash.safetensors>
```

The artifact directory holds `manifest.json`, `codebooks.f32`,
`resident.safetensors`, one `L{ll}.vq3` per MoE layer (+ optional `L{ll}.i4`,
`indexer.safetensors`), and the tokenizer. See [docs/reference/architecture.md](docs/reference/architecture.md)
for the on-disk layout and the module map.

## Run

```
# decode a prompt
cargo run --release --features rocm -- <model-dir> --prompt "Why is the sky blue?"

# benchmark N tokens (hybrid default; override the mode/policy/budget as needed)
cargo run --release --features rocm -- <model-dir> -bench 256 --mode hybrid --max-mem 115
```

Useful flags: `--max-mem <GiB>` (device budget, literal; default `free − 16 GiB`),
`--direct-vmm-dma` (raw DMA over the default pinned bounce), `--attn
auto|dense|dsa|streaming|misa`, `--trace <path>` (dump the routed-expert access
trace for the offline `replay` sim), `--no-mtp` (turn off speculative decode, which is on by
default whenever the artifact carries the MTP head — **every mode carries one** since
2026-07-31, though an artifact converted before then needs `bin/fp8_to_i4` re-run to emit
`L78.i4`), `--mtp-min-conf` (the confidence gate that makes speculation pay — default 0.8,
**1.108×**; `0` disables it and costs you ~15%). The hybrid hot/cold split has no flag — it
self-sizes with the byte-arena pool.
`rivoli --help` lists every flag with its default and its legal values (`-bench` is
accepted alongside `--bench`, so every command line recorded in docs/measurement/benchmarks.md still runs).

## Documentation

Start at **[docs/00-orientation/TOUR.md](docs/00-orientation/TOUR.md)** — two pages, and it
is enough to be useful. Then
**[docs/00-orientation/INDEX.md](docs/00-orientation/INDEX.md)**, which lists every doc with
a status and a one-line **verdict** so you can decide what *not* to read.

| | |
|---|---|
| `docs/reference/` | true about the engine today — `architecture.md` is the one meant to be read whole |
| `docs/measurement/` | how to measure, the roadmap, traces, and the append-only benchmark log |
| `docs/investigations/` | questions asked, answered and closed. Read the verdict; open the file only to re-open the question |

## Serve — OpenAI API, and llama-swap

```
cargo run --release --features rocm -- <model-dir> --port 8080 --ctx 8192
```

`POST /v1/chat/completions` (with or without `stream`), `GET /v1/models`, `GET /health`.
Loopback only, one request at a time — the GPU is sole-tenant and decodes at ~3 tok/s, so
concurrency here could only queue what the device already serialises.

Under llama-swap, which spawns the process on demand and proxies to it:

```yaml
models:
  glm-5.2-rivoli:
    cmd: /path/to/rivoli /var/db/rivoli/glm52-vq3-full --port ${PORT} --ctx 8192 --max-mem 115
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    # Pin build is ~1 min and the port does not open until the model is loaded, so the
    # health check gets connection-refused (not a 503) until it is genuinely ready.
    healthCheckTimeout: 300
```

Three things it deliberately does not do, so nobody debugs their absence:

- **No sampling.** `temperature` and `top_p` are accepted and **ignored** — the engine
  decodes greedy argmax, which is what every number in `benchmarks.md` is measured against.
  Output is deterministic no matter what the client asks for. One warning per process says so.
- **No `/v1/completions`.** A raw, unframed prompt leaves the model outside an assistant
  turn where its EOS ids are unreachable, so it runs to the token limit and then loops —
  the failure mode that invalidated every benchmark cell above 2048 tokens (see the
  retraction above). That endpoint would serve it by construction.
- **No paging.** `--ctx` allocates the KV slabs once, at startup (~51 KB/token on top of
  the expert pool); a conversation that does not fit is a 400, never a silent truncation.

The per-request log carries the same degeneration check `-bench` does — a looped response
is a broken one, and it is *faster*, so server mode is not allowed to be the path that
hides it.

## Layout

```
src/            engine — format, quant, pin (residency + streaming), gpu (forward),
                arena/hybrid (byte-arena pool + policies), stream (io_uring),
                asyncfetch/gpustream (async load‖compute overlap), math, model, ...
src/bin/        convert, fp8_to_i4, add_indexer, i4_audit, ppl, replay
kernels/*.hip   HIP kernels (moe, mla, attn, linalg, indexer, fwd, async, vmm)
examples/       dot_bench — per-format dot microbench (int4 vs int3-vq vs fp8, rocm);
                moe_bench — the MoE kernels alone, same source on both backends
tests/          kernel oracles + lib unit tests; tests/common/ the shared
                backend-neutral scaffolding (Lcg, assert_close, byte helpers)
tests/bench-matrix.sh   the mode x attn x policy matrix runner (classifies every
                cell ok/SUSPECT/DEGENERATE/CRASH/TIMEOUT; refuses to rank the rest)
docs/           ARCHITECTURE.md, PERF.md (roadmap + the class-axis profile),
                INT4.md, TRACES.md + grafana dashboard, GPU_TRACE.md; proposals:
                VULKAN.md (second backend), NPU.md (DSA/MISA),
                CACHE_ROUTE.md (top-m routing) + CACHE_PILOT.md (its prefetch)
docs/reference/modes.md        format-mode + cache-policy reference
```
