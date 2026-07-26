# rivoli

A from-scratch GLM-5.2 MoE decode engine for the **AMD Strix Halo** APU (gfx1151),
in Rust + HIP/ROCm. It streams a 256-expert / top-8 / 78-layer model through a
device-side routed-expert pool backed by unified LPDDR5 (host RAM *is* GPU memory
via GTT), decoding at ~3 tok/s on a single Strix Halo node.

**The artifact is the model.** You convert the fp8 checkpoint once into a
self-contained directory, then point `rivoli` at it — no separate config, no
external weight dirs. The machine is auto-discovered; every CLI flag is a
benchmark/diagnostic override, not required setup.

## Weight formats

| weights | format | resident? |
|---|---|---|
| routed experts | **int3-vq** (12-bit codebook idx + bf16 g64 scale) and/or **int4** (colibri, per-row scale) | streamed (pooled) |
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
format; `--cache-policy lru | 2q | arc` (default **2q**) picks the pool eviction
policy. Hybrid keeps the frequently-reused ("hot") experts int4 for its ~1.8×
faster compute and streams the rest as small cheap int3-vq slots, in one
byte-arena pool whose hot/cold split floats with the workload. **[MODES.md](MODES.md)**
is the full story — the format tradeoffs, the cache policies, the interaction
matrix, and (important) why free-running decode `tok/s` cannot rank modes.

## Build

```
cargo build --release --features rocm
```

`build.rs` compiles `kernels/*.hip` via `hipcc` — needs ROCm and a gfx1151 target.
Without `--features rocm` the crate builds host-only (formats, converter logic,
cache/replay sims), which is what CI/clippy runs.

Optional features: `otlp` (export the decode summary as one OTLP span, opt-in at
runtime via `OTEL_EXPORTER_OTLP_ENDPOINT`); `trace` (expensive correctness probes
`--checksum-x`/`XSUM` + fine-grained per-op timing). The cheap per-token PROFILE
summary is always on, no feature needed.

## Convert a checkpoint → artifact

```
# fp8 GLM-5.2 checkpoint → int3-vq artifact (learns 3 codebooks, GPU-encodes experts)
cargo run --release --features rocm --bin convert -- <fp8-dir> <out-dir> --gpu

# add int4 expert files (L{l}.i4) beside the .vq3, from a colibri-int4 source,
# to enable --mode int4 / hybrid
cargo run --release --features rocm --bin pack_i4 -- <colibri-dir> <out-dir>

# add the DSA sparse-indexer weights (enables --attn dsa; auto-detected)
cargo run --release --bin add_indexer -- <out-dir> <indexer-stash.safetensors>
```

The artifact directory holds `manifest.json`, `codebooks.f32`,
`resident.safetensors`, one `L{ll}.vq3` per MoE layer (+ optional `L{ll}.i4`,
`indexer.safetensors`), and the tokenizer. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
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
trace for the offline `replay` sim), `--hot-pct <n>` via mode config (hybrid split).
Run `rivoli` with no model dir for the full usage line.

## Layout

```
src/            engine — format, quant, pin (residency + streaming), gpu (forward),
                arena/hybrid (byte-arena pool + policies), stream (io_uring),
                asyncfetch/gpustream (async load‖compute overlap), math, model, ...
src/bin/        convert, vq3_to_i4, pack_i4, add_indexer, replay
kernels/*.hip   HIP kernels (moe, mla, attn, linalg, indexer, fwd, async, vmm)
examples/       dot_bench — per-format dot microbench (int4 vs int3-vq vs fp8)
tests/          kernel oracles + lib unit tests
docs/           ARCHITECTURE.md; proposals: VULKAN.md (second backend),
                CACHE_ROUTE.md (top-m routing) + CACHE_PILOT.md (its prefetch)
MODES.md        format-mode + cache-policy reference
```
