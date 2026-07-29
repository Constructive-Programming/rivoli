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
44 -> 8 -> 4 -> 2 cells over 512 / 2048 / 4096 / 10000 tokens (~10 h, one process per
cell). Numbers and caveats in **[benchmarks.md](benchmarks.md)**.

| | best cell | 512 | 2048 | 4096 | 10000 |
|---|---|---:|---:|---:|---:|
| tok/s | `int3-vq` + `streaming` + `2q` | 2.81 | 2.97 | 3.06 | **3.26** |

It is the only configuration that gets **faster** with context, and the reason is the
first thing to understand about this table: `streaming` attends a fixed 516 rows
(`--sinks 4 --window 512`) no matter how long the context is — 100% of context at 512
tokens, **5.2% at 10k**. It ranked 11th at 512, where it attends everything. The
throughput is real; the quality cost of discarding 95% of the context is not measured by
any number here.

Three results worth knowing before trusting a `tok/s` from this engine:

- **DSA degenerates at long context.** `int3-vq/dsa/2q` is clean at 512/2048/4096 and
  collapses at 10k: **45% of its output is a verbatim duplicate** (longest repeated block
  4544 of 10000 tokens, confirmed as real text). Its 2.31 tok/s is not a result. Detected
  only because runs are classified for repetition — no throughput metric can see it.
- **`top-m` is not free.** Fastest policy at 512 (it took all four top slots) and by 4096
  it is +1.6% over `2q` while substituting ~5.5% of experts away from the true top-K. Its
  85-90% hit rates are not comparable to the other policies'.
- **`int4` throws NaN/Inf intermittently** — 2 of 16 cells, non-reproducing, no predictive
  combination. `int3-vq` and `hybrid` were 28/28 clean. Not root-caused.

**A degenerate run benchmarks FASTER**, which is why none of this is ranked on raw
throughput: looping re-routes to the same few experts, so the hit rate rises. Measured
directly — the same cell scored 2.88 tok/s while emitting 50% duplicate output and 2.66
tok/s once the prompt was fixed. `-bench` runs therefore report a longest-repeated-block
and warn when output degenerates. See [MODES.md](MODES.md) on why free-running `tok/s`
cannot rank modes.

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
byte-arena pool whose hot/cold split floats with the workload. **[MODES.md](MODES.md)**
is the full story — the format tradeoffs, the cache policies, the interaction
matrix, and (important) why free-running decode `tok/s` cannot rank modes.

**The defaults are not the throughput winner, deliberately.** `hybrid` holds the better
perplexity (5.189 vs int3-vq's 5.275), while the matrix above measures `int3-vq` ~19%
faster at a 115 GiB budget. Which one is right depends on whether you are spending your
memory budget on quality or on slots, and the answer moves with `--max-mem`: at 100 GiB
the ranking reported in [docs/INT4.md](docs/INT4.md) is the other way round.

## Build

```
cargo build --release --features rocm
```

`build.rs` compiles `kernels/*.hip` via `hipcc` — needs ROCm and a gfx1151 target.
Without `--features rocm` the crate builds host-only (formats, converter logic,
cache/replay sims), which is what CI/clippy runs.

Optional features: `otlp` (export the decode as OTLP traces + metrics, opt-in at runtime
via `OTEL_EXPORTER_OTLP_ENDPOINT`; set `RIVOLI_SPANS` to also emit a per-token/per-layer
span timeline — see [docs/TRACES.md](docs/TRACES.md)); `trace` (expensive correctness probes
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
trace for the offline `replay` sim). The hybrid hot/cold split has no flag — it
self-sizes with the byte-arena pool.
Run `rivoli` with no model dir for the full usage line.

## Layout

```
src/            engine — format, quant, pin (residency + streaming), gpu (forward),
                arena/hybrid (byte-arena pool + policies), stream (io_uring),
                asyncfetch/gpustream (async load‖compute overlap), math, model, ...
src/bin/        convert, fp8_to_i4, vq3_to_i4, add_indexer, i4_audit, ppl, replay
kernels/*.hip   HIP kernels (moe, mla, attn, linalg, indexer, fwd, async, vmm)
examples/       dot_bench — per-format dot microbench (int4 vs int3-vq vs fp8)
tests/          kernel oracles + lib unit tests
tests/bench-matrix.sh   the mode x attn x policy matrix runner (classifies every
                cell ok/SUSPECT/DEGENERATE/CRASH/TIMEOUT; refuses to rank the rest)
docs/           ARCHITECTURE.md, PERF.md (roadmap + the class-axis profile),
                INT4.md, TRACES.md + grafana dashboard, GPU_TRACE.md; proposals:
                VULKAN.md (second backend), NPU.md (DSA/MISA),
                CACHE_ROUTE.md (top-m routing) + CACHE_PILOT.md (its prefetch)
MODES.md        format-mode + cache-policy reference
```
