# rivoli — architecture

A clean reimplementation of the GLM-5.2 MoE decode engine for AMD Strix Halo
(gfx1151). It grew an int3-vq-only core into the int4 / hybrid routed-expert path,
the byte-arena pool, and the DSA sparse indexer. For the run-mode tradeoffs see
[../MODES.md](../MODES.md).

## Weight formats

| weights | format | resident? | why |
|---|---|---|---|
| routed experts | **int3-vq** (12-bit codebook idx + bf16 g64 scale) and/or **int4** (colibri, per-row scale) | streamed | the bandwidth driver; the mode picks the format (see MODES.md) |
| shared expert | int3-vq | resident | folded into the MoE launch |
| attention projections (q_a/q_b/kv_a/kv_b/o_proj) | **fp8** e4m3 + 128-block scale | resident | native source precision, no requant loss; format-neutral to tok/s |
| dense-layer MLPs (first 3 layers) | **fp8** | resident | consistent with attention; small |
| embed / lm_head | **int8** + per-row scale | resident | unchanged; not int4 |
| norms, router gate | **f32** | resident | tiny |
| DSA indexer (wk/wq_b fp8; weights_proj/k_norm bf16→f32) | fp8 / f32 | resident (optional) | present only if the artifact carries indexer weights |

Codebooks are **per-projection** (gate/up/down) — the down_proj distribution
differs from gate/up; separate codebooks are the quality lever. 3 codebooks,
learned once, resident (~192 KiB total).

The routed-expert **format is a run-mode choice** (`--mode int3-vq|int4|hybrid`,
default hybrid). int3-vq is smallest (most residency, cheapest miss) but its dot
is a codebook gather (latency-bound); int4 is ~1.8× faster compute but 24% bigger
(fewer slots). Hybrid puts the hot set in int4 and streams the cold set as int3-vq
— one byte-arena pool, split floats with the workload. Full reasoning: MODES.md.

## On-disk artifact — one self-contained directory (built from the fp8 checkpoint)

```
<model>/
  manifest.json          # self-describing: format version, full ModelConfig,
                         #   VQ params, codebook layout, resident offset table
  codebooks.f32          # 3 × VQ_K × VQ_DIM f32 (gate, up, down)
  resident.safetensors   # every resident weight: per layer {fp8 attn proj + block
                         #   scales, fp8 dense MLP, f32 norms, f32 router gate};
                         #   global {int8 embed, int8 lm_head, f32 final norm}
  L{03..NN}.vq3          # one int3-vq file per MoE layer (see header below)
  L{03..NN}.i4           # optional int4 twin per MoE layer (for --mode int4|hybrid)
  indexer.safetensors    # optional DSA indexer weights (enables --attn dsa)
  tokenizer.json, generation_config.json
```

`convert` produces `manifest.json` + `codebooks.f32` + `resident.safetensors` +
`L{ll}.vq3`. `pack_i4` adds the `L{ll}.i4` twins from a colibri-int4 source.
`add_indexer` writes the side `indexer.safetensors`. All extra files are optional
and merged at load (`Safetensors::open_dir` unions every `*.safetensors`).

`.vq3` per-file header (little-endian): magic `"VQ3\0"`, u32 version, u32 layer,
u32 n_experts (incl. shared), u32 hidden, u32 moe_inter, u64 stride, then the six
projection sub-offsets. Each expert block is `gate‖up‖down` (12-bit indices, then
bf16 scales) at O_DIRECT-aligned stride (VQ_ALIGN = 4096); index `n_experts` is the
shared expert. Validated on open; a dim/version mismatch fails loud.

## Kernels (kernels/*.hip, gated on `rocm`)

- `common.hpp` — wave helpers, bf16/e4m3, the fp16-codebook `dot_vq_wave`, `dot_i4_wave`,
  `dot_fp8_wave` + the shared e4m3 LUT.
- `moe.hip` — `moe_gateup_vq`/`moe_down_vq` + the int4 twins, `moe_reduce` (3 codebooks).
- `mla.hip` — `mla_absorb_fp8` / `mla_value_fp8` (fp8 kv_b).
- `linalg.hip` — `gemv_vq`, `gemv_fp8` (wave-per-row + split-K for long reductions),
  `gemv_f32`, `gemv_i8`, `swiglu`, `rmsnorm`, `rope`, `vq_encode`, `gemv_i4` (microbench).
- `attn.hip` — flash attend; latent cache is **fp8-e4m3 + per-128 block scales**, roped
  key bf16. Optional `rows` gather for the sparse (DSA/streaming) path.
- `indexer.hip` — DSA lightning indexer (layernorm / index_append / index_score /
  index_pool_push / index_head_route).
- `fwd.hip` (embed_i8, append_kv fp8-latent + bf16 key, gather_rope, vadd, argmax),
  `vmm.hip`, `async.hip` (HIP stream/event/host-func bridge + the io_uring bounce arena's
  pinned-alloc + async-H2D helpers).

The cold-expert io_uring streamer is the `io-uring` crate in `src/stream.rs` (no
liburing system lib).

## Rust modules (src/)

- `math`, `model` (ModelConfig) — leaf.
- `format` — the artifact reader: manifest + `resident.safetensors` + `.vq3`/`.i4`
  mmap/index; single source of truth for the on-disk layout (mirrors the converter).
- `quant` — VQ encode/decode + fp8/int8/int4 decode oracles.
- `device` (DeviceTier/DeviceBuf/VmmBuf), `stream` (io_uring O_DIRECT streamer).
- `arena` — the two-ended byte arena backing the routed pool (cold packs low, hot high).
- `hybrid` — byte-aware residency policies (`HybridPolicy`: HybridLru/TwoQ/Arc) that
  own eviction and place each miss in the right slab.
- `cache` — single-format / offline-replay pool policies (2Q/LRU/ARC).
- `hip` — FFI surface for the kernels above.
- `pin` — resident placement (fp8 attn/dense, int8 embed/lm_head, f32 norms/gate, VQ
  shared, codebooks, optional indexer) + the routed-expert streaming pool.
- `gpu` — the decode forward pass (fp8 attention + MLA, VQ/int4 experts, one MoE launch).
- `asyncfetch` / `gpustream` — the async load‖compute overlap: GPU/io_uring completions
  bridged to futures via async signals; current-thread runtime, GPU streams are the
  concurrency.
- `attn` — attention mode (Dense / Streaming / Dsa / Misa).
- `indexer` — DSA indexer constants + device-side selection.
- `telemetry` — the always-on per-token PROFILE summary (+ optional OTLP span).
- `tokenizer`, `watchdog`, `config` (the artifact is the model — no snapshot/vq-dir knobs).

## Tools (src/bin/)

- `convert` — fp8 checkpoint → the int3-vq artifact (learns codebooks, GPU-encodes).
- `pack_i4` — colibri-int4 source → per-layer `L{l}.i4` twins (repack, not requant).
- `add_indexer` — write the DSA `indexer.safetensors` side file into an artifact.
- `replay` — offline cache-policy sim over a `--trace` dump.
