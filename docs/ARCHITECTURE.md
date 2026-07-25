# rivoli-vq — int3-vq-only decode engine (rewrite)

A clean reimplementation of the GLM-5.2 MoE decode engine for AMD Strix Halo
(gfx1151), dropping int4 entirely. Built from the accumulated knowledge of the
`int3-vq` branch, which remains the reference (`git show int3-vq:<path>`).

## Weight formats (no int4 anywhere)

| weights | format | resident? | why |
|---|---|---|---|
| routed + shared experts | **VQ-int3** (12-bit codebook idx + bf16 g64 scale) | streamed | the bandwidth driver; smaller = more residency = faster |
| attention projections (q_a/q_b/kv_a/kv_b/o_proj) | **fp8** e4m3 + 128-block scale | resident | native source precision, no requant loss; format-neutral to tok/s (resident) |
| dense-layer MLPs (first `dense` layers) | **fp8** | resident | consistent with attention; small |
| embed / lm_head | **int8** + per-row scale | resident | unchanged; not int4 |
| norms, router gate | **f32** | resident | tiny |
| DSA indexer (wk/wq_b/weights_proj/k_norm) | **bf16** | resident | unchanged |

Codebooks are **per-projection** (gate/up/down) — the down_proj distribution
differs from gate/up; separate codebooks are the quality lever. 3 codebooks,
learned once, resident (~192 KiB total).

## On-disk artifact — one self-contained directory (built from the fp8 checkpoint)

```
<model>/
  manifest.json        # self-describing: format version, full ModelConfig,
                       #   VQ params, codebook layout, resident-blob offset table
  codebooks.f32        # 3 × VQ_K × VQ_DIM f32 (gate, up, down)
  resident.bin         # every resident weight, laid out for direct tier placement:
                       #   per layer {fp8 attn proj + block scales, fp8 dense MLP,
                       #   f32 norms, bf16 indexer, f32 router gate}; global
                       #   {int8 embed, int8 lm_head, f32 final norm}
  L{03..NN}.vq3        # one file per MoE layer, self-describing header +
                       #   (n_experts + 1) expert blocks at O_DIRECT-aligned stride;
                       #   block = gate‖up‖down (12-bit indices, then bf16 scales);
                       #   index n_experts = the shared expert (folded into one launch)
```

`.vq3` per-file header (little-endian): magic `"VQ3\0"`, u32 version, u32 layer,
u32 n_experts (incl. shared), u32 hidden, u32 moe_inter, u64 stride, then the six
projection sub-offsets. Validated on open; a dim/version mismatch fails loud.

## Kernels (kernels/*.hip, gated on `rocm`)

Keep + prune: `common.hpp` (wave helpers, bf16/e4m3, **dot_vq_wave**; drop
dot_i4/dot_i8), `attn.hip` (dense-only — no sparse gather: this checkpoint has no
DSA indexer, so the indexer + StreamingLLM paths are dropped; the latent cache is
**fp8-e4m3 + per-128 block scales** (single path, no bf16 KV), roped key bf16),
`fwd.hip` (embed_i8, append_kv fp8-latent + bf16 key, gather_rope, vadd, argmax),
`vmm.hip`, `async.hip` (HIP stream/event/host-func bridge + the io_uring bounce
arena's pinned-alloc + async-H2D helpers). The cold-expert io_uring streamer is the
`io-uring` crate in `src/stream.rs` (no liburing). **Dropped: `indexer.hip`.**

Rewrite for the new formats:
- `moe.hip` — `moe_gateup_vq`/`moe_down_vq`/`moe_reduce` taking the **3 per-projection
  codebooks**. (drop int4 moe)
- `linalg.hip` — `gemv_vq`, `gemv_f32`, `rmsnorm`, `rope`, `vq_encode`; **new** `gemv_fp8`.
  (drop gemv_i4/gemv_i8)
- `mla.hip` — **new** `mla_absorb_fp8` / `mla_value_fp8` (kv_b is fp8 now). (drop int4)

## Rust modules (src/)

Keep, cleaned: `math`, `model` (ModelConfig), `device` (DeviceTier/DeviceBuf/VmmBuf),
`stream` (io_uring), `cache` (2Q pool), `tokenizer`, `watchdog`, `config` (no
snapshot/vq-dir/attn/kv-fp8 knobs; the artifact is the model).

Dropped modules: `attn` (dense-only ⇒ `AttnMode` collapses to `pos+1`, and the CPU
int4 `attention()` oracle is redundant with `tests/kernel.rs`), `indexer` (no DSA
indexer in this checkpoint), `snapshot` (int4 loader; replaced by `format`).

Rewrite:
- `format` (new) — the artifact reader: manifest + resident.bin + `.vq3` mmap/index,
  the single source of truth for the on-disk layout (mirrors the converter).
- `quant` — VQ encode/decode oracle + fp8/int8 decode helpers. (drop int4)
- `hip` — FFI surface for the kernel set above. (drop int4 launchers, add fp8)
- `pin` — resident placement (fp8 attn/dense, int8 embed) + VQ expert streaming pool.
- `gpu` — the decode forward pass (fp8 attention, VQ experts, one MoE launch).
- `main` / `config` — no int4/`--vq-dir`; the artifact IS the model.

Drop: int4 everything, the hybrid colibri-snapshot dependency, `--vq-dir` (the model
is now natively VQ), the OTLP telemetry stack, `--pre-seed`/`usage` unless it pulls
its weight, most `#[cfg(trace)]` diagnostics.

New tool: `bin/convert` — fp8 checkpoint → the unified artifact (learns the 3
codebooks, GPU-encodes experts + shared, writes fp8 resident blob + `.vq3` files +
manifest). Reuses the validated `vq_encode` GPU path.

Keep tool: `bin/replay` — offline cache-policy sim (still relevant).

## Build order (each step compiles + checkpoints)

1. Cargo.toml + build.rs (pruned)
2. leaf modules: math, model
3. quant (VQ + fp8/int8 helpers) + format (manifest + header types)
4. converter (produces a real artifact to test the loader against)
5. kernels + hip FFI (VQ moe, fp8 gemv/mla, reused attn/indexer)
6. device + stream + cache (mostly moved)
7. pin (resident fp8 + VQ streaming) + gpu (forward) + main/config
8. end-to-end run; benchmark vs the int3-vq branch
