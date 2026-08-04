---
status: live
verdict: The index. Every doc, its status, and its one-line verdict — so you can decide what NOT to open.
---

# docs index — read this, not the directory

~500 KB lives under `docs/`, and **most of it is closed investigation kept for what it
eliminated.** The verdict column exists so you can rule a file out without opening it. If
a verdict answers your question, you are done — that is the intended outcome, not a
shortcut.

Every file carries `status:` and `verdict:` front matter, and `tests/docs.rs` fails if this
table and that front matter disagree. So this page cannot quietly go stale the way the
`docs/README.md` it replaces did.

**`status:` values.** `live` — true about the engine today; a wrong one is a defect.
`closed-negative` — tried, measured, rejected; the code is gone. `closed-shipped` — tried,
measured, kept; the result is in the engine. `closed-mixed` — parts of both, banner inside.
`data` — measurements and instruments, not prose.

## Start here

| | |
|---|---|
| [`TOUR.md`](TOUR.md) | **New here? This, then stop.** The engine in two pages: what it is, why it streams, and the five things that will bite you. |

## reference/ — the engine as it is

| doc | status | verdict |
|---|---|---|
| [`architecture.md`](../reference/architecture.md) | live | The engine as it is. The one doc meant to be read whole; §8b is the INV registry, enforced by tests/invariants.rs. |
| [`modes.md`](../reference/modes.md) | live | The --mode × --cache-policy matrix and which knob does what. Quality ladder: int4 5.120 > hybrid 5.189 > int3-vq 5.275. |
| [`vulkan-kernels.md`](../reference/vulkan-kernels.md) | live | What the Vulkan backend has and what binds anyone editing a shader: 16 of 29 kernels, ~1.9x slower on --mode int3-vq --attn dense, the device requirements, the numerics and index-width rules, the mechanised-guard registry, and two OPEN fp8-dot gaps. |
| [`serving.md`](../reference/serving.md) | live | The OpenAI HTTP server (--port). Thinking defaults OFF and is a prompt prefill, not a flag; tool calling works; sampling and /v1/completions do not, on purpose. |

## measurement/ — how to measure, and what was measured

| doc | status | verdict |
|---|---|---|
| [`how-to-measure.md`](../measurement/how-to-measure.md) | live | How to measure, and the four-out-of-five lesson that says why the ISA beats a profile. |
| [`perf-roadmap.md`](../measurement/perf-roadmap.md) | live | The ranked performance roadmap. Live rows: #2 VQ_K codebook, #5 the MLA HB sweep. |
| [`traces.md`](../measurement/traces.md) | live | The --features otlp instrument: its three switches, what the engine emits, and how to read a trace. Verified end to end 2026-08-01, after it had stopped compiling. |
| [`gpu-profiling.md`](../measurement/gpu-profiling.md) | live | Why ROCm GPU profiling does not work on this part, and what to do instead. Only if you are attaching a profiler. |
| [`benchmarks.md`](../measurement/benchmarks.md) | data | Append-only measurements. Never read whole — grep for the config. The top table predates the .i4 rebuild and says so. |
| [`probes/`](../measurement/probes/README.md) | data | Standalone HIP probes that reproduce engine behaviour outside the engine, and what each one settled. |

## investigations/ — asked, answered, closed

Read the verdict. Open the file only if you are about to re-open the question. **Two rows are
`live`** — open proposals, partly built; each moves to `closed-shipped` when it is executed.

| doc | status | verdict |
|---|---|---|
| [`v4-flash-port.md`](../investigations/v4-flash-port.md) | live | The staged plan to make V4-Flash decode. S1 LANDED 2026-08-05 (.f4 repack bit-exact over 10.27 GB; a 137-golden CPU oracle with five measured blind spots). Corrects other-models.md from the real repo: experts are 148.25 GB native FP4 (138.1 GiB) so it DOES stream at ~83% residency, not "nearly fully resident"; 3.449 GB/token, since the shared expert is fp8 and resident, not FP4 and streamed; the partial fp8 KV act_quant is mandatory, not a --kv-fp8 to refuse (that flag does not exist); YaRN is per-layer, keyed to compress_ratio. DSpark/MTP is separable and out of scope. |
| [`otlp-modernization.md`](../investigations/otlp-modernization.md) | live | PARTLY BUILT. Keep OTLP — measured, no leaner path exists at 0.30 and it costs 64 crates. Run-identity labels {mode,cache_policy,attn,max_mem_gib,mtp} SHIPPED 2026-08-02, as did the §3 drops; MTP acceptance and moe-by-miss are still proposed. |
| [`int4-scales.md`](../investigations/int4-scales.md) | closed-shipped | Why int4 was unusable and how group-128 scales fixed it: PPL 73.43 → 5.120, making int4 the best-quality mode. RESOLVED. |
| [`vulkan-port.md`](../investigations/vulkan-port.md) | closed-shipped | Porting the engine to Vulkan across four phases — the journal, not the rules. Shipped and decoding; the live inventory AND every standing shader obligation moved to reference/vulkan-kernels.md on 2026-08-01. |
| [`cache-conditional-routing.md`](../investigations/cache-conditional-routing.md) | closed-negative | top-m routing: RETIRED 2026-07-30. Cost +3.63% PPL on int3-vq and +12.7% on int4 against a ~1% bar, and made every cache change an output change. |
| [`cross-layer-prefetch.md`](../investigations/cross-layer-prefetch.md) | closed-negative | LOOKA hints and the pilot prefetcher: REMOVED 2026-07-31. The veto bound on 0.9% of evictions; the prefetcher predicted at 99% precision and still cost more than it saved. |
| [`codebook-rotation.md`](../investigations/codebook-rotation.md) | closed-negative | Hadamard/QuIP rotation for int3-vq: CLOSED 2026-08-01. A per-layer codebook recovers 0.09% against a 2% bar, so there is nothing to homogenise. int3-vq is rate-limited. |
| [`npu-offload.md`](../investigations/npu-offload.md) | closed-negative | DSA indexer offload to the NPU: not worth it. The answer is in the first 40 lines; the device top-k it recommended shipped instead (−9.4 ms/token). |
| [`perf-evidence.md`](../investigations/perf-evidence.md) | closed-mixed | Phase profile and per-kernel tranches behind the roadmap. The "Where the time goes" block is STALE and inverted — see its banner. |
| [`other-models.md`](../investigations/other-models.md) | live | Both targets keep source fidelity — V4-Flash native FP4 in a new `.f4` container (~157 GB), K3 native MXFP4 (~1.45 TB) — because both ship 4-bit experts and re-quantizing to int3-vq is the lossy-on-lossy chain int4-scales.md records at PPL 73.43. NFS measures 154 MB/s against NVMe's 7.0 GB/s, so `/swarm` is the library and NVMe the working set; all three at native fidelity are 1.73 TiB against 1.69 TiB of disk. FOUR CORRECTIONS 2026-08-05 from the downloaded repo, each dated in place: V4 DOES stream (138.1 GiB of experts, ~83% residency) so §3 is inverted; `--kv-fp8` does not exist and the partial fp8 KV act_quant is mandatory not forbidden; §7's "unverified absence" is resolved and missed a 41-layer KV compressor; the shared expert is fp8, not FP4. |

## If you are writing here

- **Correct in place with a dated note.** Do not delete and do not silently overwrite. What
  an investigation ruled out is worth as much as what it found, and a correction that erases
  the prior belief makes the next reader repeat the experiment.
- **Update `status:` and `verdict:` when the answer changes**, and update the row here. The
  test will tell you if you forget one of the two.
- **A doc that goes from live to closed moves directory.** That move is the signal; leaving
  a dead mechanism described under `reference/` is how `PERF.md` came to claim the engine was
  compute-bound for a month after it wasn't.
