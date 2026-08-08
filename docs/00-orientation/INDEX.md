---
scope: engine
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

**`scope:` values — whose evidence backs the verdict.** `glm` — measured on GLM-5.2 only;
`v4` — the DeepSeek-V4 port; `engine` — model-independent (tooling, method, infra). **A
closed verdict rules its question out only for its scope.** Added 2026-08-07 after
`npu-offload.md`'s GLM-only closed-negative was read as engine-wide; a `glm`-scoped closure
says nothing about V4 and must be re-evaluated there before it forecloses anything.

## Start here

| | |
|---|---|
| [`TOUR.md`](TOUR.md) | **New here? This, then stop.** The engine in two pages: what it is, why it streams, and the five things that will bite you. |

## reference/ — the engine as it is

| doc | scope | status | verdict |
|---|---|---|---|
| [`architecture.md`](../reference/architecture.md) | glm | live | The engine as it is. The one doc meant to be read whole; §8b is the INV registry, enforced by tests/invariants.rs. |
| [`modes.md`](../reference/modes.md) | glm | live | The --mode × --cache-policy matrix and which knob does what. Quality ladder: int4 5.120 > hybrid 5.189 > int3-vq 5.275. |
| [`serving.md`](../reference/serving.md) | glm | live | The OpenAI HTTP server (--port). Thinking defaults OFF and is a prompt prefill, not a flag; tool calling works; sampling and /v1/completions do not, on purpose. |
| [`gpu-lock.md`](../reference/gpu-lock.md) | engine | live | llama-swap's pod shares /var/run/sys-gpu.lock — but only for SOME models (CORRECTED 2026-08-07, verified live 3x - whisper and the embedding model hold GTT with no lock). Until hr-fleet wraps every cmd, a GPU witness must sample mem_info_gtt_used, not just the flock and KFD. |

## measurement/ — how to measure, and what was measured

| doc | scope | status | verdict |
|---|---|---|---|
| [`how-to-measure.md`](../measurement/how-to-measure.md) | engine | live | How to measure, and the four-out-of-five lesson that says why the ISA beats a profile. |
| [`perf-roadmap.md`](../measurement/perf-roadmap.md) | glm | live | The ranked performance roadmap, re-scored 2026-08-04 on recurring cost rather than wall at today's bottleneck. Live rows: #2 VQ_K=2048 (1.189x MoE kernels at 12-bit, +18.7% relFrob, needs a real dNLL gate), #10 general-R MoE kernels. #5 DONE 2026-08-02 (HB 8→16, 2.08x kernel, −3.2 ms/tok, gated). #8 and #11 stay closed on complexity and quality — NOT on "bytes stop buying anything below the floor", which was the wrong axis. |
| [`traces.md`](../measurement/traces.md) | engine | live | The --features otlp instrument: its three switches, what the engine emits, and how to read a trace. Verified end to end 2026-08-01, after it had stopped compiling. |
| [`gpu-profiling.md`](../measurement/gpu-profiling.md) | engine | live | Why ROCm GPU profiling does not work on this part, and what to do instead. Only if you are attaching a profiler. |
| [`benchmarks.md`](../measurement/benchmarks.md) | engine | data | Append-only measurements. Never read whole — grep for the config. The top table predates the .i4 rebuild and says so. |
| [`probes/`](../measurement/probes/README.md) | glm | data | Standalone HIP probes that reproduce engine behaviour outside the engine, and what each one settled. |

## investigations/ — asked, answered, closed

Read the verdict. Open the file only if you are about to re-open the question. **Five rows are
`live`** — open proposals, partly built; each moves to `closed-shipped` when it is executed.
(This said "Two" while three were live, and two more arrived 2026-08-07. Count the rows, not
the sentence.)

| doc | scope | status | verdict |
|---|---|---|---|
| [`refactor-2026-08.md`](../investigations/refactor-2026-08.md) | engine | live | A staged 39% code reduction (36,100 -> ~22,050 lines) over two waves and nine tracks. Retiring the Vulkan backend is 6,600 of it and must run ALONE because its cfg sites reach 21 files. ~43% of this tree is TEST code and is not a line-count target: consolidating duplicated scaffolding is in scope, cutting coverage is not. 50% remains unreachable without dropping features beyond Vulkan. |
| [`v4-decode-decomposition.md`](../investigations/v4-decode-decomposition.md) | v4 | live | OPEN, three levers LANDED and the route span SPLIT. M3b (2026-08-07, launch geometry): moe 70.6 → 54.7 ms/token, +9.6% tok/s. M3c (2026-08-08, branchless e2m1/e8m0 + global-AS descriptor loads, fp4 dot loop 195 → 105 instr per 128 weight bytes): moe 54.9 → 49.6, wall 165.3 = 6.048 tok/s — the registered prediction (−4..−10, point −6) landed IN BAND at −5.3, the first of the series to do so. Output byte-identical and counters identical across every arm. M4 (2026-08-08) split route with four event-pair sub-spans, no new join: attn 53.2 | cmp 9.1 | hcn 41.2 | gate 3.2 | win 107.7 (resid 1.0) — WRONG at the headline: hcn (hyper-connections + norms) measured 41.2 against a 4–8 band and 0.8 ms of bytes, the engine's largest above-bytes excess and the new #1 lever (read the hc/norm kernels host-side next); fp8 GEMV rate demoted to #2; cmp and gate closed at budget; resid 1.0 kills the gate-D2H width micro-lever. Residue to 10 tok/s (~65 ms off the 165.3 wall): hcn +40.4 > moe +31.6 over its 18 ms byte floor (miss exposure, shared GEMV unpriced) > attn +29.3 > remainder ~31 non-tail. M2's floor (~62 ms/token ≈ 16 tok/s ceiling) stands; buckets still not certified free (wall spread ±1.5% over stock-class runs). |
| [`v4-flash-port.md`](../investigations/v4-flash-port.md) | v4 | live | The staged plan to make V4-Flash decode. S1 LANDED 2026-08-05 (.f4 repack bit-exact over 10.27 GB; a 137-golden CPU oracle with five measured blind spots). Corrects other-models.md from the real repo: experts are 148.25 GB native FP4 (138.1 GiB) so it DOES stream at ~83% residency, not "nearly fully resident"; 3.449 GB/token, since the shared expert is fp8 and resident, not FP4 and streamed; the partial fp8 KV act_quant is mandatory, not a --kv-fp8 to refuse (that flag does not exist); YaRN is per-layer, keyed to compress_ratio. DSpark/MTP is separable and out of scope. The LAYER LOOP LANDED 2026-08-05 (src/v4gpu.rs + a main.rs V4 branch + a real-weight per-layer gate) and has NOT yet run on a device; three deviations from the reference are named at their call sites (unclamped shared expert, positional block selection on the ratio-4 layers, un-rounded MoE output) and reviews caught two criticals before the GPU did. The dev-profile sweep is also RED at a2504eb. |
| [`otlp-modernization.md`](../investigations/otlp-modernization.md) | engine | live | PARTLY BUILT. Keep OTLP — measured, no leaner path exists at 0.30 and it costs 64 crates. Run-identity labels {mode,cache_policy,attn,max_mem_gib,mtp} SHIPPED 2026-08-02, as did the §3 drops; MTP acceptance and moe-by-miss are still proposed. |
| [`ffn-norm-out-envelope.md`](../investigations/ffn-norm-out-envelope.md) | v4 | live | OPEN. `ffn_norm_out` and `.out` carry no bound at all since 2026-08-07 — they report and assert only that the row was reached. The 5e-2 they used to carry was the same constant four attention tensors were re-derived away FROM, whose derived values came out 17, 275, 23 and 71. Two substitutes have been measured and REFUTED: the differing-element fraction at 1.42x (probe sweep), and, later the same day, a perturbed-golden A/B through the gate itself — a `SinkhornIterCountProbe` golden ran the gate green, with a same-tensor fraction separation of only 1.20x. The work is transcribing `hc_post` + the MoE to compute the envelope, unblocked since Track 0 released the files on 2026-08-06. |
| [`real-weights-defect-goldens.md`](../investigations/real-weights-defect-goldens.md) | v4 | closed-mixed | ANSWERED 2026-08-07, built and measured the same day — `emit --defect` exists, the loader refuses a mismatched golden (proven red live), and all seven gate arms ran with every red/green outcome matching its pre-registered prediction. The four derived attention bounds were scored THROUGH the gate for the first time; each anchor defect went red, but `attn_derot`'s 1.3x separation survived by only 1.07x at worst (24.5 vs 23), and both kv-quant defects left their home tensor `kv_entry` GREEN under its 17 bound — every red came from downstream. `SinkhornIterCountProbe` ran fully green, settling unlock #3 negatively — no live bound sees the one defect the checkpoint discriminates and the toy cannot; the gate saw its `ffn_norm_out` movement (69.6% of elements differing) only in an unbounded reported row — so `ffn-norm-out-envelope.md` still owes the transcription and cannot be closed by a perturbed golden alone. |
| [`int4-scales.md`](../investigations/int4-scales.md) | glm | closed-shipped | Why int4 was unusable and how group-128 scales fixed it: PPL 73.43 → 5.120, making int4 the best-quality mode. RESOLVED. |
| [`vulkan-port.md`](../investigations/vulkan-port.md) | glm | closed-negative | Porting the engine to Vulkan across four phases — the journal, not the rules. It shipped and decoded, then was RETIRED 2026-08-06 as an unfinished port: 16 of 29 kernels, 6 of its own 36 mode-matrix cells decoding (of 72; 36 per backend), ~1.9x slower, no DeepSeek-V4 path. Code at tag archive/vulkan-backend-hb16; the inventory and the shader rules are in vulkan-kernels.md. |
| [`vulkan-kernels.md`](../investigations/vulkan-kernels.md) | glm | closed-negative | RETIRED 2026-08-06 as an unfinished port, not a feature — 16 of 29 kernels, 6 of its own 36 mode-matrix cells decoding and 30 refusing (of 72; 36 per backend), ~1.9x slower on --mode int3-vq --attn dense, and no DeepSeek-V4 path at all. Kept as the inventory of what was and was not ported, the numerics and index-width rules, and the mechanised-guard registry; code at tag archive/vulkan-backend-hb16. |
| [`cache-conditional-routing.md`](../investigations/cache-conditional-routing.md) | glm | closed-negative | top-m routing: RETIRED 2026-07-30. Cost +3.63% PPL on int3-vq and +12.7% on int4 against a ~1% bar, and made every cache change an output change. |
| [`cross-layer-prefetch.md`](../investigations/cross-layer-prefetch.md) | glm | closed-negative | LOOKA hints and the pilot prefetcher: REMOVED 2026-07-31. The veto bound on 0.9% of evictions; the prefetcher predicted at 99% precision and still cost more than it saved. |
| [`codebook-rotation.md`](../investigations/codebook-rotation.md) | glm | closed-negative | Hadamard/QuIP rotation for int3-vq: CLOSED 2026-08-01. A per-layer codebook recovers 0.09% against a 2% bar, so there is nothing to homogenise. int3-vq is rate-limited. |
| [`npu-offload.md`](../investigations/npu-offload.md) | glm | closed-negative | DSA indexer offload to the NPU: not worth it — M1a (measured 2026-08-07) closed the decoupled window for GLM's DSA indexer (verbatim 1-step-stale costs +0.89 nats; only the diagonal-patched variant is unmeasured). The device top-k it recommended shipped instead (−9.4 ms/token). |
| [`perf-evidence.md`](../investigations/perf-evidence.md) | glm | closed-mixed | Phase profile and per-kernel tranches behind the roadmap. The "Where the time goes" block is STALE and inverted — see its banner. |
| [`other-models.md`](../investigations/other-models.md) | engine | live | Both targets keep source fidelity — V4-Flash native FP4 in a new `.f4` container (~157 GB), K3 native MXFP4 (~1.45 TB) — because both ship 4-bit experts and re-quantizing to int3-vq is the lossy-on-lossy chain int4-scales.md records at PPL 73.43. NFS measures 154 MB/s against NVMe's 7.0 GB/s, so `/swarm` is the library and NVMe the working set; all three at native fidelity are 1.73 TiB against 1.69 TiB of disk. FOUR CORRECTIONS 2026-08-05 from the downloaded repo, each dated in place: V4 DOES stream (138.1 GiB of experts, ~83% residency) so §3 is inverted; `--kv-fp8` does not exist and the partial fp8 KV act_quant is mandatory not forbidden; §7's "unverified absence" is resolved and missed a 41-layer KV compressor; the shared expert is fp8, not FP4. |

## If you are writing here

- **Correct in place with a dated note.** Do not delete and do not silently overwrite. What
  an investigation ruled out is worth as much as what it found, and a correction that erases
  the prior belief makes the next reader repeat the experiment.
- **Update `status:` and `verdict:` when the answer changes**, and update the row here. The
  test will tell you if you forget one of the two.
- **A doc that goes from live to closed moves directory.** That move is the signal; leaving
  a dead mechanism described under `reference/` is how `PERF.md` came to claim the engine was
  compute-bound for a month after it wasn't.
