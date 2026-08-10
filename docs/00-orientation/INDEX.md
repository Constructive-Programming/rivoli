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
| [`glimmer-architecture.md`](../reference/glimmer-architecture.md) | live | Muse Glimmer-30B's forward pass, precise enough to write kernels from — extracted 2026-08-10 from first-party sources only (raw config.json, the safetensors headers by range request, and transformers' own modeling_muse_glimmer.py), no summarizing fetch in the chain. 52 dense layers, 39 sliding (window 2048) + 13 full at zero-based 3,7,...,51, GQA 32Q/2KV head_dim 128. SANDWICH NORMS: four per layer, post-norms on the BRANCH before the residual add, and they are CENTERED (x*(1+w)) while the final norm and the two weightless norms are plain (x*w) — two formulas in one model. Q and K carry a WEIGHTLESS RMSNorm that ships no tensor (with_scale=False) and Q alone is then scaled by qk_scale_factor 3.87. RoPE is rotate_half (split-half), NOT rivoli's interleaved convention — a row permutation of q_proj/k_proj converts it, argued in section 6 and unproven. NoPE layers skip rotation entirely. Attention output is gated by sigmoid(gate_proj(layer input)) BEFORE o_proj. Logits are 20*tanh(x*0.196116/20), which is argmax-invariant — so every greedy gate is BLIND to it. Text side 55.71 GB bf16 = 26.51 GB/token at fp8. Fifteen traps in section 9. The DFlash drafter is a SEPARATE 2.556 B checkpoint (section 11): a 5-layer BIDIRECTIONAL cross-attention adapter that shares almost nothing with the target (32Q/8KV, plain pre-norm, weighted QK-norm) and borrows the target's embedding UNNORMED plus its lm_head; because the target is dense, break-even is N>1.1 accepted tokens per cycle, inverting the MoE-union economics that made ungated MTP a loss on GLM. |
| [`modes.md`](../reference/modes.md) | live | The --mode × --cache-policy matrix and which knob does what. Quality ladder: int4 5.120 > hybrid 5.189 > int3-vq 5.275. |
| [`vulkan-kernels.md`](../reference/vulkan-kernels.md) | live | What the Vulkan backend has and what binds anyone editing a shader: 16 of 29 kernels, ~1.9x slower on --mode int3-vq --attn dense, the device requirements, the numerics and index-width rules, the mechanised-guard registry, and two OPEN fp8-dot gaps. |
| [`serving.md`](../reference/serving.md) | live | The OpenAI HTTP server (--port). Thinking defaults OFF and is a prompt prefill, not a flag; tool calling works; sampling and /v1/completions do not, on purpose. |
| [`gpu-lock.md`](../reference/gpu-lock.md) | live | llama-swap's Vulkan pod on rh-anine now shares the SAME /var/run/sys-gpu.lock every bare-metal GPU command already flocks (TOUR.md, docs/measurement/) — no rivoli code changes, this documents the other side of an existing contract. |

## measurement/ — how to measure, and what was measured

| doc | status | verdict |
|---|---|---|
| [`how-to-measure.md`](../measurement/how-to-measure.md) | live | How to measure, and the four-out-of-five lesson that says why the ISA beats a profile. |
| [`perf-roadmap.md`](../measurement/perf-roadmap.md) | live | The ranked performance roadmap, re-scored 2026-08-04 on recurring cost rather than wall at today's bottleneck. Live rows: #2 VQ_K=2048 (1.189x MoE kernels at 12-bit, +18.7% relFrob, needs a real dNLL gate), #10 general-R MoE kernels. #5 DONE 2026-08-02 (HB 8→16, 2.08x kernel, −3.2 ms/tok, gated). #8 and #11 stay closed on complexity and quality — NOT on "bytes stop buying anything below the floor", which was the wrong axis. |
| [`traces.md`](../measurement/traces.md) | live | The --features otlp instrument: its three switches, what the engine emits, and how to read a trace. Verified end to end 2026-08-01, after it had stopped compiling. |
| [`gpu-profiling.md`](../measurement/gpu-profiling.md) | live | Why ROCm GPU profiling does not work on this part, and what to do instead. Only if you are attaching a profiler. |
| [`benchmarks.md`](../measurement/benchmarks.md) | data | Append-only measurements. Never read whole — grep for the config. The top table predates the .i4 rebuild and says so. |
| [`probes/`](../measurement/probes/README.md) | data | Standalone HIP probes that reproduce engine behaviour outside the engine, and what each one settled. |

## investigations/ — asked, answered, closed

Read the verdict. Open the file only if you are about to re-open the question. **Two rows are
`live`** — open proposals, not yet built; each moves to `closed-shipped` when it is executed.

| doc | status | verdict |
|---|---|---|
| [`glimmer-port.md`](../investigations/glimmer-port.md) | live | The implementation plan for Muse Glimmer-30B, the fourth model, sequenced after K3. A DENSE 52-layer port that bypasses the streaming machinery entirely — 26.51 GB/token fully resident at fp8 (measured from the shard headers), so the ceiling is GTT bandwidth, not NVMe, and the residency stage is deleted outright. S0 DONE and G0 MET 2026-08-10; the spec is reference/glimmer-architecture.md. S1 is blocked on K3's S1a landing the Arch seam, not on anything here. Reuse is high everywhere except attention: GQA 32Q/2KV + sliding-window locals + the sigmoid output gate is a new kernel family (rivoli is MLA-only), and S0 added a second body of work the card hid — four sandwich norms per layer in a CENTERED x*(1+w) form the engine has never implemented, plus a weightless QK-norm that ships no tensor. RoPE may cost nothing: a q/k row permutation converts split-half to rivoli's interleaved kernel, argued but unproven. DFlash break-even is N>1.1 accepted tokens because dense verification reads each weight once — the inverse of GLM's MoE-union economics, where ungated MTP was a 0.93x loss. |
| [`otlp-modernization.md`](../investigations/otlp-modernization.md) | live | PARTLY BUILT. Keep OTLP — measured, no leaner path exists at 0.30 and it costs 64 crates. Run-identity labels {mode,cache_policy,attn,max_mem_gib,mtp} SHIPPED 2026-08-02, as did the §3 drops; MTP acceptance and moe-by-miss are still proposed. |
| [`int4-scales.md`](../investigations/int4-scales.md) | closed-shipped | Why int4 was unusable and how group-128 scales fixed it: PPL 73.43 → 5.120, making int4 the best-quality mode. RESOLVED. |
| [`vulkan-port.md`](../investigations/vulkan-port.md) | closed-shipped | Porting the engine to Vulkan across four phases — the journal, not the rules. Shipped and decoding; the live inventory AND every standing shader obligation moved to reference/vulkan-kernels.md on 2026-08-01. |
| [`cache-conditional-routing.md`](../investigations/cache-conditional-routing.md) | closed-negative | top-m routing: RETIRED 2026-07-30. Cost +3.63% PPL on int3-vq and +12.7% on int4 against a ~1% bar, and made every cache change an output change. |
| [`cross-layer-prefetch.md`](../investigations/cross-layer-prefetch.md) | closed-negative | LOOKA hints and the pilot prefetcher: REMOVED 2026-07-31. The veto bound on 0.9% of evictions; the prefetcher predicted at 99% precision and still cost more than it saved. |
| [`codebook-rotation.md`](../investigations/codebook-rotation.md) | closed-negative | Hadamard/QuIP rotation for int3-vq: CLOSED 2026-08-01. A per-layer codebook recovers 0.09% against a 2% bar, so there is nothing to homogenise. int3-vq is rate-limited. |
| [`npu-offload.md`](../investigations/npu-offload.md) | closed-negative | DSA indexer offload to the NPU: not worth it. The answer is in the first 40 lines; the device top-k it recommended shipped instead (−9.4 ms/token). |
| [`perf-evidence.md`](../investigations/perf-evidence.md) | closed-mixed | Phase profile and per-kernel tranches behind the roadmap. The "Where the time goes" block is STALE and inverted — see its banner. |

## If you are writing here

- **Correct in place with a dated note.** Do not delete and do not silently overwrite. What
  an investigation ruled out is worth as much as what it found, and a correction that erases
  the prior belief makes the next reader repeat the experiment.
- **Update `status:` and `verdict:` when the answer changes**, and update the row here. The
  test will tell you if you forget one of the two.
- **A doc that goes from live to closed moves directory.** That move is the signal; leaving
  a dead mechanism described under `reference/` is how `PERF.md` came to claim the engine was
  compute-bound for a month after it wasn't.
