# Models

rivoli targets large mixture-of-experts models. Supporting a model means an artifact
converter, a per-architecture decode loop, and first-party reference goldens that the loop
is scored against. Which models belong to which code path is data in a checked census
table, not something encoded in code names.

## At a glance

| model | params[^aa] | experts[^exp] | attention | formats | tok/s[^box] | Intelligence Index[^aa] |
|---|---|---|---|---|---|---|
| GLM-5.2 | 753B / 40B | 256 (8) | MLA[^mla] | VQ3, INT4[^vq] | 3.0[^glm-tps] | [N/A](https://artificialanalysis.ai/models/glm-5-2)[^na] |
| DeepSeek-V4-flash | 284B / 13B | 256+1 (6) | MQA, SWA + blocks[^v4] | FP4 | 9.2[^v4-tps] | [N/A](https://artificialanalysis.ai/models/deepseek-v4-flash)[^na] |
| Kimi-K3 | 2.8T / 104B | 896+2 (16) | KDA + MLA[^k3] | MXFP4[^mx] | —[^none] | [17](https://artificialanalysis.ai/models/kimi-k3)[^k3-ii] |
| Muse Glimmer | 30B dense | — | GQA, SWA[^gl] | BF16, FP8[^fp8] | —[^gl-tps] | [6](https://artificialanalysis.ai/models/muse-glimmer)[^gl-ii] |
| Inkling 2 Mini | 26B / 5.4B | 128 (8) | SWA + full, sinks[^ink] | BF16[^ink-fmt] | 11.7 / 2.3[^mini-tps] | [20](https://artificialanalysis.ai/models/inkling-2-mini) |
| Inkling 2 Small | 594B / 17B | 384 (8) | SWA + full, sinks[^ink] | MXFP8[^mx] | —[^none] | [31](https://artificialanalysis.ai/models/inkling-2-small) |
| Qwen3.8-Flash-Next | 180B / 6B | 512+1 (10) | GDN + sparse[^qw] | FP8[^fp8] | —[^none] | [28](https://artificialanalysis.ai/models/qwen3-8-flash-next) |

## Pipeline and verification

| model | convert | decode | serve | PPL[^ppl] | vs. reference | smoke | open |
|---|---|---|---|---|---|---|---|
| GLM-5.2 | ✓ | ✓ | ✓ | ✓ | 32/32 ids | 12/12 | stale-read defect[^det] |
| DeepSeek-V4-flash | ✓ | ✓ | ✓[^v4-serve] | — | 30/32 ids[^tie] | 3/3 | NLL-by-position gate |
| Kimi-K3 | ✓ | ✓[^k3-dec] | ✗[^k3-serve] | — | KDA 2.3e-7[^k3-ref] | — | context alloc. vs. floor[^k3-ctx] |
| Muse Glimmer | ✓ | ✓ | ✓ | ✓ | goldens 3/3 | halves[^halves] | drafter[^dflash] |
| Inkling 2 Mini | ✓ | ✓ | ✓ | ✓ | 8/8 ids | 16/16 | — |
| Inkling 2 Small | ✓ | pending | — | — | logits 1.5e-5[^small-ref] | 6/6[^deviceless] | live GPU decode |
| Qwen3.8-Flash-Next | ✓ | ✗ | ✗ | — | anchors vendored | — | decode arm (S6–S8) |

✓ built and passing · ✗ refused at startup with the reason · — not built or not measured.
"ids" means greedy token ids identical to the first-party reference on the pinned prompt.
Every model runs text only: multi-token-prediction heads and vision towers are refused or
left out of the artifact. The repository's `docs/` hold the evidence for every row; see the
[docs index](https://github.com/Constructive-Programming/rivoli/blob/main/docs/00-orientation/INDEX.md).

[^aa]: Parameter counts (total / active per token) and Intelligence Index scores as
    published by [Artificial Analysis](https://artificialanalysis.ai/) on its v5.0.2 index,
    read on 2026-11-25. Scores from earlier index versions are on a different scale and are
    not mixed in.

[^na]: Deprecated on Artificial Analysis, with no score on the v5 index.

[^exp]: Routed experts per MoE layer (+ shared experts), with the number active per token
    in parentheses. From the model configs in the repository.

[^box]: Decode tokens per second, release build, measured under the GPU lock with an
    empty contention witness. Test machine: AMD Ryzen AI Max+ 395 (Radeon 8060S, ROCm),
    128 GB unified LPDDR5, artifacts on a local Samsung 9100 PRO 1 TB (PCIe 5.0) NVMe with
    btrfs. Numbers move with the memory budget (`--max-mem`) and the prompt; each footnote
    names its arm.

[^glm-tps]: 3.03 tok/s, mean of three arms, 512 tokens, `--max-mem 115`, 3-bit VQ,
    2026-11-20. About 2.1 GB of expert slots streamed per token.

[^v4-tps]: 9.17 tok/s, 512 tokens, `--max-mem 100`, 2026-08-16. Its prompt was not
    recorded, so the arm has not been reproduced.

[^none]: No witnessed decode speed yet.

[^gl-tps]: A bf16 figure from 2026-08-16 (2.56 tok/s) is superseded; the re-take is owed.

[^mini-tps]: 11.7 tok/s with all 24 layers resident (90 GiB budget); 2.3 tok/s at a 20 GiB
    budget, streaming 16 of 24 layers. 64 tokens, 2026-11-25. Both arms produced identical
    token ids.

[^mla]: Multi-head latent attention.

[^v4]: Multi-query attention with a shared K=V head; a 128-token sliding window plus
    compressed blocks chosen by an indexer.

[^k3]: Hybrid: 69 Kimi Delta Attention (linear) layers and 24 gated MLA layers.

[^gl]: Grouped-query attention (32 query, 2 KV heads); three sliding-window layers
    (2048 tokens) per full-attention layer.

[^ink]: Sliding-window and full-attention layers, with attention sinks.

[^qw]: Hybrid: 36 Gated DeltaNet (linear) layers and 12 indexed sparse-attention layers.

[^vq]: VQ3: 3-bit vector (codebook) quantization. INT4: 4-bit integer.

[^mx]: OCP microscaling formats: 4- or 8-bit floats sharing one scale per 32-element
    block. Kimi-K3's MXFP4 is repacked bit-exact.

[^fp8]: FP8 e4m3 with one scale per 128-element block.

[^ink-fmt]: Converted to bf16 from the checkpoint's NVFP4/MXFP8 weights.

[^ppl]: Teacher-forced perplexity scoring, the producer for paired per-token NLL
    comparisons.

[^det]: Under some timing, a GPU read of memory the NVMe has just written can return stale
    data, so long stock runs have not always reproduced their own output. `--copy-via-cpu`
    is a validated fix candidate; `--arena-refresh` is a lighter mitigation.

[^v4-serve]: Uses GLM's chat framing, a known gap.

[^tie]: Both differing positions are near-ties between the top two tokens.

[^k3-dec]: Correctness only so far: a first 16-token decode, coherent.

[^k3-serve]: No chat encoding yet.

[^k3-ref]: Linear-attention (KDA) layer outputs match the reference captures to 2.3e-7.

[^k3-ctx]: At `--ctx 4096` an allocation that grows with context is not yet charged to the
    memory floor, so the run can fail after the pin is placed.

[^k3-ii]: Low reasoning effort; the max-effort score is not listed.

[^gl-ii]: High reasoning effort.

[^halves]: Covered by the device-halves gate rather than an end-to-end smoke script.

[^dflash]: The DFlash speculative drafter has a converter and an oracle so far.

[^small-ref]: MXFP8 logits within 1.5e-5 (relative) of the bf16 reference.

[^deviceless]: Deviceless cells only.
