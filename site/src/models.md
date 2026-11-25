# Models

rivoli targets large mixture-of-experts models. Supporting a model means adding an
artifact converter, a per-architecture decode loop, and first-party reference goldens
that the loop is scored against. Which models belong to which code path is data in a
checked census table, not something encoded in code names.

| family | in the tree |
|---|---|
| GLM-5.2 (MoE) | converter, decode loop, serve; the primary parity reference |
| DeepSeek-V4-flash | converter, decode loop |
| Kimi-K3 | converter, decode loop (MXFP4 checkpoint) |
| Muse Glimmer | converter (bf16 or fp8), decode loop, DFlash drafter |
| Inkling 2 | Mini: converter, decode loop, serve. Small: converted and verified; live decode pending |
| Qwen3.8-Flash-Next | reference bytes and math vendored; refused at startup until the loop lands |

A port is "done" when its greedy output is token-identical to the first-party reference
implementation on the pinned prompt, and its quality gates pass. Status in this table
follows the repository's `docs/`, which hold the evidence for each row.
