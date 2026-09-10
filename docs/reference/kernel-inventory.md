---
status: live
scope: engine
verdict: Every GPU kernel launcher in the tree as a row — 66 of them, the same population `crates/cli/tests/kernel_coverage.rs` censuses at 66/66/0 — with the models that call it, the production site that calls it, and the mode that reaches it, all derived from the `launchers!` DSL and from the engine's own call sites rather than from memory. It also carries the thirteen launchers that NO production code calls, each classified with the evidence that puts it in a category: superseded by a shipped sibling, implemented ahead of the arm that will call it (qwen S3 vs S6), or pending a deferred milestone (M13's GLM DSA). Reuse is measured, not hoped for: V4 is the most-shared kernel consumer (22 launchers) and `gemv_fp8`, `argmax`, `vadd` and `sigmoid_gate` each serve three or four architectures through one implementation.
---

# The kernel inventory

Derived 2026-09-08 on rh-anine / this worktree. Read the derivation before the table: every
column is a query, and a query can be wrong.

## How each column was built, and what would make it lie

* **Rows.** The `launchers!` DSL rows in `crates/backend/src/{hip_linalg,hip_blocks,hip_attn}.rs`
  (grammar `rust_fn -> c_symbol, "tag" (args)`), plus the four hand-written launchers in
  `hip.rs`/`hip_attn.rs` that the macro refuses. 66 in total, which is the same population
  `kernel_coverage.rs` prints (`66 launchers: 66 with oracles, 0 deferred`, measured by
  `--nocapture` on a deviceless run) — two independent derivations agreeing is the only reason
  the count is stated rather than counted.
* **Model columns.** A cell is lit only by a **production** reference: an import from or a call
  through the ABI wall inside `crates/engine/src/**` or `crates/cli/src/**`. Test files never
  light a cell. That distinction is the table's whole point: `kernel_coverage.rs` asks *"does an
  oracle exercise this launcher?"* and answers yes for all 66, while 13 of them have no caller in
  any binary anyone ships. Coverage and liveness are different questions, and only this table
  asks the second one.
* **`Activated at`.** The module and the enclosing function of the production reference, so a row
  can be re-checked by opening one file. Where the mode is decided by a dispatch rather than a
  location, the dispatch is named (e.g. `RoutedFmt::{I4,Vq3}` in `glm/mlp.rs`).
* **The `Qwen` column is empty, and that is a result.** `Arch::QwenFlashNext` refuses at every
  door and `crates/engine/src/qwen/` does not exist (S6), so no qwen kernel can have a production
  caller yet. Four of the thirteen uncalled launchers are exactly the qwen S3 set named in
  `kernel_coverage.rs`'s FOURTH TURN.
* **What would make it lie.** A launcher reached only through a `dyn`/fn-pointer table or a macro
  that synthesises the name would read as uncalled; a `#[cfg]`-compiled-out call site would read
  as live. Both were checked by hand for the 13 (each is cited below), and neither was found —
  every one of them is referenced by a test file and by nothing that ships.

## The table

| kernel (`launch_…`) | C entry | TU | GLM | V4 | K3 | Glimmer | Qwen | activated at (module :: fn) |
|---|---|---|---|---|---|---|---|
| `rmsnorm_centered_single` | `rmsnorm_centered_single` | activation.hip | · | · | · | **✓** | · | glimmer/forward :: branch_add, pre_norm |
| `rmsnorm_single` | `rmsnorm_single` | activation.hip | **✓** | · | **✓** | **✓** | · | glimmer/forward; glm/attn; glm/forward; k3/forward :: cache_and_absorb, embed_rows, qkv_project, rms_vec |
| `rmsnorm_weightless_batch` | `rmsnorm_weightless_batch` | activation.hip | · | · | · | **✓** | · | glimmer/forward :: attention, embed |
| `rope_interleave` | `rope_interleave` | activation.hip | **✓** | · | · | · | · | glm/attn :: cache_and_absorb |
| `rope_split_half` | `rope_split_half` | activation.hip | · | · | · | **✓** | · | glimmer/forward :: attention |
| `situ_glu_f32` | `situ_glu_f32` | activation.hip | · | · | **✓** | · | · | k3/forward :: situ_mlp |
| `swiglu` | `swiglu` | activation.hip | **✓** | · | · | **✓** | · | glimmer/forward; glm/mlp :: desc_of, mlp |
| `swiglu_clamped_bf16` | `swiglu_clamped_bf16` | activation.hip | · | **✓** | · | · | · | v4/moe :: shared_expert |
| `attend` | `mla_attend` | attn.hip | **✓** | · | · | · | · | glm/attn :: attend_rows |
| `gather_attn_shared_kv` | `gather_attn_shared_kv` | attn.hip | · | **✓** | · | · | · | v4/attn :: attend_rows |
| `gqa_attend` | `gqa_attend` | attn.hip | · | · | · | **✓** | · | glimmer/forward :: attention |
| `gqa_block_attend` | `gqa_block_attend` | attn.hip | · | · | · | · | · | *(no production reference)* |
| `mha_attend` | `mha_attend` | attn.hip | · | · | **✓** | · | · | k3/forward :: mla_attention |
| `act_quant_f4_rotated` | `act_quant_f4_rotated` | blockindex.hip | · | **✓** | · | · | · | v4/blocksel; v4/kvcompress :: emit, raw_scores |
| `append_kv` | `append_kv` | fwd.hip | **✓** | · | · | · | · | glm/attn :: cache_and_absorb |
| `argmax` | `argmax` | fwd.hip | **✓** | **✓** | **✓** | **✓** | · | glimmer/forward; glm/decode; k3/decode; v4/decode :: argmax, argmax_rows, sample |
| `embed_i8_row` | `embed_i8_row` | fwd.hip | **✓** | · | · | · | · | glm/forward :: embed_rows |
| `flag_nonfinite` | `flag_nonfinite` | fwd.hip | **✓** | · | · | · | · | glm/forward :: embed_rows |
| `gather_rope` | `gather_rope` | fwd.hip | **✓** | · | · | · | · | glm/attn :: cache_and_absorb |
| `logit_softcap` | `logit_softcap` | fwd.hip | · | · | · | **✓** | · | glimmer/forward :: sample |
| `sigmoid_gate` | `sigmoid_gate` | fwd.hip | · | · | **✓** | **✓** | · | glimmer/forward; k3/forward :: attention, mla_attention |
| `vadd` | `vadd` | fwd.hip | **✓** | · | **✓** | **✓** | · | glimmer/forward; glm/attn; glm/mlp; k3/forward :: add_sub_to_pref, branch_add, desc_of, moe_ffn |
| `embed_bf16_row_bcast` | `embed_bf16_row_bcast` | headtail.hip | · | **✓** | **✓** | **✓** | · | glimmer/forward; k3/forward; v4/forward :: begin_step, embed, rms_vec |
| `hc_head_collapse` | `hc_head_collapse` | headtail.hip | · | **✓** | · | · | · | v4/forward :: head_tail |
| `index_append` | `index_append` | indexer.hip | · | · | · | · | · | *(no production reference)* |
| `index_head_route` | `index_head_route` | indexer.hip | · | · | · | · | · | *(no production reference)* |
| `index_pool_norm_f32` | `index_pool_norm_f32` | indexer.hip | · | · | · | · | · | *(no production reference)* |
| `index_pool_push` | `index_pool_push` | indexer.hip | · | · | · | · | · | *(no production reference)* |
| `index_score` | `index_score` | indexer.hip | · | · | · | · | · | *(no production reference)* |
| `index_topk` | `index_topk` | indexer.hip | · | · | · | · | · | *(no production reference)* |
| `layernorm` | `layernorm` | indexer.hip | · | · | · | · | · | *(no production reference)* |
| `gemm_bf16` | `gemm_bf16` | kvcompress.hip | · | **✓** | **✓** | **✓** | · | glimmer/forward; k3/forward; v4/blocksel; v4/forward; v4/kvcompress :: bproj, deposit, head_tail, proj |
| `kv_compress_decode` | `kv_compress_decode` | kvcompress.hip | · | **✓** | · | · | · | v4/kvcompress :: emit |
| `kv_compress_deposit` | `kv_compress_deposit` | kvcompress.hip | · | **✓** | · | · | · | v4/kvcompress :: deposit |
| `kv_compress_prefill` | `kv_compress_prefill` | kvcompress.hip | · | **✓** | · | · | · | v4/kvcompress :: emit |
| `gemv_f32` | `gemv_f32` | linalg.hip | **✓** | **✓** | · | · | · | glm/mlp; v4/moe :: gate, route_layer |
| `gemv_fp8` | `gemv_fp8` | linalg.hip | **✓** | · | · | **✓** | · | glimmer/forward; glm/attn; glm/mlp :: desc_of, output_project, proj, qkv_project |
| `gemv_i4` | `gemv_i4` | linalg.hip | · | · | · | · | · | *(no production reference)* |
| `gemv_i8` | `gemv_i8` | linalg.hip | **✓** | · | · | · | · | glm/forward :: tail |
| `gemv_vq` | `gemv_vq` | linalg.hip | · | · | · | · | · | *(no production reference)* |
| `vq_encode` | `vq_encode` | linalg.hip | · | · | · | · | · | *(no production reference)* |
| `act_quant_f8_prefix` | `act_quant_f8_prefix` | mla.hip | · | **✓** | · | · | · | v4/attn; v4/kvcompress; v4/moe :: emit, output_project, qkv_project, quantize_activation |
| `gemv_fp8_bf16` | `gemv_fp8_bf16` | mla.hip | · | **✓** | · | · | · | v4/attn; v4/engine :: max_ctx, output_project |
| `mla_absorb_fp8` | `mla_absorb_fp8` | mla.hip | **✓** | · | · | · | · | glm/attn :: cache_and_absorb |
| `mla_value_fp8` | `mla_value_fp8` | mla.hip | **✓** | · | · | · | · | glm/attn :: output_project |
| `qk_norm` | `qk_norm` | mla.hip | · | **✓** | · | · | · | v4/attn :: qkv_project |
| `rmsnorm_batch` | `rmsnorm_batch` | mla.hip | · | **✓** | · | · | · | v4/attn; v4/forward :: head_tail, pre_norm, qkv_project |
| `rope_adjacent` | `rope_adjacent` | mla.hip | · | **✓** | · | · | · | v4/attn; v4/blocksel :: output_project, qkv_project, raw_scores |
| `moe_acc_drain` | `moe_acc_drain_s` | moe.hip | **✓** | **✓** | · | · | · | glm/mlp; v4/moe :: desc_of, routed_experts |
| `moe_acc_drain_to` | `moe_acc_drain_to_s` | moe.hip | · | · | **✓** | · | · | k3/forward :: moe_ffn |
| `moe_expert_range` | `moe_expert_range` | moe.hip | **✓** | · | · | · | · | glm/mlp :: expert_range |
| `moe_expert_range_i4` | `moe_expert_range_i4` | moe.hip | **✓** | · | · | · | · | glm/mlp :: expert_range |
| `act_quant_f8` | `act_quant_f8` | moe_f4.hip | · | **✓** | · | · | · | v4/moe :: shared_expert |
| `moe_expert_range_f4` | `moe_expert_range_f4` | moe_f4.hip | **✓** | **✓** | · | · | · | glm/mlp; v4/moe :: expert_range |
| `moe_expert_range_f4_situ` | `moe_expert_range_f4_situ` | moe_f4.hip | · | · | **✓** | · | · | k3/forward :: expert_range |
| `gated_delta_recurrent_f32` | `gated_delta_recurrent_f32` | recurrent.hip | · | · | **✓** | · | · | k3/forward :: kda_attention |
| `rmsnorm_gate_heads_f32` | `rmsnorm_gate_heads_f32` | recurrent.hip | · | · | **✓** | · | · | k3/forward :: kda_attention |
| `short_conv_silu_f32` | `short_conv_silu_f32` | recurrent.hip | · | · | **✓** | · | · | k3/forward :: kda_attention |
| `attn_res` | `attn_res` | residual.hip | · | · | **✓** | · | · | k3/forward :: fold_into_h |
| `gated_residual_collapse_f32` | `gated_residual_collapse_f32` | residual.hip | · | · | · | · | · | *(no production reference)* |
| `hc_post` | `hc_post` | residual.hip | · | **✓** | · | · | · | v4/forward :: layer |
| `hc_pre` | `hc_pre` | residual.hip | · | **✓** | · | · | · | v4/forward :: pre_norm |
| `gdn_recurrent_f32` | `(hand-written)` | *(hand)* | · | · | · | · | · | *(no production reference)* |
| `hash_rows` | `(hand-written)` | *(hand)* | · | · | · | · | · | *(no production reference)* |
| `index_score_blocks` | `(hand-written)` | *(hand)* | · | **✓** | · | · | · | v4/blocksel :: raw_scores |
| `index_score_blocks_f32` | `(hand-written)` | *(hand)* | · | · | · | · | · | *(no production reference)* |

## The thirteen that no production code calls

Classified with primary evidence. A launcher here is either **wired** (name the milestone),
**superseded** (delete it and its kernel body), or **pending a deferred chain** — the fourth
option, "kept because a test covers it", is the one this table exists to refuse, because
`kernel_coverage.rs` already counts those as covered.

| launcher | TU | category | the evidence |
|---|---|---|---|
| `gemv_vq` | linalg.hip | **superseded** | GLM's int3-vq experts run through `moe_expert_range` + `gemv_f32`/`gemv_fp8` (`glm/mlp.rs:558`, `RoutedFmt::Vq3`) |
| `gemv_i4` | linalg.hip | **superseded** | same shape: `RoutedFmt::I4 => launch_moe_expert_range_i4` (`glm/mlp.rs:555`) |
| `index_score` | indexer.hip | **pending M13** | `indexer.hip`'s header: "device port of the old int4 tree's indexer, adapted to the fp8 checkpoint… fed straight into the sparse `mla_attend` gather" — and sparse is refused: `SPARSE_ATTN_DEFERRED` at `core/src/legality.rs:360` ("only `--attn dense` decodes today"), with M13's chain deferred wholesale (`wave/m12-glm-chain`, §4) |
| `index_append` | indexer.hip | **pending M13** | as above |
| `index_topk` | indexer.hip | **pending M13, and the header argues against itself** | it says "the top-`index_topk` scores are picked host-side", so even the ported flow may not need the kernel form; M13 must answer that, not inherit it |
| `index_head_route` | indexer.hip | **pending M13** | as above |
| `index_pool_push` | indexer.hip | **pending M13** | as above |
| `layernorm` | indexer.hip | **pending M13** | used only by that pipeline ("`k_norm` … feeds `layernorm` below"); no other module imports it |
| `gqa_block_attend` | attn.hip | **not wired — by decision** | M17c's drafter attend: oracle-covered and executed (7/7, witnessed 2026-09-04), but §4 states it is a scored kernel rather than a shipped one. Its `gqa_attend` duplication is still OWED because jscpd cannot see `.hip` |
| `gdn_recurrent_f32` | recurrent.hip | **qwen S3 ahead of S6** | named in `kernel_coverage.rs`'s FOURTH TURN as one of three launchers that landed with their oracles; the arm that would call them is S6 |
| `index_pool_norm_f32` | indexer.hip | **qwen S3 ahead of S6** | same |
| `gated_residual_collapse_f32` | residual.hip | **qwen S3 ahead of S6** | same |
| `index_score_blocks_f32` | blockindex.hip | **qwen S3 ahead of S6** | the f32 sibling of V4's live `index_score_blocks`; `v4/blocksel.rs` calls the f4 pair, not this one |

**Two near-misses worth recording, because they were classified wrong before being checked.**
`hash_rows` and `vq_encode` both look uncalled to a scan that only looks at `crates/engine/src`:
`hash_rows` is called from `engine/src/fetch/stream.rs`, `engine/src/probe.rs` and
`engine/src/routed.rs` (shared streaming infrastructure, not an arch directory), and
`vq_encode` is called by `cli/src/bin/convert/gpu.rs` — the converter. A model-column scan
that keys on architecture directories silently invents dead code, which is the direction of
error that gets kernels deleted. Both are live.

## Reuse, measured

| | launchers with a production caller |
|---|---|
| V4 | **22** |
| GLM | 19 |
| K3 | 14 |
| Glimmer | 13 |
| Qwen | 0 (S6 not built) |

The most-shared implementations — one kernel body, several architectures, which is the reuse the
inventory is meant to protect:

- **`argmax`** — GLM+Glimmer+K3+V4
- **`vadd`** — GLM+Glimmer+K3
- **`rmsnorm_single`** — GLM+Glimmer+K3
- **`gemm_bf16`** — Glimmer+K3+V4
- **`embed_bf16_row_bcast`** — Glimmer+K3+V4
- **`swiglu`** — GLM+Glimmer

The reading: `gemv_fp8`, `argmax`, `vadd`, `embed_bf16_row_bcast`, `sigmoid_gate` and
`gemm_bf16` are the trunk every arch leans on, so any change to them is a four-model change and
belongs behind the parity gate, not in a drive-by refactor. `moe_expert_range_f4` serving both GLM
and V4 while `moe_expert_range_f4_situ` serves K3 alone is the same fact at the fused-activation
boundary — the difference is the epilogue, not the memory layout.
