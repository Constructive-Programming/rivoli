# MTP speculative decode — build plan

Multi-Token Prediction (MTP) turns the ~60% idle GPU (decode is NVMe-bound; see
`AGENTS.md` § pipeline cost) into throughput by **drafting** the next-next token
and **verifying** a short run of drafts in one batched forward — one expert
fetch then serves multiple tokens, amortizing the NVMe wall. Built by the
`AGENTS.md` mechanism: scalar oracle → kernel → wire → validate, milestone-gated.

## Architecture (confirmed from the checkpoint + DeepSeek-V3 formulation)

GLM-5.2 ships **one** MTP layer at index `num_hidden_layers` = 78
(`num_nextn_predict_layers=1`). Its 791 tensors are a full transformer layer
(MLA attention + 256 routed experts + shared expert + DSA indexer) plus the MTP
glue:

- `model.layers.78.enorm.weight` `[6144]` — RMSNorm on the next token's
  **embedding** (e = embedding).
- `model.layers.78.hnorm.weight` `[6144]` — RMSNorm on the previous **hidden**
  `h` (h = hidden).
- `model.layers.78.eh_proj.weight` `[6144, 12288]` — Linear `2·hidden → hidden`,
  kept **bf16** (high dynamic range: mean|w|≈0.012, max≈1.5 → per-row int4 zeros
  ~all of it, 99% rel-err; validated during M0).
- `model.layers.78.shared_head.norm.weight` `[6144]` — pre-head RMSNorm.
- `model.layers.78.shared_head.head.weight` — **absent** ⇒ output head is **tied
  to the main `lm_head`** (reuse it).
- plus `input_layernorm`, `post_attention_layernorm`, the full `self_attn`
  stack, `mlp` (256 experts + gate + shared), and the `indexer`
  (`index_share_for_mtp_iteration=true` ⇒ the MTP attention reuses the main
  model's DSA top-k selection rather than running its own).

**Forward (one MTP layer, DeepSeek-V3 §MTP), drafting token t+2:**
given the main model's final hidden `h` (pre-lm_head, at the position that just
produced token t+1) and the embedding of the just-emitted token t+1,
```
h'  = eh_proj( concat[ enorm(embed(t+1)) , hnorm(h) ] )   # [hidden]; e=embed, h=hidden
h'' = mtp_layer_forward(h')     # input_layernorm→attn(reuse main top-k)→+res→post_norm→moe→+res
logits_{t+2} = lm_head( rmsnorm(h'', shared_head.norm) )
draft t+2 = argmax(logits_{t+2})
```
The attention runs at the current position with the existing KV cache; with
`index_share_for_mtp_iteration` it reuses the main model's selected rows (no
separate indexer scoring needed for the draft).

## Weights: extract + convert (M0 gating dependency)

The colibri converter skipped layer 78, so it's **not in the int4 snapshot**.
Extract it from `zai-org/GLM-5.2` by the range-request method (`AGENTS.md` §
weights): the layer-78 tensors live in HF shards ~270–282. Requantize to the
engine's formats, matching colibri's math exactly (`convert_fp8_to_int4.py`):
- fp8 e4m3 with `*.weight_scale_inv` → 128×128-block dequant to f32.
- int4 per-row: `s = max(|w|.max(axis=1)/7, 1e-8)`, `q = clip(rint(w/s), -8, 7)`,
  pack low nibble = col 2j, high nibble = col 2j+1, each `+8`; `.qs` = f32 row
  scale. (Experts, shared, attn projections.)
- eh_proj kept bf16 (see above); norms/gate/bias f32; indexer skipped (in out-idx).
- Norms/`enorm`/`hnorm`/`shared_head.norm` → f32 (widened from bf16).
- Indexer projections → bf16 (as the main indexer shard).
Write `out-mtp-*.safetensors` next to the snapshot (indexed by `snapshot.rs`).
Size ≈ one MoE layer ≈ 4.8 GB int4. **Gate M0: layer-78 tensors present +
indexed; shapes validated at load.**

## Milestones (each gated on a measured number)

- **M0 — weights.** Extract + convert layer 78 → `out-mtp`. *Gate: indexed,
  shapes checked.*
- **M1 — scalar draft oracle.** `mtp.rs`: the glue (enorm/hnorm/eh_proj) +
  reuse `attention()`/`moe_block()` at layer 78 + `shared_head.norm`+`lm_head`.
  *Gate: on the reference path, MTP drafts the token the main model actually
  emits next at a sane rate (accept rate > ~0.5 on a fixed prompt) — coherence,
  not speed.*
- **M2 — kernels.** eh_proj is a bf16 GEMV (reuse `gemv_bf16`); enorm/hnorm reuse
  `layernorm`/`rmsnorm`; the MTP layer reuses the existing attn+MoE kernels. Only
  new device piece: the concat+glue plumbing. *Gate: device draft logits match
  the scalar oracle within tolerance.*
- **M3 — speculative decode loop.** Draft with MTP → verify k drafts in one
  **batched (S=k) forward** through the main 78 layers (the union of experts is
  fetched once, reused across the k positions) → accept the longest correct
  prefix, roll back KV/indexer for rejected tokens. Needs a small-S path in the
  MLA + fused-MoE kernels (today S=1). *Gate: end-to-end tok/s > the dense
  baseline at equal quality (greedy-equivalent accepted tokens), on 512 tokens.*

  **Status (2026-07-19): mechanism DONE + greedy-equivalence VERIFIED; perf gate
  NOT met.** `GpuEngine::{forward_batch,generate_spec}` (S=2), batched fused-MoE
  kernel (`rivoli_moe_experts_batched`), MTP-KV lockstep, accept/reject rollback,
  `--spec` CLI. Validation `tests/mtp_spec.rs`: spec output byte-identical to
  greedy (24/24). **Measured (64 tok, "The sky is blue because"): spec 0.53 tok/s
  vs baseline 0.71 tok/s — SLOWER.** 44 verify rounds, 19 accepted (43.2%); expert
  hit 69.6% vs baseline 75.4%. Root cause is measured, not a bug: `forward_batch`
  runs **without cross-layer prefetch**, and baseline's prefetch is what hides the
  fetch (drives its 75% hit + ~1.4s/pass). Each S=2 round costs ~2.74s (~1.9× a
  baseline pass) with the union fetch back on the critical path, so even at 60–80%
  accept the round-count drop (64→40→36) can't beat ~1.9× per round.
  → **M4 (perf): restore prefetch in the batched path** — predict the next layer's
  union from both positions' post-attn residuals, submit the reads on the second
  ring, drain in `resolve_layer(l+1)` (mirror `forward`'s prefetch). Without it the
  batched-fetch amortization is dominated by the prefetch it gave up.

## Why this is the right lever (recap)

Decode is NVMe-read-bound and `attn`/`mlp` can't overlap (sequential residual
chain), so the GPU idles ~60% waiting on expert fetch. Batched verify amortizes
one fetch across k tokens **and** fills the idle GPU — hitting both levers the
concurrency analysis identified. Risk: acceptance rate (a 1-token MTP head is
typically ~0.6–0.85 on this family) and the batched-S kernel work. Colibri saw
MTP lose in a *compute-bound* regime; here the GPU is fetch-starved, so re-test,
don't assume.

## GPU note

M1 is CPU-only (scalar) and can be built + validated while a GPU job runs. M2/M3
need the sole-tenant GPU — queue them for when the current 10k benchmark frees
it (`AGENTS.md` § sole-tenant).
