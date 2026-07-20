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

  **M4 DONE (2026-07-19): prefetch restored, spec STILL loses — investigation
  closed with a definitive negative result.** `forward_batch` now predicts + submits
  L+1's union (per position, deduped, capped at `prefetch_depth`) exactly like
  `forward`; regression test greedy-identical *with prefetch active*. Measured: spec
  **0.55** tok/s vs baseline **0.70** (was 0.53 pre-M4; hit 69.6→73.8%, drain-wait
  down to 16ms/tok — fetch now hidden). **Root cause (definitive): decode is NVMe-
  BANDWIDTH-bound and prefetch already hides fetch LATENCY in both paths, so
  batching's fetch-*amortization* lever is moot — and batching *raises* bytes/token:
  spec reads 184 misses/tok vs baseline 159 (+16%), because a 2-position union has
  more distinct experts than one top-8 and 57% of rounds are rejects that fetched a
  wasted draft's experts.** At 43% accept the round-count drop (64→44) can't beat
  +16% bytes on the bandwidth floor. Spec would only win with much higher accept
  (>~70%) AND high union overlap — not reachable with a 1-token MTP head here.
  `--spec` stays in as a verified-correct, opt-in mechanism; not the default.

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

## INVESTIGATION CLOSED — 2026-07-20 (256-token sweep, all salvage branches dead)

Three salvage ideas were built off `mtp` and benchmarked at **256 tokens** (the
prior M3/M4 numbers were 64-token only, which hid the regime flip below). All
lose; the two branches with new device paths also have 256-token correctness
bugs. Snapshot for `--spec`: `~/glm52-snap` (int4 experts + bf16 eh_proj) — NOT
`/var/db/.../glm52-colibri-int4`, whose `out-mtp-*` are a stale all-int8
extraction that won't load.

| Config | tok/s @256 | vs base | Notes |
|--------|-----------|---------|-------|
| baseline (no spec) | **1.05** | — | warm, 86.8% hit, fetch 51% |
| overlap-gate (`--spec`) | NaN@256 | — | ran @64 (0.69, gate shut spec 50/64); KV-lockstep bug in the gate's forward/forward_batch toggle at longer context |
| warm-budget (`--spec`) | **0.95** | −10% | budget isolation ✓ (pool stayed 4634), warm gate engaged @tok86 ✓, **84% accept** ✓ — still loses |
| union-tree width-1 | 0.68 | −35% | raw always-on chain, 56% accept |
| union-tree width-2 | NaN@256 | — | S=3 tree attention geometry bug (decoupled RoPE-pos + sibling gather); never validated on device |

**Definitive finding — the bottleneck is regime-dependent, and speculation loses
in BOTH regimes for opposite reasons:**
- **Cold** (short ctx, ~63% fetch): NVMe-**bandwidth**-bound. Batched verify's
  2-position union reads **+16% bytes/tok**; at 43% accept the round-count drop
  can't beat it. (M3/M4.)
- **Warm** (long ctx, cache ~87%, fetch drops to 24–51%): **compute**-bound. The
  MTP draft head is a *full extra MoE layer (layer 78, 256 experts) every round*;
  `warm-budget` hit 84% accept yet `other` ballooned to **638 ms/tok (63%)**, so
  the draft+batched-verify compute exceeds the tokens saved.

`warm-budget`'s STRUCTURE is correct (warm-only gate + isolated MTP budget are
keepers, and it reached 84% accept). The only thing beating it is the draft's
compute cost. The single un-tested path that could flip it positive: a **cheap
approximate draft** (top-k experts only, or a distilled/low-rank layer-78)
instead of the full 256-expert forward — if the draft drops below the ~200 ms/tok
that 84% accept saves. Not pursued.

**Verdict: MTP speculation as structured (full layer-78 draft) cannot win on this
engine in either regime.** All four branches marked `deadend/*`. The lever that
actually crosses 1 tok/s is the warm expert cache (commit 16bae7f), not
speculation — speculation only steals from it.
