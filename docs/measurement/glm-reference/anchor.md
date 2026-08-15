---
status: data
scope: glm
verdict: The GLM-5.2 anchor exists and runs — transformers' own glm_moe_dsa (5.15.0, torch 2.13.0+cpu, python 3.14.6; the fp8 checkpoint declares this model_type with NO auto_map, so the in-tree class IS the shipped implementation) executed at tiny non-degenerate widths with the real structure. TWO goldens are vendored (crates/oracles/tests/glm-anchor-{1,2}.bin, 265,019 B each, byte-for-byte reproducible, salts verified distinct) and read by glm_anchor.rs with no python, no venv, no network, no device. THIRTEEN defects x 2 salts = 26 compare cells, all green, each gated on its declared-green set scoped to t0. The matrix rejected its own author three times while being built: a skipped call orphans its hook (shared_expert_off zeroes the weight instead), a structural defect may only ADD captures (first_k_dense_off REPLACED the dense layers' captures and was refused by the set-equality contract — indexer_share_off, sharing disabled, took the slot), and a norm downstream of the DSA selection cannot sit in an indexer defect's green set. GLM-specific trap recorded: the router's e_score_correction_bias is an nn.Buffer, invisible to named_parameters, and left at zero it collapses router_bias_off into a no-op — drawn explicitly. No tolerances yet: S-kernel work measures the fp32 floors from BOTH draws before any threshold is chosen, per the K3/Glimmer discipline.
---

# GLM-5.2 anchor

Generated 2026-08-15 on `rw/main`; the driver is
`crates/oracles/tests/glm_anchor_driver.py`, the deviceless gate is
`crates/oracles/tests/glm_anchor.rs` (8 tests), regeneration is
`crates/oracles/tests/glm-anchor.sh` (no GPU, no lock — CPU torch).

## Environment (pinned in the goldens' own metadata, preflight reads it from there)

| | |
|---|---|
| python | 3.14.6 |
| torch | 2.13.0+cpu |
| transformers | 5.15.0 (`glm_moe_dsa` native — the checkpoint has no `auto_map`) |
| venv | `/home/rhansen/glm-anchor/venv` (recreated; nothing in it is load-bearing beyond the pins) |
| source checkpoint | `/swarm/storage/ai/openclaw/glm52-fp8` (`model_type: glm_moe_dsa`) |

## Tiny config (REAL values in the driver's trailing comments)

vocab 61 · hidden 48 · inter 96 · moe_inter 24 · layers 6 (2 dense + 4 sparse) · heads 4 ·
routed 10 top-3 + 1 shared · scaling 2.5 · kv_lora 20 · q_lora 28 · qk_rope 8 · qk_nope 14
(qk_head 22 ≠ kv_lora 20, deliberately) · v_head 10 · index_topk 4 < prompt 12 (the DSA
selection is REAL — below index_topk the whole run rides the dense fast path) ·
index_head_dim 16 · index_n_heads 2 · indexer pattern **FSFSFS** (both mechanisms run:
full indexers at 0/2/4, cross-layer sharing at 1/3/5). Every width distinct from every
other; `glm_anchor.rs` asserts the non-degeneracy rather than trusting it.

## What is captured (census derived in the gate, never constants)

Per step (7 = prefill + 6 decode), per layer: `attn.out`, `q_resid` (q_a_layernorm out),
`kv_latent` (kv_a_layernorm out), both pre/post norms, `attend.{q,out,mask_last_row}`
(the mask row pins the selection as arithmetic, not just indices), the attention rope
pair, `topk_indices` (sorted — the SET is the contract, torch.topk order is not); on full
layers the indexer's own rope pair; on sparse layers `router.{logits,weights,topk_last}`
and `{moe,shared,experts}.out`; on dense layers `mlp.out`; plus per-step logits, prompt,
emitted ids, and the two structure vectors. 623 float + 74 int tensors per golden.

## The defect matrix (13 × 2 salts, all green)

router_softmax · router_bias_off · router_norm_topk_off · router_scaling_off ·
shared_expert_off (weight zeroed, not call skipped) · rope_half_split (the documented
V3.2 difference) · kv_norm_spans_rope · q_a_norm_off · expand_kv_swapped_split ·
attn_scale_unscaled · indexer_relu_off · indexer_select_all · indexer_share_off
(structural, extra_ok: sharing disabled, shared layers grow their own indexers).

Green sets are scoped to **t0**: a defect that shifts the argmax contaminates every later
step through the token it feeds back, so localisation is only possible on the prefill.
Router-family defects hold the dense prefix (29 green captures); attention-family defects
hold prompt + structure only; indexer defects additionally hold everything upstream of
the selection on L0 (`q_resid`, `kv_latent`, `norm.in`, `attend.q`).

## What this does NOT establish

No per-operator tolerances (S-kernel work measures fp32 floors from both draws first — a
tolerance picked to make a kernel pass is not a tolerance), and no real-weight evidence:
these goldens pin the ARITHMETIC of the reference at toy widths, not the checkpoint.
