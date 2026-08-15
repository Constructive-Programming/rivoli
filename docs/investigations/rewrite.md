---
status: live
scope: engine
verdict: The gates-first rewrite is through M0 and the substance of M1 — gates armed and red-proofed before any code, anchors vendored before the engine, arena/cache/hybrid/partition/fetch/waist ported with both feature arms verified; three M1 items (legality table, gate-taxonomy types, proptest) are deliberately deferred to their first consumers rather than built speculatively.
---

# The rewrite, milestone by milestone

The approved plan (2026-08-15) rebuilds rivoli on this orphan branch: quality gates before
code, anchors before the engine, the six-crate DAG as the layering, GLM-5.2 first, then
serve → Glimmer → V4 → K3. The old tree at `wt/glimmer-s2` @ `6b7f496e` stays live as the
parity reference. This doc is the running record; each milestone gets a dated section
stating what shipped, what its exit gate showed, and what was deliberately not done.

## M0 — gates (DONE 2026-08-15)

Commits `b56ff21`, `e922a34`, `35132d6`. Workspace + jscpd + CodeScene 10/10 + docs
registry + derived exemption ledger + CI; anchors (Glimmer ×6, K3 ×2), frozen V4 oracle,
`bin/ppl` + corpora. Exit gate: `docs/measurement/gate-red-proofs.md` — every gate shown
red, two of them fired unplanted during the port itself (the jscpd fnv1a catch, the
ledger's 0→3 refusal). OWED: the CodeScene score-half red proof, blocked on
`CS_ACCESS_TOKEN`.

## M1 — substrate (DONE 2026-08-15, three items deferred)

Commits `4b5b3da`, `3c0f577`, `74bdb20`, `285e989`, `f071c52`.

- **Pure core:** `arena`/`cache`/`hybrid` ported verbatim; `residency::partition()` is P6
  as a signature (one author, prefix-shaped, monotone in `free`, refuses below floor with
  the arithmetic). INV-1 registered.
- **Backend waist:** `hip.rs` + `gpustream.rs` + `Signal`/`block_on`/`NULL_STREAM` + all
  11 kernels; both arms verified on the box (featureless = `abi` alone; `rocm` = 11
  kernels through hipcc, clippy clean). The three `repr(C)` mirrors moved to
  `backend::abi` with the layout assert. INV-4/INV-6 arrived with the port and the
  registry gate refused the tree until §8b documented them.
- **Engine shell begins:** `fetch/` (io_uring ticketed dataflow) + `telemetry.rs` ported;
  all five instrument features declared and forwarded; `feature-matrix.sh` (quick 9/9 on
  the box) + `matrix.rs` list-drift gate, red-proofed.

Exit gate: registry non-empty, both-direction check green, INV red-proofed
(`gate-red-proofs.md` §2b). All verification with explicit exit codes after two
false-green incidents (cwd reset to the main repo; `| tail` eating a red) — both recorded
in the session memory.

**Deferred from M1, deliberately — each to its first consumer, not dropped:**
1. **Legality table** (`decide(arch, flag)`) → M4, with the CLI whose flags it judges. A
   table over flags that do not exist yet would be invented rows.
2. **Gate-taxonomy / tolerance-with-provenance types** → M2/M3, with the GLM anchor and
   the first kernel oracle that load them. The shapes are sketched in the plan; building
   them unconsumed invites a redesign the first consumer would force anyway.
3. **proptest** → with the first property that earns a generator (partition monotonicity
   is currently swept deterministically over a range; proptest joins when the arena
   relocation properties port in M2+).

## M2 — GLM artifact + config + converter + the GLM anchor (IN PROGRESS 2026-08-15)

Landed so far: `core::num` (conversions + `Scoring`), `artifact::{quant,format,schema,
arch,glm_config}` (commit `46f2153` — sniffing is identity-only, presentation policy
deliberately not ported), the three converters + `engine::device` (commit `73ecfa1`).
Still owed: the GLM anchor, artifact tests, tokenizer (deferred — coupled to
`dsv4_encoding`, arrives with the CLI at M4).

**Anchor scouting, settled 2026-08-15:**
- The fp8 source is `/swarm/storage/ai/openclaw/glm52-fp8` — `model_type: glm_moe_dsa`,
  **no `auto_map`**, so the first-party stack is transformers-native (no remote code).
- Fresh venv at `/home/rhansen/glm-anchor/venv`: torch 2.13.0+cpu, transformers 5.15.0,
  `glm_moe_dsa` in `CONFIG_MAPPING_NAMES`. CPU-only, so the anchor needs no GPU and no
  lock, same as Glimmer's.
- NOTE: the old `/home/rhansen/glimmer-anchor/venv` (pinned in the old tree's anchor.md)
  is GONE from the box — a Glimmer anchor regeneration would need its venv rebuilt at the
  pinned transformers commit fe747d88 first.
- Mechanisms the taps must capture, from reading `modeling_glm_moe_dsa.py`: MLA with
  q-LoRA (`q_a_proj→q_a_layernorm→q_b_proj`) and kv-LoRA (`kv_a_proj_with_mqa` split
  kv_lora_rank + qk_rope, `kv_a_layernorm` on the kv half only), **interleaved** RoPE
  (both attention and indexer — V3.2 uses half-split, this does not), `expand_kv`
  latent expansion, DSA indexer (own wq_b/wk/LayerNorm(eps 1e-6)/weights_proj; ReLU
  scores; head-weighted sum; `indexer_types` full/shared with `prev_topk_indices`
  cross-layer sharing), sigmoid router + `e_score_correction_bias` + group top-2 +
  norm_topk + `routed_scaling_factor` (router `weight` is ZERO-init — a tiny model must
  draw it, or every expert ties), MoE + shared expert, dense first `first_k_dense_replace`
  layers. Config `__post_init__` computes `indexer_types` from freq/offset and forces
  `head_dim = qk_rope_head_dim`.
- Tiny-config non-degeneracy plan (every width distinct, lesson 30): vocab 61, hidden 48,
  inter 96, moe_inter 24, layers 6 (2 dense + 4 sparse), heads 4, kv_heads 4, routed 10
  top-3, shared 1, kv_lora 20, q_lora 28, qk_rope 8, qk_nope 14 (sum 22 ≠ kv_lora),
  v_head 10, index_topk 4 (< PROMPT_LEN so the sparse path is exercised — the old dsa
  fast-path-below-topk lesson), index_head_dim 16, index_n_heads 2.

Exit unchanged: anchor integrity green, defect matrix (≥10 defects × 2 salts) fully
red-capable, converter round-trip byte-stable.
