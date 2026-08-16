---
status: live
scope: engine
verdict: M15 wired the V4 lightning indexer end to end with ZERO new kernels — the pin places the 374 MB of attn.indexer.* it always counted, blocksel.rs drives the already-scored spread/scorer pair per step, and the 2052-token --ctx refusal is deleted; below that boundary the arm is the pre-M15 engine byte for byte (scored-vs-positional buffers proven identical), the engine's top-k reproduces the frozen oracle's selection set on every toy row (that gate red-proofed directly, by reversing the ranking comparator — the oracle-side tie flip provably could not have reddened it, since it moved the order and not the set), and the three --attn sparse cells flipped Refuse→FallbackLoudly. Device gates (ids A/B at ctx 2048, boundary-crossing smoke) and the NLL-vs-position gate (needs M10's --ppl) are owed and listed.
---

# V4 scored block selection (M15)

**The question**: `crate::v4`'s declared deviation 2 — compressed-block selection was
POSITIONAL, agreeing with the trained-in lightning indexer on the block SET only below
`positional_context_limit(index_topk)` = 4·(512+1) = **2052 tokens**, and `check_context`
hard-refused any `--ctx` at or above that at the door. The deviation's own text promised
"the ceiling goes away when the indexer does." M15 is that landing.

## What landed (all host-verifiable claims gated on every `cargo test`)

- **`v4/pin.rs` places the indexer weights** — `attn.indexer.{wq_b, weights_proj,
  compressor.*}` on the 21 ratio-4 layers, ~374 MB that `safetensors_bytes` always counted,
  so the pool loses nothing. `wq_b` is fp8 `[8192, 1024]` on the same 128-block scale grid
  as every attention projection; the rest is f32 (converter-widened bf16, narrowed back
  exactly at engine build).
- **`v4/blocksel.rs` (new) drives the selection** through kernels that all existed and were
  all scored before this milestone: `gemv_fp8_bf16` (wq_b), `rope_adjacent`,
  `act_quant_f4_rotated` (the Hadamard-fp4 spread), `gemm_bf16` (weights_proj), and
  `index_score_blocks` — the scorer `blockindex.hip` carried since S2c with the note "not
  wired, not dead", bit-identical against the oracle in `kernel_v4_indexer.rs`. **Zero new
  launchers; the kernel census is untouched at 60/60/0.** Host work per scored step is the
  `weights_proj` scale round-trip (`[m, 64]` floats, 256 B at decode — the oracle's two
  bf16 stores around `wscale` happen there) and the top-k assembly.
- **`v4/select.rs::scored_rows`** is the pure top-k: causal mask by index, descending
  score, **ties toward the lower block index** (the frozen oracle's `topk_idx` rule),
  survivors emitted ASCENDING with a `-1` tail. Ascending is load-bearing: wherever the
  causal limit fits under `index_topk`, the scored row is byte-identical to the positional
  fill whatever the scores say.
- **The gating rule**: an engine whose `max_ctx` sits below the boundary builds NO indexer
  state and is the pre-M15 arm bit for bit. Above it, the indexer's nested compressor runs
  on every indexed-layer call from position 0 (state continuity — a compressor first
  consulted at the boundary would score against an empty cache), and the scoring kernels
  run only once `end_pos/ratio > index_topk`.
- **`check_context` deleted** (door and engine); `--ctx` on V4 is bounded by memory only.
- **Legality**: `V4_SPARSE_ATTN_IS_NOT_A_CHOICE` rewritten (its old text cited the
  positional refusal); the three sparse `--attn` cells flipped **Refuse → FallbackLoudly**
  on the `--mode` precedent — the checkpoint owns the attention, the flag toggles nothing,
  and refusing `--attn dsa` on the one arm whose attention IS natively block-sparse would
  kill exactly the runs the flag's intent describes. `decide()` stays total; the row test
  pins the flip cell by cell as `deepseek_v4_row_is_the_m15_truth`.
- **`tests/smoke-v4.sh` (new, thin)**: V4's own refusal fragment (`--mtp`), the
  fallback-flip cell (`--attn dsa` must warn AND decode ids identical to the recorded
  dense run), and a ceiling cell whose ~2.6k-token prompt prefills ACROSS 2052 — with the
  crossing asserted from the engine's own PREFILL line, so the cell cannot pass vacuously.

## The gates and where each stands

| gate | form | status |
|---|---|---|
| (a) set equality | oracle-side: `crates/oracles/tests/v4_indexer_goldens.rs` — the exported score matrix DETERMINES the exported selection (list-equal, tie rule pinned) and below-cap rows keep every legal block. Engine-side: `crates/engine/tests/v4_scored_selection.rs` — `scored_rows` over the oracle's own exported scores reproduces the oracle's set on every row, truncation reached (anti-vacuity counters ≥ 3 rows), on the toy whose `index_topk = 2` puts the cap at 12 tokens | **GREEN**, deviceless, every `cargo test` |
| (a′) below-cap byte identity | scored fill ≡ positional fill: adversarial synthetic scores (select.rs unit) AND real oracle scores (engine test), both phases | **GREEN**, deviceless |
| (b) ids at `--ctx 2048` | byte-compare `--bench 32 --ctx 2048` pre-M15 (264758c) vs M15 — by the below-cap identity the claim is IDENTITY, strictly inside M8's registered standard | **OWED — GPU session** |
| (c) NLL-vs-position across 2052 | needs M10's `--ppl` (teacher-forced NLL), unmerged in this tree | **BLOCKED on M10**; the deviceless half (keep-oldest ≠ scored on truncated rows — the sabotage's observability) is GREEN |
| smoke | `tests/smoke-v4.sh`, 3 cells | **OWED — GPU session** (needs the recorded ids capture) |

**Gate (b)'s standard, cited exactly**: `docs/investigations/rewrite.md` §M8 exit gate —
cross-engine drift quantified with the reference's own instrument and calibrated against
`old:docs/investigations/v4-decode-decomposition.md` §M9's registered intra-engine standard
(17/512 flips, flip-gap median 0.099 vs 3.19 overall, max |Δlogit| 8.14 — "the drift
resamples ties, it does not degrade"); M8 measured 30/32 argmax-identical with both flips
at near-ties (gaps 0.21 and 1.40 against an agreeing median of 3.83). M15's below-cap
claim is stronger than the standard: byte-identical selection buffers, so any id drift at
`--ctx 2048` is a defect, not drift.

## Red proofs (recorded in measurement/gate-red-proofs.md §5)

Six, on **both** sides of the gate — the split matters, because a perturbation of the
oracle-side recompute says nothing about whether the engine-side comparison resolves
anything, and the tie flip proves it: it changed the ORDER and not the SET, so the
engine-side set-equality gate could not have seen it.

*Oracle side*: the tie-rule flip reddened the list-equality on a REAL tie; score
perturbation above the boundary moves the recomputed set and below it provably cannot; the
keep-oldest sabotage disagrees with the scored selection on every boundary-crossing
fixture — which is what makes the eventual NLL-cliff red-proof non-vacuous.

*Engine side, perturbing `scored_rows` itself*: reversing the ranking comparator (keep the
LOWEST scores) reddened the set gate at `left: {16, 18} right: {17, 18}` while the
below-cap identity stayed green — resolution and specificity in one run; deleting the
ascending re-sort reddened both below-cap byte-identity tests, which is what proves the
ascending emit load-bearing rather than tidy. A sixth proof covers the rectangle check
`gather_scored` ends in: with the per-row width test removed, a compensating-ragged
buffer (comp widths 2, 1, 3 over three rows) was accepted as a valid `(3, 5)` rectangle,
so the aggregate total that stood alone before M15 was blind to exactly the raggedness a
caller-ranked selection can produce.

## What the selection can and cannot claim against the reference

Given bit-identical inputs the scorer is bit-identical to the oracle — but its inputs ride
tolerance-gated kernels (projections, pooling), so near a score TIE the engine and the
reference can legitimately pick different blocks; the oracle's own scoring note records
that no numeric tolerance can see this. Hence: set-equality where sets are determined,
plus NLL smoothness across the boundary for the rest — never a bitwise selection claim at
real depth. The attended ORDER is ascending-by-position where the reference's is
score-descending: a summation-order difference inside the online softmax over the same
set, the same class of difference every tolerance-gated kernel already carries.

## Cost, stated

Below the boundary: zero — no allocation, no launch, no byte moved. Above it, per step:
the nested compressor (two `[m,4096]×[256,4096]` bf16 GEMMs + pooling) on every
indexed-layer call, plus — only past 2052 — wq_b GEMV, rope+spread over `[m·64, 128]`, the
score kernel (`m·n_comp` threads × 8192 MACs), one 256 B and one `4·n_comp` B D2H, and a
host top-k over `n_comp ≤ max_ctx/4` candidates. Device memory when armed:
`iq [max_m, 8192]` f32 (134 MB at ctx 4096) + score `[max_m, max_ctx/4]` (16 MB) + 21
caches `[max_ctx/4, 128]` (11 MB) + narrowed weights (~98 MB) ≈ **260 MB of scratch**,
same accounting class as the existing `max_m`-sized activation scratch (charged to free
memory, not the partition — `weights_only_floor` carries what that means). tok/s deltas
above/below the boundary are the GPU session's to measure.
