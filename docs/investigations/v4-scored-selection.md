---
status: live
scope: engine
verdict: M15 wired the V4 lightning indexer end to end with ZERO new kernels — the pin places the 374 MB of attn.indexer.* it always counted, blocksel.rs drives the already-scored spread/scorer pair per step, and the 2052-token --ctx refusal is deleted; below that boundary the arm is the pre-M15 engine byte for byte (scored-vs-positional buffers proven identical), the engine's top-k reproduces the frozen oracle's selection set on every toy row (that gate red-proofed directly, by reversing the ranking comparator — the oracle-side tie flip provably could not have reddened it, since it moved the order and not the set), and the three --attn sparse cells flipped Refuse→FallbackLoudly. ON SILICON 2026-08-16 the wiring ran for the first time and both device gates passed: the ids A/B at --ctx 2048 is byte-identical pre-vs-post INCLUDING headers with the same routing work on both arms (6217 hits / 1781 misses), and smoke-v4.sh is 3/3 — its ceiling cell prefilling 2474 tokens through the scored selection on an invocation that was refused at the door before M15, then decoding 32/32 ids identical to the recorded pre-M15 capture with --attn dsa warning and changing nothing. Owed: the NLL-vs-position gate (blocked on M10's --ppl) and a CodeScene score for blocksel.rs (CS_ACCESS_TOKEN unset, attempted twice, not guessed). No tok/s is claimed — dev-profile binaries, warm cache.
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
| (b) ids at `--ctx 2048` | byte-compare `--bench 32 --ctx 2048` pre-M15 (264758c) vs M15 — by the below-cap identity the claim is IDENTITY, strictly inside M8's registered standard | **GREEN** 2026-08-16. Both arms dev-profile, isolated target dirs, `flock`ed, KFD witness 0 before and after each. `cmp` byte-identical INCLUDING headers, and both reported the same routing work (6217 hits / 1781 misses) — so the identity is of the work done, not only of the text. Capture committed as `tests/v4-bench32-ctx2048.ids` |
| (c) NLL-vs-position across 2052 | needs M10's `--ppl` (teacher-forced NLL), unmerged in this tree | **BLOCKED on M10**; the deviceless half (keep-oldest ≠ scored on truncated rows — the sabotage's observability) is GREEN |
| smoke | `tests/smoke-v4.sh`, 3 cells | **GREEN 3/3** 2026-08-16, exit 0. Witness 0 before and after; during the run the only `/dev/kfd` holder was this suite's own `rivoli` (parent chain `flock` -> the script), so no arm is discarded |

**Gate (b)'s standard, cited exactly**: `docs/investigations/rewrite.md` §M8 exit gate —
cross-engine drift quantified with the reference's own instrument and calibrated against
`old:docs/investigations/v4-decode-decomposition.md` §M9's registered intra-engine standard
(17/512 flips, flip-gap median 0.099 vs 3.19 overall, max |Δlogit| 8.14 — "the drift
resamples ties, it does not degrade"); M8 measured 30/32 argmax-identical with both flips
at near-ties (gaps 0.21 and 1.40 against an agreeing median of 3.83). M15's below-cap
claim is stronger than the standard: byte-identical selection buffers, so any id drift at
`--ctx 2048` is a defect, not drift.

## First silicon, 2026-08-16 — the scored path actually ran

Nothing in `blocksel.rs` had ever executed before this run: `kernel_v4_indexer.rs` scores
the kernels in isolation and the deviceless gate feeds `scored_rows` the oracle's scores,
so the WIRING met a GPU for the first time here. What it proves, cell by cell:

- **`refuse: mtp`** — `V4_MTP_NEEDS_A_KERNEL` fires with V4's own kernel-shaped reason
  ("missing KERNEL, not a missing head"), i.e. the arm's other refusal did not drift while
  its neighbour flipped.
- **`ceiling gone`** — `--ctx 4096` with a ~2.5k-token prompt **prefilled 2474 tokens in
  one whole-prompt pass** and decoded 12/12 finite ids. 2474 is inside the cell's asserted
  2053–4083 window and read off the engine's own `PREFILL:` line, not the prompt's word
  count. **This is the milestone**: past position 2052 the block set on all 21 indexed
  layers is decided by the indexer's scores, and the identical invocation was REFUSED AT
  THE DOOR before M15. The prefill is a single call, so the crossing happens inside one
  `attention_block` sweep rather than accumulating over decode steps.
- **`--attn dsa` falls back loudly** — the rewritten const appears verbatim in the live
  `WARN` line, then the run decodes **32/32 ids identical to the recorded pre-M15
  capture**. That is the strongest available form of "the flag toggles nothing": not that
  the warning was printed, but that the text was unchanged by printing it.

**No performance number is recorded from this session, and that is deliberate.** Both
binaries are dev-profile, and the second arm of gate (b) ran with the first arm's page
cache warm — the tok/s difference between them is a cache artifact, which the identical
hit/miss counts confirm. A real V4 tok/s delta across the boundary needs `--release`,
cold-cache symmetry and its own booking; it is not M15's to claim. The one number that IS
evidence here is the token count, because it is a correctness fact.

## Red proofs (recorded in measurement/gate-red-proofs.md §5)

Seven, on **both** sides of the gate — the split matters, because a perturbation of the
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

## Raised in review, deliberately NOT done here

Three review rounds ran before the commit; most findings were taken. These four were
declined with the reason, rather than silently:

- **`scored_rows` should be a `Sel` method.** It takes `kind`, `index_topk`, `at` and
  `offset` as loose parameters, and `blocksel` derives all four from the same engine fields
  `attn::upload_selection` independently builds its `Sel` from — which is precisely the
  re-derivation `select.rs`'s own header was written about. The fix (`Sel::scored_rows`,
  one `Sel` built in `attention_block` and passed to both) is right and should happen.
  Declined for THIS commit because it re-plumbs the device path, and the device path's only
  gate is owed to a GPU session: making the change now means the first silicon run of M15
  tests a refactor rather than the milestone.
- **`tests/smoke-lib.sh`.** ~16 lines of `smoke-v4.sh` are verbatim `smoke-glm.sh` — the
  `BIN` resolution (which has already been fixed once, in one place), `LOCK` (which has
  already moved once), and the four cell helpers. Nothing can see it: `build.rs` scans
  `crates` only and `.jscpd.json` is `format: ["rust"]`, and both line-cap walkers start at
  `crates/` too, so root `tests/*.sh` is in a permanent blind spot. Declined because the
  extraction edits `smoke-glm.sh`, a GPU gate this session cannot run. Worth doing with the
  walkers extended to `tests/` in the same change.
- **`v4_scored_selection.rs` rebuilds the toy** instead of using `common::v4::toy_fixture()`
  (cached; three builds here). jscpd cannot see a 3-line clone under its `minLines: 5`.
  Declined as test-only, and because the fixture is what three of this milestone's red
  proofs were executed against.
- **CodeScene was not run.** Attempted twice — once host-side, once again in the GPU
  session — and `CS_ACCESS_TOKEN` is unset both times, so `cs check` exits telling you to
  set a PAT. (`cs` itself is installed, 1.0.36.) The 10/10 hard gate is therefore **the one
  merge gate M15 leaves unverified**, and no score is guessed here in its place. The two
  plausible triggers are `blocksel::scored_selection` (~113 code lines over four phases —
  its own comments already mark where the seams would go) and `scored_rows`'s six
  parameters. **Score `crates/engine/src/v4/blocksel.rs` before merge**; note that
  `crates/cli/tests/codescene.rs` warns-and-skips without a license locally and hard-fails
  in CI via `RIVOLI_CS_REQUIRED=1`, so this will surface there rather than here.

## Cost, stated

Below the boundary: zero — no allocation, no launch, no byte moved. Above it, per step:
the nested compressor (two `[m,4096]×[256,4096]` bf16 GEMMs + pooling) on every
indexed-layer call, plus — only past 2052 — wq_b GEMV, rope+spread over `[m·64, 128]`, the
score kernel (`m·n_comp` threads × 8192 MACs), one 256 B and one `4·n_comp` B D2H, and a
host top-k **per query row** over `n_comp ≤ max_ctx/4` candidates. That last one is the
term to watch: `scored_rows` runs one full sort per row, so a boundary-crossing PREFILL
does `m` sorts × 21 layers rather than the one a decode step does. Decode is unaffected;
prefill above the boundary is unmeasured, and `select_nth_unstable_by` under the identical
comparator yields the identical set in O(n) if it measures badly.

**Device memory when armed, corrected 2026-08-16** — the first table said ≈260 MB and
omitted the nested compressor's two `[max_m, cd]` projection scratches, which are its
largest allocation. Recomputed from `LayerCompressor::build` and `IndexerBank::new` at
ctx 4096 (`max_m` 4095, `cd` 256 read off the artifact's own
`attn.indexer.compressor.wkv` `[256, 4096]`, `d` 128, 21 indexed layers):

| buffer | each | ×21 |
|---|---|---|
| `iq [max_m, 64·128]` f32 | — | 134.2 MB |
| `score_dev [max_m, max_ctx/4]` f32 | — | 16.8 MB |
| `w_dev [max_m, 64]` f32 | — | 1.0 MB |
| `proj_kv` + `proj_score` `[max_m, cd]` f32 | 8.39 MB | **176.1 MB** |
| `w_kv` + `w_gate` narrowed bf16 | 4.19 MB | 88.1 MB |
| `blocks [max_ctx/4, d]` f32 | 0.52 MB | **11.0 MB** |
| `cache [max_ctx/4, d]` f32 | 0.52 MB | 11.0 MB |
| `wproj` narrowed bf16 | 0.52 MB | 11.0 MB |
| pooling states | ~8 KB | 0.2 MB |

**≈ 449 MB of scratch**, not 260 — same accounting class as the existing `max_m`-sized
activation scratch (charged to free memory, not the partition — `weights_only_floor`
carries what that means). tok/s deltas above/below the boundary are the GPU session's to
measure.

**A residual the deleted door used to hide.** `--ctx` is now bounded by memory alone, and
this scratch is QUADRATIC in it: `score_dev` is `(max_ctx-1)·⌈max_ctx/4⌉·4` bytes — 16.8 MB
at 4096, 268 MB at 16384, 4.3 GB at 65536 (the checkpoint's own
`original_max_position_embeddings`) — with `iq` linear at 2.1 GB there. It fails loudly at
`DeviceBuf::new`, but AFTER `V4Pin::build` has read nine gigabytes, where `check_context`
used to refuse at the door. `main.rs`'s `--ctx` help still quotes a linear "~51 KB of
device memory per token". Not fixed here: the honest bound is a footprint estimate, which
is the same unbudgeted-activation-scratch question `V4Engine::max_m`'s doc already names
and defers, and inventing a second ceiling inside the milestone that removed the first one
would be the wrong shape.
