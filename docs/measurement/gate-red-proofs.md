---
status: data
scope: engine
verdict: Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN. M7's anchor-decode gate is proven red in BOTH halves — deviceless (an absent capture name, a tolerance under its envelope) and on device (all four recipe rows executed 2026-08-16 with observed magnitudes matching old:'s, plus two recorded operator false-greens whose lesson is part of the record). M15's scored selection carries six deviceless proofs on BOTH sides of its gate: a tie-rule flip reddening the oracle-side list-equality on a real tie (left [[9,8]] vs right [[8,9]]) — which the engine-side set gate provably could not have caught, since the set was unchanged — a reversed ranking comparator reddening that engine-vs-oracle gate (left {16,18} vs right {17,18}) while the below-cap identity stayed green, a dropped ascending emit reddening the below-cap byte-identity on both real and adversarial scores, a removed per-row rectangle check accepting a compensating-ragged buffer the aggregate total was blind to, and two standing fixtures pinning score-perturbation resolution above the boundary against provable inertness below it plus the keep-oldest sabotage's observability.
---

# M0 gate red proofs

Run 2026-08-15 on `rw/main` at commit `e922a34`, per P7: a gate that has never been red is
not evidence. Each proof was planted, observed red, reverted, and observed green again in
the same session. The exact reddening output is quoted; everything else is reproduction.

## 1. jscpd (duplication, build error)

Planted: the body of `rivoli_core::hash::fnv1a` copied into `crates/artifact/src/glimmer.rs`
as `planted_clone` (26 tokens — above the 15-token floor).

```
npx --no -- jscpd -c .jscpd.json --exitCode 7 crates   → exit 7   (red)
git checkout crates/artifact/src/glimmer.rs
npx --no -- jscpd -c .jscpd.json --exitCode 7 crates   → exit 0   (green again)
```

This gate also fired **unplanted, twice, during the M0 port itself** — the CodeScene
gate's private `fnv1a` against the ported `golden_read.rs` copy — which is a stronger
proof than the synthetic one: it caught duplication nobody believed they had written.
The fix (one owner in `rivoli-core::hash`) is commit `e922a34`.

## 2. Docs registry (front matter ↔ INDEX agreement)

Planted: `TAMPERED ` prepended to `how-to-measure.md`'s front-matter verdict, INDEX row
left untouched — the exact "corrected in one place only" drift the gate exists for.

```
test the_index_lists_every_doc_with_a_matching_verdict ... FAILED   (red)
git checkout docs/measurement/how-to-measure.md
cargo test -p rivoli --test docs                        → 3 passed  (green again)
```

The derived exemption ledger (third test in the same file) also fired unplanted during
the port: the frozen V4 oracle brought 3 ignore-marker regions and the test refused
CLAUDE.md's stale `0` until the ledger line was updated.

## 2b. Invariant registry (added 2026-08-15, M1)

Planted: the §8b table's `INV-1` renamed to `INV-9`, which breaks BOTH directions at
once — INV-9 documented with no test, `inv_1_*` tested with no row.

```
INV-[9] documented in architecture.md §8b with no `inv_<n>_*` test. ...   FAILED  (red)
revert → 1 passed                                                          (green again)
```

## 3. CodeScene (10/10 code health)

Two halves, one still owed:

- **Required-mode half, proven:** with no `CS_ACCESS_TOKEN` in the environment,
  `RIVOLI_CS_REQUIRED=1 cargo test --test codescene` panics —
  `CodeScene gate REQUIRED but cs did not run (crates/artifact/src/glimmer.rs): not JSON
  (unlicensed cs prints PAT prose here)` — naming the file and the cause. So CI (which
  sets the variable) cannot silently skip; classification is by output, and exit codes
  were measured to be uninformative (unlicensed `cs review` exits 1 like any failure).
- **Score half, OWED:** the standing fixture (`codescene-redproof/bad.rs.txt` must score
  < 10) and a planted-unhealthy-function run both require a licensed `cs`. The dev box
  has a cached cloud license JWT (exp 2026-08-31) but `cs review` demands the
  `CS_ACCESS_TOKEN` PAT, which is not in this environment. **First licensed run must
  execute both proofs and correct this doc in place with the observed scores.** Until
  then every local `cargo test` runs this gate in warn-and-skip mode, and the skip is
  invisible under libtest capture — stated here so nobody reads local green as health.

## 4. Muse Glimmer anchor decode (added 2026-08-16, M7's exit gate)

`crates/engine/tests/glimmer_anchor_decode.rs` scores the engine against the reference's own
logits; `crates/engine/tests/glimmer_anchor_widths.rs` is its deviceless half. **Both halves
are proven** — the deviceless half in its authoring session, the device half the same day
(below), on the box, under the flock.

**Proven, no GPU.** Both planted, observed red, reverted, observed green again:

```
# a renamed capture — the way a golden gate silently stops scoring
BRANCH_CAP → "post_feedforward_layernorm2.out"
  t0.L0.post_feedforward_layernorm2.out is not in the golden; it holds 1099 float
  tensors, e.g. ["t0.embed_norm.out", "t0.rope.cos", "t0.rope.sin"]     FAILED   (red)
revert → 2 passed                                                                (green again)

# a bound dropped under the envelope it was measured from
TOL 2e-1 → 1e-1
  TOL is 1.47x its measured envelope 6.8e-2. Under ~2x it reddens on a correct
  engine …                                                              FAILED   (red)
revert → 2 passed                                                                (green again)
```

**PAID 2026-08-16, same day, first GPU session.** All four rows executed on gfx1151 under the
flock, each reverted byte-identical, green after (3/3 in 0.38–0.68 s per run):

| # | mutation | observed red | `old:` said |
|---|---|---|---|
| 4 | both budgets roomy | `tight budget pinned 8 and streamed 0 with 0 fills` FAILED | `streamed 0` |
| 1 | `qk_scales` → q clamped to 1.0 | logits **6.779e-1**, branch 3.182e-1; the P4 test rightly stayed green (both arms wrong identically is still bit-identical) | 6.7e-1 |
| 2 | `rotated` predicate inverted | logits **1.113e0**, branch 5.094e-1 | 1.1e0 |
| 3 | k/v handles swapped at the attend launch | **1.023e0** / 8.364e-1 | 1.3e0 |

This tree's own green worsts (`--nocapture` — libtest eats a passing test's prints): logits
**5.742e-2** (text-1) / **6.954e-2** (text-2) vs `TOL` 2e-1; branch **2.993e-2** at L6 vs
`TOL_BRANCH` 1e-1. Note 6.954e-2 sits 2% ABOVE `old:`'s truncation-measured envelope top of
6.8e-2 — the round-to-nearest prediction ("slightly under") was wrong in direction and right
in magnitude; the margin to the bound is 2.9×, and this line is now the envelope's owner.

**Two operator false-greens during the paying, both the same trap, both recorded:** rows 4 and
1 first "ran" against a STALE binary because the mutation orphaned a variable, `warnings =
deny` failed the rebuild, and the build's exit code had been eaten by `| tail` /
`echo $?`-after-pipeline. The tell was a red-proof that refused to go red (row 4) and a red
with the WRONG failure (row 1 reddened the partition test, not the logits). Debug the
harness before the tree: check the build's exit UNPIPED, and read WHICH test failed, not
just that one did.

## 5. V4 scored-selection set gate (added 2026-08-16, M15, deviceless)

The M15 gate compares the engine's top-k (`v4::select::scored_rows`) against the frozen
oracle's own `Indexer.forward` exports on the toy (`index_topk = 2`, cap = 12 tokens, so
truncation is REACHED and counted — the "set-invariant goldens" trap the shipped 512 sets
below 2052 tokens). Six proofs; the four executed ones were reverted and the tree re-run
green, and proofs 2 and 3 stand as fixtures.

**Proofs 1-3 redden the ORACLE-side file; 4-6 redden the gate that actually compares the
engine to it.** That split is the point: proof 1's tie flip produced the same SET
(`[[9,8]]` vs `[[8,9]]`), so the engine-side set-equality gate could not have caught it —
and equally, nothing in proofs 1-3 shows the engine-side gate resolves anything at all.
A gate is proven by a perturbation of the code IT scores.

1. **Tie-rule flip** (`crates/oracles/tests/v4_indexer_goldens.rs`): the recompute's
   tie-break flipped from `.then(a.cmp(&b))` to `.then(b.cmp(&a))`. Observed red on a REAL
   tie in the fixture — `left: [[9, 8]] right: [[8, 9]]` — proving the list-equality gate
   resolves the tie rule, not merely the score order. Reverted, green (2 passed).
2. **Score perturbation, both directions** (standing fixture in the same file): promoting
   the best currently-excluded legal block past the winners MOVES the recomputed set above
   the boundary (resolution), and the same promotion on a below-cap row provably cannot
   (specificity — the below-cap identity as an executable).
3. **Keep-oldest sabotage** (`crates/engine/tests/v4_scored_selection.rs`, standing): the
   `min(k, limit)`-oldest selection — the exact bug `positional_context_limit` documents —
   disagrees with the scored set on truncated rows of every boundary-crossing fixture.
   This is the deviceless half of the M15 NLL-cliff red-proof: if it ever stops
   disagreeing, a sabotaged NLL run would BE the scored run and the cliff check would pass
   vacuously.
4. **Ranking comparator reversed** (`v4::select::scored_rows`, the code the engine-side
   gate scores): `row[b].total_cmp(&row[a])` flipped to `row[a].total_cmp(&row[b])`, i.e.
   keep the LOWEST-scoring blocks. `the_engines_selection_over_the_oracles_scores_names_
   the_oracles_set` went red on a real prefill row — `left: {16, 18} right: {17, 18}` —
   and the below-cap identity test stayed GREEN, which is the specificity half: below the
   cap the scores must not reach the selection at all. Reverted, 3 passed.
5. **Ascending re-sort removed** (same function, `sel.sort_unstable()` deleted so survivors
   come out in score order): both below-cap byte-identity tests went red — the engine one
   on real oracle scores, the `select.rs` unit one on adversarial synthetic scores — plus
   the set gate's own layout assert (`row 7: not ascending`). This is what proves the
   ascending emit is load-bearing rather than tidiness, since the SET is unchanged by it.
   Reverted, green.
6. **Per-row rectangle check removed** (`v4::select::assemble`, leaving the aggregate
   `rows * cols` total that stood alone before this proof): the compensating-ragged case
   added to `a_ragged_scored_row_is_refused_by_the_rectangle_check` — comp widths 2, 1, 3
   over a 3-row prefill, summing to exactly `3 * comp[0].len()` — was ACCEPTED as a valid
   `(3, 5)` rectangle. So the total was blind to raggedness whose entry count happens to
   balance, which `gather_scored` can be handed because its compressed rows come from a
   CALLER. The per-row width check is the fix and this is its proof; restored, 11 passed.
