---
status: data
scope: engine
verdict: Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN.
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
