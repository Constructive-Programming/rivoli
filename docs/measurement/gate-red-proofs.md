---
status: data
scope: engine
verdict: Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN. M7's anchor-decode gate is proven red in BOTH halves — deviceless (an absent capture name, a tolerance under its envelope) and on device (all four recipe rows executed 2026-08-16 with observed magnitudes matching old:'s, plus two recorded operator false-greens whose lesson is part of the record). M10's three gates split each proof into a CLASSIFIER half (paid 2026-08-16, deviceless: 6 planted defects red, --expect-red inverted both ways, the small-bucket row showing why the band alone is not the gate) and an ENGINE half (OWED, needs the device and a source mutation) — plus one gate that reddened unplanted on its own author, the argmax fold that let a leading NaN win.
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

## 5. M10: the phase profile and the teacher-forced scorer (added 2026-08-16)

`tests/ppl-gates.sh` carries three cells. **Every gate here splits into two claims that
must be proven separately, and conflating them is the trap this section exists to name:**

- **the CLASSIFIER claim** — the gate's arithmetic goes red on the defect it is for. Needs
  no device: `PPL_REPLAY_LOG` re-classifies a hand-written arm log, and the CI band is
  read from `bin/ppl`'s printed interval. **PAID below.**
- **the ENGINE claim** — the numbers that arithmetic reads are what the engine actually
  stamps. Needs the device and a source mutation. **OWED below.**

A classifier proof is *not* evidence about the engine. It is evidence that a green from
this gate means something, which is the prerequisite.

### 5a. Classifier half — PAID 2026-08-16, no GPU

`--expect-red[=FRAGMENT]` inverts the classification so a proof is judged by the SAME code
the green is. All three directions were exercised, against a `wall 390.000` line:

| planted into the PROFILE line | observed |
|---|---|
| `ffn 0.000` (the dominant bucket's accumulation dropped) | `ffn bucket is 0.000 — the accumulation was dropped` |
| `head 0.000` (a SMALL bucket dropped — **the band alone does not see this**, `other` moves 6%) | `head bucket is 0.000 — the accumulation was dropped` |
| `other -6.000` (one span stamped into two buckets) | `other is -1.538% of wall, under the -0.05% floor` |
| `other 104.000` (wall going somewhere unstamped) | `other is 26.667% of wall, over the 15.00% ceiling` |
| `other 0.000` while the four buckets leave 26 ms | `the reported other 0.000 ms disagrees with wall - buckets = 26.000 ms (eps 0.020)` |
| line deleted (the report reworded) | `no PROFILE/tok line on the run's log` |
| `ffn` renamed `moe` (the format changed) | `PROFILE/tok line did not parse` |
| healthy (`other 4.000` of `wall 390.000`) | `ok: other 1.026% of wall, census 3/3 non-zero, remainder re-derived to within 0.020 ms` |
| `attend 0.030` — a genuinely MICROSECOND bucket | green, correctly: not a dropped stamp |

And the inversion itself, all three ways:

| invocation | against | observed |
|---|---|---|
| `--expect-red=ffn bucket is` | the `ffn 0.000` line | `RED-PROOF OK: 1 cell(s) went red as demanded` (exit 0) |
| `--expect-red=ffn bucket is` | a DELETED line (a red, but a different one) | `RED-PROOF FAILED: … went red, but on 'no PROFILE/tok line…' — which does not contain 'ffn bucket is'` (exit 1) |
| `--expect-red` | a healthy line | `RED-PROOF FAILED: every cell came out green` (exit 1) |

Four of these rows exist because two independent reviews found the first draft of this
cell weaker than it claimed, and each is now a row rather than a promise:

- **The `head 0.000` row is why the cell is not just a band.** `head` is a few percent of a
  GLM token, so dropping it moves `other` by less than the band's width; only the
  per-bucket census catches it.
- **The `other 0.000` row is why the cell does not trust the number it audits.** `other_ms`
  is the ONE derived field the engine reports; the first draft read it and checked nothing
  else, so hard-coding it to zero passed the band (0% is inside it) AND the census (three
  buckets still non-zero). The classifier now re-derives `wall − (attend+ffn+fetch-wait+head)`
  from the same line and reds on a disagreement past 0.02 ms.
- **The `attend 0.030` row is why the buckets print microseconds.** At `{:.1}` ms a
  genuinely-µs bucket prints `0.0` and the census calls it dropped — and
  `ProfileSummary`'s own table says Glimmer's and V4/K3's attend buckets ARE launch-only
  µs. The cell would have gone falsely red the first day it was pointed at a second arm.
- **The wrong-reason row is why `--expect-red` takes a FRAGMENT.** §4 above records a
  red-proof that "passed" while reddening the WRONG test. Any-red-counts reproduces that
  class exactly; the fragment form does not.

The mirror admission, unfixed and deliberate: `fetch-wait` is NOT in the census, because
`0.0` is its SUCCESS value (fetch fully hidden behind resident compute is the design), so
a dropped `fetch-wait` stamp is a **known blind spot** of this cell.

TF equivalence classifier, same day, against `bin/ppl`'s printed 95% CI and the
pre-registered `±ln(1.01) = ±0.00995` nats band:

| CI fed in | verdict |
|---|---|
| `[+0.00000, +0.00000]`, `[-0.00100, +0.00120]`, `[+0.00990, +0.00994]` | GREEN |
| `[+0.01200, +0.01500]` | RED (entirely worse) |
| `[-0.05000, -0.02000]` | RED (entirely BETTER — two-sided on purpose; a rewrite that reliably beats the reference is a misaligned position, not a free lunch) |
| `[-0.00200, +0.01100]`, `[-0.90000, +0.90000]` | INCONCLUSIVE, exit 1 — never a pass |

### 5b. Engine half — OWED, needs the device

Three runs, each a source mutation, `--expect-red`, then `git checkout` and a green:

| cell | mutation | must redden with |
|---|---|---|
| `profile` | delete `self.prof.lap(Phase::Ffn, t)` after the compute-stream await (`glm/mlp.rs`) | `--expect-red='ffn bucket is'` — and this is the run that proves the stamps are wired, which 5a cannot |
| `p4` | none — `tests/ppl-gates.sh --red-proof-corpus` scores arm B on a one-word-different corpus, no rebuild | the NLL bodies differ, with the first differing position named |
| `tf` | off-by-one the forced position in `score::walk` (`tally.push(&row, own, ids[i])` → `ids[i-1]`) | the CI lands entirely outside ±0.00995 |

`p4`'s red-proof scope, stated because it is narrower than it looks: it proves the byte
comparison and the anti-vacuity hit_pct check are live. It does **not** simulate a
residency-dependent format defect — the only real one on record is `--mode hybrid`, whose
cache picks each expert's format, and hybrid does not decode in this tree (it refuses at
`FormatPlan`). When it lands, IT is this cell's red-proof, and this paragraph is the
instruction to make it so.

**A rebuild sits between the green and the red-proof of `profile` and `tf`,** which evicts
page cache (ms/miss 1.36 → 5.14, measured). It does not invalidate either proof: both
reden on a bucket going to zero or an interval moving by orders of magnitude, neither of
which a colder cache can manufacture. It WOULD invalidate a tok/s comparison across the
pair, so do not make one.

### 5c. A gate that fired unplanted, on its author

`crates/engine/src/score.rs`'s own NaN test reddened the first time it ran:
`host_argmax(&[NaN, 2.0, 1.0])` returned index **0**, not 1. The fold seeded `best` from
`row[0]`, and `NaN > x` is false in both directions, so a leading NaN wins a `>`-fold — the
`f32::max`-swallows-NaN class in its exact repo-native form. The device kernel
(`kernels/fwd.hip::argmax_reduce`) seeds with `-INFINITY` and spells `va != va` out; the
host now does the same. Unfixed, the two folds disagree on exactly one input and the
disagreement surfaces as a bogus *coherence* refusal instead of the true "non-finite
logits" one. Written down here because a gate catching its own author before the first
device run is worth more than a planted proof.
