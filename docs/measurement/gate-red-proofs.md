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

### 5a-2. The `p4` discriminator, re-specified and re-proved (2026-08-17)

The first device battery reddened `p4`, and **the red was a specification error**: the cell
demanded byte-identical NLLs across `--max-mem` 115 and 70 on an arm that does not repeat
itself. The control (same budget twice) moved 555 positions against the budget's 553.
`docs/investigations/glm-nondeterminism.md` holds the measurement; what belongs here is
that the replacement was proved before it was believed.

The cell now runs A, A' (control) and B, and reads two intervals out of `bin/ppl`. Proved
deviceless against synthetic `.nll` files carrying the observed jitter shape (761
positions, ~72% of them moving, calibrated so the control's mean lands at the measured
0.0018 nats), driving **the script's own verdict code**, not a copy:

| control CI | budget CI | verdict |
|---|---|---|
| `[-0.00214, +0.00635]` | `[-0.00665, +0.00157]` (null effect) | GREEN |
| `[-0.00214, +0.00635]` | `[+0.01017, +0.01887]` (+0.015 nats) | RED |
| `[-0.00214, +0.00635]` | `[-0.04569, -0.03405]` (budget BETTER) | RED — two-sided |
| `[+0.00300, +0.00900]` (excludes 0) | any | UNCALIBRATED, exit 1, never a pass |

Resolution bought: a ~0.0043-nat half-width, against the 0.0134–0.0172-nat quality gaps
the ladder must resolve — ~3x headroom.

**LENGTH-AWARE, added 2026-08-17 after the text-level result.** Divergence on this arm is
stochastic in the position at which it first fires — GLM is byte-identical at 32 generated
tokens and differs at 512, and the same command's first divergence moved from position 13
(loaded box) to 452 (quiet box). Two consequences are now built in rather than assumed:

- **Every verdict this cell prints carries its scored-position count**, because
  "byte-identical" and "the control spread is X" are different experiments at different
  lengths and a reader comparing two runs must be able to see that.
- **The strict branch CONFIRMS BEFORE CONVICTING, by repeating the BUDGET arm.** A
  matching control pair plus a differing B could be the budget, or it could be arm B
  catching a one-off — and convicting the budget of the engine's wobble is exactly what
  the first version of this cell did on a real device run. So a strict-branch difference
  triggers a fourth arm, on the red path only.
  **Which side to repeat is not a matter of taste, and the first attempt at this fix got it
  wrong** (it re-ran the CONTROL). Model a run as modal with probability `q`, else divergent
  in a one-off way that does not recur — which is what this looks like, being stochastic in
  the position it first fires. Under no budget effect every arm is an independent draw:

  | rule | false-conviction rate | q=0.90 | q=0.70 |
  |---|---|---:|---:|
  | `A==A'` and `A!=B` | `q²(1−q)` | 0.0810 | 0.1470 |
  | `A==A'==A''` and `A!=B` | `q³(1−q)` | 0.0729 | 0.1029 |
  | `A==A'`, `A!=B`, **`B==B''`** | ~0 | ~0 | ~0 |

  A second CONTROL only sharpens the estimate of `q`; the false conviction comes entirely
  from the B side, and it does nothing about that. Sharper still: the second control
  multiplies the true and false rates by the SAME `q`, so the likelihood ratio
  `q²/[q²(1−q)] = q³/[q³(1−q)] = 1/(1−q)` is UNCHANGED — it carries no information about the
  budget whatsoever, and only makes REDs rarer in both directions, by exactly a factor of
  `(1−q)`. A second BUDGET arm kills the confound instead, because a one-off does not repeat
  — while a REAL budget effect is a different but STABLE output, so `B==B''` still holds and
  the rule convicts. Same cost, one arm.

  Two things the rule pays for, both stated rather than discovered later:

  - **Power.** Convicting needs `q⁴` instead of `q²`, so at `q=0.90` a genuine P4 violation
    is reported UNCALIBRATED rather than RED about 34% of the time (19% before). The
    direction is safe — no path turns a real violation into GREEN, because a strict GREEN
    still requires byte-identity of A and B — and `MEM_B` streams ~1.7x the misses of
    `MEM_A`, so it is a priori the arm likelier to wobble. **Expect UNCALIBRATED, not RED,
    to be the strict branch's usual answer on a GLM-like arm.** That is the correct
    diagnosis, not a gate failure.
  - **The `≈0` rests entirely on one-offs being UNIQUE.** If the divergence turns out to be
    a race with two STABLE attractors rather than a diffuse one-off — second mode reached
    with probability `r` at `MEM_B` — then `B==B''` happens with probability `r²` and the
    false-conviction rate is `q²r²`, which at `q=0.9, r=0.3` is 0.073: no better than the
    rule it replaced. The evidence (first-divergence position moving 13→452 with host load)
    favours diffuseness but does not establish it. If `wave/fix-glm-determinism` finds a
    two-attractor race, this rule needs revisiting.

So the cell's failure mode under a non-reproducing arm is now **UNCALIBRATED (exit 1, never
a pass), not a false conviction** — which is the property that matters for the three
branches inheriting it.

**Two design errors were found and fixed while proving it**, both of the kind that produce
a confident wrong number rather than a failure:

- **The first replacement was also wrong.** It demanded the two intervals be DISJOINT. But
  A' and B are both paired against A, so both intervals carry A's noise and non-overlap
  double-counts it — the test came out ~2x less sensitive than the data supports, and a
  +0.004-nat effect passed. Writing the model out (`d(A→B)ᵢ = δ + εᵢᴮ − εᵢᴬ`) shows the
  interval already estimates δ at the right SE, so the question is whether it clears zero,
  and the control's job is to prove zero is where a null lands.
- **`bin/ppl` prints intervals in its VERDICT block too**, so grepping its whole output for
  a CI can return a verdict's interval as if it were a table row. Found when `p4` needed
  two rows; `tf` had been taking `head -1` off an untruncated grep since it was written and
  is now fixed too. Both parses assert their expected row COUNT.

And one gate-shape bug the battery exposed directly: **`p4` reddening meant `tf` never
ran** — the one cell that validates scoring against the pinned reference was lost to an
unrelated failure two cells earlier. Cells now run in subshells and the battery continues;
setup (2) and discard (3) still abort, because those say the measurement could not be
taken at all.

> **CORRECTED 2026-08-17, same day: the FIRST fix for that did not work, and this paragraph
> asserted it did.** The driver was `(run_cell "$c") || exit $?` — a subshell in a `||` list
> whose right-hand side re-raised the cell's own exit code, so a red still killed the
> battery. Reproduced with a 15-line mirror, then fixed by moving the code mapping to the
> loop and removing the `exit` on a red.
>
> The same bash rule has a second consequence worth writing down, because it cannot be
> worked around: **`set -e` is disabled INSIDE a subshell that is an operand of `||`, and it
> cannot be re-armed** — `( set +e; set -e; … )` still runs past a failing command (measured;
> only a fresh `bash -c` restores it). So every failure inside a cell must be checked
> explicitly, which is what `run_arm`/`score_arm` returning into a tested variable, the
> `cmp`s inside `if`, and the row-count assertions after each parse are for. A step added to
> a cell without an explicit status check is silently ignored.

### 5b. Engine half — OWED, needs the device

Three runs, each a source mutation, `--expect-red`, then `git checkout` and a green:

| cell | mutation | must redden with |
|---|---|---|
| `profile` | `telemetry.rs`: `Phase::Ffn => self.ffn_ns += ns,` → `Phase::Ffn => {}` — the whole ffn accumulation, gone | `--expect-red='ffn bucket is'`. This is the run that proves the stamps are WIRED, which 5a cannot |
| `p4` | none — `--red-proof-corpus` scores arm B on a one-word-different corpus, no rebuild | the budget interval EXCLUDES zero against a control that contains it. Note this is now a stronger proof than it was: the perturbation has to clear a MEASURED noise floor, not merely be non-zero |
| `tf` | `glm/decode.rs`: `tally.push(&row, own, target)?` → `tally.push(&row, own, ids[i - 1])?` — position i's row scored against position i-1's token | the CI lands entirely outside ±0.00995 |

The `tf` mutation is in `glm/decode.rs` and not in `score::walk` because **GLM does not
come through `walk`** — its whole score runs inside one `block_on`, so it keeps a bespoke
loop over the same `Tally`, and the `tf` cell scores GLM. Mutating `walk` would redden
nothing here. (`walk`'s own alignment is covered deviceless by
`walk_scores_every_position_and_forces_all_but_the_last`, which asserts the exact
`advanced == [(1,1),(2,2)]` sequence this mutation breaks.)

**Both mutations were pre-verified to COMPILE, 2026-08-16, and reverted.** That is not
ceremony: §4 above records two operator false-greens where the mutation orphaned a
variable, `warnings = deny` failed the rebuild, the exit code was eaten by a pipeline, and
the "red-proof" ran against a stale binary — a proof that refused to go red, and a red with
the wrong failure. Each of these keeps its variables used (`target` is still read by
`forward`; `ns` is still read by the other three arms), so neither can produce that
failure. Run:

```
cd .claude/worktrees/wave-m10
sed -i 's/Phase::Ffn => self.ffn_ns += ns,/Phase::Ffn => {}/' crates/engine/src/telemetry.rs
CARGO_TARGET_DIR=... cargo build --release --features teacher-forcing   # OUTSIDE the flock, exit UNPIPED
PPL_BIN=.../release/rivoli tests/ppl-gates.sh --expect-red='ffn bucket is' <artifact> profile
git checkout -- crates/engine/src/telemetry.rs        # never sed a file back
CARGO_TARGET_DIR=... cargo build --release --features teacher-forcing
PPL_BIN=.../release/rivoli tests/ppl-gates.sh <artifact> profile        # green again
```

`p4`'s red-proof scope, stated because it is narrower than it looks: it proves the paired
discriminator and the control-relative anti-vacuity check are live. It does **not** simulate
a residency-dependent format defect — the only real one on record is `--mode hybrid`, whose
cache picks each expert's format, and hybrid does not decode in this tree (it refuses at
`FormatPlan`). When it lands, IT is this cell's red-proof, and this paragraph is the
instruction to make it so.

**What is already known about `tf` before it runs, and it matters because three branches
depend on `--ppl` (2026-08-17, zero GPU):** the old tree recorded GLM int3-vq at
**PPL 5.222720** over `tests/ppl-corpus.txt` — the same 762-token corpus this tree vendored
byte-identically (`old:docs/investigations/int4-scales.md`, re-measured 2026-07-31 on the
current artifact, `--cache-policy lru --max-mem 100`). This tree's two runs give
**5.200080 / 5.209284**, mean 5.204682:

```
dPPL  -0.018038  (-0.345%)      dNLL  -0.00346 nats
      = 2.0x the measured noise floor,  0.35x the 1% quality bar
```

So the scorer's ABSOLUTE SCALE is independently corroborated. That is a weaker statement
than the `tf` cell's paired equivalence and does not replace it — the arms differ in cache
policy and budget (both output-neutral in int3-vq per INV-1, though only up to the noise
floor now that one is measured), and the old figure is a single run whose own floor is
unknown, so the residual 2x-floor gap is not attributable. But it does bound the failure it
rules out: an off-by-one target on real prose would land near the corpus entropy, order
**5 nats**, and a broken softmax would produce a non-finite NLL that `nll_of` refuses. A
0.003-nat agreement is not what either looks like.

**`profile` has since run GREEN on the device** (2026-08-17: `other` 0.265% of wall,
census 3/3, remainder re-derived to 0.001 ms on the real line — see
`docs/measurement/baseline-2026-08-16.md` for the numbers and for why that run's absolute
wall is not citeable). A green is not a red proof; the mutation below is still owed. What
the green does establish is that the stamps produce a coherent profile at all, so a
subsequent red can be attributed to the mutation rather than to the instrument.

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
