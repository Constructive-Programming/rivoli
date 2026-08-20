---
status: data
scope: engine
verdict: Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN. M7's anchor-decode gate is proven red in BOTH halves — deviceless (an absent capture name, a tolerance under its envelope) and on device (all four recipe rows executed 2026-08-16 with observed magnitudes matching old:'s, plus two recorded operator false-greens whose lesson is part of the record). M15's scored selection carries seven deviceless proofs on BOTH sides of its gate: a tie-rule flip reddening the oracle-side list-equality on a real tie (left [[9,8]] vs right [[8,9]]) — which the engine-side set gate provably could not have caught, since the set was unchanged — a reversed ranking comparator reddening that engine-vs-oracle gate (left {16,18} vs right {17,18}) while the below-cap identity stayed green, a dropped ascending emit reddening the below-cap byte-identity on both real and adversarial scores, a removed per-row rectangle check accepting a compensating-ragged buffer the aggregate total was blind to, and two standing fixtures pinning score-perturbation resolution above the boundary against provable inertness below it plus the keep-oldest sabotage's observability. M10's three gates split each proof into a CLASSIFIER half (paid 2026-08-16, deviceless: 6 planted defects red, --expect-red inverted both ways, the small-bucket row showing why the band alone is not the gate) and an ENGINE half (OWED, needs the device and a source mutation) — plus one gate that reddened unplanted on its own author, the argmax fold that let a leading NaN win. M11's fp8 gates are PAID deviceless — layer_bytes stripped of its scale grid, sniff falling back to the compiled-in block, the converter at the wrong block and with one projection class skipped, and the parity script over six runs including both refusals and a green baseline — while its DEVICE half is OWED with recipes written down: the anti-fallback assert in glimmer_fp8_decode.rs. Slot::fill's third-zip-leg guard was RETIRED rather than proven - the parameter it checked was deleted, so the truncation has no shape. M11b's id pin is PAID on the real 27 MB tokenizer, 31 of 31 cases identical to apply_chat_template and red-proofed by closing a system turn with the non-stop token; its serve door ships BOTH halves (request framing and reply channel-splitting) after the request-half-only version was written and reverted as a regression, behind SIX red-proofed pure gates including the arch dispatch itself; a prefix-monotonicity property caught two streaming P0s that no non-streaming gate could see - a raw turn header streamed then the channel wedged forever, and a partial <|eot|> at the prefix boundary. Only the live SSE round-trip is OWED on the GPU. ADDED 2026-08-17: the GLM determinism gates (§6) — the id comparator, a prompt pinned by length+md5 that fired unplanted, INV-9 under a scan_free mutation, the probe's fold-slot layout under an inserted enum variant, and the column comparator's three refusals; the determinism gate's own 512-token GREEN is OWED and standing, because the engine is red. Also recorded there: a false EXCLUSION found in this round's own instrument — a fold omitted on dense layers made two runs 'agree' about a quantity neither measured, which no gate in this repo could see. ADDED 2026-08-17 (Phase 1): the per-fold probe flags — parse refusals, spin_rows reading nothing, and the column comparator refusing two logs from different fold configurations; the dash-for-disabled rendering was recorded OWED and PAID the same day, red-proofed by printing 0 instead.
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

## 6. The GLM determinism gates (added 2026-08-17, `wave/fix-glm-determinism`)

Five checks landed with the divergence investigation. Each was planted, observed red, reverted,
observed green again — and one of them reddened *by itself*, which is recorded because a proof that
fires unbidden is worth more than one you had to arrange.

**6a. `tests/determinism-glm.sh --self-test` (the id comparator).** Two id streams differing by one
token, and a truncated stream. Both must compare unequal. The truncated case is the one that
matters: a naive `diff <(sort)`-style comparator passes it, and that substitution has been made
before.

```
FAIL: the comparator did NOT redden on differing id streams  → exit 2   (red, on a planted no-op comparator)
FAIL: the comparator did NOT redden on a TRUNCATED stream    → exit 2   (red)
SELF-TEST ok ...                                                        (green)
```

**6b. The gate's recorded prompt, pinned by length + md5.** This one **fired on its own**: the pin
was written with a placeholder hash before the real one was computed, and the check refused.

> **CORRECTED 2026-08-17, same day.** As first written the pin ran only under `--self-test`, i.e.
> never on the path it claimed to protect — review named it. It now runs on EVERY invocation, and
> was re-proved there: a planted wrong hash makes `determinism-glm.sh <artifact> 512` exit 2 before
> it reaches the artifact check, with the exit code read UNPIPED (`| head` eats it — that is a
> recorded trap in this very file).

```
FAIL: the recorded prompt changed — 317 bytes, md5 18927a780b36b029d03450d2100e9242
      expected 317 bytes, md5 c26cb59ea0b3d5c3b83fa1a1e2fa3ee5                (red, unplanted)
→ real hash inserted → SELF-TEST ok, prompt is 317 bytes                     (green)
```

**6c. INV-9 (`inv_9_a_slot_is_not_reissued_until_its_copy_lands`).** Planted: `scan_free` ignores
its `landed` predicate, i.e. the invariant's own violation.

```
assertion `left == right` failed
  left: Some(0)   right: None      FAILED   (red — an un-landed slot was issued)
revert → 1 passed                          (green again)
```

An earlier draft added a SECOND test for this invariant, whose declared red proof reddened the
existing one as well. That is what the duplication was: the two asserted the same three things.
The existing test was renamed to carry the number instead — the registry check does not care which
test carries it, only that exactly one does.

**6d. The divergence probe's fold-slot layout — RED-PROVED, THEN THE TEST WAS DELETED, and the
sequence is the point.** `Probe::fetch_fold_slot` handed out `Q::Bh`'s address and its callers
reached `Sc`/`Se` with `.add(1)`/`.add(2)`. A test asserted the contiguity, and was planted against:
a variant inserted between `Bh` and `Sc` with `NQ` bumped so it compiles — exactly the change that
would silently make one layer's folds land in another layer's slot.

```
assertion `left == right` failed: Sc must follow Bh    FAILED   (red)
revert → 1 passed                                              (green again)
```

> **SUPERSEDED 2026-08-17, same day.** Review pointed out that a test guarding an offset assumption
> is the weaker half of the choice: make the layout unrepresentable instead. `FetchFolds { bh, sc }`
> now names both pointers, `Q::Se` is fetched by name, no `.add()` remains, and the test is gone
> along with the three SAFETY paragraphs that existed only to restate the offset. **The red proof is
> kept rather than deleted** because it is the evidence that the hazard was real — the reason the
> refactor happened at all — and because a reader finding the deleted test in history should find
> why it went. This is the one row in this file whose gate no longer exists, and it is the right
> outcome: a hazard removed beats a hazard tested.

**6f. The length classifier (`classify_lengths`).** Added because review pointed out the branch was
reachable only after two real 512-token GPU arms and so could never be shown red — and because it is
the code that had just been found WRONG (it filed a divergence as a setup error). Factored out and
driven from `--self-test`; planted: the unequal-length branch returns 2 (setup) instead of 1 (RED).

```
FAIL: unequal lengths must be RED (1), not a setup error   exit 2   (red)
revert → SELF-TEST ok                                      exit 0   (green again)
```

Exit codes read UNPIPED both times. Piped through `| tail` the mutated run reports 0, which is the
trap this file already records twice.

**6e. `tests/divergence-columns.sh`'s three refusals.** All observed on real inputs:

```
mismatched headers (v2 log vs v3 log)  → FAIL: the two logs declare different columns   exit 2
no header line                         → FAIL: ... has no 'rivoli-divergence' header    exit 2
header names more columns than rows    → FAIL: row 1 has 22 fields but the header ... 14 exit 2
```

It also reproduced the coordinator's hand-written comparison exactly on the first real pair — row
12427, `pos=164 layer=24`, column `h` — which is the closest thing to a green with independent
provenance this round has.

> **CORRECTED 2026-08-17, same day.** A fourth refusal was firing on the documented-NORMAL case:
> `paste` pads the exhausted side, so once the shorter log ran out a row carried `n` fields instead
> of `2n` and the guard reported "row 12001 has 11 fields but the header names 11 columns" — a
> self-contradictory setup error on two logs of different length, which is what diverged arms always
> produce. It also made the honest "no differing row in the common prefix" message unreachable
> whenever the lengths differed. Both bodies are now truncated to the common prefix first, and the
> prefix case was verified to print its own message with exit 0.

**Owed, and standing.** The gate's own 512-token GREEN has never been observed, because the engine
is red: the first device pair diverged. That is the live red proof for the gate as a whole (§6a's
comparator is only the mechanism), and the green is owed once the defect is fixed. Recorded as owed
rather than claimed — this section exists so that distinction survives.

**One false-green caught in this round's own instrument, and it is the lesson.** The first version
of the probe folded the MoE's input `xn` on MoE layers only. GLM has three dense layers, so their
`xn` column stayed 0 in both runs — and a diff reads two equal zeros as "attention agreed" when
nothing was measured. A false EXCLUSION is the one failure mode an instrument may not have, and it
is invisible to every gate in this repo because the instrument was producing well-formed output the
whole time. Every column that cannot exist for a row now prints `-`, never 0.

**6g. The per-fold probe flags (added 2026-08-17, Phase 1).** Four checkable surfaces landed with
`--divergence-folds`; three have proofs, one is asserted and says so.

- **`Folds::parse` refusals** (`probe.rs::folds_parse_refuses_what_it_cannot_honour`). The
  assertions ARE the proof — `is_err()` on an unknown name and on two `sc` variants — because the
  failure being guarded is a silent success. Why it matters: every Phase 1 cell is "enable exactly
  one fold and see whether the pair still diverges", so a typo that quietly enabled NOTHING would
  make the cell green and be read as *"this fold is the mask"* — the precise inversion of the truth.
- **`spin_rows` reads nothing** (`fwd_kernel.rs::spin_rows_burns_time_without_reading_anything`).
  The property is invariance under the payload, plus non-zero and length-dependent so an elided
  loop or a store of 0 cannot pass. **Partly vacuous by construction and labelled so**: the kernel
  never receives the buffer pointer, so it *cannot* read it — the test guards a future signature
  change rather than today's code.
- **`tests/divergence-columns.sh`'s fold guard**, all three paths observed:

  ```
  two logs, different fold configs      → FAIL: ... DIFFERENT fold configurations   exit 2
  a log whose version implies no config → FAIL: ... declares no fold configuration  exit 2
  v2 logs (the token-164 pair)          → "folds: light (derived: v2 predates …)"   exit 0
  ```

  The derivation is what keeps the historical pair readable: v2 predates the fetch-path folds, v3
  could not disable them. Refusing them outright — the first version — would have made the one pair
  that produced a coordinate uncomparable.
- **The `-`-for-disabled rendering** (`probe.rs::an_unmeasured_column_prints_a_dash_and_never_a_zero`).
  **Recorded as OWED earlier the same day and paid before the round closed.** The obstacle was that
  `Probe` needs a `DeviceBuf`, so the rule lived inside `drain` where nothing deviceless could reach
  it; the fix was to extract `format_row` as a pure function of the drained words. Planted: print
  `0` instead of `-` for an unmeasured column — the exact shape of the bug that already occurred
  here, where `xn` was folded on MoE layers only and the dense rows' zeros read as
  "attention agreed".

  ```
  assertion `left == right` failed: a dense layer has no h   FAILED   (red)
  revert → 1 passed                                                  (green again)
  ```

  The fold words in the fixture are all NON-zero, so a `-` in the output can only come from the
  rule and never from the data; and the test closes by asserting that a fully-enabled row prints a
  hash in every column, without which a formatter that emitted `-` unconditionally would pass.

## 7. Muse Glimmer fp8 (added 2026-08-16, M11)

Five gates arrive with `--fp8`: three deviceless (two in `glimmer/geometry.rs`'s test
module, one in `crates/cli/tests/glimmer_convert.rs`), one shell (`tests/convert-parity-
glimmer-fp8.sh`), one on device (`crates/engine/tests/glimmer_fp8_decode.rs`). **The four
deviceless ones are PAID below; the device one is OWED and its recipe is written down.**

**Reverts were hand-edits, verified by `sha256sum -c`, not `git checkout`** — this branch's
work is uncommitted, so the restore command the sections above use would have discarded it.
The hash file is the evidence the tree came back byte-identical; every proof below was
followed by `sha256sum -c` reporting `OK` for the mutated file.

### 7a. `layer_bytes` charges the fp8 scale grid

Planted: the `ProjFmt::Fp8` arm reduced to `n` — the packed byte per weight with the f32
grid dropped, which is the size a reader who thought "fp8 is one byte" would write and
which under-reserves the tier by exactly what `DeviceTier::place` then bails on.

```
test glimmer::geometry::geometry_tests::layer_bytes_charges_each_format_its_own_size ... FAILED
assertion `left == right` failed
  left: 33792     (1_024 + 32_768,        the grid omitted)
 right: 33824     (1_024 + 32_768 + 8*4,  eight [1,1] grids)     (red)
revert → 8 passed                                                (green again)
```

### 7b. `ProjFmt::sniff` reads the stamp, never a compiled-in constant

Planted: `FormatMeta::load(dir).map(|m| m.fp8_block).unwrap_or(quant::FP8_BLOCK)` — the
plausible defect, since the default IS what the converter stamps today and the fallback is
invisible until someone converts at another block.

```
test glimmer::geometry::geometry_tests::sniff_reads_the_dtype_and_consults_the_manifest_only_for_fp8 ... FAILED
assertion failed: ProjFmt::sniff(&dir, &st).is_err()               (red — the stampless
                                                                    fp8 fixture was accepted)
revert → 8 passed                                                  (green again)
```

### 7c. The converter quantizes the projections, at the shipped block, and nothing else

Two mutations, each reddening a different assertion of the one test.

```
# (i) the wrong block constant — `add_quantized_fp8(&src, base, 64)`
assertion `left == right` failed: model.language_model.layers.0.mlp.gate_proj.weight_scale_inv: grid shape
  left: [2, 1]                                                     (red)
 right: [1, 1]
```

That is the fixture's discrimination claim paying out on exactly the tensor it names: `INTER`
= 96 against a 64 block gives `div_ceil(96, 64) == 2`. It also fixes the boundary the test's
own doc used to get wrong — the second grid row appears at any block **strictly below** 96,
not at 96, where `div_ceil(96, 96) == 1`.

```
# (ii) one projection class skipped — `&& !name.ends_with("mlp.down_proj.weight")`
Error: --fp8 quantized 28 tensors; this checkpoint's 4 layers imply 32   (red — the
                                                                          converter's own
                                                                          count `ensure!`)
revert (both) → 4 passed                                                 (green again)
```

### 7d. `tests/convert-parity-glimmer-fp8.sh`

Its header prescribed a red proof and claimed "the M11 record logs both runs" before any
record existed. Five runs, on six-file scratch directories holding one line each (the script
is a byte comparator; artifact-shaped content would not exercise anything more of it):

| run | input | result |
|---|---|---|
| green baseline | two distinct, identical dirs | `PARITY: every file byte-identical`, exit **0** |
| one byte flipped | `resident.safetensors` last byte `s`→`S` | `DIFFER` + `PARITY FAILED`, exit **1** |
| self-comparison | same dir twice | `REFUSED: both arguments resolve to …`, exit **66** |
| self-comparison, respelled | `dir` vs `dir/.` | same refusal, exit **66** — `realpath` earns its place |
| symlinked member | distinct dirs, `resident.safetensors` a symlink to the ref's | `REFUSED: resident.safetensors resolves to the same file in both directories`, exit **66** |
| missing member | `chat_template.jinja` absent from cand | `chat_template.jinja: MISSING (cand)` + `PARITY FAILED`, exit **1** |

The green baseline is listed deliberately: a gate proven only red is a gate that might be
unconditionally red, which is the same false evidence in the other direction.

### 7e. `glimmer_fp8_decode.rs` — OWED, device

The gate makes two claims and each needs its own mutation. Run under the flock,
`-- --test-threads=1 --nocapture`, dev profile; revert by hand-edit and `sha256sum -c`.

| # | mutation | must redden |
|---|---|---|
| 1 | `proj`'s `ProjPin::Fp8` arm dispatched to `launch_gemm_bf16` on `w.packed` cast to `*const u16` — the named silent-fallback failure, made real | the anti-fallback assert (`differing > 0`) |
| 2 | `Slot::fill`'s destination walk shortened by one (`.take(self.tensors.len() - 1)`) so the last tensor keeps layer 0's bytes | the P4 assert (`split.logits == all.logits`) |
| 3 | `quantize_artifact`'s grid census asserted against a fixture whose `inter` is under the block | the census assert — this one is a standing claim about coverage, not a defect |

Row 1 is the one that matters: a green anti-fallback assert is the ONLY structural evidence
that fp8 arithmetic ran, and it has never been red. Until it is, read a green on this file
as "the two artifacts differ", not as "the fp8 kernel is dispatched".

### 7e-bis. An operator false-green during the paying, and it is a NEW combination

Recorded on the same rule as §4's pair. Between two of the runs above, `cargo check
--workspace --all-targets` **failed** with

```
error: failed to run custom build command for `rivoli v0.1.0 (…/wave-m11/crates/cli)`
  process didn't exit successfully: … build-script-build (exit status: 101)
  --- stdout
  cargo:rerun-if-changed=…/wave-m19/crates
```

and the failure was invisible, because the command was `cargo check … | grep -E
'^(error|warning)' -A 5 | head -20; echo "CHECK OK"` — the grep matched, `head` swallowed the
rest, and the unconditional `echo` printed a green that nothing had established. **A commit
was made on it.** (It was sound; that is luck, not evidence.)

The cause is the fourth shared resource: `CARGO_TARGET_DIR` is one directory for every
worktree on this box, and `crates/cli/build.rs` bakes `env!("CARGO_MANIFEST_DIR")` at its own
compile time. A sibling worktree (`wave-m19`) compiled the build script last, so cargo reused
ITS binary, which scanned a `crates` directory this checkout does not have and panicked. The
jscpd gate did not run for that invocation at all.

**Two rules, both already in this repo's notes, that only bite together:** read the exit code
UNPIPED, and isolate `CARGO_TARGET_DIR` for any run whose green you intend to cite. Every gate
in §7 was re-run afterwards under `CARGO_TARGET_DIR=/var/db/rivoli/m11/verify-target` with the
exit code captured directly — deviceless suite exit 0 (**298 passed, 0 failed, 81 binaries**),
clippy exit 0 on both arms, `cargo fmt --check` exit 0, jscpd exit 0 (0 clones).

### 7f. `Slot::fill`'s third zip leg — RETIRED rather than proven

A gate that cannot exist needs no red proof, and this one stopped existing the same day it
was written. Kept as a record because the sequence is the lesson.

Correctness review found `Slot::fill`'s `.zip(&self.lens).zip(tails)` unguarded: `tails`
arrived as a parameter, a short one truncates a zip **silently**, and `Slot::new` asserted
only `addrs == lens`. The first fix was a runtime `ensure!` comparing the two lengths — a
real check, and one whose red proof needs a `DeviceTier` and therefore a contended GPU.

The shipped fix instead **deleted the parameter**: `Slot` now holds `Vec<(String, usize)>`,
name beside length, built by `Slot::new` from `layer_tails` and never handed in from
outside. `fill` zips two sequences that are both the slot's own and were built together, and
there is no third leg to be short. `GlimmerPin` lost its `tails` field with it.

**A guard you can delete beats a guard you must prove**, and on this box "must prove" meant
booking sole-tenant device time to demonstrate a failure the type system can refuse instead.
`Slot::new`'s `ensure!(addrs().len() == tensors.len())` **stays** and is not in the same
class: it compares two genuinely independent walks — the pin's field-by-field placement
against the config-driven tail list — which can diverge with nobody noticing.

## 6. Muse Glimmer chat framing (added 2026-08-17, M11b)

Until M11b every Glimmer prompt — bench and serve — was framed with **GLM's** chat template.
`glimmer_encoding.rs` had existed since the port and was wired to nothing; `main.rs`'s own
comment called it "a KNOWN GAP, not a decision" and said the change was owed an id-pinned
comparison first. This is that comparison.

### 6a. The id pin — PAID, deviceless

`crates/artifact/tests/glimmer_template.rs::rendered_prompts_tokenize_to_the_vendored_ids`
runs `render` → `Tokenizer::encode` → `case["ids"]` over all 31 vendored cases, on the
checkpoint's own 27 MB `tokenizer.json`.

```
RIVOLI_GLIMMER_ARTIFACT=/swarm/storage/ai/rivoli/glimmer-30b-fp8   cargo test -p rivoli-artifact --test glimmer_template --no-default-features -- --nocapture
  id pin: 31 cases tokenized identically to apply_chat_template
test result: ok. 5 passed                                                        (green)
```

**This closes a gap that had been recorded as owed and unclosable.** The sibling census test's dated
correction states the property — every special resolves to ONE id — is "true and UNVERIFIED",
and names the exact run that would close it, adding "the tiny fixture has none". A real
tokenizer is on this box now, and the run is above.

**Red proof.** Planted: `system_tail` closes with `<|eom|>` instead of `<|eot|>` — a one-token
change, and the one that decides whether a decode can STOP (only `<|eot|>` is a stop id).

```
case `plain_user` diverges at id 49 (got 57 ids, want 57)
  got  ...[392, 2540, 706, 392, 1556, 4205, 200007, 200022, …]
  want ...[392, 2540, 706, 392, 1556, 4205, 200008, 200022, …]      FAILED   (red)
revert (sha256 verified) → 5 passed                                          (green again)
```

**What the id half adds over the byte pin beside it**, since any `render` mutation reddens
both: the byte pin never calls a tokenizer, so it cannot see whether `<|start|>` became one
id or five ordinary pieces — that is decided by the `tokenizers` crate reading the
checkpoint's added-token table, outside this crate entirely. The reddening above shows ids
200007/200008 as SINGLE tokens, which is the property being pinned.

**Without `RIVOLI_GLIMMER_ARTIFACT` the test asserts its own reason** rather than returning
green having compared nothing: the 31 cases must still be present. An `eprintln!` skip would
be invisible under libtest capture, which is why "it skips loudly" is not a thing here.

### 6b. The serve door's six pure gates — PAID, deviceless

**The request half alone was written and then REVERTED.** Review found that framing a Glimmer
request with Glimmer's template while reading its reply back with GLM's would produce, for
every served reply, either an empty `content` (the GLM split hunts a `</think>` that is not
there and calls everything reasoning) or one leaking raw `to=user<|message|>` onto the user's
screen. That is a REGRESSION on a door that worked, and worse than the defect it was fixing.
**Both halves land together or neither does**, and `serve::split_channels`'s doc says so where
the next reader will be.

Both halves are pure functions of the request body and the generated text, so they live in
`serve/oai.rs` — the module whose header says it is where the pure functions and the tests are
— rather than beside their callers. (The first draft of this line justified that with "`frame_prompt`
takes a `&Ctx` holding `&mut Engine`" — false, and believing it is what left the dispatch
ungated; the real reason is that `frame_prompt` needs a 27 MB tokenizer.) Six gates, `cargo test -p rivoli --bin rivoli` — five
in `serve/glimmer.rs`, one in `serve/mod.rs`. **Each was planted, observed red, reverted, and
`sha256sum`-verified**, per this document's own standard for the word PAID:

| gate | planted defect → observed red |
|---|---|
| `a_developer_turn_is_framed_as_a_system_turn_and_not_dropped` | rewrite disabled → FAILED. Glimmer's role chain has no `else`, so an unmapped `developer` turn renders as NOTHING — instructions dropped behind a 200 |
| `the_reasoning_strength_maps_the_request_over_the_servers_default` | `if think { effort }` → `effort.or(…)` → FAILED. All five cells; **two are rivoli's invention** on a template with no thinking boolean, and an invented mapping with no gate is prose |
| `a_body_without_messages_is_refused` | `.context(…)?` → `.unwrap_or_default()` → FAILED. Otherwise `render` emits a system block and a bare generation prompt, and the model answers nobody |
| `split_glimmer_reads_the_recipient_and_never_swallows_a_reply` | two hardenings, quoted below. Eight cases, including the one that matters: text with no markers comes back WHOLE as content, because an empty `content` reads as a working server returning nothing |
| `every_prefix_of_a_generation_grows_both_channels_monotonically` | **needed no plant — it went red on the tree it was written for, twice.** See §6b-P0 |
| `the_reply_is_read_back_with_the_same_template_that_framed_it` (`serve/mod.rs`) | arms crossed → FAILED with `left: (" to=user<\|message\|>hi<\|eot\|>", "")` — content empty, the whole raw turn in reasoning. This is the DISPATCH, and until it existed both leaves were gated and the `match` pairing them was not |

#### 6b-P0. The streaming defect the monotonicity gate caught, and the one it caught next

**A P0 found by review before it shipped, then a second one found by the gate written for the
first.** Both were in `split_glimmer`, both streaming-only, and the non-streaming path was
correct throughout — so §6c's owed device recipe, a plain `curl`, would have come back green
over a broken door.

1. **The whole-text fallback fired on PREFIXES.** `render` ends the prompt at
   `<|start|>assistant`, so the first tokens a model emits are the turn header ` to=user` —
   several tokens before `<|message|>`. With no `<|message|>` yet, the fallback returned the
   raw header as content and `stream_decode` sent it. Then `<|message|>` arrived, content
   collapsed to the body, `strip_prefix(" to=user")` failed on it and on every token after, and
   `sent_c` froze. **Every streamed Glimmer reply would have been the literal text ` to=user`
   and nothing else, `finish_reason: "stop"`.** Fixed by `complete: bool`.
2. **A partial marker at the prefix boundary.** With (1) fixed, the new gate went red anyway:
   `prefix 36 reported content "The sky is blue.<", which the finished "The sky is blue." does
   not start with`. The `<` is three tokens short of `<|eot|>`; emitting it wedges the channel
   exactly as the header did. Fixed by `trim_partial_marker`, which is `delta`'s U+FFFD rule
   for markers instead of codepoints.

The gate is therefore its own red proof twice over, which is stronger evidence than a plant:
it caught duplication of a failure mode nobody had thought of yet. **Write the property, not
the cases** — a hand-written case list starts at a `<|message|>`, and both defects lived
strictly before one.

**Two operator false-greens while paying the dispatch proof, both the §4 trap again.** The
first mutation (`Arch::MuseGlimmer if false`) made the match non-exhaustive; the second
orphaned `complete`. Both were BUILD failures under `warnings = deny`, and both classifiers
read "no FAILED in the output" as green. The tell was a red-proof that refused to go red — the
signal §4 already names. A mutation for a red proof has to keep the tree COMPILING, which on a
`-D warnings` crate means keeping every binding live.

**`split_glimmer`'s two hardenings are red-proofed**, both planted, observed red, reverted,
sha256 verified identical:

```
# "earliest terminator wins" -> "try <|eot|>, else <|eom|>"
  left: ("", "A<|eom|>B")     right: ("", "A")                            FAILED  (red)
# `header.trim() == "to=self"` -> `header.contains("to=self")`
  left: ("CALL", "")          right: ("", "CALL")                         FAILED  (red)
revert both -> 1 passed                                                          (green again)
```

Read the left-hand sides: the first LEAKS `<|eom|>` into the text a user sees, which is the
whole class this function exists to stop; the second files a `to=selfcheck.run` tool call as
reasoning and drops it out of `content` entirely. Neither is hypothetical — both were the
naive spelling, and both were written that way first.

**Two scope lines are drawn in code and stated, not left to be discovered:**

- **`tools` is withheld from Glimmer's template.** It would render the ATEM
  `<atem:function_calls>` preamble while `oai::parse_tool_calls` reads GLM's `<tool_call>`
  markup and nothing else — so every tool use would return as prose with
  `finish_reason: "stop"`, a confidently wrong answer. A Glimmer tool request gets an honest
  untooled reply until an ATEM parser is ported.
- **DeepSeek-V4 still takes GLM's framing on the serve door**, and Kimi-K3's arm is
  unreachable (`main` refuses `--port` for it first). Both matches are EXHAUSTIVE, which is
  what makes the V4 gap visible instead of hidden behind an `if arch == MuseGlimmer` — the
  shape the first draft had, and the reason it was rejected.

### 6c. The serve round-trip — OWED, device

What §6b cannot reach: that a real decode through the real template TERMINATES. Needs a live
`Engine`, hence the GPU. Recipe, under the flock, dev profile:

```
rivoli /swarm/storage/ai/rivoli/glimmer-30b-fp8 --port 18174 &
curl -s localhost:18174/v1/chat/completions -d '{"model":"g","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

Green = a reply that TERMINATES on `<|eot|>` rather than running to `max_tokens`. Red proof:
point the arm at `encode_chat_turns` (the pre-M11b behaviour) and watch the reply run to the
limit — the old tree's 56-run retraction is that failure at scale.

**The second cell is the reply's SHAPE**: `content` non-empty, carrying no `<|message|>`,
`to=user` or `<|eot|>` fragment. That is the half §6b's `split_glimmer` gates on synthetic
text and only a real decode can gate on real output.

## 8. V4 scored-selection set gate (added 2026-08-16, M15, deviceless)

The M15 gate compares the engine's top-k (`v4::select::scored_rows`) against the frozen
oracle's own `Indexer.forward` exports on the toy (`index_topk = 2`, cap = 12 tokens, so
truncation is REACHED and counted — the "set-invariant goldens" trap the shipped 512 sets
below 2052 tokens). Seven proofs; the five executed ones were reverted and the tree re-run
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
6. **Prefill capture dropped from the oracle-side fixture** (`steps()` in
   `v4_indexer_goldens.rs`, standing in as the rename that would silently drop it): claim
   1 — "below the cap the indexer keeps every block it is offered", the premise the whole
   below-cap byte-identity argument rests on — examined ZERO rows while the existing
   `truncated_rows >= 5` floor was still met by the four decode steps alone, because every
   decode row here sits above the cap. Observed red only after adding the second floor:
   `only 0 non-empty below-cap rows — the prefill capture is gone and claim 1 examined
   nothing`. Reverted, 2 passed. **The lesson is the general one:** two claims on opposite
   sides of a boundary need two counters, because either side's examined-count can reach
   zero while the other's floor still passes.
7. **Per-row rectangle check removed** (`v4::select::assemble`, leaving the aggregate
   `rows * cols` total that stood alone before this proof): the compensating-ragged case
   added to `a_ragged_scored_row_is_refused_by_the_rectangle_check` — comp widths 2, 1, 3
   over a 3-row prefill, summing to exactly `3 * comp[0].len()` — was ACCEPTED as a valid
   `(3, 5)` rectangle. So the total was blind to raggedness whose entry count happens to
   balance, which `gather_scored` can be handed because its compressed rows come from a
   CALLER. The per-row width check is the fix and this is its proof; restored, 11 passed.
