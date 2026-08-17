---
status: data
scope: engine
verdict: Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN. M7's anchor-decode gate is proven red in BOTH halves — deviceless (an absent capture name, a tolerance under its envelope) and on device (all four recipe rows executed 2026-08-16 with observed magnitudes matching old:'s, plus two recorded operator false-greens whose lesson is part of the record). ADDED 2026-08-17: the GLM determinism gates (§5) — the id comparator, a prompt pinned by length+md5 that fired unplanted, INV-9 under a scan_free mutation, the probe's fold-slot layout under an inserted enum variant, and the column comparator's three refusals; the determinism gate's own 512-token GREEN is OWED and standing, because the engine is red. Also recorded there: a false EXCLUSION found in this round's own instrument — a fold omitted on dense layers made two runs 'agree' about a quantity neither measured, which no gate in this repo could see. ADDED 2026-08-17 (Phase 1): the per-fold probe flags — parse refusals, spin_rows reading nothing, and the column comparator refusing two logs from different fold configurations; the dash-for-disabled rendering was recorded OWED and PAID the same day, red-proofed by printing 0 instead.
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

## 5. The GLM determinism gates (added 2026-08-17, `wave/fix-glm-determinism`)

Five checks landed with the divergence investigation. Each was planted, observed red, reverted,
observed green again — and one of them reddened *by itself*, which is recorded because a proof that
fires unbidden is worth more than one you had to arrange.

**5a. `tests/determinism-glm.sh --self-test` (the id comparator).** Two id streams differing by one
token, and a truncated stream. Both must compare unequal. The truncated case is the one that
matters: a naive `diff <(sort)`-style comparator passes it, and that substitution has been made
before.

```
FAIL: the comparator did NOT redden on differing id streams  → exit 2   (red, on a planted no-op comparator)
FAIL: the comparator did NOT redden on a TRUNCATED stream    → exit 2   (red)
SELF-TEST ok ...                                                        (green)
```

**5b. The gate's recorded prompt, pinned by length + md5.** This one **fired on its own**: the pin
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

**5c. INV-9 (`inv_9_a_slot_is_not_reissued_until_its_copy_lands`).** Planted: `scan_free` ignores
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

**5d. The divergence probe's fold-slot layout — RED-PROVED, THEN THE TEST WAS DELETED, and the
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

**5f. The length classifier (`classify_lengths`).** Added because review pointed out the branch was
reachable only after two real 512-token GPU arms and so could never be shown red — and because it is
the code that had just been found WRONG (it filed a divergence as a setup error). Factored out and
driven from `--self-test`; planted: the unequal-length branch returns 2 (setup) instead of 1 (RED).

```
FAIL: unequal lengths must be RED (1), not a setup error   exit 2   (red)
revert → SELF-TEST ok                                      exit 0   (green again)
```

Exit codes read UNPIPED both times. Piped through `| tail` the mutated run reports 0, which is the
trap this file already records twice.

**5e. `tests/divergence-columns.sh`'s three refusals.** All observed on real inputs:

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
is red: the first device pair diverged. That is the live red proof for the gate as a whole (§5a's
comparator is only the mechanism), and the green is owed once the defect is fixed. Recorded as owed
rather than claimed — this section exists so that distinction survives.

**One false-green caught in this round's own instrument, and it is the lesson.** The first version
of the probe folded the MoE's input `xn` on MoE layers only. GLM has three dense layers, so their
`xn` column stayed 0 in both runs — and a diff reads two equal zeros as "attention agreed" when
nothing was measured. A false EXCLUSION is the one failure mode an instrument may not have, and it
is invisible to every gate in this repo because the instrument was producing well-formed output the
whole time. Every column that cannot exist for a row now prints `-`, never 0.

**5g. The per-fold probe flags (added 2026-08-17, Phase 1).** Four checkable surfaces landed with
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
