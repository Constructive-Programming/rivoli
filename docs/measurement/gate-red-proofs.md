---
status: data
scope: engine
verdict: THE RULE THIS FILE NOW OPENS WITH, extracted from the M17 battery and generalising past this repo: a plant must be observed to have CHANGED THE TREE, and to have reddened THE ASSERTION UNDER TEST, before its result is evidence -- three of seventeen M17 plants produced a result meaning something other than it looked like (one reddened the BUILD via warnings = deny with no test executed, one reddened an EARLIER assert in the same body, and one NEVER APPLIED because rustfmt had reflowed the target line, so the suite ran green on an unmodified tree and reported exit=0 with everything passing, which is indistinguishable from a plant that failed to redden). Failed plants are kept rather than deleted, because a proof that missed is evidence about the harness. Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN. M7's anchor-decode gate is proven red in BOTH halves — deviceless (an absent capture name, a tolerance under its envelope) and on device (all four recipe rows executed 2026-08-16 with observed magnitudes matching old:'s, plus two recorded operator false-greens whose lesson is part of the record). M17a's DFlash drafter oracle is registered here and recorded in glimmer-reference/anchor.md: nine deviceless plants, all re-run after the 2026-08-16 fixture re-vendor because a re-measured floor invalidates its old red proof, with the block-for-window substitution going from 1 of 10 tests red to 3; two rows the rewrite had DELETED rather than corrected -- the prefix-filter proof and the +1-ulp detection floor -- were restored and independently re-run 2026-08-17, both reproducing their pre-re-vendor results exactly, with their observed values kept in anchor.md alone. The restored prefix-filter row also carried a wrong MECHANISM (labelled salt pairing, when the planted rename cannot reach the salt assert -- dropping the LAST draft golden leaves the first correctly paired and the CENSUS assert is what reddens), and running the mirror plant to find what does exercise the pairing guard added a tenth plant: two independent guards, one plant each, both 7 of 10 red. M17b's drafter converter gate is proven red in all twelve of its tests AS FIRST LANDED by six plants, with two rows carrying the argument: a one-byte edit to the vendored checkpoint header reddens the LIVE-file comparison, which is what proves that conditional half is armed on this box rather than silently skipping, and a wrong drafter hidden_size reddens the POSITIVE pairing arm, which is what makes "the shipped drafter pairs with the shipped target" evidence rather than a run that failed early for an unrelated reason; the two refusal-only plants show each refusal arm reddening both when its guard is deleted and when its wording drifts, which a status-code-only refusal test cannot see. GREW the same day under review to 13 tests and NINE plants: the per-token budgets (drafter KV 20,480 B/token, hidden-state export 66,560 B/token, a 260.0 MiB ring at ctx 4096) were prose in a doc and gated nowhere and are now derived from the shipped config; the pin test was strengthened to derive its 36/22 verbatim-widened split from the real HEADER as well as from the census, red-proofed by a rank change that preserves every byte total; and RIVOLI_DRAFTER_CKPT_REQUIRED was added on RIVOLI_CS_REQUIRED's precedent because stating that the live comparison degrades when the mount vanishes is not the same as enforcing it -- proven in all THREE states, including the one people skip, the real path WITH the variable set, since a required mode never run in its required state is a mechanism rather than a gate. GREW AGAIN to 14 tests and ELEVEN plants with the mask-indexing gate, whose first plant is recorded as a FAILED proof: removing q_offset's use left the parameter unread, warnings = deny turned that into a compile error, and the run exited 101 without executing a single test -- an exit code alone would have read as "reddened". The replacement plant passes 0 at the CALL SITE, which is the mistake a kernel author actually makes, compiles, and reddens the gate with left: 0 right: 256. THREE MORE PLANTS (fifteen in all) close a discarded-count gap review found in the same gate -- the cache branch's strictly-bidirectional count was captured into `_` and thrown away while the commit message cited it, a guard that cannot fire -- and proving the new assertion red took all three attempts, because an assert earlier in the same test body hides every later one: two plants exited 101 with the test named while never touching the line under test, and only conditioning on q_offset separated the two fixture calls. A plant that reddens the TEST is not a plant that reddens the ASSERTION you added; read left/right, not the exit code. TWO MORE GATES AND TWO MORE PLANTS (15 tests, 17 plants) cover a SECOND fixture blindness: the reference's causal overlay has a STRICT lower edge and its bidirectional overlay an INCLUSIVE one, one row apart, and attn.hip::gqa_attend -- which already exists for Glimmer with the right grouping, layout and a start_pos that IS the q_offset -- implements the causal form with lo = pos - win + 1 (its own comment calls that off-by-one trap 14). Copying it and widening the bound drops exactly one KV row per query, 16 at the shipped geometry, and the vendored fixture changes on 0 of its 4 rows because at ctx 12 window 13 the lower edge clamps to 0 either way -- so the gate asserts the shipped arithmetic AND the blindness. A THIRD failed plant is recorded with them: rustfmt had reflowed the target line, the substitution matched nothing, and the suite ran green on an unmodified tree, reporting exit=0 15 passed -- indistinguishable from a plant that failed to redden. The invariant covering all three failures is that a plant must be observed to have CHANGED THE TREE before its result is evidence. SECTION 7 adds M17c's block-attend kernel: its census DEFERRED row is red-proofed on all three edges (row deleted while the gap exists; launcher renamed so the row names something absent; a test made to cover it, retiring the deferral), and a removed semicolon takes the BUILD to exit 101 -- which had to be run rather than assumed, because .jscpd.json is format:["rust"] so the duplication gate never scans .hip, and build.rs skips objects newer than their source so a fast rebuild proves nothing about whether hipcc ran. The kernel compiles for gfx1151 and has NEVER EXECUTED; the three defects its header names are invisible to a compiler and, by measurement, to the vendored goldens as well. Its duplication against gqa_attend is recorded as OWED rather than clean, because no gate can raise it and the factoring would refactor the GPU-parity-gated live attend path on a change this agent cannot verify. A TWELFTH plant guards the per-token budget gate's two dtype columns: the plan's three budgets are bf16 figures against an f32 engine (scratch is DeviceBuf::new(n*4), kv_bytes ends in checked_mul(4)), so the ctx-4096 export ring is 520.0 MiB and not the plan's 260.0 unless the export narrows -- writing the bf16 constant into the f32 row reddens with left: 545259520 right: 272629760.
---

# M0 gate red proofs

Run 2026-08-15 on `rw/main` at commit `e922a34`, per P7: a gate that has never been red is
not evidence. Each proof was planted, observed red, reverted, and observed green again in
the same session. The exact reddening output is quoted; everything else is reproduction.

## THE RULE, before any of the sections below

**A plant must be observed to have CHANGED THE TREE, and to have reddened THE ASSERTION UNDER TEST,
before its result is evidence.**

Extracted from the M17 battery (§6), where **three of seventeen plants produced a result that meant
something other than what it looked like** — and in every case the exit code and the test name were
exactly what a working proof produces:

| how a plant lies | the instance | what it looked like |
|---|---|---|
| it reddens the **build**, not the gate | removing a parameter's only use tripped `warnings = deny` | `exit=101`, no test executed |
| it reddens **another assertion** in the same test | an earlier `assert` in the body fires first and masks the rest | `exit=101`, correct test named, wrong line |
| it **never applied** | `rustfmt` had reflowed the target line, so the substitution matched nothing | `exit=0`, suite green **on an unmodified tree** |

The third is the dangerous one: a green run on an unmodified tree is indistinguishable from "the
plant failed to redden the gate", and both look like a finding. The practices that separate them,
cheapest first:

1. **Confirm the edit landed** — a substitution that asserts on its own match count, or a `git
   diff --stat` before the run. Never a silent `replace`.
2. **Read the failure's `left`/`right`**, not the exit code and not the test name. Name the
   assertion you intend to redden, then confirm the message is that one's.
3. **Isolate when an earlier assert can mask** — condition the plant on the specific case (here, on
   `q_offset`) rather than on something the earlier assertion also sees.
4. **Never let a pipeline eat the status** — `| tail`, `| grep`, and `echo $?`-after-a-pipeline all
   report the last stage, which is why §4's two operator false-greens went unnoticed.

Failed plants are **kept in the tables below rather than deleted**. A proof that missed is evidence
about the harness, and this section exists because that evidence turned out to be worth more than
the proofs that landed first time.

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

## 5. The DFlash drafter oracle (added 2026-08-16, M17a; re-run 2026-08-17)

`crates/oracles/tests/glimmer_draft_oracle.rs`. **Deviceless, and the battery itself lives in
`docs/measurement/glimmer-reference/anchor.md`** §The red-proof battery — kept there rather than
copied here because every row is stated next to the floor it is measured against, and a second
frozen copy of eight numbers agreeing with the first is not a check. This section is the registry
entry; that section is the record.

**Ten** plants, each applied, run, and reverted (nine as of 2026-08-16; the tenth is the mirror
prefix-filter plant below). All were re-run after the 2026-08-16 fixture
re-vendor (`sliding_window` 4 → 13), because **a re-measured floor invalidates its old red
proof.** Two rows changed meaning under the re-vendor and that change IS the finding:
`mask()` reading `block` instead of `window` went from reddening 1 of 10 tests to 3, and a mask
cell flipped inside the block went from inexpressible to reddening 3.

Two rows were deleted from the doc rather than corrected when that section was rewritten — the
prefix-filter proof and the `+1 ulp` detection floor — and were **restored and independently
re-run 2026-08-17**, both reproducing their pre-re-vendor results exactly. Their observed values
are in anchor.md §the red-proof battery and §the ladder, and are deliberately **not** repeated
here: by this section's own rule, a second frozen copy would be a number to drift rather than a
check.

**The restored prefix-filter row also carried a wrong MECHANISM, and finding that added the tenth
plant.** It was labelled "salt pairing"; review traced the positional zip and showed the planted
rename cannot reach the salt assert at all, because dropping the last draft golden leaves the
first paired with the first — correctly — and the census assert is what reddens. The mirror plant
(rename the FIRST golden out) was then run to find what does exercise the pairing guard, and it
does. Two independent guards, one plant each, both recorded against the line they fire on in
anchor.md. **The count was right and the mechanism was not**, which is the failure mode a battery
of counts cannot catch on its own.

The ulp row is a **detection-floor** record, not a passing gate: it bounds what the oracle's green
is evidence for. This repo has twice registered a red-proof whose perturbation was below the
detection floor and read the resulting green as coverage (the parity 1-ulp and single-sign-flip
rungs, `CLAUDE.md` §Gates → parity), which is why the floor is measured and written down.

## 6. The DFlash drafter converter gate (added 2026-08-17, M17b)

`crates/cli/tests/drafter_convert.rs`, **15** tests, deviceless. **All fifteen are proven red by
seventeen plants**, each applied, run, and reverted; the suite is 15/15 green before and after. The
measurements they gate are in `glimmer-reference/drafter-checkpoint.md`.

> **GREW 2026-08-17, same day, from 12 tests and six plants** — review added the thirteenth test
> (the per-token budgets, which were prose in the doc and gated nowhere) and three plants: one for
> an assertion strengthened in review, and two for the new test and the new required mode.

| # | plant | reddens |
|---|---|---|
| P1 | `norm.weight` → `nprm.weight` inside the vendored header (same length, still valid JSON) | **3** — the header FNV pin, the census set equality, and the **live-file comparison** |
| P2 | `TENSOR_BYTES` 5,111,970,304 → ...306 | **3** — the offsets-tile-the-file check, the resident pin, and the params×2 identity |
| P3 | `target_layer_ids` 5 entries → 4 in the vendored config | **3** — the `encoder.fc` 5×hidden concat, the census, the shipped-config facts |
| P4 | `hidden_size` 6656 → 6144 in the vendored config | **7** — including the POSITIVE pairing arm |
| P5 | the converter's `hidden_size` cross-check disabled | **1** — the hidden-width refusal, and only it |
| P6 | the not-an-artifact refusal reworded | **1** — the attach refusal, and only it |
| P7 | `norm.weight` retyped `[6656]` → `[6656, 1]` in the vendored header — same element count, same byte span | **5** — including the pin test's new file-derived rank check |
| P8 | `num_key_value_heads` 8 → 4 in the vendored config | **3** — the per-token budgets, the census, the shipped-config facts |
| P9 | `CKPT_DIR` pointed at an absent path, run in **both** modes | the required mode, in all three states |
| P10 | `mask_shape` made to ignore `q_offset` | **reddened the BUILD, not the gate** — see below |
| P10b | the serving call site passes `0` for `q_offset` | **1** — the mask-indexing gate, `left: 0, right: 256` |
| P11 | `sliding_window` 2048 → 8 in the vendored config | **2** — the mask-indexing gate and the shipped-config facts |
| P12 | the f32 ring row written with the bf16 constant | **1** — the per-token budget gate, `left: 545259520, right: 272629760` |
| P13 | `strict` broken globally (`kv - ctx > row + 1`) | the mask gate — at the SHIPPED assert, `left: 105, right: 120` |
| P14 | `strict` broken for `ctx == 12, row == 0` | the mask gate — at the NO-CACHE fixture assert, `left: (13, 2, 0)` |
| P14b | `strict` broken for `q_offset == 12, row == 0` | the mask gate — at the **cache-branch** assert, `left: (16, 3, 3), right: (16, 6, 3)` |
| P15 | both lower edges made strict | *(did not apply — rustfmt had reflowed the target line, so the run was on a CLEAN tree and its green was not a proof)* |
| P15b | the same, against the reflowed text | **1** — the lower-edge gate, `left: 2063, right: 2064` |
| P16 | the fixture's window 13 → 2 | **1** — the blindness assert, `left: [0,1,2,3,4], right: [1,2,3,4]` |

Every one of the fifteen is reddened by at least one plant. **Two of these rows carry the argument:**

**P1 reddens `the_vendored_header_is_the_live_checkpoints_own_bytes`.** That test is this suite's
one conditional half — with the checkpoint unmounted it has nothing to compare and degrades to the
vendored-only checks. P1 is therefore not just a proof that the comparison works; it is the proof
that the comparison **is armed on this box right now** rather than silently skipping. A suite whose
conditional half had quietly gone inert would show exactly the same all-green result.

**P4 reddens `the_shipped_drafter_pairs_with_the_shipped_target`.** That arm asserts a refusal
naming the ABSENT WEIGHTS, which the run can only reach by passing all three of the converter's
pairing cross-checks — so the refusal it dies on is the evidence that the shipped drafter and the
shipped target pair. Without P4 that arm could be passing because the run failed early for its own
unrelated reason. P4 makes the positive arm evidence rather than a coincidence.

**P7 is the plant that had to preserve the byte totals to be worth running.** The pin test was
strengthened in review 2026-08-17: it derived the 36-verbatim/22-widened split from the config
only, in a file whose whole premise is "against the real checkpoint", so it never touched a
checkpoint byte. It now computes the same split from the real header's own ranks and compares.
Proving that live needed a plant that changes a RANK without changing anything else — retyping
`norm.weight` from `[6656]` to `[6656, 1]` keeps the element count, keeps `elems * 2` bytes, and
keeps every offset — and the new assertion is the one that catches it.

**P9 is the required-mode proof, and it needs all three states to mean anything.**
`RIVOLI_DRAFTER_CKPT_REQUIRED` exists because P7 above only proves the live-header comparison was
armed *at the instant it was planted*; P7's honesty about the mount does not enforce it, and P7
says so. Following `RIVOLI_CS_REQUIRED`'s precedent exactly:

| `CKPT_DIR` | variable | result |
|---|---|---|
| absent path | unset | **passes** — the honest optional case, and the suite's other twelve still examine the vendored bytes |
| absent path | set | **panics**: `RIVOLI_DRAFTER_CKPT_REQUIRED is set but /…/no-such-drafter/model.safetensors is absent — the live-header comparison examined nothing` |
| the real path | set | **13/13 green** — which is the state that proves the required mode is satisfiable here rather than aspirational |

The third row is the one people skip. A required mode nobody has run in its required state is a
mechanism, not a gate.

**P10 IS RECORDED AS A FAILED PROOF, because that is what it was.** Deleting `q_offset`'s use
inside `mask_shape` left the parameter unread, `[workspace.lints.rust] warnings = deny` turned that
into `error: unused variable`, and the run exited 101 **without executing a single test**. An exit
code alone would have read as "the plant reddened"; reading WHICH failure it was showed the gate
had not run at all. `CLAUDE.md` names this exact trap twice — a red-proof that refuses to go red is
evidence about the harness, and `| tail`/`echo $?` after a pipeline eats the distinction. P10b is
the replacement, and it is the better plant anyway: passing `0` at the **call site** is precisely
the mistake a kernel author makes when they forget to plumb the offset, it compiles, and the gate
catches it with `left: 0, right: 256`.

**P12 exists because the budgets were bf16 figures against an f32 engine.** The plan's three
per-token numbers price 2 bytes per element — the checkpoint's dtype — while `glimmer::pin::scratch`
is `DeviceBuf::new(n * 4)` and `geometry::kv_bytes` ends in `checked_mul(4)`. The gate now asserts
BOTH columns, and P12 is the plant that proves the f32 column is not just the bf16 one relabelled:
writing 260.0 MiB where 520.0 belongs reddens with `left: 545259520, right: 272629760`. Under P5 a
budget that does not name its dtype is not yet a budget, and a gate that asserted only one column
would have let the ring be sized at half.

**P13, P14 AND P14b ARE ONE PROOF THAT TOOK THREE ATTEMPTS, AND THE REASON IS WORTH MORE THAN THE
PROOF.** Review found that the mask gate captured the cache branch's strictly-bidirectional count
into `_` and threw it away, while the commit message cited the number — a guard that cannot fire,
which is a class this repo keeps rediscovering. The count is now asserted. Proving *that assertion*
red then took three plants, because **a plant that reddens the test is not a plant that reddens the
assertion you added**:

| plant | intended target | what actually fired |
|---|---|---|
| P13 | the new cache-branch count | the SHIPPED-widths count — it asserts first and masked the rest |
| P14 | the new cache-branch count | the NO-CACHE fixture count — `ctx == 12` matches BOTH fixture calls |
| P14b | the new cache-branch count | **it**, at last — conditioning on `q_offset == 12` is what separates the two fixture calls |

An `assert` earlier in the same test body hides every later one, so "the test went red" is not
evidence about the line you are trying to gate. The discipline that falls out: **name the assertion,
then read the failure's `left`/`right` to confirm it is that one** — P13 and P14 both exited 101 with
the test named, and neither touched the assertion under test. Kept as failed attempts rather than
deleted, on the same argument as P10 above.

**P15/P15b/P16 gate a SECOND fixture blindness, and P15 is a third recorded failed plant.** The
reference's causal overlay has a STRICT lower edge (`kv_idx > q_idx - window`) and its bidirectional
overlay an INCLUSIVE one (`abs(q_idx - kv_idx) <= window`) — one row apart. `attn.hip::gqa_attend`
implements the causal form, computes `lo = pos - win + 1`, and its own comment names that off-by-one
"trap 14"; carrying that `lo` into a bidirectional kernel drops exactly one KV row per query, 16 at
the shipped geometry. **The vendored fixture cannot see it: 0 of its 4 query rows change**, because
at ctx 12 with window 13 the lower edge is negative for every row and clamps to 0 either way. So the
gate holds the shipped arithmetic AND asserts the blindness itself, with the note that if the two
edges ever differ at the fixture geometry the fixture gained the power and the test is what to
delete — P16 forces exactly that by shrinking the window to 2.

**P15 did not apply**: rustfmt had reflowed `let lo = if …` across four lines, the substitution
matched nothing, the python `assert` fired, and the suite then ran GREEN on an unmodified tree.
Reported as `exit=0, 15 passed` — which is indistinguishable from "the plant failed to redden the
gate" if you only read the counts. Third recorded instance in this file of a plant whose green meant
something other than what it looked like (P10 reddened the build; P13/P14 reddened the wrong
assert). **The invariant that catches all three: a plant must be observed to have CHANGED THE TREE
before its result is evidence.** P15b is the same plant against the reflowed text.

**P5 and P6 are the pair that keeps the refusal arms honest.** Each refusal test asserts a MESSAGE,
not merely a non-zero exit, so it reddens both when the guard is **deleted** (P5) and when its
wording **drifts** (P6). The second failure mode is the one a status-code-only refusal test cannot
see, and this repo's harness (`common::expect_refusal`) exists because of it.

## 7. The DFlash block-attend kernel and its census row (added 2026-08-17, M17c)

`crates/backend/kernels/attn.hip::gqa_block_attend` + `launch_gqa_block_attend`
(`crates/backend/src/hip_attn.rs`). **The kernel has never executed** — that is the DEFERRED row's
whole content, and these four proofs are what can be established without a device.

| # | plant | result |
|---|---|---|
| P17 | the `DEFERRED` row deleted while the launcher has no oracle | red — `1 kernel(s) have a launcher … and NO oracle …: gqa_block_attend` |
| P18 | the launcher renamed so the row names something absent | red — `DEFERRED names launch_gqa_block_attend, which is not a launcher` |
| P19 | a test made to "cover" it (`launch_gqa_block_attend(` in a test file) | red — `the deferral has outlived its reason; delete the row` |
| P20 | a semicolon removed inside the new kernel | **the BUILD fails, exit 101** |

**P17/P18/P19 exercise all three edges of the census at once**, which is what "checked at both ends"
buys: the row cannot be dropped while the gap exists, cannot name a launcher that is gone, and
cannot outlive the arrival of an oracle. A deferral is the only honest state for a kernel that has
not run, and it is the state that expires by itself.

**P20 is the one that had to be run rather than assumed.** A `.hip` file is text until something
compiles it, and two facts made "it compiles" a claim worth testing rather than believing:
`.jscpd.json` is `format: ["rust"]`, so **the duplication gate does not scan `.hip` at all**, and
`crates/backend/build.rs` skips any object newer than its source, so a rebuild reporting
`Finished in 1.28s` proves nothing about whether `hipcc` ran. Removing one semicolon takes the
build to exit 101; restoring it builds clean. So the kernel is syntactically valid HIP, compiles
for gfx1151, and is linked — and none of that says it computes the right thing.

**What P20 does NOT establish, stated because a compiling kernel reads as a working one:** no
value has been scored. The three defects the kernel's own header names — the `q_offset` branch, the
inclusive-vs-strict lower edge, and the target's `qk_scale_factor` leaking in — are invisible to a
compiler and, by measurement, invisible to the vendored goldens too (§`drafter_convert.rs`, and
`glimmer-reference/drafter-checkpoint.md`). The oracle scoring is the GPU work; the deviceless
geometry gates are what stand in for it until then.

**One duplication note, since no gate will raise it.** `gqa_block_attend` shares its tile loop and
online-softmax body with `gqa_attend` and differs in the bound, the absent ring, and `kv_len`.
jscpd cannot see it (`.hip` is unscanned) and `hip_attn.rs`'s macro region is exempt by an ignore
marker whose own argument is that *review* is the duplication gate there. So this is recorded as
**owed** rather than clean: the factoring is a shared `__device__` inline taking `(lo, hi)`, which
would put the causal-versus-bidirectional edge distinction in one place instead of two comments.
It was not done now because it would refactor the live Glimmer attend path — GPU-parity-gated,
`tests/glimmer_attend.rs`'s measured 6.56e-7 residual — on a change this agent cannot verify. Do it
once both kernels are device-green, not before.
