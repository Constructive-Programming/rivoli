---
name: rivoli-docs
description: Maintain the rivoli engine's docs/ tree — the 00-orientation / reference / measurement / investigations layout, the status+verdict front matter, and the INDEX.md that tests/docs.rs checks against it. Use when adding or editing anything under docs/, closing an investigation, changing what a doc claims, moving a doc between live and closed, or when tests/docs.rs fails.
---

# Maintaining `docs/`

The layout exists because this repo's recurring failure is **stale prose nobody can tell is
stale**: `PERF.md` called the engine compute-bound for weeks after the measurement that
inverted it, `CACHE_PILOT.md`'s header outlived its code, `quant.rs` cited a `docs/int3.md`
that was never committed. So the shelf a doc sits on states whether to trust it, and a test
enforces that the claim on the shelf matches the claim in the file.

## The contract

Every `.md` under `docs/` starts with:

```markdown
---
status: live | closed-negative | closed-shipped | closed-mixed | data
verdict: One line. What this settles, with the number. ≥20 chars.
---
```

`docs/00-orientation/INDEX.md` lists the file with **the same verdict, verbatim**.
`tests/docs.rs` fails if either half is missing or the two disagree.

**The front matter is the source; the index mirrors it.** Do not paraphrase into the index —
that is exactly what drifted nine times on this test's first run. Copy the string. Markdown
emphasis (`**5.120**` vs `5.120`) is allowed to differ; the claim is not.

## Which status, and therefore which directory

| status | means | lives in |
|---|---|---|
| `live` | true about the engine today; a wrong one is a defect | `reference/` or `measurement/` |
| `closed-negative` | tried, measured, rejected — the code is gone | `investigations/` |
| `closed-shipped` | tried, measured, kept — the result is in the engine | `investigations/` |
| `closed-mixed` | parts of both; the file carries a banner saying which | `investigations/` |
| `data` | measurements or instruments, not prose | `measurement/` |

**When status changes, the file moves.** `git mv` it, fix the index section, update inbound
references. That move is the signal a reader relies on; a dead mechanism still sitting under
`reference/` is the exact failure this layout was built to stop.

## Recipes

**Adding a doc.** Put it in the directory its status implies. Write the front matter first —
if you cannot state the verdict in one line, the doc does not yet know what it is about. Add
the row to the matching INDEX.md table. Run `cargo test --release --features rocm --test docs`.

**Changing what a doc claims.** Edit the body, update `verdict:`, mirror it into INDEX.md.
The test tells you which of the two you forgot. If the *status* changed too, move the file.

**Closing an investigation.** Set `status:` to the outcome, rewrite `verdict:` to lead with
the result and the number that carries it (`"…recovers 0.09% against a 2% bar"`, not
`"…was investigated"`). `git mv` into `investigations/`. Name the file for the *question*,
not the mechanism: `codebook-rotation.md`, not `hadamard.md`.

**Correcting something that was wrong.** Correct **in place with a dated note** — never
delete, never silently overwrite. What an investigation ruled out is worth as much as what
it found, and erasing the prior belief makes the next reader repeat the experiment. Say what
was believed, what is true, and how the error happened:

```markdown
> **CORRECTED 2026-08-01.** This said the fp8 set was deleted. It is at /swarm/storage/…;
> the error came from reading /var/db/rivoli and inferring rather than looking.
```

**A doc has gone half-live.** Split it along the seam rather than banner the whole thing.
`PERF.md` became `measurement/perf-roadmap.md` (live table), `measurement/how-to-measure.md`
(method, which never goes stale) and `investigations/perf-evidence.md` (the evidence, with
the stale block bannered). The method half had been 39 KB deep where nobody found it.

## Rules that predate the structure and still bind

- **Anything over ~20 KB opens with a STATE block** giving the current answer in ~15 lines.
  A reader should be able to stop there.
- **`reference/architecture.md` is the one doc meant to be read whole. Do not split it.**
  Its §8b is a registry `tests/invariants.rs` parses — a documented INV-*n* with no
  `inv_n_*` test, or the reverse, is a failing test. Never add one without the other, and
  do not write a literal `INV-<digit>` token into §8b prose for an invariant that no longer
  exists: the parser reads it as a live claim.
- **`measurement/benchmarks.md` was append-only until 2026-08-10 and is NOT any more.** It had
  grown to 4,070 lines of journal; it is now the verdict of each round. Record a new result as
  a short section, and when one is superseded REPLACE it and say so. Two things are not yours
  to drop: the **section titles** (155 inbound citations resolve by name) and the
  **reproducibility artefacts** — the canonical 218-token prompt, the reply md5 gates, kernel
  fingerprints, the recorded command forms. Git history holds the journal; `investigations/`
  holds the arguments. **Both halves now have a test** (`tests/docs.rs`):
  `benchmarks_citations_resolve` fails on a quoted name no longer in the file, and
  `benchmarks_stays_compact` caps the line count. Do not trust this bullet — it was prose
  alone for one day, and in that day a compaction dropped four cited anchors while two
  hand-verification passes reported zero.
- **Verdicts are for ruling files out.** The reader's best outcome is answering their
  question from the index and never opening anything.

## Checks

```bash
cargo test --release --features rocm --test docs        # front matter + index agreement
cargo test --release --features rocm --test invariants  # the INV-n registry
grep -rn "docs/[A-Za-z0-9_]*\.md" --include=*.rs .      # inbound refs, after any move
```

**What the test cannot check is whether a verdict is TRUE.** It checks one exists, is
classified, and says the same thing in both places — which turns silent drift into a visible
edit. Truth is still on you: if you change behaviour, go find the doc that describes it.
