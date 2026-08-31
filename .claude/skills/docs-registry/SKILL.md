---
name: docs-registry
description: How rivoli's docs/ registry works — status/scope/verdict front matter, the one-row-per-doc INDEX agreement, corrections in place, and superseded docs moving rather than dying. Invoke whenever you add a file under docs/, change a doc's verdict or status, correct a claim in a doc, retire or supersede one, or touch docs/00-orientation/INDEX.md.
---

# The docs registry

`docs/00-orientation/INDEX.md` is what a reader consults to decide **what NOT to open**.
The registry gate (`crates/cli/tests/docs.rs`) enforces that the index and the docs agree.
It cannot check whether a verdict is *true* — only that one exists, is well-formed, and is
the same in both places. That is why the discipline below is on you, not on the gate.

## Front matter — three keys, on every file under `docs/`

```yaml
status: live
scope: k3
verdict: the KDA anchor is vendored and the gate reads it deviceless; MLA stays exact-only
```

- **`status:`** — one of the `STATUSES` array in `crates/cli/tests/docs.rs`.
- **`scope:`** — one of the `SCOPES` array in the same file. It answers *whose evidence
  backs this verdict*.
- **`verdict:`** — the claim, in prose, long enough to rule the file out for a reader who
  will not open it (the gate rejects a verdict that is too short to do that).

**Read those two arrays in `crates/cli/tests/docs.rs` before you pick a value.** They are
the source of truth and they grow; anything this skill's prose might name is a snapshot,
and a snapshot is exactly the inherited number the house forbids.

`scope:` exists because **a closed verdict rules its question out only for its scope.**
The old tree's `npu-offload.md` carried a `closed-negative` measured on GLM-5.2 alone and
was relayed for months as if it were engine-wide. Never restate a closed verdict without
its scope, and never widen one without new evidence under the wider scope.

## INDEX agreement — exactly one row per doc

- Every doc gets **exactly ONE** INDEX row, matched by its **link target**, not by its
  title. Two rows is how a stale verdict survives: the old tree's `k3-port.md` had two
  rows carrying *different* verdicts and the gate was blind between them. A "mechanical"
  merge resolution has duplicated a row here before — after any rebase that touches
  INDEX.md, grep the per-doc rows.
- The row's **verdict cell must contain the doc's front-matter verdict** after
  normalization (markdown emphasis and link syntax dropped, whitespace collapsed). The
  check is row-scoped: a whole-index `contains` once let one doc's stale verdict be
  satisfied by *another* doc's row text.
- The row's **scope cell** is read from that same row, so a doc linked only from prose
  matches no `| [` table row and **nothing checks its scope** — that is itself a finding.
  Link from the table.
- **`TOUR.md` is the one exception**: it is linked from a prose row rather than the
  verdict table, and its verdict is its own text.

## Changing a doc

- **Corrections go in place, with a dated note.** Do not rewrite history silently and do
  not open a second doc to say the first was wrong.
- **Correct the verdict, not just the body.** A dated in-place fix that leaves the false
  claim standing in `verdict:` and in the INDEX row has corrected nothing — CLAUDE.md
  tells readers to trust the index *instead of* the doc, and the gate only checks that the
  two AGREE, so two matching lies pass. Fix the body, the `verdict:`, and the row together.
  If the wrong thing was a prose count, **delete it** rather than correcting it a third
  time; a number nothing re-derives will drift again.
- **Superseded docs MOVE directory; they are never deleted.** The old file keeps its
  provenance and its verdict follows it, and its INDEX row moves with it.
- A new doc's front matter, its INDEX row, and whatever it documents land in the **same
  change** — a doc added now and registered "next commit" is a red gate for whoever pulls.

## Verifying — the one unlocked check

```bash
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli \
  cargo test --workspace --no-default-features --test docs
```

No GPU, no lock, safe anywhere. **The explicit `CARGO_TARGET_DIR` is mandatory**
(`<checkout>` is the worktree's name, `rivoli` for the primary checkout):
`/etc/security/pam_env.conf:6` exports a shared `CARGO_TARGET_DIR` on every node and it
outranks any `.cargo/config.toml`, so without it you are sharing a target dir with every
other checkout — a build-script binary baked for a sibling's `crates` directory scanning a
tree this checkout does not have, which has already produced a green that certified
nothing (`docs/measurement/gate-red-proofs.md` §12, and §7e-bis for the defect itself).

Read the exit code **unpiped**. Then read the failure text: the gate names the file, the
key, and which of the two sides it disagreed with.
