---
status: live
scope: engine
verdict: Every doc in this tree, with status, scope, and the same verdict its front matter carries — the agreement is test-enforced by crates/cli/tests/docs.rs, so a stale row here is a red test, not a trap.
---

# Index

Start with [TOUR.md](TOUR.md). Then use the verdict column below to decide what **not** to
open. A doc's row must agree with its own front matter; `crates/cli/tests/docs.rs` fails
when they drift, when a doc is missing from here, or when a doc has two rows.

Layout: `reference/` = true today · `measurement/` = how to measure and what was measured ·
`investigations/` = asked, answered, closed. A doc that stops being true moves directory —
that move is the signal. A closed verdict rules its question out **only for its `scope:`**.

| doc | scope | status | verdict |
|---|---|---|---|
| [principles.md](../reference/principles.md) | engine | live | Why rivoli exists, as seven principles a plan can be checked against — confirmed by the owner 2026-08-12. Decode bigger-than-memory models on this one box; caching IS the space/bandwidth/compute trade; maximize hardware features over portability; the budget trades speed, never text; bytes/token is the currency; the pin is a function of free memory, not of architecture; every claim is a gate that can go red. |
