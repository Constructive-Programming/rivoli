---
status: live
scope: engine
verdict: The rewrite's architecture as built so far — the six-crate layering and what it precludes, the P6 residency contract, and the §8b invariant registry (INV-1), which grows a section per milestone rather than being written ahead of the code.
---

# Architecture

Grows a section per milestone; a section describes what is BUILT, not what is planned
(plans live in `../../docs/investigations/` when they need a record at all). The old
tree's architecture.md is the reference for everything not yet rebuilt, at the pin named
in CLAUDE.md.

## 1. The workspace

Six crates whose dependency DAG is the layering — see TOUR.md's table and the root
`Cargo.toml`'s rationale comment. The two structural consequences worth restating: core
cannot name a weight format (the old tree's format-follows-residency defect is
un-writeable here), and the featureless workspace is the default, CI-tested build.

## 2. Residency — the P6 contract

`rivoli-core::residency::partition(ordered, free, floor)` is the one author of the
placement decision. No architecture parameter exists to consult; per-model pins hold
different tensors but never decide what is resident. The returned partition is a PREFIX
of the caller's priority order — prefix-ness is what makes the decision monotone in
`free` (more memory only extends the pin) and what makes a dense cyclic model's optimal
policy (static prefix, the Belady degenerate) the same code path rather than a special
case. Below-floor budgets refuse with the arithmetic in the message; the run never
degrades.

## 8b. Invariants (INV-n) — and the mechanism that keeps this section honest

A documented invariant with no `inv_<n>_*` test, or a test naming an invariant no longer
documented, fails `crates/cli/tests/invariants.rs` — in both directions, with an
anti-vacuity floor (the registry must not be empty). The section number is inherited from
the old tree so citations travel.

| ID | invariant | test |
|---|---|---|
| INV-1 | The pin is a function of `(ordered units, free bytes, floor)` only — monotone in `free`, prefix-shaped, all-resident as the degenerate top. P6 as a gate: no architecture parameter exists to consult, so "this model is dense, so everything is pinned" cannot be expressed. | `crates/core/src/residency.rs::inv_1_the_pin_is_monotone_in_free_memory_and_nothing_else` |
