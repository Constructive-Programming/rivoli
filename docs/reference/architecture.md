---
status: live
scope: engine
verdict: The rewrite's architecture as built so far — the six-crate layering and what it precludes, the P6 residency contract, and the §8b invariant registry, which grows a section per milestone rather than being written ahead of the code.
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
| INV-1 | Routing is a pure function of `(gate logits, bias, top_k, scoring)` — never consults the cache. Inherited number, inherited meaning: the old tree's INV-1, whose violation in hybrid mode is that tree's standing open defect. In this tree the DAG enforces the stronger form (core cannot name residency) and the ported property test keeps the number alive. | `crates/core/src/routing.rs::inv_1_routing_never_consults_the_cache` |
| INV-8 | The pin is a function of `(ordered units, free bytes, floor)` only — monotone in `free`, prefix-shaped, all-resident as the degenerate top. P6 as a gate: no architecture parameter exists to consult, so "this model is dense, so everything is pinned" cannot be expressed. Numbered fresh (1 was briefly reused here before the routing port arrived carrying the original; the NEW invariant moved). | `crates/core/src/residency.rs::inv_8_the_pin_is_monotone_in_free_memory_and_nothing_else` |
| INV-4 | A device-side wait may be enqueued BEFORE its producer exists and still waits — the property `hipStreamWaitEvent` lacks, stated as a property because the retired second backend reached it by a different mechanism. Arrived with the waist port; ID inherited from the old registry so citations travel. Device test: runs only under `--features rocm` on the box. | `crates/backend/src/gpustream.rs::inv_4_wait_enqueued_before_signal_still_waits` |
| INV-6 | A wait can always be released from the HOST, so a producer that dies owing a ticket cannot hang the device (monotone CAS into signal memory). Missing in the old tree until 2026-08-01 — `hipStreamWaitValue64` has no error state, so a fetch error hung the device instead of returning. Device test, same rocm-only caveat as INV-4. | `crates/backend/src/gpustream.rs::inv_6_a_host_release_retires_an_enqueued_wait` |
| INV-5 | An expert cannot be launched without enqueueing its data dependency: `RoutedPool::submit` returns a `Ticket` per selected expert and no residency mask, so the loop has nothing to branch on — resident / missing / in-flight are one code path, and `wait_on` is the only consumer. The testable half is the encoding: `RESIDENT` is value 0, a genuinely satisfied wait, never a sentinel the consumer must recognise and skip (that branch is the `hit` mask growing back — the bool that once won a disagreement silently and launched over a slot still being written). ID inherited from the old registry; arrived with the M4 pool port, rocm-only like the pool. | `crates/engine/src/routed.rs::inv_5_every_descriptor_carries_a_ticket` |
| INV-9 | **A staging slot is never re-issued until its bounce copy has retired**, so the one host decision on the routed path that reads DEVICE PROGRESS rather than its own inputs cannot vary between two runs of one input. `AsyncFetch::take_slot`'s predicate is a timeline value, i.e. wall-clock; what makes it uniform is a barrier elsewhere — `glm::forward::run_layer` ends every layer with an unconditional `device_sync`, so at the next `submit` every prior copy has retired and `scan_free` is pure round-robin over the miss sequence. `AsyncFetch::slot_stalls()` is the observable that falsifies that precondition. **The test asserts the slot-reuse rule itself** — every slot issued exactly once, `None` when all copies are in flight, and only the slot whose timeline advanced coming back — not the barrier and not the whole-path claim; the wider statement is reasoning ON TOP of it and is argued in `docs/investigations/glm-nondeterminism.md`. **Scope:** this is the same-PROGRAM half, not the same-OUTPUT half. GLM does not currently reproduce its own OUTPUT over 512 tokens, and that property's gate is `tests/determinism-glm.sh` — length-aware, floor 512, because the defect is byte-identical at 32 tokens. Added 2026-08-17; red-proofed by making `scan_free` ignore its `landed` predicate, which turns the all-in-flight `None` into `Some(0)`. | `crates/engine/src/fetch/asyncfetch.rs::inv_9_a_slot_is_not_reissued_until_its_copy_lands` |
