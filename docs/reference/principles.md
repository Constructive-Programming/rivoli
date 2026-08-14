---
scope: engine
status: live
verdict: Why rivoli exists, as seven principles a plan can be checked against — confirmed by the owner 2026-08-12. Decode bigger-than-memory models on this one box; caching IS the space/bandwidth/compute trade; maximize hardware features over portability; the budget trades speed, never text; bytes/token is the currency; the pin is a function of free memory, not of architecture; every claim is a gate that can go red.
---

# Principles

Why this engine is being written. Confirmed in this form by the owner on 2026-08-12; the
Glimmer integration plan (`investigations/glimmer-integration.md`) is the first written
against them. A change that violates one of these is wrong even if every test passes —
and a plan stage that quietly assumes one away (S1a's "a dense model has nothing to
stream") is where that starts.

## P1 — Decode models bigger than memory, on this one box

AMD Strix Halo, unified LPDDR5 via GTT, weights streamed from NVMe **overlapped with
compute**. The overlap is the whole design; everything else serves it. A model that fits
is the degenerate happy case of the streaming path, never a separate resident-only design.

## P2 — Caching is a key principle

The engine acknowledges — and trades in — the triangle of **memory space, bandwidth, and
compute**. Hybrid mode is the canonical example, in the owner's words: GLM-5.2's int4 is
cheaper to compute but more expensive to store than vq3, so int4 is used where there is
bandwidth and space to load it, vq3 where there is not. The cache is not a transparency
layer; which format an expert or layer occupies is a first-class decision made against
that triangle.

## P3 — Maximize hardware features

We avoid universality of where we can run when it costs performance: **hardware
compatibility is always traded away for efficiency**. The Vulkan backend's retirement is
this principle enforced retroactively; gfx1151-specific kernel decisions need no apology.
If and when the NPU is valuable, we use it — the 2026-08-07 npu-offload closure is
GLM-scoped by its own `scope:` field and rules nothing out for other models.

## P4 — The memory knob trades speed, never text

`--max-mem` and cache policy move tok/s and hit rate, not output. INV-1 is this principle
as a test; hybrid's violation of it is the standing open defect precisely because it
breaks the principle, not because the text it produces is bad. Any new format-by-residency
scheme must state its position against this principle explicitly.

## P5 — Bytes per token are the currency

Decode speed = traffic ÷ bandwidth. Quantization, residency, prefetch, and speculative
decode are all traffic levers; the quality price of each is measured on paired dNLL from
`bin/ppl`, never assumed. A win counts even when hidden behind another bottleneck
(efficiency is cost); one-time costs amortize to ~0 against millions of decodes.

## P6 — The pin is a function of free memory

What is resident is decided by how much memory is free **at run time** — other tenants,
KV at the configured context, scratch — never by model architecture. "This model is dense,
so everything is pinned" and "this model routes, so experts stream" are both category
errors: the pin holds whatever the budget leaves room for, and the streaming path must be
able to carry anything the pin dropped.

## P7 — Every claim is a gate that can go red

Tolerances measured before the kernel exists; red proofs run and reverted; invariants
registered with tests (`architecture.md` §8b); closed investigations kept for what they
eliminated; a gate is proven able to fail before its green is believed. Method rather than
product — but it decides what any plan may call "done".
