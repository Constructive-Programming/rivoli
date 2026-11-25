# Principles

Seven principles, confirmed by the project owner on 2026-08-12. A change that violates
one is wrong even if every test passes.

<div class="rv-card ember">

## P1 — Decode models bigger than memory, on one box

A ROCm unified memory system, with weights streamed from NVMe **overlapped with
compute**. The overlap is the whole design; everything else serves it. A model that fits
is the trivial case of the streaming path, never a separate resident-only design.

</div>

## P2 — Caching is a key principle

The engine trades in the triangle of **memory space, bandwidth and compute**. Which
format an expert or layer occupies is a first-class decision made against that triangle,
not a detail hidden behind a cache.

## P3 — Maximize hardware features

Running everywhere is not a goal when it costs performance: **hardware compatibility is
always traded away for efficiency**. Hardware-specific kernel decisions need no apology. If
the NPU becomes valuable, rivoli will use it.

## P4 — The memory knob trades speed, never text

`--max-mem` and cache policy move tok/s and hit rate, not output. Any scheme that picks a
format based on residency must state where it stands against this principle.

## P5 — Bytes per token are the currency

Decode speed = traffic ÷ bandwidth. Quantization, residency, prefetch and speculative
decode are all traffic levers. The quality price of each is measured on paired
per-token NLL differences, never assumed. A win counts even when another bottleneck hides
it, and one-time costs amortize to nearly zero over millions of decodes.

## P6 — The pin is a function of free memory

What is resident is decided by how much memory is free **at run time**, never by model
architecture. "This model is dense, so everything is pinned" and "this model routes, so
experts stream" are both category errors. The pin holds whatever the budget leaves room
for, and the streaming path must be able to carry anything the pin dropped.

<div class="rv-card">

## P7 — Every claim is a gate that can go red

Tolerances are measured before the kernel exists. Red proofs are run and reverted.
Invariants are registered with tests. A gate is proven able to fail before its pass is
believed. This is method rather than product, but it decides what any plan may call
"done". See [Quality gates](quality.md).

</div>
