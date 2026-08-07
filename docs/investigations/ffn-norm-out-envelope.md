---
scope: v4
status: live
verdict: OPEN. `ffn_norm_out` and `.out` carry no bound at all since 2026-08-07 — they report and assert only that the row was reached. The 5e-2 they used to carry was the same constant four attention tensors were re-derived away FROM, whose derived values came out 17, 275, 23 and 71. Two substitutes have been measured and REFUTED: the differing-element fraction at 1.42x (probe sweep), and, later the same day, a perturbed-golden A/B through the gate itself — a `SinkhornIterCountProbe` golden ran the gate green, with a same-tensor fraction separation of only 1.20x. The work is transcribing `hc_post` + the MoE to compute the envelope, unblocked since Track 0 released the files on 2026-08-06.
---

# What bound do `ffn_norm_out` and `.out` actually deserve?

**STATE.** `tests/v4_loop.rs` compares the device against the oracle on six tensors per layer.
Four are bounded by a *derived* envelope. Two — `L{n}.pre.ffn_norm_out` and `L{n}.pre.out` —
pass `None`: they are read back, compared, and reported, with no pass/fail bound. That is
deliberate and it is not a hole being hidden; it is a hole being *labelled*. This document is
how it gets closed.

## Why they have no bound

They had `Some(5e-2)`. That constant was never derived for them. It is the same literal the
four attention tensors carried until they were re-derived, and the derivation moved every one
of them by three to four orders of magnitude:

| tensor | old | derived |
|---|---:|---:|
| `kv_entry` | 5e-2 | 17 |
| `q` | 5e-2 | 275 |
| `attn_derot` | 5e-2 | 23 |
| `attn_out` | 5e-2 | 71 |

> **CORRECTED 2026-08-07.** This table listed `attn_norm_out` (which was never re-derived —
> it still carries the chosen 5e-2) and omitted `attn_out`, shifting three bounds onto the
> wrong tensors. The four derived tensors and their bounds are as now shown
> (`src/v4gpu.rs::AttnStages::scored`); the body's later mention of "`attn_out` at 1.6x"
> always referred to a row this table failed to carry.

A constant already measured wrong for four siblings, left on two others only because their
envelope sat outside an earlier track's file set, is not evidence about those two. It was
removed on 2026-08-07 rather than widened — **widening a bound until the test passes is
forbidden here, and the file says so in bold.** Red on an underived bound is honest; green on
a widened one is not.

**The half that makes an unbounded comparison honest is asserting the row was reached.**
`check` now refuses an empty or length-mismatched comparison before it does anything else:

```
L0.pre.out: compared 0 against 212992 elements — the row was never reached
```

Proven live by emptying a readback. Without it, "reported, no bound" decays silently into
"never ran", which is the failure mode that makes unbounded rows dangerous. **Keep this
assertion when the bound lands.**

## What was tried instead, and refuted

`tests/v4_loop.rs` nominated the **differing-element fraction** as the statistic that might
gate these two without an envelope — it is the statistic that separated every defect an
earlier review could construct, and it beats `max_rel` on `q`, where `max_rel` is
floor-dominated and reads 1.07 for a defect that doubles every element.

Measured 2026-08-07, `v4-oracle defects --layer 0 --decode-steps 1` (7 minutes, CPU only):

- 43 defects run. **13 move nothing at all** on this probe, and only **9 move
  `ffn_norm_out`.** A statistic cannot gate a defect that does not move the tensor, so the
  fraction's in-scope subset here is 9 of 43 — *smaller* than `max_rel`'s.
- Of those 9 the weakest is **`SinkhornIterCountProbe` at 39,893/53,248 = 74.9%**, against the
  device's **28,141/53,248 = 52.85%**. A separation of **1.42x**.

1.42x is the same order as the two bounds this file already calls barely-gates — `attn_derot`
at 1.3x and `attn_out` at 1.6x. So switching statistic does not rescue the downstream tensors;
it moves the same weak separation onto a different axis, for the same underlying reason:
`hc_post` dilutes a sublayer error across four residual copies before the MoE dilutes it
again.

> **A first reading of this table said "88.5%, a real separation" and drew the opposite
> conclusion.** It was read off an INCOMPLETE run — the matrix was still executing and the
> 74.9% row had not printed. Do not quote this table from a partial log. It takes 7 minutes
> and prints `EXIT=0` when it is done.

**Also tried and refuted, 2026-08-07 — the perturbed-golden shortcut.**
[`real-weights-defect-goldens.md`](real-weights-defect-goldens.md) hoped its `--defect` flag
"may close this doc cheaply". Measured through the gate the same day: a
`SinkhornIterCountProbe` golden ran the whole `tests/v4_loop.rs` gate **GREEN** — no bounded
row breached, and everything the defect moved sat in the unbounded "reported" rows. Two
distinct fractions, not to be conflated: the defect's own footprint on `L1.pre.ffn_norm_out`
is **91.1%** of elements (golden vs perturbed golden, that doc's fixed-probe finding — the
74.9% above is probe-driven and does not transfer); what the gate saw is **69.6%** differing
(device vs perturbed golden, `max_rel` 6.35, carrying the device's own noise). Against the
same-tensor device baseline — arm 1's `L1.pre.ffn_norm_out` at **58.1%**, not the 52.85%
`L0.pre` figure — the separation is **1.20x**, weaker still than the 1.42x above. So no
statistic currently asserted, and no perturbed golden, substitutes for the envelope. The
transcription below remains the work.

## The work

Transcribe `hc_post` and the MoE into the same envelope calculation that produced 17, 275, 23
and 71 for the attention tensors, and derive the bound for `ffn_norm_out` and `.out` from it.
That is the only instrument that has worked on this gate, and nothing measured since suggests
a cheaper one exists.

**Unblocked since 2026-08-06**, when Track 0 closed with no defect and released `src/attn.rs`
and `src/v4gpu.rs` from its exclusive ownership.

Order of work:

1. `hc_post` first — it is the smaller of the two and it is where the dilution starts. Its
   four residual copies are what turn a sublayer error into a diffuse one, so the envelope
   has to model the combination, not just the magnitudes.
2. The MoE second. Top-8 of 256 with a routing weight applied to the SwiGLU intermediate; the
   reference rounds the weighted intermediate to bf16 before `w2`, and that rounding is inside
   the envelope, not outside it.
3. Derive, then flip both rows from `None` to `Some(derived)` **in the same commit as the
   derivation's record.** A bound whose derivation is not written down is the 5e-2 problem
   again with a different number.

## Hazards, each of which has already cost something here

- **Do not widen.** If the derived bound does not admit the observed difference, that is a
  finding about the device, not about the bound.
- **Keep the anti-vacuity assertion.** It is what distinguishes "no bound" from "no test".
- **Do not derive against the toy fixture.** Measured 2026-08-07 on a different question, the
  toy and the checkpoint disagree about whether a defect is observable at all — the Sinkhorn
  iteration count moves 39,893 elements on real weights and exactly zero on the toy. An
  envelope validated only against the toy inherits that blindness. See
  [`real-weights-defect-goldens.md`](real-weights-defect-goldens.md), which is the other half
  of this problem.
- **`.out` is downstream of `ffn_norm_out`.** Deriving one does not give the other for free,
  but a bound on `.out` that is *tighter* than its input's is wrong by construction and is a
  useful self-check.

## What a successful negative result looks like

"The envelope for these two is wider than the observed difference by so much that the bound
gates nothing" is a real answer, and it is the answer the 1.42x measurement above already
hints at. If that is where the derivation lands, say so with the number and leave the rows
reporting — an honest `None` beats a bound that cannot fail.
