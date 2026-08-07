---
scope: v4
status: live
verdict: OPEN, and newly justified by measurement. `v4-oracle emit` hardcodes `Defect::None`, so every derived tolerance in this repo is validated by host transcription rather than through the gate it protects. A `--defect` flag is the whole change; `Defect::ALL` already enumerates the breakages and the `defects` subcommand already drives them. The reason to build it is no longer aesthetic: on 2026-08-07 the checkpoint was measured to discriminate a defect the toy fixture is bit-blind to (39,893/53,248 elements vs exactly 0), so real-weights goldens are strictly stronger, not merely more realistic.
---

# Can a bound be derived through the gate instead of beside it?

**STATE.** `src/bin/v4-oracle.rs::emit()` writes goldens at `Defect::None` and nothing can ask
it for anything else. So a tolerance is derived by transcribing the arithmetic on the host and
reasoning about its envelope, and then *separately* the gate is run to see whether it passes.
The derivation never goes through the gate. That is the one weakness the four derived
attention bounds still have, and it is named as owed in `tests/v4_loop.rs` and tracked nowhere
else — which is why it is here.

## The change

Add `--defect <name>` to the `emit` subcommand. `Defect::ALL` already enumerates every
variant, `Defect` already parses in the `defects` path, and `Oracle::new(cfg, defect)` already
takes one. The emit path is the only thing that does not.

```
v4-oracle emit --model <dir> --out <file> --layers N --decode-steps K [--defect <name>]
```

Design constraints, each with a reason:

- **A flag, not an env var.** This repo's rule, and this is exactly the case it exists for: a
  golden emitted under a silently-active perturbation is indistinguishable from a real one.
- **The output must name the defect.** A perturbed golden that can be mistaken for
  `Defect::None` is worse than no feature. Either refuse an `--out` that does not carry the
  defect name, or write the name into the file's own header and have the loader check it.
- **`--defect None` must be spellable and must be identical to omitting the flag**, so the
  no-perturbation arm of any A/B goes through the same code path as the perturbed one.
- **Refuse an unknown name loudly**, listing the variants. A typo that silently falls back to
  `None` produces two identical goldens and an A/B that cannot fail — this repo's most-repeated
  failure shape.

## Why this is worth building, measured rather than assumed

Until 2026-08-07 the argument was "real weights are more realistic", which is the kind of
claim that does not justify work. It is now a measurement.

`Defect::SinkhornIterCountProbe` runs `hc_sinkhorn_iters - 1` Sinkhorn passes. On the toy
fixture, 19 and 20 iterations are **bit-identical** — the 4x4 matrix reaches a bitwise fixed
point, `tests/v4_oracle.rs::sinkhorn_has_converged_long_before_iteration_20` asserts exactly
that, and the variant is excluded from the oracle's defect matrix for that reason. On the
checkpoint, `v4-oracle defects --layer 0 --decode-steps 1` gives:

```
SinkhornIterCountProbe  L0.pre.ffn_norm_out(39893/53248)  L0.pre.router_weights(78/78)
                        L0.pre.ffn_out(50812/53248)       L0.pre.out(143026/212992)
```

All 78 router weights and two thirds of the block output. Convergence is to within f32
rounding, and whether the last ulp settles is weight-dependent; the toy's mixes settle and the
checkpoint's do not.

**So there exists at least one defect that a toy-fixture golden cannot see and a real-weights
golden can.** That is the concrete argument: a `--defect` flag does not merely make the
existing derivations more convincing, it reaches a class of defect the current instrument is
structurally blind to. The full accounting and the four in-place corrections that finding
forced are in the dated notes on `sinkhorn_has_converged_long_before_iteration_20`,
`Defect::SinkhornIterCountProbe`, `V4Config::hc_sinkhorn_iters`, `launch_hc_pre` and
`kernels/linalg.hip::hc_sinkhorn`.

## What it unlocks

1. **Re-derive the four attention bounds through the gate.** `attn_norm_out` 17, `q` 275,
   `kv_entry` 23, `attn_derot` 71 were each derived on the host. With `--defect`, each can be
   checked the other way round: emit a golden under a defect the bound is supposed to catch,
   and confirm the gate goes red. A bound that stays green against a defect it claims to catch
   is the finding.
2. **A real bidirectional test for every bound.** The pattern this repo already trusts — the
   defect must move the tensors it claims to and leave the rest bit-identical — becomes
   available at real dims instead of toy dims.
3. **It may close [`ffn-norm-out-envelope.md`](ffn-norm-out-envelope.md) cheaply, or prove it
   cannot be closed cheaply.** If a real-weights perturbed golden separates `ffn_norm_out`
   cleanly where the differing-element fraction managed only 1.42x, the envelope transcription
   may not be needed at all. That is worth knowing before spending the transcription.

## Cost and hazards

The flag itself is small. The cost is in the runs: `emit` reads the real checkpoint, and the
43-defect sweep is 7 minutes for one layer at one decode step. Emitting a *golden* per defect
is heavier than the sweep's in-memory comparison and should not be done for all 43 by reflex —
pick the defects a given bound claims to catch.

- **Do not let a perturbed golden reach a benchmark.** Anything under `--defect` is an
  instrument artefact. If goldens are ever cached by path, the defect name must be in the path.
- **The sweep prints `EXIT=0` when it finishes and its table inverts if read early.** A partial
  run already produced one wrong conclusion here, recorded in `ffn-norm-out-envelope.md`.
- **A defect that moves everything proves nothing.** The bidirectional half — the defect must
  leave the tensors it does not claim *bit-identical* — is the half that makes the check
  informative, and it is the half that is easy to skip when the run is expensive.
