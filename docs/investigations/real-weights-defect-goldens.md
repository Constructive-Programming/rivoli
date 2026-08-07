---
status: live
verdict: OPEN, GPU half pending — the flag EXISTS since 2026-08-07 (`emit --defect`, defect name in the file's own header, loader refuses a mismatch) and the host half is measured, bidirectionally, on five perturbed real-weights goldens; each moves exactly the tensors its toy row claims and leaves the rest bit-identical. One finding already, against this doc's own headline: the sweep's 74.9% `ffn_norm_out` figure for `SinkhornIterCountProbe` does not survive the emit chain — driven by the real residual instead of the fixed probe, its direct footprint is 1 element in 53,248, so detectability is input-dependent as well as weight-dependent. The through-the-gate runs are prepared with pre-registered predictions and not yet run.
---

# Can a bound be derived through the gate instead of beside it?

**STATE.** `src/bin/v4-oracle.rs::emit()` writes goldens at `Defect::None` and nothing can ask
it for anything else. So a tolerance is derived by transcribing the arithmetic on the host and
reasoning about its envelope, and then *separately* the gate is run to see whether it passes.
The derivation never goes through the gate. That is the one weakness the four derived
attention bounds still have, and it is named as owed in `tests/v4_loop.rs` and tracked nowhere
else — which is why it is here.

> **SUPERSEDED 2026-08-07.** The paragraph above describes the state this doc was opened
> against. The flag now exists and the host half is measured — see "built, and the host half
> measured" below; only the through-the-gate GPU runs remain.

## The change

Add `--defect <name>` to the `emit` subcommand. `Defect::ALL` already enumerates every
variant, `Defect` already parses in the `defects` path, and `Oracle::new(cfg, defect)` already
takes one. The emit path is the only thing that does not.

> **CORRECTED 2026-08-07.** "`Defect` already parses in the `defects` path" was wrong —
> `defects` enumerates `Defect::breakages()`; nothing anywhere parsed a name. The parser is
> new (`Defect::from_flag`).

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

## 2026-08-07 — built, and the host half measured. The GPU half is prepared, not run.

### What exists now

`v4-oracle emit --defect <name>`, honoring all four constraints above, plus one hazard rule:

- The name is parsed by `Defect::from_flag` (new — this doc's claim that "`Defect` already
  parses in the `defects` path" was **wrong**: `defects` enumerates `Defect::breakages()`,
  nothing anywhere parsed a name). An unknown name refuses and lists all 51 variants;
  `tests/v4_oracle.rs::defect_from_flag_roundtrips_and_refuses_loudly` holds both directions.
- The name is written into the file's metadata under `golden::DEFECT_KEY` — for **every**
  emit, `None` included, so "declared unperturbed" and "predates the flag" stay
  distinguishable. `GoldenSet::defect()`/`expect_defect()` are the loader half;
  `tests/v4_loop.rs::open_goldens` calls `expect_defect` against `RIVOLI_V4_GOLDENS_DEFECT`
  (default `None`) before comparing anything, and the check itself is unit-tested in
  `src/v4oracle/golden.rs` (a perturbed header cannot pass as `None`; a legacy keyless file
  reads as `None` — safe, because only pre-flag binaries produced keyless files).
- `--defect None` measured byte-identical to omitting the flag, on the real checkpoint.
- A breakage additionally requires the `--out` file name to carry the defect name — the
  "perturbed golden at a neutral path" hazard above, enforced at emit rather than by
  discipline. `v4-goldens.bin` as a default out for a perturbed emit is therefore impossible.
- `v4-oracle cmp <base> <perturbed>` prints every tensor of two golden files, zeros included,
  plus each file's declared defect — the instrument for the bidirectional half below. It
  brands an identical pair as unable to A/B anything (the failure shape the typo-refusal
  exists for), refuses a pair emitted at different prompts, and scales `max_rel` by the
  **base** side — scaling by the perturbed side would understate an inflating defect and
  explode on a collapsing one.
- The other consumer of this file format, `docs/measurement/probes/v4_attn_amplification.py`,
  now refuses any golden whose header declares a defect — its whole output is the envelope of
  a *correct* implementation, and it had no notion of the key (found by review 2026-08-07;
  refusal proven red against a perturbed file, green against base).

Cost correction: one `--layers 2 --decode-steps 1` emit is **~20 s** on the real checkpoint
(release binary), not minutes — the hazard is not the emit cost, it is still the gate runs.

### Host-side bidirectional evidence, real weights, emit chain

Base golden re-emitted at `Defect::None`: all 53 tensors bit-identical to the deployed
`/var/db/rivoli/v4-goldens-l2.bin` (which differs only by the new metadata key). Five
perturbed goldens, chosen as each derived bound's own weakest-in-scope defect
(`AttnStages::scored`'s table) plus the Sinkhorn probe; every pair byte-diffed against base
before comparison. Counts are `changed/total` from `v4-oracle cmp`:

| defect | anchors bound | moves (first directly-hit tensors) | stays bit-identical (the claim) |
|---|---|---|---|
| `RopeHalfSplit` | `kv_entry` 17 | `L0.pre.q` 46203/425984, `L0.pre.kv_entry` 712/6656, everything downstream | 13 rows: embed, `L0.*.in`, `L0.*.attn_norm_out`, all 4 `head.probe.*`, all 4 `router_indices` |
| `RopeFirstDims` | `q` 275 | `L0.pre.q` 80282/425984, `L0.pre.kv_entry` 1071/6656, downstream | same 13 rows |
| `SkipKvActQuant` | `attn_derot` 23 | `L0.pre.kv_entry` 5470/6656, downstream | same 13 **plus `L0.pre.q` and `L0.dec0.q`** — the q path never sees the kv quant |
| `KvActQuantWholeTensor` | `attn_out` 71 | `L0.pre.kv_entry` 781/6656 (the wrongly-quantized rope dims), downstream | same 15 rows |
| `SinkhornIterCountProbe` | (none — the `ffn_norm_out` question) | `L0.pre.ffn_norm_out` **1**/53248, `L0.pre.router_weights` 6/78, then chain-amplified: `L1.pre.attn_norm_out` 11662/53248, `L1.pre.ffn_norm_out` 48526/53248 | **26 rows**, including the ENTIRE `L0` attention half and the entire `L0.dec0` phase |

Every row matches the toy expectation table (`tests/v4_oracle.rs::expect`) transported to the
emit chain, with the one structural difference the chain adds: `L1.*.in` inherits `L0.*.out`,
so L1 tensors move for inherited reasons and only the L0 rows carry the isolation claim.

### The finding: the sweep's headline number was an artefact of its fixed probe

This doc's justification quoted `SinkhornIterCountProbe` at 39,893/53,248 (74.9%) of
`L0.pre.ffn_norm_out`, from `v4-oracle defects` — which drives a **fixed synthetic probe**.
On the emit chain, driven by the model's own embedding of the same prompt, the direct
footprint is **1 element in 53,248**, and the entire `L0.dec0` phase (a different fresh
embed) moves **nothing at all** — the Sinkhorn iteration reaches its bitwise fixed point on
that input exactly as it does on the toy. So detectability of this defect is
**input-dependent as well as weight-dependent**, and the sweep's fraction column does not
transfer to the goldens the gate actually consumes. The justification for the flag survives
(the checkpoint does discriminate where the toy cannot), but any plan to close
[`ffn-norm-out-envelope.md`](ffn-norm-out-envelope.md) with this defect must use the
chain-amplified L1 numbers (48,526/53,248 = 91.1% on `L1.pre.ffn_norm_out`), not the probe
sweep's 74.9%.

### The GPU half, prepared and NOT run — with predictions registered before measuring

The goldens are reproducible in ~20 s each (deterministic host arithmetic; the `None` emit
was verified byte-identical across runs and tensor-identical to the deployed fixture):

```
cargo run --release --bin v4-oracle -- emit --layers 2 --decode-steps 1 \
    --out v4-goldens-l2-none.bin                                    # and once per defect:
    --defect RopeHalfSplit          --out v4-goldens-l2-ropehalfsplit.bin
    --defect RopeFirstDims          --out v4-goldens-l2-ropefirstdims.bin
    --defect SkipKvActQuant         --out v4-goldens-l2-skipkvactquant.bin
    --defect KvActQuantWholeTensor  --out v4-goldens-l2-kvactquantwholetensor.bin
    --defect SinkhornIterCountProbe --out v4-goldens-l2-sinkhornitercountprobe.bin
```

(One prepared set currently also sits in the 2026-08-07 session scratchpad, but the commands
above are the durable record — a session temp path is not.) Build outside the lock
(`cargo test --features rocm --test v4_loop --no-run`), then per arm, dev profile,
`--test-threads=1`, flocked:

1. **Baseline, expected GREEN** — `RIVOLI_V4_GOLDENS=<dir>/v4-goldens-l2-none.bin`. Proves
   the new-format golden passes through the same code path as the perturbed arms.
2. **Loader-check arming, expected RED AT LOAD** — the `ropehalfsplit` file with
   `RIVOLI_V4_GOLDENS_DEFECT` unset: must panic "emitted under --defect RopeHalfSplit"
   before any comparison. This is the prove-the-gate-can-go-red precondition.
3. **`RopeHalfSplit` + matching env var** — predict RED with `kv_entry` rows breaching 17
   (host transcription puts the defect at 116 on that tensor) and `q` rows breaching 275.
4. **`RopeFirstDims`** — predict RED with `q` rows breaching 275 (defect at 1,482).
5. **`SkipKvActQuant`** — the marginal one: the defect sits at 26.8 against `attn_derot`'s
   23 (1.17x). Predict red on `attn_derot`, **but a green here is the named finding** — a
   bound that stays green against the very defect its derivation anchored on.
6. **`KvActQuantWholeTensor`** — same shape: 94.6 against `attn_out`'s 71 (1.33x). Red
   predicted, green is the finding.
7. **`SinkhornIterCountProbe`** — predict **GREEN overall**: no bounded tensor moves enough
   (`router_weights` moved ≤ 4.3e-3 base-relative against its 1e-2 bound; `ffn_norm_out`
   and `.out` carry no bound). A full green here would answer unlock #3 negatively: the one
   defect the checkpoint discriminates and the toy cannot is invisible to every bound the
   gate currently asserts, so the envelope transcription is still owed.

Caveat on 3–7: the numbers behind the predictions are per-element `max_rel` from the host
transcription at `L0.pre`, while `cmp`'s column is tensor-scale-relative — they are not the
same statistic, which is exactly why the runs go through the gate instead of being declared
from this table.
