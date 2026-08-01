---
status: closed-shipped
verdict: Why int4 was unusable and how group-128 scales fixed it: PPL 73.43 → 5.120, making int4 the best-quality mode. RESOLVED.
---

# int4: why `--mode int4` was unusable, and the fix

**Status: RESOLVED, 2026-07-27.** The cause was a *format* choice, not a bug — one symmetric
scale per **6144-weight row**. Replacing it with one scale per **128 weights** fixed it
outright. The investigation below is preserved because what it *eliminated* is worth as much
as what it found, and because two of its findings outlive int4 entirely (§1, §9).

| mode | before (per-row) | after (group-128) |
|---|---:|---:|
| `int4` | 73.43 | **5.120** |
| `hybrid` | 11.55 | **5.189** |
| `int3-vq` *(control — never reads `.i4`)* | 5.275434 | **5.275434** |

The control reproduced to six decimals, so the `.i4` artifact was the only variable. **int4 is
now the best-quality mode in the engine, and hybrid dominates int3-vq on both axes** — better
PPL (5.189 vs 5.275) *and* faster (2.72 vs 2.62 tok/s), which is what hybrid was designed for
and had never delivered. Cost: `.i4` grows ~6% (slot 18,915,328 → 20,054,016 B), so fewer
experts are resident and int4's hit rate falls 74.6 → 68.1% — int4 is now the best quality and
the *slowest*; hybrid is the best overall. See §6 for why group size and not a scale tweak,
and §10 for the confirming measurement.

The rest of this document describes the **pre-fix** state, in the present tense as written.

> Ranking before the fix: **int3-vq (5.28) > hybrid (11.55) > int4 (73.43)**

Everything below is from one 2026-07-27 device block, one binary, one artifact. Nothing is
compared across sessions except where a control licenses it, and those controls are named.

**Reproduction.** Artifact `/var/db/rivoli/glm52-vq3-full` (`i4_source` stamp: tool
`fp8_to_i4`, chain `fp8->int4`, src `/swarm/storage/ai/openclaw/glm52-fp8`, layers [3,78]) —
that stamp was the authoritative way to tell the two `.i4` sets apart, since one overwrote
the other in place. Engine: this branch (`feat/fp8-i4`) for A0–A5/M1/H, `41522f6` for P.

> **RE-STAMPED later the same day (2026-07-31).** The artifact carries an `i4_source` again
> — `layers [3, 79]`, same tool/chain/src, group 128 — because `bin/fp8_to_i4 --from 78
> --to 79` was run to add the MTP head's slab, and stamping is what that tool does. Two
> traps came out of it, both now guarded:
>
> - A subrange run merges with a PRIOR stamp, but there was none, so it wrote `[78, 79]`
>   over a full set — a claim NARROWER than the `.i4` on disk, which reads as "only layer 78
>   is fp8-derived" and is worse than no claim. The tool now warns when the range it stamps
>   does not cover the `L*.i4` present. The `[3, 79]` value was restored by hand on the
>   evidence below plus this section's own record of the original `[3, 78]` run.
> - `moe_i4_real_data_vs_fp8_ground_truth` had been **passing by skipping** the whole time
>   the stamp was absent (no stamp → early return). Restoring it made the test actually run.
>   A provenance-gated test reports "ok" when the provenance is missing; that is the gate
>   working as designed and is worth knowing when reading a green suite.
>
> **CORRECTION, 2026-07-31: the stamp had gone missing, and it was never the strongest
> evidence.** For a period that artifact's `manifest.json` carried no `i4_source` at all —
> a stamp is one JSON field and JSON fields go missing. The **slab length** does not:
> `ExpertSet::open` requires
> `len == (n_experts + 1) * i4_expert_stride`, and the stride is a function of the group
> size, so the file's 5,153,882,112 B = 257 × 20,054,016 identifies group 128 and nothing
> else (group 64 would be 21,233,664 per expert, per-row 18,915,328). Length is the
> discriminator; the stamp is the label on it. `main` no longer refuses an unstamped set —
> see the correction under §10 below.

```
rivoli <artifact> --ppl tests/ppl-corpus.txt --ppl-out out.nll \
    --mode int4 --cache-policy lru --max-mem 100 [--moe-gain <g>]
```

---

## 1. Read this first: the degeneration gate is not a quality metric

`docs/measurement/benchmarks.md` gates cells on the **distinct-token ratio** of a completion — "degenerate
greedy output = a severe bug, disqualified from ranking". That gate is not measuring model
quality, and this block shows it inverting.

**Within this block, the two metrics move in opposite directions across the gain sweep** —
that is the supported claim, and it is enough on its own.

*(A stronger-sounding claim — "distinct-ratio ranked int4 above hybrid" — does NOT come from
this block: **H's distinct-ratio was never measured** (§2, §7). The int4-above-hybrid ordering
is from the 2026-07-26 observation that opened this investigation, where int4 scored 0.074 and
hybrid 0.040 while PPL here ranks hybrid 6× better. Cross-session, no control named, so it is
recorded as context and not leaned on.)*

Worse, across the gain sweep the two metrics move monotonically in *opposite* directions:

| `--moe-gain` | 1.00 | 0.96 | 0.93 | 0.91 | 0.86 |
|---|---:|---:|---:|---:|---:|
| PPL | 73.4 | 96.3 | 115.5 | 145.8 | **216.4** |
| distinct | 0.126 | 0.203 | 0.260 | 0.265 | **0.324** |

Attenuating the MoE branch makes the model a **strictly worse predictor** (PPL triples,
+14.5 SE) while making its free-running text **visibly more diverse** (distinct 2.6×). The
gate would have passed the worst model in the block more readily than the best.

Distinct-ratio detects *repetition*. Repetition is one failure mode among many, and it is
suppressible by interventions that damage the model. **Use teacher-forced PPL to rank; use
distinct-ratio only to flag a run as unreadable, never to compare two runs.** This needs
revisiting on its own account, independent of int4.

The failure is not a simple sign flip, which is what makes it dangerous. On the *other* side
of the same knob the metrics agree: A5 (g=1.10) improves both PPL (73.4 → 46.1) and distinct
(0.126 → 0.277). So distinct-ratio is not reliably anti-correlated with quality — it is
**uncorrelated in a way that happens to look right about half the time**, which is exactly
the profile of a proxy that survives casual validation and then inverts a ranking.

---

## 2. The arms

One block, one binary, `--cache-policy lru --max-mem 100` unless stated, 762 predicted
tokens (`tests/ppl-corpus.txt`), free-running 256 tokens for `distinct`.

| arm | mode | knob | PPL | hit% | distinct | note |
|---|---|---|---:|---:|---:|---|
| **C** | int3-vq | — | **5.2754** | 73.60 | 0.474 | in-session coherence anchor |
| **A0** | int4 | g=1.00 | **73.4306** | 79.60 | 0.126 | the state under investigation |
| A1 | int4 | g=0.96 | 96.3148 | 80.54 | 0.203 | |
| A2 | int4 | g=0.93 | 115.5266 | 81.27 | 0.260 | = the artifact-rewrite equivalent |
| A3 | int4 | g=0.91 | 145.7549 | 82.47 | 0.265 | = the old set's measured centre |
| A4 | int4 | g=0.86 | 216.4227 | 84.62 | 0.324 | |
| **A5** | int4 | g=1.10 | **46.0831** | 77.09 | 0.277 | **amplifying HELPS, −7.7 SE** |
| **P** | int4 | pre-merge engine `41522f6` | **73.4306** | 79.60 | *not run* | **bit-identical to A0** |
| **M1** | int4 | `--max-mem 80` | **73.4306** | 70.52 | *not run* | **bit-identical to A0** |
| **H** | hybrid | — | **11.5518** | 78.29 | *not run* | PPL-only; see §7 |

`M2` (`--max-mem 118`) failed on its own: `rivoli_vmm_alloc(108869943296) failed (-2)`, a
101.4 GiB pool OOM. Not retried — M1 already settles invariance bit-identically.

Hit% rises monotonically with PPL across the whole gain sweep (79.6 → 84.6). Two
independently measured quantities moving together, neither fitted to the other. It is
routing collapse, visible **under teacher forcing** — teacher forcing pins the input tokens
but the router still runs on the model's own damaged hidden states.

---

## 3. What is exonerated, and how strongly

Four claims, four different strengths. They are not interchangeable, and one of them proves
much less than it looks.

| claim | evidence | strength |
|---|---|---|
| fp8 decode geometry | `gate_proj` `weight_scale_inv` is **[16,48]**, `down_proj` is **[48,16]** — mirrored, each tracking its own weight orientation. Read from the safetensors header by a script with **no rivoli code in path**. A transposed convention cannot satisfy both; `dequant_fp8`'s shape assert would fire and the conversion would die on the first expert. | **fully independent** |
| layer / expert mapping | **1275 projections** (25 layers × 17 experts × 3) cross-checking `.i4` against `.vq3` at the *same artifact coordinates*. Touches no fp8, no `dequant_fp8`, no `model.layers.{l}` string. All cos ≥ 0.90, worst 0.9582. | **could have failed, didn't** |
| GPU int4 kernel | Full `down(silu(gate·x) ⊙ up·x)` against an **f64 reference from the fp8 checkpoint**: rel-L2 0.2951, **gain 1.0009**, max_err/max\|ref\| 0.1603. Now `tests/kernel.rs::moe_i4_real_data_vs_fp8_ground_truth`. | independent of `matvec_i4` |
| this morning's merges | **P bit-identical to A0 — 0/762 tokens differ, max \|dNLL\| = 0.** | **conclusive** |
| arena / residency / binding | **M1 bit-identical to A0 across a 9pp hit swing** (70.52% vs 79.60%) — far more eviction and refill, byte-for-byte same output. | **conclusive** |
| `.i4` bytes = intent | **390** projections bit-exact (`--verify`, 13 layers × 10 experts × 3) **plus 54** from targeted per-layer runs (6 × 3 × 3). Two separate runs — `--verify` alone prints 390. | **circular by construction** — the audit re-derives through the same code the converter used. States self-consistency only. |

The last row is why the first two exist. **A check that resolves its inputs through the code
under test cannot see an error in that resolution.** Bit identity, R-against-ground-truth and
layer mapping were all instances. What breaks it is a consumer reaching the same data by a
different route — which is what `.vq3` provided.

Two controls together close the cross-session licence: C reproduces the historical int3-vq
PPL to **six decimals** (fixing harness, corpus, settings, `.vq3`, attention, routing, pool),
and P is bit-identical to A0 (fixing the entire engine). **Between the old-`.i4` and new-`.i4`
int4 runs, the artifact is the only variable** — which is the comparison §4 needs.

Stated precisely, because the looser form ("the only variable, full stop") is not licensed:
C and A0 differ in more than the artifact — different kernel (`launch_moe_expert_range_i4` vs
`..._range`), different slot offsets, 15.3 → 18.9 MB/expert and hence different pool geometry.
P fixes the engine *version* and M1 fixes *residency*; neither exonerates the int4 decode
*path*, which rests on a single-expert, single-`x` GPU test. That test is strong evidence, not
proof, and §5's mechanism is what makes a decode bug unnecessary as an explanation.

### Falsified: branch gain

Pre-registered before any arm completed, with a stated falsifier. The falsifier fired.
Attenuation is **monotonically harmful** — no basin, no interior optimum — and A2 landed
exactly on the artifact-rewrite equivalent (0.93), so this is a clean falsification and not a
near-miss on the wrong constant. A5 then showed the *opposite* tail helps (g=1.10 → PPL 46.1,
−7.7 SE): the branch is under-contributing, not over-contributing.

---

## 4. The confirmed inversion

The `.i4` set was rebuilt straight from fp8 (`bin/fp8_to_i4`, chain `fp8->int4`), replacing
`fp8 -> vq3 -> int4`. The new weights are **strictly more accurate** by every scale-sensitive
measure against f64 fp8 ground truth:

**The "old `.i4`" rows below are RECONSTRUCTED**, not read from the retired artifact — it was
overwritten in place. `bin/i4_audit` re-derives them as `quant_i4(vq_decode_proj(.vq3))`, which
is what `bin/vq3_to_i4` did. A sound proxy, but by this document's own §3 teaching it resolves
through some of the same code, so it is a reconstruction and is labelled as one.

> **2026-08-01: `i4_audit` no longer re-derives them.** The "old i4" rows and the `VQ_GAIN`
> attenuation pre-flight built on them were removed — they head-to-headed the shipped set
> against a generation nothing can produce any more (`bin/vq3_to_i4` is deleted), and the
> branch-gain hypothesis they fed is falsified outright by §3, "Falsified: branch gain"
> (attenuation is monotonically harmful; no interior optimum). **The rows below and every
> number they printed stand as recorded**, here and in `measurement/benchmarks.md` — a
> reconstruction that has already served its comparison does not need a standing tool. The
> live arms are unchanged: `--scan`, `--verify`, `--xcheck`, `--scale-study`.

| | rel-L2 (whole row) | bulk (\|w\| ≤ p99) | tail | gain |
|---|---:|---:|---:|---:|
| new `.i4` | **0.205** | **0.215** | **0.065** | **1.0008** |
| old `.i4` (vq3-derived) | 0.250 | 0.261 | 0.093 | 0.9766 |

Better on the whole row, better on the bulk, better on the tail, and unbiased where the old
was shrunk 2.3%. The two error stages close in quadrature to 0.2%
(`sqrt(0.250² − 0.159²) = 0.193` vs `0.205 / 1.063 = 0.193`), leaving no unexplained residual.

**Strictly better weights, 8× worse model.** That is not a paradox once §5 is read.

---

## 5. The mechanism: one scale for 6144 weights

`quant.rs::quant_i4` uses **one symmetric scale per output row**, `s = amax/7`, with
`q = round(w/s) + 8` clamped to `[0,15]`. Consequences:

- **A row is 6144 weights** (gate/up) or 2048 (down). One outlier sets the step for all of them.
- **Nibble 0 is unreachable** — `round(w/s) ∈ [-7,7]`, so 15 of 16 levels are used.
- The overload point lands at **4.3–4.9σ** (4.86σ inferred from the measured `amax/median`
  = 7.2 with `median|w| = 0.6745σ`; 4.31σ from `amax/p99.9` = 1.31 — the spread is the tail
  being heavier than Gaussian). The MSE-optimal overload for a uniform quantizer on a unit
  Gaussian is `(N/2)·Δ*` with `Δ* = 0.3352σ` (Max's optimum, derived for N = 16): **2.68σ at 16
  levels**; the 15 levels actually used give 7.5·Δ* = **2.51σ** by that formula. So the
  quantizer is loaded roughly **1.6–1.9× too wide** — which overlaps §6's *empirical* optimum
  (α ≈ 0.55–0.65 ⇒ 1.54–1.82×), and the empirical number is the one to trust: this σ chain
  assumes Gaussian tails, and §5's own measured zero-fraction (0.465) is ~1.7× what that
  assumption predicts, so the weights are visibly heavier-tailed than the model.

Everything below `s/2` rounds to nibble 8 = zero. Measured fraction of weights sent to zero:

| | mean zero-frac | rows past 50% zeros |
|---|---:|---:|
| `L03 e0 down_proj`, new `.i4` | 0.465 | **603** / 6144 |
| `L03 e0 down_proj`, old `.i4` | 0.435 | **115** / 6144 |

**5.2× more near-dead rows.** The means differ by only 7%, but the distribution sits right at
the threshold, so the count past it moves five-fold. (An earlier read of this table compared
the shared expert's counts, saw the same order of magnitude, and wrongly generalised — the
routed down-projections are where it bites.) No row is *entirely* dead; the worst observed is
0.9995.

**Why the old chain worked, by accident — and an open question that matters.** Re-quantizing
`.vq3` handed `quant_i4` an already-clipped `amax` (effective α ≈ 0.94, ~7% fewer weights
zeroed). The old `.i4` was not better designed; it was pre-conditioned upstream.

**But which upstream property did it?** Two candidates, and they are not distinguished here:
(a) `.vq3`'s scale per **64** weights, or (b) `quant_vq`'s **least-squares scale refit**, which
is MMSE-like and shrinks by `1 − relL2²` — measured 0.9766, and the mechanism this repo has
actually evidenced (`bin/i4_audit`'s `VQ_GAIN` — removed 2026-08-01, see §2's note; the
0.9766 is recorded in docs/measurement/benchmarks.md). MMSE shrinkage biting harder in
the tail than at the median would drop `amax/median` 7.2 → 6.8 with **no appeal to group size
at all**. This is load-bearing: if the benefit came from (b), then group-wise int4 *without* an
MMSE/LS scale fit will not reproduce it, and 365 GB gets rewritten on the wrong half of the
mechanism. **Cheap discriminator, run it first:** re-quantize one projection with group-64
*scalar* scales and no LS refit, and see whether `amax/median` and the zero-fraction move. §6's
recommendation rests on the industry-practice argument, which is independent of this — but the
*explanation* above should not be quoted as settled until that measurement exists.

**A5 is NOT explained by this, and the doc should not pretend otherwise.** The tempting story —
sparsification weakens the branch, so amplifying recovers it — is refuted by §4's own gain
column: the new `.i4` is measured at gain **1.0008**, i.e. *unbiased*, and its chain gains
(1.001 / 1.006 / 1.066) are *higher* than the old set's (0.894 / 0.949 / 0.902) which decoded
coherently. So g = 1.10 pushes the branch ~10% **above** fp8 ground truth and PPL improves by
7.7 SE. That is an anomaly, it is the one arm that moved PPL the right way, and it is
**unexplained**. Anyone continuing this should start there.

And it explains the mode ranking without appeal to gain, binding, or the merges. Hybrid is
better than int4 (11.55 vs 73.43) because its cold slab is `.vq3` at group-64.

---

## 6. The conclusion that supersedes tuning α

An α sweep (`s = α·amax/7`) was measured on real rows and does help — the optimum is
**α ≈ 0.55–0.65** in all 27 cells sampled, taking rel-L2 from 0.205 to ~0.13 and beating
`.vq3` by 17–28% on 24 of 27. **Do not implement it.** It is tuning a constant inside a
scheme that is coarser than anything in current practice.

**Group-wise scales at 32–128 weights are the industry standard.** For AutoGPTQ, 4-bit with
**group size 128** is the recommended configuration; AutoAWQ uses **asymmetric quantization
with group size 128**. On INT4 at group-128, AWQ cut the perplexity penalty from 4.57 to 1.17
— a ~74% reduction in degradation. Marlin-class kernels support 32/64/128 with in-kernel
dequant.

> **Our `.vq3` at group-64 is *finer* than the industry standard. Our int4 at per-row-6144 is
> far coarser than anything in practice.** That single fact explains the whole ranking.

*Sourcing:* the group-128 configurations and the AWQ 4.57 → 1.17 figure are confirmed
([Quantized Instruction-Tuned LLMs](https://arxiv.org/html/2409.11055v1),
[Quantization without Tears](https://arxiv.org/pdf/2411.13918)). A specific
per-channel-vs-group LLaMA-7B perplexity pair was cited to me second-hand and I **could not
verify it**, so it is deliberately not quoted here. Confirm exact group sizes against
whichever checkpoint is adopted rather than trusting this paragraph.

**Recommendation:** implement group-wise int4 (group 64 or 128, asymmetric) rather than tuning
per-row. Prefer importing a published int4 checkpoint for this model family over re-deriving
one — that would be an importer, not a re-derivation.

> **Superseded by §10.** Group-128 *symmetric* was implemented and shipped, taking int4 from
> PPL 73.43 to 5.120 without importing anything, so the import path was never needed and
> `bin/pack_i4` was never written — do not go looking for it (docs that reference one are
> stale). Asymmetric quantisation remains untested; see "Still open".

**What that touches** — this is not a converter-only change:

| file | why |
|---|---|
| `kernels/moe.hip` (`moe_gateup_i4`, `moe_down_i4`) | applies the scale **outside** the dot (`d.gate_scale[j]`). Group scales break exactly that. `dot_vq_wave` in the same file already does per-group scaling — use it as the template. |
| `src/quant.rs` | `i4_row_bytes`, `quant_i4`, `dequant_i4`, `matvec_i4`, `i4_proj_bytes`, `i4_expert_bytes`, `i4_expert_stride`, `i4_slot_offsets` all encode per-row layout. |
| `tests/artifact.rs` | asserts on-disk bytes equal `quant_i4`'s output; rewritten with it. |
| `src/bin/i4_audit.rs` | every mode slices via `off[k*2+1]`. |
| `docs/reference/modes.md` | group scales grow bytes/expert, moving the **18.9 MB/expert** figure — which sets pool slots and hit rate, i.e. the whole int4-vs-int3-vq residency tradeoff. |

**Acceptance gate.** `--mode int4` must reach PPL within ~10% of int3-vq's 5.28 on
`tests/ppl-corpus.txt` at `--max-mem 100`, and `bin/i4_audit`'s `>50%-zero` row count for
`L03 e0 down_proj` must collapse from **603**. If group-wise int4 cannot clear the first, the
mode should be removed rather than shipped.

---

## 7. Gaps and untested

- **α** — measured in weight space, never tested end-to-end. Superseded; do not spend on it.
- **Group-wise int4** — the actual recommendation, untested here.
- **H's distinct-ratio** — not run. `followup.sh` ran hybrid PPL-only. Not backfilled because
  docs/reference/modes.md makes hybrid's free-running numbers the least interpretable of the set (its
  numerics are a function of cache state).
- **P and M1 distinct** — not run; both are bit-identical to A0, so a generation pass would
  reproduce A0's text exactly.
- **Expert-dependent attenuation** — `--moe-gain` is uniform to ±0.1%; the old set scattered
  ±3.4% per expert. The gain falsification does not exclude a *non-uniform* attenuation story,
  though §5 makes it unnecessary.

## 8. Tools left behind

- `bin/i4_audit` — `--scan` (all 197,376,000 scales), `--verify` (wide bit-identity),
  `--xcheck` (`.i4` ↔ `.vq3`, the one that can fail), `--scale-study` (α sweep, weight and
  output space), plus dead-row and bulk/tail statistics.
- `tests/artifact.rs::i4_bytes_are_what_the_checkpoint_quantizes_to` — exact, CPU-only,
  provenance-gated on `i4_source`. (Still gated: a TEST may reasonably demand the label,
  since it is asserting what the bytes were derived *from*, which length cannot show. The
  ENGINE only needs the group size, which length does show — see the §10 amendment.)
- `tests/kernel.rs::moe_i4_real_data_vs_fp8_ground_truth` — independent f64 ground truth.
- `--moe-gain <g>` — MoE-branch gain, `vadd`-identical at 1.0. **Decision: kept, diagnostic
  only.** Its hypothesis is falsified (§3) and it is *not* a fix — A5's g=1.10 gain is an
  epiphenomenon of over-sparsification (§5), and "fixing" int4 with it would paper over the
  cause. Kept solely because it is the instrument behind five of the ten arms above, so
  deleting it makes the central falsification unreproducible. If the next agent disagrees,
  `git revert` the commit that added it — `launch_vadd` is already the g==1.0 path.

## 9. Operational traps hit (both the same lesson)

1. **`pkill -f "upper.sh"` matched its own invoking shell** — the pattern was *in* the command
   running it. Killed the whole process group mid-command.
2. **A process-pattern probe returned a false negative**, so a second copy of a 100 GiB job was
   launched on top of a still-running one, both writing the same output file.

> **A pattern probe returning nothing is not proof of absence** — exactly as `kill -0`
> returning success is not proof of health. Key progress on output *content*.

---

## 10. The fix, and the measurement that confirmed it

**Change.** `quant_i4` now emits one f32 scale per `I4_GROUP = 128` weights along the input
dim instead of one per output row, and the scale applies *inside* the dot — per group, in both
`dot_i4_wave` paths (the dword fast path and the scalar tail). `I4_GROUP` is a named constant,
so 64 (matching `.vq3`) can be swept without touching anything else.

**The discriminator, run before spending the rebuild.** §6 left one question open that a
365 GB conversion was riding on: did the old `.i4` behave better because `.vq3` carries
group-of-64 scales, or because `quant_vq` refits its scales by least squares? Those predict
opposite things about group-wise RTN. `i4_audit --scale-study` scores every candidate against
f64 fp8 truth (expert 7 `down_proj`, 6144×2048):

| scheme | W relL2 | zeros | gain |
|---|---:|---:|---:|
| per-row `amax/7` *(the shipped defect)* | 0.1521 | 20.73% | 1.0012 |
| LS refit (as `quant_vq`) | 0.1482 | 20.26% | 0.9994 |
| **GROUP-128 `amax/7`** | **0.1190** | **16.21%** | 1.0133 |
| per-row best α (oracle) | 0.1097 | 13.62% | 0.9881 |

**The refit buys 2.6%; group-128 buys 21.8%.** Group size was the mechanism. The old set
inherited its advantage from `.vq3`'s granularity, not from the refit — so group-wise scales
were the right build, and α tuning was correctly abandoned.

Note the study does *not* cleanly favour group-128 over per-row at α≈0.60 (0.1190 vs 0.1152).
Weight-space metrics could not decide this, which is consistent with §4: they never predicted
decode quality here. PPL decided.

**Confirmation** — one session, one binary, `--max-mem 100`, 762 teacher-forced tokens, plus
512-token free-running:

| mode | PPL | hit % | tok/s | distinct |
|---|---:|---:|---:|---:|
| `int4` | **5.120** | 64.40 | 2.01 | 0.279 |
| `hybrid` | **5.189** | 73.50 | **2.72** | 0.138 |
| `int3-vq` *(control)* | 5.275434 | 73.60 | 2.62 | 0.465 |

int4's completion went from a four-sentence verbatim loop to correct Rayleigh-scattering
physics. **And §1 lands one final time: hybrid has the worst distinct-ratio of the three
(0.138) and the second-best perplexity.** Its repetition is greedy-decode attractor behaviour
on a healthy model. A distinct-ratio gate would now reject the best config in the engine.

**A provenance gap closed while validating.** `I4Source` gained a `group` field and
`fp8_to_i4` wrote it, but nothing read it back. The engine indexes `scale[o*ngroups + i/G]`, so
a set quantised at a different `G` is a differently-*shaped* array — reading it does not fault,
it yields `rel_l2=NaN`, and the ground-truth oracle then reports "SYSTEMATIC gain error",
sending the reader after a numerics bug that is really a stale artifact. The engine and the
test now refuse, naming both group sizes and the remedy. Verified in both directions: refusing
the per-row artifact, passing the rebuilt one. *A provenance field that records without
enforcing is the failure it exists to prevent.*

> **AMENDED 2026-07-31.** That aphorism is right about a stamp that *disagrees* and wrong
> about one that is *absent*, and the code had conflated them: the `None` arm refused to
> load, which locked out the reference artifact — bytes provably correct — because a JSON
> field went missing. It also contradicted `format.rs::I4Source`'s own doc comment, which
> already said such a set "has a different `.i4` file size and is rejected by
> `ExpertSet::open`, so this is a diagnosis, not a load-time guard."
>
> The behaviour now matches that comment. **Unstamped** logs a line and defers to the slab
> length, which proves the group size exactly and before any expert is read. **A stamp that
> positively disagrees still bails** — that is a claim in conflict with the binary, not the
> absence of one. Sharpened: *a provenance field should enforce what it asserts, and assert
> nothing when it is silent.*

**Still open.** `I4_GROUP = 64` is untested and is what `.vq3` uses — worth one sweep, since it
costs another ~6% in size and the engine is bandwidth-bound. Asymmetric quantisation (a
zero-point) is untested; the current scheme is symmetric and uses 15 of 16 codes. Neither was
needed to close this.
