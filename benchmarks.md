# Benchmarks

512-token greedy decode, GLM-5.2 full artifact (`/var/db/rivoli/glm52-vq3-full`),
AMD Strix Halo gfx1151. Matrix: `--mode {int3-vq,int4,hybrid}` × `--cache-policy
{lru,2q,arc}`, 9 runs.

**Fixed across every run** (for comparability): same prompt
(*"Explain, step by step, how a transformer neural network processes a sentence."*),
`--attn dense`, `--max-mem 115`. Only `--mode` and `--cache-policy` vary.
Binary: release + `--features rocm`. GPU sole-tenant (k3s stopped).

## Results

Degenerate output (greedy collapse into repetition) is treated as a **severe bug and
a FAIL**, not a data point — a degenerate run routes to the same few experts, which
*inflates* hit% and tok/s, so the broken run looks fastest. Runs are gated on output
quality (distinct-token ratio of the completion) *before* their speed counts.

| mode | policy | tok/s | hit % | output | verdict |
|---|---|---:|---:|---|---|
| int3-vq | lru | 2.67 | 78.0 | coherent (distinct 0.74) | ✅ ok |
| int3-vq | 2q  | 2.68 | 77.9 | coherent (distinct 0.74) | ✅ ok |
| int3-vq | arc | — | — | — | ❌ **CRASH** |
| int4 | lru | ~~3.50~~ | ~~91.9~~ | **degenerate (distinct 0.04)** | ❌ **BUG** |
| int4 | 2q  | ~~3.28~~ | ~~91.2~~ | **degenerate (distinct 0.04)** | ❌ **BUG** |
| int4 | arc | — | — | — | ❌ **CRASH** |
| hybrid | lru | 2.89 | 83.0 | coherent (distinct 0.57) | ✅ ok |
| hybrid | 2q  | 2.55 | 80.7 | coherent (distinct 0.60) | ✅ ok |
| hybrid | arc | — | — | — | ❌ **CRASH** |

**5 of 9 runs fail** *as originally run.* int4's struck-through tok/s are *disqualified*
— they are the degeneration artifact, not real throughput. **int4's degeneration was
later root-caused to the colibri `.i4` source and FIXED** (see BUG 1); with vq3-derived
`.i4`, `--mode int4` decodes coherently. `arc` still crashes (BUG 2, open).

### Valid runs only — ranked

Among the runs that produced coherent output, best first:

| # | mode | policy | tok/s | hit % |
|---|---|---|---:|---:|
| 1 | hybrid | lru | **2.89** | 83.0 |
| 2 | int3-vq | 2q | 2.68 | 77.9 |
| 3 | int3-vq | lru | 2.67 | 78.0 |
| 4 | hybrid | 2q | 2.55 | 80.7 |

`hybrid + lru` is the only coherent config that clears int3-vq — it keeps the frequent
experts in int4 (faster compute) while streaming the rest as accurate int3-vq, and the
byte-arena fits ~6902 slots vs all-int4's 5596. Note `hybrid + 2q` is *slower* than
int3-vq here; among coherent runs the policy interaction is small and inside run-to-run
routing noise. Do not read fine tok/s differences as decisive (see caveat).

### Per-token profile (valid runs, ms/tok)

| mode/policy | wall | route | moe (gpu) | fetch (hidden) | miss/tok | GB/tok |
|---|---:|---:|---:|---:|---:|---:|
| int3-vq / lru | 375 | 115 | 242 (233) | 188 (95%) | 131.9 | 2.02 |
| int3-vq / 2q  | 373 | 114 | 237 (228) | 186 (95%) | 132.7 | 2.04 |
| hybrid / lru  | 346 | 114 | 215 (206) | 152 (94%) | 101.9 | 1.56 |
| hybrid / 2q   | 392 | 115 | 254 (244) | 192 (95%) | 115.6 | 1.77 |

Fetch is ~95% hidden behind compute in every valid run — the engine is compute-bound
(route + moe-gpu), not fetch-bound, at this budget. hybrid/lru's edge is fewer
misses/tok (101.9 vs ~132) → less exposed fetch and less MoE work.

---

## Severe bugs found

### BUG 1 — `int4` mode degenerates (all-experts int4)

`--mode int4` collapses into verbatim repetition within the 512 tokens:

> *"# The following is a simple example of how to use the neural network to process a
> sentence and how to use it to train them # The following is a simple example of how
> to use the neural network to process a sentence…"* (repeats to EOS)

distinct-token ratio **0.04** (vs ~0.74 for the coherent int3-vq baseline on the
identical prompt). Reproduces under both `lru` and `2q`, so it is **not** policy-related.

**Root cause: the `.i4` experts were sourced from the wrong checkpoint.** The int4
compute path is bit-correct (GPU kernel matches CPU `matvec_i4` on real bytes to cosine
1.0000 — test `moe_i4_real_data_matches_cpu`; no blowup/NaN in a residual-norm trace).
The defect is the *data*: `.i4` was built by `pack_i4` copying **colibri's** int4, which
is a different/worse quantization of the experts than the **vq3** the rest of the model
(attention, router, embed) comes from. Reconstructing actual weight rows and regressing
proved it:

| int4 source | fidelity vs vq3 (R) | per-row scale |
|---|---|---|
| colibri (`pack_i4`) | **0.96** | 5–9% inflated |
| vq3 self-requant (`quant_i4` of the faithful weights) | **0.98** | matches `amax/7` |

colibri's experts are ~4% off the vq3 experts the glm52-fp8 router was routing for;
running that mismatch under greedy decode compounds into repetition collapse. (cosine
alone hid it — cosine is scale-blind; the regression + self-requant exposed it.)

**Fix — `bin/vq3_to_i4`: re-derive `.i4` from our faithful `.vq3` weights** (decode each
expert to f32, re-quantize with `quant_i4`), replacing colibri as the `.i4` source.
After regenerating the artifact's `.i4`, `--mode int4` decodes **coherently**:

> *"…Use the sentence 'The cat sat on the mat' as an example. We need to explain, step
> by step, how a transformer neural network processes a sentence…"*

hit rate 73% (coherent-level, vs the degenerate 92%), no repetition. `pack_i4` is now
deprecated as the `.i4` source. `--mode int4` is un-gated (it works). Hybrid benefits
too — its int4 hot experts are now the self-consistent vq3-derived set.

**Hybrid was never broken (a 24-token probe misled me).** Over a warm 128-token run,
hybrid's int4 share of routed launches climbs as the pool fills (11%→17%+), i.e. it
*does* promote hot experts into the int4 tier; the ~6900-slot pool just needs ~60 tokens
to fill before anything cycles through the 2Q ghost to promote.

### BUG 2 — `--cache-policy arc` crashes in every mode

```
Error: expert not resident after alloc (batch exceeds pool — raise --max-mem)
```

(`pin.rs:836`) — after a batch's misses are admitted+placed, some batch key is not
resident. It is **not** a real capacity problem: the pool built fine (~5596–6900 slots)
and a per-layer batch is only ~9 experts; `lru` and `2q` handle the identical budget
without issue.

`HybridArc` (`src/hybrid.rs`) is the only policy that fails, and its `Spec` unit test
passes — but that test models single-tier byte accounting, **not** the two-ended arena
(cold packs from the low end, hot from the high end, split floats, cross-tier grow
compacts). So the bug is in the arc↔arena interaction the unit test doesn't cover:
arc's `evict_until_fits` frees enough *total* bytes but its `protect`-is-a-no-op +
`p`-driven tier choice can leave a batch slot unplaceable in the arena (wrong-end holes
compaction can't resolve, or a just-admitted/hit slot evicted within the batch).
`lru`/`2q` don't hit it because their eviction stays within the entered segment.

**Fix direction:** make `HybridArc` respect the same per-batch protection `2q` uses
(don't evict keys touched this batch) and evict from the tier whose arena end actually
needs to shrink; add an arena-backed replay/unit test so the invariant is covered off-GPU.

---

## Measurement caveat (why the gates matter)

Free-running greedy `tok/s` **cannot** rank modes on its own — this run is the proof:
`int4` posted the highest tok/s (3.50) and highest hit% (91.9%) purely *because* it
degenerated. Always gate on output quality first, then compare speed among survivors.
For a trustworthy speed number use a fixed forced-token bench; for residency use
`replay <trace> <n_slots> [--sweep]`; for pure per-format compute use
`examples/dot_bench.rs`. See [MODES.md](MODES.md).

*Generated 2026-07-25. Reproduce: `--mode <m> --cache-policy <p> -bench 512 --attn dense
--max-mem 115 --prompt "<above>"`.*
