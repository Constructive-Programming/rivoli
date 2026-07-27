# Benchmarks

512-token greedy decode, GLM-5.2 full artifact (`/var/db/rivoli/glm52-vq3-full`),
AMD Strix Halo gfx1151. Matrix: `--mode {int3-vq,int4,hybrid}` × `--cache-policy
{lru,2q,arc}`, 9 runs.

**Fixed across every run** (for comparability): same prompt
(*"Explain, step by step, how a transformer neural network processes a sentence."*),
`--attn dense`, `--max-mem 115`. Only `--mode` and `--cache-policy` vary.
Binary: release + `--features rocm`. GPU sole-tenant (k3s stopped).
`.i4` experts are the **vq3-derived** set (`vq3_to_i4`); see "int4 provenance" below.

## Results — all coherent, no crashes

Output quality is gated first (degenerate greedy output = a severe bug, disqualified
from ranking) via the distinct-token ratio of the completion. Every cell passed.

| mode | policy | tok/s | hit % | distinct | output |
|---|---|---:|---:|---:|---|
| int3-vq | lru | 2.76 | 78.0 | 0.74 | ✅ coherent |
| int3-vq | 2q  | 2.77 | 77.9 | 0.74 | ✅ coherent |
| int3-vq | arc | 2.77 | 77.9 | 0.74 | ✅ coherent |
| int4 | lru | 2.28 | 75.9 | 0.62 | ✅ coherent |
| int4 | 2q  | 2.39 | 76.3 | 0.62 | ✅ coherent |
| int4 | arc | 2.29 | 76.0 | 0.62 | ✅ coherent |
| hybrid | lru | **2.85** | 80.6 | 0.66 | ✅ coherent |
| hybrid | 2q  | 2.66 | 76.7 | 0.65 | ✅ coherent |
| hybrid | arc | 2.51 | 75.7 | 0.58 | ✅ coherent |

**9/9 pass.** (An earlier run of this matrix had 5/9 failures — int4 degenerated and
arc crashed; both are now fixed, see "Bugs found and fixed" below.)

### Ranked (all coherent)

1. **hybrid / lru — 2.85 tok/s** (80.6% hit) — the fastest coherent config.
2. int3-vq / 2q · arc — 2.77
3. int3-vq / lru — 2.76
4. hybrid / 2q — 2.66
5. hybrid / arc — 2.51
6. int4 / 2q — 2.39
7. int4 / arc — 2.29
8. int4 / lru — 2.28

**hybrid+lru wins** — the byte-arena packs the highest effective residency (80.6% hit,
fewest misses/tok), and its hot experts run int4's faster compute. **int3-vq is
policy-insensitive** (2.76–2.77 across all three). **all-int4 is the slowest** despite
faster per-expert compute: its 18.9 MB experts (vs vq3's 15.3 MB) fit fewer pool slots →
more misses → more fetch + more MoE work. `hybrid+arc` (2.51) trails `hybrid+lru` — arc's
adaptive split holds a smaller working set here.

### Per-token profile (ms/tok)

| mode/policy | wall | route | moe (gpu) | fetch (hidden) | miss/tok | GB/tok |
|---|---:|---:|---:|---:|---:|---:|
| int3-vq / lru | 363 | 114 | 232 (223) | 177 (95%) | 131.9 | 2.02 |
| int3-vq / 2q  | 361 | 115 | 226 (217) | 175 (95%) | 132.7 | 2.04 |
| int3-vq / arc | 361 | 115 | 226 (217) | 174 (95%) | 132.4 | 2.03 |
| int4 / lru | 439 | 109 | 310 (301) | 255 (96%) | 144.5 | 2.22 |
| int4 / 2q  | 419 | 110 | 285 (277) | 243 (97%) | 142.4 | 2.18 |
| int4 / arc | 437 | 110 | 305 (296) | 256 (97%) | 144.2 | 2.21 |
| hybrid / lru | 351 | 115 | 220 (210) | 159 (94%) | 116.2 | 1.78 |
| hybrid / 2q  | 375 | 115 | 242 (233) | 194 (95%) | 139.7 | 2.14 |
| hybrid / arc | 398 | 116 | 262 (253) | 218 (96%) | 145.8 | 2.24 |

Fetch is ~95% hidden behind compute everywhere — the engine is compute-bound (route +
moe-gpu), not fetch-bound, at this budget. hybrid/lru's edge is the fewest misses/tok
(116 vs ~132–145) → lowest fetch and lowest MoE wall.

---

## Bugs found and fixed

### int4 degeneration — WRONG `.i4` SOURCE (fixed)

`--mode int4` used to collapse into repetition from token 0 (distinct-token ratio 0.04).
The int4 compute path is bit-correct (GPU kernel matches CPU `matvec_i4` on real bytes to
cosine 1.0000 — test `moe_i4_real_data_matches_cpu`; no blowup/NaN). The defect was the
*data*: `.i4` was built by `pack_i4` copying **colibri's** int4, a different/worse
quantization of the experts than the **vq3** the rest of the model uses — reconstructing
weight rows and regressing showed colibri int4 at R≈0.96 vs vq3 (per-row scales 5–9%
inflated) vs a vq3 self-requant at R≈0.98. The mismatched experts, run under the
glm52-fp8 router, compound into greedy collapse. (cosine hid it — cosine is scale-blind.)
Fix: `bin/vq3_to_i4` re-derives `.i4` from the faithful `.vq3` weights; `pack_i4` is
deprecated as the `.i4` source. int4 now decodes coherently (rows above).

### `arc` crash — batch-eviction (fixed)

`--cache-policy arc` used to crash (`expert not resident after alloc`, `pin.rs`) — and so
did int4/lru at 512 tokens. General bug in all three policies: `submit_layer` protects
each hit then admits each miss, but a miss's eviction could reclaim a key touched earlier
in the *same* batch (a prior hit or admitted miss), which the pin then can't resolve. arc
triggered it readily (adaptive `p` drives one tier small enough for a 9-expert batch to
drain past its MRU end); int4/lru hit it via eviction pressure (bigger experts → fewer
slots). Fix: each policy keeps a per-batch `pinned` set (`begin_batch` clears it, protect
+ admit add to it); `OrderedSet::{peek,pop}_lru_skip` skip pinned keys during eviction.
All three arc cells and int4/lru now run clean (above).

### int4 provenance — MEASURED, and it inverts hybrid's stated premise

These int4/hybrid numbers use `.i4` re-derived from **vq3** (itself a lossy 3-bit
quantization). `bin/vq3_to_i4` does this deliberately: colibri's own int4 was a mismatched
per-row RTN quantization (R≈0.96 against vq3, scales 5–9% inflated) that made all-int4
decode degenerate under the fp8 router. So the chain in the artifact anyone actually runs
is **fp8 → vq3 → int4**, and **the int4 set cannot be better than the vq3 it was derived
from, by construction.**

That is no longer just a caveat — it is measured. Teacher-forced perplexity on a fixed
762-token corpus, `--max-mem 100`, LRU, no substitution:

| mode | PPL | hit% |
|---|---:|---:|
| int3-vq | **5.275** | 73.67% |
| int4 | **9.083** | 69.67% |

int4 is **72% worse in perplexity** before any cache policy touches it. This is the
arithmetic of double quantization, not a surprise — but it inverts the design rationale on
record for hybrid mode. Hybrid is described as putting the hot set in int4 to buy accuracy
along with int4's ~1.8× compute. **In this artifact int4 has no accuracy to offer**: it is
strictly a re-quantization of the vq3 set, so hybrid currently trades quality *away* for
compute rather than buying quality with it. `docs/CACHE_ROUTE.md` carried the same inverted
claim ("int4 is both more accurate and ~1.8× faster") and has been corrected.

Every int4 or hybrid quality number in this file must be read as *this artifact's* int4,
not as rivoli's int4.

**The fix path, which now exists.** The colibri sister project switched its converter to
**group-scaled int4 (gs64) by default** at commit `21cbc29` (2026-07-24), for exactly this
defect: per-row int4 measured **−9.3pp mean acc_norm** against **−2.2…−3.4pp** for
group-scaled. A gs64 container would plausibly give a faithful int4 genuinely better than
vq3 and restore hybrid's premise. Recommended source:
`mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`. That is a **`pack_i4` job, not a
`vq3_to_i4` job** — `pack_i4` imports a colibri container directly, which is the whole
point, since the defect that deprecated it was per-row scaling and gs64 removes it.
Until then, `--mode int4` and `--mode hybrid` quality numbers are bounded above by vq3.

### `quant_i4`'s `amax/7` is loaded ~1.8× too wide — and that, not provenance, is int4's deficit

The `.i4` set was rebuilt straight from fp8 (`bin/fp8_to_i4`, chain `fp8->int4`), removing
the second quantization stage. The weights got measurably closer to ground truth **and
decode quality got worse**, which is the shape of a bug that better weights unmask. It is
not one. `bin/i4_audit` measures the whole path against the ORIGINAL fp8 checkpoint in
f64 — never against `matvec_i4`, so no convention the producer and consumer share can
cancel — and every hypothesis that would have made it a defect is refuted:

| check | result |
|---|---|
| on-disk bytes == `quant_i4(dequant_fp8(ckpt))` | **bit-exact**, routed and shared, all 3 projections (now `tests/artifact.rs`) |
| all 197,376,000 per-row scales in the set | 0 non-finite, 0 zero, 0 negative, 0 `amax==0` dead rows; range 2.35e-5 … 4.51e-1 |
| new vs old `.i4` vs fp8 truth, whole row | rel-L2 **0.205 vs 0.250** — new is strictly better |
| … restricted to the BULK (`\|w\| ≤ p99`), same positions | **0.215 vs 0.261** — new is better *there too* |
| … restricted to the TAIL | **0.065 vs 0.093** — and there |
| per-row `amax/median` (fp8 vs vq3-decoded rows) | 7.2 vs 6.8 — the fp8 step is **6.3% coarser**, as predicted |

So the "fp8 keeps outliers, coarsens the step, wrecks the bulk" mechanism is **real in its
premise and wrong in its conclusion**: the coarser step costs ~6%, and dropping a whole
quantization stage buys far more. The errors add in quadrature and close exactly —
`sqrt(0.250² − 0.159²) = 0.193` against `0.205 / 1.063 = 0.193`, agreement 0.2%, leaving
no unexplained residual for a defect to hide in.

**The one systematic difference is GAIN.** `quant_vq` refits its scale by least squares,
so it is MMSE-like and shrinks: gain `= 1 − relL2²` (measured 0.9766 vs predicted 0.9754).
`quant_i4` is plain round-to-nearest and is unbiased (gain 1.0000). The old `.i4`
inherited vq3's shrink; compounded over gate‖up‖down and silu that is **~9% on the whole
expert chain** (0.921 vs 1.007). Every configuration that ever decoded coherently ran a
~9%-attenuated MoE branch; the new set is the first at full gain. That is a real change in
the model, and it is the *only* one — but it is a property of the quantizers, not a bug.

**The actual defect is the loading factor, and the fix is one constant.** `s = amax/7`
puts the quantizer's overload point at ~4.6σ; the MSE optimum for a 15-level uniform
quantizer on Gaussian-ish data is ~2.7σ. Sweeping `s = α·amax/7` against fp8 truth over
27 cells (layers 3/40/77 × experts 0/128/shared × 3 projections):

| α | 1.00 (shipped) | 0.80 | 0.70 | **0.60** | 0.50 | vq3 |
|---|---:|---:|---:|---:|---:|---:|
| rel-L2 (L3 e0 gate) | 0.2054 | 0.1648 | 0.1461 | **0.1314** | 0.1304 | 0.1589 |
| gain | 1.0008 | 0.9989 | 0.9971 | **0.9907** | 0.9761 | 0.9766 |

The optimum sits at **α = 0.55–0.65 in all 27 cells** (gate/up 0.55–0.60, down 0.65), and
a per-row search buys only ~4% over a single global constant — so no percentile, no sort,
no tunable. At α = 0.60 int4 beats vq3 by **17–28% in rel-L2 on 24 of 27 cells** (the three
weak ones are the shared expert of the late layers, where it merely ties), and the
output-space error `y = W·x` moves the same way, so this is not a weight-space artifact.
`quant_i4` already clamps to `[0,15]`, so a smaller `s` saturates correctly with no other
change.

**This inverts the recommendation above.** int4 is not bounded above by vq3 because of
double quantization — it was bounded above by an absmax scale set 1.8× too wide, and the
`fp8->int4` set already removed the other half of the problem. Importing a gs64 container
is no longer the only route. Not yet implemented: the measurement is the deliverable, and
a quality run must confirm it before 365 GB is rewritten.

**Two verification gaps this closed.** `moe_i4_real_data_matches_cpu` compares our kernel
to our own `matvec_i4`, so a convention both share is invisible to it; and no test asserted
what the bytes MEAN. `tests/artifact.rs::i4_bytes_are_what_the_checkpoint_quantizes_to` is
now the exact gate (bit identity, CPU-only, provenance-checked), and
`tests/kernel.rs::moe_i4_real_data_vs_fp8_ground_truth` is the coarse independent one.
The latter is *deliberately* coarse — two aggregate statistics over 6144 outputs cannot see
corruption confined to a few percent of rows, which is what the sibling test's max-abs is
for. Its doc says so rather than claiming a resolution it does not have.

---

## `top-m` offline screen (CACHE_ROUTE, arXiv:2412.00099)

Offline replay of captured v2 routing traces under cache-conditional substitution. **No
engine change is involved** — this is `bin/replay` over a fixed trace, so it is free of
the decode-trajectory confound below. Three 512-token captures, one per mode, same prompt
as above, `--attn dense`, `--max-mem 100` (not 115 — the node was shared). Each trace is
39,600 routing decisions = (16 prompt + 512 generated) × 75 MoE layers. Policy LRU.

`J` = sacred prefix (always selected, resident or not). `M` = candidate window eligible
for residency promotion. `swap%` = share of chosen slots outside the true top-K, i.e. the
quality cost. The `M = top_k = 8` control column reproduces each baseline to +0.00pp at
0.0% swap, which is the invariant proving the substitution is driven by the real router
ranking.

| mode | slots | baseline | J=2, M=12 *(paper defaults)* | J=4, M=10 *(cheapest passing)* | J=1, M=32 *(max, not a recommendation)* |
|---|---:|---:|---|---|---|
| int3-vq | 5,870 | 72.70% | **+15.24 pp** (17.8% swap) | +8.93 pp (9.6%) | +24.03 pp (38.7%) |
| int4 | 4,744 | 71.15% | **+15.05 pp** (17.6% swap) | +8.67 pp (9.4%) | +25.37 pp (37.7%) |
| hybrid | 5,852 | 74.35% | **+15.13 pp** (17.3% swap) | +8.92 pp (9.4%) | +22.69 pp (36.3%) |

Relative miss removal at the widest window: 88.0 / 87.9 / 88.5% — well past the paper's
">50% cache-miss reduction", and essentially **mode-independent**.

**Effective pool size is the useful framing — and the swap figure travels with it, always
in the same sentence.** Hit rate is still climbing steeply with capacity in our operating
region (int3-vq: 4,744→66.42%, 5,852→72.60%, 8,000→81.13%, 12,000→90.37%), so converting
slots into hits is worth a lot. Read against that curve:

> `top-m` at J=2/M=12 buys what growing the pool from 5,852 to ~10,950 slots would — an
> **~1.9× effective pool — at 17.8% swap, and a MEASURED +3.63% perplexity** (see Quality
> below), which is 3.6× the ~1% acceptance bar. J=4/M=10 is worth ~1.4× at 9.6% swap, at a
> perplexity cost the data cannot yet resolve.

Pool growth is free; substitution is not. **J=2/M=12 — the paper's own defaults — is
therefore not a shippable operating point here**, and the residency headline must never be
quoted without that. 17.8% swap means nearly one chosen expert in five is not the one the
router asked for, and it now has a price attached.

**And we cannot simply buy those slots**, which is why the steep curve matters here
specifically: the box has ~120 GiB and this capture already ran at `--max-mem 100`. There
is no meaningful room to grow the pool into, so a policy that raises the yield per slot is
worth roughly what a pool we cannot build would be.

**CACHE_ROUTE's prediction that int4/hybrid would benefit more than int3-vq is NOT
SUPPORTED, and the mechanism is simpler than the prediction was:** absolute gain tracks
*headroom*. hybrid starts from the highest baseline (74.35%) and therefore wins least;
int4 starts lowest and wins most at the widest window. Slot size never enters into it.

Reading the modes against each other at their own slot counts is a trap: each mode decodes
its own trajectory, so the traces are different workloads. At *matched* capacity the int4
trace is ~4.7pp more cacheable than the int3-vq trace, which fully accounts for int4's
apparent resilience to having 19% fewer slots. With one trajectory per mode, no cross-mode
claim here should be leaned on — always compare at matched capacity, which `replay` now
prints by default for exactly this reason.

**What this screen does NOT say.** There is no quality term anywhere in it. `swap%` is a
proxy for the quality cost, not a measurement of it — a cell at 38.7% swap runs a
different expert than the model chose more than a third of the time, and the perplexity
consequence is unmeasured. The screen says "do not stop", not "ship (J=1, M=32)".

**M is capped at 32 by the capture, not by the method.** `TRACE_WINDOW` is 32, and
`bin/replay` clamps M to the recorded window width while the engine can rank as far as
`n_experts`. The two are clamp-for-clamp identical for every M ≤ 32, which covers the whole
grid above — but sweeping M past 32 requires recapturing with a wider `TRACE_WINDOW`, or
the simulator and the engine are no longer measuring the same policy.

### CACHE_PILOT: the offline oracle cannot screen it

Reported for completeness, and as a negative result about the *method*. A perfect
next-layer predictor reaches ~100% hit at every horizon and every mode — vacuously, since
a decision needs 8 keys and 8 admissions fit in any pool holding one batch. Its
speculative admissions equal the baseline's misses (int3-vq: 86,477 vs 86,478), so it is
the same bytes moved earlier, which restates CACHE_PILOT's thesis rather than testing it.
**The pilot's risk is recall, recall is unobservable offline, and LOOKA is its only gate.**

A modelled predictor (keeps the top `k` ranked true experts, fills the rest with
distractors from the ranks just outside the true set) prices the false positives, on the
int3-vq trace:

| recall | hit% | vs baseline | bytes vs baseline |
|---|---:|---:|---:|
| 4/8 (50%) | 84.18% | +11.48 pp | 1.74× |
| 6/8 (75%) — nearest colibri's measured 71.6% | 91.49% | +18.79 pp | 1.34× |
| 8/8 (100%) — the vacuous ceiling | 99.99% | +27.30 pp | 1.00× |

L+2 tracks L+1 within 0.07pp throughout. That means the *horizon* costs essentially
nothing in residency terms — but the model holds recall fixed across horizons, so it says
nothing about whether real recall survives the longer reach. That is exactly LOOKA's
question. **Do not read "L+2 is free" as the pilot's main risk being retired.** It is not
retired; it is unmeasured, and reaching further is precisely where a real predictor is
expected to lose recall. Treat every row as an upper bound: these errors are independent
across decisions, and a real predictor's are correlated.

### DECISION: `top-m` ships opt-in and UNCERTIFIED

The powered run, `int3-vq`, **5,184 teacher-forced positions**, `--max-mem 100`, shared
baseline, one process per cell. This is the run that decided the feature.

Baseline (lru): **PPL 4.130637**, hit 72.25%.

| cell | PPL | dPPL% | mean dNLL | sd | SE | 95% CI (nats) | worse% | hit% | swap% | verdict |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| J=4/M=9 | 4.15252 | **+0.529%** | +0.00528 | 0.2700 | 0.00375 | [−0.00207, +0.01263] | 52.3% | **77.69%** (+5.44pp) | 5.79% | **INCONCLUSIVE** — interval contains zero |
| J=4/M=10 | 4.16786 | +0.901% | +0.00897 | 0.3011 | 0.00418 | [+0.00077, +0.01717] | 54.3% | 81.47% (+9.22pp) | 9.80% | **COST ESTABLISHED, MAGNITUDE UNRESOLVED** |

**J=4/M=9 is what ships**, and its verdict is INCONCLUSIVE rather than "small cost
confirmed": the interval contains zero, so no cost is established at all, and it is equally
not certified within the bar. **J=4/M=10 is not ship-able** — its lower bound clears zero,
so its cost *is* real, and buying more text would refine that number without changing the
decision. Note J=4/M=10's point estimate (+0.901%) sits under the bar; it is excluded on
the upper bound and on the established-cost finding, not on the headline.

**The knob defaults are J=4/M=9, not the paper's J=2/M=12.** That matters because `top-m`
ships opt-in: a user who enables the policy without passing knobs would otherwise have
received the one configuration this program rejected (+3.63% on int3-vq, outright FAIL on
int4). The paper's values remain reachable explicitly.

**Shipped opt-in, not as the default.** The interval **contains zero**, so `top-m` is not
significantly worse than baseline; its upper bound of +1.27% overshoots the pre-registered
~1% bar, so it is not certified within budget either. The point estimate is half the bar —
what fails is the uncertainty, not the result. Promoting it to default needs ~12,840 tokens
(~3.4 h sole-tenant for baseline plus one cell), and at this point estimate it might still
miss.

**Relaxing the bar to the paper's own +0.1–3.0% band was considered and declined**, because
it would have passed immediately and the ~1% figure was fixed before any data existed.
Moving a threshold after seeing the result it fails is post-hoc. See `MODES.md`.

### The engine and the simulator implement the same policy — including one forward prediction

Checked before any quality number was trusted, because if it failed the entire offline
screen above would be measuring a policy the engine does not run. Two independent
implementations of the substitution rule — `bin/replay`'s `substitute` and the engine's
`route_into` — on **different text** (the screen's 512-token trace vs the perplexity
corpus):

| (J, M) | simulator | engine | |
|---|---|---|---|
| J=4/M=10 | +8.93pp hit, 9.6% swap | +8.98pp hit, 9.65% swap | retrodiction |
| J=2/M=12 | +15.24pp hit, 17.8% swap | +15.18pp hit, 17.62% swap | retrodiction |
| **J=4/M=9** | **+5.35pp hit, 5.7% swap** | **+5.44pp hit, 5.79% swap** | **forward prediction** |

Agreement to 0.09pp on hit and 0.2pp on swap. These are **deterministic counts over a run**,
not statistical estimates of a small effect, so they carry no power caveat and are not
subject to the ambiguity that limits the quality numbers.

**The third row is a different and stronger class of evidence than the first two.** Those
were retrodictions — cells the engine had already run, checked against the simulator
afterwards. J=4/M=9 the simulator predicted *before the cell existed*: it was chosen off the
offline grid precisely because it was the lowest-swap cell still clearing the residency
screen, and the engine then returned it to within 0.09pp. An offline model that makes a
successful **forward** prediction is what justifies using the screen to choose future (J, M)
without re-measuring every candidate on device — which matters, because each device cell is
~44 minutes and the offline grid is milliseconds.

### Quality — teacher-forced perplexity, and what it does NOT establish

762 predicted tokens, fixed corpus, one process per cell, paired per-token NLL.
`dPPL%` is the headline; the paired **mean dNLL ± SE** is the evidence.

`bin/ppl` reports one of four verdicts, and they are four different next actions rather than
a severity scale. **PASS** — upper bound below the bar; ship-able. **FAIL** — lower bound
above the bar; rejected. **COST ESTABLISHED, MAGNITUDE UNRESOLVED** — interval clears zero
but not the bar; the cost is real, its size is not known, and more text refines the number
without changing the decision, because "not demonstrably within budget" is already enough
not to ship. **INCONCLUSIVE** — interval straddles zero; nothing is established and more
text could genuinely change the answer. The last two are the pair worth keeping apart: one
says stop measuring, the other says measure more if you care, and flattening them is how a
decision gets relitigated later as "we never checked properly".

int3-vq — baseline PPL 5.275434, hit 73.67%:

| cell | PPL | dPPL% | mean dNLL | SE | 95% CI (nats) | worse% | hit% | swap% |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| J=4/M=10 | 5.32864 | +1.009% | +0.01003 | 0.01092 | [−0.01136, +0.03143] | 52.6% | 82.65% | 9.65% |
| J=2/M=12 | 5.46686 | +3.629% | +0.03564 | 0.01474 | [+0.00676, +0.06453] | 57.0% | 88.85% | 17.62% |

int4 — baseline PPL 9.083032, hit 69.67%. **Read with the provenance caveat above: this
artifact's int4 is vq3-derived and 72% worse in absolute PPL before any policy acts, so
these are not evidence about `top-m` in a well-quantized int4 mode.**

| cell | PPL | dPPL% | mean dNLL | SE | 95% CI (nats) | worse% | hit% | swap% |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| J=4/M=10 | 9.27330 | +2.095% | +0.02073 | 0.01206 | [−0.00290, +0.04436] | 54.1% | 79.95% | 9.55% |
| J=2/M=12 | 10.23659 | +12.700% | +0.11956 | 0.01730 | [+0.08565, +0.15347] | 61.7% | — | 17.6% |

**J=2/M=12 — the paper's own defaults — is DECIDED AND REJECTED. It does not get
re-measured.** It fails outright on int4 at +12.700% with the interval entirely past the
bar (6.91 SE above zero), and on int3-vq its lower bound is +0.68% around a +3.63% point
estimate (2.42 SE above zero). Nothing about a larger corpus rescues a cell whose interval
is already above the bar; more text would only tighten it around a failing value.

**The one FAIL in this program was not better measured than the cells around it — it was
just enormous.** A reader scanning a single FAIL beside several INCONCLUSIVEs will infer the
FAIL rested on stronger evidence. It did not: `int4 J=2/M=12` earned that label from a
+12.7% effect on a run that was underpowered by exactly the same margin as every other cell
here. The asymmetry is effect size, not measurement quality.

**Three of the four cells are UNDERPOWERED, and that is the headline.** One standard error
exceeds the 0.00995-nat bar (a 1% PPL change) in every cell, so at 762 tokens the
experiment cannot resolve the acceptance question at any point estimate. An underpowered
null is **not** evidence of no harm. Only `int4 J=2/M=12` is decided: its interval lies
entirely above the bar, a genuine FAIL at +12.7%.

**What survives: cost rises with swap, within a fixed quantization.** Measured against a
common baseline on the same text, so the two swap levels are directly comparable. In
int3-vq, 9.65% swap → +1.01% and 17.62% swap → +3.63%; the high-swap point is 2.42 SE above
zero, the low-swap point only 0.92 SE — so what is established is that the high-swap
configuration costs something real, and that cost grows faster than swap does. The same
shape appears in int4 (1.72 SE and 6.91 SE).

**The cost per unit of swap is quantization-dependent — established, but by one of the two
comparisons only.** Difference-of-differences between the arms at matched swap, independent
runs so `SE = sqrt(SE₁² + SE₂²)`:

| matched swap | int4 − int3-vq | SE of difference | |
|---|---:|---:|---|
| ~9.6% | +0.01070 | 0.01627 | 0.66 SE — **too noisy to contribute** |
| ~17.6% | +0.08392 | 0.02273 | **3.69 SE — significant past p<0.001** |

The high-swap pair carries this on its own. The low-swap pair contributes nothing: both of
its estimates are individually indistinguishable from zero, and the "roughly 2×" ratio that
can be read off them inherits that uncertainty rather than escaping it — an earlier draft of
this section claimed it and should not have.

**The durable conclusion is a transfer warning: a (J, M) validated on one quantization does
not carry over to another, and any future mode must be re-measured rather than inheriting a
setting.** What remains unresolved is the *magnitude* of the gap at the low-swap operating
point we would actually ship, which needs both arms powered — roughly twice the n of either
arm alone.

A mechanism suggests itself — a less faithful quantization has less quality headroom to give
away before substitution starts to hurt — and it fits the direction and the widening with
swap. It is not tested here, and it is worth noting that its plausibility is exactly what
made the unsupported low-swap ratio tempting in the first place.

---

## Per-kernel round: matched A/B, `examples/dot_bench`

Same instrument binary in both arms; the only difference is the kernels. Three
**interleaved** repeats (base/fix/base/fix/base/fix) so drift shows up as spread inside
an arm rather than as the effect. GLM-5.2 dims from the manifest: H=64, qk_head_dim=256
(nope 192 + rope 64), v_head_dim=256, kv_lora_rank=512, 78 layers.

**Controls first — kernels this branch does not touch, measured in the same runs.**
Without these the deltas below are unreadable:

| control | base (3) | fix (3) | Δ |
|---|---|---|---:|
| `lm_head` | 8128.3 / 8114.2 / 8127.7 µs | 8121.2 / 8118.9 / 8132.3 | **+0.01%** |
| `rmsnorm` | 7.7 / 7.7 / 7.9 µs | 7.7 / 7.7 / 7.7 | **~0%** |

**Noise floor ≈ 0.1% on the big kernels** (~7% on kernels of a few tens of µs, e.g.
`argmax`). Everything below is judged against that.

| kernel | base µs | fix µs | Δ | GB/s |
|---|---:|---:|---:|---|
| `mla_absorb` | 72.00 | **36.50** | **−49.3% (1.97×)** | 87.4 → **172.3** | †
| `mla_value` | 33.73 | **27.03** | **−19.9% (1.25×)** | 248.6 → **310.3** |
| `mla_attend` nr512 | 258.03 | **227.17** | **−12.0%** | — |
| `mla_attend` nr2048 | 876.30 | **778.53** | **−11.2%** | — |
| `o_proj` | 541.55 | **528.95** | **−2.3%** | 184.7 → **190.6** |

† **CORRECTION (2026-07-26, "DSA indexer round" below): 36.50 µs is a cache-resident
figure, and so is its 172.3 GB/s.** The rig replays one 14.7 MB `kv_b` weight, which Strix
Halo's 32 MB MALL serves; with 4 rotating copies the same kernel measures **45.64 µs**, and
the engine holds 78 distinct `kv_b`. **The A/B above is unaffected** — both arms replayed
the same single weight, so the −49.3% delta stands and is what this table was for. What is
wrong is using 36.50 µs as an absolute per-layer cost, which the `×78` projection below and
docs/PERF.md both do. The same defect is present in every absolute µs figure in this
section; only the deltas are safe.

Arms are non-overlapping for every row. o_proj is the weakest and was pooled over two
separate experiments (6 samples/arm) because its effect is close to the between-run drift
of its own baseline: all 6 base samples (534.2–548.1) sit above all 6 fix samples
(524.7–531.7).

### Per-token, from the microbench — the prediction, since superseded

×78 attention layers. Recorded as the *prediction* because the in-engine run below
measured something different, and the gap is the interesting part.

| | Δ/call | ×78 |
|---|---:|---:|
| `mla_absorb` | −35.50 µs | −2.77 ms |
| `mla_attend` | −30.87 µs | −2.41 ms |
| `o_proj` | −12.60 µs | −0.98 ms |
| `mla_value` | −6.70 µs | −0.52 ms |
| **predicted total** | | **−6.68 ms/tok** |

**Report kernel work in the unit the budget is denominated in.** A large multiple on a
72 µs kernel is a small number of milliseconds, and a subject line saying "1.97×" outlives
the body that qualifies it.

## In-engine confirmation — the number a merge decision rests on

`-bench 256 --mode int3-vq --cache-policy lru --max-mem 100 --attn dense`, fixed prompt,
**interleaved** base/fix/base/fix. Same binary except the kernels.

**Why `-bench` is a fixed-token bench here despite being greedy decode:** in `int3-vq`
residency cannot reach the numerics, and all four changes are bit-identical, so both arms
*must* decode the same tokens. Verified, not assumed — see below.

| run | wall | **route** | **moe-gpu** (control) | fetch | miss/tok |
|---|---:|---:|---:|---:|---:|
| base.1 | 368 | **112** | 232 | 204 | 157.36 |
| fix.1 | 382 | **103** | **253** | **224** | 157.36 |
| base.2 | 366 | **112** | 230 | 202 | 157.36 |
| fix.2 | 357 | **104** | 230 | 202 | 157.36 |

**Clean pair (base.2 / fix.2, both uncontaminated): `route` 112 → 104, control flat at
230, wall 366 → 357, 2.73 → 2.80 tok/s.** Miss counts identical to the decimal in every
run (157.36/tok, 118115 hit / 45085 miss, 73.8%), so the arms are comparable.

**Measured −8.5 ms in `route` against a −6.68 ms prediction — the microbench UNDER-predicted.**
That is the opposite of the direction assumed throughout this work ("microbench caches are
friendlier than the engine's, so treat it as an upper bound"), and it should be assumed
*less* here still, because the prediction used `mla_attend` at nr=512 while this run's
context only reaches ~272. **That assumption is not supported by this measurement, and no
mechanism for the surplus is offered here.** The decomposition below establishes where
*part* of it comes from and leaves the rest explicitly unexplained.

### Interleaving flipped the conclusion — the concrete case

**Round 1 alone reads as a 4% REGRESSION**: wall 368 → 382, and `moe-gpu` — the control —
moved 232 → 253, *further than the signal did*. Round 2 shows `fix.2` at moe-gpu 230 and
fetch 202, matching base exactly. `fix.1` was an I/O outlier; every contaminated number sat
in the fetch-coupled buckets while `route` read 103/104 in both rounds regardless.

Had the slot allowed only one round, the honest report would have been the opposite
conclusion — a bit-identical change apparently slowing the engine by 4%. Interleaved arms
are not a refinement here; they are the difference between the right answer and the wrong
one.

### Choosing a control bucket: `route` is insulated, `moe-gpu` is not

The control was badly chosen and it is worth writing down why, because the reasoning
generalises. **Attention runs entirely on resident weights**, so `route` never waits on the
streamer and is structurally insulated from fetch variance. **`moe-gpu` absorbs stalls on
streamed experts**, so it moves with NVMe and page-cache state for reasons having nothing
to do with the kernels under test — which is precisely what it did. A control has to be
insensitive to the noise source, not merely untouched by the change.

### End-to-end bit-identity — evidence the repo has no test for

The generated text is **byte-identical between arms** (1251 bytes, timestamps stripped):
**256 greedy argmaxes over a 154,880-way vocabulary, every one landing on the same token.**
A single-ULP shift anywhere in attention would eventually flip a near-tie and diverge the
sequence. Every kernel oracle in this repo compares at `1e-3 * mx + 1e-3`, two to three
orders of magnitude looser than bit-identity, while `attn.hip` states bit-identity as a
requirement ("greedy decode needs it"). **This run is the only end-to-end check of that
property that exists**, and it is a by-product of an A/B rather than a test. A golden-bits
test remains an open gap.

### Decomposition: `fp8_dot_strided` reaches further than o_proj

The −8.5 ms exceeded prediction, so a third arm isolated the cause: `nofp8` is identical to
`fix` except `fp8_dot_strided` reverted to the signed divide, with the MLA and attend
changes retained. Interleaved, 2 rounds.

| arm | route (r1, r2) | attributable to |
|---|---|---|
| base | 112, 112 | — |
| `nofp8` | 106, 106 | MLA + attend = **−6.0 ms** |
| `fix` | 103, 104 | fp8 helper = **−2.5 ms** |
| | | **total −8.5 ms** |

`fix` reproduced 103/104 across two independent sessions, and the parts sum to the whole.

**The shared-helper reach is confirmed. `fp8_dot_strided` is worth −2.5 ms, 2.5× the
−0.98 ms that o_proj alone accounts for** — because it is the shared helper behind *every*
fp8 block-scaled GEMV in `route`: `o_proj`, `q_a`, `q_b`, `kv_a` and the dense MLP. **This
matters for what gets optimised next: PERF.md described that lever as an o_proj fix, and it
is a route-wide one.** Any future change to this helper — load widening, x re-read tiling —
inherits the same multiplier.

**Verdict: the shared-helper mechanism accounts for most of the gap, not all of it.**
Against the prediction table above, per component:

| component | predicted | measured | surplus |
|---|---:|---:|---:|
| fp8 helper (predicted as o_proj alone) | −0.98 | **−2.50** | **+1.52** |
| MLA + attend (−2.77 −0.52 −2.41) | −5.70 | **−6.00** | +0.30 |
| total | −6.68 | −8.50 | +1.82 |

**The fp8 helper's extra reach explains 1.52 of the 1.82 ms — ~84%.** MLA+attend came in
0.30 ms over, which is at the edge of what a 1 ms-resolution bucket can resolve.

Two honest limits. Counting thread-iterations puts the non-o_proj fp8 GEMVs at ~0.5×
o_proj, predicting ~−1.46 ms where −2.50 was measured, so the *size* of the reach is not
fully derived even though its existence is now measured. And the prediction's attend term
came from nr=512 while this run averages shorter context, so MLA+attend's context-adjusted
expectation is below −5.70 and its true surplus is larger than +0.30. **The residue is left
unexplained.** No second mechanism is proposed for it: one unverified explanation is a
caveat, two stacked is a story.

### Caveat: `--max-mem 100`, and what transfers

This ran at `--max-mem 100` (to stay clear of concurrent agents), not the 115 behind the
351 ms profile at the top of this file — hence 157 miss/tok and 2.41 GB/tok here against
116 and 1.78 there, and wall ~366 vs 351. **`route` transfers** (112 measured vs 115
recorded; attention is resident-only and budget-insensitive). **`wall` and `moe` do not** —
they are dominated by the miss rate, which the budget sets.

### Convert to per-token even when you don't need the number — it forces contact with ground truth

The `mla_absorb` and `mla_value` rows above were first measured at **guessed dims**:
`run_mla` had been called with nope=128, vh=128, qh=192. The manifest says **192 / 256 /
256**. Those kernels read `H*nope*kvl` and `H*vh*kvl`, so the bench was moving **4.2 MB
where the engine moves 6.3 and 8.4** — a different working set, a different cache regime,
and a per-token figure that would have been wrong by a factor no reader could have
recovered from the report.

**Nothing in the measurement caught it. The conversion did.** The A/B was clean, the arms
were non-overlapping, the controls were flat, and the numbers were internally consistent —
a wrong-shape benchmark is still a perfectly self-consistent benchmark. What surfaced the
error was multiplying by call counts, because that required opening `manifest.json`, and
the manifest disagreed. That it happened to be *understating* the win is luck, not a
mitigating factor.

So: **do the per-token conversion as a matter of course, including when the ratio is the
only thing you plan to quote.** Its value is not the arithmetic. It is that the arithmetic
cannot be done without going and looking at what the engine actually runs. Any step that
forces contact with ground truth is worth more than the step's own output — and a
microbench, which fabricates its own inputs, has no other moment where that contact is
compulsory.

### The instruments agreed — and that is *why* the earlier refusal was right

The first o_proj measurement was taken without a matched baseline, and 185 (in-engine,
recorded in PERF.md) → 189 (microbench) was **not** reported as an improvement. The
matched run later measured the base at **184.7 GB/s** against that 185, `mla_value` at
**248.6** against its recorded 254, `mla_absorb` at **87.4** against 99, and `mla_attend`
at 258 µs × 78 = **20.1 ms** against the recorded ~20 ms. The instruments agree closely,
so the original comparison would have given roughly the right answer.

**It was still the wrong thing to do, and this is the strongest evidence in this file for
staging a matched arm even when the old number looks fine: agreement between two
instruments is something you DEMONSTRATE, and none of these agreements were knowable
before both arms had been run.** Had o_proj's true effect been the 2% it turned out to be
and the instrument gap been 3% the other way, the uncontrolled comparison would have
reported a regression as an improvement. The cost of staging the baseline was one extra
build; the cost of not staging it is unbounded and invisible.

### Closed questions

- **`rmsnorm`'s `dim3(1)` launch is not a problem.** A single workgroup on a 40-CU part is
  a striking thing to find on a hot path, and it is **7.7 µs — 0.05% of the `tail`
  bucket.** At hidden=6144 there is simply not enough work for the geometry to matter. Do
  not re-flag it from the launch shape alone; it has been measured.
- **`mla_value` was not a healthy reference.** PERF.md judged `mla_absorb`'s 99 GB/s
  against "`mla_value`'s 254" — but `mla_value` carried the same 64-bit divide, so the
  yardstick was depressed too. Post-fix: 172.3 vs 310.3, absorb still ~1.8× off its
  sibling, so its load-width restructure remains worth doing. **Check whether a reference
  point is itself healthy before measuring against it.**

### Open question: half of `tail` is in none of its kernels

`tail` measures ~16 ms/tok. Measured: `lm_head` **8.12 ms**, `argmax` **0.088 ms**,
`rmsnorm` **0.008 ms** — **~8.2 ms total, leaving ~7.8 ms unattributed.** Those are the
only three kernels in the bucket, so **`tail` cannot be fixed by optimising them**: the
best case on `lm_head` (117 → 256 GB/s) is 8.12 → 3.71 ms, ~4.4 ms.

Candidate, **named but not measured**: these rows time 60 back-to-back launches behind a
single sync, while the engine pays a `device_sync` and a logits readback *per token*, so
per-token launch/sync/readback overhead would land in the bucket and not in any row here.
That is a hypothesis with a plausible mechanism, which is exactly the status the four
per-kernel mechanism errors below started from. Measure before acting on it.

## Read the ISA before you book the device

**The GPU is the scarce resource here; the compiler is not.** hipcc will answer a large
class of kernel questions on the CPU, in seconds, with no queue — and it answers some of
them *better* than a bench would, because it gives you the mechanism rather than a number.
Both of these are CPU-only and need no device:

```sh
# 1. The gfx1151 ISA for a kernel translation unit.
hipcc --offload-arch=gfx1151 -O3 --cuda-device-only -S kernels/linalg.hip -o /tmp/k.s
awk '/^gemv_fp8_splitk:/,/^\.Lfunc_end/' /tmp/k.s > /tmp/kernel.s   # isolate one kernel
awk '/Inner Loop Header/,/s_cbranch_execnz/' /tmp/kernel.s          # isolate its hot loop

# 2. Registers, scratch, spills, occupancy.
hipcc --offload-arch=gfx1151 -O3 -Rpass-analysis=kernel-resource-usage -c kernels/attn.hip -o /dev/null
```

What they are good for, from the per-kernel round that produced them:

- **Instruction mix of the hot loop.** Count `v_` (VALU) against `global_load`/`ds_load`.
  A loop with 44 VALU ops around 5 FMAs is not memory-bound no matter what its GB/s says.
- **Whether a register array actually landed in registers.** `ScratchSize` and
  `VGPRs Spill` are the whole answer, and a spill silently converts a "move it to
  registers" optimization into a slowdown.
- **Divergence.** Count `s_and_saveexec_b32`.

### When you remove one cost, count the cost you may have added — in the same instrument

The divergence check caught a change that would have been a regression, and the way it
nearly got through is the point. Moving `mla_latent_attend`'s accumulator from LDS to
registers was supposed to delete an LDS read-modify-write, and it did:

| version | `s_and_saveexec_b32` | `ds_store` |
|---|---:|---:|
| baseline (`acc` in LDS) | 6 | 4 |
| `acc` in registers, bound `i < kvl` | **37** | 2 |
| `acc` in registers, bound `k < nacc` | **4** | 2 |

The success criterion — "did the `ds_store` go away" — is **green on the middle row**,
which is the version that added 31 exec-mask save/restores by predicating all 16 unrolled
steps on a lane-divergent bound. It would plausibly have been slower than the code it
replaced while displaying the exact signature of the win it was aiming for. A wall-clock
bench would not have attributed it either: you would have seen "slower" and suspected
register pressure, not the exec mask.

**So: whatever cost you set out to remove, measure the neighbouring costs in the same
pass.** Removing LDS traffic can add divergence; moving to registers can add spills;
unrolling can add instruction-cache pressure. The instrument that shows the win is
usually one grep away from the instrument that shows the offsetting loss.

### Two ways an instruction count lies

Both of these came up in one afternoon and both will recur:

- **Unroll factors differ between the versions you are comparing.** Normalize before
  quoting a ratio. `mla_absorb_fp8`'s loops were unrolled ×3 before the fix and ×2 after;
  the raw block sizes (498 vs 52) suggest ~10×, per iteration it is ~6×. Count a
  once-per-iteration op — `ds_load`, or the weight load — to recover the factor.
- **Guarded paths inflate the static count when the guard is not taken at real dims.**
  The same kernel's 498 instructions include a full 64-bit Newton-Raphson division behind
  `v_cmpx_ne_u64`, which is **dead** at GLM dims (`row` ≤ 24576 and `block` = 128 both fit
  in 32 bits). The static number is real and the dynamic cost is smaller. When a count
  spans a branch, say which side runs.

### The signed-division signature — grep for the right thing

**Do not conclude "no divide in the loop" by grepping for `v_rcp_iflag_f32`.** LLVM
strength-reduces a division by a loop-invariant runtime value into a magic multiply, so
the reciprocal disappears while the cost does not. What survives for a **signed** divide
is the quotient correction, and that is what to look for:

```
v_mul_hi_u32 / v_mul_lo_u32 / v_cndmask_b32 / v_max_i32 / v_xor_b32 / v_ashrrev_i32
```

`gemv_fp8_splitk` had eight of those per iteration around five FMAs, from
`scalerow[i0 / block]` where both operands are `int`. Replacing it with a shift (the fp8
tile is a power of two, and the launchers now enforce it) took the loop from 44 VALU to
29 with the memory ops unchanged at 7.

**A 64-bit divide is a different and much larger animal.** `size_t / int` promotes to a
64-bit unsigned division, which LLVM *cannot* fold to a magic multiply — it emits an
inline Newton-Raphson reciprocal (seed constant `0x5f7ffffc`) plus a 32-bit fast path
guarded by `v_cmpx_ne_u64`. `mla_absorb_fp8` had one of these in its `d` loop, from
`kvb_scale[(row / block) * sc_cols + ...]` with `size_t row`: **498 static instructions
around 10 memory ops.** Read such counts carefully — the 64-bit path is *not taken* at
GLM dims (both operands fit in 32 bits), so the static number overstates the dynamic
cost, and the honest claim is "a runtime division per iteration was removed", with the
magnitude left to measurement.

### Re-opening the load-widening dead end — and why that was legitimate

`kernels/common.hpp` records that widening the fp8 loads "was a wash". That note came
from `d5e5932`, whose own commit message says the GEMVs were **decode**-bound at the
time — and the LDS e4m3 LUT *in that same commit* removed the decode bound. **The
conclusion outlived the conditions that produced it.** That is the standard for re-testing
a logged dead end: not "let's try again", but a specific reason the original measurement
no longer applies.

The re-examination also narrowed the question. The ISA shows the x-side load is already
`global_load_b128` — LLVM vectorized the four `x[i0+k]` reads by itself, so "widen the
loads" has silently been half-done since before the note was written. Only the weight
side is `b32`, and at 4 fp8/lane that is 128 B/wave = exactly one cache line, which is
not obviously worth widening. The open question is therefore **x re-read amplification**
(every block streams all of x for its slice of weights), which is a different lever from
load width and was never what the dead end tested.

## Running these benches — detach anything multi-cell

**A GPU run longer than the agent harness's background-task lifetime must be detached into
its own process group, or a task reap kills the engine with it.** This is invisible from
the code and cost a cell before it was understood.

Concrete numbers from the run that hit it: a 5,185-token perplexity cell is **~44 minutes**
(~2,613 s of scoring plus ~100 s of pin build), and the harness stopped the task at
**~60 minutes**. One cell fits; two never can. The engine was a child of that task, so it
died with it — `base` had completed, `j4m9` was killed 12 minutes into scoring, `j4m10`
never started.

The fix is `setsid`, and it applies to any multi-cell sweep regardless of how the script is
invoked:

```sh
setsid nohup ./tests/ppl-sweep-powered.sh <out-dir> > resume.out 2>&1 < /dev/null &
disown
```

**Verify detachment rather than assuming it** — "I ran setsid" and "it is actually
detached" are different claims, and the second is the one that matters:

```sh
ps -o pid,ppid,pgid,cmd -C rivoli
# PID 2005651  PPID 2005649  PGID 2005649  -> own process group, not a harness child
```

If `PGID` equals the process's own `PID` (and `PPID` is not the harness shell), the run
will survive a task reap.

Two related traps from the same run, both of which produce a confident wrong number rather
than an error:

- **Watchers must be keyed on content, not existence.** `--ppl-out` creates its file
  *before* writing 5,184 lines, so a watcher testing `[ -f x.nll ]` can fire on a partial
  file and hand the analysis a truncated cell. Key on line count.
- **Do not re-run a completed cell "for consistency".** After a partial failure, mixing a
  surviving cell with relaunched ones is legitimate *because* of the prefix checksum below,
  not in spite of it. Reproducing a verified artifact costs ~44 minutes of sole-tenant
  device time to learn nothing.

---

## A first-failure build hid a second one, and the fix caught it in the wild

`build.rs` used to compile shaders in sorted order and abort on the first failure, so a
change breaking several shaders reported one. It was fixed to compile everything and fail
once with the whole list — and the confirmation arrived unprompted.

The fix was verified with a synthetic break (set `ROWS_PER_BLOCK = 6`, watch two `#error`s
report instead of one). Then, on the next rebase, a deliberate-break check that had
previously printed **DID NOT FIRE** for `argmax_reduce`'s power-of-two coupling started
firing correctly — because `append_kv`'s `#error` was no longer masking it. A real case,
not a constructed one, and the accumulation change is what surfaced it.

The general form is worth keeping: **a check that stops at the first failure reports a
floor, not a count.** "The build passes now", after fixing the one error it named, is a
weaker statement than it sounds — nothing established that error was the only one. The
same applies to any first-failure-abort harness. Ask of a check: *if there were three
problems, would this tell me three?*

---

## Measurement caveat

Free-running greedy `tok/s` cannot rank modes on its own: a degenerate run routes to the
same few experts → inflated hit% → artificially *fast* (the earlier int4 rows posted the
highest tok/s *because* they degenerated). Always gate on output quality first, then
compare speed among survivors. For residency use `replay <trace> <n_slots> [--sweep]`; for
pure per-format compute use `examples/dot_bench.rs`. See [MODES.md](MODES.md).

*Generated 2026-07-26. Reproduce: `--mode <m> --cache-policy <p> -bench 512 --attn dense
--max-mem 115 --prompt "<above>"`.*

---

## DSA indexer round: `examples/indexer_bench`

Instrument for the NPU-offload gates (docs/NPU.md M0/M1), gfx1151 sole tenant, 2026-07-26.
Interpretation lives in [docs/NPU.md](docs/NPU.md); the rows and the methodology are here.
`--attn dsa` dims from the manifest: index_n_heads 32, index_head_dim 128, index_topk 2048,
and **21 FULL indexer layers** of 78 (`indexer_types` is 21 full / 57 shared, so a
per-token figure is ×21, not ×78).

### Controls

All from the round's final run unless a superseded run is named.

| control | result |
|---|---|
| `o_proj` fp8 [6144×16384] vs the 528.95 µs / 190.6 GB/s recorded in "Per-kernel round" | **519.4 µs / 193.8 GB/s** (1.8%) |
| `index_score` nt=32768, 21 rotating key slabs vs one replayed slab | **237.2 vs 208.7 µs (1.14×)** — run at nt=32768 only; the ≤4k rows are launch-bound (GB/s *rises* with nt: 7.5 / 30.5 / 32.8 / 35.4), so the rotation is not what makes them what they are |
| `index_score` output read back — finite, varying | ok |

A fourth check, the score-D2H round-trip against seeded bytes, is an `assert!` in the rig:
it aborts the run on failure and prints nothing on success, so it is not a reported control.
It compares the first 8 elements only.

### Rows (µs per call, per full layer)

| kernel | µs | note |
|---|---:|---|
| indexer key path (`gemv_fp8` wk + `layernorm` + `rope` + `index_append`) | 15.32 | 20.48 with a sync per call |
| `gemv_fp8` wq_b [4096×2048] | 78.27 | 107.2 GB/s |
| `gemv_f32` weights_proj [32×6144] | 34.74 | 22.6 GB/s, 32 output rows — grid-starved |
| `index_score` nt=128 / 2048 / 4096 / 8192 / 16384 / 32768 | 4.4 / 17.2 / 31.9 / 59.2 / 115.0 / 239.1 | 35 GB/s at long context |
| host score D2H + CPU top-k + row upload, same contexts | 18.1 / 81.9 / 160.2 / 183.0 / 353.2 / 553.6 | distribution-dependent — see below |
| `gemv_fp8` q_b [16384×2048] | 213.35 | 4 rotating copies |
| `mla_absorb_fp8` | 45.64 | 4 rotating copies |
| MoE batch, 9 vq3 experts + reduce | 1261.88 | 138.0 MB → 109.4 GB/s |
| dense fp8 SwiGLU MLP | 1174.67 | |

### Three methodology lessons, all of which cost a wrong answer first

- **Replaying one weight measures the MALL, not the bus.** A single 33.5 MB `q_b` timed at
  **372 GB/s — above the 256 GB/s bus**, which is only possible from the 32 MB MALL. With 4
  rotating copies it is 213.35 µs (157 GB/s). The same defect moved `weights_proj` 19.4 →
  34.7 µs and `mla_absorb_fp8` 36.04 → 45.64 µs. **The 36.50 µs `mla_absorb` figure recorded
  in "Per-kernel round" above is therefore cache-resident** — sound for the A/B it was made
  for, wrong as an absolute per-layer cost. Rotate before quoting an absolute.
- **A window must contain all the independent work, not a subset.** Scoping the exact
  overlap window to "kv_proj + KV-append" gave 22.6 µs and refuted a design; the full set of
  selection-independent phase-1 launches is **291.25 µs** and clears it. Under-scoping is
  not conservative — it produces a confident false negative.
- **Comparison-driven host code is distribution-dependent.** The D2H + `topk_into` + row
  upload over 32768 scores totals **162 µs** on a tie-heavy array (superseded run `m0m1-v2`)
  and **554 µs** on a plausible heavy-tailed one (final run) — a 3.4× spread on what turned
  out to be the single largest cost in the analysis. Synthesise the distribution
  deliberately and say which one you used.

### A GPU∥GPU probe cannot answer a GPU∥NPU bandwidth question

`index_score` on the null stream against the MoE batch on a `hipStreamNonBlocking` stream
measured, in superseded run `m0m1-v2`, three arms: the two workloads timed apart summed to
**2505.6 µs**, the both-on-the-null-stream control ran in **2453.4 µs** (1.02× vs the sum —
so the serial arm was genuinely serial), and the concurrent arm ran in **2625.0 µs** (0.95×,
i.e. *slower* than serial). That result was determined before it ran:
`index_score` at nt=32768 launches 32768 workgroups and the MoE batch ~9000, so each alone
over-subscribes all 40 CUs and neither can finish sooner concurrently no matter how much
DRAM bandwidth is spare. It measures compute-unit contention. The probe was deleted rather
than left printing a confident 0.95×; the bandwidth question is answered arithmetically from
the GB/s rows instead.


### In-engine confirmation, `--attn dsa` (2026-07-26/27)

`--attn dsa --mode hybrid --cache-policy lru --max-mem 115 -bench 48`, sole tenant, with two
always-on buckets added to `dsa_select_layer`. Both ride joins the path already pays: the
indexer's GPU span comes from a HIP-event pair read behind the existing `device_sync`, and
the host clock starts *after* that sync so the GPU wait is not double-counted. Guarded to
`--attn dsa` — under misa the head-route syncs inside the event bracket, which would fold
host time into a GPU-timeline number.

| | run A | run B |
|---|---:|---:|
| prompt tokens / mean nt during decode | 2432 / 2456 | 5185 / 5209 |
| wall ms/token | **391** | **438** |
| route (post-selection attention + host routing) | 156 | 158 |
| moe wall (gpu) | 201 (192) | 242 (232) |
| indexer GPU ms/tok — µs/layer | 4.1 — 194.9 | 4.6 — 218.1 |
| indexer host ms/tok — µs/layer | 4.5 — 214.2 | 7.0 — 334.1 |
| scoring layers/token | 21.0 | 21.0 |
| tok/s · hit% · miss/tok · GB/tok | 2.56 · 81.4 · 111.4 · 1.71 | 2.28 · 76.9 · 138.9 · 2.13 |
| residual (wall − route − moe − indexer) | 25.4 | 26.4 |

Interpretation, and the extrapolations built on these rows, live in
[docs/NPU.md](docs/NPU.md) "In-engine confirmation" — not repeated here. Three methodology
points belong with the rows, though:

- **`route` is flat, 156 → 158 ms, across a 2.1× context increase** — first direct evidence
  that DSA caps the attend at `index_topk` rows. `route` is the right bucket to read across
  runs for the reason this file already gives above: attention runs on resident weights and
  is structurally insulated from fetch variance.
- **The microbench under-predicts the indexer's GPU span by 27%** (1.271× and 1.264×, two
  contexts, agreeing to 0.6%) — size solid, **mechanism unestablished**. The rig's own
  launch-overhead measurement (5.16 µs for a four-kernel group with one sync) under-predicts
  the ~41 µs surplus by 4–8×, so "launch bubbles" does not account for it. This is the
  second unexplained microbench under-prediction of ~27% in this file; the earlier one
  (route tranche, above) is a ratio of two *deltas* in which fixed per-launch overhead
  cancels, so the two cannot share a cause and neither corroborates the other.
- **The host round-trip is 2.0–2.2× its isolated microbench** at matched nt, so even a
  deliberately realistic synthetic distribution understated it. A harder real distribution
  and in-situ CPU-cache contention from the streamer moving 1.7–2.1 GB/token both fit; not
  separated here.

**A wall series across contexts is not obtainable from runs like these.** Run A's prompt is the first
12,000 characters of run B's — wholly contained in it — so context length and prompt
content are perfectly confounded — and reaching any longer context requires more text, so
the confound is structural, not an artifact of this pair. The +47 ms of wall came with hit%
81.4 → 76.9 and ms/miss 76 → 134; n = 2 cannot apportion it. Compare `route` across runs, not
`wall`.

### Device top-k (`index_topk`) vs the host round-trip, 2026-07-27

`examples/indexer_bench`, gfx1151 sole tenant. Controls that run: `o_proj` 520.22 µs /
193.5 GB/s, rotation 1.16× — both ok. Correctness gate
`tests/kernel.rs::index_topk_matches_host_selection` passes on all 10 cases, including a
sentinel-tail assertion that nothing is written past `min(k,nt)`.

Both implementations timed in the same rig, on the same buffer, on the same data, µs per
full layer (host → device):

| nt | dense (few ties) | scattered (heavy ties, random order) | sorted-sparse (**artifact**) |
|---:|---:|---:|---:|
| 2456 | 86.6 → 35.8 (2.42×) | 54.4 → 45.6 (1.19×) | 28.8 → 45.3 (0.64×) |
| 4096 | 107.7 → 32.5 (3.32×) | 61.2 → 54.3 (1.13×) | 32.9 → 51.0 (0.65×) |
| 5209 | 101.4 → 41.2 (2.46×) | 74.9 → 59.7 (1.25×) | 41.1 → 60.9 (0.67×) |
| 8192 | 126.4 → 52.7 (2.40×) | 96.6 → 82.7 (1.17×) | 46.8 → 79.2 (0.59×) |
| 16384 | 344.5 → 83.3 (4.14×) | 144.3 → 126.6 (1.14×) | 65.7 → 127.6 (0.52×) |
| 32768 | 578.1 → 157.6 (3.67×) | 191.8 → 215.0 (0.89×) | 144.6 → 215.0 (0.67×) |

**A fixture can look like a finding — this one did.** An earlier revision of this section
measured only the third column and reported the kernel as 1.6–1.9× *slower* than the CPU,
attributing it to ties making quickselect cheaper. `topk_into` seeds its index workspace
with the identity permutation and orders by (score desc, index asc); that fixture's
non-zero values descend from index 0, so the identity **is** the sorted order and both
`select_nth_unstable_by` and the trailing `sort_by` got an already-sorted slice — their
best case, unavailable to the kernel. The `scattered` column holds the tie structure fixed
and randomises order: the ratio moves 0.64× → 1.19×, so **~1.8× of that "regression" was
the fixture.** When timing a comparison-based algorithm, randomise the input order or you
are measuring your generator.

**Corrected reading.** Ties cut both ways — cheaper for quickselect, dearer for the radix
histogram (tied keys collide on one LDS bin) — so the kernel runs 2.4–4.1× faster on
dense data, 1.13–1.25× on tie-heavy, and 0.89× at tie-heavy 32k. Never quote a single
speedup without the distribution.

**Caveats on precision.** The host column is 20 iterations reported as a bare mean with no
dispersion, and it is non-monotonic (107.7 µs at nt=4096 against 101.4 at 5209) and
disagrees with the earlier `m0_host` row by up to ~30% at some contexts while matching to
~1% at others. Ratios here are good to about one significant figure, not two. The device
kernel is also single-workgroup, so its absolute cost is one CU's serial sweep; the
LDS-contention hypothesis names a lever but occupancy is the larger structural bound.

Interpretation and what it means for wiring: docs/NPU.md § "The device top-k, measured".

### Device top-k WIRED: three-arm in-engine A/B, 2026-07-27

`--attn dsa --mode hybrid --cache-policy lru --max-mem 115 -bench 128`, the same
2432-token prompt as "In-engine confirmation" above, gfx1151 sole tenant. Arms selected by
`RIVOLI_TOPK` from **one binary** — no build differs between them — run **interleaved**
(host, device, device-nosync, twice). Greedy decode is deterministic, so every arm generates
the same tokens and the same expert-miss sequence: the arms are PAIRED, and `116.79
miss/tok` is identical across all seven runs.

**Read the buckets, not the wall.** `wall = route + moe + idx_gpu + idx_host + unbucketed`
is an identity here and closes to ±0.4 ms on every run below. `moe` carries **7–10 ms of
within-arm spread** with no proposed mechanism, against effects of 9.4 and 2.5 ms — so at
n=2 the wall cannot resolve either change, while the buckets that respond to them can. The
unbucketed column is included for exactly this reason; omitting it is what made an earlier
revision of this section wrong twice.

| arm | rep | wall | route | moe | idx_gpu | idx_host | unbucketed |
|---|---|---:|---:|---:|---:|---:|---:|
| `host` | r1 | 446.9 | 154.4 | 247 | 4.10 | 11.20 | 30.20 |
| `host` | r2 | 451.8 | 155.9 | 254 | 4.11 | 9.01 | 28.78 |
| `device` | r1 | 443.7 | 154.5 | 254 | 4.82 | — | 30.38 |
| `device` | r2 | 433.5 | 155.3 | 245 | 4.82 | — | 28.38 |
| `device-nosync` | r1 | 434.5 | 167.1 | 248 | 4.82 | — | 14.58 |
| `device-nosync` | r2 | 441.7 | 165.4 | 255 | 4.82 | — | 16.48 |

Per-layer: `idx_gpu` 195.2 / 195.9 µs (host), 229.3 / 229.6 (device), 229.7 / 229.3
(nosync); `idx_host` **533.2 / 428.9 µs**, host arm only.

**The two wins, costed separately. Both are real and they differ by 4×.**

| change | measured in | r1 | r2 | mean | wall delta (for contrast) |
|---|---|---:|---:|---:|---:|
| `host` → `device` (the top-k) | indexer bucket | −10.48 | −8.30 | **−9.4** | −3.2 / −18.3 |
| `device` → `device-nosync` (the sync) | route + unbucketed | −3.20 | −1.80 | **−2.5** | −9.2 / **+8.2** |

**The top-k: −9.4 ms/token, 2.1% of wall.** The indexer bucket is the only one that
responds — `idx_host` (11.20, 9.01) goes to zero and `idx_gpu` rises 0.72 for the kernel —
and **the unbucketed remainder is unchanged to ±0.4 ms in both replicates**, i.e. nothing
else moved. Per-replicate agreement is 2.18 ms. The wall deltas for the same change are
−3.2 and −18.3, a 15 ms spread, entirely from `moe` going +7.0 then −9.0. This is a
measurement, not a prediction: `idx_host` is host wall time the engine spends with the GPU
idle, and on this arm it stops existing.

**The sync: −2.5 ms/token, 0.6% of wall.** `route` rises +12.60 / +10.10 as the wait
relocates to the gate-logits D2H, and the unbucketed remainder falls −15.80 / −11.90; the
difference is the win. Same sign in both replicates. **Its wall delta changes sign
(−9.2, +8.2) purely because `moe` swings −6.0 / +10.0** — a 16 ms swing against a mechanism
bounded at 3 dense layers × 229 µs = **0.7 ms**, i.e. 14× more movement than the change can
physically cause. `moe` is noise here and must be kept out of the comparator.

**The default keeps the sync anyway**, and this is a judgement not a measurement: −2.5 ms is
0.6% of wall at n=2, against making `route` incomparable with every historical row in this
file. Re-run at n≥4; if the −2.5 holds, flip it.

**Two corrections this section previously got wrong, recorded because both are instructive.**
*(1)* An earlier revision headlined the top-k at −11.2 ms from the `host` → `device-nosync`
wall delta, calling the −9.4 a "prediction" the wall "confirmed". That inverted the
evidence: −9.4 is the direct measurement and −11.2 is a proxy carrying `moe`'s noise plus
the sync's own −2.5. It also attributed a figure from an arm that is not the shipped default
to the shipped default. *(2)* The same revision reported the sync deletion as worth
"nothing, sign reverses" — which was `moe` noise admitted into the comparator for a 2.5 ms
effect, and was inconsistent with this section's own withdrawal of the `moe` story below.
The rule that would have caught both: **decide which buckets can respond to the change
before looking at any of them.**

**Note the three rows are not three measurements.** Row 3 (`host` → `device-nosync`) is
exactly row 1 + row 2 per replicate; three arms give **two independent contrasts**. Quoting
a "solid" third row is selecting the pair that happened to land closest.

**A mechanism proposed and refuted, recorded so nobody re-derives it.** In r1 the `device`
arm's `moe` rose 247 → 254, almost exactly cancelling the top-k win, and the obvious story
was that the baseline's ~11 ms of CPU top-k had been doubling as head start for the fetch
reaper. **In r2 the same comparison went the other way (254 → 245).** Fitted to one
replicate; withdrawn. It is the first hypothesis anyone will form from the r1 column.

**The instrument that reproduced, and the one that did not.** `idx_gpu` is 195.2 and 195.9
µs/layer — 0.15% and 0.51% from the 194.9 recorded in a different session (mean 0.33%).
`idx_host` is **533.2 and 428.9 µs/layer: 24% apart, same binary, same arm, same prompt,
forty minutes apart**, against 214.2 in that earlier session. **The quantity this branch
denominates its entire prize in is the unstable one.**

**What is NOT explained about that instability.** These runs saw `28–30 ms/miss` where the
earlier session saw `76.1`, and one reading is the streamer serving from page cache and
pushing ~1.8 GB/token through the CPU concurrently with the CPU top-k. **Two facts in the
same table cut against it:** this session is 15% SLOWER at the wall (447–452 vs 391) with
`moe` 24% higher (247–255 vs 201) at identical flags and prompt, and `idx_host` moved 24%
*within* this session at constant ms/miss. The two replicates are also one session, not two
independent observations. **No mechanism is established.** The usable consequence: the SIZE
of the top-k win is machine-state dependent; its existence is not.

**Correctness, and why the exit status is not the evidence.** `RIVOLI_TOPK=verify` runs both
selections per full layer and compares: **10,752 full layers matched the host selection
exactly** — 21 full layers × 512 scoring tokens (384 prefill past `index_topk` = 2048, plus
128 decode), i.e. every layer that could have run. A sentinel parked one slot past
`min(topk, nt)` survived all 10,752, so the kernel did not over-select. The count is quoted
rather than the exit status because an earlier revision of this gate **exited 0 having
compared zero layers** whenever the context stayed under `index_topk`. The repaired gate was
then confirmed to fail: `RIVOLI_TOPK=verify … -bench 4` on the default short prompt exits 1
with `compared 0 layers: the context never exceeded index_topk=2048`. The comparison loop
was rewritten during review, so the gate was re-run afterwards on the shipped binary:
**8,736 layers matched** at `-bench 32` (21 × [384 prefill + 32 decode]). All seven runs
generated **byte-identical output** (564 chars, sha256 `778387fa557c4e9d…`), coherent prose.

**What is NOT established.** One prompt, one context (nt ≈ 2496 mean), n = 2 per arm, and
`moe`'s 7–10 ms spread is uncharacterised — until it is, no wall-level effect below ~15 ms
is measurable on this rig by wall alone, which affects any future A/B in this file. This
session's wall is 15% above the earlier one at identical flags, and that is unexplained.
`--attn misa` takes the device path, skips the timing bracket, and was never run. A paired
`--ppl` across arms would beat identical greedy text as an equality check and was not run.
Under-selection is caught only incidentally (stale rows differ); a poison-fill of
`rows_buf[0..nr]` before the launch would make it explicit.
