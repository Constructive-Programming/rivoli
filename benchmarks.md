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

## Measurement caveat

Free-running greedy `tok/s` cannot rank modes on its own: a degenerate run routes to the
same few experts → inflated hit% → artificially *fast* (the earlier int4 rows posted the
highest tok/s *because* they degenerated). Always gate on output quality first, then
compare speed among survivors. For residency use `replay <trace> <n_slots> [--sweep]`; for
pure per-format compute use `examples/dot_bench.rs`. See [MODES.md](MODES.md).

*Generated 2026-07-26. Reproduce: `--mode <m> --cache-policy <p> -bench 512 --attn dense
--max-mem 115 --prompt "<above>"`.*
