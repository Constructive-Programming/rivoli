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

### int4 provenance (in progress)

These int4/hybrid numbers use `.i4` re-derived from **vq3** (itself a lossy 3-bit
quantization). The higher-fidelity source is the original **GLM-5.2-FP8** checkpoint;
`fp8_to_i4` (deriving `.i4` straight from fp8 via `quant_i4`) is pending a re-download of
that checkpoint, after which int4/hybrid will be re-benched against this baseline.

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
> **~1.9× effective pool — at 17.8% swap, quality cost unmeasured.** J=4/M=10 is worth
> ~1.4× at 9.6% swap.

Pool growth is free; substitution is not. Quoted without the swap number, "1.9× effective
pool" reads as costless — 17.8% swap means nearly one chosen expert in five is not the one
the router asked for.

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

---

## Measurement caveat

Free-running greedy `tok/s` cannot rank modes on its own: a degenerate run routes to the
same few experts → inflated hit% → artificially *fast* (the earlier int4 rows posted the
highest tok/s *because* they degenerated). Always gate on output quality first, then
compare speed among survivors. For residency use `replay <trace> <n_slots> [--sweep]`; for
pure per-format compute use `examples/dot_bench.rs`. See [MODES.md](MODES.md).

*Generated 2026-07-26. Reproduce: `--mode <m> --cache-policy <p> -bench 512 --attn dense
--max-mem 115 --prompt "<above>"`.*
