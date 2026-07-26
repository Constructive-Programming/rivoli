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

## Measurement caveat

Free-running greedy `tok/s` cannot rank modes on its own: a degenerate run routes to the
same few experts → inflated hit% → artificially *fast* (the earlier int4 rows posted the
highest tok/s *because* they degenerated). Always gate on output quality first, then
compare speed among survivors. For residency use `replay <trace> <n_slots> [--sweep]`; for
pure per-format compute use `examples/dot_bench.rs`. See [MODES.md](MODES.md).

*Generated 2026-07-26. Reproduce: `--mode <m> --cache-policy <p> -bench 512 --attn dense
--max-mem 115 --prompt "<above>"`.*
