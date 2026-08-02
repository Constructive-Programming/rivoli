---
status: data
verdict: Append-only measurements. Never read whole — grep for the config. The top table predates the .i4 rebuild and says so.
---

# Benchmarks

> ## STATE — read this, then grep for your config
>
> **Append-only measurement log, 132 KB. Do not read whole.** `grep -n "^## " docs/measurement/benchmarks.md`
> for the map. Newest results are at the BOTTOM.
>
> - **Quality ladder (current):** int4 **5.120** > hybrid **5.189** > int3-vq **5.275**.
>   The mode table immediately below predates the `.i4` rebuild and its int4/hybrid rows do
>   NOT describe the current artifact.
> - **THE PROMPT FRAMING CHANGED ON 2026-08-01, so free-running text is not comparable
>   across that date.** `encode_chat` emitted GLM-4's `<|role|>\n{content}` and ended the
>   prompt at `<|assistant|>\n`; this checkpoint's `chat_template.jinja` has **no separator
>   after the role token** and ends at `<|assistant|><think></think>`. Every `-bench` run
>   above was therefore one token off-template per turn and carried no thinking prefill.
>   Throughput rows are unaffected in kind (the work per token did not change) but their
>   *text*, acceptance rates and hit rates all move, because the model is being asked a
>   differently-tokenized question. Re-measure before comparing a new run to an old one.
>   `--ppl` numbers are NOT affected: it scores a corpus through `encode`, never the chat
>   framing. See `tests/artifact.rs::chat_framing_matches_the_checkpoint_template`.
> - **Throughput (DECODE):** ~2.6–2.8 tok/s int3-vq, ~2.7 hybrid, ~2.1 int4 (larger slot →
>   fewer resident experts). Fetch is 96–98% hidden; decode is MoE-compute-bound.
> - **PREFILL is a different regime, and it was never in this block.** Token-major prefill
>   re-reads experts **154.75 per token** — 77 layers separate two demands for the same one
>   and the pool evicts in the gap. `--layer-major-prefill` (opt-in, 2026-08-02) reorders it
>   to **28.20 reads/token**, the compulsory floor, for **2.15×** on prefill wall, output
>   byte-identical, every `--attn` mode. Decode is unchanged by it (it cannot be reordered —
>   token T+1's input is T's argmax) beyond a one-off ~2.7 s warm-up. See "Layer-major
>   prefill" at the bottom. Reads and wall are measured; that what now bounds prefill is
>   LPDDR5 expert re-reads is INFERRED from the 5.66x-reads-for-2.15x-wall gap, not measured.
> - **Speculative decode:** **1.108× gated** (`--mtp-min-conf 0.8`, the default); 0.93–0.95×
>   ungated. See "The MTP confidence gate" at the bottom — it supersedes the earlier
>   "Speculative decode (`--mtp`)" section, which measured only the ungated form.
> - **`distinct` / `longest repeated block` do not measure quality** — see the note under
>   "Results" below before using either to judge a run.
> - Every run: sole-tenant GPU, no `cargo build` between arms of a pair.

512-token greedy decode, GLM-5.2 full artifact (`/var/db/rivoli/glm52-vq3-full`),
AMD Strix Halo gfx1151. Matrix: `--mode {int3-vq,int4,hybrid}` × `--cache-policy
{lru,2q,arc}`, 9 runs.

**Fixed across every run** (for comparability): same prompt
(*"Explain, step by step, how a transformer neural network processes a sentence."*),
`--attn dense`, `--max-mem 115`. Only `--mode` and `--cache-policy` vary.
Binary: release + `--features rocm`. GPU sole-tenant (k3s stopped).
`.i4` experts are the **vq3-derived** set (`vq3_to_i4`); see "int4 provenance" below.

> **STALE — and the staleness notice was itself stale, corrected 2026-07-31.** Every int4
> and hybrid row below was produced with the *old* per-row-scaled `.i4` set and does not
> describe the current artifact. The number this banner used to quote — "`--mode int4` now
> measures PPL 73.43" — was the **pre-fix** figure and stopped being true the same day it
> was written: `docs/investigations/int4-scales.md` is headed *"Status: RESOLVED, 2026-07-27"*, group-128 scales
> took int4 from **73.43 → 5.120** and hybrid from 11.55 → 5.189, and int4 became the
> best-quality mode in the engine. Re-measured 2026-07-31 on the current artifact:
> **int4 5.154898 against int3-vq's 5.222720**. Read `docs/investigations/int4-scales.md` §0 and §10, and the
> `--mode int4` section near the end of this file, before quoting any int4 number from
> anywhere above.

## Results — all coherent, no crashes

Output quality is gated first via the distinct-token ratio of the completion. Every cell
passed. **Do not rank on this metric, and do not treat a low score as a bug report.**
Measured 2026-07-27: across a branch-gain sweep PPL tripled (73 → 216) while distinct-ratio
doubled (0.126 → 0.324) — monotone in OPPOSITE directions. It detects repetition, one
failure mode among many, and repetition is suppressible by changes that damage the model.
Rank on teacher-forced PPL; use this only to flag a run unreadable. See `docs/investigations/int4-scales.md` §1.

> **This gate has now misled three separate investigations, so it is worth stating what it
> cannot do.** `distinct` and `longest repeated block` fire IDENTICALLY on (a) a clean
> greedy repetition loop, (b) *spliced corruption* — half-copies from context cutting
> mid-phrase — and (c) legitimate prose that restates a paragraph on purpose. The three
> demand completely different responses and the metric cannot tell them apart:
>
> - §10 of `docs/investigations/int4-scales.md`: hybrid scored the WORST distinct-ratio of the three modes
>   (0.138) with the second-best PPL. A distinct-ratio gate would have rejected the best
>   config in the engine.
> - 2026-07-31: `distinct 0.193 / longest repeated block 77` on int3-vq was read as
>   "repetition collapse" and sent an investigation after a KV/attention bug that does not
>   exist. The text was spliced, not looped, and that distinction was the whole diagnosis.
> - 2026-07-31: int4 and int3-vq scored `distinct 0.279` vs `0.264` and **`longest repeated
>   block 77` for BOTH** on the same prompt — one produced correct physics with a
>   deliberate `**Corrected Version:**` restatement, the other produced wreckage.
>
> **Read the text.** The metric flags a run for reading; it does not diagnose it, and an
> earlier framing of degeneration as "a severe bug, disqualified from ranking" is retired
> as of 2026-07-31 — that is a hypothesis to test, not a verdict to record.

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

> **RETRACTED 2026-07-30.** This read: "Fetch is ~95% hidden behind compute everywhere —
> the engine is compute-bound (route + moe-gpu), not fetch-bound, at this budget." Both
> halves are wrong. The hidden-% column is computed from `compute_gpu_ns`, a HipEvent
> **bracket** spanning the whole MoE phase, so the fetch stalls *inside* it were counted as
> compute and thus as fetch hidden. Real ceiling is `compute/fetch_wall` ≈ **57%**, and the
> engine is **fetch-bound**: 2.25 GB/token at ~12 GB/s is ~181 ms of transfer against
> 117 ms of compute. The per-miss cost is 1239 µs, measured by bucketing the bracket by
> per-layer miss count. See ARCHITECTURE.md §3. **The hidden-% column in the table above is
> unreliable; the ms/tok, misses/tok and tok/s columns are unaffected.**

hybrid/lru's edge is the fewest misses/tok (116 vs ~132–145) → lowest fetch and lowest MoE
wall. That ordering *is* still meaningful — and reads more directly now: under a fetch-bound
engine, fewest bytes moved wins, which is exactly what the tok/s column shows.

### DIRECT vs BOUNCE ablation (2026-07-30, int3-vq/dense/lru, `--max-mem 115`, 512 tok)

Run to test whether the per-miss cost was serialized H2D staging (all bounce copies share
one FIFO fetch stream). It is not — removing the copies made it **2.2× worse**:

| | 0-miss layer | per-miss | ms/tok | tok/s |
|---|--:|--:|--:|--:|
| bounce (default) | 1563 µs | **1239 µs** | 386 | **2.59** |
| `--direct-vmm-dma` | 1525 µs | **2709 µs** | 837 | **1.19** |

DIRECT's 2709 µs/miss is 15.34 MB / 5.66 GB/s — one expert at the array's *serialized*
rate. Bounce's 1239 µs is 12.4 GB/s, ~2.2× that, so the concurrent-submit path is working
as designed. The zero-miss rows being equal (1563 vs 1525 µs) also shows the flag does not
change where kernels read from — it only flips the streamer's DMA destination.

> **2026-08-01: `--direct-vmm-dma` was DELETED and the staging hop is now unconditional.**
> This measurement is why, and it is not superseded — it is the reason the flag went. Misses
> are the entire design, and the DIRECT row more than doubles the cost of one. The amdgpu
> EFAULT on O_DIRECT DMA into VMM pages that the flag *also* worked around no longer
> reproduces on kernel 6.18.38, so nothing was left holding it up. **The table stands as the
> record**; `src/fetch/stream.rs`'s header carries both these numbers and the
> `get_user_pages` history, because that history is what forbids ever making the arena
> device-local. Re-adding a direct destination on **Vulkan** is a separate, still-open
> question — the pool is host memory there, so no H2D copy is being saved
> (`investigations/vulkan-port.md`, increment 1 finding 2).

---

## Bugs found and fixed

### fp8 block scale MIS-APPLIED at `block < 4` — every fp8 GEMV (fixed)

**A numerics bug, not a perf one, and it sat in the helper every fp8 block-scaled GEMV
goes through.** `common.hpp::fp8_dot_strided` reads four fp8 weights per lane as one dword
and applies **one** block scale to all four:

```c
float s = scalerow[i0 >> bsh];              // ONE scale for columns i0..i0+3
acc += s * (x[i0]*lut[..] + x[i0+1]*lut[..] + x[i0+2]*lut[..] + x[i0+3]*lut[..]);
```

That is only the right scale when the scale tile is at least a quad wide. At `block` 1 or
2 the columns past the tile boundary belong to *later* tiles and silently took `i0`'s —
three of four at block=1, the upper two at block=2. Affected `gemv_fp8`,
`gemv_fp8_splitk` and `mla_value_fp8`, i.e. `o_proj`, `q_a`, `q_b`, `kv_a`, the dense MLP
and the MLA value projection.

**The guard test was actively asserting the broken domain.** `blk_shift` needs a
power-of-two tile, so guard 1003 rejects non-powers-of-two — and 1 and 2 *are* powers of
two, so they passed. `gemv_fp8_rejects_non_power_of_two_block` goes further and
**explicitly requires block=1 to be ACCEPTED** ("1 is a power of two (bsh = 0,
`i >> 0` == `i / 1`), so it must be ACCEPTED"). The launcher's contract said the input was
legal while the kernel computed it wrong, which is the worst of the two failure modes: no
error code, no fault, just wrong numbers.

**Why nothing caught it.** Every oracle shape in the suite used `block = 128`. A tile that
wide can never expose a quad straddling two tiles, so the entire fp8 oracle set was
structurally incapable of seeing this — the same class of blind spot as the `fnv` note
below, where an instrument is green because its inputs cannot reach the defect.

**Found** by a `block = 2` shape added to `mla_fp8_matches_reference` while restructuring
`mla_absorb_fp8` — the quad restructure needs exactly the same `block >= 4` precondition,
which is what prompted asking whether the existing helper had it. It did not.
`mla_value` at block=2 failed at **err 2.2e-1 against tol 1.5e-3**; `gemv_fp8` at block=2
failed at **err 6.2e-1**. Both are hard failures, not tolerance grazes.

**Fix**, one line, in the helper rather than the callers:

```c
int n4 = (block >= 4) ? (i_dim >> 2) : 0;   // narrow tile → per-column tail path
```

The per-column tail loop below it was always correct, so zeroing `n4` hands it the whole
row. **Bit-identical at `block >= 4`** — the engine runs 128, and the generated ISA for
the hot loop is byte-for-byte unchanged, so no shipped model's numerics moved. Post-fix
margins: `mla_value` block=2 at 12318×, `gemv_fp8` block=2 and block=1 green.

**TWO TWINS ARE KNOWN-BROKEN AND DELIBERATELY LEFT.** Both are recorded in
`kernels/common.hpp` at the fix site and in docs/measurement/perf-roadmap.md #4:

- **`kernels/vk/fp8.glsl::fp8_dot_strided` has the identical bug** — same loop, same
  unconditional `n4 = i_dim >> 2`, same one-scale-per-quad. **And `tests/vk.rs`'s
  `oracle_fp8_dot_strided` MIRRORS the kernel, including this behaviour** (`let s =
  scalerow[i0 / block];` for the whole quad). So the Vulkan suite is **structurally blind
  to it**: the oracle and the kernel agree *because they share the defect*, and adding a
  block<4 shape there would pass while still being wrong. Fixing it means changing the
  shader **and** its oracle together. Not done here — this branch had no Vulkan device
  slot, and an unrunnable fix to a second backend is worse than a recorded divergence.
- **`rivoli_gemv_fp8` still lacks the `i_dim % 4` guard** its `w4` cast needs (rows are
  `i_dim` bytes apart, so a ragged `i_dim` misaligns three rows in four). `src/vk.rs`
  enforces it; `rivoli_mla_value_fp8` now enforces the equivalent `kvl % 4` (guard 1002,
  a parity gap this same work found and fixed). The requirement is **conditional** — at
  `block < 4` the cast is never reached — which is why it is a guard question rather than
  a second bug. Not added because a launcher guard that fires hard-fails a decode, and
  this branch was barred from running the engine to confirm no live projection dim trips
  it. Every GLM dim is a multiple of 4; the exposure is a future config, not this one.

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
> **SUPERSEDED 2026-07-27 — see `docs/investigations/int4-scales.md`.** Two claims below are now measured false:
> that `.i4` "cannot be better than the vq3 it was derived from, by construction", and that the
> deficit is "the arithmetic of double quantization". The set was rebuilt from fp8 and is
> **strictly more accurate** — and **8× worse end to end** (PPL 73.43 vs 5.28). The real cause
> is per-row scaling (one scale per 6144 weights). The gs64/`pack_i4` recommendation at the end
> of this section turns out to be **right for the wrong reason**, and `docs/investigations/int4-scales.md` re-endorses
> it on the correct one.

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
compute rather than buying quality with it. `docs/investigations/cache-conditional-routing.md` carried the same inverted
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
> **SUPERSEDED 2026-07-27 — see `docs/investigations/int4-scales.md`.** The measurements below stand; the
> *recommendation* does not. Tuning α is tuning a constant inside a per-row scheme that is far
> coarser than any current practice (group-wise at 32–128 is standard). **Do not implement α.**
> The end-to-end test this section called for was run: `--mode int4` measures PPL 73.43 against
> int3-vq's 5.28, and a branch-gain sweep falsified the attenuation hypothesis outright.

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

**Chain-level accuracy is x-draw noise; chain-level GAIN is not.** Scored through the
full `down(silu(gate·x) ⊙ up·x)` chain on L3 experts 0/7/shared (`bin/i4_audit`, the GPU
test's seed), new vs old `.i4` rel-L2 is 0.295/0.256/0.242 vs 0.290/0.305/0.296 — the sign
flips with the expert, i.e. within draw noise. The gains over the same cells are
**1.001/1.066/1.006 vs 0.894/0.949/0.902**, consistent in sign and size. So from the
model's point of view the reliable thing that changed when the `.i4` set was rebuilt is
the ~9% attenuation going away, not the accuracy. Weight-space rel-L2 (0.205 vs 0.250)
stays the trustworthy accuracy number; the chain statistic is too noisy to rank on.

### Pre-flight for the attenuation arm: specified correctly, but 0.9766 is not the knob value

Before booking device time to test whether the model needs an attenuated MoE branch, the
arm has to be shown to produce the attenuation it claims. `bin/i4_audit` scores the
shipped nibbles with every stored per-row scale multiplied by `VQ_GAIN = 0.9766` —
vq3's shrink without vq3's error — through the full expert chain, 9 cells (layers
3/40/77 × experts 0/7/shared):

| | branch factor vs the unattenuated new `.i4` | sd | range |
|---|---:|---:|---|
| arm (stored scale ×0.9766) | **0.9282** | 0.0010 | 0.9271–0.9297 |
| the old `fp8→vq3→int4` set | **0.9124** | 0.0341 | 0.8391–0.9483 |

**The arm is well specified and is *cleaner* than what it mimics.** Its effect is uniform
to ±0.1% across layers and across routed vs shared experts, which is what an arm should
be. The old set's attenuation is the same size but scatters ±3.4%, because its extra
quantization error also passes through `silu` and contributes apparent attenuation that
varies per expert. So the knob isolates attenuation; it does not reproduce the old set's
per-expert texture, and if what the model needs is expert-DEPENDENT attenuation the knob
cannot show it.

**The correction that matters: the per-projection constant is not the branch constant.**
0.9766 per projection compounds to **0.9282** at the branch output — `0.9766³ = 0.9314`,
and `silu` compression takes off another 0.35%. Setting a branch-level `--moe-gain` to
0.9766 would under-attenuate by 5% and land outside the old set's central value.
A branch-gain sweep must be specified in branch units: **1.00 / 0.96 / 0.93 / 0.90 /
0.86**, where 0.93 is the faithful equivalent of the artifact rewrite and 0.91 is the old
set's centre.

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
Moving a threshold after seeing the result it fails is post-hoc. See `docs/reference/modes.md`.

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
docs/measurement/perf-roadmap.md both do. The same defect is present in every absolute µs figure in this
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

### ~~Open question: half of `tail` is in none of its kernels~~ — ANSWERED

> **Closed by the CLASS axis (docs/measurement/perf-roadmap.md).** The unattributed time is decode-loop HOST
> CPU, not a hidden kernel — measured at ~6 ms/tok of a total 6.2 ms host compute, itemised
> as kernel launch, tokio poll *(term deleted 2026-08-01 with tokio itself; 2.4 ms of a
> 338 ms token, so the conclusion is unchanged)*, `submit_layer` and `route_into`. The
> candidate named below
> (per-token launch/sync/readback overhead) was right. It is also DEMOTED by the same
> measurement: host compute is under 2% of wall, so this is not where the engine is slow.
> The analysis below is kept for the reasoning, not as a live task.

### Original: half of `tail` is in none of its kernels

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

**That open question is now CLOSED — refuted.** `SPLITK_ROWS` tiling was implemented and
swept 1/2/4/8; R=8 cuts x traffic 8× and is the **slowest** arm (+11%), R=2 the best at
−1.4%, inside a noise band wider than the effect. The tiling was reverted. See
docs/measurement/perf-roadmap.md follow-up #2 for the table. The load-width lever, meanwhile, paid where it
was correctly aimed: `mla_absorb_fp8`'s single-byte weight load became a dword and the
kernel went 35.9 → 25.7 µs. Two levers, one refuted and one confirmed, out of the same
ISA pass.

### Divide by the peak before you book the slot

**402 MB of x plus 100 MB of weights in 529 µs is 950 GB/s, on a 256 GB/s part.** The
traffic figure that motivated the x re-read item was 3.7× over the hardware limit, which
means it was never DRAM traffic — `x` is 64 KB, it lives in cache, and the kernel was
never paying the bus cost the hypothesis charged it. **One division would have retired the
item without a device slot.** It cost four builds and twenty interleaved runs instead.

This generalises past bandwidth. Any figure a hypothesis rests on — GB/s, instructions
per byte, bytes per wave — has a hardware ceiling next to it, and checking is free.
`mla_value_fp8` reads 8.4 MB in 26.1 µs = **322 GB/s, also over peak**: it is re-reading a
14.7 MB `kv_b` that the bench leaves resident across its 60 iterations, so "310 GB/s" was
never a DRAM number either and `mla_absorb`'s 1.8× gap to it was never a 1.8× bus gap.
Both numbers are still valid *comparators* between arms of the same bench. Neither is a
roofline. Say which one you are quoting.

### A fingerprint is the only instrument that shows bit-identity

`assert_close` cannot tell a bit-identical restructure from a reassociating one — both
pass, and the margin print does not distinguish them either. `examples/dot_bench` now
prints an FNV-1a hash of each kernel's raw output bytes, and the absorb restructure's
claim rests on it: `0925c147afeea3fb`, unchanged across 14 interleaved runs of both arms.

It only works if the inputs vary. `run_fp8` and `run_mla` used constant `x`, `q` and
`clat` — correct for throughput, since traffic does not depend on values, but a constant
input leaves the output insensitive to summation order, and the fingerprint would have
been green for a change that reassociated. **The instrument and the input generator are
one instrument**; a fingerprint over degenerate data is a fingerprint of nothing.

### Measuring on a contended bus — report min, and keep a control arm

These runs shared the machine with a memory-heavy conversion job, and `o_proj` (100 MB
streamed) ranged **515 → 1141 µs on a single unchanged arm**. The MLA kernels (6-14 MB,
cache-resident) held to ±3% through the same window. Three things kept the conclusions:
interleaving the arms so drift hits both, quoting **min-of-N** rather than mean (the
minimum is the least-contended sample; the mean measures the neighbour), and carrying an
**unchanged control kernel** in the same binary — `mla_value` at 26.1 µs in both arms, and
`lm_head` at 8026 vs 8038 µs, are what license reading a 1% difference at all.

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
pure per-format compute use `examples/dot_bench.rs`. See [docs/reference/modes.md](docs/reference/modes.md).

*Generated 2026-07-26. Reproduce: `--mode <m> --cache-policy <p> -bench 512 --attn dense
--max-mem 115 --prompt "<above>"`.*

---

## DSA indexer round: `examples/indexer_bench`

Instrument for the NPU-offload gates (docs/investigations/npu-offload.md M0/M1), gfx1151 sole tenant, 2026-07-26.
The rig itself is deleted (`77b5500:examples/indexer_bench.rs`) — superseded by the
engine's in-engine indexer buckets, which refuted its GPU-span figure by 27%. Every row
below stands as recorded; re-running them means restoring the file.
Interpretation lives in [docs/investigations/npu-offload.md](docs/investigations/npu-offload.md); the rows and the methodology are here.
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
[docs/investigations/npu-offload.md](docs/investigations/npu-offload.md) "In-engine confirmation" — not repeated here. Three methodology
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

Interpretation and what it means for wiring: docs/investigations/npu-offload.md § "The device top-k, measured".

### Device top-k WIRED: three-arm in-engine A/B, 2026-07-27

`--attn dsa --mode hybrid --cache-policy lru --max-mem 115 -bench 128`, the same
2432-token prompt as "In-engine confirmation" above, gfx1151 sole tenant. Arms selected by
`RIVOLI_TOPK` from **one binary** — no build differs between them — run **interleaved**
(host, device, device-nosync, twice). **The switch no longer exists**: `device` shipped,
`host`/`device-nosync`/`verify` were deleted once these rows were recorded
(`77b5500:src/gpu.rs` restores them). Every figure below stands as measured; the PROFILE
line no longer prints a `[topk=…]` tag or an `idx_host` term, so rows recorded after
2026-07-30 carry neither. Greedy decode is deterministic, so every arm generates
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

## RETRACTION: the 512->10k matrix's long-context results are invalid

**Everything below at 2048 tokens and above measured degeneration, not throughput.** The
headline — `int3-vq/streaming/2q` "the only cell that gets FASTER with context" — is an
artifact. Reclassifying all 58 cells from their logs with a structural-repetition check:

| round | tokens | valid | degenerate |
|---|---:|---:|---:|
| 1 | 512 | **42/42** | 0 |
| 2 | 2048 | 7/8 | 1 |
| 3 | 4096 | **0/4** | **4** |
| 4 | 10000 | **0/2** | **2** |

The winning cell was degenerating from 2048 onward and the detectors passed it every time:

| tokens | tok/s | hit % | most-repeated line | distinct-word |
|---:|---:|---:|---|---:|
| 512 | 2.81 | 81.6 | (none) | 0.474 |
| 2048 | 2.97 | 83.2 | `- Mechanism.` x38 | 0.366 |
| 4096 | 3.06 | 85.4 | `- Mechanism.` x53 | 0.288 |
| 10000 | 3.26 | 89.6 | `**Memory Product.**` **x329** | 0.244 |

**The rising hit rate was the tell, and it was reported as a success.** Throughput and hit
rate climb monotonically *because* the output collapses: a template loop re-routes to the
same experts. Every "streaming scales with context" conclusion drawn from this is void.

**Why both detectors missed it.** The loop had a VARYING SLOT — `**Memory Phase:**`,
`**Memory State:**`, `**Memory Status:**`, … — so `detect_loop` found no verbatim cycle and
`longest_repeated_block` capped at 142 tokens. Both are exact matchers. A near-miss loop
with one changing token is the most common real degeneration shape and neither could see it.

**The instrument that would have caught it was rejected on a misread.** Distinct-word ratio
falls monotonically here (0.474 -> 0.244) and separates the healthy band (0.42-0.53) from
the broken one (0.12-0.29) cleanly. It was excluded because docs/investigations/int4-scales.md records that a
distinct-token gate INVERTS — hybrid has the worst ratio and the second-best perplexity.
That warning is about ranking *healthy* configs, where the ratio does not track quality.
Generalising it to "never use distinct ratio" cost three rounds of device time. It is an
alarm, not a ranking metric, and `telemetry::repetition_report` now uses it as one.

**What survives.** Round 1 (512 tokens, 42 cells) — the mode ranking, top-m's substitution
cost, and int4's intermittent NaN. Everything context-dependent needs re-running, and not
with free-running decode: at 4096 tokens EVERY configuration degenerated, so the fixed
forced-token harness is no longer optional for long-context work.

**What was right for the wrong reason.** "DSA degenerates at 10k" is true, but not
distinctive — dense and streaming degenerate at 4096 too. DSA's is merely the only one an
exact matcher could see.

## Benchmark matrix: mode x attn x cache-policy, 115 GiB, 512 -> 10k tokens

Bracket 44 -> 8 -> 4 -> 2 at 512 / 2048 / 4096 / 10000 tokens, `--max-mem 115`, one
process per cell, ~10 h total. Full data in `tests/bench-matrix.sh`'s output dir.

**Result: `int3-vq` + `streaming` + `2q` — 3.26 tok/s at 10k, and the only cell that gets
FASTER with context** (2.81 -> 2.97 -> 3.06 -> 3.26, +16% over 512 -> 10k).

### The four findings that matter

**1. DSA degenerates catastrophically at 10k context.** `int3-vq/dsa/2q` was clean at 512,
2048 and 4096 (longest-repeated-block 6, 14, 16 tokens) and then collapsed at 10k:
**lrb 4544 of 10000 — 45% of the output is a verbatim duplicate**, verified as real text
(a 6000+ char block recurring), not a hash collision. Its 2.31 tok/s is therefore not a
result. `streaming` at the same length is clean (lrb 142, 1.4%). This is a QUALITY failure
in the sparse-attention path, invisible to any throughput metric, and it was caught only
because the run classifies output. The indexer is the suspect: it scores every cached
token and selects a subset, and something about that selection at ~10k positions loses the
context the model needs.

**2. `streaming` wins by doing less, and the benchmark cannot see the cost.**
`--sinks 4 --window 512` attends **516 rows regardless of context** — 100% of context at
512 tokens, 25% at 2048, 12.6% at 4096, **5.2% at 10k**. It ranked 11th at 512 (where it
attends everything) and 1st from 2048 on. The throughput is real; whether the output is
worth having at 5% context coverage is unmeasured here. Treat it as an approximation that
happens to be fast, not an optimisation.

**3. DSA does not bound its cost the way its design implies.** It fell 14-21% from 512 to
4096 — the steepest declines in the bracket — because the attend is capped but the INDEXER
scores every cached token (4.7 ms/tok over 16.836 scoring layers at 10k — a RUN AVERAGE,
not a contradiction of NPU.md's 21.000: layers below `index_topk`=2048 return dense without
scoring, so the first ~2048 positions score nothing and 21 x (10000-2048)/10000 = 16.7).
Its cost curve
resembles dense's. Any fix belongs in the indexer, not the attend kernel.

**4. `top-m`'s premium collapses to noise as context grows.** Fastest cell at 512
(3.22 tok/s, all four top-4 slots) and by 4096 it is +1.6% over `dsa/2q` (2.54 vs 2.50)
while substituting **5.48-5.62%** of experts away from the true top-K. It buys its 85-86%
hit rates by routing to whatever is resident; that hit rate is not comparable to the other
policies'. At 512 tokens the substitution looked free. It is not.

### Mode ranking at 115 GiB, and a correction

| mode | ok | tok/s @512 | mean |
|---|---:|---|---:|
| `int3-vq` | 16/16 | 2.70-3.22 | **2.91** |
| `hybrid` | 12/12 | 2.12-2.76 | 2.45 |
| `int4` | **14/16** | 1.77-2.42 | 2.10 |

int3-vq beats hybrid by 19% here, where docs/investigations/int4-scales.md records hybrid ahead (2.72 vs 2.62 at
`--max-mem 100`). Two things differ: the budget (115 GiB gives int3-vq's 15.34 MB experts
~6900 slots vs int4's 5274, so a larger pool favours the smaller format) and **the
prompt** — those numbers were free-running on "The sky is blue because", where hybrid
degenerated to lrb 256/512 and int3-vq stayed at 20. Hybrid is the mode that confound
flattered most. The speed half of that comparison needs re-measuring; hybrid still holds
the better perplexity (5.189 vs 5.275), which this matrix does not measure.

### Non-finite logits: intermittent, NOT int4-specific, and not yet reproduced under instrumentation

> **Superseding the section below, which called it int4-specific.** It is not. Five
> occurrences across two modes: `int4/dense/arc`, `int4/streaming/2q`, and
> **`int3-vq/streaming/2q` three times** — a config the matrix reported 16/16 clean. By
> attention mode: streaming 3, dense 2. By policy: arc 2, 2q 3, **lru 0, top-m 0**.
>
> **It is a race.** Adding a per-layer host copy (`--checksum-x`) makes it vanish
> entirely — 0 non-finite across 78 layers of every position on a config that fails
> without it. A Heisenbug a sync closes is a timing fault.
>
> **Ruled out:** fetch/reaper failure (no poison, no reaper error in any failing log),
> prefetch-admitting-before-load (no prefetch path exists), tier migration on hit (both
> `HybridArc::get` and `HybridTwoQ::get` explicitly keep a hit in its tier).
>
> **The invariant holds in normal operation.** A `trace`-only check now verifies on every
> cache HIT that the key's bytes actually landed since admission: ~3.7M checks, zero
> violations. Poison-filling freshly admitted slots (`0x7FC0_7FC0`, quiet NaN in f32 and
> both bf16 halves) loaded 15,656 slots correctly with coherent output. So this is not a
> systematic read-before-write.
>
> **Not reproduced since instrumenting: 52 consecutive clean runs.** That is weaker
> evidence than it looks, and the reason is a mistake worth recording: the first 28 runs
> were `dense`-only when 3 of 5 failures were `streaming`; the 16 poison runs add ~78
> `device_sync` calls per token and almost certainly mask the race, exactly as
> `--checksum-x` does; and the 6 uninstrumented control runs at 2048 tokens on the one
> twice-failing cell were **underpowered by design** — that cell fails ~1 in 7, so
> P(0 in 6) = 0.40. **Fifteen runs of that cell (~3 h) is the minimum for a 90% chance of
> one event.** Three separate underpowered experiments were run in this investigation
> before anyone computed the power; compute it first.
>
> **Leading remaining suspect:** arena compaction (`src/arena.rs` slot relocation), which
> `arc` and `2q` exercise far more than `lru` because their tier boundary floats. Matches
> the 5/5 policy correlation. Not verified.
>
> **The part that is worse than the crash.** NaN needs a slot that was NEVER written. On a
> warm pool the same race reads the evicted expert's bytes instead: finite, plausible,
> silently wrong. A clean run is not evidence of absence, and some published numbers could
> be quietly affected.

### Original (int4-specific framing, superseded above): `int4` throws NaN/Inf intermittently

**2 of 16 int4 cells** died with `logits are non-finite`: `int4/dense/arc` and
`int4/streaming/2q`. `int3-vq` and `hybrid` were 28/28 clean, and neither failing cell
reproduced in round 2. No combination predicts it — `2q` passed under dense and failed
under streaming, `arc` did the reverse — which is the signature of a timing-dependent
fault rather than a logic one, plausibly widened by int4's 20.05 MB fetches taking ~30%
longer than vq3's. **`hybrid` is unaffected despite using int4 for its hot experts**,
which points at the all-int4 residency path rather than the int4 decode kernel. Not yet
root-caused.

### Method notes

- **The prompt was manufacturing the degeneration.** On the default short prompt,
  hybrid/dense/lru returned lrb 256/512 at 2.88 tok/s; on a prompt that sustains a long
  answer, lrb 18 at 2.66. **The broken run benchmarked 8% FASTER** — looping re-routes to
  the same experts, the hit rate rises. Round 1 was restarted for this reason and returned
  0 degenerate cells out of 44.
- **Do not select rounds on raw tok/s.** A pure top-8-by-tok/s at 512 would have cut at
  2.91 and **eliminated `int3-vq/streaming/2q` at 2.81 — the eventual winner.** At short
  context the capped-attention modes carry their overhead without their benefit. Rounds
  were seeded to preserve the mode and attention comparisons instead.

## Speculative decode (`--mtp`): the batched verify pass LOSES 7%, and why

**2026-07-31**, int3-vq / dense / 2q, `--max-mem 115`, 128 tokens, chat-framed prompt,
sole tenant. Both runs back-to-back from the same binary with no rebuild between them
(a `cargo build` evicts page cache and moved `ms/miss` 1.36 → 5.14 in a discarded pair).
MTP ran FIRST, so any residual page-cache advantage went to the baseline.

| | tok/s | ms/tok | route | moe | moe gpu | miss/tok | GB/tok | hit |
|---|---|---|---|---|---|---|---|---|
| sequential | **2.69** | 372.3 | 101.5 | 251 | 243 | 147.3 | 2.26 | 75.3% |
| `--mtp` (2-row verify) | **2.50** | 399.4 | 84.4 | 292 | 285 | 197.2 | 3.02 | 74.1% |

**0.93×.** Completions are BYTE-IDENTICAL (203 bytes, `cmp` clean) — which is the point of
the exercise as much as the speed is: every batched kernel is bit-identical per row
(`tests/kernel.rs::batched_rows_are_bit_identical_to_single_rows`), row 0 of a verify pass
is the real token, and a union expert a row did not route to carries weight 0 and is
skipped. So the speculative engine has no freedom to differ, and `diff` is the whole
quality gate.

Acceptance 38/90 = 42.2%, i.e. **1.422 tokens per verify pass**.

### The two halves pull opposite ways

- **Attention batches for free, as predicted.** `route` (the gate-D2H wait, which absorbs
  the attention GPU time) went 101.5 → 84.4 ms/tok = **0.83×**. Per PASS that is 1.18× for
  two token rows: `q_a`/`q_b`/`kv_a`/`o_proj`/`kv_b`/`lm_head` are dense weights read once
  per layer whatever the row count, so the second row costs arithmetic and nothing else.
- **The MoE does not.** 251 → 292 ms/tok = 1.16×, or **1.66× per pass**. A batched pass
  routes twice and must launch the UNION: measured 13.49 routed + 1 shared = 14.5 experts
  against a single row's 9, i.e. **1.61×**. Per expert the second row IS free (0-miss
  layers: 2585 µs / 14.5 experts = 178 µs, against the baseline's 1582 / 9 = 176 µs), but
  1.61× the experts is 1.61× the weight reads, and the weight read is the whole cost.

### Break-even is 1.53 tokens/pass — the run landed at 1.422

Blending the two at their measured shares of the wall (MoE 67%, attention 27%, tail 6%):

```
cost(verify pass) / cost(sequential pass) = 0.67·1.74 + 0.27·1.09 + 0.06·1.2 = 1.53
```

so the pass has to yield 1.53 tokens to break even — **~53% acceptance**. This run gave
42.2%; the earlier sequential measurement of the same head over the same 128 tokens gave
53.5% (68/127). The two samples' 95% intervals overlap, so this is not a structural loss
so much as a coin flip that landed on the wrong side. It is nonetheless the measurement,
and 0.93× is the number.

### The prediction was wrong, and the error is worth naming

The pre-implementation estimate was **1.27–1.33×**. It applied the union factor (1.687) to
FETCH and the row-batching factor (1.079) to COMPUTE — but compute scales with the union
too, because each of the union's experts needs its own weight read. Correcting that term
alone gives 1.61/1.422 = 1.13× on the MoE, which is what happened (1.16× measured). The
fetch half of that estimate was fine (2.26 → 3.02 GB/tok = 1.34×, still 97% hidden).

Second time a fetch/compute projection on this engine has gone the wrong way. Re-measure.

### What would flip it

Only the acceptance rate. Skipping zero-weight rows inside the expert kernels would not
help: `R=2` costs 1.079× of `R=1`, so the per-row arithmetic is ~8% of an expert launch
and the other 92% is the weight read, which the union forces regardless. GLM-5.2 ships
`num_nextn_predict_layers = 1`, so a deeper head is not available either — depth-2 chained
drafts were measured at 4.4%, which is why the pass is 2 rows and not 3.

### 512-token confirmation, and a normal prompt

`--prompt "What causes the seasons on Earth?"`, same config, 2026-07-31:

| | tok/s | ms/tok | route | moe | miss/tok | GB/tok | hit |
|---|---|---|---|---|---|---|---|
| `--no-mtp` | **2.63** | — | — | — | — | — | 76.1% |
| default (speculative) | **2.49** | 401.6 | 87.2 | 289 | 181.8 | 2.79 | 75.5% |

**0.95×**, and completions BYTE-IDENTICAL over all 512 tokens (2195 bytes, `cmp` clean).
Acceptance 161/350 = 46.0% = 1.460 tokens/pass, against the 1.53 break-even. Confidence
separates cleanly and monotonically: 6% / 24% / 30% / 57% / 91% over n = 32/88/84/58/88.

**Neither run terminates.** Both hit the 512 cap without emitting EOS, both degrade the
same way. Byte-identical with and without speculation, so it is NOT the verify pass.

### It is not a repetition loop, and it is not the engine

The `distinct 0.193 / longest repeated block 77` metrics say "repetition collapse" and
that reading is WRONG — they fire on a clean loop and on this, and the difference is the
whole clue. The text is SPLICED, not looped: `the Northern and Southern Hem` cuts
mid-phrase into a sentence from two paragraphs earlier; `away from the Polaris` where the
source clause said "the Sun"; `**23.5 degrees**` re-emitted as `**23. 5 degrees**`; and
` it is AI-generated text. Please verify the information.` appears mid-sentence though the
phrase is in no part of the context. Half-copies from context plus training-frequency
attractors — not an attractor the sampler is stuck in.

Root-caused 2026-07-31 as **model behaviour, not an engine fault**:

- Teacher-forced PPL reproduces the recorded gate figure EXACTLY — 5.222720, mean NLL
  1.653018 (docs/reference/architecture.md "What gates this change"), six decimal places.
- NLL is FLAT by position over 762 predicted tokens, in fact slightly improving
  (Pearson r = -0.105; bucket means 1.893 / 1.469 / 2.083 / 2.013 / 2.006 / 1.263 / 1.168 /
  1.329). A KV row-index, rope-position or attention-window bug climbs with position. This
  does not, and it stays healthy well past the ~40 tokens where free-running corrupts.
- The two 512-token runs are byte-identical at 97,542 vs 77,733 misses — different cache
  pressure, same output, so no expert-stream race.
- **`nll_forced` and `generate` call the same `self.forward(token, pos)`.** There is no
  code path free-running takes that teacher-forcing does not. The engine cannot be
  behaving differently; only the token sequence differs.

The mechanism is in the same data: **52/762 = 6.8% of teacher-forced tokens sit above
NLL 5** — near-random even given a known-good prefix. Greedy has no escape from those, and
one bad argmax conditions everything after it.

What this does NOT establish is how much of that 6.8% is int3-vq damage versus GLM-5.2
under greedy: the repo has no unquantized arm, so the attribution is unmeasured. (An earlier
draft of this paragraph cited "the ladder is int3-vq 5.28 > hybrid 11.55 > int4 73.43" —
those are the PRE-fix numbers and the ordering is inverted. Post-fix it is int4 5.120 >
hybrid 5.189 > int3-vq 5.275. Every arm is still quantized, which is the part that
mattered, but quoting the dead ladder is exactly the drift this file keeps catching.) If it
is quantization, the lever is a sampler (temperature/top-p) — the engine is greedy-only —
not a kernel. **Measured 2026-07-31, and it moves this a long way:** int4 on the same
prompt produces correct physics for 512 tokens and self-corrects mid-completion. So the
6.8% is at least substantially int3-vq's format, not the checkpoint's ceiling.

**A caution on method.** A first attempt to localise the fork by teacher-forcing the model
on its OWN output was discarded as confounded: it fed raw text through `--ppl` while the
generation had been chat-framed, and re-tokenized instead of replaying the emitted ids, so
positions did not align. Its "NLL 16.6" spikes were tokenizer boundaries. Replaying the
ids needs a `--ppl` that takes ids, which does not exist.

## `--mode int4` vs int3-vq — the point estimate favours int4, the test cannot confirm it

**2026-07-31**, one binary, one session, `tests/ppl-corpus.txt` (762 teacher-forced
tokens), 2q / `--max-mem 115` / dense. `--ppl` never enters `generate`, so speculative
decode is not a variable.

| mode | PPL | mean NLL | hit % | 762 tok in | slot bytes | pool |
|---|---:|---:|---:|---:|---:|---:|
| int3-vq | 5.222720 | 1.653018 | 78.17 | 283.9 s | 15,335,424 | 6888 |
| **int4** | **5.154898** | **1.639947** | 70.86 | 365.3 s | 20,054,016 | 5274 |

Paired (`bin/ppl`): mean dNLL **−0.01307**, sd 0.5279, SE 0.01913, 95% CI
**[−0.05056, +0.02441]**, worse% 57.3, dPPL −1.299%.

**INCONCLUSIVE, and it must be reported that way.** The interval straddles zero and SE
exceeds the 1%-PPL bar of 0.00995 nats, so this corpus cannot resolve the question at any
point estimate — `bin/ppl` says so itself and asks for ~2021 tokens. int4 being ahead on
PPL is consistent with docs/investigations/int4-scales.md §10's independent 5.120 vs 5.275434, but 762 tokens do
not establish it here. `tests/ppl-corpus-5000.txt` exists and would settle it.

Note `worse%` 57.3 with a NEGATIVE mean dNLL: int4 is individually worse on most tokens
and wins by a wide margin on a minority. That is a distribution worth knowing about before
reading −1.3% as a uniform improvement.

**int4 is the slower mode**, and structurally so: its slot is 20.05 MB against int3-vq's
15.34 MB, so the same budget holds 5274 experts instead of 6888 and the hit rate falls
78.2 → 70.9%. 762 tokens took 365 s against 284 s.

### Free-running, the difference is not 1.3% — it is night and day

Same prompt that corrupted under int3-vq ("What causes the seasons on Earth?"), 512 tokens:

| mode | distinct | longest repeated block | the completion |
|---|---:|---:|---|
| int3-vq | 0.264 | 77 | spliced wreckage: half-copies cutting mid-phrase, `**23. 5 degrees**`, boilerplate with no source in context |
| **int4** | 0.279 | 77 | correct physics end to end, and it CATCHES ITS OWN SLIP — emits "tilted toward the Earth", then a `**Corrected Version:**` block restating the paragraph with "tilted toward the **Sun**", then continues into an accurate Q&A on insolation angle and daylight hours |

**The metrics cannot see this.** distinct 0.264 vs 0.279 is nothing, and `longest repeated
block` is 77 for BOTH — for int3-vq it is corruption, for int4 it is the deliberate
`**Corrected Version:**` restatement, which is legitimate prose. Third independent
demonstration of docs/investigations/int4-scales.md §1, after §10's hybrid case and the 2026-07-31 root-cause:
the degeneration gate does not measure quality, and here it ranks a coherent completion
level with a broken one.

Neither reaches EOS in 512, but for opposite reasons — int3-vq cannot escape its loop,
int4 is working through a self-generated Q&A series and would plausibly continue for a
while. int4 is 2.06 tok/s against int3-vq's 2.63.

**So the 1.3% PPL gap badly understates the practical gap.** Teacher-forcing pins the
prefix and measures one step; free-running compounds every step into its own input, and
that is where a small per-token edge turns into the difference between physics and
splices. A PPL delta this size is not a safe proxy for decode quality in either direction.

### The `.i4` set on the reference artifact is group-128, proven by its length

`manifest.json` carries no `i4_source`, and `main` used to REFUSE to load on that basis.
It no longer does, because the absence proves nothing while the slab length proves
everything: `ExpertSet::open` requires `len == (n_experts + 1) * i4_expert_stride`, and the
stride is a function of the group size. The file is **5,153,882,112 B = 257 × 20,054,016**,
which only group 128 produces — group 64 would be 21,233,664 and per-row 18,915,328 per
expert. `format.rs::I4Source` already said the stamp "is a diagnosis, not a load-time
guard"; the `None` arm had made it one. A stamp that POSITIVELY disagrees still bails.

---

## The MTP confidence gate — 1.108×, 2026-07-31

Supersedes "Speculative decode (`--mtp`)" above as the verdict on the feature: that section
measured only the **ungated** form, which is still a loss. Shared-GPU caveat: these ran
under `flock /tmp/rivoli-gpu.lock` with another tenant present, so pin-build times varied
33 s → 164 s and absolute tok/s is noisier than the sole-tenant runs above. **Rank on
tokens/pass and on the within-batch pairing, not on tok/s across sections** — tokens/pass
is pure loop arithmetic and is contention-immune.

512 tokens, `--mode int3-vq --attn dense --cache-policy 2q --max-mem 115`, memory-systems
prompt (the coherent one; the seasons prompt trips the degeneration warning and is a
different regime — see below).

| arm | tok/s | tokens/pass | speculated | acceptance |
|---|---:|---:|---:|---:|
| `--no-mtp` (sequential) | 2.68 | 1.000 | — | — |
| `--mtp-min-conf 0` (ungated) | 2.66 | 1.657 | 100% | 65.7% |
| **`--mtp-min-conf 0.8` (default)** | **2.97** | 1.459 | **50%** | 66.4% |

**1.108× over sequential**, and the three arms are **byte-identical** (1965 B of generated
text, `cmp` clean across all three). Gating LOWERS tokens/pass while RAISING throughput —
half the passes are cheap single-row ones — so tokens/pass stops being the figure of merit
once the gate is on.

### Why a fixed threshold is safe: the calibration does not move

`RIVOLI_MTP_PROBE=1` buckets acceptance by the draft's own top-1 probability and, separately,
by the MAIN MODEL's confidence in the token the draft was tested against.

> **The probe was deleted 2026-08-01 and this table cannot be re-run as written.** It was an
> env var, which `CLAUDE.md` forbids — invisible to `--help`, absent from the recorded
> command line right here, silently active in a build that looks stock. It had answered its
> question: "de-quantize the draft head" is REFUTED (46.0% → 49.4% acceptance, 49% → 54%
> target-conditioned, Δ = 3.4 pp ± 7.4 — noise). Recover it from tag
> `archive/mtp-target-probe`. **The `draft conf` rows are still measurable from a stock
> build**: `mtp_bins` is the DRAFT-side histogram and was deliberately kept, because it sizes
> the live `--mtp-min-conf 0.8` gate that is on by default and worth 1.108×. Only the
> `target conf` rows needed the probe.

| | seasons (int3-vq) | memory (int3-vq) | seasons (int4) |
|---|---:|---:|---:|
| n | 350 | 309 | 342 |
| acceptance | 46.0% | **65.7%** | 49.4% |
| accept @ draft conf ≥0.8 | **91%** | **91%** | **91%** |
| accept @ draft conf 0.6–0.8 | 57% | 57% | 66% |
| accept @ target conf ≥0.8 | 49% | 76% | 54% |
| share of drafts ≥0.8 | 25% | 52% | 23% |

The ≥0.8 bin lands at 91% across two prompts and two quantizations. What moves is the
**mass**, not the curve — so the threshold does not need per-prompt tuning, which was the
main hazard the gate was expected to carry. Against a ~1.53× verify pass the break-even is
~53%, so 0.8 clears it with margin and 0.6 (57%) would not.

### Two things this refuted

**`d`, the draft pass, is 16–19 ms ≈ 0.045 of a sequential pass** — it had been *inferred*
at 0.01 from closing the cost model. Measured, it is 4× that, but still small enough that a
gate placed BEFORE the draft would buy ~2% and would also go blind: the post-draft gate
scores its skipped drafts for free against the plain pass's own `t1`, so bins it stops
speculating on keep filling. Post-draft gating has no explore/exploit problem; pre-draft
gating would.

**"De-quantize the draft head" is refuted.** The hypothesis was that int3-vq damage to layer
78 was costing acceptance (published MTP designs report 85–90%). Rebuilding the head at int4
moved acceptance 46.0% → 49.4% (Δ = 3.4 pp, SE 3.8, CI [−4.0, +10.8]) and target-conditioned
49% → 54% (Δ = 5 pp, CI [−3.4, +13.4]). **Both within noise.** Acceptance tracks the TEXT,
not the head's precision: 65.7% on coherent generation versus 46–49% on the sample that
trips the degeneration warning, where the model itself is near-random (6.8% of teacher-forced
tokens sit above NLL 5) and greedy cannot escape a bad argmax. Caveat: `--mode int4` changes
the main model as well as the head, so this is not a head-only comparison — it is the one
that was available, and it shows no movement.

int4 is not the throughput play regardless: 1.80 tok/s at 66.4% expert hit, against
int3-vq's 2.68 at 72.2%.

### Hybrid is not output-neutral under cache changes — 2026-07-31

Found while checking the batched int4 kernel: hybrid sequential and hybrid speculative
produce different text. The kernel is NOT the cause. Controls, all `--mode hybrid --attn
dense --cache-policy 2q --bench 512`, seasons prompt:

| run | flags | bytes | expert hit | vs previous |
|---|---|---:|---:|---|
| A | `--no-mtp --max-mem 115` | 2100 | 70.9% | — |
| B | `--no-mtp --max-mem 115` (repeat) | 2100 | 70.9% | **byte-identical to A** |
| C | `--no-mtp --max-mem 70` | 2167 | 53.4% | **differs from B at line 2** |

So hybrid is deterministic run-to-run and **not** stable across cache configurations, with
no speculation involved in any of the three. `int3-vq` and `int4` are byte-identical under
speculation (1965 B and 2003 B, sequential vs gated vs ungated) — they are single-format.

Mechanism: `Pin::submit_layer` fills `fmt` from the HOT/COLD slab placement and `gpu.rs`
branches on it to choose `moe_expert_range_i4` vs the VQ launcher, so in hybrid **residency
selects the arithmetic**. INV-1 ("routing never consults the cache") holds and its test
passes; it was being read as the stronger claim that cache changes are output-neutral, which
is true only for the single-format modes.

Consequence for measurement: **do not A/B quality across `--max-mem` or `--cache-policy` in
hybrid** — the arms are different numerics, not the same model under different pressure.
Open defect; fixing it means binding format to expert identity rather than residency, which
changes what `hybrid` is.

### The demand fetch is at the drive's ceiling; the drive is idle 35% of the token — 2026-08-01

Chasing "can the disk fetches run fully concurrently with the rest of the system". Baseline
`--mode int3-vq --attn dense --cache-policy 2q --max-mem 100 -bench 64`: 441.9 ms/tok wall,
`moe` 328 ms, `fetch` 286 ms, `io_wait` 286 ms, 187.75 miss/tok × 15.34 MB = 2.88 GB/tok.

**The MoE phase IS the disk.** Summing `moe_us_by_miss` over the run gives 20.5 s of MoE
bracket against 18.3 s of `io_wait` — 89%. A zero-miss layer is 1585 µs, so all-resident
compute is ~2 s of that 20.5 s.

Drive characterised with `docs/measurement/probes/fetch_batch.hip`, which reproduces the engine's shape
exactly (`hipHostMalloc` bounce buffers, submit-*m*-drain-all-*m*, random 15.3 MB O_DIRECT
reads across the 75 layer files, GPU kept busy streaming LPDDR5):

| | QD1 | QD2 | QD4 | QD8 |
|---|---:|---:|---:|---:|
| GPU busy | **7.7** | 12.1 | 13.0 | 10.9 |
| GPU idle | 8.8 | 12.6 | 13.3 | 13.7 |

Engine achieves ~10 GB/s. Weighting the table by the engine's own miss distribution predicts
15.8 s against the measured 18.3 s — inside the probe's run-to-run spread, which is **±25% at
QD1 alone** (7.7–12.5 GB/s across runs of the same probe). An early "26% unexplained" was
that spread plus an idle-GPU comparison; it did not survive matching the load.

Three hypotheses tested and killed, in order:

| hypothesis | measurement | verdict |
|---|---|---|
| the bounce→VMM copy is the serial tail | 0.18 ms/expert at 87 GB/s (`fetch_stream_ops.hip`) vs a ~1.3 ms read | no |
| the pinned arena costs read bandwidth | pinned vs pageable within noise at every QD | no |
| splitting a read raises QD for free | one expert split 2 ways: 1.94 → 1.44 ms — but only 18% of layers miss exactly once | **~2% overall; dropped** |

`--direct-vmm-dma` re-ablated on the same build: 900.8 ms/tok, 1.11 tok/s vs bounce's 2.26,
`io_wait` 761 ms — confirms bounce as the default and, incidentally, exposed the broken
hidden-fetch metric (it printed **99% hidden** on this arm). *(2026-08-01: both are gone —
the flag, because this arm is the whole case against it; and `fetch_hidden_pct`, because a
metric whose ordering disagrees with throughput is not reporting a small error. See the
DIRECT vs BOUNCE ablation above.)*

**What is left is the duty cycle: 18.3 s of NVMe inside a 28.3 s decode.** The ring only has
work between a layer's routing and its MoE launch. Filling the idle 35% needs the routing
known a layer ahead — prediction, not a fetch change. Floor if it were free:
184.3 GB ÷ 12.5 GB/s ≈ 14.7 s, i.e. ~1.9×.

### Deleting the per-read `Signal`: perf-neutral, and it fixed a hang — 2026-08-01

`asyncfetch` armed one `Signal` (a `hipLaunchHostFunc`, a host round trip recorded INTO the
fetch stream) per cold read. Nothing awaited it — `gpu.rs` took the `Vec<Signal>` and dropped
it — since the ticketed dataflow moved the dependency onto the device. Probe cost: 7.2 µs of
enqueue per read inside the `io_wait` clock, plus 4% of the fetch stream's drain.

Paired A/B, both binaries built BEFORE either arm ran and alternated with no build between
(the first attempt showed 2.26 → 2.54 and was an artifact of the build evicting page cache —
exactly the trap CLAUDE.md warns about):

| arm | tok/s | n | mean | sd |
|---|---|---:|---:|---:|
| old | 2.50, 2.76, 2.24, 2.68, 2.34 | 5 | 2.504 | 0.222 |
| new | 2.44, 2.60, 2.59, 2.47, 2.50, 2.27 | 6 | 2.478 | 0.121 |

**Perf-neutral**: Δ = −0.026 ± 0.111 tok/s (SE of the difference), nowhere near resolvable.
Output byte-identical, hit/miss identical at 29353/16094, `moe_us_by_miss` histogram identical
bucket for bucket.

Note the run-to-run spread on this bench is **±0.2 tok/s, ~9%** — wider than most of what
gets ranked in PERF.md. At n=5 the new arm looked three times more consistent (sd 0.070) and
that was tempting to attribute to the deleted host callbacks; the sixth run (2.27) took the
sd to 0.121 and F to 3.4, p ≈ 0.10. **There is no variance finding here**, only a reminder
that this bench needs n ≥ 5 per arm before any single-digit percentage is worth saying.

The value is elsewhere: the reaper's poison path resolved those signals and **never signalled
the tickets**, so a fetch error left every consumer parked on a `hipStreamWaitValue64` nothing
would ever satisfy — the device hung instead of the decode returning the error. Teardown now
calls `Timeline::release` (host-side monotone store on ROCm, `vkSignalSemaphore` on Vulkan).
Registered as **INV-6**, with a test per backend that proves the mechanism rather than the
bookkeeping: a host release retires a wait enqueued against a value no stream will ever write.

### Cross-layer prefetch: the predictor works, the window does not — 2026-08-01

`--features rocm,pred-probe`, then `--pred-probe --mode int3-vq --attn dense
--cache-policy 2q --max-mem 100
--no-mtp -bench 64`. `--no-mtp` deliberately: with speculation on, the union carries two
routers' picks and a row-0 prediction would be scored against a denominator it never saw.

At the top of each MoE layer — before attention adds into the residual — run the layer's own
router on `post_ln(x)` and compare its top-8 against what the real router picks afterwards.
That is exactly the information a prefetch issued into the attention window would have.

| | |
|---|---:|
| recall on the top-k | **83.9%** (37236/44400) |
| **recall on the MISSES** | **82.7%** (12563/15191) |
| reads it would issue | 16306 |
| of those, wasted | **23.0%** |

Better than LOOKA's 77.2% at L+1 (2026-07-30), which is what the mechanism predicts: this
predictor is missing only the attention residual where LOOKA was missing that *and* the
previous layer's MoE output. **Prediction accuracy was never the blocker.**

The economics, per pass (74 passes, 15.335 MB/read, measured 1.32 ms/miss):

| | reads/pass | drive time |
|---|---:|---:|
| demand misses today | 205.3 | 271 ms |
| prefetch would issue | 220.4 | 291 ms |
| — useful (a demand miss, started early) | 169.8 | — |
| — **wasted** | **50.7** | **+67 ms/token** |
| demand misses still unpredicted | 35.5 | 47 ms |

Gain ceiling is the idle window, **85 ms/token** (`route_wait`, the host blocked on the gate
D2H = attention GPU time, with an empty io_uring ring throughout). The waste costs 67 ms and
the predictor's own rmsnorm+gemv+D2H ~6 ms, so the net ceiling is **~+12 ms on a 397 ms
token — 3%** — and that assumes the window is filled perfectly.

**It cannot be. The window is 1.13 ms per layer and one 15.3 MB expert read is ~2 ms at
QD1.** It fits 0.74 of a single read where a layer needs 2.9, and a layer's fetch ends when
its LAST read lands, so starting a subset early moves the batch by much less than the window.
The prefetch necessarily spills into the MoE phase — the saturated regime `b372cd4` measured.

So three explanations have now been offered for why this does not pay, and two are false:

| explanation | verdict |
|---|---|
| "on a bandwidth-bound path overlap creates no bandwidth" (`b372cd4`) | **false** — the drive idles 35% of every token |
| "the predictor cannot see far enough" | **false** — 82.7% recall on misses |
| the idle window is shorter than one expert read, and the waste costs more than it returns | **holds** |

`b372cd4`'s A/B result stands; its stated reason does not, and it prefetched during the MoE
phase rather than into the idle window, so it never tested the case its conclusion was read
as closing. What would change the answer is a smaller expert or more compute per layer —
PERF.md #2, not prefetch.

## Layer-major prefill (`--layer-major-prefill`) — prefill 2.15x, output byte-identical (2026-08-02)

Prefill the prompt LAYER-MAJOR: every token through layer L before any token reaches L+1,
instead of walking all 78 layers per token. Same arithmetic, different order.

```
./target/release/rivoli /var/db/rivoli/glm52-vq3-full -bench 16 --mode int3-vq \
  --attn dense --cache-policy 2q --max-mem 115 --prompt "$(head -c 3200 tests/ppl-corpus-5000.txt)" \
  --dump-ids <out>            # and the same again with --layer-major-prefill
```
658-token chat-framed prompt, MTP on (the default), both arms inside ONE lock hold so the
page cache is not disturbed between them.

| | token-major | layer-major | |
|---|---:|---:|---|
| prefill wall | 282.7 s | **131.7 s** | **2.15x** |
| expert reads (prefill) | 104 991 | **18 558** | **5.66x fewer** |
| reads/token | 159.56 | **28.20** | |
| prefill hit rate | 73.8 % | **94.7 %** | |
| total wall, 16 tokens | 287.8 s | **139.6 s** | 2.06x |
| total misses | 107 102 | **22 059** | 4.9x fewer |
| decode | 393.3 ms/pass | 610.5 ms/pass | **1.55x WORSE — see below** |
| token ids | 16 ids | 16 ids | **byte-identical** |

### The read count is the mechanism, and it lands on the floor

18 558 reads is the **compulsory** count — one read per distinct `(layer, expert)` pair, what
no policy can beat. The offline prediction from the preserved 769-token trace was 18 474 for
the model's 75 MoE layers; this run is a 658-token prompt and *includes* the MTP head's own
MoE layer, so ~18.5k is the same number. Token-major does not fail to cache — it fails to
cache LONG ENOUGH: layer L's experts are evicted somewhere in the next 77 layers, so the next
token re-reads them. One layer's experts are 256 x 15.34 MB = 3.93 GB against a 6874-slot
pool, so layer-major never has to evict anything it is about to want.

### It is byte-identical, and that is checked rather than argued

`tests/layer-major-neutrality.sh` is the standing gate, and it asserts BOTH halves: token ids
identical AND reads/token actually reduced. Neither alone is a gate — the first passes on a
build where the flag does nothing, the second on a build that produces garbage cheaply.

### What it does NOT buy: the LPDDR5 re-read. This is where the rest of the win is.

The reads-only arithmetic said 13-15x and that estimate was WRONG, because it priced NVMe
bytes and FLOPs and not expert weights re-read from RAM. A pass is still `MAXROW` = 2 rows
wide, so each 2-row pass streams its experts out of LPDDR5 exactly as before; layer-major
only halves how often (2 rows share one read instead of 1). Per layer per token that is
~115 MB against ~138 MB — 17%, not 6x. Once the fetch stops dominating, THAT traffic is the
bound, and 2.15x is what the two effects come to together.

Widening a pass needs general-`R` MoE kernels. `moe_gateup_vq`/`moe_down_vq` and the int4
twins are templated at `R <= 2` and return 1004 above it, and a genuinely wide `R` cannot be
one more `acc[R]` register slot: at R=769 the activations are 18.9 MB, larger than L2, so the
x-side would stream from DRAM once per output-row block and cost more than the weights it
saved. It wants LDS tiling on both operands — a real GEMM kernel, plus its Vulkan twin.

### The surprise: decode starts COLD after a layer-major prefill

**Decode got 1.55x slower per pass (393.3 -> 610.5 ms), and it is cache shape, not a bug.**
Token-major prefill ends with the last token having walked all 78 layers, so the pool's
most-recent entries span every layer — exactly what the first decode token wants. Layer-major
ends holding layer 77's experts for every token, so decode restarts cold at layers 0..76.

At `-bench 16` that cold start IS the whole decode, so this is the worst case for the metric;
total misses still fell 4.9x and total wall still halved. But the trade is real and its sign
depends on the workload: long prompt + short answer wins big, short prompt + long answer may
not. **Not measured at a realistic generation length, which is the missing number and the
reason the flag is off by default.**

### Incidental: `d` was reported against the wrong denominator

The token-major arm printed `MTP draft cost: 19.5 ms/draft over 671 drafts = 255.8% of decode
wall`. `mtp_draft_ns`/`mtp_draft_n` accumulate across the PREFILL's per-token drafts but are
reported as a share of DECODE wall, so a 658-token prompt charged 657 prefill drafts against
a 16-token decode. Predates this work — the same rebase bug `fetch_ns` had — and was
invisible only because both arms mis-counted equally until layer-major started using the
tail-less `mtp_fill`. Rebased with `hit0`/`miss0`/`fetch0`/`io0`.

### Coverage: every attention mode, and the offline sim it cross-validates

`tests/layer-major-neutrality.sh`, `--no-mtp`, `-bench 8`, 2q, `--max-mem 115`. Wall times
here are INDICATIVE ONLY — this script takes the lock per arm (so it does not reserve a
shared GPU for an hour), which means another tenant's run can land between the two arms. The
id comparison and the read counts are immune to that; only the seconds are not. The 2.15x
headline above came from a pair inside one lock hold.

| arm | prompt | reads/token | prefill wall | token ids |
|---|---:|---:|---:|---|
| `--attn dense`, as shipped | 658 | 154.64 -> **27.83** (5.56x) | 251.2 -> 129.2 s | identical |
| `--attn dsa`, as shipped | 658 | 154.97 -> **27.83** (5.57x) | 256.1 -> 130.9 s | identical |
| `--attn dsa`, `index_topk=64` | 197 | 170.64 -> **78.31** (2.18x) | 78.2 -> 45.8 s | identical |
| `--attn streaming`, `--window 64` | 197 | 168.69 -> **77.87** (2.17x) | 80.1 -> 45.0 s | identical |
| `--attn dense`, MTP on | 658 | 159.56 -> **28.20** (5.66x) | 282.7 -> 131.7 s | identical |

**`bin/replay` is validated by this, from a completely different code path.** The offline sim
predicted **154.75** reads/token for a 769-token prefill at 2Q/6874 slots; the engine reports
**154.64** for a 658-token one. 0.07% apart. Every cache-policy conclusion in this file rests
on that simulator and nothing had previously checked it against the engine's own counter.

**The `index_topk=64` row is the one that matters for correctness, and it is why the shadow
artifact exists.** At the shipped 2048 a 658-token prompt never leaves `dsa_select_layer`'s
dense fast path, so the two `as shipped` rows never write the indexer's `sel` buffer at all —
they exercise the position-keyed `last_nr`/`last_dense` (where the old row-slot scheme would
already have handed a shared layer the wrong `nr`) but not the cross-pass selection reuse.
Lowering `index_topk` to 64 makes ~133 of 197 positions select, store into `sel`, and have a
later shared layer read it back. That row passing is the evidence the IndexShare re-keying is
right; the other two are not.

**The win grows with prompt length, measured rather than argued.** Layer-major reads the
compulsory count, which is ~78 x distinct-per-layer and therefore roughly FIXED in absolute
terms — 15 427 reads at 197 tokens, 18 310 at 658. Per token that is 78.31 and 27.83. Token-
major is ~155/token at both lengths, because it re-reads per token by construction. So the
ratio is 2.18x at 197 tokens and 5.56x at 658, and keeps climbing.

**Streaming needed `--window 64` for the same reason dsa needed the shadow.** `streaming_rows`
returns the whole causal prefix while `nt <= sinks + window`, and the default window is 8192
— so a 197-token run at the shipped default is `--attn dense` wearing a different flag. With
a 64-wide window the per-row selection actually runs, and layer-major needs nothing from it:
the set is a pure function of `(pos + r, sinks, window)`, rebuilt every pass, nothing
outliving the call. That is the property dsa's IndexShare did NOT have.

> One `--attn dsa` as-shipped arm at 197 tokens failed with an empty log — a signal from this
> session's own cleanup catching the arm before it wrote anything, not an engine fault. The
> same configuration passes at 658 tokens in the table above.

### The decode "regression" is a ONE-OFF ~2.7 s warm-up, and the fix for it FAILED (2026-08-02)

The table above reports layer-major decode at **610.5 ms/pass against token-major's 390.8**,
which reads like a 1.55x steady-state regression. It is not one. `-bench 16` is 13 passes and
a cold start front-loads, so that measurement cannot tell a one-off penalty from a rate
difference — a distinction the first write-up of this section did not draw, and should have.

Same pair at `-bench 128` (the model hits EOS at 25 tokens, so 22 passes):

| passes | token-major | layer-major | gap/pass | gap x passes |
|---:|---:|---:|---:|---:|
| 13 | 390.8 ms | 610.5 ms | 219.7 ms | **2.86 s** |
| 22 | 490.3 ms | 608.0 ms | 117.7 ms | **2.59 s** |

**The per-pass gap HALVED while the total held.** A steady-state penalty would have kept the
gap at ~220 ms/pass and totalled 4.8 s over 22 passes; instead the product is flat at ~2.7 s.
Layer-major's own ms/pass barely moved (610.5 -> 608.0) while token-major's ROSE (390.8 ->
490.3) as the context grew — the two are converging, which is what a warm-up looks like and
what a rate difference does not. Against the 152 s the prefill saves on the same run
(282.9 -> 130.6 s), 2.7 s is **1.8%**. Output identical at both lengths (16 and 25 ids).

**The mitigation this file previously proposed was implemented, measured and REVERTED.**
Sweep layer-major over tokens `0..n-2`, then run the last prompt token through all 78 layers
as one ordinary pass, so the pool ends in the shape token-major leaves it. Measured: decode
**618.5 ms/pass against 610.5 without it** — no change — at a cost of **+383 expert reads**
(28.20 -> 28.79/token). Correct but useless, so it is not in the tree.

Why it could not work, which is worth more than the change was: a token-major prefill does
not leave the pool holding THE LAST TOKEN's experts across all layers, it leaves roughly the
last **ten** tokens' — 6874 slots / (78 layers x 9 experts) = 9.8. Decode therefore finds ten
tokens of recent routing at every layer. One closing pass supplies one, so layers 0..50 still
hold a single token's 9 experts each and decode still misses them. Reproducing the
token-major end state needs a ~10-token close, not a 1-token one — and at a fixed 2.7 s the
thing it would buy is not worth the reads.
