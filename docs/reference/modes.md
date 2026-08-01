---
status: live
verdict: The --mode × --cache-policy matrix and which knob does what. Quality ladder: int4 5.120 > hybrid 5.189 > int3-vq 5.275.
---

# Run modes

Rivoli's routed-expert pool has **two orthogonal knobs**:

1. **Format mode** (`--mode int3-vq | int4 | hybrid`, default `hybrid`) — the
   quantization the routed experts decode from.
2. **Cache policy** (`--cache-policy lru | 2q | arc`, default `2q`) — how the pool
   decides which experts stay resident.

They are independent choices, but the *hybrid* format mode constrains what a cache
policy has to do (it must place each expert in the right format slab). The matrix at
the bottom is the whole story; the sections explain the tradeoffs behind it.

Everything here is about the **routed** experts (256/layer, streamed on demand). The
always-resident set (attention, dense MLPs, shared expert, codebooks) is unaffected.

> **The format mode decides whether speculative decode runs. Since 2026-07-31 every mode
> can.** The MTP head is checkpoint layer 78 and rides the routed pool like any other
> layer, so it needs its expert slab in **every format the run opens**. `bin/convert` emits
> `L78.vq3`; `bin/fp8_to_i4` now emits `L78.i4` (its range bound used to stop one layer
> short). So:
>
> | `--mode` | opens | MTP head | speculative decode |
> |---|---|---|---|
> | `int3-vq` | `.vq3` | present | **on** (default; `--no-mtp` opts out) |
> | `int4` | `.i4` | present *(needs `L78.i4`)* | **on** |
> | `hybrid` *(default)* | both | present *(needs both)* | **on** |
>
> **An artifact converted before 2026-07-31 has no `L78.i4`** — `int4`/`hybrid` on one log
> "speculative decode OFF: this artifact carries no MTP head" and decode sequentially,
> which is a missing slab rather than a missing feature. Re-run `bin/fp8_to_i4` to emit it.
>
> This table said "absent / off" for `int4` and `hybrid` until 2026-07-31, and called the
> feature a 0.93–0.95× loss "so nothing is being lost in the meantime". Gated on draft
> confidence it measures **1.108×** — see `docs/reference/architecture.md` §13.

---

## Format modes

An "expert" is one MoE feed-forward block. The three modes differ in how its weights
are quantized, which sets three things: **compute speed**, **on-device size** (→ how
many fit in the pool → hit rate), and **stream cost** (bytes per miss).

### `int3-vq` — vector-quantized 3-bit
- **Size:** ~15.3 MB/expert — the smallest, so the **most slots fit** → highest hit
  rate at a given budget, and the **cheapest miss** (smallest read).
- **Compute:** the dot is a random codebook gather (`idx → cb[idx]`). Even with the
  fp16 codebook L1-resident, it's **latency/throughput-bound on the gather** — the
  slow path. Microbench: ~353–383 GElem/s.
- **Needs:** `L{l}.vq3` + `codebooks.f32` in the artifact.
- **Best when:** the pool is residency/bandwidth-bound (small budget, big working set).

### `int4` — 4-bit, one f32 scale per 128 weights
> **History (2026-07-27):** this mode used ONE scale per 6144-weight row and was
> unusable — PPL 73.43 against int3-vq's 5.28. The defect was the format, not a bug:
> a single outlier set the step for a whole row and rounded the bulk to zero (603 rows
> past 50% zeros on one projection). Group-128 scales fixed it outright, **PPL 73.43 →
> 5.120**, and int4 is now the best-quality mode in the engine. See `docs/investigations/int4-scales.md`.
- **Size:** ~20.1 MB/expert (~32% bigger) — **fewer slots fit** → lower hit rate →
  **more misses** → more fetch + more host-gated compute bubbles.
- **Compute:** sequential nibble decode, no gather. **~1.8× faster than int3-vq**
  (microbench: ~669–677 GElem/s, confound-free — see caveat below). No codebook.
- **Needs:** `L{l}.i4` in the artifact.
- **Best when:** the working set is *fully resident* (huge budget) so the residency
  penalty vanishes and only the faster compute remains. As a whole-run mode it
  usually **loses** — the residency hit outweighs the compute win.

### `hybrid` — int4 hot, int3-vq cold *(default)*
> **Measured 2026-07-28: PPL 5.189** — the best mode in the engine OVERALL, being the only
> one that beats int3-vq on both axes at once (5.189 vs 5.275 *and* 2.72 vs 2.62 tok/s).
> Not the best perplexity: that is int4 at **5.120**, which this line used to claim for
> hybrid while listing a better number two clauses later. int4 is best quality and slowest;
> hybrid is best overall. (Corrected 2026-07-31.) The
> earlier 11.55 was measured against the per-row-scaled `.i4` set that `docs/investigations/int4-scales.md` §10
> replaced with group-128 scales; it does not describe the current artifact.
Two physical slabs. The cache policy routes each expert to one:
- **HOT slab = int4** — the *frequently reused* experts. They get int4's fast compute
  **and** stay resident (no fetch, no bubbles) — the one place int4's speed actually
  pays.
- **COLD slab = int3-vq** — the *probation / rarely reused* experts. Small (more fit)
  and cheap to stream.
- **Needs:** *both* `L{l}.vq3` and `L{l}.i4`.
- **Idea:** spend int4's size premium only where its compute win lands (the hot set),
  and keep residency high everywhere else with cheap vq3 slots. Best of both.

#### Hybrid's numerics depend on cache state — read this before comparing two hybrid runs

In `int3-vq` and `int4` every expert decodes from the *same* format, so residency decides
which **reads** happen and never which **arithmetic** runs. Output is a function of the
weights and the token prefix alone.

**Hybrid breaks that.** Residency decides which *slab* an expert lands in, and the slab
decides its *format* — int4 or int3-vq — so the numerics themselves become a function of
cache state. Two consequences, and the distinction between them matters:

- **Reproducible at a fixed configuration.** Same prompt, same `--max-mem`, same
  `--cache-policy`, same budget: the pool starts empty and residency evolves
  identically, so the output is identical run to run. Hybrid is not nondeterministic.
- **NOT comparable across budgets or policies.** Change `--max-mem` or the policy and
  you change which experts sit in which format — which changes the
  arithmetic. **Two hybrid runs at different budgets are not the same computation**, so a
  difference in output or quality between them cannot be attributed to the thing you were
  varying. The format mix moved underneath the comparison.

So an A/B across budgets is sound in the single-format modes and is confounded by
construction in hybrid. If you need a cross-budget quality comparison, run it in
`int3-vq`, where residency cannot touch the numerics.

**There is no split flag.** `--hot-pct` existed briefly, was replaced by a cold-slab
floor, and was deleted outright along with the fixed-partition cache variants and the
replay `--hybrid` split simulator (`c876a8d`) — the split self-sizes now. The hot/cold
boundary is emergent: the two-ended byte arena packs COLD from the low end and HOT from
the high end, and the boundary **floats** with the policy's tier decisions
(`src/memory/arena.rs`, `src/memory/pin.rs`).

The reason a flat percent was fragile still stands and is why nothing replaced it: too
high and the cold/probation slab collapses (measured cliff around n_cold < ~600 slots).
**A consequence worth knowing before you plan an experiment:** there is now no knob to
sweep the split with, and `--max-mem` is *not* a substitute — changing the budget changes
which experts sit in which format, i.e. changes the arithmetic being compared. Sweeping
the split again means re-introducing a floor override first.

---

## Cache policies

The routed pool is smaller than the working set, so it evicts. The policy owns *which
expert to evict*. Scan resistance matters: a MoE layer touches many one-shot experts;
a pure-recency policy lets those flush the genuinely-hot ones.

| Policy | Signal | Scan-resistant | Notes |
|---|---|---|---|
| `lru` | recency only | no | Simplest. One list, evict the least-recently-used. |
| `2q` *(default)* | recency + a 2nd-access promotion | yes | A1in probation (one-shots age out) + Am frequent + A1out ghost. |
| `arc` | self-tuning recency/frequency balance | yes | T1/T2 + B1/B2 ghosts + adaptive target `p`. |

**Those three are the whole list.** `--cache-policy` parses `lru | 2q | arc` and nothing
else; every one of them is output-neutral by construction (**INV-1**, `architecture.md`
§8b).

### `top-m` was a fourth policy. RETIRED 2026-07-30, removed from the engine.

> **This section described a selectable policy until 2026-08-01.** It does not any more —
> `--cache-policy top-m`, `--route-j`, `--route-m`, `RouteAdvice`, the `route_into`
> substitution and the `swap%` counter are all deleted from the engine, and `bin/replay`'s
> (J, M) substitution grid went with them on 2026-08-01. The prose below is kept in the
> past tense because *what it ruled out is the durable part*; nothing here is a knob.

`top-m` was cache-conditional MoE routing
([arXiv:2412.00099](https://arxiv.org/abs/2412.00099)). **Every other policy answers "what
do I evict". This one answered "given that a miss costs a 15 MB read, is the 5th-ranked
expert that is already resident better than the 4th-ranked one that is not?"** The top-`J`
ranked experts were always selected; the remaining `top_k − J` slots preferred experts that
were already resident *and* ranked inside the top-`M` window; anything left fell back to
plain rank order. Expert weights were untouched — the cache reordered *selection* only and
never rewrote a gate value. Knobs were `--route-j` (**4**) and `--route-m` (**9**) — the
measured cell, NOT the paper's J=2/M=12, which was rejected on this workload.

**It was the only policy whose choice was not output-neutral**, and that is why it is gone.
`lru`, `2q` and `arc` change only which bytes are read and when; the decoded tokens are
identical. `top-m` changed which experts ran, so *every* cache change became a potential
output change and each one needed a perplexity run to price.

**The measured trade** (int3-vq, 5,184 teacher-forced positions, `--max-mem 100`, at
`--route-j 4 --route-m 9`):

| | |
|---|---|
| hit rate | 72.25% → **77.69%** (+5.44pp) at 5.79% swap |
| perplexity | 4.1306 → 4.1525, **+0.529%** |
| 95% CI | **[−0.21%, +1.27%]** |
| verdict | **UNCERTIFIED** against the ~1% bar |

Read both halves of that honestly. The interval **contained zero**, so `top-m` was *not
significantly worse* than the baseline — and also *not certified within budget*, because
the upper bound of +1.27% overshot the pre-registered ~1% bar. The point estimate was half
the bar; the uncertainty is what failed, not the measurement. Under the paper's own
reference band (+0.1–3.0%) it would have passed comfortably. **Relaxing the bar to that
band was considered and DECLINED** — the ~1% figure was fixed in the plan before any data
existed, and moving a threshold after seeing a result that misses it is post-hoc reasoning.
Recorded so it is not re-proposed as an oversight. Certifying it as the default would have
needed ~**12,840 teacher-forced tokens** (~2.5× the corpus used), about **3.4 h** of
sole-tenant device time for a baseline plus one cell, and at that point estimate it might
*still* have missed.

Later measurement settled it against the bar outright — **+3.63% PPL on int3-vq and +12.7%
on int4** — but the deciding argument was the structural one above, not the number. Full
record in
[`docs/investigations/cache-conditional-routing.md`](../investigations/cache-conditional-routing.md);
the (J, M) grid and the certification arithmetic are tabulated in
[`docs/measurement/benchmarks.md`](../measurement/benchmarks.md) §"`top-m` offline screen"
and §"DECISION". Recover the offline screen from tag `archive/replay-oracle-prefetch`.

**`lru`, `2q` and `arc` are byte-identical to before `top-m` existed, and to after.** The
substitution sat behind an early return, and with it deleted routing is a pure function of
(logits, bias, `top_k`) — which is now the tested invariant, not a claim.

### What `hybrid` demands of a policy
The HOT (int4) slab should hold the *frequent* experts. But **an expert's format only
changes on a (re)fetch — i.e. on a miss** (a hit returns the existing slot; no
migration). So the policy's only lever is **which slab a miss lands in**, and to fill
HOT correctly it needs a *frequency memory that survives eviction*:

- **`2q` / `arc` fit naturally.** Both promote to the frequent tier only when a
  **ghost** key is re-accessed — which is a miss. Promotion *coincides with the fetch*,
  so the fetch goes straight into the int4 slab. Zero wasted work. In hybrid the tier
  caps are **dynamic and byte-aware** (there is no `::fixed` variant; the tier mapping
  below is real, the fixed-partition constructors are not): each segment maps to a slab, and
  an insert only evicts from its own segment, so a reuse stays in one slab. This trades
  the policy's adaptivity for two right-sized slabs.
- **`lru` has no ghost and no frequency signal**, so on its own it can't tell a hot
  miss from a cold one. Hybrid-LRU adds a tiny **frequency-counter admission**: an
  all-time access count per key (~19k keys × u16 ≈ 38 KB); on a miss, `count ≥
  threshold → HOT`, else COLD. LRU still does the *eviction* (per slab); frequency does
  the *placement*. Promotion still rides the miss path — no extra fetch.

**Shared caveat (all policies):** a frequent expert that stays resident in COLD and
keeps *hitting* never misses, so it never migrates to int4 — served fast (no fetch) but
at vq3 compute. None of the policies proactively upgrade a sticky-cold expert; that
would need speculative refetch. Acceptable, but real.

---

## Interaction matrix

Rows = format mode, columns = cache policy. The grid is complete — these three policies are
all there are. (It once carried a fourth column for `top-m`, which was single-format only;
`config.rs::validate` rejected `top-m` + `hybrid` outright. Both the policy and that
validator are gone — `Config::validate` was an `Ok(())` stub after the retirement and was
deleted 2026-08-01; `validate_backend`, which refuses `--mode int4|hybrid` under Vulkan,
is a different function and is still live.)

| | `lru` | `2q` *(default)* | `arc` |
|---|---|---|---|
| **`int3-vq`** | 1 vq3 slab, LRU evict | 1 vq3 slab, dynamic 2Q | 1 vq3 slab, dynamic ARC |
| **`int4`** | 1 int4 slab, LRU evict | 1 int4 slab, dynamic 2Q | 1 int4 slab, dynamic ARC |
| **`hybrid`** *(default)* | 2 slabs; **freq-counter** places (HOT/COLD), per-slab LRU evict | 2 slabs; **`HybridTwoQ`** (A1in→COLD/vq3, Am→HOT/int4) | 2 slabs; **`HybridArc`** (T1→COLD/vq3, T2→HOT/int4) |

- **Single-format rows** (`int3-vq`, `int4`): the policy runs unmodified
  (`hybrid::make`, dynamic), one slab.
- **Hybrid row:** the same dynamic policy runs with two strides and reports a
  `Tier` (Cold/Hot) per insert; the pool maps that to a slab. The 2Q split still sizes the
  A1in probation queue and the A1out ghost, at `TwoQSplit::default()` — **kin 8% / kout
  20%**, in `src/memory/cache.rs`, in every mode.

  > **CORRECTED 2026-08-01.** This said "`--2q-kin`/`--2q-kout` are live in every mode".
  > The *split* is; the two engine flags are not. They were deleted that day, having never
  > appeared in `docs/`, `tests/`, the README or any script, and `TwoQSplit::default()` was
  > the only value ever passed. The measured optimum is a broad plateau (kin 6–10% × kout
  > 15–25%, all within 0.1pp), which is why nothing was ever lost by not exposing it —
  > `src/memory/cache.rs` carries that measurement beside the constant. To sweep the split
  > offline, `bin/replay` keeps its own `--kin`/`--kout` and prints the full kin×kout grid;
  > that is now the only way to move it.

---

## How to actually compare these — measurement caveat

**Free-running decode `tok/s` cannot rank format modes or splits.** Greedy decode on a
lossy model is fragile: a run that degenerates into repetition routes to the *same* few
experts every token → high hit rate → artificially *fast*. Measured example: hot-88%
degenerated ("…atmosphere of Earth (" ×30) → 89% hit → 3.24 tok/s, while a *coherent*
hot-85% run → 80% hit → 2.61 tok/s. **The faster number was the broken output.** So a
tok/s sweep partly measures *which config degenerated*, not efficiency.

Use instead:
- **The residency sim** — `replay <trace> <n_slots> [--sweep]` replays *one fixed trace*
  through the same byte-aware policies the engine runs and reports the resident-hit %,
  isolating residency from the decode trajectory. Confound-free; milliseconds. Ranks
  policies and 2Q Kin/Kout, not compute.
- **A fixed forced-token bench** (same tokens every run → same trace) for a trustworthy
  wall-clock number per config.
- **The dot microbench** (`examples/dot_bench.rs`) for pure per-format compute (this is
  where int4's 1.8× is established, with no routing in the loop).
- **Output quality** (perplexity on fixed text / longer prompts) — *orthogonal* to the
  split; never inferred from decode tok/s. Use `--ppl <text> --ppl-out <path>`, which
  scores **teacher-forced** (every position is fed the known next token, never the model's
  own argmax) and writes one NLL per position; `bin/ppl` then compares runs **paired** at
  each position. Pairing is what gives it the resolution — a ~1% perplexity bar is ~0.01
  nats of mean ΔNLL, which two independently measured perplexities cannot separate from
  sampling noise at a few hundred tokens. Report the standard error, not just the mean:
  an underpowered null is not evidence of no harm.

**And in `hybrid`, a cross-budget or cross-policy quality comparison is confounded by
construction** — see "Hybrid's numerics depend on cache state" above. Vary the budget in
`int3-vq`, where residency cannot reach the arithmetic.
