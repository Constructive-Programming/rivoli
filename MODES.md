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
- **Size:** ~20.1 MB/expert (~32% bigger) — **fewer slots fit** → lower hit rate →
  **more misses** → more fetch + more host-gated compute bubbles.
- **Compute:** sequential nibble decode, no gather. **~1.8× faster than int3-vq**
  (microbench: ~669–677 GElem/s, confound-free — see caveat below). No codebook.
- **Needs:** `L{l}.i4` in the artifact.
- **Best when:** the working set is *fully resident* (huge budget) so the residency
  penalty vanishes and only the faster compute remains. As a whole-run mode it
  usually **loses** — the residency hit outweighs the compute win.

### `hybrid` — int4 hot, int3-vq cold *(default)*
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
  `--cache-policy`, same `--hot-pct`: the pool starts empty and residency evolves
  identically, so the output is identical run to run. Hybrid is not nondeterministic.
- **NOT comparable across budgets or policies.** Change `--max-mem`, the policy, or
  `--hot-pct` and you change which experts sit in which format — which changes the
  arithmetic. **Two hybrid runs at different budgets are not the same computation**, so a
  difference in output or quality between them cannot be attributed to the thing you were
  varying. The format mix moved underneath the comparison.

So an A/B across budgets is sound in the single-format modes and is confounded by
construction in hybrid. If you need a cross-budget quality comparison, run it in
`int3-vq`, where residency cannot touch the numerics.

**`--hot-pct <n>`** (hybrid only; errors in other modes) sets the byte split — n% of
the pool to the int4 HOT slab. **Default:** a *cold-slab floor* (probation must hold
≳2 tokens' worth of experts so it can't starve) with the rest to HOT — i.e. "push hot
as high as possible without starving cold." A flat percent is fragile: too high and
the cold/probation slab collapses (measured cliff around n_cold < ~600 slots).

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
| `top-m` **(opt-in, single-format modes only)** | router rank + residency | n/a | **Changes which experts RUN, not just which are cached.** Not output-neutral. See below. |

### `top-m` — cache-conditional routing (opt-in, not the default)

`--cache-policy top-m`, cache-conditional MoE routing
([arXiv:2412.00099](https://arxiv.org/abs/2412.00099)). **Every other policy answers "what
do I evict". This one answers "given that a miss costs a 15 MB read, is the 5th-ranked
expert that is already resident better than the 4th-ranked one that is not?"** The top-`J`
ranked experts are always selected; the remaining `top_k − J` slots prefer experts that are
already resident *and* ranked inside the top-`M` window; anything left falls back to plain
rank order. Expert weights are untouched — the cache reorders *selection* only and never
rewrites a gate value. Knobs `--route-j` (**4**) and `--route-m` (**9**) — the measured
cell, NOT the paper's J=2/M=12, which was rejected on this workload (see below).

**This is the first policy whose choice is not output-neutral.** `lru`, `2q` and `arc`
change only which bytes are read and when; the decoded tokens are identical. `top-m`
changes which experts run, so it changes the output. That is why it is opt-in.

**The measured trade** (int3-vq, 5,184 teacher-forced positions, `--max-mem 100`, at
`--route-j 4 --route-m 9`):

| | |
|---|---|
| hit rate | 72.25% → **77.69%** (+5.44pp) at 5.79% swap |
| perplexity | 4.1306 → 4.1525, **+0.529%** |
| 95% CI | **[−0.21%, +1.27%]** |
| verdict | **UNCERTIFIED** against the ~1% bar |

Read both halves of that honestly. The interval **contains zero**, so `top-m` is *not
significantly worse* than the baseline — and it is also *not certified within budget*,
because the upper bound of +1.27% overshoots the pre-registered ~1% bar. The point
estimate is half the bar; the uncertainty is what fails, not the measurement. Under the
paper's own reference band (+0.1–3.0%) it would pass comfortably.

**Making it the default requires certification first.** That needs roughly **12,840
teacher-forced tokens** (~2.5× the corpus used), about **3.4 h of sole-tenant device time**
for a baseline plus one cell — and at the current point estimate it may *still* miss. Until
someone buys that, `top-m` stays opt-in.

**Relaxing the bar to the paper's ≤3% band was considered and DECLINED.** It would have
passed immediately, and that is precisely the reason not to do it: the ~1% figure was fixed
in the plan before any data existed, and moving a threshold after seeing a result that
misses it is post-hoc reasoning. Recorded here so it is not re-proposed as an oversight.

**Not available in `--mode hybrid`** — the rank-driven tier rule it would need is parked on
an artifact precondition (see `docs/CACHE_ROUTE.md`). The engine rejects the combination
rather than silently falling back. `top-m` is also incompatible with `--trace`, because
substitution breaks the invariant the v2 trace format promises.

**`lru`, `2q` and `arc` are byte-identical to before `top-m` existed.** The substitution
sits behind an early return, and the residency predicate is provably never consulted when
routing advice is absent. Choosing any of the three leaves the engine exactly as it was.

### What `hybrid` demands of a policy
The HOT (int4) slab should hold the *frequent* experts. But **an expert's format only
changes on a (re)fetch — i.e. on a miss** (a hit returns the existing slot; no
migration). So the policy's only lever is **which slab a miss lands in**, and to fill
HOT correctly it needs a *frequency memory that survives eviction*:

- **`2q` / `arc` fit naturally.** Both promote to the frequent tier only when a
  **ghost** key is re-accessed — which is a miss. Promotion *coincides with the fetch*,
  so the fetch goes straight into the int4 slab. Zero wasted work. In hybrid the tier
  caps are **fixed** (`TwoQ::fixed` / `Arc::fixed`): each segment maps to a slab, and
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

Rows = format mode, columns = cache policy. `--hot-pct` applies only to the hybrid row.

| | `lru` | `2q` *(default)* | `arc` |
|---|---|---|---|
| **`int3-vq`** | 1 vq3 slab, LRU evict | 1 vq3 slab, dynamic 2Q | 1 vq3 slab, dynamic ARC |
| **`int4`** | 1 int4 slab, LRU evict | 1 int4 slab, dynamic 2Q | 1 int4 slab, dynamic ARC |
| **`hybrid`** *(default)* | 2 slabs; **freq-counter** places (HOT/COLD), per-slab LRU evict | 2 slabs; **`TwoQ::fixed`** (A1in→COLD/vq3, Am→HOT/int4) | 2 slabs; **`Arc::fixed`** (T1→COLD/vq3, T2→HOT/int4) |

- **Single-format rows** (`int3-vq`, `int4`): the policy runs unmodified (`cache::make`,
  dynamic), one slab. `--hot-pct` is rejected.
- **Hybrid row:** the policy runs in its **fixed-partition** variant and reports a
  `Tier` (Cold/Hot) per insert; the pool maps that to a slab. `--2q-kout` still sizes
  the ghost; `--2q-kin` is inert in hybrid (the split comes from `--hot-pct`).

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
