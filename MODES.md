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

### `int4` — colibri 4-bit, per-row scale
- **Size:** ~18.9 MB/expert (~24% bigger) — **fewer slots fit** → lower hit rate →
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
  split; never inferred from decode tok/s.
