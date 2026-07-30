# rivoli — CACHE_PILOT: router-piloted cross-layer expert prefetch

Status: **PARKED — but reason 1 has LIFTED (2026-07-29). Still do not build; reason 2
stands.**

**1. ~~Blocked on a faithful int4~~ — LIFTED.** This reason said the machinery existed to
make an L+2 **int4 promotion** affordable, while `.i4` was re-derived from `.vq3`
(`bin/vq3_to_i4`) and therefore strictly *less* faithful than the vq3 it came from —
int3-vq PPL 5.275 against int4 9.083. Promoting to int4 degraded quality, so building
machinery to promote *more, earlier* would have been an elaborate way to make the model
worse.

That artifact is gone. `bin/fp8_to_i4` derives `.i4` from the original fp8 and group-128
scales replaced the per-row ones: **int4 PPL 5.120, hybrid 5.189, int3-vq 5.275**
(`INT4.md` §10). int4 is now the most faithful of the three, so promotion no longer costs
quality and this objection no longer applies. (The precondition text named a `pack_i4`
container that was never written; the shipped path is `fp8_to_i4`. The slot is also
20.1 MB now, not 18.9.)

**2. It has no acceptance criteria of its own (pre-existing) — STILL BLOCKING.** It is preliminary work for
[CACHE_ROUTE.md](CACHE_ROUTE.md) (`--cache-policy top-m`) and is accepted or removed with
it. See "Acceptance is not local" below.

Note the offline screen cannot lift the remaining reason: it saturates by construction and
has no power over this work at all (see "Headroom" below). The one thing worth doing here
is still **LOOKA** (Step 1) — its recall numbers are a durable fact about
the model, cheap, and useful whatever happens to the loader.

Unaffected: single-format `top-m` (`int3-vq`, `int4`) needs none of this machinery.

Sister-project evidence: colibri's `PILOT`/`PILOT_REAL` (`c/colibri.c`), measured on
*this node* (rh-anine).

## What this is, and why it is not the prefetch we deleted

We deleted prefetch at `b372cd4` after a full-model A/B: ON 1.03 tok/s / 78.3% hit vs
OFF 1.03 tok/s / 76.9%. The conclusion recorded then — *its whole contribution was
`preloading` (reads that OVERLAP compute, not reads avoided), and on a bandwidth-bound
path overlap creates no bandwidth* — is correct and is independently confirmed by
colibri: their `COUPLE` experiment states flatly that readahead hints "cannot move
engine hit% by construction", and at scale measured them harmful (0.48 → 0.35 tok/s
from page-cache thrashing).

**CACHE_PILOT is a different mechanism.** It predicts a future layer's experts while
the current layer is still computing, and performs the *actual load into the pool*,
keyed for that layer. When decode arrives, the expert is already resident and the
demand read **never happens**. The lever is residency — fewer total bytes — not
earlier bytes.

The evidence that this is the real mechanism is a natural experiment, not a claim.
Colibri shipped the eviction guard with its comparison inverted (#474), which silently
degraded their real loads to hint-only. The field report (#490) isolated the damage:
**LRU hit share 27–38% → 15%, hit rate 84–86% → 74.5%, 0.95 → 0.82 tok/s.** Fixing the
polarity (#497) restored 0.82 → 0.89. What was lost was cache occupancy.

On **rh-anine**, colibri measured `PIPE=1 PILOT_REAL=1` at hit **67% → 83%** and
expert-disk time **22.9 s → 9.1 s (−60%)** (commit `248d78b`).

## Why the payoff differs for us

> **RETRACTED 2026-07-30 — this section had it backwards, and it inverts the plan's
> payoff.** It read: "Colibri is disk-bound; we are not, any more. Our fetch is **92%
> hidden** … So do **not** expect the win to show up as fetch wall."
>
> We *are* disk-bound. The "92% hidden" figure derives from `compute_gpu_ns`, a HipEvent
> **bracket** over the whole MoE phase that contains the fetch stalls it was used to rule
> out — so stall time was being counted as compute, and therefore as fetch hidden. Bucketing
> the bracket by per-layer miss count (38,400 layer-instances) separates them: a zero-miss
> layer costs 1563 µs → **117 ms/token of real compute** (corroborated independently by
> `moe_bench.rs` at 113 ms), while each miss adds **1239 µs**. Against 2.25 GB/token of
> fetch at ~12 GB/s ⇒ **~181 ms of transfer vs 117 ms of compute**. See ARCHITECTURE.md §3.
>
> **So expect the win exactly where this said not to look: fetch wall.** Fewer misses is
> now the *only* lever that raises the ceiling — perfect overlap alone floors at ~283 ms/tok
> (~3.5 tok/s), and past that the term to cut is bytes moved. That makes this plan more
> valuable than when it was written, not less, but for the opposite reason: raise hit rate
> to move fewer bytes, not to unblock host-gated launches.

Where it should show up instead:

1. **Compute bubbles.** A miss host-gates the per-expert launch; that is the
   already-measured mechanism by which lower hit rate inflates `moe-gpu` (the all-int4
   investigation concluded exactly this: "bigger int4 experts → fewer slots → lower
   hit% → more fetch + host-gated compute BUBBLES that inflate moe-gpu and hide the
   faster compute"). Fewer misses ⇒ fewer bubbles ⇒ lower `moe-gpu`.
2. **Bytes.** 0.96 GB/tok is still real DRAM/NVMe traffic on a shared LPDDR5 bus that
   the MoE dot competes for.
3. **Format.** For `top-m` the speculative load is what makes an int4 promotion
   affordable at all — 18.9 MB cannot land on the critical path.

## Architectural fit

We are better placed than colibri for the *prediction* half and worse placed for the
*admission* half.

**Cheap here:** routing is already on the host (`gpu.rs:55` `route_into`), and every
layer's router gate is resident f32 (`gate_w`, used at `gpu.rs:789` via
`launch_gemv_f32`). Predicting a future layer costs one extra `rmsnorm` + `gemv_f32`
(256×6144 — trivial) on state we already have.

**Expensive here:** we have one unified byte `Arena` (`arena.rs`) with a floating
hot/cold split, not colibri's per-layer LRU. A speculative admission evicts from the
*same* pool the current layer is using, and can trigger *compaction* (a synchronous
device memcpy that relocates a boundary slot). The correctness rule in `submit_spine` —
allocate all of a batch's miss slots before resolving final slots or issuing reads, so
a read never targets a slot that later moves — must extend to speculative slots or we
reintroduce the silent-corruption class the `inflight` guard was written for.

**Headroom is the known risk, not a gate.** Colibri measured a +3.1pp recall
improvement producing *literally zero* tok/s change on a cache-starved host, and our VQ
working set (16,457 experts against a ~5,000-slot pool) is closer to that regime than
theirs.

**The offline replay step does NOT quantify this, and an earlier draft claimed it did.**
The oracle ceiling saturates by construction: a decision needs `top_k` keys, `top_k`
admissions fit in any pool that holds one batch, so a perfect predictor removes every miss
and the number is a tautology — and at 100% recall the speculative admissions *are* the
baseline misses, so it is the same bytes moved earlier, which is this document's thesis
restated rather than tested. **This work's gate is recall, recall is unobservable offline,
and LOOKA (Step 1) is the only thing that measures it.** `bin/replay` does print a
*modelled* recall curve — at recall `r` the predictor still names `top_k` experts, so
every false negative is also a false positive, drawn from the ranks just outside the true
set in that decision's own window — and that curve prices the wasted bytes. It is an upper
bound (its errors are independent; real ones are correlated), so treat it as the shape of
the trade, and read the real `r` off LOOKA.

## Step 1 — LOOKA (instrumentation)

Zero-effect counters behind the `trace` feature (dev instrumentation, not a permanent
CLI flag):

- After layer L's attention, apply the **target layer's** post-attention norm and router
  gate to L's post-attention residual; take top-`k`. Instrument **both horizons** —
  L+1 and L+2 — because `top-m` needs L+2 and the recall difference between them is a
  number nobody has.
- Stash the prediction; when the target layer actually routes (`gpu.rs:797`), compare
  against the real `sel` and accumulate recall.
- Also accumulate the **baseline**: recall of "the same experts that layer chose for the
  previous token", the null hypothesis any predictor must beat.
- Report all of it beside the existing PROFILE summary.

Colibri's numbers on GLM-5.2 (48 greedy tokens) are the yardstick: **71.6%** at L+1 vs
**41.3%** for previous-token, with 79.4% as the skip-attention-only upper bound. Our
model is the same architecture but int3-vq quantized, and L+2 recall is unmeasured
anywhere — LOOKA exists to produce those numbers, and they feed the horizon and K
choices in the loader rather than a go/no-go.

**Cost note:** the prediction needs its own D2H of the pilot logits. Every existing
profile bucket wraps a join the forward pass already pays; this one does not. Fold the
pilot logits into the *same* `copy_out_into` as the next gate read if possible, and if
not, account the added sync honestly in the profile rather than letting it land in
`other`.

### MEASURED 2026-07-30 — `src/looka.rs`, `--features rocm,trace`

int3-vq / dense / lru, `--max-mem 115`, 128 tokens, chat-framed prompt:

| | recall | vs null | n |
|---|--:|--:|--:|
| **L+1** | **77.2%** | +46.4pp | 120,600 |
| **L+2** | **68.9%** | +38.2pp | 120,600 |
| prev-token (null hypothesis) | 30.8% | — | 120,000 |

**The gate passes.** We beat colibri's 71.6% at L+1 on a *quantized* model, and our null is
weaker than theirs (30.8% vs 41.3%), so the margin over free is 2.5× rather than 1.7×.

**L+2 = 68.9% answers the open horizon question** — only 8.3pp behind L+1, so `top-m`'s
two-layer lead time is affordable and does not need a separate cheaper predictor.

Implementation notes for anyone extending it:
- The pilot runs on layer L's **post-attention residual**, i.e. before L's MoE output is
  added back. That staleness is deliberate — it is the state a real prefetcher would have,
  since L+h's true input depends on MoE results not yet computed. Correcting for it would
  measure a predictor nobody can build.
- It is launched **outside** the MoE event bracket (gpu.rs: pilot ~1104, bracket opens
  ~1419), verified by the per-layer miss buckets being unchanged (0-miss 1576 µs here vs
  1563 µs on the default build).
- **`route` is not comparable in trace builds** — it reads 9.1 ms here vs 102.6 ms by
  default, because the pilot's D2H drains the stream earlier and the attention wait that
  `route_wait_ns` absorbed relocates into the pilot's sync bucket. (Which incidentally
  shows `route` was ~90% attention wait, not host routing.)

### What this does NOT settle — read before starting Step 2

Recall passing is necessary, not sufficient, and the disk-bound finding
(ARCHITECTURE.md §3) changed the economics after this plan was written. Prefetch's classic
payoff is hiding latency; ours is **bandwidth**-limited, so mispredicted bytes are a direct
throughput tax rather than merely wasted work.

At 77.2% recall, covering today's 140.79 useful misses/token takes ~182 speculative loads:
**2.16 → 2.80 GB/token, +29% on the exact resource that is the bottleneck.** A loader that
fetches all of top-`k` is therefore plausibly net-NEGATIVE even at this recall.

What Step 2 actually needs is **precision on a subset** — speculate only where confidence is
high. The next measurement is recall broken down by **prediction rank**: if the top 2–3
predictions carry ~95% precision, a confidence-gated prefetcher moves few enough wasted
bytes to win. That number does not exist yet, and LOOKA is the right place to add it.

## Step 2 — the speculative loader

One component, two callers: CACHE_PILOT asks it to load a predicted expert in the cheap
format, `top-m` asks it to load a window member as int4. Design constraints, each
traceable to a measured failure:

- **Demand reads must never queue behind speculative ones.** Colibri satisfies this
  with a second ring, having measured **two pools beat both** the layered and unified
  fusions (#78) — but that result is about *worker-thread pools* (PIPE pthreads vs the
  pilot thread), and our I/O is one io_uring plus one reaper thread, not a pool. **Do
  not port the second ring on their evidence.** Try a depth cap or priority split on
  the existing ring first, measure demand-read latency under speculative load, and add
  a second ring only if that fails. A ring we do not add is a ring we do not have to
  poison-handle, drain, and shut down.
- **Strictly-future layers only.** Speculation writes only layers `> current`; the
  demand path claims the current layer and waits out in-flight speculation for it.
  Slots stay hidden (no key published) until the read completes, so `submit_spine`
  can never resolve a slot that is mid-write.
- **Eviction guard, correct polarity.** Before a speculation takes the arena's victim:
  if the victim is warm and hotter than the predicted expert by a hysteresis margin,
  **drop the speculation** — do not displace the warm resident. Ask the policy for that
  judgement; do not add a parallel heat counter. `HybridLru` already tracks `freq` with
  decay (`hybrid.rs:87`, `:103`) and `top-m` replaces it with router rank
  ([CACHE_ROUTE.md](CACHE_ROUTE.md)), so the guard must read whatever the active policy
  exposes. The polarity that matters is *protect the victim*; colibri's inverted
  version ("speculation must beat the victim") drops ~100% of speculations once the
  cache fills, because a correct speculation is by definition colder than the resident
  it would replace. Unguarded, they measured **+9–18% bytes for +0.5–0.7pt hit** — a
  net loss.
- **Never trigger compaction speculatively.** A speculative admission that would force
  a cross-tier grow/compaction must be dropped instead. Compaction is a synchronous
  device memcpy on the decode path; paying it for a guess is indefensible.
- **Non-fatal.** A mispredicted or failed speculative read must never propagate an
  error into decode.

**Do not build:** N parallel speculative workers (colibri swept 1/4/8: barrier flat at
~7.5 s regardless, and more workers *hurt* — hit 86→82%, expert-disk 15.5→19.6 s;
diagnosis "latency/depth-bound, not throughput-bound"), or fusion into the existing
demand pool (above).

**Not planned: two-step (shared-expert-corrected) prediction.** Colibri measured
+3.1pp recall (71.6 → 76.7%) and *zero* end-to-end change on a cache-starved host, and
here it would cost a real expert compute per layer since our shared expert is folded
into the MoE launch. If `top-m` lands and recall is the binding constraint, plan it
then.

## Why L+2

`top-m` promotes to int4, and colibri diagnosed their prefetch barrier as latency-bound
because *"1 layer of decode compute ~6 ms < 1 expert load ~10 ms"* — which is also why
throwing workers at it did nothing. An 18.9 MB int4 promotion needs more slack than one
layer of compute buys, so the horizon is two layers. Prediction error grows with the
horizon; that trade is what LOOKA's dual-horizon counters measure.

## Measurement discipline

The house rule applies and is not negotiable here: **free-running decode tok/s cannot
rank this change.** Prediction quality shifts routing, which shifts hit rate, which
shifts the degeneration behaviour of a greedy run — the confound that already
invalidated one hot_pct sweep and two earlier conclusions.

- Residency and byte claims: `bin/replay` on a **fixed trace**, or a **fixed forced-token**
  bench (same tokens every run).
- Compute claims: the `moe-gpu` bucket at matched miss counts, or `examples/dot_bench`.
- Output quality: unchanged by construction *for CACHE_PILOT alone* — it is cache
  placement only, so decoded token IDs must be byte-identical with it on and off.
  Assert that; colibri verified sha256-identical full-model cold-stream output across
  guard-on/guard-off/off. (`top-m` does change output. That is its trade, not ours.)

## Build order (shared with `top-m`)

One program, not two:

1. Extend the `--trace` format (adds the top-M candidate window `top-m` needs).
2. Offline replay: the (J, M) miss-reduction grid in
   [CACHE_ROUTE.md](CACHE_ROUTE.md), plus the oracle ceiling and the modelled recall
   curve. One GPU capture, then free. This screens **`top-m`**; it cannot screen the
   pilot (see "Headroom" above) — step 3 does that.
3. LOOKA recall counters, both horizons (Step 1 above).
4. The speculative loader (Step 2 above).
5. `top-m` routing + rank-driven tiering, then the pilot prediction driving promotion.

## Acceptance is not local

There is no CACHE_PILOT-only bar to clear. The combined feature is judged by
[CACHE_ROUTE.md](CACHE_ROUTE.md)'s acceptance criteria, and **if `top-m` fails them,
this comes out with it** — the prediction, the speculative loader, and any ring or
guard added for it. Do not leave a prefetcher behind as an orphan feature "since it is
already written": that is exactly how the last prefetch survived long enough to need
deleting at `b372cd4`.

The one thing worth recording either way is the LOOKA measurement. Recall numbers for
GLM-5.2-int3-vq at L+1 and L+2 are a durable fact about the model that belongs in
`benchmarks.md` whatever happens to the code.
