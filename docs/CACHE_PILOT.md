# rivoli — CACHE_PILOT: router-piloted cross-layer expert prefetch

Status: **PARKED — blocked on an artifact precondition. Do not build.**

There are now **two independent reasons** not to build this, and they are not the same
reason wearing different hats.

**1. It is blocked on a faithful int4 (new, and decisive on its own).** This document's
entire purpose is to make an L+2 **int4 promotion** affordable — the prediction and the
speculative loader exist to move an 18.9 MB int4 load off the critical path. But in the
artifact we run, `.i4` is re-derived from `.vq3` (`bin/vq3_to_i4`), so int4 is strictly
*less* faithful than the vq3 it came from: int3-vq PPL 5.275 against int4 9.083 on a fixed
teacher-forced corpus (`../benchmarks.md`, "int4 provenance"). **Promotion to int4
therefore degrades quality here, and this machinery exists to do more promotion, earlier.**
Building it now would be an elaborate, expensive mechanism for making the model worse —
the most costly possible way to be wrong. The precondition is an int4 more faithful than
vq3; a group-scaled (gs64) `pack_i4` container plausibly supplies one, at which point this
reason lifts and the design below stands as written.

**2. It has no acceptance criteria of its own (pre-existing).** It is preliminary work for
[CACHE_ROUTE.md](CACHE_ROUTE.md) (`--cache-policy top-m`) and is accepted or removed with
it. See "Acceptance is not local" below.

Note the offline screen cannot lift either reason: it saturates by construction and has no
power over this work at all (see "Headroom" below). The one thing worth doing here before
the precondition is met is **LOOKA** (Step 1) — its recall numbers are a durable fact about
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

Colibri is disk-bound; we are not, any more. Our fetch is **92% hidden** behind the
async expert stream and the bottleneck is compute (route 134 ms + moe-gpu 226 ms of a
388 ms wall). So do **not** expect the win to show up as fetch wall.

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
