# rivoli — CACHE_PILOT: router-piloted cross-layer expert prefetch

Status: **DELETED FROM THE ENGINE 2026-07-31. This document is now a record, not a guide to
live code.** Step 1 (LOOKA) and the `--hint-k` eviction-veto layer were built, measured
inert, and removed; nothing described below still exists in `src/`. The measurements stand
and are the reason it went — read this before proposing cross-layer prefetch again.

> **THE ANSWER, 2026-08-01 — and it is not the one this document spent 400 lines expecting.**
> Cross-layer prefetch is closed, but **not** because prediction is hard and **not** because
> the drive is saturated. Both of those were the stated reasons and both are false:
>
> - The pre-attention router predicts a layer's own misses at **82.7% recall**
>   (`RIVOLI_PRED_PROBE=1`, measured — better than LOOKA's 77.2% at L+1).
> - The drive is **idle 35% of every token** (`ARCHITECTURE.md` §3), so there is spare
>   bandwidth. `b372cd4` prefetched into the *busy* window, which is why it saw nothing.
>
> **The blocker is that the idle window is 1.13 ms and one expert read is ~2 ms.** It fits
> 0.74 of a single read where a layer needs 2.9, and a layer's fetch ends when its LAST read
> lands. Meanwhile the 23% of a top-8 prefetch that goes unused costs **+67 ms/token** of
> extra drive time against a **≤85 ms/token** ceiling. Net ceiling ~3%, unreachable.
>
> Full numbers in "Feasibility, settled" below. What would change the answer is a **smaller
> expert** or **more compute per layer** — i.e. PERF.md #2, not prefetch.

**Why it was removed rather than left default-off.** It was ~1,100 lines across
`src/looka.rs`, `src/hybrid.rs` (the `Hint`/`HintSet` types, the cap, the decay, four trait
methods per policy), `src/cache.rs` (a second, advisory skip set in the eviction scan, with
a fallback that had to be right at eleven call sites), `src/gpu.rs` (per-layer rmsnorm +
gemv + a blocking D2H), plus two registered invariants (INV-2, INV-3) with their tests. That
is a lot of load-bearing machinery — the eviction scan carried two authorities with
different failure modes — kept alive for a mechanism that moved hit rate by at most +0.1pp.
Its own measurement says why it cannot be rescued by tuning: at ~965 slots a key survives
~138 layers, so a 1-2 layer veto binds only on keys already at the LRU end, and reaching the
eviction horizon would need a precision LOOKA does not have there (77.2% at L+1, 68.9% at
L+2, falling).

The offline simulator in `bin/replay` is NOT affected: its `Pilot` is a modelled predictor
over a trace, self-contained, and still the right tool for asking the recall question.

The history below is kept because it records *why* this was parked twice and what changed.

> **CORRECTED 2026-07-31.** This line used to read "SHIPPED — `--pilot-k`, always on,
> default 3", and all three claims were wrong against the code:
> - **The flag is `--hint-k`, not `--pilot-k`.** There is no `--pilot-k`.
> - **Its default is 0, i.e. OFF** (`gpu::DEFAULT_HINT_K`), not 3.
> - It is off *because it was measured inert*: the vetoes bind (23/220/137 at 40/30/25 GiB,
>   none ever dropped for cap) but touch **0.9% of evictions**, moving hit rate by at most
>   +0.1pp while costing 3-8% throughput for the pilot's per-layer rmsnorm+gemv+D2H. At
>   ~965 slots a key survives ~138 layers, so a 1-2 layer veto can only bind on a key
>   already at the LRU end — and a next-layer prediction is warm, so it sits at the far end.
>
> The mechanism is not broken and the plumbing is kept: `--hint-k 3` re-enables it, and
> output is bit-identical at every value. What is retired is the claim that it is on.

Both parking reasons are resolved. Reason 1 (no faithful int4) lifted 2026-07-29. Reason 2
("no acceptance criteria of its own — accepted or removed with `top-m`") was overtaken by
measurement: LOOKA produced a standalone gate (L+1 recall 77.2%, `p@0` 99%) and the loader
produced a standalone result (+8–14% tok/s at K=1, output bit-identical), so this no longer
depends on `top-m` landing. What remains coupled to `top-m` is the *int4 promotion* variant,
not the prefetcher.

The paragraph below arguing prefetch "cannot move hit% by construction" is about the
DELETED readahead-hint prefetch, not this one — the distinction it draws turned out to be
the right one, and the measured hit-rate rise (78.0 → 79.8%) is the confirmation.

**1. ~~Blocked on a faithful int4~~ — LIFTED.** This reason said the machinery existed to
make an L+2 **int4 promotion** affordable, while `.i4` was re-derived from `.vq3`
(`bin/vq3_to_i4`) and therefore strictly *less* faithful than the vq3 it came from —
int3-vq PPL 5.275 against int4 9.083.

> **`bin/vq3_to_i4` NO LONGER EXISTS** (deleted; `docs/ARCHITECTURE.md` §11 flags every
> doc that still cites it). `.i4` is built by `bin/fp8_to_i4` straight from the fp8 source,
> and since the group-128 fix (`docs/INT4.md`, RESOLVED 2026-07-27) int4 is the
> best-QUALITY mode in the engine: **PPL 5.120 against int3-vq's 5.275**, re-measured
> 5.154898 vs 5.222720 on 2026-07-31. The premise this bullet was lifted *around* — that
> promoting to int4 degrades quality — is now false in the opposite direction. Anything
> below reasoning from "int4 is less faithful" is reasoning from a dead artifact. Promoting to int4 degraded quality, so building
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

> **CORRECTED 2026-08-01 — "fewer total bytes" is wrong, and the `b372cd4` verdict is
> narrower than it reads.**
>
> A prefetched expert's bytes still cross the PCIe/NVMe path exactly once. Prefetch does not
> move fewer bytes; it moves the same bytes *earlier*. (It can raise hit rate for LATER
> tokens by improving pool occupancy — `b372cd4` measured that at +1.4pp — but that is a
> second-order effect, not the lever.) So the distinction this section draws between
> CACHE_PILOT and the deleted prefetch does not exist: both are "earlier bytes".
>
> Which does **not** make prefetch worthless, because the premise underneath the deletion
> was never checked. `b372cd4` concluded "on a bandwidth-bound NVMe path overlapping a read
> creates no bandwidth". That holds only where the drive is BUSY, and its own code says
> where it issued:
>
> > `// Submit L+1's predicted-expert reads NOW (non-blocking) while this layer's demand`
> > `// reads are still in flight, so both batches sit in the device queue together.`
>
> It prefetched **during the MoE phase — the one window where the drive is saturated.**
> Measured 2026-08-01: `route_wait` is 84.8 ms/token of host blocked on the gate D2H, i.e.
> attention GPU time, ~1.13 ms per layer, and the io_uring ring is EMPTY throughout. The
> drive idles **35% of every token** and prefetch was aimed at the other 65%.
>
> The unexploited window is worth at most **85 ms/token, ~1.28×**, capped by the attention
> duration rather than the read duration (starting a 2.0 ms read 1.13 ms early gains
> 1.13 ms, not 2.0). Against that: the pilot's own per-layer rmsnorm+gemv+D2H cost 3–8%
> (above), mispredicted reads evict against a working set ~3× the pool, and the staging ring
> is 16 slots for a ~3-slot layer.
>
> **`b372cd4`'s result stands for the window it tested.** It is not evidence about issuing
> at the top of layer L+1, where `x_{L+1}` is already exact and only the attention residual
> is missing — strictly better information than that implementation had (it predicted from
> L's residual *before* L's MoE contribution). See "Feasibility" below.

## Feasibility, settled — MEASURED 2026-08-01 (`RIVOLI_PRED_PROBE=1`)

The predictor is **not** the problem. Run at the top of each MoE layer on `post_ln(x)` — the
layer input, before attention adds into it — against `--mode int3-vq --attn dense --max-mem
100 --no-mtp -bench 64`:

| | |
|---|---:|
| recall on the top-k | **83.9%** (37236/44400) |
| **recall on the MISSES** (the only ones a prefetch could save) | **82.7%** (12563/15191) |
| reads it would issue | 16306 |
| of those, wasted (no row routed there) | **23.0%** |

That beats LOOKA's 77.2% at L+1, which is what one would expect: this predictor is missing
only the attention residual, where LOOKA was missing that *and* the previous layer's MoE
output. **Prediction accuracy is not what killed cross-layer prefetch.**

### The economics, and they do not work

Per pass (74 passes, 15.335 MB/read, measured 1.32 ms/miss):

| | reads/pass | drive time |
|---|---:|---:|
| demand misses today | 205.3 | 271 ms |
| prefetch would issue | 220.4 | 291 ms |
| — useful (a demand miss, started early) | 169.8 | — |
| — **wasted** | **50.7** | **+67 ms/token** |
| demand misses it would still not predict | 35.5 | 47 ms |

Against a gain ceiling of **85 ms/token** (the whole idle window) the 23% waste costs
**67 ms/token**, and the predictor's own per-layer rmsnorm+gemv+D2H costs ~6 ms. Net ceiling
**~+12 ms on a 397 ms token, ~3%** — while assuming the idle window is exploited perfectly.

### Why it cannot be exploited perfectly, which is the real blocker

**The idle window is 1.13 ms per layer. One 15.3 MB expert read takes ~2 ms at QD1.** The
window fits 0.74 of a single read and a layer needs 2.9 of them. Worse, a layer's fetch ends
when its *last* read lands, so starting a subset early moves the batch by much less than the
window. The prefetch necessarily spills into the MoE phase — which is the saturated regime
`b372cd4` measured, and where its wasted 23% is a straight tax on the bottleneck.

### So: closed, and now for the right reason

Three explanations have been offered for why cross-layer prefetch does not pay here. Two are
wrong:

- ~~"On a bandwidth-bound path overlap creates no bandwidth"~~ (`b372cd4`) — the drive idles
  35% of every token. There IS spare bandwidth.
- ~~"The predictor cannot see far enough"~~ — 82.7% recall on misses, measured above.
- **The idle window is shorter than one expert read, and a top-8 prefetch's waste costs more
  drive time than the window can return.** This one holds.

It also says what would change the answer, which none of the plumbing work does: a **smaller
expert** (the window fits a larger fraction of one) or a **longer window** (more compute per
layer to hide behind). Both are the same lever as PERF.md #2, and neither is prefetch.

The probe stays in `gpu.rs` behind `RIVOLI_PRED_PROBE=1`, off by default. It is ~60 lines
and it is the evidence; re-run it before re-opening this.

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
high. Measured below.

### Precision by prediction rank — MEASURED 2026-07-30

Same run. `p@r` is the marginal precision of the pilot's r-th ranked guess (gate score
descending); cumulative is the precision of speculating on the whole prefix.

| rank r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **L+1** `p@r` | **99%** | 96% | 93% | 87% | 78% | 67% | 55% | 42% |
| **L+2** `p@r` | **97%** | 92% | 85% | 76% | 66% | 55% | 45% | 36% |

Cumulative L+1: top1 98.7 · top2 97.4 · top3 95.8 · top4 93.6 · top5 90.5 · top6 86.6 ·
top7 82.1 · top8 77.2%. Cumulative L+2: top1 96.7 → top8 68.9%. (Cumulative top-8 equals
aggregate recall exactly on both horizons — the rank and recall counters are independent,
so that agreement is a self-check, not a tautology.)

> **REVISED 2026-07-30, after `--pilot` shipped and was measured.** The derivation below
> assumes the device is bandwidth-SATURATED, so a wasted speculative fetch displaces a
> useful one one-for-one. **Measurement falsified that assumption.** Counting the
> speculative reads (which the profile initially omitted), `--pilot` moved **2.02 → 2.29
> GB/token, +13%, while wall fell 397.8 → 351.1 ms/token, −12%** — 5.08 → 6.52 GB/s
> sustained. More bytes, less time: there was device headroom, and the binding constraint
> is nearer **queue depth than bandwidth**.
>
> The engine is still fetch-bound (~181 ms transfer vs 117 ms compute) and the "95% hidden"
> retraction is untouched — that rests on the miss-bucket linearity and the `moe_wall`
> arithmetic, not on saturation. What falls is only the *tax* argument. The 50% figure is
> therefore a CONSERVATIVE floor, not the true break-even: a wrong speculation that fills
> otherwise-idle device time costs less than a full 1.24 ms, so ranks below 50% may still
> pay. **The top-7 gate is worth measuring rather than ruled out.**
>
> Why the error: `2.25 GB / 386 ms = 5.83 GB/s` matched the documented "5.8–6.7 GB/s at
> QD≥4" too neatly, and that coincidence was allowed to outweigh the marginal (12.4 GB/s)
> and in-fetch-window (10.9 GB/s) figures measured in the same session, which both said the
> device was faster than the aggregate implied.

#### The break-even, and why it is 50%

On a bandwidth-saturated device one expert is 15.34 MB ≈ **1.24 ms** — which is also the
measured stall a miss imposes (ARCHITECTURE.md §3). So per speculation:

- **right** → the fetch leaves the critical path, saving ~1.24 ms of stall; bytes unchanged,
  since it was going to be fetched anyway.
- **wrong** → ~1.24 ms of bandwidth spent displacing a real fetch on a device with no spare.

Net = `1.24 × (2p − 1)` ms, **positive iff p > 50%**. Judge the *marginal* `p@r`, not the
cumulative: adding rank r is priced at `p@r` alone.

| horizon | speculate on | cumulative precision | first rank rejected |
|---|---|--:|---|
| **L+1** | ranks 0–6 (**top-7**) | 82.1% | r7 @ 42% |
| **L+2** | ranks 0–5 (**top-6**) | 78.5% | r6 @ 45% |

**Modelled payoff**, L+1 top-7: `Σ(2·p@r − 1)` over the gated ranks = 4.50, so
`4.50 × 1.24 ms × f_nonres` per layer. At the measured miss fraction (`f_nonres` ≈ 0.235)
that is ~1.3 ms/layer ⇒ **~98 ms/token**, which would land 386 → ~288 ms/tok, essentially
on the ~283 ms overlap floor. Treat this as a model, not a result: it assumes wasted bytes
displace useful ones one-for-one, that a correct prefetch hides the stall completely, and it
does **not** price wrong admissions evicting warm residents — a second-order cost that
pushes the real break-even above 50%. The eviction guard in Step 2 exists for exactly that,
and its hysteresis margin should be tuned against this curve.

`p@0` = 99% at L+1 is the standout: the single top prediction is right essentially always
and clears any plausible break-even. **If Step 2 wants a minimum viable increment, it is
"prefetch rank 0 of L+1 only"** — one expert per layer, ~99% precision, no confidence
tuning, and it exercises every piece of loader plumbing (hidden slots, eviction guard,
demand-priority) at the lowest possible waste.

## Step 2a — SHIPPED: the speculative loader (`--pilot-k`, always on, default 3)

Built 2026-07-30 off the precision curve above. Landed first as rank-0-only to prove the
plumbing, then widened to a configurable **top-K** and turned **always on**, default
**K=3** — `p@0..2` are 99/96/93%, the ranks that are right nearly every time, while p@3
onward fall away (87, 78, 67, 55, 42%). `--pilot-k 0` disables it.

In-flight speculations are capped at K: anything still airborne from the previous layer
counts against the budget, so a slow fetch throttles new speculation rather than letting the
queue grow. Predictions are consumed in RANK order, so a narrower budget keeps the best
guesses.

**Where the speculation is allocated is the whole correctness story.** `Pin::submit_spine`
takes a `spec_req` and allocates the predicted slot in **phase 1b, inside the same batch as
that layer's demand misses**. That is not a convenience: phase 1b is where evictions and
compaction relocate slots, and phase 1c resolves final addresses only after it. Allocating
a speculative slot anywhere else — later, or out of band — lets a read target a slot that
subsequently moves, which is exactly the silent-corruption class the `inflight` guard was
written for.

**The read-before-write trap, and the substitution that avoids it.** `alloc` publishes the
key immediately, so the next layer's phase 1a finds it via `routed.get` and counts it a
hit — and a hit is handed `Signal::ready()`, meaning nothing downstream waits. With bytes
still in the air that is precisely the failure the READ-BEFORE-WRITE detector exists to
report. `submit_spine` therefore tracks the one in-flight speculation and **substitutes the
real Signal** for that expert's slot in phase 2, so the expert stream awaits it exactly as
it would a demand miss. The detector is also routed around for that single key, since it is
the one case where an unlanded hit is legitimate rather than a bug.

Other invariants, each traceable to a constraint above:
- **Pinned while in flight.** `protect` is called on the in-flight speculative key before
  phase 1b, the only window guaranteed to precede any `alloc`, so this batch's evictions
  cannot reclaim a slot that already has a read queued against it.
- **One in flight, maximum.** If the previous speculation has neither landed nor been
  consumed, this layer skips rather than tracking two. A layer (~3.5 ms) comfortably
  exceeds a fetch (~1.4 ms), so the skip is rare.
- **Already-resident predictions issue nothing** (~76% of them), which is what keeps the
  feature cheap.
- **Speculative read goes last in the batch** — the cheap approximation of "demand reads
  never queue behind speculative ones". io_uring submits the whole SQ at once so this is
  not a hard priority; a real priority split on the ring is the upgrade if it shows up.
- **Eviction guard is implicit, not the one this document asks for.** Every key the layer
  needs is protected by phase 1a and the in-flight spec is pinned, so a speculation can only
  take a victim nobody in this batch wants — but there is no hotness *hysteresis*, so it can
  still displace a warm resident a LATER layer wants. Watch expert-hit%: if it falls, that
  guard is the fix. This is the known ceiling of the MVP.

### MEASURED — int3-vq/dense/lru, `--max-mem 115`, 64 tokens

| | baseline | `--pilot` |
|---|--:|--:|
| tok/s | 2.51 | **2.71 – 2.85** (2 runs) |
| ms/tok | 397.8 | 351.1 / 369.6 |
| expert hit | 78.0% | **79.8%** |
| demand misses/tok | 131.91 | **121.42** |
| bytes/tok (incl. speculative) | 2.02 GB | **2.29 GB** |
| live precision | — | **98.7%** (1749/1772) |

**~+8–14% throughput.** The range is two pilot runs against ONE baseline; the spread is
run-to-run timing noise, not a difference in behaviour — hits, misses and speculation counts
are bit-identical between the two pilot runs, so the cache path is fully deterministic and
only wall time moves. A tighter number needs more baseline samples.

Live precision 98.7% sits on LOOKA's predicted `p@0` of 99%, which is the check that the
loader prefetches what the predictor named.

**Verified**: output token IDs are bit-identical to the baseline (`--dump-ids`, diff clean),
and zero READ-BEFORE-WRITE reports under `--features trace`.

Two defects the first run caught, both fixed:
1. **7 READ-BEFORE-WRITE reports**, all for one key. A speculation consumed *while still in
   flight* reached neither path that sets the loaded flag — not a demand miss (nothing
   pushed it to `pending_loaded`), and it had not landed before use (so the batch-start
   `mark_loaded` never ran). The key stayed marked unloaded and every later token hitting it
   re-reported. The bytes had landed; only the flag was missing. Fixed by pushing the
   consumed key to `pending_loaded` exactly as a demand miss does.
2. **The profile undercounted its own fetches.** `fetch_n` tracked demand misses only, so
   `gb_per_tok`/`ms_per_miss` omitted every speculative read — understating the first run by
   0.43 GB/token and making prefetch appear to REDUCE bytes moved. It increases them ~13%.
   Now summed over demand + speculative; the hit-rate denominator is left on demand alone so
   it cannot flatter itself.

`--pilot` is a runtime flag on default builds; the pilot's rmsnorm+gemv runs one horizon
(L+1) outside `trace` and both under it, so LOOKA keeps reporting the L+2 curve.
`spec_issued`/`spec_used` are reported as live precision and should track `p@0`; a gap means
the loader is prefetching something other than what the predictor named.

## Step 2 — the speculative loader (full design, for the top-7 gate)

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
