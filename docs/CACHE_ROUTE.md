# rivoli — `top-m`: cache-conditional MoE routing (arXiv:2412.00099)

Status: **REMOVED FROM THE ENGINE 2026-07-30.** `--cache-policy top-m`, `--route-j`,
`--route-m`, `RouteAdvice`, the `route_into` substitution and `swap%` are all deleted. See
the retirement record at the END of this file for why, and for the LOOKA hint layer that
replaced it. **Everything between here and there describes a mechanism the engine no longer
has** — kept because it records the design, the measurements that rejected it, and the
reasoning that a future proposal would otherwise repeat.

It was a fourth `--cache-policy` alongside `lru | 2q | arc`, and the only cache mechanism
that ever **changed which experts run**. That property is now forbidden and tested against
(**INV-1**, ARCHITECTURE.md §8b).

Source: Skliar, van Rozendaal, Lepert, Boinovski, van Baalen, Nagel, Whatmough,
Ehteshami Bejnordi — *Mixture of Cache-Conditional Experts for Efficient Mobile Device
Inference*, TMLR (June 2025), [arXiv:2412.00099](https://arxiv.org/abs/2412.00099).
Training-free. Reported: **>50% cache-miss reduction**, perplexity **+0.1–3%**,
downstream task accuracy loss **<0.1%**, ~2× speedup on mobile.

## The idea

Every other policy here answers "what do I evict". This one answers "given that a miss
costs a 15–19 MB read, is the 5th-ranked expert that is already resident better than the
4th-ranked one that is not?" On a bandwidth-bound engine the answer is usually yes, and
the paper's contribution is doing it without wrecking quality:

1. Rank experts by the router as usual.
2. **Sacred top-J** — the J highest-ranked experts are always selected, cached or not.
   (Paper uses J=1 for Mixtral/Phi, **J=2 for Qwen/DeepSeek-class routers**; GLM-5.2 is
   in that class.)
3. **Top-M window** — only experts ranked within the top M are eligible to be promoted
   for being resident. This is what stops a cache-resident but genuinely irrelevant
   expert from being run.
4. Fill the remaining `top_k − J` slots preferring resident experts inside the window,
   falling back to rank order.
5. **Expert weights come from the original router scores.** The cache only reorders
   *selection*; it never rewrites the gate values of the experts that do run.

Optional per-token variant: instead of a fixed M, accumulate sorted router probabilities
until a cumulative mass `p` is reached and use that as the window.

## Why it drops in cleanly here

Our routing is already on the host and already separates ranking from weighting —
`route_into` (`gpu.rs:55`) computes `scores[e] = sigmoid(logit)`, ranks by
`choice[e] = scores[e] + bias[e]`, and `topk_into` selects; the MoE weights are then
built from `scores[e]` alone (`gpu.rs:~814`). That is exactly the paper's split, so the
change is confined to the selection step and the weight path is untouched.

Residency is queryable without mutation: `HybridPolicy::contains(&self, k)` takes
`&self` and does *not* refresh recency (`get` is the mutating one), so the router can
ask "is `expert_key(layer, e)` resident" for the whole candidate window with no side
effects on the eviction clock.

Two rivoli-specific bonuses beyond the paper's model: every avoided miss also avoids an
**admission**, therefore an **eviction**, and in `--mode hybrid` sometimes a
**compaction** (a synchronous device memcpy relocating a boundary slot).

~~And since our fetch is already 92% hidden, the win lands as fewer host-gated compute
bubbles in `moe-gpu` rather than as fetch wall.~~ **Retracted 2026-07-30:** "92% hidden"
came from a bracket that contained the very stalls it was used to rule out. The engine is
fetch-bound (~181 ms of transfer vs 117 ms of compute), so the win lands **as fetch wall**,
which is the larger prize. See ARCHITECTURE.md §3 and CACHE_PILOT.md.

## Design

**A fourth policy — but not a fourth policy implementation.** `--cache-policy top-m`
adds an arm to `hybrid::make` (`hybrid.rs:57`) that constructs the existing
`HybridLru` (the paper evaluates on LRU) configured with a rank-driven tier rule instead
of the frequency threshold, and with the routing advice switched on.

Do **not** fork a `HybridTopM` struct. This repo deleted three duplicate policy
families in `08db745`; re-adding one by copy-paste is the same mistake with a new name.
`HybridLru` gains a tier-rule field; everything else is shared.

**Staging note, because this sentence overstates what the first step does.** The
tier-rule replacement described here is the *hybrid* rank-driven tiering, and it is a
later step. In the shipped single-format work `HybridLru` gains an **advice** field only;
`LRU_HOT_THRESHOLD` and the `freq` map are untouched and still drive admission. That is
correct for `int3-vq` and `int4`, where both tiers hold the same format and the tier
choice is cosmetic — and the frequency threshold *cannot* be replaced until the hybrid
work lands. Read "gains a tier-rule field" as describing the end state, not step 2.

What `top-m` adds is one trait method the other policies leave defaulted:

```rust
// hybrid.rs — default None, so lru/2q/arc are untouched.
trait HybridPolicy {
    // ...existing...
    fn route_advice(&self) -> Option<(usize, usize)> { None }  // (j, m)
}
```

`route_into` gains the advice plus a residency predicate. When the advice is `None` the
function is bit-identical to today's behaviour — that is the regression guarantee.

Knobs `--route-j` and `--route-m`. **They default to J=4/M=9, the cell measured and
shipped on — NOT the paper's J=2/M=12.** The paper's values were the defaults until they
were measured here: +3.63% perplexity on int3-vq with the cost established as real, and an
outright FAIL on int4 at +12.7%. Shipping opt-in with a rejected default would have handed
the worst measured configuration to anyone who enabled the policy without passing knobs.
The paper's values stay reachable explicitly; they are just not the accident. The paper's cumulative-mass variant (a per-token window from a
probability threshold `p`) is **not** planned — it is a second mechanism for the knob
fixed M already provides. Add it if M turns out to be workload-dependent.

**Weights.** No change. The chosen set feeds the existing `scores[e]` →
`norm_topk_prob` sum-normalize → `routed_scale` path. Note that normalizing over a
*substituted* set does shift the weights relative to the unsubstituted set — that is
not a deviation from the paper, it is the model's own top-k normalization applied to the
set that actually runs.

## Mode integration (`--mode int3-vq | int4 | hybrid`)

**Single-format modes** (`int3-vq`, `int4`): uniform slot stride, residency is a
boolean, and the behaviour is the paper exactly.

> ## UNPARKED (2026-07-29) — the artifact precondition is now MET
>
> **This block parked the hybrid tier rule on "an int4 more faithful than vq3", which the
> then-current vq3-derived `.i4` could not be by construction. That artifact is gone.**
>
> `bin/fp8_to_i4` now derives `.i4` from the original fp8 directly, and group-128 scales
> replaced the per-row ones: **int4 PPL 5.120, hybrid 5.189, int3-vq 5.275**
> (`docs/INT4.md` §10). int4 is now the *most* faithful of the three, not the least.
>
> So the argument below — that promoting the sacred top-J into int4 would be "precisely
> inverted", putting the most important experts in the least faithful format — **no longer
> holds against the current artifact.** It was correct when written, against a different
> `.i4`.
>
> The rule is therefore unblocked and unbuilt. Its acceptance test is unchanged and is
> still owed: the hybrid A/B of the rank tier rule against the frequency threshold, on
> hit% and the `moe-gpu` bucket. Note also that the precondition text named a
> `pack_i4` container that was never written; the shipped path is `fp8_to_i4`.
>
> One caveat before building it: `config.rs::validate` currently rejects `top-m` +
> `--mode hybrid` outright, precisely because this rule does not exist. Implementing the
> rule means lifting that guard, and the guard is what stops a silent fallback to the
> frequency threshold being credited to `top-m`.
>
> Unaffected: **single-format `top-m` (`int3-vq`, `int4`) ships independently of this.**
> There is no promotion, no tiering and no format change anywhere in its path.

**Hybrid mode: the router rank *is* the tier rule.** Today `HybridLru` promotes to the
Hot (int4) tier on an access-frequency threshold (`hybrid.rs:150`). Under `top-m` that
heuristic is redundant, because the router already tells us which experts matter — so
replace it rather than layer on top of it:

- **Sacred top-J → int4.** These run every time this layer is visited and are the
  highest-value experts in the layer. int4 is ~1.8× faster to compute (isolated: gate/up
  int4 669 vs vq3 353 GElem/s).
  **It is NOT more accurate in the artifact we run, and an earlier draft of this line
  claimed it was.** `bin/vq3_to_i4` re-derives `.i4` from our own `.vq3`, so the chain is
  fp8 → vq3 (lossy 3-bit) → int4 and the int4 set cannot be better than the vq3 it came
  from, by construction. Measured on a 762-token teacher-forced corpus at the time: int4
  PPL 9.083 vs
  int3-vq 5.275. So promoting to int4 buys speed and *costs* quality here. See
  "int4 provenance" in `../benchmarks.md` for the fix path (a group-scaled colibri
  container), after which this bullet's original premise would hold again.
- **Top-M window → marked for int4 load, horizon L+2.** Membership in the window is a
  prediction that this expert is about to be worth running. Issue its int4 load
  speculatively, two layers ahead.
- **Demand misses → int3-vq.** A miss on an expert we need *now* fetches the small
  format (15.3 MB vs 18.9 MB), because the only thing that matters on the critical path
  is ending the stall.
- **Everything else** stays whatever it was admitted as and ages out under LRU.

The steady state is emergent, not configured: an expert that keeps reappearing in the
top-M window keeps getting promoted and stays int4; a one-off expert is fetched as
int3-vq and ages out. That is the "natural mix" — **frequency-of-use expressed through
router rank, with no frequency counter to tune.** `LRU_HOT_THRESHOLD` and the `freq`
map (`hybrid.rs:87`) can be deleted for this policy.

**Why L+2 and not L+1.** The horizon is the fix for the one failure colibri diagnosed
but did not solve: their pilot looks one layer ahead, and *"1 layer of decode compute
~6 ms < 1 expert load ~10 ms"* — which is why their prefetch barrier is latency-bound
and why throwing workers at it did nothing (flat at 1/4/8 workers). An 18.9 MB int4
promotion needs more slack than one layer of compute provides. Two layers gives it.

**[CACHE_PILOT.md](CACHE_PILOT.md) is the machinery that makes this possible**, and it
exists for this policy: promotion needs a cross-layer prediction (which experts will be
in the window at L+2) and a speculative loader (to fetch 18.9 MB off the critical
path). It is preliminary work here, not a separate feature — it carries no acceptance
criteria of its own and is judged, and removed, with `top-m`. See Acceptance.

**Testable prediction, worth stating before measuring:** cache-conditional routing
should help *more* in `int4` and `hybrid` than in `int3-vq`, because int4 slots are
18.9 MB vs 15.3 MB, so the same budget holds fewer of them and the miss rate it is
attacking is higher. If the measurement contradicts this, the implementation is
suspect, not the prediction.

**MEASURED: it is not supported.** On the offline grid (512-token traces, `--max-mem
100`, LRU), relative miss removal at the widest window is **88.0% int3-vq / 87.9% int4 /
88.5% hybrid** — the effect is essentially mode-independent. In *absolute* pp the ranking
is the opposite of the prediction at the paper's own defaults (J=2, M=12): int3-vq
+15.24, hybrid +15.13, int4 +15.05. Absolute gain simply tracks headroom — hybrid starts
from the highest baseline (74.35% vs 72.70% vs 71.15%) so it has the least to win.

Do not claim the prediction confirmed, and do not treat this as the implementation being
suspect either: the mechanism is intact.

**Why it washed — and a trap to avoid when reading these numbers.** It is tempting to
observe that int4 holds 4,744 slots against int3-vq's 5,870 (19% fewer) for only 1.55pp
less hit, and conclude the working set is so concentrated that slot count barely matters.
**That conclusion is wrong**, and it is wrong because comparing each mode at *its own*
capacity conflates two different things: how many slots the mode gets, and what its trace
is. Every mode decodes its own trajectory, so the three traces are not the same workload.

Separating them (LRU, same trace, matched capacity):

| | cap 4,744 | cap 5,852 |
|---|---:|---:|
| int3-vq trace | 66.42% | 72.60% |
| int4 trace | 71.15% | 76.07% |
| hybrid trace | 68.28% | 74.35% |

Slot count matters **a lot** — for the int3-vq trace, +23% slots is +6.18pp — and int4's
trace is simply ~4.7pp more cacheable than int3-vq's at equal capacity. The decomposition
is exact: −6.18pp from the smaller pool, +4.73pp from the easier trace, net −1.45pp
against the −1.55pp observed. There is no concentration effect doing the work.

So: **read the cross-mode grid at matched capacity, not at each mode's own slot count.**
With one 512-token trajectory per mode, the trace-to-trace differences here are of the
same order as plausible run-to-run variance, and none of the cross-mode claims should be
leaned on. The *within*-trace (J, M) results are unaffected by any of this — each grid is
scored against its own baseline — which is why the screen itself stands.

**The steep saturation curve is good news for `top-m`, not bad, and it is the strongest
honest argument for the feature.** Hit rate is still climbing steeply with pool size right
where we operate, so squeezing more out of the slots we have is worth a great deal — and
**we cannot buy those slots.** The box has ~120 GiB total and the capture already ran at
`--max-mem 100`; there is no meaningful headroom left to grow the pool into. `top-m` is
valuable in direct proportion to how hard capacity is to add, and on this node it is
nearly impossible to add at all.

Stated in the currency the engine thinks in, and **never state the first half without the
second**:

> `top-m` at the paper's defaults (J=2, M=12) buys what growing the pool from 5,852 to
> ~10,950 slots would — an **~1.9× effective pool — at 17.8% swap, with the quality cost
> unmeasured.** The cheapest passing cell (J=4, M=10) is worth ~1.4× at 9.6% swap.

Pool growth is free; substitution is not. The moment "1.9× effective pool" is quoted
without the swap figure beside it, it reads as costless, and it is not — 17.8% swap means
nearly one chosen expert in five is not what the router asked for.

## Staging

Shared with [CACHE_PILOT.md](CACHE_PILOT.md) — see its build order; these are one
program, not two. The pilot's LOOKA counters and speculative loader land between steps
1 and 2 below, because promotion cannot be implemented without them.

**1. Trace format first — this is a real prerequisite.** Today's `--trace` records the
selections that actually happened. Evaluating substitution offline needs the *candidate
window*: per routing decision, the top-M expert ids (and their `choice` scores). Extend
the trace writer, then extend `bin/replay` to replay a captured trace under substitution
and report the hit-rate delta for a grid of (J, M). This answers "how much miss
reduction is available on our workload" with **zero engine risk** and (after one capture)
no further GPU time. It is the cheap place to stop **`top-m`** before anything is written
into the engine.

**It does not screen CACHE_PILOT, and an earlier draft of this section wrongly said it
did.** The offline oracle — perfect knowledge of the next decision's experts — saturates:
a decision needs `top_k` keys, `top_k` admissions fit in any pool that holds one batch, so
a perfect predictor removes every miss by construction and the number is a tautology. It
is worse than uninformative, because at 100% recall the speculative admission set *is* the
baseline miss set — the same bytes and the same evictions, merely earlier. The pilot's
entire risk is **recall**, and recall is unobservable offline; LOOKA (build order step 3)
is its real gate. What replay can offer is a *modelled* recall curve (`bin/replay` prints
one), which prices the false positives a degraded predictor emits — but that is analysis,
not a bar, and it is an upper bound because its errors are independent where real ones are
correlated.

**2. Engine implementation.** The `hybrid::make` arm + the tier rule + the
`route_advice` hook + `route_into` substitution + `swap%`. Assert the `None` path is
unchanged.

**3. Quality measurement.** Unlike CACHE_PILOT, **this changes output.** Perplexity on a fixed
text corpus, swept over (J, M), in each mode. The paper's reference point is +0.1–3%
perplexity for >50% miss reduction; if we are paying materially more than that, either
M is too wide or J is too small.

**4. Mode matrix.** Measure all three modes. In `hybrid`, the rank-driven tier rule
needs its own A/B against the frequency threshold it replaces: same trace, same budget,
compare hit%, the resulting int4/int3-vq resident mix, and `moe-gpu`.

## Counters

One number, not three. **`swap%`** — the fraction of chosen slots that were not in the
true top-K — reported in the PROFILE summary whenever the policy is `top-m`, alongside
the existing hit%, `GB/tok`, and `moe-gpu`.

**Denominator: total chosen slots over the profiled decode window**, which equals
`hits + misses` because `submit_layer` looks each selected expert up exactly once. That is
deliberately the same denominator `hit%` uses, so the two numbers printed beside each
other are directly comparable, and it is what `bin/replay` reports too. Per-decision and
per-token means would both differ once batches vary; pick this one and stay with it.

## `top-m` is incompatible with `--trace`, and the engine rejects the combination

Not an implementation detail — a real interaction between two features this document
treats as independent. The v2 trace format promises `window[..top_k] == sel`;
`submit_spine` debug-asserts it and `bin/replay` hard-fails a trace that violates it.
Substitution is *precisely* what breaks that prefix, because `sel` is then no longer the
rank-order head of the window. A `--trace --cache-policy top-m` run would therefore either
trip the assert or write a corrupt capture that the next (J, M) screen would read as
ground truth. The config layer refuses the pair outright.

If a trace of a *substituted* run is ever wanted, it needs its own format that records the
pre-substitution ranking and the post-substitution selection separately. Do not quietly
relax the prefix invariant to allow it — that invariant is what makes a captured trace
trustworthy.

Colibri reports `swap%`, `route_agree` and `route_kl`; the first two are the same
measurement (a chosen slot outside the true top-K *is* a swap, so
`route_agree = (K − swaps)/K = 1 − swap%`). `route_kl` is the rigorous version and is
worth adding if a writeup needs it, but `swap%` is what you tune (J, M) against.

## Risks

- **Self-reinforcing lock-in.** Preferring residents makes them recently-used, which
  keeps them resident, which makes them preferred. The pool can collapse onto a small
  expert set and stop exploring. Sacred top-J is the structural guard (the strongest J
  experts always run regardless); `swap%` is the instrument — watch it climb over a long
  run, that is the signature. Note the rank-driven tier rule *reduces* this risk versus
  the frequency threshold it replaces: rank is recomputed from the router every token,
  where a frequency counter only ratchets up.
- **Quality regression is silent.** A cache-aware run that degenerates into repetition
  routes to the same few experts, so it shows *higher* hit rate and *faster* tok/s. This
  is precisely the confound documented in MODES.md that already invalidated a hot_pct
  sweep. **Never rank (J, M) by free-running decode tok/s.** Perplexity on fixed text
  for quality; the replay sim or a fixed forced-token bench for residency.
- **Interaction with the router bias.** We rank by `sigmoid + bias`, where the bias is
  the model's load-balancing term. Substitution is applied on top of a ranking that
  already encodes balancing pressure; if `swap%` correlates with particular experts,
  the bias may be fighting the cache preference.
- **Two effects, one policy.** Routing substitution and rank-driven int4 promotion are
  separable and should be measured separately (substitution alone, then substitution +
  promotion), or a win from one will be credited to the other. Colibri runs an explicit
  2×2 for the routing/prefetch pair; ours is the same discipline with the axes renamed.

## Acceptance

**This section covers `top-m` and [CACHE_PILOT.md](CACHE_PILOT.md) together.** The
pilot is preliminary work for this policy and has no separate bar; there is one
decision for the whole program.

- Replay (fixed trace, per mode): **≥ +5pp absolute hit-rate improvement** at some (J, M)
  on the grid, or the feature is not worth the routing complexity — say so and stop.
  (This bar replaces an earlier "miss reduction ≥20pp", which was a spec bug: our baseline
  miss rate is ~19–24%, so a 20 *percentage-point* reduction was at or past the arithmetic
  maximum. Absolute hit pp is also directly comparable to the hit% column in
  `benchmarks.md`.) Report **both** numbers for every grid cell — absolute pp on hit, and
  the relative % of misses removed — so the result stays comparable to the paper's
  ">50% cache-miss reduction" even though the relative figure is not the bar.
  This grid is a **screen for whether to build**, not final acceptance; the perplexity,
  hybrid tier-rule A/B and `moe-gpu` gates below still decide.
- Engine: `--cache-policy lru|2q|arc` byte-identical to today.
- Quality: perplexity delta within ~1% of the unsubstituted baseline at the chosen
  (J, M), measured on fixed text, in every mode it ships for.
- Hybrid: the rank-driven tier rule beats the frequency threshold it replaces on hit%
  and `moe-gpu` at a matched budget, and the resident int4/int3-vq mix is stable rather
  than drifting to one format.
- MODES.md gains the policy and the honest quality trade; this is the first policy whose
  choice is not output-neutral, and that has to be documented where users pick it.

**If these are not met, CACHE_PILOT comes out too** — the prediction, the speculative
loader, the eviction guard, and any ring added for it. There is no fallback position
where the prefetcher stays because it is already written. That is precisely how the
last prefetch survived until `b372cd4`. What survives a negative result is the
measurements: the (J, M) grid, the perplexity sweep, and the LOOKA recall numbers, all
in `benchmarks.md`.

---

## RETIRED 2026-07-30 — `top-m` removed from the engine

`--cache-policy top-m`, `--route-j`/`--route-m`, `RouteAdvice`, `route_into`'s substitution
and the `swap%` counter are deleted (~180 refs / 16 files). What replaced it: the LOOKA
hint layer (docs/CACHE_PILOT.md), which steers **eviction** instead of **selection**.

**Why it went.** Measured cost was +3.63% perplexity on int3-vq and an outright fail on int4
at +12.7%, against an acceptance bar of ~1%. But the deciding argument is structural, not
the number: because top-m made routing cache-conditional, *every* cache change became a
potential output change, and each one needed a perplexity run to price. With it gone,
routing is a pure function of (logits, bias, top_k) — now tested as **INV-1** — so a hint, a
policy swap or a budget change is output-bit-identical BY CONSTRUCTION. The acceptance test
for the whole hint layer is "the token IDs never move", which is checkable in one diff.

`bin/replay`'s (J, M) grid is kept as the historical screen that produced the numbers above.
It now models a mechanism the engine does not have; it is not a knob anything can enable.

### And the hint layer that replaced it is INERT at the default operating point

Measured the same day (int3-vq/lru, `--max-mem 115`, `--hint-k 0/3/8`): output identical at
every K (as INV-1 guarantees), and **expert hit 78.0% at all three** — unchanged to three
significant figures — while tok/s drifted DOWN 2.75 / 2.68 / 2.48 (the pilot's cost, nothing
bought).

The wiring is fine, which was checked rather than assumed: of **46,305** hints offered,
**36,529 (78.9%)** named an already-resident key — tracking the hit rate exactly, so the
vetoes have plenty to protect. They simply never bind:

- ~**7.2 evictions per layer** against a **6920-slot** pool, so a given key's chance of
  being chosen as victim inside a 1–2 layer veto window is ~0.1%;
- and LRU evicts the *coldest* key, while an expert predicted for the NEXT layer is warm by
  construction — it is at the opposite end of the queue from where eviction happens.

**The hint horizon is 1–2 layers; the eviction horizon is thousands.** A veto is the right
mechanism only where those overlap.

Three ways to make it bind, in order of how much they are worth testing:
1. **Raise the pressure.** At a small `--max-mem` the eviction horizon collapses toward the
   hint horizon. This is one cheap run and it decides whether the layer has ANY operating
   point; do it before anything else.
2. **Lengthen the horizon.** Predicting L+8 rather than L+2 would reach the eviction
   horizon, but precision falls with distance (77.2% at L+1, 68.9% at L+2) and a prediction
   names a *specific* layer, so this trades directly against accuracy.
3. **Let hints drive ADMISSION, not just protection.** This is the only option that helps a
   MISSING expert — a veto cannot protect what is not resident. It is also what the deleted
   `--pilot-k` preloader did, and that measured flat while moving 2.4× the bytes. Do not
   revisit it without a mechanism for the bytes.
