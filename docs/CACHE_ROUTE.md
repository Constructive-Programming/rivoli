# rivoli — `top-m`: cache-conditional MoE routing (arXiv:2412.00099)

Status: **proposed, not started.** Adds a fourth `--cache-policy` alongside
`lru | 2q | arc`, and is the first cache mechanism that **changes which experts run**.

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
**compaction** (a synchronous device memcpy relocating a boundary slot). And since our
fetch is already 92% hidden, the win lands as fewer host-gated compute bubbles in
`moe-gpu` rather than as fetch wall — the same mechanism the all-int4 investigation
identified.

## Design

**A fourth policy — but not a fourth policy implementation.** `--cache-policy top-m`
adds an arm to `hybrid::make` (`hybrid.rs:57`) that constructs the existing
`HybridLru` (the paper evaluates on LRU) configured with a rank-driven tier rule instead
of the frequency threshold, and with the routing advice switched on.

Do **not** fork a `HybridTopM` struct. This repo deleted three duplicate policy
families in `08db745`; re-adding one by copy-paste is the same mistake with a new name.
`HybridLru` gains a tier-rule field; everything else is shared.

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

Knobs, defaulting to the paper's values for this router class: `--route-j` (2) and
`--route-m` (12). The paper's cumulative-mass variant (a per-token window from a
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

**Hybrid mode: the router rank *is* the tier rule.** Today `HybridLru` promotes to the
Hot (int4) tier on an access-frequency threshold (`hybrid.rs:150`). Under `top-m` that
heuristic is redundant, because the router already tells us which experts matter — so
replace it rather than layer on top of it:

- **Sacred top-J → int4.** These run every time this layer is visited and are the
  highest-value experts in the layer. int4 is both more accurate and ~1.8× faster to
  compute (isolated: gate/up int4 669 vs vq3 353 GElem/s).
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
