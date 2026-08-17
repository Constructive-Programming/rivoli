---
status: live
scope: glm
verdict: THE MECHANISM IS A WRONG-BYTES READ IN THE ROUTED POOL, and the discriminator that says so has run on device. Coordinate: two arms at identical flags, quiet box, witness clean, ids diverged at body position 157; the divergence logs' first differing row is pos=164 nrow=1 layer=24, where `h` and `x` differ while `xn`, `gl`, `pk` and `sl` are IDENTICAL and relocs=0, misses=1. Attention output, router logits, picks and slot placement all agree — the same experts were selected and placed in the same slots, and the bytes read out of those slots hashed differently. So attention is out of frame, routing is out of frame (INV-1 holds), placement is out of frame, and relocation is not merely refuted by the barrier argument but ABSENT from the event row. BOTH previously named candidates stay refuted by construction: MoE accumulation cannot do it (per-term saturating fixed point, wrapping u64 atomicAdd, drain summing lanes at a fixed stride, so the resident/miss lane split is exactly equivalent to one accumulator) and relocation-vs-in-flight-read cannot (unconditional per-layer device_sync, ticketed miss kernels, host-awaited lanes, submit relocating only before it resolves any read). No float atomic exists in any kernel, so given identical bytes the arithmetic is bit-reproducible. THE RATE IS MEASURED AND THE RANKING QUANTITY IS THE FIRST DIVERGING POSITION, NEVER THE COUNT: per-run first divergences 236, 362, 375 with a fourth arm clean to 762 give 1 event per 299 (matched pair) or 1 per 578 (conservative, keeping the censored arm), exact 95% Poisson interval 1 per [206, 2804]; P(detect) = 1-exp(-2n/rate) predicts the greedy observations it was not fitted on, 0-of-3 pairs at 32 tokens and 3-of-3 at 512. Contention is NOT required — this pair ran on a quiet box. Positions 0..235 are bit-identical across two budgets and two source trees, so the engine reproduces until an event fires, and the wake then GROWS rather than decays. WHAT REMAINS is which HOP delivers the wrong bytes: the NVMe read, the bounce arena, or the DMA into the slot. Three fetch-path folds now split them (`bh` the arena after the read, `sc` the slot after the copy, `se` the same slot at end of layer) at ~10-15% throughput cost, and the rate being per-read rather than per-second is why a slower instrumented run should still fire. NOT ESTABLISHED: the ROCm 7.14/clang 23.1 toolchain is NOT exonerated — both binaries were built under it, so that experiment varied source and held toolchain fixed.
---

# GLM does not reproduce itself: the rate, and why both suspects are innocent

**Supersedes the page of this name on `wave/m10-spine`**, which measured the defect and
bounded its magnitude. Every measurement there stands. What changes here is (a) the rate and
the ranking quantity, (b) the attribution: that page's scope statement — "the wobble is
confined to the routed expert pool … those two candidates are NOT separated" — names two
candidates that **cannot be the mechanism in this tree**.

## The coordinate, measured on device 2026-08-17

Two arms, `--bench 512 --mode int3-vq --attn dense --max-mem 115`, release with
`corruption-probe`, **quiet box**, contention witness empty before and after. Generated ids
diverged at body position 157. `tests/divergence-columns.sh` on the two divergence logs:

```
FIRST DIVERGENCE at row 12427:  pos=164 nrow=1 layer=24
  -> column 5 (h) moved first

  column   A                    B
  xn       7e4e946650f24190     7e4e946650f24190
  h        9c7e5614ddb784fc     bb1a15e399e82fbf       <-- DIFFERS
  x        5f7ac21d78086fba     c4d30b6a9ae85370       <-- DIFFERS
  gl       18d6d2d88f7ef0f8     18d6d2d88f7ef0f8
  pk       7384d5fcc94ff963     7384d5fcc94ff963
  sl       ea11f614bd198904     ea11f614bd198904
  misses   1                    1
  relocs   0                    0
```

**Read by the decision rule this instrument was built around:** `xn` identical, so attention and
its KV cache produced the same MoE input. `gl` identical, so the router saw the same logits. `pk`
identical, so it picked the same experts — INV-1 holds at the event. `sl` identical, so the pool
put those experts in the same slots. And `h` differs. The same experts, in the same slots, fed the
same input, produced different SwiGLU intermediates; `x` moves as a consequence. **The bytes read
out of those slots differed.** Attention, routing and placement are out of frame.

**`relocs=0` on the event row.** Relocation is not merely refuted by the barrier argument above —
it is *absent from the event*. **`misses=1`**, so exactly one cold read landed on that layer at
that token, which is the obvious suspect and is what the three new fetch-path folds are aimed at.

**Contention is not required.** The box was quiet and the event still fired at token 164. That
matches the rate model (an event per ~300 token-forwards) rather than a load-triggered race, and
it is what makes an instrumented — therefore slower — run still worth running: the rate is per
READ, not per second.

Two notes on the run itself, both worth keeping:

- **The arms generated different lengths** (25,272 vs 22,386 log rows) because they hit EOS at
  different points once diverged. `tests/determinism-glm.sh` classified that as a setup error, and
  it is not — a length difference IS a divergence. Fixed: lengths are now compared to each other
  before either is compared to `ngen`.
- **The gate refused this configuration**, correctly, on its own precondition: with no `--prompt`
  the engine's default hits EOS at 276 tokens, so a 512-token floor is unreachable. Fixed by
  recording a long-form prompt VERBATIM in the gate, pinned by length and md5 in its `--self-test`
  so it cannot drift — the placeholder-prompt failure that makes
  `docs/measurement/baseline-2026-08-16.md` unreproducible from its own record.

## Two corrections to the record, before anything is built on it

**1. The toolchain is NOT exonerated.** The inherited page and the brief that opened this
milestone both concluded that because the pinned OLD binary diverges too, "the ROCm 7.14 /
clang 23.1 upgrade did not introduce it". That does not follow. **Both** binaries were built
on this box under 7.14, the only installed toolchain, so the experiment varied SOURCE and held
TOOLCHAIN FIXED. What it establishes is two narrower things: it is not a rewrite regression,
and the upgrade did not RETIRE a pre-existing defect. Whether the upgrade introduced it is
**untested**, and testing it needs the old source under the old toolchain, which is not
installed. Treat toolchain as an open variable.

**2. The ranking quantity is the FIRST DIVERGING POSITION, never the differing count.** Past
the first event the two runs are computing over different state, so every later position
differs whether or not anything else went wrong. "496 of 512 ids differ" is **at least one**
event plus its wake — not 496 events, not a severity, and not comparable between runs: it is a
function of *where* the first event landed and nothing else. A count-based rate would have
said the loaded box was 8× worse than the quiet one; the honest statement is that its first
event arrived 35× sooner. `tests/nll-divergence.sh` computes the first position and refuses to
report a count as a rate.

## The rate, measured

Four teacher-forced arms are vendored at `docs/measurement/glm-divergence-evidence/` (33 KB,
one f32 per predicted position). They came from `wave/m10-spine`'s `p4` cell on 2026-08-17.
**Their provenance is what each file's own header says and no more** — an earlier draft of this
paragraph asserted a uniform set of flags that the headers do not carry, which is the failure this
whole page is about:

| file | header, verbatim | what is NOT recorded |
|---|---|---|
| `a.nll` | `mode=int3-vq policy=2q attn=dense max_mem=115 corpus=1253f29de5a93559 tokens=762 hit_pct=78.2591` | profile, binary, date |
| `a2.nll` | same, `hit_pct=78.2597` | same |
| `b.nll` | same but `max_mem=70`, `hit_pct=62.0900` | same |
| `ref.nll` | `mode=int3-vq policy=2q moe_gain=1 tokens=762 hit_pct=78.1293` | **no `attn`, no `max_mem`, and NO CORPUS HASH** — a different writer, i.e. the pinned old binary, not the same `p4` invocation |

`ref.nll`'s missing corpus hash is the one that matters: the property `wave/m10-spine` cited as
making its numbers reproducible ("the FNV-1a the engine logs and writes into the `.nll` header")
is absent from exactly the arm whose provenance is hardest to check. So `ref` is used below only
where it cannot carry a conclusion on its own, and the rate is taken from `a`/`a2`, which are the
only two arms whose headers agree in every field but the hit rate.

**`a2`'s hit rate differs from `a`'s (78.2597 vs 78.2591) at the same budget on the same corpus,
and that is a result, not noise in the bookkeeping.** Teacher forcing fixes the token sequence,
so routing should be identical — unless the hidden state diverges, which changes the gate logits,
which changes the picks, which changes what hits. The hit rates differing is independent
confirmation that the event is real and lands upstream of routing.

Reproduce every number below with:

```
tests/nll-divergence.sh docs/measurement/glm-divergence-evidence/{a,a2,b,ref}.nll
tests/nll-divergence.sh --se    docs/measurement/glm-divergence-evidence/{a,a2}.nll
tests/nll-divergence.sh --power 2 598      # matched-pair reading
tests/nll-divergence.sh --power 3 1735     # conservative reading
```

| pair | ndiff | first | \|dNLL\| at the event |
|---|---:|---:|---|
| a vs a2 (**same budget, the control**) | 526 | **236** | 1.21e-2 |
| a vs b (115 vs 70) | 526 | 236 | 1.21e-2 |
| a vs ref (old binary) | 526 | 236 | 1.21e-2 |
| a2 vs b | 400 | **362** | 3.20e-2 |
| a2 vs ref | 400 | 362 | 3.20e-2 |
| b vs ref | 387 | **375** | 2.04e-3 |

Four readings, in order of how load-bearing they are:

**Positions 0..235 are bit-identical in all four arms.** Two budgets (78.26% and 62.09% hit —
1.7× the misses) and two source trees agree exactly for 236 positions. So the engine *is*
reproducible and *is* budget-neutral (P4, INV-1) right up until an event fires. Whatever this
is, it is rare, not pervasive.

**The events are per-RUN, not per-position.** Every pair involving `a` breaks at 236 because
`a` is the arm that broke there; among the others the earliest disagreement is 362, then 375.
Reading first-disagreements as minima: `a` at 236, `a2` at 362, one of `{b, ref}` at 375, the
last censored past 762. **236 is not a structural boundary** — which is worth saying because
it looked like one for an hour, and the attention split plan was the obvious suspect. It is
not: `mla_plan_splits` at GLM's `H=64, HB=16` changes cut at `nr` = 193 and 241, so 236 sits
mid-plateau with `(n_splits=3, tps=5)` on both sides. **Split-KV is exonerated as the trigger.**

**The rate, and the inference behind it.** The events are per-run, and only `a`'s is read
directly: `a` vs everything breaks at 236, so `event(a) = 236`. `a2` vs `b` and `a2` vs `ref` both
break at **362**, which is `min(event(a2), event(b))` and `min(event(a2), event(ref))`; if
`event(a2)` were later, then `event(b)` and `event(ref)` would both have to be exactly 362, so
`event(a2) = 362`. `b` vs `ref` then breaks at 375, giving `min(event(b), event(ref)) = 375` with
the other censored past 762. **That chain is the whole basis for the numbers below and was
implicit in an earlier draft** — a reviewer read "the matched pair gives two events" as
self-contradictory, correctly, because a pair yields one first-divergence unless the other arm's
event is inferred this way.

Two readings, and neither is buried:

| reading | events / exposure | rate | exact 95% Poisson interval |
|---|---|---|---|
| matched pair only (`a`, `a2`, both at `--max-mem 115`) | 2 / 598 | 1 per **299** | 1 per [83, 2469] |
| all four arms, the clean one contributing its full 762 | 3 / 1735 | 1 per **578** | 1 per [206, 2804] |

The second mixes configurations (`b` at a different budget, `ref` a different binary) but keeps
the censored arm's exposure, so it is the **conservative** one and it is what
`tests/determinism-glm.sh` is sized against. A pair diverges if either arm has an event, so
P(detect) = 1 − exp(−2n/rate):

| ngen | at 1/299 | at 1/578 | at the conservative interval's pessimistic end |
|---:|---:|---:|---:|
| 32 | 19% | 10% | 2% |
| 256 | 82% | 59% | 17% |
| 512 | 97% | 83% | 31% |

80% power needs ngen 241 at the optimistic reading and **465 at the conservative one**, which is
why the gate's floor is 512 rather than 256 — an earlier draft set it at 256 off the matched
reading alone. **And the interval is the point: at its pessimistic end even 512 tokens is 31%
power.** A green bounds the rate; it does not prove determinism.

**That curve was fitted on teacher-forced NLLs and predicts the greedy id observations it was
not fitted on.** It also disposes of the "32 tokens is safe" reading: there is no safe length,
only a probability, and the 32-vs-512 contrast needs no extra mechanism to explain.

**The wake GROWS, it does not decay.** `a` vs `a2`, median |dNLL| by 60-position window from
the event: 2.8e-3, 2.6e-3, 1.7e-2, 3.6e-2, 3.3e-2, 2.0e-2, 3.5e-2, 3.5e-2, 3.8e-2. One event,
then two KV caches drifting further apart forever. And the magnitude AT the event — 1.2e-2 to
3.2e-2 nats — is far too large for a rounding difference and is the size of one expert's
contribution to one position going wrong.

### Why teacher forcing and greedy decode disagree about the magnitude, and which to use

Teacher forcing re-anchors every position to the committed corpus, so a perturbation propagates
only *numerically*, through the KV cache. Greedy decode feeds its own output back, so one flipped
argmax rewrites the tail. Same defect, two magnitudes. **A teacher-forced probe measures
perturbation MAGNITUDE; a greedy probe measures divergence TIMING.** Pick per question; do not
compare their numbers.

**The magnitude, derived from the vendored pair rather than quoted.** `a` vs `a2`: mean |dNLL|
**0.035252**, mean signed dNLL **−0.004249** (`--se`). An earlier draft of this section put the
teacher-forced wobble at "0.0018 nats", which is `wave/m10-spine`'s figure — and it is neither
of these, because it was computed as `ln(5.209284/5.200080)` over a **different** `p4` invocation
(hit 78.2643/78.2352, against these files' 78.2591/78.2597). Quoting it beside CIs derived from
*these* files presented two runs as one body of evidence. It is dropped rather than corrected a
second time.

### The independence check `wave/m10-spine` owed, now paid — and what it changes

That page carried a boxed caveat: `bin/ppl` reports `SE = sd/sqrt(n)`, which assumes the
per-position dNLLs are independent; this defect makes one event contaminate every later position,
so the assumption is exactly the one it violates; the true half-width could be ~3.9× wider; and
*"a lag-1 autocorrelation or a block-bootstrap SE over any control pair's two `.nll` files settles
it… those files were not retained… preserve that directory and do this check before any downstream
branch leans on the multiples."*

**This branch vendored those exact files, so the check is done** (`--se`, 2026-08-17). Two
deterministic diagnostics, no RNG:

| pair | mean dNLL | naive SE | lag-1 ρ | block SE (L=50) |
|---|---:|---:|---:|---:|
| `a2 − a` (control, same budget) | −0.004249 | 0.00302 | **−0.0797** | 0.00165 = 0.55× naive |
| `b − a` (70 vs 115) | −0.003367 | 0.00232 | **−0.0093** | 0.00214 = 0.92× naive |

**ρ is NEGATIVE, so the feared inflation does not occur: the naive SE is conservative here, not
optimistic.** That retires the caveat, and it is retired by measurement rather than by
supersession — which matters, because this page's first draft said "every measurement there
stands" and silently dropped the obligation. `wave/m10-spine`'s ~3.9× worry is **refuted for these
two pairs**; it remains untested for a pair whose event lands early enough to leave a long
correlated tail, and `--se` is one command on any future pair.

**One conclusion has to be corrected, and it is one of mine.** Under the naive SE both CIs contain
zero, which the first draft read as "the wobble is symmetric noise, not a bias". Under the tighter
block SE the control pair's CI is [−0.00748, −0.00101] — it **excludes** zero. So a pair's mean
dNLL is genuinely non-zero: one run's event shifts its whole tail in one direction. What that does
*not* mean is a bias, because the sign is a property of which arm had the event and where, and n=1
constrains it not at all.

**The `--max-mem` conclusion survives, and on better evidence than "both contain zero".** The
control pair (same budget) and the budget pair move by the same amount in the same direction
(−0.0042 vs −0.0034, ratio 1.3). Comparing them to *each other* is what says there is no budget
effect — a comparison that does not depend on either interval, and the one the re-specified `p4`
cell exists to make.

## What is REFUTED, and by what

Both refutations are structural — read off the code and the kernels — which is why they are
stated as refutations and not as further exclusions. Both are also *reasoning*, not
measurement, which is why `--divergence-log` carries the columns that let a device run check
them (see `Cols::reloc`).

### MoE accumulation order cannot do it

Checked in this tree rather than trusted, because the standing claim in
`docs/reference/architecture.md` is exactly the kind of inherited assurance this repo distrusts:

- `common.hpp::moe_fixed` clamps **each term independently** (`llrintf(fmin(fmax(v, ±MAX)) ·
  2^44)`), so saturation cannot depend on arrival order.
- `moe.hip:124`/`:371` accumulate with `atomicAdd` on `unsigned long long`. Integer addition is
  associative and commutative, and `common.hpp`'s width argument bounds Σ over ≤16 clamped
  terms at 2^62 — a full binade of slack, so no wrap occurs either.
- `moe_acc_drain_impl` sums the `MOE_ACC_ROWS` lane blocks into a `long long` at a fixed stride
  and converts **once**, in `double`.

So the resident-lane / miss-lane split (`glm/mlp.rs`, two streams, two accumulator blocks) is
*exactly equivalent* to one accumulator: which lane a contribution lands in is a
residency-dependent decision that provably cannot change the sum. Batching maximal resident
runs into one launch is the same argument.

Stronger, and the load-bearing generalisation: **there is no float atomic in any kernel in the
tree.** `grep atomic kernels/*.hip *.hpp` yields the u64 MoE `atomicAdd`, the `atomicCAS` on
the non-finite flag, and a u32 histogram in the DSA indexer that `--attn dense` never launches.
**Given identical input bytes, this engine's arithmetic is bit-reproducible** — so the defect
has to be a wrong-bytes read.

### Arena relocation vs an in-flight read cannot do it *in this tree*

The prior hypothesis — a read outliving its layer and having its slot `memcpy`'d out from under
it, pins not stopping compaction, seen as 9 corrupted reads in 8452 at `--max-mem 30` —
describes the OLD tree. Here three barriers close it, and they compose:

1. `glm/forward.rs::run_layer` ends **every** layer with `device_sync()` =
   `hipDeviceSynchronize` — the whole device, all streams.
2. Every miss kernel is enqueued behind `hipStreamWaitValue64` on its ticket, and the reaper
   signals that timeline on the fetch stream only *after* enqueueing the bounce→slot copy.
3. `launch_moe` host-awaits both lanes (`hipLaunchHostFunc`-backed) before returning, so all of
   layer L's copies have executed before layer L+1 submits anything.

So when `admit_misses` evicts, frees and relocates, **the device is idle**. And `submit`'s
phase order forbids the remaining shape independently: all relocation happens in
`admit_misses`, and `resolve` computes final slots and issues reads only afterwards, so a read
never targets a slot that later moves. `relocate`'s `memcpy_dtod` is a blocking `hipMemcpy`,
which suffices *because* nothing else is in flight — note it would NOT suffice otherwise, since
rivoli's streams are `hipStreamNonBlocking` and the null stream carries no implicit ordering
against them.

## What three read-only audits found, so nobody repeats them

Audits of (a) the sync and memory-visibility substrate, (b) the host residency path, (c) the
whole non-MoE GLM path, 2026-08-17. The negative results are the useful part:

- **No hash-order, time-derived, address-derived or unstable-sort decision** reaches an
  eviction, a tier, a relocation order or a read order. The one `retain` over a `HashMap` (the
  LFU halving) is order-INVARIANT — its closure reads and writes only the entry's own value.
  Recency is a `BTreeMap` on a monotonic tick. `route_into`'s comparator carries an index
  tiebreak, so top-k ties resolve to lowest index.
- **The `Arena`'s relocation sequence is a pure function of its `(alloc, free)` sequence.** Its
  free lists are `Vec` with `pop`/`swap_remove`: history-dependent, fully determined.
- **Argmax is a single-block reduction** whose combine resolves an exact tie to `min(index)`
  explicitly, with `__syncthreads` between rounds and no atomics. A tie cannot resolve
  differently across runs. Do not revisit this.
- **No buffer in the GLM path is read before it is written**, padding included: every producer
  and consumer is parameterised on `nrow` (the row count is a *template* parameter, never a
  grid dimension), `moe_acc` and `argmax_dev` are explicitly zeroed at construction, and
  `moe_hidden`'s two lanes index by the ABSOLUTE descriptor index so they are disjoint. The one
  full-buffer D2H of a half-uninitialised staging buffer (`gate_logits_host` at `nrow == 1`) is
  consumed only in its written half.
- **The KV cache reads exactly `pos + r + 1` rows**, no padding, and the split plan is a pure
  function of `(H, nr)`.

There is no live ordering hole and no unwritten read. **The defect is not statically visible**,
which is why this milestone's deliverable is instruments and a gate rather than a fix.

## The instruments

### `tests/nll-divergence.sh` — deviceless, and what it does and does not re-derive

First-divergence position, magnitude at the event, the wake profile, the SE diagnostics, and the
rate/power curve — all from the vendored files, with no device. It compares the recorded value
TEXT, with no tolerance: the question is bit-reproducibility, and a per-position tolerance would
hide exactly the small early perturbation whose position is the measurement.

**It does not make every number on this page derived, and the first draft claimed it did.** What
it re-derives is the divergence table, the wake profile, the SE diagnostics, and the rate and
power from `(events, exposure)` — which it now takes explicitly, so a censored arm cannot be
dropped silently. Everything below is INHERITED from `wave/m10-spine`'s prose or from the brief
that opened this milestone, is not checkable in this tree, and is marked as such wherever it
appears:

| inherited number | what it is | why it is not checkable here |
|---|---|---|
| 61/512, 496/512, 247/512; first divergence 13, 265, 452 | the greedy-decode observations | the id dumps were not retained |
| 0-of-3 pairs at 32 tokens, 3-of-3 at 512 | the greedy pair outcomes | same |
| 70.6% MTP acceptance, `p0.8+` at 89% | `wave/m12-glm-chain`'s measurement | another branch |
| 7.7 GB/s at QD1, ~35% ring idle | the old tree's fetch probes | no probe in this tree |
| 9 corrupted reads in 8452 at `--max-mem 30` | the old tree's relocation finding | archive only |
| ~148 misses/token, ~20 MB/expert | the old tree's poison-fill comment | superseded below by a DERIVED 130.4 and 14.625 MiB |

The greedy numbers are load-bearing for one claim only — that the defect reaches generated text
and is length-dependent — and the rate derived here independently predicts their 32-vs-512
pattern, which is the strongest check available without re-running them.

### `tests/divergence-columns.sh` — read a pair of logs BY COLUMN

Two logs in, the first differing row and **which column moved first** out. It exists as a script
rather than a snippet for a recorded reason: it *was* a snippet in this doc, and the operator
running the first real pair could not find it and wrote the comparison a second time. A procedure
that gets re-derived by whoever needs it is not recorded, whatever the doc says.

It takes the column names from **each file's own header** rather than from a list of its own, so it
cannot drift from the writer and it reads v2 and v3 logs alike; it refuses two logs whose headers
disagree, since those came from different builds and are not comparable. It also does not require
equal lengths — after diverging, the arms generate different numbers of tokens.

### `--divergence-log` — a coordinate AND a mechanism

Forward-ported from archive `544fea7` (reachable from `archive/belady-residency-bound`), not
cherry-picked: that commit's `--checksum-x` and `--checksum-route` were written against a
single-file `gpu.rs`, and they are folded here into one flag and one file format. Two of its
design decisions are inherited verbatim and are why it can be pointed at this bug at all:

- **The fold is XOR** (`rivoli_core::hash::xor_fold` and its device twin
  `kernels/fwd.hip::hash_rows`), because XOR is commutative *and* associative and so is
  bit-identical whatever order the atomics land in. A float sum would report a difference from
  scheduling jitter alone.
- **Nothing touches the host or the disk mid-run.** The predecessor copied the residual to the
  host every layer and produced a CLEAN run on a configuration that reproduced without it — the
  tool built for the bug could not be used on it.

What is new is that it discriminates rather than only localising. Three device-folded quantities
per layer cut the layer at the two seams the refuted candidates sat either side of:

| column | quantity | folded for | a difference here, with the earlier columns equal, means |
|---|---|---|---|
| `xn` | the MLP's input (post-attention rmsnorm) | every layer | attention or its KV cache; the MLP has not run |
| `h` | `moe_hidden`, the SwiGLU intermediate | MoE layers | the gate/up expert BYTES, or that kernel |
| `x` | the residual at layer exit | every layer | the down projection, the accumulator or the drain |

`xn` is folded on GLM's 3 dense layers too, which is not an incidental detail: folding it only
on the 75 MoE layers left the dense rows' `xn` at 0 in both runs, and a diff reads two equal
zeros as "attention agreed" when nothing was measured. Every column that cannot exist for a
layer prints `-`, never 0 — a false EXCLUSION is the one failure mode an instrument must not
have, and this one had it until it was caught here.

plus five host columns that cost nothing because routing is already a host function of
host-resident data: `gl` (what the router saw, over exact bytes), `pk` (what it picked), `sl`
(WHERE the pool put each expert — arena **offsets**, never addresses, since the VMM base
differs per run), and the layer's `misses` and `relocs` deltas. Diff two logs: first differing
LINE is the coordinate, first differing COLUMN names the mechanism.
`fwd_kernel.rs::hash_rows_matches_the_host_fold` scores the kernel against the host fold and
pins bit-exactness, one-ULP sensitivity and permutation sensitivity — every conclusion here
will be read off a pair of these hashes, and an instrument nobody checked is a source of
confident wrong answers.

**`trace` deliberately does not imply the probe's feature.** The archived predecessor did,
because its flag had always lived under `trace`; `--divergence-log` is new, so there is no flag
to keep, and implying it would arm the probe in exactly the build whose extra per-layer
`device_sync` invalidates the measurement. **Never debug this under `--trace`.**

## The gate

`tests/determinism-glm.sh`: two runs, one binary, byte-identical arguments, ids compared.
`ngen` has a hard floor of **256**, and the floor is the power calculation above rather than a
round number — the gate prints the curve when it refuses. Default 512. A green **bounds the
rate at that length; it does not prove determinism**, and the gate says so in its own output.
It carries a per-arm contention witness (now shared with `tests/parity-glm.sh` via
`tests/gpu-witness.sh` — a two-copy false-green guard is one copy away from not guarding),
never builds, and refuses a short arm, since two short arms of equal length would otherwise be
a green over a decode that never happened.

Red proof, both halves, run 2026-08-17:

- **Mechanical, deviceless:** `--self-test` feeds the gate's own comparator two id streams
  differing by one token, and a truncated stream, and fails if either compares equal. Reddens
  on both. A proof about the comparator, labelled as such.
- **Live:** the defect is present, so a 512-token arm on this tree IS the red proof, and the
  table above is its record.

**INV-9** (`docs/reference/architecture.md` §8b, paired with
`asyncfetch.rs::inv_9_slot_handout_is_deterministic_under_the_layer_barrier`) covers the
same-PROGRAM half — that no host decision on the routed path can make two runs of one input
diverge — and scopes itself explicitly away from the same-OUTPUT half, which is false today.
Red-proofed by making `scan_free` ignore `landed`.

## One capability gap that shapes experiment 2

**`--divergence-log` can only be pointed at GREEDY decode on this branch.** It is wired into
`run_bench`, and there is no `--ppl` / teacher-forcing decode path in this tree at all — the
`teacher-forcing` feature is declared empty here and the `.nll` evidence above came from
`wave/m10-spine`'s `score.rs`. That matters for how experiment 2 is read:

- **Greedy is still interpretable**, and this is the important half: the first differing LINE is a
  divergence that happened *during that forward, on identical inputs*, so its (layer, column) is a
  real coordinate. Every line after it is trivially different and carries nothing.
- **What is not reachable is the better measurement.** Teacher-forced, both arms stay re-anchored
  to the corpus, so a second and third event remain visible instead of being buried under the
  first one's wake — a run would yield several coordinates rather than one. That needs the probe
  wired into a TF walk, i.e. it lands when this branch and `wave/m10-spine`'s scorer meet.

So experiment 2 gives ONE coordinate per pair, and the ladder below should be read as
"one sample decides which arm we are on", not as "one run settles the mechanism".

## What the instrument now measures, and the one question left

Three fetch-path folds were added after the coordinate came in, to split the one remaining
question — **which hop delivers the wrong bytes**:

| column | folded where | a difference means |
|---|---|---|
| `bh` | the pinned bounce arena, on the fetch stream, the moment the NVMe read completes and **before** the copy | the DRIVE delivered different bytes |
| `sc` | the pool slot, on the fetch stream, immediately **after** the copy | with `bh` equal: the COPY |
| `se` | the same pool slot again at end of layer, after both MoE lanes are awaited | with `sc == se` and `h` differing: the bytes AT REST are RIGHT, so the kernel read them **too early** — a ticket/timeline ordering failure, not a bad payload. With `sc != se`: something wrote the slot after the copy landed |

All three are XOR folds at **full width**, so none can miss a corruption by sampling, and all three
are stream-ordered around the copy with **no host sync and no barrier** — which is the only reason
an instrument may be pointed at this defect at all.

`se` is deliberately narrower than the others: it folds only the experts the layer MISSED. Folding
the whole batch would be ~9 experts × 14.6 MiB × 78 layers ≈ 10 GB/token of extra device reads
against ~2 GB for the misses. **So `se` is silent about a RESIDENT expert corrupted on an earlier
token, and that limit must be stated with any result it produces.** If `h` differs while all three
of `bh`, `sc`, `se` agree, that is the surviving reading and it points at a resident expert.

**Cost, and why it is worth paying.** Three full-width folds add ~6 GB/token of device reads on top
of the ~2 GB the fetch path already moves. Those reads come from system RAM at LPDDR5 bandwidth
rather than from NVMe, so the arithmetic is ~40–60 ms/token against the 388 ms the committed
baseline records — **roughly 10–15%**, not the 2× a naive reading of the byte volume suggests.
And the masking risk is bounded by the measurement above: the event fired on a QUIET box, and the
rate is per READ rather than per second, so a run 15% slower issues the same number of reads and
should fire at a similar TOKEN position. **If it does not fire, that is itself informative** — it
would mean the instrument perturbs, which is the `--checksum-x` failure and would have to be
recorded as such rather than read as a green.

### Still open, and cheap

- **Grep any surviving stderr from the diverging runs for `READ-BEFORE-WRITE`.**
  `routed.rs::touch_hits` carries a live, **unconditional** detector for "the policy counted this
  expert a HIT but its bytes never landed since admission". If it fired, it names the coordinate
  independently; if it was silent, the premature-read half is excluded at zero cost. **Its blind
  spot must be stated with the result:** it detects bytes that never arrived, never bytes that
  arrived and were then clobbered — which is precisely what `se` is for.
- **Log `slot_stalls()` per token.** It should be identically 0; INV-9 rests on that. Non-zero
  means the two runs stopped being the same program, and it is the only host-side amplifier left.
- **A budget high enough that the routed pool never evicts**, if one exists on this box — the
  control pair going byte-identical would put the fault squarely in the fetch path. Note the weak
  counter-evidence already in hand: `b` at `--max-mem 70` (1.7× the misses) had its event LATER
  than `a` at 115, which does not support a rate that scales with miss count. n=1 per budget.
- **The toolchain remains untested.** It needs the old source under the old toolchain, which is not
  installed.

## Defects found on the way

**Fixed here.** `asyncfetch.rs`'s `debug_assert_eq!(sub, 0, "VQ expert read must be
block-aligned")` was the only thing "checking" two properties the fetch path depends on, and
under `--release` — which is what every benchmark and every divergence run is built with — it
checked neither. It is replaced by `RoutedGeom::check_reads_fit_their_slots`, one pass over the
read table at `open()`: every read must start `ALIGN`-aligned (or the bounce→slot copy lands the
aligned superset at the slot BASE and shifts all six projection pointers) and its superset must
fit one `stride` (or the copy spills into the neighbouring slot and corrupts another resident
expert — precisely the shape of defect under investigation). Both hold today by arithmetic in
another crate (`VQ_ALIGN == ALIGN == 4096`, blocks at `VQ_ALIGN + e·stride`), so the check is
expected to stay silent; "holds by arithmetic elsewhere" is exactly the coupling a repack or a
new format breaks quietly. This is the repo's most common review finding wearing a new hat, in
the fetch path of an open defect.

**Open, one `ensure!` each, both inert for the real artifact.** Nothing asserts
`lm_head.o_dim == cfg.vocab`: `tail()` writes `logits` at width `o_dim` while `argmax_rows`
reduces over `cfg.vocab`, so a smaller `o_dim` reduces over uninitialised memory and a larger
one writes past `logits` into whatever `hipMalloc` handed out next (which is `argmax_dev`).

**Corrected, not a defect.** `glm/engine.rs`'s comment describes `moe_acc` as
`[MOE_ACC_ROWS][MAXROW][hidden]`, but the live lane stride is `nrow · hidden`, matching the
drain's own `n`. Writer and reader agree and every drain zeroes exactly what it reads, so it is
correct — but a third consumer trusting the comment would alias.

## What this costs the record

The old tree's byte-identity claims — gated MTP at `--mtp-min-conf 0.8`, the parity gates, the
quality ladder's A/Bs — were measured on an engine that does not reproduce itself over long
runs. They are not thereby wrong; they are **unproven at any length where an event is likely**,
and the power table says what that means for a given length. Any future byte-identity claim on
GLM must state its token count. `wave/m12-glm-chain`'s MTP losslessness gate is the immediate
casualty: at 70.6% acceptance it is well past break-even and cannot close until the baseline
reproduces.
