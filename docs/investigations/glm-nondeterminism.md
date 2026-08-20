---
status: live
scope: glm
verdict: SUPERSEDED 2026-08-18 by glm-nondeterminism-closeout.md — read that instead; this is the lab notebook, kept for how each arm was run and what each probe measured. Its leading claim is REFUTED (the defect is not in the bounce arena: `--direct-vmm-dma` had no arena and still diverged 91/512 @420); the completed ablation matrix and its one mislabel correction are in the body; the isolated path is exonerated (1e6 repro reads, 0 mismatches — the hazard needs the engine's own concurrency); the ranking quantity is the FIRST DIVERGING POSITION, never the count, and the rate scales per READ. No fix was claimed here — the surviving operational rule and the `--copy-via-cpu` fix candidate are the closeout's §6.
---

# GLM does not reproduce itself: the bytes read out of the pool slots differ

**Supersedes the page of this name on `wave/m10-spine`**, which measured the defect and bounded its
magnitude. Every measurement there stands — including the independence check it recorded as OWED,
which this page pays rather than inherits (see below). What changes here is three things: the rate
and the ranking quantity; the attribution, since that page's two named candidates **cannot be the
mechanism in this tree**; and the mechanism itself, now localised on device to a wrong-bytes read in
the routed-expert pool.

The one question left is which HOP delivers them — the NVMe read, the bounce arena, or the DMA —
and the instrument that splits those is built but has not run.

## The coordinate, measured on device 2026-08-17

Two arms, `--bench 512 --mode int3-vq --attn dense --max-mem 115`, release with
`corruption-probe`, **quiet box**, contention witness empty before and after. Generated ids
diverged at body position 157. `tests/divergence-columns.sh` on the two divergence logs:

```
FIRST DIVERGENCE at row 12427:  pos=164 nrow=1 layer=24
  -> column 5 (h) moved first

  column   A                    B
  pos      164                  164
  nrow     1                    1
  layer    24                   24
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
  the engine's default ("The sky is blue because", 11 prompt tokens) hits EOS well before 512 — 276,
  281 and 318 generated ids observed across retained runs, and *that spread is itself the defect*,
  since diverged arms stop at different points. So a 512-token floor is unreachable there. Fixed by
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

80% power needs ngen 241 at the optimistic reading and **465 at the conservative one**, which is why
`tests/determinism-glm.sh`'s floor is **512** — an earlier draft set it at 256 off the matched
reading alone, and 256 is now nowhere in the gate. **And the interval is the point: at its pessimistic end even 512 tokens is 31%
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
| 236,570/70,030 and 714,436/206,564 hit/miss; 42,666 and 122,538 log rows | the two heavy-probe pairs | the logs were not retained here; the ZERO-EVENTS result is the load-bearing part and the counts are corroborating detail |
| int4: body line 28, 95,351 misses | the int4 pair | same — the ids were not retained here |
| ~148 misses/token, ~20 MB/expert | the old tree's poison-fill comment | **superseded by derivation** — see the note below |

**The two numbers that were inherited and are now derived**, because the cost figures above rest on
them and an earlier draft quoted the old tree's: **misses/token = 130.4** (75 MoE layers × `top_k` 8
= 600 routed lookups, at the 78.2591% hit rate in these `.nll` headers) and **expert stride =
15,335,424 B = 14.625 MiB** (`L03.vq3` is 3,941,208,064 B, the layout is `hbytes + blocks · stride`,
and `blocks` is **257** — 256 routed plus the shared expert, as `artifact/src/format/layer.rs` states
outright; 4096 + 257 × 15,335,424 reproduces the file size exactly). The old tree's "~20 MB per
expert" is the **`.i4`** stride, and assuming 256 blocks instead of 257 gives 15,395,328 B, which is
not a multiple of 4096 — under which `RoutedGeom::check_reads_fit_their_slots` would refuse the real
artifact at startup. Both mistakes were made and caught here; this is the inherited-number failure
this repo keeps writing down.

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

## One capability gap that shapes the next run

**`--divergence-log` can only be pointed at GREEDY decode on this branch.** It is wired into
`run_bench`, and there is no `--ppl` / teacher-forcing decode path in this tree at all — the
`teacher-forcing` feature is declared empty here and the `.nll` evidence above came from
`wave/m10-spine`'s `score.rs`. That matters for how the next instrumented run is read:

- **Greedy is still interpretable**, and this is the important half: the first differing LINE is a
  divergence that happened *during that forward, on identical inputs*, so its (layer, column) is a
  real coordinate. Every line after it is trivially different and carries nothing.
- **What is not reachable is the better measurement.** Teacher-forced, both arms stay re-anchored
  to the corpus, so a second and third event remain visible instead of being buried under the
  first one's wake — a run would yield several coordinates rather than one. That needs the probe
  wired into a TF walk, i.e. it lands when this branch and `wave/m10-spine`'s scorer meet.

So an instrumented pair gives ONE coordinate, and the hop table above should be read as "one sample
decides which hop we are on", not as "one run settles the mechanism".

## The completed ablation matrix — and a MISLABEL that inverts its inference

| intervention | reads the arena? | outcome |
|---|---|---|
| v2 light · `sc` · `sc-nop` · `se` · `xa,ac` | no | RED (169 / 236 / 292 / 301 / 292) |
| `bh-nop` — launch + dispatch acquire, no read | no | RED @292 |
| `bh-decoy` — same bytes, NOT the arena. **NOT same duration — see the correction below** | no | RED **@12** (earliest in the investigation) |
| `bh-line` — **stride 32, one dword per 128 B line** | yes, ~1/32 of the bytes | RED **@704** (latest) |
| **`bh` — every byte of the just-written region** | yes, all | **CLEAN 1536** |
| `bh,sc,se` | yes, all | CLEAN (512 + 1536) |
| `--pinned-coherent` (fine-grained arena, **flag observed to apply**) | — | RED 225/512 @283 vs control 220/512 @291 |

Two conclusions are solid. **It is not a dispatch/ordering acquire** (`bh-nop` red). And
**host→device visibility of the arena is refuted with the intervention observed to apply** —
`hipHostGetFlags` returned `0x40000000`, coherent-bit true, so `hipHostMallocCoherent` is not a
no-op on ROCm 7.14 and the arena really was fine-grained. That branch is closed properly rather
than by assumption, and the read-back is now automatic for any candidate fix.

> **CORRECTION 2026-08-19 — the `bh-decoy` cell never held duration constant, so "not bandwidth
> or delay" was never established.** The decoy is `probe.rs`'s `DeviceBuf` — DEVICE memory
> (`hipMalloc`), read at ~135–220 GB/s — while `bh` reads the PINNED HOST arena at the ~13 GB/s
> the arena-refresh cost implies (~1.15 ms per miss against ~0.1 ms for the decoy). The arm that
> was recorded as the equal-duration control delivered roughly a TENTH of the delay. The
> delay/bandwidth hypothesis was therefore open the whole time, and it is what the
> `--fetch-settle-us` (pure host time, no device work) and `--arena-refresh-decoy` (the same
> touch aimed at a second pinned-HOST arena the NVMe never writes) arms exist to close. The
> `sc`-ladder decoys are unaffected: there both the slot and the decoy are device memory.

**`bh` is therefore a Heisenberg probe, not a fix candidate** — ~10% of throughput on a hash nobody
reads.

### The mislabel, and what it costs

`bh-line` was recorded as "one cache line of the arena" and read as "bulk: no", giving
*"one line insufficient, all lines sufficient ⇒ a per-LINE effect"*. **That is not what the arm
does.** `LINE_F32 = 32` f32 = a **128 B stride**, so `bh-line` reads one dword every 128 B **across
the whole region** — it touches *every* line, at ~1/32 of the bytes. Both it and `bh` touch every
line.

So the per-line reading is not established. What the matrix actually says is that **stride-32
sampling is insufficient and stride-1 is sufficient**, and `bh-line` is the LATEST red in the
investigation (704 against a ~292 median) — a **dose-response in bytes read**, not a threshold in
lines touched.

**The granularity is therefore unknown, and that is now the cheapest decisive question.** One
hypothesis with a concrete prediction: a single dword touch pulls only the containing **64 B
sector**, not both halves of a 128 B line, so stride 32 repairs half of each line and stride 16
would repair all of it. If so, **stride 16 is clean at 1/16 of the bytes** — a fix, where `bh` is
only a probe.

`--divergence-folds bh-line:N` makes that a sweep. `N ∈ {1, 4, 8, 16, 32}` names the granularity;
the stride is part of the fold label, so two strides are two experiments and
`tests/divergence-columns.sh` refuses to mix them. The default is 32 and is a *default*, not a
constant, for exactly this reason — and `Folds`'s `Default` is hand-written because a derived one
would give stride 0, a `-line` arm that reads NOTHING whose clean result would read as
"sampling suffices".

### Phase 3B — copy by kernel — is built and is the other live branch

`--copy-by-kernel` moves bounce→slot bytes with an ordinary shader copy instead of
`hipMemcpyAsync`. The matrix demands a device-side read of the arena's own bytes in bulk, and a
shader copy is exactly that — reading through the normal vector-memory path that a copy engine may
bypass. **If it comes back clean it is both the conviction and a fix at roughly equal bandwidth.**

On the same footing as `--pinned-coherent`: a flag and not a feature, so the protocol compares two
release binaries differing in one argument. And it **counts what it actually did** —
`copies=[memcpy N / kernel M]` — because an intervention that never applied and one that does not
work produce the same red, which has already cost this investigation two rounds. That read-back is
the standing lesson and applies to every candidate fix from here.

## PHASE 1 RESULT: `bh` is the suppressor — the defect is in the BOUNCE ARENA

All 1536-token pairs, quiet box, binaries built before any arm:

| folds | tok/s | reads the arena? | outcome |
|---|---|---|---|
| v2 light | 1.61 / 1.84 | no | DIVERGED @169 |
| `se` | 2.16 / 2.04 | no | DIVERGED @301 |
| `sc` | 2.46 / 2.58 | no | DIVERGED @236 |
| `sc-nop` | 2.57 / 2.59 | no | DIVERGED @292 |
| `xa,ac` | 2.32 / 2.41 | no | DIVERGED @292 |
| **`bh`** | **2.19 / 2.39** | **YES** | **CLEAN over 1536** |
| `bh,sc,se` | 1.25 / 1.70 | YES | CLEAN (512 and 1536) |

**Every cell that does not touch the bounce arena diverges; both cells that do are clean.** And it
is not a slowdown artifact — `bh` runs at 2.19–2.39 tok/s and suppresses while the light probe runs
*slower* at 1.61–1.84 and diverges, so duration is ruled out by the data rather than by argument.
`sc` — the post-copy slot read that the visibility-at-the-copy story predicted would be the
suppressor — **does not suppress**. That prediction was wrong.

`ac` also resolved coordinate 2, in the opposite direction from the guess attached to it: at pos 287
L18 `xa` was identical (closing the rescaled-residual alternative `xa` was added for) with `h`, `ac`
and `x` all differing. So gate/up read wrong there, versus down at coordinate 2 — **not
sub-region-specific: whichever part of the slot is read while stale.**

## The missing ordering guarantee, named

The chain, with the code that assumes each step:

1. **`rivoli_pinned_alloc`** (`kernels/async.hip`) allocates the arena
   `hipHostMalloc(..., hipHostMallocDefault)` — **no coherence property requested**.
2. io_uring O_DIRECT has the **NVMe device DMA** into that arena. The CQE establishes that those
   writes are visible **to the CPU**: the kernel's completion path carries the barriers for that,
   and btrfs's own `datasum` verification then reads them successfully on the CPU, which is why
   storage is clean.
3. **`Streamer::reap`** (`fetch/stream.rs`) consumes the CQE and enqueues
   `hipMemcpyAsync(H2D)` **back to back, with nothing between them.**
4. That copy's reader is the **GPU's copy path**, not the CPU.

**Nothing in that chain establishes that the agent in (4) observes the writes from (2).** The CPU is
neither the producer nor the consumer, so the CQE's guarantee — the only one present — is about the
wrong agent. For **fine-grained** host memory the platform provides the missing guarantee by
snooping; for **coarse-grained** it does not, and an acquire is required at a synchronisation point.
`hipHostMallocDefault` does not state which it yields, and on ROCm that has been runtime- and
environment-dependent, so **this is not asserted here — it is A/B-ed** (`--pinned-coherent`).

A stale line is exactly what the reuse pattern arranges: a staging slot's *previous* contents were
themselves read through the GPU on an earlier fetch, so the GPU may hold a cached copy of that host
range when the NVMe DMA overwrites it. The corrupted payload would then be a **previous expert's
weights** — finite, plausible, silently wrong, which is the recorded symptom class exactly.

### Why a CPU-side fence is NOT the fix

It was proposed as candidate (a) and as potentially shippable. **It cannot work**, and the reason is
the same one that makes the gap real: an acquire/`sfence` on the reaper thread orders the *CPU's own*
accesses. The producer is a third-party device's DMA and the consumer is the GPU's copy engine —
**neither is the CPU**, and the CPU's view was already correct (btrfs verified it). A CPU fence adds
no ordering between those two agents. If it appeared to help it would be by perturbing timing, which
is the class of accidental fix this investigation exists to reject.

### The candidates, and which are architectural

| candidate | flag | architectural? |
|---|---|---|
| **fine-grained arena** — `hipHostMallocCoherent` | `--pinned-coherent` | **YES.** Changes the memory TYPE, so the platform provides the guarantee. Expect a bandwidth cost on the bounce→GTT copy: fine-grained host memory is not GPU-cached. **Measure it.** |
| **a bare kernel launch before the copy** | `--divergence-folds bh-nop` | **Probably** — an HSA dispatch performs a system-scope acquire, which is a documented dispatch property rather than a side effect of reading. But relying on the *scope* of that acquire is a weaker contract than owning the memory type |
| one cache line of the arena | `--divergence-folds bh-line` | **NO — mitigation.** Works, if it works, by touching memory. Label it as such |
| a CPU fence | — | **NO — not a fix at all**, see above |

**`bh-nop` is the decisive cheap cell and it should run next.** It has the launch and its acquire and
reads nothing of the arena. If it suppresses, what repairs the hazard is the *dispatch*, not the
bytes — which makes the coherent allocation (or an explicit cache operation) the fix and any read
incidental. If `bh-nop` fires while `bh` suppresses, it really is about reading those bytes, and the
memory-type story needs revisiting.

Note the ladder now exists at **both** positions (`bh`/`bh-nop`/`bh-decoy`/`bh-line` and the `sc`
set) from one enum, because it is the same question asked at two points.

## The control, and a SECOND coordinate with the opposite signature

**Control, same box and day, same pinned prompt (md5 `18927a780b36b029d03450d2100e9242`), all
binaries built before any arm:** the unprobed pair diverged at body line **66** of 512, and the
**v2 light-probe pair diverged at body line 169**. So the unmodified engine still diverges today
and **the light probe does not suppress** — which isolates the suppressor to the v3 *hop* folds
specifically and gives every ablation below a live baseline.

The light pair's first differing row:

```
row 11814   pos=186  nrow=1  layer=35   misses=3  relocs=1
  A   xn bd7368d84ea4b46f   h b7d6b00bf231b134   x 568ecb4fe448726f
  B   xn bd7368d84ea4b46f   h b7d6b00bf231b134   x 327c32c8d8c34960
      gl b545f4899bbd9bde   pk c2d211a6885ec420   sl 27e82e49106795e3   (identical)
```

**Only `x` moved; `h` is IDENTICAL** — the opposite of the token-164 event, where `h` differed and
`x` followed.

### What that means, and it is sharper than "`h` matching proves nothing"

The reading offered with this result was that `h` is folded at one instant while the kernel reads
the slot at another, so a corruption landing in between leaves `h` identical — and therefore that
"`h` differing proves wrong bytes; `h` matching proves nothing". **The first half is right and the
second half is too weak, because `h` is not a fold of the bytes.**

`h` is `moe_hidden` — the *output* of the gate/up pass, folded after both MoE lanes are awaited. It
is a **consumer-output** quantity: a function of what the kernel actually consumed, computed by the
consumer itself. The gate/up kernel is deterministic given `(xn, gate/up weights)`, and `xn` is
identical here. So `h` identical **does** prove the gate and up weights were read identically. The
"fold at a different instant" caveat applies to `bh`/`sc`/`se`, which fold BYTES at a chosen moment;
it does not apply to `h`, `x`, `ac` or `xn`.

That distinction is now in the log's own header, in `tests/divergence-columns.sh`'s key, and at the
flag, because getting it wrong in either direction misleads:

- a **bytes-at-an-instant** column agreeing (`bh`, `sc`, `se`) → proves only that the bytes matched
  when it looked. **No null on those exonerates a hop.**
- a **consumer-output** column agreeing (`xa`, `xn`, `h`, `ac`, `x`) → the kernel is deterministic,
  so equal output over equal other-inputs means the bytes it consumed were equal.

So at this coordinate: gate/up read the right bytes, and everything between `h` and `x` is
suspect — the **down projection**, the fixed-point accumulator, or the drain. The accumulator is
refuted at the kernel (per-term saturating fixed point, wrapping u64 `atomicAdd`, lane split exactly
equivalent to one accumulator) and the drain is deterministic given `moe_acc`. **The surviving
reading is that the DOWN projection read different bytes** — the same slot as gate/up, a different
sub-region (`down_indices`/`down_scales`), a *separate kernel launch*, and therefore a different
instant. One mechanism, two signatures, depending on which part of the slot was stale when it was
read.

**Two gaps this exposed, both now closed rather than argued about.**

1. `xn` is a **norm** of the residual, and rmsnorm is scale-invariant — so `xn` agreeing does not
   strictly rule out a rescaled residual, which would leave `xn` identical and `x` different. That
   is the alternative explanation for this exact coordinate, and it is a real (if implausible) hole.
   **`--divergence-folds xa`** folds the residual *before* the norm and closes it.
2. `h` and `x` cannot separate "the down projection read wrong bytes" from "the drain".
   **`--divergence-folds ac`** folds the fixed-point accumulator after both lanes are awaited and
   *before* the drain — the consumer-output witness for pass 2, exactly as `h` is for pass 1. With
   it, this coordinate resolves in one run.

Both are cheap (6,144 and ~24,576 elements against `h`'s ~98,000) and both are **opt-in**, so the
light probe stays byte-identical to the configuration just proven not to suppress.

### Relocation carries no signal at all

`relocs > 0` occurs in **14,338 of 42,666 rows (33.6%)** of an ordinary run. The token-164 event
fired at `relocs = 0` and this one at `relocs = 1`. Neither value carries weight, and the column
stays only so that a future coordinate can be checked against it rather than assumed clean.

## Hop 1 is largely exonerated, by a check that was already running

The owner ran the kernel-log audit: **no MCE, no KFD resets, no amdgpu/SDMA faults, no NVMe AERs,
no btrfs checksum errors**, across every window. On its own a clean log is weak evidence. Here it is
not, and the reason is the filesystem: the artifact sits on **plain btrfs with `datasum` ON** (no
`C` attribute on the slabs or their directory), and btrfs verifies checksums on **direct-IO** reads
after the data lands. The engine reads io_uring O_DIRECT into a `hipHostMalloc` pinned arena, so
**every one of the ~70,000–95,000 expert reads per arm is already a storage-integrity test**, and it
reports clean.

Bad bytes on the drive, or corruption in the NVMe→host DMA, would have EIO'd and logged. **They did
not.** What that does NOT cover is anything after the verification: the arena sitting in host memory,
and the `hipMemcpyAsync` from it into GTT.

Two consequences:

- **Phase 4's userspace storage repro drops sharply in value** — btrfs is already running that
  experiment continuously, at scale, and it is clean.
- **`bh` is not thereby redundant.** btrfs verifies at read completion; `bh` folds on the fetch
  stream some time later, so `bh` differing would mean the arena changed *after* btrfs blessed it.
  That is a real and otherwise-unwatched window, and it is why F1 keeps its place — with the
  asymmetry above attached, since `bh` agreeing still cannot acquit.

The surviving suspects are the **bounce→GTT copy** and the **consumer reading before that copy is
coherently visible** — neither of which produces a kernel log entry, so a clean audit is exactly what
they predict. That promotes F2 and Phase 2 to the main line, and pulls Phase 3's copy-by-kernel and
blocking-`hipMemcpy` ablations forward.

## THE HEAVY PROBE SUPPRESSES THE DEFECT — and that is itself the strongest clue yet

Two instrumented pairs on a quiet box, all three fetch-path folds on, **2,048 tokens and ZERO
events**: 512 tokens (identical ids, identical hit/miss 236,570/70,030, all 42,666 log rows equal)
and 1,536 tokens (identical ids, hit/miss 714,436/206,564, all 122,538 rows equal).

**The probability, re-derived here rather than quoted.** A pair diverges if EITHER arm has an event,
so the exposure is 2 × 2,048 = 4,096 arm-tokens, and P(zero) = exp(−rate × exposure):

| rate | P(zero), exposure counted once per pair | P(zero), per ARM — the correct reading |
|---|---:|---:|
| matched, 1 per 299 | 0.106% | **0.00011%** |
| conservative, 1 per 578 | 2.89% | **0.084%** |

The figures first reported for this result — 0.11% / 2.89% — are the middle column, i.e. the
single-arm reading, which is conservative in the right direction; the correct per-arm numbers are
smaller still and the conclusion is stronger than it was stated.

**But it is not airtight, and this is the honest bound.** The rate itself has a wide interval
(exact 95% Poisson, 1 per [206, 2804] on the conservative reading). At its **pessimistic end**,
P(zero over 4,096 arm-tokens) is **23%** — so a clean 2,048-token result is suggestive rather than
conclusive on its own. Each Phase 1 cell adds another sample against the same rate, which is the
cheapest way to close that gap; a single further clean heavy pair takes it well under 5%.

The variable is the probe's **weight**, not its presence: the light probe — the per-layer columns
with no fetch-path folds — is what produced the token-164 coordinate above. The heavy probe's
distinguishing ingredient is **a read of the pool slot immediately after the copy**.

**Working hypothesis: a device-side VISIBILITY problem on the DMA.** The consumer occasionally reads
a slot whose `hipMemcpyAsync` has been signalled but whose bytes are not yet coherently visible to
it, and any extra read at that point forces the ordering that makes them visible. Every other
observation fits: host bookkeeping is correct, `READ-BEFORE-WRITE` never fires, the tickets are in
place, `relocs=0`, and routing and slot identity are identical while only the payload differs.

**This also indicts the instrument I built.** The all-folds-on configuration is the `--checksum-x`
failure reproduced in new code — a probe that dilates its subject until the subject stops
misbehaving. It is why `--divergence-folds` now defaults to NONE and enables one fold at a time, and
why the log records which folds produced it and `tests/divergence-columns.sh` refuses to compare two
configurations.

## `--mode int4` diverges too: not a vq3 problem, and the rate is per READ

An unprobed 512-token pair in `--mode int4` diverged at body line **28**. Different stride
(19.125 MiB against 14.625), different kernel (`launch_moe_expert_range_i4`), no codebooks — the
same defect. Every arm before this compared int3-vq against itself, so **"vq3-specific" is retired**.

It also bears on the rate model, and the claim has to be made at the strength the evidence supports.
int4 took **95,351 misses** against int3-vq's 70,030 over the same 512 tokens (+36%) and diverged at
token 28 rather than 320–452. That is **one sample per mode**, so it cannot establish a per-read
rate on its own — an exponential with mean 320 produces a first event at 28 about 8% of the time, so
the observation is unsurprising under a per-TOKEN model too. What it does do is remove the reason to
prefer per-token — the mode with more reads failed sooner. It cannot go further than that, and an
earlier draft of this paragraph overreached by saying the modes "differ in reads per token and in
nothing else that should matter" three sentences after listing three differences. They also differ
in BYTES (95,351 × 19.125 MiB against 70,030 × 14.625 MiB is **1.78×**, not the 1.36× of the read
count) and in the kernel. So this single pair cannot separate per-read from per-byte either. **Per-read is now the working model, not a measured
result**, and it is what makes an instrumented run worth its throughput cost and what the storage
repro exploits to turn a 20-minute-per-arm hunt into millions of iterations per hour. Two more int4
pairs would settle it cheaply.

## What the instrument now measures, and the one question left

Three fetch-path folds split the one remaining question — **which hop delivers the wrong bytes** —
and each is now enabled INDEPENDENTLY by `--divergence-folds`, because all three at once suppress
the defect (above). The default is none.

| column | folded where | a difference means |
|---|---|---|
| `bh` | the pinned bounce arena, on the fetch stream, the moment the NVMe read completes and **before** the copy | the DRIVE delivered different bytes |
| `sc` | the pool slot, on the fetch stream, immediately **after** the copy | with `bh` equal: the COPY |
| `se` | ALL the layer's slots, at end of layer, each at its own index offset | with `bh` and `sc` equal: a slot was wrong AT REST and not the one just copied — a resident expert, or a write after the copy landed |

**Read them CROSS-RUN — A's column against B's — and never against each other.** `sc` folds the one
slot just copied; `se` folds all ~9 the layer used. They are different quantities and would differ
on every row, so a within-run `sc == se` test (which an earlier draft of this section invited) sends
an operator after an innocent hop. With all three equal across runs and `h` differing, the reading
is: every byte the batch used was identical at rest in BOTH runs, so the kernel read one of them
before it landed — a ticket/timeline ordering failure.

All three are XOR folds at **full width**, so none can miss a corruption by sampling. `bh` and `sc`
are stream-ordered around the copy on the fetch stream; `se` is on the null stream at end of layer,
after both MoE lanes have been awaited. **No fold adds a host sync of its own** — but `Probe::drain`
does add one `device_sync` per pass, and that is deliberate and argued at the code: without it the
async null-stream clear of the fold slab could race the reaper's fetch-stream folds. It lands where
`run_layer`'s per-layer sync has already idled the device, so it adds no barrier that was not
already there.

`se` folds every expert the layer used, each at a distinct index offset (`i_base = j·n`), which
matters twice. It is **not** blind to a resident expert corrupted on an earlier token — `bh`/`sc`
see only the ~1.7 of 9 slots that were read, and that gap is the whole reason `se` exists. And it is
not invariant under two payloads being **swapped between slots**: without the offset the fold mixes
only the within-slot index, so a crossed destination — exactly the "wrong bytes in a slot" class
under investigation — would leave every fold agreeing and be misread as something else. That was a
real hole, found by review 2026-08-17.

**Cost, re-derived — and the first version of this paragraph was wrong twice.**

The byte volume is **~14.4 GB/token**: `bh` and `sc` fold one 14.625 MiB expert per cold read at
130.4 reads/token (2.0 GB each), and `se` folds all 9 of the batch's slots on each of the 75 MoE
layers (10.4 GB). An earlier draft said ~6 GB, which was all three at misses-only — `se` had been
widened to the union without the number moving with it, and `--divergence-log`'s cost then appeared
in the INDEX verdict, the place CLAUDE.md tells readers to trust *instead of* the doc.

Against the only effective-bandwidth figure this repo has measured — **~135 GB/s** on this box
(`docs/measurement/baseline-2026-08-16.md`, Glimmer bf16 fully resident) — that is ~106 ms/token, or
**~27%** of the 388 ms/token the same baseline records for GLM int3-vq.

**That estimate is arithmetic, not a measurement, and it only became credible after a bug was
fixed.** The fold kernel originally did one `atomicXor` per element against a SINGLE global u64.
At ~936 folds/token × 3.8 M elements that is 3.6e9 same-address atomics — which serialise, so no
bandwidth figure bounds them at all, and at any plausible rate the instrument would have run ~1–4 s
per token: several times slower than the thing it measures, which is the `--checksum-x` failure with
a different mechanism. `hash_rows` now reduces in shared memory and issues one atomic per block
(3744× fewer), which is what makes it bandwidth-bound and the number above meaningful. The reduction
is exact — XOR is associative and commutative — and `fwd_kernel.rs` still scores the kernel against
`rivoli_core::hash::xor_fold_from` bit for bit.

**Measure it on the first instrumented run** rather than trusting the paragraph above:
`AsyncFetch::io_wait_ns` already reports the reaper's slack, and the run's own tok/s says what the
folds cost.

> **FALSIFIED 2026-08-17, the same day, and left here rather than deleted because being wrong in a
> stated way is the point of stating it.** This paragraph continued: *"The masking risk is bounded
> by the coordinate measurement: the event fired on a QUIET box, and the rate is per READ rather
> than per second, so a run ~27% slower issues the same number of reads and should fire at a similar
> TOKEN position. If it does not fire, that is itself informative — it would mean the instrument
> perturbs."*
>
> **It did not fire.** 2,048 instrumented tokens, zero events — the section above. So the masking
> risk was NOT bounded, the prediction was wrong, and the "if it does not fire" clause is the one
> part that held: the instrument perturbs. Two further corrections travel with it — "the rate is per
> READ" is now a working model rather than a fact (below), and a ~27% cost estimate cannot bound a
> masking risk in the first place, because what suppresses is evidently the *shape* of the added
> work and not its duration. That is exactly what Phase 2's ladder was built to separate.

### `crates/engine/examples/arena_repro.rs` — the hop at rate, and the AMD exhibit

```
flock /var/run/sys-gpu.lock -c \
  '<target>/release/examples/arena_repro /var/db/rivoli/glm52-vq3-full/L03.vq3 1000000'
# ...and the same with --coherent, which is the fix arm
```

The engine reproduces the defect at roughly one event per 40,000–80,000 reads — a 20-minute arm per
sample. This drives **the engine's own `Streamer`** (same ring, same pinned arena, same
`hipMemcpyAsync`, not a re-implementation) in a hot loop and verifies every read, turning the same
statistics into minutes. If it fires it is the minimal driver-level exhibit: no model, no MoE, no
scheduling.

Three properties make it faithful, and each is a way it could have been useless:

1. **Staging slots are reused round-robin** — the condition the hypothesis needs, since a slot's
   previous contents were themselves read through the GPU.
2. **Consecutive uses of a slot carry DIFFERENT payloads** (17 block offsets against 16 slots,
   coprime). Refill a slot with the same bytes and a stale line returns the right answer for the
   wrong reason.
3. **It never reads the arena.** Verification folds the DESTINATION after the copy — the `sc`
   position, measured not to suppress. A repro that verified at the arena would be a repro of
   nothing, because reading the arena is precisely what makes the defect vanish.

The reference folds come from **buffered** reads of the same ranges, so a mismatch means the
O_DIRECT + copy path delivered something the ordinary read path does not. A clean run prints its own
rule-of-three bound, so a null is a number rather than a shrug.

**It is NOT deviceless**, contrary to the plan that called for it: `hipHostMalloc` initialises the
HIP runtime and both the copy and the verifying fold are device work, so it holds a KFD entry and
appears to every witness. **Take the flock.** What it avoids is the 281 GB load and the decode, so it
is cheap to run *between* GPU cells — not free of them.

### Phase 1's matrix, and the asymmetry it must be read under

One fold at a time, 1536-token pairs, against the live control above:

| cell | `--divergence-folds` | fires ⇒ | goes clean ⇒ |
|---|---|---|---|
| F1 | `bh` | if `bh` DIFFERS, the arena changed after btrfs blessed it | **not an acquittal of storage** — see the asymmetry |
| F2 | `sc` | the post-copy read is not the mask | `sc` is the mask; the hazard is at the copy/visibility boundary |
| F3 | `se` | the control behaved like the unprobed engine, as expected | the hypothesis is wrong — `se` runs *after* the consumer and should not be able to suppress anything |

**Whichever fold turns RED→GREEN is the mask, and its position names the mechanism.** But a cell
that fires convicts nothing on its own either: `bh`/`sc`/`se` are bytes-at-an-instant folds, so
their nulls are not acquittals, and only their DIFFERENCES carry a verdict.

### Phase 2: WHAT about the post-copy read repairs the hazard?

Four alternatives occupy the same pipeline position, so exactly one may be selected. Each rung
removes one ingredient, read against `sc`, the known suppressor:

| `--divergence-folds` | launch | duration & bandwidth | touches the slot |
|---|---|---|---|
| `sc-nop` | yes | ~0 | no |
| `sc-decoy` | yes | same as `sc` | **no** |
| `sc-line` | yes | ~1/32 of `sc` | yes, every cache line |
| `sc` | yes | full | yes |

**Run `sc-nop` FIRST.** If a bare kernel launch at that position suppresses, the mechanism is the
stream boundary's own cache maintenance and every other arm is confounded — the folds are enqueued
*before* the timeline signal the consuming kernel waits on, so any launch there also delays that
signal. Every outcome, including the two that are not a clean win:

| result | reading |
|---|---|
| `sc-nop` suppresses | the LAUNCH BOUNDARY, not duration and not this slot. Go to Phase 3; the other arms add nothing |
| `sc-nop` fires, `sc-decoy` suppresses | duration or bandwidth — a **timing** hazard. The fix is a delay or a stream-ordering change, not a read |
| `sc-nop` and `sc-decoy` fire, `sc-line` suppresses | **touching the bytes** repairs them: a cache/visibility effect, and `sc-line` is then a candidate fix at ~1/32 of the reads |
| everything below `sc` fires | what `sc` does that none of them do is the full-width read of *this* slot. Nothing cheaper serves as a fix; go to Phase 3 |
| nothing fires, `sc` included | the suppression did not reproduce. The rate's interval permits it (23% at the pessimistic end) — re-run the control before reading anything into the ladder |

**`sc-decoy` replaced an equal-duration spin kernel, and why is worth recording.** The first version
burned the same trip count with no memory access, claiming equal duration "by construction". Review
refuted it from this tree's own comment: `hash_rows` is *bandwidth-bound*, so removing 100% of the
bandwidth and keeping the ALU makes the arm far SHORTER — and an arm that then failed to suppress
would have been read as "the hazard is not time" when it never delivered the time. Folding a decoy
buffer of the same size holds duration and bandwidth constant while still never touching the slot.
The spin kernel is deleted; `sc-nop` is a one-element fold of that same decoy, so there is no second
kernel to drift.

**`sc-line` sweeps one element per 128 B cache line across the WHOLE slot**, not the first line —
covering the slot is what lets it be a candidate fix at all. 128 B is measured in this tree
(`kernels/linalg.hip`, `kernels/mla.hip`); an earlier constant said 64 B with no citation, which
would have swept every other line. Its column renders `~<hash>` because it sees ~1/32 of the bytes.

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
