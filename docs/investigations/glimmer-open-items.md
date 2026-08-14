---
scope: glimmer
status: live
verdict: Everything the Muse Glimmer port owes after S3's layer loop landed 2026-08-13, in one list, because the obligations had spread across four commit messages, two verdict lines and three source comments. THE SHORT VERSION -- five open items, all of them stage work (S4, S5 and the rest of G3); six of the eleven were closed the day this file was written, and two of those six had been recorded as open on evidence that was already stale. The three live CLASSES this file opened with -- ungated correctness, fixture blind spots, unpriced cost -- are all empty. (1) UNGATED CORRECTNESS: EMPTY as of 2026-08-13. The eps transposition -- the item this register called its sharpest -- turned out to be gated already, against the chain gate's tolerance of the day, closed by the softcap tolerance work one commit earlier; the 'nothing reddens' recorded here was a measurement taken before that change and never re-run, which is the same stale-fact defect as the wrong citation it replaced. A second, LOCALISING gate was added anyway (Glimmer::branch plus a per-layer score against the oracle), and the route this file predicted -- scoring the reference's own captures -- is REFUTED by measurement: there the eps signal is 1.6e-3 to 1.3e-2 against a bf16 weight floor of 4.7e-3 to 3.0e-2, i.e. 0.2x to 0.6x the noise at every layer of both salts. Both surviving margins were thin because the FIXTURE's branch sits at mean(x2) ~ O(1) where the two epsilons nearly agree -- which is a second reason to do (2), and (2) then re-derived both tolerances so the old figures no longer describe anything (the prose counts are deleted rather than corrected a third time; tests/glimmer_chain.rs's TOL and TOL_BRANCH carry their own measured separations). (2) FIXTURE-GEOMETRY BLIND SPOTS: CLOSED 2026-08-13 by widening the toy checkpoint from (2 heads, 1 kv, head_dim = dim) to (4, 2, dim/2). Both were UNCONSTRUCTIBLE rather than merely uncaught -- at head_dim = hidden the two candidate softmax scales are the same NUMBER, and at hkv = 1 the block and modulo KV broadcasts are the same FUNCTION, so no tolerance could have separated either. They now measure 3.7e-1 and 9.5e-1. Six test binaries share the fixture and all pass unchanged; the widening also improved every clean floor in the chain gate by 2.7-5.5x, so its three tolerances and its whole 13-row mutation census were re-derived rather than carried over. (3) UNPRICED COST: the KV cache item is CLOSED 2026-08-13 -- glimmer_gpu::runtime_bytes is subtracted from the budget BEFORE the tier is sized, a budget that cannot cover it is refused by name, the slot count is one function read by both the allocation and the accounting, and the residency line moved below the tokenizer because the footprint is a function of the context; gated deviceless at the shipped widths plus a partition-boundary assertion, with one gap stated rather than papered over (removing the subtraction leaves every test green, since its effect needs a genuinely near-full device). Prefill is CLOSED too: Glimmer::prefill is layer-major and chunked at 256, so GlimmerPin::layer is called once per layer per chunk instead of once per token -- measured 6 fills against 900 token-major on the fixture, red-proved by reverting the order. It is a pure reorder (every numeric gate was bit-for-bit unchanged, which is the check), so no output comparison can see it and Glimmer::slot_stats exposes the pin's fill counter for the assertion. Chunked rather than whole-prompt because the batch's residual streams stay live: whole-prompt is 3.49 GB at the 131072 ceiling and 3.1's own gate caught it at 7.144 GB, where chunking costs 6.8 MB for 99.6% of the saving. The MATH is still m=1 / tq=1; batching it needs a rows dimension on the centered norm and a per-row rope position, changes the arithmetic, and first makes the launcher's ring union hazard reachable -- separate work. (4) STAGE WORK, six, mostly S4 and S5. G3's comparison against MUSE GLIMMER rather than against rivoli's own transcription is DONE 2026-08-13 (tests/glimmer_reference.rs): the anchor driver now exports all 107 parameter tensors, both goldens regenerate BYTE-IDENTICAL so the change is provably additive, and rivoli's loop reproduces the reference's own logits on the reference's own weights over 7 steps x 2 salts, worst 4.8e-2, all 14 emitted tokens exact, red-proved at 1.3e0 / 1.1e0 / 6.7e-1. It carries a bf16 weight-rounding term MEASURED at 9.3e-3 to 6.8e-2 (rivoli stores projections bf16, the reference computed f32), which sets its tolerance and costs it resolution -- so it and the finer chain gate are complements, not substitutes. Its load-bearing by-product is that an independent f64 transcription of sections 3-5 reproduces the reference to 3.8e-6, the first evidence here that the architecture doc is right about the WHOLE chain. AND IT DID NOT CLOSE THE EPS ITEM, which this file had expected it to: the transposition is green on the reference's own weights too, so it is not a fixture artefact. S4 is now PART-CLOSED (2026-08-14): the 59.553 GB checkpoint is downloaded, structurally verified and converted bf16-verbatim to a 55.71 GB artifact (418 verbatim + 209 norms widened + 809 vision skipped, reconciling to the index's 1436); the chat template is hand-ported as artifact::glimmer_encoding and pinned byte-for-byte against 24 cases rendered by the checkpoint's OWN apply_chat_template, with 11 red proofs; and tie_word_embeddings: false is gated on ADDRESSES (the two [vocab, hidden] tensors are 2,689,662,976 B each, 1.252x i32::MAX, contents distinct), red-proved by aliasing the head onto the embedding. Item 3 is CLOSED too (2026-08-14): three greedy runs on the bf16 artifact, sole tenant with a KFD/GTT witness, all 52 layers pinned at 55.712 GB -- 1.486 tok/s at 96 tokens, 2.261 at 500 (warm-up amortizing), and 1.804 on a run that TERMINATED NATURALLY at 110 of a 400 limit, which is what GLM did in none of 56 runs before its framing was fixed. The text is coherent and it is the reasoning channel: every run completes the generation prompt with ' to=self<|message|>' as Reasoning strength: high asks, and the 500-token run writes three correct sentences and then checks them. One finding is for the SERVING layer, not S4: the run that stopped stopped inside the reasoning channel, because <|eom|> (200007) is deliberately not a stop id -- so a bounded CLI decode yields reasoning and never the answer, and there is no way to lower the reasoning strength from the CLI. What S4 still owes is the dNLL ladder per format and the first quantized format, and the ladder cannot start before the format exists. Register items 4.2, 4.3 and 4.4 are all CLOSED 2026-08-14. 4.3: a 7661-token prompt with a needle at token ~53 -- 7608 tokens back, outside every sliding window, so only the 13 full-attention layers can carry it across a ring that wrapped 3.7x -- and the model reproduced TANGERINE-4417 EXACTLY, coherently. A behavioural probe at shipped width, not a new numeric gate (that exists at sliding_window=2). It also exposed the real cost of an unbatched prefill: 0.416 s/token at 1201 and 0.561 at 7661, i.e. 75-101% of a decode step, because section 3.2 batched the FETCH and not the arithmetic -- so a 7661-token prompt is 72 MINUTES before the first output token, which makes batching the prefill math the highest-value S5 item by a wide margin. 4.4: sixteen bare null_mut() in stream position (not the 'about fourteen' this file estimated), all converted, with a test that lists the three genuine pointer survivors by reason. 4.2: CLOSED BY REFUTATION plus a price -- the anchor's logits max at 0.24 where 20*tanh(x/20) is arithmetically the identity (0.0002%), so disabling the softcap moves total variation from 1.249e-3 to 1.249e-3 and NO tolerance could ever see it; priced instead on the real model, where it cuts max|logit| 36.27 -> 18.96 and the top1-top2 gap 33.46 -> 18.05, leaves total variation at 1.92e-8 (both saturate), and moves the RUNNER-UP token by 15.4 nats -- so the instrument for it is teacher-forced NLL on non-argmax tokens, i.e. item 4's dNLL ladder, which needs a Glimmer --ppl path that does not exist. S5 remains gated on G4, but the STREAMING curve is now measured on bf16 and it is the first evidence Glimmer streams at all: 1.804 tok/s all-resident against 0.638 / 0.453 / 0.355 at --max-mem 45 / 30 / 15 (9 / 26 / 42 layers streaming), so the pin is worth 5.1x end to end. Marginal streaming bandwidth RISES with the streamed count -- 8.6 -> 15.2 -> 18.0 GB/s -- so the per-fill fixed cost dominates small streamed sets and amortizes, which is exactly what S5's prefetch item is about, measured before the lever exists. The first version of that table was NON-MONOTONIC and it was PAGE CACHE, not a finding: the first arm paid the cold NFS read for the whole 55.712 GB file, 82.81 s cold against 37.90 and 37.64 s warm -- a 2.2x confound, bigger than most effects being measured, reproducing to 0.8% once warm. OPEN AND NEW 2026-08-14 (section 4b): a SIGSEGV inside libamdhip64's VMM path, ~3 in 52 runs, and THREE CAPTURED CORES say two of the three are on ALLOCATION (rivoli_vmm_alloc under VmmBuf::new -> DeviceTier::new -> GlimmerPin::build) rather than teardown. Read with coredumpctl+gdb, which works where a live attach does not (ptrace_scope=1). A teardown-join fix was attempted and REFUTED by its own measurement -- 2-in-27 became 1-in-25, indistinguishable, and the next crash was an alloc one -- so it is kept for the real hazard it closes and explicitly NOT as this bug's fix. Not contention, not the probability-space metrics, and not test order (that claim was withdrawn the same hour). NOT Glimmer-specific either: VmmBuf is every model's tier allocator, and glimmer_reference only surfaces it by building ~14 engines per run -- not GPU contention (the crashing run's KFD/GTT witness is identical to three passing ones), not the probability-space additions (host arithmetic, no device call). A first attempt to pin the trigger on test ORDER was WITHDRAWN the same hour: 1-in-5 against 0-in-10 looked like a discriminator until the same configuration returned 0-in-10, so the two arms are indistinguishable at that sample size. It matters because this is the gate that scores rivoli against Muse Glimmer's own logits, and a suite that fails a few runs in a hundred invites re-running until green. Also carried here: the branch is 50+ commits unpushed, the checkpoint lives at /swarm/storage/ai/rivoli because /swarm/storage/ai/models is root-owned, and the GPU node label is left at disabled with llama-swap Pending.
---

# Muse Glimmer — open items after S3

**Why this file exists.** S3's layer loop landed on 2026-08-13 and its four review rounds
produced more open items than they closed. Those items were recorded where they were found —
in commit messages, in two `verdict:` lines, and in source comments — which is the shape this
repo's own history says rots: *a written-down gap is not an enforced one*, and a gap written
down in five places is one nobody can count. This is the count.

`glimmer-integration.md` remains the plan. This is its outstanding-work register, and every
row here should either move into that plan's stage sections or be closed and deleted.

---

## 1. Ungated correctness

### 1.1 The eps assignment — **CLOSED 2026-08-13**

`Glimmer::pre_norm` reads `eps_pre` (1e-5) and `Glimmer::branch_add` reads `eps_post` (1e-8),
assigned by position. Transposing them now reddens **two** tests in `glimmer_chain.rs`.

> **AND IT WAS ALREADY CLOSED WHEN THIS FILE CALLED IT THE SHARPEST OPEN ITEM.** The
> transposition reddens `the_loop_matches_a_host_reference_at_every_position` at **3.673e-5**
> against its 2e-5 bound — and what closed it was the *softcap* tolerance work one commit
> earlier, which tightened that bound from 1e-4 and made the comparison per-position. The
> "nothing in the tree reddens" recorded here came from a measurement taken **before** that
> change and never re-run. A stale measurement carried forward as a fact; the same defect class
> as the wrong `glimmer_head.rs` citation it replaced.

**What was added anyway, and why it still earns its place.** `Glimmer::branch()` and
`every_layers_branch_matches_the_oracle_and_that_is_where_the_eps_lives` score each layer's
post-FFN branch against the host oracle, selecting a layer by truncating the config. It
LOCALISES the defect to a layer and is a second, independent catch.

**Both margins were thin, and §2's widening is what fixed that.** The cause was the FIXTURE:
its branch sat at `mean(x²)` ~ O(1), where 1e-5 and 1e-8 are nearly the same number, against
the reference's ~1e-3 where `glimmer_norm.rs` measures 41.8-56.6x.

> **The margins are not restated here, deliberately** (2026-08-14). This paragraph carried
> "1.8x and 1.6x" past the widening that re-derived both tolerances from scratch — so did the
> `verdict:` and the INDEX row, which `CLAUDE.md` tells readers to trust INSTEAD of opening the
> doc. A prose number nothing checks has now been wrong in this file twice. `TOL` and
> `TOL_BRANCH` in `tests/glimmer_chain.rs` carry their own measured separations in their doc
> blocks, and those are re-derived whenever the fixture moves; read them there.

> **The route this file predicted — scoring the reference's own captures — is REFUTED, measured.**
> `tests/glimmer_reference.rs` reads each layer's branch against Muse Glimmer's capture, and there
> the transposition is **1.6e-3 – 1.3e-2** against a bf16 weight floor of **4.7e-3 – 3.0e-2**: the
> signal is 0.2x to 0.6x the noise at every layer of both salts. rivoli stores bf16 and the
> reference computed f32, and that rounding lands on the same tensor at the same magnitude. The
> comparison that works is the one with no weight term — engine against oracle, both reading the
> same artifact.

---

## 2. Fixture-geometry blind spots — **BOTH CLOSED 2026-08-13**

`tests/common/mod.rs::glimmer_fixture` went from `(heads, kv_heads, head_dim) = (2, 1, dim)` to
**`(4, 2, dim / 2)`**, and the geometry now carries three inequalities with the reason for each
written at the line. Six test binaries share that fixture; all pass unchanged.

| was | now measured |
|---|---|
| `attn_scale` from `hidden` instead of `head_dim` — an exact NO-OP, `head_dim == hidden` | **3.7e-1** |
| the KV broadcast as `head % hkv` — UNCONSTRUCTIBLE, one KV head made every mapping identical | **9.5e-1** |

**Unconstructible is a stronger word than uncaught, and it is the right one.** No tolerance and
no additional gate could have found either: at `head_dim = hidden` the two scales are the same
number, and at `hkv = 1` the two broadcasts are the same function. 4 over 2 is the smallest
geometry that separates `[0,0,1,1]` from `[0,1,0,1]`.

**It also widened the eps margins, which was the second reason to do it** (§1.1). The branch
comparison went from 4.8x separation to **11.6x**, and every clean floor in `glimmer_chain.rs`
improved 2.7-5.5x — so all three of that file's tolerances were re-derived from scratch rather
than carried over, along with its whole mutation census.

---

## 3. Unpriced cost

### 3.1 The KV cache is allocated outside the budget — **CLOSED 2026-08-13**

`glimmer_gpu::runtime_bytes(gt, n_ctx)` computes the KV cache and activation scratch, and
`Glimmer::new` subtracts it from the budget **before** `GlimmerPin::build` sizes the tier — which
is the point that matters, since the tier is what `guard_capacity` checks against free memory and
everything after it was an unguarded `hipMalloc`. A budget that cannot cover the footprint is
refused by name, with both numbers and what to change.

The slot count is now ONE function (`slots_of`, derived from `window_of`) read by both the
allocation and the accounting — jscpd reported the second copy the moment it was written, which
is the gate reaching the same conclusion.

**The `residency:` line moved below the tokenizer, and it had to.** The footprint is a function
of the context, and the context is a function of the prompt, so before the prompt exists there is
no honest split to print. It still comes after every refusal — that ordering is an earlier
review's and its reason is unchanged. `glimmer_flags.rs`'s `--max-mem` test was re-keyed onto the
budget line, which carries the flag's own VALUE; that is its third marker and the first that does
not name a downstream sentence.

**Gated, and one gap stated rather than papered over.** `runtime_bytes` is asserted deviceless at
the shipped widths (3-5 GiB at the 131072 ceiling; sub-linear past the window, since the sliding
rings cap); the refusal is red-proved; and the PARTITION boundary is gated — charging the
footprint must move the split off all-resident. **Removing the subtraction itself leaves every
test green**, because its effect is only observable on a genuinely near-full device, which no test
can force. It is covered by its two consequences, not directly, and the test says so.

---

### 3.2 Prefill is token-major — **CLOSED 2026-08-13**

`Glimmer::prefill` is layer-major and chunked: every position in a 256-token chunk goes through
layer `l` before any reaches `l+1`. **`GlimmerPin::layer(l)` is called once per layer per chunk
instead of once per layer per token**, and a streamed Glimmer layer is a synchronous 967.942 MB
host memcpy.

Measured on the fixture, 300 tokens with layers streaming: **6 fills, against 900 token-major**.
Red-proved by reverting the loop order.

**It is a reorder and nothing else — the same launches with the same arguments.** Every gate that
compares numbers was bit-for-bit unchanged when it landed (2.705e-7, 7.114e-7, 3.693e-2, 3.984e-2,
4.769e-2, all identical), which is the check that the reorder is a reorder. The corollary is that
**no numeric gate can see this property at all**, so `Glimmer::slot_stats` exposes the pin's fill
counter and `tests/glimmer_loop.rs` asserts against `streamed * (chunks + 1)` — the STREAMED
layers, plus one visit each for the decode step. (Written here and on `slot_stats` as
`n_layers * chunks` until review checked it against the code, 2026-08-14.)

**Chunked at 256 because whole-prompt batching is a memory trade with a flat payoff.** The
residual streams of a batch must all stay live; at the 131072 ceiling that is `n_ctx * hidden * 4`
= **3.49 GB**, on top of the KV cache — the first version did exactly that and §3.1's own gate
caught it at 7.144 GB. Chunking costs 6.8 MB and keeps 99.6% of the fetch saving.

**What is NOT done: the math is still `m = 1` and `tq = 1`.** Batching the projections into one
GEMM per chunk and the attends into one `gqa_attend` needs a rows dimension on
`rmsnorm_centered_single` and a per-row position on `rope_split_half`, and it CHANGES the
arithmetic — so every tolerance in the chain and reference gates would need re-deriving. That also
makes the launcher's `ring_cap >= win + tq - 1` union hazard reachable for the first time, and it
has no numeric gate. A separate piece of work with its own review.

---

## 4. Stage work

### 4.1 G3 — compare against MUSE GLIMMER, not against rivoli's transcription

`tests/glimmer_chain.rs` scores the engine against a host reference *written from
`glimmer-architecture.md`*. That says the loop feeds the kernels correctly. It says nothing
about whether either matches the model, and a defect a single author transcribes into both
sides passes it — the file names three such shared readings.

The anchor goldens hold 1,099 captured intermediates from the real
`MuseGlimmerForConditionalGeneration`, plus `prompt.ids`, `emitted.ids` and the tiny config.
**They hold no parameters**, so the engine cannot be run on them — which is the whole blocker,
and the fix is already specified in the driver's own docstring:

> *"Only `gate_proj` today. Items 4-5 and G3 will want q/k/v/o and the MLP; extend this rather
> than starting a second file, and note that the whole tiny model is ~475k floats, so exporting
> all of it is ~1.9 MB per salt — a decision to make deliberately, not by accretion."*

> **DONE 2026-08-13.** `tests/glimmer_reference.rs`. The driver now exports all 107 parameter
> tensors (2,065,185 B per salt, 18x the old dump), both text goldens regenerate BYTE-IDENTICAL so
> the change is provably additive, and the weight sets' provenance census is structural — it names
> every expected tensor rather than counting them.
>
> **rivoli's loop reproduces Muse Glimmer's own logits on Muse Glimmer's own weights**, 7 steps x 2
> salts, worst **4.8e-2**, with all 14 emitted tokens matching exactly. Red-proved at 1.3e0 (k/v
> swap), 1.1e0 (NoPE inverted) and 6.7e-1 (`qk_scale` dropped).
>
> **Two things this cost, both recorded at the gate.** (a) The comparison carries a bf16
> weight-rounding term — rivoli stores projections as bf16, the reference computed in f32 — MEASURED
> at 9.3e-3 to 6.8e-2 by running the chain in f64 twice, once exact and once bf16-rounded. That sets
> the tolerance at 2e-1 and costs resolution: defects under ~1e-1 belong to `glimmer_chain.rs`,
> which has no weight term. (b) The anchor's random prompt drew the multimodal placeholder ids
> (`image_token_id` 59, `video_token_id` 58), which the wrapper substitutes before the text stack
> ever sees them — so salt 1's raw `prompt.ids` disagree with what was actually embedded at two
> positions, and feeding them to a text-only decode scores 8.7e-1. The effective ids are RECOVERED
> from the reference's own first capture rather than hardcoded.
>
> **The load-bearing by-product:** an independent f64 transcription of §3-§5 reproduces the
> reference to **3.8e-6**. That is the first evidence in this tree that the architecture doc is
> right about the whole chain — and it is what says `glimmer_chain.rs`'s oracle scores against a
> correct reading rather than a shared misreading.

### 4.2 G3 — the probability-space softcap check — **CLOSED 2026-08-14, by REFUTATION plus a price**

The check was built: `total_variation` and per-token NLL now score every reference step in
`glimmer_reference.rs`, measured and bounded like every other tolerance there.

> **THE ANCHOR CANNOT PRICE THE SOFTCAP, and that is measured, not suspected.** Disabling the
> softcap on this fixture moves total variation from 1.249e-3 to 1.249e-3 and dNLL from 5.316e-3
> to 5.311e-3. Not a small signal — NO signal, and no tolerance could ever recover it:
>
> | | max &#124;logit&#124; | what `20·tanh(x/20)` does there |
> |---|---|---|
> | this anchor | **0.24** | changes it by **0.0002%** — arithmetically the identity |
> | the real 30B, trained prompt | **36.27** | compresses to **18.96**, a 48% cut |
>
> `tanh` is linear near zero and the anchor never leaves that region. `glimmer-integration.md`
> S4 item 4 predicted this in as many words ("the anchor provably cannot"); this is the
> measurement behind the prediction.

**So it was priced on the real model instead** — `--prompt "Say hello."`, one step, `cap = 20`
against `cap = 1e30`, everything else identical:

| | capped | uncapped |
|---|---|---|
| max &#124;logit&#124; | 18.96 | 36.27 |
| top1−top2 logit gap | 18.05 | 33.46 |
| argmax | 328 | 328 — **the same**, as S2 proved it must be |
| total variation between the two | **1.92e-8** | — invisible; both saturate at `p(top1) = 1.0` |
| runner-up probability | 1.45e-8 | 2.95e-15 — a factor of **4.9 million** |
| runner-up NLL | 18.05 nats | 33.46 nats — **15.4 nats apart** |
| entropy | 4.05e-7 nats | 1.28e-13 nats — a factor of 3.2 million |

**The softcap is invisible to greedy decode, invisible to total variation on a confident prompt,
and enormous in the tail.** That named the right instrument: teacher-forced NLL on NON-argmax
tokens. It did not exist for this model, so it was built — and it settles the question outright.

> **PRICED, 2026-08-14, on `tests/ppl-corpus.txt` (762 tokens), same artifact, same corpus, the
> softcap the only difference:**
>
> | | PPL | mean dNLL | 95% CI (nats) |
> |---|---:|---:|---|
> | softcap **on** (`cap = 20`) | **7.008490** | — | — |
> | softcap **off** (`cap = 1e30`) | **55.467635** | **+2.06868** | **[+1.80699, +2.33036]** |
>
> **The softcap is worth 2.07 nats per token, and removing it costs 691% of PPL** — while
> leaving greedy output bit-identical, because it is argmax-invariant. The interval is entirely
> on the worse side, so this is a decisive FAIL for the uncapped arm rather than an
> underpowered null. `bin/ppl` notes the SE (0.13351) exceeds its 1%-bar of 0.00995 nats: true,
> and irrelevant here — the effect is **15x the SE**. A 1% question on this corpus would need
> more text; a 691% one does not.

**bf16 PPL 7.008490 is the ladder's first rung**, and every format from here is compared against
it on this corpus with `bin/ppl`'s paired statistics.

**What the two new metrics DO buy**, stated narrowly. TV is a second norm on the same logits in
the space the output lives in (`worst_rel` is relative-max, this is L1 on the distribution), so a
defect concentrated on one high-probability token moves it far more. Bound 5e-3 at ~3.4x the
measured clean value; red-proved. dNLL's resolution is **a ~10% temperature error** — measured:
x1.05 on the engine's logits scores 1.2e-2 and passes, x1.10 scores 2.231e-2 and reddens. Coarse,
because the anchor's 61-way distribution is nearly uniform, and tightening it would leave 1.6x
over clean — the exact thinness that made §1.1's gates red on a correct engine.

### 4.3 G3 — a decode crossing position 2048 — **CLOSED 2026-08-14**

The ring's first eviction at shipped width. `glimmer_chain.rs` wraps the fixture's window four
times, which is the same mechanism at `sliding_window = 2`; this needed S4's weights, which now
exist.

> **A needle probe, because it discriminates where a smoke test does not.** A 7661-token framed
> prompt carrying `the vault access code is TANGERINE-4417` at token ~53 — **7608 tokens before
> the end, so outside every one of the 39 sliding layers' 2048-token windows.** Only the 13
> full-attention layers can carry it, across a ring that wrapped **3.7 times**. A ring that
> mis-evicted would corrupt the residual stream and make a specific string from the far past
> unrecoverable.
>
> **The model reproduced `TANGERINE-4417` exactly**, and the surrounding text is coherent and
> on-task ("The user gives a huge list of maintenance logs, all routine inspection with no…").
> Control at 1201 tokens — inside the window, so the sliding layers can see it too — also
> retrieved it.
>
> **This is a behavioural probe at shipped width, not a new numeric gate**, and it is worth being
> exact about that: the numeric ring gate exists at `sliding_window = 2` in `glimmer_chain.rs`.
> What this adds is that nothing about the mechanism is width-dependent — no 2048-sized block
> assumption, no index that overflows, no eviction that fires early.
>
> **And it exposed the cost of an unbatched prefill**, which is the more actionable finding:
>
> | prompt | n_ctx | total | prefill s/token |
> |---:|---:|---:|---:|
> | 1201 | 1249 | 526.5 s | 0.416 |
> | 7661 | 7709 | 4327.8 s | 0.561 |
>
> Prefill costs **75-101% of a decode step per token**, because the math is still `m = 1`,
> `tq = 1` (§3.2 batched the FETCH, not the arithmetic) and at all-resident there are no fetches
> left to save. A 7661-token prompt is **72 minutes before the first output token**. The growth
> from 0.416 to 0.561 is the 13 full-attention layers being O(n) while the 39 sliding ones cap at
> 2048. **This makes batching the prefill math the highest-value S5 item by a wide margin** — it
> is the difference between long context being usable and not, independent of tok/s.

### 4.4 The `NULL_STREAM` source census — **CLOSED 2026-08-14**

`src/backend.rs` defines `NULL_STREAM` so a deliberate null reads as a decision, and nothing
made the call sites use it.

> **Sixteen, not "about fourteen".** The estimate in this row was prose nobody had counted —
> the same class of defect as the margins in §1.1. Actual: **9 in `gpu.rs`** (four
> `Event::record`, five launcher tails), **6 in `f4gpu.rs`** — which imports `NULL_STREAM` at
> line 109 and spells it correctly six lines from a bare one — and **1 in `attn.rs`**, the guard
> test module whose own doc argues the null is honest. It is, and the constant is what says so.
>
> Every site was classified by READING it. Three occurrences are pointers and stay: `gpu.rs`'s
> indexer pool pointer (absent means "no pool", not "no stream"), `memory/device.rs`'s two
> `hipMalloc` out-parameters, and `fetch/stream.rs`'s io_uring destination array.
>
> **The substitution is value-identical by construction** — `NULL_STREAM` IS
> `std::ptr::null_mut()` — so compilation is the proof and no behaviour can have moved.
>
> `tests/kernel_coverage.rs::a_bare_null_mut_is_never_a_stream` keeps it, listing the three
> survivors with their reasons rather than a count. Red-proved by reintroducing one null.
> **Comment lines are skipped, the OPPOSITE of the jscpd marker rule** — there a comment mention
> was live and two corrections were spent establishing it; here counting prose would make the
> census a function of how often the convention is discussed.
>
> **rustfmt then reflowed two `gpu.rs` calls**, because dropping the `crate::backend::` prefix
> shortened lines enough for them to collapse. The committed file was fmt-clean, so those hunks
> were mine; fixed here, and jscpd re-run after the reflow rather than before.

### 4.5 S4 — real weights — **items 1, 2 and 5 CLOSED 2026-08-14**

The full checkpoint bf16-verbatim, then the first quantized format by dNLL.

> **Item 1 — the checkpoint is converted.** `meta-models/Muse-Glimmer-30B`, 59.553 GB in two
> shards, verified structurally before use (each shard's payload ends exactly at its file size,
> and the index's tensor total plus the two headers is the byte count on disk). It lives at
> `/swarm/storage/ai/rivoli/muse-glimmer-30b` — **NOT** `/swarm/storage/ai/models`, which is
> `root:root` and not writable here. `convert_glimmer` produced a 55.71 GB artifact at
> `/swarm/storage/ai/rivoli/glimmer-30b-bf16`: **418 tensors bf16 verbatim, 209 norms widened to
> f32, 809 vision skipped**, which reconciles exactly — 209 is `52 x 4 + 1`, and 627 text + 809
> vision is the index's 1436. Both EOS ids (`[200001, 200008]`) reached the artifact.
>
> **Item 2 — the chat template is ported and pinned** (`51bb252`). `artifact::glimmer_encoding`,
> gated byte-for-byte against 24 cases rendered by `apply_chat_template` on the checkpoint
> itself, plus the ids, with 11 red proofs. The tools block the plan predicted is there: it is
> the ATEM protocol, ~2 KB of preamble per system turn.
>
> **Item 5 — `tie_word_embeddings: false` is gated.** `the_head_and_the_embedding_are_two_tensors_at_two_addresses`
> asserts the pin places both `[vocab, hidden]` tensors at DIFFERENT addresses and that
> `global_bytes` charges both; red-proved by aliasing `head` onto `embed` in `GlimmerPin::build`.
> Measured on the shipped artifact: each is **2,689,662,976 B = 1.252x `i32::MAX`**, and their
> contents differ (blake2b `916305e9…` vs `f614edea…`). The check is on ADDRESSES, because
> aliasing is a placement property and equal contents would still be two tensors.

> **Item 3 — read, and it decodes.** Three greedy runs on the bf16 artifact, sole tenant under
> the flock, each with a KFD/GTT witness recorded (0 holders, 18.7 MB GTT before the two that
> matter). All 52 layers pin at 55.712 GB with an `auto` budget of 100 GiB, so these are
> ALL-RESIDENT numbers and say nothing about streaming.
>
> | prompt | emitted | s | tok/s | ended |
> |---|---|---|---|---|
> | NVMe-vs-quantized, 81 tok framed | 96 (limit) | 64.62 | **1.486** | hit the limit |
> | same | 500 (limit) | 221.13 | **2.261** | hit the limit |
> | "Say hello.", 59 tok framed | **110 of 400** | 60.98 | 1.804 | **EOS** |
>
> The spread is warm-up amortizing over a longer run, not variance in the loop.
>
> **The text is coherent and it is the model reasoning.** Every run completes the generation
> prompt `<|start|>assistant` with ` to=self<|message|>` — the reasoning channel the system
> block's `Reasoning strength: high` asks for — and the content is real: the 500-token run
> reaches "Modern NVMe SSDs provide multi-GB/s sequential bandwidth that can be overlapped with
> GPU compute via prefetching", writes three well-formed sentences that answer the question, then
> goes back to counting them. It restates the task several times, which is a reasoning-scratchpad
> pattern rather than a loop — read, not scored, since `distinct`/repeated-block are banned here.
>
> **The third run TERMINATED NATURALLY at 110 of 400**, which is the property this whole item is
> about: it is what GLM did in none of 56 runs before its framing was fixed, and it is the direct
> payoff of wiring the template into `run_glimmer`.
>
> **One finding for the serving layer, not for S4.** The run that stopped stopped *inside the
> reasoning channel* — so a bounded CLI decode yields the reasoning and never the answer. A
> reasoning turn ends with `<|eom|>` (200007), which is deliberately NOT one of the two stop ids,
> so the loop is right to run through it; ending the whole turn there is the model's choice. A
> server that wants the answer has to treat `<|eom|>` as "continue into the next channel" rather
> than as a stop, and there is no way to ask for a lower reasoning strength from the CLI today —
> `run_glimmer` takes `GlimmerChatOpts::default()`, which is the template's own `high`.

> **The streaming path is measured, and it is the first evidence Glimmer streams at all.**
> Same prompt at `--bench 24`, one `--max-mem` per arm, warm page cache:
>
> | GiB | pinned/streamed | tok/s | s/token | streamed GB/token | marginal GB/s |
> |---:|---:|---:|---:|---:|---:|
> | auto (100) | 52 / 0 | **1.804** | 0.554 | 0 | — |
> | 45 | 43 / 9 | 0.638 | 1.567 | 8.71 | 8.6 |
> | 30 | 26 / 26 | 0.453 | 2.208 | 25.17 | 15.2 |
> | 15 | 10 / 42 | 0.355 | 2.817 | 40.65 | 18.0 |
>
> **The pin is worth 5.1× end to end**, and marginal streaming bandwidth RISES with the streamed
> count (8.6 → 18.0 GB/s) — the per-fill fixed cost dominates a small streamed set and amortizes.
> That is the quantity §4.6's prefetch item is about, measured before the lever is built.
> `GLIMMER_STREAM_SLOTS` is 1 throughout.
>
> **This is not an S5 result and does not open S5** — nothing was built or changed. It
> characterises what S4 already shipped, which is what makes the later comparison possible.
>
> **The first version of the table was NON-MONOTONIC — 26 streamed layers beat 9 — and it was
> PAGE CACHE, not a finding.** The 45 GiB arm ran first and paid the cold NFS read for the whole
> 55.712 GB file: **82.81 s cold against 37.90 and 37.64 s warm, a 2.2× confound**, larger than
> most effects being measured and reproducing to 0.8% once warm. `CLAUDE.md` warns not to
> `cargo build` between arms; this is the same hazard from arm ORDER alone, on a file too big to
> stay resident across a cold start. **A sweep over budgets needs a discarded warm-up arm.**

**And this is what decides the first quantized format.** bf16 FITS on this machine — all 52
layers pin — so the reason to quantize here is not capacity, it is that any streaming at bf16
costs 3–5×. int4+g128 at 13.65 GB/token would put the whole model under **14 GB**, i.e.
all-resident at a budget where bf16 streams 42 of 52 layers and runs at 0.355 tok/s. That is a
measured argument for int4 over fp8 (26.51 GB/token), where the plan had only a table.

**Still open here:** item 4 (the dNLL ladder per format, and the softcap priced on trained
logits) and the first quantized format itself — int4+g128 at 13.65 GB/token is the only row that
plausibly decodes at interactive speed on this GTT; fp8 at 26.51 is the quality anchor. Item 4
cannot start before the format exists, since a ladder needs a second rung.

### 4.6 S5 — performance

Speculative decode (divides weight traffic by accepted length — the biggest lever on a
bandwidth-bound dense model), prefetch (perfect, since the schedule is known before the run),
the async slot fill, and giving the four stream-less launchers a stream so the loop can leave
the null stream. **Note the ordering constraint recorded on `Glimmer::layer`:** S5's prefetch
makes `GlimmerPin::layer`'s captured-pointer hazard reachable, and the `Handles` resolution is
what currently keeps it unreachable.

---

## 4b. OPEN — SIGSEGV inside HIP's VMM path, ~3 in 52 runs, and it is on ALLOC

**Found 2026-08-14 by a full-suite sweep.** `cargo test --test glimmer_reference` dies with
SIGSEGV. That file builds ~14 engines per run, so it cycles VMM allocate/free far harder than
anything else in the tree — which is why it surfaced here and nowhere else.

**Three cores, captured by systemd-coredump and read with `coredumpctl debug` — `ptrace_scope`
blocks a live attach but not a core.** All three land in `libamdhip64.so.7`, two at the same
instruction offset, and the frame below is rivoli's:

| core | rivoli frame | under |
|---|---|---|
| 2041319 | `rivoli_vmm_free` | `Drop for VmmBuf` → `DeviceTier` → `GlimmerPin` → `Glimmer` |
| 2019129 | **`rivoli_vmm_alloc`** | `VmmBuf::new` → `DeviceTier::new` → `GlimmerPin::build` |
| 2091304 | **`rivoli_vmm_alloc`** | same, via `common::decode_one` |

**Two of three are on ALLOCATION, not teardown** — `hipMemCreate`/`hipMemAddressReserve`/
`hipMemMap` on a fresh 1.4–1.7 MB tier, sizes the fixture uses constantly.

> **A fix was attempted and REFUTED by its own measurement.** `Drop for VmmBuf` called
> `hipMemUnmap`/`hipMemRelease` with no join, which is a real hazard `glimmer_gpu.rs` had already
> documented and half-fixed (error path only). Adding `device_sync()` there changed the rate from
> **2 in 27 to 1 in 25 — indistinguishable**, and the crash that followed it was an ALLOC one. So
> "unsynchronised teardown corrupts the runtime and the next allocation trips over it" is not
> supported. The join is kept, described accurately at the call site, because the hazard it
> closes is real on its own; it is not this bug's fix.

**What is ruled out:** GPU contention (the crashing run's KFD/GTT witness was identical to three
passing runs in the same batch); the probability-space metrics added the same day (host
arithmetic over a 61-element vector, no device call); and test ORDER — a 1-in-5-against-0-in-10
reading looked like a discriminator until the same configuration returned 0-in-10, so the arms
are indistinguishable at that sample size and the claim was withdrawn.

**Why it matters beyond flakiness:** this is the gate that scores rivoli against Muse Glimmer's
own logits, and a ~6% failure rate invites re-running until green — which is how a real
regression gets attributed to a flake. **It is also not Glimmer-specific**: `VmmBuf` is the
tier allocator for every model here, and nothing else builds enough engines per process to hit it.

**Next:** `rivoli_vmm_alloc` asks for `hipMemAllocationGranularityMinimum` rather than
`...Recommended`, which is the one rivoli-side choice in that function that is not forced;
whether the runtime is stable at minimum granularity under rapid reserve/free churn is the
question to put to a standalone reproducer, outside the test harness.

---

## 5. Housekeeping, needing the owner

* **The branch is 50 commits ahead of `origin/main` and unpushed** (`wt/glimmer-s2`).
* **The GPU node label is left at `disabled`** — `kubectl label node rh-anine.hr-home.xyz
  hr-home.xyz/rocm=true --overwrite` restores it; `ai/llama-swap` has been `Pending` since it
  was flipped.
