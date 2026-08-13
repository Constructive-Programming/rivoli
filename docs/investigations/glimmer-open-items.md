---
scope: glimmer
status: live
verdict: Everything the Muse Glimmer port owes after S3's layer loop landed 2026-08-13, in one list, because the obligations had spread across four commit messages, two verdict lines and three source comments. THE SHORT VERSION -- five open items, all of them stage work (S4, S5 and the rest of G3); six of the eleven were closed the day this file was written, and two of those six had been recorded as open on evidence that was already stale. The three live CLASSES this file opened with -- ungated correctness, fixture blind spots, unpriced cost -- are all empty. (1) UNGATED CORRECTNESS: EMPTY as of 2026-08-13. The eps transposition -- the item this register called its sharpest -- turned out to be gated already, at 3.673e-5 against the chain gate's 2e-5, closed by the softcap tolerance work one commit earlier; the 'nothing reddens' recorded here was a measurement taken before that change and never re-run, which is the same stale-fact defect as the wrong citation it replaced. A second, LOCALISING gate was added anyway (Glimmer::branch plus a per-layer score against the oracle), and the route this file predicted -- scoring the reference's own captures -- is REFUTED by measurement: there the eps signal is 1.6e-3 to 1.3e-2 against a bf16 weight floor of 4.7e-3 to 3.0e-2, i.e. 0.2x to 0.6x the noise at every layer of both salts. Both surviving margins are thin (1.8x and 1.6x) because the FIXTURE's branch sits at mean(x2) ~ O(1) where the two epsilons nearly agree -- which is a second reason to do (2). (2) FIXTURE-GEOMETRY BLIND SPOTS: CLOSED 2026-08-13 by widening the toy checkpoint from (2 heads, 1 kv, head_dim = dim) to (4, 2, dim/2). Both were UNCONSTRUCTIBLE rather than merely uncaught -- at head_dim = hidden the two candidate softmax scales are the same NUMBER, and at hkv = 1 the block and modulo KV broadcasts are the same FUNCTION, so no tolerance could have separated either. They now measure 3.7e-1 and 9.5e-1. Six test binaries share the fixture and all pass unchanged; the widening also improved every clean floor in the chain gate by 2.7-5.5x, so its three tolerances and its whole 13-row mutation census were re-derived rather than carried over. (3) UNPRICED COST: the KV cache item is CLOSED 2026-08-13 -- glimmer_gpu::runtime_bytes is subtracted from the budget BEFORE the tier is sized, a budget that cannot cover it is refused by name, the slot count is one function read by both the allocation and the accounting, and the residency line moved below the tokenizer because the footprint is a function of the context; gated deviceless at the shipped widths plus a partition-boundary assertion, with one gap stated rather than papered over (removing the subtraction leaves every test green, since its effect needs a genuinely near-full device). Prefill is CLOSED too: Glimmer::prefill is layer-major and chunked at 256, so GlimmerPin::layer is called once per layer per chunk instead of once per token -- measured 6 fills against 900 token-major on the fixture, red-proved by reverting the order. It is a pure reorder (every numeric gate was bit-for-bit unchanged, which is the check), so no output comparison can see it and Glimmer::slot_stats exposes the pin's fill counter for the assertion. Chunked rather than whole-prompt because the batch's residual streams stay live: whole-prompt is 3.49 GB at the 131072 ceiling and 3.1's own gate caught it at 7.144 GB, where chunking costs 6.8 MB for 99.6% of the saving. The MATH is still m=1 / tq=1; batching it needs a rows dimension on the centered norm and a per-row rope position, changes the arithmetic, and first makes the launcher's ring union hazard reachable -- separate work. (4) STAGE WORK, six, mostly S4 and S5. G3's comparison against MUSE GLIMMER rather than against rivoli's own transcription is DONE 2026-08-13 (tests/glimmer_reference.rs): the anchor driver now exports all 107 parameter tensors, both goldens regenerate BYTE-IDENTICAL so the change is provably additive, and rivoli's loop reproduces the reference's own logits on the reference's own weights over 7 steps x 2 salts, worst 4.8e-2, all 14 emitted tokens exact, red-proved at 1.3e0 / 1.1e0 / 6.7e-1. It carries a bf16 weight-rounding term MEASURED at 9.3e-3 to 6.8e-2 (rivoli stores projections bf16, the reference computed f32), which sets its tolerance and costs it resolution -- so it and the finer chain gate are complements, not substitutes. Its load-bearing by-product is that an independent f64 transcription of sections 3-5 reproduces the reference to 3.8e-6, the first evidence here that the architecture doc is right about the WHOLE chain. AND IT DID NOT CLOSE THE EPS ITEM, which this file had expected it to: the transposition is green on the reference's own weights too, so it is not a fixture artefact. Also carried here: the branch is 50 commits unpushed, and the GPU node label is left at disabled with llama-swap Pending.
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

**Both margins are thin, and both are measured** — 1.8x over tolerance for the logits, 1.6x for
the branch. The cause is the FIXTURE: its branch sits at `mean(x²)` ~ O(1), where 1e-5 and 1e-8
are nearly the same number. The reference's sits at ~1e-3, where `glimmer_norm.rs` measures
41.8-56.6x. **§2 widens this** — a fixture with a realistic branch statistic turns two thin
gates into two comfortable ones, which is a second reason to do that work.

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
counter and `tests/glimmer_loop.rs` asserts against `n_layers * chunks`.

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

### 4.2 G3 — the probability-space softcap check

S2 proved argmax-invariance, so no greedy gate can see the softcap. `Glimmer::logits()` is the
accessor for it and exists; the check compares softmax/NLL against the reference and requires a
`softcap_off` run to redden. §4.1 supplies the reference side.

> Partly paid already: the chain gate red-proves a deleted softcap (4.1e0) and the `tanh` alone
> (9.9e-5), and that second figure is what set its tolerance. What is missing is the comparison
> in probability space rather than in logits.

### 4.3 G3 — a decode crossing position 2048

The ring's first eviction. `glimmer_chain.rs` wraps the fixture's window four times, which is
the same mechanism at `sliding_window = 2`; this is the shipped-width version and needs S4's
weights.

### 4.4 The `NULL_STREAM` source census

`src/backend.rs` defines `NULL_STREAM` so a deliberate null reads as a decision, and about
fourteen `src/` call sites still pass a bare `std::ptr::null_mut()` in the stream position —
including one in a file that imports `NULL_STREAM` and uses it eight lines away. One test, no
device, `kernel_coverage.rs`'s style.

**Deferred to the owner:** it edits three engine files this port does not own.

### 4.5 S4 — real weights

The full checkpoint bf16-verbatim, then the first quantized format by dNLL. Carries the chat
template (hand-ported and byte-pinned; the artifact drops it) and the 2.69 GB `lm_head`, which
is 1.25x `i32::MAX` bytes and untested at every stage.

### 4.6 S5 — performance

Speculative decode (divides weight traffic by accepted length — the biggest lever on a
bandwidth-bound dense model), prefetch (perfect, since the schedule is known before the run),
the async slot fill, and giving the four stream-less launchers a stream so the loop can leave
the null stream. **Note the ordering constraint recorded on `Glimmer::layer`:** S5's prefetch
makes `GlimmerPin::layer`'s captured-pointer hazard reachable, and the `Handles` resolution is
what currently keeps it unreachable.

---

## 5. Housekeeping, needing the owner

* **The branch is 50 commits ahead of `origin/main` and unpushed** (`wt/glimmer-s2`).
* **The GPU node label is left at `disabled`** — `kubectl label node rh-anine.hr-home.xyz
  hr-home.xyz/rocm=true --overwrite` restores it; `ai/llama-swap` has been `Pending` since it
  was flipped.
