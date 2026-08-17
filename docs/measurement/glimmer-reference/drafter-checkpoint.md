---
scope: glimmer
status: data
verdict: The DFlash drafter checkpoint is on this box and was CONVERTED end to end, so M17b's numbers are measured rather than derived from the spec. It carries 58 tensors, NOT the spec's 59 -- eleven per layer x 5 plus encoder.fc, encoder.output_norm_enc and norm -- every one BF16, 2,555,985,152 parameters, 5,111,970,304 tensor bytes under a 6,296-byte header for a 5,111,976,608-byte file. The 59th tensor does not exist: DrafterConfig::census's header had predicted the discrepancy and named an `fc` bias as the candidate, and there is no bias, so the spec's PROSE is wrong and its own per-layer enumeration (which derives 58) is right. encoder.fc is [6656, 33280] = [hidden, 5 x hidden], the only place the length of target_layer_ids is visible in a SHAPE rather than in the config. convert_glimmer_drafter ran against the real checkpoint 2026-08-17 in 174 s on CPU/NFS, writing 36 tensors bf16 verbatim and 22 norms widened to f32, and the output artifact's tensor half is 5,112,132,608 B -- EXACTLY the pin this gate derives from the census, confirming the rank rule (1-D is a norm and is held f32) adds 162,304 B and nothing else. So the resident pin is 4.761 GiB, not the plan's 5.1 GiB, which was the file's size in GB read as GiB and overstated a P6 budget by 7.1% (5,476,083,302 B claimed against 5,112,132,608 B actual -- CORRECTED 2026-08-17 from 7.6%, a figure that followed from none of these numbers and was invented inside the paragraph correcting a different unit slip). Three M17c budgets are now derived from the shipped config rather than inherited: drafter KV is 2 x 8 kv_heads x 128 head_dim x 5 layers x 2 B = 20,480 B = 20.0 KiB/token exactly; the target-side hidden-state export is 5 x 6656 x 2 B = 66,560 B/token; and a ctx-4096 ring is 272,629,760 B = 260.0 MiB -- BUT ALL THREE ARE bf16 FIGURES AND THIS ENGINE'S CACHES ARE f32 (glimmer::pin::scratch is DeviceBuf::new(n*4) and geometry::kv_bytes ends in checked_mul(4)), so what the drafter COSTS at checkpoint dtype and what the engine ALLOCATES differ by exactly 2x: KV 40.0 KiB/token, export 133,120 B/token, and the ring 520.0 MiB rather than 260.0 unless the export deliberately narrows to bf16 on the way out of the residual stream -- a decision with a quality cost, not a formatting choice, and M17c's to make explicitly. Both columns are asserted, each labelled: under P5 a budget that does not say which dtype it is in is not yet a budget, and all three of the plan's did not. The checkpoint itself is NOT vendored -- only its 6,304-byte safetensors header, which carries every name, shape, dtype and offset and so is the whole census and none of the weights, pinned by length and FNV-1a and RECOMPUTED from the live file whenever the mount is present. All THIRTEEN gates in crates/cli/tests/drafter_convert.rs are proven red by nine plants (gate-red-proofs.md section 6; twelve gates and six plants as first landed, grown the same day under review), including the two that matter most: a one-byte header edit reddens the live-file comparison, which is the proof that conditional half is armed on this box rather than silently skipping, and a wrong drafter hidden_size reddens the POSITIVE pairing arm, which is what makes "the shipped drafter pairs with the shipped target" evidence instead of a run that failed early for its own reasons. What this does NOT establish: no VALUE was scored -- the conversion is a byte copy plus a widen, the drafter's arithmetic is scored by the CPU oracle on the tiny anchor fixtures, and nothing here has run a drafter forward pass. The TARGET's own name census still has no real-index gate. WHY THIS ARM'S GATES MEAN WHAT THEY SAY, recorded because M17c/d set their gates here: Glimmer decode reproduces itself byte-for-byte, both fully pinned and while streaming, so a lossless-drafting gate ("greedy ids identical with and without drafting") is EXPRESSIBLE on this arm and an acceptance histogram here is a measurement rather than a sample -- neither of which currently holds on GLM, whose decode does not reproduce itself and which has a blocking defect with its own owner. And DFlash economics must be priced against BOTH target dtypes, because a dense verify pass reads every weight once regardless of row count, so a cheaper fp8 target makes the drafter's FIXED cost a larger fraction of the step and break-even N is a function of the target's dtype. The supporting GLM-determinism and fp8 throughput FIGURES are other tracks' measurements, cited in the body and NOT owned here: this doc is scope glimmer and must not become the tree's only home for a scope-glm defect. THE LARGEST THING THE REAL CONFIG CONTRADICTS IN THE PLAN, measured before the kernel exists: the mask M17c must build is NOT the one the anchor pins. The reference's overlay is abs(q_idx - kv_idx) <= sliding_window with q_idx = row + q_offset, and q_offset is the cache's query offset when a cache is present and ZERO when it is not -- the anchor ran use_cache=False (a fresh DFlashCache reports kv_length 0), so every vendored golden pins the q_offset=0 branch. At the fixture's ctx 12 / block 4 / window 13 that is 13 of 16 block-vs-block pairs, which is what M17a re-vendored to obtain; at the SHIPPED ctx 4096 / block 16 / window 2048 the same expression gives ZERO of 256, because q_idx <= 15 cannot reach kv >= ctx once ctx > window -- the block does not attend itself at all and the drafter degenerates to a context-reader that still produces finite logits and passes every shape check. The cache branch (q_offset = ctx) gives 256 of 256 with 120 strictly bidirectional. The anchor CANNOT arbitrate because the two indexings differ even at the tiny geometry (13 of 16 vs 16 of 16, 0 masked context columns vs 3), so the goldens pin one by value and it is the one that does not survive scaling -- M17a's re-vendor was necessary and correct and still leaves the property pinned in a regime that inverts at production widths. M17c therefore carries q_offset as an EXPLICIT kernel argument, 0 for anchor parity and ctx in decode, never a default.
---

# The DFlash drafter checkpoint, measured

`/swarm/storage/ai/rivoli/muse-glimmer-30b-assistant/` — the Muse Glimmer 30B **assistant**
checkpoint, which is the DFlash block drafter. It arrived after `DrafterConfig` was written
against `glimmer-architecture.md` §11, which is the useful order: the schema was a prediction
before it was a reading.

## The file

| | |
|---|---|
| `model.safetensors` | **5,111,976,608 B** |
| header | 8-byte length prefix + **6,296 B** of JSON, `__metadata__` `{"format": "pt"}` |
| tensors | **58**, every one **BF16** |
| tensor bytes | **5,111,970,304** — and `8 + 6,296 + 5,111,970,304` is the file size exactly |
| parameters | **2,555,985,152** (the "2.556 B" of the spec, from the census rather than the card) |

The offsets **tile the file from 0 with no gap and no overlap**, which is what turns a census of
shapes into a claim about a file. A header can carry correct shapes and still not describe
5,111,976,608 bytes; only the offsets say it does.

### 58 tensors, not 59 — the spec's prose is wrong

`DrafterConfig::census`'s own header predicted this discrepancy before the file existed: the
spec's prose says 59 while its own per-layer enumeration derives 58, and the converter compares
by exact SET equality so that "whichever tensor accounts for the difference (an `fc` bias is the
candidate) is NAMED by the gate on first contact". **There is no bias.** The set is exactly:

| group | count | names |
|---|---|---|
| per layer, × 5 | 55 | `self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, `mlp.{gate,up,down}_proj`, `input_layernorm`, `post_attention_layernorm` |
| model-level | 3 | `encoder.fc`, `encoder.output_norm_enc`, `norm` |

11 × 5 + 3 = 58. The prediction was right and the prose is wrong.

**And two absences are as load-bearing as the 58 presences**: there is no `embed_tokens` and no
`lm_head`. The drafter *borrows* the target's, which is why there is no standalone drafter
artifact — a drafter alone decodes nothing — and why the converter treats their **presence** in a
checkpoint as an error rather than a bonus.

### `encoder.fc [6656, 33280]` is the 5×hidden concat

The tensor that makes this a DFlash drafter rather than a small dense target. It takes the
**concatenation** of the target's hidden state at each of `target_layer_ids` `[1, 13, 25, 37, 49]`
and projects it back to one hidden width, so its input is `5 × 6656 = 33280`. That product is the
**only place in the checkpoint where the length of `target_layer_ids` appears in a shape rather
than in the config**, which is why the gate derives it (`[h, ids.len() * h]`) as well as spelling
it out: a config listing the wrong number of target layers reddens instead of being agreed with.

### The shipped config, for the record

`hidden 6656` · `inter 19968` · `5 layers` · `32 Q / 8 KV heads` · `head_dim 128` ·
`block_size 16` · `mask_token_id 201818` · `sliding_window 2048` · `rope_theta 500000` ·
`layer_types` all five `sliding_attention` · `rms_norm_eps 1e-5`.

`32 × 128 = 4096 ≠ 6656`: **the head width and the hidden width are different here**, which is the
trap every Glimmer fixture in this tree is built to keep visible. A converter or kernel that
derived one from the other passes a fixture where they coincide.

## The conversion, run end to end 2026-08-17

Not a synthetic fixture — the real 5.1 GB file. The target side is a scratch directory holding
**only** `manifest.json`, copied from `/swarm/storage/ai/rivoli/glimmer-30b-bf16/`, because
`refuse_before_writing` reaches the target through `GlimmerConfig::load` and nothing else. This
kept the production artifact untouched: nothing was written into the shared NFS artifact.

```bash
cp /swarm/storage/ai/rivoli/glimmer-30b-bf16/manifest.json  $PROBE/manifest.json
convert_glimmer_drafter /swarm/storage/ai/rivoli/muse-glimmer-30b-assistant  $PROBE
# convert_glimmer_drafter: 36 tensors bf16 verbatim, 22 norms widened to f32,
#                          58 tensors total, 5112138812 B -> $PROBE/drafter/resident.safetensors
```

**174 seconds, CPU and NFS only, no GPU, exit 0.** Dev profile, so `debug_assert!` was live.

| | derived from the census | observed in the artifact |
|---|---|---|
| tensors | 58 | 58 |
| bf16 verbatim | 36 | 36 `BF16` |
| norms widened to f32 | 22 | 22 `F32` |
| tensor bytes | **5,112,132,608** | **5,112,132,608** |
| file bytes | — | 5,112,138,812 = `8 + 6,196` header + the above |

> **What of this run IS gated, and what is only recorded.** The derived pin — 5,112,132,608 B —
> is asserted in `drafter_convert.rs` from the census and, since 2026-08-17, independently from
> the real header's own ranks. The *output artifact's* total (5,112,138,812 B) and its 6,196-byte
> header are a **one-off observation** with no test re-asserting them: that suite gates the INPUT
> checkpoint, and the output header's size is the writer's business, not the model's. A
> regeneration that changed the output header would not redden anything, and should not.

The two sides agreeing **exactly** is the measurement. 22 is `4 × 5 + 2` — both layernorms plus
`q_norm`/`k_norm` per layer, plus `encoder.output_norm_enc` and `norm` — covering **81,152**
elements, and widening those from bf16 to f32 adds `81,152 × 2 = 162,304 B` and nothing else. So
the artifact is the checkpoint's tensor bytes plus 162,304, and the house rank rule ("1-D is a
norm and is held f32, everything else is a projection and stays bf16") is confirmed against a real
file rather than asserted.

### The pin is 4.761 GiB, not 5.1 GiB

```
5,112,132,608 B / 2^30 = 4.7611 GiB
```

The wave plan said "5.1 GiB". That is the checkpoint's size in **GB** read as **GiB**, and the
overstatement is **7.1%**, derived rather than eyeballed:

```
5.1 GiB      = 5,476,083,302 B
pin          = 5,112,132,608 B
overstatement =   363,950,694 B  =  7.119%
```

> **CORRECTED 2026-08-17, same day.** This said **7.6%**, which follows from none of the numbers
> above — review recomputed it and got 7.119%. Neither 5.1/4.7612 nor 5.112/4.7612 yields 7.6%
> either, so the figure was not a mis-pairing of real quantities; it was invented. **That is this
> repo's most-named defect committed inside the very paragraph correcting a different unit slip**,
> which is why the arithmetic is now written out instead of stated. A percentage with no visible
> numerator is a number nobody can check.

P6 spends this against free memory, where the pin is a function of what is free and never of
architecture. Under P6 the drafter is simply **more resident bytes**, never a
special case; 4.761 GiB is what it costs.

## The three M17c budgets, derived

Stated in the plan and now derived from the shipped config, so none of them is an inherited
number:

| budget | derivation | value |
|---|---|---|
| drafter KV per token | `2 (K,V) × 8 kv_heads × 128 head_dim × 5 layers × 2 B` | **20,480 B = 20.0 KiB** exactly |
| target hidden-state export per token | `5 target layers × 6656 hidden × 2 B` | **66,560 B** |
| export ring at ctx 4096 | `66,560 × 4096` | **272,629,760 B = 260.0 MiB** exactly |

All three match the plan, and each is exact rather than rounded — 20,480 B *is* 20.0 KiB, the ring
*is* 260.0 MiB — which is worth saying because "~20 KiB" and "≈260 MiB" read like estimates.

> **CORRECTED 2026-08-17, same day: all three are bf16 figures, and this engine's caches are f32.**
> The plan prices 2 bytes per element, which is the *checkpoint's* dtype. The Glimmer engine
> allocates its KV cache and its residual stream through
> `glimmer::pin::scratch(n) = DeviceBuf::new(n * 4)`, and `geometry::kv_bytes` ends in
> `checked_mul(4)`. So what the drafter would COST at checkpoint dtype and what the engine would
> ALLOCATE differ by exactly 2×:
>
> | budget | at bf16 (the plan) | as this engine allocates (f32) |
> |---|---|---|
> | drafter KV / token | 20,480 B = **20.0 KiB** | 40,960 B = **40.0 KiB** |
> | hidden-state export / token | 66,560 B | 133,120 B |
> | export ring at ctx 4096 | 272,629,760 B = **260.0 MiB** | 545,259,520 B = **520.0 MiB** |
>
> **The ring is the one that matters: 520.0 MiB, not 260.0, unless the export deliberately narrows
> to bf16 on the way out of the residual stream.** That narrowing is a decision with a quality cost
> attached, not a formatting choice, and it is M17c's to make explicitly — the residual stream is
> f32 for a reason and the drafter consumes what the target hands it. Both columns are now asserted
> in `drafter_convert.rs`, each labelled, because M17c must allocate from one and size a ring from
> the other. **Under P5 bytes/token is the currency, so a budget that does not say which dtype it
> is in is not yet a budget** — and all three of the plan's did not.

## Why this arm's gates mean what they say

Recorded here because M17c and M17d will set tolerance and byte-identity gates from this file, and
the same gates are **not** currently available on the GLM arm.

> **Every figure in this section is another track's measurement, reported to M17 by the
> coordinator 2026-08-17 and NOT witnessed here.** They are cited because they change what M17c/d
> may claim, and they are cited *as citations*: this file is `scope: glimmer`, and a GLM-scoped
> defect must not have its only home in a Glimmer doc where a scope-filtered reader would never
> find it. **The GLM determinism figures belong in a `scope: glm` doc owned by the agent holding
> that root cause, and the fp8 throughput pair belongs to the M11 fp8 track's own measurement
> record.** Until those exist, treat the numbers below as unwitnessed here and go to those owners
> before citing them anywhere else.

**Glimmer decode is byte-identical to itself** — fully pinned, and also while streaming 21 of 52
layers through one slot, so identity survives the residency path rather than only the resident
one. Teacher-forced twice with identical flags it reproduces PPL 7.008490 exactly. **GLM does
not**: 512-token decode has been reported differing at 496 of 512 ids after one contention event
and at 61 of 512 on a quiet box, with the old tree worse at 247 of 512 — so it is neither a
rewrite regression nor the ROCm upgrade, it is a hard blocking defect with its own owner.

The consequence for DFlash is direct and it is the reason the economics are testable here first:

- **A lossless-drafting gate is expressible on this arm.** "Greedy ids byte-identical with and
  without drafting" is a real assertion when the no-drafting baseline reproduces itself. On an arm
  that diverges from itself, that gate cannot distinguish a broken verify from the baseline's own
  noise, and the only honest form left is a distributional one.
- **A tolerance means a tolerance.** A bound measured against a reproducible reference is a bound;
  measured against a drifting one it is a bound plus an unknown.
- **An acceptance histogram is a measurement rather than a sample.** Acceptance length is the whole
  break-even question (dense verify inverts the economics — break-even is N > 1.1 accepted tokens,
  not GLM's 53%), and a histogram from a decode that does not repeat is not one number.

**Price the drafter against BOTH target dtypes, not bf16.** Glimmer fp8 shipped at **5.28 tok/s
against bf16's 2.45 — 2.16×**, both fully pinned over 512 tokens with no silent fallback
(`Fp8 { block: 128 }` projections in a 28.5 GiB tier, against `Bf16` in 51.9 GiB). A dense verify
pass reads every weight once regardless of how many rows it verifies, so making the target pass
2.16× cheaper makes the drafter's **fixed** cost a correspondingly larger fraction of the step.
The break-even N is therefore a function of the target's dtype, and quoting one number for it
without saying which dtype it was measured at is the mistake to avoid at M17e.

## The mask M17c must build is NOT the mask the anchor pins

**Measured 2026-08-17, before the kernel exists — which is the only order in which it can prevent
anything.** This is the largest thing the real config contradicts in the plan.

The reference's overlay is one line, `masking_utils.py::sliding_window_bidirectional_overlay`: "a
token can attend to any other token if their absolute distance is within the (inclusive) sliding
window size", i.e. `abs(q_idx - kv_idx) <= sliding_window`.

`q_idx` is `row + q_offset`, and `_preprocess_mask_arguments` sets
`q_offset = past_key_values.get_query_offset(layer_idx)` **when a cache is present and `0` when it
is not**. Both branches reach the same overlay, so the mask's entire meaning turns on which branch
built it — and nothing in the shapes, the byte counts or the census says which.

**The S1b anchor is the no-cache branch.** Its own recorded reference behaviour is that a fresh
`DFlashCache` reports `kv_length` 0 and the correct 2D mask only works with `use_cache=False`, so
every vendored draft golden pins **`q_offset = 0`**.

Both branches, counted over block-vs-block pairs (`kv >= ctx`), strictly-bidirectional pairs among
them (a query attending a **later** row of its own block — the only positive evidence of
bidirectionality, since a causal mask permits none), and masked context columns:

| geometry | `q_offset` | block pairs attending | strictly bidirectional | ctx cols masked |
|---|---|---|---|---|
| fixture: ctx 12, block 4, window 13 | **0** (the anchor) | **13 of 16** | 3 | 0 |
| fixture: ctx 12, block 4, window 13 | 12 (cache) | 16 of 16 | 6 | 3 |
| **shipped: ctx 4096, block 16, window 2048** | **0** (anchor form) | **0 of 256** | **0** | 32,632 |
| **shipped: ctx 4096, block 16, window 2048** | 4096 (cache) | **256 of 256** | 120 | 32,888 |

**Read the third row.** With `q_offset = 0`, `q_idx` is at most `block − 1` = 15, so no query can
reach a key at `kv >= ctx` once `ctx > window`. **The block does not attend itself at all.** A
kernel that transliterates the anchor's mask is not a block drafter — it is a context-reader that
would still produce finite logits, still pass every shape check in this file, and still decode. The
cache branch restores the property in full.

**And the anchor cannot arbitrate**, which is why this needed measuring rather than reading. The two
indexings differ *even at the fixture's geometry* — 13 of 16 against 16 of 16, and 0 masked context
columns against 3 — so the goldens pin one of them **by value**, and it is the one that does not
survive scaling. `dflash.rs`'s `mask` docstring had anticipated the shape of the question ("whether
that off-window indexing is desirable at real context lengths is a serving-path question the cache
answers, not this fixture"); this is that question answered, in the direction that costs something.

**There is an irony worth stating, because it bounds what M17a bought.** M17a's headline finding was
that the old fixtures pinned bidirectionality by *no value at all* — the block-vs-block submatrix
summed to exactly 0.0 — and the re-vendor to `sliding_window` 13 fixed that, correctly and
necessarily. But the property it now pins by value is pinned **in the `q_offset = 0` regime**, where
it holds for a reason that inverts at production widths. The fixture is strictly better than it was
and still cannot answer this.

**So M17c carries `q_offset` as an explicit kernel argument**: `0` when scoring against the vendored
goldens, so anchor parity stays exact, and `ctx` in the decode path. Not a default and not an
inferred value — the two regimes differ by 256 attending pairs at ctx 4096, and a wrong default is
invisible to every other gate here.
`drafter_convert.rs::the_serving_mask_indexes_queries_by_position_and_the_anchor_pins_the_other_branch`
holds both arms, red-proofed by passing `0` at the serving call site (which is the mistake, and it
reddens with `left: 0, right: 256`).

## What is vendored, and why only the header

`crates/cli/tests/drafter-checkpoint-header.bin` — **6,304 bytes**: the 8-byte length prefix and
the header JSON. FNV-1a `0xf0c2c64967d79b88`.

That is every name, shape, dtype and byte offset — **the whole census and none of the weights**.
So `crates/cli/tests/drafter_convert.rs` runs with no NFS, no 5 GB read and no device, and still
fails on a checkpoint whose tensor set is not this one.

**The pin is recomputed from the live file, not compared against a frozen copy of itself.** When
the mount is present, `the_vendored_header_is_the_live_checkpoints_own_bytes` reads the live
6,304 bytes, checks the file's size, and compares byte-for-byte as well as by FNV. A pin checked
only against a copy of itself is decoration; that lesson is why this half exists.

**Its one conditional, stated rather than latent:** with the mount absent there is nothing to
compare, so the test degrades to the vendored-only checks the other eleven already make. It is
never *vacuous* — it always parses and counts the vendored header — but its extra power depends on
the mount, and a green run does not by itself say which of the two it got. Plant P1 below is the
evidence that it is armed on this box.

## Red proofs

All twelve gates are proven red, by six plants. The battery, the coverage matrix and the two
plants worth reading are in **`docs/measurement/gate-red-proofs.md` §6** — not repeated here,
because a second copy of the same table is a thing to drift rather than a check.

## What this does NOT establish

- **No value was scored.** The conversion is a byte copy plus a widen. Nothing here has run a
  drafter forward pass; the drafter's *arithmetic* is scored by the CPU oracle
  (`crates/oracles/tests/glimmer_draft_oracle.rs`) against the tiny anchor fixtures, at toy widths
  with drawn weights. The two halves do not overlap: this file gates the tensors, that one gates
  the math, and **neither has yet compared a real weight to a real activation.** That is M17c.
- **No target-side name census against a real index.** `crates/cli/tests/glimmer_convert.rs` uses
  a synthetic index it writes itself, so a name or shape wrong in *both* the schema and the
  fixture is still uncaught for the target. This file is the shape that gate should take when the
  target's checkpoint work lands.
- **Nothing about the artifact re-opening in the engine.** `DrafterConfig::load` round-trips the
  published manifest inside the converter, and that is where it stops; no pin has placed these
  bytes on a device.
- **Nothing about decode legality.** A Glimmer artifact carrying a `drafter/` sub-artifact is what
  will make `--mtp` legal on this arm, and that flip is M17d's, behind its own economics gate.
