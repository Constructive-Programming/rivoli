---
scope: glimmer
status: live
verdict: What is LEFT of the Muse Glimmer-30B integration, re-planned 2026-08-12 against reference/principles.md after the owner corrected the port's central assumption — the pin is a function of free memory, so S1a's all-resident GlimmerPin ("a dense model has nothing to stream") violates P6 and dense makes streaming MORE load-bearing, not less: every weight is read every token (53.02 GB bf16 / 26.51 fp8 / 13.65 int4), there is no routed union to hide behind, and the resident fraction times bandwidth IS the tok/s model. Four stages remain: R1 residency contract + budget-aware pin (cyclic access makes LRU pathological — hit rate 0 at any deficit — and Belady degenerates to a STATIC prefix partition, so the policy axis collapses; output across budgets gated identical at zero tolerance on the tiny model), S3 layer loop against that contract from day one (sandwich norms x*(1+w) have NO kernel yet; gate operand scored vs gate_proj.out captures; streams passed, not null; G3 owes the softcap probability-space check nothing at S2 could make), S4 real weights (chat template byte-pinned, dNLL ladder per format, the 2.69 GB lm_head is 1.25x i32::MAX bytes and untested at every stage), S5 performance (speculative decode DIVIDES weight traffic by accepted length — the biggest single lever on a bandwidth-bound dense model; prefetch is PERFECT because the schedule is known before the run; NPU re-opened per P3, the closure is GLM-scoped). R1 IMPLEMENTED 2026-08-12 and then substantially corrected by three reviews: the contract is GlimmerPin::layer(l), one shape whether pinned or streamed, gated at 24 (budget, layer, pass) resolutions all byte-identical to all-resident; --max-mem is ACCEPTED and no longer hidden, and run_glimmer reports the partition AFTER every refusal (above them, device_budget's hipMemGetInfo let a low free-GTT reading preempt every architectural refusal with a memory error, and made the flag suite a GPU arm silently). THE REVIEWS OVERTURNED TWO OF MY ARGUMENTS, not just my code. GLIMMER_STREAM_SLOTS is 1, not 2, and the slot count is NOT a correctness property: kernel launches are asynchronous, so no finite slot count establishes write-after-read ordering -- only a dependency does -- and with a synchronous fill a second slot buys no overlap while costing one extra streamed layer (967.942 MB) every token, because the floor charges every slot unconditionally. The invariant a caller owes is now stated on layer() and S5 must add the fence in the same change as any slot increase. Second, R1 had introduced a real regression: only the PINNED prefix got the dtype and shape checks, so which invariants a layer received became a function of --max-mem, and a q_proj stored transposed is byte-identical in length and was accepted under a low budget while refused at full residency; every layer's headers are now validated at build. Also: partition asks for what it uses (the version first recorded as a fix guarded an unreachable branch and left the real streaming-path over-allocation); Slot writes to each tensor's own address, dropping a base-is-lowest assumption that would have wrapped under --release; the per-layer figure is 967.942 MB, not the checkpoint's bf16-norm 967.889 MB, and is now pinned by a shipped-widths test; the >2 GiB gate tested DeviceBuf/hipMemcpy while the pin places through DeviceTier/VmmBuf and now runs the pin's path; and G-R1(b) was WITHDRAWN rather than repaired, because its test re-implemented the slot map as a local closure and its premise dissolved. Still open: the attention-weights pin-vs-slot decision, which needs a decode to measure. Supersedes glimmer-port.md's S3+ sections ON SWITCH-OVER, which is the owner's call; S0-S2 records stay where they are.
---

# Muse Glimmer — what is left, planned against the principles

**Read `reference/principles.md` first; every stage below cites the principle it serves.**
This plan replaces `glimmer-port.md`'s remaining stages when the owner switches over;
S0–S2's records (done, gated, reviewed twice) stay in that file and are not restated here.

## Why the re-plan

`glimmer-port.md` S1a shipped `GlimmerPin` as "the resident weight set. **All of it**" — 55.7
GB bf16 placed unconditionally, no budget parameter, on the stated ground that "a dense model
has nothing to stream." **That is P6 inverted.** The pin is a function of free memory at run
time (other tenants have held 41 GB of this GTT; KV at 131072 context is 1.7 GB; the DFlash
drafter is another 2.6 GB) — and dense makes the streaming path MORE central than GLM's, not
less:

| | GLM-5.2 (MoE) | Glimmer (dense) |
|---|---|---|
| weights read per token | the routed UNION (top-8 of 256 + shared) | **every weight, every token** |
| bytes/token | budget-dependent | **53.02 GB bf16 / 26.51 fp8 / 13.65 int4+g128** (§7, arch doc) |
| access order | router-dependent, skewed | **fixed cyclic** — known before the run starts |
| what the cache exploits | routing skew | **nothing to exploit; partition and prefetch** |

Two consequences the old plan never priced:

1. **Cyclic access is LRU's pathological case.** At budget M < N layers, LRU evicts exactly
   the layer needed next: hit rate → **0**, not M/N. Belady on a cyclic scan (evict the
   just-used block — its next use is farthest) is equivalent to a **static partition**: pin a
   fixed prefix, stream the rest, hit fraction = M/N. So Glimmer collapses the
   `--cache-policy` axis to one correct answer, and any dynamic policy can only match or lose
   to it. R1 builds the partition, not a cache.
2. **The tok/s model is one line.** `t_token ≈ resident_bytes/BW_gtt +
   streamed_bytes/BW_nvme`, and the NVMe term is ~an order of magnitude more expensive per
   byte — so the resident fraction is the whole performance story, which is exactly why P6
   is not a nicety here. (Both bandwidths get MEASURED numbers in R1 before any tok/s claim;
   this plan deliberately quotes none it does not have.)

## R1 — the residency contract, and a pin that takes a budget

**Serves P6, P1, P4. Blocks S3 (the loop consumes the contract) and S4 (real weights do not
fit).**

1. **The contract first, one seam:** a per-layer source the loop calls —
   resolved device pointers for layer `l`, plus the stream the caller must order against.
   Behind it, two implementations: *pinned* (pointers into the tier, null-equivalent
   ordering) and *streamed* (a slot the prefetcher filled, ordered on the fetch stream).
   The loop cannot tell which it got; that indistinguishability is what the P4 gate below
   asserts. `RoutedPool`'s slot is three projections — exactly a dense SwiGLU MLP — so the
   MLP side reuses that machinery; **attention weights (170.4 MB/layer: q/gate/o at 54.5 MB,
   k/v at 3.4) have no slot shape and need one**, or the arithmetic decision below removes
   the need.
2. **`GlimmerPin::build(budget)`**: embed + lm_head + every norm are unconditionally
   resident (5.38 GB bf16 + 13 KB — read every token, tiny vs the layers); then whole layers
   in ascending order until the budget stops; the remainder registered with the streamer.
   Refuse a budget below the floor (residents + KV at max context + scratch + one streamed
   layer's double-buffer) with the number in the message, per `run_glimmer`'s refusal style.
3. **Attention-weights decision, made with arithmetic and then measured:** pinning ALL
   attention weights costs 8.86 GB bf16 / ~2.2 GB int4 and removes the second slot class
   entirely (only MLPs stream — 82% of layer bytes). Take it if the floor stays acceptable
   at the formats that matter; otherwise build the second slot class. Record the decision
   and the measurement here either way.
4. **Prefetch is a schedule, not a predictor.** Next layer is known before the run starts —
   GLM's whole unpredictability apparatus is dead weight here. Depth-k cyclic prefetch on
   the fetch stream, overlap gated in S5, correctness gated now: a deliberate off-by-one in
   the schedule (fetch l+1's bytes into l's slot) must redden the R1 gates.

### R1 as built — 2026-08-12, after three reviews

**The contract is `GlimmerPin::layer(l) -> &GlimmerLayerPin`.** Same twelve-field shape whether
the layer was pinned or arrived through a slot; the caller cannot tell, and that is P4 expressed
as a type. `tests/glimmer_residency.rs` gates it: **24 (budget, layer, pass) resolutions, all
byte-identical to the all-resident pin.**

**`--max-mem` is ACCEPTED** (and no longer hidden from Glimmer's `--help`); `run_glimmer` reports
the partition from `GlimmerTextConfig::partition`, the same arithmetic the pin uses, **after every
refusal** — see the review findings below for why the ordering is load-bearing.
**`--cache-policy` stays refused for a new reason**: the old one ("a dense model streams nothing")
became false the moment a budget could leave layers streaming; the standing one is that cyclic
access makes LRU evict exactly the layer needed next and Belady degenerates to a fixed prefix.

#### What three reviews changed, and two of them were my arguments rather than my code

1. **`GLIMMER_STREAM_SLOTS` is 1, not 2, and the slot count is NOT a correctness property.** The
   first version asserted "two is a correctness requirement rather than a tuning choice — one slot
   would be refilled while a kernel still reads it", and even shipped a `const` assert for it. Two
   reviews independently showed the argument does not work: **kernel launches are asynchronous**,
   so a host running two layers ahead overwrites slot 0 under a live kernel with two slots exactly
   as with one. **No finite slot count establishes write-after-read ordering — only a dependency
   does.** With R1's synchronous fill a second slot buys no overlap and costs one extra streamed
   layer every token (967.942 MB), because `floor_bytes` charges every slot unconditionally and
   each one pins one layer fewer. **S5 raises the count and adds the fence in the same change.**
2. **The invariant a caller owes is now stated on `layer()`**: before requesting a layer that maps
   to slot `s`, retire every kernel reading `s`'s previous occupant. There is still no ticket, but
   for a narrower reason than first given — a ticket expresses fill-then-read, which is the
   dependency that genuinely does not exist while the fill is synchronous; the one that DOES exist
   runs the other way and belongs in an event or a `device_sync`. The original argument ("valid for
   the same reason as `DeviceTier::place`") was false, and `place`'s running before any kernel
   exists was the tell.
3. **Every layer's headers are validated at build, pinned or not.** This was a real regression R1
   introduced: only the pinned prefix went through the dtype and shape checks two reviews added on
   2026-08-11, so **which invariants a layer received became a function of `--max-mem`**. A
   `q_proj` stored `[6656, 4096]` is byte-identical in length to `[4096, 6656]` and was accepted
   outright under a low budget while being refused at full residency — a transposed matrix, fluent
   wrong text, no error. Headers only: 12 index lookups per layer, no bytes, no device.
4. **`partition` asks for what it uses.** The version this plan first recorded "fixed" an
   over-allocation on the all-pinned path that **the arithmetic makes unreachable**, and left the
   real one on the streaming path — up to a whole layer plus slack of GTT allocated and never
   written, which also feeds `guard_capacity` and can turn a workable budget into a refusal.
5. **`Slot` writes to each tensor's own address.** It briefly stored `base = addrs[0]` plus
   pointer-subtracted offsets, which rested on the first placement being the lowest address —
   true for a bump allocator, promised by nothing, and an underflow into `base.add(huge)` if it
   ever changed, silently under `--release`. Writing to the address directly needs no ordering
   assumption and no arithmetic, and bounds each write by the placement it targets.
6. **`GlimmerLayerPin` is still not `Copy`, and the note now says what that does not buy:**
   `Bf16Weight` is `Copy` and the norms are bare pointers, so `let q = pin.layer(5)?.q` extracts a
   handle that outlives the borrow. The type narrows the mistake; it does not forbid it.
7. **967.942 MB, not 967.889 MB, per layer.** The figure was the CHECKPOINT's — with the four norms
   at bf16 — while the converter widens them to f32. `resident_bytes` prices that widening at 2.782
   MB two functions away, and 55.712 GB only reconciles with the corrected value, so the file
   contradicted itself. `..._at_the_shipped_widths` now pins all three totals as a test.

**The streaming source is still the mmap, not io_uring** — `ExpertSet`'s O_DIRECT path needs a
layer-blocked artifact `convert_glimmer` does not emit. `ponytail:` ceiling named at the call site.
**S5 must account for the page cache before quoting any bytes-from-disk number**, and note the
mmap has a second cost: a fault cannot return `Err`, so an NVMe error is SIGBUS rather than a
handled failure.

**Still owed at R1:** the attention-weights pin-vs-slot decision (§3 above) is unmade — whole
layers stream, including their 170.4 MB of attention weights — and it needs a decode to measure.

**G-R1 — met when** (a) tiny-model decode output is **bit-identical across every budget**
from all-resident to the floor — P4 as a gate, zero tolerance, and the tiny checkpoint CAN
exercise it (force a budget that pins only some of its 8 layers); (b) the schedule
off-by-one defect reddens it; (c) allocation at 55.7 GB with a real-machine budget fails
LOUDLY at build, not at layer 40 of the first decode; (d) a **>2 GiB single-tensor
alloc+copy test exists and passes** — `lm_head` is 2,689,662,976 bytes = 1.25× `i32::MAX`,
`DeviceBuf::new`/`copy_in_at` were traced size_t-clean by review but no test allocates past
2 GiB anywhere in the tree.

> **(a) AS WRITTEN IS NOT REACHABLE AT R1, and `tests/glimmer_residency.rs` carries the
> substitute.** It asks for DECODE output across budgets; there is no decode until S3, whose
> whole content is the layer loop. Waiting would ship R1's partition ungated, so the gate is
> per-layer instead: **`layer(l)` resolves to byte-identical tensors at every budget**, all
> twelve tensors, every layer, two passes per budget so a refilled slot is re-read. That is
> stronger than (a) where they overlap — it localises a wrong partition to the tensor rather
> than to the run — and strictly weaker in one way that matters: **a loop can consume correct
> bytes in the wrong ORDER, and only a decode sees that.** S3 still owes the end-to-end form,
> and this is not a substitute for it there.
>
> **(b) was WITHDRAWN, and that is the honest outcome rather than a gap.** It asked for a
> schedule off-by-one to redden. The test written for it re-implemented the slot map as a local
> closure with its own `const SLOTS = 2`, so it proved a property of `%` and could not observe the
> shipped constant at all — both reviews found that independently, and the doc claiming "reducing
> `SLOT_COUNT` to 1 reddens it" was false. It was then deleted rather than repaired, because
> finding 1 above dissolved the premise: the slot count is not what makes slot reuse safe, so
> there is no schedule property here to gate. **What replaces it is the invariant on `layer()`
> plus S5's obligation to add the fence** — and a real gate is possible without a decode (a
> long-running kernel over a slot's span, a host refill, assert the readback is clean), which S3
> should carry.
>
> (c) is `partition`'s floor refusal, gated with no device at every boundary including one byte
> under — and now at the SHIPPED widths too, because review showed several fixture-width
> assertions cannot fail there: 1 MiB of alignment slack is 99.6% of every fixture budget, so a
> floor charging zero slots still passed. (d) is `#[ignore]`d — it allocates 2.69 GB of GTT, so a
> suite count NEVER includes it and must not be quoted as if it did; run it with `--ignored` under
> the flock. It also tested the wrong allocator until review: `DeviceBuf`/`hipMemcpy`, while the
> pin places through `DeviceTier::place` into a `VmmBuf`. It now runs the pin's own path.

## S3 — the layer loop, against the contract from day one

**Serves P7 mainly; the loop is where every §9 trap either dies or ships.** Module
`glimmer_gpu.rs` (the seam argument in `glimmer-port.md` stands), consuming R1's contract —
written so that "all resident" is a budget value, never a code path.

What the loop must get right, each with its gate:

1. **Sandwich norms — the missing kernel.** Four per layer, post-norms on the BRANCH before
   the residual add, and they are CENTERED: `x*(1+w)` — **no kernel in the tree computes
   that form** (`rmsnorm_single`/`batch` are plain `x*w`). One new kernel (or a flag-free
   second entry point, the `rope_split_half` precedent — a wrong-form call site must not be
   one bool away), tolerance row measured from the anchor's norm buckets BEFORE it exists,
   S2-style.
2. **Weightless QK-norm + `qk_scale_factor` 3.87 on Q alone** — trap 2's territory; scored
   against `q_norm.out`/`k_norm.out` captures.
3. **The gate operand.** S2 proved the kernel and explicitly not the wiring; the anchor
   captures both `input_layernorm.out` and `attn.gate_proj.out` per (step, layer), so the
   loop is scored against `gate_proj.out` directly — the realistic trap-4 miswiring
   (`gate_proj` of the pre-norm residual or the post-attention norm) differs mostly by a
   scale and ONLY this capture catches it.
4. **Streams, not null.** `sigmoid_gate` and `logit_softcap` take the trailing stream now;
   the loop passes its compute stream at both call sites. A null there is the unordered-read
   bug `linalg.hip`'s swiglu note describes, and no fixture can see it — review round 2's
   finding, standing S3 contract.
5. **NoPE pattern [L,L,L,G]**, window 2048 ring on sliding layers, **two EOS ids**
   (`[200001, 200008]` — a scalar-EOS port stops on one), softcap present in the tail.

**G3 — met when** teacher forcing, greedy decode, and incremental-with-KV match the tiny
model at **zero tolerance**; a decode crossing position 2048 matches a from-scratch prefill
(the ring's first eviction); a pattern-shift defect reddens global layers only; **the
probability-space softcap check exists** — S2 proved argmax-invariance means no greedy gate
can see the softcap, so G3 compares softmax/NLL against the tiny reference and a
`softcap_off` engine run must redden it. All of G3 runs at every budget G-R1 sweeps, so the
loop is never green only-resident.

## S4 — real weights

**Serves P5 (the quality ladder) and closes what the tiny model cannot price.**

1. Convert the full checkpoint bf16-verbatim (exists) — then the **first quantized format**,
   chosen by arithmetic then priced by dNLL: int4+g128 at 13.65 GB/token is the only row
   that plausibly decodes at interactive speed on this GTT; fp8 at 26.51 is the quality
   anchor. `WMat`'s note applies: if a format addition ever pays the payload-struct hop,
   delete the two exemptions.
2. **Chat template, hand-ported and byte-pinned** — the artifact drops it (it lives only in
   the source checkpoint), GLM's drifted for months, and this is an "agentic" model: expect
   a tools block in the pin.
3. Bounded greedy run **read by a human** — `distinct`/repeated-block are banned instruments
   here (three investigations misled).
4. **dNLL ladder per format** from `bin/ppl` paired stats, the 5000-token corpus if the
   762-token one is underpowered — and the **softcap priced on trained logits** (the anchor
   provably cannot; S2's `ExactOnly` row says so), closing G3's IOU with a real number.
5. `tie_word_embeddings: false` — both 2.69 GB tensors ship; assert both are placed and
   DIFFERENT (a port that aliases them saves 2.7 GB and is silently a different model).

**G4 — met when** the full checkpoint round-trips bit-exact where verbatim, the template pin
holds byte-for-byte, a human has read the output, and every format in the ladder has a
paired-dNLL row whose interval does not straddle zero (or is recorded inconclusive).

## S5 — performance, each lever priced

**Serves P2, P3, P5. Nothing here starts until G4; every number lands in
`measurement/benchmarks.md` with its command line.**

| lever | why it is plausibly large | what decides it |
|---|---|---|
| **speculative decode (DFlash)** | dense verification reads each weight ONCE per pass, so N accepted tokens **divide weight traffic by N** — on a bandwidth-bound model that is a direct tok/s multiplier; break-even N>1.1 (arch §11), the inverse of GLM's MoE-union economics | port the 2.556 B drafter (separate checkpoint, 5-layer bidirectional cross-attn, borrows embed UNNORMED + lm_head); measure accepted-N on real prompts |
| **prefetch overlap** | the schedule is perfect (R1), so streamed bytes should hide entirely behind compute until NVMe saturates | measure ms/layer streamed vs pinned at fixed budget; the gap is the unhidden remainder |
| **format per residency class** | P2's hybrid lever, deterministic per budget: resident layers int4 (cheap compute), streamed layers a smaller format (bandwidth-bound side) | decide against P4 FIRST — output would vary across budgets, hybrid's documented defect in milder form; if taken, document as a mode, never a default |
| **NPU** | dense decode is one long sequential GEMV stream — the NPU-shaped workload; the npu-offload closure is **GLM-scoped by its own `scope:` field** | its own measurement, `scope: glimmer`, using the closure's method |
| **fusion** (gate into attend, softcap into head GEMV) | one pass over [rows][4096] / 202048 floats each | price AFTER the fixtures exist to catch a fused wrong answer; S2 refused both for exactly that reason |

**G5 — met when** tok/s is reported at ≥3 budgets × ≥2 formats with the witness discipline
(sole tenant, flock, GTT sampled), the one-line tok/s model above is validated or corrected
against those points, and each declined lever carries its measured reason here.

## Decisions this plan leaves open, deliberately

| decision | stage | what settles it |
|---|---|---|
| attention weights: pin always vs second slot class | R1 | floor arithmetic at int4/fp8, then a measurement |
| first quantized format | S4 | dNLL ladder + GB/token table |
| hybrid-format-by-budget: take it or refuse it | S5 | P4 stance, in writing, before any code |
| NPU | S5 | its own scoped measurement |

## What is NOT in this plan

S0–S2 (done, gated, reviewed — `glimmer-port.md` keeps the record); the vision tower (3.84
GB, out of scope until text decodes); serving-layer integration (`serving.md` is GLM-scoped;
Glimmer inherits it only after G5); K3, which proceeds independently on its own plan.
