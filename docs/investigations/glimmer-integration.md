---
scope: glimmer
status: live
verdict: What is LEFT of the Muse Glimmer-30B integration, re-planned 2026-08-12 against reference/principles.md after the owner corrected the port's central assumption — the pin is a function of free memory, so S1a's all-resident GlimmerPin ("a dense model has nothing to stream") violates P6 and dense makes streaming MORE load-bearing, not less: every weight is read every token (53.02 GB bf16 / 26.51 fp8 / 13.65 int4), there is no routed union to hide behind, and the resident fraction times bandwidth IS the tok/s model. Four stages remain: R1 residency contract + budget-aware pin (cyclic access makes LRU pathological — hit rate 0 at any deficit — and Belady degenerates to a STATIC prefix partition, so the policy axis collapses; output across budgets gated identical at zero tolerance on the tiny model), S3 layer loop against that contract from day one (sandwich norms x*(1+w) have NO kernel yet; gate operand scored vs gate_proj.out captures; streams passed, not null; G3 owes the softcap probability-space check nothing at S2 could make), S4 real weights (chat template byte-pinned, dNLL ladder per format, the 2.69 GB lm_head is 1.25x i32::MAX bytes and untested at every stage), S5 performance (speculative decode DIVIDES weight traffic by accepted length — the biggest single lever on a bandwidth-bound dense model; prefetch is PERFECT because the schedule is known before the run; NPU re-opened per P3, the closure is GLM-scoped). R1 IMPLEMENTED 2026-08-12: the contract is GlimmerPin::layer(l), one shape whether pinned or streamed; --max-mem is now ACCEPTED and run_glimmer reports the partition from the same arithmetic the pin uses; --cache-policy is still refused but for a NEW reason (the old one, 'a dense model streams nothing', became false the moment a budget could leave layers streaming). Four decisions recorded: NO ticket because the fill is a synchronous host memcpy and an always-satisfied dependency object is the hit-mask mistake asyncfetch.rs already paid for; SLOT_COUNT=2 as a correctness floor, not a tuning knob; GlimmerLayerPin deliberately NOT Copy so a stale streamed pin is a compile error; and the source is the mmap rather than io_uring because O_DIRECT needs a layer-blocked artifact convert_glimmer does not emit yet, so S5 must account for the page cache before quoting any bytes-from-disk number. G-R1(a) as planned was NOT reachable - it asks for decode output and there is no decode until S3 - so the gate is per-layer BYTE identity across every budget instead: stronger where it overlaps, and weaker in one way S3 still owes, since a loop can consume correct bytes in the wrong ORDER. Still open at R1: the attention-weights pin-vs-slot decision, which needs a decode to measure. Supersedes glimmer-port.md's S3+ sections ON SWITCH-OVER, which is the owner's call; S0-S2 records stay where they are.
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

### R1 as built — 2026-08-12

**The contract is `GlimmerPin::layer(l) -> &GlimmerLayerPin`.** Same twelve-field shape whether
the layer was pinned or arrived through a slot; the caller cannot tell, and that is P4 expressed
as a type. Four decisions worth the record, each of which could have gone the other way:

1. **No ticket, because the fill is synchronous.** Under HIP's unified addressing the tier is
   host-writable, so a fill is the same host `memcpy` `DeviceTier::place` performs and it has
   completed when `layer()` returns. Handing back an always-satisfied `Ticket` would be a
   second, host-side encoding of "is this ready?" that can never disagree with reality — which
   is precisely what `fetch/asyncfetch.rs`'s `Ticket` doc records the `hit: Vec<bool>` mask
   costing the GLM path. **S5 makes the fill asynchronous and must add a real ticket then**; the
   signature is shaped to accept exactly that change and nothing else.
2. **`SLOT_COUNT = 2`, a correctness floor rather than a tuning knob.** A launch reads layer `l`
   while the next fill targets `l+1`; one slot would overwrite bytes a kernel is still reading —
   the read-outlives-its-slot defect still open on the GLM arena. More slots buy prefetch DEPTH,
   which is worth nothing until the fill is async, and would cost `SLOT_COUNT × 967.889 MB`.
3. **`GlimmerLayerPin` is deliberately NOT `Copy`**, alone among the pin structs here. A copied
   pin of a streamed layer stays valid-looking after its slot is refilled; borrowing from a
   `&mut self` method makes holding a stale one a compile error.
4. **The streaming source is the mmap, not io_uring.** `ExpertSet` streams per-layer sidecar
   files whose blocks are aligned for O_DIRECT; a safetensors tensor starts wherever the header
   left it, so the same path needs `convert_glimmer` to emit an aligned layer-blocked output.
   `ponytail:` buffered I/O through the page cache, ceiling named at the call site. **Upgrade
   path: a layer-blocked artifact turns `Slot::fill` into an `AsyncFetch` submit.** Until then
   the page cache is doing the caching, and **S5 must account for that before quoting any
   bytes-from-disk number.**

**The globals stay resident at every budget, by arithmetic.** embed + lm_head + final norm are
5.380 GB against a layer's 0.968, and each is read once per TOKEN — streaming them frees 5.4 GB
and pays on every token, so they are in the floor and a budget below it is refused.

**`--max-mem` is now ACCEPTED for Glimmer** and `run_glimmer` reports the partition it implies
from `GlimmerPin::partition` — the same arithmetic the pin uses, so the operator's line cannot
disagree with the pin's split. **`--cache-policy` is still refused, for a new reason**: the old
one ("a dense model streams nothing, so there is no pool") became false the moment a budget
could leave layers streaming. The standing reason is that the policy question has one answer
here — cyclic access makes LRU evict exactly the layer needed next (hit rate 0), and Belady
degenerates to a fixed subset where every subset of size `k` scores `k/n`.

**Still owed at R1, and not done:** the attention-weights decision (§3 above) is unmade — the
current partition streams whole layers including their 170.4 MB of attention weights, which is
the simple thing and not necessarily the right one. It needs the floor arithmetic at the formats
that matter, then a measurement, and it cannot be settled before S3 gives it a decode to
measure.

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
> (b) is likewise arithmetic rather than a broken build: the slot map's correctness property is
> that it is injective over any `SLOT_COUNT` consecutive streamed layers, and the red proof
> asserts that a single slot collides on every consecutive pair — so reducing `SLOT_COUNT` to 1
> reddens it. (c) is `partition`'s floor refusal, gated with no device at every boundary
> including one byte under. (d) is `#[ignore]`d by default: it allocates 2.69 GB of GTT, which
> is not something a routine `cargo test` on a shared GPU should do; run it explicitly under
> the flock.

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
