---
status: live
verdict: The staged plan to make V4-Flash decode. S1 LANDED 2026-08-05 (.f4 repack bit-exact over 10.27 GB; a 137-golden CPU oracle with five measured blind spots). Corrects other-models.md from the real repo: experts are 148.25 GB native FP4 (138.1 GiB) so it DOES stream at ~83% residency, not "nearly fully resident"; 3.449 GB/token, since the shared expert is fp8 and resident, not FP4 and streamed; the partial fp8 KV act_quant is mandatory, not a --kv-fp8 to refuse (that flag does not exist); YaRN is per-layer, keyed to compress_ratio. DSpark/MTP is separable and out of scope.
---

# DeepSeek-V4-Flash-0731 → first decode, first benchmark

**Status:** staged, S1 launching 2026-08-04.
**End goal:** the model decodes coherent text under rivoli, produces a benchmark, and we can
say what its quality is and where its performance goes.

This is a hypothesis to test, not a spec to satisfy. **A well-evidenced negative — "this
stage cannot be done at acceptable cost, here is the measurement" — is a successful
outcome.** Several claims below are inferences from the reference source and are marked
`INFERRED`; disproving one is worth more than implementing around it.

---

## 0. Ground truth, and what it corrects

Weights: `/var/db/rivoli/deepseek-v4-flash-0731` (167 GB, 48 shards).
**Reference implementation: `inference/model.py` in that directory — 961 lines, and it is the
spec.** `docs/investigations/other-models.md` §3/§4 was written from the config and the
paper, before the repo was downloaded. Read `model.py` over the doc wherever they disagree.

Measured from all 72,317 safetensors headers (2026-08-04):

| role | bytes |
|---|---:|
| routed + shared experts (main model) | **148.25 GB** |
| MTP/DSpark experts | 10.34 GB |
| attention core (fp8) | 4.60 GB |
| embed / head / norms | 2.12 GB |
| KV compressor | 0.53 GB · indexer 0.28 · mHC 0.14 · router 0.11 |
| **total** | **166.88 GB** |

One routed expert (w1+w2+w3 incl. scales) is **13.37 MB**; top-6 × 43 layers = **3.449
GB/token** of stream traffic.

> **CORRECTED 2026-08-05, by S1a and S1b independently.** This said **4.02 GB/token**, and
> the row above lumped the shared expert in with the routed ones. Both errors are the same
> mistake: the shared expert is **not FP4 and not streamed**. On disk
> `shared_experts.w1.weight` is `F8_E4M3[2048,4096]` with `F8_E8M0[16,32]` scales — fp8 at
> 128×128 blocking, **25.17 MB**, because `MoE.__init__` passes `expert_dtype` only to the
> *routed* experts. It is resident, so it is not per-token traffic at all. The old figure
> counted it at the routed expert's 13.37 MB *and* as streamed. Residency is unchanged
> (~84% at `--max-mem` 115); the artifact total (148.25 GB = 147.17 routed + 1.08 shared)
> was reproduced exactly from S1a's converted artifact rather than from the index.
>
> Consequence for `.f4`: the container holds `n_experts` blocks and **no shared block**,
> unlike `.vq3`/`.i4`. A block past that boundary would be the wrong *arithmetic*, not
> merely the wrong bytes.

### Three corrections that change the work

1. **The streaming verdict inverts.** `other-models.md` §3 called V4 "nearly fully resident…
   would barely stream" from a ~120 GiB artifact against a ~115 GiB `--max-mem`. That 120 GiB
   is the **int3-vq** figure, and §6 then decided to keep **native FP4**. At the format
   actually chosen the experts are **148.25 GB = 138.1 GiB**, which does *not* fit the pool:
   ~83% capacity residency, against GLM's ~41%. It streams — less than GLM, but it streams.
   Per-token traffic is **3.449 GB, not 3.1** (see the correction above; this first said
   4.02). The two sections were computed at different
   bitrates and never reconciled.

2. **`--kv-fp8` is not simply "refuse".** The reference *deliberately* fp8-quantizes the KV
   entry, and does it **partially**: `act_quant(kv[..., :-rd], 64, …)` quantizes only the
   **non-RoPE dims [0:448)** at block 64 and leaves the RoPE dims **[448:512) in bf16**,
   to match QAT. So the correct behaviour is a *mandatory partial* quantization, not an
   optional whole-tensor one. Applying rivoli's existing whole-tensor `--kv-fp8` would both
   double-quantize and corrupt the positional dims — which is exactly the llama.cpp failure
   (`=`-loops and `"Mirror …"` noise, no crash). Omitting it entirely is also wrong: it
   diverges from what the model was trained against.

3. **YaRN is per-layer and keyed to compression.** `Attention.__init__`: a layer with
   `compress_ratio != 0` uses `compress_rope_theta = 160000` **with** YaRN
   (`original_seq_len = 65536`); a layer with ratio 0 uses `rope_theta = 10000` and **no**
   YaRN (`original_seq_len = 0` disables the interpolation branch). Two `freqs_cis` tables,
   selected per layer. `other-models.md` lists "YaRN" as one flat gap.

Also worth knowing, and absent from the doc: `swiglu_limit: 10.0` is a **clamped** SwiGLU
(rivoli's is unclamped — silent-wrong, not a crash); expert scales are `F8_E8M0` blocked
**32 along the input dim only** (the config's `weight_block_size: [128,128]` applies to the
*fp8 attention* tensors, a different scheme — and `F4_GROUP=32` in the shipped `.f4`
primitives already matches); and there are **three** `mtp.{0,1,2}` blocks each with a full
256-expert FFN, though the config says `num_nextn_predict_layers: 1`.

### Scope cut that makes this tractable

`Transformer.forward` (line 913) is the main path. **DSpark/MTP is `forward_spec` (929) and
is entirely separable** — `mtp.*`, `markov_head`, `confidence_head` and the `dspark_*` config
fields are all speculative decode, which rivoli gates off by default anyway. **Stages 1–4
target `Transformer.forward` only.** That removes 10.34 GB of weights and three subsystems
from the critical path. Do not build DSpark.

---

## S1 — Foundation. No GPU. Two agents, disjoint files.

The whole port's risk is that **every defect here is silent-wrong**: wrong router scoring,
missing QK-norm, missing output de-rotation, unclamped SwiGLU, mis-scaled FP4 — none crash,
all produce fluent wrong text. `distinct`/`longest repeated block` cannot see any of them
(CLAUDE.md, and it has misled three investigations here already). **So S1b builds the gate
before S2/S3 build the thing being gated.**

### S1a — artifact: config, naming, `.f4` converter

Owns: `src/artifact/{model,config,format,quant}.rs`, `src/bin/convert.rs`.

1. `ModelConfig` accepts V4. Today it refuses with `missing field kv_lora_rank`; the five
   absent fields are `kv_lora_rank`, `qk_nope_head_dim`, `v_head_dim`, `intermediate_size`,
   `first_k_dense_replace`. They are absent *because V4 is not MLA and has no dense layers* —
   so make them optional and derive an explicit architecture discriminant from
   `model_type`/`architectures`, rather than defaulting them to 0 and letting a GLM-shaped
   path run on a V4 config. **A default that silently produces a runnable-looking config is
   the failure mode to avoid here.**
2. Tensor naming. V4 uses the reference scheme, not HuggingFace's:
   `layers.{l}.ffn.experts.{e}.w{1,3,2}` (= gate/up/down), `.scale` not `.weight_scale_inv`,
   `attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b}`, `attn_norm`/`ffn_norm`, `embed.weight`,
   `head.weight`. Full table in `other-models.md` §7 — now verified against the complete
   index, and §7's "unverified absence" note is **resolved**: `ffn.gate.tid2eid` (3 layers,
   `I64[129280, 6]`), `attn.indexer.*` (21 layers) and the MTP block all exist.
3. `.f4` container. The primitives shipped in `6859e61` (e2m1 LUT, e8m0 decode, `F4_GROUP`
   32). Source experts are `I8[2048, 2048]` (FP4 nibbles, 2/byte) with `F8_E8M0[2048, 128]`
   scales — a **repack, not a requantization**: values pass through untouched. Assert that
   bit-exactly, on real tensors, both directions (§"what the reviews hunt" #9).
4. Attention tensors are already `F8_E4M3` + `F8_E8M0` scales at 128×128, which is the
   block size the resident path uses. Convert without requantizing.

**Deliverable:** a `.f4` artifact for at least layers 0–2 (enough for S1b to score), and the
byte accounting reproduced from the artifact rather than from the index.

### S1b — the oracle: a CPU transliteration of `Transformer.forward`

Owns: a new `src/bin/` or `tests/` oracle + its fixtures. **Must not touch `gpu.rs`,
`attn.rs` or any kernel.**

Transliterate the reference's main path to f32 on CPU, reading safetensors directly, and emit
**per-layer golden activations** for a short fixed prompt. This is the same pattern the repo
already trusts: `math.rs`'s frozen `route_into_pre` oracle and `glsl_numerics.rs`'s
transliterations, both `jscpd:ignore`d because being a verbatim copy is the point.

Cover, in reference order — `Attention.forward` is lines 490–548 and each of these is a
separate silent-wrong risk:

- `wq_a → q_norm → wq_b`, then **QK-norm** `q *= rsqrt(q.square().mean(-1) + eps)` — note
  this is RMS normalization with **no learnable weight**, applied after the unflatten to
  (heads, head_dim).
- RoPE on **the last 64 dims only** of q and of kv.
- `wkv → kv_norm`, then the **partial fp8 act_quant of dims [0:448) at block 64** (§0.2).
- The KV cache is a **ring**: `kv_cache[:, start_pos % win]` with `win = 128`, sized
  `window_size + max_seq_len // compress_ratio`, compressed region appended after `win`.
- `attn_sink` (`F32[64]`, per head) as an extra logit in the softmax **denominator only**.
- **Output de-rotation**: `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)`.
- Grouped low-rank output: `o.view(b, s, 8, -1)`, einsum against
  `wo_a.view(8, 1024, -1)`, then `wo_b`.
- Per-layer YaRN/theta selection (§0.3), and the `Compressor`/`Indexer` pair
  (`Indexer` exists **only** where `compress_ratio == 4`: 41 layers carry a compressor, 21
  of them an indexer. **`compress_ratios` has 46 entries for 43 layers** — the trailing
  three are the MTP blocks — and the ratio-4 indices are `[2,4,…,42]`, so the **last layer
  is ratio-4, not the ratio-0 tail the shape suggests**. S1b's first cut had 40 alternating
  entries and silently lost layer 42's compressor and indexer.)
- `Block.hc_pre`/`hc_post` (mHC, 4 streams, 20 Sinkhorn iters) and `Gate`'s hash path for
  `layer_id < 3` (`tid2eid`, selection bypasses the scores; the gate still produces weights).
- `Expert.forward` with `swiglu_limit = 10.0`.

**Beware the gate that cannot fail.** State, for each golden, what would have to be true for
it to reject a wrong implementation — then verify that by breaking the oracle deliberately
and checking it disagrees *at every case the defect touches and agrees at every case it does
not*. A model that disagrees everywhere proves nothing. Do this on CPU now; it is free here
and expensive later.

**The most-trusted case is the blind spot.** Layer 0 is the one everyone will check. It has
`compress_ratio = 0` — no compressor, no indexer, no YaRN, base theta. It is therefore the
*least* representative layer in the model. Produce goldens for a ratio-0 layer, a ratio-4
layer (with indexer) and a ratio-128 layer (without), and say which is which.

---

## S1 — LANDED 2026-08-05 (`3d32071`)

S1a: `ModelConfig` and `V4Config` are **separate types** refusing each other by name before
serde reads a dimension, so the five GLM-only fields stay *required* and no zero-filled
`ModelConfig` is constructible. `.f4` repack proved at **768 experts, 0 of 10,267,668,480
bytes differing**. Artifact for layers 0–2 at `/var/db/rivoli/v4-f4-l0-2`.

S1b: `src/v4oracle/` — **137 float + 18 int goldens** over a ratio-0, a ratio-4 and a
ratio-128 layer, from the real checkpoint. 27 tests. Its five *measured* blind spots and the
`.compress_idxs`-is-order-not-set warning are in the module docs; read them before trusting
a comparison.

Union clippy is clean for the first time since `9182ffc`/`6859e61`.

## S2 — the kernels. Three streams, gated against S1b's goldens.

**None of these touch `src/gpu.rs`'s layer loop.** Each delivers kernels plus a host harness
that scores them against the oracle, in the shape `tests/kernel.rs` already uses. Wiring the
layer loop is S3, deliberately: three agents editing the decode loop in parallel produces a
merge nobody can review, and a kernel that cannot be scored without the full forward pass is
a kernel whose first failure is a whole wrong model.

- **S2a — the MoE half.** `dot_f4` (e2m1 nibbles + e8m0 group-32 scales) and the
  `moe_gateup_f4`/`moe_down_f4` pair, **clamped** SwiGLU (`swiglu_limit = 10.0`), hash
  routing for layers 0–2 (`tid2eid`, `I64[129280,6]`; selection bypasses the scores, the
  gate still produces the *weights*), and mHC (`hc_pre`/`hc_post`, 4 streams, 20 Sinkhorn
  iterations). Note the shared expert is **fp8, not FP4** — a different kernel, already in
  the tree.
- **S2b — attention core.** MQA with one 512-d shared K=V entry for all 64 heads: the
  weightless QK-norm, RoPE on the last 64 dims with **adjacent-pair** complex packing (NOT
  rivoli's half-split — S1b flagged this), the partial `act_quant` on kv dims [0:448) at
  block 64, the KV ring, `attn_sink` in the denominator only, the output **de-rotation**,
  and the grouped `wo_a`/`wo_b`. Scoreable against the ratio-0 layers alone, which need no
  compressor.
- **S2c — compressor and indexer.** Both `Compressor` branches including
  `overlap_transform`, the `Indexer` (ratio-4 layers only), the two per-layer `freqs_cis`
  tables, and the window/compress top-k index generation. Scored on the ratio-4 and
  ratio-128 goldens.

**DECIDED 2026-08-05 — `wo_a` is bf16 in S2, not fp8.** It ships fp8+scale on disk, but
`convert.py` dequantizes it and `Attention.forward` does a **bf16 einsum** with no activation
quantization. An fp8 GEMV there would be faster and would *not* match the oracle bit-for-bit
— which would put an unexplained delta into every attention comparison S2b makes, so a real
bug and the format choice become indistinguishable at exactly the moment that distinction
matters most. Match the reference now; re-open fp8 in S4 as a **measured** perf lever with
the oracle available to price its error. This is the same discipline as not ranking on
free-running tok/s: do not let two variables move at once.

### The duplication gate cannot see `kernels/` — found 2026-08-05 by S2c2

**`f2e4m3_rne` and `fast_round_scale` were written twice, independently**, by two agents who
never saw each other's work: `kernels/common.hpp` (S2a, `03b956f`) and `kernels/mla.hip` as
`v4_f2e4m3_rne` / `v4_round_scale` (S2b, `e76e0d4`). Same function, same subnormal-tie
argument, same bit surgery, arrived at separately — because rivoli's own `f32_to_e4m3`
rounds half-away-from-zero where V4 was trained against RNE, so both needed a replacement.
`v4_block_sum` and `v4_rbf16` are a third pair.

**`build.rs:618` is `const SCAN: &[&str] = &["src", "tests", "build.rs"]`.** `kernels/` is
not scanned, and the doc comment above it says so — "this repo's own Rust". So the gate this
repo treats as absolute (*"Duplication is a build error… `.jscpd.json` carries no
`threshold`, so there is no budget"*) is **structurally blind to the HIP and GLSL sources**,
which is where the most numerics-sensitive code in the engine lives. The fp4-ownership
decision prevented the duplication it named and not the one beside it.

Not a bug in the gate — a known scope whose implication nobody had drawn. **S3 lifts one
copy of each into `common.hpp`**; it needs `mla.hip` edits that no S2 agent was permitted to
make. — *DONE 2026-08-05; §"Landed by S3" has the result. Two things are worth carrying
forward: the lift moved `f2e4m3_rne` out from under `mla.hip`'s `contract(off)` and an ISA
diff caught the FMA that appeared, and the first pass still left a FOURTH hand copy
(`v4_act_quant`'s clamp-and-encode loop) eight lines below its own note saying the
duplication was gone. Both were found by review, neither by a test.* Whether jscpd should
scan `kernels/` at all is a separate question: the two backends'
ABI walls are already `jscpd:ignore`d for being deliberate copies, so turning it on there
would need its own exemption pass first.

### A hole S3 inherits unless it acts — recorded 2026-08-05 by S2c

**The shipped goldens at `index_topk = 512` are set-invariant, and a set comparison against
them cannot see a wrong ranking.** Measured at real weights, layer 2, 13 tokens:
`indexer_truncated = 0` and the selected sets are `[[-1,-1,-1], [-1,-1,13], [-1,13,14],
[13,14,15]]` — determined **entirely by the causal mask**. `IndexerNoWeights` and
`IndexerNoRelu` both move `.indexer_scores` and leave the set bit-identical. So the gate
accepts an arbitrarily wrong ranking, confirmed rather than argued.

Lowering `index_topk` fixes it — `indexer_truncated = 13`, and row 12 then selects a
strictly *older* block than row 11, which is the assertion S2c pins, because a monotonic
pick would just be re-testing the causal mask and calling it a ranking test. But reaching
truncation at the **shipped** 512 needs **≥2052 tokens**.

Consequence: the oracle's ranking code is proven discriminating; the *shipped goldens* are
not. **Any stage scored only against them inherits the hole.** If S3 leans on
`.compress_idxs`, it needs either a long-prompt golden or the lowered-`index_topk` probe
wired in. This is the same shape as the recorded trap where a `--attn dsa` A/B under 2048
tokens passes vacuously.

Related, from S2b: `Io.freqs` is a raw pointer that cannot distinguish the ratio-0 table
from the YaRN one — mixing them is `Defect::RopeNoYarn`, fluent and wrong. And `Scratch`
sizing is unchecked: a scratch allocated for decode and handed a `Prefill` overruns every
buffer.

### The compressor gate cannot resolve `act_quant`'s arguments — DECIDED 2026-08-05

S2c2's compressor kernel is verified: three of four cells **bit-identical** to the oracle,
the fourth off by a single e4m3 boundary flip on **5 of 32768 elements (0.0153%)** — with
`rope_tail = 1` proving the arithmetic upstream of quantization is exact, `quant_dims = 16`
being exactly one e4m3 step, and `want=3.5 got=3.25` being adjacent codes in the `[2,4)`
binade. Four independent predictions, none tuned. The original "16 ULP" was a **unit error
in the harness**, not a kernel defect.

What remains red is the defect sweep, and it is a **coverage** result:

| defect | separation |
|---|---|
| `CompressorNoOverlap` / `RopeHalfSplit` / `CompressorRopeAtBlockEnd` / `RopeAllDims` | 31,324 – 32,848 |
| `SkipKvActQuant` | **8** |
| `KvActQuantNoRoundScale` | **22** |
| `KvActQuantBlock128` | **INERT — covers it not at all** |

Every *compressor* defect separates enormously; only the three `act_quant` **argument**
defects sit at or below the quantizer's own step, because they perturb it by less than one
of its steps.

> **CORRECTED 2026-08-05 by S2c-indexer, on hardware — the SECOND correction to this
> section, and it should be read with the first.** The coordinator has already recorded that
> the non-coverage is **13 cells**, not the handful this table lists, `NoBf16Rounding` at
> `sep=16` on both ratio-128 cells included. This adds the part that correction does not
> carry, and one detail it rounds off.
>
> **The scale-invariance argument below is too strong, and the data disproving it was in the
> same output it was written from.** It says `KvActQuantBlock128` is inert "for a reason no
> threshold can fix" — ue8m0 scales are powers of two, e4m3 is exactly scale-invariant under
> them, so *no* gate can see it at *any* threshold. That holds at three of the four cells. At
> `ratio4/prefill` it does not:
>
> ```
> ratio4/prefill clean:              max=16  differing=5/32768
> ratio4/prefill KvActQuantBlock128: max=16  differing=6/32768   want=3.5 got=3.25
> ```
>
> Same max, **different element count** — so `broken != clean` and the defect is observable.
> Invariance is exact only while both blockings keep every value inside e4m3's range; at a
> rounding boundary they diverge, which is what `3.5` against `3.25` in the `[2,4)` binade is.
>
> **Why it survived three readings is the part worth keeping.** The claim is an argument from
> first principles — powers of two, therefore exactly invariant — and it is *nearly* true.
> That was enough that its author, the coordinator quoting it as settled, and S2c-indexer on
> first pass all accepted it without checking it against a `differing=` count already on
> screen. A derivation that is right in kind and wrong at the boundary is harder to catch than
> a wrong number, because nobody looks.
>
> **The detail:** this table lists **three** `act_quant` defects. Four are measured under the
> floor — `KvActQuantWholeTensor` (29 and 38) appears nowhere here.
>
> The live list is `BELOW_RESOLUTION` in `tests/v4_compress_kernel.rs`, which asserts each
> entry's measured separation exactly. Not restated here, so the numbers have one home.

**DECISION: record this as named non-coverage. Do NOT lower `RESOLVABLE`.** Lowering it to
admit `sep=8` is the budget-not-measurement move S2c2 spent this round undoing, and it would
buy nothing real: `KvActQuantBlock128` is inert *for a reason already in this document* —
ue8m0 scales are powers of two and e4m3 is exactly scale-invariant under them, so **no gate
scored against this oracle can see that defect**, at any threshold. It is S1b's documented
blind spot reappearing one layer out, not a new gap.

`act_quant` is S2b's kernel and is verified there: its 8/8 run includes the subnormal-tie
fixture engineered to separate RNE from half-away-from-zero, which is the property that
actually matters. So the honest statement is that **this gate verifies the compressor and
delegates `act_quant` to S2b's** — not that `act_quant` is unverified.

> **How the RED suite shipped — the coordinator's error, recorded 2026-08-05.** I merged S2
> having run `docs`, `invariants` and `v4_oracle` but *not* the three kernel suites, and
> wrote "measured" in the merge commit. The full suite had already failed to complete twice
> that session — once from a GPU contention error of mine, once from the known `gpustream`
> hang — and the merge went ahead anyway. The rule this breaks is already in CLAUDE.md; what
> is new is that a **green subset was read as a green suite**.
>
> A second error of mine sits on top of it: I then reported the failing set as "exactly the
> cells this section rules out", having read the DECIDED table as a record of what *fails*
> when it was a record of what had been *considered*. The measured figure is 13 cells, and it
> is S2c-indexer's below, not mine.

### The Hadamard basis order — SETTLED 2026-08-05 by S2c-indexer, and it was load-bearing

S1b flagged `numerics::hadamard_rotate`'s basis order as **its single highest-risk
inference**: `model.py:256` imports `hadamard_transform` from `fast_hadamard_transform`, the
package is not vendored with the checkpoint, and `inference/requirements.txt` does not pin a
version — so the order could not be read off the reference. It was resolved by reading the
**package**, not by agreeing with the oracle:

1. `fast_hadamard_transform.hadamard_transform`'s own docstring: *"Equivalent to
   `F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale`"*. The package ships that
   equivalence as executable code (`hadamard_transform_ref`) and its test suite asserts the
   CUDA kernel matches it **elementwise** at `dim = 128` — which is what excludes a
   permutation, since a reordered output disagrees maximally on random input.
2. Checked in **both** sdists that could satisfy the unpinned requirement (1.0.4.post1 and
   1.1.0); docstring and reference body are character-identical, so the missing pin does not
   reopen it.
3. `scipy.linalg.hadamard` is **Sylvester's construction** — natural/Kronecker order, not
   sequency — by its docstring and its source.

The oracle was **right**. `tests/v4_hadamard_basis.rs` now pins `hadamard_rotate` to an
explicitly-constructed Sylvester matrix bit-for-bit and carries the chain.

**The part worth banking is that nothing shipped could have told us.** `hadamard_rotate` was
patched to the sequency order — a pure permutation, still orthogonal, still symmetric — and
the whole CPU suite re-run: `v4_oracle.rs` 27/27, `v4_compress.rs` 7/7 and
`v4_compress_probe.rs` 4/4 **all passed**, the ranking probe included. Two reasons, and the
second generalises past this defect:

* `hadamard_is_its_own_inverse` cannot separate the candidates because **both** orderings are
  symmetric, and symmetry is exactly the condition for involution. Now asserted.
* **Every oracle test is self-relative.** A `Defect` arm and its baseline both run through the
  same primitive, so an error the oracle and its own defect matrix share cancels. That is the
  limit of comparing an implementation to an oracle, and the only way past it is to compare
  the oracle to something that did not come from the oracle's source — the same gap
  `v4compress.rs`'s `jscpd:ignore` region names for `freqs_cis`/`window_topk`/`compress_topk`,
  which is **still open**.

**The quantizer is what makes the basis order observable at all**, and that is the durable
lesson: an orthogonal rotation is invisible to a dot product, so before `fp4_act_quant` there
is nothing to be right or wrong about. The order becomes load-bearing only through *which
coordinates share a block-32 scale*. Any future change to the fp4 blocking re-opens this
question; a change to the rotation alone does not.

It mattered. Measured in the **bf16 score** `Indexer.forward` computes, over 64 row pairs:
before quantization the two orders are **bit-identical on all 64** (`(Hq)·(Hk) = q·k` for any
orthogonal `H`); with `fp4_act_quant`'s block-32 grouping they differ on **56 of 64**, by a
median **7%** and a maximum **104%** of the larger score, against a ~0.8% bf16 step. The
whole mechanism is *which coordinates share an fp4 scale*.

Two review findings from this round are worth keeping as method. The negative control
("sequency fails the same gate") **did not call the function under test** — it compared two
locally-built matrices to each other, and so would have stayed green under the exact defect
it claimed to catch; all three reviews found it, and the gate and its control are now one
test so a rename cannot separate them. And the first cut scored the impact in "bf16 ulps"
using `|v| · 2^-8`, which is one binade wrong — 8 significand bits give a spacing of `2^-7`
of the binade *base* — and that inflated figure had already been copied into three documents.
Counting distinct **bf16 scores** needs no ulp arithmetic at all, and is the quantity the
selection actually reads.

### The indexer's device half — S2c-indexer, 2026-08-05. MEASURED.

`kernels/v4indexer.hip` (`v4_indexer_spread`, `v4_indexer_score`), the fp4 activation
quantizer in `common.hpp` (`f2e2m1`, `fp4_quant_roundtrip`), `Geom::indexer`, and
`tests/v4_indexer_kernel.rs`. **8/8 on hardware, every comparison BIT-IDENTICAL:**

| comparison | result |
|---|---|
| `indexer_spread` vs the oracle's `hadamard_rotate` + `fp4_act_quant_inplace` | **1152/1152** |
| the same over 60 binades of block scale | **7680/7680** |
| `indexer_score` on the checkpoint's own compressed KV | **48/48** |
| `indexer_score` base / no-weights / no-fp4 | **20/20** each |

Read the score rows narrowly. They are against a host transliteration of `model.py:425-427`,
**not** against `Oracle::indexer`, which still carries the confirmed head-sum defect below —
so they are an arithmetic-and-plumbing verdict, not a correctness verdict. The **spread** rows
are against the oracle and are a real verdict.

**The `contract(off)` pragma is verified in the ISA and was run RED FIRST**: 1
`v_fmac_f32_e32` with the pragma removed, 0 with it restored. The counts and the counting
trap that comes with them — a naive `v_fma|v_mac|v_mad` grep reports 10 against 9, because 9
of those are pure ADDRESS arithmetic the pragma neither does nor should touch — are stated
once, at the pragma in `kernels/v4indexer.hip`. Worth knowing elsewhere: `mla.hip`'s
equivalent note counts "fma-class" without excluding integer MADs, so its 14 may be
similarly inflated.

**Requirement 5 is discharged.** `Geom` split into `GeomAbi` (the `repr(C)` mirror, still
28 bytes with its layout assert) plus a `Quantize` field, and `compress` matches on it
exhaustively with no wildcard. `Geom::indexer` refuses every `LayerKind` but `Overlap`,
since an `Indexer` exists only at ratio 4. The hazard is that
`Geom::attention(Overlap, index_head_dim, …)` and `Geom::indexer(Overlap, index_head_dim, …)`
agree on **all six integers the kernel sees** — no dimension guard can separate them, which
is why the finish is a field and not an argument.

**Three findings from review that no amount of running would have surfaced sooner:**

1. **FMA contraction would have broken the bit-exactness claim.** `dot += q[i]*kv[i]`
   contracts to `v_fmac_f32` under hipcc's default `-ffp-contract=fast-honor-pragmas` — one
   rounding per term where the host does two. Every comparison would have failed, and failed
   *looking like a numerics bug*. `#pragma clang fp contract(off)` added, scoped to the
   scoring kernel; `mla.hip`, `linalg.hip` and `attn.hip` already do this and mla.hip records
   an ISA verification. **That verification has not been repeated here** — it needs the ISA
   dump, which is a build-time step, and is the first thing to do on GO.
2. **A guard called unreachable was reachable.** The spread launcher's `d % 32` check was
   commented as unreachable behind the power-of-two check; `d = 16` clears the latter and
   fires the former. Comment corrected, case added to the test.
3. `v4i_rbf16` would have been the **third** copy of `bf16f(f2bf16(x))`. Lifted to
   `common.hpp::rbf16` instead — three copies down to two without touching `mla.hip`.
   Requirement 11 now covers the **block-quantize loop** as well (`mla.hip::v4_act_quant`,
   `linalg.hip`, and this file are three spellings of seed-amax → `fast_round_scale` →
   roundtrip).

**Two things this stage does NOT cover, both named rather than papered over:**

* **The score chain is gated against a host transliteration of `model.py:425-427`, not
  against `Oracle::indexer`.** The oracle computes that chain internally but exposes neither
  the roped-and-spread `q` nor the scaled `weights`, so the kernel cannot be handed its
  intermediates. That makes it a *second* transliteration, and a misreading shared with the
  oracle is invisible — the same gap `v4compress.rs`'s `jscpd:ignore` region names for
  `freqs_cis`. **Making `Oracle::linear` public would close it**, and is left to whoever owns
  that file. The compressed KV the comparison runs on is real (`CompState::cache` after the
  oracle's own indexer compressor).
* **An oracle fidelity bug — CONFIRMED 2026-08-05 by the coordinator, on CPU torch.**
  `Oracle::indexer` sums the per-head products as a bf16 RUNNING fold,
  `acc = bf16(acc + bf16(dot·w))`. `torch.sum` over a bf16 tensor accumulates through
  `acc_type` — **f32** — and rounds **once**; only the output dtype is bf16.
  `torch bf16 .sum() == f32-accumulate-round-once` measured **True**. Blast radius is
  `.indexer_scores` and `.compress_idxs`: line 1241 is the only running-fold site in
  `forward.rs`.

  > **CORRECTED 2026-08-05 by S2c-indexer, against the reference.** The magnitudes first
  > recorded here (62.6% of scores disagreeing, mean signed Δ −0.0048, "a systematic downward
  > drift ~70× larger than for signed summands") were measured on the premise that the
  > summands are **non-negative** — "relu'd, times *positive* weights". They are not.
  > `weights_proj` is a bare `ColumnParallelLinear` with **no activation** (model.py:400),
  > scaled only by the positive scalar at :424, so `weights` is **signed** and
  > `relu(dot) * weight` is signed. The summands can cancel, and no claim about the
  > *direction* of the error survives; the quoted percentages describe a distribution the
  > model does not have. **The fix is unaffected** — `acc_type` is a property of the
  > reduction, not of its input — and the error still compounds rather than averaging out,
  > which is what `tests/v4_indexer_kernel.rs::host_score_accumulates_in_f32_not_bf16` pins:
  > 63 of 64 terms at a quarter of a bf16 ulp vanish entirely, 1.0 against 1.125.

  **Not settled by the fix:** the bf16 fold pinned the summation ORDER as a side effect and an
  f32 accumulator does not. Torch's reduction is vectorized and tree-shaped, so its partial
  sums differ from an ascending fold and can land either side of the final rounding. The
  kernel and its host reference agree with each other exactly; **neither is pinned to torch's
  ordering, and nothing in this repo could tell.**

  **The kernel and `tests/v4_indexer_kernel.rs::host_score` were corrected to match torch on
  2026-08-05; the oracle's own fix is owned by whoever owns `src/v4oracle/**`.** Until it
  lands the two disagree, and a comparison against the current indexer goldens is not
  evidence in either direction. Note the shape: this is the SECOND instance in one stage of
  the same root cause the Hadamard finding exposed — every oracle test is self-relative, so
  an error the oracle shares with its own defect matrix cancels, and only a comparison
  against something outside the oracle's own source can see it.

**e8m0 (requirement 15): the premise needs correcting before the gap can be closed.** The
indexer consumes **no e8m0 scale bytes at all** — its weights are fp8 (`wq_b`, which ships a
`.scale`) and bf16 (`weights_proj`, `wkv`, `wgate`); there is no packed fp4 weight tensor on
this path, so `dot_f4_wave_r`'s `e8m0f` is never called. Verified against the checkpoint
index: layer 2 carries exactly seven `attn.indexer.*` tensors and only `wq_b` has a `.scale`.
What this path *does* exercise is the same exponent domain through `fast_round_scale`, so
`fp4_block_scale_covers_sixty_binades_not_two` sweeps 60 binades of block scale against the
2 the suite reaches today. **`e8m0f`'s decode of a scale BYTE — including `0x00` and `0xff` —
remains uncovered, and nothing on the indexer path can reach it.** Requirement 10 is still
the only thing that will.

### `v4_compress_kernel` was RED from the S2 merge — fixed 2026-08-05 by S2c-indexer

Reproduced at `ae2dd33` before touching anything: **6 passed, 2 failed, exit 101**. Both
failures were bookkeeping in the sense that no kernel was wrong, and neither was bookkeeping
in what it revealed.

**1. A decision written into prose but not into the assertion.**
`each_in_scope_defect_is_further_from_the_gpu_than_the_clean_oracle_is` demanded
`sep >= RESOLVABLE` on cells the *"compressor gate cannot resolve `act_quant`'s arguments —
DECIDED 2026-08-05"* section above had already ruled out. The decision stands and
`RESOLVABLE` was **not** lowered — that is the budget-not-measurement move that section spent
a round undoing. The non-coverage now lives at the assertion as `BELOW_RESOLUTION`, an
**expected value, not a skip**: each entry must reproduce its measured separation *exactly*,
so a cell that gains resolution fires; each entry must be *reached*, so a dead one cannot
absorb a regression; and an unrecorded cell below the floor still fails.

**The measured non-coverage is broader than that section recorded** — 13 cells, not the
three it tabulated at one cell. Two it names nowhere: `KvActQuantWholeTensor` (29 and 38),
and `NoBf16Rounding` at `sep=16` on **both** ratio-128 cells, which is exactly one e4m3 step
and is not an `act_quant` argument at all. The full table is in `BELOW_RESOLUTION`; it is not
restated here, so the number lives in one place.

**And that section's scale-invariance argument is not universal.** It calls
`KvActQuantBlock128` inert "for a reason no threshold can fix" — ue8m0 scales are powers of
two, e4m3 is exactly scale-invariant under them. True at three of four cells. At
`ratio4/prefill` the defect is **live**: `sep=16`, 6 of 32768 elements, `want=3.5 got=3.25`,
adjacent codes in the `[2,4)` binade. Invariance is exact only while both blockings keep
every value inside the format's range; at a rounding boundary they diverge. An entry reaches
`BELOW_RESOLUTION` only if `broken != clean`, so that row is itself the disproof of "at any
threshold".

**2. A guard working correctly, and it was telling us something.**
`the_ratio_0_rope_table_reproduces_the_no_yarn_defect_exactly` fails at `ratio4/decode` with
a **perfect** impersonation — max=0, bit-identical against the defect-injected oracle — and a
separation from the clean oracle of only **8 codes**, half an e4m3 step. The anti-vacuity half
is right to reject that: the cell cannot distinguish "consulted the wrong table" from "did not
consult the table at all".

**`RopeNoYarn` is S3 requirement 4** — `Io.freqs` cannot tell the ratio-0 table from the YaRN
one — one of the five defects here that produce fluent wrong output. So the honest record is:
`ratio4/decode` is **non-separating at a measured sep=8, and this suite cannot see requirement
4 at RATIO-4 decode** — not "at decode", since `ratio128/decode` is unrecorded and still
required to separate. `ratio4/prefill` separates at **31,215** and is still required; that
cell is what gates the requirement. Not papered into green.

Every one of the four new gates was proved able to fire by deliberate break before the fix
was committed — a changed recorded value, a removed record, a dead record, and the no-yarn
expected value.

### Convergence between two reviews is not confirmation — S2b, 2026-08-05

S2b committed with two of three reviews, judging the third's question already answered. It
was wrong, and the shape of the miss is worth more than the bug.

The correctness reviewer found that **`head_dim == qk_rope_head_dim`** — "rotate the whole
head", an ordinary-looking config — builds a valid `Dims` and passes *every* check, because
`(512 - 512).is_multiple_of(64)` is `0.is_multiple_of(64)`, which is **true**. That is the
same `is_multiple_of`-admits-zero property the zero-extent sweep was added for, landing on
the one extent the sweep structurally cannot reach: `head_dim - rope_head_dim` is
**derived**, no config field holds it, and it is what `act_quant` sizes on. It surfaced as
opaque guard code 1001 at first launch — precisely the failure the sweep exists to prevent.

So the previous commit's claim that the test "proves MEMBERSHIP: every extent the kernels
index with is in that list" was false when written. Nine extents, not eight.

**The generalisable part:** ponytail and code-quality both converged on "the list is 6 of 8"
— a *counting* error. Only the reviewer tracing execution paths found that the list itself
was the wrong list. Two reviews agreeing reads like confirmation and is not; they can share
a blind spot exactly as an oracle and an implementation can. Run the third.

### `debug_assert!` is dead in every build this project runs — found 2026-08-05

`Cargo.toml`'s `[profile.release]` sets `lto` and `opt-level` and **no `debug-assertions`**;
there is no `.cargo/config.toml` overriding it, and CLAUDE.md prescribes `--release` for every
build, test and clippy run. So all **36 `debug_assert!` occurrences in `src/`** are compiled
out of every binary anyone here has ever run. They are not weak checks; they are absent ones.

Distribution: `artifact/quant.rs` 17, `gpu.rs` 4, `v4compress.rs` 2, `artifact/model.rs` 2,
`fetch/stream.rs` 2, and one each in `fetch/asyncfetch.rs`, `backend/vk.rs` and the three
`v4oracle` files.

Two of them are load-bearing by their own documentation: `v4compress.rs`'s doc says they are
"what ENFORCES the bsz=1 scope cut". They enforce nothing. That is this repo's most common
review finding — *a comment asserting a check that does not exist* — reproduced 36 times by a
profile setting rather than by any one author.

**RESOLVED 2026-08-05 by changing the habit, not the code.** Two repairs were considered and
both rejected. Setting `debug-assertions = true` on `[profile.release]` was rejected because
that profile is what every number in `docs/measurement/benchmarks.md` was measured under, and
bounds and overflow checks on the hot path change the thing being measured. A `debug` Cargo
feature with a `debug_check!` macro was written and then **reverted**: a feature cannot set
`debug-assertions` (it is a profile flag), so it required rewriting all 32 call sites to a
private macro — 32 sites of churn, a second spelling to learn, and a macro whose only job was
to reimplement a profile that already exists.

The actual defect was the instruction in CLAUDE.md, which prescribed `--release` for *every*
build, test and clippy run. Cargo's dev profile already sets `debug-assertions`. So the rule
is now: **develop on the dev profile, and use `--release` for benchmarks and performance
evaluation only.** The 32 `debug_assert!`s are live again for anyone following it, with no
code change at all.

What remains true and worth carrying: a `debug_assert!` fires only for someone who follows
that rule, so a check that must hold in a shipped binary is an `assert!`/`ensure!` and pays
its cost. `v4compress.rs`'s pair — whose doc calls them "what ENFORCES the bsz=1 scope cut" —
should be promoted on that argument, not on this one.

### FMA contraction is a SECOND uncontrolled source of oracle-vs-kernel divergence — 2026-08-05

`build.rs:67` passes hipcc exactly `--offload-arch=<arch> -O3 -fPIC`. **There is no
`-ffp-contract` flag anywhere in the build**, and clang's HIP default is
`-ffp-contract=fast`. So every kernel in this tree contracts multiply-add into FMA unless
its own file blocks it — `mla.hip` does, in one place, deliberately; nothing else accounts
for it.

Measured by the head-tail stage on the mHC blend at `dim 4096 × s 8`: **3 disagreements
against a plain multiply-add host reference, 0 against `mul_add`**. Its first bitwise
assertion was written against the *non*-contracted form, which works out to roughly a
**1-in-5 spurious red per run** — and a spurious red there is indistinguishable in a log from
a real regression.

This sits *on top of* `wave_sum` re-association, which this port already knew about. Two
independent sources, and the build controls neither. A host reference for any contracted
expression must use `mul_add`, and the assertion should carry a bound that fires if
contraction ever stops rather than silently passing.

### `assert_close` over bitwise at real dims — RETRACTED and replaced, 2026-08-05

An earlier instruction here — relayed by the coordinator — said the device-side head gate
must be **bitwise, not `assert_close`**, on the argument that both mHC-rsqrt defects
(4.899e-3 and 4.284e-4) sit under one bf16 ulp (7.81e-3) and a tolerance near the bf16 floor
misses them. **That was derived at toy dims and is wrong at real ones.** At `dim 4096` a
*correct* wave-reduced kernel already differs from the oracle on ~0.08% of bf16 elements, so
a bitwise gate rejects correct code.

The hardware settled it: the per-copy rsqrt defect was injected into the kernel and **caught
at dim 256 and 512, and not at 1024**. So the resolution is a property of the dimension, not
of the tolerance, and the right instrument is a defect ladder run at the dims where the
defect is separable — not a bitwise assertion at the dims where it is not. This is
requirement 16 ("toy-dim bit-exactness does not predict bit-exactness at depth") arriving
from the other direction: toy-dim *separability* does not predict separability at depth
either.

### Integration checkpoint — VERIFIED on the merged tree, 2026-08-05

`c4367a9` (indexer) + `590cd65` (loading) merged into the integration branch and then
**re-run here**, under `flock`, with the KFD witness re-checked *inside* the lock (the lock is
advisory; a foreign holder is invisible to it alone):

| suite | result |
|---|---|
| `v4_oracle` | 27 passed |
| `v4_attn` | 8 passed |
| `v4_kernel` | 17 passed |
| `v4_compress_kernel` | 8 passed (was 6 passed / 2 failed / exit 101 at `78796eb`) |
| `v4_indexer_kernel` | 8 passed |
| `v4_pin` | 1 passed — union over BOTH fixtures |
| `v4_loading` | 10 passed |
| `invariants` / `docs` | 1 / 2 passed |

**82 tests, overall rc=0**, plus union clippy at 0 findings. This is the check `78796eb` did
not get: a green subset was written up there as a green suite, and the kernel suites were in
fact red. The `v4_pin` row is worth reading precisely — it is one test, and it passes only if
`all.hash && all.scored && all.indexer && all.compressor_only`, which no single fixture can
satisfy. A machine holding only `v4-f4-l0-2` fails it rather than reporting coverage it lacks.

### ALL FIVE PREREQUISITES LANDED — 117 tests green on the merged tree, 2026-08-05

Four branches merged with **zero conflicts** and re-verified here, per-binary, under `flock`
with the KFD witness taken inside the lock:

| | | | |
|---|---|---|---|
| `v4_attn` 8 | `v4_kernel` 17 | `v4_compress_kernel` 8 | `v4_indexer_kernel` 8 |
| `v4_pin` 1 | `v4_hadamard_basis` 4 | `v4_attn_host` 9 | `v4_compress` 7 |
| `v4_oracle` 32 | `v4_loading` 10 | `v4_artifact` 2 | `v4_compress_probe` 4 |
| `v4_head_tail` 4 | `invariants` 1 | `docs` 2 | **117 total, rc=0** |

Union clippy silent. The five things S3 found missing when it tried to wire the loop —
the `.f4` reader, the V4 resident loader, bf16 `embed`/`lm_head`, `hc_head`, and
compressed-layer attention — all now exist and are gated.

**`cargo test --release --features rocm` HANGS and cannot be quoted.** Reproduced
2026-08-05: **zero `test result` lines in 10 minutes**, stopping at `Running unittests
src/lib.rs` — CLAUDE.md's recorded intermittent `gpustream` hang. Teardown was clean (0 KFD
holders, flock free), so it costs time rather than wedging the device. **The per-binary sweep
above is the replacement and runs in under a minute.** Any suite-wide count quoted from that
command — including "243 rocm tests green" earlier in this session — was almost certainly
per-suite, and should be read that way.

**Two device-free debts are owed, and are acceptance criteria for the layer loop.** Both
were traded for an enforcing construction whose only correct home is the loop; landing them
anywhere else produces a `pub fn` with no caller, which is what `compress_slot` *was* when it
was deleted.

| owed | cost until paid |
|---|---|
| `Io` built by something that takes `LayerKind` and calls `rope_for_layer` itself | nothing detects `Defect::RopeNoYarn`, on the one cell measured invisible to the numeric gate |
| the compressor's placer computing `window + start_pos / ratio`, with a test | requirement 2 implemented nowhere, asserted nowhere; appending is right only until a step is skipped |

**And the compressed-layer end-to-end test is still unwritten.** `tests/v4_attn.rs` pins
`LAYER` to ratio-0, so the `io.cache` tail layout and compressed columns reaching
`sparse_attn` are executed by nothing, on **41 of 43 layers**. `Cell` in
`v4_compress_kernel.rs` is compressor-only and the oracle's `attention` is private, so an
oracle comparison must drive `run_layer`: load a real layer's fp8 attention set and its
compressor, run `compress`, place its output at both destinations, then `attention`.

## S3 — wire the layer loop, first decode.

**Requirements banked from S2, each measured rather than supposed.** These are conditions on
the wiring, not suggestions:

*Correctness, will produce fluent wrong output if missed:*
1. **`rmsnorm` must bf16-round its output.** V4's `RMSNorm` returns bf16; rivoli's does not.
   — *CORRECTED 2026-08-05: this points at the wrong kernel. See §"Requirement 1 does not
   need a change to GLM's `rmsnorm`" below.*
   Worth **7.5e-3 on a 3.1 max (0.24%)**, and supplying it host-side makes the mHC chain
   reproduce the goldens *exactly* — so the requirement is both real and sufficient. Shared
   with GLM, which is why no S2 agent could close it.
2. **`compress` returns the block COUNT, not the block INDEX.** At decode the block belongs
   at cache slot `start_pos / ratio`. A caller that appends is correct only while it never
   skips a step — and speculation is on by default.
3. **`score_state` must be `-inf`-initialised, `kv_state` zeroed.** After a prefill shorter
   than `ratio`, slots the decode pool reads were never written; zeros make them live
   entries with weight `exp(0-m)`.
4. **`Io.freqs` cannot distinguish the ratio-0 table from the YaRN one** — it is a raw
   pointer. Mixing them is `Defect::RopeNoYarn`: fluent and wrong.
5. **`Geom::indexer` must land with the fp4 finish**, never before. The indexer's compressor
   has the attention compressor's *shape* and a different *algorithm*; a `Geom` built for it
   passes every guard and runs block-64 e4m3 where fp4 over all 128 is due.

*Structural, will fault or silently corrupt:*
6. **`Scratch` sizing is unchecked** — a decode-sized scratch handed a `Prefill` overruns
   every buffer. A `capacity` field is three lines. — *LANDED 2026-08-05 as `Scratch::rows`.*
7. **`Dims`' public fields make `from_config`'s validation bypassable** — including the
   derived-extent check above — **and the test fixture already bypasses it.** That is the
   path S3 will copy by default. — *LANDED 2026-08-05: `Dims::validate`, called from
   `attention` too. Sealing the fields was rejected — it stops a struct literal and not a
   later `d.head_dim = 0`.*
8. `x`/`h` must be 16-byte aligned: unchecked, faults rather than falling back. `wexpert`,
   `h`, `descs` are indexed by **absolute** expert id and sized `n_desc`, not `e_count`.
   `hc_post`'s `y` must not alias `residual`. `Buffers` enforces only `scratch_rows`.
9. **`act_quant` runs on the null stream** — S2b's launcher takes no stream argument. Must
   be fixed before an overlapped layer loop, or the streaming design is defeated silently.
10. `tid2eid` entries and e8m0 `0x00`/`0xff` scale bytes must be rejected **at load**; the
    kernels cannot.

*Housekeeping S3 owns because no S2 agent was permitted to:*
11. **Lift one copy each of `f2e4m3_rne` and `fast_round_scale` into `common.hpp`** (see the
    jscpd blind-spot note above), plus `v4_block_sum`/`v4_rbf16`. Needs `mla.hip` edits.
    — *LANDED 2026-08-05, and it was not free: see §"Landed by S3".*
12. Prefill and decode index **different spaces** (absolute positions vs ring slots) and
    nothing in the types says so.
13. `mod v4` is `#[cfg(feature = "rocm")]` and `vk.rs` has no counterparts, so
    `tests/kernel_coverage.rs` does not see these launchers. Whether V4 gets a Vulkan arm is
    S3's call; **stubs would claim a parity nothing has measured.**

*Known-thin coverage S3 inherits as written, not as hoped:*
14. The **fast-path→tail handoff in the MoE kernels is exercised by nothing** — neither
    kernel runs both paths in one call at toy dims, while the launcher's `% 128` guard
    admits e.g. 384, which would hit it. A real gap in shipped code.
15. **e8m0 exercises 2 distinct codes of 254** (`119..=120`); a decode bug on any other is
    invisible to the whole suite. Toy-fixture artifact; the real checkpoint spreads wider.
16. **Toy-dim bit-exactness does not predict bit-exactness at depth.** At real dims there
    are 16× the terms, f32 re-association grows and bf16 flips become likely. Do not build
    a gate on the 0.000e0 results.

### The 16 requirements are conditions on a loop that has five missing prerequisites — S3, 2026-08-05

**The list above is a list of ways to get the wiring wrong. None of its entries is a thing
that has to be BUILT for the wiring to exist**, and five such things do not exist. Read as
a work plan the section reads as "connect the parts"; the parts do not meet. Each of these
was checked in the tree, not inferred:

1. **There is no `.f4` reader.** `ExpertHeader::from_bytes` (`format.rs:804`) hard-rejects
   any magic but `VQ3_MAGIC`; `ExpertSet::open` (`:939`) computes its length check as
   `(n_experts + 1) * stride` and a `.f4` file has exactly `n_experts` blocks;
   `ExpertSet::open_routed` (`:862`) matches only `"vq3"`/`"i4"`. The in-source note at
   `format.rs:804` says the first TWO must be relaxed **together**; `open_routed`'s extension
   match is a third it does not mention. `shared_block`
   must become unreachable for `.f4` rather than return the block past the end.
2. **There is no V4 resident-weight loader.** `Pin::build` takes `cfg: &'a ModelConfig` and
   `LayerPin` is MLA-shaped (`q_a/q_b/kv_a/kv_b/o_proj`). Nothing loads
   `resident.safetensors`' V4 tensors — the fp8 attention set, `attn_sink`, the four norms,
   the gate (`weight` plus `bias` **or** `tid2eid`), the six `hc_*` parameters, the
   compressor's `ape/wkv/wgate/norm`, the indexer's tensors, or the fp8 shared expert.
   `bin/convert_v4` **writes** all of them; nothing reads them.
3. **`embed` and `head` are int8 on the pin and bf16 in the artifact.**
   `launch_embed_i8_row` and `launch_gemv_i8` are the only two kernels for those roles;
   `convert_v4.rs`'s `MODEL_LEVEL` emits `embed.weight` and `head.weight` as bf16. Two kernels are missing.
   `v4_dense_gemm_bf16` is closer than it looks — runtime `(m, n, k)`, a bf16 `[n,k]`
   weight, so at `m=1, n=vocab, k=hidden` it computes a bf16 head GEMV correctly. The
   objection to reusing it is SHAPE, not capability: one wave per output element is a
   129,280-wave launch over a one-row activation. That is a performance argument carrying
   no measurement yet, so the honest instruction is "call it first, then price it", not
   "write a kernel".
4. **`hc_head` exists nowhere** — not in `kernels/`, not in `src/`, and deliberately not in
   the oracle (`forward.rs:1783`: the head tail "is NOT transliterated here"). So the last
   three ops of `Transformer.forward` — `hc_head`, the final `RMSNorm`, `ParallelHead` —
   have neither an implementation nor a golden. **The first decode's logits are ungated by
   construction**, and no tolerance chosen against per-layer goldens changes that.
5. **`attn::v4::attention` cannot express a compressed layer, which is 41 of the 43.** It
   derives the selection shape itself — `(seqlen, seqlen.min(window))` at prefill,
   `(1, window)` at decode — and *rejects* anything else. A compressed layer's selection is
   `cat([window_topk, compress_topk])`, so the width is `window + n_comp` and the `kv_src`
   it attends must be `[ring ‖ compressed]`, which is a different buffer in each phase —
   `Attention.forward`'s `start_pos == 0` arm builds `torch.cat([kv, kv_compress])` and
   attends *that*, while the decode arm attends `self.kv_cache` whole. The function's
   own doc says "for a `compress_ratio == 0` layer"; what was not drawn is that this makes
   it unable to run **every layer but 0 and 1**.

None of these is hard. Together they are more than "wire the loop", and the estimate that
put 16 conditions and no prerequisites in one stage is the thing to correct.

### The pre-indexer shortcut is narrower than it sounds — S3, 2026-08-05

The briefing S3 was given (and the reading that makes a pre-indexer decode sound free)
was: *"`index_topk` is 512 and the ratio-4 layers compress 4 tokens per block, so below
~2052 tokens `index_topk` never truncates and block selection is purely positional —
`compress_topk` in the oracle is a pure positional function."* **The two halves are each
true of a different code path and joining them is false.**

`compress_topk` (`v4oracle/forward.rs:1308`) is positional and takes no scores, and its own
doc scopes it to "where `compress_ratio != 4`". `Oracle::attention` (`:1437-1462`) selects
`lw.indexer` when it is `Some`, which `bin/v4-oracle.rs`'s `load_layer` pins to exactly the
ratio-4 layers (a `match (ratio == 4, ck.has_prefix("…indexer."))` that `bail!`s on either
mismatch);
`compress_topk` is the **ratio-128 fallback**. `Oracle::indexer` (`:1173`) computes the
full `[s, n_comp]` score matrix at **every** prompt length — there is no length gate on
that path.

What is true, and it is the useful half: `k = index_topk.min(n_comp)` with
`n_comp = (start_pos + s) / ratio`, so truncation begins at `ratio * (index_topk + 1)` =
**2052 total positions** (asserted by `tests/v4_compress.rs::indexer_topk_never_cuts_at_the_emit_prompt`). Below it the selected
**set** is every causally-legal block — fixed by the mask, not by any score — so a *set*
comparison cannot distinguish a right ranking from a wrong one, which is the hole recorded
above. Two consequences the loose phrasing hides:

- The engine may generate the set positionally below 2052 and attend the same blocks. It
  may **not** expect `.compress_idxs` to compare: `topk_idx` returns **score-ordered** rows
  (`forward.rs:776-782`, and `tests/v4_compress_probe.rs:331`), so the golden is a
  permutation of the engine's positional order even when both are right. Compare as a SET.
- Same set, different **order**, is not the same arithmetic. `sparse_attn` folds an online
  softmax over the rows in the order given, so a positional engine and a score-ordered
  oracle differ in the low bits of every compressed layer's output. That is a floor under
  any tolerance at real dims, and it is not re-association from the kernels — it is a
  *deliberate* difference this shortcut introduces.
- 2052 is a bound on `start_pos + seqlen`, not on the prompt. A 13-token prompt crosses it
  after 2039 generated tokens.

### Requirement 1 does not need a change to GLM's `rmsnorm` — S3, 2026-08-05

Requirement 1 reads "rivoli's `rmsnorm` does not bf16-round … Shared with GLM, which is why
no S2 agent could close it." **The kernel that must not change is not the kernel the V4 loop
should call.** `mla.hip::v4_rmsnorm` (S2b's) already stores `rbf16(w[i] * (row[i] * rs))`,
is one block per row, and is in-place. `linalg.hip::rmsnorm` (GLM's) is out-of-place, does
not round, and is **single-row** — `dim3(1)`, one mean over its whole `n`, so handing it
`s * dim` takes a joint statistic over every token and reads the norm weight past its
allocation (found by review 2026-08-05, `tests/v4_kernel.rs:1165`).

So the requirement is satisfied by *selection*, not by an edit: `attn_norm` and `ffn_norm`
go through `launch_v4_rmsnorm`, in place on `hc_pre`'s `y` (which is `[s, dim]` scratch, not
the residual — the residual is the `[s, hc, dim]` `h` that `hc_post` reads). The measured
7.5e-3 gap and GLM's kernel both stay exactly where they are.

### Landed by S3 — 2026-08-05

Requirements 6, 7 and 11; the reasoning is at the code (`common.hpp`'s V4 helper block,
`attn::v4::Scratch::rows`, `attn::v4::Dims::validate`). **Requirement 11 did not close the
hole it came from:** lifting four functions by hand is not a mechanism, `build.rs` still
does not scan `kernels/`, and the next pair of parallel agents can re-derive the same
arithmetic with nothing to stop them.

**Requirement 11 was not free, and neither problem was found by a test.** Two review
findings, both real:

- **The lift introduced an FMA.** `mla.hip` opens a file-scope `#pragma clang fp
  contract(off)`; `common.hpp` is included ABOVE it, and clang attaches FP options per
  expression at parse time, so pointing `v4_act_quant` at `common.hpp::f2e4m3_rne` put its
  subnormal branch under default contraction. Counted inside `v4_act_quant` at
  `--offload-arch=gfx1151 -O3`: **7 fma-class instructions at 78796eb, 8 after the lift, 7
  with a per-function `contract(off)` restored** — run in that order so the check was seen
  to go red. No value moved (the branch bounds `a` to (2^-10, 2^-6) where `a * 512.0f` is
  exact), but `mla.hip`'s "VERIFIED IN THE ISA" note and the `ULP_BUDGET = 1` it justifies
  in `tests/v4_attn.rs` had both quietly become false. Restored rather than documented away.
- **A fourth copy survived the first pass.** `v4_act_quant`'s clamp-and-encode loop was
  `common.hpp::act_quant_roundtrip` open-coded — same divide, same deliberate ternaries (so
  a NaN propagates instead of being laundered into -448 by `fminf`), same encode — sitting
  eight lines below the note claiming the duplication was lifted. Now calls the shared one.
  `moe.hip::bf16r` was a fifth: an identical body to `rbf16` under a transposed name, both
  in scope in one translation unit.

The **only** remaining ISA deltas against 78796eb are a clamp-order swap in `v4_act_quant`
(`act_quant_roundtrip` tests the low bound first; identical for every input including NaN)
and one added `s_barrier` each in `v4_rmsnorm`/`v4_qk_norm` from the trailing
`__syncthreads()` — 1012 → 1024 bytes of code.

**Still not verified on a GPU.** `tests/v4_attn.rs`, `tests/v4_kernel.rs` and
`tests/v4_compress_kernel.rs` are what would show a numerics change and none has been run.
The ISA diff bounds the risk; it does not replace them.

**Named gap this leaves.** Nothing in the tree asserts the no-contraction property — it is
a hand reading of the ISA in a comment, and this stage broke it without anything going red.
A check would be a grep of the compiled `.s` for fma-class instructions inside the V4 kernel
symbols. Not built here; recorded so the next reader knows the comment is the only guard.

### Separation predicted at real dims, BEFORE measuring — S3, 2026-08-05

Requirement 16 says toy-dim bit-exactness does not predict bit-exactness at depth, and
forbids building a gate on S2's `0.000e0`. Stated here so it cannot be fitted afterwards.

**A ratio-0 layer, one layer, real dims.** ≥99% of elements bit-identical; the remainder at
exactly **1 bf16 ULP**; none above 2. The reasoning: the kernels reduce in a wave/LDS tree
and the oracle folds sequentially, so the f32 disagreement grows from ~16× more terms — but
every stage boundary re-truncates to bf16 (eps ≈ 2^-8), so an f32 delta of ~2^-20 relative
only survives the store when the value sits within that of a bf16 rounding boundary. If the
measured spread exceeds 2 ULP on a ratio-0 layer, the cause is a defect and not depth, and
that is the prediction's whole point.

**A compressed layer is predicted WORSE, and for a reason that is not re-association.** Two
sources stack on top of the above:

1. `sparse_attn` folds an online softmax over the selected rows **in the order given**.
   Below 2052 positions a pre-indexer engine iterates blocks positionally and the oracle
   iterates them in `topk_idx`'s score order — the same set, a different fold. That is a
   permutation of ~`n_comp` terms, not a tree-vs-sequential difference, and it moves the
   running max as well as the sum.
2. The compressed rows arrive through the compressor's own pooling and partial `act_quant`,
   so they carry S2c's measured floor (three of four cells bit-identical, the fourth one
   e4m3 boundary flip on 0.0153% of elements) before attention touches them.

So: **1–4 bf16 ULP on the compressed layers, with the excess over the ratio-0 prediction
attributable to fold ORDER rather than to depth.** The way to tell those two apart is to
feed the engine the oracle's own `.compress_idxs` order for one run — if the spread drops
back to the ratio-0 prediction, source 1 is confirmed and the remaining gap is source 2.
That control run is the measurement to make first; without it a compressed-layer number
cannot be attributed at all.

### Two guards were built and then withdrawn — S3, 2026-08-05

`14a9009` shipped a `RopeTable` newtype (a `*const f32` tagged `yarn: bool`, checked in
`attention` against the layer class) and a `Dims::compress_slot` (the compressed block's row
and count, per phase). Both had tests; both were mutated to red before being trusted. Both
are **deleted**. What ruled them out is worth as much as what they did:

- **`RopeTable` diagnosed a mismatch that construction can prevent.**
  `v4compress::rope_for_layer(compressed, rope_theta, kind)` already keys on `LayerKind` —
  the same `compressor_ratio().is_some()` predicate the guard used — and moves theta and
  `original_seq_len` together. A caller who builds from `kind` cannot produce the mismatch;
  the tag was a second place to state the same fact, and therefore a second place to state
  it wrongly.
- **`compress_slot` had one caller and it used the function backwards** — the "destination"
  was read as a memcpy source. Its decode arm, the half carrying requirement 2, had no
  caller at all.

**Both withdrawals cost real coverage, and neither cost is theoretical.** `attention` no
longer detects `Defect::RopeNoYarn`, on the one path this port has *measured* as invisible
to its numeric gate (`ratio4/decode`, sep 8 against a `RESOLVABLE` of 64). And requirement
2's decode slot — `start_pos / ratio`, never "the next free one", the rule speculative
decode breaks by construction — is now implemented nowhere and asserted nowhere; it survives
only as prose on `v4compress::compress`. Both were device-free tests. The trade is only
sound if the enforcing construction actually lands:

- `Io` must be built by something that takes `kind` and calls `rope_for_layer` itself, so
  the two-table selection has exactly one site.
- Whatever places the compressor's output must compute the decode slot, and be tested on it.

Until then these are two guards removed and not yet replaced, which is a worse position than
before `14a9009` — recorded here so it is a decision with a deadline rather than a deletion
that quietly became permanent.

### DEBT: the enforcing construction, handed back rather than landed — S3, 2026-08-05

`2445645` deleted `RopeTable` and `Dims::compress_slot` on the argument that construction
should prevent both defects instead of diagnosing them. The construction was **not** landed
in that stretch, and this is the explicit hand-back rather than a silent deferral.

**Why not landed, and it is not scope.** Both replacements have exactly one correct home —
the layer loop — and the layer loop does not exist. Landing them anywhere else produces a
`pub fn` with no caller, which is precisely what `compress_slot` was when it was deleted:
one caller that used it backwards, and a decode arm with none. Re-adding a callerless
helper to discharge a debt created by deleting a callerless helper is a loop, not progress.

**What is owed, and what it costs until it is paid:**

| owed | until then |
|---|---|
| `Io` built by something that takes `LayerKind` and calls `v4compress::rope_for_layer` itself, so the two-table selection has ONE site | nothing detects `Defect::RopeNoYarn`, on the one cell measured invisible to the numeric gate (`ratio4/decode`, sep 8 against `RESOLVABLE` 64) |
| whoever places the compressor's output computing the decode slot as `window + start_pos / ratio`, with a test | requirement 2 is implemented nowhere and asserted nowhere; a caller that appends is right only until it skips a step, and speculative decode skips by construction |

> **PARTLY DISCHARGED 2026-08-05 by the compressed-layer cell, and one entry is corrected.**
> Both rules now have a *caller* and a *test*, in `tests/v4_attn.rs`: `Gpu::new` builds the
> rotary table through `rope_for_layer` keyed on `LayerKind` (and
> `the_two_rope_table_constructions_agree_on_the_un_yarned_table` pins the un-YaRN'd arm
> against `v4_rope_table_ratio0`, a comparison nothing in the tree performed before), and
> `Gpu::compress_and_place` computes both destinations with `COMP_SLOTS` asserting the decode
> slot. Measured: **a compressed-layer numeric gate separates a wrong rotary table
> (`RopeNoYarn`) at 33,461 ULP.**
>
> **Read that narrowly.** It means the mistake will be VISIBLE once something drives the layer
> loop through such a gate. It does NOT mean anything in the engine detects it — nothing does,
> the owed item ("`Io` built by something that takes `LayerKind`") is untouched, and the cell
> this table actually names, `ratio4/decode`, is still the measured-invisible one. Both
> callers here are the TEST harness, so neither debt is retired; what has changed is that the
> loop now has an executable specification of both rules to copy, with a seen-red record. The
> harness is therefore a second implementation of the loop's placement rules, and it and the
> future engine can drift with nothing able to notice.
>
> **The second entry's stated reason was wrong** and the correction matters more than the
> discharge: "speculative decode skips by construction" is false — `compress` refuses
> `seqlen > 1` at decode, so a speculating engine calls it once per position and a gap is a
> bug, not a mode. And the requirement is **not observable through attention output at all**;
> see the measurement above. A layer loop reviewed against numeric goldens alone would ship
> the append rule green.

Both were **device-free** tests before deletion. Whoever builds the layer loop owns both,
and this table is the acceptance criterion.

### The full `cargo test --release --features rocm` hangs — S3, 2026-08-05

Attempted on the merged tree; produced **zero** `test result` lines in 10 minutes and had to
be killed. It reaches `Running unittests src/lib.rs` and stops there. That is CLAUDE.md's
recorded intermittent `gpustream` hang, reproduced. Teardown was clean — 0 KFD holders and
the flock free afterwards — so it wastes time rather than wedging the device.

Consequence for anyone quoting a suite-wide number: **run the test binaries individually.**
A per-binary sweep of all twelve V4 suites is 105 tests and takes under a minute:

```
v4_attn 8 · v4_kernel 17 · v4_compress_kernel 8 · v4_indexer_kernel 8 · v4_pin 1
v4_hadamard_basis 4 · v4_attn_host 9 · v4_compress 7 · v4_oracle 27 · v4_loading 10
v4_artifact 2 · v4_compress_probe 4        — all rc=0 at `2445645`
```

### The compressed-layer cell — PREDICTED BEFORE MEASURING, S3-e2e, 2026-08-05

The gap: `tests/v4_attn.rs` pinned `LAYER` to ratio-0, so the `io.cache` tail layout, the
prefill persist copy, the decode slot and compressed columns reaching `sparse_attn` were
executed by nothing, on **41 of 43 layers**. The cell added here drives toy layer 3
(`compress_ratio == 8`, `NonOverlap`, no indexer) through `compress` → both placements →
`attention`, against `Oracle::run_layer`.

**Why the toy and not `/var/db/rivoli/v4-f4-l3-5`.** What is uncovered is *plumbing* — which
buffer holds the compressed rows, at which offset, written when — and plumbing is
dimension-independent. The real-weights arm would cost a V4 `LayerW` loader and the full MoE
per step (`forward.rs:1091`: 3.4 GB of experts per layer, which is why `Oracle::compressor`
was made `pub` rather than driven through `run_layer`), and requirement 16 already says
toy-dim bit-exactness does not predict real-dim bit-exactness — so a real-dims number here
would not license one either. Recorded as a scope decision, not an oversight: **this cell
says nothing about the compressed path at `head_dim = 512`.**

**Ratio-4 is deliberately not the cell**, and that is the sharpest limit. `Oracle::attention`
selects `lw.indexer` at ratio 4 and `topk_idx` returns **score-ordered** rows, while
`v4_topk_idxs` returns them positionally. Same set below 2052 positions, different fold
order through `sparse_attn`'s softmax — so every disagreement on a ratio-4 cell would be
uninterpretable, which is §"The pre-indexer shortcut is narrower than it sounds" arriving at
its consequence. `attention` branches on neither the ratio nor `has_indexer`, only on
`Sel::shape`'s `n_comp`, so the plumbing under test is the same on both classes; the
**arithmetic** at ratio 4 is not covered.

**Stated before the hardware ran, so it cannot be fitted afterwards:**

| | predicted |
|---|---|
| ratio-0 cell after the harness refactor | unchanged, floor **0** — no arithmetic moved |
| `q`, `kv_entry` on the compressed layer | **bit-identical** |
| `compressed` (the pooled blocks, read back from where `sparse_attn` indexed them) | **bit-identical** |
| `attn_derot`, `attn_out` | **bit-identical**, floor **0** |
| the 19-defect separation sweep | every entry ≥ 1000 bf16 ULP; tightest margin from a compressor defect |

The reasoning, and the fallback that would falsify it: the ratio-0 cell already measures 0 at
these dims. A compressed layer adds a fold over `ents = 8` pooling entries per feature, an
RMSNorm over 256, and `topk` growing 8 → 9..12 columns. Tree-vs-sequential re-association is
~1e-7 relative against a bf16 step of ~0.4%, so the expected flip count over a 256-element
block is ~0.006. `v4_compress_kernel.rs` measured 3 of 4 cells bit-identical at **real** dims
over 32768 elements; 256 elements is ~128x less exposure. **The one uncontrolled source is
`expf` versus `f32::exp`** in the pooling softmax, which that suite names as a bound it cannot
supply. If it bites, `compressed` moves by exactly one e4m3 step (**16 bf16 codes**) on a
handful of elements — over `ULP_BUDGET = 1`, and the honest response is a recorded registry
entry, never a widened budget.

**Found before the hardware, and it corrects two shipped comments.** `src/attn.rs`'s launch
sequence says of the QK-norm's position "The oracle cannot see this order", and
`tests/v4_attn.rs`'s header listed it beside `KvActQuantBlock128` as one of two defects
"invisible to these goldens". Measured device-free on the compressed layer,
`Defect::QkNormAfterRope` moves **four** goldens on 4–13% of their elements
(`q` rel 7.4e-3, `attn_out` rel 9.1e-3). The mathematical argument is sound — RoPE rotates
adjacent pairs so it preserves `q.square().mean(-1)`, and a scalar commutes with a rotation —
but `Oracle::qk_norm` computes that statistic in **bf16** (`forward.rs:768`, faithfully: it is
bf16 in the reference), so `rs` is quantized to ~0.4% steps and the two orders land on
different steps. It is a *rounding* difference, which
`tests/v4_oracle.rs::qk_norm_order_is_a_rounding_difference_not_an_arithmetic_one` already
bounds against dropping bf16 rounding entirely — it is not an *invisibility*, and the two
exclusions were being carried as one fact. `KvActQuantBlock128` really is bit-inert here and
is now asserted so.

The generalisable part: both claims were arguments from exact arithmetic about a pipeline that
rounds. That is the same shape as the `KvActQuantBlock128` scale-invariance derivation this
document already records as "right in kind and wrong at the boundary" — the third time in this
port that a first-principles equivalence has been taken for a bitwise one.

**What the three reviews caught, and two of them are this document's own named failure mode.**

- **A guard whose condition cannot occur, added in response to the previous review.** The
  first round left the compressed sweep without the ratio-0 sweep's schedule-length assert;
  the fix moved it into `reach` so neither caller could forget it. It is a **no-op**: no
  `Defect` can change the step count, because `drive_script` pushes one `Phase` per script
  entry unconditionally and nothing in `run_layer` is defect-conditional. Kept, re-scoped, and
  re-worded to the condition that IS reachable — a caller pairing `refs` and `mine` from the
  two cells' different-length scripts, which `zip` truncates silently. `moved`'s
  presence-mismatch arms are dead for the same kind of reason (`.compressed`'s existence is a
  pure function of `(seqlen, start_pos, ratio)`) and now say so instead of advertising
  themselves as the instrument's strongest protection.
- **A synthetic state sold as a production one.** `COMP_SKIP_TO` justified its deliberate gap
  with "speculative decode advances `start_pos` by more than one and produces exactly this gap
  in production". **Retracted.** `compress` refuses `seqlen > 1` at `start_pos > 0`, so an
  engine accepting two speculative tokens calls it once per POSITION; a gap is a bug, not a
  mode. The gap is still *necessary* — a block is skipped only when positions are, and the
  decode slot `start_pos / ratio` agrees with "next free slot" on every contiguous script — but
  what it certifies is narrower than it read. At position 31 both implementations pool a block
  from slots last written at positions {16, 9, 10, 11, 12, 13, 14, 31} rather than 24..31, and
  RoPE it at 24. That is a **state-machine** probe: it shows engine and oracle implement the
  same deposit/emit/slide machine and that the block lands at the right slot, not that the
  value is what `model.py` computes for a real sequence. The reference-faithful evidence is the
  prefill and the contiguous decodes.
- **A silent re-fixturing of the ratio-0 cell.** The per-layer `wo_a` seed (`…-fp8-L{layer}`)
  changed layer 0's synthetic weights, against which `floor == 0` and the `r >= 8` separations
  are asserted exactly and dated. One hoisted RNG under the original name restores them. The
  same change carried a comment on the `h-{tag}` seed arguing against exactly this.

**A process failure to record: a review subagent ran the GPU tests without the lock.** It was
given Bash and not told the device was held; it ran three GPU tests plus a `cargo check`. Any
contention in that window is explained by it. Its numbers are not adopted here — the measured
results below are from the run under `flock` with the KFD witness taken inside it. **The
prohibition belongs in the review prompt**, because a subagent inherits Bash and is covered by
nothing the parent does: *"Do not run anything that touches the GPU; read the code and the
captured logs."* Reviews are meant to read code, so it costs nothing.

### MEASURED on gfx1151, 2026-08-05 — every prediction held, and one gate did not

Under `flock`, KFD witness taken inside the lock (empty on every run). Per-binary sweep,
**117 passed over 14 suites**; the only failure is the pre-existing dev-profile
`v4_loading::magic_separates_the_formats_when_the_length_cannot`
(`src/artifact/quant.rs:128`, `i_dim 32 not a multiple of VQ_GROUP`), which predates this work
and is in neither file it touches.

| predicted | measured |
|---|---|
| ratio-0 floor unchanged at 0 | **0 ULP**, all three steps, every stage `differing=0` — the hoisted-RNG fix restored the dated numbers |
| `q`, `kv_entry` bit-identical | **0/24576 and 0/3072**, every step |
| `compressed` bit-identical | **0/256** at all three emitting steps (prefill, 15, 31) |
| `attn_derot`, `attn_out` bit-identical, floor 0 | **0 ULP over 7 steps** |
| 19-defect sweep ≥ 1000 ULP | **32,932–34,562**; tightest `KvActQuantWholeTensor` at 32,932 |
| tightest margin from a compressor defect | **wrong** — it is `KvActQuantWholeTensor`; the two compressor defects sit at 33,790 and 33,584, mid-field |

The `expf`-versus-`f32::exp` fallback did not bite: the pooling softmax agreed exactly at
these dims, so no registry entry was needed and `ULP_BUDGET` was not touched.

**THE NEGATIVE RESULT, and it is the most useful thing here: requirement 2 is invisible to
the attention output.** Injecting the append rule — the decode slot as "the next free slot"
instead of `window_size + start_pos / ratio` — left **every numeric golden bit-identical**,
`attn_out` included, on a script built specifically to expose it. The reason is structural,
not a fixture accident:

```
start_pos 31, ratio 8, window 8, n_comp = 4  ->  selection names cache rows 8,9,10,11
correct: [8]=b0 [9]=b1 [10]=0  [11]=b3        oracle: [0]=b0 [1]=b1 [2]=0  [3]=b3
append:  [8]=b0 [9]=b1 [10]=b3 [11]=0
attended MULTISET in all three: { b0, b1, b3, zero-row }
```

The two rules differ by a **permutation** of the compressed region. The positional selection
covers rows `0..n_comp`, which is exactly the highest slot the correct rule uses, so append
never writes outside the selection and never overwrites a live row — and `sparse_attn`'s
softmax over a set is permutation-invariant. **More gaps do not help**: the multiset is
invariant under every permutation the two rules can produce. So the `COMP_SKIP_TO` gap did not
buy what its comment claimed, for the *second* time (the first was the retracted
speculative-decode framing).

**Two premises, and both expire.** This holds only while (a) the compressed selection is the
positional full prefix — `Sel::n_comp` REFUSES past `index_topk` rather than truncating — and
(b) unwritten rows read as an agreed zero on both sides. Once the score-ordered `Indexer`
lands, the selected SET is content-chosen and a permuted region changes which blocks are
attended, not merely their fold order; and a compressed region reused across sequences without
clearing holds stale rows rather than zeros, so the two rules read different values. The
conclusion is scoped, not structural: **re-measure when either premise goes.** Stated because
an unscoped version would be cited to justify not re-testing — and note that the engine's
obligation to clear the compressed region between sequences is asserted nowhere today.

What it buys instead, and what now gates the requirement: an assertion that block `b` sits at
cache row `window + b`, read from the ring **directly** and compared against a slot spelled by
hand in `COMP_SLOTS` — a source independent of the arithmetic under test. That goes red under
the append rule and nothing else does. One of its two anti-vacuity arms was proved able to fire (the
"emitted a block whose destination COMP_SLOTS does not name" arm). The other — "a COMP_SLOTS
row names a step this script skips" — is reached only by a row whose `start_pos` is absent
from the script; the break recorded above instead names a step that IS in the script but emits
nothing, which short-circuits earlier. Recorded as a gap rather than claimed. Note the shape: a numeric oracle comparison, however tight, cannot see a **layout**
error that permutes rows the selection covers uniformly. Requirement 2 could have been
declared covered on a green run here, and would not have been.

**SEEN-RED RECORD.** No gate below is trusted green without having been watched go red.

| break | gate that fired | evidence |
|---|---|---|
| drop the prefill persist copy into `io.cache` | `attn_derot` at the FIRST decode step | 2031/2048 differing, rel 4.2e-1 |
| drop the prefill `s.kv` tail copy | `attn_derot` at prefill | 10150/24576, rel 6.7e-1 |
| decode slot → "next free" | `COMP_SLOTS` layout assertion | "block 3 is not at cache row 11" — **and no numeric golden moved** |
| drop a `COMP_SLOTS` row | "a step emitted a block whose destination COMP_SLOTS does not name" | — |
| add a `COMP_SLOTS` row for a non-emitting step | the golden lookup for `.compressed` | — |
| engine forgets the layer compresses (`Sel.kind` → `Plain`) | the anti-vacuity assertion | "0 compressed columns past a window half of 12, largest selection index 11" |
| ratio-0 rope table on a compressed layer (requirement 4, in the ENGINE) | `q` at prefill | 4375/24576, rel 1.06 |
| poison probe 2 aimed at a SELECTED block | the bit-identical arm | paraphrase: the probe reported MOVED where the table requires identical |
| a `COMP_SLOTS` row naming a start_pos absent from the script | "a COMP_SLOTS row names a step this script skips" | — |
| a `COMP_SLOTS` row naming an in-script step that emits nothing | "COMP_SLOTS names start_pos 14 but the reference emits no block there — the table is wrong, not the oracle" | — |
| a stray write to a compressed row PAST the selection | "cache row 13 names no block this script emits and must never be written" | **caught by nothing else** — every numeric golden passed |

**Two more dead guards were found this way, making four in this stage.** `Gpu::poke`'s bounds
assert could not fire (its callers are gated by `base.n_comp < capacity` at the call site); it
moved to `Gpu::cache_row`, whose rows come from a hand-written table and are bounded by
nothing. And the first version of the unwritten-row check tested only the SKIPPED block's row
— a placement writing a duplicate there is caught earlier by `assert_within`, because that row
is inside the selection. Verified by injecting exactly that "belt and braces" placement: it
died in the numeric comparison and never reached the check. Widened to every compressed row
`COMP_SLOTS` does not name, it states something nothing else does, and a stray write to row 13
proves it.

The pattern is worth naming: **three of the four dead guards were added in response to a
review finding.** A reviewer says "X is unchecked", the obvious check gets written, and nobody
asks whether X was reachable. The question that catches it is the one this stage's brief
opens with, applied to the *fix* and not only to the code under review.

Each break was checked for effectiveness before its result was believed — the append break's
patch was confirmed present in the source while the suite was green, which is what turned a
"the gate works" reading into the finding above.

**One self-inflicted loss, recorded because the rule that prevents it is already written
down.** The layout assertion was added *after* the break script started and was not staged;
the script's `git checkout -- tests/v4_attn.rs` (reverting a break) took the new work with it,
and the next break then ran against the old file and passed for the wrong reason. CLAUDE.md's
"stage before you inject a break" exists for exactly this. Re-applied and committed before any
further injection.

## S4 — benchmark, quality assessment, ranked perf work.

Scoped after S2 reports. S4's quality assessment ranks on **paired dNLL from `bin/ppl`**, not
on a PPL point estimate and never on free-running tok/s — and note that at 5k tokens this
engine has a ~40% silent-corruption rate on GLM (`benchmarks.md`, "Long runs are
NON-DETERMINISTIC"), so **every arm gets repeated** before any number is quotable.

---

## Standing rules for every stage

- **The GPU is sole-tenant and the coordinator holds it.** Do all non-GPU work first, then
  report ready and wait for an explicit GO. Check occupancy with `/sys/class/kfd/kfd/proc/`,
  never `pgrep`. Wrap every GPU command in `flock /var/run/sys-gpu.lock -c '…'`; build
  *outside* the lock.
- Commit freely to your own branch. **Never push, never merge**, never touch another
  worktree. No `git stash` — it is shared across worktrees. Restore files with
  `git checkout -- <file>`, never with `sed`.
- Before **every** commit, run ponytail / code-quality / correctness reviews as subagents and
  address or explicitly decline each finding.
- Build with the union features, not just `rocm`:
  `cargo clippy --release --features rocm,otlp,teacher-forcing,pred-probe,trace --all-targets`.
- Duplication is a build error (`jscpd`, no budget). Transliterated oracles are the exception
  and carry `jscpd:ignore-start` with the argument in place.
- Comments carry the *why* and the measurement. A comment asserting a check that does not
  exist is the most common finding in this repo's reviews — if a comment says verified,
  something must verify it.
