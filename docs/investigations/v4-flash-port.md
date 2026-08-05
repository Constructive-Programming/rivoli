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
make. Whether jscpd should scan `kernels/` at all is a separate question: the two backends'
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

### The indexer's device half — S2c-indexer, 2026-08-05. WRITTEN, NOT YET RUN.

`kernels/v4indexer.hip` (`v4_indexer_spread`, `v4_indexer_score`), the fp4 activation
quantizer in `common.hpp` (`f2e2m1`, `fp4_quant_roundtrip`), `Geom::indexer`, and
`tests/v4_indexer_kernel.rs`. **The GPU was held by the coordinator throughout, so not one
of these kernels has executed.** Everything below is a compile-and-review result; the
measured column is empty on purpose.

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

## S3 — wire the layer loop, first decode.

**Requirements banked from S2, each measured rather than supposed.** These are conditions on
the wiring, not suggestions:

*Correctness, will produce fluent wrong output if missed:*
1. **`rmsnorm` must bf16-round its output.** V4's `RMSNorm` returns bf16; rivoli's does not.
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
   every buffer. A `capacity` field is three lines.
7. **`Dims`' public fields make `from_config`'s validation bypassable** — including the
   derived-extent check above — **and the test fixture already bypasses it.** That is the
   path S3 will copy by default.
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
