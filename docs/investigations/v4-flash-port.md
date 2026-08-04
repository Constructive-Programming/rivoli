---
status: live
verdict: The staged plan to make V4-Flash decode. Corrects other-models.md on three points measured from the downloaded repo: experts are 148.25 GB native FP4 (138.1 GiB) so it DOES stream at ~83% residency, not "nearly fully resident"; 4.02 GB/token not 3.1; the partial fp8 KV act_quant is mandatory, not a --kv-fp8 to refuse; YaRN is per-layer, keyed to compress_ratio. DSpark/MTP is separable and out of scope.
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

One routed expert (w1+w2+w3 incl. scales) is **13.37 MB**; top-6 + 1 shared × 43 layers =
**4.02 GB/token**.

### Three corrections that change the work

1. **The streaming verdict inverts.** `other-models.md` §3 called V4 "nearly fully resident…
   would barely stream" from a ~120 GiB artifact against a ~115 GiB `--max-mem`. That 120 GiB
   is the **int3-vq** figure, and §6 then decided to keep **native FP4**. At the format
   actually chosen the experts are **148.25 GB = 138.1 GiB**, which does *not* fit the pool:
   ~83% capacity residency, against GLM's ~41%. It streams — less than GLM, but it streams.
   Per-token traffic is **4.02 GB, not 3.1**. The two sections were computed at different
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
  (`Indexer` exists **only** where `compress_ratio == 4`, which is why 21 of 41 have one).
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

## S2 — attention frontend on GPU. Gated against S1b. GPU required.
## S3 — mHC + hash routing + clamped SwiGLU + the FP4 MoE kernel. Gated against S1b.
## S4 — first decode, benchmark, quality assessment, ranked perf work.

Scoped after S1 reports. S4's quality assessment ranks on **paired dNLL from `bin/ppl`**, not
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
