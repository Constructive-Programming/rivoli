---
status: live
scope: qwen
verdict: The seven S0 math questions are ANSWERED from primary sources at a pinned revision whose config bytes are VENDORED and hash-gated in tree, and the n-gram hash spec is MEASURED byte-exact against the official fp8 checkpoint rather than transcribed -- GDN decay is exp of -exp A times softplus of Wa x plus b with the 48 V heads owning a 128-key by 128-value state, a SIGMOID output gate over a ONES-centred gated RMSNorm whose 256 settling bytes are vendored, conv kernel 4 with the newest tap LAST and SiLU after; the QSA indexer scores a sum over heads of ReLU of q dot kbar over 512 mean-pooled 4-token blocks, partial-RoPEd at the block-START position, and is provably DENSE at or under 2051 tokens from the reference's own topk of min block-topk and complete-blocks; hyper-connections are Qwen's own Gated Residual, NOT Sinkhorn mHC, with hc_lowrank 320 = hidden 2560 over 8; the hash is splitmix64 bigram plus trigram whose 3 multipliers and 16 prime head vocabularies were re-derived independently and matched the checkpoint's I64 buffers exactly, the byte-exact match at index 0 being the EVIDENCE that ple-layer-index is 0; the router is plain softmax with NO bias correction; partial RoPE covers the FIRST 64 of 256 dims, paired split-half. The traps table now covers all nine named classes at two or more defect rows each, and carries a class-to-row map so that is checkable rather than counted; it is headed by the layer-types ALIAS -- the checkpoint's full-attention entries are the 12 layers that RUN the indexer, and a dense reading of them is bit-identical below 2051 tokens and therefore invisible to every oracle, parity window and smoke cell this port plans. The headline OPEN taps are the n-gram fp8 scale direction, the block-alignment disagreement between report and code, and the per-operator fp32 boundaries, which are tolerances and are owed to S2 on 2 or more weight draws.
---

# Qwen3.8-Flash-Next (`qwen4_exp`): the math, from primary sources

S0's math extraction. Every answer carries its source; where two primary sources disagree
the disagreement is named and, where a weight settles it, **measured**. The traps table is
written so S2's defect matrix can be built straight off it.

## Sources and their pins

| tag | what | pin |
|---|---|---|
| **TR** | `tech_report.pdf`, 28 pp., pdfTeX, created 2026-08-26 | `github.com/QwenLM/Qwen3.8-Flash-Next` @ `6988587`, blob `f26e9dd6ea95`, 2,366,146 B |
| **MOD** | `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, 2707 lines | `huggingface/transformers` tag **v5.16.1** @ `93c8b7b48596`, blob `7af639771c07`, 124,924 B |
| **CFG** | `.../configuration_qwen4_exp.py`, 334 lines | same tag, blob `3dd586b84458`, 16,463 B |
| **ROPE** | `src/transformers/modeling_rope_utils.py` (`_compute_yarn_parameters`) | same tag |
| **CKPT** | `config.json`, `generation_config.json`, `model.safetensors.index.json` (152,089 tensors), safetensors headers and byte ranges | `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| **CARD** | `README.md` | same revision |
| ~~LC~~ | `ggml-org/llama.cpp` PR **#27742** (merged, head `eaf93765572e`) | **TRANSLITERATION — cross-check only, never a golden source** |

> **CORRECTED 2026-08-31.** This section's heading read "nothing vendored here; S0 item 1
> still owes the vendoring", which was false in the commit that wrote it: `config.json`
> (72,514 B) and `generation_config.json` (202 B) were vendored beside it under
> `docs/measurement/qwen-reference/`, at this revision, and `tensor-families.tsv` records
> their sizes and hashes. Every **CKPT** citation of a config field below is therefore
> re-derivable in tree — read the vendored file, not the network — and
> `crates/artifact/tests/qwen_names.rs` recomputes FNV-1a over the live bytes on every
> deviceless run, which is what census item 1 asked for and what closed it.

**MOD/CFG line citations carry their enclosing symbol**, as `MOD:519 (GatedDeltaNet.forward)`.
Class names drop the `Qwen4ExpText` prefix every one of them carries, so
`GatedDeltaNet` is `Qwen4ExpTextGatedDeltaNet`; module-level functions are named in full, and
so is `Qwen4ExpTextModel`, the one class whose stripped name (`Model`) would say nothing.
Reason: a patch bump moves lines and the symbol survives it, so a stale pointer degrades to a
hint instead of sending a reader to unrelated code. Every citation in this file was re-derived
mechanically against the pinned blobs on 2026-08-31 (`git hash-object` on the fetched files
reproduces `7af639771c07…` and `3dd586b84458…` exactly) after review found ~12 that did not
land — two of them inside `load_balancing_loss_func`, an aux-loss accumulator with nothing to
do with the claims they were attached to.

The model repo carries **no `auto_map`** and no `modeling_*.py`: the in-tree transformers
class IS the shipped implementation, as with GLM (`glm-reference/anchor.md:20`). The model
dir does not exist at v5.15.0, so **v5.16.1 is the earliest released tag that has it** —
the anchor venv the other four ports share cannot be reused unpinned.

Byte measurements below are `Range:`-read from CKPT; absolute offset is
`8 + header_len + data_offset`, with `header_len` 39264 (`model-00001`), 143816
(`model-00005`), 2688 (`model-00037`). No shard was downloaded in full: the largest read is
20,480 B.

> **CORRECTED 2026-08-31.** This said "Only headers and ≤256-byte ranges were read", which
> its own T1 row contradicts — `hc_norm.weight` is BF16 [10240], and a mean over 10240
> elements needs all 20,480 bytes of it. The numbers reproduce (re-read at review); it was the
> read-policy sentence that was wrong, and that sentence is the recipe for reproducing the one
> measurement in this file that overrides a primary source. The per-tensor provenance table in
> **T1 provenance** below now gives shard, offsets, `n` and dtype for all four, so each is
> re-fetchable byte-for-byte, and the 256 B that settle T1 are vendored in tree as
> `docs/measurement/qwen-reference/linear_attn_norm_l0.bin`.

## 1 — GDN exact math, and how it differs from KDA

**Decay — softplus, not a safe-gate.** `α_t = exp[ −exp(A) · softplus(W_α x_t + b_α) ]`
(TR Eq. 10, p.4) = `g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)` with
`α = exp(g)`, `g` carried through the rule (MOD:519 in `GatedDeltaNet.forward`; the rule
itself is MOD:266-398, the two module-level fallbacks `torch_chunk_gated_delta_rule` and
`torch_recurrent_gated_delta_rule`); the `.float()` is load-bearing (MOD:518, the comment
right above it: without it `A` may be `-inf` in fp16). Write gate `β_t = σ(W_β x_t)`
(TR Eq. 9; MOD:517, `GatedDeltaNet.forward`).

**`dt_bias` shape** is `(linear_num_value_heads,)` = **(48,)** — per-V-head scalar, not
per-channel (MOD:432 in `GatedDeltaNet.__init__`, the `nn.Parameter(torch.ones(num_v_heads))`
line; CKPT `layers.0.linear_attn.dt_bias` **BF16 [48]**, `A_log` likewise **BF16 [48]**, whose
own `uniform_(0.01,16)` init is the neighbouring MOD:435 this used to cite by mistake). Both are *learned*: at layer 0, `dt_bias` mean −3.29008 / min −8.0 / max
+2.53125, `A_log` mean +1.53425 / min −3.57812 / max +5.0625. The `torch.ones` and
`uniform_(0.01,16)` in `__init__` are init placeholders only.

**L2 norm placement.** `q_t = L2Norm(SiLU(ShortConv(W_q x)))`, same for `k`;
`v_t = SiLU(ShortConv(W_v x))` — **V is never L2-normed** (TR Eq. 6-8). It happens *inside*
the rule (`use_qk_l2norm_in_kernel=True` at both call sites, MOD:534 and MOD:547 in
`GatedDeltaNet.forward`), `eps=1e-6`, `dim=-1`, and **before** the `1/sqrt(d_k)` scale the rule
applies to `q` alone (MOD:259-262 `l2norm`; MOD:295-296 in
`torch_chunk_gated_delta_rule`). Pipeline:
`Linear → depthwise conv → SiLU → L2Norm(1e-6) → ×1/sqrt(128)`, minus the scale for `k`.

**Output gate: SIGMOID.** `o_t = W_o[ σ(W_z x_t) ⊙ RMSNorm(y_t) ]` (TR Eq. 11), explicitly
"Unlike the original GDN, which uses a SiLU output gate, we use the bounded sigmoid gate"
(TR p.4); code takes `config.output_gate_type or config.hidden_act`
(MOD:437-439, `GatedDeltaNet.__init__`) and CKPT sets **`output_gate_type: "sigmoid"`**
(CFG:193-195 in `Qwen4ExpTextConfig.validate_architecture`, which admits only
`sigmoid`/`silu`). Order inside the gated norm (MOD:192-201, `RMSNormGated.forward`): `x̂ = x·rsqrt(mean(x²)+eps)` in **fp32**, then `w ⊙ x̂` cast
back, then `⊙ σ(z.float())`, with `z` from `in_proj_z` (CKPT **BF16 [6144, 2560]**) reshaped
`(48, 128)`.

**The 48 V heads own the state.** `linear_num_key_heads: 16`, `linear_num_value_heads: 48`;
QK are `repeat_interleave(48/16 = 3, dim=2)` to 48 *before* the rule (MOD:520-522,
`GatedDeltaNet.forward`), so the
state is `[batch][48][d_k 128][d_v 128]` = 48 × 16,384 f32 per layer, over 36 GDN layers.
TR p.3 pins the layout — "GDN maintains a state `S_t ∈ R^{d_k×d_v}`, which is the
**transpose** of the state convention used in the original formulation" — and MOD:391-392
(`torch_recurrent_gated_delta_rule`) indexes it `[key][value]` (`sum(dim=-2)` over the key
axis), which is already rivoli's
coalescing order, so **no transpose is owed**. A `[128][128]` state passes every shape check
either way, so score both readings against the reference output.

Against KDA (scope: k3): no per-channel decay, no 7-kwarg fla entry point — fla-lineage
`chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`, per-head scalar gates, and
MOD:266-398's vendored torch fallbacks are the honest reference.

## 2 — GDN short convolution

Depthwise (`groups = conv_dim`), **bias-free**, `kernel_size = linear_conv_kernel_dim = 4`
over `conv_dim = 2·key_dim + value_dim = 2·2048 + 6144 = 10240` channels (MOD:419-428,
`GatedDeltaNet.__init__`;
CKPT `linear_attn.conv1d.weight` **BF16 [10240, 1, 4]**).

**Activation is AFTER the conv**, and it is `config.hidden_act = "silu"` — *not* the output
gate's sigmoid (MOD:229-233 `causal_conv1d_update`, MOD:254-255 `causal_conv1d_fn`, and the two
call sites that pass `activation=self.activation`, MOD:477-483 and MOD:490-496 in
`GatedDeltaNet.forward`). Prefill: `F.conv1d(..., padding=3)` then `[:, :, :seq_len]`
(MOD:245-253, `causal_conv1d_fn`).

**Tap order.** `F.conv1d` is cross-correlation (no kernel flip), so with 3 zeros of left
padding `out[t] = Σ_{j=0..3} w[j]·x[t−3+j]`: **`w[3]` multiplies the CURRENT token, `w[0]`
the oldest (t−3)**. The decode path agrees — the cache holds `conv_kernel_size = 4` columns,
`cat([state(4), x_t(1)])` → `padding=0` conv → last output, i.e. taps
`{state[1], state[2], state[3], x_t}` (MOD:217-233, `causal_conv1d_update`; `cache_utils.py`
`DynamicLayer.update_conv_state`, same tag) — so the state is allocated **4 wide while only
3 columns are live**, a natural off-by-one for a port that keeps 3.

Fused-projection split: `in_proj_qkv` (CKPT **BF16 [10240, 2560]**) is
`torch.split(..., [key_dim, key_dim, value_dim])` = **Q rows 0..2047, K 2048..4095, V
4096..10239**, each then reshaped head-major (MOD:502-515, `GatedDeltaNet.forward`). LC#27742 names this a live trap
— "three different segmentations, one correct and two deliberately wrong" — that its own
arch test could not distinguish.

## 3 — QSA indexer

**`layer_types` in the checkpoint is `full_attention`, and those 12 layers RUN THE INDEXER.**
CKPT's `text_config.layer_types` is 36 `"linear_attention"` + 12 `"full_attention"` at indices
3, 7, …, 47 — and the reference never uses that second spelling:
`Qwen4ExpTextConfig.__post_init__` rewrites it before validation,

```
# The real checkpoint contains "full_attention" entries for layers that are actually using an indexer
elif "full_attention" in self.layer_types:
    self.layer_types = ["qwen_sparse_attention" if layer == "full_attention" else layer ...]
```

(CFG:180-184 in `Qwen4ExpTextConfig.__post_init__`, comment included verbatim), and
`validate_architecture` (CFG:190) then admits **only**
`{linear_attention, qwen_sparse_attention}` (CFG:190). MOD compounds it by using the identical
string `"full_attention"` as a **causal-mask dictionary key** (MOD:1404, 1423 in
`Qwen4ExpTextModel.forward`) — a different namespace, same spelling. So a port that reads `layer_types`
at face value builds dense global attention on 12 of 48 layers and skips the indexer: see trap
**T19**, and note that it is the one defect in this file no planned oracle can see, because the
two implementations are bit-identical below 2051 cached tokens (next subsection but one).

MQA: `indexer_n_heads: 4` query heads + `indexer_kv_heads: 1` shared key head,
`indexer_head_dim: 128`; `index_qk_proj` is one **BF16 [640, 2560]** matrix
(MOD:623-627, `QSAIndexer.__init__`) split `[4·128 | 1·128]` (MOD:645-650,
`QSAIndexer.forward`; CKPT). **Scoring — no learned per-head weights.**
`I_ib = Σ_{h=1..4} ReLU(⟨q_i^h, k̄_b⟩)` when `p_b + r − 1 ≤ i`, else `−∞` (TR Eq. 15, p.6) =
`matmul(q, K̄ᵀ)` in **fp32**, `.transpose(-1,-2)`, `relu(...).sum(dim=-1)`, `/ sqrt(128)`
(MOD:690-693, `QSAIndexer.forward`). **No `w_h`**; the sum is over heads. TR omits the `1/sqrt(d)` — a positive
uniform scale that cannot move the top-k, but pin the code's order for a bit-exact fp32
oracle.

**RoPE applies to indexer q and to the pooled k, at different positions.** `q`:
`RMSNorm → PRoPE(q̂, i)` at the query's own position (MOD:651-652). `k`: raw un-normed keys
are cached, then per block **mean-pooled in fp32** (`AvgPool`, TR Eq. 13 = MOD:681
`pooled_keys = key_groups.float().mean(dim=1)`), cast back, `k_layernorm` (MOD:682), and only
then `PRoPE(k̄_b, p_b)` at the **block's FIRST token position** (MOD:683-688, where
`group_starts = block_token_indices[:, 0]` is the position handed to `apply_rotary_pos_emb`;
all in `QSAIndexer.forward`; TR Eq. 13-14, whose reason is that pooling first "avoids averaging token
representations with different rotary phases"). Partial RoPE covers 64 of the 128 indexer
dims, "matching the rotary dimension used in the core attention module" (TR p.6). So it is
**pooled keys, not score-then-aggregate**, and the indexer cache holds **one 128-dim
un-normed, un-roped key per token** across 12 QSA layers.

**`compress_ratio: 4` means 4 TOKENS per block, and both readings of the budget hold at
once**: `block_topk = indexer_budget // indexer_compress_ratio = 2048/4 = 512` (MOD:622,
`QSAIndexer.__init__`) →
**512 blocks × 4 tokens = 2048 tokens**. TR Eq. 16 writes `K_B = ⌈K/r⌉` where the code
floors; CFG:223-224 (`validate_architecture`) requires exact divisibility, so they agree here
and only here.

**Block boundaries.** `num_complete_blocks = |visible| // 4` (MOD:672); the last
`|visible| mod 4` tokens are **always included and never scored** (MOD:700-701, the `tail`
concatenated after the top-k); and the selection buffer is
`indexer_budget + compress_ratio − 1 = 2051` wide (MOD:661-666, all in `QSAIndexer.forward`;
TR Eq. 19). The
corollary S2 should exploit, **from the first-party source alone**: MOD:695 selects

```
selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
```

so whenever `num_complete_blocks ≤ block_topk = 512` the top-k is TOTAL — every complete block
is selected — and the `mod 4` tail is concatenated unconditionally (MOD:700-701). With
`num_complete_blocks = |visible| // 4`, that holds for every `|visible| ≤ 2051`. **So at ≤ 2051
cached tokens QSA is dense by construction**, and a below-budget window is a *free exact
oracle* over the whole selection-and-masking path. LC#27742 measured 0.0 max logit delta over
2051 rows, which CORROBORATES this and is not the source of it — the pin table above labels LC
a transliteration, cross-check only, and a two-line structural argument from MOD needs no
number of theirs.

**The same fact is the blind spot in every gate this port plans.** A dense-global
implementation of the 12 `full_attention`-spelled layers (see the ALIAS at the head of this
section) is bit-identical to QSA in exactly this regime — which is the regime of the free
oracle, of the planned ~64-token parity window and of the smoke decode cell. Any cell that must
distinguish the two needs **> 2051 cached tokens**, and nothing else does.

The output is a mask **combined with** the causal mask (`&` for
bool, `+` for float), never a replacement, with `−1` slots scattered to a dropped column
(MOD:704-717 in `QSAIndexer.forward`; consumed at MOD:793-797 in `Attention.forward`).

## 4 — Hyper-connections: Qwen's **Gated Residual (GR)**, not Sinkhorn mHC

TR §2.2 (pp. 10-11) derives GR, drops the branch-mixing operator `H_res` outright ("`H_res`
adds little") and finds the static terms `H_⋆^s` bring "no improvement for GR". So:
**low-rank dynamic**, with no counterpart to DeepSeek-V4's doubly-stochastic `H_res`
(scope: v4) — LC#27742 refused a shared base class for the same reason.

Per sublayer, with `n_r = hc_count = 4` and `d = 2560` (TR Eq. 30-34 ≡ MOD:941-969, the
whole of `GatedResidual`):

```
R̂_i = RMSNorm(R_i; γ_i)                       group RMSNorm, group_size = d      (30)
G   = unvec σ( W_u  SiLU( (1/n_r) W_d vec(R̂) ) )                                 (31)
x   = (1/n_r) Σ_i G_i ⊙ R̂_i                   MEAN, not sum                      (32)
s   = 2 σ( (1/n_r) W_w vec(R̂) ) ∈ R^{n_r}                                        (33)
R'_i = R_i + s_i · y,   y = F(x)              R_i is PRE-norm                    (34)
```

**Where `hc_lowrank: 320` enters**: it is the bottleneck rank `r = d/8 = 2560/8 = 320`
(TR after Eq. 32) — `W_d ∈ R^{r × n_r d}` = CKPT `input_mix_weight_down` **BF16 [320,
10240]**, `W_u ∈ R^{n_r d × r}` = `input_mix_weight_up` **BF16 [10240, 320]**, `W_w` =
`block_inject_weight` **BF16 [4, 10240]**, `hc_norm` **BF16 [10240]**.

**Pre/post per sublayer**: two independent GR modules per layer,
`attn_hyper_connection` and `mlp_hyper_connection` (MOD:1204-1205,
`DecoderLayer.__init__`), each reading before its sublayer and writing after (MOD:1222-1243,
`DecoderLayer.forward`). GR **replaces** pre-normalisation (TR p.11:
"Eq. (24) loses its Norm, and widening adds no normalization layer") — and CKPT indeed has
no `input_layernorm` / `post_attention_layernorm` tensor.

**Width-4 stream layout**: `hidden_states = inputs_embeds.repeat(1, 1, hc_count)` — seeded
by **repeating** the token embedding 4× into one contiguous 10240 vector, branch-major
(`unflatten(-1, (4, 2560))`) (MOD:1417, `Qwen4ExpTextModel.forward`). A fifth GR with
`use_combine=False`, `hyper_connection_mixer`, collapses 4→1 at the end and **is the model's
only final norm**; there is no `model.norm` (declared MOD:1330 and applied MOD:1430, in
`Qwen4ExpTextModel.__init__` and `.forward`; `Qwen4ExpTextModel.__init__` is MOD:1323-1332 in full, and it has
no `self.norm`; CKPT agrees).

> **CITATIONS CORRECTED 2026-08-31.** The two pointers here were MOD:1495 and MOD:1508, which
> both land inside `load_balancing_loss_func` — a router aux-loss accumulator. Both CLAIMS are
> true (re-verified against the pinned blob), and "no final norm exists" is recorded nowhere
> else in this tree, so a reader who checked the citation and found a `bincount` would have
> dropped a load-bearing fact. `lm_head.weight` is its own tensor and
`tie_word_embeddings: false`, so despite MOD's `_tied_weights_keys` the head is **untied**.

## 5 — Hashed n-gram embeddings (PLE)

`ngram_size: 3`, `heads_per_ngram: 8` → `ngram_heads = (3−1)·8 = 16`;
`ple_embed_dim: 2560` → `head_dim_per_ngram = 2560/16 = 160` (MOD:1019-1050,
`NGramEmbedding.__init__`).

**Hash** (MOD:976-995, the module-level `_PRIME_1`, `_splitmix64` and
`_build_layer_multipliers`; mixing at MOD:1098-1110 in `NGramEmbedding.forward`).
`half_bound = ((2^63−1) // vocab_size) // 2` with `vocab_size = 248320`;
`base_seed = seed + 10007·ple_layer_index` with `seed` **defaulting to 1234** (CFG:156 — CKPT
does not set it); for `i = 0,1,2`,
`m_i = 2·(splitmix64(base_seed + 0x9E3779B97F4A7C15·(i+1)) mod half_bound) + 1`, always odd.
Then with `t_s` = the token history shifted right by `s`, all in int64:

- bigram heads 0..7:   `mix = (t_0·m_0) XOR (t_1·m_1)`
- trigram heads 8..15: `mix = (t_0·m_0) XOR (t_1·m_1) XOR (t_2·m_2)`
- head `h`: `id = (mix mod V_h) + off_h`, where `off_h` is the exclusive prefix sum of the
  head vocabularies (MOD:1041-1042) and `V_h` is the **`(global_head_idx + 1)`-th prime
  strictly greater than `ngram_vocab_size_base − 1 = 19,999,999`**, with

  ```
  global_head_idx = ple_layer_index · ngram_heads + head_idx          (MOD:1038)
  V_h             = _find_nth_prime_after(base − 1, global_head_idx + 1)   (MOD:1039)
  ```

**The layer index is in BOTH halves of the per-PLE-layer indexing, and stating only one of
them is the trap.** `base_seed = seed + 10007·ple_layer_index` carries it for the multipliers;
`global_head_idx` carries it for the vocabularies. This file said "`V_h` is the `(h+1)`-th
prime", which is the `ple_layer_index == 0` specialisation — true for THIS checkpoint, because
`ple_layer_ids` has exactly one entry, and silently wrong for any later model or milestone with
two PLE layers: the multipliers would reseed while the primes restarted at 20000003, colliding
both layers' 16 heads onto the same table rows. That is a plausible-looking output, not a
crash. **And the specialisation is the evidence**: index 0 reproduces the checkpoint's
20000003…20000171 and `[23703573157769, 20109073645365, 8052911324071]` byte-exactly while
index 1 gives 20000213…20000429, so the byte-exact match IS the proof that
`ple_layer_index == 0` — which is the more useful thing to have written down.

**Gate owed (S4, deviceless).** The whole derivation is ~30 lines of integer arithmetic over
`u64`, so it belongs in Rust beside the converter rather than in a golden that pins only the
three multipliers — a wrong `global_head_idx` passes such a golden, because the multipliers do
not depend on `head_idx` at all. The gate re-derives all 16 primes, the 16 offsets and the 3
multipliers from `seed` and `ple_layer_index` and byte-compares them against the I64 values
recorded in `qwen-reference/tensor-families.tsv`, both at index 0 (must match) and at index 1
(must NOT).

**This is MEASURED, not transcribed** — three independent re-derivations against CKPT's I64
buffers and shard geometry (a fourth, at review, reproduced them again from scratch):

| quantity | re-derived | checkpoint |
|---|---|---|
| `layer_multipliers` I64 [3] | `[23703573157769, 20109073645365, 8052911324071]` | identical ✓ |
| `ngram_heads_vocab_sizes` I64 [16] | 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 | identical ✓ |
| `ngram_heads_offsets` I64 [16] | exclusive prefix sums, last 300001275 | identical ✓ |
| padded rows | `ceil(320,001,446 / 128)·128 = 320,001,536` | 128 shards × **F8_E4M3 [2500012, 160]** = 320,001,536 ✓ |

320,001,536 × 160 = **51.2 G elements** — the "51B n-gram table" figure — and it is the
*only* non-expert tensor in CKPT carrying a scale: one **BF16 [1] `weight_scale` =
1.9931793212890625e-4**, against the routed experts' block-128 `weight_scale_inv` grids,
which are **BF16 and per projection `[out/128, in/128]`**: `down_proj` (weight 2560×640) is
**[20, 5]** and `gate_proj`/`up_proj` (weight 640×2560) are **[5, 20]**.

> **CORRECTED 2026-08-31.** This read "the experts' `weight_scale_inv` **BF16 [20, 5]**", which
> is `down_proj`'s orientation stated for all three — and the sibling
> `qwen-reference/tensor-families.tsv` contradicts it on its own rows. A converter that built
> the grid as [20, 5] for `gate_proj`/`up_proj` would index a [5, 20] grid transposed, giving
> each 128×128 block a different block's scale: non-crashing, silently wrong dequant across
> 80.5 GB — two thirds of the expert weight — and invisible to any gate that checks only
> tensor counts and dtypes. The orientation is a **converter** assert at S4, per projection. `split_ngram_parts: 128` is the shard count; MOD has no `shard_` handling, so shard
reassembly is a checkpoint-side convention (OPEN-2).

**Injection point — `ple_layer_ids` is ONE-INDEXED.** `ple_layer_ids: [2]`, taken as
`layer_idx + 1 in config.ple_layer_ids` (MOD:1202, `DecoderLayer.__init__`); CFG:240-246
rejects ids outside
`[1, num_hidden_layers]` and CFG:248-254 requires a `linear_attention` host; TR p.15 says
"We place it at Layer 2, allowing host-memory prefetching to overlap with the computation of
the first layer"; CKPT's only PLE tensors sit under **`layers.1.`**. Host = **`layer_idx =
1`, the second layer**.

**Semantics and scaling on inject** (MOD:1176-1189 in `PLELayer.forward`, with the read at
MOD:1217-1220 in `DecoderLayer.forward`; no TR coverage, code is the only
source): the output is the **full 10240-wide** hc stream, added at the **top** of the layer
*before* the attn GR read — `hidden_states += ple(...)`. Inside:
`e = ngram_embed(ids)` (16 × 160 → 2560); `k̂ = norm_key(W_k e)` and
`q̂ = norm_query(hidden)`, both `∈ R^{4×2560}`; `g = (k̂ ⊙ q̂).sum(-1)/sqrt(2560)` per branch,
then the **signed square root** `g ← sign(g)·sqrt(max(|g|, 1e-6))`;
`v = σ(g) · W_v e` broadcast over branches; and
`out = v + SiLU(DilatedDepthwiseConv(norm_conv(v)))`, the conv being
`PLELayer._short_conv` (MOD:1150-1167, whose `F.silu` is MOD:1164). There is **no scalar multiplier on
inject** — the gate *is* the scaling.

The PLE conv is `ple_conv_kernel_size = 4` with **`dilation = ngram_size = 3`**, depthwise
over 10240 channels, bias-free, state width `(4−1)·3 = 9`, so token `t` reads taps
`{t−9, t−6, t−3, t}` with `w[0..3]` — the same newest-last convention as GDN (CKPT
`ple.conv1d.weight` **BF16 [10240, 1, 4]**). History is padded with **`config.eos_token_id`
= 248044** (CKPT `text_config`, scalar; MOD:1076 is where an absent cache is filled with it)
and `_shift_right_ignore_eos` resets the n-gram context at every EOS, so an n-gram never spans
a document boundary (MOD:1053-1067);
`generation_config.json`'s `eos_token_id: [248046, 248044]` is a *different* object and the
hash uses the config scalar. **CKPT's I64 tensors hold exactly the three buffers above**,
one set, layer 1 only — and being derivable is what made the table a check, not a copy.

## 6 — Router

**Softmax, not sigmoid, and NO bias correction.** `Qwen4ExpTextTopKRouter` is
`F.linear(h, W)` with `W` = **BF16 [512, 2560]** and *no bias tensor at all*, then
`softmax(dim=-1, dtype=float32)` over all `num_experts = 512`, then
`topk(num_experts_per_tok = 10)` (MOD:898-916, the whole of `TopKRouter`). No `e_score_correction_bias` exists in the
module or in CKPT, so the GLM/DeepSeek scar — a router bias that is an `nn.Buffer`, invisible
to `named_parameters` (`glm-reference/anchor.md`) — **does not transfer**; the absence here is
structural. `norm_topk_prob` is absent from CKPT's config and **defaults to `True`**
(CFG:163), so the 10 kept probabilities are renormalised to sum to 1 *after* the top-k
(MOD:912-913, `TopKRouter.forward`). TR says nothing about router scoring; §3.1 only notes it stays on AdamW.

**Shared-expert gating** (MOD:919-938, `SparseMoeBlock`): one shared `Qwen4ExpTextMLP` at
`shared_expert_intermediate_size = 640` (= `moe_intermediate_size`), scaled by
`σ(shared_expert_gate(h))` (**BF16 [1, 2560]**), then **added** to the routed sum. CKPT
stores routed experts **per-expert** (`mlp.experts.{0..511}.{gate,up,down}_proj`, F8_E4M3 +
block-128 `weight_scale_inv`) where MOD expects a fused 3-D `gate_up_proj` — the converter
owes the fuse, and `gate` precedes `up` in MOD's `chunk(2, dim=-1)` (MOD:889,
`Experts.forward`).

## 7 — Partial RoPE

`head_dim: 256`, `partial_rotary_factor: 0.25` → `dim = int(256 · 0.25) = **64**`;
`inv_freq = 1 / theta^(arange(0, 64, 2) / 64)`, 32 frequencies, `theta = rope_theta = 1e7`
(MOD:96-116, `RotaryEmbedding.compute_default_rope_parameters`; CKPT `rope_parameters`). `emb = cat(freqs, freqs)` → `rotary_dim = 64`.

**Split-half, over the FIRST 64 dims.** `apply_rotary_pos_emb` takes `rotary_dim` from
`cos.shape[-1]`, splits `q_rope = q[..., :64]` / `q_nope = q[..., 64:]`, rotates only
`q_rope` with `rotate_half` — which pairs `i` with `i+32` inside that 64 — and concatenates
back (MOD:566-570 `rotate_half`, MOD:591-600 `apply_rotary_pos_emb`). Dims 0..63 rotate as 32 `(i, i+32)` pairs; dims 64..255 pass through.
**Not interleaved `(i, i+1)`**, and **not** the last 64.

mRoPE (`mrope_interleaved: true`, `mrope_section: [11, 11, 10]`, summing to the 32
frequencies) is **inert for text-only v1**: 2-D `position_ids` expand to 3 identical grids
(MOD:121-124, `RotaryEmbedding.forward`), so `apply_interleaved_mrope` overwrites `freqs[0]`
slices with identical values (MOD:140-155); vision is what makes it bite.
`attention_scaling = 1.0` under `rope_type: "default"` (MOD:113, whose own comment says
"Unused in this type of RoPE").

**YaRN is opt-in and absent from every released config.** CKPT sets `rope_type: "default"`;
the only primary statement of the factor-4 form is CARD §"Processing Ultra-Long Texts" —
`{"rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0,
"original_max_position_embeddings": 262144}`, keeping `mrope_interleaved`/`mrope_section`,
called **static** YaRN and advised against for short contexts. Mechanics are then
transformers' generic `_compute_yarn_parameters` (ROPE): NTK-by-parts over `dim = 64`,
`beta_fast = 32`, `beta_slow = 1`, `truncate = True`,
`inv_freq = interp·(1−ε) + extrap·ε` with `ε = 1 − linear_ramp(low, high, 32)`, plus — the
part a port forgets — a non-unit **`attention_factor = 0.1·ln(4) + 1 =
1.138629436111989`** on both `cos` and `sin`. At `theta = 1e7`, `dim = 64`,
`original_max = 262144`: `low = 14`, `high = 22`. Deferred for v1; the form is pinned so the
deferral is a milestone, not a guess.

## Traps table (each row = a wrong-but-plausible reading, and how S2 catches it)

The plan's requirement is **≥2 rows for each of the nine named classes** (GDN decay form, conv
tap order, output-gate activation, indexer relu/pool/budget, hc transpose + lowrank order,
router bias/norm/w1w3, n-gram seed/order/layer, partial-rope width, eps homes) — not ≥18 rows
total.

> **EXTENDED 2026-08-31 (S1 review).** The table met the count and not the rule: the
> distribution was 5/5/5/2/2 across four classes and 1-or-0 across the rest, so a matrix built
> off it would read complete while carrying zero coverage of the GDN decay FORM, of where the
> epsilon lives, and of the router's gate/up order. Six rows were added for exactly those gaps
> (T20-T25) plus T19 for the `layer_types` alias, which had no row at all and is the only trap
> here that is invisible to every gate this port currently plans.

**The nine classes against the defect rows that cover them** — so the ≥2 rule is checkable
here rather than asserted, and S2's matrix has a census to answer instead of a count to hit.
"Rows" means defect rows in the matrix, which is what the plan counts; several table rows below
name two.

| class | defect rows | table rows |
|---|---|---|
| GDN decay form | `gdn_decay_sigmoid_gate`, `gdn_decay_clamped_lower_bound`, `dt_bias_broadcast_wrong` | T20, T17 |
| conv tap order | `conv_tap_reversed`, `conv_tap_rotated_by_one` | T2, T21 |
| output-gate activation | `output_gate_silu`, `output_gate_tanh`, `output_gate_on_normed_stream` | T5, T22 |
| indexer relu/pool/budget | `indexer_relu_off`, `indexer_sum_over_blocks`, `indexer_rope_at_block_end`, `indexer_rope_before_pool`, `indexer_budget_confused`, `qsa_layer_is_dense` | T6, T7, T8, T19 |
| hc transpose + lowrank order | `hc_res_mixer_added`, `hc_static_weights`, `hc_sum_not_mean`, `hc_residual_uses_normed`, `hc_lowrank_transposed` | T9, T10, T11 |
| router bias/norm/w1w3 | `router_sigmoid`, `router_bias_added`, `router_no_renorm`, `expert_gate_up_swapped` | T15, T24, T25 |
| n-gram seed/order/layer | `ngram_seed_wrong`, `ngram_mod_round`, `ngram_mix_add`, `ngram_order_swapped`, `ple_layer_off_by_one` | T12, T13, T14 |
| partial-rope width | `rope_interleaved_pairs`, `rope_on_tail_dims` | T16 |
| eps homes | `eps_outside_sqrt`, `eps_from_the_wrong_config_field` | T23 |

Four further classes carry rows without being on the plan's list — norm offset (T1), fused-QKV
segmentation (T3), attention gate split (T4), state axis order (T18) — and each of those is a
defect this tree has been bitten by on another port, which is why they were written first.

| # | wrong-but-plausible reading | why it is tempting | S2 defect row that reddens it |
|---|---|---|---|
| T1 | GDN output norm is zero-centred `(1+w)` like every other norm | TR p.4 says "the same formulation is consistently applied to all other RMSNorm layers", and Fig. 2 labels it "Zero-Centered RMSNorm" | **MEASURED — code wins.** `layers.0.linear_attn.norm.weight` (n=128) is mean **+0.96677**, min +0.875, max +1.02344 → ones-centred, matching `RMSNormGated`'s bare `w ⊙ x̂` (MOD:198 — MOD:199 is the line AFTER, which applies the gate). Compare with `layers.0.attn_hyper_connection.hc_norm.weight` (n=10240) mean **−0.06355**, min −5.9375, max +6.625 → zero-centred, matching `(1.0 + w)` (MOD:177, `RMSNorm.forward`). Bytes, offsets and the in-tree fixture are in **T1 provenance** below. Two rows: `gdn_norm_unit_offset` and `hc_norm_no_offset`, each shown to redden only its own norm |
| T2 | conv taps oldest-last (`w[0]` = current token) | a natural read of a causal window, and a mirror-symmetric weight makes it invisible | `conv_tap_reversed`: reverse `w` along the kernel axis; must move GDN q/k/v and the PLE output. Fixture must use a **non-palindromic** kernel or the row is vacuous |
| T3 | fused `in_proj_qkv` split as `[V|K|Q]`, or head-interleaved | 10240 = 2·2048+6144 admits several segmentations; LC#27742 reports its own arch test could not tell three of them apart | `qkv_segmentation_swapped` ×2 (two wrong orders). Requires **distinct** per-head weight draws — equal draws make it green |
| T4 | attention `q_proj` is `[query(6144) | gate(6144)]` | the reshape is easy to miss | It is **per head**: `view(..., 24, 512)` then `chunk(2, dim=-1)` (MOD:805-807, `Attention.forward`), so head `h` owns rows `[512h, 512h+256)` for query and `[512h+256, 512h+512)` for gate. `attn_gate_block_split` must redden |
| T5 | GDN output gate is SiLU | it is fla/GDN's own default, and `RMSNormGated`'s `activation` parameter defaults to `"silu"` | `output_gate_silu`: TR p.4 and `output_gate_type: "sigmoid"` both name it; σ vs SiLU differ on all inputs, so a value row suffices |
| T6 | indexer scores are `Σ_h w_h·ReLU(...)` with learned head weights | the plan's own phrasing, and DSA-family indexers do carry them | `index_qk_proj` is the **only** indexer projection (CKPT: 3 tensors/layer). `indexer_head_weights` cannot even be planted without inventing a tensor — instead plant `indexer_relu_off` and `indexer_sum_over_blocks` (sum on the wrong axis) |
| T7 | pooled indexer keys get RoPE at the block's LAST token, or before pooling | "block position" is ambiguous, and pooling-then-roping is the unusual order | `indexer_rope_at_block_end` and `indexer_rope_before_pool`. TR Eq. 13-14 and MOD:679-688 (`QSAIndexer.forward`) both say pool → norm → RoPE at `p_b` = first token |
| T8 | budget 2048 means 2048 blocks, or 512 tokens | "512 blocks or 2048 tokens" is the same fact stated twice | `indexer_budget_confused`: at ≤2051 cached tokens QSA **must be bit-identical to dense** — from MOD:695's `topk(min(block_topk, num_complete_blocks))`, corroborated by LC#27742's measured 0.0 max logit delta over 2051 rows; above it, selection must change. Both directions in one row, and the ABOVE direction is the only one with any power (see T19) |
| T9 | hyper-connections are static mHC with a Sinkhorn-normalised `H_res` | DeepSeek-V4 (scope: v4) is in this tree and looks adjacent | TR p.11 drops `H_res` outright; CKPT has **no** `n_r × n_r` tensor. `hc_res_mixer_added` / `hc_static_weights` |
| T10 | GR read collapses branches by **sum**, or reads the *normed* stream into the residual | mean vs sum differs by exactly 4×, and `hyper_input` vs `hyper_input_normed` is one identifier apart | `hc_sum_not_mean` (4× on every layer, so it compounds) and `hc_residual_uses_normed` (MOD:969 returns the **pre**-norm `hyper_input`) |
| T11 | `hc_lowrank: 320` is arbitrary, or is the *output* width | it is `d/8`; a transposed `W_d`/`W_u` still type-checks in a shape-agnostic port | shapes are asserted from CKPT: down **[320, 10240]**, up **[10240, 320]**. `hc_lowrank_transposed` |
| T12 | PLE injects at 0-indexed layer 2 | `ple_layer_ids: [2]` reads as an index everywhere else in the config | **1-indexed**: CFG:240-246 rejects `0`, and CKPT's PLE tensors are under `layers.1.` `ple_layer_off_by_one` must redden, and a converter that emits `layers.2.ple.*` must fail its census |
| T13 | the n-gram hash mixes with `+` or a single multiplier, or uses `mod 20,000,000` | 20M is the config field name and the round number invites it | the 16 vocabularies are **16 distinct primes** and the multipliers are splitmix64-derived — re-derive them and byte-compare the I64 buffers (done above). `ngram_seed_wrong`, `ngram_mod_round`, `ngram_mix_add` |
| T14 | bigram and trigram heads share a hash, or the order is trigram-first | both blocks read `shifted_tokens[0..]` from the same list | heads 0..7 are **bigram**, 8..15 **trigram** (MOD:1098-1100, where `start_idx = (ngram - 2) * heads_per_ngram`). `ngram_order_swapped` — and because each head has its own prime, a swap changes every id |
| T15 | router uses sigmoid scoring, or a top-k bias correction | every other MoE in this tree has one | `router_sigmoid`, `router_bias_added`. There is no bias tensor to zero, so the plant must ADD one — and adding one must redden |
| T16 | partial RoPE is interleaved `(i, i+1)`, or covers the last 64 dims | GLM/Qwen2 lineage is interleaved in places, and `mrope_interleaved: true` invites the confusion (it is about the *mRoPE sections*, not the pairing) | `rope_interleaved_pairs`, `rope_on_tail_dims`. Fixture needs `head_dim > rotary_dim` or both are vacuous |
| T17 | `dt_bias`/`A_log` are per-channel, or fixed at their init values | `torch.ones` / `uniform_(0.01,16)` sit right there in `__init__` | both are **[48]** and learned (measured ranges above). `dt_bias_broadcast_wrong` |
| T18 | the GDN state is `[value][key]` | it is square `[128][128]`; every shape check passes either way (scar: `square-tensors-hide-their-axis-order`) | score BOTH readings against the reference output, as K3's KDA fixture does; TR p.3 pins `[d_k × d_v]` |
| **T19** | `layer_types`' 12 `full_attention` entries mean dense global attention, so those layers need no indexer | it is the checkpoint's own spelling, it is the spelling every other HF config uses for dense attention, and MOD uses the identical string as a causal-mask key (MOD:1404, 1423) | **The reference ALIASES it**: CFG:180-184 rewrites `full_attention` → `qwen_sparse_attention` before `validate_architecture` (CFG:190) will accept the config, with the comment "layers that are actually using an indexer". `qsa_layer_is_dense` must redden — and it can ONLY redden **above 2051 cached tokens**, because below that QSA selects every block (MOD:695) and dense is bit-identical. So this row's fixture carries its own context length, and a matrix run only at the free-oracle width scores it vacuously. Partial backstop: census item 3 sees 36 unconsumed indexer tensors — but not if the converter emits them and the arm ignores them |
| **T20** | GDN decay is a "safe gate" — `g = -exp(A) · sigmoid(...)`, or a clamped lower bound like KDA's | fla's GDN family has both forms, K3's KDA (scope: k3) carries `gate_lower_bound` in this very tree, and `softplus` vs `sigmoid` differ only in scale on small inputs | `gdn_decay_sigmoid_gate` and `gdn_decay_clamped_lower_bound`: MOD:519 is `-A_log.float().exp() * F.softplus(a.float() + dt_bias)` with no clamp anywhere, and CKPT carries no `gate_lower_bound` field. Both plants must redden a decay-sensitive value row; the fixture needs `\|a + dt_bias\|` large enough that softplus and sigmoid separate |
| **T21** | the conv kernel is time-reversed as a whole (`w[3..0]`), not just the tap order | T2 plants a REVERSAL of the whole kernel and a palindromic weight hides it; the subtler case is a kernel rotated by ONE tap, which survives a symmetry check | `conv_tap_rotated_by_one`: shift `w` cyclically by one along the kernel axis. It changes every output while leaving the weight multiset intact, so a fixture that checks only sums or norms passes it. Applies to GDN q/k/v (kernel 4, dense taps) and to the PLE conv (kernel 4, **dilation 3**), where the rotation moves the taps from `{t−9,t−6,t−3,t}` to `{t−6,t−3,t,t+3}` — a FUTURE tap, so this plant must also be caught by a causality assert |
| **T22** | the GDN output gate is `tanh`, or `sigmoid` applied to the normed stream rather than to `z` | TR calls it "the bounded sigmoid gate" and `tanh` is the other bounded gate; and `RMSNormGated` takes both `hidden_states` and `gate`, so which argument the activation wraps is one identifier apart | `output_gate_tanh` and `output_gate_on_normed_stream`. MOD:199 is `hidden_states * ACT2FN[activation](gate.to(float32))` — the activation wraps `gate`, `gate` is `in_proj_z`'s output, and `tanh` vs σ differ on every input including sign. Distinct from T5 (SiLU), which is the fla default rather than the other bounded gate |
| **T23** | the RMS epsilon goes outside the square root (`x/(rms+eps)`), or inside a *different* norm than the one that owns it | `rsqrt(mean(x²)+eps)` and `1/(sqrt(mean(x²))+eps)` are one paren apart and agree to ~1e-7 on well-scaled inputs; the model has FIVE eps homes (`rms_norm_epsilon` for the trunk norms, `layer_norm_epsilon` for `RMSNormGated`, `1e-6` hard-coded in `l2norm`, `1e-6` in the PLE signed-sqrt clamp, `1e-6` in the vision LayerNorms) | `eps_outside_sqrt` and `eps_from_the_wrong_config_field`: MOD:170 and MOD:197 both put it INSIDE (`rsqrt(... + eps)`), `l2norm` (MOD:261) carries its own literal, and `GatedDeltaNet.__init__` (MOD:437-438) passes `layer_norm_epsilon` and not `rms_norm_eps`. The fixture needs a row whose RMS is small enough for eps to bite — a well-scaled draw makes both plants green, which is what makes this class worth two rows |
| **T24** | the fused expert `gate_up_proj` is `[up \| gate]`, so the SiLU lands on the wrong half | CKPT ships `gate_proj` and `up_proj` as SEPARATE per-expert tensors and MOD expects one fused 3-D `gate_up_proj`, so the converter chooses the order — and "w1/w3" naming across families disagrees about which is which | `expert_gate_up_swapped`: MOD:889-890 is `gate, up = linear(...).chunk(2, -1)` then `act_fn(gate) * up`, so `gate` is the FIRST half. Swapping applies SiLU to `up` — non-crashing, changes every routed token. Needs distinct gate/up draws; equal draws make it green. Pairs with the scale-grid orientation ([20,5] vs [5,20], §5) as the two silent expert-side converter defects |
| **T25** | the router's top-10 probabilities are used as softmax gave them, with no renormalisation | `norm_topk_prob` is **ABSENT from CKPT's config**, so a port that reads only the file finds nothing to honour — and the ten kept probabilities sum to ≈1 anyway on a peaked router, so the error is a small per-token scale | `router_no_renorm`: CFG:163 defaults `norm_topk_prob: bool = True` and MOD:912-913 divides by the kept sum. The plant is to SKIP the division; the fixture must use a FLAT router logit draw, where the kept ten sum to ≈10/512 and the un-renormalised output is ~50x too small. A peaked draw makes this row vacuous, which is the whole trap: the config field is absent, the default is on, and the error hides on realistic weights |

## T1 provenance — the four measured weights, re-fetchable byte-for-byte

**Every measured-weight statistic in this file comes from these four tensors, and T1 is the one
place this doc OVERRIDES a primary source with a measurement** ("code wins" over TR p.4's
zero-centred claim). It also drives a defect row a kernel will be scored against
(`gdn_norm_unit_offset`), so where the bytes came from is part of the evidence and not a
footnote. All four are in `model-00001-of-00131.safetensors`, whose `header_len` is 39264, so
`absolute = 8 + 39264 + data_offset` = `data_offset + 39272`. Shard assignment is from CKPT's
`model.safetensors.index.json`; `data_offsets` are from that shard's own safetensors header.

| tensor (under `model.language_model.`) | dtype, n | `data_offsets` | absolute byte range | measured |
|---|---|---|---|---|
| `layers.0.linear_attn.norm.weight` | BF16, 128 | `[110796992, 110797248]` | `[110836264, 110836520)` | mean **+0.9667663574**, min +0.875, max +1.0234375 → **ones-centred** |
| `layers.0.attn_hyper_connection.hc_norm.weight` | BF16, 10240 | `[13209600, 13230080]` | `[13248872, 13269352)` | mean **−0.06355**, min −5.9375, max +6.625 → **zero-centred** |
| `layers.0.linear_attn.dt_bias` | BF16, 48 | `[26419296, 26419392]` | `[26458568, 26458664)` | mean **−3.29008**, min −8.0, max +2.53125 → learned, not `torch.ones` |
| `layers.0.linear_attn.A_log` | BF16, 48 | `[26337280, 26337376]` | `[26376552, 26376648)` | mean **+1.53425**, min −3.57812, max +5.0625 → learned, not `uniform_(0.01,16)` |

**The first row's bytes are VENDORED**, because it is the row that overrules the tech report:
`docs/measurement/qwen-reference/linear_attn_norm_l0.bin`, 256 B, sha256 `ba3568fc5413…`,
fnv1a64 `7159c1b3a9e167f7`, recorded in `tensor-families.tsv`'s VENDORED BYTES block.
`crates/artifact/tests/qwen_names.rs` decodes it on every deviceless run, re-derives the
verdict from the values rather than reading the digits above
(`the_gdn_output_norm_weight_is_ones_centred`), and recomputes the hash from the live file — so
S2 inherits a checked fact instead of a paragraph. The discriminant it gates is `min > 0.5`,
which separates ones- from zero-centred by a margin no narrowing can close; the digits are
pinned transitively by the hash.

The other three are not vendored — the table above is enough to re-fetch each with one HTTP
range request against `resolve/<the CKPT revision>/model-00001-of-00131.safetensors`. Vendoring
`hc_norm` would cost 20,480 B for a statistic that contradicts no source, and the two [48]
tensors settle T17, which the shapes alone already decide.

## OPEN — what the sources do not settle (these become S2 taps)

1. **`ngram_embedding.weight_scale` direction and granularity.** One BF16 scalar
   (1.9931793212890625e-4) named `weight_scale`, against the experts' `weight_scale_inv`.
   `448 × 1.99e-4 ≈ 0.089` makes `w = q · scale` plausible, but that is arithmetic, not a
   source. **Tap**: recover a row's dequantised norm from a forward capture.
2. **Shard reassembly for the 128 `shard_N.weight` tensors.** MOD has no `shard_` handling;
   CFG carries only the count. Row-major concatenation in *numeric* order is the obvious
   read, and `shard_10` before `shard_2` under a lexical sort is exactly the silent kind of
   wrong. **Tap**: reconstruct one known id's row two ways.
3. **Block alignment: TR vs code.** TR Eq. 13 sets `p_b = b·r` (absolute multiples of `r`);
   MOD:675-677 (`QSAIndexer.forward`) chunks the **visible-index list** by `r`. Identical for a plain causal row
   from position 0, divergent under left padding or any non-contiguous mask. **Tap**: a
   left-padded batch.
4. **fp32 boundaries.** MOD forces fp32 in the RMSNorms, the gated-norm activation, both
   softmaxes, the indexer matmul, the block-key pooling and the delta rule
   (`mamba_ssm_dtype: "float32"`). Which of those a bf16 kernel may collapse is a
   *tolerance* question with no source — measurement, per operator, ≥2 draws, at S2.
5. **Recurrent- and residual-state dtype in flight.** CKPT sets `mamba_ssm_dtype:
   "float32"`; TR p.2 says "the residual state supports FP8 storage" of the *hc* stream.
   Neither is a decode-time contract. **Tap**: A/B against the parity window.
6. **Chat template / thinking framing.** `chat_template.jinja` is in CKPT (8,952 B) and is
   track D's; nothing here constrains it. The GLM drift scar applies verbatim.
7. **YaRN activation.** No released config enables it (CARD only); form pinned above.
8. **Top-k tie-breaking** in both the router's `topk(10)` of 512 and the indexer's
   `topk(512)` of blocks — `torch.topk`'s order on exact ties is unspecified and bf16 ties
   are not rare. **Tap**: measure the tie rate before assuming it cannot move an argmax.
9. **`heads_per_ngram: 8` vs `split_ngram_parts: 128` vs
   `make_ngram_vocab_size_divisible_by: 128`** are three unrelated 8/128s in one subsystem,
   disambiguated only by the code paths cited; a census assert on each is owed at S4.
10. **MTP and vision counts.** MOD:1256 (`Qwen4ExpPreTrainedModel._keys_to_ignore_on_load_unexpected`) excludes `^mtp.*` by name and CKPT's `mtp.*` /
    `model.visual.*` families are structurally separate, but the counts (1 MTP layer with
    its own 512 experts; 27 vision blocks) are S4 census data, not settled here.
    *(2026-08-31: the PARAMETER halves are settled and gated —
    `qwen-reference/tensor-families.tsv` declares 2,607,304,448 MTP and 448,931,056 vision
    parameters, re-derived from shape × count by
    `qwen_names.rs::the_parameter_counts_split_by_v1_status_are_re_derivable`. What stays open
    is the converter's by-name exhaustiveness over those families, which is item 3 of the
    census and not a number.)*

## Worklog

**2026-08-31 (later, S1 review pass)** — every MOD/CFG line citation re-derived mechanically
against the pinned blobs, whose identity was re-confirmed locally (`git hash-object` reproduces
`7af639771c07…` for 124,924 B and `3dd586b84458…` for 16,463 B). Twelve pointers moved, two of
them out of `load_balancing_loss_func`, and each surviving citation gained its enclosing symbol
so a patch bump degrades it to a hint. Corrected in place, each at its own site: the
read-policy sentence (a 20,480 B read is not a "≤256-byte range"), the expert scale-grid
orientation ([20,5] is `down_proj` alone; `gate_proj`/`up_proj` are [5,20], which the sibling
tsv's own rows said all along), the heading's "nothing vendored here", and the n-gram prime rule
(stated in its `global_head_idx` form, with the byte-exact match at index 0 recorded as the
EVIDENCE that `ple_layer_index == 0` rather than as an assumption). Added: the `layer_types`
ALIAS — the checkpoint's `full_attention` entries are the layers that RUN the indexer, aliased
by CFG:180-184, invisible to every planned oracle below 2051 tokens — as §3's opening subsection
and trap T19; six trap rows closing the per-class gaps (T20-T25); the ≤2051-dense argument
restated from MOD:695 with LC#27742 demoted to corroboration; a class→defect-row map so the ≥2
rule is checkable; and **T1 provenance**, with the 256 bytes that settle T1 vendored in tree and
gated by `crates/artifact/tests/qwen_names.rs`. Nothing here is a tolerance still.

**2026-08-31** — written for S0 from TR, MOD/CFG/ROPE at transformers v5.16.1 `93c8b7b4`
and CKPT @ `236dfdf2`. LC#27742 was read only as a cross-check; it agreed on the sigmoid
gate, the 2051-row dense-QSA oracle, the 51.2 G-element table and the non-Sinkhorn
hyper-connection, and contributed trap T3 — every one of those now stated from MOD or TR
instead, with LC as corroboration. The n-gram hash became measurement rather than
transcription the same day — three independent re-derivations matched CKPT's I64 buffers and
shard geometry byte-exactly — and T1's TR-vs-MOD disagreement was settled by reading 128
BF16 weights out of `model-00001`. Nothing here is a tolerance; every tolerance is owed to
S2 on ≥2 weight draws.
