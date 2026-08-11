---
scope: glimmer
status: live
verdict: Muse Glimmer-30B's forward pass, precise enough to write kernels from — extracted 2026-08-10 from first-party sources only (raw config.json, the safetensors headers by range request, and transformers' own modeling_muse_glimmer.py), no summarizing fetch in the chain. 52 dense layers, 39 sliding (window 2048) + 13 full at zero-based 3,7,...,51, GQA 32Q/2KV head_dim 128. SANDWICH NORMS: four per layer, post-norms on the BRANCH before the residual add, and they are CENTERED (x*(1+w)) while the final norm and the two weightless norms are plain (x*w) — two formulas in one model. Q and K carry a WEIGHTLESS RMSNorm that ships no tensor (with_scale=False) and Q alone is then scaled by qk_scale_factor 3.87. RoPE is rotate_half (split-half), NOT rivoli's interleaved convention — a row permutation of q_proj/k_proj converts it, argued in section 6 and unproven. NoPE layers skip rotation entirely. Attention output is gated by sigmoid(gate_proj(layer input)) BEFORE o_proj. Logits are 20*tanh(x*0.196116/20), which is argmax-invariant — so every greedy gate is BLIND to it. Text side 55.71 GB bf16 = 26.51 GB/token at fp8. Fifteen traps in section 9. The DFlash drafter is a SEPARATE 2.556 B checkpoint (section 11): a 5-layer BIDIRECTIONAL cross-attention adapter that shares almost nothing with the target (32Q/8KV, plain pre-norm, weighted QK-norm) and borrows the target's embedding UNNORMED plus its lm_head; because the target is dense, break-even is N>1.1 accepted tokens per cycle, inverting the MoE-union economics that made ungated MTP a loss on GLM.
---

# Muse Glimmer-30B — the forward pass

Extracted 2026-08-10. `docs/investigations/glimmer-port.md` is the plan; this is the
specification it builds against. Text side only — the vision tower is out of scope.

**Provenance, one paragraph.** Every fact here comes from a first-party artifact fetched raw
with `curl`, not through a summarizing fetch layer. Sources: `config.json` and
`chat_template.jinja` from `meta-models/Muse-Glimmer-30B` pinned at
**`f84ecc3a0ea984a4c04542a84269e3d065350a6e`** (`lastModified` 2026-08-10T08:15:02Z); tensor
names, shapes and dtypes from `model.safetensors.index.json` **plus the safetensors headers
themselves**, read by HTTP range request (first 8 bytes little-endian `u64` = header length,
then that many bytes) so shapes are the shard's own and not the index's summary; arithmetic
from `transformers` `src/transformers/models/muse_glimmer/modeling_muse_glimmer.py` on
`main`, cited below as `M:<line>`. The repo ships **no** `modeling_*.py` of its own —
`muse_glimmer` is native to transformers 5.15.0.dev0, so the library *is* the first-party
implementation. **The announcement and model card were used for nothing**: they were the
plan's original provenance and §10 records where they were wrong.

> **PINNED 2026-08-11.** `main` moves, so "on `main`" was not a citable source and every
> `M:<line>` below was a reference to a file that could shift under it. The reference is now
> installed and pinned at transformers commit
> **`fe747d88a3296bd94d426db2717f232f9d4afdb7`** (`5.16.0.dev0`, one minor past the
> `5.15.0.dev0` this was written against), in a venv at `/home/rhansen/glimmer-anchor/venv`
> kept SEPARATE from K3's — that one is `4.56.2` and its version *is* the provenance of its
> vendored goldens, so upgrading it in place would invalidate them while leaving every byte
> and every gate green.
>
> **Re-verified against the installed file at that commit, not assumed:** the four
> `MuseGlimmerTextCenteredRMSNorm` per layer with `rms_norm_eps` on the two pre-norms and
> `post_norm_eps` on the two post-norms; `qk_norm` weightless with Q alone scaled by
> `qk_scale_factor` (still `M:323`, so the line citations still resolve); `rotate_half`;
> `attn_output * torch.sigmoid(self.gate_proj(hidden_states))` before `o_proj`; the
> `output_multiplier` → `/ softcap` → `tanh` → `* softcap` logit path; and the weightless
> `embed_norm`. Nothing in §1–§11 changed across the two versions.
>
> Torch here is **CPU** (`2.13.0+cpu`), not K3's `2.13.0+rocm7.2`. Deliberate: this reference
> is plain PyTorch with no triton kernel behind it — K3 needed a GPU because fla's KDA ops are
> triton-only — so a tiny-dims golden reproduces on CPU and costs the shared device nothing.

## 1. Shapes

```
hidden 6656 · layers 52 · vocab 202048 · head_dim 128 · heads 32 Q / 2 KV (groups 16)
intermediate 19968 (SwiGLU, silu) · sliding_window 2048 · max_position 131072
rms_norm_eps 1e-5 · post_norm_eps 1e-8 · attention_bias false · tie_word_embeddings false
qk_scale_factor 3.87 · output_multiplier 0.19611613513818404 (= 1/sqrt(26))
final_logit_softcapping 20.0 · rope_theta 500000.0 (rope_type "default", no scaling)
```

`num_attention_heads * head_dim = 4096 ≠ hidden 6656`. **`q_proj` is not square and
`head_dim` is not `hidden / n_heads`** — any code carrying that identity is wrong here and
must fail at parse, not at decode.

Per-layer tensors, all `BF16`, verified from the shard headers:

| tensor | shape |
|---|---|
| `input_layernorm` / `post_attention_layernorm` | `[6656]` |
| `pre_feedforward_layernorm` / `post_feedforward_layernorm` | `[6656]` |
| `self_attn.q_proj` | `[4096, 6656]` |
| `self_attn.k_proj` / `self_attn.v_proj` | `[256, 6656]` |
| `self_attn.gate_proj` | `[4096, 6656]` |
| `self_attn.o_proj` | `[6656, 4096]` |
| `mlp.gate_proj` / `mlp.up_proj` | `[19968, 6656]` |
| `mlp.down_proj` | `[6656, 19968]` |

Global: `model.language_model.embed_tokens.weight` `[202048, 6656]`,
`lm_head.weight` `[202048, 6656]`, `model.language_model.norm.weight` `[6656]`.
**Sliding and full layers are shape-identical** — the layer kind changes masking and RoPE
only, never a tensor.

## 2. The layer map, which needs no inference

`config.json` ships **two explicit 52-entry arrays**. Do not derive the pattern from a
modulus; consume the arrays.

- `layer_types[i]` ∈ {`sliding_attention`, `full_attention`}, pattern `[s,s,s,full]` × 13.
  Zero-based full layers = **3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51** (n=13);
  sliding n=39. Note **layer 51, the last layer, is full**.
- `layer_rope_theta[i]` = `500000.0` on sliding layers, **`0` on full layers**.

`layer_rope_theta` is consumed **as a boolean**, not as a per-layer base: the model builds one
cos/sin table from `config.rope_parameters.rope_theta` (M:513) and passes
`position_embeddings if self.config.layer_rope_theta[i] else None` (M:520). So there is **one**
RoPE table at θ=500000 and a per-layer on/off flag — not 52 tables. A port that reads the
top-level `rope_parameters.rope_theta` and applies it to all 52 layers produces fluent wrong
text on 13 of them.

## 3. The decoder layer — sandwich norms

`M:395-417`, verbatim in structure:

```
residual = h
h = input_layernorm(h)                  # centered, eps 1e-5
h = self_attn(h)                        # §4
h = post_attention_layernorm(h)         # centered, eps 1e-8  -- ON THE BRANCH
h = residual + h

residual = h
h = pre_feedforward_layernorm(h)        # centered, eps 1e-5
h = mlp(h)                              # silu(gate(h)) * up(h) -> down    M:156
h = post_feedforward_layernorm(h)       # centered, eps 1e-8  -- ON THE BRANCH
h = residual + h
```

**Four norms per layer, and the two post-norms normalise the branch output before the residual
add** — they are not applied to the residual stream. rivoli's existing layers are pre-norm
only, two per layer. **The two eps values differ by three orders of magnitude and are assigned
by position**: `rms_norm_eps` 1e-5 on the two pre-norms, `post_norm_eps` 1e-8 on the two
post-norms (M:381-384).

## 4. Attention

`M:326-371`. Per layer, with `hidden_states` = the post-`input_layernorm` activation:

```
q = q_proj(h)  -> view [.., 32, 128]        k = k_proj(h) -> [.., 2, 128]
v = v_proj(h) -> [.., 2, 128]

q = qk_norm(q) * 3.87                       # WEIGHTLESS RMSNorm over head_dim, THEN scale
k = qk_norm(k)                              # normed, NOT scaled       M:341-342

if layer_rope_theta[i] != 0:                # NoPE layers skip entirely  M:345-347
    q, k = apply_rotary_pos_emb(q, k, cos, sin)      # rotate_half, §6

k, v = cache.update(k, v)                   # cache stores POST-norm, POST-rope k
attn = softmax(q @ kᵀ * head_dim**-0.5 + mask) @ v    # scaling = 1/sqrt(128)   M:274,302
                                            # sliding layers mask to a 2048 window
attn = attn.reshape(.., 4096)
attn = attn * sigmoid(gate_proj(h))         # gate reads h, NOT attn      M:369
out  = o_proj(attn)                                                      # M:370
```

Four things here each produce fluent wrong text on their own:

1. **`qk_norm` ships no tensor.** It is `MuseGlimmerRMSNorm(eps=rms_norm_eps,
   with_scale=False)` (M:323) — a weightless RMSNorm over the last dim (`head_dim`=128),
   applied per head to **both** q and k. There is no `q_norm`/`k_norm` in the checkpoint
   because there is no weight to ship. **Absence of a tensor is not absence of an operation**;
   a port that enumerates tensors to decide what to implement will silently skip this.
2. **`qk_scale_factor` = 3.87 multiplies Q only**, after the norm (M:341). K does not get it.
   It is *not* a replacement for the softmax scale: `scaling` is still `head_dim**-0.5`
   (M:302) and both apply, for an effective Q factor of `3.87 / sqrt(128)` = 0.342.
3. **The gate is computed from the layer input, not the attention output** (M:369).
   `gate_proj` is `[4096, 6656]`, consumes `h`, and its `sigmoid` multiplies the attention
   output elementwise before `o_proj`. A port that gates on `attn` has the right shapes,
   the right tensor, and the wrong model.
4. **The KV cache holds post-qk_norm, post-RoPE K.** Order is norm → scale → rope → cache.

`repeat_kv` (M:249) broadcasts each of the 2 KV heads to 16 Q heads by
`expand(b, kv_heads, n_rep, s, d).reshape(b, kv_heads*n_rep, s, d)` — so Q head `j` uses KV
head `j // 16`, **not** `j % 2`. Both mappings type-check and decode fluently.

## 5. Embedding, final norm, and the logit path

- **The embedding is normed by a weightless RMSNorm** (`MuseGlimmerTextNormedEmbedding`,
  M:436-444, eps `rms_norm_eps`). Its own comment says why it cannot be folded into the
  embedding matrix: *"cannot be merged to the embedding matrix, as Dflash implem needs to
  embed without the norm"* (M:439) — so the drafter shares this matrix **unnormed**. Fold it
  and S6 breaks.
- **Two norm formulas coexist.** The four per-layer norms and nothing else are
  `MuseGlimmerTextCenteredRMSNorm`: `_norm(x) * (1.0 + w)` with `w` initialised to **zeros**
  (M:128-138). The final `model.norm` (M:465), the `qk_norm`, and the embedding norm are
  `MuseGlimmerRMSNorm`: `_norm(x) * w`, weight ones (M:117-121). Applying `* w` to a centered
  norm's zero-centered weight multiplies the residual stream by ≈0 — that one crashes into
  garbage, which makes it the *safe* member of this list. The reverse substitution does not.
- **Logits** (M:1253-1260):
  `logits = 20.0 * tanh(lm_head(h) * 0.19611613513818404 / 20.0)`.

**The logit path is argmax-invariant, and that is a gate blind spot worth naming.**
`output_multiplier` > 0 and `tanh` is strictly increasing, so the multiplier and the softcap
**cannot change which token greedy decoding picks**. A greedy-decode gate, a teacher-forced
argmax check and a byte-identical-output comparison are all blind to omitting them entirely.
They *do* change every probability: NLL/perplexity, any sampling, and any confidence gate
(`top1_prob`, the `--mtp-min-conf` analogue) are all wrong without them. **G3 must therefore
carry a probability-space check, not only greedy equality** — this is the §G "name the blind
spot" obligation for this model, and it is not layer 0.

`_tied_weights_keys` maps `lm_head.weight` to the embedding (M:1145), but this checkpoint sets
`tie_word_embeddings: false` and ships **both** tensors separately (2.690 GB each). Assert
untied from config; do not infer from the class.

## 6. RoPE convention — and the permutation that may make rivoli's kernel correct

Glimmer uses **`rotate_half`** (M:216-220): for a head vector `x[0..127]`, pairs are
`(x[i], x[i+64])` sharing frequency `inv_freq[i]`, since `emb = cat(freqs, freqs)` (M:209).
rivoli's `rope_interleave` (`kernels/linalg.hip:285`) uses the **interleaved** convention:
pairs `(x[2i], x[2i+1])` with frequency `i`. These are different permutations of the same
arithmetic, and applying one where the other is meant leaves the text fluent.

**Proposal, unproven — S1b must settle it before any kernel work depends on it.** Define
within each head the permutation `P: y[2i] = x[i], y[2i+1] = x[i+64]` for `i` in `0..63`.
Interleaved RoPE on `y` computes exactly split-half RoPE on `x`, because both pair the same
two elements against the same frequency. So **permuting the rows of `q_proj` and `k_proj`
within each head at conversion time should let rivoli's existing interleaved kernel run
unmodified.** The argument that it is safe to do at conversion time:

- `v_proj` is never rotated, so `o_proj`'s input basis is untouched — only `q_proj` and
  `k_proj` rows move.
- `gate_proj` acts elementwise on the attention output, not on q or k, so it is unaffected.
- `qk_norm` is an RMS over all 128 dims of the head; a permutation within the head does not
  change the mean of squares, so the norm **commutes** with `P` and may be applied either
  side of it.

If this holds, item 2 of S2 collapses from a new kernel to a converter permutation plus a
per-layer on/off flag. **It is an argument, not a measurement**: G1b owes it a numeric
fixture that reddens when `P` is replaced by identity.

## 7. Byte accounting

From the shard headers, summed by hand; the total reconciles with the index's own
`total_size` 59,553,253,376, which is the check that the sum is right.

| | tensors | bytes (bf16) |
|---|---|---|
| text side | 627 | **55.710 GB** (27.855 B params) |
| vision tower + adapter | 809 | 3.844 GB (1.922 B params) |
| total | 1436 | 59.553 GB — matches index metadata exactly |

Text side: **967.889 MB per layer** × 52 = 50.330 GB, plus embed 2.690 GB + lm_head 2.690 GB
+ 13312 B of final norm. Per layer: MLP 797.4 MB (three `[19968,6656]`-class matrices),
attention 170.4 MB (`q`/`gate`/`o` at 54.5 MB each, `k`/`v` at 3.4 MB), norms 53 KB.

**Decode traffic, weights read once per token** (dense — there is nothing to route):

| format | layers + lm_head |
|---|---|
| bf16 | 53.020 GB/token |
| fp8 | **26.510 GB/token** |
| int4 + group-128 fp16 scales | 13.648 GB/token |

**KV traffic is not negligible and it is asymmetric.** 512 elements/token/layer (2 heads × 128
× {k,v}). The 39 sliding layers are capped by the window: **81.79 MB total at bf16**, forever.
The 13 full layers are not: at 131072 context they read **1.745 GB/token at bf16**, 0.872 GB
at fp8 — i.e. at long context the full layers' KV costs more than a fifth of the fp8 weight
traffic, from 25% of the layers.

## 8. Tokenizer and chat template

`vocab_size` 202048. `bos` `<|begin_of_text|>` (id 200000); `generation_config` gives
**`eos_token_id: [200001, 200008]`** — two ids, and a port that reads a scalar EOS stops on
one of them. `pad` `<|finetune_right_pad|>` (200018).

`chat_template.jinja` (7167 bytes) is a Harmony-class framing and must be hand-ported
byte-for-byte, with a pinning test, per the `tokenizer.rs` precedent:

- Turns are `<|start|>{role}<|message|>{content}{<|eot|> or <|eom|>}`. `<|eom|>` closes a turn
  followed by another of the **same role**; `<|eot|>` closes otherwise.
- Assistant turns carry a **recipient**: `to=self` for `reasoning_content`, `to={tool.name}`
  for tool calls, `to=user` (or absent) otherwise. Tool calls render as an `ATEM`
  `<invoke>` XML-ish block; the macro **raises** if `arguments` is a JSON string rather
  than a mapping.
- When no system message is present the template **injects one**, carrying
  `Knowledge cutoff: 2026-01-04.`, a current date, `Reasoning strength: {high|…}.`, the tool
  definitions block, and a `# Valid recipients: …` line.
- The generation prompt is bare `<|start|>assistant` — **no `<|message|>`** — so the model
  emits its own recipient token first. A port that appends `<|message|>` removes the model's
  ability to choose `to=self`, silently disabling reasoning.
- `tokenizer_config.json` also carries a `response_template` describing how to parse the
  reply back (open patterns `to=user<|message|>` / `to=self<|message|>`, close `<|eot|>`/
  `<|eom|>`). That is the parser contract for `serve.rs`.

## 9. Traps — each one runs clean and produces a wrong model

Each is a G2/G3 defect-run candidate. None crashes; none is visible to `distinct` or longest
repeated block.

1. Reading top-level `rope_theta` for all layers instead of `layer_rope_theta[i]` as a flag —
   rotates the 13 NoPE layers.
2. Skipping `qk_norm` because no tensor ships for it.
3. Applying `qk_scale_factor` to K as well as Q, or instead of the `1/sqrt(128)` softmax scale.
4. Gating on the attention output instead of the layer input.
5. Substituting plain RMSNorm (`* w`) for the centered form (`* (1+w)`) on the four per-layer
   norms — or the reverse on the final norm.
6. Using one eps for all four per-layer norms (1e-5 vs 1e-8 by position).
7. Applying the post-norms to the residual stream instead of the branch.
8. Dropping the post-norms entirely (a two-norm pre-norm layer is the shape rivoli already has,
   so this is the *easy* mistake).
9. Interleaved vs split-half RoPE, or applying §6's permutation to `v_proj`/`o_proj` too.
10. KV head broadcast as `j % 2` instead of `j // 16`.
11. Omitting the embedding's weightless norm — or folding it into the matrix, which breaks the
    drafter later.
12. Omitting `output_multiplier` / softcap: **invisible to every greedy gate** (§5).
13. Treating EOS as a scalar when `generation_config` lists two ids.
14. Off-by-one on the sliding window. **SETTLED**: `masking_utils.sliding_window_overlay` is
    `kv_idx > q_idx - sliding_window` and-ed with causal `kv_idx <= q_idx`, so position `p`
    attends to **`[p-2047, p]` — exactly 2048 rows, inclusive of `p` itself**. The library's
    own docstring confirms it: at `sliding_window=3`, row 4 sees `{2,3,4}`. A 2048-row ring
    buffer is therefore exactly right, and the current token's own K/V must be in it.

    > **CORRECTED 2026-08-11 — "exactly right" is true PER QUERY ROW, and a ring is not sized
    > per row.** A launch covering `T` query rows dereferences the union of their windows,
    > `[p₀-2047, p₀+T-1]`, which is **2047 + T** distinct positions. At 2048 slots and `T = 2`
    > the oldest row the first query still wants has already been overwritten by the newest —
    > inside one launch, with every shape right and no error. Decode is `T = 1` and unaffected;
    > a **prefill chunk is not**, and layer-major prefill is this engine's default. Found by
    > two independent reviews of S2 item 1 the day the kernel landed, not by its gate: the
    > reference hands one query row per sliding step, so no golden can reach the case.
    > `rivoli_gqa_attend` now refuses `ring_cap < win + tq - 1`.
15. Assuming `head_dim == hidden / n_heads` (128 vs 208).

## 10. Where the model card was wrong

The plan's §1 was built from the announcement and HF model card. Recorded because the failure
mode is the point, not to score it:

| card / blog said | first-party says |
|---|---|
| "approximately 4-bit precision", "under 20 GB", K-Quant variants | the HF repo is **BF16, 59.553 GB**, two shards. The 4-bit artifacts are separate GGUF releases; rivoli converts from the BF16 |
| "gated attention: yes" (form unspecified) | confirmed, and specified: `sigmoid(gate_proj(layer_input))` before `o_proj` — but **`config.json` carries no flag for it**, so it is discoverable only from weights + code |
| context "131,072+" | exactly 131072, `rope_type: "default"` — **no scaling scheme at all** |
| ~29.6 B params "including vision encoder" | 29.777 B total, of which **27.855 B is the text side** |
| 52 layers, hidden 6656, 32/2 heads, head_dim 128, inter 19968, window 2048, vocab 202048 | **all confirmed** |
| — (absent from every card) | `qk_norm`, `qk_scale_factor`, `output_multiplier`, `final_logit_softcapping`, `post_norm_eps`, the sandwich norms, the centered-norm formula, the normed embedding. **Eight load-bearing facts, none of them in the marketing surface.** |

## 11. The DFlash drafter

Published as a **separate checkpoint**: `meta-models/Muse-Glimmer-30B-assistant`, single
`model.safetensors`, 5,111,976,608 B, **2.556 B params, 59 tensors, all BF16**. Implemented
first-party in transformers at `models/muse_glimmer_assistant/` (derived from Exaone4);
`models/dflash/` does not exist. Independently implemented in vLLM
(`v1/spec_decode/dflash.py`) and SGLang (`srt/models/dflash.py`), **both only for the Qwen3
flavour** — neither wires up Muse Glimmer, so there is no serving reference for this pairing.

```
block_size 16 · layers 5 · hidden 6656 · inter 19968 · head_dim 128
heads 32 Q / 8 KV  (4:1 — NOT the target's 16:1)
all 5 layers sliding_attention, window 2048 · rope_theta 500000 · rms_norm_eps 1e-5
mask_token_id 201818 · target_layer_ids [1, 13, 25, 37, 49]  (zero-based, of 52)
```

**It is not a small decode model. It is a 5-layer cross-attention adapter**, and almost every
property differs from the target:

| | target | drafter |
|---|---|---|
| KV groups | 32Q/2KV | **32Q/8KV** |
| per-layer norms | four, sandwich, centered | **two, plain pre-norm** — no post-FFN norm |
| QK-norm | weightless, ships no tensor | **has weights**, `q_norm`/`k_norm` `[128]` |
| causality | causal | **bidirectional** (`is_causal = False`) |
| embed / lm_head | owns both | **owns neither** — borrows the target's |

Tensors: `encoder.fc.weight [6656, 33280]` (33280 = 5 × 6656, one column block per
`target_layer_ids` entry), `encoder.output_norm_enc.weight [6656]`, `norm.weight [6656]`, and
per layer `q_proj [4096,6656]`, `k_proj`/`v_proj `[1024,6656]`, `o_proj [6656,4096]`,
`q_norm`/`k_norm [128]`, `input_layernorm`, `post_attention_layernorm`, and a SwiGLU MLP.

**The cycle**, one forward pass with no denoising loop:

1. The target's decode emits hidden states at layers 1/13/25/37/49 **for every accepted
   token**, not only the last. Concatenate along the feature axis → `[n, 33280]`.
2. `H_t = output_norm_enc(fc(concat))` → `[n, 6656]`. Computed **once**, shared by all 5 layers.
3. Draft input is `[last_accepted_token] + 15 × MASK(201818)`, embedded from the target's
   embedding matrix **raw — the weightless embed-norm of §5 is deliberately skipped**.
4. Each draft layer computes `Q` from the 16 draft rows only, but `K`/`V` from
   `concat(H_t, draft_rows)` — so the target context enters as extra **K/V entries** and
   bypasses `Q`, `o_proj` and the FFN entirely.
5. Attention is **bidirectional** across the 16-row block (window 2048 still applies).
6. Logits come from the **target's** `lm_head`; slice off index 0 → **15 candidates**.

**Q and K/V have different sequence lengths in the same call** (16 vs `ctx+16`), so RoPE is
applied with `cos/sin` built over the full range and **Q taking the tail slice**. Off by
`ctx_len` here is a silent quality loss, not a crash.

**Acceptance is stock assisted decoding and is lossless**: greedy takes the longest prefix
matching the target's argmax, plus one bonus token; sampling uses Leviathan rejection
sampling. DFlash contributes no verification rule of its own.

**The economics here are the inverse of GLM's, and this is the load-bearing point.** rivoli's
MTP experience — 2 rows costing 1.61× the experts, ungated `--mtp` landing at 0.93× — is
*MoE-union* economics and **does not transfer to a dense model**. A dense verify pass reads
every weight exactly once regardless of row count, so per cycle the cost is one target read
plus one drafter read, against `N` accepted tokens:

> speedup ≈ `N × 26.51 / (26.51 + 2.56)` = `N × 0.91` at fp8

**Break-even is `N > 1.1`** — the drafter pays for itself at barely more than one accepted
token per cycle, and τ=2 already returns 1.8×. That is why S6 is worth doing here when the
equivalent was a loss on GLM. Two cautions before any of this is registered as a prediction:
the drafter adds ~20 KiB/token of its own KV (5 × 8 × 128 × 2 × 2 B, ≈38% of the target's
53 KiB/token), and the target must now export 5 hidden states per accepted token —
**66,560 B/token** of extra traffic and a new output path through the decode loop that does
not exist today.

Published numbers, none of them on this hardware or this pairing: the drafter card reports
**3.1× on an RTX 5090** (74.9 → 233.4 tok/s, llama.cpp, against the *K-Quant-17GB* target with
a *quantized* drafter) and 1.5–1.8× on Apple M4/M5 Max via ExecuTorch. The paper's τ≈6.54 /
4.9× average is **H200, Qwen3-4B, dense**. The paper's §5.4 also shows acceptance decaying
with context beyond the drafter's ~4K training window unless long-context fine-tuned — so any
band registered from short prompts will not hold at 131k.

## 12. Two keys the schema does not bind, recorded so they are not rediscovered

- **`out_hidden_size: 6144` (wrapper level).** Not the text model's `hidden_size` 6656, and
  not mentioned anywhere else in this doc. It belongs to the vision path — `vision_adapter`
  /`vision_projection` project the tower's 1536 through `projector_hidden_size` 4096 — so it
  is out of scope here. Recorded because a reader who finds a second "hidden size" in the
  config and assumes it is the trunk's will build every projection 512 too narrow. Flagged by
  review 2026-08-11 as absent from this doc.
- **`text_config.eos_token_id` is the scalar `200001`**, while `generation_config.json` lists
  **two** ids, `[200001, 200008]`. That is trap 13 sitting live in the shipped files. The
  schema deliberately does **not** bind the `text_config` key: EOS comes from the tokenizer's
  `generation_config`, and binding the scalar here is precisely how a port comes to stop on
  one of the two.

## 13. Still open

- **Sustained resident-GEMV bandwidth at these shapes on gfx1151.** BLOCKED 2026-08-10, and
  the block is the protocol working, not a fault: `/var/run/sys-gpu.lock` was **held**
  (`flock -w 1` exited 1 — captured, not swallowed) and `gpu_busy_percent` sat at 100 across
  six one-second samples, with llama-swap holding 41.24 GB of GTT across
  `qwen3-embedding-4b`, `qwen3.6-medium` and `whisper` — and **zero
  `/sys/class/kfd/kfd/proc/` entries**, the Vulkan-tenant blind spot again, so a kfd-only
  check would have called this machine free. No number was taken in preference to one that
  would have to be discarded. This is the only unmet part of G0; `reference/gpu-lock.md`
  documents the shared-lock contract to re-run under.
- Whether §6's permutation actually holds. It is an argument; G1b owes it a fixture.
