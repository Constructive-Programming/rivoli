---
status: live
scope: inkling
verdict: The Inkling 2 text-stack and loop math is EXTRACTED from first-party transformers sources at pinned SHA d8f10a82 with file:line citations, the checkpoint-to-HF name mapping and the NVFP4/MXFP8 dequant semantics are READ from the loader's own conversion ops rather than inferred, the Mini tensor census is DERIVED from the live shard header (1244 tensors, 85 rows, closure asserted in the TSV), and the traps table is written for the defect matrix -- MXFP8 scales are row-wise [1,32] despite the config's [32,32] declaration, w13 rows are gate/up INTERLEAVED, the even e2m1 nibble is LOW, L2Clip adds 1e-6 in the denominator, the router adds 1e-20, layer 0 IS a block boundary, and pass 2+ restart from the pre-loop states with the layer-6 boundary state SKIPPED.
---

# Inkling 2 (`inkling2` / `inkling2_loop`): the math, from primary sources

S0's math extraction for the Inkling 2 port (plan of record:
`docs/investigations/inkling2-port.md`). Every answer carries its source at the pinned
revision; the traps table (§8) is written so the anchor's defect matrix can be built
straight off it. Nothing here is a measurement of this box.

## 1. Sources and their pins

| tag | what | pin |
|---|---|---|
| **MOD** | `src/transformers/models/inkling2/modeling_inkling2.py`, 1882 lines | `huggingface/transformers` `main` @ `d8f10a82213a3a9db48385aba3a4b57ab535de50` |
| **LOOP** | `.../inkling2_loop/modular_inkling2_loop.py`, 720 lines (the modular source; `modeling_inkling2_loop.py` is generated from it and agrees) | same SHA |
| **CFG** | `.../inkling2/configuration_inkling2.py`, 200 lines | same SHA |
| **LCFG** | `.../inkling2_loop/configuration_inkling2_loop.py`, 205 lines | same SHA |
| **CONV** | `src/transformers/conversion_mapping.py`, 2875 lines — the checkpoint→HF weight mapping | same SHA |
| **LOAD** | `src/transformers/core_model_loading.py` — the conversion ops (dequant, interleave, query merge, unpad) | same SHA |
| **MINI** | `thinkingmachines/Inkling-2-Mini` `config.json` + quant/tokenizer pins | @ `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`, vendored under `docs/measurement/inkling2-reference/mini/` |
| **SMALL** | `thinkingmachines/Inkling-2-Small-MXFP8` same set | @ `14f2cb29a5d0f4c1fda8f645fa339e936c2147a2`, vendored under `.../small-mxfp8/` |
| **BLOG** | `huggingface.co/blog/inkling2mini` — states the original checkpoint is NOT directly transformers-compatible and points at the conversion path | fetched 2026-11-23 |

The blog's literal link (`convert_inkling2_weights_to_hf.py` at `refs/pull/51050/head`)
404s at every ref tried (2026-11-23): the script was never merged. What DID merge
(PR #51050, commit `046bdb8cc2ce`, 2026-11-21) is the on-load conversion machinery —
**CONV** + **LOAD** — so `from_pretrained` on the ORIGINAL checkpoint is the first-party
path, and it is the anchor's weight-preparation path (§9). No standalone script is needed
or cited.

## 2. Identity

- **Family**: Inkling 2, Thinking Machines Lab, released 2026-11. Two members in scope:
  **Mini** (`inkling2_loop`, 24 stored layers, 12 of them looped ×3 → 48 applications)
  and **Small** (`inkling2`, 48 layers, no loop). Owner scope 2026-11-23: both members,
  Mini first; text-only v1 (vision tower pinned but unspent, image input refused);
  MTP/DSpark speculator deferred behind `Flag::Mtp`'s refusal.
- **Small source is `Inkling-2-Small-MXFP8`**, not an NVFP4 Small: transformers' Small
  path keeps MXFP8 experts quantized (CONV:301–352, the `inkling2_text` expert transforms
  feed `gate_up_proj_scale_inv`), and its `hf_quant_config.json` declares MXFP8 for all
  48 layers, zero NVFP4. The first-party anchor requirement selects this checkpoint.
- **Tokenizer**: tiktoken BPE rank file `tiktoken/tokenizer.model`, 3,386,494 B,
  200,055 lines (base64 token + rank), sha256
  `f170372dd655728909228220d77cf6b8104a59f38a91262a7d2c4027906ec0d8` —
  **byte-identical between Mini and Small**, vendored ONCE at
  `docs/measurement/inkling2-reference/tiktoken.model`. Same family as K3
  (`crates/artifact/src/tiktoken.rs`). Specials occupy ids 199998–200057; EOS is
  **200006** `<|content_model_end_sampling|>`, pad **199999** `<|endoftext|>`, the image
  placeholder is 200054 `<|unused_200054|>`. `tokenizer_config.json`:
  `PreTrainedTokenizerFast`, `model_max_length 1048576`,
  `clean_up_tokenization_spaces false`, `fix_mistral_regex false`, 15
  `extra_special_tokens`. The members' 26 MB `tokenizer.json` files are derived
  fast-tokenizer serializations — not vendored, not read by the runtime.
- **Chat template**: `chat_template.jinja`, 6868 B, sha256
  `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`,
  **byte-identical between members**, vendored per member.
- **Vocab**: stored padded to **201024**; `unpadded_vocab_size` **200058**; `lm_head`
  keeps only the first 200058 rows (LOAD:856–884, `Inkling2UnpadVocab`). Embed and
  unembed are separate tensors, **never tied** (MOD:1466 comment; `_tied_weights_keys={}`).

## 3. Text-stack math (MOD unless noted)

Shapes below use Mini's numbers (H=2304, 24 Q heads / 8 KV heads, head_dim 128,
d_rel 16, rel_extent 1024, I=1152, E=128, top-8); Small doubles H to 4608, Q to 48,
I to 2304, E to 384, layers to 48.

### Norms and clips
- **RMSNorm** (101–118): fp32 variance mean, `rsqrt(var+eps)`, then `weight*hidden`,
  cast back to input dtype (T5-style: weight applied AFTER the rsqrt, in fp32).
  `rms_norm_eps 1e-6`.
- **UnweightedRMSNorm** (121–130): fp32 `rsqrt(square.mean+eps)`, no weight.
- **L2Clip** (133–144): fp32; `norm = vector_norm(dim=-1)`;
  `hidden * (threshold/(norm+1e-6)).clamp(max=1.0)` — the **+1e-6 is in the
  denominator**, a trap (§8 T4).
- **ScaledL2Clip** (147–159): learned `scale [heads,dim]` (initialized ones) applied
  first, then L2Clip.
- **LogScale** (162–177): `τ = 1 + α·log(max((pos+1)/n_floor, 1))`,
  `hidden.float() * τ[...,None,None]`; α=0.1, `n_floor 8192` for BOTH members
  (the class default is 128000; both shipped configs override to 8192 — §8 T9).
- **LogScaledL2Clip** (180–193): own `scale`, `l2_clip(hidden.float()*scale.float())`,
  then LogScale.

### Relative position bias (196–227)
`linear(hidden) → [heads, d_rel]` (`wr_du`, [384,2304] = 24×16), norm (LogScaledL2Clip
on full layers, ScaledL2Clip on sliding; threshold `rel_l2_clip 8.0`),
`rel_logits = (states @ proj[d_rel, rel_extent]).transpose(1,2)` (`rel_logits_proj.proj`
[16,1024]), gathered at `clamp(distance, 0, extent−1)`, `masked_fill` zero outside
`[0, extent)`. Distances come from `get_relative_distance` (1325–1344):
`q_positions[:,None] − kv_positions[None,:]`, offsets from the cache's
`get_mask_sizes`/`get_query_offset` of the first full (or first sliding) cache layer.
Sliding layers use `rel_extent = sliding_window` (1024); full layers `config.rel_extent`
(1024) — equal here, but the SOURCE of the extent differs and the defect matrix must
plant them apart.

### Short convolutions (286–412)
- **ShortConvolution** (286–350): depthwise causal k=4, no bias, output
  `(conv + residual)/√(k+1) = √5`. Sliding layers' k/v sconv weights are
  `.to(fp32)` at construction (464, 875, 878, 934, 937). Checkpoint
  `k_sconv.weight [1024,1,4]` (8 KV heads × 128 = 1024 channels).
- **DynamicShortConvolution** (353–412): `static_weight [k,hidden]` (zeros at init)
  plus `weight_proj(hidden)` per KV head (`[32,2304]` = 8 heads × 4 taps); tap index is
  backward distance (tap 0 = current token); output `(windowed sum + residual)/√(2k+1) = 3`.
  Full layers only. Checkpoint `k_sconv.static_w [4,1024]` + `k_sconv.weight_proj`.
- Post-attention and post-MLP convolutions (`attn_sconv`, `mlp_sconv`, `[2304,1,4]`)
  are ShortConvolution at conv_idx 2 and 3, fp32, applied AFTER the attention/MLP
  output and BEFORE the residual add (`number_of_conv_states 4`: 0=k, 1=v, 2=post_attn,
  3=post_mlp).

### Projections and attention
- **QueryProjection** (415–432): `wq_du` linear → heads → norm (LogScaledL2Clip if
  `log_scaling` else ScaledL2Clip, threshold `qk_l2_clip 22.627416997969522` = 16√2).
  The learned clip scale is `q_l2_clip.scale [24,128]`.
- **KeyValueProjection** (435–485): linear (`wk_dv`/`wv_dv` [1024,2304]) → sconv
  (Dynamic if full else Short.to(fp32)) → heads → norm: keys get **L2Clip**
  (`qk_l2_clip`), values get **UnweightedRMSNorm** (eps) — `use_value_norm_no_gamma
  true` is the checkpoint's spelling of that choice.
- **GatedOutputProjection** (488–500):
  `sigmoid(gate_proj(hidden).float()) · attn_out` → `wo_ud` linear. `gate_attn`
  [3072,2304] is the gate; 3072 = 24 heads × 128.
- **eager_attention_forward** (515–546): `scaling = head_dim^−0.5`;
  `QKᵀ·scaling + rel_bias + mask`; **attention sink**: `sink_proj(hidden)`
  (`attn_sink_proj` [24,2304]) produces ONE extra logit per head,
  `cat[logits, sinks].float()` softmax, then the sink column is dropped (it carries
  zero value). GQA `repeat_kv` (503).
- **SlidingAttention** (549–639): `is_sliding` from `layer_types`; q_proj
  `log_scaling=False`, k/v `dynamic=False`, `rel_extent=sliding_window`, sink present,
  rel_bias `log_scaling=False`. Window 1024.
- **FullAttention** (642–730): q_proj `log_scaling=True`, k/v `dynamic=True`,
  `rel_extent=config.rel_extent`, rel_bias `log_scaling=True`; forward takes SEPARATE
  `key_hidden_states`/`value_hidden_states` (the full layer's three distinct norms).
- **Layer types** (CFG `__post_init__`): `"hybrid"` (full) iff `(i+1)%6==0` else
  `"hybrid_sliding"` → full at 5,11,17,23 (Mini) / 5,11,…,47 (Small).

### MoE
- **Softcap** (733–739): `tanh(x/cap)·cap`.
- **SoftcappedSwiGLU** (742–753): fp32
  `gate_softcap(gate) · sigmoid(RAW gate) · up_softcap(up)` — the sigmoid sees the
  UNCAPPED gate; caps 4.0 (gate) / 25.0 (up).
  **S4 reuse (MEASURED 2026-11-24):** this is K3's SiTU-GLU — the existing `situ_glu_f32`
  kernel (`b1·tanh(g/b1)·sigmoid(g)·b2·tanh(u/b2)`) at b1=4, b2=25, BIT-IDENTICAL under the
  division-form sigmoid (`crates/oracles/src/inkling.rs::softcap_swiglu_is_situ_glu`, diff 0.0).
  The only gap from the torch reference is `torch.sigmoid`'s 1-ulp (1.19e-7) implementation
  choice, amplified by the gate cap to ~1.9e-6 at the output. **No new activation kernel.**
- **Experts** (757–793): `gate_up_proj [E,2I,H]`, `down_proj [E,H,I]`; per-expert
  `F.linear` then `_apply_gate` = `chunk(2, dim=-1)` on the 2I output (gate = first I,
  up = last I); accumulate top-k outputs in **fp32** via `index_add_`, cast back at the
  end. `n_shared_experts 0` — no shared expert.
- **TopKRouter** (796–816): everything fp32. `weight [E,H]` is an fp32 parameter
  (checkpoint stores BF16, upcast on load — `_keep_in_fp32_modules_strict`);
  `e_score_correction_bias` is an **nn.Buffer** (checkpoint `mlp.gate.bias`, F32 [128]);
  `scores = sigmoid(logits)`; `indices = topk(scores + bias, k, sorted=False)`;
  `weights = scores.gather(indices) / (sum + 1e-20) · routed_scaling_factor` (8.0).
  The bias shifts SELECTION only, never the weights — a trap (§8 T6).

### Block attention residual (833–858)
`query` zeros fp32 `[H]` (1 query) or `[nq,H]` (3 on full layers);
UnweightedRMSNorm over `cat[block_states, partial.unsqueeze(-2)].float()`; attend with
`scaling = H^−0.5`, **no mask, `is_causal=False`, `num_key_value_groups=1`**; returns
unbind. `attn_res_normalize_values true` is the checkpoint spelling of the value norm
here. This is a SECOND attention per layer over the block-state bank, not over tokens.

### Decoder layers and the block bank
- **SlidingDecoderLayer** (861–915): `is_block_boundary = layer_idx % 6 == 0` —
  **includes layer 0** (§8 T7). `attention_input` = BlockAttentionResidual(1 query) if
  `layer_idx>0` else None; `mlp_input`(1 query); `attn_norm`/`mlp_norm` RMSNorm.
  Forward: `block_residual = hidden`; attention_input → hidden;
  self_attn(input_layernorm(hidden)); post_attn_conv; `residual = hidden.float()`;
  **at a boundary** append `block_residual` to `block_states` and `residual = attn_out
  ONLY`; else `residual = block_residual.float() + attn_out`; `mlp_input(block_states,
  hidden)` where hidden is the POST-ATTENTION residual; mlp(post_attention_layernorm);
  post_mlp_conv; return `residual + mlp_out`.
- **FullDecoderLayer** (918–980): `attention_input` = **3 queries** if `layer_idx>0`
  else None; separate `q/k/v_input_norm`; if None, q=k=v=hidden, else three distinct
  query outputs; rest identical.
- **TextModel** (1348–1442): `block_states` starts zeros `[...,0,H]`; per layer, at a
  boundary append the layer INPUT hidden (1433–1434); after the last layer
  `final_input(block_states, hidden)[0]` → `norm` → cast to input dtype.
  `attn_res.attn_queries.i` feeds layer **i+1**'s attention_input (CONV offset=1,
  §6); `mlp_queries.i` feeds layer i; `output_query` is `final_input.query`.
- **NormedEmbedding** (1316–1322): embed + `embed_norm` RMSNorm.
- **ForCausalLM** (1446–1529): `lm_head Linear(H, 200058, no bias)`,
  `logits = lm_head(hidden).float()` → **final_logit_softcapping 30.0**.

## 4. Loop math (LOOP)

Mini only. `loop_config {core_start 6, core_end 18, num_passes 3, gate_rank 64,
is_varloop false}`; LCFG:139–171 reads these into `loop_layer_index_start/end`,
`loop_num_passes`, `loop_gate_rank` (`is_varloop` is never read — pin it, don't knob it).
`validate_architecture`: passes≥1, start>0, start%6==0, end%6==0.
`total_num_hidden_layers = 24 + (3−1)·12 = 48` (cache slots, not weights).

- **MessageGate** (172–190): `norm` = UnweightedRMSNorm; `down_proj Linear(2H→64, no
  bias)` (bf16); `forget_gate_proj`/`message_gate_proj Linear(64→H, no bias, fp32)`.
  Forward: cast both inputs fp32; `cat[norm(hidden), norm(message)]` → cast to
  down_proj dtype (bf16) → down_proj → fp32; `forget = tanh(forget_proj(·))`,
  `gain = tanh(message_proj(·))`; return `(1−forget)*hidden + gain*message` in fp32.
  Checkpoint: `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight` (§6).
- **MessageAttentionResidual** (192–217): `query` zeros fp32 **[12,4,H]**; forward
  `cat[block_states, partial.unsqueeze(-2)].float()` → UnweightedRMSNorm → attend with
  `query[loop_layer_idx]` expanded, `scaling = H^−0.5`, no mask → `transpose(1,2)` →
  4 messages in order **(q, k, v, mlp)**.
- **SlidingLoopDecoderLayer** (231–301): 4 message gates;
  `attention_input(block_states, hidden)[0]`; if messages None, q=k=v=
  `input_layernorm(hidden)`; else each of q/k/v = `input_layernorm(gate(hidden, msg))`
  — **ONE shared norm** after gating. Attention → post_attn_conv → residual/boundary
  logic identical to base; `mlp_input(...)[0]` → `mlp_message_gate` →
  post_attention_layernorm → mlp → post_mlp_conv.
- **FullLoopDecoderLayer** (307–375): gates applied to EACH of the 3 attention_input
  outputs, then the three separate norms; rest identical.
- **LoopTextModel.forward** (445–585):
  1. Cache extended ONCE when `len(cache.layers)==24`: `deepcopy` of
     `layers[6:18] * (passes−1)` appended (494) → **48 slots**.
  2. Pre-loop `layers[:6]`; save `pre_loop_hidden_states, pre_loop_block_states`;
     `num_pre_loop_blocks = 1` (516) — only layer 0 is a boundary in 0..5.
  3. Pass 1 over `layers[6:18]`, messages=None. Boundaries 6 and 12 append → bank has
     3 states.
  4. For `loop_idx in 1..passes−1`:
     `previous_block_states = block_states[..., num_pre_loop_blocks+1:, :]` (537) —
     **skips the layer-6 boundary state; only the layer-12 state survives** (§8 T8);
     `previous_hidden_states = hidden`; restart hidden/block_states from the saved
     pre-loop pair; `cache_offset = 24 + (loop_idx−1)·12 − 6` (542) → pass 2 offset 18,
     pass 3 offset 30. Per layer:
     `messages = message_input(previous_block_states, previous_hidden_states,
     loop_layer_idx)` — previous states are FIXED within a pass (no intra-pass
     recurrence).
  5. Post-loop `layers[18:]`; `final_input` → norm.
- `_keep_in_fp32_modules_strict` (389–401): `e_score_correction_bias`, `mlp.gate.weight`,
  `post_attention_convolution`, `post_mlp_convolution`, `sconv.conv1d`,
  `attention_input`, `mlp_input`, `final_input`, `message_input`, `message_gate_proj`,
  `forget_gate_proj`.
- Wrappers (586–720): `Inkling2LoopForCausalLM`, `Inkling2LoopModel` (adds
  `vision_tower`), `Inkling2LoopForConditionalGeneration` — all delegate to the base
  forwards. `Inkling2LoopTopKRouter` (377) is a bare `pass` subclass.

## 5. Quantization ABI (LOAD:772–820, `Inkling2LoopDequantizeExperts`)

The Mini checkpoint mixes two expert formats; the dequant below is the FIRST-PARTY
oracle, transcribed from the loader, and the converter's repack must invert it exactly.

### NVFP4 (Mini layers 0–19)
- `w13_weight` U8 `[128,2304,1152]`: TWO e2m1 codes per byte, **even element in the LOW
  nibble** (`byte & 0xF` first, `byte >> 4` second); logical `[E,2I,H]`.
  `w2_weight` U8 `[128,2304,576]`, logical `[E,H,I]`.
- E2M1 LUT (LOAD:781): `(0, .5, 1, 1.5, 2, 3, 4, 6, −0, −.5, −1, −1.5, −2, −3, −4, −6)`.
- `*_weight_scale` F8_E4M3 `[E,rows,cols/16]`: one e4m3 scale per **16 values along the
  INPUT (last) dim**.
- `*_weight_amax` F32 `[E]`, `*_weight_outer_scale_denominator` F32 `[1]`:
  `value = code_value × block_scale × (amax[e] / denominator)`.
- This is NOT rivoli's `.f4` (MX-FP4: e2m1 + e8m0, block 32, `quant/f4.rs`). New scale
  ABI; repack, never requantize. `hf_quant_config.json` records the quantizer side
  (`fouroversix_mae` scale selection, per-token activation amax, block 16) — provenance
  for the format, not runtime behavior.

### MXFP8 (Mini layers 20–23; ALL Small layers)
- `w13_weight` F8_E4M3 `[128,2304,2304]` (one byte per element), `w2_weight`
  `[128,2304,1152]`.
- `*_weight_scale` U8 `[E,rows,cols/32]`: `value = fp8_e4m3 × 2^(scale−127)`, one
  power-of-two scale per **32 values along the INPUT dim** — **ROW-WISE [1,32]**.
- **The declared `weight_block_shape [32,32]` in `hf_quant_config.json` is WRONG for
  the stored layout**: a true [32,32] layout would store `[128,72,72]` scales (663,552
  elements); the file holds `[128,2304,72]` (21,233,664) — 32× more. The shape is
  authoritative (§8 T1). Small's root `quantization_config` independently declares
  `weight_block_size [1,32]`, agreeing with the shapes.

### Fusion and layout ops (CONV + LOAD)
- `w13` rows are **gate/up INTERLEAVED** in the checkpoint: `Interleave(dim=1)`
  (LOAD:232–256) DE-interleaves (`reshape [E,I,2,H] → transpose → [E,2,I,H] →
  reshape [E,2I,H]`) into `[gate_all; up_all]`, which `chunk(2)` then splits correctly.
  The scale tensor gets the SAME interleave (CONV:365–370). `w2` is a plain rename.
- `unembed.weight` → `lm_head.weight` keeps the first **200058** rows
  (LOAD:856–884).
- Query banks: `middle_cycle_cross_pass_queries` F32 [24,H] holds per-loop-layer
  **[q, mlp]** pairs; `middle_cycle_cross_pass_kv_queries` F32 [24,H] holds **[k, v]**;
  `Inkling2LoopMergeMessageQueries` (LOAD:823–853) unflattens each to [12,2,H] and
  concatenates `[q, kv, mlp]` → **[12,4,H]** in order (q,k,v,mlp).
- `attn_res.attn_queries.N` → `layers.(N+1).attention_input.query`
  (`Inkling2IndexOffsetWeightRename`, LOAD:1266, offset=1; CONV:408–412).

## 6. Checkpoint → transformers name mapping (CONV:301–418, 2654–2707)

The checkpoint names are TML's; transformers renames on load. The table is the
converter's Rosetta stone (source pattern → target; regex semantics as in CONV).

| checkpoint | transformers |
|---|---|
| `model.llm.unembed.weight` | `lm_head.weight` (unpad to 200058) |
| `model.llm.` | `model.` (text) / `model.language_model.` (multimodal wrapper) |
| `embed.weight` | `embed_tokens.weight` |
| `embed_norm.` | `embed_tokens.embed_norm.` |
| `mlp.experts.w13_weight` (+scale/amax/denominator) | `mlp.experts.gate_up_proj` (dequant + deinterleave) |
| `mlp.experts.w2_weight` (+…) | `mlp.experts.down_proj` (dequant) |
| `mlp.gate.bias` | `mlp.gate.e_score_correction_bias` |
| `.attn.` | `.self_attn.` |
| `wq_du` / `wk_dv` / `wv_dv` | `q_proj.linear` / `k_proj.linear` / `v_proj.linear` |
| `k_sconv.` / `v_sconv.` | `k_proj.sconv.` / `v_proj.sconv.` |
| `wr_du` | `relative_position_bias.linear` |
| `rel_logits_proj.` | `relative_position_bias.` |
| `q_l2_clip.` | `q_proj.norm.` |
| `rel_l2_clip.` | `relative_position_bias.norm.` |
| `wo_ud` / `gate_attn` | `o_proj.linear` / `o_proj.gate_proj` |
| `attn_sink_proj` | `sink_proj` |
| `attn_sconv.` / `mlp_sconv.` | `post_attention_convolution.` / `post_mlp_convolution.` |
| `(convolution\|sconv).weight` (not `_proj`) | `…conv1d.weight` |
| `…static_w` / `…weight_proj` | `…static_weight` / `…weight_proj.weight` |
| `mlp_norm` / `attn_norm` | `post_attention_layernorm` / `input_layernorm` |
| `(q\|k\|v)_input_norm` | `…_input_layernorm` |
| `attn_res.attn_queries.N` | `layers.(N+1).attention_input.query` |
| `attn_res.mlp_queries.N` | `layers.N.mlp_input.query` |
| `attn_res.output_query` | `final_input.query` |
| `controllers.attn.` / `.k.` / `.v.` / `.mlp.` | `q_message_gate.` / `k_…` / `v_…` / `mlp_message_gate.` |
| `_message_gate.down_weight` | `…down_proj.weight` |
| `_message_gate.gain_weight` | `…message_gate_proj.weight` |
| `_message_gate.forget_weight` | `…forget_gate_proj.weight` |
| `middle_cycle_cross_pass{,_kv}_queries` | `message_input.query` (merge, §5) |
| `model.visual.` | `model.vision_tower.` (EXCLUDED from text v1; full vision rename table at CONV:311–351) |

Small (`inkling2`) keeps MXFP8 quantized: `w13_weight_scale` →
`gate_up_proj_scale_inv`, `w2_weight_scale` → `down_proj_scale_inv` (CONV:365–375),
handled by the finegrained-fp8 quantizer the same PR touched. Mini (`inkling2_loop`)
REPLACES those transforms with dequant-on-load (CONV:2654–2676, "Temporary" comment).

## 7. Config facts (CFG, LCFG, MINI, SMALL)

**attribute_map** (CFG:74–81, LCFG:85–92) — the JSON spellings differ from the class
attributes; the Rust config type reads the JSON spellings:
`sliding_window→sliding_window_size`, `num_local_experts→n_routed_experts`,
`sconv_kernel_size→conv_kernel_size`, `model_max_length→max_position_embeddings`,
`intermediate_size→moe_intermediate_size`, `routed_scaling_factor→route_scale`.

**Mini text**: hidden 2304, 24 layers, vocab 201024 / unpadded 200058, 24Q/8KV
head_dim 128, d_rel 16, rel_extent 1024, `log_scaling_n_floor 8192`, alpha 0.1,
rms_norm_eps 1e-6, qk_l2_clip 22.627416997969522, rel_l2_clip 8.0, sconv_kernel_size 4,
sliding_window_size 1024, n_layers_per_attn_res_block 6, 128 experts top-8,
n_shared_experts 0, intermediate_size 1152, softcaps 4.0/25.0/30.0, route_scale 8.0,
scoring_func sigmoid, use_gate_bias true, swa 24Q/8KV/128,
`local_layer_ids [0,1,2,3,4,6,…,22]` (all but 5,11,17,23).
Root: `dtype bfloat16`, `architectures [Inkling2LoopForConditionalGeneration]`,
`model_type inkling2_loop`, eos 200006, `quantization_config {quant_method modelopt,
quant_algo NVFP4}` (the generic header; `hf_quant_config.json`, `quant_method inkling2`,
is authoritative — its per-tensor groups are keyed by **`tensors`**, not `module_names`).
Vision: `inkling2_vision`, hidden 1024, 23 layers, 16 heads, patch 20, merge 2,
projection 4096, posemb grid 32, mlp_ratio 4.0, layer_norm_eps 1e-5.
MTP: hidden 2304, 3 layers, intermediate 9216, rope_theta 1e6, block_size 8,
markov_rank 256, rnn_hidden, target_layer_ids [5,23,41,48] — DEFERRED, refused.
`generation_config`: `{do_sample true, temperature 1.0, eos 200006, pad 199999}`.

**Small text**: hidden 4608, 48 layers, 48Q/8KV head_dim 128, 384 experts top-8,
intermediate_size 2304, `log_scaling_n_floor 8192` (CORRECTION 2026-11-23: an earlier
note said 128000, the class default; the shipped config explicitly sets 8192, same as
Mini), extra fields `pre_norm_pre_affine false`, `final_norm_pre_affine false`;
vision block has no `model_type`/`text_hidden_size` keys (defaults apply) and an extra
`temporal_patch_size 2`; MTP 5 layers, intermediate 18432, target_layer_ids
[5,11,17,23,29,35,41,48]. Root `quantization_config {quant_method fp8,
weight_block_size [1,32], scale_fmt ue8m0, modules_to_not_convert [unembed, visual,
attn.*]}`. `generation_config`: `{eos 200006, pad 199999}` (no sampling keys).

**Checkpoint fields with NO transformers class counterpart** — assert them as pinned
behavior in the Rust `validate`, never invent knobs: `q_bias`, `o_bias`,
`use_value_norm_no_gamma`, `attn_res_normalize_values`, `n_shared_experts`,
`local_layer_ids`, `use_gate_bias`, `pre_norm_pre_affine`, `final_norm_pre_affine`,
`loop_config.is_varloop`.

## 8. Traps (the defect matrix's seed list)

| # | trap | where | plant that reddens |
|---|---|---|---|
| T1 | MXFP8 scales are row-wise [1,32]; the quant config DECLARES [32,32] | §5, header shapes | read scales as [32,32] blocks → 32× wrong scale count, expert outputs scale-wrong |
| T2 | w13 rows are gate/up INTERLEAVED; even e2m1 nibble is LOW | LOAD:232, 781–800 | treat w13 as [gate;up] concat, or high nibble first → gate/up swapped, sign-class errors |
| T3 | NVFP4 scale is e4m3 × amax/denominator, NOT e8m0 block-32 | §5 | reuse `.f4` MX-FP4 path → wrong scale ABI entirely |
| T4 | L2Clip denominator is `norm + 1e-6` | MOD:133–144 | drop the epsilon → diverges at near-zero norms |
| T5 | router weights divide by `sum + 1e-20`; bias shifts selection ONLY | MOD:796–816 | add bias to weights, or drop epsilon |
| T6 | softcap SwiGLU: sigmoid sees the RAW gate, tanh caps are separate | MOD:742–753 | sigmoid the capped gate |
| T7 | layer 0 IS a block boundary; boundary residual = attn_out ONLY | MOD:861–915 | start boundaries at 6, or add block_residual at boundaries |
| T8 | pass 2+ skip the layer-6 boundary state (`num_pre_loop_blocks+1`) | LOOP:537 | pass the whole bank → messages attend an extra state |
| T9 | n_floor is 8192 in BOTH shipped configs (class default 128000) | MINI/SMALL config | use the class default → LogScale wrong past 8192 |
| T10 | sink logit joins the softmax then is DROPPED (zero value) | MOD:515–546 | add sink as a value, or softmax without it |
| T11 | full layers: 3 distinct norms/queries; sliding: ONE shared norm after gating | MOD:918–980, LOOP:231–301 | share the full layer's norm |
| T12 | attn_queries index i feeds layer i+1; mlp_queries index i feeds layer i | CONV:408–412 | off-by-one on the query bank |
| T13 | message order is (q,k,v,mlp); the two banks store [q,mlp] and [k,v] | LOAD:823–853 | interleave as (q,mlp,k,v) |
| T14 | lm_head keeps 200058 of 201024 rows; embed keeps all 201024 | LOAD:856–884 | unpad the embed too, or keep padded logits |
| T15 | sconv residual divides by √5 (static) / 3 (dynamic), fp32 weights | MOD:286–412 | drop the residual or the divisor |
| T16 | cache_offset pass 2 = 18, pass 3 = 30; conv states and KV are per-pass copies | LOOP:494, 542 | reuse pass-1 cache → cross-pass contamination |
| T17 | rel_extent source differs (sliding_window vs config) though both are 1024 | MOD:549–730 | plant unequal extents to catch a shared read |
| T18 | value norm is UNWEIGHTED RMSNorm; key norm is L2Clip | MOD:435–485 | swap them |

## 9. Anchor consequence

Because transformers converts the ORIGINAL checkpoint on load (§1, CONV+LOAD), the
first-party anchor runs `Inkling2LoopForConditionalGeneration.from_pretrained` on the
local Mini source with NO hand conversion in the golden path — the dequant, interleave,
query merge and unpad are the reference's own code. The rivoli converter (S2) is then
scored TWICE: by the census (exact counts, named exclusions) and by byte-comparing its
dequanted experts against the loader's `Inkling2LoopDequantizeExperts` output on real
tensors. Goldens are generated once (CPU fp32) and vendored deviceless, per CLAUDE.md.