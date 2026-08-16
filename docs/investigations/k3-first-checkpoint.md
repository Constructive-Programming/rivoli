---
status: live
scope: k3
verdict: The real Kimi-K3 checkpoint arrived 2026-08-16 and the schema survived it — every field K3TextConfig requires is present at the level it expects, num_nextn_predict_layers IS 0 (no MTP head, the assertion holds), the quantization block is MXFP4 e2m1 + e8m0 group_size 32 as assumed, and the vendored config.json is BYTE-IDENTICAL to the shipped one (md5 e0b7be1d…, index sha256 a1c52106… matching its pinned revision). Two things the synthetic gates could not see: the checkpoint ships NO tokenizer.json — it is tiktoken (163,584 ranks + 16 special ids in tokenizer_config.json) — so convert_k3's aux list refused the real source at the LAST step of a 1.42 TiB run, and Tokenizer::load reads tokenizer.json unconditionally for every arch, which blocks the first K3 decode on a tiktoken loader that does not exist rather than on the GPU. Conversion census: 497,220 tensors = 494,592 routed (1,446,456,066,048 B) + 2,460 resident (113,509,540,864 B) + 168 vision (894,717,952 B); one .f4 layer is exactly 4096 + 896 × 17,547,264 = 15,722,352,640 B, verified against the source at 0 differing bytes.
---

# Kimi-K3: the first real checkpoint

The K3 completion wave's first two steps — convert, then record — from the post-rewrite
feature-wave plan. (That plan is not in this tree, so its milestone numbers are not cited
here as if they were.) Everything below was read off `/swarm/storage/ai/rivoli/kimi-k3-src` on
2026-08-16, the day the download finished. **This arm had never touched a real checkpoint** —
every K3 gate to date built its own fixture — so the point of this document is the list of
things a fixture cannot tell you.

## STATE

- **The schema is right.** `K3TextConfig`'s ~40 required fields are all present, at the
  nesting level it expects. No `missing field`, no wrong level, no guessed key.
- **No MTP head.** `num_nextn_predict_layers` is `0`. The `validate_flags` assertion holds and
  the plan's "K3 spec decode stays Refuse" cell keeps its stated reason.
- **The vendored config is the shipped config, byte for byte**, and the index matches its
  pinned revision's sha256. The schema work done against the vendored copy was done against
  the real thing.
- **The converter could not finish**, for one reason, now fixed: it copied `tokenizer.json`,
  and Kimi-K3 has none.
- **The first K3 decode is not blocked on the GPU.** It is blocked on a tiktoken tokenizer
  loader. See §4.
- Conversion is running; §5 carries the census to assert it against.

## 1. Config, field by field

`config.json` is the `KimiK3ForConditionalGeneration` multimodal wrapper the port expected:
`text_config` carries the text model, `vision_config` is its sibling, and the WRAPPER is the
level that names the architecture (`kimi_k3`), which is the level `arch_of_named` reads.

Every value `K3TextConfig` binds, confronted with the file:

| field | shipped | the port assumed |
|---|---|---|
| `num_hidden_layers` | 93 | 93 ✓ |
| `hidden_size` | 7168 | 7168 ✓ |
| `routed_expert_hidden_size` | 3584 | the latent, ≠ hidden ✓ |
| `moe_intermediate_size` | 3072 | 3072 ✓ |
| `intermediate_size` | 33792 | dense layer 0 ✓ |
| `vocab_size` | 163840 | 163840 ✓ |
| `num_experts` | 896 | 896 ✓ |
| `num_experts_per_token` | 16 | 16 — and `top_k` is 50, a *different key* ✓ |
| `num_shared_experts` | 2 | a count of experts, not of tensors ✓ |
| `first_k_dense_replace` | 1 | layer 0 dense ✓ |
| `num_nextn_predict_layers` | **0** | **0 — asserted, and it holds** ✓ |
| `tie_word_embeddings` | false | false, and present in `text_config` ✓ |
| `mla_use_nope` / `mla_use_output_gate` | true / true | both true ✓ |
| `rope_theta` | **absent** | must be absent ✓ |
| `kv_lora_rank` / `q_lora_rank` | 512 / 1536 | 512 is exactly the kernel cap ✓ |
| `num_attention_heads` = `num_key_value_heads` | 96 = 96 | not MQA ✓ |
| `linear_attn_config.gate_lower_bound` | **-5.0** | negative, and it multiplies ✓ |
| `linear_attn_config.use_full_rank_gate` | true, one level down | the level was the old bug ✓ |
| `linear_attn_config.full_attn_layers` / `.kda_layers` | 24 / 69 | partition 1..=93, one-based ✓ |
| `hidden_act` | `situ` | ✓ |
| `activation_situ_linear_beta` | 25.0 | the long spelling ✓ |
| `moe_router_activation_func` / `topk_method` | `sigmoid` / `noaux_tc` | ✓ |
| `num_expert_group` / `topk_group` | 1 / 1 | degenerate, asserted ✓ |
| `rms_norm_eps` | 1e-05 | that key, not `rms_eps` ✓ |
| `dtype` | `bfloat16`, inside `text_config` | that level ✓ |

**Nothing to correct.** Every hazard `k3_config.rs`'s comments were written against — the
`top_k` collision, the `activation_situ_linear_beta` spelling, `use_full_rank_gate`'s level,
the latent-vs-hidden confusion — was a real hazard and the port is on the right side of each.

That is not luck. `docs/measurement/k3-reference/config.json` is **byte-identical** to the
shipped file (`md5 e0b7be1dc855c77fcaaae29940ec6d07`, both), so the port was reading this
config all along. The claim is worth recording precisely because it is the one an artifact
"checked only against a frozen copy of itself" would also make — here the frozen copy and the
live file were compared directly, and the live file's index carries
`sha256 a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`, matching the
revision `tensor-families.tsv` pins.

### Quantization config: as assumed

```
format             mxfp4-pack-quantized
num_bits           4          group_size    32
type               float      symmetric     true
strategy           group      scale_dtype   torch.uint8
```

e2m1 nibbles with e8m0 group scales at 32 — exactly what `.f4` holds, so the repack is a
copy. And `k3_config.rs`'s reason for *not* binding this block is confirmed by the file: its
`ignore` list names `self_attn`, `shared_experts`, `mlp.(gate|up|gate_up|down)_proj`,
`lm_head`, `vision_tower`, `mm_projector` — and **omits** `routed_expert_down_proj`,
`routed_expert_up_proj` and `block_sparse_moe.gate.weight`, all three of which ship BF16
under a `targets: ["Linear"]` that claims them. The block mis-declares its own scope; the
converter is right to drive off `.weight_packed` instead.

### Tensor names and shapes: as assumed

The 60 families reduce exactly as `tensor-families.tsv` records (48 text-side), and the
shapes `confront_moe_trunk` demands are the shapes on disk — `gate.weight [896, 7168]`,
`routed_expert_down_proj [3584, 7168]`, `routed_expert_up_proj [7168, 3584]`,
`routed_expert_norm [3584]`, the fused shared MLP at `[6144, 7168]`·2 + `[7168, 6144]`.
Experts are `w1`/`w3`/`w2` with `.weight_packed [o, i/2]` and `.weight_scale [o, i/32]`, both
`U8`:

```
w1  packed [3072, 1792]  scale [3072, 112]     (3584 -> 3072, the latent in)
w3  packed [3072, 1792]  scale [3072, 112]
w2  packed [3584, 1536]  scale [3584,  96]     (3072 -> 3584, back to the latent)
```

`F4_NAMING_K3` and `vq_expert_layout(3584, 3072)` describe these exactly.

## 2. The vision side is skipped — and the counter that "proves" it cannot

168 tensors, 894,717,952 B — `vision_tower.*` (27 encoder blocks) and `mm_projector.*`,
siblings of `language_model` in the name tree. `convert_k3::is_vision` skips them by prefix.
The port implements the text arm only.

> The k3 tree once estimated this at "~600 MB" with nothing to derive it from. The measured
> figure is 894,717,952 B — 0.83 GiB. Recorded here so the estimate stops being repeated.

**But `skipped_vision` is structurally zero on this checkpoint, and the first draft of this
section claimed the opposite.** It said the run "PRINTS the count, so the exclusion is an
observation rather than an assumption". Measured: the 168 vision tensors sit **alone** in
shards 95 and 96, which hold no other tensor, so `in_scope` rejects everything in them and
`Safetensors::open_indexed` opens **94 of 96 shards**. `write_resident` never sees a vision
name and logs `0 vision skipped` — whatever the filter does. The 168 / 894,717,952 figures
above are from the index, computed here; the run does not witness them.

That matters beyond the wording, because `is_vision` is the SAME function in `in_scope` and in
the resident loop. Break it and both ends fail together: shards 95–96 start being opened *and*
stop being filtered, 0.83 GiB of vision tower lands in `resident.safetensors`, `kept > 0`
passes, and the vision counter still reads 0.

So the guard added here does not consult `is_vision` at all. Every text-side tensor carries
`language_model.` (`K3_TEXT_PREFIX`, or `lm_head` directly beneath it — `k3_names.rs` pins
that across all 60 families) while the multimodal siblings do not, so `write_resident` refuses
any in-scope name lacking that prefix. Same defect, independent question.

> The gate that appeared to cover this — `k3_convert.rs`'s `assert!(log.contains("2 vision"))`
> — is green only because the fixture writes every tensor into ONE shard, so the vision names
> are in an opened file. That is this milestone's own defect (§3) reproduced inside the test
> that was supposed to catch it: **a fixture free to choose its shard layout proves nothing
> about a checkpoint that chose a different one.** The assert is kept — it still shows the
> counter works when the tensors are reachable — but it is no longer read as covering the real
> checkpoint, and the prefix guard is what does.

## 3. What the synthetic gates could not see: there is no `tokenizer.json`

**Kimi-K3 ships no `tokenizer.json`.** `tokenizer_config.json` declares
`tokenizer_class: "TikTokenTokenizer"`, and the vocabulary is `tiktoken.model` — 163,584
lines of `<base64> <rank>`, leaving 256 slots under `vocab_size` 163840 for special tokens, of
which 16 are declared in `added_tokens_decoder`:

```
163584 [BOS]           163587 <|open|>        163590 [start_header_id]   163602 <|media_begin|>
163585 [EOS]           163588 <|close|>       163591 [end_header_id]     163603 <|media_content|>
163586 <|end_of_msg|>  163589 <|sep|>         163593 [EOT]               163604 <|media_end|>
163605 <|media_pad|>   163649 <osagent_mode>  163838 [UNK]               163839 [PAD]
```

`convert_k3`'s aux list was `["tokenizer.json", "tokenizer_config.json"]`. `finish_artifact`
copies with `fs::copy` and `ensure!`s the result, and it runs **last** — so the converter
would have read 1.419 TiB and written 1.419 TiB (1.316 TiB of `.f4` plus 0.103 TiB of
resident set) over some hours, and then refused. It is now
`["tiktoken.model", "tokenizer_config.json", "generation_config.json"]`, each argued at the
call site.

`generation_config.json` is the third correction: the old comment said K3 ships none. It
does, it carries `eos_token_id: 163586`, and `Tokenizer::load` reads stop ids from that exact
filename — the file whose absence `finish_artifact`'s own comment calls the "ZERO stop
tokens, behind a 56-run retraction" defect.

**Why nothing caught it.** `crates/cli/tests/k3_convert.rs` wrote its own `tokenizer.json`
into the fixture. A fixture free to invent its inputs proves the tool agrees with the
fixture. The gate now writes the three files the checkpoint actually ships **and asserts
`tokenizer.json` is absent from the artifact** — the half a presence check cannot state,
since "did the names I listed arrive" is green under either list.

> `eos_token_id` disagrees across the checkpoint's own files: `tokenizer_config.json` names
> `[EOS]` (163585) as `eos_token`, while `config.json` and `generation_config.json` both say
> `163586` (`<|end_of_msg|>`). The two generation-side files agree with each other and with
> the XTML framing, so 163586 is the one to decode against; recorded because a stop token
> taken from the wrong file is silent.

## 4. The first K3 decode is blocked on the tokenizer, not on the GPU

`crates/cli/src/main.rs` calls `Tokenizer::load(&a.model)` unconditionally, for every
architecture, before the arch match — and `Tokenizer::load` opens `{model}/tokenizer.json`
through the `tokenizers` crate. K3 has no such file and `convert_k3` cannot conjure one, so
**a K3 artifact cannot be opened by the CLI at all today**, `--bench` included.

This contradicts the plan's sequencing, which put "first real decode, correctness-only"
directly after the conversion. The decode is not GPU-gated; it is gated on a tiktoken
vocabulary loader (`tiktoken.model` ranks + the 16 special ids from `tokenizer_config.json`)
that this tree does not have. That is a self-contained, deviceless, testable piece of work
and it is the next K3 milestone, ahead of any device time.

## 5. Conversion census — the numbers to assert against

Read from the 96 shard headers, not from prose. The sum of tensor byte-extents equals
`metadata.total_size` exactly, and every shard satisfies `8 + header + data == file length`,
so no shard is truncated:

| set | tensors | bytes | GiB |
|---|---:|---:|---:|
| routed (→ `.f4`) | 494,592 | 1,446,456,066,048 | 1347.12 |
| resident (→ `resident.safetensors`) | 2,460 | 113,509,540,864 | 105.71 |
| vision (skipped) | 168 | 894,717,952 | 0.83 |
| **total** | **497,220** | **1,560,860,324,864** | **1453.66** |

Derived expectations, each checkable on the finished artifact:

- One expert is `3 × (5,505,024 + 344,064) = 17,547,264 B`, which is `4284 × 4096` — already
  `VQ_ALIGN`-aligned, so the repack pads nothing and `.f4` carries the routed bytes exactly.
- One layer file is `4096 + 896 × 17,547,264 = 15,722,352,640 B`. **Observed on `L01.f4`:
  15,722,352,640 B.**
- 92 MoE layer files (layers 1..92; layer 0 is dense), totalling `1,446,456,442,880 B`.
- `resident.safetensors` = 113,509,540,864 B of payload plus its header.
- `write_resident`'s own guard wants at least `92 × 896 × 6 = 494,592` routed tensors
  skipped, which is exactly the routed count — the guard is tight, not slack.

### The verify is against the SOURCE — and once, against neither side

`--verify` re-reads each written `.f4` **file** and byte-compares every expert span against
the source safetensors. It is not a comparison of the artifact with a copy of itself.
`L01.f4`: `896 experts, 0 bytes differ`.

But writer and verifier both walk `F4Expert::spans`, so a wrong LAYOUT would be written and
re-read consistently and this check could not see it. That gap is closed once, by hand, on
`L02.f4`: expert-block offsets re-derived from `config.json` alone (gate‖up‖down, each
`packed ‖ scales`, padded to `VQ_ALIGN`) by a reader that does not link rivoli, then compared
against the shard bytes read straight out of the source safetensors.

```
independently derived: expert bytes=17547264 stride=17547264 pad=0   (matches f4_expert_stride)
magic: b'FP4\0'
experts 0, 1, 447, 895 — 6/6 spans each, 24/24 byte-identical to source
```

Zero padding is the part worth keeping: `17,547,264 = 4284 × 4096` exactly, so `.f4` carries
the routed bytes with nothing added, and the artifact's routed total is the source's routed
total rather than merely proportional to it.

### Throughput, disclaimed

Two figures, and they measure different things. The one-layer canary took **8 min 56 s**, but
that included the resident-set write and the manifest. In steady state the interval between
consecutive layer files — one 15.7 GiB write plus the previous layer's 15.7 GiB verify read,
against a 15.7 GiB source read — was **2 min 39 s** (L02→L03), i.e. ~300 MB/s of NFS traffic.

**Extrapolating that to ~4 h for the set is an extrapolation from one interval, taken while
this document's own verification pass was competing for the same mount.** It is written down
to size the job, not to be cited; the real total belongs here when the run ends. The artifact
is NFS-hosted by the owner's Q1 decision, correctness-first, with **decode perf explicitly
disclaimed** — none of these numbers say anything about tok/s.

## 6. The gates added here, and their red proofs

Three, each shown red before its green was believed (P7):

| gate | what it refuses | proven red by |
|---|---|---|
| `require_aux` (`format/meta.rs`), called before any shard is opened | an aux name the source does not have | removing each of the three files in turn; each refusal names the file. Anti-vacuity: with all three present the same invocation reaches a *later* refusal (`--from 0` → "dense prefix"), so the rows cannot be green against a converter that refuses everything |
| the `language_model.` prefix guard (`convert_k3::write_resident`) | an in-scope tensor outside the text arm | injecting `audio_tower.encoder.proj.weight` into the fixture shard — a model-level name `is_vision` does not match, so it tests the guard and not that function. Neutering the guard's condition to `true` makes the converter run to completion and the row FAIL; restored, green |
| the docs registry row for this file | verdict drift between the doc and `INDEX.md` | editing the INDEX row's copy of `num_nextn_predict_layers IS 0` to `IS 2`; `docs.rs` failed naming this file and both strings, and went green on restore |

The sidecar assertion in `k3_convert.rs` is stated as a **set equality** rather than three
presence checks, because a per-name `is_file()` loop is green under any list that is a
superset of the names the test spells — it could not see `tokenizer.json` coming back. Each
sidecar's bytes are also compared against the source's, since presence under a right name says
nothing about content.

## Open, in order

1. **Tiktoken tokenizer loader** — deviceless, blocks every K3 decode (§4).
2. **First K3 decode**, correctness-only, once (1) lands: ids finite, `--ctx` at or under the
   `ATTEND_MAX_KV` 8192 ceiling, small token count.
3. **Chat encoding** — `encoding_k3.py` is a legitimate first-party source, and a port needs
   an id-pinned golden against it rather than a hand-read. Not started; `--port` still
   refuses, on a message corrected the same day to stop claiming K3 ships no encoder.
