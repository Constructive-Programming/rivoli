---
status: live
scope: k3
verdict: The real Kimi-K3 checkpoint arrived 2026-08-16 and the schema survived it — every field K3TextConfig requires is present at the level it expects, num_nextn_predict_layers IS 0 (no MTP head, the assertion holds), the quantization block is MXFP4 e2m1 + e8m0 group_size 32 as assumed, and the vendored config.json is BYTE-IDENTICAL to the shipped one (md5 e0b7be1d…, index sha256 a1c52106… matching its pinned revision). Two things the synthetic gates could not see: the checkpoint ships NO tokenizer.json — it is tiktoken (163,584 ranks + 16 special ids in tokenizer_config.json) — so convert_k3's aux list refused the real source at the LAST step of a 1.42 TiB run, and Tokenizer::load read tokenizer.json unconditionally for every arch, so no K3 artifact could be opened at all -- FIXED 2026-08-17 by a first-party-gated tiktoken loader (95 id-pinned cases; 16 gates, all deviceless -- 8 assert in CI and 8 skip without the checkpoint; 17 red proofs, five of which corrected the plan; both encode caps ported and fuzzed against the first-party encoder at 8,372 texts / 3,523,794 ids / zero mismatches), which measured two corrections to its own plan: round-trip stays GREEN under a broken pre-tokenizer, and the pat_str Han intersections are invisible to id equality because no token in the vocabulary mixes Han with non-Han. Conversion census: 497,220 tensors = 494,592 routed (1,446,456,066,048 B) + 2,460 resident (113,509,540,864 B) + 168 vision (894,717,952 B); one .f4 layer is exactly 4096 + 896 × 17,547,264 = 15,722,352,640 B, verified against the source at 0 differing bytes.
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
- **The first K3 decode was not blocked on the GPU** — it was blocked on a tiktoken tokenizer
  loader, because `Tokenizer::load` reads `tokenizer.json` unconditionally and K3 ships none.
  §4 is the diagnosis. **CLOSED 2026-08-17** by the loader in §7; the sentence is kept in the
  past tense rather than deleted, because "the artifact existed before anything could read it"
  is the shape of the finding and the next arm can repeat it.
- **The tiktoken loader SHIPPED** — 16 deviceless gates, 8 of which assert in CI, 17 red proofs
  (§7). Both encode caps are ported and differentially fuzzed against the first-party encoder at
  **8,372 texts / 3,523,794 ids / zero mismatches**. Five proofs refused to redden where
  predicted and two of my own harnesses produced false greens; every correction is recorded.
- **A concurrent GPU pin starves this job, and starvation reads as death.** GTT is system RAM
  here, so a ~115 GiB pin and a large streaming CPU job cannot share the machine — and the
  flock does not express it, because this job never touches the device. Two observers called
  a live process dead off an empty `ps | grep`. §5.
- Conversion FINISHED: 92 layers, every one verified at 0 differing bytes, artifact 1.4188 TiB.
  §5 carries the census it was asserted against.

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

### A concurrent GPU pin starves this job — and starvation reads exactly like death

Mid-run the converter stopped emitting log lines for roughly 40 minutes. Two observers
independently concluded it had been killed, and **both were wrong**: it was alive the whole
time, at its original PID, and resumed on its own.

What actually happened is the scheduling constraint worth keeping. **On this box GTT is
system RAM**, so a decode's pin is not device-side — a ~115 GiB pin took 123 of 124 GB, drove
`buff/cache` from 98 GB to 1 GB, and pushed the machine into swap. This converter's own
anonymous footprint is bounded and small (the 1 GiB `LAYER_WINDOW`), but it moves ~47 GiB per
layer through the page cache and mmaps 94 shards. With no cache left it made progress far too
slowly to log. A large pin and a large streaming CPU job **cannot run concurrently on this
machine**, and the GPU flock does not express that: the flock guards the *device*, and this
job never takes it because it never touches the device.

> **The diagnostic error is the more useful half.** Both observers ran a *pattern* probe
> (`ps … | grep convert_k3`) which returned nothing while that exact process was running, and
> read the empty result as death. One then fitted an OOM-kill narrative to it — a 115 GiB pin
> starting within seconds of the last log line — and recorded it as fact. Nothing was killed;
> there was no OOM. Had a restart been issued, a second converter would have started writing
> into the same output directory while the first was mid-layer.
>
> The rule this yields: **an empty pattern probe is not evidence of absence.** Liveness is
> `/proc/<pid>/stat` for the PID you launched and recorded — it carries `ppid`, `pgrp`,
> `session` and state in one read and cannot match the wrong thing. Detachment likewise is
> read there (`ppid=1`, `pgrp == session == pid`, `tty=0`), not inferred from having typed
> `setsid`. The launch verification here was correct and was doubted anyway, because the later
> probe was believed over the earlier evidence.

**Owed, and deliberately not applied to the running job:** the converter should tee a
timestamp and `free -g` into its log once per layer. Both misreadings came from having only
"no new log lines" to go on; a per-layer memory line separates *starved* from *dead* without
any process probe at all. It is additive and belongs to whoever next touches this tool.

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

## 7. The tiktoken loader — SHIPPED 2026-08-17

> **This section was written as a SCOPE and is kept as one**, with the outcome folded in
> beneath: the plan below is what was built, and the two places reality contradicted it are
> called out where they occur rather than quietly edited away. The conversion finished (92
> layers, §5), the loader landed, and **a K3 artifact now opens** —
> `crates/artifact/src/tiktoken.rs`, gated by `crates/artifact/tests/k3_tokenizer.rs`,
> **deviceless, so unlike everything else on this arm it runs in CI**.

**Separate the two questions first, because conflating them is the trap.** K3 ships *two*
first-party Python files and they are not the same milestone:

| file | what it defines | milestone |
|---|---|---|
| `tokenization_kimi.py` + `tiktoken.model` | **the tokenizer** — text ↔ ids | this one, and it is small |
| `encoding_k3.py` | **the chat framing** — messages → XTML → text | later, still refused |

So `encoding_k3.py` is *not* the id-pinned source for the loader. It renders a string; the
tokenizer turns strings into ids. A loader gated against `encoding_k3.py` would be gated
against the wrong reference.

### What the loader needs, and it is fully specified

`TikTokenTokenizer` is ~60 lines of construction, all of it data this artifact now carries:

- `mergeable_ranks` = the 163,584 `<base64> <rank>` lines of `tiktoken.model`.
- **special tokens are positional, not listed**: ids `163584..163840` — `num_base_tokens ..
  num_base_tokens + 256` — named from `added_tokens_decoder` where an entry exists and
  `<|reserved_token_{i}|>` where it does not. The 16 named ones are in
  `tokenizer_config.json`. This is why both files had to be copied.
- `n_vocab` = 163,840, which is `vocab_size` — a free cross-check against the config.
- a `pat_str` of 8 alternatives.

### The two traps, both in `pat_str`

1. **`\s+(?!\S)` is a negative lookahead.** Rust's `regex` crate does not support lookaround
   at all, so the obvious dependency is the wrong one; `fancy-regex` is the shape that can
   express this. Silently dropping the alternative changes trailing-whitespace tokenization —
   different ids, no error.
2. **`[\p{Lu}…&&[^\p{Han}]]` is character-class intersection.** Rust `regex` does support
   `&&`, which makes this look safe, but it must be confirmed through whatever engine handles
   trap 1 rather than assumed to survive the switch.

Both traps corrupt ids without failing, which is the class this repo gates rather than tests.

### Gate design

The house already has the right shape twice — `v4_encoding_gold.rs` and Glimmer's
`glimmer_template_driver.py` + `glimmer-chat-cases.json` (112.8 KB of vendored cases). Follow
it:

1. **Vendor id-pinned cases generated by the first-party stack**, `tokenization_kimi.py`
   against the shipped `tiktoken.model` — never a transliteration, per the anchor rule. Cases
   must cover what the traps break: trailing and interior whitespace runs, `\r\n`, Han text,
   Han adjacent to Latin (the intersection clauses exist for exactly this), digit runs longer
   than 3, the `'s`/`'ll` contractions in both cases, and every one of the 16 named special
   tokens plus at least one `<|reserved_token_N|>`.
2. **Round-trip is necessary but far too weak on its own** — a consistently wrong encoder
   round-trips perfectly. The gate is *id equality against the vendored goldens*.
3. **Assert the boundary explicitly**: `n_vocab == 163840 == config.vocab_size`, and
   `<|end_of_msg|> == 163586 == generation_config.eos_token_id`. A loader that is right about
   text and wrong about where the special block starts produces fluent output that never
   stops.
4. **Red proofs**, each shown red before the green is believed: drop the `\s+(?!\S)`
   alternative; drop one `&&[^\p{Han}]]` intersection; shift the special-token base by one.
   Each must redden id equality, and #3 must additionally redden the boundary assertion.
5. Deviceless, so it runs in CI unlike everything else on this arm.

The eos discrepancy in §3 is a live hazard for step 3: `tokenizer_config.json` says `[EOS]`
163585 and the two generation-side files say 163586. Pin 163586 and pin the disagreement.

### What shipped, and the measured red-proof matrix

**Sixteen gates, all deviceless — and exactly EIGHT assert in CI.** Nine live in
`crates/artifact/tests/k3_tokenizer.rs`; seven more are unit tests inside
`crates/artifact/src/tiktoken.rs`. The seven units plus one integration gate
(`pat_str`/constant equality) need no checkpoint and never skip. The other eight read
`tiktoken.model` out of the **artifact** — not the source, so they also gate what `convert_k3`
shipped — and skip without it, which in CI is all eight.

> This count has been wrong three times (8, then 9, then 13) as gates were added and as a whole
> test binary turned out to be missing from the tally. It is now **derived by script rather than
> written** — `#[test]` counted in both files, minus those that reach `load_or_skip`/`absent` — and
> that is the only reason to trust it. An earlier draft also claimed the censuses make a *skipped*
> run safe, which is retracted below.

Every perturbation was applied, run, and reverted. **LIB** = the seven `src/tiktoken.rs` unit
tests (checkpoint-free); the rest are the nine integration gates — **pat** = constant equality ·
**prov** = vocabulary provenance · **ids** = id equality · **spec** = special block · **bnd** =
boundary+eos · **cap** = both encode caps · **rt** = round-trip · **trip** = vocabulary tripwires ·
**door** = `Tokenizer::load` seam.

| perturbation | LIB | pat | prov | ids | spec | bnd | cap | rt | trip | door |
|---|---|---|---|---|---|---|---|---|---|---|
| RP1 drop `\s+(?!\S)` | **R** | **R** | ok | **R** | ok | ok | **R** | *ok* | ok | ok |
| RP2a–d drop ANY ONE of the four `&&[^\p{Han}]]` | **R** | **R** | ok | *ok* | ok | ok | ok | ok | ok | ok |
| RP3 special base `+1` | ok | ok | ok | **R** | *ok* | ok | ok | **R** | ok | ok |
| RP4 `SPECIAL_SLOTS` 256→255 | ok | **R** | ok | † | † | † | † | † | † | † |
| RP5 `MAX_SAME_CLASS_RUN` −1 | ok | **R** | ok | ok | ok | ok | ok | ok | ok | ok |
| RP6 `RIVOLI_K3_REQUIRED=1`, bogus path | ok | ok | **R** | **R** | **R** | **R** | **R** | **R** | **R** | **R** |
| RP7 golden's vocabulary FNV-1a flipped 1 bit | ok | ok | **R** | ok | ok | ok | ok | ok | ok | ok |
| RP8 inner cap `>` → `>=` | **R** | ok | ok | ok | ok | ok | **R** | ok | ok | ok |
| RP9 `n_vocab` −1 | ok | ok | ok | ok | ok | **R** | ok | ok | ok | ok |
| RP10′ sniff a filename K3 does not ship | ok | ok | ok | ok | ok | ok | ok | ok | ok | **R** |
| RP11 drop "XTML" from `hf()`'s refusal | ok | ok | ok | ok | ok | ok | ok | ok | ok | **R** |
| RP14 a special spelling prefixes another | **R** | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| RP15 reserved-name format `<\|reserved_{id}\|>` | ok | ok | ok | **R** | **R** | ok | ok | ok | ok | ok |
| RP16′ outer cap disabled | ok | ok | ok | ok | ok | ok | **R** | ok | ok | ok |
| RP17 two ids share one spelling | **R** | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| RP12/13 typo in a driver constant copy | — | — | — | — | — | — | — | — | — | — |

Every gate now has at least one perturbation aimed at **its own** claim: `bnd` gets RP9 (not RP4,
see †), `spec` gets RP15, `cap` gets RP8 and RP16′, `door` gets RP10′/RP11, `trip` gets RP14/RP17
as unit tests, `prov` gets RP7. RP12/13 sit outside the matrix — they redden the *driver*, which
exits 1 and refuses to write goldens at all.

**† is not a red.** With `SPECIAL_SLOTS = 255`, `assemble`'s `named_outside` guard fires (`[PAD]`
163839 falls outside `163584..163839`), `Vocab::load` returns `Err`, and every
`load_or_skip(..).expect(..)` panics on the SAME refusal. Those cells are one load failure, not
eight gates seeing their own claims. I had credited RP4 as the boundary gate's proof; **RP9** is.

> **My own red-proof harness produced two false greens, and both were build failures read as
> "nothing reddened".** A shell helper that counts `FAILED` lines cannot tell a clean run from a
> tree that did not compile, and `-D warnings` makes that likely: deleting `Tokenizer::load`'s
> tiktoken arm leaves the enum variant unconstructed (`dead-code`), and disabling the outer cap
> orphans `char_chunks`. Both perturbations were replaced with ones that compile (RP10′, RP16′).
> The lesson is the day's other one restated: **check that the thing ran before believing what it
> reports.** A perturbation the compiler refuses is itself a result — `-D warnings` is a stronger
> guard there than any test — but it must be recorded as that, not as a silent green.

*Italics mark a proof that refused to redden where predicted; each was chased, not accepted.*

**Three cells in italics are the findings.** Each is a red proof that *refused* to redden where
it was predicted to, which P7 treats as evidence about the tree rather than a pass — so each was
chased rather than accepted.

**Two rows overturn claims this section originally made, and both are the useful part.**

**1. Round-trip is not merely weak — it is measurably blind.** Row 1 breaks the lookahead,
which changes ids on real text, and `decode_round_trips_every_case` stays **green**. The
prediction that "a consistently wrong encoder round-trips perfectly" is no longer an argument
for preferring id equality; it is an observation with a run behind it.

**2. RP3 does not redden the boundary assert, and the plan above was wrong to require that
too — twice, since the expectation was restated on resume.** The mechanism: the special block is
built by looking a name up *by* its computed id (`named.get(&id)`), so shifting the base shifts
which SLOT holds an id while `<|end_of_msg|> → 163586` still resolves correctly. The boundary
gate asks exactly that question, so it cannot see the shift. What the shift does break is the
*reserved* spellings and the case ids — caught by **spec**, **ids** and **rt**. The boundary
gate's own proof is **RP4**, which moves `SPECIAL_SLOTS` and reddens it. A gate needs a
perturbation aimed at its own claim, not at a neighbouring one.

**3. Trap 2 cannot be caught by id equality at all, and the plan above was wrong to require
it.** The red proof *refused to redden* the id gate, which per P7 is evidence about the tree
rather than a pass — so it was chased. Removing an intersection does change the pre-tokenizer
(`"hello你好"` becomes one piece instead of two), but the ids are **identical**, confirmed at
the reference level in python: 0 of 12 Han-boundary texts differ. The reason is structural —
**no token in the 163,584-entry vocabulary mixes Han with non-Han**, so no byte-pair merge can
cross the boundary and BPE *reconstructs* exactly the split the intersection would have made.

So id equality is not what protects that trap — **two pattern-level gates are**: the string
equality, and the behavioural split assertion in `src/tiktoken.rs`'s unit tests, which is the
stronger because it observes the effect rather than the text. Both are checkpoint-free and assert
in CI.

> **The behavioural probe was itself wrong twice, and this is the third statement of it.** There
> are FOUR intersection clauses and one probe does not see them all. `"hello你好"` — the original —
> detects only alt-2's LOWERCASE clause; `"A你好"`, which review proposed as the fix, detects three
> of four and misses alt-2's UPPERCASE one; removing that clause changes **nothing** across six
> Han-boundary probes. Measured over all four clauses × ten probes, the covering probe is
> **`"AB你好c"` → `["AB", "你好", "c"]`**, which reddens on any one of the four. Until it landed,
> two of the four clauses had no behavioural gate at all — which is exactly the "only guard" claim
> this section had just finished retracting, true again for half the clauses. `no_vocabulary_token_mixes_han_with_non_han` is then a *tripwire* on the premise, not the
guard an earlier draft called it. If a future vocabulary ships a Han-mixing
token, it reddens and tells the reader that id equality has *started* covering the
intersections. Same shape as Glimmer's `qk_scale_on_k` exclusion: invisible by algebra, not by
resolution.

**What RP5, RP6 and RP7 added.** RP5 pins `MAX_SAME_CLASS_RUN` against the reference's
`MAX_NO_WHITESPACES_CHARS` — it changes ids once it trips, which is the property that earned
`PAT_STR` its string equality, and nothing else would notice a typo because the chunking gate
drives the cap explicitly. RP7 pins the *vocabulary* the goldens came from, recomputed with
`rivoli_core::hash::fnv1a` from the `tiktoken.model` about to be scored — provenance nothing
recomputes is decoration, and without it every id assertion could be scored against a different
vocabulary sitting at the same path. RP6 is the one that matters most for reading a green run:

> **A skip is a PASS, and this file claimed otherwise.** `load_or_skip` returning `None` leads to
> a bare `return`, which libtest scores green — so on a box with no checkpoint the suite reports
> 9/9 having checked one string constant. The censuses defend a *partial* run after a successful
> load; they say nothing about a run that loaded nothing. `RIVOLI_K3_REQUIRED=1` turns every skip
> into a panic (`crates/cli/tests/codescene.rs`'s `RIVOLI_CS_REQUIRED` is the precedent and the
> carve-out argument), and RP6 is that flag against a bogus artifact path: **within the
> integration binary**, 8 red and only its checkpoint-free gate green. The six unit tests are
> unaffected — they need no artifact — so five of thirteen stay green by design. Red-proofing the flag is also what found the last gate still
> carrying its own inline skip — it was the single test that stayed green, which is precisely
> what a proof is for. **Nothing arms the flag today**: there is no K3 smoke script, so that is
> owed, and it is the difference between "runs in CI" and "asserts in CI".

> The driver had the same disease it was built to detect. Its first draft *spelled* the
> reserved-slot expectations itself and put 163839 among them — which is named `[PAD]`, so the
> golden asserted a spelling the reference does not have. The Rust gate caught it on its first
> run. Both sides now derive from the reference's own special map, and a negative case
> (`<|reserved_token_163839|>` must NOT be one id) keeps the distinction live.

### Two gate results that are not about ids

**CodeScene could not be discharged and is owed.** `cs` is installed but unlicensed here, so the
gate warn-and-skips by design (`RIVOLI_CS_REQUIRED=1` correctly turns that into a failure — I ran
it). It therefore did NOT score this module, matching CLAUDE.md's standing note that the
score-below-10 half waits on `CS_ACCESS_TOKEN`. Self-assessed against the band the tree records
instead: 12 primitive arguments across the methods, against a threshold of 11 — inside it. The
biggest contributor was `encode_capped`'s two bare `usize` caps, now one [`EncodeCaps`] value,
which drops it to 10. That refactor stands on its own argument regardless of the gate: the caps are
a pair the reference nests, both are `usize`, and swapping them **compiles and passes every short
fixture** — at `chars = 25_000, run = 400_000` the inner cap simply never trips. Same hazard
`ArtifactDirs` exists for.

**The 800-line soft cap fired at 821 lines** and its instruction is that the next edit shrink the
file, so the unit tests moved to `src/tiktoken/tests.rs` (666 + 164), following `v4_encoding`'s
`mod tests;` precedent. They are the part that moves without splitting an argument across files —
and they are also the only K3 gates that assert in CI, which is worth having in one obvious place.

### The seam, which is where the bug actually was

`Tokenizer::load` sniffs by FILE — `tokenizer.json` first (what the other three checkpoints
ship, and what every recorded benchmark id was produced with), then `tiktoken.model`, and it
names both when neither is present. `Vocabulary` is a two-arm enum rather than a trait object
**because the arms are not interchangeable**: every chat-framing method is built on
`token_to_id`, which only the HuggingFace arm has. A trait would force the tiktoken arm to
answer `None`, and `encode_chat_turns` already responds to missing chat tokens by encoding RAW
with a warning — so the abstraction would have converted "this model has no chat framing" into
a silently unframed prompt. It refuses instead, and a gate asserts the refusal names both the
format and why.

## Open, in order

1. ~~Tiktoken tokenizer loader~~ — **done 2026-08-17**, §7. A K3 artifact opens.
2. **First K3 decode**, correctness-only: ids finite, sane text, no crash, `--ctx` at or under
   the `ATTEND_MAX_KV` 8192 ceiling, small token count. **Not a reproducibility claim** — GLM
   decode is on record as failing to reproduce itself over long runs (496 of 512 ids differing
   after one event; 61/512 on a quiet box), the cause sits in the routed expert pool, and K3
   streams ~92% of a 1.3 TiB routed set against GLM's 22%. So instability is the expected
   default here, and any byte-identity gate for K3 needs a token count and an A-vs-A control
   arm before it means anything. Perf is disclaimed: the artifact is NFS-resident by the
   owner's Q1 decision, so no tok/s from it is citeable.
3. **Chat encoding** (independent of 2) — `encoding_k3.py` is the first-party source and a port
   needs an id-pinned golden against it rather than a hand-read. Not started; `--port` still
   refuses. Note the loader does **not** unblock this: it turns strings into ids, and the
   framing that produces the string is the separate milestone §7 opens by distinguishing.
