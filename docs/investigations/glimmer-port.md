---
scope: glimmer
status: live
- **S1b is DONE and G1b is MET (2026-08-11).** The anchor runs the first-party stack on CPU;
  four goldens across two draws and two modes are vendored, with 14 defect runs each gated on
  what they must leave bit-identical. §S1b, §"The anchor itself".
- **S2 item 1 is DONE (2026-08-12)**: `gqa_attend` and `tests/glimmer_attend.rs`, scored against
  the `attend` tolerance row that was measured **before the kernel existed** (`anchor.md`
  §tolerances). §S2 item 1 carries what it measured, what two reviews found in it, and what it
  still does not cover. **Next is S2 item 2, per-layer RoPE.**
- **This block said "Next is S1b" for a day after S1b landed** — the front-matter verdict was
  updated and the STATE block was not, which is the one place a reader looks first. Both of us
  corrected it independently on the same day, from opposite branches.
- **rivoli had never quantized a checkpoint before, and Glimmer is where it must.** GLM, V4
  and K3 all ship pre-quantized and every converter *copies*; `quant.rs` had a
  `dequant_fp8_block` and no inverse. `quantize_fp8_block` now exists — but choosing its
  scales is **our** quality decision with no upstream to defer to, so the shipped converter
  writes bf16 and S5 owes a dNLL gate the other three ports did not. §S1a item 2.
- **The `SafeWriter` owned-bytes problem is deferred, not solved.** `copy_verbatim` borrows
  the mapped source, so a 55.7 GB bf16 set streams through with no host copy. An fp8 pass
  produces **owned** bytes — ~26 GB resident until `write` — and that is when the two-pass
  shape K3's plan specified comes due.
- **§1 below is superseded.** It is kept as written, with a correction column, because what
  the card got wrong is the reusable lesson. Read `glimmer-architecture.md` instead.
- **The design is inverted relative to every model this engine runs.** Glimmer is dense.
  At fp8 the whole text model is ~26 GB — resident with ~80 GB to spare. The NVMe expert
  streaming, byte arena, residency cache and Belady work are **bypassed, not ported**: a
  Glimmer layer is 52 instances of the no-pool/no-ticket path GLM's 3 dense layers already
  take (`src/gpu.rs:2043` — "attention + MLP were all launches, nothing blocked"). §2.
- **The one new kernel family is attention.** rivoli is MLA-with-q-LoRA and nothing else —
  no GQA, no MHA, no per-layer RoPE, no trained sliding window (`[wt]
  docs/investigations/other-models.md` §"The attention half is not [reusable]"). Glimmer
  needs GQA 32Q/2KV with a 2048-row ring KV on 3 of every 4 layers and full KV on the
  fourth, a sigmoid output gate, and RoPE on local layers only. §3, §S2.
- **The layer body is NOT the shape rivoli already has.** S0 found four norms per layer, not
  two: post-norms applied to the **branch** before each residual add, in a **centered**
  (`x*(1+w)`) form the engine has never implemented — while the final norm and two weightless
  norms keep the plain `x*w` form. Two formulas, two eps values (1e-5 pre, 1e-8 post), and a
  weightless QK-norm **that ships no tensor at all**. `glimmer-architecture.md` §3, §4.
- **RoPE may cost nothing.** Glimmer uses `rotate_half`; rivoli's kernel is interleaved. A
  row permutation of `q_proj`/`k_proj` within each head converts one to the other, and the
  argument that it is safe at conversion time is in `glimmer-architecture.md` §6. **It is an
  argument, not a measurement** — G1b owes it a fixture that reddens on identity.
- **Every greedy gate is blind to the logit path.** `20*tanh(x*0.196116/20)` is
  argmax-invariant, so greedy equality, teacher-forced argmax and byte-identical output all
  pass with it omitted — while every probability, NLL and confidence value is wrong. G3 must
  carry a probability-space check. This is this model's §G blind spot, and it is not layer 0.
- **The FFN is already there.** GLM's dense layers run `gemv_fp8(gate)+gemv_fp8(up)+swiglu+
  gemv_fp8(down)` with runtime dims (`src/gpu.rs:2003-2043`); 6656→19968 is that code with
  different numbers, behind one stale load-time guard (`MAX_FUSED_INTER`, already removed
  on `[wt]`). §3.
- **Speed lives or dies on two numbers:** achieved resident-GEMV bandwidth at Glimmer shapes
  (unmeasured — S0 item 8), and DFlash block acceptance. The theoretical LPDDR5X ceiling
  puts undrafted decode at **≤ ~10 tok/s fp8 / ~20 tok/s int4**; the drafted ceiling is a
  multiple of that because a dense model verifies 17 rows for one weight read — the MoE
  batching-pays-the-union penalty that gates GLM's MTP at 1.108× **does not apply**. §4.
- **Fluent wrong text is still the failure mode.** A wrong window boundary, a rotated
  global layer, or a mis-broadcast KV head all decode fluently. Gates are numeric or they
  are nothing; the K3 gate model (`k3-port.md` §G — prove it can go red, name the blind
  spot) binds every gate here.

## 1. Ground truth as first believed — SUPERSEDED 2026-08-10 by `reference/glimmer-architecture.md`

> **CORRECTED 2026-08-10.** This table was built from the announcement, the HF model card and
> the DFlash abstract, all through a summarizing fetch layer, and it is left in place because
> the *shape* of its errors is the lesson. S0 replaced it with first-party sources. The
> headline shapes were right; **the card omitted eight load-bearing operations entirely** —
> QK-norm, `qk_scale_factor`, `output_multiplier`, `final_logit_softcapping`, `post_norm_eps`,
> the sandwich-norm structure, the centered-norm formula, and the normed embedding. Two rows
> were wrong outright: the release is **BF16, 59.553 GB** (not "~4-bit, under 20 GB" — those
> are separate GGUFs), and the context is **exactly 131072 with no scaling scheme**, not
> "131,072+". The full corrected specification, with the traps each of these produces, is
> `docs/reference/glimmer-architecture.md`; its §10 tabulates card-vs-truth.

| field | claimed value | consequence if true |
|---|---|---|
| architecture | dense causal transformer, 52 layers | no router, no experts, no arena — §2 |
| hidden / FFN inter | 6656 / 19968 (SwiGLU) | existing fp8 dense-MLP path; both divide 128 |
| heads | 32 Q / 2 KV, head dim 128 | q_proj is 6656→4096, **non-square**; nothing in any existing config expresses `head_dim ≠ hidden/n_heads` — new config fields |
| attention pattern | [local, local, local, global] repeating | 39 local / 13 global; per-layer kind table |
| sliding window | 2048, local layers only | ring KV, 2048 rows × 2 heads — bounded |
| positional | RoPE θ=500,000 on **local layers only**; global layers NoPE | per-layer RoPE toggle; K3's NoPE-scored-not-rotated is precedent but NOT the same rule — here the dims are presumably not rotated at all. S0 must settle which |
| gated attention | yes, form unspecified | assume per-head sigmoid gate before o_proj (K3's gated-MLA shape); **S0 must confirm form and placement** |
| context | 131,072+ | global-layer KV at full context ≈ 1.7 GB (§5); "+" means a scaling scheme S0 must find |
| vocab | 202,048 = 200,000 BPE + 2,048 special | logits row 808 KB; int8 lm_head ≈ 1.34 GB, read whole every token |
| tie_word_embeddings | **unknown** | ±1.34 GB resident and ±traffic |
| release format | BF16 safetensors; K-Quant GGUFs exist | rivoli converts from BF16; GGUFs are irrelevant except as a quality cross-check |
| drafter | DFlash: 5 layers, block 16, 32Q/8KV, separate release | MAXROW 2→17; drafter conditions on target-model context features — **the interface is the paper's §method, unread. S0 item** |
| vision | ViT-G/14, ~1.8 B, 50 blocks, ≤4096 visual tokens | converter skips explicitly; `serve.rs` image refusal stays |
| chat template | unknown; "agentic" implies tool framing | hand-port + byte-level pinning test, per `tokenizer.rs` precedent |

## 2. The inversion — what a dense model does to this engine

Everything rivoli exists for — overlap of NVMe expert streams with resident compute — is
unused here. `LayerMlp::Dense` (`src/memory/pin.rs:101-108`) already models a layer that
touches no pool, no cache, no fetch, no ticket: Glimmer is that arm 52 times. Consequences,
each worth stating because it deletes a stage the other ports needed:

- **No residency stage.** No cache policy, no `--max-mem` sensitivity, no Belady question.
  A `GlimmerPin` owns everything; the `ArenaPool` remainder goes unused. (`resident_bytes`'s
  `(n_pin - dense_layers) * shared` term is zero-correct, but `Pin::build` still opens
  `ExpertSet`s that won't exist — one reason `GlimmerPin` is its own type, §S1a.)
- **Single format, so quality A/Bs are safe.** The hybrid-mode defect — residency selecting
  the arithmetic (INV-1 exception, `architecture.md` §8b) — cannot occur: there is no
  per-expert format choice because there are no experts.
- **Batched rows are nearly free.** GLM's MoE pays the union of routed experts across rows
  (measured 1.61× for 2 rows); a dense verify pass reads each weight once regardless of
  `nrow`. This is why DFlash block-16 is the headline perf lever (§4) and why the
  `--mtp-min-conf`-style acceptance economics need re-deriving rather than copying.
- **The bottleneck moves to GTT read bandwidth.** Undrafted decode is one full read of
  ~25 GB (fp8) per token. NVMe numbers (12.39–14.76 GB/s at the expert shape) are
  irrelevant; the governing number is sustained resident-GEMV GB/s, which rivoli has never
  needed to measure at these shapes. S0 item 8 measures it with the existing
  benchmark-only `gemv_i4`/`gemv_fp8` harness before any prediction is registered.

## 3. What is new, what is reused

Against the `[wt]` tree; **re-verify each row when K3 merges** (K3's S2/S3 will have moved
some of these).

| piece | status |
|---|---|
| fully-resident layer path (no pool/fetch/ticket) | **reuse** — `LayerMlp::Dense` arm, 3→52 layers |
| dense SwiGLU MLP at runtime dims | **reuse** — `src/gpu.rs:2003-2043`; delete stale `MAX_FUSED_INTER` guard (done on `[wt]`) |
| fp8 block-scaled GEMV (+`_r2`, `_splitk`) | **reuse verbatim** — `kernels/linalg.hip:40-102` |
| int8 embed / lm_head, per-artifact vocab | **reuse** — `cfg.vocab` is already data-driven; 202048 is a size note, not a code change |
| `Arch` enum, `ArchConfig` refusal-tested configs, per-arch help | **reuse** — add third/fourth arm (`[wt] src/arch.rs`) |
| streaming `SafeWriter`, `Safetensors::open_indexed`, converter-per-model | **reuse** — arrives with K3 S1a; Glimmer's artifact is the simplest yet: `manifest.json` + `resident.safetensors`, **no expert files at all** |
| `FormatMeta` | **modify** — it hard-asserts VQ params on artifacts that have no VQ tensors; needs a nullable VQ section |
| tokenizers-crate loading, EOS from `generation_config` | **reuse** — `src/artifact/tokenizer.rs` |
| chat template | **new hand-port** + its own byte-level pinning test (the GLM template drifted for months; `preserve_order` note in `Cargo.toml`) |
| ring-buffer windowed KV | **adapt** — V4's `Sel { win, kind, … }` + ring cache (`[wt] src/v4gpu.rs:355-380`) is the closest machinery, minus V4's compression |
| per-layer RoPE toggle, θ=500k | **new plumbing** — today RoPE is unconditional with one scalar θ; V4's `Defect::RopeNoYarn` note is the design memo |
| **GQA attention kernel (32Q/2KV, dense + windowed)** | **NEW kernel family** — `mla_latent_attend` shares no memory layout; the bulk of S2 |
| per-head output gate | **new kernel, known shape** — K3 S2 ships gated MLA first; same pattern, different site |
| MAXROW=2 verify loop → block-17, LCP acceptance | **generalise** — `Draft`/`Span`/`forward_inner` already parameterise rows and layer ranges; acceptance walk and scratch sizing are the work |
| 5-layer drafter forward | **new, small** — `Span { layers: 0..5 }` carries it; the block-diffusion draft step itself is new arithmetic (S0 reads the paper) |
| expert streaming, io_uring, arena, hybrid cache, router, KDA, AttnRes, MLA | **not used** |
| perception encoder | **out of scope** — §S6 |

## 4. Traffic and the predicted operating point — REGISTER IN S0, NOT HERE

**Superseded by measurement 2026-08-10** — the numbers below are now summed from the
safetensors headers (`glimmer-architecture.md` §7), and the sum reconciles with the index's
own `total_size`, which is what makes it a check rather than an estimate. The original
estimates (~25 GB/token fp8, ~13 int4) were close; they are replaced, not corrected, because
nothing depended on them yet.

Per-token weight traffic, undrafted, all-resident: **bf16 53.020 · fp8 26.510 · int4+g128
13.648 GB/token** (52 × 967.889 MB of layers + 2.690 GB lm_head).

KV is asymmetric and the asymmetry is the interesting part: the 39 sliding layers are capped
by the window at **81.79 MB total, forever**, while the 13 full layers read **1.745 GB/token
at bf16** (0.872 fp8) at 131k context. At long context a quarter of the layers cost more than
a fifth of the fp8 weight traffic.

Ceiling arithmetic: at the LPDDR5X theoretical ~256 GB/s, fp8 undrafted ≤ ~10 tok/s. The
*achieved* number is S0 item 8; register the predicted band from it and hold S5 to that band.
DFlash multiplies this by (accepted tokens per verify pass); the card claims 3.1× on an
RTX 5090 — treat as vendor-optimistic until G6 measures acceptance here.

## 5. Memory budget — not binding, assert it anyway

fp8 text model ~26 GB + KV ~1.8 GB + drafter (~1 GB?) + scratch ≪ `--max-mem 115`. The
budget check stays (a wrong config must still refuse), but unlike K3 nothing here is
capacity-planned. tmpfs and contention discipline per CLAUDE.md still apply.

## Stages. K3's gate model (`k3-port.md` §G) binds: met or not met, prove each gate can go red.

### S0 — ground truth. No code, no weights. **DONE 2026-08-10 except item 5.**

1. **Done.** `config.json`, `chat_template.jinja`, `tokenizer_config.json`,
   `generation_config.json` and the safetensors **headers** (by HTTP range request, so shapes
   are the shards' own) pulled raw at pin `f84ecc3a0ea984a4c04542a84269e3d065350a6e`.
2. **Done.** DFlash settled from the paper, the drafter checkpoint's own header, and
   transformers' `models/muse_glimmer_assistant/`. §S6, `glimmer-architecture.md` §11.
3. **Done.** NoPE confirmed as *no rotation at all* (not K3's cached-and-scored); gate is
   `sigmoid(gate_proj(layer input))` before `o_proj`; window is `[p-2047, p]` — 2048 rows
   inclusive of `p`, from `masking_utils.sliding_window_overlay` and its own docstring;
   no context-scaling scheme (`rope_type: "default"`, exactly 131072);
   `tie_word_embeddings: false` with both matrices shipped.
4. **Done.** `docs/reference/glimmer-architecture.md`, line-cited to
   `modeling_muse_glimmer.py`, with no summarizing fetch anywhere in the chain.
5. **MOVED to S5. Not done, and it was never a G0 item.**

   > **CORRECTED 2026-08-10.** This was written as a G0 item ("register the predicted band
   > from a measured number") and that was a placement error, not a discovery. G0 asks
   > whether the *forward pass* is known; sustained GB/s is a throughput fact that **nothing
   > in S1, S2, S3 or S4 consumes** — the §4 economics that matter early (fp8-vs-int4, the
   > DFlash break-even at N>1.1) are ratios of byte counts and need no absolute bandwidth.
   > Left where it was, a contended GPU would have blocked all correctness work for a number
   > only S5 spends. It is now a **precondition of S5/G5**, where it binds. The move does not
   > weaken it: no throughput claim may be registered without it, exactly as before.

   Attempted twice 2026-08-10 and correctly refused both times: `/var/run/sys-gpu.lock` held
   (`flock -w` → exit 1, captured rather than swallowed — the false-green trap),
   `gpu_busy_percent` 100 across six one-second samples, llama-swap holding 41.24 then
   64.02 GB of GTT across `qwen3-embedding-4b`, `qwen3.6-medium` and `whisper`, with **zero
   `/sys/class/kfd/kfd/proc/` entries** — a kfd-only check would have called the machine free
   and produced a number that had to be discarded. Re-run under `reference/gpu-lock.md`'s
   shared-lock contract.
6. **Done, and it moved.** The least representative layer is **not** layer 0 — every layer is
   structurally identical here, so the blind spot is arithmetic, not positional: the logit
   path is argmax-invariant (§STATE). Gates must cover a sliding layer, a full layer, a
   window-boundary crossing, layer 51 (full, and last), **and probability space**.

### G0 — **MET 2026-08-10**

Every unknown has a recorded answer and a first-party source, and
`reference/glimmer-architecture.md` exists and is line-cited. What G0 does **not** assert,
so that the next reader does not over-trust it: two things there are arguments rather than
measurements — §6's RoPE permutation (G1b owes it a fixture) and the DFlash break-even (a
byte ratio, not an observed τ) — and the bandwidth item is now S5's, above.

**S1 is blocked, but not on G0.** It is sequenced behind Kimi-K3: `Arch`/`ArchConfig`, the
streaming `SafeWriter` and the per-arch converter shape all arrive with K3's own S1a, which
is in progress on `wt/k3-s1a`. Building that seam here would collide with it. When K3's S1a
lands, re-verify §3's reuse table against the merged tree before starting — several rows
name files K3 will have moved.

### S1a — artifact. **DONE 2026-08-11**, all four items. No GPU except item 3's gate.

1. **DONE.** `Arch::MuseGlimmer` + recogniser tests both directions; `GlimmerConfig` /
   `GlimmerTextConfig` with no defaulted fields; `main.rs` dispatch that parses and then
   refuses. The shipped binary reads the unmodified `config.json` and reports the layer map
   before bailing. Config vendored at `docs/measurement/glimmer-reference/config.json`
   (HF revision `f84ecc3`) and the schema is pinned to it by a test that always runs.

   Two things this produced that were not in the plan:

   - **The pairing invariant is enforced at the load boundary.** `layer_types` and
     `layer_rope_theta` are independent arrays and a layer slides IFF it is rotated; they can
     disagree, and a disagreement is not a shape error anywhere downstream. It is the
     strongest claim the config alone can make, so it is made where the file is read rather
     than left to an S1b fixture.
   - **`head_dim` had to become a first-class field, and a test asserts the inequality.**
     32 × 128 = 4096 against `hidden_size` 6656 — the only config here whose head dim is not
     `hidden / n_heads`, so a later simplification to the derived form builds 208-wide heads.
     Asserted as `head_dim != hidden / n_heads` so it reddens for that reason.

   Gates: a **26-row defect run** over the vendored config plus an inserted-key assertion,
   each row asserted on its own refusal *message*; and `every_glimmer_field_is_required`,
   which compares the whole set of keys whose removal is refused against the set tolerated as
   absent, so a field that stops being required moves between them. Both proven able to go
   red. `output_multiplier` and `final_logit_softcapping` are checked at the load boundary
   **because** they are argmax-invariant — no greedy gate downstream can see them wrong.

   Two things the red-proofs corrected, which is the argument for running them rather than
   asserting a gate is sound:

   - `every_glimmer_field_is_required` **does not** redden on a `#[serde(default)]` for
     `head_dim` — the default 0 is caught by `validate`'s width loop, so the checkpoint is
     still refused and no defect exists. It reddens on `attention_bias`, where the default is
     the acceptable value. The doc originally claimed the `head_dim` case and was wrong.
   - Two defect rows drafted from the review were themselves wrong and were cut: `/dtype` set
     to `bfloat16` is a no-op, and `rope_parameters.rope_theta = 0` trips the pairing check
     before it can reach the narrowing check it was meant to exercise.

   Incidental: jscpd caught the f32-narrowing loop against K3's, factored to
   `ensure_f32_positive` rather than exempted — one rule about the hardware, unlike the
   dimension serde renames beside it where the shared text is a coincidence of four
   checkpoints.
2. `src/bin/convert_glimmer.rs`: BF16 → fp8-block resident artifact via streaming
   `SafeWriter`; skip `vision_tower`/projector explicitly; copy tokenizer +
   `generation_config`. No expert files; `FormatMeta` gains its nullable VQ section.

   **The quantizer landed first (2026-08-11): `quant::quantize_fp8_block`.** Writing it
   surfaced something the plan had assumed away — **rivoli has never quantized a checkpoint
   before.** GLM ships fp8, V4 ships fp8 + e8m0, K3 ships mxfp4; every existing converter
   *copies* a decision the publisher made, and `quant.rs` accordingly had `dequant_fp8_block`
   and no inverse. Glimmer ships BF16 and nothing else, so **the scale per tile is our
   choice** — a quality decision with no upstream to defer to. Two consequences, both new:

   - **S5 owes a dNLL gate the other three ports did not.** "Is fp8 good enough for Glimmer"
     is now a question about *rivoli's* quantizer, not about the checkpoint. Single format, so
     the A/B is safe (no hybrid-residency hazard) — `bin/ppl` on the 5000-token corpus against
     a bf16 artifact as the reference arm.
   - **A bf16 resident artifact is the honest fallback**, and worth pricing before the fp8
     path is assumed. `Dtype::Bf16` is already accepted, the converter becomes a near-verbatim
     copy — which makes `SafeWriter`'s owned-bytes problem disappear with it — and there is no
     quantization question at all. It costs 53.02 GB/token against 26.51 and runs on the slow
     bf16 GEMV. Correctness first, then earn the halving with the gate above.

   Gates on the quantizer: round-trip bounded by e4m3's **half-ULP relative to each value**
   (the loose form, against the tile amax, would pass a quantizer that collapsed every small
   value in a tile to zero); an all-zero tile takes scale 1.0, since 0.0 makes the encode
   `0/0` and `f32_to_e4m3(NaN)` is a finite-looking `0x7f`; non-finite inputs refuse rather
   than encode. Proven able to go red on three defects — halved scale, transposed scale grid,
   zero-scale zero-tile — each reddening the cases it touches and staying green elsewhere.
   **`convert_glimmer` landed 2026-08-11, bf16 verbatim.** It quantizes nothing: norms widen
   to f32 (the house convention every arch follows), everything else is `copy_verbatim` at
   BF16, and the three vision families are skipped by an explicit predicate whose *count* the
   run prints — so their exclusion is an observation, not an assumption. `SafeWriter` borrows,
   so the 55.7 GB set streams through with no host copy and **the owned-bytes problem is
   deferred rather than solved**; it returns with the fp8 pass.

   Two plan items turned out to be work nothing needed. **`FormatMeta` needs no nullable VQ
   section** — `current()` stamps the compiled-in VQ constants and `load()` compares them
   against the same constants, so they always agree and are inert on an artifact with no VQ
   tensors. And the tensor-name list is now **one** constant (`GLIMMER_LAYER_TENSORS`) read by
   both converter and test, after jscpd caught the second copy: a duplicated list of *names*
   is precisely what `tests/k3_names.rs` exists to prevent.

   Gate: `tests/glimmer_convert.rs` runs the real binary over a synthetic four-layer
   checkpoint — byte-identity per verbatim tensor, correct *values* (not just lengths) for
   every widened norm, vision absent, the artifact re-opening as the same model through
   `load_config` + `FormatMeta::load`, and the two refusals that must fire before 55 GB is
   written (incomplete checkpoint by name; out_dir == src_dir, which is a SIGBUS risk because
   the writer maps the source while it writes). Proven red on two defects: un-skipped vision,
   and norms copied verbatim instead of widened. **What it does not establish is tensor
   NAMES** — the fixture is built from the same list the converter checks, so a name wrong in
   both is wrong in both.

   **That gap is now closed, and the demonstration is worth keeping.**
   `tests/glimmer_names.rs` pins every name against the shipped
   `model.safetensors.index.json`, reduced to 40 families and vendored at
   `docs/measurement/glimmer-reference/tensor-families.tsv` (sha256 in its header; dtype and
   shape are the shard headers' own, by HTTP Range). It asserts the reduction is of the real
   checkpoint (1436 tensors, all BF16, summing to `metadata.total_size` 59,553,253,376), that
   every name and shape matches — `q_proj`/`gate_proj` share `[4096, 6656]` and `k_proj`/
   `v_proj` share `[256, 6656]`, so those two pairs are separable only by NAME — and that
   **every family is either implemented or deliberately skipped** (627 text / 809 vision),
   which is what makes "we skip the vision half" a measurement rather than an assumption.

   Red-proof: mis-transliterating `self_attn.gate_proj` as `self_attn.output_gate` fails
   `glimmer_names` twice with precise messages **while `glimmer_convert` stays green** — the
   two tests are not redundant, and this is the evidence.
3. **DONE 2026-08-11.** `GlimmerPin` / `GlimmerLayerPin`, its own type per the V4/K3
   precedent, in `memory/pin.rs` beside `Pin` and `F4Pin` so the four private placers
   (`place_bf16`, `place_f32`, `dims2`) stay private.

   **`build` takes no `capacity` and no cache policy, and that absence is the port.** Both
   parameters exist on the other two pins to divide a budget between a resident tier and a
   streaming pool. There is no pool: the resident set *is* the model, its size is a function
   of the config alone, and a device that cannot hold it cannot run Glimmer at any setting.
   `DeviceTier::new` already refuses what does not fit and names the shortfall.

   Two corrections to this item as planned:

   - **"int8 embed/lm_head placement reused" was wrong** — that is GLM's, and it presumes an
     int8-quantized artifact. `convert_glimmer` writes bf16 verbatim (item 2, "correctness
     first"), so the reuse is V4's `place_bf16`/`Bf16Weight`. The int8 question returns only
     if the fp8 pass also quantizes the embedding, which it need not.
   - **"its own tensor-name table" would have been a third copy.** The names were already one
     constant; what the pin additionally needed was *shapes*, and `tests/glimmer_names.rs`
     had its own copy of those. Both now read
     `GlimmerTextConfig::layer_tensor_shape` — one table, and the names test is what
     validates it, since it resolves every entry against the shipped
     `model.safetensors.index.json`. So the table the pin checks placements against is not a
     belief about the checkpoint; it is confronted with it.

   `GlimmerTextConfig::resident_bytes` is derived from that same table and **sizes the tier**,
   which is what makes it load-bearing rather than documentation: under-count it and
   `DeviceTier::place` bails partway through a 55.7 GB load. **55,712,344,064 bytes**, and
   `tests/glimmer_names.rs` reproduces it from the vendored index's own shapes — G1a's fourth
   clause, met without a device. It is 2.782 MB above the checkpoint's text half because the
   209 norms widen bf16→f32.

   Gates: `tests/glimmer_pin.rs`, a **GPU arm** (own file, so `glimmer_convert`'s two
   deviceless tests stay runnable in CI, which has no rocm job). It converts the shared
   fixture and then pins it, which is the point of sharing — a pin test on its own checkpoint
   would establish nothing about the pipeline. It asserts **the bytes arrived**, not only the
   dims: the tier is a host-fillable VMM allocation, so every pointer the pin hands out is
   readable from the test, and every tensor in the fixture carries a distinct blob. That is
   what separates `q_proj` from `self_attn.gate_proj` and `k_proj` from `v_proj` — within
   each pair the shapes are identical, so a field wired to the wrong name is a *value*
   failure and nothing else can see it. Plus the refusal: a config implying different dims
   must name the tensor, and the defect chosen (`num_key_value_heads` doubled) makes the tier
   LARGER on purpose, so what fires is the shape check rather than a capacity bail.

   One drift guard the pin cannot derive: it names all twelve per-layer tensors as struct
   fields, so a thirteenth entry in `GLIMMER_LAYER_TENSORS` would be placed by nothing. The
   test asserts the count is 12 and says why.

   **Proven red on three defects, each reddening only its own test:**

   | defect | reddens | and this is the point |
   |---|---|---|
   | `attn_gate` placed from `self_attn.q_proj`'s name | the placement test, on VALUES | the shape check passed — both are `[4096, 6656]`, so nothing but the bytes can see it |
   | the shape `ensure!` deleted | the refusal test only | "a config implying different dims must be refused" |
   | `resident_bytes` counting norms at bf16 | the accounting test, deviceless | 55,709,575,168 against 55,712,344,064 — the 2.782 MB of widening, exactly |

   And one accidental red worth keeping, because it says something about the refusal test's
   shape. The first run failed with *"the refusal must name the tensor, got: refusing to
   start: 5.3 GiB GPU memory already in use by another tenant"* — `GlimmerPin::build` did
   return `Err`, so an `is_err()` assertion would have passed **vacuously on a machine with a
   busy GPU**. It asserts on the message instead, and that is why it failed rather than
   lying. (The tenant was llama-swap's `qwen3-embedding-4b`, holding 5.3 GiB with **zero**
   `/sys/class/kfd/kfd/proc/` entries — the blind spot `kfd-blind-to-vulkan-tenants` records,
   found again here. `GET /unload` freed it; the flock alone would not have.)
4. **DONE 2026-08-11.** `run_glimmer` — the config parse, the layer-map log, nine flag
   refusals and the decode bail, all in one function; the dispatch arm is now a call.

   **The refusals are written BEFORE the decode path, which inverts the order V4 and K3
   set.** `run_v4`'s bails were written alongside its decode and K3's comment predicts the
   same for `run_k3`. That order is what the failure needs: `Arch::hidden_flags` is consumed
   only for clap's `.hide(true)`, so the parser still accepts every flag it hides, and a
   branch that omits a refusal compiles clean, passes clippy, and silently takes a knob it
   cannot honour.

   **A dense model refuses more than V4 does, and the extra two are the residency knobs.**
   `--cache-policy` picks an eviction policy for a pool that does not exist, and `--max-mem`
   sized the budget whose remainder GREW that pool — `GlimmerPin::build` takes no capacity at
   all. Accepted, both would be read by nothing, and a line in `benchmarks.md` carrying
   `--max-mem 70` would be a claim about a constraint that never applied. `--trace` joins
   them: it dumps the routed-expert *access* trace. Nine total, against V4's three.

   `--window` is the one that is dangerous rather than merely inert. This model HAS a sliding
   window — 2048 rows on 39 of 52 layers — but it comes from `sliding_window` in the config
   and is a property of how the weights were trained. A flag that reads as if it sets that
   and does not is worse than one that names nothing.

   Incidental: clap's defaults for `--sinks`/`--window`/`--misa-heads`/`--cache-policy` are
   now named constants, because a refusal has to compare against the same value clap
   defaulted to and the alternative is the literal written twice.

   Gate: `tests/glimmer_flags.rs`, which runs the **shipped binary** over the converted
   fixture — the only thing that can tell a hidden flag from a rejected one. Three tests, and
   the second is the one that earns its place: **a bare run must reach the decode bail with
   no refusal firing.** A table of nine `bail!`s where one compares against the wrong default
   refuses a user who passed nothing, which reads as "this model is broken", and the
   per-flag test cannot see it. Both halves proven red — deleting the `--cache-policy`
   condition reddens the first test only; comparing `--window` against 2048 instead of its
   default reddens **both**, the bare-run test being the one that says why.

   Two findings from the first run, neither predictable from reading the code: **`--port` and
   `--bench` are mutually exclusive in clap**, so a refusal test that passes `-bench` can
   never reach the `--port` bail (it dies in the parser); and `tracing`'s fmt layer writes to
   **stdout** while `anyhow`'s report goes to stderr, so the layer-map log and the bail that
   follows it land on different streams. A test reading one stream gets half the run.

**G1a — met when** conversion round-trips bit-exact on sampled tensors; GLM/V4/K3 artifacts
still open byte-identically (test opens them); a config missing or contradicting any
load-bearing field refuses at startup, proven by feeding it one; resident byte accounting
reproduced from the artifact.

### G1a — **MET 2026-08-11.** Clause by clause, with what backs each

| clause | met by | note |
|---|---|---|
| conversion round-trips bit-exact **on sampled tensors** | `glimmer_convert` | stronger than asked: **every** verbatim tensor is compared byte for byte, and every widened norm by *value* rather than by length — a length check passes on a zeroed tensor |
| GLM/V4/K3 artifacts still open (test opens them) | `arch_artifacts` | `arch_of_artifact` **and** the full `load_config`, so a `validate` that tightened for everybody fails here. GLM and V4 open on this machine; **K3 SKIPs — no artifact exists yet** — and the test asserts at least one was present, so it cannot go green having opened nothing |
| a config contradicting a load-bearing field refuses **at startup**, proven by feeding it one | `glimmer_flags` | the *binary*, not the type. The 26-row defect run proves it about `GlimmerConfig`; this proves the dispatch parses before it bails, which was a comment and is now a gate. The defect fed is the pairing invariant, the one no downstream shape check could catch |
| resident byte accounting reproduced from the artifact | `glimmer_names` | **55,712,344,064**, computed from the vendored index's own shapes rather than written down |

**The blind spot, named as the gate model requires.** Every one of these is about *bytes and
names*, and not one of them evaluates an arithmetic expression. The artifact is bf16-verbatim,
so "round-trips bit-exact" is a statement about `memcpy` — it would hold identically if every
tensor in the checkpoint were noise. Nothing here can see a wrong window boundary, a rotated
NoPE layer, a mis-broadcast KV head, or the argmax-invariant logit path. That is S1b's whole
job, and G1a being met says only that the inputs to S1b are the ones the checkpoint ships.

### The review round, 2026-08-11 — what four reviews found after G1a was declared MET

Ponytail, correctness, testing and adversarial, all run against `k3-s1a..HEAD`. **Two CI
breaks and one silent-wrongness class**, none of which any gate on this branch could see.

| finding | who | what it was |
|---|---|---|
| **`tests/glimmer_flags.rs` had no `#![cfg(feature = "rocm")]`** | all four | It runs the shipped binary, and a featureless `rivoli` is a refusal stub — so the ONE CI job this repo has (`cargo test --release --locked`, featureless) went **3/3 red**. Verified by running it. The file's header claimed "deviceless, and that is a property of where the bail sits"; deviceless was true and featureless was not |
| **`tests/arch_artifacts.rs` asserted `opened > 0` unconditionally** | testing, adversarial | `/var/db/rivoli` does not exist on a CI runner, so the anti-vacuity assert *was* the CI break. And "skips loudly" was false — libtest captures stdout of PASSING tests, so a run that degraded from two architectures to one looked identical to a full one. Now: no artifacts → print and return; some → `EXPECTED_PRESENT` |
| **All five f32 norms were placed with NO extent check** | correctness, adversarial | `place_f32` discards the shape and the norm fields are bare `*const f32`. A norm shorter than `hidden` is placed into a tier sized for the full width and handed to S2's RMSNorm as a `hidden`-long array — reading inter-placement padding and the next tensor's bytes. **In bounds of the slab, no error, a scaled-wrong residual stream.** The adversarial reviewer ran the shipped converter on a `[7]`-instead-of-`[8]` norm: `exit 0`. `place_glimmer_norm` now checks it, and `glimmer_pin_refuses_a_norm_that_is_not_hidden_long` converts exactly that artifact and pins it |
| **Nine refusals compared VALUES, so passing a default was accepted** | adversarial | `--mode hybrid --cache-policy 2q --window 8192` fired zero refusals — and `tests/mode-matrix.sh` passes mode/policy/attn explicitly, so it is the ordinary case. "Was this flag typed" and "does it hold a non-default value" are different questions and only the first is the refusal's. `parse_args` now returns clap's `ValueSource::CommandLine` set; the table asks about presence. The gap sat exactly between the two existing tests — one passed only non-defaults, the other passed nothing |
| **`--moe-gain` was accepted and read by nothing** | testing | The one row the table missed, and the one flag with no `Config` field — so nothing downstream could have refused it. `--mode` is refused with "there are no routed experts"; this is the same argument |
| **The refusal message claimed "hidden from --help" for four flags that were not hidden** | correctness | Fixed on the other side: `MuseGlimmer` now has its own `hidden_flags` arm with nine entries, split from the shared V4/K3 one. A flag a model cannot honour should not be advertised in that model's help |
| **`generation_config.json` was called load-bearing while its absence was a `WARNING`** | correctness, adversarial | And the branch's own fixture shipped without it, so every green run certified the artifact shape in which **trap 13 (the scalar EOS) is live**. `REQUIRED_AUX` refuses before any weight is read; the fixture now ships all four |
| **The skipped-vision count was a function of the checkpoint's SHARD boundaries** | testing | `open_indexed` selects whole shards, and `want` excludes vision — so a vision-only shard is never opened and its tensors never reach `names()`. The count is now taken from the index. The single-shard fixture could not have shown this |
| two comments asserted guarantees the code did not make | ponytail, testing | `is_vision`'s doc argued from a "positive predicate" that is `!is_vision(n)` — the real behaviour is the inverse, an unrecognised vision prefix is COPIED. And `glimmer_names`' counts do not gate the converter's predicate at all |
| the tensor count was wrong in two comments | ponytail, correctness | "Five projections and four norms" is nine against a list of twelve, and "5 norms per layer" against four. `pin.rs` had it right, so **the tree disagreed with itself about the length of the one constant that exists to stop that** |

**One finding was wrong and is recorded as such.** Both ponytail and I considered replacing the
flag bundle with `&Args`; it does not compile — `Config` moves four of that struct's `String`s
just above the dispatch, so `a` is partially moved and `&a` will not borrow. That is the same
constraint `run_v4`'s doc already records. (The bundle is gone anyway: presence-based refusals
need only the id set.)

**Declined, with the argument:** `quantize_fp8_block` has no production caller and ponytail
proposed deleting it until the fp8 pass. Kept — it is inert, its three red-proofs are the
thing that would be lost, and the plan sequenced it deliberately at item 2.

**Still open, recorded rather than fixed:** nothing ties the vendored fixtures to HF revision
`f84ecc3`, so a revision that ADDS a tensor family would be copied verbatim into the artifact,
never placed by the pin, and every gate would stay green (adversarial). `convert_glimmer` has
no membership test against `GLIMMER_LAYER_TENSORS` on the way in, and no provenance stamp — GLM
has `I4Source` and Glimmer has no analogue. **That belongs in S1b**, beside the goldens that
will also be revision-pinned.

**Two things G1a did not ask for and this stage produced anyway**, both because the plan
underestimated what a *fourth* architecture costs: the tensor-name pinning
(`glimmer_names`, against the shipped index — a name wrong in both converter and test is
wrong in both), and the flag refusals arriving before the decode path rather than with it.

### S1b — gate harness. No GPU. Must not touch kernels.

Tiny-model goldens emitted from the **first-party HF modeling code** at tiny dims (the
anchor rule from K3 G1b: a reference attesting to itself is not independence — here the
first-party stack is plain `transformers`, so the anchor is cheap). Per-operator fixtures
for: GQA attend (dense + windowed), ring-KV append/evict at the 2048 boundary, per-layer
RoPE table, output gate, SwiGLU at 6656/19968, and the DFlash draft step.

**G1b — met when** every golden has a recorded defect run reddening exactly where the defect
touches, covering: a local layer, a global layer, a window-boundary crossing, layer 0,
layer 51.

#### Item 0 — the vendored index. **DONE 2026-08-11.** No GPU, no venv, no network at test time.

Not in the plan as written; it arrived from K3's side of a cross-session exchange and it had
to come first, because every later golden is compared against a tensor set this port believes
in. `model.safetensors.index.json` is now vendored whole (132674 B, sha256 `7d817b4d…`,
verified against the live fetch *before* pinning) beside the reduction it produced.

**What was wrong.** `tensor-families.tsv` is a hand reduction, and
`glimmer_names.rs` checked it against three constants — `40` families, `1436` tensors,
`59_553_253_376` bytes — each of which the TSV's own header already declared, taken from an
index that was not in the tree. Two frozen copies agreeing with each other while neither
described anything on disk. **A structural test catches a field that MOVED; one that was
ADDED is precisely what it cannot see**, so an upstream revision growing a tensor family gave
a TSV that never mentioned it, a test that never asked, and green. K3 found the same shape
three times in two days (`k3-port.md`); this is the fourth.

**What it is now.** Families and counts are derived from the vendored index and diffed against
the checked-in TSV, both directions named. The byte total is read from the index and compared
against a sum over TSV *shapes* — which came from the shard headers by HTTP Range, since the
index carries neither dtype nor shape. **The TSV deliberately stays a checked-in artifact
rather than becoming a cache of the derivation**: derive both sides and the comparison
degenerates into testing `product()`, and the hand-transcription errors it currently catches
stop being covered.

Four red-proofs, each restored: an added family (`self_attn.qk_norm` injected into the index)
names the family; a per-family count drift names the row; a shape drift fires the byte
cross-check, proving it is not a tautology; and **`metadata.total_size` deleted PANICS** rather
than defaulting to 0 against an empty sum, which would have been a gate that reads as coverage
and is zero.

Vendoring is available here only because the file is 132 KB. K3's is 60 MB and can only diff
against a fresh fetch — that is a size accident, not a better design, and the note in
`glimmer_names.rs` says so where someone might copy it.

#### Ordering defect found in passing, and fixed — the budget check preceded the refusals

`tests/glimmer_flags.rs` failed all three cases while a sibling agent held ~100 GB of GTT for a
benchmark: `check_budget` sat above the architecture dispatch, so a Glimmer artifact given
`--attn dsa` reported *"only 12.9 GB available; need more than the 17 GB OS reserve"* instead
of saying the flag does not apply. **Whether a flag applies is a fact about argv and the
manifest and cannot depend on free RAM**, and the wrong message points the reader at the
machine rather than at their command line. The check now runs after the arch is known and
Muse Glimmer skips it. Proven both directions under real load, at **1–2 GB available**:
`glimmer_flags` 3/3 green where it had been 3/3 red, and V4 and GLM artifacts still refuse
with the OS-reserve message.

That suite called itself deviceless and was in fact load-dependent. **No CI here would ever
have caught it** — there is no rocm job, so nothing runs it at all; it took a contended
machine and an unrelated benchmark. `run_glimmer` now has no budget check, which is correct
only while it bails before allocating: when S3 gives it a pin build, the check belongs at the
top of that build, after the refusals and before the 55.7 GB.

#### The anchor itself. **DONE 2026-08-11. G1b MET.** No GPU — this reference is plain PyTorch.

`docs/measurement/glimmer-reference/anchor.md` is the record; `tests/glimmer_anchor_driver.py`
produced the goldens, `tests/glimmer-anchor.sh` reproduces them, `tests/glimmer_anchor.rs` reads
them with no python, no venv, no network and no device.

**Every fixture this stage asked for exists**, and one the plan did not name: the two weightless
norms, which ship no tensor and are therefore the two a port can omit without the checkpoint
complaining. GQA attend dense and windowed (layers 3 and 7 full against 0-2, 4-6 sliding); ring-KV
append and evict at the window boundary — recorded as a *shape*, since a sliding layer's cache
holds exactly `sliding_window` rows from the first decode step while a full layer's grows 12→18;
the per-layer RoPE table with q and k captured on both sides of the rotation, on rotated layers
only; the output gate and the gated value entering `o_proj` separately, so the gate's POSITION is
pinned and not just its value; the SwiGLU; and one full DFlash draft step.

**G1b's clauses, each met by a recorded run:** 14 defects × 2 weight draws = 28 runs, every one
gated on the captures it must leave bit-identical. A local layer (`full_layers_slide` holds layers
0-2 entirely), a global layer (`rope_on_nope_layers` and `window_off_by_one` both localise onto 3
and 7), a window-boundary crossing (`window_off_by_one`, at window 4 across 18 positions), layer 0
(green in nine of the fourteen), and the last layer (7 here, standing for 51 — chosen so the
`[w,w,w,full]` rule puts a full layer both at the end and not at the end).

**The finding.** `softcap_off` moves 7 of 1103 captures and leaves `emitted.ids` bit-identical. The
argmax-invariance of `T*tanh(x*mult/T)` stops being an argument in §9 of the architecture doc and
becomes a measurement — **every greedy decode gate in this repo is provably blind to a wrong logit
scale on this model**, and nothing but a value-level fixture will ever catch it.

**Two reference behaviours found by running it, neither visible by reading.** The DFlash drafter's
default attention mask is built from `noise_embeds` and is block-wide while K/V span
context+block, so the reference raises; and supplying the correct 2D mask only works with
`use_cache=False`, because the mask builder takes `kv_length` from `past_key_values` and a fresh
`DFlashCache` reports 0. Both are recorded in `anchor.md` and in the driver at the call site. A
port's first draft call hits both.

**Green sets are scoped to step 0, and that is a fact about the model, not caution.** A defect that
shifts the argmax changes the token fed into step 1, so from t1 onward even layer 0 differs for a
reason unrelated to where the defect lives. Only the prefill can localise. The first version of the
matrix declared unscoped green sets and all of them failed at t6 on exactly this.

**Three defects in the harness, found by the harness.** Rope captures were first numbered by CALL
index, which counts only rotated layers — so six rotated layers of eight were labelled L0..L5 and
every NoPE golden was mislabelled; the layer index now comes from the attention module, which is
the only place it is in scope. The weightless-norm tap was a patched class method, so
`qk_norm_off` and `embed_norm_off` deleted their own evidence and died in the tap census instead of
reddening anything; they are forward hooks now, which fire around whatever `forward` currently is.
And `draft_causal` first set `self_attn.is_causal = True` and moved **zero** captures, because
`eager_attention_forward` never reads that flag — causality is entirely in the mask.

**What this does NOT establish.** No tolerances: K3's anchor measured per-operator fp32 rounding
floors and derived thresholds from them, and this one has not, because there is no Glimmer kernel
yet to need them. S2 must measure the floors the same way before choosing any threshold — a
tolerance picked to make a kernel pass is not a tolerance. And no real weights: every number is a
deterministic draw at toy widths.

#### Three things this stage changed outside its own files

- **`src/v4oracle/golden.rs` → `src/golden.rs`.** Its own doc said to move it out of `v4oracle/`
  "if a third model arrives" rather than grow a third magic under a name that says V4. Muse Glimmer
  is that third model. `read_k3` and `read_glimmer` are now one `read_anchor` behind two names.
- **`tests/common/golden_read.rs`.** `float`/`shape_of`/`fnv1a` and the `Vendored` byte-pin were
  written in `k3_anchor.rs`, and the jscpd gate rejected the second copy — which is how it surfaced
  that `f4_loop.rs` had been carrying a **third** all along. All three now share one facade, which
  re-exports `GoldenSet` so the module preambles stop being identical too.
- **`tests/docs.rs` identified docs by BASENAME.** Two ports now ship an `anchor.md`, and the
  duplicate-row check reported the new one as a duplicate of K3's on the day it was added. The
  false positive was the visible half; the dangerous half is that it would have handed one doc the
  OTHER doc's row, so a scope check could pass by reading a cell belonging to a different port. It
  matches on the path under `docs/` now, and was red-proved both ways.

### S2 — kernels. Each item gates before the next.

Order: **ring/dense GQA attend → per-layer RoPE → output gate → wire the existing MLP →
lm_head/logits sizing.**

1. **GQA attend** — the new family. **DONE 2026-08-12**: `gqa_attend` in `kernels/attn.hip`,
   gated by `tests/glimmer_attend.rs` and scored against `tolerance::GLIMMER`'s `attend` row,
   **`Rel(1.64e-4)`, measured before the kernel existed** (`anchor.md` §tolerances). One kernel,
   two row-sources as planned (`ring_cap = 0` is a cache indexed by position, `ring_cap = win`
   is the ring), and the broadcast trap is the thing the gate is built around.

   **The bound is DERIVED, and no `Sel` was borrowed.** The plan said to take V4's descriptor;
   at 131072 context a `[tq][s]` mask array is larger than the model, so the kernel takes
   `(start_pos, win, ring_cap)` and computes `j ∈ [pos - win + 1, pos]` itself. The golden's
   captured mask is then compared **against** that derivation rather than fed to it —
   `the_derived_bound_reproduces_the_captured_mask`.

   **What it measured**, over both goldens, all 8 layers, all 7 steps (112 cases, plus 72 ring
   cases and 6 launcher guards): worst absolute re-association error **6.56e-7**, smallest
   wrong-mapping signal **0.335**. Five decades apart, so `MAX_ABS` is 10× the floor and there
   is no judgement call in it. This is the first Glimmer tolerance and it answers the
   front matter's "no tolerances yet" for this operator only.

   **Two red proofs, run and reverted** (both recorded in the commit and the test's comments):
   `head % hkv` for `head / group` reddens the kernel and ring tests at **1.20** and 1.03 —
   that is trap 10, and it is the one that decodes fluently. `pos - win` for `pos - win + 1`
   reddens the kernel at **1.07** and leaves the mask test **green**, because the mask test
   restates the rule in Rust: trap 14 needs both halves and neither alone is the trap.

   **Two reference behaviours the plan did not name**, both found by running the gate:
   `attend.out` is captured after transformers' own `attn_output.transpose(1, 2)`, so it is
   already `[rows, heads, dim]` while `q`/`k_cache`/`v_cache` are `[heads, rows, dim]` — and at
   6 heads against 12 rows, reading it heads-first does not fail a shape check by accident.
   And `DynamicSlidingWindowLayer` keeps the last `window - 1` rows and returns
   `cat(kept, new)`, so **a sliding layer's decode cache is the WINDOW, not the sequence**; the
   offset is derived from the capture's own shape. A port that assumes the reference stores
   modulo-indexed rows misreads every sliding golden.

   **Still open here.** The fp8/bf16 KV dtype decision is untouched — the kernel takes f32 K/V
   and S0's traffic arithmetic has not been applied to it. The occupancy is deliberate and
   marked: one subgroup per (query row, Q head) is 32 blocks at decode against 40 CUs, with the
   whole KV sweep serialised inside each. `mla_latent_attend`'s two levers — HB head-tiling
   (all 16 Q heads of a group share one KV head, so the LDS tile would be read once per group
   rather than once per head) and split-KV with a combine pass — are the upgrade path, and S5
   is where they get priced rather than guessed.
2. **Per-layer RoPE** — table with θ=500k applied on locals only; the global-layer rule is
   whatever S0 settled, asserted positively (K3's `mla_use_nope` lesson: assert the flag,
   never default it).
3. **Output gate** — K3's gated-MLA kernel pattern at the GQA site.
4. **MLP wiring** — no new kernel; delete the guard, bind the dims.
5. **lm_head at 202048** — existing `gemv_i8`; check `ARGMAX_BYTES`/logit scratch at
   808 KB/row × MAXROW.

**G2 — met when** each kernel passes its S1b fixture and its defect run, in order.

### S3 — layer loop, first decode.

`src/glimmer_gpu.rs` — a module, not a branch in `gpu.rs` (the V4 rationale: no shared
per-layer step; and jscpd panics on clones at zero budget — factor when it fires, do not
pre-design a four-model skeleton). Name every deviation from the reference at its call site.

**G3 — met when** teacher forcing, greedy decode, and incremental-with-KV match the tiny
model at zero tolerance; **a decode crossing position 2048 matches a from-scratch prefill**
(the ring's first eviction is the blind spot); the [L,L,L,G] pattern is proven by a
deliberate pattern-shift defect going red on global layers only.

### S4 — real weights.

Convert the full checkpoint; hand-port the chat template with a byte-level pinning test
(tool framing included — "agentic" model, expect a tools block; GLM's template drifted for
months without one); bounded greedy run **read by a human** (not `distinct`, not
repeated-block); byte accounting from the artifact; determinism across two runs.

**G4 — met when** all of the above hold. This is the first point Glimmer "runs".

### S5 — throughput.

**Precondition (moved here from S0 item 5, 2026-08-10):** measure sustained resident-GEMV
GB/s at Glimmer shapes on gfx1151 and register the predicted band from it *before* the first
decode run. Sole-tenant, under `/var/run/sys-gpu.lock`, with a contention witness sampled per
arm — and the witness must read `mem_info_gtt_used`, not only `/sys/class/kfd/kfd/proc/`,
which is blind to the Vulkan tenants that actually occupy this machine.

Undrafted first: measure against that registered band; miss is explained and recorded or
the gate is not met. Then the int4 lever (`fp8_to_i4` path, group-128 — the int4-scales
lesson says group scales are what made int4 usable) with a paired dNLL gate from `bin/ppl`
on the 5000-token corpus, single format so the A/B is safe. Layer-major prefill: verify it
composes with windowed layers or record why not.

**G5 — met when** throughput is inside the registered band and output is byte-identical to
G4 at the same settings.

### S6 — DFlash drafter. After G5; independently useful.

Spec: `glimmer-architecture.md` §11, first-party-verified 2026-08-10 (checkpoint
`meta-models/Muse-Glimmer-30B-assistant`, 2.556 B, and transformers'
`models/muse_glimmer_assistant/`). **Neither vLLM nor SGLang wires DFlash to Glimmer** — both
implement only the Qwen3 flavour — so there is no serving reference for this pairing.

1. **Drafter artifact.** Second converter target: 5 layers, its own tensor names, its own
   config asserts. It shares almost nothing with the target — 32Q/**8**KV, plain two-norm
   pre-norm layers (no sandwich), **weighted** `q_norm`/`k_norm [128]` where the target's are
   weightless — so nothing may be defaulted from `GlimmerConfig`.
2. **The target must export 5 hidden states per accepted token** (zero-based layers
   1/13/25/37/49), not just the last. That output path does not exist today and costs
   **66,560 B/token**. Build it before the drafter, since G6 cannot start without it.
3. **A bidirectional 16-row attention with two sequence lengths.** Q is the 16 draft rows;
   K/V is `concat(projected_target_context, draft_rows)`. RoPE builds `cos/sin` over the full
   range and Q takes the **tail slice** — off by `ctx_len` is silent quality loss. This is a
   different kernel from S2's causal GQA, not a parameterisation of it.
4. **Embedding must be readable raw.** The drafter embeds without the target's weightless
   embed-norm (§5). If the converter ever folds that norm into the matrix, the drafter is
   unusable — the first-party code carries a comment saying exactly this.
5. `MAXROW` 2→17: re-size every scratch sized off it (`ARGMAX_BYTES`, logits, row-selection
   arrays).
6. Acceptance: longest-common-prefix walk replacing the hardwired `d == t1`
   (`src/gpu.rs:3067`), plus the bonus token. Lossless by construction.
7. Register the drafted band from **measured** acceptance before claiming any multiplier.

**Why this is worth doing here when the equivalent was a loss on GLM.** A dense verify reads
every weight once regardless of row count, so per cycle the cost is one target read plus one
drafter read: `speedup ≈ N × 26.51/(26.51+2.56)` = `N × 0.91` at fp8, **break-even at
N > 1.1**. GLM's 0.93× ungated `--mtp` was the *MoE union* — 2 rows costing 1.61× the experts
— and that penalty does not exist without experts. Do not carry the `--mtp-min-conf` gate
over on cargo: its whole purpose was to avoid paying the union on a low-confidence draft, and
at break-even 1.1 there is little left to gate. Keep the gate-after-draft *scoring* so the
acceptance histogram never goes blind; re-derive whether any gate is warranted from measured
τ. Published figures are 3.1× (RTX 5090, llama.cpp, quantized both sides) and τ≈6.5 / 4.9×
(H200, Qwen3-4B) — **neither is this hardware nor this pairing**, and the paper shows
acceptance decaying past the drafter's ~4K training window, so a band registered on short
prompts will not hold at 131k.

**G6 — met when** drafted output is byte-identical to undrafted greedy at the same prompt,
acceptance and tok/s are recorded, and the gate at zero acceptance degrades to exactly
undrafted speed (prove it by feeding a shuffled drafter).

**Out of scope, recorded:** the perception encoder and multimodal serving. `serve.rs`'s
image-refusal reminder stays correct for Glimmer and must name the right reason. A vision
stage would be its own plan; nothing above depends on it.

## Standing rules

`CLAUDE.md` § "Measurement discipline" and § "Build and test" in full: dev profile for
development, flock + contention witness per arm, no `cargo build` between arms, instruments
behind a feature and a flag, jscpd at zero budget, the feature union for anything touching
`telemetry.rs`/`eval.rs`/`gpu.rs`, and no GPU CI — the gates above are the whole safety net.
Name kernels and types for what they do (`gqa_ring_attend`, not `glimmer_attend`); the
`Arch` variant is the only place the model's name belongs.
