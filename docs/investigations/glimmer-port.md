---
scope: glimmer
status: live
verdict: The implementation plan for Muse Glimmer-30B, the fourth model, sequenced after K3. A DENSE 52-layer port that bypasses the streaming machinery entirely — ~25 GB/token fully resident at fp8, so the ceiling is GTT bandwidth, not NVMe. Reuse is high everywhere except attention: GQA 32Q/2KV + sliding-window locals + per-head output gate is a new kernel family (rivoli is MLA-only), per-layer RoPE-on/off is new plumbing, and the DFlash block-16 drafter generalises MAXROW=2 to 17 — cheap here precisely because the model is dense, so a 17-row verify costs one weight read. S0 NOT STARTED: every §1 number is model-card provenance through a summarizing fetch, unverified — the exact trap K3's G0 was reopened for twice.
---

# Muse Glimmer-30B — implementation plan

**Glimmer is a capability this engine must have.** This is the plan to get there, not an
assessment of whether to. It is sequenced **after Kimi-K3** and builds on the `wt/k3-s1a`
lineage (V4 + K3): `Arch`/`ArchConfig`, the streaming `SafeWriter`, per-arch converters and
refusal-tested config parsing are assumed present, and line references marked `[wt]` are to
that branch, not `main`.

Target: `meta-models/Muse-Glimmer-30B`, **text-only** (the 1.8 B perception encoder is out of
scope — §S6). 52 layers, ~29.6 B total incl. vision, ~26.5 B on the text decode path.
Apache 2.0, BF16 safetensors on HF, plus a separately-released 5-layer DFlash drafter.

## STATE

- **S0 not started.** Every number in §1 was read 2026-08-10 from the announcement, the HF
  model card and the DFlash abstract (arXiv 2602.06036) **through a summarizing fetch layer**.
  K3's G0 was reopened twice for exactly this class of provenance. Nothing here is
  implementation-grade until S0 pins `config.json`, the tokenizer files and the raw paper.
- **The design is inverted relative to every model this engine runs.** Glimmer is dense.
  At fp8 the whole text model is ~26 GB — resident with ~80 GB to spare. The NVMe expert
  streaming, byte arena, residency cache and Belady work are **bypassed, not ported**: a
  Glimmer layer is 52 instances of the no-pool/no-ticket path GLM's 3 dense layers already
  take (`src/gpu.rs:2043` — "attention + MLP were all launches, nothing blocked"). §2.
- **The one new kernel family is attention.** rivoli is MLA-with-q-LoRA and nothing else —
  no GQA, no MHA, no per-layer RoPE, no trained sliding window (`[wt]
  docs/investigations/other-models.md` §"The attention half is not [reusable]"). Glimmer
  needs GQA 32Q/2KV with a 2048-token ring KV on 3 of every 4 layers and full KV on the
  fourth, a per-head sigmoid output gate, and RoPE on local layers only. §3, §S2.
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

## 1. Ground truth as currently believed — ALL UNVERIFIED

Read 2026-08-10: announcement (`research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model`),
model card (`huggingface.co/meta-models/Muse-Glimmer-30B`), DFlash abstract
(`arxiv.org/abs/2602.06036`). **Each row below is an S0 item: confirm against raw
`config.json` / tokenizer files / paper PDF, record the answer and its source, or correct it.**

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

Per-token weight traffic, undrafted, all-resident (arithmetic from §1's unverified shapes;
carried to S0 as a template, not a prediction):

- fp8: ~23.8 GB layers + ~1.34 GB int8 lm_head ≈ **~25 GB/token**
- int4 (group-128, `fp8_to_i4` pipeline exists): ≈ **~13 GB/token**
- KV reads at 131k context: 13 global layers × ctx × 1 KB ≈ 1.7 GB/token — *not* negligible
  at full context; bounded 82 MB total for all 39 locals.

Ceiling arithmetic: at the LPDDR5X theoretical ~256 GB/s, fp8 undrafted ≤ ~10 tok/s. The
*achieved* number is S0 item 8; register the predicted band from it and hold S5 to that band.
DFlash multiplies this by (accepted tokens per verify pass); the card claims 3.1× on an
RTX 5090 — treat as vendor-optimistic until G6 measures acceptance here.

## 5. Memory budget — not binding, assert it anyway

fp8 text model ~26 GB + KV ~1.8 GB + drafter (~1 GB?) + scratch ≪ `--max-mem 115`. The
budget check stays (a wrong config must still refuse), but unlike K3 nothing here is
capacity-planned. tmpfs and contention discipline per CLAUDE.md still apply.

## Stages. K3's gate model (`k3-port.md` §G) binds: met or not met, prove each gate can go red.

### S0 — ground truth. No code, no weights.

1. Pin `config.json` + tokenizer files + safetensors index from HF; resolve every §1 row.
2. Read the DFlash paper (not the abstract): drafter interface to the target (which context
   features, injected where), draft step arithmetic, acceptance rule.
3. Settle NoPE-vs-scored on global layers, gate form/placement, window semantics (sink
   tokens? inclusive boundary?), context-scaling scheme, `tie_word_embeddings`.
4. Extract `docs/reference/glimmer-architecture.md` from first-party modeling code,
   provenance-marked like `k3-architecture.md` (raw source, line-cited; no summarizing
   fetches).
5. Measure sustained resident-GEMV GB/s at Glimmer shapes on this part (benchmark kernels,
   flock + witness). Register the §4 band from it.
6. Layer-0 blind spot check: which layer is *least* representative here? (First global
   layer, layer 3 if the pattern starts local — the gates of §G rule 2 must cover a local
   layer, a global layer, a window-boundary crossing, layer 0 and layer 51.)

**G0 — met when** every §1 row has a recorded answer and a first-party source, the reference
doc exists, and the predicted band is registered from a measured number.

### S1a — artifact. No GPU.

1. `Arch::MuseGlimmer` arm + recogniser test (both directions); `GlimmerConfig: ArchConfig`
   with **no defaulted fields** — `head_dim`, `num_key_value_heads`, the layer-pattern
   descriptor and window size are asserted, not inferred (`hidden/n_heads` ≠ 128 here; any
   code that still assumes that identity must fail loudly at parse, not at decode).
2. `src/bin/convert_glimmer.rs`: BF16 → fp8-block resident artifact via streaming
   `SafeWriter`; skip `vision_tower`/projector explicitly; copy tokenizer +
   `generation_config`. No expert files; `FormatMeta` gains its nullable VQ section.
3. `GlimmerPin`: own type (per the V4/K3 precedent — shares nothing with the pool); int8
   embed/lm_head placement reused; its own tensor-name table (V4's §7 records what
   assuming HF-style naming cost).
4. `run_glimmer` dispatch with hand-written refusals for every flag that doesn't apply
   (`--cache-policy`, `--mode`, `--attn dsa/misa`, `--hint-*` descendants) — V4's bespoke
   bails are the template; omitting them compiles clean and silently accepts.

**G1a — met when** conversion round-trips bit-exact on sampled tensors; GLM/V4/K3 artifacts
still open byte-identically (test opens them); a config missing or contradicting any
load-bearing field refuses at startup, proven by feeding it one; resident byte accounting
reproduced from the artifact.

### S1b — gate harness. No GPU. Must not touch kernels.

Tiny-model goldens emitted from the **first-party HF modeling code** at tiny dims (the
anchor rule from K3 G1b: a reference attesting to itself is not independence — here the
first-party stack is plain `transformers`, so the anchor is cheap). Per-operator fixtures
for: GQA attend (dense + windowed), ring-KV append/evict at the 2048 boundary, per-layer
RoPE table, output gate, SwiGLU at 6656/19968, and the DFlash draft step.

**G1b — met when** every golden has a recorded defect run reddening exactly where the defect
touches, covering: a local layer, a global layer, a window-boundary crossing, layer 0,
layer 51.

### S2 — kernels. Each item gates before the next.

Order: **ring/dense GQA attend → per-layer RoPE → output gate → wire the existing MLP →
lm_head/logits sizing.**

1. **GQA attend** — the new family. One kernel, two row-sources: full cache (global) or
   ring window (local). KV is 2 heads × 128; 16 Q heads share each KV head — the broadcast
   is the correctness trap (a transposed group mapping decodes fluently). Borrow V4's `Sel`
   descriptor and ring bookkeeping; the fp8/bf16 KV dtype decision comes from S0's traffic
   arithmetic.
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

Undrafted first: measure against the S0-registered band; miss is explained and recorded or
the gate is not met. Then the int4 lever (`fp8_to_i4` path, group-128 — the int4-scales
lesson says group scales are what made int4 usable) with a paired dNLL gate from `bin/ppl`
on the 5000-token corpus, single format so the A/B is safe. Layer-major prefill: verify it
composes with windowed layers or record why not.

**G5 — met when** throughput is inside the registered band and output is byte-identical to
G4 at the same settings.

### S6 — DFlash drafter. After G5; independently useful.

1. Drafter artifact: second converter target (5 layers, own tensor names, own config
   asserts).
2. `MAXROW` 2→17: re-size every scratch sized off it (`ARGMAX_BYTES`, logits, row-selection
   arrays); the verify already reads weights once for N rows — dense makes this near-free.
3. Acceptance: longest-common-prefix walk replacing the hardwired `d == t1` (`src/gpu.rs:3067`);
   per-block confidence gating **re-derived, not copied** from `--mtp-min-conf` (the 1.108×
   economics were MoE-union economics; keep the gate-after-draft scoring so the histogram
   never goes blind).
4. Register the drafted band from measured acceptance before claiming any multiplier.

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
