---
status: closed-shipped
scope: inkling
verdict: S0-S8 COMPLETE (MINI BF16, MINI MXFP8, AND SMALL MXFP8 WITNESSED ON HOT NVME END TO END; `tests/smoke-inkling-small.sh` 17/17 GREEN + RED-PROOFED) -- the Inkling 2 port (owner scope 2026-11-23 both members, Mini first, text-only v1, MTP refused) has both members built, audited, gated, and benchmarked on hot NVMe. Mini bf16 (`inkling2_loop`, 48.6 GiB tier): 8/8 greedy token-identical to first-party transformers on "Hi"; witnessed release 64-token decode on "The sky is blue because" (GTT 17-18 MiB pre-arm, empty KFD witness) 11.69 / 11.59 tok/s pinned (90 GiB, 24/24 layers, 85.6 ms/tok) and 2.25 tok/s streaming (20 GiB, 8/24 pinned), 64/64 ids identical A=A'=B; TF `--ppl` on TOUR.md (830 tok, PPL 9.6174) byte-identical A=A'=B; `tests/smoke-inkling.sh` 16/16 green + red-proofed. Mini MXFP8 (`convert_inkling --mxfp8`, 27 GB on disk, 3.0 GiB resident + 23.6 GiB pool, layers 20-23 repacked bit-exact and 0-19 quantized NVFP4->MXFP8, sniffed automatically via `has_mxfp8_source`): `inkling_mini_mxfp8.rs` 3/3 GPU green; **8/8 token-identical** to first-party transformers on "Hi"; witnessed release 64-token decode **14.75 / 14.30 tok/s** at 64 GiB (100% experts fit, 67.8 ms/tok, +26% vs bf16 at half the memory; 14.13 tok/s to EOS at 130 tok) and **12.60 tok/s** at 12 GiB streaming (37.2% experts pinned, 64/64 ids identical A=A'=B); paired dNLL vs Mini bf16 on TOUR.md (830 tok) mean `-0.00415` nats, 95% CI `[-0.01257, +0.00427]` (upper bound `< 0.00995` 1% bar, worse% `48.6%`, PPL `9.5776` vs `9.6174`). Small MXFP8 (`inkling2`, 564 GiB routed set on `/var/db/rivoli/inkling-2-small-mxfp8-art`, 996/996 resident match, L00/L47 0 bytes differ): fused `moe_gateup_mxfp8_situ` + `moe_down_mxfp8_weighted` kernels (79/79/0 census, 64 callers, 15 uncalled, 13 oracle-validated; 6/6 GPU oracle + 4/4 composition + §22h red-proofed); witnessed release 64-token decode **2.67 / 2.66 tok/s** at 100 GiB (15.5% pinned, 76.6% hit, 374.0 ms/tok) and **1.37 tok/s** at 32 GiB (3.4% pinned, 35.6% hit, 731.8 ms/tok), **64/64 ids identical** A=A'=B. The Small serve cell closed 2026-11-25 (`SMOKE GREEN: 17 cells`, exit 0): non-stream reply `'Hi there'`, `enable_thinking=false` content `'Hi'`, SSE frames + `[DONE]`, clean shutdown.
---

# Inkling 2: port plan of record

## Why this file exists

Inkling 2 (Thinking Machines Lab, 2026-11) is the fifth model family in scope. Its text
stack is unlike the four on record: looped decoder layers (Mini applies 12 stored layers
three times), block-attention-residual query banks instead of plain residuals, short
convolutions on every k/v path, attention sinks, softcapped SwiGLU, and TWO expert quant
ABIs in one checkpoint (NVFP4 layers 0–19, MXFP8 layers 20–23). The math extraction and
the checkpoint→HF name mapping live in
[`docs/reference/inkling2-architecture.md`](../reference/inkling2-architecture.md); this
file carries the plan, the pins, and the census answers. The Qwen port
([qwen-flash-next-port.md](qwen-flash-next-port.md)) is the refusing-first precedent:
recognition + legality + doors land RED before any engine code exists.

## Owner scope decisions (2026-11-23)

1. **Both members, Mini first.** One port plan for the shared `inkling2` text stack;
   Mini is the iteration vehicle (19.5 GB, fits), Small becomes the P1
   bigger-than-memory streaming case (~600 GB MXFP8 source).
2. **Text-only v1.** The vision tower's weights are pinned and counted but EXCLUDED by
   name from the converter; image input is refused loudly at the door.
3. **MTP/DSpark deferred.** `mtp.safetensors` is downloaded and hash-verified but never
   opened by the converter; `Flag::Mtp` refuses with its own const.
4. **Small source = `Inkling-2-Small-MXFP8`**, not an NVFP4 Small: transformers' Small
   path keeps MXFP8 experts quantized and its quant config declares MXFP8 for all 48
   layers, zero NVFP4. The first-party-anchor rule selects this checkpoint. NVFP4
   machinery is still built for Mini.

## Stage legend

| stage | what | state |
|---|---|---|
| S0 | identity: tokenizer TYPE, pins vendored, Mini weights verified, math + name mapping extracted, this plan registered | **built 2026-11-23** |
| S1 | refusing-first landing: Arch variants + sniff + config types + legality rows + doors + golden magics + pin gate, each refusal red-proofed, deviceless battery green | **built 2026-11-23** |
| S2 | Mini artifact converter: exact consumed/emitted counts, exclusions BY NAME (vision families, `mtp.*`), NVFP4/MXFP8 dequant per the first-party oracle, w13 de-interleave | **built 2026-11-24** |
| S3 | anchors: transformers `main` @ `d8f10a82` + torch CPU venv; tiny-config fp32 goldens from the first-party text model, the decode pair being the model's own CACHED step with every loop pass captured; defect matrix off the traps table; carved fp32 tolerance floors on ≥2 draws (anchor.md) | **built 2026-11-24** |
| S4 | kernels: twelve ORACLE_VALIDATED Rust kernels (clips ×3, dequants ×2, sconv, sconv_dynamic, sink_attend, message_gate, vaxpy, kv_scatter, rel_bias_gather), no C twin (owner 2026-11-24); MoE = `situ_glu_f32` reuse, RMSNorm = `rmsnorm_single` reuse; census both ends (78/63/15/12); red proofs §22a–g | **built 2026-11-24** |
| S5 | Mini engine arm (looped decode, dynamic sconv, block residual, MessageGate, MoE, P6 residency partition, tiktoken/chat/serve, TF `--ppl`, `inkling_engine_anchor.rs`, `tests/smoke-inkling.sh` green + red-proofed, 8/8 first-party parity, P4 16/16 invariance, S5 ponytail+correctness audit closed) | **built 2026-11-25** |
| S6 | Small streaming arm (`RoutedFmt::MxFp8`, `convert_inkling_small` verified 0 bytes differ on real 12.61 GB L00, fused `moe_expert_range_mxfp8_situ` kernels 6/6 GPU green, two-stream `RoutedPool` in `InklingEngine::open_mxfp8`, legality/CLI/serve cutover, `inkling_small_mxfp8.rs` 4/4 GPU green, `tests/smoke-inkling-small.sh`, S6 audit closed) | **built 2026-11-25** |
| S7 | witnessed release decode + P4 invariance on hot NVMe for BOTH Mini (11.69 / 11.59 tok/s pinned, 2.25 tok/s streamed, P4 64/64 ids + 830/830 NLL bytes identical) AND Small (2.67 / 2.66 tok/s at 100 GiB, 1.37 tok/s at 32 GiB, P4 64/64 ids identical, clean KFD/GTT witness), all 11 Inkling GPU suites (56/56 green) + §22h red proofs, full 48-layer Small artifact verified on hot NVMe; Small `smoke-inkling-small.sh` **17/17 green** (serve cell included) | **measured 2026-11-25** |
| S8 | closeout: verdicts corrected in place, §22h red proofs + hot-NVMe Small timings recorded, census 1–8 answered, deviceless + CodeScene 10/10 + jscpd 0-clone battery green; `tests/smoke-inkling-small.sh` **17/17 green** with the serve cell (`SMOKE GREEN: 17 cells`, exit 0), flipped `closed-shipped` | **closed 2026-11-25** |

S2-before-S3 is deliberate and does NOT violate the TDD ordering: the converter is
scored by the census (exact counts from the derived TSV) and by the FIRST-PARTY dequant
oracle (`core_model_loading.py:772–820`, already extracted), both of which exist before
the converter is written. The model goldens (S3) score the KERNELS (S4), not the
converter's byte layout — and because transformers converts the original checkpoint on
load, S3 needs no rivoli converter at all. The two orderings are decoupled; the converter
additionally gets a byte-level cross-check against the loader's dequanted experts.

## Identity and pins

Vendored under `docs/measurement/inkling2-reference/` (bytes + sha256 + FNV-1a64 in the
TSV header and the pin gate; FNV-1a is `crates/core/src/hash.rs::fnv1a`, no sha256
dependency in the gate):

| pin | revision | note |
|---|---|---|
| `thinkingmachines/Inkling-2-Mini` | `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` | lastModified 2026-11-23T08:08:49Z |
| `thinkingmachines/Inkling-2-Small-MXFP8` | `14f2cb29a5d0f4c1fda8f645fa339e936c2147a2` | lastModified 2026-11-20T14:10:39Z |
| `huggingface/transformers` | `d8f10a82213a3a9db48385aba3a4b57ab535de50` (`main`) | inkling2 merged via PR #51050, `046bdb8cc2ce` 2026-11-21 |
| `tiktoken.model` | 3,386,494 B, sha256 `f170372d…6ec0d8` | ONE copy; byte-identical between members |
| Mini shard | `model-00001-of-00001.safetensors` 19,526,019,168 B, sha256 `65e60f51…3539d1` | on `/swarm/storage/ai/rivoli/inkling-2-mini-src/`, recomputed and matched to the HF LFS oid |
| Mini MTP | `mtp.safetensors` 635,801,416 B, sha256 `7c3b6513…10a1d2` | verified, DEFERRED, never opened |

`docs/measurement/inkling2-reference/mini/tensor-families.tsv` is DERIVED from the live
shard header by script: 85 homogeneous (family, dtype, shape) rows; **1244 tensors** =
822 TEXT (18,889,358,496 B) + 422 EXCLUDED-VISION (636,512,256 B); closure asserts
header+payload = file size exactly. The converter's S2 counts come from this table, not
from memory.

**Name-mapping resolution (2026-11-23):** the blog's literal
`convert_inkling2_weights_to_hf.py` link 404s at every ref — no standalone script was
ever merged. What merged is the on-load conversion: `conversion_mapping.py`
(`inkling2` :301, `inkling2_text` :353, loop built programmatically :2654–2707) +
`core_model_loading.py` (`Interleave` :232, `Inkling2LoopDequantizeExperts` :772,
`Inkling2LoopMergeMessageQueries` :823, `Inkling2UnpadVocab` :856,
`Inkling2IndexOffsetWeightRename` :1266). The full rename table and dequant semantics
are in the math doc §5–6.

## S0 registration red proof (2026-11-23)

The docs gate was run with both docs written but SCOPES/INDEX unregistered:
`every_doc_declares_a_status_and_a_verdict` reddened on `scope 'inkling' is not one of
[glm, v4, k3, engine, glimmer, qwen]` for BOTH docs, and
`the_index_lists_every_doc_with_a_matching_verdict` reddened on both docs missing from
the INDEX (exit 101; log `/var/cache/rivoli/scratch/inkling-docs-red.log`). Adding
`"inkling"` to SCOPES and the two INDEX rows turned it green, 4/4
(`inkling-docs-green.log`). The gate demonstrably refuses an unregistered port.

## S1 landing and red proofs (2026-11-23)

The refusing-first surface is built and the deviceless battery is green:
`cargo test --workspace --no-default-features`, `cargo build -p rivoli --no-default-features`
(the jscpd unit) and `cargo clippy --workspace --all-targets --no-default-features` all exit 0
(`inkling-s1-test.log`, `inkling-s1-build.log`, `inkling-s1-clippy.log` under
`/var/cache/rivoli/scratch/`). `inkling_config.rs` scores CodeScene 10.0. The new gates ran
non-vacuously: the sniff test, the schema test, all four pin-gate cells (including the live
shard-header cell, which read the 19.5 GB shard's header on `/swarm`), and the legality
refusal test.

Two jscpd/CodeScene falls forced structure rather than exemptions, both argued in place:
the two wrapper `validate` bodies share one `validate_member` helper (a cloned
`validate_root`-call pair would be a 5-line clone), and that helper takes FOUR arguments —
the root pair bundled as one tuple, the expected spelling derived from `arch` inside — because
the six-argument form scored 9.68 ("Excess Number of Function Arguments", Rust threshold 4).
Mini's root-dtype `ensure!` is one line, not the five-line block `glimmer_config.rs:227` writes
for the same bfloat16 rule: jscpd normalizes the message literal, so the block form clones it.

Each S1 gate was shown red, then reverted (logs `inkling-red-{a,b,c,d}-*.log`):

| gate | plant | observed red |
|---|---|---|
| (a) legality | `inkling2`'s `--ctx` cell flipped to `Outcome::Support` | `inkling2_rows_refuse_every_flag_as_of_s1` exit 101: `--ctx` left `Support`, right `Refuse(INKLING_ARM_NOT_BUILT)` |
| (b) sniff | `inkling2_loop_text` admitted in `from_manifest_str` | `inkling_accepts_its_root_spellings_and_refuses_the_nested_ones` exit 101: left `Some(Inkling2Loop)`, right `None` |
| (c) config | block-end complement predicate inverted (`!= 0` → `== 0`) | schema test exit 101: Mini `local_layer_ids` 20 entries vs derived 4 |
| (d) pin gate | recorded `mini/config.json` fnv1a64 last digit `f`→`0` | `the_vendored_files_hash_as_the_header_records` exit 101: live `5279eda68ce19c6f` ≠ header `…60` |

The docs-registration red proof is S0's (above); SCOPES already carries `inkling` and the
INDEX row tracks this verdict.

## S2 landing and red proofs (2026-11-24)

The Mini converter is built and the deviceless battery is green (test/build/clippy exit 0,
CodeScene 10.0 on every new file). Three pieces, each gated:

- **`census::inkling`** (`crates/artifact/src/census/inkling.rs`): classifies every shard name by
  (pattern, dtype, shape) — the key is NOT the pattern alone, because `attn_queries.{I}` and the
  expert weights split into two rows by storage format. `confront` checks every family's count
  both ends and each tensor's byte length; `tests/inkling_census.rs` scores it on the live shard
  (1244 names → 822 TEXT / 422 vision) and deviceless (a full synthetic expansion passes; a dropped
  tensor and an unknown name both redden).
- **`inkling_dequant`** (`crates/artifact/src/inkling_dequant.rs`): the NVFP4 (e2m1 low-nibble-first,
  e4m3 block-16 scale, per-expert amax/denominator) and MXFP8 (e4m3 × 2^(u8−127), row-wise [1,32])
  dequant, w13 de-interleave, and (q,k,v,mlp) query merge — transcribed from LOAD:772–853.
  `tests/inkling_dequant.rs` bit-compares the bf16 output (fnv1a64 + 64 spot values) against
  `dequant-golden.json`, vendored from the ACTUAL transformers ops run on real experts (torch
  2.15.0+cpu, transformers @ d8f10a82, venv at `/var/cache/rivoli/scratch/inkling-venv`). The
  multiply association matches LOAD:813–814 exactly so the bf16 rounding agrees bit-for-bit.
- **`convert_inkling`** (`crates/cli/src/bin/convert_inkling.rs`): consumed 822 TEXT tensors,
  excluded 422 vision by name, never opened `mtp.safetensors`, and emitted 693 tensors (176 expert
  tensors fused to 48 projections, 50 query tensors to 49, unembed unpadded to 200058 rows). The
  run (`inkling-s2-convert.log`, ~19 min dev profile) wrote a 52.1 GB bf16 artifact to
  `/var/cache/rivoli/scratch/inkling-mini-artifact`. `tests/inkling_convert.rs` compares the
  artifact's names+shapes to `text-model-names.tsv` — the first-party `Inkling2LoopForCausalLM`
  state dict (meta device), 693 params / 47 patterns — both ends.

The 8-column census contract is now shared between qwen and inkling via `census.rs` (jscpd).

| gate | plant | observed red |
|---|---|---|
| (e) census | `embed.weight` row count 1→2 (total kept consistent so load passes) | live confront exit 101: observed 1, census records 2 |
| (f) dequant | NVFP4 nibble order swapped (high first, trap T2) | deviceless cells red (1.0≠0.5) AND first-party cross-check red (nvfp4_w13 bf16 fnv) |
| (g) convert | `text-model-names.tsv` lm_head rows 200058→200059 | both cells exit 101: census shape and artifact mismatch |

## S3 landing and red proofs (2026-11-24)

The first-party anchor is built; the full record (stack pins, tiny config, goldens, defect
matrix, tolerance floors, vacuous plants) is `docs/measurement/inkling2-reference/anchor.md`.
`crates/oracles/tests/inkling-anchor.sh` regenerates it (venv torch 2.15.0+cpu + transformers @
`d8f10a82`); `inkling_anchor_driver.py` runs the reference; `inkling_anchor.rs` is the deviceless
gate (7 cells). Three goldens vendored byte-pinned and reproduced 3 of 3; a five-trap defect
matrix reddens on both salts with declared first-touched buckets; tolerance floors carved
fp32-vs-fp64 (the reference is fp32-locked, so a whole-model fp64 run shows zero diff).
**Correction 2026-11-24 (review):** `--mode decode` was a plain 17-token forward wearing the name —
it is now the model's own CACHED step (prefill `0..16` with `use_cache=True` and NO hooks, then ONE
token stepped against that cache, so the conv-state shift, the KV append and the passes'
`cache_offset` all execute), the loop layers `6`/`11`/`17` store all three of their passes, and
both clip floors — previously `0`, because no captured vector reached `threshold` — are carved on
an input scaled into the clamp branch. A golden now holds 16 float tensors, regenerated and
reproduced 3 of 3 by `inkling-anchor.sh` (pins and 7-operator floors in `anchor.md`).

| gate | plant | observed red |
|---|---|---|
| (h) anchor byte pin | last byte of `inkling-anchor-decode-inkling-1.bin` XOR 0xFF | `the_vendored_goldens_match_their_byte_pins` exit 101: FNV-1a mismatch |

The defect matrix's own red is every row's `--compare`; two plants (L2Clip epsilon, router
bias-in-weights) were measured vacuous at tiny widths / random init and replaced — recorded in
anchor.md.

## S4 kernel plan (reuse measured, 2026-11-24)

The CPU operator references are `crates/oracles/src/inkling.rs` (6 tests, each encoding its
trap). **Owner direction 2026-11-24: no C versions of the kernels.** Inkling's new kernels are
Rust-only, validated against the first-party CPU oracle and the S3 anchor through an oracle
suite — NOT the C→Rust bit-identity harness (which scores the legacy ports against tag
`archive/hip-kernels`). The census gained an `ORACLE_VALIDATED` category (`kernel_coverage.rs`):
a waived kernel still owes a `=>` launchers row and an oracle suite that cites it, checked both
ends; it is waived only the C-twin gate row.

**Measured reuses (no new kernel):**
- MoE activation = `situ_glu_f32` (SoftcappedSwiGLU is SiTU-GLU; bit-identical, math doc §MoE).
- RMSNorm = `rmsnorm_single` (T5-style weight-after-rsqrt; `weight·(x·inv)` commutes to the
  kernel's `(x·inv)·weight`).

**Genuinely new kernels (Rust + oracle suite, no C twin):**
- L2Clip / ScaledL2Clip / LogScaledL2Clip (the q/k/rel norm clips; T4/T9).
- Sink attention (softmax joins a sink logit then DROPS it; T10) — **single-query decode step
  BUILT, then BATCHED** (below): one launch over `heads` blocks with the GQA head→kv-head map, an
  on-device sink logit and a parallel block softmax; the rel-bias fold still arrives as the `bias`
  array `rel_bias_gather` fills.
- Relative-position bias (`rel_logits_proj` + the rel L2Clip).
- Short convolution (inkling's `(conv+residual)/√(k+1)`, no SiLU — distinct from K3's
  `short_conv_silu_f32`; T15). **Static decode step BUILT** (below); the dynamic variant follows.
- Loop message gates (MessageGate down/gain/forget; the per-pass recurrence; T8/T16). **Pointwise
  combine BUILT** (below); the down/forget/gain GEMVs and the per-pass recurrence are S5.
- Block attention residual (query-bank attend; cf. K3 `attn_res`). **Reuse, not a new kernel** (below).
- NVFP4 / MXFP8 expert dequant (the S2 `inkling_dequant` oracle is the CPU reference). **BOTH
  BUILT** (below).

Each lands as a complete Rust unit (kernel + kernarg + handle + launcher + oracle suite); the
census refuses a kernel missing any of handle / launcher / oracle. **The clip family is BUILT
and GPU-verified (all three)**: `l2_clip`, `scaled_l2_clip` (sliding-layer q/k norm, learned
scale first), and `log_scaled_l2_clip` (full-layer q/k norm, scaled clip + LogScale τ),
`kernel_inkling_l2clip.rs`, 8 tests against `rivoli_oracles::inkling::{l2_clip, scaled_l2_clip,
log_scaled_l2_clip}`, flock + clean witness — the oracle match (one loop over both τ arms),
clip-down-only, aliasing (its own in-place launch), unit-scale and unit-τ reductions, and the
1001/1006 guards. The host launchers share a `ClipGuard` trait so each wrapper is one line
(CodeScene 10.0); the suite's `clip_launch` dispatches all three through one four-arg helper.
Red proofs: a bogus `ORACLE_VALIDATED` row reddens the census (i); removing the clip clamp
reddens the GPU oracle (j). The attention/conv/loop/dequant kernels follow on the same path.
**Correction 2026-11-24 (S5 wiring):** the clips are BATCHED — each kernel takes `rows` blocks,
one block per row/head, and block `bid()` selects its row at offset `r·n`, so a prefill's q/k/rel
norms are ONE launch rather than one per token; the kernargs and their sizes are unchanged
(`L2Clip` 24 B, `ScaledL2Clip` 32 B with its `[rows, n]` scale, `LogScaledL2Clip` 40 B), and each
launcher gained a leading `rows: Rows`.

**The MXFP8 decode is BUILT and GPU-verified**: `mxfp8_dequant` (one e4m3 value per byte, one
power-of-two scale per 32 values along the input dim, ROW-WISE `[1,32]`, T1),
`kernel_inkling_dequant.rs`, 3 tests BIT-FOR-BIT against `rivoli_artifact::inkling_dequant::Mxfp8`
(the decode is elementwise, so the fp8 LUT + `ldexp(1,e)` match `e4m3_to_f32` + `powi` exactly),
flock + clean witness — the bitwise oracle over four expert/row/width grids, the T1 raw-exponent
trap, and the 1001 geometry guards. Red proof (k, `gate-red-proofs.md` §22a): the scale bias
`−127`→`−128` reddens the oracle and the T1 trap while the guard holds. The host geometry carries
the fp8 LUT's LDS (`GEMV_FP8_LDS_BYTES`); the kernel reuses `gpu::fp8_lut`.

**The NVFP4 decode is BUILT and GPU-verified**: `nvfp4_dequant` (two E2M1 codes per byte, the LOW
nibble the even element, one e4m3 scale per 16 values, per-expert factor `amax[e]/denom`; T2/T3),
3 tests BIT-FOR-BIT against `rivoli_artifact::inkling_dequant::Nvfp4` — the bitwise oracle over
four grids (pinning the `(code·scale)·factor` association, LOAD:813), the T2 low-nibble-first trap,
and the 1001/1006 geometry+denom guards. Red proof (`gate-red-proofs.md` §22b): swapping the
nibbles reddens the oracle and the T2 trap while MXFP8 and the guards hold. The suite builds
every case through `Nvfp4Case::new` (denom fixed at the shipped 448) and maps the case's fields
into the CPU `Nvfp4` oracle so the device case and the reference cannot drift; the host geometry
shares `elementwise_fp8_lds` with MXFP8.

**The static sconv decode step is BUILT and GPU-verified**: `sconv` (a depthwise causal k-tap
cross-correlation of the window `[past k−1, current]` plus the residual, divided by `√(k+1)`; T15),
`kernel_inkling_sconv.rs`, 3 tests against `rivoli_oracles::inkling::sconv` — the oracle over five
channel/tap grids (RELATIVE, the k-tap dot is a short reduction that may contract to FMA), the T15
`√5` divisor trap, and the 1001/1006 geometry+divisor guards. Red proof (`gate-red-proofs.md`
§22c): dropping the residual reddens the oracle and the T15 trap while the guard holds. The kernel
is one thread per channel, no LDS. **Correction 2026-11-24:** the state SHIFT moved into the kernel
(it takes the running `state [channels, k−1]` plus `current` and advances the state in place, so the
layer loop keeps no window copy), and the dynamic variant is resolved in engine `pin.rs`'s
`LayerSconv::Dynamic` (a static tap bank plus a `weight_proj` per side, divisor `√(2k+1)`) rather
than left to S5.

**The single-query sink attention is BUILT and GPU-verified**: `sink_attend` (one block stripes the
`n` QK dots into LDS logits, thread 0 runs the max-subtracted softmax over `[logits, sink]` and
drops the sink column, then the threads stripe the `d` weight·V outputs; T10),
`kernel_inkling_sink_attend.rs`, 3 tests against `rivoli_oracles::inkling::sink_attend` — the oracle
over five key/head grids (RELATIVE, the dots and the AV sum are reductions), the T10
sink-absorbs-mass trap, and the 1001/1006 guards including the LDS budget — now
`(n + one reduction block)·4 ≤ 65536`, because the parallel block softmax keeps a reduction scratch
beside the logits. Red proof
(`gate-red-proofs.md` §22d): dropping the sink from the denominator reddens the oracle and the T10
trap while the guard holds. **Correction 2026-11-24 (S5 wiring):** the GQA repeat and the
multi-query batch are INSIDE the kernel now — one launch covers `heads` blocks, each block maps its
head to a kv head and computes one query row's sink softmax over a shared LDS logit block, and the
per-head sink logit arrives as a `*const f32` read ON DEVICE, so the engine never reads it back and
the layer loop carries no `device_sync`. The rel-bias fold still arrives as the `bias` array, which
the engine gathers with `rel_bias_gather`.

**The MessageGate pointwise combine is BUILT and GPU-verified**: `message_gate`
(`out = (1−tanh(forget_logit))·hidden + tanh(gain_logit)·message`, fp32; MOD:196–200),
`kernel_inkling_message_gate.rs`, 3 tests against `rivoli_oracles::inkling::message_gate` — the
oracle over five hidden widths (RELATIVE, ocml's tanh is ~1 ulp from the host's), the
forget-suppresses/gain-keeps trap, and the 1001 guard. Red proof (`gate-red-proofs.md` §22e):
using `f` instead of `1−f` reddens the oracle and the trap while the guard holds. The down/forget/
gain projections are GEMVs (the engine's linalg) and the per-pass recurrence (T8/T16) is S5.

**The remaining S4 items are reuse + S5 integration, not new kernels** (analysis 2026-11-24, to be
witnessed in S5):
- *Block attention residual* (MOD:843–858): `norm(cat[block_states, partial])` then a sink-less
  attend (`sinks=None`, scaling H^−0.5, no mask, groups=1). The norm is `rmsnorm_weightless_batch`
  (UnweightedRMSNorm); the attend is `sink_attend` with `sink = −∞` — `exp(−∞−m)=0` removes the
  sink column's mass, leaving the ordinary softmax. **Kernel half witnessed**:
  `sink_attend_negative_infinity_sink_is_sinkless` (GPU == CPU oracle at sink=−∞, 2026-11-24). The
  1/3-query bank and the block-bank norm are per-query calls + `rmsnorm_weightless_batch` (S5).
- *Dynamic short convolution* (MOD:353–412): the static `sconv` kernel at `divisor = √(2k+1) = 3`
  with the weight formed as `static_w + weight_proj(hidden)` (a GEMV); the conv-state shift is
  engine bookkeeping.
- *Relative-position bias* (MOD:196–227): the rel norm is the clip family (`scaled_l2_clip` /
  `log_scaled_l2_clip`, threshold rel_l2_clip 8.0); `wr_du` and `rel_logits_proj` are GEMVs; the
  `clamp(distance, 0, extent−1)` gather + `masked_fill` is engine indexing folded into `sink_attend`'s
  `bias`. T17 (the extent's SOURCE differs sliding vs full) is an S5 config trap.

So S4's new-kernel work is **ten ORACLE_VALIDATED kernels** (three clips, two dequants, sconv,
sink_attend, message_gate, and the pair the S5 wiring forced into existence — `kv_scatter`, which
appends one token's projected k/v into the per-head cache, and `rel_bias_gather`, which gathers a
window's distances), each with a recorded red proof (§22a–g plus the clip i/j), plus the measured
reuses (`situ_glu_f32`, `rmsnorm_single`). The census holds both ends (76 launchers, 10
oracle-validated, 16 classified uncalled). The attention/conv/gate families are covered; what
remains is the S5 engine arm that wires them with the position/cache/block-bank/loop logic.

## S5 landing (begun 2026-11-24)

The engine arm's deviceless foundation is landed: `crates/engine/src/inkling/geometry.rs` (the
`inkling` module registered in `engine/lib.rs`) carries the loop's trap-carrying arithmetic as pure
functions over plain data — `application_schedule` (Mini's 6 + 12·3 + 6 = 48 applications of 24
stored layers; Small's plain 0..48), `is_block_boundary`/`is_block_end` (layer 0 IS a boundary, T7),
`sconv_static_divisor`/`sconv_dynamic_divisor` (√5 vs 3, T15), `qk_scale`, and `loop_cache_offset`
(T16: pass 2 → 18, pass 3 → 30, LOOP:542), `cache_layer_count` (48 cache layers for Mini), and the
f32 scratch byte counts (residual/KV/sconv-state), the streamed-unit `expert_layer_bytes`, and
`dense_layer_elements` (the resident per-layer footprint; full vs sliding differ by T11's norms).
Twelve deviceless tests score these rules. The `rocm`-gated `engine.rs` allocates the decode scratch
(`InklingScratch`: the residual stream plus the pre-loop and previous-pass copies the recurrence
restarts from, the block bank, the four short-convolution states of EACH cache layer, a
`ctx`-long K and V per cache layer, and the attention/logit buffers) from those byte counts; the
rocm build is green. The 2026-11-24 review caught both sizings: ONE shared sconv state let a loop
pass continue the window a previous pass left behind and let two layers of one pass share it, so the
state is now indexed by cache layer; and the K/V pair had been allocated from the K+V layer total
into each buffer, a 2× over-allocation.
`names.rs` carries the artifact weight-name layout (the converter's rename-target census): the
per-layer attention/MoE/norm suffixes, the full layers' three q/k/v input norms (T11), the
block-residual query, and the globals (`lm_head.weight`, the final/message query banks). `pin.rs`
carries `InklingLayerPin` (the resolved per-layer dense weights — bf16 projections, f32
norms/sconv/queries) and `place_layer`, which resolves them from the artifact by name; the routed
experts are the streamed unit, placed separately. A review correction gave the FULL layers their
DYNAMIC k/v convolution (`static_weight` plus a `weight_proj` per side, divisor √(2k+1) = 3) instead
of the static one — the two forms are a `LayerSconv` enum so a mis-classified layer cannot run the
wrong conv — and made `attention_input.query` a field of every layer with `i > 0` (`[hidden]` on a
sliding layer, `[3, hidden]` on a full one), which is where the block-residual bank's three queries
live. `forward.rs` carries `proj` (the bf16 GEMM-at-m=1 projection), `logit_tail` (final norm +
lm_head + softcap), the per-head `q_norm_sliding`/`k_norm`/`v_norm` (T4/T18), `gated_output` (the
sigmoid output gate + o_proj), and `attend_gqa` (the batched sink softmax, T10). The same review
reshaped the last two: `sink_attend` is launched ONCE over `heads` blocks with the GQA head→kv-head
map and a parallel block softmax inside the kernel, and it reads the per-head sink logit from DEVICE
memory — the `[heads]` vector `sink_proj` wrote stays a device pointer, so the layer loop has no
per-layer readback and no `device_sync`. The clip family was batched at the same time (each kernel
takes `rows` blocks, one per row/head, so a prefill's q/k norms are one launch rather than one per
token), and `rel_bias_gather` gained the window's `start` (`key_pos = start + t`), which is what
makes a window at `pos >= sliding_window_size` gather the right distances instead of reading from
the cache's origin. The `SinkAttend` jscpd ABI exemption the unbatched field order needed is GONE —
its q/k/v fields no longer collide with the test `AttendIo` helper — so the exemption ledger is back
at 10. `l2_clip`, `scaled_l2_clip`, `log_scaled_l2_clip` and `sink_attend` now have production
callers (the inventory's uncalled table shrank to 16).
The `sconv` kernel was redesigned self-contained (2026-11-24): it takes the running `state`
`[channels, k−1]` + `current` and advances the state in place (shift left one, append current), so
the layer loop keeps no separate window copy; the oracle returns `(out, advanced_state)` and the
suite checks the advance bitwise (§22c re-recorded with a residual plant and a state-advance plant).
The KV cache is per-head-deinterleaved `[kv_heads, ctx, head_dim]` so `sink_attend` reads each head's
`n` vectors contiguous (no gather); the append is the new `kv_scatter` kernel (2026-11-24, §22f),
which scatters one token's projected k/v into the head blocks at `pos`; the rel-bias distance gather
is the new `rel_bias_gather` kernel (2026-11-24, §22g). The attention scratch (k/v, rel_states,
rel_logits, bias, sink, gate) and the static `kv_path` helper (proj → sconv → norm → kv_scatter) are
landed; `sconv`/`kv_scatter` now have production callers. An S2 converter correction (2026-11-24):
the f32-consumed norms/clip-scales/sconv weights are widened bf16→f32 (the launchers take f32 and
`place_f32` requires F32), and rel_proj is transposed to `[rel_extent, d_rel]` for the m=heads GEMM
(`gemm_bf16` computes x·wᵀ); the target-census TSV carries the transposed shape. The expert path is
zero-copy and fused: the source bytes are the mmap'd slices, and the dequant writes bf16 DIRECTLY
with the `w13` de-interleave folded in, so no f32 plane of a 128-expert layer is ever allocated. The
attention sub-block is wired (`attention`: input norm(s) → q/kv/rel-bias/sink → GQA sink attend →
gated output, the sink read ON DEVICE). On 2026-11-25 the remaining S5 pieces landed and were audited:
`sconv_dynamic` and `vaxpy` kernels (12 ORACLE_VALIDATED, 78/63/15/12 census), Inkling inline-specials
`tiktoken.rs` + `inkling_encoding.rs` + `serve/inkling.rs`, the full `InklingPin` (`residency::partition`),
`InklingEngine`, `decode.rs` (loop passes, MessageGates, block-res, MoE, TF `--ppl`), `seam.rs` /
`legality.rs` / CLI / serve wiring, `inkling_engine_anchor.rs` (3/3 GPU green against S3 golden),
8/8 greedy token parity with `Inkling2LoopForConditionalGeneration.from_pretrained` on `"Hi"`
(`[200001, 200008, 371, 150409, 150239, 150330, 196980, 4896]`), P4 residency invariance (90 GiB 24/24
pinned vs 20 GiB 8/24 streamed: 16/16 identical ids), TF `--ppl` row-coherent on 830/830 tokens of
`TOUR.md` (PPL 9.617419, mean NLL 2.263576), release 11.95 tok/s pinned, and `tests/smoke-inkling.sh`
(16/16 green + `--expect-red` red-proofed).

## S6 landing (2026-11-25)

The Small-MXFP8 streaming arm (`Arch::Inkling2`: 48 layers, non-looped, 384 MXFP8 routed experts
top-8, hidden 4608, inter 2304) is built and verified:

- **`.mxfp8` container & repacker**: `crates/artifact/src/quant/mxfp8.rs` (`MXFP8_GROUP = 32`,
  32,845,824 B/expert = exact multiple of `VQ_ALIGN`, 12,612,800,512 B per 384-expert layer),
  `RoutedFmt::MxFp8` (`MX8\0` magic, no shared block), `format/mxfp8_repack.rs` (de-interleaves
  `w13_weight` AND `w13_weight_scale` into contiguous gate/up spans, refuses `0xFF` ue8m0 NaN),
  shared `inkling_convert.rs` rename/transform rules, `crates/artifact/tests/mxfp8_loading.rs`
  (4/4 deviceless green), and `crates/cli/src/bin/convert_inkling_small.rs` (sharded `open_indexed`,
  `--from`/`--to`/`--verify`/`--verify-only`/`--resident-only`/`--skip-resident`). Verified on
  real Small-MXFP8 layer 0 (`model-00001..00048`): `L00.mxfp8` 12,612,800,512 B, 384 experts,
  **0 bytes differ**.
- **Fused MXFP8 SiTU-GLU kernels**: `crates/kernels/src/prelude/dot_mxfp8.rs` (row-wise `[1,32]`
  ue8m0 wave dot using LDS `fp8_lut` + `e8m0f`) and `crates/kernels/src/moe_mxfp8.rs`
  (`moe_gateup_mxfp8_situ` + `moe_down_mxfp8_weighted`, `fp8_lut()` before any early return, f32
  intermediate and pass-2 weight multiply), launched via `launch_moe_expert_range_mxfp8_situ`
  (dynamic LDS `GEMV_FP8_LDS_BYTES`). **Corrected 2026-11-25 (Wave 2 perf):** the e4m3 weights
  now decode arithmetically in registers (`E4M3_REBIAS`), so neither kernel builds the LDS
  `fp8_lut`, `MxFp8Row` carries no `lut`, and the launcher passes `LdsBytes(0)` — the S6 record
  above is what shipped, not what the tree does now. Kernel census updated to **79/79/0** (98
  shipped Rust kernels, 64 production callers, 15 classified uncalled, 13 `ORACLE_VALIDATED`); GPU oracle
  `crates/engine/tests/kernel_inkling_moe_mxfp8.rs` 6/6 green under flock (including T1 raw
  exponent trap, gate/up swap separation, offset window with `w == 0` skip, and guard codes
  1001–1006).
- **Engine, legality & CLI cutover**: `InklingPin::build_mxfp8` charges globals + dense +
  `PIN_SLACK` + `ScratchSizes` to `PoolPlan` before sizing `RoutedPool`; `moe_mxfp8.rs` runs
  residents-first on `compute_stream` (acc row 0) overlapped with per-miss tickets on `miss_stream`
  (acc row 1) into `launch_moe_acc_drain_to`; `legality.rs` flips `Arch::Inkling2` to live streaming
  and deletes `INKLING_ARM_NOT_BUILT`; `seam.rs`/`main.rs`/`bench.rs`/`serve/mod.rs` share
  `Engine::Inkling` and Inkling chat framing across both members.
- **Gates**: `crates/engine/tests/inkling_small_mxfp8.rs` 4/4 GPU green (with `--features
  teacher-forcing`): door check, TF finite NLL, P4 bit-identical ids + logits across pinned (0
  warm fills) vs O_DIRECT streaming (36 fills), and bf16 vs MXFP8 cross-check (worst rel logit
  diff `1.48e-5`). `tests/smoke-inkling-small.sh` landed.

## S7 witnessed measurements & full Small artifact (2026-11-25)

All Mini device measurements ran under `tests/gpu-witness.sh` (`flock /var/run/sys-gpu.lock`,
empty KFD witness, pre-arm GTT 17–18 MiB) on the release+`teacher-forcing` binary (`225ff6f`,
zero builds between arms):

| arm | budget | residency | throughput / NLL | wall breakdown (ms/tok) | comparison |
|---|---|---|---|---|---|
| Mini bench A (`"The sky is blue because"`, 64 tok) | 90 GiB | 24/24 layers pinned (48.6 GiB tier, 3.2 GiB floor) | **11.69 tok/s** (24192 hits, 0 misses) | 85.559 = attend 1.964 + ffn 77.451 + head 5.988 + other 0.155 (0.18%) | baseline |
| Mini bench A' (control, identical flags) | 90 GiB | 24/24 layers pinned | **11.59 tok/s** (24192 hits, 0 misses) | 86.261 = attend 1.949 + ffn 78.214 + head 5.944 + other 0.154 (0.18%) | **64/64 ids identical** to A |
| Mini bench B (P4 streaming) | 20 GiB | 8/24 pinned, 16/24 streamed (18.2 GiB tier) | **2.25 tok/s** (6048 hits, 18144 misses) | 444.413 = attend 2.344 + ffn 436.625 + head 5.050 + other 0.394 (0.09%) | **64/64 ids identical** to A |
| Mini TF `--ppl` (`TOUR.md`, 830 tok) A / A' / B | 90 / 90 / 20 GiB | 100% / 100% / 25% hits | **PPL 9.617419** (mean NLL 2.263576) | — | **830/830 NLL bytes identical** A=A'=B |
| Mini MXFP8 bench A (`"The sky is blue because"`, 64 tok, hot NVMe) | 64 GiB | 3072/3072 experts (100%, 23.6 GiB pool, 3.0 GiB resident) | **14.75 tok/s** (23393 hits, 799 cold fills) | 67.790 = attend 1.710 + ffn 60.863 + head 5.075 + other 0.142 (0.21%) | **+26% vs bf16** at half RAM; 8/8 first-party `"Hi"` parity |
| Mini MXFP8 bench A' (control, hot NVMe) | 64 GiB | 3072/3072 experts (100%) | **14.30 tok/s** (23393 hits, 799 cold fills) | 69.937 = attend 1.708 + ffn 62.890 + head 5.194 + other 0.145 (0.21%) | **64/64 ids identical** to A (14.13 tok/s to EOS at 130 tok) |
| Mini MXFP8 bench B (P4 streaming, hot NVMe) | 12 GiB | 1143/3072 experts (37.2%, 8.8 GiB pool) | **12.60 tok/s** (22252 hits, 1940 misses) | 79.360 = attend 1.696 + ffn 72.436 + head 5.079 + other 0.149 (0.19%) | **64/64 ids identical** to A (**5.6×** bf16 20 GiB streaming) |
| Mini MXFP8 TF `--ppl` (`TOUR.md`, 830 tok) | 64 GiB | 99.15% hits | **PPL 9.577597** (mean NLL 2.259427) | — | paired dNLL vs bf16: `-0.00415` nats, 95% CI `[-0.01257, +0.00427]` (PASS, worse% `48.6%`) |
| Small bench A (`"The sky is blue because"`, 64 tok, hot NVMe) | 100 GiB | 2849/18432 experts (15.5%, 87.4 GiB pool, 12.4 GiB resident) | **2.67 tok/s** (18526 hits, 5666 misses = 76.6% hit) | 374.041 = attend 1.854 + ffn 361.597 + head 10.552 + other 0.037 (0.01%) | baseline (prefill 18 tok in 10.8 s) |
| Small bench A' (control, identical flags, hot NVMe) | 100 GiB | 2849/18432 experts (15.5%) | **2.66 tok/s** (18526 hits, 5666 misses = 76.6% hit) | 375.975 = attend 2.261 + ffn 362.992 + head 10.665 + other 0.057 (0.02%) | **64/64 ids identical** to A |
| Small bench B (P4 streaming, hot NVMe) | 32 GiB | 626/18432 experts (3.4%, 19.4 GiB pool) | **1.37 tok/s** (8607 hits, 15585 misses = 35.6% hit) | 731.836 = attend 2.439 + ffn 718.711 + head 10.619 + other 0.067 (0.01%) | **64/64 ids identical** to A |
Host loadavg during the Small hot-NVMe arms was `8.81 → 11.08` (A), `11.08 → 12.53` (A'), and
`12.53 → 11.46` (B) on 32 cores (pre-arm GTT 17–18 MiB, KFD witness empty across all three).
Note on `--max-mem 115` on this 124 GiB RAM host: `115 GiB` pinned + 12 GiB resident OS/page
pressure triggered heavy host swap thrashing during prefill; `100 GiB` (leaving ~24 GiB headroom)
runs cleanly at 2.67 tok/s.

- **All 11 Inkling GPU test binaries** (`kernel_inkling_{l2clip,dequant,sconv,sink_attend,message_gate,vaxpy,kv_scatter,rel_bias,moe_mxfp8}`, `inkling_engine_anchor`, `inkling_small_mxfp8` with `--features teacher-forcing`): **56/56 tests pass** under `flock`.
- **P7 red proofs paid (§22h in `docs/measurement/gate-red-proofs.md`)**: planting `scale.wrapping_sub(1)` in `dot_mxfp8_wave` reddened `kernel_inkling_moe_mxfp8` (3/6 failed, guards+swap held) AND `inkling_small_mxfp8` (`bf16_resident_experts_match_the_streamed_mxfp8_ones` failed on argmax `[43, 1, 35, 24]` vs `[15, 35, 55, 53]`); restored byte-identical and re-verified green. Planting a wrong refusal fragment in `tests/smoke-inkling-small.sh` exited 0 under `--expect-red='without the expected message'`, then restored byte-identical.
- **Full Small-MXFP8 checkpoint & artifact**: all 48 shards (`619,380,084,424 B`, rev `14f2cb29`) downloaded to cold storage `/swarm/storage/ai/rivoli/inkling-2-small-mxfp8-src`, converted to `/swarm/storage/ai/rivoli/inkling-2-small-mxfp8-art`, and promoted to hot NVMe `/var/db/rivoli/inkling-2-small-mxfp8-art`:
  - `resident.safetensors` (~13 GB): **996/996** tensor names and shapes match first-party `Inkling2ForCausalLM` meta `state_dict` (0 missing, 0 extra, 0 shape mismatches; 422 vision and 192 expert tensors excluded by name).
  - `L00.mxfp8`–`L47.mxfp8`: 48 files × `12,612,800,512 B` each (`~564 GiB` routed set); `L00.mxfp8` and `L47.mxfp8` verified byte-for-byte against source shards (`384 experts, 0 bytes differ`).
  - `tests/smoke-inkling-small.sh` **17/17 green** (`SMOKE GREEN: 17 cells`, exit 0, 2026-11-25) on `/var/db/rivoli/inkling-2-small-mxfp8-art` at `--max-mem 100`: all 6 refusals with the table's own wording, `--cache-policy 2q`, the `--mode int4` loud MXFP8 fallback, `bench` (4 ids), `determinism A-vs-A` (identical ids), P4 (identical ids at 100 vs 24 GiB), and the serve cell end to end -- readiness after load, `/v1/models`, non-stream reply `'Hi there'`, `enable_thinking=false` content `'Hi'`, SSE frames + `[DONE]`, clean shutdown. Red-proofed by a planted refusal fragment under `--expect-red='without the expected message'`, then restored byte-identical.
## Integration surface (S1's edit list)

- `crates/core/src/legality.rs`: Arch variants (proposed `Inkling2Loop` for Mini,
  `Inkling2` for Small; `name()`'s docs-scope contract must be checked first — two
  arches share ONE scope `inkling`), `ARCH_COUNT` 5→7, `ALL`, ordinal/name/summary;
  shared `INKLING_ARM_NOT_BUILT` + `INKLING_MTP_NOT_LOADED` consts, ONE row fn called by
  both `decide` arms (jscpd); dated correction of `QWEN_ARM_NOT_BUILT`'s "only row that
  refuses every cell" sentence; ARMLESS grows to 3.
- `crates/artifact/src/arch.rs`: root-only sniff (`Inkling2LoopForConditionalGeneration`
  | `inkling2_loop`, `Inkling2ForConditionalGeneration` | `inkling2`); nested
  `inkling2_loop_text`/`inkling2_text` refused at the sniff, asserted in config validate.
- `crates/artifact/src/inkling_config.rs` (new, qwen_config.rs template): two wrapper
  types over ONE shared text config; `validate` asserts root+nested spellings, the
  attribute_map JSON spellings, loop validate rules, and the behavior-pinned extra
  fields (math doc §7).
- `crates/artifact/src/schema.rs` tests against the vendored configs; pin gate mirroring
  `tests/qwen_names.rs:456–480` over the TSV header (`include_bytes!`, length + FNV-1a).
- Doors: `crates/cli/src/main.rs:179`, `bench.rs:112`, `serve/mod.rs:314`/`:365`, each
  with fragment tests; `crates/oracles/src/golden.rs` `magic_of` exhaustive match forces
  two new 8-byte MAGIC consts.
- `crates/cli/tests/docs.rs` SCOPES gains `"inkling"`; INDEX gains exactly one row per
  doc — landed in THIS change (S0), not "next commit".
- Engine seam: NO qwen arm exists in `seam.rs` ArchCfg/Engine; likewise no inkling arm
  at S1. Engine work starts S5.

Naming exception (argued as in the qwen plan): arch-scoped modules
(`inkling_config.rs`, `convert_inkling.rs`, `inkling_encoding.rs`, `engine/src/inkling/`)
name a PORT BOUNDARY; kernels, traits and structs are behaviour+ABI named (skill §8).
Model membership is data in the checked census tables.

## Skill census (model-port §1–8), current answers

1. **Vendored pins, recomputed**: S0 vendored both members' config/tokenizer/generation/
   quant pins + the shared tiktoken file; the pin GATE (`crates/artifact/tests/inkling_pins.rs`,
   recompute FNV-1a from the live vendored bytes, compare to the TSV header) landed S1 and is
   red-proofed (d, above). CLOSED (S1).
2. **Tokenizer TYPE first**: settled BEFORE weights moved — tiktoken BPE rank file,
   200,055 lines, byte-identical between members; chat template pinned per member.
   CLOSED (S0).
3. **Converter exact counts, exclusions by name**: CLOSED (S2). `convert_inkling` consumed
   822 TEXT tensors and excluded 422 EXCLUDED-VISION by name on the real shard (census confront,
   `consumed == 822` guard); `mtp.safetensors` never opened. The output's 693 names+shapes equal
   the first-party text model's state dict (`tests/inkling_convert.rs`).
4. **First-party anchors + defect matrix**: CLOSED (S3). The DEQUANT oracle is first-party-
   verified (S2, `dequant-golden.json`); the MODEL anchor (S3) runs transformers' own
   `Inkling2LoopForCausalLM` over a tiny config, vendors three byte-pinned goldens reproduced
   3 of 3, a five-trap defect matrix reddening on both draws with declared first-touched buckets,
   and carved fp32-vs-fp64 tolerance floors (the reference is fp32-locked). Record:
   `docs/measurement/inkling2-reference/anchor.md`; gate `crates/oracles/tests/inkling_anchor.rs`.
5. **Kernel census both ends**: CLOSED (S4+S5+S6). `kernel_coverage.rs` checks all 79 launchers
   (64 callers, 15 classified uncalled, 13 `ORACLE_VALIDATED`) both ends against `docs/reference/kernel-inventory.md`.
6. **Registration in the SAME change**: this change adds `"inkling"` to SCOPES, both
   docs, and their INDEX rows. CLOSED with S0.
7. **Thin smoke WITH the exit gate**: CLOSED (S5). `tests/smoke-inkling.sh` landed with Mini's exit gate
   (16/16 green + `--expect-red` red-proofed 2026-11-25).
8. **Behaviour+ABI names**: CLOSED (S4+S5); every kernel and ABI struct is behaviour+ABI named; port-boundary
   modules argued above.

## Measurement posture

Both Mini's and Small's witnessed hot-NVMe measurements on this box are recorded in the S7 section
above (64-token release decode, A-vs-A' control and P4 A-vs-B budget invariance, clean KFD witness,
17–18 MiB pre-arm GTT; Mini also carries 830-token byte-identical TF NLL).