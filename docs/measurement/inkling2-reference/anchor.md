---
status: live
scope: inkling
verdict: THE S3 ANCHOR IS BUILT -- the Inkling 2 Mini first-party anchor runs transformers' own inkling2_loop text model (Inkling2LoopForCausalLM @ d8f10a82, torch 2.15.0+cpu) over a tiny config derived from the real Mini config, on CPU fp32, and vendors three goldens (two decode draws + one prefill) byte-pinned and reproduced 3 of 3 by inkling-anchor.sh. The DECODE goldens are the model's own CACHED step (`use_cache=True`: a 16-token prefill whose cache is not captured, then ONE token stepped against it), so every bucket is `[1, 1, …]` and the incremental body -- the conv-state shift, the KV append, the per-pass cache offsets -- is what they pin; each looped capture layer (6, 11, 17) stores all three passes. The defect matrix has five forward-operable traps (l2clip_no_clamp, softcap_sigmoid_capped, sconv_no_divisor, router_no_route_scale, logscale_wrong_floor), each reddening against the None golden on BOTH salts with its declared first-touched bucket; logscale first touches layer 5 not layer 0 because the sliding layers carry no LogScale. The tolerance floors are CARVED fp32-vs-fp64 per operator (RMSNorm, SoftcappedSwiGLU, both clips on an input scaled into the clamp branch, the static short convolution, the loop MessageGate, and the sink softmax on a single query row) because the reference is fp32-LOCKED -- the residual stream and most params force .float(), so a whole-model fp64 run computes fp32 internally and shows zero diff. Two plants were measured VACUOUS and replaced: the L2Clip +1e-6 epsilon (sub-fp32 at these norms) and the router bias-in-weights (e_score_correction_bias is a zero buffer at random init). A third vacuity was the CARVE rather than the model: the L2Clip floor was exactly 0 because every captured head vector sat under threshold, so the carve now scales its input until the clamp is entered. The model forward-pass goldens score the S4 kernels; the converter-level traps T1/T2/T3/T12/T13/T14 are gated by S2's dequant/census/name gates, not here.
---

# The Inkling 2 Mini S3 anchor

What produced these files, what they pin, and what was measured from them.
`crates/oracles/tests/inkling-anchor.sh` regenerates everything;
`crates/oracles/tests/inkling_anchor_driver.py` is what it runs;
`crates/oracles/tests/inkling_anchor.rs` is the deviceless gate that reads what it vendors.

## The stack, pinned

| what | value |
|---|---|
| reference | `transformers.models.inkling2_loop` @ `d8f10a82213a3a9db48385aba3a4b57ab535de50` |
| torch | 2.15.0+cpu |
| venv | `/var/cache/rivoli/scratch/inkling-venv` |
| entry point | `Inkling2LoopForCausalLM` (text model; v1 is text-only) |
| dtype | fp32 (the reference forces `.float()` throughout — see the floors) |
| conv path | reference PyTorch (`causal_conv1d` absent, asserted by the driver) |

No GPU, no flock: the short convolutions fall back to the reference body when `causal_conv1d` is
absent, which is the arithmetic this anchor pins. The driver asserts the fast path is absent
rather than trusting the venv.

## The tiny config

`docs/measurement/inkling2-reference/mini/anchor-config.json`, derived from the real Mini config
by `/var/cache/rivoli/scratch/gen_inkling_tiny.py`. Widths shrink (hidden 256, 8 Q / 2 KV heads,
head_dim 32, d_rel 4, rel_extent 64, 8 experts top-2, intermediate 128, vocab 256 / unpadded 250);
the structure stays the real model's: 24 layers, block 6, loop `[6,18)` ×3, the block-end
complement `local_layer_ids`. Two forced non-width overrides, each so an operator is observable
within the short golden window:

- `log_scaling_n_floor = 4` (real 8192). LogScale activates only past the floor; at the real floor
  a 17-token window never reaches it and the `logscale_wrong_floor` defect would be vacuous.
- `qk_l2_clip_threshold = sqrt(head_dim)·sqrt(2)` (the real rule, 16√2 at head_dim 128; 8 at 32).

## The goldens

| vendored file | bytes | FNV-1a | holds |
|---|---|---|---|
| `inkling-anchor-decode-inkling-1.bin` | 18002 | `c06e4a7ef0b2f164` | cached decode step, salt inkling-1 |
| `inkling-anchor-decode-inkling-2.bin` | 18002 | `b05a31b0f65ac4be` | cached decode step, salt inkling-2 |
| `inkling-anchor-prefill-inkling-1.bin` | 263406 | `abcbffe60fc880b3` | 16-position prefill, salt inkling-1 |
Each golden holds **16 float tensors**: `logits` and six `model.layers.{L}.hidden` for the capture
layers `(0, 5, 6, 11, 17, 23)` — the first sliding/MoE layer, the first full block-end, the first
loop layer, a full block-end inside the loop, the last loop layer, and the last full block-end —
plus the nine loop-pass tensors `model.layers.{6,11,17}.pass{1,2,3}.hidden`. The widths are the
tiny config's 256 hidden and 250 unpadded logits; the ROW count is the protocol's: `[1, 16, …]` for
the prefill and `[1, 1, …]` for the decode step. The container is the `RIVILGLD` layout
`crates/oracles/src/golden.rs` reads (the magic landed at S1). The two decode draws are independent
per-tensor on `to_bits` (`draws_are_independent`, whose shared-tensor count is now 16), not merely
different hashes.

### The decode protocol is the model's own cached step

`--mode decode` does NOT run a 17-token forward and call it a decode. It runs
`model(input_ids=ids[:, :16], use_cache=True)` with NO hooks, then registers the hooks and steps
`ids[:, 16:17]` against that cache. That is the incremental body and nothing else:
`causal_conv1d_update`'s conv-state shift instead of a full-window convolution, the KV append onto
the prefill's keys/values, and the loop passes' per-pass `cache_offset` (T16). The prefill's cache
is the reference's own hybrid `DynamicCache` — 48 layer entries, because the text model appends
`loop_num_passes − 1` deep copies of the loop layers' caches, plus four conv states per layer. Both
protocols are recorded in each golden's metadata (`protocol`, `prefill_seq`, `capture_passes`,
`loop_passes`), so a reader never has to infer which body produced the bytes.

### Every loop pass is captured

Layers `6`, `11` and `17` sit inside `[6, 18)`, so the model applies each of them
`loop_num_passes` times in ONE forward. `model.layers.{L}.hidden` keeps the LAST pass (the value
the following layer consumed) and every pass is also stored as
`model.layers.{L}.pass{p}.hidden`. The driver asserts the call count per looped layer, and
`inkling_anchor.rs` asserts the three passes are not bit-identical — a golden that kept only the
last pass could not see the recurrence at all.

## The defect matrix — five forward-operable traps, both salts

Each row plants one trap in the reference body, captures a golden, and `--compare` asserts the
declared first-touched bucket changed. Converter-level traps (T1/T2/T3/T12/T13/T14) are gated by
S2, not here.

| defect | trap | first touched | changed |
|---|---|---|---|
| `l2clip_no_clamp` | T4 family | `model.layers.0.hidden` | 16 |
| `softcap_sigmoid_capped` | T6 | `model.layers.0.hidden` | 16 |
| `sconv_no_divisor` | T15 | `model.layers.0.hidden` | 16 |
| `router_no_route_scale` | T5 family | `model.layers.0.hidden` | 16 |
| `logscale_wrong_floor` | T9 | `model.layers.5.hidden` | 15 |

A decode golden carries 16 tensors, so the four layer-0 traps reach every bucket (16/16), while
`logscale_wrong_floor` touches 15/16 and first touches layer 5: the sliding layers (including layer
0) carry `ScaledL2Clip`, not `LogScaledL2Clip`, so that trap stays invisible until the first full
layer. That localization is the matrix discriminating WHERE the trap lives, preserved under the
cached-decode protocol.

### Two plants and one carve measured vacuous

- **L2Clip `+1e-6` epsilon** (T4 literal): at these norms the epsilon is sub-fp32 — the captured
  tensors were bit-identical (maxdiff 0). Replaced by removing the `clamp(max=1.0)`, which scales
  short vectors UP and is observable.
- **Router bias-in-weights** (T5 literal): `e_score_correction_bias` is a zero buffer at random
  init, so adding it to the weights changes nothing. Replaced by dropping `routed_scaling_factor`
  (route_scale 8), observable.
- **The clip CARVE** (not the model): both clip floors were `0` because the captured vectors never
  reached `threshold`, so the measured floor described an identity. The carve scales its input
  until the clamp is entered and asserts it did — see the tolerances section.

## Tolerances — carved fp32 floors, and the fp32-locked finding

The inkling reference is **fp32-locked**: the decoder residual stream does
`residual = hidden_states.float()` and most params are hardcoded `dtype=torch.float32`. A
whole-model fp64 run therefore computes fp32 internally and shows ZERO diff — the qwen
"double the whole model" approach does not apply. The floor lives in each operator's own fp32
arithmetic, measured by CARVING: a pre-hook captures the fp32 inputs a real forward fed the
operator, then the operator is recomputed fp32 vs fp64 on those inputs.

| operator | salt `inkling-1` | salt `inkling-2` |
|---|---|---|
| `Inkling2LoopRMSNorm` | 2.384186e-07 | 2.384186e-07 |
| `Inkling2LoopSoftcappedSwiGLU` | 1.490116e-08 | 2.980232e-08 |
| `Inkling2LoopScaledL2Clip` (scaled ×8.15 / ×8.504 into clamp) | 2.384186e-07 | 2.384186e-07 |
| `Inkling2LoopL2Clip` (scaled ×17.93 / ×21.97 into clamp) | 2.384186e-07 | 2.384186e-07 |
| `Inkling2LoopShortConvolution` (`(conv + residual)/√(k+1)`) | 2.980232e-08 | 2.980232e-08 |
| `Inkling2LoopMessageGate` (`(1−tanh f)·h + tanh g·m`) | 2.384186e-07 | 1.192093e-07 |
| `sink_attend` (1 query, 8 heads, 17 keys, sink logit) | 2.384186e-07 | 2.384186e-07 |

Measured 2026-11-24 via `--tolerance` on both draws. Three things changed on 2026-11-24, and all
three are about the carve being NON-VACUOUS rather than about new arithmetic:
- The carve now runs under the anchor's DECODE protocol (a one-token step against the prefill's
  cache), so the streams it measures are the ones the shipped decode path feeds.
- **Both clip floors were previously `0`, and a `0` floor measures nothing.** At the tiny widths
  every captured head vector's norm sits below `threshold`, so `clamp(max=1.0)` is the identity and
  fp32 == fp64 exactly. The carve now scales the captured input by `2·threshold / max_norm` and
  ASSERTS that the scaled row enters the clip branch before it reports a floor.
- `MessageGate` re-implements the weightless norm inside the carve: the reference's own
  `UnweightedRMSNorm` casts to fp32 first, which would make the "fp64" arm compute in fp32 and
  report `0` — the fp32-locked trap one level down. The same rule applies to every projection the
  carve touches: widen the weight per arm, never let the reference's `.float()` bind the fp64 arm.

The floors are expected to repeat across the two draws because they are fp32 rounding constants
(2^-22, 2^-24), a property of the arithmetic rather than of the weight values; a draw-dependent
floor is itself a finding, not noise. The discrete router is excluded: its top-k selection is not a
continuous rounding error.

## Red proofs

| gate | plant | observed red |
|---|---|---|
| (h) byte pin | last byte of `inkling-anchor-decode-inkling-1.bin` XOR 0xFF | `the_vendored_goldens_match_their_byte_pins` exit 101: FNV-1a mismatch |

The defect matrix's own red is every row's `--compare` (a defect that changed nothing, or changed
its first-touched bucket absent, fails); the script exits non-zero if any row fails or any
vendored golden does not reproduce.

## OPEN and OWED

- The per-operator tolerance table here covers the seven continuous candidates carved so far —
  RMSNorm, SwiGLU, both clips (on a scaled, non-vacuous input), the static short convolution, the
  loop MessageGate, and the sink softmax with its AV sum — each on both draws and under the decode
  protocol. It does NOT yet carve `Inkling2LoopLogScaledL2Clip` as a composition, the DYNAMIC short
  convolution (`√(2k+1)` with a projected tap bank), or the final logit softcap; those are the rows
  still owed, and each owes its own non-vacuity argument before its number is quoted.
- The real-checkpoint forward golden (the 19.5 GB source through `from_pretrained`) is the S5
  parity gate's reference, not this tiny-config operator anchor.
