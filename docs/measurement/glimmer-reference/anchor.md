---
scope: glimmer
status: data
verdict: The S1b anchor exists and runs — Muse Glimmer's own first-party stack (transformers 5.16.0.dev0 at commit fe747d88, torch 2.13.0+cpu, python 3.14.6) executed at tiny widths but the REAL structure, and it needs NO GPU because this reference is plain PyTorch with a CPU path for every operator. FOUR files are vendored (tests/glimmer-anchor-{text,draft}-{1,2}.bin; text 643,957 B, draft 72,145 B), two weight draws x two modes, each reproduced byte-for-byte on a later run, and read by tests/glimmer_anchor.rs with no python, no venv, no network and no device. FOURTEEN defect runs are scored at both draws, 28 runs in all, and each is GATED on the captures it must leave bit-identical rather than merely on having changed something. THE FINDING THAT ONLY THIS CAN SEE - softcap_off moves 7 of 1103 captures and leaves emitted.ids identical, so the argmax-invariant logit path is not an argument but a measurement, and every greedy gate in this repo is provably blind to it. Two reference behaviours were discovered by running it: the DFlash drafter's default mask is block-wide against context+block K/V and RAISES, and passing the correct 2D mask only works with use_cache=False because a fresh DFlashCache reports kv_length 0. The green sets are scoped to step 0 because a defect that shifts the argmax contaminates every later step through the token it feeds back - localisation is only possible on the prefill. Five deviations are declared in the metadata and pinned by the test - eager attention, fp32 against a bf16 checkpoint, the ForConditionalGeneration wrapper (the softcap lives only there), shrunk special-token ids, and output_multiplier kept at the released value rather than recomputed.
---

# The Muse Glimmer S1b anchor

**What it is.** Goldens emitted by `transformers.models.muse_glimmer` and
`.muse_glimmer_assistant` themselves, at a tiny config, captured on the way past by
`tests/glimmer_anchor_driver.py`. **Nothing in the driver re-implements the model.**
`docs/reference/glimmer-architecture.md` was extracted by *reading* the modeling code, and a
fixture derived from that reading would put one misreading in the spec, in the golden, and in the
kernel checked against it.

**What it is not.** `tests/glimmer_anchor.rs` is a **fixture-integrity gate, not a correctness
gate**. It compares no rivoli output to anything, because at S1b there is no Glimmer kernel to
score — so the literal answer to "what wrong implementation passes this" is every one. What it does
is hold the files to the shape S2 will reach for, refuse a file that is not the one this doc
describes, and refuse a tiny config that has stopped matching the real checkpoint.

## The environment, and why it needs no GPU

```
transformers 5.16.0.dev0 @ fe747d88a3296bd94d426db2717f232f9d4afdb7
torch 2.13.0+cpu   numpy 2.5.2   python 3.14.6
venv: /home/rhansen/glimmer-anchor/venv   (separate from K3's, which has no muse_glimmer at all)
```

**Torch is CPU on purpose.** K3's anchor needs a device because its KDA ops are triton kernels with
no CPU path; this reference is plain PyTorch throughout, so regeneration takes no GPU lock and can
run beside a benchmark. The bytes are vendored anyway — reading them needs no python either, and a
gate that needs a device on this machine blocks correctness work that never wanted one.

`preflight_env()` refuses to generate under a different env, and **reads the pin out of the
vendored golden** rather than restating it, so a deliberate re-pin needs no edit to the driver.
Proven red by running the driver under K3's venv: it named all four drifted versions and stopped.

## The tiny config

Every structural field is real; only widths shrink. Depth and structure are where the traps live.

| | tiny | real | why this one |
|---|---|---|---|
| `hidden_size` | 72 | 6656 | **not** `num_heads * head_dim` — the real model's shape, and the assumption a port is most likely to collapse |
| `num_attention_heads` / `num_key_value_heads` | 6 / 2 | 32 / 2 | group 3: neither 1 (MHA, no broadcast) nor equal to the kv count |
| `head_dim` | 8 | 128 | |
| `intermediate_size` | 216 | 19968 | keeps the real 3x hidden |
| `num_hidden_layers` | 8 | 52 | so "every 4th backward from the last" lands on 3 and 7 — the `[w,w,w,full]` pattern twice, a full layer that is not the last, and a last layer that is |
| `sliding_window` | 4 | 2048 | crossable by the 18 positions generated. A window the sequence never reaches tests the dense path and passes vacuously |
| `vocab_size` | 61 | 202048 | prime, so it collides with no other width |
| `rms_norm_eps` / `post_norm_eps` | **1e-5 / 1e-8** | same | REAL. The two-eps sandwich is the trap the anchor exists for |
| `qk_scale_factor`, `output_multiplier`, `final_logit_softcapping`, `rope_theta` | **real** | | |

`tests/glimmer_anchor.rs::the_tiny_config_kept_the_real_values` compares every "real" field against
the vendored `config.json` rather than against constants, so an upstream revision that moves one
fails the gate instead of being agreed with.

## What is captured

Prefill of 12 tokens, then 6 decode steps; 1099 float tensors and 4 int tensors per text golden.
Per step and layer: the four sandwich norms, `qk_norm` for q and k separately (so the norm is
separable from the 3.87 scale), q/k on both sides of the rotation **on rotated layers only**, the
attend's post-cache K/V, its mask and weights and output, the sigmoid gate, the gated value
entering `o_proj`, the SwiGLU output. Per step: the RoPE table, the weightless embed norm, the final
norm, the logits. Plus `prompt.ids`, `emitted.ids`, and the per-layer sliding/roped flags.

**The ring-KV fixture is a shape, and it shows eviction.** On a sliding layer the prefill still sees
all 12 rows and is windowed by the *mask*; from the first decode step the cache itself holds exactly
`sliding_window` = 4 rows. On a full layer it grows: 12, 13, … 18. A port may truncate during
prefill instead and get the same numbers; what it may not do is keep more than the window after.

The draft mode captures one DFlash step: the concatenated target context at
`len(target_layer_ids) * hidden`, the encoder projection, the block embedded **raw** from the
target's embedding matrix, per-layer Q at block length against K/V at context+block, the two
per-layer norms, and logits from the **target's** lm_head.

## Two reference behaviours found by running it

Neither is in `glimmer-architecture.md` §11, because neither is visible by reading:

1. **The drafter's default attention mask is unusable.** `MuseGlimmerAssistantModel.forward` builds
   its masks from `inputs_embeds=noise_embeds`, so the mask is block-wide while K/V span
   context+block, and the reference raises on the add. The caller owes it a 2D mask of length
   `context + block`.
2. **…and that mask only works with `use_cache=False`.**
   `create_bidirectional_sliding_window_mask` takes `kv_length` from `past_key_values` when one is
   present, and a freshly built `DFlashCache` reports 0 — so the correct 16-wide mask comes back 4
   wide and the same error returns. A port's first draft call will hit both.

## The defect matrix — 14 defects x 2 draws, all green

Each defect declares the captures it must leave **bit-identical**, and `--compare` asserts both
halves: that something moved, and that nothing else did. **A defect run that reddens everything
proves nothing about where the arithmetic lives.**

| defect | moved / held | what it proves localises |
|---|---|---|
| `post_norm_eps_shared` | 917 / 186 | the two eps are separable: everything into and through the first attention holds |
| `norm_not_centered` | 1029 / 74 | `x*(1+w)` against `x*w` |
| `qk_scale_on_k` | 973 / 130 | **q holds while k moves** — the scale is on Q alone |
| `qk_norm_off` | 1026 / 77 | a norm that ships no tensor, so a port can omit it silently |
| `rope_interleaved` | 1022 / 81 | rotate_half against rivoli's interleaved convention. The table holds; only its application moves |
| `rope_on_nope_layers` | 842 / 261 (+56 new) | trap #1. The NoPE layers begin *calling* the rotation, so 56 captures exist only under the defect |
| `window_off_by_one` | 1024 / 79 | **the full layers' masks hold** — a window change must not reach them |
| `full_layers_slide` | 968 / 135 | layers 0-2 precede the first full layer and hold entirely |
| `gate_disabled` | 1017 / 86 | everything into and through the attend holds; only the gate and after moves |
| `kv_broadcast_blocked` | 1018 / 85 | the attend's *inputs* hold — this separates a broadcast bug from a projection bug, which look identical one tensor later |
| `softcap_off` | **7 / 1096** | see below |
| `embed_norm_off` | 1030 / 73 | the other weightless norm |
| `draft_context_unprojected` | 38 / 12 | **`attend.q` holds**: the context enters as extra K/V and bypasses Q entirely |
| `draft_causal` | 40 / 10 | bidirectional against causal. Setting `is_causal` does nothing — causality is entirely in the mask |

**`softcap_off` is the one only this fixture can see.** It moves the 7 logit tensors and leaves the
other 1096 captures — including `emitted.ids` — bit-identical. The argmax-invariance of
`T*tanh(x*mult/T)` stops being an argument in §9 and becomes a measurement, and the corollary is
sharp: **every greedy decode gate in this repo is provably blind to a wrong logit scale on this
model.** Nothing but a value-level fixture will catch it.

**Green sets are scoped to step 0, and that is a fact about the model.** A defect that shifts the
argmax changes the token fed into step 1, so from t1 onward even layer 0 differs for a reason that
has nothing to do with where the defect lives. Only the prefill, whose input is the fixed prompt,
can localise anything. The first version of the matrix declared unscoped green sets and every one
of them failed at t6 on exactly this.

## Reproduction

```bash
GLIMMER_ANCHOR_VENV=/home/rhansen/glimmer-anchor/venv tests/glimmer-anchor.sh
```

Regenerates all 28 runs, `cmp`s each clean run against its vendored twin, prints the matrix, and
fails if any defect reddens nothing, violates its green set, or if the two salts ever produce
identical files. Measured 2026-08-11: all four clean runs reproduce byte-for-byte.

The `tests/*.bin` files are read by `cargo test --test glimmer_anchor` — no feature, no device.
**When the byte pins in that file's `GOLDENS` table fail after a deliberate regeneration, update
the constants and say so here.** Re-vendoring is a reviewed change, not a side effect of running
the driver.

## Declared deviations

Each is in the goldens' own metadata and pinned by the test:

1. `_attn_implementation` forced to `eager` — the semantics the fused paths approximate, and the
   only one the driver can tap.
2. **fp32, while the checkpoint is bf16.** Right for a reference — the point is to pin arithmetic,
   not to reproduce one accumulation order — but it is where S2's tolerance decision starts.
3. `MuseGlimmerForConditionalGeneration` on text-only input, because the logit softcap lives only
   on that wrapper (M:1253-1260). The vision tower is built at toy widths and never runs.
4. `bos`/`eos`/`pad` and the image/video token ids shrunk into the tiny vocab.
5. `output_multiplier` kept at the released 0.19611613513818404 rather than `1/sqrt(hidden/256)`
   recomputed at hidden 72 — it is a config value the port reads, not a formula it evaluates.

## What this does NOT establish

- **No tolerances.** K3's anchor measured per-operator fp32 rounding floors and derived thresholds;
  this one has not, because there is no Glimmer kernel yet to need them. When S2 writes one, the
  floors have to be measured the same way before any threshold is chosen — a tolerance picked to
  make a kernel pass is not a tolerance.
- **No real weights.** Every number here comes from a deterministic draw at toy widths. The
  checkpoint's own arithmetic — bf16 accumulation, the real 6656/19968 shapes — is untouched.
- **Nothing about the conversion path.** The safetensors reduction is gated separately by
  `tests/glimmer_names.rs` against the vendored `model.safetensors.index.json`.
