---
scope: k3
status: data
verdict: The S1b anchor exists and runs — Kimi-K3's own first-party stack (modeling_kimi_linear.py at the pinned revision + fla-core 0.5.2 + transformers 4.56.2 + torch 2.13.0+rocm7.2) executed on gfx1151 at tiny widths but the REAL 93-layer structure, emitting 228 named activations; the decode golden is vendored at tests/k3-anchor-decode.bin (286 KiB, reproduced byte-for-byte on a later independent run) and read by tests/k3_anchor.rs with no GPU, no python and no network. Eleven defect runs, each GATED on the layers it must leave bit-identical rather than merely on having changed something: MlaLoraEps1e5 (the C reference's 1e-5 against first-party 1e-6) costs 2.2e-5 relative and reddens nothing before the first MLA layer; MlaScaleFromNope, KdaNoQkL2Norm, KdaGateLowerBoundOff, KdaStateLayout, KdaBetaSigmoidOutside, ExpertW1W3Swap, DenseMlpGateUpSwap, RouterBiasInWeight, LatentNormAfterUp and AttnResNormalisedValues each redden from their own first layer on. Four deviations are declared in the golden's metadata and pinned by the test: attention is eager (the reference's __init__ force-overwrites the field to flash_attention_2, so the driver overrides it after construction), the routed experts are plain nn.Linear (MXFP4 is anchored on real bytes by repack-one-expert.md instead), the text model runs without the vision wrapper, and the reference runs in fp32 while the checkpoint is bf16 — which is where S2's tolerance decision starts. KDA cannot run on CPU at all — fla's chunk_kda/fused_recurrent_kda are triton kernels and its pure-torch naive twin takes none of the seven kwargs the model passes — which is why the goldens are GENERATED on a GPU once and vendored rather than regenerated in a gate.
---

# The S1b anchor: goldens from Kimi-K3's own stack

**Measured 2026-08-11.** G1b calls this anchor *mandatory*, and G0 item 11 says why: four of the
twelve traps are **not attestable from the checkpoint**, because the KDA inner arithmetic lives in
`fla-core` and the MXFP4 unpack in `compressed-tensors`. Everything rivoli knows about those came
from reading a third-party C reference — and that reading has already been wrong once, on the MLA
LoRA-norm eps. A fixture derived from it would put the same misreading in the spec, in the golden,
and in the kernel scored against the golden. So the golden comes from *running* the reference.

## What runs

| | |
|---|---|
| reference | `moonshotai/Kimi-K3` @ `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` — `modeling_kimi_linear.py`, `configuration_kimi_k3.py` |
| entry point | `KimiLinearForCausalLM`, built from a tiny config **derived from the vendored `config.json`** |
| stack | torch 2.13.0+rocm7.2, transformers 4.56.2, fla-core 0.5.2, **pytorch-triton-rocm 3.5.1** |
| device | AMD Radeon 8060S (gfx1151), ROCm 7.2 |
| driver | `tests/k3_anchor_driver.py`; `tests/k3-anchor.sh` reproduces every run below |

The reference `.py` files are **not vendored** — same argument as `repack-one-expert.md`: the
revision is pinned, so the recipe is worth more than the copy. Their sha256 prefixes are written
into every golden's metadata and **pinned by `tests/k3_anchor.rs`** (`ref_modeling_sha256_16` =
`9e3564c70ac21854`, `ref_config_sha256_16` = `735eb9ebe593e17d`), so a golden regenerated against a
later revision fails rather than passing with a hash nothing compares. The vendored `config.json`
is byte-identical to the one at that revision (`9710e121a58d03ac`), re-downloaded and compared
2026-08-11.

> **CORRECTED 2026-08-11, same day.** The stack row said **triton 3.7.1**, read off a
> `pip install triton` line. Wrong package: `pytorch-triton-rocm` was force-reinstalled over it
> afterwards and is what runs, and the golden's own metadata says `3.5.1`. The venv holds both
> dist-infos, so which one `import triton` resolves is install-order dependent — the file is the
> authority, and the test now pins the value.

## Tiny widths, real structure

Depth is nearly free and structure is where the traps live; width is what costs. So **93 layers,
the real 1-based `kda_layers`/`full_attn_layers` partition (all 93 entries compared against the
vendored config), `first_k_dense_replace` 1, `attn_res_block_size` 12, `num_shared_experts` 2, both
`situ` betas, `gate_lower_bound` −5.0, `short_conv_kernel_size` 4** — every one inherited from the
real config and asserted twice: at generation time, and again from the file, with the driver
recording which fields it checked (`structural_asserted`) so the test cannot claim one the driver
skipped.

> **CORRECTED 2026-08-11 after review, and this one had teeth.** The first widths were
> `hidden_size` 128, `intermediate_size` 128, `moe_intermediate_size` 32,
> `routed_expert_hidden_size` 64, `kv_lora_rank` 16, `qk_nope_head_dim` 16 — and they made **four
> pairs accidentally equal that the real config separates**:
>
> | pair | tiny (first) | real |
> |---|---|---|
> | `kv_lora_rank` vs `qk_nope_head_dim` | 16 == 16 | 512 vs 128 |
> | `2 · moe_intermediate_size` vs `routed_expert_hidden_size` | 64 == 64 | 6144 vs 3584 |
> | `hidden_size` vs `intermediate_size` | 128 == 128 | 7168 vs 33792 |
> | `hidden_size` vs KDA `num_heads · head_dim` | 128 == 128 | 7168 vs 12288 |
>
> "Only widths shrink" was the stated premise and it was violated: **a width that makes two
> structurally distinct quantities equal deletes a structural distinction.** A port reading the KV
> latent width off `qk_nope_head_dim`, or the shared expert's width off the latent rather than
> `num_shared_experts · moe_intermediate_size` — the `[hidden, 2·moe_inter]` coupling this port has
> a recorded trap for, and which `tests/k3_anchor.rs` claimed to pin — produced a **bit-identical**
> fixture, not merely a shape-valid one. Found before any kernel was scored against it.

Current widths: hidden 7168→**192**, latent 3584→**96** (the ratio of 2 kept), `moe_intermediate_size`
3072→**24**, `intermediate_size` 33792→**256**, 896 experts→**8**, top-16→**top-2**, vocab
163840→**256**, MLA `(nope, rope, v)` `(128, 64, 128)`→**(16, 8, 16)**, `kv_lora_rank`
512→**24**, `q_lora_rank` 1536→**32**, KDA 96 heads→**4** at `head_dim` **32**.

Equalities the real config *does* have are kept: `qk_nope_head_dim == v_head_dim`,
`num_attention_heads == linear_attn_config.num_heads`, and `latent == hidden / 2`. `head_dim` stays
a power of two because fla's triton kernels block over K and V and refuse degenerate widths. One
collision remains on purpose: MLA's `kv_b_proj` output (128) equals the KDA projection width;
breaking it needs a non-power-of-two head count, and a port confusing MLA's KV expansion with KDA's
projection is confusing two different layer families. Token ids move too (bos/eos/pad
163584/163586/163839 → 250/251/255): `nn.Embedding` refuses a `padding_idx` outside
`num_embeddings`, and `pad` stays the last row so the reference's zeroing of it still applies.

## The four declared deviations

Each is in the golden's metadata and pinned by `tests/k3_anchor.rs`, because a golden that hides
what produced it is worse than no golden.

1. **Attention is `eager`** (`attn_implementation`). `KimiLinearModel.__init__` overwrites whatever
   `_attn_implementation` you pass with `flash_attention_2` and logs "Ignoring the provided
   attention implementation"; `KimiMLAAttention.forward` reads the field at call time, so the driver
   sets it back *after* construction. Eager is the semantics flash approximates, needs no second
   GPU-only kernel, and skips only the pad-then-slice of `value_states`, which is numerically
   neutral.
2. **Routed experts are plain `nn.Linear`** (`quantized=no`) — `quantization_config` is dropped, so
   nothing here is MXFP4. Not a gap: the unpack is anchored on **real bytes** by
   `repack-one-expert.md`, and a group-32 scale grid does not exist at `moe_inter` 24.
3. **No vision wrapper** (`entry_point=KimiLinearForCausalLM`, not
   `KimiK3ForConditionalGeneration`). rivoli refuses vision and the wrapper adds no text-side
   arithmetic. Recorded because nothing else in the file would distinguish the two.
4. **fp32, while the checkpoint is bf16** (`dtype=torch.float32`). Right for a reference — the point
   is to pin arithmetic, not to reproduce one accumulation order — but it is where S2's tolerance
   decision starts, so it is in the metadata. *(Added 2026-08-11 after review: three deviations were
   declared and this fourth was not.)*

## Why the goldens are vendored rather than regenerated

**KDA cannot run on CPU.** `chunk_kda`/`fused_recurrent_kda` are triton kernels; on a CPU tensor
triton raises `Pointer argument (at 0) cannot be accessed from Triton`, and with no GPU present at
all its driver refuses first (`0 active drivers`). fla ships a pure-torch `naive_recurrent_kda`,
and it is **not** a substitute: it takes none of `A_log`, `dt_bias`, `use_qk_l2norm_in_kernel`,
`use_beta_sigmoid_in_kernel`, `use_gate_in_kernel`, `safe_gate` or `lower_bound` — all seven moved
*inside* the kernel, which is precisely the arithmetic no document attests to. Substituting it would
mean hand-transliterating those, the exact act this anchor exists to avoid.

So the golden is generated on a GPU **once** and the bytes are vendored; `tests/k3_anchor.rs` reads
them with no device, no python and no network, and pins their length and an FNV-1a over them so a
regeneration is a deliberate, reviewed change rather than a silent one. `tests/k3-anchor.sh` ends by
`cmp`-ing its fresh `None` decode golden against the vendored file: on a later independent run it
printed **"vendored decode golden reproduced byte-for-byte"**. That is a statement about this GPU,
this driver and these versions, not a cross-machine contract.

**Only the decode golden is vendored** (292,781 B = 286 KiB, `tests/k3-anchor-decode.bin`). The
prefill golden is **1,299,802 B = 1.24 MiB** — every per-token tensor is eight times wider at
`--seq 8` — and its consumer does not exist: S2 item 5 defers chunked prefill outright ("no chunked
prefill exists in the reference"), so vendoring it now would be 1.24 MiB nothing reads. Its recipe
is `K3_ANCHOR_MODES=prefill tests/k3-anchor.sh`.

## What is captured

Six layers, chosen for what each one *is* — `0` (KDA, the only dense `mlp`, an attn-res block
start), `1` (first MoE layer), `3` (first MLA layer, 1-based 4), `12` (an attn-res block start that
is not layer 0), `91` and `92` (the two **adjacent** MLA layers the real map ends with) — plus the
model-level tail. Every submodule output of those layers, **223 float and 5 int tensors** in the
decode golden (220 + 5 in prefill, which has no incoming recurrent state), plus two operator
boundaries no forward hook can see:

* **KDA** — `q`/`k`/`v` after the short convolutions, the log-space gate, `beta`, `A_log`,
  `dt_bias`, the incoming recurrent state, and both outputs. Everything between those two points
  lives in fla's triton kernel and in no document. This IS the S2 KDA fixture.
* **AttnRes** — `prefix_sum`, the accumulated `block_residual` stack and the fold's output, for
  both per-layer folds and the model-level one.

> **AttnRes was missing entirely until review 2026-08-11**, and it is S2's *first* kernel.
> `_apply_attn_res` reads `proj.weight.squeeze(0)`, `norm.weight` and `norm.variance_epsilon`
> **inline** and never calls either module — and a `register_forward_hook` only fires from
> `Module.__call__`. So the four AttnRes modules per captured layer and the two model-level ones
> produced *nothing*, the golden held no `*_res_*` tensor at all, and three comments said the
> opposite. Three reviewers found it independently. The fold is now captured by wrapping the
> reference's free function, the way the KDA ops already were, and the driver **asserts that every
> registered hook fired at least once** — which is the check that would have caught it. That
> assertion earned itself on its first run, by catching five more dead hooks on the `experts`
> `ModuleList`.

Two things are deliberately not captured. The **individual routed experts** are excluded — not for
size but because `moe_infer` only calls experts that won tokens, so which expert modules fire is
routing-dependent, and any defect that moves the routing changes the golden's tensor *set*. The
first defect matrix reported `inf` for most layers on four of five defects for exactly that reason,
drowning the signal; `topk_idx`/`topk_weight` and the block output are the routing fixture and are
always present. The **prefill activations in decode mode** are skipped too: the warm-up pass runs
unhooked, since the prefill golden already holds them.

## The defect runs

`--defect` perturbs the reference after construction. **A layer downstream of a defect is expected
to redden** — this is one forward pass, so a perturbation at layer 3 reaches layer 92 by
construction. "Stays green elsewhere" can only mean *upstream*, and upstream is where the
localisation lives.

**The green cells are gated, not just recorded.** `--compare` fails if a defect changed nothing, if
it reddened any layer the driver's `EXPECT_GREEN` declares must stay bit-identical, or if the two
runs captured different tensors. Until 2026-08-11 the only automated check was "some cell is
non-zero", so a regression that broke the localisation would have printed a matrix nobody reads and
exited 0 — review found that, and it was the load-bearing half of §G rule 1.

Decode, `--seq 8`, one step. `differing` is out of the layer's captured tensor count; `max_rel` is
the **row maximum** of `max|a−b|` over the tensor's own scale:

| defect | 0 | 1 | 3 | 12 | 91 | 92 | model | max_rel |
|---|---|---|---|---|---|---|---|---|
| `MlaLoraEps1e5` | **0/38** | **0/47** | 20/30 | 43/47 | 29/30 | 29/30 | 6/6 | 2.2e−5 |
| `MlaScaleFromNope` | **0/38** | **0/47** | 16/30 | 43/47 | 30/30 | 29/30 | 6/6 | 1.3e+0 |
| `ExpertW1W3Swap` | **0/38** | 4/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 3.0e+0 |
| `RouterBiasInWeight` | **0/38** | 5/47 | 26/30 | 44/47 | 30/30 | 30/30 | 6/6 | 2.4e+0 |
| `LatentNormAfterUp` | **0/38** | 4/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 2.0e+2 |
| `DenseMlpGateUpSwap` | 6/38 | 42/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 2.3e+0 |
| `AttnResNormalisedValues` | 8/38 | 42/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 1.8e+0 |
| `KdaNoQkL2Norm` | 15/38 | 42/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 4.2e+1 |
| `KdaGateLowerBoundOff` | 15/38 | 42/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 3.3e+0 |
| `KdaStateLayout` | 15/38 | 42/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 3.3e+0 |
| `KdaBetaSigmoidOutside` | 15/38 | 42/47 | 27/30 | 44/47 | 30/30 | 30/30 | 6/6 | 7.0e+0 |

Every bold cell is asserted. Read the rows:

* **`MlaLoraEps1e5`** — MLA's two LoRA norms at the C reference's 1e-5 instead of first-party's
  1e-6 (they are constructed *without* `config.rms_norm_eps`, which is 1e-5, so `KimiRMSNorm`'s own
  1e-6 default is the right value). Layers 0 and 1 are **bit-identical**: they are KDA, upstream of
  the first MLA layer. The cost is **2.2e-5 relative** — real, detectable, and far below anything a
  tolerance-based fixture would have flagged. This is the divergence G0 item 11 found, priced.
* **`MlaScaleFromNope`** — softmax scale over `qk_nope_head_dim` instead of `q_head_dim`, trap 8.
  The rope dims NoPE never rotates are still scored, so they still count in the scale. Same green
  cells as the eps defect and five orders of magnitude louder, which gives the MLA fixture a second
  witness at a completely different magnitude.
* **`ExpertW1W3Swap`** — gate and up swapped in every routed expert, the one repack error that is
  byte-clean; `repack-one-expert.md` could only pin that `w2` is not in the up slot, and this is the
  numerical oracle `V4_PROJ`'s doc says is needed. Layer 0 untouched — it has no routed experts —
  and layer 1 reddens 4 of 47: its MoE block and what follows, not its attention.
* **`RouterBiasInWeight`** — takes the routing weight from the biased score instead of the unbiased
  one (trap 6). Layer 1 reddens 5 of 47 at 3.5e-2, and `topk_idx` does **not** move: the bias steers
  selection only, so a wrong weight is a small, silent error. The most easily-missed of the eleven.
* **`LatentNormAfterUp`** — RMSNorm the latent sandwich's output *after* the up projection instead
  of the aggregate before it. Shape-valid in both directions because a norm is width-generic, and
  the **loudest defect in the set at 2.0e+2**.
* **`DenseMlpGateUpSwap`** — the same gate/up swap in the single dense layer, so a defect exists
  that touches **layer 0 alone** at first: 6 of 38 there, everything after by propagation.
* **`AttnResNormalisedValues`** — mixes the *normalised* sources instead of the raw concatenation.
  `_apply_attn_res` normalises `v` only to score it and then mixes `v_float`; mixing `k` is a
  one-character slip that leaves every shape intact. Reddens 8 of layer 0's 38 — layer 0's MLP fold
  is the first one that runs, since its `block_residual` starts empty.
* **The four KDA kwargs** — `use_qk_l2norm_in_kernel`, `lower_bound` (the −5.0 gate bound, trap 4:
  it *multiplies* the sigmoid rather than clamping), the stored state's (K,V)-vs-(V,K) axis order,
  and `use_beta_sigmoid_in_kernel`. These are the arithmetic that exists only inside fla's kernel,
  and each reddens 15 of layer 0's 38 tensors while leaving the 23 upstream of the KDA op alone —
  the earliest a defect can localise here. Without these runs nothing showed the goldens were
  sensitive to that arithmetic at all; a kernel that omitted the gate bound entirely could have
  matched.

`KdaStateLayout` transposes the state the wrapper *returns* rather than flipping
`transpose_state_layout`, which is the kwarg that names the choice. That was the first
implementation and it was abandoned on measurement: with the kwarg flipped, fla went into triton for
**25 minutes without finishing** against ~30 s for every other defect, writing new cache entries
throughout — it asks for a kernel variant nobody will ship, and a defect run that costs half an hour
is one nobody re-runs. The state's axis order is invisible to any shape assertion, because
`head_k_dim == head_dim` in the tiny model *and* in the real one (128 == 128), so only its values
can carry it.

Prefill (`--seq 8`, `chunk_kda`) gives the same green cells with the same localisation. Row maxima
differ as expected — `KdaNoQkL2Norm` reaches 4.8e+1 and `MlaLoraEps1e5` 3.8e-5, both larger because
the chunked path compounds over eight positions — and `KdaStateLayout` is *smaller* at 1.2e+0, since
in prefill there is no incoming state for the transposition to have corrupted.

G1b's coverage classes are met by this table: a KDA layer (0, 1, 12), an MLA layer (3, 91, 92),
layer 0 (`DenseMlpGateUpSwap`), and layer 92 (every row).

## The harness is red-proved too

`tests/k3_anchor.rs` was proved able to go red both ways on 2026-08-11: truncating the vendored
golden to 100,000 bytes failed every test at load ("failed to fill whole buffer", exit 101), and
mutating one assertion in each test — expecting `flash_attention_2`, `num_shared_experts` 3,
`dt_bias` `[64]`, and a five-layer capture list — failed exactly those and nothing else.

Two harness-side guards were added the same day and both fired on their first real run, which is the
only evidence that matters for a guard: the every-hook-fired assertion caught five dead hooks on the
`experts` `ModuleList`, and `build.rs`'s jscpd gate rejected the two config-check loops at 35 tokens.
That duplication was written into them from the start and only crossed jscpd's 15-token floor once
`cargo fmt` broke their one-liner `assert!`s across four lines each — the formatter made it visible,
it did not create it, and the fix was to factor the loops into one function. (CLAUDE.md said the
opposite until 2026-08-11 and is corrected in place.)

**What this does NOT reject.** `tests/k3_anchor.rs` is a fixture-*integrity* gate: it compares no
rivoli output to the golden, because at S1b there is no K3 kernel to score. G1b's remaining owed
item is the per-operator **tolerances** — the anchor gives exact bytes from one implementation, in
fp32, on one GPU, and what a HIP kernel may differ by is undecided.

One residual limit, disclosed rather than fixed: the vendored golden is **one salt and one
position**. A kernel bug degenerate at these particular values — a zero crossing in `beta`, a tie in
the top-2 selection — is invisible to it. The cheap mitigation, when S2 needs it, is a second decode
golden at a different `--salt`.

## Re-running it

```bash
python3 -m venv venv && venv/bin/pip install \
    --index-url https://download.pytorch.org/whl/rocm7.2 torch pytorch-triton-rocm
venv/bin/pip install transformers==4.56.2 einops safetensors fla-core
R=9f62e4e9fffbd0a83ddd60e1c209d828994b3569
for f in configuration_kimi_k3.py modeling_kimi_linear.py; do
    curl -sSLO "https://huggingface.co/moonshotai/Kimi-K3/resolve/$R/$f"; done
K3_ANCHOR_VENV=$PWD/venv K3_ANCHOR_REF=$PWD/ref tests/k3-anchor.sh
```

The install is ~6.2 GB of ROCm wheels and does not belong in `/tmp` — that is 63 GB of tmpfs *in
RAM* on this box, and filling it shrinks `--max-mem auto` silently before anything fails loudly.
`--device cpu` gets a config, weight-init or defect-injection error out of the driver in seconds
without taking the GPU lock; it cannot produce a golden, which is the point.
