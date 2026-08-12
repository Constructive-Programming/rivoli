---
scope: k3
status: data
verdict: The S1b anchor exists and runs, and its per-operator TOLERANCES are measured — Kimi-K3's own first-party stack (modeling_kimi_linear.py at the pinned revision + fla-core 0.5.2 + transformers 4.56.2 + torch 2.13.0+rocm7.2) executed on gfx1151 at tiny widths but the REAL 93-layer structure. TWO independent weight draws are vendored (tests/k3-anchor-decode-k3-anchor-{1,2}.bin, 332,009 B each, each reproduced byte-for-byte on a later run) and read by tests/k3_anchor.rs with no GPU, no python and no network; eleven defect runs are scored against both and each is GATED on the layers it must leave bit-identical. THE TOLERANCE FINDING: MLA is exact-only, because the C reference's LoRA-norm eps moves that operator by 1.92e-5 while its own fp32 rounding floor is 5.74e-5 — a margin of 0.33x — the eps sits BELOW the floor — so no threshold admits a correct kernel and rejects that eps, and the eps must be pinned structurally instead. Downstream it sits at 0.3-0.9x the floor, i.e. BELOW the reference's own rounding error, so a tolerance-based fixture could not have seen it anywhere — which is why this anchor is exact bytes. Every other operator has 16,000x to 3.3M x of room and is set at 10x its floor, where the floor is the MAX over both draws: attn_res 7.1e-4, mla_attend 4.1e-4, moe_latent 6.3e-4, moe_route 6.0e-4, kda_op 6.3e-4, dense_mlp 9.4e-6. Floors come from running the same reference in fp64 with all 276 fla modules held at fp32 (a plain model.double() dies in triton), except kda_op's, which is chunk_kda against fused_recurrent_kda over 69 layers because fla returns an fp32 recurrent state whatever the input dtype. Four deviations are declared in the metadata and pinned by the test: eager attention, unquantized experts, no vision wrapper, fp32 against the checkpoint's bf16.
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

That check **enumerates the vendored files and prints a census** (`verified: N of M`), rather than
looping over the salts it was asked to run. Driving it from `SALTS` meant a narrowed run —
`K3_ANCHOR_SALTS=k3-anchor-1`, which halves a ~25 min GPU-locked regeneration and is a reasonable
thing to want — printed one cheerful "reproduced byte-for-byte" and exited 0 while the other
vendored golden went unchecked, output indistinguishable from a full pass. The skip is legitimate;
being quiet about it was the defect. A run that reproduces **nothing** now fails outright, since
that is what an empty `$OUT` or a mistyped `--out` looks like from outside.

**Only the decode golden is vendored** (332,009 B = 324 KiB each, `tests/k3-anchor-decode-k3-anchor-1.bin`
and `-2.bin`). The
prefill golden is **1,299,802 B = 1.24 MiB** — every per-token tensor is eight times wider at
`--seq 8` — and its consumer does not exist: S2 item 5 defers chunked prefill outright ("no chunked
prefill exists in the reference"), so vendoring it now would be 1.24 MiB nothing reads. Its recipe
is `K3_ANCHOR_MODES=prefill tests/k3-anchor.sh`.

## What is captured

Six layers, chosen for what each one *is* — `0` (KDA, the only dense `mlp`, an attn-res block
start), `1` (first MoE layer), `3` (first MLA layer, 1-based 4), `12` (an attn-res block start that
is not layer 0), `91` and `92` (the two **adjacent** MLA layers the real map ends with) — plus the
model-level tail. Every submodule output of those layers, **272 float and 5 int tensors** in the
decode golden, plus the operator boundaries no forward hook can see:

> **CORRECTED 2026-08-12.** This line said **235**, which was true for about a day. Item 2 added 27
> captures and item 3 ten more without it being touched, so the file understated its own contents by
> 37 while `tests/k3_anchor.rs` asserted the real number — the count is gated there, and was right
> throughout. The arithmetic, so the next addition has somewhere to attach: 223 originally, +12
> `.fold` (item 1), +21 MLA attention-core and +6 `o_proj.in_gated` (item 2), +10 latent-norm input
> and weight (item 3). **Prefill's count is no longer restated here**: it was given as "232 + 5" and
> that too was a snapshot of one day's driver. It differs from decode by the incoming recurrent
> state, which is the durable fact.

* **KDA** — `q`/`k`/`v` after the short convolutions, the log-space gate, `beta`, `A_log`,
  `dt_bias`, the incoming recurrent state, and both outputs. Everything between those two points
  lives in fla's triton kernel and in no document. This IS the S2 KDA fixture.
* **The MLA attention CORE** — `q`, `k`, `v`, the additive `mask`, the **`scaling` as a value**, the
  output and the softmax `probs`, for each of the three captured MLA layers. `eager_attention_forward`
  is a module-level free function, so no hook fired for it and the golden held the projections
  around the attention with nothing from inside it. Added 2026-08-11 for S2 item 2 — see below.
* **AttnRes** — `prefix_sum`, the accumulated `block_residual` stack, **the fold** and the fold's
  output, for both per-layer folds and the model-level one. Twelve folds: two per captured layer
  plus the model-level one, minus layer 0's `self_attention_res`, which §3's layer loop guards on a
  non-empty block stack.

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
>
> **CORRECTED 2026-08-11, same day, by S2 item 1 trying to use it.** The fix above captured
> `in.prefix_sum`, `in.block_residual` and `out` — and **not the scoring vector**, which left the
> fixture unusable for the exact purpose it had just been created for. AttnRes is the one operator
> here whose inputs do not determine its output: `out` is
> `softmax(<RMSNorm(v), norm.weight * proj.weight>) @ v`, and neither weight was in the file, so
> there was no way to get from the captured inputs to the captured output. The kernel could not be
> scored against it. `wrap_attn_res` now also captures `{tag}.fold`, taking the golden from 223 to
> **235** float tensors and forcing a re-vendor.
>
> The **product**, not the two factors. `fold[i] = norm[i] * proj[i]` is a load-time collapse the
> port does in its loader, so the kernel never sees the factors; a fixture carrying them would be
> scoring an elementwise multiply no kernel performs. The eps is deliberately NOT captured — it is
> `config.rms_norm_eps`, which `the_tiny_configs_kept_the_real_structure` already pins against the
> real checkpoint, and `tests/k3_kernels.rs` reads it from the golden's own `tiny_config` rather
> than from a literal.
>
> The generalisable form: **a fixture is only usable if its captures span the operator's whole
> boundary**, and "inputs and outputs" is not that boundary whenever a weight sits between them.
> Three reviewers found the missing hook; none noticed the fixture it produced could not be used,
> because nothing had tried to use one yet.

### The same gap, twice more, found by S2 item 2 (2026-08-11)

**The MLA attention core had no fixture at all**, and it hid the three traps §5 spends most of its
words on. `eager_attention_forward` is a module-level free function — `KimiMLAAttention.forward`
resolves it from module globals at call time — so `register_forward_hook` never fired and the golden
carried the projections either side of the attention and nothing from within it. No scores, no
softmax, no pre-gate output. Unscoreable: the softmax scale over the full 192 rather than
`qk_nope`; the unrotated rope dims still being scored, which §5 names "the silent bug"; and
causality, which lives entirely in the additive mask.

Fixed by `wrap_mla_attention`, the same setattr `wrap_attn_res` uses. **`scaling` is captured as a
one-element tensor rather than left in metadata**, which is the difference between a trap a fixture
reads and one a reader has to remember: 0.20412 = 1/sqrt(24) against 0.25 for the nope-only
misreading.

**And the gate sat in a gap one level down.** `o_proj`'s OUTPUT was captured, its INPUT was not — so
trap 10 (MLA gates with no norm; KDA norms then gates) had nothing on either side of it, and a port
that normed before gating would have matched every tensor in the file. A
`register_forward_pre_hook` on `o_proj` takes the input; a forward hook cannot, because it hands
you the output. It fires on the KDA layers too, which is better than intended — both sides of the
contrast are now in one file.

> **A wrong assumption caught by its own fixture.** The first version of
> `the_gate_ordering_is_the_one_mla_uses` asserted that a KDA layer has no output-gate projection.
> It went red immediately: KDA has a `g_proj` too, 128 wide. Trap 10 is not "one gates and the other
> does not" — **both gate, and the difference is the order**. KDA carries an `o_norm` and normalises
> first; MLA has no norm on that path at all, and the presence or absence of `o_norm` is the
> contrast that is actually in the file. What no fixture here can reach is KDA's *intermediate*:
> its norm and gate are fused inside fla's `FusedRMSNormGated`, so proving the order directly is
> S2 item 5's, on the KDA operator boundary.

**The generalisation, now that it has happened three times:** any operator whose input is a DERIVED
value rather than another module's captured output has no fixture, and a forward hook cannot give
it one. Free functions need a setattr wrap; module inputs need a pre-hook. Both were found the same
way — by trying to write the kernel and discovering there was nothing to launch with.

### A fourth time, and the generalisation predicted it (2026-08-12, S2 item 3)

**The latent sandwich's aggregate.** §6's order is `down(x)` → experts in latent space → RMSNorm the
**aggregate** → `up(...)`, and the golden held the three module outputs and nothing between them.
`routed_expert_norm` is fed `moe_infer`'s return — a method call, not a module call — and `.experts`
is unhooked on purpose, so the one operator in this sandwich whose arithmetic is neither a plain
matmul nor shared with another model had **no input at all**. Its weight went missing with it, for
the reason `.fold` did: an input and an output do not determine an operator when a weight sits
between them. `wrap_latent_sandwich` captures both, by pre-hook, +10 tensors.

Two inputs are deliberately **not** captured, both because the file already holds them:
`routed_expert_up_proj`'s is `routed_expert_norm`'s output, and `routed_expert_down_proj`'s is
`post_attention_layernorm`'s (reference `:964-966`, `:1035-1037` — the block is called on the normed
hidden state and on nothing else). A second copy under a second name reads as corroboration and is a
tautology.

**The two projection WEIGHTS are also not captured, and that is a decision.** At `[96,192]` and
`[192,96]` they are 36,864 floats per MoE layer against the whole golden's ~70,000, so capturing
them at the five MoE layers would roughly quadruple both vendored files. What they would buy is an
anchor-scored GEMV, and that comparison is weak where it is not free: rivoli's trunk matmul is
`gemm_bf16`, whose weights are **bf16**, while this reference runs fp32 — one of the four declared
deviations at the top of this file. A bf16 weight is ~2⁻⁹ off its fp32 twin, **1.95e−3 against
`moe_latent`'s 2.9e−4 tolerance**, so the fixture could only be stated seven times looser than the
operator's own bar, and it would still be at hidden 192 rather than 7168.
`tests/k3_kernels.rs::the_trunk_gemv_matches_an_f64_dot_at_k3_widths` scores that kernel against an
f64 dot at the **real** widths instead, which is both cheaper and wider. Revisit this only if
something needs the reference's particular matrix.

> **This is the first item where the anchor's fp32-vs-bf16 deviation bit**, and it is worth stating
> as a rule for the rest of S2: **an operator whose rivoli kernel rounds to bf16 cannot be scored
> against this anchor at the anchor's own tolerance.** Items 1 and 2 were unaffected because
> `attn_res` and `mha_attend` are f32 throughout. Item 3 is affected twice — once for the GEMVs
> above, and once for the norm, which is why it uses `linalg.hip::rmsnorm_single` (f32 store) and
> not `mla.hip::rmsnorm_batch` (bf16 store, V4's `RMSNorm.forward`). `k3-port.md` item 3 named the
> batch kernel on the grounds that it is already width-generic; it is, and the width was never the
> problem.

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

A second review the same day found **both directions of that gate still had a hole**, and they are
closed. A declared-green layer that was never CAPTURED scored as green, because an absent layer read
as zero differing tensors — so dropping a layer from `CAPTURE_LAYERS`, or naming one outside it,
would have rested the localisation claim on an empty set. And the positive half was only "something,
somewhere, differs": nothing asserted the first captured layer PAST the green boundary actually
reddened, so a perturbation that missed its operator and merely disturbed something downstream read
as a localised, detected defect while the arithmetic the cell prices went unexercised. Both are
asserted now, and all eleven recorded rows satisfy the second (checked against the matrix above
before it was added, so it gates rather than merely passing). The gate arms were exercised without a
GPU by rebuilding synthetic defect goldens from the vendored bytes through the driver's own reader
and writer — which is how a crash in the new arm was found: `per` also holds the model-level `model`
key, and sorting the whole key set by `int` raises on it.

Decode, `--seq 8`, one step. `differing` is out of the layer's captured tensor count; `max_rel` is
the **row maximum** of `max|a−b|` over the tensor's own scale:

> **Denominators re-derived 2026-08-11** when S2 item 1 added the twelve `.fold` captures: 38→39,
> 47→49, 30→32, 6→7, summing to the +12. **Every numerator is unchanged**, and that is the useful
> half — the folds are weights no defect perturbs, so they land green everywhere and no defect's
> localisation moved. Confirmed structurally as well: the regenerated goldens are bit-identical to
> the old ones on all 223 pre-existing captures and on every metadata field, so the driver change
> is provably additive rather than merely believed to be.
>
> **Scope of that re-derivation: salt-1 decode only.** The salt-2 and both prefill matrices were
> not re-run — the regeneration lost the GPU flock at 19 of 48 runs (`-E 66`, 900 s) to another
> tenant. Their numerators are unaffected by the same argument, but their denominators as printed
> by a fresh `--compare` will be the grown ones, not the numbers a reader would derive from this
> table. Re-run `tests/k3-anchor.sh` when the device is free and this note goes away.

| defect | 0 | 1 | 3 | 12 | 91 | 92 | model | max_rel |
|---|---|---|---|---|---|---|---|---|
| `MlaLoraEps1e5` | **0/40** | **0/50** | 26/40 | 44/50 | 35/40 | 35/40 | 6/7 | 2.2e−5 |
| `MlaScaleFromNope` | **0/40** | **0/50** | 20/40 | 44/50 | 37/40 | 36/40 | 6/7 | 1.3e+0 |
| `ExpertW1W3Swap` | **0/40** | 4/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 3.0e+0 |
| `RouterBiasInWeight` | **0/40** | 5/50 | 32/40 | 45/50 | 36/40 | 36/40 | 6/7 | 2.4e+0 |
| `LatentNormAfterUp` | **0/40** | 4/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 2.0e+2 |
| `DenseMlpGateUpSwap` | 6/40 | 43/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 2.3e+0 |
| `AttnResNormalisedValues` | 8/40 | 43/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 1.8e+0 |
| `KdaNoQkL2Norm` | 16/40 | 43/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 4.2e+1 |
| `KdaGateLowerBoundOff` | 16/40 | 43/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 3.3e+0 |
| `KdaStateLayout` | 16/40 | 43/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 3.3e+0 |
| `KdaBetaSigmoidOutside` | 16/40 | 43/50 | 33/40 | 45/50 | 36/40 | 36/40 | 6/7 | 7.0e+0 |

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

  > **A review asked to delete this one, 2026-08-11, and the argument is good enough to record
  > rather than dismiss.** Its green set is identical to `MlaLoraEps1e5`'s, it sets no tolerance
  > ceiling (1.3e+0 against that defect's 2.22e-5), and "a second witness at a different magnitude"
  > is undercut by MLA turning out to be scored **bit-exactly**, where magnitude does not enter. It
  > was kept because deleting it is only free if its 21 reddened tensors at layer 3 are a SUBSET of
  > the eps defect's 27 — the matrix above shows counts, not sets, so nothing here proves that, and
  > a defect whose red set is merely smaller still localises something the other does not. The check
  > is `--compare` on the two defect goldens against each other rather than against `None`, and it
  > needs a GPU because only the `None` goldens are vendored. **Do that before cutting it.**
* **`ExpertW1W3Swap`** — gate and up swapped in every routed expert, the one repack error that is
  byte-clean; `repack-one-expert.md` could only pin that `w2` is not in the up slot, and this is the
  numerical oracle `V4_PROJ`'s doc says is needed. Layer 0 untouched — it has no routed experts —
  and layer 1 reddens 5 of 52: its MoE block and what follows, not its attention.
* **`RouterBiasInWeight`** — takes the routing weight from the biased score instead of the unbiased
  one (trap 6). Layer 1 reddens 6 of 52 at 3.5e-2, and `topk_idx` does **not** move: the bias steers
  selection only, so a wrong weight is a small, silent error. The most easily-missed of the eleven.
* **`LatentNormAfterUp`** — RMSNorm the latent sandwich's output *after* the up projection instead
  of the aggregate before it. Shape-valid in both directions because a norm is width-generic, and
  the **loudest defect in the set at 2.0e+2**.
* **`DenseMlpGateUpSwap`** — the same gate/up swap in the single dense layer, so a defect exists
  that touches **layer 0 alone** at first: 6 of 40 there, everything after by propagation.
* **`AttnResNormalisedValues`** — mixes the *normalised* sources instead of the raw concatenation.
  `_apply_attn_res` normalises `v` only to score it and then mixes `v_float`; mixing `k` is a
  one-character slip that leaves every shape intact. Reddens 8 of layer 0's 38 — layer 0's MLP fold
  is the first one that runs, since its `block_residual` starts empty.
* **The four KDA kwargs** — `use_qk_l2norm_in_kernel`, `lower_bound` (the −5.0 gate bound, trap 4:
  it *multiplies* the sigmoid rather than clamping), the stored state's (K,V)-vs-(V,K) axis order,
  and `use_beta_sigmoid_in_kernel`. These are the arithmetic that exists only inside fla's kernel,
  and each reddens 16 of layer 0's 40 tensors while leaving the 24 upstream of the KDA op alone —
  the earliest a defect can localise here. Without these runs nothing showed the goldens were
  sensitive to that arithmetic at all; a kernel that omitted the gate bound entirely could have
  matched.

> **CORRECTED 2026-08-11, and it is the useful correction of this round.** The bullet above and
> `defect_kda_gate_lower_bound_off`'s docstring both said the gate bound's *form* — multiply, not
> clamp — is arithmetic "nothing outside fla's kernel attests to". **fla's own docstring attests to
> it**, at `fla/ops/kda/chunk.py:250-256`, and gives both forms explicitly: with `lower_bound` set
> and `safe_gate=True` the activation changes from `-exp(A_log) * softplus(g + dt_bias)` to
> `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, "which naturally clamps the output to
> `[lower_bound, 0)`". So **S2 has a written reference for this term and should port from it**,
> rather than inferring the shape of the expression from a red cell. What the defect run still buys
> is unchanged and still needed: proof that the golden is *sensitive* to the term, since a docstring
> can be stale in a way bytes cannot.
>
> Two facts from the same source that a port needs. **The defect cannot isolate `lower_bound`**: fla
> raises `ValueError` unless `lower_bound` is set whenever `safe_gate=True and use_gate_in_kernel`
> (`chunk.py:394`), so dropping the bound forces `safe_gate=False` in the same run and the cell
> attests to the PAIR. A kernel that got the bound right and the clamp wrong is not distinguished
> here. And fla range-checks `-5 <= lower_bound < 0`, so the real config's **−5.0 sits exactly on
> the inclusive end of its own safe range** — a port that treats the bound as exclusive, or that
> nudges it for stability, is out of the range fla will accept.

`KdaStateLayout` transposes the state the wrapper *returns* rather than flipping
`transpose_state_layout`, which is the kwarg that names the choice. That was the first
implementation and it was abandoned on measurement: with the kwarg flipped, fla went into triton for
**25 minutes without finishing** against ~30 s for every other defect, writing new cache entries
throughout — it asks for a kernel variant nobody will ship, and a defect run that costs half an hour
is one nobody re-runs. The state's axis order is invisible to any shape assertion, because
`head_k_dim == head_dim` in the tiny model *and* in the real one (128 == 128), so only its values
can carry it.

> **MEASURED 2026-08-12 by S2 item 5a, which had to know: the state these goldens carry is
> `[value][key]`.** `transpose_state_layout=True` names the choice and this is what it does. Scoring
> both interpretations of `in.initial_state` against `out.o` through the recurrence separates them by
> six orders — 2.5e-7 with the transpose against 2.2e-1 to 5.6e-1 without, unanimous over layers 0, 1
> and 12 at both draws. So the kwarg's effect is now attested by the bytes and not only by its name,
> which is what `KdaStateLayout` was for; and the ambiguity was real rather than theoretical, since
> `k3-architecture.md` §4 writes the recurrence as `S[key][value]` and a port reading that as the
> stored layout gets an order-1 error with every shape check green. **rivoli's kernel keeps
> `[key][value]` and transposes in the fixture instead** — the reason is coalescing and it is argued
> at `kernels/recurrent.hip`.

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
rivoli output to a golden, because at S1b there is no K3 kernel to score.

## The per-operator tolerances

**Measured 2026-08-11, not chosen.** The table lives in `tests/common/k3_tolerance.rs` so S2's kernel
tests and the anchor's own gate share one copy, and each row's *policy* is derived from two measured
numbers rather than written down. `tests/k3_anchor.rs` fails if a row's numbers stop supporting its
policy — which is what stops a tolerance being widened the first time a kernel disagrees.

**The floor: what a correct implementation cannot beat.** Run the identical reference at double
precision and diff it against the fp32 run; the difference is the fp32 run's own rounding error, and
an independent correct kernel in fp32 that associates its sums differently lands in the same
neighbourhood.

```bash
tests/k3_anchor_driver.py --mode decode --defect None --dtype float64 --out fp64.bin ...
tests/k3_anchor_driver.py --by-operator fp64.bin tests/k3-anchor-decode-k3-anchor-1.bin
```

`--dtype float64` is not a mode anyone ships; it exists to produce that number. It required work:
**a plain `model.double()` dies inside triton** with `fp_downcast_rounding should be set only for
truncating fp conversions`, and not only in the KDA op — `ShortConvolution` and `FusedRMSNormGated`
are fla modules too, and all three refuse fp64. So the run holds **276 fla modules at fp32** while
the rest of the model is double. What survives is a genuine fp64 reference for AttnRes, MLA, the
latent sandwich, SiTU/MoE, the norms and the head: four of S2's five items.

**KDA needs its own floor**, and would have even if fp64 had worked: the kernel returns an **fp32
recurrent state whatever the input dtype**. So its floor is the disagreement between the two paths
fla itself ships for the same recurrence — `chunk_kda` over eight positions at once against
`fused_recurrent_kda` one position at a time, same weights, same tokens, worst of all **69** KDA
layers:

```bash
tests/k3_anchor_driver.py --mode kda-equiv ...     # writes nothing; prints one number per layer
```

**The ceiling: the weakest defect the tolerance must still catch.** Per operator, the smallest signal
among the defect runs that *target* it — another operator's defect leaking downstream is not what
this operator's tolerance is for.

**This table is the ONE-DRAW reading and FIVE of its rows have since moved.** `mla` was re-measured
across both draws and split in two when item 2 captured the attention core; **`attn_res`,
`moe_latent` and `moe_route` were left one-draw until item 3 re-ran the fp64 island on draw 2
(2026-08-12) and found every operator's floor LARGER there, by 2.5-5×.** The superseding numbers are
in *"Re-measured on both draws"* below, and `tests/common/k3_tolerance.rs` is what the tests read.
Kept here because the change — a 1.3× margin becoming 0.33× — is the finding, and deleting the
before makes the after unreadable.

| operator | one-draw floor (below) | draw 2 | now, and policy |
|---|---|---|---|
| `attn_res` | 1.571e−5 | **7.052e−5** | 7.052e−5, `Rel(7.1e-4)` |
| `moe_latent` | 2.851e−5 | **6.287e−5** | 6.287e−5, `Rel(6.3e-4)` |
| `moe_route` | 2.472e−5 | **5.956e−5** | 5.956e−5, `Rel(6.0e-4)` |

`dense_mlp` was already the max and `kda_op`'s floor is a different measurement. **The
`weakest own defect` column was right by luck** — it wants the MIN over draws, and draw 1 gave the
smaller signal on every row, so the one-draw habit understated the floors and left the defects
correct. Checked rather than assumed. And **not caused by item 3's new captures**: the `moe_latent`
floor is 2.851e−5 / 6.287e−5 whether or not the two new tensors are in the bucket, measured both
ways.

| operator | fp32 floor | weakest own defect | margin | policy |
|---|---|---|---|---|
| `attn_res` | 1.57e−5 | 1.80e+0 `AttnResNormalisedValues` | 114,000× | `Rel(1.6e-4)` |
| **`mla`** *(superseded)* | **1.70e−5** | **2.22e−5 `MlaLoraEps1e5`** | **1.3×** | **exact only** |
| `moe_latent` | 2.85e−5 | 2.05e+2 `LatentNormAfterUp` | 7.2M× | `Rel(2.9e-4)` |
| `moe_route` | 2.47e−5 | 2.23e+0 `RouterBiasInWeight` | 90,000× | `Rel(2.5e-4)` |
| `kda_op` | 6.30e−5 | 1.75e+0 `KdaBetaSigmoidOutside` | 27,700× | `Rel(6.3e-4)` |
| `dense_mlp` | 9.37e−7 | 1.28e+0 `DenseMlpGateUpSwap` | 1.4M× | `Rel(9.4e-6)` |

**MLA is the finding.** The C reference's LoRA-norm eps moves that operator by 2.22e−5 while the
operator's own fp32 rounding floor is 1.70e−5 — a margin of **1.3×**. There is no threshold that
admits a correct kernel and rejects that eps, so **the eps cannot be settled numerically at all.**
Two consequences, both for S2/S3:

* MLA's fixture is scored **bit-exactly**, and `Policy::ExactOnly` says so in code.
* The eps has to be pinned **structurally** — read the constant and assert it — because no amount of
  numerical care will detect it. `KimiMLAAttention` constructs both LoRA norms *without* passing
  `config.rms_norm_eps`, so the right value is `KimiRMSNorm`'s own 1e-6 default, and that is a fact
  about the source rather than about any output.

It is also the retrospective justification for this anchor being exact bytes: **had S1b shipped
tolerance-based fixtures, the divergence G0 item 11 found would have been invisible to its own
gate.** Measured downstream, the same defect sits at 0.3–0.9× the floor on every other operator —
i.e. *below* the reference's own rounding error — so nothing outside MLA could have seen it either.

Every other operator has four to seven orders of magnitude of room, so the tolerances are set at
10× the floor and the gate requires 30× of clearance under the defect. Three red-proofs on
2026-08-11: marking `mla` as a `Rel` fails ("no Rel tolerance is defensible"), setting a tolerance
within 30× of its defect fails, and setting one below its own floor fails.

> **CORRECTED 2026-08-11 by review: the gate on this table could not express most of the ratio
> line.** The constants were `tol >= floor*9.9`, `tol <= defect/30`, and `ExactOnly` iff
> `margin < 3.0` — but a `Rel` value satisfying the first two exists only when
> `margin >= 9.9*30 = 297`, so **every margin in `[3.0, 297)` had no admissible policy at all**, and
> the two error messages each told the author to do what the other refused. An operator at floor
> 1e-5 and defect 1e-3 (margin 100) needed `t >= 9.9e-5` and `t <= 3.33e-5` simultaneously. Nothing
> was in that band — `mla` is 1.31× and the rest above 27,000× — so the gate was green while being
> unusable for the next operator measured, which is precisely what the paragraph below tells S2 to
> do. The `ExactOnly` boundary is now DERIVED from the other two constants rather than written by
> hand, so the branches partition the line with no gap. Two smaller ones from the same review: the
> tolerances are 10× the floor **within two-significant-figure rounding** (they run 9.998× for
> `kda_op` to 10.185× for `attn_res`, because each is written to 2 s.f. against a floor recorded to
> 4) — an undocumented `9.9` was the only thing admitting `kda_op` while four comments claimed a
> flat 10× — and the upper side of that rule was unbounded, so `Rel(5.0e-2)` on `attn_res`, 3183×
> its floor, passed. Both are bounded and stated now.

> **RE-MEASURED 2026-08-11 for S2 item 2, and one row changed materially.**
>
> The attention core is now captured, so `operator_of` splits **`mla_attend`** out of `mla`: they
> answer different questions. `mla` covers the projections and the LoRA norms, where the eps
> divergence lives; `mla_attend` covers `eager_attention_forward`'s boundary, which a fixture feeds
> the reference's OWN q/k/v — so the eps cannot reach it, and a GPU reduction can never be
> bit-exact with torch in any case. Inheriting `mla`'s `ExactOnly` would have made item 2
> ungateable for a reason that does not apply to it.
>
> | operator | floor draw 1 | floor draw 2 | floor | weakest targeting defect | policy |
> |---|---|---|---|---|---|
> | `mla` | 1.801e−5 | **5.742e−5** | 5.742e−5 | `MlaLoraEps1e5` 1.923e−5 | ExactOnly |
> | `mla_attend` | 2.320e−5 | **4.103e−5** | 4.103e−5 | `MlaScaleFromNope` 6.578e−1 | Rel(4.10e−4) |
>
> **`mla`'s finding got stronger, and the old number was optimistic.** It was recorded as floor
> 1.697e−5 against a 2.22e−5 defect — a margin of 1.3×, i.e. "no threshold admits a correct kernel
> and rejects that eps". Both numbers were single-draw. Measured across BOTH draws, with the bucket
> the new captures give it, the margin is **0.33×**: the eps divergence sits *below* the operator's
> own fp32 rounding floor. That is strictly stronger than un-thresholdable — it is
> **indistinguishable from rounding** — and it points the same way: the eps must be pinned by
> reading it, and no numeric gate can stand in for that.
>
> The two draws are **3.2× apart** for `mla` and 1.8× for `mla_attend`. That is the same one-draw
> trap Muse Glimmer's `attend` floor exposed, hit again on a different model — a floor is the max
> over draws or it is not a floor.
>
> **`MlaLoraEps1e5` is excluded from `mla_attend`'s defect set** on PROVENANCE. It perturbs the
> LoRA norms, which are *upstream* of the attention; it reaches this bucket only by changing the
> q/k/v the attention is handed, and a fixture feeds the reference's own q/k/v, so the defect cannot
> reach the kernel under test at all. It is excluded because it does not target this operator.
>
> > **CORRECTED 2026-08-12.** This excluded it "for a sharper reason than judgement: it moves that
> > bucket by 3.031e−5, which is *below* the 4.103e−5 floor." **That rule does not survive being
> > applied twice.** On the `mla` row the same defect moves 1.923e−5 under a 5.742e−5 floor — also
> > below — so a magnitude rule would drop it there too, leaving `mla` with no targeting defect and
> > turning its ExactOnly into a `Rel`. Two rows cannot read the same evidence shape and reach
> > opposite conclusions. The provenance argument above is the one this file's own methodology
> > already states (*the weakest signal among the defects that TARGET this operator*); that the
> > signal is also below the floor is corroboration, not the reason. Found by adversarial review;
> > `tests/common/k3_tolerance.rs` carries the same correction at the row itself.
> >
> > Stated plainly because it cuts the other way too: `Rel(4.10e-4)` is 13.5× above what
> > `MlaLoraEps1e5` moves this bucket by, so **the bucket-level gate provably cannot see that eps.**
> > Correct for the fixture, wrong to generalise — S3 must pin the eps by reading the constant.

**Seven operators have a row; the driver classifies ELEVEN — and the four without one are a GAP, not a
decision.** `operator_of` also emits `kda_trunk` (a KDA layer's projections and norms, as opposed to
the recurrence itself), `norm`, `residual` and `head`. Nobody measured a floor for those: the six
are the distinct kernels S2 and S3 will write, and the other four are buckets the comparator uses to
say *where* a defect landed. So **S2 must not score those four against a threshold** — compare them
exactly, or measure the floor the same way (`--dtype float64`, then `--by-operator`) and add a row.
`k3_anchor.rs` now asserts the table holds exactly the measured six in both directions, so a row for
an unmeasured operator is a visible edit rather than a number that arrived from nowhere.

## The width blind spot every fixture built on this inherits

**Every capture here is at the tiny model's widths, and for a kernel that is not merely a smaller
version of the real thing — it can be a structurally different code path.** Found 2026-08-11 while
gating S2 item 1, and credited to the Muse Glimmer port, which hit the identical shape in its own
attend fixture (all captures at `head_dim` 8 against a real 128, so a per-lane accumulator only ever
exercised its first register).

For `attn_res` the numbers are stark. `hidden` is **192** here and **7168** in the real model, and
the kernel launches 256 threads — so in every golden-backed case the strided loop
`for (i = t; i < n; i += blockDim.x)` runs **at most one iteration**, and 64 of the 256 threads run
none. At the real width each thread runs 28. Separately, every AttnRes capture is one token, so
`blockIdx.x`'s strides into `src` and `out` are multiplied by zero throughout — while layer-major
prefill, the default, passes the whole prompt at once.

Both gaps were then demonstrated rather than argued. `tests/k3_kernels.rs` carries a synthetic
sweep — `n` in {192, 257, 1000, 7168} × `nsrc` in {2, 9} × `tokens` in {1, 3}, scored against the
same f64 host oracle the golden tests validate — and two deliberate kernel breaks were caught by
**that sweep alone**, with all twelve folds across both draws staying green:

| break | golden suite | width sweep |
|---|---|---|
| `out` ignores `blockIdx` — every token writes block 0 | green | **red** |
| score reduction truncated after one pass — `n > 256` silently dropped | green | **red** |

**This is a property of anchoring on a tiny model, not a defect in this anchor**, and shrinking
widths remains the right trade: depth and structure are where the traps live, and width is what
costs. But it means **a golden-backed fixture is necessary and not sufficient for any kernel whose
loop structure depends on a width**, which is most of them. Every remaining S2 item — MLA over 192
head dims, the latent sandwich at 3584, the MoE at 7168, KDA's 96 heads × 128 — owes the same
synthetic sweep beside its golden fixture. Score it against a host oracle the goldens have already
validated, so the sweep inherits the reference's evidence instead of asserting a fresh claim.

## Two salts, because one draw proves less than it looks like

The residual limit was that the golden is **one weight draw at one position**: a kernel bug
degenerate at those particular values — a routed weight near zero, a `beta` saturating the gate —
hides completely, and one draw cannot show that a defect's localisation is a property of the
arithmetic rather than of the numbers it landed on.

So **two goldens are vendored** (`--salt k3-anchor-1` and `-2`, 332,009 B each — identical shapes,
independent values), `tests/k3-anchor.sh` scores all eleven defects against both, and every test in
`tests/k3_anchor.rs` loops over both. The salt-2 matrix reproduces salt-1's green cells exactly:
`MlaLoraEps1e5` and `MlaScaleFromNope` leave layers 0 and 1 bit-identical, `ExpertW1W3Swap`,
`RouterBiasInWeight` and `LatentNormAfterUp` leave layer 0 bit-identical. The localisation is a fact
about the arithmetic.

Degeneracy is now **asserted rather than hoped for**, on each draw: no routed weight below 5% of the
largest (an expert weighted at ~0 makes its own arithmetic unscoreable), `|beta| < 8` (beyond that
`sigmoid` is within 4e-4 of its limits and the delta-rule update is pinned), `A_log` inside
`log(uniform(1,16))`, `dt_bias` inside its draw range, logits finite and not all equal.

Two of those were weaker than they read, both corrected 2026-08-11 by review. **`A_log`'s range
check accepted an ALL-ZEROS vector**, since `log(1) = 0` is inside `log(uniform(1,16))` — and a
constant `A_log` makes every head decay identically, so a kernel ignoring the term would match;
only the FNV pin stood between that and a pass. It must now also be non-constant. And **the
"two independent draws" claim was checked by comparing the two FILES' hashes**, which carries no
information: each golden embeds its own `salt` string in its metadata, so the bytes differ whatever
the weights did. A driver refactor that passed a literal salt to `init_weights`, or an `_gen` that
stopped mixing the salt into its seed, would put bit-identical tensors in both files and pass
everything — while the fixture claimed the second draw whose entire purpose is that a bug degenerate
at one draw's values cannot hide. It is now compared per tensor, on `to_bits` over every float the
two goldens share, and all of them must differ — **except a named structural class**, which must be
bit-identical instead. That class exists because item 2's captures introduced values that are not
drawn at all: the attention `mask` is causality (all zero at a decode step) and `scaling` is the
config constant `1/sqrt(qk_nope + qk_rope)`. Asserting them EQUAL is the stronger statement — a
`scaling` that varied between draws would mean it had stopped being a config constant. The
exemption list is itself checked for being non-empty, so it cannot go stale into exempting nothing.
Item 3's two captures are ordinary draws: the aggregate is downstream of the weights and the norm
weight is drawn from `sha256(salt/name)` like every other parameter.

> **CORRECTED 2026-08-12.** This said "all 223 floats … and all 223 must differ", a count that was
> stale by 49 and, more importantly, described a test that had already been rewritten: a flat
> "everything differs" rule went red the moment `mask` and `scaling` were captured. The count is
> deliberately not restated here — `tests/k3_anchor.rs` asserts it and this file quoting it is how
> it went wrong twice.

**The vendored `config.json` is pinned by a hash the gate RECOMPUTES**, added 2026-08-11 after Muse
Glimmer's port found the same hole on its own side (its HF revision was a prose claim matched by
nothing). The goldens' metadata records `real_config_sha256_16` of the file the run consumed, and
`k3_anchor.rs` pins that value — but both are frozen copies, so a `config.json` updated to a later
revision left them agreeing with each other while no longer describing the file on disk. The
structural test catches a revision that moves a field it reads; a revision that ADDS one is exactly
what it cannot see, and the red-proof showed that directly — injecting a new top-level field left
`the_tiny_configs_kept_the_real_structure` GREEN and only the new hash went red. FNV-1a rather than
sha256, on the same argument as the golden bytes: there is no sha256 in this tree and a
dev-dependency for one tripwire is not worth it. That the Rust hash over `include_str!` agrees with
python's over the raw file is also a check that the include performs no transformation.

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

**The venv is a shared resource, and the driver now refuses to generate in the wrong one.** As of
2026-08-11 a second env exists on this machine for Muse Glimmer's S1b, at transformers
**5.15.0.dev0** — `muse_glimmer` is native to that version, so it is not a choice on their side —
while these goldens are **4.56.2**. `K3_ANCHOR_VENV` pointing at the wrong one is a two-character
mistake, and without a check the symptom is a `cmp` mismatch at the *end* of a ~25 min GPU-locked
regeneration, reported as "DIFFERS … find out why it moved", which invites suspecting the driver.
`preflight_env()` runs before the reference is even loaded and names the drifted package. The
versions it compares against are **read out of the vendored golden**, not restated in the driver —
those bytes already carry what produced them and `k3_anchor.rs` already asserts it, so a third copy
is the shape that made CLAUDE.md's exemption count wrong three times over. A deliberate re-pin
therefore needs no edit to the driver: `K3_ANCHOR_ALLOW_ENV_DRIFT=1`, regenerate, re-vendor, and the
new bytes are the new pin — plus the version strings in `k3_anchor.rs`, which is the gate on them.

The install is ~6.2 GB of ROCm wheels and does not belong in `/tmp` — that is 63 GB of tmpfs *in
RAM* on this box, and filling it shrinks `--max-mem auto` silently before anything fails loudly.
`--device cpu` gets a config, weight-init or defect-injection error out of the driver in seconds
without taking the GPU lock; it cannot produce a golden, which is the point.
