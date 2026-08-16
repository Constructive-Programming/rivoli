---
scope: glimmer
status: data
verdict: The S1b anchor exists and runs — Muse Glimmer's own first-party stack (transformers 5.16.0.dev0 at commit fe747d88, torch 2.13.0+cpu, python 3.14.6) executed at tiny widths but the REAL structure, and it needs NO GPU because this reference is plain PyTorch with a CPU path for every operator. SIX files are vendored (tests/glimmer-anchor-{text,draft,weights}-{1,2}.bin; text 643,957 B, draft 72,145 B, weights 113,035 B), two weight draws x two modes plus a WEIGHT SET per draw, each reproduced byte-for-byte on a later run, and read by tests/glimmer_anchor.rs with no python, no venv, no network and no device. THE WEIGHT SETS WERE ADDED 2026-08-13 for S3 item 3 (--dump-weights): gate_proj is 72->48 and a layer captures only 18 rows, so 18 equations against 72 unknowns per output element is underdetermined by 4x and EVERY candidate operand admits a weight that fits the captures exactly -- the recover-and-predict shape the sandwich norms use is not weaker here but VACUOUS, because a norm is elementwise and a projection is not. They go in their own files, not into the goldens, so the four pinned FNVs above did not move; verified by regenerating both goldens with the flag on and finding them byte-identical. They were first vendored with no length, no FNV, no census and no regeneration path, which review caught the same day -- glimmer-anchor.sh now regenerates and cmps them and glimmer_anchor.rs pins their bytes. FOURTEEN defect runs are scored at both draws, 28 runs in all, and each is GATED on the captures it must leave bit-identical rather than merely on having changed something. THE FINDING THAT ONLY THIS CAN SEE - softcap_off moves 7 of 1103 captures and leaves emitted.ids identical, so the argmax-invariant logit path is not an argument but a measurement, and every greedy gate in this repo is provably blind to it. Two reference behaviours were discovered by running it: the DFlash drafter's default mask is block-wide against context+block K/V and RAISES, and passing the correct 2D mask only works with use_cache=False because a fresh DFlashCache reports kv_length 0. The green sets are scoped to step 0 because a defect that shifts the argmax contaminates every later step through the token it feeds back - localisation is only possible on the prefill. Five deviations are declared in the metadata and pinned by the test - eager attention, fp32 against a bf16 checkpoint, the ForConditionalGeneration wrapper (the softcap lives only there), shrunk special-token ids, and output_multiplier kept at the released value rather than recomputed. TOLERANCES ADDED 2026-08-11 for S2 item 1: per-operator fp32 floors for all thirteen buckets, from --dtype float64 (the whole model in double, no island needed) against each fp32 golden, measured at BOTH draws because attend's floor is 2.1x apart between them - 7.819e-6 and 1.639e-5 - so a one-draw floor would have set the threshold at half what a correct kernel needs. SIX rows are tabled as of 2026-08-12 -- attend, rope, o_proj, logits, norm and qk_norm; attend's is floor 1.639e-5, weakest targeting defect 2.086e0 (kv_broadcast_blocked), Rel(1.64e-4). (This said "ONE row is tabled" through five more landing; the count is stated once here and was gated by old:glimmer_tolerance.rs, nowhere else -- CORRECTED 2026-08-16: that file does not exist in the rewrite and no table_covers_exactly caller covers GLIMMER here, so the six-row set is gated in neither direction in this tree.) qk_scale_on_k is EXCLUDED from that set at 6.232e-4 (38x the floor, which would have forced ExactOnly) because (s*q).k and q.(s*k) are the same product - the defect is invisible to this kernel by algebra, not by resolution. CORRECTED 2026-08-12: this said it "is caught in qk_norm/proj instead" at 2.16e0, and measuring it for S3 item 2's row showed all of that false -- qk_norm moves 4.324e-4/2.825e-4 and proj 4.185e-4/2.449e-4, the 2.16e0 was qk_norm_off's attend figure off the wrong row, and and it does not implement trap 3 at all. CORRECTED AGAIN 2026-08-13: the replacement claim -- that it is caught NOWHERE at Rel strength because an RMS norm is scale-invariant -- is ALSO false. The norm cancels a scalar only up to the eps term, a residue of sqrt(1+(s^2-1)*eps/(s^2*m+eps))-1 which reproduces the measured 2.825e-4 and 6.232e-4 figures and is 3.7x-7.9x the qk_norm row's own Rel(7.85e-5). So the defect IS caught there; the exclusion now rests on MARGIN (36x the floor against the 297x a Rel policy needs) and on every m here coming from toy widths with no real weights, the residue falling as 1/m to 0.06x tol at m~1. S4 re-derives it. The other twelve buckets have floors and no row: a floor is half a row, and S2 compares them exactly until the other half is reasoned through. DRAFT-MODE FLOORS MEASURED 2026-08-16 for M17a: the same fp64 protocol at --mode draft, both salts -- 11 compute captures floored at 2.0248e-6..3.5313e-6 while mask/noise_embeds/candidates came back exactly 0.0 (copies and picks, compared bit-exactly, no tolerance) -- and the CPU drafter oracle scores every capture at 10x these (3 s.f.), with a five-defect matrix in crates/oracles/tests/glimmer_draft_oracle.rs proving the comparison reddens. THE GATE'S OWN LADDER was then measured and both halves recorded: a +1 ulp f32 flip on one encoder.fc element moves encoder.out by 7e-13 relative (2.980119e-7 -> 2.980126e-7) and is CORRECTLY invisible at a 2.33e-5 threshold, so this gate witnesses structure and never a one-ulp conversion error, while an attention-scale slip to 1/sqrt(head_dim+1) reddens it. AND A FALSE GREEN WAS FOUND AND CLOSED: the scoring helper folded (got-want).abs() with f64::max, which returns the OTHER argument on NaN, so an all-NaN capture scored 0.0 -- a perfect match -- proven by running the clean gate GREEN on a capture forced to all-NaN with the guard removed, and red with it. Third recorded reintroduction of that trap in this tree, and the first where the reference side was already guarded and the got side was not. THREE FIXTURE LIMITS were then found by review and are now standing assertions rather than surprises: (1) NO QUERY EVER ATTENDS THE BLOCK -- window 4, block 4, ctx 12, so the furthest query reaches kv 7 of 16 and the block-vs-block submatrix sums to exactly 0.0 on both salts, which means glimmer-architecture.md section 11 step 5 (attention is BIDIRECTIONAL across the block) is pinned by the mask PATTERN alone and by no value, and every draft floor and defect signal was measured on a pattern the real model does not produce; (2) sliding_window and block_size are BOTH 4, so every value comparison scores green with either read for the other -- closed by a gate that widens the window and requires the mask to move; (3) the reference pins its rotary tables to FLOAT32 even under --dtype float64 (inv_freq via torch.arange dtype=torch.float, re-cast inside maybe_autocast(enabled=False)), so the floors have zero contribution from the rope table and could never price one computed otherwise -- the oracle now builds that table in f32 in the reference's own order. Limits 1 and 2 need a re-vendor at sliding_window >= ctx + block to close.
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

> **CORRECTED 2026-08-16.** That venv no longer exists on this machine. M17a's draft-mode floor
> runs were made under `/home/rhansen/glm-anchor/venv` (transformers 5.15.0) with
> `--no-preflight`, after checking the substitution the only way that makes it citeable:
> that env regenerates both vendored draft goldens with all 50 captures bit-identical, and the
> assistant modeling files are byte-identical between 5.15.0 and the pinned `fe747d88`. The pin
> above is still what the goldens WERE produced under and is still what `preflight_env()`
> enforces — it is not re-establishable today. See §Draft-mode floors.

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

> **CORRECTED 2026-08-16.** This command cannot run as written: the venv it names is gone (see
> the note under §Environment). The driver's own path in this tree is
> `crates/oracles/tests/glimmer_anchor_driver.py`, and the substitute env needs `--no-preflight`
> because `preflight_env()` correctly refuses it.

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

## Tolerances — the fp32 floors, measured 2026-08-11 for S2

Added when S2 item 1 (the GQA attend kernel) came to need a threshold. **Measured before the kernel
existed**, which is the only order in which the number means anything.

```bash
# The floor: run the identical reference in double and diff it against the fp32 golden.
tests/glimmer_anchor_driver.py --mode text --defect None --salt glimmer-anchor-1 \
    --dtype float64 --out fp64-1.bin
tests/glimmer_anchor_driver.py --by-operator fp64-1.bin tests/glimmer-anchor-text-1.bin
# The signal a threshold must stay under: the same report, clean against each defect run.
tests/glimmer_anchor_driver.py --by-operator target/glimmer-anchor/text-1-None.bin \
    target/glimmer-anchor/text-1-kv_broadcast_blocked.bin
```

**`--dtype float64` is the whole model in double, with no island.** K3's anchor has to hold every
fla module at fp32 because its KDA ops are triton kernels that refuse double; this reference is
plain PyTorch, so one flag covers every operator at once. Weights are untouched by it —
`init_weights` draws into an explicit f32 buffer and widens — so an fp64 run sees numerically
identical weights and differs *only* in accumulation, which is the property being measured.

**Floors, max over both draws** (`--mode text`; the draft mode has no floors yet):

| operator | draw 1 | draw 2 | floor |
|---|---|---|---|
| `attend` | 7.819e−6 | 1.639e−5 | **1.639e−5** |
| `o_proj` | 7.643e−6 | 8.293e−6 | 8.293e−6 |
| `qk_norm` | 7.845e−6 | 6.295e−6 | 7.845e−6 |
| `norm` | 6.082e−6 | 7.701e−6 | 7.701e−6 |
| `proj` | 3.307e−6 | 4.779e−6 | 4.779e−6 |
| `rope` | 4.490e−6 | 4.773e−6 | 4.773e−6 |
| `mlp` | 4.532e−6 | 4.744e−6 | 4.744e−6 |
| `logits` | 3.061e−6 | 3.520e−6 | 3.520e−6 |
| `gate` | 3.408e−6 | 3.501e−6 | 3.501e−6 |
| `final_norm` | 3.045e−6 | 3.496e−6 | 3.496e−6 |
| `embed_norm`, `rope_table`, `ids` | 0 | 0 | below what the container can see |

**A floor measured at one draw is not a floor.** `attend` came out **2.1× apart** between the two
draws — same arithmetic, different numbers for the softmax to round. Taking draw 1 alone would have
placed the threshold at half what a correct kernel can need, and that failure mode is a *correct*
kernel that cannot pass. Every floor above is the max over both. (K3's table was measured at one
salt and is not known to be draw-robust; that is now an open item against it.)

**The three zeros are "below the container", not "exact".** `Capture.add` stores f32 because the
golden container does, so an fp64 run is rounded on the way out and every comparison carries one f32
rounding — relative 2⁻²⁴ = **6.0e−8**. Every non-zero floor above sits 50–130× clear of that, so it
is measuring arithmetic; a bucket landing near 6e−8 would be measuring the container and must not
become a threshold until the container is widened.

### The rows that exist, and why the others do not

**Six rows exist as of 2026-08-12** — `attend`, `rope`, `o_proj`, `logits`, `norm` and `qk_norm`;
`logits` is `ExactOnly`. (This said "Four" through two more landing. The set was gated by
`old:tests/glimmer_tolerance.rs::table_covers_exactly` and by nothing else, so a count in prose is
a number that rots — this is the third place it had to be corrected in one round.)
> **CORRECTED 2026-08-16.** `glimmer_tolerance.rs` does not exist in the rewrite. The only
> `table_covers_exactly` caller here is K3's (`crates/oracles/tests/k3_anchor.rs`), so the GLIMMER
> six-row set is gated in NEITHER direction in this tree. The claim was carried over from the old
> tree with the doc; porting that gate is Glimmer-side work, not M17's. `softcap_off` moves `logits` by 4.993e-5 / 4.879e-5 by draw, only **13.9x** the floor
against the 297x a `Rel` policy needs, and the reason is the TINY MODEL rather than the instrument:
at untrained weights the logits sit in `tanh`'s linear region where `20*tanh(x*0.196/20)` is nearly
the identity. **This anchor therefore cannot price the logit path**; S4's trained weights can.
`ids` moves by exactly 0.000e0 at both draws, which is section 5's argmax-invariance as a
measurement. `o_proj`'s row comes from `gate_disabled` at 3.759e0 / 3.688e0 (weakest 3.688e0,
margin 444,700x) and `rope`'s from `rope_interleaved` / `rope_on_nope_layers` (weakest 1.811e0,
margin 379,000x).

The original note's reasoning, kept because it is the reusable part (its two-row inventory went
stale the day the table above reached four and is dropped): a floor is half a row — the other
half is deciding which defects the operator is *answerable for*, and that is per-kernel
reasoning. An operator earns its row when its item lands; until then S2 compares it exactly,
enforced from the other side by `tests/glimmer_tolerance.rs`, which fails on a row for an
unanalysed operator.

**`norm`'s defect set, added 2026-08-12 for S3 item 1 — measured BEFORE the kernel existed.** The
bucket is exactly the four sandwich norms: **224 tensors = 4 norms x 8 layers x 7 steps**, with
`final_norm` (7) and `qk_norm` (112) as separate buckets and separate call sites. That separation is
load-bearing, because **this model carries two norm formulas** — the four sandwich norms are
CENTERED, `x*(1+w)`, while the final norm and the two weightless norms are plain `x*w` (§5); the four norms and their two eps are §3.

| defect | draw 1 | draw 2 | weaker |
|---|---|---|---|
| `norm_not_centered` — `x*w` where `x*(1+w)` belongs, i.e. the FORM | 1.139e0 | 1.131e0 | 1.131e0 |
| `post_norm_eps_shared` — `rms_norm_eps` where `post_norm_eps` belongs | 2.571e−2 | 2.024e−2 | **2.024e−2** |

Floor **7.701e−6**, weakest **2.024e−2**, margin **2,628×** against the 297× a `Rel` policy needs,
so `Rel(7.70e-5)`.

**Both are counted and neither is excluded — the opposite call to `attend`'s.** There the excluded
defect was invisible to the kernel *by algebra*. Here the eps defect is 56× weaker than the form
defect and it is still one this operator is answerable for: the two eps differ by three orders of
magnitude (1e-5 against 1e-8) and a kernel handed the wrong one computes a genuinely different
value. It is also the conservative choice, since it is the number that sets the row.

Worth recording for whoever writes the kernel: `norm_not_centered` also moves `qk_norm` (2.434e0 /
2.201e0), `embed_norm` (1.995e0 / 1.962e0) and `final_norm` (1.738e0 / 1.695e0) — those are
**downstream contamination, not targeting**, since a contaminated hidden state reaches every later
norm. Pricing this operator against them would be the leak this file's own tolerance rule forbids.

**`qk_scale_on_k` is excluded from `attend`'s defect set, and the exclusion decided the policy.** It
moves `attend` by only 6.232e−4, just 38× the floor — under the 297× a `Rel` tolerance needs, so
counting it would have forced `attend` to `ExactOnly`. It must not: `(s·q)·k` and `q·(s·k)` are the
same product, so moving the scale across the dot is invisible to this kernel **by algebra**, and
6.232e−4 is the rounding difference between two spellings of one number. Pricing an operator against
a defect it provably cannot distinguish would have made this kernel exact-only on a false premise.

> **CORRECTED 2026-08-12, by running `--by-operator` for S3 item 2's row.** This closed with "the
> defect is real and is caught where it is *not* equivalent — the qk-norm runs between the scale and
> the product, so `qk_norm` and `proj` see it at 2.16e0." **All three claims are false.** Measured,
> `qk_scale_on_k` moves `qk_norm` by 4.324e−4 / 2.825e−4 and `proj` by 4.185e−4 / 2.449e−4 — not
> 2.16e0, which appears to be `qk_norm_off`'s `attend` figure (2.153e0) read off the wrong row.
>
> **So this defect is caught NOWHERE at `Rel` strength**: its largest movement anywhere is `attend`'s
> 6.232e−4, 38× that floor. The exclusion above still stands on the algebra, but the consoling clause
> does not, and an exclusion justified by "it is caught elsewhere" needs the elsewhere to exist.
>
> **The reason is that the defect does not implement the trap it is named for.** `defect_qk_scale_on_k`
> multiplies `k_proj`'s OUTPUT by 3.87 — upstream of the weightless qk-norm, which is scale-invariant
> but for its eps, so the norm removes it and the model is barely perturbed. The residue is the eps
> term alone: `(eps/2)/mean(k²)·(1−1/3.87²)` at mean(k²) ≈ 0.016 predicts 2.9e−4, which is what came
> out at both draws. §9 trap 3's damaging form is the MIRROR of what Q gets — `qk_norm(k) * 3.87`,
> after the norm — and that has no defect run, so **nothing in this anchor gates trap 3**. Fixing it
> means re-vendoring, which is a reviewed change and is left as an open item rather than folded into
> S3 item 2.

**`rope`'s defect set, added 2026-08-12 for S2 item 2.** Two defects target the rotation and **both
are counted** — unlike `attend`'s excluded `qk_scale_on_k`, there is no algebraic identity hiding
either from a rope kernel. Regenerated at both salts and scored with `--by-operator` against the
clean run:

| defect | draw 1 | draw 2 | weaker |
|---|---|---|---|
| `rope_interleaved` (the pairing convention) | 2.505e0 | 2.214e0 | 2.214e0 |
| `rope_on_nope_layers` (rotating the 13 θ=0 layers) | 2.011e0 | 1.811e0 | **1.811e0** |

The weakest is the row's, giving a margin of **379,000×** over the floor. Both defect runs also
re-derived the clean goldens from scratch, and those came back **byte-identical to the vendored
`text-1`/`text-2`** — an unplanned second check of the reproducibility claim this file makes.

What remains are genuine attend wrongnesses, each shown as the weaker of the two draws:
`kv_broadcast_blocked` **2.086e0**, `window_off_by_one` 2.187e0, `full_layers_slide` 2.282e0. The
weakest is the row's `weakest_defect`, giving a margin of 127,000× — `Rel` is comfortably founded.

Three defects change the capture set or a mask's shape (`window_off_by_one` and `full_layers_slide`
resize masks, `rope_on_nope_layers` adds 56 captures), and `--by-operator` **skips and counts** those
rather than refusing the pair. It refused at first, which silently dropped exactly the two rows
`attend` needed — the sweep printed no `attend` line for them at all, which reads as "the window
defect leaves attention alone".

**`qk_norm`'s defect set, added 2026-08-12 for S3 item 2.** Two defects touch the bucket and only
one of them TARGETS it. Regenerated at both salts (the clean runs again came back byte-identical to
the vendored goldens, a third independent check of the reproducibility claim) and scored with
`--by-operator`:

| defect | draw 1 | draw 2 | weaker |
|---|---|---|---|
| `qk_norm_off` — the norm skipped, §9 trap 2 | 1.575e0 | 1.483e0 | **1.483e0** |
| `qk_scale_on_k` — EXCLUDED, see above | 4.324e−4 | 2.825e−4 | 2.825e−4 |

Floor **7.845e−6**, weakest **1.483e0**, margin **189,000×** — `Rel(7.85e-5)`.

**The exclusion is by LOCALITY, not by algebra, and the distinction matters.** `attend` excludes the
same defect because `(s·q)·k` and `q·(s·k)` are one product — an identity. Here the scale is applied
to `k_proj`'s output, i.e. to this operator's INPUT, and a correct norm reproduces the reference from
whatever input it is given. An operator cannot be priced against a defect upstream of it. Counting it
would have forced `ExactOnly` at 36× the floor, which is the same false-premise trap `attend`'s row
records — arrived at twice, by two different routes.

**What this row does NOT cover: the 3.87 scale itself.** `qk_norm.q` is captured BEFORE it, so no
tensor in this bucket can see it. It is visible one capture later, in `q.pre_rope` (the `proj`
bucket), which the driver takes on entry to `apply_rotary_pos_emb` — after norm and after scale. A
port that omits the scale, or folds it into the softmax scale instead, reddens there and not here.

### Draft-mode floors, measured 2026-08-16 for M17a

The same fp64-against-fp32 protocol, `--mode draft`, both salts — the CPU drafter oracle
(`crates/oracles/src/dflash.rs`) computes in f64 and is scored per capture at 10× these:

```bash
$VENV/bin/python glimmer_anchor_driver.py --mode draft --salt glimmer-anchor-$n \
    --defect None --dtype float64 --out draft-$n-f64.bin --no-preflight
# then per-capture max|f32−f64| / max|f64|, worst over both salts and all five layers
```

| capture | floor | | capture | floor |
|---|---|---|---|---|
| `encoder.out` | 2.3281e−6 | | `attend.out` | 3.4292e−6 |
| `input_layernorm.out` | 2.0248e−6 | | `post_attention_layernorm.out` | 2.4606e−6 |
| `attend.q` | 3.5134e−6 | | `mlp.out` | 3.5313e−6 |
| `attend.k` | 3.3121e−6 | | `final_norm.out` / `last_hidden` | 2.4320e−6 |
| `attend.v` | 2.3172e−6 | | `logits` | 2.6476e−6 |

`attend.mask`, `noise_embeds`, `block_ids` and `candidates` came back at exactly 0.0 — they
are copies, thresholds or integer picks, not arithmetic, and the oracle compares them
bit-exactly with no tolerance at all. The thresholds (10×, written to 3 s.f. because FOUR of
these floors round outside `FLOOR_MULT`'s (9.9, 10.2) band at 2 s.f. — `encoder.out` 9.879×,
`input_layernorm.out` 9.878×, `final_norm.out` 9.868× and, worst, `logits` at 9.820×; at 3 s.f.
every ratio lands in 9.976×–10.012×. **CORRECTED 2026-08-16, same day:** this said "two of these
floors", a count restated rather than recomputed, and review recomputing it found four — the
worst of them not one of the two that had been named), the oracle-side defect signals they
must catch, and the five-defect matrix that proves the comparison can redden all live in
`crates/oracles/tests/glimmer_draft_oracle.rs`; the fp64 runs were made under
`/home/rhansen/glm-anchor/venv` (transformers 5.15.0 — the 5.16.0.dev0 venv named above no
longer exists on this machine) after verifying that env regenerates both vendored draft
goldens with all 50 captures bit-identical (only the metadata version strings differ — the
assistant modeling files are byte-identical between 5.15.0 and the pinned `fe747d88`).

### What the draft oracle's gate can and cannot see — the ladder, measured 2026-08-16

Each rung was planted in the tree, run, and reverted. The instrument is the clean gate
(`the_oracle_reproduces_every_captured_value_of_both_goldens`); where a number is quoted it was
read off the assert's own message with that row's tolerance driven to zero, which is how a
green gate is made to state what it actually measured.

| rung | perturbation | `encoder.out` rel | gate |
|---|---|---|---|
| 0 | none (the shipped oracle) | 2.980119e−7 | green — 78× inside its 2.33e−5 tolerance |
| 1 | +1 ulp on one f32 element of `encoder.fc` | 2.980126e−7 | **green, and correctly so** |
| 2 | attention scale `1/√head_dim` → `1/√(head_dim+1)` | — | red |

**Rung 1 is the floor and it is the useful number**: a one-ulp weight flip moves the worst
relative difference by 7e−13, five orders of magnitude below the threshold. That is not a hole
— a single-ulp f32 perturbation is far inside the reference's OWN fp32 noise, which is what the
floor measures — but it does bound what this gate is evidence for. It cannot witness a
conversion that is off by an ulp; it witnesses structure. (Recorded because this repo has twice
registered a red-proof whose perturbation was below the detection floor and read the resulting
green as a passing gate.)

The whole battery, each plant applied to the tree, run, and reverted (2026-08-16):

| plant | gate | result |
|---|---|---|
| one capture forced to all-NaN, got-side guard PRESENT | clean forward | red at the finite guard |
| the same, guard REMOVED | clean forward | **GREEN — all 39 tolerances passed** |
| a well-formed extra row in `DRAFT_ORACLE` | `table_covers_exactly` | red |
| attention scale `1/√head_dim` → `1/√(head_dim+1)` | clean forward | red |
| `+1 ulp` on one f32 element of `encoder.fc` | clean forward | green (2.980126e−7 vs 2.980119e−7) |
| one recorded L0 signal raised 2.146e−1 → 2.60e−1 | defect matrix | red at the signal bar |
| `mask()` reading `block` instead of `window` | whole binary | red at **exactly one** test — the window gate — and green at the other nine |
| a draft golden renamed out of the prefix filter | whole binary | red (salt pairing, 7 of 10) |

The `block`-for-`window` row is the one worth keeping: nine value gates stayed green under a
substitution that is a 128× error in the real model, and only the gate written for that collapse
caught it. That is what "every value comparison scores green with either read for the other"
means measured rather than argued.

**A false green was found and closed the same day.** The scoring helper folded
`(got − want).abs()` with `f64::max`, which RETURNS THE OTHER ARGUMENT ON NaN — so a capture
that computed only NaN scored 0.0, a perfect match. Proven both ways rather than argued: with
one capture forced to all-NaN and the guard removed, the clean gate **passed** (all 39 captures
green); with the guard in place it fails on that capture before any tolerance is consulted.
This is the third recorded reintroduction of the same trap in this tree
(`crates/engine/tests/common/scoring.rs` carries the first two), and the first where the
reference side was already guarded and the got side was not.

### What the draft goldens cannot see, found by review 2026-08-16

Three limits of the FIXTURE, not of the oracle. Each is now a standing assertion in
`crates/oracles/tests/glimmer_draft_oracle.rs` or a stated exception in `dflash.rs`, so none
can be rediscovered as a surprise:

- **No query ever attends the block.** `sliding_window 4`, `block_size 4`, captured context 12
  → `kv_len` 16, the block's own K/V rows at 12..16, and the mask `|q_row − kv| ≤ 4` lets the
  furthest query (row 3) reach kv **7**. Measured: the block-vs-block submatrix sums to exactly
  0.0 on both salts. **§11 step 5 — attention is bidirectional across the block — is therefore
  pinned by the mask PATTERN alone and by no value in this golden.** `defect_causal_mask`'s red
  is the re-selection of CONTEXT rows; a port that is block-causal and context-correct is
  indistinguishable here. Every draft-mode floor and every defect signal was measured on an
  attention pattern the real model (window 2048, block 16) does not produce. Closing it needs a
  re-vendor at `sliding_window ≥ ctx + block`.
- **`sliding_window` and `block_size` are both 4**, one field apart in the config, so every
  value comparison scores green with either substituted for the other — the `[D][D]` axis-order
  trap in a new place. Verified by substitution. Closed by a gate that widens the window by one
  and requires the mask to move; the collapse itself needs a re-vendor.
- **The reference's rotary tables are float32 even under `--dtype float64`**
  (`modeling_muse_glimmer_assistant.py:352` pins `inv_freq` to `torch.arange(..., dtype=
  torch.float)`, and `:358-363` re-casts inside `maybe_autocast(enabled=False)  # Force
  float32`). So the fp64 run that measured these floors used the SAME f32 table as the fp32
  run, and **the floors have zero contribution from the rope table** — they cannot price one
  computed any other way. The oracle originally built it in f64, ~2e−7 relative from the
  reference (inside `attend.q`'s 3.51e−5, but on an argument the floor never made, and growing
  linearly with position). It now builds the table in f32 in the reference's own order.

## What this does NOT establish

- **Tolerances for twelve of the thirteen buckets.**
  > **CORRECTED 2026-08-16.** Seven, not twelve — six rows landed 2026-08-12 (§the rows that
  > exist, above). The count is the same one that rotted three times in that section; read the
  > table, not this bullet.

  Their floors are measured and tabled above, but
  a floor alone is not a threshold, and no row exists for them. Compare them exactly.
- **No draft-mode floors.** Both `--dtype float64` runs were `--mode text`.
  > **CORRECTED 2026-08-16.** No longer true: M17a measured the draft-mode floors at both
  > salts — §Draft-mode floors above — and the drafter oracle's tolerances are set from them.
- **No real weights.** Every number here comes from a deterministic draw at toy widths. The
  checkpoint's own arithmetic — bf16 accumulation, the real 6656/19968 shapes — is untouched.
- **Nothing about the conversion path.** The safetensors reduction is gated separately by
  `tests/glimmer_names.rs` against the vendored `model.safetensors.index.json`.
