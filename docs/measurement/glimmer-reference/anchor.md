---
scope: glimmer
status: data
verdict: The S1b anchor exists and runs — Muse Glimmer's own first-party stack (transformers 5.15.0 from PyPI -- no commit, pinned instead by assistant_modeling_sha256 c31755fb..d709ed -- torch 2.13.0+cpu, python 3.14.6; RE-VENDORED 2026-08-16 from transformers 5.16.0.dev0 at commit fe747d88) executed at tiny widths but the REAL structure, and it needs NO GPU because this reference is plain PyTorch with a CPU path for every operator. SIX files are vendored (tests/glimmer-anchor-{text,draft,weights}-{1,2}.bin; text 644,019 B, draft 72,208 B, weights 2,065,247 B, all re-vendored 2026-08-16), two weight draws x two modes plus a WEIGHT SET per draw, each reproduced byte-for-byte on a later run, and read by tests/glimmer_anchor.rs with no python, no venv, no network and no device. THE WEIGHT SETS WERE ADDED 2026-08-13 for S3 item 3 (--dump-weights): gate_proj is 72->48 and a layer captures only 18 rows, so 18 equations against 72 unknowns per output element is underdetermined by 4x and EVERY candidate operand admits a weight that fits the captures exactly -- the recover-and-predict shape the sandwich norms use is not weaker here but VACUOUS, because a norm is elementwise and a projection is not. They go in their own files, not into the goldens, so the four pinned FNVs above did not move; verified by regenerating both goldens with the flag on and finding them byte-identical. They were first vendored with no length, no FNV, no census and no regeneration path, which review caught the same day -- glimmer-anchor.sh now regenerates and cmps them and glimmer_anchor.rs pins their bytes. FOURTEEN defect runs are scored at both draws, 28 runs in all, and each is GATED on the captures it must leave bit-identical rather than merely on having changed something. THE FINDING THAT ONLY THIS CAN SEE - softcap_off moves 7 of 1103 captures and leaves emitted.ids identical, so the argmax-invariant logit path is not an argument but a measurement, and every greedy gate in this repo is provably blind to it. Two reference behaviours were discovered by running it: the DFlash drafter's default mask is block-wide against context+block K/V and RAISES, and passing the correct 2D mask only works with use_cache=False because a fresh DFlashCache reports kv_length 0. The green sets are scoped to step 0 because a defect that shifts the argmax contaminates every later step through the token it feeds back - localisation is only possible on the prefill. Five deviations are declared in the metadata and pinned by the test - eager attention, fp32 against a bf16 checkpoint, the ForConditionalGeneration wrapper (the softcap lives only there), shrunk special-token ids, and output_multiplier kept at the released value rather than recomputed. TOLERANCES ADDED 2026-08-11 for S2 item 1: per-operator fp32 floors for all thirteen buckets, from --dtype float64 (the whole model in double, no island needed) against each fp32 golden, measured at BOTH draws because attend's floor is 2.1x apart between them - 7.819e-6 and 1.639e-5 - so a one-draw floor would have set the threshold at half what a correct kernel needs. SIX rows are tabled as of 2026-08-12 -- attend, rope, o_proj, logits, norm and qk_norm; attend's is floor 1.639e-5, weakest targeting defect 2.086e0 (kv_broadcast_blocked), Rel(1.64e-4). (This said "ONE row is tabled" through five more landing; the count is stated once here and was gated by old:glimmer_tolerance.rs, nowhere else -- CORRECTED 2026-08-16: that file does not exist in the rewrite and no table_covers_exactly caller covers GLIMMER here, so the six-row set is gated in neither direction in this tree.) qk_scale_on_k is EXCLUDED from that set at 6.232e-4 (38x the floor, which would have forced ExactOnly) because (s*q).k and q.(s*k) are the same product - the defect is invisible to this kernel by algebra, not by resolution. CORRECTED 2026-08-12: this said it "is caught in qk_norm/proj instead" at 2.16e0, and measuring it for S3 item 2's row showed all of that false -- qk_norm moves 4.324e-4/2.825e-4 and proj 4.185e-4/2.449e-4, the 2.16e0 was qk_norm_off's attend figure off the wrong row, and and it does not implement trap 3 at all. CORRECTED AGAIN 2026-08-13: the replacement claim -- that it is caught NOWHERE at Rel strength because an RMS norm is scale-invariant -- is ALSO false. The norm cancels a scalar only up to the eps term, a residue of sqrt(1+(s^2-1)*eps/(s^2*m+eps))-1 which reproduces the measured 2.825e-4 and 6.232e-4 figures and is 3.7x-7.9x the qk_norm row's own Rel(7.85e-5). So the defect IS caught there; the exclusion now rests on MARGIN (36x the floor against the 297x a Rel policy needs) and on every m here coming from toy widths with no real weights, the residue falling as 1/m to 0.06x tol at m~1. S4 re-derives it. The other twelve buckets have floors and no row: a floor is half a row, and S2 compares them exactly until the other half is reasoned through. DRAFT-MODE FLOORS MEASURED 2026-08-16 for M17a: the same fp64 protocol at --mode draft, both salts -- 11 compute captures floored at 2.0248e-6..3.5313e-6 while mask/noise_embeds/candidates came back exactly 0.0 (copies and picks, compared bit-exactly, no tolerance) -- and the CPU drafter oracle scores every capture at 10x these (3 s.f.), with a five-defect matrix in crates/oracles/tests/glimmer_draft_oracle.rs proving the comparison reddens. THE GATE'S OWN LADDER was then measured and both halves recorded: a +1 ulp f32 flip on one encoder.fc element moves encoder.out by 7e-13 relative (2.980119e-7 -> 2.980126e-7) and is CORRECTLY invisible at a 2.33e-5 threshold, so this gate witnesses structure and never a one-ulp conversion error, while an attention-scale slip to 1/sqrt(head_dim+1) reddens it. AND A FALSE GREEN WAS FOUND AND CLOSED: the scoring helper folded (got-want).abs() with f64::max, which returns the OTHER argument on NaN, so an all-NaN capture scored 0.0 -- a perfect match -- proven by running the clean gate GREEN on a capture forced to all-NaN with the guard removed, and red with it. Third recorded reintroduction of that trap in this tree, and the first where the reference side was already guarded and the got side was not. THREE FIXTURE LIMITS were then found by review and are now standing assertions rather than surprises: (1) NO QUERY EVER ATTENDS THE BLOCK -- window 4, block 4, ctx 12, so the furthest query reaches kv 7 of 16 and the block-vs-block submatrix sums to exactly 0.0 on both salts, which means glimmer-architecture.md section 11 step 5 (attention is BIDIRECTIONAL across the block) is pinned by the mask PATTERN alone and by no value, and every draft floor and defect signal was measured on a pattern the real model does not produce; (2) sliding_window and block_size are BOTH 4, so every value comparison scores green with either read for the other -- closed by a gate that widens the window and requires the mask to move; (3) the reference pins its rotary tables to FLOAT32 even under --dtype float64 (inv_freq via torch.arange dtype=torch.float, re-cast inside maybe_autocast(enabled=False)), so the floors have zero contribution from the rope table and could never price one computed otherwise -- the oracle now builds that table in f32 in the reference's own order. LIMITS 1 AND 2 ARE NOW CLOSED BY A RE-VENDOR (2026-08-16): sliding_window 4 -> 13 in the drafter's tiny config, so 13 of the 16 block-vs-block pairs attend and 3 stay masked -- the block genuinely attends itself (row 0 attends row 1, which a causal mask forbids) AND the window still binds, where w>=15 would have made the mask all ones and unable to fail. EVERY draft floor was re-measured under it (attend.q, attend.out and final_norm.out rose 1.35x-1.37x while encoder.out and attend.v did not move at all, because the encoder never sees the mask and V is projected before attention) and every defect signal re-measured: CausalMask went 1.471e0 -> 2.018e0 at attend.out and 6.231e-1 -> 1.075e0 at the logits, measuring for the first time the property it is named for. THE RED PROOF THAT MATTERS: substituting block for window used to redden 1 of 10 tests and now reddens 3, so the blind spot was closed by changing the fixture rather than by adding an assertion. The stack moved too -- the fe747d88 venv is gone and the only one left is transformers 5.15.0 from PyPI, which has no commit -- so the provenance gate now pins assistant_modeling_sha256 (the digest of the four muse_glimmer_assistant/*.py that actually run) by VALUE for every file -- a dual sha-or-digest form was written and killed by review the same day, because the sha branch was unreachable and a 40-hex shape check is strictly weaker than a value check, and the substitution was bridged by regenerating at the OLD geometry and reproducing (1103 text + 50 draft + 107 weights) x 2 salts = 2,520 of 2,520 captures across all six vendored files bit-identically -- CORRECTED 2026-08-16 from "2,420", which was a sum this verdict stated rather than added up. All six were re-vendored together because the one-environment invariant spans them; new pins are text 644,019 B, draft 72,208 B, weights 2,065,247 B. Limit 3 (the f32 rope table) stands and the oracle matches it.
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

### Draft-mode floors — RE-MEASURED 2026-08-16 at the re-vendored geometry

> **The first set of these floors is superseded, not merely refined.** They were measured at
> `sliding_window 4`, where no query ever reached the block (below), so every one of them priced
> an attention pattern the real model does not produce. Do not carry a number out of the
> superseded table; it is kept only in git history.

**The re-vendor.** `sliding_window` 4 → **13** in the drafter's tiny config
(`glimmer_anchor_lib.py`). ctx is `PROMPT_LEN` 12 and `block_size` is 4, so `kv_len` is 16 and
the block's own K/V rows sit at 12..16. The reference indexes queries by ROW (`q_offset = 0`,
no cache) while RoPE places them at `ctx..`, so the mask is `|q_row − kv| ≤ w`:

| w | block-vs-block pairs attending | window binds? |
|---|---|---|
| 4 (old) | **0 of 16** | yes, but the block is out of reach |
| 13 (new) | **13 of 16** | yes — 3 pairs masked |
| ≥ 15 | 16 of 16 | **no** — the mask is all ones and cannot fail |

13 is the **minimum** value producing a strictly-bidirectional pair — a query attending a LATER
block row, exactly what a causal mask forbids — and minimal on purpose: 3 of the 16 block pairs
stay masked, so the window still binds inside the block where w ≥ 15 would be all ones and unable
to fail.

**The cost, and why it is forced rather than chosen.** At w ≥ 12 the CONTEXT half of the mask is
all ones, so this fixture no longer exercises window-masking of the context. Swept over w at
ctx 12 / block 4:

| w | 2–8 | 9 | 10 | 11 | 12 | **13** | 14 | 15+ |
|---|---|---|---|---|---|---|---|---|
| context columns masked | 31–6 | 3 | 1 | 0 | 0 | **0** | 0 | 0 |
| strictly-bidirectional pairs | 0 | 0 | 0 | 0 | 0 | **3** | 5 | 6 |

The two ranges never meet: context masking needs w ≤ 10, bidirectionality needs w ≥ 13. That
follows from the reference's own mask form — `q_offset = 0` indexes queries by ROW (0..block)
while K/V spans `ctx + block`, so any w letting q0 reach kv = ctx also lets it reach every
kv < ctx. **No geometry with this mask has both, at any ctx.** §11 step 5 is the property under
test, so the block wins, and the context-window blind spot is now asserted explicitly in
`glimmer_draft_oracle.rs` rather than left latent.

```bash
$VENV/bin/python glimmer_anchor_driver.py --mode draft --salt glimmer-anchor-$n \
    --defect None --dtype float64 --out draft-$n-f64.bin --no-preflight
# then per-capture max|f32−f64| / max|f64|, worst over both salts and all five layers
```

| capture | floor | 10× tol | vs old |
|---|---|---|---|
| `encoder.out` | 2.3281e−6 | 2.33e−5 | 1.00× |
| `input_layernorm.out` | 2.4993e−6 | 2.50e−5 | 1.23× |
| `attend.q` | 4.7961e−6 | 4.80e−5 | **1.37×** |
| `attend.k` | 2.9202e−6 | 2.92e−5 | 0.88× |
| `attend.v` | 2.3172e−6 | 2.32e−5 | 1.00× |
| `attend.out` | 4.6989e−6 | 4.70e−5 | **1.37×** |
| `post_attention_layernorm.out` | 2.7168e−6 | 2.72e−5 | 1.10× |
| `mlp.out` | 3.7631e−6 | 3.76e−5 | 1.07× |
| `final_norm.out` / `last_hidden` | 3.2815e−6 | 3.28e−5 | **1.35×** |
| `logits` | 2.9504e−6 | 2.95e−5 | 1.11× |

The movement is itself evidence the measurement is real: `encoder.out` and `attend.v` did not
move at all (the encoder does not see the mask, and V is projected before attention), while the
three captures downstream of the widened attention rose 1.35–1.37×. At 3 s.f. every ratio lands
in 9.99×–10.01×, inside `FLOOR_MULT`'s (9.9, 10.2). `attend.mask`, `noise_embeds`, `block_ids`,
`prompt.ids`, `target_layer_ids` and `candidates` came back at exactly 0.0 — copies and integer
picks, compared bit-exactly with no tolerance.

**The defect signals moved with the geometry, which is the whole reason the re-vendor mattered.**
Min over both salts, at L0, now asserted as data by `glimmer_draft_oracle.rs`:

| defect | attend.out | logits | was (old geometry) |
|---|---|---|---|
| `CausalMask` | **2.018e0** | **1.075e0** | 1.471e0 / 6.231e-1 |
| `RopeUntailed` | 2.173e-1 | **4.015e-1** | 2.212e-1 / 2.213e-1 |
| `EncoderNormSkipped` | 1.201e0 | 7.023e-1 | 1.106e0 / 3.237e-1 |
| `TargetGrouping` | 1.015e0 | 1.180e0 | 1.217e0 / 1.022e0 |

`CausalMask` is the one to read: at the old window it could only re-select CONTEXT rows, because
the block was unreachable. It now costs 1.37× more at `attend.out` and 1.73× more at the logits,
and for the first time it is measuring the property it is named for.

### Provenance: the stack changed under the re-vendor, and 2,420 captures say it did not matter

The venv holding transformers at `fe747d88` no longer exists on this machine. Everything above
was produced under `/home/rhansen/glm-anchor/venv` — **torch 2.13.0+cpu, transformers 5.15.0
from PyPI, python 3.14.6** — which has no `direct_url.json` and therefore no commit at all.

Rather than weaken the gate that required a 40-hex sha, the driver now also records
**`assistant_modeling_sha256`**: the digest of the four
`models/muse_glimmer_assistant/*.py` that actually run, name-then-bytes, sorted —
`c31755fb…d709ed`, pinned by value in `glimmer_anchor.rs`. Every vendored file must match that digest — one
assertion, not a choice: the dual form this briefly had was killed by review the same day,
because every file carries `transformers_commit` = `"unknown"` so the sha branch was unreachable,
the driver emits the digest unconditionally so it is not a fallback, and a 40-hex *shape* check is
strictly weaker than a *value* check anyway. `transformers_commit` is still required to agree
across all four goldens. A content hash is the stronger pin of the two: a
repository revision does not tell you which four files were installed.

The bridge was measured before anything was re-vendored under the new stack. Regenerating under
5.15.0 at the OLD geometry reproduced **every capture of every vendored file bit-identically** —
**(1103 text + 50 draft + 107 weights) × 2 salts = 2,520 in all** — with only the
metadata version strings differing. So the only numbers that moved in this re-vendor are the ones
the geometry change was for. All six files were re-vendored together, because
`the_anchor_goldens_record_what_produced_them` requires one environment across all of them and
two files pinned to different stacks cannot be compared to each other.

New byte pins: text 644,019 B, draft 72,208 B, weights 2,065,247 B.

### The red-proof battery, RE-RUN at the re-vendored geometry (2026-08-16)

Every plant above was re-applied after the re-vendor. All still redden — and two rows changed
meaning, which is the measurement that justifies the whole re-vendor:

| plant | old geometry | new geometry |
|---|---|---|
| all-NaN capture, guard PRESENT | red | red |
| all-NaN capture, guard REMOVED | **green (the hole)** | **green (the hole)** |
| extra unconsulted tolerance row | red | red |
| attention scale `1/√(hd+1)` | red | red |
| a recorded L0 signal raised | red | red |
| **`mask()` reading `block` instead of `window`** | red at **1** of 10 tests | **red at 3 of 10** |
| **a mask cell flipped INSIDE the block** | *(inexpressible — the block was never attended)* | **red at 3 of 10** |

The `block`-for-`window` row is the direct evidence. At the old geometry a 128×-in-the-real-model
substitution was visible only to the one gate written for it, and nine value gates scored it
green. At the new geometry the value gates see it too. The blind spot is closed, and it was
closed by changing the FIXTURE, not by adding an assertion.

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
