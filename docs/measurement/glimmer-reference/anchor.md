---
scope: glimmer
status: data
verdict: The S1b anchor exists and runs — Muse Glimmer's own first-party stack (transformers 5.15.0 from PyPI -- no commit, pinned instead by assistant_modeling_sha256 c31755fb..d709ed -- torch 2.13.0+cpu, python 3.14.6; RE-VENDORED 2026-08-16 from transformers 5.16.0.dev0 at commit fe747d88) executed at tiny widths but the REAL structure, and it needs NO GPU because this reference is plain PyTorch with a CPU path for every operator. SIX files are vendored (crates/oracles/tests/glimmer-anchor-{text,draft,weights}-{1,2}.bin; text 644,019 B, draft 72,208 B, weights 2,065,247 B, all re-vendored 2026-08-16), two weight draws x two modes plus a WEIGHT SET per draw, each reproduced byte-for-byte on a later run, and read by crates/oracles/tests/glimmer_anchor.rs with no python, no venv, no network and no device (PATHS CORRECTED 2026-08-17: this verdict and the INDEX row carried the OLD tree's tests/ layout while the body had already been corrected to the rewrite's crates/oracles/tests/ -- the verdict is the half CLAUDE.md tells readers to trust INSTEAD of the doc, and docs.rs only checks the two agree with each other). THE WEIGHT SETS WERE ADDED 2026-08-13 for S3 item 3 (--dump-weights): gate_proj is 72->48 and a layer captures only 18 rows, so 18 equations against 72 unknowns per output element is underdetermined by 4x and EVERY candidate operand admits a weight that fits the captures exactly -- the recover-and-predict shape the sandwich norms use is not weaker here but VACUOUS, because a norm is elementwise and a projection is not. They go in their own files, not into the goldens, so the four pinned FNVs above did not move; verified by regenerating both goldens with the flag on and finding them byte-identical. They were first vendored with no length, no FNV, no census and no regeneration path, which review caught the same day -- crates/oracles/tests/glimmer-anchor.sh now regenerates and cmps them and glimmer_anchor.rs pins their bytes. FOURTEEN defect runs are scored at both draws, 28 runs in all, and each is GATED on the captures it must leave bit-identical rather than merely on having changed something. THE FINDING THAT ONLY THIS CAN SEE - softcap_off moves 7 of 1103 captures and leaves emitted.ids identical, so the argmax-invariant logit path is not an argument but a measurement, and every greedy gate in this repo is provably blind to it. Two reference behaviours were discovered by running it: the DFlash drafter's default mask is block-wide against context+block K/V and RAISES, and passing the correct 2D mask only works with use_cache=False because a fresh DFlashCache reports kv_length 0. The green sets are scoped to step 0 because a defect that shifts the argmax contaminates every later step through the token it feeds back - localisation is only possible on the prefill. Five deviations are declared in the metadata and pinned by the test - eager attention, fp32 against a bf16 checkpoint, the ForConditionalGeneration wrapper (the softcap lives only there), shrunk special-token ids, and output_multiplier kept at the released value rather than recomputed. TOLERANCES ADDED 2026-08-11 for S2 item 1: per-operator fp32 floors for all thirteen buckets, from --dtype float64 (the whole model in double, no island needed) against each fp32 golden, measured at BOTH draws because attend's floor is 2.1x apart between them - 7.819e-6 and 1.639e-5 - so a one-draw floor would have set the threshold at half what a correct kernel needs. SIX rows are tabled as of 2026-08-12 -- attend, rope, o_proj, logits, norm and qk_norm; attend's is floor 1.639e-5, weakest targeting defect 2.086e0 (kv_broadcast_blocked), Rel(1.64e-4). (This said "ONE row is tabled" through five more landing; the count is stated once here and was gated by old:glimmer_tolerance.rs, nowhere else -- CORRECTED 2026-08-16: that file does not exist in the rewrite and no table_covers_exactly caller covers GLIMMER here, so the six-row set is gated in neither direction in this tree.) qk_scale_on_k is EXCLUDED from that set at 6.232e-4 (38x the floor, which would have forced ExactOnly) because (s*q).k and q.(s*k) are the same product - the defect is invisible to this kernel by algebra, not by resolution. CORRECTED 2026-08-12: this said it "is caught in qk_norm/proj instead" at 2.16e0, and measuring it for S3 item 2's row showed all of that false -- qk_norm moves 4.324e-4/2.825e-4 and proj 4.185e-4/2.449e-4, the 2.16e0 was qk_norm_off's attend figure off the wrong row, and and it does not implement trap 3 at all. CORRECTED AGAIN 2026-08-13: the replacement claim -- that it is caught NOWHERE at Rel strength because an RMS norm is scale-invariant -- is ALSO false. The norm cancels a scalar only up to the eps term, a residue of sqrt(1+(s^2-1)*eps/(s^2*m+eps))-1 which reproduces the measured 2.825e-4 and 6.232e-4 figures and is 3.7x-7.9x the qk_norm row's own Rel(7.85e-5). So the defect IS caught there; the exclusion now rests on MARGIN (36x the floor against the 297x a Rel policy needs) and on every m here coming from toy widths with no real weights, the residue falling as 1/m to 0.06x tol at m~1. S4 re-derives it. The other twelve buckets have floors and no row: a floor is half a row, and S2 compares them exactly until the other half is reasoned through. DRAFT-MODE FLOORS MEASURED 2026-08-16 for M17a: the same fp64 protocol at --mode draft, both salts -- 11 compute captures floored at 2.0248e-6..3.5313e-6 while mask/noise_embeds/candidates came back exactly 0.0 (copies and picks, compared bit-exactly, no tolerance) -- and the CPU drafter oracle scores every capture at 10x these (3 s.f.), with a five-defect matrix in crates/oracles/tests/glimmer_draft_oracle.rs proving the comparison reddens. THE GATE'S OWN LADDER was then measured and both halves recorded: a +1 ulp f32 flip on one encoder.fc element moves encoder.out by 7e-13 relative (2.980119e-7 -> 2.980126e-7) and is CORRECTLY invisible at a 2.33e-5 threshold, so this gate witnesses structure and never a one-ulp conversion error, while an attention-scale slip to 1/sqrt(head_dim+1) reddens it. AND A FALSE GREEN WAS FOUND AND CLOSED: the scoring helper folded (got-want).abs() with f64::max, which returns the OTHER argument on NaN, so an all-NaN capture scored 0.0 -- a perfect match -- proven by running the clean gate GREEN on a capture forced to all-NaN with the guard removed, and red with it. Third recorded reintroduction of that trap in this tree, and the first where the reference side was already guarded and the got side was not. THREE FIXTURE LIMITS were then found by review and are now standing assertions rather than surprises: (1) NO QUERY EVER ATTENDS THE BLOCK -- window 4, block 4, ctx 12, so the furthest query reaches kv 7 of 16 and the block-vs-block submatrix sums to exactly 0.0 on both salts, which means glimmer-architecture.md section 11 step 5 (attention is BIDIRECTIONAL across the block) is pinned by the mask PATTERN alone and by no value, and every draft floor and defect signal was measured on a pattern the real model does not produce; (2) sliding_window and block_size are BOTH 4, so every value comparison scores green with either read for the other -- closed by a gate that widens the window and requires the mask to move; (3) the reference pins its rotary tables to FLOAT32 even under --dtype float64 (inv_freq via torch.arange dtype=torch.float, re-cast inside maybe_autocast(enabled=False)), so the floors have zero contribution from the rope table and could never price one computed otherwise -- the oracle now builds that table in f32 in the reference's own order. LIMITS 1 AND 2 ARE NOW CLOSED BY A RE-VENDOR (2026-08-16): sliding_window 4 -> 13 in the drafter's tiny config, so 13 of the 16 block-vs-block pairs attend and 3 stay masked -- the block genuinely attends itself (row 0 attends row 1, which a causal mask forbids) AND the window still binds, where w>=15 would have made the mask all ones and unable to fail. EVERY draft floor was re-measured under it (attend.q, attend.out and final_norm.out rose 1.35x-1.37x while encoder.out and attend.v did not move at all, because the encoder never sees the mask and V is projected before attention) and every defect signal re-measured: CausalMask went 1.471e0 -> 2.018e0 at attend.out and 6.231e-1 -> 1.075e0 at the logits, measuring for the first time the property it is named for. THE RED PROOF THAT MATTERS: substituting block for window used to redden 1 of 10 tests and now reddens 3, so the blind spot was closed by changing the fixture rather than by adding an assertion. The stack moved too -- the fe747d88 venv is gone and the only one left is transformers 5.15.0 from PyPI, which has no commit -- so the provenance gate now pins assistant_modeling_sha256 (the digest of the four muse_glimmer_assistant/*.py that actually run) by VALUE for every file -- a dual sha-or-digest form was written and killed by review the same day, because the sha branch was unreachable and a 40-hex shape check is strictly weaker than a value check, and the substitution was bridged by regenerating at the OLD geometry and reproducing (1103 text + 50 draft + 107 weights) x 2 salts = 2,520 of 2,520 captures across all six vendored files bit-identically -- CORRECTED 2026-08-16 from "2,420", which was a sum this verdict stated rather than added up. All six were re-vendored together because the one-environment invariant spans them; new pins are text 644,019 B, draft 72,208 B, weights 2,065,247 B. Limit 3 (the f32 rope table) stands and the oracle matches it. THE DIGEST'S SELF-CONSISTENCY HOLE IS CLOSED BY MEASUREMENT 2026-08-17: the gate compares each golden's assistant_modeling_sha256 against a constant that had been copied from the driver's own output, so a wrong digest would have agreed with itself forever -- recomputing it by an independent implementation over the LIVE venv gives c31755fbbb88c190d5dc7768a3cfab3f6b0a7c85bfdbaf806be49c322dd709ed over exactly four files, matching the pin, so the constant is the digest of the files that actually run. That recomputation is independent in EXECUTION and not in algorithm design -- it reuses the driver's glob/sort/name-then-bytes recipe on purpose, so it closes the self-consistency loop but would reproduce rather than expose a blind spot in the recipe's own file selection. The residual is that this is a one-off recomputation and not a standing gate; making it standing needs the same conditional-resource mechanism drafter_convert.rs uses for the checkpoint (RIVOLI_DRAFTER_CKPT_REQUIRED, on RIVOLI_CS_REQUIRED's precedent) and is owed, so a re-vendor must re-run the recorded command because the gate cannot. A LIMIT 4 was found 2026-08-17 and no re-vendor closes it, because it is about which code path the reference takes rather than about the widths: q_idx is row + q_offset, the reference sets q_offset from the cache when one is present and to ZERO when one is not, and these goldens are the no-cache branch (a fresh DFlashCache reports kv_length 0, which section Two reference behaviours already records). At ctx 12 / block 4 / window 13 that is the 13-of-16 block-vs-block pairs the re-vendor was for; at the SHIPPED ctx 4096 / block 16 / window 2048 the identical expression gives 0 of 256 and the block does not attend itself at all, while the cache branch gives 256 of 256. The goldens DISTINGUISH the two branches at this geometry (16 of 16 with 3 context columns masked) and pin the one that does not survive scaling -- so the re-vendor replaced bidirectionality-pinned-by-no-value with bidirectionality-pinned-in-the-q_offset-0-regime, strictly better and still not the serving regime. Table and gate: glimmer-reference/drafter-checkpoint.md.
---

# The Muse Glimmer S1b anchor

**What it is.** Goldens emitted by `transformers.models.muse_glimmer` and
`.muse_glimmer_assistant` themselves, at a tiny config, captured on the way past by
`crates/oracles/tests/glimmer_anchor_driver.py`. **Nothing in the driver re-implements the model.**
`docs/reference/glimmer-architecture.md` was extracted by *reading* the modeling code, and a
fixture derived from that reading would put one misreading in the spec, in the golden, and in the
kernel checked against it.

**What it is not.** `crates/oracles/tests/glimmer_anchor.rs` is a **fixture-integrity gate, not a correctness
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

`crates/oracles/tests/glimmer_anchor_text.rs::the_tiny_config_kept_the_real_values` compares every "real" field against
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
GLIMMER_ANCHOR_VENV=/home/rhansen/glm-anchor/venv crates/oracles/tests/glimmer-anchor.sh
```

> **CORRECTED 2026-08-16, PATHS FIXED 2026-08-17.** The command as first written named
> `/home/rhansen/glimmer-anchor/venv`, which is gone (see the note under §Environment), and
> `tests/…`, which is the OLD tree's layout — every path in this document is repo-root-relative
> and the anchor moved under `crates/oracles/` in the rewrite. Both are corrected above and in
> the tolerance commands below. The substitute env additionally needs `--no-preflight`, because
> `preflight_env()` correctly refuses a stack it was not pinned to.

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
T=crates/oracles/tests
$VENV/bin/python $T/glimmer_anchor_driver.py --mode text --defect None \
    --salt glimmer-anchor-1 --dtype float64 --out fp64-1.bin --no-preflight
$VENV/bin/python $T/glimmer_anchor_driver.py --by-operator fp64-1.bin \
    $T/glimmer-anchor-text-1.bin
# The signal a threshold must stay under: the same report, clean against each defect run.
$VENV/bin/python $T/glimmer_anchor_driver.py --by-operator \
    target/glimmer-anchor/text-1-None.bin \
    target/glimmer-anchor/text-1-kv_broadcast_blocked.bin
```

**`--dtype float64` is the whole model in double, with no island.** K3's anchor has to hold every
fla module at fp32 because its KDA ops are triton kernels that refuse double; this reference is
plain PyTorch, so one flag covers every operator at once. Weights are untouched by it —
`init_weights` draws into an explicit f32 buffer and widens — so an fp64 run sees numerically
identical weights and differs *only* in accumulation, which is the property being measured.

**Floors, max over both draws** (`--mode text`; the draft mode's own floors are §Draft-mode
floors below — this said "the draft mode has no floors yet" until M17a measured them 2026-08-16):

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
enforced from the other side by `old:tests/glimmer_tolerance.rs`, which fails on a row for an
unanalysed operator.
> **CORRECTED 2026-08-17.** `old:` prefix added: that file exists only in the old tree, as the
> note under §The rows that exist already records. Nothing in the rewrite enforces this
> direction for GLIMMER — the sentence describes a gate that is not here.

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
$VENV/bin/python crates/oracles/tests/glimmer_anchor_driver.py --mode draft --salt glimmer-anchor-$n \
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

### `weakest_defect` SWEPT 2026-08-17 — the column is a valid bound and its stated rule was wrong

Six of the ten `weakest_defect` values in `DRAFT_ORACLE` had never been re-measured after the
re-vendor, and five of those six had floors that moved, so their captures changed by an unmeasured
amount. They were left standing on the argument that all ten sit ~4 decades above tolerance and
`tolerances_leave_room`'s verdict cannot turn on them. **That argument holds — and it was the wrong
thing to rely on, because it is not a measurement.** All four forward defects were therefore swept
across all ten operators and both salts.

The instrument: a temporary `#[test]` running `case.run(&params, defect)` for each of the four,
scoring every capture with the same `worst_rel` the clean gate uses, taking **max over layers** per
operator and then **min over the two salts** (the binding draw — a tolerance must catch the defect
on every draw). Planted, run, reverted.

**The measured matrix**, min over salts, worst layer, `rel`-vs-golden:

| operator | `CausalMask` | `RopeUntailed` | `EncoderNormSkipped` | `TargetGrouping` |
|---|---|---|---|---|
| `encoder.out` | **2.980e−7** | **2.980e−7** | 9.832e−1 | **2.980e−7** |
| `input_layernorm.out` | 1.064e0 | 3.359e−1 | 7.467e−1 | 9.943e−1 |
| `attend.q` | 1.376e0 | 6.170e−1 | 7.352e−1 | 1.364e0 |
| `attend.k` | 1.278e0 | 2.687e−1 | 7.848e−1 | 1.031e0 |
| `attend.v` | 1.212e0 | 2.434e−1 | 1.308e0 | 8.678e−1 |
| `attend.out` | 2.027e0 | 5.294e−1 | 1.425e0 | 1.673e0 |
| `post_attention_layernorm.out` | 9.642e−1 | 3.488e−1 | 7.489e−1 | 1.050e0 |
| `mlp.out` | 1.452e0 | 4.506e−1 | 1.133e0 | 1.544e0 |
| `final_norm.out` | 1.036e0 | 3.298e−1 | 6.005e−1 | 1.013e0 |
| `logits` | 1.075e0 | 4.016e−1 | 7.023e−1 | 1.180e0 |

**The good news first: every declared value is at or below the measured weakest signal for the
defects its row is answerable for, so no gate's verdict was ever wrong.** Declared-over-tolerance
runs **8,132×** (`post_attention_layernorm.out`) to **56,336×** (`attend.v`) — 3.91 to 4.75
decades, so "~4 decades above tolerance" was right — and the measured minima are all at or above
declared, which only widens those margins.

**Three findings the sweep produced, none of them visible from the declared column:**

1. **`encoder.out` cannot take the stated rule at all.** Three of the four defects leave it at
   **2.980e−7** — the CLEAN oracle's own floor, the identical number §the ladder records for rung
   0 at salt 1. So a literal "min over all four forward defects" would set its `weakest_defect`
   *below its own 2.33e−5 tolerance* and `tolerances_leave_room` would refuse the table. Its
   declared 9.832e−1 is `EncoderNormSkipped`'s and could only ever have been: the encoder does not
   see the mask, the grouping, or Q's RoPE. §11 step 4 — Q never sees the context — is why, and
   the three floor readings are that structural claim measured rather than argued.
2. **The docstring's "neither is a transcription of the other" is false for seven of ten rows.**
   `DRAFT_ORACLE`'s column is supposed to be the worst LAYER while the defect matrix's table is
   L0. For `attend.q` the docstring even offers itself an example — "on a row whose worst layer IS
   L0 the two coincide, e.g. `attend.q` at 5.031e−1 — coincidence, not transcription". The sweep
   says `attend.q`'s worst-layer minimum is **6.170e−1**, not 5.031e−1, so its worst layer is
   **not** L0 and the declared number *is* the L0 figure transcribed. Same for
   `input_layernorm.out`, `attend.k`, `attend.out`, `post_attention_layernorm.out`, `mlp.out` and
   `final_norm.out`. Only `encoder.out`, `attend.v` (1.307e0 against a measured 1.308e0) and
   `logits` (4.015e−1 against 4.016e−1) are the worst-layer numbers they claim to be.
3. **So the column is a LOWER BOUND, not the quantity its name and docstring describe** — and a
   lower bound is exactly what `tolerances_leave_room` needs, which is why nothing broke. It is
   now written down as one.

**What is owed, and why it was not done here.** The strong fix is to make this sweep a standing
test, so the column can never drift again: a per-operator table of the defects each operator is
**answerable for** (a real design statement — finding 1 shows it cannot be "all of them"), plus an
assertion that the non-answerable defects leave the capture at the clean floor. That needs
`Case`, `scored` and `worst_rel` moved into `glimmer_anchor_common/mod.rs` so a second binary can
reach them without jscpd reporting the copy, and `glimmer_draft_oracle.rs` is already **822 lines
against the 800 soft cap**, whose contract is that the next edit shrinks it. Adding ~60 lines
there would break that rule to fix this one. Recorded as owed rather than done, with the matrix
above standing in for it until then.

### Provenance: the stack changed under the re-vendor, and the bridge says it did not matter

> **CORRECTED 2026-08-17.** This heading read "2,420 captures", the same restated-not-added sum
> the verdict was corrected for the day before, three lines above a body that says 2,520. The
> count is now stated **once**, in the body, where it is derived from its three addends — a
> heading that carries a number is a third place for it to rot.

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

> **THE SELF-CONSISTENCY HOLE, CLOSED BY MEASUREMENT 2026-08-17.** The gate compares each golden's
> `assistant_modeling_sha256` against a constant in `glimmer_anchor.rs` — and that constant was
> copied from the driver's own output, so a driver whose digest computation was wrong would have
> written a wrong digest into the metadata and into the constant, and the two would have agreed
> forever. The gate proved the goldens shared an environment; it did not prove the recorded digest
> was the digest of the files that ran.
>
> **It is now measured, by a separate execution over the live venv rather than through the driver:**
>
> **Independent in EXECUTION, not in algorithm design** — a distinction review drew 2026-08-17 and
> it is the honest one. The command below uses the same glob/sort/name-then-bytes idiom as
> `environment()`, deliberately, because reproducing the recipe is what makes the comparison
> meaningful. So it proves the constant is not a copy of a copy, and it does NOT probe the recipe
> itself: if `*.py` silently excluded a file that runs (a nested subpackage, a compiled variant),
> this check reproduces that blind spot rather than exposing it. What is closed is the
> self-consistency loop; auditing the file set is a separate question.
>
>
> ```bash
> python3 -c 'import hashlib, pathlib
> src = pathlib.Path("/home/rhansen/glm-anchor/venv/lib/python3.14/site-packages/transformers/models/muse_glimmer_assistant")
> h = hashlib.sha256()
> for f in sorted(src.glob("*.py")):
>     h.update(f.name.encode()); h.update(f.read_bytes())
> print(len(list(src.glob("*.py"))), h.hexdigest())'
> # 4 c31755fbbb88c190d5dc7768a3cfab3f6b0a7c85bfdbaf806be49c322dd709ed
> ```
>
> **Four files, and the digest matches the pinned constant exactly.** So the constant is the digest
> of `__init__.py`, `configuration_muse_glimmer_assistant.py`,
> `modeling_muse_glimmer_assistant.py` and `modular_muse_glimmer_assistant.py` as installed — the
> four that actually run — and not merely a number agreeing with itself.
>
> **The residual, stated rather than left implicit:** this is a ONE-OFF recomputation recorded here,
> not a standing gate. Making it standing needs the venv present, which is the same conditional-
> resource problem `crates/cli/tests/drafter_convert.rs` solves for the checkpoint with
> `RIVOLI_DRAFTER_CKPT_REQUIRED` on `RIVOLI_CS_REQUIRED`'s precedent; the same mechanism applies
> here and is owed. Until then: a re-vendor must re-run the command above, because the gate cannot.

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
| `draft-2` renamed out of the prefix filter | red (**7 of 10**) | red (**7 of 10**) — at the CENSUS assert |
| `draft-1` renamed out of the prefix filter | *(not run)* | red (**7 of 10**) — at the SALT-PAIRING assert |
| `+1 ulp` on one f32 element of `encoder.fc` | green (below the floor — §the ladder) | green (below the floor — §the ladder) |
| **`mask()` reading `block` instead of `window`** | red at **1** of 10 tests | **red at 3 of 10** |
| **a mask cell flipped INSIDE the block** | *(inexpressible — the block was never attended)* | **red at 3 of 10** |

> **RESTORED 2026-08-17.** The prefix-filter and `+1 ulp` rows, and the whole §ladder section
> below, were **deleted** rather than corrected when this section replaced its predecessor — the
> verdict went on citing the ulp numbers, so the doc promised evidence it no longer carried. They
> are back, and both were **RE-RUN at the re-vendored geometry** rather than copied forward,
> because a re-measured floor invalidates its old red proof. Both reproduced their old results
> exactly; the runs are recorded in §the ladder.

The `block`-for-`window` row is the direct evidence. At the old geometry a 128×-in-the-real-model
substitution was visible only to the one gate written for it, and nine value gates scored it
green. At the new geometry the value gates see it too. The blind spot is closed, and it was
closed by changing the FIXTURE, not by adding an assertion.

The two prefix-filter rows are the anti-vacuity half of the same battery, and **which of them you
plant decides which guard you are testing** — a distinction this document got wrong until
2026-08-17.

`draft_goldens()` selects the vendored files by a **name prefix** (`is_mode`,
`glimmer_anchor_common/mod.rs`), preserving `GOLDENS` order, and `draft_cases()` zips that
selection against `WEIGHT_SETS` **positionally**. So renaming a golden out is not one failure but
two, depending on which one goes:

| plant | survivor pairs with | fires at | message |
|---|---|---|---|
| `draft-2` → `xdraft-2` | `draft-1` + `weights-1` — **correctly** | `cases()`'s census, `glimmer_draft_oracle.rs:156` | `draft case census — both salts must be present` |
| `draft-1` → `xdraft-1` | `draft-2` + `weights-1` — **wrong salt** | `draft_cases()`'s pairing assert, `:174` | `draft-2: paired with weights-1` / `left: "glimmer-anchor-2"` |

Both were run 2026-08-17 (whole binary, `--no-default-features`) and both come back **7 of 10 red,
3 green** — the same count, from different asserts. The three that stay green are the three that
consult no golden: `the_draft_tolerances_leave_room`, `the_draw_gate_reds_on_a_wrong_seed`, and
`the_salted_draws_regenerate_both_vendored_weight_sets_bit_for_bit`, which reads `WEIGHT_SETS`
directly.

> **CORRECTED 2026-08-17.** The restored row was labelled "salt pairing" — inherited from the
> deleted version — and the paragraph beneath it said the plant "mispairs the survivor with the
> wrong weight set". Review traced the zip and that is **false for the `draft-2` plant**: dropping
> the LAST draft entry leaves the first zipped with the first, which is the correct pair, so the
> salt assert never runs and the census assert is what reddens. The mirror plant was then run to
> find out what actually exercises the pairing guard, and it is `draft-1`. Both are now recorded,
> each against the line it fires on. The count was right and the mechanism was not — which is
> exactly the class of error this file keeps a red-proof battery to avoid.

The useful bound: **a silently halved fixture set is caught, but only by the seven tests that read
a golden — and the two guards that catch it are independent, so a change that deleted either would
still leave the battery reading 7 of 10 on one plant.**

### The gate's detection floor — the ladder, re-run 2026-08-17

> **RESTORED 2026-08-17**, and **re-measured** rather than carried: this section was deleted with
> the re-vendor, while the verdict kept quoting its numbers. Every figure below was read off a
> fresh run at `sliding_window 13`.

Each rung is planted in the tree, run, and reverted. The instrument is the clean gate
(`the_oracle_reproduces_every_captured_value_of_both_goldens`); the rel figures were read by
adding one `eprintln!` beside its assert and running with `--nocapture`, which is how a green gate
is made to state what it actually measured.

| rung | perturbation | `encoder.out` rel, salt 1 | salt 2 | gate |
|---|---|---|---|---|
| 0 | none (the shipped oracle) | 2.980119e−7 | 3.147616e−7 | green — **74× inside** its 2.33e−5 tolerance |
| 1 | +1 ulp on one f32 element of `encoder.fc` | 2.980126e−7 | 3.1476353e−7 | **green, and correctly so** |
| 2 | attention scale `1/√head_dim` → `1/√(head_dim+1)` | — | — | red (from the battery above, 2026-08-16) |

**The re-vendor did not move this ladder, and that is itself a measurement.** Rung 0 at salt 1 came
back at **exactly** the pre-re-vendor 2.980119e−7 — expected, because `encoder.out` is the one
capture the widened window cannot reach (the encoder concatenates target hidden states and never
sees a mask), and its floor is the one that moved 1.00× in the table above. The two agree, so the
old rung numbers were re-derived rather than trusted. Salt 2 is the worse draw and is quoted
because the gate is scored on both; 74× is the margin against the binding one.

**Rung 1 is the floor and it is the useful number**: a one-ulp weight flip moves the worst relative
difference by 7e−13, five orders of magnitude below the threshold. That is not a hole — a
single-ulp f32 perturbation is far inside the reference's OWN fp32 noise, which is what the floor
measures — but it does bound what this gate is evidence for. **It cannot witness a conversion that
is off by an ulp; it witnesses structure.** Recorded because this repo has twice registered a
red-proof whose perturbation was below the detection floor and read the resulting green as a
passing gate (the parity 1-ulp and single-sign-flip rungs, `CLAUDE.md` §Gates → parity).

The worst rung-0 rel over both salts, per operator, from the same run — every one 29–74× inside
its tolerance, which is what "a correct oracle sits at ~the floor" looks like measured:

| operator | worst rel | tol | margin |
|---|---|---|---|
| `encoder.out` | 3.148e−7 | 2.33e−5 | 74× |
| `input_layernorm.out` | 6.980e−7 | 2.50e−5 | 36× |
| `attend.q` | 1.347e−6 | 4.80e−5 | 36× |
| `attend.k` | 9.986e−7 | 2.92e−5 | 29× |
| `attend.v` | 6.837e−7 | 2.32e−5 | 34× |
| `attend.out` | 1.260e−6 | 4.70e−5 | 37× |
| `post_attention_layernorm.out` | 7.037e−7 | 2.72e−5 | 39× |
| `mlp.out` | 1.180e−6 | 3.76e−5 | 32× |
| `final_norm.out` | 7.606e−7 | 3.28e−5 | 43× |
| `logits` | 6.876e−7 | 2.95e−5 | 43× |

**A false green was found and closed 2026-08-16.** The scoring helper folded `(got − want).abs()`
with `f64::max`, which RETURNS THE OTHER ARGUMENT ON NaN — so a capture that computed only NaN
scored 0.0, a perfect match. Proven both ways rather than argued: with one capture forced to
all-NaN and the guard removed, the clean gate **passed** (all 39 captures green); with the guard in
place it fails on that capture before any tolerance is consulted. Third recorded reintroduction of
the same trap in this tree (`crates/engine/tests/common/scoring.rs` carries the first two), and the
first where the reference side was already guarded and the got side was not. Both sides are guarded
now — `Case::want` for the reference, `worst_rel` for the oracle.

### Limit 4, found 2026-08-17: the goldens pin the NO-CACHE mask branch

A fourth fixture limit, and unlike limits 1 and 2 **no re-vendor closes it** — it is a property of
which code path the reference takes, not of the widths.

The reference's overlay is `abs(q_idx - kv_idx) <= sliding_window`, and `q_idx` is `row + q_offset`
where `masking_utils.py::_preprocess_mask_arguments` sets `q_offset` from the cache when one is
present and to **0** when one is not. §Two reference behaviours above already records that the
correct 2D mask only works with `use_cache=False`, because a fresh `DFlashCache` reports `kv_length`
0 — so **every vendored draft golden is the `q_offset = 0` branch.**

At this fixture's ctx 12 / block 4 / window 13 that gives the 13-of-16 block-vs-block pairs the
re-vendor was for. **At the shipped ctx 4096 / block 16 / window 2048 the identical expression gives
0 of 256**: `q_idx <= 15` cannot reach `kv >= ctx` once `ctx > window`, so the block does not attend
itself at all. The cache branch (`q_offset = ctx`) gives 256 of 256, 120 of them strictly
bidirectional.

**The goldens distinguish the two branches and pin the one that does not survive scaling** — at this
geometry the cache branch would give 16 of 16 with 3 context columns masked, so this is not a case
the fixture is blind to; it is a case the fixture answers, in the direction serving does not want.
That bounds what the M17a re-vendor bought: it replaced "bidirectionality pinned by no value" with
"bidirectionality pinned by value in the `q_offset = 0` regime", which is strictly better and still
not the serving regime. The full table, the arithmetic and the standing gate are
`glimmer-reference/drafter-checkpoint.md` §The mask M17c must build.

### What the draft goldens still cannot see — limit 3, live

> **RESTORED 2026-08-17.** Limits 1 and 2 were closed by the re-vendor and their record is above.
> **Limit 3 was not closed and was deleted anyway**, while the verdict kept asserting it stands —
> so the one live limit was the one the body stopped carrying. It is a property of the REFERENCE,
> not of the geometry, and no re-vendor can move it.

- **The reference's rotary tables are float32 even under `--dtype float64`.**
  `modeling_muse_glimmer_assistant.py:352` builds `inv_freq` with
  `torch.arange(..., dtype=torch.float)`, and `:358-363` re-casts with `.float()` inside
  `maybe_autocast(enabled=False)  # Force float32`. So the fp64 run that measured every floor in
  this document used the **same f32 table** as the fp32 run, and **the floors have zero
  contribution from the rope table** — they cannot price one computed any other way. The oracle
  originally built it in f64 and sat ~2e−7 relative from the reference (inside `attend.q`'s
  tolerance, but on an argument the floor never made, and growing linearly with position). It now
  builds the table **in f32 in the reference's own order** — `inv_freq = 1/(θ^(2f/dim))` once, then
  `inv_freq × position`, never `position / θ^(2f/dim)` — and the exception is stated at
  `crates/oracles/src/dflash.rs:264-267` rather than left to be rediscovered.
  **Consequence for M17c:** a device rope table computed in any other order or width is outside
  what this anchor prices. It needs its own comparison, not this one's tolerance.

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
- **Nothing about the conversion path.** The TARGET's safetensors reduction is gated by
  `crates/cli/tests/glimmer_convert.rs`, on a synthetic four-layer checkpoint whose index that
  test writes itself.
  > **CORRECTED TWICE, 2026-08-17.** This first said `tests/glimmer_names.rs` — the old tree's
  > name and path, and no such file exists here. The first correction repointed it at
  > `glimmer_convert.rs` but kept the clause "against the vendored
  > `model.safetensors.index.json`", and review caught that as a path fix carrying a claim that
  > is no longer true of the target: **there is no vendored `model.safetensors.index.json` in
  > this tree at all**, and `glimmer_convert.rs`'s own header says so, naming the real-index gate
  > as work that "arrives with the real-checkpoint work". A name or shape wrong in BOTH the
  > schema and the fixture is therefore still uncaught for the target.
  >
  > For the **DRAFTER** it is now caught: M17b vendors the shipped assistant checkpoint's own
  > 6,304-byte safetensors header and gates `DrafterConfig::census()` against it
  > (`crates/cli/tests/drafter_convert.rs`, `glimmer-reference/drafter-checkpoint.md`). That is
  > the shape the target's gate should take when its checkpoint work lands.

