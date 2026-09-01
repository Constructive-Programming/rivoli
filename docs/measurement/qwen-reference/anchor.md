---
scope: qwen
status: data
verdict: The S2 Qwen3.8-Flash-Next anchor exists and runs, its per-operator TOLERANCES are measured on both weight draws, and it needed NO GPU — transformers' own qwen4_exp (5.16.1, torch 2.13.0+cpu, python 3.14.6) ships pure-torch GatedDeltaNet bodies, so the whole matrix is CPU at a measured 252 s on nproc 32, and the driver ASSERTS fla, kernels and causal_conv1d are all absent rather than trusting the venv. FOUR goldens are vendored (crates/oracles/tests/qwen-anchor-{decode-qwen-anchor-1,decode-qwen-anchor-2,prefill-qwen-anchor-1,ngram-real}.bin at 338,050 / 338,050 / 399,035 / 2,891 B), all four reproduced byte-for-byte by a later independent run, and read deviceless by three test binaries (12 tests, no python, no venv, no network). THIRTY-EIGHT defect rows across 13 classes x 2 salts = 76 compare cells, all green in BOTH directions — each row's declared first-touched bucket reddens AND every captured bucket upstream of it stays bit-identical. NINE tolerance rows sit at 10x their floors, each floor measured fp32-against-fp64 at both draws and taken as the max: gdn_conv 7.638e-6, gdn_op 1.4959e-5, gdn_out_norm 1.6539e-5, gdn_proj 1.1838e-5, hc_attn 1.4249e-5, indexer 1.139e-5, moe 1.5243e-5, moe_route 1.9242e-5, qsa_proj 1.0408e-5, with the weakest targeting defect l2norm_eps_dropped at 1.610e-2 (margin 1077x). THE TOLERANCE FINDING: gdn_op has TWO honest floors and the second is the one a HIP port will be — the reference's chunked and recurrent bodies disagree by 9.420e-5 over 36 layers, 6.3x the fp64 number, at which the l2-norm eps margin falls to 171x and forces ExactOnly, so a CHUNKED prefill kernel needs its own prefill-mode GDN fixture or the 1e-6 eps pinned by reading it as K3's MLA eps is. ngram_hash gets no row at all because its floor is exactly 0.000e0 — the ids are int64 and fp32 and fp64 produce the same integers. THREE INHERITED DEFECTS were found by reading the vendored bytes back and are fixed: the prefill golden held THIRTEEN tensors under one name (the indexer norm runs once per query position and nothing had ever read that file), one width-audit row asserted a distinction the real config does not have (half_head_dim == indexer_head_dim == 128, which is also 2*rotary_dim), and one asserted an equality that was tautological by construction. FIVE red proofs are recorded and were observed left/right rather than by exit code, and one of them arms the per-class rule against bytes from a driver whose own rule was weaker; TWO FURTHER PLANTS LIED and are recorded rather than deleted — one returned a full green on an unmodified tree because its substitution never applied, and the other changed the tree correctly while the GENERATOR refused it, so the artifact under test never moved and the green meant nothing. A THIRD false green was found 2026-08-31 by reading the harness rather than its output and is NOT yet discharged: --mode indexer-identity had no pristine arm at all — _install installs wrap_indexer on every model, and the arm meant to be the reference asked for variant="identity", a string matching no branch, so BOTH arms ran the transcription and the mode reported bit_identical True on every revision it was ever run on; it is fixed by a REFERENCE_INDEXER object() sentinel compared with is, so no string can collapse the two arms again, and BOTH the re-run and the red proof were paid 2026-09-01 -- baseline exit 0 True, planted (taps.py md5 moved) exit 1 FALSE, reverted (md5 restored) exit 0 True, with a score-rescale plant rejected as a proof because topk is invariant under it and it would have reported green on a modified tree. The frozen teacher-forced window stays OWED — it needs the 172.76 GiB checkpoint, which is not on this box.
---

# The Qwen3.8-Flash-Next S2 anchor

What produced these files, what they pin, what was measured from them, and what is still owed.
`crates/oracles/tests/qwen-anchor.sh` regenerates everything below; the three deviceless gates that
read the result are `crates/oracles/tests/qwen_anchor.rs` (provenance, byte pins, structural
survival, the width audit, the tolerance table), `qwen_anchor_fixtures.rs` (operator fixtures,
values, the capture and defect censuses) and `qwen_anchor_windows.rs` (the prefill window and the
real-parameter n-gram hash).

## Why there is no GPU in this anchor, and why that is a measurement

K3's anchor takes the device because fla's KDA ops are triton kernels with no CPU path. This
reference is different: `Qwen4ExpTextGatedDeltaNet` ships **pure-torch bodies inside transformers**
(MOD:266-398) that the kernel-from-hub decorator selects whenever `fla` is absent. So the whole
matrix runs on CPU — **252 s on `nproc` 32**, measured 2026-08-31 as the span from the first
written golden to the last of a full run, which makes it a lower bound since it excludes the
script's own startup — and **no coordinator GPU lease was taken or needed**.

> The figure this replaced was inherited prose, and both halves of it were wrong to restate:
> `qwen-anchor.sh`'s header claimed "~9 min ... ~2 GB peak RSS". The time is 2.1x the measurement
> and the RSS had never been measured at all, so it is deleted rather than corrected — a number
> nothing re-derives drifts, and this file is where that rule is supposed to bind.

That is only true while the venv stays clean, so the driver does not trust it: `preflight_env`
**asserts `fla`, `kernels` and `causal_conv1d` are all absent** and the assertion is recorded in
every golden as `forbidden_absent`. A venv with `fla` installed silently produces goldens of a
different computation.

## The stack, pinned

| what | value |
|---|---|
| entry point | `Qwen4ExpForCausalLM` |
| transformers | **5.16.1** exactly — the `qwen4_exp` model dir does not exist at 5.15 |
| torch / python | `2.13.0+cpu` / `3.14.6` |
| `modeling_qwen4_exp.py` sha256/16 | `77fec77d87f2a0eb` |
| `configuration_qwen4_exp.py` sha256/16 | `26b47995740e3bc5` |
| real `config.json` sha256/16 | `c22eb0a053eed62e` |
| checkpoint | `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13` |

The model repo carries no `auto_map` and no `modeling_*.py`, so **the in-tree transformers class IS
the shipped implementation** and hashing those two files is hashing the reference.

**The config provenance check recomputes rather than compares two frozen copies.** K3's and
Glimmer's anchors pin an FNV-1a of `config.json` beside a constant, which leaves two frozen numbers
agreeing with each other while nothing is recomputed from the live file. Here each golden already
carries `real_config_sha256_16`, so `qwen_anchor.rs` hashes the **live** file and compares against
what the **bytes** say. There is no third constant to update, and a one-digit config edit fails
immediately (red proof 2 below).

## The four selectors, and why each is pinned by value

None of these is visible in the numbers and each defaults to something other than what was run.

* **`attn_implementation` = `eager`.** The reference force-overwrites `_attn_implementation` in
  `__init__`; the driver overrides it afterwards and the metadata records what the model **ended up
  with**. This is the trap K3's anchor paid for already.
* **`experts_implementation` = `eager`.** `torch._grouped_mm` refuses `Double`, so the fp64 floor run
  cannot use it — and a floor measured against a different experts path is not this run's floor. The
  fp32 run therefore matches it. The two paths' own disagreement is measured separately (below).
* **`gdn_entry`.** A decode golden records **both** bodies
  (`torch_chunk_gated_delta_rule,torch_recurrent_gated_delta_rule`) because it is a warm 16-token
  prefill followed by one token; the prefill golden records **only the chunked one**. That asymmetry
  is what the two-floor finding rests on.
* **`conv_entry`.** `causal_conv1d_fn` / `causal_conv1d_update`, the module-level functions the
  reference hands `conv1d.weight.squeeze(1)` to — it never calls the `nn.Conv1d`, so a forward hook
  on that module fires zero times and the conv is captured by wrapping the functions instead.

## The tiny config

Widths only, plus two forced non-width overrides. Depth and structure are the real model's: all 48
layers, the real `layer_types` (in the checkpoint's `full_attention` spelling, so the alias rewrite
at CFG:180-184 is exercised), the real interval, `ple_layer_ids: [2]`, `hc_count: 4`, both conv
kernels at 4, `indexer_compress_ratio: 4`, `ngram_size: 3` and `heads_per_ngram: 8`.

The two non-width overrides:

* **`mrope_section` `[2, 1, 1]`** — it must index into `rotary_dim // 2` frequencies, so the real
  `[11, 11, 10]` (summing to 32 for 64 rotary dims) becomes a sum of 4 for the tiny 8, keeping the
  real shape of two equal sections and one smaller. mRoPE stays inert for text either way
  (MOD:140-155 overwrites `freqs[0]` slices with identical values), which is the point.
* **`rope_theta` 10.0, and `rope_theta` is a WIDTH here.** At 1e7 over 8 rotary dims,
  `inv_freq = theta ** -(0, .25, .5, .75)` is `(1, 1.78e-2, 3.16e-4, 5.62e-6)`, so over positions
  0..15 three of the four frequency pairs are numerically inert and **both partial-rope defect rows
  would be vacuous**. At theta 10 the four angles at position 15 are 15.0, 8.4, 4.7 and 2.7
  radians — spread across the circle, which is what the real theta buys at the real width.
* **`indexer_budget` 8, which is what makes the selection path exist.** `block_topk = budget //
  ratio` and MOD:695's `topk(min(block_topk, complete_blocks))` make QSA **dense by construction**
  whenever the visible set is at most `budget + ratio - 1` — 2051 tokens at the real budget, which is
  why `qwen-architecture.md` records T19's alias as invisible to every planned oracle. At budget 8
  that ceiling is 11, so a 16-position window holds 11 dense queries and 5 selective ones.

### The width audit, at both ends

`assert_width_audit` runs at generation time and `qwen_anchor.rs` re-runs it from the recorded
`widths` and `real_widths` maps. Three lists, **25 rows**, each checked in both maps:

| list | rows | checked |
|---|---|---|
| `distinct` | 18 | distinct in the tiny widths **and** in the real ones |
| `equal` | 4 | equal in both — equalities the model HAS, kept so the fixture stays like it |
| `collide_in_real` | 3 | equal in real, **distinct** in tiny — collisions the fixture deliberately breaks |

> **CORRECTED 2026-08-31 — the third list did not exist, and its absence made one row of the first
> list false.** The inherited audit listed `half_head_dim` vs `indexer_head_dim` among "pairs the
> real config keeps DISTINCT". They are **both 128 in the real config**, which is also
> `2 * rotary_dim` (64) there — so a port reading the index head width off either `head_dim / 2` or
> twice the rotary width produces a **bit-identical fixture at real widths**, and the list asserted
> the opposite. The deviation was argued in the tiny config's own comment and nowhere in the audit's
> data, which is the split between a rule in a comment and a list somewhere else that this anchor
> refuses everywhere else.
>
> Evaluating all nineteen rows against the real config found **two further collisions the tiny
> config already broke and nobody had written down**: `hc_width` vs `conv_dim` (both 10240 —
> `4 * 2560` against `2*2048 + 6144`) and `index_qk_width` vs `moe_intermediate_size` (both 640 —
> `(4+1)*128` against a routed expert width). Coverage nobody records is coverage the next width
> edit deletes for free.
>
> **A fifth `equal` row died in the same pass because it could not fail**: it compared
> `ple_conv_dilation` to `ngram_size`, and `derived_widths` computes the former **as** the latter. A
> gate row that compares a value to itself is decoration, and this one read as coverage of the
> dilated conv's stride. The claim is real and now lives where it can fail —
> `ple_conv_dilation_observed`, read off the **constructed** `ple.conv1d` and pinned at 3.
> `assert_width_audit` now refuses any row naming the same key twice.

The three asymmetries S3 is most likely to read the wrong way are asserted individually on top of
the loops, so a failure names the trap rather than a row index:

| fact | real | tiny |
|---|---|---|
| GDN key→value repeat vs attention `n_rep` | 48/16 = **3** vs 24/2 = 12 | 9/3 = **3** vs 4/2 = 2 |
| rope widths: `rotary_dim`, `head_dim/2`, `indexer_head_dim` | 64, **128, 128** | 8, 16, 24 (and `2*8 != 24`) |
| routed vs shared expert width | **640 == 640** | 28 == 28, kept |

## The goldens

| vendored file | bytes | FNV-1a | holds |
|---|---|---|---|
| `qwen-anchor-decode-qwen-anchor-1.bin` | 338,050 | `0xee980262f256e9b5` | 314 float + 11 int, 7 layers |
| `qwen-anchor-decode-qwen-anchor-2.bin` | 338,050 | `0x013a839417a2a7f1` | the second draw, same tensor set |
| `qwen-anchor-prefill-qwen-anchor-1.bin` | 399,035 | `0x7cf7e4d4f811a461` | 57 float + 2 int, layer 3 only |
| `qwen-anchor-ngram-real.bin` | 2,891 | `0x640056bb550961bf` | 6 int tensors, FULL width |

Captured layers are `0, 1, 2, 3, 27, 46, 47`, chosen for what each **is**: 0 the first GDN layer and
first MoE; **1 the PLE host** (`ple_layer_ids: [2]` is one-indexed — MOD:1202 tests `layer_idx + 1`
— so a capture list written from the config's digits would watch the wrong layer); 2 the GDN layer
immediately downstream of the injection, which gives every PLE-side defect a captured green boundary
at layer 0 and a captured red at layer 1; 3 the first QSA layer; 27 a mid-pattern QSA layer; 46 and
47 the last GDN and last QSA.

**The two draws are compared per tensor, on `to_bits`, over all 314 shared floats** — not by
comparing the two files, which carry their own salt strings in metadata and are therefore guaranteed
to differ whatever the weights did.

### Capture commands

```bash
QWEN_ANCHOR_VENV=/var/cache/rivoli/scratch/qwen-anchor/venv \
  crates/oracles/tests/qwen-anchor.sh          # everything below, ~10 min, CPU, no lock
```

Individually, from `crates/oracles/tests`, with `PYTHONDONTWRITEBYTECODE=1` (an NFS `.pyc` served
from a stale mtime shadowed an edited source while this was being authored — the same scar the port
doc records for stale cargo binaries, one language over):

```bash
$PY qwen_anchor_driver.py --config ../../../docs/measurement/qwen-reference/config.json \
    --mode decode --defect None --salt qwen-anchor-1 --out gold-decode-qwen-anchor-1-None.bin
$PY qwen_anchor_driver.py --config … --mode prefill --defect None --salt qwen-anchor-1 \
    --capture-layers 3 --out gold-prefill-qwen-anchor-1-None.bin
$PY qwen_anchor_driver.py --config … --mode ngram-real --out gold-ngram-real.bin
$PY qwen_anchor_driver.py --tolerance-table <matrix dir>     # the nine rows, derived
```

### Reproducibility, witnessed

The four vendored files were **regenerated by a later independent full run and `cmp`'d
byte-for-byte** — `vendored goldens verified: 4 of 4`, script exit 0, 2026-08-31. The script
enumerates the expected set rather than globbing it, so an unvendored golden is a named failure
instead of a silently shorter loop.

**Re-run 2026-09-01, after the harness was split into five modules, and it is the split that the
run was for.** `qwen_anchor_lib` and `qwen_anchor_driver` had crossed the 800-line soft cap, so
the scoring half moved to `qwen_anchor_compare` and the environment half to
`qwen_anchor_provenance`. A move refactor is exactly the change that can relocate arithmetic
without any static check noticing: resolving every free name and importing every module proves
they LOAD, and proves nothing about what they compute. `vendored goldens verified: 4 of 4`,
byte-for-byte, on all four — decode at both salts, prefill, and the ngram micro-anchor. The
script is `set -euo pipefail`, so reaching that block at all means the 76 defect-matrix compare
cells, the tolerance table and the two equivalence modes had already passed; a green there is a
green for the whole run, not for its last stage.

## The defect matrix — both columns

**38 rows across 13 classes, at 2 salts = 76 compare cells, all green in both directions.**

| class | rows | class | rows |
|---|---|---|---|
| gdn decay form | 3 | n-gram seed/order/layer | 5 |
| conv tap order | 2 | partial-rope width | 2 |
| output-gate activation | 3 | eps homes | 2 |
| indexer relu/pool/budget | 7 | norm offset | 2 |
| hc transpose + lowrank order | 4 | fused-QKV segmentation | 2 |
| router bias/norm/w1w3 | 4 | attention gate split | 1 |
| | | state axis order | 1 |

Each row declares ONE first-touched bucket (`EXPECT_FIRST_TOUCH`), and `_gate_first_touch` derives
**both** halves of the localisation claim from it, refusing a run in which:

* the defect changed **nothing** — that is not a defect run;
* the declared bucket was not **captured** — an uncaptured bucket scores green through a default and
  the whole claim would rest on an empty set, which is the fail-open K3's review found;
* the declared bucket stayed **bit-identical** — the RED column;
* any captured bucket strictly **upstream** of it moved — the HOLD column.

The comparator also aborts if the two runs captured different tensor SETS, because a set mismatch is
a broken harness rather than a defect finding. That distinction was measured into existence on K3,
where individually-captured routed experts made four of five defects report `inf`.

**The hold column's anti-vacuity is checked deviceless too.** `qwen_anchor_fixtures.rs` counts, for
every row, the captured buckets upstream of its declared one, and asserts that the rows with an
**empty** upstream set are **exactly** those whose bucket is the globally-first captured bucket —
an equivalence, so the exception is derived rather than listed. That is **32 of 38 rows with a real
hold column**; the six without are the hyper-connection and eps rows, which first touch layer 0's
`hc_attn`, the first module of the first layer (`ngram_hash` and `ple_inject` are absent there
because the PLE host is layer 1). Their red half still has content; only their hold half is empty,
and that is a property of where the model starts.

### Two things the matrix rejected while being built

* **A per-expert capture makes the tensor set routing-dependent.** `mlp.experts.act_fn` is an
  `nn.SiLU` *instance*, so it is a submodule, and `Qwen4ExpTextExperts.forward` calls it once per HIT
  expert — its call count is a property of the routing, so any defect that moves the routing changes
  the golden's tensor set rather than its numbers. Found as a **duplicate name** in the container's
  own reader. It is excluded from hooking.
* **`in_proj_*` and `out_proj` cannot share a `gdn_proj` bucket.** With both under one bucket,
  `gdn_decay_sigmoid_gate` reddened 2 of its 6 tensors and the first-touch gate correctly refused the
  run: `out_proj` and the block output are DOWNSTREAM of the recurrence. They are now `gdn_out`.

### The indexer transcription, and its own gate

`qwen_anchor_taps.indexer_variant_forward` is the **only** copy of reference arithmetic in this
anchor — three defect rows substitute one line of it each — so it is run against the reference at
`variant=None` and any difference at all is a failure.

> **CORRECTED 2026-08-31: this gate had no pristine arm, and the `bit_identical True` recorded
> here could not have been anything else.** `_install` installs `wrap_indexer` on every model it
> builds, so the arm meant to be the reference had to ask for no transcription explicitly — and it
> did not. It passed `variant="identity"`, a string matching no branch inside
> `indexer_variant_forward`, so both arms ran the same transcribed body and
> `Qwen4ExpTextQSAIndexer.forward` never executed. The mode compared the transcription against
> itself on every revision it was ever run on. The fix is a `REFERENCE_INDEXER` sentinel — a bare
> `object()` compared with `is`, so that no string can silently collapse the two arms again — and
> `wrap_indexer` leaves `cls.forward` alone when it sees it. `restore_reference` is called at the
> top of `_install` and carries `("Qwen4ExpTextQSAIndexer", "forward")`, so the second model's
> `orig` is the pristine reference rather than the first model's wrapper.
>
> **Both were paid 2026-09-01 on rh-anine, and the red was witnessed.** Three arms, with the
> `qwen_anchor_taps.py` md5 checked at each so a plant that fails to move the tree cannot pass for
> one that did: baseline `f7610a5d…` exit 0 `bit_identical True`; planted `5c2b3b6e…` (moved)
> exit 1 **`False`**; reverted `f7610a5d…` (restored) exit 0 `True`. The plant forced the
> `variant=None` path onto the `rope_at_block_end` branch — a defect row already proven to redden
> its own bucket. A score-rescale plant was REJECTED as a proof because `topk` is invariant under
> it, so the mask and logits stay bit-identical and it would have reported green on a modified
> tree. Full record: `gate-red-proofs.md` section 14.

## Tolerances

Every number is **derived by a recorded command**, not transcribed:
`qwen_anchor_driver.py --tolerance-table <matrix dir>` reads the matrix's own goldens and prints the
floors, weakest defects and margins that `crates/oracles/tests/common/tolerance.rs`'s `QWEN` block
carries. Floors are the **max over the two draws**; weakest defects the **min** over the two.

### Per-operator fp32 floors, fp32 against fp64, both draws

The fp64 run doubles the whole model **after** `init_weights`, so the drawn values are bit-identical
to the fp32 run's and the only difference measured is arithmetic. There is no fp32 island to carve
out — the reference bodies are plain torch — but it works **only** with `experts_implementation=eager`.

| operator | draw 1 | draw 2 | floor (max) | row? |
|---|---|---|---|---|
| `gdn_conv` | 7.638e-6 | 7.006e-6 | **7.638e-6** | yes |
| `gdn_op` | 1.496e-5 | 1.316e-5 | **1.4959e-5** | yes (see below) |
| `gdn_out` | 6.884e-6 | 1.617e-5 | 1.617e-5 | no — downstream |
| `gdn_out_norm` | 1.436e-5 | 1.654e-5 | **1.6539e-5** | yes |
| `gdn_proj` | 1.184e-5 | 7.006e-6 | **1.1838e-5** | yes |
| `hc_attn` | 1.425e-5 | 1.002e-5 | **1.4249e-5** | yes |
| `hc_collapse` | 6.480e-6 | 7.433e-6 | 7.433e-6 | no — localisation bucket |
| `hc_mlp` | 1.141e-5 | 7.209e-6 | 1.141e-5 | no — downstream |
| `head` | 6.660e-6 | 8.390e-6 | 8.390e-6 | no — localisation bucket |
| `indexer` | 1.139e-5 | 9.601e-6 | **1.139e-5** | yes |
| `moe` | 1.304e-5 | 1.524e-5 | **1.5243e-5** | yes |
| `moe_route` | 1.924e-5 | 9.782e-6 | **1.9242e-5** | yes |
| `moe_shared` | 1.012e-4 | 2.026e-5 | 1.012e-4 | no — localisation bucket |
| `ngram_hash` | **0.000e0** | **0.000e0** | 0.000e0 | no — see below |
| `ple_inject` | 4.049e-7 | 3.064e-7 | 4.049e-7 | no — downstream |
| `qk_norm` | 8.167e-6 | 1.165e-5 | 1.165e-5 | no — no targeting defect |
| `qsa` | 8.408e-6 | 1.260e-5 | 1.260e-5 | no — downstream |
| `qsa_proj` | 1.041e-5 | 7.265e-6 | **1.0408e-5** | yes |
| `residual` | 7.068e-6 | 7.031e-6 | 7.068e-6 | no — localisation bucket |

**A floor measured at one draw is not a floor.** `gdn_out` differs 2.3x between draws and
`moe_shared` 5.0x; taking the smaller would place a threshold at a fraction of what a correct kernel
can need, and the failure mode is a kernel that is right and cannot pass.

### The nine rows

| operator | floor | weakest targeting defect | margin | tolerance |
|---|---|---|---|---|
| `gdn_conv` | 7.6380e-6 | `gdn_conv_tap_rotated` 1.9282e0 | 252,441x | `Rel(7.64e-5)` |
| `gdn_op` | 1.4959e-5 | `l2norm_eps_dropped` 1.6102e-2 | **1077x** | `Rel(1.50e-4)` |
| `gdn_out_norm` | 1.6539e-5 | `gdn_norm_unit_offset` 9.7290e-1 | 58,824x | `Rel(1.65e-4)` |
| `gdn_proj` | 1.1838e-5 | `gdn_qkv_head_interleaved` 1.7165e0 | 144,996x | `Rel(1.18e-4)` |
| `hc_attn` | 1.4249e-5 | `eps_outside_sqrt` 7.2782e-2 | **5108x** | `Rel(1.42e-4)` |
| `indexer` | 1.1390e-5 | `indexer_budget_one_block_short` 1.0000e0 | 87,800x | `Rel(1.14e-4)` |
| `moe` | 1.5243e-5 | `expert_gate_up_swapped` 8.2217e-1 | 53,938x | `Rel(1.52e-4)` |
| `moe_route` | 1.9242e-5 | `router_sigmoid` 2.8638e-1 | 14,883x | `Rel(1.92e-4)` |
| `qsa_proj` | 1.0408e-5 | `attn_gate_block_split` 1.4696e0 | 141,202x | `Rel(1.04e-4)` |

`gdn_conv_tap_rotated` is the weakest of its class because a **cyclic rotation leaves the weight
multiset intact** — it is the one a symmetry check passes. `router_sigmoid` (2.864e-1) is weaker
here than `router_no_renorm` (4.15e0) because the flat untrained draw makes the renormalisation
cheap; a trained router inverts that, so neither row should be inherited to S4 without re-measuring.

### The load-bearing finding: `gdn_op` has two floors

The reference ships **two implementations of the same recurrence**, so there are two honest floors:

* **1.4959e-5** — fp32 against fp64 through `torch_recurrent_gated_delta_rule`, the per-token path
  the DECODE goldens were produced by and the one rivoli's decode kernel is;
* **9.420e-5** — `torch_chunk_gated_delta_rule` against `torch_recurrent_gated_delta_rule` on
  identical weights, worst over all **36** GDN layers. Two real implementations of one recurrence
  disagreeing, which is what a HIP port will be — and **6.3x** the fp64 number, the same relationship
  K3 measured for KDA (6.301e-5 against a 5.99e-6 fp64 island).

The row uses the first, and the argument is that fixture and kernel associate the same way: a decode
golden produced by the per-token path, scored against a per-token kernel, is not sensitive to the
chunked path's association.

**What that leaves OWED is stated rather than hidden.** At the chunked floor the weakest targeting
defect (`l2norm_eps_dropped`, 1.610e-2) is only **171x** the floor, under the 297x a `Rel` policy
needs — so a **chunked prefill kernel** scored against this fixture would have to be `ExactOnly`,
which a kernel scored against a host oracle cannot satisfy. S3's prefill path therefore needs its
own prefill-mode GDN fixture, **or** the l2-norm eps pinned by READING it (1e-6, hard-coded at
MOD:261, not a config field) exactly as K3's MLA eps is. The prefill golden here captures layer 3
only, which is QSA, so it does **not** close this.

### One more measured floor, and one operator with none

* **`grouped_mm` against the eager expert loop**, same weights and same tokens: logits move by
  **4.673e-6**. That is the cost of the selector this anchor pins, measured rather than assumed.
* **`ngram_hash`'s floor is exactly 0.000e0 at both draws** and no `Rel` value is expressible: the
  hashed row ids are int64, so fp32 and fp64 runs produce the **same integers**. A hash is exact or
  it is wrong. S3 and S4 compare those ids bit-for-bit — which `golden::diff`'s int section already
  does — and the five n-gram defect rows (weakest `ngram_seed_wrong` at 1.691e0) are what make that
  comparison non-vacuous.

## The prefill window: both selection regimes in one pass

Salt 1, layer 3 only, 16 positions. All seven layers at 16 positions is 2.7 MB of vendored bytes for
a claim that lives in one layer's indexer, and the full-depth structure is already pinned by the two
decode goldens.

The evidence is a **ladder of thirteen `k_layernorm` captures** whose leading dimensions are the
per-query complete-block count `(query + 1) / compress_ratio`:

| calls | queries | blocks | against `block_topk` 2 |
|---|---|---|---|
| 1-4 | 3, 4, 5, 6 | 1 | dense |
| 5-8 | 7, 8, 9, 10 | 2 | dense |
| 9-12 | 11, 12, 13, 14 | 3 | **selective** |
| 13 | 15 | 4 | **selective** |

Queries 0, 1 and 2 have no complete block and never reach the norm — which is why 8 calls are dense
while the dense TOTAL is 11, and 8 + 5 + 3 = 16 is asserted as such.

> **The ladder exists only because an inherited defect was found on 2026-08-31.** All thirteen
> results were being written under **one name**. `qwen_anchor_lib.read_golden` asserts against
> duplicates and its comment says of the alternative "the Rust reader keeps both. Latent; loud" — but
> **nothing had ever read the prefill golden back**, so the latent half was what shipped.
> `golden_read::float` returns the FIRST match, so twelve of the thirteen were unreachable by name
> and any census over distinct names was silently short.
>
> The fix is in `Capture.add`: the first call keeps the bare name and the k-th gets `#k`, and a
> `repeated_captures` metadata map records every name that recurred. **Decode bytes are unchanged by
> it, verified by regeneration and `cmp` rather than argued** — no module is called twice in a decode
> forward, which is what that map being `{}` states.
>
> The alternative fix — excluding the norm from hooking, as `act_fn` is — was rejected on a
> measurement: `act_fn`'s call count is a function of the ROUTING, so a moved routing changes the
> tensor set; this count is a function of the window and the compress ratio, both fixed by the
> config, and the ladder **is** the dense/selective boundary the fixture exists to show.

## The real-parameter n-gram anchor

`ngram_vocab_size_base` is **20,000,000** in the checkpoint and 1024 in the tiny config, so the tiny
fixture keeps the RULE while the table fits in a golden and this anchor is the rule at the real
numbers. Six int tensors, generated at full width with `ple_embed_dim` 16 — the **one declared
deviation**, recorded in the metadata as `embedding_dim_deviation`, because the ids do not depend on
it.

Checked as the rule, not as 32 constants: 16 head vocabularies that are strictly increasing primes
at or above the base (20000003, 20000023, …), offsets that are their exclusive prefix sum,
`total_vocab_size` 320,001,446 equal to their sum, and 320,001,536 embedding rows — the next
multiple of `make_ngram_vocab_size_divisible_by` 128.

**The strong check is the per-head band**: each head owns `[offset, offset + vocab)` and every
hashed id must land in its **own** head's band. A hash that applied head 5's offset to head 3's id
stays inside `[0, total)` and passes a global range test. `ngram_order_swapped` is the row that
prices this in the tiny matrix.

The pinned window carries the **real** `eos_token_id` 248044 **twice**: it pads the n-gram history
(MOD:1076) and resets the context at every document boundary (MOD:1053-1067), and the tiny config
cannot hold that id at all because 248044 does not fit a 512-row vocab. `seed` 1234 — a config-class
default **absent** from the checkpoint's file, trap T25 — is what `layer_multipliers` is a function
of, and the gate asserts both the value and the absence.

## Red proofs

Five plants, each **observed to have changed the tree** and to have reddened **the assertion added**,
read left/right rather than by exit code.

All five were run 2026-08-31 against the gate **as it ships** — plants 1-4 were first run against the
single-file version and **re-run after the split into three binaries and the factoring into
`common/golden_read.rs`**, because a restructured gate is a different gate and its old proofs say
nothing about it. Plant 5 exists only after the split; it covers the per-class rule added in the
same pass.

| # | plant | tree change observed | what reddened |
|---|---|---|---|
| 1 | truncate the vendored decode-1 golden by one byte | 338,050 → 338,049 B, `cmp` rc 1 | `qwen-anchor-1: length` **left 338049 right 338050**; the other tests failed at `load`, which is the loader refusing a short file |
| 2 | `full_attention_interval` 4 → 5 in the vendored real config | `git diff` shows the one line | `tiny config lost full_attention_interval` **left Number(4) right Number(5)**, and the recompute check **left "9b4c47d576200cb0" right "c22eb0a053eed62e"** |
| 3 | delete one `EXPECT_FIRST_TOUCH` row, regenerate, re-vendor | 41,097 → 41,052 B, the `diff` names line 703 | `gdn_conv_tap_rotated is in class conv tap order with no first_touch entry` |
| 4 | `gdn_op`'s `Rel` from 1.50e-4 to 1.50e-6 | `git diff` shows the row | `gdn_op: tolerance 1.5e-6 is 0.100x its 1.4959e-5 floor, outside the 9.9..10.2x the rule places it at` |
| 5 | **two** mutations: weaken the driver's own `>= 2` rule to `>= 1`, then move `l2norm_eps_dropped` from `eps homes` into `gdn decay form` | the `diff` shows both; the driver then reports `38 rows, 13 classes, eps homes = 1` | `eps homes carries 1 defect row(s); the plan's rule is >= 2 PER CLASS` |

**Plant 5 needed two mutations and that is the finding, not an inconvenience.** The first attempt
moved the row alone and the run came back GREEN — because
`qwen_anchor_defects.assert_class_coverage` **already enforces `>= 2` for all thirteen classes at
generation time** and refused to write the golden (`regen_rc=1`, `AssertionError: class 'eps homes'
has only 1 defect row(s), the rule is >= 2`, observed). So the generating side's rule is stricter
than the plan's nine-class version, and it is red-proofed by that observation.

That left the Rust assertion needing a scenario it can actually see, and it has one: **a golden
vendored from a driver revision whose own rule was weaker.** Weakening the python rule stands in for
exactly that, and under it the Rust gate reddens. Without the second mutation the assertion would
have been an unreachable duplicate of a check one file over — decoration next to a real gate, which
is what this anchor removed a tautological width-audit row for.

> **TWO plants LIED, and both are recorded rather than deleted — they fail in DIFFERENT ways.**
>
> **(i) The substitution never applied.** The first attempt at plant 3 used a nested-quoting python
> `-c` whose string broke; the `SyntaxError` went to the same stream as everything else, nothing was
> modified, and the suite ran **green, 12 passed, exit 0** — indistinguishable from a plant that
> failed to redden. Caught only by the grep count printed beside it still reading `1`.
>
> **(ii) The tree changed and the ARTIFACT did not.** The first attempt at plant 5 modified
> `qwen_anchor_defects.py` correctly — the `diff` proves it — but the regeneration then failed
> (`regen_rc=1`), so the vendored golden still held the OLD metadata and the gate read bytes the
> plant had never reached. **Green, exit 0, on a genuinely modified tree.** This is the failure mode
> the "observe the tree changed" rule does not by itself catch, and the addition it forces is:
> **observe that the ARTIFACT UNDER TEST changed too.** Here that means reading the generator's exit
> code, which is why every plant above prints `regen_rc`.
>
> These are the fourth and fifth instances of this class on record here (three of seventeen M17
> plants, `docs/measurement/gate-red-proofs.md` §10), and the invariant they confirm is the one that
> file now opens with — extended by (ii).
>
> **A verification LINE also lied in the same round.** Plant 4's restore check was
> `git diff | grep -c rel_row` with an `echo "0 changed lines = restored"` after it. It printed
> **9** — the whole `QWEN` block is new, so every row shows as an addition — and the echo asserted
> nothing. The restore was re-verified by grepping the file for the planted value (0 occurrences) and
> re-running the suite green.

One further red is not a plant: **`indexer_scores_separate_the_blocks` reddened on its own author.**
Its first form demanded all four block scores distinct and went red at `qwen-anchor-2`, whose scores
are `[0.0, 0.0, 0.712, 1.244]`. The tie is not a degenerate draw — the per-block score is
`relu(score).sum(dim=-1) / sqrt(d)`, so **a block whose every raw score is negative scores exactly
0.0**, and two such blocks tie exactly. The assertion was narrowed to the claim that matters: the
score at rank `block_topk - 1` is strictly above the one at rank `block_topk`, by a real margin.

## Declared deviations

Every one is in the metadata and pinned by a test.

1. **`eager` attention** — the reference forces `flash_attention_2`; the driver overrides it.
2. **`eager` experts** — `_grouped_mm` refuses `Double`, so the floor run cannot use it; measured
   cost 4.673e-6 on the logits.
3. **fp32 against the checkpoint's fp8/bf16.**
4. **`ple_embed_dim` 16** in the n-gram anchor only; the ids do not depend on it.
5. **Shrunk special-token ids** in the tiny config (bos 501, eos 500) — 248044 does not fit a
   512-row vocab. The **real** eos appears in the n-gram anchor, which is where it is load-bearing.
6. **`indexer_head_dim` 24 breaks a real 128 == 128 equality** — the deliberate width deviation,
   now listed in `collide_in_real` and argued there.

## OPEN and OWED

* **The frozen teacher-forced window is OWED.** It needs the 172.76 GiB checkpoint, which is not on
  this box (the whole HF cache here is 3.7 MB and holds only Glimmer's config).
* **A real-byte fp8 block-128 slice dequant micro-anchor is OWED**, for the same reason. The
  vendored real bytes that DO exist — `linear_attn_norm_l0.bin`, 256 B — are already fully consumed
  by `crates/artifact/tests/qwen_names.rs::the_gdn_output_norm_weight_is_ones_centred`, which settles
  the ones-vs-zero-centred form, so this anchor cites it rather than duplicating it.
* **A chunked-prefill GDN fixture, or the l2-norm eps pinned structurally** — the `gdn_op` two-floor
  item above. This is the one owed thing that blocks a tolerance claim rather than a nice-to-have.
* **`torch.topk`'s tie-break on equal block scores is OPEN and unpriced.** The relu makes exact ties
  normal (measured above), so when the budget cut falls INSIDE a tie, which blocks are selected is
  decided by the sort and not by the model — a port with a different tie-break diverges on a fixture
  where every captured score is bit-identical. No defect row covers it, because perturbing a
  tie-break is not a perturbation of the reference's arithmetic, and this fixture cannot show it
  because its own cut is clean. S3 must settle it by reading the reference's selection, not by
  scoring against these bytes.
* **`qk_norm` has a measured floor (1.165e-5) and no row**, because no defect targets it yet. A floor
  is half a row; the other half is deciding which defects TARGET the operator, and that is per-kernel
  work. **Do not score it against a threshold** — compare it exactly until a row exists.
* **The tolerance rows are untrained-draw measurements.** `router_sigmoid` being weaker than
  `router_no_renorm` is an artefact of a flat router; S4 re-derives both.
