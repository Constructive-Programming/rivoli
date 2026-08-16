---
status: data
scope: glimmer
verdict: M11's fp8 arm, measured where it can be measured without a GPU. The artifact arithmetic RECONCILES to the byte — layer_bytes predicts 967.942 MB bf16 and 484.142 MB fp8, totals 55.712 GB and 30.555 GB, and the two artifacts on NFS are 55,712,428,144 B and 30,554,903,564 B, i.e. the prediction plus a safetensors header that grows by 141 B for each of the 416 scale tensors fp8 adds. The pre-registered quality band is Q2's, DERIVED not measured: the old tree's int4-to-hybrid PPL gap 5.120 to 5.189 is 0.013387 nats of log-PPL, and the old tree's own fp8-vs-bf16 paired run (-0.00026, 95% CI [-0.00701, +0.00649]) sits inside it with 2.06x of margin on the upper bound. CONVERTER BYTE-PARITY IS PAID: this tree's --fp8 reproduces the old tree's 30,554,903,564-byte artifact and all five aux files with identical sha256, so the two converters agree on every tile scale and every e4m3 rounding with no tolerance anywhere. TWO gates remain unpaid and each is named with what blocks it: paired dNLL (blocked on M10's --ppl, which has zero commits) and tok/s plus partition bit-identity (need the GPU). The old tree's 4.714 tok/s fp8 figure is NOT a baseline to beat or cite - it was recorded with a non-empty contention witness and discarded by its own author.
---

# Muse Glimmer fp8 — M11

`convert_glimmer --fp8` quantizes the **416** per-layer projections (52 layers × 8) to
fp8-e4m3 with one f32 scale per 128×128 tile. `embed_tokens`, `lm_head` and every norm are
untouched, which is why the halving is 54.8% and not 50%. The engine sniffs the artifact's
projection dtype at open — there is no flag and no legality cell, **the artifact IS the
model**.

## 1. The pre-registration, written before any quality run

**Q2's answer is the equivalence band: the old tree's int4→hybrid dNLL gap.** That gap is
not recorded anywhere as a paired figure, so it is DERIVED from the recorded ladder rather
than quoted, and the derivation is stated so a reader can reject it rather than inherit it:

| | |
|---|---|
| old tree's ladder (`old:docs/reference/modes.md` front matter) | int4 **5.120** < hybrid **5.189** < int3-vq 5.275, PPL |
| the int4→hybrid step in nats | `ln(5.189) − ln(5.120)` = **0.013387** |
| **pre-registered band for Glimmer fp8** | **\|mean dNLL\| ≤ 0.01339 nats**, paired, `bin/ppl` |

**What that derivation assumes, said out loud:** the two PPLs are arm-level figures over one
corpus, so their log-difference equals the paired mean dNLL only because both arms scored the
same positions. They did. It is still a derived number and not a measured paired interval,
which is exactly the difference this band is being used to judge — recorded here so the
weaker provenance travels with it.

**The reference measured this pair already, on the old engine**: fp8-vs-bf16 paired mean
dNLL **−0.00026**, 95% CI **[−0.00701, +0.00649]** over 762 teacher-forced positions
(`old:docs/measurement/benchmarks.md`, "the S4 ladder"). Its upper bound is **2.06×** inside
the band above. That is a prediction for this tree's run, not a substitute for it: the
rewrite's Glimmer arm is a different engine through a different seam.

**An interval straddling zero is inconclusive, not a pass** — the rule this repo has broken
before. The band is on the magnitude; the power question is `bin/ppl`'s own.

## 2. What the bytes already say, with no GPU

`geometry::layer_bytes` is the only place that sizes a Glimmer layer, and both formats
reconcile against the two artifacts on `/swarm/storage/ai/rivoli/` — **an arithmetic check
that costs nothing and is independent of every kernel:**

| | bf16 | fp8 |
|---|---:|---:|
| one layer, `layer_bytes` | 967,942,144 B (**967.942 MB**) | 484,142,464 B (**484.142 MB**) |
| globals (embed + lm_head bf16, final norm f32) | 5,379,352,576 B | same — `--fp8` does not touch them |
| predicted weight total, 52 layers | 55,712,344,064 B (**55.712 GB**) | 30,554,760,704 B (**30.555 GB**) |
| `resident.safetensors` on NFS | **55,712,428,144 B** | **30,554,903,564 B** |
| residual = safetensors header | 84,080 B | 142,860 B |

The header difference — 58,780 B for the **416** `weight_scale_inv` entries fp8 adds, ≈141 B
per entry — is the whole discrepancy, and 141 B is the length of one safetensors index entry
for a name of this shape. So `layer_bytes`' fp8 arm is right at real dims, and the existing
reference artifact is what this converter's `--fp8` produces to within its own header.

**This is also the cheap independent check against M11's named failure mode.** An fp8
dispatch that silently fell back to bf16 would still show ~1.0× tok/s and pass every P4
check; what it cannot fake is the tier. `GlimmerPin::build` now logs the sniffed format
beside the tier size, so a run claiming fp8 while reporting a ~52 GiB tier is caught at
startup:

```
partition: 52 of 52 layers pinned, 0 streamed through 0 slot(s) (Fp8 { block: 128 } projections, 28.5 GiB tier)
```

**28.456 GiB fp8 against 51.886 GiB bf16.** Log the line with every timing arm.

### Scale-grid geometry at real dims

Every projection is multi-tile on **both** axes at the shipped block, which is worth stating
because the fixture-scale gates only reach 2 tiles:

| projection | shape | grid at block 128 |
|---|---|---|
| `q_proj`, `self_attn.gate_proj` | `[4096, 6656]` | `[32, 52]` |
| `k_proj`, `v_proj` | `[256, 6656]` | `[2, 52]` |
| `o_proj` | `[6656, 4096]` | `[52, 32]` |
| `mlp.gate_proj`, `mlp.up_proj` | `[19968, 6656]` | `[156, 52]` |
| `mlp.down_proj` | `[6656, 19968]` | `[52, 156]` |

The anchor fixture the device gate runs on has `intermediate_size` **216**, above the block,
so its MLP projections are `[2, 1]` and `[1, 2]` — `gemv_fp8`'s row-tile index
(`o >> blk_shift(block)`) and its `sc_cols > 1` column stride are both live there, at 2
tiles. Nothing below the real artifact exercises 156.

### And a different kernel entirely

`rivoli_gemv_fp8` is a **dispatcher**: at `i_dim >= 4096` it forwards to
`rivoli_gemv_fp8_splitk`, a separate body with its own scale-row addressing and an LDS
combine in place of the wave reduction. Every real Glimmer projection is over that threshold
(`i_dim` 6656 ×6, 4096 for `o_proj`, 19968 for `mlp.down_proj`), and **every fixture
projection is under it** (72, 48, 216). So `glimmer_fp8_decode.rs` exercises the wave-per-row
kernel and the shipped model exercises only the split-K one.

That is a division of labour, not a hole — `crates/engine/tests/kernel.rs::gemv_fp8_matches_oracle`
scores split-K against the host oracle at `(128, 16384)` and a ragged `(130, 8192)`. What has
no coverage anywhere is split-K **through the Glimmer seam**, and the first thing that will
run it is the real-artifact decode. Read the fixture gate's green accordingly.

## 3. Converter byte-parity — the M11 (a) gate

`tests/convert-parity-glimmer-fp8.sh <reference> <candidate>`, red-proofed over six runs
(`gate-red-proofs.md` §5d). The reference is
`/swarm/storage/ai/rivoli/glimmer-30b-fp8`, written by the old tree's converter; the
candidate is this tree's `--fp8` over the same checkpoint.

**PAID 2026-08-16: every file byte-identical, nothing to argue.**

```
tests/convert-parity-glimmer-fp8.sh /swarm/storage/ai/rivoli/glimmer-30b-fp8 \
                                    /var/db/rivoli/m11/candidate-fp8
PARITY: every file byte-identical                                          exit 0
```

| file | bytes | sha256 (both sides) |
|---|---:|---|
| `resident.safetensors` | 30,554,903,564 | `0ec15657a77f2d66a42286a872986647a745d5515a7991a3aadda77ae1c05116` |
| `manifest.json` | 5,249 | `936fbb0cad7aa6f85c4e2cf0bbf45b7a5b8672e2abee4b4cae20663f530b35e2` |
| `tokenizer.json` | 28,129,897 | `c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d` |
| `tokenizer_config.json` | 79,936 | `781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5` |
| `generation_config.json` | 202 | `1fa51889b1f8d3659802dedaa27e005b81e5c58483f13ecf13f2d97306bc6e35` |
| `chat_template.jinja` | 9,992 | `cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678` |

**What this establishes, and it is more than "the port compiles".** The reference artifact was
written by the OLD tree's converter; this candidate by the rewrite's. Byte-identity over
30.5 GB means the two agree on the tile scale of all **29,536 grids × 416 projections**, on
e4m3's rounding at every one of 25,163,726,336 weights, on the manifest's key order and its
`format` stamp, and on which tensors take which path — with no tolerance anywhere. A scale
convention that differed by so much as one tile would show here as a differing hash.

**The converter run itself**, recorded so the next one can be budgeted:

| | |
|---|---|
| command | `convert_glimmer --fp8 /swarm/storage/ai/rivoli/muse-glimmer-30b <out>` |
| profile | `--release` (a data-production pass; every check on it is `ensure!`, so no `debug_assert!` is lost) |
| wall clock | **38 min** (22:55:09 → 23:33:10), source on NFS, output on local NVMe |
| log | `convert_glimmer: 2 tensors bf16 verbatim, 416 projections quantized to fp8, 209 norms widened to f32, 809 vision tensors skipped` — 416 = 52 × 8, 209 = 52 × 4 + 1, 2 = `embed_tokens` + `lm_head` |
| peak RSS | **38.2 GB** observed |
| parity run | 8 m 45 s (61 GB hashed, one side over NFS) |

> **The module doc's "~25.2 GB" is a claim about OWNED bytes and the observed 38.2 GB does
> not contradict it — but neither is it the number to budget against.** The owned payload is
> exactly 25,169,806,336 B (416 packed projections plus their grids); the rest is the mmap'd
> 57 GB source, whose pages count in RSS as they are touched and are not reclaimed under no
> pressure. **Budget the peak, not the payload.** Recorded here rather than corrected in the
> converter's header, because the header's sentence is about why `--fp8` owns bytes the bf16
> pass does not, and that sentence is true.

## 4. Owed, and what each is blocked on

| gate | blocked on |
|---|---|
| paired dNLL fp8-vs-bf16, band above | **M10's `--ppl`/`--ppl-out` teacher-forcing path.** `bin/ppl` (the comparer) is ported; its producer is not, and `wave/m10-spine` has zero commits. Nothing in M11 can substitute for it: a decode gate can show fp8 differs from bf16, never that it differs by the right amount |
| tok/s ≥ 1.7× bf16 (expect ~1.9×) | the GPU, sole-tenant, witness per arm |
| partition bit-identity across two `--max-mem` | the GPU (fixture-scale half is `glimmer_fp8_decode.rs`) |
| `glimmer_fp8_decode.rs`'s own red proofs | the GPU — recipes in `gate-red-proofs.md` §5e/§5f |

**The bf16 baseline to beat is 2.56 tok/s** (`baseline-2026-08-16.md`, release, 512 tokens,
fully resident). **The old tree's fp8 number is not a baseline.** It recorded 4.714 tok/s
over 220 tokens and 141.9 s of wall clock against bf16's 306.4 — and then discarded both,
because `/tmp/hiptest` (root, not ours) held the device for the duration. An arm with a
non-empty witness is not a lower bound anyone may cite, and it is repeated here only so the
number is not re-inherited from that page as though it had survived.

**Why the paired dNLL is load-bearing for CORRECTNESS and not only for quality.** The device
gate's two assertions are budget-invariance and fp8≠bf16; a scale grid swapped between two
projections of identical shape (`q_proj`/`self_attn.gate_proj`, `k_proj`/`v_proj`,
`mlp.gate_proj`/`mlp.up_proj`) stays finite, stays budget-invariant, and stays unequal to
bf16 — green in both. At real dims that defect is a large systematic quality loss, far
outside the band above. The dNLL run is the gate that sees it.
