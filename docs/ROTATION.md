# rivoli — ROTATION: incoherence processing for the int3-vq codebook

Status: **CLOSED NEGATIVE 2026-08-01.** Both arguments for rotation are measured and both
fail. `src/bin/vq_study.rs` is the instrument; the numbers are in "Result" below. Nothing
was built beyond the measurement, which is the point.

> **STATE, in fifteen lines.**
> - The question: would a randomized Hadamard rotation (QuIP incoherence processing) improve
>   int3-vq, which is the worst mode in the engine at **PPL 5.275** against int4's 5.120 and
>   hybrid's 5.189?
> - `src/artifact/quant.rs` says rotation "measured no gain on these already-well-conditioned
>   weights (docs/int3.md)". **`docs/int3.md` was never committed** — not in `main`, not in
>   any commit in history. The claim has no surviving evidence.
> - But the claim is probably RIGHT, on the axis it names. See "What the outlier data already
>   says" — these weights are ~16% more outlier-heavy than a perfect Gaussian, not 5×.
> - What that measurement does NOT cover is the **global codebook**, and that is where the
>   remaining case for rotation lives. See "The two arguments".
> - The gating experiment was therefore **not** a rotation experiment: how much distortion
>   does a per-layer codebook recover over the global one? **Answer: 0.09% median, 0.24%
>   worst, against a 2% bar.** The shared codebook costs essentially nothing, so there is
>   nothing for rotation to homogenise and nothing for a per-layer codebook to recover.
> - The fp8 ground truth **is available** at `/swarm/storage/ai/openclaw/glm52-fp8` (581 GB,
>   141 shards) — verified `glm_moe_dsa`, 78 layers, hidden 6144, 256 experts, e4m3 with
>   `[128,128]` block scales. It reads at **62 MB/s over NFS**, which is the real constraint:
>   sample, never sweep.

## The two arguments for rotation, which are not the same argument

**1. Outliers set the scale (the classic QuIP argument).** One large weight in a
`VQ_GROUP=64` block forces `amax` up, and every other weight in the block is then quantized
against a range it does not need. Rotation smears outliers across the row, `amax` falls
toward RMS, and the same bits buy more resolution.

**2. One codebook must fit 75 layers (the argument specific to this engine).**
`load_codebooks` reads exactly `3 · VQ_K · VQ_DIM` floats — **three global codebooks**, one
per projection, shared by all 256 experts of all 75 MoE layers. k-means fits the *average*
subvector distribution; every layer that deviates from the average pays. Rotation drives
every layer's distribution toward the same isotropic Gaussian, so one codebook fits all of
them equally.

These need different measurements, and only the first has been done.

## What the outlier data already says — argument 1 is weak here

Measured, on the real weights (`benchmarks.md`, `docs/INT4.md` §, per-row, fp8):

| statistic | measured | a perfect Gaussian row of 6144 would give |
|---|--:|--:|
| `amax / median|w|` | **7.2** | **≈ 6.2** (`√(2 ln 6144) / 0.6745`) |
| implied overload point | 4.86σ | 4.18σ |

**So these rows are about 16% more outlier-heavy than Gaussian noise — not 3×, not 10×.**
Incoherence processing exists for weights and activations that are *pathologically* spiky;
GLM-5.2's expert weights are not. A rotation would move `amax/median` from ~7.2 toward ~6.2,
worth on the order of **0.2 bits**, and only if the group scale were the binding constraint.

It is not, for a second reason: that 7.2 is a **per-row** figure, and the shipped quantiser
scales **per group of 64**. `amax` over 64 Gaussian samples sits at ~2.9σ against a row's
4.2σ, so the fine group already collects most of the win a rotation would deliver.

**Conclusion: the deferral was right about what it measured.** Any proposal that leads with
"rotation removes outliers" is arguing against this table and needs to say why.

## What has NOT been measured — argument 2

Nothing in the record prices the global codebook. The distortion of a shared codebook splits
into two parts, and only the sum has ever been observed:

```
D_global  =  D_rate        (what 12 bits per 4 weights can do at all)
          +  D_mismatch    (what this layer loses by sharing a codebook with 74 others)
```

`D_mismatch` is the entire budget rotation could recover on argument 2 — and it is also
exactly what a **per-layer codebook** recovers, at **zero bytes per expert**, with no
rotation, no kernel change, and no activation-side transform. The codebook lives in the file
header; only the current layer's three are live at a time, so the L1 footprint is unchanged.

**If `D_mismatch` is small, both options die together** and int3-vq is rate-limited — which
points at `VQ_K=8192`, at `VQ_GROUP=32`, or at simply using hybrid, which already dominates
int3-vq on both axes (PPL 5.189 *and* 2.72 vs 2.62 tok/s).

## The gating experiment

One number, CPU-only, no GPU lock, no engine change:

1. Read N experts per layer from the **fp8 checkpoint** across a spread of layers (early /
   middle / late). *(An earlier draft said `.i4`, from when the fp8 set was believed
   deleted — see Constraints.)*
2. Re-encode each with `quant_vq` against **(a)** the shipped global codebook, **(b)** a
   codebook k-means'd on that layer alone, and **(c)** the refit control. Same `VQ_K`,
   `VQ_DIM`, `VQ_GROUP` and training-set size — the only variable is who the codebook was
   trained on.
3. Report relative L2 per layer for all three.

`D_mismatch = D_global − D_per_layer`, per layer. **Acceptance: if the median layer recovers
< 2% relative L2, stop** — rotation cannot beat a per-layer codebook at its own argument, and
neither is worth building.

Only if that clears does the rotation arm make sense, and then it is a three-way at fixed
rate: global · per-layer · global-after-rotation.

## Result — measured 2026-08-01, `vq_study`

`--layers 6,34,74 --experts 6 --rows 32 --kmeans-iters 25 --stride 48`, fp8 ground truth,
relative L2 of a full encode→decode round trip through the shipped `quant_vq` /
`vq_decode_proj`:

| layer | shipped | refit (control) | per-layer | recovered |
|--:|--:|--:|--:|--:|
| 6 | 0.1565 / 0.1564 / 0.1549 | 0.1575 / 0.1574 / 0.1563 | 0.1573 / 0.1573 / 0.1561 | 0.16 / 0.06 / 0.15% |
| 34 | 0.1551 / 0.1551 / 0.1551 | 0.1562 / 0.1562 / 0.1565 | 0.1563 / 0.1562 / 0.1563 | −0.03 / −0.03 / 0.08% |
| 74 | 0.1551 / 0.1551 / 0.1557 | 0.1562 / 0.1564 / 0.1569 | 0.1560 / 0.1562 / 0.1565 | 0.12 / 0.09 / 0.24% |

*(gate / up / down in each cell.)* **Median recovered 0.09%, max 0.24%, against a 2% bar —
an order of magnitude below it, and three of nine cells are negative.**

**Two controls make this readable rather than suggestive.**

1. **The refit control is +0.76% off the shipped codebook.** A codebook fitted by this tool
   on pooled subvectors reproduces the shipped one to within a percent, so the per-layer
   column is being compared against a faithful stand-in and not against this tool's own
   incompetence. An early run without this control reported **−7.3%** and meant nothing: at
   3 k-means iterations the per-layer fit was simply unconverged, and the study was
   measuring the fitter.
2. **The training-set size is equalized.** Pooling *n* layers would otherwise hand the
   control *n* times the subvectors, and k-means improves with data — a naive pool wins for
   a reason that has nothing to do with crossing layers. The pooled arm is strided down to
   one layer's count, so the only variable left is whether the training data crossed a layer
   boundary.

**What it means.** Relative L2 sits at **0.155–0.157 everywhere** — across layer 6, 34 and
74, across all three projections, whichever codebook is used. The subvector distributions
of this model are *already* homogeneous across depth. So:

- **Per-layer codebooks: dead.** They recover 0.09%. The zero-byte fix has nothing to fix.
- **Rotation as a homogeniser: dead**, and for a stronger reason — there is nothing to
  homogenise. Rotation's job on argument 2 was to make every layer look alike; they already
  do.
- **Rotation as an outlier fix: already weak** on the `amax/median` table above (7.2 measured
  against 6.2 for pure Gaussian noise), and per-group-of-64 scaling collects most of that.

**int3-vq's 15.5% relative L2 is the RATE.** 12 bits per 4 weights plus a bf16 scale per 64
is what it buys, and no codebook rearrangement moves it. The levers that remain change the
rate or the format: `VQ_K` (a kernel change — see Constraints), `VQ_GROUP` 64→32 (+0.25 bpw),
or hybrid, which already dominates int3-vq on both axes at PPL 5.189 vs 5.275 and 2.72 vs
2.62 tok/s.

**Do not re-open on argument 2 without new evidence about the weights themselves** — a
different checkpoint, or a fine-tune that pushes layers apart. The measurement is cheap
(~2 minutes, CPU + NFS, no GPU lock), so re-run it rather than re-reasoning about it.

## Constraints the next person will hit

- **The fp8 checkpoint is at `/swarm/storage/ai/openclaw/glm52-fp8`.**

  > **CORRECTED 2026-08-01, same day this file was written.** The first version of this
  > section said the fp8 set was deleted and that every study had to be comparative against
  > `.i4`. That was wrong — it was inferred from `/var/db/rivoli` holding only the converted
  > artifact, without looking further. Use fp8 as ground truth; `i4_audit`'s ground-truth
  > modes run. The `.i4`-as-reference fallback below is no longer needed and is kept only
  > because it remains a valid *comparative* reference if the NFS mount is unavailable.

  **The constraint is bandwidth, not availability: 62 MB/s over NFS** (measured, 512 MB cold
  read). The whole set is ~2.6 hours end to end, so nothing here sweeps it. Safetensors is
  seekable and the index maps tensor → shard, so read only the experts you sample: one
  expert is `3 × 6144 × 2048` fp8 ≈ **37.7 MB ≈ 0.6 s**. The gating experiment's 8 experts ×
  6 layers is ~1.8 GB ≈ **30 seconds** of I/O. Budget by expert count, not by layer count.

  Sample several experts per layer, not one — a single expert conflates a *per-layer*
  codebook with a *per-expert* one, and only the first is affordable as header data
  (per-expert would be 256 × 75 × 3 codebooks).
- **`VQ_INDEX_BITS` is load-bearing in the packing.** 12 bits packs two indices per 3 bytes.
  11 or 13 bits needs an arbitrary-bit-boundary packer *on the MoE gather*, which is the hot
  path. Any `VQ_K` change is a kernel change, not a constant change.
- **Rotation needs an activation-side transform too.** gate/up share the post-attention
  rmsnorm output, so one Walsh–Hadamard per layer covers every expert (and `hidden = 6144 =
  512 · 12`, so `H₅₁₂ ⊗ H₁₂` fits exactly). `down` is the hard one: its output rejoins the
  residual stream, so either un-rotate before the add or keep the residual rotated.
- **`E8 fell short` is evidence, not an aside.** `quant.rs` records that a fixed E8 lattice
  underperformed the learned codebook. E8 is optimal *for isotropic Gaussian* data, so that
  result measures how far these subvectors are from isotropic — the same quantity argument 2
  is about. If rotation ever does pay, the prize is not a better codebook but **no codebook
  at all**: a lattice is computed, not gathered, which would delete the L1 pressure that
  `PERF.md` item #2 is trying to relieve by shrinking `VQ_K`.
