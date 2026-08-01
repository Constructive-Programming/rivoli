# rivoli — ROTATION: incoherence processing for the int3-vq codebook

Status: **EXPLORATION, nothing built.** This document exists to stop the first measurement
being the wrong one.

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
> - The gating experiment is therefore **not** a rotation experiment. It is one number:
>   how much distortion does a per-layer codebook recover over the global one?
> - **The fp8 ground truth is gone** (365 GB, deleted; 252 GB free on the volume). Every
>   study below is comparative against `.i4`, and says so.

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

1. Decode N experts from `.i4` across a spread of layers (early / middle / late).
2. Re-encode each with `quant_vq` against **(a)** the shipped global codebook, and **(b)** a
   codebook k-means'd on that layer alone. Same `VQ_K`, same `VQ_DIM`, same `VQ_GROUP`, same
   rate — the only variable is who the codebook was trained on.
3. Report relative L2 per layer for both.

`D_mismatch = D_global − D_per_layer`, per layer. **Acceptance: if the median layer recovers
< 2% relative L2, stop** — rotation cannot beat a per-layer codebook at its own argument, and
neither is worth building.

Only if that clears does the rotation arm make sense, and then it is a three-way at fixed
rate: global · per-layer · global-after-rotation.

## Constraints the next person will hit

- **The fp8 checkpoint is deleted.** `i4_audit`'s ground-truth modes cannot run. Use `.i4` as
  the reference: it is built by `fp8_to_i4` straight from fp8 with group-128 scales and
  measures **5.120 PPL against vq3's 5.275**, so it is the most faithful surviving artifact.
  This understates absolute error and is fine for a **comparative** study, where every arm
  sees the same reference — say so in any number that leaves this file.
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
