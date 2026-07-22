# Group-scaled int3 experts — build plan (our own quant, Path B)

> **STATUS (2026-07-22): the implemented design PIVOTED from this plan.** M1/M2
> ship a *learned vector-quantized* int3 (d=4 subvectors, 4096-entry k-means
> codebook, 12-bit indices + bf16 g64 scales), not the scalar int3-g128 below —
> the VQ codebook beats per-row int4 by ~12–20% RMS on real converted weights
> where scalar int3 only tied it. E8 lattice + Hadamard rotation stayed deferred
> (measured no gain). Source of truth for the built format: `src/quant.rs`
> (oracle + `.i3` loader), `src/bin/fp82vq.rs` (converter), `kernels/*` (`dot_vq_wave`,
> `gemv_vq`, `moe_*_vq`). Sections below are the original scalar plan, kept for
> rationale; a full rewrite lands after the end-to-end tok/s + quality run.

Stream the routed MoE experts in a **uniform group-scaled int3** container we
build ourselves from the fp8 checkpoint, replacing colibri per-row int4. Decode is
NVMe-read-bound and expert bytes are the whole wall; fewer bytes/expert cuts fetch
**and** — because experts get smaller — lets the pool hold ~22% more of them, so
hit rate rises too (bytes buy *fetch × residency*, which compounds). We roll our
own instead of adopting a GGUF because every off-the-shelf sub-4-bit GLM-5.2 is
either a 7-format dynamic mix (unsloth UD) or barely smaller than int4 (Q3_K_M =
0.95×); a uniform format we control is **one dequant kernel**. Built by the
`AGENTS.md` mechanism: scalar oracle → HIP kernel → wire → validate.

## Why int3 beats the shipped int4 (the whole bet)

Not "3 bits > 4 bits" — a *better quantizer* at fewer bits. Colibri's own ablation:

| container | MMLU-class Δ | bytes vs int4 |
|---|---|---|
| shipped per-row int4 | −9.3pp | 1.00× |
| **uniform int3-g64 (scalar)** | **−7.5pp** | 0.75× |
| int3-g64-e8-rot | −5.9pp | 0.75× |

**Plain uniform int3 with group scales already beats per-row int4** — no lattice,
no rotation needed for v1. The lever is scale granularity: a per-row int4 scale
spans thousands of weights, so one outlier coarsens the step for the whole row; a
g128 scale spans 128 weights, so precision lands where the weights live. E8 +
rotation (the −5.9pp tier) is a **deferred** quality upgrade, not v1.

## Source (verified) — fp8, the true weights

`zai-org/GLM-5.2-FP8` — 141 shards, 756 GB, the checkpoint the colibri int4
snapshot was itself built from (same shard numbering). Quantizing it directly
avoids the double-quant loss of going through Q8/GGUF. Confirmed layout:

- experts `model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.weight` = **F8_E4M3**
- scales `..._weight_scale_inv` = **F32, 128×128 block** (gate/up weight `[2048,6144]`
  → scale `[16,48]`; down `[6144,2048]` → `[48,16]`). Dequant: `w_f32 = fp8_e4m3(q) ·
  scale_inv[r//128, c//128]` (confirm scale-vs-inverse convention against colibri's
  converter in M1).

Downloading now to `/swarm/storage/home/old-data/glm52-fp8/` (NFS, 8.8 T free —
fine for a one-time source read; O_DIRECT decode reads the int3 *output* from local
NVMe, never NFS). `sudo` isn't available for a nicer path; revisit if wanted.

## Target format — uniform, per-layer, one read per expert

`<local-nvme>/glm52-int3/L{layer:02}.i3`, one file per MoE layer (3..77 → 75
files). Each file: 256 experts at a **fixed stride**; each expert is its three
projections concatenated:

```
expert[e] = gate ‖ up ‖ down                     # each = [packed int3 rows][f16 group scales]
```

- **Symmetric group-scaled int3.** Along the input (reduction) dim, groups of
  **G = 128** share one f16 scale; value ∈ [−4,3], `w = q · s`, `s = max|w_group|/4`
  (exact rounding/clamp pinned by the M1 oracle). Group along in-dim so the GEMV
  applies the scale per accumulation window. Scales are **finer than the fp8's own
  128×128 blocks** (128 weights/scale vs 16384), so quality ≥ the fp8 blocking.
- **Packing:** 8 consecutive int3 → 3 bytes. 3.125 bpw incl. scales → **~0.78×** int4.
  Per expert ≈ 14.75 MB (vs 18.9), a layer file ≈ 3.78 GB, total **≈ 283 GB**.
- **One O_DIRECT read per expert:** `pread(off = e·STRIDE, len = STRIDE)`. Fixed size
  ⇒ **no index, no per-expert offsets, no shard-straddle** — deletes `plan_group`
  and the safetensors offset-chasing; slots straight into `moe_table`/`stream_expert`
  (offset becomes `e·STRIDE`). 75 fds, held open, no `ulimit` issue.
- Scales live in the file next to their weights; the weight/`.qs` split disappears.

Chose per-**layer** over per-**expert** files (19,200): same single-read goal, 75
fds not 19,200, uniform stride reuses the existing streaming machinery. mmap
rejected for the cold path — the ~283 GB working set dwarfs 128 GB RAM, so mmap
just thrashes page cache we copy to GPU VMM and never re-read; O_DIRECT single-read
wins (`ponytail:` documented in code at the loader).

## Scope — experts only

The resident tier (MLA attention, DSA indexer, dense layers, shared expert, embed,
lm_head, ~10 GB) loads once and never streams, so Phase 1 converts **routed
experts only** and leaves the resident tier on the colibri int4 snapshot. Only the
**MoE fused GEMV** needs an int3 variant; MLA/indexer/dense/lm_head kernels are
untouched. Router (`gate`) stays int4 and selects int3 experts — same cross-source
mixing the engine already does (int4 experts + bf16 indexer). Coherence of int4
router/attention + int3 experts is a **validation gate (M3)**, not an assumption.

## Milestones (each gated on a measured number)

**M0 — source + layout (DONE / in flight).** fp8 source verified (naming, F8_E4M3,
128×128 F32 scales, experts per-tensor sliceable). Download running. **Gate:** all
141 shards present + checksummed against the index sizes.

**M1 — scalar oracle: fp8→fp32 dequant + fp32→group-int3 quant/dequant (`quant.rs`),
CPU-only.** Port fp8-e4m3 unpack + block-scale (mirror colibri's converter; cite),
then the symmetric g128 int3 quant and its inverse. **Gate:** round-trip a real
expert — `dequant(quant(fp8_dequant(w)))` within the expected int3 error, and the
quant matches a NumPy reference bit-for-bit on indices. Builds/validates without a
GPU (run alongside GPU jobs).

**M2 — converter (`bin/fp82int3`), offline host tool.** Read the fp8 shards locally
(mmap), and per MoE layer per expert: dequant each projection (fp8·block-scale) →
group-int3 quantize → pack → append to `L{ll}.i3` at the uniform stride, on **local
NVMe** (nvme0: 378 GB + 283 GB < 1.8 TB). **Gate:** a sampled expert dequants (via
M1) to the same fp32 (±int3 error) whether taken through the converter or straight
from the fp8; report layer-file size ≈ 3.78 GB, total ≈ 283 GB.

**M3 — HIP int3 MoE kernel + wire, validated vs oracle then end-to-end.** Add
`moe_fused_i3.hip` (group-int3 dequant fused into the GEMV, modeled on `gemv_i4` but
3-bit unpack + per-group scales), its `hip.rs` launcher, and an `--int3-experts <dir>`
knob (printed line 1). Build `moe_table` from the uniform stride; `stream_expert`
issues the single pread; resident tier still from the colibri snapshot. **Gates, in
order:** (1) kernel vs M1 oracle, max abs err ≤ `1e-3·max_ref + 1e-3`
(`tests/kernel_test.rs`); (2) decode real tokens — **coherent**, and a quality
spot-check vs the int4 baseline on a fixed prompt set (the coherence risk, measured);
(3) tok/s + bytes/tok — expect fetch ↓ ~22%, residency ↑, tok/s ↑ from 1.00 @128.

**M4 — cut over / keep both.** Behind the knob, keep int4 streaming until M3.3 passes
3 consecutive 512-tok runs (stability gate); then default to int3, delete `plan_group`
and the safetensors expert path if nothing else needs them.

## Deferred

- **E8 lattice + FWHT rotation** (`DEFERRED`, colibri #452): the −7.5→−5.9pp tier.
  Same container + a grid LUT and an inverse-FWHT in the kernel. Do only if v1's
  scalar int3 quality is short.
- **Resident tier → int3** (`DEFERRED`): saves startup RAM, not decode time; also
  needs int3 MLA/indexer/dense kernels. Only to fully drop the colibri dependency.
- **Dual-SSD mirror** (separate unit, colibri #421): split the int3 layer files
  across nvme0/nvme1 by `(layer,e)` hash. Composes; do after int3 lands. Stacks:
  fewer bytes × two drives.
- **g64 instead of g128** if quality is short (halves the group, +0.125 bpw, finer).

## Risks

1. **Cross-quant coherence** (int4 router/attention + int3 experts) — M3.2 gate.
   Fallback: convert router+attention from fp8 too (converter-only, no new kernel).
2. **Scalar int3 quality** — the ablation says it beats per-row int4, but on GLM-5.2
   specifically it's unmeasured until M3.2. Knobs if short: g64, then E8+rotation.
3. **fp8 scale convention** (`_scale_inv` = scale or reciprocal) — get it wrong and
   every weight is off by a block factor. M1 validates against colibri's converter
   and a known-good dequant.
