# AGENTS.md — how we build units in rivoli

rivoli is a from-scratch Rust + ROCm/HIP inference engine for GLM-5.2 (MLA + DSA
MoE, 744B) on a single Strix Halo APU (gfx1151, unified LPDDR5X, model streamed
from NVMe). This file is the **method** — the repeatable mechanism that landed
MLA decode, the DSA/MISA indexer, and the fp8 KV cache. Follow it for every new
unit (the next one is MTP speculative decode). Architecture, milestones, and
measured numbers live in `PLAN.md`; this is how we work, not what we're building.

## The mechanism (scalar oracle → kernel → wire → validate), milestone-gated

Every unit is built in this order. Do not skip a rung; each gate is a **measured
number**, not a vibe.

1. **Scalar reference oracle first.** Implement the unit as a plain-Rust,
   correctness-first scalar function (the `attn.rs` / `indexer.rs` / `moe.rs`
   reference path). It never has to be fast — it only has to be *right*, because
   it is the oracle every GPU kernel is checked against. Port the reference
   implementation faithfully from the source of truth (the HF modeling code +
   `config.json`), matching math exactly: norm type & epsilon, RoPE variant,
   scale factors, reduction order where it matters. Cite the source in a comment
   (e.g. "mirrors modeling_glm_moe_dsa.py"). Gate: coherent output on the
   reference path.

2. **HIP kernel, validated bit-for-bit (or within a stated tolerance) against
   the oracle.** Write the kernel in `kernels/*.hip`, launch it via an FFI
   wrapper in `hip.rs`, and add a test in `tests/kernel_test.rs` that runs the
   kernel and the **crate's own scalar primitive** over the same inputs and
   asserts they agree. Build the reference from the crate's real functions, never
   a re-implementation in the test — that's what makes "no oracle drift" true.
   Tolerance: bit-exact for integer/quant paths and fixed-function ops; a stated
   relative tolerance (e.g. `1e-3·max_ref + 1e-3`) only where float reduction
   order or a lossy format (bf16/fp8) legitimately differs — and say why in the
   test. Gate: max abs error within tolerance across realistic dims.

3. **Wire into the engine.** Thread the unit through `engine.rs` (scalar path)
   and/or `gpu.rs` (resident device path), a CLI knob in `config.rs`/`main.rs`
   (see zero-knob rule), and the resident-weight placement in `pin.rs` if it has
   weights. Fail at **construction**, not mid-decode, when a prerequisite is
   missing (e.g. a required shard absent) — loud and early.

4. **Validate end-to-end.** Decode real tokens and confirm the output is
   coherent and matches the reference where it must (e.g. modes that are
   equivalent by construction must be bit-identical — the "dense == dsa below
   the sparsity threshold" invariant). Gate: the milestone's tok/s / correctness
   number in `PLAN.md`.

## Deferred-optimization discipline

The reference stays simple on purpose. When you see an optimization the oracle
doesn't need, DO NOT inline it — leave a `DEFERRED (profiling Pn): …` comment
naming the optimization, the win, and the milestone where it lands (the P1/P2/P3
notes in `attn.rs` are the model). This keeps the oracle readable and the kernel
the single place the tiling/coalescing lives.

## Weights the snapshot doesn't have

The int4 snapshot (`glm52-colibri-int4`) was produced by the colibri converter,
which **skipped** the DSA indexer and the MTP layer (78). When a unit needs
weights that aren't in the snapshot, **range-request extract them from the HF
checkpoint** instead of re-downloading it: fetch `model.safetensors.index.json`,
then for each needed tensor do a ranged GET of its shard header (first 8 bytes =
header length, then the JSON) and a second ranged GET of the tensor's byte span.
This pulled the 110 indexer tensors (412 MB) without touching the ~1.5 TB repo
(`scratchpad/extract_indexer.py` is the reference). HF checkpoint weights are
fp8-e4m3 with 128×128 block scales; to feed the int4 MoE kernel they must be
requantized to our per-row int4 layout the way the converter does.

Store extracted shards next to the snapshot as `out-idx-*.safetensors` /
`out-mtp-*.safetensors`; `snapshot.rs` indexes any `out-*.safetensors`. When
root write to `/var/db/llama-server/...` isn't available in-session, use the
`~/glm52-snap` symlink-overlay (real snapshot symlinks + the new shard) and set
`RIVOLI_SNAPSHOT`; tests read that or skip cleanly.

## Testing & CI

- `./test.sh` = `cargo fmt --check` + `cargo clippy --all-targets --features rocm
  -- -D warnings` + `cargo test --features rocm`. Run it before every commit.
- Kernel tests are `rocm`-gated and need the **GPU**; the scalar/lib tests do
  not. The whole engine compiles and lib-tests **without** a GPU (CPU dev build,
  no `rocm` feature) — keep it that way (feature-gate all HIP behind `rocm`).
- Tests that need the real snapshot read `RIVOLI_SNAPSHOT`/`~/glm52-snap` and
  **skip with a printed note** when it's absent, so bare CI stays green.

## The GPU is sole-tenant — plan work around it

- Exactly **one** GPU process at a time (the startup guard refuses to start if
  another tenant holds >1 GiB). A benchmark or decode run **owns the GPU** for
  its duration; you cannot run kernel tests or a second decode alongside it.
- **CPU work parallelizes with a running GPU job**: `cargo build`/`clippy`/`check`
  (hipcc compiles kernels without the device), scalar/lib tests, weight
  extraction, code review, docs. Do that work while the GPU is busy; queue the
  GPU steps for when it frees.
- For parallel *authoring*, spawn subagents in **git worktrees** (isolated
  branches) doing build+clippy only — never GPU tests — then merge and run the
  GPU validation yourself, serially. (This is how MISA-GPU was built alongside
  the fp8 work.) Prefer disjoint file ownership to minimize merge conflicts.
- When another change is in flight on shared files, add **additive** kernels/FFI
  (a new `*_fp8` variant, a new launcher) rather than changing an existing
  signature — it keeps the build green and avoids clobbering the parallel work.

## Conventions (see also PLAN.md § Conventions)

- **No `unwrap`/`expect` outside tests** (workspace lint, deny). No bare
  catch-all that swallows an error silently — surface it.
- **Zero-knob config**: no env vars, no config files; CLI flags only, and the
  full discovered config is printed as the **first line** of every run. A "GPU
  number" with zero kernel launches is reported as CPU fallback, loudly.
- Bump the crate version after code/Dockerfile changes that ship an image;
  `build.sh` refuses to push an existing tag. `.githooks` run fmt+clippy
  (pre-commit) / tests (pre-push).
- Commit messages: `type(scope): summary`, body explains the *why* and the
  validation done. End with the Co-Authored-By / Claude-Session trailers.

## What the pipeline actually costs (so you optimize the right thing)

Decode is **NVMe-read-bound**, not compute-bound: per token the GPU is ~35–42%
busy; ~60% is spent streaming MoE experts from NVMe. `fetch` (NVMe DMA) overlaps
GPU compute via cross-layer prefetch; `attn` and `mlp` are a sequential
residual-dependency chain and **cannot** overlap each other. So the real levers
are fetch **bandwidth** (queue depth / striping), prefetch **recall**, expert
**cache hit** (bigger pool / fp8 KV frees room), and filling the idle GPU
(speculative/MTP decode). Optimize fetch before compute.
